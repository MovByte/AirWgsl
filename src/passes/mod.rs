//! Canonicalization: the passes that turn a pipeline stage's AIR modules into one self-contained function (plan 5.1
//! stage 5).
//!
//! [`canonicalize`] runs, in order:
//!
//! 1. [`link`]: definitions of the stage module's declarations are copied in from the other modules (imported
//!    libraries); a declaration no module defines stays a declaration.
//! 2. [`specialize`]: function constants get their values (the initializers of their `air.fc_initializer` globals)
//!    and `air.is_function_constant_defined` calls become constants.
//! 3. [`static_init`]: the constructors `llvm.global_ctors` lists (the `air.static_init` functions that compute
//!    constant-address-space globals from function constants) run in the interpreter, and their results become the
//!    initializers of the globals they wrote.
//! 4. [`inline`]: every call of a defined function in the entry point is inlined, transitively. A call cycle is a
//!    refusal (MSL forbids recursion, and WGSL has none).
//! 5. Until nothing changes: [`sroa`] splits stack objects along the byte ranges their accesses use, [`mem2reg`]
//!    promotes stack objects accessed only whole to SSA values, [`fold`] evaluates instructions with constant
//!    operands (with the interpreter's exact semantics) and loads from constant globals, and [`dce`] removes dead
//!    instructions, stack objects that are only written, and unreachable blocks.
//! 6. Extraction: a new module holds the entry point and what it refers to.
//!
//! Every pass preserves the function's meaning in LangRef's sense: the result refines the input (it may replace
//! undef or poison by a value, and remove undefined behavior, never add any). The corpus check runs each function
//! before and after on the interpreter with random inputs and compares by refinement ([`crate::interp::compare`]).
//!
//! What canonicalization cannot do is a [`Refusal`] whose reason starts with the item of the compiler's tracked
//! refusal list it falls under ([`Item`]), `[namespace:id]`, and whose kind is that item's.
//!
//! # The canonical form
//!
//! What [`Canonical`] guarantees to the stages after it (the guarantees marked *checked* are verified before
//! [`canonicalize`] returns; a violation is a refusal, never a silently different form):
//!
//! - The module defines exactly one function, the entry point, at index [`Canonical::function`] (*checked*). Every
//!   other function is a declaration: an `air.*` or `llvm.*` intrinsic, an external symbol that only metadata names
//!   (a function an entry-point record refers to, such as a stitching trait function, is kept as an external
//!   declaration without its body), or one the code refers to without calling it. Code that refers to a defined
//!   function as a value (a function pointer) is refused.
//! - The function calls only declarations (*checked*): `air.*` and `llvm.*` intrinsics, and the functions no linked
//!   module defines, which the lowering classifies (Metal runtime-library helpers `___metal_*` that Metal links at
//!   pipeline creation, visible functions the pipeline did not link, post-tessellation control-point fetches) and
//!   refuses or lowers. Functions through pointers (visible function tables) stay calls through pointers. No
//!   `air.is_function_constant_defined` or `air.normalize_function_constant_predicate` call remains, nor any
//!   `llvm.lifetime.*`, `llvm.assume`, `llvm.experimental.noalias.scope.decl` or `llvm.dbg.*` call (*checked*).
//! - Function constants are values: an `air.fc_initializer` global holds the constant's value as its initializer (or
//!   `undef` when it has none) and is `constant`; a load of one through a constant pointer is folded (*checked*).
//!   Constant-address-space globals are `constant` with the contents the static initializers left in them, when no
//!   function other than the initializers writes that address space. `llvm.global_ctors` and `llvm.compiler.used`
//!   are gone.
//! - The entry point's record in the AIR entry-point metadata (`!air.kernel`, `!air.vertex`, `!air.fragment`, ...)
//!   is kept with the operands it refers to (argument records, and the `__metal_implicit_fc_pred_*` globals of
//!   arguments that carry a function-constant predicate, whose initializers now hold the predicate values: an
//!   argument is present exactly when [`specialize::predicate_present`] says so); the other module-level AIR
//!   metadata (`!air.compile_options`, `!air.version`, `!air.language_version`, `!air.sampler_states`,
//!   `!llvm.module.flags`, ...) is copied, without the records that name a global the canonical module does not keep
//!   (the `!air.function_constants` records of folded constants); a list left empty goes.
//!   The AIR metadata decoder (package S2) reads these from the canonical module.
//! - Stack objects left (`alloca`) are those accessed in ways that need memory: dynamic indices, pointers that
//!   escape (into calls, stores of the pointer, `ptrtoint`, `select` or `phi`), volatile or atomic accesses, or
//!   accesses of different types over overlapping bytes. The static allocas of inlined functions are in the entry
//!   block; other allocas stay where the front end put them.
//! - The control flow graph has only reachable blocks, and a block whose only predecessor ends in an unconditional
//!   branch to it is merged into it. SSA values (phis included) stay as LLVM defines them; `freeze` stays.
//! - Constants are LLVM's canonical constants. Arithmetic, comparisons, casts other than `ptrtoint`, vector and
//!   aggregate operations, `select`, `freeze`, calls of declarations without memory effects, and loads from constant
//!   globals are folded whenever all their operands are constants and the interpreter determines the result exactly:
//!   a defined value without choice bits, a wholly `undef` value, or poison. Results the semantics leave open (a
//!   NaN's sign and payload, the sign of a zero under `nsz`, `llvm.fmuladd`'s fusion) are not chosen, except
//!   `freeze`'s, which is one fixed value: zero bits where its operand is undef or poison. Results that depend on
//!   where a global lies (`ptrtoint`, ordered comparisons of pointers) are not folded.
//! - The alias-analysis metadata (`!alias.scope`, `!noalias`) of inlined instructions is dropped (it would otherwise
//!   share one callee's scopes between its copies); other metadata attachments stay; `!dbg` locations of inlined
//!   instructions are dropped.

pub mod copy;
pub mod dce;
pub mod fold;
pub mod inline;
pub mod link;
pub mod mem2reg;
pub mod reflection;
pub mod specialize;
pub mod sroa;
pub mod static_init;
pub mod util;
pub mod values;

use crate::api::{FunctionConstant, Location, Refusal, RefusalKind};
use crate::ir::{GlobalId, Metadata, Module, NamedMetadata, Value};

/// A canonicalized entry point.
#[derive(Clone, Debug)]
pub struct Canonical {
    pub module: Module,
    /// Index of the entry point in `module.functions`.
    pub function: usize,
    /// Where the stage module's global variables went: stage-module index to `module.globals` index, for the ones
    /// the canonical module keeps. (Linking appends to the stage module, so its own indices stay valid through the
    /// passes.)
    pub globals: std::collections::BTreeMap<u32, u32>,
}

/// Fixed-point iterations of SROA, mem2reg, folding and DCE before canonicalization gives up.
const ROUNDS: usize = 64;

/// The named metadata lists of entry points: each operand is a record whose first operand is the function.
pub const ENTRY_LISTS: &[&str] = &[
    "air.vertex",
    "air.fragment",
    "air.kernel",
    "air.visible",
    "air.intersection",
    "air.mesh",
    "air.object",
    "air.tile",
];

/// Canonicalizes function `entry` of `modules[0]`, linked with the other modules, with function constant values
/// `constants` (constants without a value stay undef, as Metal leaves them: see [`specialize`]).
// Refusals go back to the caller by value, as everywhere in the compiler.
#[allow(clippy::result_large_err)]
pub fn canonicalize(modules: &[Module], entry: &str, constants: &[FunctionConstant]) -> Result<Canonical, Refusal> {
    let records = modules.first().map(specialize::records).unwrap_or_default();
    canonicalize_with(modules, entry, &records, constants)
}

/// [`canonicalize`] for modules the AIR semantic layer decoded: the function constants are its typed records (the
/// IR the canonicalization works on is each module's [`crate::air::AirModule::ir`]).
#[allow(clippy::result_large_err)]
pub fn canonicalize_air(
    modules: &[crate::air::AirModule],
    entry: &crate::air::EntryPoint,
    constants: &[FunctionConstant],
) -> Result<Canonical, Refusal> {
    let Some(first) = modules.first() else {
        return Err(refusal(Item::ModuleUnreadable, "pipeline", &entry.name, "no modules"));
    };
    let records: Vec<specialize::ConstantRecord> = first
        .function_constants
        .iter()
        .map(|c| specialize::ConstantRecord {
            global: c.global,
            type_name: c.type_name.clone(),
            name: c.name.clone(),
            index: c.index as u64,
        })
        .collect();
    let irs: Vec<Module> = modules.iter().map(|a| a.ir.clone()).collect();
    canonicalize_with(&irs, &entry.name, &records, constants)
}

#[allow(clippy::result_large_err)]
fn canonicalize_with(
    modules: &[Module],
    entry: &str,
    records: &[specialize::ConstantRecord],
    constants: &[FunctionConstant],
) -> Result<Canonical, Refusal> {
    let Some(first) = modules.first() else {
        return Err(refusal(Item::ModuleUnreadable, "pipeline", entry, "no modules"));
    };
    let mut m = link::link(first, &modules[1..], entry)?;
    let f = m
        .function_by_name(entry)
        .filter(|&i| !m.functions[i].is_declaration())
        .ok_or_else(|| refusal(Item::ModuleUnreadable, "function", entry, "no linked module defines it"))?;
    specialize::specialize(&mut m, records, constants)?;
    static_init::run(&mut m)?;
    inline::inline_all(&mut m, f)?;
    simplify(&mut m, f)?;
    let out = extract(&m, f);
    check(&out.module, out.function)?;
    Ok(out)
}

/// The fixed point of SROA, mem2reg, folding and DCE on function `f`.
#[allow(clippy::result_large_err)]
pub fn simplify(m: &mut Module, f: usize) -> Result<(), Refusal> {
    for _ in 0..ROUNDS {
        let mut changed = false;
        changed |= fold::fold(m, f)?;
        changed |= dce::dce(m, f);
        changed |= sroa::sroa(m, f)?;
        changed |= mem2reg::mem2reg(m, f);
        changed |= dce::dce(m, f);
        if !changed {
            return Ok(());
        }
    }
    let name = m.functions[f].common.name.clone();
    Err(refusal(Item::NonCanonical, "canonicalization", &name, "the passes did not reach a fixed point"))
}

/// A module holding function `f` of `m` and what it refers to (module-level metadata included, entry-point lists
/// reduced to `f`'s record).
pub fn extract(m: &Module, f: usize) -> Canonical {
    let mut out = copy::empty_like(m);
    let mut c = copy::Copier::new(m, copy::Reach::CodeOnly);
    c.metadata_functions_as_declarations = true;
    let GlobalId::Function(nf) = c.global(&mut out, GlobalId::Function(f as u32)) else { unreachable!() };
    c.finish(&mut out);
    // The entry point's records bring what they name (argument predicate globals) along.
    c.reach = copy::Reach::All;
    let mut lists = Vec::new();
    for nm in &m.named_metadata {
        if !ENTRY_LISTS.contains(&nm.name.as_str()) {
            continue;
        }
        let mut ops = Vec::new();
        for &op in &nm.ops {
            if record_function(m, op) == Some(f)
                && let Some(d) = c.md(&mut out, op)
            {
                ops.push(d);
            }
        }
        if !ops.is_empty() {
            lists.push(NamedMetadata { name: nm.name.clone(), ops });
        }
    }
    c.finish(&mut out);
    c.reach = copy::Reach::CodeOnly;
    let skip = ["llvm.global_ctors", "llvm.compiler.used", "llvm.used"];
    let mut others = Vec::new();
    for nm in &m.named_metadata {
        if ENTRY_LISTS.contains(&nm.name.as_str()) || skip.contains(&nm.name.as_str()) {
            continue;
        }
        // A record that names a global the canonical module does not keep (a function constant's global, once its
        // value is folded) describes nothing left: it goes, rather than staying with a null operand.
        let kept: std::collections::HashSet<GlobalId> = c.globals.keys().copied().collect();
        let names_dropped = |op: crate::ir::MdId| -> bool {
            let Metadata::Tuple { ops, .. } = m.md(op) else { return false };
            ops.iter()
                .flatten()
                .any(|&o| matches!(m.md(o), Metadata::Value { value: Value::Global(g), .. } if !kept.contains(g)))
        };
        let mut ops = Vec::new();
        for &op in nm.ops.iter().filter(|&&op| !names_dropped(op)) {
            if let Some(d) = c.md(&mut out, op) {
                ops.push(d);
            }
        }
        if !ops.is_empty() {
            others.push(NamedMetadata { name: nm.name.clone(), ops });
        }
    }
    c.finish(&mut out);
    // Keep the source order of named metadata.
    let order: Vec<&str> = m.named_metadata.iter().map(|n| n.name.as_str()).collect();
    let mut all: Vec<NamedMetadata> = lists.into_iter().chain(others).collect();
    all.sort_by_key(|n| order.iter().position(|o| *o == n.name).unwrap_or(usize::MAX));
    out.named_metadata = all;
    let globals = c
        .globals
        .iter()
        .filter_map(|(s, d)| match (s, d) {
            (GlobalId::Var(a), GlobalId::Var(b)) => Some((*a, *b)),
            _ => None,
        })
        .collect();
    Canonical { module: out, function: nf as usize, globals }
}

/// The function an entry-point record (`!{ptr @f, ...}`) names.
pub fn record_function(m: &Module, rec: crate::ir::MdId) -> Option<usize> {
    let Metadata::Tuple { ops, .. } = m.md(rec) else { return None };
    let first = (*ops.first()?)?;
    match m.md(first) {
        Metadata::Value { value: Value::Global(GlobalId::Function(i)), .. } => Some(*i as usize),
        _ => None,
    }
}

/// Checks the canonical form's guarantees that later stages rely on.
#[allow(clippy::result_large_err)]
fn check(m: &Module, f: usize) -> Result<(), Refusal> {
    let func = &m.functions[f];
    let name = &func.common.name;
    for (i, g) in m.functions.iter().enumerate() {
        if i != f && !g.is_declaration() {
            return Err(refusal(
                Item::NonCanonical,
                &format!("function @{}", g.common.name),
                name,
                "a defined function other than the entry point is referred to (as a function pointer)",
            ));
        }
    }
    let fc_globals: Vec<u32> = specialize::records(m).iter().map(|r| r.global).collect();
    for b in &func.blocks {
        for &id in &b.insts {
            match &func.inst(id).op {
                crate::ir::Op::Call { callee: Value::Global(GlobalId::Function(c)), .. } => {
                    let callee = &m.functions[*c as usize].common.name;
                    let gone = ["air.is_function_constant_defined", "air.normalize_function_constant_predicate"];
                    let gone_llvm =
                        ["llvm.lifetime.", "llvm.assume", "llvm.experimental.noalias.scope.decl", "llvm.dbg."];
                    if gone.iter().any(|g| callee.starts_with(g)) || gone_llvm.iter().any(|g| callee.starts_with(g)) {
                        return Err(refusal_at(
                            m,
                            f,
                            id,
                            Item::NonCanonical,
                            &format!("call of @{}", callee),
                            "the canonical form has no such call, but the passes left it",
                        ));
                    }
                }
                crate::ir::Op::Load { ptr: Value::Global(GlobalId::Var(g)), .. } if fc_globals.contains(g) => {
                    return Err(refusal_at(
                        m,
                        f,
                        id,
                        Item::FunctionConstants,
                        "function constant load",
                        "a load of a function constant global, which the passes fold",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// The tracked refusal-list items canonicalization refuses with ([`crate::wgsl::tracked`]): each refusal's reason
/// starts with its item, `[namespace:id]`, and its kind is the item's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    /// A call cycle: MSL has no recursion and WGSL cannot express it.
    CallCycle,
    /// A static initializer canonicalization cannot run at compile time.
    StaticInit,
    /// A function constant record or query canonicalization cannot resolve.
    FunctionConstants,
    /// A function constant value of another type or size than the constant's, which Metal rejects.
    FunctionConstantType,
    /// A `bool` function constant value with a byte other than 0 or 1.
    FunctionConstantBool,
    /// A linked module that does not match the stage module (data layout).
    LinkedModule,
    /// The stage modules lack the entry function.
    ModuleUnreadable,
    /// A form the passes should have removed but did not (the canonical form's checks), or a limit of them.
    NonCanonical,
}

impl Item {
    pub fn tag(self) -> &'static str {
        match self {
            Item::CallCycle => "compiler:call-cycle",
            Item::StaticInit => "family:static-init",
            Item::FunctionConstants => "family:function-constants",
            Item::FunctionConstantType => "compiler:function-constant-type",
            Item::FunctionConstantBool => "compiler:function-constant-bool-value",
            Item::LinkedModule => "compiler:linked-module",
            Item::ModuleUnreadable => "compiler:module-unreadable",
            Item::NonCanonical => "compiler:non-canonical-input",
        }
    }

    pub fn kind(self) -> RefusalKind {
        match self {
            Item::CallCycle => RefusalKind::NoExactLowering,
            Item::FunctionConstantType | Item::LinkedModule | Item::ModuleUnreadable => RefusalKind::RejectedByDevice,
            _ => RefusalKind::NotImplemented,
        }
    }
}

pub(crate) fn refusal(item: Item, construct: &str, function: &str, reason: &str) -> Refusal {
    Refusal {
        kind: item.kind(),
        construct: construct.to_string(),
        location: Some(Location { function: function.to_string(), instruction: None }),
        reason: format!("[{}] {}", item.tag(), reason),
        reference: None,
    }
}

pub(crate) fn refusal_at(
    m: &Module,
    f: usize,
    i: crate::ir::InstId,
    item: Item,
    construct: &str,
    reason: &str,
) -> Refusal {
    Refusal {
        kind: item.kind(),
        construct: construct.to_string(),
        location: Some(Location {
            function: m.functions[f].common.name.clone(),
            instruction: Some(crate::ir::print::print_instruction(m, f, i)),
        }),
        reason: format!("[{}] {}", item.tag(), reason),
        reference: None,
    }
}
