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
//! # The canonical form
//!
//! What [`Canonical`] guarantees to the stages after it:
//!
//! - The module defines exactly one function, the entry point, at index [`Canonical::function`]. Every other
//!   function is a declaration: an `air.*` or `llvm.*` intrinsic, or an external symbol that no linked module defines
//!   and that is only referred to, never called (a call of one is a refusal).
//! - No call in the function calls a defined function; no `air.is_function_constant_defined` or
//!   `air.normalize_function_constant_predicate` call remains; no load reads an `air.fc_initializer` global; no
//!   `llvm.lifetime.*`, `llvm.assume`, `llvm.experimental.noalias.scope.decl` or `llvm.dbg.*` call remains.
//! - Globals keep their address spaces; constant-address-space globals the static initializers wrote are `constant`
//!   with the computed initializer. `llvm.global_ctors` and `llvm.compiler.used` are gone.
//! - The entry point's record in the AIR entry-point metadata (`!air.kernel`, `!air.vertex`, `!air.fragment`, ...)
//!   is kept with the operands it refers to (argument records, and the `__metal_implicit_fc_pred_*` globals of
//!   arguments that carry a function-constant predicate, whose initializers now hold the predicate values: see
//!   [`specialize::predicate_present`]); the other module-level AIR metadata (`!air.compile_options`,
//!   `!air.version`, `!air.language_version`, `!air.sampler_states`, `!llvm.module.flags`, ...) is copied, with
//!   references to globals that were not kept turned into null operands. The AIR metadata decoder (package S2) reads
//!   these from the canonical module.
//! - Stack objects: every `alloca` is in the entry block with a constant count. Those left are accessed in ways that
//!   need memory: dynamic indices, escaping pointers (calls, pointer stores), or mixed types over the same bytes.
//! - The control flow graph has only reachable blocks; SSA values (phis included) stay as LLVM defines them; `freeze`
//!   stays.
//! - Constants are LLVM's canonical constants; no instruction has all-constant operands where the interpreter
//!   defines its result exactly (instructions whose result the semantics leave open are kept, not folded to a
//!   choice, except `freeze`, whose result is one fixed value: it folds to zero bits for undef or poison).

pub mod copy;
pub mod dce;
pub mod fold;
pub mod inline;
pub mod link;
pub mod mem2reg;
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
    let Some(first) = modules.first() else {
        return Err(refusal(RefusalKind::RejectedByDevice, "pipeline", entry, "no modules"));
    };
    let mut m = link::link(first, &modules[1..], entry)?;
    let f = m
        .function_by_name(entry)
        .filter(|&i| !m.functions[i].is_declaration())
        .ok_or_else(|| refusal(RefusalKind::RejectedByDevice, "function", entry, "no linked module defines it"))?;
    specialize::specialize(&mut m, constants)?;
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
    Err(refusal(RefusalKind::NotImplemented, "canonicalization", &name, "the passes did not reach a fixed point"))
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
        let ops: Vec<_> = nm.ops.iter().filter_map(|&op| c.md(&mut out, op)).collect();
        others.push(NamedMetadata { name: nm.name.clone(), ops });
    }
    c.finish(&mut out);
    // Keep the source order of named metadata.
    let order: Vec<&str> = m.named_metadata.iter().map(|n| n.name.as_str()).collect();
    let mut all: Vec<NamedMetadata> = lists.into_iter().chain(others).collect();
    all.sort_by_key(|n| order.iter().position(|o| *o == n.name).unwrap_or(usize::MAX));
    out.named_metadata = all;
    Canonical { module: out, function: nf as usize }
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
                RefusalKind::NotImplemented,
                &format!("function @{}", g.common.name),
                name,
                "a defined function other than the entry point is referred to (as a function pointer)",
            ));
        }
    }
    for b in &func.blocks {
        for &id in &b.insts {
            if let crate::ir::Op::Call { callee: Value::Global(GlobalId::Function(c)), .. } = &func.inst(id).op {
                let callee = &m.functions[*c as usize].common.name;
                if !callee.starts_with("air.") && !callee.starts_with("llvm.") {
                    return Err(refusal_at(
                        m,
                        f,
                        id,
                        RefusalKind::NotImplemented,
                        &format!("call of @{}", callee),
                        "no linked module defines the function, and it is not an intrinsic",
                    ));
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn refusal(kind: RefusalKind, construct: &str, function: &str, reason: &str) -> Refusal {
    Refusal {
        kind,
        construct: construct.to_string(),
        location: Some(Location { function: function.to_string(), instruction: None }),
        reason: reason.to_string(),
        reference: None,
    }
}

pub(crate) fn refusal_at(
    m: &Module,
    f: usize,
    i: crate::ir::InstId,
    kind: RefusalKind,
    construct: &str,
    reason: &str,
) -> Refusal {
    Refusal {
        kind,
        construct: construct.to_string(),
        location: Some(Location {
            function: m.functions[f].common.name.clone(),
            instruction: Some(crate::ir::print::print_instruction(m, f, i)),
        }),
        reason: reason.to_string(),
        reference: None,
    }
}
