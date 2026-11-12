//! Static initialization: running the module's constructors at compile time.
//!
//! The front end computes constant-address-space globals that depend on function constants (`constant int x =
//! is_function_constant_defined(c) ? c : 7;`, the `__metal_implicit_fc_pred_*` argument predicates) in constructors:
//! functions in section `air.static_init`, listed in `llvm.global_ctors`. Once the function constants have values
//! those are compile-time computations. They run here in the interpreter, without uninterpreted calls, in
//! `llvm.global_ctors` order (by priority, then position); every global they write becomes `constant` with the
//! written contents as its initializer, and the constructor list is dropped (extraction leaves it out).
//!
//! A constructor that writes a global outside the constant address space (2) would initialize memory the program
//! can change, which no measured module does: refused. So is a constructor the interpreter cannot evaluate
//! exactly, and a function in section `air.static_init` that `llvm.global_ctors` does not list.

use crate::api::Refusal;
use crate::interp::memory::Layout;
use crate::interp::{self, Memory, Options};
use crate::ir::layout::DataLayout;
use crate::ir::{ConstKind, GlobalId, Module, Value};

use super::{Item, refusal, values};

/// The constructors `llvm.global_ctors` lists, in the order they run.
pub fn constructors(m: &Module) -> Vec<usize> {
    let Some(g) = m.globals.iter().find(|g| g.common.name == "llvm.global_ctors") else { return Vec::new() };
    let Some(Value::Const(c)) = g.initializer else { return Vec::new() };
    let ConstKind::Array(items) = &m.constant(c).kind else { return Vec::new() };
    let mut out: Vec<(u64, usize, usize)> = Vec::new();
    for (pos, &it) in items.iter().enumerate() {
        let Value::Const(e) = it else { continue };
        let ConstKind::Struct(fields) = &m.constant(e).kind else { continue };
        let prio = super::util::const_int(m, fields[0]).unwrap_or(65535) as u64;
        if let Value::Global(GlobalId::Function(f)) = fields[1] {
            out.push((prio, pos, f as usize));
        }
    }
    out.sort();
    out.into_iter().map(|(_, _, f)| f).collect()
}

/// Runs the constructors and folds what they wrote into initializers.
#[allow(clippy::result_large_err)]
pub fn run(m: &mut Module) -> Result<(), Refusal> {
    let ctors = constructors(m);
    for (i, f) in m.functions.iter().enumerate() {
        if f.section.as_deref() == Some("air.static_init") && !f.is_declaration() && !ctors.contains(&i) {
            return Err(refusal(
                Item::StaticInit,
                "air.static_init",
                &f.common.name,
                "a static initializer that llvm.global_ctors does not list",
            ));
        }
    }
    if ctors.is_empty() {
        return Ok(());
    }
    let sealed = !writes_constant_space(m, &ctors);
    let mut mem = Memory::new();
    // Every global's region exists before the constructors run, so their writes show against the initial contents.
    let mut before = Vec::new();
    for g in 0..m.globals.len() as u32 {
        if m.globals[g as usize].initializer.is_none() {
            continue;
        }
        if let Ok(r) = interp::global_region(m, g, &mut mem) {
            before.push((g, r, mem.region(r).bytes.clone()));
        }
    }
    let opts = Options::default();
    for &f in &ctors {
        if let Err(e) = interp::run(m, f, &[], &mut mem, &opts) {
            return Err(refusal(
                Item::StaticInit,
                "air.static_init",
                &m.functions[f].common.name,
                &format!("the static initializer cannot be evaluated at compile time: {}", e),
            ));
        }
    }
    let dl = DataLayout::parse(&m.datalayout).map_err(|e| refusal(Item::ModuleUnreadable, "datalayout", "", &e))?;
    for (g, r, initial) in before {
        if mem.region(r).bytes == initial {
            // Unchanged. With no write to the constant address space outside the constructors, the global keeps
            // its initializer for good.
            let gv = &mut m.globals[g as usize];
            if sealed && gv.common.addrspace == 2 && !gv.externally_initialized {
                gv.constant = true;
            }
            continue;
        }
        let gv = &m.globals[g as usize];
        let ty = gv.value_type;
        if gv.common.addrspace != 2 {
            return Err(refusal(
                Item::StaticInit,
                &format!("global @{}", gv.common.name),
                "air.static_init",
                "a static initializer writes a global outside the constant address space",
            ));
        }
        let layout = Layout { dl: &dl, types: &m.types };
        let size = layout.store_size(ty).unwrap_or(0) as usize;
        let now = mem.region(r).bytes[..size].to_vec();
        let v = layout.decode(ty, &now).map_err(|e| refusal(Item::StaticInit, "air.static_init", "", &e))?;
        let name = gv.common.name.clone();
        let init = values::to_const(m, &mem, ty, &v)
            .map_err(|e| refusal(Item::StaticInit, &format!("global @{}", name), "air.static_init", &e))?;
        let gv = &mut m.globals[g as usize];
        gv.initializer = Some(init);
        gv.constant = true;
        gv.externally_initialized = false;
    }
    // A global the constructors created without an initializer (a declaration) cannot have been written: writing it
    // would need its region, which only exists for globals with initializers above or through a run that failed.
    Ok(())
}

/// Whether a function other than the constructors may write memory in the constant address space (2): a store, an
/// atomic, or a call that may write memory, with a pointer of that address space.
fn writes_constant_space(m: &Module, ctors: &[usize]) -> bool {
    use crate::ir::{AttrKind, AttrSet, Attribute, Op, Type, TypeId};
    let effects = |set: &AttrSet| -> u64 {
        if set.has(AttrKind::ReadNone) || set.has(AttrKind::ReadOnly) {
            return 0;
        }
        match set.get(AttrKind::Memory) {
            Some(Attribute::Int(_, me)) => *me,
            _ => 0xff,
        }
    };
    for (fi, f) in m.functions.iter().enumerate() {
        if f.is_declaration() || ctors.contains(&fi) {
            continue;
        }
        let in_constant_space = |v: Value| -> bool {
            let ty: Option<TypeId> = match v {
                Value::Inst(i) => Some(f.inst(i).ty),
                Value::Arg(a) => Some(f.args[a as usize].ty),
                Value::Const(c) => Some(m.constant(c).ty),
                Value::Global(g) => return m.global(g).addrspace == 2,
                _ => None,
            };
            ty.is_some_and(|t| matches!(m.types.get(m.types.scalar(t)), Type::Ptr(2)))
        };
        for b in &f.blocks {
            for &i in &b.insts {
                let writes = match &f.inst(i).op {
                    Op::Store { ptr, .. } | Op::AtomicRmw { ptr, .. } | Op::CmpXchg { ptr, .. } => {
                        in_constant_space(*ptr)
                    }
                    Op::Call { callee, args, attrs, .. } => {
                        let decl = match callee {
                            Value::Global(GlobalId::Function(c)) => {
                                effects(m.attrs(m.functions[*c as usize].attrs.function))
                            }
                            _ => 0xff,
                        };
                        (decl & effects(m.attrs(attrs.function))) & 0xaa != 0
                            && args.iter().any(|a| in_constant_space(*a))
                    }
                    _ => false,
                };
                if writes {
                    return true;
                }
            }
        }
    }
    false
}
