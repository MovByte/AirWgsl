//! What pipeline reflection needs from canonicalization ([`crate::reflect::Inputs`]): the values the static
//! initializer leaves in the globals an entry point's record names, and the arguments the canonical function no
//! longer uses.
//!
//! The values come from the canonical module, whose globals hold what the function constants and the
//! `air.static_init` constructors made of them; they are keyed by the stage module's global indices, which is how
//! the AIR semantic layer names them ([`crate::air::Predicate::global`], [`crate::air::Slot::Constant`]).

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::interp::{self, Memory};
use crate::ir::layout::DataLayout;
use crate::ir::{GlobalId, MdId, Metadata, Module, Type, Value};
use crate::reflect::Inputs;

use super::{Canonical, ENTRY_LISTS, record_function, specialize};

/// The reflection inputs of canonical entry point `c`:
///
/// - `predicates`: for every global the entry point's record names, whether its value has a set bit (the rule by
///   which Metal keeps an argument that carries `[[function_constant(p)]]`; see [`specialize`]). A value with
///   undef bits counts them clear, as Metal drops an argument whose predicate has no value.
/// - `slots`: for every such global of an integer type of at most 32 bits whose value is defined, that value (a
///   binding index or count given by a function constant, or computed from one by the static initializer).
/// - `unused_arguments`: the arguments of the canonical function that no instruction uses.
///
/// A global whose value the canonicalization could not determine has no entry; reflection then refuses the
/// binding or argument that needs it, naming the global.
pub fn inputs(c: &Canonical) -> Inputs {
    let m = &c.module;
    let back: BTreeMap<u32, u32> = c.globals.iter().map(|(s, d)| (*d, *s)).collect();
    let mut predicates = BTreeMap::new();
    let mut slots = BTreeMap::new();
    let dl = DataLayout::parse(&m.datalayout).unwrap_or_default();
    let mut mem = Memory::new();
    for g in record_globals(m, c.function) {
        let Some(&stage) = back.get(&g) else { continue };
        let Ok(r) = interp::global_region(m, g, &mut mem) else { continue };
        let ty = m.globals[g as usize].value_type;
        let size = dl.store_size(&m.types, ty).unwrap_or(0) as usize;
        let bytes = &mem.region(r).bytes[..size.min(mem.region(r).bytes.len())];
        predicates.insert(stage, specialize::predicate_present(bytes));
        if let Type::Int(w) = m.types.get(ty)
            && *w <= 32
            && bytes.iter().all(|b| b.is_plain())
        {
            let mut v = 0u32;
            for (k, b) in bytes.iter().enumerate() {
                v |= (b.v as u32) << (8 * k);
            }
            let mask = if *w == 32 { u32::MAX } else { (1u32 << w) - 1 };
            slots.insert(stage, v & mask);
        }
    }
    let func = &m.functions[c.function];
    let mut used = HashSet::new();
    for b in &func.blocks {
        for &i in &b.insts {
            for v in func.inst(i).op.operands() {
                if let Value::Arg(a) = v {
                    used.insert(a);
                }
            }
        }
    }
    let unused_arguments: BTreeSet<u32> = (0..func.args.len() as u32).filter(|a| !used.contains(a)).collect();
    Inputs { predicates, slots, unused_arguments }
}

/// The global variables the entry-point records of function `f` refer to, at any depth.
fn record_globals(m: &Module, f: usize) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let mut seen = HashSet::new();
    let mut stack: Vec<MdId> = Vec::new();
    for nm in m.named_metadata.iter().filter(|n| ENTRY_LISTS.contains(&n.name.as_str())) {
        stack.extend(nm.ops.iter().copied().filter(|&r| record_function(m, r) == Some(f)));
    }
    while let Some(md) = stack.pop() {
        if !seen.insert(md) {
            continue;
        }
        match m.md(md) {
            Metadata::Value { value: Value::Global(GlobalId::Var(g)), .. } => {
                out.insert(*g);
            }
            Metadata::Tuple { ops, .. } => stack.extend(ops.iter().flatten().copied()),
            Metadata::ArgList(items) => stack.extend(items.iter().copied()),
            _ => {}
        }
    }
    out
}
