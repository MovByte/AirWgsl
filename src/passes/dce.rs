//! Dead code elimination.
//!
//! Live are the instructions with effects (stores, calls that may write memory or not return, atomics, fences,
//! volatile accesses, terminators) and, transitively, their operands; the rest goes, phi cycles included. Removing
//! a non-volatile load is removing at most undefined behavior (an access the program may not make), which LangRef
//! allows. Calls of declarations count as free of effects when they neither access memory nor fail to return
//! (`memory(none)`, or `memory(read)`, with `willreturn` and `nounwind`). `llvm.lifetime.*`, `llvm.assume`,
//! `llvm.experimental.noalias.scope.decl` and `llvm.dbg.*` go too: each only removes undefined behavior or carries
//! no meaning.
//!
//! A stack object that nothing reads (every use is a store to it, a memset of it, a copy into it, a lifetime marker,
//! or a pointer derived from it only for those) goes with all of its stores.

use std::collections::HashSet;

use crate::ir::{AttrKind, Attribute, GlobalId, InstId, Module, Op, Value};

use super::util::remove_unreachable_blocks;

/// Removes dead code in function `f`; returns whether anything changed.
pub fn dce(m: &mut Module, f: usize) -> bool {
    let mut changed = remove_unreachable_blocks(&mut m.functions[f]);
    changed |= dead_allocas(m, f);
    let func = &m.functions[f];
    let mut live: HashSet<InstId> = HashSet::new();
    let mut work: Vec<InstId> = Vec::new();
    for b in &func.blocks {
        for &i in &b.insts {
            if has_effect(m, f, i) {
                live.insert(i);
                work.push(i);
            }
        }
    }
    while let Some(i) = work.pop() {
        for v in func.inst(i).op.operands() {
            if let Value::Inst(u) = v
                && live.insert(u)
            {
                work.push(u);
            }
        }
    }
    let func = &mut m.functions[f];
    for b in &mut func.blocks {
        let before = b.insts.len();
        b.insts.retain(|i| live.contains(i));
        changed |= b.insts.len() != before;
    }
    changed
}

/// Whether an instruction must stay whatever its uses.
fn has_effect(m: &Module, f: usize, i: InstId) -> bool {
    let inst = m.functions[f].inst(i);
    match &inst.op {
        Op::Store { .. } | Op::AtomicRmw { .. } | Op::CmpXchg { .. } | Op::Fence { .. } | Op::VaArg { .. } => true,
        Op::Load { volatile, atomic, .. } => *volatile || atomic.is_some(),
        Op::Call { callee, attrs, .. } => match callee {
            Value::Global(GlobalId::Function(c)) => {
                let d = &m.functions[*c as usize];
                let n = d.common.name.as_str();
                if n.starts_with("llvm.lifetime.")
                    || n == "llvm.assume"
                    || n.starts_with("llvm.experimental.noalias.scope.decl")
                    || n.starts_with("llvm.dbg.")
                {
                    return false;
                }
                if !d.is_declaration() {
                    return true;
                }
                let fa = m.attrs(d.attrs.function);
                let ca = m.attrs(attrs.function);
                let returns = (fa.has(AttrKind::WillReturn) || ca.has(AttrKind::WillReturn))
                    && (fa.has(AttrKind::NoUnwind) || ca.has(AttrKind::NoUnwind));
                let reads_only = [fa, ca].iter().any(|s| {
                    s.has(AttrKind::ReadNone)
                        || s.has(AttrKind::ReadOnly)
                        || matches!(s.get(AttrKind::Memory), Some(Attribute::Int(_, me)) if me & 0xaa == 0)
                });
                !(returns && reads_only)
            }
            _ => true,
        },
        op => op.is_terminator(),
    }
}

/// Removes stack objects that are written but never read, with their writes.
fn dead_allocas(m: &mut Module, f: usize) -> bool {
    let func = &m.functions[f];
    let uses = super::util::use_map(func);
    let users = |_: &crate::ir::Function, v: Value| -> Vec<InstId> {
        match v {
            Value::Inst(i) => {
                let mut u = uses[i.0 as usize].clone();
                u.dedup();
                u
            }
            _ => Vec::new(),
        }
    };
    let mut doomed: HashSet<InstId> = HashSet::new();
    for b in &func.blocks {
        for &a in &b.insts {
            if !matches!(func.inst(a).op, Op::Alloca { .. }) {
                continue;
            }
            // Pointers derived from the object, and whether every use only writes it.
            let mut derived = vec![a];
            let mut set: HashSet<InstId> = HashSet::from([a]);
            let mut ok = true;
            let mut k = 0;
            while k < derived.len() && ok {
                let p = derived[k];
                k += 1;
                for u in users(func, Value::Inst(p)) {
                    let op = &func.inst(u).op;
                    match op {
                        Op::Store { ptr, value, volatile: false, atomic: None, .. } => {
                            if *value == Value::Inst(p) || *ptr != Value::Inst(p) {
                                ok = false;
                            }
                        }
                        Op::Gep { ptr, .. } if *ptr == Value::Inst(p) => {
                            if set.insert(u) {
                                derived.push(u);
                            }
                        }
                        Op::Cast { op: crate::ir::CastOp::BitCast | crate::ir::CastOp::AddrSpaceCast, .. } => {
                            if set.insert(u) {
                                derived.push(u);
                            }
                        }
                        Op::Call { callee: Value::Global(GlobalId::Function(c)), args, .. } => {
                            let n = m.functions[*c as usize].common.name.as_str();
                            let write_only = n.starts_with("llvm.lifetime.")
                                || (n.starts_with("llvm.memset.") && args[0] == Value::Inst(p))
                                || ((n.starts_with("llvm.memcpy.") || n.starts_with("llvm.memmove."))
                                    && args[0] == Value::Inst(p)
                                    && args[1] != Value::Inst(p));
                            let volatile = n.starts_with("llvm.mem")
                                && args.get(3).is_some_and(|v| super::util::const_int(m, *v) != Some(0));
                            if !write_only || volatile {
                                ok = false;
                            }
                        }
                        _ => ok = false,
                    }
                }
            }
            if ok {
                // The object and every instruction that only writes it or derives a pointer to it.
                for &p in &derived {
                    doomed.insert(p);
                    for u in users(func, Value::Inst(p)) {
                        doomed.insert(u);
                    }
                }
            }
        }
    }
    if doomed.is_empty() {
        return false;
    }
    let func = &mut m.functions[f];
    for b in &mut func.blocks {
        b.insts.retain(|i| !doomed.contains(i));
    }
    true
}
