//! Promotion of stack objects to SSA values (LLVM's `PromoteMemToReg`): phis at the iterated dominance frontier of
//! the blocks that write the object, then a walk of the dominator tree that replaces each load by the value last
//! stored (Cytron et al., "Efficiently computing static single assignment form and the control dependence graph").
//!
//! An `alloca` is promoted when it is in the entry block with a count of one and every use is a non-volatile,
//! non-atomic load of exactly its type, a store of exactly its type to it, or a lifetime marker on it. Its memory
//! starts uninitialized, so a load before any store reads `undef` (LangRef `alloca`); `llvm.lifetime.start` makes
//! the contents uninitialized again and `llvm.lifetime.end` makes later loads undefined, both of which `undef`
//! refines.

use std::collections::{HashMap, HashSet};

use crate::ir::{ConstKind, GlobalId, InstId, Module, Op, TypeId, Value};

use super::util::{dominance_frontiers, dominators, kind_const, remove_unreachable_blocks, successors};

/// Promotes what it can in function `f`; returns whether anything changed.
pub fn mem2reg(m: &mut Module, f: usize) -> bool {
    remove_unreachable_blocks(&mut m.functions[f]);
    let allocas = promotable(m, f);
    if allocas.is_empty() {
        return false;
    }
    let func = &m.functions[f];
    let nblocks = func.blocks.len();
    let idom = dominators(func);
    let df = dominance_frontiers(func, &idom);
    let index: HashMap<InstId, usize> = allocas.iter().enumerate().map(|(k, (a, _))| (*a, k)).collect();
    // Which blocks define which object.
    let mut defs: Vec<Vec<usize>> = vec![Vec::new(); allocas.len()];
    for (b, block) in func.blocks.iter().enumerate() {
        for &i in &block.insts {
            if let Some(a) = target(m, f, i, &index)
                && !matches!(func.inst(i).op, Op::Load { .. })
                && !defs[a].contains(&b)
            {
                defs[a].push(b);
            }
        }
    }
    // Phi placement.
    let mut phis: Vec<HashMap<usize, InstId>> = vec![HashMap::new(); nblocks];
    let mut new_phis = Vec::new();
    for (a, (_, ty)) in allocas.iter().enumerate() {
        let mut work = defs[a].clone();
        let mut placed: HashSet<usize> = HashSet::new();
        while let Some(b) = work.pop() {
            for &d in &df[b] {
                if placed.insert(d) {
                    new_phis.push((d, a, *ty));
                    if !defs[a].contains(&d) {
                        work.push(d);
                    }
                }
            }
        }
    }
    for (b, a, ty) in new_phis {
        let func = &mut m.functions[f];
        let p = super::util::new_inst(func, ty, Op::Phi { incoming: Vec::new(), fmf: Default::default() });
        func.blocks[b].insts.insert(0, p);
        phis[b].insert(a, p);
    }
    // Renaming over the dominator tree.
    let undefs: Vec<Value> = allocas.iter().map(|(_, ty)| kind_const(m, *ty, ConstKind::Undef)).collect();
    let mut children = vec![Vec::new(); nblocks];
    for (b, d) in idom.iter().enumerate() {
        if let Some(p) = *d
            && p != b
        {
            children[p].push(b);
        }
    }
    let mut replace: HashMap<InstId, Value> = HashMap::new();
    let mut remove: Vec<InstId> = Vec::new();
    let resolve = super::util::resolve;
    let mut stack: Vec<(usize, Vec<Value>)> = vec![(0, undefs.clone())];
    while let Some((b, mut cur)) = stack.pop() {
        for (&a, &p) in &phis[b] {
            cur[a] = Value::Inst(p);
        }
        let insts = m.functions[f].blocks[b].insts.clone();
        for i in insts {
            let Some(a) = target(m, f, i, &index) else { continue };
            match &m.functions[f].inst(i).op {
                Op::Load { .. } => {
                    replace.insert(i, resolve(&replace, cur[a]));
                }
                Op::Store { value, .. } => {
                    cur[a] = resolve(&replace, *value);
                }
                _ => {
                    // A lifetime marker.
                    cur[a] = undefs[a];
                }
            }
            remove.push(i);
        }
        for s in successors(&m.functions[f], b) {
            for (&a, &p) in &phis[s] {
                let v = resolve(&replace, cur[a]);
                if let Op::Phi { incoming, .. } = &mut m.functions[f].insts[p.0 as usize].op {
                    incoming.push((v, crate::ir::BlockId(b as u32)));
                }
            }
        }
        for &c in children[b].iter().rev() {
            stack.push((c, cur.clone()));
        }
    }
    let func = &mut m.functions[f];
    super::util::apply_replacements(func, &replace);
    let dead: HashSet<InstId> = remove.into_iter().chain(allocas.iter().map(|(a, _)| *a)).collect();
    super::util::remove_insts(func, &dead);
    true
}

/// The promoted object an instruction accesses, if it is a load, store or lifetime marker of one.
fn target(m: &Module, f: usize, i: InstId, index: &HashMap<InstId, usize>) -> Option<usize> {
    let func = &m.functions[f];
    let p = match &func.inst(i).op {
        Op::Load { ptr, .. } => *ptr,
        Op::Store { ptr, .. } => *ptr,
        Op::Call { args, .. } if is_lifetime(m, &func.inst(i).op) => *args.last()?,
        _ => return None,
    };
    match p {
        Value::Inst(a) => index.get(&a).copied(),
        _ => None,
    }
}

fn is_lifetime(m: &Module, op: &Op) -> bool {
    matches!(op, Op::Call { callee: Value::Global(GlobalId::Function(c)), .. }
        if m.functions[*c as usize].common.name.starts_with("llvm.lifetime."))
}

/// The promotable allocas of `f` and their types.
fn promotable(m: &Module, f: usize) -> Vec<(InstId, TypeId)> {
    let func = &m.functions[f];
    let Some(entry) = func.blocks.first() else { return Vec::new() };
    let mut cands: HashMap<InstId, TypeId> = HashMap::new();
    for &i in &entry.insts {
        if let Op::Alloca { allocated, count, inalloca: false, .. } = &func.inst(i).op
            && super::util::const_int(m, *count) == Some(1)
            && m.types.is_first_class(*allocated)
            && m.types.is_sized(*allocated)
        {
            cands.insert(i, *allocated);
        }
    }
    if cands.is_empty() {
        return Vec::new();
    }
    let value_type = |v: Value| -> Option<TypeId> {
        match v {
            Value::Inst(i) => Some(func.inst(i).ty),
            Value::Arg(a) => Some(func.args[a as usize].ty),
            Value::Const(c) => Some(m.constant(c).ty),
            _ => None,
        }
    };
    let mut bad: HashSet<InstId> = HashSet::new();
    for b in &func.blocks {
        for &i in &b.insts {
            let inst = func.inst(i);
            let ok_use = |a: InstId, operand_k: usize| -> bool {
                let ty = cands[&a];
                match &inst.op {
                    Op::Load { volatile: false, atomic: None, .. } => operand_k == 0 && inst.ty == ty,
                    Op::Store { value, volatile: false, atomic: None, .. } => {
                        operand_k == 1 && *value != Value::Inst(a) && value_type(*value) == Some(ty)
                    }
                    Op::Call { args, .. } if is_lifetime(m, &inst.op) => operand_k == args.len() - 1,
                    _ => false,
                }
            };
            for (k, v) in inst.op.operands().iter().enumerate() {
                if let Value::Inst(a) = v
                    && cands.contains_key(a)
                    && !ok_use(*a, k)
                {
                    bad.insert(*a);
                }
            }
        }
    }
    let mut out: Vec<(InstId, TypeId)> = cands.into_iter().filter(|(a, _)| !bad.contains(a)).collect();
    out.sort_by_key(|(a, _)| a.0);
    out
}
