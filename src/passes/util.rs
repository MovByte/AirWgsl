//! Editing functions in place: operands, uses, the CFG and dominance, and constants built the way LLVM's constant
//! constructors build them (`ConstantInt::get`, `ConstantStruct::get`, `ConstantArray::get`, `ConstantVector::get`).
//!
//! An instruction removed from a function leaves its arena slot unused: blocks list the live instructions, and every
//! analysis here walks blocks.

use std::collections::HashMap;

use crate::ir::{BlockId, ConstId, ConstKind, Constant, Function, InstId, Module, Op, Type, TypeId, Value};

/// The value operands of an instruction, mutably (callee included), in [`Op::operands`] order.
pub fn operands_mut(op: &mut Op) -> Vec<&mut Value> {
    match op {
        Op::FNeg { value, .. } | Op::Cast { value, .. } | Op::Freeze { value } => vec![value],
        Op::Binary { lhs, rhs, .. } | Op::ICmp { lhs, rhs, .. } | Op::FCmp { lhs, rhs, .. } => vec![lhs, rhs],
        Op::Select { cond, t, f, .. } => vec![cond, t, f],
        Op::Phi { incoming, .. } => incoming.iter_mut().map(|(v, _)| v).collect(),
        Op::Alloca { count, .. } => vec![count],
        Op::Load { ptr, .. } => vec![ptr],
        Op::Store { value, ptr, .. } => vec![value, ptr],
        Op::Gep { ptr, indices, .. } => std::iter::once(ptr).chain(indices.iter_mut()).collect(),
        Op::ExtractElement { vec, idx } => vec![vec, idx],
        Op::InsertElement { vec, elt, idx } => vec![vec, elt, idx],
        Op::ShuffleVector { a, b, .. } => vec![a, b],
        Op::ExtractValue { agg, .. } => vec![agg],
        Op::InsertValue { agg, value, .. } => vec![agg, value],
        Op::Call { callee, args, bundles, .. } => {
            let mut v: Vec<&mut Value> = args.iter_mut().collect();
            for b in bundles.iter_mut() {
                v.extend(b.inputs.iter_mut());
            }
            v.push(callee);
            v
        }
        Op::AtomicRmw { ptr, value, .. } => vec![ptr, value],
        Op::CmpXchg { ptr, cmp, new, .. } => vec![ptr, cmp, new],
        Op::Fence { .. } | Op::Unreachable | Op::Br { .. } => vec![],
        Op::VaArg { list } => vec![list],
        Op::Ret { value } => value.iter_mut().collect(),
        Op::CondBr { cond, .. } => vec![cond],
        Op::Switch { cond, .. } => vec![cond],
        Op::IndirectBr { addr, .. } => vec![addr],
    }
}

/// The successor blocks of a terminator, mutably.
pub fn successors_mut(op: &mut Op) -> Vec<&mut BlockId> {
    match op {
        Op::Br { dest } => vec![dest],
        Op::CondBr { t, f, .. } => vec![t, f],
        Op::Switch { default, cases, .. } => std::iter::once(default).chain(cases.iter_mut().map(|(_, b)| b)).collect(),
        Op::IndirectBr { dests, .. } => dests.iter_mut().collect(),
        _ => vec![],
    }
}

/// The terminator of block `b`.
pub fn terminator(f: &Function, b: usize) -> Option<InstId> {
    f.blocks[b].insts.last().copied().filter(|&i| f.inst(i).op.is_terminator())
}

pub fn successors(f: &Function, b: usize) -> Vec<usize> {
    match terminator(f, b) {
        Some(t) => f.inst(t).op.successors().into_iter().map(|s| s.0 as usize).collect(),
        None => Vec::new(),
    }
}

/// Predecessors of every block (with repeats for multiple edges, as phis count them).
pub fn predecessors(f: &Function) -> Vec<Vec<usize>> {
    let mut p = vec![Vec::new(); f.blocks.len()];
    for b in 0..f.blocks.len() {
        for s in successors(f, b) {
            p[s].push(b);
        }
    }
    p
}

/// Blocks reachable from the entry, in reverse postorder.
pub fn reverse_postorder(f: &Function) -> Vec<usize> {
    let n = f.blocks.len();
    let mut seen = vec![false; n];
    let mut post = Vec::with_capacity(n);
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    if n == 0 {
        return post;
    }
    seen[0] = true;
    while let Some(&mut (b, ref mut k)) = stack.last_mut() {
        let succ = successors(f, b);
        if *k < succ.len() {
            let s = succ[*k];
            *k += 1;
            if !seen[s] {
                seen[s] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(b);
            stack.pop();
        }
    }
    post.reverse();
    post
}

/// Immediate dominators (Cooper, Harvey and Kennedy, "A Simple, Fast Dominance Algorithm"), for reachable blocks;
/// the entry's is itself, unreachable blocks have `None`.
pub fn dominators(f: &Function) -> Vec<Option<usize>> {
    let rpo = reverse_postorder(f);
    let mut order = vec![usize::MAX; f.blocks.len()];
    for (i, &b) in rpo.iter().enumerate() {
        order[b] = i;
    }
    let preds = predecessors(f);
    let mut idom: Vec<Option<usize>> = vec![None; f.blocks.len()];
    if rpo.is_empty() {
        return idom;
    }
    idom[rpo[0]] = Some(rpo[0]);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new: Option<usize> = None;
            for &p in &preds[b] {
                if idom[p].is_none() {
                    continue;
                }
                new = Some(match new {
                    None => p,
                    Some(cur) => {
                        let (mut x, mut y) = (p, cur);
                        while x != y {
                            while order[x] > order[y] {
                                x = idom[x].unwrap();
                            }
                            while order[y] > order[x] {
                                y = idom[y].unwrap();
                            }
                        }
                        x
                    }
                });
            }
            if new.is_some() && idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom
}

/// Dominance frontiers of the reachable blocks.
pub fn dominance_frontiers(f: &Function, idom: &[Option<usize>]) -> Vec<Vec<usize>> {
    let preds = predecessors(f);
    let mut df = vec![Vec::new(); f.blocks.len()];
    for b in 0..f.blocks.len() {
        let Some(ib) = idom[b] else { continue };
        let ps: Vec<usize> = preds[b].iter().copied().filter(|p| idom[*p].is_some()).collect();
        if ps.len() < 2 {
            continue;
        }
        for p in ps {
            let mut r = p;
            while r != ib {
                if !df[r].contains(&b) {
                    df[r].push(b);
                }
                let next = idom[r].unwrap();
                if next == r {
                    break;
                }
                r = next;
            }
        }
    }
    df
}

/// The block each live instruction is in.
pub fn inst_blocks(f: &Function) -> HashMap<InstId, usize> {
    let mut m = HashMap::new();
    for (b, block) in f.blocks.iter().enumerate() {
        for &i in &block.insts {
            m.insert(i, b);
        }
    }
    m
}

/// How many times each instruction's result is used by live instructions.
pub fn use_counts(f: &Function) -> Vec<u32> {
    let mut n = vec![0u32; f.insts.len()];
    for b in &f.blocks {
        for &i in &b.insts {
            for v in f.inst(i).op.operands() {
                if let Value::Inst(u) = v {
                    n[u.0 as usize] += 1;
                }
            }
        }
    }
    n
}

/// The live instructions that use `v`.
pub fn users(f: &Function, v: Value) -> Vec<InstId> {
    let mut out = Vec::new();
    for b in &f.blocks {
        for &i in &b.insts {
            if f.inst(i).op.operands().contains(&v) {
                out.push(i);
            }
        }
    }
    out
}

/// The users of each instruction (by instruction index), among the live instructions; an instruction that uses a
/// value twice is listed twice.
pub fn use_map(f: &Function) -> Vec<Vec<InstId>> {
    let mut u = vec![Vec::new(); f.insts.len()];
    for b in &f.blocks {
        for &i in &b.insts {
            for v in f.inst(i).op.operands() {
                if let Value::Inst(d) = v {
                    u[d.0 as usize].push(i);
                }
            }
        }
    }
    u
}

/// Follows a replacement map to its end.
pub fn resolve(map: &HashMap<InstId, Value>, mut v: Value) -> Value {
    while let Value::Inst(i) = v {
        match map.get(&i) {
            Some(&n) if n != v => v = n,
            _ => break,
        }
    }
    v
}

/// Replaces every use of each instruction in `map` by its (resolved) replacement, in one pass over the function.
pub fn apply_replacements(f: &mut Function, map: &HashMap<InstId, Value>) {
    if map.is_empty() {
        return;
    }
    let live: Vec<InstId> = f.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
    for i in live {
        for o in operands_mut(&mut f.insts[i.0 as usize].op) {
            if let Value::Inst(x) = *o
                && map.contains_key(&x)
            {
                *o = resolve(map, *o);
            }
        }
    }
}

/// Removes every instruction in `dead` from the blocks.
pub fn remove_insts(f: &mut Function, dead: &std::collections::HashSet<InstId>) {
    if dead.is_empty() {
        return;
    }
    for b in &mut f.blocks {
        b.insts.retain(|i| !dead.contains(i));
    }
}

/// Replaces every use of `from` in live instructions by `to`.
pub fn replace_uses(f: &mut Function, from: Value, to: Value) {
    let live: Vec<InstId> = f.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
    for i in live {
        for o in operands_mut(&mut f.insts[i.0 as usize].op) {
            if *o == from {
                *o = to;
            }
        }
    }
}

/// Removes instruction `i` from its block (its slot stays, unused).
pub fn remove_inst(f: &mut Function, i: InstId) {
    for b in &mut f.blocks {
        if let Some(k) = b.insts.iter().position(|&x| x == i) {
            b.insts.remove(k);
            return;
        }
    }
}

/// Removes the phi operands that come from block `pred` into block `b` (one per edge removed).
pub fn remove_phi_edge(f: &mut Function, b: usize, pred: usize) {
    let ids: Vec<InstId> = f.blocks[b].insts.clone();
    for i in ids {
        if let Op::Phi { incoming, .. } = &mut f.insts[i.0 as usize].op {
            if let Some(k) = incoming.iter().position(|(_, p)| p.0 as usize == pred) {
                incoming.remove(k);
            }
        } else {
            break;
        }
    }
}

/// Removes blocks unreachable from the entry, renumbering the rest and dropping phi operands from removed blocks.
/// Returns whether anything changed.
pub fn remove_unreachable_blocks(f: &mut Function) -> bool {
    let rpo = reverse_postorder(f);
    let mut reach = vec![false; f.blocks.len()];
    for b in rpo {
        reach[b] = true;
    }
    if reach.iter().all(|&r| r) {
        return false;
    }
    let mut map = vec![u32::MAX; f.blocks.len()];
    let mut n = 0u32;
    for (b, &r) in reach.iter().enumerate() {
        if r {
            map[b] = n;
            n += 1;
        }
    }
    let old = std::mem::take(&mut f.blocks);
    for (b, block) in old.into_iter().enumerate() {
        if reach[b] {
            f.blocks.push(block);
        }
    }
    let live: Vec<InstId> = f.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
    for i in live {
        let op = &mut f.insts[i.0 as usize].op;
        if let Op::Phi { incoming, .. } = op {
            incoming.retain(|(_, p)| map[p.0 as usize] != u32::MAX);
            for (_, p) in incoming.iter_mut() {
                *p = BlockId(map[p.0 as usize]);
            }
        }
        for s in successors_mut(op) {
            *s = BlockId(map[s.0 as usize]);
        }
    }
    // Use-list order refers to the old block uses; it has no meaning for the program.
    for b in &mut f.blocks {
        b.use_order = None;
    }
    true
}

/// Appends an instruction to the arena (not to a block).
pub fn new_inst(f: &mut Function, ty: TypeId, op: Op) -> InstId {
    f.insts.push(crate::ir::Instruction { ty, name: None, op, md: Vec::new(), dbg: None });
    InstId(f.insts.len() as u32 - 1)
}

// Constants -----------------------------------------------------------------------------------------------------------

pub fn int_const(m: &mut Module, ty: TypeId, v: u128) -> Value {
    let bits = m.types.int_bits(ty).unwrap_or(64);
    let n = bits.div_ceil(64).max(1) as usize;
    let v = if bits >= 128 { v } else { v & ((1u128 << bits) - 1) };
    let words: Vec<u64> = (0..n).map(|i| if i < 2 { (v >> (64 * i)) as u64 } else { 0 }).collect();
    Value::Const(m.intern_const(Constant { ty, kind: ConstKind::Int(words) }))
}

pub fn i1_const(m: &mut Module, b: bool) -> Value {
    let t = m.types.int(1);
    int_const(m, t, b as u128)
}

pub fn kind_const(m: &mut Module, ty: TypeId, kind: ConstKind) -> Value {
    Value::Const(m.intern_const(Constant { ty, kind }))
}

/// The null value of a type (`Constant::getNullValue`).
pub fn null_const(m: &mut Module, ty: TypeId) -> Value {
    let kind = match m.types.get(ty) {
        Type::Int(_) => {
            return int_const(m, ty, 0);
        }
        Type::Ptr(_) => ConstKind::NullPtr,
        Type::Token => ConstKind::TokenNone,
        Type::Half | Type::BFloat | Type::Float | Type::Double => ConstKind::Float(0),
        _ => ConstKind::Zero,
    };
    kind_const(m, ty, kind)
}

fn is_null(m: &Module, v: Value) -> bool {
    match v {
        Value::Const(c) => match &m.constant(c).kind {
            ConstKind::Zero | ConstKind::NullPtr | ConstKind::TokenNone => true,
            ConstKind::Int(w) => w.iter().all(|x| *x == 0),
            // +0.0 only: -0.0 is not a null value.
            ConstKind::Float(b) => *b == 0,
            _ => false,
        },
        _ => false,
    }
}

fn kind_is(m: &Module, v: Value, f: impl Fn(&ConstKind) -> bool) -> bool {
    matches!(v, Value::Const(c) if f(&m.constant(c).kind))
}

/// `ConstantStruct::get`.
pub fn struct_const(m: &mut Module, ty: TypeId, elems: Vec<Value>) -> Value {
    if !elems.is_empty() {
        if elems.iter().all(|&e| is_null(m, e)) {
            return kind_const(m, ty, ConstKind::Zero);
        }
        if elems.iter().all(|&e| kind_is(m, e, |k| matches!(k, ConstKind::Poison))) {
            return kind_const(m, ty, ConstKind::Poison);
        }
        if elems.iter().all(|&e| kind_is(m, e, |k| matches!(k, ConstKind::Undef))) {
            return kind_const(m, ty, ConstKind::Undef);
        }
    } else {
        return kind_const(m, ty, ConstKind::Zero);
    }
    kind_const(m, ty, ConstKind::Struct(elems))
}

/// `ConstantArray::get` and `ConstantVector::get`: data sequences for simple element types.
pub fn sequence_const(m: &mut Module, ty: TypeId, elems: Vec<Value>) -> Value {
    let is_vector = m.types.is_vector(ty);
    if elems.is_empty() {
        return kind_const(m, ty, ConstKind::Zero);
    }
    let c = elems[0];
    if elems.iter().all(|&e| e == c) {
        if kind_is(m, c, |k| matches!(k, ConstKind::Poison)) {
            return kind_const(m, ty, ConstKind::Poison);
        }
        if kind_is(m, c, |k| matches!(k, ConstKind::Undef)) {
            return kind_const(m, ty, ConstKind::Undef);
        }
        if is_null(m, c) {
            return kind_const(m, ty, ConstKind::Zero);
        }
    }
    let elem = match m.types.get(ty) {
        Type::Array { elem, .. } | Type::Vector { elem, .. } => *elem,
        _ => unreachable!("a sequence constant of a non-sequence type"),
    };
    let simple = matches!(
        m.types.get(elem),
        Type::Int(8 | 16 | 32 | 64) | Type::Half | Type::BFloat | Type::Float | Type::Double
    );
    if simple {
        let bits: Option<Vec<u64>> = elems
            .iter()
            .map(|&e| match e {
                Value::Const(k) => match &m.constant(k).kind {
                    ConstKind::Int(w) => Some(w[0]),
                    ConstKind::Float(b) => Some(*b as u64),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        if let Some(bits) = bits {
            let kind = if bits.iter().all(|b| *b == 0) {
                ConstKind::Zero
            } else if is_vector {
                ConstKind::DataVector(bits)
            } else {
                ConstKind::DataArray(bits)
            };
            return kind_const(m, ty, kind);
        }
    }
    kind_const(m, ty, if is_vector { ConstKind::Vector(elems) } else { ConstKind::Array(elems) })
}

/// The integer value of a constant (`ConstantInt`, or a splat of one).
pub fn const_int(m: &Module, v: Value) -> Option<u128> {
    let Value::Const(c) = v else { return None };
    match &m.constant(c).kind {
        ConstKind::Int(w) => {
            let mut x = 0u128;
            for (i, word) in w.iter().enumerate().take(2) {
                x |= (*word as u128) << (64 * i);
            }
            Some(x)
        }
        ConstKind::Zero if m.types.is_int(m.constant(c).ty) => Some(0),
        _ => None,
    }
}

pub fn const_id(v: Value) -> Option<ConstId> {
    match v {
        Value::Const(c) => Some(c),
        _ => None,
    }
}

/// Merges each block into its predecessor when that predecessor ends with an unconditional branch to it and is its
/// only predecessor (the phis of such a block have one operand each and become it). Returns whether any merged.
pub fn merge_blocks(f: &mut Function) -> bool {
    let mut changed = false;
    loop {
        let preds = predecessors(f);
        let mut found = None;
        for (b, p) in preds.iter().enumerate() {
            if b == 0 || p.len() != 1 || p[0] == b {
                continue;
            }
            let a = p[0];
            if let Some(t) = terminator(f, a)
                && matches!(f.inst(t).op, Op::Br { .. })
            {
                found = Some((a, b, t));
                break;
            }
        }
        let Some((a, b, t)) = found else { return changed };
        // Phis of b: one operand each.
        let ids = f.blocks[b].insts.clone();
        let mut rest = Vec::new();
        for i in ids {
            if let Op::Phi { incoming, .. } = &f.inst(i).op {
                let v = incoming[0].0;
                replace_uses(f, Value::Inst(i), v);
            } else {
                rest.push(i);
            }
        }
        f.blocks[a].insts.retain(|&x| x != t);
        f.blocks[a].insts.extend(rest);
        f.blocks[b].insts.clear();
        // Successors of b now have a as their predecessor.
        let succ = successors(f, a);
        for s in succ {
            let ids = f.blocks[s].insts.clone();
            for i in ids {
                if let Op::Phi { incoming, .. } = &mut f.insts[i.0 as usize].op {
                    for (_, p) in incoming.iter_mut() {
                        if p.0 as usize == b {
                            *p = BlockId(a as u32);
                        }
                    }
                } else {
                    break;
                }
            }
        }
        // b is now unreachable and empty; drop it.
        let mut map: Vec<u32> = (0..f.blocks.len() as u32).collect();
        for x in map.iter_mut().skip(b + 1) {
            *x -= 1;
        }
        f.blocks.remove(b);
        let live: Vec<InstId> = f.blocks.iter().flat_map(|bb| bb.insts.iter().copied()).collect();
        for i in live {
            let op = &mut f.insts[i.0 as usize].op;
            if let Op::Phi { incoming, .. } = op {
                for (_, p) in incoming.iter_mut() {
                    *p = BlockId(map[p.0 as usize]);
                }
            }
            for s in successors_mut(op) {
                *s = BlockId(map[s.0 as usize]);
            }
        }
        for bb in &mut f.blocks {
            bb.use_order = None;
        }
        changed = true;
    }
}
