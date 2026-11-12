//! Constant folding with the interpreter's exact semantics, and the simplifications that follow from constants.
//!
//! An instruction whose operands are all constants or globals is evaluated by the interpreter
//! ([`crate::interp::eval_instruction`]); its result replaces it when it is exactly determined: a value with no undef
//! bit (or entirely `undef`) and no choice bit, or poison. A result the semantics leave open (a NaN's payload, the
//! sign of a zero under `nsz`, `llvm.fmuladd`'s fusion) is not chosen here, except for `freeze`, whose result is
//! one fixed value of the program's choosing: zero bits where its operand is undef or poison, the interpreter's
//! choice. Undefined behavior is never folded. Calls fold only for declarations without memory effects.
//!
//! Loads fold from globals that are `constant` and not `externally_initialized`: their contents are the initializer
//! for the whole run. Results that would depend on where the interpreter placed a global (`ptrtoint` of a global, an
//! ordering comparison of addresses) are not folded.
//!
//! The simplifications: a conditional branch or switch on a constant becomes a branch (the dropped edges leave their
//! phis), a `select` on a constant condition becomes the chosen operand, a phi whose operands are all one value (or
//! itself) becomes that value, a cast to the operand's own type and a `getelementptr` with all-zero indices become
//! their operand, and blocks no longer reachable go.

use std::collections::{HashMap, HashSet};

use crate::api::Refusal;
use crate::interp::{self, Memory, Options, Scalar, Val};
use crate::ir::{AttrKind, Attribute, BlockId, CastOp, ConstExpr, ConstKind, GlobalId, InstId, Module, Op, Value};

use super::util::{apply_replacements, const_int, remove_insts, remove_unreachable_blocks, resolve, reverse_postorder};
use super::values;

/// Folds what it can in function `f`; returns whether anything changed.
#[allow(clippy::result_large_err)]
pub fn fold(m: &mut Module, f: usize) -> Result<bool, Refusal> {
    let mut changed = false;
    let mut mem = Memory::new();
    let opts = Options::default();
    loop {
        let mut round = false;
        // Folded instructions map to their values; later instructions read their operands through the map, and
        // the map applies to the whole function once per round.
        let mut map: HashMap<InstId, Value> = HashMap::new();
        let mut dead: HashSet<InstId> = HashSet::new();
        for b in reverse_postorder(&m.functions[f]) {
            let insts = m.functions[f].blocks[b].insts.clone();
            for i in insts {
                if !map.is_empty() {
                    let op = &mut m.functions[f].insts[i.0 as usize].op;
                    for o in super::util::operands_mut(op) {
                        if let Value::Inst(x) = *o
                            && map.contains_key(&x)
                        {
                            *o = resolve(&map, *o);
                        }
                    }
                }
                if let Some(v) = simplify(m, f, i) {
                    map.insert(i, v);
                    dead.insert(i);
                    continue;
                }
                if !foldable(m, f, i) {
                    continue;
                }
                let Ok(Some(v)) = interp::eval_instruction(m, f, i, &mut mem, &opts) else { continue };
                let is_freeze = matches!(m.functions[f].inst(i).op, Op::Freeze { .. });
                let ty = m.functions[f].inst(i).ty;
                if !determined(m, ty, &v, is_freeze) {
                    continue;
                }
                let Ok(c) = values::to_const(m, &mem, ty, &v) else { continue };
                map.insert(i, c);
                dead.insert(i);
            }
        }
        if !dead.is_empty() {
            apply_replacements(&mut m.functions[f], &map);
            remove_insts(&mut m.functions[f], &dead);
            round = true;
        }
        round |= branches(m, f);
        if remove_unreachable_blocks(&mut m.functions[f]) {
            round = true;
        }
        if super::util::merge_blocks(&mut m.functions[f]) {
            round = true;
        }
        if !round {
            return Ok(changed);
        }
        changed = true;
    }
}

/// Whether instruction `i` is one the interpreter may evaluate alone.
fn foldable(m: &Module, f: usize, i: InstId) -> bool {
    let inst = m.functions[f].inst(i);
    let ops = inst.op.operands();
    if !ops.iter().all(|v| matches!(v, Value::Const(_) | Value::Global(_))) {
        return false;
    }
    if !ops.iter().all(|&v| address_free(m, v)) {
        return false;
    }
    match &inst.op {
        Op::FNeg { .. }
        | Op::Binary { .. }
        | Op::FCmp { .. }
        | Op::Select { .. }
        | Op::Gep { .. }
        | Op::ExtractElement { .. }
        | Op::InsertElement { .. }
        | Op::ShuffleVector { .. }
        | Op::ExtractValue { .. }
        | Op::InsertValue { .. }
        | Op::Freeze { .. } => true,
        Op::Cast { op, .. } => !matches!(op, CastOp::PtrToInt),
        Op::ICmp { lhs, .. } => {
            // Pointer comparisons depend on the layout unless both sides are null.
            !matches!(lhs, Value::Global(_)) && !ops.iter().any(|&v| is_pointer_const(m, v))
        }
        Op::Load { ptr, volatile: false, atomic: None, .. } => constant_object(m, *ptr),
        Op::Call { callee: Value::Global(GlobalId::Function(c)), attrs, .. } => {
            let d = &m.functions[*c as usize];
            d.is_declaration() && (no_memory(m.attrs(d.attrs.function)) || no_memory(m.attrs(attrs.function)))
        }
        _ => false,
    }
}

fn no_memory(set: &crate::ir::AttrSet) -> bool {
    set.has(AttrKind::ReadNone) || matches!(set.get(AttrKind::Memory), Some(Attribute::Int(_, 0)))
}

fn is_pointer_const(m: &Module, v: Value) -> bool {
    match v {
        Value::Global(_) => true,
        Value::Const(c) => {
            let k = m.constant(c);
            m.types.is_ptr_or_ptr_vector(k.ty) && !matches!(k.kind, ConstKind::NullPtr | ConstKind::Zero)
        }
        _ => false,
    }
}

/// Whether a constant's value does not depend on where globals are (no `ptrtoint` inside).
fn address_free(m: &Module, v: Value) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        let Value::Const(c) = v else { continue };
        if !seen.insert(c) {
            continue;
        }
        match &m.constant(c).kind {
            ConstKind::Expr(ConstExpr::Cast { op: CastOp::PtrToInt, .. }) => return false,
            ConstKind::Expr(e) => match e {
                ConstExpr::Cast { value, .. } => stack.push(*value),
                ConstExpr::Binary { lhs, rhs, .. } => stack.extend([*lhs, *rhs]),
                ConstExpr::Gep { ptr, indices, .. } => {
                    stack.push(*ptr);
                    stack.extend(indices.iter().copied());
                }
                ConstExpr::ExtractElement { vec, idx } => stack.extend([*vec, *idx]),
                ConstExpr::InsertElement { vec, elt, idx } => stack.extend([*vec, *elt, *idx]),
                ConstExpr::ShuffleVector { a, b, .. } => stack.extend([*a, *b]),
            },
            ConstKind::Struct(v) | ConstKind::Array(v) | ConstKind::Vector(v) => stack.extend(v.iter().copied()),
            _ => {}
        }
    }
    true
}

/// Whether a pointer constant points into a global whose contents never change.
fn constant_object(m: &Module, p: Value) -> bool {
    let base = match p {
        Value::Global(g) => Some(g),
        Value::Const(c) => match &m.constant(c).kind {
            ConstKind::Expr(ConstExpr::Gep { ptr: Value::Global(g), .. }) => Some(*g),
            ConstKind::Expr(ConstExpr::Cast {
                op: CastOp::BitCast | CastOp::AddrSpaceCast,
                value: Value::Global(g),
            }) => Some(*g),
            _ => None,
        },
        _ => None,
    };
    match base {
        Some(GlobalId::Var(i)) => {
            let g = &m.globals[i as usize];
            g.constant && !g.externally_initialized && g.initializer.is_some()
        }
        _ => false,
    }
}

/// Whether a value of type `ty` is exactly determined: each scalar is poison, wholly undef, or defined without
/// choice bits (`freeze` may keep its choices).
fn determined(m: &Module, ty: crate::ir::TypeId, v: &Val, freeze: bool) -> bool {
    use crate::ir::Type;
    let width = |t: crate::ir::TypeId| -> Option<u32> {
        match m.types.get(t) {
            Type::Int(w) => Some(*w),
            Type::Half | Type::BFloat => Some(16),
            Type::Float => Some(32),
            Type::Double => Some(64),
            Type::Ptr(_) => Some(64),
            _ => None,
        }
    };
    let scalar_ok = |t: crate::ir::TypeId, s: &Scalar| -> bool {
        let Some(w) = width(t) else { return false };
        let full = crate::interp::value::mask(w);
        s.poison || ((s.undef == 0 || s.undef & full == full) && (freeze || s.choice & full == 0))
    };
    match (m.types.get(ty), v) {
        (Type::Vector { elem, .. }, Val::V(l)) => l.iter().all(|s| scalar_ok(*elem, s)),
        (Type::Array { elem, .. }, Val::A(a)) => a.iter().all(|x| determined(m, *elem, x, freeze)),
        (Type::Struct { .. } | Type::LiteralStruct { .. }, Val::A(a)) => {
            let elems = m.types.struct_elems(ty).unwrap_or(&[]);
            a.len() == elems.len() && a.iter().zip(elems).all(|(x, e)| determined(m, *e, x, freeze))
        }
        (_, Val::S(s)) => scalar_ok(ty, s),
        (_, Val::Token) => true,
        _ => false,
    }
}

/// Structural simplifications of instruction `i`: the value it equals, if any.
fn simplify(m: &mut Module, f: usize, i: InstId) -> Option<Value> {
    let func = &m.functions[f];
    let inst = func.inst(i);
    match &inst.op {
        Op::Phi { incoming, .. } => {
            let mut v: Option<Value> = None;
            for (x, _) in incoming {
                if *x == Value::Inst(i) {
                    continue;
                }
                match v {
                    None => v = Some(*x),
                    Some(y) if y == *x => {}
                    Some(_) => return None,
                }
            }
            v
        }
        Op::Select { cond, t, f: fv, .. } => match const_int(m, *cond) {
            Some(c) if m.types.int_bits(m.constant(super::util::const_id(*cond)?).ty) == Some(1) => {
                Some(if c & 1 != 0 { *t } else { *fv })
            }
            _ => None,
        },
        Op::Cast { op: CastOp::BitCast | CastOp::AddrSpaceCast, value, .. } => {
            let vt = match value {
                Value::Inst(x) => func.inst(*x).ty,
                Value::Arg(a) => func.args[*a as usize].ty,
                Value::Const(c) => m.constant(*c).ty,
                _ => return None,
            };
            (vt == inst.ty).then_some(*value)
        }
        Op::Gep { ptr, indices, .. } => {
            let pt = match ptr {
                Value::Inst(x) => func.inst(*x).ty,
                Value::Arg(a) => func.args[*a as usize].ty,
                _ => return None,
            };
            let zero = indices.iter().all(|&x| const_int(m, x) == Some(0));
            (zero && pt == inst.ty).then_some(*ptr)
        }
        _ => None,
    }
}

/// Branches and switches on constants become unconditional; returns whether any did.
fn branches(m: &mut Module, f: usize) -> bool {
    let mut changed = false;
    for b in 0..m.functions[f].blocks.len() {
        let Some(t) = super::util::terminator(&m.functions[f], b) else { continue };
        let op = m.functions[f].inst(t).op.clone();
        let (keep, all): (BlockId, Vec<BlockId>) = match &op {
            Op::CondBr { cond, t: tb, f: fb } => {
                let Value::Const(c) = cond else { continue };
                let ConstKind::Int(w) = &m.constant(*c).kind else { continue };
                (if w[0] & 1 != 0 { *tb } else { *fb }, vec![*tb, *fb])
            }
            Op::Switch { cond, default, cases } => {
                let Some(v) = const_int(m, *cond) else { continue };
                let dest = cases.iter().find(|(c, _)| const_int(m, Value::Const(*c)) == Some(v)).map(|(_, d)| *d);
                (dest.unwrap_or(*default), op.successors())
            }
            _ => continue,
        };
        // Every edge but one to the kept block goes, with its phi operands.
        let mut kept_one = false;
        for s in all {
            if s == keep && !kept_one {
                kept_one = true;
                continue;
            }
            super::util::remove_phi_edge(&mut m.functions[f], s.0 as usize, b);
        }
        m.functions[f].insts[t.0 as usize].op = Op::Br { dest: keep };
        changed = true;
    }
    changed
}
