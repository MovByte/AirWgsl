//! Interpreter values as constants, and constants as interpreter values.
//!
//! A constant is LLVM's: each scalar is a number, `undef` or `poison` as a whole. An interpreter value can also be
//! partly undef (some bits) or carry choice bits; as a constant it becomes the value with zeros in its undef bits and
//! its chosen bits, which refines it (LangRef: undef may be replaced by any value, and a choice by any allowed one).

use crate::interp::memory::{Memory, RegionKind};
use crate::interp::value::mask;
use crate::interp::{Scalar, Val};
use crate::ir::{ConstExpr, ConstKind, Constant, GepFlags, GlobalId, Module, Type, TypeId, Value};

use super::util::{kind_const, sequence_const, struct_const};

/// The constant of type `ty` that refines `v`. Pointers into global regions of `mem` (the memory of a run of `m`)
/// become `getelementptr i8, ptr @g, i64 offset` expressions; other pointers with provenance have no constant form.
pub fn to_const(m: &mut Module, mem: &Memory, ty: TypeId, v: &Val) -> Result<Value, String> {
    match (m.types.get(ty).clone(), v) {
        (Type::Vector { elem, .. }, Val::V(lanes)) => {
            let mut out = Vec::with_capacity(lanes.len());
            for l in lanes {
                out.push(scalar_const(m, mem, elem, l)?);
            }
            Ok(sequence_const(m, ty, out))
        }
        (Type::Array { elem, .. }, Val::A(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(to_const(m, mem, elem, it)?);
            }
            Ok(sequence_const(m, ty, out))
        }
        (Type::Struct { .. } | Type::LiteralStruct { .. }, Val::A(items)) => {
            let elems = m.types.struct_elems(ty).ok_or("an opaque struct")?.to_vec();
            let mut out = Vec::with_capacity(items.len());
            for (k, it) in items.iter().enumerate() {
                out.push(to_const(m, mem, elems[k], it)?);
            }
            Ok(struct_const(m, ty, out))
        }
        (Type::Token, Val::Token) => Ok(kind_const(m, ty, ConstKind::TokenNone)),
        (_, Val::S(s)) => scalar_const(m, mem, ty, s),
        (t, v) => Err(format!("no constant of type {:?} for {:?}", t, v)),
    }
}

fn scalar_const(m: &mut Module, mem: &Memory, ty: TypeId, s: &Scalar) -> Result<Value, String> {
    if s.poison {
        return Ok(kind_const(m, ty, ConstKind::Poison));
    }
    let t = m.types.get(ty).clone();
    let width = match &t {
        Type::Int(w) => *w,
        Type::Half | Type::BFloat => 16,
        Type::Float => 32,
        Type::Double => 64,
        Type::Ptr(a) => {
            crate::ir::layout::DataLayout::parse(&m.datalayout).map(|dl| dl.pointer_spec(*a).bits).unwrap_or(64)
        }
        other => return Err(format!("no scalar constant of type {:?}", other)),
    };
    if s.undef & mask(width) == mask(width) {
        return Ok(kind_const(m, ty, ConstKind::Undef));
    }
    let bits = s.bits & !s.undef & mask(width);
    match t {
        Type::Int(w) => {
            let n = w.div_ceil(64).max(1) as usize;
            let words = (0..n).map(|i| if i < 2 { (bits >> (64 * i)) as u64 } else { 0 }).collect();
            Ok(kind_const(m, ty, ConstKind::Int(words)))
        }
        Type::Ptr(a) => {
            if s.prov == 0 {
                if bits == 0 {
                    return Ok(kind_const(m, ty, ConstKind::NullPtr));
                }
                let i64t = m.types.int(64);
                let addr = kind_const(m, i64t, ConstKind::Int(vec![bits as u64]));
                return Ok(kind_const(
                    m,
                    ty,
                    ConstKind::Expr(ConstExpr::Cast { op: crate::ir::CastOp::IntToPtr, value: addr }),
                ));
            }
            let r = mem.region(s.prov);
            let base = match r.kind {
                RegionKind::Global(i) => Value::Global(GlobalId::Var(i)),
                RegionKind::Function(i) => Value::Global(GlobalId::Function(i)),
                _ => return Err(format!("a pointer to {}, which has no constant form", r.label)),
            };
            if m.global(match base {
                Value::Global(g) => g,
                _ => unreachable!(),
            })
            .addrspace
                != a
            {
                return Err("a pointer constant in another address space than its object".into());
            }
            let off = (bits as u64).wrapping_sub(r.base);
            if off == 0 {
                return Ok(base);
            }
            let i8t = m.types.int(8);
            let i64t = m.types.int(64);
            let idx = kind_const(m, i64t, ConstKind::Int(vec![off]));
            Ok(kind_const(
                m,
                ty,
                ConstKind::Expr(ConstExpr::Gep {
                    source: i8t,
                    flags: GepFlags(0),
                    inrange: None,
                    ptr: base,
                    indices: vec![idx],
                }),
            ))
        }
        _ => Ok(kind_const(m, ty, ConstKind::Float(bits))),
    }
}

/// The value of a constant that refers to no global, or `None`.
pub fn const_val(m: &Module, v: Value) -> Option<Val> {
    let Value::Const(c) = v else { return None };
    let k: &Constant = m.constant(c);
    let ty = k.ty;
    let width = |t: TypeId| -> Option<u32> {
        match m.types.get(t) {
            Type::Int(w) => Some(*w),
            Type::Half | Type::BFloat => Some(16),
            Type::Float => Some(32),
            Type::Double => Some(64),
            Type::Ptr(a) => crate::ir::layout::DataLayout::parse(&m.datalayout).ok().map(|dl| dl.pointer_spec(*a).bits),
            _ => None,
        }
    };
    let fill = |leaf: &dyn Fn(TypeId) -> Option<Scalar>| -> Option<Val> { fill(m, ty, leaf) };
    Some(match &k.kind {
        ConstKind::Int(w) => {
            if w.len() > 2 {
                return None;
            }
            let mut x = 0u128;
            for (i, word) in w.iter().enumerate() {
                x |= (*word as u128) << (64 * i);
            }
            splat(m, ty, Scalar::int(x))
        }
        ConstKind::Float(b) => splat(m, ty, Scalar::int(*b)),
        ConstKind::NullPtr => splat(m, ty, Scalar::ptr(0, 0)),
        ConstKind::Zero => fill(&|_| Some(Scalar::int(0)))?,
        ConstKind::Undef => fill(&|t| width(t).map(Scalar::undef))?,
        ConstKind::Poison => fill(&|_| Some(Scalar::POISON))?,
        ConstKind::TokenNone => Val::Token,
        ConstKind::Struct(items) | ConstKind::Array(items) => {
            Val::A(items.iter().map(|&i| const_val(m, i)).collect::<Option<Vec<_>>>()?)
        }
        ConstKind::Vector(items) => {
            Val::V(items.iter().map(|&i| const_val(m, i).and_then(|v| v.scalar())).collect::<Option<Vec<_>>>()?)
        }
        ConstKind::DataArray(b) => Val::A(b.iter().map(|x| Val::S(Scalar::int(*x as u128))).collect()),
        ConstKind::DataVector(b) => Val::V(b.iter().map(|x| Scalar::int(*x as u128)).collect()),
        _ => return None,
    })
}

fn splat(m: &Module, ty: TypeId, s: Scalar) -> Val {
    match m.types.get(ty) {
        Type::Vector { len, .. } => Val::V(vec![s; *len as usize]),
        _ => Val::S(s),
    }
}

fn fill(m: &Module, ty: TypeId, leaf: &dyn Fn(TypeId) -> Option<Scalar>) -> Option<Val> {
    Some(match m.types.get(ty) {
        Type::Vector { len, elem, .. } => Val::V(vec![leaf(*elem)?; *len as usize]),
        Type::Array { len, elem } => {
            if *len > 1 << 20 {
                return None;
            }
            Val::A(vec![fill(m, *elem, leaf)?; *len as usize])
        }
        Type::Struct { .. } | Type::LiteralStruct { .. } => {
            let elems = m.types.struct_elems(ty)?;
            Val::A(elems.iter().map(|&e| fill(m, e, leaf)).collect::<Option<Vec<_>>>()?)
        }
        Type::Token => Val::Token,
        _ => Val::S(leaf(ty)?),
    })
}
