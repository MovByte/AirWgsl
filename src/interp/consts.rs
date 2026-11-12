//! Constants as values, and the metadata operands the interpreter reads.

use super::Error;
use super::exec::{Frame, Interp, R, TyRef, map_lanes, zip_lanes};
use super::ops::{self, Prim};
use super::value::{Scalar, Val};
use crate::ir::{ConstExpr, ConstId, ConstKind, MdId, Metadata, Module, Type, TypeId, Value};

/// A frame for evaluating module-level values (constants refer to no function).
pub(super) fn detached() -> Frame {
    Frame { f: usize::MAX, args: Vec::new(), vals: Vec::new(), allocas: Vec::new(), cur: None }
}

fn unsupported(what: impl Into<String>) -> Error {
    Error::Unsupported { what: what.into(), location: String::new() }
}

impl Interp<'_> {
    pub fn eval_const(&mut self, c: ConstId) -> R<Val> {
        if let Some(v) = self.consts.get(&c) {
            return Ok(v.clone());
        }
        let v = self.const_value(c)?;
        // Constants that refer to globals are cached too: a global's region never moves.
        self.consts.insert(c, v.clone());
        Ok(v)
    }

    fn operand(&mut self, v: Value) -> R<Val> {
        match v {
            Value::Const(c) => self.eval_const(c),
            Value::Global(g) => Ok(Val::S(self.global_ptr(g)?)),
            _ => Err(unsupported("a constant operand that is not a constant")),
        }
    }

    fn const_value(&mut self, c: ConstId) -> R<Val> {
        let m = self.m;
        let k = m.constant(c);
        let ty = k.ty;
        Ok(match &k.kind {
            ConstKind::Int(words) => {
                if words.len() > 2 {
                    return Err(unsupported(format!("a {}-word integer constant", words.len())));
                }
                let mut v = 0u128;
                for (i, w) in words.iter().enumerate() {
                    v |= (*w as u128) << (64 * i);
                }
                splat_or_scalar(m, ty, Scalar::int(v))
            }
            ConstKind::Float(bits) => splat_or_scalar(m, ty, Scalar::int(*bits)),
            ConstKind::NullPtr => splat_or_scalar(m, ty, Scalar::ptr(0, 0)),
            ConstKind::TokenNone => Val::Token,
            ConstKind::Zero => fill(m, ty, &|_| Some(Scalar::int(0)))?,
            ConstKind::Undef => fill(m, ty, &|t| self.prim(TyRef::Id(t)).map(|p| Scalar::undef(p.width())))?,
            ConstKind::Poison => fill(m, ty, &|_| Some(Scalar::POISON))?,
            ConstKind::Struct(items) | ConstKind::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for v in items {
                    out.push(self.operand(*v)?);
                }
                Val::A(out)
            }
            ConstKind::Vector(items) => {
                let mut out = Vec::with_capacity(items.len());
                for v in items {
                    match self.operand(*v)? {
                        Val::S(s) => out.push(s),
                        _ => return Err(unsupported("a vector constant of non-scalars")),
                    }
                }
                Val::V(out)
            }
            ConstKind::DataArray(bits) => Val::A(bits.iter().map(|b| Val::S(Scalar::int(*b as u128))).collect()),
            ConstKind::DataVector(bits) => Val::V(bits.iter().map(|b| Scalar::int(*b as u128)).collect()),
            ConstKind::Expr(e) => self.const_expr(ty, e)?,
            ConstKind::BlockAddress { .. } => return Err(unsupported("blockaddress")),
            ConstKind::DsoLocalEquivalent(g) | ConstKind::NoCfi(g) => Val::S(self.global_ptr(*g)?),
            ConstKind::PtrAuth { .. } => return Err(unsupported("ptrauth constants")),
        })
    }

    fn const_expr(&mut self, ty: TypeId, e: &ConstExpr) -> R<Val> {
        let fr = detached();
        match e {
            ConstExpr::Cast { op, value } => {
                let st = self.type_of(&fr, *value);
                let v = self.operand(*value)?;
                self.cast_value(&fr, *op, 0, crate::ir::FastMath(0), st, ty, v)
            }
            ConstExpr::Binary { op, lhs, rhs, flags } => {
                let p = self.prim_of(&fr, TyRef::Id(ty))?;
                let a = self.operand(*lhs)?;
                let b = self.operand(*rhs)?;
                let mut err = None;
                let r = zip_lanes(&a, &b, |x, y| {
                    let r = match p {
                        Prim::Float(fmt) => Ok(ops::float_binop(*op, crate::ir::FastMath(0), fmt, x, y)),
                        _ => ops::int_binop(*op, *flags, p.width(), x, y),
                    };
                    r.unwrap_or_else(|u| {
                        err.get_or_insert(u);
                        Scalar::POISON
                    })
                });
                if let Some(u) = err {
                    return Err(Error::Ub { reason: u.0, location: String::new() });
                }
                Ok(r)
            }
            ConstExpr::Gep { source, flags, ptr, indices, .. } => {
                let base = self.operand(*ptr)?;
                let mut idx = Vec::new();
                for i in indices {
                    let w = self.prim_of(&fr, self.type_of(&fr, *i))?.width();
                    idx.push((self.operand(*i)?, w));
                }
                let asp = match self.prim_of(&fr, self.type_of(&fr, *ptr))? {
                    Prim::Ptr(a, _) => a,
                    _ => return Err(unsupported("getelementptr on a non-pointer")),
                };
                self.gep(&fr, *source, asp, &base, &idx, *flags)
            }
            ConstExpr::ExtractElement { vec, idx } => {
                let v = self.operand(*vec)?.lanes();
                let i = self.operand(*idx)?.scalar().unwrap_or(Scalar::POISON);
                Ok(Val::S(if i.is_defined() && (i.bits as usize) < v.len() {
                    v[i.bits as usize]
                } else {
                    Scalar::POISON
                }))
            }
            ConstExpr::InsertElement { vec, elt, idx } => {
                let mut v = self.operand(*vec)?.lanes();
                let e = self.operand(*elt)?.scalar().unwrap_or(Scalar::POISON);
                let i = self.operand(*idx)?.scalar().unwrap_or(Scalar::POISON);
                if i.is_defined() && (i.bits as usize) < v.len() {
                    v[i.bits as usize] = e;
                } else {
                    v = vec![Scalar::POISON; v.len()];
                }
                Ok(Val::V(v))
            }
            ConstExpr::ShuffleVector { a, b, mask } => {
                let x = self.operand(*a)?.lanes();
                let y = self.operand(*b)?.lanes();
                let all: Vec<Scalar> = x.into_iter().chain(y).collect();
                Ok(Val::V(
                    mask.iter()
                        .map(|&k| if k < 0 || k as usize >= all.len() { Scalar::POISON } else { all[k as usize] })
                        .collect(),
                ))
            }
        }
    }

    /// A cast of a whole value (scalar or vector) from type `st` to `dt`.
    #[allow(clippy::too_many_arguments)]
    pub fn cast_value(
        &mut self,
        fr: &Frame,
        op: crate::ir::CastOp,
        flags: u8,
        fmf: crate::ir::FastMath,
        st: TyRef,
        dt: TypeId,
        v: Val,
    ) -> R<Val> {
        let sp = self.prim_of(fr, st)?;
        let dp = self.prim_of(fr, TyRef::Id(dt))?;
        let sv = self.is_vector(st);
        let dv = self.m.types.is_vector(dt);
        if op == crate::ir::CastOp::BitCast && (sv || dv) && (sv != dv || sp.width() != dp.width()) {
            let lanes = v.lanes();
            let total = sp.width() * lanes.len() as u32;
            let out = super::memory::bitcast_lanes(&lanes, sp.width(), dp.width(), total / dp.width());
            return Ok(if dv { Val::V(out) } else { Val::S(out[0]) });
        }
        Ok(map_lanes(&v, |a| ops::cast(op, flags, fmf, sp, dp, a)))
    }
}

/// A scalar constant of a scalar type, or its splat for a vector type (`splat (i32 1)` is an integer constant of
/// vector type).
fn splat_or_scalar(m: &Module, ty: TypeId, s: Scalar) -> Val {
    match m.types.get(ty) {
        Type::Vector { len, .. } => Val::V(vec![s; *len as usize]),
        _ => Val::S(s),
    }
}

/// A value of type `ty` with every scalar leaf `leaf(scalar type)`.
fn fill(m: &Module, ty: TypeId, leaf: &dyn Fn(TypeId) -> Option<Scalar>) -> R<Val> {
    let one = |t: TypeId| leaf(t).ok_or_else(|| unsupported(format!("a constant of type {:?}", m.types.get(t))));
    Ok(match m.types.get(ty) {
        Type::Vector { len, elem, .. } => Val::V(vec![one(*elem)?; *len as usize]),
        Type::Array { len, elem } => {
            if *len > 1 << 24 {
                return Err(unsupported(format!("an array constant of {} elements", len)));
            }
            let one = fill(m, *elem, leaf)?;
            Val::A(vec![one; *len as usize])
        }
        Type::Struct { .. } | Type::LiteralStruct { .. } => {
            let elems = m.types.struct_elems(ty).ok_or_else(|| unsupported("a constant of an opaque struct"))?.to_vec();
            let mut out = Vec::new();
            for e in elems {
                out.push(fill(m, e, leaf)?);
            }
            Val::A(out)
        }
        Type::Token => Val::Token,
        Type::TargetExt { .. } => return Err(unsupported("a constant of a target extension type")),
        _ => Val::S(one(ty)?),
    })
}

/// The integer of a `!{iN v}` metadata node (or a value wrapper).
pub fn md_int(m: &Module, md: MdId) -> Option<u64> {
    let inner = match m.md(md) {
        Metadata::Tuple { ops, .. } => ops.first().copied().flatten()?,
        _ => md,
    };
    match m.md(inner) {
        Metadata::Value { value: Value::Const(c), .. } => match &m.constant(*c).kind {
            ConstKind::Int(w) => w.first().copied(),
            _ => None,
        },
        _ => None,
    }
}

/// The ranges of `!range` metadata: pairs `[lo, hi)`.
pub fn md_ranges(m: &Module, md: MdId) -> Vec<(u128, u128)> {
    let Metadata::Tuple { ops, .. } = m.md(md) else { return Vec::new() };
    let ints: Vec<u128> = ops
        .iter()
        .filter_map(|o| {
            let o = (*o)?;
            match m.md(o) {
                Metadata::Value { value: Value::Const(c), .. } => match &m.constant(*c).kind {
                    ConstKind::Int(w) => {
                        let mut v = 0u128;
                        for (i, x) in w.iter().enumerate().take(2) {
                            v |= (*x as u128) << (64 * i);
                        }
                        Some(v)
                    }
                    _ => None,
                },
                _ => None,
            }
        })
        .collect();
    ints.chunks(2).filter(|p| p.len() == 2).map(|p| (p[0], p[1])).collect()
}
