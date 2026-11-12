//! Calls of declarations: LLVM intrinsics (LangRef "Intrinsic Functions"), the AIR functions whose results the
//! Metal Shading Language Specification defines exactly, and uninterpreted functions for the rest.
//!
//! AIR integer functions follow MSL Table 6.2 with the edge behavior the numerics package measured on Metal and
//! implements in WGSL (`src/num/wgsl/int.wgsl`); where MSL leaves a result unspecified (`clamp` with `minval >
//! maxval`, `extract_bits` past the width), the interpreter computes the value that library returns and marks it a
//! choice. Conversions follow MSL 2.23 and 8.6 as `src/num` lowers them: to floating point rounds to nearest even, to
//! an integer truncates, NaN converts to 0 (to `true` for `bool`); an out-of-range conversion to an integer, which
//! MSL leaves unspecified, saturates and is marked a choice.

use super::exec::{Frame, Interp, R, TyRef, map_lanes, select, zip_lanes};
use super::memory::{Byte, RegionKind};
use super::ops::{self, Prim};
use super::value::{Scalar, Val, mask};
use super::{Externals, float};
use crate::ir::{AttrKind, Attribute, FastMath, InstId, Op, Type, TypeId, Value};
use float::Format;

/// Whether a call was evaluated here.
enum Done {
    Value(Option<Val>),
    NotHere,
}

fn s(v: &Val) -> Scalar {
    v.scalar().unwrap_or(Scalar::POISON)
}

impl Interp<'_> {
    pub fn intrinsic(&mut self, fr: &mut Frame, id: InstId, target: usize, args: Vec<Val>) -> R<Option<Val>> {
        let m = self.m;
        let name = m.functions[target].common.name.as_str();
        let inst = m.functions[fr.f].inst(id);
        let fmf = match &inst.op {
            Op::Call { fmf, .. } => *fmf,
            _ => FastMath(0),
        };
        let done = if let Some(rest) = name.strip_prefix("llvm.") {
            self.llvm(fr, rest, inst.ty, fmf, &args)?
        } else if let Some(rest) = name.strip_prefix("air.") {
            self.air(fr, id, rest, inst.ty, &args)?
        } else {
            Done::NotHere
        };
        match done {
            Done::Value(v) => Ok(v),
            Done::NotHere => self.external(fr, id, target, args),
        }
    }

    fn ret_prim(&self, fr: &Frame, ty: TypeId) -> R<Prim> {
        self.prim_of(fr, TyRef::Id(ty))
    }

    // LLVM intrinsics -----------------------------------------------------------------------------------------------

    fn llvm(&mut self, fr: &mut Frame, rest: &str, ty: TypeId, fmf: FastMath, a: &[Val]) -> R<Done> {
        let base = rest.split('.').next().unwrap_or("");
        let two = rest.split('.').take(2).collect::<Vec<_>>().join(".");
        let v = match base {
            "memcpy" | "memmove" => {
                self.memcpy(fr, &s(&a[0]), &s(&a[1]), &s(&a[2]), base == "memcpy")?;
                None
            }
            "memset" => {
                self.memset(fr, &s(&a[0]), &s(&a[1]), &s(&a[2]))?;
                None
            }
            "lifetime" => {
                let p = s(&a[a.len() - 1]);
                self.lifetime(fr, &p, rest.starts_with("lifetime.start"))?;
                None
            }
            "assume" => {
                let c = s(&a[0]);
                if !c.is_defined() || c.bits & 1 == 0 {
                    return Err(self.ub(fr, "llvm.assume of a condition that is not true"));
                }
                None
            }
            "dbg" | "donothing" | "sideeffect" => None,
            "experimental" if two == "experimental.noalias" => None,
            "expect" | "annotation" | "launder" | "strip" => Some(a[0].clone()),
            "var" if two == "var.annotation" => None,
            "ptr" if two == "ptr.annotation" => Some(a[0].clone()),
            "abs" => {
                let w = self.ret_prim(fr, ty)?.width();
                let flag = s(&a[1]);
                Some(map_lanes(&a[0], |x| {
                    if x.poison {
                        return Scalar::POISON;
                    }
                    if x.undef != 0 {
                        return Scalar::undef(w);
                    }
                    let min = 1u128 << (w - 1);
                    if x.bits == min {
                        return if flag.bits & 1 != 0 { Scalar::POISON } else { keep_choice(*x, Scalar::int(min), w) };
                    }
                    let v = if x.bits & min != 0 { x.bits.wrapping_neg() & mask(w) } else { x.bits };
                    keep_choice(*x, Scalar::int(v), w)
                }))
            }
            "smax" | "smin" | "umax" | "umin" => {
                let w = self.ret_prim(fr, ty)?.width();
                let pred = match base {
                    "smax" => crate::ir::IntPredicate::Sgt,
                    "smin" => crate::ir::IntPredicate::Slt,
                    "umax" => crate::ir::IntPredicate::Ugt,
                    _ => crate::ir::IntPredicate::Ult,
                };
                Some(zip_lanes(&a[0], &a[1], |x, y| {
                    let c = ops::icmp(pred, false, w, x, y);
                    s(&select(&Val::S(c), &Val::S(*x), &Val::S(*y)))
                }))
            }
            "uadd" | "usub" | "sadd" | "ssub" if rest.contains(".sat.") => {
                let w = self.ret_prim(fr, ty)?.width();
                Some(zip_lanes(&a[0], &a[1], |x, y| defined2(x, y, w, |p, q| sat_addsub(base, p, q, w))))
            }
            "ctpop" | "bswap" | "bitreverse" => {
                let w = self.ret_prim(fr, ty)?.width();
                Some(map_lanes(&a[0], |x| {
                    defined1(x, w, |v| match base {
                        "ctpop" => v.count_ones() as u128,
                        "bswap" => v.swap_bytes() >> (128 - w),
                        _ => v.reverse_bits() >> (128 - w),
                    })
                }))
            }
            "ctlz" | "cttz" => {
                let w = self.ret_prim(fr, ty)?.width();
                let poison_zero = s(&a[1]).bits & 1 != 0;
                Some(map_lanes(&a[0], |x| {
                    if x.is_defined() && x.bits == 0 && poison_zero {
                        return Scalar::POISON;
                    }
                    defined1(x, w, |v| {
                        if base == "ctlz" {
                            (v.leading_zeros() - (128 - w)) as u128
                        } else {
                            v.trailing_zeros().min(w) as u128
                        }
                    })
                }))
            }
            "fshl" | "fshr" => {
                let w = self.ret_prim(fr, ty)?.width();
                let (x, y, z) = (a[0].lanes(), a[1].lanes(), a[2].lanes());
                let lanes: Vec<Scalar> = (0..x.len())
                    .map(|k| {
                        let (p, q, r) = (x[k], y[k], z[k]);
                        if p.poison || q.poison || r.poison {
                            return Scalar::POISON;
                        }
                        if p.undef | q.undef | r.undef != 0 {
                            return Scalar::undef(w);
                        }
                        let sh = (r.bits % w as u128) as u32;
                        let v = if sh == 0 {
                            if base == "fshl" { p.bits } else { q.bits }
                        } else if base == "fshl" {
                            ((p.bits << sh) | (q.bits >> (w - sh))) & mask(w)
                        } else {
                            ((q.bits >> sh) | (p.bits << (w - sh))) & mask(w)
                        };
                        let mut o = Scalar::int(v);
                        if p.choice | q.choice | r.choice != 0 {
                            o.choice = mask(w);
                        }
                        o
                    })
                    .collect();
                Some(if self.m.types.is_vector(ty) { Val::V(lanes) } else { Val::S(lanes[0]) })
            }
            "fabs" | "copysign" | "sqrt" | "fma" | "fmuladd" | "minnum" | "maxnum" | "minimum" | "maximum"
            | "minimumnum" | "maximumnum" | "floor" | "ceil" | "trunc" | "rint" | "nearbyint" | "round"
            | "roundeven" => {
                let Prim::Float(f) = self.ret_prim(fr, ty)? else {
                    return Err(self.unsupported(fr, "a float intrinsic on integers"));
                };
                Some(self.llvm_float(base, f, fmf, a))
            }
            _ => return Ok(Done::NotHere),
        };
        Ok(Done::Value(v))
    }

    fn llvm_float(&self, base: &str, f: Format, fmf: FastMath, a: &[Val]) -> Val {
        let lanes: Vec<Vec<Scalar>> = a.iter().map(Val::lanes).collect();
        let n = lanes[0].len();
        let out: Vec<Scalar> = (0..n)
            .map(|k| {
                let ops: Vec<Scalar> = lanes.iter().map(|l| if l.len() == 1 { l[0] } else { l[k] }).collect();
                match base {
                    "fabs" => bitwise_float(&ops[0], |b| b & !(f.sign_bit() as u128)),
                    "copysign" => {
                        let (x, y) = (ops[0], ops[1]);
                        if x.poison || y.poison {
                            return Scalar::POISON;
                        }
                        let sb = f.sign_bit() as u128;
                        Scalar {
                            bits: (x.bits & !sb) | (y.bits & sb),
                            undef: (x.undef & !sb) | (y.undef & sb),
                            choice: (x.choice & !sb) | (y.choice & sb),
                            prov: 0,
                            poison: false,
                        }
                    }
                    "sqrt" => ops::float_math(f, fmf, &ops, &|v| float::sqrt(f, v[0])),
                    "fma" => ops::float_math(f, fmf, &ops, &|v| float::fma(f, v[0], v[1], v[2])),
                    "fmuladd" => {
                        // LangRef: fused or not, as the implementation likes; the interpreter fuses and marks the bits
                        // in which the unfused result differs.
                        let fused = ops::float_math(f, fmf, &ops, &|v| float::fma(f, v[0], v[1], v[2]));
                        let unfused = ops::float_math(f, fmf, &ops, &|v| {
                            let p = float::binary(f, float::Arith::Mul, v[0], v[1]);
                            float::binary(f, float::Arith::Add, p, v[2])
                        });
                        if fused.poison || unfused.poison {
                            return Scalar::POISON;
                        }
                        let mut r = fused;
                        r.choice |= (fused.bits ^ unfused.bits) | unfused.choice;
                        r.choice &= !r.undef;
                        r
                    }
                    "minnum" | "maxnum" | "minimum" | "maximum" | "minimumnum" | "maximumnum" => {
                        minmax(base, f, fmf, &ops[0], &ops[1])
                    }
                    _ => {
                        let mode = match base {
                            "floor" => float::Integral::Down,
                            "ceil" => float::Integral::Up,
                            "trunc" => float::Integral::Zero,
                            "round" => float::Integral::TiesAway,
                            _ => float::Integral::TiesEven,
                        };
                        ops::float_math(f, fmf, &ops, &|v| float::round_integral(f, v[0], mode))
                    }
                }
            })
            .collect();
        if matches!(a[0], Val::V(_)) { Val::V(out) } else { Val::S(out[0]) }
    }

    fn memcpy(&mut self, fr: &Frame, dst: &Scalar, src: &Scalar, len: &Scalar, no_overlap: bool) -> R<()> {
        if !len.is_defined() {
            return Err(self.ub(fr, "a memory transfer of an undef or poison length"));
        }
        let n = len.bits as u64;
        if n == 0 {
            return Ok(());
        }
        let bytes = self.mem.read(src, n).map_err(|e| self.ub(fr, e))?;
        if no_overlap {
            let (a, b) = (dst.bits as u64, src.bits as u64);
            if a < b.wrapping_add(n) && b < a.wrapping_add(n) && dst.prov == src.prov {
                return Err(self.ub(fr, "llvm.memcpy of overlapping ranges"));
            }
        }
        self.mem.write(dst, &bytes).map_err(|e| self.ub(fr, e))
    }

    fn memset(&mut self, fr: &Frame, dst: &Scalar, val: &Scalar, len: &Scalar) -> R<()> {
        if !len.is_defined() {
            return Err(self.ub(fr, "llvm.memset of an undef or poison length"));
        }
        let n = len.bits as usize;
        if n == 0 {
            return Ok(());
        }
        let b = if val.poison {
            Byte::POISON
        } else {
            Byte {
                v: val.bits as u8,
                undef: val.undef as u8,
                choice: val.choice as u8,
                poison: false,
                frag: 0,
                prov: 0,
            }
        };
        self.mem.write(dst, &vec![b; n]).map_err(|e| self.ub(fr, e))
    }

    /// `llvm.lifetime.start`/`end` (LangRef 21): on the first byte of a stack object, make it alive (uninitialized)
    /// or dead; on anything else, fill the object with poison.
    fn lifetime(&mut self, fr: &Frame, p: &Scalar, start: bool) -> R<()> {
        if !p.is_defined() {
            return Err(self.ub(fr, "a lifetime marker on a pointer that is not well defined"));
        }
        let id = if p.prov != 0 { Some(p.prov) } else { self.mem.region_at(p.bits as u64) };
        let Some(id) = id else { return Err(self.ub(fr, "a lifetime marker on a pointer to no object")) };
        let r = self.mem.region_mut(id);
        if r.kind == RegionKind::Alloca && p.bits as u64 == r.base {
            if start {
                r.live = true;
                r.bytes.iter_mut().for_each(|b| *b = Byte::UNDEF);
            } else {
                r.live = false;
            }
        } else {
            r.bytes.iter_mut().for_each(|b| *b = Byte::POISON);
        }
        Ok(())
    }

    // AIR -----------------------------------------------------------------------------------------------------------

    fn air(&mut self, fr: &mut Frame, id: InstId, rest: &str, ty: TypeId, a: &[Val]) -> R<Done> {
        let parts: Vec<&str> = rest.split('.').collect();
        let head = parts[0];
        let v = match head {
            "convert" => {
                if parts.len() != 5 {
                    return Err(self.unsupported(fr, "an air.convert with a rounding mode or saturation suffix"));
                }
                let dk = parts[1].chars().next().unwrap_or('?');
                let sk = parts[3].chars().next().unwrap_or('?');
                let st = self.type_of(fr, operand(self, fr, id, 0));
                let sp = self.prim_of(fr, st)?;
                let dp = self.ret_prim(fr, ty)?;
                let kind_ok = |k: char, p: Prim| match p {
                    Prim::Float(_) => k == 'f',
                    Prim::Int(_) => k == 's' || k == 'u',
                    Prim::Ptr(..) => false,
                };
                if !kind_ok(dk, dp) || !kind_ok(sk, sp) {
                    return Err(self.unsupported(fr, "an air.convert whose kinds do not match its types"));
                }
                if sk == 's' && sp == Prim::Int(1) {
                    // The front end converts bool as unsigned; a signed i1 source has no definition to follow.
                    return Err(self.unsupported(fr, "an air.convert from a signed i1"));
                }
                Some(map_lanes(&a[0], |x| convert(dk, dp, sk, sp, x)))
            }
            "is_function_constant_defined" => {
                let p = s(&a[0]);
                let Some(defined) = &self.opts.defined_constants else {
                    return Err(
                        self.unsupported(fr, "air.is_function_constant_defined without function constant values")
                    );
                };
                let name = match (p.prov, p.is_defined()) {
                    (r, true) if r != 0 => match self.mem.region(r).kind {
                        RegionKind::Global(g) => self.m.globals[g as usize].common.name.clone(),
                        _ => return Err(self.unsupported(fr, "air.is_function_constant_defined of a non-global")),
                    },
                    _ => return Err(self.unsupported(fr, "air.is_function_constant_defined of an unknown pointer")),
                };
                Some(Val::S(Scalar::bool(defined.contains(&name))))
            }
            "abs" | "abs_diff" | "add_sat" | "sub_sat" | "clamp" | "hadd" | "rhadd" | "mad_hi" | "mul_hi"
            | "mad_sat" | "max" | "min" | "max3" | "min3" | "median3" | "mul24" | "mad24" | "extract_bits"
            | "insert_bits" | "clz" | "ctz" | "popcount" | "reverse_bits" | "rotate" => {
                let Prim::Int(w) = self.ret_prim(fr, ty)? else { return Ok(Done::NotHere) };
                let signed = parts.get(1) == Some(&"s");
                if matches!(head, "clz" | "ctz") && s(&a[1]).bits & 1 != 0 {
                    return Err(self.unsupported(fr, format!("air.{} with its i1 operand true", head)));
                }
                Some(self.air_int(fr, head, signed, w, a)?)
            }
            "atomic" => return self.air_atomic(fr, id, &parts, ty, a),
            // MSL 6.5 fabs: the absolute value, a sign-bit operation (the call's fast-math flags apply).
            "fabs" => {
                let Prim::Float(f) = self.ret_prim(fr, ty)? else { return Ok(Done::NotHere) };
                let fmf = match &self.m.functions[fr.f].inst(id).op {
                    Op::Call { fmf, .. } => *fmf,
                    _ => FastMath(0),
                };
                Some(map_lanes(&a[0], |x| {
                    let r = bitwise_float(x, |b| b & !(f.sign_bit() as u128));
                    if r.poison || r.undef != 0 {
                        return r;
                    }
                    let (v, o) = (r.bits as u64, x.bits as u64);
                    let nan = fmf.0 & FastMath::NNAN != 0 && (f.is_nan(v) || f.is_nan(o));
                    let inf = fmf.0 & FastMath::NINF != 0 && (f.is_inf(v) || f.is_inf(o));
                    if nan || inf { Scalar::POISON } else { r }
                }))
            }
            // The predicate of an argument's [[function_constant]]: the identity (see passes::specialize for the
            // measurements that pin down what Metal makes of the stored value).
            "normalize_function_constant_predicate" => Some(a[0].clone()),
            _ => return Ok(Done::NotHere),
        };
        Ok(Done::Value(v))
    }

    fn air_int(&self, fr: &Frame, f: &str, signed: bool, w: u32, a: &[Val]) -> R<Val> {
        let lanes: Vec<Vec<Scalar>> = a.iter().map(Val::lanes).collect();
        let n = lanes[0].len();
        let m = mask(w);
        let sx = |v: u128| -> i128 { sext(v, w) };
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            // Scalar operands (clz's flag, extract_bits' offset and count) apply to every lane.
            let ops: Vec<Scalar> = lanes.iter().map(|l| if l.len() == 1 { l[0] } else { l[k] }).collect();
            let data = match f {
                "clz" | "ctz" | "extract_bits" => 1,
                "insert_bits" => 2,
                _ => ops.len(),
            };
            if ops.iter().any(|o| o.poison) {
                out.push(Scalar::POISON);
                continue;
            }
            if ops.iter().any(|o| o.undef != 0) {
                out.push(Scalar::undef(w));
                continue;
            }
            let v: Vec<u128> = ops.iter().take(data).map(|o| o.bits & m).collect();
            let mut choice = ops.iter().any(|o| o.choice != 0);
            let lim_s = |x: i128| -> u128 {
                let hi = (1i128 << (w - 1)) - 1;
                let lo = -(1i128 << (w - 1));
                (x.clamp(lo, hi) as u128) & m
            };
            let r: u128 = match f {
                "abs" => {
                    if signed {
                        (sx(v[0]).unsigned_abs()) & m
                    } else {
                        v[0]
                    }
                }
                "abs_diff" => {
                    if signed {
                        (sx(v[0]) - sx(v[1])).unsigned_abs() & m
                    } else {
                        v[0].abs_diff(v[1])
                    }
                }
                "add_sat" => {
                    if signed {
                        lim_s(sx(v[0]) + sx(v[1]))
                    } else {
                        (v[0] + v[1]).min(m)
                    }
                }
                "sub_sat" => {
                    if signed {
                        lim_s(sx(v[0]) - sx(v[1]))
                    } else {
                        v[0].saturating_sub(v[1])
                    }
                }
                "hadd" | "rhadd" => {
                    let r = if f == "rhadd" { 1 } else { 0 };
                    if signed { ((sx(v[0]) + sx(v[1]) + r) >> 1) as u128 & m } else { (v[0] + v[1] + r as u128) >> 1 }
                }
                "mul_hi" | "mad_hi" => {
                    let hi = if signed {
                        let p = sx(v[0]).wrapping_mul(sx(v[1]));
                        mul_hi_signed(sx(v[0]), sx(v[1]), w).unwrap_or((p >> w) as u128 & m)
                    } else {
                        mul_hi_unsigned(v[0], v[1], w)
                    };
                    if f == "mad_hi" { hi.wrapping_add(v[2]) & m } else { hi }
                }
                "mad_sat" => {
                    if signed {
                        mad_sat_signed(sx(v[0]), sx(v[1]), sx(v[2]), w)
                    } else {
                        mad_sat_unsigned(v[0], v[1], v[2], w)
                    }
                }
                "max" | "min" => {
                    let lt = if signed { sx(v[0]) < sx(v[1]) } else { v[0] < v[1] };
                    if (f == "max") == lt { v[1] } else { v[0] }
                }
                "max3" | "min3" | "median3" | "clamp" => {
                    let key = |x: u128| -> i128 { if signed { sx(x) } else { x as i128 } };
                    let mx = |x: u128, y: u128| if key(x) < key(y) { y } else { x };
                    let mn = |x: u128, y: u128| if key(y) < key(x) { y } else { x };
                    match f {
                        "max3" => mx(v[0], mx(v[1], v[2])),
                        "min3" => mn(v[0], mn(v[1], v[2])),
                        "median3" => mx(mn(v[0], v[1]), mn(mx(v[0], v[1]), v[2])),
                        _ => {
                            // MSL Table 6.2: min(max(x, minval), maxval); undefined if minval > maxval.
                            if key(v[1]) > key(v[2]) {
                                choice = true;
                            }
                            mn(mx(v[0], v[1]), v[2])
                        }
                    }
                }
                "mul24" | "mad24" => {
                    if w != 32 {
                        return Err(self.unsupported(fr, "mul24 or mad24 on a type other than 32-bit"));
                    }
                    let p = if signed {
                        (sext(v[0] & 0xff_ffff, 24) * sext(v[1] & 0xff_ffff, 24)) as u128 & m
                    } else {
                        ((v[0] & 0xff_ffff) * (v[1] & 0xff_ffff)) & m
                    };
                    if f == "mad24" { p.wrapping_add(v[2]) & m } else { p }
                }
                "clz" => (v[0].leading_zeros() - (128 - w)) as u128,
                "ctz" => v[0].trailing_zeros().min(w) as u128,
                "popcount" => v[0].count_ones() as u128,
                "reverse_bits" => v[0].reverse_bits() >> (128 - w),
                "rotate" => {
                    let r = (v[1] % w as u128) as u32;
                    if r == 0 { v[0] } else { ((v[0] << r) | (v[0] >> (w - r))) & m }
                }
                "extract_bits" | "insert_bits" => {
                    // Offset and count are 32-bit operands.
                    let (off, cnt) =
                        if f == "extract_bits" { (ops[1].bits, ops[2].bits) } else { (ops[2].bits, ops[3].bits) };
                    let (off, cnt) = (off & 0xffff_ffff, cnt & 0xffff_ffff);
                    // MSL Table 6.2 defines the result for offset + bits within the width; beyond it, the clamped
                    // WGSL form is the value computed, as a choice.
                    if off + cnt > w as u128 {
                        choice = true;
                    }
                    let o = off.min(w as u128) as u32;
                    let c = cnt.min((w - o) as u128) as u32;
                    if f == "extract_bits" {
                        if c == 0 {
                            0
                        } else {
                            let field = (v[0] >> o) & mask(c);
                            if signed { (sext(field, c) as u128) & m } else { field }
                        }
                    } else if c == 0 {
                        v[0]
                    } else {
                        let fm = mask(c) << o;
                        (v[0] & !fm) | ((v[1] << o) & fm)
                    }
                }
                _ => return Err(self.unsupported(fr, format!("air.{}", f))),
            };
            let mut o = Scalar::int(r & m);
            if choice {
                o.choice = m;
            }
            out.push(o);
        }
        Ok(if matches!(a[0], Val::V(_)) { Val::V(out) } else { Val::S(out[0]) })
    }

    /// `air.atomic.{global,local}.<op>[.s|.u].<type>(ptr, [value,] order, scope, volatile)`: one invocation's view,
    /// where every atomic is the plain read-modify-write. The compare-exchange forms are not evaluated.
    fn air_atomic(&mut self, fr: &Frame, id: InstId, parts: &[&str], ty: TypeId, a: &[Val]) -> R<Done> {
        if parts.len() < 3 || !matches!(parts[1], "global" | "local") {
            return Ok(Done::NotHere);
        }
        let op = parts[2];
        let p = s(&a[0]);
        match op {
            "load" => {
                let size = self.layout().store_size(ty).unwrap_or(1);
                let v = self.load(fr, ty, &p, size)?;
                Ok(Done::Value(Some(v)))
            }
            "store" => {
                let TyRef::Id(t) = self.type_of(fr, operand(self, fr, id, 1)) else {
                    return Err(self.unsupported(fr, "air.atomic store of a pointer"));
                };
                let size = self.layout().store_size(t).unwrap_or(1);
                self.store(fr, t, &a[1], &p, size)?;
                Ok(Done::Value(None))
            }
            "xchg" | "add" | "sub" | "and" | "or" | "xor" | "min" | "max" => {
                let signed = parts.get(3) == Some(&"s");
                let rmw = match (op, signed) {
                    ("xchg", _) => crate::ir::RmwOp::Xchg,
                    ("add", _) => crate::ir::RmwOp::Add,
                    ("sub", _) => crate::ir::RmwOp::Sub,
                    ("and", _) => crate::ir::RmwOp::And,
                    ("or", _) => crate::ir::RmwOp::Or,
                    ("xor", _) => crate::ir::RmwOp::Xor,
                    ("min", true) => crate::ir::RmwOp::Min,
                    ("min", false) => crate::ir::RmwOp::UMin,
                    ("max", true) => crate::ir::RmwOp::Max,
                    _ => crate::ir::RmwOp::UMax,
                };
                let prim = self.prim_of(fr, TyRef::Id(ty))?;
                if matches!(prim, Prim::Float(_)) && rmw != crate::ir::RmwOp::Xchg {
                    return Ok(Done::NotHere);
                }
                let size = self.layout().store_size(ty).unwrap_or(1);
                let old = self.load(fr, ty, &p, size)?;
                let new = self.rmw(fr, rmw, prim, &old, &a[1])?;
                self.store(fr, ty, &new, &p, size)?;
                Ok(Done::Value(Some(old)))
            }
            _ => Ok(Done::NotHere),
        }
    }

    // Uninterpreted functions ----------------------------------------------------------------------------------------

    fn external(&mut self, fr: &mut Frame, id: InstId, target: usize, args: Vec<Val>) -> R<Option<Val>> {
        let m = self.m;
        let decl = &m.functions[target];
        let name = decl.common.name.clone();
        if self.opts.externals == Externals::Refuse {
            return Err(self.unsupported(fr, format!("a call of {}, which has no evaluated definition", name)));
        }
        let inst = m.functions[fr.f].inst(id);
        let call_attrs = match &inst.op {
            Op::Call { attrs, .. } => attrs.function,
            _ => crate::ir::AttrSetId::EMPTY,
        };
        let me = memory_effects(m.attrs(decl.attrs.function)).min(memory_effects(m.attrs(call_attrs)));
        let loc = |l: usize| (me >> (2 * l)) & 3;
        let (mut arg, mut inacc, errno, other) = (loc(0), loc(1), loc(2), loc(3));
        let has_ptr = args.iter().any(|v| {
            let mut p = Vec::new();
            collect_ptrs(v, &mut p);
            !p.is_empty()
        }) || m
            .types
            .function(decl.ty)
            .is_some_and(|(_, ps, _)| ps.iter().any(|&t| m.types.is_ptr_or_ptr_vector(t)));
        let known = effect_free(&name, has_ptr);
        match known {
            // A classified family: its effects are what the classification says, whatever the declaration states.
            Some(Effect::Value) => {
                arg = 0;
                inacc = 0;
            }
            Some(Effect::Event) => {
                arg = 0;
                inacc = 2;
            }
            None => {
                if other & 2 != 0 || errno & 2 != 0 {
                    return Err(self.unsupported(
                        fr,
                        format!("a call of {}, which may write memory the interpreter models", name),
                    ));
                }
                if other & 1 != 0 {
                    return Err(self.unsupported(fr, format!("a call of {}, which may read any memory", name)));
                }
            }
        }
        // Pointer arguments, and what the call may do through them.
        let mut ptrs = Vec::new();
        for v in &args {
            collect_ptrs(v, &mut ptrs);
        }
        let mut digest = Fnv::new();
        digest.str(&name);
        let mut indefinite = false;
        for v in &args {
            self.hash_val(&mut digest, v, &mut indefinite);
        }
        if arg != 0 {
            for p in &ptrs {
                if p.poison || p.undef != 0 {
                    return Err(self.ub(fr, format!("a call of {} with a pointer that is not well defined", name)));
                }
                if p.choice != 0 {
                    self.choice_path = true;
                }
                let r = if p.prov != 0 { Some(p.prov) } else { self.mem.region_at(p.bits as u64) };
                let Some(r) = r else {
                    return Err(self.ub(fr, format!("a call of {} with a pointer to no object", name)));
                };
                let region = self.mem.region(r);
                let is_object = matches!(region.kind, RegionKind::Object | RegionKind::Function(_));
                if arg & 2 != 0 && !is_object {
                    return Err(self.unsupported(
                        fr,
                        format!("a call of {}, which may write {} through an argument", name, region.label),
                    ));
                }
                if arg & 1 != 0 {
                    if is_object {
                        digest.u64(region.version);
                    } else {
                        for b in &region.bytes {
                            digest.bytes(&[b.v, b.undef, b.choice, b.poison as u8, b.frag]);
                            digest.u64(b.prov as u64);
                        }
                    }
                }
            }
        }
        let writes = arg & 2 != 0 || inacc & 2 != 0;
        if writes {
            if arg & 2 != 0 {
                for p in &ptrs {
                    let r = if p.prov != 0 { p.prov } else { self.mem.region_at(p.bits as u64).unwrap_or(0) };
                    if r != 0 {
                        self.mem.region_mut(r).version += 1;
                    }
                }
            }
            self.event(&name, args.clone());
            digest.u64(self.events.len() as u64);
        }
        self.uninterpreted += 1;
        let ty = inst.ty;
        if self.m.types.is_void(ty) {
            return Ok(None);
        }
        let seed = digest.finish();
        let mut k = 0u64;
        let v = self.opaque_value(fr, ty, seed, &mut k, indefinite, &name)?;
        Ok(Some(v))
    }

    fn hash_val(&self, h: &mut Fnv, v: &Val, indefinite: &mut bool) {
        match v {
            Val::S(sc) => self.hash_scalar(h, sc, indefinite),
            Val::V(l) => {
                h.u64(l.len() as u64);
                for sc in l {
                    self.hash_scalar(h, sc, indefinite);
                }
            }
            Val::A(items) => {
                h.u64(items.len() as u64);
                for it in items {
                    self.hash_val(h, it, indefinite);
                }
            }
            Val::Md(md) => h.u64(md.0 as u64 | 1 << 40),
            Val::Token => h.u64(7),
        }
    }

    fn hash_scalar(&self, h: &mut Fnv, sc: &Scalar, indefinite: &mut bool) {
        if sc.poison || sc.undef != 0 {
            *indefinite = true;
            return;
        }
        if sc.prov != 0 {
            // A pointer hashes as its object and offset, which are the same in two runs of the same function.
            let r = self.mem.region(sc.prov);
            h.str(&r.label);
            h.u64((sc.bits as u64).wrapping_sub(r.base));
        } else {
            h.u64(sc.bits as u64);
            h.u64((sc.bits >> 64) as u64);
        }
    }

    /// A value of type `ty` determined by `seed`: each scalar leaf takes the next bits of the stream. Pointer leaves
    /// point to an object made for that value.
    fn opaque_value(&mut self, fr: &Frame, ty: TypeId, seed: u64, k: &mut u64, indefinite: bool, name: &str) -> R<Val> {
        let types = &self.m.types;
        Ok(match types.get(ty).clone() {
            Type::Vector { len, elem, .. } => {
                let mut out = Vec::new();
                for _ in 0..len {
                    out.push(s(&self.opaque_value(fr, elem, seed, k, indefinite, name)?));
                }
                Val::V(out)
            }
            Type::Array { len, elem } => {
                let mut out = Vec::new();
                for _ in 0..len {
                    out.push(self.opaque_value(fr, elem, seed, k, indefinite, name)?);
                }
                Val::A(out)
            }
            Type::Struct { .. } | Type::LiteralStruct { .. } => {
                let elems = types.struct_elems(ty).unwrap_or(&[]).to_vec();
                let mut out = Vec::new();
                for e in elems {
                    out.push(self.opaque_value(fr, e, seed, k, indefinite, name)?);
                }
                Val::A(out)
            }
            Type::Ptr(a) => {
                *k += 1;
                if indefinite {
                    return Ok(Val::S(Scalar::undef(64)));
                }
                let h = mix(seed, *k);
                let label = format!("object from {} #{:016x}", name, h);
                let found = self.mem.regions().find(|(_, r)| r.label == label).map(|(i, _)| i);
                let r = match found {
                    Some(r) => r,
                    None => self.mem.add(RegionKind::Object, a, Vec::new(), false, label),
                };
                Val::S(self.mem.pointer(r, 0))
            }
            Type::Token => Val::Token,
            _ => {
                let w = self.prim_of(fr, TyRef::Id(ty))?.width();
                *k += 1;
                if indefinite {
                    Val::S(Scalar::undef(w))
                } else {
                    // Any fixed function is an uninterpreted function; this one keeps its values in the ranges shaders
                    // use them in (integers below 64, as sizes and indices; floats in [-4, 4), never NaN or
                    // infinite), so that runs reach past them instead of looping on a huge size or turning fast-math
                    // code into poison.
                    let h = mix(seed, *k);
                    let bits = match self.prim_of(fr, TyRef::Id(ty))? {
                        Prim::Float(f) => f.from_f64(((h % 8000) as f64 - 4000.0) / 1000.0) as u128,
                        _ => (h % 64) as u128 & mask(w),
                    };
                    Val::S(Scalar::int(bits))
                }
            }
        })
    }
}

/// The interpreter's view of a family whose declaration states no memory effects but whose definition touches no
/// memory the module can address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Effect {
    /// A value computed from the arguments and from other invocations or the fixed-function state (SIMD-group and
    /// quad-group functions, derivatives).
    Value,
    /// An effect outside the module's memory: synchronization, fragment discard.
    Event,
}

/// What the interpreter may assume of an `air.*` function whose declaration does not state its memory effects.
///
/// - The numeric families of `tables/intrinsics.toml` (math-float, math-geometric, integer, relational, conversion,
///   pack-unpack, derivative: MSL 6.4 to 6.8 and 6.15) are functions of their operands with no memory access; the
///   ones that take a pointer (`frexp`, `modf`, `sincos` with an out-parameter) are not classified here.
/// - Synchronization (MSL 6.9: `wg.barrier`, `simdgroup.barrier`) and `discard_fragment` (MSL 6.10) act on other
///   invocations and the pipeline, never on memory the shader addresses: events.
/// - SIMD-group and quad-group functions (MSL 6.12, 6.12.2) are values computed from the operands of other
///   invocations, which a single invocation's run does not see: values of the operands. So are the SIMD-group
///   matrix functions without pointer operands (MSL 6.7).
fn effect_free(name: &str, has_pointer_operand: bool) -> Option<Effect> {
    let n = name.strip_prefix("air.")?;
    let event = ["wg.barrier", "simdgroup.barrier", "discard_fragment"];
    if event.iter().any(|p| n.starts_with(p)) {
        return Some(Effect::Event);
    }
    if has_pointer_operand {
        return None;
    }
    // MSL 6.7: SIMD-group matrix functions without pointer operands (init, multiply, multiply-accumulate) are values.
    if ["simd_", "quad_", "simdgroup_matrix_"].iter().any(|p| n.starts_with(p)) {
        return Some(Effect::Value);
    }
    let base = n.strip_prefix("fast_").unwrap_or(n);
    let base = base.split('.').next().unwrap_or("");
    const NUMERIC: &[&str] = &[
        // math-float
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "copysign",
        "cos",
        "cosh",
        "cospi",
        "divide",
        "exp",
        "exp2",
        "exp10",
        "fabs",
        "fdim",
        "floor",
        "fma",
        "fmax",
        "fmax3",
        "fmedian3",
        "fmin",
        "fmin3",
        "fmod",
        "fract",
        "frexp",
        "ilogb",
        "ldexp",
        "log",
        "log2",
        "log10",
        "modf",
        "nextafter",
        "pow",
        "powr",
        "rint",
        "round",
        "rsqrt",
        "sin",
        "sincos",
        "sinh",
        "sinpi",
        "sqrt",
        "tan",
        "tanh",
        "tanpi",
        "trunc",
        "saturate",
        "mix",
        "smoothstep",
        "step",
        "sign",
        "clamp",
        // math-geometric
        "dot",
        "distance",
        "distance_squared",
        "length",
        "length_squared",
        "normalize",
        "cross",
        "reflect",
        "refract",
        "faceforward",
        // integer
        "abs",
        "abs_diff",
        "add_sat",
        "hadd",
        "mad_hi",
        "mad_sat",
        "max",
        "max3",
        "median3",
        "min",
        "min3",
        "mul_hi",
        "rhadd",
        "sub_sat",
        "mad24",
        "mul24",
        "extract_bits",
        "insert_bits",
        "clz",
        "ctz",
        "popcount",
        "reverse_bits",
        "rotate",
        // relational, conversion, pack-unpack, derivative
        "all",
        "any",
        "convert",
        "pack",
        "unpack",
        "dfdx",
        "dfdy",
        "fwidth",
    ];
    if NUMERIC.contains(&base) {
        return Some(Effect::Value);
    }
    None
}

/// The `MemoryEffects` of an attribute set: unknown (all read and write) when it has none.
fn memory_effects(set: &crate::ir::AttrSet) -> u64 {
    if set.has(AttrKind::ReadNone) {
        return 0;
    }
    match set.get(AttrKind::Memory) {
        Some(Attribute::Int(_, me)) => *me,
        _ => 0xff,
    }
}

fn collect_ptrs(v: &Val, out: &mut Vec<Scalar>) {
    match v {
        Val::S(sc) if sc.prov != 0 || (sc.bits != 0 && !sc.poison) => {
            if sc.prov != 0 {
                out.push(*sc)
            }
        }
        Val::V(l) => out.extend(l.iter().filter(|x| x.prov != 0).copied()),
        Val::A(items) => items.iter().for_each(|i| collect_ptrs(i, out)),
        _ => {}
    }
}

/// The operand `k` of call `id` (its value operand, for types).
fn operand(it: &Interp<'_>, fr: &Frame, id: InstId, k: usize) -> Value {
    match &it.m.functions[fr.f].inst(id).op {
        Op::Call { args, .. } => args[k],
        _ => Value::Arg(0),
    }
}

fn sext(v: u128, w: u32) -> i128 {
    if w >= 128 {
        return v as i128;
    }
    let sh = 128 - w;
    ((v << sh) as i128) >> sh
}

fn keep_choice(from: Scalar, mut to: Scalar, w: u32) -> Scalar {
    if from.choice != 0 {
        to.choice = mask(w);
    }
    to
}

fn defined1(x: &Scalar, w: u32, f: impl Fn(u128) -> u128) -> Scalar {
    if x.poison {
        return Scalar::POISON;
    }
    if x.undef != 0 {
        return Scalar::undef(w);
    }
    keep_choice(*x, Scalar::int(f(x.bits & mask(w)) & mask(w)), w)
}

fn defined2(x: &Scalar, y: &Scalar, w: u32, f: impl Fn(u128, u128) -> u128) -> Scalar {
    if x.poison || y.poison {
        return Scalar::POISON;
    }
    if x.undef != 0 || y.undef != 0 {
        return Scalar::undef(w);
    }
    let mut r = Scalar::int(f(x.bits & mask(w), y.bits & mask(w)) & mask(w));
    if x.choice | y.choice != 0 {
        r.choice = mask(w);
    }
    r
}

fn sat_addsub(base: &str, p: u128, q: u128, w: u32) -> u128 {
    let m = mask(w);
    let hi = (1i128 << (w - 1)) - 1;
    let lo = -(1i128 << (w - 1));
    match base {
        "uadd" => (p + q).min(m),
        "usub" => p.saturating_sub(q),
        "sadd" => ((sext(p, w) + sext(q, w)).clamp(lo, hi) as u128) & m,
        _ => ((sext(p, w) - sext(q, w)).clamp(lo, hi) as u128) & m,
    }
}

fn mul_hi_unsigned(a: u128, b: u128, w: u32) -> u128 {
    if w <= 64 {
        (a * b) >> w
    } else {
        // 128-bit operands do not occur (AIR integers are at most 64 bits).
        0
    }
}

fn mul_hi_signed(a: i128, b: i128, w: u32) -> Option<u128> {
    if w <= 64 { Some(((a * b) >> w) as u128 & mask(w)) } else { None }
}

fn mad_sat_unsigned(a: u128, b: u128, c: u128, w: u32) -> u128 {
    let m = mask(w);
    match a.checked_mul(b).and_then(|p| p.checked_add(c)) {
        Some(v) => v.min(m),
        None => m,
    }
}

fn mad_sat_signed(a: i128, b: i128, c: i128, w: u32) -> u128 {
    let hi = (1i128 << (w - 1)) - 1;
    let lo = -(1i128 << (w - 1));
    let v = a.checked_mul(b).and_then(|p| p.checked_add(c)).unwrap_or(if (a < 0) != (b < 0) { lo } else { hi });
    (v.clamp(lo, hi) as u128) & mask(w)
}

/// A bitwise operation on a float's encoding (not a math operation: no NaN rules).
fn bitwise_float(x: &Scalar, f: impl Fn(u128) -> u128) -> Scalar {
    if x.poison {
        return Scalar::POISON;
    }
    let mut r = *x;
    r.bits = f(x.bits) & !x.undef;
    r
}

/// LLVM's `minnum`/`maxnum` (IEEE 754-2008 minNum/maxNum with -0 < +0; a signaling NaN operand gives a NaN),
/// `minimum`/`maximum` (NaN if either operand is NaN) and `minimumnum`/`maximumnum` (a NaN operand, quiet or
/// signaling, yields the other operand).
pub fn minmax(name: &str, f: Format, fmf: FastMath, a: &Scalar, b: &Scalar) -> Scalar {
    let is_max = name.starts_with("max");
    let kind = name.trim_start_matches("min").trim_start_matches("max");
    ops::float_math(f, fmf, &[*a, *b], &|v| {
        let (x, y) = (v[0], v[1]);
        let (xn, yn) = (f.is_nan(x), f.is_nan(y));
        match kind {
            "num" => {
                if f.is_snan(x) || f.is_snan(y) || (xn && yn) {
                    return f.preferred_nan();
                }
            }
            "imum" => {
                if xn || yn {
                    return f.preferred_nan();
                }
            }
            _ => {
                if xn && yn {
                    return f.preferred_nan();
                }
            }
        }
        if xn {
            return y;
        }
        if yn {
            return x;
        }
        // Order with -0 below +0.
        let key = |v: u64| -> (f64, bool) { (f.to_f64(v), !f.is_neg(v)) };
        let (kx, ky) = (key(x), key(y));
        let x_less = kx.0 < ky.0 || (kx.0 == ky.0 && !kx.1 && ky.1);
        if is_max == x_less { y } else { x }
    })
}

/// One lane of `air.convert.<dk>.<dty>.<sk>.<sty>`.
fn convert(dk: char, dp: Prim, sk: char, sp: Prim, x: &Scalar) -> Scalar {
    if x.poison {
        return Scalar::POISON;
    }
    let (sw, dw) = (sp.width(), dp.width());
    if x.undef != 0 {
        return Scalar::undef(dw);
    }
    let r = match (dk, dp, sk, sp) {
        ('f', Prim::Float(df), 'f', Prim::Float(sf)) => {
            let v = x.bits as u64;
            if sf.is_nan(v) {
                // A NaN converts to a NaN; its payload follows the NaN rules (the preferred NaN is chosen).
                return ops::nan_result(df, &[]);
            }
            Scalar::int(df.from_f64(sf.to_f64(v)) as u128)
        }
        (_, Prim::Int(1), 'f', Prim::Float(sf)) => {
            // bool(x) is x != 0, and NaN converts to true (measured by the numerics package).
            let v = x.bits as u64;
            Scalar::bool(sf.is_nan(v) || !sf.is_zero(v))
        }
        (_, Prim::Int(_), 'f', Prim::Float(sf)) => {
            let v = x.bits as u64;
            if sf.is_nan(v) {
                Scalar::int(0)
            } else {
                match ops::float_to_int(sf, v, dw, dk == 's') {
                    Some(r) => Scalar::int(r),
                    None => {
                        // Out of range: MSL leaves the result unspecified; the saturated value is computed.
                        let neg = sf.is_neg(v);
                        let sat = if dk == 's' {
                            if neg { 1u128 << (dw - 1) } else { mask(dw - 1) }
                        } else if neg {
                            0
                        } else {
                            mask(dw)
                        };
                        Scalar { bits: sat, undef: 0, choice: mask(dw), prov: 0, poison: false }
                    }
                }
            }
        }
        ('f', Prim::Float(df), _, Prim::Int(_)) => {
            let (neg, mag) = if sk == 's' {
                let v = sext(x.bits & mask(sw), sw);
                (v < 0, v.unsigned_abs())
            } else {
                (false, x.bits & mask(sw))
            };
            Scalar::int(float::from_int(df, neg, mag) as u128)
        }
        (_, Prim::Int(1), _, Prim::Int(_)) => Scalar::bool(x.bits & mask(sw) != 0),
        (_, Prim::Int(_), _, Prim::Int(_)) => {
            let v = if sk == 's' { sext(x.bits & mask(sw), sw) as u128 } else { x.bits & mask(sw) };
            Scalar::int(v & mask(dw))
        }
        _ => Scalar::undef(dw),
    };
    if x.choice != 0 && !r.poison {
        let mut r = r;
        r.choice = mask(dw);
        return r;
    }
    r
}

/// FNV-1a, 64-bit.
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// SplitMix64 of `seed + k`: the k-th word of a value's stream.
fn mix(seed: u64, k: u64) -> u64 {
    let mut z = seed.wrapping_add(k.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
