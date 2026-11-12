//! The value semantics of LLVM's arithmetic, bitwise, comparison and conversion instructions (LangRef "Instruction
//! Reference", release 21.1.8), element by element.
//!
//! Operands follow [`super::value`]: poison operands give poison (division takes its divisor first), undef bits are
//! carried exactly through bitwise operations and shifts by a defined amount, and widen to the bits an operation can
//! reach otherwise (the low bits of an addition, subtraction or multiplication below the lowest undef bit stay exact;
//! every other result bit becomes undef; a floating-point operation with an undef bit gives an undef result). Where
//! some value of the undef bits would make the result poison (a wrapping `nuw` addition, an out-of-range shift), the
//! result is poison: a set of outcomes that contains poison admits every outcome. Where some value would make the
//! operation undefined behavior (a zero divisor), it is.
//!
//! Choice bits spread as undef bits would, but never make an operation poison or undefined: the interpreter computes
//! with the value it chose.

use super::float::{self, Arith, Format};
use super::value::{Scalar, mask};
use crate::ir::{BinOp, CastOp, FastMath, FloatPredicate, IntPredicate, flags};

/// Undefined behavior, with what triggered it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ub(pub String);

/// The kind of one scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prim {
    Int(u32),
    Float(Format),
    /// A pointer: address space and width in bits.
    Ptr(u32, u32),
}

impl Prim {
    pub fn width(self) -> u32 {
        match self {
            Prim::Int(w) => w,
            Prim::Float(f) => f.width(),
            Prim::Ptr(_, w) => w,
        }
    }
}

fn sext(v: u128, w: u32) -> i128 {
    if w >= 128 {
        return v as i128;
    }
    let sh = 128 - w;
    ((v << sh) as i128) >> sh
}

fn sign_bit(w: u32) -> u128 {
    1u128 << (w - 1)
}

/// Unsigned bounds of the values an operand may take.
fn urange(s: &Scalar, w: u32) -> (u128, u128) {
    (s.bits & !s.undef & mask(w), (s.bits | s.undef) & mask(w))
}

/// Signed bounds of the values an operand may take.
fn srange(s: &Scalar, w: u32) -> (i128, i128) {
    let sb = sign_bit(w);
    let known = s.bits & !s.undef & mask(w);
    let lo = known | (s.undef & sb);
    let hi = (known | (s.undef & !sb)) & mask(w);
    (sext(lo, w), sext(hi, w))
}

/// Whether the operand may take value `v`.
fn may_be(s: &Scalar, v: u128, w: u32) -> bool {
    (s.bits ^ v) & !s.undef & mask(w) == 0
}

/// Marks every bit of `r` as a choice when an operand has a choice bit.
fn spread_choice(mut r: Scalar, w: u32, ops: &[&Scalar]) -> Scalar {
    if !r.poison && ops.iter().any(|o| o.choice != 0) {
        r.choice = mask(w) & !r.undef;
    }
    r
}

fn all_undef(w: u32) -> Scalar {
    Scalar::undef(w)
}

/// An integer binary operator on `w`-bit operands.
pub fn int_binop(op: BinOp, fl: u8, w: u32, a: &Scalar, b: &Scalar) -> Result<Scalar, Ub> {
    let m = mask(w);
    let nuw = fl & flags::NUW != 0;
    let nsw = fl & flags::NSW != 0;
    let exact = fl & flags::EXACT != 0;
    match op {
        BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => {
            let signed = matches!(op, BinOp::SDiv | BinOp::SRem);
            if b.poison {
                return Err(Ub(format!("{} by a poison divisor", op.name())));
            }
            if may_be(b, 0, w) {
                return Err(Ub(format!(
                    "{} by zero{}",
                    op.name(),
                    if b.undef != 0 { " (an undef divisor)" } else { "" }
                )));
            }
            if a.poison {
                return Ok(Scalar::POISON);
            }
            if signed && may_be(a, sign_bit(w), w) && may_be(b, m, w) {
                return Err(Ub(format!("{} overflow (the minimum value by -1)", op.name())));
            }
            if a.undef != 0 || b.undef != 0 {
                return Ok(if exact { Scalar::POISON } else { spread_choice(all_undef(w), w, &[a, b]) });
            }
            let (x, y) = (a.bits & m, b.bits & m);
            let (q, r) = if signed {
                let (sx, sy) = (sext(x, w), sext(y, w));
                // Rust's / and % truncate toward zero, as LLVM's sdiv and srem do.
                ((sx / sy) as u128 & m, (sx % sy) as u128 & m)
            } else {
                (x / y, x % y)
            };
            if exact && matches!(op, BinOp::UDiv | BinOp::SDiv) && r != 0 {
                return Ok(Scalar::POISON);
            }
            let v = if matches!(op, BinOp::UDiv | BinOp::SDiv) { q } else { r };
            return Ok(spread_choice(Scalar::int(v), w, &[a, b]));
        }
        _ => {}
    }
    if a.poison || b.poison {
        return Ok(Scalar::POISON);
    }
    let r = match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul => {
            let u = (a.undef | b.undef) & m;
            if u != 0 {
                if (nuw || nsw) && may_wrap(op, nuw, nsw, w, a, b) {
                    return Ok(Scalar::POISON);
                }
                let low = u.trailing_zeros().min(w);
                let (x, y) = (a.bits & !a.undef, b.bits & !b.undef);
                let v = match op {
                    BinOp::Add => x.wrapping_add(y),
                    BinOp::Sub => x.wrapping_sub(y),
                    _ => x.wrapping_mul(y),
                } & mask(low);
                Scalar { bits: v, undef: m & !mask(low), choice: 0, prov: 0, poison: false }
            } else {
                let (x, y) = (a.bits & m, b.bits & m);
                if (nuw || nsw) && may_wrap(op, nuw, nsw, w, a, b) {
                    return Ok(Scalar::POISON);
                }
                let v = match op {
                    BinOp::Add => x.wrapping_add(y),
                    BinOp::Sub => x.wrapping_sub(y),
                    _ => x.wrapping_mul(y),
                } & m;
                Scalar::int(v)
            }
        }
        BinOp::Shl | BinOp::LShr | BinOp::AShr => {
            if b.undef != 0 {
                let (_, hi) = urange(b, w);
                // Some value of the amount reaches the width: poison is among the outcomes.
                if hi >= w as u128 {
                    return Ok(Scalar::POISON);
                }
                return Ok(spread_choice(all_undef(w), w, &[a, b]));
            }
            let s = b.bits & m;
            if s >= w as u128 {
                return Ok(Scalar::POISON);
            }
            let s = s as u32;
            let (x, xu) = (a.bits & m, a.undef & m);
            match op {
                BinOp::Shl => {
                    if s > 0 && (nuw || nsw) {
                        let out_mask = m & !(m >> s);
                        // The bits shifted out (and, for nsw, the new sign bit) must be known.
                        let check = if nsw { out_mask | (sign_bit(w) >> s) } else { out_mask };
                        if xu & check != 0 {
                            return Ok(Scalar::POISON);
                        }
                        if nuw && x & out_mask != 0 {
                            return Ok(Scalar::POISON);
                        }
                        if nsw && sext((x << s) & m, w) >> s != sext(x, w) {
                            return Ok(Scalar::POISON);
                        }
                    }
                    let mut r = Scalar::int((x << s) & m);
                    r.undef = (xu << s) & m;
                    r.bits &= !r.undef;
                    r.choice = (a.choice << s) & m & !r.undef;
                    return Ok(spread_choice_from(r, w, b));
                }
                BinOp::LShr | BinOp::AShr => {
                    if exact && s > 0 && (x | xu) & mask(s) != 0 {
                        return Ok(Scalar::POISON);
                    }
                    let mut r = if op == BinOp::LShr {
                        let mut r = Scalar::int(x >> s);
                        r.undef = xu >> s;
                        r.choice = (a.choice & m) >> s;
                        r
                    } else {
                        let fill = m & !(m >> s);
                        let neg = x & sign_bit(w) != 0;
                        let su = xu & sign_bit(w) != 0;
                        let mut r = Scalar::int((x >> s) | if neg { fill } else { 0 });
                        r.undef = (xu >> s) | if su { fill } else { 0 };
                        let sc = a.choice & sign_bit(w) != 0;
                        r.choice = ((a.choice & m) >> s) | if sc { fill } else { 0 };
                        r
                    };
                    r.bits &= !r.undef;
                    r.choice &= !r.undef;
                    return Ok(spread_choice_from(r, w, b));
                }
                _ => unreachable!(),
            }
        }
        BinOp::And | BinOp::Or | BinOp::Xor => {
            let ka0 = !a.bits & !a.undef & m;
            let kb0 = !b.bits & !b.undef & m;
            let ka1 = a.bits & !a.undef & m;
            let kb1 = b.bits & !b.undef & m;
            let u = (a.undef | b.undef) & m;
            let (bits, undef) = match op {
                BinOp::And => {
                    let und = u & !(ka0 | kb0);
                    (a.bits & b.bits & m & !und, und)
                }
                BinOp::Or => {
                    if fl & flags::DISJOINT != 0 && (a.bits | a.undef) & (b.bits | b.undef) & m != 0 {
                        return Ok(Scalar::POISON);
                    }
                    let und = u & !(ka1 | kb1);
                    ((a.bits | b.bits) & m & !und, und)
                }
                _ => ((a.bits ^ b.bits) & m & !u, u),
            };
            let known = match op {
                BinOp::And => ka0 | kb0,
                BinOp::Or => ka1 | kb1,
                _ => 0,
            };
            // A choice bit stays a choice where the other operand does not force the result.
            let choice = match op {
                BinOp::Xor => (a.choice | b.choice) & m,
                _ => (a.choice | b.choice) & m & !known,
            } & !undef;
            return Ok(Scalar { bits, undef, choice, prov: 0, poison: false });
        }
        _ => unreachable!("not an integer operator"),
    };
    Ok(spread_choice(r, w, &[a, b]))
}

fn spread_choice_from(r: Scalar, w: u32, amount: &Scalar) -> Scalar {
    if amount.choice != 0 { spread_choice(r, w, &[amount]) } else { r }
}

/// Whether some values of the operands make `op` wrap in the senses its flags forbid.
fn may_wrap(op: BinOp, nuw: bool, nsw: bool, w: u32, a: &Scalar, b: &Scalar) -> bool {
    let m = mask(w);
    let (alo, ahi) = urange(a, w);
    let (_, bhi) = urange(b, w);
    let (salo, sahi) = srange(a, w);
    let (sblo, sbhi) = srange(b, w);
    let smin = if w >= 128 { i128::MIN } else { -(1i128 << (w - 1)) };
    let smax = if w >= 128 { i128::MAX } else { (1i128 << (w - 1)) - 1 };
    let sfits = |v: Option<i128>| v.is_some_and(|v| v >= smin && v <= smax);
    match op {
        BinOp::Add => {
            (nuw && ahi.checked_add(bhi).is_none_or(|s| s > m))
                || (nsw && !(sfits(salo.checked_add(sblo)) && sfits(sahi.checked_add(sbhi))))
        }
        BinOp::Sub => (nuw && alo < bhi) || (nsw && !(sfits(salo.checked_sub(sbhi)) && sfits(sahi.checked_sub(sblo)))),
        BinOp::Mul => {
            // The largest values the operands may take give the largest product.
            let uw = nuw && ahi.checked_mul(bhi).is_none_or(|p| p > m);
            let sw = nsw && {
                let corners =
                    [salo.checked_mul(sblo), salo.checked_mul(sbhi), sahi.checked_mul(sblo), sahi.checked_mul(sbhi)];
                !corners.iter().all(|c| sfits(*c))
            };
            uw || sw
        }
        _ => false,
    }
}

/// `icmp` of two `w`-bit operands (pointers compare their addresses).
pub fn icmp(pred: IntPredicate, samesign: bool, w: u32, a: &Scalar, b: &Scalar) -> Scalar {
    if a.poison || b.poison {
        return Scalar::POISON;
    }
    let m = mask(w);
    if samesign {
        let sb = sign_bit(w);
        if (a.undef | b.undef) & sb != 0 || (a.bits ^ b.bits) & sb != 0 {
            return Scalar::POISON;
        }
    }
    let decided = |lt: bool, gt: bool, eq: bool| -> bool {
        match pred {
            IntPredicate::Eq => eq,
            IntPredicate::Ne => !eq,
            IntPredicate::Ugt | IntPredicate::Sgt => gt,
            IntPredicate::Uge | IntPredicate::Sge => gt || eq,
            IntPredicate::Ult | IntPredicate::Slt => lt,
            IntPredicate::Ule | IntPredicate::Sle => lt || eq,
        }
    };
    let r = if a.undef == 0 && b.undef == 0 {
        let (x, y) = (a.bits & m, b.bits & m);
        let ord = match pred {
            IntPredicate::Sgt | IntPredicate::Sge | IntPredicate::Slt | IntPredicate::Sle => {
                sext(x, w).cmp(&sext(y, w))
            }
            _ => x.cmp(&y),
        };
        Scalar::bool(decided(ord.is_lt(), ord.is_gt(), ord.is_eq()))
    } else {
        // The result is defined when every value of the undef bits gives the same answer.
        let differ = (a.bits ^ b.bits) & !(a.undef | b.undef) & m != 0;
        let outcome = match pred {
            IntPredicate::Eq | IntPredicate::Ne => {
                if differ {
                    Some(decided(false, false, false))
                } else {
                    None
                }
            }
            IntPredicate::Ugt | IntPredicate::Uge | IntPredicate::Ult | IntPredicate::Ule => {
                let (alo, ahi) = urange(a, w);
                let (blo, bhi) = urange(b, w);
                range_outcome(alo, ahi, blo, bhi, &decided)
            }
            _ => {
                let (alo, ahi) = srange(a, w);
                let (blo, bhi) = srange(b, w);
                range_outcome(alo, ahi, blo, bhi, &decided)
            }
        };
        match outcome {
            Some(v) => Scalar::bool(v),
            None => Scalar::undef(1),
        }
    };
    spread_choice(r, 1, &[a, b])
}

fn range_outcome<T: Ord + Copy>(
    alo: T,
    ahi: T,
    blo: T,
    bhi: T,
    decided: &dyn Fn(bool, bool, bool) -> bool,
) -> Option<bool> {
    if ahi < blo {
        Some(decided(true, false, false))
    } else if alo > bhi {
        Some(decided(false, true, false))
    } else if alo == ahi && blo == bhi && alo == blo {
        Some(decided(false, false, true))
    } else {
        // Both answers are possible unless the predicate is true (or false) for all of lt, eq and gt that remain.
        let mut seen = Vec::new();
        if alo < bhi {
            seen.push(decided(true, false, false));
        }
        if ahi > blo {
            seen.push(decided(false, true, false));
        }
        if alo <= bhi && blo <= ahi {
            seen.push(decided(false, false, true));
        }
        if seen.iter().all(|&v| v == seen[0]) { Some(seen[0]) } else { None }
    }
}

// Floating point ------------------------------------------------------------------------------------------------------

/// The NaN a floating-point math operation returns (LangRef "Behavior of Floating-Point NaN values"): the sign is a
/// choice; the quiet bit is set and the payload is zero or copied from a NaN operand, so the payload bits that some
/// NaN operand has set are choices; with a signaling NaN operand the quiet bit may stay clear, so it is a choice too.
/// The interpreter chooses the preferred NaN.
pub fn nan_result(fmt: Format, operands: &[u64]) -> Scalar {
    let mut choice = fmt.sign_bit() as u128;
    for &o in operands {
        if fmt.is_nan(o) {
            choice |= (o & fmt.frac_mask() & !fmt.quiet_bit()) as u128;
            if fmt.is_snan(o) {
                choice |= fmt.quiet_bit() as u128;
            }
        }
    }
    Scalar { bits: fmt.preferred_nan() as u128, undef: 0, choice, prov: 0, poison: false }
}

/// Applies `nnan` and `ninf` to an operation's operands and result: poison when one is NaN or infinite.
fn fmf_poison(fmt: Format, fmf: FastMath, operands: &[u64], result: u64) -> bool {
    let nnan = fmf.0 & FastMath::NNAN != 0;
    let ninf = fmf.0 & FastMath::NINF != 0;
    operands.iter().chain(std::iter::once(&result)).any(|&v| (nnan && fmt.is_nan(v)) || (ninf && fmt.is_inf(v)))
}

/// Evaluates a floating-point math operation `f` on defined operands under fast-math flags `fmf`: poison per
/// `nnan`/`ninf`, the NaN rules for a NaN result, and under `nsz` the sign of a zero operand or result as a choice.
pub fn float_math(fmt: Format, fmf: FastMath, operands: &[Scalar], f: &dyn Fn(&[u64]) -> u64) -> Scalar {
    let w = fmt.width();
    if operands.iter().any(|o| o.poison) {
        return Scalar::POISON;
    }
    let refs: Vec<&Scalar> = operands.iter().collect();
    if operands.iter().any(|o| o.undef != 0) {
        return spread_choice(all_undef(w), w, &refs);
    }
    let vals: Vec<u64> = operands.iter().map(|o| o.bits as u64).collect();
    let r = f(&vals);
    if fmf_poison(fmt, fmf, &vals, r) {
        return Scalar::POISON;
    }
    let mut out = if fmt.is_nan(r) { nan_result(fmt, &vals) } else { Scalar::int(r as u128) };
    if fmf.0 & FastMath::NSZ != 0 {
        // Each zero operand may be taken with either sign; collect the bits on which the results differ.
        let zeros: Vec<usize> = (0..vals.len()).filter(|&i| fmt.is_zero(vals[i])).collect();
        for combo in 1u32..(1 << zeros.len()) {
            let mut alt = vals.clone();
            for (k, &i) in zeros.iter().enumerate() {
                if combo & (1 << k) != 0 {
                    alt[i] ^= fmt.sign_bit();
                }
            }
            let r2 = f(&alt);
            if fmt.is_nan(r2) != fmt.is_nan(r) {
                // A zero's sign that decides between a NaN and a number: every bit is open.
                out.choice = mask(w);
            } else if !fmt.is_nan(r2) {
                out.choice |= ((r ^ r2) & fmt.mask()) as u128;
            }
        }
        if fmt.is_zero(r) {
            out.choice |= fmt.sign_bit() as u128;
        }
    }
    let mut out = spread_choice_keep(out, w, &refs);
    out.choice &= mask(w);
    out
}

/// Like [`spread_choice`], but keeps choice bits the operation set itself.
fn spread_choice_keep(r: Scalar, w: u32, ops: &[&Scalar]) -> Scalar {
    if r.poison {
        return r;
    }
    let c = r.choice;
    let mut r = spread_choice(r, w, ops);
    r.choice |= c;
    r
}

/// A floating-point binary operator.
pub fn float_binop(op: BinOp, fmf: FastMath, fmt: Format, a: &Scalar, b: &Scalar) -> Scalar {
    let ar = match op {
        BinOp::FAdd => Arith::Add,
        BinOp::FSub => Arith::Sub,
        BinOp::FMul => Arith::Mul,
        BinOp::FDiv => Arith::Div,
        BinOp::FRem => Arith::Rem,
        _ => unreachable!("not a floating-point operator"),
    };
    float_math(fmt, fmf, &[*a, *b], &|v| float::binary(fmt, ar, v[0], v[1]))
}

/// `fneg`: flips the sign bit, NaN included (it is not a math operation); `nnan`/`ninf` still apply.
pub fn fneg(fmf: FastMath, fmt: Format, a: &Scalar) -> Scalar {
    if a.poison {
        return Scalar::POISON;
    }
    if a.undef == 0 && fmf_poison(fmt, fmf, &[a.bits as u64], a.bits as u64 ^ fmt.sign_bit()) {
        return Scalar::POISON;
    }
    let mut r = *a;
    r.bits ^= fmt.sign_bit() as u128;
    r.bits &= !r.undef;
    if fmf.0 & FastMath::NSZ != 0 && a.undef == 0 && fmt.is_zero(a.bits as u64) {
        r.choice |= fmt.sign_bit() as u128;
    }
    r
}

/// `fcmp`.
pub fn fcmp(pred: FloatPredicate, fmf: FastMath, fmt: Format, a: &Scalar, b: &Scalar) -> Scalar {
    if a.poison || b.poison {
        return Scalar::POISON;
    }
    if a.undef != 0 || b.undef != 0 {
        return spread_choice(Scalar::undef(1), 1, &[a, b]);
    }
    let (x, y) = (a.bits as u64, b.bits as u64);
    let nnan = fmf.0 & FastMath::NNAN != 0;
    let ninf = fmf.0 & FastMath::NINF != 0;
    if [x, y].iter().any(|&v| (nnan && fmt.is_nan(v)) || (ninf && fmt.is_inf(v))) {
        return Scalar::POISON;
    }
    use FloatPredicate as P;
    use std::cmp::Ordering::*;
    let ord = float::compare(fmt, x, y);
    let v = match pred {
        P::False => false,
        P::True => true,
        P::Oeq => ord == Some(Equal),
        P::Ogt => ord == Some(Greater),
        P::Oge => matches!(ord, Some(Greater | Equal)),
        P::Olt => ord == Some(Less),
        P::Ole => matches!(ord, Some(Less | Equal)),
        P::One => matches!(ord, Some(Less | Greater)),
        P::Ord => ord.is_some(),
        P::Uno => ord.is_none(),
        P::Ueq => matches!(ord, None | Some(Equal)),
        P::Ugt => matches!(ord, None | Some(Greater)),
        P::Uge => matches!(ord, None | Some(Greater | Equal)),
        P::Ult => matches!(ord, None | Some(Less)),
        P::Ule => matches!(ord, None | Some(Less | Equal)),
        P::Une => ord != Some(Equal),
    };
    spread_choice(Scalar::bool(v), 1, &[a, b])
}

// Conversions ---------------------------------------------------------------------------------------------------------

/// A cast of one element from `from` to `to`. `ptr_to_int` and `int_to_ptr` follow LangRef: the address is
/// truncated or zero-extended; `inttoptr` gives a pointer without provenance. `bitcast` between scalars of equal
/// width keeps the bits (vector bitcasts are the caller's).
pub fn cast(op: CastOp, fl: u8, fmf: FastMath, from: Prim, to: Prim, a: &Scalar) -> Scalar {
    if a.poison {
        return Scalar::POISON;
    }
    let fw = from.width();
    let tw = to.width();
    let tm = mask(tw);

    match (op, from, to) {
        (CastOp::Trunc, Prim::Int(_), Prim::Int(_)) => {
            let dropped = mask(fw) & !tm;
            let nuw = fl & flags::NUW != 0;
            let nsw = fl & flags::NSW != 0;
            if nuw && ((a.bits | a.undef) & dropped != 0) {
                return Scalar::POISON;
            }
            if nsw {
                // The dropped bits and the new sign bit must all be known and equal.
                let check = dropped | sign_bit(tw);
                if a.undef & check != 0 || sext(a.bits & tm, tw) != sext(a.bits & mask(fw), fw) {
                    return Scalar::POISON;
                }
            }
            Scalar { bits: a.bits & tm, undef: a.undef & tm, choice: a.choice & tm, prov: 0, poison: false }
        }
        (CastOp::ZExt, Prim::Int(_), Prim::Int(_)) => {
            if fl & flags::NNEG != 0 && (a.bits | a.undef) & sign_bit(fw) != 0 {
                return Scalar::POISON;
            }
            Scalar { prov: 0, ..*a }
        }
        (CastOp::SExt, Prim::Int(_), Prim::Int(_)) => {
            let ext = tm & !mask(fw);
            let sb = sign_bit(fw);
            Scalar {
                bits: a.bits | if a.bits & sb != 0 { ext } else { 0 },
                undef: a.undef | if a.undef & sb != 0 { ext } else { 0 },
                choice: a.choice | if a.choice & sb != 0 { ext } else { 0 },
                prov: 0,
                poison: false,
            }
        }
        (CastOp::FPToUI | CastOp::FPToSI, Prim::Float(f), Prim::Int(_)) => {
            if a.undef != 0 {
                return spread_choice(all_undef(tw), tw, &[a]);
            }
            match float_to_int(f, a.bits as u64, tw, op == CastOp::FPToSI) {
                Some(v) => spread_choice(Scalar::int(v), tw, &[a]),
                // LangRef: a value that does not fit (NaN and infinities included) gives poison.
                None => Scalar::POISON,
            }
        }
        (CastOp::UIToFP | CastOp::SIToFP, Prim::Int(_), Prim::Float(f)) => {
            if op == CastOp::UIToFP && fl & flags::NNEG != 0 && (a.bits | a.undef) & sign_bit(fw) != 0 {
                return Scalar::POISON;
            }
            if a.undef != 0 {
                return spread_choice(all_undef(tw), tw, &[a]);
            }
            let (neg, mag) = if op == CastOp::SIToFP {
                let v = sext(a.bits & mask(fw), fw);
                (v < 0, v.unsigned_abs())
            } else {
                (false, a.bits & mask(fw))
            };
            let r = float::from_int(f, neg, mag);
            if fmf_poison(f, fmf, &[], r) {
                return Scalar::POISON;
            }
            spread_choice(Scalar::int(r as u128), tw, &[a])
        }
        (CastOp::FPTrunc | CastOp::FPExt, Prim::Float(ff), Prim::Float(tf)) => {
            if a.undef != 0 {
                return spread_choice(all_undef(tw), tw, &[a]);
            }
            let x = a.bits as u64;
            if fmf_poison(ff, fmf, &[x], 0) {
                return Scalar::POISON;
            }
            if ff.is_nan(x) {
                // The NaN rules, with the operand's payload carried to the result's fraction.
                let payload = if tf.frac_bits >= ff.frac_bits {
                    (x & ff.frac_mask() & !ff.quiet_bit()) << (tf.frac_bits - ff.frac_bits)
                } else {
                    (x & ff.frac_mask() & !ff.quiet_bit()) >> (ff.frac_bits - tf.frac_bits)
                };
                let mut n = nan_result(tf, &[]);
                n.choice |= payload as u128;
                if ff.is_snan(x) {
                    n.choice |= tf.quiet_bit() as u128;
                }
                return spread_choice_keep(n, tw, &[a]);
            }
            let r = tf.from_f64(ff.to_f64(x));
            if fmf_poison(tf, fmf, &[], r) {
                return Scalar::POISON;
            }
            let mut out = Scalar::int(r as u128);
            if fmf.0 & FastMath::NSZ != 0 && tf.is_zero(r) {
                out.choice |= tf.sign_bit() as u128;
            }
            spread_choice_keep(out, tw, &[a])
        }
        (CastOp::PtrToInt, Prim::Ptr(..), Prim::Int(_)) => {
            Scalar { bits: a.bits & tm, undef: a.undef & tm, choice: a.choice & tm, prov: 0, poison: false }
        }
        (CastOp::IntToPtr, Prim::Int(_), Prim::Ptr(..)) => {
            Scalar { bits: a.bits & tm, undef: a.undef & tm, choice: a.choice & tm, prov: 0, poison: false }
        }
        (CastOp::BitCast | CastOp::AddrSpaceCast, _, _) => {
            let mut r = *a;
            if !matches!(to, Prim::Ptr(..)) {
                r.prov = 0;
            }
            r
        }
        _ => unreachable!("a cast between kinds the verifier rejects"),
    }
}

/// Truncation toward zero of a floating-point value to a `w`-bit integer, or `None` when it does not fit.
pub fn float_to_int(f: Format, x: u64, w: u32, signed: bool) -> Option<u128> {
    if f.is_nan(x) || f.is_inf(x) {
        return None;
    }
    let t = f.to_f64(x).trunc();
    // Every finite value of the formats here is exact in binary64, and so is its truncation.
    let (neg, mag) = match float::Format::decode(float::DOUBLE, t.to_bits()) {
        float::Class::Finite { neg, mant, exp } => {
            if mant == 0 {
                (false, 0u128)
            } else if exp >= 0 {
                if exp >= 75 {
                    return None;
                }
                (neg, (mant as u128) << exp)
            } else {
                (neg, (mant as u128) >> (-exp))
            }
        }
        _ => return None,
    };
    if signed {
        let lim = 1u128 << (w - 1);
        if (!neg && mag >= lim) || (neg && mag > lim) {
            return None;
        }
        let v = if neg { (mag as i128).wrapping_neg() as u128 } else { mag };
        Some(v & mask(w))
    } else {
        if (neg && mag != 0) || (w < 128 && mag >> w != 0) {
            return None;
        }
        Some(mag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: u128) -> Scalar {
        Scalar::int(v)
    }

    #[test]
    fn undef_bits_follow_langref_examples() {
        let u = Scalar::undef(32);
        let x = s(0x00ff_00f0);
        // and with undef keeps the known-zero bits.
        let r = int_binop(BinOp::And, 0, 32, &x, &u).unwrap();
        assert_eq!(r.undef, 0x00ff_00f0);
        assert_eq!(r.bits, 0);
        // or with undef keeps the known-one bits.
        let r = int_binop(BinOp::Or, 0, 32, &x, &u).unwrap();
        assert_eq!(r.undef, !0x00ff_00f0u32 as u128);
        assert_eq!(r.bits, 0x00ff_00f0);
        // add, sub and xor with undef are undef.
        assert_eq!(int_binop(BinOp::Add, 0, 32, &x, &u).unwrap().undef, mask(32));
        assert_eq!(int_binop(BinOp::Xor, 0, 32, &u, &u).unwrap().undef, mask(32));
        // sdiv by undef is undefined behavior; sdiv of undef is a value.
        assert!(int_binop(BinOp::SDiv, 0, 32, &x, &u).is_err());
        assert!(int_binop(BinOp::SDiv, 0, 32, &u, &x).is_ok());
        // or i8 undef, 255 is 255.
        let r = int_binop(BinOp::Or, 0, 8, &Scalar::undef(8), &s(255)).unwrap();
        assert!(r.is_defined());
        assert_eq!(r.bits, 255);
    }

    #[test]
    fn poison_rules() {
        assert!(int_binop(BinOp::Sub, flags::NUW, 32, &s(0), &s(1)).unwrap().poison);
        assert!(int_binop(BinOp::Add, flags::NSW, 8, &s(127), &s(1)).unwrap().poison);
        assert!(!int_binop(BinOp::Add, flags::NUW, 8, &s(127), &s(1)).unwrap().poison);
        assert!(int_binop(BinOp::Shl, 0, 32, &s(1), &s(32)).unwrap().poison);
        assert!(int_binop(BinOp::LShr, flags::EXACT, 32, &s(3), &s(1)).unwrap().poison);
        assert!(int_binop(BinOp::Or, flags::DISJOINT, 32, &s(3), &s(1)).unwrap().poison);
        assert!(int_binop(BinOp::And, 0, 32, &Scalar::POISON, &s(0)).unwrap().poison);
        assert!(int_binop(BinOp::SDiv, 0, 8, &s(0x80), &s(0xff)).is_err());
        assert_eq!(int_binop(BinOp::SRem, 0, 8, &s(0xf9), &s(2)).unwrap().bits, 0xff);
        assert!(int_binop(BinOp::Shl, flags::NSW, 8, &s(0x40), &s(1)).unwrap().poison);
        assert!(!int_binop(BinOp::Shl, flags::NSW, 8, &s(0xc0), &s(1)).unwrap().poison);
        assert!(int_binop(BinOp::Mul, flags::NUW, 8, &s(16), &s(16)).unwrap().poison);
        assert!(int_binop(BinOp::Mul, flags::NSW, 8, &s(0xff), &s(0x80)).unwrap().poison);
        assert!(!int_binop(BinOp::Mul, flags::NSW, 8, &s(0xff), &s(0x7f)).unwrap().poison);
    }

    #[test]
    fn comparisons_with_undef_decide_when_they_can() {
        // (undef & 3) < 4 is true for every value.
        let lowbits = int_binop(BinOp::And, 0, 32, &Scalar::undef(32), &s(3)).unwrap();
        let r = icmp(IntPredicate::Ult, false, 32, &lowbits, &s(4));
        assert!(r.is_defined());
        assert_eq!(r.bits, 1);
        assert!(!icmp(IntPredicate::Slt, false, 32, &Scalar::undef(32), &s(4)).is_defined());
        assert!(icmp(IntPredicate::Ult, true, 32, &s(0x8000_0000), &s(1)).poison);
    }

    #[test]
    fn float_nan_and_flags() {
        let f = float::SINGLE;
        let one = s(1.0f32.to_bits() as u128);
        let zero = s(0);
        let inf = s(f32::INFINITY.to_bits() as u128);
        // 0/0 is the preferred NaN with the sign a choice.
        let r = float_binop(BinOp::FDiv, FastMath(0), f, &zero, &zero);
        assert_eq!(r.bits as u64, f.preferred_nan());
        assert_eq!(r.choice, f.sign_bit() as u128);
        assert!(float_binop(BinOp::FDiv, FastMath(FastMath::NNAN), f, &zero, &zero).poison);
        assert!(float_binop(BinOp::FAdd, FastMath(FastMath::NINF), f, &inf, &one).poison);
        // 1 / -0 under nsz may be either infinity.
        let nz = s(0x8000_0000);
        let r = float_binop(BinOp::FDiv, FastMath(FastMath::NSZ), f, &one, &nz);
        assert_eq!(r.choice, f.sign_bit() as u128);
        assert!(f.is_inf(r.bits as u64));
    }

    #[test]
    fn conversions() {
        let f = float::SINGLE;
        assert_eq!(float_to_int(f, (-1.5f32).to_bits() as u64, 32, true), Some((-1i128) as u128 & mask(32)));
        assert_eq!(float_to_int(f, (-1.5f32).to_bits() as u64, 32, false), None);
        assert_eq!(float_to_int(f, (-0.5f32).to_bits() as u64, 32, false), Some(0));
        assert_eq!(float_to_int(f, 2147483648.0f32.to_bits() as u64, 32, true), None);
        assert_eq!(float_to_int(f, (-2147483648.0f32).to_bits() as u64, 32, true), Some(0x8000_0000));
        let r = cast(CastOp::SIToFP, 0, FastMath(0), Prim::Int(32), Prim::Float(f), &s(0xffff_ffff));
        assert_eq!(r.bits as u32, (-1.0f32).to_bits());
        let t = cast(CastOp::Trunc, flags::NSW, FastMath(0), Prim::Int(32), Prim::Int(8), &s(0x80));
        assert!(t.poison);
        let t = cast(CastOp::Trunc, flags::NSW, FastMath(0), Prim::Int(32), Prim::Int(8), &s(0xffff_ff80));
        assert_eq!(t.bits, 0x80);
    }
}
