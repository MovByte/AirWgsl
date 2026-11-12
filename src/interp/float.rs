//! IEEE-754 binary formats on raw bits: decoding, correctly rounded results (round to nearest, ties to even) and the
//! core operations of LangRef's "Floating-Point Semantics" for `half`, `bfloat`, `float` and `double`.
//!
//! `double` arithmetic is the host's, which is IEEE binary64 with round to nearest even and subnormals preserved.
//! The narrower formats compute in binary64 and round once more: for addition, subtraction, multiplication,
//! division and square root that second rounding is innocuous because binary64 carries at least `2p + 2` bits for
//! each of them (p = 24, 11 and 8; Figueroa, "When is double rounding innocuous?", SIGNUM 1995), so the result is the
//! correctly rounded result of the narrow format. Fused multiply-add rounds the exact `a * b + c` once: the product
//! of two narrow values is exact in binary64, and the sum is split into its binary64 rounding and the exact error
//! (Knuth's TwoSum), whose sign settles the ties.

use crate::ir::FloatKind;

/// A binary interchange format: exponent and fraction field widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    pub exp_bits: u32,
    pub frac_bits: u32,
}

pub const HALF: Format = Format { exp_bits: 5, frac_bits: 10 };
pub const BFLOAT: Format = Format { exp_bits: 8, frac_bits: 7 };
pub const SINGLE: Format = Format { exp_bits: 8, frac_bits: 23 };
pub const DOUBLE: Format = Format { exp_bits: 11, frac_bits: 52 };

impl Format {
    /// The format of a floating-point kind, for the four kinds the interpreter evaluates.
    pub fn of(kind: FloatKind) -> Option<Format> {
        match kind {
            FloatKind::Half => Some(HALF),
            FloatKind::BFloat => Some(BFLOAT),
            FloatKind::Float => Some(SINGLE),
            FloatKind::Double => Some(DOUBLE),
            _ => None,
        }
    }

    pub fn width(self) -> u32 {
        1 + self.exp_bits + self.frac_bits
    }

    pub fn bias(self) -> i32 {
        (1 << (self.exp_bits - 1)) - 1
    }

    pub fn sign_bit(self) -> u64 {
        1 << (self.exp_bits + self.frac_bits)
    }

    pub fn exp_mask(self) -> u64 {
        ((1u64 << self.exp_bits) - 1) << self.frac_bits
    }

    pub fn frac_mask(self) -> u64 {
        (1u64 << self.frac_bits) - 1
    }

    /// The quiet bit (the fraction's top bit).
    pub fn quiet_bit(self) -> u64 {
        1 << (self.frac_bits - 1)
    }

    /// Every bit of the encoding.
    pub fn mask(self) -> u64 {
        if self.width() == 64 { u64::MAX } else { (1u64 << self.width()) - 1 }
    }

    pub fn infinity(self, neg: bool) -> u64 {
        self.exp_mask() | if neg { self.sign_bit() } else { 0 }
    }

    /// The preferred NaN: quiet bit set, payload zero (LangRef "Behavior of Floating-Point NaN values").
    pub fn preferred_nan(self) -> u64 {
        self.exp_mask() | self.quiet_bit()
    }

    pub fn is_nan(self, bits: u64) -> bool {
        bits & self.exp_mask() == self.exp_mask() && bits & self.frac_mask() != 0
    }

    pub fn is_snan(self, bits: u64) -> bool {
        self.is_nan(bits) && bits & self.quiet_bit() == 0
    }

    pub fn is_inf(self, bits: u64) -> bool {
        bits & !self.sign_bit() & self.mask() == self.exp_mask()
    }

    pub fn is_zero(self, bits: u64) -> bool {
        bits & !self.sign_bit() & self.mask() == 0
    }

    pub fn is_neg(self, bits: u64) -> bool {
        bits & self.sign_bit() != 0
    }

    pub fn decode(self, bits: u64) -> Class {
        let neg = self.is_neg(bits);
        let e = ((bits & self.exp_mask()) >> self.frac_bits) as i32;
        let f = bits & self.frac_mask();
        if e == (1 << self.exp_bits) - 1 {
            return if f == 0 { Class::Inf { neg } } else { Class::Nan { neg, quiet: f & self.quiet_bit() != 0 } };
        }
        let emin = 1 - self.bias();
        if e == 0 {
            // Zero or subnormal: f * 2^(emin - frac_bits).
            Class::Finite { neg, mant: f, exp: emin - self.frac_bits as i32 }
        } else {
            Class::Finite { neg, mant: f | (1u64 << self.frac_bits), exp: e - self.bias() - self.frac_bits as i32 }
        }
    }

    /// The value of a finite encoding as a binary64 (exact for every format here).
    pub fn to_f64(self, bits: u64) -> f64 {
        if self == DOUBLE {
            return f64::from_bits(bits);
        }
        match self.decode(bits) {
            Class::Nan { neg, .. } => {
                // The payload moves to the top of the binary64 fraction, as a widening conversion does.
                let payload = (bits & self.frac_mask()) << (52 - self.frac_bits);
                f64::from_bits(DOUBLE.exp_mask() | payload | if neg { DOUBLE.sign_bit() } else { 0 })
            }
            Class::Inf { neg } => {
                if neg {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            }
            Class::Finite { neg, mant, exp } => {
                let v = mant as f64 * 2f64.powi(exp);
                if neg { -v } else { v }
            }
        }
    }

    /// Rounds the finite binary64 `x` to this format (round to nearest, ties to even). NaN and infinity map to the
    /// format's NaN (payload truncated, quiet bit kept) and infinity.
    pub fn from_f64(self, x: f64) -> u64 {
        self.from_f64_delta(x, 0)
    }

    /// Rounds `x + d` where `d` is a value of magnitude below half a unit in the last place of `x` with sign
    /// `delta` (-1, 0 or 1): the rounding of an exact sum that binary64 rounded to `x`.
    pub fn from_f64_delta(self, x: f64, delta: i8) -> u64 {
        if self == DOUBLE {
            return x.to_bits();
        }
        let b = x.to_bits();
        match DOUBLE.decode(b) {
            Class::Nan { neg, .. } => {
                let payload = (b & DOUBLE.frac_mask()) >> (52 - self.frac_bits);
                self.exp_mask() | payload | if neg { self.sign_bit() } else { 0 }
            }
            Class::Inf { neg } => self.infinity(neg),
            Class::Finite { neg, mant, exp } => {
                // The delta's sign is relative to the value; make it relative to the magnitude.
                let d = if neg { -delta } else { delta };
                round(self, neg, mant as u128, exp, d)
            }
        }
    }
}

/// A decoded encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Nan {
        neg: bool,
        quiet: bool,
    },
    Inf {
        neg: bool,
    },
    /// The value `(-1)^neg * mant * 2^exp`; zero has `mant` 0.
    Finite {
        neg: bool,
        mant: u64,
        exp: i32,
    },
}

/// The encoding nearest to `(-1)^neg * (mant + d) * 2^exp` (ties to even), where `d` is 0 or a magnitude below half
/// a unit of `mant`'s last place whose sign is `delta`. A nonzero `delta` needs `mant` to carry at least two more
/// bits than the format keeps at that exponent, which every caller's binary64 value does for the narrower formats.
pub fn round(fmt: Format, neg: bool, mant: u128, exp: i32, delta: i8) -> u64 {
    let sign = if neg { fmt.sign_bit() } else { 0 };
    if mant == 0 {
        return sign;
    }
    let msb = 127 - mant.leading_zeros() as i32;
    let lead = exp + msb;
    let emin = 1 - fmt.bias();
    let frac = fmt.frac_bits as i32;
    let mut lsb = (lead - frac).max(emin - frac);
    let shift = lsb - exp;
    let mut q: u128 = if shift <= 0 {
        debug_assert!(delta == 0, "an inexact value without guard bits");
        mant << (-shift) as u32
    } else if shift > 128 {
        // Below half the smallest subnormal: rounds to zero (a delta cannot reach the halfway point).
        0
    } else {
        let s = shift as u32;
        let q = if s == 128 { 0 } else { mant >> s };
        let rem = if s == 128 { mant } else { mant & ((1u128 << s) - 1) };
        let half = 1u128 << (s - 1);
        let up = rem > half || (rem == half && (delta > 0 || (delta == 0 && q & 1 == 1)));
        if up { q + 1 } else { q }
    };
    if q >> (fmt.frac_bits + 1) != 0 {
        q >>= 1;
        lsb += 1;
    }
    if q >> fmt.frac_bits == 0 {
        // Subnormal or zero.
        return sign | q as u64;
    }
    let biased = lsb + frac + fmt.bias();
    if biased >= (1 << fmt.exp_bits) - 1 {
        return sign | fmt.exp_mask();
    }
    sign | ((biased as u64) << fmt.frac_bits) | (q as u64 & fmt.frac_mask())
}

/// The encoding nearest to the integer `(-1)^neg * mag`.
pub fn from_int(fmt: Format, neg: bool, mag: u128) -> u64 {
    round(fmt, neg, mag, 0, 0)
}

/// Splits an exact binary64 sum `a + b` into its rounding `s` and the sign of the error `(a + b) - s`
/// (Knuth's TwoSum; exact unless the sum overflows).
pub fn two_sum(a: f64, b: f64) -> (f64, i8) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    let d = if err > 0.0 {
        1
    } else if err < 0.0 {
        -1
    } else {
        0
    };
    (s, d)
}

/// Result of a binary operation in IEEE semantics, without the NaN rules (callers apply those): the encoding of the
/// correctly rounded result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// `a op b` in `fmt`, both finite or infinite or NaN; NaN results come back as some NaN (callers replace them).
pub fn binary(fmt: Format, op: Arith, a: u64, b: u64) -> u64 {
    let x = fmt.to_f64(a);
    let y = fmt.to_f64(b);
    let r = match op {
        Arith::Add => x + y,
        Arith::Sub => x - y,
        Arith::Mul => x * y,
        Arith::Div => x / y,
        // fmod is exact, so the binary64 remainder of narrow operands is representable in the narrow format.
        Arith::Rem => x % y,
    };
    fmt.from_f64(r)
}

/// `sqrt(a)` correctly rounded.
pub fn sqrt(fmt: Format, a: u64) -> u64 {
    fmt.from_f64(fmt.to_f64(a).sqrt())
}

/// `a * b + c` with one rounding.
pub fn fma(fmt: Format, a: u64, b: u64, c: u64) -> u64 {
    let (x, y, z) = (fmt.to_f64(a), fmt.to_f64(b), fmt.to_f64(c));
    if fmt == DOUBLE {
        return x.mul_add(y, z).to_bits();
    }
    let p = x * y;
    if !p.is_finite() || !z.is_finite() {
        return fmt.from_f64(p + z);
    }
    let (s, d) = two_sum(p, z);
    if s == 0.0 {
        // An exact zero (binary64 sums are exact when they cancel); its sign follows IEEE's rules for the sum.
        return fmt.from_f64(s);
    }
    fmt.from_f64_delta(s, d)
}

/// IEEE `compareQuiet`: `None` when unordered.
pub fn compare(fmt: Format, a: u64, b: u64) -> Option<std::cmp::Ordering> {
    if fmt.is_nan(a) || fmt.is_nan(b) {
        return None;
    }
    fmt.to_f64(a).partial_cmp(&fmt.to_f64(b))
}

/// Round to an integral value in the given mode, exact (no rounding error is possible): IEEE `roundToIntegral*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integral {
    /// Ties to even (`rint`, `nearbyint`, `roundeven`).
    TiesEven,
    /// Ties away from zero (`round`).
    TiesAway,
    /// Toward zero (`trunc`).
    Zero,
    /// Toward negative infinity (`floor`).
    Down,
    /// Toward positive infinity (`ceil`).
    Up,
}

pub fn round_integral(fmt: Format, a: u64, mode: Integral) -> u64 {
    if fmt.is_nan(a) || fmt.is_inf(a) || fmt.is_zero(a) {
        return a;
    }
    let x = fmt.to_f64(a);
    let r = match mode {
        Integral::TiesEven => {
            let t = x.trunc();
            let diff = (x - t).abs();
            if diff > 0.5 || (diff == 0.5 && t % 2.0 != 0.0) { t + x.signum() } else { t }
        }
        Integral::TiesAway => x.round(),
        Integral::Zero => x.trunc(),
        Integral::Down => x.floor(),
        Integral::Up => x.ceil(),
    };
    // A zero result keeps the operand's sign (IEEE 5.9: roundToIntegral of a negative value in (-1, 0) is -0).
    let r = if r == 0.0 { 0.0f64.copysign(x) } else { r };
    fmt.from_f64(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }

    #[test]
    fn single_rounding_matches_the_host() {
        let mut s = 1;
        for _ in 0..200_000 {
            let x = f64::from_bits(lcg(&mut s));
            if x.is_nan() {
                continue;
            }
            assert_eq!(SINGLE.from_f64(x), (x as f32).to_bits() as u64, "{:e}", x);
        }
        for e in -160..-120 {
            for k in 0..64 {
                let x = (k as f64 + 0.5) * 2f64.powi(e);
                assert_eq!(SINGLE.from_f64(x), (x as f32).to_bits() as u64, "{:e}", x);
            }
        }
    }

    #[test]
    fn single_arithmetic_matches_the_host() {
        let mut s = 7;
        for _ in 0..200_000 {
            let a = lcg(&mut s) as u32;
            let b = (lcg(&mut s) >> 17) as u32;
            let c = (lcg(&mut s) >> 9) as u32;
            let (x, y, z) = (f32::from_bits(a), f32::from_bits(b), f32::from_bits(c));
            let ok = |r: u64, h: f32| {
                let h = h.to_bits() as u64;
                r == h || (SINGLE.is_nan(r) && SINGLE.is_nan(h))
            };
            assert!(ok(binary(SINGLE, Arith::Add, a as u64, b as u64), x + y));
            assert!(ok(binary(SINGLE, Arith::Sub, a as u64, b as u64), x - y));
            assert!(ok(binary(SINGLE, Arith::Mul, a as u64, b as u64), x * y));
            assert!(ok(binary(SINGLE, Arith::Div, a as u64, b as u64), x / y));
            assert!(ok(binary(SINGLE, Arith::Rem, a as u64, b as u64), x % y));
            assert!(ok(sqrt(SINGLE, a as u64), x.sqrt()));
            assert!(ok(fma(SINGLE, a as u64, b as u64, c as u64), x.mul_add(y, z)), "{:e} {:e} {:e}", x, y, z);
        }
    }

    #[test]
    fn half_round_trips_and_ties_to_even() {
        for h in 0u64..0x10000 {
            if HALF.is_nan(h) {
                continue;
            }
            assert_eq!(HALF.from_f64(HALF.to_f64(h)), h);
        }
        // 1 + 2^-11 is halfway between 1 and 1 + 2^-10: even is 1.
        assert_eq!(HALF.from_f64(1.0 + 2f64.powi(-11)), 0x3c00);
        assert_eq!(HALF.from_f64(1.0 + 3.0 * 2f64.powi(-11)), 0x3c02);
        // The largest finite half is 65504; 65520 is the overflow threshold.
        assert_eq!(HALF.from_f64(65519.0), 0x7bff);
        assert_eq!(HALF.from_f64(65520.0), 0x7c00);
        // Half the smallest subnormal rounds to zero, a bit more rounds up.
        assert_eq!(HALF.from_f64(2f64.powi(-25)), 0);
        assert_eq!(HALF.from_f64(2f64.powi(-25) * 1.0001), 1);
        assert_eq!(HALF.from_f64(-2f64.powi(-25)), 0x8000);
    }

    #[test]
    fn integers_round_once() {
        // 2^24 + 1 is a tie in binary32 and goes to even; 2^53 + 1 likewise in binary64.
        assert_eq!(from_int(SINGLE, false, (1 << 24) + 1), 0x4b80_0000);
        assert_eq!(from_int(SINGLE, false, (1 << 24) + 3), 0x4b80_0002);
        assert_eq!(from_int(DOUBLE, false, (1 << 53) + 1), (9007199254740992.0f64).to_bits());
        let mut s = 3;
        for _ in 0..100_000 {
            let v = lcg(&mut s);
            assert_eq!(from_int(SINGLE, false, v as u128), (v as f32).to_bits() as u64);
            assert_eq!(from_int(DOUBLE, false, v as u128), (v as f64).to_bits());
            let w = v >> (v % 64);
            assert_eq!(from_int(SINGLE, false, w as u128), (w as f32).to_bits() as u64);
        }
        // 0x1_0000_0000_0000_0801 needs the sticky bit to round up in binary64.
        let v: u128 = (1 << 64) | 0x801;
        assert_eq!(from_int(DOUBLE, false, v), (v as f64).to_bits());
    }

    #[test]
    fn half_fma_is_singly_rounded() {
        // Exhaustive over a slice: compare with an exact rational computation.
        let mut s = 11;
        for _ in 0..50_000 {
            let a = lcg(&mut s) & 0xffff;
            let b = lcg(&mut s) & 0xffff;
            let c = lcg(&mut s) & 0xffff;
            if [a, b, c].iter().any(|&v| HALF.is_nan(v) || HALF.is_inf(v)) {
                continue;
            }
            let r = fma(HALF, a, b, c);
            // Exact: a*b + c as a scaled integer (exponents of half are at least -24).
            let dec = |v: u64| match HALF.decode(v) {
                Class::Finite { neg, mant, exp } => (if neg { -(mant as i128) } else { mant as i128 }, exp),
                _ => unreachable!(),
            };
            let (ma, ea) = dec(a);
            let (mb, eb) = dec(b);
            let (mc, ec) = dec(c);
            let e = (ea + eb).min(ec);
            let exact = (ma * mb) << (ea + eb - e) as u32;
            let exact = exact + (mc << (ec - e) as u32);
            let want = round(HALF, exact < 0, exact.unsigned_abs(), e, 0);
            if exact == 0 {
                continue;
            }
            assert_eq!(r, want, "{:x} {:x} {:x}", a, b, c);
        }
    }
}
