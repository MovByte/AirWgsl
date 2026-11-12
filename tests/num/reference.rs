//! CPU references, implemented independently of the WGSL library.
//!
//! binary32 arithmetic, rounding to integral values and conversions are the host CPU's (AArch64, FPCR at its reset
//! value: round to nearest even, no flush to zero, NaN propagation rather than default NaN). binary16 arithmetic uses
//! the CPU's half-precision instructions (FEAT_FP16). Functions the CPU does not provide are written here from their
//! definitions: correctly rounded rsqrt by exact integer comparison, and the MSL selection functions from the text of
//! MSL Table 6.4.

#[cfg(not(target_arch = "aarch64"))]
compile_error!("the numerics references use AArch64 floating-point instructions");

use core::arch::asm;

/// The floating-point control register. The references assume 0: round to nearest even, no flush to zero, no
/// default-NaN mode, no alternate half-precision format.
pub fn fpcr() -> u64 {
    let v: u64;
    // SAFETY: reading FPCR has no side effects.
    unsafe { asm!("mrs {}, fpcr", out(reg) v) };
    v
}

pub fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

// binary32 arithmetic as explicit instructions: the compiler may commute an operator, which changes which NaN operand
// the result carries.
macro_rules! s_binary {
    ($name:ident, $insn:literal) => {
        pub fn $name(a: u32, b: u32) -> u32 {
            let r: u32;
            // SAFETY: base AArch64 floating-point instructions on the named registers only.
            unsafe {
                asm!(
                    "fmov {x:s}, {a:w}",
                    "fmov {y:s}, {b:w}",
                    concat!($insn, " {x:s}, {x:s}, {y:s}"),
                    "fmov {r:w}, {x:s}",
                    a = in(reg) a,
                    b = in(reg) b,
                    r = out(reg) r,
                    x = out(vreg) _,
                    y = out(vreg) _,
                )
            };
            r
        }
    };
}

s_binary!(add, "fadd");
s_binary!(sub, "fsub");
s_binary!(mul, "fmul");
s_binary!(div, "fdiv");

pub fn sqrt(a: u32) -> u32 {
    let r: u32;
    // SAFETY: as above.
    unsafe {
        asm!(
            "fmov {x:s}, {a:w}",
            "fsqrt {x:s}, {x:s}",
            "fmov {r:w}, {x:s}",
            a = in(reg) a,
            r = out(reg) r,
            x = out(vreg) _,
        )
    };
    r
}

/// fma(a, b, c) = a * b + c rounded once (FMADD Sd, Sn=a, Sm=b, Sa=c).
pub fn fma(a: u32, b: u32, c: u32) -> u32 {
    let r: u32;
    // SAFETY: as above.
    unsafe {
        asm!(
            "fmov {x:s}, {a:w}",
            "fmov {y:s}, {b:w}",
            "fmov {z:s}, {c:w}",
            "fmadd {x:s}, {x:s}, {y:s}, {z:s}",
            "fmov {r:w}, {x:s}",
            a = in(reg) a,
            b = in(reg) b,
            c = in(reg) c,
            r = out(reg) r,
            x = out(vreg) _,
            y = out(vreg) _,
            z = out(vreg) _,
        )
    };
    r
}

pub fn quiet(a: u32) -> u32 {
    if f(a).is_nan() { a | 0x0040_0000 } else { a }
}

/// Exact sign of (m * 2^e)^2 * x - 1 for positive m, x = xm * 2^xe: -1, 0 or 1.
fn cmp_square_times(m: u64, e: i32, xm: u64, xe: i32) -> i32 {
    // (m^2 * xm) * 2^(2e + xe) versus 1.
    let p = (m as u128) * (m as u128) * (xm as u128);
    let k = 2 * e + xe;
    if p == 0 {
        return -1;
    }
    let bits = 128 - p.leading_zeros() as i32;
    // p * 2^k has its leading one at bit (bits - 1 + k).
    if bits - 1 + k > 0 {
        return 1;
    }
    if bits - 1 + k < 0 {
        return -1;
    }
    // p * 2^k is in [1, 2): it equals 1 exactly when p is a power of two.
    if p.is_power_of_two() { 0 } else { 1 }
}

/// Decompose a positive finite binary32 value as m * 2^e with integer m.
fn decompose(a: u32) -> (u64, i32) {
    let e = (a >> 23) & 0xff;
    let frac = (a & 0x7f_ffff) as u64;
    if e == 0 { (frac, -149) } else { (frac | 0x80_0000, e as i32 - 150) }
}

/// 1 / sqrt(a), correctly rounded, with the IEEE 754-2019 rSqrt special cases.
pub fn rsqrt(a: u32) -> u32 {
    let x = f(a);
    if x.is_nan() {
        return quiet(a);
    }
    if a & 0x7fff_ffff == 0 {
        return a | 0x7f80_0000;
    }
    if x < 0.0 {
        return 0x7fc0_0000;
    }
    if x.is_infinite() {
        return 0;
    }
    let (xm, xe) = decompose(a);
    // A candidate within a few ulps, then exact midpoint tests.
    let approx = (1.0 / (x as f64).sqrt()) as f32;
    let mut c = approx.to_bits();
    // The midpoint between c and its successor: (2 * cm + 1) * 2^(ce - 1).
    let above = |c: u32| {
        let (cm, ce) = decompose(c);
        cmp_square_times(2 * cm + 1, ce - 1, xm, xe)
    };
    // Move down while the value lies below the midpoint under c, up while it lies above the one over c.
    loop {
        if c > 1 && above(c - 1) > 0 {
            c -= 1;
            continue;
        }
        if above(c) < 0 {
            c += 1;
            continue;
        }
        break;
    }
    c
}

// binary16 arithmetic on the CPU (FEAT_FP16 instructions).
macro_rules! h_binary {
    ($name:ident, $insn:literal) => {
        pub fn $name(a: u16, b: u16) -> u16 {
            let r: u32;
            // SAFETY: FEAT_FP16 is part of the AArch64 baseline of every Apple silicon CPU; the instructions only
            // touch the named registers.
            unsafe {
                asm!(
                    ".arch_extension fp16",
                    "fmov {x:h}, {a:w}",
                    "fmov {y:h}, {b:w}",
                    concat!($insn, " {x:h}, {x:h}, {y:h}"),
                    "fmov {r:w}, {x:h}",
                    a = in(reg) a as u32,
                    b = in(reg) b as u32,
                    r = out(reg) r,
                    x = out(vreg) _,
                    y = out(vreg) _,
                )
            };
            r as u16
        }
    };
}

h_binary!(h_add, "fadd");
h_binary!(h_sub, "fsub");
h_binary!(h_mul, "fmul");
h_binary!(h_div, "fdiv");

pub fn h_sqrt(a: u16) -> u16 {
    let r: u32;
    // SAFETY: as for the binary operations.
    unsafe {
        asm!(
            ".arch_extension fp16",
            "fmov {x:h}, {a:w}",
            "fsqrt {x:h}, {x:h}",
            "fmov {r:w}, {x:h}",
            a = in(reg) a as u32,
            r = out(reg) r,
            x = out(vreg) _,
        )
    };
    r as u16
}

/// a * b + c rounded once (FMADD Hd, Hn=a, Hm=b, Ha=c).
pub fn h_fma(a: u16, b: u16, c: u16) -> u16 {
    let r: u32;
    // SAFETY: as for the binary operations.
    unsafe {
        asm!(
            ".arch_extension fp16",
            "fmov {x:h}, {a:w}",
            "fmov {y:h}, {b:w}",
            "fmov {z:h}, {c:w}",
            "fmadd {x:h}, {x:h}, {y:h}, {z:h}",
            "fmov {r:w}, {x:h}",
            a = in(reg) a as u32,
            b = in(reg) b as u32,
            c = in(reg) c as u32,
            r = out(reg) r,
            x = out(vreg) _,
            y = out(vreg) _,
            z = out(vreg) _,
        )
    };
    r as u16
}

/// binary32 to binary16 (FCVT Hd, Sn), round to nearest even.
pub fn f32_to_f16(a: u32) -> u16 {
    let r: u32;
    // SAFETY: FCVT between single and half precision is in the AArch64 baseline.
    unsafe {
        asm!(
            "fmov {x:s}, {a:w}",
            "fcvt {y:h}, {x:s}",
            "umov {r:w}, {y:v}.h[0]",
            a = in(reg) a,
            r = out(reg) r,
            x = out(vreg) _,
            y = out(vreg) _,
        )
    };
    r as u16
}

/// binary16 to binary32 (FCVT Sd, Hn), exact.
pub fn f16_to_f32(h: u16) -> u32 {
    let r: u32;
    // SAFETY: as above.
    unsafe {
        asm!(
            "dup {y:v}.8h, {a:w}",
            "fcvt {x:s}, {y:h}",
            "fmov {r:w}, {x:s}",
            a = in(reg) h as u32,
            r = out(reg) r,
            x = out(vreg) _,
            y = out(vreg) _,
        )
    };
    r
}

/// Unsigned and signed 32-bit integers to binary16 in one rounding (UCVTF/SCVTF Hd, Wn).
pub fn u32_to_f16(v: u32) -> u16 {
    let r: u32;
    // SAFETY: as for the binary operations.
    unsafe {
        asm!(
            ".arch_extension fp16",
            "ucvtf {x:h}, {a:w}",
            "fmov {r:w}, {x:h}",
            a = in(reg) v,
            r = out(reg) r,
            x = out(vreg) _,
        )
    };
    r as u16
}

pub fn i32_to_f16(v: u32) -> u16 {
    let r: u32;
    // SAFETY: as for the binary operations.
    unsafe {
        asm!(
            ".arch_extension fp16",
            "scvtf {x:h}, {a:w}",
            "fmov {r:w}, {x:h}",
            a = in(reg) v,
            r = out(reg) r,
            x = out(vreg) _,
        )
    };
    r as u16
}

pub fn u64_to_f16(v: u64) -> u16 {
    let r: u32;
    // SAFETY: as for the binary operations.
    unsafe {
        asm!(
            ".arch_extension fp16",
            "ucvtf {x:h}, {a:x}",
            "fmov {r:w}, {x:h}",
            a = in(reg) v,
            r = out(reg) r,
            x = out(vreg) _,
        )
    };
    r as u16
}

pub fn i64_to_f16(v: u64) -> u16 {
    let r: u32;
    // SAFETY: as for the binary operations.
    unsafe {
        asm!(
            ".arch_extension fp16",
            "scvtf {x:h}, {a:x}",
            "fmov {r:w}, {x:h}",
            a = in(reg) v,
            r = out(reg) r,
            x = out(vreg) _,
        )
    };
    r as u16
}

/// MSL fmax: "Returns y if x < y, otherwise returns x. If one argument is a NaN, fmax() returns the other argument.
/// If both arguments are NaNs, fmax() returns a NaN." Two NaNs give the first signaling NaN quieted, else the first.
pub fn fmax(x: u32, y: u32) -> u32 {
    match (f(x).is_nan(), f(y).is_nan()) {
        (true, true) => nan2(x, y),
        (true, false) => y,
        (false, true) => x,
        _ => {
            if f(x) < f(y) {
                y
            } else {
                x
            }
        }
    }
}

/// MSL fmin: "Returns y if y < x, otherwise it returns x", with fmax's NaN rules.
pub fn fmin(x: u32, y: u32) -> u32 {
    match (f(x).is_nan(), f(y).is_nan()) {
        (true, true) => nan2(x, y),
        (true, false) => y,
        (false, true) => x,
        _ => {
            if f(y) < f(x) {
                y
            } else {
                x
            }
        }
    }
}

fn is_snan(a: u32) -> bool {
    f(a).is_nan() && a & 0x0040_0000 == 0
}

/// The NaN two NaN operands give: the CPU's own choice, observed through an FADD.
pub fn nan2(a: u32, b: u32) -> u32 {
    let r = add(a, b);
    debug_assert!(f(r).is_nan() && (is_snan(a) || is_snan(b) || r == a || r == b));
    r
}

/// ldexp(x, n), correctly rounded: the product is exact in binary64 for |n| <= 400 and is rounded once to binary32.
pub fn ldexp(a: u32, n: i32) -> u32 {
    let x = f(a);
    if x.is_nan() {
        return quiet(a);
    }
    let n = n.clamp(-400, 400);
    let p = (x as f64) * 2f64.powi(n);
    (p as f32).to_bits()
}
