//! The math functions (math.wgsl) against the system libm in binary64.
//!
//! Errors are measured in ulps as MSL 8.4 defines them, against the binary64 reference (itself within one binary64
//! ulp, about 2^-29 binary32 ulps). Special values are checked bit for bit against rules written here from the texts
//! the library follows: MSL Table 6.4 where MSL defines a result, otherwise C99 Annex F.9 (WG14 N1256) and OpenCL C
//! 3.0 section 7.5.1. A NaN operand must come back quieted with its payload, an invalid operation as the default NaN.
//!
//! Every function must stay within 1 ulp, which is inside every bound MSL states for it (Table 8.1 precise; Table 8.2
//! fast, whose smallest ulp bounds are 2 for log2 and 3 for exp and exp2, and whose absolute bounds for sin, cos,
//! sinpi, cospi, log and log2 are implied by 1 ulp on their domains).

use crate::gpu::{kernel_source, par_range, Gpu, Kernel, MAX_ITEMS};
use crate::inputs::{f32_any, Rng, SPECIAL_F32};
use crate::prelude;
use crate::reference as r;
use std::sync::Mutex;

unsafe extern "C" {
    fn __sinpi(x: f64) -> f64;
    fn __cospi(x: f64) -> f64;
    fn __tanpi(x: f64) -> f64;
    fn __exp10(x: f64) -> f64;
}

const DNAN: u32 = 0x7fc0_0000;
const PINF: u32 = 0x7f80_0000;
const NINF: u32 = 0xff80_0000;
const ONE: u32 = 0x3f80_0000;

fn is_nan(a: u32) -> bool {
    a & 0x7fff_ffff > 0x7f80_0000
}

fn f(a: u32) -> f64 {
    f32::from_bits(a) as f64
}

/// The value of a binary32 result for error measurement; infinity counts as 2^128, the next value after the largest
/// finite one.
fn value(y: u32) -> f64 {
    match y {
        PINF => 2f64.powi(128),
        NINF => -(2f64.powi(128)),
        _ => f(y),
    }
}

/// MSL 8.4: "If x is a real number that lies between two finite consecutive floating-point numbers a and b, without
/// being equal to one of them, then ulp(x) = |b - a|, otherwise ulp(x) is the distance between the two non-equal
/// finite floating-point numbers nearest x."
pub fn ulp32(x: f64) -> f64 {
    let ax = x.abs();
    let max = f32::MAX as f64;
    if ax >= max {
        return max - f(f32::MAX.to_bits() - 1);
    }
    let c = ax as f32;
    let lo = if (c as f64) > ax { f32::from_bits(c.to_bits() - 1) } else { c };
    let hi = f32::from_bits(lo.to_bits() + 1);
    if lo as f64 == ax {
        if lo.to_bits() == 0 {
            return f(1);
        }
        let below = f32::from_bits(lo.to_bits() - 1);
        return (hi as f64 - lo as f64).min(lo as f64 - below as f64);
    }
    hi as f64 - lo as f64
}

/// The same definition on the binary16 grid.
pub fn ulp16(x: f64) -> f64 {
    let ax = x.abs();
    if ax >= 65504.0 {
        return 32.0;
    }
    if ax < 2f64.powi(-14) {
        return 2f64.powi(-24);
    }
    let e = ax.log2().floor() as i32;
    let spacing = 2f64.powi(e - 10);
    // A power of two: the spacing below is half the spacing above.
    if ax == 2f64.powi(e) && e > -14 { spacing / 2.0 } else { spacing }
}

pub fn value16(h: u32) -> f64 {
    match h {
        0x7c00 => 65536.0,
        0xfc00 => -65536.0,
        _ => f(r::f16_to_f32(h as u16)),
    }
}

/// The expected bits of a special case, or `None` when the result is measured in ulps.
type Special = fn(&[u32]) -> Option<u32>;

pub struct Fun {
    pub name: &'static str,
    pub arity: usize,
    reference: fn(&[f64]) -> f64,
    special: Special,
}

fn nan_in(a: &[u32]) -> Option<u32> {
    a.iter().position(|&x| is_nan(x)).map(|_| {
        // Two NaN operands: the first signaling NaN, else the first quiet NaN, quieted.
        a.iter().copied().find(|&x| is_nan(x) && x & 0x0040_0000 == 0).or_else(|| a.iter().copied().find(|&x| is_nan(x))).unwrap()
            | 0x0040_0000
    })
}

fn zero_of(a: u32) -> u32 {
    a & 0x8000_0000
}

fn sp_exp(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] {
        PINF => Some(PINF),
        NINF => Some(0),
        x if x & 0x7fff_ffff == 0 => Some(ONE),
        _ => None,
    })
}

fn sp_log(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] {
        PINF => Some(PINF),
        ONE => Some(0),
        x if x & 0x7fff_ffff == 0 => Some(NINF),
        x if x >> 31 != 0 => Some(DNAN),
        _ => None,
    })
}

fn sp_odd_zero_inf_nan(a: &[u32]) -> Option<u32> {
    // f(+-0) = +-0 and f(+-inf) is invalid: sin, tan.
    nan_in(a).or(match a[0] & 0x7fff_ffff {
        0 => Some(a[0]),
        PINF => Some(DNAN),
        _ => None,
    })
}

fn sp_cos(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] & 0x7fff_ffff {
        0 => Some(ONE),
        PINF => Some(DNAN),
        _ => None,
    })
}

fn int_parity(a: u32) -> Option<bool> {
    let x = f32::from_bits(a);
    if x.is_finite() && x == x.trunc() { Some((x.abs() as f64 % 2.0) == 1.0) } else { None }
}

fn half_int_parity(a: u32) -> Option<bool> {
    let x = f32::from_bits(a) as f64;
    let t = x - 0.5;
    if x.is_finite() && t == t.trunc() { Some(t.rem_euclid(2.0) == 1.0) } else { None }
}

// OpenCL C 3.0 7.5.1: sinpi(+-0) = +-0; sinpi(+n) = +0 for positive integers n; sinpi(-n) = -0 for negative ones.
fn sp_sinpi(a: &[u32]) -> Option<u32> {
    nan_in(a).or(if a[0] & 0x7fff_ffff == PINF {
        Some(DNAN)
    } else if int_parity(a[0]).is_some() {
        Some(zero_of(a[0]))
    } else if let Some(odd) = half_int_parity(a[0]) {
        // sin(pi (n + 1/2)) = (-1)^n exactly.
        Some(if odd { 0xbf80_0000 } else { ONE })
    } else {
        None
    })
}

// cospi(+-0) = 1; cospi(n + 0.5) = +0.
fn sp_cospi(a: &[u32]) -> Option<u32> {
    nan_in(a).or(if a[0] & 0x7fff_ffff == PINF {
        Some(DNAN)
    } else if let Some(odd) = int_parity(a[0]) {
        Some(if odd { 0xbf80_0000 } else { ONE })
    } else if half_int_parity(a[0]).is_some() {
        Some(0)
    } else {
        None
    })
}

// tanpi(+-0) = +-0; tanpi(n) = copysign(0, n) for even n, copysign(0, -n) for odd n; tanpi(n + 0.5) = +inf for even
// n, -inf for odd n.
fn sp_tanpi(a: &[u32]) -> Option<u32> {
    nan_in(a).or(if a[0] & 0x7fff_ffff == PINF {
        Some(DNAN)
    } else if let Some(odd) = int_parity(a[0]) {
        Some(if odd { zero_of(a[0]) ^ 0x8000_0000 } else { zero_of(a[0]) })
    } else if let Some(odd) = half_int_parity(a[0]) {
        Some(if odd { NINF } else { PINF })
    } else {
        None
    })
}

fn sp_asin(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] & 0x7fff_ffff {
        0 => Some(a[0]),
        x if x > ONE => Some(DNAN),
        _ => None,
    })
}

fn sp_acos(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] {
        ONE => Some(0),
        x if x & 0x7fff_ffff > ONE => Some(DNAN),
        _ => None,
    })
}

fn sp_atan(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] & 0x7fff_ffff {
        0 => Some(a[0]),
        _ => None,
    })
}

fn sp_sinh(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] & 0x7fff_ffff {
        0 | PINF => Some(a[0]),
        _ => None,
    })
}

fn sp_cosh(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] & 0x7fff_ffff {
        0 => Some(ONE),
        PINF => Some(PINF),
        _ => None,
    })
}

fn sp_tanh(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] & 0x7fff_ffff {
        0 => Some(a[0]),
        PINF => Some(zero_of(a[0]) | ONE),
        _ => None,
    })
}

fn sp_acosh(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] {
        ONE => Some(0),
        PINF => Some(PINF),
        x if x >> 31 != 0 || x < ONE => Some(DNAN),
        _ => None,
    })
}

fn sp_atanh(a: &[u32]) -> Option<u32> {
    nan_in(a).or(match a[0] & 0x7fff_ffff {
        0 => Some(a[0]),
        ONE => Some(zero_of(a[0]) | PINF),
        x if x > ONE => Some(DNAN),
        _ => None,
    })
}

// C99 F.9.1.4 (the results that are multiples of pi are measured in ulps).
fn sp_atan2(a: &[u32]) -> Option<u32> {
    if let Some(n) = nan_in(a) {
        return Some(n);
    }
    let (y, x) = (a[0], a[1]);
    let ay = y & 0x7fff_ffff;
    let ax = x & 0x7fff_ffff;
    if ay == 0 && x >> 31 == 0 {
        return Some(y);
    }
    if ax == PINF && ay != PINF && x >> 31 == 0 {
        return Some(zero_of(y));
    }
    None
}

fn is_odd_int(a: u32) -> bool {
    int_parity(a) == Some(true)
}

fn is_int(a: u32) -> bool {
    int_parity(a).is_some()
}

// C99 F.9.4.4.
fn sp_pow(a: &[u32]) -> Option<u32> {
    let (x, y) = (a[0], a[1]);
    let ax = x & 0x7fff_ffff;
    let ay = y & 0x7fff_ffff;
    let sx = x >> 31 != 0;
    let sy = y >> 31 != 0;
    if ay == 0 || x == ONE {
        return Some(ONE);
    }
    if let Some(n) = nan_in(a) {
        return Some(n);
    }
    if ax == 0 {
        return Some(match (sy, is_odd_int(y)) {
            (true, true) => x | PINF,
            (true, false) => PINF,
            (false, true) => x,
            (false, false) => 0,
        });
    }
    if ay == PINF {
        if x == 0xbf80_0000 {
            return Some(ONE);
        }
        return Some(if (ax < ONE) == sy { PINF } else { 0 });
    }
    if x == NINF {
        return Some(match (sy, is_odd_int(y)) {
            (true, true) => 0x8000_0000,
            (true, false) => 0,
            (false, true) => NINF,
            (false, false) => PINF,
        });
    }
    if x == PINF {
        return Some(if sy { 0 } else { PINF });
    }
    if sx && !is_int(y) {
        return Some(DNAN);
    }
    None
}

// OpenCL C 3.0 7.5.1.
fn sp_powr(a: &[u32]) -> Option<u32> {
    let (x, y) = (a[0], a[1]);
    let ax = x & 0x7fff_ffff;
    let ay = y & 0x7fff_ffff;
    if let Some(n) = nan_in(a) {
        return Some(n);
    }
    if x >> 31 != 0 && ax != 0 {
        return Some(DNAN);
    }
    let sy = y >> 31 != 0;
    if ax == 0 {
        return Some(if ay == 0 { DNAN } else if sy { PINF } else { 0 });
    }
    if ax == PINF {
        return Some(if ay == 0 { DNAN } else if sy { 0 } else { PINF });
    }
    if x == ONE {
        return Some(if ay == PINF { DNAN } else { ONE });
    }
    if ay == 0 {
        return Some(ONE);
    }
    if ay == PINF {
        return Some(if (ax < ONE) == sy { PINF } else { 0 });
    }
    None
}

macro_rules! fun1 {
    ($name:literal, $reference:expr, $special:expr) => {
        Fun { name: $name, arity: 1, reference: |x| $reference(x[0]), special: $special }
    };
}

pub fn functions() -> Vec<Fun> {
    vec![
        fun1!("exp", f64::exp, sp_exp),
        fun1!("exp2", f64::exp2, sp_exp),
        fun1!("exp10", |x| unsafe { __exp10(x) }, sp_exp),
        fun1!("log", f64::ln, sp_log),
        fun1!("log2", f64::log2, sp_log),
        fun1!("log10", f64::log10, sp_log),
        fun1!("sin", f64::sin, sp_odd_zero_inf_nan),
        fun1!("cos", f64::cos, sp_cos),
        fun1!("tan", f64::tan, sp_odd_zero_inf_nan),
        fun1!("sinpi", |x| unsafe { __sinpi(x) }, sp_sinpi),
        fun1!("cospi", |x| unsafe { __cospi(x) }, sp_cospi),
        fun1!("tanpi", |x| unsafe { __tanpi(x) }, sp_tanpi),
        fun1!("asin", f64::asin, sp_asin),
        fun1!("acos", f64::acos, sp_acos),
        fun1!("atan", f64::atan, sp_atan),
        fun1!("sinh", f64::sinh, sp_sinh),
        fun1!("cosh", f64::cosh, sp_cosh),
        fun1!("tanh", f64::tanh, sp_tanh),
        fun1!("asinh", f64::asinh, sp_sinh),
        fun1!("acosh", f64::acosh, sp_acosh),
        fun1!("atanh", f64::atanh, sp_atanh),
        Fun { name: "atan2", arity: 2, reference: |x| x[0].atan2(x[1]), special: sp_atan2 },
        Fun { name: "pow", arity: 2, reference: |x| x[0].powf(x[1]), special: sp_pow },
        Fun { name: "powr", arity: 2, reference: |x| x[0].powf(x[1]), special: sp_powr },
    ]
}

pub fn function(name: &str) -> Fun {
    functions().into_iter().find(|f| f.name == name).unwrap()
}

/// Largest error seen, where, and failures.
#[derive(Default)]
struct Tally {
    max: f64,
    at: Vec<u32>,
    failures: u64,
    first: Vec<String>,
    count: u64,
}

impl Tally {
    fn merge(&mut self, o: Tally) {
        if o.max > self.max {
            self.max = o.max;
            self.at = o.at;
        }
        self.failures += o.failures;
        self.count += o.count;
        for s in o.first {
            if self.first.len() < 12 {
                self.first.push(s);
            }
        }
    }

    fn fail(&mut self, s: String) {
        self.failures += 1;
        if self.first.len() < 12 {
            self.first.push(s);
        }
    }
}

/// Check one result: special values exactly, the rest within `bound` ulps (`half` selects the binary16 grid).
fn assess(fun: &Fun, ops: &[u32], y: u32, half: bool, bound: f64, t: &mut Tally) {
    t.count += 1;
    let wide: Vec<u32> = if half { ops.iter().map(|&h| r::f16_to_f32(h as u16)).collect() } else { ops.to_vec() };
    if let Some(want) = (fun.special)(&wide) {
        let want = if half { r::f32_to_f16(want) as u32 } else { want };
        if y != want {
            t.fail(format!("{}({:08x?}) = {:08x}, want {:08x}", fun.name, ops, y, want));
        }
        return;
    }
    let refv = (fun.reference)(&wide.iter().map(|&a| f(a)).collect::<Vec<_>>());
    let nan = if half { y & 0x7fff > 0x7c00 } else { is_nan(y) };
    if refv.is_nan() || nan {
        if !(refv.is_nan() && nan) {
            t.fail(format!("{}({:08x?}) = {:08x}, reference {}", fun.name, ops, y, refv));
        }
        return;
    }
    // The reference rounded to the result format: equal bits are no error (this covers results that overflow).
    let rounded = (refv as f32).to_bits();
    let rounded = if half { r::f32_to_f16(rounded) as u32 } else { rounded };
    let (yv, u) = if half { (value16(y), ulp16(refv)) } else { (value(y), ulp32(refv)) };
    let e = (yv - refv).abs() / u;
    // Equal bits are the correctly rounded result even where the distance is not finite (an overflow to infinity).
    let err = if y == rounded { e.min(0.5) } else { e };
    if std::env::var("NM_DEBUG").is_ok() && err > 0.4 {
        eprintln!("debug {}({:08x?}) = {:08x} rounded {:08x} ref {:e} u {:e} err {}", fun.name, ops, y, rounded, refv, u, err);
    }
    if err > t.max {
        t.max = err;
        t.at = ops.to_vec();
    }
    if err > bound {
        t.fail(format!("{}({:08x?}) = {:08x} ({}), reference {}, {} ulp", fun.name, ops, y, yv, refv, err));
    }
}

fn report(what: &str, t: Tally, bound: f64) {
    assert_eq!(t.failures, 0, "{}: {} of {} fail\n{}", what, t.failures, t.count, t.first.join("\n"));
    eprintln!(
        "{}: {} inputs, max error {:.6} ulp at {:08x?} (bound {}) ({})",
        what,
        t.count,
        t.max,
        t.at,
        bound,
        Gpu::get().adapter
    );
}

fn item(fun: &Fun, half: bool) -> String {
    format!("nm_{}_{}", if half { "f16" } else { "f32" }, fun.name)
}

/// Run a function over operand lists and assess every result.
fn run_lists(fun: &Fun, half: bool, ops: &[Vec<u32>], bound: f64) {
    let it = item(fun, half);
    let args: Vec<&str> = ["nm_t_in0[i]", "nm_t_in1[i]"][..fun.arity].to_vec();
    let src = kernel_source(false, &prelude(&[&it]), "", &format!("nm_t_out[i] = {}({});", it, args.join(", ")));
    let k = Kernel::new(&src, 1);
    let total = ops[0].len();
    let tally = Mutex::new(Tally::default());
    let mut start = 0;
    while start < total {
        let n = (total - start).min(MAX_ITEMS as usize);
        let slices: Vec<&[u32]> = ops.iter().map(|v| &v[start..start + n]).collect();
        let out = k.run(&slices, [0, 0, 0], n as u32);
        par_range(n, |js| {
            let mut t = Tally::default();
            for j in js {
                let o: Vec<u32> = slices.iter().map(|s| s[j]).collect();
                assess(fun, &o, out[j], half, bound, &mut t);
            }
            tally.lock().unwrap().merge(t);
        });
        start += n;
    }
    report(&format!("{} {}", if half { "binary16" } else { "binary32" }, fun.name), tally.into_inner().unwrap(), bound);
}

/// Every binary32 (or, with `half`, binary16) input of a one-operand function.
fn sweep(fun: &Fun, half: bool, bound: f64) {
    let it = item(fun, half);
    let src = kernel_source(false, &prelude(&[&it]), "", &format!("nm_t_out[i] = {}(nm_t_p.y + i);", it));
    let k = Kernel::new(&src, 1);
    let end: u64 = if half { 1 << 16 } else { 1 << 32 };
    let tally = Mutex::new(Tally::default());
    let mut base = 0u64;
    while base < end {
        let n = (end - base).min(MAX_ITEMS as u64) as u32;
        let out = k.run(&[], [base as u32, 0, 0], n);
        par_range(n as usize, |js| {
            let mut t = Tally::default();
            for j in js {
                assess(fun, &[(base + j as u64) as u32], out[j], half, bound, &mut t);
            }
            tally.lock().unwrap().merge(t);
        });
        base += n as u64;
    }
    report(&format!("{} {} (every input)", if half { "binary16" } else { "binary32" }, fun.name), tally.into_inner().unwrap(), bound);
}

fn sample_unary(n: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    let mut v: Vec<u32> = SPECIAL_F32.to_vec();
    // Integers, half-integers and values near multiples of pi/2 for the trigonometric functions.
    for k in 0..4096u32 {
        v.push((k as f32 * 0.5).to_bits());
        v.push((-(k as f32) * 0.5).to_bits());
        v.push(((k as f64) * std::f64::consts::FRAC_PI_2) as f32 as u32);
    }
    while v.len() < n {
        v.push(f32_any(&mut rng));
    }
    vec![v]
}

fn sample_binary(n: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    let mut a = Vec::new();
    let mut b = Vec::new();
    let extra = [0x3f00_0000u32, 0xbf00_0000, 0x4040_0000, 0xc040_0000, 0x3eaa_aaab, 0x4b00_0001, 0xcb00_0001];
    for &x in SPECIAL_F32.iter().chain(extra.iter()) {
        for &y in SPECIAL_F32.iter().chain(extra.iter()) {
            a.push(x);
            b.push(y);
        }
    }
    while a.len() < n {
        let x = f32_any(&mut rng);
        // Exponents for pow that keep results finite most of the time, and integer exponents for negative bases.
        let y = match rng.below(4) {
            0 => ((rng.below(64) as i32 - 32) as f32).to_bits(),
            1 => (((rng.u32() as f64 / u32::MAX as f64) * 8.0 - 4.0) as f32).to_bits(),
            _ => f32_any(&mut rng),
        };
        a.push(x);
        b.push(y);
    }
    vec![a, b]
}

/// The quick check of every function: special values and a sample.
#[test]
fn binary32_functions_sample() {
    for fun in functions() {
        let ops = if fun.arity == 1 { sample_unary(1 << 18, 11) } else { sample_binary(1 << 18, 12) };
        run_lists(&fun, false, &ops, 1.0);
    }
}

/// Every binary16 input of each one-operand function, rounded once to binary16 (MSL states no half-precision bound
/// for the target, numerics.toml generic decision; Table 8.3 asks 1 ulp from Apple family 4).
#[test]
fn binary16_functions_all() {
    for fun in functions().into_iter().filter(|f| f.arity == 1) {
        sweep(&fun, true, 1.0);
    }
}

#[test]
fn binary16_binary_functions_sample() {
    let mut rng = Rng::new(13);
    let mut a = Vec::new();
    let mut b = Vec::new();
    for _ in 0..(1 << 18) {
        a.push(rng.below(1 << 16));
        b.push(if rng.below(2) == 0 { r::f32_to_f16(((rng.below(16) as i32 - 8) as f32).to_bits()) as u32 } else { rng.below(1 << 16) });
    }
    for fun in functions().into_iter().filter(|f| f.arity == 2) {
        run_lists(&fun, true, &[a.clone(), b.clone()], 1.0);
    }
}

macro_rules! exhaustive {
    ($test:ident, $name:literal) => {
        #[test]
        #[ignore = "exhaustive: run with --ignored"]
        fn $test() {
            sweep(&function($name), false, 1.0);
        }
    };
}

exhaustive!(exp_all, "exp");
exhaustive!(exp2_all, "exp2");
exhaustive!(exp10_all, "exp10");
exhaustive!(log_all, "log");
exhaustive!(log2_all, "log2");
exhaustive!(log10_all, "log10");
exhaustive!(sin_all, "sin");
exhaustive!(cos_all, "cos");
exhaustive!(tan_all, "tan");
exhaustive!(sinpi_all, "sinpi");
exhaustive!(cospi_all, "cospi");
exhaustive!(tanpi_all, "tanpi");
exhaustive!(asin_all, "asin");
exhaustive!(acos_all, "acos");
exhaustive!(atan_all, "atan");
exhaustive!(sinh_all, "sinh");
exhaustive!(cosh_all, "cosh");
exhaustive!(tanh_all, "tanh");
exhaustive!(asinh_all, "asinh");
exhaustive!(acosh_all, "acosh");
exhaustive!(atanh_all, "atanh");

macro_rules! long_binary {
    ($test:ident, $name:literal) => {
        #[test]
        #[ignore = "long: run with --ignored"]
        fn $test() {
            run_lists(&function($name), false, &sample_binary(1 << 26, 14), 1.0);
        }
    };
}

long_binary!(atan2_long, "atan2");
long_binary!(pow_long, "pow");
long_binary!(powr_long, "powr");
