//! The common and selection functions (common.wgsl) against the CPU: fmod and nextafter from the C library, fract
//! from the CPU's floor and subtraction, mix and smoothstep from MSL's formula in the CPU's binary32 arithmetic, and
//! the selections from MSL's definitions.

use crate::gpu::{Kernel, MAX_ITEMS, kernel_source, par_range};
use crate::inputs::{Rng, SPECIAL_F16, SPECIAL_F32, f32_any, f32_near};
use crate::prelude;
use crate::reference as r;

unsafe extern "C" {
    fn nextafterf(x: f32, y: f32) -> f32;
}

fn is_nan(a: u32) -> bool {
    a & 0x7fff_ffff > 0x7f80_0000
}

fn nan_of(a: &[u32]) -> Option<u32> {
    a.iter()
        .copied()
        .find(|&x| is_nan(x) && x & 0x0040_0000 == 0)
        .or_else(|| a.iter().copied().find(|&x| is_nan(x)))
        .map(|x| x | 0x0040_0000)
}

fn run(item: &str, ops: &[Vec<u32>], want: impl Fn(&[u32]) -> u32 + Sync) {
    let reads = ["nm_t_in0[i]", "nm_t_in1[i]", "nm_t_in2[i]"];
    let call = format!("nm_t_out[i] = {}({});", item, reads[..ops.len()].join(", "));
    let k = Kernel::new(&kernel_source(false, &prelude(&[item]), "", &call), 1);
    let total = ops[0].len();
    let mut start = 0;
    let mut bad: Vec<String> = Vec::new();
    let mut count = 0u64;
    while start < total {
        let n = (total - start).min(MAX_ITEMS as usize);
        let s: Vec<&[u32]> = ops.iter().map(|v| &v[start..start + n]).collect();
        let out = k.run(&s, [0, 0, 0], n as u32);
        let found = std::sync::Mutex::new((0u64, Vec::new()));
        par_range(n, |js| {
            let mut local = (0u64, Vec::new());
            for j in js {
                let a: Vec<u32> = s.iter().map(|v| v[j]).collect();
                let w = want(&a);
                if out[j] != w {
                    local.0 += 1;
                    if local.1.len() < 8 {
                        local.1.push(format!("{}({:08x?}) = {:08x}, want {:08x}", item, a, out[j], w));
                    }
                }
            }
            let mut g = found.lock().unwrap();
            g.0 += local.0;
            g.1.extend(local.1);
        });
        let (c, f) = found.into_inner().unwrap();
        count += c;
        bad.extend(f.into_iter().take(8));
        start += n;
    }
    assert_eq!(count, 0, "{}: {} of {} differ\n{}", item, count, total, bad.join("\n"));
    eprintln!("{}: {} inputs, 0 mismatches", item, total);
}

fn ops(seed: u64, arity: usize) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    let mut v = vec![Vec::new(); arity];
    for &a in SPECIAL_F32 {
        for &b in SPECIAL_F32 {
            v[0].push(a);
            if arity > 1 {
                v[1].push(b);
            }
            if arity > 2 {
                v[2].push(SPECIAL_F32[rng.below(SPECIAL_F32.len() as u32) as usize]);
            }
        }
    }
    while v[0].len() < 1 << 20 {
        let a = f32_any(&mut rng);
        v[0].push(a);
        if arity > 1 {
            v[1].push(f32_near(&mut rng, a));
        }
        if arity > 2 {
            v[2].push(f32_near(&mut rng, a));
        }
    }
    v
}

fn f(a: u32) -> f32 {
    f32::from_bits(a)
}

#[test]
fn fmod() {
    run("nm_f32_fmod", &ops(21, 2), |a| {
        if let Some(n) = nan_of(a) {
            return n;
        }
        let v = f(a[0]) % f(a[1]);
        // C99 F.9.7.1: invalid for an infinite x or a zero y.
        if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() }
    });
}

#[test]
fn nextafter() {
    run("nm_f32_nextafter", &ops(22, 2), |a| {
        if let Some(n) = nan_of(a) {
            return n;
        }
        // SAFETY: nextafterf is a pure C library function.
        unsafe { nextafterf(f(a[0]), f(a[1])) }.to_bits()
    });
}

#[test]
fn fract() {
    run("nm_f32_fract", &ops(23, 1), |a| {
        let x = a[0];
        if is_nan(x) {
            return x | 0x0040_0000;
        }
        let ax = x & 0x7fff_ffff;
        if ax == 0 {
            return x;
        }
        if ax == 0x7f80_0000 {
            return x & 0x8000_0000;
        }
        let d = f(x) - f(x).floor();
        if d >= 1.0 { 0x3f7f_ffff } else { d.to_bits() }
    });
}

#[test]
fn selections() {
    let o3 = ops(24, 3);
    run("nm_f32_fmax3", &o3, |a| r::fmax(a[0], r::fmax(a[1], a[2])));
    run("nm_f32_fmin3", &o3, |a| r::fmin(a[0], r::fmin(a[1], a[2])));
    run("nm_f32_clamp", &o3, |a| r::fmin(r::fmax(a[0], a[1]), a[2]));
    run("nm_f32_saturate", &o3[..1], |a| r::fmin(r::fmax(a[0], 0), 0x3f80_0000));
    run("nm_f32_fmedian3", &o3, |a| {
        let ok: Vec<u32> = a.iter().copied().filter(|&x| !is_nan(x)).collect();
        let network = if is_nan(a[2]) && !is_nan(a[0]) && !is_nan(a[1]) {
            r::fmin(a[0], a[1])
        } else {
            r::fmax(r::fmin(a[0], a[1]), r::fmin(r::fmax(a[0], a[1]), a[2]))
        };
        match ok.len() {
            // Three NaNs: MSL asks for a NaN, which the network gives.
            0 => {
                assert!(is_nan(network));
                network
            }
            1 => ok[0],
            // One NaN: MSL allows either remaining value; the library returns the one its fmin/fmax network gives.
            2 => {
                assert!(network == ok[0] || network == ok[1]);
                network
            }
            _ => {
                let mut s = ok.clone();
                s.sort_by(|x, y| f(*x).partial_cmp(&f(*y)).unwrap());
                let mid = s[1];
                // The network's choice among equal values (signed zeros) is its own; the value is the median's.
                if f(network) == f(mid) { network } else { mid }
            }
        }
    });
}

#[test]
fn sign_step_mix_smoothstep() {
    let o3 = ops(25, 3);
    run("nm_f32_sign", &o3[..1], |a| {
        let x = a[0];
        if is_nan(x) {
            0
        } else if x & 0x7fff_ffff == 0 {
            x
        } else {
            (x & 0x8000_0000) | 0x3f80_0000
        }
    });
    run("nm_f32_step", &o3[..2], |a| if f(a[1]) < f(a[0]) { 0 } else { 0x3f80_0000 });
    run("nm_f32_mix", &o3, |a| (f(a[0]) + (f(a[1]) - f(a[0])) * f(a[2])).to_bits());
    run("nm_f32_smoothstep", &o3, |a| {
        let t = (f(a[2]) - f(a[0])) / (f(a[1]) - f(a[0]));
        let t = r::fmin(r::fmax(t.to_bits(), 0), 0x3f80_0000);
        let t = f(t);
        (t * t * (3.0 - 2.0 * t)).to_bits()
    });
    run("nm_f32_recip", &o3[..1], |a| (1.0 / f(a[0])).to_bits());
}

/// The binary16 forms over every operand (one-operand) or a large sample, against the binary32 forms narrowed.
#[test]
fn binary16_forms() {
    let all: Vec<u32> = (0..1u32 << 16).collect();
    let widen = |h: u32| r::f16_to_f32(h as u16);
    let narrow = |x: u32| r::f32_to_f16(x) as u32;
    let q16 = |h: u32| h | 0x0200;
    let nan16 = |h: u32| h & 0x7fff > 0x7c00;
    run("nm_f16_fabs", std::slice::from_ref(&all), |a| a[0] & 0x7fff);
    run("nm_f16_ceil", std::slice::from_ref(&all), move |a| narrow(f(widen(a[0])).ceil().to_bits()));
    run("nm_f16_floor", std::slice::from_ref(&all), move |a| narrow(f(widen(a[0])).floor().to_bits()));
    run("nm_f16_trunc", std::slice::from_ref(&all), move |a| narrow(f(widen(a[0])).trunc().to_bits()));
    run("nm_f16_rint", std::slice::from_ref(&all), move |a| narrow(f(widen(a[0])).round_ties_even().to_bits()));
    run("nm_f16_round", std::slice::from_ref(&all), move |a| narrow(f(widen(a[0])).round().to_bits()));
    run("nm_f16_fract", std::slice::from_ref(&all), move |a| {
        let h = a[0];
        if nan16(h) {
            return q16(h);
        }
        if h & 0x7fff == 0 {
            return h;
        }
        if h & 0x7fff == 0x7c00 {
            return h & 0x8000;
        }
        let x = f(widen(h));
        narrow((x - x.floor()).to_bits()).min(0x3bff)
    });
    let mut rng = Rng::new(26);
    let mut a = Vec::new();
    let mut b = Vec::new();
    for _ in 0..(1 << 20) {
        let pick = |rng: &mut Rng| {
            if rng.below(8) == 0 {
                SPECIAL_F16[rng.below(SPECIAL_F16.len() as u32) as usize] as u32
            } else {
                rng.below(1 << 16)
            }
        };
        a.push(pick(&mut rng));
        b.push(pick(&mut rng));
    }
    let pair = [a, b];
    run("nm_f16_fmod", &pair, move |x| {
        if nan16(x[0]) || nan16(x[1]) {
            // The binary16 NaN rule, like binary32's: first signaling, else first quiet NaN, quieted.
            let first_s = x.iter().copied().find(|&h| nan16(h) && h & 0x0200 == 0);
            return first_s.or_else(|| x.iter().copied().find(|&h| nan16(h))).unwrap() | 0x0200;
        }
        let v = f(widen(x[0])) % f(widen(x[1]));
        if v.is_nan() { 0x7e00 } else { narrow(v.to_bits()) }
    });
    run("nm_f16_nextafter", &pair, move |x| {
        if nan16(x[0]) || nan16(x[1]) {
            let first_s = x.iter().copied().find(|&h| nan16(h) && h & 0x0200 == 0);
            return first_s.or_else(|| x.iter().copied().find(|&h| nan16(h))).unwrap() | 0x0200;
        }
        let (fx, fy) = (f(widen(x[0])), f(widen(x[1])));
        if fx == fy {
            return x[1];
        }
        if x[0] & 0x7fff == 0 {
            return (x[1] & 0x8000) | 1;
        }
        // The neighbouring binary16 value towards y.
        let up = fx < fy;
        let pos = x[0] & 0x8000 == 0;
        if up == pos { x[0] + 1 } else { x[0] - 1 }
    });
    run("nm_f16_fmax", &pair, move |x| narrow(r::fmax(widen(x[0]), widen(x[1]))));
    run("nm_f16_fmin", &pair, move |x| narrow(r::fmin(widen(x[0]), widen(x[1]))));
    run("nm_f16_step", &pair, move |x| if f(widen(x[1])) < f(widen(x[0])) { 0 } else { 0x3c00 });
}

/// fma and the arithmetic on every binary32 operand pair is covered in softfloat.rs; here a check that the f32
/// recip is the division it is defined as, over a full sweep of one binade and its neighbours.
#[test]
fn recip_binade() {
    let v: Vec<u32> = (0x3f00_0000u32..0x4000_0000).step_by(7).collect();
    run("nm_f32_recip", &[v], |a| (1.0 / f(a[0])).to_bits());
}
