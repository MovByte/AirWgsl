//! Pack and unpack (MSL 6.15, conversion rules 8.7.1 and 8.7.7) and the integer edges, lowered and run on the GPU:
//! against exact references written here, and against what Metal returned (tests/num/data/metal-pack.txt).

use crate::gpu::{kernel_source, par_range, Kernel, MAX_ITEMS};
use crate::inputs::Rng;
use crate::reference as r;
use airwgsl::api::HostCaps;
use airwgsl::num::{lower, FastMath, Op, Operand, Rounding, Scalar, Target, Ty};
use std::collections::BTreeMap;

const DATA: &str = include_str!("data/metal-pack.txt");

/// Lower `name` for operands of `tys` and return a kernel over items of `stride` input words: the operands' words
/// are read from nm_t_in0, the result's words written to nm_t_out.
fn kernel(name: &str, tys: &[Ty], ret: Ty, stride: u32) -> Kernel {
    let mut caps = HostCaps::default();
    caps.features.insert("shader-f16".into());
    let soft = Rounding::SoftFloat;
    let mut exprs = Vec::new();
    let mut base = 0;
    for t in tys {
        let w = |k: u32| format!("nm_t_in0[i * {}u + {}u]", stride, base + k);
        exprs.push(if t.lanes == 1 { w(0) } else { format!("{}({})", t.carrier(), (0..t.lanes).map(w).collect::<Vec<_>>().join(", ")) });
        base += t.lanes;
    }
    let ops: Vec<Operand> = exprs.iter().zip(tys).map(|(e, t)| Operand { expr: e, ty: *t, constant: None }).collect();
    let l = lower(Op::Call(name), &ops, Some(ret), FastMath::NONE, Target { rounding: &soft, host: &caps }, "v").unwrap();
    let v = l.value.unwrap();
    let mut body = l.stmts.join("\n");
    for k in 0..ret.lanes {
        let e = if ret.lanes == 1 { v.clone() } else { format!("({})[{}]", v, k) };
        body += &format!("\nnm_t_out[i * {}u + {}u] = {};", ret.lanes, k, e);
    }
    Kernel::new(&kernel_source(false, &l.prelude.emit().unwrap(), "", &body), ret.lanes)
}

fn run(k: &Kernel, inputs: &[u32], stride: usize, outs: usize) -> Vec<Vec<u32>> {
    let mut res = Vec::new();
    for chunk in inputs.chunks(stride * (MAX_ITEMS as usize / 8)) {
        let n = chunk.len() / stride;
        let out = k.run(&[chunk], [0, 0, 0], n as u32);
        res.extend(out.chunks(outs).map(|c| c.to_vec()));
    }
    res
}

/// The integer nearest num / den, ties to even.
fn rne(num: u128, den: u128) -> u128 {
    let q = num / den;
    let rem = num % den;
    if 2 * rem > den || (2 * rem == den && q & 1 == 1) { q + 1 } else { q }
}

/// MSL Table 8.7 with exact arithmetic (8.7.1.2: conversions to unorm round correctly): the unsigned code of a
/// binary32 value.
fn unorm_code(x: u32, maxv: u32) -> u32 {
    let f = r::f(x);
    if f.is_nan() || f <= 0.0 {
        return 0;
    }
    if f >= 1.0 {
        return maxv;
    }
    let (m, e) = if x >> 23 == 0 { (x as u128, -149) } else { ((x & 0x7f_ffff | 0x80_0000) as u128, (x >> 23) as i32 - 150) };
    // x * maxv = m * maxv / 2^-e, below 2^-60 when -e >= 100 (m * maxv < 2^40).
    if -e >= 100 {
        return 0;
    }
    rne(m * maxv as u128, 1u128 << (-e) as u32) as u32
}

fn snorm_code(x: u32, maxv: u32, bits: u32) -> u32 {
    if r::f(x).is_nan() {
        return 0;
    }
    let mag = unorm_code(x & 0x7fff_ffff, maxv);
    (if x >> 31 != 0 { mag.wrapping_neg() } else { mag }) & ((1 << bits) - 1)
}

/// Metal's 565 packing: the product rounded to binary32 before the nearest integer (measured).
fn code_565_metal(x: u32, maxv: u32) -> u32 {
    if r::f(x).is_nan() {
        return 0;
    }
    r::f(r::mul(x, (maxv as f32).to_bits())).clamp(0.0, maxv as f32).round_ties_even() as u32
}

/// The binary16 value nearest num / den (num, den > 0, the quotient below 65504), exactly.
fn f16_nearest(num: u64, den: u64) -> u32 {
    if num == 0 {
        return 0;
    }
    // Find e with 2^e <= q < 2^(e+1); significand step 2^(max(e, -14) - 10).
    // num / den < 2^e  <=>  num * 2^24 < den * 2^(e + 24)
    let mut e = 15i32;
    while e > -24 && ((num as u128) << 24) < ((den as u128) << (e + 24)) {
        e -= 1;
    }
    let step = e.max(-14) - 10;
    // code = nearest integer to q / 2^step
    let (n, d) = if step >= 0 { (num as u128, (den as u128) << step) } else { ((num as u128) << (-step), den as u128) };
    let code = rne(n, d) as u32;
    // Assemble: value = code * 2^step.
    if step == -24 {
        return code; // subnormal (or the carry into the smallest normal, which has the same bits)
    }
    let biased = (step + 10 + 15) as u32;
    // code is in [2^10, 2^11]; a carry to 2^11 moves to the next binade, which the addition below performs.
    ((biased - 1) << 10) + code
}

#[test]
fn normalized_conversions_are_exact() {
    let mut rng = Rng::new(41);
    // pack: every format on random and boundary components.
    let edge: Vec<u32> = (0..256u32)
        .flat_map(|k| {
            let half = (k as f64 + 0.5) / 255.0;
            let h = (half as f32).to_bits();
            [h - 1, h, h + 1, ((k as f32) / 127.0).to_bits()]
        })
        .chain([0, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x7fc0_0000, 0x7f80_0000, 0xff80_0000, 1, 0x3f7f_ffff])
        .collect();
    for (name, comps, signed) in [
        ("air.pack.unorm4x8.v4f32", vec![(8u32, 255u32); 4], false),
        ("air.pack.snorm4x8.v4f32", vec![(8, 127); 4], true),
        ("air.pack.unorm2x16.v2f32", vec![(16, 65535); 2], false),
        ("air.pack.snorm2x16.v2f32", vec![(16, 32767); 2], true),
        ("air.pack.unorm.rgb10a2.v4f32", vec![(10, 1023), (10, 1023), (10, 1023), (2, 3)], false),
        ("air.pack.unorm.rgb565.v3f32", vec![(5, 31), (6, 63), (5, 31)], false),
    ] {
        let n = comps.len() as u32;
        let ret = if name.contains("565") { Scalar::I16 } else { Scalar::I32 };
        let k = kernel(name, &[Ty::vector(Scalar::F32, n)], Ty::scalar(ret), n);
        let mut inputs = Vec::new();
        for _ in 0..(1 << 18) {
            for _ in 0..n {
                inputs.push(match rng.below(3) {
                    0 => edge[rng.below(edge.len() as u32) as usize],
                    1 => ((rng.u32() as f64 / u32::MAX as f64 * 2.4 - 1.2) as f32).to_bits(),
                    _ => rng.u32(),
                });
            }
        }
        let got = run(&k, &inputs, n as usize, 1);
        for (j, g) in got.iter().enumerate() {
            let mut want = 0u32;
            let mut shift = 0;
            for (c, &(bits, maxv)) in comps.iter().enumerate() {
                let x = inputs[j * n as usize + c];
                let code = if signed { snorm_code(x, maxv, bits) } else { unorm_code(x, maxv) };
                want |= code << shift;
                shift += bits;
            }
            assert_eq!(g[0], want, "{} {:08x?}", name, &inputs[j * n as usize..(j + 1) * n as usize]);
        }
        eprintln!("{}: {} items exact", name, got.len());
    }
    // unpack: every code of every format, to binary32 and binary16.
    for (fmt, comps, signed) in [
        ("unorm4x8", vec![(8u32, 255u32); 4], false),
        ("snorm4x8", vec![(8, 127); 4], true),
        ("unorm2x16", vec![(16, 65535); 2], false),
        ("snorm2x16", vec![(16, 32767); 2], true),
        ("unorm.rgb10a2", vec![(10, 1023), (10, 1023), (10, 1023), (2, 3)], false),
        ("unorm.rgb565", vec![(5, 31), (6, 63), (5, 31)], false),
    ] {
        for half in [false, true] {
            let n = comps.len() as u32;
            let vt = Ty::vector(if half { Scalar::F16 } else { Scalar::F32 }, n);
            let name = format!("air.unpack.{}.{}", fmt, vt.suffix());
            let pt = if fmt.contains("565") { Scalar::I16 } else { Scalar::I32 };
            let k = kernel(&name, &[Ty::scalar(pt)], vt, 1);
            // Words that put every code in every component: code k in all components at once.
            let maxbits = comps.iter().map(|c| c.0).max().unwrap();
            let words: Vec<u32> = (0..1u32 << maxbits)
                .map(|code| {
                    let mut w = 0;
                    let mut shift = 0;
                    for &(bits, _) in &comps {
                        w |= (code & ((1 << bits) - 1)) << shift;
                        shift += bits;
                    }
                    w & if pt == Scalar::I16 { 0xffff } else { u32::MAX }
                })
                .collect();
            let got = run(&k, &words, 1, n as usize);
            for (j, g) in got.iter().enumerate() {
                let mut shift = 0;
                for (c, &(bits, maxv)) in comps.iter().enumerate() {
                    let code = (words[j] >> shift) & ((1 << bits) - 1);
                    shift += bits;
                    let v = if signed && code >> (bits - 1) == 1 { code as i64 - (1 << bits) } else { code as i64 };
                    let want = if half {
                        let m = f16_nearest(v.unsigned_abs(), maxv as u64);
                        let m = if v < 0 { m | 0x8000 } else { m };
                        // max(-1, ...)
                        if v < 0 && m > 0xbc00 { 0xbc00 } else { m }
                    } else {
                        let q = v as f32 / maxv as f32;
                        if q < -1.0 { 0xbf80_0000 } else { q.to_bits() }
                    };
                    assert_eq!(g[c], want, "{} code {} component {}", name, code, c);
                }
            }
            eprintln!("{}: every code exact", name);
        }
    }
}

#[test]
fn srgb_conversions() {
    // unpack: every code against the binary64 power, which is within 2^-50 of the exact value: the binary32 and
    // binary16 results must be the nearest values unless the binary64 value is that close to a midpoint.
    for half in [false, true] {
        let vt = Ty::vector(if half { Scalar::F16 } else { Scalar::F32 }, 4);
        let name = format!("air.unpack.unorm4x8.srgb.{}", vt.suffix());
        let k = kernel(&name, &[Ty::scalar(Scalar::I32)], vt, 1);
        let words: Vec<u32> = (0..256u32).map(|c| c | c << 8 | c << 16 | c << 24).collect();
        let got = run(&k, &words, 1, 4);
        for c in 0..256usize {
            let x = c as f64 / 255.0;
            let lin = if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) };
            let want = if half { r::f32_to_f16((lin as f32).to_bits()) as u32 } else { (lin as f32).to_bits() };
            for comp in 0..3 {
                assert_eq!(got[c][comp], want, "{} code {}", name, c);
            }
            // Alpha is linear (MSL 8.7.7).
            let a = if half { f16_nearest(c as u64, 255) } else { (c as f32 / 255.0).to_bits() };
            assert_eq!(got[c][3], a, "{} alpha {}", name, c);
        }
        eprintln!("{}: every code", name);
    }
    // pack: every binary32 value in [0, 1] in steps, plus every threshold and its neighbours, against binary64 with
    // MSL's rule floor(255 c + 0.5) where binary64 decides it, and MSL's precision bound (|exact - result| < 1)
    // everywhere.
    let k = kernel("air.pack.unorm4x8.srgb.v4f32", &[Ty::vector(Scalar::F32, 4)], Ty::scalar(Scalar::I32), 4);
    let mut xs: Vec<u32> = (0..=0x3f80_0000u32).step_by(61).collect();
    xs.extend([0x8000_0000u32, 0xbf80_0000, 0x3f80_0001, 0x7f80_0000, 0x7fc0_0000, 0xff80_0000]);
    let inputs: Vec<u32> = xs.iter().flat_map(|&x| [x, x, x, x]).collect();
    let got = run(&k, &inputs, 4, 1);
    let srgb = |x: f64| if x < 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    let mut decided = 0u64;
    let bad = std::sync::Mutex::new(Vec::new());
    par_range(xs.len(), |js| {
        for j in js {
            let x = r::f(xs[j]);
            let c = if x.is_nan() || x <= 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { srgb(x as f64) };
            let v = 255.0 * c + 0.5;
            let code = got[j][0] & 0xff;
            let alpha = got[j][0] >> 24;
            let ok_bound = (255.0 * c - code as f64).abs() < 1.0;
            let ok_rule = (v - v.floor()).abs() < 1e-9 || code == v.floor() as u32;
            if !(ok_bound && ok_rule && alpha == unorm_code(xs[j], 255) && got[j][0] & 0xff_ffff == code * 0x01_0101) {
                bad.lock().unwrap().push(format!("{:08x}: {:08x}, binary64 {}", xs[j], got[j][0], v));
            }
        }
    });
    decided += xs.len() as u64;
    let bad = bad.into_inner().unwrap();
    assert!(bad.is_empty(), "{}", bad[..bad.len().min(8)].join("\n"));
    eprintln!("air.pack.unorm4x8.srgb.v4f32: {} values", decided);
}

// Against Metal ---------------------------------------------------------------------------------------------------

struct Measured {
    ins: Vec<u32>,
    outs: Vec<u32>,
}

fn measured(name: &str) -> Vec<Measured> {
    DATA.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (lhs, rhs) = l.split_once(" -> ")?;
            let mut it = lhs.split_whitespace();
            if it.next()? != name {
                return None;
            }
            let hex = |s: &str| u32::from_str_radix(s, 16).unwrap();
            Some(Measured { ins: it.map(hex).collect(), outs: rhs.split_whitespace().map(hex).collect() })
        })
        .collect()
}

/// Every measured pack and unpack case: equal bits, or (for unpack) Metal within MSL's 1.5-ulp bound of the exact
/// quotient where the library has the nearest value; counted per function.
#[test]
fn pack_and_unpack_match_metal() {
    let mut caps = BTreeMap::new();
    let packs = [
        ("pack_float_to_unorm4x8_f4", "air.pack.unorm4x8.v4f32", 4u32, false),
        ("pack_float_to_snorm4x8_f4", "air.pack.snorm4x8.v4f32", 4, false),
        ("pack_float_to_srgb_unorm4x8_f4", "air.pack.unorm4x8.srgb.v4f32", 4, false),
        ("pack_float_to_unorm10a2_f4", "air.pack.unorm.rgb10a2.v4f32", 4, false),
        ("pack_half_to_unorm4x8_h4", "air.pack.unorm4x8.v4f16", 4, true),
        ("pack_half_to_snorm4x8_h4", "air.pack.snorm4x8.v4f16", 4, true),
        ("pack_half_to_unorm10a2_h4", "air.pack.unorm.rgb10a2.v4f16", 4, true),
        ("pack_float_to_unorm2x16_f2", "air.pack.unorm2x16.v2f32", 2, false),
        ("pack_float_to_snorm2x16_f2", "air.pack.snorm2x16.v2f32", 2, false),
        ("pack_half_to_unorm2x16_h2", "air.pack.unorm2x16.v2f16", 2, true),
        ("pack_half_to_snorm2x16_h2", "air.pack.snorm2x16.v2f16", 2, true),
        ("pack_float_to_unorm565_f3", "air.pack.unorm.rgb565.v3f32", 3, false),
        ("pack_half_to_unorm565_h3", "air.pack.unorm.rgb565.v3f16", 3, true),
    ];
    for (mname, air, n, half) in packs {
        let cases = measured(mname);
        let et = if half { Scalar::F16 } else { Scalar::F32 };
        let ret = if air.contains("565") { Scalar::I16 } else { Scalar::I32 };
        let k = kernel(air, &[Ty::vector(et, n)], Ty::scalar(ret), n);
        // The measured inputs are float4; a half kernel converted them with half(float4) (round to nearest even).
        let inputs: Vec<u32> = cases
            .iter()
            .flat_map(|c| c.ins[..n as usize].iter().map(move |&x| if half { r::f32_to_f16(x) as u32 } else { x }))
            .collect();
        let got = run(&k, &inputs, n as usize, 1);
        let mut differ = Vec::new();
        let mut within = 0;
        for (c, g) in cases.iter().zip(&got) {
            if g[0] == c.outs[0] {
                continue;
            }
            // 565: Metal rounds the product to binary32 first; MSL 8.7.1.2 asks the correctly rounded code.
            if mname.contains("565") && !half {
                let mut w = 0;
                let mut shift = 0;
                for (k, (bits, maxv)) in [(5u32, 31u32), (6, 63), (5, 31)].into_iter().enumerate() {
                    w |= code_565_metal(c.ins[k], maxv) << shift;
                    shift += bits;
                }
                if w == c.outs[0] {
                    within += 1;
                    continue;
                }
            }
            // sRGB: MSL 8.7.7 states the procedure floor(255 c + 0.5), which the library follows exactly, and bounds
            // an implementation's result by |255 c - result| < 1, which Metal's codes meet.
            if mname.contains("srgb") {
                let srgb = |x: f64| if x < 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
                let ok = (0..3).all(|k| {
                    let x = r::f(c.ins[k]) as f64;
                    let cl = if x.is_nan() || x <= 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { srgb(x) };
                    (255.0 * cl - ((c.outs[0] >> (8 * k)) & 0xff) as f64).abs() < 1.0
                }) && c.outs[0] >> 24 == g[0] >> 24;
                if ok {
                    within += 1;
                    continue;
                }
            }
            differ.push(format!("{} {:08x?}: Metal {:08x}, library {:08x}", mname, &c.ins[..n as usize], c.outs[0], g[0]));
        }
        caps.insert(mname, (cases.len(), within));
        assert!(differ.is_empty(), "{}", differ[..differ.len().min(10)].join("\n"));
    }
    let unpacks = [
        ("unpack_unorm4x8_to_float", "air.unpack.unorm4x8.v4f32", 4u32, false, 255u32),
        ("unpack_snorm4x8_to_float", "air.unpack.snorm4x8.v4f32", 4, false, 127),
        ("unpack_unorm4x8_srgb_to_float", "air.unpack.unorm4x8.srgb.v4f32", 4, false, 255),
        ("unpack_unorm2x16_to_float", "air.unpack.unorm2x16.v2f32", 2, false, 65535),
        ("unpack_snorm2x16_to_float", "air.unpack.snorm2x16.v2f32", 2, false, 32767),
        ("unpack_unorm10a2_to_float", "air.unpack.unorm.rgb10a2.v4f32", 4, false, 1023),
        ("unpack_unorm4x8_to_half", "air.unpack.unorm4x8.v4f16", 4, true, 255),
        ("unpack_snorm4x8_to_half", "air.unpack.snorm4x8.v4f16", 4, true, 127),
        ("unpack_unorm4x8_srgb_to_half", "air.unpack.unorm4x8.srgb.v4f16", 4, true, 255),
        ("unpack_unorm2x16_to_half", "air.unpack.unorm2x16.v2f16", 2, true, 65535),
        ("unpack_snorm2x16_to_half", "air.unpack.snorm2x16.v2f16", 2, true, 32767),
        ("unpack_unorm10a2_to_half", "air.unpack.unorm.rgb10a2.v4f16", 4, true, 1023),
        ("unpack_unorm565_to_float", "air.unpack.unorm.rgb565.v3f32", 3, false, 31),
        ("unpack_unorm565_to_half", "air.unpack.unorm.rgb565.v3f16", 3, true, 31),
    ];
    for (mname, air, n, half, _) in unpacks {
        let cases = measured(mname);
        let et = if half { Scalar::F16 } else { Scalar::F32 };
        let pt = if air.contains("565") { Scalar::I16 } else { Scalar::I32 };
        let k = kernel(air, &[Ty::scalar(pt)], Ty::vector(et, n), 1);
        let inputs: Vec<u32> = cases.iter().map(|c| if pt == Scalar::I16 { c.ins[0] & 0xffff } else { c.ins[0] }).collect();
        let got = run(&k, &inputs, 1, n as usize);
        let mut within = 0;
        let mut differ = Vec::new();
        for (c, g) in cases.iter().zip(&got) {
            for comp in 0..n as usize {
                let (m, o) = (c.outs[comp], g[comp]);
                if m == o {
                    continue;
                }
                let val = |b: u32| if half { r::f(r::f16_to_f32(b as u16)) as f64 } else { r::f(b) as f64 };
                // sRGB to linear: MSL 8.7.7 bounds the result by converting it back, |r - c| <= 0.5 in code units.
                if mname.contains("srgb") && comp < 3 {
                    let x = val(m);
                    let back = 255.0 * if x < 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
                    let code = (c.ins[0] >> (8 * comp)) & 0xff;
                    if (back - code as f64).abs() <= 0.5 {
                        within += 1;
                        continue;
                    }
                }
                // Metal's value within MSL Table 8.6's 1.5 ulp of the library's correctly rounded one.
                let ulp = if half { crate::math::ulp16(val(o)) } else { crate::math::ulp32(val(o)) };
                if (val(m) - val(o)).abs() <= 1.5 * ulp {
                    within += 1;
                } else {
                    differ.push(format!("{} {:08x} component {}: Metal {:08x}, library {:08x}", mname, c.ins[0], comp, m, o));
                }
            }
        }
        caps.insert(mname, (cases.len(), within));
        assert!(differ.is_empty(), "{}", differ[..differ.len().min(10)].join("\n"));
    }
    for (k, (n, d)) in caps {
        eprintln!("{}: {} cases match Metal ({} explained differences: Metal's own rounding within MSL's bound, or its 565 product rounding)", k, n, d);
    }
}

/// The integer edges MSL leaves undefined or implementation-defined, against Metal. Where MSL defines the result
/// the library must match; where it does not (extract_bits and insert_bits with offset + bits past the width) the
/// cases are reported, not required.
#[test]
fn integer_edges_match_metal() {
    let i32t = Ty::scalar(Scalar::I32);
    let i16t = Ty::scalar(Scalar::I16);
    let i8t = Ty::scalar(Scalar::I8);
    let mut undefined = 0;
    for (mname, air, tys, ret, defined) in [
        ("rotate_i32", "air.rotate.i32", vec![i32t, i32t], i32t, None),
        ("rotate_u16", "air.rotate.i16", vec![i16t, i16t], i16t, None),
        ("rotate_i8", "air.rotate.i8", vec![i8t, i8t], i8t, None),
        ("mul24_i32", "air.mul24.s.i32", vec![i32t, i32t], i32t, None),
        ("mul24_u32", "air.mul24.u.i32", vec![i32t, i32t], i32t, None),
        ("clz_u32", "air.clz.i32", vec![i32t, Ty::scalar(Scalar::I1)], i32t, None),
        ("ctz_u16", "air.ctz.i16", vec![i16t, Ty::scalar(Scalar::I1)], i16t, None),
        ("abs_i32", "air.abs.s.i32", vec![i32t], i32t, None),
        ("absdiff_i32", "air.abs_diff.s.i32", vec![i32t, i32t], i32t, None),
        ("hadd_i32", "air.hadd.s.i32", vec![i32t, i32t], i32t, None),
        ("rhadd_i32", "air.rhadd.s.i32", vec![i32t, i32t], i32t, None),
        ("mulhi_i32", "air.mul_hi.s.i32", vec![i32t, i32t], i32t, None),
        ("extract_bits_u32", "air.extract_bits.u.i32", vec![i32t, i32t, i32t], i32t, Some(1)),
        ("extract_bits_i32", "air.extract_bits.s.i32", vec![i32t, i32t, i32t], i32t, Some(1)),
        ("insert_bits_u32", "air.insert_bits.u.i32", vec![i32t, i32t, i32t, i32t], i32t, Some(2)),
    ] {
        let cases = measured(mname);
        let stride = 4;
        let k = kernel(air, &tys, ret, stride);
        let mut inputs = Vec::new();
        for c in &cases {
            let (u, v) = (c.ins[0], c.ins[1]);
            let w: Vec<u32> = match mname {
                "rotate_u16" => vec![u & 0xffff, v & 0xffff, 0, 0],
                "rotate_i8" => vec![u & 0xff, v & 0xff, 0, 0],
                "ctz_u16" => vec![u & 0xffff, 0, 0, 0],
                "clz_u32" | "abs_i32" => vec![u, 0, 0, 0],
                "extract_bits_u32" | "extract_bits_i32" => vec![u, v & 63, (v >> 8) & 63, 0],
                "insert_bits_u32" => vec![u, 0xffff_ffff, v & 63, (v >> 8) & 63],
                _ => vec![u, v, 0, 0],
            };
            inputs.extend(w);
        }
        let got = run(&k, &inputs, stride as usize, 1);
        let mut differ = Vec::new();
        for (j, (c, g)) in cases.iter().zip(&got).enumerate() {
            if g[0] == c.outs[0] {
                continue;
            }
            if let Some(off_at) = defined {
                let w = &inputs[j * stride as usize..];
                if w[off_at] + w[off_at + 1] > 32 {
                    undefined += 1;
                    continue;
                }
            }
            differ.push(format!("{} {:08x?}: Metal {:08x}, library {:08x}", mname, c.ins, c.outs[0], g[0]));
        }
        assert!(differ.is_empty(), "{}", differ[..differ.len().min(10)].join("\n"));
        eprintln!("{}: {} cases", mname, cases.len());
    }
    eprintln!("extract_bits and insert_bits past the width (undefined in MSL Table 6.2): {} cases differ", undefined);
}
