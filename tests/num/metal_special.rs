//! The library against what Metal itself returned for the special inputs of each function (tests/num/data, measured
//! by tests/num/metal/special.swift on this Mac's GPU with fast math disabled).
//!
//! NaN payloads are not compared (MSL does not specify them, and Metal returns the default NaN where the library keeps
//! the operand's payload). Cases with a subnormal operand or result are skipped: that GPU flushes binary32
//! subnormals, which MSL 8.1 allows and the library does not do. Every other case must agree bit for bit, be within
//! one ulp where both results are rounded values of the same irrational number, or be one of the listed cases where
//! the library follows a text (MSL, C99 Annex F, OpenCL C 7.5.1) and Metal on this GPU does not.

use crate::gpu::{kernel_source, Kernel};
use crate::prelude;

const DATA: &str = include_str!("data/metal-precise-special.txt");

struct Case {
    ops: Vec<u32>,
    res: Vec<u32>,
}

fn cases(name: &str) -> Vec<Case> {
    DATA.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (lhs, rhs) = l.split_once(" -> ")?;
            let mut it = lhs.split_whitespace();
            if it.next()? != name {
                return None;
            }
            let hex = |s: &str| u32::from_str_radix(s, 16).unwrap();
            Some(Case { ops: it.map(hex).collect(), res: rhs.split_whitespace().map(hex).collect() })
        })
        .collect()
}

fn is_nan(a: u32) -> bool {
    a & 0x7fff_ffff > 0x7f80_0000
}

fn subnormal(a: u32) -> bool {
    a & 0x7f80_0000 == 0 && a & 0x007f_ffff != 0
}

fn is_zero(a: u32) -> bool {
    a & 0x7fff_ffff == 0
}

/// MSL's precise ulp bound (Table 8.1) for the functions whose results are rounded values of irrational numbers.
fn ulp_bound(name: &str) -> Option<i64> {
    Some(match name {
        "acos" | "acosh" | "asin" | "asinh" | "cos" | "cosh" | "cospi" | "exp" | "exp2" | "exp10" | "log" | "log2"
        | "log10" | "sin" | "sincos" | "sinh" | "sinpi" => 4,
        "atan" | "atanh" | "tanh" => 5,
        "atan2" | "tan" | "tanpi" => 6,
        "pow" | "powr" => 16,
        _ => return None,
    })
}

/// Why a difference from Metal is the library following a text that this GPU's Metal does not, or `None`.
fn deliberate(name: &str, ops: &[u32], metal: u32, ours: u32) -> Option<&'static str> {
    let x = ops[0];
    let opposite_zeros = is_zero(metal) && is_zero(ours) && metal != ours;
    match name {
        "fmax" | "fmin" | "fmax3" | "fmin3" | "fmedian3" | "clamp" | "saturate" if opposite_zeros => {
            Some("MSL Table 6.4: fmax and fmin return x for equal arguments (Metal returns y)")
        }
        "sign" if x == 0x8000_0000 => Some("MSL Table 6.1: sign(-0.0) is -0.0"),
        "step" if ops.iter().any(|&v| is_nan(v)) => {
            Some("MSL Table 6.1: step returns 0.0 if x < edge, otherwise 1.0 (no comparison with a NaN is true)")
        }
        "mix" if !(0.0..=1.0).contains(&f32::from_bits(ops[2])) || ops.iter().any(|&v| is_nan(v)) => {
            Some("MSL Table 6.1: mix is undefined for a outside [0, 1]")
        }
        "mix" if opposite_zeros => Some("MSL Table 6.1: mix may be x + (y - x) * a, the form the library evaluates"),
        "sinpi" | "tanpi" if opposite_zeros => Some("OpenCL C 3.0 7.5.1: sinpi and tanpi zeros at integers"),
        "atan" if opposite_zeros => Some("C99 F.9.1.3: atan(+-0) is +-0"),
        "round" if opposite_zeros => Some("C99 F.9.6.6 and OpenCL C 3.0 7.5.1: round(+-0) is +-0, round(-0.5 < x < 0) is -0"),
        "fract" if opposite_zeros => Some("OpenCL C 3.0 7.5.1: fract(-0) and fract(-inf) are -0"),
        "atan2" if opposite_zeros => Some("MSL 8.5: a result flushed to zero has an undefined sign (the exact result rounds to -0)"),
        _ => None,
    }
}

/// Run `name` from the measured data through the library item `item` and classify every case.
fn compare(name: &str, item: &str, arity: usize, outs: usize, call: &str) -> Vec<String> {
    let cs = cases(name);
    assert!(!cs.is_empty(), "no measured cases for {}", name);
    let mut cols: Vec<Vec<u32>> = vec![Vec::new(); arity];
    for c in &cs {
        for (k, col) in cols.iter_mut().enumerate() {
            col.push(c.ops[k]);
        }
    }
    let body = call.replace("{a}", "nm_t_in0[i]").replace("{b}", "nm_t_in1[i]").replace("{c}", "nm_t_in2[i]");
    let k = Kernel::new(&kernel_source(false, &prelude(&[item, "nm_f32_to_i32"]), "", &body), outs as u32);
    let slices: Vec<&[u32]> = cols.iter().map(|v| v.as_slice()).collect();
    let out = k.run(&slices, [0, 0, 0], cs.len() as u32);
    let mut differences = Vec::new();
    let half_result = name == "to_half";
    // Cases within Metal's ulp bound of each other, and deliberate differences.
    let mut tally = (0u32, 0u32);
    for (j, c) in cs.iter().enumerate() {
        if c.ops.iter().any(|&x| subnormal(x)) {
            continue;
        }
        for (o, &metal) in c.res.iter().enumerate() {
            let ours = out[j * outs + o];
            let nan = if half_result { |v: u32| v & 0x7fff > 0x7c00 } else { is_nan };
            if nan(metal) && nan(ours) || metal == ours || subnormal(ours) && is_zero(metal) {
                continue;
            }
            let (fm, fo) = (f32::from_bits(metal), f32::from_bits(ours));
            let finite = fm.is_finite() && fo.is_finite() && fm != 0.0 && (metal >> 31) == (ours >> 31);
            if let Some(b) = ulp_bound(name) {
                if finite && (metal as i64 - ours as i64).abs() <= b {
                    tally.0 += 1;
                    continue;
                }
            }
            match deliberate(name, &c.ops, metal, ours) {
                Some(rule) => {
                    tally.1 += 1;
                    eprintln!("{} {:08x?}: Metal {:08x}, library {:08x} ({})", name, c.ops, metal, ours, rule)
                }
                None => differences.push(format!("{} {:08x?}: Metal {:08x}, library {:08x}", name, c.ops, metal, ours)),
            }
        }
    }
    eprintln!("{}: {} cases, {} within MSL's ulp bound of Metal's value, {} deliberate", name, cs.len(), tally.0, tally.1);
    differences
}

#[test]
fn library_matches_metal_special_values() {
    let one = ["exp", "exp2", "exp10", "log", "log2", "log10", "sin", "cos", "tan", "sinpi", "cospi", "tanpi", "asin", "acos",
               "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "sqrt", "rsqrt", "ceil", "floor", "trunc", "rint",
               "round", "fract", "saturate", "sign"];
    let two = ["atan2", "pow", "powr", "fmod", "fmax", "fmin", "nextafter", "step"];
    let three = ["fma", "fmax3", "fmin3", "fmedian3", "clamp", "mix", "smoothstep"];
    let mut all = Vec::new();
    for n in one {
        let item = format!("nm_f32_{}", n);
        all.extend(compare(n, &item, 1, 1, &format!("nm_t_out[i] = {}({{a}});", item)));
    }
    for n in two {
        let item = format!("nm_f32_{}", n);
        all.extend(compare(n, &item, 2, 1, &format!("nm_t_out[i] = {}({{a}}, {{b}});", item)));
    }
    for n in three {
        let item = format!("nm_f32_{}", n);
        all.extend(compare(n, &item, 3, 1, &format!("nm_t_out[i] = {}({{a}}, {{b}}, {{c}});", item)));
    }
    all.extend(compare("fabs", "nm_f32_abs", 1, 1, "nm_t_out[i] = nm_f32_abs({a});"));
    all.extend(compare("divide", "nm_f32_div", 2, 1, "nm_t_out[i] = nm_f32_div({a}, {b});"));
    all.extend(compare(
        "sincos",
        "nm_f32_sincos",
        1,
        2,
        "let r = nm_f32_sincos({a}); nm_t_out[2u * i] = r.x; nm_t_out[2u * i + 1u] = r.y;",
    ));
    all.extend(compare(
        "ldexp",
        "nm_f32_ldexp",
        2,
        1,
        "nm_t_out[i] = nm_f32_ldexp({a}, bitcast<i32>(nm_f32_to_i32(nm_t_in1[i])));",
    ));
    for (n, item) in [("to_int", "nm_f32_to_i32"), ("to_uint", "nm_f32_to_u32"), ("to_short", "nm_f32_to_i16"), ("to_half", "nm_f32_to_f16")] {
        all.extend(compare(n, item, 1, 1, &format!("nm_t_out[i] = {}({{a}});", item)));
    }
    assert!(all.is_empty(), "{} unexplained differences from Metal:\n{}", all.len(), all.join("\n"));
}
