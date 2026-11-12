//! The lowering rules: every name of the numeric families in tables/intrinsics.toml lowers to valid WGSL or to a
//! refusal that says why, the decisions follow numerics.toml and the policy, and lowered code computes what MSL
//! defines (integer functions, conversions and comparisons against references written here from MSL Table 6.2, MSL
//! 2.23 and 8.6, and IEEE comparison semantics).

use crate::gpu::{Gpu, Kernel, MAX_WORDS, par_range};
use crate::inputs::{Rng, SPECIAL_F32, f32_any};
use crate::reference as r;
use airwgsl::api::{HostCaps, RefusalKind};
use airwgsl::num::{
    Column, FPred, FastMath, How, Inst, Lowered, Op, Operand, Rounding, Scalar, Target, Ty, lower, selftest,
};
use std::cmp::Ordering;
use std::collections::BTreeMap;

fn host(f16: bool) -> HostCaps {
    let mut h = HostCaps::default();
    if f16 {
        h.features.insert("shader-f16".into());
    }
    h
}

fn words(t: Ty) -> u32 {
    t.lanes * if t.scalar == Scalar::I64 { 2 } else { 1 }
}

/// A carrier value of `t` read from `nm_t_in0` at word `base`.
fn load(t: Ty, base: u32) -> String {
    let w = |k: u32| format!("nm_t_in0[i * nm_t_p.y + {}u]", base + k);
    let elem = |j: u32| match t.scalar {
        Scalar::I1 => format!("({} != 0u)", w(j)),
        Scalar::I64 => format!("vec2<u32>({}, {})", w(2 * j), w(2 * j + 1)),
        _ => w(j),
    };
    if t.lanes == 1 {
        elem(0)
    } else {
        format!("{}({})", t.carrier(), (0..t.lanes).map(elem).collect::<Vec<_>>().join(", "))
    }
}

/// Statements storing a carrier value of `t` to `nm_t_out` at the item's words.
fn store(t: Ty, v: &str) -> String {
    let n = words(t);
    let mut s = String::new();
    for j in 0..t.lanes {
        let e = if t.lanes == 1 { v.to_string() } else { format!("{}[{}]", v, j) };
        match t.scalar {
            Scalar::I1 => s += &format!("nm_t_out[i * {}u + {}u] = select(0u, 1u, {});\n", n, j, e),
            Scalar::I64 => {
                s += &format!("nm_t_out[i * {}u + {}u] = ({}).x;\n", n, 2 * j, e);
                s += &format!("nm_t_out[i * {}u + {}u] = ({}).y;\n", n, 2 * j + 1, e);
            }
            _ => s += &format!("nm_t_out[i * {}u + {}u] = {};\n", n, j, e),
        }
    }
    s
}

/// Lower `op` over operands of `tys` and run it on the GPU for every item of `inputs` (each the operands' words,
/// concatenated). Returns the result words per item.
fn run_lowered(
    op: Op,
    tys: &[Ty],
    ret: Ty,
    flags: FastMath,
    rounding: &Rounding,
    inputs: &[Vec<u32>],
) -> (Lowered, Vec<Vec<u32>>) {
    run_lowered_on(&host(Gpu::get().has_f16), op, tys, ret, flags, rounding, inputs)
}

fn run_lowered_on(
    caps: &HostCaps,
    op: Op,
    tys: &[Ty],
    ret: Ty,
    flags: FastMath,
    rounding: &Rounding,
    inputs: &[Vec<u32>],
) -> (Lowered, Vec<Vec<u32>>) {
    let target = Target { rounding, host: caps };
    let mut base = 0;
    let mut exprs = Vec::new();
    for t in tys {
        exprs.push(load(*t, base));
        base += words(*t);
    }
    let operands: Vec<Operand> =
        exprs.iter().zip(tys).map(|(e, t)| Operand { expr: e, ty: *t, constant: None }).collect();
    let l = lower(op, &operands, Some(ret), flags, target, "nm_v").unwrap_or_else(|e| panic!("{:?}: {}", op, e));
    let body = format!("{}\n{}", l.stmts.join("\n"), store(ret, l.value.as_deref().unwrap()));
    let enable = l.enables.contains("f16");
    let src = crate::gpu::kernel_source(enable, &l.prelude.emit().unwrap(), "", &body);
    let k = Kernel::new(&src, words(ret));
    // Items may carry more words than the operands use; the stride is the item length.
    let stride = inputs.first().map_or(base, |i| i.len() as u32);
    assert!(stride >= base);
    let mut results = Vec::new();
    let per_run = (MAX_WORDS / stride.max(words(ret)).max(1) as u64) as usize;
    for chunk in inputs.chunks(per_run) {
        let flat: Vec<u32> = chunk.iter().flatten().copied().collect();
        let out = k.run(&[&flat], [stride, 0, 0], chunk.len() as u32);
        results.extend(out.chunks(words(ret) as usize).map(|c| c.to_vec()));
    }
    (l, results)
}

fn check(what: &str, inputs: &[Vec<u32>], got: &[Vec<u32>], want: impl Fn(&[u32]) -> Vec<u32> + Sync) {
    let bad = std::sync::Mutex::new((0u64, Vec::new()));
    par_range(inputs.len(), |js| {
        let mut local = (0u64, Vec::new());
        for j in js {
            let w = want(&inputs[j]);
            if w != got[j] {
                local.0 += 1;
                if local.1.len() < 8 {
                    local.1.push(format!("{}: in {:08x?} got {:08x?} want {:08x?}", what, inputs[j], got[j], w));
                }
            }
        }
        let mut g = bad.lock().unwrap();
        g.0 += local.0;
        g.1.extend(local.1);
    });
    let (n, first) = bad.into_inner().unwrap();
    assert_eq!(n, 0, "{} of {} differ\n{}", n, inputs.len(), first.join("\n"));
}

// Every name of the numeric families -----------------------------------------------------------------------------

fn parse_suffix(s: &str) -> Option<Ty> {
    let (lanes, rest) = match s.strip_prefix('v') {
        Some(rest) => {
            let end = rest.find(|c: char| !c.is_ascii_digit())?;
            (rest[..end].parse().ok()?, &rest[end..])
        }
        None => (1, s),
    };
    let sc = match rest {
        "i1" => Scalar::I1,
        "i8" => Scalar::I8,
        "i16" => Scalar::I16,
        "i32" => Scalar::I32,
        "i64" => Scalar::I64,
        "f16" => Scalar::F16,
        "f32" => Scalar::F32,
        _ => return None,
    };
    Some(Ty::vector(sc, lanes))
}

/// Operand and result types of an air name, as AIR declares them (the declarations probed with the Metal compiler:
/// extract_bits and insert_bits take i32 offset and count, clz and ctz an i1 flag, ldexp an integer vector, sincos a
/// pointer, dot returns the scalar).
fn signature(name: &str) -> Option<(Vec<Ty>, Ty)> {
    let rest = name.strip_prefix("air.")?;
    let parts: Vec<&str> = rest.split('.').collect();
    let head = parts[0].strip_prefix("fast_").unwrap_or(parts[0]);
    let last = parse_suffix(parts.last()?)?;
    let i32t = Ty::scalar(Scalar::I32);
    let n = |k: usize| vec![last; k];
    Some(match head {
        "pack" | "unpack" => {
            let fmt = parts[1..parts.len() - 1].join(".");
            let packed = Ty::scalar(if fmt == "unorm.rgb565" { Scalar::I16 } else { Scalar::I32 });
            if head == "pack" { (vec![last], packed) } else { (vec![packed], last) }
        }
        "convert" => {
            if parts.len() != 5 {
                return None;
            }
            (vec![parse_suffix(parts[4])?], parse_suffix(parts[2])?)
        }
        "dot" => (n(2), last.element()),
        "all" | "any" => (n(1), Ty::scalar(Scalar::I1)),
        "dfdx" | "dfdy" | "fwidth" => (n(1), last),
        "sincos" => (vec![last, last], last),
        "ldexp" => (vec![last, Ty::vector(Scalar::I32, last.lanes)], last),
        "atan2" | "fmax" | "fmin" | "fmod" | "pow" | "powr" | "nextafter" | "copysign" | "step" | "fdim" | "divide" => {
            (n(2), last)
        }
        "fma" | "fmax3" | "fmin3" | "fmedian3" | "mix" | "smoothstep" => (n(3), last),
        "clamp" if last.scalar.is_float() => (n(3), last),
        "clz" | "ctz" => (vec![last, Ty::scalar(Scalar::I1)], last),
        "popcount" | "reverse_bits" | "abs" => (n(1), last),
        "rotate" | "abs_diff" | "add_sat" | "sub_sat" | "hadd" | "rhadd" | "mul_hi" | "max" | "min" | "mul24" => {
            (n(2), last)
        }
        "clamp" | "mad_hi" | "mad_sat" | "max3" | "min3" | "median3" | "mad24" => (n(3), last),
        "extract_bits" => (vec![last, i32t, i32t], last),
        "insert_bits" => (vec![last, last, i32t, i32t], last),
        _ if parts.len() == 2 => (n(1), last),
        _ => return None,
    })
}

const NUMERIC_FAMILIES: &[&str] =
    &["math-float", "math-geometric", "integer", "conversion", "relational", "derivative", "pack-unpack"];

/// Every name of the numeric families in tables/intrinsics.toml (or the file NM_INTRINSICS_TOML names), with its
/// family. The table lists names one per [[name]] row, or, in its grammar form, as [[function]] names followed by
/// one type token per slot from the slot's [[domain]] (tokens whose element a domain reclassifies into another
/// family, such as f64, are left out).
fn numeric_names() -> Vec<(String, String)> {
    let path = std::env::var("NM_INTRINSICS_TOML")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/tables/intrinsics.toml").into());
    let table: toml::Value = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let arr = |k: &str| table.get(k).and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let s = |v: &toml::Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let mut out = Vec::new();
    for n in arr("name") {
        let fam = s(&n, "family");
        if NUMERIC_FAMILIES.contains(&fam.as_str()) {
            out.push((s(&n, "name"), fam));
        }
    }
    let domains = arr("domain");
    let families = arr("family");
    for f in arr("function") {
        let fam = s(&f, "family");
        if !NUMERIC_FAMILIES.contains(&fam.as_str()) {
            continue;
        }
        let slots: Vec<String> = f
            .get("slots")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let mut names = vec![s(&f, "name")];
        for kind in slots {
            let fam_row = families.iter().find(|x| s(x, "id") == fam);
            let dom_id = fam_row
                .and_then(|r| r.get("domains"))
                .and_then(|d| d.get(&kind))
                .and_then(|v| v.as_str())
                .map(String::from);
            let dom = domains
                .iter()
                .find(|d| match &dom_id {
                    Some(id) => &s(d, "id") == id,
                    None => d
                        .get("default_for")
                        .and_then(|v| v.as_array())
                        .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(kind.as_str()))),
                })
                .unwrap_or_else(|| panic!("no domain for slot {}", kind));
            let reclassified: Vec<String> = dom
                .get("reclassify")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|p| p.as_array().and_then(|p| p[0].as_str()).map(String::from)).collect())
                .unwrap_or_default();
            let mut tokens = Vec::new();
            for e in dom.get("elements").and_then(|v| v.as_array()).unwrap() {
                let e = e.as_str().unwrap();
                if reclassified.iter().any(|r| r == e) {
                    continue;
                }
                for w in dom.get("widths").and_then(|v| v.as_array()).unwrap() {
                    let w = w.as_integer().unwrap();
                    tokens.push(if w == 1 { e.to_string() } else { format!("v{}{}", w, e) });
                }
            }
            names = names.iter().flat_map(|n| tokens.iter().map(move |t| format!("{}.{}", n, t))).collect();
        }
        out.extend(names.into_iter().map(|n| (n, fam.clone())));
    }
    out
}

#[test]
fn every_numeric_name_lowers_to_valid_wgsl_or_a_refusal() {
    let names = numeric_names();
    assert!(names.len() > 1000, "{} names", names.len());
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    let mut module = String::from("enable f16;\n");
    let mut prelude = airwgsl::num::prelude::Prelude::new();
    let caps = host(true);
    let soft = Rounding::SoftFloat;
    for (k, (name, fam)) in names.iter().enumerate() {
        let Some((tys, ret)) = signature(name) else {
            // Bare stems are entries of the front end's name table; modules always carry the type suffix.
            tally.entry(format!("{} without a type suffix", fam)).and_modify(|n| *n += 1).or_insert(1);
            continue;
        };
        for flags in [FastMath::FAST, FastMath::NONE] {
            let params: Vec<String> = tys.iter().enumerate().map(|(j, t)| format!("p{}: {}", j, t.carrier())).collect();
            let exprs: Vec<String> = (0..tys.len()).map(|j| format!("p{}", j)).collect();
            let operands: Vec<Operand> =
                exprs.iter().zip(&tys).map(|(e, t)| Operand { expr: e, ty: *t, constant: None }).collect();
            match lower(Op::Call(name), &operands, Some(ret), flags, Target { rounding: &soft, host: &caps }, "t") {
                Ok(l) => {
                    prelude.extend(&l.prelude);
                    let f = format!(
                        "fn case_{}_{}({}) -> {} {{\n{}\nreturn {};\n}}\n",
                        k,
                        flags.finite_only() as u8,
                        params.join(", "),
                        ret.carrier(),
                        l.stmts.join("\n"),
                        l.value.unwrap()
                    );
                    module.push_str(&f);
                    *tally.entry(format!("{} lowered ({:?})", fam, l.rule.how)).or_insert(0) += 1;
                }
                Err(e) => {
                    *tally.entry(format!("{} refused: {}", fam, e.reason)).or_insert(0) += 1;
                    assert!(
                        e.reason.contains("rounding mode")
                            || e.reason.contains("mul24 and mad24")
                            || e.reason.contains("derivative of an integer")
                            || e.reason.contains("has no MSL function")
                            || e.reason.contains("packed format takes"),
                        "{}: unexpected refusal {}",
                        name,
                        e
                    );
                }
            }
        }
    }
    module.push_str(&prelude.emit().unwrap());
    let m = naga::front::wgsl::parse_str(&module).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&module)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&m)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&module)));
    // A second WGSL front end: Dawn's tint, when NM_TINT names its executable.
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("numerics-all-names.wgsl");
    std::fs::write(&path, &module).unwrap();
    if let Ok(tint) = std::env::var("NM_TINT") {
        let out = std::process::Command::new(&tint).arg("--format").arg("wgsl").arg(&path).output().unwrap();
        assert!(out.status.success(), "tint rejects the module: {}", String::from_utf8_lossy(&out.stderr));
        eprintln!("tint accepts {} ({} bytes)", path.display(), module.len());
    }
    for (k, v) in &tally {
        eprintln!("{:5} {}", v, k);
    }
}

#[test]
fn decisions_follow_the_table_and_the_policy() {
    let caps = host(true);
    let soft = Rounding::SoftFloat;
    let cases = selftest::cases();
    let want: Vec<u32> = cases.iter().map(|c| c.want).collect();
    let native = Rounding::Native(selftest::evaluate(&cases, &want).native_rounding().unwrap());
    let f32t = Ty::scalar(Scalar::F32);
    let f16t = Ty::scalar(Scalar::F16);
    let rule = |op: Op, n: usize, ty: Ty, flags: FastMath, rounding: &Rounding| {
        let ops: Vec<Operand> = (0..n).map(|_| Operand { expr: "x", ty, constant: None }).collect();
        lower(op, &ops, Some(ty), flags, Target { rounding, host: &caps }, "t").unwrap().rule
    };
    // Fast-math addition: polyfill, soft-float by default, native under a rounding proof.
    let r = rule(Op::Inst(Inst::FAdd), 2, f32t, FastMath::FAST, &soft);
    assert_eq!((r.row, r.column, r.how), (Some("add"), Column::FastF32, How::SoftFloat));
    assert_eq!(rule(Op::Inst(Inst::FAdd), 2, f32t, FastMath::FAST, &native).how, How::NativeRne);
    // Precise addition is soft-float under every policy.
    assert_eq!(rule(Op::Inst(Inst::FAdd), 2, f32t, FastMath::NONE, &native).how, How::SoftFloat);
    // A native row (division with fast math).
    assert_eq!(rule(Op::Inst(Inst::FDiv), 2, f32t, FastMath::FAST, &soft).how, How::Native);
    // A polyfill row that is not about rounding stays soft-float under the native policy.
    let r = rule(Op::Call("air.fast_sin.f32"), 1, f32t, FastMath::FAST, &native);
    assert_eq!((r.row, r.how), (Some("sin"), How::SoftFloat));
    // The precise variant by name, whatever the flags.
    let r = rule(Op::Call("air.sin.f32"), 1, f32t, FastMath::FAST, &soft);
    assert_eq!((r.column, r.how), (Column::PreciseF32, How::SoftFloat));
    // fma has no fast_ variant: the flags decide; the native policy uses fma only with a fused-rounding proof.
    assert_eq!(rule(Op::Call("air.fma.f32"), 3, f32t, FastMath::FAST, &native).how, How::NativeRne);
    // Half: native with fast flags, the exact library when NaN and infinity must be honored.
    assert_eq!(rule(Op::Call("air.sin.f16"), 1, f16t, FastMath::FAST, &soft).how, How::Native);
    assert_eq!(rule(Op::Call("air.sin.f16"), 1, f16t, FastMath::NONE, &soft).how, How::SoftFloat);
    // Explicit rounding conversions are refused.
    let e = lower(
        Op::Call("air.convert.s.i32.f.f32.rtz"),
        &[Operand { expr: "x", ty: f32t, constant: None }],
        Some(Ty::scalar(Scalar::I32)),
        FastMath::FAST,
        Target { rounding: &soft, host: &caps },
        "t",
    )
    .unwrap_err();
    assert_eq!(e.kind, RefusalKind::NotImplemented);
}

// Integer functions ------------------------------------------------------------------------------------------------

/// The value of an n-bit carrier word as signed or unsigned.
fn val(w: u64, n: u32, signed: bool) -> i128 {
    let m = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let w = w & m;
    if signed && w >> (n - 1) & 1 == 1 { w as i128 - (1i128 << n) } else { w as i128 }
}

fn wrap(v: i128, n: u32) -> u64 {
    let m = if n == 64 { u64::MAX as u128 } else { (1u128 << n) - 1 };
    (v as u128 & m) as u64
}

fn sat(v: i128, n: u32, signed: bool) -> u64 {
    let (lo, hi) = if signed { (-(1i128 << (n - 1)), (1i128 << (n - 1)) - 1) } else { (0, (1i128 << n) - 1) };
    wrap(v.clamp(lo, hi), n)
}

/// MSL Table 6.2, from its text.
fn int_ref(f: &str, n: u32, signed: bool, a: &[u64]) -> u64 {
    let v = |k: usize| val(a[k], n, signed);
    let floor_half = |s: i128| s.div_euclid(2);
    match f {
        "abs" => wrap(v(0).abs(), n),
        "abs_diff" => wrap((v(0) - v(1)).abs(), n),
        "add_sat" => sat(v(0) + v(1), n, signed),
        "sub_sat" => sat(v(0) - v(1), n, signed),
        "hadd" => wrap(floor_half(v(0) + v(1)), n),
        "rhadd" => wrap(floor_half(v(0) + v(1) + 1), n),
        // Unsigned 64-bit products need 128 bits unsigned, beyond i128.
        "mul_hi" if n == 64 && !signed => wrap(((a[0] as u128 * a[1] as u128) >> 64) as i128, n),
        "mad_hi" if n == 64 && !signed => wrap(((a[0] as u128 * a[1] as u128) >> 64) as i128 + v(2), n),
        "mad_sat" if n == 64 && !signed => {
            match (a[0] as u128).checked_mul(a[1] as u128).and_then(|p| p.checked_add(a[2] as u128)) {
                Some(p) if p <= u64::MAX as u128 => p as u64,
                _ => u64::MAX,
            }
        }
        "mul_hi" => wrap((v(0) * v(1)) >> n, n),
        "mad_hi" => wrap(((v(0) * v(1)) >> n) + v(2), n),
        "mad_sat" => sat(v(0) * v(1) + v(2), n, signed),
        "max" => wrap(v(0).max(v(1)), n),
        "min" => wrap(v(0).min(v(1)), n),
        "max3" => wrap(v(0).max(v(1).max(v(2))), n),
        "min3" => wrap(v(0).min(v(1).min(v(2))), n),
        "median3" => {
            let mut s = [v(0), v(1), v(2)];
            s.sort();
            wrap(s[1], n)
        }
        "clamp" => wrap(v(0).max(v(1)).min(v(2)), n),
        _ => unreachable!(),
    }
}

fn int_inputs(n: u32, arity: usize, rng: &mut Rng, count: usize) -> Vec<Vec<u64>> {
    let m = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let edge = [0u64, 1, 2, m, m - 1, m >> 1, (m >> 1) + 1, (m >> 1) - 1];
    (0..count)
        .map(|_| {
            (0..arity)
                .map(|_| if rng.below(3) == 0 { edge[rng.below(edge.len() as u32) as usize] } else { rng.next() & m })
                .collect()
        })
        .collect()
}

fn to_words(v: u64, n: u32) -> Vec<u32> {
    if n == 64 { vec![v as u32, (v >> 32) as u32] } else { vec![v as u32] }
}

/// Decode an item's operand words into per-lane values: operand k, lane l.
fn decode(words_in: &[u32], arity: usize, lanes: u32, n: u32) -> Vec<Vec<u64>> {
    let w = if n == 64 { 2 } else { 1 };
    (0..lanes as usize)
        .map(|l| {
            (0..arity)
                .map(|k| {
                    let at = (k * lanes as usize + l) * w;
                    if n == 64 { words_in[at] as u64 | (words_in[at + 1] as u64) << 32 } else { words_in[at] as u64 }
                })
                .collect()
        })
        .collect()
}

fn run_int(name: &str, tys: &[Ty], ret: Ty, items: Vec<Vec<u32>>, want: impl Fn(&[u32]) -> Vec<u32> + Sync) {
    let soft = Rounding::SoftFloat;
    let (_, got) = run_lowered(Op::Call(name), tys, ret, FastMath::NONE, &soft, &items);
    check(name, &items, &got, want);
    eprintln!("{}: {} items, 0 mismatches", name, items.len());
}

fn items_for(n: u32, arity: usize, lanes: u32, rng: &mut Rng, count: usize) -> Vec<Vec<u32>> {
    (0..count)
        .map(|_| {
            let vals = int_inputs(n, arity, rng, lanes as usize);
            (0..arity).flat_map(|k| vals.iter().flat_map(move |l| to_words(l[k], n))).collect()
        })
        .collect()
}

#[test]
fn integer_functions_match_msl_table_6_2() {
    let mut rng = Rng::new(31);
    for f in [
        "abs", "abs_diff", "add_sat", "sub_sat", "hadd", "rhadd", "mul_hi", "mad_hi", "mad_sat", "max", "min", "max3",
        "min3", "median3", "clamp",
    ] {
        let arity = match f {
            "abs" => 1,
            "mad_hi" | "mad_sat" | "max3" | "min3" | "median3" | "clamp" => 3,
            _ => 2,
        };
        for (sc, n) in [(Scalar::I8, 8), (Scalar::I16, 16), (Scalar::I32, 32), (Scalar::I64, 64)] {
            for signed in [true, false] {
                let lanes = [1, 3][rng.below(2) as usize];
                let ty = Ty::vector(sc, lanes);
                let name = format!("air.{}.{}.{}", f, if signed { "s" } else { "u" }, ty.suffix());
                let items = items_for(n, arity, lanes, &mut rng, 1 << 14);
                run_int(&name, &vec![ty; arity], ty, items, |w| {
                    decode(w, arity, lanes, n)
                        .iter()
                        .flat_map(|l| to_words(if f == "abs" && !signed { l[0] } else { int_ref(f, n, signed, l) }, n))
                        .collect()
                });
            }
        }
    }
}

#[test]
fn bit_functions_match_msl_table_6_2() {
    let mut rng = Rng::new(32);
    for (sc, n) in [(Scalar::I8, 8u32), (Scalar::I16, 16), (Scalar::I32, 32), (Scalar::I64, 64)] {
        let ty = Ty::vector(sc, 2);
        let flag = Ty::scalar(Scalar::I1);
        let m = |v: u64| if n == 64 { v } else { v & ((1u64 << n) - 1) };
        // clz and ctz: the bit count for a zero operand (MSL Table 6.2).
        let items: Vec<Vec<u32>> = items_for(n, 1, 2, &mut rng, 1 << 14)
            .into_iter()
            .map(|mut w| {
                w.push(0);
                w
            })
            .collect();
        run_int(&format!("air.clz.{}", ty.suffix()), &[ty, flag], ty, items.clone(), |w| {
            decode(w, 1, 2, n).iter().flat_map(|l| to_words((m(l[0]).leading_zeros() - (64 - n)) as u64, n)).collect()
        });
        run_int(&format!("air.ctz.{}", ty.suffix()), &[ty, flag], ty, items, |w| {
            decode(w, 1, 2, n).iter().flat_map(|l| to_words(m(l[0]).trailing_zeros().min(n) as u64, n)).collect()
        });
        let items = items_for(n, 1, 2, &mut rng, 1 << 14);
        run_int(&format!("air.popcount.{}", ty.suffix()), &[ty], ty, items.clone(), |w| {
            decode(w, 1, 2, n).iter().flat_map(|l| to_words(m(l[0]).count_ones() as u64, n)).collect()
        });
        run_int(&format!("air.reverse_bits.{}", ty.suffix()), &[ty], ty, items, |w| {
            decode(w, 1, 2, n).iter().flat_map(|l| to_words(l[0].reverse_bits() >> (64 - n), n)).collect()
        });
        // rotate: "Bits shifted off the left side of the element are shifted back in from the right", the count
        // taken modulo the width.
        let items = items_for(n, 2, 2, &mut rng, 1 << 14);
        run_int(&format!("air.rotate.{}", ty.suffix()), &[ty, ty], ty, items, |w| {
            decode(w, 2, 2, n)
                .iter()
                .flat_map(|l| {
                    let k = (l[1] % n as u64) as u32;
                    let v = m(l[0]);
                    let r = if k == 0 { v } else { m((v << k) | (v >> (n - k))) };
                    to_words(r, n)
                })
                .collect()
        });
        // extract_bits and insert_bits where MSL defines them (offset + bits within the width).
        let i32t = Ty::scalar(Scalar::I32);
        let mut ex = Vec::new();
        let mut ins = Vec::new();
        for _ in 0..1 << 14 {
            let off = rng.below(n + 1);
            let bits = rng.below(n - off + 1);
            let x: Vec<u32> = items_for(n, 1, 2, &mut rng, 1).remove(0);
            let y: Vec<u32> = items_for(n, 1, 2, &mut rng, 1).remove(0);
            ex.push([x.clone(), vec![off, bits]].concat());
            ins.push([x, y, vec![off, bits]].concat());
        }
        for signed in [true, false] {
            let su = if signed { "s" } else { "u" };
            run_int(&format!("air.extract_bits.{}.{}", su, ty.suffix()), &[ty, i32t, i32t], ty, ex.clone(), |w| {
                let wl = if n == 64 { 4 } else { 2 };
                let (off, bits) = (w[wl], w[wl + 1]);
                decode(&w[..wl], 1, 2, n)
                    .iter()
                    .flat_map(|l| {
                        let v = m(l[0]);
                        let r = if bits == 0 {
                            0
                        } else {
                            let f = (v >> off) & if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
                            if signed && f >> (bits - 1) & 1 == 1 {
                                m(f | !(if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 }))
                            } else {
                                f
                            }
                        };
                        to_words(r, n)
                    })
                    .collect()
            });
            run_int(&format!("air.insert_bits.{}.{}", su, ty.suffix()), &[ty, ty, i32t, i32t], ty, ins.clone(), |w| {
                let wl = if n == 64 { 4 } else { 2 };
                let (off, bits) = (w[2 * wl], w[2 * wl + 1]);
                decode(&w[..2 * wl], 2, 2, n)
                    .iter()
                    .flat_map(|l| {
                        // An offset of the full width (with 0 bits) inserts nothing: MSL leaves only offset + bits
                        // above the width undefined, so the shifts must not overflow there.
                        let mask =
                            if bits == 64 { u64::MAX } else { ((1u64 << bits) - 1).checked_shl(off).unwrap_or(0) };
                        to_words(m((l[0] & !mask) | (l[1].checked_shl(off).unwrap_or(0) & mask)), n)
                    })
                    .collect()
            });
        }
    }
}

// Conversions and comparisons ----------------------------------------------------------------------------------------

/// A random value of a scalar type as carrier words.
fn value_of(s: Scalar, rng: &mut Rng) -> Vec<u32> {
    match s {
        Scalar::F32 => vec![f32_any(rng)],
        Scalar::F16 => vec![if rng.below(4) == 0 { r::f32_to_f16(f32_any(rng)) as u32 } else { rng.below(1 << 16) }],
        Scalar::I1 => vec![rng.below(2)],
        Scalar::I64 => vec![rng.u32(), if rng.below(2) == 0 { rng.u32() } else { (rng.below(3) as i32 - 1) as u32 }],
        _ => vec![rng.u32() & ((1u64 << s.bits()) - 1) as u32],
    }
}

/// MSL static_cast (2.23, 8.6) on one value; out-of-range float to integer saturates (Rust's `as` does exactly that,
/// and converts NaN to 0, as MSL 8.6 requires).
fn convert_ref(dk: char, d: Scalar, sk: char, s: Scalar, w: &[u32]) -> Vec<u32> {
    // The source value as an exact real: f64 holds every binary32, binary16 and 32-bit integer; i64 uses i128.
    let as_int = |w: &[u32]| -> i128 {
        match (s, sk) {
            (Scalar::I64, 's') => (w[0] as u64 | (w[1] as u64) << 32) as i64 as i128,
            (Scalar::I64, _) => (w[0] as u64 | (w[1] as u64) << 32) as i128,
            (_, 's') => val(w[0] as u64, s.bits(), true),
            _ => w[0] as i128 & ((1i128 << s.bits()) - 1),
        }
    };
    let float = |w: &[u32]| -> f32 { if s == Scalar::F16 { r::f(r::f16_to_f32(w[0] as u16)) } else { r::f(w[0]) } };
    let out_int = |v: i128| -> Vec<u32> { to_words(wrap(v, d.bits()), d.bits()) };
    if dk == sk && d == s {
        return w.to_vec();
    }
    match (dk, d) {
        (_, Scalar::I1) => {
            vec![if sk == 'f' { (float(w) != 0.0 || float(w).is_nan()) as u32 } else { (as_int(w) != 0) as u32 }]
        }
        ('f', Scalar::F32) => {
            if sk == 'f' {
                vec![float(w).to_bits()]
            } else if s == Scalar::I64 {
                vec![if sk == 's' { (as_int(w) as i64 as f32).to_bits() } else { (as_int(w) as u64 as f32).to_bits() }]
            } else {
                vec![(as_int(w) as f64 as f32).to_bits()]
            }
        }
        ('f', _) => {
            if sk == 'f' {
                vec![r::f32_to_f16(float(w).to_bits()) as u32]
            } else if s == Scalar::I64 {
                vec![if sk == 's' { r::i64_to_f16(as_int(w) as i64 as u64) } else { r::u64_to_f16(as_int(w) as u64) }
                    as u32]
            } else if sk == 's' {
                vec![r::i32_to_f16(as_int(w) as i32 as u32) as u32]
            } else {
                vec![r::u32_to_f16(as_int(w) as u32) as u32]
            }
        }
        (_, _) if sk == 'f' => {
            let x = float(w);
            let v: i128 = match (dk, d) {
                ('s', Scalar::I8) => x as i8 as i128,
                ('s', Scalar::I16) => x as i16 as i128,
                ('s', Scalar::I32) => x as i32 as i128,
                ('s', _) => x as i64 as i128,
                (_, Scalar::I8) => x as u8 as i128,
                (_, Scalar::I16) => x as u16 as i128,
                (_, Scalar::I32) => x as u32 as i128,
                _ => x as u64 as i128,
            };
            out_int(v)
        }
        _ => out_int(as_int(w)),
    }
}

#[test]
fn conversions_follow_msl_static_cast() {
    let mut rng = Rng::new(33);
    let types = [
        (Scalar::F32, 'f'),
        (Scalar::F16, 'f'),
        (Scalar::I1, 'u'),
        (Scalar::I8, 's'),
        (Scalar::I8, 'u'),
        (Scalar::I16, 's'),
        (Scalar::I16, 'u'),
        (Scalar::I32, 's'),
        (Scalar::I32, 'u'),
        (Scalar::I64, 's'),
        (Scalar::I64, 'u'),
    ];
    for &(d, dk) in &types {
        for &(s, sk) in &types {
            let lanes = 2;
            let (dt, st) = (Ty::vector(d, lanes), Ty::vector(s, lanes));
            let name = format!("air.convert.{}.{}.{}.{}", dk, dt.suffix(), sk, st.suffix());
            let items: Vec<Vec<u32>> = (0..1 << 13)
                .map(|k| {
                    (0..lanes)
                        .flat_map(|_| {
                            if s == Scalar::F32 && k % 4 == 0 {
                                vec![SPECIAL_F32[rng.below(SPECIAL_F32.len() as u32) as usize]]
                            } else {
                                value_of(s, &mut rng)
                            }
                        })
                        .collect()
                })
                .collect();
            let sw = if s == Scalar::I64 { 2 } else { 1 };
            run_int(&name, &[st], dt, items, |w| {
                (0..lanes as usize).flat_map(|l| convert_ref(dk, d, sk, s, &w[l * sw..(l + 1) * sw])).collect()
            });
        }
    }
}

#[test]
fn comparisons_are_ieee() {
    let mut rng = Rng::new(34);
    let preds = [
        (FPred::Oeq, (|a: f32, b: f32| a == b) as fn(f32, f32) -> bool),
        (FPred::Ogt, |a, b| a > b),
        (FPred::Oge, |a, b| a >= b),
        (FPred::Olt, |a, b| a < b),
        (FPred::Ole, |a, b| a <= b),
        (FPred::One, |a, b| matches!(a.partial_cmp(&b), Some(Ordering::Less | Ordering::Greater))),
        (FPred::Ord, |a, b| !a.is_nan() && !b.is_nan()),
        (FPred::Ueq, |a, b| a.is_nan() || b.is_nan() || a == b),
        (FPred::Ugt, |a, b| !matches!(a.partial_cmp(&b), Some(Ordering::Less | Ordering::Equal))),
        (FPred::Uge, |a, b| !matches!(a.partial_cmp(&b), Some(Ordering::Less))),
        (FPred::Ult, |a, b| !matches!(a.partial_cmp(&b), Some(Ordering::Greater | Ordering::Equal))),
        (FPred::Ule, |a, b| !matches!(a.partial_cmp(&b), Some(Ordering::Greater))),
        (FPred::Une, |a, b| a != b),
        (FPred::Uno, |a, b| a.is_nan() || b.is_nan()),
    ];
    for s in [Scalar::F32, Scalar::F16] {
        let ty = Ty::vector(s, 2);
        let items: Vec<Vec<u32>> = (0..1 << 15)
            .map(|_| {
                let a = value_of(s, &mut rng);
                let b = if rng.below(4) == 0 { a.clone() } else { value_of(s, &mut rng) };
                [a, b, value_of(s, &mut rng), value_of(s, &mut rng)].concat()
            })
            .map(|v| vec![v[0], v[2], v[1], v[3]])
            .collect();
        for (p, f) in preds {
            let soft = Rounding::SoftFloat;
            let (_, got) = run_lowered(
                Op::Inst(Inst::FCmp(p)),
                &[ty, ty],
                Ty::vector(Scalar::I1, 2),
                FastMath::NONE,
                &soft,
                &items,
            );
            let val = |w: u32| if s == Scalar::F16 { r::f(r::f16_to_f32(w as u16)) } else { r::f(w) };
            check(&format!("fcmp {:?} {:?}", p, s), &items, &got, |w| {
                vec![f(val(w[0]), val(w[2])) as u32, f(val(w[1]), val(w[3])) as u32]
            });
        }
    }
}

// Policies on this device ------------------------------------------------------------------------------------------

fn subnormal32(x: u32) -> bool {
    x & 0x7f80_0000 == 0 && x & 0x007f_ffff != 0
}

/// Fast-math +, *, fma and a polyfilled function, lowered under the default policy (bit-exact everywhere) and under
/// the native policy this device's self-test supports (round to nearest even; subnormal results may be flushed, as
/// MSL 8.1 allows).
#[test]
fn policies_compute_correctly_on_this_device() {
    let mut rng = Rng::new(35);
    let ty = Ty::vector(Scalar::F32, 4);
    let items: Vec<Vec<u32>> = (0..1 << 16)
        .map(|_| {
            (0..12)
                .map(|_| {
                    loop {
                        let v = f32_any(&mut rng);
                        // Fast math: NaN and infinity operands are undefined (MSL 8.1).
                        if r::f(v).is_finite() {
                            break v;
                        }
                    }
                })
                .collect()
        })
        .collect();
    let cases = selftest::cases();
    let want: Vec<u32> = cases.iter().map(|c| c.want).collect();
    let native = Rounding::Native(selftest::evaluate(&cases, &want).native_rounding().unwrap());
    for (label, rounding) in [("soft-float", Rounding::SoftFloat), ("native", native)] {
        for (op, arity, f) in [
            (Op::Inst(Inst::FAdd), 2, (|a: &[u32]| r::add(a[0], a[1])) as fn(&[u32]) -> u32),
            (Op::Inst(Inst::FMul), 2, |a| r::mul(a[0], a[1])),
            (Op::Call("air.fma.v4f32"), 3, |a| r::fma(a[0], a[1], a[2])),
        ] {
            let tys = vec![ty; arity];
            let (l, got) = run_lowered(op, &tys, ty, FastMath::FAST, &rounding, &items);
            let mut flushed = 0;
            for (inp, g) in items.iter().zip(&got) {
                for lane in 0..4 {
                    let a: Vec<u32> = (0..arity).map(|k| inp[k * 4 + lane]).collect();
                    let w = f(&a);
                    if r::f(w).is_nan() || !r::f(w).is_finite() {
                        continue;
                    }
                    if g[lane] != w {
                        // Only the native policy may flush (MSL 8.5): a subnormal result to a zero of either sign, or
                        // subnormal operands to zeros of either sign before an otherwise conforming operation.
                        // "Subnormal before rounding": whether the exact value is below the smallest normal. binary64
                        // holds binary32 products exactly; a sum is the exact pair (s, e) of TwoSum.
                        let m = 2f64.powi(-126);
                        let tiny = |b: &[u32]| -> bool {
                            let x: Vec<f64> = b.iter().map(|&v| r::f(v) as f64).collect();
                            let (p, c) = match arity {
                                3 => (x[0] * x[1], x[2]),
                                _ if matches!(op, Op::Inst(Inst::FMul)) => (x[0] * x[1], 0.0),
                                _ => (x[0], x[1]),
                            };
                            let s = p + c;
                            let bb = s - p;
                            let e = (p - (s - bb)) + (c - bb);
                            if s == 0.0 && e == 0.0 {
                                return false;
                            }
                            s.abs() < m || (s.abs() == m && e != 0.0 && (e < 0.0) != (s < 0.0))
                        };
                        let flush = |b: &[u32], v: u32| if tiny(b) { vec![v, 0, 0x8000_0000] } else { vec![v] };
                        let mut allowed = flush(&a, w);
                        let subs: Vec<usize> = (0..arity).filter(|&k| subnormal32(a[k])).collect();
                        for mask in 0..(1u32 << (2 * subs.len())) {
                            let mut b = a.clone();
                            for (j, &k) in subs.iter().enumerate() {
                                b[k] = (mask >> (2 * j) & 1) << 31;
                            }
                            allowed.extend(flush(&b, f(&b)));
                        }
                        assert!(
                            label == "native" && allowed.contains(&g[lane]),
                            "{} {:?}: {:08x?} got {:08x} want {:08x}",
                            label,
                            op,
                            a,
                            g[lane],
                            w
                        );
                        flushed += 1;
                    }
                }
            }
            eprintln!(
                "{} {:?} ({:?}): {} lanes, {} differ only by flushing subnormals",
                label,
                op,
                l.rule.how,
                items.len() * 4,
                flushed
            );
        }
    }
}

/// binary16 arithmetic lowered with fast flags: native f16 on a shader-f16 device (the half row's decision; any
/// correctly rounded result conforms and this device's f16 is checked to give it), and the exact binary16 library
/// for precise calls.
#[test]
fn half_lowering_on_this_device() {
    assert!(Gpu::get().has_f16, "the device offers shader-f16");
    let mut rng = Rng::new(36);
    let ty = Ty::vector(Scalar::F16, 2);
    let items: Vec<Vec<u32>> = (0..1 << 16).map(|_| (0..4).map(|_| rng.below(1 << 16)).collect()).collect();
    for flags in [FastMath::FAST, FastMath::NONE] {
        for (op, f) in [
            (Inst::FAdd, (|a: u16, b: u16| r::h_add(a, b)) as fn(u16, u16) -> u16),
            (Inst::FSub, |a, b| r::h_sub(a, b)),
            (Inst::FMul, |a, b| r::h_mul(a, b)),
            (Inst::FDiv, |a, b| r::h_div(a, b)),
        ] {
            let soft = Rounding::SoftFloat;
            let (l, got) = run_lowered(Op::Inst(op), &[ty, ty], ty, flags, &soft, &items);
            for (inp, g) in items.iter().zip(&got) {
                for lane in 0..2 {
                    let (a, b) = (inp[lane] as u16, inp[2 + lane] as u16);
                    let w = f(a, b) as u32;
                    let nan = |h: u32| h & 0x7fff > 0x7c00;
                    let special = |h: u16| h & 0x7c00 == 0x7c00;
                    if flags.finite_only()
                        && (special(a) || special(b) || special(w as u16) || (w & 0x7c00 == 0 && w & 0x3ff != 0))
                    {
                        // Fast math: NaN and infinity undefined, subnormals may be flushed.
                        continue;
                    }
                    assert!(
                        g[lane] == w || nan(g[lane]) && nan(w),
                        "{:?} {:?}: {:04x} {:04x} got {:04x} want {:04x}",
                        flags,
                        op,
                        a,
                        b,
                        g[lane],
                        w
                    );
                }
            }
            eprintln!(
                "binary16 {:?} with {}: {:?}, {} lanes",
                op,
                if flags.finite_only() { "fast flags" } else { "no flags" },
                l.rule.how,
                items.len() * 2
            );
        }
    }
}

/// binary16 on a device without shader-f16: fast-math arithmetic is the exact binary16 library under the default
/// policy, and binary32 arithmetic rounded to binary16 under the native policy (exact for +, - and * when binary32
/// rounds to nearest even); transcendental functions evaluate natively in binary32 and round to binary16 (MSL states
/// no half-precision bound for the target), checked here to be within one binary16 ulp of the binary64 value.
#[test]
fn half_fallback_without_shader_f16() {
    let caps = host(false);
    let mut rng = Rng::new(37);
    let ty = Ty::vector(Scalar::F16, 2);
    let finite = |rng: &mut Rng| loop {
        let h = rng.below(1 << 16);
        if h & 0x7c00 != 0x7c00 {
            break h;
        }
    };
    let items: Vec<Vec<u32>> = (0..1 << 16).map(|_| (0..4).map(|_| finite(&mut rng)).collect()).collect();
    let cases = selftest::cases();
    let want: Vec<u32> = cases.iter().map(|c| c.want).collect();
    let native = Rounding::Native(selftest::evaluate(&cases, &want).native_rounding().unwrap());
    for rounding in [Rounding::SoftFloat, native] {
        for (op, f) in [
            (Inst::FAdd, (|a: u16, b: u16| r::h_add(a, b)) as fn(u16, u16) -> u16),
            (Inst::FSub, |a, b| r::h_sub(a, b)),
            (Inst::FMul, |a, b| r::h_mul(a, b)),
            (Inst::FDiv, |a, b| r::h_div(a, b)),
        ] {
            let (l, got) = run_lowered_on(&caps, Op::Inst(op), &[ty, ty], ty, FastMath::FAST, &rounding, &items);
            assert!(l.enables.is_empty() && l.requirements.features.is_empty(), "no f16 needed");
            // Sums, differences and products of binary16 values are never binary32 subnormals (they are multiples
            // of 2^-24, or at least 2^-48), so a host that flushes binary32 subnormals cannot change them.
            let mut mismatches = 0;
            for (inp, g) in items.iter().zip(&got) {
                for lane in 0..2 {
                    let (a, b) = (inp[lane] as u16, inp[2 + lane] as u16);
                    let w = f(a, b) as u32;
                    if w & 0x7c00 == 0x7c00 {
                        continue;
                    }
                    if g[lane] != w {
                        mismatches += 1;
                    }
                }
            }
            assert_eq!(mismatches, 0, "{:?} {:?}", op, l.rule);
            eprintln!("binary16 without shader-f16, {:?} ({:?}): {} lanes exact", op, l.rule.how, items.len() * 2);
        }
    }
    // A transcendental function: binary32 evaluation rounded to binary16.
    let soft = Rounding::SoftFloat;
    let (l, got) = run_lowered_on(&caps, Op::Call("air.sin.v2f16"), &[ty], ty, FastMath::FAST, &soft, &items);
    let mut max = 0f64;
    for (inp, g) in items.iter().zip(&got) {
        for lane in 0..2 {
            let x = r::f(r::f16_to_f32(inp[lane] as u16)) as f64;
            let y = r::f(r::f16_to_f32(g[lane] as u16)) as f64;
            let e = (y - x.sin()).abs() / crate::math::ulp16(x.sin());
            max = max.max(e);
        }
    }
    eprintln!("air.sin.v2f16 without shader-f16 ({:?}): max error {} binary16 ulp", l.rule.how, max);
    assert!(max <= 1.0);
}

/// sincos on vectors (MSL defines it for every float vector type): the sines are the value and the cosines the
/// store, lane by lane equal to the library's sin and cos.
#[test]
fn vector_sincos_returns_sines_and_stores_cosines() {
    let mut rng = Rng::new(38);
    let ty = Ty::vector(Scalar::F32, 3);
    let caps = host(true);
    let soft = Rounding::SoftFloat;
    let ops = [
        Operand {
            expr: "vec3<u32>(nm_t_in0[3u * i], nm_t_in0[3u * i + 1u], nm_t_in0[3u * i + 2u])",
            ty,
            constant: None,
        },
        Operand { expr: "0u", ty, constant: None },
    ];
    let l = lower(
        Op::Call("air.sincos.v3f32"),
        &ops,
        Some(ty),
        FastMath::NONE,
        Target { rounding: &soft, host: &caps },
        "v",
    )
    .unwrap();
    assert_eq!(l.stores.len(), 1);
    let mut p = l.prelude.clone();
    p.require("nm_f32_sin").unwrap();
    p.require("nm_f32_cos").unwrap();
    let body = format!(
        "{}\nlet s = {};\nlet c = {};\nfor (var k = 0u; k < 3u; k++) {{ let x = nm_t_in0[3u * i + k]; nm_t_out[6u * i + k] = select(1u, 0u, s[k] == nm_f32_sin(x)); nm_t_out[6u * i + 3u + k] = select(1u, 0u, c[k] == nm_f32_cos(x)); }}",
        l.stmts.join("\n"),
        l.value.unwrap(),
        l.stores[0].value
    );
    let k = Kernel::new(&crate::gpu::kernel_source(false, &p.emit().unwrap(), "", &body), 6);
    let input: Vec<u32> = (0..3 * 4096).map(|_| f32_any(&mut rng)).collect();
    let out = k.run(&[&input], [0, 0, 0], 4096);
    assert!(out.iter().all(|&w| w == 0), "a lane differs from the library's sin or cos");
}
