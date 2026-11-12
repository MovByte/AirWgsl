//! The lowering rules: `air.*` and `llvm.*` numeric calls and floating-point instructions to WGSL over the carriers of
//! [`super::Ty::carrier`].

// The refusal is the compiler's only failure value (api::compile) and is returned by value throughout.
#![allow(clippy::result_large_err)]

use super::rows::{Decision, ROUNDING_ONLY, row_of};
use super::{Column, FPred, FastMath, How, Inst, Lowered, Op, Operand, Rounding, Rule, Scalar, Store, Target, Ty};
use crate::api::{HostRequirements, Refusal, RefusalKind};
use crate::num::prelude::Prelude;
use std::collections::BTreeSet;

struct Ctx<'a> {
    construct: String,
    target: Target<'a>,
    tmp: &'a str,
    stmts: Vec<String>,
    prelude: Prelude,
    requirements: HostRequirements,
    enables: BTreeSet<String>,
    next: usize,
}

type R<T> = Result<T, Refusal>;

/// What a rule produced besides the shared state: the value, stores and the rule.
struct Out {
    value: Option<String>,
    stores: Vec<Store>,
    rule: Rule,
}

fn out(value: String, rule: Rule) -> Out {
    Out { value: Some(value), stores: Vec::new(), rule }
}

fn exact() -> Rule {
    Rule { row: None, column: Column::Exact, how: How::Exact }
}

impl Ctx<'_> {
    fn refuse(&self, kind: RefusalKind, reason: impl Into<String>, reference: Option<&str>) -> Refusal {
        Refusal {
            kind,
            construct: self.construct.clone(),
            location: None,
            reason: reason.into(),
            reference: reference.map(String::from),
        }
    }

    fn malformed(&self, what: impl Into<String>) -> Refusal {
        self.refuse(RefusalKind::NotImplemented, format!("operands do not match the function: {}", what.into()), None)
    }

    /// Bind an expression to a fresh `let` and return its name.
    fn bind(&mut self, expr: &str) -> String {
        let name = format!("{}_{}", self.tmp, self.next);
        self.next += 1;
        self.stmts.push(format!("let {} = {};", name, expr));
        name
    }

    fn need(&mut self, item: &str) -> R<()> {
        self.prelude
            .require(item)
            .map_err(|e| self.refuse(RefusalKind::NotImplemented, format!("library defect: {}", e.0), None))
    }

    fn uses_f16(&mut self) {
        self.requirements.features.insert("shader-f16".into());
        self.enables.insert("f16".into());
    }
}

/// Lower one operation. Operand expressions may be any WGSL expressions in their carriers; each is evaluated once,
/// bound to a `let` named `{tmp}_{n}`. `ret` is the result type (`None` for a call without one).
pub fn lower(
    op: Op,
    args: &[Operand],
    ret: Option<Ty>,
    flags: FastMath,
    target: Target,
    tmp: &str,
) -> Result<Lowered, Refusal> {
    let construct = match op {
        Op::Call(name) => name.to_string(),
        Op::Inst(i) => format!("{:?}", i).to_lowercase(),
    };
    let mut c = Ctx {
        construct,
        target,
        tmp,
        stmts: Vec::new(),
        prelude: Prelude::new(),
        requirements: HostRequirements::default(),
        enables: BTreeSet::new(),
        next: 0,
    };
    let bound: Vec<String> = args.iter().map(|a| c.bind(a.expr)).collect();
    let tys: Vec<Ty> = args.iter().map(|a| a.ty).collect();
    let o = match op {
        Op::Inst(i) => lower_inst(&mut c, i, &bound, &tys, args, ret, flags)?,
        Op::Call(name) => lower_call(&mut c, name, &bound, &tys, ret, flags)?,
    };
    Ok(Lowered {
        stmts: c.stmts,
        value: o.value,
        stores: o.stores,
        prelude: c.prelude,
        requirements: c.requirements,
        enables: c.enables,
        rule: o.rule,
    })
}

// Lanes ------------------------------------------------------------------------------------------------------------

fn lane(v: &str, ty: Ty, i: u32) -> String {
    if ty.lanes == 1 { v.to_string() } else { format!("{}[{}]", v, i) }
}

/// Build a value of `ty` from one expression per lane.
fn build(ty: Ty, lanes: Vec<String>) -> String {
    if ty.lanes == 1 {
        lanes.into_iter().next().unwrap_or_default()
    } else {
        format!("{}({})", ty.carrier(), lanes.join(", "))
    }
}

/// Apply `f` lane by lane; scalar operands are broadcast to every lane.
fn per_lane(ty: Ty, ops: &[(&str, Ty)], f: impl Fn(&[String]) -> String) -> String {
    let lanes = (0..ty.lanes)
        .map(|i| {
            let a: Vec<String> = ops.iter().map(|(v, t)| lane(v, *t, if t.lanes == 1 { 0 } else { i })).collect();
            f(&a)
        })
        .collect();
    build(ty, lanes)
}

fn check_lanes(c: &Ctx, ty: Ty, tys: &[Ty]) -> R<()> {
    for t in tys {
        if t.lanes != 1 && t.lanes != ty.lanes {
            return Err(c.malformed(format!("{} lanes against {}", t.lanes, ty.lanes)));
        }
    }
    Ok(())
}

// Floating point ---------------------------------------------------------------------------------------------------

/// How one floating-point lane is computed.
enum FloatImpl {
    /// A library function over the carriers.
    Lib(String),
    /// A WGSL expression over native values: `{0}`, `{1}`, `{2}` are the operands.
    Native(&'static str),
}

/// The numerics.toml decision for a floating-point operation and how it is carried out.
fn decide(c: &Ctx, row: &'static str, scalar: Scalar, fast: bool) -> R<(Rule, bool)> {
    let r = row_of(row)
        .ok_or_else(|| c.refuse(RefusalKind::NotImplemented, format!("no numerics.toml row {}", row), None))?;
    let (column, d) = match (scalar, fast) {
        (Scalar::F32, true) => (Column::FastF32, r.fast_f32),
        (Scalar::F32, false) => (Column::PreciseF32, r.precise_f32),
        (Scalar::F16, true) => (Column::Half, r.half),
        // MSL 8.1 asks NaN and infinity support with fast math disabled for half as for single precision, which
        // native WGSL f16 does not give: the exact binary16 library, as the generic precise decision does for f32.
        (Scalar::F16, false) => (Column::Half, Decision::SoftFloat),
        _ => return Err(c.malformed("a floating-point operation on an integer type")),
    };
    match d {
        Decision::Native => Ok((Rule { row: Some(r.id), column, how: How::Native }, true)),
        Decision::SoftFloat => Ok((Rule { row: Some(r.id), column, how: How::SoftFloat }, false)),
        Decision::Polyfill => {
            if let Rounding::Native(n) = c.target.rounding
                && ROUNDING_ONLY.contains(&r.id)
                && (r.id != "fma" || n.fused_fma())
            {
                return Ok((Rule { row: Some(r.id), column, how: How::NativeRne }, true));
            }
            Ok((Rule { row: Some(r.id), column, how: How::SoftFloat }, false))
        }
        Decision::Refusal => {
            Err(c.refuse(RefusalKind::NoExactLowering, format!("numerics.toml row {} refuses it", r.id), None))
        }
    }
}

fn float_scalar(t: Scalar) -> &'static str {
    if t == Scalar::F16 { "f16" } else { "f32" }
}

/// One lane of a native operation: operands converted from carriers to WGSL floats, the expression, and back.
fn native_lane(c: &mut Ctx, scalar: Scalar, pattern: &str, ops: &[String], int_ops: &[usize]) -> R<String> {
    let f16_native = scalar == Scalar::F16 && c.target.has_f16();
    let conv: Vec<String> = ops
        .iter()
        .enumerate()
        .map(|(k, v)| {
            if int_ops.contains(&k) {
                format!("bitcast<i32>({})", v)
            } else if scalar == Scalar::F32 {
                format!("bitcast<f32>({})", v)
            } else if f16_native {
                format!("nm_h({})", v)
            } else {
                format!("nm_hw({})", v)
            }
        })
        .collect();
    let mut e = pattern.to_string();
    // Highest index first, so that {1} does not match inside {10}.
    for (k, v) in conv.iter().enumerate().rev() {
        e = e.replace(&format!("{{{}}}", k), v);
    }
    // Literals in patterns are written as f32; in f16 code they become f16.
    e = e.replace("{T}", if f16_native { "f16" } else { "f32" });
    Ok(if scalar == Scalar::F32 {
        format!("bitcast<u32>({})", e)
    } else if f16_native {
        c.uses_f16();
        c.need("nm_h")?;
        c.need("nm_hb")?;
        format!("nm_hb({})", e)
    } else {
        c.need("nm_hw")?;
        c.need("nm_hn")?;
        format!("nm_hn({})", e)
    })
}

/// Lower a floating-point operation lane by lane with either implementation.
fn float_op(c: &mut Ctx, ty: Ty, ops: &[(&str, Ty)], imp: FloatImpl, int_ops: &[usize]) -> R<String> {
    let lanes: Vec<Vec<String>> = (0..ty.lanes)
        .map(|i| ops.iter().map(|(v, t)| lane(v, *t, if t.lanes == 1 { 0 } else { i })).collect::<Vec<_>>())
        .collect();
    let mut exprs = Vec::new();
    for l in lanes {
        exprs.push(match &imp {
            FloatImpl::Lib(f) => {
                c.need(f)?;
                let args: Vec<String> = l
                    .iter()
                    .enumerate()
                    .map(|(k, v)| if int_ops.contains(&k) { format!("bitcast<i32>({})", v) } else { v.clone() })
                    .collect();
                format!("{}({})", f, args.join(", "))
            }
            FloatImpl::Native(p) => native_lane(c, ty.scalar, p, &l, int_ops)?,
        });
    }
    Ok(build(ty, exprs))
}

/// A math function stem: its numerics.toml row, operand count, the operand that is an integer (ldexp's exponent),
/// its native WGSL form, and whether AIR has a `fast_` variant of it.
struct MathFn {
    stem: &'static str,
    row: &'static str,
    arity: usize,
    int_op: Option<usize>,
    native: &'static str,
    has_fast: bool,
}

const fn m(stem: &'static str, arity: usize, native: &'static str, has_fast: bool) -> MathFn {
    MathFn { stem, row: stem, arity, int_op: None, native, has_fast }
}

/// The `air.*` math functions (intrinsics.toml family math-float). `native` is used where the row decides native;
/// "" marks a stem whose native form is the exact library function (a WGSL built-in would change the definition:
/// round's halfway rule, fract's open interval, nextafter).
const MATH: &[MathFn] = &[
    m("acos", 1, "acos({0})", true),
    m("acosh", 1, "acosh({0})", true),
    m("asin", 1, "asin({0})", true),
    m("asinh", 1, "asinh({0})", true),
    m("atan", 1, "atan({0})", true),
    m("atan2", 2, "atan2({0}, {1})", true),
    m("atanh", 1, "atanh({0})", true),
    m("ceil", 1, "ceil({0})", true),
    m("clamp", 3, "clamp({0}, {1}, {2})", true),
    m("cos", 1, "cos({0})", true),
    m("cosh", 1, "cosh({0})", true),
    m("cospi", 1, "cos({0} * {T}(3.14159265358979323846))", true),
    m("exp", 1, "exp({0})", true),
    m("exp10", 1, "exp2({0} * {T}(3.32192809488736234787))", true),
    m("exp2", 1, "exp2({0})", true),
    m("fabs", 1, "abs({0})", true),
    m("floor", 1, "floor({0})", true),
    m("fma", 3, "fma({0}, {1}, {2})", false),
    m("fmax", 2, "max({0}, {1})", true),
    m("fmax3", 3, "max({0}, max({1}, {2}))", true),
    m("fmedian3", 3, "max(min({0}, {1}), min(max({0}, {1}), {2}))", true),
    m("fmin", 2, "min({0}, {1})", true),
    m("fmin3", 3, "min({0}, min({1}, {2}))", true),
    m("fmod", 2, "({0} % {1})", true),
    m("fract", 1, "", true),
    MathFn { stem: "ldexp", row: "ldexp", arity: 2, int_op: Some(1), native: "ldexp({0}, {1})", has_fast: true },
    m("log", 1, "log({0})", true),
    m("log10", 1, "(log2({0}) * {T}(0.301029995663981195214))", true),
    m("log2", 1, "log2({0})", true),
    m("mix", 3, "mix({0}, {1}, {2})", false),
    m("nextafter", 2, "", false),
    m("pow", 2, "pow({0}, {1})", true),
    m("powr", 2, "pow({0}, {1})", true),
    m("rint", 1, "round({0})", true),
    m("round", 1, "", true),
    m("rsqrt", 1, "inverseSqrt({0})", true),
    m("saturate", 1, "saturate({0})", true),
    m("sign", 1, "sign({0})", false),
    m("sin", 1, "sin({0})", true),
    m("sinh", 1, "sinh({0})", true),
    m("sinpi", 1, "sin({0} * {T}(3.14159265358979323846))", true),
    m("sqrt", 1, "sqrt({0})", true),
    m("tan", 1, "tan({0})", true),
    m("tanh", 1, "tanh({0})", true),
    m("tanpi", 1, "tan({0} * {T}(3.14159265358979323846))", true),
    m("trunc", 1, "trunc({0})", true),
];

/// The library function of a stem for a float type.
fn lib_name(stem: &str, scalar: Scalar) -> String {
    let stem = if stem == "fabs" && scalar == Scalar::F32 { "abs" } else { stem };
    format!("nm_{}_{}", float_scalar(scalar), stem)
}

/// Native forms valid only on a host proven to round to nearest even (rows in ROUNDING_ONLY).
fn rne_native(stem: &str) -> Option<&'static str> {
    match stem {
        "fma" => Some("fma({0}, {1}, {2})"),
        "fract" => Some("min({0} - floor({0}), {T}(0.99999994039535522461))"),
        _ => None,
    }
}

fn lower_math(c: &mut Ctx, f: &MathFn, fast_name: bool, b: &[String], tys: &[Ty], flags: FastMath) -> R<Out> {
    if b.len() != f.arity + usize::from(f.stem == "sincos") {
        return Err(c.malformed(format!("{} operands", b.len())));
    }
    let ty = tys[0];
    check_lanes(c, ty, tys)?;
    if !ty.scalar.is_float() {
        return Err(c.malformed("an integer type"));
    }
    if fast_name && !f.has_fast {
        return Err(c.malformed("no fast_ variant exists"));
    }
    // The fast or precise variant: the name says when AIR has both, the call's flags otherwise.
    let fast = if f.has_fast && ty.scalar == Scalar::F32 { fast_name } else { fast_name || flags.finite_only() };
    let (rule, native) = decide(c, f.row, ty.scalar, fast)?;
    let ops: Vec<(&str, Ty)> = b.iter().map(String::as_str).zip(tys.iter().copied()).collect();
    let int_ops: Vec<usize> = f.int_op.into_iter().collect();
    let pattern = if rule.how == How::NativeRne { rne_native(f.stem).unwrap_or(f.native) } else { f.native };
    let imp = if native && !pattern.is_empty() {
        FloatImpl::Native(pattern)
    } else {
        FloatImpl::Lib(lib_name(f.stem, ty.scalar))
    };
    let rule = if native && pattern.is_empty() { Rule { how: How::Exact, ..rule } } else { rule };
    Ok(out(float_op(c, ty, &ops, imp, &int_ops)?, rule))
}

fn lower_sincos(c: &mut Ctx, fast_name: bool, b: &[String], tys: &[Ty], flags: FastMath) -> R<Out> {
    let ty = tys[0];
    if b.len() != 2 || !ty.scalar.is_float() {
        return Err(c.malformed("sincos takes a float scalar or vector and a pointer"));
    }
    let s = ty.scalar;
    let fast = if s == Scalar::F32 { fast_name } else { fast_name || flags.finite_only() };
    let (rule, native) = decide(c, "sincos", s, fast)?;
    // One (sin, cos) pair per lane; the sines are the value, the cosines are stored through the pointer operand.
    let mut pairs = Vec::new();
    for i in 0..ty.lanes {
        let x = lane(&b[0], ty, i);
        pairs.push(if native {
            let sin = native_lane(c, s, "sin({0})", std::slice::from_ref(&x), &[])?;
            let cos = native_lane(c, s, "cos({0})", &[x], &[])?;
            c.bind(&format!("vec2<u32>({}, {})", sin, cos))
        } else {
            let f = lib_name("sincos", s);
            c.need(&f)?;
            c.bind(&format!("{}({})", f, x))
        });
    }
    let sines = build(ty, pairs.iter().map(|p| format!("{}.x", p)).collect());
    let cosines = build(ty, pairs.iter().map(|p| format!("{}.y", p)).collect());
    Ok(Out { value: Some(sines), stores: vec![Store { operand: 1, value: cosines, ty }], rule })
}

/// dot: the sum of the products, left to right (MSL Table 6.11).
fn lower_dot(c: &mut Ctx, b: &[String], tys: &[Ty], flags: FastMath) -> R<Out> {
    if b.len() != 2 || tys[0] != tys[1] || !tys[0].scalar.is_float() {
        return Err(c.malformed("dot takes two vectors of the same float type"));
    }
    let ty = tys[0];
    let s = ty.scalar;
    let (rule, native) = decide(c, "dot", s, flags.finite_only())?;
    let n = ty.lanes;
    let expr = if native {
        // MSL states no bound for dot and WGSL's is the sum of products (numerics.toml row dot).
        let terms: Vec<String> = (0..n).map(|i| format!("({{{}}} * {{{}}})", 2 * i, 2 * i + 1)).collect();
        let pattern = terms.join(" + ");
        let ops: Vec<String> = (0..n).flat_map(|i| [lane(&b[0], ty, i), lane(&b[1], ty, i)]).collect();
        native_lane(c, s, &pattern, &ops, &[])?
    } else {
        let mul = format!("nm_{}_mul", float_scalar(s));
        let add = format!("nm_{}_add", float_scalar(s));
        c.need(&mul)?;
        c.need(&add)?;
        let mut acc = format!("{}({}, {})", mul, lane(&b[0], ty, 0), lane(&b[1], ty, 0));
        for i in 1..n {
            acc = format!("{}({}, {}({}, {}))", add, acc, mul, lane(&b[0], ty, i), lane(&b[1], ty, i));
        }
        acc
    };
    Ok(out(expr, rule))
}

/// fadd, fsub, fmul, fdiv, frem: rows add, sub, mul, div (recip for a 1.0 numerator), fmod.
fn lower_arith(c: &mut Ctx, i: Inst, b: &[String], tys: &[Ty], args: &[Operand], flags: FastMath) -> R<Out> {
    let ty = tys[0];
    if b.len() != 2 || tys[1] != ty || !ty.scalar.is_float() {
        return Err(c.malformed("two operands of one float type"));
    }
    let s = ty.scalar;
    let one = if s == Scalar::F32 { 0x3f80_0000 } else { 0x3c00 };
    let (row, native, stem) = match i {
        Inst::FAdd => ("add", "({0} + {1})", "add"),
        Inst::FSub => ("sub", "({0} - {1})", "sub"),
        Inst::FMul => ("mul", "({0} * {1})", "mul"),
        Inst::FDiv if args[0].constant == Some(one) && ty.lanes == 1 => ("recip", "({0} / {1})", "div"),
        Inst::FDiv => ("div", "({0} / {1})", "div"),
        _ => ("fmod", "({0} % {1})", "fmod"),
    };
    let fast = flags.finite_only();
    let (rule, use_native) = decide(c, row, s, fast)?;
    let ops: Vec<(&str, Ty)> = b.iter().map(String::as_str).zip(tys.iter().copied()).collect();
    // binary16 without shader-f16 computes natively in binary32 and rounds to binary16. For +, - and * that is the
    // correctly rounded binary16 result when binary32 rounds to nearest even (binary32 has at least 2 * 11 + 2
    // significand bits); otherwise, and for / and fmod, the exact binary16 library gives the correctly rounded
    // result, which the half row's native decision also admits.
    let f16_fallback = s == Scalar::F16 && !c.target.has_f16();
    let fallback_native =
        matches!(c.target.rounding, Rounding::Native(_)) && matches!(i, Inst::FAdd | Inst::FSub | Inst::FMul);
    let imp = if use_native && !(f16_fallback && !fallback_native) {
        FloatImpl::Native(native)
    } else {
        FloatImpl::Lib(format!("nm_{}_{}", float_scalar(s), stem))
    };
    let rule = match (&imp, rule.how) {
        (FloatImpl::Lib(_), How::Native) => Rule { how: How::SoftFloat, ..rule },
        (FloatImpl::Native(_), How::Native) if f16_fallback => Rule { how: How::NativeRne, ..rule },
        _ => rule,
    };
    let expr = if let (FloatImpl::Lib(f), "recip") = (&imp, row) {
        c.need(f)?;
        format!("{}({}u, {})", f, one, b[1])
    } else {
        float_op(c, ty, &ops, imp, &[])?
    };
    Ok(out(expr, rule))
}

fn fcmp_lane(s: Scalar, p: FPred, a: &str, b: &str) -> String {
    let (a, b) = if s == Scalar::F16 {
        (format!("nm_f16_to_f32({})", a), format!("nm_f16_to_f32({})", b))
    } else {
        (a.into(), b.into())
    };
    let oeq = format!("nm_f32_oeq({}, {})", a, b);
    let olt = format!("nm_f32_olt({}, {})", a, b);
    let ogt = format!("nm_f32_olt({}, {})", b, a);
    let ole = format!("nm_f32_ole({}, {})", a, b);
    let oge = format!("nm_f32_ole({}, {})", b, a);
    let uno = format!("nm_f32_uno({}, {})", a, b);
    match p {
        FPred::False => "false".into(),
        FPred::True => "true".into(),
        FPred::Oeq => oeq,
        FPred::Ogt => ogt,
        FPred::Oge => oge,
        FPred::Olt => olt,
        FPred::Ole => ole,
        FPred::One => format!("(!{} && !{})", uno, oeq),
        FPred::Ord => format!("!{}", uno),
        FPred::Ueq => format!("({} || {})", uno, oeq),
        FPred::Ugt => format!("!{}", ole),
        FPred::Uge => format!("!{}", olt),
        FPred::Ult => format!("!{}", oge),
        FPred::Ule => format!("!{}", ogt),
        FPred::Une => format!("!{}", oeq),
        FPred::Uno => uno,
    }
}

fn lower_inst(
    c: &mut Ctx,
    i: Inst,
    b: &[String],
    tys: &[Ty],
    args: &[Operand],
    ret: Option<Ty>,
    flags: FastMath,
) -> R<Out> {
    match i {
        Inst::FAdd | Inst::FSub | Inst::FMul | Inst::FDiv | Inst::FRem => lower_arith(c, i, b, tys, args, flags),
        Inst::FNeg => {
            let ty = tys[0];
            if !ty.scalar.is_float() {
                return Err(c.malformed("fneg of an integer type"));
            }
            let bit = if ty.scalar == Scalar::F32 { "0x80000000u" } else { "0x8000u" };
            Ok(out(per_lane(ty, &[(&b[0], ty)], |a| format!("({} ^ {})", a[0], bit)), exact()))
        }
        Inst::FCmp(p) => {
            let ty = tys[0];
            if b.len() != 2 || tys[1] != ty || !ty.scalar.is_float() {
                return Err(c.malformed("fcmp of two operands of one float type"));
            }
            for item in ["nm_f32_oeq", "nm_f32_olt", "nm_f32_ole", "nm_f32_uno"] {
                c.need(item)?;
            }
            if ty.scalar == Scalar::F16 {
                c.need("nm_f16_to_f32")?;
            }
            // Comparisons are exact on the bits in every mode: WGSL may flush subnormal operands (15.7.2).
            let rty = Ty::vector(Scalar::I1, ty.lanes);
            Ok(out(per_lane(rty, &[(&b[0], ty), (&b[1], ty)], |a| fcmp_lane(ty.scalar, p, &a[0], &a[1])), exact()))
        }
        Inst::FPToSI | Inst::FPToUI | Inst::SIToFP | Inst::UIToFP | Inst::FPTrunc | Inst::FPExt => {
            let rty = ret.ok_or_else(|| c.malformed("a conversion without a result type"))?;
            let (dk, sk) = match i {
                Inst::FPToSI => ('s', 'f'),
                Inst::FPToUI => ('u', 'f'),
                Inst::SIToFP => ('f', 's'),
                Inst::UIToFP => ('f', 'u'),
                _ => ('f', 'f'),
            };
            lower_convert(c, dk, rty, sk, tys[0], &b[0])
        }
    }
}

// Conversions ------------------------------------------------------------------------------------------------------

/// One lane of a conversion between kinds 'f', 's', 'u' (MSL 2.23 static_cast and 8.6: to floating point rounds to
/// nearest even, to an integer toward zero, NaN to 0; out-of-range values, which MSL leaves undefined, saturate;
/// integer to integer sign- or zero-extends by the source kind and wraps).
fn convert_lane(c: &mut Ctx, dk: char, d: Scalar, sk: char, s: Scalar, v: &str) -> R<String> {
    use Scalar::*;
    let ext32 = |v: &str, s: Scalar, sk: char| -> String {
        match (s, sk) {
            (I1, 's') => format!("select(0u, 0xffffffffu, {})", v),
            (I1, _) => format!("select(0u, 1u, {})", v),
            (I8 | I16, 's') => format!("u32(nm_sext({}, {}u))", v, s.bits()),
            (I64, _) => format!("{}.x", v),
            _ => v.to_string(),
        }
    };
    let ext64 = |v: &str, s: Scalar, sk: char| -> String {
        match (s, sk) {
            (I64, _) => v.to_string(),
            (_, 's') => format!("nm_sext64({})", ext32(v, s, sk)),
            _ => format!("vec2<u32>({}, 0u)", ext32(v, s, sk)),
        }
    };
    if matches!((s, sk), (I8 | I16, 's')) {
        c.need("nm_sext")?;
    }
    Ok(match (dk, d, sk, s) {
        ('f', F32, 'f', F32) | ('f', F16, 'f', F16) => v.to_string(),
        ('f', F16, 'f', F32) => {
            c.need("nm_f32_to_f16")?;
            format!("nm_f32_to_f16({})", v)
        }
        ('f', F32, 'f', F16) => {
            c.need("nm_f16_to_f32")?;
            format!("nm_f16_to_f32({})", v)
        }
        // Float to integer.
        (_, I1, 'f', _) => {
            // bool(x) is x != 0: NaN converts to true (measured, tests/num/data int_to_bool and to_bool).
            let w = if s == F16 { format!("nm_f16_to_f32({})", v) } else { v.to_string() };
            if s == F16 {
                c.need("nm_f16_to_f32")?;
            }
            c.need("nm_f32_oeq")?;
            format!("!nm_f32_oeq({}, 0u)", w)
        }
        (_, _, 'f', _) => {
            let w = if s == F16 {
                c.need("nm_f16_to_f32")?;
                format!("nm_f16_to_f32({})", v)
            } else {
                v.to_string()
            };
            let f = format!("nm_f32_to_{}{}", if dk == 's' { "i" } else { "u" }, d.bits());
            c.need(&f)?;
            format!("{}({})", f, w)
        }
        // Integer to float.
        ('f', _, _, _) => {
            let fmt = float_scalar(d);
            let (f, arg) = if s == I64 {
                (format!("nm_{}_from_{}64", fmt, if sk == 's' { "i" } else { "u" }), v.to_string())
            } else {
                (format!("nm_{}_from_{}32", fmt, if sk == 's' { "i" } else { "u" }), ext32(v, s, sk))
            };
            c.need(&f)?;
            format!("{}({})", f, arg)
        }
        // Integer to integer.
        (_, I1, _, _) => {
            if s == I64 {
                format!("any({} != vec2<u32>(0u, 0u))", v)
            } else {
                format!("({} != 0u)", ext32(v, s, sk))
            }
        }
        (_, I64, _, _) => {
            if sk == 's' && s != I64 {
                c.need("nm_sext64")?;
            }
            ext64(v, s, sk)
        }
        (_, d, _, _) => {
            let x = ext32(v, s, sk);
            if d.bits() < 32 { format!("({} & {}u)", x, (1u32 << d.bits()) - 1) } else { x }
        }
    })
}

fn kind_of(c: &Ctx, k: &str, ty: Scalar) -> R<char> {
    let ch = match k {
        "f" => 'f',
        "s" => 's',
        "u" => 'u',
        _ => return Err(c.malformed(format!("conversion kind {}", k))),
    };
    if (ch == 'f') != ty.is_float() {
        return Err(c.malformed(format!("kind {} with type {}", k, ty.suffix())));
    }
    Ok(ch)
}

fn lower_convert(c: &mut Ctx, dk: char, dty: Ty, sk: char, sty: Ty, v: &str) -> R<Out> {
    if dty.lanes != sty.lanes {
        return Err(c.malformed("conversion between different lane counts"));
    }
    if (dk == 'f') != dty.scalar.is_float() || (sk == 'f') != sty.scalar.is_float() {
        return Err(c.malformed("conversion kinds do not match the types"));
    }
    let lanes: Vec<String> = (0..dty.lanes).map(|i| lane(v, sty, i)).collect();
    let mut exprs = Vec::new();
    for l in lanes {
        exprs.push(convert_lane(c, dk, dty.scalar, sk, sty.scalar, &l)?);
    }
    Ok(out(build(dty, exprs), exact()))
}

// Integer functions ------------------------------------------------------------------------------------------------

/// air integer functions (intrinsics.toml family integer, MSL Table 6.2): (name, signed form exists, operand count).
const INT_FNS: &[(&str, usize)] = &[
    ("abs", 1),
    ("abs_diff", 2),
    ("add_sat", 2),
    ("clamp", 3),
    ("hadd", 2),
    ("mad_hi", 3),
    ("mad_sat", 3),
    ("max", 2),
    ("max3", 3),
    ("median3", 3),
    ("min", 2),
    ("min3", 3),
    ("mul_hi", 2),
    ("rhadd", 2),
    ("sub_sat", 2),
    ("mad24", 3),
    ("mul24", 2),
    ("extract_bits", 3),
    ("insert_bits", 4),
    ("clz", 2),
    ("ctz", 2),
    ("popcount", 1),
    ("reverse_bits", 1),
    ("rotate", 2),
];

/// One lane of an integer function. `sg` is the signedness from the name (`None` for clz, ctz, popcount,
/// reverse_bits, rotate).
fn int_lane(c: &mut Ctx, f: &str, sg: Option<bool>, s: Scalar, a: &[String]) -> R<String> {
    let n = s.bits();
    let su = if sg == Some(true) { "s" } else { "u" };
    let wide = s == Scalar::I64;
    let call = |c: &mut Ctx, name: String, args: &[&str]| -> R<String> {
        c.need(&name)?;
        Ok(if wide {
            format!("{}({})", name, args.join(", "))
        } else {
            format!("{}({}, {}u)", name, args.join(", "), n)
        })
    };
    let pre = if wide { "nm_i64_" } else { "nm_int_" };
    let a: Vec<&str> = a.iter().map(String::as_str).collect();
    Ok(match f {
        "abs" if sg == Some(false) => a[0].to_string(),
        "abs" => call(c, format!("{}abs_s", pre), &a[..1])?,
        "abs_diff" => call(c, format!("{}absdiff_{}", pre, su), &a[..2])?,
        "add_sat" => call(c, format!("{}add_sat_{}", pre, su), &a[..2])?,
        "sub_sat" => call(c, format!("{}sub_sat_{}", pre, su), &a[..2])?,
        "hadd" => call(c, format!("{}hadd_{}", pre, su), &a[..2])?,
        "rhadd" => call(c, format!("{}rhadd_{}", pre, su), &a[..2])?,
        "mul_hi" => call(c, format!("{}mul_hi_{}", pre, su), &a[..2])?,
        "mad_hi" => {
            let hi = call(c, format!("{}mul_hi_{}", pre, su), &a[..2])?;
            if wide {
                c.need("nm_add64")?;
                format!("nm_add64({}, {})", hi, a[2])
            } else {
                format!("(({} + {}) & {}u)", hi, a[2], if n == 32 { u32::MAX } else { (1u32 << n) - 1 })
            }
        }
        "mad_sat" => call(c, format!("{}mad_sat_{}", pre, su), &a[..3])?,
        "max" => call(c, format!("{}max_{}", pre, su), &a[..2])?,
        "min" => call(c, format!("{}min_{}", pre, su), &a[..2])?,
        "max3" => {
            let inner = call(c, format!("{}max_{}", pre, su), &[a[1], a[2]])?;
            call(c, format!("{}max_{}", pre, su), &[a[0], &inner])?
        }
        "min3" => {
            let inner = call(c, format!("{}min_{}", pre, su), &[a[1], a[2]])?;
            call(c, format!("{}min_{}", pre, su), &[a[0], &inner])?
        }
        "median3" => call(c, format!("{}median3_{}", pre, su), &a[..3])?,
        // MSL Table 6.2: clamp returns min(max(x, minval), maxval).
        "clamp" => {
            let mx = call(c, format!("{}max_{}", pre, su), &a[..2])?;
            call(c, format!("{}min_{}", pre, su), &[&mx, a[2]])?
        }
        "clz" => {
            if wide {
                call(c, "nm_i64_clz".into(), &a[..1])?
            } else {
                call(c, "nm_int_clz".into(), &a[..1])?
            }
        }
        "ctz" => call(c, format!("{}ctz", pre), &a[..1])?,
        "popcount" => call(c, format!("{}popcount", pre), &a[..1])?,
        "reverse_bits" => call(c, format!("{}reverse_bits", pre), &a[..1])?,
        "rotate" => call(c, format!("{}rotate", pre), &a[..2])?,
        "extract_bits" => {
            if wide {
                c.need("nm_i64_extract_bits")?;
                format!("nm_i64_extract_bits({}, {}, {}, {})", a[0], a[1], a[2], sg == Some(true))
            } else {
                call(c, format!("nm_int_extract_bits_{}", su), &a[..3])?
            }
        }
        "insert_bits" => {
            if wide {
                c.need("nm_i64_insert_bits")?;
                format!("nm_i64_insert_bits({}, {}, {}, {})", a[0], a[1], a[2], a[3])
            } else {
                call(c, "nm_int_insert_bits".into(), &a[..4])?
            }
        }
        "mul24" | "mad24" => {
            if n != 32 {
                return Err(c.malformed("mul24 and mad24 are defined for 32-bit integers (MSL Table 6.2)"));
            }
            // MSL Table 6.2: the product of the low 24 bits; outside the 24-bit range the result is
            // implementation-defined, and Metal multiplies the low 24 bits, sign-extended for the signed form
            // (measured, tests/num/data/metal-pack.txt), which this reproduces.
            let p = if sg == Some(true) {
                c.need("nm_sext")?;
                format!("u32(nm_sext({}, 24u) * nm_sext({}, 24u))", a[0], a[1])
            } else {
                format!("(({} & 0xffffffu) * ({} & 0xffffffu))", a[0], a[1])
            };
            if f == "mad24" { format!("({} + {})", p, a[2]) } else { p }
        }
        _ => return Err(c.refuse(RefusalKind::NotImplemented, format!("integer function {}", f), None)),
    })
}

fn lower_int(c: &mut Ctx, f: &str, sg: Option<bool>, arity: usize, b: &[String], tys: &[Ty]) -> R<Out> {
    if b.len() != arity {
        return Err(c.malformed(format!("{} operands", b.len())));
    }
    let ty = tys[0];
    if ty.scalar.is_float() || ty.scalar == Scalar::I1 {
        return Err(c.malformed("an integer function on a non-integer type"));
    }
    // extract_bits and insert_bits take scalar 32-bit offset and count; clz and ctz an i1 zero-is-poison flag,
    // which only widens what is allowed for zero, where the result is the width either way.
    let (data_ops, ty_ops): (usize, Vec<Ty>) = match f {
        "clz" | "ctz" => (1, tys[..1].to_vec()),
        "extract_bits" => (1, tys[..1].to_vec()),
        "insert_bits" => (2, tys[..2].to_vec()),
        _ => (arity, tys.to_vec()),
    };
    check_lanes(c, ty, &ty_ops)?;
    let lanes = ty.lanes;
    let mut exprs = Vec::new();
    for i in 0..lanes {
        let mut a: Vec<String> =
            (0..data_ops).map(|k| lane(&b[k], tys[k], if tys[k].lanes == 1 { 0 } else { i })).collect();
        if matches!(f, "extract_bits" | "insert_bits") {
            for k in data_ops..arity {
                let t = tys[k];
                if t.scalar == Scalar::I64 {
                    a.push(format!("{}.x", lane(&b[k], t, if t.lanes == 1 { 0 } else { i })));
                } else {
                    a.push(lane(&b[k], t, if t.lanes == 1 { 0 } else { i }));
                }
            }
        }
        exprs.push(int_lane(c, f, sg, ty.scalar, &a)?);
    }
    Ok(out(build(ty, exprs), exact()))
}

// Names --------------------------------------------------------------------------------------------------------------

fn parse_suffix(s: &str) -> Option<Ty> {
    let (lanes, rest) = match s.strip_prefix('v') {
        Some(r) => {
            let end = r.find(|ch: char| !ch.is_ascii_digit())?;
            (r[..end].parse().ok()?, &r[end..])
        }
        None => (1, s),
    };
    let scalar = match rest {
        "i1" => Scalar::I1,
        "i8" => Scalar::I8,
        "i16" => Scalar::I16,
        "i32" => Scalar::I32,
        "i64" => Scalar::I64,
        "f16" => Scalar::F16,
        "f32" => Scalar::F32,
        _ => return None,
    };
    Some(Ty::vector(scalar, lanes))
}

fn expect_suffix(c: &Ctx, suffix: &str, ty: Ty) -> R<()> {
    if parse_suffix(suffix) != Some(ty) {
        return Err(c.malformed(format!("suffix {} with operand type {}", suffix, ty.suffix())));
    }
    Ok(())
}

fn not_numeric(c: &Ctx) -> Refusal {
    c.refuse(RefusalKind::NotImplemented, "not a function of the numerics families", None)
}

fn lower_call(c: &mut Ctx, name: &str, b: &[String], tys: &[Ty], ret: Option<Ty>, flags: FastMath) -> R<Out> {
    if tys.is_empty() {
        return Err(c.malformed("no operands"));
    }
    if let Some(rest) = name.strip_prefix("air.") {
        return lower_air(c, rest, b, tys, ret, flags);
    }
    if let Some(rest) = name.strip_prefix("llvm.") {
        return lower_llvm(c, rest, b, tys, flags);
    }
    Err(not_numeric(c))
}

fn lower_air(c: &mut Ctx, rest: &str, b: &[String], tys: &[Ty], ret: Option<Ty>, flags: FastMath) -> R<Out> {
    let parts: Vec<&str> = rest.split('.').collect();
    let head = parts[0];
    match head {
        "convert" => {
            // air.convert.<dk>.<dty>.<sk>.<sty>; suffixed rounding or saturation forms are OpenCL entries MSL never
            // emits (intrinsics.toml family conversion-explicit-rounding).
            if parts.len() != 5 {
                return Err(c.refuse(
                    RefusalKind::NotImplemented,
                    "conversion with an explicit rounding mode or saturation, which MSL does not produce",
                    Some("MSL 2.23"),
                ));
            }
            let dty = ret.ok_or_else(|| c.malformed("a conversion without a result type"))?;
            expect_suffix(c, parts[2], dty)?;
            expect_suffix(c, parts[4], tys[0])?;
            let dk = kind_of(c, parts[1], dty.scalar)?;
            let sk = kind_of(c, parts[3], tys[0].scalar)?;
            lower_convert(c, dk, dty, sk, tys[0], &b[0])
        }
        "dot" => {
            if parts.len() != 2 {
                return Err(c.malformed("dot without a type suffix"));
            }
            expect_suffix(c, parts[1], tys[0])?;
            lower_dot(c, b, tys, flags)
        }
        "all" | "any" => {
            if parts.len() != 2 || tys[0].scalar != Scalar::I1 {
                return Err(c.malformed("all and any take a boolean vector"));
            }
            expect_suffix(c, parts[1], tys[0])?;
            Ok(out(format!("{}({})", head, b[0]), exact()))
        }
        "dfdx" | "dfdy" | "fwidth" => {
            if parts.len() != 2 {
                return Err(c.malformed("derivative without a type suffix"));
            }
            let ty = tys[0];
            expect_suffix(c, parts[1], ty)?;
            // MSL returns a high precision partial derivative, WGSL's fine variant (intrinsics.toml family
            // derivative); WGSL has derivatives of f32 only, so binary16 values widen exactly and round back.
            let w = match head {
                "dfdx" => "dpdxFine",
                "dfdy" => "dpdyFine",
                _ => "fwidthFine",
            };
            if ty.scalar == Scalar::F32 {
                let ft = if ty.lanes == 1 { "f32".to_string() } else { format!("vec{}<f32>", ty.lanes) };
                Ok(out(format!("bitcast<{}>({}(bitcast<{}>({})))", ty.carrier(), w, ft, b[0]), exact_native()))
            } else if ty.scalar == Scalar::F16 {
                c.need("nm_hw")?;
                c.need("nm_hn")?;
                let ops = [(b[0].as_str(), ty)];
                Ok(out(per_lane(ty, &ops, |a| format!("nm_hn({}(nm_hw({})))", w, a[0])), exact_native()))
            } else {
                Err(c.malformed("a derivative of an integer type"))
            }
        }
        "pack" | "unpack" => lower_pack(c, head == "pack", &parts[1..], b, tys, ret),
        _ => {
            // Integer functions: air.<fn>.<s|u>.<type> or air.<fn>.<type>.
            if let Some(&(f, arity)) = INT_FNS.iter().find(|(f, _)| *f == head)
                && !tys[0].scalar.is_float()
            {
                let (sg, suffix) = match parts.as_slice() {
                    [_, "s", t] => (Some(true), *t),
                    [_, "u", t] => (Some(false), *t),
                    [_, t] if matches!(f, "clz" | "ctz" | "popcount" | "reverse_bits" | "rotate") => (None, *t),
                    _ => return Err(c.malformed("integer function name")),
                };
                expect_suffix(c, suffix, tys[0])?;
                return lower_int(c, f, sg, arity, b, tys);
            }
            let (fast, stem) = match head.strip_prefix("fast_") {
                Some(s) => (true, s),
                None => (false, head),
            };
            if parts.len() != 2 {
                return Err(c.malformed("math function without a single type suffix"));
            }
            expect_suffix(c, parts[1], tys[0])?;
            if stem == "sincos" {
                return lower_sincos(c, fast, b, tys, flags);
            }
            match MATH.iter().find(|f| f.stem == stem) {
                Some(f) => lower_math(c, f, fast, b, tys, flags),
                None => Err(not_numeric(c)),
            }
        }
    }
}

fn exact_native() -> Rule {
    Rule { row: None, column: Column::Exact, how: How::Native }
}

/// llvm.* intrinsics with numeric semantics (the corpus's non-air callees).
fn lower_llvm(c: &mut Ctx, rest: &str, b: &[String], tys: &[Ty], flags: FastMath) -> R<Out> {
    let parts: Vec<&str> = rest.split('.').collect();
    let ty = tys[0];
    let suffix = parts.last().copied().unwrap_or("");
    let name = parts[..parts.len() - 1].join(".");
    expect_suffix(c, suffix, ty)?;
    let fast = flags.finite_only();
    let float_math = |c: &mut Ctx, stem: &str| -> R<Out> {
        let f = MATH.iter().find(|f| f.stem == stem).ok_or_else(|| not_numeric(c))?;
        // llvm intrinsics have no fast_ name: the flags decide.
        let pseudo =
            MathFn { stem: f.stem, row: f.row, arity: f.arity, int_op: f.int_op, native: f.native, has_fast: false };
        lower_math(c, &pseudo, false, b, tys, if fast { FastMath::FAST } else { FastMath::NONE })
    };
    if ty.scalar.is_float() {
        return match name.as_str() {
            "fabs" => {
                let m = if ty.scalar == Scalar::F32 { "0x7fffffffu" } else { "0x7fffu" };
                Ok(out(per_lane(ty, &[(&b[0], ty)], |a| format!("({} & {})", a[0], m)), exact()))
            }
            "copysign" => {
                let (m, s) =
                    if ty.scalar == Scalar::F32 { ("0x7fffffffu", "0x80000000u") } else { ("0x7fffu", "0x8000u") };
                Ok(out(
                    per_lane(ty, &[(&b[0], ty), (&b[1], tys[1])], |a| {
                        format!("(({} & {}) | ({} & {}))", a[0], m, a[1], s)
                    }),
                    exact(),
                ))
            }
            "sqrt" | "exp2" | "fma" | "floor" | "ceil" | "trunc" | "rint" | "round" | "sin" | "cos" | "exp" | "log"
            | "log2" | "log10" | "pow" => float_math(c, &name),
            "fmuladd" => float_math(c, "fma"),
            "nearbyint" | "roundeven" => float_math(c, "rint"),
            "maxnum" => float_math(c, "fmax"),
            "minnum" => float_math(c, "fmin"),
            _ => Err(not_numeric(c)),
        };
    }
    let s = ty.scalar;
    let n = s.bits();
    match name.as_str() {
        "abs" => lower_int(c, "abs", Some(true), 1, &b[..1], &tys[..1]),
        "ctlz" => lower_int(c, "clz", None, 2, b, tys),
        "cttz" => lower_int(c, "ctz", None, 2, b, tys),
        "ctpop" => lower_int(c, "popcount", None, 1, b, tys),
        "bitreverse" => lower_int(c, "reverse_bits", None, 1, b, tys),
        "smax" => lower_int(c, "max", Some(true), 2, b, tys),
        "smin" => lower_int(c, "min", Some(true), 2, b, tys),
        "umax" => lower_int(c, "max", Some(false), 2, b, tys),
        "umin" => lower_int(c, "min", Some(false), 2, b, tys),
        "uadd.sat" => lower_int(c, "add_sat", Some(false), 2, b, tys),
        "usub.sat" => lower_int(c, "sub_sat", Some(false), 2, b, tys),
        "sadd.sat" => lower_int(c, "add_sat", Some(true), 2, b, tys),
        "ssub.sat" => lower_int(c, "sub_sat", Some(true), 2, b, tys),
        "bswap" if n == 16 || n == 32 || n == 64 => {
            let f = if n == 64 { "nm_i64_bswap" } else { "nm_int_bswap" };
            c.need(f)?;
            Ok(out(
                per_lane(ty, &[(&b[0], ty)], |a| {
                    if n == 64 { format!("{}({})", f, a[0]) } else { format!("{}({}, {}u)", f, a[0], n) }
                }),
                exact(),
            ))
        }
        "fshl" | "fshr" if b.len() == 3 => {
            let f = format!("nm_{}_{}", if n == 64 { "i64" } else { "int" }, name);
            c.need(&f)?;
            let ops = [(b[0].as_str(), ty), (b[1].as_str(), tys[1]), (b[2].as_str(), tys[2])];
            let e = per_lane(ty, &ops, |a| {
                if n == 64 {
                    format!("{}({}, {}, {})", f, a[0], a[1], a[2])
                } else {
                    format!("{}({}, {}, {}, {}u)", f, a[0], a[1], a[2], n)
                }
            });
            Ok(out(e, exact()))
        }
        _ => Err(not_numeric(c)),
    }
}

// Pack and unpack -----------------------------------------------------------------------------------------------------

/// A packed format of MSL 6.15: its components (bits each, largest code, signed), and whether rgb uses sRGB.
struct PackFmt {
    comps: &'static [(u32, u32)],
    signed: bool,
    srgb: bool,
    /// The packed type: i32, or i16 for 565.
    packed: Scalar,
}

fn pack_format(parts: &[&str]) -> Option<(PackFmt, &'static str)> {
    const C4X8: &[(u32, u32)] = &[(8, 255), (8, 255), (8, 255), (8, 255)];
    const S4X8: &[(u32, u32)] = &[(8, 127), (8, 127), (8, 127), (8, 127)];
    const C2X16: &[(u32, u32)] = &[(16, 65535), (16, 65535)];
    const S2X16: &[(u32, u32)] = &[(16, 32767), (16, 32767)];
    const C1010102: &[(u32, u32)] = &[(10, 1023), (10, 1023), (10, 1023), (2, 3)];
    const C565: &[(u32, u32)] = &[(5, 31), (6, 63), (5, 31)];
    let f = |comps, signed, srgb, packed| PackFmt { comps, signed, srgb, packed };
    Some(match parts {
        ["unorm4x8"] => (f(C4X8, false, false, Scalar::I32), "unorm4x8"),
        ["snorm4x8"] => (f(S4X8, true, false, Scalar::I32), "snorm4x8"),
        ["unorm4x8", "srgb"] => (f(C4X8, false, true, Scalar::I32), "unorm4x8.srgb"),
        ["unorm2x16"] => (f(C2X16, false, false, Scalar::I32), "unorm2x16"),
        ["snorm2x16"] => (f(S2X16, true, false, Scalar::I32), "snorm2x16"),
        ["unorm", "rgb10a2"] => (f(C1010102, false, false, Scalar::I32), "unorm.rgb10a2"),
        ["unorm", "rgb565"] => (f(C565, false, false, Scalar::I16), "unorm.rgb565"),
        _ => return None,
    })
}

fn lower_pack(c: &mut Ctx, pack: bool, parts: &[&str], b: &[String], tys: &[Ty], ret: Option<Ty>) -> R<Out> {
    let (suffix, fmt_parts) = match parts.split_last() {
        Some((s, f)) if parse_suffix(s).is_some() => (*s, f),
        _ => return Err(c.malformed("pack or unpack without a type suffix")),
    };
    let Some((fmt, _)) = pack_format(fmt_parts) else {
        // The other names of the front end's table (1x8, 2x8, 1x16, 4x16, rg11b10f, rgb9e5) have no MSL function
        // at the target's language version that produces them.
        return Err(c.refuse(
            RefusalKind::NotImplemented,
            format!(
                "{} {} has no MSL function at the target's language version",
                if pack { "pack" } else { "unpack" },
                fmt_parts.join(".")
            ),
            Some("MSL 6.15"),
        ));
    };
    let vty = parse_suffix(suffix).unwrap_or(Ty::scalar(Scalar::F32));
    if !vty.scalar.is_float() || vty.lanes as usize != fmt.comps.len() {
        // The front end's grammar admits any float type here; MSL's functions take one vector size per format.
        return Err(c.malformed(format!(
            "the packed format takes {} float components, not {}",
            fmt.comps.len(),
            suffix
        )));
    }
    let half = vty.scalar == Scalar::F16;
    if pack {
        if b.len() != 1 || tys[0] != vty {
            return Err(c.malformed("pack takes one float vector"));
        }
        if ret.is_some_and(|r| r != Ty::scalar(fmt.packed)) {
            return Err(c.malformed("pack result type"));
        }
        let mut shift = 0;
        let mut terms = Vec::new();
        for (i, &(bits, maxv)) in fmt.comps.iter().enumerate() {
            let mut x = lane(&b[0], vty, i as u32);
            if half {
                c.need("nm_f16_to_f32")?;
                x = format!("nm_f16_to_f32({})", x);
            }
            let code = if fmt.srgb && i < 3 {
                c.need("nm_srgb_code")?;
                format!("nm_srgb_code({})", x)
            } else if fmt.signed {
                c.need("nm_snorm_code")?;
                format!("nm_snorm_code({}, {}u, {}u)", x, maxv, bits)
            } else {
                c.need("nm_unorm_code")?;
                format!("nm_unorm_code({}, {}u)", x, maxv)
            };
            terms.push(if shift == 0 { code } else { format!("({} << {}u)", code, shift) });
            shift += bits;
        }
        return Ok(out(format!("({})", terms.join(" | ")), exact()));
    }
    if b.len() != 1 || tys[0] != Ty::scalar(fmt.packed) {
        return Err(c.malformed("unpack takes one packed integer"));
    }
    if ret.is_some_and(|r| r != vty) {
        return Err(c.malformed("unpack result type"));
    }
    let mut shift = 0;
    let mut lanes = Vec::new();
    let fl = if half { "f16" } else { "f32" };
    for (i, &(bits, maxv)) in fmt.comps.iter().enumerate() {
        let code = format!("(({} >> {}u) & {}u)", b[0], shift, (1u32 << bits) - 1);
        lanes.push(if fmt.srgb && i < 3 {
            let f = format!("nm_srgb_{}", fl);
            c.need(&f)?;
            format!("{}({})", f, code)
        } else if fmt.signed {
            let f = format!("nm_snorm_{}", fl);
            c.need(&f)?;
            format!("{}({}, {}u, {}u)", f, code, maxv, bits)
        } else {
            let f = format!("nm_unorm_{}", fl);
            c.need(&f)?;
            format!("{}({}, {}u)", f, code, maxv)
        });
        shift += bits;
    }
    Ok(out(build(vty, lanes), exact()))
}
