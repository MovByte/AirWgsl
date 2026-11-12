//! The decisions of the numerics table (tables/numerics.toml), one row per MSL builtin or operator.
//!
//! The table is the authority; this copy is what the lowering reads at run time, and a unit test checks that it
//! equals the table row for row, so a change to the table that is not carried here fails the build's tests.

/// A decision column of the numerics table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Decision {
    /// WGSL's operation is at least as tight as MSL's bound: emit it.
    Native,
    /// Emit an implementation from correctly rounded operations; on the default policy these are the soft-float
    /// library's.
    Polyfill,
    /// The integer soft-float library (precise functions, which must honor NaN and infinity).
    SoftFloat,
    /// No exact lowering.
    Refusal,
}

/// One row: its id (the MSL function name without `fast_`, or the operator) and its three decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub id: &'static str,
    /// Single precision with fast math (MSL Table 8.2).
    pub fast_f32: Decision,
    /// Single precision with fast math disabled (MSL Table 8.1, NaN and infinity required).
    pub precise_f32: Decision,
    /// Half precision.
    pub half: Decision,
}

use Decision::{Native as N, Polyfill as P, SoftFloat as S};

const fn row(id: &'static str, fast_f32: Decision) -> Row {
    Row { id, fast_f32, precise_f32: S, half: N }
}

/// The rows of numerics.toml, in its order.
pub const ROWS: &[Row] = &[
    row("add", P),
    row("sub", P),
    row("mul", P),
    row("recip", P),
    row("div", N),
    row("acos", P),
    row("acosh", P),
    row("asin", P),
    row("asinh", P),
    row("atan", P),
    row("atan2", P),
    row("atanh", N),
    row("ceil", N),
    row("copysign", P),
    row("cos", P),
    row("cosh", P),
    row("cospi", P),
    row("divide", N),
    row("exp", P),
    row("exp2", P),
    row("exp10", P),
    row("fabs", N),
    row("fdim", P),
    row("floor", N),
    row("fma", P),
    row("fmax", N),
    row("fmin", N),
    row("fmax3", N),
    row("fmin3", N),
    row("fmedian3", P),
    row("fmod", N),
    row("fract", P),
    row("frexp", N),
    row("ilogb", P),
    row("ldexp", N),
    row("log", N),
    row("log2", P),
    row("log10", P),
    row("modf", N),
    row("nextafter", P),
    row("pow", P),
    row("powr", P),
    row("rint", N),
    row("round", P),
    row("rsqrt", N),
    row("sin", P),
    row("sincos", P),
    row("sinh", P),
    row("sinpi", P),
    row("sqrt", P),
    row("tan", P),
    row("tanh", P),
    row("tanpi", P),
    row("trunc", N),
    row("clamp", N),
    row("mix", N),
    row("saturate", N),
    row("sign", N),
    row("smoothstep", N),
    row("step", N),
    row("dot", N),
    row("cross", N),
    row("length", N),
    row("length_squared", N),
    row("distance", N),
    row("distance_squared", N),
    row("normalize", N),
    row("faceforward", N),
    row("reflect", P),
    row("refract", N),
];

pub fn row_of(id: &str) -> Option<&'static Row> {
    ROWS.iter().find(|r| r.id == id)
}

/// The fast-math rows that are polyfill only because WGSL leaves the rounding direction open (numerics.toml rule R1
/// and the row analyses of add, sub, mul, fma, fdim and fract). On a host whose self-test showed round-to-nearest-even
/// for the operations involved, these may use native WGSL arithmetic; every other polyfill row keeps its polyfill.
pub const ROUNDING_ONLY: &[&str] = &["add", "sub", "mul", "fma", "fdim", "fract"];

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Vec<(String, String, String, String)> {
        let mut rows = Vec::new();
        let mut cur: Option<[String; 4]> = None;
        let value = |line: &str, key: &str| -> Option<String> {
            let rest = line.strip_prefix(key)?.trim_start();
            let rest = rest.strip_prefix('=')?.trim();
            Some(rest.trim_matches('"').to_string())
        };
        for line in text.lines() {
            if line.starts_with("[[") {
                if let Some(r) = cur.take() {
                    rows.push((r[0].clone(), r[1].clone(), r[2].clone(), r[3].clone()));
                }
                if line == "[[row]]" {
                    cur = Some(Default::default());
                }
                continue;
            }
            if let Some(r) = cur.as_mut() {
                for (k, key) in ["id", "fast_f32", "precise_f32", "half"].iter().enumerate() {
                    if let Some(v) = value(line, key)
                        && line.starts_with(&format!("{} ", key))
                    {
                        r[k] = v;
                    }
                }
            }
        }
        if let Some(r) = cur.take() {
            rows.push((r[0].clone(), r[1].clone(), r[2].clone(), r[3].clone()));
        }
        rows
    }

    fn name(d: Decision) -> &'static str {
        match d {
            Decision::Native => "native",
            Decision::Polyfill => "polyfill",
            Decision::SoftFloat => "soft-float",
            Decision::Refusal => "refusal",
        }
    }

    #[test]
    fn rows_equal_the_numerics_table() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tables/numerics.toml");
        let text = std::fs::read_to_string(path).unwrap();
        let table = parse(&text);
        assert_eq!(table.len(), ROWS.len(), "row count");
        for (t, r) in table.iter().zip(ROWS) {
            assert_eq!(t.0, r.id);
            assert_eq!(t.1, name(r.fast_f32), "{} fast_f32", r.id);
            assert_eq!(t.2, name(r.precise_f32), "{} precise_f32", r.id);
            assert_eq!(t.3, name(r.half), "{} half", r.id);
        }
    }

    #[test]
    fn rounding_only_rows_are_fast_polyfills() {
        for id in ROUNDING_ONLY {
            assert_eq!(row_of(id).unwrap().fast_f32, Decision::Polyfill, "{}", id);
        }
    }
}
