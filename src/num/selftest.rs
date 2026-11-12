//! The host rounding self-test: does the device's native binary32 `+`, `-`, `*` and `fma` round to nearest even?
//!
//! MSL's correctly rounded `+ - *` admit round-to-nearest-even or round-toward-zero results (MSL 8.2), while WGSL's
//! admit either neighbour (WGSL 15.7.4.1, numerics.toml rule R1). The default lowering is therefore integer
//! soft-float. This probe gathers the evidence for the alternative: it runs native WGSL arithmetic on cases built to
//! tell the rounding rules apart and compares every result with round-to-nearest-even. A host whose results all
//! agree may be given [`Rounding::Native`](super::Rounding) through [`Report::native_rounding`]; the choice of policy
//! is the owner's (numerics.toml question 5.3-rounding-direction).
//!
//! How the probe keeps its answers meaningful:
//!
//! - Operands come from a storage buffer at run time, so no compiler can fold the operations with its own rounding.
//! - Each invocation performs exactly one operation and stores its result, so there is nothing to reassociate or to
//!   fuse (WGSL 15.7.5 permits both); `fma` is tested through the `fma` built-in.
//! - The cases are exact ties with even and odd neighbours (where round-to-nearest-even differs from ties-away and from
//!   every directed mode), values just above and below a tie, both signs (directed modes are asymmetric), carries into
//!   the exponent, cancellation, overflow at the largest finite value, products of few-bit significands with exact
//!   ties, fma cases whose single rounding differs from a rounded product followed by a rounded sum, and a seeded
//!   random set. Expected values are computed on the CPU, which rounds to nearest even.
//! - Results that are subnormal, and operands that are, are reported separately: MSL lets a device flush them
//!   (MSL 8.1), so they do not decide the rounding question.
//!
//! A probe cannot be exhaustive (2^64 operand pairs): passing it is evidence, not proof.

use super::NativeRounding;

/// The operation of a case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeOp {
    Add,
    Sub,
    Mul,
    Fma,
}

impl ProbeOp {
    fn code(self) -> u32 {
        match self {
            ProbeOp::Add => 0,
            ProbeOp::Sub => 1,
            ProbeOp::Mul => 2,
            ProbeOp::Fma => 3,
        }
    }
}

/// One case: operands as binary32 bits and the round-to-nearest-even result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Case {
    pub op: ProbeOp,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub want: u32,
}

fn f(b: u32) -> f32 {
    f32::from_bits(b)
}

fn subnormal(x: u32) -> bool {
    x & 0x7f80_0000 == 0 && x & 0x007f_ffff != 0
}

fn expected(op: ProbeOp, a: u32, b: u32, c: u32) -> u32 {
    match op {
        ProbeOp::Add => (f(a) + f(b)).to_bits(),
        ProbeOp::Sub => (f(a) - f(b)).to_bits(),
        ProbeOp::Mul => (f(a) * f(b)).to_bits(),
        ProbeOp::Fma => f(a).mul_add(f(b), f(c)).to_bits(),
    }
}

/// splitmix64, for a reproducible random part.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// The probe's cases, in a fixed order.
pub fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let mut push = |op: ProbeOp, a: u32, b: u32, c: u32| {
        let want = expected(op, a, b, c);
        // Only finite cases: NaN payloads say nothing about rounding.
        if f(want).is_nan() {
            return;
        }
        v.push(Case { op, a, b, c, want });
    };
    // Additions and subtractions around exact ties: a has significand m (even or odd), b is half an ulp of a, a
    // little more, or a little less, with every sign combination and exponents across the range.
    for e in [1u32, 20, 100, 126, 127, 128, 150, 200, 253] {
        for m in [0u32, 1, 2, 3, 0x7f_fffe, 0x7f_ffff, 0x40_0000, 0x40_0001, 0x12_3457, 0x12_3456] {
            let a = (e << 23) | m;
            // Half an ulp of a is 2^(e - 151) when e >= 25.
            if e < 25 {
                continue;
            }
            let half = (e - 24) << 23;
            for b in [half, half | 1, half - 1, half | 0x40_0000, (e - 23) << 23, (e - 25) << 23 | 0x7f_ffff] {
                for sa in [0u32, 0x8000_0000] {
                    for sb in [0u32, 0x8000_0000] {
                        push(ProbeOp::Add, a | sa, b | sb, 0);
                        push(ProbeOp::Sub, a | sa, b | sb, 0);
                        push(ProbeOp::Add, b | sb, a | sa, 0);
                    }
                }
            }
        }
    }
    // Carries into the exponent and cancellation.
    for (a, b) in [
        (0x3fff_ffffu32, 0x3400_0000u32),
        (0x3f7f_ffff, 0x3380_0000),
        (0x3f80_0000, 0xbf7f_ffff),
        (0x4b7f_ffff, 0x3f00_0000),
    ] {
        push(ProbeOp::Add, a, b, 0);
        push(ProbeOp::Sub, a, b ^ 0x8000_0000, 0);
        push(ProbeOp::Add, a ^ 0x8000_0000, b ^ 0x8000_0000, 0);
    }
    // Overflow at the largest finite value: a tie rounds to even, which is infinity.
    for b in [0x7300_0000u32, 0x72ff_ffff, 0x7300_0001] {
        push(ProbeOp::Add, 0x7f7f_ffff, b, 0);
        push(ProbeOp::Add, 0xff7f_ffff, b ^ 0x8000_0000, 0);
    }
    // Products of few-bit significands: (2^12 + i)(2^12 + j) has up to 26 significant bits, so ties and near-ties at
    // binary32's 24 are common.
    for i in (1u32..64).step_by(3) {
        for j in (1u32..64).step_by(5) {
            for e in [127u32, 100, 150] {
                let a = (e << 23) | (i << 11);
                let b = (127 << 23) | (j << 11) | (i & 1);
                push(ProbeOp::Mul, a, b, 0);
                push(ProbeOp::Mul, a | 0x8000_0000, b, 0);
                // fma: the product exactly, minus its rounded value, leaves the rounding error, which an unfused
                // evaluation loses.
                let p = (f(a) * f(b)).to_bits();
                push(ProbeOp::Fma, a, b, p ^ 0x8000_0000);
                push(ProbeOp::Fma, a, b, 0x3380_0000);
                push(ProbeOp::Fma, a | 0x8000_0000, b, 0x3380_0000);
            }
        }
    }
    // A seeded random set over normal operands with nearby exponents.
    let mut r = Mix(0x005e_ed0f_f001);
    for k in 0..4096u32 {
        let x = r.next();
        let ea = 64 + (x as u32 % 128);
        let eb = (ea as i32 + ((x >> 8) as i32 % 30) - 15).clamp(1, 254) as u32;
        let a = ((x >> 32) as u32 & 0x807f_ffff) | (ea << 23);
        let b = ((x >> 20) as u32 & 0x807f_ffff) | (eb << 23);
        let c = ((x >> 3) as u32 & 0x807f_ffff) | (((ea + eb).saturating_sub(127).clamp(1, 254)) << 23);
        let op = [ProbeOp::Add, ProbeOp::Sub, ProbeOp::Mul, ProbeOp::Fma][(k % 4) as usize];
        push(op, a, b, c);
    }
    // Subnormal results, reported separately.
    for b in [0x0080_0001u32, 0x0080_0003, 0x0100_0000] {
        push(ProbeOp::Sub, 0x0080_0000 | 2, b, 0);
        push(ProbeOp::Mul, 0x0080_0001, 0x3f00_0001, 0);
    }
    v
}

/// The probe shader: binding 0 holds one `vec4<u32>` (op, a, b, c) per case, binding 1 receives the results.
/// Dispatch `ceil(cases / 64)` workgroups.
pub fn probe_wgsl() -> String {
    "@group(0) @binding(0) var<storage, read> nm_st_in: array<vec4<u32>>;\n\
     @group(0) @binding(1) var<storage, read_write> nm_st_out: array<u32>;\n\
     @compute @workgroup_size(64)\n\
     fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n\
     \x20   let i = g.x;\n\
     \x20   if (i >= arrayLength(&nm_st_in)) {\n\
     \x20       return;\n\
     \x20   }\n\
     \x20   let c = nm_st_in[i];\n\
     \x20   let x = bitcast<f32>(c.y);\n\
     \x20   let y = bitcast<f32>(c.z);\n\
     \x20   let z = bitcast<f32>(c.w);\n\
     \x20   var r: f32;\n\
     \x20   switch (c.x) {\n\
     \x20       case 0u: {\n\
     \x20           r = x + y;\n\
     \x20       }\n\
     \x20       case 1u: {\n\
     \x20           r = x - y;\n\
     \x20       }\n\
     \x20       case 2u: {\n\
     \x20           r = x * y;\n\
     \x20       }\n\
     \x20       default: {\n\
     \x20           r = fma(x, y, z);\n\
     \x20       }\n\
     \x20   }\n\
     \x20   nm_st_out[i] = bitcast<u32>(r);\n\
     }\n"
    .to_string()
}

/// The probe input for `cases`: four words per case.
pub fn probe_input(cases: &[Case]) -> Vec<u32> {
    cases.iter().flat_map(|c| [c.op.code(), c.a, c.b, c.c]).collect()
}

/// Per-operation tallies over normal operands and results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub cases: u32,
    pub failures: u32,
}

/// The outcome of a probe run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub add_sub: Tally,
    pub mul: Tally,
    pub fma: Tally,
    /// Cases with a subnormal operand or result: how many there were and how many agreed with gradual underflow.
    pub subnormal: Tally,
    /// The first failing cases with the device's results.
    pub failures: Vec<(Case, u32)>,
}

/// Compare the device's results with the expected ones.
pub fn evaluate(cases: &[Case], results: &[u32]) -> Report {
    let mut r = Report::default();
    for (c, &got) in cases.iter().zip(results) {
        let ok = got == c.want;
        let sub = subnormal(c.a) || subnormal(c.b) || (c.op == ProbeOp::Fma && subnormal(c.c)) || subnormal(c.want);
        let t = if sub {
            &mut r.subnormal
        } else {
            match c.op {
                ProbeOp::Add | ProbeOp::Sub => &mut r.add_sub,
                ProbeOp::Mul => &mut r.mul,
                ProbeOp::Fma => &mut r.fma,
            }
        };
        t.cases += 1;
        if !ok {
            t.failures += 1;
            if r.failures.len() < 32 {
                r.failures.push((*c, got));
            }
        }
    }
    r
}

impl Report {
    /// The native rounding policy this run supports: `None` unless every normal add, sub and mul case rounded to
    /// nearest even; fma is admitted separately when all its cases rounded once to nearest even.
    pub fn native_rounding(&self) -> Option<NativeRounding> {
        if self.add_sub.cases == 0 || self.mul.cases == 0 || self.add_sub.failures != 0 || self.mul.failures != 0 {
            return None;
        }
        Some(NativeRounding {
            fused_fma: self.fma.cases != 0 && self.fma.failures == 0,
            evidence: format!(
                "add/sub {}/{}, mul {}/{}, fma {}/{} round to nearest even; subnormal cases {}/{} with gradual underflow",
                self.add_sub.cases - self.add_sub.failures,
                self.add_sub.cases,
                self.mul.cases - self.mul.failures,
                self.mul.cases,
                self.fma.cases - self.fma.failures,
                self.fma.cases,
                self.subnormal.cases - self.subnormal.failures,
                self.subnormal.cases
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_contain_ties_that_separate_the_rounding_modes() {
        let cs = cases();
        assert!(cs.len() > 5000);
        // An exact tie with an odd significand rounds up in magnitude under nearest-even and down under toward-zero.
        let tie = cs.iter().find(|c| c.op == ProbeOp::Add && c.a == (127 << 23 | 1) && c.b == 103 << 23).unwrap();
        assert_eq!(tie.want, 127 << 23 | 2);
        // The fused case keeps the product's rounding error.
        assert!(cs.iter().any(|c| c.op == ProbeOp::Fma && c.want != (f(c.a) * f(c.b) + f(c.c)).to_bits()));
    }

    #[test]
    fn a_report_without_failures_admits_native_rounding() {
        let cs = cases();
        let want: Vec<u32> = cs.iter().map(|c| c.want).collect();
        let r = evaluate(&cs, &want);
        let n = r.native_rounding().unwrap();
        assert!(n.fused_fma());
        let mut wrong = want.clone();
        let k = cs.iter().position(|c| c.op == ProbeOp::Mul && !subnormal(c.want)).unwrap();
        wrong[k] ^= 1;
        assert!(evaluate(&cs, &wrong).native_rounding().is_none());
    }
}
