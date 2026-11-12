//! The binary32 and binary16 soft-float library against the CPU, bit for bit (NaN payloads included).

use crate::gpu::{Gpu, Kernel, MAX_ITEMS, kernel_source, par_range};
use crate::inputs::{Rng, SPECIAL_F16, f32_any, f32_near};
use crate::{prelude, reference as r};
use std::sync::Mutex;

/// Mismatches seen by a sweep: the count and the first few (inputs, got, expected).
#[derive(Default)]
struct Mismatches {
    count: u64,
    first: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>,
}

impl Mismatches {
    fn note(&mut self, inputs: Vec<u32>, got: Vec<u32>, want: Vec<u32>) {
        self.count += 1;
        if self.first.len() < 12 {
            self.first.push((inputs, got, want));
        }
    }

    fn check(self, what: &str, total: u64) {
        let hex = |v: &[u32]| v.iter().map(|w| format!("{:#010x}", w)).collect::<Vec<_>>().join(" ");
        let detail: Vec<String> = self
            .first
            .iter()
            .map(|(i, g, w)| format!("  in [{}] got [{}] want [{}]", hex(i), hex(g), hex(w)))
            .collect();
        assert_eq!(self.count, 0, "{}: {} of {} differ\n{}", what, self.count, total, detail.join("\n"));
        eprintln!("{}: {} inputs, 0 mismatches ({})", what, total, Gpu::get().adapter);
    }
}

/// Every u32 input of a one-operand operation, generated on the GPU: `expr` reads the operand as `a`; the output has
/// `stride` words per input (`out` names them `nm_t_out[i * stride + k]`).
fn sweep_u32(
    what: &str,
    items: &[&str],
    body: &str,
    stride: u32,
    first: u64,
    count: u64,
    want: impl Fn(u32) -> Vec<u32> + Sync,
) {
    let src = kernel_source(false, &prelude(items), "", &format!("let a = nm_t_p.y + i;\n{}", body));
    let k = Kernel::new(&src, stride);
    let m = Mutex::new(Mismatches::default());
    let mut base = first;
    let end = first + count;
    while base < end {
        let n = (end - base).min(MAX_ITEMS as u64) as u32;
        let out = k.run(&[], [base as u32, 0, 0], n);
        let st = stride as usize;
        par_range(n as usize, |range| {
            let mut local = Mismatches::default();
            for j in range {
                let a = (base + j as u64) as u32;
                let got = &out[j * st..(j + 1) * st];
                let w = want(a);
                if w.as_slice() != got {
                    local.note(vec![a], got.to_vec(), w);
                }
            }
            let mut g = m.lock().unwrap();
            g.count += local.count;
            for x in local.first {
                if g.first.len() < 12 {
                    g.first.push(x);
                }
            }
        });
        base += n as u64;
    }
    m.into_inner().unwrap().check(what, count);
}

/// Operations over explicit operand lists (up to three operands per item).
fn run_lists(what: &str, items: &[&str], expr: &str, ops: &[Vec<u32>], want: impl Fn(&[u32]) -> u32 + Sync) {
    let reads = ["nm_t_in0[i]", "nm_t_in1[i]", "nm_t_in2[i]"];
    let mut e = expr.to_string();
    for (k, name) in ["a", "b", "c"].iter().enumerate().take(ops.len()) {
        e = e.replace(&format!("{{{}}}", name), reads[k]);
    }
    let src = kernel_source(false, &prelude(items), "", &format!("nm_t_out[i] = {};", e));
    let k = Kernel::new(&src, 1);
    let total = ops[0].len();
    let mut m = Mismatches::default();
    let mut start = 0;
    while start < total {
        let n = (total - start).min(MAX_ITEMS as usize);
        let slices: Vec<&[u32]> = ops.iter().map(|v| &v[start..start + n]).collect();
        let out = k.run(&slices, [0, 0, 0], n as u32);
        let found = Mutex::new(Mismatches::default());
        par_range(n, |js| {
            let mut local = Mismatches::default();
            for j in js {
                let args: Vec<u32> = slices.iter().map(|s| s[j]).collect();
                let w = want(&args);
                if out[j] != w {
                    local.note(args, vec![out[j]], vec![w]);
                }
            }
            let mut g = found.lock().unwrap();
            g.count += local.count;
            for x in local.first {
                if g.first.len() < 12 {
                    g.first.push(x);
                }
            }
        });
        let f = found.into_inner().unwrap();
        m.count += f.count;
        for x in f.first {
            if m.first.len() < 12 {
                m.first.push(x);
            }
        }
        start += n;
    }
    m.check(what, total as u64);
}

fn operands(seed: u64, n: usize, arity: usize) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    let mut v = vec![Vec::with_capacity(n); arity];
    for _ in 0..n {
        let a = f32_any(&mut rng);
        v[0].push(a);
        if arity > 1 {
            v[1].push(f32_near(&mut rng, a));
        }
        if arity > 2 {
            let b = v[1][v[1].len() - 1];
            // The addend near the product's magnitude makes cancellation and ties likely.
            let p = r::mul(a, b);
            v[2].push(if rng.below(2) == 0 { f32_near(&mut rng, p) } else { f32_any(&mut rng) });
        }
    }
    v
}

#[test]
fn reference_cpu_is_in_its_default_mode() {
    assert_eq!(r::fpcr(), 0, "the references assume FPCR = 0");
}

#[test]
fn reference_rsqrt_is_self_consistent() {
    // 1/sqrt(4^k) is exact; 1/sqrt(2) rounds to 0x3f3504f3 (the binary32 nearest 0.70710678118654752).
    assert_eq!(r::rsqrt(0x3f80_0000), 0x3f80_0000);
    assert_eq!(r::rsqrt(0x4080_0000), 0x3f00_0000);
    assert_eq!(r::rsqrt(0x4000_0000), 0x3f35_04f3);
}

const QUICK: usize = 1 << 20;
const LONG: usize = 1 << 28;

macro_rules! binary_f32 {
    ($quick:ident, $long:ident, $what:literal, $item:literal, $reference:path) => {
        #[test]
        fn $quick() {
            run_lists($what, &[$item], concat!($item, "({a}, {b})"), &operands(1, QUICK, 2), |x| {
                $reference(x[0], x[1])
            });
        }

        #[test]
        #[ignore = "long: run with --ignored"]
        fn $long() {
            run_lists($what, &[$item], concat!($item, "({a}, {b})"), &operands(2, LONG, 2), |x| $reference(x[0], x[1]));
        }
    };
}

binary_f32!(f32_add, f32_add_long, "binary32 add", "nm_f32_add", r::add);
binary_f32!(f32_sub, f32_sub_long, "binary32 sub", "nm_f32_sub", r::sub);
binary_f32!(f32_mul, f32_mul_long, "binary32 mul", "nm_f32_mul", r::mul);
binary_f32!(f32_div, f32_div_long, "binary32 div", "nm_f32_div", r::div);
binary_f32!(f32_fmax, f32_fmax_long, "binary32 fmax", "nm_f32_fmax", r::fmax);
binary_f32!(f32_fmin, f32_fmin_long, "binary32 fmin", "nm_f32_fmin", r::fmin);

#[test]
fn f32_fma() {
    run_lists("binary32 fma", &["nm_f32_fma"], "nm_f32_fma({a}, {b}, {c})", &operands(3, QUICK, 3), |x| {
        r::fma(x[0], x[1], x[2])
    });
}

#[test]
#[ignore = "long: run with --ignored"]
fn f32_fma_long() {
    run_lists("binary32 fma", &["nm_f32_fma"], "nm_f32_fma({a}, {b}, {c})", &operands(4, LONG, 3), |x| {
        r::fma(x[0], x[1], x[2])
    });
}

#[test]
fn f32_compare() {
    let ops = operands(5, QUICK, 2);
    let pred = |name: &str, f: fn(f32, f32) -> bool| {
        run_lists(
            &format!("binary32 {}", name),
            &[&format!("nm_f32_{}", name)],
            &format!("u32(nm_f32_{}({{a}}, {{b}}))", name),
            &ops,
            |x| f(r::f(x[0]), r::f(x[1])) as u32,
        )
    };
    pred("oeq", |a, b| a == b);
    pred("olt", |a, b| a < b);
    pred("ole", |a, b| a <= b);
    pred("uno", |a, b| a.is_nan() || b.is_nan());
}

#[test]
fn f32_ldexp() {
    let mut rng = Rng::new(6);
    let a: Vec<u32> = (0..QUICK).map(|_| f32_any(&mut rng)).collect();
    let n: Vec<u32> = (0..QUICK)
        .map(|_| match rng.below(3) {
            0 => rng.below(600).wrapping_sub(300),
            1 => rng.below(64).wrapping_sub(32),
            _ => rng.u32(),
        })
        .collect();
    run_lists("binary32 ldexp", &["nm_f32_ldexp"], "nm_f32_ldexp({a}, bitcast<i32>({b}))", &[a, n], |x| {
        r::ldexp(x[0], x[1] as i32)
    });
}

// Every binary32 input.
macro_rules! exhaustive_f32 {
    ($name:ident, $what:literal, $item:literal, $reference:expr) => {
        #[test]
        #[ignore = "exhaustive: run with --ignored"]
        fn $name() {
            sweep_u32($what, &[$item], concat!("nm_t_out[i] = ", $item, "(a);"), 1, 0, 1 << 32, |a| {
                vec![$reference(a)]
            });
        }
    };
}

exhaustive_f32!(f32_sqrt_all, "binary32 sqrt", "nm_f32_sqrt", r::sqrt);
exhaustive_f32!(f32_rsqrt_all, "binary32 rsqrt", "nm_f32_rsqrt", r::rsqrt);
exhaustive_f32!(f32_floor_all, "binary32 floor", "nm_f32_floor", |a| r::f(a).floor().to_bits());
exhaustive_f32!(f32_ceil_all, "binary32 ceil", "nm_f32_ceil", |a| r::f(a).ceil().to_bits());
exhaustive_f32!(f32_trunc_all, "binary32 trunc", "nm_f32_trunc", |a| r::f(a).trunc().to_bits());
exhaustive_f32!(f32_rint_all, "binary32 rint", "nm_f32_rint", |a| r::f(a).round_ties_even().to_bits());
exhaustive_f32!(f32_round_all, "binary32 round", "nm_f32_round", |a| r::f(a).round().to_bits());
exhaustive_f32!(f32_to_f16_all, "binary32 to binary16", "nm_f32_to_f16", |a| r::f32_to_f16(a) as u32);
exhaustive_f32!(f32_to_i32_all, "binary32 to i32", "nm_f32_to_i32", |a| r::f(a) as i32 as u32);
exhaustive_f32!(f32_to_u32_all, "binary32 to u32", "nm_f32_to_u32", |a| r::f(a) as u32);
exhaustive_f32!(f32_to_i16_all, "binary32 to i16", "nm_f32_to_i16", |a| r::f(a) as i16 as u16 as u32);
exhaustive_f32!(f32_to_u16_all, "binary32 to u16", "nm_f32_to_u16", |a| r::f(a) as u16 as u32);
exhaustive_f32!(f32_to_i8_all, "binary32 to i8", "nm_f32_to_i8", |a| r::f(a) as i8 as u8 as u32);
exhaustive_f32!(f32_to_u8_all, "binary32 to u8", "nm_f32_to_u8", |a| r::f(a) as u8 as u32);
exhaustive_f32!(u32_to_f32_all, "u32 to binary32", "nm_f32_from_u32", |a| (a as f32).to_bits());
exhaustive_f32!(i32_to_f32_all, "i32 to binary32", "nm_f32_from_i32", |a| (a as i32 as f32).to_bits());
exhaustive_f32!(u32_to_f16_all, "u32 to binary16", "nm_f16_from_u32", |a| r::u32_to_f16(a) as u32);
exhaustive_f32!(i32_to_f16_all, "i32 to binary16", "nm_f16_from_i32", |a| r::i32_to_f16(a) as u32);

#[test]
#[ignore = "exhaustive: run with --ignored"]
fn f32_to_64_all() {
    sweep_u32(
        "binary32 to i64",
        &["nm_f32_to_i64"],
        "let v = nm_f32_to_i64(a); nm_t_out[2u * i] = v.x; nm_t_out[2u * i + 1u] = v.y;",
        2,
        0,
        1 << 32,
        |a| {
            let v = r::f(a) as i64 as u64;
            vec![v as u32, (v >> 32) as u32]
        },
    );
    sweep_u32(
        "binary32 to u64",
        &["nm_f32_to_u64"],
        "let v = nm_f32_to_u64(a); nm_t_out[2u * i] = v.x; nm_t_out[2u * i + 1u] = v.y;",
        2,
        0,
        1 << 32,
        |a| {
            let v = r::f(a) as u64;
            vec![v as u32, (v >> 32) as u32]
        },
    );
}

/// 64-bit integers to binary32 and binary16: random values over every magnitude.
#[test]
fn int64_to_float() {
    let mut rng = Rng::new(7);
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    for _ in 0..QUICK {
        let bits = rng.below(65);
        let v = if bits == 0 { 0 } else { rng.next() >> (64 - bits) };
        let v = match rng.below(4) {
            0 => v | 1,
            1 => v & !((1u64 << rng.below(40)) - 1),
            _ => v,
        };
        lo.push(v as u32);
        hi.push((v >> 32) as u32);
    }
    let ops = [lo, hi];
    let v = |x: &[u32]| x[0] as u64 | (x[1] as u64) << 32;
    run_lists("u64 to binary32", &["nm_f32_from_u64"], "nm_f32_from_u64(vec2<u32>({a}, {b}))", &ops, |x| {
        (v(x) as f32).to_bits()
    });
    run_lists("i64 to binary32", &["nm_f32_from_i64"], "nm_f32_from_i64(vec2<u32>({a}, {b}))", &ops, |x| {
        (v(x) as i64 as f32).to_bits()
    });
    run_lists("u64 to binary16", &["nm_f16_from_u64"], "nm_f16_from_u64(vec2<u32>({a}, {b}))", &ops, |x| {
        r::u64_to_f16(v(x)) as u32
    });
    run_lists("i64 to binary16", &["nm_f16_from_i64"], "nm_f16_from_i64(vec2<u32>({a}, {b}))", &ops, |x| {
        r::i64_to_f16(v(x)) as u32
    });
}

/// Every binary16 input of the one-operand operations.
#[test]
fn f16_unary_all() {
    sweep_u32("binary16 to binary32", &["nm_f16_to_f32"], "nm_t_out[i] = nm_f16_to_f32(a);", 1, 0, 1 << 16, |a| {
        vec![r::f16_to_f32(a as u16)]
    });
    sweep_u32("binary16 sqrt", &["nm_f16_sqrt"], "nm_t_out[i] = nm_f16_sqrt(a);", 1, 0, 1 << 16, |a| {
        vec![r::h_sqrt(a as u16) as u32]
    });
    sweep_u32("binary16 rsqrt", &["nm_f16_rsqrt"], "nm_t_out[i] = nm_f16_rsqrt(a);", 1, 0, 1 << 16, |a| {
        // rsqrt rounded once to binary16: from the exact binary32 square root of the widened value, the correctly
        // rounded reciprocal square root is found by exact comparison against neighbouring binary16 midpoints.
        vec![h_rsqrt(a as u16) as u32]
    });
}

/// Correctly rounded binary16 1/sqrt(h), with the rSqrt special cases.
fn h_rsqrt(h: u16) -> u16 {
    let x = r::f16_to_f32(h);
    let xf = r::f(x);
    if xf.is_nan() {
        return h | 0x0200;
    }
    if h & 0x7fff == 0 {
        return h | 0x7c00;
    }
    if xf < 0.0 {
        return 0x7e00;
    }
    if xf.is_infinite() {
        return 0;
    }
    // A nearby candidate, then the exact test: the value lies below a midpoint m exactly when m^2 * x > 1, and
    // binary64 holds m^2 * x exactly (m has at most 12 significant bits, x at most 11).
    let t = 1.0 / (xf as f64).sqrt();
    let mut c = r::f32_to_f16((t as f32).to_bits());
    let val = |c: u16| r::f(r::f16_to_f32(c)) as f64;
    let mid_above = |c: u16| (val(c) + val(c + 1)) / 2.0;
    loop {
        if c > 0 && mid_above(c - 1) * mid_above(c - 1) * (xf as f64) > 1.0 {
            c -= 1;
            continue;
        }
        if mid_above(c) * mid_above(c) * (xf as f64) < 1.0 {
            c += 1;
            continue;
        }
        break;
    }
    c
}

/// Every pair of binary16 operands of add, sub, mul and div.
macro_rules! exhaustive_f16 {
    ($name:ident, $what:literal, $item:literal, $reference:path) => {
        #[test]
        #[ignore = "exhaustive: run with --ignored"]
        fn $name() {
            sweep_u32(
                $what,
                &[$item],
                concat!("nm_t_out[i] = ", $item, "(a & 0xffffu, a >> 16u);"),
                1,
                0,
                1 << 32,
                |a| vec![$reference(a as u16, (a >> 16) as u16) as u32],
            );
        }
    };
}

exhaustive_f16!(f16_add_all, "binary16 add", "nm_f16_add", r::h_add);
exhaustive_f16!(f16_sub_all, "binary16 sub", "nm_f16_sub", r::h_sub);
exhaustive_f16!(f16_mul_all, "binary16 mul", "nm_f16_mul", r::h_mul);
exhaustive_f16!(f16_div_all, "binary16 div", "nm_f16_div", r::h_div);

/// binary16 binary operations on a sample of pairs (the full sweep is f16_*_all).
#[test]
fn f16_binary_sample() {
    let mut rng = Rng::new(8);
    let mut a = Vec::new();
    let mut b = Vec::new();
    for _ in 0..QUICK {
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
    let ops = [a, b];
    run_lists("binary16 add", &["nm_f16_add"], "nm_f16_add({a}, {b})", &ops, |x| {
        r::h_add(x[0] as u16, x[1] as u16) as u32
    });
    run_lists("binary16 sub", &["nm_f16_sub"], "nm_f16_sub({a}, {b})", &ops, |x| {
        r::h_sub(x[0] as u16, x[1] as u16) as u32
    });
    run_lists("binary16 mul", &["nm_f16_mul"], "nm_f16_mul({a}, {b})", &ops, |x| {
        r::h_mul(x[0] as u16, x[1] as u16) as u32
    });
    run_lists("binary16 div", &["nm_f16_div"], "nm_f16_div({a}, {b})", &ops, |x| {
        r::h_div(x[0] as u16, x[1] as u16) as u32
    });
}

#[test]
fn f16_fma_sample() {
    let mut rng = Rng::new(9);
    let n = if cfg!(debug_assertions) { QUICK } else { 1 << 26 };
    let mut v = vec![Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..n {
        for k in v.iter_mut() {
            k.push(if rng.below(8) == 0 {
                SPECIAL_F16[rng.below(SPECIAL_F16.len() as u32) as usize] as u32
            } else {
                rng.below(1 << 16)
            });
        }
    }
    run_lists("binary16 fma", &["nm_f16_fma"], "nm_f16_fma({a}, {b}, {c})", &v, |x| {
        r::h_fma(x[0] as u16, x[1] as u16, x[2] as u16) as u32
    });
}
