//! The fast-math rows numerics.toml marks native, measured on this host: the lowering emits WGSL's own operation,
//! whose accuracy WGSL bounds at least as tightly as MSL; the host's measured error must be within MSL's bound
//! (MSL Table 8.2). A host outside it is a host bug, and the row would move to the polyfill on that host.

use crate::gpu::{kernel_source, Gpu, Kernel, MAX_ITEMS};
use crate::inputs::Rng;
use crate::math::ulp32;
use crate::reference as r;
use airwgsl::api::HostCaps;
use airwgsl::num::{lower, FastMath, How, Inst, Op, Operand, Rounding, Scalar, Target, Ty};

fn native_kernel(name: &str, arity: usize) -> Kernel {
    native_op(Op::Call(name), arity)
}

fn native_op(op: Op, arity: usize) -> Kernel {
    let caps = HostCaps::default();
    let soft = Rounding::SoftFloat;
    let t = Ty::scalar(Scalar::F32);
    let names = ["nm_t_in0[i]", "nm_t_in1[i]"];
    let ops: Vec<Operand> = names[..arity].iter().map(|e| Operand { expr: e, ty: t, constant: None }).collect();
    let l = lower(op, &ops, Some(t), FastMath::FAST, Target { rounding: &soft, host: &caps }, "v").unwrap();
    assert_eq!(l.rule.how, How::Native, "{:?} is a native row", op);
    let body = format!("{}\nnm_t_out[i] = {};", l.stmts.join("\n"), l.value.unwrap());
    Kernel::new(&kernel_source(false, &l.prelude.emit().unwrap(), "", &body), 1)
}

/// Inputs uniform in log scale over [2^lo, 2^hi), with random signs when `signed`.
fn log_uniform(rng: &mut Rng, n: usize, lo: i32, hi: i32, signed: bool) -> Vec<u32> {
    (0..n)
        .map(|_| {
            let e = (lo + rng.below((hi - lo) as u32) as i32 + 127) as u32;
            let s = if signed { rng.below(2) << 31 } else { 0 };
            s | e << 23 | (rng.u32() & 0x7f_ffff)
        })
        .collect()
}

#[test]
fn native_rows_meet_msl_fast_bounds_on_this_host() {
    let mut rng = Rng::new(51);
    let n = MAX_ITEMS as usize / 16;
    // x / y (fdiv with fast flags, row div): at most 2.5 ulp for y in [2^-126, 2^126] (Table 8.2).
    let x = log_uniform(&mut rng, n, -60, 60, true);
    let y = log_uniform(&mut rng, n, -60, 60, true);
    let mut report = Vec::new();
    let got = native_op(Op::Inst(Inst::FDiv), 2).run(&[&x, &y], [0, 0, 0], n as u32);
    let mut max = 0f64;
    for ((&a, &b), &g) in x.iter().zip(&y).zip(&got) {
        let want = r::f(a) as f64 / r::f(b) as f64;
        max = max.max((r::f(g) as f64 - want).abs() / ulp32(want));
    }
    report.push(("divide", max, 2.5));
    // rsqrt: at most 2 ulp (Table 8.2).
    let a = log_uniform(&mut rng, n, -126, 127, false);
    let got = native_kernel("air.fast_rsqrt.f32", 1).run(&[&a], [0, 0, 0], n as u32);
    let mut max = 0f64;
    for (&x, &g) in a.iter().zip(&got) {
        let want = 1.0 / (r::f(x) as f64).sqrt();
        max = max.max((r::f(g) as f64 - want).abs() / ulp32(want));
    }
    report.push(("rsqrt", max, 2.0));
    // log: absolute error at most 2^-21 on [0.5, 2], otherwise at most 3 ulp (Table 8.2).
    let a = log_uniform(&mut rng, n, -126, 127, false);
    let got = native_kernel("air.fast_log.f32", 1).run(&[&a], [0, 0, 0], n as u32);
    let (mut max_abs, mut max_ulp) = (0f64, 0f64);
    for (&x, &g) in a.iter().zip(&got) {
        let xv = r::f(x) as f64;
        let want = xv.ln();
        let err = (r::f(g) as f64 - want).abs();
        if (0.5..=2.0).contains(&xv) {
            max_abs = max_abs.max(err);
        } else {
            max_ulp = max_ulp.max(err / ulp32(want));
        }
    }
    report.push(("log on [0.5, 2] (absolute, in units of 2^-21)", max_abs / 2f64.powi(-21), 1.0));
    report.push(("log elsewhere", max_ulp, 3.0));
    // Exact rows: ceil, floor, trunc, rint, fabs, fmax, fmin, ldexp (correctly rounded or exact selections).
    let a: Vec<u32> = log_uniform(&mut rng, n, -30, 40, true);
    for (name, f) in [
        ("air.fast_ceil.f32", (|x: f32| x.ceil()) as fn(f32) -> f32),
        ("air.fast_floor.f32", |x| x.floor()),
        ("air.fast_trunc.f32", |x| x.trunc()),
        ("air.fast_rint.f32", |x| x.round_ties_even()),
        ("air.fast_fabs.f32", |x| x.abs()),
    ] {
        let got = native_kernel(name, 1).run(&[&a], [0, 0, 0], n as u32);
        let bad = a.iter().zip(&got).filter(|(x, g)| f(r::f(**x)).to_bits() != **g && !(f(r::f(**x)) == 0.0 && r::f(**g) == 0.0)).count();
        report.push((name, bad as f64, 0.0));
    }
    let b: Vec<u32> = log_uniform(&mut rng, n, -30, 40, true);
    for (name, f) in [("air.fast_fmax.f32", (|x: f32, y: f32| x.max(y)) as fn(f32, f32) -> f32), ("air.fast_fmin.f32", |x, y| x.min(y))] {
        let got = native_kernel(name, 2).run(&[&a, &b], [0, 0, 0], n as u32);
        let bad = a.iter().zip(&b).zip(&got).filter(|((x, y), g)| f(r::f(**x), r::f(**y)).to_bits() != **g).count();
        report.push((name, bad as f64, 0.0));
    }
    eprintln!("native rows on {}:", Gpu::get().adapter);
    for (what, err, bound) in &report {
        eprintln!("  {}: {} (bound {})", what, err, bound);
    }
    for (what, err, bound) in report {
        assert!(err <= bound, "{}: {} exceeds MSL's bound {}", what, err, bound);
    }
}
