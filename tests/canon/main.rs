//! Canonicalization tests: each pass's work on IR fixtures (`ir/*.ll`, assembled to `ir/*.bc` by `ir/build.sh`),
//! checked by structure and by running the function before and after on the interpreter; the function constant
//! rules against what Metal does (`data/function-constants.txt`, measured by `metal/function_constants.swift`).
//!
//! The corpus-wide check is `examples/canon_check.rs`; the comparison with real Metal is `examples/metal_diff.rs`
//! with `metal/run_kernels.swift`.

mod harness;

use airwgsl::api::{FunctionConstant, RefusalKind};
use airwgsl::interp::{Byte, Externals, Scalar, Val};
use airwgsl::ir::{Module, Op};
use airwgsl::passes::{self, Canonical, specialize};
use harness::{Inputs, Rng, Verdict};

fn fixture(bytes: &[u8]) -> Module {
    airwgsl::parse_bitcode(bytes).expect("fixture bitcode")
}

fn ops(c: &Canonical) -> Vec<Op> {
    let f = &c.module.functions[c.function];
    f.blocks.iter().flat_map(|b| b.insts.iter().map(|&i| f.inst(i).op.clone())).collect()
}

fn count(c: &Canonical, pred: impl Fn(&Op) -> bool) -> usize {
    ops(c).iter().filter(|o| pred(o)).count()
}

/// Runs `test` of the original and of the canonical module on the same inputs and requires refinement for each,
/// with at least one defined original run.
fn check_refines(m: &Module, c: &Canonical, constants: &[FunctionConstant], runs: u64) {
    let f = m.function_by_name("test").unwrap();
    let kinds = harness::arg_kinds(m, f);
    let mut defined = 0;
    for seed in 0..runs {
        let mut rng = Rng(seed * 7919 + 1);
        let invs: Vec<Inputs> = (0..2).map(|t| harness::inputs(m, f, &kinds, &mut rng, t)).collect();
        let a = harness::run(m, f, &kinds, &invs, Some(constants), Externals::Refuse);
        let b = harness::run(&c.module, c.function, &kinds, &invs, None, Externals::Refuse);
        match harness::compare(&a, &b) {
            Verdict::Refines => defined += 1,
            Verdict::FirstUb(_) => {}
            other => panic!("seed {}: {:?}", seed, other),
        }
    }
    assert!(defined > 0, "every original run had undefined behavior");
}

fn canon(m: &Module, constants: &[FunctionConstant]) -> Canonical {
    passes::canonicalize(std::slice::from_ref(m), "test", constants).unwrap_or_else(|r| panic!("{}", r))
}

#[test]
fn inlining_removes_calls_and_keeps_results() {
    let m = fixture(include_bytes!("ir/inline.bc"));
    let c = canon(&m, &[]);
    assert_eq!(count(&c, |o| matches!(o, Op::Call { .. })), 0);
    assert_eq!(count(&c, |o| matches!(o, Op::Alloca { .. })), 0);
    assert!(count(&c, |o| matches!(o, Op::Phi { .. })) >= 3);
    check_refines(&m, &c, &[], 16);
}

#[test]
fn recursion_is_refused() {
    let m = fixture(include_bytes!("ir/recursion.bc"));
    let r = passes::canonicalize(std::slice::from_ref(&m), "test", &[]).unwrap_err();
    assert_eq!(r.kind, RefusalKind::NoExactLowering);
    assert!(r.reason.contains("recursion"));
}

#[test]
fn mem2reg_places_phis_and_reads_undef_before_stores() {
    let m = fixture(include_bytes!("ir/mem2reg.bc"));
    let c = canon(&m, &[]);
    assert_eq!(count(&c, |o| matches!(o, Op::Alloca { .. } | Op::Load { .. } | Op::Store { .. })), 0);
    assert_eq!(count(&c, |o| matches!(o, Op::Phi { .. })), 3);
    check_refines(&m, &c, &[], 16);
}

#[test]
fn sroa_splits_and_promotes_what_constant_offsets_reach() {
    let m = fixture(include_bytes!("ir/sroa.bc"));
    let c = canon(&m, &[]);
    // Only the dynamically indexed array stays in memory.
    assert_eq!(count(&c, |o| matches!(o, Op::Alloca { .. })), 1);
    assert_eq!(count(&c, |o| matches!(o, Op::Call { .. })), 0);
    check_refines(&m, &c, &[], 16);
}

#[test]
fn folding_uses_exact_semantics_only() {
    let m = fixture(include_bytes!("ir/fold.bc"));
    let c = canon(&m, &[]);
    // One block: the constant branch and switch are gone.
    assert_eq!(c.module.functions[c.function].blocks.len(), 1);
    // The mutable global's load stays; the NaN-producing division stays (its sign and payload are choices).
    assert_eq!(count(&c, |o| matches!(o, Op::Load { .. })), 1);
    assert_eq!(count(&c, |o| matches!(o, Op::Binary { op: airwgsl::ir::BinOp::FDiv, .. })), 1);
    check_refines(&m, &c, &[], 8);
}

#[test]
fn linking_brings_definitions_and_their_dependencies() {
    let main = fixture(include_bytes!("ir/link_main.bc"));
    let lib = fixture(include_bytes!("ir/link_lib.bc"));
    let c = passes::canonicalize(&[main.clone(), lib.clone()], "test", &[]).unwrap();
    assert_eq!(count(&c, |o| matches!(o, Op::Call { .. } | Op::Load { .. })), 0);
    // helper(x) = (x + 100) << 1.
    let mut mem = airwgsl::interp::Memory::new();
    let out = airwgsl::interp::run(
        &c.module,
        c.function,
        &[Val::S(Scalar::int(5))],
        &mut mem,
        &airwgsl::interp::Options::default(),
    )
    .unwrap();
    assert_eq!(out.ret, Some(Val::S(Scalar::int(210))));
    // Without the library the call is refused: nothing defines @helper.
    let r = passes::canonicalize(&[main], "test", &[]).unwrap_err();
    assert!(r.construct.contains("@helper"));
}

#[test]
fn static_initializers_fold_into_constants() {
    let m = fixture(include_bytes!("ir/static_init.bc"));
    let with = |i: u64, ty: u64, v: &[u8]| FunctionConstant { index: i, name: None, data_type: ty, value: v.to_vec() };
    // fc_i = 5, fc_b = true: derived = 10, the predicate keeps its argument.
    let consts = [with(0, 29, &[5, 0, 0, 0]), with(1, 53, &[1])];
    let c = canon(&m, &consts);
    let stores: Vec<Op> = ops(&c).into_iter().filter(|o| matches!(o, Op::Store { .. })).collect();
    assert_eq!(stores.len(), 1);
    let Op::Store { value, .. } = &stores[0] else { unreachable!() };
    assert_eq!(passes::util::const_int(&c.module, *value), Some(10));
    let pred = c.module.globals.iter().position(|g| g.common.name == "pred").unwrap();
    let mut mem = airwgsl::interp::Memory::new();
    let r = airwgsl::interp::global_region(&c.module, pred as u32, &mut mem).unwrap();
    assert!(specialize::predicate_present(&mem.region(r).bytes[..1]));
    assert!(c.module.globals[pred].constant);
    check_refines(&m, &c, &consts, 4);
    // fc_i unset: the default 7 applies; fc_b unset: the predicate is undef and drops its argument.
    let c = canon(&m, &[]);
    let Some(Op::Store { value, .. }) = ops(&c).into_iter().find(|o| matches!(o, Op::Store { .. })) else { panic!() };
    assert_eq!(passes::util::const_int(&c.module, value), Some(14));
    let pred = c.module.globals.iter().position(|g| g.common.name == "pred").unwrap();
    let mut mem = airwgsl::interp::Memory::new();
    let r = airwgsl::interp::global_region(&c.module, pred as u32, &mut mem).unwrap();
    assert!(!specialize::predicate_present(&mem.region(r).bytes[..1]));
    // A value of another type is refused, as Metal refuses it.
    let r = passes::canonicalize(std::slice::from_ref(&m), "test", &[with(0, 33, &[5, 0, 0, 0])]).unwrap_err();
    assert_eq!(r.kind, RefusalKind::RejectedByDevice);
    // The canonical module keeps no constructor, no function constant global and no other function body.
    assert!(c.module.globals.iter().all(|g| g.common.name != "llvm.global_ctors" && !g.common.name.ends_with(".init")));
    assert!(c.module.functions.iter().enumerate().all(|(i, f)| i == c.function || f.is_declaration()));
}

/// The specialization rules against the measured Metal behavior.
#[test]
fn function_constants_follow_metal() {
    let data = include_str!("data/function-constants.txt");
    let mut predicates = 0;
    let mut typed = 0;
    for line in data.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split(' ').collect();
        match f[0] {
            "predicate" => {
                let bytes: Vec<Byte> = if f[2] == "unset" {
                    vec![Byte::UNDEF; 4]
                } else {
                    (0..f[2].len())
                        .step_by(2)
                        .map(|k| Byte::data(u8::from_str_radix(&f[2][k..k + 2], 16).unwrap()))
                        .collect()
                };
                assert_eq!(specialize::predicate_present(&bytes), f[3] == "kept", "{}", line);
                predicates += 1;
            }
            "typed" => {
                let want = specialize::data_type(f[1]).unwrap();
                let given: u64 = f[2].parse().unwrap();
                // Metal accepts exactly the matching type.
                assert_eq!(f[4] == "error", want != given, "{}", line);
                typed += 1;
            }
            "missing" => {
                // A used constant without a value: no error, the use is undef (Metal deletes the store of it).
                assert_eq!(f[2], "ok", "{}", line);
            }
            _ => panic!("unknown line {}", line),
        }
    }
    assert!(predicates >= 20 && typed >= 9);
}

/// The canonicalization over the generated corpus's fuzz kernels, when the corpus is on this machine
/// (`AIRWGSL_CORPUS`, the directory holding `build/air`).
#[test]
#[ignore]
fn corpus_fuzz_kernels() {
    let Some(root) = std::env::var_os("AIRWGSL_CORPUS") else { return };
    let dir = std::path::Path::new(&root).join("build/air/3.1");
    let mut checked = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if !name.starts_with("fuzz_") || !name.ends_with(".air") {
            continue;
        }
        let m = airwgsl::parse_bitcode(&std::fs::read(&p).unwrap()).unwrap();
        for (f, entry) in harness::entry_points(&m) {
            let c = passes::canonicalize(std::slice::from_ref(&m), &entry, &[]).unwrap();
            let kinds = harness::arg_kinds(&m, f);
            let mut rng = Rng(checked as u64 + 1);
            let invs: Vec<Inputs> = (0..4).map(|t| harness::inputs(&m, f, &kinds, &mut rng, t)).collect();
            let a = harness::run(&m, f, &kinds, &invs, None, Externals::Refuse);
            let b = harness::run(&c.module, c.function, &kinds, &invs, None, Externals::Refuse);
            assert!(matches!(harness::compare(&a, &b), Verdict::Refines), "{} {}", name, entry);
            checked += 1;
        }
    }
    assert!(checked > 0);
}
