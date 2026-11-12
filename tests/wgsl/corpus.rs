//! The generated corpus's kernels in the exact subset, run on Metal, wgpu and Dawn with random inputs.
//!
//! `AW_CORPUS` names the build directory of `corpus/build.py` (holding `modules.tsv` and `air/`). Every kernel of the
//! language-version 3.1 modules that this crate compiles and whose function uses only operations MSL defines exactly
//! (integer arithmetic, integer and integer-to-float conversions, relational functions, bit moves through memory,
//! control flow; no floating-point arithmetic, no floating-point math or derivatives, no float-to-integer or
//! float-to-float conversions, whose out-of-range or rounding cases MSL leaves open, no division or shift by a
//! variable amount, which random inputs make undefined, no poison vector lane, which as_type of a 3-component vector
//! exposes, no pointer-integer conversion, whose address values are the implementation's, and no threadgroup
//! memory, which the probes race on) is run: every buffer it reads gets
//! random bytes, every buffer it writes starts as a fill pattern, one threadgroup of 64 threads, and all buffers must
//! be byte-identical. Without `AW_CORPUS` the test reports that it did not run.

use super::harness::{bytes, compile_kernel, dawn_node, first_difference, run_dawn, run_metal, run_wgpu};
use airwgsl::air::{Access, AirModule, Role, Slot, Stage};
use airwgsl::ir::{BinOp, CastOp, GlobalId, Module, Op, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn is_const(v: Value) -> bool {
    matches!(v, Value::Const(_))
}

/// Whether function `fi` stays in the exact subset.
fn exact(m: &Module, fi: usize) -> bool {
    let f = &m.functions[fi];
    // Threadgroup memory is shared by the threadgroup's threads; the probes use it without barriers, so what a thread
    // reads there depends on the order other threads ran in.
    if f.args.iter().any(|a| m.types.pointer_addrspace(a.ty) == Some(3)) {
        return false;
    }
    for inst in &f.insts {
        for v in inst.op.operands() {
            if let Value::Global(GlobalId::Var(g)) = v
                && m.globals[g as usize].common.addrspace == 3
            {
                return false;
            }
        }
        match &inst.op {
            Op::Binary { op, .. } if op.is_float() => return false,
            Op::FNeg { .. } => return false,
            // An address as an integer: its value beyond the alignment the region gives is the implementation's.
            Op::Cast { op: CastOp::PtrToInt | CastOp::IntToPtr, .. } => return false,
            // A poison lane (as_type of a 3-component vector reads its fourth, undefined lane).
            Op::ShuffleVector { mask, .. } if mask.contains(&-1) => return false,
            Op::Cast { op: CastOp::FPToSI | CastOp::FPToUI | CastOp::FPTrunc | CastOp::FPExt, .. } => return false,
            Op::Binary { op: BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem, rhs, .. } if !is_const(*rhs) => {
                // Division by zero is undefined: random divisors include zero.
                return false;
            }
            Op::Binary { op: BinOp::Shl | BinOp::LShr | BinOp::AShr, rhs, .. } if !is_const(*rhs) => {
                // A shift by the width or more is poison: random amounts include such.
                return false;
            }
            Op::Call { callee: Value::Global(GlobalId::Function(c)), .. } => {
                let name = &m.functions[*c as usize].common.name;
                let fam = airwgsl::wgsl::tracked::family_of(name);
                let ok = match fam {
                    Some("integer") | Some("relational") => true,
                    Some("conversion") => {
                        // air.convert.<dk>.<dty>.<sk>.<sty>: integer sources only, and no float-to-float.
                        let p: Vec<&str> = name.split('.').collect();
                        p.len() == 6 && p[4] != "f"
                    }
                    _ => {
                        name.starts_with("llvm.lifetime")
                            || name.starts_with("llvm.memcpy")
                            || name.starts_with("llvm.memset")
                            || name.starts_with("llvm.experimental.noalias")
                    }
                };
                // Integer functions whose results MSL leaves open for some inputs.
                // (mul24 and mad24 beyond 24 bits; extract_bits and insert_bits past the width.)
                if !ok || ["mad24", "mul24", "extract_bits", "insert_bits"].iter().any(|n| name.contains(n)) {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

#[test]
fn generated_corpus_exact_kernels_match_metal() {
    let Some(root) = std::env::var_os("AW_CORPUS").map(PathBuf::from) else {
        eprintln!("generated corpus differential not run: AW_CORPUS unset");
        return;
    };
    let limit: usize = std::env::var("AW_CORPUS_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    // Run every STRIDE-th exact kernel (all by default); the others are counted.
    let stride: usize = std::env::var("AW_CORPUS_STRIDE").ok().and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
    let mut seen_exact = 0usize;
    let mut sampled_out = 0usize;
    let tsv = std::fs::read_to_string(root.join("modules.tsv")).unwrap();
    let mut modules: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for line in tsv.lines().filter(|l| !l.starts_with('#')) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() >= 3 && c[1] == "3.1" {
            modules.entry(c[0].to_string()).or_default().push(c[2].rsplit('/').next().unwrap().to_string());
        }
    }
    let dawn = dawn_node();
    let (mut ran, mut skipped_inexact, mut not_compiled) = (0usize, 0usize, 0usize);
    let tmp = super::harness::tmp_dir().join("corpus");
    let _ = std::fs::create_dir_all(&tmp);
    'outer: for (path, functions) in modules {
        let air = std::fs::read(root.join(&path)).unwrap();
        let source = std::fs::read_to_string(root.join(path.replace(".air", ".metal"))).unwrap();
        let decoded = AirModule::parse(&air).unwrap();
        let m = &decoded.ir;
        for f in functions {
            if ran >= limit {
                break 'outer;
            }
            let Some(e) = decoded.entry_point(&f) else { continue };
            if e.stage != Stage::Kernel {
                continue;
            }
            if !exact(m, e.function as usize) {
                skipped_inexact += 1;
                continue;
            }
            // Buffers by Metal index: written ones start as a pattern, read ones as random bytes; each threadgroup
            // buffer index gets 1024 bytes.
            let mut buffers = vec![vec![0u8; 4]; 31];
            let mut tg: Vec<u64> = Vec::new();
            for a in &e.arguments {
                let Role::Buffer(b) = &a.role else { continue };
                let Slot::Value(idx) = b.location.index else { continue };
                let idx = idx as usize;
                if b.address_space == Some(3) {
                    if tg.len() <= idx {
                        tg.resize(idx + 1, 0);
                    }
                    tg[idx] = 1024;
                } else {
                    buffers[idx] = if b.access == Access::Read {
                        bytes(idx as u64 * 31 + a.position as u64, 65536)
                    } else {
                        vec![0xa5; 65536]
                    };
                }
            }
            let Ok(c) = compile_kernel(&air, &f, [64, 1, 1], &tg) else {
                not_compiled += 1;
                continue;
            };
            seen_exact += 1;
            if !(seen_exact - 1).is_multiple_of(stride) {
                sampled_out += 1;
                continue;
            }
            let dir = tmp.join(format!("{}-{}", path.replace('/', "_"), f));
            let _ = std::fs::create_dir_all(&dir);
            let reference = run_metal(&dir, &source, &f, [64, 1, 1], [1, 1, 1], &tg, &buffers);
            let w = run_wgpu(&c, [1, 1, 1], &buffers);
            for (k, (a, b)) in reference.iter().zip(&w).enumerate() {
                if let Some(i) = first_difference(a, b) {
                    panic!("{} {}: buffer {} differs at byte {} on wgpu (WGSL in {})", path, f, k, i, dir.display());
                }
            }
            if let Some(dn) = &dawn {
                std::fs::write(dir.join("k.wgsl"), &c.wgsl[0].source).unwrap();
                let d = run_dawn(dn, &dir, &c, [1, 1, 1], &buffers);
                for (k, (a, b)) in reference.iter().zip(&d).enumerate() {
                    if let Some(i) = first_difference(a, b) {
                        panic!("{} {}: buffer {} differs at byte {} on Dawn", path, f, k, i);
                    }
                }
            }
            ran += 1;
        }
    }
    eprintln!(
        "generated corpus: {} exact kernels equal on Metal and wgpu{}; {} more exact kernels compiled but not run \
         (stride {}); {} kernels outside the exact subset; {} exact-subset kernels refused",
        ran,
        if dawn.is_some() { " and Dawn" } else { " (Dawn not run)" },
        sampled_out,
        stride,
        skipped_inexact,
        not_compiled
    );
}
