//! The interpreter against real Metal on kernels whose results MSL defines bit for bit.
//!
//! Usage:
//!
//! - `metal_diff prepare JOBS.tsv DIR [THREADS]`: for each job (`AIR_PATH<TAB>ENTRY`, a kernel of the generated
//!   corpus whose MSL source sits next to the module as `.metal`), writes random input bytes and a zeroed output
//!   buffer to DIR, picks function constant values when the module has function constants, and writes
//!   DIR/manifest.tsv for `tests/canon/metal/run_kernels.swift`.
//! - `metal_diff compare DIR REPORT.tsv`: runs each job's canonical function, and its original, on the interpreter
//!   with only exact semantics (no uninterpreted calls) for the same threads and inputs, and compares the output
//!   buffer bit for bit with what Metal wrote. Bytes the interpreter leaves undef, poison or a choice are counted
//!   apart, never compared. Kernels in the exact subset (no floating-point operation besides loads and stores: MSL
//!   and LLVM define every integer result exactly) report `equal` or `differs`, and any difference fails; kernels
//!   that compute in floating point report `float-equal` or `float-differs`, which is information only: Metal on
//!   this GPU flushes subnormals and resolves the open bits of fast-math results its own way.
//!
//! The kernels follow the probe harness of `corpus/generate.py`: `in_bytes` at buffer 0, results at buffer 30 (64
//! bytes per thread), `thread_position_in_grid` as the thread index; other buffers (function-constant-gated
//! arguments) get the input bytes too. Threads run one after another on one memory on the interpreter, which is a
//! schedule Metal's own allows for these kernels (no thread reads what another writes).

#[path = "../tests/canon/harness.rs"]
mod harness;

use std::collections::BTreeMap;
use std::io::Write;

use airwgsl::api::FunctionConstant;
use airwgsl::interp::Externals;
use harness::{ArgKind, Inputs, Rng};

const IN_BYTES: usize = 1 << 14;

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|k| u8::from_str_radix(&s[k..k + 2], 16).unwrap()).collect()
}

fn prepare(jobs: &str, dir: &str, threads: u64) {
    std::fs::create_dir_all(dir).unwrap();
    let mut manifest = std::fs::File::create(format!("{}/manifest.tsv", dir)).unwrap();
    let mut n = 0;
    for (k, line) in std::fs::read_to_string(jobs).unwrap().lines().enumerate() {
        let mut p = line.split('\t');
        let (path, entry) = (p.next().unwrap(), p.next().unwrap());
        let source = path.trim_end_matches(".air").to_string() + ".metal";
        let lang = path.split('/').rev().nth(1).unwrap_or("3.1").to_string();
        let Ok(data) = std::fs::read(path) else { continue };
        let Ok(m) = airwgsl::parse_bitcode(&data) else { continue };
        let Some(f) = m.function_by_name(entry) else { continue };
        let mut rng = Rng(fnv(path) ^ fnv(entry));
        let constants = harness::constant_values(&m, &mut rng);
        let input: Vec<u8> = (0..IN_BYTES).map(|_| rng.next() as u8).collect();
        let in_path = format!("{}/{}.in", dir, k);
        std::fs::write(&in_path, &input).unwrap();
        let idx = harness::buffer_indices(&m, f);
        let mut buffers = Vec::new();
        let mut out = None;
        for b in idx.iter() {
            let Some(b) = b else { continue };
            if *b == 30 {
                let out_path = format!("{}/{}.out", dir, k);
                std::fs::write(&out_path, vec![0u8; (threads * 64) as usize]).unwrap();
                buffers.push(format!("{}={}", b, out_path));
                out = Some(format!("{}={}", b, out_path));
            } else {
                buffers.push(format!("{}={}", b, in_path));
            }
        }
        let Some(out) = out else { continue };
        let cs = if constants.is_empty() {
            "-".to_string()
        } else {
            constants
                .iter()
                .map(|c| format!("{}:{}:{}", c.index, c.data_type, hex(&c.value)))
                .collect::<Vec<_>>()
                .join(",")
        };
        writeln!(
            manifest,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            k,
            source,
            lang,
            entry,
            threads,
            buffers.join(","),
            out,
            cs,
            path
        )
        .unwrap();
        n += 1;
    }
    println!("jobs={}", n);
}

fn compare(dir: &str, report: &str) -> bool {
    let mut rep = std::fs::File::create(report).unwrap();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut indefinite_bytes = 0usize;
    for line in std::fs::read_to_string(format!("{}/manifest.tsv", dir)).unwrap().lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (id, entry, threads, out, cs, path) = (f[0], f[3], f[4].parse::<u64>().unwrap(), f[6], f[7], f[8]);
        let out_path = out.split('=').nth(1).unwrap();
        let (status, detail) = match std::fs::read(format!("{}.metal", out_path)) {
            Err(_) => ("metal-error".to_string(), "Metal produced no output (see the runner's log)".to_string()),
            Ok(metal) => {
                let constants: Vec<FunctionConstant> = if cs == "-" {
                    Vec::new()
                } else {
                    cs.split(',')
                        .map(|c| {
                            let p: Vec<&str> = c.split(':').collect();
                            FunctionConstant {
                                index: p[0].parse().unwrap(),
                                name: None,
                                data_type: p[1].parse().unwrap(),
                                value: unhex(p[2]),
                            }
                        })
                        .collect()
                };
                let r = one(path, entry, threads, f[5], &constants, &metal, &mut indefinite_bytes);
                (r.0, r.1)
            }
        };
        *counts.entry(status.clone()).or_default() += 1;
        writeln!(rep, "{}\t{}\t{}\t{}\t{}", id, path, entry, status, detail).unwrap();
    }
    let s: Vec<String> = counts.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
    println!("{} indefinite_bytes={}", s.join(" "), indefinite_bytes);
    !counts.contains_key("differs") && !counts.contains_key("refusal")
}

fn one(
    path: &str,
    entry: &str,
    threads: u64,
    buffers: &str,
    constants: &[FunctionConstant],
    metal: &[u8],
    indefinite: &mut usize,
) -> (String, String) {
    let m = airwgsl::parse_bitcode(&std::fs::read(path).unwrap()).unwrap();
    let canon = match airwgsl::passes::canonicalize(std::slice::from_ref(&m), entry, constants) {
        Ok(c) => c,
        Err(r) => return ("refusal".into(), r.to_string()),
    };
    let bound: BTreeMap<u64, Vec<u8>> = buffers
        .split(',')
        .map(|b| {
            let (i, p) = b.split_once('=').unwrap();
            (i.parse().unwrap(), std::fs::read(p).unwrap())
        })
        .collect();
    let mut results = Vec::new();
    for (module, func, consts) in
        [(&canon.module, canon.function, None), (&m, m.function_by_name(entry).unwrap(), Some(constants))]
    {
        let idx = harness::buffer_indices(module, func);
        let kinds = harness::arg_kinds(module, func);
        let invs: Vec<Inputs> = (0..threads)
            .map(|t| {
                let mut buffers = Vec::new();
                let mut values = Vec::new();
                for (a, k) in kinds.iter().enumerate() {
                    match k {
                        ArgKind::Buffer(_) => {
                            buffers.push(Some(
                                idx[a].and_then(|b| bound.get(&b).cloned()).unwrap_or_else(|| vec![0u8; IN_BYTES]),
                            ));
                            values.push(None);
                        }
                        ArgKind::Object => {
                            buffers.push(None);
                            values.push(None);
                        }
                        ArgKind::Value => {
                            buffers.push(None);
                            values.push(Some(airwgsl::interp::Val::S(airwgsl::interp::Scalar::int(t as u128))));
                        }
                    }
                }
                Inputs { buffers, values }
            })
            .collect();
        let run = harness::run(module, func, &kinds, &invs, consts, Externals::Refuse);
        if let Err(e) = &run.result {
            return (
                if matches!(e, airwgsl::interp::Error::Ub { .. }) { "interp-ub" } else { "interp-unsupported" }.into(),
                e.to_string(),
            );
        }
        let out_arg = idx.iter().position(|b| *b == Some(30)).unwrap();
        let label = format!("buffer argument {}", out_arg);
        let region = run.memory.regions().find(|(_, r)| r.label == label).unwrap().1.bytes.clone();
        results.push(region);
    }
    let exact = exact_subset(&canon.module, canon.function);
    let tag = |s: &str| if exact { s.to_string() } else { format!("float-{}", s) };
    let (canonical, original) = (&results[0], &results[1]);
    for (k, (c, o)) in canonical.iter().zip(original.iter()).enumerate() {
        if c != o && o.is_plain() {
            return ("differs".into(), format!("canonical and original interpreter runs differ at byte {}", k));
        }
    }
    let mut first = None;
    for (k, b) in canonical.iter().enumerate() {
        if !b.is_plain() {
            *indefinite += 1;
            continue;
        }
        if metal.get(k) != Some(&b.v) && first.is_none() {
            first = Some(k);
        }
    }
    match first {
        None => (tag("equal"), String::new()),
        Some(k) => (
            tag("differs"),
            format!(
                "byte {} (thread {}): Metal {:02x}, interpreter {:02x}",
                k,
                k / 64,
                metal.get(k).copied().unwrap_or(0),
                canonical[k].v
            ),
        ),
    }
}

/// Whether function `f` computes no floating-point value: no instruction other than a load or store has a
/// floating-point operand or result.
fn exact_subset(m: &airwgsl::ir::Module, f: usize) -> bool {
    let func = &m.functions[f];
    let fp = |t: airwgsl::ir::TypeId| m.types.is_fp_or_fp_vector(t);
    let ty = |v: airwgsl::ir::Value| -> Option<airwgsl::ir::TypeId> {
        match v {
            airwgsl::ir::Value::Inst(i) => Some(func.inst(i).ty),
            airwgsl::ir::Value::Arg(a) => Some(func.args[a as usize].ty),
            airwgsl::ir::Value::Const(c) => Some(m.constant(c).ty),
            _ => None,
        }
    };
    func.blocks.iter().flat_map(|b| b.insts.iter()).all(|&i| {
        let inst = func.inst(i);
        match &inst.op {
            airwgsl::ir::Op::Load { .. } | airwgsl::ir::Op::Store { .. } => true,
            op => !fp(inst.ty) && op.operands().iter().all(|&v| ty(v).is_none_or(|t| !fp(t))),
        }
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("prepare") if args.len() >= 4 => {
            prepare(&args[2], &args[3], args.get(4).and_then(|t| t.parse().ok()).unwrap_or(16))
        }
        Some("compare") if args.len() >= 4 => {
            if !compare(&args[2], &args[3]) {
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("usage: metal_diff prepare JOBS.tsv DIR [THREADS] | metal_diff compare DIR REPORT.tsv");
            std::process::exit(2);
        }
    }
}
