//! Canonicalizes every entry point of a corpus and checks, on the interpreter, that the canonical function refines
//! the original: both run on the same generated inputs (and function constant values), and the canonical run's
//! return value, events and memory must refine the original's (`interp::compare`).
//!
//! Usage: `canon_check JOBS.tsv REPORT.tsv [THREADS] [SEEDS]`
//!
//! Each line of JOBS.tsv is `BITCODE_PATH<TAB>ENTRY` (raw or wrapped bitcode). For each job and seed the original
//! runs with uninterpreted calls for declarations without evaluated definitions: its constructors, then four
//! invocations of the entry point on one memory. REPORT.tsv gets one line per job: path, entry, status, detail.
//! Seeds cycle through three ways of filling buffers (floats, small integers, random bytes; see the harness).
//! Statuses: `ok` (refines for every seed, and at least one seed's original run was defined), `ok-original-ub` (every
//! seed's original run had undefined behavior, which any result refines: the job shows nothing; a seed whose original
//! run reaches the step limit, `AIRWGSL_STEP_LIMIT` instructions per invocation, shows nothing either), `refusal` (canonicalization refused, with the construct), `unchecked`
//! (the interpreter cannot run the original: the reason), `mismatch`, `canonical-error` and `panic` (failures).
//! The totals go to stdout; the exit status is non-zero if any job failed.

#[path = "../tests/canon/harness.rs"]
mod harness;

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use airwgsl::interp::Externals;
use harness::{Rng, Verdict};

thread_local! {
    static LAST: std::cell::RefCell<Option<(String, std::rc::Rc<airwgsl::ir::Module>)>> = const { std::cell::RefCell::new(None) };
}

/// The module at `path`, parsed once for consecutive jobs on it.
fn module(path: &str) -> Result<std::rc::Rc<airwgsl::ir::Module>, String> {
    if let Some(m) = LAST.with(|l| l.borrow().as_ref().filter(|(p, _)| p == path).map(|(_, m)| m.clone())) {
        return Ok(m);
    }
    let data = std::fs::read(path).map_err(|e| format!("read: {}", e))?;
    let m = std::rc::Rc::new(airwgsl::parse_bitcode(&data).map_err(|e| format!("parse: {}", e))?);
    LAST.with(|l| *l.borrow_mut() = Some((path.to_string(), m.clone())));
    Ok(m)
}

fn check(path: &str, entry: &str, seeds: u64) -> (String, String) {
    let m = match module(path) {
        Ok(m) => m,
        Err(e) => return ("unchecked".into(), e),
    };
    let m: &airwgsl::ir::Module = &m;
    let Some(f) = m.function_by_name(entry) else { return ("unchecked".into(), "no such function".into()) };
    let kinds = harness::arg_kinds(m, f);
    let mut ub = None;
    let mut limit = None;
    let mut defined = 0;
    for seed in 0..seeds {
        let fill = [harness::Fill::Floats, harness::Fill::SmallInts, harness::Fill::Random][(seed % 3) as usize];
        let mut rng = Rng(fnv(path) ^ fnv(entry) ^ seed.wrapping_mul(0x9e37_79b9));
        let has_fc = !airwgsl::passes::specialize::records(m).is_empty();
        let constants = if has_fc { harness::constant_values(m, &mut rng) } else { Vec::new() };
        let canon = match airwgsl::passes::canonicalize(std::slice::from_ref(m), entry, &constants) {
            Ok(c) => c,
            Err(r) => return ("refusal".into(), format!("{}", r)),
        };
        let invs: Vec<harness::Inputs> = (0..4)
            .map(|t| {
                let mut r2 = Rng(rng.0 ^ 0x5555);
                let mut i = harness::inputs_with(m, f, &kinds, &mut r2, 0, fill);
                let mut r3 = Rng(rng.0 ^ t);
                let vi = harness::inputs_with(m, f, &kinds, &mut r3, t * 4, fill);
                i.values = vi.values;
                i
            })
            .collect();
        let a = harness::run(m, f, &kinds, &invs, Some(&constants), Externals::Uninterpreted);
        let ckinds = harness::arg_kinds(&canon.module, canon.function);
        if ckinds != kinds {
            return ("canonical-error".into(), "the canonical function's arguments bind differently".into());
        }
        let b = harness::run(&canon.module, canon.function, &kinds, &invs, None, Externals::Uninterpreted);
        match harness::compare(&a, &b) {
            Verdict::Refines => defined += 1,
            Verdict::FirstUb(r) => ub = Some(r),
            Verdict::FirstLimit(r) => limit = Some(r),
            Verdict::FirstUnsupported(r) => return ("unchecked".into(), r),
            Verdict::Differs(d) => {
                let st = if b.result.is_err() { "canonical-error" } else { "mismatch" };
                return (st.into(), format!("seed {}: {}", seed, d));
            }
        }
    }
    // A job is checked when some seed's original run was defined; a job whose every run had undefined behavior
    // shows nothing.
    match (defined, ub, limit) {
        (0, Some(r), _) => ("ok-original-ub".into(), r),
        (0, None, Some(l)) => ("unchecked".into(), format!("limit: {}", l)),
        _ => ("ok".into(), String::new()),
    }
}

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: canon_check JOBS.tsv REPORT.tsv [THREADS] [SEEDS]");
        std::process::exit(2);
    }
    let jobs: Vec<(String, String)> = std::fs::read_to_string(&args[1])
        .expect("jobs")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut p = l.split('\t');
            (p.next().unwrap().to_string(), p.next().unwrap_or("").to_string())
        })
        .collect();
    let threads: usize = args.get(3).and_then(|t| t.parse().ok()).unwrap_or(4);
    let seeds: u64 = args.get(4).and_then(|t| t.parse().ok()).unwrap_or(2);
    let jobs = Arc::new(jobs);
    let next = Arc::new(AtomicUsize::new(0));
    let results = Arc::new(Mutex::new(vec![(String::new(), String::new()); jobs.len()]));
    let mut handles = Vec::new();
    for _ in 0..threads {
        let jobs = jobs.clone();
        let next = next.clone();
        let results = results.clone();
        handles.push(
            std::thread::Builder::new()
                .stack_size(64 << 20)
                .spawn(move || {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs.len() {
                            break;
                        }
                        let (p, e) = &jobs[i];
                        let t0 = std::time::Instant::now();
                        if std::env::var_os("CANON_CHECK_TRACE").is_some() {
                            eprintln!("start {} {}", p, e);
                        }
                        let r = std::panic::catch_unwind(|| check(p, e, seeds)).unwrap_or_else(|x| {
                            let msg = x
                                .downcast_ref::<String>()
                                .cloned()
                                .or_else(|| x.downcast_ref::<&str>().map(|s| s.to_string()));
                            ("panic".into(), msg.unwrap_or_default())
                        });
                        let ms = t0.elapsed().as_millis();
                        results.lock().unwrap()[i] =
                            (r.0, if ms > 10_000 { format!("{} [{} ms]", r.1, ms) } else { r.1 });
                    }
                })
                .unwrap(),
        );
    }
    for h in handles {
        h.join().unwrap();
    }
    let results = results.lock().unwrap();
    let mut report = std::fs::File::create(&args[2]).expect("report");
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (i, (status, detail)) in results.iter().enumerate() {
        *counts.entry(status.clone()).or_default() += 1;
        writeln!(report, "{}\t{}\t{}\t{}", jobs[i].0, jobs[i].1, status, detail.replace(['\t', '\n'], " ")).unwrap();
    }
    let summary: Vec<String> = counts.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
    println!("jobs={} {}", jobs.len(), summary.join(" "));
    let failed = ["mismatch", "canonical-error", "panic"].iter().any(|s| counts.contains_key(*s));
    if failed {
        std::process::exit(1);
    }
}
