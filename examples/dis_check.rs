//! Compares this crate's printed IR with `llvm-dis` output for a corpus of bitcode modules.
//!
//! Usage: `dis_check JOBS.tsv REPORT.tsv [THREADS]`
//!
//! Each line of JOBS.tsv is `BITCODE_PATH<TAB>LL_PATH`, the `.ll` file being `llvm-dis` (LLVM 21.1.8) output for the
//! bitcode. The comparison is textual, line for line, under one normalization: the `; ModuleID = '...'` line that
//! `llvm-dis` writes first is dropped, because it names the input file rather than anything in the module. Comments
//! (`; Function Attrs:`, `; preds = ...`, `; Unknown intrinsic`) are compared like everything else.
//!
//! REPORT.tsv gets one line per module: path, status (`ok`, `mismatch`, `error`) and detail. The totals go to stdout;
//! the exit status is non-zero unless every module is `ok`.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn check(bc: &str, ll: &str) -> (String, String) {
    let data = match std::fs::read(bc) {
        Ok(d) => d,
        Err(e) => return ("error".into(), format!("read {}: {}", bc, e)),
    };
    let expected = match std::fs::read(ll) {
        Ok(d) => String::from_utf8_lossy(&d).into_owned(),
        Err(e) => return ("error".into(), format!("read {}: {}", ll, e)),
    };
    let module = match airwgsl::parse_bitcode(&data) {
        Ok(m) => m,
        Err(e) => return ("error".into(), e.to_string()),
    };
    let got = airwgsl::ir::print::print_module(&module);
    let exp_lines: Vec<&str> = expected.lines().filter(|l| !l.starts_with("; ModuleID = ")).collect();
    let got_lines: Vec<&str> = got.lines().collect();
    for i in 0..exp_lines.len().max(got_lines.len()) {
        let e = exp_lines.get(i).copied().unwrap_or("<end>");
        let g = got_lines.get(i).copied().unwrap_or("<end>");
        if e != g {
            let clip = |s: &str| s.chars().take(300).collect::<String>().replace('\t', " ");
            return ("mismatch".into(), format!("line {}: expected [{}] got [{}]", i + 1, clip(e), clip(g)));
        }
    }
    ("ok".into(), String::new())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dis_check JOBS.tsv REPORT.tsv [THREADS]");
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
    let jobs = Arc::new(jobs);
    let next = Arc::new(AtomicUsize::new(0));
    let results = Arc::new(Mutex::new(vec![(String::new(), String::new()); jobs.len()]));
    let mut handles = Vec::new();
    for _ in 0..threads {
        let jobs = jobs.clone();
        let next = next.clone();
        let results = results.clone();
        handles.push(std::thread::spawn(move || {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                let (bc, ll) = &jobs[i];
                let r = std::panic::catch_unwind(|| check(bc, ll)).unwrap_or_else(|_| ("error".into(), "panic".into()));
                results.lock().unwrap()[i] = r;
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let results = results.lock().unwrap();
    let mut report = std::fs::File::create(&args[2]).expect("report");
    let (mut ok, mut mismatch, mut error) = (0, 0, 0);
    for (i, (status, detail)) in results.iter().enumerate() {
        match status.as_str() {
            "ok" => ok += 1,
            "mismatch" => mismatch += 1,
            _ => error += 1,
        }
        writeln!(report, "{}\t{}\t{}", jobs[i].0, status, detail).unwrap();
    }
    println!("modules={} ok={} mismatch={} error={}", jobs.len(), ok, mismatch, error);
    if ok != jobs.len() {
        std::process::exit(1);
    }
}
