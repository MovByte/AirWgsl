//! Decodes the AIR metadata of a corpus of modules and tallies the results.
//!
//! Usage: `air_check LIST REPORT.tsv [THREADS] [mutate]`
//!
//! LIST holds one bitcode path per line (raw or wrapped bitcode: `.bc` or `.air`). Every module must either decode
//! into an [`airwgsl::air::AirModule`] or be refused with a named construct. REPORT.tsv gets one line per module:
//! path, `ok` or `refused`, the refused construct and the reason; a crash of the decoder is reported as `panic`. The
//! totals, entry point counts by stage and refusals by construct go to stdout. The exit status is non-zero when any
//! module panics.
//!
//! With `mutate`, every module that decodes is then mutated once per distinct `air.*` string of its metadata: the
//! string is renamed to `air.mutated_key`, and the decoder must refuse the module naming that string. A mutation
//! that still decodes, or is refused for another reason, is a hole in the closed grammar and is reported as
//! `hole` with the original string.

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use airwgsl::air::AirModule;

enum Outcome {
    Ok { stages: Vec<&'static str>, mutations: usize, holes: Vec<String> },
    Refused { construct: String, reason: String },
    Error(String),
}

/// Renames each distinct `air.*` string of the module's metadata in turn; returns the number of mutations and the
/// strings whose renaming the decoder did not refuse by name.
fn mutate(ir: &airwgsl::ir::Module) -> (usize, Vec<String>) {
    use airwgsl::ir::{MdId, Metadata};
    const MUTATED: &[u8] = b"air.mutated_key";
    let mut n = 0;
    let mut holes = Vec::new();
    for (i, md) in ir.metadata.iter().enumerate() {
        let Metadata::String(s) = md else { continue };
        if !s.starts_with(b"air.") {
            continue;
        }
        let mut x = ir.clone();
        x.set_md(MdId(i as u32), Metadata::String(MUTATED.to_vec()));
        x.rebuild_md_unique();
        n += 1;
        match AirModule::decode(x) {
            Err(r) if r.construct == "air.mutated_key" || r.reason.contains("air.mutated_key") => {}
            other => holes.push(format!(
                "{}: {}",
                String::from_utf8_lossy(s),
                match other {
                    Ok(_) => "decoded".to_string(),
                    Err(r) => r.to_string(),
                }
            )),
        }
    }
    (n, holes)
}

fn check(path: &str, mutation: bool) -> Outcome {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => return Outcome::Error(format!("read: {}", e)),
    };
    match AirModule::parse(&data) {
        Ok(m) => Outcome::Ok {
            mutations: 0,
            holes: Vec::new(),
            stages: m
                .entry_points
                .iter()
                .map(|e| match e.stage {
                    airwgsl::air::Stage::Vertex => "vertex",
                    airwgsl::air::Stage::Fragment => "fragment",
                    airwgsl::air::Stage::Kernel => "kernel",
                    airwgsl::air::Stage::Visible => "visible",
                    airwgsl::air::Stage::Intersection => "intersection",
                    airwgsl::air::Stage::Mesh => "mesh",
                    airwgsl::air::Stage::Object => "object",
                })
                .collect(),
        }
        .with_mutations(if mutation { Some(mutate(&m.ir)) } else { None }),
        Err(r) => Outcome::Refused { construct: r.construct.clone(), reason: r.to_string() },
    }
}

impl Outcome {
    fn with_mutations(self, m: Option<(usize, Vec<String>)>) -> Outcome {
        match (self, m) {
            (Outcome::Ok { stages, .. }, Some((mutations, holes))) => Outcome::Ok { stages, mutations, holes },
            (o, _) => o,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: air_check LIST REPORT.tsv [THREADS]");
        std::process::exit(2);
    }
    let jobs: Vec<String> = std::fs::read_to_string(&args[1])
        .expect("list")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    let threads: usize = args.get(3).and_then(|t| t.parse().ok()).unwrap_or(4);
    let mutation = args.get(4).map(String::as_str) == Some("mutate");
    let jobs = Arc::new(jobs);
    let next = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<Option<Outcome>>>> = Arc::new(Mutex::new((0..jobs.len()).map(|_| None).collect()));
    let mut handles = Vec::new();
    for _ in 0..threads {
        let (jobs, next, results) = (jobs.clone(), next.clone(), results.clone());
        handles.push(std::thread::spawn(move || {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                let r = std::panic::catch_unwind(|| check(&jobs[i], mutation)).unwrap_or_else(|p| {
                    let msg = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    Outcome::Error(format!("panic: {}", msg))
                });
                results.lock().unwrap()[i] = Some(r);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let results = results.lock().unwrap();
    let mut report = std::fs::File::create(&args[2]).expect("report");
    let (mut ok, mut refused, mut errors) = (0, 0, 0);
    let (mut mutations, mut holes) = (0usize, 0usize);
    let mut stages: BTreeMap<&str, usize> = BTreeMap::new();
    let mut constructs: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for (i, r) in results.iter().enumerate() {
        match r.as_ref().unwrap() {
            Outcome::Ok { stages: s, mutations: n, holes: h } => {
                ok += 1;
                for st in s {
                    *stages.entry(st).or_default() += 1;
                }
                mutations += n;
                holes += h.len();
                writeln!(report, "{}\tok\t\t", jobs[i]).unwrap();
                for hole in h {
                    writeln!(report, "{}\thole\t\t{}", jobs[i], hole.replace('\t', " ")).unwrap();
                }
            }
            Outcome::Refused { construct, reason } => {
                refused += 1;
                let e = constructs.entry(construct.clone()).or_insert((0, reason.clone()));
                e.0 += 1;
                writeln!(report, "{}\trefused\t{}\t{}", jobs[i], construct, reason.replace('\t', " ")).unwrap();
            }
            Outcome::Error(e) => {
                errors += 1;
                writeln!(report, "{}\terror\t\t{}", jobs[i], e.replace('\t', " ")).unwrap();
            }
        }
    }
    println!("modules={} ok={} refused={} error={}", jobs.len(), ok, refused, errors);
    if mutation {
        println!("mutations={} refused-by-name={} holes={}", mutations, mutations - holes, holes);
    }
    for (s, n) in &stages {
        println!("entry points: {} {}", s, n);
    }
    for (c, (n, example)) in &constructs {
        println!("refused {} x{}: {}", c, n, example.chars().take(400).collect::<String>());
    }
    if errors != 0 || holes != 0 {
        std::process::exit(1);
    }
}
