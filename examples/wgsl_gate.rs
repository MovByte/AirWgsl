//! The phase-1 gate's compiler check over a corpus: every function either compiles to WGSL that naga accepts (and,
//! in a second step, Tint), or is refused with an item of the tracked refusal list. A panic is a crash and fails the
//! gate.
//!
//! Usage: `wgsl_gate CORPUS_ROOT OUT_DIR [THREADS] [LIMIT]`
//!
//! `CORPUS_ROOT` is the output directory of `corpus/pipeline.py`: the generated corpus is read from
//! `build/modules.tsv` (modules under `build/`), the system corpus from `system/modules.tsv` (modules in
//! `system/bitcode/<sha256>.bc`). Each function is lowered on its own ([`airwgsl::wgsl::compile_entry`]): a kernel
//! with a 64 x 1 x 1 threadgroup and 4096 bytes for each threadgroup buffer index, a vertex or fragment function with
//! its own varyings.
//!
//! With `AW_TINT` naming Dawn's `tint` executable, each distinct generated module is also checked by Tint
//! (`tint --format wgsl`: parsing, resolution and the uniformity analysis).
//!
//! `OUT_DIR` gets `results.tsv` (corpus, module, function, status, tracked item, kind, construct, reason),
//! `wgsl/<key>.wgsl` for each distinct generated module (keyed by content) and `wgsl.lst` listing them, and `summary.txt` (counts by
//! status and by tracked item). The exit status is non-zero when any function crashed or produced WGSL naga rejects.

use airwgsl::HostCaps;
use airwgsl::wgsl::{self, hooks, tracked};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct Job {
    corpus: &'static str,
    sha: String,
    path: PathBuf,
    offset: u64,
    size: u64,
    functions: Vec<String>,
}

/// A 128-bit content key for the module text (two SipHash passes with different keys).
fn content_key(data: &[u8]) -> String {
    use std::hash::{Hash, Hasher};
    let mut a = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut a);
    let mut b = std::collections::hash_map::DefaultHasher::new();
    0x5eedu64.hash(&mut b);
    data.hash(&mut b);
    format!("{:016x}{:016x}", a.finish(), b.finish())
}

fn read_jobs(root: &Path) -> Vec<Job> {
    let mut by_module: BTreeMap<(String, &'static str), Job> = BTreeMap::new();
    for (corpus, tsv) in [("generated", "build/modules.tsv"), ("system", "system/modules.tsv")] {
        let Ok(text) = std::fs::read_to_string(root.join(tsv)) else {
            eprintln!("no {}", tsv);
            continue;
        };
        for line in text.lines() {
            if line.starts_with('#') {
                continue;
            }
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() < 7 {
                continue;
            }
            let (entry, sha) = (c[2], c[4].to_string());
            let function =
                if corpus == "generated" { entry.rsplit('/').next().unwrap().to_string() } else { entry.to_string() };
            let (path, offset, size) = if corpus == "generated" {
                (root.join("build").join(c[0]), c[5].parse().unwrap_or(0), c[6].parse().unwrap_or(0))
            } else {
                (root.join("system/bitcode").join(format!("{}.bc", sha)), 0, 0)
            };
            let j = by_module.entry((sha.clone(), corpus)).or_insert(Job {
                corpus,
                sha,
                path,
                offset,
                size,
                functions: Vec::new(),
            });
            if !j.functions.contains(&function) {
                j.functions.push(function);
            }
        }
    }
    by_module.into_values().collect()
}

struct Outcome {
    status: &'static str,
    item: String,
    kind: String,
    construct: String,
    reason: String,
    wgsl: Option<String>,
}

fn refused(r: &airwgsl::Refusal) -> Outcome {
    let item = match tracked::item_of(r) {
        Some((ns, id)) => format!("{}:{}", ns.prefix(), id),
        None => "UNTRACKED".into(),
    };
    Outcome {
        status: "refused",
        item,
        kind: format!("{:?}", r.kind),
        construct: r.construct.clone(),
        reason: r.reason.replace(['\t', '\n'], " "),
        wgsl: None,
    }
}

fn naga_check(src: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    // WGSL's core built-ins (unpack2x16float and pack2x16float among them) are always available in WebGPU; naga
    // gates them behind a capability.
    let mut caps = naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32;
    if src.starts_with("enable f16;") || src.contains("\nenable f16;") {
        caps |= naga::valid::Capabilities::SHADER_FLOAT16;
    }
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
        .validate(&module)
        .map_err(|e| e.emit_to_string(src))?;
    Ok(())
}

/// Tint's WGSL front end and resolver (Dawn's validation of a shader module) on a file.
fn tint_check(tint: &Path, file: &Path) -> Result<(), String> {
    let out = std::process::Command::new(tint)
        .args(["--format", "wgsl", "-o", "/dev/null"])
        .arg(file)
        .output()
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).lines().take(6).collect::<Vec<_>>().join(" | ").replace('\t', " "))
    }
}

fn lower(module: &airwgsl::ir::Module, bitcode: &[u8], function: &str, rules: &hooks::Rules) -> Outcome {
    let entries = match wgsl::entry::entries(module) {
        Ok(e) => e,
        Err(r) => return refused(&r),
    };
    let fi = match module.function_by_name(function) {
        Some(fi) => fi,
        // The system index also lists the data symbols of library modules: they are not functions.
        None if module.globals.iter().any(|g| g.common.name == function) => {
            return Outcome {
                status: "not-a-function",
                item: String::new(),
                kind: String::new(),
                construct: String::new(),
                reason: "a global variable of the module".into(),
                wgsl: None,
            };
        }
        // Library modules (archive members) are listed under their member names: they hold no entry point.
        None if entries.is_empty() => {
            return Outcome {
                status: "not-a-function",
                item: String::new(),
                kind: String::new(),
                construct: String::new(),
                reason: "a library module listed under its archive member name".into(),
                wgsl: None,
            };
        }
        // Binary archives name a module's one entry point differently from its IR function.
        None if entries.len() == 1 => entries[0].function,
        None => return refused(&tracked::refuse("module-unreadable", function, None, "no such function")),
    };
    let function = module.functions[fi].common.name.as_str();
    let Some(entry) = entries.into_iter().find(|e| e.function == fi) else {
        // Not a pipeline stage: a function linked into stages (visible, intersection, stitchable or library code).
        let has = |n: &str| {
            module.named_metadata.iter().any(|nm| {
                nm.name == n
                    && nm.ops.iter().any(|&o| match module.md(o) {
                        airwgsl::ir::Metadata::Tuple { ops, .. } => ops.first().copied().flatten().is_some_and(|f| {
                            matches!(module.md(f), airwgsl::ir::Metadata::Value { value: airwgsl::ir::Value::Global(airwgsl::ir::GlobalId::Function(x)), .. } if *x as usize == fi)
                        }),
                        _ => false,
                    })
            })
        };
        let r = if has("air.intersection") {
            tracked::refuse_family("ray-tracing", function, None)
        } else if has("air.visible") {
            tracked::refuse_family("visible-stitchable", function, None)
        } else {
            tracked::refuse(
                "non-canonical-input",
                function,
                None,
                "a function that is not an entry point is linked into one",
            )
        };
        return refused(&r);
    };
    let ft = match entry.stage {
        airwgsl::MetalStage::Vertex => 1,
        airwgsl::MetalStage::Fragment => 2,
        _ => 3,
    };
    let am = airwgsl::AirModule {
        hash: [0; 32],
        bitcode: Arc::from(bitcode),
        entry: Some(airwgsl::EntryPoint { function: function.to_string(), function_type: ft, constants: Vec::new() }),
    };
    let inp = match wgsl::input(&am) {
        Ok(i) => i,
        Err(r) => return refused(&r),
    };
    let mut host = HostCaps::default();
    host.features.insert("shader-f16".into());
    match wgsl::compile_entry(&inp, &host, rules, [64, 1, 1], &[4096; 31]) {
        Ok((src, _)) => match naga_check(&src) {
            Ok(()) => Outcome {
                status: "compiled",
                item: String::new(),
                kind: String::new(),
                construct: String::new(),
                reason: String::new(),
                wgsl: Some(src),
            },
            Err(e) => Outcome {
                status: "naga-rejected",
                item: String::new(),
                kind: String::new(),
                construct: String::new(),
                reason: e.lines().take(6).collect::<Vec<_>>().join(" | ").replace('\t', " "),
                wgsl: Some(src),
            },
        },
        Err(r) => refused(&r),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: wgsl_gate CORPUS_ROOT OUT_DIR [THREADS] [LIMIT]");
        std::process::exit(2);
    }
    let root = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let limit: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    std::fs::create_dir_all(out.join("wgsl")).unwrap();
    let mut jobs = read_jobs(&root);
    jobs.truncate(limit);
    eprintln!("{} modules, {} functions", jobs.len(), jobs.iter().map(|j| j.functions.len()).sum::<usize>());
    let jobs = Arc::new(jobs);
    let next = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    // Content key of each generated module and, once checked, Tint's verdict (an error message or none).
    let written: Arc<Mutex<HashMap<String, Option<Option<String>>>>> = Arc::new(Mutex::new(HashMap::new()));
    let tint: Option<PathBuf> = std::env::var_os("AW_TINT").map(PathBuf::from);
    eprintln!(
        "tint: {}",
        tint.as_ref().map(|t| t.display().to_string()).unwrap_or_else(|| "not run (set AW_TINT)".into())
    );
    std::panic::set_hook(Box::new(|_| {}));
    let mut handles = Vec::new();
    for _ in 0..threads {
        let (jobs, next, results, written, out, tint) =
            (jobs.clone(), next.clone(), results.clone(), written.clone(), out.clone(), tint.clone());
        handles.push(std::thread::spawn(move || {
            let rules = hooks::registered();
            loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= jobs.len() {
                    break;
                }
                let j = &jobs[k];
                let data = match std::fs::read(&j.path) {
                    Ok(d) => d,
                    Err(e) => {
                        let mut r = results.lock().unwrap();
                        for f in &j.functions {
                            r.push(format!("{}\t{}\t{}\tio-error\t\t\t\t{}", j.corpus, j.sha, f, e));
                        }
                        continue;
                    }
                };
                let bc: Vec<u8> =
                    if j.size > 0 { data[j.offset as usize..(j.offset + j.size) as usize].to_vec() } else { data };
                let parsed = std::panic::catch_unwind(|| airwgsl::parse_bitcode(&bc));
                let mut lines = Vec::new();
                match parsed {
                    Ok(Ok(module)) => {
                        for f in &j.functions {
                            let o = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                lower(&module, &bc, f, &rules)
                            }));
                            let o = match o {
                                Ok(o) => o,
                                Err(p) => Outcome {
                                    status: "crash",
                                    item: String::new(),
                                    kind: String::new(),
                                    construct: String::new(),
                                    reason: p
                                        .downcast_ref::<String>()
                                        .cloned()
                                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                                        .unwrap_or_default()
                                        .replace(['\t', '\n'], " "),
                                    wgsl: None,
                                },
                            };
                            let mut hash = String::new();
                            let mut o = o;
                            if let Some(src) = &o.wgsl {
                                hash = content_key(src.as_bytes());
                                let first = {
                                    let mut w = written.lock().unwrap();
                                    if w.contains_key(&hash) {
                                        false
                                    } else {
                                        w.insert(hash.clone(), None);
                                        true
                                    }
                                };
                                let path = out.join("wgsl").join(format!("{}.wgsl", hash));
                                if first {
                                    std::fs::write(&path, src).unwrap();
                                    let verdict = tint
                                        .as_ref()
                                        .filter(|_| o.status == "compiled")
                                        .and_then(|t| tint_check(t, &path).err());
                                    written.lock().unwrap().insert(hash.clone(), Some(verdict));
                                }
                                // Wait for the thread that checks this text first.
                                let verdict = loop {
                                    if let Some(Some(v)) = written.lock().unwrap().get(&hash) {
                                        break v.clone();
                                    }
                                    std::thread::yield_now();
                                };
                                if let Some(e) = verdict {
                                    o.status = "tint-rejected";
                                    o.reason = e;
                                }
                            }
                            lines.push(format!(
                                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                                j.corpus,
                                j.sha,
                                f,
                                o.status,
                                o.item,
                                o.kind,
                                o.construct.replace('\t', " "),
                                o.reason,
                                hash
                            ));
                        }
                    }
                    Ok(Err(e)) => {
                        let r = tracked::refuse("module-unreadable", "module", None, e.to_string());
                        let o = refused(&r);
                        for f in &j.functions {
                            lines.push(format!(
                                "{}\t{}\t{}\trefused\t{}\t{}\tmodule\t{}\t",
                                j.corpus, j.sha, f, o.item, o.kind, o.reason
                            ));
                        }
                    }
                    Err(_) => {
                        for f in &j.functions {
                            lines.push(format!("{}\t{}\t{}\tcrash\t\t\t\treader panic\t", j.corpus, j.sha, f));
                        }
                    }
                }
                results.lock().unwrap().extend(lines);
                if k % 500 == 0 {
                    eprintln!("{} / {}", k, jobs.len());
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let mut results = results.lock().unwrap().clone();
    results.sort();
    let mut f = std::fs::File::create(out.join("results.tsv")).unwrap();
    writeln!(f, "corpus\tmodule\tfunction\tstatus\titem\tkind\tconstruct\treason\twgsl").unwrap();
    let mut by_status: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut by_item: BTreeMap<(String, String), usize> = BTreeMap::new();
    for l in &results {
        writeln!(f, "{}", l).unwrap();
        let c: Vec<&str> = l.split('\t').collect();
        *by_status.entry((c[0].to_string(), c[3].to_string())).or_default() += 1;
        if c[3] == "refused" {
            *by_item.entry((c[0].to_string(), c[4].to_string())).or_default() += 1;
        }
    }
    let mut lst = std::fs::File::create(out.join("wgsl.lst")).unwrap();
    for h in written.lock().unwrap().keys() {
        let p = out.join("wgsl").join(format!("{}.wgsl", h));
        writeln!(lst, "{}", std::fs::canonicalize(&p).unwrap_or(p).display()).unwrap();
    }
    let mut s = String::new();
    s.push_str("By status:\n");
    for ((c, st), n) in &by_status {
        s.push_str(&format!("  {:10} {:15} {}\n", c, st, n));
    }
    s.push_str("Refusals by tracked item:\n");
    let mut items: Vec<_> = by_item.iter().collect();
    items.sort_by(|a, b| b.1.cmp(a.1));
    for ((c, it), n) in items {
        s.push_str(&format!("  {:10} {:50} {}\n", c, it, n));
    }
    std::fs::write(out.join("summary.txt"), &s).unwrap();
    print!("{}", s);
    let bad = by_status.iter().any(|((_, st), n)| {
        (st == "crash" || st == "naga-rejected" || st == "tint-rejected" || st == "io-error") && *n > 0
    }) || by_item.keys().any(|(_, i)| i == "UNTRACKED");
    std::process::exit(if bad { 1 } else { 0 });
}
