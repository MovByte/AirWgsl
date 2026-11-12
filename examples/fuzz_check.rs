//! Feeds truncated and corrupted inputs to the readers and checks they return errors, never panic or hang.
//!
//! Usage: `fuzz_check FILES.txt CASES_PER_FILE [THREADS]`
//!
//! FILES.txt lists bitcode modules (raw or wrapped) and metallibs, one path per line. For each file the readers see
//! every truncation at a set of lengths (all of the first 512 bytes, then evenly spaced), and CASES_PER_FILE copies
//! with random corruption (bit flips, byte overwrites, byte runs set to 0x00 or 0xFF, and splices of the file into
//! itself) from a fixed-seed generator, so runs are reproducible. Each parse runs on a thread with a 2 MiB stack, the
//! default for spawned threads, and is timed. Panics and parses slower than 10 seconds are reported with the case
//! that caused them; the exit status is non-zero if there is any. Each case is announced on stderr before it runs, so
//! a case that ends the process (an abort) can be found; a single allocation over 4 GiB is reported there and aborts.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The system allocator, refusing single allocations over 4 GiB (an input-sized allocation gone wrong).
struct Guard;

const HUGE: usize = 4 << 30;

fn report_huge() {
    let _ = std::io::stderr().write_all(b"fuzz_check: allocation over 4 GiB\n");
}

unsafe impl GlobalAlloc for Guard {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > HUGE {
            report_huge();
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > HUGE {
            report_huge();
            return std::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > HUGE {
            report_huge();
            return std::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Guard = Guard;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }
}

fn mutate(data: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut d = data.to_vec();
    if d.is_empty() {
        return d;
    }
    let edits = 1 + rng.below(8);
    for _ in 0..edits {
        match rng.below(5) {
            0 => {
                let i = rng.below(d.len());
                d[i] ^= 1 << rng.below(8);
            }
            1 => {
                let i = rng.below(d.len());
                d[i] = rng.next() as u8;
            }
            2 => {
                let i = rng.below(d.len());
                let n = 1 + rng.below(16).min(d.len() - i);
                let v = if rng.below(2) == 0 { 0 } else { 0xff };
                let end = (i + n).min(d.len());
                for b in &mut d[i..end] {
                    *b = v;
                }
            }
            3 => {
                // Overwrite a 32-bit word with a boundary value.
                let i = rng.below(d.len().saturating_sub(4).max(1));
                let v: u32 = [0, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0xffff][rng.below(6)];
                for (k, b) in v.to_le_bytes().iter().enumerate() {
                    if i + k < d.len() {
                        d[i + k] = *b;
                    }
                }
            }
            _ => {
                // Copy a run of the file over another place.
                let a = rng.below(d.len());
                let b = rng.below(d.len());
                let n = 1 + rng.below(64);
                for k in 0..n {
                    if a + k < d.len() && b + k < d.len() {
                        d[b + k] = data[a + k];
                    }
                }
            }
        }
    }
    d
}

/// Parses `bytes` with every reader that applies; returns whether the parse succeeded.
fn parse_all(bytes: &[u8]) -> bool {
    let mut ok = false;
    if let Ok(m) = airwgsl::parse_bitcode(bytes) {
        // A module that reads must also print.
        let _ = airwgsl::ir::print::print_module(&m);
        ok = true;
    }
    if let Ok(lib) = airwgsl::parse_metallib(bytes) {
        for f in &lib.functions {
            if let Some(module) = f.module
                && let Ok(m) = airwgsl::parse_bitcode(module)
            {
                let _ = airwgsl::ir::print::print_module(&m);
            }
        }
        ok = true;
    }
    if airwgsl::mtlb::parse_container(bytes, "input.metallib").is_ok() {
        ok = true;
    }
    let _ = airwgsl::mtlb::embedded_metallibs(bytes, &[]);
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: fuzz_check FILES.txt CASES_PER_FILE [THREADS]");
        std::process::exit(2);
    }
    let files: Vec<String> =
        std::fs::read_to_string(&args[1]).expect("files").lines().filter(|l| !l.is_empty()).map(String::from).collect();
    let cases: usize = args[2].parse().expect("cases");
    let threads: usize = args.get(3).and_then(|t| t.parse().ok()).unwrap_or(4);
    let files = Arc::new(files);
    let next = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    let counts = Arc::new(Mutex::new((0usize, 0usize, 0usize)));
    let mut handles = Vec::new();
    for _ in 0..threads {
        let (files, next, failures, counts) = (files.clone(), next.clone(), failures.clone(), counts.clone());
        handles.push(std::thread::spawn(move || {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= files.len() {
                    break;
                }
                let data = match std::fs::read(&files[i]) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let mut inputs: Vec<(String, Vec<u8>)> = Vec::new();
                let mut lengths: Vec<usize> = (0..data.len().min(512)).collect();
                let step = (data.len() / 64).max(1);
                lengths.extend((512..data.len()).step_by(step));
                for n in lengths {
                    inputs.push((format!("truncate {}", n), data[..n].to_vec()));
                }
                let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (i as u64 + 1).wrapping_mul(0x1000_0000_01b3));
                for c in 0..cases {
                    inputs.push((format!("mutation {}", c), mutate(&data, &mut rng)));
                }
                for (what, input) in inputs {
                    eprintln!("start\t{}\t{}", files[i], what);
                    let start = Instant::now();
                    let r = std::thread::Builder::new()
                        .stack_size(2 << 20)
                        .spawn(move || std::panic::catch_unwind(|| parse_all(&input)))
                        .unwrap()
                        .join();
                    let elapsed = start.elapsed();
                    let mut c = counts.lock().unwrap();
                    match r {
                        Ok(Ok(true)) => c.0 += 1,
                        Ok(Ok(false)) => c.1 += 1,
                        _ => {
                            c.2 += 1;
                            failures.lock().unwrap().push(format!("{}\t{}\tpanic", files[i], what));
                        }
                    }
                    if elapsed > Duration::from_secs(10) {
                        failures.lock().unwrap().push(format!("{}\t{}\tslow {:?}", files[i], what, elapsed));
                    }
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let (ok, err, panics) = *counts.lock().unwrap();
    let failures = failures.lock().unwrap();
    for f in failures.iter() {
        println!("{}", f);
    }
    println!(
        "files={} cases={} parsed={} rejected={} panics={} failures={}",
        files.len(),
        ok + err + panics,
        ok,
        err,
        panics,
        failures.len()
    );
    if !failures.is_empty() {
        std::process::exit(1);
    }
}
