//! Reads every file of a system corpus with this crate's container readers and compares what they find with the
//! corpus index: the libraries and module tables in each file, and every module's name, offset, size and SHA-256.
//!
//! Usage: `container_check FILES_DIR INDEX.tsv UNITS.tsv MODULES.tsv REPORT.tsv`
//!
//! FILES_DIR holds the files under their root-relative paths. INDEX.tsv lists the files (`path sha256 size layout
//! ...`), UNITS.tsv the libraries and module tables found in them (`path location kind sha256 size entries stubs
//! modules ...`), MODULES.tsv their modules (`path location entry type module_sha256 offset size`). A file is read
//! with `parse_container`; files no container format matches (the shared cache) and the parts of other files outside
//! the libraries the format names are scanned with `embedded_metallibs`. For each file the multiset of units (kind,
//! SHA-256, size, entries with a module, entries without one, distinct modules) and of modules (entry, type, SHA-256,
//! offset, size) must equal the index's.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use airwgsl::mtlb::{self, Item, Metallib};

type Unit = (String, String, usize, usize, usize, usize);
type Module = (String, String, String, usize, usize);

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn read_rows(path: &str) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .expect("index")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split('\t').map(String::from).collect())
        .collect()
}

fn library_unit(lib: &Metallib) -> Unit {
    let entries = lib.functions.iter().filter(|f| f.module.is_some()).count();
    let stubs = lib.functions.len() - entries;
    let modules: BTreeSet<u64> = lib.functions.iter().filter(|f| f.module.is_some()).map(|f| f.module_offset).collect();
    ("mtlb".into(), hex(&mtlb::sha256(lib.data)), lib.data.len(), entries, stubs, modules.len())
}

fn library_modules(base: usize, lib: &Metallib, out: &mut Vec<Module>) {
    for f in &lib.functions {
        if let Some(m) = f.module {
            let offset = base + (lib.header.module_list.offset + f.module_offset) as usize;
            out.push((f.name.clone(), f.function_type.to_string(), hex(&mtlb::sha256(m)), offset, m.len()));
        }
    }
}

/// Units and modules of one file, and its layout summary.
fn read_file(data: &[u8], path: &str) -> Result<(Vec<Unit>, Vec<Module>, String), String> {
    let items = match mtlb::parse_container(data, path) {
        Ok(items) => items,
        Err(e) if e.message == "no AIR container format matches" => Vec::new(),
        Err(e) => return Err(format!("refused: {}", e)),
    };
    let mut units = Vec::new();
    let mut modules = Vec::new();
    let mut covered = Vec::new();
    let mut table: Option<(String, usize)> = None;
    let mut layout = Vec::new();
    for item in &items {
        match item {
            Item::Metallib { location, offset, lib } => {
                units.push(library_unit(lib));
                library_modules(*offset, lib, &mut modules);
                covered.push((*offset, lib.data.len()));
                layout.push(location.clone());
            }
            Item::Module { location, name, offset, bytes } => {
                modules.push((name.clone(), "-".into(), hex(&mtlb::sha256(bytes)), *offset, bytes.len()));
                let kind = location.rsplit('/').next().unwrap_or(location).to_string();
                match &mut table {
                    Some((k, n)) if *k == kind => *n += 1,
                    Some((k, _)) => return Err(format!("modules of two kinds ({} and {}) in one file", k, kind)),
                    None => table = Some((kind, 1)),
                }
            }
            Item::Other { location, .. } => layout.push(location.clone()),
        }
    }
    if let Some((kind, n)) = &table {
        units.push((kind.clone(), hex(&mtlb::sha256(data)), data.len(), *n, 0, *n));
        layout.push(format!("{}x{}", kind, n));
    }
    let embedded = mtlb::embedded_metallibs(data, &covered);
    for (at, lib) in &embedded {
        units.push(library_unit(lib));
        library_modules(*at, lib, &mut modules);
    }
    if !embedded.is_empty() {
        layout.push(format!("embedded x{}", embedded.len()));
    }
    Ok((units, modules, layout.join(" ")))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!("usage: container_check FILES_DIR INDEX.tsv UNITS.tsv MODULES.tsv REPORT.tsv");
        std::process::exit(2);
    }
    let files = read_rows(&args[2]);
    let mut want_units: BTreeMap<String, Vec<Unit>> = BTreeMap::new();
    for r in read_rows(&args[3]) {
        let n = |i: usize| r[i].parse::<usize>().expect("number");
        want_units.entry(r[0].clone()).or_default().push((r[2].clone(), r[3].clone(), n(4), n(5), n(6), n(7)));
    }
    let mut want_modules: BTreeMap<String, Vec<Module>> = BTreeMap::new();
    for r in read_rows(&args[4]) {
        let n = |i: usize| r[i].parse::<usize>().expect("number");
        want_modules.entry(r[0].clone()).or_default().push((r[2].clone(), r[3].clone(), r[4].clone(), n(5), n(6)));
    }
    let mut report = std::fs::File::create(&args[5]).expect("report");
    let (mut ok, mut bad) = (0usize, 0usize);
    let (mut n_units, mut n_modules) = (0usize, 0usize);
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for row in &files {
        let path = &row[0];
        let data = match std::fs::read(format!("{}/{}", args[1], path)) {
            Ok(d) => d,
            Err(e) => {
                bad += 1;
                writeln!(report, "{}\terror\t{}", path, e).unwrap();
                continue;
            }
        };
        let result = read_file(&data, path).and_then(|(mut units, mut modules, layout)| {
            units.sort();
            modules.sort();
            let mut wu = want_units.get(path).cloned().unwrap_or_default();
            let mut wm = want_modules.get(path).cloned().unwrap_or_default();
            wu.sort();
            wm.sort();
            if units != wu {
                return Err(format!(
                    "units differ: reader {:?}, index {:?}",
                    units.iter().take(3).collect::<Vec<_>>(),
                    wu.iter().take(3).collect::<Vec<_>>()
                ));
            }
            if modules != wm {
                let only_ours: Vec<_> = modules.iter().filter(|m| !wm.contains(m)).take(2).collect();
                let only_index: Vec<_> = wm.iter().filter(|m| !modules.contains(m)).take(2).collect();
                return Err(format!(
                    "modules differ ({} vs {}): reader only {:?}, index only {:?}",
                    modules.len(),
                    wm.len(),
                    only_ours,
                    only_index
                ));
            }
            for u in &units {
                *kinds.entry(u.0.clone()).or_default() += 1;
            }
            Ok((units.len(), modules.len(), layout))
        });
        match result {
            Ok((u, m, layout)) => {
                ok += 1;
                n_units += u;
                n_modules += m;
                writeln!(report, "{}\tok\t{} units, {} modules: {}", path, u, m, layout).unwrap();
            }
            Err(e) => {
                bad += 1;
                writeln!(report, "{}\tmismatch\t{}", path, e).unwrap();
            }
        }
    }
    let kinds: Vec<String> = kinds.iter().map(|(k, n)| format!("{}={}", k, n)).collect();
    println!(
        "files={} ok={} mismatch={} units={} ({}) modules={}",
        files.len(),
        ok,
        bad,
        n_units,
        kinds.join(","),
        n_modules
    );
    if bad != 0 {
        std::process::exit(1);
    }
}
