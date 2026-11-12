//! The argument activity reflection reports with the canonicalization's inputs, for comparison with real Metal's.
//!
//! Usage: `reflect_activity JOBS.tsv OUT.tsv`
//!
//! Each line of JOBS.tsv names a pipeline real Metal reflected (`tools/metal-reflection/record.m`, from the jobs
//! `examples/reflect_corpus.rs` writes, converted by `tests/canon/reflect_activity.py jobs`, which also compares the
//! result with Metal's): `ID<TAB>METALLIB<TAB>STAGES<TAB>CONSTANTS`, where STAGES is a comma list of
//! `stage:function` (`kernel`, `vertex`, `fragment`) and CONSTANTS a comma list of `index:mtldatatype:hex` or `-`.
//! For every stage the function's module is canonicalized with those constants ([`airwgsl::passes::canonicalize_air`])
//! and reflected twice: with no inputs, and with [`airwgsl::passes::reflection::inputs`]. OUT.tsv gets one line per
//! argument of the function's own (vertex-descriptor and stage-in buffers excluded): id, stage, name, index, whether
//! it is active without the inputs and with them, and, for comparison, whether the front end's function uses it as
//! it stands and after specialization and simplification without inlining (`-` where the argument's position is
//! not known by its name and index); or a line naming the stage's refusal.

use std::io::Write;

use airwgsl::air::AirModule;
use airwgsl::api::FunctionConstant;
use airwgsl::reflect::{self, Inputs, StageInput};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|k| u8::from_str_radix(&s[k..k + 2], 16).unwrap()).collect()
}

/// The first binding index of an argument's role, when it has a numeric one.
fn role_index(r: &airwgsl::air::Role) -> Option<u64> {
    use airwgsl::air::{Role, Slot};
    let loc = match r {
        Role::Buffer(b) | Role::IndirectBuffer(b) => Some(b.location),
        Role::IndirectConstant(l)
        | Role::Texture { location: l, .. }
        | Role::Sampler(l)
        | Role::InstanceAccelerationStructure(l, _)
        | Role::PrimitiveAccelerationStructure(l, _)
        | Role::IntersectionFunctionTable(l, _)
        | Role::VisibleFunctionTable(l, _) => Some(*l),
        Role::Imageblock(i) => i.location,
        _ => None,
    }?;
    match loc.index {
        Slot::Value(v) => Some(v as u64),
        Slot::Constant(_) => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: reflect_activity JOBS.tsv OUT.tsv");
        std::process::exit(2);
    }
    let mut out = std::fs::File::create(&args[2]).expect("out");
    for line in std::fs::read_to_string(&args[1]).expect("jobs").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (id, lib, stages, cs) = (f[0], f[1], f[2], f[3]);
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
        let data = std::fs::read(lib).expect("metallib");
        let ml = airwgsl::parse_metallib(&data).expect("metallib");
        for st in stages.split(',').filter(|s| !s.is_empty()) {
            let (stage, name) = st.split_once(':').unwrap();
            let Some(bytes) = ml.functions.iter().find(|x| x.name == name).and_then(|x| x.module) else {
                writeln!(out, "{}\t{}\t-\t-\t-\t-\tno module", id, stage).unwrap();
                continue;
            };
            let m = match AirModule::parse(bytes) {
                Ok(m) => m,
                Err(r) => {
                    writeln!(out, "{}\t{}\t-\t-\t-\t-\trefused: {}", id, stage, r).unwrap();
                    continue;
                }
            };
            let Some(e) = m.entry_point(name).cloned() else { continue };
            let c = match airwgsl::passes::canonicalize_air(std::slice::from_ref(&m), &e, &constants) {
                Ok(c) => c,
                Err(r) => {
                    writeln!(out, "{}\t{}\t-\t-\t-\t-\tcanonicalization refused: {}", id, stage, r).unwrap();
                    continue;
                }
            };
            let inputs = airwgsl::passes::reflection::inputs(&c);
            // Two narrower notions of "unused", for comparison: in the front end's function as it is, and after
            // specialization and simplification without inlining.
            let uses = |m: &airwgsl::ir::Module, f: usize| -> std::collections::BTreeSet<u32> {
                let func = &m.functions[f];
                let mut used = std::collections::BTreeSet::new();
                for b in &func.blocks {
                    for &i in &b.insts {
                        for v in func.inst(i).op.operands() {
                            if let airwgsl::ir::Value::Arg(a) = v {
                                used.insert(a);
                            }
                        }
                    }
                }
                used
            };
            let fi = e.function as usize;
            let used_front = uses(&m.ir, fi);
            let used_local = {
                let mut ir = m.ir.clone();
                let recs: Vec<airwgsl::passes::specialize::ConstantRecord> = m
                    .function_constants
                    .iter()
                    .map(|c| airwgsl::passes::specialize::ConstantRecord {
                        global: c.global,
                        type_name: c.type_name.clone(),
                        name: c.name.clone(),
                        index: c.index as u64,
                    })
                    .collect();
                let ok = airwgsl::passes::specialize::specialize(&mut ir, &recs, &constants).is_ok()
                    && airwgsl::passes::static_init::run(&mut ir).is_ok()
                    && airwgsl::passes::simplify(&mut ir, fi).is_ok();
                if ok { Some(uses(&ir, fi)) } else { None }
            };
            let base = reflect::stage(&m, &e, &Inputs::default(), StageInput::None);
            let ours = reflect::stage(&m, &e, &inputs, StageInput::None);
            match ours {
                Err(r) => writeln!(out, "{}\t{}\t-\t-\t-\t-\treflection refused: {}", id, stage, r).unwrap(),
                Ok(o) => {
                    for a in &o.arguments {
                        let b = base
                            .as_ref()
                            .ok()
                            .and_then(|b| b.arguments.iter().find(|x| x.name == a.name && x.index == a.index));
                        let base_active = b.map(|x| if x.active { "1" } else { "0" }).unwrap_or("-");
                        // The argument's position, by its name and index among the entry point's records.
                        let pos = e
                            .arguments
                            .iter()
                            .find(|x| {
                                x.name.as_deref() == Some(a.name.as_str()) && role_index(&x.role) == Some(a.index)
                            })
                            .map(|x| x.position);
                        let front = pos.map(|p| if used_front.contains(&p) { "1" } else { "0" }).unwrap_or("-");
                        let local = match (&used_local, pos) {
                            (Some(u), Some(p)) => {
                                if u.contains(&p) {
                                    "1"
                                } else {
                                    "0"
                                }
                            }
                            _ => "-",
                        };
                        writeln!(
                            out,
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                            id,
                            stage,
                            a.name,
                            a.index,
                            base_active,
                            if a.active { "1" } else { "0" },
                            front,
                            local
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
}
