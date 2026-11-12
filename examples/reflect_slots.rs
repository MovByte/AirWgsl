//! Checks that canonicalization gives reflection a value for every binding index or count a global holds.
//!
//! Usage: `reflect_slots LIST`
//!
//! LIST holds bitcode paths, one per line. For every entry point of every module the AIR semantic layer decodes, the
//! function is canonicalized with every function constant zero (as `examples/reflect_corpus.rs` sets them for real
//! Metal) and [`airwgsl::passes::reflection::inputs`] must hold a value for each [`airwgsl::air::Slot::Constant`] its
//! arguments and outputs name, and a predicate value for each predicated argument. Prints the totals; the exit status
//! is non-zero if a slot or predicate has no value.

use airwgsl::air::{AirModule, OutputRole, Role, Slot};
use airwgsl::api::FunctionConstant;

fn slots(e: &airwgsl::air::EntryPoint) -> Vec<u32> {
    let mut out = Vec::new();
    let mut add = |s: Slot| {
        if let Slot::Constant(g) = s {
            out.push(g);
        }
    };
    for a in &e.arguments {
        match &a.role {
            Role::Buffer(b) | Role::IndirectBuffer(b) => {
                add(b.location.index);
                add(b.location.count);
            }
            Role::IndirectConstant(l)
            | Role::Texture { location: l, .. }
            | Role::Sampler(l)
            | Role::InstanceAccelerationStructure(l, _)
            | Role::PrimitiveAccelerationStructure(l, _)
            | Role::IntersectionFunctionTable(l, _)
            | Role::VisibleFunctionTable(l, _)
            | Role::CommandBuffer(l)
            | Role::ComputePipelineState(l)
            | Role::RenderPipelineState(l)
            | Role::VertexInput(l)
            | Role::PatchInput(l)
            | Role::StageIn(Some(l)) => {
                add(l.index);
                add(l.count);
            }
            Role::RenderTarget { index, .. } => add(*index),
            _ => {}
        }
    }
    for o in &e.outputs {
        if let OutputRole::RenderTarget { index, .. } = o.role {
            add(index);
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let list = std::fs::read_to_string(&args[1]).expect("list");
    let (mut entries, mut slot_count, mut missing, mut predicates, mut refused) = (0, 0, 0, 0, 0);
    for path in list.lines().filter(|l| !l.is_empty()) {
        let Ok(m) = AirModule::parse(&std::fs::read(path).expect("read")) else { continue };
        let zeros: Vec<FunctionConstant> = m
            .function_constants
            .iter()
            .map(|c| {
                let ty = m.ir.globals[c.global as usize].value_type;
                let dl = airwgsl::ir::layout::DataLayout::parse(&m.ir.datalayout).unwrap();
                let size = dl.store_size(&m.ir.types, ty).unwrap_or(0) as usize;
                FunctionConstant {
                    index: c.index as u64,
                    name: None,
                    data_type: airwgsl::passes::specialize::data_type(&c.type_name).unwrap_or(0),
                    value: vec![0; size],
                }
            })
            .collect();
        for e in &m.entry_points {
            if m.ir.functions[e.function as usize].is_declaration() {
                continue;
            }
            entries += 1;
            let c = match airwgsl::passes::canonicalize_air(std::slice::from_ref(&m), e, &zeros) {
                Ok(c) => c,
                Err(_) => {
                    refused += 1;
                    continue;
                }
            };
            let inputs = airwgsl::passes::reflection::inputs(&c);
            for g in slots(e) {
                slot_count += 1;
                if !inputs.slots.contains_key(&g) {
                    missing += 1;
                    println!("no slot value: {} {} @{}", path, e.name, m.ir.globals[g as usize].common.name);
                }
            }
            for a in &e.arguments {
                if let Some(p) = &a.predicate {
                    predicates += 1;
                    if !inputs.predicates.contains_key(&p.global) {
                        missing += 1;
                        println!(
                            "no predicate value: {} {} @{}",
                            path, e.name, m.ir.globals[p.global as usize].common.name
                        );
                    }
                }
            }
        }
    }
    println!(
        "entries={} canonicalization-refused={} slots={} predicates={} missing={}",
        entries, refused, slot_count, predicates, missing
    );
    if missing > 0 {
        std::process::exit(1);
    }
}
