//! The rule interface the lowering packages register with, exercised by a rule written for this test only: a 32-bit
//! atomic add on device memory. The rule declares the atomic effect on its pointer operand, so the memory model
//! declares the buffer `array<atomic<u32>>` and makes every other access to it atomic; the kernel then runs on Metal,
//! wgpu and Dawn and the counters (sums, independent of the order of the adds) must agree.

use super::harness::{air, case_dir, dawn_node, first_difference, run_dawn, run_metal, run_wgpu};
use airwgsl::api::*;
use airwgsl::num::Rounding;
use airwgsl::wgsl::hooks::{CallRule, CallSite, Effect, RuleCx, Rules};
use std::sync::Arc;

struct AtomicAdd;

impl CallRule for AtomicAdd {
    fn handles(&self, name: &str) -> bool {
        name == "air.atomic.global.add.u.i32"
    }

    fn effects(&self, _call: &CallSite<'_>) -> Vec<(usize, Effect)> {
        vec![(0, Effect::Atomic)]
    }

    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> Result<Option<String>, Refusal> {
        let p = cx.pointer(call.args[0])?;
        assert_eq!(p.regions.len(), 1, "the test kernel adds into one buffer");
        let v = cx.operand(call.args[1])?;
        let name = cx.region_name(p.regions[0]);
        let r = cx.fresh();
        cx.line(format!("let {} = atomicAdd(&{}[{} >> 2u], {});", r, name, p.off, v));
        Ok(Some(r))
    }
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void count(device atomic_uint* counters [[buffer(0)]], device uint* seen [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&counters[i % 8u], i * 3u + 1u, memory_order_relaxed);
    seen[i] = uint(i < 8u);
}
"#;

#[test]
fn a_registered_rule_lowers_its_calls_and_makes_the_region_atomic() {
    let dir = case_dir("hooks");
    let bc = air(&dir, SOURCE);
    let m = AirModule {
        hash: [0; 32],
        bitcode: Arc::from(&bc[..]),
        entry: Some(EntryPoint { function: "count".into(), function_type: 3, constants: Vec::new() }),
    };
    let desc = PipelineDesc {
        kind: PipelineKind::Compute(ComputeState {
            threads_per_threadgroup: [64, 1, 1],
            non_uniform_grid: false,
            threadgroup_memory_lengths: Vec::new(),
            stage_input: None,
        }),
        bindings: Vec::new(),
    };
    // Without the rule the call is refused with its family.
    let r = compile(std::slice::from_ref(&m), &desc, &HostCaps::default()).unwrap_err();
    assert!(r.reason.starts_with("[family:atomic-int32]"), "{}", r);
    let rules = Rules::new().with(Box::new(AtomicAdd));
    let c = airwgsl::wgsl::compile_with(&[m], &desc, &HostCaps::default(), &rules, &Rounding::SoftFloat).unwrap();
    assert!(c.wgsl[0].source.contains("array<atomic<u32>>"), "{}", c.wgsl[0].source);
    let counters = vec![0u8; 32];
    let seen = vec![0xa5u8; 256 * 4];
    let buffers = [counters, seen];
    let reference = run_metal(&dir, SOURCE, "count", [64, 1, 1], [4, 1, 1], &[], &buffers);
    let w = run_wgpu(&c, [4, 1, 1], &buffers);
    for (a, b) in reference.iter().zip(&w) {
        assert_eq!(first_difference(a, b), None);
    }
    if let Some(dn) = dawn_node() {
        let d = run_dawn(&dn, &dir, &c, [4, 1, 1], &buffers);
        for (a, b) in reference.iter().zip(&d) {
            assert_eq!(first_difference(a, b), None);
        }
    }
}
