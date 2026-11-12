//! Threadgroup and SIMD-group barriers (MSL 6.10.1): `air.wg.barrier(i32 flags, i32 scope)` and
//! `air.simdgroup.barrier(i32 flags, i32 scope)`.
//!
//! A barrier is an execution barrier for the threads of the threadgroup (SIMD-group) and, per bit of `mem_flags`
//! (MSL Table 6.13, values measured from the target's front end), a fence ordering their accesses to one kind of
//! memory:
//!
//! | flag | value | WGSL |
//! | --- | --- | --- |
//! | `mem_none` | 0 | `workgroupBarrier()` (an execution barrier; its workgroup-memory ordering only adds order) |
//! | `mem_device` | 1 | `storageBarrier()`: control barrier, acquire-release on storage memory, workgroup scope |
//! | `mem_threadgroup` | 2 | `workgroupBarrier()`: control barrier, acquire-release on workgroup memory |
//! | `mem_texture` | 4 | none beyond the execution barrier: MSL orders only `access::read_write` textures with it, which the A10X does not have (D1: `readWriteTextureSupport` = 0) |
//! | `mem_threadgroup_imageblock` | 8 | none beyond the execution barrier: imageblocks are above the A10X's family |
//! | `mem_object_data` | 16 | none beyond the execution barrier: object functions are above the A10X's family |
//!
//! Several flags give one WGSL barrier each, one after the other: every access before the first is ordered before
//! every access after the last, for each kind of memory, which is the single MSL barrier's ordering. The scope operand
//! is `1` for both barriers in the target's AIR; MSL 3.1 has no other (thread scopes and barrier memory orders are
//! Metal 3.2 and 4.1), so any other value is refused.
//!
//! Uniformity: WGSL's synchronization built-ins exist in compute shaders only and must be reached in uniform control
//! flow as WGSL's uniformity analysis proves it (WGSL 15.2, 17.11). MSL requires every thread of the threadgroup to
//! reach a threadgroup barrier, but on Apple GPUs a thread that has ended no longer takes part (MSL 6.10.1), which WGSL
//! has no form for. The rule checks each finished kernel module with WGSL's analysis ([`super::uniform`]) and refuses
//! the stage where a barrier is not proven uniform. A SIMD-group barrier is lowered as the workgroup barrier only
//! because of that proof: in control flow uniform across the workgroup every thread of each SIMD-group reaches the
//! call together, and the workgroup barrier's synchronization and ordering include the SIMD-group's.

use super::uniform;
use crate::api::{Location, MetalStage, Refusal};
use crate::ir::{ConstKind, Value};
use crate::wgsl::hooks::{CallRule, CallSite, RuleCx};
use crate::wgsl::tracked;

/// The measured `mem_flags` bits.
const DEVICE: u64 = 1;
const THREADGROUP: u64 = 2;
const TEXTURE: u64 = 4;
const IMAGEBLOCK: u64 = 8;
const OBJECT_DATA: u64 = 16;

/// The WGSL barriers of a `mem_flags` value.
pub fn barriers(flags: u64) -> Option<Vec<&'static str>> {
    if flags & !(DEVICE | THREADGROUP | TEXTURE | IMAGEBLOCK | OBJECT_DATA) != 0 {
        return None;
    }
    let mut out = Vec::new();
    if flags & DEVICE != 0 {
        out.push("storageBarrier();");
    }
    if flags & THREADGROUP != 0 || out.is_empty() {
        out.push("workgroupBarrier();");
    }
    Some(out)
}

/// The lowering of `air.wg.barrier` and `air.simdgroup.barrier`.
pub struct Barriers;

fn loc(cx: &dyn RuleCx, call: &CallSite<'_>) -> Option<Location> {
    let fi = cx.function();
    Some(Location {
        function: cx.module().functions[fi].common.name.clone(),
        instruction: Some(crate::ir::print::print_instruction(cx.module(), fi, call.inst)),
    })
}

fn int_const(cx: &dyn RuleCx, v: Value) -> Option<u64> {
    match v {
        Value::Const(c) => match &cx.module().constant(c).kind {
            ConstKind::Int(w) => Some(w[0]),
            _ => None,
        },
        _ => None,
    }
}

impl CallRule for Barriers {
    fn handles(&self, name: &str) -> bool {
        name == "air.wg.barrier" || name == "air.simdgroup.barrier"
    }

    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> Result<Option<String>, Refusal> {
        let refuse = |cx: &dyn RuleCx, detail: String| {
            tracked::refuse("sync-operand", call.name.to_string(), loc(cx, call), detail)
        };
        if call.args.len() != 2 {
            return Err(refuse(cx, format!("{} operands where the target's front end passes 2", call.args.len())));
        }
        let Some(flags) = int_const(cx, call.args[0]) else {
            return Err(refuse(cx, "memory flags that are not a constant".into()));
        };
        match int_const(cx, call.args[1]) {
            Some(1) => {}
            other => {
                return Err(refuse(
                    cx,
                    format!("scope {:?}: the target's front end passes 1 for MSL 3.1 barriers", other),
                ));
            }
        }
        let Some(lines) = barriers(flags) else {
            return Err(refuse(cx, format!("memory flags {:#x} outside MSL's mem_flags", flags)));
        };
        for l in lines {
            cx.line(l.to_string());
        }
        Ok(None)
    }

    fn check_stage(&self, stage: MetalStage, function: &str, wgsl: &str) -> Result<(), Refusal> {
        if !uniform::has_synchronization(wgsl) {
            return Ok(());
        }
        let loc = Some(Location { function: function.to_string(), instruction: None });
        if stage != MetalStage::Kernel {
            return Err(tracked::refuse(
                "barrier-stage",
                "air.wg.barrier",
                loc,
                format!(
                    "a barrier in a {:?} function: WGSL has synchronization functions in compute shaders only",
                    stage
                ),
            ));
        }
        match uniform::check(wgsl) {
            Ok(f) if f.is_empty() => Ok(()),
            Ok(f) => Err(tracked::refuse(
                "barrier-uniformity",
                f[0].call.clone(),
                loc,
                format!(
                    "WGSL's uniformity analysis cannot prove {} to be in uniform control flow: it depends on {}",
                    f[0].call, f[0].cause
                ),
            )),
            Err(e) => Err(tracked::refuse(
                "uniformity-text",
                "air.wg.barrier",
                loc,
                format!("the uniformity analysis does not read the generated module: {}", e),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_map_to_wgsl_barriers() {
        assert_eq!(barriers(0), Some(vec!["workgroupBarrier();"]));
        assert_eq!(barriers(1), Some(vec!["storageBarrier();"]));
        assert_eq!(barriers(2), Some(vec!["workgroupBarrier();"]));
        assert_eq!(barriers(3), Some(vec!["storageBarrier();", "workgroupBarrier();"]));
        assert_eq!(barriers(4), Some(vec!["workgroupBarrier();"]));
        assert_eq!(barriers(5), Some(vec!["storageBarrier();"]));
        assert_eq!(barriers(8), Some(vec!["workgroupBarrier();"]));
        assert_eq!(barriers(16), Some(vec!["workgroupBarrier();"]));
        assert_eq!(barriers(32), None);
        assert_eq!(tracked::family_of("air.wg.barrier"), Some("threadgroup-barrier"));
        assert_eq!(tracked::family_of("air.simdgroup.barrier"), Some("simdgroup-barrier"));
    }
}
