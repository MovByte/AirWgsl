//! Indirect command buffers encoded from shaders (MSL 6.17).
//!
//! The A10X encodes render and compute commands from shaders (D1: `supportsGFXIndirectCommandBuffers` and
//! `supportsCMPIndirectCommandBuffers` are true). WebGPU has no indirect command buffers, so the plan executes them
//! in the executor from the indirect command buffer's memory, with GPU-encoded commands written into that memory by
//! the shader through the memory model (plan 5.7). The layout of that memory is the plugin's to define, and the wire
//! schema does not define it yet (its `MTLIndirectRenderCommand` and `MTLIndirectComputeCommand` entries write "in a
//! layout this schema does not define yet"). Until it does, every encoding function (family `icb-encoding`) and every
//! used command-buffer or pipeline-state argument is refused with the tracked item `icb-layout`; the commands the A10X
//! cannot encode from shaders at all (tessellation, mesh, render barriers: `icb-encoding-unsupported`) stay refused as
//! rejected by the device. An unused argument of these kinds encodes nothing and is lowered to no binding.

use crate::air::Argument;
use crate::api::{Location, MetalStage, Refusal};
use crate::ir::Value;
use crate::wgsl::hooks::{CallRule, CallSite, RuleCx, StageCx, StageRule};
use crate::wgsl::tracked;

/// The argument record kinds of indirect command buffers and the pipeline states encoded into them.
pub const KINDS: &[&str] = &["air.command_buffer", "air.compute_pipeline_state", "air.render_pipeline_state"];

/// The refusal of an indirect-command-buffer construct.
pub fn refusal(construct: &str, function: &str) -> Refusal {
    tracked::refuse(
        "icb-layout",
        construct.to_string(),
        Some(Location { function: function.to_string(), instruction: None }),
        "the wire schema does not define the memory layout of an indirect command buffer, which commands encoded \
         from a shader are written in",
    )
}

/// Whether IR argument `arg` of function `fi` is used.
pub fn arg_used(m: &crate::ir::Module, fi: usize, arg: u32) -> bool {
    m.functions[fi].insts.iter().any(|i| i.op.operands().contains(&Value::Arg(arg)))
}

/// The refusal of the encoding functions.
pub struct Encoding;

impl CallRule for Encoding {
    fn handles(&self, name: &str) -> bool {
        tracked::family_of(name) == Some("icb-encoding")
    }

    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> Result<Option<String>, Refusal> {
        let fi = cx.function();
        Err(refusal(call.name, &cx.module().functions[fi].common.name))
    }
}

/// The refusal of used command-buffer and pipeline-state arguments.
pub struct Arguments;

impl StageRule for Arguments {
    fn handles_argument(&self, _stage: MetalStage, kind: &str) -> bool {
        KINDS.contains(&kind)
    }

    fn argument(&self, cx: &mut dyn StageCx, arg: u32, record: &Argument) -> Result<String, Refusal> {
        let fi = cx.function();
        if record.unused || !arg_used(cx.module(), fi, arg) {
            return Ok("0u".into());
        }
        Err(refusal(crate::wgsl::io::role_key(&record.role), &cx.module().functions[fi].common.name))
    }
}
