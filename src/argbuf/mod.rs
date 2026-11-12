//! Argument buffers (MSL 2.13, plan 5.5) on the A10X's tier 1 (D1: `argumentBuffersSupport` = 0,
//! `supportsArgumentBuffersTier2` = false).
//!
//! On tier 1 an argument buffer is encoded only through `MTLArgumentEncoder`. The plugin's encoder writes, into the
//! guest buffer at the offsets of the AIR struct layout (`air.struct_type_info`), the raw bytes of each constant
//! member and, for each texture, sampler or buffer member, a 64-bit handle word: the object's wire handle (WebMetal
//! wire schema, "Handles"). Apart from the handle words the buffer is ordinary memory, so an argument buffer whose
//! members are all constants (`air.indirect_constant`, at any nesting) is a buffer argument of the memory model: it is
//! bound at its buffer index and read at the layout's offsets, which is exactly what the encoder wrote.
//!
//! A resource member is refused, naming what is missing for an exact lowering:
//!
//! - a buffer member (`compiler:argument-buffer-buffer`): the shader dereferences the member's address, the bound
//!   buffer plus the offset of `setBuffer:offset:atIndex:`, but the handle word carries only the buffer's handle, so
//!   the wire schema gives the executor no offset to bind the member with;
//! - a texture or sampler member (`compiler:argument-buffer-handle`): the texture and sampler lowering takes its
//!   handles from entry arguments only ([`crate::wgsl::hooks::RuleCx::handle_arg`]); a handle loaded from an argument
//!   buffer needs a binding sourced from the member (`BindingSource::ArgumentBufferMember`) through that lowering;
//! - command buffers and pipeline states (`compiler:icb-layout`, [`crate::icb`]), function tables
//!   ([`crate::ftable`]) and ray-tracing objects (family `ray-tracing`), as those constructs are refused elsewhere.
//!
//! An argument buffer the function does not use is lowered to no binding.

use crate::air::{Argument, IndirectArgument, Role, Slot, StructInfo};
use crate::api::{Location, MetalStage, Refusal};
use crate::wgsl::hooks::{StageCx, StageRule};
use crate::wgsl::tracked;

/// What a member of an argument buffer needs, most fundamental first: the order in which refusals are reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Need {
    RayTracing,
    FunctionTable,
    CommandBuffer,
    Buffer,
    Handle,
    Unknown,
}

fn need(a: &Argument) -> Option<Need> {
    Some(match &a.role {
        Role::IndirectConstant(_) => return None,
        Role::Buffer(_) | Role::IndirectBuffer(_) => Need::Buffer,
        Role::Texture { .. } | Role::Sampler(_) => Need::Handle,
        Role::CommandBuffer(_) | Role::ComputePipelineState(_) | Role::RenderPipelineState(_) => Need::CommandBuffer,
        Role::VisibleFunctionTable(..) => Need::FunctionTable,
        Role::InstanceAccelerationStructure(..)
        | Role::PrimitiveAccelerationStructure(..)
        | Role::IntersectionFunctionTable(..) => Need::RayTracing,
        _ => Need::Unknown,
    })
}

/// The most fundamental need of a struct's members, with the member's name.
fn needs(s: &StructInfo, out: &mut Option<(Need, String)>) {
    for m in &s.members {
        if let Some(IndirectArgument::Resource(a)) = &m.indirect
            && let Some(n) = need(a)
            && out.as_ref().is_none_or(|(o, _)| n < *o)
        {
            *out = Some((n, m.name.clone()));
        }
        if let Some(nested) = &m.nested {
            needs(nested, out);
        }
    }
}

/// The lowering of argument buffers.
pub struct ArgumentBuffers;

impl StageRule for ArgumentBuffers {
    fn handles_argument(&self, _stage: MetalStage, kind: &str) -> bool {
        kind == "air.indirect_buffer"
    }

    fn argument(&self, cx: &mut dyn StageCx, arg: u32, record: &Argument) -> Result<String, Refusal> {
        let fi = cx.function();
        let fname = cx.module().functions[fi].common.name.clone();
        let loc = || Some(Location { function: fname.clone(), instruction: None });
        if record.unused || !crate::icb::arg_used(cx.module(), fi, arg) {
            return Ok("0u".into());
        }
        let Role::IndirectBuffer(b) = &record.role else {
            unreachable!("the rule handles air.indirect_buffer records")
        };
        let Some(info) = &b.struct_info else {
            return Err(tracked::refuse(
                "entry-metadata",
                "air.indirect_buffer",
                loc(),
                "an argument buffer without air.struct_type_info",
            ));
        };
        let mut worst = None;
        needs(info, &mut worst);
        if let Some((n, member)) = worst {
            return Err(match n {
                Need::RayTracing => tracked::refuse_family("ray-tracing", member, loc()),
                Need::FunctionTable => crate::ftable::refusal("air.visible_function_table", &fname),
                Need::CommandBuffer => crate::icb::refusal("air.command_buffer", &fname),
                Need::Buffer => tracked::refuse(
                    "argument-buffer-buffer",
                    member,
                    loc(),
                    "a buffer member of an argument buffer: its handle word has no offset for the binding",
                ),
                Need::Handle => tracked::refuse(
                    "argument-buffer-handle",
                    member,
                    loc(),
                    "a texture or sampler member of an argument buffer: the texture lowering binds handles of entry \
                     arguments only",
                ),
                Need::Unknown => tracked::refuse(
                    "argument-buffer",
                    member,
                    loc(),
                    "an argument-buffer member of a kind the argument-buffer lowering does not know",
                ),
            });
        }
        let index = match b.location.index {
            Slot::Value(v) => v as u64,
            Slot::Constant(_) => {
                return Err(tracked::refuse_family("function-constants", "air.indirect_buffer", loc()));
            }
        };
        let arg_ty = cx.module().functions[fi].args[arg as usize].ty;
        let space = b.address_space.or(cx.module().types.pointer_addrspace(arg_ty)).unwrap_or(1);
        if space != 1 && space != 2 {
            return Err(tracked::refuse(
                "entry-metadata",
                "air.indirect_buffer",
                loc(),
                format!("an argument buffer in address space {}", space),
            ));
        }
        Ok(cx.buffer_region(arg, index, space))
    }
}
