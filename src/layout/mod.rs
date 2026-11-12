//! Binding assignment (plan 5.5).
//!
//! Group 0 holds the executor's per-draw uniform block: one `array<vec4<u32>, N>` whose fields ([`UniformValue`]) are
//! allocated as the lowering asks for them (residual offsets of buffer bindings, base vertex and instance, ...).
//! Group 1 holds the guest resources the specialized shader actually uses, one binding each, numbered densely in the
//! order of the stages and their Metal argument-table indices.
//!
//! Buffers are storage bindings: read-only when no stage writes through them, `read_write` otherwise, and of
//! `atomic<u32>` words when the memory model makes their region atomic. Uniform bindings are never used for guest data.
//!
//! **Alias variants.** Metal lets one buffer be bound at several indices of a stage; WebGPU rejects a writable binding
//! that overlaps another binding of the same usage scope. Slots that the pipeline description puts in one alias group
//! share one WGSL variable; each slot keeps its own residual offset (bytes from the shared binding's aligned base to the
//! slot's Metal offset), so the memory model addresses the same bytes through either slot. The binding's source is the
//! group's lowest slot, and the executor binds the buffer from an aligned base at or below every member's offset and
//! covering every member's range, as it computed the group. A buffer that is also a vertex or index buffer of the draw
//! and is written by the shader has no WebGPU form (plan 5.5) and is refused.
//!
//! **Limits.** A stage that needs more storage buffers than the host's `maxStorageBuffersPerShaderStage` (or the
//! per-stage limits where the host reports them) is refused with both values: the exact coalescing of plan 5.5 (copying
//! read-only buffers into one arena in stream order) is an executor pass that is not implemented yet.

// The refusal is the compiler's only failure value (api::compile) and is returned by value throughout.
#![allow(clippy::result_large_err)]

use crate::api::{
    BindGroupLayout, BindingEntry, BindingLayout, BindingResource, BindingSource, BufferBindingType, HostCaps,
    MetalStage, Refusal, ShaderStage, ShaderStages, StageBindings, UniformBlock, UniformField, UniformValue,
};
use crate::wgsl::tracked;
use std::collections::BTreeMap;

/// Group of the executor's uniform block.
pub const UNIFORM_GROUP: u32 = 0;
/// Group of the guest resources.
pub const RESOURCE_GROUP: u32 = 1;

/// The WGSL name of the uniform block variable.
pub const UNIFORM_VAR: &str = "aw_u";

/// A storage-buffer variable: one or more Metal slots of one stage.
#[derive(Clone, Debug)]
pub struct BufferVar {
    pub stage: MetalStage,
    /// The WGSL variable.
    pub name: String,
    /// The Metal buffer indices it serves.
    pub slots: Vec<u64>,
    pub written: bool,
    pub atomic: bool,
    pub binding: u32,
}

/// The pipeline's layout, built while the stages are lowered.
#[derive(Default)]
pub struct Layout {
    fields: Vec<UniformField>,
    /// Stages that read the uniform block.
    uniform_stages: ShaderStages,
    buffers: Vec<BufferVar>,
    resources: Vec<ResourceVar>,
}

/// A binding a lowering rule asked for (a texture, a sampler, an argument-buffer member).
#[derive(Clone, Debug)]
pub struct ResourceVar {
    pub stage: MetalStage,
    pub name: String,
    /// The WGSL type of the variable (`texture_2d<f32>`, `sampler`, ...).
    pub wgsl_type: String,
    pub resource: BindingResource,
    pub source: BindingSource,
    pub binding: u32,
}

fn shader_stage(s: MetalStage) -> ShaderStage {
    match s {
        MetalStage::Vertex => ShaderStage::Vertex,
        MetalStage::Fragment => ShaderStage::Fragment,
        MetalStage::Kernel | MetalStage::Tile => ShaderStage::Compute,
    }
}

impl Layout {
    pub fn new() -> Layout {
        Layout::default()
    }

    /// The WGSL expression of a uniform field (a `u32`), allocating it on first use.
    pub fn uniform(&mut self, stage: MetalStage, value: UniformValue) -> String {
        self.uniform_stages = self.uniform_stages | ShaderStages::from(shader_stage(stage));
        let offset = match self.fields.iter().find(|f| f.value == value) {
            Some(f) => f.offset,
            None => {
                let offset = 4 * self.fields.len() as u64;
                self.fields.push(UniformField { offset, value });
                offset
            }
        };
        format!("{}[{}][{}]", UNIFORM_VAR, offset / 16, (offset / 4) % 4)
    }

    /// The storage variable serving buffer `index` of `stage` (creating it), from the stage's bindings: slots of one
    /// alias group share a variable.
    pub fn buffer(&mut self, stage: MetalStage, index: u64, bindings: Option<&StageBindings>) -> String {
        if let Some(b) = self.buffers.iter().find(|b| b.stage == stage && b.slots.contains(&index)) {
            return b.name.clone();
        }
        let group = bindings.and_then(|sb| sb.buffers.iter().find(|s| s.index == index)).and_then(|s| s.alias_group);
        if let Some(g) = group
            && let Some(b) = self.buffers.iter_mut().find(|b| {
                b.stage == stage
                    && bindings
                        .map(|sb| sb.buffers.iter().any(|s| s.alias_group == Some(g) && b.slots.contains(&s.index)))
                        .unwrap_or(false)
            })
        {
            b.slots.push(index);
            b.slots.sort();
            return b.name.clone();
        }
        let prefix = match stage {
            MetalStage::Vertex => "v",
            MetalStage::Fragment => "f",
            MetalStage::Kernel => "k",
            MetalStage::Tile => "t",
        };
        let name = format!("aw_b{}{}", prefix, index);
        self.buffers.push(BufferVar {
            stage,
            name: name.clone(),
            slots: vec![index],
            written: false,
            atomic: false,
            binding: 0,
        });
        name
    }

    /// The WGSL name of the binding variable for a resource source of `stage`.
    pub fn resource_name(stage: MetalStage, source: &BindingSource) -> String {
        let s = match stage {
            MetalStage::Vertex => "v",
            MetalStage::Fragment => "f",
            MetalStage::Kernel => "k",
            MetalStage::Tile => "t",
        };
        match source {
            BindingSource::MetalTexture { index, .. } => format!("aw_t{}{}", s, index),
            BindingSource::MetalSampler { index, .. } => format!("aw_s{}{}", s, index),
            BindingSource::ArgumentBufferMember { buffer_index, argument_id, .. } => {
                format!("aw_ab{}{}_{}", s, buffer_index, argument_id)
            }
            BindingSource::ConstantSampler { encoding } => format!("aw_cs{}{:x}", s, encoding),
            BindingSource::MetalBuffer { index, .. } => format!("aw_rb{}{}", s, index),
            BindingSource::Uniforms(_) => format!("aw_ru{}", s),
        }
    }

    /// Add a binding a rule asked for, once per (stage, source).
    pub fn add_resource(&mut self, r: ResourceVar) {
        if !self.resources.iter().any(|x| x.stage == r.stage && x.source == r.source) {
            self.resources.push(r);
        }
    }

    /// Record how a variable is accessed (after the stage's memory analysis).
    pub fn note_access(&mut self, name: &str, written: bool, atomic: bool) {
        if let Some(b) = self.buffers.iter_mut().find(|b| b.name == name) {
            b.written |= written;
            b.atomic |= atomic;
        }
    }

    pub fn buffer_var(&self, name: &str) -> Option<&BufferVar> {
        self.buffers.iter().find(|b| b.name == name)
    }

    /// Number the bindings and check them against the host's limits and the draw's other uses of the buffers.
    pub fn finish(&mut self, host: &HostCaps, bindings: &[StageBindings]) -> Result<(), Refusal> {
        self.buffers.sort_by_key(|b| (b.stage, b.slots[0]));
        for (i, b) in self.buffers.iter_mut().enumerate() {
            b.binding = i as u32;
        }
        let first = self.buffers.len() as u32;
        self.resources.sort_by_key(|r| r.stage);
        for (i, r) in self.resources.iter_mut().enumerate() {
            r.binding = first + i as u32;
        }
        let mut per_stage: BTreeMap<MetalStage, u64> = BTreeMap::new();
        for b in &self.buffers {
            *per_stage.entry(b.stage).or_default() += 1;
            if b.written {
                let sb = bindings.iter().find(|s| s.stage == b.stage);
                for slot in &b.slots {
                    if sb
                        .and_then(|s| s.buffers.iter().find(|x| x.index == *slot))
                        .is_some_and(|x| x.also_vertex_or_index_buffer)
                    {
                        return Err(tracked::refuse(
                            "vertex-buffer-written",
                            format!("buffer {}", slot),
                            None,
                            "a buffer written by the shader is also a vertex or index buffer of the draw: no WebGPU \
                             usage scope allows both (plan 5.5)",
                        ));
                    }
                }
            }
        }
        for (stage, n) in per_stage {
            let names: &[&str] = match stage {
                MetalStage::Vertex => &["maxStorageBuffersInVertexStage", "maxStorageBuffersPerShaderStage"],
                MetalStage::Fragment => &["maxStorageBuffersInFragmentStage", "maxStorageBuffersPerShaderStage"],
                _ => &["maxStorageBuffersPerShaderStage"],
            };
            for name in names {
                if let Some(&limit) = host.limits.get(*name)
                    && n > limit
                {
                    return Err(tracked::refuse(
                        "host-limit",
                        name.to_string(),
                        None,
                        format!("{:?} stage needs {} storage buffers; the host's {} is {}", stage, n, name, limit),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The uniform block's size in bytes (a multiple of 16), 0 when no field is used.
    pub fn uniform_size(&self) -> u64 {
        (4 * self.fields.len() as u64).div_ceil(16) * 16
    }

    /// The WGSL declarations a stage's module needs: the uniform block (when used by any stage) and the stage's
    /// buffers.
    pub fn declarations(&self, stage: MetalStage) -> String {
        let mut out = String::new();
        if !self.fields.is_empty() && self.uniform_stages.contains(ShaderStages::from(shader_stage(stage))) {
            out.push_str(&format!(
                "@group({}) @binding(0) var<uniform> {}: array<vec4<u32>, {}>;\n",
                UNIFORM_GROUP,
                UNIFORM_VAR,
                self.uniform_size() / 16
            ));
        }
        for b in self.buffers.iter().filter(|b| b.stage == stage) {
            let elem = if b.atomic { "atomic<u32>" } else { "u32" };
            let access = if b.written || b.atomic { "read_write" } else { "read" };
            out.push_str(&format!(
                "@group({}) @binding({}) var<storage, {}> {}: array<{}>;\n",
                RESOURCE_GROUP, b.binding, access, b.name, elem
            ));
        }
        for r in self.resources.iter().filter(|r| r.stage == stage) {
            out.push_str(&format!(
                "@group({}) @binding({}) var {}: {};\n",
                RESOURCE_GROUP, r.binding, r.name, r.wgsl_type
            ));
        }
        out
    }

    /// The pipeline's binding layout.
    pub fn binding_layout(&self) -> BindingLayout {
        let mut groups = Vec::new();
        if !self.fields.is_empty() {
            groups.push(BindGroupLayout {
                group: UNIFORM_GROUP,
                entries: vec![BindingEntry {
                    binding: 0,
                    visibility: self.uniform_stages,
                    resource: BindingResource::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: self.uniform_size(),
                    },
                    source: BindingSource::Uniforms(UniformBlock {
                        fields: self.fields.clone(),
                        size: self.uniform_size(),
                    }),
                }],
            });
        }
        if !self.buffers.is_empty() || !self.resources.is_empty() {
            let mut entries: Vec<BindingEntry> = self
                .buffers
                .iter()
                .map(|b| BindingEntry {
                    binding: b.binding,
                    visibility: ShaderStages::from(shader_stage(b.stage)),
                    resource: BindingResource::Buffer {
                        ty: if b.written || b.atomic {
                            BufferBindingType::Storage
                        } else {
                            BufferBindingType::ReadOnlyStorage
                        },
                        has_dynamic_offset: false,
                        // A runtime-sized array of u32 needs at least one element.
                        min_binding_size: 4,
                    },
                    source: BindingSource::MetalBuffer { stage: b.stage, index: b.slots[0] },
                })
                .collect();
            entries.extend(self.resources.iter().map(|r| BindingEntry {
                binding: r.binding,
                visibility: ShaderStages::from(shader_stage(r.stage)),
                resource: r.resource.clone(),
                source: r.source.clone(),
            }));
            groups.push(BindGroupLayout { group: RESOURCE_GROUP, entries });
        }
        BindingLayout { groups }
    }

    pub fn buffers(&self) -> &[BufferVar] {
        &self.buffers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::BufferSlot;

    #[test]
    fn alias_groups_share_a_variable() {
        let sb = StageBindings {
            stage: MetalStage::Kernel,
            buffers: vec![
                BufferSlot { index: 0, alias_group: Some(1), also_vertex_or_index_buffer: false },
                BufferSlot { index: 2, alias_group: Some(1), also_vertex_or_index_buffer: false },
                BufferSlot { index: 3, alias_group: None, also_vertex_or_index_buffer: false },
            ],
            textures: vec![],
            samplers: vec![],
            argument_buffers: vec![],
        };
        let mut l = Layout::new();
        let a = l.buffer(MetalStage::Kernel, 0, Some(&sb));
        let b = l.buffer(MetalStage::Kernel, 2, Some(&sb));
        let c = l.buffer(MetalStage::Kernel, 3, Some(&sb));
        assert_eq!(a, b);
        assert_ne!(a, c);
        l.note_access(&a, true, false);
        l.finish(&HostCaps::default(), std::slice::from_ref(&sb)).unwrap();
        let bl = l.binding_layout();
        assert_eq!(bl.groups.len(), 1);
        assert_eq!(bl.groups[0].entries.len(), 2);
        assert!(l.declarations(MetalStage::Kernel).contains("var<storage, read_write> aw_bk0: array<u32>"));
    }

    #[test]
    fn uniform_fields_are_packed_in_words() {
        let mut l = Layout::new();
        let e0 = l.uniform(MetalStage::Kernel, UniformValue::ResidualOffset { stage: MetalStage::Kernel, index: 0 });
        let e1 = l.uniform(MetalStage::Kernel, UniformValue::BaseVertex);
        let e0b = l.uniform(MetalStage::Kernel, UniformValue::ResidualOffset { stage: MetalStage::Kernel, index: 0 });
        assert_eq!(e0, "aw_u[0][0]");
        assert_eq!(e1, "aw_u[0][1]");
        assert_eq!(e0, e0b);
        assert_eq!(l.uniform_size(), 16);
    }

    #[test]
    fn storage_limit_is_refused_with_both_values() {
        let mut l = Layout::new();
        for i in 0..3 {
            l.buffer(MetalStage::Kernel, i, None);
        }
        let mut host = HostCaps::default();
        host.limits.insert("maxStorageBuffersPerShaderStage".into(), 2);
        let r = l.finish(&host, &[]).unwrap_err();
        assert!(r.reason.contains("needs 3") && r.reason.contains("is 2"), "{}", r.reason);
    }
}
