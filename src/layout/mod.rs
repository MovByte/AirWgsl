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

    /// The WGSL expression of a uniform field (a `u32`), allocating it on first use. For a value of several words
    /// ([`UniformValue::words`]) it is the first word; [`Layout::uniform_words`] gives each.
    pub fn uniform(&mut self, stage: MetalStage, value: UniformValue) -> String {
        self.uniform_words(stage, value).swap_remove(0)
    }

    /// The WGSL expressions (each a `u32`) of the words of a uniform field, allocating it on first use. Fields are
    /// packed word after word in the order they are first asked for.
    pub fn uniform_words(&mut self, stage: MetalStage, value: UniformValue) -> Vec<String> {
        self.uniform_stages = self.uniform_stages | ShaderStages::from(shader_stage(stage));
        let words = value.words();
        let offset = match self.fields.iter().find(|f| f.value == value) {
            Some(f) => f.offset,
            None => {
                let offset = self.fields.last().map(|f| f.offset + 4 * f.value.words()).unwrap_or(0);
                self.fields.push(UniformField { offset, value });
                offset
            }
        };
        (0..words)
            .map(|k| {
                let o = offset + 4 * k;
                format!("{}[{}][{}]", UNIFORM_VAR, o / 16, (o / 4) % 4)
            })
            .collect()
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
            BindingSource::MetalTextureLevel { index, level, .. } => format!("aw_t{}{}_l{}", s, index, level),
            BindingSource::MetalSampler { index, .. } => format!("aw_s{}{}", s, index),
            BindingSource::ArgumentBufferMember { buffer_index, argument_id, .. } => {
                format!("aw_ab{}{}_{}", s, buffer_index, argument_id)
            }
            BindingSource::ConstantSampler { encoding } => format!("aw_cs{}{:x}", s, encoding),
            BindingSource::MetalBuffer { index, .. } => format!("aw_rb{}{}", s, index),
            BindingSource::Uniforms(_) => format!("aw_ru{}", s),
            BindingSource::FetchSnapshot { attachment } => format!("aw_fb{}{}", s, attachment),
            BindingSource::TessellationVertices => format!("aw_tv{}", s),
            BindingSource::ControlPointIndices => format!("aw_tc{}", s),
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
        self.check_storage_textures(host)
    }

    /// The stages' storage textures against the host's limits. A texture written at a level computed at run time takes
    /// one storage binding per level ([`BindingSource::MetalTextureLevel`]); when those bindings pass the limit, the
    /// refusal names the textures and their levels.
    fn check_storage_textures(&self, host: &HostCaps) -> Result<(), Refusal> {
        let mut per_stage: BTreeMap<MetalStage, BTreeMap<u64, u64>> = BTreeMap::new();
        let mut other: BTreeMap<MetalStage, u64> = BTreeMap::new();
        for r in &self.resources {
            if !matches!(r.resource, BindingResource::StorageTexture { .. }) {
                continue;
            }
            match r.source {
                BindingSource::MetalTextureLevel { index, .. } => {
                    *per_stage.entry(r.stage).or_default().entry(index).or_default() += 1;
                }
                _ => *other.entry(r.stage).or_default() += 1,
            }
        }
        let stages: std::collections::BTreeSet<MetalStage> = per_stage.keys().chain(other.keys()).copied().collect();
        for stage in stages {
            let levels = per_stage.get(&stage).cloned().unwrap_or_default();
            let n = levels.values().sum::<u64>() + other.get(&stage).copied().unwrap_or(0);
            let names: &[&str] = match stage {
                MetalStage::Vertex => &["maxStorageTexturesInVertexStage", "maxStorageTexturesPerShaderStage"],
                MetalStage::Fragment => &["maxStorageTexturesInFragmentStage", "maxStorageTexturesPerShaderStage"],
                _ => &["maxStorageTexturesPerShaderStage"],
            };
            for name in names {
                let Some(&limit) = host.limits.get(*name) else { continue };
                if n <= limit {
                    continue;
                }
                let multi: Vec<String> = levels
                    .iter()
                    .filter(|(_, k)| **k > 1)
                    .map(|(index, k)| format!("texture {} binds {} levels", index, k))
                    .collect();
                if multi.is_empty() {
                    return Err(tracked::refuse(
                        "host-limit",
                        name.to_string(),
                        None,
                        format!("{:?} stage needs {} storage textures; the host's {} is {}", stage, n, name, limit),
                    ));
                }
                return Err(tracked::refuse(
                    "texture-write-level",
                    name.to_string(),
                    None,
                    format!(
                        "{:?} stage needs {} storage textures: a texture written at a level computed at run time has \
                         a storage binding for each of its levels ({}); the host's {} is {}",
                        stage,
                        n,
                        multi.join(", "),
                        name,
                        limit
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The uniform block's size in bytes (a multiple of 16), 0 when no field is used.
    pub fn uniform_size(&self) -> u64 {
        self.fields.last().map(|f| f.offset + 4 * f.value.words()).unwrap_or(0).div_ceil(16) * 16
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
            // A buffer a rule asked for is storage the stage only reads.
            let space = match r.resource {
                BindingResource::Buffer { .. } => "<storage, read>",
                _ => "",
            };
            out.push_str(&format!(
                "@group({}) @binding({}) var{} {}: {};\n",
                RESOURCE_GROUP, r.binding, space, r.name, r.wgsl_type
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
        // A three-word value takes the next three words; the field after it follows them.
        let origin = l.uniform_words(MetalStage::Kernel, UniformValue::StageInOrigin);
        assert_eq!(origin, ["aw_u[0][2]", "aw_u[0][3]", "aw_u[1][0]"]);
        assert_eq!(l.uniform(MetalStage::Kernel, UniformValue::BaseInstance), "aw_u[1][1]");
        assert_eq!(l.uniform_size(), 32);
        let bl = l.binding_layout();
        let BindingSource::Uniforms(block) = &bl.groups[0].entries[0].source else { panic!("no uniform block") };
        assert_eq!(block.fields.iter().map(|f| f.offset).collect::<Vec<_>>(), [0, 4, 8, 20]);
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
