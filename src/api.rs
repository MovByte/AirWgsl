//! The compiler interface: the AIR modules of a pipeline and the state that changes generated code go in; WGSL, the
//! binding layout the executor builds, the pass structure, reflection and the host requirements come out.
//!
//! [`compile`] is the only entry point and [`Refusal`] its only failure: there is no fallback path and no
//! best-effort mode. Metal values (pixel formats, vertex formats, blend factors and the rest) are carried as the raw
//! numbers Metal uses, named in each field's documentation; WebGPU values use the WebGPU specification's vocabulary
//! (W3C Candidate Recommendation Draft, 15 September 2026, section 8 "Resource Binding"). Feature, limit and WGSL
//! language-feature names are the specification's strings.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

// Input ---------------------------------------------------------------------------------------------------------------

/// One AIR module of the pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AirModule {
    /// SHA-256 of `bitcode`, the key the guest registered the module under.
    pub hash: [u8; 32],
    /// The module's LLVM bitcode as a metallib stores it for one function.
    pub bitcode: Arc<[u8]>,
    /// The function of this module the pipeline uses as a stage, or `None` for a module only linked into another
    /// module's function (linked functions, preloaded libraries).
    pub entry: Option<EntryPoint>,
}

/// A function that is a pipeline stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryPoint {
    /// Name of the function in the module.
    pub function: String,
    /// `MTLFunctionType` of the function.
    pub function_type: u64,
    /// Function constant values the bitcode was not already specialized with; empty when it was.
    pub constants: Vec<FunctionConstant>,
}

/// One function constant value, by index, or by name when `name` is set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionConstant {
    pub index: u64,
    pub name: Option<String>,
    /// `MTLDataType` of the value.
    pub data_type: u64,
    /// The value's bytes, as Metal lays out a value of that type.
    pub value: Vec<u8>,
}

/// Everything besides the modules that changes the generated code. The executor keys its pipeline variants on it.
///
/// The math options (fast math, denormals) are not here: they are recorded in each module's AIR compile options and
/// read from there.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineDesc {
    pub kind: PipelineKind,
    /// The resources bound to each stage at the draw or dispatch the variant is for.
    pub bindings: Vec<StageBindings>,
}

/// The pipeline's kind and its kind-specific state.
#[derive(Clone, Debug, PartialEq)]
pub enum PipelineKind {
    Render(RenderState),
    Compute(ComputeState),
    /// A tile render pipeline (`MTLTileRenderPipelineDescriptor`).
    Tile(TileState),
}

/// Render pipeline state, from `MTLRenderPipelineDescriptor` and the draw.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderState {
    pub vertex_descriptor: Option<VertexDescriptor>,
    /// Slot `i` at position `i`.
    pub color_attachments: Vec<ColorAttachment>,
    /// `MTLPixelFormat`.
    pub depth_attachment_pixel_format: u64,
    /// `MTLPixelFormat`.
    pub stencil_attachment_pixel_format: u64,
    pub raster_sample_count: u64,
    pub alpha_to_coverage_enabled: bool,
    pub alpha_to_one_enabled: bool,
    pub rasterization_enabled: bool,
    /// `MTLPrimitiveTopologyClass`.
    pub input_primitive_topology: u64,
    /// `MTLPrimitiveType` of the draw, when the variant is for one.
    pub primitive_type: Option<u64>,
    pub tessellation: Option<Tessellation>,
    pub max_vertex_amplification_count: u64,
}

/// `MTLVertexDescriptor`: buffer layouts and attributes, slot `i` at position `i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexDescriptor {
    pub layouts: Vec<VertexBufferLayout>,
    pub attributes: Vec<VertexAttribute>,
}

/// `MTLVertexBufferLayoutDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexBufferLayout {
    pub stride: u64,
    /// `MTLVertexStepFunction`.
    pub step_function: u64,
    pub step_rate: u64,
}

/// `MTLVertexAttributeDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    /// `MTLVertexFormat`.
    pub format: u64,
    pub offset: u64,
    pub buffer_index: u64,
}

/// `MTLRenderPipelineColorAttachmentDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorAttachment {
    /// `MTLPixelFormat`.
    pub pixel_format: u64,
    pub blending_enabled: bool,
    /// `MTLBlendOperation`.
    pub rgb_blend_operation: u64,
    /// `MTLBlendOperation`.
    pub alpha_blend_operation: u64,
    /// `MTLBlendFactor`.
    pub source_rgb_blend_factor: u64,
    /// `MTLBlendFactor`.
    pub destination_rgb_blend_factor: u64,
    /// `MTLBlendFactor`.
    pub source_alpha_blend_factor: u64,
    /// `MTLBlendFactor`.
    pub destination_alpha_blend_factor: u64,
    /// `MTLColorWriteMask`.
    pub write_mask: u64,
}

/// The tessellation fields of `MTLRenderPipelineDescriptor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tessellation {
    /// `MTLTessellationPartitionMode`.
    pub partition_mode: u64,
    pub max_tessellation_factor: u64,
    pub factor_scale_enabled: bool,
    /// `MTLTessellationFactorFormat`.
    pub factor_format: u64,
    /// `MTLTessellationControlPointIndexType`.
    pub control_point_index_type: u64,
    /// `MTLTessellationFactorStepFunction`.
    pub factor_step_function: u64,
    /// `MTLWinding`.
    pub output_winding_order: u64,
}

/// Compute pipeline state, from `MTLComputePipelineDescriptor` and the dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputeState {
    /// `MTLSize` of the dispatch's threadgroup: WGSL fixes the workgroup size in the shader.
    pub threads_per_threadgroup: [u64; 3],
    /// Whether the dispatch is `dispatchThreads:` (grid not a multiple of the threadgroup size).
    pub non_uniform_grid: bool,
    /// `setThreadgroupMemoryLength:atIndex:` lengths, index `i` at position `i`.
    pub threadgroup_memory_lengths: Vec<u64>,
    pub stage_input: Option<StageInputDescriptor>,
}

/// `MTLStageInputOutputDescriptor`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageInputDescriptor {
    /// `MTLBufferLayoutDescriptor` entries: (stride, `MTLStepFunction`, step rate), slot `i` at position `i`.
    pub layouts: Vec<(u64, u64, u64)>,
    /// `MTLAttributeDescriptor` entries: (`MTLAttributeFormat`, offset, buffer index), slot `i` at position `i`.
    pub attributes: Vec<(u64, u64, u64)>,
    /// `MTLIndexType`.
    pub index_type: u64,
    pub index_buffer_index: u64,
}

/// Tile render pipeline state.
#[derive(Clone, Debug, PartialEq)]
pub struct TileState {
    /// `MTLPixelFormat` of each color attachment, slot `i` at position `i`.
    pub color_attachment_pixel_formats: Vec<u64>,
    pub raster_sample_count: u64,
    pub threadgroup_size_matches_tile_size: bool,
}

/// A Metal shader stage, as bindings are numbered per stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MetalStage {
    Vertex,
    Fragment,
    Kernel,
    Tile,
}

/// What is bound to one stage's argument tables.
#[derive(Clone, Debug, PartialEq)]
pub struct StageBindings {
    pub stage: MetalStage,
    pub buffers: Vec<BufferSlot>,
    pub textures: Vec<TextureSlot>,
    pub samplers: Vec<SamplerSlot>,
    pub argument_buffers: Vec<ArgumentBufferSlot>,
}

/// A bound buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferSlot {
    /// Buffer argument table index.
    pub index: u64,
    /// Slots of the draw that hold overlapping ranges of the same buffer share an alias group; `None` when the slot
    /// aliases no other.
    pub alias_group: Option<u32>,
    /// The same buffer is also bound as a vertex or index buffer of the draw.
    pub also_vertex_or_index_buffer: bool,
}

/// A bound texture; its kind decides the WGSL texture type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureSlot {
    /// Texture argument table index.
    pub index: u64,
    /// `MTLTextureType` of the bound texture.
    pub texture_type: u64,
    /// `MTLPixelFormat` of the bound texture (or view).
    pub pixel_format: u64,
    pub sample_count: u64,
    /// `MTLTextureSwizzle` red, green, blue, alpha of the bound texture.
    pub swizzle: [u8; 4],
}

/// A bound sampler: the fields of its `MTLSamplerDescriptor` that WGSL fixes or the shader must emulate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerSlot {
    /// Sampler argument table index.
    pub index: u64,
    /// `MTLSamplerMinMagFilter`.
    pub min_filter: u64,
    /// `MTLSamplerMinMagFilter`.
    pub mag_filter: u64,
    /// `MTLSamplerMipFilter`.
    pub mip_filter: u64,
    /// `MTLSamplerAddressMode` for s, t and r.
    pub address_modes: [u64; 3],
    /// `MTLSamplerBorderColor`.
    pub border_color: u64,
    pub normalized_coordinates: bool,
    /// `MTLCompareFunction`.
    pub compare_function: u64,
    pub max_anisotropy: u64,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub lod_average: bool,
}

/// An argument buffer and the resources its handle words referred to at the draw or dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct ArgumentBufferSlot {
    /// Buffer argument table index of the argument buffer.
    pub index: u64,
    /// (argument id, texture) for each texture member.
    pub textures: Vec<(u64, TextureSlot)>,
    /// (argument id, sampler) for each sampler member.
    pub samplers: Vec<(u64, SamplerSlot)>,
    /// (argument id, buffer) for each buffer member.
    pub buffers: Vec<(u64, BufferSlot)>,
}

/// What the WebGPU adapter offers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostCaps {
    /// `GPUFeatureName` strings the device was created with.
    pub features: BTreeSet<String>,
    /// `GPUSupportedLimits` attribute names and values of the device.
    pub limits: BTreeMap<String, u64>,
    /// WGSL language extension names (`navigator.gpu.wgslLanguageFeatures`).
    pub wgsl_language_features: BTreeSet<String>,
}

// Output --------------------------------------------------------------------------------------------------------------

/// A compiled pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct Compiled {
    /// One WGSL shader module per WebGPU stage.
    pub wgsl: Vec<Stage>,
    pub layout: BindingLayout,
    /// How the executor runs the pipeline's draws or dispatches.
    pub passes: Vec<PassPlan>,
    pub reflection: Reflection,
    pub requirements: HostRequirements,
}

/// A WebGPU shader stage. Values follow `GPUShaderStage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// A set of WebGPU shader stages (`GPUShaderStageFlags`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShaderStages(pub u32);

impl ShaderStages {
    /// `GPUShaderStage.VERTEX`.
    pub const VERTEX: ShaderStages = ShaderStages(0x1);
    /// `GPUShaderStage.FRAGMENT`.
    pub const FRAGMENT: ShaderStages = ShaderStages(0x2);
    /// `GPUShaderStage.COMPUTE`.
    pub const COMPUTE: ShaderStages = ShaderStages(0x4);

    pub fn contains(self, other: ShaderStages) -> bool {
        self.0 & other.0 == other.0
    }
}

impl core::ops::BitOr for ShaderStages {
    type Output = ShaderStages;
    fn bitor(self, other: ShaderStages) -> ShaderStages {
        ShaderStages(self.0 | other.0)
    }
}

impl From<ShaderStage> for ShaderStages {
    fn from(s: ShaderStage) -> ShaderStages {
        match s {
            ShaderStage::Vertex => ShaderStages::VERTEX,
            ShaderStage::Fragment => ShaderStages::FRAGMENT,
            ShaderStage::Compute => ShaderStages::COMPUTE,
        }
    }
}

/// One generated WGSL module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage {
    pub stage: ShaderStage,
    pub entry_point: String,
    /// WGSL source text.
    pub source: String,
}

/// The bind group layouts of the pipeline and where each binding's resource comes from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BindingLayout {
    pub groups: Vec<BindGroupLayout>,
}

/// One bind group layout (`GPUBindGroupLayoutDescriptor`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BindGroupLayout {
    /// The `@group` index.
    pub group: u32,
    pub entries: Vec<BindingEntry>,
}

/// One entry (`GPUBindGroupLayoutEntry`) and its source.
#[derive(Clone, Debug, PartialEq)]
pub struct BindingEntry {
    /// The `@binding` index.
    pub binding: u32,
    pub visibility: ShaderStages,
    pub resource: BindingResource,
    pub source: BindingSource,
}

/// The binding's resource layout: exactly one of the `GPUBindGroupLayoutEntry` members.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingResource {
    /// `GPUBufferBindingLayout`.
    Buffer { ty: BufferBindingType, has_dynamic_offset: bool, min_binding_size: u64 },
    /// `GPUSamplerBindingLayout`.
    Sampler { ty: SamplerBindingType },
    /// `GPUTextureBindingLayout`.
    Texture { sample_type: TextureSampleType, view_dimension: TextureViewDimension, multisampled: bool },
    /// `GPUStorageTextureBindingLayout`; `format` is a `GPUTextureFormat` string.
    StorageTexture { access: StorageTextureAccess, format: String, view_dimension: TextureViewDimension },
}

/// `GPUBufferBindingType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferBindingType {
    /// "uniform"
    Uniform,
    /// "storage"
    Storage,
    /// "read-only-storage"
    ReadOnlyStorage,
}

/// `GPUSamplerBindingType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SamplerBindingType {
    /// "filtering"
    Filtering,
    /// "non-filtering"
    NonFiltering,
    /// "comparison"
    Comparison,
}

/// `GPUTextureSampleType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureSampleType {
    /// "float"
    Float,
    /// "unfilterable-float"
    UnfilterableFloat,
    /// "depth"
    Depth,
    /// "sint"
    Sint,
    /// "uint"
    Uint,
}

/// `GPUTextureViewDimension`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureViewDimension {
    /// "1d"
    D1,
    /// "2d"
    D2,
    /// "2d-array"
    D2Array,
    /// "cube"
    Cube,
    /// "cube-array"
    CubeArray,
    /// "3d"
    D3,
}

/// `GPUStorageTextureAccess`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageTextureAccess {
    /// "write-only"
    WriteOnly,
    /// "read-only"
    ReadOnly,
    /// "read-write"
    ReadWrite,
}

/// Where the executor gets the resource for a binding. Emulations add sources; a source the executor does not know
/// is a failure it reports, never a binding it leaves empty.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BindingSource {
    /// A buffer argument table slot.
    MetalBuffer { stage: MetalStage, index: u64 },
    /// A texture argument table slot.
    MetalTexture { stage: MetalStage, index: u64 },
    /// A sampler argument table slot.
    MetalSampler { stage: MetalStage, index: u64 },
    /// A member of an argument buffer, by argument id.
    ArgumentBufferMember { stage: MetalStage, buffer_index: u64, argument_id: u64 },
    /// A sampler declared in the shader (`constexpr sampler`), by its AIR sampler-state encoding.
    ConstantSampler { encoding: u64 },
    /// The executor's per-draw uniform block, laid out as described.
    Uniforms(UniformBlock),
    /// The snapshot of color attachment `attachment` that a framebuffer-fetch draw reads ([`PassPlan::PerPrimitive`]):
    /// a texture of the attachment's stored format, sample count and size, holding the attachment's contents as they
    /// were before the primitives the current WebGPU pass draws.
    FetchSnapshot { attachment: u32 },
}

/// The executor's per-draw uniform block: what each field holds and where.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UniformBlock {
    pub fields: Vec<UniformField>,
    /// Size in bytes.
    pub size: u64,
}

/// One field of the per-draw uniform block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniformField {
    /// Byte offset in the block.
    pub offset: u64,
    pub value: UniformValue,
}

/// What a uniform field holds. Each value is a `u32` unless stated otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UniformValue {
    /// The draw's base vertex.
    BaseVertex,
    /// The draw's base instance.
    BaseInstance,
    /// Bytes from a buffer binding's aligned base to the Metal offset (WebGPU requires aligned binding offsets).
    ResidualOffset { stage: MetalStage, index: u64 },
    /// Length in bytes of the buffer bound at the slot, from the offset (for `air.buffer_size` and bounds).
    BufferLength { stage: MetalStage, index: u64 },
    /// The origin of the dispatch's stage-in region (`setStageInRegion:`): three `u32`, x, y and z, at consecutive
    /// offsets.
    StageInOrigin,
    /// The size of the dispatch's stage-in region (`setStageInRegion:`): three `u32`, width, height and depth, at
    /// consecutive offsets.
    StageInSize,
}

impl UniformValue {
    /// The number of `u32` words the value occupies in the block.
    pub fn words(&self) -> u64 {
        match self {
            UniformValue::StageInOrigin | UniformValue::StageInSize => 3,
            _ => 1,
        }
    }
}

/// How the executor runs the pipeline's work. Emulations add plans; a plan the executor does not know is a failure it
/// reports, never ignored.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PassPlan {
    /// Each draw or dispatch is one WebGPU draw or dispatch in the current pass.
    Direct,
    /// Framebuffer fetch (plan 5.9): the fragment function reads the color attachments `fetched` (ascending) through
    /// [`BindingSource::FetchSnapshot`] bindings, and each fragment must see the writes of every earlier primitive of
    /// the pass, its own draw's included. The executor draws the primitives of a draw in primitive order, in groups:
    /// before each group it ends the WebGPU pass, copies each fetched attachment into its snapshot, and begins a pass
    /// that loads the attachments; the group is drawn in that pass with the snapshots bound.
    ///
    /// One primitive per group is exact for any geometry. A group of several primitives is exact when no two of them
    /// cover a common pixel, which `overlap` proves for a draw when it is present ([`crate::fetch::overlap`]): its
    /// [`OverlapProver::groups`] gives groups that may be drawn together. Without it every group is one primitive.
    PerPrimitive { fetched: Vec<u32>, overlap: Option<OverlapProver> },
}

pub use crate::fetch::overlap::OverlapProver;

/// Pipeline reflection, field for field the wire protocol's `Reflection` that the guest builds Metal's reflection
/// objects from. Raw values are Metal's (`MTLFunctionType`, `MTLArgumentType`, `MTLBindingAccess`, `MTLDataType`,
/// `MTLTextureType`). Nested types are indices into the stage's `types`; [`NO_TYPE`] is none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reflection {
    pub stages: Vec<StageReflection>,
}

/// The "no type" index of [`ArgumentReflection`], [`TypeReflection`] and [`StructMemberReflection`] type fields.
pub const NO_TYPE: u32 = 0xffff_ffff;

/// The reflection of one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageReflection {
    /// `MTLFunctionType`.
    pub function_type: u64,
    pub arguments: Vec<ArgumentReflection>,
    pub types: Vec<TypeReflection>,
    /// The built-in arguments Metal's reflection lists for the stage (`builtInArguments` of a kernel,
    /// `vertexBuiltInArguments` of a vertex function), in the order the function declares them. `None` when not
    /// reported: a stage Metal keeps no list for, or one with a built-in whose `MTLBuiltInArgument` is not measured.
    pub builtins: Option<Vec<BuiltinReflection>>,
}

/// One `MTLBuiltInArgument`: the values of `-[MTLBuiltInArgument initWithName:type:access:active:index:dataType:
/// builtInType:]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuiltinReflection {
    pub name: String,
    /// `MTLArgumentType` (15 for every built-in, measured).
    pub r#type: u64,
    /// `MTLBindingAccess`.
    pub access: u64,
    pub active: bool,
    pub index: u64,
    /// `MTLDataType` of the argument.
    pub data_type: u64,
    /// `MTLBuiltInArgumentType`.
    pub builtin_type: u64,
}

/// One `MTLArgument`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArgumentReflection {
    pub name: String,
    /// `MTLArgumentType`.
    pub r#type: u64,
    /// `MTLBindingAccess`.
    pub access: u64,
    pub index: u64,
    pub active: bool,
    pub buffer_alignment: u64,
    pub buffer_data_size: u64,
    /// `MTLDataType`.
    pub buffer_data_type: u64,
    pub buffer_struct_type: u32,
    pub buffer_pointer_type: u32,
    pub buffer_indirect_argument_type: u32,
    pub threadgroup_memory_alignment: u64,
    pub threadgroup_memory_data_size: u64,
    /// `MTLTextureType`.
    pub texture_type: u64,
    /// `MTLDataType`.
    pub texture_data_type: u64,
    pub is_depth_texture: bool,
    pub array_length: u64,
    pub indirect_constant_alignment: u64,
    pub indirect_constant_data_size: u64,
    /// `MTLDataType`.
    pub indirect_constant_data_type: u64,
    /// The AIR address space of a buffer (1 device, 2 constant; the `isConstantBuffer` of its `MTLPointerType`), 0 when
    /// not reported.
    pub buffer_address_space: u32,
    /// `MTLDataType`: the element type of threadgroup memory (`threadgroupMemoryDataType`), 0 when not reported.
    pub threadgroup_memory_data_type: u64,
}

/// One `MTLType`; `data_type` selects which members are meaningful (struct: `members`; array: `element_*`,
/// `array_length`, `stride`, `argument_index_stride`; pointer: `element_*`, `access`, `alignment`, `data_size`,
/// `element_is_argument_buffer`, `address_space`; texture reference: `texture_*`, `access`, `is_depth_texture`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TypeReflection {
    /// `MTLDataType`.
    pub data_type: u64,
    pub members: Vec<StructMemberReflection>,
    /// `MTLDataType`.
    pub element_data_type: u64,
    pub element_type: u32,
    pub array_length: u64,
    pub stride: u64,
    pub argument_index_stride: u64,
    /// `MTLBindingAccess`.
    pub access: u64,
    pub alignment: u64,
    pub data_size: u64,
    pub element_is_argument_buffer: bool,
    /// `MTLDataType`.
    pub texture_data_type: u64,
    /// `MTLTextureType`.
    pub texture_type: u64,
    pub is_depth_texture: bool,
    /// A pointer's AIR address space (1 device, 2 constant), 0 when not reported.
    pub address_space: u32,
}

/// One `MTLStructMember`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StructMemberReflection {
    pub name: String,
    pub offset: u64,
    /// `MTLDataType`.
    pub data_type: u64,
    pub r#type: u32,
    pub argument_index: u64,
}

/// What the compiled pipeline needs from the WebGPU device. The executor compares it with the device and reports a
/// shortfall naming the feature or limit; it never runs the pipeline without it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostRequirements {
    /// `GPUFeatureName` strings.
    pub features: BTreeSet<String>,
    /// Minimum values of `GPUSupportedLimits` attributes.
    pub limits: BTreeMap<String, u64>,
    /// WGSL language extension names.
    pub wgsl_language_features: BTreeSet<String>,
}

// Failure -------------------------------------------------------------------------------------------------------------

/// Why the compiler did not produce a pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub kind: RefusalKind,
    /// The construct refused: an `air.*` intrinsic or metadata name, an LLVM instruction or type, a pipeline field.
    pub construct: String,
    pub location: Option<Location>,
    /// What is missing or wrong, in a sentence.
    pub reason: String,
    /// The specification clause that makes it so (Metal Shading Language, WGSL or WebGPU section), when one does.
    pub reference: Option<String>,
}

/// How a refusal reaches the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalKind {
    /// Real Metal on the device rejects this input too: the executor returns the error Metal returns.
    RejectedByDevice,
    /// The construct has no exact WGSL form on any host.
    NoExactLowering,
    /// The construct needs a WebGPU feature or limit this host lacks (named in `construct` or `reason`).
    HostUnsupported,
    /// The compiler does not implement the construct yet.
    NotImplemented,
}

/// Where in the input a refused construct is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Location {
    /// The function containing it.
    pub function: String,
    /// The instruction, in LLVM's textual form, when the construct is one.
    pub instruction: Option<String>,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}: {}", self.kind, self.construct, self.reason)?;
        if let Some(loc) = &self.location {
            write!(f, " (in {}", loc.function)?;
            if let Some(i) = &loc.instruction {
                write!(f, ": {}", i)?;
            }
            write!(f, ")")?;
        }
        if let Some(r) = &self.reference {
            write!(f, " [{}]", r)?;
        }
        Ok(())
    }
}

impl std::error::Error for Refusal {}

// Entry point ---------------------------------------------------------------------------------------------------------

/// Compile the pipeline made of `stage_modules` with the state in `desc` for a device with `host`'s capabilities.
///
/// Vertex, fragment and compute pipelines are lowered by [`crate::wgsl`] with the rules the lowering packages have
/// registered; every construct without a lowering is a [`Refusal`] naming an item of the tracked refusal list
/// ([`crate::wgsl::tracked`]).
// Pipelines are compiled once per variant, so returning the refusal by value costs nothing that matters.
#[allow(clippy::result_large_err)]
pub fn compile(stage_modules: &[AirModule], desc: &PipelineDesc, host: &HostCaps) -> Result<Compiled, Refusal> {
    crate::wgsl::compile(stage_modules, desc, host)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module() -> AirModule {
        AirModule {
            hash: [0; 32],
            bitcode: Arc::from(&[0xde, 0xc0, 0x17, 0x0b][..]),
            entry: Some(EntryPoint { function: "main0".into(), function_type: 3, constants: Vec::new() }),
        }
    }

    fn compute_desc() -> PipelineDesc {
        PipelineDesc {
            kind: PipelineKind::Compute(ComputeState {
                threads_per_threadgroup: [64, 1, 1],
                non_uniform_grid: false,
                threadgroup_memory_lengths: Vec::new(),
                stage_input: None,
            }),
            bindings: Vec::new(),
        }
    }

    #[test]
    fn compile_refuses_an_unreadable_module() {
        let refusal = compile(&[module()], &compute_desc(), &HostCaps::default()).unwrap_err();
        assert_eq!(refusal.kind, RefusalKind::RejectedByDevice);
        assert!(refusal.reason.starts_with("[compiler:module-unreadable]"), "{}", refusal);
    }

    #[test]
    fn shader_stage_flags_follow_the_webgpu_constants() {
        assert_eq!(ShaderStages::from(ShaderStage::Vertex).0, 0x1);
        assert_eq!(ShaderStages::from(ShaderStage::Fragment).0, 0x2);
        assert_eq!(ShaderStages::from(ShaderStage::Compute).0, 0x4);
        let both = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        assert!(both.contains(ShaderStages::FRAGMENT));
        assert!(!both.contains(ShaderStages::COMPUTE));
    }
}
