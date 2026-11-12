//! The AIR semantic layer: what an AIR module's metadata says about its functions, as typed values.
//!
//! [`AirModule::parse`] reads a module's bitcode (through [`crate::bitcode`]) and decodes every named metadata entry
//! AIR defines: the entry points of each stage (`air.vertex`, `air.fragment`, `air.kernel`, `air.visible`,
//! `air.intersection`, `air.mesh`, `air.object`) with their outputs and argument records, the function constants,
//! the constant samplers, the compile options, the AIR and language versions, the module flags and the records of
//! linked and stitched code (`air.visible_function_references`, `air.signature`, `air.ci`,
//! `air.external_initializations`). [`crate::api::AirModule`] is the carrier the executor hands the compiler; this
//! module's [`AirModule`] is its decoded form, holding the IR alongside.
//!
//! The grammar is closed. Every named metadata entry, module flag and record key must be one the grammar names
//! ([`grammar`]); anything else is a [`Refusal`] that names it. A key the grammar knows in a record that does not
//! admit it is refused the same way: no part of the metadata is skipped. Layouts were measured on the system and
//! generated corpora; their meanings are checked against real Metal's reflection ([`crate::reflect`]).

// Refusals are the failure of decoding a module, made once per module; they are returned by value like
// `crate::compile` returns them.
#![allow(clippy::result_large_err)]

pub mod desc;
pub mod grammar;
pub mod md;
pub mod sampler;
pub mod types;

use crate::api::{Refusal, RefusalKind};
use crate::ir::{MdId, Module};
use md::{Cursor, Operand, refusal};

pub use sampler::SamplerDescriptor;
pub use types::{IndirectArgument, Member, StructInfo};

// Module ----------------------------------------------------------------------------------------------------------

/// A decoded AIR module.
#[derive(Clone, Debug)]
pub struct AirModule {
    /// The module's IR.
    pub ir: Module,
    /// `air.version`: (major, minor, patch). `None` in modules that carry no AIR records (libraries of plain
    /// functions linked into others).
    pub air_version: Option<(u32, u32, u32)>,
    /// `air.language_version`: the source language and its (major, minor, patch).
    pub language_version: Option<LanguageVersion>,
    pub compile_options: CompileOptions,
    pub flags: ModuleFlags,
    /// `llvm.ident`: the producer strings.
    pub ident: Vec<String>,
    /// `air.source_file_name`: one per source the module was compiled or linked from.
    pub source_file_names: Vec<String>,
    pub entry_points: Vec<EntryPoint>,
    /// `air.function_constants`.
    pub function_constants: Vec<FunctionConstant>,
    /// `air.sampler_states`.
    pub sampler_states: Vec<ConstantSampler>,
    /// `air.visible_function_references`.
    pub visible_function_references: Vec<VisibleFunctionReference>,
    /// `air.external_initializations`.
    pub external_initializations: Vec<ExternalInitialization>,
    /// `air.signature`: each linked-constants signature's entries.
    pub signatures: Vec<Vec<desc::SignatureEntry>>,
    /// `air.ci`: CoreImage kernel descriptions.
    pub core_image: Vec<desc::CoreImageKernel>,
    /// The type descriptions the stitching records refer to.
    pub type_descriptions: desc::TypeDescriptions,
}

/// `air.language_version`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LanguageVersion {
    /// The language name (`Metal`).
    pub language: String,
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

/// `air.compile_options`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileOptions {
    /// `air.compile.fast_math_enable` (true) or `air.compile.fast_math_disable` (false); `None` when neither.
    pub fast_math: Option<bool>,
    /// `air.compile.denorms_disable`.
    pub denorms_disable: bool,
    /// `air.compile.framebuffer_fetch_enable`.
    pub framebuffer_fetch_enable: bool,
    /// `air.compile.framebuffer_fetch_disable`: the target's compiler writes it in place of
    /// `air.compile.framebuffer_fetch_enable` when framebuffer fetch is off (libGPUCompilerImpl 21H461,
    /// `AIRMetadataPrunerPass::run` at 0x1c4396ce4 and `upgradeAIRModule` at 0x1c43b2808 choose between the two
    /// strings).
    pub framebuffer_fetch_disable: bool,
    /// `air.compile.native_double_disable`.
    pub native_double_disable: bool,
}

/// `llvm.module.flags`: each flag's merge behavior and value.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModuleFlags {
    /// `SDK Version`: [major, minor].
    pub sdk_version: Option<(u32, u32)>,
    pub wchar_size: Option<u32>,
    pub frame_pointer: Option<u32>,
    pub pic_level: Option<u32>,
    /// `air.max_device_buffers`, `air.max_constant_buffers`, `air.max_threadgroup_buffers`, `air.max_textures`,
    /// `air.max_read_write_textures`, `air.max_samplers`: the argument-table sizes the front end compiled against.
    pub max_device_buffers: Option<u32>,
    pub max_constant_buffers: Option<u32>,
    pub max_threadgroup_buffers: Option<u32>,
    pub max_textures: Option<u32>,
    pub max_read_write_textures: Option<u32>,
    pub max_samplers: Option<u32>,
}

/// A function constant (`air.function_constants`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionConstant {
    /// The `externally_initialized` global the value is written to, by index into [`Module::globals`].
    pub global: u32,
    /// MSL type name.
    pub type_name: String,
    pub name: String,
    /// `[[function_constant(index)]]`.
    pub index: u32,
    /// The `i1` some modules add after the index.
    pub flag: Option<bool>,
}

/// A constant sampler (`air.sampler_states`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstantSampler {
    /// The global holding the sampler-state word, by index into [`Module::globals`].
    pub global: u32,
}

impl ConstantSampler {
    /// The sampler-state word: the global's `i64` initializer. AIR the target writes holds one `i64`; a global of
    /// another form (newer compilers write `[2 x i64]`) is refused when the word is needed, as the target cannot
    /// decode it.
    pub fn word(&self, m: &Module) -> Result<u64, Refusal> {
        sampler_word(m, self.global)
    }

    /// The descriptor the target builds from the word, or the target's failure for it.
    pub fn descriptor(&self, m: &Module) -> Result<SamplerDescriptor, Refusal> {
        sampler::decode(self.word(m)?)
    }
}

/// `air.visible_function_reference`: a function referenced by name for linking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleFunctionReference {
    /// The declared function, by index into [`Module::functions`].
    pub function: u32,
    pub name: String,
}

/// `air.externally_initialized`: a global the runtime initializes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalInitialization {
    pub global: u32,
    pub type_name: String,
    pub name: String,
}

// Entry points ----------------------------------------------------------------------------------------------------

/// A function's stage, the named metadata that lists it. `function_type` gives `MTLFunctionType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Vertex,
    Fragment,
    Kernel,
    Visible,
    Intersection,
    Mesh,
    Object,
}

impl Stage {
    /// `MTLFunctionType` (iPhoneOS SDK `MTLLibrary.h`).
    pub fn function_type(self) -> u64 {
        match self {
            Stage::Vertex => 1,
            Stage::Fragment => 2,
            Stage::Kernel => 3,
            Stage::Visible => 5,
            Stage::Intersection => 6,
            Stage::Mesh => 7,
            Stage::Object => 8,
        }
    }

    fn named(name: &str) -> Option<Stage> {
        Some(match name {
            "air.vertex" => Stage::Vertex,
            "air.fragment" => Stage::Fragment,
            "air.kernel" => Stage::Kernel,
            "air.visible" => Stage::Visible,
            "air.intersection" => Stage::Intersection,
            "air.mesh" => Stage::Mesh,
            "air.object" => Stage::Object,
            _ => return None,
        })
    }
}

/// An entry point: a function listed in a stage's named metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryPoint {
    /// By index into [`Module::functions`].
    pub function: u32,
    pub name: String,
    pub stage: Stage,
    /// The return value, or its members.
    pub outputs: Vec<Output>,
    pub arguments: Vec<Argument>,
    /// `early_fragment_tests` (fragment functions).
    pub early_fragment_tests: bool,
    /// `air.patch` (post-tessellation vertex functions).
    pub patch: Option<Patch>,
    /// `air.max_work_group_size` (kernels).
    pub max_work_group_size: Option<u32>,
    /// The tags of an intersection function, in order (`air.triangle`, `air.instancing`, `air.triangle_data`...).
    pub intersection_tags: Vec<IntersectionTag>,
    /// `air.stitching_info` (stitchable visible functions).
    pub stitching_info: Option<desc::StitchingInfo>,
}

/// `air.patch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Patch {
    pub kind: PatchKind,
    /// `air.patch_control_point`, when given.
    pub control_points: Option<u32>,
}

/// The patch type (`MTLPatchType`: triangle 1, quad 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchKind {
    Triangle,
    Quad,
}

/// One tag of an intersection function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntersectionTag {
    BoundingBox,
    Triangle,
    Instancing,
    MultiLevelInstancing,
    TriangleData,
    WorldSpaceData,
    /// An `i32` that follows `air.multi_level_instancing`: the instancing levels.
    Levels(u32),
}

/// A binding index or count: a number, or a global that gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Value(u32),
    /// A global, by index into [`Module::globals`], that holds the value once function constants are set: a
    /// function constant's own global, or one the module's static initializer computes from constants (the system
    /// corpus has render-target indices in `__metal_implicit_attr_int_expr` globals).
    Constant(u32),
}

/// `air.location_index`: the first binding index and the number of indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    pub index: Slot,
    pub count: Slot,
}

/// The access of a buffer or texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// `air.read`.
    Read,
    /// `air.write`.
    Write,
    /// `air.read_write`.
    ReadWrite,
    /// `air.sample` (textures).
    Sample,
}

/// An interpolation qualifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// `air.flat`.
    Flat,
    /// Sampling (`air.center`, `air.centroid`, `air.sample`) and perspective (`air.perspective`,
    /// `air.no_perspective`).
    Smooth { sampling: Sampling, perspective: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    Center,
    Centroid,
    Sample,
}

/// A predicate on a function constant (`air.function_constant`): the record exists when the global, which the
/// module's static initializer computes from the constants, is true.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Predicate {
    /// By index into [`Module::globals`].
    pub global: u32,
    pub type_name: String,
    /// The constant expression's name in the source.
    pub name: String,
}

/// One argument of an entry point (or one resource member of an argument buffer).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Argument {
    /// The LLVM argument number (for an argument-buffer member: the member's position among the resources).
    pub position: u32,
    pub role: Role,
    /// `air.arg_type_name`.
    pub type_name: Option<String>,
    /// `air.arg_name`.
    pub name: Option<String>,
    /// `air.arg_type_size`.
    pub type_size: Option<u32>,
    /// `air.arg_type_align_size`.
    pub type_align: Option<u32>,
    /// `air.arg_unused`.
    pub unused: bool,
    pub predicate: Option<Predicate>,
}

/// What an argument is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    /// `air.buffer`: a pointer to device, constant or threadgroup memory.
    Buffer(Buffer),
    /// `air.indirect_buffer`: an argument buffer.
    IndirectBuffer(Buffer),
    /// `air.indirect_constant`: a constant member of an argument buffer.
    IndirectConstant(Location),
    /// `air.texture`; `raster_order_group` is `air.raster_order_group`.
    Texture { location: Location, access: Access, raster_order_group: Option<u32> },
    /// `air.sampler`.
    Sampler(Location),
    /// `air.imageblock`.
    Imageblock(Imageblock),
    /// `air.instance_acceleration_structure`, with its access when the record states one.
    InstanceAccelerationStructure(Location, Option<Access>),
    /// `air.primitive_acceleration_structure`.
    PrimitiveAccelerationStructure(Location, Option<Access>),
    /// `air.intersection_function_table`.
    IntersectionFunctionTable(Location, Option<Access>),
    /// `air.visible_function_table`.
    VisibleFunctionTable(Location, Option<Access>),
    /// `air.command_buffer`: an indirect command buffer (argument-buffer member).
    CommandBuffer(Location),
    /// `air.compute_pipeline_state` (argument-buffer member).
    ComputePipelineState(Location),
    /// `air.render_pipeline_state` (argument-buffer member).
    RenderPipelineState(Location),
    /// `air.vertex_input`: a vertex attribute.
    VertexInput(Location),
    /// `air.patch_input`: a per-patch attribute of a post-tessellation vertex function.
    PatchInput(Location),
    /// `air.patch_control_point_input`: the control points.
    PatchControlPointInput(desc::ControlPointInput),
    /// `air.stage_in`: a kernel's stage-in data, or one attribute of it when the record has a location.
    StageIn(Option<Location>),
    /// `air.fragment_input`: a varying, matched to a vertex output by name. `interpolant` is set for an MSL
    /// `interpolant<T, P>` (`air.interpolation_function`), which the function interpolates itself: whether `P` is
    /// perspective-correct.
    FragmentInput { name: String, interpolation: Option<Interpolation>, interpolant: Option<bool> },
    /// `air.render_target`: a color attachment read by a fragment function (framebuffer fetch).
    RenderTarget { index: Slot, raster_order_group: Option<u32> },
    /// `air.payload`: an object-to-mesh payload, with the layout of its struct.
    Payload(Option<StructInfo>),
    /// `air.mesh`: a mesh function's output mesh.
    Mesh(desc::MeshInfo),
    /// `air.mesh_grid_properties`.
    MeshGridProperties,
    /// `air.visible_input`: an input of a stitched visible function.
    VisibleInput,
    /// A builtin input, by its key, with the interpolation it carries (fragment `air.position`).
    Builtin { key: &'static str, interpolation: Option<Interpolation> },
}

/// A buffer argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub location: Location,
    pub access: Access,
    /// `air.address_space`: 1 device, 2 constant, 3 threadgroup. Two system modules omit it; the parameter's pointer
    /// type then carries the address space.
    pub address_space: Option<u32>,
    /// `air.buffer_size`: the size of the pointee when the front end knows it.
    pub buffer_size: Option<u32>,
    /// `air.struct_type_info`: the pointee's layout when it is a struct.
    pub struct_info: Option<StructInfo>,
}

/// An imageblock argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Imageblock {
    /// `implicit` or `explicit` layout.
    pub layout: String,
    /// `air.imageblock_data_size`.
    pub data_size: Option<u32>,
    pub struct_info: Option<StructInfo>,
    pub location: Option<Location>,
}

/// One output of an entry point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub role: OutputRole,
    pub type_name: Option<String>,
    pub name: Option<String>,
    pub predicate: Option<Predicate>,
    /// `air.invariant`.
    pub invariant: bool,
    /// `air.shared`.
    pub shared: bool,
    /// `air.raster_order_group`.
    pub raster_order_group: Option<u32>,
}

/// What an output is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputRole {
    /// `air.vertex_output`: a varying, by its linkage name (`user(...)` or `generated(...)`).
    VertexOutput(String),
    /// `air.render_target`: color attachment `index`, dual-source blending index `blend_index`.
    RenderTarget { index: Slot, blend_index: u32 },
    /// `air.depth` with its `air.depth_qualifier` (`air.any`, `air.greater`, `air.less`).
    Depth(&'static str),
    /// `air.clip_distance` with `air.clip_distance_array_size`.
    ClipDistance(Option<u32>),
    /// `air.visible_output`.
    VisibleOutput,
    /// `air.mesh_vertex_data`: a mesh vertex attribute, by its number and linkage name.
    MeshVertexData { index: u32, name: String },
    /// `air.mesh_primitive_data`: a mesh primitive attribute.
    MeshPrimitiveData { index: u32, name: String },
    /// Any other builtin output, by its key.
    Builtin(&'static str),
}

// Decoding --------------------------------------------------------------------------------------------------------

/// The items of one record, each taken at most once; [`Items::finish`] refuses any not taken.
struct Items<'m> {
    items: Vec<grammar::Item<'m>>,
    taken: Vec<bool>,
    what: String,
    function: Option<String>,
}

impl<'m> Items<'m> {
    fn read(c: &mut Cursor<'m>, context: grammar::Context) -> Result<Items<'m>, Refusal> {
        let mut items: Vec<grammar::Item<'m>> = Vec::new();
        while !c.at_end() {
            let item = grammar::item(c, context)?;
            if items.iter().any(|i| i.key == item.key) {
                return Err(refusal(
                    RefusalKind::NotImplemented,
                    item.key,
                    c.function.as_deref(),
                    format!("{}: the key appears twice", c.what),
                ));
            }
            items.push(item);
        }
        let taken = vec![false; items.len()];
        Ok(Items { items, taken, what: format!("{} {}", c.what, c.render()), function: c.function.clone() })
    }

    fn take(&mut self, key: &str) -> Option<Vec<Operand<'m>>> {
        let i = self.items.iter().position(|it| it.key == key)?;
        self.taken[i] = true;
        Some(self.items[i].ops.clone())
    }

    fn flag(&mut self, key: &str) -> bool {
        self.take(key).is_some()
    }

    fn has(&self, key: &str) -> bool {
        self.items.iter().any(|it| it.key == key)
    }

    /// The one key of `keys` present, if exactly one is; a refusal when several are.
    fn one_of(&mut self, keys: &[&'static str]) -> Result<Option<&'static str>, Refusal> {
        let present: Vec<&'static str> = keys.iter().copied().filter(|k| self.has(k)).collect();
        match present.len() {
            0 => Ok(None),
            1 => {
                self.take(present[0]);
                Ok(Some(present[0]))
            }
            _ => Err(self.error(present[1], &format!("only one of {}", present.join(", ")))),
        }
    }

    fn error(&self, construct: &str, expected: &str) -> Refusal {
        refusal(
            RefusalKind::NotImplemented,
            construct,
            self.function.as_deref(),
            format!("{}: the AIR grammar admits {} here", self.what, expected),
        )
    }

    fn require(&mut self, key: &str, role: &str) -> Result<Vec<Operand<'m>>, Refusal> {
        match self.take(key) {
            Some(ops) => Ok(ops),
            None => Err(self.error(key, &format!("{} only with {}", role, key))),
        }
    }

    fn string(&mut self, key: &str) -> Option<String> {
        self.take(key).map(|ops| match ops[0] {
            Operand::Str(s) => String::from_utf8_lossy(s).into_owned(),
            _ => unreachable!("the grammar gives {} a string", key),
        })
    }

    fn int(&mut self, key: &str) -> Option<u32> {
        self.take(key).map(|ops| match ops[0] {
            Operand::Int { value, .. } => value as u32,
            _ => unreachable!("the grammar gives {} an i32", key),
        })
    }

    fn finish(self) -> Result<(), Refusal> {
        for (i, it) in self.items.iter().enumerate() {
            if !self.taken[i] {
                return Err(self.error(it.key, "no such key in a record of this role"));
            }
        }
        Ok(())
    }
}

fn slot(op: Operand<'_>) -> Slot {
    match op {
        Operand::Int { value, .. } => Slot::Value(value as u32),
        Operand::Global(g) => Slot::Constant(g),
        _ => unreachable!("the grammar admits an i32 or a global for a slot"),
    }
}

fn location(items: &mut Items<'_>) -> Option<Location> {
    items.take("air.location_index").map(|ops| Location { index: slot(ops[0]), count: slot(ops[1]) })
}

fn require_location(items: &mut Items<'_>, role: &str) -> Result<Location, Refusal> {
    let ops = items.require("air.location_index", role)?;
    Ok(Location { index: slot(ops[0]), count: slot(ops[1]) })
}

fn predicate(m: &Module, items: &mut Items<'_>) -> Result<Option<Predicate>, Refusal> {
    let Some(ops) = items.take("air.function_constant") else { return Ok(None) };
    let Operand::Node(id) = ops[0] else { unreachable!("the grammar gives air.function_constant a tuple") };
    let mut c = Cursor::node(m, id, format!("air.function_constant !{}", id.0), items.function.as_deref())?;
    let global = c.global_ref("air.function_constant")?;
    let type_name = c.string("air.function_constant type name")?;
    let name = c.string("air.function_constant name")?;
    c.finish()?;
    Ok(Some(Predicate { global, type_name, name }))
}

fn interpolation(items: &mut Items<'_>) -> Result<Option<Interpolation>, Refusal> {
    if items.flag("air.flat") {
        for k in ["air.center", "air.centroid", "air.sample", "air.perspective", "air.no_perspective"] {
            if items.has(k) {
                return Err(items.error(k, "air.flat without a sampling or perspective qualifier"));
            }
        }
        return Ok(Some(Interpolation::Flat));
    }
    let sampling = match items.one_of(&["air.center", "air.centroid", "air.sample"])? {
        None => None,
        Some("air.center") => Some(Sampling::Center),
        Some("air.centroid") => Some(Sampling::Centroid),
        Some(_) => Some(Sampling::Sample),
    };
    let perspective = items.one_of(&["air.perspective", "air.no_perspective"])?.map(|k| k == "air.perspective");
    match (sampling, perspective) {
        (None, None) => Ok(None),
        (Some(sampling), Some(perspective)) => Ok(Some(Interpolation::Smooth { sampling, perspective })),
        (None, Some(_)) => {
            Err(items.error("air.perspective", "a perspective qualifier only with a sampling qualifier"))
        }
        (Some(_), None) => Err(items.error("air.center", "a sampling qualifier only with a perspective qualifier")),
    }
}

fn access(items: &mut Items<'_>, keys: &[&'static str], role: &str) -> Result<Access, Refusal> {
    match items.one_of(keys)? {
        Some("air.read") => Ok(Access::Read),
        Some("air.write") => Ok(Access::Write),
        Some("air.read_write") => Ok(Access::ReadWrite),
        Some("air.sample") => Ok(Access::Sample),
        _ => Err(items.error(role, &format!("{} only with one of {}", role, keys.join(", ")))),
    }
}

fn optional_access(items: &mut Items<'_>) -> Result<Option<Access>, Refusal> {
    Ok(match items.one_of(&["air.read", "air.write", "air.read_write"])? {
        None => None,
        Some("air.read") => Some(Access::Read),
        Some("air.write") => Some(Access::Write),
        Some(_) => Some(Access::ReadWrite),
    })
}

fn struct_info(m: &Module, items: &mut Items<'_>, depth: u32) -> Result<Option<StructInfo>, Refusal> {
    match items.take("air.struct_type_info") {
        None => Ok(None),
        Some(ops) => {
            let Operand::Node(id) = ops[0] else { unreachable!("the grammar gives air.struct_type_info a tuple") };
            Ok(Some(types::struct_info(m, id, items.function.as_deref(), depth)?))
        }
    }
}

fn buffer(m: &Module, items: &mut Items<'_>, role: &str) -> Result<Buffer, Refusal> {
    let location = require_location(items, role)?;
    let access = access(items, &["air.read", "air.write", "air.read_write"], role)?;
    let address_space = items.int("air.address_space");
    let buffer_size = items.int("air.buffer_size");
    let struct_info = struct_info(m, items, 0)?;
    Ok(Buffer { location, access, address_space, buffer_size, struct_info })
}

/// The builtin input keys (arguments that are values the stage provides).
const BUILTIN_INPUTS: &[&str] = &[
    "air.position",
    "air.render_target_array_index",
    "air.viewport_array_index",
    "air.distance",
    "air.vertex_id",
    "air.instance_id",
    "air.base_vertex",
    "air.base_instance",
    "air.amplification_id",
    "air.amplification_count",
    "air.front_facing",
    "air.point_coord",
    "air.primitive_id",
    "air.barycentric_coord",
    "air.sample_id",
    "air.sample_mask_in",
    "air.patch_id",
    "air.position_in_patch",
    "air.thread_position_in_grid",
    "air.thread_position_in_threadgroup",
    "air.thread_index_in_threadgroup",
    "air.threadgroup_position_in_grid",
    "air.threadgroups_per_grid",
    "air.threads_per_grid",
    "air.threads_per_threadgroup",
    "air.dispatch_threads_per_threadgroup",
    "air.thread_index_in_simdgroup",
    "air.threads_per_simdgroup",
    "air.simdgroup_index_in_threadgroup",
    "air.simdgroups_per_threadgroup",
    "air.dispatch_simdgroups_per_threadgroup",
    "air.thread_index_in_quadgroup",
    "air.quadgroup_index_in_threadgroup",
    "air.quadgroups_per_threadgroup",
    "air.dispatch_quadgroups_per_threadgroup",
    "air.stage_in_grid_origin",
    "air.stage_in_grid_size",
    "air.origin",
    "air.direction",
    "air.min_distance",
    "air.max_distance",
    "air.geometry_id",
    "air.object_to_world_transform",
    "air.geometry_intersection_function_table_offset",
];

/// The role keys of an argument record: exactly one is present.
const ARGUMENT_ROLES: &[&str] = &[
    "air.buffer",
    "air.indirect_buffer",
    "air.indirect_constant",
    "air.texture",
    "air.sampler",
    "air.imageblock",
    "air.instance_acceleration_structure",
    "air.primitive_acceleration_structure",
    "air.intersection_function_table",
    "air.visible_function_table",
    "air.command_buffer",
    "air.compute_pipeline_state",
    "air.render_pipeline_state",
    "air.vertex_input",
    "air.patch_input",
    "air.patch_control_point_input",
    "air.stage_in",
    "air.fragment_input",
    "air.render_target",
    "air.payload",
    "air.mesh",
    "air.mesh_grid_properties",
    "air.visible_input",
];

/// Decodes an argument record. `member` is true for a resource member of an argument buffer.
pub(crate) fn decode_argument(m: &Module, id: MdId, function: Option<&str>, member: bool) -> Result<Argument, Refusal> {
    let what = if member { format!("argument-buffer member !{}", id.0) } else { format!("argument !{}", id.0) };
    let mut c = Cursor::node(m, id, what, function)?;
    let position = c.i32("argument number")?;
    let mut items = Items::read(&mut c, grammar::Context::Argument)?;
    let pred = predicate(m, &mut items)?;
    let type_name = items.string("air.arg_type_name");
    let name = items.string("air.arg_name");
    let type_size = items.int("air.arg_type_size");
    let type_align = items.int("air.arg_type_align_size");
    let unused = items.flag("air.arg_unused");
    let roles: Vec<&'static str> =
        ARGUMENT_ROLES.iter().chain(BUILTIN_INPUTS).copied().filter(|k| items.has(k)).collect();
    let role_key = match roles.as_slice() {
        [one] => *one,
        [] => return Err(items.error("argument", "an argument record only with a role key")),
        [_, second, ..] => {
            // `air.position` and the sampling qualifier `air.sample` are not roles on their own.
            return Err(items.error(second, &format!("one role key (found {})", roles.join(", "))));
        }
    };
    items.take(role_key);
    let role = match role_key {
        "air.buffer" => Role::Buffer(buffer(m, &mut items, role_key)?),
        "air.indirect_buffer" => Role::IndirectBuffer(buffer(m, &mut items, role_key)?),
        "air.indirect_constant" => Role::IndirectConstant(require_location(&mut items, role_key)?),
        "air.texture" => {
            let location = require_location(&mut items, role_key)?;
            let access = access(&mut items, &["air.sample", "air.read", "air.write", "air.read_write"], role_key)?;
            Role::Texture { location, access, raster_order_group: items.int("air.raster_order_group") }
        }
        "air.sampler" => Role::Sampler(require_location(&mut items, role_key)?),
        "air.imageblock" => {
            let ops = items.items.iter().find(|i| i.key == "air.imageblock").unwrap().ops.clone();
            let Operand::Str(layout) = ops[0] else { unreachable!("the grammar gives air.imageblock a string") };
            let data_size = items.int("air.imageblock_data_size");
            let struct_info = struct_info(m, &mut items, 0)?;
            let location = location(&mut items);
            Role::Imageblock(Imageblock {
                layout: String::from_utf8_lossy(layout).into_owned(),
                data_size,
                struct_info,
                location,
            })
        }
        "air.instance_acceleration_structure" => {
            Role::InstanceAccelerationStructure(require_location(&mut items, role_key)?, optional_access(&mut items)?)
        }
        "air.primitive_acceleration_structure" => {
            Role::PrimitiveAccelerationStructure(require_location(&mut items, role_key)?, optional_access(&mut items)?)
        }
        "air.intersection_function_table" => {
            Role::IntersectionFunctionTable(require_location(&mut items, role_key)?, optional_access(&mut items)?)
        }
        "air.visible_function_table" => {
            Role::VisibleFunctionTable(require_location(&mut items, role_key)?, optional_access(&mut items)?)
        }
        "air.command_buffer" => Role::CommandBuffer(require_location(&mut items, role_key)?),
        "air.compute_pipeline_state" => Role::ComputePipelineState(require_location(&mut items, role_key)?),
        "air.render_pipeline_state" => Role::RenderPipelineState(require_location(&mut items, role_key)?),
        "air.vertex_input" => Role::VertexInput(require_location(&mut items, role_key)?),
        "air.patch_input" => Role::PatchInput(require_location(&mut items, role_key)?),
        "air.patch_control_point_input" => {
            let ops = items.items.iter().find(|i| i.key == role_key).unwrap().ops.clone();
            let nodes: Vec<MdId> = ops
                .iter()
                .map(|o| match o {
                    Operand::Node(n) => *n,
                    _ => unreachable!("the grammar gives air.patch_control_point_input tuples"),
                })
                .collect();
            Role::PatchControlPointInput(desc::control_points(m, &nodes, function)?)
        }
        "air.stage_in" => Role::StageIn(location(&mut items)),
        "air.fragment_input" => {
            let ops = items.items.iter().find(|i| i.key == role_key).unwrap().ops.clone();
            let Operand::Str(name) = ops[0] else { unreachable!("the grammar gives air.fragment_input a string") };
            let name = String::from_utf8_lossy(name).into_owned();
            if items.flag("air.interpolation_function") {
                let perspective = match items.one_of(&["air.perspective", "air.no_perspective"])? {
                    Some(k) => k == "air.perspective",
                    None => {
                        return Err(items.error(
                            "air.interpolation_function",
                            "air.interpolation_function only with air.perspective or air.no_perspective",
                        ));
                    }
                };
                Role::FragmentInput { name, interpolation: None, interpolant: Some(perspective) }
            } else {
                Role::FragmentInput { name, interpolation: interpolation(&mut items)?, interpolant: None }
            }
        }
        "air.render_target" => {
            let ops = items.items.iter().find(|i| i.key == role_key).unwrap().ops.clone();
            Role::RenderTarget { index: slot(ops[0]), raster_order_group: items.int("air.raster_order_group") }
        }
        "air.payload" => Role::Payload(struct_info(m, &mut items, 0)?),
        "air.mesh" => {
            let ops = items.items.iter().find(|i| i.key == role_key).unwrap().ops.clone();
            let Operand::Node(n) = ops[0] else { unreachable!("the grammar gives air.mesh a tuple") };
            Role::Mesh(desc::mesh(m, n, function)?)
        }
        "air.mesh_grid_properties" => Role::MeshGridProperties,
        "air.visible_input" => Role::VisibleInput,
        key => {
            // Of the builtin inputs, only a fragment function's position and barycentric coordinates carry an
            // interpolation qualifier in the corpora; any other builtin with one is outside the measured grammar.
            let interpolation =
                if matches!(key, "air.position" | "air.barycentric_coord") { interpolation(&mut items)? } else { None };
            Role::Builtin { key, interpolation }
        }
    };
    items.finish()?;
    Ok(Argument { position, role, type_name, name, type_size, type_align, unused, predicate: pred })
}

/// The builtin output keys.
const BUILTIN_OUTPUTS: &[&str] = &[
    "air.position",
    "air.point_size",
    "air.render_target_array_index",
    "air.viewport_array_index",
    "air.sample_mask",
    "air.stencil",
    "air.accept_intersection",
    "air.distance",
];

pub(super) fn decode_output(m: &Module, id: MdId, function: Option<&str>) -> Result<Output, Refusal> {
    let mut c = Cursor::node(m, id, format!("output !{}", id.0), function)?;
    let mut items = Items::read(&mut c, grammar::Context::Output)?;
    let pred = predicate(m, &mut items)?;
    let type_name = items.string("air.arg_type_name");
    let name = items.string("air.arg_name");
    let invariant = items.flag("air.invariant");
    let shared = items.flag("air.shared");
    let raster_order_group = items.int("air.raster_order_group");
    let roles: Vec<&'static str> = [
        "air.vertex_output",
        "air.render_target",
        "air.depth",
        "air.clip_distance",
        "air.visible_output",
        "air.mesh_vertex_data",
        "air.mesh_primitive_data",
    ]
    .iter()
    .chain(BUILTIN_OUTPUTS)
    .copied()
    .filter(|k| items.has(k))
    .collect();
    let role_key = match roles.as_slice() {
        [one] => *one,
        [] => return Err(items.error("output", "an output record only with a role key")),
        [_, second, ..] => return Err(items.error(second, &format!("one role key (found {})", roles.join(", ")))),
    };
    let ops = items.take(role_key).unwrap();
    let role = match role_key {
        "air.vertex_output" => {
            let Operand::Str(s) = ops[0] else { unreachable!("the grammar gives air.vertex_output a string") };
            OutputRole::VertexOutput(String::from_utf8_lossy(s).into_owned())
        }
        "air.render_target" => {
            let Operand::Int { value, .. } = ops[1] else { unreachable!("the grammar gives an i32") };
            OutputRole::RenderTarget { index: slot(ops[0]), blend_index: value as u32 }
        }
        "air.depth" => {
            items.require("air.depth_qualifier", "air.depth")?;
            match items.one_of(&["air.any", "air.greater", "air.less"])? {
                Some(q) => OutputRole::Depth(q),
                None => return Err(items.error("air.depth_qualifier", "air.any, air.greater or air.less after it")),
            }
        }
        "air.clip_distance" => OutputRole::ClipDistance(items.int("air.clip_distance_array_size")),
        "air.visible_output" => OutputRole::VisibleOutput,
        "air.mesh_vertex_data" | "air.mesh_primitive_data" => {
            let (Operand::Int { value, .. }, Operand::Str(s)) = (ops[0], ops[1]) else {
                unreachable!("the grammar gives an i32 and a string")
            };
            let (index, name) = (value as u32, String::from_utf8_lossy(s).into_owned());
            if role_key == "air.mesh_vertex_data" {
                OutputRole::MeshVertexData { index, name }
            } else {
                OutputRole::MeshPrimitiveData { index, name }
            }
        }
        key => OutputRole::Builtin(key),
    };
    items.finish()?;
    Ok(Output { role, type_name, name, predicate: pred, invariant, shared, raster_order_group })
}

fn list(m: &Module, id: MdId, what: &str, function: Option<&str>) -> Result<Vec<MdId>, Refusal> {
    let mut c = Cursor::node(m, id, what.to_string(), function)?;
    let mut out = Vec::new();
    while !c.at_end() {
        out.push(c.node_id(what)?);
    }
    Ok(out)
}

fn decode_entry(m: &Module, descs: &mut desc::TypeDescriptions, stage: Stage, id: MdId) -> Result<EntryPoint, Refusal> {
    let what = format!("{} entry !{}", stage_name(stage), id.0);
    let mut c = Cursor::node(m, id, what.clone(), None)?;
    let function = c.function_ref("entry point function")?;
    let name = m.functions[function as usize].common.name.clone();
    c.function = Some(name.clone());
    let fname = Some(name.as_str());
    let outputs_id = c.node_id("entry point outputs")?;
    let arguments_id = c.node_id("entry point arguments")?;
    let mut outputs = Vec::new();
    for o in list(m, outputs_id, "entry point outputs", fname)? {
        outputs.push(decode_output(m, o, fname)?);
    }
    let mut arguments = Vec::new();
    for a in list(m, arguments_id, "entry point arguments", fname)? {
        arguments.push(decode_argument(m, a, fname, false)?);
    }
    let mut entry = EntryPoint {
        function,
        name: name.clone(),
        stage,
        outputs,
        arguments,
        early_fragment_tests: false,
        patch: None,
        max_work_group_size: None,
        intersection_tags: Vec::new(),
        stitching_info: None,
    };
    // The entry's tags.
    while !c.at_end() {
        match (stage, c.peek()) {
            (Stage::Fragment, Some(Operand::Str(b"early_fragment_tests"))) if !entry.early_fragment_tests => {
                c.advance();
                entry.early_fragment_tests = true;
            }
            (Stage::Vertex, Some(Operand::Node(n))) if entry.patch.is_none() => {
                c.advance();
                let mut p = Cursor::node(m, n, format!("air.patch !{}", n.0), fname)?;
                if p.peek_str() != Some(b"air.patch") {
                    return Err(p.error("air.patch", "air.patch"));
                }
                p.advance();
                let kind = match p.bytes("air.patch")? {
                    b"triangle" => PatchKind::Triangle,
                    b"quad" => PatchKind::Quad,
                    _ => {
                        return Err(refusal(
                            RefusalKind::NotImplemented,
                            "air.patch",
                            fname,
                            "the patch type is neither triangle nor quad",
                        ));
                    }
                };
                let mut control_points = None;
                if p.peek_str() == Some(b"air.patch_control_point") {
                    p.advance();
                    control_points = Some(p.i32("air.patch_control_point")?);
                }
                p.finish()?;
                entry.patch = Some(Patch { kind, control_points });
            }
            (Stage::Kernel, Some(Operand::Node(n))) if entry.max_work_group_size.is_none() => {
                c.advance();
                let mut p = Cursor::node(m, n, format!("air.max_work_group_size !{}", n.0), fname)?;
                if p.peek_str() != Some(b"air.max_work_group_size") {
                    return Err(p.error("air.max_work_group_size", "air.max_work_group_size"));
                }
                p.advance();
                entry.max_work_group_size = Some(p.i32("air.max_work_group_size")?);
                p.finish()?;
            }
            (Stage::Visible, Some(Operand::Node(n))) if entry.stitching_info.is_none() => {
                c.advance();
                entry.stitching_info = Some(desc::stitching_info(m, descs, n, fname)?);
            }
            (Stage::Intersection, Some(Operand::Str(s))) => {
                let tag = match s {
                    b"air.bounding_box" => IntersectionTag::BoundingBox,
                    b"air.triangle" => IntersectionTag::Triangle,
                    b"air.instancing" => IntersectionTag::Instancing,
                    b"air.multi_level_instancing" => IntersectionTag::MultiLevelInstancing,
                    b"air.triangle_data" => IntersectionTag::TriangleData,
                    b"air.world_space_data" => IntersectionTag::WorldSpaceData,
                    _ => return Err(c.error(String::from_utf8_lossy(s).into_owned(), "an intersection tag")),
                };
                c.advance();
                entry.intersection_tags.push(tag);
            }
            (Stage::Intersection, Some(Operand::Int { bits: 32, value }))
                if entry.intersection_tags.last() == Some(&IntersectionTag::MultiLevelInstancing) =>
            {
                c.advance();
                entry.intersection_tags.push(IntersectionTag::Levels(value as u32));
            }
            _ => return Err(c.error(format!("{} entry tag", stage_name(stage)), "a tag of this stage's entry")),
        }
    }
    Ok(entry)
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Vertex => "air.vertex",
        Stage::Fragment => "air.fragment",
        Stage::Kernel => "air.kernel",
        Stage::Visible => "air.visible",
        Stage::Intersection => "air.intersection",
        Stage::Mesh => "air.mesh",
        Stage::Object => "air.object",
    }
}

fn single(name: &str, ops: &[MdId]) -> Result<MdId, Refusal> {
    match ops {
        [one] => Ok(*one),
        _ => Err(refusal(
            RefusalKind::NotImplemented,
            name,
            None,
            format!("!{} has {} entries; the AIR grammar has one", name, ops.len()),
        )),
    }
}

fn decode_flags(m: &Module, ops: &[MdId], flags: &mut ModuleFlags) -> Result<(), Refusal> {
    for &id in ops {
        let mut c = Cursor::node(m, id, format!("llvm.module.flags !{}", id.0), None)?;
        let behavior = c.i32("module flag behavior")?;
        let key = c.string("module flag key")?;
        // Merge behaviors (`Module::ModFlagBehavior`) as measured: 1 Error, 2 Warning, 7 Max, 8 Min.
        let (expected, slot): (u32, &mut Option<u32>) = match key.as_str() {
            "wchar_size" => (1, &mut flags.wchar_size),
            "frame-pointer" => (7, &mut flags.frame_pointer),
            "PIC Level" => (8, &mut flags.pic_level),
            "air.max_device_buffers" => (7, &mut flags.max_device_buffers),
            "air.max_constant_buffers" => (7, &mut flags.max_constant_buffers),
            "air.max_threadgroup_buffers" => (7, &mut flags.max_threadgroup_buffers),
            "air.max_textures" => (7, &mut flags.max_textures),
            "air.max_read_write_textures" => (7, &mut flags.max_read_write_textures),
            "air.max_samplers" => (7, &mut flags.max_samplers),
            "SDK Version" => {
                if behavior != 2 {
                    return Err(c.error("SDK Version", "merge behavior 2 (Warning)"));
                }
                let pair = sdk_version(m, &mut c)?;
                c.finish()?;
                if flags.sdk_version.replace(pair).is_some() {
                    return Err(c.error("SDK Version", "one SDK Version flag"));
                }
                continue;
            }
            other => {
                return Err(refusal(
                    RefusalKind::NotImplemented,
                    other,
                    None,
                    "a module flag the AIR grammar does not name",
                ));
            }
        };
        if behavior != expected {
            return Err(c.error(key.clone(), &format!("merge behavior {}", expected)));
        }
        let value = c.i32(&key)?;
        c.finish()?;
        if slot.replace(value).is_some() {
            return Err(c.error(key, "the flag once"));
        }
    }
    Ok(())
}

fn sdk_version(m: &Module, c: &mut Cursor<'_>) -> Result<(u32, u32), Refusal> {
    use crate::ir::{ConstKind, Metadata, Type, Value};
    if let Some(Operand::Other(id)) = c.peek()
        && let Metadata::Value { ty, value: Value::Const(k) } = m.md(id)
        && let Type::Array { len: 2, elem } = m.types.get(*ty)
        && m.types.get(*elem) == &Type::Int(32)
        && let ConstKind::DataArray(v) = &m.constant(*k).kind
    {
        c.advance();
        return Ok((v[0] as u32, v[1] as u32));
    }
    Err(c.error("SDK Version", "[2 x i32]"))
}

impl AirModule {
    /// Reads `bitcode` (a raw or wrapped LLVM bitcode module) and decodes its AIR metadata.
    pub fn parse(bitcode: &[u8]) -> Result<AirModule, Refusal> {
        let ir = crate::bitcode::parse_bitcode(bitcode).map_err(|e| Refusal {
            kind: RefusalKind::NotImplemented,
            construct: "bitcode".into(),
            location: None,
            reason: e.to_string(),
            reference: None,
        })?;
        AirModule::decode(ir)
    }

    /// Decodes the AIR metadata of `ir`.
    pub fn decode(ir: Module) -> Result<AirModule, Refusal> {
        let m = &ir;
        let mut air_version = None;
        let mut language_version = None;
        let mut compile_options = CompileOptions::default();
        let mut flags = ModuleFlags::default();
        let mut ident = Vec::new();
        let mut source_file_names = Vec::new();
        let mut entry_points = Vec::new();
        let mut function_constants = Vec::new();
        let mut sampler_states = Vec::new();
        let mut visible_function_references = Vec::new();
        let mut external_initializations = Vec::new();
        let mut signatures = Vec::new();
        let mut core_image = Vec::new();
        let mut type_descriptions = desc::TypeDescriptions::default();
        for named in &m.named_metadata {
            let name = named.name.as_str();
            if let Some(stage) = Stage::named(name) {
                for &id in &named.ops {
                    entry_points.push(decode_entry(m, &mut type_descriptions, stage, id)?);
                }
                continue;
            }
            match name {
                "llvm.module.flags" => decode_flags(m, &named.ops, &mut flags)?,
                "llvm.ident" => {
                    for &id in &named.ops {
                        let mut c = Cursor::node(m, id, "llvm.ident", None)?;
                        ident.push(c.string("llvm.ident")?);
                        c.finish()?;
                    }
                }
                "air.version" => {
                    let mut c = Cursor::node(m, single(name, &named.ops)?, "air.version", None)?;
                    air_version = Some((c.i32(name)?, c.i32(name)?, c.i32(name)?));
                    c.finish()?;
                }
                "air.language_version" => {
                    let mut c = Cursor::node(m, single(name, &named.ops)?, "air.language_version", None)?;
                    let language = c.string(name)?;
                    language_version = Some(LanguageVersion {
                        language,
                        major: c.i32(name)?,
                        minor: c.i32(name)?,
                        patch: c.i32(name)?,
                    });
                    c.finish()?;
                }
                "air.source_file_name" => {
                    for &id in &named.ops {
                        let mut c = Cursor::node(m, id, "air.source_file_name", None)?;
                        source_file_names.push(String::from_utf8_lossy(c.bytes(name)?).into_owned());
                        c.finish()?;
                    }
                }
                "air.compile_options" => {
                    for &id in &named.ops {
                        let mut c = Cursor::node(m, id, "air.compile_options", None)?;
                        let key = c.string(name)?;
                        c.finish()?;
                        let o = &mut compile_options;
                        let dup = match key.as_str() {
                            "air.compile.fast_math_enable" => o.fast_math.replace(true).is_some(),
                            "air.compile.fast_math_disable" => o.fast_math.replace(false).is_some(),
                            "air.compile.denorms_disable" => std::mem::replace(&mut o.denorms_disable, true),
                            "air.compile.framebuffer_fetch_enable" => {
                                std::mem::replace(&mut o.framebuffer_fetch_enable, true) || o.framebuffer_fetch_disable
                            }
                            "air.compile.framebuffer_fetch_disable" => {
                                std::mem::replace(&mut o.framebuffer_fetch_disable, true) || o.framebuffer_fetch_enable
                            }
                            "air.compile.native_double_disable" => {
                                std::mem::replace(&mut o.native_double_disable, true)
                            }
                            _ => {
                                return Err(refusal(
                                    RefusalKind::NotImplemented,
                                    key,
                                    None,
                                    "a compile option the AIR grammar does not name",
                                ));
                            }
                        };
                        if dup {
                            return Err(refusal(
                                RefusalKind::NotImplemented,
                                key,
                                None,
                                "air.compile_options sets the option twice or both ways",
                            ));
                        }
                    }
                }
                "air.function_constants" => {
                    for &id in &named.ops {
                        let mut c = Cursor::node(m, id, format!("air.function_constants !{}", id.0), None)?;
                        let global = c.global_ref(name)?;
                        let type_name = c.string(name)?;
                        let cname = c.string(name)?;
                        let index = c.i32(name)?;
                        let flag = match c.peek() {
                            Some(Operand::Int { bits: 1, value }) => {
                                c.advance();
                                Some(value != 0)
                            }
                            _ => None,
                        };
                        c.finish()?;
                        function_constants.push(FunctionConstant { global, type_name, name: cname, index, flag });
                    }
                }
                "air.sampler_states" => {
                    for &id in &named.ops {
                        let mut c = Cursor::node(m, id, format!("air.sampler_states !{}", id.0), None)?;
                        if c.peek_str() != Some(b"air.sampler_state") {
                            return Err(c.error("air.sampler_states", "air.sampler_state"));
                        }
                        c.advance();
                        let global = c.global_ref("air.sampler_state")?;
                        c.finish()?;
                        sampler_states.push(ConstantSampler { global });
                    }
                }
                "air.visible_function_references" => {
                    for &id in &named.ops {
                        let mut c = Cursor::node(m, id, format!("air.visible_function_references !{}", id.0), None)?;
                        if c.peek_str() != Some(b"air.visible_function_reference") {
                            return Err(c.error(name, "air.visible_function_reference"));
                        }
                        c.advance();
                        let function = c.function_ref(name)?;
                        let fname = c.string(name)?;
                        c.finish()?;
                        visible_function_references.push(VisibleFunctionReference { function, name: fname });
                    }
                }
                "air.external_initializations" => {
                    for &id in &named.ops {
                        let mut c = Cursor::node(m, id, format!("air.external_initializations !{}", id.0), None)?;
                        if c.peek_str() != Some(b"air.externally_initialized") {
                            return Err(c.error(name, "air.externally_initialized"));
                        }
                        c.advance();
                        let global = c.global_ref(name)?;
                        let type_name = c.string(name)?;
                        let gname = c.string(name)?;
                        c.finish()?;
                        external_initializations.push(ExternalInitialization { global, type_name, name: gname });
                    }
                }
                "air.signature" => {
                    for &id in &named.ops {
                        signatures.push(desc::signature(m, id)?);
                    }
                }
                "air.ci" => {
                    for &id in &named.ops {
                        core_image.push(desc::core_image(m, id)?);
                    }
                }
                other => {
                    return Err(refusal(
                        RefusalKind::NotImplemented,
                        other,
                        None,
                        "named metadata the AIR grammar does not name",
                    ));
                }
            }
        }
        Ok(AirModule {
            ir,
            air_version,
            language_version,
            compile_options,
            flags,
            ident,
            source_file_names,
            entry_points,
            function_constants,
            sampler_states,
            visible_function_references,
            external_initializations,
            signatures,
            core_image,
            type_descriptions,
        })
    }

    /// The entry point for function `name`.
    pub fn entry_point(&self, name: &str) -> Option<&EntryPoint> {
        self.entry_points.iter().find(|e| e.name == name)
    }
}

/// The sampler-state word in global `g`: an `i64` constant.
fn sampler_word(m: &Module, g: u32) -> Result<u64, Refusal> {
    use crate::ir::{ConstKind, Type, Value};
    let gv = &m.globals[g as usize];
    if gv.constant
        && m.types.get(gv.value_type) == &Type::Int(64)
        && let Some(Value::Const(c)) = gv.initializer
        && let ConstKind::Int(words) = &m.constant(c).kind
    {
        return Ok(words[0]);
    }
    Err(refusal(
        RefusalKind::NotImplemented,
        "air.sampler_state",
        None,
        format!("@{} is not an i64 constant, the sampler-state form the target decodes", gv.common.name),
    ))
}

#[cfg(test)]
pub(crate) mod tests;
