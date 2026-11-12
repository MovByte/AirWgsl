//! Pipeline reflection built from AIR metadata, in the shape of the wire protocol's `Reflection` ([`crate::api`]):
//! one [`StageReflection`] per stage, its `MTLArgument`s, and the `MTLType` tree each argument refers to.
//!
//! The values are Metal's as real Metal reports them for the same AIR: every rule here is checked against
//! `MTLComputePipelineReflection` and `MTLRenderPipelineReflection` recorded on real Metal for the corpora
//! (`tools/metal-reflection/record.m`, `examples/reflect_corpus.rs`).
//!
//! Types are numbered as the wire protocol numbers them: the stage's `types` list is filled by a depth-first walk of
//! the arguments in order, each type taking the next index when the walk first reaches it and its children numbered
//! after it; for a buffer argument the walk visits `bufferStructType`, then `bufferPointerType`. A type reached twice
//! is listed twice.
//!
//! Some inputs come from outside the module's metadata: the values of function-constant predicates
//! (`air.function_constant`), which the module's static initializer computes from the constants, the values of
//! function constants that give binding indices, and the arguments the canonicalized function leaves unused.
//! [`Inputs`] carries them; a record that needs a value it does not have is refused.

// Refusals are the failure of decoding a module, made once per module; they are returned by value like
// `crate::compile` returns them.
#![allow(clippy::result_large_err)]

pub mod names;

use std::collections::{BTreeMap, BTreeSet};

use crate::air::{
    Access, AirModule, Argument, Buffer, EntryPoint, IndirectArgument, Location, Member, Role, Slot, Stage, StructInfo,
};
use crate::api::{
    ArgumentReflection, BuiltinReflection, Location as RefusalLocation, NO_TYPE, Refusal, RefusalKind,
    StageInputDescriptor, StageReflection, StructMemberReflection, TypeReflection, VertexDescriptor,
};
use names::data_type as dt;

/// `MTLArgumentType` values.
pub mod argument_type {
    pub const BUFFER: u64 = 0;
    pub const THREADGROUP_MEMORY: u64 = 1;
    pub const TEXTURE: u64 = 2;
    pub const SAMPLER: u64 = 3;
    pub const IMAGEBLOCK_DATA: u64 = 16;
    pub const IMAGEBLOCK: u64 = 17;
    pub const VISIBLE_FUNCTION_TABLE: u64 = 24;
    pub const PRIMITIVE_ACCELERATION_STRUCTURE: u64 = 25;
    pub const INSTANCE_ACCELERATION_STRUCTURE: u64 = 26;
    pub const INTERSECTION_FUNCTION_TABLE: u64 = 27;
}

/// The `MTLArgumentType` of every built-in argument (measurement reflect-private, every case).
pub const BUILTIN_ARGUMENT_TYPE: u64 = 15;

/// The AIR address spaces a pointer's reflection records (`isConstantBuffer` is the second).
pub mod address_space {
    pub const DEVICE: u32 = 1;
    pub const CONSTANT: u32 = 2;
    pub const THREADGROUP: u32 = 3;
}

/// The `MTLBuiltInArgumentType` Metal's reflection lists a built-in argument with, by AIR key: `Some(value)`, or `None`
/// for a built-in the list omits. Measured on an Apple M4 (measurement reflect-private) for every kernel and vertex
/// built-in the front end writes; the values are those of 21H461's `MTLBuiltInArgumentTypeString` table (Metal
/// 0x187b31604 indexes 25 names at 0x1e642d1a8: 0 Invalid, 1 VertexID ... 20 ThreadsPerSimdgroup). A key that is not
/// listed here is not measured.
pub fn builtin_type(key: &str) -> Option<Option<u64>> {
    Some(match key {
        "air.vertex_id" => Some(1),
        "air.instance_id" => Some(2),
        "air.base_vertex" => Some(3),
        "air.base_instance" => Some(4),
        "air.thread_position_in_grid" => Some(5),
        "air.threads_per_grid" => Some(6),
        "air.thread_position_in_threadgroup" => Some(7),
        "air.thread_index_in_threadgroup" => Some(8),
        "air.threads_per_threadgroup" => Some(9),
        "air.threadgroup_position_in_grid" => Some(10),
        "air.threadgroups_per_grid" => Some(11),
        "air.stage_in_grid_origin" => Some(15),
        "air.stage_in_grid_size" => Some(16),
        "air.thread_index_in_quadgroup" => Some(17),
        "air.thread_index_in_simdgroup" => Some(18),
        "air.threads_per_simdgroup" => Some(20),
        "air.simdgroup_index_in_threadgroup"
        | "air.simdgroups_per_threadgroup"
        | "air.dispatch_simdgroups_per_threadgroup"
        | "air.quadgroup_index_in_threadgroup"
        | "air.quadgroups_per_threadgroup"
        | "air.dispatch_quadgroups_per_threadgroup"
        | "air.dispatch_threads_per_threadgroup" => None,
        _ => return None,
    })
}

/// `MTLBindingAccess` values.
pub mod access {
    pub const READ_ONLY: u64 = 0;
    pub const READ_WRITE: u64 = 1;
    pub const WRITE_ONLY: u64 = 2;
}

/// The `bufferAlignment` real Metal reports for a vertex-descriptor or stage-input buffer.
pub const VERTEX_BUFFER_ALIGNMENT: u64 = 4;

/// The smallest `bufferAlignment` real Metal reports for a buffer in the `constant` address space.
pub const CONSTANT_BUFFER_MIN_ALIGNMENT: u64 = 4;

/// What reflection needs that the module's metadata does not hold.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Inputs {
    /// The value of each predicate global (`air.function_constant`), by global index: the module's static
    /// initializer computes them from the function constants.
    pub predicates: BTreeMap<u32, bool>,
    /// The value of each function-constant global used as a binding index or count, by global index.
    pub slots: BTreeMap<u32, u32>,
    /// Argument numbers the function does not use once specialized and canonicalized (inlined, dead code removed).
    /// Real Metal reports such arguments inactive although the front end did not mark them `air.arg_unused`: its
    /// compiler finds them unused after optimizing. Reflection reports an argument active when the front end did not
    /// mark it unused and it is not in this set.
    pub unused_arguments: BTreeSet<u32>,
}

struct Builder<'a> {
    module: &'a AirModule,
    entry: &'a EntryPoint,
    inputs: &'a Inputs,
    types: Vec<TypeReflection>,
}

fn refuse(entry: &EntryPoint, construct: &str, reason: String) -> Refusal {
    Refusal {
        kind: RefusalKind::NotImplemented,
        construct: construct.into(),
        location: Some(RefusalLocation { function: entry.name.clone(), instruction: None }),
        reason,
        reference: None,
    }
}

impl<'a> Builder<'a> {
    fn refuse(&self, construct: &str, reason: String) -> Refusal {
        refuse(self.entry, construct, reason)
    }

    fn slot(&self, s: Slot) -> Result<u64, Refusal> {
        match s {
            Slot::Value(v) => Ok(v as u64),
            Slot::Constant(g) => match self.inputs.slots.get(&g) {
                Some(v) => Ok(*v as u64),
                None => Err(self.refuse(
                    "air.location_index",
                    format!(
                        "the binding is given by function constant @{}, whose value reflection was not given",
                        self.module.ir.globals[g as usize].common.name
                    ),
                )),
            },
        }
    }

    fn present(&self, arg: &Argument) -> Result<bool, Refusal> {
        match &arg.predicate {
            None => Ok(true),
            Some(p) => match self.inputs.predicates.get(&p.global) {
                Some(v) => Ok(*v),
                None => Err(self.refuse(
                    "air.function_constant",
                    format!(
                        "argument {} exists only when {} is true; reflection was not given the predicate's value",
                        arg.name.as_deref().unwrap_or("?"),
                        p.name
                    ),
                )),
            },
        }
    }

    fn push(&mut self, t: TypeReflection) -> u32 {
        self.types.push(t);
        (self.types.len() - 1) as u32
    }

    fn type_name<'n>(&self, arg: &'n Argument) -> Result<&'n str, Refusal> {
        arg.type_name
            .as_deref()
            .ok_or_else(|| self.refuse("air.arg_type_name", format!("argument {} has no type name", arg.position)))
    }

    /// The `MTLDataType` of a data type name.
    fn data(&self, name: &str) -> Result<u64, Refusal> {
        names::data_type(name)
            .ok_or_else(|| self.refuse("air.arg_type_name", format!("the type name {} has no MTLDataType rule", name)))
    }

    /// A struct type and its members, numbered in pre-order.
    fn struct_type(&mut self, info: &StructInfo, argument_buffer: bool) -> Result<u32, Refusal> {
        let index = self.push(TypeReflection { data_type: dt::STRUCT, ..Default::default() });
        let mut members = Vec::new();
        for m in &info.members {
            members.push(self.member(m, argument_buffer)?);
        }
        self.types[index as usize].members = members;
        Ok(index)
    }

    /// The type entry a member's value has, and its `MTLDataType`. `array_length` describes an array of it.
    fn member(&mut self, m: &Member, argument_buffer: bool) -> Result<StructMemberReflection, Refusal> {
        let argument_index = match &m.indirect {
            None => 0,
            Some(IndirectArgument::Aggregate(id)) => *id as u64,
            Some(IndirectArgument::Resource(arg)) => match &arg.role {
                Role::Texture { location, .. }
                | Role::Sampler(location)
                | Role::IndirectConstant(location)
                | Role::CommandBuffer(location)
                | Role::ComputePipelineState(location)
                | Role::RenderPipelineState(location)
                | Role::PrimitiveAccelerationStructure(location, _)
                | Role::InstanceAccelerationStructure(location, _)
                | Role::Buffer(Buffer { location, .. }) => self.slot(location.index)?,
                other => {
                    return Err(self.refuse(
                        "air.indirect_argument",
                        format!("member {}: argument-buffer member role {:?} has no reflection rule", m.name, other),
                    ));
                }
            },
        };
        let (data_type, r#type) = self.member_type(m, argument_buffer)?;
        Ok(StructMemberReflection { name: m.name.clone(), offset: m.offset as u64, data_type, r#type, argument_index })
    }

    fn member_type(&mut self, m: &Member, argument_buffer: bool) -> Result<(u64, u32), Refusal> {
        // An array: a C array member (array_length) or an `array<T, N>` type name.
        if m.array_length > 0 || names::array(&m.type_name).is_some() {
            let index = self.push(TypeReflection { data_type: dt::ARRAY, ..Default::default() });
            let (length, element_name) = if m.array_length > 0 {
                (m.array_length as u64, m.type_name.as_str())
            } else {
                let (e, n) = names::array(&m.type_name).unwrap();
                (n, e)
            };
            let element = Member { array_length: 0, type_name: element_name.to_string(), ..m.clone() };
            let (element_data_type, element_type) = self.element_type(&element, argument_buffer)?;
            let t = &mut self.types[index as usize];
            t.array_length = length;
            t.element_data_type = element_data_type;
            t.element_type = element_type;
            if argument_buffer {
                // In an argument buffer every leaf member of an element takes an id: the stride in ids is their
                // count. An array of resources (or of structs holding any) has no byte stride.
                t.argument_index_stride = id_count(&element);
                t.stride = if has_resource(&element) { 0 } else { m.size as u64 };
            } else {
                t.stride = m.size as u64;
            }
            return Ok((dt::ARRAY, index));
        }
        self.element_type(m, argument_buffer)
    }

    fn element_type(&mut self, m: &Member, argument_buffer: bool) -> Result<(u64, u32), Refusal> {
        if let Some(nested) = &m.nested {
            return Ok((dt::STRUCT, self.struct_type(nested, argument_buffer)?));
        }
        if let Some(IndirectArgument::Resource(arg)) = &m.indirect {
            match &arg.role {
                Role::Texture { access, .. } => {
                    let tex = self.texture_name(&m.type_name)?;
                    let index = self.push(TypeReflection {
                        data_type: dt::TEXTURE,
                        texture_data_type: tex.data_type,
                        texture_type: tex.texture_type,
                        is_depth_texture: tex.is_depth,
                        access: texture_access(*access),
                        element_type: NO_TYPE,
                        ..Default::default()
                    });
                    return Ok((dt::TEXTURE, index));
                }
                Role::Sampler(_) => return Ok((dt::SAMPLER, NO_TYPE)),
                Role::CommandBuffer(_) => return Ok((dt::INDIRECT_COMMAND_BUFFER, NO_TYPE)),
                Role::ComputePipelineState(_) => return Ok((dt::COMPUTE_PIPELINE, NO_TYPE)),
                Role::RenderPipelineState(_) => return Ok((dt::RENDER_PIPELINE, NO_TYPE)),
                Role::PrimitiveAccelerationStructure(..) => {
                    return Ok((dt::PRIMITIVE_ACCELERATION_STRUCTURE, NO_TYPE));
                }
                Role::InstanceAccelerationStructure(..) => {
                    return Ok((dt::INSTANCE_ACCELERATION_STRUCTURE, NO_TYPE));
                }
                Role::Buffer(b) => {
                    let element_data_type =
                        if b.struct_info.is_some() { dt::STRUCT } else { self.pointee_data_type(&m.type_name, arg)? };
                    let mut alignment = arg.type_align.unwrap_or(0) as u64;
                    if b.address_space == Some(2) {
                        alignment = alignment.max(CONSTANT_BUFFER_MIN_ALIGNMENT);
                    }
                    let index = self.push(TypeReflection {
                        data_type: dt::POINTER,
                        element_data_type,
                        element_type: NO_TYPE,
                        access: buffer_access(b.access),
                        alignment,
                        data_size: arg.type_size.unwrap_or(0) as u64,
                        // Measured for device and constant pointers in an argument buffer (reflect-private, bindings).
                        address_space: b.address_space.unwrap_or(0),
                        ..Default::default()
                    });
                    if let Some(info) = &b.struct_info {
                        // The pointee of an argument-buffer pointer is a plain struct: its members take no ids.
                        let s = self.struct_type(info, false)?;
                        self.types[index as usize].element_type = s;
                    }
                    return Ok((dt::POINTER, index));
                }
                Role::IndirectConstant(_) => {}
                other => {
                    return Err(self.refuse(
                        "air.indirect_argument",
                        format!("member {}: argument-buffer member role {:?} has no reflection rule", m.name, other),
                    ));
                }
            }
        }
        Ok((self.data(&m.type_name)?, NO_TYPE))
    }

    /// The data type of a pointer's pointee. A pointee the front end gives no size for and whose name is not a data
    /// type (`void`, `rasterization_rate_map_data`) reflects as `MTLDataTypeNone`, with size 0 and alignment 0 before
    /// the `constant` minimum (measured on real Metal for both names, as arguments and as argument-buffer members).
    fn pointee_data_type(&self, name: &str, arg: &Argument) -> Result<u64, Refusal> {
        if arg.type_size.is_none() && names::data_type(name).is_none() {
            return Ok(dt::NONE);
        }
        self.data(name)
    }

    fn texture_name(&self, name: &str) -> Result<names::TextureName, Refusal> {
        names::texture(name)
            .ok_or_else(|| self.refuse("air.arg_type_name", format!("{} is not a texture type name", name)))
    }

    fn argument(&mut self, arg: &Argument) -> Result<Option<ArgumentReflection>, Refusal> {
        if !self.present(arg)? {
            return Ok(None);
        }
        let name = arg.name.clone().unwrap_or_default();
        let active = !arg.unused && !self.inputs.unused_arguments.contains(&arg.position);
        let base =
            |r#type: u64, access: u64, location: &Location, b: &Builder| -> Result<ArgumentReflection, Refusal> {
                Ok(ArgumentReflection {
                    name: name.clone(),
                    r#type,
                    access,
                    index: b.slot(location.index)?,
                    active,
                    array_length: b.slot(location.count)?,
                    buffer_struct_type: NO_TYPE,
                    buffer_pointer_type: NO_TYPE,
                    buffer_indirect_argument_type: NO_TYPE,
                    ..Default::default()
                })
            };
        let r = match &arg.role {
            Role::Buffer(b) | Role::IndirectBuffer(b) => {
                let indirect = matches!(arg.role, Role::IndirectBuffer(_));
                let threadgroup = b.address_space == Some(3);
                let mut r = base(
                    if threadgroup { argument_type::THREADGROUP_MEMORY } else { argument_type::BUFFER },
                    buffer_access(b.access),
                    &b.location,
                    self,
                )?;
                let size = arg.type_size.unwrap_or(0) as u64;
                let mut align = arg.type_align.unwrap_or(0) as u64;
                if b.address_space == Some(2) {
                    // Real Metal reports at least 4-byte alignment for a `constant` buffer (measured: `constant
                    // uchar *` and `constant ushort *` reflect alignment 4, `device` ones 1 and 2).
                    align = align.max(CONSTANT_BUFFER_MIN_ALIGNMENT);
                }
                if threadgroup {
                    r.threadgroup_memory_alignment = align;
                    r.threadgroup_memory_data_size = size;
                    // The pointee's data type, a struct (atomics included: the front end gives them struct type
                    // information) as MTLDataTypeStruct (measurement reflect-private, case bindings).
                    r.threadgroup_memory_data_type = if b.struct_info.is_some() {
                        dt::STRUCT
                    } else {
                        self.pointee_data_type(self.type_name(arg)?, arg)?
                    };
                    return Ok(Some(r));
                }
                r.buffer_alignment = align;
                r.buffer_data_size = size;
                r.buffer_address_space = b.address_space.unwrap_or(0);
                let element_data_type;
                let mut element_struct = None;
                if let Some(info) = &b.struct_info {
                    element_data_type = dt::STRUCT;
                    r.buffer_struct_type = self.struct_type(info, indirect)?;
                    element_struct = Some(info);
                } else {
                    element_data_type = self.pointee_data_type(self.type_name(arg)?, arg)?;
                }
                r.buffer_data_type = element_data_type;
                let pointer = self.push(TypeReflection {
                    data_type: dt::POINTER,
                    element_data_type,
                    access: r.access,
                    alignment: align,
                    data_size: size,
                    element_is_argument_buffer: indirect,
                    element_type: NO_TYPE,
                    address_space: r.buffer_address_space,
                    ..Default::default()
                });
                if let Some(info) = element_struct {
                    let s = self.struct_type(info, indirect)?;
                    self.types[pointer as usize].element_type = s;
                }
                r.buffer_pointer_type = pointer;
                r
            }
            Role::Texture { location, access, .. } => {
                let tex =
                    self.texture_name(names::array(self.type_name(arg)?).map_or(self.type_name(arg)?, |a| a.0))?;
                let mut r = base(argument_type::TEXTURE, texture_access(*access), location, self)?;
                r.texture_type = tex.texture_type;
                r.texture_data_type = tex.data_type;
                r.is_depth_texture = tex.is_depth;
                r
            }
            Role::Sampler(location) => base(argument_type::SAMPLER, access::READ_ONLY, location, self)?,
            // Acceleration structures and function tables reflect read-only whatever access the record states
            // (measured: an `intersection_function_table` recorded `air.read_write` reflects access 0).
            Role::InstanceAccelerationStructure(location, _) => {
                base(argument_type::INSTANCE_ACCELERATION_STRUCTURE, access::READ_ONLY, location, self)?
            }
            Role::PrimitiveAccelerationStructure(location, _) => {
                base(argument_type::PRIMITIVE_ACCELERATION_STRUCTURE, access::READ_ONLY, location, self)?
            }
            Role::IntersectionFunctionTable(location, _) => {
                base(argument_type::INTERSECTION_FUNCTION_TABLE, access::READ_ONLY, location, self)?
            }
            Role::VisibleFunctionTable(location, _) => {
                base(argument_type::VISIBLE_FUNCTION_TABLE, access::READ_ONLY, location, self)?
            }
            Role::Imageblock(ib) if ib.layout == "explicit" => {
                // An explicit imageblock argument has no binding index in AIR; real Metal reflects index 0 and
                // array length 1.
                let location = ib.location.unwrap_or(Location { index: Slot::Value(0), count: Slot::Value(1) });
                base(argument_type::IMAGEBLOCK, access::READ_ONLY, &location, self)?
            }
            Role::Builtin { .. }
            | Role::VertexInput(_)
            | Role::PatchInput(_)
            | Role::PatchControlPointInput(_)
            | Role::StageIn(_)
            | Role::FragmentInput { .. }
            | Role::RenderTarget { .. } => return Ok(None),
            other => {
                return Err(self.refuse(
                    "argument",
                    format!("argument {}: role {:?} has no reflection rule", arg.position, role_name(other)),
                ));
            }
        };
        Ok(Some(r))
    }
}

fn role_name(r: &Role) -> &'static str {
    match r {
        Role::Imageblock(_) => "air.imageblock",
        Role::InstanceAccelerationStructure(..) => "air.instance_acceleration_structure",
        Role::PrimitiveAccelerationStructure(..) => "air.primitive_acceleration_structure",
        Role::IntersectionFunctionTable(..) => "air.intersection_function_table",
        Role::VisibleFunctionTable(..) => "air.visible_function_table",
        Role::CommandBuffer(_) => "air.command_buffer",
        Role::ComputePipelineState(_) => "air.compute_pipeline_state",
        Role::RenderPipelineState(_) => "air.render_pipeline_state",
        Role::Payload(_) => "air.payload",
        Role::Mesh(_) => "air.mesh",
        Role::MeshGridProperties => "air.mesh_grid_properties",
        Role::VisibleInput => "air.visible_input",
        Role::IndirectConstant(_) => "air.indirect_constant",
        _ => "argument",
    }
}

/// The argument-buffer ids one element of `m` takes: one per leaf member.
fn id_count(m: &Member) -> u64 {
    match &m.nested {
        Some(nested) => nested.members.iter().map(|x| id_count(x) * elements(x)).sum(),
        None => 1,
    }
}

/// Whether `m` is, or holds, an argument-buffer resource (a texture, sampler, buffer pointer or other object).
fn has_resource(m: &Member) -> bool {
    if let Some(nested) = &m.nested {
        return nested.members.iter().any(has_resource);
    }
    matches!(&m.indirect, Some(IndirectArgument::Resource(arg)) if !matches!(arg.role, Role::IndirectConstant(_)))
}

/// The number of elements of a member: its array length, 1 when it is not an array.
fn elements(m: &Member) -> u64 {
    if m.array_length > 0 {
        m.array_length as u64
    } else if let Some((_, n)) = names::array(&m.type_name) {
        n
    } else {
        1
    }
}

fn buffer_access(a: Access) -> u64 {
    match a {
        Access::Read | Access::Sample => access::READ_ONLY,
        Access::ReadWrite => access::READ_WRITE,
        Access::Write => access::WRITE_ONLY,
    }
}

fn texture_access(a: Access) -> u64 {
    buffer_access(a)
}

/// `MTLVertexFormatInvalid` (= `MTLAttributeFormatInvalid`).
const VERTEX_FORMAT_INVALID: u64 = 0;
/// `MTLStepFunctionThreadPositionInGridXIndexed` and `...YIndexed`: the step functions that read the index buffer.
const STEP_GRID_X_INDEXED: u64 = 7;
const STEP_GRID_Y_INDEXED: u64 = 8;

/// A stage-input descriptor as reflection reads it: the argument name prefix, the stride per buffer index, and the
/// (format, buffer index) per attribute index.
type Described<'a> = (&'a str, Vec<u64>, Vec<(u64, u64)>);

/// The stage-input descriptor of the pipeline, when it has one.
#[derive(Clone, Copy, Debug)]
pub enum StageInput<'a> {
    None,
    /// A render pipeline's `MTLVertexDescriptor` (vertex functions).
    Vertex(&'a VertexDescriptor),
    /// A compute pipeline's `MTLStageInputOutputDescriptor` (kernels).
    Compute(&'a StageInputDescriptor),
}

/// The reflection of one entry point: its argument-table arguments and their types.
///
/// With a stage-input descriptor, real Metal lists before the function's own arguments one buffer argument per
/// buffer index the descriptor's attributes read (an attribute with format 0, invalid, reads none), in increasing
/// index order, named `vertexBuffer.<n>` (vertex descriptor) or `stageInBuffer.<n>` (compute stage input) by that
/// order. Each is a read-only buffer of a struct whose size is the layout's stride, 4-byte aligned, with no struct
/// type; it is active when the function reads an attribute the buffer feeds, except in a post-tessellation vertex
/// function, where it is reported inactive. The function's own arguments follow in the order the function declares
/// them.
pub fn stage(
    module: &AirModule,
    entry: &EntryPoint,
    inputs: &Inputs,
    input: StageInput<'_>,
) -> Result<StageReflection, Refusal> {
    let mut b = Builder { module, entry, inputs, types: Vec::new() };
    let mut arguments = Vec::new();
    let described: Option<Described> = match input {
        StageInput::None => None,
        StageInput::Vertex(vd) => Some((
            "vertexBuffer",
            vd.layouts.iter().map(|l| l.stride).collect(),
            vd.attributes.iter().map(|a| (a.format, a.buffer_index)).collect(),
        )),
        StageInput::Compute(sd) => {
            if sd.layouts.iter().any(|l| matches!(l.1, STEP_GRID_X_INDEXED | STEP_GRID_Y_INDEXED)) {
                return Err(b.refuse(
                    "MTLStageInputOutputDescriptor",
                    "an indexed step function reads an index buffer, whose reflection is not measured".into(),
                ));
            }
            Some((
                "stageInBuffer",
                sd.layouts.iter().map(|l| l.0).collect(),
                sd.attributes.iter().map(|a| (a.0, a.2)).collect(),
            ))
        }
    };
    if let Some((prefix, strides, attributes)) = described {
        let mut buffers: Vec<u64> = attributes.iter().filter(|a| a.0 != VERTEX_FORMAT_INVALID).map(|a| a.1).collect();
        buffers.sort_unstable();
        buffers.dedup();
        let mut read = Vec::new();
        for arg in &entry.arguments {
            match &arg.role {
                Role::VertexInput(location) | Role::StageIn(Some(location)) | Role::PatchInput(location) => {
                    read.push(b.slot(location.index)?)
                }
                Role::PatchControlPointInput(cp) => {
                    for a in &cp.attributes {
                        read.push(b.slot(a.location.index)?);
                    }
                }
                _ => {}
            }
        }
        for (n, &buffer) in buffers.iter().enumerate() {
            let stride = strides.get(buffer as usize).copied().unwrap_or(0);
            // Real Metal reports a post-tessellation vertex function's buffers inactive, though the function reads
            // the attributes they feed (measured over every post-tessellation function of the generated corpus).
            let active = entry.patch.is_none()
                && attributes
                    .iter()
                    .enumerate()
                    .any(|(i, a)| a.0 != VERTEX_FORMAT_INVALID && a.1 == buffer && read.contains(&(i as u64)));
            // The described buffers are device buffers (isConstantBuffer 0, measurement reflect-private, cases
            // vertex-descriptor and stage-input-descriptor).
            let pointer = b.push(TypeReflection {
                data_type: dt::POINTER,
                element_data_type: dt::STRUCT,
                access: access::READ_ONLY,
                alignment: VERTEX_BUFFER_ALIGNMENT,
                data_size: stride,
                element_type: NO_TYPE,
                address_space: address_space::DEVICE,
                ..Default::default()
            });
            arguments.push(ArgumentReflection {
                name: format!("{}.{}", prefix, n),
                r#type: argument_type::BUFFER,
                access: access::READ_ONLY,
                index: buffer,
                active,
                array_length: 1,
                buffer_alignment: VERTEX_BUFFER_ALIGNMENT,
                buffer_data_size: stride,
                buffer_data_type: dt::STRUCT,
                buffer_struct_type: NO_TYPE,
                buffer_pointer_type: pointer,
                buffer_indirect_argument_type: NO_TYPE,
                buffer_address_space: address_space::DEVICE,
                ..Default::default()
            });
        }
    }
    for arg in &entry.arguments {
        if let Some(r) = b.argument(arg)? {
            arguments.push(r);
        }
    }
    let builtins = b.builtins()?;
    Ok(StageReflection { function_type: entry.stage.function_type(), arguments, types: b.types, builtins })
}

impl Builder<'_> {
    /// The built-in arguments of a kernel or a vertex function, as Metal's reflection lists them (measurement
    /// reflect-private): in declaration order, each with type 15, access 0 (read only) and index 0, active when the
    /// function reads it, and the argument type's data type; built-ins without an `MTLBuiltInArgumentType` are left
    /// out. `None` for other stages (Metal keeps no list of a fragment function's built-ins) and for a function with a
    /// built-in that is not measured.
    fn builtins(&self) -> Result<Option<Vec<BuiltinReflection>>, Refusal> {
        if !matches!(self.entry.stage, Stage::Kernel | Stage::Vertex) {
            return Ok(None);
        }
        let mut out = Vec::new();
        for arg in &self.entry.arguments {
            let Role::Builtin { key, .. } = &arg.role else { continue };
            let Some(listed) = builtin_type(key) else { return Ok(None) };
            let Some(builtin_type) = listed else { continue };
            out.push(BuiltinReflection {
                name: arg.name.clone().unwrap_or_default(),
                r#type: BUILTIN_ARGUMENT_TYPE,
                access: access::READ_ONLY,
                active: !arg.unused && !self.inputs.unused_arguments.contains(&arg.position),
                index: 0,
                data_type: self.data(self.type_name(arg)?)?,
                builtin_type,
            });
        }
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel of the AIR layer's tests: `constant ushort *a [[buffer(2)]]`, `texture2d<half> t [[texture(3)]]`,
    /// `sampler s [[sampler(1)]]` and a builtin.
    #[test]
    fn reflects_a_kernel() {
        let a = AirModule::decode(crate::air::tests::valid()).unwrap();
        let r = stage(&a, a.entry_point("k").unwrap(), &Inputs::default(), StageInput::None).unwrap();
        assert_eq!(r.function_type, 3);
        assert_eq!(r.arguments.len(), 3, "the builtin is not an argument-table argument");
        let buf = &r.arguments[0];
        assert_eq!((buf.name.as_str(), buf.r#type, buf.index, buf.access), ("a", 0, 2, 0));
        // A `constant` buffer reflects at least 4-byte alignment.
        assert_eq!((buf.buffer_alignment, buf.buffer_data_size, buf.buffer_data_type), (4, 2, 41));
        let p = &r.types[buf.buffer_pointer_type as usize];
        assert_eq!((p.data_type, p.element_data_type, p.alignment, p.data_size), (60, 41, 4, 2));
        let tex = &r.arguments[1];
        assert_eq!((tex.r#type, tex.index, tex.texture_type, tex.texture_data_type), (2, 3, 2, 16));
        let s = &r.arguments[2];
        assert_eq!((s.r#type, s.index), (3, 1));
        // The constant buffer's address space, on the binding and its pointer type.
        assert_eq!((buf.buffer_address_space, p.address_space), (2, 2));
        // `uint i [[thread_position_in_grid]]`: MTLBuiltInArgumentThreadPositionInGrid, MTLDataTypeUInt.
        let builtin = BuiltinReflection {
            name: "i".into(),
            r#type: 15,
            access: 0,
            active: true,
            index: 0,
            data_type: 33,
            builtin_type: 5,
        };
        assert_eq!(r.builtins, Some(vec![builtin]));
    }

    /// Threadgroup memory reflects its element's data type; built-ins without an `MTLBuiltInArgumentType` are left
    /// out, an unused one is inactive, and a function with a built-in that is not measured reports none.
    #[test]
    fn builtins_and_threadgroup_memory() {
        let tg = [
            "#0",
            "air.buffer",
            "air.location_index",
            "#0",
            "#1",
            "air.read_write",
            "air.address_space",
            "#3",
            "air.arg_type_size",
            "#16",
            "air.arg_type_align_size",
            "#16",
            "air.arg_type_name",
            "float4",
            "air.arg_name",
            "t",
        ];
        let lid = ["#1", "air.thread_index_in_threadgroup", "air.arg_type_name", "ushort", "air.arg_name", "l"];
        let sg = ["#2", "air.simdgroup_index_in_threadgroup", "air.arg_type_name", "uint", "air.arg_name", "s"];
        let records = vec![tg.to_vec(), lid.to_vec(), sg.to_vec()];
        let a = AirModule::decode(crate::air::tests::entry("air.kernel", &[], &records)).unwrap();
        let inputs = Inputs { unused_arguments: [1].into_iter().collect(), ..Default::default() };
        let r = stage(&a, a.entry_point("k").unwrap(), &inputs, StageInput::None).unwrap();
        let t = &r.arguments[0];
        assert_eq!((t.r#type, t.threadgroup_memory_data_type, t.buffer_address_space), (1, 6, 0));
        let builtins = r.builtins.unwrap();
        assert_eq!(builtins.len(), 1, "{builtins:?}");
        assert_eq!((builtins[0].name.as_str(), builtins[0].active), ("l", false));
        assert_eq!((builtins[0].builtin_type, builtins[0].data_type), (8, 41));

        let patch = ["#0", "air.patch_id", "air.arg_type_name", "uint", "air.arg_name", "p"];
        let a = AirModule::decode(crate::air::tests::entry("air.kernel", &[], &[patch.to_vec()])).unwrap();
        let r = stage(&a, a.entry_point("k").unwrap(), &Inputs::default(), StageInput::None).unwrap();
        assert_eq!(r.builtins, None);
    }

    /// The vertex-descriptor buffers of a measured case: `vertex float4 v1(In in [[stage_in]], constant float *u
    /// [[buffer(3)]], device const float *w [[buffer(0)]])` with attributes 0 and 2 in buffer 5 (stride 32),
    /// attribute 1 in buffer 2 (stride 8) and an attribute 5 the function does not read in buffer 7 (stride 12). Real
    /// Metal (Apple M4, macOS 27.0.1) reflects `vertexBuffer.0` (index 2, size 8, active), `vertexBuffer.1` (index 5,
    /// size 32, active), `vertexBuffer.2` (index 7, size 12, inactive), then `u` and `w` in declaration order.
    #[test]
    fn vertex_descriptor_buffers_come_first_in_buffer_order() {
        use crate::api::{VertexAttribute, VertexBufferLayout};
        let rec = |v: &[&'static str]| v.to_vec();
        let m = crate::air::tests::entry(
            "air.vertex",
            &[rec(&["air.position", "air.arg_type_name", "float4"])],
            &[
                rec(&["#0", "air.vertex_input", "air.location_index", "#0", "#1", "air.arg_type_name", "float4"]),
                rec(&["#1", "air.vertex_input", "air.location_index", "#1", "#1", "air.arg_type_name", "uchar2"]),
                rec(&["#2", "air.vertex_input", "air.location_index", "#2", "#1", "air.arg_type_name", "half"]),
                rec(&[
                    "#3",
                    "air.buffer",
                    "air.location_index",
                    "#3",
                    "#1",
                    "air.read",
                    "air.address_space",
                    "#2",
                    "air.arg_type_size",
                    "#4",
                    "air.arg_type_align_size",
                    "#4",
                    "air.arg_type_name",
                    "float",
                    "air.arg_name",
                    "u",
                ]),
                rec(&[
                    "#4",
                    "air.buffer",
                    "air.location_index",
                    "#0",
                    "#1",
                    "air.read",
                    "air.address_space",
                    "#1",
                    "air.arg_type_size",
                    "#4",
                    "air.arg_type_align_size",
                    "#4",
                    "air.arg_type_name",
                    "float",
                    "air.arg_name",
                    "w",
                ]),
            ],
        );
        let a = AirModule::decode(m).unwrap();
        let layout = |stride| VertexBufferLayout { stride, step_function: 1, step_rate: 1 };
        let attr = |format, buffer_index| VertexAttribute { format, offset: 0, buffer_index };
        let mut vd = VertexDescriptor { layouts: vec![layout(0); 8], attributes: vec![attr(0, 0); 6] };
        vd.layouts[5] = layout(32);
        vd.layouts[2] = VertexBufferLayout { stride: 8, step_function: 2, step_rate: 1 };
        vd.layouts[7] = layout(12);
        vd.attributes[0] = attr(31, 5);
        vd.attributes[1] = attr(1, 2);
        vd.attributes[2] = attr(53, 5);
        vd.attributes[5] = attr(28, 7);
        let r = stage(&a, a.entry_point("k").unwrap(), &Inputs::default(), StageInput::Vertex(&vd)).unwrap();
        let got: Vec<(&str, u64, u64, bool)> =
            r.arguments.iter().map(|x| (x.name.as_str(), x.index, x.buffer_data_size, x.active)).collect();
        assert_eq!(
            got,
            [
                ("vertexBuffer.0", 2, 8, true),
                ("vertexBuffer.1", 5, 32, true),
                ("vertexBuffer.2", 7, 12, false),
                ("u", 3, 4, true),
                ("w", 0, 4, true),
            ]
        );
        let p = &r.types[r.arguments[2].buffer_pointer_type as usize];
        assert_eq!((p.element_data_type, p.alignment, p.data_size, p.element_type), (1, 4, 12, NO_TYPE));
    }
}
