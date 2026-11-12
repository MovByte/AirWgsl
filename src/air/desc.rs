//! The records AIR nests below entry points and module tables: type descriptions and stitching records of
//! stitchable visible functions (`air.stitching_info`), linked-constant signatures (`air.signature`), CoreImage
//! kernel descriptions (`air.ci`), mesh output descriptions (`air.mesh`) and patch control-point inputs
//! (`air.patch_control_point_input`).
//!
//! Like the argument records, these are decoded completely: every tuple's head key and operand sequence must be one
//! the grammar below names, as measured on the system and generated corpora, or the module is refused naming it.
//!
//! Type descriptions form a graph (a struct's fields refer to other descriptions, a pointer to its pointee); they are
//! decoded once each into [`TypeDescriptions`] and referred to by index.

use std::collections::HashMap;

use super::md::{Cursor, Operand, refusal};
use super::{Location, Output, Predicate, Slot, StructInfo};
use crate::api::{Refusal, RefusalKind};
use crate::ir::{MdId, Module, TypeId, Value};

/// Index of a type description in [`TypeDescriptions::types`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DescId(pub u32);

/// The type descriptions of a module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TypeDescriptions {
    pub types: Vec<TypeDesc>,
    by_node: HashMap<MdId, DescId>,
}

/// An attribute of a type description (the tuple in its fourth operand).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeAttribute {
    /// `air.address_space`.
    AddressSpace(u32),
    /// `air.render_target_attr`.
    RenderTarget(u32),
}

/// One type description. Every one starts with its size and alignment in bytes and its attributes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeDesc {
    pub size: u32,
    pub align: u32,
    pub attributes: Vec<TypeAttribute>,
    pub kind: TypeDescKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeDescKind {
    /// `air.void_type`, `air.bool_type`, `air.char_type`, `air.uchar_type`, `air.short_type`, `air.ushort_type`,
    /// `air.int_type`, `air.uint_type`, `air.long_type`, `air.ulong_type`, `air.half_type`, `air.float_type`,
    /// `air.sampler_type`: the key.
    Scalar(&'static str),
    /// `air.vector_type`: element and count.
    Vector { element: DescId, count: u32 },
    /// `air.matrix_type`: the column description and the two counts as recorded (columns, rows).
    Matrix { column: DescId, columns: u32, rows: u32 },
    /// `air.struct_type`.
    Struct { name: String, fields: Vec<Field> },
    /// `air.array_type`.
    Array { element: DescId, count: u32 },
    /// `air.array_ref_of_type`: an `array_ref<T>`.
    ArrayRef { element: DescId },
    /// `air.pointer_type`.
    Pointer { pointee: DescId },
    /// `air.lvalue_reference_type`.
    LvalueReference { referent: DescId },
    /// `air.texture_1d_type`, `air.texture_2d_type`, `air.texture_2d_array_type`, `air.texture_3d_type`: the key,
    /// the component type and the access (`sample`, `read`, `write`...).
    Texture { key: &'static str, component: DescId, access: String },
    /// `air.imageblock_type`: the layout string and the element description.
    Imageblock { layout: String, element: DescId },
}

/// `air.record_field`: one field of a struct description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub offset: u32,
    /// The field record's second `i32`. In every corpus record it is a power of two dividing the offset (at most
    /// 16); the front end's rule for it is not established.
    pub offset_align: u32,
    pub ty: DescId,
    pub name: String,
    pub attributes: Vec<TypeAttribute>,
}

const SCALARS: &[&str] = &[
    "air.void_type",
    "air.bool_type",
    "air.char_type",
    "air.uchar_type",
    "air.short_type",
    "air.ushort_type",
    "air.int_type",
    "air.uint_type",
    "air.long_type",
    "air.ulong_type",
    "air.half_type",
    "air.float_type",
    "air.sampler_type",
];

const TEXTURES: &[&str] =
    &["air.texture_1d_type", "air.texture_2d_type", "air.texture_2d_array_type", "air.texture_3d_type"];

/// The head keys of nested records this module decodes (for the grammar's check against the target's names).
pub const KEYS: &[&str] = &[
    "air.void_type",
    "air.bool_type",
    "air.char_type",
    "air.uchar_type",
    "air.short_type",
    "air.ushort_type",
    "air.int_type",
    "air.uint_type",
    "air.long_type",
    "air.ulong_type",
    "air.half_type",
    "air.float_type",
    "air.sampler_type",
    "air.texture_1d_type",
    "air.texture_2d_type",
    "air.texture_2d_array_type",
    "air.texture_3d_type",
    "air.vector_type",
    "air.matrix_type",
    "air.struct_type",
    "air.record_field",
    "air.array_type",
    "air.array_ref_of_type",
    "air.pointer_type",
    "air.lvalue_reference_type",
    "air.imageblock_type",
    "air.address_space",
    "air.render_target_attr",
    "air.stitching_info",
    "air.stitching_type",
    "air.stitching_argument",
    "air.linked_constants_signature",
    "air.module_signature",
    "air.ci_builtin",
    "air.ci_texture",
    "air.ci_sampler",
    "air.ci_pointer",
    "air.ci_matrix",
    "air.ci_imageblock",
    "air.location_count",
    "air.mesh_type_info",
    "air.mesh_vertex_data",
    "air.mesh_primitive_data",
    "air.triangle",
    "air.patch_control_point_function",
];

fn head<'m>(c: &Cursor<'m>) -> Option<&'m [u8]> {
    c.peek_str()
}

fn expect_head(c: &mut Cursor<'_>, key: &str) -> Result<(), Refusal> {
    if head(c) != Some(key.as_bytes()) {
        return Err(c.error(key, key));
    }
    c.advance();
    Ok(())
}

impl TypeDescriptions {
    /// The description in tuple `id`, decoding it (and what it refers to) the first time.
    pub fn decode(&mut self, m: &Module, id: MdId, function: Option<&str>) -> Result<DescId, Refusal> {
        if let Some(&d) = self.by_node.get(&id) {
            return Ok(d);
        }
        if self.by_node.len() > 1 << 20 {
            return Err(refusal(
                RefusalKind::NotImplemented,
                "type description",
                function,
                "too many type descriptions",
            ));
        }
        let index = DescId(self.types.len() as u32);
        // A placeholder breaks cycles: a struct whose field points back at it refers to its own index.
        self.types.push(TypeDesc { size: 0, align: 0, attributes: Vec::new(), kind: TypeDescKind::Scalar("") });
        self.by_node.insert(id, index);
        let mut c = Cursor::node(m, id, format!("type description !{}", id.0), function)?;
        let key: &'static str = match head(&c).and_then(|h| KEYS.iter().find(|k| k.as_bytes() == h)) {
            Some(k) => k,
            None => return Err(c.error("type description", "an air.*_type key")),
        };
        c.advance();
        let size = c.i32(key)?;
        let align = c.i32(key)?;
        let attributes = attributes(m, &mut c, function)?;
        let kind = if SCALARS.contains(&key) {
            TypeDescKind::Scalar(key)
        } else if TEXTURES.contains(&key) {
            let component = self.decode(m, c.node_id(key)?, function)?;
            let access = c.string(key)?;
            TypeDescKind::Texture { key, component, access }
        } else {
            match key {
                "air.vector_type" => {
                    let element = self.decode(m, c.node_id(key)?, function)?;
                    TypeDescKind::Vector { element, count: c.i32(key)? }
                }
                "air.matrix_type" => {
                    let column = self.decode(m, c.node_id(key)?, function)?;
                    TypeDescKind::Matrix { column, columns: c.i32(key)?, rows: c.i32(key)? }
                }
                "air.struct_type" => {
                    let name = c.string(key)?;
                    let list = c.node_id(key)?;
                    let mut l = Cursor::node(m, list, format!("air.struct_type fields !{}", list.0), function)?;
                    let mut fields = Vec::new();
                    while !l.at_end() {
                        let f = l.node_id("air.record_field")?;
                        fields.push(self.field(m, f, function)?);
                    }
                    TypeDescKind::Struct { name, fields }
                }
                "air.array_type" => {
                    let element = self.decode(m, c.node_id(key)?, function)?;
                    TypeDescKind::Array { element, count: c.i32(key)? }
                }
                "air.array_ref_of_type" => {
                    TypeDescKind::ArrayRef { element: self.decode(m, c.node_id(key)?, function)? }
                }
                "air.pointer_type" => TypeDescKind::Pointer { pointee: self.decode(m, c.node_id(key)?, function)? },
                "air.lvalue_reference_type" => {
                    TypeDescKind::LvalueReference { referent: self.decode(m, c.node_id(key)?, function)? }
                }
                "air.imageblock_type" => {
                    let layout = c.string(key)?;
                    TypeDescKind::Imageblock { layout, element: self.decode(m, c.node_id(key)?, function)? }
                }
                _ => return Err(c.error(key, "a type description key")),
            }
        };
        c.finish()?;
        self.types[index.0 as usize] = TypeDesc { size, align, attributes, kind };
        Ok(index)
    }

    fn field(&mut self, m: &Module, id: MdId, function: Option<&str>) -> Result<Field, Refusal> {
        let mut c = Cursor::node(m, id, format!("air.record_field !{}", id.0), function)?;
        expect_head(&mut c, "air.record_field")?;
        let offset = c.i32("air.record_field")?;
        let offset_align = c.i32("air.record_field")?;
        let ty = self.decode(m, c.node_id("air.record_field")?, function)?;
        let name = c.string("air.record_field")?;
        let attributes = attributes(m, &mut c, function)?;
        c.finish()?;
        Ok(Field { offset, offset_align, ty, name, attributes })
    }
}

/// The attribute operand of a type description or field: null, or a tuple of attribute records.
fn attributes(m: &Module, c: &mut Cursor<'_>, function: Option<&str>) -> Result<Vec<TypeAttribute>, Refusal> {
    match c.advance() {
        Some(Operand::Null) => Ok(Vec::new()),
        Some(Operand::Node(list)) => {
            let mut l = Cursor::node(m, list, format!("type attributes !{}", list.0), function)?;
            let mut out = Vec::new();
            while !l.at_end() {
                let a = l.node_id("type attribute")?;
                let mut r = Cursor::node(m, a, format!("type attribute !{}", a.0), function)?;
                let attr = match head(&r) {
                    Some(b"air.address_space") => {
                        r.advance();
                        TypeAttribute::AddressSpace(r.i32("air.address_space")?)
                    }
                    Some(b"air.render_target_attr") => {
                        r.advance();
                        TypeAttribute::RenderTarget(r.i32("air.render_target_attr")?)
                    }
                    _ => return Err(r.error("type attribute", "air.address_space or air.render_target_attr")),
                };
                r.finish()?;
                out.push(attr);
            }
            Ok(out)
        }
        _ => Err(refusal(
            RefusalKind::NotImplemented,
            "type attributes",
            function,
            format!("{}: no attribute operand", c.what),
        )),
    }
}

// Stitching ---------------------------------------------------------------------------------------------------------

/// `air.stitching_type`: how a value of a type crosses a stitching boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StitchingType {
    pub ty: DescId,
    /// The functions that load an argument, store a return value, destroy a value and copy one from a buffer
    /// (the last absent for some types), by index into [`Module::functions`].
    pub load: u32,
    pub store: u32,
    pub destroy: u32,
    pub copy_from_buffer: Option<u32>,
}

/// `air.stitching_argument`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StitchingArgument {
    pub ty: StitchingType,
    pub name: Option<String>,
}

/// `air.stitching_info` of a stitchable visible function: its result's stitching type (`None` for void) and its
/// arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StitchingInfo {
    pub result: Option<StitchingType>,
    pub arguments: Vec<StitchingArgument>,
}

fn stitching_type(
    m: &Module,
    descs: &mut TypeDescriptions,
    id: MdId,
    function: Option<&str>,
) -> Result<StitchingType, Refusal> {
    let mut c = Cursor::node(m, id, format!("air.stitching_type !{}", id.0), function)?;
    expect_head(&mut c, "air.stitching_type")?;
    let ty = descs.decode(m, c.node_id("air.stitching_type")?, function)?;
    let load = c.function_ref("air.stitching_type")?;
    let store = c.function_ref("air.stitching_type")?;
    let destroy = c.function_ref("air.stitching_type")?;
    let copy_from_buffer = match c.peek() {
        Some(Operand::Null) => {
            c.advance();
            None
        }
        _ => Some(c.function_ref("air.stitching_type")?),
    };
    c.finish()?;
    Ok(StitchingType { ty, load, store, destroy, copy_from_buffer })
}

pub(super) fn stitching_info(
    m: &Module,
    descs: &mut TypeDescriptions,
    id: MdId,
    function: Option<&str>,
) -> Result<StitchingInfo, Refusal> {
    let mut c = Cursor::node(m, id, format!("air.stitching_info !{}", id.0), function)?;
    expect_head(&mut c, "air.stitching_info")?;
    let result = match c.peek() {
        Some(Operand::Null) => {
            c.advance();
            None
        }
        _ => Some(stitching_type(m, descs, c.node_id("air.stitching_info")?, function)?),
    };
    let mut arguments = Vec::new();
    while !c.at_end() {
        let a = c.node_id("air.stitching_info")?;
        let mut r = Cursor::node(m, a, format!("air.stitching_argument !{}", a.0), function)?;
        expect_head(&mut r, "air.stitching_argument")?;
        let ty = stitching_type(m, descs, r.node_id("air.stitching_argument")?, function)?;
        let name = if r.at_end() { None } else { Some(r.string("air.stitching_argument")?) };
        r.finish()?;
        arguments.push(StitchingArgument { ty, name });
    }
    Ok(StitchingInfo { result, arguments })
}

// Linked constants ----------------------------------------------------------------------------------------------------

/// One entry of `air.linked_constants_signature`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignatureEntry {
    /// `air.module_signature`.
    Module(String),
    /// A linked constant: its value (a constant of the module), name and index.
    Constant { ty: TypeId, value: Value, name: String, index: u32 },
}

pub(super) fn signature(m: &Module, id: MdId) -> Result<Vec<SignatureEntry>, Refusal> {
    let mut c = Cursor::node(m, id, format!("air.signature !{}", id.0), None)?;
    expect_head(&mut c, "air.linked_constants_signature")?;
    let mut out = Vec::new();
    while !c.at_end() {
        let e = c.node_id("air.linked_constants_signature")?;
        let mut r = Cursor::node(m, e, format!("air.linked_constants_signature entry !{}", e.0), None)?;
        if head(&r) == Some(b"air.module_signature") {
            r.advance();
            out.push(SignatureEntry::Module(r.string("air.module_signature")?));
        } else {
            // The constant is the entry's first operand, as metadata wrapping a constant value.
            let first = match m.md(e) {
                crate::ir::Metadata::Tuple { ops, .. } => ops.first().copied().flatten(),
                _ => None,
            };
            let (ty, value) = match first.map(|f| m.md(f)) {
                Some(crate::ir::Metadata::Value { ty, value: value @ Value::Const(_) }) => (*ty, *value),
                _ => return Err(r.error("air.linked_constants_signature", "a constant")),
            };
            r.advance();
            let name = r.string("air.linked_constants_signature")?;
            let index = r.i32("air.linked_constants_signature")?;
            out.push(SignatureEntry::Constant { ty, value, name, index });
        }
        r.finish()?;
    }
    Ok(out)
}

// CoreImage -------------------------------------------------------------------------------------------------------

/// One argument or output record of an `air.ci` kernel description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreImageRecord {
    /// The leading argument number (arguments only).
    pub position: Option<u32>,
    /// `air.ci_builtin`, `air.ci_texture`, `air.ci_sampler`, `air.ci_pointer`, `air.ci_matrix` or `air.ci_imageblock`.
    pub role: &'static str,
    /// `air.read`, `air.write` or `air.sample`, when present.
    pub access: Option<&'static str>,
    /// `air.location_count`.
    pub location_count: Option<u32>,
    pub type_size: Option<u32>,
    pub type_align: Option<u32>,
    pub struct_info: Option<StructInfo>,
    pub type_name: Option<String>,
    pub name: Option<String>,
}

const CI_ROLES: &[&str] =
    &["air.ci_builtin", "air.ci_texture", "air.ci_sampler", "air.ci_pointer", "air.ci_matrix", "air.ci_imageblock"];

fn ci_record(m: &Module, id: MdId) -> Result<CoreImageRecord, Refusal> {
    let mut c = Cursor::node(m, id, format!("air.ci record !{}", id.0), None)?;
    let position = match c.peek() {
        Some(Operand::Int { bits: 32, value }) => {
            c.advance();
            Some(value as u32)
        }
        _ => None,
    };
    let role: &'static str = match head(&c).and_then(|h| CI_ROLES.iter().find(|k| k.as_bytes() == h)) {
        Some(k) => k,
        None => return Err(c.error("air.ci", "an air.ci_* role key")),
    };
    c.advance();
    let mut r = CoreImageRecord {
        position,
        role,
        access: None,
        location_count: None,
        type_size: None,
        type_align: None,
        struct_info: None,
        type_name: None,
        name: None,
    };
    while !c.at_end() {
        let key = match head(&c) {
            Some(k) => String::from_utf8_lossy(k).into_owned(),
            None => return Err(c.error("air.ci", "a key")),
        };
        let fresh = match key.as_str() {
            "air.read" | "air.write" | "air.sample" if role == "air.ci_texture" || role == "air.ci_pointer" => {
                c.advance();
                let access = match key.as_str() {
                    "air.read" => "air.read",
                    "air.write" => "air.write",
                    _ => "air.sample",
                };
                r.access.replace(access).is_none()
            }
            "air.location_count" if role == "air.ci_texture" => {
                c.advance();
                r.location_count.replace(c.i32(&key)?).is_none()
            }
            "air.arg_type_size" if role == "air.ci_pointer" => {
                c.advance();
                r.type_size.replace(c.i32(&key)?).is_none()
            }
            "air.arg_type_align_size" if role == "air.ci_pointer" => {
                c.advance();
                r.type_align.replace(c.i32(&key)?).is_none()
            }
            "air.struct_type_info" if role == "air.ci_pointer" => {
                c.advance();
                let n = c.node_id(&key)?;
                r.struct_info.replace(super::types::struct_info(m, n, None, 0)?).is_none()
            }
            "air.arg_type_name" => {
                c.advance();
                r.type_name.replace(c.string(&key)?).is_none()
            }
            "air.arg_name" => {
                c.advance();
                r.name.replace(c.string(&key)?).is_none()
            }
            _ => return Err(c.error(key, "a key of this air.ci record")),
        };
        if !fresh {
            return Err(c.error(key, "each key once"));
        }
    }
    Ok(r)
}

/// `air.ci`: a CoreImage kernel's outputs and arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreImageKernel {
    pub function: u32,
    pub outputs: Vec<CoreImageRecord>,
    pub arguments: Vec<CoreImageRecord>,
}

pub(super) fn core_image(m: &Module, id: MdId) -> Result<CoreImageKernel, Refusal> {
    let mut c = Cursor::node(m, id, format!("air.ci !{}", id.0), None)?;
    let function = c.function_ref("air.ci")?;
    let mut lists = [Vec::new(), Vec::new()];
    for list in &mut lists {
        let l = c.node_id("air.ci")?;
        let mut lc = Cursor::node(m, l, format!("air.ci list !{}", l.0), None)?;
        while !lc.at_end() {
            list.push(ci_record(m, lc.node_id("air.ci")?)?);
        }
    }
    c.finish()?;
    let [outputs, arguments] = lists;
    Ok(CoreImageKernel { function, outputs, arguments })
}

// Mesh and patch inputs ---------------------------------------------------------------------------------------------

/// `air.mesh_type_info`: a mesh function's output mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshInfo {
    pub vertex_outputs: Vec<Output>,
    pub primitive_outputs: Vec<Output>,
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// The topology key (`air.triangle`).
    pub topology: &'static str,
}

pub(super) fn mesh(m: &Module, id: MdId, function: Option<&str>) -> Result<MeshInfo, Refusal> {
    let mut c = Cursor::node(m, id, format!("air.mesh !{}", id.0), function)?;
    expect_head(&mut c, "air.mesh_type_info")?;
    let mut lists = [Vec::new(), Vec::new()];
    for list in &mut lists {
        let l = c.node_id("air.mesh_type_info")?;
        let mut lc = Cursor::node(m, l, format!("air.mesh_type_info outputs !{}", l.0), function)?;
        while !lc.at_end() {
            list.push(super::decode_output(m, lc.node_id("air.mesh_type_info")?, function)?);
        }
    }
    let max_vertices = c.i32("air.mesh_type_info")?;
    let max_primitives = c.i32("air.mesh_type_info")?;
    let topology = match head(&c) {
        Some(b"air.triangle") => "air.triangle",
        _ => return Err(c.error("air.mesh_type_info", "the topology key air.triangle")),
    };
    c.advance();
    c.finish()?;
    let [vertex_outputs, primitive_outputs] = lists;
    Ok(MeshInfo { vertex_outputs, primitive_outputs, max_vertices, max_primitives, topology })
}

/// One control-point attribute of a post-tessellation vertex function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlPointAttribute {
    pub location: Location,
    pub type_name: String,
    pub name: String,
    pub predicate: Option<Predicate>,
}

/// `air.patch_control_point_input`: the function that reads a control point and its attributes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlPointInput {
    pub function: u32,
    pub attributes: Vec<ControlPointAttribute>,
}

pub(super) fn control_points(m: &Module, nodes: &[MdId], function: Option<&str>) -> Result<ControlPointInput, Refusal> {
    let (first, rest) = nodes.split_first().expect("the grammar gives one or more tuples");
    let mut c = Cursor::node(m, *first, format!("air.patch_control_point_input !{}", first.0), function)?;
    expect_head(&mut c, "air.patch_control_point_function")?;
    let f = c.function_ref("air.patch_control_point_function")?;
    c.finish()?;
    let mut attributes = Vec::new();
    for &n in rest {
        let mut a = Cursor::node(m, n, format!("control-point attribute !{}", n.0), function)?;
        let mut predicate = None;
        if head(&a) == Some(b"air.function_constant") {
            a.advance();
            let p = a.node_id("air.function_constant")?;
            let mut pc = Cursor::node(m, p, format!("air.function_constant !{}", p.0), function)?;
            predicate = Some(Predicate {
                global: pc.global_ref("air.function_constant")?,
                type_name: pc.string("air.function_constant")?,
                name: pc.string("air.function_constant")?,
            });
            pc.finish()?;
        }
        expect_head(&mut a, "air.location_index")?;
        let slot = |a: &mut Cursor<'_>| match a.advance() {
            Some(Operand::Int { bits: 32, value }) => Ok(Slot::Value(value as u32)),
            Some(Operand::Global(g)) => Ok(Slot::Constant(g)),
            _ => Err(refusal(
                RefusalKind::NotImplemented,
                "air.location_index",
                function,
                "a control-point attribute location that is not an i32 or a function constant",
            )),
        };
        let location = Location { index: slot(&mut a)?, count: slot(&mut a)? };
        expect_head(&mut a, "air.arg_type_name")?;
        let type_name = a.string("air.arg_type_name")?;
        expect_head(&mut a, "air.arg_name")?;
        let name = a.string("air.arg_name")?;
        a.finish()?;
        attributes.push(ControlPointAttribute { location, type_name, name, predicate });
    }
    Ok(ControlPointInput { function: f, attributes })
}
