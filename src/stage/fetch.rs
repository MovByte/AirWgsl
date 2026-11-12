//! Attribute fetch: `[[stage_in]]` vertex attributes described by an `MTLVertexDescriptor`, and kernel stage-in
//! attributes described by an `MTLStageInputOutputDescriptor`, read by the generated code from the bound buffers
//! (vertex pulling, plan 5.10).
//!
//! Every attribute is pulled: the attribute's buffer slot is bound as a read-only storage binding of the stage (the
//! same binding the memory model uses for that slot) and the generated code reads the element's bytes at
//! `offset + index * stride` from the Metal binding offset, extracts the format's components and converts them to the
//! attribute's type. Byte addresses need no alignment: a component that straddles two words is assembled from both.
//! WebGPU's native vertex input is not used; its formats and alignment rules cover a subset, and a native path would
//! be an optimization with its own equivalence test (plan 5.10).
//!
//! # Formats and attribute types
//!
//! The pipeline is created exactly for these combinations and refused with "Vertex attribute ... cannot be read using
//! MTLAttributeFormat..." otherwise. Measured on real Metal (Apple M4, macOS 27.0.1), and the rule is the framework's
//! own on the target build: 21H461's `-[MTLVertexDescriptorInternal validateWithVertexFunction:error:
//! renderPipelineDescriptor:]` (0x187bb874c) checks each active attribute's data type against
//! `isVertexFormatInteger(format)` this way, with that message, before any compiler runs:
//!
//! | Format | Attribute scalar |
//! | --- | --- |
//! | half, float, normalized (including the 10-10-10-2 and BGRA forms) | `float`, `half` |
//! | unsigned 8- or 16-bit integer | `float`, `half`, `uint`, `ushort`, `uchar` |
//! | signed 8- or 16-bit integer | `float`, `half`, `int`, `short`, `char` |
//! | unsigned 32-bit integer | `uint`, `ushort`, `uchar` |
//! | signed 32-bit integer | `int`, `short`, `char` |
//!
//! The measured conversions (every code of each 8-, 10- and 16-bit format, and 65536 binary32 and binary16
//! patterns): an integer attribute receives the component's low bits after zero- or sign-extension; a float attribute
//! receives the exact value of an integer component, and for a normalized component the correctly rounded `c / (2^n -
//! 1)` (unorm) or `max(-1, c / (2^(n-1) - 1))` (snorm), except that the 10-bit components of `Int1010102Normalized`
//! are `max(-1, c * RN(1/511))` with a correctly rounded product; a half attribute receives the correctly rounded
//! value rounded once, to nearest even, to binary16; binary32 to binary16 rounds to nearest even with a NaN becoming `0x7e00`; binary16 to binary32 is exact
//! with a NaN becoming `0x7fc00000`; a component fetched into its own type keeps its bits. Components the format lacks
//! are taken from (0, 0, 0, 1) and components the attribute lacks are dropped. `UInt1010102Normalized` and
//! `Int1010102Normalized` hold x, y, z and w from the least significant bit (10, 10, 10 and 2 bits);
//! `UChar4Normalized_BGRA` holds the bytes of z, y, x and w in memory order. These match the V1 goldens of
//! `vertex.formats`.
//!
//! # Fetch index
//!
//! - Render (`MTLVertexStepFunction`): per-vertex fetches element `vertex_id` (which includes the base vertex, as
//!   WebGPU's `vertex_index` does); per-instance with step rate `r` fetches `base_instance + (instance_id -
//!   base_instance) / r` (V1 `vertex.step-base-instance` and a measurement with base instance 3 and rate 2); constant
//!   fetches element 0 whatever the base instance (measured).
//! - Compute stage-in (`MTLStepFunction`): `ThreadPositionInGridX` with step rate `r` fetches `origin.x + min(gid.x /
//!   r, size.x - 1)` for the stage-in region (`setStageInRegion:`) with origin and size, `ThreadPositionInGridY` the
//!   same in y, the indexed forms fetch `index[min(gid / r, size - 1)] + origin` from the index buffer, and constant
//!   fetches element 0 (V1 `compute.stage-in-region`, and measurements of rates 1 to 3, origins in x and y, both
//!   index types and threads beyond the region). MSL defines the region only as the grid over which threads that
//!   read stage-in data are launched; the clamp of threads beyond it is the measured behavior.
//!
//! # What real Metal rejects
//!
//! The framework's descriptor validation (measured) rejects an attribute the descriptor does not describe, a stride of
//! zero, a per-vertex step rate other than one, a per-instance step rate of zero, a constant step rate other than
//! zero, and stage-in step functions other than constant and the four thread-position forms. The A10X reports no
//! dynamic attribute stride (D1 `supportsDynamicAttributeStride = false`), and the packed float formats
//! `FloatRG11B10` and `FloatRGB9E5` are listed from Apple5 in Apple's feature set tables (the V1 case requires
//! Apple5; the A10X is Apple3). Each of these is refused as on the device.
//!
//! Alignment is not validated on the target: 21H461's Metal.framework checks no stride or offset alignment (the
//! descriptor validation above and `newSerializedDescriptor`), and its only use of `supportsUnalignedVertexFetch`, which
//! the A10X answers NO, is `MTLGetCompilerOptions` (0x187a9acdc) choosing a compiler option bit, so the pipeline is
//! created. The API validation layer (MetalTools) alone checks, at draw and stage-in dispatch time, that a stride is a
//! multiple of 4 on such a device (`-[MTLDebugRenderCommandEncoder validateCommonDrawErrors:instanceCount:baseInstance:
//! maxVertexID:]`, "providing stride of %u to buffer-layout index %u, but device only supports strides that are a
//! multiple of 4 bytes."), which the guest's own validation layer reports against the plugin device. Pipelines with
//! unaligned strides or offsets are therefore compiled; the generated code reads the addressed bytes, as a device
//! with unaligned fetch does (measured), where the A10X's result is undefined.

use super::lib;
use crate::air::{Argument, Location, Role, Slot};
use crate::api::{Location as At, PipelineKind, Refusal, UniformValue, VertexDescriptor};
use crate::ir::{Module, Type, TypeId};
use crate::wgsl::hooks::StageCx;
use crate::wgsl::tracked;

type R<T> = Result<T, Refusal>;

/// The encoding of a format's components.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    UInt,
    SInt,
    UNorm,
    SNorm,
    Half,
    Float,
    /// `UInt1010102Normalized`.
    UNorm1010102,
    /// `Int1010102Normalized`.
    SNorm1010102,
    /// `UChar4Normalized_BGRA`.
    UNormBgra,
    /// `FloatRG11B10`, `FloatRGB9E5`.
    PackedFloat,
}

/// One `MTLVertexFormat` (and the `MTLAttributeFormat` of the same value).
#[derive(Clone, Copy, Debug)]
pub struct Format {
    pub value: u64,
    pub name: &'static str,
    pub kind: Kind,
    pub components: u32,
    /// Bits of each component (the packed forms: of the word).
    pub bits: u32,
}

const fn f(value: u64, name: &'static str, kind: Kind, components: u32, bits: u32) -> Format {
    Format { value, name, kind, components, bits }
}

/// `MTLVertexFormat` (iPhoneOS SDK `MTLVertexDescriptor.h`; `MTLAttributeFormat` in `MTLStageInputOutputDescriptor.h`
/// has the same values).
pub const FORMATS: &[Format] = &[
    f(1, "UChar2", Kind::UInt, 2, 8),
    f(2, "UChar3", Kind::UInt, 3, 8),
    f(3, "UChar4", Kind::UInt, 4, 8),
    f(4, "Char2", Kind::SInt, 2, 8),
    f(5, "Char3", Kind::SInt, 3, 8),
    f(6, "Char4", Kind::SInt, 4, 8),
    f(7, "UChar2Normalized", Kind::UNorm, 2, 8),
    f(8, "UChar3Normalized", Kind::UNorm, 3, 8),
    f(9, "UChar4Normalized", Kind::UNorm, 4, 8),
    f(10, "Char2Normalized", Kind::SNorm, 2, 8),
    f(11, "Char3Normalized", Kind::SNorm, 3, 8),
    f(12, "Char4Normalized", Kind::SNorm, 4, 8),
    f(13, "UShort2", Kind::UInt, 2, 16),
    f(14, "UShort3", Kind::UInt, 3, 16),
    f(15, "UShort4", Kind::UInt, 4, 16),
    f(16, "Short2", Kind::SInt, 2, 16),
    f(17, "Short3", Kind::SInt, 3, 16),
    f(18, "Short4", Kind::SInt, 4, 16),
    f(19, "UShort2Normalized", Kind::UNorm, 2, 16),
    f(20, "UShort3Normalized", Kind::UNorm, 3, 16),
    f(21, "UShort4Normalized", Kind::UNorm, 4, 16),
    f(22, "Short2Normalized", Kind::SNorm, 2, 16),
    f(23, "Short3Normalized", Kind::SNorm, 3, 16),
    f(24, "Short4Normalized", Kind::SNorm, 4, 16),
    f(25, "Half2", Kind::Half, 2, 16),
    f(26, "Half3", Kind::Half, 3, 16),
    f(27, "Half4", Kind::Half, 4, 16),
    f(28, "Float", Kind::Float, 1, 32),
    f(29, "Float2", Kind::Float, 2, 32),
    f(30, "Float3", Kind::Float, 3, 32),
    f(31, "Float4", Kind::Float, 4, 32),
    f(32, "Int", Kind::SInt, 1, 32),
    f(33, "Int2", Kind::SInt, 2, 32),
    f(34, "Int3", Kind::SInt, 3, 32),
    f(35, "Int4", Kind::SInt, 4, 32),
    f(36, "UInt", Kind::UInt, 1, 32),
    f(37, "UInt2", Kind::UInt, 2, 32),
    f(38, "UInt3", Kind::UInt, 3, 32),
    f(39, "UInt4", Kind::UInt, 4, 32),
    f(40, "Int1010102Normalized", Kind::SNorm1010102, 4, 32),
    f(41, "UInt1010102Normalized", Kind::UNorm1010102, 4, 32),
    f(42, "UChar4Normalized_BGRA", Kind::UNormBgra, 4, 8),
    f(45, "UChar", Kind::UInt, 1, 8),
    f(46, "Char", Kind::SInt, 1, 8),
    f(47, "UCharNormalized", Kind::UNorm, 1, 8),
    f(48, "CharNormalized", Kind::SNorm, 1, 8),
    f(49, "UShort", Kind::UInt, 1, 16),
    f(50, "Short", Kind::SInt, 1, 16),
    f(51, "UShortNormalized", Kind::UNorm, 1, 16),
    f(52, "ShortNormalized", Kind::SNorm, 1, 16),
    f(53, "Half", Kind::Half, 1, 16),
    f(54, "FloatRG11B10", Kind::PackedFloat, 3, 32),
    f(55, "FloatRGB9E5", Kind::PackedFloat, 3, 32),
];

pub fn format(value: u64) -> Option<&'static Format> {
    FORMATS.iter().find(|f| f.value == value)
}

/// The scalar type of an attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    Float,
    Half,
    UInt(u32),
    SInt(u32),
}

impl Scalar {
    fn name(self) -> String {
        match self {
            Scalar::Float => "float".into(),
            Scalar::Half => "half".into(),
            Scalar::UInt(8) => "uchar".into(),
            Scalar::UInt(16) => "ushort".into(),
            Scalar::UInt(_) => "uint".into(),
            Scalar::SInt(8) => "char".into(),
            Scalar::SInt(16) => "short".into(),
            Scalar::SInt(_) => "int".into(),
        }
    }
}

/// The scalar type and lane count of an attribute of IR type `t` and MSL type name `type_name`.
pub fn attribute_type(m: &Module, t: TypeId, type_name: Option<&str>) -> Option<(Scalar, u32)> {
    let (lanes, e) = match m.types.get(t) {
        Type::Vector { len, elem, .. } if (2..=4).contains(len) => (*len, *elem),
        Type::Vector { .. } => return None,
        _ => (1, t),
    };
    let scalar = match m.types.get(e) {
        Type::Float => Scalar::Float,
        Type::Half => Scalar::Half,
        Type::Int(b @ (8 | 16 | 32)) => {
            let base = type_name?.trim_end_matches(|c: char| c.is_ascii_digit());
            match base {
                "uint" | "ushort" | "uchar" => Scalar::UInt(*b),
                "int" | "short" | "char" => Scalar::SInt(*b),
                _ => return None,
            }
        }
        _ => return None,
    };
    Some((scalar, lanes))
}

/// Whether real Metal creates a pipeline that reads `format` into an attribute of scalar type `s` (measured; see the
/// module documentation).
pub fn compatible(format: &Format, s: Scalar) -> bool {
    match format.kind {
        Kind::UInt if format.bits <= 16 => matches!(s, Scalar::Float | Scalar::Half | Scalar::UInt(_)),
        Kind::SInt if format.bits <= 16 => matches!(s, Scalar::Float | Scalar::Half | Scalar::SInt(_)),
        Kind::UInt => matches!(s, Scalar::UInt(_)),
        Kind::SInt => matches!(s, Scalar::SInt(_)),
        _ => matches!(s, Scalar::Float | Scalar::Half),
    }
}

fn rejected(construct: impl Into<String>, at: &Option<At>, detail: impl Into<String>) -> Refusal {
    tracked::refuse("vertex-descriptor", construct, at.clone(), detail)
}

/// The WGSL of a byte loader over a storage variable: the little-endian `u32` at any byte address.
fn loader(cx: &mut dyn StageCx, var: &str) -> String {
    let name = format!("aw_vload_{}", var);
    cx.module_item(
        &name,
        format!(
            "fn {n}(a: u32) -> u32 {{\n    let w = a >> 2u;\n    let s = (a & 3u) * 8u;\n    if (s == 0u) {{\n        \
             return {v}[w];\n    }}\n    return ({v}[w] >> s) | ({v}[w + 1u] << (32u - s));\n}}\n",
            n = name,
            v = var
        ),
    );
    name
}

fn mask(bits: u32) -> String {
    if bits >= 32 { "0xffffffffu".into() } else { format!("{}u", (1u64 << bits) - 1) }
}

/// The raw components of the element at byte address `addr` (a WGSL `u32` expression), each a `u32` holding the
/// component's bits zero-extended, in x, y, z, w order.
fn raw_components(load: &str, addr: &str, format: &Format) -> Vec<String> {
    let at = |off: u32| if off == 0 { format!("{}({})", load, addr) } else { format!("{}({} + {}u)", load, addr, off) };
    match format.kind {
        Kind::UNorm1010102 | Kind::SNorm1010102 => {
            let w = at(0);
            [(0, 10), (10, 10), (20, 10), (30, 2)]
                .iter()
                .map(|(o, b)| format!("extractBits({}, {}u, {}u)", w, o, b))
                .collect()
        }
        Kind::UNormBgra => [2, 1, 0, 3].iter().map(|k| format!("extractBits({}, 0u, 8u)", at(*k))).collect(),
        _ => {
            let bytes = format.bits / 8;
            (0..format.components)
                .map(|k| {
                    let w = at(k * bytes);
                    if format.bits == 32 { w } else { format!("extractBits({}, 0u, {}u)", w, format.bits) }
                })
                .collect()
        }
    }
}

/// Convert one raw component (`bits` wide) of `format` to the carrier of scalar `s`.
fn convert(cx: &mut dyn StageCx, raw: &str, format: &Format, bits: u32, s: Scalar) -> String {
    let sext = |raw: &str| format!("(bitcast<i32>({} << {}u) >> {}u)", raw, 32 - bits, 32 - bits);
    match (format.kind, s) {
        (Kind::UInt, Scalar::UInt(w)) => {
            if w >= bits {
                raw.to_string()
            } else {
                format!("({} & {})", raw, mask(w))
            }
        }
        (Kind::SInt, Scalar::SInt(w)) => {
            if w == bits {
                raw.to_string()
            } else if w < bits {
                format!("({} & {})", raw, mask(w))
            } else {
                format!("(bitcast<u32>{} & {})", sext(raw), mask(w))
            }
        }
        (Kind::UInt, Scalar::Float) => format!("bitcast<u32>(f32({}))", raw),
        (Kind::SInt, Scalar::Float) => format!("bitcast<u32>(f32{})", sext(raw)),
        (Kind::UInt, Scalar::Half) => {
            lib(cx, "aw_vf_int_f16");
            format!("aw_vf_int_f16(i32({}))", raw)
        }
        (Kind::SInt, Scalar::Half) => {
            lib(cx, "aw_vf_int_f16");
            format!("aw_vf_int_f16{}", sext(raw))
        }
        (Kind::Float, Scalar::Float) | (Kind::Half, Scalar::Half) => raw.to_string(),
        (Kind::Float, Scalar::Half) => {
            lib(cx, "aw_vf_f32_to_f16");
            format!("aw_vf_f32_to_f16({})", raw)
        }
        (Kind::Half, Scalar::Float) => {
            lib(cx, "aw_vf_f16_to_f32");
            format!("aw_vf_f16_to_f32({})", raw)
        }
        (Kind::SNorm1010102, Scalar::Float) if bits == 10 => {
            cx.prelude("nm_f32_mul");
            lib(cx, "aw_vf_snorm10_f32");
            format!("aw_vf_snorm10_f32({})", raw)
        }
        (Kind::UNorm | Kind::UNormBgra | Kind::UNorm1010102 | Kind::SNorm | Kind::SNorm1010102, _) => {
            let signed = matches!(format.kind, Kind::SNorm | Kind::SNorm1010102);
            let f = match (signed, s) {
                (false, Scalar::Half) => "aw_vf_unorm_f16",
                (false, _) => "aw_vf_unorm_f32",
                (true, Scalar::Half) => "aw_vf_snorm_f16",
                (true, _) => "aw_vf_snorm_f32",
            };
            lib(cx, f);
            format!("{}({}, {}u)", f, raw, bits)
        }
        _ => unreachable!("compatible() admits only the combinations above"),
    }
}

/// The attribute value (in the carrier of `scalar` x `lanes`) of the element at byte address `addr`.
pub fn element(cx: &mut dyn StageCx, load: &str, addr: &str, format: &Format, scalar: Scalar, lanes: u32) -> String {
    let raw = raw_components(load, addr, format);
    let widths: Vec<u32> = match format.kind {
        Kind::UNorm1010102 | Kind::SNorm1010102 => vec![10, 10, 10, 2],
        Kind::UNormBgra => vec![8; 4],
        _ => vec![format.bits; raw.len()],
    };
    let one = match scalar {
        Scalar::Float => "0x3f800000u",
        Scalar::Half => "0x3c00u",
        _ => "1u",
    };
    let lanes_v: Vec<String> = (0..lanes as usize)
        .map(|k| match raw.get(k) {
            Some(r) => convert(cx, r, format, widths[k], scalar),
            None if k == 3 => one.to_string(),
            None => "0u".to_string(),
        })
        .collect();
    if lanes == 1 { lanes_v[0].clone() } else { format!("vec{}<u32>({})", lanes, lanes_v.join(", ")) }
}

/// The record of a stage-in attribute: its attribute index.
fn attribute_index(at: &Option<At>, loc: &Location, kind: &str) -> R<u64> {
    let index = match loc.index {
        Slot::Value(v) => v as u64,
        Slot::Constant(_) => return Err(tracked::refuse_family("function-constants", kind, at.clone())),
    };
    if loc.count != Slot::Value(1) {
        return Err(tracked::refuse("vertex-input", kind, at.clone(), "an attribute spanning several locations"));
    }
    Ok(index)
}

/// The location of the entry function and the IR type of argument `arg`.
fn arg_info(cx: &dyn StageCx, arg: u32) -> (Option<At>, TypeId) {
    let m = cx.module();
    let fi = cx.function();
    (super::loc(m, fi), m.functions[fi].args[arg as usize].ty)
}

/// `MTLBufferLayoutStrideDynamic`.
const STRIDE_DYNAMIC: u64 = u64::MAX;

/// Lower a `[[stage_in]]` vertex attribute (`air.vertex_input`).
pub fn vertex_input(cx: &mut dyn StageCx, arg: u32, record: &Argument) -> R<String> {
    let (at, ty) = arg_info(cx, arg);
    let at = &at;
    let Role::VertexInput(loc) = &record.role else { unreachable!("registered for air.vertex_input") };
    let attr = attribute_index(at, loc, "air.vertex_input")?;
    let construct = format!("attribute({})", attr);
    let (scalar, lanes) = attribute_type(cx.module(), ty, record.type_name.as_deref())
        .ok_or_else(|| tracked::refuse("vertex-input", construct.clone(), at.clone(), "an attribute type"))?;
    let vd: Option<&VertexDescriptor> = match &cx.desc().kind {
        PipelineKind::Render(r) => r.vertex_descriptor.as_ref(),
        _ => None,
    };
    let Some(vd) = vd else {
        return Err(rejected(
            construct,
            at,
            "the function has a [[stage_in]] attribute and the pipeline no vertex descriptor",
        ));
    };
    let a = vd
        .attributes
        .get(attr as usize)
        .copied()
        .filter(|a| a.format != 0)
        .ok_or_else(|| rejected(construct.clone(), at, "the vertex descriptor does not describe the attribute"))?;
    let fmt = check_format(at, &construct, a.format, scalar, lanes)?;
    if a.buffer_index >= 31 {
        return Err(rejected(construct, at, format!("buffer index {} is not below 31", a.buffer_index)));
    }
    let layout = vd.layouts.get(a.buffer_index as usize).copied().unwrap_or(crate::api::VertexBufferLayout {
        stride: 0,
        step_function: 1,
        step_rate: 1,
    });
    let stride = check_stride(at, &construct, a.buffer_index, layout.stride)?;
    let index = match (layout.step_function, layout.step_rate) {
        (0, 0) => "0u".to_string(),
        (1, 1) => cx.builtin("vertex_index", "u32"),
        (2, r) if r > 0 => {
            let iid = cx.builtin("instance_index", "u32");
            let base = cx.uniform(UniformValue::BaseInstance);
            if r == 1 { iid } else { format!("({b} + ({i} - {b}) / {r}u)", b = base, i = iid, r = r) }
        }
        (3 | 4, _) => return Err(tracked::refuse_family("tessellation", construct, at.clone())),
        (s, r) => {
            return Err(rejected(
                construct,
                at,
                format!("step function {} with step rate {} (the framework's descriptor validation rejects it)", s, r),
            ));
        }
    };
    fetch(cx, arg, a.buffer_index, stride, a.offset, &index, fmt, scalar, lanes)
}

fn check_format(at: &Option<At>, construct: &str, value: u64, scalar: Scalar, lanes: u32) -> R<&'static Format> {
    let fmt =
        format(value).ok_or_else(|| rejected(construct, at, format!("format {} is not an MTLVertexFormat", value)))?;
    if fmt.kind == Kind::PackedFloat {
        return Err(rejected(
            construct,
            at,
            format!(
                "MTLVertexFormat{} is supported from Apple5 (Metal feature set tables); the A10X is Apple3",
                fmt.name
            ),
        ));
    }
    if !compatible(fmt, scalar) {
        return Err(rejected(
            construct,
            at,
            format!(
                "an attribute of type {}{} cannot be read using MTLAttributeFormat{} (measured on Metal)",
                scalar.name(),
                if lanes > 1 { lanes.to_string() } else { String::new() },
                fmt.name
            ),
        ));
    }
    Ok(fmt)
}

fn check_stride(at: &Option<At>, construct: &str, buffer: u64, stride: u64) -> R<u64> {
    if stride == STRIDE_DYNAMIC {
        return Err(rejected(
            construct,
            at,
            "a dynamic attribute stride: the A10X reports supportsDynamicAttributeStride = false (D1)",
        ));
    }
    if stride == 0 {
        return Err(rejected(construct, at, format!("buffer {} has no stride", buffer)));
    }
    if stride > u32::MAX as u64 {
        return Err(rejected(construct, at, format!("buffer {} has stride {}", buffer, stride)));
    }
    Ok(stride)
}

/// Emit the fetch of one attribute from Metal buffer `buffer` at element `index` (a WGSL `u32`).
#[allow(clippy::too_many_arguments)]
fn fetch(
    cx: &mut dyn StageCx,
    arg: u32,
    buffer: u64,
    stride: u64,
    offset: u64,
    index: &str,
    fmt: &Format,
    scalar: Scalar,
    lanes: u32,
) -> R<String> {
    let (var, base) = cx.buffer(buffer)?;
    let load = loader(cx, &var);
    let addr = format!("aw_va{}", arg);
    cx.prologue(format!("let {} = {} + {} * {}u + {}u;", addr, base, index, stride, offset));
    Ok(element(cx, &load, &addr, fmt, scalar, lanes))
}

/// Lower a kernel's `[[stage_in]]` attribute (`air.stage_in` with a location).
pub fn stage_in(cx: &mut dyn StageCx, arg: u32, record: &Argument) -> R<String> {
    let (at, ty) = arg_info(cx, arg);
    let at = &at;
    let Role::StageIn(Some(loc)) = &record.role else {
        return Err(tracked::refuse(
            "vertex-input",
            "air.stage_in",
            at.clone(),
            "a stage-in record without a location",
        ));
    };
    let attr = attribute_index(at, loc, "air.stage_in")?;
    let construct = format!("attribute({})", attr);
    let (scalar, lanes) = attribute_type(cx.module(), ty, record.type_name.as_deref())
        .ok_or_else(|| tracked::refuse("vertex-input", construct.clone(), at.clone(), "an attribute type"))?;
    let si = match &cx.desc().kind {
        PipelineKind::Compute(c) => c.stage_input.clone(),
        _ => None,
    };
    let Some(si) = si else {
        return Err(rejected(
            construct,
            at,
            "the kernel has a [[stage_in]] attribute and the pipeline no stage input descriptor",
        ));
    };
    let (format_value, offset, buffer) =
        si.attributes.get(attr as usize).copied().filter(|a| a.0 != 0).ok_or_else(|| {
            rejected(construct.clone(), at, "the stage input descriptor does not describe the attribute")
        })?;
    let fmt = check_format(at, &construct, format_value, scalar, lanes)?;
    if buffer >= 31 {
        return Err(rejected(construct, at, format!("buffer index {} is not below 31", buffer)));
    }
    let (stride, step, rate) = si.layouts.get(buffer as usize).copied().unwrap_or((0, 5, 1));
    let stride = check_stride(at, &construct, buffer, stride)?;
    let index = match (step, rate) {
        (0, 0) => "0u".to_string(),
        (5..=8, r) if r > 0 => {
            let axis = if step == 5 || step == 7 { 0 } else { 1 };
            let gid = cx.builtin("global_invocation_id", "vec3<u32>");
            let origin = cx.uniform_words(UniformValue::StageInOrigin);
            let size = cx.uniform_words(UniformValue::StageInSize);
            let c = ["x", "y"][axis];
            let pos = format!("min({}.{} / {}u, {} - 1u)", gid, c, r, size[axis]);
            if step <= 6 {
                format!("({} + {})", origin[axis], pos)
            } else {
                let (ivar, ibase) = cx.buffer(si.index_buffer_index)?;
                let iload = loader(cx, &ivar);
                let element = match si.index_type {
                    0 => format!("extractBits({}({} + {} * 2u), 0u, 16u)", iload, ibase, pos),
                    1 => format!("{}({} + {} * 4u)", iload, ibase, pos),
                    t => return Err(rejected(construct, at, format!("index type {}", t))),
                };
                format!("({} + {})", element, origin[axis])
            }
        }
        (s, r) => {
            return Err(rejected(
                construct,
                at,
                format!("stage-in step function {} with step rate {} (the framework's validation rejects it)", s, r),
            ));
        }
    };
    fetch(cx, arg, buffer, stride, offset, &index, fmt, scalar, lanes)
}

/// The `MTLVertexFormat` whose components are an attribute's own type: `float4` reads `Float4`, `ushort2` reads
/// `UShort2`, `half` reads `Half`.
pub fn natural_format(scalar: Scalar, lanes: u32) -> &'static Format {
    let (kind, bits) = match scalar {
        Scalar::Float => (Kind::Float, 32),
        Scalar::Half => (Kind::Half, 16),
        Scalar::UInt(b) => (Kind::UInt, b),
        Scalar::SInt(b) => (Kind::SInt, b),
    };
    FORMATS.iter().find(|f| f.kind == kind && f.bits == bits && f.components == lanes).expect("every type has a format")
}

/// One buffer's layout for attributes read in their own types: the buffer index, the stride, and each attribute's
/// (index, `MTLVertexFormat`, offset).
pub type NaturalLayout = (u64, u64, Vec<(u64, u64, u64)>);

/// A descriptor that reads each `[[stage_in]]` attribute of an entry point in its own type from one buffer, in
/// attribute order with each attribute at a multiple of its component size: the pipeline state for compiling a
/// function on its own (the corpus checks). `None` when the function has no attributes or one without a type this
/// lowering reads. The buffer is the highest index below 31 that no buffer argument uses.
pub fn natural_layout(m: &Module, e: &crate::air::EntryPoint) -> Option<NaturalLayout> {
    let fi = e.function as usize;
    let mut attrs: Vec<(u64, &'static Format)> = Vec::new();
    let mut used = [false; 31];
    for a in &e.arguments {
        match &a.role {
            Role::VertexInput(l) | Role::StageIn(Some(l)) => {
                let Slot::Value(index) = l.index else { return None };
                let (s, n) = attribute_type(m, m.functions[fi].args[a.position as usize].ty, a.type_name.as_deref())?;
                attrs.push((index as u64, natural_format(s, n)));
            }
            Role::Buffer(b) | Role::IndirectBuffer(b) => {
                if let Slot::Value(i) = b.location.index
                    && (i as usize) < used.len()
                {
                    used[i as usize] = true;
                }
            }
            _ => {}
        }
    }
    if attrs.is_empty() {
        return None;
    }
    let buffer = (0..31u64).rev().find(|i| !used[*i as usize])?;
    attrs.sort_by_key(|a| a.0);
    let mut offset = 0u64;
    let mut out = Vec::new();
    for (index, f) in attrs {
        let size = (f.bits / 8 * f.components) as u64;
        let align = (f.bits / 8) as u64;
        offset = offset.div_ceil(align) * align;
        out.push((index, f.value, offset));
        offset += size;
    }
    Some((buffer, offset.div_ceil(4).max(1) * 4, out))
}
