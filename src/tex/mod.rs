//! Texture and sampler lowering (plan 5.5, 5.7, 6.3, 6.4): the `air.*` texture functions of the families
//! `texture-sample`, `texture-read-write`, `texture-gather`, `texture-query` and `is_null_texture` of
//! `texture-null`, over texture and sampler arguments and constant samplers.
//!
//! # Bindings
//!
//! A texture argument becomes one WebGPU binding of the Metal texture slot. Its WGSL type follows the AIR texture type
//! and the texture the pipeline description binds at the slot ([`crate::api::TextureSlot`]), which is pipeline-variant
//! state: the texel class and filterability of the stored format ([`format`]), a storage format for write access,
//! and the view dimension. The executor stores 1D textures as 2D textures of height 1 and binds 1D and 1D array
//! textures as 2D and 2D array views (WebMetal `host/src/mem/texture.rs`); a cube texture that the function reads or
//! writes texel by texel is bound as a 2D array view of its faces, since WGSL loads and stores no cube views.
//!
//! A written texture is bound by level ([`crate::api::BindingSource::MetalTextureLevel`]): a storage binding shows
//! one level (WebGPU 8.2.1), and Metal writes at a level operand (`write(color, coord, lod)`, which the A10X's
//! compiler takes computed at run time for every texture type but 1D and 1D array: its `metal_texture` header
//! constrains `lod` to a constant 0 only for those, and on macOS before Metal 2.3). A write at a constant level binds
//! that level; a write at a level computed at run time binds every level of the bound texture (a view's own levels,
//! from its first) and selects among them with a `switch`, whose default is the out-of-range level that MSL 4.1
//! section 6.13 ignores. The levels count against the host's storage textures per stage: a stage that would bind more
//! is refused with the textures and their levels named, never bound with fewer levels. The level count is variant
//! state ([`crate::api::TextureSlot::mipmap_level_count`]), so `get_num_mip_levels` of a written texture is a
//! constant.
//!
//! A texture slot the description does not list holds no texture: the argument is a null texture. Only
//! `is_null_texture` is defined for it (MSL 4.1 section 6.13.18: "The behavior of calling any texture member function
//! with a null texture is undefined"), so its other calls are lowered for the declared type with the canonical
//! binding of that type (a filterable float, sint or uint texture, a depth texture, or an `rgba32` storage texture),
//! for which the executor binds a placeholder.
//!
//! A sampler argument becomes a binding of the slot, a constant sampler (`air.sampler_state`) a binding of its encoding
//! ([`crate::api::BindingSource::ConstantSampler`]); the binding type is "comparison" for the compare functions, and
//! otherwise "filtering" when a filter of the bound descriptor is linear and "non-filtering" when not. A sampler slot
//! the description does not list holds no sampler (sampling with it is undefined); it is bound as "non-filtering".
//!
//! # Exactness
//!
//! Each call is lowered either to the WGSL function with the same definition, or to texel fetches the shader does
//! itself when WebGPU has no counterpart: sampling an integer texture (WGSL samples float textures only), the
//! `clampToZero` address mode, unnormalized coordinates, and formats the executor stores in another format. The
//! fetch path is exact for nearest filtering without mipmaps, where the result is one texel of level 0 selected by
//! the address modes; a fetch with linear filtering, a mipmap filter or anisotropy is refused, because Metal states no
//! filter-weight precision or mipmap selection rule to reproduce (WebMetal `tables/formats.toml` question
//! 5.7-filter-weight-precision; refusal row `address-modes-with-mips-or-anisotropy`). Out-of-bounds reads return the
//! values MSL 4.1 section 6.13 defines and out-of-bounds writes are skipped; WGSL leaves both to the implementation,
//! so every read and write is bounds-checked. The level of a read or write is in bounds below the level count of the
//! texture the function is given, a view's own count: Apple silicon writes some out-of-range levels (measured on the
//! Apple M4: it takes the level modulo 2^16 and lets a write past a view's last level land in the parent's level),
//! which MSL ignores and the A10X's own behavior is not measured for, so the lowering follows MSL; reads at those
//! levels are out of bounds on the M4 as well (WebMetal reference case `texture.write-levels`).
//!
//! Sampler settings the A10X does not support (mirrorClampToEdge, border colors), cube texture arrays, 2D
//! multisample arrays and `access::read_write` textures are refused as the device rejects them (the getters of the
//! WebMetal device profile). The sparse residency flag of each result reads "resident": no A10X texture is sparse.
//!
//! # Swizzles
//!
//! A bound texture's swizzle, composed after the format's own (A8Unorm reads as (0, 0, 0, r)), is applied by the
//! texture view when the host has `texture-component-swizzle` (the executor sets the view's swizzle; the compiled
//! pipeline then requires the feature) and by the shader otherwise.

// The refusal is the compiler's only failure value (api::compile) and is returned by value throughout.
#![allow(clippy::result_large_err)]

pub mod format;
mod lower;

use crate::air::{Access, Role, Slot};
use crate::api::{
    BindingResource, BindingSource, MetalStage, Refusal, RefusalKind, SamplerBindingType, SamplerSlot,
    StorageTextureAccess, TextureSampleType, TextureSlot, TextureViewDimension,
};
use crate::ir::{ConstExpr, ConstKind, GlobalId, Module, Op, Type, TypeId, Value};
use crate::wgsl::hooks::{CallRule, CallSite, RuleCx};
use crate::wgsl::tracked;
use format::{Class, Shim, Stored, Texel, Unavailable};

type R<T> = Result<T, Refusal>;

/// The texture and sampler rule set.
pub struct Textures;

impl CallRule for Textures {
    fn handles(&self, name: &str) -> bool {
        let Some(n) = parse(name) else { return false };
        let family = tracked::family_of(name);
        match n.op {
            Op2::IsNull => family == Some("texture-null"),
            _ => matches!(family, Some("texture-sample" | "texture-read-write" | "texture-gather" | "texture-query")),
        }
    }

    fn claims_argument(&self, kind: &str) -> bool {
        kind == "air.texture" || kind == "air.sampler"
    }

    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> R<Option<String>> {
        let name = parse(call.name).expect("handles() parsed the name");
        lower::lower(cx, call, &name)
    }
}

// Names ---------------------------------------------------------------------------------------------------------------

/// What a texture function does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op2 {
    Sample,
    SampleCompare,
    Gather,
    GatherCompare,
    Read,
    Write,
    Width,
    Height,
    Depth,
    ArraySize,
    MipLevels,
    Samples,
    IsNull,
}

/// The AIR texture type a function names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dim {
    D1,
    D1Array,
    D2,
    D2Array,
    D2Ms,
    D2MsArray,
    D3,
    Cube,
    CubeArray,
    Buffer,
}

impl Dim {
    pub(crate) fn arrayed(self) -> bool {
        matches!(self, Dim::D1Array | Dim::D2Array | Dim::D2MsArray | Dim::CubeArray)
    }

    /// `MTLTextureType` of a texture of this AIR type.
    fn texture_type(self) -> u64 {
        match self {
            Dim::D1 => 0,
            Dim::D1Array => 1,
            Dim::D2 => 2,
            Dim::D2Array => 3,
            Dim::D2Ms => 4,
            Dim::Cube => 5,
            Dim::CubeArray => 6,
            Dim::D3 => 7,
            Dim::D2MsArray => 8,
            Dim::Buffer => 9,
        }
    }
}

/// The element type of a function's texel operand or result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Elem {
    None,
    F32,
    F16,
    /// 32-bit integers, signed or not.
    I32(bool),
    /// 16-bit integers, signed or not.
    I16(bool),
}

/// A parsed texture function name: `air.<op>_<texture|depth>_<dim>[_grad][.i16][.s|.u][.<type>]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Name {
    pub op: Op2,
    pub depth: bool,
    pub dim: Dim,
    pub grad: bool,
    /// The coordinates are 16-bit integers (`.i16`).
    pub short: bool,
    pub elem: Elem,
    /// A write's rounding mode for floating-point pixel types: `.rte` (`Some(true)`) or `.rtz` (`Some(false)`), from
    /// `-ftexture-write-rounding-mode` (MSL 4.1 section 1.6.7); `None` is the hardware's native rounding.
    pub round: Option<bool>,
}

pub(crate) fn parse(name: &str) -> Option<Name> {
    let rest = name.strip_prefix("air.")?;
    let (base, suffix) = rest.split_once('.').map(|(b, s)| (b, Some(s))).unwrap_or((rest, None));
    const OPS: [(&str, Op2); 13] = [
        ("sample_compare_", Op2::SampleCompare),
        ("sample_", Op2::Sample),
        ("gather_compare_", Op2::GatherCompare),
        ("gather_", Op2::Gather),
        ("read_", Op2::Read),
        ("write_", Op2::Write),
        ("get_width_", Op2::Width),
        ("get_height_", Op2::Height),
        ("get_depth_", Op2::Depth),
        ("get_array_size_", Op2::ArraySize),
        ("get_num_mip_levels_", Op2::MipLevels),
        ("get_num_samples_", Op2::Samples),
        ("is_null_", Op2::IsNull),
    ];
    let (op, rest) = OPS.iter().find_map(|(p, op)| base.strip_prefix(p).map(|r| (*op, r)))?;
    let (depth, rest) = match rest.strip_prefix("texture_") {
        Some(r) => (false, r),
        None => (true, rest.strip_prefix("depth_")?),
    };
    let (dims, grad) = match rest.strip_suffix("_grad") {
        Some(d) => (d, true),
        None => (rest, false),
    };
    let dim = match dims {
        "1d" => Dim::D1,
        "1d_array" => Dim::D1Array,
        "2d" => Dim::D2,
        "2d_array" => Dim::D2Array,
        "2d_ms" => Dim::D2Ms,
        "2d_ms_array" => Dim::D2MsArray,
        "3d" => Dim::D3,
        "cube" => Dim::Cube,
        "cube_array" => Dim::CubeArray,
        "buffer_1d" => Dim::Buffer,
        _ => return None,
    };
    if grad && op != Op2::Sample && op != Op2::SampleCompare {
        return None;
    }
    let mut short = false;
    let mut signed: Option<bool> = None;
    let mut elem = Elem::None;
    let mut round = None;
    for tok in suffix.map(|s| s.split('.').collect::<Vec<_>>()).unwrap_or_default() {
        match tok {
            "rte" | "rtz"
                if op == Op2::Write && round.is_none() && !short && elem == Elem::None && signed.is_none() =>
            {
                round = Some(tok == "rte")
            }
            "i16" if elem == Elem::None && signed.is_none() => short = true,
            "s" | "u" if elem == Elem::None && signed.is_none() => signed = Some(tok == "s"),
            "v4f32" | "f32" if signed.is_none() && elem == Elem::None => elem = Elem::F32,
            "v4f16" if signed.is_none() && elem == Elem::None => elem = Elem::F16,
            "v4i32" if elem == Elem::None => elem = Elem::I32(signed?),
            "v4i16" if elem == Elem::None => elem = Elem::I16(signed?),
            _ => return None,
        }
    }
    Some(Name { op, depth, dim, grad, short, elem, round })
}

// Refusals ------------------------------------------------------------------------------------------------------------

fn refuse(id: &str, construct: &str, detail: impl Into<String>) -> Refusal {
    tracked::refuse(id, construct.to_string(), None, detail)
}

/// A setting the A10X rejects, with the device profile getter that says so.
fn above_family(construct: &str, what: &str, getter: &str) -> Refusal {
    refuse(
        "texture-above-family",
        construct,
        format!("{}: the A10X does not support it (WebMetal device profile, {})", what, getter),
    )
}

// Constants -----------------------------------------------------------------------------------------------------------

/// The value of an integer constant operand, sign-extended from its width.
pub(crate) fn const_int(m: &Module, v: Value) -> Option<i64> {
    let Value::Const(c) = v else { return None };
    let k = m.constant(c);
    let bits = match m.types.get(k.ty) {
        Type::Int(b) => *b,
        _ => return None,
    };
    match &k.kind {
        ConstKind::Int(w) => {
            let raw = *w.first()? as i64;
            Some(if bits >= 64 { raw } else { (raw << (64 - bits)) >> (64 - bits) })
        }
        ConstKind::Zero => Some(0),
        _ => None,
    }
}

/// The value of a `float` constant operand.
pub(crate) fn const_f32(m: &Module, v: Value) -> Option<f32> {
    let Value::Const(c) = v else { return None };
    let k = m.constant(c);
    if !matches!(m.types.get(k.ty), Type::Float) {
        return None;
    }
    match &k.kind {
        ConstKind::Float(b) => Some(f32::from_bits(*b as u32)),
        ConstKind::Zero => Some(0.0),
        _ => None,
    }
}

/// The lanes of an integer vector (or scalar) constant operand.
pub(crate) fn const_ivec(m: &Module, v: Value) -> Option<Vec<i32>> {
    let Value::Const(c) = v else { return None };
    let k = m.constant(c);
    let (len, elem) = match m.types.get(k.ty) {
        Type::Vector { len, elem, .. } => (*len as usize, *elem),
        Type::Int(_) => return const_int(m, v).map(|x| vec![x as i32]),
        _ => return None,
    };
    let bits = match m.types.get(elem) {
        Type::Int(b) => *b,
        _ => return None,
    };
    let sext = |raw: u64| -> i32 { ((raw << (64 - bits)) as i64 >> (64 - bits)) as i32 };
    match &k.kind {
        ConstKind::Zero => Some(vec![0; len]),
        ConstKind::DataVector(e) => Some(e.iter().map(|x| sext(*x)).collect()),
        ConstKind::Vector(vals) => vals.iter().map(|x| const_int(m, *x).map(|y| y as i32)).collect(),
        _ => None,
    }
}

/// The global a constant pointer operand names, through constant casts.
fn global_of(m: &Module, v: Value) -> Option<u32> {
    match v {
        Value::Global(GlobalId::Var(g)) => Some(g),
        Value::Const(c) => match &m.constant(c).kind {
            ConstKind::Expr(ConstExpr::Cast { value, .. }) => global_of(m, *value),
            _ => None,
        },
        _ => None,
    }
}

/// The `air.sampler_state` word of a constant sampler operand: a constant `i64` global the module lists in
/// `air.sampler_states`.
fn sampler_state(m: &Module, v: Value) -> Option<u64> {
    let g = global_of(m, v)?;
    let gv = &m.globals[g as usize];
    use crate::ir::Metadata;
    let names_g = |md: &Metadata| matches!(md, Metadata::Value { value, .. } if global_of(m, *value) == Some(g));
    let listed = m.named_metadata.iter().any(|nm| {
        nm.name == "air.sampler_states"
            && nm.ops.iter().any(|&op| match &m.metadata[op.0 as usize] {
                Metadata::Tuple { ops, .. } => ops.iter().flatten().any(|o| names_g(&m.metadata[o.0 as usize])),
                _ => false,
            })
    });
    if !listed || !gv.constant || !matches!(m.types.get(gv.value_type), Type::Int(64)) {
        return None;
    }
    let init = gv.initializer?;
    const_int(m, init).map(|x| x as u64)
}

// Arguments -----------------------------------------------------------------------------------------------------------

/// The component type a texture argument is declared with (`air.arg_type_name`, `texture2d<float, sample>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Component {
    Float,
    Sint,
    Uint,
}

/// A texture operand: the argument and what is bound at its slot.
#[derive(Clone, Debug)]
pub(crate) struct Tex {
    pub arg: u32,
    pub index: u64,
    pub access: Access,
    pub component: Component,
    pub slot: Option<TextureSlot>,
    /// The bound format and its storage on the host.
    pub format: Option<(&'static format::Format, Stored)>,
}

fn component_of(type_name: Option<&str>, depth: bool) -> Component {
    if depth {
        return Component::Float;
    }
    let t = type_name.unwrap_or("");
    let arg = t.split_once('<').map(|(_, r)| r.split([',', '>']).next().unwrap_or("").trim()).unwrap_or("");
    match arg {
        "int" | "short" | "char" => Component::Sint,
        "uint" | "ushort" | "uchar" => Component::Uint,
        _ => Component::Float,
    }
}

/// The texture argument an operand is.
pub(crate) fn texture(cx: &dyn RuleCx, name: &Name, construct: &str, v: Value) -> R<Tex> {
    let Some(arg) = cx.handle_arg(v) else {
        return Err(refuse(
            "texture-dynamic-handle",
            construct,
            "the texture operand is not one texture argument (a select or phi of textures, an array element or a \
             null texture handle)",
        ));
    };
    let Some(rec) = cx.arg_record(arg) else {
        return Err(refuse("entry-metadata", construct, "a texture operand without an argument record"));
    };
    let Role::Texture { location, access, .. } = rec.role else {
        return Err(refuse("entry-metadata", construct, "the texture operand's argument is not a texture"));
    };
    let index = match (location.index, location.count) {
        (Slot::Value(i), Slot::Value(1)) => i as u64,
        (Slot::Value(_), Slot::Value(_)) => {
            return Err(refuse("texture-dynamic-handle", construct, "an array of textures"));
        }
        _ => return Err(tracked::refuse_family("function-constants", construct.to_string(), None)),
    };
    let component = component_of(rec.type_name.as_deref(), name.depth);
    let slot = cx.bindings().and_then(|b| b.textures.iter().find(|t| t.index == index)).copied();
    let format = match &slot {
        None => None,
        Some(s) => {
            let f = format::format(s.pixel_format).ok_or_else(|| {
                refuse(
                    "texture-slot-mismatch",
                    construct,
                    format!("the bound pixel format {} is not one a shader can bind on the A10X", s.pixel_format),
                )
            })?;
            let stored = format::stored(s.pixel_format, cx.host()).map_err(|u| match u {
                Unavailable::NotProvided(why) => refuse("texture-slot-mismatch", construct, why),
                Unavailable::Host(feature) => refuse(
                    "texture-host-format",
                    construct,
                    format!("the bound format {} needs the WebGPU feature {}", f.name, feature),
                ),
            })?;
            Some((f, stored))
        }
    };
    let t = Tex { arg, index, access, component, slot, format };
    check_slot(&t, name, construct)?;
    Ok(t)
}

/// The bound texture agrees with the argument's type: Metal's validation layer rejects a draw that binds another
/// texture type or a format of another class, and without it such an access is undefined.
fn check_slot(t: &Tex, name: &Name, construct: &str) -> R<()> {
    let (Some(slot), Some((f, _))) = (&t.slot, &t.format) else { return Ok(()) };
    if slot.mipmap_level_count == 0 {
        return Err(refuse("texture-slot-mismatch", construct, "a bound texture of no mipmap levels"));
    }
    let want = name.dim.texture_type();
    // A 2D argument also takes a 2D view of one face or slice; Metal requires the view's type to match.
    if slot.texture_type != want {
        return Err(refuse(
            "texture-slot-mismatch",
            construct,
            format!("the bound texture's MTLTextureType {} is not the argument's {}", slot.texture_type, want),
        ));
    }
    let ok = match (f.class, t.component, name.depth) {
        (Class::Depth, _, true) => true,
        (_, _, true) => false,
        // MSL 4.1 section 2.9: a float texture's data is a normalized, half or single-precision format, which the
        // depth formats are (WebMetal formats.toml question 5.5-depth-texture-as-color).
        (Class::Float | Class::Depth, Component::Float, false) => true,
        (Class::Sint, Component::Sint, false) => true,
        (Class::Uint | Class::Stencil, Component::Uint, false) => true,
        _ => false,
    };
    if !ok {
        return Err(refuse(
            "texture-slot-mismatch",
            construct,
            format!("the bound format {} does not have the argument's texel type", f.name),
        ));
    }
    Ok(())
}

/// A sampler operand.
#[derive(Clone, Debug)]
pub(crate) struct Smp {
    pub source: BindingSource,
    /// The descriptor: the bound sampler's, a constant sampler's, or `None` when no sampler is bound.
    pub desc: Option<SamplerSlot>,
}

pub(crate) fn sampler(cx: &dyn RuleCx, construct: &str, v: Value) -> R<Smp> {
    let stage = cx.stage();
    if let Some(arg) = cx.handle_arg(v) {
        let Some(rec) = cx.arg_record(arg) else {
            return Err(refuse("entry-metadata", construct, "a sampler operand without an argument record"));
        };
        let Role::Sampler(location) = rec.role else {
            return Err(refuse("entry-metadata", construct, "the sampler operand's argument is not a sampler"));
        };
        let index = match (location.index, location.count) {
            (Slot::Value(i), Slot::Value(1)) => i as u64,
            (Slot::Value(_), Slot::Value(_)) => {
                return Err(refuse("texture-dynamic-handle", construct, "an array of samplers"));
            }
            _ => return Err(tracked::refuse_family("function-constants", construct.to_string(), None)),
        };
        let desc = cx.bindings().and_then(|b| b.samplers.iter().find(|s| s.index == index)).copied();
        return Ok(Smp { source: BindingSource::MetalSampler { stage, index }, desc });
    }
    let Some(word) = sampler_state(cx.module(), v) else {
        return Err(refuse(
            "texture-dynamic-handle",
            construct,
            "the sampler operand is not one sampler argument or constant sampler",
        ));
    };
    // A word the target's Metal fails to decode (a reserved bit set by a newer setting, such as a reduction mode) is a
    // sampler the A10X does not support.
    let d = crate::air::sampler::decode(word).map_err(|r| {
        refuse(
            "texture-above-family",
            construct,
            format!("{} (setDescriptorFromBitmask in the target's Metal fails on it)", r.reason),
        )
    })?;
    let half = |b: u16| half_to_f32(b);
    Ok(Smp {
        source: BindingSource::ConstantSampler { encoding: word },
        desc: Some(SamplerSlot {
            index: u64::MAX,
            min_filter: d.min_filter,
            mag_filter: d.mag_filter,
            mip_filter: d.mip_filter,
            address_modes: d.address_modes,
            border_color: d.border_color,
            normalized_coordinates: d.normalized_coordinates,
            compare_function: d.compare_function,
            max_anisotropy: d.max_anisotropy,
            lod_min_clamp: half(d.lod_min_clamp),
            lod_max_clamp: half(d.lod_max_clamp),
            lod_average: false,
        }),
    })
}

/// IEEE binary16 bits to binary32 (exact).
fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = match e {
        0 if m == 0 => s,
        0 => {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
        0x1f => s | 0x7f80_0000 | (m << 13),
        _ => s | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Settings of a bound sampler the A10X rejects.
pub(crate) fn check_sampler(d: &SamplerSlot, axes: usize, construct: &str) -> R<()> {
    for &m in &d.address_modes[..axes] {
        match m {
            1 => {
                return Err(above_family(
                    construct,
                    "the mirrorClampToEdge address mode",
                    "getters.supportsMirrorClampToEdgeSamplerMode = false",
                ));
            }
            5 => {
                return Err(above_family(
                    construct,
                    "the clampToBorderColor address mode",
                    "getters.supportsBlackOrWhiteSamplerBorderColors and isCustomBorderColorSupported = false",
                ));
            }
            0 | 2..=4 => {}
            _ => return Err(refuse("texture-slot-mismatch", construct, format!("address mode {}", m))),
        }
    }
    if d.min_filter > 1 || d.mag_filter > 1 || d.mip_filter > 2 {
        return Err(refuse(
            "texture-filter-mode",
            construct,
            format!(
                "filter codes min {} mag {} mip {}: a filter with no MTLSamplerDescriptor enumerator (MSL \
                 filter::bicubic writes code 2)",
                d.min_filter, d.mag_filter, d.mip_filter
            ),
        ));
    }
    Ok(())
}

// Bindings ------------------------------------------------------------------------------------------------------------

/// The WGSL texture types of a binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum View {
    /// A sampled texture: dimension, sample type, multisampled.
    Sampled(TextureViewDimension, TextureSampleType, bool),
    /// A write-only storage texture of a `GPUTextureFormat`.
    Storage(TextureViewDimension, &'static str),
}

pub(crate) fn wgsl_type(v: View) -> String {
    use TextureSampleType as S;
    use TextureViewDimension as D;
    match v {
        View::Sampled(d, S::Depth, ms) => match (d, ms) {
            (D::D2, true) => "texture_depth_multisampled_2d".into(),
            (D::D2, false) => "texture_depth_2d".into(),
            (D::D2Array, _) => "texture_depth_2d_array".into(),
            (D::Cube, _) => "texture_depth_cube".into(),
            _ => "texture_depth_cube_array".into(),
        },
        View::Sampled(d, s, ms) => {
            let t = match s {
                S::Sint => "i32",
                S::Uint => "u32",
                _ => "f32",
            };
            let base = match (d, ms) {
                (D::D2, true) => "texture_multisampled_2d",
                (D::D1, _) => "texture_1d",
                (D::D2, _) => "texture_2d",
                (D::D2Array, _) => "texture_2d_array",
                (D::Cube, _) => "texture_cube",
                (D::CubeArray, _) => "texture_cube_array",
                (D::D3, _) => "texture_3d",
            };
            format!("{}<{}>", base, t)
        }
        View::Storage(d, f) => {
            let base = match d {
                D::D2Array => "texture_storage_2d_array",
                D::D3 => "texture_storage_3d",
                _ => "texture_storage_2d",
            };
            format!("{}<{}, write>", base, f)
        }
    }
}

/// Declare the texture's binding with the given view; returns the WGSL variable. A storage view is one level: level
/// `level` of the bound texture, counted from a view's first level (WebGPU 8.2.1: a storage binding's view has one
/// level); a sampled view shows every level of the bound texture, and `level` is 0.
pub(crate) fn bind_texture(cx: &mut dyn RuleCx, t: &Tex, view: View, level: u32) -> R<String> {
    let stage = cx.stage();
    let (resource, source) = match view {
        View::Sampled(d, s, ms) => (
            BindingResource::Texture { sample_type: s, view_dimension: d, multisampled: ms },
            BindingSource::MetalTexture { stage, index: t.index },
        ),
        View::Storage(d, f) => (
            BindingResource::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: f.to_string(),
                view_dimension: d,
            },
            BindingSource::MetalTextureLevel { stage, index: t.index, level },
        ),
    };
    debug_assert!(level == 0 || matches!(view, View::Storage(..)));
    let ty = wgsl_type(view);
    cx.resource(resource, source, &ty)
}

/// The mipmap levels a function addresses in the texture: the bound texture's (a view's own), or `None` when no texture
/// is bound at the slot.
pub(crate) fn levels(t: &Tex) -> Option<u64> {
    t.slot.map(|s| s.mipmap_level_count)
}

pub(crate) fn bind_sampler(cx: &mut dyn RuleCx, s: &Smp, ty: SamplerBindingType) -> R<String> {
    let wgsl = if ty == SamplerBindingType::Comparison { "sampler_comparison" } else { "sampler" };
    cx.resource(BindingResource::Sampler { ty }, s.source.clone(), wgsl)
}

/// The native binding type of a sampler used for plain sampling.
pub(crate) fn plain_sampler_type(s: &Smp) -> SamplerBindingType {
    match &s.desc {
        Some(d) if d.min_filter == 1 || d.mag_filter == 1 || d.mip_filter == 2 => SamplerBindingType::Filtering,
        _ => SamplerBindingType::NonFiltering,
    }
}

// Uses ----------------------------------------------------------------------------------------------------------------

/// How the function uses a texture or sampler handle, over all its texture calls: one binding serves every use, so
/// uses that need different WebGPU bindings of one slot are refused.
#[derive(Default)]
pub(crate) struct Uses {
    /// The texture is read or written texel by texel.
    pub texel: bool,
    /// The texture is sampled or gathered.
    pub sampled: bool,
    /// The sampler is used by a compare function.
    pub compare: bool,
    /// The sampler is used by another sampling function.
    pub plain: bool,
}

/// The uses of the texture argument `arg` (when `texture`) or of the sampler `source`.
pub(crate) fn uses(cx: &dyn RuleCx, arg: Option<u32>, source: Option<&BindingSource>) -> Uses {
    let m = cx.module();
    let f = &m.functions[cx.function()];
    let mut u = Uses::default();
    for inst in &f.insts {
        let Op::Call { callee: Value::Global(GlobalId::Function(c)), args, .. } = &inst.op else { continue };
        let Some(n) = parse(&m.functions[*c as usize].common.name) else { continue };
        if let (Some(a), Some(&tv)) = (arg, args.first())
            && cx.handle_arg(tv) == Some(a)
        {
            match n.op {
                Op2::Read | Op2::Write => u.texel = true,
                Op2::Sample | Op2::SampleCompare | Op2::Gather | Op2::GatherCompare => u.sampled = true,
                _ => {}
            }
        }
        if let (Some(src), Some(&sv)) = (source, args.get(1))
            && matches!(n.op, Op2::Sample | Op2::SampleCompare | Op2::Gather | Op2::GatherCompare)
        {
            let same = match src {
                BindingSource::MetalSampler { .. } => cx
                    .handle_arg(sv)
                    .and_then(|a| cx.arg_record(a))
                    .is_some_and(|r| matches!((&r.role, src), (Role::Sampler(l), BindingSource::MetalSampler { index, .. }) if l.index == Slot::Value(*index as u32))),
                BindingSource::ConstantSampler { encoding } => sampler_state(m, sv) == Some(*encoding),
                _ => false,
            };
            if same {
                if matches!(n.op, Op2::SampleCompare | Op2::GatherCompare) {
                    u.compare = true;
                } else {
                    u.plain = true;
                }
            }
        }
    }
    u
}

// Helpers shared by the lowerings ---------------------------------------------------------------------------------

/// The texel class a sampled binding of the texture reads, from the bound format (or the declared type when no
/// texture is bound).
pub(crate) fn sample_type(cx: &dyn RuleCx, t: &Tex, depth_arg: bool) -> TextureSampleType {
    use TextureSampleType as S;
    if depth_arg {
        return S::Depth;
    }
    match &t.format {
        None => match t.component {
            Component::Float => S::Float,
            Component::Sint => S::Sint,
            Component::Uint => S::Uint,
        },
        Some((_, st)) => match st.texel {
            Texel::Uint => S::Uint,
            Texel::Sint => S::Sint,
            Texel::Depth => S::UnfilterableFloat,
            Texel::UnfilterableFloat => S::UnfilterableFloat,
            Texel::Float32 if !st.filterable(cx.host()) => S::UnfilterableFloat,
            Texel::Float | Texel::Float32 => S::Float,
        },
    }
}

/// The format's shim, when a texture is bound.
pub(crate) fn shim(t: &Tex) -> Shim {
    t.format.map(|(_, s)| s.shim).unwrap_or(Shim::None)
}

/// The components the bound format specifies (all when no texture is bound).
pub(crate) fn components(t: &Tex) -> format::Components {
    t.format.map(|(f, _)| f.components).unwrap_or(format::RGBA)
}

/// The stage's kind for implicit LOD: only fragment functions compute one (MSL 4.1 section 6.13).
pub(crate) fn fragment(cx: &dyn RuleCx) -> bool {
    cx.stage() == MetalStage::Fragment
}

/// The result type's carrier of a `{ T, i8 }` texture result.
pub(crate) fn result_struct(cx: &mut dyn RuleCx, ret: TypeId) -> R<String> {
    cx.carrier(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_parse_as_the_front_end_writes_them() {
        let n = parse("air.sample_texture_2d.v4f32").unwrap();
        assert_eq!((n.op, n.depth, n.dim, n.grad, n.elem), (Op2::Sample, false, Dim::D2, false, Elem::F32));
        let n = parse("air.sample_compare_depth_cube_array_grad.f32").unwrap();
        assert_eq!((n.op, n.depth, n.dim, n.grad), (Op2::SampleCompare, true, Dim::CubeArray, true));
        let n = parse("air.read_texture_2d_ms.i16.u.v4i16").unwrap();
        assert_eq!((n.op, n.dim, n.short, n.elem), (Op2::Read, Dim::D2Ms, true, Elem::I16(false)));
        let n = parse("air.write_texture_2d.rtz.v4f32").unwrap();
        assert_eq!((n.op, n.round, n.elem), (Op2::Write, Some(false), Elem::F32));
        assert_eq!(parse("air.write_texture_2d.i16.rte.v4f16").map(|n| n.round), None);
        assert_eq!(parse("air.write_texture_2d.rte.i16.v4f16").map(|n| n.round), Some(Some(true)));
        assert_eq!(parse("air.write_texture_2d.rte.i16.u.v4i32").map(|n| n.elem), Some(Elem::I32(false)));
        assert!(parse("air.read_texture_2d.rte.v4f32").is_none());
        let n = parse("air.write_texture_buffer_1d.s.v4i32").unwrap();
        assert_eq!((n.op, n.dim, n.elem), (Op2::Write, Dim::Buffer, Elem::I32(true)));
        assert_eq!(parse("air.get_num_mip_levels_texture_1d").unwrap().op, Op2::MipLevels);
        assert_eq!(parse("air.is_null_depth_2d_ms").unwrap().op, Op2::IsNull);
        assert!(parse("air.gather_texture_2d_grad.v4f32").is_none());
        assert!(parse("air.sample_texture_4d.v4f32").is_none());
        assert!(parse("air.get_num_samples").is_none());
        assert!(parse("air.read_texture_2d.v4i32").is_none());
    }

    #[test]
    fn the_rule_claims_the_texture_families_only() {
        let r = Textures;
        assert!(r.handles("air.sample_texture_2d.v4f32"));
        assert!(r.handles("air.get_width_texture_2d"));
        assert!(r.handles("air.is_null_texture_2d"));
        assert!(r.handles("air.gather_compare_depth_2d.f32"));
        assert!(!r.handles("air.get_null_texture_2d"));
        assert!(!r.handles("air.fence_texture_2d"));
        assert!(!r.handles("air.calculate_clamped_lod_texture_2d"));
        assert!(!r.handles("air.get_num_samples.i32"));
        assert!(r.claims_argument("air.texture") && r.claims_argument("air.sampler"));
        assert_eq!(tracked::compiler_item("texture-above-family").kind, RefusalKind::RejectedByDevice);
        assert!(!r.claims_argument("air.buffer"));
    }

    #[test]
    fn binary16_widens_exactly() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0x7bff), 65504.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(half_to_f32(0x4b80), 15.0);
        assert!(half_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn wgsl_types() {
        use TextureSampleType as S;
        use TextureViewDimension as D;
        assert_eq!(wgsl_type(View::Sampled(D::D2, S::Float, false)), "texture_2d<f32>");
        assert_eq!(wgsl_type(View::Sampled(D::D2, S::Uint, true)), "texture_multisampled_2d<u32>");
        assert_eq!(wgsl_type(View::Sampled(D::D2Array, S::Depth, false)), "texture_depth_2d_array");
        assert_eq!(wgsl_type(View::Storage(D::D2Array, "rgba8unorm")), "texture_storage_2d_array<rgba8unorm, write>");
    }
}
