//! Framebuffer fetch (plan 5.9): a fragment function's color inputs (`[[color(m)]]`, AIR `air.render_target`
//! arguments) as reads of snapshot textures, and the pass plan that keeps the snapshots current.
//!
//! # What Metal defines
//!
//! A color input reads the current value of the fragment's own pixel in color attachment `m`, or of its own sample
//! for a multisampled attachment, including the values earlier primitives of the same draw wrote (MSL 4.1 section
//! 5.2.3.4; D2 question `5.9-fetch-per-sample`). The A10X supports it (D1 `supportsProgrammableBlending = true`) and
//! reports no raster order groups (D1 `supportsRasterOrderGroups = false`); a color input in a raster order group is
//! refused with the function's other raster order groups ([`crate::stage::fragment::raster_order_groups`]).
//!
//! The value a color input of type `T` reads from an attachment of pixel format `F` was measured on real Metal (Apple
//! M4, macOS 27.0.1) for every pixel of a 16 x 16 attachment filled with arbitrary texels, for 12 formats x 14 types
//! and for the 16-bit integer types on 5 more formats (`tests/fetch/probe.rs` repeats it against the lowering):
//!
//! - The class of `T` (float and half; int and short; uint and ushort) must be the class of `F`, else pipeline
//!   creation fails with "input of type T is not compatible with a F color attachment" (21H461's `MTLCompiler` holds
//!   the same "%s of type %s is not compatible with a %s color attachment" message). `T` may have fewer or more
//!   components than `F`.
//! - The components `F` has read as the format's value converted to `T`: floats of normalized and float formats are
//!   the correctly rounded binary32 value, halves the correctly rounded binary16 value of the format's exact value;
//!   integers are the stored integer, and a 16-bit integer type saturates a wider value (an `R32Uint` texel
//!   0x7a55300b reads as `ushort` 0xffff).
//! - Components `F` lacks read 0, except alpha, which reads 1: what WebGPU's `textureLoad` returns for them (WebGPU
//!   26.1 texel conversion).
//! - sRGB formats read the linear value through Metal's sRGB conversion, whose precision MSL 8.7.7 bounds (the value
//!   converted back must lie within 0.5 of the code); the M4 differs from the exact formula by up to a few percent
//!   at small codes, inside that bound. WebGPU's sRGB conversion is held to the same bound by the tests.
//! - In a multisampled pass each sample reads its own value: the function runs once per group of samples with equal
//!   contents (M4), which for the color result is per-sample execution, while the varyings stay interpolated where
//!   per-pixel execution interpolates them ([D2 `5.9-fetch-per-sample`]).
//!
//! # Lowering
//!
//! Each fetched attachment gets a binding of [`BindingSource::FetchSnapshot`]: a texture of the attachment's stored
//! WebGPU format and sample count (`texture_2d<T>` or `texture_multisampled_2d<T>`), read with `textureLoad` at the
//! fragment's pixel (`@builtin(position)`, whose `x` and `y` lie inside the pixel) and, when multisampled, at
//! `@builtin(sample_index)`. Using `sample_index` makes WebGPU run the function once per sample (WebGPU 23.2.10);
//! varyings keep their own sampling qualifiers, as on Metal. The texel becomes the input's type by the rules above;
//! WebGPU's conversions of normalized and float formats to `f32` are the exact values Metal's are (checked over
//! every code by the tests), and a binary16 lane rounds the binary32 value to nearest even, which equals rounding the
//! format's exact value because binary32 has at least 2 x 11 + 2 significand bits.
//!
//! Formats the executor stores in another WebGPU format with a shader conversion ([`crate::tex::format::Shim`]) are
//! refused: their snapshot holds the stored texels.
//!
//! # Pass plan
//!
//! A pipeline whose fragment stage reads a snapshot gets [`PassPlan::PerPrimitive`] naming the fetched attachments,
//! with the overlap prover of [`overlap`] when its vertex function qualifies.

pub mod overlap;

use crate::air::{Argument, Role, Slot};
use crate::api::{
    BindingLayout, BindingResource, BindingSource, MetalStage, PassPlan, PipelineDesc, PipelineKind, Refusal,
    TextureSampleType, TextureViewDimension,
};
use crate::ir::{Module, Type, Value};
use crate::stage::fragment::{Class, color_format};
use crate::tex::format::{Shim, Texel, stored};
use crate::wgsl::hooks::{StageCx, StageRule};
use crate::wgsl::tracked;

type R<T> = Result<T, Refusal>;

/// The stage rule lowering color inputs.
pub struct Fetch;

impl StageRule for Fetch {
    fn handles_argument(&self, stage: MetalStage, kind: &str) -> bool {
        stage == MetalStage::Fragment && kind == "air.render_target"
    }

    fn argument(&self, cx: &mut dyn StageCx, arg: u32, record: &Argument) -> R<String> {
        color_input(cx, arg, record)
    }
}

fn loc(m: &Module, fi: usize) -> Option<crate::api::Location> {
    Some(crate::api::Location { function: m.functions[fi].common.name.clone(), instruction: None })
}

/// The lane types a color input can have (MSL 5.2.3.4 admits `half`, `float`, `short`, `ushort`, `int` and `uint`
/// scalars and vectors; the front end rejects `char` and `uchar`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    F32,
    F16,
    I32,
    I16,
    U32,
    U16,
}

fn lane_of(m: &Module, el: crate::ir::TypeId, type_name: Option<&str>) -> Option<Lane> {
    let base = type_name.map(|t| t.trim_end_matches(|c: char| c.is_ascii_digit()));
    Some(match (m.types.get(el), base) {
        (Type::Float, _) => Lane::F32,
        (Type::Half, _) => Lane::F16,
        (Type::Int(32), Some("int")) => Lane::I32,
        (Type::Int(32), Some("uint")) => Lane::U32,
        (Type::Int(16), Some("short")) => Lane::I16,
        (Type::Int(16), Some("ushort")) => Lane::U16,
        _ => return None,
    })
}

fn class_of(l: Lane) -> Class {
    match l {
        Lane::F32 | Lane::F16 => Class::Float,
        Lane::I32 | Lane::I16 => Class::SInt,
        Lane::U32 | Lane::U16 => Class::UInt,
    }
}

/// Whether the function's code uses IR argument `arg`.
fn uses(m: &Module, fi: usize, arg: u32) -> bool {
    m.functions[fi].insts.iter().any(|i| i.op.operands().contains(&Value::Arg(arg)))
}

fn raster_samples(desc: &PipelineDesc) -> u64 {
    match &desc.kind {
        PipelineKind::Render(r) => r.raster_sample_count,
        _ => 1,
    }
}

/// Lower color input `arg` (record `record`): its value in the carrier of its type.
// Pipelines are compiled once per variant, so returning the refusal by value costs nothing that matters.
#[allow(clippy::result_large_err)]
fn color_input(cx: &mut dyn StageCx, arg: u32, record: &Argument) -> R<String> {
    let m = cx.module();
    let fi = cx.function();
    let at = loc(m, fi);
    let Role::RenderTarget { index, .. } = &record.role else { unreachable!("air.render_target") };
    let color = match index {
        Slot::Value(v) => *v as u64,
        Slot::Constant(_) => return Err(tracked::refuse_family("function-constants", "air.render_target", at)),
    };
    let construct = format!("color({})", color);
    let ty = m.functions[fi].args[arg as usize].ty;
    let (n, el) = match m.types.get(ty) {
        Type::Vector { len, elem, .. } => (*len, *elem),
        _ => (1, ty),
    };
    let lane = lane_of(m, el, record.type_name.as_deref()).filter(|_| n <= 4).ok_or_else(|| {
        tracked::refuse("stage-builtin", construct.clone(), at.clone(), "a color input type MSL does not admit")
    })?;
    let carrier = if n == 1 { "u32".to_string() } else { format!("vec{}<u32>", n) };
    // An input the function does not read needs no snapshot: Metal's observable behavior is the same with and without
    // it (measured on an Apple M4: the sample groups a multisampled function runs for are those of a function without
    // the input).
    if record.unused || !uses(m, fi, arg) {
        return Ok(format!("{}()", carrier));
    }
    let PipelineKind::Render(rs) = &cx.desc().kind else { unreachable!("a fragment stage of a render pipeline") };
    let format = rs.color_attachments.get(color as usize).map(|a| a.pixel_format).unwrap_or(0);
    let Some((class, comps)) = color_format(format) else {
        return Err(tracked::refuse(
            "fetch-attachment",
            construct,
            at,
            format!("color attachment {} has pixel format {}, which is not a color-renderable format", color, format),
        ));
    };
    if class_of(lane) != class {
        return Err(tracked::refuse(
            "fetch-format-class",
            construct,
            at,
            format!(
                "input of type {} is not compatible with pixel format {} (measured on Metal)",
                record.type_name.as_deref().unwrap_or("?"),
                format
            ),
        ));
    }
    let st = match stored(format, cx.host()) {
        Ok(s) if s.shim == Shim::None => s,
        Ok(s) => {
            return Err(tracked::refuse(
                "fetch-attachment",
                construct,
                at,
                format!(
                    "pixel format {} is stored as {} with the shader conversion {:?}: the snapshot holds the stored \
                     texels",
                    format, s.webgpu, s.shim
                ),
            ));
        }
        Err(why) => {
            return Err(tracked::refuse(
                "fetch-attachment",
                construct,
                at,
                format!("pixel format {} has no WebGPU storage on this host: {:?}", format, why),
            ));
        }
    };
    let samples = raster_samples(cx.desc());
    if samples > 1
        && cx.entry().arguments.iter().any(|a| {
            matches!(a.role, Role::Builtin { key: "air.sample_mask_in", .. }) && !a.unused && uses(m, fi, a.position)
        })
    {
        return Err(tracked::refuse(
            "fetch-sample-mask",
            construct,
            at,
            "a multisampled function reading a color input and [[sample_mask]]",
        ));
    }
    let (sample_type, scalar) = match st.texel {
        Texel::Uint => (TextureSampleType::Uint, "u32"),
        Texel::Sint => (TextureSampleType::Sint, "i32"),
        Texel::Float | Texel::Float32 | Texel::UnfilterableFloat => (TextureSampleType::UnfilterableFloat, "f32"),
        Texel::Depth => unreachable!("a color-renderable format reads as color"),
    };
    let multisampled = samples > 1;
    let wgsl_type =
        if multisampled { format!("texture_multisampled_2d<{}>", scalar) } else { format!("texture_2d<{}>", scalar) };
    let snapshot = cx.resource(
        BindingResource::Texture { sample_type, view_dimension: TextureViewDimension::D2, multisampled },
        BindingSource::FetchSnapshot { attachment: color as u32 },
        &wgsl_type,
    )?;
    let position = cx.builtin("position", "vec4<f32>");
    let sample = if multisampled { cx.builtin("sample_index", "u32") } else { "0u".to_string() };
    let texel = format!("aw_fetch{}", arg);
    cx.prologue(format!("let {} = textureLoad({}, vec2<u32>({}.xy), {});", texel, snapshot, position, sample));
    let lanes = ["x", "y", "z", "w"];
    let mut parts = Vec::new();
    for (k, c) in lanes.iter().enumerate().take(n as usize) {
        let v = format!("{}.{}", texel, c);
        parts.push(if k as u32 >= comps {
            // Components the format lacks: 0, and 1 for alpha.
            let one = k == 3;
            match lane {
                Lane::F32 => if one { "0x3f800000u" } else { "0u" }.to_string(),
                Lane::F16 => if one { "0x3c00u" } else { "0u" }.to_string(),
                _ => if one { "1u" } else { "0u" }.to_string(),
            }
        } else {
            match lane {
                Lane::F32 => format!("bitcast<u32>({})", v),
                Lane::F16 => {
                    cx.prelude("nm_f32_to_f16");
                    format!("nm_f32_to_f16(bitcast<u32>({}))", v)
                }
                Lane::U32 | Lane::I32 => {
                    if scalar == "u32" {
                        v
                    } else {
                        format!("bitcast<u32>({})", v)
                    }
                }
                Lane::U16 => format!("min({}, 0xffffu)", v),
                Lane::I16 => format!("(bitcast<u32>(clamp({}, -32768i, 32767i)) & 0xffffu)", v),
            }
        });
    }
    Ok(if n == 1 { parts.remove(0) } else { format!("{}({})", carrier, parts.join(", ")) })
}

/// The attachments a compiled layout reads snapshots of, ascending.
pub fn fetched(layout: &BindingLayout) -> Vec<u32> {
    let mut v: Vec<u32> = layout
        .groups
        .iter()
        .flat_map(|g| &g.entries)
        .filter_map(|e| match e.source {
            BindingSource::FetchSnapshot { attachment } => Some(attachment),
            _ => None,
        })
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// The pass plan of a compiled pipeline: [`PassPlan::PerPrimitive`] when the layout reads snapshots, with the
/// overlap prover `prover` gives for the vertex stage; [`PassPlan::Direct`] otherwise.
pub fn plan(layout: &BindingLayout, prover: impl FnOnce() -> Option<overlap::OverlapProver>) -> PassPlan {
    let fetched = fetched(layout);
    if fetched.is_empty() { PassPlan::Direct } else { PassPlan::PerPrimitive { fetched, overlap: prover() } }
}

/// Color attachments for lowering a fragment function on its own (corpus checks, [`crate::wgsl::compile_entry`]):
/// each attachment the function reads gets a 32-bit format of its input's class (float, signed or unsigned integer)
/// with as many components as the function's output to it has (1, 2 or 4, at most the output's lane count; 4 without
/// an output), which Metal accepts for that input and output (`src/fetch/mod.rs`, `src/stage/fragment.rs`). Other
/// indices have no attachment.
pub fn corpus_attachments(m: &Module, entry: &crate::air::EntryPoint) -> Vec<crate::api::ColorAttachment> {
    use crate::air::OutputRole;
    let fi = entry.function as usize;
    let lanes = |t: crate::ir::TypeId| match m.types.get(t) {
        Type::Vector { len, .. } => *len,
        _ => 1,
    };
    let ret = m.types.function(m.functions[fi].ty).map(|(r, _, _)| r);
    let members: Vec<crate::ir::TypeId> = match ret.map(|r| (r, m.types.get(r))) {
        Some((_, Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. })) => elems.clone(),
        Some((r, _)) => vec![r],
        None => Vec::new(),
    };
    let mut out: Vec<crate::api::ColorAttachment> = Vec::new();
    for a in &entry.arguments {
        let Role::RenderTarget { index: Slot::Value(color), .. } = a.role else { continue };
        let ty = m.functions[fi].args[a.position as usize].ty;
        let el = match m.types.get(ty) {
            Type::Vector { elem, .. } => *elem,
            _ => ty,
        };
        let Some(lane) = lane_of(m, el, a.type_name.as_deref()) else { continue };
        let written = entry.outputs.iter().enumerate().find_map(|(k, o)| match o.role {
            OutputRole::RenderTarget { index: Slot::Value(i), blend_index: 0 } if i == color => {
                members.get(k).map(|t| lanes(*t))
            }
            _ => None,
        });
        let comps = match written.unwrap_or(4) {
            1 => 0,
            2 | 3 => 1,
            _ => 2,
        };
        let pixel_format = match class_of(lane) {
            Class::Float => [55, 105, 125][comps],
            Class::UInt => [53, 103, 123][comps],
            Class::SInt => [54, 104, 124][comps],
        };
        let i = color as usize;
        if out.len() <= i {
            out.resize(i + 1, crate::api::ColorAttachment { pixel_format: 0, ..unblended(0) });
        }
        out[i] = unblended(pixel_format);
    }
    out
}

fn unblended(pixel_format: u64) -> crate::api::ColorAttachment {
    crate::api::ColorAttachment {
        pixel_format,
        blending_enabled: false,
        rgb_blend_operation: 0,
        alpha_blend_operation: 0,
        source_rgb_blend_factor: 1,
        destination_rgb_blend_factor: 0,
        source_alpha_blend_factor: 1,
        destination_alpha_blend_factor: 0,
        write_mask: 0xf,
    }
}
