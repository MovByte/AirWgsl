//! Fragment-stage interface: color outputs and their attachment formats, dual-source outputs, the coverage mask,
//! `discard_fragment`, the sample count and positions, and early fragment tests.

use super::loc;
use crate::air::{EntryPoint, Interpolation, Output, OutputRole, Role, Sampling, Slot};
use crate::api::{ColorAttachment, PipelineDesc, PipelineKind, Refusal, RefusalKind};
use crate::ir::{GlobalId, Module, Op, Type, TypeId, Value};
use crate::wgsl::hooks::{CallRule, CallSite, OutputField, RuleCx, StageCx};
use crate::wgsl::tracked;

type R<T> = Result<T, Refusal>;

/// The class of values a color output or attachment holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Float,
    SInt,
    UInt,
}

/// Metal's rule for a color output and its attachment's `MTLPixelFormat`: the class the output's type must have and
/// the components it must have. Measured on real Metal (Apple M4, macOS 27.0.1): the pipeline is refused with "output
/// of type ... is not compatible with a ... color attachment" for another class and "fragment shader color output does
/// not have enough components for the pixel format" (for A8Unorm: "requires 4 outputs") for fewer components. The
/// same rule holds in WebGPU (validating GPUFragmentState). `None` for a format that is not color renderable.
pub fn color_format(pixel_format: u64) -> Option<(Class, u32)> {
    use Class::*;
    Some(match pixel_format {
        1 => (Float, 4),
        10 | 11 | 12 | 20 | 22 | 25 | 55 => (Float, 1),
        13 | 23 | 53 => (UInt, 1),
        14 | 24 | 54 => (SInt, 1),
        30 | 31 | 32 | 60 | 62 | 65 | 105 => (Float, 2),
        33 | 63 | 103 => (UInt, 2),
        34 | 64 | 104 => (SInt, 2),
        40 | 92 | 93 | 554 | 555 => (Float, 3),
        41 | 42 | 43 | 70 | 71 | 72 | 80 | 81 | 90 | 94 | 110 | 112 | 115 | 125 | 552 | 553 => (Float, 4),
        73 | 91 | 113 | 123 => (UInt, 4),
        74 | 114 | 124 => (SInt, 4),
        _ => return None,
    })
}

/// `MTLBlendFactor` values that read the second source (`Source1Color`, `OneMinusSource1Color`, `Source1Alpha`,
/// `OneMinusSource1Alpha`).
fn uses_source1(f: u64) -> bool {
    (15..=18).contains(&f)
}

/// Whether an attachment's blending reads the second source.
fn blends_source1(a: &ColorAttachment) -> bool {
    a.blending_enabled
        && [
            a.source_rgb_blend_factor,
            a.destination_rgb_blend_factor,
            a.source_alpha_blend_factor,
            a.destination_alpha_blend_factor,
        ]
        .iter()
        .any(|f| uses_source1(*f))
}

/// Whether an attachment's color blend factors read the source alpha (`SourceAlpha`, `OneMinusSourceAlpha`,
/// `SourceAlphaSaturated`, `Source1Alpha`, `OneMinusSource1Alpha`): WebGPU then requires a 4-component output.
fn blends_source_alpha(a: &ColorAttachment) -> bool {
    a.blending_enabled
        && [a.source_rgb_blend_factor, a.destination_rgb_blend_factor].iter().any(|f| matches!(f, 4 | 5 | 10 | 17 | 18))
}

fn color_attachments(desc: &PipelineDesc) -> &[ColorAttachment] {
    match &desc.kind {
        PipelineKind::Render(r) => &r.color_attachments,
        _ => &[],
    }
}

fn msl_class(m: &Module, el: TypeId, type_name: Option<&str>) -> Option<Class> {
    match m.types.get(el) {
        Type::Float | Type::Half => Some(Class::Float),
        Type::Int(8 | 16 | 32) => {
            let base = type_name?.trim_end_matches(|c: char| c.is_ascii_digit());
            match base {
                "int" | "short" | "char" => Some(Class::SInt),
                "uint" | "ushort" | "uchar" => Some(Class::UInt),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A color output (`[[color(m)]]`, `[[color(m), index(i)]]`, MSL 5.2.3.5 Table 5.7) as `@location(m)`, checked against
/// the attachment's format. A second-source output (`index(1)`) becomes `@blend_src(1)` at location 0 when the
/// attachment's blending reads the second source (WGSL `dual_source_blending`, WebGPU feature `dual-source-blending`),
/// and is not declared when nothing reads it.
pub fn color_output(cx: &mut dyn StageCx, record: &Output, ty: TypeId) -> R<Option<OutputField>> {
    let m = cx.module();
    let fi = cx.function();
    let at = loc(m, fi);
    let OutputRole::RenderTarget { index: slot, blend_index } = &record.role else { unreachable!("air.render_target") };
    let (slot, blend_index) = (*slot, *blend_index);
    let color = match slot {
        Slot::Value(v) => v as u64,
        Slot::Constant(_) => return Err(tracked::refuse_family("function-constants", "air.render_target", at)),
    };
    let construct = format!("color({}), index({})", color, blend_index);
    let (n, el) = match m.types.get(ty) {
        Type::Vector { len, elem, .. } => (*len, *elem),
        _ => (1, ty),
    };
    let class = msl_class(m, el, record.type_name.as_deref()).filter(|_| n <= 4).ok_or_else(|| {
        tracked::refuse("stage-builtin", construct.clone(), at.clone(), "an output type without a WGSL form")
    })?;
    // The lane type: binary32, binary16, or an integer of the width.
    let lane_bits = match m.types.get(el) {
        Type::Float => 0,
        Type::Half => 1,
        Type::Int(b) => *b,
        _ => unreachable!("msl_class admits float, half and integers"),
    };
    let attachments = color_attachments(cx.desc()).to_vec();
    let attachment = attachments.get(color as usize).filter(|a| a.pixel_format != 0).copied();
    if let Some(a) = attachment
        && let Some((want, comps)) = color_format(a.pixel_format)
    {
        if want != class {
            return Err(tracked::refuse(
                "color-output-format",
                construct,
                at.clone(),
                format!(
                    "a {:?} output is not compatible with pixel format {} (measured on Metal)",
                    class, a.pixel_format
                ),
            ));
        }
        if n < comps {
            return Err(tracked::refuse(
                "color-output-format",
                construct,
                at.clone(),
                format!(
                    "{} components for pixel format {}, which needs {} (measured on Metal)",
                    n, a.pixel_format, comps
                ),
            ));
        }
        if n < 4 && blends_source_alpha(&a) {
            return Err(tracked::refuse(
                "color-output-alpha",
                construct,
                at.clone(),
                "blend factors read the source alpha of an output without an alpha component: WebGPU rejects the \
                 pipeline, and Metal states no alpha for it",
            ));
        }
    }
    // Dual-source blending: in use when attachment 0 blends with the second source.
    let second = cx.entry().outputs.iter().any(|o| matches!(o.role, OutputRole::RenderTarget { blend_index: 1, .. }));
    let dual = second && attachments.first().is_some_and(blends_source1);
    if blend_index != 0 && !dual {
        // Nothing reads the second source.
        return Ok(None);
    }
    let mut location = format!("@location({})", color);
    if dual {
        if color != 0 || blend_index > 1 {
            return Err(tracked::refuse_row(
                "dual-source-blending",
                RefusalKind::NoExactLowering,
                construct,
                at.clone(),
                "WGSL blend_src outputs are only at location 0, and a pipeline that blends with the second source has \
                 no other color output",
                "WebGPU 10.3.1 validating GPUFragmentState",
            ));
        }
        if !cx.host().features.contains("dual-source-blending") {
            return Err(tracked::refuse_row(
                "dual-source-blending",
                RefusalKind::HostUnsupported,
                construct,
                at.clone(),
                "the host lacks the dual-source-blending feature",
                "WGSL 4.1.1 dual_source_blending",
            ));
        }
        cx.require_feature("dual-source-blending");
        cx.enable("dual_source_blending");
        location = format!("@location(0) @blend_src({})", blend_index);
    }
    let lane = |i: u32| if n == 1 { "{}".to_string() } else { format!("{{}}[{}]", i) };
    let vec = |s: &str| if n == 1 { s.to_string() } else { format!("vec{}<{}>", n, s) };
    let (scalar, value) = match (lane_bits, class) {
        (0, _) => ("f32", format!("bitcast<{}>({{}})", vec("f32"))),
        (1, _) => {
            // Widening binary16 to binary32 is exact.
            cx.prelude("nm_f16_to_f32");
            let parts: Vec<String> = (0..n).map(|i| format!("bitcast<f32>(nm_f16_to_f32({}))", lane(i))).collect();
            ("f32", if n == 1 { parts[0].clone() } else { format!("{}({})", vec("f32"), parts.join(", ")) })
        }
        (b, Class::SInt) => {
            if b != 32 {
                cx.prelude("nm_sext");
            }
            let parts: Vec<String> = (0..n)
                .map(|i| {
                    if b == 32 { format!("bitcast<i32>({})", lane(i)) } else { format!("nm_sext({}, {}u)", lane(i), b) }
                })
                .collect();
            ("i32", if n == 1 { parts[0].clone() } else { format!("{}({})", vec("i32"), parts.join(", ")) })
        }
        _ => ("u32", "{}".to_string()),
    };
    Ok(Some(OutputField { decl: format!("{} {{name}}: {}", location, vec(scalar)), value }))
}

/// Whether the fragment function runs once per sample in both APIs: it reads `[[sample_id]]` or a sample-interpolated
/// input (MSL 5.4; WebGPU 23.2.10).
fn per_sample(e: &EntryPoint) -> bool {
    e.arguments.iter().any(|a| match &a.role {
        Role::Builtin { key: "air.sample_id", .. } => !a.unused,
        Role::FragmentInput {
            interpolation: Some(Interpolation::Smooth { sampling: Sampling::Sample, .. }), ..
        } => !a.unused,
        _ => false,
    })
}

fn raster_samples(desc: &PipelineDesc) -> u64 {
    match &desc.kind {
        PipelineKind::Render(r) => r.raster_sample_count,
        _ => 1,
    }
}

/// `[[sample_mask]]` input (MSL Table 5.5: the samples covered by the primitive). Real Metal gives the whole coverage
/// mask, also when the function runs per sample (measured: 0xf and 0x5 for full and half-covered pixels, with and
/// without `[[sample_id]]`). WGSL's `sample_mask` input is the coverage mask when the shader runs per pixel; when it
/// runs per sample WGSL 13.3.1.1 lets it be either the coverage mask or the one sample's bit. With one sample per
/// pixel the two are the same.
pub fn sample_mask_in(cx: &mut dyn StageCx, arg: u32) -> R<String> {
    let m = cx.module();
    let fi = cx.function();
    if per_sample(cx.entry()) && raster_samples(cx.desc()) > 1 {
        return Err(tracked::refuse(
            "sample-mask-input",
            "air.sample_mask_in",
            loc(m, fi),
            "the coverage mask of a function that runs per sample: WGSL lets it be the one sample's bit, Metal gives \
             the whole mask",
        ));
    }
    if m.types.int_bits(m.functions[fi].args[arg as usize].ty) != Some(32) {
        return Err(tracked::refuse("stage-builtin", "air.sample_mask_in", loc(m, fi), "an unexpected type"));
    }
    cx.require_feature("core-features-and-limits");
    Ok(cx.builtin("sample_mask", "u32"))
}

/// `[[sample_mask]]` output (MSL Table 5.7): the coverage mask is ANDed with it, as WebGPU 23.2.11 does with the
/// `sample_mask` output. WebGPU refuses a `sample_mask` output with alpha to coverage, whose combination Metal does not
/// state.
pub fn sample_mask_out(cx: &mut dyn StageCx, ty: TypeId) -> R<Option<OutputField>> {
    let m = cx.module();
    let fi = cx.function();
    if let PipelineKind::Render(r) = &cx.desc().kind
        && r.alpha_to_coverage_enabled
    {
        return Err(tracked::refuse(
            "sample-mask-output",
            "air.sample_mask",
            loc(m, fi),
            "a sample mask output with alpha to coverage: WebGPU rejects the pipeline (validating \
             GPURenderPipelineDescriptor)",
        ));
    }
    if m.types.int_bits(ty) != Some(32) {
        return Err(tracked::refuse("stage-builtin", "air.sample_mask", loc(m, fi), "an unexpected type"));
    }
    cx.require_feature("core-features-and-limits");
    Ok(Some(OutputField { decl: "@builtin(sample_mask) {name}: u32".into(), value: "{}".into() }))
}

/// The package's call rule: `discard_fragment` and the sample queries.
pub struct FragmentCalls;

/// Metal's default sample positions (`Metal.framework`'s `__defaultSamplePositions_N` tables, plan 6.4): WebGPU's
/// standard pattern for 4 samples (limits.toml question 6.4-sample-positions-4x) and the pixel center for 1.
fn sample_positions(count: u64) -> Option<&'static [(f32, f32)]> {
    match count {
        1 => Some(&[(0.5, 0.5)]),
        2 => Some(&[(0.75, 0.75), (0.25, 0.25)]),
        4 => Some(&[(0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)]),
        _ => None,
    }
}

impl CallRule for FragmentCalls {
    fn handles(&self, name: &str) -> bool {
        matches!(name, "air.discard_fragment" | "air.get_num_samples.i32" | "air.get_sample_position.v2f32")
    }

    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> Result<Option<String>, Refusal> {
        let fi = cx.function();
        let function = cx.module().functions[fi].common.name.clone();
        let at = || Some(crate::api::Location { function: function.clone(), instruction: None });
        match call.name {
            // MSL 6.11.1.3: the fragment's output and its later buffer and texture writes are discarded; from MSL 2.3
            // the thread keeps executing for derivatives, which is WGSL's discard (demotion to a helper invocation,
            // WGSL 9.4.6). For modules of MSL 2.2 and earlier, where the thread terminates and derivatives after a
            // partial discard are undefined, the demotion is a valid refinement (intrinsics.toml question
            // 5.6-discard).
            "air.discard_fragment" => {
                cx.line("discard;".into());
                Ok(None)
            }
            _ => {
                // The trailing operand is the attachment the query is about; the front end passes 0.
                let zero = match call.args.last() {
                    Some(Value::Const(c)) => {
                        matches!(&cx.module().constant(*c).kind, crate::ir::ConstKind::Int(w) if w.first() == Some(&0))
                    }
                    _ => false,
                };
                if !zero {
                    return Err(tracked::refuse(
                        "stage-builtin",
                        call.name,
                        at(),
                        "a sample query whose attachment operand is not the constant 0 the front end passes",
                    ));
                }
                let count = match cx.desc().map(|d| &d.kind) {
                    Some(PipelineKind::Render(r)) => r.raster_sample_count,
                    _ => {
                        return Err(tracked::refuse(
                            "stage-builtin",
                            call.name,
                            at(),
                            "a sample query outside a render pipeline",
                        ));
                    }
                };
                if call.name == "air.get_num_samples.i32" {
                    // MSL Table 6.20: the sample count of the multisampled color attachment, the pipeline's raster
                    // sample count.
                    return Ok(Some(format!("{}u", count)));
                }
                // MSL Table 6.20: the sample's position in the pixel; the executor refuses passes with programmed
                // positions (refusal custom-sample-positions), so these are the defaults. An index outside the
                // pattern has no defined result; it reads the last position.
                let positions = sample_positions(count).ok_or_else(|| {
                    tracked::refuse("stage-builtin", call.name, at(), format!("a sample count of {}", count))
                })?;
                let s = cx.operand(call.args[0])?;
                let t = cx.fresh();
                let table: Vec<String> =
                    positions.iter().map(|(x, y)| format!("vec2<u32>({}u, {}u)", x.to_bits(), y.to_bits())).collect();
                cx.line(format!(
                    "let {} = array<vec2<u32>, {}>({})[min({}, {}u)];",
                    t,
                    positions.len(),
                    table.join(", "),
                    s,
                    positions.len() - 1
                ));
                Ok(Some(t))
            }
        }
    }
}

/// Whether the function calls an `air.*` function with an effect outside the fragment's own outputs, or discards.
fn effects(m: &Module, fi: usize) -> Option<String> {
    let f = &m.functions[fi];
    for inst in &f.insts {
        if let Op::Call { callee: Value::Global(GlobalId::Function(c)), .. } = &inst.op {
            let name = &m.functions[*c as usize].common.name;
            if name == "air.discard_fragment" || name.starts_with("air.write_") || name.starts_with("air.atomic") {
                return Some(name.clone());
            }
        }
    }
    None
}

/// `[[early_fragment_tests]]` (MSL 5.1.2): the depth and stencil tests run before the fragment function, which may
/// then output no depth or stencil value (MSL 5.1.2 makes either an error, which is refused as on the device). WebGPU gives
/// no control over when they run, but they decide the same results when the function can neither change the coverage
/// nor have effects beyond its outputs: the samples that pass and their depth, stencil and color writes are then the
/// same whether the tests run before or after it. A function that discards, writes buffers or textures, uses atomics,
/// outputs a coverage mask, or runs with alpha to coverage needs the emulation of plan 5.7 (the comparison in
/// the shader against a fetched copy of depth under the serialization of plan 5.9), which is not implemented.
pub fn early_fragment_tests(m: &Module, fi: usize, e: &EntryPoint, desc: &PipelineDesc, buffer_writes: bool) -> R<()> {
    let refuse = |what: String| {
        tracked::refuse(
            "early-fragment-tests",
            "early_fragment_tests",
            loc(m, fi),
            format!("early fragment tests with {}: the order of the tests decides the result", what),
        )
    };
    if buffer_writes {
        return Err(refuse("buffer writes".into()));
    }
    if let Some(name) = effects(m, fi) {
        return Err(refuse(format!("a call of {}", name)));
    }
    for o in &e.outputs {
        match o.role {
            OutputRole::Depth(_) | OutputRole::Builtin("air.stencil") => {
                return Err(tracked::refuse(
                    "early-fragment-tests-output",
                    "early_fragment_tests",
                    loc(m, fi),
                    "a fragment function with early fragment tests that outputs depth or stencil",
                ));
            }
            OutputRole::Builtin("air.sample_mask") => return Err(refuse("a coverage mask output".into())),
            _ => {}
        }
    }
    if let PipelineKind::Render(r) = &desc.kind
        && r.alpha_to_coverage_enabled
    {
        return Err(refuse("alpha to coverage".into()));
    }
    Ok(())
}

/// Raster order groups (`[[raster_order_group(n)]]` on a fragment function's color outputs, color inputs, textures
/// and device buffers; AIR `air.raster_order_group`). The A10X reports none (D1
/// `supportsRasterOrderGroups = false`), and what its driver does with a function that declares them, refuse the
/// pipeline or build it without the ordering, is decided by its compiler back end (AGXCompilerCore in
/// MTLCompilerService), which only the real compiler service can show (WebMetal packages G4 and G5). A WebGPU draw
/// orders no overlapping fragments, so a lowering would drop the ordering the function asks for: the function is
/// refused, as WebMetal's executor refuses the pipeline.
pub fn raster_order_groups(m: &Module, fi: usize, e: &EntryPoint) -> R<()> {
    let outputs = e.outputs.iter().filter_map(|o| o.raster_order_group.map(|g| ("a color output", g)));
    let arguments = e.arguments.iter().filter_map(|a| match &a.role {
        Role::RenderTarget { raster_order_group: Some(g), .. } => Some(("a color input", *g)),
        Role::Texture { raster_order_group: Some(g), .. } => Some(("a texture", *g)),
        Role::Buffer(b) | Role::IndirectBuffer(b) => b.raster_order_group.map(|g| ("a buffer", g)),
        _ => None,
    });
    match outputs.chain(arguments).next() {
        Some((what, group)) => Err(tracked::refuse(
            "raster-order-group",
            format!("raster_order_group({})", group),
            loc(m, fi),
            format!(
                "{} in a raster order group: the A10X reports none, its driver's answer for such a function is its \
                 compiler back end's (WebMetal packages G4 and G5), and a WebGPU draw has no fragment ordering",
                what
            ),
        )),
        None => Ok(()),
    }
}
