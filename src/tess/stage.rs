//! Post-tessellation vertex functions (MSL 5.1.1.1): the vertex function of a tessellation pipeline, run once per
//! point the tessellator generates, as a WGSL vertex function over the point records of [`super::kernel`].
//!
//! The draw is indexed by the kernels' indices, so `vertex_index` selects the point's record: its position in the
//! patch (u and v in 16.16 fixed point), its patch identifier and its instance identifier. The function's inputs:
//!
//! - `[[position_in_patch]]`: u and v, and w = 1 - u - v on a triangle, each the exact binary32 value of its fixed-point
//!   coordinate (the M4 reports these exact values, V1 `tessellation.*`); the record's integers are converted by
//!   integer arithmetic.
//! - `[[patch_id]]`: the patch's identifier: the patch index buffer's element for the patch, else the draw's patch
//!   index (`patchStart` onward), measured (V1 `tessellation.draws`).
//! - `[[instance_id]]`: the instance, base instance included; `[[base_instance]]`: the draw's base instance.
//! - `[[stage_in]]` per-patch attributes (`air.patch_input`): fetched from their buffer at the patch identifier, with
//!   the vertex descriptor's layout of step function `MTLVertexStepFunctionPerPatch`.
//! - `patch_control_point<T>` (`air.patch_control_point_input`): control point `i` of the patch is element `patch_id *
//!   N + i` of buffers with step function `MTLVertexStepFunctionPerPatchControlPoint`, `N` the patch's control-point
//!   count; for a pipeline with a `tessellationControlPointIndexType` (drawn with `drawIndexedPatches`), the element
//!   the control-point index buffer holds at that position (16- or 32-bit, from the draw's offset), measured with
//!   patch index buffers, patch starts and both index types. The function reads a control point through a call of its
//!   control-point function (`air.patch_control_point_function`), which [`ControlPoints`] lowers to fetches of the
//!   attributes in the function's result order; `air.get_num_patch_control_points` is `N`.
//!
//! Real Metal requires a step rate of 1 with the patch step functions (its vertex descriptor validation asserts
//! "stepRate(...) must be one if stepFunction is MTLVertexStepFunctionPerPatch", measured). Other step functions for
//! these attributes have not been measured and are refused as not implemented.

use crate::air::{Argument, PatchKind, Role};
use crate::api::{
    BindingResource, BindingSource, BufferBindingType, Location as At, PipelineKind, Refusal, Tessellation,
    UniformValue, VertexDescriptor,
};
use crate::ir::Type;
use crate::stage::fetch::{self, attribute_type, element};
use crate::wgsl::hooks::{CallRule, CallSite, RuleCx, StageCx};
use crate::wgsl::tracked;

type R<T> = Result<T, Refusal>;

/// The private variable holding the point's record.
const RECORD: &str = "aw_tess_record";
/// The private variable holding the patch's control-point count.
const COUNT: &str = "aw_tess_count";

fn at(cx: &dyn StageCx) -> Option<At> {
    crate::stage::loc(cx.module(), cx.function())
}

fn not_implemented(construct: impl Into<String>, at: Option<At>, reason: impl Into<String>) -> Refusal {
    tracked::refuse("post-tessellation-input", construct, at, reason)
}

/// The pipeline's tessellation state, which a post-tessellation vertex function needs.
fn tessellation(cx: &dyn StageCx) -> R<Tessellation> {
    match &cx.desc().kind {
        PipelineKind::Render(r) => r.tessellation.ok_or_else(|| {
            tracked::refuse(
                "pipeline-kind",
                "air.patch",
                at(cx),
                "a post-tessellation vertex function in a pipeline without tessellation state",
            )
        }),
        _ => Err(tracked::refuse(
            "pipeline-kind",
            "air.patch",
            at(cx),
            "a post-tessellation function outside a render pipeline",
        )),
    }
}

/// Start the interface of a post-tessellation vertex function: the point's record and the patch's control-point
/// count, read once into private variables the arguments and the control-point fetches use.
pub fn begin(cx: &mut dyn StageCx) -> R<()> {
    tessellation(cx)?;
    let name = cx.resource(
        BindingResource::Buffer {
            ty: BufferBindingType::ReadOnlyStorage,
            has_dynamic_offset: false,
            min_binding_size: 16,
        },
        BindingSource::TessellationVertices,
        "array<u32>",
    )?;
    let vi = cx.builtin("vertex_index", "u32");
    cx.module_item(RECORD, format!("var<private> {}: vec4<u32>;\n", RECORD));
    cx.prologue(format!(
        "{r} = vec4<u32>({n}[4u * {vi}], {n}[4u * {vi} + 1u], {n}[4u * {vi} + 2u], {n}[4u * {vi} + 3u]);",
        r = RECORD,
        n = name,
        vi = vi
    ));
    let count = control_point_count(cx);
    cx.module_item(COUNT, format!("var<private> {}: u32;\n", COUNT));
    cx.prologue(format!("{} = {};", COUNT, count));
    cx.module_item(
        "aw_tess_fixed",
        "// The binary32 bits of a 16.16 fixed-point value of at most 1: its integer converts exactly, then the \
         exponent drops by 16.\nfn aw_tess_fixed(x: u32) -> u32 {\n    if (x == 0u) {\n        return 0u;\n    }\n    \
         return bitcast<u32>(f32(x)) - 0x8000000u;\n}\n"
            .into(),
    );
    Ok(())
}

/// Lower an argument of a post-tessellation vertex function, or `None` for one the general interface lowers.
pub fn argument(cx: &mut dyn StageCx, arg: u32, record: &Argument) -> R<Option<String>> {
    let quad = cx.entry().patch.map(|p| p.kind == PatchKind::Quad).unwrap_or(false);
    let (lanes, word) = {
        let m = cx.module();
        let t = m.functions[cx.function()].args[arg as usize].ty;
        let lanes = match m.types.get(t) {
            Type::Vector { len, elem, .. } if matches!(m.types.get(*elem), Type::Float) => *len,
            _ => 0,
        };
        let bits = m.types.int_bits(t);
        (lanes, bits)
    };
    let word = |w: String| match word {
        Some(32) => Some(w),
        Some(16) => Some(format!("({} & 0xffffu)", w)),
        _ => None,
    };
    let e = match &record.role {
        Role::Builtin { key: "air.position_in_patch", .. } => match (quad, lanes) {
            (true, 2) => format!("vec2<u32>(aw_tess_fixed({r}.x), aw_tess_fixed({r}.y))", r = RECORD),
            (false, 3) => format!(
                "vec3<u32>(aw_tess_fixed({r}.x), aw_tess_fixed({r}.y), aw_tess_fixed(0x10000u - {r}.x - {r}.y))",
                r = RECORD
            ),
            _ => {
                return Err(fetch::rejected(
                    "air.position_in_patch",
                    &at(cx),
                    "position_in_patch must be float2 on a quad patch and float3 on a triangle patch (Metal \
                     Programming Guide, Table 12-1)",
                ));
            }
        },
        Role::Builtin { key: k @ ("air.patch_id" | "air.instance_id"), .. } => {
            let lane = if *k == "air.patch_id" { "z" } else { "w" };
            word(format!("{}.{}", RECORD, lane))
                .ok_or_else(|| not_implemented(*k, at(cx), "a built-in of other than 16 or 32 bits"))?
        }
        Role::Builtin { key: "air.base_instance", .. } => {
            let w = cx.uniform(UniformValue::BaseInstance);
            word(w)
                .ok_or_else(|| not_implemented("air.base_instance", at(cx), "a built-in of other than 16 or 32 bits"))?
        }
        Role::PatchInput(location) => patch_input(cx, arg, record, location)?,
        Role::PatchControlPointInput(_) => {
            control_points(cx)?;
            // The argument is only passed to the control-point function, whose calls [`ControlPoints`] lowers.
            "0u".into()
        }
        Role::Builtin { key: k @ ("air.vertex_id" | "air.base_vertex"), .. } => {
            return Err(fetch::rejected(
                *k,
                &at(cx),
                "a post-tessellation vertex function has no vertex identifier (Metal Programming Guide, Table 12-1)",
            ));
        }
        _ => return Ok(None),
    };
    Ok(Some(e))
}

fn descriptor(cx: &dyn StageCx) -> Option<&VertexDescriptor> {
    match &cx.desc().kind {
        PipelineKind::Render(r) => r.vertex_descriptor.as_ref(),
        _ => None,
    }
}

/// The attribute at `attr` with its layout: (format, offset, buffer, stride), checked for `step`.
fn attribute(cx: &dyn StageCx, attr: u64, step: u64, construct: &str) -> R<(u64, u64, u64, u64)> {
    let at = at(cx);
    let Some(vd) = descriptor(cx) else {
        return Err(fetch::rejected(
            construct,
            &at,
            "the function has a [[stage_in]] attribute and the pipeline no vertex descriptor",
        ));
    };
    let a = vd
        .attributes
        .get(attr as usize)
        .copied()
        .filter(|a| a.format != 0)
        .ok_or_else(|| fetch::rejected(construct, &at, "the vertex descriptor does not describe the attribute"))?;
    if a.buffer_index >= 31 {
        return Err(fetch::rejected(construct, &at, format!("buffer index {} is not below 31", a.buffer_index)));
    }
    let layout = vd.layouts.get(a.buffer_index as usize).copied().unwrap_or(crate::api::VertexBufferLayout {
        stride: 0,
        step_function: 1,
        step_rate: 1,
    });
    if layout.step_function != step {
        return Err(not_implemented(
            construct,
            at,
            format!(
                "step function {} for an attribute of a post-tessellation vertex function (only {} is measured)",
                layout.step_function, step
            ),
        ));
    }
    if layout.step_rate != 1 {
        return Err(fetch::rejected(
            construct,
            &at,
            format!("step rate {} with a patch step function, which must be one (measured)", layout.step_rate),
        ));
    }
    let stride = fetch::check_stride(&at, construct, a.buffer_index, layout.stride)?;
    Ok((a.format, a.offset, a.buffer_index, stride))
}

/// A per-patch `[[stage_in]]` attribute.
fn patch_input(cx: &mut dyn StageCx, arg: u32, record: &Argument, location: &crate::air::Location) -> R<String> {
    let at = at(cx);
    let attr = fetch::attribute_index(&at, location, "air.patch_input")?;
    let construct = format!("attribute({})", attr);
    let ty = {
        let m = cx.module();
        attribute_type(m, m.functions[cx.function()].args[arg as usize].ty, record.type_name.as_deref())
    };
    let (scalar, lanes) =
        ty.ok_or_else(|| tracked::refuse("vertex-input", construct.clone(), at.clone(), "an attribute type"))?;
    // MTLVertexStepFunctionPerPatch.
    let (format, offset, buffer, stride) = attribute(cx, attr, 3, &construct)?;
    let fmt = fetch::check_format(&at, &construct, format, scalar, lanes)?;
    let (var, base) = cx.buffer(buffer)?;
    let load = fetch::loader(cx, &var);
    let addr = format!("aw_tess_pa{}", arg);
    cx.prologue(format!("let {} = {} + {}.z * {}u + {}u;", addr, base, RECORD, stride, offset));
    Ok(element(cx, &load, &addr, fmt, scalar, lanes))
}

/// The number of control points of a patch: the function's `[[patch(type, N)]]`, else the draw's.
fn control_point_count(cx: &mut dyn StageCx) -> String {
    match cx.entry().patch.and_then(|p| p.control_points) {
        Some(n) => format!("{}u", n),
        None => cx.uniform(UniformValue::PatchControlPoints),
    }
}

/// Module functions fetching each control-point attribute: `aw_tess_cp<k>(i, p)`, attribute `k` of the function's
/// control-point record for control point `i` of patch `p`.
fn control_points(cx: &mut dyn StageCx) -> R<()> {
    let at = at(cx);
    let Some(cpi) = cx.entry().arguments.iter().find_map(|a| match &a.role {
        Role::PatchControlPointInput(c) => Some(c.clone()),
        _ => None,
    }) else {
        return Ok(());
    };
    let members: Option<Vec<_>> = {
        let m = cx.module();
        let ret = m.types.function(m.functions[cpi.function as usize].ty).map(|f| f.0);
        match ret.map(|r| m.types.get(r)) {
            Some(Type::Struct { body: Some(elems), .. } | Type::LiteralStruct { elems, .. }) => Some(
                elems.iter().zip(&cpi.attributes).map(|(&t, a)| attribute_type(m, t, Some(&a.type_name))).collect(),
            ),
            _ => None,
        }
    };
    let members = members.filter(|m| m.len() == cpi.attributes.len()).ok_or_else(|| {
        not_implemented(
            "air.patch_control_point_function",
            at.clone(),
            "a control-point record that is not a struct of its attributes",
        )
    })?;
    // A pipeline with a control-point index type draws with drawIndexedPatches, one without with drawPatches: the
    // validation layer rejects the other combinations ("tessellationControlPointIndexType of
    // MTLTessellationControlPointIndexTypeUInt16 not valid for drawPatches", measured), whose results are undefined.
    let index_type = match &cx.desc().kind {
        PipelineKind::Render(r) => r.tessellation.map(|t| t.control_point_index_type).unwrap_or(0),
        _ => 0,
    };
    // The element of the control point's attributes.
    let element_index = match index_type {
        0 => format!("p * {} + i", COUNT),
        kind @ (1 | 2) => {
            let name = cx.resource(
                BindingResource::Buffer {
                    ty: BufferBindingType::ReadOnlyStorage,
                    has_dynamic_offset: false,
                    min_binding_size: 4,
                },
                BindingSource::ControlPointIndices,
                "array<u32>",
            )?;
            let offset = cx.uniform(UniformValue::ControlPointIndexOffset);
            let load = fetch::loader(cx, &name);
            if kind == 1 {
                format!("extractBits({}({} + (p * {} + i) * 2u), 0u, 16u)", load, offset, COUNT)
            } else {
                format!("{}({} + (p * {} + i) * 4u)", load, offset, COUNT)
            }
        }
        kind => {
            return Err(fetch::rejected(
                "tessellationControlPointIndexType",
                &at,
                format!("tessellationControlPointIndexType {} is not an MTLTessellationControlPointIndexType", kind),
            ));
        }
    };
    for (k, (a, ty)) in cpi.attributes.iter().zip(members).enumerate() {
        if a.predicate.is_some() {
            return Err(tracked::refuse_family("function-constants", "air.patch_control_point_input", at.clone()));
        }
        let attr = fetch::attribute_index(&at, &a.location, "air.patch_control_point_input")?;
        let construct = format!("attribute({})", attr);
        let (scalar, lanes) =
            ty.ok_or_else(|| tracked::refuse("vertex-input", construct.clone(), at.clone(), "an attribute type"))?;
        // MTLVertexStepFunctionPerPatchControlPoint.
        let (format, offset, buffer, stride) = attribute(cx, attr, 4, &construct)?;
        let fmt = fetch::check_format(&at, &construct, format, scalar, lanes)?;
        let (var, base) = cx.buffer(buffer)?;
        let load = fetch::loader(cx, &var);
        let value = element(cx, &load, "a", fmt, scalar, lanes);
        let carrier = if lanes == 1 { "u32".to_string() } else { format!("vec{}<u32>", lanes) };
        cx.module_item(
            &format!("aw_tess_cp{}", k),
            format!(
                "fn aw_tess_cp{k}(i: u32, p: u32) -> {carrier} {{\n    let a = {base} + ({element_index}) * {stride}u + {offset}u;\n    \
                 return {value};\n}}\n"
            ),
        );
    }
    Ok(())
}

/// The calls of a post-tessellation vertex function's control-point function and of `air.get_num_patch_control_points`.
pub struct ControlPoints;

/// Whether `name` is a control-point function: the front end names it `<record type>.MTL_CONTROL_POINT_FN` and
/// declares it externally defined; the entry's `air.patch_control_point_function` names it.
fn is_control_point_function(name: &str) -> bool {
    name.ends_with(".MTL_CONTROL_POINT_FN")
}

impl CallRule for ControlPoints {
    fn handles(&self, name: &str) -> bool {
        is_control_point_function(name) || name == "air.get_num_patch_control_points"
    }

    fn lower(&self, cx: &mut dyn RuleCx, call: &CallSite<'_>) -> R<Option<String>> {
        let at = Some(At { function: cx.module().functions[cx.function()].common.name.clone(), instruction: None });
        let tessellated =
            cx.desc().is_some_and(|d| matches!(&d.kind, PipelineKind::Render(r) if r.tessellation.is_some()));
        if !tessellated {
            return Err(tracked::refuse(
                "pipeline-kind",
                call.name.to_string(),
                at,
                "a control-point read outside a tessellation pipeline",
            ));
        }
        if call.name == "air.get_num_patch_control_points" {
            return Ok(Some(COUNT.to_string()));
        }
        let index = cx.operand(call.args[0])?;
        let count = match cx.module().types.get(call.ret) {
            Type::Struct { body: Some(elems), .. } | Type::LiteralStruct { elems, .. } => elems.len(),
            _ => return Err(not_implemented(call.name, at, "a control-point function that does not return a struct")),
        };
        let fields: Vec<String> = (0..count).map(|k| format!("aw_tess_cp{}({}, {}.z)", k, index, RECORD)).collect();
        let carrier = cx.carrier(call.ret)?;
        Ok(Some(format!("{}({})", carrier, fields.join(", "))))
    }
}

/// The pipeline state for compiling a post-tessellation vertex function on its own (the corpus checks): integer
/// partitioning at the default maximum factor of 16, no control-point index type, and a vertex descriptor reading the
/// per-patch attributes in their own types from one buffer (`MTLVertexStepFunctionPerPatch`) and the control-point
/// attributes from another (`MTLVertexStepFunctionPerPatchControlPoint`), the two highest indices below 31 no buffer
/// argument uses.
pub fn corpus_state(
    m: &crate::ir::Module,
    e: &crate::air::EntryPoint,
) -> Option<(Tessellation, Option<VertexDescriptor>)> {
    e.patch?;
    let tessellation = Tessellation {
        partition_mode: 1,
        max_tessellation_factor: 16,
        factor_scale_enabled: false,
        factor_format: 0,
        control_point_index_type: 0,
        factor_step_function: 0,
        output_winding_order: 0,
    };
    let fi = e.function as usize;
    let mut used = [false; 31];
    let mut patch: Vec<(u64, &'static fetch::Format)> = Vec::new();
    let mut points: Vec<(u64, &'static fetch::Format)> = Vec::new();
    for a in &e.arguments {
        match &a.role {
            Role::Buffer(b) | Role::IndirectBuffer(b) => {
                if let crate::air::Slot::Value(i) = b.location.index
                    && (i as usize) < used.len()
                {
                    used[i as usize] = true;
                }
            }
            Role::PatchInput(l) => {
                let crate::air::Slot::Value(index) = l.index else { return Some((tessellation, None)) };
                let (s, n) = attribute_type(m, m.functions[fi].args[a.position as usize].ty, a.type_name.as_deref())?;
                patch.push((index as u64, fetch::natural_format(s, n)));
            }
            Role::PatchControlPointInput(cpi) => {
                let ret = m.types.function(m.functions[cpi.function as usize].ty).map(|f| f.0)?;
                let elems = match m.types.get(ret) {
                    Type::Struct { body: Some(elems), .. } | Type::LiteralStruct { elems, .. } => elems.clone(),
                    _ => return Some((tessellation, None)),
                };
                for (t, attr) in elems.iter().zip(&cpi.attributes) {
                    let crate::air::Slot::Value(index) = attr.location.index else { return Some((tessellation, None)) };
                    let (s, n) = attribute_type(m, *t, Some(&attr.type_name))?;
                    points.push((index as u64, fetch::natural_format(s, n)));
                }
            }
            _ => {}
        }
    }
    if patch.is_empty() && points.is_empty() {
        return Some((tessellation, None));
    }
    let mut free = (0..31u64).rev().filter(|i| !used[*i as usize]);
    let (patch_buffer, point_buffer) = (free.next()?, free.next()?);
    let mut vd = VertexDescriptor { layouts: Vec::new(), attributes: Vec::new() };
    for (attrs, buffer, step) in [(patch, patch_buffer, 3), (points, point_buffer, 4)] {
        let mut attrs = attrs;
        attrs.sort_by_key(|a| a.0);
        let mut offset = 0u64;
        for (index, f) in attrs {
            let size = (f.bits / 8 * f.components) as u64;
            let align = (f.bits / 8) as u64;
            offset = offset.div_ceil(align) * align;
            vd.attributes.resize(
                vd.attributes.len().max(index as usize + 1),
                crate::api::VertexAttribute { format: 0, offset: 0, buffer_index: 0 },
            );
            vd.attributes[index as usize] =
                crate::api::VertexAttribute { format: f.value, offset, buffer_index: buffer };
            offset += size;
        }
        vd.layouts.resize(
            vd.layouts.len().max(buffer as usize + 1),
            crate::api::VertexBufferLayout { stride: 0, step_function: 1, step_rate: 1 },
        );
        vd.layouts[buffer as usize] =
            crate::api::VertexBufferLayout { stride: offset.div_ceil(4).max(1) * 4, step_function: step, step_rate: 1 };
    }
    Some((tessellation, Some(vd)))
}
