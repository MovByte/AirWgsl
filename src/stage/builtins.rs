//! Built-in values that need more than a direct WGSL built-in.

use super::{from_vec3, loc};
use crate::air::Argument;
use crate::api::{PipelineKind, Refusal, UniformValue};
use crate::ir::{ConstKind, Module, Op, TypeId, Value};
use crate::wgsl::hooks::{OutputField, StageCx};
use crate::wgsl::tracked;

type R<T> = Result<T, Refusal>;

/// `[[grid_origin]]` and `[[grid_size]]` of a kernel with stage-in data (MSL 5.2.3.6, Table 5.8): the origin and
/// size of the stage-in region the dispatch sets (`setStageInRegion:`), which the executor passes in the uniform block
/// (measured on Metal: the values read are the region's).
pub fn stage_in_region(cx: &mut dyn StageCx, arg: u32, record: &Argument) -> R<String> {
    let m = cx.module();
    let fi = cx.function();
    let key = crate::wgsl::io::role_key(&record.role);
    let value = if key == "air.stage_in_grid_origin" { UniformValue::StageInOrigin } else { UniformValue::StageInSize };
    let t = m.functions[fi].args[arg as usize].ty;
    let words = cx.uniform_words(value);
    let w: [String; 3] = [words[0].clone(), words[1].clone(), words[2].clone()];
    from_vec3(cx.module(), &w, t).ok_or_else(|| {
        tracked::refuse("stage-builtin", key, loc(cx.module(), fi), "an unexpected type for the built-in")
    })
}

/// `[[thread_index_in_threadgroup]]`: MSL 5.2.3.6 defines it as `ly * Sx + lx` for a 2D threadgroup; for a 3D
/// threadgroup real Metal gives `(lz * Sy + ly) * Sx + lx` (measured on an Apple M4 for 4x2x2, 3x5x2 and 2x2x8
/// threadgroups), which is WGSL's `local_invocation_index` (WGSL 13.3.1.1) in every case.
pub fn thread_index(cx: &mut dyn StageCx, arg: u32) -> R<String> {
    let m = cx.module();
    let fi = cx.function();
    let t = m.functions[fi].args[arg as usize].ty;
    let bits = m.types.int_bits(t);
    let name = cx.builtin("local_invocation_index", "u32");
    match bits {
        Some(32) => Ok(name),
        Some(16) => Ok(format!("({} & 0xffffu)", name)),
        _ => Err(tracked::refuse(
            "stage-builtin",
            "air.thread_index_in_threadgroup",
            loc(cx.module(), fi),
            "an unexpected type for the built-in",
        )),
    }
}

/// `[[thread_position_in_grid]]` and `[[thread_position_in_threadgroup]]` of a kernel (MSL 5.2.3.6: the position in
/// the grid is the threadgroup's position times the threadgroup size plus the position in the threadgroup). In a
/// dimension where the pipeline variant's threadgroup size is 1, the position in the threadgroup is 0 and the position
/// in the grid is the threadgroup's position, so those components are WGSL's `workgroup_id` and the constant 0
/// (WGSL 13.3.1.1 defines global_invocation_id the same way): the values are the same, and WGSL's uniformity analysis
/// sees that they are uniform across the workgroup, which barriers under branches on them need.
pub fn kernel_position(cx: &mut dyn StageCx, arg: u32, record: &Argument) -> R<String> {
    let key = crate::wgsl::io::role_key(&record.role);
    let ws = match &cx.desc().kind {
        PipelineKind::Compute(c) => c.threads_per_threadgroup,
        _ => unreachable!("a kernel is lowered in a compute pipeline"),
    };
    let grid = key == "air.thread_position_in_grid";
    let lanes: Vec<String> = (0..3)
        .map(|d| {
            let c = ["x", "y", "z"][d];
            match (ws[d] == 1, grid) {
                (true, true) => format!("{}.{}", cx.builtin("workgroup_id", "vec3<u32>"), c),
                (true, false) => "0u".to_string(),
                (false, true) => format!("{}.{}", cx.builtin("global_invocation_id", "vec3<u32>"), c),
                (false, false) => format!("{}.{}", cx.builtin("local_invocation_id", "vec3<u32>"), c),
            }
        })
        .collect();
    let fi = cx.function();
    let t = cx.module().functions[fi].args[arg as usize].ty;
    let (n, bits) = match cx.module().types.get(t) {
        crate::ir::Type::Vector { len, elem, .. } => (*len as usize, cx.module().types.int_bits(*elem)),
        _ => (1, cx.module().types.int_bits(t)),
    };
    if n > 3 || !matches!(bits, Some(16 | 32)) {
        return Err(tracked::refuse("stage-builtin", key, loc(cx.module(), fi), "an unexpected type for the built-in"));
    }
    let lanes: Vec<String> =
        lanes[..n].iter().map(|x| if bits == Some(16) { format!("({} & 0xffffu)", x) } else { x.clone() }).collect();
    if n == 1 {
        return Ok(lanes[0].clone());
    }
    // A lane a branch reads keeps its own uniformity (the workgroup_id lanes are uniform).
    cx.argument_lanes(arg, lanes.clone());
    Ok(format!("vec{}<u32>({})", n, lanes.join(", ")))
}

/// The refusal for `[[point_coord]]`: the position inside a point needs the point expansion.
pub fn point_coord(cx: &mut dyn StageCx) -> Refusal {
    tracked::refuse(
        "point-size",
        "air.point_coord",
        loc(cx.module(), cx.function()),
        "[[point_coord]] needs the points drawn as quads (plan 5.7), which needs Metal's point coverage rule \
         (limits.toml question 5.7-point-rasterization) and an executor pass plan",
    )
}

/// Whether the draws of this pipeline variant can be points.
fn draws_points(cx: &dyn StageCx) -> bool {
    match &cx.desc().kind {
        PipelineKind::Render(r) => match r.input_primitive_topology {
            1 => true,
            0 => matches!(r.primitive_type, None | Some(0)),
            _ => false,
        },
        _ => false,
    }
}

/// The value returned in member `index` of the return value `v`, through `insertvalue` and constant structs.
fn returned_member(m: &Module, fi: usize, v: Value, index: u32) -> Option<Value> {
    match v {
        Value::Inst(i) => match &m.functions[fi].inst(i).op {
            Op::InsertValue { agg, value, indices } => match indices.as_slice() {
                [k] if *k == index => Some(*value),
                [_] => returned_member(m, fi, *agg, index),
                _ => None,
            },
            _ => None,
        },
        Value::Const(c) => match &m.constant(c).kind {
            ConstKind::Struct(members) => members.get(index as usize).copied(),
            _ => None,
        },
        _ => None,
    }
}

/// `[[point_size]]` (MSL 5.2.3.3, Table 5.4). Line and triangle draws do not use it. WebGPU rasterizes a point as the
/// one pixel containing it (WebGPU 23.2.5.1). A Metal point of size 1 sampled once per pixel covers the pixel whose
/// center lies within half a pixel of it in both axes, which is that pixel; MSL does not define the point's shape
/// (the V1 case raster.points takes coverage as exact only where shapes agree), and plan 5.7 and limits.toml question
/// 5.7-point-rasterization take points of size 1 as native. So the output is dropped when every return writes the
/// constant 1.0 and the pipeline has one sample per pixel; any other size, and points under multisampling (where a
/// Metal point covers samples of the pixels around it), need the quad expansion and are refused.
pub fn point_size(cx: &mut dyn StageCx, index: usize, _ty: TypeId) -> R<Option<OutputField>> {
    if !draws_points(cx) {
        return Ok(None);
    }
    let m = cx.module();
    let fi = cx.function();
    let refuse = |detail: &str| tracked::refuse("point-size", "air.point_size", loc(m, fi), detail.to_string());
    let samples = match &cx.desc().kind {
        PipelineKind::Render(r) => r.raster_sample_count,
        _ => 1,
    };
    if samples > 1 {
        return Err(refuse("points drawn with more than one sample per pixel need the quad expansion (plan 5.7)"));
    }
    let f = &m.functions[fi];
    let mut returns = 0;
    for inst in &f.insts {
        if let Op::Ret { value: Some(v) } = &inst.op {
            returns += 1;
            let size = returned_member(m, fi, *v, index as u32);
            let one = size.is_some_and(|s| match s {
                Value::Const(c) => matches!(m.constant(c).kind, ConstKind::Float(b) if b == 0x3f80_0000),
                _ => false,
            });
            if !one {
                return Err(refuse(
                    "a point size that is not the constant 1.0 needs the quad expansion (plan 5.7), which needs Metal's \
                     point coverage rule (limits.toml question 5.7-point-rasterization) and an executor pass plan",
                ));
            }
        }
    }
    if returns == 0 {
        return Err(refuse("no return writes the point size"));
    }
    Ok(None)
}
