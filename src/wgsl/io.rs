//! The stage interface: entry-point arguments and results as WGSL entry parameters, bindings and return values.
//!
//! These are the direct mappings, which need no emulation pass: buffer and threadgroup arguments (memory regions),
//! the built-in values with an exact WGSL counterpart, user varyings matched by name, render-target and depth outputs.
//! Everything else (vertex input attributes, point size on point topologies, clip distances, layered rendering,
//! sample masks, dual-source outputs, framebuffer fetch, textures, samplers, argument buffers) is refused with the
//! tracked item of the package that lowers it. Package S6b owns the stage interface and replaces or extends these
//! rules; the mappings here are each justified by the clause cited at the rule.

use super::entry::{Entry, Item, Record};
use super::tracked;
use crate::api::{
    ComputeState, HostCaps, Location, MetalStage, PipelineDesc, PipelineKind, Refusal, RefusalKind, StageBindings,
    UniformValue,
};
use crate::ir::{Module, Type, TypeId};
use crate::layout::Layout;
use crate::mem::{ArgRegion, RegionKind};
use std::collections::{BTreeMap, HashMap};

/// The interpolation of a varying, as WGSL spells it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interp(pub String);

/// A varying: its WGSL type and interpolation, and its location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Varying {
    pub location: u32,
    /// WGSL type (`f32`, `vec4<f32>`, `vec2<i32>`, ...).
    pub wgsl: String,
    /// `@interpolate(...)`, or empty for the WGSL default.
    pub interp: String,
}

/// The varyings of a render pipeline, by name (`user(...)`, `generated(...)`).
pub type Varyings = BTreeMap<String, Varying>;

/// The lowered interface of one entry point.
#[derive(Default)]
pub struct Interface {
    /// WGSL entry parameters.
    pub params: Vec<String>,
    /// Statements binding `aw_aN` (the IR arguments), in argument order.
    pub prologue: Vec<String>,
    /// Memory regions of arguments.
    pub regions: HashMap<u32, ArgRegion>,
    /// The WGSL return type and its struct declaration, when the stage returns values.
    pub ret_type: Option<String>,
    pub ret_decl: String,
    /// Per return member: how the IR value's member becomes the output field (`{}` is the member's carrier).
    pub ret_fields: Vec<Option<String>>,
    /// `@workgroup_size(...)` arguments.
    pub workgroup_size: Option<[u64; 3]>,
    /// The numerics library items the conversions need.
    pub prelude: Vec<&'static str>,
    pub enables: Vec<String>,
    /// Fragment functions that must not have side effects for the lowering to be exact (`early_fragment_tests`).
    pub early_fragment_tests: bool,
}

fn loc(m: &Module, fi: usize) -> Option<Location> {
    Some(Location { function: m.functions[fi].common.name.clone(), instruction: None })
}

/// The scalar kind and lane count of an MSL type name (`float4`, `ushort2`, `half`, `int`, ...).
fn msl_type(name: &str) -> Option<(&'static str, u32)> {
    let digits = name.trim_start_matches(|c: char| c.is_ascii_alphabetic() || c == '_');
    let base = &name[..name.len() - digits.len()];
    let lanes = if digits.is_empty() { 1 } else { digits.parse().ok()? };
    let kind = match base {
        "float" => "float",
        "half" => "half",
        "int" | "short" | "char" => "sint",
        "uint" | "ushort" | "uchar" => "uint",
        "bool" => "bool",
        _ => return None,
    };
    Some((kind, lanes))
}

/// The lanes and element type of an IR scalar or vector type.
fn lanes(m: &Module, t: TypeId) -> (u32, TypeId) {
    match m.types.get(t) {
        Type::Vector { len, elem, .. } => (*len, *elem),
        _ => (1, t),
    }
}

fn vec_of(scalar: &str, n: u32) -> String {
    if n == 1 { scalar.to_string() } else { format!("vec{}<{}>", n, scalar) }
}

/// A `vec3<u32>` built-in (`w`) as the carrier of IR type `t` (uint, uint2, uint3, ushort, ushort2, ushort3).
fn from_vec3(m: &Module, w: &str, t: TypeId) -> Option<String> {
    let (n, e) = lanes(m, t);
    let bits = m.types.int_bits(e)?;
    let sw = match n {
        1 => format!("{}.x", w),
        2 => format!("{}.xy", w),
        3 => w.to_string(),
        _ => return None,
    };
    match bits {
        32 => Some(sw),
        16 => Some(if n == 1 { format!("({} & 0xffffu)", sw) } else { format!("({} & vec{}<u32>(0xffffu))", sw, n) }),
        _ => None,
    }
}

/// A `u32` built-in as the carrier of IR type `t` (uint or ushort).
fn from_u32(m: &Module, w: &str, t: TypeId) -> Option<String> {
    match m.types.int_bits(t)? {
        32 => Some(w.to_string()),
        16 => Some(format!("({} & 0xffffu)", w)),
        _ => None,
    }
}

/// The WGSL type, interpolation and carrier conversions of a varying of IR type `t`, MSL type name `tn`.
struct VaryingType {
    wgsl: String,
    integer: bool,
    /// Output field from the carrier (`{}`).
    out: String,
    /// Carrier from the input parameter (`{}`).
    inp: String,
}

fn varying_type(m: &Module, t: TypeId, tn: Option<&str>, prelude: &mut Vec<&'static str>) -> Option<VaryingType> {
    let (n, e) = lanes(m, t);
    if n > 4 {
        return None;
    }
    let lane_out = |f: &dyn Fn(&str) -> String| -> String {
        if n == 1 {
            f("{}")
        } else {
            let parts: Vec<String> = (0..n).map(|i| f(&format!("{{}}[{}]", i))).collect();
            format!("{}({})", "VEC", parts.join(", "))
        }
    };
    match m.types.get(e) {
        Type::Float => Some(VaryingType {
            wgsl: vec_of("f32", n),
            integer: false,
            out: format!("bitcast<{}>({{}})", vec_of("f32", n)),
            inp: format!("bitcast<{}>({{}})", vec_of("u32", n)),
        }),
        Type::Half => {
            // Widening binary16 to binary32 is exact; the fragment side rounds back to nearest even.
            prelude.push("nm_f16_to_f32");
            prelude.push("nm_f32_to_f16");
            let out = lane_out(&|x| format!("bitcast<f32>(nm_f16_to_f32({}))", x)).replace("VEC", &vec_of("f32", n));
            let inp = lane_out(&|x| format!("nm_f32_to_f16(bitcast<u32>({}))", x)).replace("VEC", &vec_of("u32", n));
            Some(VaryingType { wgsl: vec_of("f32", n), integer: false, out, inp })
        }
        // bool varyings: WGSL has no bool user-defined IO, so they travel as flat 0 or 1.
        Type::Int(1) => {
            let out = lane_out(&|x| format!("select(0u, 1u, {})", x)).replace("VEC", &vec_of("u32", n));
            let inp = lane_out(&|x| format!("({} != 0u)", x)).replace("VEC", &vec_of("bool", n));
            Some(VaryingType { wgsl: vec_of("u32", n), integer: true, out, inp })
        }
        Type::Int(b @ (8 | 16 | 32)) => {
            let signed = matches!(tn.and_then(msl_type), Some(("sint", _)));
            let mask = if *b == 32 { None } else { Some((1u64 << b) - 1) };
            if signed {
                prelude.push("nm_sext");
                let out = lane_out(&|x| {
                    if *b == 32 { format!("bitcast<i32>({})", x) } else { format!("nm_sext({}, {}u)", x, b) }
                })
                .replace("VEC", &vec_of("i32", n));
                let inp = lane_out(&|x| match mask {
                    Some(mk) => format!("(bitcast<u32>({}) & {}u)", x, mk),
                    None => format!("bitcast<u32>({})", x),
                })
                .replace("VEC", &vec_of("u32", n));
                Some(VaryingType { wgsl: vec_of("i32", n), integer: true, out, inp })
            } else {
                let inp = match mask {
                    Some(mk) => lane_out(&|x| format!("({} & {}u)", x, mk)).replace("VEC", &vec_of("u32", n)),
                    None => "{}".to_string(),
                };
                Some(VaryingType { wgsl: vec_of("u32", n), integer: true, out: "{}".into(), inp })
            }
        }
        _ => None,
    }
}

/// `@interpolate` of a fragment input record (MSL 5.4: center, centroid or sample with perspective or
/// no_perspective, or flat; WGSL: perspective or linear with center, centroid or sample, or flat). Metal's flat
/// interpolation takes the first vertex of the primitive, as WGSL's `flat` (sampling `first`) does (plan 5.3).
fn interpolation(r: &Record, integer: bool) -> Option<String> {
    if r.has("air.flat") || integer {
        return Some("@interpolate(flat)".into());
    }
    let ty = if r.has("air.no_perspective") { "linear" } else { "perspective" };
    let sampling = if r.has("air.centroid") {
        "centroid"
    } else if r.has("air.sample") {
        "sample"
    } else {
        "center"
    };
    Some(format!("@interpolate({}, {})", ty, sampling))
}

/// Assign varying locations for a pipeline: the fragment inputs, sorted by name, then the vertex outputs no fragment
/// input reads (which a vertex-only lowering still declares).
pub fn varyings(m_v: Option<(&Module, &Entry)>, m_f: Option<(&Module, &Entry)>) -> Result<Varyings, Refusal> {
    let mut out = Varyings::new();
    let mut prelude = Vec::new();
    if let Some((m, e)) = m_f {
        let f = &m.functions[e.function];
        let mut names: Vec<(String, TypeId, Record)> = Vec::new();
        for (i, r) in e.args.iter().enumerate() {
            if let Some(r) = r
                && r.kind() == "air.fragment_input"
                && r.has("air.interpolation_function")
            {
                return Err(tracked::refuse_row(
                    "pull-model-interpolation",
                    RefusalKind::NoExactLowering,
                    "air.fragment_input",
                    loc(m, e.function),
                    "an interpolant argument (interpolate_at_* functions)",
                    "WGSL 13.3.1.4",
                ));
            }
            if let Some(r) = r
                && r.kind() == "air.fragment_input"
                && let Some(n) = r.str_after("air.fragment_input")
            {
                names.push((n.to_string(), f.args[i].ty, r.clone()));
            }
        }
        names.sort_by(|a, b| a.0.cmp(&b.0));
        for (k, (n, t, r)) in names.into_iter().enumerate() {
            let vt = varying_type(m, t, r.type_name(), &mut prelude).ok_or_else(|| {
                tracked::refuse("varying-mismatch", n.clone(), loc(m, e.function), "a varying type without a WGSL form")
            })?;
            let interp = interpolation(&r, vt.integer).unwrap_or_default();
            out.insert(n, Varying { location: k as u32, wgsl: vt.wgsl, interp });
        }
    }
    if let Some((m, e)) = m_v {
        let ret = m.functions[e.function].ty;
        let ret = m.types.function(ret).map(|(r, _, _)| r).unwrap();
        let members: Vec<TypeId> = match m.types.get(ret) {
            Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. } => elems.clone(),
            _ => vec![ret],
        };
        let mut extra: Vec<(String, TypeId, Option<String>)> = Vec::new();
        for (k, r) in e.outputs.iter().enumerate() {
            if r.kind() == "air.vertex_output"
                && let Some(n) = r.str_after("air.vertex_output")
                && let Some(t) = members.get(k)
            {
                match out.get(n) {
                    Some(v) => {
                        let vt = varying_type(m, *t, r.type_name(), &mut prelude);
                        if vt.map(|vt| vt.wgsl) != Some(v.wgsl.clone()) {
                            return Err(tracked::refuse(
                                "varying-mismatch",
                                n.to_string(),
                                loc(m, e.function),
                                "the vertex output and the fragment input of this name have different types",
                            ));
                        }
                    }
                    None if m_f.is_none() => extra.push((n.to_string(), *t, r.type_name().map(String::from))),
                    None => {}
                }
            }
        }
        extra.sort_by(|a, b| a.0.cmp(&b.0));
        let base = out.len() as u32;
        for (k, (n, t, tn)) in extra.into_iter().enumerate() {
            let vt = varying_type(m, t, tn.as_deref(), &mut prelude).ok_or_else(|| {
                tracked::refuse("varying-mismatch", n.clone(), loc(m, e.function), "a varying type without a WGSL form")
            })?;
            let interp = if vt.integer { "@interpolate(flat)".to_string() } else { String::new() };
            out.insert(n, Varying { location: base + k as u32, wgsl: vt.wgsl, interp });
        }
    }
    Ok(out)
}

/// The context of one stage's interface lowering.
pub struct Ctx<'a> {
    pub m: &'a Module,
    pub entry: &'a Entry,
    pub desc: &'a PipelineDesc,
    pub host: &'a HostCaps,
    pub bindings: Option<&'a StageBindings>,
    pub varyings: &'a Varyings,
    /// IR arguments that some instruction uses.
    pub used: &'a [bool],
}

fn stage_refusal(m: &Module, fi: usize, what: &str, detail: &str) -> Refusal {
    tracked::refuse("stage-builtin", what.to_string(), loc(m, fi), detail.to_string())
}

/// Lower the interface of `cx.entry`.
pub fn lower(cx: &Ctx, layout: &mut Layout) -> Result<Interface, Refusal> {
    let m = cx.m;
    let e = cx.entry;
    let fi = e.function;
    let f = &m.functions[fi];
    let mut it = Interface::default();
    let stage = e.stage;
    let mut gid = false;
    let mut lid = false;
    let mut wid = false;
    let mut nwg = false;
    let mut lidx = false;
    let mut vid = false;
    let mut iid = false;
    let mut pos = false;
    let mut ff = false;
    let mut sid = false;
    let compute: Option<&ComputeState> = match &cx.desc.kind {
        PipelineKind::Compute(c) => Some(c),
        _ => None,
    };
    if stage == MetalStage::Kernel {
        let c = compute.ok_or_else(|| {
            tracked::refuse("pipeline-kind", "kernel", loc(m, fi), "a kernel in a non-compute pipeline")
        })?;
        if c.non_uniform_grid {
            return Err(tracked::refuse(
                "non-uniform-dispatch",
                "dispatchThreads",
                loc(m, fi),
                "WGSL dispatches whole workgroups; the edge guard is the stage-interface lowering's",
            ));
        }
        it.workgroup_size = Some(c.threads_per_threadgroup);
    }
    for (i, a) in f.args.iter().enumerate() {
        let Some(r) = &e.args[i] else {
            return Err(tracked::refuse(
                "entry-metadata",
                format!("argument {}", i),
                loc(m, fi),
                "an argument without a record",
            ));
        };
        let name = format!("aw_a{}", i);
        let kind = r.kind();
        let unused = r.has("air.arg_unused") || !cx.used[i];
        let t = a.ty;
        let bind = |it: &mut Interface, expr: String| it.prologue.push(format!("let {} = {};", name, expr));
        match kind {
            "air.buffer" => {
                if unused {
                    continue;
                }
                let index = r.int_after("air.location_index", 0).ok_or_else(|| {
                    tracked::refuse("entry-metadata", kind.to_string(), loc(m, fi), "a buffer without a location index")
                })?;
                let space = r.int_after("air.address_space", 0).unwrap_or(1) as u32;
                if space == 3 {
                    // A [[threadgroup(index)]] argument: memory of the length the dispatch sets for the index.
                    let len = compute
                        .and_then(|c| c.threadgroup_memory_lengths.get(index as usize).copied())
                        .filter(|&l| l > 0);
                    let Some(len) = len else {
                        return Err(tracked::refuse(
                            "threadgroup-memory-length",
                            format!("threadgroup({})", index),
                            loc(m, fi),
                            "no threadgroup memory length is set for this index",
                        ));
                    };
                    it.regions.insert(
                        i as u32,
                        ArgRegion {
                            kind: RegionKind::ThreadgroupArg { arg: i as u32, index },
                            name: format!("aw_tga{}", index),
                            base: "0u".into(),
                            words: Some(len.div_ceil(4)),
                        },
                    );
                    bind(&mut it, "0u".into());
                    continue;
                }
                if space != 1 && space != 2 {
                    return Err(tracked::refuse(
                        "entry-metadata",
                        kind.to_string(),
                        loc(m, fi),
                        format!("a buffer in address space {}", space),
                    ));
                }
                let var = layout.buffer(stage, index, cx.bindings);
                // Plan 5.4: the binding starts at an aligned base below the Metal offset; the residual is the region's
                // first byte.
                let base = layout.uniform(stage, UniformValue::ResidualOffset { stage, index });
                it.regions.insert(
                    i as u32,
                    ArgRegion {
                        kind: RegionKind::Buffer { arg: i as u32, index, space },
                        name: var,
                        base: base.clone(),
                        words: None,
                    },
                );
                bind(&mut it, base);
            }
            "air.texture" | "air.sampler" => {
                it.regions.insert(
                    i as u32,
                    ArgRegion {
                        kind: RegionKind::Handle { arg: i as u32 },
                        name: String::new(),
                        base: "0u".into(),
                        words: Some(0),
                    },
                );
                if !unused {
                    return Err(tracked::refuse(
                        "resource-argument",
                        kind.to_string(),
                        loc(m, fi),
                        "texture and sampler arguments are lowered by package S6a",
                    ));
                }
            }
            "air.indirect_buffer" => {
                if !unused {
                    return Err(tracked::refuse(
                        "argument-buffer",
                        kind.to_string(),
                        loc(m, fi),
                        "argument buffers are lowered by package S6d",
                    ));
                }
            }
            // Kernel built-ins (MSL 5.2.3.6, Table 5.8; WGSL 13.3.1.1). For a dispatch whose grid is a whole number of
            // threadgroups (the non-uniform case is refused above), MSL 5.2.3.6 defines the thread position in the grid
            // as threadgroup position times threadgroup size plus thread position in the threadgroup, which is WGSL's
            // global_invocation_id; the grid size is the threadgroup count times the threadgroup size.
            "air.thread_position_in_grid"
            | "air.thread_position_in_threadgroup"
            | "air.threadgroup_position_in_grid"
            | "air.threadgroups_per_grid"
            | "air.threads_per_grid"
            | "air.threads_per_threadgroup"
            | "air.dispatch_threads_per_threadgroup"
                if stage == MetalStage::Kernel =>
            {
                if unused {
                    continue;
                }
                let ws = it.workgroup_size.unwrap();
                let w = match kind {
                    "air.thread_position_in_grid" => {
                        gid = true;
                        "aw_gid".to_string()
                    }
                    "air.thread_position_in_threadgroup" => {
                        lid = true;
                        "aw_lid".into()
                    }
                    "air.threadgroup_position_in_grid" => {
                        wid = true;
                        "aw_wid".into()
                    }
                    "air.threadgroups_per_grid" => {
                        nwg = true;
                        "aw_nwg".into()
                    }
                    "air.threads_per_grid" => {
                        nwg = true;
                        format!("(aw_nwg * vec3<u32>({}u, {}u, {}u))", ws[0], ws[1], ws[2])
                    }
                    // The threadgroup size the dispatch gives, which this pipeline variant fixes (plan 5.5).
                    _ => format!("vec3<u32>({}u, {}u, {}u)", ws[0], ws[1], ws[2]),
                };
                let e = from_vec3(m, &w, t)
                    .ok_or_else(|| stage_refusal(m, fi, kind, "an unexpected type for the built-in"))?;
                bind(&mut it, e);
            }
            // MSL 5.2.3.6: the thread index in the threadgroup is ly * Sx + lx for a 2D threadgroup, WGSL's
            // local_invocation_index (WGSL 13.3.1.1) for a threadgroup of depth 1. A 3D threadgroup's linearization is
            // not stated by MSL and is refused until measured.
            "air.thread_index_in_threadgroup" if stage == MetalStage::Kernel => {
                if unused {
                    continue;
                }
                let ws = it.workgroup_size.unwrap();
                if ws[2] != 1 {
                    return Err(stage_refusal(
                        m,
                        fi,
                        kind,
                        "the linear thread index of a 3D threadgroup is not specified by MSL",
                    ));
                }
                lidx = true;
                let e = from_u32(m, "aw_lidx", t).ok_or_else(|| stage_refusal(m, fi, kind, "an unexpected type"))?;
                bind(&mut it, e);
            }
            // MSL 5.2.3.1, Table 5.2: vertex_id and instance_id include the base vertex and base instance; WebGPU's
            // vertex_index and instance_index include firstVertex/baseVertex and firstInstance (WebGPU 23.2,
            // WGSL 13.3.1.1).
            "air.vertex_id" | "air.instance_id" if stage == MetalStage::Vertex => {
                if unused {
                    continue;
                }
                let w = if kind == "air.vertex_id" {
                    vid = true;
                    "aw_vid"
                } else {
                    iid = true;
                    "aw_iid"
                };
                let e = from_u32(m, w, t).ok_or_else(|| stage_refusal(m, fi, kind, "an unexpected type"))?;
                bind(&mut it, e);
            }
            "air.base_vertex" | "air.base_instance" if stage == MetalStage::Vertex => {
                if unused {
                    continue;
                }
                let v = if kind == "air.base_vertex" { UniformValue::BaseVertex } else { UniformValue::BaseInstance };
                let w = layout.uniform(stage, v);
                let e = from_u32(m, &w, t).ok_or_else(|| stage_refusal(m, fi, kind, "an unexpected type"))?;
                bind(&mut it, e);
            }
            // MSL 5.2.3.4, Table 5.5: (x, y, z, 1/w) window-relative coordinates; WebGPU 23.2.5 fragment position: framebuffer x and y,
            // depth, and the perspective divisor 1/w.
            "air.position" if stage == MetalStage::Fragment => {
                if r.has("air.centroid") || r.has("air.sample") {
                    return Err(stage_refusal(m, fi, kind, "a fragment position at centroid or sample location"));
                }
                if unused {
                    continue;
                }
                pos = true;
                bind(&mut it, "bitcast<vec4<u32>>(aw_pos)".into());
            }
            "air.front_facing" if stage == MetalStage::Fragment => {
                if unused {
                    continue;
                }
                ff = true;
                bind(&mut it, "aw_ff".into());
            }
            "air.sample_id" if stage == MetalStage::Fragment => {
                if unused {
                    continue;
                }
                sid = true;
                let e = from_u32(m, "aw_sid", t).ok_or_else(|| stage_refusal(m, fi, kind, "an unexpected type"))?;
                bind(&mut it, e);
            }
            "air.fragment_input" if r.has("air.interpolation_function") => {
                return Err(tracked::refuse_row(
                    "pull-model-interpolation",
                    RefusalKind::NoExactLowering,
                    kind,
                    loc(m, fi),
                    "an interpolant argument (interpolate_at_* functions)",
                    "WGSL 13.3.1.4",
                ));
            }
            "air.position_in_patch" | "air.patch_id" | "air.patch_control_point_input" => {
                return Err(tracked::refuse_family("tessellation", kind.to_string(), loc(m, fi)));
            }
            "air.fragment_input" if stage == MetalStage::Fragment => {
                let n = r.str_after(kind).unwrap_or("");
                let v = cx.varyings.get(n).ok_or_else(|| {
                    tracked::refuse("varying-mismatch", n.to_string(), loc(m, fi), "no vertex output of this name")
                })?;
                let mut pre = Vec::new();
                let vt = varying_type(m, t, r.type_name(), &mut pre).ok_or_else(|| {
                    tracked::refuse("varying-mismatch", n.to_string(), loc(m, fi), "a varying type without a WGSL form")
                })?;
                it.prelude.extend(pre);
                let p = format!("aw_in{}", v.location);
                it.params.push(format!("@location({}) {} {}: {}", v.location, v.interp, p, v.wgsl));
                bind(&mut it, vt.inp.replace("{}", &p));
            }
            "air.render_target" if stage == MetalStage::Fragment => {
                return Err(tracked::refuse(
                    "framebuffer-fetch",
                    kind.to_string(),
                    loc(m, fi),
                    "a color attachment read in the shader",
                ));
            }
            "air.vertex_input" | "air.stage_in" => {
                return Err(tracked::refuse(
                    "vertex-input",
                    kind.to_string(),
                    loc(m, fi),
                    "stage_in attributes are lowered by package S6b",
                ));
            }
            _ => {
                if kind.starts_with("air.") {
                    return Err(match tracked::family_of(kind) {
                        Some("metadata-builtins") | None => stage_refusal(m, fi, kind, "no direct WGSL built-in"),
                        Some(fam) => tracked::refuse_family(fam, kind.to_string(), loc(m, fi)),
                    });
                }
                return Err(tracked::refuse(
                    "entry-metadata",
                    kind.to_string(),
                    loc(m, fi),
                    "an unrecognized argument record",
                ));
            }
        }
    }
    let p = &mut it.params;
    if gid {
        p.push("@builtin(global_invocation_id) aw_gid: vec3<u32>".into());
    }
    if lid {
        p.push("@builtin(local_invocation_id) aw_lid: vec3<u32>".into());
    }
    if wid {
        p.push("@builtin(workgroup_id) aw_wid: vec3<u32>".into());
    }
    if nwg {
        p.push("@builtin(num_workgroups) aw_nwg: vec3<u32>".into());
    }
    if lidx {
        p.push("@builtin(local_invocation_index) aw_lidx: u32".into());
    }
    if vid {
        p.push("@builtin(vertex_index) aw_vid: u32".into());
    }
    if iid {
        p.push("@builtin(instance_index) aw_iid: u32".into());
    }
    if pos {
        p.push("@builtin(position) aw_pos: vec4<f32>".into());
    }
    if ff {
        p.push("@builtin(front_facing) aw_ff: bool".into());
    }
    if sid {
        p.push("@builtin(sample_index) aw_sid: u32".into());
    }
    for fl in &e.flags {
        match fl {
            Item::Str(s) if s == "early_fragment_tests" => it.early_fragment_tests = true,
            Item::Str(s) => {
                return Err(tracked::refuse(
                    "entry-metadata",
                    s.clone(),
                    loc(m, fi),
                    "an unrecognized entry attribute",
                ));
            }
            _ => {}
        }
    }
    outputs(cx, &mut it)?;
    Ok(it)
}

fn outputs(cx: &Ctx, it: &mut Interface) -> Result<(), Refusal> {
    let m = cx.m;
    let e = cx.entry;
    let fi = e.function;
    let ret = m.types.function(m.functions[fi].ty).map(|(r, _, _)| r).unwrap();
    if m.types.is_void(ret) {
        if !e.outputs.is_empty() {
            return Err(tracked::refuse("entry-metadata", "outputs", loc(m, fi), "output records for a void function"));
        }
        return Ok(());
    }
    let (members, is_struct): (Vec<TypeId>, bool) = match m.types.get(ret) {
        Type::LiteralStruct { elems, .. } | Type::Struct { body: Some(elems), .. } => (elems.clone(), true),
        _ => (vec![ret], false),
    };
    if members.len() != e.outputs.len() {
        return Err(tracked::refuse(
            "entry-metadata",
            "outputs",
            loc(m, fi),
            "output records do not match the return type",
        ));
    }
    let topology_has_points = match &cx.desc.kind {
        PipelineKind::Render(r) => !matches!(r.input_primitive_topology, 2 | 3),
        _ => true,
    };
    let mut fields = Vec::new();
    let mut convs = Vec::new();
    for (k, (r, t)) in e.outputs.iter().zip(&members).enumerate() {
        let kind = r.kind();
        let member = if is_struct { format!("{{}}.m{}", k) } else { "{}".to_string() };
        let field = format!("o{}", k);
        let conv = |c: &str| c.replace("{}", &member);
        match kind {
            // MSL 5.2.3.3, Table 5.4 position (float4) and WGSL position output; clip space and the viewport
            // transform agree: MSL 1.7 (NDC (-1,-1) lower left, visible z in [0, 1], viewport origin top-left with
            // y down) and WebGPU 23.2.5 (the same NDC box; framebuffer y = 0.5 * (-ndc.y + 1) * height).
            "air.position" if e.stage == MetalStage::Vertex => {
                let inv = if r.has("air.invariant") { "@invariant " } else { "" };
                fields.push(format!("@builtin(position) {}{}: vec4<f32>", inv, field));
                convs.push(Some(conv("bitcast<vec4<f32>>({})")));
            }
            "air.vertex_output" if e.stage == MetalStage::Vertex => {
                let n = r.str_after(kind).unwrap_or("");
                match cx.varyings.get(n) {
                    Some(v) => {
                        let mut pre = Vec::new();
                        let vt = varying_type(m, *t, r.type_name(), &mut pre).ok_or_else(|| {
                            tracked::refuse(
                                "varying-mismatch",
                                n.to_string(),
                                loc(m, fi),
                                "a varying type without a WGSL form",
                            )
                        })?;
                        it.prelude.extend(pre);
                        fields.push(format!("@location({}) {} {}: {}", v.location, v.interp, field, v.wgsl));
                        convs.push(Some(conv(&vt.out)));
                    }
                    // No fragment input reads it: the output is not declared.
                    None => convs.push(None),
                }
            }
            // MSL 5.2.3.3, Table 5.4: the point size is the size of a point primitive; line and triangle topologies do
            // not use it.
            "air.point_size" if e.stage == MetalStage::Vertex && !topology_has_points => convs.push(None),
            "air.clip_distance" if !cx.host.features.contains("clip-distances") => {
                return Err(tracked::refuse_row(
                    "clip-distances",
                    RefusalKind::HostUnsupported,
                    kind,
                    loc(m, fi),
                    "the host lacks the clip-distances feature",
                    "WGSL 4.1.1 clip_distances",
                ));
            }
            "air.stencil" => {
                return Err(tracked::refuse_row(
                    "stencil-export",
                    RefusalKind::NoExactLowering,
                    kind,
                    loc(m, fi),
                    "WGSL has no stencil reference output",
                    "WGSL 13.3.1.1",
                ));
            }
            // MSL Table 5.7 color(n) and WGSL @location(n) on a fragment output: the value goes to color attachment n.
            "air.render_target" if e.stage == MetalStage::Fragment => {
                let color = r.int_after(kind, 0).unwrap_or(0);
                let index = r.int_after(kind, 1).unwrap_or(0);
                if index != 0 {
                    return Err(stage_refusal(
                        m,
                        fi,
                        "air.render_target index(1)",
                        "dual-source outputs are lowered by package S6b",
                    ));
                }
                let (n, el) = lanes(m, *t);
                let (scalar, c) = match (m.types.get(el), r.type_name().and_then(msl_type)) {
                    (Type::Float, _) => ("f32", format!("bitcast<{}>({{}})", vec_of("f32", n))),
                    (Type::Half, _) => {
                        it.prelude.push("nm_f16_to_f32");
                        let parts: Vec<String> = (0..n)
                            .map(|i| {
                                if n == 1 {
                                    "bitcast<f32>(nm_f16_to_f32({}))".to_string()
                                } else {
                                    format!("bitcast<f32>(nm_f16_to_f32({{}}[{}]))", i)
                                }
                            })
                            .collect();
                        (
                            "f32",
                            if n == 1 {
                                parts[0].clone()
                            } else {
                                format!("{}({})", vec_of("f32", n), parts.join(", "))
                            },
                        )
                    }
                    (Type::Int(b @ (8 | 16 | 32)), Some(("sint", _))) => {
                        let parts: Vec<String> = (0..n)
                            .map(|i| {
                                let x = if n == 1 { "{}".to_string() } else { format!("{{}}[{}]", i) };
                                if *b == 32 {
                                    format!("bitcast<i32>({})", x)
                                } else {
                                    it.prelude.push("nm_sext");
                                    format!("nm_sext({}, {}u)", x, b)
                                }
                            })
                            .collect();
                        (
                            "i32",
                            if n == 1 {
                                parts[0].clone()
                            } else {
                                format!("{}({})", vec_of("i32", n), parts.join(", "))
                            },
                        )
                    }
                    (Type::Int(8 | 16 | 32), _) => ("u32", "{}".to_string()),
                    _ => return Err(stage_refusal(m, fi, kind, "an output type without a WGSL form")),
                };
                fields.push(format!("@location({}) {}: {}", color, field, vec_of(scalar, n)));
                convs.push(Some(conv(&c)));
            }
            // MSL Table 5.7 depth(...) and WGSL frag_depth; the any/greater/less qualifier is a hint (plan 5.7).
            "air.depth" if e.stage == MetalStage::Fragment && matches!(m.types.get(*t), Type::Float) => {
                fields.push(format!("@builtin(frag_depth) {}: f32", field));
                convs.push(Some(conv("bitcast<f32>({})")));
            }
            _ => {
                return Err(match tracked::family_of(kind) {
                    Some("metadata-builtins") | None => stage_refusal(m, fi, kind, "no direct WGSL output"),
                    Some(fam) => tracked::refuse_family(fam, kind.to_string(), loc(m, fi)),
                });
            }
        }
    }
    if fields.is_empty() {
        // Every output is unread: the stage returns nothing.
        it.ret_fields = convs;
        return Ok(());
    }
    let name = if e.stage == MetalStage::Vertex { "aw_VOut" } else { "aw_FOut" };
    it.ret_decl =
        format!("struct {} {{\n{}}}\n\n", name, fields.iter().map(|f| format!("    {},\n", f)).collect::<String>());
    it.ret_type = Some(name.to_string());
    it.ret_fields = convs;
    Ok(())
}
