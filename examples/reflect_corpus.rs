//! Builds pipeline reflection from AIR for every pipeline function of a set of metallibs, and the jobs that make real
//! Metal report its reflection for the same pipelines.
//!
//! Usage: `reflect_corpus jobs METALLIBS JOBS.jsonl OURS.jsonl SIDECAR.tsv DUMPDIR`, then, after the recorder ran,
//! `reflect_corpus specialized SIDECAR.tsv OURS2.jsonl`. `reflect_corpus extract CONTAINERS OUTDIR` first writes the
//! metallibs held inside other files (universal binaries, Mach-O sections, pipeline libraries) out as files.
//!
//! METALLIBS lists metallib paths, one per line. For every vertex, fragment and kernel function the tool decodes the
//! function's module ([`airwgsl::air::AirModule`]) and writes:
//!
//! - to JOBS.jsonl, the pipeline `tools/metal-reflection/record.m` creates on real Metal: a compute pipeline for a
//!   kernel; for a vertex function, a render pipeline with no fragment function and a vertex descriptor that feeds
//!   each `air.vertex_input` from one buffer; for a fragment function, a render pipeline whose vertex function is
//!   generated MSL writing every varying the fragment function reads (matched by its `user(...)` or
//!   `generated(...)` name), with a color attachment for each `air.render_target` output or input, and depth and
//!   stencil attachments when the function writes depth or stencil. Function constants are set to zero bytes.
//! - to OURS.jsonl, this crate's reflection of the same stage ([`airwgsl::reflect::stage`]) in the recorder's
//!   format, or the refusal.
//!
//! A function of a module with function constants is specialized by real Metal (every constant zero), and the
//! recorder writes the specialized function's bitcode (`bitcodeData`) to DUMPDIR; its OURS.jsonl line is `pending`
//! and SIDECAR.tsv lists it. The `specialized` pass reflects those modules, as the guest receives them: a module
//! specialized by the framework, whose predicates are resolved.
//!
//! A function this tool cannot set a pipeline up for gets a `skip` line in OURS.jsonl naming the reason and no job.
//! `tools/metal-reflection/compare.py` compares the two outputs.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write;

use airwgsl::air::{AirModule, EntryPoint, OutputRole, Role, Slot, Stage};
use airwgsl::api::{StageInputDescriptor, StageReflection, VertexAttribute, VertexBufferLayout, VertexDescriptor};
use airwgsl::reflect::{self, Inputs, StageInput};

/// A JSON value, written with sorted keys.
enum J {
    N(u64),
    B(bool),
    S(String),
    A(Vec<J>),
    O(BTreeMap<String, J>),
}

fn obj(pairs: Vec<(&str, J)>) -> J {
    J::O(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn write_json(out: &mut String, j: &J) {
    match j {
        J::N(n) => {
            let _ = write!(out, "{}", n);
        }
        J::B(b) => out.push_str(if *b { "true" } else { "false" }),
        J::S(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    c if (c as u32) < 0x20 => {
                        let _ = write!(out, "\\u{:04x}", c as u32);
                    }
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        J::A(v) => {
            out.push('[');
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(out, x);
            }
            out.push(']');
        }
        J::O(m) => {
            out.push('{');
            for (i, (k, v)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(out, &J::S(k.clone()));
                out.push(':');
                write_json(out, v);
            }
            out.push('}');
        }
    }
}

fn json(j: &J) -> String {
    let mut s = String::new();
    write_json(&mut s, j);
    s
}

/// The recorder's form of a stage's reflection.
fn stage_json(stage: &str, r: &StageReflection) -> J {
    let args = r
        .arguments
        .iter()
        .map(|a| {
            let mut m = vec![
                ("name", J::S(a.name.clone())),
                ("type", J::N(a.r#type)),
                ("access", J::N(a.access)),
                ("index", J::N(a.index)),
                ("active", J::B(a.active)),
                ("array_length", J::N(a.array_length)),
            ];
            match a.r#type {
                0 => {
                    m.push(("buffer_alignment", J::N(a.buffer_alignment)));
                    m.push(("buffer_data_size", J::N(a.buffer_data_size)));
                    m.push(("buffer_data_type", J::N(a.buffer_data_type)));
                    m.push(("buffer_struct_type", J::N(a.buffer_struct_type as u64)));
                    m.push(("buffer_pointer_type", J::N(a.buffer_pointer_type as u64)));
                }
                1 => {
                    m.push(("threadgroup_memory_alignment", J::N(a.threadgroup_memory_alignment)));
                    m.push(("threadgroup_memory_data_size", J::N(a.threadgroup_memory_data_size)));
                }
                2 => {
                    m.push(("texture_type", J::N(a.texture_type)));
                    m.push(("texture_data_type", J::N(a.texture_data_type)));
                    m.push(("is_depth_texture", J::B(a.is_depth_texture)));
                }
                _ => {}
            }
            obj(m)
        })
        .collect();
    let types = r
        .types
        .iter()
        .map(|t| {
            let mut m = vec![("data_type", J::N(t.data_type))];
            match t.data_type {
                1 => m.push((
                    "members",
                    J::A(
                        t.members
                            .iter()
                            .map(|x| {
                                obj(vec![
                                    ("name", J::S(x.name.clone())),
                                    ("offset", J::N(x.offset)),
                                    ("data_type", J::N(x.data_type)),
                                    ("argument_index", J::N(x.argument_index)),
                                    ("type", J::N(x.r#type as u64)),
                                ])
                            })
                            .collect(),
                    ),
                )),
                2 => {
                    m.push(("element_data_type", J::N(t.element_data_type)));
                    m.push(("array_length", J::N(t.array_length)));
                    m.push(("stride", J::N(t.stride)));
                    m.push(("argument_index_stride", J::N(t.argument_index_stride)));
                    m.push(("element_type", J::N(t.element_type as u64)));
                }
                60 => {
                    m.push(("element_data_type", J::N(t.element_data_type)));
                    m.push(("access", J::N(t.access)));
                    m.push(("alignment", J::N(t.alignment)));
                    m.push(("data_size", J::N(t.data_size)));
                    m.push(("element_is_argument_buffer", J::B(t.element_is_argument_buffer)));
                    m.push(("element_type", J::N(t.element_type as u64)));
                }
                58 => {
                    m.push(("texture_data_type", J::N(t.texture_data_type)));
                    m.push(("texture_type", J::N(t.texture_type)));
                    m.push(("access", J::N(t.access)));
                    m.push(("is_depth_texture", J::B(t.is_depth_texture)));
                }
                _ => {}
            }
            obj(m)
        })
        .collect();
    obj(vec![("stage", J::S(stage.into())), ("arguments", J::A(args)), ("types", J::A(types))])
}

/// `MTLVertexFormat` (= `MTLAttributeFormat`) of an attribute's MSL type.
fn vertex_format(t: &str) -> Option<u64> {
    Some(match t {
        "uchar2" => 1,
        "uchar3" => 2,
        "uchar4" => 3,
        "char2" => 4,
        "char3" => 5,
        "char4" => 6,
        "ushort2" => 13,
        "ushort3" => 14,
        "ushort4" => 15,
        "short2" => 16,
        "short3" => 17,
        "short4" => 18,
        "half2" => 25,
        "half3" => 26,
        "half4" => 27,
        "float" => 28,
        "float2" => 29,
        "float3" => 30,
        "float4" => 31,
        "int" => 32,
        "int2" => 33,
        "int3" => 34,
        "int4" => 35,
        "uint" => 36,
        "uint2" => 37,
        "uint3" => 38,
        "uint4" => 39,
        "uchar" => 45,
        "char" => 46,
        "ushort" => 49,
        "short" => 50,
        "half" => 53,
        _ => return None,
    })
}

/// `MTLPixelFormat` for a color output or input of the MSL type `t`: the R, RG or RGBA format of the type's
/// component type with no more components than `t` has.
fn color_format(t: &str) -> Option<u64> {
    let scalar = t.trim_end_matches(['2', '3', '4']);
    let n = match t.as_bytes().last() {
        Some(b'2') => 1,
        Some(b'3') | Some(b'4') => 2,
        _ => 0,
    };
    let formats: [u64; 3] = match scalar {
        "float" => [55, 105, 125],
        "half" => [25, 65, 115],
        "int" => [54, 104, 124],
        "uint" => [53, 103, 123],
        "short" => [24, 64, 114],
        "ushort" => [23, 63, 113],
        "char" => [14, 34, 74],
        "uchar" => [13, 33, 73],
        _ => return None,
    };
    if n == 2 && t.ends_with('3') {
        // A three-component output needs a format of at most three components: there is none among the R, RG and
        // RGBA integer and float formats, so use RG and leave the third component unwritten.
        return Some(formats[1]);
    }
    Some(formats[n])
}

/// The bytes of a zero value of a function constant's MSL type, as `setConstantValue:type:atIndex:` reads it.
fn constant_value(t: &str) -> Option<(u64, usize)> {
    let data_type = reflect::names::data_type(t)?;
    let scalar = t.trim_end_matches(['2', '3', '4']);
    let n: usize = match t.as_bytes().last() {
        Some(b'2') => 2,
        Some(b'3') => 4,
        Some(b'4') => 4,
        _ => 1,
    };
    let size = match scalar {
        "bool" | "char" | "uchar" => 1,
        "short" | "ushort" | "half" | "bfloat" => 2,
        "int" | "uint" | "float" => 4,
        "long" | "ulong" => 8,
        _ => return None,
    };
    Some((data_type, size * n))
}

/// The vertex function a fragment function's varyings are written by, as MSL source.
fn vertex_source(e: &EntryPoint) -> Result<(String, bool), String> {
    let mut members = String::from("    float4 wm_position [[position]];\n");
    let mut layered = false;
    for a in &e.arguments {
        match &a.role {
            Role::FragmentInput { name, .. } => {
                let t = a.type_name.as_deref().ok_or("varying without a type name")?;
                if let Some(user) = name.strip_prefix("user(").and_then(|s| s.strip_suffix(')')) {
                    let _ = writeln!(members, "    {} wm_{} [[user({})]];", t, members.len(), user);
                } else if let Some(g) = name.strip_prefix("generated(").and_then(|s| s.strip_suffix(')')) {
                    let digits: String = g.chars().take_while(|c| c.is_ascii_digit()).collect();
                    let len: usize = digits.parse().map_err(|_| format!("generated name {}", name))?;
                    let ident = g.get(digits.len()..digits.len() + len).ok_or(format!("generated name {}", name))?;
                    let _ = writeln!(members, "    {} {};", t, ident);
                } else {
                    return Err(format!("varying name {}", name));
                }
            }
            Role::Builtin { key: "air.render_target_array_index", .. } => {
                layered = true;
                members.push_str("    uint wm_layer [[render_target_array_index]];\n");
            }
            Role::Builtin { key: "air.viewport_array_index", .. } => {
                layered = true;
                members.push_str("    uint wm_viewport [[viewport_array_index]];\n");
            }
            _ => {}
        }
    }
    let src = format!(
        "#include <metal_stdlib>\nusing namespace metal;\nstruct WMOut {{\n{}}};\nvertex WMOut wm_vertex() {{ WMOut o = {{}}; return o; }}\n",
        members
    );
    Ok((src, layered))
}

/// The value of a binding slot in the job's pipeline: a number, or a function constant the job sets to zero.
fn slot_value(m: &AirModule, slot: &Slot) -> Option<u32> {
    match slot {
        Slot::Value(v) => Some(*v),
        Slot::Constant(g) => m.function_constants.iter().any(|c| c.global == *g).then_some(0),
    }
}

fn buffer_indices(e: &EntryPoint) -> Vec<u32> {
    e.arguments
        .iter()
        .filter_map(|a| match &a.role {
            Role::Buffer(b) | Role::IndirectBuffer(b) => match b.location.index {
                Slot::Value(v) => Some(v),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// What a job needs besides its JSON: the vertex descriptor it sets (reflection lists its buffers) and whether its
/// function is specialized by real Metal.
struct Setup {
    vertex_descriptor: Option<VertexDescriptor>,
    stage_input: Option<StageInputDescriptor>,
    specialized: bool,
}

fn job(lib: &str, m: &AirModule, e: &EntryPoint, id: &str, dump: &str) -> Result<(J, Setup), String> {
    let mut j: Vec<(&str, J)> = vec![("id", J::S(id.into())), ("library", J::S(lib.into()))];
    let mut setup = Setup { vertex_descriptor: None, stage_input: None, specialized: false };
    if !m.function_constants.is_empty() {
        let mut cs = Vec::new();
        for c in &m.function_constants {
            let (t, n) = constant_value(&c.type_name).ok_or(format!("function constant type {}", c.type_name))?;
            cs.push(obj(vec![("index", J::N(c.index as u64)), ("type", J::N(t)), ("hex", J::S("00".repeat(n)))]));
        }
        j.push(("constants", J::A(cs)));
        j.push(("dump", J::S(dump.into())));
        setup.specialized = true;
    }
    match e.stage {
        Stage::Kernel => {
            j.push(("kind", J::S("compute".into())));
            j.push(("function", J::S(e.name.clone())));
            let inputs: Vec<_> = e
                .arguments
                .iter()
                .filter_map(|a| match &a.role {
                    Role::StageIn(Some(l)) => Some((l, a.type_name.as_deref().unwrap_or(""))),
                    _ => None,
                })
                .collect();
            if !inputs.is_empty() {
                let used = buffer_indices(e);
                let buffer = (0..31u32).rev().find(|b| !used.contains(b)).ok_or("no free stage-in buffer index")?;
                let mut attrs = Vec::new();
                let mut sd = StageInputDescriptor {
                    layouts: vec![(0, 5, 1); buffer as usize + 1],
                    attributes: Vec::new(),
                    index_type: 0,
                    index_buffer_index: 0,
                };
                sd.layouts[buffer as usize] = (64, 5, 1);
                for (l, t) in inputs {
                    let Slot::Value(index) = l.index else { return Err("stage-in index from a constant".into()) };
                    let format = vertex_format(t).ok_or(format!("stage-in type {}", t))?;
                    if sd.attributes.len() <= index as usize {
                        sd.attributes.resize(index as usize + 1, (0, 0, 0));
                    }
                    sd.attributes[index as usize] = (format, 0, buffer as u64);
                    attrs.push(obj(vec![
                        ("index", J::N(index as u64)),
                        ("format", J::N(format)),
                        ("offset", J::N(0)),
                        ("buffer", J::N(buffer as u64)),
                    ]));
                }
                // MTLStepFunctionThreadPositionInGridX (5).
                let layout = obj(vec![
                    ("buffer", J::N(buffer as u64)),
                    ("stride", J::N(64)),
                    ("step_function", J::N(5)),
                    ("step_rate", J::N(1)),
                ]);
                j.push(("stage_input", obj(vec![("layouts", J::A(vec![layout])), ("attributes", J::A(attrs))])));
                setup.stage_input = Some(sd);
            }
        }
        Stage::Vertex => {
            j.push(("kind", J::S("render".into())));
            j.push(("vertex", obj(vec![("function", J::S(e.name.clone()))])));
            // Each input with the step function of its buffer: per vertex (1), or for a post-tessellation function
            // per patch control point (4) and per patch (3).
            let mut inputs: Vec<(Slot, String, u64)> = Vec::new();
            for a in &e.arguments {
                let t = a.type_name.clone().unwrap_or_default();
                match &a.role {
                    Role::VertexInput(l) => inputs.push((l.index, t, 1)),
                    Role::PatchInput(l) => inputs.push((l.index, t, 3)),
                    Role::PatchControlPointInput(cp) => {
                        for x in &cp.attributes {
                            inputs.push((x.location.index, x.type_name.clone(), 4));
                        }
                    }
                    _ => {}
                }
            }
            if let Some(patch) = &e.patch {
                // Pow2 partitioning, factor 16, half factors, no control-point index buffer, constant factor step,
                // clockwise winding: MTLRenderPipelineDescriptor's defaults.
                let _ = patch;
                j.push((
                    "tessellation",
                    obj(vec![
                        ("partition", J::N(0)),
                        ("max_factor", J::N(16)),
                        ("factor_format", J::N(0)),
                        ("control_point_index_type", J::N(0)),
                        ("step_function", J::N(0)),
                        ("winding", J::N(0)),
                    ]),
                ));
            }
            if !inputs.is_empty() {
                let used = buffer_indices(e);
                let mut free = (0..31u32).rev().filter(|b| !used.contains(b));
                let mut buffer_of: BTreeMap<u64, u32> = BTreeMap::new();
                let mut vd = VertexDescriptor { layouts: Vec::new(), attributes: Vec::new() };
                let mut attrs = Vec::new();
                for (slot, t, step) in &inputs {
                    let buffer = match buffer_of.get(step) {
                        Some(b) => *b,
                        None => {
                            let b = free.next().ok_or("no free vertex buffer index")?;
                            buffer_of.insert(*step, b);
                            b
                        }
                    };
                    let Slot::Value(index) = slot else { return Err("vertex input index from a constant".into()) };
                    let format = vertex_format(t).ok_or(format!("vertex input type {}", t))?;
                    attrs.push(obj(vec![
                        ("index", J::N(*index as u64)),
                        ("format", J::N(format)),
                        ("offset", J::N(0)),
                        ("buffer", J::N(buffer as u64)),
                    ]));
                    if vd.attributes.len() <= *index as usize {
                        vd.attributes
                            .resize(*index as usize + 1, VertexAttribute { format: 0, offset: 0, buffer_index: 0 });
                    }
                    vd.attributes[*index as usize] = VertexAttribute { format, offset: 0, buffer_index: buffer as u64 };
                }
                let mut layouts = Vec::new();
                for (&step, &buffer) in &buffer_of {
                    if vd.layouts.len() <= buffer as usize {
                        vd.layouts.resize(
                            buffer as usize + 1,
                            VertexBufferLayout { stride: 0, step_function: 1, step_rate: 1 },
                        );
                    }
                    vd.layouts[buffer as usize] = VertexBufferLayout { stride: 64, step_function: step, step_rate: 1 };
                    layouts.push(obj(vec![
                        ("buffer", J::N(buffer as u64)),
                        ("stride", J::N(64)),
                        ("step_function", J::N(step)),
                        ("step_rate", J::N(1)),
                    ]));
                }
                j.push(("vertex_descriptor", obj(vec![("layouts", J::A(layouts)), ("attributes", J::A(attrs))])));
                setup.vertex_descriptor = Some(vd);
            }
            if e.outputs.is_empty() {
                // A vertex function that returns nothing is valid only without rasterization.
                j.push(("rasterization", J::N(0)));
            } else if e.outputs.iter().any(|o| matches!(o.role, OutputRole::Builtin("air.point_size"))) {
                // Writing the point size requires the point topology class.
                j.push(("topology", J::N(1)));
            } else if e.outputs.iter().any(|o| {
                matches!(o.role, OutputRole::Builtin("air.render_target_array_index" | "air.viewport_array_index"))
            }) {
                j.push(("topology", J::N(3)));
            }
        }
        Stage::Fragment => {
            if e.arguments.iter().any(|a| matches!(a.role, Role::Imageblock(_))) {
                return Err("fragment imageblock".into());
            }
            let (src, layered) = vertex_source(e)?;
            j.push(("kind", J::S("render".into())));
            j.push(("vertex", obj(vec![("source", J::S(src)), ("function", J::S("wm_vertex".into()))])));
            j.push(("fragment", obj(vec![("function", J::S(e.name.clone()))])));
            if layered {
                j.push(("topology", J::N(3)));
            }
            let mut colors: BTreeMap<u64, u64> = BTreeMap::new();
            let mut depth = false;
            let mut stencil = false;
            for o in &e.outputs {
                match &o.role {
                    OutputRole::RenderTarget { index, .. } => {
                        let i = &slot_value(m, index)
                            .ok_or("render target index from an expression the static initializer computes")?;
                        let t = o.type_name.as_deref().unwrap_or("");
                        colors.insert(*i as u64, color_format(t).ok_or(format!("color output type {}", t))?);
                    }
                    OutputRole::Depth(_) => depth = true,
                    OutputRole::Builtin("air.stencil") => stencil = true,
                    _ => {}
                }
            }
            for a in &e.arguments {
                if let Role::RenderTarget { index, .. } = &a.role {
                    let i = &slot_value(m, index)
                        .ok_or("render target index from an expression the static initializer computes")?;
                    let t = a.type_name.as_deref().unwrap_or("");
                    colors.insert(*i as u64, color_format(t).ok_or(format!("color input type {}", t))?);
                }
            }
            j.push(("colors", J::A(colors.into_iter().map(|(i, f)| J::A(vec![J::N(i), J::N(f)])).collect())));
            if depth {
                j.push(("depth", J::N(252)));
            }
            if stencil {
                j.push(("stencil", J::N(253)));
            }
        }
        _ => return Err("not a vertex, fragment or kernel function".into()),
    }
    Ok((obj(j), setup))
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Vertex => "vertex",
        Stage::Fragment => "fragment",
        _ => "kernel",
    }
}

/// This crate's reflection of `entry` in `m`, as an output line.
fn reflect_line(id: &str, m: &AirModule, entry: &str, input: StageInput<'_>) -> J {
    let Some(e) = m.entry_point(entry) else {
        return obj(vec![
            ("id", J::S(id.into())),
            ("status", J::S("skip".into())),
            ("reason", J::S("no entry point".into())),
        ]);
    };
    match reflect::stage(m, e, &Inputs::default(), input) {
        Ok(r) => {
            let mut stages = Vec::new();
            if e.stage == Stage::Fragment {
                // The generated vertex function has no arguments.
                stages.push(stage_json("vertex", &StageReflection { function_type: 1, ..Default::default() }));
            }
            stages.push(stage_json(stage_name(e.stage), &r));
            obj(vec![("id", J::S(id.into())), ("status", J::S("ok".into())), ("stages", J::A(stages))])
        }
        Err(r) => {
            obj(vec![("id", J::S(id.into())), ("status", J::S("refused".into())), ("refusal", J::S(r.to_string()))])
        }
    }
}

fn status_of(line: &J) -> String {
    if let J::O(m) = line
        && let Some(J::S(s)) = m.get("status")
    {
        return s.clone();
    }
    String::new()
}

fn input_of(setup: &Setup) -> StageInput<'_> {
    match (&setup.vertex_descriptor, &setup.stage_input) {
        (Some(vd), _) => StageInput::Vertex(vd),
        (None, Some(sd)) => StageInput::Compute(sd),
        _ => StageInput::None,
    }
}

type Triples = Vec<(u64, u64, u64)>;

/// Encodes a job's stage input for the sidecar: `V` or `C`, then `stride/step/rate` per layout and
/// `format/offset/buffer` per attribute.
fn encode_input(setup: &Setup) -> String {
    let (kind, layouts, attributes): (&str, Triples, Triples) = match (&setup.vertex_descriptor, &setup.stage_input) {
        (Some(vd), _) => (
            "V",
            vd.layouts.iter().map(|l| (l.stride, l.step_function, l.step_rate)).collect(),
            vd.attributes.iter().map(|a| (a.format, a.offset, a.buffer_index)).collect(),
        ),
        (None, Some(sd)) => ("C", sd.layouts.clone(), sd.attributes.clone()),
        _ => return String::new(),
    };
    let f = |v: &[(u64, u64, u64)]| v.iter().map(|x| format!("{}/{}/{}", x.0, x.1, x.2)).collect::<Vec<_>>().join(",");
    format!("{}|{}|{}", kind, f(&layouts), f(&attributes))
}

/// The stage input of a sidecar row: a vertex descriptor or a stage-input descriptor.
enum Decoded {
    None,
    Vertex(VertexDescriptor),
    Compute(StageInputDescriptor),
}

fn decode_input(s: &str) -> Option<Decoded> {
    if s.is_empty() {
        return Some(Decoded::None);
    }
    let mut p = s.split('|');
    let (kind, l, a) = (p.next()?, p.next()?, p.next()?);
    let triples = |x: &str| -> Option<Vec<(u64, u64, u64)>> {
        x.split(',')
            .filter(|x| !x.is_empty())
            .map(|t| {
                let n: Vec<u64> = t.split('/').map(|n| n.parse().ok()).collect::<Option<Vec<u64>>>()?;
                Some((n[0], n[1], n[2]))
            })
            .collect()
    };
    let (layouts, attributes) = (triples(l)?, triples(a)?);
    Some(match kind {
        "V" => Decoded::Vertex(VertexDescriptor {
            layouts: layouts
                .iter()
                .map(|x| VertexBufferLayout { stride: x.0, step_function: x.1, step_rate: x.2 })
                .collect(),
            attributes: attributes
                .iter()
                .map(|x| VertexAttribute { format: x.0, offset: x.1, buffer_index: x.2 })
                .collect(),
        }),
        _ => Decoded::Compute(StageInputDescriptor { layouts, attributes, index_type: 0, index_buffer_index: 0 }),
    })
}

fn jobs_mode(args: &[String]) {
    let libs = std::fs::read_to_string(&args[2]).expect("list");
    let mut jobs = std::fs::File::create(&args[3]).expect("jobs");
    let mut ours = std::fs::File::create(&args[4]).expect("ours");
    let mut sidecar = std::fs::File::create(&args[5]).expect("sidecar");
    let dumps = &args[6];
    std::fs::create_dir_all(dumps).expect("dump directory");
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut n = 0usize;
    for lib in libs.lines().filter(|l| !l.is_empty()) {
        let data = std::fs::read(lib).expect("read metallib");
        let ml = match airwgsl::parse_metallib(&data) {
            Ok(m) => m,
            Err(e) => {
                *counts.entry(format!("metallib error: {}", e)).or_default() += 1;
                continue;
            }
        };
        for f in &ml.functions {
            let Some(bytes) = f.module else { continue };
            if f.alias_of.is_some() {
                continue;
            }
            let id = format!("{}#{}", lib, f.name);
            let line = match AirModule::parse(bytes) {
                Err(r) => {
                    obj(vec![("id", J::S(id)), ("status", J::S("refused".into())), ("refusal", J::S(r.to_string()))])
                }
                Ok(m) => match m.entry_point(&f.name) {
                    None => obj(vec![
                        ("id", J::S(id)),
                        ("status", J::S("skip".into())),
                        (
                            "reason",
                            J::S(format!(
                                "not a vertex, fragment or kernel function: the module lists no entry point for it \
                                 (metallib function type {})",
                                f.function_type
                            )),
                        ),
                    ]),
                    Some(e) => {
                        n += 1;
                        let dump = format!("{}/{}.bc", dumps, n);
                        match job(lib, &m, e, &id, &dump) {
                            Err(reason) => {
                                obj(vec![("id", J::S(id)), ("status", J::S("skip".into())), ("reason", J::S(reason))])
                            }
                            Ok((jb, setup)) => {
                                writeln!(jobs, "{}", json(&jb)).unwrap();
                                if setup.specialized {
                                    writeln!(sidecar, "{}\t{}\t{}\t{}", id, dump, f.name, encode_input(&setup))
                                        .unwrap();
                                    obj(vec![("id", J::S(id)), ("status", J::S("pending".into()))])
                                } else {
                                    reflect_line(&id, &m, &f.name, input_of(&setup))
                                }
                            }
                        }
                    }
                },
            };
            *counts.entry(status_of(&line)).or_default() += 1;
            writeln!(ours, "{}", json(&line)).unwrap();
        }
    }
    for (k, n) in counts {
        println!("{} {}", k, n);
    }
}

/// Reflects the specialized modules real Metal produced (`bitcodeData` of the specialized function, written by the
/// recorder) for the jobs whose functions have function constants.
fn specialized_mode(args: &[String]) {
    let rows = std::fs::read_to_string(&args[2]).expect("sidecar");
    let mut ours = std::fs::File::create(&args[3]).expect("ours");
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in rows.lines().filter(|l| !l.is_empty()) {
        let p: Vec<&str> = row.split('\t').collect();
        let (id, dump, function) = (p[0], p[1], p[2]);
        let input = decode_input(p.get(3).copied().unwrap_or("")).expect("sidecar stage input");
        let line = match std::fs::read(dump) {
            Err(_) => obj(vec![
                ("id", J::S(id.into())),
                ("status", J::S("skip".into())),
                ("reason", J::S("real Metal wrote no specialized bitcode".into())),
            ]),
            Ok(bytes) => match AirModule::parse(&bytes) {
                Err(r) => obj(vec![
                    ("id", J::S(id.into())),
                    ("status", J::S("refused".into())),
                    ("refusal", J::S(r.to_string())),
                ]),
                Ok(m) => reflect_line(
                    id,
                    &m,
                    function,
                    match &input {
                        Decoded::None => StageInput::None,
                        Decoded::Vertex(vd) => StageInput::Vertex(vd),
                        Decoded::Compute(sd) => StageInput::Compute(sd),
                    },
                ),
            },
        };
        *counts.entry(status_of(&line)).or_default() += 1;
        writeln!(ours, "{}", json(&line)).unwrap();
    }
    for (k, n) in counts {
        println!("{} {}", k, n);
    }
}

/// Writes every metallib inside the listed container files (universal binaries, Mach-O sections, pipeline
/// libraries) to OUTDIR as its own file, so that real Metal can load it, and prints the paths written.
fn extract_mode(args: &[String]) {
    let list = std::fs::read_to_string(&args[2]).expect("list");
    let out = &args[3];
    std::fs::create_dir_all(out).expect("output directory");
    for path in list.lines().filter(|l| !l.is_empty()) {
        let data = std::fs::read(path).expect("read container");
        // Metallibs a container format holds, else metallibs embedded at any offset (the shared cache's data).
        let libs: Vec<(String, &[u8])> = match airwgsl::mtlb::parse_container(&data, path) {
            Ok(items) => items
                .into_iter()
                .filter_map(|item| match item {
                    airwgsl::mtlb::Item::Metallib { location, lib, .. } => Some((location, lib.data)),
                    _ => None,
                })
                .collect(),
            Err(_) => airwgsl::mtlb::embedded_metallibs(&data, &[])
                .into_iter()
                .map(|(offset, lib)| (format!("{:#x}", offset), lib.data))
                .collect(),
        };
        for (location, bytes) in libs {
            let name: String = format!("{}#{}", path, location)
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' { c } else { '_' })
                .collect();
            let file = format!("{}/{}.metallib", out, name);
            std::fs::write(&file, bytes).expect("write metallib");
            println!("{}", file);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("extract") if args.len() == 4 => extract_mode(&args),
        Some("jobs") if args.len() == 7 => jobs_mode(&args),
        Some("specialized") if args.len() == 4 => specialized_mode(&args),
        _ => {
            eprintln!("usage: reflect_corpus extract CONTAINERS OUTDIR");
            eprintln!("       reflect_corpus jobs METALLIBS JOBS.jsonl OURS.jsonl SIDECAR.tsv DUMPDIR");
            eprintln!("       reflect_corpus specialized SIDECAR.tsv OURS.jsonl");
            std::process::exit(2);
        }
    }
}
