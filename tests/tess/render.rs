//! Tessellated draws through the whole pipeline: an MSL post-tessellation vertex function and fragment function
//! compiled by `airwgsl::compile`, the tessellator's kernels on wgpu, and the generated triangles drawn with the
//! compiled WGSL, against the same draw on this Mac's Metal (`metal_tess.swift`), compared byte for byte.
//!
//! The vertex function places each point at a window position hashed from its position in the patch, its patch
//! identifier and its instance, so that the triangles overlap, and writes a flat code mixing that hash with a
//! control point and the patch's per-patch attribute. The render target is the last code written at each pixel: it
//! depends on every triangle, its vertex order (the flat value is the first vertex's) and the order of the triangles.

use super::draw::Draw;
use super::kernel::{dispatch, read};
use super::stage::{air, case_dir, gpu, host, module, render_desc};
use airwgsl::api::*;
use airwgsl::tess::{self, factors::f32_to_half};
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

const SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;
struct CP { float4 p [[attribute(0)]]; };
struct PIn { patch_control_point<CP> cp; uint tag [[attribute(1)]]; };
struct VOut { float4 position [[position]]; uint code [[flat]]; };
static uint mixed(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
static VOut place(uint a, uint b, uint c, uint patch, uint instance, uint tag, float4 cp0, float4 cp1) {
    uint h = mixed(a ^ mixed(b ^ mixed(c ^ mixed(patch * 31u + instance))));
    // Window positions on a grid of 1/64: exact in binary32 on both sides.
    float2 xy = float2(float(h & 0x7fu), float((h >> 7) & 0x7fu)) / 64.0f - 1.0f + cp0.xy;
    VOut o;
    o.position = float4(xy, 0.5f, 1.0f);
    o.code = (h | 1u) ^ tag ^ as_type<uint>(cp1.w);
    return o;
}
[[patch(quad, 4)]] vertex VOut quad_vertex(PIn in [[stage_in]], float2 p [[position_in_patch]], uint patch [[patch_id]],
                                           uint instance [[instance_id]]) {
    return place(as_type<uint>(p.x), as_type<uint>(p.y), 0u, patch, instance, in.tag, in.cp[0].p, in.cp[3].p);
}
[[patch(triangle, 3)]] vertex VOut triangle_vertex(PIn in [[stage_in]], float3 p [[position_in_patch]],
                                                   uint patch [[patch_id]], uint instance [[instance_id]]) {
    return place(as_type<uint>(p.x), as_type<uint>(p.y), as_type<uint>(p.z), patch, instance, in.tag, in.cp[1].p,
                 in.cp[2].p);
}
fragment uint code_fragment(VOut in [[stage_in]]) { return in.code; }
"#;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

/// One full-pipeline draw.
struct Scene {
    draw: Draw,
    /// 0 none, 1 UInt16, 2 UInt32 control-point indices (drawIndexedPatches).
    indexed: u64,
    control_points: u32,
    /// Buffer 1: a float4 per control point, as bit patterns.
    cp: Vec<u32>,
    /// Buffer 2: a uint per patch.
    tags: Vec<u32>,
    cp_indices: Vec<u32>,
}

fn metal_runner() -> PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let exe = super::stage::tmp_dir().join("metal_tess");
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/tess/metal_tess.swift");
        let o = Command::new("swiftc").args(["-O", "-o"]).arg(&exe).arg(&src).output().expect("swiftc");
        assert!(o.status.success(), "swiftc: {}", String::from_utf8_lossy(&o.stderr));
        exe
    })
    .clone()
}

fn metal(name: &str, s: &Scene) -> Vec<u8> {
    let dir = case_dir(&format!("tess-{name}"));
    let src = dir.join("tess.metal");
    std::fs::write(&src, SOURCE).unwrap();
    let w = |v: &[u32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
    let d = &s.draw;
    let spec = format!(
        "source {}\nvertex {}\nfragment code_fragment\nquad {}\nmode {}\nmax {}\nwinding {}\nindexed {}\npoints {}\n\
         patches {}\ninstances {}\nbase {}\nwidth {}\nheight {}\nfactors {}\ncp {}\ntags {}\ncpindices {}\n",
        src.display(),
        if d.quad { "quad_vertex" } else { "triangle_vertex" },
        d.quad as u32,
        d.mode,
        d.max_factor,
        d.winding,
        s.indexed,
        s.control_points,
        d.patch_count,
        d.instance_count,
        d.base_instance,
        WIDTH,
        HEIGHT,
        w(&d.factors.iter().map(|&h| h as u32).collect::<Vec<_>>()),
        w(&s.cp),
        w(&s.tags),
        w(&s.cp_indices),
    );
    let spec_path = dir.join("spec.txt");
    std::fs::write(&spec_path, spec).unwrap();
    let out = dir.join("metal.bin");
    let o = Command::new(metal_runner()).arg(&spec_path).arg(&out).output().expect("metal_tess");
    assert!(o.status.success(), "metal_tess: {}", String::from_utf8_lossy(&o.stderr));
    std::fs::read(&out).unwrap()
}

fn compile(s: &Scene) -> Compiled {
    let dir = case_dir("tess-air");
    let bitcode = air(&dir, SOURCE, "ios-metal2.4");
    let d = &s.draw;
    let vertex = if d.quad { "quad_vertex" } else { "triangle_vertex" };
    let vd = VertexDescriptor {
        layouts: (0..3)
            .map(|i| match i {
                1 => VertexBufferLayout { stride: 16, step_function: 4, step_rate: 1 },
                2 => VertexBufferLayout { stride: 4, step_function: 3, step_rate: 1 },
                _ => VertexBufferLayout { stride: 0, step_function: 1, step_rate: 1 },
            })
            .collect(),
        attributes: vec![
            VertexAttribute { format: 31, offset: 0, buffer_index: 1 },
            VertexAttribute { format: 36, offset: 0, buffer_index: 2 },
        ],
    };
    // MTLPixelFormatR32Uint, triangle topology class.
    let mut desc = render_desc(53, 3, 1, Some(vd));
    if let PipelineKind::Render(r) = &mut desc.kind {
        r.tessellation = Some(Tessellation {
            partition_mode: d.mode as u64,
            max_tessellation_factor: d.max_factor as u64,
            factor_scale_enabled: false,
            factor_format: 0,
            control_point_index_type: s.indexed,
            // MTLTessellationFactorStepFunctionPerPatch.
            factor_step_function: 1,
            output_winding_order: d.winding as u64,
        });
    }
    let modules = [module(&bitcode, vertex, 1), module(&bitcode, "code_fragment", 2)];
    airwgsl::compile(&modules, &desc, &host()).unwrap_or_else(|r| panic!("compile: {r}"))
}

/// The uniform block: base instance, control-point count, and zero residual offsets (every buffer is bound from its
/// start).
fn uniforms(block: &UniformBlock, s: &Scene, lengths: &[(u64, u64)]) -> Vec<u8> {
    let mut b = vec![0u8; block.size as usize];
    for f in &block.fields {
        let v: u32 = match &f.value {
            UniformValue::BaseInstance => s.draw.base_instance,
            UniformValue::PatchControlPoints => s.control_points,
            UniformValue::ResidualOffset { .. } | UniformValue::ControlPointIndexOffset => 0,
            UniformValue::BufferLength { index, .. } => {
                lengths.iter().find(|l| l.0 == *index).map(|l| l.1 as u32).unwrap()
            }
            other => panic!("uniform {other:?}"),
        };
        b[f.offset as usize..f.offset as usize + 4].copy_from_slice(&v.to_le_bytes());
    }
    b
}

fn render_wgpu(s: &Scene, c: &Compiled) -> Vec<u8> {
    let g = gpu();
    let dev = &g.device;
    assert_eq!(c.passes, vec![PassPlan::Tessellation { quad: s.draw.quad }]);
    let k = dispatch(&s.draw);
    let bytes = |w: &[u32]| -> Vec<u8> {
        let mut b: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        if b.is_empty() {
            b = vec![0; 4];
        }
        b
    };
    let cp_bytes = if s.indexed == 1 {
        let mut b: Vec<u8> = s.cp_indices.iter().flat_map(|&x| (x as u16).to_le_bytes()).collect();
        while !b.len().is_multiple_of(4) || b.is_empty() {
            b.push(0);
        }
        b
    } else {
        bytes(&s.cp_indices)
    };
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let init = |contents: &[u8], usage| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents, usage })
    };
    let cp = init(&bytes(&s.cp), storage);
    let tags = init(&bytes(&s.tags), storage);
    let cpi = init(&cp_bytes, storage);
    let lengths = [(1u64, 4 * s.cp.len() as u64), (2, 4 * s.tags.len() as u64)];
    let source = |st: ShaderStage| c.wgsl.iter().find(|w| w.stage == st).unwrap();
    let vs = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vertex"),
        source: wgpu::ShaderSource::Wgsl(source(ShaderStage::Vertex).source.clone().into()),
    });
    let fs = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fragment"),
        source: wgpu::ShaderSource::Wgsl(source(ShaderStage::Fragment).source.clone().into()),
    });
    let pipeline = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &vs,
            entry_point: Some(&source(ShaderStage::Vertex).entry_point),
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Cw,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &fs,
            entry_point: Some(&source(ShaderStage::Fragment).entry_point),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Uint,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });
    let mut groups = Vec::new();
    let uniform_buffers: Vec<wgpu::Buffer> = c
        .layout
        .groups
        .iter()
        .flat_map(|g| &g.entries)
        .filter_map(|e| match &e.source {
            BindingSource::Uniforms(u) => Some(init(&uniforms(u, s, &lengths), wgpu::BufferUsages::UNIFORM)),
            _ => None,
        })
        .collect();
    for grp in &c.layout.groups {
        let layout = pipeline.get_bind_group_layout(grp.group);
        let entries: Vec<wgpu::BindGroupEntry> = grp
            .entries
            .iter()
            .map(|e| {
                let buffer = match &e.source {
                    BindingSource::Uniforms(_) => &uniform_buffers[0],
                    BindingSource::MetalBuffer { index: 1, .. } => &cp,
                    BindingSource::MetalBuffer { index: 2, .. } => &tags,
                    BindingSource::TessellationVertices => &k.vertices,
                    BindingSource::ControlPointIndices => &cpi,
                    other => panic!("binding source {other:?}"),
                };
                wgpu::BindGroupEntry { binding: e.binding, resource: buffer.as_entire_binding() }
            })
            .collect();
        groups.push((
            grp.group,
            dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries }),
        ));
    }
    let target = dev.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let out = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (WIDTH * HEIGHT * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut e = dev.create_command_encoder(&Default::default());
    {
        let mut pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        for (i, grp) in &groups {
            pass.set_bind_group(*i, grp, &[]);
        }
        pass.set_index_buffer(k.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(&k.draw, 0);
    }
    e.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &out,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(WIDTH * 4), rows_per_image: None },
        },
        wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
    );
    g.queue.submit([e.finish()]);
    read(dev, &g.queue, &out, (WIDTH * HEIGHT * 4) as u64).iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn scene(quad: bool, mode: u32, factors: &[f32], patches: u32, instances: u32, indexed: u64) -> Scene {
    let per = if quad { 6 } else { 4 };
    let n = if quad { 4 } else { 3 };
    let mut halves = Vec::new();
    for p in 0..patches as usize {
        for k in 0..per {
            halves.push(f32_to_half(factors[(p * per + k) % factors.len()]));
        }
    }
    // Control points: xy offsets of a few 1/64 steps and w a distinct pattern; patches share none.
    let points = (patches * n + 4) as usize;
    let mut cp = Vec::new();
    for i in 0..points {
        let off = ((i % 5) as f32 - 2.0) / 64.0;
        cp.extend_from_slice(&[off.to_bits(), (-off).to_bits(), 0, (0x3f80_0000u32 + 977 * i as u32)]);
    }
    let tags = (0..patches + 2).map(|p| 0x9e37_79b9u32.wrapping_mul(p + 1)).collect();
    // A control-point index buffer reversing each patch's control points and shifting them by one patch.
    let cp_indices = (0..patches * n).map(|i| (i / n) * n + (n - 1 - i % n) + n).collect();
    let draw = Draw {
        quad,
        mode,
        max_factor: 16,
        winding: 0,
        step: 1,
        patch_count: patches,
        instance_count: instances,
        base_instance: if instances > 1 { 2 } else { 0 },
        factors: halves,
        ..Draw::default()
    };
    Scene { draw, indexed, control_points: n, cp, tags, cp_indices }
}

#[test]
fn tessellated_draws_equal_metal_through_the_pipeline() {
    let cases: Vec<(&str, Scene)> = vec![
        ("quad-integer", scene(true, 1, &[3.0, 5.0, 7.0, 2.0, 4.0, 6.0], 1, 1, 0)),
        ("quad-odd", scene(true, 2, &[3.3, 5.1, 7.7, 2.2, 4.9, 6.1], 2, 1, 0)),
        ("quad-even-instances", scene(true, 3, &[2.5, 6.0, 1.5, 9.0, 4.5, 3.5], 2, 2, 0)),
        ("quad-pow2-indexed16", scene(true, 0, &[3.0, 5.0, 2.0, 8.0, 4.0, 2.0], 3, 1, 1)),
        ("triangle-integer", scene(false, 1, &[6.0, 3.0, 5.0, 4.0], 1, 1, 0)),
        ("triangle-odd-indexed32", scene(false, 2, &[5.0, 3.0, 7.0, 5.0], 2, 1, 2)),
        ("triangle-even-instances", scene(false, 3, &[2.5, 7.5, 3.0, 6.0], 3, 2, 0)),
        ("triangle-minimum", scene(false, 1, &[1.0, 1.0, 1.0, 1.0], 1, 1, 0)),
    ];
    for (name, s) in &cases {
        let c = compile(s);
        let settings = s.draw.settings();
        for dp in 0..s.draw.patch_count {
            let h = s.draw.factors_of(dp, 0);
            assert!(!tess::counts(&settings, s.draw.patch_type(), &h, 1.0).batched, "{name}: a batched patch");
        }
        let want = metal(name, s);
        let got = render_wgpu(s, &c);
        assert!(want.iter().any(|&b| b != 0), "{name}: Metal drew nothing");
        if let Some(i) = got.chunks(4).zip(want.chunks(4)).position(|(a, b)| a != b) {
            panic!("{name}: pixel {} ({}, {}) differs", i, i as u32 % WIDTH, i as u32 / WIDTH);
        }
    }
    eprintln!("{} tessellated draws: wgpu equals Metal on every pixel", cases.len());
}
