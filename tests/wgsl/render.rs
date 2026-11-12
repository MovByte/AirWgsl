//! A render pipeline: a vertex function placing one triangle that covers the viewport and passing flat integer
//! varyings, and a fragment function writing its pixel position and the varyings to an integer color target and to a
//! device buffer. Every value is exact (pixel centers, the first vertex's flat values, integer formats), so the
//! target and the buffer must be byte-identical on Metal, wgpu and Dawn.

use super::harness::{RUN, air, case_dir, dawn_node, first_difference, gpu, tmp_dir};
use airwgsl::api::*;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct V {
    float4 pos [[position]];
    uint a [[flat]];
    int2 b [[flat]];
    ushort c [[flat]];
};

vertex V vs(uint vid [[vertex_id]]) {
    V o;
    float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));
    o.pos = float4(p * 2.0f - 1.0f, 0.5f, 1.0f);
    o.a = vid * 7u + 1u;
    o.b = int2(-3, int(vid) - 5);
    o.c = ushort(vid + 40000u);
    return o;
}

struct F {
    uint4 color [[color(0)]];
};

fragment F fs(V in [[stage_in]], device uint* out [[buffer(0)]]) {
    uint x = uint(in.pos.x);
    uint y = uint(in.pos.y);
    F f;
    f.color = uint4(x, y, in.a, uint(in.b.y) ^ uint(in.c));
    out[y * 16u + x] = x * 1000u + y + (in.a << 20) + uint(in.b.x);
    return f;
}
"#;

const W: u32 = 16;
const H: u32 = 16;

fn metal(dir: &std::path::Path, buffer: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let exe = tmp_dir().join("metal_render");
    if !exe.exists() {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wgsl/metal_render.swift");
        let o = Command::new("swiftc").args(["-O", "-o"]).arg(&exe).arg(&src).output().unwrap();
        assert!(o.status.success(), "swiftc: {}", String::from_utf8_lossy(&o.stderr));
    }
    let src = dir.join("ref.metal");
    std::fs::write(&src, SOURCE).unwrap();
    let b = dir.join("metal0.bin");
    std::fs::write(&b, buffer).unwrap();
    let o =
        Command::new(&exe).arg(&src).args(["vs", "fs", &W.to_string(), &H.to_string(), "3"]).arg(&b).output().unwrap();
    assert!(o.status.success(), "metal_render: {}", String::from_utf8_lossy(&o.stderr));
    (std::fs::read(dir.join("ref.metal.target")).unwrap(), std::fs::read(dir.join("metal0.bin.out")).unwrap())
}

fn compile_pipeline(bc: &[u8]) -> Compiled {
    let module = |f: &str, t: u64| AirModule {
        hash: [0; 32],
        bitcode: Arc::from(bc),
        entry: Some(EntryPoint { function: f.into(), function_type: t, constants: Vec::new() }),
    };
    let desc = PipelineDesc {
        kind: PipelineKind::Render(RenderState {
            vertex_descriptor: None,
            color_attachments: vec![ColorAttachment {
                pixel_format: 123, // MTLPixelFormatRGBA32Uint
                blending_enabled: false,
                rgb_blend_operation: 0,
                alpha_blend_operation: 0,
                source_rgb_blend_factor: 1,
                destination_rgb_blend_factor: 0,
                source_alpha_blend_factor: 1,
                destination_alpha_blend_factor: 0,
                write_mask: 0xf,
            }],
            depth_attachment_pixel_format: 0,
            stencil_attachment_pixel_format: 0,
            raster_sample_count: 1,
            alpha_to_coverage_enabled: false,
            alpha_to_one_enabled: false,
            rasterization_enabled: true,
            input_primitive_topology: 3,
            primitive_type: Some(3),
            tessellation: None,
            max_vertex_amplification_count: 1,
        }),
        bindings: Vec::new(),
    };
    compile(&[module("vs", 1), module("fs", 2)], &desc, &HostCaps::default()).unwrap_or_else(|r| panic!("{}", r))
}

fn wgpu_run(c: &Compiled, buffer: &[u8]) -> (Vec<u8>, Vec<u8>) {
    use wgpu::util::DeviceExt;
    let _guard = RUN.lock().unwrap_or_else(|e| e.into_inner());
    let g = gpu();
    let d = &g.device;
    let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
    let vs = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(c.wgsl[0].source.as_str().into()),
    });
    let fs = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(c.wgsl[1].source.as_str().into()),
    });
    let pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &vs,
            entry_point: Some(&c.wgsl[0].entry_point),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &fs,
            entry_point: Some(&c.wgsl[1].entry_point),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Uint,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let target = d.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Uint,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let sb = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: buffer,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let mut groups: std::collections::BTreeMap<u32, Vec<(u32, wgpu::Buffer)>> = Default::default();
    for grp in &c.layout.groups {
        for e in &grp.entries {
            match &e.source {
                BindingSource::Uniforms(u) => {
                    let ub = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: &vec![0u8; u.size as usize],
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    groups.entry(grp.group).or_default().push((e.binding, ub));
                }
                BindingSource::MetalBuffer { stage: MetalStage::Fragment, index: 0 } => {
                    groups.entry(grp.group).or_default().push((e.binding, sb.clone()))
                }
                other => panic!("binding {:?}", other),
            }
        }
    }
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let view = target.create_view(&Default::default());
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        for (grp, es) in &groups {
            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(*grp),
                entries: &es
                    .iter()
                    .map(|(b, buf)| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() })
                    .collect::<Vec<_>>(),
            });
            pass.set_bind_group(*grp, &bg, &[]);
        }
        pass.draw(0..3, 0..1);
    }
    let tread = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (W * H * 16) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &tread,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 16), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    let bread = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&sb, 0, &bread, 0, buffer.len() as u64);
    g.queue.submit([enc.finish()]);
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("wgpu validation: {}", e);
    }
    let read = |b: &wgpu::Buffer| {
        let s = b.slice(..);
        s.map_async(wgpu::MapMode::Read, |_| {});
        d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        s.get_mapped_range().to_vec()
    };
    (read(&tread), read(&bread))
}

#[test]
fn vertex_and_fragment_pipeline() {
    let dir = case_dir("render");
    let bc = air(&dir, SOURCE);
    let c = compile_pipeline(&bc);
    std::fs::write(dir.join("vs.wgsl"), &c.wgsl[0].source).unwrap();
    std::fs::write(dir.join("fs.wgsl"), &c.wgsl[1].source).unwrap();
    let buffer = vec![0xa5u8; (W * H * 4) as usize];
    let (mt, mb) = metal(&dir, &buffer);
    let (wt, wb) = wgpu_run(&c, &buffer);
    assert_eq!(first_difference(&mt, &wt), None, "target differs on wgpu; WGSL in {}", dir.display());
    assert_eq!(first_difference(&mb, &wb), None, "buffer differs on wgpu");
    // Every pixel was shaded.
    assert!(mb.chunks(4).all(|w| w != [0xa5; 4]));
    if let Some(dn) = dawn_node() {
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wgsl/dawn_render.cjs");
        let mut args: Vec<String> = Vec::new();
        for grp in &c.layout.groups {
            for e in &grp.entries {
                match &e.source {
                    BindingSource::Uniforms(u) => {
                        let p = dir.join("uniforms.bin");
                        std::fs::write(&p, vec![0u8; u.size as usize]).unwrap();
                        args.push(format!("{}:{}:uniform:{}", grp.group, e.binding, p.display()));
                    }
                    _ => {
                        let p = dir.join("dawn0.bin");
                        std::fs::write(&p, &buffer).unwrap();
                        args.push(format!("{}:{}:storage:{}", grp.group, e.binding, p.display()));
                    }
                }
            }
        }
        let o = Command::new("node")
            .arg(script)
            .arg(dn)
            .arg(dir.join("vs.wgsl"))
            .arg(dir.join("fs.wgsl"))
            .arg(&c.wgsl[0].entry_point)
            .args([W.to_string(), H.to_string(), "3".into()])
            .arg(dir.join("dawn.target"))
            .args(&args)
            .output()
            .unwrap();
        assert!(o.status.success(), "dawn_render: {}", String::from_utf8_lossy(&o.stderr));
        let dt = std::fs::read(dir.join("dawn.target")).unwrap();
        let db = std::fs::read(dir.join("dawn0.bin.out")).unwrap();
        assert_eq!(first_difference(&mt, &dt), None, "target differs on Dawn");
        assert_eq!(first_difference(&mb, &db), None, "buffer differs on Dawn");
        eprintln!("render: Metal = wgpu = Dawn");
    } else {
        eprintln!("render: Metal = wgpu (Dawn not run: AW_DAWN_NODE unset)");
    }
}
