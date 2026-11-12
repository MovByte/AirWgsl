//! A small executor for framebuffer-fetch pipelines, playing the executor's part of [`PassPlan::PerPrimitive`] on wgpu
//! and on Dawn: before each group of primitives the pass ends, the target is copied into the snapshot texture and a
//! pass that loads the target begins; the group's draws read the snapshot. Groups are single primitives (the baseline)
//! or the overlap prover's. The same scene also runs on this Mac's Metal (`metal_fetch.swift`) as the reference.

use super::stage::{Slot, case_dir, dawn_node, gpu, uniform_bytes};
use airwgsl::api::*;
use airwgsl::fetch::overlap::{Draw as ProverDraw, Primitive, Topology, Viewport};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

static RUN: Mutex<()> = Mutex::new(());
static DAWN: Mutex<()> = Mutex::new(());

/// A color attachment format of the scenes: (WebGPU name, `MTLPixelFormat`, bytes per texel).
#[derive(Clone, Copy, Debug)]
pub struct Format {
    pub webgpu: &'static str,
    pub metal: u64,
    pub bpp: u32,
}

pub const R32UINT: Format = Format { webgpu: "r32uint", metal: 53, bpp: 4 };
pub const R16UINT: Format = Format { webgpu: "r16uint", metal: 23, bpp: 2 };
pub const RGBA8UNORM: Format = Format { webgpu: "rgba8unorm", metal: 70, bpp: 4 };
pub const RGBA16FLOAT: Format = Format { webgpu: "rgba16float", metal: 115, bpp: 8 };

/// One draw: vertices `start..start + count` of each of `instances` instances, with vertex buffer 0.
#[derive(Clone)]
pub struct SceneDraw {
    pub start: u32,
    pub count: u32,
    pub instances: u32,
    pub vertices: Vec<u8>,
}

pub struct Scene {
    pub compiled: Compiled,
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    /// Initial texels (single-sampled targets), else cleared to `clear`.
    pub initial: Option<Vec<u8>>,
    pub clear: [f64; 4],
    /// One-plus-one blending.
    pub blend: bool,
    /// Whether the fragment function writes the attachment (write mask all, else none).
    pub write: bool,
    /// `MTLPrimitiveType`: 0 point, 1 line, 2 line strip, 3 triangle, 4 triangle strip.
    pub primitive: u64,
    pub draws: Vec<SceneDraw>,
    /// The fragment buffer at index 0, shared by the draws and read back.
    pub fragment_buffer: Option<Vec<u8>>,
}

/// What a scene produced: the target (per sample for a multisampled one) and the fragment buffer.
#[derive(Debug, PartialEq, Eq)]
pub struct Output {
    pub target: Vec<u8>,
    pub buffer: Option<Vec<u8>>,
}

/// One WebGPU draw of a step: scene draw `draw`, vertices `first..first + count`, instances from `first_instance`.
#[derive(Clone, Debug)]
pub struct Call {
    pub draw: usize,
    pub first: u32,
    pub count: u32,
    pub first_instance: u32,
    pub instances: u32,
}

fn topology(primitive: u64) -> (Topology, bool, &'static str) {
    match primitive {
        0 => (Topology::Point, false, "point-list"),
        1 => (Topology::Line, false, "line-list"),
        2 => (Topology::Line, true, "line-strip"),
        3 => (Topology::Triangle, false, "triangle-list"),
        4 => (Topology::Triangle, true, "triangle-strip"),
        p => panic!("primitive type {}", p),
    }
}

/// The primitives of a draw in order (instance-major), with each one's first vertex.
fn primitives(s: &Scene, d: &SceneDraw) -> Vec<(Primitive, u32)> {
    let (t, strip, _) = topology(s.primitive);
    let n = t.vertices() as u32;
    let (count, step) = if strip { (d.count.saturating_sub(n - 1), 1) } else { (d.count / n, n) };
    let mut out = Vec::new();
    for inst in 0..d.instances {
        for i in 0..count {
            let first = d.start + i * step;
            let mut v = [0u32; 3];
            for (k, x) in v.iter_mut().enumerate().take(n as usize) {
                *x = first + k as u32;
            }
            out.push((Primitive { vertices: v, instance: inst }, first));
        }
    }
    out
}

/// The steps of a scene: each primitive alone (`prove` false) or the overlap prover's groups.
pub fn steps(s: &Scene, prove: bool) -> Vec<Vec<Call>> {
    let (t, _, _) = topology(s.primitive);
    let n = t.vertices() as u32;
    let prover = match &s.compiled.passes[..] {
        [PassPlan::PerPrimitive { overlap, .. }] => overlap.clone(),
        other => panic!("not a fetch plan: {:?}", other),
    };
    let mut steps = Vec::new();
    for (k, d) in s.draws.iter().enumerate() {
        let prims = primitives(s, d);
        let groups: Vec<Range<usize>> = if prove {
            let p = prover.as_ref().expect("an overlap prover");
            let buffers = [(0u64, d.vertices.as_slice())];
            let draw = ProverDraw {
                topology: t,
                buffers: &buffers,
                base_vertex: 0,
                base_instance: 0,
                viewport: Viewport { x: 0.0, y: 0.0, width: s.width as f64, height: s.height as f64 },
                target: [s.width, s.height],
            };
            let list: Vec<Primitive> = prims.iter().map(|p| p.0).collect();
            p.groups(&draw, &list)
        } else {
            (0..prims.len()).map(|i| i..i + 1).collect()
        };
        let (_, strip, _) = topology(s.primitive);
        for g in groups {
            let mut calls: Vec<Call> = Vec::new();
            for i in g {
                let (p, first) = prims[i];
                // The next primitive of the same instance continues the previous call's vertices: a list's after its
                // last vertex, a strip's at its last n - 1 vertices.
                let next = |c: &Call| if strip { c.first + c.count + 1 - n } else { c.first + c.count };
                match calls.last_mut() {
                    Some(c) if c.first_instance == p.instance && first == next(c) => c.count = first + n - c.first,
                    _ => calls.push(Call { draw: k, first, count: n, first_instance: p.instance, instances: 1 }),
                }
            }
            steps.push(calls);
        }
    }
    steps
}

fn bindings(layout: &BindingLayout) -> Vec<(u32, &BindingEntry)> {
    layout.groups.iter().flat_map(|g| g.entries.iter().map(move |e| (g.group, e))).collect()
}

fn padded(data: &[u8]) -> Vec<u8> {
    let mut v = data.to_vec();
    while !v.len().is_multiple_of(4) || v.is_empty() {
        v.push(0);
    }
    v
}

fn uniform(block: &UniformBlock, d: &SceneDraw, fragment: Option<&Vec<u8>>) -> Vec<u8> {
    let mut slots = vec![Slot { stage: MetalStage::Vertex, index: 0, data: d.vertices.clone(), offset: 0 }];
    if let Some(f) = fragment {
        slots.push(Slot { stage: MetalStage::Fragment, index: 0, data: f.clone(), offset: 0 });
    }
    let draw = super::stage::Draw::new(d.start, d.count, slots);
    uniform_bytes(block, &draw, None)
}

fn wgpu_format(f: &str) -> wgpu::TextureFormat {
    match f {
        "r32uint" => wgpu::TextureFormat::R32Uint,
        "r32sint" => wgpu::TextureFormat::R32Sint,
        "r32float" => wgpu::TextureFormat::R32Float,
        "r16uint" => wgpu::TextureFormat::R16Uint,
        "r8unorm" => wgpu::TextureFormat::R8Unorm,
        "rg16sint" => wgpu::TextureFormat::Rg16Sint,
        "rgba8unorm" => wgpu::TextureFormat::Rgba8Unorm,
        "rgba8unorm-srgb" => wgpu::TextureFormat::Rgba8UnormSrgb,
        "bgra8unorm" => wgpu::TextureFormat::Bgra8Unorm,
        "rgba8uint" => wgpu::TextureFormat::Rgba8Uint,
        "rgba8sint" => wgpu::TextureFormat::Rgba8Sint,
        "rgba16sint" => wgpu::TextureFormat::Rgba16Sint,
        "rgba16float" => wgpu::TextureFormat::Rgba16Float,
        "rgba32uint" => wgpu::TextureFormat::Rgba32Uint,
        "rgba32sint" => wgpu::TextureFormat::Rgba32Sint,
        f => panic!("format {}", f),
    }
}

fn read(d: &wgpu::Device, b: &wgpu::Buffer) -> Vec<u8> {
    let s = b.slice(..);
    s.map_async(wgpu::MapMode::Read, |_| {});
    d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    s.get_mapped_range().to_vec()
}

const READ_SAMPLES: &str = "@group(0) @binding(0) var t: texture_multisampled_2d<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let w = textureDimensions(t).x;
    let n = textureNumSamples(t);
    for (var s = 0u; s < n; s = s + 1u) {
        o[(g.y * w + g.x) * n + s] = textureLoad(t, g.xy, i32(s)).x;
    }
}
";

/// One binding of a call: (group, binding, the buffer made for it, its source).
type CallBinding<'a> = (u32, u32, Option<wgpu::Buffer>, &'a BindingSource);

/// Run the steps on wgpu.
pub fn run_wgpu(s: &Scene, steps: &[Vec<Call>]) -> Output {
    use wgpu::util::DeviceExt;
    let _guard = RUN.lock().unwrap_or_else(|e| e.into_inner());
    let g = gpu();
    let d = &g.device;
    let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = |src: &str| {
        d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
    };
    let vs = shader(&s.compiled.wgsl[0].source);
    let fs = shader(&s.compiled.wgsl[1].source);
    let mut layouts = Vec::new();
    for grp in &s.compiled.layout.groups {
        while layouts.len() < grp.group as usize {
            layouts.push(d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] }));
        }
        let entries: Vec<wgpu::BindGroupLayoutEntry> = grp
            .entries
            .iter()
            .map(|e| wgpu::BindGroupLayoutEntry {
                binding: e.binding,
                visibility: wgpu::ShaderStages::from_bits_truncate(e.visibility.0),
                ty: match &e.resource {
                    BindingResource::Buffer { ty, .. } => wgpu::BindingType::Buffer {
                        ty: match ty {
                            BufferBindingType::Uniform => wgpu::BufferBindingType::Uniform,
                            BufferBindingType::Storage => wgpu::BufferBindingType::Storage { read_only: false },
                            BufferBindingType::ReadOnlyStorage => wgpu::BufferBindingType::Storage { read_only: true },
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    BindingResource::Texture { sample_type, multisampled, .. } => wgpu::BindingType::Texture {
                        sample_type: match sample_type {
                            TextureSampleType::Uint => wgpu::TextureSampleType::Uint,
                            TextureSampleType::Sint => wgpu::TextureSampleType::Sint,
                            _ => wgpu::TextureSampleType::Float { filterable: false },
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: *multisampled,
                    },
                    r => panic!("resource {:?}", r),
                },
                count: None,
            })
            .collect();
        layouts.push(d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries }));
    }
    let refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
    let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &refs,
        immediate_size: 0,
    });
    let format = wgpu_format(s.format.webgpu);
    let one = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    let targets = [Some(wgpu::ColorTargetState {
        format,
        blend: s.blend.then_some(wgpu::BlendState { color: one, alpha: one }),
        write_mask: if s.write { wgpu::ColorWrites::ALL } else { wgpu::ColorWrites::empty() },
    })];
    let pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &vs,
            entry_point: Some(&s.compiled.wgsl[0].entry_point),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: match topology(s.primitive).2 {
                "point-list" => wgpu::PrimitiveTopology::PointList,
                "line-list" => wgpu::PrimitiveTopology::LineList,
                "line-strip" => wgpu::PrimitiveTopology::LineStrip,
                "triangle-list" => wgpu::PrimitiveTopology::TriangleList,
                _ => wgpu::PrimitiveTopology::TriangleStrip,
            },
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState { count: s.samples, ..Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &fs,
            entry_point: Some(&s.compiled.wgsl[1].entry_point),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    });
    let size = wgpu::Extent3d { width: s.width, height: s.height, depth_or_array_layers: 1 };
    let texture = |usage| {
        d.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: s.samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = texture(
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    // WebGPU requires RENDER_ATTACHMENT of a multisampled texture (createTexture validation).
    let snapshot = texture(
        wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING
            | if s.samples > 1 { wgpu::TextureUsages::RENDER_ATTACHMENT } else { wgpu::TextureUsages::empty() },
    );
    if let Some(init) = &s.initial {
        g.queue.write_texture(
            color.as_image_copy(),
            init,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(s.width * s.format.bpp),
                rows_per_image: None,
            },
            size,
        );
    }
    let fragment = s.fragment_buffer.as_ref().map(|f| {
        d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &padded(f),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    });
    let snapshot_view = snapshot.create_view(&Default::default());
    let mut enc = d.create_command_encoder(&Default::default());
    let view = color.create_view(&Default::default());
    if s.initial.is_none() {
        let c = s.clear;
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: c[0], g: c[1], b: c[2], a: c[3] }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    for step in steps {
        enc.copy_texture_to_texture(color.as_image_copy(), snapshot.as_image_copy(), size);
        // The buffers of each call's bindings first, then the bind groups that borrow them.
        let resources: Vec<Vec<CallBinding>> = step
            .iter()
            .map(|call| {
                let sd = &s.draws[call.draw];
                bindings(&s.compiled.layout)
                    .into_iter()
                    .map(|(grp, e)| {
                        let owned = match &e.source {
                            BindingSource::Uniforms(u) => {
                                Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: None,
                                    contents: &uniform(u, sd, s.fragment_buffer.as_ref()),
                                    usage: wgpu::BufferUsages::UNIFORM,
                                }))
                            }
                            BindingSource::MetalBuffer { stage: MetalStage::Vertex, index: 0 } => {
                                Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: None,
                                    contents: &padded(&sd.vertices),
                                    usage: wgpu::BufferUsages::STORAGE,
                                }))
                            }
                            BindingSource::MetalBuffer { stage: MetalStage::Fragment, index: 0 }
                            | BindingSource::FetchSnapshot { .. } => None,
                            other => panic!("binding source {:?}", other),
                        };
                        (grp, e.binding, owned, &e.source)
                    })
                    .collect()
            })
            .collect();
        let groups: Vec<Vec<wgpu::BindGroup>> = resources
            .iter()
            .map(|rs| {
                (0..layouts.len())
                    .map(|k| {
                        let es: Vec<wgpu::BindGroupEntry> = rs
                            .iter()
                            .filter(|r| r.0 == k as u32)
                            .map(|(_, b, owned, src)| wgpu::BindGroupEntry {
                                binding: *b,
                                resource: match (owned, src) {
                                    (Some(buf), _) => buf.as_entire_binding(),
                                    (None, BindingSource::FetchSnapshot { .. }) => {
                                        wgpu::BindingResource::TextureView(&snapshot_view)
                                    }
                                    (None, _) => fragment.as_ref().unwrap().as_entire_binding(),
                                },
                            })
                            .collect();
                        d.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layouts[k],
                            entries: &es,
                        })
                    })
                    .collect()
            })
            .collect();
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        for (call, bgs) in step.iter().zip(&groups) {
            for (k, bg) in bgs.iter().enumerate() {
                pass.set_bind_group(k as u32, bg, &[]);
            }
            pass.draw(call.first..call.first + call.count, call.first_instance..call.first_instance + call.instances);
        }
    }
    let mut reads = Vec::new();
    if s.samples == 1 {
        let row = (s.width * s.format.bpp).div_ceil(256) * 256;
        let b = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * s.height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &b,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            size,
        );
        reads.push((b, row, s.width * s.format.bpp));
    } else {
        let n = (s.width * s.height * s.samples * 4) as u64;
        let out = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let module = shader(READ_SAMPLES);
        let cp = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let tv = color.create_view(&Default::default());
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &cp.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tv) },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        {
            let mut cpass = enc.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&cp);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(s.width, s.height, 1);
        }
        let b = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&out, 0, &b, 0, n);
        reads.push((b, n as u32, n as u32));
    }
    if let Some(f) = &fragment {
        let b = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: f.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(f, 0, &b, 0, f.size());
        reads.push((b, f.size() as u32, f.size() as u32));
    }
    g.queue.submit([enc.finish()]);
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("wgpu validation: {}\n{}", e, s.compiled.wgsl.iter().map(|w| w.source.as_str()).collect::<String>());
    }
    let mut outs = Vec::new();
    for (b, row, len) in &reads {
        let raw = read(d, b);
        let mut packed = Vec::new();
        for y in 0..raw.len() as u32 / row {
            packed.extend_from_slice(&raw[(y * row) as usize..(y * row + len) as usize]);
        }
        outs.push(packed);
    }
    let buffer = if fragment.is_some() {
        let mut b = outs.pop();
        if let (Some(b), Some(f)) = (&mut b, &s.fragment_buffer) {
            b.truncate(f.len());
        }
        b
    } else {
        None
    };
    Output { target: outs.pop().unwrap(), buffer }
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Run the steps on Dawn.
pub fn run_dawn(dawn: &Path, dir: &Path, s: &Scene, steps: &[Vec<Call>], tag: &str) -> Output {
    let mut files = 0;
    let mut file = |data: &[u8]| {
        let p = dir.join(format!("{}{}.bin", tag, files));
        files += 1;
        std::fs::write(&p, data).unwrap();
        json_str(&p.display().to_string())
    };
    let mut groups = Vec::new();
    for g in &s.compiled.layout.groups {
        let entries: Vec<String> = g
            .entries
            .iter()
            .map(|e| {
                let r = match &e.resource {
                    BindingResource::Buffer { ty, .. } => format!(
                        "\"buffer\":{{\"type\":\"{}\"}}",
                        match ty {
                            BufferBindingType::Uniform => "uniform",
                            BufferBindingType::Storage => "storage",
                            BufferBindingType::ReadOnlyStorage => "read-only-storage",
                        }
                    ),
                    BindingResource::Texture { sample_type, multisampled, .. } => format!(
                        "\"texture\":{{\"sampleType\":\"{}\",\"viewDimension\":\"2d\",\"multisampled\":{}}}",
                        match sample_type {
                            TextureSampleType::Uint => "uint",
                            TextureSampleType::Sint => "sint",
                            _ => "unfilterable-float",
                        },
                        multisampled
                    ),
                    r => panic!("resource {:?}", r),
                };
                format!("{{\"binding\":{},\"visibility\":{},{}}}", e.binding, e.visibility.0, r)
            })
            .collect();
        groups.push(format!("{{\"group\":{},\"entries\":[{}]}}", g.group, entries.join(",")));
    }
    let mut step_json = Vec::new();
    for step in steps {
        let mut calls = Vec::new();
        for call in step {
            let sd = &s.draws[call.draw];
            let mut binds = Vec::new();
            for (grp, e) in bindings(&s.compiled.layout) {
                let (kind, f) = match &e.source {
                    BindingSource::Uniforms(u) => ("uniform", file(&uniform(u, sd, s.fragment_buffer.as_ref()))),
                    BindingSource::MetalBuffer { stage: MetalStage::Vertex, index: 0 } => {
                        ("storage", file(&padded(&sd.vertices)))
                    }
                    BindingSource::MetalBuffer { stage: MetalStage::Fragment, index: 0 } => ("fragment", "null".into()),
                    BindingSource::FetchSnapshot { .. } => ("snapshot", "null".into()),
                    other => panic!("binding source {:?}", other),
                };
                binds.push(format!(
                    "{{\"group\":{},\"binding\":{},\"kind\":\"{}\",\"file\":{}}}",
                    grp, e.binding, kind, f
                ));
            }
            calls.push(format!(
                "{{\"bindings\":[{}],\"first\":{},\"count\":{},\"firstInstance\":{},\"instances\":{}}}",
                binds.join(","),
                call.first,
                call.count,
                call.first_instance,
                call.instances
            ));
        }
        step_json.push(format!("[{}]", calls.join(",")));
    }
    let vs = dir.join(format!("{}vs.wgsl", tag));
    std::fs::write(&vs, &s.compiled.wgsl[0].source).unwrap();
    let fs = dir.join(format!("{}fs.wgsl", tag));
    std::fs::write(&fs, &s.compiled.wgsl[1].source).unwrap();
    let out = dir.join(format!("{}dawn.out", tag));
    let initial = s.initial.as_ref().map(|i| file(i)).unwrap_or_else(|| "null".into());
    let fragment = s.fragment_buffer.as_ref().map(|f| file(&padded(f))).unwrap_or_else(|| "null".into());
    let scene = format!(
        "{{\"vs\":{},\"fs\":{},\"entry\":{},\"groups\":[{}],\"format\":\"{}\",\"bpp\":{},\"width\":{},\"height\":{},\"samples\":{},\"initial\":{},\"clear\":[{}],\"blend\":{},\"write\":{},\"topology\":\"{}\",\"fragmentBuffer\":{},\"steps\":[{}],\"out\":{},\"features\":[]}}",
        json_str(&vs.display().to_string()),
        json_str(&fs.display().to_string()),
        json_str(&s.compiled.wgsl[0].entry_point),
        groups.join(","),
        s.format.webgpu,
        s.format.bpp,
        s.width,
        s.height,
        s.samples,
        initial,
        s.clear.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","),
        s.blend,
        s.write,
        topology(s.primitive).2,
        fragment,
        step_json.join(","),
        json_str(&out.display().to_string()),
    );
    let sp = dir.join(format!("{}scene.json", tag));
    std::fs::write(&sp, scene).unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fetch/dawn_fetch.cjs");
    let _guard = DAWN.lock().unwrap_or_else(|e| e.into_inner());
    let marker = PathBuf::from(format!("{}.complete", out.display()));
    let mut attempt = 0;
    loop {
        let _ = std::fs::remove_file(&marker);
        let o = Command::new("node").arg(&script).arg(dawn).arg(&sp).output().unwrap();
        let complete = marker.exists();
        match (o.status.success(), complete, o.status.code()) {
            (true, _, _) | (false, true, None) => break,
            // dawn.node ended by a signal before completing (seen under load, as in tests/stage): up to two more
            // runs, and a third failure fails the test.
            (false, false, None) if attempt < 2 => attempt += 1,
            _ => panic!(
                "dawn_fetch ({}): {}{}",
                o.status,
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            ),
        }
    }
    let buffer = s.fragment_buffer.as_ref().map(|f| {
        let mut b = std::fs::read(format!("{}.buffer", out.display())).unwrap();
        b.truncate(f.len());
        b
    });
    Output { target: std::fs::read(&out).unwrap(), buffer }
}

/// The Metal runner, built once.
fn metal_runner() -> PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let exe = super::stage::tmp_dir().join("metal_fetch_fb");
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fetch/metal_fetch.swift");
        let o = Command::new("swiftc").args(["-O", "-o"]).arg(&exe).arg(&src).output().expect("swiftc");
        assert!(o.status.success(), "swiftc: {}", String::from_utf8_lossy(&o.stderr));
        exe
    })
    .clone()
}

/// Run the scene on this Mac's Metal from MSL `source`; `Err` with Metal's message when it refuses the pipeline.
pub fn run_metal(dir: &Path, s: &Scene, source: &str, vertex: &str, fragment: &str) -> Result<Output, String> {
    let src = dir.join("metal.metal");
    std::fs::write(&src, source).unwrap();
    let mut draws = Vec::new();
    for (k, d) in s.draws.iter().enumerate() {
        let p = dir.join(format!("metal-v{}.bin", k));
        std::fs::write(&p, &d.vertices).unwrap();
        draws.push(format!(
            "{{\"vertices\":{},\"start\":{},\"count\":{},\"instances\":{}}}",
            json_str(&p.display().to_string()),
            d.start,
            d.count,
            d.instances
        ));
    }
    let initial = s.initial.as_ref().map(|i| {
        let p = dir.join("metal-initial.bin");
        std::fs::write(&p, i).unwrap();
        json_str(&p.display().to_string())
    });
    let fb = s.fragment_buffer.as_ref().map(|f| {
        let p = dir.join("metal-fragment.bin");
        std::fs::write(&p, f).unwrap();
        json_str(&p.display().to_string())
    });
    let out = dir.join("metal.out");
    let scene = format!(
        "{{\"source\":{},\"vertex\":\"{}\",\"fragment\":\"{}\",\"format\":{},\"bpp\":{},\"width\":{},\"height\":{},\"samples\":{},\"initial\":{},\"clear\":[{}],\"blend\":{},\"write\":{},\"primitive\":{},\"fragmentBuffer\":{},\"draws\":[{}],\"out\":{}}}",
        json_str(&src.display().to_string()),
        vertex,
        fragment,
        s.format.metal,
        s.format.bpp,
        s.width,
        s.height,
        s.samples,
        initial.unwrap_or_else(|| "null".into()),
        s.clear.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","),
        s.blend,
        s.write,
        s.primitive,
        fb.unwrap_or_else(|| "null".into()),
        draws.join(","),
        json_str(&out.display().to_string()),
    );
    let sp = dir.join("metal-scene.json");
    std::fs::write(&sp, scene).unwrap();
    let o = Command::new(metal_runner()).arg(&sp).output().unwrap();
    assert!(o.status.success(), "metal_fetch: {}", String::from_utf8_lossy(&o.stderr));
    let stdout = String::from_utf8_lossy(&o.stdout);
    if let Some(r) = stdout.strip_prefix("refused: ") {
        return Err(r.trim().to_string());
    }
    let buffer = s.fragment_buffer.as_ref().map(|_| std::fs::read(format!("{}.buffer", out.display())).unwrap());
    Ok(Output { target: std::fs::read(&out).unwrap(), buffer })
}

/// Run a scene with each primitive alone and with the prover's groups (when it has a prover) on wgpu and, when
/// `AW_DAWN_NODE` names it, on Dawn; all runs must agree. Returns the output and whether Dawn ran.
pub fn render(name: &str, s: &Scene) -> (Output, bool) {
    let dir = case_dir(name);
    let base = steps(s, false);
    let w = run_wgpu(s, &base);
    let proved = matches!(&s.compiled.passes[..], [PassPlan::PerPrimitive { overlap: Some(_), .. }]);
    let grouped = proved.then(|| steps(s, true));
    if let Some(g) = &grouped {
        assert_eq!(run_wgpu(s, g), w, "{}: the prover's groups differ from the baseline on wgpu", name);
    }
    let dawn = dawn_node();
    if let Some(dn) = &dawn {
        assert_eq!(run_dawn(dn, &dir, s, &base, "base-"), w, "{}: Dawn and wgpu differ", name);
        if let Some(g) = &grouped {
            assert_eq!(run_dawn(dn, &dir, s, g, "grouped-"), w, "{}: the prover's groups differ on Dawn", name);
        }
    }
    (w, dawn.is_some())
}
