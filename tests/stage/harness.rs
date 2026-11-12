//! A small executor for the stage tests: draws and dispatches of compiled pipelines on wgpu (wgpu 29, whose wgpu-core
//! is the one wgpu-native v29 wraps) and on Dawn through its Node.js binding when `AW_DAWN_NODE` names `dawn.node`.
//!
//! It plays the executor's part of the contract: every binding of the compiled layout is served from its source (the
//! uniform block field by field, Metal buffer slots bound from offset 0 with the Metal offset as the residual), the
//! vertex stage has no WebGPU vertex buffers (attributes are pulled), and indexed draws use WebGPU's index buffer with
//! the strip index format set for strip topologies. Integer color targets are read back byte for byte; a 4-sample
//! target is read per sample with a compute shader; a depth target is copied out.

use airwgsl::api::*;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

pub fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("stage-tests");
    std::fs::create_dir_all(&d).unwrap();
    d
}

pub fn case_dir(name: &str) -> PathBuf {
    let d = tmp_dir().join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `dawn.node`, when the environment names it.
pub fn dawn_node() -> Option<PathBuf> {
    std::env::var_os("AW_DAWN_NODE").map(PathBuf::from)
}

/// The target's AIR for an MSL source (`std` is the `-std` option).
pub fn air(dir: &Path, source: &str, std: &str) -> Vec<u8> {
    let src = dir.join("s.metal");
    std::fs::write(&src, source).unwrap();
    let out = dir.join("s.air");
    let o = Command::new("xcrun")
        .args(["-sdk", "iphoneos", "metal", &format!("-std={}", std), "-target", "air64-apple-ios17.7", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .output()
        .expect("xcrun metal");
    assert!(o.status.success(), "metal: {}", String::from_utf8_lossy(&o.stderr));
    std::fs::read(&out).unwrap()
}

pub fn module(bitcode: &[u8], function: &str, function_type: u64) -> AirModule {
    AirModule {
        hash: [0; 32],
        bitcode: Arc::from(bitcode),
        entry: Some(EntryPoint { function: function.into(), function_type, constants: Vec::new() }),
    }
}

pub fn host() -> HostCaps {
    let mut h = HostCaps::default();
    h.features.insert("core-features-and-limits".into());
    h
}

pub fn color(pixel_format: u64) -> ColorAttachment {
    ColorAttachment {
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

/// A render pipeline description: one color attachment, the given topology class and raster sample count.
pub fn render_desc(pixel_format: u64, topology: u64, samples: u64, vd: Option<VertexDescriptor>) -> PipelineDesc {
    PipelineDesc {
        kind: PipelineKind::Render(RenderState {
            vertex_descriptor: vd,
            color_attachments: vec![color(pixel_format)],
            depth_attachment_pixel_format: 0,
            stencil_attachment_pixel_format: 0,
            raster_sample_count: samples,
            alpha_to_coverage_enabled: false,
            alpha_to_one_enabled: false,
            rasterization_enabled: true,
            input_primitive_topology: topology,
            primitive_type: None,
            tessellation: None,
            max_vertex_amplification_count: 1,
        }),
        bindings: Vec::new(),
    }
}

/// A Metal buffer slot bound for a draw or dispatch.
#[derive(Clone)]
pub struct Slot {
    pub stage: MetalStage,
    pub index: u64,
    pub data: Vec<u8>,
    /// The Metal offset.
    pub offset: u64,
}

/// An index buffer: bytes, `uint32` or `uint16`, Metal byte offset.
#[derive(Clone)]
pub struct Indices {
    pub data: Vec<u8>,
    pub uint32: bool,
    pub offset: u64,
}

#[derive(Clone)]
pub struct Draw {
    pub vertex_start: u32,
    /// Vertex count, or index count for an indexed draw.
    pub count: u32,
    pub instance_count: u32,
    pub base_instance: u32,
    pub base_vertex: i32,
    pub indices: Option<Indices>,
    pub slots: Vec<Slot>,
}

impl Draw {
    pub fn new(vertex_start: u32, count: u32, slots: Vec<Slot>) -> Draw {
        Draw { vertex_start, count, instance_count: 1, base_instance: 0, base_vertex: 0, indices: None, slots }
    }
}

/// A render pass with one pipeline.
pub struct Scene {
    pub compiled: Compiled,
    /// `GPUTextureFormat` of the color target: "r16uint", "r32uint" or "rgba32uint".
    pub format: &'static str,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub clear: [f64; 4],
    /// A depth32float attachment cleared to this value, tested with "always" and written.
    pub depth: Option<f64>,
    /// `GPUPrimitiveTopology`.
    pub topology: &'static str,
    pub draws: Vec<Draw>,
}

/// What a render pass produced: the color target (per sample for a 4-sample target, sample-major within a pixel) and
/// the depth target.
#[derive(Debug, PartialEq, Eq)]
pub struct Rendered {
    pub color: Vec<u8>,
    pub depth: Option<Vec<u8>>,
}

fn bytes_per_pixel(format: &str) -> u32 {
    match format {
        "r16uint" => 2,
        "r32uint" => 4,
        "rgba32uint" => 16,
        f => panic!("format {}", f),
    }
}

/// The uniform block of a draw or dispatch.
pub fn uniform_bytes(block: &UniformBlock, d: &Draw, region: Option<([u32; 3], [u32; 3])>) -> Vec<u8> {
    let mut bytes = vec![0u8; block.size as usize];
    for f in &block.fields {
        let slot = |stage: &MetalStage, index: &u64| {
            d.slots.iter().find(|s| s.stage == *stage && s.index == *index).unwrap_or_else(|| panic!("slot {}", index))
        };
        let words: Vec<u32> = match &f.value {
            UniformValue::BaseVertex => vec![d.base_vertex as u32],
            UniformValue::BaseInstance => vec![d.base_instance],
            UniformValue::ResidualOffset { stage, index } => vec![slot(stage, index).offset as u32],
            UniformValue::BufferLength { stage, index } => {
                let s = slot(stage, index);
                vec![(s.data.len() as u64 - s.offset) as u32]
            }
            UniformValue::StageInOrigin => region.expect("a stage-in region").0.to_vec(),
            UniformValue::StageInSize => region.expect("a stage-in region").1.to_vec(),
            other => panic!("uniform {:?}", other),
        };
        for (k, w) in words.iter().enumerate() {
            let o = f.offset as usize + 4 * k;
            bytes[o..o + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
    bytes
}

/// (group, binding, source) of every binding.
fn bindings(layout: &BindingLayout) -> Vec<(u32, u32, BindingSource)> {
    layout.groups.iter().flat_map(|g| g.entries.iter().map(move |e| (g.group, e.binding, e.source.clone()))).collect()
}

fn padded(data: &[u8]) -> Vec<u8> {
    let mut v = data.to_vec();
    while !v.len().is_multiple_of(4) || v.is_empty() {
        v.push(0);
    }
    v
}

// wgpu -----------------------------------------------------------------------------------------------------------

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

static RUN: Mutex<()> = Mutex::new(());

/// Dawn runs (one Node.js process each) are serialized too: concurrent dawn.node processes beside the wgpu device have
/// ended without a report under load.
static DAWN: Mutex<()> = Mutex::new(());

pub fn gpu() -> &'static Gpu {
    static G: OnceLock<Gpu> = OnceLock::new();
    G.get_or_init(|| {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no WebGPU adapter: the stage tests need one");
        let features = adapter.features() & wgpu::Features::DUAL_SOURCE_BLENDING;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("stage tests"),
            required_features: features,
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("request_device");
        Gpu { device, queue }
    })
}

fn wgpu_format(f: &str) -> wgpu::TextureFormat {
    match f {
        "r16uint" => wgpu::TextureFormat::R16Uint,
        "r32uint" => wgpu::TextureFormat::R32Uint,
        "rgba32uint" => wgpu::TextureFormat::Rgba32Uint,
        f => panic!("format {}", f),
    }
}

fn topology(t: &str) -> wgpu::PrimitiveTopology {
    match t {
        "point-list" => wgpu::PrimitiveTopology::PointList,
        "line-list" => wgpu::PrimitiveTopology::LineList,
        "line-strip" => wgpu::PrimitiveTopology::LineStrip,
        "triangle-list" => wgpu::PrimitiveTopology::TriangleList,
        "triangle-strip" => wgpu::PrimitiveTopology::TriangleStrip,
        t => panic!("topology {}", t),
    }
}

fn read_buffer(d: &wgpu::Device, b: &wgpu::Buffer) -> Vec<u8> {
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
    for (var s = 0u; s < 4u; s = s + 1u) {
        o[(g.y * w + g.x) * 4u + s] = textureLoad(t, g.xy, i32(s)).x;
    }
}
";

pub fn render_wgpu(s: &Scene) -> Rendered {
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
    let fs = s.compiled.wgsl.get(1).map(|st| shader(&st.source));
    let format = wgpu_format(s.format);
    let strip = s.topology.ends_with("strip");
    let indexed32 = s.draws.iter().find_map(|d| d.indices.as_ref().map(|i| i.uint32));
    let targets = [Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })];
    let pipeline = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &vs,
            entry_point: Some(&s.compiled.wgsl[0].entry_point),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: topology(s.topology),
            strip_index_format: if strip {
                indexed32.map(|u| if u { wgpu::IndexFormat::Uint32 } else { wgpu::IndexFormat::Uint16 })
            } else {
                None
            },
            ..Default::default()
        },
        depth_stencil: s.depth.map(|_| wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: s.samples, ..Default::default() },
        fragment: fs.as_ref().map(|fs| wgpu::FragmentState {
            module: fs,
            entry_point: Some(&s.compiled.wgsl[1].entry_point),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    });
    let size = wgpu::Extent3d { width: s.width, height: s.height, depth_or_array_layers: 1 };
    let target = d.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: s.samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth = s.depth.map(|_| {
        d.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    });
    // One bind group set per draw.
    let mut per_draw = Vec::new();
    for draw in &s.draws {
        let mut groups: BTreeMap<u32, Vec<(u32, wgpu::Buffer)>> = BTreeMap::new();
        for (grp, binding, source) in bindings(&s.compiled.layout) {
            let buffer = match &source {
                BindingSource::Uniforms(u) => d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &uniform_bytes(u, draw, None),
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
                BindingSource::MetalBuffer { stage, index } => {
                    let slot = draw
                        .slots
                        .iter()
                        .find(|x| x.stage == *stage && x.index == *index)
                        .unwrap_or_else(|| panic!("no buffer for {:?} {}", stage, index));
                    d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: &padded(&slot.data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                }
                other => panic!("binding source {:?}", other),
            };
            groups.entry(grp).or_default().push((binding, buffer));
        }
        let index = draw.indices.as_ref().map(|i| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &padded(&i.data),
                usage: wgpu::BufferUsages::INDEX,
            })
        });
        let bgs: Vec<(u32, wgpu::BindGroup)> = groups
            .iter()
            .map(|(grp, es)| {
                (
                    *grp,
                    d.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipeline.get_bind_group_layout(*grp),
                        entries: &es
                            .iter()
                            .map(|(b, buf)| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() })
                            .collect::<Vec<_>>(),
                    }),
                )
            })
            .collect();
        per_draw.push((bgs, index));
    }
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let view = target.create_view(&Default::default());
        let dview = depth.as_ref().map(|t| t.create_view(&Default::default()));
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: s.clear[0],
                        g: s.clear[1],
                        b: s.clear[2],
                        a: s.clear[3],
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: dview.as_ref().map(|v| wgpu::RenderPassDepthStencilAttachment {
                view: v,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(s.depth.unwrap() as f32),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        for (draw, (bgs, index)) in s.draws.iter().zip(&per_draw) {
            for (grp, bg) in bgs {
                pass.set_bind_group(*grp, bg, &[]);
            }
            let instances = draw.base_instance..draw.base_instance + draw.instance_count;
            match (&draw.indices, index) {
                (Some(i), Some(ib)) => {
                    let (fmt, size) =
                        if i.uint32 { (wgpu::IndexFormat::Uint32, 4) } else { (wgpu::IndexFormat::Uint16, 2) };
                    pass.set_index_buffer(ib.slice(..), fmt);
                    let first = (i.offset / size) as u32;
                    pass.draw_indexed(first..first + draw.count, draw.base_vertex, instances);
                }
                _ => pass.draw(draw.vertex_start..draw.vertex_start + draw.count, instances),
            }
        }
    }
    let bpp = bytes_per_pixel(s.format);
    let mut reads = Vec::new();
    if s.samples == 1 {
        let row = (s.width * bpp).div_ceil(256) * 256;
        let b = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * s.height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &b,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            size,
        );
        reads.push((b, row, s.width * bpp));
    } else {
        assert_eq!((s.samples, s.format), (4, "r16uint"), "multisample readback is for 4-sample r16uint targets");
        let n = (s.width * s.height * 4 * 4) as u64;
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
        let tv = target.create_view(&Default::default());
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
    if let Some(dt) = &depth {
        let row = (s.width * 4).div_ceil(256) * 256;
        let b = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * s.height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: dt,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &b,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            size,
        );
        reads.push((b, row, s.width * 4));
    }
    g.queue.submit([enc.finish()]);
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("wgpu validation: {}\n{}", e, s.compiled.wgsl.iter().map(|w| w.source.as_str()).collect::<String>());
    }
    let mut outs = Vec::new();
    for (b, row, len) in &reads {
        let raw = read_buffer(d, b);
        let rows = raw.len() as u32 / row;
        let mut packed = Vec::new();
        for y in 0..rows {
            packed.extend_from_slice(&raw[(y * row) as usize..(y * row + len) as usize]);
        }
        outs.push(packed);
    }
    let depth = if s.depth.is_some() { outs.pop() } else { None };
    Rendered { color: outs.pop().unwrap(), depth }
}

// Dawn -----------------------------------------------------------------------------------------------------------

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n"))
}

/// Run a scene on Dawn: the scene is written as JSON with its buffers in files, and `dawn_scene.cjs` replays it.
pub fn render_dawn(dawn: &Path, dir: &Path, s: &Scene) -> Rendered {
    let mut files = 0;
    let mut file = |data: &[u8]| {
        let p = dir.join(format!("b{}.bin", files));
        files += 1;
        std::fs::write(&p, data).unwrap();
        json_str(&p.display().to_string())
    };
    let mut draws = Vec::new();
    for draw in &s.draws {
        let mut binds = Vec::new();
        for (grp, binding, source) in bindings(&s.compiled.layout) {
            let (kind, data) = match &source {
                BindingSource::Uniforms(u) => ("uniform", uniform_bytes(u, draw, None)),
                BindingSource::MetalBuffer { stage, index } => {
                    let slot = draw.slots.iter().find(|x| x.stage == *stage && x.index == *index).unwrap();
                    ("storage", padded(&slot.data))
                }
                other => panic!("binding source {:?}", other),
            };
            binds.push(format!(
                "{{\"group\":{},\"binding\":{},\"kind\":\"{}\",\"file\":{}}}",
                grp,
                binding,
                kind,
                file(&data)
            ));
        }
        let index = match &draw.indices {
            Some(i) => format!(
                "{{\"file\":{},\"format\":\"{}\",\"first\":{}}}",
                file(&padded(&i.data)),
                if i.uint32 { "uint32" } else { "uint16" },
                i.offset / if i.uint32 { 4 } else { 2 }
            ),
            None => "null".into(),
        };
        draws.push(format!(
            "{{\"bindings\":[{}],\"index\":{},\"first\":{},\"count\":{},\"instances\":{},\"firstInstance\":{},\"baseVertex\":{}}}",
            binds.join(","),
            index,
            draw.vertex_start,
            draw.count,
            draw.instance_count,
            draw.base_instance,
            draw.base_vertex
        ));
    }
    let vs = dir.join("vs.wgsl");
    std::fs::write(&vs, &s.compiled.wgsl[0].source).unwrap();
    let fs = s.compiled.wgsl.get(1).map(|w| {
        let p = dir.join("fs.wgsl");
        std::fs::write(&p, &w.source).unwrap();
        json_str(&p.display().to_string())
    });
    let scene = format!(
        "{{\"vs\":{},\"fs\":{},\"entry\":{},\"format\":\"{}\",\"width\":{},\"height\":{},\"samples\":{},\"clear\":[{}],\"depth\":{},\"topology\":\"{}\",\"draws\":[{}],\"color\":{},\"depthOut\":{}}}",
        json_str(&vs.display().to_string()),
        fs.unwrap_or_else(|| "null".into()),
        json_str(&s.compiled.wgsl[0].entry_point),
        s.format,
        s.width,
        s.height,
        s.samples,
        s.clear.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","),
        s.depth.map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
        s.topology,
        draws.join(","),
        json_str(&dir.join("dawn.color").display().to_string()),
        json_str(&dir.join("dawn.depth").display().to_string()),
    );
    let sp = dir.join("scene.json");
    std::fs::write(&sp, scene).unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/stage/dawn_scene.cjs");
    {
        let _guard = DAWN.lock().unwrap_or_else(|e| e.into_inner());
        let mut attempt = 0;
        loop {
            let o = Command::new("node").arg(&script).arg(dawn).arg(&sp).output().unwrap();
            let complete = String::from_utf8_lossy(&o.stdout).contains("complete");
            match (o.status.success(), complete, o.status.code()) {
                (true, _, _) => break,
                // dawn.node ended by a signal after its outputs were written and checked: the run is complete.
                (false, true, None) => {
                    eprintln!("dawn_scene: dawn.node ended with {} after completing the scene", o.status);
                    break;
                }
                // Ended by a signal before completing (seen under load, never for a scene that fails on its own):
                // the scene runs once more, and a second failure fails the test.
                (false, false, None) if attempt == 0 => {
                    eprintln!(
                        "dawn_scene: dawn.node ended with {} before completing; running the scene again",
                        o.status
                    );
                    attempt += 1;
                }
                _ => panic!(
                    "dawn_scene ({}): {}{}",
                    o.status,
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                ),
            }
        }
    }
    Rendered {
        color: std::fs::read(dir.join("dawn.color")).unwrap(),
        depth: s.depth.map(|_| std::fs::read(dir.join("dawn.depth")).unwrap()),
    }
}

/// Render on wgpu and, when available, on Dawn; both must agree. Returns the wgpu result and whether Dawn ran.
pub fn render(name: &str, s: &Scene) -> (Rendered, bool) {
    let dir = case_dir(name);
    for (k, w) in s.compiled.wgsl.iter().enumerate() {
        std::fs::write(dir.join(format!("stage{}.wgsl", k)), &w.source).unwrap();
    }
    let w = render_wgpu(s);
    let mut dawn = false;
    if let Some(dn) = dawn_node() {
        let d = render_dawn(&dn, &dir, s);
        assert!(d == w, "{}: Dawn and wgpu differ (WGSL in {})", name, dir.display());
        dawn = true;
    }
    (w, dawn)
}

// Compute ----------------------------------------------------------------------------------------------------------

/// Run a compiled kernel with `groups` workgroups on wgpu and (when available) Dawn; returns each slot's bytes.
pub fn dispatch(
    name: &str,
    c: &Compiled,
    groups: [u32; 3],
    slots: &[Slot],
    region: Option<([u32; 3], [u32; 3])>,
) -> (Vec<Vec<u8>>, bool) {
    use wgpu::util::DeviceExt;
    let dir = case_dir(name);
    std::fs::write(dir.join("k.wgsl"), &c.wgsl[0].source).unwrap();
    let draw = Draw::new(0, 0, slots.to_vec());
    let w = {
        let _guard = RUN.lock().unwrap_or_else(|e| e.into_inner());
        let g = gpu();
        let d = &g.device;
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(c.wgsl[0].source.as_str().into()),
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some(&c.wgsl[0].entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        let bufs: Vec<wgpu::Buffer> = slots
            .iter()
            .map(|s| {
                d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &padded(&s.data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
            })
            .collect();
        let mut groups_map: BTreeMap<u32, Vec<(u32, wgpu::Buffer)>> = BTreeMap::new();
        for (grp, binding, source) in bindings(&c.layout) {
            let b = match &source {
                BindingSource::Uniforms(u) => d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &uniform_bytes(u, &draw, region),
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
                BindingSource::MetalBuffer { index, .. } => {
                    bufs[slots.iter().position(|s| s.index == *index).unwrap()].clone()
                }
                other => panic!("binding source {:?}", other),
            };
            groups_map.entry(grp).or_default().push((binding, b));
        }
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (grp, es) in &groups_map {
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
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        let reads: Vec<wgpu::Buffer> = bufs
            .iter()
            .map(|b| {
                let r = d.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: b.size(),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                enc.copy_buffer_to_buffer(b, 0, &r, 0, b.size());
                r
            })
            .collect();
        g.queue.submit([enc.finish()]);
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("wgpu validation: {}\n{}", e, c.wgsl[0].source);
        }
        reads.iter().zip(slots).map(|(r, s)| read_buffer(d, r)[..s.data.len()].to_vec()).collect::<Vec<_>>()
    };
    let mut dawn = false;
    if let Some(dn) = dawn_node() {
        let mut args: Vec<String> = Vec::new();
        for (grp, binding, source) in bindings(&c.layout) {
            match &source {
                BindingSource::Uniforms(u) => {
                    let p = dir.join("uniforms.bin");
                    std::fs::write(&p, uniform_bytes(u, &draw, region)).unwrap();
                    args.push(format!("{}:{}:uniform:{}", grp, binding, p.display()));
                }
                BindingSource::MetalBuffer { index, .. } => {
                    let s = slots.iter().find(|s| s.index == *index).unwrap();
                    let p = dir.join(format!("dawn{}.bin", index));
                    std::fs::write(&p, &s.data).unwrap();
                    let written = c.layout.groups.iter().flat_map(|g| &g.entries).any(|e| {
                        e.binding == binding
                            && matches!(e.resource, BindingResource::Buffer { ty: BufferBindingType::Storage, .. })
                    });
                    args.push(format!(
                        "{}:{}:{}:{}",
                        grp,
                        binding,
                        if written { "storage" } else { "read-only-storage" },
                        p.display()
                    ));
                }
                other => panic!("binding source {:?}", other),
            }
        }
        let shader = dir.join("k.wgsl");
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wgsl/dawn_run.cjs");
        let _guard = DAWN.lock().unwrap_or_else(|e| e.into_inner());
        let o = Command::new("node")
            .arg(script)
            .arg(dn)
            .arg(&shader)
            .arg(&c.wgsl[0].entry_point)
            .arg(format!("{},{},{}", groups[0], groups[1], groups[2]))
            .args(&args)
            .output()
            .unwrap();
        assert!(o.status.success(), "dawn_run ({}): {}", o.status, String::from_utf8_lossy(&o.stderr));
        for (k, s) in slots.iter().enumerate() {
            let used = bindings(&c.layout)
                .iter()
                .any(|(_, _, src)| matches!(src, BindingSource::MetalBuffer { index, .. } if *index == s.index));
            if used {
                let got = std::fs::read(dir.join(format!("dawn{}.bin.out", s.index))).unwrap();
                assert_eq!(got, w[k], "{}: Dawn and wgpu differ for buffer {}", name, s.index);
            }
        }
        dawn = true;
    }
    (w, dawn)
}

// V1 goldens -------------------------------------------------------------------------------------------------------

/// The V1 reference harness directory (WebMetal `tests/reference`): `AW_V1_REFERENCE`, else its place next to this
/// crate in the UniXNU tree.
pub fn reference() -> Option<PathBuf> {
    let p = match std::env::var_os("AW_V1_REFERENCE") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../drivers/WebMetal/tests/reference"),
    };
    p.join("shaders").is_dir().then_some(p)
}

/// The device key of the goldens the stage tests compare with.
pub const GOLDEN_DEVICE: &str = "apple-m4-macos-27.0.1-26a434";

/// A JSON value (the subset the golden files use).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(k),
            _ => None,
        }
    }
    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a,
            _ => &[],
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Json {
        self.ws();
        match self.b[self.i] {
            b'{' => {
                self.i += 1;
                let mut m = BTreeMap::new();
                loop {
                    self.ws();
                    if self.b[self.i] == b'}' {
                        self.i += 1;
                        return Json::Obj(m);
                    }
                    let Json::Str(k) = self.value() else { panic!("object key") };
                    self.ws();
                    assert_eq!(self.b[self.i], b':');
                    self.i += 1;
                    let v = self.value();
                    m.insert(k, v);
                    self.ws();
                    if self.b[self.i] == b',' {
                        self.i += 1;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                loop {
                    self.ws();
                    if self.b[self.i] == b']' {
                        self.i += 1;
                        return Json::Arr(a);
                    }
                    a.push(self.value());
                    self.ws();
                    if self.b[self.i] == b',' {
                        self.i += 1;
                    }
                }
            }
            b'"' => {
                self.i += 1;
                let mut s = Vec::new();
                while self.b[self.i] != b'"' {
                    if self.b[self.i] == b'\\' {
                        self.i += 1;
                        match self.b[self.i] {
                            b'n' => s.push(b'\n'),
                            b't' => s.push(b'\t'),
                            b'u' => {
                                let h = std::str::from_utf8(&self.b[self.i + 1..self.i + 5]).unwrap();
                                let c = char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap_or('?');
                                s.extend_from_slice(c.to_string().as_bytes());
                                self.i += 4;
                            }
                            c => s.push(c),
                        }
                    } else {
                        s.push(self.b[self.i]);
                    }
                    self.i += 1;
                }
                self.i += 1;
                Json::Str(String::from_utf8_lossy(&s).into_owned())
            }
            b't' => {
                self.i += 4;
                Json::Bool(true)
            }
            b'f' => {
                self.i += 5;
                Json::Bool(false)
            }
            b'n' => {
                self.i += 4;
                Json::Null
            }
            _ => {
                let start = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    self.i += 1;
                }
                Json::Num(std::str::from_utf8(&self.b[start..self.i]).unwrap().parse().unwrap())
            }
        }
    }
}

pub fn parse_json(text: &str) -> Json {
    Parser { b: text.as_bytes(), i: 0 }.value()
}

/// A golden case file of the release variant.
pub struct Golden {
    pub case: String,
    records: Vec<Json>,
}

impl Golden {
    pub fn load(reference: &Path, case: &str) -> Golden {
        let p = reference.join("goldens").join(GOLDEN_DEVICE).join("release").join(format!("{}.json", case));
        let j = parse_json(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {}", p.display(), e)));
        assert_eq!(j.get("status").and_then(|s| s.str()), Some("ok"), "{} did not run to its end", case);
        Golden { case: case.into(), records: j.get("records").unwrap().arr().to_vec() }
    }

    fn record(&self, name: &str) -> &Json {
        self.records
            .iter()
            .find(|r| r.get("name").and_then(|n| n.str()) == Some(name))
            .unwrap_or_else(|| panic!("{}: no record {}", self.case, name))
    }

    /// The bytes of a data record.
    pub fn data(&self, name: &str) -> Vec<u8> {
        let h = self.record(name).get("hex").and_then(|h| h.str()).unwrap();
        (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    /// The comparison rule of a record.
    pub fn rule(&self, name: &str) -> String {
        self.record(name).get("rule").and_then(|r| r.get("compare")).and_then(|c| c.str()).unwrap().to_string()
    }
}

/// The AIR of one of the V1 shader files, compiled as the harness's build does (`-std=ios-metal2.4`).
pub fn v1_air(reference: &Path, file: &str) -> Vec<u8> {
    static CACHE: Mutex<BTreeMap<String, Vec<u8>>> = Mutex::new(BTreeMap::new());
    let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = c.get(file) {
        return b.clone();
    }
    let dir = tmp_dir().join("v1-air");
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join(format!("{}.air", file));
    let o = Command::new("xcrun")
        .args(["-sdk", "iphoneos", "metal", "-std=ios-metal2.4", "-target", "air64-apple-ios17.7", "-c"])
        .arg(reference.join("shaders").join(format!("{}.metal", file)))
        .arg("-o")
        .arg(&out)
        .output()
        .expect("xcrun metal");
    assert!(o.status.success(), "metal: {}", String::from_utf8_lossy(&o.stderr));
    let b = std::fs::read(&out).unwrap();
    c.insert(file.into(), b.clone());
    b
}

/// The rows `y0..y0+h`, columns `x0..x0+w` of a tightly packed image with `bpp` bytes per pixel.
pub fn region(image: &[u8], width: u32, bpp: u32, x0: u32, y0: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for y in y0..y0 + h {
        let s = ((y * width + x0) * bpp) as usize;
        out.extend_from_slice(&image[s..s + (w * bpp) as usize]);
    }
    out
}

pub fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// The first differing 32-bit word of two byte strings.
pub fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let (wa, wb) = (words(a), words(b));
    (0..wa.len().max(wb.len())).find(|&i| wa.get(i) != wb.get(i))
}

/// The metal runner (`metal_fetch.swift`), built once.
pub fn metal_fetch() -> PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let exe = tmp_dir().join("metal_fetch");
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/stage/metal_fetch.swift");
        let o = Command::new("swiftc").args(["-O", "-o"]).arg(&exe).arg(&src).output().expect("swiftc");
        assert!(o.status.success(), "swiftc: {}", String::from_utf8_lossy(&o.stderr));
        exe
    })
    .clone()
}
