//! The tessellator's compute kernels (`tess::kernel`) on wgpu against the CPU tessellator, record for record, over
//! every draw of the V1 cases.

use super::draw::{self, Draw};
use super::stage::{gpu, reference};
use airwgsl::tess::{self, kernel};
use wgpu::util::DeviceExt;

/// What the kernels produced: the point records, the indices, the draw buffer.
struct Out {
    vertices: Vec<u32>,
    indices: Vec<u32>,
    draw: Vec<u32>,
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn read(d: &wgpu::Device, q: &wgpu::Queue, b: &wgpu::Buffer, size: u64) -> Vec<u32> {
    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = d.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(b, 0, &staging, 0, size);
    q.submit([e.finish()]);
    let s = staging.slice(..);
    s.map_async(wgpu::MapMode::Read, |_| {});
    d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    words(&s.get_mapped_range())
}

/// The kernels' output buffers on the GPU.
pub struct Kernels {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub draw: wgpu::Buffer,
}

/// The kernel parameters of a draw. The factor buffer is bound from offset 0 and the draw's offset goes into the
/// parameters, as the executor does for a Metal offset below the storage alignment.
pub fn params(d: &Draw) -> kernel::Params {
    let mut p = kernel::Params::new(&d.settings(), d.patch_type()).with_scale(d.scale);
    p.factor_offset = d.offset;
    p.instance_stride = d.instance_stride;
    p.step = d.step;
    p.patch_start = d.patch_start;
    p.patch_count = d.patch_count;
    p.instance_count = d.instance_count;
    p.base_instance = d.base_instance;
    if !d.patch_indices.is_empty() {
        p.patch_index_offset = Some(0);
    }
    p
}

/// Run the three dispatches for a draw.
pub fn dispatch(d: &Draw) -> Kernels {
    let g = gpu();
    let dev = &g.device;
    let p = params(d);
    let (max_points, max_indices) = kernel::bounds(&d.settings(), d.patch_type());
    let slots = p.slots();
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let mut factor_bytes: Vec<u8> = d.factors.iter().flat_map(|h| h.to_le_bytes()).collect();
    while !factor_bytes.len().is_multiple_of(4) || factor_bytes.is_empty() {
        factor_bytes.push(0);
    }
    let init = |label: &str, contents: &[u8], usage| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
    };
    let params = init("params", &p.bytes(), wgpu::BufferUsages::UNIFORM);
    let factors = init("factors", &factor_bytes, storage);
    let pi: Vec<u8> = if d.patch_indices.is_empty() {
        vec![0; 4]
    } else {
        d.patch_indices.iter().flat_map(|w| w.to_le_bytes()).collect()
    };
    let patch_indices = init("patch indices", &pi, storage);
    let zeroed = |label: &str, words: u64, extra| {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 4 * words.max(1),
            usage: storage | extra,
            mapped_at_creation: false,
        })
    };
    let slot_buffer = zeroed("slots", (slots * kernel::SLOT_WORDS) as u64, wgpu::BufferUsages::empty());
    let vertex_words = slots as u64 * max_points as u64 * kernel::VERTEX_WORDS as u64;
    let vertices = zeroed("vertices", vertex_words, wgpu::BufferUsages::empty());
    let index_words = slots as u64 * max_indices as u64;
    let indices = zeroed("indices", index_words, wgpu::BufferUsages::INDEX);
    let draw_buffer = zeroed("draw", kernel::DRAW_WORDS as u64, wgpu::BufferUsages::INDIRECT);
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(kernel::WGSL.into()),
    });
    let pipeline = |entry: &str| {
        dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipelines = [pipeline(kernel::ENTRY_COUNT), pipeline(kernel::ENTRY_SCAN), pipeline(kernel::ENTRY_GENERATE)];
    let resources = [&params, &factors, &patch_indices, &slot_buffer, &vertices, &indices, &draw_buffer];
    let mut e = dev.create_command_encoder(&Default::default());
    for (k, pl) in pipelines.iter().enumerate() {
        let layout = pl.get_bind_group_layout(0);
        // Each entry point's derived layout holds only the bindings it uses.
        let used: &[u32] = match k {
            0 => &[0, 1, 2, 3, 6],
            1 => &[0, 3, 6],
            _ => &[0, 1, 2, 3, 4, 5],
        };
        let entries: Vec<wgpu::BindGroupEntry> = used
            .iter()
            .map(|&b| wgpu::BindGroupEntry { binding: b, resource: resources[b as usize].as_entire_binding() })
            .collect();
        let group =
            dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        let mut pass = e.begin_compute_pass(&Default::default());
        pass.set_pipeline(pl);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(if k == 1 { 1 } else { slots.div_ceil(kernel::WORKGROUP) }, 1, 1);
    }
    g.queue.submit([e.finish()]);
    Kernels { vertices, indices, draw: draw_buffer }
}

fn run(d: &Draw) -> Out {
    let g = gpu();
    let k = dispatch(d);
    let draw = read(&g.device, &g.queue, &k.draw, 4 * kernel::DRAW_WORDS as u64);
    let points = draw[kernel::DRAW_POINTS as usize] as u64;
    let index_count = draw[0] as u64;
    Out {
        vertices: if points == 0 { Vec::new() } else { read(&g.device, &g.queue, &k.vertices, 16 * points) },
        indices: if index_count == 0 { Vec::new() } else { read(&g.device, &g.queue, &k.indices, 4 * index_count) },
        draw,
    }
}

/// The CPU tessellator's records and indices for a draw, as the kernels lay them out.
fn expected(d: &Draw) -> (Vec<u32>, Vec<u32>, bool) {
    let settings = d.settings();
    let (mut v, mut ix, mut batched) = (Vec::new(), Vec::new(), false);
    for i in 0..d.instance_count {
        let instance_id = d.base_instance + i;
        for dp in d.patch_start..d.patch_start + d.patch_count {
            let halves = d.factors_of(dp, instance_id);
            let scale = if d.scale_enabled { d.scale } else { 1.0 };
            batched |= tess::counts(&settings, d.patch_type(), &halves, scale).batched;
            let patch = tess::tessellate(&settings, d.patch_type(), &halves, scale);
            let base = (v.len() / 4) as u32;
            for p in &patch.points {
                v.extend_from_slice(&[p[0], p[1], d.patch_id(dp), instance_id]);
            }
            ix.extend(patch.indices.iter().map(|&k| base + k));
        }
    }
    (v, ix, batched)
}

#[test]
fn kernels_equal_the_cpu_tessellator_on_wgpu() {
    let Some(reference) = reference() else {
        eprintln!("not run: WebMetal's tests/reference is not next to this crate and AW_V1_REFERENCE is unset");
        return;
    };
    let mut n = 0;
    for case in super::CASES {
        for (name, d, _) in draw::load(&reference, case) {
            let got = run(&d);
            let (v, ix, batched) = expected(&d);
            assert_eq!(got.draw[1..5], [1, 0, 0, 0], "{case} {name}: indirect arguments");
            assert_eq!(got.draw[kernel::DRAW_BATCHED as usize] != 0, batched, "{case} {name}: batched");
            assert_eq!(got.vertices.len(), v.len(), "{case} {name}: points");
            if let Some(k) = got.vertices.iter().zip(&v).position(|(a, b)| a != b) {
                panic!("{case} {name}: point record word {k}: {} on the GPU, {} on the CPU", got.vertices[k], v[k]);
            }
            assert_eq!(got.indices.len(), ix.len(), "{case} {name}: indices");
            if let Some(k) = got.indices.iter().zip(&ix).position(|(a, b)| a != b) {
                panic!("{case} {name}: index {k}: {} on the GPU, {} on the CPU", got.indices[k], ix[k]);
            }
            n += 1;
        }
    }
    eprintln!("{n} draws: the kernels' point records and indices equal the CPU tessellator's");
}

/// No factors give a patch more points or indices than [`kernel::bounds`], which sizes the executor's buffers.
#[test]
fn bounds_hold_over_a_factor_grid() {
    use airwgsl::tess::{PartitionMode, PatchType, Settings, factors::f32_to_half};
    let values =
        [0.5f32, 1.0, 1.01, 1.5, 2.0, 2.5, 3.0, 3.99, 5.0, 7.5, 8.0, 9.0, 11.0, 15.0, 15.9, 16.0, 31.0, 63.9, 64.0];
    for max in [8u32, 16, 32, 64] {
        for mode in
            [PartitionMode::Pow2, PartitionMode::Integer, PartitionMode::FractionalOdd, PartitionMode::FractionalEven]
        {
            if mode == PartitionMode::Pow2 && !max.is_power_of_two() {
                continue;
            }
            let s = Settings { partition: mode, max_factor: max, scale_enabled: false, clockwise: true };
            for patch in [PatchType::Triangle, PatchType::Quad] {
                let (bp, bi) = kernel::bounds(&s, patch);
                for (k, &a) in values.iter().enumerate() {
                    for &b in values.iter().skip(k % 3).step_by(3) {
                        let h = [
                            f32_to_half(a),
                            f32_to_half(b),
                            f32_to_half(a),
                            f32_to_half(b),
                            f32_to_half(b),
                            f32_to_half(a),
                        ];
                        let c = tess::counts(&s, patch, &h, 1.0);
                        assert!(
                            c.points <= bp && c.indices <= bi,
                            "{mode:?} {max} {patch:?} {a} {b}: {c:?} > ({bp}, {bi})"
                        );
                    }
                }
            }
        }
    }
}
