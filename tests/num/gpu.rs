//! Running WGSL compute kernels on a real WebGPU implementation (wgpu with its Metal backend) and reading the results
//! back as u32 words.

use std::sync::{Mutex, OnceLock};

/// Error scopes are a stack per device; kernel creation in concurrent tests is serialized so that each pops its own.
static CREATE: Mutex<()> = Mutex::new(());

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Adapter name, backend and driver, for test reports.
    pub adapter: String,
    pub has_f16: bool,
}

/// The largest number of invocations, and output words, one run handles.
pub const MAX_ITEMS: u32 = 1 << 24;

impl Gpu {
    /// The shared device. A test that needs the GPU fails when no adapter exists: the numerics proofs are not skipped.
    pub fn get() -> &'static Gpu {
        static GPU: OnceLock<Gpu> = OnceLock::new();
        GPU.get_or_init(|| {
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.backends = wgpu::Backends::METAL;
            let instance = wgpu::Instance::new(desc);
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            }))
            .expect("no WebGPU adapter: the numerics GPU tests need one");
            let info = adapter.get_info();
            let has_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
            let mut limits = wgpu::Limits::default();
            limits.max_storage_buffer_binding_size = (MAX_ITEMS as u64) * 4 * 4;
            limits.max_buffer_size = (MAX_ITEMS as u64) * 4 * 4;
            let features = if has_f16 { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() };
            let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("numerics tests"),
                required_features: features,
                required_limits: limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
            .expect("request_device");
            Gpu { device, queue, adapter: format!("{} ({:?}, {})", info.name, info.backend, info.driver_info), has_f16 }
        })
    }
}

/// The kernel frame every test kernel uses. `BODY` sees `i` (the item index), the inputs `nm_t_in0..2`, the
/// parameters `nm_t_p` (x is the item count) and writes `nm_t_out`.
pub fn kernel_source(enable_f16: bool, prelude: &str, helpers: &str, body: &str) -> String {
    format!(
        "{}{}\n{}\n@group(0) @binding(0) var<storage, read> nm_t_in0: array<u32>;\n\
         @group(0) @binding(1) var<storage, read> nm_t_in1: array<u32>;\n\
         @group(0) @binding(2) var<storage, read> nm_t_in2: array<u32>;\n\
         @group(0) @binding(3) var<uniform> nm_t_p: vec4<u32>;\n\
         @group(0) @binding(4) var<storage, read_write> nm_t_out: array<u32>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) w: vec3<u32>) {{\n\
         let i = g.x + g.y * w.x * 256u;\n\
         if (i >= nm_t_p.x) {{ return; }}\n\
         {}\n}}\n",
        if enable_f16 { "enable f16;\n" } else { "" },
        prelude,
        helpers,
        body
    )
}

pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Output words per item.
    stride: u32,
    out: wgpu::Buffer,
    readback: wgpu::Buffer,
    params: wgpu::Buffer,
    dummy: wgpu::Buffer,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Kernel {
    /// Compile `source`. A shader the implementation rejects fails the test with the implementation's message.
    pub fn new(source: &str, stride: u32) -> Kernel {
        let gpu = Gpu::get();
        let device = &gpu.device;
        let _serial = CREATE.lock().unwrap_or_else(|e| e.into_inner());
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("numerics kernel"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(4, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("kernel rejected: {}", e);
        }
        let bytes = MAX_ITEMS as u64 * stride as u64 * 4;
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Kernel { pipeline, layout, stride, out, readback, params, dummy }
    }

    /// Run `n` items (at most [`MAX_ITEMS`]) with up to three input arrays and parameters y, z, w; return
    /// `n * stride` output words.
    pub fn run(&self, inputs: &[&[u32]], params: [u32; 3], n: u32) -> Vec<u32> {
        assert!(n <= MAX_ITEMS && inputs.len() <= 3);
        let gpu = Gpu::get();
        let device = &gpu.device;
        let ins: Vec<wgpu::Buffer> = inputs
            .iter()
            .map(|data| {
                let b = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (data.len().max(4) * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue.write_buffer(&b, 0, bytes_of(data));
                b
            })
            .collect();
        gpu.queue.write_buffer(&self.params, 0, bytes_of(&[n, params[0], params[1], params[2]]));
        let input = |k: usize| ins.get(k).unwrap_or(&self.dummy).as_entire_binding();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input(0) },
                wgpu::BindGroupEntry { binding: 1, resource: input(1) },
                wgpu::BindGroupEntry { binding: 2, resource: input(2) },
                wgpu::BindGroupEntry { binding: 3, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.out.as_entire_binding() },
            ],
        });
        let groups = n.div_ceil(256).max(1);
        let (gx, gy) = if groups <= 65535 { (groups, 1) } else { (4096, groups.div_ceil(4096)) };
        let bytes = n as u64 * self.stride as u64 * 4;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&self.out, 0, &self.readback, 0, bytes.max(4));
        gpu.queue.submit([enc.finish()]);
        let slice = self.readback.slice(0..bytes.max(4));
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let words = {
            let view = slice.get_mapped_range();
            let mut v = vec![0u32; (bytes / 4) as usize];
            for (w, c) in v.iter_mut().zip(view[..bytes as usize].chunks_exact(4)) {
                *w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            v
        };
        self.readback.unmap();
        words
    }
}

pub fn bytes_of(words: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and any byte pattern of it is valid; the slice covers exactly the words.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) }
}

/// Split the index range 0..n across the available CPU threads.
pub fn par_range(n: usize, f: impl Fn(std::ops::Range<usize>) + Sync) {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1);
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let mut start = 0;
        while start < n {
            let end = (start + per).min(n);
            let f = &f;
            s.spawn(move || f(start..end));
            start = end;
        }
    });
}
