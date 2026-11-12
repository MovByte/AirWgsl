//! Running one compute function three ways: real Metal on this Mac (the reference), the compiled WGSL on wgpu (wgpu
//! 29, whose wgpu-core is the one wgpu-native v29 wraps), and the compiled WGSL on Dawn through its Node.js binding
//! when `AW_DAWN_NODE` names `dawn.node`. Buffers are compared byte for byte.
//!
//! The MSL source is compiled twice: by `xcrun metal` for `air64-apple-ios17.7` (the target's AIR, which this crate
//! compiles to WGSL) and by the Mac's Metal at run time (the reference). Both are Apple's front end for the same
//! source; the tests use only operations whose result MSL defines exactly.

use airwgsl::api::*;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

pub fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wgsl-tests");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A unique scratch directory for one case.
pub fn case_dir(name: &str) -> PathBuf {
    let d = tmp_dir().join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The target's AIR for an MSL source.
pub fn air(dir: &Path, source: &str) -> Vec<u8> {
    let src = dir.join("k.metal");
    std::fs::write(&src, source).unwrap();
    let out = dir.join("k.air");
    let o = Command::new("xcrun")
        .args(["-sdk", "iphoneos", "metal", "-std=metal3.1", "-target", "air64-apple-ios17.7", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .output()
        .expect("xcrun metal");
    assert!(o.status.success(), "metal: {}", String::from_utf8_lossy(&o.stderr));
    std::fs::read(&out).unwrap()
}

/// A compute pipeline for `function` of the module.
#[allow(clippy::result_large_err)]
pub fn compile_kernel(bitcode: &[u8], function: &str, threads: [u64; 3], tg: &[u64]) -> Result<Compiled, Refusal> {
    let m = AirModule {
        hash: [0; 32],
        bitcode: Arc::from(bitcode),
        entry: Some(EntryPoint { function: function.into(), function_type: 3, constants: Vec::new() }),
    };
    let desc = PipelineDesc {
        kind: PipelineKind::Compute(ComputeState {
            threads_per_threadgroup: threads,
            non_uniform_grid: false,
            threadgroup_memory_lengths: tg.to_vec(),
            stage_input: None,
        }),
        bindings: Vec::new(),
    };
    let mut host = HostCaps::default();
    host.features.insert("shader-f16".into());
    compile(&[m], &desc, &host)
}

/// The executor's uniform block for a dispatch whose buffers are bound from offset 0.
fn uniforms(layout: &BindingLayout, buffers: &[Vec<u8>]) -> Option<(u32, u32, Vec<u8>)> {
    for g in &layout.groups {
        for e in &g.entries {
            if let BindingSource::Uniforms(u) = &e.source {
                let mut bytes = vec![0u8; u.size as usize];
                for f in &u.fields {
                    let v: u32 = match &f.value {
                        UniformValue::ResidualOffset { .. } | UniformValue::BaseVertex | UniformValue::BaseInstance => {
                            0
                        }
                        UniformValue::BufferLength { index, .. } => buffers[*index as usize].len() as u32,
                        other => panic!("uniform {:?}", other),
                    };
                    bytes[f.offset as usize..f.offset as usize + 4].copy_from_slice(&v.to_le_bytes());
                }
                return Some((g.group, e.binding, bytes));
            }
        }
    }
    None
}

/// (group, binding, Metal buffer index, writable) of each buffer binding.
fn buffer_bindings(layout: &BindingLayout) -> Vec<(u32, u32, u64, bool)> {
    let mut out = Vec::new();
    for g in &layout.groups {
        for e in &g.entries {
            if let (BindingSource::MetalBuffer { index, .. }, BindingResource::Buffer { ty, .. }) =
                (&e.source, &e.resource)
            {
                out.push((g.group, e.binding, *index, *ty == BufferBindingType::Storage));
            }
        }
    }
    out
}

// Real Metal ------------------------------------------------------------------------------------------------------

fn metal_runner() -> PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let exe = tmp_dir().join("metal_run");
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wgsl/metal_run.swift");
        let o = Command::new("swiftc").args(["-O", "-o"]).arg(&exe).arg(&src).output().expect("swiftc");
        assert!(o.status.success(), "swiftc: {}", String::from_utf8_lossy(&o.stderr));
        exe
    })
    .clone()
}

/// Run on this Mac's Metal: returns each buffer's contents afterwards.
pub fn run_metal(
    dir: &Path,
    source: &str,
    function: &str,
    threads: [u64; 3],
    groups: [u32; 3],
    tg: &[u64],
    buffers: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let src = dir.join("ref.metal");
    std::fs::write(&src, source).unwrap();
    let mut files = Vec::new();
    for (i, b) in buffers.iter().enumerate() {
        let p = dir.join(format!("metal{}.bin", i));
        std::fs::write(&p, b).unwrap();
        files.push(p);
    }
    let tgs =
        if tg.is_empty() { "-".to_string() } else { tg.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",") };
    let o = Command::new(metal_runner())
        .arg(&src)
        .arg(function)
        .arg(format!("{},{},{}", threads[0], threads[1], threads[2]))
        .arg(format!("{},{},{}", groups[0], groups[1], groups[2]))
        .arg(tgs)
        .args(&files)
        .output()
        .unwrap();
    assert!(o.status.success(), "metal_run: {}", String::from_utf8_lossy(&o.stderr));
    files.iter().map(|f| std::fs::read(f.with_extension("bin.out")).unwrap()).collect()
}

// wgpu ---------------------------------------------------------------------------------------------------------

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Error scopes are per device; runs are serialized so that each pops its own.
static RUN: Mutex<()> = Mutex::new(());

pub fn gpu() -> &'static Gpu {
    static G: OnceLock<Gpu> = OnceLock::new();
    G.get_or_init(|| {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no WebGPU adapter: the WGSL differential tests need one");
        let features = adapter.features() & wgpu::Features::SHADER_F16;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wgsl tests"),
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

/// Run the compiled kernel on wgpu with `groups` workgroups.
pub fn run_wgpu(c: &Compiled, groups: [u32; 3], buffers: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use wgpu::util::DeviceExt;
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
    let mut gpu_bufs: Vec<Option<wgpu::Buffer>> = Vec::new();
    for b in buffers {
        let mut data = b.clone();
        while data.len() % 4 != 0 || data.is_empty() {
            data.push(0);
        }
        gpu_bufs.push(Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })));
    }
    let mut entries: std::collections::BTreeMap<u32, Vec<(u32, wgpu::Buffer)>> = Default::default();
    if let Some((grp, bind, bytes)) = uniforms(&c.layout, buffers) {
        let ub = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        entries.entry(grp).or_default().push((bind, ub));
    }
    for (grp, bind, index, _) in buffer_bindings(&c.layout) {
        entries.entry(grp).or_default().push((bind, gpu_bufs[index as usize].clone().unwrap()));
    }
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        for (grp, es) in &entries {
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
    let mut reads = Vec::new();
    for b in gpu_bufs.iter().flatten() {
        let r = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: b.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(b, 0, &r, 0, b.size());
        reads.push(r);
    }
    g.queue.submit([enc.finish()]);
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("wgpu validation: {}\n{}", e, c.wgsl[0].source);
    }
    let mut out = Vec::new();
    for (k, r) in reads.iter().enumerate() {
        let s = r.slice(..);
        s.map_async(wgpu::MapMode::Read, |_| {});
        d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        out.push(s.get_mapped_range()[..buffers[k].len()].to_vec());
    }
    out
}

// Dawn -----------------------------------------------------------------------------------------------------------

/// `dawn.node`, when the environment names it.
pub fn dawn_node() -> Option<PathBuf> {
    std::env::var_os("AW_DAWN_NODE").map(PathBuf::from)
}

pub fn run_dawn(dawn: &Path, dir: &Path, c: &Compiled, groups: [u32; 3], buffers: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let shader = dir.join("k.wgsl");
    std::fs::write(&shader, &c.wgsl[0].source).unwrap();
    let mut args: Vec<String> = Vec::new();
    if let Some((grp, bind, bytes)) = uniforms(&c.layout, buffers) {
        let p = dir.join("uniforms.bin");
        std::fs::write(&p, bytes).unwrap();
        args.push(format!("{}:{}:uniform:{}", grp, bind, p.display()));
    }
    let mut files = vec![None; buffers.len()];
    for (grp, bind, index, writable) in buffer_bindings(&c.layout) {
        let p = dir.join(format!("dawn{}.bin", index));
        std::fs::write(&p, &buffers[index as usize]).unwrap();
        args.push(format!(
            "{}:{}:{}:{}",
            grp,
            bind,
            if writable { "storage" } else { "read-only-storage" },
            p.display()
        ));
        files[index as usize] = Some(p);
    }
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wgsl/dawn_run.cjs");
    let o = Command::new("node")
        .arg(script)
        .arg(dawn)
        .arg(&shader)
        .arg(&c.wgsl[0].entry_point)
        .arg(format!("{},{},{}", groups[0], groups[1], groups[2]))
        .args(&args)
        .output()
        .unwrap();
    assert!(o.status.success(), "dawn_run: {}\n{}", String::from_utf8_lossy(&o.stderr), c.wgsl[0].source);
    // Buffers the shader does not use keep their contents.
    files
        .iter()
        .enumerate()
        .map(|(k, f)| match f {
            Some(p) => std::fs::read(p.with_extension("bin.out")).unwrap(),
            None => buffers[k].clone(),
        })
        .collect()
}

/// Deterministic test bytes (SplitMix64).
pub fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)) as u8
        })
        .collect()
}

/// The first differing byte of two buffers.
pub fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i))
}

/// Which backends ran a case.
pub struct Ran {
    pub dawn: bool,
}

/// Compile and run a kernel on Metal, wgpu and (when available) Dawn, and require byte-identical buffers.
pub fn differential(
    name: &str,
    source: &str,
    function: &str,
    threads: [u64; 3],
    groups: [u32; 3],
    tg: &[u64],
    buffers: &[Vec<u8>],
) -> Ran {
    let dir = case_dir(name);
    let bc = air(&dir, source);
    let c = compile_kernel(&bc, function, threads, tg).unwrap_or_else(|r| panic!("{}: {}", name, r));
    std::fs::write(dir.join("k.wgsl"), &c.wgsl[0].source).unwrap();
    let reference = run_metal(&dir, source, function, threads, groups, tg, buffers);
    let w = run_wgpu(&c, groups, buffers);
    for (k, (a, b)) in reference.iter().zip(&w).enumerate() {
        if let Some(i) = first_difference(a, b) {
            panic!(
                "{}: buffer {} differs at byte {} (Metal {:02x?}, wgpu {:02x?}); WGSL in {}",
                name,
                k,
                i,
                &a[i..(i + 16).min(a.len())],
                &b[i..(i + 16).min(b.len())],
                dir.join("k.wgsl").display()
            );
        }
    }
    let mut ran = Ran { dawn: false };
    if let Some(dn) = dawn_node() {
        let dw = run_dawn(&dn, &dir, &c, groups, buffers);
        for (k, (a, b)) in reference.iter().zip(&dw).enumerate() {
            if let Some(i) = first_difference(a, b) {
                panic!(
                    "{}: buffer {} differs at byte {} on Dawn (Metal {:02x?}, Dawn {:02x?})",
                    name,
                    k,
                    i,
                    &a[i..(i + 16).min(a.len())],
                    &b[i..(i + 16).min(b.len())]
                );
            }
        }
        ran.dawn = true;
    }
    ran
}
