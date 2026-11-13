//! Running one compute function with textures and samplers three ways: real Metal on this Mac (the reference), the
//! compiled WGSL on wgpu (whose wgpu-core is the one wgpu-native v29 wraps), and the compiled WGSL on Dawn through its
//! Node.js binding when `AW_DAWN_NODE` names `dawn.node`.
//!
//! The case states the textures and samplers Metal binds; the compiler gets them as the pipeline description's
//! texture and sampler slots, and the WebGPU side binds what the compiled layout asks for: each texture slot's texels
//! in the format the executor stores it in ([`airwgsl::tex::format::stored`]), uploaded as Metal's bytes, and each
//! sampler as the executor maps a Metal sampler that WebGPU expresses (address modes clamp-to-edge, repeat and
//! mirror-repeat; a sampler without mipmaps reads level 0, so its LOD range is [0, 0], which the cases use only with
//! equal minification and magnification filters). A slot with no texture gets a 1 x 1 placeholder of the binding's
//! type, as the executor binds for a null texture.

use airwgsl::api::*;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

pub fn case_dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tex-tests").join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The target's AIR for an MSL source.
pub fn air(dir: &Path, source: &str, flags: &[&str]) -> Vec<u8> {
    let src = dir.join("k.metal");
    std::fs::write(&src, source).unwrap();
    let out = dir.join("k.air");
    let o = Command::new("xcrun")
        .args(["-sdk", "iphoneos", "metal", "-std=metal3.1", "-target", "air64-apple-ios17.7", "-c"])
        .args(flags)
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .output()
        .expect("xcrun metal");
    assert!(o.status.success(), "metal: {}", String::from_utf8_lossy(&o.stderr));
    std::fs::read(&out).unwrap()
}

/// A texture the case binds.
#[derive(Clone)]
pub struct TexIn {
    pub index: u64,
    /// `MTLTextureType`.
    pub ty: u64,
    /// `MTLPixelFormat`.
    pub format: u64,
    pub w: u32,
    pub h: u32,
    /// Depth of a 3D texture, else 1.
    pub d: u32,
    pub array: u32,
    pub levels: u32,
    pub bpp: u32,
    /// Every level, each level's slices one after the other, rows tightly packed.
    pub data: Vec<u8>,
    pub swizzle: [u8; 4],
    /// The levels of the view the function is given, (first level, count); `None` binds the texture itself.
    pub view_levels: Option<(u32, u32)>,
    /// The function writes the texture; every level is read back and compared.
    pub write: bool,
    pub samples: u32,
}

impl TexIn {
    pub fn new(index: u64, ty: u64, format: u64, size: [u32; 3], levels: u32, bpp: u32, data: Vec<u8>) -> TexIn {
        let (d, array) = if ty == 7 { (size[2], 1) } else { (1, size[2]) };
        TexIn {
            index,
            ty,
            format,
            w: size[0],
            h: size[1],
            d,
            array,
            levels,
            bpp,
            data,
            swizzle: [2, 3, 4, 5],
            view_levels: None,
            write: false,
            samples: 1,
        }
    }

    fn slices(&self) -> u32 {
        (if self.ty == 5 { 6 } else { 1 }) * self.array
    }

    /// The WebGPU size: 1D textures are stored as 2D textures of height 1.
    fn size(&self) -> [u32; 3] {
        [self.w, if matches!(self.ty, 0 | 1) { 1 } else { self.h }, if self.ty == 7 { self.d } else { self.slices() }]
    }

    fn slot(&self) -> TextureSlot {
        TextureSlot {
            index: self.index,
            texture_type: self.ty,
            pixel_format: self.format,
            sample_count: self.samples as u64,
            swizzle: self.swizzle,
            mipmap_level_count: self.bound_levels().1 as u64,
        }
    }

    /// The levels of the texture the function is given: (first level, count).
    fn bound_levels(&self) -> (u32, u32) {
        self.view_levels.unwrap_or((0, self.levels))
    }
}

/// A sampler descriptor with the fields the cases set.
pub fn sampler(index: u64, filter: u64, mip: u64, modes: [u64; 3], normalized: bool, compare: u64) -> SamplerSlot {
    SamplerSlot {
        index,
        min_filter: filter,
        mag_filter: filter,
        mip_filter: mip,
        address_modes: modes,
        border_color: 0,
        normalized_coordinates: normalized,
        compare_function: compare,
        max_anisotropy: 1,
        lod_min_clamp: 0.0,
        lod_max_clamp: f32::MAX,
        lod_average: false,
    }
}

pub struct Case<'a> {
    pub name: &'a str,
    pub source: &'a str,
    pub function: &'a str,
    pub threads: [u64; 3],
    pub groups: [u32; 3],
    pub buffers: Vec<Vec<u8>>,
    pub textures: Vec<TexIn>,
    /// Texture slots bound to no texture.
    pub nulls: Vec<u64>,
    pub samplers: Vec<SamplerSlot>,
    /// Compiler options for both compilations (the reference then loads a library built for this Mac).
    pub flags: Vec<&'a str>,
}

impl Case<'_> {
    pub fn new<'a>(name: &'a str, source: &'a str, function: &'a str) -> Case<'a> {
        Case {
            name,
            source,
            function,
            threads: [1, 1, 1],
            groups: [1, 1, 1],
            buffers: Vec::new(),
            textures: Vec::new(),
            nulls: Vec::new(),
            samplers: Vec::new(),
            flags: Vec::new(),
        }
    }
}

pub fn host() -> HostCaps {
    let mut h = HostCaps::default();
    h.features.insert("shader-f16".into());
    h
}

/// Compile the case's kernel with its bindings as the pipeline description.
#[allow(clippy::result_large_err)]
pub fn compile_case(c: &Case<'_>, bitcode: &[u8]) -> Result<Compiled, Refusal> {
    compile_case_on(c, bitcode, &host())
}

/// [`compile_case`] for a host with the capabilities `host`.
#[allow(clippy::result_large_err)]
pub fn compile_case_on(c: &Case<'_>, bitcode: &[u8], host: &HostCaps) -> Result<Compiled, Refusal> {
    let m = AirModule {
        hash: [0; 32],
        bitcode: Arc::from(bitcode),
        entry: Some(EntryPoint { function: c.function.into(), function_type: 3, constants: Vec::new() }),
    };
    let desc = PipelineDesc {
        kind: PipelineKind::Compute(ComputeState {
            threads_per_threadgroup: c.threads,
            non_uniform_grid: false,
            threadgroup_memory_lengths: Vec::new(),
            stage_input: None,
        }),
        bindings: vec![StageBindings {
            stage: MetalStage::Kernel,
            buffers: (0..c.buffers.len() as u64)
                .map(|index| BufferSlot { index, alias_group: None, also_vertex_or_index_buffer: false })
                .collect(),
            textures: c.textures.iter().map(|t| t.slot()).collect(),
            samplers: c.samplers.clone(),
            argument_buffers: Vec::new(),
        }],
    };
    compile(&[m], &desc, host)
}

// Real Metal ------------------------------------------------------------------------------------------------------

fn metal_runner() -> PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let exe = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("metal_tex");
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/tex/metal_tex.swift");
        let o = Command::new("swiftc").args(["-O", "-o"]).arg(&exe).arg(&src).output().expect("swiftc");
        assert!(o.status.success(), "swiftc: {}", String::from_utf8_lossy(&o.stderr));
        exe
    })
    .clone()
}

/// The buffers after the dispatch, then level 0 of each written texture.
pub fn run_metal(dir: &Path, c: &Case<'_>) -> Vec<Vec<u8>> {
    let src = dir.join("ref.metal");
    std::fs::write(&src, c.source).unwrap();
    let mut spec = String::new();
    if !c.flags.is_empty() {
        let lib = dir.join("ref.metallib");
        let o = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-std=metal3.1"])
            .args(&c.flags)
            .arg(&src)
            .arg("-o")
            .arg(&lib)
            .output()
            .expect("xcrun metal");
        assert!(o.status.success(), "metal: {}", String::from_utf8_lossy(&o.stderr));
        spec.push_str(&format!("library {}\n", lib.display()));
    }
    spec += &format!(
        "function {}\nthreads {} {} {}\ngroups {} {} {}\n",
        c.function, c.threads[0], c.threads[1], c.threads[2], c.groups[0], c.groups[1], c.groups[2]
    );
    let mut files = Vec::new();
    for (i, b) in c.buffers.iter().enumerate() {
        let p = dir.join(format!("metal{}.bin", i));
        std::fs::write(&p, b).unwrap();
        spec.push_str(&format!("buffer {} {}\n", i, p.display()));
        files.push(p.with_extension("bin.out"));
    }
    for t in &c.textures {
        let p = dir.join(format!("mtex{}.bin", t.index));
        std::fs::write(&p, &t.data).unwrap();
        spec.push_str(&format!(
            "texture {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
            t.index,
            t.ty,
            t.format,
            t.w,
            t.h,
            t.d,
            t.array,
            t.levels,
            t.bpp,
            t.write as u32,
            t.swizzle[0],
            t.swizzle[1],
            t.swizzle[2],
            t.swizzle[3],
            t.bound_levels().0,
            t.bound_levels().1,
            p.display()
        ));
        if t.write {
            let r = dir.join(format!("mtex{}.out", t.index));
            spec.push_str(&format!("readback {} {}\n", t.index, r.display()));
            files.push(r);
        }
    }
    for n in &c.nulls {
        spec.push_str(&format!("null {}\n", n));
    }
    for s in &c.samplers {
        spec.push_str(&format!(
            "sampler {} {} {} {} {} {} {} {} {} {} {} {}\n",
            s.index,
            s.min_filter,
            s.mag_filter,
            s.mip_filter,
            s.address_modes[0],
            s.address_modes[1],
            s.address_modes[2],
            s.normalized_coordinates as u32,
            s.compare_function,
            s.max_anisotropy,
            s.lod_min_clamp,
            if s.lod_max_clamp == f32::MAX { 3.4028234663852886e38 } else { s.lod_max_clamp as f64 }
        ));
    }
    let sp = dir.join("spec.txt");
    std::fs::write(&sp, spec).unwrap();
    let o = Command::new(metal_runner()).arg(&src).arg(&sp).output().unwrap();
    assert!(o.status.success(), "{}: metal_tex: {}", c.name, String::from_utf8_lossy(&o.stderr));
    files.iter().map(|f| std::fs::read(f).unwrap()).collect()
}

// The WebGPU binding plan -----------------------------------------------------------------------------------------

/// One binding of the compiled layout, as both WebGPU runners create it.
#[derive(Clone, Debug)]
pub enum Bind {
    Uniform(Vec<u8>),
    Buffer {
        index: usize,
        writable: bool,
    },
    Texture {
        /// Bindings with one key share the texture (the levels of a written Metal texture).
        key: u64,
        format: String,
        dimension: &'static str,
        size: [u32; 3],
        levels: u32,
        bpp: u32,
        data: Vec<u8>,
        view: &'static str,
        /// The levels the view shows: (first level, count).
        view_levels: (u32, u32),
        storage: bool,
        readback: bool,
    },
    Sampler {
        modes: [&'static str; 3],
        mag: &'static str,
        min: &'static str,
        mip: &'static str,
        lod: (f32, f32),
        compare: Option<&'static str>,
    },
}

fn view_name(d: TextureViewDimension) -> &'static str {
    match d {
        TextureViewDimension::D1 => "1d",
        TextureViewDimension::D2 => "2d",
        TextureViewDimension::D2Array => "2d-array",
        TextureViewDimension::Cube => "cube",
        TextureViewDimension::CubeArray => "cube-array",
        TextureViewDimension::D3 => "3d",
    }
}

fn mode(m: u64) -> &'static str {
    match m {
        0 => "clamp-to-edge",
        2 => "repeat",
        3 => "mirror-repeat",
        _ => panic!("address mode {} has no WebGPU sampler", m),
    }
}

fn compare_name(f: u64) -> &'static str {
    ["never", "less", "equal", "less-equal", "greater", "not-equal", "greater-equal", "always"][f as usize]
}

/// The plan for the compiled layout: (group, binding, what).
pub fn plan(c: &Case<'_>, compiled: &Compiled) -> Vec<(u32, u32, Bind)> {
    let mut out = Vec::new();
    for g in &compiled.layout.groups {
        for e in &g.entries {
            let b = match (&e.source, &e.resource) {
                (BindingSource::Uniforms(u), _) => {
                    let mut bytes = vec![0u8; u.size as usize];
                    for f in &u.fields {
                        let v: u32 = match &f.value {
                            UniformValue::BufferLength { index, .. } => c.buffers[*index as usize].len() as u32,
                            _ => 0,
                        };
                        bytes[f.offset as usize..f.offset as usize + 4].copy_from_slice(&v.to_le_bytes());
                    }
                    Bind::Uniform(bytes)
                }
                (BindingSource::MetalBuffer { index, .. }, BindingResource::Buffer { ty, .. }) => {
                    Bind::Buffer { index: *index as usize, writable: *ty == BufferBindingType::Storage }
                }
                (
                    src @ (BindingSource::MetalTexture { index, .. } | BindingSource::MetalTextureLevel { index, .. }),
                    r,
                ) => {
                    let level = match src {
                        BindingSource::MetalTextureLevel { level, .. } => Some(*level),
                        _ => None,
                    };
                    let (view, storage) = match r {
                        BindingResource::Texture { view_dimension, .. } => (*view_dimension, None),
                        BindingResource::StorageTexture { view_dimension, format, .. } => {
                            (*view_dimension, Some(format.clone()))
                        }
                        _ => panic!("texture source with {:?}", r),
                    };
                    match c.textures.iter().find(|t| t.index == *index) {
                        Some(t) => {
                            let stored = airwgsl::tex::format::stored(t.format, &host()).unwrap();
                            // The executor's views: a storage binding is one level of the bound view, a sampled
                            // binding every level of it.
                            let (base, count) = t.bound_levels();
                            Bind::Texture {
                                key: *index,
                                format: stored.webgpu.to_string(),
                                dimension: if t.ty == 7 { "3d" } else { "2d" },
                                size: t.size(),
                                levels: t.levels,
                                bpp: t.bpp,
                                data: t.data.clone(),
                                view: view_name(view),
                                view_levels: match level {
                                    Some(l) => (base + l, 1),
                                    None => (base, count),
                                },
                                storage: storage.is_some(),
                                readback: t.write,
                            }
                        }
                        None => {
                            // The executor's placeholder for a null texture.
                            let (format, bpp) = match (r, &storage) {
                                (_, Some(f)) => (f.clone(), 16),
                                (BindingResource::Texture { sample_type: TextureSampleType::Depth, .. }, _) => {
                                    ("depth32float".to_string(), 4)
                                }
                                (BindingResource::Texture { sample_type: TextureSampleType::Sint, .. }, _) => {
                                    ("rgba32sint".to_string(), 16)
                                }
                                (BindingResource::Texture { sample_type: TextureSampleType::Uint, .. }, _) => {
                                    ("rgba32uint".to_string(), 16)
                                }
                                _ => ("rgba8unorm".to_string(), 4),
                            };
                            let layers = if matches!(view, TextureViewDimension::Cube) { 6 } else { 1 };
                            assert_eq!(level.unwrap_or(0), 0, "a placeholder has one level");
                            Bind::Texture {
                                key: u64::MAX - out.len() as u64,
                                format,
                                dimension: if view == TextureViewDimension::D3 { "3d" } else { "2d" },
                                size: [1, 1, layers],
                                levels: 1,
                                bpp,
                                data: vec![0; bpp as usize * layers as usize],
                                view: view_name(view),
                                view_levels: (0, 1),
                                storage: storage.is_some(),
                                readback: false,
                            }
                        }
                    }
                }
                (src @ (BindingSource::MetalSampler { .. } | BindingSource::ConstantSampler { .. }), r) => {
                    let BindingResource::Sampler { ty } = r else { panic!("sampler source with {:?}", r) };
                    let d = match src {
                        BindingSource::MetalSampler { index, .. } => {
                            *c.samplers.iter().find(|s| s.index == *index).expect("a bound sampler")
                        }
                        BindingSource::ConstantSampler { encoding } => {
                            let d = airwgsl::air::sampler::decode(*encoding).unwrap();
                            sampler(
                                0,
                                d.min_filter,
                                d.mip_filter,
                                d.address_modes,
                                d.normalized_coordinates,
                                d.compare_function,
                            )
                        }
                        _ => unreachable!(),
                    };
                    let f = |x: u64| if x == 1 { "linear" } else { "nearest" };
                    Bind::Sampler {
                        modes: [mode(d.address_modes[0]), mode(d.address_modes[1]), mode(d.address_modes[2])],
                        mag: f(d.mag_filter),
                        min: f(d.min_filter),
                        mip: if d.mip_filter == 2 { "linear" } else { "nearest" },
                        lod: if d.mip_filter == 0 { (0.0, 0.0) } else { (d.lod_min_clamp, d.lod_max_clamp.min(32.0)) },
                        compare: if *ty == SamplerBindingType::Comparison {
                            Some(compare_name(d.compare_function))
                        } else {
                            None
                        },
                    }
                }
                other => panic!("binding {:?}", other),
            };
            out.push((g.group, e.binding, b));
        }
    }
    out
}

// wgpu ------------------------------------------------------------------------------------------------------------

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

static RUN: Mutex<()> = Mutex::new(());

pub fn gpu() -> &'static Gpu {
    static G: OnceLock<Gpu> = OnceLock::new();
    G.get_or_init(|| {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no WebGPU adapter: the texture differential tests need one");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("texture tests"),
            required_features: adapter.features() & wgpu::Features::SHADER_F16,
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("request_device");
        Gpu { device, queue }
    })
}

fn wgpu_format(name: &str) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as F;
    match name {
        "r8unorm" => F::R8Unorm,
        "rg8unorm" => F::Rg8Unorm,
        "rgba8unorm" => F::Rgba8Unorm,
        "rgba8unorm-srgb" => F::Rgba8UnormSrgb,
        "rgba8snorm" => F::Rgba8Snorm,
        "rgba8uint" => F::Rgba8Uint,
        "rgba8sint" => F::Rgba8Sint,
        "r16uint" => F::R16Uint,
        "r16float" => F::R16Float,
        "rgba16uint" => F::Rgba16Uint,
        "rgba16float" => F::Rgba16Float,
        "r32uint" => F::R32Uint,
        "r32sint" => F::R32Sint,
        "r32float" => F::R32Float,
        "rg32float" => F::Rg32Float,
        "rgba32uint" => F::Rgba32Uint,
        "rgba32sint" => F::Rgba32Sint,
        "rgba32float" => F::Rgba32Float,
        "depth32float" => F::Depth32Float,
        "depth16unorm" => F::Depth16Unorm,
        other => panic!("format {} is not in the test table", other),
    }
}

fn wgpu_view(v: &str) -> wgpu::TextureViewDimension {
    use wgpu::TextureViewDimension as V;
    match v {
        "2d" => V::D2,
        "2d-array" => V::D2Array,
        "cube" => V::Cube,
        "3d" => V::D3,
        _ => V::D1,
    }
}

fn wgpu_mode(m: &str) -> wgpu::AddressMode {
    match m {
        "repeat" => wgpu::AddressMode::Repeat,
        "mirror-repeat" => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::ClampToEdge,
    }
}

fn wgpu_filter(f: &str) -> wgpu::FilterMode {
    if f == "linear" { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest }
}

fn wgpu_compare(c: &str) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as C;
    match c {
        "never" => C::Never,
        "less" => C::Less,
        "equal" => C::Equal,
        "less-equal" => C::LessEqual,
        "greater" => C::Greater,
        "not-equal" => C::NotEqual,
        "greater-equal" => C::GreaterEqual,
        _ => C::Always,
    }
}

fn binding_type(r: &BindingResource) -> wgpu::BindingType {
    match r {
        BindingResource::Buffer { ty, has_dynamic_offset, min_binding_size } => wgpu::BindingType::Buffer {
            ty: match ty {
                BufferBindingType::Uniform => wgpu::BufferBindingType::Uniform,
                BufferBindingType::Storage => wgpu::BufferBindingType::Storage { read_only: false },
                BufferBindingType::ReadOnlyStorage => wgpu::BufferBindingType::Storage { read_only: true },
            },
            has_dynamic_offset: *has_dynamic_offset,
            min_binding_size: std::num::NonZeroU64::new(*min_binding_size),
        },
        BindingResource::Sampler { ty } => wgpu::BindingType::Sampler(match ty {
            SamplerBindingType::Filtering => wgpu::SamplerBindingType::Filtering,
            SamplerBindingType::NonFiltering => wgpu::SamplerBindingType::NonFiltering,
            SamplerBindingType::Comparison => wgpu::SamplerBindingType::Comparison,
        }),
        BindingResource::Texture { sample_type, view_dimension, multisampled } => wgpu::BindingType::Texture {
            sample_type: match sample_type {
                TextureSampleType::Float => wgpu::TextureSampleType::Float { filterable: true },
                TextureSampleType::UnfilterableFloat => wgpu::TextureSampleType::Float { filterable: false },
                TextureSampleType::Depth => wgpu::TextureSampleType::Depth,
                TextureSampleType::Sint => wgpu::TextureSampleType::Sint,
                TextureSampleType::Uint => wgpu::TextureSampleType::Uint,
            },
            view_dimension: wgpu_view(view_name(*view_dimension)),
            multisampled: *multisampled,
        },
        BindingResource::StorageTexture { format, view_dimension, .. } => wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu_format(format),
            view_dimension: wgpu_view(view_name(*view_dimension)),
        },
    }
}

/// The `GPUBindGroupLayoutEntry` members of a binding, as JSON.
fn layout_json(r: &BindingResource) -> String {
    match r {
        BindingResource::Buffer { ty, .. } => format!(
            "\"buffer\":{{\"type\":\"{}\"}}",
            match ty {
                BufferBindingType::Uniform => "uniform",
                BufferBindingType::Storage => "storage",
                BufferBindingType::ReadOnlyStorage => "read-only-storage",
            }
        ),
        BindingResource::Sampler { ty } => format!(
            "\"sampler\":{{\"type\":\"{}\"}}",
            match ty {
                SamplerBindingType::Filtering => "filtering",
                SamplerBindingType::NonFiltering => "non-filtering",
                SamplerBindingType::Comparison => "comparison",
            }
        ),
        BindingResource::Texture { sample_type, view_dimension, multisampled } => format!(
            "\"texture\":{{\"sampleType\":\"{}\",\"viewDimension\":\"{}\",\"multisampled\":{}}}",
            match sample_type {
                TextureSampleType::Float => "float",
                TextureSampleType::UnfilterableFloat => "unfilterable-float",
                TextureSampleType::Depth => "depth",
                TextureSampleType::Sint => "sint",
                TextureSampleType::Uint => "uint",
            },
            view_name(*view_dimension),
            multisampled
        ),
        BindingResource::StorageTexture { format, view_dimension, .. } => format!(
            "\"storageTexture\":{{\"access\":\"write-only\",\"format\":\"{}\",\"viewDimension\":\"{}\"}}",
            format,
            view_name(*view_dimension)
        ),
    }
}

/// Create the compiled layout's bind group layouts, the pipeline, and a bind group of textures and samplers as the
/// case binds them (multisampled textures are created as render attachments and not uploaded), under a validation
/// error scope: the WebGPU validation of layouts (WebGPU 8.1.1) and bind groups (8.3.1). Returns the first error.
pub fn validate_bindings(c: &Case<'_>, compiled: &Compiled) -> Option<String> {
    let _guard = RUN.lock().unwrap_or_else(|e| e.into_inner());
    let g = gpu();
    let d = &g.device;
    let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(compiled.wgsl[0].source.as_str().into()),
    });
    let ngroups = compiled.layout.groups.iter().map(|g| g.group + 1).max().unwrap_or(0);
    let bgls: Vec<wgpu::BindGroupLayout> = (0..ngroups)
        .map(|gi| {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = compiled
                .layout
                .groups
                .iter()
                .filter(|g| g.group == gi)
                .flat_map(|g| g.entries.iter())
                .map(|e| wgpu::BindGroupLayoutEntry {
                    binding: e.binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: binding_type(&e.resource),
                    count: None,
                })
                .collect();
            d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries })
        })
        .collect();
    let refs: Vec<Option<&wgpu::BindGroupLayout>> = bgls.iter().map(Some).collect();
    let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &refs,
        immediate_size: 0,
    });
    let _pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &module,
        entry_point: Some(&compiled.wgsl[0].entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut views = Vec::new();
    let mut samplers = Vec::new();
    for grp in &compiled.layout.groups {
        for e in &grp.entries {
            match (&e.source, &e.resource) {
                (BindingSource::MetalTexture { index, .. }, BindingResource::Texture { view_dimension, .. }) => {
                    let t = c.textures.iter().find(|t| t.index == *index).expect("a bound texture");
                    let stored = airwgsl::tex::format::stored(t.format, &host()).unwrap();
                    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING;
                    if t.samples > 1 {
                        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
                    }
                    let size = t.size();
                    let tex = d.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: size[2] },
                        mip_level_count: t.levels,
                        sample_count: t.samples,
                        dimension: if t.ty == 7 { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 },
                        format: wgpu_format(stored.webgpu),
                        usage,
                        view_formats: &[],
                    });
                    let v = tex.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu_view(view_name(*view_dimension))),
                        ..Default::default()
                    });
                    views.push((grp.group, e.binding, v));
                }
                (BindingSource::MetalSampler { .. } | BindingSource::ConstantSampler { .. }, r) => {
                    let BindingResource::Sampler { ty } = r else { continue };
                    let s = d.create_sampler(&wgpu::SamplerDescriptor {
                        compare: (*ty == SamplerBindingType::Comparison).then_some(wgpu::CompareFunction::Less),
                        ..Default::default()
                    });
                    samplers.push((grp.group, e.binding, s));
                }
                _ => {}
            }
        }
    }
    for (gi, bgl) in bgls.iter().enumerate() {
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
        for (grp, b, v) in &views {
            if *grp as usize == gi {
                entries.push(wgpu::BindGroupEntry { binding: *b, resource: wgpu::BindingResource::TextureView(v) });
            }
        }
        for (grp, b, s) in &samplers {
            if *grp as usize == gi {
                entries.push(wgpu::BindGroupEntry { binding: *b, resource: wgpu::BindingResource::Sampler(s) });
            }
        }
        if entries.len()
            == compiled.layout.groups.iter().filter(|g| g.group as usize == gi).map(|g| g.entries.len()).sum::<usize>()
        {
            let _ = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: bgl, entries: &entries });
        }
    }
    pollster::block_on(scope.pop()).map(|e| e.to_string())
}

pub fn run_wgpu(c: &Case<'_>, compiled: &Compiled, plan: &[(u32, u32, Bind)]) -> Vec<Vec<u8>> {
    use wgpu::util::DeviceExt;
    let _guard = RUN.lock().unwrap_or_else(|e| e.into_inner());
    let g = gpu();
    let d = &g.device;
    let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(compiled.wgsl[0].source.as_str().into()),
    });
    // The executor's layout: the compiled binding layout, group by group.
    let bgls: Vec<wgpu::BindGroupLayout> = (0..compiled.layout.groups.iter().map(|g| g.group + 1).max().unwrap_or(0))
        .map(|gi| {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = compiled
                .layout
                .groups
                .iter()
                .filter(|g| g.group == gi)
                .flat_map(|g| g.entries.iter())
                .map(|e| wgpu::BindGroupLayoutEntry {
                    binding: e.binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: binding_type(&e.resource),
                    count: None,
                })
                .collect();
            d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries })
        })
        .collect();
    let refs: Vec<Option<&wgpu::BindGroupLayout>> = bgls.iter().map(Some).collect();
    let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &refs,
        immediate_size: 0,
    });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &module,
        entry_point: Some(&compiled.wgsl[0].entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    let bufs: Vec<wgpu::Buffer> = c
        .buffers
        .iter()
        .map(|b| {
            let mut data = b.clone();
            while data.len() % 4 != 0 || data.is_empty() {
                data.push(0);
            }
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        })
        .collect();
    enum Res {
        Buf(wgpu::Buffer),
        View(wgpu::TextureView),
        Smp(wgpu::Sampler),
    }
    let mut res: Vec<(u32, u32, Res)> = Vec::new();
    // (texture, size, bytes per texel, levels, 3D) of each texture read back.
    let mut readbacks: Vec<(wgpu::Texture, [u32; 3], u32, u32, bool)> = Vec::new();
    let mut textures: std::collections::HashMap<u64, wgpu::Texture> = std::collections::HashMap::new();
    let storage_keys: std::collections::HashSet<u64> = plan
        .iter()
        .filter_map(|(_, _, b)| match b {
            Bind::Texture { key, storage: true, .. } => Some(*key),
            _ => None,
        })
        .collect();
    for (grp, bind, b) in plan {
        let r = match b {
            Bind::Uniform(bytes) => Res::Buf(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })),
            Bind::Buffer { index, .. } => Res::Buf(bufs[*index].clone()),
            Bind::Texture {
                key,
                format,
                dimension,
                size,
                levels,
                bpp,
                data,
                view,
                view_levels,
                storage: _,
                readback,
            } => {
                if let Some(tex) = textures.get(key) {
                    res.push((
                        *grp,
                        *bind,
                        Res::View(tex.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu_view(view)),
                            base_mip_level: view_levels.0,
                            mip_level_count: Some(view_levels.1),
                            ..Default::default()
                        })),
                    ));
                    continue;
                }
                let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC;
                if storage_keys.contains(key) {
                    usage |= wgpu::TextureUsages::STORAGE_BINDING;
                }
                let tex = d.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: size[2] },
                    mip_level_count: *levels,
                    sample_count: 1,
                    dimension: if *dimension == "3d" { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 },
                    format: wgpu_format(format),
                    usage,
                    view_formats: &[],
                });
                let mut off = 0usize;
                for l in 0..*levels {
                    let w = (size[0] >> l).max(1);
                    let h = (size[1] >> l).max(1);
                    let layers = if *dimension == "3d" { (size[2] >> l).max(1) } else { size[2] };
                    let n = (w * h * layers * bpp) as usize;
                    g.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &tex,
                            mip_level: l,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &data[off..off + n],
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(w * bpp),
                            rows_per_image: Some(h),
                        },
                        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
                    );
                    off += n;
                }
                let v = tex.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu_view(view)),
                    base_mip_level: view_levels.0,
                    mip_level_count: Some(view_levels.1),
                    ..Default::default()
                });
                if *readback {
                    readbacks.push((tex.clone(), *size, *bpp, *levels, *dimension == "3d"));
                }
                textures.insert(*key, tex);
                Res::View(v)
            }
            Bind::Sampler { modes, mag, min, mip, lod, compare } => {
                Res::Smp(d.create_sampler(&wgpu::SamplerDescriptor {
                    label: None,
                    address_mode_u: wgpu_mode(modes[0]),
                    address_mode_v: wgpu_mode(modes[1]),
                    address_mode_w: wgpu_mode(modes[2]),
                    mag_filter: wgpu_filter(mag),
                    min_filter: wgpu_filter(min),
                    mipmap_filter: if *mip == "linear" {
                        wgpu::MipmapFilterMode::Linear
                    } else {
                        wgpu::MipmapFilterMode::Nearest
                    },
                    lod_min_clamp: lod.0,
                    lod_max_clamp: lod.1,
                    compare: compare.map(wgpu_compare),
                    anisotropy_clamp: 1,
                    border_color: None,
                }))
            }
        };
        res.push((*grp, *bind, r));
    }
    let mut enc = d.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        let mut groups: Vec<u32> = res.iter().map(|r| r.0).collect();
        groups.dedup();
        for grp in groups {
            let entries: Vec<wgpu::BindGroupEntry> = res
                .iter()
                .filter(|r| r.0 == grp)
                .map(|(_, b, r)| wgpu::BindGroupEntry {
                    binding: *b,
                    resource: match r {
                        Res::Buf(x) => x.as_entire_binding(),
                        Res::View(v) => wgpu::BindingResource::TextureView(v),
                        Res::Smp(s) => wgpu::BindingResource::Sampler(s),
                    },
                })
                .collect();
            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgls[grp as usize],
                entries: &entries,
            });
            pass.set_bind_group(grp, &bg, &[]);
        }
        pass.dispatch_workgroups(c.groups[0], c.groups[1], c.groups[2]);
    }
    let mut reads = Vec::new();
    for b in &bufs {
        let r = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: b.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(b, 0, &r, 0, b.size());
        reads.push(r);
    }
    // Every level of a texture read back, one buffer each, joined into one output.
    let mut level_reads: Vec<Vec<(wgpu::Buffer, [u32; 3], u32)>> = Vec::new();
    for (tex, size, bpp, levels, three_d) in &readbacks {
        let mut per_level = Vec::new();
        for l in 0..*levels {
            let extent =
                [(size[0] >> l).max(1), (size[1] >> l).max(1), if *three_d { (size[2] >> l).max(1) } else { size[2] }];
            let row = (extent[0] * bpp).div_ceil(256) * 256;
            let r = d.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (row * extent[1] * extent[2]) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: l,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &r,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(extent[1]),
                    },
                },
                wgpu::Extent3d { width: extent[0], height: extent[1], depth_or_array_layers: extent[2] },
            );
            per_level.push((r, extent, row));
        }
        level_reads.push(per_level);
    }
    g.queue.submit([enc.finish()]);
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("{}: wgpu validation: {}\n{}", c.name, e, compiled.wgsl[0].source);
    }
    let mut out = Vec::new();
    for (k, r) in reads.iter().enumerate() {
        let s = r.slice(..);
        s.map_async(wgpu::MapMode::Read, |_| {});
        d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        out.push(s.get_mapped_range()[..c.buffers[k].len()].to_vec());
    }
    for (per_level, (_, _, bpp, _, _)) in level_reads.iter().zip(&readbacks) {
        let mut v = Vec::new();
        for (r, extent, row) in per_level {
            let s = r.slice(..);
            s.map_async(wgpu::MapMode::Read, |_| {});
            d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let all = s.get_mapped_range();
            let tight = (extent[0] * bpp) as usize;
            for z in 0..extent[2] {
                for y in 0..extent[1] {
                    let o = ((z * extent[1] + y) * row) as usize;
                    v.extend_from_slice(&all[o..o + tight]);
                }
            }
        }
        out.push(v);
    }
    out
}

// Dawn ------------------------------------------------------------------------------------------------------------

pub fn dawn_node() -> Option<PathBuf> {
    std::env::var_os("AW_DAWN_NODE").map(PathBuf::from)
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

pub fn run_dawn(dawn: &Path, dir: &Path, c: &Case<'_>, compiled: &Compiled, plan: &[(u32, u32, Bind)]) -> Vec<Vec<u8>> {
    let shader = dir.join("k.wgsl");
    std::fs::write(&shader, &compiled.wgsl[0].source).unwrap();
    let mut files = vec![None; c.buffers.len()];
    let mut tex_out = Vec::new();
    let mut entries = Vec::new();
    for (n, (grp, bind, b)) in plan.iter().enumerate() {
        let e = match b {
            Bind::Uniform(bytes) => {
                let p = dir.join(format!("u{}.bin", n));
                std::fs::write(&p, bytes).unwrap();
                format!("{{\"kind\":\"uniform\",\"file\":{}}}", json_str(&p.display().to_string()))
            }
            Bind::Buffer { index, writable } => {
                let p = dir.join(format!("dawn{}.bin", index));
                std::fs::write(&p, &c.buffers[*index]).unwrap();
                files[*index] = Some(p.clone());
                format!(
                    "{{\"kind\":\"{}\",\"file\":{}}}",
                    if *writable { "storage" } else { "read-only-storage" },
                    json_str(&p.display().to_string())
                )
            }
            Bind::Texture { key, format, dimension, size, levels, bpp, data, view, view_levels, storage, readback } => {
                let p = dir.join(format!("dtex{}.bin", n));
                std::fs::write(&p, data).unwrap();
                let first = !plan[..n].iter().any(|(_, _, b)| matches!(b, Bind::Texture { key: k, .. } if k == key));
                let rb = if *readback && first {
                    let r = dir.join(format!("dtex{}.out", n));
                    tex_out.push(r.clone());
                    format!(",\"readback\":{}", json_str(&r.display().to_string()))
                } else {
                    String::new()
                };
                let storage_key =
                    plan.iter().any(|(_, _, b)| matches!(b, Bind::Texture { key: k, storage: true, .. } if k == key));
                format!(
                    "{{\"kind\":\"texture\",\"key\":\"{}\",\"format\":\"{}\",\"dimension\":\"{}\",\"size\":[{},{},{}],\
                     \"levels\":{},\"bpp\":{},\"view\":\"{}\",\"baseMipLevel\":{},\"mipLevelCount\":{},\"storage\":{},\
                     \"storageUsage\":{},\"file\":{}{}}}",
                    key,
                    format,
                    dimension,
                    size[0],
                    size[1],
                    size[2],
                    levels,
                    bpp,
                    view,
                    view_levels.0,
                    view_levels.1,
                    storage,
                    storage_key,
                    json_str(&p.display().to_string()),
                    rb
                )
            }
            Bind::Sampler { modes, mag, min, mip, lod, compare } => format!(
                "{{\"kind\":\"sampler\",\"desc\":{{\"addressModeU\":\"{}\",\"addressModeV\":\"{}\",\"addressModeW\":\"{}\",\
                 \"magFilter\":\"{}\",\"minFilter\":\"{}\",\"mipmapFilter\":\"{}\",\"lodMinClamp\":{},\"lodMaxClamp\":{}{}}}}}",
                modes[0],
                modes[1],
                modes[2],
                mag,
                min,
                mip,
                lod.0,
                lod.1,
                compare.map(|c| format!(",\"compare\":\"{}\"", c)).unwrap_or_default()
            ),
        };
        entries.push(e.replacen('{', &format!("{{\"group\":{},\"binding\":{},", grp, bind), 1));
    }
    let ngroups = compiled.layout.groups.iter().map(|g| g.group + 1).max().unwrap_or(0);
    let layouts: Vec<String> = (0..ngroups)
        .map(|gi| {
            let es: Vec<String> = compiled
                .layout
                .groups
                .iter()
                .filter(|g| g.group == gi)
                .flat_map(|g| g.entries.iter())
                .map(|e| format!("{{\"binding\":{},\"visibility\":4,{}}}", e.binding, layout_json(&e.resource)))
                .collect();
            format!("[{}]", es.join(","))
        })
        .collect();
    let features: Vec<String> = compiled.requirements.features.iter().map(|f| json_str(f)).collect();
    let plan_json = format!(
        "{{\"shader\":{},\"entry\":{},\"groups\":[{},{},{}],\"features\":[{}],\"layouts\":[{}],\"bindings\":[{}]}}",
        json_str(&shader.display().to_string()),
        json_str(&compiled.wgsl[0].entry_point),
        c.groups[0],
        c.groups[1],
        c.groups[2],
        features.join(","),
        layouts.join(","),
        entries.join(",")
    );
    let pp = dir.join("plan.json");
    std::fs::write(&pp, plan_json).unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/tex/dawn_tex.cjs");
    let o = Command::new("node").arg(script).arg(dawn).arg(&pp).output().unwrap();
    assert!(
        o.status.success(),
        "{}: dawn_tex: {}\n{}",
        c.name,
        String::from_utf8_lossy(&o.stderr),
        compiled.wgsl[0].source
    );
    let mut out: Vec<Vec<u8>> = files
        .iter()
        .enumerate()
        .map(|(k, f)| match f {
            Some(p) => std::fs::read(p.with_extension("bin.out")).unwrap(),
            None => c.buffers[k].clone(),
        })
        .collect();
    for t in tex_out {
        out.push(std::fs::read(t).unwrap());
    }
    out
}

pub fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i))
}

/// A comparison of Metal's outputs (buffers, then written textures) with a WebGPU run's: the first mismatch.
pub type Check = dyn Fn(&[Vec<u8>], &[Vec<u8>]) -> Option<String>;

/// Which backends ran a case.
pub struct Ran {
    pub dawn: bool,
}

/// Compile and run on Metal, wgpu and (when available) Dawn, and compare each WebGPU run with Metal by `check`.
pub fn differential_with(c: &Case<'_>, check: &Check) -> Ran {
    let dir = case_dir(c.name);
    let bc = air(&dir, c.source, &c.flags);
    let compiled = compile_case(c, &bc).unwrap_or_else(|r| panic!("{}: {}", c.name, r));
    std::fs::write(dir.join("k.wgsl"), &compiled.wgsl[0].source).unwrap();
    let reference = run_metal(&dir, c);
    let p = plan(c, &compiled);
    let w = run_wgpu(c, &compiled, &p);
    if let Some(m) = check(&reference, &w) {
        panic!("{}: wgpu: {}; WGSL in {}", c.name, m, dir.join("k.wgsl").display());
    }
    let mut ran = Ran { dawn: false };
    if let Some(dn) = dawn_node() {
        let dw = run_dawn(&dn, &dir, c, &compiled, &p);
        if let Some(m) = check(&reference, &dw) {
            panic!("{}: Dawn: {}; WGSL in {}", c.name, m, dir.join("k.wgsl").display());
        }
        ran.dawn = true;
    }
    ran
}

/// Byte-identical outputs.
pub fn exact(a: &[Vec<u8>], b: &[Vec<u8>]) -> Option<String> {
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        if let Some(i) = first_difference(x, y) {
            return Some(format!(
                "output {} differs at byte {} (Metal {:02x?}, WebGPU {:02x?})",
                k,
                i,
                &x[i..(i + 16).min(x.len())],
                &y[i..(i + 16).min(y.len())]
            ));
        }
    }
    if a.len() != b.len() {
        return Some(format!("{} outputs against {}", a.len(), b.len()));
    }
    None
}

pub fn differential(c: &Case<'_>) -> Ran {
    differential_with(c, &exact)
}

/// Little-endian bytes of `u32`s and `f32`s.
pub fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
