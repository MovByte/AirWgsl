//! The stage rules' decisions: what links and what Metal refuses (varyings, color outputs against their formats), the
//! cases without an exact lowering, and the generated code of the sample queries, dual-source outputs, point sizes and
//! the linear thread index of a 3D threadgroup (run against real Metal on this Mac).

use super::harness::*;
use airwgsl::api::*;
use airwgsl::wgsl::tracked;
use std::process::Command;

fn item(r: &Refusal) -> String {
    tracked::item_of(r).map(|(_, id)| id).unwrap_or_default()
}

fn validate(src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}\n{}", e.emit_to_string(src), src));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{:?}\n{}", e, src));
}

fn pipeline(bc: &[u8], vs: &str, fs: &str, desc: &PipelineDesc, host: &HostCaps) -> Result<Compiled, Refusal> {
    let c = compile(&[module(bc, vs, 1), module(bc, fs, 2)], desc, host)?;
    for s in &c.wgsl {
        validate(&s.source);
    }
    Ok(c)
}

const VARYINGS: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct VI { float4 pos [[position]]; int x [[user(x)]]; };
struct VF { float4 pos [[position]]; float x [[user(x)]]; };
struct VS { float4 pos [[position]]; short x [[user(x)]]; };
struct FU { float4 pos [[position]]; uint x [[user(x)]]; };
struct FH { float4 pos [[position]]; half x [[user(x)]]; };
struct FN { float4 pos [[position]]; float y [[user(y)]]; };
struct FV { float4 pos [[position]]; float4 x [[user(x)]]; half z [[user(z)]]; };
vertex VI v_int(uint vid [[vertex_id]]) { VI o; o.pos = float4(0, 0, 0, 1); o.x = -int(vid); return o; }
vertex VF v_float(uint vid [[vertex_id]]) { VF o; o.pos = float4(0, 0, 0, 1); o.x = float(vid); return o; }
vertex VS v_short(uint vid [[vertex_id]]) { VS o; o.pos = float4(0, 0, 0, 1); o.x = short(vid); return o; }
fragment uint4 f_uint(FU in [[stage_in]]) { return uint4(in.x); }
fragment uint4 f_half(FH in [[stage_in]]) { return uint4(as_type<ushort>(in.x)); }
fragment uint4 f_other(FN in [[stage_in]]) { return uint4(as_type<uint>(in.y)); }
fragment uint4 f_unread(FV in [[stage_in]]) { return uint4(1); }
"#;

/// Real Metal links an int output to a uint input of the same name and refuses other type differences and inputs no
/// output writes (measured: "Fragment input(s) ... mismatching vertex shader output type(s) or not written by vertex
/// shader"); an input the fragment function does not read links whatever its type and whether or not an output
/// writes it.
#[test]
fn varyings_link_as_metal_links_them() {
    let dir = case_dir("varyings");
    let bc = air(&dir, VARYINGS, "metal3.1");
    let desc = render_desc(123, 3, 1, None);
    pipeline(&bc, "v_int", "f_uint", &desc, &host()).unwrap();
    pipeline(&bc, "v_int", "f_unread", &desc, &host()).unwrap();
    for (vs, fs) in [("v_float", "f_half"), ("v_short", "f_uint"), ("v_int", "f_other")] {
        let e = pipeline(&bc, vs, fs, &desc, &host()).unwrap_err();
        assert_eq!(
            (e.kind, item(&e).as_str()),
            (RefusalKind::RejectedByDevice, "varying-mismatch"),
            "{} {}: {}",
            vs,
            fs,
            e
        );
    }
}

const OUTPUTS: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; };
vertex V vt(uint vid [[vertex_id]]) { V o; o.pos = float4(0, 0, 0, 1); return o; }
fragment float f_float(V in [[stage_in]]) { return 0.25f; }
fragment float3 f_float3(V in [[stage_in]]) { return float3(0.25f); }
fragment int4 f_int4(V in [[stage_in]]) { return int4(1); }
struct D { float4 c [[color(0), index(0)]]; float4 s [[color(0), index(1)]]; };
fragment D f_dual(V in [[stage_in]]) { D d; d.c = float4(0.5f); d.s = float4(0.25f); return d; }
struct M { uint c [[color(0)]]; uint m [[sample_mask]]; };
fragment M f_mask(V in [[stage_in]]) { M m; m.c = 1; m.m = 5; return m; }
"#;

fn desc_with(pf: u64, f: impl FnOnce(&mut RenderState)) -> PipelineDesc {
    let mut d = render_desc(pf, 3, 1, None);
    if let PipelineKind::Render(r) = &mut d.kind {
        f(r);
    }
    d
}

/// Color outputs against their attachment's format (Metal's measured rule, the same as WebGPU's), dual-source
/// outputs, and the coverage mask with alpha to coverage.
#[test]
fn color_outputs_follow_the_attachment_format() {
    let dir = case_dir("outputs");
    let bc = air(&dir, OUTPUTS, "metal3.1");
    let rejected = |fs: &str, pf: u64| {
        let e = pipeline(&bc, "vt", fs, &render_desc(pf, 3, 1, None), &host()).unwrap_err();
        assert_eq!(
            (e.kind, item(&e).as_str()),
            (RefusalKind::RejectedByDevice, "color-output-format"),
            "{}: {}",
            fs,
            e
        );
    };
    rejected("f_float", 70); // RGBA8Unorm needs 4 components.
    rejected("f_float", 53); // R32Uint needs an unsigned output.
    rejected("f_int4", 123); // RGBA32Uint needs an unsigned output.
    rejected("f_float3", 1); // A8Unorm needs 4 components.
    pipeline(&bc, "vt", "f_float", &render_desc(10, 3, 1, None), &host()).unwrap();
    pipeline(&bc, "vt", "f_float3", &render_desc(40, 3, 1, None), &host()).unwrap();
    // Source-alpha blend factors with an output without alpha.
    let e = pipeline(
        &bc,
        "vt",
        "f_float3",
        &desc_with(40, |r| {
            r.color_attachments[0].blending_enabled = true;
            r.color_attachments[0].source_rgb_blend_factor = 4;
        }),
        &host(),
    )
    .unwrap_err();
    assert_eq!(item(&e), "color-output-alpha", "{}", e);
    // Dual source: the second output is declared only when blending reads it, and then needs the feature.
    let plain = pipeline(&bc, "vt", "f_dual", &render_desc(70, 3, 1, None), &host()).unwrap();
    assert!(!plain.wgsl[1].source.contains("blend_src"), "{}", plain.wgsl[1].source);
    let dual = desc_with(70, |r| {
        r.color_attachments[0].blending_enabled = true;
        r.color_attachments[0].destination_rgb_blend_factor = 16;
    });
    let e = pipeline(&bc, "vt", "f_dual", &dual, &host()).unwrap_err();
    assert_eq!((e.kind, item(&e).as_str()), (RefusalKind::HostUnsupported, "dual-source-blending"), "{}", e);
    let mut h = host();
    h.features.insert("dual-source-blending".into());
    let c = pipeline(&bc, "vt", "f_dual", &dual, &h).unwrap();
    let fs = &c.wgsl[1].source;
    assert!(fs.contains("enable dual_source_blending;") && fs.contains("@location(0) @blend_src(1)"), "{}", fs);
    assert!(c.requirements.features.contains("dual-source-blending"));
    // A coverage mask output with alpha to coverage, which WebGPU refuses.
    pipeline(&bc, "vt", "f_mask", &render_desc(53, 3, 4, None), &host()).unwrap();
    let e = pipeline(&bc, "vt", "f_mask", &desc_with(53, |r| r.alpha_to_coverage_enabled = true), &host()).unwrap_err();
    assert_eq!(item(&e), "sample-mask-output", "{}", e);
}

const FRAGMENT: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; };
vertex V vt(uint vid [[vertex_id]]) { V o; o.pos = float4(0, 0, 0, 1); return o; }
[[early_fragment_tests]] fragment uint f_early(V in [[stage_in]]) { return 1; }
[[early_fragment_tests]] fragment uint f_early_discard(V in [[stage_in]]) {
    if (in.pos.x < 2.0f) discard_fragment();
    return 1;
}
[[early_fragment_tests]] fragment uint f_early_write(V in [[stage_in]], device uint *b [[buffer(0)]]) {
    b[uint(in.pos.x)] = 1;
    return 1;
}
fragment uint f_mask_in_sample(V in [[stage_in]], uint m [[sample_mask]], uint s [[sample_id]]) { return m + s; }
fragment uint4 f_samples(V in [[stage_in]], uint s [[sample_id]]) {
    float2 p = get_sample_position(s);
    return uint4(as_type<uint2>(p), get_num_samples(), s);
}
"#;

/// Early fragment tests are exact without effects or coverage changes, and refused otherwise; the coverage-mask input
/// of a per-sample function with several samples has no exact lowering.
#[test]
fn early_tests_and_the_coverage_mask_input() {
    let dir = case_dir("fragment");
    let bc = air(&dir, FRAGMENT, "metal3.1");
    pipeline(&bc, "vt", "f_early", &render_desc(53, 3, 1, None), &host()).unwrap();
    for fs in ["f_early_discard", "f_early_write"] {
        let e = pipeline(&bc, "vt", fs, &render_desc(53, 3, 1, None), &host()).unwrap_err();
        assert_eq!(item(&e), "early-fragment-tests", "{}: {}", fs, e);
    }
    pipeline(&bc, "vt", "f_mask_in_sample", &render_desc(53, 3, 1, None), &host()).unwrap();
    let e = pipeline(&bc, "vt", "f_mask_in_sample", &render_desc(23, 3, 4, None), &host()).unwrap_err();
    assert_eq!((e.kind, item(&e).as_str()), (RefusalKind::NoExactLowering, "sample-mask-input"), "{}", e);
}

/// get_num_samples is the raster sample count and get_sample_position the default pattern, which for 4 samples is
/// the positions real Metal reports (the V1 golden of msaa.coverage-4x, `default-sample-positions`): rendered per
/// sample on wgpu and Dawn, each sample reads its own position.
#[test]
fn sample_queries_give_the_default_pattern() {
    let dir = case_dir("samples");
    let bc = air(&dir, FRAGMENT, "metal3.1");
    let c = pipeline(&bc, "vt", "f_samples", &render_desc(123, 3, 1, None), &host()).unwrap();
    assert!(c.wgsl[1].source.contains("1065353216u") || c.wgsl[1].source.contains("1056964608u"));
    let four = pipeline(&bc, "vt", "f_samples", &render_desc(123, 3, 4, None), &host()).unwrap();
    let src = &four.wgsl[1].source;
    let table = [(0.375f32, 0.125f32), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)];
    for (x, y) in table {
        assert!(src.contains(&format!("vec2<u32>({}u, {}u)", x.to_bits(), y.to_bits())), "{}", src);
    }
    if let Some(r) = reference() {
        let g = Golden::load(&r, "msaa.coverage-4x");
        let measured = words(&g.data("default-sample-positions"));
        let ours: Vec<u32> = table.iter().flat_map(|(x, y)| [x.to_bits(), y.to_bits()]).collect();
        assert_eq!(measured, ours, "Metal's default 4-sample positions");
    } else {
        eprintln!("sample positions: the V1 golden was not compared (no reference harness)");
    }
}

const POINTS: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct P { float4 pos [[position]]; uint id [[flat]]; float ps [[point_size]]; };
vertex P v_one(uint vid [[vertex_id]]) { P o; o.pos = float4(0, 0, 0, 1); o.id = vid; o.ps = 1.0f; return o; }
vertex P v_two(uint vid [[vertex_id]]) { P o; o.pos = float4(0, 0, 0, 1); o.id = vid; o.ps = 2.0f; return o; }
vertex P v_read(uint vid [[vertex_id]], constant float *s [[buffer(0)]]) {
    P o; o.pos = float4(0, 0, 0, 1); o.id = vid; o.ps = s[vid]; return o;
}
fragment uint f_id(P in [[stage_in]]) { return in.id; }
fragment uint f_coord(P in [[stage_in]], float2 c [[point_coord]]) { return as_type<uint>(c.x); }
"#;

/// Points of the constant size 1 with one sample are WebGPU's points; other sizes, sizes read from memory, points with
/// several samples and point_coord are refused; line and triangle pipelines drop the size.
#[test]
fn point_sizes() {
    let dir = case_dir("points");
    let bc = air(&dir, POINTS, "metal3.1");
    let points = render_desc(53, 1, 1, None);
    pipeline(&bc, "v_one", "f_id", &points, &host()).unwrap();
    pipeline(&bc, "v_two", "f_id", &render_desc(53, 3, 1, None), &host()).unwrap();
    for (vs, fs, d) in [
        ("v_two", "f_id", points.clone()),
        ("v_read", "f_id", points.clone()),
        ("v_one", "f_id", render_desc(23, 1, 4, None)),
        ("v_one", "f_coord", points.clone()),
    ] {
        let e = pipeline(&bc, vs, fs, &d, &host()).unwrap_err();
        assert_eq!(item(&e), "point-size", "{} {}: {}", vs, fs, e);
    }
}

const KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(device uint *o [[buffer(0)]], uint i [[thread_index_in_threadgroup]],
              uint3 l [[thread_position_in_threadgroup]], uint3 g [[threadgroup_position_in_grid]]) {
    uint group = (g.z * 2u + g.y) * 2u + g.x;
    uint slot = group * 64u + (l.z * 4u + l.y) * 4u + l.x;
    o[slot] = i + 1u;
}
"#;

/// [[thread_index_in_threadgroup]] of a 3D threadgroup equals real Metal's (this Mac's GPU, through the WGSL
/// tests' kernel runner).
#[test]
fn linear_thread_index_of_a_3d_threadgroup() {
    let dir = case_dir("thread-index");
    let bc = air(&dir, KERNEL, "metal3.1");
    for threads in [[4u64, 2, 2], [3, 4, 4], [2, 2, 4]] {
        let desc = PipelineDesc {
            kind: PipelineKind::Compute(ComputeState {
                threads_per_threadgroup: threads,
                non_uniform_grid: false,
                threadgroup_memory_lengths: Vec::new(),
                stage_input: None,
            }),
            bindings: Vec::new(),
        };
        let c = compile(&[module(&bc, "k", 3)], &desc, &host()).unwrap_or_else(|e| panic!("{}", e));
        let out = vec![0u8; 8 * 64 * 4];
        let (w, dawn) = dispatch(
            "thread-index",
            &c,
            [2, 2, 2],
            &[Slot { stage: MetalStage::Kernel, index: 0, data: out.clone(), offset: 0 }],
            None,
        );
        // Real Metal.
        let runner = tmp_dir().join("metal_run");
        if !runner.exists() {
            let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wgsl/metal_run.swift");
            let o = Command::new("swiftc").args(["-O", "-o"]).arg(&runner).arg(&src).output().unwrap();
            assert!(o.status.success(), "swiftc: {}", String::from_utf8_lossy(&o.stderr));
        }
        let src = dir.join("k.metal");
        std::fs::write(&src, KERNEL).unwrap();
        let buf = dir.join("b0.bin");
        std::fs::write(&buf, &out).unwrap();
        let o = Command::new(&runner)
            .arg(&src)
            .arg("k")
            .arg(format!("{},{},{}", threads[0], threads[1], threads[2]))
            .arg("2,2,2")
            .arg("-")
            .arg(&buf)
            .output()
            .unwrap();
        assert!(o.status.success(), "metal_run: {}", String::from_utf8_lossy(&o.stderr));
        let metal = std::fs::read(dir.join("b0.bin.out")).unwrap();
        assert_eq!(first_difference(&w[0], &metal), None, "{:?}: thread index differs from Metal", threads);
        eprintln!("thread index {:?}: wgpu{} = Metal", threads, if dawn { " = Dawn" } else { "" });
    }
}

const ROWS: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void rows(device uint *o [[buffer(0)]], constant uint &n [[buffer(1)]], threadgroup uint *t [[threadgroup(0)]],
                 uint3 gid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    if (gid.y < n) {
        t[lid.x] = gid.x * 3u + gid.y + lid.y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[gid.y * 16u + gid.x] = t[(lid.x + 1u) % 8u];
    }
}
"#;

/// In a dimension where the variant's threadgroup size is 1, the thread position in the grid is the threadgroup's
/// position and the position in the threadgroup is 0: emitted that way, a barrier under a branch on them is uniform
/// for WGSL, and the results equal real Metal's. With a threadgroup size above 1 in that dimension the branch is not
/// uniform and the barrier is refused.
#[test]
fn thread_positions_in_dimensions_of_size_1() {
    let dir = case_dir("rows");
    let bc = air(&dir, ROWS, "metal3.1");
    let desc = |threads: [u64; 3]| PipelineDesc {
        kind: PipelineKind::Compute(ComputeState {
            threads_per_threadgroup: threads,
            non_uniform_grid: false,
            threadgroup_memory_lengths: vec![32],
            stage_input: None,
        }),
        bindings: Vec::new(),
    };
    let e = compile(&[module(&bc, "rows", 3)], &desc([8, 2, 1]), &host()).unwrap_err();
    assert_eq!(item(&e), "barrier-uniformity", "{}", e);
    let c = compile(&[module(&bc, "rows", 3)], &desc([8, 1, 1]), &host()).unwrap_or_else(|e| panic!("{}", e));
    validate(&c.wgsl[0].source);
    let out = vec![0xa5u8; 16 * 16 * 4];
    let n = 5u32.to_le_bytes().to_vec();
    let slots = [
        Slot { stage: MetalStage::Kernel, index: 0, data: out.clone(), offset: 0 },
        Slot { stage: MetalStage::Kernel, index: 1, data: n.clone(), offset: 0 },
    ];
    let (w, dawn) = dispatch("rows", &c, [2, 8, 1], &slots, None);
    let runner = tmp_dir().join("metal_run");
    if !runner.exists() {
        let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wgsl/metal_run.swift");
        let o = Command::new("swiftc").args(["-O", "-o"]).arg(&runner).arg(&src).output().unwrap();
        assert!(o.status.success(), "swiftc: {}", String::from_utf8_lossy(&o.stderr));
    }
    let src = dir.join("rows.metal");
    std::fs::write(&src, ROWS).unwrap();
    let (b0, b1) = (dir.join("b0.bin"), dir.join("b1.bin"));
    std::fs::write(&b0, &out).unwrap();
    std::fs::write(&b1, &n).unwrap();
    let o =
        Command::new(&runner).arg(&src).arg("rows").args(["8,1,1", "2,8,1", "32"]).arg(&b0).arg(&b1).output().unwrap();
    assert!(o.status.success(), "metal_run: {}", String::from_utf8_lossy(&o.stderr));
    let metal = std::fs::read(dir.join("b0.bin.out")).unwrap();
    assert_eq!(first_difference(&w[0], &metal), None, "rows differ from Metal");
    eprintln!("thread positions in dimensions of size 1: wgpu{} = Metal", if dawn { " = Dawn" } else { "" });
}
