//! The framebuffer-fetch cases.

use super::harness::*;
use super::stage::{Golden, air, case_dir, color, host, module, reference, render_desc, v1_air, words};
use airwgsl::api::*;
use airwgsl::wgsl::tracked;
use std::path::Path;
use std::process::Command;

fn note(case: &str, dawn: bool, what: &str) {
    if dawn {
        eprintln!("{}: wgpu = Dawn = {}", case, what);
    } else {
        eprintln!("{}: wgpu = {} (Dawn not run: AW_DAWN_NODE unset)", case, what);
    }
}

fn item(r: &Refusal) -> String {
    tracked::item_of(r).map(|(_, id)| id).unwrap_or_default()
}

fn pipeline(vbc: &[u8], vs: &str, fbc: &[u8], fs: &str, desc: &PipelineDesc) -> Result<Compiled, Refusal> {
    compile(&[module(vbc, vs, 1), module(fbc, fs, 2)], desc, &host())
}

fn has_prover(c: &Compiled) -> bool {
    matches!(&c.passes[..], [PassPlan::PerPrimitive { overlap: Some(_), .. }])
}

/// `RVertexAt` of the V1 harness (`RenderSupport.m`): a 48-byte `RVertex`.
fn rvertex(px: f64, py: f64, w: u32, h: u32, id: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for f in [(px / w as f64 * 2.0 - 1.0) as f32, (1.0 - py / h as f64 * 2.0) as f32, 0.0, 1.0] {
        out.extend_from_slice(&f.to_le_bytes());
    }
    for u in [id, 0, 0, 0] {
        out.extend_from_slice(&u.to_le_bytes());
    }
    for f in [1.0f32, 0.0, 0.0, 0.0] {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// V1 `render.framebuffer-fetch`: three overlapping triangles in one draw and a fourth in a second draw, each
/// fragment computing `previous * 3 + id` on R32Uint; Metal's result is 58 everywhere.
#[test]
fn v1_render_framebuffer_fetch() {
    let Some(r) = reference() else {
        eprintln!("render.framebuffer-fetch: not run: the V1 reference harness is not at AW_V1_REFERENCE");
        return;
    };
    let g = Golden::load(&r, "render.framebuffer-fetch");
    assert_eq!(g.rule("result"), "exact");
    let c = pipeline(
        &v1_air(&r, "render"),
        "rnd_vertex_triangles",
        &v1_air(&r, "render_pass"),
        "rnd_fragment_fetch",
        &render_desc(R32UINT.metal, 3, 1, None),
    )
    .unwrap();
    assert!(matches!(&c.passes[..], [PassPlan::PerPrimitive { fetched, .. }] if fetched == &[0]), "{:?}", c.passes);
    let mut v = Vec::new();
    for t in 0..4 {
        v.extend(rvertex(-1.0, -1.0, 8, 8, t + 1));
        v.extend(rvertex(24.0, -1.0, 8, 8, t + 1));
        v.extend(rvertex(-1.0, 24.0, 8, 8, t + 1));
    }
    let scene = Scene {
        compiled: c,
        format: R32UINT,
        width: 8,
        height: 8,
        samples: 1,
        initial: None,
        clear: [0.0; 4],
        blend: false,
        write: true,
        primitive: 3,
        draws: vec![
            SceneDraw { start: 0, count: 9, instances: 1, vertices: v.clone() },
            SceneDraw { start: 9, count: 3, instances: 1, vertices: v },
        ],
        fragment_buffer: None,
    };
    let (out, dawn) = render("v1-framebuffer-fetch", &scene);
    assert_eq!(words(&out.target), words(&g.data("result")), "replay and Metal differ");
    note("render.framebuffer-fetch", dawn, "Metal golden");
}

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct FV { float4 position; uint4 data; };
struct FOut { float4 position [[position]]; uint id [[flat]]; };
vertex FOut fvs(const device FV *v [[buffer(0)]], uint vid [[vertex_id]], uint iid [[instance_id]]) {
    FOut o;
    o.position = v[vid].position;
    o.id = v[vid].data.x + iid * 16u;
    return o;
}
vertex FOut fvs_scaled(const device FV *v [[buffer(0)]], uint vid [[vertex_id]]) {
    FOut o;
    float4 p = v[vid].position;
    o.position = float4(p.xy * dot(p.zw, p.zw) * 0.5f, 0.0f, 1.0f);
    o.id = v[vid].data.x;
    return o;
}
fragment uint ffs_order(FOut in [[stage_in]], uint previous [[color(0)]]) {
    return previous * 3u + in.id;
}
fragment float4 ffs_blend(FOut in [[stage_in]], float4 c [[color(0)]]) {
    return c;
}
fragment ushort ffs_order16(FOut in [[stage_in]], ushort previous [[color(0)]]) {
    return previous * 3 + ushort(in.id);
}
fragment ushort ffs_order16_position(FOut in [[stage_in]], ushort previous [[color(0)]]) {
    return previous * 3 + ushort(in.id) + ushort(fract(in.position.x) * 8.0f) * 16 + ushort(fract(in.position.y) * 8.0f);
}
fragment half4 ffs_half(FOut in [[stage_in]], half4 c [[color(0)]]) {
    return c * 2.0h + half4(half(in.id));
}
"#;

/// An `FV` vertex at pixel `(px, py)` of a `w` x `h` target.
fn fv(px: f64, py: f64, w: u32, h: u32, id: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for f in [(px / w as f64 * 2.0 - 1.0) as f32, (1.0 - py / h as f64 * 2.0) as f32, 0.0, 1.0] {
        out.extend_from_slice(&f.to_le_bytes());
    }
    for u in [id, 0, 0, 0] {
        out.extend_from_slice(&u.to_le_bytes());
    }
    out
}

fn tri(v: &mut Vec<u8>, p: [(f64, f64); 3], w: u32, h: u32, id: u32) {
    for (x, y) in p {
        v.extend(fv(x, y, w, h, id));
    }
}

/// Compile `MSL` for the target and the given functions and state.
fn compile_msl(dir: &Path, vs: &str, fs: &str, desc: &PipelineDesc) -> Result<Compiled, Refusal> {
    let bc = air(dir, MSL, "ios-metal2.4");
    pipeline(&bc, vs, &bc, fs, desc)
}

/// Run on wgpu (both plans) and Dawn, and on Metal; all must agree.
fn differential(name: &str, s: &Scene, vs: &str, fs: &str) {
    let (out, dawn) = render(name, s);
    let m = run_metal(&case_dir(&format!("{}-metal", name)), s, MSL, vs, fs).expect("Metal accepts the pipeline");
    assert_eq!(out.target, m.target, "{}: replay and Metal differ", name);
    note(name, dawn, "Metal");
}

/// Overlapping and disjoint triangles in one instanced draw: the result depends on primitive order where they overlap,
/// and the prover groups the disjoint ones.
#[test]
fn overlapping_and_disjoint_triangles() {
    let dir = case_dir("overlap-air");
    let c = compile_msl(&dir, "fvs", "ffs_order", &render_desc(R32UINT.metal, 3, 1, None)).unwrap();
    assert!(has_prover(&c));
    let (w, h) = (32, 32);
    let mut v = Vec::new();
    tri(&mut v, [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)], w, h, 1);
    tri(&mut v, [(4.0, 4.0), (24.0, 4.0), (4.0, 24.0)], w, h, 2);
    tri(&mut v, [(26.0, 1.0), (31.0, 1.0), (26.0, 6.0)], w, h, 3);
    tri(&mut v, [(25.0, 25.0), (31.0, 25.0), (25.0, 31.0)], w, h, 4);
    tri(&mut v, [(2.0, 2.0), (30.0, 3.0), (3.0, 30.0)], w, h, 5);
    tri(&mut v, [(1.0, 26.0), (6.0, 26.0), (1.0, 31.0)], w, h, 6);
    let s = Scene {
        compiled: c,
        format: R32UINT,
        width: w,
        height: h,
        samples: 1,
        initial: None,
        clear: [0.0; 4],
        blend: false,
        write: true,
        primitive: 3,
        draws: vec![SceneDraw { start: 0, count: 18, instances: 2, vertices: v }],
        fragment_buffer: None,
    };
    let base = steps(&s, false).len();
    let grouped = steps(&s, true).len();
    assert_eq!(base, 12);
    assert!(grouped < base, "the prover grouped nothing: {} groups", grouped);
    eprintln!("overlapping-and-disjoint: {} primitives in {} groups", base, grouped);
    differential("overlapping-and-disjoint", &s, "fvs", "ffs_order");
}

/// A triangle strip whose consecutive triangles share edges and whose later triangles overlap earlier ones.
#[test]
fn triangle_strip() {
    let dir = case_dir("strip-air");
    let c = compile_msl(&dir, "fvs", "ffs_order", &render_desc(R32UINT.metal, 3, 1, None)).unwrap();
    let (w, h) = (24, 24);
    let mut v = Vec::new();
    let pts = [(0.0, 0.0), (0.0, 20.0), (12.0, 2.0), (10.0, 22.0), (23.0, 0.0), (2.0, 23.0), (20.0, 23.0)];
    for (k, (x, y)) in pts.iter().enumerate() {
        v.extend(fv(*x, *y, w, h, k as u32 + 1));
    }
    let s = Scene {
        compiled: c,
        format: R32UINT,
        width: w,
        height: h,
        samples: 1,
        initial: None,
        clear: [0.0; 4],
        blend: false,
        write: true,
        primitive: 4,
        draws: vec![SceneDraw { start: 0, count: 7, instances: 1, vertices: v }],
        fragment_buffer: None,
    };
    differential("triangle-strip", &s, "fvs", "ffs_order");
}

/// One-plus-one blending of the fetched value onto the attachment: overlapping triangles double the initial texels
/// once or twice, saturating.
#[test]
fn blending_with_fetched_values() {
    let dir = case_dir("blend-air");
    let mut desc = render_desc(RGBA8UNORM.metal, 3, 1, None);
    if let PipelineKind::Render(r) = &mut desc.kind {
        let a = &mut r.color_attachments[0];
        a.blending_enabled = true;
        a.rgb_blend_operation = 0;
        a.alpha_blend_operation = 0;
        (a.source_rgb_blend_factor, a.destination_rgb_blend_factor) = (1, 1);
        (a.source_alpha_blend_factor, a.destination_alpha_blend_factor) = (1, 1);
    }
    let c = compile_msl(&dir, "fvs", "ffs_blend", &desc).unwrap();
    let (w, h) = (16, 16);
    let mut v = Vec::new();
    tri(&mut v, [(-1.0, -1.0), (30.0, -1.0), (-1.0, 30.0)], w, h, 1);
    tri(&mut v, [(3.0, 3.0), (16.0, 3.0), (3.0, 16.0)], w, h, 2);
    let initial: Vec<u8> = (0..w * h * 4).map(|i| ((i * 37 + 11) & 127) as u8).collect();
    let s = Scene {
        compiled: c,
        format: RGBA8UNORM,
        width: w,
        height: h,
        samples: 1,
        initial: Some(initial),
        clear: [0.0; 4],
        blend: true,
        write: true,
        primitive: 3,
        draws: vec![SceneDraw { start: 0, count: 6, instances: 1, vertices: v }],
        fragment_buffer: None,
    };
    differential("blending-with-fetched", &s, "fvs", "ffs_blend");
}

/// `half4` color input of an RGBA16Float attachment with half arithmetic on the fetched value.
#[test]
fn half_arithmetic_on_rgba16float() {
    let dir = case_dir("half-air");
    let c = compile_msl(&dir, "fvs", "ffs_half", &render_desc(RGBA16FLOAT.metal, 3, 1, None)).unwrap();
    let (w, h) = (16, 16);
    let mut v = Vec::new();
    tri(&mut v, [(-1.0, -1.0), (30.0, -1.0), (-1.0, 30.0)], w, h, 1);
    tri(&mut v, [(2.0, 2.0), (14.0, 5.0), (5.0, 14.0)], w, h, 3);
    // Small integers: every result is exact in binary16.
    let initial: Vec<u8> = (0..w * h * 4).flat_map(|i| half_bits((i % 7) as f32).to_le_bytes()).collect();
    let s = Scene {
        compiled: c,
        format: RGBA16FLOAT,
        width: w,
        height: h,
        samples: 1,
        initial: Some(initial),
        clear: [0.0; 4],
        blend: false,
        write: true,
        primitive: 3,
        draws: vec![SceneDraw { start: 0, count: 6, instances: 1, vertices: v }],
        fragment_buffer: None,
    };
    differential("half-on-rgba16float", &s, "fvs", "ffs_half");
}

/// binary16 bits of a small integer.
fn half_bits(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let e = x.log2().floor() as i32;
    let m = ((x / 2f32.powi(e) - 1.0) * 1024.0) as u16;
    (((e + 15) as u16) << 10) | m
}

/// A 4-sample R16Uint attachment (WebGPU's 32-bit integer formats are not multisampled): triangles whose edges cross
/// pixels, so that samples of a pixel differ, each fetching its own sample; a second function also reads the
/// fragment's position, which Metal gives at the pixel centre.
#[test]
fn multisampled_per_sample_fetch() {
    let dir = case_dir("msaa-air");
    let c = compile_msl(&dir, "fvs", "ffs_order16", &render_desc(R16UINT.metal, 3, 4, None)).unwrap();
    assert!(c.wgsl[1].source.contains("sample_index"), "{}", c.wgsl[1].source);
    let (w, h) = (8, 8);
    let mut v = Vec::new();
    tri(&mut v, [(0.3, 0.2), (7.6, 1.4), (1.1, 7.7)], w, h, 1);
    tri(&mut v, [(0.0, 8.0), (8.0, 0.4), (7.9, 7.9)], w, h, 2);
    tri(&mut v, [(-1.0, -1.0), (20.0, -1.0), (-1.0, 20.0)], w, h, 3);
    let s = Scene {
        compiled: c,
        format: R16UINT,
        width: w,
        height: h,
        samples: 4,
        initial: None,
        clear: [0.0; 4],
        blend: false,
        write: true,
        primitive: 3,
        draws: vec![SceneDraw { start: 0, count: 9, instances: 1, vertices: v }],
        fragment_buffer: None,
    };
    differential("msaa-per-sample", &s, "fvs", "ffs_order16");
    let c = compile_msl(&dir, "fvs", "ffs_order16_position", &render_desc(R16UINT.metal, 3, 4, None)).unwrap();
    let s = Scene { compiled: c, ..s };
    differential("msaa-per-sample-position", &s, "fvs", "ffs_order16_position");
}

/// Random small triangles, many overlapping: the prover's groups equal the baseline on wgpu and Dawn, and both equal
/// Metal.
#[test]
fn random_triangles_grouped_equal_baseline() {
    let dir = case_dir("random-air");
    let c = compile_msl(&dir, "fvs", "ffs_order", &render_desc(R32UINT.metal, 3, 1, None)).unwrap();
    let (w, h) = (48, 48);
    let mut state = 0x2545f491u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut v = Vec::new();
    for id in 0..60u32 {
        let (x, y) = ((next() % 44) as f64 + 0.37, (next() % 44) as f64 + 0.61);
        let s = 2.0 + (next() % 6) as f64;
        tri(&mut v, [(x, y), (x + s, y + 0.5), (x + 0.25, y + s)], w, h, id + 1);
    }
    let s = Scene {
        compiled: c,
        format: R32UINT,
        width: w,
        height: h,
        samples: 1,
        initial: None,
        clear: [0.0; 4],
        blend: false,
        write: true,
        primitive: 3,
        draws: vec![SceneDraw { start: 0, count: 180, instances: 1, vertices: v }],
        fragment_buffer: None,
    };
    let groups = steps(&s, true).len();
    eprintln!("random-triangles: 60 primitives in {} groups", groups);
    assert!(groups < 60);
    differential("random-triangles", &s, "fvs", "ffs_order");
}

/// The prover needs a position the GPU computes exactly: a fast-math `dot` (native WGSL arithmetic) disqualifies the
/// vertex function; the same function compiled without fast math qualifies (the soft-float lowering is exact).
#[test]
fn prover_needs_exactly_lowered_positions() {
    let dir = case_dir("prover-air");
    let desc = render_desc(R32UINT.metal, 3, 1, None);
    let c = compile_msl(&dir, "fvs_scaled", "ffs_order", &desc).unwrap();
    assert!(!has_prover(&c), "a fast-math position computation got a prover");
    let src = dir.join("precise.metal");
    std::fs::write(&src, MSL).unwrap();
    let out = dir.join("precise.air");
    let o = Command::new("xcrun")
        .args(["-sdk", "iphoneos", "metal", "-std=ios-metal2.4", "-fno-fast-math", "-target", "air64-apple-ios17.7"])
        .arg("-c")
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let bc = std::fs::read(&out).unwrap();
    let c = pipeline(&bc, "fvs_scaled", &bc, "ffs_order", &desc).unwrap();
    assert!(has_prover(&c), "a precise position computation got no prover");
    let (w, h) = (16, 16);
    let mut v = Vec::new();
    for k in 0..4 {
        let o = k as f64 * 4.0;
        tri(&mut v, [(16.0 + o, 16.0), (19.0 + o, 16.0), (16.0 + o, 19.0)], w, h, k + 1);
    }
    tri(&mut v, [(12.0, 12.0), (31.0, 12.0), (12.0, 31.0)], w, h, 9);
    let s = Scene {
        compiled: c,
        format: R32UINT,
        width: w,
        height: h,
        samples: 1,
        initial: None,
        clear: [0.0; 4],
        blend: false,
        write: true,
        primitive: 3,
        draws: vec![SceneDraw { start: 0, count: 15, instances: 1, vertices: v }],
        fragment_buffer: None,
    };
    let (out, dawn) = render("prover-precise", &s);
    let m = run_metal(&case_dir("prover-precise-metal"), &s, MSL, "fvs_scaled", "ffs_order").unwrap();
    assert_eq!(out.target, m.target);
    note("prover-precise", dawn, "Metal");
}

/// The refusals: a type class Metal refuses, a raster order group (the A10X reports none, and its driver's answer is
/// its compiler's), `[[sample_mask]]` with a multisampled fetch.
#[test]
fn refusals() {
    let dir = case_dir("refusals");
    let src = r#"
#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; };
vertex V rv(uint vid [[vertex_id]]) { V v; v.position = float4(0, 0, 0, 1); return v; }
fragment void rclass(V in [[stage_in]], int c [[color(0)]], device int *o [[buffer(0)]]) { o[0] = c; }
fragment uint rrog(V in [[stage_in]], uint c [[color(0), raster_order_group(0)]]) { return c + 1u; }
fragment uint rmask(V in [[stage_in]], uint c [[color(0)]], uint m [[sample_mask]]) { return c + m; }
fragment uint runused(V in [[stage_in]], uint c [[color(0)]]) { return 7u; }
"#;
    let bc = air(&dir, src, "ios-metal2.4");
    let r = pipeline(&bc, "rv", &bc, "rclass", &render_desc(R32UINT.metal, 3, 1, None)).unwrap_err();
    assert_eq!((item(&r), r.kind), ("fetch-format-class".to_string(), RefusalKind::RejectedByDevice), "{}", r);
    let r = pipeline(&bc, "rv", &bc, "rrog", &render_desc(R32UINT.metal, 3, 1, None)).unwrap_err();
    assert_eq!((item(&r), r.kind), ("raster-order-group".to_string(), RefusalKind::NotImplemented), "{}", r);
    let r = pipeline(&bc, "rv", &bc, "rmask", &render_desc(R32UINT.metal, 3, 4, None)).unwrap_err();
    assert_eq!(item(&r), "fetch-sample-mask", "{}", r);
    // One sample: the mask is the pixel's coverage either way.
    pipeline(&bc, "rv", &bc, "rmask", &render_desc(R32UINT.metal, 3, 1, None)).unwrap();
    // An input the function does not read takes no snapshot and no per-primitive plan.
    let c = pipeline(&bc, "rv", &bc, "runused", &render_desc(R32UINT.metal, 3, 1, None)).unwrap();
    assert_eq!(c.passes, vec![PassPlan::Direct]);
    // Metal refuses the class mismatch too.
    let s = Scene {
        compiled: c,
        format: R32UINT,
        width: 4,
        height: 4,
        samples: 1,
        initial: None,
        clear: [0.0; 4],
        blend: false,
        write: false,
        primitive: 3,
        draws: Vec::new(),
        fragment_buffer: None,
    };
    let refused = run_metal(&dir, &s, src, "rv", "rclass").unwrap_err();
    assert!(refused.contains("not compatible"), "{}", refused);
    let _ = color(0);
}

/// Every type a color input can have read from 15 formats, against Metal on this Mac: the fetched lanes go to a
/// fragment buffer. A type of another class must be refused as Metal refuses it; the others read Metal's values bit
/// for bit, except that sRGB conversion is held to MSL 8.7.7's bound (the value converted back lies within 0.5 of the
/// code) on both.
#[test]
fn input_types_against_formats() {
    // (WebGPU format, MTLPixelFormat, bytes per texel, class).
    let formats: &[(&str, u64, u32, char)] = &[
        ("r32uint", 53, 4, 'u'),
        ("rgba8uint", 73, 4, 'u'),
        ("r16uint", 23, 2, 'u'),
        ("rgba32uint", 123, 16, 'u'),
        ("rg16sint", 64, 4, 's'),
        ("r32sint", 54, 4, 's'),
        ("rgba8sint", 74, 4, 's'),
        ("rgba16sint", 114, 8, 's'),
        ("rgba32sint", 124, 16, 's'),
        ("r8unorm", 10, 1, 'f'),
        ("rgba8unorm", 70, 4, 'f'),
        ("bgra8unorm", 80, 4, 'f'),
        ("rgba16float", 115, 8, 'f'),
        ("r32float", 55, 4, 'f'),
        ("rgba8unorm-srgb", 71, 4, 'f'),
    ];
    let types: &[(&str, char)] = &[
        ("uint", 'u'),
        ("uint2", 'u'),
        ("uint4", 'u'),
        ("ushort", 'u'),
        ("ushort4", 'u'),
        ("int", 's'),
        ("int4", 's'),
        ("short2", 's'),
        ("short4", 's'),
        ("float", 'f'),
        ("float2", 'f'),
        ("float4", 'f'),
        ("half", 'f'),
        ("half4", 'f'),
    ];
    let (w, h) = (16u32, 16u32);
    let mut dawn_ran = false;
    let mut checked = 0;
    for &(wf, mf, bpp, class) in formats {
        let mut initial: Vec<u8> = (0..w * h * bpp).map(|i| ((i * 37 + 11) & 255) as u8).collect();
        match wf {
            // Finite values without binary32 subnormals.
            "r32float" => initial.chunks_mut(4).for_each(|c| c[3] = (c[3] & 0x3f) | 0x20),
            "rgba16float" => initial.chunks_mut(2).for_each(|c| c[1] &= 0x3f),
            _ => {}
        }
        // Each format with the types of its class and the first type of another class.
        let mut chosen: Vec<&(&str, char)> = types.iter().filter(|t| t.1 == class).collect();
        chosen.push(types.iter().find(|t| t.1 != class).unwrap());
        for &&(ty, tclass) in &chosen {
            let n: usize = ty.chars().last().and_then(|c| c.to_digit(10)).map_or(1, |d| d as usize);
            let scalar = ty.trim_end_matches(|c: char| c.is_ascii_digit());
            let mut body = String::new();
            for k in 0..n {
                let lane = if n == 1 { "c".to_string() } else { format!("c[{}]", k) };
                let conv = match scalar {
                    "float" | "int" => format!("as_type<uint>({})", lane),
                    "half" => format!("uint(as_type<ushort>({}))", lane),
                    "short" => format!("uint(as_type<ushort>({}))", lane),
                    _ => format!("uint({})", lane),
                };
                body.push_str(&format!("    o[p * 4 + {}] = {};\n", k, conv));
            }
            let src = format!(
                "#include <metal_stdlib>\nusing namespace metal;\nstruct V {{ float4 position [[position]]; }};\n\
                 vertex V vs(uint vid [[vertex_id]]) {{\n    float2 q = float2(vid == 1 ? 3.0 : -1.0, vid == 2 ? -3.0 : 1.0);\n    V v; v.position = float4(q, 0, 1); return v;\n}}\n\
                 fragment void fs(V in [[stage_in]], {} c [[color(0)]], device uint *o [[buffer(0)]]) {{\n    uint p = uint(in.position.y) * {} + uint(in.position.x);\n{}}}\n",
                ty, w, body
            );
            let name = format!("types-{}-{}", wf, ty);
            let dir = case_dir(&name);
            let bc = air(&dir, &src, "ios-metal2.4");
            let compiled = pipeline(&bc, "vs", &bc, "fs", &render_desc(mf, 3, 1, None));
            let s = |c: Compiled| Scene {
                compiled: c,
                format: Format { webgpu: wf, metal: mf, bpp },
                width: w,
                height: h,
                samples: 1,
                initial: Some(initial.clone()),
                clear: [0.0; 4],
                blend: false,
                write: false,
                primitive: 3,
                draws: vec![SceneDraw { start: 0, count: 3, instances: 1, vertices: vec![0; 4] }],
                fragment_buffer: Some(vec![0xab; (w * h * 16) as usize]),
            };
            let metal = |c: Compiled| run_metal(&dir, &s(c), &src, "vs", "fs");
            match compiled {
                Err(r) => {
                    assert_ne!(tclass, class, "{}: refused: {}", name, r);
                    assert_eq!(item(&r), "fetch-format-class", "{}: {}", name, r);
                    // Metal refuses it too (Metal compiles the source; the scene's pipeline is not used).
                    let dummy = Compiled {
                        wgsl: Vec::new(),
                        layout: BindingLayout::default(),
                        passes: vec![PassPlan::Direct],
                        reflection: Reflection::default(),
                        requirements: HostRequirements::default(),
                    };
                    let refused = metal(dummy).expect_err("Metal accepts what the compiler refuses");
                    assert!(refused.contains("not compatible"), "{}: {}", name, refused);
                }
                Ok(c) => {
                    assert_eq!(tclass, class, "{}: compiled a type Metal refuses", name);
                    let scene = s(c.clone());
                    let (out, dawn) = render(&name, &scene);
                    dawn_ran |= dawn;
                    let m = metal(c).expect("Metal accepts the pipeline");
                    let (got, want) = (words(out.buffer.as_ref().unwrap()), words(m.buffer.as_ref().unwrap()));
                    if wf == "rgba8unorm-srgb" {
                        for (k, (g, mw)) in got.iter().zip(&want).enumerate() {
                            let (pixel, lane) = (k / 4, k % 4);
                            // Lanes beyond the input and the alpha lane (stored linearly) are exact.
                            if lane >= n || lane == 3 {
                                assert_eq!(g, mw, "{} word {}", name, k);
                                continue;
                            }
                            let code = initial[pixel * 4 + lane] as f64;
                            for v in [g, mw] {
                                let x =
                                    if scalar == "half" { half_value(*v as u16) } else { f32::from_bits(*v) as f64 };
                                let back = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
                                assert!(
                                    (back * 255.0 - code).abs() <= 0.5,
                                    "{} lane {}: {} from code {}",
                                    name,
                                    k,
                                    x,
                                    code
                                );
                            }
                        }
                    } else if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
                        panic!("{}: word {}: replay {:08x}, Metal {:08x}", name, i, got[i], want[i]);
                    }
                    checked += 1;
                }
            }
        }
    }
    eprintln!(
        "input types: {} format/type pairs equal Metal on wgpu{}",
        checked,
        if dawn_ran { " and Dawn" } else { " (Dawn not run)" }
    );
}

fn half_value(b: u16) -> f64 {
    let e = ((b >> 10) & 0x1f) as i32;
    let m = (b & 0x3ff) as f64;
    let v = if e == 0 { m / 1024.0 * 2f64.powi(-14) } else { (1.0 + m / 1024.0) * 2f64.powi(e - 15) };
    if b & 0x8000 != 0 { -v } else { v }
}
