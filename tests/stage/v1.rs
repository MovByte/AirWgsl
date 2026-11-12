//! The stage-interface cases of the V1 reference harness (WebMetal `tests/reference`), replayed with the compiled
//! pipelines on wgpu and Dawn and compared with the goldens recorded on real Metal under FORMAT.md's rules: records
//! whose rule is `exact` must be equal; records whose rule is `none` are compared too where the behavior was measured
//! (and say so), and reported otherwise.
//!
//! Each case builds the same pipeline state, buffers and draws as its Objective-C source in `src/cases/` and reads the
//! same region of the target.

use super::harness::*;
use airwgsl::api::*;
use airwgsl::stage::fetch::{FORMATS, Kind};
use airwgsl::wgsl::tracked;

const RGBA32UINT: u64 = 123;
const R32UINT: u64 = 53;
const R16UINT: u64 = 23;

fn skip(case: &str) -> Option<std::path::PathBuf> {
    let r = reference();
    if r.is_none() {
        eprintln!("{}: not run: the V1 reference harness is not at AW_V1_REFERENCE or next to this crate", case);
    }
    r
}

/// `RVertex` of shaders/render.metal: position, data, extra (48 bytes).
#[derive(Clone, Copy)]
struct RVertex {
    position: [f32; 4],
    data: [u32; 4],
    extra: [f32; 4],
}

/// RVertexAt of RenderSupport.m.
fn vertex_at(px: f64, py: f64, z: f64, w: u32, h: u32, id: u32) -> RVertex {
    RVertex {
        position: [(px / w as f64 * 2.0 - 1.0) as f32, (1.0 - py / h as f64 * 2.0) as f32, z as f32, 1.0],
        data: [id, 0, 0, 0],
        extra: [1.0, 0.0, 0.0, 0.0],
    }
}

fn vertex_bytes(v: &[RVertex]) -> Vec<u8> {
    let mut out = Vec::new();
    for x in v {
        for f in x.position {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for u in x.data {
            out.extend_from_slice(&u.to_le_bytes());
        }
        for f in x.extra {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    out
}

fn u32_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn pipeline(bc: &[u8], vs: &str, fs: &str, desc: &PipelineDesc) -> Result<Compiled, Refusal> {
    compile(&[module(bc, vs, 1), module(bc, fs, 2)], desc, &host())
}

fn vslot(index: u64, data: Vec<u8>, offset: u64) -> Slot {
    Slot { stage: MetalStage::Vertex, index, data, offset }
}

fn expect_equal(case: &str, record: &str, got: &[u8], want: &[u8]) {
    if let Some(i) = first_difference(got, want) {
        let (g, w) = (words(got), words(want));
        panic!(
            "{} {}: word {} differs: replay {:08x?}, Metal {:08x?}",
            case,
            record,
            i,
            &g[i..(i + 4).min(g.len())],
            &w[i..(i + 4).min(w.len())]
        );
    }
}

fn note_dawn(case: &str, dawn: bool) {
    if dawn {
        eprintln!("{}: wgpu = Dawn = Metal", case);
    } else {
        eprintln!("{}: wgpu = Metal (Dawn not run: AW_DAWN_NODE unset)", case);
    }
}

fn item(r: &Refusal) -> String {
    tracked::item_of(r).map(|(_, id)| id).unwrap_or_default()
}

fn fetch_desc(format: u64) -> PipelineDesc {
    render_desc(
        RGBA32UINT,
        0,
        1,
        Some(VertexDescriptor {
            layouts: vec![
                VertexBufferLayout { stride: 0, step_function: 1, step_rate: 1 },
                VertexBufferLayout { stride: 16, step_function: 1, step_rate: 1 },
            ],
            attributes: vec![VertexAttribute { format, offset: 0, buffer_index: 1 }],
        }),
    )
}

/// `vertex.formats`: every format fetched as uint4, int4 or float4; the 16 fetched values are row 0 of the target.
#[test]
fn vertex_formats() {
    let Some(r) = skip("vertex.formats") else { return };
    let g = Golden::load(&r, "vertex.formats");
    let bc = v1_air(&r, "render");
    let mut dawn = false;
    let mut checked = 0;
    for f in FORMATS.iter().filter(|f| f.kind != Kind::PackedFloat) {
        let input = g.data(&format!("{}.input", f.name));
        let mut fetches = Vec::new();
        match f.kind {
            Kind::UInt => fetches.push("uint"),
            Kind::SInt => fetches.push("int"),
            _ => {}
        }
        if matches!(f.kind, Kind::UInt | Kind::SInt) && f.bits <= 16 || !matches!(f.kind, Kind::UInt | Kind::SInt) {
            fetches.push("float");
        }
        for fetch in fetches {
            let c = pipeline(&bc, &format!("rnd_vertex_fetch_{}", fetch), "rnd_fragment_id_data", &fetch_desc(f.value))
                .unwrap_or_else(|e| panic!("{} as {}: {}", f.name, fetch, e));
            let scene = Scene {
                compiled: c,
                format: "rgba32uint",
                width: 16,
                height: 16,
                samples: 1,
                clear: [0.0; 4],
                depth: None,
                topology: "point-list",
                draws: vec![Draw::new(0, 16, vec![vslot(1, input.clone(), 0)])],
            };
            let (out, d) = render(&format!("vertex-formats-{}-{}", f.name, fetch), &scene);
            dawn = d;
            let name = format!("{}.as-{}4", f.name, fetch);
            expect_equal("vertex.formats", &name, &out.color[..256], &g.data(&name));
            checked += 1;
        }
    }
    assert_eq!(checked, 67);
    note_dawn("vertex.formats", dawn);
}

/// `vertex.formats-packed-float`: FloatRG11B10 and FloatRGB9E5 are Apple5 formats; the A10X (Apple3) refuses them.
#[test]
fn vertex_formats_packed_float() {
    let Some(r) = skip("vertex.formats-packed-float") else { return };
    let bc = v1_air(&r, "render");
    for f in [54, 55] {
        let e = pipeline(&bc, "rnd_vertex_fetch_float", "rnd_fragment_id_data", &fetch_desc(f)).unwrap_err();
        assert_eq!((e.kind, item(&e).as_str()), (RefusalKind::RejectedByDevice, "vertex-descriptor"), "{}", e);
    }
}

fn two_buffer_desc(step: u64, rate: u64) -> PipelineDesc {
    render_desc(
        RGBA32UINT,
        0,
        1,
        Some(VertexDescriptor {
            layouts: vec![
                VertexBufferLayout { stride: 0, step_function: 1, step_rate: 1 },
                VertexBufferLayout { stride: 8, step_function: 1, step_rate: 1 },
                VertexBufferLayout { stride: 8, step_function: step, step_rate: rate },
            ],
            attributes: vec![
                VertexAttribute { format: 37, offset: 0, buffer_index: 1 },
                VertexAttribute { format: 37, offset: 0, buffer_index: 2 },
            ],
        }),
    )
}

/// `vertex.step-functions`: per-instance with step rate 2 and constant.
#[test]
fn vertex_step_functions() {
    let Some(r) = skip("vertex.step-functions") else { return };
    let g = Golden::load(&r, "vertex.step-functions");
    let bc = v1_air(&r, "render");
    let per_vertex: Vec<u32> = (0..16).flat_map(|i| [100 + i, 200 + i]).collect();
    let per_instance: Vec<u32> = (0..16).flat_map(|i| [300 + i, 400 + i]).collect();
    let mut dawn = false;
    for (name, step, rate) in [("per-instance-rate-2", 2, 2), ("constant", 0, 0)] {
        let c = pipeline(&bc, "rnd_vertex_fetch_two", "rnd_fragment_id_data", &two_buffer_desc(step, rate)).unwrap();
        let mut draw =
            Draw::new(0, 4, vec![vslot(1, u32_bytes(&per_vertex), 0), vslot(2, u32_bytes(&per_instance), 0)]);
        draw.instance_count = 5;
        let scene = Scene {
            compiled: c,
            format: "rgba32uint",
            width: 16,
            height: 16,
            samples: 1,
            clear: [0.0; 4],
            depth: None,
            topology: "point-list",
            draws: vec![draw],
        };
        let (out, d) = render(&format!("step-{}", name), &scene);
        dawn = d;
        expect_equal("vertex.step-functions", name, &region(&out.color, 16, 16, 0, 0, 4, 5), &g.data(name));
    }
    note_dawn("vertex.step-functions", dawn);
}

/// `vertex.step-base-instance`: the per-instance fetch index with a nonzero base instance. The golden's rule is
/// `none` (Apple's documentation does not state it); the fetch index is the one measured here and with base instance
/// 3 and step rate 2 again (src/stage/fetch.rs), so the replay must equal it.
#[test]
fn vertex_step_base_instance() {
    let Some(r) = skip("vertex.step-base-instance") else { return };
    let g = Golden::load(&r, "vertex.step-base-instance");
    let bc = v1_air(&r, "render");
    let per_vertex: Vec<u32> = (0..4).flat_map(|i| [i, i]).collect();
    let per_instance: Vec<u32> = (0..16).flat_map(|i| [300 + i, 300 + i]).collect();
    let c = pipeline(&bc, "rnd_vertex_fetch_two", "rnd_fragment_id_data", &two_buffer_desc(2, 2)).unwrap();
    let mut draw = Draw::new(0, 2, vec![vslot(1, u32_bytes(&per_vertex), 0), vslot(2, u32_bytes(&per_instance), 0)]);
    draw.instance_count = 4;
    draw.base_instance = 3;
    let scene = Scene {
        compiled: c,
        format: "rgba32uint",
        width: 16,
        height: 16,
        samples: 1,
        clear: [0.0; 4],
        depth: None,
        topology: "point-list",
        draws: vec![draw],
    };
    let (out, dawn) = render("step-base-instance", &scene);
    assert_eq!(g.rule("fetched"), "none");
    expect_equal("vertex.step-base-instance", "fetched", &region(&out.color, 16, 16, 0, 3, 2, 4), &g.data("fetched"));
    note_dawn("vertex.step-base-instance", dawn);
}

/// `vertex.bytes-and-buffer-offset`: inline bytes, then a buffer at offsets 0, 40 and 64.
#[test]
fn vertex_bytes_and_buffer_offset() {
    let Some(r) = skip("vertex.bytes-and-buffer-offset") else { return };
    let g = Golden::load(&r, "vertex.bytes-and-buffer-offset");
    let bc = v1_air(&r, "render");
    let desc = render_desc(
        RGBA32UINT,
        0,
        1,
        Some(VertexDescriptor {
            layouts: vec![
                VertexBufferLayout { stride: 0, step_function: 1, step_rate: 1 },
                VertexBufferLayout { stride: 8, step_function: 1, step_rate: 1 },
            ],
            attributes: vec![VertexAttribute { format: 37, offset: 0, buffer_index: 1 }],
        }),
    );
    let c = pipeline(&bc, "rnd_vertex_fetch_uint", "rnd_fragment_id_data", &desc).unwrap();
    let bytes = u32_bytes(&(0..8).map(|i| 0x1000 + i).collect::<Vec<_>>());
    let buffered = u32_bytes(&(0..32).map(|i| 0x2000 + i).collect::<Vec<_>>());
    let mut draws = vec![Draw::new(0, 4, vec![vslot(1, bytes, 0)])];
    for (k, offset) in [0u64, 40, 64].iter().enumerate() {
        let mut d = Draw::new(0, 4, vec![vslot(1, buffered.clone(), *offset)]);
        d.base_instance = k as u32 + 1;
        draws.push(d);
    }
    let scene = Scene {
        compiled: c,
        format: "rgba32uint",
        width: 16,
        height: 16,
        samples: 1,
        clear: [0.0; 4],
        depth: None,
        topology: "point-list",
        draws,
    };
    let (out, dawn) = render("bytes-and-offset", &scene);
    expect_equal(
        "vertex.bytes-and-buffer-offset",
        "fetched",
        &region(&out.color, 16, 16, 0, 0, 4, 4),
        &g.data("fetched"),
    );
    note_dawn("vertex.bytes-and-buffer-offset", dawn);
}

/// `vertex.attribute-stride`: MTLBufferLayoutStrideDynamic. The A10X reports no dynamic attribute stride (D1), so
/// the pipeline is refused as on the device.
#[test]
fn vertex_attribute_stride() {
    let Some(r) = skip("vertex.attribute-stride") else { return };
    let bc = v1_air(&r, "render");
    let desc = render_desc(
        RGBA32UINT,
        0,
        1,
        Some(VertexDescriptor {
            layouts: vec![
                VertexBufferLayout { stride: 0, step_function: 1, step_rate: 1 },
                VertexBufferLayout { stride: u64::MAX, step_function: 1, step_rate: 1 },
            ],
            attributes: vec![VertexAttribute { format: 37, offset: 0, buffer_index: 1 }],
        }),
    );
    let e = pipeline(&bc, "rnd_vertex_fetch_uint", "rnd_fragment_id_data", &desc).unwrap_err();
    assert_eq!((e.kind, item(&e).as_str()), (RefusalKind::RejectedByDevice, "vertex-descriptor"), "{}", e);
}

/// `draw.vertex-instance-ids`: vertex_id and instance_id with vertexStart, instanceCount and baseInstance.
#[test]
fn draw_vertex_instance_ids() {
    let Some(r) = skip("draw.vertex-instance-ids") else { return };
    let g = Golden::load(&r, "draw.vertex-instance-ids");
    let bc = v1_air(&r, "render");
    let c = pipeline(&bc, "rnd_vertex_ids", "rnd_fragment_id_data", &render_desc(RGBA32UINT, 0, 1, None)).unwrap();
    let mut a = Draw::new(0, 4, vec![]);
    a.instance_count = 2;
    let mut b = Draw::new(5, 6, vec![]);
    b.instance_count = 3;
    b.base_instance = 9;
    let scene = Scene {
        compiled: c,
        format: "rgba32uint",
        width: 16,
        height: 16,
        samples: 1,
        clear: [0.0; 4],
        depth: None,
        topology: "point-list",
        draws: vec![a, b],
    };
    let (out, dawn) = render("vertex-instance-ids", &scene);
    expect_equal("draw.vertex-instance-ids", "ids", &out.color, &g.data("ids"));
    note_dawn("draw.vertex-instance-ids", dawn);
}

/// `draw.indexed`: uint16 and uint32 indices at offsets, base vertex (one negative) and base instance.
#[test]
fn draw_indexed() {
    let Some(r) = skip("draw.indexed") else { return };
    let g = Golden::load(&r, "draw.indexed");
    let bc = v1_air(&r, "render");
    let c = pipeline(&bc, "rnd_vertex_ids", "rnd_fragment_id_data", &render_desc(RGBA32UINT, 0, 1, None)).unwrap();
    let i16: Vec<u8> = [0xdeadu16, 0xbeef, 0xf00d, 0, 1, 2, 5, 9, 3].iter().flat_map(|x| x.to_le_bytes()).collect();
    let i32b = u32_bytes(&[0xdeadbeef, 0xfeedface, 10, 12, 14, 7]);
    let idx = |data: &Vec<u8>, uint32, offset| Some(Indices { data: data.clone(), uint32, offset });
    let mut a = Draw::new(0, 6, vec![]);
    a.indices = idx(&i16, false, 6);
    a.instance_count = 2;
    a.base_vertex = 4;
    a.base_instance = 7;
    let mut b = Draw::new(0, 4, vec![]);
    b.indices = idx(&i32b, true, 8);
    b.base_vertex = -3;
    b.base_instance = 12;
    let mut c3 = Draw::new(0, 3, vec![]);
    c3.indices = idx(&i16, false, 8);
    // Point lists take no strip index format, so the draws may use both index formats in one pass.
    let scene = Scene {
        compiled: c,
        format: "rgba32uint",
        width: 16,
        height: 16,
        samples: 1,
        clear: [0.0; 4],
        depth: None,
        topology: "point-list",
        draws: vec![a, b, c3],
    };
    let (out, dawn) = render("indexed", &scene);
    let merged = out.color;
    expect_equal("draw.indexed", "ids", &merged, &g.data("ids"));
    note_dawn("draw.indexed", dawn);
}

/// `draw.primitive-restart`: 0xFFFF and 0xFFFFFFFF split indexed triangle strips (the line strips' coverage is not
/// defined by Metal and is not compared).
#[test]
fn draw_primitive_restart() {
    let Some(r) = skip("draw.primitive-restart") else { return };
    let g = Golden::load(&r, "draw.primitive-restart");
    let bc = v1_air(&r, "render");
    let corners = [[1.0, 1.0], [1.0, 5.0], [7.0, 1.0], [7.0, 5.0], [9.0, 8.0], [9.0, 14.0], [14.0, 8.0], [14.0, 14.0]];
    let v: Vec<RVertex> =
        corners.iter().enumerate().map(|(i, c)| vertex_at(c[0], c[1], 0.0, 16, 16, i as u32 / 4 + 1)).collect();
    let c = pipeline(&bc, "rnd_vertex_triangles", "rnd_fragment_id", &render_desc(R32UINT, 3, 1, None)).unwrap();
    let s16: Vec<u8> = [0u16, 1, 2, 3, 0xffff, 4, 5, 6, 7].iter().flat_map(|x| x.to_le_bytes()).collect();
    let s32 = u32_bytes(&[0, 1, 2, 3, 0xffffffff, 4, 5, 6, 7]);
    let mut dawn = false;
    for (name, data, uint32) in [("triangle-strip.uint16", s16, false), ("triangle-strip.uint32", s32, true)] {
        let mut d = Draw::new(0, 9, vec![vslot(0, vertex_bytes(&v), 0)]);
        d.indices = Some(Indices { data, uint32, offset: 0 });
        let scene = Scene {
            compiled: c.clone(),
            format: "r32uint",
            width: 16,
            height: 16,
            samples: 1,
            clear: [0.0; 4],
            depth: None,
            topology: "triangle-strip",
            draws: vec![d],
        };
        let (out, dn) = render(&format!("restart-{}", name), &scene);
        dawn = dn;
        expect_equal("draw.primitive-restart", name, &out.color, &g.data(name));
    }
    note_dawn("draw.primitive-restart", dawn);
}

/// A triangle covering a whole w x h target (FCover of RenderFragment.m).
fn cover(w: u32, h: u32, id: u32, z: f64) -> Vec<RVertex> {
    vec![
        vertex_at(-1.0, -1.0, z, w, h, id),
        vertex_at(3.0 * w as f64, -1.0, z, w, h, id),
        vertex_at(-1.0, 3.0 * h as f64, z, w, h, id),
    ]
}

/// `render.discard`: discard_fragment for pixels with (x + y) % 3 == 0.
#[test]
fn render_discard() {
    let Some(r) = skip("render.discard") else { return };
    let g = Golden::load(&r, "render.discard");
    let bc = v1_air(&r, "render");
    let c = pipeline(&bc, "rnd_vertex_triangles", "rnd_fragment_discard", &render_desc(R32UINT, 0, 1, None)).unwrap();
    let scene = Scene {
        compiled: c,
        format: "r32uint",
        width: 8,
        height: 8,
        samples: 1,
        clear: [9.0, 0.0, 0.0, 0.0],
        depth: None,
        topology: "triangle-list",
        draws: vec![Draw::new(0, 3, vec![vslot(0, vertex_bytes(&cover(8, 8, 4, 0.0)), 0)])],
    };
    let (out, dawn) = render("discard", &scene);
    expect_equal("render.discard", "result", &out.color, &g.data("result"));
    note_dawn("render.discard", dawn);
}

fn ulps(a: u32, b: u32) -> u64 {
    let key = |x: u32| {
        if x & 0x8000_0000 != 0 { 0x8000_0000u64 - (x & 0x7fff_ffff) as u64 } else { 0x8000_0000u64 + x as u64 }
    };
    key(a).abs_diff(key(b))
}

/// `render.interpolation`: center_perspective, center_no_perspective, centroid_perspective and flat. The flat value
/// comes from the first vertex in both (the provoking vertex Metal uses, measured, and WGSL's `flat` with sampling
/// `first`) and must be equal; the smooth values have rule `none` (Metal states no interpolation precision) and their
/// distance to Metal's is reported.
#[test]
fn render_interpolation() {
    let Some(r) = skip("render.interpolation") else { return };
    let g = Golden::load(&r, "render.interpolation");
    let bc = v1_air(&r, "render");
    let c = pipeline(&bc, "rnd_vertex_interpolated", "rnd_fragment_interpolated", &render_desc(RGBA32UINT, 0, 1, None))
        .unwrap();
    let mut v = vec![
        vertex_at(-1.0, -1.0, 0.0, 8, 8, 1),
        vertex_at(24.0, -1.0, 0.0, 8, 8, 2),
        vertex_at(-1.0, 24.0, 0.0, 8, 8, 3),
    ];
    let values = [0.0f32, 1.0, 0.5];
    let ws = [1.0f32, 2.0, 4.0];
    for k in 0..3 {
        v[k].extra[1] = values[k];
        for i in 0..4 {
            v[k].position[i] *= ws[k];
        }
    }
    let scene = |v: &[RVertex]| Scene {
        compiled: c.clone(),
        format: "rgba32uint",
        width: 8,
        height: 8,
        samples: 1,
        clear: [0.0; 4],
        depth: None,
        topology: "triangle-list",
        draws: vec![Draw::new(0, 3, vec![vslot(0, vertex_bytes(v), 0)])],
    };
    let (out, dawn) = render("interpolation-distinct", &scene(&v));
    let want = words(&g.data("distinct-vertex-values"));
    let got = words(&out.color);
    let mut worst = [0u64; 3];
    for p in 0..64 {
        assert_eq!(got[p * 4 + 3], want[p * 4 + 3], "flat value of pixel {} (provoking vertex)", p);
        for q in 0..3 {
            worst[q] = worst[q].max(ulps(got[p * 4 + q], want[p * 4 + q]));
        }
    }
    eprintln!(
        "render.interpolation: largest distance to Metal in ulps: perspective {}, no_perspective {}, centroid {} (rule none)",
        worst[0], worst[1], worst[2]
    );
    for x in v.iter_mut() {
        x.extra[1] = 0.75;
        x.data[0] = 6;
    }
    let (out, _) = render("interpolation-equal", &scene(&v));
    let flat: Vec<u8> = out.color.chunks(16).flat_map(|p| p[12..16].to_vec()).collect();
    expect_equal("render.interpolation", "equal-vertex-values.flat", &flat, &g.data("equal-vertex-values.flat"));
    let smooth_equal = first_difference(&out.color, &g.data("equal-vertex-values.smooth")).is_none();
    eprintln!("render.interpolation: equal-vertex smooth values equal Metal's: {} (rule none)", smooth_equal);
    note_dawn("render.interpolation", dawn);
}

/// `raster.triangle-coverage`: coverage by pixel-centre containment.
#[test]
fn raster_triangle_coverage() {
    let Some(r) = skip("raster.triangle-coverage") else { return };
    let g = Golden::load(&r, "raster.triangle-coverage");
    let bc = v1_air(&r, "render");
    let tris: [[f64; 6]; 6] = [
        [1.125, 1.5625, 9.375, 2.0625, 3.4375, 10.9375],
        [14.875, 1.125, 8.75, 13.3125, 15.375, 13.0],
        [1.25, 13.125, 32.0, 13.4375, 1.25, 13.9375],
        [10.0625, 5.4375, 12.25, 7.5625, 11.15625, 6.5],
        [6.25, 5.9375, 6.875, 6.625, 6.25, 6.8125],
        [12.5625, -2.8125, 20.125, 4.3125, 11.5, 6.375],
    ];
    let v: Vec<RVertex> = tris
        .iter()
        .enumerate()
        .flat_map(|(t, p)| (0..3).map(move |k| vertex_at(p[2 * k], p[2 * k + 1], 0.0, 16, 16, t as u32 + 1)))
        .collect();
    let c = pipeline(&bc, "rnd_vertex_triangles", "rnd_fragment_id", &render_desc(R32UINT, 0, 1, None)).unwrap();
    let scene = Scene {
        compiled: c,
        format: "r32uint",
        width: 16,
        height: 16,
        samples: 1,
        clear: [0.0; 4],
        depth: None,
        topology: "triangle-list",
        draws: vec![Draw::new(0, 18, vec![vslot(0, vertex_bytes(&v), 0)])],
    };
    let (out, dawn) = render("triangle-coverage", &scene);
    expect_equal("raster.triangle-coverage", "coverage", &out.color, &g.data("coverage"));
    note_dawn("raster.triangle-coverage", dawn);
}

/// `raster.points`: the point sizes come from a buffer, so the compiler cannot show they are 1; sizes other than 1
/// need the quad expansion, which is refused (limits.toml question 5.7-point-rasterization).
#[test]
fn raster_points() {
    let Some(r) = skip("raster.points") else { return };
    let bc = v1_air(&r, "render");
    let e = pipeline(&bc, "rnd_vertex", "rnd_fragment_point_id", &render_desc(R32UINT, 0, 1, None)).unwrap_err();
    assert_eq!((e.kind, item(&e).as_str()), (RefusalKind::NotImplemented, "point-size"), "{}", e);
}

/// `depth-stencil.fragment-depth`: [[depth(any)]], [[depth(greater)]] and [[depth(less)]] written to Depth32Float.
#[test]
fn depth_stencil_fragment_depth() {
    let Some(r) = skip("depth-stencil.fragment-depth") else { return };
    let g = Golden::load(&r, "depth-stencil.fragment-depth");
    let bc = v1_air(&r, "render");
    let mut dawn = false;
    for (fs, z, name) in [
        ("rnd_fragment_depth_any", 0.5, "any"),
        ("rnd_fragment_depth_greater", 0.25, "greater"),
        ("rnd_fragment_depth_less", 0.75, "less"),
    ] {
        let mut desc = render_desc(R32UINT, 0, 1, None);
        if let PipelineKind::Render(rs) = &mut desc.kind {
            rs.depth_attachment_pixel_format = 252;
        }
        let c = pipeline(&bc, "rnd_vertex_triangles", fs, &desc).unwrap();
        let scene = Scene {
            compiled: c,
            format: "r32uint",
            width: 16,
            height: 4,
            samples: 1,
            clear: [0.0; 4],
            depth: Some(1.0),
            topology: "triangle-list",
            draws: vec![Draw::new(0, 3, vec![vslot(0, vertex_bytes(&cover(16, 4, 1, z)), 0)])],
        };
        let (out, dn) = render(&format!("depth-{}", name), &scene);
        dawn = dn;
        let rec = format!("{}.depth", name);
        expect_equal("depth-stencil.fragment-depth", &rec, out.depth.as_ref().unwrap(), &g.data(&rec));
    }
    note_dawn("depth-stencil.fragment-depth", dawn);
}

/// `msaa.coverage-4x`: coverage, the [[sample_mask]] input and output, and [[sample_id]] with 4 samples at the default
/// positions, compared where the golden is unambiguous (samples, or for the mask input whole pixels, within 1/16 pixel
/// of an edge are masked with the golden's own classification). The case renders to R32Uint, which WebGPU cannot
/// multisample (its format table); every value it writes is below 2^16, so the replay renders to R16Uint, which WebGPU
/// can, and reads the samples back as u32: the stage interface's results are the same.
#[test]
fn msaa_coverage_4x() {
    let Some(r) = skip("msaa.coverage-4x") else { return };
    let g = Golden::load(&r, "msaa.coverage-4x");
    let bc = v1_air(&r, "render");
    let classes = words(&g.data("classification"));
    const AMBIGUOUS: u32 = 0xffff_ffff;
    let mask = |measured: &[u8], whole: bool| -> Vec<u8> {
        let mut m = words(measured);
        for p in 0..64 {
            let any = (0..4).any(|s| classes[p * 4 + s] == AMBIGUOUS);
            for s in 0..4 {
                if if whole { any } else { classes[p * 4 + s] == AMBIGUOUS } {
                    m[p * 4 + s] = AMBIGUOUS;
                }
            }
        }
        u32_bytes(&m)
    };
    let tri = [1.0625, 0.8125, 7.4375, 2.3125, 2.1875, 7.6875];
    let mut dawn = false;
    for (fs, mask_value, record, whole) in [
        ("rnd_fragment_id", 0, "coverage.unambiguous", false),
        ("rnd_fragment_mask_in", 0, "sample-mask-input.unambiguous-pixels", true),
        ("rnd_fragment_sample_id", 0, "sample-id.unambiguous", false),
        ("rnd_fragment_mask_out", 0x5, "sample-mask-output-0x5.unambiguous", false),
    ] {
        let c = pipeline(&bc, "rnd_vertex_triangles", fs, &render_desc(R16UINT, 0, 4, None)).unwrap();
        let v: Vec<RVertex> = (0..3)
            .map(|k| {
                let mut x = vertex_at(tri[2 * k], tri[2 * k + 1], 0.0, 8, 8, 1);
                x.data[1] = mask_value;
                x
            })
            .collect();
        let scene = Scene {
            compiled: c,
            format: "r16uint",
            width: 8,
            height: 8,
            samples: 4,
            clear: [0.0; 4],
            depth: None,
            topology: "triangle-list",
            draws: vec![Draw::new(0, 3, vec![vslot(0, vertex_bytes(&v), 0)])],
        };
        let (out, dn) = render(&format!("msaa-{}", fs), &scene);
        dawn = dn;
        expect_equal("msaa.coverage-4x", record, &mask(&out.color, whole), &g.data(record));
    }
    note_dawn("msaa.coverage-4x", dawn);
}

fn compute_desc(threads: [u64; 3], stage_input: Option<StageInputDescriptor>) -> PipelineDesc {
    PipelineDesc {
        kind: PipelineKind::Compute(ComputeState {
            threads_per_threadgroup: threads,
            non_uniform_grid: false,
            threadgroup_memory_lengths: Vec::new(),
            stage_input,
        }),
        bindings: Vec::new(),
    }
}

/// `compute.thread-positions`: the kernel built-ins for dispatchThreadgroups.
#[test]
fn compute_thread_positions() {
    let Some(r) = skip("compute.thread-positions") else { return };
    let g = Golden::load(&r, "compute.thread-positions");
    let bc = v1_air(&r, "compute");
    let c = compile(&[module(&bc, "compute_positions", 3)], &compute_desc([4, 2, 1], None), &host()).unwrap();
    let count = 12 * 4 * 2;
    let slots = vec![
        Slot { stage: MetalStage::Kernel, index: 0, data: vec![0; count * 64], offset: 0 },
        Slot { stage: MetalStage::Kernel, index: 1, data: u32_bytes(&[12, 4, 2, 0]), offset: 0 },
    ];
    let (out, dawn) = dispatch("thread-positions", &c, [3, 2, 2], &slots, None);
    expect_equal("compute.thread-positions", "records", &out[0], &g.data("records"));
    note_dawn("compute.thread-positions", dawn);
}

/// `compute.stage-in-region`: stage-in attributes fetched through the region set by setStageInRegion:.
#[test]
fn compute_stage_in_region() {
    let Some(r) = skip("compute.stage-in-region") else { return };
    let g = Golden::load(&r, "compute.stage-in-region");
    let bc = v1_air(&r, "compute");
    let si = StageInputDescriptor {
        layouts: vec![(0, 0, 0), (8, 5, 1)],
        attributes: vec![(13, 0, 1), (28, 4, 1)],
        index_type: 0,
        index_buffer_index: 0,
    };
    let c = compile(&[module(&bc, "compute_stage_in", 3)], &compute_desc([6, 2, 1], Some(si)), &host())
        .unwrap_or_else(|e| panic!("{}", e));
    let slots = vec![
        Slot { stage: MetalStage::Kernel, index: 0, data: vec![0; 6 * 2 * 16], offset: 0 },
        Slot { stage: MetalStage::Kernel, index: 1, data: g.data("input"), offset: 0 },
    ];
    let (out, dawn) = dispatch("stage-in-region", &c, [1, 1, 1], &slots, Some(([5, 0, 0], [6, 2, 1])));
    expect_equal("compute.stage-in-region", "fetched", &out[0], &g.data("fetched"));
    note_dawn("compute.stage-in-region", dawn);
}

/// `compute.dispatch-threads-nonuniform`: dispatchThreads needs nonuniform threadgroup sizes, which the A10X does not
/// support (D1); the variant is refused as on the device.
#[test]
fn compute_dispatch_threads_nonuniform() {
    let Some(r) = skip("compute.dispatch-threads-nonuniform") else { return };
    let bc = v1_air(&r, "compute");
    let mut desc = compute_desc([4, 2, 2], None);
    if let PipelineKind::Compute(cs) = &mut desc.kind {
        cs.non_uniform_grid = true;
    }
    let e = compile(&[module(&bc, "compute_positions", 3)], &desc, &host()).unwrap_err();
    assert_eq!((e.kind, item(&e).as_str()), (RefusalKind::RejectedByDevice, "non-uniform-dispatch"), "{}", e);
}
