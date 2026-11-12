//! Attribute fetch compared with real Metal on this Mac for every code of the 8-, 10- and 16-bit formats, 65536
//! binary32 patterns (NaNs, infinities, rounding boundaries and random values), 32-bit integers truncated to narrower
//! attributes, and elements at addresses that are not multiples of 4. Each combination is drawn as points on Metal
//! (tests/stage/metal_fetch.swift), on wgpu and on Dawn, and the fetched attributes must be byte-identical; a
//! combination Metal refuses must be refused by the compiler as on the device.

use super::harness::*;
use airwgsl::api::*;
use airwgsl::wgsl::tracked;
use std::process::Command;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct P {
    float4 pos [[position]];
    uint4 data [[flat]];
    float ps [[point_size]];
};

static float4 at(uint v) {
    return float4((float(v % 256u) + 0.5f) / 128.0f - 1.0f, 1.0f - (float(v / 256u) + 0.5f) / 128.0f, 0.0f, 1.0f);
}

#define FETCH(T, WIDEN) \
    struct I_##T { T v [[attribute(0)]]; }; \
    vertex P v_##T(I_##T in [[stage_in]], uint vid [[vertex_id]]) { \
        P o; o.pos = at(vid); o.data = WIDEN; o.ps = 1.0f; return o; \
    }

FETCH(float4, as_type<uint4>(in.v))
FETCH(half4, uint4(as_type<ushort4>(in.v)))
FETCH(uint4, in.v)
FETCH(int4, as_type<uint4>(in.v))
FETCH(ushort4, uint4(in.v))
FETCH(short4, uint4(as_type<ushort4>(in.v)))
FETCH(uchar4, uint4(in.v))
FETCH(char4, uint4(as_type<uchar4>(in.v)))
FETCH(float2, uint4(as_type<uint2>(in.v), 0xeeeeeeeeu, 0xeeeeeeeeu))

fragment uint4 f_data(P in [[stage_in]]) { return in.data; }
"#;

/// Every 16-bit code, little-endian.
fn codes16() -> Vec<u8> {
    (0..=0xffffu16).flat_map(|c| c.to_le_bytes()).collect()
}

fn codes8() -> Vec<u8> {
    (0..=0xffu8).collect()
}

/// Every 10-bit code in x, other codes in y and z, every 2-bit code in w.
fn codes1010102() -> Vec<u8> {
    (0..1024u32)
        .flat_map(|c| (c | ((c * 7) & 1023) << 10 | ((c * 13) & 1023) << 20 | (c & 3) << 30).to_le_bytes())
        .collect()
}

/// Binary32 patterns: NaNs of both signs and payloads, infinities, zeros, values at binary16's overflow, subnormal
/// and rounding boundaries, then random words.
fn floats() -> Vec<u8> {
    let mut v: Vec<u32> = vec![
        0x7fc0_0000,
        0xffc0_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fa0_0000,
        0x7fff_ffff,
        0x7f80_0000,
        0xff80_0000,
        0,
        0x8000_0000,
        0x477f_e000,
        0x477f_f000,
        0x477f_efff,
        0x477f_f001,
        0x3300_0000,
        0x3300_0001,
        0x32ff_ffff,
        0x387f_c000,
        0x387f_e000,
        0x3880_0000,
        0x3f80_1000,
        0x3f80_3000,
        0x3f80_1001,
        0x3f80_0fff,
        0x0000_0001,
        0x0080_0000,
    ];
    let b = bytes(0x5eed, 4 * 65536);
    v.extend(b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).take(65536 - v.len()));
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Deterministic bytes (SplitMix64).
fn bytes(seed: u64, n: usize) -> Vec<u8> {
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

struct Case {
    format: u64,
    name: &'static str,
    attr: &'static str,
    stride: u64,
    offset: u64,
    input: Vec<u8>,
}

fn run(c: &Case, bc: &[u8], dir: &std::path::Path) -> bool {
    let count = ((c.input.len() as u64 - c.offset) / c.stride) as u32;
    let vertex = format!("v_{}", c.attr);
    let input = dir.join(format!("{}-{}.in", c.name, c.attr));
    std::fs::write(&input, &c.input).unwrap();
    let out = dir.join(format!("{}-{}.metal.out", c.name, c.attr));
    let src = dir.join("fetch.metal");
    let o = Command::new(metal_fetch())
        .arg(&src)
        .arg(&vertex)
        .arg("f_data")
        .arg(c.format.to_string())
        .arg(c.stride.to_string())
        .arg(c.offset.to_string())
        .arg(&input)
        .arg(count.to_string())
        .arg(&out)
        .output()
        .unwrap();
    assert!(o.status.success(), "metal_fetch: {}", String::from_utf8_lossy(&o.stderr));
    let refused = String::from_utf8_lossy(&o.stdout).starts_with("refused");
    let desc = render_desc(
        123,
        1,
        1,
        Some(VertexDescriptor {
            layouts: vec![
                VertexBufferLayout { stride: 0, step_function: 1, step_rate: 1 },
                VertexBufferLayout { stride: c.stride, step_function: 1, step_rate: 1 },
            ],
            attributes: vec![VertexAttribute { format: c.format, offset: c.offset, buffer_index: 1 }],
        }),
    );
    let compiled = compile(&[module(bc, &vertex, 1), module(bc, "f_data", 2)], &desc, &host());
    if refused {
        let e = compiled.expect_err("Metal refuses the combination; the compiler must too");
        assert_eq!(e.kind, RefusalKind::RejectedByDevice, "{}", e);
        assert_eq!(tracked::item_of(&e).map(|i| i.1), Some("vertex-descriptor".into()), "{}", e);
        return false;
    }
    let compiled = compiled.unwrap_or_else(|e| panic!("{} as {}: {}", c.name, c.attr, e));
    let reference = std::fs::read(&out).unwrap();
    let height = count.div_ceil(256);
    let scene = Scene {
        compiled,
        format: "rgba32uint",
        width: 256,
        height,
        samples: 1,
        clear: [0.0; 4],
        depth: None,
        topology: "point-list",
        draws: vec![Draw::new(
            0,
            count,
            vec![Slot { stage: MetalStage::Vertex, index: 1, data: c.input.clone(), offset: 0 }],
        )],
    };
    let (got, _) = render(&format!("fetch-{}-{}", c.name, c.attr), &scene);
    let got = &got.color[..(count * 16) as usize];
    if let Some(i) = first_difference(got, &reference) {
        let (g, r) = (words(got), words(&reference));
        panic!(
            "{} as {}: element {} differs: compiled {:08x?}, Metal {:08x?}",
            c.name,
            c.attr,
            i / 4,
            &g[i / 4 * 4..i / 4 * 4 + 4],
            &r[i / 4 * 4..i / 4 * 4 + 4]
        );
    }
    true
}

#[test]
fn fetch_conversions_match_metal() {
    let dir = case_dir("fetch");
    std::fs::write(dir.join("fetch.metal"), SOURCE).unwrap();
    let bc = air(&dir, SOURCE, "metal3.1");
    let mut cases = Vec::new();
    let mut add =
        |format: u64, name: &'static str, attrs: &[&'static str], stride: u64, offset: u64, input: &Vec<u8>| {
            for a in attrs {
                cases.push(Case { format, name, attr: a, stride, offset, input: input.clone() });
            }
        };
    let (c16, c8, c10, f) = (codes16(), codes8(), codes1010102(), floats());
    let all_types = ["float4", "half4", "uint4", "int4", "ushort4", "short4", "uchar4", "char4"];
    add(51, "UShortNormalized", &["float4", "half4", "uint4"], 2, 0, &c16);
    add(52, "ShortNormalized", &["float4", "half4"], 2, 0, &c16);
    add(49, "UShort", &all_types, 2, 0, &c16);
    add(50, "Short", &all_types, 2, 0, &c16);
    add(53, "Half", &["float4", "half4", "uint4"], 2, 0, &c16);
    add(47, "UCharNormalized", &["float4", "half4"], 1, 0, &c8);
    add(48, "CharNormalized", &["float4", "half4"], 1, 0, &c8);
    add(45, "UChar", &all_types, 1, 0, &c8);
    add(46, "Char", &all_types, 1, 0, &c8);
    add(40, "Int1010102Normalized", &["float4", "half4", "int4"], 4, 0, &c10);
    add(41, "UInt1010102Normalized", &["float4", "half4"], 4, 0, &c10);
    add(42, "UChar4Normalized_BGRA", &["float4", "half4"], 4, 0, &c10);
    add(28, "Float", &["float4", "half4", "float2", "uint4"], 4, 0, &f);
    add(36, "UInt", &["uint4", "ushort4", "uchar4", "float4"], 4, 0, &f);
    add(32, "Int", &["int4", "short4", "char4", "half4"], 4, 0, &f);
    // Elements at addresses that are not multiples of 4.
    add(14, "UShort3-stride6", &["uint4"], 6, 0, &c16);
    add(2, "UChar3-stride3", &["uint4", "float4"], 3, 0, &c8);
    add(30, "Float3-offset2-stride14", &["float4"], 14, 2, &f[..14 * 4096].to_vec());
    add(26, "Half3-offset1-stride7", &["half4", "float4"], 7, 1, &c16[..7 * 4096].to_vec());
    let mut compared = 0;
    let mut refused = 0;
    for c in &cases {
        if run(c, &bc, &dir) {
            compared += 1;
        } else {
            refused += 1;
        }
    }
    eprintln!(
        "fetch: {} combinations equal to Metal{}, {} refused as Metal refuses them",
        compared,
        if dawn_node().is_some() { " on wgpu and Dawn" } else { " on wgpu (Dawn not run: AW_DAWN_NODE unset)" },
        refused
    );
}
