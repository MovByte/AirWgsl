//! The cases. Each kernel writes its results to buffer 1 (or into a texture that is read back).

use super::harness::*;
use airwgsl::api::SamplerSlot;

fn report(name: &str, r: Ran) {
    if !r.dawn {
        eprintln!("{}: Dawn not run (set AW_DAWN_NODE)", name);
    }
}

/// R32Uint texels `0xa0LLYYXX` (level, row, column) for every level of a `w` x `h` texture with `slices` slices.
fn r32_levels(w: u32, h: u32, slices: u32, levels: u32) -> Vec<u8> {
    let mut v = Vec::new();
    for l in 0..levels {
        let (lw, lh) = ((w >> l).max(1), (h >> l).max(1));
        for s in 0..slices {
            for y in 0..lh {
                for x in 0..lw {
                    v.push(0xa000_0000 | (l << 24) | (s << 16) | (y << 8) | x);
                }
            }
        }
    }
    u32s(&v)
}

/// 8 x 8 RGBA32Float, texel (x, y) = (x + 1, y + 1, 100 + x + 8y, 0.5).
fn rgba32_8x8() -> Vec<u8> {
    let mut v = Vec::new();
    for y in 0..8 {
        for x in 0..8 {
            v.extend_from_slice(&[x as f32 + 1.0, y as f32 + 1.0, 100.0 + (x + 8 * y) as f32, 0.5]);
        }
    }
    f32s(&v)
}

/// Texel-centre coordinates (k + 0.5) / 8 inside, at the edges and one and two periods outside.
fn centres() -> Vec<f32> {
    let ks = [-17i32, -9, -8, -1, 0, 3, 7, 8, 12, 15, 16, 23];
    let vs = [-6i32, 2, 9];
    let mut c = Vec::new();
    for v in vs {
        for k in ks {
            c.push((k as f32 + 0.5) / 8.0);
            c.push((v as f32 + 0.5) / 8.0);
        }
    }
    c
}

const READ_2D: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(texture2d<uint, access::read> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
              device uint4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = t.read(q[i].xy, q[i].z);
    if (i == 0) {
        o[32] = uint4(t.get_width(0), t.get_height(1), t.get_num_mip_levels(), t.get_width(2));
    }
}
"#;

#[test]
fn read_2d_in_and_out_of_bounds() {
    let mut q = Vec::new();
    for (x, y, l) in
        [(0, 0, 0), (3, 2, 0), (1, 1, 1), (0, 0, 2), (4, 0, 0), (0, 4, 0), (2, 0, 1), (0, 0, 3), (0, 0, 99)]
    {
        q.extend_from_slice(&[x, y, l, 0u32]);
    }
    while q.len() < 32 * 4 {
        q.extend_from_slice(&[7, 7, 7, 0]);
    }
    let mut c = Case::new("read_2d", READ_2D, "k");
    c.threads = [32, 1, 1];
    c.buffers = vec![u32s(&q), vec![0; 33 * 16]];
    c.textures = vec![TexIn::new(0, 2, 53, [4, 4, 1], 3, 4, r32_levels(4, 4, 1, 3))];
    report(c.name, differential(&c));
}

const READ_FLOAT: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(texture2d<float, access::read> t [[texture(0)]], texture2d<half, access::read> h [[texture(1)]],
              device const uint4 *q [[buffer(0)]], device float4 *o [[buffer(1)]], device half4 *oh [[buffer(2)]],
              uint i [[thread_position_in_grid]])
{
    o[i] = t.read(q[i].xy);
    oh[i] = h.read(q[i].xy);
}
"#;

#[test]
fn read_unorm_formats_with_defaults() {
    // Every RGBA8Unorm code, and R8Unorm read through half (components it lacks: 0 and alpha 1).
    let data: Vec<u8> = (0..=255u8).collect();
    let mut q = Vec::new();
    for i in 0..64u32 {
        q.extend_from_slice(&[i % 8, i / 8, 0, 0]);
    }
    q.extend_from_slice(&[8, 0, 0, 0]);
    q.extend_from_slice(&[0, 9, 0, 0]);
    let n = q.len() / 4;
    let mut c = Case::new("read_unorm", READ_FLOAT, "k");
    c.threads = [n as u64, 1, 1];
    c.buffers = vec![u32s(&q), vec![0; n * 16], vec![0; n * 8]];
    c.textures = vec![
        TexIn::new(0, 2, 70, [8, 8, 1], 1, 4, data.iter().cycle().take(256).copied().collect()),
        TexIn::new(1, 2, 10, [8, 8, 1], 1, 1, (0..64u8).map(|x| x.wrapping_mul(37)).collect()),
    ];
    report(c.name, differential(&c));
}

const READ_TYPES: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void r1(texture1d<uint, access::read> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
               device uint4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(q[i].x); if (i == 0) o[15] = uint4(t.get_width(), t.get_num_mip_levels(), 0, 0); }
kernel void r1a(texture1d_array<uint, access::read> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
                device uint4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(q[i].x, q[i].y); if (i == 0) o[15] = uint4(t.get_width(), t.get_array_size(), 0, 0); }
kernel void r2a(texture2d_array<uint, access::read> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
                device uint4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(q[i].xy, q[i].z, q[i].w); if (i == 0) o[15] = uint4(t.get_width(), t.get_array_size(), t.get_num_mip_levels(), 0); }
kernel void r3(texture3d<uint, access::read> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
               device uint4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(q[i].xyz, q[i].w); if (i == 0) o[15] = uint4(t.get_width(1), t.get_height(1), t.get_depth(1), t.get_num_mip_levels()); }
kernel void rc(texturecube<uint, access::read> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
               device uint4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(q[i].xy, q[i].z, q[i].w); if (i == 0) o[15] = uint4(t.get_width(), t.get_height(1), 0, 0); }
"#;

fn read_case(function: &str, tex: TexIn, queries: &[[u32; 4]]) {
    let mut q: Vec<u32> = queries.iter().flatten().copied().collect();
    while q.len() < 15 * 4 {
        q.extend_from_slice(&queries[0]);
    }
    let name = format!("read_{}", function);
    let mut c = Case::new(&name, READ_TYPES, function);
    c.threads = [15, 1, 1];
    c.buffers = vec![u32s(&q), vec![0; 16 * 16]];
    c.textures = vec![tex];
    report(c.name, differential(&c));
}

#[test]
fn read_1d_arrays_3d_and_cube() {
    read_case(
        "r1",
        TexIn::new(0, 0, 53, [8, 1, 1], 1, 4, r32_levels(8, 1, 1, 1)),
        &[[0, 0, 0, 0], [7, 0, 0, 0], [8, 0, 0, 0], [100, 0, 0, 0]],
    );
    read_case(
        "r1a",
        TexIn::new(0, 1, 53, [8, 1, 3], 1, 4, r32_levels(8, 1, 3, 1)),
        &[[0, 0, 0, 0], [5, 2, 0, 0], [8, 1, 0, 0], [1, 3, 0, 0]],
    );
    read_case(
        "r2a",
        TexIn::new(0, 3, 53, [4, 4, 2], 2, 4, r32_levels(4, 4, 2, 2)),
        &[[0, 0, 0, 0], [3, 3, 1, 0], [1, 1, 1, 1], [2, 0, 1, 1], [0, 0, 2, 0], [0, 0, 0, 2]],
    );
    let mut d3 = Vec::new();
    for l in 0..2u32 {
        let s = 4 >> l;
        for z in 0..s {
            for y in 0..s {
                for x in 0..s {
                    d3.push(0xb000_0000 | (l << 24) | (z << 16) | (y << 8) | x);
                }
            }
        }
    }
    read_case(
        "r3",
        TexIn::new(0, 7, 53, [4, 4, 4], 2, 4, u32s(&d3)),
        &[[0, 0, 0, 0], [3, 2, 1, 0], [1, 1, 1, 1], [0, 0, 4, 0], [2, 0, 0, 1], [0, 0, 0, 2]],
    );
    read_case(
        "rc",
        TexIn::new(0, 5, 53, [4, 4, 1], 2, 4, r32_levels(4, 4, 6, 2)),
        &[[0, 0, 0, 0], [3, 1, 5, 0], [1, 1, 2, 1], [0, 0, 6, 0], [4, 0, 0, 0], [0, 0, 0, 2]],
    );
}

const WRITE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void wf(texture2d<float, access::write> t [[texture(0)]], device const float4 *v [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(v[i], uint2(i % 4, i / 4)); t.write(float4(9), uint2(4 + i, 0)); }
kernel void wh(texture2d<half, access::write> t [[texture(0)]], device const float4 *v [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(half4(v[i]), uint2(i % 4, i / 4)); }
kernel void wi(texture2d<int, access::write> t [[texture(0)]], device const int4 *v [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(v[i], uint2(i % 4, i / 4)); }
kernel void wa(texture2d_array<uint, access::write> t [[texture(0)]], device const uint4 *v [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(v[i], uint2(i % 4, (i / 4) % 4), i / 16); }
kernel void w3(texture3d<float, access::write> t [[texture(0)]], device const float4 *v [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(v[i], uint3(i % 2, (i / 2) % 2, i / 4)); t.write(float4(5), uint3(0, 0, 2 + i)); }
kernel void w1(texture1d<float, access::write> t [[texture(0)]], device const float4 *v [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(v[i], i); }
"#;

/// Float values around the rounding points of 8-bit unorm codes (k + 0.5) / 255, and out of range.
fn rounding_values(n: usize) -> Vec<f32> {
    let mut v = Vec::new();
    for i in 0..n * 4 {
        let k = (i * 7 % 255) as f32;
        v.push(match i % 8 {
            0 => (k + 0.5) / 255.0,
            1 => (k + 0.5) / 255.0 + 1e-7,
            2 => -0.25,
            3 => 1.75,
            4 => k / 255.0,
            5 => (k + 0.49999) / 255.0,
            6 => 0.5,
            _ => f32::from_bits(0x3e80_0001 + i as u32),
        });
    }
    v
}

fn write_case(function: &str, tex: TexIn, values: Vec<u8>, n: u64) {
    write_case_with(function, tex, values, n, &[])
}

fn write_case_with(function: &str, tex: TexIn, values: Vec<u8>, n: u64, flags: &[&str]) {
    let name = format!("write_{}_{}_{}", function, tex.format, flags.len());
    let mut c = Case::new(&name, WRITE, function);
    c.flags = flags.to_vec();
    c.threads = [n, 1, 1];
    c.buffers = vec![values];
    let mut t = tex;
    t.write = true;
    c.textures = vec![t];
    report(c.name, differential(&c));
}

#[test]
fn writes_round_saturate_and_skip_out_of_bounds() {
    write_case("wf", TexIn::new(0, 2, 70, [4, 4, 1], 1, 4, vec![0; 64]), f32s(&rounding_values(16)), 16);
    write_case(
        "wf",
        TexIn::new(0, 2, 72, [4, 4, 1], 1, 4, vec![0; 64]),
        f32s(&rounding_values(16).iter().map(|x| x * 2.0 - 1.0).collect::<Vec<_>>()),
        16,
    );
    let halves: Vec<f32> =
        (0..64).map(|i| f32::from_bits(0x3f80_0000 + 0x1000 * i + if i % 2 == 0 { 0x1000 } else { 0xfff })).collect();
    write_case("wh", TexIn::new(0, 2, 115, [4, 4, 1], 1, 8, vec![0; 128]), f32s(&halves), 16);
    // Float values to binary16 with explicit rounding modes (the native mode is refused).
    for mode in ["-ftexture-write-rounding-mode=rte", "-ftexture-write-rounding-mode=rtz"] {
        let mut v = halves.clone();
        v[32..40].copy_from_slice(&[65519.0, 65520.0, 1e9, -7e4, 1e-6, -3e-7, 6.1e-5, f32::from_bits(1)]);
        write_case_with("wf", TexIn::new(0, 2, 115, [4, 4, 1], 1, 8, vec![0; 128]), f32s(&v), 16, &[mode]);
    }
    let ints: Vec<u32> = (0..64)
        .map(|i: u32| [0x7fff_ffff, 0x8000_0000, 200, 0xffff_ff00, 127, 128, 0xffff_ff80, 3][(i % 8) as usize])
        .collect();
    write_case("wi", TexIn::new(0, 2, 74, [4, 4, 1], 1, 4, vec![0; 64]), u32s(&ints), 16);
    let uints: Vec<u32> = (0..128).map(|i: u32| i * 0x0101_0101).collect();
    write_case("wa", TexIn::new(0, 3, 123, [4, 4, 2], 1, 16, vec![0; 512]), u32s(&uints), 32);
    write_case(
        "w3",
        TexIn::new(0, 7, 125, [2, 2, 2], 1, 16, vec![0; 128]),
        f32s(&(0..32).map(|i| i as f32).collect::<Vec<_>>()),
        8,
    );
    write_case(
        "w1",
        TexIn::new(0, 0, 125, [8, 1, 1], 1, 16, vec![0; 128]),
        f32s(&(0..48).map(|i| i as f32 * 0.5).collect::<Vec<_>>()),
        12,
    );
}

const WRITE_LEVELS: &str = r#"
#include <metal_stdlib>
using namespace metal;
// q[i] = (x, y, slice, face or z, level).xyzw with the level in w.
kernel void l2(texture2d<uint, access::write> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(uint4(0xc0de0000u | i, 0, 0, 0), q[i].xy, q[i].w); }
kernel void l2s(texture2d<uint, access::write> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
                uint i [[thread_position_in_grid]])
{ t.write(uint4(0xc0de0000u | i, 0, 0, 0), ushort2(q[i].xy), ushort(q[i].w)); }
kernel void la(texture2d_array<uint, access::write> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(uint4(0xc0de0000u | i, 0, 0, 0), q[i].xy, q[i].z, q[i].w); }
kernel void l3(texture3d<uint, access::write> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(uint4(0xc0de0000u | i, 0, 0, 0), q[i].xyz, q[i].w); }
kernel void lc(texturecube<uint, access::write> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(uint4(0xc0de0000u | i, 0, 0, 0), q[i].xy, q[i].z, q[i].w); }
kernel void lf(texture2d<float, access::write> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
               uint i [[thread_position_in_grid]])
{ t.write(float4(0.25 * float(i % 5), 1.0, 0.5, 0.0), q[i].xy, q[i].w); }
kernel void lk(texture2d<uint, access::write> t [[texture(0)]], device uint *o [[buffer(1)]],
               uint i [[thread_position_in_grid]])
{
    t.write(uint4(0xc0de0000u | i, 0, 0, 0), uint2(i, 0), 2);
    t.write(uint4(7), uint2(i, 1), 9);
    if (i == 0) {
        o[0] = t.get_num_mip_levels();
        o[1] = t.get_width(1);
        o[2] = t.get_height(2);
    }
}
"#;

/// R32Uint texels `0xa0LLZZXX`-style for every level of a `w` x `h` x `d` 3D texture: level, depth, row, column.
fn r32_volume(w: u32, h: u32, d: u32, levels: u32) -> Vec<u8> {
    let mut v = Vec::new();
    for l in 0..levels {
        let (lw, lh, ld) = ((w >> l).max(1), (h >> l).max(1), (d >> l).max(1));
        for z in 0..ld {
            for y in 0..lh {
                for x in 0..lw {
                    v.push(0xa000_0000 | (l << 24) | (z << 16) | (y << 8) | x);
                }
            }
        }
    }
    u32s(&v)
}

/// One kernel writing texture 0 at the queried coordinates and levels, compared with Metal level by level. The
/// queries keep to levels this Mac's GPU leaves alone when they are out of range: a level whose low 16 bits, added to
/// a view's first level, name a level of the parent is written there by Apple silicon, where MSL 4.1 section 6.13
/// ignores the write (WebMetal's reference case texture.write-levels records it).
fn level_case(name: &str, function: &str, tex: TexIn, queries: &[[u32; 4]]) {
    let mut c = Case::new(name, WRITE_LEVELS, function);
    c.threads = [queries.len() as u64, 1, 1];
    c.buffers = vec![u32s(&queries.concat())];
    let mut t = tex;
    t.write = true;
    c.textures = vec![t];
    report(c.name, differential(&c));
}

/// Out-of-range levels of a texture of `levels` levels that Apple silicon ignores as MSL does.
const OUT_OF_RANGE: [u32; 6] = [31, 32, 0xffff, 0x7fff_ffff, 0xffff_fffc, 0xffff_ffff];

#[test]
fn writes_at_levels_computed_at_run_time() {
    let mut q2 = vec![
        [1, 1, 0, 0],
        [2, 1, 0, 1],
        [1, 0, 0, 2],
        [0, 0, 0, 3],
        [7, 7, 0, 0],
        [3, 3, 0, 1],
        [4, 0, 0, 1],
        [2, 0, 0, 2],
        [3, 3, 0, 4],
        [0, 1, 0, 5],
    ];
    q2.extend(OUT_OF_RANGE.iter().map(|&l| [0, 0, 0, l]));
    level_case("levels_2d", "l2", TexIn::new(0, 2, 53, [8, 8, 1], 4, 4, r32_levels(8, 8, 1, 4)), &q2);
    // 16-bit coordinates and level: the level is converted to ushort in the function.
    let q2s = [[1, 1, 0, 0], [2, 1, 0, 0x1_0001], [0, 0, 0, 3], [0, 0, 0, 0xffff], [1, 1, 0, 0x2_0004]];
    level_case("levels_2d_short", "l2s", TexIn::new(0, 2, 53, [8, 8, 1], 4, 4, r32_levels(8, 8, 1, 4)), &q2s);
    // A view of levels 1 and 2: its level 0 is the parent's level 1; its level 2 would be the parent's level 3.
    let mut view = TexIn::new(0, 2, 53, [8, 8, 1], 4, 4, r32_levels(8, 8, 1, 4));
    view.view_levels = Some((1, 2));
    level_case(
        "levels_2d_view",
        "l2",
        view,
        &[[3, 2, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0], [2, 0, 0, 1], [0, 0, 0, 3], [1, 1, 0, 0xffff_ffff]],
    );
    let mut qa = vec![[1, 0, 1, 0], [0, 1, 2, 1], [0, 0, 0, 2], [3, 3, 2, 0], [0, 0, 3, 0], [0, 0, 0, 3]];
    qa.extend(OUT_OF_RANGE.iter().map(|&l| [1, 1, 1, l]));
    level_case("levels_2d_array", "la", TexIn::new(0, 3, 53, [4, 4, 3], 3, 4, r32_levels(4, 4, 3, 3)), &qa);
    let mut array_view = TexIn::new(0, 3, 53, [4, 4, 3], 3, 4, r32_levels(4, 4, 3, 3));
    array_view.view_levels = Some((1, 2));
    level_case(
        "levels_2d_array_view",
        "la",
        array_view,
        &[[1, 1, 2, 0], [0, 0, 1, 1], [0, 0, 0, 2], [2, 0, 0, 0], [0, 0, 3, 1]],
    );
    let mut q3 = vec![[1, 2, 3, 0], [1, 0, 1, 1], [0, 0, 0, 2], [0, 0, 1, 2], [0, 0, 0, 3]];
    q3.extend(OUT_OF_RANGE.iter().map(|&l| [0, 0, 0, l]));
    level_case("levels_3d", "l3", TexIn::new(0, 7, 53, [4, 4, 4], 3, 4, r32_volume(4, 4, 4, 3)), &q3);
    let mut qc = vec![[3, 0, 0, 0], [1, 1, 2, 1], [0, 0, 5, 2], [0, 0, 6, 1], [0, 0, 1, 3]];
    qc.extend(OUT_OF_RANGE.iter().map(|&l| [1, 1, 3, l]));
    level_case("levels_cube", "lc", TexIn::new(0, 5, 53, [4, 4, 1], 3, 4, r32_levels(4, 4, 6, 3)), &qc);
    // A float write converted to RGBA8Unorm at each level.
    let qf: Vec<[u32; 4]> = (0..3).flat_map(|l| (0..4 >> l).map(move |x| [x, 0, 0, l])).chain([[0, 0, 0, 3]]).collect();
    level_case("levels_rgba8", "lf", TexIn::new(0, 2, 70, [4, 4, 1], 3, 4, vec![0x11; 84]), &qf);
}

#[test]
fn writes_at_constant_levels_and_level_queries() {
    let run = |name: &str, t: TexIn| {
        let mut c = Case::new(name, WRITE_LEVELS, "lk");
        c.threads = [4, 1, 1];
        c.buffers = vec![vec![0; 4], vec![0; 12]];
        let mut t = t;
        t.write = true;
        c.textures = vec![t];
        // The writes bind level 2 alone; the write at level 9 is out of range and has no binding.
        let bc = air(&case_dir(name), WRITE_LEVELS, &[]);
        let compiled = compile_case(&c, &bc).unwrap();
        let mut levels: Vec<u32> = compiled
            .layout
            .groups
            .iter()
            .flat_map(|g| &g.entries)
            .filter_map(|e| match e.source {
                airwgsl::api::BindingSource::MetalTextureLevel { level, .. } => Some(level),
                _ => None,
            })
            .collect();
        levels.sort();
        // Level 0 serves the size queries.
        assert_eq!(levels, [0, 2], "{}: the level bindings", name);
        report(c.name, differential(&c));
    };
    run("levels_constant", TexIn::new(0, 2, 53, [8, 8, 1], 4, 4, r32_levels(8, 8, 1, 4)));
    let mut view = TexIn::new(0, 2, 53, [16, 16, 1], 5, 4, r32_levels(16, 16, 1, 5));
    view.view_levels = Some((1, 3));
    run("levels_constant_view", view);
}

#[test]
fn reads_through_level_range_views() {
    // Levels 1 to 3 of a 5-level texture: the view's level 3 is the parent's level 4, out of the view.
    let mut t = TexIn::new(0, 2, 53, [16, 16, 1], 5, 4, r32_levels(16, 16, 1, 5));
    t.view_levels = Some((1, 3));
    let mut c = Case::new("read_level_view", READ_2D, "k");
    let q: Vec<[u32; 4]> = vec![
        [0, 0, 0, 0],
        [7, 7, 0, 0],
        [3, 3, 1, 0],
        [1, 1, 2, 0],
        [0, 0, 3, 0],
        [0, 0, 0xffff_ffff, 0],
        [8, 0, 0, 0],
    ];
    c.threads = [q.len() as u64, 1, 1];
    c.buffers = vec![u32s(&q.concat()), vec![0; 33 * 16]];
    c.textures = vec![t];
    report(c.name, differential(&c));
}

#[test]
fn level_bindings_past_the_storage_texture_limit_are_refused() {
    let mut c = Case::new("levels_limit", WRITE_LEVELS, "l2");
    c.buffers = vec![vec![0; 16]];
    let mut t = TexIn::new(0, 2, 53, [4096, 4096, 1], 13, 4, Vec::new());
    t.write = true;
    c.textures = vec![t];
    let bc = air(&case_dir(c.name), WRITE_LEVELS, &[]);
    let mut h = host();
    h.limits.insert("maxStorageTexturesPerShaderStage".into(), 8);
    let r = compile_case_on(&c, &bc, &h).unwrap_err();
    assert_eq!(r.kind, airwgsl::api::RefusalKind::HostUnsupported, "{}", r);
    assert!(r.reason.starts_with("[compiler:texture-write-level]"), "{}", r);
    assert!(r.reason.contains("texture 0 binds 13 levels") && r.reason.contains("is 8"), "{}", r);
    h.limits.insert("maxStorageTexturesPerShaderStage".into(), 13);
    let compiled = compile_case_on(&c, &bc, &h).unwrap();
    let storage = compiled
        .layout
        .groups
        .iter()
        .flat_map(|g| &g.entries)
        .filter(|e| matches!(e.resource, airwgsl::api::BindingResource::StorageTexture { .. }))
        .count();
    assert_eq!(storage, 13);
}

const SAMPLE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void s2(texture2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
               device const float2 *c [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.sample(s, c[i]); }
kernel void s2o(texture2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
                device const float2 *c [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.sample(s, c[i], int2(-2, 3)); }
kernel void su(texture2d<uint, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
               device const float2 *c [[buffer(0)]], device uint4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.sample(s, c[i], int2(1, -1)); }
kernel void sk(texture2d<float, access::sample> t [[texture(0)]], device const float2 *c [[buffer(0)]],
               device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    constexpr sampler repeat(address::repeat);
    constexpr sampler mirrored(address::mirrored_repeat, filter::nearest);
    constexpr sampler zero(address::clamp_to_zero);
    constexpr sampler pixel(coord::pixel, address::clamp_to_zero);
    float2 p = c[i];
    o[4 * i + 0] = t.sample(repeat, p);
    o[4 * i + 1] = t.sample(mirrored, p);
    o[4 * i + 2] = t.sample(zero, p);
    o[4 * i + 3] = t.sample(pixel, p * 8.0);
}
"#;

fn sample_case(
    name: &str,
    function: &str,
    tex: TexIn,
    smp: Option<SamplerSlot>,
    coords: Vec<f32>,
    out_per: usize,
    elem: usize,
) {
    let n = coords.len() / 2;
    let mut c = Case::new(name, SAMPLE, function);
    c.threads = [n as u64, 1, 1];
    c.buffers = vec![f32s(&coords), vec![0; n * out_per * elem]];
    c.textures = vec![tex];
    c.samplers = smp.into_iter().collect();
    report(c.name, differential(&c));
}

#[test]
fn sample_address_modes_at_texel_centres() {
    let t = || TexIn::new(0, 2, 125, [8, 8, 1], 1, 16, rgba32_8x8());
    for (name, m) in [
        ("sample_clamp", [0, 0, 0]),
        ("sample_repeat", [2, 2, 2]),
        ("sample_mirror", [3, 3, 3]),
        ("sample_mixed", [2, 3, 0]),
        ("sample_zero", [4, 4, 4]),
        ("sample_zero_s", [4, 2, 0]),
    ] {
        sample_case(name, "s2", t(), Some(sampler(0, 0, 0, m, true, 0)), centres(), 1, 16);
    }
    sample_case("sample_offset", "s2o", t(), Some(sampler(0, 0, 0, [2, 3, 0], true, 0)), centres(), 1, 16);
    sample_case("sample_zero_offset", "s2o", t(), Some(sampler(0, 0, 0, [4, 4, 0], true, 0)), centres(), 1, 16);
    // R32Float: outside the texture clampToZero reads (0, 0, 0, 1), the components it lacks taking their defaults.
    let r: Vec<u8> = f32s(&(0..64).map(|i| 2.0 + i as f32).collect::<Vec<_>>());
    sample_case(
        "sample_zero_r32",
        "s2",
        TexIn::new(0, 2, 55, [8, 8, 1], 1, 4, r),
        Some(sampler(0, 0, 0, [4, 4, 4], true, 0)),
        centres(),
        1,
        16,
    );
}

#[test]
fn sample_unnormalized_integer_and_constant_samplers() {
    let pixel: Vec<f32> = centres().iter().map(|x| x * 8.0).collect();
    let t = || TexIn::new(0, 2, 125, [8, 8, 1], 1, 16, rgba32_8x8());
    sample_case("sample_pixel", "s2", t(), Some(sampler(0, 0, 0, [0, 0, 0], false, 0)), pixel.clone(), 1, 16);
    sample_case("sample_pixel_zero", "s2", t(), Some(sampler(0, 0, 0, [4, 4, 0], false, 0)), pixel, 1, 16);
    sample_case("sample_constexpr", "sk", t(), None, centres(), 4, 16);
    let u = r32_levels(8, 8, 1, 1);
    for (name, m) in
        [("sample_uint_repeat", [2, 2, 2]), ("sample_uint_mirror", [3, 0, 0]), ("sample_uint_zero", [4, 4, 4])]
    {
        sample_case(
            name,
            "su",
            TexIn::new(0, 2, 53, [8, 8, 1], 1, 4, u.clone()),
            Some(sampler(0, 0, 0, m, true, 0)),
            centres(),
            1,
            16,
        );
    }
}

const LEVELS: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(texture2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
              device const float4 *c [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[3 * i + 0] = t.sample(s, c[i].xy, level(c[i].z));
    o[3 * i + 1] = t.sample(s, c[i].xy, bias(c[i].z));
    o[3 * i + 2] = t.sample(s, c[i].xy);
}
"#;

#[test]
fn sample_levels_outside_fragment_functions() {
    // 8 x 8 R32Float with 4 levels, every texel of level L = 10 (L + 1); levels away from the halfway points.
    let mut data = Vec::new();
    for l in 0..4u32 {
        let w = 8 >> l;
        data.extend(std::iter::repeat_n(10.0 * (l + 1) as f32, (w * w) as usize));
    }
    let lv = [0.0f32, 0.25, 0.75, 1.0, 1.25, 1.75, 2.0, 3.0, 7.0, -2.0];
    let c4: Vec<f32> = lv.iter().flat_map(|l| [0.3, 0.6, *l, 0.0]).collect();
    let n = lv.len();
    let mut c = Case::new("sample_levels", LEVELS, "k");
    c.threads = [n as u64, 1, 1];
    c.buffers = vec![f32s(&c4), vec![0; n * 48]];
    c.textures = vec![TexIn::new(0, 2, 55, [8, 8, 1], 4, 4, f32s(&data))];
    c.samplers = vec![sampler(0, 0, 1, [0, 0, 0], true, 0)];
    report(c.name, differential(&c));
}

const GATHER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void g(texture2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
              device const float2 *c [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[5 * i + 0] = t.gather(s, c[i], int2(0), component::x);
    o[5 * i + 1] = t.gather(s, c[i], int2(0), component::y);
    o[5 * i + 2] = t.gather(s, c[i], int2(0), component::z);
    o[5 * i + 3] = t.gather(s, c[i], int2(0), component::w);
    o[5 * i + 4] = t.gather(s, c[i], int2(1, -2), component::x);
}
kernel void gu(texture2d<uint, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
               device const float2 *c [[buffer(0)]], device uint4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.gather(s, c[i], int2(0), component::x); }
"#;

#[test]
fn gathers() {
    // Gather coordinates between texel centres (texel corners) and at centres.
    let mut coords = centres();
    for k in [0, 1, 4, 7, 8, 9] {
        coords.extend_from_slice(&[k as f32 / 8.0, (k + 2) as f32 / 8.0]);
    }
    let n = coords.len() / 2;
    for (name, m) in [("gather_clamp", [0, 0, 0]), ("gather_repeat", [2, 3, 0]), ("gather_zero", [4, 4, 0])] {
        let mut c = Case::new(name, GATHER, "g");
        c.threads = [n as u64, 1, 1];
        c.buffers = vec![f32s(&coords), vec![0; n * 80]];
        c.textures = vec![TexIn::new(0, 2, 125, [8, 8, 1], 1, 16, rgba32_8x8())];
        c.samplers = vec![sampler(0, 0, 0, m, true, 0)];
        report(c.name, differential(&c));
    }
    let mut c = Case::new("gather_uint", GATHER, "gu");
    c.threads = [n as u64, 1, 1];
    c.buffers = vec![f32s(&coords), vec![0; n * 16]];
    c.textures = vec![TexIn::new(0, 2, 53, [8, 8, 1], 1, 4, r32_levels(8, 8, 1, 1))];
    c.samplers = vec![sampler(0, 0, 0, [2, 2, 0], true, 0)];
    report(c.name, differential(&c));
}

const DEPTH: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(depth2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
              device const float4 *q [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[2 * i] = float4(t.sample_compare(s, q[i].xy, q[i].z), t.sample_compare(s, q[i].xy, q[i].z, level(0)), 0, 0);
    o[2 * i + 1] = t.gather_compare(s, q[i].xy, q[i].z);
}
kernel void p(depth2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
              device const float4 *q [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = float4(t.sample(s, q[i].xy), t.sample(s, q[i].xy, level(1)), t.gather(s, q[i].xy).xw); }
kernel void r(depth2d<float, access::read> t [[texture(0)]], device const uint4 *q [[buffer(0)]],
              device float *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(q[i].xy, q[i].z); }
"#;

#[test]
fn depth_compare_gather_and_read() {
    // Depth16Unorm: WebGPU uploads depth from buffers only into depth16unorm (WebMetal formats.toml question
    // 6.6-depth-copies); the depth of texel i is code 1024 i + 512.
    let d: Vec<u8> = (0..64u16).flat_map(|i| (1024 * i + 512).to_le_bytes()).collect();
    let mut q = Vec::new();
    for (k, r) in [(0, 0.2f32), (9, 0.5), (27, 0.43), (63, 0.99), (40, 0.0), (5, 1.0)] {
        q.extend_from_slice(&[((k % 8) as f32 + 0.5) / 8.0, ((k / 8) as f32 + 0.5) / 8.0, r, 0.0]);
        q.extend_from_slice(&[(k % 8) as f32 / 8.0, (k / 8) as f32 / 8.0, r, 0.0]);
    }
    let n = q.len() / 4;
    for (name, f) in [("depth_less", 1u64), ("depth_greater_equal", 6), ("depth_never", 0)] {
        let mut c = Case::new(name, DEPTH, "k");
        c.threads = [n as u64, 1, 1];
        c.buffers = vec![f32s(&q), vec![0; n * 32]];
        c.textures = vec![TexIn::new(0, 2, 250, [8, 8, 1], 1, 2, d.clone())];
        c.samplers = vec![sampler(0, 0, 0, [0, 0, 0], true, f)];
        report(c.name, differential(&c));
    }
    let mut c = Case::new("depth_plain", DEPTH, "p");
    c.threads = [n as u64, 1, 1];
    c.buffers = vec![f32s(&q), vec![0; n * 16]];
    let mut two = d.clone();
    two.extend((0..16u16).flat_map(|i| (4000 * i).to_le_bytes()));
    c.textures = vec![TexIn::new(0, 2, 250, [8, 8, 1], 2, 2, two)];
    c.samplers = vec![sampler(0, 0, 1, [2, 0, 0], true, 0)];
    report(c.name, differential(&c));
    let mut c = Case::new("depth_read", DEPTH, "r");
    let rq: Vec<u32> =
        [[0u32, 0, 0, 0], [7, 7, 0, 0], [8, 0, 0, 0], [0, 0, 1, 0], [3, 4, 0, 0]].iter().flatten().copied().collect();
    c.threads = [5, 1, 1];
    c.buffers = vec![u32s(&rq), vec![0; 20]];
    c.textures = vec![TexIn::new(0, 2, 250, [8, 8, 1], 1, 2, d)];
    report(c.name, differential(&c));
}

const SWIZZLE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(texture2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
              device const float2 *c [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[4 * i + 0] = t.sample(s, c[i]);
    o[4 * i + 1] = t.read(uint2(c[i] * 8.0) % 8);
    o[4 * i + 2] = t.gather(s, c[i], int2(0), component::y);
    o[4 * i + 3] = t.gather(s, c[i], int2(0), component::w);
}
"#;

#[test]
fn swizzled_views_and_shader_formats() {
    let coords = centres();
    let n = coords.len() / 2;
    let run = |name: &str, format: u64, bpp: u32, data: Vec<u8>, swizzle: [u8; 4]| {
        let mut c = Case::new(name, SWIZZLE, "k");
        c.threads = [n as u64, 1, 1];
        c.buffers = vec![f32s(&coords), vec![0; n * 64]];
        let mut t = TexIn::new(0, 2, format, [8, 8, 1], 1, bpp, data);
        t.swizzle = swizzle;
        c.textures = vec![t];
        c.samplers = vec![sampler(0, 0, 0, [2, 2, 2], true, 0)];
        let srgb = matches!(format, 11 | 31);
        report(c.name, if srgb { differential_with(&c, &srgb_bound) } else { differential(&c) });
    };
    let bytes = |bpp: usize| -> Vec<u8> { (0..64 * bpp).map(|i| (i * 53 % 256) as u8).collect() };
    run("swizzle_rgba32", 125, 16, rgba32_8x8(), [5, 1, 2, 0]);
    run("swizzle_rgba8", 70, 4, bytes(4), [4, 4, 3, 2]);
    run("format_a8", 1, 1, bytes(1), [2, 3, 4, 5]);
    run("format_a8_swizzled", 1, 1, bytes(1), [5, 2, 1, 5]);
    run("format_r8_srgb", 11, 1, bytes(1), [2, 3, 4, 5]);
    run("format_rg8_srgb", 31, 2, bytes(2), [2, 3, 4, 5]);
    run("format_b5g6r5", 40, 2, bytes(2), [2, 3, 4, 5]);
    run("format_a1bgr5", 41, 2, bytes(2), [2, 3, 4, 5]);
    run("format_abgr4", 42, 2, bytes(2), [2, 3, 4, 5]);
    run("format_bgr5a1", 43, 2, bytes(2), [2, 3, 4, 5]);
}

const NULL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void k(texture2d<float> a [[texture(0)]], texture2d<float> b [[texture(1)]], device uint *o [[buffer(0)]])
{ o[0] = is_null_texture(a); o[1] = is_null_texture(b); }
"#;

#[test]
fn null_textures() {
    let mut c = Case::new("null", NULL, "k");
    c.buffers = vec![vec![0xee; 8]];
    c.textures = vec![TexIn::new(0, 2, 70, [1, 1, 1], 1, 4, vec![1, 2, 3, 4])];
    c.nulls = vec![1];
    report(c.name, differential(&c));
}

const OTHER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void s1(texture1d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
               device const float4 *c [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.sample(s, c[i].x); }
kernel void s3(texture3d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
               device const float4 *c [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.sample(s, c[i].xyz); }
kernel void sa(texture2d_array<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
               device const float4 *c [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.sample(s, c[i].xy, uint(c[i].z)); }
kernel void sc(texturecube<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
               device const float4 *c [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{ o[i] = t.sample(s, c[i].xyz); }
"#;

#[test]
fn sample_1d_3d_arrays_and_cubes() {
    let run = |name: &str, f: &str, t: TexIn, m: [u64; 3], coords: Vec<f32>| {
        let n = coords.len() / 4;
        let mut c = Case::new(name, OTHER, f);
        c.threads = [n as u64, 1, 1];
        c.buffers = vec![f32s(&coords), vec![0; n * 16]];
        c.textures = vec![t];
        c.samplers = vec![sampler(0, 0, 0, m, true, 0)];
        report(c.name, differential(&c));
    };
    let vals = |n: usize| f32s(&(0..n * 4).map(|i| i as f32).collect::<Vec<_>>());
    let c1: Vec<f32> = [-0.6f32, 0.0625, 0.4375, 0.9375, 1.3].iter().flat_map(|x| [*x, 0.0, 0.0, 0.0]).collect();
    run("sample_1d", "s1", TexIn::new(0, 0, 125, [8, 1, 1], 1, 16, vals(8)), [2, 0, 0], c1.clone());
    run("sample_1d_zero", "s1", TexIn::new(0, 0, 125, [8, 1, 1], 1, 16, vals(8)), [4, 0, 0], c1);
    let c3: Vec<f32> = [[0.125f32, 0.375, 0.625, 0.0], [0.875, 0.125, 1.375, 0.0], [-0.125, 0.625, 0.375, 0.0]]
        .iter()
        .flatten()
        .copied()
        .collect();
    run("sample_3d", "s3", TexIn::new(0, 7, 125, [4, 4, 4], 1, 16, vals(64)), [0, 2, 3], c3.clone());
    run("sample_3d_zero", "s3", TexIn::new(0, 7, 125, [4, 4, 4], 1, 16, vals(64)), [4, 4, 4], c3);
    let ca: Vec<f32> = [[0.125f32, 0.375, 0.0, 0.0], [0.875, 0.625, 1.0, 0.0], [0.375, 0.125, 2.0, 0.0]]
        .iter()
        .flatten()
        .copied()
        .collect();
    run("sample_2d_array", "sa", TexIn::new(0, 3, 125, [4, 4, 3], 1, 16, vals(48)), [0, 0, 0], ca);
    // Face centres and points well inside each face (MSL 4.1 Table 6.23 face order).
    let cc: Vec<f32> = [
        [1.0f32, 0.3, -0.2, 0.0],
        [-1.0, -0.4, 0.1, 0.0],
        [0.2, 1.0, 0.3, 0.0],
        [0.1, -1.0, -0.3, 0.0],
        [0.3, 0.2, 1.0, 0.0],
        [-0.1, 0.35, -1.0, 0.0],
    ]
    .iter()
    .flatten()
    .copied()
    .collect();
    run("sample_cube", "sc", TexIn::new(0, 5, 125, [4, 4, 1], 1, 16, vals(96)), [0, 0, 0], cc);
}

/// The sRGB encoding of MSL 4.1 section 8.7.7, unrounded, in code units.
fn srgb_encode(c: f64) -> f64 {
    255.0 * if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// MSL 4.1 section 8.7.7's bound for sRGB-to-linear conversion: the result encoded back without rounding is within
/// 0.5 of the original code. Lanes that differ must both meet it for the code Metal's value encodes to; all other bytes
/// are compared exactly.
fn srgb_bound(a: &[Vec<u8>], b: &[Vec<u8>]) -> Option<String> {
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        for (i, (p, q)) in x.chunks(4).zip(y.chunks(4)).enumerate() {
            if p == q {
                continue;
            }
            let m = f32::from_le_bytes(p.try_into().unwrap()) as f64;
            let w = f32::from_le_bytes(q.try_into().unwrap()) as f64;
            let code = srgb_encode(m).round();
            if !(0.0..=1.0).contains(&m) || (srgb_encode(m) - code).abs() > 0.5 || (srgb_encode(w) - code).abs() > 0.5 {
                return Some(format!(
                    "output {} lane {}: Metal {} and WebGPU {} are not one sRGB code's decode",
                    k, i, m, w
                ));
            }
        }
    }
    None
}

const LAYOUTS: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void ms(texture2d_ms<float, access::read> t [[texture(0)]], device float4 *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(uint2(i, 0), i & 3) + float(t.get_num_samples()); }
kernel void msu(texture2d_ms<uint, access::read> t [[texture(0)]], device uint4 *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(uint2(i, 0), i & 3); }
kernel void dms(depth2d_ms<float, access::read> t [[texture(0)]], device float *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(uint2(i, 0), i & 3); }
kernel void d(depth2d<float> t [[texture(0)]], sampler s [[sampler(0)]], device float *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{ o[i] = t.sample(s, float2(i)) + t.read(uint2(i, 0)) + t.gather(s, float2(i)).x; }
kernel void dc(depthcube<float> t [[texture(0)]], sampler s [[sampler(0)]], device float *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{ o[i] = t.sample_compare(s, float3(1, 0, 0), 0.5) + t.get_width(); }
kernel void ddf(texture2d<float, access::read> t [[texture(0)]], device float4 *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{ o[i] = t.read(uint2(i, 0)); }
"#;

/// The compiled layouts of multisampled and depth textures (null, and bound with formats of each class) pass WebGPU's
/// validation of bind group layouts and bind groups (WebGPU 8.1.1, 8.3.1): a multisampled float binding is
/// "unfilterable-float", never "float", and depth formats bind as "depth" or, through float textures,
/// "unfilterable-float".
#[test]
fn multisampled_and_depth_layouts_are_valid() {
    let dir = case_dir("layouts");
    let bc = air(&dir, LAYOUTS, &[]);
    let ms = |format: u64| {
        let mut t = TexIn::new(0, 4, format, [4, 4, 1], 1, 4, Vec::new());
        t.samples = 4;
        t
    };
    let cases: Vec<(&str, Vec<TexIn>, Vec<SamplerSlot>)> = vec![
        ("ms", vec![], vec![]),
        ("ms", vec![ms(70)], vec![]),
        ("ms", vec![ms(115)], vec![]),
        ("ms", vec![ms(55)], vec![]),
        ("ms", vec![ms(252)], vec![]),
        ("msu", vec![], vec![]),
        ("msu", vec![ms(73)], vec![]),
        ("dms", vec![], vec![]),
        ("dms", vec![ms(252)], vec![]),
        ("d", vec![TexIn::new(0, 2, 252, [4, 4, 1], 1, 4, Vec::new())], vec![]),
        ("dc", vec![TexIn::new(0, 5, 250, [4, 4, 1], 1, 2, Vec::new())], vec![sampler(0, 1, 0, [0, 0, 0], true, 1)]),
        ("ddf", vec![TexIn::new(0, 2, 252, [4, 4, 1], 1, 4, Vec::new())], vec![]),
        ("ddf", vec![TexIn::new(0, 2, 250, [4, 4, 1], 1, 2, Vec::new())], vec![]),
    ];
    for (f, textures, samplers) in cases {
        let mut c = Case::new("layouts", LAYOUTS, f);
        c.buffers = vec![vec![0; 64]];
        let bound = !textures.is_empty();
        c.textures = textures;
        c.samplers = samplers;
        if !bound {
            // Null textures: the layout alone is validated.
            c.nulls = vec![0];
        }
        let compiled = compile_case(&c, &bc).unwrap_or_else(|r| panic!("{}: {}", f, r));
        for g in &compiled.layout.groups {
            for e in &g.entries {
                if let airwgsl::api::BindingResource::Texture { sample_type, multisampled: true, .. } = &e.resource {
                    assert_ne!(*sample_type, airwgsl::api::TextureSampleType::Float, "{}: {:?}", f, e);
                }
            }
        }
        if bound && let Some(e) = validate_bindings(&c, &compiled) {
            panic!("{} ({} bound): WebGPU validation: {}\n{}", f, c.textures[0].format, e, compiled.wgsl[0].source);
        }
    }
}
