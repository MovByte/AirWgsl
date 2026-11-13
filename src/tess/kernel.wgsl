// The tessellator on the GPU: the same computation as src/tess/factors.rs and src/tess/hw.rs, one invocation per
// patch of a draw (instance-major). Every step is integer arithmetic, binary32 values included (as bit patterns),
// so the results are exact on every WebGPU implementation (WGSL leaves infinities and NaNs of float arithmetic
// indeterminate).
//
// Three entry points run in order, each in its own dispatch:
//   tess_count:    the points and indices of each patch, and whether any patch has a batched blade;
//   tess_scan:     one workgroup: each patch's first point and index, and the drawIndexedIndirect arguments;
//   tess_generate: each patch's point records and indices at its place.

struct Params {
    // 0 triangle, 1 quad.
    quad: u32,
    // MTLTessellationPartitionMode.
    partition_mode: u32,
    // Metal's clamp of the processed factors, binary32 bit patterns.
    lower: u32,
    upper: u32,
    // tessellationOutputWindingOrder is MTLWindingClockwise.
    clockwise: u32,
    // tessellationFactorScaleEnabled, and the scale as a binary16 bit pattern.
    scale_enabled: u32,
    scale: u32,
    // MTLTessellationFactorStepFunction.
    step: u32,
    // Byte offset of the draw's factors in factor_words (the Metal offset less the binding's aligned base).
    factor_offset: u32,
    instance_stride: u32,
    patch_start: u32,
    patch_count: u32,
    base_instance: u32,
    // Patches times instances.
    slots: u32,
    // Whether the draw has a patch index buffer, and the byte offset of its first element in patch_index_words.
    indexed: u32,
    patch_index_offset: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> factor_words: array<u32>;
@group(0) @binding(2) var<storage, read> patch_index_words: array<u32>;
// Per patch: its points and indices (tess_count), then its first point and first index (tess_scan).
@group(0) @binding(3) var<storage, read_write> slot_words: array<u32>;
// Per point: u and v in 16.16 fixed point, the patch identifier, the instance identifier.
@group(0) @binding(4) var<storage, read_write> vertices: array<u32>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;
// drawIndexedIndirect arguments (index count, instance count, first index, base vertex, first instance), then the
// batched flag and the point count.
@group(0) @binding(6) var<storage, read_write> draw_words: array<atomic<u32>>;

const FXP_ONE: u32 = 0x10000u;
const FXP_ONE_THIRD: u32 = 0x5555u;
const FXP_TWO_THIRDS: u32 = 0xaaaau;
const FXP_ONE_HALF: u32 = 0x8000u;
const F32_ONE: u32 = 0x3f800000u;
// 1 + 2^-16 and 1 + 2^-17.
const F32_ONE_PLUS_EPSILON: u32 = 0x3f800080u;
const F32_ONE_PLUS_HALF_EPSILON: u32 = 0x3f800040u;
const WHOLE_BLADE_POINTS: i32 = 36;

const FIXED_RECIPROCAL = array<u32, 65>(
    0xffffffffu, 0x10000u, 0x8000u, 0x5555u, 0x4000u, 0x3333u, 0x2aabu, 0x2492u, 0x2000u, 0x1c72u, 0x199au, 0x1746u,
    0x1555u, 0x13b1u, 0x1249u, 0x1111u, 0x1000u, 0xf0fu, 0xe39u, 0xd79u, 0xccdu, 0xc31u, 0xba3u, 0xb21u, 0xaabu,
    0xa3du, 0x9d9u, 0x97bu, 0x925u, 0x8d4u, 0x889u, 0x842u, 0x800u, 0x7c2u, 0x788u, 0x750u, 0x71cu, 0x6ebu, 0x6bdu,
    0x690u, 0x666u, 0x63eu, 0x618u, 0x5f4u, 0x5d1u, 0x5b0u, 0x591u, 0x572u, 0x555u, 0x539u, 0x51fu, 0x505u, 0x4ecu,
    0x4d5u, 0x4beu, 0x4a8u, 0x492u, 0x47eu, 0x46au, 0x457u, 0x444u, 0x432u, 0x421u, 0x410u, 0x400u);
const FINAL_POINT_POSITION = array<i32, 33>(
    0, 32, 16, 8, 17, 4, 18, 9, 19, 2, 20, 10, 21, 5, 22, 11, 23, 1, 24, 12, 25, 6, 26, 13, 27, 3, 28, 14, 29, 7, 30,
    15, 31);
const LOOP_START = array<i32, 33>(
    1, 1, 17, 9, 9, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
const LOOP_END = array<i32, 33>(
    0, 0, 17, 17, 25, 25, 25, 25, 29, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
    31, 31, 31, 32);

// Factors -------------------------------------------------------------------------------------------------------------

// A binary16 value as a binary32 bit pattern, exactly.
fn half_bits(h: u32) -> u32 {
    let sign = (h >> 15u) << 31u;
    let e = (h >> 10u) & 0x1fu;
    var m = h & 0x3ffu;
    if (e == 0u) {
        if (m == 0u) {
            return sign;
        }
        var x = 113u; // 127 - 15 + 1
        loop {
            if ((m & 0x400u) != 0u) {
                break;
            }
            m = m << 1u;
            x = x - 1u;
        }
        return sign | (x << 23u) | ((m & 0x3ffu) << 13u);
    }
    if (e == 0x1fu) {
        return sign | 0x7f800000u | (m << 13u);
    }
    return sign | ((e + 112u) << 23u) | (m << 13u);
}

fn is_nan(f: u32) -> bool {
    return (f & 0x7f800000u) == 0x7f800000u && (f & 0x7fffffu) != 0u;
}

// The product of two binary32 values that came from binary16 ones, which binary32 holds exactly.
fn mul_bits(a: u32, b: u32) -> u32 {
    let sign = (a ^ b) & 0x80000000u;
    if (is_nan(a) || is_nan(b)) {
        return 0x7fc00000u;
    }
    let za = (a & 0x7fffffffu) == 0u;
    let zb = (b & 0x7fffffffu) == 0u;
    let ia = (a & 0x7fffffffu) == 0x7f800000u;
    let ib = (b & 0x7fffffffu) == 0x7f800000u;
    if (ia || ib) {
        if (za || zb) {
            return 0x7fc00000u;
        }
        return sign | 0x7f800000u;
    }
    if (za || zb) {
        return sign;
    }
    // Both normal with at most 11 significant bits: the 22-bit product is exact.
    let sa = ((a & 0x7fffffu) | 0x800000u) >> 13u;
    let sb = ((b & 0x7fffffu) | 0x800000u) >> 13u;
    let p = sa * sb;
    var e = i32((a >> 23u) & 0xffu) + i32((b >> 23u) & 0xffu) - 127;
    var frac: u32;
    if (p >= 0x200000u) {
        frac = (p - 0x200000u) << 2u;
        e = e + 1;
    } else {
        frac = (p - 0x100000u) << 3u;
    }
    return sign | (u32(e) << 23u) | frac;
}

// Metal's clamp: NaN and anything below `lower` to `lower`, anything above `upper` to `upper`.
fn clamp_bits(f: u32) -> u32 {
    if (is_nan(f) || (f & 0x80000000u) != 0u || f < params.lower) {
        return params.lower;
    }
    return min(f, params.upper);
}

// The ceiling of a binary32 value of at least 1.
fn ceil_bits(f: u32) -> u32 {
    let e = (f >> 23u) - 127u;
    if (e >= 23u) {
        return f;
    }
    let mask = (1u << (23u - e)) - 1u;
    if ((f & mask) == 0u) {
        return f;
    }
    return (f & ~mask) + (mask + 1u);
}

// The integer part of a binary32 value of at least 1.
fn trunc_bits(f: u32) -> u32 {
    let e = (f >> 23u) - 127u;
    return ((f & 0x7fffffu) | 0x800000u) >> (23u - e);
}

// The reference's floatToIDotF<15, 16, false> for a binary32 value of at least 1 and below 2^15: round to nearest even.
fn float_to_fixed(f: u32) -> u32 {
    let unbiased = i32((f >> 23u) & 0xffu) - 127;
    var output = (f & 0x7fffffu) | 0x800000u;
    let extra = 7 - unbiased;
    if (extra >= 0) {
        let lsb = 1u << u32(extra);
        let mask = lsb - 1u;
        let half = lsb >> 1u;
        if ((output & lsb) != 0u || (output & mask) > half) {
            output = output + half;
        }
        return output >> u32(extra);
    }
    return output << u32(-extra);
}

fn read_half(byte_address: u32) -> u32 {
    let w = factor_words[byte_address >> 2u];
    return select(w & 0xffffu, w >> 16u, (byte_address & 2u) != 0u);
}

// Factors -------------------------------------------------------------------------------------------------------------

struct Patch {
    culled: bool,
    minimum: bool,
    // Partitioning of the tessellator: 0 integer (Metal's pow2 and integer), 1 fractional odd, 2 fractional even.
    partitioning: u32,
    outside: array<u32, 4>,
    inside: array<u32, 2>,
    outside_odd: array<bool, 4>,
    inside_odd: array<bool, 2>,
    outside_points: array<i32, 4>,
    inside_points: array<i32, 2>,
    inside_edge_point_base: i32,
    points: i32,
}

struct Ctx {
    inv_floor: u32,
    inv_ceil: u32,
    fraction: u32,
    half_points: i32,
    split: i32,
}

var<private> parity_odd: bool;
var<private> pf: Patch;
var<private> outside_ctx: array<Ctx, 4>;
var<private> inside_ctx: array<Ctx, 2>;
var<private> point_base: u32;
var<private> index_base: u32;
var<private> slot_patch: u32;
var<private> slot_instance: u32;
var<private> generate: bool;
// The reference's index patching (m_IndexPatchContext, m_IndexPatchContext2).
var<private> patched: bool;
var<private> patched2: bool;
var<private> p_inside_delta: i32;
var<private> p_inside_bad: i32;
var<private> p_inside_replacement: i32;
var<private> p_outside_base: i32;
var<private> p_outside_delta: i32;
var<private> p_outside_bad: i32;
var<private> p_outside_replacement: i32;
var<private> p2_base: i32;
var<private> p2_end: i32;
var<private> p2_bad: i32;
var<private> p2_replacement: i32;

fn remove_msb(val: i32) -> i32 {
    var check: i32;
    if (val <= 0xffff) {
        check = select(0x8000, 0x80, val <= 0xff);
    } else {
        check = select(i32(0x80000000u), 0x800000, val <= 0xffffff);
    }
    for (var i = 0; i < 8; i++) {
        if ((val & check) != 0) {
            return val & ~check;
        }
        check = i32(u32(check) >> 1u);
    }
    return 0;
}

fn fxp_ceil(x: u32) -> u32 {
    if ((x & 0xffffu) != 0u) {
        return (x & 0x7fff0000u) + FXP_ONE;
    }
    return x;
}

fn num_points_for(f: u32) -> i32 {
    if (parity_odd) {
        return i32((fxp_ceil(FXP_ONE_HALF + (f + 1u) / 2u) * 2u) >> 16u);
    }
    return i32((fxp_ceil((f + 1u) / 2u) * 2u) >> 16u) + 1;
}

fn factor_context(f: u32) -> Ctx {
    var c: Ctx;
    var half = (f + 1u) / 2u;
    if (parity_odd || half == FXP_ONE_HALF) {
        half = half + FXP_ONE_HALF;
    }
    let floor_half = half & 0x7fff0000u;
    let ceil_half = fxp_ceil(half);
    c.fraction = half - floor_half;
    c.half_points = i32(ceil_half >> 16u);
    if (ceil_half == floor_half) {
        c.split = c.half_points + 1;
    } else if (parity_odd) {
        if (floor_half == FXP_ONE) {
            c.split = 0;
        } else {
            c.split = (remove_msb(i32(floor_half >> 16u) - 1) << 1u) + 1;
        }
    } else {
        c.split = (remove_msb(i32(floor_half >> 16u)) << 1u) + 1;
    }
    var floor_segments = i32((floor_half * 2u) >> 16u);
    var ceil_segments = i32((ceil_half * 2u) >> 16u);
    if (parity_odd) {
        floor_segments = floor_segments - 1;
        ceil_segments = ceil_segments - 1;
    }
    c.inv_floor = FIXED_RECIPROCAL[floor_segments];
    c.inv_ceil = FIXED_RECIPROCAL[ceil_segments];
    return c;
}

fn place(c: Ctx, point_in: i32) -> u32 {
    var point = point_in;
    var flip = false;
    if (point >= c.half_points) {
        point = (c.half_points << 1u) - point;
        if (parity_odd) {
            point = point - 1;
        }
        flip = true;
    }
    if (point == c.half_points) {
        return FXP_ONE_HALF;
    }
    let on_ceil = u32(point);
    var on_floor = on_ceil;
    if (point > c.split) {
        on_floor = on_floor - 1u;
    }
    var location = (on_floor * c.inv_floor) * (FXP_ONE - c.fraction) + (on_ceil * c.inv_ceil) * c.fraction;
    location = (location + FXP_ONE_HALF) >> 16u;
    if (flip) {
        location = FXP_ONE - location;
    }
    return location;
}

// The patch's processed factors and sizes (factors.rs process, then hw.rs quad_process or tri_process).
fn process(halves: array<u32, 6>) {
    pf.culled = false;
    pf.minimum = false;
    let edges = select(3u, 4u, params.quad != 0u);
    let insides = select(1u, 2u, params.quad != 0u);
    var f: array<u32, 6>;
    for (var i = 0u; i < 6u; i++) {
        f[i] = half_bits(halves[i]);
        if (params.scale_enabled != 0u) {
            f[i] = mul_bits(f[i], half_bits(params.scale));
        }
    }
    for (var i = 0u; i < edges; i++) {
        if (is_nan(f[i]) || (f[i] & 0x80000000u) != 0u || (f[i] & 0x7fffffffu) == 0u) {
            pf.culled = true;
            return;
        }
    }
    var outside: array<u32, 4>;
    var inside: array<u32, 2>;
    for (var i = 0u; i < 6u; i++) {
        var v = clamp_bits(f[i]);
        if (params.partition_mode == 0u && (v & 0x7fffffu) != 0u) {
            v = (v & 0x7f800000u) + 0x800000u;
        }
        if (i < edges) {
            outside[i] = v;
        } else if (i < edges + insides) {
            inside[i - edges] = v;
        }
    }
    let integer = params.partition_mode <= 1u;
    pf.partitioning = select(params.partition_mode - 1u, 0u, integer);
    let odd_partitioning = pf.partitioning == 1u;
    if (integer) {
        for (var i = 0u; i < 4u; i++) {
            outside[i] = ceil_bits(outside[i]);
        }
    }
    if (odd_partitioning) {
        // Force a picture frame when any factor exceeds 1 (a triangle's inside factor is not consulted).
        var frame = false;
        for (var i = 0u; i < edges; i++) {
            frame = frame || outside[i] > F32_ONE_PLUS_HALF_EPSILON;
        }
        if (params.quad != 0u) {
            frame = frame || inside[0] > F32_ONE_PLUS_HALF_EPSILON || inside[1] > F32_ONE_PLUS_HALF_EPSILON;
        }
        if (frame) {
            inside[0] = max(inside[0], F32_ONE_PLUS_EPSILON);
            inside[1] = max(inside[1], F32_ONE_PLUS_EPSILON);
        }
    }
    if (integer) {
        inside[0] = ceil_bits(inside[0]);
        inside[1] = ceil_bits(inside[1]);
    }
    for (var i = 0u; i < 4u; i++) {
        pf.outside_odd[i] = select(odd_partitioning, (trunc_bits(outside[i]) & 1u) != 0u, integer);
        pf.outside[i] = float_to_fixed(outside[i]);
    }
    for (var i = 0u; i < 2u; i++) {
        pf.inside_odd[i] =
            select(odd_partitioning, (trunc_bits(inside[i]) & 1u) != 0u && inside[i] != F32_ONE, integer);
        pf.inside[i] = float_to_fixed(inside[i]);
    }
    if (integer || odd_partitioning) {
        var all_one = true;
        for (var i = 0u; i < edges; i++) {
            all_one = all_one && pf.outside[i] == FXP_ONE;
        }
        for (var i = 0u; i < insides; i++) {
            all_one = all_one && pf.inside[i] == FXP_ONE;
        }
        if (all_one) {
            pf.minimum = true;
            pf.points = i32(edges);
            return;
        }
    }
    for (var i = 0u; i < edges; i++) {
        parity_odd = pf.outside_odd[i];
        outside_ctx[i] = factor_context(pf.outside[i]);
    }
    for (var i = 0u; i < insides; i++) {
        parity_odd = pf.inside_odd[i];
        inside_ctx[i] = factor_context(pf.inside[i]);
    }
    var points = 0;
    for (var i = 0u; i < edges; i++) {
        parity_odd = pf.outside_odd[i];
        pf.outside_points[i] = num_points_for(pf.outside[i]);
        points = points + pf.outside_points[i];
    }
    points = points - i32(edges);
    for (var i = 0u; i < insides; i++) {
        parity_odd = pf.inside_odd[i];
        pf.inside_points[i] = max(num_points_for(pf.inside[i]), select(3, 4, pf.inside_odd[i]));
    }
    pf.inside_edge_point_base = points;
    if (params.quad != 0u) {
        points = points + (pf.inside_points[0] - 2) * (pf.inside_points[1] - 2);
    } else {
        let n = pf.inside_points[0];
        let rings = (n >> 1u) - 1;
        if (pf.inside_odd[0]) {
            points = points + 3 * (rings * (rings + 1) - rings);
        } else {
            points = points + 3 * (rings * (rings + 1)) + 1;
        }
    }
    pf.points = points;
}

// Output --------------------------------------------------------------------------------------------------------------

fn define_point(u: u32, v: u32, k: i32) {
    if (generate) {
        let at = (point_base + u32(k)) * 4u;
        vertices[at] = u;
        vertices[at + 1u] = v;
        vertices[at + 2u] = slot_patch;
        vertices[at + 3u] = slot_instance;
    }
}

fn patch_index(index_in: i32) -> i32 {
    var index = index_in;
    if (patched) {
        if (index >= p_outside_base) {
            if (index == p_outside_bad) {
                index = p_outside_replacement;
            } else {
                index = index + p_outside_delta;
            }
        } else if (index == p_inside_bad) {
            index = p_inside_replacement;
        } else {
            index = index + p_inside_delta;
        }
    } else if (patched2) {
        if (index >= p2_base) {
            if (index == p2_bad) {
                index = p2_replacement;
            } else {
                index = p2_end - index;
            }
        } else if (index == p2_bad) {
            index = p2_replacement;
        }
    }
    return index;
}

fn define_index(index: i32, at: i32) {
    if (generate) {
        indices[index_base + u32(at)] = point_base + u32(patch_index(index));
    }
}

// A triangle in the reference's clockwise vertex order: stored for Metal's winding (MTLWindingClockwise is the
// reference's counter-clockwise output).
fn reference_triangle(i0: i32, i1: i32, i2: i32, at: i32) {
    define_index(i0, at);
    if (params.clockwise != 0u) {
        define_index(i2, at + 1);
        define_index(i1, at + 2);
    } else {
        define_index(i1, at + 1);
        define_index(i2, at + 2);
    }
}

// A triangle in the vertex order of MTLWindingClockwise.
fn metal_triangle(x: i32, y: i32, z: i32, at: i32) {
    define_index(x, at);
    if (params.clockwise != 0u) {
        define_index(y, at + 1);
        define_index(z, at + 2);
    } else {
        define_index(z, at + 1);
        define_index(y, at + 2);
    }
}

fn stitch_transition(base_in: i32, inside_base: i32, inside_half_in: i32, inside_odd: bool, outside_base: i32,
                     outside_half_in: i32, outside_odd: bool) {
    var base = base_in;
    let inside_half = select(inside_half_in, inside_half_in - 1, inside_odd);
    let outside_half = select(outside_half_in, outside_half_in - 1, outside_odd);
    var outside = outside_base;
    var inside = inside_base;
    let i_start = min(LOOP_START[inside_half], LOOP_START[outside_half]);
    let i_end = max(LOOP_END[inside_half], LOOP_END[outside_half]);
    if (FINAL_POINT_POSITION[0] < outside_half) {
        reference_triangle(outside, outside + 1, inside, base);
        base = base + 3;
        outside = outside + 1;
    }
    for (var i = i_start; i <= i_end; i++) {
        if (FINAL_POINT_POSITION[i] < inside_half) {
            reference_triangle(inside, outside, inside + 1, base);
            base = base + 3;
            inside = inside + 1;
        }
        if (FINAL_POINT_POSITION[i] < outside_half) {
            reference_triangle(outside, outside + 1, inside, base);
            base = base + 3;
            outside = outside + 1;
        }
    }
    if (inside_odd != outside_odd || inside_odd) {
        if (inside_odd == outside_odd) {
            reference_triangle(inside, outside, inside + 1, base);
            base = base + 3;
            reference_triangle(outside, outside + 1, inside + 1, base);
            base = base + 3;
            inside = inside + 1;
            outside = outside + 1;
        } else if (!inside_odd) {
            reference_triangle(outside, outside + 1, inside, base);
            base = base + 3;
            outside = outside + 1;
        } else {
            reference_triangle(inside, outside, inside + 1, base);
            base = base + 3;
            inside = inside + 1;
        }
    }
    for (var i = i_end; i >= i_start; i--) {
        if (FINAL_POINT_POSITION[i] < outside_half) {
            reference_triangle(outside, outside + 1, inside, base);
            base = base + 3;
            outside = outside + 1;
        }
        if (FINAL_POINT_POSITION[i] < inside_half) {
            reference_triangle(inside, outside, inside + 1, base);
            base = base + 3;
            inside = inside + 1;
        }
    }
    if (FINAL_POINT_POSITION[0] < outside_half) {
        reference_triangle(outside, outside + 1, inside, base);
    }
}

// Quad patches --------------------------------------------------------------------------------------------------------

fn quad_points() {
    var offset = 0;
    for (var edge = 0; edge < 4; edge++) {
        let end = pf.outside_points[edge] - 1;
        parity_odd = pf.outside_odd[edge];
        for (var p = 0; p < end; p++) {
            let q = select(end - p, p, edge == 1 || edge == 2);
            let param = place(outside_ctx[edge], q);
            if ((edge & 1) != 0) {
                define_point(param, select(0u, FXP_ONE, edge == 3), offset);
            } else {
                define_point(select(0u, FXP_ONE, edge == 2), param, offset);
            }
            offset = offset + 1;
        }
    }
    let nu = pf.inside_points[0];
    let nv = pf.inside_points[1];
    let rings = min(nu, nv) >> 1u;
    for (var ring = 1; ring < rings; ring++) {
        let start = ring;
        let end = array<i32, 2>(nu - 1 - start, nv - 1 - start);
        for (var edge = 0; edge < 4; edge++) {
            let a0 = edge & 1;
            let a1 = (edge + 1) & 1;
            let perp_point = select(end[a0], start, edge < 2);
            parity_odd = pf.inside_odd[a0];
            let perp = place(inside_ctx[a0], perp_point);
            parity_odd = pf.inside_odd[a1];
            for (var p = start; p < end[a1]; p++) {
                let q = select(end[a1] - (p - start), p, edge == 1 || edge == 2);
                let param = place(inside_ctx[a1], q);
                if (a1 != 0) {
                    define_point(perp, param, offset);
                } else {
                    define_point(param, perp, offset);
                }
                offset = offset + 1;
            }
        }
    }
    if (nu > nv && !pf.inside_odd[1]) {
        let start = rings;
        let end = nu - 1 - start;
        parity_odd = pf.inside_odd[0];
        for (var p = start; p <= end; p++) {
            define_point(place(inside_ctx[0], p), FXP_ONE_HALF, offset);
            offset = offset + 1;
        }
    } else if (nv >= nu && !pf.inside_odd[0]) {
        let start = rings;
        let end = nv - 1 - start;
        parity_odd = pf.inside_odd[1];
        for (var p = end; p >= start; p--) {
            define_point(FXP_ONE_HALF, place(inside_ctx[1], p), offset);
            offset = offset + 1;
        }
    }
}

// The point index of inside grid point (i, j) (hw.rs QuadGrid::index).
fn grid_index(i: i32, j: i32) -> i32 {
    let nu = pf.inside_points[0];
    let nv = pf.inside_points[1];
    let rings = min(nu, nv) >> 1u;
    let r = min(min(i, j), min(nu - 1 - i, nv - 1 - j));
    if (r < rings) {
        let end_u = nu - 1 - r;
        let end_v = nv - 1 - r;
        let lu = end_u - r;
        let lv = end_v - r;
        let b = pf.inside_edge_point_base + 2 * (r - 1) * (nu + nv - 2) - 4 * r * (r - 1);
        if (i == r && j > r) {
            return b + (end_v - j);
        }
        if (j == r && i < end_u) {
            return b + lv + (i - r);
        }
        if (i == end_u && j < end_v) {
            return b + lv + lu + (j - r);
        }
        return b + 2 * lv + lu + (end_u - i);
    }
    let b = pf.inside_edge_point_base + 2 * (rings - 1) * (nu + nv - 2) - 4 * rings * (rings - 1);
    if (nu > nv && !pf.inside_odd[1]) {
        return b + (i - rings);
    }
    return b + (nv - 1 - rings - j);
}

// The grid point of quad (i, j) of blade k at da steps along a and db along b from its corner nearest the center.
fn blade_corner(k: i32, i: i32, j: i32, da: i32, db: i32) -> i32 {
    switch k {
        case 0: { return grid_index(i + 1 - da, j + db); }
        case 1: { return grid_index(i + 1 - db, j + 1 - da); }
        case 2: { return grid_index(i + da, j + 1 - db); }
        default: { return grid_index(i + db, j + da); }
    }
}

fn quad_emit(k: i32, i: i32, j: i32, at: i32) {
    let n = blade_corner(k, i, j, 0, 0);
    let x = blade_corner(k, i, j, 1, 0);
    let f = blade_corner(k, i, j, 1, 1);
    let y = blade_corner(k, i, j, 0, 1);
    metal_triangle(x, f, n, at);
    metal_triangle(n, f, y, at + 3);
}

// A blade's outer and inner index ranges (first, step, count) and center quad (hw.rs quad_blade).
struct Blade {
    outer: vec3<i32>,
    inner: vec3<i32>,
    center: bool,
}

fn range_down(lo: i32, hi: i32) -> vec3<i32> {
    return vec3<i32>(hi, -1, max(hi - lo + 1, 0));
}

fn range_up(lo: i32, hi: i32) -> vec3<i32> {
    return vec3<i32>(lo, 1, max(hi - lo + 1, 0));
}

fn quad_blade(nu: i32, nv: i32, k: i32) -> Blade {
    var b: Blade;
    b.center = false;
    switch k {
        case 0: {
            b.outer = range_down(1, (nu + 1) / 2 - 2);
            b.inner = range_up(select(nv / 2, nv / 2 - 1, nv % 2 == 0), nv - 3);
            b.center = nu % 2 == 0 && nv % 2 == 0;
        }
        case 1: {
            b.outer = range_down(1, (nv + 1) / 2 - 2);
            b.inner = range_down(1, select((nu + 1) / 2 - 2, nu / 2 - 1, nu % 2 == 0));
        }
        case 2: {
            b.outer = range_up(nu / 2, nu - 3);
            b.inner = range_down(1, select((nv + 1) / 2 - 2, nv / 2 - 1, nv % 2 == 0));
        }
        default: {
            b.outer = range_up(nv / 2, nv - 3);
            b.inner = range_up(select(nu / 2, nu / 2 - 1, nu % 2 == 0), nu - 3);
        }
    }
    return b;
}

fn quad_connectivity() {
    let nu = pf.inside_points[0];
    let nv = pf.inside_points[1];
    let rows = array<i32, 2>((nu + 1) >> 1u, (nv + 1) >> 1u);
    let degenerate = array<i32, 2>(select(-1, rows[1] - 1, !pf.inside_odd[1]), select(-1, rows[0] - 1, !pf.inside_odd[0]));
    var inside_base_cur = pf.inside_edge_point_base;
    var outside_base_cur = 0;
    let inside_edge = array<i32, 2>(nu - 2, nv - 2);
    let edge0_inside = inside_base_cur;
    let edge0_outside = outside_base_cur;
    var at = 0;
    for (var edge = 0; edge < 4; edge++) {
        let parity = (edge + 1) & 1;
        let triangles = inside_edge[parity] + pf.outside_points[edge] - 2;
        var inside_base: i32;
        var outside_base: i32;
        if (edge == 3) {
            if (degenerate[parity] == 1) {
                p2_base = inside_base_cur + 1;
                p2_bad = outside_base_cur + pf.outside_points[edge] - 1;
                p2_replacement = edge0_outside;
                p2_end = (p2_base << 1u) - 1;
                inside_base = p2_base;
                outside_base = outside_base_cur;
                patched2 = true;
            } else {
                p_inside_delta = inside_base_cur;
                p_inside_bad = inside_edge[parity] - 1;
                p_inside_replacement = edge0_inside;
                p_outside_base = p_inside_bad + 1;
                p_outside_delta = outside_base_cur - p_outside_base;
                p_outside_bad = p_outside_base + pf.outside_points[edge] - 1;
                p_outside_replacement = edge0_outside;
                inside_base = 0;
                outside_base = p_outside_base;
                patched = true;
            }
        } else if (edge == 2 && degenerate[parity] == 1) {
            p2_base = inside_base_cur;
            p2_bad = -1;
            p2_replacement = -1;
            p2_end = p2_base << 1u;
            inside_base = p2_base;
            outside_base = outside_base_cur;
            patched2 = true;
        } else {
            inside_base = inside_base_cur;
            outside_base = outside_base_cur;
        }
        stitch_transition(at, inside_base, inside_ctx[parity].half_points, pf.inside_odd[parity], outside_base,
                          outside_ctx[edge].half_points, pf.outside_odd[edge]);
        patched = false;
        patched2 = false;
        at = at + 3 * triangles;
        outside_base_cur = outside_base_cur + pf.outside_points[edge] - 1;
        if (edge == 2 && degenerate[parity] == 1) {
            inside_base_cur = inside_base_cur - (inside_edge[parity] - 1);
        } else {
            inside_base_cur = inside_base_cur + inside_edge[parity] - 1;
        }
    }
    for (var k = 0; k < 4; k++) {
        let b = quad_blade(nu, nv, k);
        if (b.center) {
            quad_emit(k, nu / 2 - 1, nv / 2 - 1, at);
            at = at + 6;
        }
        for (var o = 0; o < b.outer.z; o++) {
            for (var n = 0; n < b.inner.z; n++) {
                let a = b.outer.x + o * b.outer.y;
                let c = b.inner.x + n * b.inner.y;
                if (k % 2 == 0) {
                    quad_emit(k, a, c, at);
                } else {
                    quad_emit(k, c, a, at);
                }
                at = at + 6;
            }
        }
    }
}

fn quad_counts() -> vec3<u32> {
    if (pf.culled) {
        return vec3<u32>(0u);
    }
    if (pf.minimum) {
        return vec3<u32>(4u, 6u, 0u);
    }
    let nu = pf.inside_points[0];
    let nv = pf.inside_points[1];
    var triangles = 0;
    let inside_edge = array<i32, 2>(nu - 2, nv - 2);
    for (var edge = 0; edge < 4; edge++) {
        triangles = triangles + inside_edge[(edge + 1) & 1] + pf.outside_points[edge] - 2;
    }
    triangles = triangles + 2 * max(nu - 3, 0) * max(nv - 3, 0);
    var batched = false;
    for (var k = 0; k < 4; k++) {
        let b = quad_blade(nu, nv, k);
        let a = b.outer.z + select(0, 1, b.center);
        batched = batched || (a > 0 && b.inner.z > 0 && (a + 1) * (b.inner.z + 1) > WHOLE_BLADE_POINTS);
    }
    return vec3<u32>(u32(pf.points), u32(3 * triangles), select(0u, 1u, batched));
}

// Triangle patches ----------------------------------------------------------------------------------------------------

fn tri_points() {
    var offset = 0;
    for (var edge = 0; edge < 3; edge++) {
        let end = pf.outside_points[edge] - 1;
        parity_odd = pf.outside_odd[edge];
        for (var p = 0; p < end; p++) {
            let q = select(end - p, p, (edge & 1) != 0);
            let param = place(outside_ctx[edge], q);
            if (edge == 0) {
                define_point(0u, param, offset);
            } else {
                define_point(param, select(0u, FXP_ONE - param, edge == 2), offset);
            }
            offset = offset + 1;
        }
    }
    parity_odd = pf.inside_odd[0];
    let n = pf.inside_points[0];
    let rings = n >> 1u;
    for (var ring = 1; ring < rings; ring++) {
        let start = ring;
        let end = n - 1 - start;
        for (var edge = 0; edge < 3; edge++) {
            var perp = place(inside_ctx[0], start);
            perp = (perp * FXP_TWO_THIRDS + FXP_ONE_HALF) >> 16u;
            for (var p = start; p < end; p++) {
                let q = select(end - (p - start), p, (edge & 1) != 0);
                let param = place(inside_ctx[0], q);
                let shifted = param - (perp + 1u) / 2u;
                if (edge == 0) {
                    define_point(perp, shifted, offset);
                } else if (edge == 1) {
                    define_point(shifted, perp, offset);
                } else {
                    define_point(shifted, FXP_ONE - shifted - perp, offset);
                }
                offset = offset + 1;
            }
        }
    }
    if (!parity_odd) {
        define_point(FXP_ONE_THIRD, FXP_ONE_THIRD, offset);
    }
}

fn tri_ring_base(r: i32) -> i32 {
    let n = pf.inside_points[0];
    return pf.inside_edge_point_base + 3 * (r - 1) * (n - 1) - 3 * r * (r - 1);
}

// The point (a, b) of corner k's blade (hw.rs TriGrid::point).
fn tri_point(k: i32, a: i32, b: i32) -> i32 {
    let n = pf.inside_points[0];
    let d = max(a, b);
    if (d == 0) {
        return tri_ring_base(n >> 1u);
    }
    let o = select(0, 1, pf.inside_odd[0]);
    let l = 2 * d - o;
    let r = (n - 1 - l) / 2;
    var e: i32;
    var p: i32;
    if (a == d) {
        e = k;
        p = d - b;
    } else {
        e = (k + 2) % 3;
        p = d - o + a;
    }
    if (p == l) {
        e = (e + 1) % 3;
        p = 0;
    }
    return tri_ring_base(r) + e * l + p;
}

fn tri_connectivity() {
    let n = pf.inside_points[0];
    var inside_base_cur = pf.inside_edge_point_base;
    var outside_base_cur = 0;
    let inside_edge = n - 2;
    let edge0_inside = inside_base_cur;
    let edge0_outside = outside_base_cur;
    var at = 0;
    for (var edge = 0; edge < 3; edge++) {
        let triangles = inside_edge + pf.outside_points[edge] - 2;
        var inside_base: i32;
        var outside_base: i32;
        if (edge == 2) {
            p_inside_delta = inside_base_cur;
            p_inside_bad = inside_edge - 1;
            p_inside_replacement = edge0_inside;
            p_outside_base = p_inside_bad + 1;
            p_outside_delta = outside_base_cur - p_outside_base;
            p_outside_bad = p_outside_base + pf.outside_points[edge] - 1;
            p_outside_replacement = edge0_outside;
            patched = true;
            inside_base = 0;
            outside_base = p_outside_base;
        } else {
            inside_base = inside_base_cur;
            outside_base = outside_base_cur;
        }
        stitch_transition(at, inside_base, inside_ctx[0].half_points, pf.inside_odd[0], outside_base,
                          outside_ctx[edge].half_points, pf.outside_odd[edge]);
        patched = false;
        at = at + 3 * triangles;
        outside_base_cur = outside_base_cur + pf.outside_points[edge] - 1;
        inside_base_cur = inside_base_cur + inside_edge - 1;
    }
    let odd = pf.inside_odd[0];
    let rings = n >> 1u;
    let a_max = rings - 2;
    if (odd) {
        let c = tri_ring_base(rings - 1);
        if (a_max == 0) {
            metal_triangle(c + 1, c, c + 2, at);
        } else {
            metal_triangle(c + 2, c + 1, c, at);
        }
        at = at + 3;
    }
    for (var k = 0; k < 3; k++) {
        for (var a = select(0, 1, odd); a <= a_max; a++) {
            for (var b = 0; b <= a_max; b++) {
                let nn = tri_point(k, a, b);
                let x = tri_point(k, a + 1, b);
                let f = tri_point(k, a + 1, b + 1);
                let y = tri_point(k, a, b + 1);
                metal_triangle(x, f, nn, at);
                metal_triangle(nn, f, y, at + 3);
                at = at + 6;
            }
        }
    }
}

fn tri_counts() -> vec3<u32> {
    if (pf.culled) {
        return vec3<u32>(0u);
    }
    if (pf.minimum) {
        return vec3<u32>(3u, 3u, 0u);
    }
    let n = pf.inside_points[0];
    var triangles = 0;
    for (var edge = 0; edge < 3; edge++) {
        triangles = triangles + n - 2 + pf.outside_points[edge] - 2;
    }
    let span = (n >> 1u) - 1;
    let odd = pf.inside_odd[0];
    let quads = select(span * span, (span - 1) * span, odd);
    triangles = triangles + 6 * quads + select(0, 1, odd);
    let batched = (span + 1) * (span + 1) > WHOLE_BLADE_POINTS;
    return vec3<u32>(u32(pf.points), u32(3 * triangles), select(0u, 1u, batched));
}

// Slots ---------------------------------------------------------------------------------------------------------------

// Load slot s's patch: its factors as the step function addresses them, its patch and instance identifiers.
fn load_slot(s: u32) {
    let instance = s / params.patch_count;
    let draw_patch = params.patch_start + s % params.patch_count;
    slot_instance = params.base_instance + instance;
    slot_patch = draw_patch;
    if (params.indexed != 0u) {
        slot_patch = patch_index_words[(params.patch_index_offset >> 2u) + draw_patch];
    }
    let size = select(8u, 12u, params.quad != 0u);
    var at = params.factor_offset;
    if (params.step == 1u || params.step == 3u) {
        at = at + draw_patch * size;
    }
    if (params.step == 2u || params.step == 3u) {
        at = at + slot_instance * params.instance_stride;
    }
    var halves: array<u32, 6>;
    let edges = select(3u, 4u, params.quad != 0u);
    let insides = select(1u, 2u, params.quad != 0u);
    for (var i = 0u; i < edges + insides; i++) {
        halves[i] = read_half(at + 2u * i);
    }
    process(halves);
}

@compute @workgroup_size(64)
fn tess_count(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if (s >= params.slots) {
        return;
    }
    generate = false;
    load_slot(s);
    let c = select(tri_counts(), quad_counts(), params.quad != 0u);
    slot_words[2u * s] = c.x;
    slot_words[2u * s + 1u] = c.y;
    if (c.z != 0u) {
        atomicOr(&draw_words[5], 1u);
    }
}

var<workgroup> sums: array<vec2<u32>, 256>;

@compute @workgroup_size(256)
fn tess_scan(@builtin(local_invocation_index) t: u32) {
    let per = (params.slots + 255u) / 256u;
    let first = t * per;
    let last = min(first + per, params.slots);
    var total = vec2<u32>(0u);
    for (var s = first; s < last; s++) {
        total = total + vec2<u32>(slot_words[2u * s], slot_words[2u * s + 1u]);
    }
    sums[t] = total;
    workgroupBarrier();
    // Exclusive prefix of the chunk totals.
    var before = vec2<u32>(0u);
    for (var i = 0u; i < t; i++) {
        before = before + sums[i];
    }
    for (var s = first; s < last; s++) {
        let c = vec2<u32>(slot_words[2u * s], slot_words[2u * s + 1u]);
        slot_words[2u * s] = before.x;
        slot_words[2u * s + 1u] = before.y;
        before = before + c;
    }
    if (t == 255u) {
        atomicStore(&draw_words[0], before.y);
        atomicStore(&draw_words[1], 1u);
        atomicStore(&draw_words[2], 0u);
        atomicStore(&draw_words[3], 0u);
        atomicStore(&draw_words[4], 0u);
        atomicStore(&draw_words[6], before.x);
    }
}

@compute @workgroup_size(64)
fn tess_generate(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x;
    if (s >= params.slots) {
        return;
    }
    generate = true;
    patched = false;
    patched2 = false;
    point_base = slot_words[2u * s];
    index_base = slot_words[2u * s + 1u];
    load_slot(s);
    if (pf.culled) {
        return;
    }
    if (params.quad != 0u) {
        if (pf.minimum) {
            define_point(0u, 0u, 0);
            define_point(FXP_ONE, 0u, 1);
            define_point(FXP_ONE, FXP_ONE, 2);
            define_point(0u, FXP_ONE, 3);
            reference_triangle(0, 1, 3, 0);
            reference_triangle(3, 1, 2, 3);
            return;
        }
        quad_points();
        quad_connectivity();
    } else {
        if (pf.minimum) {
            define_point(0u, FXP_ONE, 0);
            define_point(0u, 0u, 1);
            define_point(FXP_ONE, 0u, 2);
            reference_triangle(1, 2, 0, 0);
            return;
        }
        tri_points();
        tri_connectivity();
    }
}
