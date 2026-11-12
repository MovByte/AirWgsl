// Conversions between binary32, binary16 and integers, and binary16 arithmetic, on bit patterns.
//
// A binary16 value is carried in the low 16 bits of a u32 with the high bits zero. Conversions to floating point
// round to nearest even with gradual underflow, and float-to-half does not flush half subnormals (MSL 8.6). Float to
// integer rounds toward zero and converts NaN to 0 (MSL 8.6); MSL gives no result for values outside the integer
// type's range, and these conversions saturate. A 64-bit integer is vec2<u32>(low word, high word).

fn nm_f16_isnan(h: u32) -> bool {
    return (h & 0x7fffu) > 0x7c00u;
}

fn nm_f16_issnan(h: u32) -> bool {
    return nm_f16_isnan(h) && (h & 0x0200u) == 0u;
}

fn nm_f16_nan2(a: u32, b: u32) -> u32 {
    if (nm_f16_issnan(a)) {
        return a | 0x0200u;
    }
    if (nm_f16_issnan(b)) {
        return b | 0x0200u;
    }
    if (nm_f16_isnan(a)) {
        return a;
    }
    return b;
}

fn nm_f16_nan3(a: u32, b: u32, c: u32) -> u32 {
    if (nm_f16_issnan(a)) {
        return a | 0x0200u;
    }
    if (nm_f16_issnan(b)) {
        return b | 0x0200u;
    }
    if (nm_f16_issnan(c)) {
        return c | 0x0200u;
    }
    if (nm_f16_isnan(a)) {
        return a;
    }
    if (nm_f16_isnan(b)) {
        return b;
    }
    return c;
}

// binary16 to binary32, exact. A NaN keeps its sign and payload and is quieted.
fn nm_f16_to_f32(h: u32) -> u32 {
    let s = (h & 0x8000u) << 16u;
    let e = (h >> 10u) & 0x1fu;
    let f = h & 0x3ffu;
    if (e == 0x1fu) {
        if (f == 0u) {
            return s | 0x7f800000u;
        }
        return s | 0x7fc00000u | (f << 13u);
    }
    if (e == 0u) {
        if (f == 0u) {
            return s;
        }
        let p = 31u - countLeadingZeros(f);
        return s | ((p + 103u) << 23u) | ((f << (23u - p)) & 0x7fffffu);
    }
    return s | ((e + 112u) << 23u) | (f << 13u);
}

// Round to nearest even and pack a binary16 value. s is the sign bit in place (0x8000 or 0); the value is
// m * 2^(e - 45) with m < 2^31, e the biased exponent when m's leading one is at bit 30.
fn nm_f16_rpack(s: u32, e: i32, m: u32) -> u32 {
    if (e >= 0x1f) {
        return s | 0x7c00u;
    }
    var exp = e;
    var sig = m;
    if (exp <= 0) {
        sig = nm_shr32_jam(sig, u32(1 - exp));
        exp = 1;
    }
    let rb = sig & 0xfffffu;
    sig = (sig + 0x80000u) >> 20u;
    if (rb == 0x80000u) {
        sig = sig & ~1u;
    }
    return s | ((u32(exp - 1) << 10u) + sig);
}

// binary32 to binary16, rounding to nearest even (MSL 8.6). A NaN keeps its sign and the high bits of its payload
// and is quieted.
fn nm_f32_to_f16(a: u32) -> u32 {
    let s = (a >> 16u) & 0x8000u;
    let e = (a >> 23u) & 0xffu;
    let f = a & 0x7fffffu;
    if (e == 0xffu) {
        if (f == 0u) {
            return s | 0x7c00u;
        }
        return s | 0x7e00u | (f >> 13u);
    }
    if ((a & 0x7fffffffu) == 0u) {
        return s;
    }
    let n = nm_f32_norm(a & 0x7fffffffu);
    return nm_f16_rpack(s, n.e - 112, n.m << 7u);
}

// Round an unrounded binary32-convention result (see f32.wgsl) directly to binary16: one rounding.
fn nm_f16_finish(r: nm_ur) -> u32 {
    if (r.k != 0u) {
        return nm_f32_to_f16(r.m);
    }
    let lz = countLeadingZeros(r.m) - 1u;
    return nm_f16_rpack(r.s << 15u, r.e - 112 - i32(lz), r.m << lz);
}

// binary16 arithmetic, correctly rounded: the exact operands widen to binary32, the binary32 core computes the
// unrounded result, and it is rounded once to binary16.
fn nm_f16_add(a: u32, b: u32) -> u32 {
    if (nm_f16_isnan(a) || nm_f16_isnan(b)) {
        return nm_f16_nan2(a, b);
    }
    return nm_f16_finish(nm_f32_add_ur(nm_f16_to_f32(a), nm_f16_to_f32(b)));
}

fn nm_f16_sub(a: u32, b: u32) -> u32 {
    if (nm_f16_isnan(a) || nm_f16_isnan(b)) {
        return nm_f16_nan2(a, b);
    }
    return nm_f16_finish(nm_f32_sub_ur(nm_f16_to_f32(a), nm_f16_to_f32(b)));
}

fn nm_f16_mul(a: u32, b: u32) -> u32 {
    if (nm_f16_isnan(a) || nm_f16_isnan(b)) {
        return nm_f16_nan2(a, b);
    }
    return nm_f16_finish(nm_f32_mul_ur(nm_f16_to_f32(a), nm_f16_to_f32(b)));
}

fn nm_f16_div(a: u32, b: u32) -> u32 {
    if (nm_f16_isnan(a) || nm_f16_isnan(b)) {
        return nm_f16_nan2(a, b);
    }
    return nm_f16_finish(nm_f32_div_ur(nm_f16_to_f32(a), nm_f16_to_f32(b)));
}

fn nm_f16_sqrt(a: u32) -> u32 {
    if (nm_f16_isnan(a)) {
        return a | 0x0200u;
    }
    return nm_f16_finish(nm_f32_sqrt_ur(nm_f16_to_f32(a)));
}

fn nm_f16_rsqrt(a: u32) -> u32 {
    if (nm_f16_isnan(a)) {
        return a | 0x0200u;
    }
    return nm_f16_finish(nm_f32_rsqrt_ur(nm_f16_to_f32(a)));
}

fn nm_f16_fma(a: u32, b: u32, c: u32) -> u32 {
    if (nm_f16_isnan(a) || nm_f16_isnan(b) || nm_f16_isnan(c)) {
        let inv = ((a & 0x7fffu) == 0x7c00u && (b & 0x7fffu) == 0u) || ((a & 0x7fffu) == 0u && (b & 0x7fffu) == 0x7c00u);
        if (nm_f16_isnan(c) && !nm_f16_issnan(c) && inv) {
            return 0x7e00u;
        }
        return nm_f16_nan3(c, a, b);
    }
    return nm_f16_finish(nm_f32_fma_ur(nm_f16_to_f32(a), nm_f16_to_f32(b), nm_f16_to_f32(c)));
}

// Integers to binary32.
fn nm_f32_from_u32(v: u32) -> u32 {
    if (v == 0u) {
        return 0u;
    }
    let lz = countLeadingZeros(v);
    let t = v << lz;
    return nm_f32_rpack(0u, 158 - i32(lz), (t >> 1u) | (t & 1u));
}

fn nm_f32_from_i32(v: u32) -> u32 {
    let s = v >> 31u;
    return (s << 31u) | nm_f32_from_u32(select(v, 0u - v, s != 0u));
}

fn nm_neg64(v: vec2<u32>) -> vec2<u32> {
    return nm_sub64(vec2<u32>(0u, 0u), v);
}

fn nm_f32_from_u64(v: vec2<u32>) -> u32 {
    if (nm_is0_64(v)) {
        return 0u;
    }
    let lz = nm_clz64(v);
    return nm_f32_rpack(0u, 190 - i32(lz), nm_shr64_jam(nm_shl64(v, lz), 33u).x);
}

fn nm_f32_from_i64(v: vec2<u32>) -> u32 {
    let s = v.y >> 31u;
    return (s << 31u) | nm_f32_from_u64(select(v, nm_neg64(v), s != 0u));
}

// Integers to binary16, rounded once.
fn nm_f16_from_u32(v: u32) -> u32 {
    if (v == 0u) {
        return 0u;
    }
    let lz = countLeadingZeros(v);
    let t = v << lz;
    return nm_f16_rpack(0u, 46 - i32(lz), (t >> 1u) | (t & 1u));
}

fn nm_f16_from_i32(v: u32) -> u32 {
    let s = v >> 31u;
    return (s << 15u) | nm_f16_from_u32(select(v, 0u - v, s != 0u));
}

fn nm_f16_from_u64(v: vec2<u32>) -> u32 {
    if (nm_is0_64(v)) {
        return 0u;
    }
    let lz = nm_clz64(v);
    return nm_f16_rpack(0u, 78 - i32(lz), nm_shr64_jam(nm_shl64(v, lz), 33u).x);
}

fn nm_f16_from_i64(v: vec2<u32>) -> u32 {
    let s = v.y >> 31u;
    return (s << 15u) | nm_f16_from_u64(select(v, nm_neg64(v), s != 0u));
}

// binary32 to integers: toward zero, NaN to 0, saturating.
fn nm_f32_to_i32(a: u32) -> u32 {
    if (nm_f32_isnan(a)) {
        return 0u;
    }
    let e = (a >> 23u) & 0xffu;
    let neg = (a >> 31u) != 0u;
    if (e < 127u) {
        return 0u;
    }
    if (e >= 158u) {
        return select(0x7fffffffu, 0x80000000u, neg);
    }
    let m = (a & 0x7fffffu) | 0x800000u;
    var v: u32;
    if (e >= 150u) {
        v = m << (e - 150u);
    } else {
        v = m >> (150u - e);
    }
    return select(v, 0u - v, neg);
}

fn nm_f32_to_u32(a: u32) -> u32 {
    if (nm_f32_isnan(a)) {
        return 0u;
    }
    let e = (a >> 23u) & 0xffu;
    if (e < 127u || (a >> 31u) != 0u) {
        return 0u;
    }
    if (e >= 159u) {
        return 0xffffffffu;
    }
    let m = (a & 0x7fffffu) | 0x800000u;
    if (e >= 150u) {
        return m << (e - 150u);
    }
    return m >> (150u - e);
}

fn nm_f32_to_i64(a: u32) -> vec2<u32> {
    if (nm_f32_isnan(a)) {
        return vec2<u32>(0u, 0u);
    }
    let e = (a >> 23u) & 0xffu;
    let neg = (a >> 31u) != 0u;
    if (e < 127u) {
        return vec2<u32>(0u, 0u);
    }
    if (e >= 190u) {
        return select(vec2<u32>(0xffffffffu, 0x7fffffffu), vec2<u32>(0u, 0x80000000u), neg);
    }
    let m = (a & 0x7fffffu) | 0x800000u;
    var v: vec2<u32>;
    if (e >= 150u) {
        v = nm_shl64(vec2<u32>(m, 0u), e - 150u);
    } else {
        v = vec2<u32>(m >> (150u - e), 0u);
    }
    return select(v, nm_neg64(v), neg);
}

fn nm_f32_to_u64(a: u32) -> vec2<u32> {
    if (nm_f32_isnan(a)) {
        return vec2<u32>(0u, 0u);
    }
    let e = (a >> 23u) & 0xffu;
    if (e < 127u || (a >> 31u) != 0u) {
        return vec2<u32>(0u, 0u);
    }
    if (e >= 191u) {
        return vec2<u32>(0xffffffffu, 0xffffffffu);
    }
    let m = (a & 0x7fffffu) | 0x800000u;
    if (e >= 150u) {
        return nm_shl64(vec2<u32>(m, 0u), e - 150u);
    }
    return vec2<u32>(m >> (150u - e), 0u);
}

// Saturating narrowing of the 32-bit results above to 8- and 16-bit carriers (low bits, high bits zero).
fn nm_f32_to_i16(a: u32) -> u32 {
    return u32(clamp(bitcast<i32>(nm_f32_to_i32(a)), -32768, 32767)) & 0xffffu;
}

fn nm_f32_to_u16(a: u32) -> u32 {
    return min(nm_f32_to_u32(a), 0xffffu);
}

fn nm_f32_to_i8(a: u32) -> u32 {
    return u32(clamp(bitcast<i32>(nm_f32_to_i32(a)), -128, 127)) & 0xffu;
}

fn nm_f32_to_u8(a: u32) -> u32 {
    return min(nm_f32_to_u32(a), 0xffu);
}
