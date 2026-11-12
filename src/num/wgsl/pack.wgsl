// MSL pack and unpack functions (MSL 6.15) with the normalized conversion rules of MSL 8.7.1 and 8.7.7. MSL 8.7.1.2:
// "In Metal 2, all conversions to and from unormalized data types round correctly", so both directions are exact:
// a code from a float is the integer nearest x * max (ties to even) of the exact product, and a float from a code is
// the value nearest code / max. Component 0 occupies the low bits of the packed word (measured,
// tests/num/data/metal-pack.txt, where Metal also rounds correctly except in its 565 packing).

// A binary32 value to an unsigned normalized code with largest code maxv: NaN to 0, clamp to [0, 1] (Table 8.7).
fn nm_unorm_code(x: u32, maxv: u32) -> u32 {
    if (nm_f32_isnan(x) || (x >> 31u) != 0u || x == 0u) {
        return 0u;
    }
    if (x >= 0x3f800000u) {
        return maxv;
    }
    // x = m * 2^(e - 150) with e <= 126; x * maxv = m * maxv * 2^-(150 - e), m * maxv < 2^40.
    let n = nm_f32_norm(x);
    let sh = u32(150 - n.e);
    if (sh >= 42u) {
        return 0u;
    }
    let p = nm_mul32w(n.m, maxv);
    let q = nm_shr64(p, sh);
    let rem = nm_sub64(p, nm_shl64(q, sh));
    let half = nm_shl64(vec2<u32>(1u, 0u), sh - 1u);
    let tie = rem.x == half.x && rem.y == half.y;
    if (nm_lt64(half, rem) || (tie && (q.x & 1u) != 0u)) {
        return q.x + 1u;
    }
    return q.x;
}

// A binary32 value to a signed normalized code of `bits` bits: NaN to 0, clamp to [-1, 1], nearest even (Table 8.7).
fn nm_snorm_code(x: u32, maxv: u32, bits: u32) -> u32 {
    if (nm_f32_isnan(x)) {
        return 0u;
    }
    let mag = nm_unorm_code(x & 0x7fffffffu, maxv);
    return select(mag, 0u - mag, (x >> 31u) != 0u) & ((1u << bits) - 1u);
}

// A code to the binary32 or binary16 value nearest code / maxv (Table 8.6).
fn nm_unorm_f32(c: u32, maxv: u32) -> u32 {
    return nm_f32_div(nm_f32_from_u32(c), nm_f32_from_u32(maxv));
}

fn nm_unorm_f16(c: u32, maxv: u32) -> u32 {
    return nm_f16_finish(nm_f32_div_ur(nm_f32_from_u32(c), nm_f32_from_u32(maxv)));
}

// max(-1.0, code / maxv) for a signed code of `bits` bits (Table 8.6).
fn nm_snorm_f32(c: u32, maxv: u32, bits: u32) -> u32 {
    let q = nm_f32_div(nm_f32_from_i32(u32(nm_sext(c, bits))), nm_f32_from_u32(maxv));
    return nm_f32_fmax(q, 0xbf800000u);
}

fn nm_snorm_f16(c: u32, maxv: u32, bits: u32) -> u32 {
    let q = nm_f16_finish(nm_f32_div_ur(nm_f32_from_i32(u32(nm_sext(c, bits))), nm_f32_from_u32(maxv)));
    return nm_f16_fmax(q, 0xbc00u);
}

// Linear to sRGB code (MSL 8.7.7): NaN and values at or below 0 give 0, values above 1 give 255; otherwise the count
// of thresholds at or below x.
fn nm_srgb_code(x: u32) -> u32 {
    if (nm_f32_isnan(x) || (x >> 31u) != 0u) {
        return 0u;
    }
    let t = min(x, 0x3f800000u);
    var lo = 0u;
    var hi = 255u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (nm_srgb_thresholds[mid] <= t) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn nm_srgb_f32(c: u32) -> u32 {
    return nm_srgb_to_linear_f32[c & 0xffu];
}

fn nm_srgb_f16(c: u32) -> u32 {
    return nm_srgb_to_linear_f16[c & 0xffu];
}
