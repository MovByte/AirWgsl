// Attribute fetch conversions (vertex input and compute stage-in). Values are carriers: u32 holding the bits of the
// attribute's scalar zero-extended (src/wgsl/types.rs). Each function starts with "fn aw_vf_" at the start of a line
// and is emitted with the functions it calls.
//
// The results were measured on real Metal for every input of the 8-, 10- and 16-bit formats (see src/stage/fetch.rs):
// normalized integers are converted to binary32 and binary16 with one correct rounding of c / (2^n - 1), and of
// max(-1, c / (2^(n-1) - 1)), except the 10-bit components of Int1010102Normalized to binary32 (aw_vf_snorm10_f32);
// integers to binary16 round to nearest even; binary32 to binary16 rounds to nearest even and a NaN becomes the quiet
// NaN 0x7e00; binary16 to binary32 is exact and a NaN becomes the quiet NaN 0x7fc00000.

// The correctly rounded (to nearest even) binary32 bits of c / m for 0 < c < m < 2^31: long division producing the
// leading one, the 23 bits after it and one guard bit; the remainder is the sticky bit.
fn aw_vf_div_f32(c: u32, m: u32) -> u32 {
    var r = c;
    var e = 127u;
    loop {
        r = r << 1u;
        e = e - 1u;
        if (r >= m) {
            break;
        }
    }
    r = r - m;
    var q = 1u;
    for (var i = 0u; i < 24u; i = i + 1u) {
        r = r << 1u;
        q = q << 1u;
        if (r >= m) {
            r = r - m;
            q = q | 1u;
        }
    }
    let guard = q & 1u;
    let mant = q >> 1u;
    let up = guard & (select(0u, 1u, r != 0u) | (mant & 1u));
    return ((e << 23u) | (mant & 0x7fffffu)) + up;
}

// The correctly rounded (to nearest even) binary16 bits of c / m for 0 < c < m < 2^31, subnormal results included:
// long division from the 2^-1 place down to one guard bit below the unit in the last place, which is 2^-(p+10) when
// the leading one is at 2^-p with p <= 14 and 2^-24 (the subnormal unit) otherwise.
fn aw_vf_div_f16(c: u32, m: u32) -> u32 {
    var r = c;
    var q = 0u;
    var k = 0u;
    var lead = 0u;
    var last = 25u;
    loop {
        k = k + 1u;
        r = r << 1u;
        q = q << 1u;
        if (r >= m) {
            r = r - m;
            q = q | 1u;
            if (lead == 0u) {
                lead = k;
                if (k <= 14u) {
                    last = k + 11u;
                }
            }
        }
        if (k == last) {
            break;
        }
    }
    let guard = q & 1u;
    let mant = q >> 1u;
    let res = mant + (guard & (select(0u, 1u, r != 0u) | (mant & 1u)));
    if (lead != 0u && lead <= 14u) {
        // res is in [2^10, 2^11]: the leading one is the implicit bit; a carry to 2^11 raises the exponent.
        return ((15u - lead) << 10u) + res - 1024u;
    }
    // A multiple of 2^-24: a subnormal, or 0x0400 (the smallest normal) when it rounds up to 2^-14.
    return res;
}

// unorm: c / (2^n - 1), n <= 16, correctly rounded to binary32.
fn aw_vf_unorm_f32(c: u32, n: u32) -> u32 {
    let m = (1u << n) - 1u;
    if (c == 0u) {
        return 0u;
    }
    if (c == m) {
        return 0x3f800000u;
    }
    return aw_vf_div_f32(c, m);
}

// snorm: max(-1, c / (2^(n-1) - 1)) for the n-bit two's complement code c, n <= 16, correctly rounded to binary32.
fn aw_vf_snorm_f32(c: u32, n: u32) -> u32 {
    let m = (1u << (n - 1u)) - 1u;
    let neg = (c >> (n - 1u)) != 0u;
    let mag = select(c, (1u << n) - c, neg);
    if (mag == 0u) {
        return 0u;
    }
    let sign = select(0u, 0x80000000u, neg);
    if (mag >= m) {
        return sign | 0x3f800000u;
    }
    return sign | aw_vf_div_f32(mag, m);
}

// Int1010102Normalized's 10-bit components to binary32. Real Metal (measured for every code) rounds f32(c) times the
// binary32 nearest to 1/511 (0x3b004020) and clamps the product at -1; the result differs from the correctly rounded
// c / 511 for 96 of the 1024 codes. nm_f32_mul (the numerics library) is the correctly rounded binary32 product.
fn aw_vf_snorm10_f32(c: u32) -> u32 {
    let v = bitcast<i32>(c << 22u) >> 22u;
    let r = nm_f32_mul(bitcast<u32>(f32(v)), 0x3b004020u);
    // A negative product below -1 (bits above those of -1.0) clamps to -1.
    return select(r, 0xbf800000u, r > 0xbf800000u);
}

// unorm to binary16, one correct rounding.
fn aw_vf_unorm_f16(c: u32, n: u32) -> u32 {
    let m = (1u << n) - 1u;
    if (c == 0u) {
        return 0u;
    }
    if (c == m) {
        return 0x3c00u;
    }
    return aw_vf_div_f16(c, m);
}

// snorm to binary16, one correct rounding.
fn aw_vf_snorm_f16(c: u32, n: u32) -> u32 {
    let m = (1u << (n - 1u)) - 1u;
    let neg = (c >> (n - 1u)) != 0u;
    let mag = select(c, (1u << n) - c, neg);
    if (mag == 0u) {
        return 0u;
    }
    let sign = select(0u, 0x8000u, neg);
    if (mag >= m) {
        return sign | 0x3c00u;
    }
    return sign | aw_vf_div_f16(mag, m);
}

// binary32 bits to binary16 bits, rounding to nearest even; a NaN becomes 0x7e00.
fn aw_vf_f32_to_f16(a: u32) -> u32 {
    let s = (a >> 16u) & 0x8000u;
    let e = (a >> 23u) & 0xffu;
    let f = a & 0x7fffffu;
    if (e == 0xffu) {
        return select(s | 0x7c00u, 0x7e00u, f != 0u);
    }
    // Biased binary16 exponent of the value's leading one (binary32 subnormals are far below binary16's range).
    let he = i32(e) - 112;
    if (he >= 31) {
        return s | 0x7c00u;
    }
    if (e == 0u || he < -10) {
        // Zero, or below half the smallest binary16 subnormal (2^-25): rounds to zero. 2^-25 itself is he == -10
        // with f == 0, which the subnormal path rounds to even (zero).
        return s;
    }
    let m = f | 0x800000u;
    if (he <= 0) {
        // Subnormal: the 24-bit significand shifted right by 14 - he places to the 2^-24 unit.
        let sh = u32(14 - he);
        let q = m >> sh;
        let rem = m & ((1u << sh) - 1u);
        let half = 1u << (sh - 1u);
        let up = select(0u, 1u, rem > half || (rem == half && (q & 1u) != 0u));
        return s | (q + up);
    }
    let q = m >> 13u;
    let rem = m & 0x1fffu;
    let up = select(0u, 1u, rem > 0x1000u || (rem == 0x1000u && (q & 1u) != 0u));
    // q includes the leading one at bit 10; adding the exponent above it lets a carry out of the mantissa raise the
    // exponent (to infinity at the top).
    return s | (((u32(he) << 10u) | (q & 0x3ffu)) + up);
}

// binary16 bits to binary32 bits, exact; a NaN becomes 0x7fc00000.
fn aw_vf_f16_to_f32(h: u32) -> u32 {
    let s = (h & 0x8000u) << 16u;
    let e = (h >> 10u) & 0x1fu;
    let f = h & 0x3ffu;
    if (e == 0x1fu) {
        return select(s | 0x7f800000u, 0x7fc00000u, f != 0u);
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

// An integer of magnitude below 2^24 (sign-extended in an i32) to binary16, rounding to nearest even.
fn aw_vf_int_f16(v: i32) -> u32 {
    return aw_vf_f32_to_f16(bitcast<u32>(f32(v)));
}
