// MSL's selection, rounding and common functions with NaN and infinity support (the precise variants), on binary32
// and binary16 bit patterns. Definitions are MSL's (Tables 6.1 and 6.4); where MSL gives no result for a special
// value, C99 Annex F.9 and OpenCL C 3.0 7.5.1 (see math.wgsl).

// fmod: x - y * trunc(x / y), always exact (C99 F.9.7.1 for the special values).
fn nm_f32_fmod(x: u32, y: u32) -> u32 {
    if (nm_f32_isnan(x) || nm_f32_isnan(y)) {
        return nm_f32_nan2(x, y);
    }
    let ax = x & 0x7fffffffu;
    let ay = y & 0x7fffffffu;
    let s = x & 0x80000000u;
    if (ax == 0x7f800000u || ay == 0u) {
        return nm_f32_dnan;
    }
    if (ay == 0x7f800000u || ax < ay) {
        return x;
    }
    let nx = nm_f32_norm(ax);
    let ny = nm_f32_norm(ay);
    // |x| mod |y| = 2^(ey - 150) * ((mx * 2^(ex - ey)) mod my), reduced eight bits at a time.
    var r = nx.m % ny.m;
    var d = nx.e - ny.e;
    loop {
        if (d <= 0) {
            break;
        }
        let k = u32(min(d, 8));
        r = (r << k) % ny.m;
        d = d - i32(k);
    }
    if (r == 0u) {
        return s;
    }
    return nm_f32_nrpack(s >> 31u, ny.e, r << 7u);
}

// fract: x - floor(x), never 1 (OpenCL C 3.0 7.5.1: fract(+-0) = +-0, fract(+-inf) = +-0).
fn nm_f32_fract(x: u32) -> u32 {
    if (nm_f32_isnan(x)) {
        return x | 0x00400000u;
    }
    let ax = x & 0x7fffffffu;
    if (ax == 0u) {
        return x;
    }
    if (ax == 0x7f800000u) {
        return x & 0x80000000u;
    }
    return nm_f32_fmin(nm_f32_sub(x, nm_f32_floor(x)), 0x3f7fffffu);
}

// nextafter (MSL Table 6.4): y when x equals y (zeros take y's sign); a NaN when either is NaN.
fn nm_f32_nextafter(x: u32, y: u32) -> u32 {
    if (nm_f32_isnan(x) || nm_f32_isnan(y)) {
        return nm_f32_nan2(x, y);
    }
    if (nm_f32_oeq(x, y)) {
        return y;
    }
    if ((x & 0x7fffffffu) == 0u) {
        return (y & 0x80000000u) | 1u;
    }
    let up = nm_f32_olt(x, y);
    let pos = (x >> 31u) == 0u;
    return select(x - 1u, x + 1u, up == pos);
}

fn nm_f32_fmax3(x: u32, y: u32, z: u32) -> u32 {
    return nm_f32_fmax(x, nm_f32_fmax(y, z));
}

fn nm_f32_fmin3(x: u32, y: u32, z: u32) -> u32 {
    return nm_f32_fmin(x, nm_f32_fmin(y, z));
}

// fmedian3: the middle value, NaNs treated as missing data (MSL 6.6). The fmin/fmax network gives that: with two NaNs
// the third value, with three a NaN, and with one NaN one of the other two, which MSL leaves open. For a NaN third
// operand the network would give the larger; the smaller is returned instead, the choice Metal makes (measured,
// tests/num/data).
fn nm_f32_fmedian3(x: u32, y: u32, z: u32) -> u32 {
    if (nm_f32_isnan(z) && !nm_f32_isnan(x) && !nm_f32_isnan(y)) {
        return nm_f32_fmin(x, y);
    }
    return nm_f32_fmax(nm_f32_fmin(x, y), nm_f32_fmin(nm_f32_fmax(x, y), z));
}

// clamp: fmin(fmax(x, minval), maxval) (MSL Table 6.1).
fn nm_f32_clamp(x: u32, lo: u32, hi: u32) -> u32 {
    return nm_f32_fmin(nm_f32_fmax(x, lo), hi);
}

fn nm_f32_saturate(x: u32) -> u32 {
    return nm_f32_clamp(x, 0u, 0x3f800000u);
}

// sign: 1.0 if x > 0, -0.0 if x = -0.0, +0.0 if x = +0.0, -1.0 if x < 0, 0.0 for a NaN (MSL Table 6.1).
fn nm_f32_sign(x: u32) -> u32 {
    if (nm_f32_isnan(x)) {
        return 0u;
    }
    if ((x & 0x7fffffffu) == 0u) {
        return x;
    }
    return (x & 0x80000000u) | 0x3f800000u;
}

// step: 0.0 if x < edge, otherwise 1.0 (MSL Table 6.1).
fn nm_f32_step(edge: u32, x: u32) -> u32 {
    return select(0x3f800000u, 0u, nm_f32_olt(x, edge));
}

// mix: x + (y - x) * a, one of the two forms MSL Table 6.1 allows.
fn nm_f32_mix(x: u32, y: u32, a: u32) -> u32 {
    return nm_f32_add(x, nm_f32_mul(nm_f32_sub(y, x), a));
}

// smoothstep: t = clamp((x - edge0) / (edge1 - edge0), 0, 1); t * t * (3 - 2 * t) (MSL Table 6.1).
fn nm_f32_smoothstep(e0: u32, e1: u32, x: u32) -> u32 {
    let t = nm_f32_clamp(nm_f32_div(nm_f32_sub(x, e0), nm_f32_sub(e1, e0)), 0u, 0x3f800000u);
    return nm_f32_mul(nm_f32_mul(t, t), nm_f32_sub(0x40400000u, nm_f32_mul(0x40000000u, t)));
}

fn nm_f32_recip(x: u32) -> u32 {
    return nm_f32_div(0x3f800000u, x);
}

// binary16 forms. Selections, rounding to integral values, fmod and ldexp are exact on the widened values, so
// narrowing their binary32 results is exact; nextafter steps on the binary16 grid.
fn nm_f16_w(h: u32) -> u32 {
    return nm_f16_to_f32(h);
}

fn nm_f16_fabs(h: u32) -> u32 {
    return h & 0x7fffu;
}

fn nm_f16_ceil(h: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_ceil(nm_f16_w(h)));
}

fn nm_f16_floor(h: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_floor(nm_f16_w(h)));
}

fn nm_f16_trunc(h: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_trunc(nm_f16_w(h)));
}

fn nm_f16_rint(h: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_rint(nm_f16_w(h)));
}

fn nm_f16_round(h: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_round(nm_f16_w(h)));
}

fn nm_f16_fmax(x: u32, y: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_fmax(nm_f16_w(x), nm_f16_w(y)));
}

fn nm_f16_fmin(x: u32, y: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_fmin(nm_f16_w(x), nm_f16_w(y)));
}

fn nm_f16_fmax3(x: u32, y: u32, z: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_fmax3(nm_f16_w(x), nm_f16_w(y), nm_f16_w(z)));
}

fn nm_f16_fmin3(x: u32, y: u32, z: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_fmin3(nm_f16_w(x), nm_f16_w(y), nm_f16_w(z)));
}

fn nm_f16_fmedian3(x: u32, y: u32, z: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_fmedian3(nm_f16_w(x), nm_f16_w(y), nm_f16_w(z)));
}

fn nm_f16_clamp(x: u32, lo: u32, hi: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_clamp(nm_f16_w(x), nm_f16_w(lo), nm_f16_w(hi)));
}

fn nm_f16_saturate(x: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_saturate(nm_f16_w(x)));
}

fn nm_f16_sign(x: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_sign(nm_f16_w(x)));
}

fn nm_f16_step(edge: u32, x: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_step(nm_f16_w(edge), nm_f16_w(x)));
}

fn nm_f16_fmod(x: u32, y: u32) -> u32 {
    if (nm_f16_isnan(x) || nm_f16_isnan(y)) {
        return nm_f16_nan2(x, y);
    }
    return nm_f32_to_f16(nm_f32_fmod(nm_f16_w(x), nm_f16_w(y)));
}

fn nm_f16_ldexp(h: u32, n: i32) -> u32 {
    if (nm_f16_isnan(h)) {
        return h | 0x0200u;
    }
    return nm_f32_to_f16(nm_f32_ldexp(nm_f16_w(h), n));
}

// fract on binary16: the binary32 difference is exact; its rounding to binary16 can reach 1, which fract excludes.
fn nm_f16_fract(h: u32) -> u32 {
    if (nm_f16_isnan(h)) {
        return h | 0x0200u;
    }
    let ah = h & 0x7fffu;
    if (ah == 0u) {
        return h;
    }
    if (ah == 0x7c00u) {
        return h & 0x8000u;
    }
    let w = nm_f16_w(h);
    return min(nm_f32_to_f16(nm_f32_sub(w, nm_f32_floor(w))), 0x3bffu);
}

fn nm_f16_nextafter(x: u32, y: u32) -> u32 {
    if (nm_f16_isnan(x) || nm_f16_isnan(y)) {
        return nm_f16_nan2(x, y);
    }
    let wx = nm_f16_w(x);
    let wy = nm_f16_w(y);
    if (nm_f32_oeq(wx, wy)) {
        return y;
    }
    if ((x & 0x7fffu) == 0u) {
        return (y & 0x8000u) | 1u;
    }
    let up = nm_f32_olt(wx, wy);
    let pos = (x >> 15u) == 0u;
    return select(x - 1u, x + 1u, up == pos);
}

// mix and smoothstep on binary16 evaluate MSL's formula with binary32 operations and round once more; MSL states no
// half-precision accuracy for them (numerics.toml rows mix, smoothstep).
fn nm_f16_mix(x: u32, y: u32, a: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_mix(nm_f16_w(x), nm_f16_w(y), nm_f16_w(a)));
}

fn nm_f16_smoothstep(e0: u32, e1: u32, x: u32) -> u32 {
    return nm_f32_to_f16(nm_f32_smoothstep(nm_f16_w(e0), nm_f16_w(e1), nm_f16_w(x)));
}
