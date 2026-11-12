// MSL math functions on binary32 and binary16 bit patterns, with NaN and infinity support (the precise functions of
// MSL Table 8.1, also used for the fast-math rows the numerics table marks polyfill).
//
// Each function computes in extended precision (xf.wgsl) and rounds once, to binary32 or directly to binary16.
// Special values: MSL's own definitions where it gives them (Table 6.1, 6.4); where MSL is silent, C99 Annex F.9
// (WG14 N1256) and, for the functions C99 lacks (sinpi, cospi, tanpi, exp10, powr), OpenCL C 3.0 section 7.5.1,
// whose flush-to-zero rules MSL 8.5 reproduces. A NaN operand gives that NaN quieted (C99 F.9 recommended
// practice); an invalid operation gives the default NaN.

// A function result: final binary32 bits (k == 1; exact in binary16 too, or a NaN) or an extended value to round.
struct nm_mr {
    x: nm_xf,
    bits: u32,
    k: u32,
}

fn nm_mr_bits(b: u32) -> nm_mr {
    return nm_mr(nm_xf(vec2<u32>(0u, 0u), 0, 0u), b, 1u);
}

fn nm_mr_xf(x: nm_xf) -> nm_mr {
    return nm_mr(x, 0u, 0u);
}

fn nm_mr_f32(r: nm_mr) -> u32 {
    if (r.k != 0u) {
        return r.bits;
    }
    return nm_xf_to_f32(r.x);
}

fn nm_mr_f16(r: nm_mr) -> u32 {
    if (r.k != 0u) {
        return nm_f32_to_f16(r.bits);
    }
    return nm_xf_to_f16(r.x);
}

fn nm_mr_neg(r: nm_mr, neg: bool) -> nm_mr {
    if (!neg) {
        return r;
    }
    if (r.k != 0u) {
        return nm_mr_bits(r.bits ^ 0x80000000u);
    }
    return nm_mr_xf(nm_xf_neg(r.x));
}

fn nm_xf_one() -> nm_xf {
    return nm_xf(vec2<u32>(0u, 0x80000000u), 0, 0u);
}

// Exponentials. Results past 2^128 are infinite and results below 2^-151 round to zero, so those arguments are
// answered without evaluation.
fn nm_exp2_mr(a: u32) -> nm_mr {
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (a == 0x7f800000u) {
        return nm_mr_bits(a);
    }
    let aa = a & 0x7fffffffu;
    if (aa >= 0x43000000u) {
        if ((a >> 31u) == 0u) {
            return nm_mr_bits(0x7f800000u);
        }
        if (aa >= 0x43180000u) {
            return nm_mr_bits(0u);
        }
    }
    return nm_mr_xf(nm_xf_exp2(nm_xf_from_f32(a)));
}

// e^x and 10^x: x times log2(e) or log2(10) in extended precision, then 2^y. pos and neg are the magnitudes past
// which the result overflows or underflows to zero.
fn nm_expk_mr(a: u32, k: nm_xf, pos: u32, neg: u32) -> nm_mr {
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (a == 0x7f800000u) {
        return nm_mr_bits(a);
    }
    let aa = a & 0x7fffffffu;
    if ((a >> 31u) == 0u && aa >= pos) {
        return nm_mr_bits(0x7f800000u);
    }
    if ((a >> 31u) != 0u && aa >= neg) {
        return nm_mr_bits(0u);
    }
    return nm_mr_xf(nm_xf_exp2(nm_xf_mul(nm_xf_from_f32(a), k)));
}

// e^x: e^89 > 2^128; e^-105 < 2^-151.
fn nm_exp_mr(a: u32) -> nm_mr {
    return nm_expk_mr(a, nm_xf_log2e, 0x42b20000u, 0x42d20000u);
}

// 10^x: 10^39 > 2^128; 10^-46 < 2^-151.
fn nm_exp10_mr(a: u32) -> nm_mr {
    return nm_expk_mr(a, nm_xf_log2_10, 0x421c0000u, 0x42380000u);
}

// Logarithms: kind 0 natural, 1 base 2, 2 base 10.
fn nm_log_mr(a: u32, kind: u32) -> nm_mr {
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if ((a & 0x7fffffffu) == 0u) {
        return nm_mr_bits(0xff800000u);
    }
    if ((a >> 31u) != 0u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    if (a == 0x7f800000u) {
        return nm_mr_bits(a);
    }
    let p = nm_xf_ln_parts(nm_xf_from_f32(a));
    if (kind == 1u) {
        return nm_mr_xf(nm_xf_add(nm_xf_from_i32(p.big_e), nm_xf_mul(p.ln_m, nm_xf_log2e)));
    }
    let ln = nm_xf_add(nm_xf_mul(nm_xf_from_i32(p.big_e), nm_xf_ln2), p.ln_m);
    if (kind == 2u) {
        return nm_mr_xf(nm_xf_mul(ln, nm_xf_ln10_inv));
    }
    return nm_mr_xf(ln);
}

// Whether a finite binary32 value is an integer, and whether it is an odd one.
fn nm_f32_is_int(a: u32) -> bool {
    let e = (a >> 23u) & 0xffu;
    if (e >= 150u) {
        return true;
    }
    if (e < 127u) {
        return (a & 0x7fffffffu) == 0u;
    }
    return (a & ((1u << (150u - e)) - 1u)) == 0u;
}

fn nm_f32_is_odd_int(a: u32) -> bool {
    let e = (a >> 23u) & 0xffu;
    if (e < 127u || e > 150u || !nm_f32_is_int(a)) {
        return false;
    }
    // The units bit: the implicit one when e == 150, otherwise fraction bit 150 - e.
    if (e == 150u) {
        return (a & 1u) != 0u;
    }
    return ((((a & 0x7fffffu) | 0x800000u) >> (150u - e)) & 1u) != 0u;
}

// |x|^y = 2^(y log2 |x|) for finite non-zero x and finite y, with the given sign.
fn nm_pow_core(x: u32, y: u32, neg: bool) -> nm_mr {
    let p = nm_xf_ln_parts(nm_xf_from_f32(x & 0x7fffffffu));
    let l = nm_xf_add(nm_xf_from_i32(p.big_e), nm_xf_mul(p.ln_m, nm_xf_log2e));
    let z = nm_xf_mul(nm_xf_from_f32(y), l);
    if (!nm_xf_is0(z) && z.e >= 8) {
        // |z| >= 256: overflow or a result below every subnormal.
        if (z.s == 0u) {
            return nm_mr_bits(select(0x7f800000u, 0xff800000u, neg));
        }
        return nm_mr_bits(select(0u, 0x80000000u, neg));
    }
    let r = nm_xf_exp2(z);
    return nm_mr_xf(nm_xf(r.m, r.e, select(0u, 1u, neg)));
}

// pow: C99 F.9.4.4.
fn nm_pow_mr(x: u32, y: u32) -> nm_mr {
    let ax = x & 0x7fffffffu;
    let ay = y & 0x7fffffffu;
    let sx = (x >> 31u) != 0u;
    let sy = (y >> 31u) != 0u;
    if (ay == 0u || x == 0x3f800000u) {
        return nm_mr_bits(0x3f800000u);
    }
    if (nm_f32_isnan(x) || nm_f32_isnan(y)) {
        return nm_mr_bits(nm_f32_nan2(x, y));
    }
    let yinf = ay == 0x7f800000u;
    let odd = !yinf && nm_f32_is_odd_int(y);
    if (ax == 0u) {
        if (sy) {
            return nm_mr_bits(select(0x7f800000u, x | 0x7f800000u, odd));
        }
        return nm_mr_bits(select(0u, x, odd));
    }
    if (yinf) {
        if (x == 0xbf800000u) {
            return nm_mr_bits(0x3f800000u);
        }
        // |x| < 1 with y = -inf, or |x| > 1 with y = +inf, grows without bound.
        if ((ax < 0x3f800000u) == sy) {
            return nm_mr_bits(0x7f800000u);
        }
        return nm_mr_bits(0u);
    }
    if (ax == 0x7f800000u) {
        let mag = select(0x7f800000u, 0u, sy);
        return nm_mr_bits(select(mag, mag | 0x80000000u, sx && odd));
    }
    if (sx && !nm_f32_is_int(y)) {
        return nm_mr_bits(nm_f32_dnan);
    }
    return nm_pow_core(x, y, sx && odd);
}

// powr: OpenCL C 3.0 7.5.1.
fn nm_powr_mr(x: u32, y: u32) -> nm_mr {
    let ax = x & 0x7fffffffu;
    let ay = y & 0x7fffffffu;
    if (nm_f32_isnan(x) || nm_f32_isnan(y)) {
        return nm_mr_bits(nm_f32_nan2(x, y));
    }
    if ((x >> 31u) != 0u && ax != 0u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    let sy = (y >> 31u) != 0u;
    if (ax == 0u) {
        if (ay == 0u) {
            return nm_mr_bits(nm_f32_dnan);
        }
        return nm_mr_bits(select(0u, 0x7f800000u, sy));
    }
    if (ax == 0x7f800000u) {
        if (ay == 0u) {
            return nm_mr_bits(nm_f32_dnan);
        }
        return nm_mr_bits(select(0x7f800000u, 0u, sy));
    }
    if (x == 0x3f800000u) {
        if (ay == 0x7f800000u) {
            return nm_mr_bits(nm_f32_dnan);
        }
        return nm_mr_bits(0x3f800000u);
    }
    if (ay == 0u) {
        return nm_mr_bits(0x3f800000u);
    }
    if (ay == 0x7f800000u) {
        if ((ax < 0x3f800000u) == sy) {
            return nm_mr_bits(0x7f800000u);
        }
        return nm_mr_bits(0u);
    }
    return nm_pow_core(x, y, false);
}

// sin, cos and tan of a finite binary32 value after reduction.
fn nm_sin_mr(a: u32) -> nm_mr {
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if ((a & 0x7fffffffu) == 0x7f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    if ((a & 0x7fffffffu) == 0u) {
        return nm_mr_bits(a);
    }
    let r = nm_xf_trig_reduce(a);
    switch (r.q) {
        case 0u: {
            return nm_mr_xf(nm_xf_sin_k(r.theta));
        }
        case 1u: {
            return nm_mr_xf(nm_xf_cos_k(r.theta));
        }
        case 2u: {
            return nm_mr_xf(nm_xf_neg(nm_xf_sin_k(r.theta)));
        }
        default: {
            return nm_mr_xf(nm_xf_neg(nm_xf_cos_k(r.theta)));
        }
    }
}

fn nm_cos_mr(a: u32) -> nm_mr {
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if ((a & 0x7fffffffu) == 0x7f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    if ((a & 0x7fffffffu) == 0u) {
        return nm_mr_bits(0x3f800000u);
    }
    let r = nm_xf_trig_reduce(a);
    switch (r.q) {
        case 0u: {
            return nm_mr_xf(nm_xf_cos_k(r.theta));
        }
        case 1u: {
            return nm_mr_xf(nm_xf_neg(nm_xf_sin_k(r.theta)));
        }
        case 2u: {
            return nm_mr_xf(nm_xf_neg(nm_xf_cos_k(r.theta)));
        }
        default: {
            return nm_mr_xf(nm_xf_sin_k(r.theta));
        }
    }
}

fn nm_tan_mr(a: u32) -> nm_mr {
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if ((a & 0x7fffffffu) == 0x7f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    if ((a & 0x7fffffffu) == 0u) {
        return nm_mr_bits(a);
    }
    let r = nm_xf_trig_reduce(a);
    let s = nm_xf_sin_k(r.theta);
    let c = nm_xf_cos_k(r.theta);
    if ((r.q & 1u) == 0u) {
        return nm_mr_xf(nm_xf_div(s, c));
    }
    return nm_mr_xf(nm_xf_neg(nm_xf_div(c, s)));
}

// Reduction for the pi functions: |x| = (q + 2 r) / 2 exactly, q an integer kept mod 4, |r| <= 1/4; theta = r pi.
// Called for |x| < 2^23.
fn nm_pi_reduce(aa: u32) -> nm_xf_trig {
    let sp = nm_xf_round_int(nm_xf_ldexp(nm_xf_from_f32(aa), 1));
    let r = nm_xf_ldexp(sp.f, -1);
    return nm_xf_trig(u32(sp.k) & 3u, nm_xf_mul(r, nm_xf_pi));
}

// The parity of an integer binary32 magnitude in [2^23, 2^24); larger ones are even.
fn nm_f32_big_int_odd(aa: u32) -> bool {
    return aa < 0x4b800000u && (aa & 1u) != 0u;
}

// sinpi: sinpi(+-0) = +-0, +0 for positive integers, -0 for negative ones (OpenCL C 3.0 7.5.1).
fn nm_sinpi_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa == 0x7f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    let neg = (a >> 31u) != 0u;
    if (aa >= 0x4b000000u) {
        return nm_mr_bits(a & 0x80000000u);
    }
    let r = nm_pi_reduce(aa);
    if (nm_xf_is0(r.theta) && (r.q & 1u) == 0u) {
        return nm_mr_bits(a & 0x80000000u);
    }
    var v: nm_xf;
    switch (r.q) {
        case 0u: {
            v = nm_xf_sin_k(r.theta);
        }
        case 1u: {
            v = nm_xf_cos_k(r.theta);
        }
        case 2u: {
            v = nm_xf_neg(nm_xf_sin_k(r.theta));
        }
        default: {
            v = nm_xf_neg(nm_xf_cos_k(r.theta));
        }
    }
    return nm_mr_neg(nm_mr_xf(v), neg);
}

// cospi: cospi(+-0) = 1, cospi(n + 0.5) = +0 (OpenCL C 3.0 7.5.1).
fn nm_cospi_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa == 0x7f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    if (aa >= 0x4b000000u) {
        return nm_mr_bits(select(0x3f800000u, 0xbf800000u, nm_f32_big_int_odd(aa)));
    }
    let r = nm_pi_reduce(aa);
    if (nm_xf_is0(r.theta) && (r.q & 1u) != 0u) {
        return nm_mr_bits(0u);
    }
    switch (r.q) {
        case 0u: {
            return nm_mr_xf(nm_xf_cos_k(r.theta));
        }
        case 1u: {
            return nm_mr_xf(nm_xf_neg(nm_xf_sin_k(r.theta)));
        }
        case 2u: {
            return nm_mr_xf(nm_xf_neg(nm_xf_cos_k(r.theta)));
        }
        default: {
            return nm_mr_xf(nm_xf_sin_k(r.theta));
        }
    }
}

// tanpi: tanpi(+-0) = +-0; tanpi(n) = copysign(0, n) for even n and copysign(0, -n) for odd n; tanpi(n + 0.5) is +inf
// for even n and -inf for odd n (OpenCL C 3.0 7.5.1). All of these are odd in x, so |x| is evaluated and the sign
// applied.
fn nm_tanpi_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa == 0x7f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    let neg = (a >> 31u) != 0u;
    if (aa >= 0x4b000000u) {
        return nm_mr_neg(nm_mr_bits(select(0u, 0x80000000u, nm_f32_big_int_odd(aa))), neg);
    }
    let r = nm_pi_reduce(aa);
    if (nm_xf_is0(r.theta)) {
        var b = 0u;
        switch (r.q) {
            case 0u: {
                b = 0u;
            }
            case 1u: {
                b = 0x7f800000u;
            }
            case 2u: {
                b = 0x80000000u;
            }
            default: {
                b = 0xff800000u;
            }
        }
        return nm_mr_neg(nm_mr_bits(b), neg);
    }
    let s = nm_xf_sin_k(r.theta);
    let c = nm_xf_cos_k(r.theta);
    var v: nm_xf;
    if ((r.q & 1u) == 0u) {
        v = nm_xf_div(s, c);
    } else {
        v = nm_xf_neg(nm_xf_div(c, s));
    }
    return nm_mr_neg(nm_mr_xf(v), neg);
}

// Inverse trigonometric functions (C99 F.9.1).
fn nm_atan_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa == 0u) {
        return nm_mr_bits(a);
    }
    let neg = (a >> 31u) != 0u;
    if (aa == 0x7f800000u) {
        return nm_mr_neg(nm_mr_xf(nm_xf_pi_2), neg);
    }
    return nm_mr_neg(nm_mr_xf(nm_xf_atan_ratio(nm_xf_from_f32(aa), nm_xf_one())), neg);
}

fn nm_atan2_mr(y: u32, x: u32) -> nm_mr {
    if (nm_f32_isnan(y) || nm_f32_isnan(x)) {
        return nm_mr_bits(nm_f32_nan2(y, x));
    }
    let ay = y & 0x7fffffffu;
    let ax = x & 0x7fffffffu;
    let ny = (y >> 31u) != 0u;
    let nx = (x >> 31u) != 0u;
    var v: nm_xf;
    if (ay == 0u) {
        if (!nx) {
            return nm_mr_bits(y);
        }
        v = nm_xf_pi;
    } else if (ax == 0u) {
        v = nm_xf_pi_2;
    } else if (ay == 0x7f800000u) {
        if (ax == 0x7f800000u) {
            if (nx) {
                v = nm_xf_3pi_4;
            } else {
                v = nm_xf_pi_4;
            }
        } else {
            v = nm_xf_pi_2;
        }
    } else if (ax == 0x7f800000u) {
        if (!nx) {
            return nm_mr_bits(y & 0x80000000u);
        }
        v = nm_xf_pi;
    } else {
        v = nm_xf_atan_ratio(nm_xf_from_f32(ay), nm_xf_from_f32(ax));
        if (nx) {
            v = nm_xf_sub(nm_xf_pi, v);
        }
    }
    return nm_mr_neg(nm_mr_xf(v), ny);
}

// sqrt(1 - x^2) for |x| <= 1, from the exact product (1 - |x|)(1 + |x|).
fn nm_xf_cosine_of(ax: nm_xf) -> nm_xf {
    let one = nm_xf_one();
    let d = nm_xf_mul(nm_xf_sub(one, ax), nm_xf_add(one, ax));
    if (nm_xf_is0(d)) {
        return d;
    }
    return nm_xf_sqrt(d);
}

fn nm_asin_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa > 0x3f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    if (aa == 0u) {
        return nm_mr_bits(a);
    }
    let ax = nm_xf_from_f32(aa);
    return nm_mr_neg(nm_mr_xf(nm_xf_atan_ratio(ax, nm_xf_cosine_of(ax))), (a >> 31u) != 0u);
}

fn nm_acos_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa > 0x3f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    if (a == 0x3f800000u) {
        return nm_mr_bits(0u);
    }
    let ax = nm_xf_from_f32(aa);
    let v = nm_xf_atan_ratio(nm_xf_cosine_of(ax), ax);
    if ((a >> 31u) != 0u) {
        return nm_mr_xf(nm_xf_sub(nm_xf_pi, v));
    }
    return nm_mr_xf(v);
}

// Hyperbolic functions (C99 F.9.2).
fn nm_sinh_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa == 0u || aa == 0x7f800000u) {
        return nm_mr_bits(a);
    }
    let neg = (a >> 31u) != 0u;
    if (aa >= 0x42b40000u) {
        // |x| >= 90: sinh(x) > 2^128.
        return nm_mr_bits((a & 0x80000000u) | 0x7f800000u);
    }
    let ax = nm_xf_from_f32(aa);
    if (aa < 0x3f800000u) {
        return nm_mr_neg(nm_mr_xf(nm_xf_mul(ax, nm_xf_poly_sinh(nm_xf_mul(ax, ax)))), neg);
    }
    let e = nm_xf_exp(ax);
    let v = nm_xf_ldexp(nm_xf_sub(e, nm_xf_div(nm_xf_one(), e)), -1);
    return nm_mr_neg(nm_mr_xf(v), neg);
}

fn nm_cosh_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa >= 0x42b40000u) {
        return nm_mr_bits(0x7f800000u);
    }
    let e = nm_xf_exp(nm_xf_from_f32(aa));
    return nm_mr_xf(nm_xf_ldexp(nm_xf_add(e, nm_xf_div(nm_xf_one(), e)), -1));
}

fn nm_tanh_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa == 0u) {
        return nm_mr_bits(a);
    }
    let neg = (a >> 31u) != 0u;
    if (aa >= 0x41800000u) {
        // |x| >= 16: 1 - tanh|x| < 2^-46, which rounds to 1.
        return nm_mr_bits((a & 0x80000000u) | 0x3f800000u);
    }
    let ax = nm_xf_from_f32(aa);
    var v: nm_xf;
    if (aa < 0x3c800000u) {
        v = nm_xf_mul(ax, nm_xf_poly_tanh(nm_xf_mul(ax, ax)));
    } else if (aa < 0x3f0ccccdu) {
        // |x| < 0.55: tanh = E / (E + 2), E = e^2|x| - 1.
        let e = nm_xf_expm1(nm_xf_ldexp(ax, 1));
        v = nm_xf_div(e, nm_xf_add(e, nm_xf_from_i32(2)));
    } else {
        let t = nm_xf_exp(nm_xf_ldexp(ax, 1));
        v = nm_xf_sub(nm_xf_one(), nm_xf_div(nm_xf_from_i32(2), nm_xf_add(t, nm_xf_one())));
    }
    return nm_mr_neg(nm_mr_xf(v), neg);
}

fn nm_asinh_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa == 0u || aa == 0x7f800000u) {
        return nm_mr_bits(a);
    }
    let neg = (a >> 31u) != 0u;
    let ax = nm_xf_from_f32(aa);
    let one = nm_xf_one();
    let x2 = nm_xf_mul(ax, ax);
    var v: nm_xf;
    if (aa < 0x3f000000u) {
        // ln(1 + u), u = |x| + x^2 / (1 + sqrt(1 + x^2)).
        let u = nm_xf_add(ax, nm_xf_div(x2, nm_xf_add(one, nm_xf_sqrt(nm_xf_add(one, x2)))));
        v = nm_xf_ln1p(u);
    } else {
        v = nm_xf_ln(nm_xf_add(ax, nm_xf_sqrt(nm_xf_add(x2, one))));
    }
    return nm_mr_neg(nm_mr_xf(v), neg);
}

fn nm_acosh_mr(a: u32) -> nm_mr {
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if ((a >> 31u) != 0u || a < 0x3f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    if (a == 0x3f800000u) {
        return nm_mr_bits(0u);
    }
    if (a == 0x7f800000u) {
        return nm_mr_bits(a);
    }
    let x = nm_xf_from_f32(a);
    let one = nm_xf_one();
    if (a < 0x40000000u) {
        // ln(1 + u), u = (x - 1) + sqrt((x - 1)(x + 1)); both factors are exact.
        let xm1 = nm_xf_sub(x, one);
        let u = nm_xf_add(xm1, nm_xf_sqrt(nm_xf_mul(xm1, nm_xf_add(x, one))));
        return nm_mr_xf(nm_xf_ln1p(u));
    }
    return nm_mr_xf(nm_xf_ln(nm_xf_add(x, nm_xf_sqrt(nm_xf_sub(nm_xf_mul(x, x), one)))));
}

fn nm_atanh_mr(a: u32) -> nm_mr {
    let aa = a & 0x7fffffffu;
    if (nm_f32_isnan(a)) {
        return nm_mr_bits(a | 0x00400000u);
    }
    if (aa > 0x3f800000u) {
        return nm_mr_bits(nm_f32_dnan);
    }
    if (aa == 0u) {
        return nm_mr_bits(a);
    }
    if (aa == 0x3f800000u) {
        return nm_mr_bits((a & 0x80000000u) | 0x7f800000u);
    }
    let neg = (a >> 31u) != 0u;
    let ax = nm_xf_from_f32(aa);
    var v: nm_xf;
    if (aa < 0x3e2e147bu) {
        // |x| < 0.17: the atanh series directly.
        v = nm_xf_mul(ax, nm_xf_poly_atanh(nm_xf_mul(ax, ax)));
    } else {
        let one = nm_xf_one();
        v = nm_xf_ldexp(nm_xf_ln(nm_xf_div(nm_xf_add(one, ax), nm_xf_sub(one, ax))), -1);
    }
    return nm_mr_neg(nm_mr_xf(v), neg);
}

// Binary32 entry points.
fn nm_f32_exp(a: u32) -> u32 {
    return nm_mr_f32(nm_exp_mr(a));
}

fn nm_f32_exp2(a: u32) -> u32 {
    return nm_mr_f32(nm_exp2_mr(a));
}

fn nm_f32_exp10(a: u32) -> u32 {
    return nm_mr_f32(nm_exp10_mr(a));
}

fn nm_f32_log(a: u32) -> u32 {
    return nm_mr_f32(nm_log_mr(a, 0u));
}

fn nm_f32_log2(a: u32) -> u32 {
    return nm_mr_f32(nm_log_mr(a, 1u));
}

fn nm_f32_log10(a: u32) -> u32 {
    return nm_mr_f32(nm_log_mr(a, 2u));
}

fn nm_f32_pow(x: u32, y: u32) -> u32 {
    return nm_mr_f32(nm_pow_mr(x, y));
}

fn nm_f32_powr(x: u32, y: u32) -> u32 {
    return nm_mr_f32(nm_powr_mr(x, y));
}

fn nm_f32_sin(a: u32) -> u32 {
    return nm_mr_f32(nm_sin_mr(a));
}

fn nm_f32_cos(a: u32) -> u32 {
    return nm_mr_f32(nm_cos_mr(a));
}

fn nm_f32_tan(a: u32) -> u32 {
    return nm_mr_f32(nm_tan_mr(a));
}

fn nm_f32_sinpi(a: u32) -> u32 {
    return nm_mr_f32(nm_sinpi_mr(a));
}

fn nm_f32_cospi(a: u32) -> u32 {
    return nm_mr_f32(nm_cospi_mr(a));
}

fn nm_f32_tanpi(a: u32) -> u32 {
    return nm_mr_f32(nm_tanpi_mr(a));
}

fn nm_f32_asin(a: u32) -> u32 {
    return nm_mr_f32(nm_asin_mr(a));
}

fn nm_f32_acos(a: u32) -> u32 {
    return nm_mr_f32(nm_acos_mr(a));
}

fn nm_f32_atan(a: u32) -> u32 {
    return nm_mr_f32(nm_atan_mr(a));
}

fn nm_f32_atan2(y: u32, x: u32) -> u32 {
    return nm_mr_f32(nm_atan2_mr(y, x));
}

fn nm_f32_sinh(a: u32) -> u32 {
    return nm_mr_f32(nm_sinh_mr(a));
}

fn nm_f32_cosh(a: u32) -> u32 {
    return nm_mr_f32(nm_cosh_mr(a));
}

fn nm_f32_tanh(a: u32) -> u32 {
    return nm_mr_f32(nm_tanh_mr(a));
}

fn nm_f32_asinh(a: u32) -> u32 {
    return nm_mr_f32(nm_asinh_mr(a));
}

fn nm_f32_acosh(a: u32) -> u32 {
    return nm_mr_f32(nm_acosh_mr(a));
}

fn nm_f32_atanh(a: u32) -> u32 {
    return nm_mr_f32(nm_atanh_mr(a));
}

// sin and cos together: (sin, cos).
fn nm_f32_sincos(a: u32) -> vec2<u32> {
    return vec2<u32>(nm_f32_sin(a), nm_f32_cos(a));
}

// Binary16 entry points: the same evaluation rounded once to binary16.
fn nm_f16_exp(h: u32) -> u32 {
    return nm_mr_f16(nm_exp_mr(nm_f16_to_f32(h)));
}

fn nm_f16_exp2(h: u32) -> u32 {
    return nm_mr_f16(nm_exp2_mr(nm_f16_to_f32(h)));
}

fn nm_f16_exp10(h: u32) -> u32 {
    return nm_mr_f16(nm_exp10_mr(nm_f16_to_f32(h)));
}

fn nm_f16_log(h: u32) -> u32 {
    return nm_mr_f16(nm_log_mr(nm_f16_to_f32(h), 0u));
}

fn nm_f16_log2(h: u32) -> u32 {
    return nm_mr_f16(nm_log_mr(nm_f16_to_f32(h), 1u));
}

fn nm_f16_log10(h: u32) -> u32 {
    return nm_mr_f16(nm_log_mr(nm_f16_to_f32(h), 2u));
}

fn nm_f16_pow(x: u32, y: u32) -> u32 {
    return nm_mr_f16(nm_pow_mr(nm_f16_to_f32(x), nm_f16_to_f32(y)));
}

fn nm_f16_powr(x: u32, y: u32) -> u32 {
    return nm_mr_f16(nm_powr_mr(nm_f16_to_f32(x), nm_f16_to_f32(y)));
}

fn nm_f16_sin(h: u32) -> u32 {
    return nm_mr_f16(nm_sin_mr(nm_f16_to_f32(h)));
}

fn nm_f16_cos(h: u32) -> u32 {
    return nm_mr_f16(nm_cos_mr(nm_f16_to_f32(h)));
}

fn nm_f16_tan(h: u32) -> u32 {
    return nm_mr_f16(nm_tan_mr(nm_f16_to_f32(h)));
}

fn nm_f16_sinpi(h: u32) -> u32 {
    return nm_mr_f16(nm_sinpi_mr(nm_f16_to_f32(h)));
}

fn nm_f16_cospi(h: u32) -> u32 {
    return nm_mr_f16(nm_cospi_mr(nm_f16_to_f32(h)));
}

fn nm_f16_tanpi(h: u32) -> u32 {
    return nm_mr_f16(nm_tanpi_mr(nm_f16_to_f32(h)));
}

fn nm_f16_asin(h: u32) -> u32 {
    return nm_mr_f16(nm_asin_mr(nm_f16_to_f32(h)));
}

fn nm_f16_acos(h: u32) -> u32 {
    return nm_mr_f16(nm_acos_mr(nm_f16_to_f32(h)));
}

fn nm_f16_atan(h: u32) -> u32 {
    return nm_mr_f16(nm_atan_mr(nm_f16_to_f32(h)));
}

fn nm_f16_atan2(y: u32, x: u32) -> u32 {
    return nm_mr_f16(nm_atan2_mr(nm_f16_to_f32(y), nm_f16_to_f32(x)));
}

fn nm_f16_sinh(h: u32) -> u32 {
    return nm_mr_f16(nm_sinh_mr(nm_f16_to_f32(h)));
}

fn nm_f16_cosh(h: u32) -> u32 {
    return nm_mr_f16(nm_cosh_mr(nm_f16_to_f32(h)));
}

fn nm_f16_tanh(h: u32) -> u32 {
    return nm_mr_f16(nm_tanh_mr(nm_f16_to_f32(h)));
}

fn nm_f16_asinh(h: u32) -> u32 {
    return nm_mr_f16(nm_asinh_mr(nm_f16_to_f32(h)));
}

fn nm_f16_acosh(h: u32) -> u32 {
    return nm_mr_f16(nm_acosh_mr(nm_f16_to_f32(h)));
}

fn nm_f16_atanh(h: u32) -> u32 {
    return nm_mr_f16(nm_atanh_mr(nm_f16_to_f32(h)));
}

fn nm_f16_sincos(h: u32) -> vec2<u32> {
    return vec2<u32>(nm_f16_sin(h), nm_f16_cos(h));
}
