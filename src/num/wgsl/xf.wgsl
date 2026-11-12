// Extended precision: binary floating point with a 64-bit significand and a 32-bit exponent, used inside the
// transcendental functions. The functions reduce their argument exactly or with errors near 2^-64, evaluate
// series in this format and round once to binary32 or binary16 at the end, so the final error stays a hair above
// half an ulp (bounded per function in tests/num/math.rs).
//
// nm_xf: value = (-1)^s * m * 2^(e - 63) with m normalized (bit 63 set), or zero when m is 0. Each operation
// rounds its result to nearest (ties away), a relative error of at most 2^-64.

struct nm_xf {
    m: vec2<u32>,
    e: i32,
    s: u32,
}

fn nm_xf_is0(a: nm_xf) -> bool {
    return (a.m.x | a.m.y) == 0u;
}

fn nm_xf_neg(a: nm_xf) -> nm_xf {
    return nm_xf(a.m, a.e, a.s ^ 1u);
}

fn nm_xf_ldexp(a: nm_xf, k: i32) -> nm_xf {
    return nm_xf(a.m, a.e + k, a.s);
}

// A finite binary32 value, exactly.
fn nm_xf_from_f32(a: u32) -> nm_xf {
    let aa = a & 0x7fffffffu;
    if (aa == 0u) {
        return nm_xf(vec2<u32>(0u, 0u), 0, a >> 31u);
    }
    let n = nm_f32_norm(aa);
    return nm_xf(vec2<u32>(0u, n.m << 8u), n.e - 127, a >> 31u);
}

fn nm_xf_from_i32(k: i32) -> nm_xf {
    if (k == 0) {
        return nm_xf(vec2<u32>(0u, 0u), 0, 0u);
    }
    var v = u32(k);
    if (k < 0) {
        v = 0u - v;
    }
    let lz = countLeadingZeros(v);
    return nm_xf(vec2<u32>(0u, v << lz), 31 - i32(lz), select(0u, 1u, k < 0));
}

// Round to nearest even, to binary32 bits.
fn nm_xf_to_f32(a: nm_xf) -> u32 {
    if (nm_xf_is0(a)) {
        return a.s << 31u;
    }
    return nm_f32_rpack(a.s, clamp(a.e, -400, 400) + 127, nm_shr64_jam(a.m, 33u).x);
}

// Round to nearest even, to binary16 bits.
fn nm_xf_to_f16(a: nm_xf) -> u32 {
    if (nm_xf_is0(a)) {
        return a.s << 15u;
    }
    return nm_f16_rpack(a.s << 15u, clamp(a.e, -400, 400) + 15, nm_shr64_jam(a.m, 33u).x);
}

// Round a 64-bit significand with a round bit; a carry out renormalizes.
fn nm_xf_round(s: u32, e: i32, m: vec2<u32>, up: bool) -> nm_xf {
    if (!up) {
        return nm_xf(m, e, s);
    }
    let r = nm_add64(m, vec2<u32>(1u, 0u));
    if (nm_is0_64(r)) {
        return nm_xf(vec2<u32>(0u, 0x80000000u), e + 1, s);
    }
    return nm_xf(r, e, s);
}

fn nm_xf_mul(a: nm_xf, b: nm_xf) -> nm_xf {
    let s = a.s ^ b.s;
    if (nm_xf_is0(a) || nm_xf_is0(b)) {
        return nm_xf(vec2<u32>(0u, 0u), 0, s);
    }
    let p = nm_mul64w(a.m, b.m);
    if ((p.w >> 31u) != 0u) {
        return nm_xf_round(s, a.e + b.e + 1, vec2<u32>(p.z, p.w), (p.y >> 31u) != 0u);
    }
    let hi = vec2<u32>((p.z << 1u) | (p.y >> 31u), (p.w << 1u) | (p.z >> 31u));
    return nm_xf_round(s, a.e + b.e, hi, ((p.y >> 30u) & 1u) != 0u);
}

// 96-bit logical shifts by n in [0, 95].
fn nm_shr96(a: vec3<u32>, n: u32) -> vec3<u32> {
    if (n == 0u) {
        return a;
    }
    if (n < 32u) {
        return vec3<u32>((a.x >> n) | (a.y << (32u - n)), (a.y >> n) | (a.z << (32u - n)), a.z >> n);
    }
    if (n == 32u) {
        return vec3<u32>(a.y, a.z, 0u);
    }
    if (n < 64u) {
        let k = n - 32u;
        return vec3<u32>((a.y >> k) | (a.z << (32u - k)), a.z >> k, 0u);
    }
    if (n == 64u) {
        return vec3<u32>(a.z, 0u, 0u);
    }
    return vec3<u32>(a.z >> (n - 64u), 0u, 0u);
}

fn nm_shl96(a: vec3<u32>, n: u32) -> vec3<u32> {
    if (n == 0u) {
        return a;
    }
    if (n < 32u) {
        return vec3<u32>(a.x << n, (a.y << n) | (a.x >> (32u - n)), (a.z << n) | (a.y >> (32u - n)));
    }
    if (n == 32u) {
        return vec3<u32>(0u, a.x, a.y);
    }
    if (n < 64u) {
        let k = n - 32u;
        return vec3<u32>(0u, a.x << k, (a.y << k) | (a.x >> (32u - k)));
    }
    if (n == 64u) {
        return vec3<u32>(0u, 0u, a.x);
    }
    return vec3<u32>(0u, 0u, a.x << (n - 64u));
}

fn nm_clz96(a: vec3<u32>) -> u32 {
    if (a.z != 0u) {
        return countLeadingZeros(a.z);
    }
    if (a.y != 0u) {
        return 32u + countLeadingZeros(a.y);
    }
    return 64u + countLeadingZeros(a.x);
}

fn nm_xf_add(a: nm_xf, b: nm_xf) -> nm_xf {
    if (nm_xf_is0(a)) {
        return b;
    }
    if (nm_xf_is0(b)) {
        return a;
    }
    var x = a;
    var y = b;
    if (b.e > a.e || (b.e == a.e && nm_lt64(a.m, b.m))) {
        x = b;
        y = a;
    }
    let d = x.e - y.e;
    if (d >= 96) {
        return x;
    }
    let big = vec3<u32>(0u, x.m.x, x.m.y);
    let small = nm_shr96(vec3<u32>(0u, y.m.x, y.m.y), u32(d));
    var r: vec3<u32>;
    var e = x.e;
    if (x.s == y.s) {
        // Halve both so the sum stays below 2^96; the bit dropped from small is below 2^-95 of the result.
        r = nm_add96(nm_shr96(big, 1u), nm_shr96(small, 1u));
        e = e + 1;
    } else {
        // Alignments by 0 or 1 place are exact, so a cancelling difference is exact.
        r = nm_sub96(big, small);
        if ((r.x | r.y | r.z) == 0u) {
            return nm_xf(vec2<u32>(0u, 0u), 0, 0u);
        }
    }
    let lz = nm_clz96(r);
    r = nm_shl96(r, lz);
    return nm_xf_round(x.s, e - i32(lz), vec2<u32>(r.y, r.z), (r.x >> 31u) != 0u);
}

fn nm_xf_sub(a: nm_xf, b: nm_xf) -> nm_xf {
    return nm_xf_add(a, nm_xf_neg(b));
}

// a / b for non-zero b: 64 quotient bits by restoring division and a round bit.
fn nm_xf_div(a: nm_xf, b: nm_xf) -> nm_xf {
    let s = a.s ^ b.s;
    if (nm_xf_is0(a)) {
        return nm_xf(vec2<u32>(0u, 0u), 0, s);
    }
    var r = a.m;
    var e = a.e - b.e;
    // The remainder is carry * 2^64 + r and stays below 2 * b.m.
    var carry = false;
    if (nm_lt64(r, b.m)) {
        carry = (r.y >> 31u) != 0u;
        r = nm_shl64(r, 1u);
        e = e - 1;
    }
    var q = vec2<u32>(0u, 0u);
    for (var i = 0u; i < 64u; i = i + 1u) {
        q = nm_shl64(q, 1u);
        if (carry || !nm_lt64(r, b.m)) {
            r = nm_sub64(r, b.m);
            q.x = q.x | 1u;
        }
        carry = (r.y >> 31u) != 0u;
        r = nm_shl64(r, 1u);
    }
    return nm_xf_round(s, e, q, carry || !nm_lt64(r, b.m));
}

// sqrt(a) for a > 0: 64 root bits, digit by digit over a 128-bit radicand, truncated.
fn nm_xf_sqrt(a: nm_xf) -> nm_xf {
    let odd = (a.e & 1) != 0;
    // Radicand m * 2^64 for an odd exponent, m * 2^63 for an even one.
    var rad = vec4<u32>(0u, 0u, a.m.x, a.m.y);
    if (!odd) {
        rad = vec4<u32>(0u, a.m.x << 31u, (a.m.y << 31u) | (a.m.x >> 1u), a.m.y >> 1u);
    }
    var res = vec2<u32>(0u, 0u);
    var rem = vec3<u32>(0u, 0u, 0u);
    for (var k = 63i; k >= 0i; k = k - 1i) {
        let w = rad[u32(k) >> 4u];
        let pair = (w >> (2u * (u32(k) & 15u))) & 3u;
        rem = nm_shl96(rem, 2u);
        rem.x = rem.x | pair;
        let trial = nm_shl96(vec3<u32>(res.x, res.y, 0u), 2u);
        res = nm_shl64(res, 1u);
        let t = vec3<u32>(trial.x | 1u, trial.y, trial.z);
        if (nm_le96(t, rem)) {
            rem = nm_sub96(rem, t);
            res.x = res.x | 1u;
        }
    }
    return nm_xf(res, (a.e - select(0, 1, odd)) >> 1u, 0u);
}

// Nearest integer k (ties away from zero) to x with |x| < 2^30, and x - k, exact.
struct nm_xf_split {
    k: i32,
    f: nm_xf,
}

fn nm_xf_round_int(x: nm_xf) -> nm_xf_split {
    if (nm_xf_is0(x) || x.e < -1) {
        return nm_xf_split(0, x);
    }
    var ki = 1u;
    if (x.e >= 0) {
        let sh = u32(63 - x.e);
        ki = nm_shr64(x.m, sh).x + ((nm_shr64(x.m, sh - 1u).x) & 1u);
    }
    let k = select(i32(ki), -i32(ki), x.s != 0u);
    return nm_xf_split(k, nm_xf_add(x, nm_xf_from_i32(-k)));
}

// 2^x for |x| < 2^9: x = k + f with |f| <= 1/2, 2^f = e^(f ln 2) by its Taylor series.
fn nm_xf_exp2(x: nm_xf) -> nm_xf {
    let r = nm_xf_round_int(x);
    return nm_xf_ldexp(nm_xf_poly_exp(nm_xf_mul(r.f, nm_xf_ln2)), r.k);
}

// e^x for |x| < 2^8.
fn nm_xf_exp(x: nm_xf) -> nm_xf {
    return nm_xf_exp2(nm_xf_mul(x, nm_xf_log2e));
}

// e^x - 1 for |x| < 2^8, without cancellation for small x.
fn nm_xf_expm1(x: nm_xf) -> nm_xf {
    if (nm_xf_is0(x) || x.e < -4) {
        return nm_xf_mul(x, nm_xf_poly_expm1(x));
    }
    return nm_xf_sub(nm_xf_exp(x), nm_xf_from_i32(1));
}

// ln x for x > 0: x = M * 2^E with M in [sqrt(1/2), sqrt(2)), ln M = 2 atanh(t), t = (M - 1) / (M + 1).
struct nm_xf_lnparts {
    big_e: i32,
    ln_m: nm_xf,
}

fn nm_xf_ln_parts(x: nm_xf) -> nm_xf_lnparts {
    // sqrt(2) * 2^63 = 0xb504f333f9de6484...
    var mm = nm_xf(x.m, 0, 0u);
    var big_e = x.e;
    if (x.m.y > 0xb504f333u || (x.m.y == 0xb504f333u && x.m.x >= 0xf9de6484u)) {
        mm = nm_xf(x.m, -1, 0u);
        big_e = big_e + 1;
    }
    let one = nm_xf_from_i32(1);
    let t = nm_xf_div(nm_xf_sub(mm, one), nm_xf_add(mm, one));
    let ln_m = nm_xf_ldexp(nm_xf_mul(t, nm_xf_poly_atanh(nm_xf_mul(t, t))), 1);
    return nm_xf_lnparts(big_e, ln_m);
}

fn nm_xf_ln(x: nm_xf) -> nm_xf {
    let p = nm_xf_ln_parts(x);
    return nm_xf_add(nm_xf_mul(nm_xf_from_i32(p.big_e), nm_xf_ln2), p.ln_m);
}

fn nm_xf_log2(x: nm_xf) -> nm_xf {
    let p = nm_xf_ln_parts(x);
    return nm_xf_add(nm_xf_from_i32(p.big_e), nm_xf_mul(p.ln_m, nm_xf_log2e));
}

// ln(1 + u) for u >= 0: 2 atanh(u / (2 + u)) while that ratio is at most 3 - 2 sqrt 2, otherwise ln of 1 + u.
fn nm_xf_ln1p(u: nm_xf) -> nm_xf {
    if (nm_xf_is0(u)) {
        return u;
    }
    // sqrt(2) - 1 = 0.4142..: 0xd413cccfe7799211 * 2^-65
    let small = u.e < -2 || (u.e == -2 && (u.m.y < 0xd413cccfu || (u.m.y == 0xd413cccfu && u.m.x <= 0xe7799211u)));
    if (small) {
        let t = nm_xf_div(u, nm_xf_add(nm_xf_from_i32(2), u));
        return nm_xf_ldexp(nm_xf_mul(t, nm_xf_poly_atanh(nm_xf_mul(t, t))), 1);
    }
    return nm_xf_ln(nm_xf_add(nm_xf_from_i32(1), u));
}

// atan(y) for 0 <= y <= 1: atan(k/8) + atan(t), t = (y - k/8) / (1 + y k/8), |t| <= 1/16.
fn nm_xf_atan01(y: nm_xf) -> nm_xf {
    // k = round(8y)
    let k = nm_xf_round_int(nm_xf_ldexp(y, 3)).k;
    if (k == 0) {
        return nm_xf_mul(y, nm_xf_poly_atan(nm_xf_mul(y, y)));
    }
    let c = nm_xf_ldexp(nm_xf_from_i32(k), -3);
    let t = nm_xf_div(nm_xf_sub(y, c), nm_xf_add(nm_xf_from_i32(1), nm_xf_mul(y, c)));
    return nm_xf_add(nm_xf_atan_tab(u32(k)), nm_xf_mul(t, nm_xf_poly_atan(nm_xf_mul(t, t))));
}

// atan(num / den) for num, den >= 0, not both zero, computing only the quotient that is at most 1.
fn nm_xf_atan_ratio(num: nm_xf, den: nm_xf) -> nm_xf {
    if (nm_xf_is0(num)) {
        return num;
    }
    if (nm_xf_is0(den)) {
        return nm_xf_pi_2;
    }
    let num_larger = num.e > den.e || (num.e == den.e && nm_lt64(den.m, num.m));
    if (num_larger) {
        return nm_xf_sub(nm_xf_pi_2, nm_xf_atan01(nm_xf_div(den, num)));
    }
    return nm_xf_atan01(nm_xf_div(num, den));
}

// sin and cos of |x| <= pi/4.
fn nm_xf_sin_k(x: nm_xf) -> nm_xf {
    return nm_xf_mul(x, nm_xf_poly_sin(nm_xf_mul(x, x)));
}

fn nm_xf_cos_k(x: nm_xf) -> nm_xf {
    return nm_xf_poly_cos(nm_xf_mul(x, x));
}

// Reduction of a finite binary32 x for the trigonometric functions: x = (q + r) * pi/2 with integer q (only q mod 4
// is kept) and |r| <= 1/2; theta = r * pi/2. Arguments up to pi/4 are used as they are. Larger ones are multiplied
// exactly by enough bits of 2/pi (the bits whose products are multiples of 4 are skipped, and those past the
// window contribute less than 2^-103).
struct nm_xf_trig {
    q: u32,
    theta: nm_xf,
}

fn nm_xf_trig_reduce(a: u32) -> nm_xf_trig {
    let aa = a & 0x7fffffffu;
    let s = a >> 31u;
    // pi/4 rounded down to binary32 is 0x3f490fda.
    if (aa <= 0x3f490fdau) {
        return nm_xf_trig(0u, nm_xf_from_f32(a));
    }
    let n = nm_f32_norm(aa);
    let big_e = n.e - 150;
    var j0 = 0u;
    if (big_e >= 2) {
        j0 = u32(big_e - 2) / 32u;
    }
    // P = M * (W[j0] .. W[j0+4]) as 6 words, least significant first.
    var p: array<u32, 6>;
    var carry = 0u;
    for (var k = 0u; k < 5u; k = k + 1u) {
        let prod = nm_mul32w(n.m, nm_twobypi_word(j0 + 4u - k));
        let lo = prod.x + carry;
        p[k] = lo;
        carry = prod.y + select(0u, 1u, lo < prod.x);
    }
    p[5] = carry;
    // The binary point of P lies at bit b = 32 (j0 + 5) - E.
    let b = i32(32u * (j0 + 5u)) - big_e;
    // 192 bits shifted so that the binary point sits at bit 128 of a 6-word value: integer bits from word 4 on.
    let sh = 128 - b;
    var w: array<u32, 6>;
    for (var k = 0i; k < 6i; k = k + 1i) {
        // word k of P << sh (sh may be negative: a right shift)
        let bitpos = 32 * k - sh;
        let wi = bitpos >> 5u;
        let off = u32(bitpos & 31);
        var lo = 0u;
        var hi = 0u;
        if (wi >= 0 && wi < 6) {
            lo = p[wi];
        }
        if (wi + 1 >= 0 && wi + 1 < 6) {
            hi = p[wi + 1];
        }
        if (off == 0u) {
            w[k] = lo;
        } else {
            w[k] = (lo >> off) | (hi << (32u - off));
        }
    }
    var q = w[4] & 3u;
    // Fraction in w[3] w[2] w[1] w[0] (128 bits); at least one half: take r - 1.
    var f = vec4<u32>(w[0], w[1], w[2], w[3]);
    var neg = false;
    if ((w[3] >> 31u) != 0u) {
        q = (q + 1u) & 3u;
        neg = true;
        // 2^128 - f
        let nx = ~f.x + 1u;
        let c0 = select(0u, 1u, nx == 0u);
        let ny = ~f.y + c0;
        let c1 = select(0u, 1u, ny == 0u && c0 == 1u);
        let nz = ~f.z + c1;
        let c2 = select(0u, 1u, nz == 0u && c1 == 1u);
        let nw = ~f.w + c2;
        f = vec4<u32>(nx, ny, nz, nw);
    }
    // Normalize the 128-bit fraction to 64 bits (r = f * 2^-128). A zero fraction would need x to be an exact
    // multiple of pi/2, which no non-zero binary32 value is; it is still answered rather than misread.
    if ((f.x | f.y | f.z | f.w) == 0u) {
        return nm_xf_trig(q, nm_xf(vec2<u32>(0u, 0u), 0, 0u));
    }
    var lz = 0u;
    if (f.w != 0u) {
        lz = countLeadingZeros(f.w);
    } else if (f.z != 0u) {
        lz = 32u + countLeadingZeros(f.z);
    } else if (f.y != 0u) {
        lz = 64u + countLeadingZeros(f.y);
    } else {
        lz = 96u + countLeadingZeros(f.x);
    }
    // The top 64 significant bits of f.
    var top = vec2<u32>(0u, 0u);
    let words = array<u32, 4>(f.x, f.y, f.z, f.w);
    let start = 127u - lz;
    for (var t = 0u; t < 64u; t = t + 1u) {
        let bitp = i32(start) - i32(t);
        var bit = 0u;
        if (bitp >= 0) {
            bit = (words[u32(bitp) >> 5u] >> (u32(bitp) & 31u)) & 1u;
        }
        top = nm_shl64(top, 1u);
        top.x = top.x | bit;
    }
    let r = nm_xf(top, i32(start) - 128, select(0u, 1u, neg) ^ s);
    // An odd multiple of pi/2 flips with the sign of x: x = -(q' + r') pi/2 gives q = -q'.
    if (s != 0u) {
        q = (4u - q) & 3u;
    }
    return nm_xf_trig(q, nm_xf_mul(r, nm_xf_pi_2));
}
