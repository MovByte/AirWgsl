// IEEE 754 binary32 arithmetic on u32 bit patterns.
//
// Values never pass through a WGSL f32: an implementation may assume no NaN or infinity occurs at run time and may
// flush subnormals (WGSL 15.7.2), so every operation here works on the bits. Results are exactly those of IEEE 754
// round-to-nearest-even with gradual underflow, which MSL permits (MSL 8.1, 8.2).
//
// NaN results follow IEEE 754-2019 6.2.3: an operation with NaN operands returns one of them, quieted, with its
// payload; when several operands are NaN the choice follows the order of FPProcessNaNs in the Arm Architecture
// Reference Manual (the first signaling NaN, otherwise the first quiet NaN). An invalid operation returns the default
// NaN 0x7fc00000. MSL does not support signaling NaNs (MSL 8.1); they are quieted like any other NaN operand.
//
// Rounding convention: an unrounded finite result is (sign, exponent e, significand m) with value m * 2^(e - 157)
// and m < 2^31. A normalized m has its leading one at bit 30, so e is then the biased exponent and bits 6..0 of m are
// the round and sticky bits.

const nm_f32_dnan: u32 = 0x7fc00000u;

fn nm_f32_isnan(a: u32) -> bool {
    return (a & 0x7fffffffu) > 0x7f800000u;
}

fn nm_f32_issnan(a: u32) -> bool {
    return nm_f32_isnan(a) && (a & 0x00400000u) == 0u;
}

// The NaN result of a two-operand operation where a or b is NaN.
fn nm_f32_nan2(a: u32, b: u32) -> u32 {
    if (nm_f32_issnan(a)) {
        return a | 0x00400000u;
    }
    if (nm_f32_issnan(b)) {
        return b | 0x00400000u;
    }
    if (nm_f32_isnan(a)) {
        return a;
    }
    return b;
}

// The NaN result of a three-operand operation where a, b or c is NaN.
fn nm_f32_nan3(a: u32, b: u32, c: u32) -> u32 {
    if (nm_f32_issnan(a)) {
        return a | 0x00400000u;
    }
    if (nm_f32_issnan(b)) {
        return b | 0x00400000u;
    }
    if (nm_f32_issnan(c)) {
        return c | 0x00400000u;
    }
    if (nm_f32_isnan(a)) {
        return a;
    }
    if (nm_f32_isnan(b)) {
        return b;
    }
    return c;
}

// An unrounded result (k == 0: sign s, exponent e, significand m) or a finished binary32 result (k == 1: bits m).
struct nm_ur {
    s: u32,
    e: i32,
    m: u32,
    k: u32,
}

fn nm_ur_done(bits: u32) -> nm_ur {
    return nm_ur(0u, 0, bits, 1u);
}

// Round to nearest even and pack. Overflow gives infinity; results below the normal range round at the subnormal
// precision.
fn nm_f32_rpack(s: u32, e: i32, m: u32) -> u32 {
    if (e >= 0xff) {
        return (s << 31u) | 0x7f800000u;
    }
    var exp = e;
    var sig = m;
    if (exp <= 0) {
        sig = nm_shr32_jam(sig, u32(1 - exp));
        exp = 1;
    }
    let rb = sig & 0x7fu;
    sig = (sig + 0x40u) >> 7u;
    if (rb == 0x40u) {
        sig = sig & ~1u;
    }
    // The leading one of sig, at bit 23, carries into the exponent field; so does a rounding carry.
    return (s << 31u) | ((u32(exp - 1) << 23u) + sig);
}

// Normalize a non-zero significand (m < 2^31) and round.
fn nm_f32_nrpack(s: u32, e: i32, m: u32) -> u32 {
    let lz = countLeadingZeros(m) - 1u;
    return nm_f32_rpack(s, e - i32(lz), m << lz);
}

fn nm_f32_finish(r: nm_ur) -> u32 {
    if (r.k != 0u) {
        return r.m;
    }
    return nm_f32_nrpack(r.s, r.e, r.m);
}

// Significand with the leading one at bit 23 and its biased exponent (below 1 for subnormals) of a finite non-zero
// magnitude.
struct nm_f32n {
    m: u32,
    e: i32,
}

fn nm_f32_norm(a: u32) -> nm_f32n {
    let e = i32(a >> 23u);
    let f = a & 0x7fffffu;
    if (e != 0) {
        return nm_f32n(f | 0x800000u, e);
    }
    let lz = countLeadingZeros(f) - 8u;
    return nm_f32n(f << lz, 1 - i32(lz));
}

// Sum of two finite values.
fn nm_f32_add_core(a: u32, b: u32) -> nm_ur {
    let sa = a >> 31u;
    let sb = b >> 31u;
    let absa = a & 0x7fffffffu;
    let absb = b & 0x7fffffffu;
    var ea = i32(absa >> 23u);
    var eb = i32(absb >> 23u);
    var ma = absa & 0x7fffffu;
    var mb = absb & 0x7fffffu;
    if (ea == 0) {
        ea = 1;
    } else {
        ma = ma | 0x800000u;
    }
    if (eb == 0) {
        eb = 1;
    } else {
        mb = mb | 0x800000u;
    }
    var mx = ma << 6u;
    var mn = mb << 6u;
    var ex = ea;
    var en = eb;
    var s = sa;
    if (absa < absb) {
        mx = mb << 6u;
        mn = ma << 6u;
        ex = eb;
        en = ea;
        s = sb;
    }
    mn = nm_shr32_jam(mn, u32(ex - en));
    var m: u32;
    if (sa == sb) {
        m = mx + mn;
        if (m == 0u) {
            return nm_ur_done(s << 31u);
        }
    } else {
        m = mx - mn;
        if (m == 0u) {
            return nm_ur_done(0u);
        }
    }
    return nm_ur(s, ex + 1, m, 0u);
}

fn nm_f32_add_ur(a: u32, b: u32) -> nm_ur {
    if (nm_f32_isnan(a) || nm_f32_isnan(b)) {
        return nm_ur_done(nm_f32_nan2(a, b));
    }
    let ia = (a & 0x7fffffffu) == 0x7f800000u;
    let ib = (b & 0x7fffffffu) == 0x7f800000u;
    if (ia) {
        if (ib && a != b) {
            return nm_ur_done(nm_f32_dnan);
        }
        return nm_ur_done(a);
    }
    if (ib) {
        return nm_ur_done(b);
    }
    return nm_f32_add_core(a, b);
}

fn nm_f32_sub_ur(a: u32, b: u32) -> nm_ur {
    if (nm_f32_isnan(a) || nm_f32_isnan(b)) {
        return nm_ur_done(nm_f32_nan2(a, b));
    }
    return nm_f32_add_ur(a, b ^ 0x80000000u);
}

fn nm_f32_mul_ur(a: u32, b: u32) -> nm_ur {
    if (nm_f32_isnan(a) || nm_f32_isnan(b)) {
        return nm_ur_done(nm_f32_nan2(a, b));
    }
    let s = (a ^ b) & 0x80000000u;
    let aa = a & 0x7fffffffu;
    let ab = b & 0x7fffffffu;
    if (aa == 0x7f800000u || ab == 0x7f800000u) {
        if (aa == 0u || ab == 0u) {
            return nm_ur_done(nm_f32_dnan);
        }
        return nm_ur_done(s | 0x7f800000u);
    }
    if (aa == 0u || ab == 0u) {
        return nm_ur_done(s);
    }
    let na = nm_f32_norm(aa);
    let nb = nm_f32_norm(ab);
    // The exact 48-bit product, reduced to 31 bits with a sticky bit.
    let p = nm_mul32w(na.m, nb.m);
    return nm_ur(s >> 31u, na.e + nb.e - 126, nm_shr64_jam(p, 17u).x, 0u);
}

fn nm_f32_div_ur(a: u32, b: u32) -> nm_ur {
    if (nm_f32_isnan(a) || nm_f32_isnan(b)) {
        return nm_ur_done(nm_f32_nan2(a, b));
    }
    let s = (a ^ b) & 0x80000000u;
    let aa = a & 0x7fffffffu;
    let ab = b & 0x7fffffffu;
    if (aa == 0x7f800000u) {
        if (ab == 0x7f800000u) {
            return nm_ur_done(nm_f32_dnan);
        }
        return nm_ur_done(s | 0x7f800000u);
    }
    if (ab == 0x7f800000u) {
        return nm_ur_done(s);
    }
    if (ab == 0u) {
        if (aa == 0u) {
            return nm_ur_done(nm_f32_dnan);
        }
        return nm_ur_done(s | 0x7f800000u);
    }
    if (aa == 0u) {
        return nm_ur_done(s);
    }
    let na = nm_f32_norm(aa);
    let nb = nm_f32_norm(ab);
    var r = na.m;
    var e = na.e - nb.e + 127;
    if (r < nb.m) {
        r = r << 1u;
        e = e - 1;
    }
    // Restoring division: 31 quotient bits, the first of them 1, and a sticky bit for the remainder.
    var q = 0u;
    for (var i = 0u; i < 31u; i = i + 1u) {
        q = q << 1u;
        if (r >= nb.m) {
            r = r - nb.m;
            q = q | 1u;
        }
        r = r << 1u;
    }
    if (r != 0u) {
        q = q | 1u;
    }
    return nm_ur(s >> 31u, e, q, 0u);
}

fn nm_f32_sqrt_ur(a: u32) -> nm_ur {
    if (nm_f32_isnan(a)) {
        return nm_ur_done(a | 0x00400000u);
    }
    if ((a & 0x7fffffffu) == 0u) {
        return nm_ur_done(a);
    }
    if ((a >> 31u) != 0u) {
        return nm_ur_done(nm_f32_dnan);
    }
    if (a == 0x7f800000u) {
        return nm_ur_done(a);
    }
    let n = nm_f32_norm(a);
    // a = m * 2^(e - 150); make the exponent even so that a = mp * 2^(2k) with mp in [2^23, 2^25).
    var mp = n.m;
    var t = n.e - 150;
    if ((t & 1) != 0) {
        mp = mp << 1u;
        t = t - 1;
    }
    let k = t / 2;
    // Digit-by-digit square root of mp * 2^26: 26 result bits, two radicand bits per step.
    var res = 0u;
    var rem = 0u;
    for (var j = 25i; j >= 0i; j = j - 1i) {
        var pair = 0u;
        if (j >= 13i) {
            pair = (mp >> u32(2i * j - 26i)) & 3u;
        }
        rem = (rem << 2u) | pair;
        let trial = (res << 2u) | 1u;
        res = res << 1u;
        if (rem >= trial) {
            rem = rem - trial;
            res = res | 1u;
        }
    }
    return nm_ur(0u, k + 139, (res << 5u) | select(0u, 1u, rem != 0u), 0u);
}

// a * b + c with a single rounding (MSL Table 6.4 fma: "Edge case behavior is per the IEEE 754-2008 standard").
fn nm_f32_fma_ur(a: u32, b: u32, c: u32) -> nm_ur {
    let na_ = nm_f32_isnan(a);
    let nb_ = nm_f32_isnan(b);
    let nc_ = nm_f32_isnan(c);
    let aa = a & 0x7fffffffu;
    let ab = b & 0x7fffffffu;
    let ac = c & 0x7fffffffu;
    let invalid_product = (aa == 0x7f800000u && ab == 0u) || (aa == 0u && ab == 0x7f800000u);
    if (na_ || nb_ || nc_) {
        // FPMulAdd: NaNs in the order addend, multiplicand, multiplier; a quiet NaN addend with an invalid product
        // gives the default NaN.
        if (nc_ && !nm_f32_issnan(c) && invalid_product) {
            return nm_ur_done(nm_f32_dnan);
        }
        return nm_ur_done(nm_f32_nan3(c, a, b));
    }
    if (invalid_product) {
        return nm_ur_done(nm_f32_dnan);
    }
    let sp = (a ^ b) & 0x80000000u;
    let pinf = aa == 0x7f800000u || ab == 0x7f800000u;
    if (pinf) {
        if (ac == 0x7f800000u && (c & 0x80000000u) != sp) {
            return nm_ur_done(nm_f32_dnan);
        }
        return nm_ur_done(sp | 0x7f800000u);
    }
    if (ac == 0x7f800000u) {
        return nm_ur_done(c);
    }
    if (aa == 0u || ab == 0u) {
        // An exact zero product: (+-0) + c.
        if (ac == 0u) {
            if (sp == (c & 0x80000000u)) {
                return nm_ur_done(c);
            }
            return nm_ur_done(0u);
        }
        return nm_ur_done(c);
    }
    let na = nm_f32_norm(aa);
    let nb = nm_f32_norm(ab);
    // Product: 48 bits placed at bits 62..15 of 64 (value pp * 2^xp); addend at bits 62..39 (value pc * 2^xc).
    let pp = nm_shl64(nm_mul32w(na.m, nb.m), 15u);
    let xp = na.e + nb.e - 315;
    if (ac == 0u) {
        let lz = nm_clz64(pp);
        return nm_ur(sp >> 31u, xp + 190 - i32(lz), nm_shr64_jam(nm_shl64(pp, lz), 33u).x, 0u);
    }
    let nc = nm_f32_norm(ac);
    let sc = c & 0x80000000u;
    let pc = nm_shl64(vec2<u32>(nc.m, 0u), 39u);
    let xc = nc.e - 189;
    var big = pp;
    var small = pc;
    var x = xp;
    var d = xp - xc;
    var sbig = sp;
    if (xc > xp || (xc == xp && nm_lt64(pp, pc))) {
        big = pc;
        small = pp;
        x = xc;
        d = xc - xp;
        sbig = sc;
    }
    small = nm_shr64_jam(small, u32(d));
    var sum: vec2<u32>;
    if (sp == sc) {
        sum = nm_add64(big, small);
    } else {
        if (nm_lt64(big, small)) {
            // Only possible when the exponents differ by at most one; the shifted operand has at least 15 trailing
            // zero bits, so the alignment lost nothing.
            sum = nm_sub64(small, big);
            sbig = sbig ^ 0x80000000u;
        } else {
            sum = nm_sub64(big, small);
        }
        if (nm_is0_64(sum)) {
            return nm_ur_done(0u);
        }
    }
    let lz = nm_clz64(sum);
    var m: u32;
    if (lz <= 33u) {
        m = nm_shr64_jam(sum, 33u - lz).x;
    } else {
        m = nm_shl64(sum, lz - 33u).x;
    }
    return nm_ur(sbig >> 31u, x + 190 - i32(lz), m, 0u);
}

// 1 / sqrt(a), correctly rounded (MSL Table 8.1 rsqrt). IEEE 754-2019 9.2 rSqrt: rSqrt(+-0) is +-infinity, rSqrt of
// a negative non-zero value is invalid.
fn nm_f32_rsqrt_ur(a: u32) -> nm_ur {
    if (nm_f32_isnan(a)) {
        return nm_ur_done(a | 0x00400000u);
    }
    if ((a & 0x7fffffffu) == 0u) {
        return nm_ur_done(a | 0x7f800000u);
    }
    if ((a >> 31u) != 0u) {
        return nm_ur_done(nm_f32_dnan);
    }
    if (a == 0x7f800000u) {
        return nm_ur_done(0u);
    }
    let n = nm_f32_norm(a);
    // a = mp * 2^(2k), mp in [2^23, 2^25). Y = floor(2^38 / sqrt(mp)) is the largest Y with Y^2 * mp <= 2^76; it is
    // found bit by bit, keeping D = 2^76 - Y^2 * mp in three words.
    var mp = n.m;
    var t = n.e - 150;
    if ((t & 1) != 0) {
        mp = mp << 1u;
        t = t - 1;
    }
    let k = t / 2;
    var y = 0u;
    var d = vec3<u32>(0u, 0u, 0x1000u);
    for (var bit = 26i; bit >= 0i; bit = bit - 1i) {
        let b = u32(bit);
        // (Y + 2^b)^2 * mp - Y^2 * mp = ((2Y + 2^b) * mp) << b
        let inc = nm_shl64to96(nm_mul32w((y << 1u) + (1u << b), mp), b);
        if (nm_le96(inc, d)) {
            d = nm_sub96(d, inc);
            y = y | (1u << b);
        }
    }
    let sticky = select(0u, 1u, (d.x | d.y | d.z) != 0u);
    return nm_ur(0u, 115 - k, (y << 4u) | sticky, 0u);
}

fn nm_f32_add(a: u32, b: u32) -> u32 {
    return nm_f32_finish(nm_f32_add_ur(a, b));
}

fn nm_f32_sub(a: u32, b: u32) -> u32 {
    return nm_f32_finish(nm_f32_sub_ur(a, b));
}

fn nm_f32_mul(a: u32, b: u32) -> u32 {
    return nm_f32_finish(nm_f32_mul_ur(a, b));
}

fn nm_f32_div(a: u32, b: u32) -> u32 {
    return nm_f32_finish(nm_f32_div_ur(a, b));
}

fn nm_f32_sqrt(a: u32) -> u32 {
    return nm_f32_finish(nm_f32_sqrt_ur(a));
}

fn nm_f32_fma(a: u32, b: u32, c: u32) -> u32 {
    return nm_f32_finish(nm_f32_fma_ur(a, b, c));
}

fn nm_f32_rsqrt(a: u32) -> u32 {
    return nm_f32_finish(nm_f32_rsqrt_ur(a));
}

fn nm_f32_neg(a: u32) -> u32 {
    return a ^ 0x80000000u;
}

fn nm_f32_abs(a: u32) -> u32 {
    return a & 0x7fffffffu;
}

fn nm_f32_copysign(a: u32, b: u32) -> u32 {
    return (a & 0x7fffffffu) | (b & 0x80000000u);
}

// Comparisons. A key maps the bits to an unsigned value in the order of the represented numbers (-0 below +0, so
// zeros are compared separately).
fn nm_f32_key(a: u32) -> u32 {
    if ((a >> 31u) != 0u) {
        return ~a;
    }
    return a | 0x80000000u;
}

fn nm_f32_uno(a: u32, b: u32) -> bool {
    return nm_f32_isnan(a) || nm_f32_isnan(b);
}

fn nm_f32_oeq(a: u32, b: u32) -> bool {
    return !nm_f32_uno(a, b) && (a == b || ((a | b) & 0x7fffffffu) == 0u);
}

fn nm_f32_olt(a: u32, b: u32) -> bool {
    return !nm_f32_uno(a, b) && ((a | b) & 0x7fffffffu) != 0u && nm_f32_key(a) < nm_f32_key(b);
}

fn nm_f32_ole(a: u32, b: u32) -> bool {
    return nm_f32_olt(a, b) || nm_f32_oeq(a, b);
}

// MSL fmax: "Returns y if x < y, otherwise returns x. If one argument is a NaN, fmax() returns the other argument. If
// both arguments are NaNs, fmax() returns a NaN." (MSL Table 6.4.)
fn nm_f32_fmax(x: u32, y: u32) -> u32 {
    let nx = nm_f32_isnan(x);
    let ny = nm_f32_isnan(y);
    if (nx && ny) {
        return nm_f32_nan2(x, y);
    }
    if (nx) {
        return y;
    }
    if (ny) {
        return x;
    }
    return select(x, y, nm_f32_olt(x, y));
}

// MSL fmin: "Returns y if y < x, otherwise it returns x", with the NaN rules of fmax.
fn nm_f32_fmin(x: u32, y: u32) -> u32 {
    let nx = nm_f32_isnan(x);
    let ny = nm_f32_isnan(y);
    if (nx && ny) {
        return nm_f32_nan2(x, y);
    }
    if (nx) {
        return y;
    }
    if (ny) {
        return x;
    }
    return select(x, y, nm_f32_olt(y, x));
}

// Rounding to an integral value: a NaN operand is returned quieted, infinities and integral values unchanged.
fn nm_f32_trunc(a: u32) -> u32 {
    let e = (a >> 23u) & 0xffu;
    if (e == 0xffu) {
        return select(a, a | 0x00400000u, nm_f32_isnan(a));
    }
    if (e >= 150u) {
        return a;
    }
    if (e < 127u) {
        return a & 0x80000000u;
    }
    return a & ~((1u << (150u - e)) - 1u);
}

fn nm_f32_floor(a: u32) -> u32 {
    let e = (a >> 23u) & 0xffu;
    if (e == 0xffu) {
        return select(a, a | 0x00400000u, nm_f32_isnan(a));
    }
    if (e >= 150u) {
        return a;
    }
    let neg = (a >> 31u) != 0u;
    if (e < 127u) {
        if ((a & 0x7fffffffu) == 0u) {
            return a;
        }
        return select(0u, 0xbf800000u, neg);
    }
    let mask = (1u << (150u - e)) - 1u;
    if ((a & mask) == 0u) {
        return a;
    }
    // Moving a negative value's magnitude to the next integer may carry into the exponent, which is the right result.
    return select(a & ~mask, (a & ~mask) + mask + 1u, neg);
}

fn nm_f32_ceil(a: u32) -> u32 {
    let e = (a >> 23u) & 0xffu;
    if (e == 0xffu) {
        return select(a, a | 0x00400000u, nm_f32_isnan(a));
    }
    if (e >= 150u) {
        return a;
    }
    let neg = (a >> 31u) != 0u;
    if (e < 127u) {
        if ((a & 0x7fffffffu) == 0u) {
            return a;
        }
        return select(0x3f800000u, 0x80000000u, neg);
    }
    let mask = (1u << (150u - e)) - 1u;
    if ((a & mask) == 0u) {
        return a;
    }
    return select((a & ~mask) + mask + 1u, a & ~mask, neg);
}

// Round half to even (MSL rint).
fn nm_f32_rint(a: u32) -> u32 {
    let e = (a >> 23u) & 0xffu;
    if (e == 0xffu) {
        return select(a, a | 0x00400000u, nm_f32_isnan(a));
    }
    if (e >= 150u) {
        return a;
    }
    let s = a & 0x80000000u;
    if (e < 126u) {
        return s;
    }
    if (e == 126u) {
        // 0.5 <= |a| < 1: exactly one half rounds to the even zero.
        return select(s | 0x3f800000u, s, (a & 0x7fffffu) == 0u);
    }
    let n = 150u - e;
    let mask = (1u << n) - 1u;
    let half = 1u << (n - 1u);
    let r = a & mask;
    let base = a & ~mask;
    if (r > half || (r == half && ((base >> n) & 1u) != 0u)) {
        return base + mask + 1u;
    }
    return base;
}

// Round half away from zero (MSL round).
fn nm_f32_round(a: u32) -> u32 {
    let e = (a >> 23u) & 0xffu;
    if (e == 0xffu) {
        return select(a, a | 0x00400000u, nm_f32_isnan(a));
    }
    if (e >= 150u) {
        return a;
    }
    let s = a & 0x80000000u;
    if (e < 126u) {
        return s;
    }
    if (e == 126u) {
        return s | 0x3f800000u;
    }
    let n = 150u - e;
    let mask = (1u << n) - 1u;
    let base = a & ~mask;
    if ((a & mask) >= (1u << (n - 1u))) {
        return base + mask + 1u;
    }
    return base;
}

// x * 2^n, correctly rounded (MSL ldexp).
fn nm_f32_ldexp(a: u32, n: i32) -> u32 {
    let aa = a & 0x7fffffffu;
    if (aa >= 0x7f800000u) {
        return select(a, a | 0x00400000u, nm_f32_isnan(a));
    }
    if (aa == 0u) {
        return a;
    }
    // Scaling by more than 2^±400 is past every rounding boundary of binary32 either way.
    let k = clamp(n, -400, 400);
    let v = nm_f32_norm(aa);
    return nm_f32_rpack(a >> 31u, v.e + k, v.m << 7u);
}
