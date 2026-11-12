// MSL integer functions (MSL Table 6.2) on the integer carriers: an n-bit value (n = 8, 16 or 32) is a u32 holding
// its bits in the low n bits with the high bits zero; a 64-bit value is vec2<u32>(low word, high word). Signedness
// belongs to the operation, as in AIR. Every result is returned in the same canonical form.

fn nm_mask(n: u32) -> u32 {
    return select((1u << n) - 1u, 0xffffffffu, n >= 32u);
}

// The low n bits of x, sign-extended.
fn nm_sext(x: u32, n: u32) -> i32 {
    let s = 32u - n;
    return bitcast<i32>(x << s) >> s;
}

fn nm_int_abs_s(x: u32, n: u32) -> u32 {
    return u32(abs(nm_sext(x, n))) & nm_mask(n);
}

// |x - y| without modulo overflow, as the unsigned type.
fn nm_int_absdiff_s(x: u32, y: u32, n: u32) -> u32 {
    let a = nm_sext(x, n);
    let b = nm_sext(y, n);
    return select(u32(b) - u32(a), u32(a) - u32(b), a > b) & nm_mask(n);
}

fn nm_int_absdiff_u(x: u32, y: u32, n: u32) -> u32 {
    return select(y - x, x - y, x > y) & nm_mask(n);
}

fn nm_int_add_sat_s(x: u32, y: u32, n: u32) -> u32 {
    let a = nm_sext(x, n);
    let b = nm_sext(y, n);
    if (n < 32u) {
        let lim = 1 << (n - 1u);
        return u32(clamp(a + b, -lim, lim - 1)) & nm_mask(n);
    }
    let s = a + b;
    if ((a < 0) == (b < 0) && (s < 0) != (a < 0)) {
        return select(0x7fffffffu, 0x80000000u, a < 0);
    }
    return u32(s);
}

fn nm_int_add_sat_u(x: u32, y: u32, n: u32) -> u32 {
    let s = x + y;
    if (n < 32u) {
        return min(s, nm_mask(n));
    }
    return select(s, 0xffffffffu, s < x);
}

fn nm_int_sub_sat_s(x: u32, y: u32, n: u32) -> u32 {
    let a = nm_sext(x, n);
    let b = nm_sext(y, n);
    if (n < 32u) {
        let lim = 1 << (n - 1u);
        return u32(clamp(a - b, -lim, lim - 1)) & nm_mask(n);
    }
    let s = a - b;
    if ((a < 0) != (b < 0) && (s < 0) != (a < 0)) {
        return select(0x7fffffffu, 0x80000000u, a < 0);
    }
    return u32(s);
}

fn nm_int_sub_sat_u(x: u32, y: u32, n: u32) -> u32 {
    return select(x - y, 0u, y > x);
}

// (x + y) >> 1 and (x + y + 1) >> 1 without modulo overflow.
fn nm_int_hadd_s(x: u32, y: u32, n: u32) -> u32 {
    let a = nm_sext(x, n);
    let b = nm_sext(y, n);
    return u32((a >> 1u) + (b >> 1u) + (a & b & 1)) & nm_mask(n);
}

fn nm_int_hadd_u(x: u32, y: u32, n: u32) -> u32 {
    return (x >> 1u) + (y >> 1u) + (x & y & 1u);
}

fn nm_int_rhadd_s(x: u32, y: u32, n: u32) -> u32 {
    let a = nm_sext(x, n);
    let b = nm_sext(y, n);
    return u32((a >> 1u) + (b >> 1u) + ((a | b) & 1)) & nm_mask(n);
}

fn nm_int_rhadd_u(x: u32, y: u32, n: u32) -> u32 {
    return (x >> 1u) + (y >> 1u) + ((x | y) & 1u);
}

// The high half of the 2n-bit product.
fn nm_int_mul_hi_s(x: u32, y: u32, n: u32) -> u32 {
    let a = nm_sext(x, n);
    let b = nm_sext(y, n);
    if (n < 32u) {
        return u32((a * b) >> n) & nm_mask(n);
    }
    let p = nm_mul32w(x, y).y;
    return p - select(0u, y, a < 0) - select(0u, x, b < 0);
}

fn nm_int_mul_hi_u(x: u32, y: u32, n: u32) -> u32 {
    if (n < 32u) {
        return (x * y) >> n;
    }
    return nm_mul32w(x, y).y;
}

// a * b + c, saturated.
fn nm_int_mad_sat_s(a: u32, b: u32, c: u32, n: u32) -> u32 {
    let x = nm_sext(a, n);
    let y = nm_sext(b, n);
    let z = nm_sext(c, n);
    if (n < 32u) {
        let lim = 1 << (n - 1u);
        return u32(clamp(x * y + z, -lim, lim - 1)) & nm_mask(n);
    }
    let p = nm_add64(nm_smul64(a, b), nm_sext64(c));
    return nm_sat64_to_i32(p);
}

fn nm_int_mad_sat_u(a: u32, b: u32, c: u32, n: u32) -> u32 {
    if (n < 32u) {
        return min(a * b + c, nm_mask(n));
    }
    let p = nm_add64(nm_mul32w(a, b), vec2<u32>(c, 0u));
    return select(p.x, 0xffffffffu, p.y != 0u);
}

fn nm_int_max_s(x: u32, y: u32, n: u32) -> u32 {
    return select(x, y, nm_sext(x, n) < nm_sext(y, n));
}

fn nm_int_min_s(x: u32, y: u32, n: u32) -> u32 {
    return select(x, y, nm_sext(y, n) < nm_sext(x, n));
}

fn nm_int_max_u(x: u32, y: u32, n: u32) -> u32 {
    return select(x, y, x < y);
}

fn nm_int_min_u(x: u32, y: u32, n: u32) -> u32 {
    return select(x, y, y < x);
}

fn nm_int_median3_s(x: u32, y: u32, z: u32, n: u32) -> u32 {
    return nm_int_max_s(nm_int_min_s(x, y, n), nm_int_min_s(nm_int_max_s(x, y, n), z, n), n);
}

fn nm_int_median3_u(x: u32, y: u32, z: u32, n: u32) -> u32 {
    return max(min(x, y), min(max(x, y), z));
}

fn nm_int_clz(x: u32, n: u32) -> u32 {
    return countLeadingZeros(x & nm_mask(n)) - (32u - n);
}

fn nm_int_ctz(x: u32, n: u32) -> u32 {
    return min(countTrailingZeros(x), n);
}

fn nm_int_popcount(x: u32, n: u32) -> u32 {
    return countOneBits(x & nm_mask(n));
}

fn nm_int_reverse_bits(x: u32, n: u32) -> u32 {
    return reverseBits(x) >> (32u - n);
}

// Rotate left by i modulo the width.
fn nm_int_rotate(v: u32, i: u32, n: u32) -> u32 {
    let r = i & (n - 1u);
    if (r == 0u) {
        return v;
    }
    return ((v << r) | (v >> (n - r))) & nm_mask(n);
}

// extract_bits and insert_bits: WGSL extractBits and insertBits clamp offset and count to the width, which is
// MSL's result wherever MSL defines one (offset + bits within the width).
fn nm_int_extract_bits_s(x: u32, offset: u32, bits: u32, n: u32) -> u32 {
    return u32(extractBits(nm_sext(x, n), offset, bits)) & nm_mask(n);
}

fn nm_int_extract_bits_u(x: u32, offset: u32, bits: u32, n: u32) -> u32 {
    return extractBits(x, offset, bits) & nm_mask(n);
}

fn nm_int_insert_bits(base: u32, ins: u32, offset: u32, bits: u32, n: u32) -> u32 {
    return insertBits(base, ins, offset, bits) & nm_mask(n);
}

// 64-bit operations.
fn nm_sext64(x: u32) -> vec2<u32> {
    return vec2<u32>(x, select(0u, 0xffffffffu, (x >> 31u) != 0u));
}

// The exact 64-bit product of two signed 32-bit values.
fn nm_smul64(x: u32, y: u32) -> vec2<u32> {
    let p = nm_mul32w(x, y);
    let hi = p.y - select(0u, y, (x >> 31u) != 0u) - select(0u, x, (y >> 31u) != 0u);
    return vec2<u32>(p.x, hi);
}

fn nm_sat64_to_i32(p: vec2<u32>) -> u32 {
    let neg = (p.y >> 31u) != 0u;
    // In range when the high word is the sign extension of the low word.
    if (p.y == select(0u, 0xffffffffu, (p.x >> 31u) != 0u)) {
        return p.x;
    }
    return select(0x7fffffffu, 0x80000000u, neg);
}

fn nm_i64_slt(a: vec2<u32>, b: vec2<u32>) -> bool {
    let ah = bitcast<i32>(a.y);
    let bh = bitcast<i32>(b.y);
    return ah < bh || (ah == bh && a.x < b.x);
}

fn nm_i64_neg(a: vec2<u32>) -> vec2<u32> {
    return nm_sub64(vec2<u32>(0u, 0u), a);
}

fn nm_i64_isneg(a: vec2<u32>) -> bool {
    return (a.y >> 31u) != 0u;
}

// The low 64 bits of a 64-bit product (signed and unsigned agree).
fn nm_i64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = nm_mul32w(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}

fn nm_i64_abs_s(a: vec2<u32>) -> vec2<u32> {
    return select(a, nm_i64_neg(a), nm_i64_isneg(a));
}

fn nm_i64_max_s(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return select(a, b, nm_i64_slt(a, b));
}

fn nm_i64_min_s(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return select(a, b, nm_i64_slt(b, a));
}

fn nm_i64_max_u(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return select(a, b, nm_lt64(a, b));
}

fn nm_i64_min_u(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return select(a, b, nm_lt64(b, a));
}

fn nm_i64_median3_s(x: vec2<u32>, y: vec2<u32>, z: vec2<u32>) -> vec2<u32> {
    return nm_i64_max_s(nm_i64_min_s(x, y), nm_i64_min_s(nm_i64_max_s(x, y), z));
}

fn nm_i64_median3_u(x: vec2<u32>, y: vec2<u32>, z: vec2<u32>) -> vec2<u32> {
    return nm_i64_max_u(nm_i64_min_u(x, y), nm_i64_min_u(nm_i64_max_u(x, y), z));
}

fn nm_i64_absdiff_s(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return select(nm_sub64(b, a), nm_sub64(a, b), nm_i64_slt(b, a));
}

fn nm_i64_absdiff_u(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return select(nm_sub64(b, a), nm_sub64(a, b), nm_lt64(b, a));
}

fn nm_i64_add_sat_s(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let s = nm_add64(a, b);
    if (nm_i64_isneg(a) == nm_i64_isneg(b) && nm_i64_isneg(s) != nm_i64_isneg(a)) {
        return select(vec2<u32>(0xffffffffu, 0x7fffffffu), vec2<u32>(0u, 0x80000000u), nm_i64_isneg(a));
    }
    return s;
}

fn nm_i64_add_sat_u(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let s = nm_add64(a, b);
    return select(s, vec2<u32>(0xffffffffu, 0xffffffffu), nm_lt64(s, a));
}

fn nm_i64_sub_sat_s(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let s = nm_sub64(a, b);
    if (nm_i64_isneg(a) != nm_i64_isneg(b) && nm_i64_isneg(s) != nm_i64_isneg(a)) {
        return select(vec2<u32>(0xffffffffu, 0x7fffffffu), vec2<u32>(0u, 0x80000000u), nm_i64_isneg(a));
    }
    return s;
}

fn nm_i64_sub_sat_u(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return select(nm_sub64(a, b), vec2<u32>(0u, 0u), nm_lt64(a, b));
}

fn nm_i64_sar1(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>((a.x >> 1u) | (a.y << 31u), u32(bitcast<i32>(a.y) >> 1u));
}

fn nm_i64_hadd_s(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return nm_add64(nm_add64(nm_i64_sar1(a), nm_i64_sar1(b)), vec2<u32>(a.x & b.x & 1u, 0u));
}

fn nm_i64_hadd_u(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return nm_add64(nm_add64(nm_shr64(a, 1u), nm_shr64(b, 1u)), vec2<u32>(a.x & b.x & 1u, 0u));
}

fn nm_i64_rhadd_s(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return nm_add64(nm_add64(nm_i64_sar1(a), nm_i64_sar1(b)), vec2<u32>((a.x | b.x) & 1u, 0u));
}

fn nm_i64_rhadd_u(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return nm_add64(nm_add64(nm_shr64(a, 1u), nm_shr64(b, 1u)), vec2<u32>((a.x | b.x) & 1u, 0u));
}

// The high 64 bits of the 128-bit product.
fn nm_i64_mul_hi_u(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = nm_mul64w(a, b);
    return vec2<u32>(p.z, p.w);
}

fn nm_i64_mul_hi_s(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    var h = nm_i64_mul_hi_u(a, b);
    if (nm_i64_isneg(a)) {
        h = nm_sub64(h, b);
    }
    if (nm_i64_isneg(b)) {
        h = nm_sub64(h, a);
    }
    return h;
}

fn nm_i64_mad_sat_u(a: vec2<u32>, b: vec2<u32>, c: vec2<u32>) -> vec2<u32> {
    let p = nm_mul64w(a, b);
    let lo = nm_add64(vec2<u32>(p.x, p.y), c);
    let carry = nm_lt64(lo, c);
    if ((p.z | p.w) != 0u || carry) {
        return vec2<u32>(0xffffffffu, 0xffffffffu);
    }
    return lo;
}

fn nm_i64_mad_sat_s(a: vec2<u32>, b: vec2<u32>, c: vec2<u32>) -> vec2<u32> {
    // The 128-bit signed product, then a 128-bit sum with c sign-extended, then saturation to 64 bits.
    let p = nm_mul64w(a, b);
    var hi = nm_i64_mul_hi_s(a, b);
    let lo = nm_add64(vec2<u32>(p.x, p.y), c);
    let carry = select(0u, 1u, nm_lt64(lo, c));
    hi = nm_add64(nm_add64(hi, vec2<u32>(carry, 0u)), select(vec2<u32>(0u, 0u), vec2<u32>(0xffffffffu, 0xffffffffu), nm_i64_isneg(c)));
    let ext = select(vec2<u32>(0u, 0u), vec2<u32>(0xffffffffu, 0xffffffffu), nm_i64_isneg(lo));
    if (hi.x == ext.x && hi.y == ext.y) {
        return lo;
    }
    return select(vec2<u32>(0xffffffffu, 0x7fffffffu), vec2<u32>(0u, 0x80000000u), nm_i64_isneg(hi));
}

fn nm_i64_clz(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(nm_clz64(a), 0u);
}

fn nm_i64_ctz(a: vec2<u32>) -> vec2<u32> {
    if (a.x != 0u) {
        return vec2<u32>(countTrailingZeros(a.x), 0u);
    }
    return vec2<u32>(32u + countTrailingZeros(a.y), 0u);
}

fn nm_i64_popcount(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(countOneBits(a.x) + countOneBits(a.y), 0u);
}

fn nm_i64_reverse_bits(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(reverseBits(a.y), reverseBits(a.x));
}

fn nm_i64_rotate(v: vec2<u32>, i: vec2<u32>) -> vec2<u32> {
    let r = i.x & 63u;
    if (r == 0u) {
        return v;
    }
    let l = nm_shl64(v, r);
    let rr = nm_shr64(v, 64u - r);
    return vec2<u32>(l.x | rr.x, l.y | rr.y);
}

// extract_bits and insert_bits on 64 bits; offset and bits are 32-bit and clamped to the width as WGSL's are.
fn nm_i64_extract_bits(x: vec2<u32>, offset: u32, bits: u32, signed: bool) -> vec2<u32> {
    let o = min(offset, 64u);
    let c = min(bits, 64u - o);
    if (c == 0u) {
        return vec2<u32>(0u, 0u);
    }
    let t = nm_shl64(nm_shr64(x, o), 64u - c);
    if (!signed) {
        return nm_shr64(t, 64u - c);
    }
    // Arithmetic shift right by 64 - c.
    let sh = 64u - c;
    let neg = (t.y >> 31u) != 0u;
    let r = nm_shr64(t, sh);
    if (!neg || sh == 0u) {
        return r;
    }
    let fill = nm_shl64(vec2<u32>(0xffffffffu, 0xffffffffu), 64u - sh);
    return vec2<u32>(r.x | fill.x, r.y | fill.y);
}

fn nm_i64_insert_bits(base: vec2<u32>, ins: vec2<u32>, offset: u32, bits: u32) -> vec2<u32> {
    let o = min(offset, 64u);
    let c = min(bits, 64u - o);
    if (c == 0u) {
        return base;
    }
    let ones = nm_shr64(vec2<u32>(0xffffffffu, 0xffffffffu), 64u - c);
    let m = nm_shl64(ones, o);
    let v = nm_shl64(vec2<u32>(ins.x & ones.x, ins.y & ones.y), o);
    return vec2<u32>((base.x & ~m.x) | v.x, (base.y & ~m.y) | v.y);
}

// LLVM funnel shifts (LangRef llvm.fshl, llvm.fshr): the concatenation a:b shifted by s modulo the width, keeping the
// high (fshl) or low (fshr) half.
fn nm_int_fshl(a: u32, b: u32, s: u32, n: u32) -> u32 {
    let k = s % n;
    if (k == 0u) {
        return a;
    }
    return ((a << k) | (b >> (n - k))) & nm_mask(n);
}

fn nm_int_fshr(a: u32, b: u32, s: u32, n: u32) -> u32 {
    let k = s % n;
    if (k == 0u) {
        return b;
    }
    return ((b >> k) | (a << (n - k))) & nm_mask(n);
}

fn nm_i64_fshl(a: vec2<u32>, b: vec2<u32>, s: vec2<u32>) -> vec2<u32> {
    let k = s.x & 63u;
    if (k == 0u) {
        return a;
    }
    let l = nm_shl64(a, k);
    let r = nm_shr64(b, 64u - k);
    return vec2<u32>(l.x | r.x, l.y | r.y);
}

fn nm_i64_fshr(a: vec2<u32>, b: vec2<u32>, s: vec2<u32>) -> vec2<u32> {
    let k = s.x & 63u;
    if (k == 0u) {
        return b;
    }
    let r = nm_shr64(b, k);
    let l = nm_shl64(a, 64u - k);
    return vec2<u32>(l.x | r.x, l.y | r.y);
}

// Byte swap (llvm.bswap) of 16-, 32- and 64-bit values.
fn nm_int_bswap(x: u32, n: u32) -> u32 {
    let w = (x << 24u) | ((x & 0xff00u) << 8u) | ((x >> 8u) & 0xff00u) | (x >> 24u);
    return w >> (32u - n);
}

fn nm_i64_bswap(x: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(nm_int_bswap(x.y, 32u), nm_int_bswap(x.x, 32u));
}
