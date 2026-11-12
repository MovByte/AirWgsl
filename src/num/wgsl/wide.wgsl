// Multiword unsigned integer arithmetic on u32 words.
//
// A 64-bit value is vec2<u32>(low word, high word); a 96-bit value is vec3<u32>(low, middle, high) and a 128-bit
// value vec4<u32>(lowest .. highest). WGSL has no wider integer type, and shifts by the full word width are taken
// modulo 32, so every shift below guards its zero and full-width cases.

// The 64-bit product of two 32-bit words, from four 16-bit partial products.
fn nm_mul32w(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return vec2<u32>((p00 & 0xffffu) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}

fn nm_add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn nm_sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

fn nm_lt64(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

fn nm_is0_64(a: vec2<u32>) -> bool {
    return (a.x | a.y) == 0u;
}

// Leading zero bits of a 64-bit value (64 for zero).
fn nm_clz64(a: vec2<u32>) -> u32 {
    if (a.y != 0u) {
        return countLeadingZeros(a.y);
    }
    return 32u + countLeadingZeros(a.x);
}

// Left shift by n in [0, 63].
fn nm_shl64(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) {
        return a;
    }
    if (n < 32u) {
        return vec2<u32>(a.x << n, (a.y << n) | (a.x >> (32u - n)));
    }
    return vec2<u32>(0u, a.x << (n - 32u));
}

// Logical right shift by n in [0, 63].
fn nm_shr64(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) {
        return a;
    }
    if (n < 32u) {
        return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
    }
    return vec2<u32>(a.y >> (n - 32u), 0u);
}

// Right shift by any n, OR-ing every bit shifted out into bit 0 (a sticky bit for rounding).
fn nm_shr32_jam(a: u32, n: u32) -> u32 {
    if (n == 0u) {
        return a;
    }
    if (n < 32u) {
        return (a >> n) | select(0u, 1u, (a << (32u - n)) != 0u);
    }
    return select(0u, 1u, a != 0u);
}

fn nm_shr64_jam(a: vec2<u32>, n: u32) -> vec2<u32> {
    if (n == 0u) {
        return a;
    }
    if (n < 64u) {
        let r = nm_shr64(a, n);
        let back = nm_shl64(r, n);
        return vec2<u32>(r.x | select(0u, 1u, back.x != a.x || back.y != a.y), r.y);
    }
    return vec2<u32>(select(0u, 1u, !nm_is0_64(a)), 0u);
}

// The full 128-bit product of two 64-bit values.
fn nm_mul64w(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let ll = nm_mul32w(a.x, b.x);
    let lh = nm_mul32w(a.x, b.y);
    let hl = nm_mul32w(a.y, b.x);
    let hh = nm_mul32w(a.y, b.y);
    // Word 1 collects ll.y + lh.x + hl.x; word 2 collects lh.y + hl.y + hh.x and the carries; word 3 is hh.y.
    let m1 = nm_add64(vec2<u32>(ll.y, 0u), vec2<u32>(lh.x, 0u));
    let m1b = nm_add64(m1, vec2<u32>(hl.x, 0u));
    let m2 = nm_add64(vec2<u32>(lh.y, 0u), vec2<u32>(hl.y, 0u));
    let m2b = nm_add64(m2, vec2<u32>(hh.x, 0u));
    let m2c = nm_add64(m2b, vec2<u32>(m1b.y, 0u));
    return vec4<u32>(ll.x, m1b.x, m2c.x, hh.y + m2c.y);
}

fn nm_add96(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
    let x = a.x + b.x;
    let c0 = select(0u, 1u, x < a.x);
    let y0 = a.y + b.y;
    let c1 = select(0u, 1u, y0 < a.y);
    let y = y0 + c0;
    let c2 = select(0u, 1u, y < y0);
    return vec3<u32>(x, y, a.z + b.z + c1 + c2);
}

fn nm_sub96(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
    let x = a.x - b.x;
    let w0 = select(0u, 1u, a.x < b.x);
    let y0 = a.y - b.y;
    let w1 = select(0u, 1u, a.y < b.y);
    let y = y0 - w0;
    let w2 = select(0u, 1u, y0 < w0);
    return vec3<u32>(x, y, a.z - b.z - w1 - w2);
}

fn nm_le96(a: vec3<u32>, b: vec3<u32>) -> bool {
    if (a.z != b.z) {
        return a.z < b.z;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    return a.x <= b.x;
}

// A 64-bit value shifted left by n in [0, 32] into 96 bits.
fn nm_shl64to96(a: vec2<u32>, n: u32) -> vec3<u32> {
    if (n == 0u) {
        return vec3<u32>(a.x, a.y, 0u);
    }
    if (n < 32u) {
        return vec3<u32>(a.x << n, (a.y << n) | (a.x >> (32u - n)), a.y >> (32u - n));
    }
    return vec3<u32>(0u, a.x, a.y);
}
