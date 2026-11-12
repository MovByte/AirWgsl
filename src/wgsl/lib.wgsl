// Library items of the emitter (64-bit integer division and arithmetic shift). A 64-bit value is vec2<u32>(low
// word, high word), as in the numerics library. Division by zero, and the signed quotient of the most negative value
// by -1, are undefined behavior in LLVM; these return what the loop computes for them.

// Unsigned 64-bit quotient and remainder by restoring long division, one bit per step.
fn aw_udivrem64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    var q = vec2<u32>(0u, 0u);
    var r = vec2<u32>(0u, 0u);
    for (var i: i32 = 63; i >= 0; i--) {
        // r = r << 1 | bit i of a
        r = vec2<u32>(r.x << 1u, (r.y << 1u) | (r.x >> 31u));
        let bit = select((a.y >> u32(i - 32)) & 1u, (a.x >> u32(i)) & 1u, i < 32);
        r.x = r.x | bit;
        let ge = r.y > b.y || (r.y == b.y && r.x >= b.x);
        if (ge) {
            let lo = r.x - b.x;
            r = vec2<u32>(lo, r.y - b.y - select(0u, 1u, r.x < b.x));
            if (i < 32) {
                q.x = q.x | (1u << u32(i));
            } else {
                q.y = q.y | (1u << u32(i - 32));
            }
        }
    }
    return vec4<u32>(q, r);
}

fn aw_neg64(a: vec2<u32>) -> vec2<u32> {
    let lo = ~a.x + 1u;
    return vec2<u32>(lo, ~a.y + select(0u, 1u, lo == 0u));
}

// Signed 64-bit quotient (truncated toward zero) and remainder (sign of the dividend).
fn aw_sdivrem64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let na = (a.y >> 31u) != 0u;
    let nb = (b.y >> 31u) != 0u;
    let ua = select(a, aw_neg64(a), na);
    let ub = select(b, aw_neg64(b), nb);
    let qr = aw_udivrem64(ua, ub);
    let q = select(qr.xy, aw_neg64(qr.xy), na != nb);
    let r = select(qr.zw, aw_neg64(qr.zw), na);
    return vec4<u32>(q, r);
}

// Arithmetic right shift by n in [0, 63].
fn aw_sar64(a: vec2<u32>, n: u32) -> vec2<u32> {
    let s = bitcast<i32>(a.y);
    if (n == 0u) {
        return a;
    }
    if (n < 32u) {
        return vec2<u32>((a.x >> n) | (a.y << (32u - n)), bitcast<u32>(s >> n));
    }
    return vec2<u32>(bitcast<u32>(s >> (n - 32u)), bitcast<u32>(s >> 31u));
}
