//! Kernels in the exact subset: integer arithmetic of every width (unsigned wrapping arithmetic; signed operations
//! only where MSL defines them, so no signed overflow), conversions, control flow, private, threadgroup and device
//! memory with sub-word stores by neighboring threads.

use super::harness::{bytes, differential};

const HEAD: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

const OPS: &str = r#"
#define OPS(U, S, W) \
kernel void ops_##U(device const U* a [[buffer(0)]], device const U* b [[buffer(1)]], device U* out [[buffer(2)]], \
                    uint i [[thread_position_in_grid]]) { \
    U x = a[i], y = b[i]; \
    U yn = y == U(0) ? U(1) : y; \
    S sx = S(x), sy = S(yn); \
    if (sx == S(W(1) << (sizeof(U) * 8 - 1)) && sy == S(-1)) sy = S(1); \
    uint n = sizeof(U) * 8; uint s = uint(y) % n; \
    device U* o = out + i * 26; \
    o[0] = U(W(x) + W(y)); o[1] = U(W(x) - W(y)); o[2] = U(W(x) * W(y)); \
    o[3] = U(W(x) / W(yn)); o[4] = U(W(x) % W(yn)); o[5] = U(sx / sy); o[6] = U(sx % sy); \
    o[7] = U(W(x) << s); o[8] = U(W(x) >> s); o[9] = U(sx >> s); \
    o[10] = x & y; o[11] = x | y; o[12] = x ^ y; o[13] = U(~x); \
    o[14] = U(x < y); o[15] = U(sx < S(y)); o[16] = U(x >= y); o[17] = U(sx >= S(y)); o[18] = U(x == y); \
    o[19] = min(x, y); o[20] = max(x, y); o[21] = U(min(sx, S(y))); o[22] = U(max(sx, S(y))); \
    o[23] = U(sx < S(0) ? -W(sx) : W(sx)); o[24] = U(clz(x)); o[25] = U(popcount(y)); \
}
OPS(uchar, char, uint)
OPS(ushort, short, uint)
OPS(uint, int, uint)
OPS(ulong, long, ulong)
"#;

fn run_ops(ty: &str, size: usize) {
    let n = 256;
    let a = bytes(1 + size as u64, n * size);
    let mut b = bytes(100 + size as u64, n * size);
    // Some divisors of -1 and of the most negative value, and some zero.
    for k in 0..8 {
        let at = k * 29 % n * size;
        let v: u8 = if k % 2 == 0 { 0xff } else { 0 };
        for j in 0..size {
            b[at + j] = v;
        }
    }
    let out = vec![0xa5u8; n * size * 26];
    let r = differential(
        &format!("ops_{}", ty),
        &format!("{}{}", HEAD, OPS),
        &format!("ops_{}", ty),
        [64, 1, 1],
        [4, 1, 1],
        &[],
        &[a, b, out],
    );
    eprintln!("ops_{}: Metal = wgpu{}", ty, if r.dawn { " = Dawn" } else { " (Dawn not run: AW_DAWN_NODE unset)" });
}

#[test]
fn integer_operations_char() {
    run_ops("uchar", 1);
}

#[test]
fn integer_operations_short() {
    run_ops("ushort", 2);
}

#[test]
fn integer_operations_int() {
    run_ops("uint", 4);
}

#[test]
fn integer_operations_long() {
    run_ops("ulong", 8);
}

const CONVERT: &str = r#"
kernel void convert(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    uint x = a[2 * i];
    uint y = a[2 * i + 1];
    ulong l = (ulong(y) << 32) | x;
    device uint* o = out + i * 20;
    o[0] = uint(char(x)); o[1] = uint(short(x)); o[2] = uint(uchar(x)); o[3] = uint(ushort(x));
    o[4] = uint(long(int(x)) >> 32); o[5] = uint(ulong(x) >> 32);
    o[6] = uint(l); o[7] = uint(l >> 32); o[8] = uint(long(l) >> 40); o[9] = uint(short(l));
    o[10] = uint(long(char(x)) >> 3); o[11] = uint(l % 1000003ul); o[12] = uint((l / 7ul) >> 3);
    o[13] = uint(long(l) / long(-13)); o[14] = uint(bool(x & 4u)); o[15] = popcount(x) + clz(y);
    o[16] = uint(as_type<ushort2>(x).y) * 3u + uint(as_type<uchar4>(y).z);
    o[17] = as_type<uint>(ushort2(ushort(x), ushort(y)));
    o[18] = uint(rotate(l, ulong(x % 64u)) >> 7);
    o[19] = uint(mulhi(x, y)) ^ uint(mulhi(int(x), int(y)));
}
"#;

#[test]
fn integer_conversions() {
    let n = 256;
    let a = bytes(7, n * 8);
    let out = vec![0xa5u8; n * 4 * 20];
    differential("convert", &format!("{}{}", HEAD, CONVERT), "convert", [64, 1, 1], [4, 1, 1], &[], &[a, out]);
}

const FLOW: &str = r#"
kernel void flow(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    uint x = a[i];
    if (x % 97u == 5u) {
        return;
    }
    uint acc = 0;
    for (uint j = 0; j < (x & 15u); j++) {
        if ((x >> j) & 1u) {
            continue;
        }
        acc += j * 3u;
        if (acc > 40u) {
            break;
        }
        for (uint k = 0; k < j; k++) {
            if (((x >> k) & 3u) == 3u) {
                acc ^= k;
                break;
            }
            acc += 1u;
        }
    }
    switch (x % 7u) {
        case 0: acc += 100u; break;
        case 1: case 2: acc *= 3u; break;
        case 4: acc -= 5u; [[fallthrough]];
        case 5: acc += 7u; break;
        default: acc = ~acc;
    }
    uint w = x | 1u;
    for (uint n = 0; n < 300u && w > 1u; n++) {
        w = (w & 1u) ? 3u * w + 1u : w / 2u;
        acc += 1u;
        if (w == 16u) {
            acc += 1000u;
            break;
        }
    }
    out[i] = acc;
}
"#;

#[test]
fn control_flow() {
    let n = 256;
    let a = bytes(11, n * 4);
    let out = vec![0xa5u8; n * 4];
    differential("flow", &format!("{}{}", HEAD, FLOW), "flow", [64, 1, 1], [4, 1, 1], &[], &[a, out]);
}

const MEMORY: &str = r#"
struct P {
    char c;
    short s;
    uchar u[3];
    packed_float3 f;
    half h;
};

kernel void memory(device const uint* a [[buffer(0)]], device P* out [[buffer(1)]], device uchar* bytes [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    uint x = a[i];
    uint arr[16];
    for (uint k = 0; k < 16; k++) {
        arr[k] = x * (k + 1);
    }
    uint idx = x % 16u;
    arr[idx] ^= 0xabcdu;
    P p;
    p.c = char(x);
    p.s = short(x >> 8);
    p.u[0] = uchar(arr[(idx + 3u) % 16u]);
    p.u[1] = uchar(x >> 16);
    p.u[2] = 7;
    p.f = packed_float3(as_type<float>(x & 0x3fffffffu), 1.0f, 2.0f);
    p.h = as_type<half>(ushort(ushort(x) & 0x7bffu));
    out[i] = p;
    bytes[i] = uchar(arr[idx]);
    bytes[1024 + 3 * i + 1] = uchar(x >> 24);
    device ushort* hw = (device ushort*)(bytes + 2048);
    hw[i] = ushort(x * 7u);
}
"#;

#[test]
fn memory_sub_word_and_private_arrays() {
    let n = 256;
    let a = bytes(13, n * 4);
    // sizeof(P) is 24: c at 0, s at 2, u at 4, f at 8, h at 20, padded to the float alignment.
    let out = vec![0xa5u8; n * 24];
    let b = vec![0x5au8; 2048 + 2 * n];
    differential("memory", &format!("{}{}", HEAD, MEMORY), "memory", [64, 1, 1], [4, 1, 1], &[], &[a, out, b]);
}

const THREADGROUP: &str = r#"
kernel void tg(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]], threadgroup uint* t [[threadgroup(0)]],
               uint i [[thread_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
    threadgroup ushort s[128];
    threadgroup uchar c[64];
    t[l] = a[i];
    s[2 * l] = ushort(a[i]);
    s[2 * l + 1] = ushort(a[i] >> 16);
    c[l] = uchar(a[i] >> 3);
    out[i] = t[l] + uint(s[2 * l]) * 3u + uint(s[2 * l + 1]) + uint(c[l]) * 5u;
}
"#;

#[test]
fn threadgroup_memory_without_barriers() {
    let n = 256;
    let a = bytes(17, n * 4);
    let out = vec![0xa5u8; n * 4];
    differential("tg", &format!("{}{}", HEAD, THREADGROUP), "tg", [64, 1, 1], [4, 1, 1], &[256], &[a, out]);
}

const POINTERS: &str = r#"
struct Pair {
    uint a;
    uchar b[6];
};

kernel void pointers(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]], device uint* out [[buffer(2)]],
                     device Pair* pairs [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    uint x = a[i];
    // A pointer into one of two buffers, chosen at run time.
    device const uint* p = (x & 1u) ? a : b;
    uint v = p[(i + 3u) % 64u];
    // Pointers into one of two private arrays.
    uint u[8] = {};
    uint w[8];
    for (uint k = 0; k < 8; k++) {
        w[k] = x ^ k;
    }
    thread uint* q = (x & 2u) ? u : w;
    q[x % 8u] += 5u;
    uint s = 0;
    for (uint k = 0; k < 8; k++) {
        s = s * 31u + u[k] + w[k] * 3u;
    }
    // Whole-struct copies (memcpy) and a zeroed struct (memset) through device memory.
    Pair z = {};
    z.b[x % 6u] = uchar(x);
    pairs[i] = z;
    Pair t = pairs[i];
    t.a = t.a * 3u + x;
    if ((x & 4u) != 0u) {
        pairs[i] = t;
    }
    // The address's low bits: alignment is preserved.
    ulong addr = ulong(reinterpret_cast<ulong>(p + (x & 3u)));
    out[i] = v + s + uint(addr & 3ul) * 1000u;
}
"#;

#[test]
fn pointer_selection_and_copies() {
    let n = 64;
    let a = bytes(19, n * 4);
    let b = bytes(23, n * 4);
    let out = vec![0xa5u8; n * 4];
    let pairs = vec![0x5au8; n * 12];
    differential(
        "pointers",
        &format!("{}{}", HEAD, POINTERS),
        "pointers",
        [64, 1, 1],
        [1, 1, 1],
        &[],
        &[a, b, out, pairs],
    );
}
