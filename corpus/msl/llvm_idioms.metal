// Source patterns the compiler lowers to generic LLVM intrinsics rather than air.* calls:
// aggregate copies and fills in every address-space pair, integer and float min/max/abs and
// saturation idioms, rotates, byte swaps, multiply-add contraction, assumptions, and inlined
// helpers with non-aliasing pointer arguments.
// profiles: default; -fno-fast-math
#include <metal_stdlib>
using namespace metal;

struct Big { float4 v[6]; int k[5]; };

// variant 0: {"harness": true}
kernel void v0(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = i; }

// variant 1: {"idiom": "aggregate copies between address spaces"}
kernel void v1(device Big *d [[buffer(0)]], constant Big *c [[buffer(1)]], uint i [[thread_index_in_threadgroup]])
{
    threadgroup Big tg[2];
    thread Big t = d[i];
    thread Big u = c[i & 1];
    tg[i & 1] = d[i + 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    thread Big w = tg[(i + 1) & 1];
    d[i + 2] = c[0];
    d[i + 3] = tg[0];
    d[i + 4] = t;
    d[i + 5].k[i % 5] = u.k[0] + w.k[1];
}

// variant 2: {"idiom": "zero fills of thread, threadgroup and device memory"}
kernel void v2(device Big *d [[buffer(0)]], device float *o [[buffer(1)]], uint i [[thread_index_in_threadgroup]])
{
    float arr[64] = {};
    threadgroup Big tg[1];
    if (i == 0)
        tg[0] = Big{};
    d[i] = Big{};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    arr[i & 63] = tg[0].v[i % 6].x;
    o[i] = arr[(i * 7) & 63];
}

// variant 3: {"idiom": "overlapping copies"}
kernel void v3(device uchar *d [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    thread uchar buf[96];
    for (uint k = 0; k < 96; k++)
        buf[k] = d[i + k];
    for (uint k = 0; k < 64; k++)
        buf[k + (i & 7)] = buf[k];
    for (uint k = 0; k < 96; k++)
        d[i * 96 + k] = buf[k];
}

// variant 4: {"idiom": "integer min, max, abs and saturating subtraction"}
kernel void v4(device const int *a [[buffer(0)]], device const uint *b [[buffer(1)]], device const short *c [[buffer(2)]],
               device const ushort *d [[buffer(3)]], device const char *e [[buffer(4)]], device const uchar *f [[buffer(5)]],
               device const long *g [[buffer(6)]], device const ulong *h [[buffer(7)]], device ulong *o [[buffer(8)]],
               uint i [[thread_position_in_grid]])
{
    int ai = a[i], aj = a[i + 1];
    uint bi = b[i], bj = b[i + 1];
    short ci = c[i], cj = c[i + 1];
    ushort di = d[i], dj = d[i + 1];
    char ei = e[i], ej = e[i + 1];
    uchar fi = f[i], fj = f[i + 1];
    long gi = g[i], gj = g[i + 1];
    ulong hi = h[i], hj = h[i + 1];
    o[i * 16 + 0] = ulong(ai < aj ? ai : aj) + ulong(ai > aj ? ai : aj) + ulong(ai < 0 ? -ai : ai);
    o[i * 16 + 1] = ulong(bi < bj ? bi : bj) + ulong(bi > bj ? bi : bj) + ulong(bi > bj ? bi - bj : 0u);
    o[i * 16 + 2] = ulong(ci < cj ? ci : cj) + ulong(ci > cj ? ci : cj) + ulong(ci < 0 ? short(-ci) : ci);
    o[i * 16 + 3] = ulong(di < dj ? di : dj) + ulong(di > dj ? di : dj) + ulong(di > dj ? ushort(di - dj) : ushort(0));
    o[i * 16 + 4] = ulong(ei < ej ? ei : ej) + ulong(ei > ej ? ei : ej);
    o[i * 16 + 5] = ulong(fi < fj ? fi : fj) + ulong(fi > fj ? fi : fj);
    o[i * 16 + 6] = ulong(gi < gj ? gi : gj) + ulong(gi > gj ? gi : gj) + ulong(gi < 0 ? -gi : gi);
    o[i * 16 + 7] = (hi < hj ? hi : hj) + (hi > hj ? hi : hj) + (hi > hj ? hi - hj : 0ul);
}

// variant 5: {"idiom": "vector integer min, max and abs"}
kernel void v5(device const int4 *a [[buffer(0)]], device const uint4 *b [[buffer(1)]], device int4 *o [[buffer(2)]],
               uint i [[thread_position_in_grid]])
{
    int4 x = a[i], y = a[i + 1];
    uint4 p = b[i], q = b[i + 1];
    o[i] = select(x, y, x > y) + select(-x, x, x >= 0) + int4(select(p, q, p < q));
}

// variant 6: {"idiom": "rotates, byte swaps and bit counts written out"}
kernel void v6(device const uint *a [[buffer(0)]], device const ushort *b [[buffer(1)]], device const uchar *c [[buffer(2)]],
               device const ulong *d [[buffer(3)]], device uint *o [[buffer(4)]], uint i [[thread_position_in_grid]])
{
    uint x = a[i];
    uint n = a[i + 1] & 31;
    ushort y = b[i];
    uchar z = c[i];
    ulong w = d[i];
    uint rot = (x << n) | (x >> ((32 - n) & 31));
    ushort roty = ushort((y << 3) | (y >> 13));
    uchar rotz = uchar((z << 2) | (z >> 6));
    ulong rotw = (w << 7) | (w >> 57);
    uint bswap = (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
    uint bits = 0;
    for (uint v = x; v != 0; v &= v - 1)
        bits++;
    o[i * 8 + 0] = rot;
    o[i * 8 + 1] = roty;
    o[i * 8 + 2] = rotz;
    o[i * 8 + 3] = uint(rotw) ^ uint(rotw >> 32);
    o[i * 8 + 4] = bswap;
    o[i * 8 + 5] = bits;
}

// variant 7: {"idiom": "float multiply-add, abs, min, max, sqrt and exp2 written out"}
kernel void v7(device const float4 *a [[buffer(0)]], device const half4 *b [[buffer(1)]], device float4 *o [[buffer(2)]],
               uint i [[thread_position_in_grid]])
{
    float4 x = a[i], y = a[i + 1], z = a[i + 2];
    half4 p = b[i], q = b[i + 1], r = b[i + 2];
    float s = x.x * y.x + z.x;
    float2 s2 = x.xy * y.xy + z.xy;
    float3 s3 = x.xyz * y.xyz + z.xyz;
    half h = p.x * q.x + r.x;
    half2 h2 = p.xy * q.xy + r.xy;
    half3 h3 = p.xyz * q.xyz + r.xyz;
    half4 h4 = p * q + r;
    o[i * 8 + 0] = x * y + z;
    o[i * 8 + 1] = float4(s, s2) + float4(s3, 0);
    o[i * 8 + 2] = float4(h4) + float4(float(h), float2(h2), 0) + float4(float3(h3), 0);
    o[i * 8 + 3] = fmax(x, y) + fmin(x, z) + fabs(x) + sqrt(y) + exp2(z);
    o[i * 8 + 4] = float4(fmax(p, q)) + float4(fmin(p, r)) + float4(max(p.x, q.x), min(p.y, r.y), 0, 0);
}

// variant 8: {"idiom": "assumptions"}
kernel void v8(device const uint *a [[buffer(0)]], device uint *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    uint n = a[i];
    __builtin_assume(n > 0 && n < 64);
    uint acc = 0;
    for (uint k = 0; k < n; k++)
        acc += a[k];
    o[i] = acc;
}

// variant 9: {"idiom": "inlined helpers with non-aliasing pointer parameters"}
static void scale(device float *__restrict dst, const device float *__restrict src, uint n, float f)
{
    for (uint k = 0; k < n; k++)
        dst[k] = src[k] * f;
}
kernel void v9(device float *a [[buffer(0)]], device const float *b [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    scale(a + i * 16, b + i * 16, 16, 2.0f);
    scale(a + i * 16 + 8, b, 8, 0.5f);
}

// variant 10: {"idiom": "local arrays with bounded lifetimes inside loops"}
kernel void v10(device const float *a [[buffer(0)]], device float *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float acc = 0;
    for (uint k = 0; k < 4; k++) {
        float tmp[32];
        for (uint m = 0; m < 32; m++)
            tmp[m] = a[(i + k + m) & 255];
        acc += tmp[(i + k) & 31];
    }
    o[i] = acc;
}
