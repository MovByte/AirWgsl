// Address spaces and pointer patterns: arithmetic, casts, aliasing, selection between regions,
// pointers in structs, packed and aligned layouts, integer round trips of pointers.
#include <metal_stdlib>
using namespace metal;

// variant 0: {"harness": true}
kernel void v0(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = i; }

// variant 1: {"pattern": "device, constant, threadgroup and thread pointers with arithmetic"}
kernel void v1(device float *d [[buffer(0)]], constant float *c [[buffer(1)]], uint i [[thread_index_in_threadgroup]])
{
    threadgroup float tg[32];
    thread float local[4] = { 1, 2, 3, 4 };
    thread float *lp = local + (i & 3);
    threadgroup float *tp = tg + (i & 31);
    *tp = c[i] + *lp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *dp = d + i * 2;
    dp[1] = tg[(i + 1) & 31];
}

// variant 2: {"pattern": "reinterpret casts between element types and byte offsets"}
kernel void v2(device uchar *bytes [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    device uint *u = (device uint *)(bytes + 4 * i);
    device half2 *h = reinterpret_cast<device half2 *>(bytes + 64 + 4 * i);
    device ushort *s = (device ushort *)(bytes + 1);
    *u = as_type<uint>(*h);
    s[i] = ushort(i);
    bytes[3 + i] = uchar(i);
}

// variant 3: {"pattern": "select and phi between pointers into different buffers"}
kernel void v3(device float *a [[buffer(0)]], device float *b [[buffer(1)]], device float *o [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    device float *p = (i & 1) ? a : b;
    device float *q = a;
    for (uint k = 0; k < (i & 3); k++)
        q = (q[k] > 0.0f) ? b + k : q + 1;
    o[i] = p[i] + *q;
}

// variant 4: {"pattern": "structs with pointers, nested structs, arrays of structs"}
struct Node { device float *data; uint count; };
struct Outer { float3 a; packed_float3 b; float2x2 m; Node n; int arr[3]; };
kernel void v4(device Outer *s [[buffer(0)]], device float *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    Outer v = s[i];
    o[i] = v.a.x + v.b.y + v.m[1][0] + v.n.data[i % max(v.n.count, 1u)] + float(v.arr[i % 3]);
    s[i].arr[1] = int(i);
}

// variant 5: {"pattern": "packed vectors, alignas, bool and long members"}
struct alignas(16) Packed { packed_half3 h; packed_char4 c; packed_ushort2 u; bool f; long l; ulong ul; };
kernel void v5(device Packed *p [[buffer(0)]], device float *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    Packed v = p[i];
    o[i] = float(v.h.x) + float(v.c.y) + float(v.u.x) + (v.f ? 1.0f : 0.0f) + float(v.l) + float(v.ul);
    p[i + 1].h = packed_half3(half(i));
}

// variant 6: {"pattern": "pointer to integer and back"}
kernel void v6(device float *a [[buffer(0)]], device ulong *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    ulong addr = reinterpret_cast<ulong>(a + i);
    o[i] = addr;
    device float *back = reinterpret_cast<device float *>(addr + 4);
    *back = 1.0f;
}

// variant 7: {"pattern": "constant address space globals and program-scope arrays"}
constant float table7[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
constant float4 vec7 = float4(1, 2, 3, 4);
kernel void v7(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    o[i] = table7[i & 7] + vec7[i & 3];
}

// variant 8: {"pattern": "unions and memcpy-style copies of aggregates"}
union U8 { float f; uint u; half2 h; };
struct Big8 { float4 v[8]; };
kernel void v8(device U8 *u [[buffer(0)]], device Big8 *a [[buffer(1)]], device Big8 *b [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    U8 x = u[i];
    x.u ^= 1u;
    u[i + 1] = x;
    b[i] = a[i];
}

// variant 9: {"pattern": "dynamic indexing of thread arrays and vectors"}
kernel void v9(device float *o [[buffer(0)]], device const uint *idx [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    float arr[16];
    for (uint k = 0; k < 16; k++)
        arr[k] = float(k * i);
    float4 v = float4(1, 2, 3, 4);
    v[idx[i] & 3] = 7.0f;
    o[i] = arr[idx[i] & 15] + v[idx[i + 1] & 3];
}

// variant 10: {"pattern": "buffer arrays and array_ref"}
kernel void v10(device float *o [[buffer(0)]], constant array<float4, 4> &m [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = m[i & 3].x;
}

// variant 11: {"pattern": "volatile accesses"}
kernel void v11(device float *a [[buffer(0)]], volatile device uint *v [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    a[i] = 1.0f;
    v[i] = v[i + 1] + 1;
}

// variant 12: {"pattern": "atomic types inside structs in device memory"}
struct Counters { atomic_uint hits; atomic_int balance; uint plain; };
kernel void v12(device Counters *c [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    atomic_fetch_add_explicit(&c->hits, 1u, memory_order_relaxed);
    atomic_fetch_sub_explicit(&c->balance, int(i), memory_order_relaxed);
    c->plain = i;
}
