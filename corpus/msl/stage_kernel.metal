// Compute-stage interface: every thread-position builtin in each accepted type, threadgroup
// memory arguments and function attributes.
#include <metal_stdlib>
using namespace metal;

// variant 0: {"harness": true}
kernel void v0(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = i; }

// variant 1: {"attribute": "grid builtins as uint3"}
kernel void v1(device uint *o [[buffer(0)]], uint3 a [[thread_position_in_grid]], uint3 b [[thread_position_in_threadgroup]],
               uint3 c [[threadgroup_position_in_grid]], uint3 d [[threads_per_grid]], uint3 e [[threads_per_threadgroup]],
               uint3 f [[threadgroups_per_grid]], uint3 g [[dispatch_threads_per_threadgroup]], uint3 h [[grid_origin]],
               uint3 k [[grid_size]])
{
    o[a.x] = b.x + c.y + d.z + e.x + f.y + g.z + h.x + k.y;
}

// variant 2: {"attribute": "grid builtins as ushort2"}
kernel void v2(device uint *o [[buffer(0)]], ushort2 a [[thread_position_in_grid]], ushort2 b [[thread_position_in_threadgroup]],
               ushort2 c [[threadgroup_position_in_grid]], ushort2 d [[threads_per_grid]], ushort2 e [[threads_per_threadgroup]],
               ushort2 f [[threadgroups_per_grid]])
{
    o[a.x] = b.x + c.y + d.x + e.y + f.x;
}

// variant 3: {"attribute": "linear indices and SIMD-group/quad-group builtins"}
kernel void v3(device uint *o [[buffer(0)]], uint a [[thread_index_in_threadgroup]], uint b [[thread_index_in_simdgroup]],
               uint c [[thread_index_in_quadgroup]], uint d [[simdgroup_index_in_threadgroup]],
               uint e [[quadgroup_index_in_threadgroup]], uint f [[threads_per_simdgroup]],
               uint g [[simdgroups_per_threadgroup]], uint h [[quadgroups_per_threadgroup]],
               uint k [[dispatch_simdgroups_per_threadgroup]], uint m [[dispatch_quadgroups_per_threadgroup]])
{
    o[a] = b + c + d + e + f + g + h + k + m;
}

// variant 4: {"attribute": "ushort linear indices"}
kernel void v4(device uint *o [[buffer(0)]], ushort a [[thread_index_in_threadgroup]], ushort b [[thread_index_in_simdgroup]],
               ushort c [[simdgroup_index_in_threadgroup]], ushort d [[threads_per_simdgroup]])
{
    o[a] = b + c + d;
}

// variant 5: {"attribute": "threadgroup(n) memory arguments"}
kernel void v5(device float *o [[buffer(0)]], threadgroup float *tg0 [[threadgroup(0)]], threadgroup float4 *tg1 [[threadgroup(1)]],
               uint i [[thread_index_in_threadgroup]])
{
    tg0[i] = float(i);
    tg1[i] = float4(i);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = tg0[i ^ 1] + tg1[i ^ 2].w;
}

// variant 6: {"attribute": "max_total_threads_per_threadgroup"}
[[max_total_threads_per_threadgroup(64)]] kernel void v6(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = i; }

// variant 7: {"attribute": "stage_in on a kernel with stage_in_origin and stage_in_size"}
struct KIn7 { float4 a [[attribute(0)]]; ushort2 b [[attribute(1)]]; };
kernel void v7(KIn7 in [[stage_in]], device float4 *o [[buffer(0)]], uint2 org [[stage_in_origin]], uint2 sz [[stage_in_size]],
               uint i [[thread_position_in_grid]])
{
    o[i] = in.a + float4(float(in.b.x + org.x + sz.y));
}

// variant 8: {"attribute": "threadgroup variables in function scope, arrays and structs"}
struct TG8 { float a; int b[3]; };
kernel void v8(device float *o [[buffer(0)]], uint i [[thread_index_in_threadgroup]])
{
    threadgroup TG8 t[16];
    threadgroup float f[64];
    t[i & 15].a = float(i);
    t[i & 15].b[i % 3] = int(i);
    f[i & 63] = float(i);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = t[(i + 1) & 15].a + float(t[i & 15].b[0]) + f[(i + 7) & 63];
}

// variant 9: {"attribute": "thread_execution_width"}
kernel void v9(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]], uint w [[thread_execution_width]]) { o[i] = w; }

// variant 10: {"attribute": "stage_in on a kernel"}
struct KIn10 { float4 a [[attribute(0)]]; ushort2 b [[attribute(1)]]; half3 c [[attribute(2)]]; };
kernel void v10(KIn10 in [[stage_in]], device float4 *o [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    o[i] = in.a + float4(float(in.b.x)) + float4(float3(in.c), 0);
}
