// Vertex-stage interface: input attributes, builtins and every output attribute.
// Variant markers follow corpus/generate.py: each variant is one entry point named v<N>.
#include <metal_stdlib>
using namespace metal;

// variant 0: {"harness": true}
vertex float4 v0(uint vid [[vertex_id]]) { return float4(float(vid)); }

// variant 1: {"attribute": "vertex_id instance_id base_vertex base_instance"}
vertex float4 v1(uint vid [[vertex_id]], uint iid [[instance_id]], uint bv [[base_vertex]], uint bi [[base_instance]])
{
    return float4(float(vid), float(iid), float(bv), float(bi));
}

// variant 2: {"attribute": "ushort vertex_id instance_id"}
vertex float4 v2(ushort vid [[vertex_id]], ushort iid [[instance_id]]) { return float4(float(vid + iid)); }

// variant 3: {"attribute": "stage_in attribute(n) of every vertex component type"}
struct VIn3 {
    float a0 [[attribute(0)]]; float2 a1 [[attribute(1)]]; float3 a2 [[attribute(2)]]; float4 a3 [[attribute(3)]];
    half a4 [[attribute(4)]]; half2 a5 [[attribute(5)]]; half3 a6 [[attribute(6)]]; half4 a7 [[attribute(7)]];
    int a8 [[attribute(8)]]; int2 a9 [[attribute(9)]]; uint3 a10 [[attribute(10)]]; uint4 a11 [[attribute(11)]];
    short2 a12 [[attribute(12)]]; ushort4 a13 [[attribute(13)]]; char4 a14 [[attribute(14)]]; uchar2 a15 [[attribute(15)]];
};
vertex float4 v3(VIn3 in [[stage_in]])
{
    return float4(in.a0) + float4(in.a1, 0, 0) + float4(in.a2, 0) + in.a3 + float4(in.a4) + float4(float2(in.a5), 0, 0)
         + float4(float3(in.a6), 0) + float4(in.a7) + float4(float(in.a8)) + float4(float2(in.a9), 0, 0)
         + float4(float3(in.a10), 0) + float4(in.a11) + float4(float2(in.a12), 0, 0) + float4(in.a13)
         + float4(in.a14) + float4(float2(in.a15), 0, 0);
}

// variant 4: {"attribute": "position point_size clip_distance invariant user"}
struct VOut4 {
    float4 pos [[position, invariant]];
    float size [[point_size]];
    float clip [[clip_distance]] [2];
    float4 color [[user(color)]];
};
vertex VOut4 v4(uint vid [[vertex_id]])
{
    VOut4 o;
    o.pos = float4(float(vid));
    o.size = 2.0f;
    o.clip[0] = 1.0f;
    o.clip[1] = -1.0f;
    o.color = float4(1);
    return o;
}

// variant 5: {"attribute": "interpolation qualifiers on outputs"}
struct VOut5 {
    float4 pos [[position]];
    float a [[flat]]; float b [[center_perspective]]; float c [[center_no_perspective]];
    float d [[centroid_perspective]]; float e [[centroid_no_perspective]];
    float f [[sample_perspective]]; float g [[sample_no_perspective]];
    int h [[flat]]; uint2 k [[user(k), flat]]; half4 m [[user(m)]];
};
vertex VOut5 v5(uint vid [[vertex_id]])
{
    VOut5 o;
    float x = float(vid);
    o.pos = float4(x); o.a = x; o.b = x; o.c = x; o.d = x; o.e = x; o.f = x; o.g = x;
    o.h = int(vid); o.k = uint2(vid); o.m = half4(x);
    return o;
}

// variant 6: {"attribute": "render_target_array_index viewport_array_index"}
struct VOut6 { float4 pos [[position]]; uint layer [[render_target_array_index]]; uint vp [[viewport_array_index]]; };
vertex VOut6 v6(uint vid [[vertex_id]], uint iid [[instance_id]])
{
    VOut6 o;
    o.pos = float4(float(vid));
    o.layer = iid;
    o.vp = iid & 1;
    return o;
}

// variant 7: {"attribute": "amplification_id amplification_count"}
vertex float4 v7(uint vid [[vertex_id]], ushort amp [[amplification_id]], ushort count [[amplification_count]])
{
    return float4(float(vid + amp + count));
}

// variant 8: {"attribute": "vertex buffers of every address space and access"}
struct U8 { float4x4 m; float4 v; };
vertex float4 v8(uint vid [[vertex_id]], const device float4 *p [[buffer(0)]], constant U8 &u [[buffer(1)]],
                 device float *w [[buffer(2)]])
{
    w[vid] = p[vid].x;
    return u.m * p[vid] + u.v;
}

// variant 9: {"attribute": "half, int and uint vertex outputs"}
struct VOut9 { float4 pos [[position]]; half2 h [[user(h)]]; int k [[user(k)]]; uint3 n [[user(n)]]; };
vertex VOut9 v9(uint vid [[vertex_id]])
{
    VOut9 o;
    o.pos = float4(float(vid));
    o.h = half2(half(vid));
    o.k = int(vid);
    o.n = uint3(vid);
    return o;
}
