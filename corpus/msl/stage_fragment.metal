// Fragment-stage interface: input builtins, render-target inputs, and every output attribute.
#include <metal_stdlib>
using namespace metal;

struct FIn { float4 pos [[position]]; };

// variant 0: {"harness": true}
fragment float4 v0(FIn in [[stage_in]]) { return in.pos; }

// variant 1: {"attribute": "front_facing point_coord sample_id sample_mask primitive_id"}
fragment float4 v1(FIn in [[stage_in]], bool ff [[front_facing]], float2 pc [[point_coord]], uint sid [[sample_id]],
                   uint mask [[sample_mask]], uint pid [[primitive_id]])
{
    return float4(pc, float(sid + mask + pid), ff ? 1.0f : 0.0f) + in.pos;
}

// variant 2: {"attribute": "barycentric_coord render_target_array_index viewport_array_index"}
fragment float4 v2(FIn in [[stage_in]], float3 bc [[barycentric_coord]], uint layer [[render_target_array_index]],
                   uint vp [[viewport_array_index]])
{
    return float4(bc, float(layer + vp));
}

// variant 3: {"attribute": "color(n) outputs of each type, index(n) dual-source"}
struct FOut3 {
    float4 c0 [[color(0)]]; half4 c1 [[color(1)]]; int4 c2 [[color(2)]]; uint4 c3 [[color(3)]];
    short2 c4 [[color(4)]]; ushort c5 [[color(5)]];
};
fragment FOut3 v3(FIn in [[stage_in]])
{
    FOut3 o;
    o.c0 = in.pos; o.c1 = half4(in.pos); o.c2 = int4(in.pos); o.c3 = uint4(in.pos);
    o.c4 = short2(in.pos.xy); o.c5 = ushort(in.pos.x);
    return o;
}

// variant 4: {"attribute": "dual-source blending index(1)"}
struct FOut4 { float4 a [[color(0), index(0)]]; float4 b [[color(0), index(1)]]; };
fragment FOut4 v4(FIn in [[stage_in]]) { return { in.pos, in.pos * 0.5f }; }

// variant 5: {"attribute": "depth(any) stencil sample_mask outputs"}
struct FOut5 { float4 c [[color(0)]]; float d [[depth(any)]]; uint s [[stencil]]; uint m [[sample_mask]]; };
fragment FOut5 v5(FIn in [[stage_in]]) { return { in.pos, in.pos.z, 1u, 0xffu }; }

// variant 6: {"attribute": "depth(greater)"}
struct FOut6 { float4 c [[color(0)]]; float d [[depth(greater)]]; };
fragment FOut6 v6(FIn in [[stage_in]]) { return { in.pos, in.pos.z + 0.1f }; }

// variant 7: {"attribute": "depth(less)"}
struct FOut7 { float4 c [[color(0)]]; float d [[depth(less)]]; };
fragment FOut7 v7(FIn in [[stage_in]]) { return { in.pos, in.pos.z - 0.1f }; }

// variant 8: {"attribute": "early_fragment_tests"}
[[early_fragment_tests]] fragment float4 v8(FIn in [[stage_in]], device atomic_uint *count [[buffer(0)]])
{
    atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    return in.pos;
}

// variant 9: {"attribute": "color(n) render-target inputs (programmable blending)"}
fragment float4 v9(FIn in [[stage_in]], float4 d0 [[color(0)]], half4 d2 [[color(2)]])
{
    return d0 * 0.5f + float4(d2);
}

// variant 10: {"attribute": "interpolation qualifiers on inputs"}
struct FIn10 {
    float4 pos [[position]];
    float a [[flat]]; float b [[center_perspective]]; float c [[center_no_perspective]];
    float d [[centroid_perspective]]; float e [[centroid_no_perspective]];
    float f [[sample_perspective]]; float g [[sample_no_perspective]]; int h [[flat]];
};
fragment float4 v10(FIn10 in [[stage_in]])
{
    return float4(in.a + in.b + in.c + in.d, in.e + in.f + in.g, float(in.h), 1);
}

// variant 11: {"attribute": "fragment buffers, textures and samplers, sampler arrays"}
fragment float4 v11(FIn in [[stage_in]], constant float4 *u [[buffer(0)]], texture2d<float> t [[texture(0)]],
                    array<sampler, 2> s [[sampler(0)]])
{
    return t.sample(s[0], in.pos.xy) + t.sample(s[1], in.pos.xy) + u[0];
}

// variant 12: {"attribute": "discard with side effects after"}
fragment float4 v12(FIn in [[stage_in]], device float *w [[buffer(0)]])
{
    if (in.pos.x > 4.0f)
        discard_fragment();
    w[uint(in.pos.y)] = in.pos.x;
    return in.pos;
}
