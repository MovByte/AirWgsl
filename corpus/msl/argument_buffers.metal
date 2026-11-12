// Argument buffers: resources, constants, arrays and nesting inside [[id(n)]] structures.
#include <metal_stdlib>
using namespace metal;

// variant 0: {"harness": true}
kernel void v0(device uint *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = i; }

// variant 1: {"pattern": "textures, samplers, buffers and constants in an argument buffer"}
struct Args1 {
    texture2d<float> t [[id(0)]];
    sampler s [[id(1)]];
    device float *b [[id(2)]];
    float4 k [[id(3)]];
    uint n [[id(4)]];
};
kernel void v1(device Args1 &a [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = a.t.sample(a.s, float2(i), level(0)) + a.k + float4(a.b[i % max(a.n, 1u)]);
}

// variant 2: {"pattern": "arrays of textures and samplers with dynamic indexing"}
struct Args2 {
    array<texture2d<float>, 4> t [[id(0)]];
    array<sampler, 2> s [[id(4)]];
    texture2d<float, access::write> out [[id(6)]];
};
kernel void v2(constant Args2 &a [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
    float4 c = a.t[gid.x & 3].sample(a.s[gid.y & 1], float2(gid), level(0));
    a.out.write(c, gid);
}

// variant 3: {"pattern": "nested argument buffer structures and arrays of structures"}
struct Material { texture2d<half> albedo; constant float4 *params; };
struct Scene { array<Material, 3> mats [[id(0)]]; depth2d<float> shadow [[id(6)]]; };
kernel void v3(device Scene &sc [[buffer(0)]], device half4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    Material m = sc.mats[i % 3];
    o[i] = m.albedo.read(uint2(i, 0)) * half4(m.params[0]) + half4(sc.shadow.read(uint2(i, 0)));
}

// variant 4: {"pattern": "argument buffer of buffers (pointer table)"}
struct Table4 { device float *ptrs [[id(0)]] [4]; };
kernel void v4(constant Table4 &t [[buffer(0)]], device float *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = t.ptrs[i & 3][i];
}

// variant 5: {"pattern": "fragment-stage argument buffer with render-target output"}
struct FArgs5 { texture2d<float> t [[id(0)]]; sampler s [[id(1)]]; };
struct FIn5 { float4 pos [[position]]; };
fragment float4 v5(FIn5 in [[stage_in]], constant FArgs5 &a [[buffer(0)]])
{
    return a.t.sample(a.s, in.pos.xy);
}

// variant 6: {"pattern": "array of argument buffers"}
struct Item6 { texture2d<float> t [[id(0)]]; float scale [[id(1)]]; };
kernel void v6(device Item6 *items [[buffer(0)]], device float4 *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = items[i & 7].t.read(uint2(i, 0)) * items[i & 7].scale;
}
