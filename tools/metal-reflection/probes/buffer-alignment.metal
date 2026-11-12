// Buffer alignment real Metal reflects for small element types, by address space and stage.
// Measured (Apple M4, macOS 27.0.1, air64-apple-ios17.0): `constant` buffers reflect bufferAlignment
// max(alignment, 4) in every stage; `device` buffers reflect the element's alignment.
#include <metal_stdlib>
using namespace metal;
kernel void k0(constant uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]], constant ushort *c [[buffer(2)]],
               constant half *d [[buffer(3)]], constant char2 *e [[buffer(4)]], device uint *o [[buffer(5)]],
               uint i [[thread_position_in_grid]])
{
    o[i] = a[i] + b[i] + c[i] + uint(d[i]) + uint(e[i].x);
}
kernel void k1(constant float *a [[buffer(0)]], constant bool *b [[buffer(1)]], device ushort *c [[buffer(2)]],
               device uint *o [[buffer(5)]], uint i [[thread_position_in_grid]])
{
    o[i] = uint(a[i]) + b[i] + c[i];
}
vertex float4 v0(constant uchar *a [[buffer(0)]], device const uchar *b [[buffer(1)]], constant ushort *c [[buffer(2)]],
                 uint vid [[vertex_id]])
{
    return float4(a[vid] + b[vid] + c[vid]);
}
fragment float4 f0(float4 p [[position]], constant uchar *a [[buffer(0)]], device const uchar *b [[buffer(1)]],
                   constant ushort *c [[buffer(2)]], device const ushort *d [[buffer(3)]], constant half *e [[buffer(4)]],
                   constant float *g [[buffer(5)]])
{
    uint i = uint(p.x);
    return float4(a[i] + b[i] + c[i] + d[i] + e[i] + g[i]);
}
