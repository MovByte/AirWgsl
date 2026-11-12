// Pointer members of an argument buffer: real Metal applies the `constant` alignment minimum of 4 to them as to
// buffer arguments (measured: `constant uchar *` and `constant half *` members reflect alignment 4, a `device ushort *`
// member 2).
#include <metal_stdlib>
using namespace metal;
struct AB {
    constant uchar *p [[id(0)]];
    device ushort *q [[id(1)]];
    constant half *h [[id(2)]];
};
kernel void k(device AB &ab [[buffer(0)]], device uint *o [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    o[i] = ab.p[i] + ab.q[i] + uint(ab.h[i]);
}
