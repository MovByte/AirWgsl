#!/usr/bin/env python3
"""Measurement air-probe: which air.* functions the Metal front end emits per MSL builtin.

Each case is one MSL function (case_<id>) that calls one builtin; one more case lists
every air.* metadata string and named metadata node of the module. The source is compiled
with this Mac's Metal toolchain for air64-apple-ios17.0 at -std=metal3.1, once with fast
math (the default) and once with -fno-fast-math, disassembled with llvm-dis, and the
air.* callees of each case function are recorded. The result file records the toolchain
version and the macOS build. The front end used here is the 32023 line of the target's
(32023.198 on 21H461); the intrinsic table checks every recorded name against the
target's own name sources, so a name only this toolchain emits is reported, not adopted.

Usage: air_probe.py --llvm-dis PATH --work DIR [--out PATH]
"""

import argparse
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

HEADER = """#include <metal_stdlib>
using namespace metal;
"""

# Kernel arguments shared by the arithmetic cases.
KARGS = ("device {T}* o [[buffer(0)]], device const {T}* a [[buffer(1)]], "
         "device const {T}* b [[buffer(2)]], device const {T}* c [[buffer(3)]], "
         "uint i [[thread_position_in_grid]]")

CASES = []


def kernel(cid, body, T="float", extra=""):
    args = KARGS.format(T=T) + (", " + extra if extra else "")
    CASES.append((cid, "kernel void case_%s(%s) { %s }" % (cid, args, body), body))


def raw(cid, src, summary):
    CASES.append((cid, src, summary))


FLOAT_UNARY = ["acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil", "cos", "cosh", "cospi",
               "exp", "exp2", "exp10", "fabs", "floor", "fract", "log", "log2", "log10", "rint",
               "round", "rsqrt", "sin", "sinh", "sinpi", "sqrt", "tan", "tanh", "tanpi", "trunc",
               "saturate", "sign"]
FLOAT_BINARY = ["atan2", "copysign", "fdim", "fmax", "fmin", "fmod", "pow", "powr", "nextafter",
                "step", "divide"]
FLOAT_TERNARY = ["fma", "fmax3", "fmin3", "fmedian3", "clamp", "mix", "smoothstep"]
INT_UNARY = ["abs", "clz", "ctz", "popcount", "reverse_bits"]
INT_BINARY = ["absdiff", "addsat", "hadd", "max", "min", "mulhi", "rhadd", "rotate", "subsat",
              "mul24"]
INT_TERNARY = ["clamp", "madhi", "madsat", "max3", "min3", "median3", "mad24"]

for T, tag in (("float", "f"), ("half", "h")):
    for f in FLOAT_UNARY:
        kernel("math_%s_%s" % (f, tag), "o[i] = %s(a[i]);" % f, T)
    for f in FLOAT_BINARY:
        kernel("math_%s_%s" % (f, tag), "o[i] = %s(a[i], b[i]);" % f, T)
    for f in FLOAT_TERNARY:
        kernel("math_%s_%s" % (f, tag), "o[i] = %s(a[i], b[i], c[i]);" % f, T)
    kernel("math_ldexp_%s" % tag, "o[i] = ldexp(a[i], int(b[i]));", T)
    kernel("math_frexp_%s" % tag, "int e; o[i] = frexp(a[i], e) + %s(e);" % T, T)
    kernel("math_ilogb_%s" % tag, "o[i] = %s(ilogb(a[i]));" % T, T)
    kernel("math_modf_%s" % tag, "%s ip; o[i] = modf(a[i], ip) + ip;" % T, T)
    kernel("math_sincos_%s" % tag, "%s cv; o[i] = sincos(a[i], cv) + cv;" % T, T)
    kernel("op_add_%s" % tag, "o[i] = a[i] + b[i];", T)
    kernel("op_div_%s" % tag, "o[i] = a[i] / b[i];", T)
    kernel("rel_isnan_%s" % tag, "o[i] = isnan(a[i]) ? 1 : 0;", T)
    kernel("rel_isinf_%s" % tag, "o[i] = isinf(a[i]) ? 1 : 0;", T)
    kernel("rel_isfinite_%s" % tag, "o[i] = isfinite(a[i]) ? 1 : 0;", T)
    kernel("rel_isnormal_%s" % tag, "o[i] = isnormal(a[i]) ? 1 : 0;", T)
    kernel("rel_isordered_%s" % tag, "o[i] = isordered(a[i], b[i]) ? 1 : 0;", T)
    kernel("rel_isunordered_%s" % tag, "o[i] = isunordered(a[i], b[i]) ? 1 : 0;", T)
    kernel("rel_signbit_%s" % tag, "o[i] = signbit(a[i]) ? 1 : 0;", T)
    kernel("rel_select_%s" % tag, "o[i] = select(a[i], b[i], c[i] > 0);", T)

for f in ["dot", "distance", "distance_squared", "length", "length_squared", "normalize"]:
    kernel("geom_%s" % f, "o[i] = float4(%s(a[i]%s));" % (f, ", b[i]" if f.startswith(("dot", "distance")) else ""),
           "float4")
kernel("geom_cross", "o[i] = cross(a[i], b[i]);", "float3")
kernel("geom_faceforward", "o[i] = faceforward(a[i], b[i], c[i]);", "float4")
kernel("geom_reflect", "o[i] = reflect(a[i], b[i]);", "float4")
kernel("geom_refract", "o[i] = refract(a[i], b[i], c[i].x);", "float4")
kernel("rel_all", "o[i] = all(a[i] > b[i]) ? 1 : 0;", "float4")
kernel("rel_any", "o[i] = any(a[i] > b[i]) ? 1 : 0;", "float4")
kernel("matrix_determinant", "float4x4 m(a[i], b[i], c[i], a[i] + b[i]); o[i] = float4(determinant(m));", "float4")
kernel("matrix_transpose", "float4x4 m(a[i], b[i], c[i], a[i] + b[i]); o[i] = transpose(m)[1];", "float4")

for T, tag in (("int", "i32s"), ("uint", "i32u"), ("short", "i16s"), ("ushort", "i16u"),
               ("char", "i8s"), ("uchar", "i8u"), ("long", "i64s")):
    for f in INT_UNARY:
        kernel("int_%s_%s" % (f, tag), "o[i] = %s(%s(a[i]));" % (T, f), T)
    for f in INT_BINARY:
        if f in ("mul24",) and T not in ("int", "uint"):
            continue
        kernel("int_%s_%s" % (f, tag), "o[i] = %s(%s(a[i], b[i]));" % (T, f), T)
    for f in INT_TERNARY:
        if f in ("mad24",) and T not in ("int", "uint"):
            continue
        kernel("int_%s_%s" % (f, tag), "o[i] = %s(%s(a[i], b[i], c[i]));" % (T, f), T)
    if T in ("int", "uint"):
        kernel("int_extract_bits_%s" % tag, "o[i] = extract_bits(a[i], uint(b[i]) & 15u, 8u);", T)
        kernel("int_insert_bits_%s" % tag, "o[i] = insert_bits(a[i], b[i], uint(c[i]) & 15u, 8u);", T)
    kernel("op_div_%s" % tag, "o[i] = a[i] / b[i];", T)

# Conversions (static_cast) and reinterpretation.
for src_t, dst_t, tag in (("float", "int", "f2s"), ("float", "uint", "f2u"), ("int", "float", "s2f"),
                          ("uint", "float", "u2f"), ("float", "half", "f2h"), ("half", "float", "h2f"),
                          ("half", "short", "h2s"), ("char", "float", "c2f"), ("float", "uchar", "f2uc"),
                          ("long", "float", "l2f"), ("float", "long", "f2l")):
    raw("conv_%s" % tag,
        "kernel void case_conv_%s(device %s* o [[buffer(0)]], device const %s* a [[buffer(1)]], "
        "uint i [[thread_position_in_grid]]) { o[i] = static_cast<%s>(a[i]); }" % (tag, dst_t, src_t, dst_t),
        "static_cast<%s>(%s)" % (dst_t, src_t))
raw("conv_as_type", "kernel void case_conv_as_type(device uint* o [[buffer(0)]], device const float* a "
    "[[buffer(1)]], uint i [[thread_position_in_grid]]) { o[i] = as_type<uint>(a[i]); }", "as_type<uint>(float)")

# Pack and unpack.
for f, ret, arg in (("pack_float_to_unorm4x8", "uint", "float4"), ("pack_float_to_snorm4x8", "uint", "float4"),
                    ("pack_float_to_unorm2x16", "uint", "float2"), ("pack_float_to_snorm2x16", "uint", "float2"),
                    ("pack_half_to_unorm4x8", "uint", "half4"), ("pack_float_to_srgb_unorm4x8", "uint", "float4")):
    raw("pack_%s" % f, "kernel void case_pack_%s(device %s* o [[buffer(0)]], device const %s* a [[buffer(1)]], "
        "uint i [[thread_position_in_grid]]) { o[i] = %s(a[i]); }" % (f, ret, arg, f), f)
for f, ret in (("unpack_unorm4x8_to_float", "float4"), ("unpack_snorm4x8_to_float", "float4"),
               ("unpack_unorm2x16_to_float", "float2"), ("unpack_snorm2x16_to_float", "float2"),
               ("unpack_unorm4x8_to_half", "half4"), ("unpack_unorm4x8_srgb_to_float", "float4"),
               ("unpack_unorm10a2_to_float", "float4"), ("unpack_unorm565_to_float", "float3")):
    raw("unpack_%s" % f, "kernel void case_unpack_%s(device %s* o [[buffer(0)]], device const %s* a [[buffer(1)]], "
        "uint i [[thread_position_in_grid]]) { o[i] = %s(a[i]); }" % (f, ret, "ushort" if "565" in f else "uint", f), f)

# Atomics on device and threadgroup memory.
for op in ("fetch_add", "fetch_sub", "fetch_and", "fetch_or", "fetch_xor", "fetch_min", "fetch_max", "exchange"):
    for T in ("int", "uint"):
        raw("atomic_%s_%s_device" % (op, T),
            "kernel void case_atomic_%s_%s_device(device atomic_%s* a [[buffer(0)]], uint i [[thread_position_in_grid]]) "
            "{ atomic_%s_explicit(&a[0], %s(i), memory_order_relaxed); }" % (op, T, T, op, T),
            "atomic_%s_explicit(device atomic_%s*)" % (op, T))
    raw("atomic_%s_uint_threadgroup" % op,
        "kernel void case_atomic_%s_uint_threadgroup(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) "
        "{ threadgroup atomic_uint t; atomic_%s_explicit(&t, i, memory_order_relaxed); "
        "threadgroup_barrier(mem_flags::mem_threadgroup); o[i] = atomic_load_explicit(&t, memory_order_relaxed); }" % (op, op),
        "atomic_%s_explicit(threadgroup atomic_uint*)" % op)
raw("atomic_load_store_device", "kernel void case_atomic_load_store_device(device atomic_uint* a [[buffer(0)]], "
    "uint i [[thread_position_in_grid]]) { uint v = atomic_load_explicit(&a[0], memory_order_relaxed); "
    "atomic_store_explicit(&a[1], v + i, memory_order_relaxed); }", "atomic_load_explicit, atomic_store_explicit")
raw("atomic_cmpxchg_device", "kernel void case_atomic_cmpxchg_device(device atomic_uint* a [[buffer(0)]], "
    "uint i [[thread_position_in_grid]]) { uint e = 0; atomic_compare_exchange_weak_explicit(&a[0], &e, i, "
    "memory_order_relaxed, memory_order_relaxed); }", "atomic_compare_exchange_weak_explicit")
raw("atomic_float_add_device", "kernel void case_atomic_float_add_device(device atomic_float* a [[buffer(0)]], "
    "uint i [[thread_position_in_grid]]) { atomic_fetch_add_explicit(&a[0], 1.0f, memory_order_relaxed); }",
    "atomic_fetch_add_explicit(device atomic_float*)")
raw("atomic_bool_device", "kernel void case_atomic_bool_device(device atomic_bool* a [[buffer(0)]], "
    "uint i [[thread_position_in_grid]]) { atomic_store_explicit(&a[0], true, memory_order_relaxed); }",
    "atomic_store_explicit(device atomic_bool*)")
raw("atomic_ulong_max_device", "kernel void case_atomic_ulong_max_device(device atomic_ulong* a [[buffer(0)]], "
    "uint i [[thread_position_in_grid]]) { atomic_max_explicit(&a[0], ulong(i), memory_order_relaxed); }",
    "atomic_max_explicit(device atomic_ulong*)")

# Barriers.
for flags, tag in (("mem_flags::mem_none", "none"), ("mem_flags::mem_device", "device"),
                   ("mem_flags::mem_threadgroup", "threadgroup"), ("mem_flags::mem_texture", "texture"),
                   ("mem_flags::mem_device | mem_flags::mem_threadgroup", "device_threadgroup")):
    raw("barrier_threadgroup_%s" % tag, "kernel void case_barrier_threadgroup_%s(device uint* o [[buffer(0)]], "
        "uint i [[thread_position_in_grid]]) { threadgroup_barrier(%s); o[i] = i; }" % (tag, flags),
        "threadgroup_barrier(%s)" % flags)
raw("barrier_simdgroup", "kernel void case_barrier_simdgroup(device uint* o [[buffer(0)]], uint i "
    "[[thread_position_in_grid]]) { simdgroup_barrier(mem_flags::mem_threadgroup); o[i] = i; }",
    "simdgroup_barrier(mem_flags::mem_threadgroup)")

# SIMD-group and quad-group functions.
for f in ("simd_sum", "simd_product", "simd_max", "simd_min", "simd_prefix_inclusive_sum",
          "simd_prefix_exclusive_sum", "simd_broadcast_first", "quad_sum", "quad_max", "quad_broadcast_first"):
    kernel("simd_%s" % f, "o[i] = %s(a[i]);" % f)
for f, arg in (("simd_shuffle", "ushort(i)"), ("simd_shuffle_down", "ushort(1)"), ("simd_shuffle_up", "ushort(1)"),
               ("simd_shuffle_xor", "ushort(1)"), ("simd_broadcast", "ushort(0)"), ("quad_shuffle", "ushort(i)"),
               ("quad_shuffle_xor", "ushort(1)"), ("quad_broadcast", "ushort(0)"), ("simd_shuffle_rotate_down", "ushort(1)")):
    kernel("simd_%s" % f, "o[i] = %s(a[i], %s);" % (f, arg))
kernel("simd_shuffle_and_fill_down", "o[i] = simd_shuffle_and_fill_down(a[i], b[i], ushort(1));")
kernel("simd_and", "o[i] = simd_and(a[i]);", "uint")
kernel("simd_ballot", "o[i] = uint(uint64_t(simd_ballot(a[i] > 0)));", "uint")
kernel("simd_all", "o[i] = simd_all(a[i] > 0) ? 1 : 0;", "uint")
kernel("simd_is_first", "o[i] = simd_is_first() ? 1 : 0;", "uint")
kernel("simd_active_threads_mask", "o[i] = uint(uint64_t(simd_active_threads_mask()));", "uint")

# Compute built-in inputs, read through metadata rather than calls; recorded for completeness.
raw("compute_builtins", "kernel void case_compute_builtins(device uint* o [[buffer(0)]], uint i "
    "[[thread_position_in_grid]], uint l [[thread_index_in_threadgroup]], uint w [[threads_per_simdgroup]], "
    "uint s [[thread_index_in_simdgroup]]) { o[i] = l + w + s; }", "thread_position_in_grid and related attributes")

# Fragment functions.
FRAG_ARGS = "float4 p [[position]], float2 uv [[user(uv)]]"
for f in ("dfdx", "dfdy", "fwidth"):
    for T, tag in (("float", "f"), ("half", "h")):
        raw("frag_%s_%s" % (f, tag), "fragment float4 case_frag_%s_%s(%s) { return float4(%s(%s(uv.x))); }"
            % (f, tag, FRAG_ARGS, f, T), "%s(%s)" % (f, T))
raw("frag_discard", "fragment float4 case_frag_discard(%s) { if (uv.x > 0.5) discard_fragment(); return p; }"
    % FRAG_ARGS, "discard_fragment()")
raw("frag_num_samples", "fragment float4 case_frag_num_samples(%s) { return float4(get_num_samples()); }"
    % FRAG_ARGS, "get_num_samples()")
raw("frag_sample_position", "fragment float4 case_frag_sample_position(%s, uint sid [[sample_id]]) "
    "{ return float4(get_sample_position(sid), 0, 1); }" % FRAG_ARGS, "get_sample_position(sample_id)")
raw("frag_fetch", "struct FIn { float4 p [[position]]; float2 uv [[user(uv)]]; };\n"
    "fragment float4 case_frag_fetch(FIn in [[stage_in]], float4 d [[color(0)]]) { return d * in.uv.x; }",
    "[[color(0)]] input (framebuffer fetch)")
raw("frag_interpolant", "struct VIn { interpolant<float2, interpolation::perspective> v [[user(v)]]; };\n"
    "fragment float4 case_frag_interpolant(VIn in [[stage_in]], uint sid [[sample_id]]) { "
    "return float4(in.v.interpolate_at_center() + in.v.interpolate_at_centroid() + "
    "in.v.interpolate_at_sample(sid) + in.v.interpolate_at_offset(float2(0.5)), 0, 1); }",
    "interpolant<T, P> interpolate_at_*")

# Textures. Kernel cases use explicit-LOD sampling; the fragment case samples implicitly.
TEX = [
    ("1d", "texture1d<float>", "float(i)", "uint(i)", None),
    ("1d_array", "texture1d_array<float>", "float(i), 0", "uint(i), 0", None),
    ("2d", "texture2d<float>", "float2(i)", "uint2(i)", "0"),
    ("2d_array", "texture2d_array<float>", "float2(i), 0", "uint2(i), 0", "0"),
    ("3d", "texture3d<float>", "float3(i)", "uint3(i)", "0"),
    ("cube", "texturecube<float>", "float3(i)", "uint2(i), 0", "0"),
    ("cube_array", "texturecube_array<float>", "float3(i), 0", "uint2(i), 0, 0", "0"),
]
for dim, ty, coord, icoord, lod in TEX:
    lev = ", level(0)" if lod is not None else ""
    raw("tex_sample_%s" % dim, "kernel void case_tex_sample_%s(device float4* o [[buffer(0)]], %s t [[texture(0)]], "
        "sampler s [[sampler(0)]], uint i [[thread_position_in_grid]]) { o[i] = t.sample(s, %s%s); }"
        % (dim, ty, coord, lev), "%s::sample" % ty)
    rd_lod = ", 0" if dim in ("2d", "2d_array", "3d", "cube", "cube_array") else ""
    raw("tex_read_%s" % dim, "kernel void case_tex_read_%s(device float4* o [[buffer(0)]], %s t [[texture(0)]], "
        "uint i [[thread_position_in_grid]]) { o[i] = t.read(%s%s); }" % (dim, ty, icoord, rd_lod), "%s::read" % ty)
    wty = ty.replace("<float>", "<float, access::write>")
    raw("tex_write_%s" % dim, "kernel void case_tex_write_%s(%s t [[texture(0)]], uint i [[thread_position_in_grid]]) "
        "{ t.write(float4(1), %s%s); }" % (dim, wty, icoord, rd_lod), "%s::write" % wty)
    raw("tex_query_%s" % dim, "kernel void case_tex_query_%s(device uint* o [[buffer(0)]], %s t [[texture(0)]], "
        "uint i [[thread_position_in_grid]]) { o[i] = t.get_width()%s; }"
        % (dim, ty, " + t.get_num_mip_levels()" if lod is not None else ""), "%s::get_width and get_num_mip_levels" % ty)
    if dim in ("2d", "2d_array", "cube", "cube_array"):
        raw("tex_gather_%s" % dim, "kernel void case_tex_gather_%s(device float4* o [[buffer(0)]], %s t [[texture(0)]], "
            "sampler s [[sampler(0)]], uint i [[thread_position_in_grid]]) { o[i] = t.gather(s, %s); }" % (dim, ty, coord),
            "%s::gather" % ty)
raw("tex_sample_2d_implicit", "fragment float4 case_tex_sample_2d_implicit(%s, texture2d<float> t [[texture(0)]], "
    "sampler s [[sampler(0)]]) { return t.sample(s, uv); }" % FRAG_ARGS, "texture2d<float>::sample (implicit LOD)")
raw("tex_sample_2d_options", "fragment float4 case_tex_sample_2d_options(%s, texture2d<float> t [[texture(0)]], "
    "sampler s [[sampler(0)]]) { return t.sample(s, uv, bias(1.0), int2(1, -1)) + t.sample(s, uv, "
    "gradient2d(float2(1), float2(1))) + t.sample(s, uv, min_lod_clamp(1.0)); }" % FRAG_ARGS,
    "texture2d<float>::sample with bias, offset, gradient and min_lod_clamp")
raw("tex_sample_2d_half", "fragment half4 case_tex_sample_2d_half(%s, texture2d<half> t [[texture(0)]], "
    "sampler s [[sampler(0)]]) { return t.sample(s, uv); }" % FRAG_ARGS, "texture2d<half>::sample")
raw("tex_read_2d_uint", "kernel void case_tex_read_2d_uint(device uint4* o [[buffer(0)]], texture2d<uint> t "
    "[[texture(0)]], uint i [[thread_position_in_grid]]) { o[i] = t.read(uint2(i)); }", "texture2d<uint>::read")
raw("tex_read_2d_ms", "kernel void case_tex_read_2d_ms(device float4* o [[buffer(0)]], texture2d_ms<float> t "
    "[[texture(0)]], uint i [[thread_position_in_grid]]) { o[i] = t.read(uint2(i), 0) + float4(t.get_num_samples()); }",
    "texture2d_ms<float>::read and get_num_samples")
raw("tex_buffer", "kernel void case_tex_buffer(device float4* o [[buffer(0)]], texture_buffer<float, access::read_write> t "
    "[[texture(0)]], uint i [[thread_position_in_grid]]) { o[i] = t.read(i); t.write(o[i], i); }",
    "texture_buffer<float>::read and write")
raw("tex_read_write_2d", "kernel void case_tex_read_write_2d(texture2d<float, access::read_write> t [[texture(0)]], "
    "uint i [[thread_position_in_grid]]) { float4 v = t.read(uint2(i)); t.fence(); t.write(v, uint2(i)); }",
    "texture2d<float, access::read_write>::read, fence and write")
raw("tex_null", "kernel void case_tex_null(device uint* o [[buffer(0)]], texture2d<float> t [[texture(0)]], "
    "uint i [[thread_position_in_grid]]) { o[i] = is_null_texture(t) ? 1 : 0; }", "is_null_texture")
for dim, ty, coord in (("depth_2d", "depth2d<float>", "float2(i)"), ("depth_2d_array", "depth2d_array<float>", "float2(i), 0"),
                       ("depth_cube", "depthcube<float>", "float3(i)")):
    raw("tex_sample_compare_%s" % dim, "kernel void case_tex_sample_compare_%s(device float* o [[buffer(0)]], %s t "
        "[[texture(0)]], sampler s [[sampler(0)]], uint i [[thread_position_in_grid]]) { o[i] = t.sample_compare(s, %s, "
        "0.5, level(0)) + t.sample(s, %s, level(0)); }" % (dim, ty, coord, coord), "%s::sample_compare and sample" % ty)
    raw("tex_gather_compare_%s" % dim, "kernel void case_tex_gather_compare_%s(device float4* o [[buffer(0)]], %s t "
        "[[texture(0)]], sampler s [[sampler(0)]], uint i [[thread_position_in_grid]]) { o[i] = t.gather_compare(s, %s, "
        "0.5); }" % (dim, ty, coord), "%s::gather_compare" % ty)
raw("tex_calculate_lod", "fragment float4 case_tex_calculate_lod(%s, texture2d<float> t [[texture(0)]], "
    "sampler s [[sampler(0)]]) { return float4(t.calculate_clamped_lod(s, uv), t.calculate_unclamped_lod(s, uv), 0, 1); }"
    % FRAG_ARGS, "texture2d<float>::calculate_clamped_lod and calculate_unclamped_lod")
raw("sampler_constexpr", "constexpr sampler cs(address::clamp_to_zero, filter::linear, coord::pixel);\n"
    "kernel void case_sampler_constexpr(device float4* o [[buffer(0)]], texture2d<float> t [[texture(0)]], "
    "uint i [[thread_position_in_grid]]) { o[i] = t.sample(cs, float2(i), level(0)); }", "constexpr sampler")

# Function constants.
raw("function_constant", "constant bool fc [[function_constant(0)]];\nkernel void case_function_constant("
    "device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = is_function_constant_defined(fc) "
    "? (fc ? 1 : 2) : 3; }", "function_constant and is_function_constant_defined")

# Address spaces: one non-inlined function per MSL address space; its signature shows the
# LLVM address space number the front end assigns.
for space in ("thread", "device", "constant", "threadgroup", "threadgroup_imageblock"):
    raw("as_%s" % space, "__attribute__((noinline)) float case_as_%s(%s float* p) { return p[1]; }" % (space, space),
        "pointer parameter in the %s address space" % space)

# Vertex stage outputs and inputs.
raw("vertex_basic", "struct VOut { float4 p [[position]]; float ps [[point_size]]; float2 uv [[user(uv)]]; };\n"
    "vertex VOut case_vertex_basic(uint vid [[vertex_id]], uint iid [[instance_id]], device const float4* v [[buffer(0)]]) "
    "{ VOut o; o.p = v[vid + iid]; o.ps = 2.0; o.uv = o.p.xy; return o; }", "vertex function with position, point_size, user output")


def parse_functions(text):
    """{function name: (sorted air.* callees, sorted pointer address spaces of the signature)}."""
    out = {}
    for m in re.finditer(r"^define [^@\n]*@([-\w.$]+)\(([^\n]*)\)[^\n]*\{\n.*?^}", text, re.M | re.S):
        body = m.group(0)
        callees = set(re.findall(r"@(air\.[\w.]+)\(", body))
        spaces = {int(n) if n else 0 for n in re.findall(r"\bptr(?: addrspace\((\d+)\))?", m.group(2))}
        name = m.group(1)
        # C++ helpers are mangled; recover the case name they were declared with.
        cm = re.search(r"(case_\w+?)P", name) if name.startswith("_Z") else None
        out[cm.group(1) if cm else name] = (sorted(callees), sorted(spaces))
    return out


def tool_version(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llvm-dis", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "air_probe.toml"))
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    src = os.path.join(a.work, "probes.metal")
    with open(src, "w") as f:
        f.write(HEADER + "\n".join(c[1] for c in CASES) + "\n")
    results = {}
    for mode, flags in (("fast", []), ("precise", ["-fno-fast-math"])):
        air = os.path.join(a.work, "probes-%s.air" % mode)
        subprocess.run(["xcrun", "-sdk", "iphoneos", "metal", "-target", "air64-apple-ios17.0", "-std=metal3.1",
                        "-c", src, "-o", air] + flags, check=True)
        text = subprocess.run([a.llvm_dis, "-o", "-", air], check=True, capture_output=True, text=True).stdout
        results[mode] = parse_functions(text)
        results[mode]["(module-metadata)"] = (sorted(set(re.findall(r'!"(air\.[^"]*)"', text)) |
                                                   {"air." + n for n in re.findall(r"^!air\.([\w.]+) = ", text, re.M)}), [])
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gen"))
    import tomlout
    meta = {
        "id": "air-probe",
        "script": "tables/measure/air_probe.py",
        "compiler": tool_version(["xcrun", "-sdk", "iphoneos", "metal", "--version"]),
        "sdk": subprocess.run(["xcrun", "-sdk", "iphoneos", "--show-sdk-version"], capture_output=True,
                              text=True).stdout.strip(),
        "target": "air64-apple-ios17.0, -std=metal3.1",
        "host": "macOS %s (%s)" % (platform.mac_ver()[0], subprocess.run(["sw_vers", "-buildVersion"],
                                                                         capture_output=True, text=True).stdout.strip()),
        "disassembler": " ".join(subprocess.run([a.llvm_dis, "--version"], capture_output=True,
                                                text=True).stdout.split()[:4]),
    }
    parts = ["# Generated by tables/measure/air_probe.py. Do not edit by hand.\n", tomlout.table("meta", meta)]
    for cid, _, summary in CASES:
        fn = "case_" + cid
        if fn not in results["fast"] or fn not in results["precise"]:
            raise SystemExit("case %s missing from the compiled module" % cid)
        item = {
            "id": cid,
            "msl": summary,
            "fast": results["fast"][fn][0] or ["(none)"],
            "precise": results["precise"][fn][0] or ["(none)"],
        }
        if results["fast"][fn][1]:
            item["pointer_address_spaces"] = results["fast"][fn][1]
        parts.append(tomlout.array_item("case", item))
    parts.append(tomlout.array_item("case", {
        "id": "module-metadata",
        "msl": "every air.* metadata string and named metadata node of the probe module",
        "fast": results["fast"]["(module-metadata)"][0],
        "precise": results["precise"]["(module-metadata)"][0],
    }))
    with open(a.out, "w") as f:
        f.write("\n".join(parts))
    print("wrote %s (%d cases)" % (a.out, len(CASES) + 1))


if __name__ == "__main__":
    main()
