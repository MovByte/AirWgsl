#!/usr/bin/env python3
"""Generate the MSL sources of the AIR corpus.

Usage:

    generate.py OUT_DIR

Writes ``OUT_DIR/msl/<group>.metal`` and ``OUT_DIR/manifest.json``. Every group is one source file
in which each variant is its own entry point (``v<N>``), so that the AIR of a variant can be
attributed to it after compilation, and so that ``corpus/build.py`` can drop the variants the
compiler rejects and keep the rest. Variant ``v0`` of every group is the harness alone (same
stage, same arguments, no probe): when it fails to compile at a language version the group
records "harness rejected" for that version, which keeps a harness limitation from being read as
a rejection of the probed builtin.

The probes sweep the whole MSL value-type space (``bool``, the integer types, ``half``,
``float``, ``bfloat``, scalars and vectors of 2, 3 and 4) for every builtin, rather than the
types the specification lists: which combinations exist is measured by compiling, not assumed.
Texture probes sweep every texture type, access qualifier, component type, coordinate type and
method form of the Metal Shading Language standard library. Results are recorded by
``build.py``; the coverage report (``coverage.py``) compares them with the system corpus and the
target compiler.

The catalog follows the headers of the Metal Shading Language standard library chapter by
chapter (math, common, integer, relational, geometric, matrix, pack, pixel formats, atomics,
compute, SIMD-group, quad-group, SIMD-group matrix, graphics, interpolation, textures, samplers,
imageblocks, command buffers, visible function tables, ray tracing, mesh, tessellation,
function constants, conversions, address spaces). It lists each builtin with the
``__metal_*`` entry of the target's standard library headers that it reaches (``REACHES``), so
``coverage.py`` can check the catalog against the target's own headers.
"""

import json
import os
import sys

# Metal Shading Language versions the iOS 17 compiler accepts, with the driver flag for each.
VERSIONS = {
    "1.1": "ios-metal1.1", "1.2": "ios-metal1.2", "2.0": "ios-metal2.0", "2.1": "ios-metal2.1",
    "2.2": "ios-metal2.2", "2.3": "ios-metal2.3", "2.4": "ios-metal2.4", "3.0": "metal3.0",
    "3.1": "metal3.1",
}

SCALARS = ["bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong",
           "half", "float", "bfloat"]
INTS = ["char", "uchar", "short", "ushort", "int", "uint", "long", "ulong"]
FLOATS = ["half", "float", "bfloat"]
WIDTHS = [1, 2, 3, 4]

PRELUDE = """#include <metal_stdlib>
using namespace metal;

// Results leave the function through these so that nothing is folded away.
template <typename X> static inline void sink(device uchar *r, X v) { *(device X *)r = v; }
template <typename X> static inline float4 fold(X v)
{
    thread uchar *p = (thread uchar *)&v;
    return float4(float(p[0]), float(p[sizeof(X) - 1]), 0.0f, 1.0f);
}
struct FragIn { float4 pos [[position]]; };
"""


def vt(scalar, width):
    return scalar if width == 1 else f"{scalar}{width}"


def all_types():
    return [(s, w) for s in SCALARS for w in WIDTHS]


class Group:
    def __init__(self, name, chapter, min_version="1.1", preamble="", options=()):
        self.name = name
        self.chapter = chapter
        self.min_version = min_version
        self.preamble = preamble
        self.options = list(options)
        self.variants = []

    def add(self, params, function, reaches=()):
        self.variants.append({"params": params, "function": function, "reaches": list(reaches)})


# Entry-point builders. ``args`` is a list of (type, name, attribute) for extra parameters.
# Probe inputs arrive through one untyped buffer and are reached through typed pointers
# (``a0``, ``a1``...), so the buffer-argument rules of a language version never decide whether a
# builtin is accepted for a type.
def _inputs(space, inputs):
    return "".join(f"    {space} const {t} *a{k} = ({space} const {t} *)(in_bytes + {k * 4096});\n"
                   for k, t in enumerate(inputs))


def kernel_fn(name, body, inputs=(), extra=()):
    params = ["device const uchar *in_bytes [[buffer(0)]]"]
    params += [f"{t} {n} {a}".strip() for t, n, a in extra]
    params += ["device uchar *r [[buffer(30)]]", "uint i [[thread_position_in_grid]]"]
    return f"kernel void {name}({', '.join(params)})\n{{\n{_inputs('device', inputs)}{body}\n}}\n"


def fragment_fn(name, body, inputs=(), extra=(), ret="return acc;"):
    params = ["FragIn in [[stage_in]]", "constant uchar *in_bytes [[buffer(0)]]"]
    params += [f"{t} {n} {a}".strip() for t, n, a in extra]
    return (f"fragment float4 {name}({', '.join(params)})\n{{\n    uint i = uint(in.pos.x);\n"
            f"    float4 acc = float4(0.0f);\n{_inputs('constant', inputs)}{body}\n    {ret}\n}}\n")


def vertex_fn(name, body, inputs=(), extra=()):
    params = ["constant uchar *in_bytes [[buffer(0)]]"]
    params += [f"{t} {n} {a}".strip() for t, n, a in extra]
    params += ["uint i [[vertex_id]]"]
    return (f"vertex float4 {name}({', '.join(params)})\n{{\n    float4 acc = float4(0.0f);\n"
            f"{_inputs('constant', inputs)}{body}\n    return acc;\n}}\n")


def k_sink(expr):
    return f"    sink(r + i * 64, {expr});"


def f_sink(expr):
    return f"    acc += fold({expr});"


# ---------------------------------------------------------------------------------------------
# Math, common, integer, relational, geometric (one variant per builtin x type x namespace)
# ---------------------------------------------------------------------------------------------

MATH_UNARY = ["acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil", "cos", "cosh", "cospi",
              "exp", "exp2", "exp10", "fabs", "abs", "floor", "fract", "ilogb", "log", "log2",
              "log10", "rint", "round", "rsqrt", "sin", "sinh", "sinpi", "sqrt", "tan", "tanh",
              "tanpi", "trunc", "saturate", "sign", "isfinite", "isinf", "isnan", "isnormal",
              "signbit", "all", "any", "length", "length_squared", "normalize",
              "clz", "ctz", "popcount", "reverse_bits", "determinant", "transpose"]
MATH_BINARY = ["atan2", "copysign", "fdim", "fmax", "fmin", "fmod", "max", "min", "pow", "powr",
               "nextafter", "divide", "step", "isordered", "isunordered", "distance",
               "distance_squared", "dot", "cross", "reflect", "absdiff", "addsat", "subsat",
               "hadd", "rhadd", "mulhi", "mul24", "rotate"]
MATH_TERNARY = ["fma", "fmax3", "fmin3", "fmedian3", "max3", "min3", "median3", "clamp", "mix",
                "smoothstep", "faceforward", "madhi", "madsat", "mad24", "select"]

REACH = {  # MSL builtin -> __metal_* entry it reaches in the target headers
    "abs": "abs", "fabs": "fabs", "absdiff": "absdiff", "addsat": "addsat", "subsat": "subsat",
    "hadd": "hadd", "rhadd": "rhadd", "mulhi": "mulhi", "mul24": "mul24", "madhi": "madhi",
    "madsat": "madsat", "mad24": "mad24", "rotate": "rotate", "clz": "clz", "ctz": "ctz",
    "popcount": "popcount", "reverse_bits": "reverse_bits", "max": "max", "min": "min",
    "fmax": "fmax", "fmin": "fmin", "fmax3": "fmax3", "fmin3": "fmin3", "fmedian3": "fmedian3",
    "max3": "max3", "min3": "min3", "median3": "median3", "clamp": "clamp", "mix": "mix",
    "saturate": "saturate", "sign": "sign", "select": "select", "all": "all", "any": "any",
    "dot": "dot", "divide": "divide", "nextafter": "nextafter", "ilogb": "ilogb",
    "extract_bits": "extract_bits", "insert_bits": "insert_bits", "ldexp": "ldexp",
    "frexp": "frexp", "sincos": "sincos", "fma": "fma", "fmod": "fmod", "fract": "fract",
}
for fn in ["acos", "acosh", "asin", "asinh", "atan", "atanh", "atan2", "ceil", "cos", "cosh",
           "cospi", "exp", "exp2", "exp10", "floor", "log", "log2", "log10", "pow", "powr", "rint",
           "round", "rsqrt", "sin", "sinh", "sinpi", "sqrt", "tan", "tanh", "tanpi", "trunc"]:
    REACH[fn] = fn


def math_groups():
    groups = []
    for arity, names in ((1, MATH_UNARY), (2, MATH_BINARY), (3, MATH_TERNARY)):
        for fn in names:
            g = Group(f"math_{fn}", "math")
            g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
            for ns in ("", "metal::fast::", "metal::precise::"):
                for s, w in all_types():
                    t = vt(s, w)
                    # Mixed forms: the last argument scalar for vector calls of functions that
                    # accept it (clamp, mix, step, smoothstep, select with a bool mask).
                    forms = [[t] * arity]
                    if fn == "select":
                        forms = [[t, t, vt("bool", w)]]
                    if w > 1 and fn in ("clamp", "mix", "smoothstep", "step", "min", "max",
                                        "fmin", "fmax", "fmax3", "fmin3", "max3", "min3",
                                        "ldexp", "rotate"):
                        forms.append([t] * (arity - 1) + [s] if fn not in ("step", "smoothstep")
                                     else [s] * (arity - 1) + [t])
                    if fn == "transpose" or fn == "determinant":
                        if s not in ("half", "float") or w == 1:
                            continue
                        forms = [[f"{s}{w}x{w}"]]
                    for form in forms:
                        args = ", ".join(f"a{k}[i]" for k in range(len(form)))
                        params = {"fn": fn, "ns": ns.strip(":") or "default", "args": form}
                        g.add(params, kernel_fn(f"v{len(g.variants)}", k_sink(f"{ns}{fn}({args})"), form),
                              [REACH[fn]] if fn in REACH else [])
            groups.append(g)
    # Out-parameter and mixed-type math.
    g = Group("math_outparams", "math")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    for ns in ("", "metal::fast::", "metal::precise::"):
        for s, w in all_types():
            t = vt(s, w)
            it = vt("int", w)
            for fn, body, inputs in (
                ("sincos", f"    {t} c; sink(r + i * 64, {ns}sincos(a0[i], c)); sink(r + i * 64 + 32, c);", [t]),
                ("modf", f"    {t} ip; sink(r + i * 64, {ns}modf(a0[i], ip)); sink(r + i * 64 + 32, ip);", [t]),
                ("frexp", f"    {it} e; sink(r + i * 64, {ns}frexp(a0[i], e)); sink(r + i * 64 + 32, e);", [t]),
                ("ldexp", k_sink(f"{ns}ldexp(a0[i], a1[i])"), [t, it]),
                ("refract", k_sink(f"{ns}refract(a0[i], a1[i], a2[i])"), [t, t, s]),
                ("extract_bits", k_sink("extract_bits(a0[i], a1[i], a2[i])"), [t, "uint", "uint"]),
                ("insert_bits", k_sink("insert_bits(a0[i], a1[i], a2[i], a3[i])"), [t, t, "uint", "uint"]),
            ):
                if fn in ("extract_bits", "insert_bits") and ns:
                    continue
                g.add({"fn": fn, "ns": ns.strip(":") or "default", "type": t},
                      kernel_fn(f"v{len(g.variants)}", body, inputs), [REACH[fn]] if fn in REACH else [])
    groups.append(g)
    return groups


def conversion_groups():
    """Every scalar/vector conversion by static_cast, implicit conversion and as_type."""
    g = Group("convert_static_cast", "conversion")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    for s, w in all_types():
        for d in SCALARS:
            src, dst = vt(s, w), vt(d, w)
            g.add({"from": src, "to": dst},
                  kernel_fn(f"v{len(g.variants)}", k_sink(f"static_cast<{dst}>(a0[i])"), [src]))
    gb = Group("convert_as_type", "conversion")
    gb.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    sizes = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2, "int": 4, "uint": 4,
             "long": 8, "ulong": 8, "half": 2, "float": 4, "bfloat": 2}
    for s, w in all_types():
        for d, dw in all_types():
            # as_type requires equal sizes; 3-component vectors occupy 4 elements.
            if sizes[s] * (4 if w == 3 else w) != sizes[d] * (4 if dw == 3 else dw):
                continue
            gb.add({"from": vt(s, w), "to": vt(d, dw)},
                   kernel_fn(f"v{len(gb.variants)}", k_sink(f"as_type<{vt(d, dw)}>(a0[i])"), [vt(s, w)]))
    return [g, gb]


# ---------------------------------------------------------------------------------------------
# Packing and pixel-format types
# ---------------------------------------------------------------------------------------------

PACK = ["pack_float_to_snorm4x8", "pack_float_to_unorm4x8", "pack_float_to_snorm2x16",
        "pack_float_to_unorm2x16", "pack_float_to_srgb_unorm4x8", "pack_float_to_unorm10a2",
        "pack_float_to_snorm10a2", "pack_float_to_unorm565", "pack_half_to_snorm4x8",
        "pack_half_to_unorm4x8", "pack_half_to_snorm2x16", "pack_half_to_unorm2x16",
        "pack_half_to_srgb_unorm4x8", "pack_half_to_unorm10a2", "pack_half_to_snorm10a2",
        "pack_half_to_unorm565"]
UNPACK = ["unpack_snorm4x8_to_float", "unpack_unorm4x8_to_float", "unpack_snorm2x16_to_float",
          "unpack_unorm2x16_to_float", "unpack_unorm4x8_srgb_to_float", "unpack_unorm10a2_to_float",
          "unpack_snorm10a2_to_float", "unpack_unorm565_to_float", "unpack_snorm4x8_to_half",
          "unpack_unorm4x8_to_half", "unpack_snorm2x16_to_half", "unpack_unorm2x16_to_half",
          "unpack_unorm4x8_srgb_to_half", "unpack_unorm10a2_to_half", "unpack_snorm10a2_to_half",
          "unpack_unorm565_to_half"]
PIXEL_FORMATS = ["r8unorm", "r8snorm", "r16unorm", "r16snorm", "rg8unorm", "rg8snorm",
                 "rg16unorm", "rg16snorm", "rgba8unorm", "srgba8unorm", "rgba8snorm",
                 "rgba16unorm", "rgba16snorm", "rgb10a2", "rg11b10f", "rgb9e5"]


def pack_reach(fn):
    """The __metal_pack/unpack entry of a pack_* or unpack_* function."""
    kind = "pack" if fn.startswith("pack") else "unpack"
    fmt = fn.split("_to_")[1] if kind == "pack" else fn.split("_to_")[0][len("unpack_"):]
    fmt = fmt.replace("srgb_unorm4x8", "unorm4x8_srgb").replace("unorm10a2", "unorm_rgb10a2")
    fmt = fmt.replace("unorm565", "unorm_rgb565")
    return f"{kind}_{fmt}"


PIXEL_REACH = {"r8unorm": "unorm1x8", "r8snorm": "snorm1x8", "r16unorm": "unorm1x16", "r16snorm": "snorm1x16",
               "rg8unorm": "unorm2x8", "rg8snorm": "snorm2x8", "rg16unorm": "unorm2x16", "rg16snorm": "snorm2x16",
               "rgba8unorm": "unorm4x8", "srgba8unorm": "unorm4x8_srgb", "rgba8snorm": "snorm4x8",
               "rgba16unorm": "unorm4x16", "rgba16snorm": "snorm4x16", "rgb10a2": "unorm_rgb10a2",
               "rg11b10f": "unorm_rg11b10f", "rgb9e5": "unorm_rgb9e5"}


def pack_groups():
    g = Group("pack", "pack")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    for fn in PACK:
        for s, w in all_types():
            g.add({"fn": fn, "type": vt(s, w)},
                  kernel_fn(f"v{len(g.variants)}", k_sink(f"{fn}(a0[i])"), [vt(s, w)]), [pack_reach(fn)])
    for fn in UNPACK:
        for s in ("ushort", "uint", "int", "ulong"):
            g.add({"fn": fn, "type": s}, kernel_fn(f"v{len(g.variants)}", k_sink(f"{fn}(a0[i])"), [s]),
                  [pack_reach(fn)])
    gp = Group("pixel_formats", "pixel")
    gp.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    for fmt in PIXEL_FORMATS:
        for s, w in all_types():
            t = vt(s, w)
            body = (f"    device {fmt}<{t}> *p = (device {fmt}<{t}> *)(r + 1024);\n"
                    f"    {t} v = p[i];\n    p[i + 1] = a0[i];\n    sink(r + i * 64, v);")
            gp.add({"format": fmt, "type": t}, kernel_fn(f"v{len(gp.variants)}", body, [t]),
                   [f"pack_{PIXEL_REACH[fmt]}", f"unpack_{PIXEL_REACH[fmt]}"])
    return [g, gp]


# ---------------------------------------------------------------------------------------------
# Atomics
# ---------------------------------------------------------------------------------------------

ATOMIC_TYPES = ["atomic_int", "atomic_uint", "atomic_bool", "atomic<float>", "atomic<ulong>",
                "atomic<long>", "atomic<half>", "atomic<char>", "atomic<short>", "atomic<ushort>"]
ATOMIC_VALUE = {"atomic_int": "int", "atomic_uint": "uint", "atomic_bool": "bool",
                "atomic<float>": "float", "atomic<ulong>": "ulong", "atomic<long>": "long",
                "atomic<half>": "half", "atomic<char>": "char", "atomic<short>": "short",
                "atomic<ushort>": "ushort"}
ATOMIC_OPS = ["store", "load", "exchange", "compare_exchange_weak", "fetch_add", "fetch_sub",
              "fetch_and", "fetch_or", "fetch_xor", "fetch_min", "fetch_max", "min", "max"]
ORDERS = ["memory_order_relaxed", "memory_order_seq_cst", "memory_order_acquire",
          "memory_order_release", "memory_order_acq_rel"]


def atomic_call(op, ptr, v, order):
    if op == "store":
        return f"    atomic_store_explicit({ptr}, {v}, {order});"
    if op == "load":
        return f"    sink(r + i * 64, atomic_load_explicit({ptr}, {order}));"
    if op == "compare_exchange_weak":
        return (f"    auto e = {v};\n"
                f"    sink(r + i * 64, atomic_compare_exchange_weak_explicit({ptr}, &e, {v}, {order}, {order}));")
    if op in ("min", "max"):
        return f"    atomic_{op}_explicit({ptr}, {v}, {order});"
    return f"    sink(r + i * 64, atomic_{op}_explicit({ptr}, {v}, {order}));"


def atomic_groups():
    groups = []
    for space in ("device", "threadgroup"):
        g = Group(f"atomic_{space}", "atomic")
        if space == "device":
            g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["int"]))
        else:
            g.add({"harness": True}, kernel_fn("v0", "    threadgroup int tg[4]; tg[i & 3] = a0[i]; sink(r + i * 64, tg[(i + 1) & 3]);", ["int"]))
        for at in ATOMIC_TYPES:
            vty = ATOMIC_VALUE[at]
            for op in ATOMIC_OPS:
                for order in ORDERS:
                    if space == "device":
                        decl = f"    device {at} *p = (device {at} *)(r + 4096);\n"
                        ptr = "&p[i & 7]"
                    else:
                        decl = f"    threadgroup {at} tg[8];\n"
                        ptr = "&tg[i & 7]"
                    body = decl + atomic_call(op, ptr, "a0[i]", order)
                    g.add({"type": at, "op": op, "order": order},
                          kernel_fn(f"v{len(g.variants)}", body, [vty]),
                          [f"atomic_{op}_explicit" if op not in ("min", "max") else f"atomic_{op}_explicit"])
        groups.append(g)
    return groups


# ---------------------------------------------------------------------------------------------
# Synchronization, SIMD-group, quad-group, SIMD-group matrices
# ---------------------------------------------------------------------------------------------

MEM_FLAGS = ["mem_flags::mem_none", "mem_flags::mem_device", "mem_flags::mem_threadgroup",
             "mem_flags::mem_texture", "mem_flags::mem_threadgroup_imageblock",
             "mem_flags::mem_object_data",
             "mem_flags::mem_device | mem_flags::mem_threadgroup",
             "mem_flags::mem_device | mem_flags::mem_texture",
             "mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture"]

SIMD_REDUCE = ["sum", "product", "min", "max", "and", "or", "xor", "prefix_exclusive_sum",
               "prefix_exclusive_product", "prefix_inclusive_sum", "prefix_inclusive_product",
               "broadcast_first"]
SIMD_SHUFFLE = ["shuffle", "shuffle_down", "shuffle_up", "shuffle_xor", "shuffle_rotate_down",
                "shuffle_rotate_up", "broadcast"]
SIMD_FILL = ["shuffle_and_fill_down", "shuffle_and_fill_up"]


def sync_groups():
    groups = []
    g = Group("barriers", "compute")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["int"]))
    for fn in ("threadgroup_barrier", "simdgroup_barrier"):
        for flags in MEM_FLAGS:
            g.add({"fn": fn, "flags": flags},
                  kernel_fn(f"v{len(g.variants)}", f"    {fn}({flags});\n    sink(r + i * 64, a0[i]);", ["int"]),
                  ["wg_barrier" if fn == "threadgroup_barrier" else "simdgroup_barrier"])
    groups.append(g)
    for prefix in ("simd", "quad"):
        for stage in ("kernel", "fragment"):
            g = Group(f"{prefix}group_{stage}", prefix + "group")
            build = kernel_fn if stage == "kernel" else fragment_fn
            put = k_sink if stage == "kernel" else f_sink
            g.add({"harness": True}, build("v0", put("a0[i]"), ["int"]))
            for s, w in all_types():
                t = vt(s, w)
                for op in SIMD_REDUCE:
                    g.add({"fn": f"{prefix}_{op}", "type": t},
                          build(f"v{len(g.variants)}", put(f"{prefix}_{op}(a0[i])"), [t]), [f"{prefix}_{op}"])
                for op in SIMD_SHUFFLE:
                    g.add({"fn": f"{prefix}_{op}", "type": t},
                          build(f"v{len(g.variants)}", put(f"{prefix}_{op}(a0[i], ushort(a1[i]))"), [t, "ushort"]),
                          [f"{prefix}_{op}"])
                for op in SIMD_FILL:
                    g.add({"fn": f"{prefix}_{op}", "type": t, "modulo": False},
                          build(f"v{len(g.variants)}", put(f"{prefix}_{op}(a0[i], a1[i], ushort(a2[i]))"), [t, t, "ushort"]),
                          [f"{prefix}_{op}"] + (["get_simdgroup_size"] if prefix == "simd" else []))
                    g.add({"fn": f"{prefix}_{op}", "type": t, "modulo": True},
                          build(f"v{len(g.variants)}", put(f"{prefix}_{op}(a0[i], a1[i], ushort(a2[i]), ushort(a2[i + 1]))"), [t, t, "ushort"]),
                          [f"{prefix}_{op}"])
            for fn, expr in ((f"{prefix}_all", f"{prefix}_all(a0[i])"), (f"{prefix}_any", f"{prefix}_any(a0[i])"),
                             (f"{prefix}_ballot", f"static_cast<{'simd' if prefix == 'simd' else 'quad'}_vote::vote_t>({prefix}_ballot(a0[i]))"),
                             (f"{prefix}_active_threads_mask", f"static_cast<{prefix}_vote::vote_t>({prefix}_active_threads_mask())"),
                             (f"{prefix}_is_first", f"{prefix}_is_first()"),
                             (f"{prefix}_is_helper_thread", f"{prefix}_is_helper_thread()"),
                             (f"{prefix}_vote_all", f"{prefix}_ballot(a0[i]).all()"),
                             (f"{prefix}_vote_any", f"{prefix}_ballot(a0[i]).any()")):
                g.add({"fn": fn}, build(f"v{len(g.variants)}", put(expr), ["bool"]),
                      [fn.replace("_vote_", "_vote_")])
            groups.append(g)
    g = Group("simdgroup_matrix", "simdgroup_matrix")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    for s in ("float", "half", "bfloat", "int", "short"):
        mt = f"simdgroup_matrix<{s}, 8, 8>"
        for space in ("device", "threadgroup"):
            src = "a0" if space == "device" else "tg"
            pre = "" if space == "device" else f"    threadgroup {s} tg[64];\n    tg[i & 63] = a0[i];\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            dst = f"(device {s} *)(r + 4096)" if space == "device" else "tg"
            body = (pre + f"    {mt} m;\n    simdgroup_load(m, {src}, 8, ulong2(0, 0), false);\n"
                    f"    {mt} t;\n    simdgroup_load(t, {src}, 8, ulong2(0, 0), true);\n"
                    f"    {mt} d = make_filled_simdgroup_matrix<{s}, 8, 8>(a0[i]);\n"
                    f"    {mt} e = {mt}(a0[i]);\n"
                    f"    simdgroup_multiply_accumulate(d, m, t, e);\n    simdgroup_multiply(e, m, d);\n"
                    f"    simdgroup_store(e, {dst}, 8, ulong2(0, 0), false);\n"
                    f"    simdgroup_store(d, {dst}, 8, ulong2(0, 0), true);")
            g.add({"type": s, "space": space}, kernel_fn(f"v{len(g.variants)}", body, [s]),
                  ["simdgroup_matrix_8x8_load", "simdgroup_matrix_8x8_store", "simdgroup_matrix_8x8_init_filled",
                   "simdgroup_matrix_8x8_init_diag", "simdgroup_matrix_8x8_multiply_accumulate"])
    groups.append(g)
    gm = Group("simdgroup_matrix_mixed", "simdgroup_matrix", min_version="2.3")
    gm.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    mtypes = ("half", "float", "bfloat")
    for d in mtypes:
        for a in mtypes:
            for b in mtypes:
                for c in mtypes:
                    body = (f"    simdgroup_matrix<{a}, 8, 8> ma = make_filled_simdgroup_matrix<{a}, 8, 8>({a}(a0[i]));\n"
                            f"    simdgroup_matrix<{b}, 8, 8> mb = make_filled_simdgroup_matrix<{b}, 8, 8>({b}(a0[i]));\n"
                            f"    simdgroup_matrix<{c}, 8, 8> mc = make_filled_simdgroup_matrix<{c}, 8, 8>({c}(a0[i]));\n"
                            f"    simdgroup_matrix<{d}, 8, 8> md;\n"
                            f"    simdgroup_multiply_accumulate(md, ma, mb, mc);\n"
                            f"    simdgroup_store(md, (device {d} *)(r + 4096), 8);")
                    gm.add({"d": d, "a": a, "b": b, "c": c}, kernel_fn(f"v{len(gm.variants)}", body, ["float"]),
                           ["simdgroup_matrix_8x8_multiply_accumulate"])
    groups.append(gm)
    return groups


# ---------------------------------------------------------------------------------------------
# Graphics: derivatives, discard, samples, interpolation, rasterization rate maps
# ---------------------------------------------------------------------------------------------

def graphics_groups():
    g = Group("fragment_builtins", "graphics")
    g.add({"harness": True}, fragment_fn("v0", f_sink("a0[i]"), ["float"]))
    for fn in ("dfdx", "dfdy", "fwidth"):
        for s, w in all_types():
            t = vt(s, w)
            g.add({"fn": fn, "type": t}, fragment_fn(f"v{len(g.variants)}", f_sink(f"{fn}(a0[i])"), [t]), [fn])
    g.add({"fn": "discard_fragment"},
          fragment_fn(f"v{len(g.variants)}", "    if (a0[i] > 0.5f) discard_fragment();", ["float"]), ["discard_fragment"])
    g.add({"fn": "get_num_samples"}, fragment_fn(f"v{len(g.variants)}", f_sink("get_num_samples()"), ["float"]), ["get_num_samples"])
    g.add({"fn": "get_sample_position"},
          fragment_fn(f"v{len(g.variants)}", f_sink("get_sample_position(uint(a0[i]))"), ["float"]), ["get_sample_position"])
    gi = Group("interpolants", "interpolation", preamble="")
    gi.add({"harness": True}, fragment_fn("v0", f_sink("a0[i]"), ["float"]))
    for s, w in all_types():
        t = vt(s, w)
        for interp in ("interpolation::perspective", "interpolation::no_perspective"):
            for call, reach in (("interpolate_at_center()", "center"), ("interpolate_at_centroid()", "centroid"),
                                ("interpolate_at_sample(uint(a0[i]))", "sample"),
                                ("interpolate_at_offset(float2(a0[i]))", "offset")):
                suffix = "perspective" if interp.endswith("::perspective") else "no_perspective"
                name = f"v{len(gi.variants)}"
                struct = f"S{name}"
                fn = (f"struct {struct} {{ float4 pos [[position]]; interpolant<{t}, {interp}> v [[user(v)]]; }};\n"
                      f"fragment float4 {name}({struct} in [[stage_in]], constant float *a0 [[buffer(0)]])\n{{\n"
                      f"    uint i = uint(in.pos.x);\n    return fold(in.v.{call}) + float4(a0[i]);\n}}\n")
                gi.add({"type": t, "interpolation": interp, "at": call.split("(")[0]}, fn,
                       [f"interpolate_{reach}_{suffix}"])
    gr = Group("rasterization_rate", "graphics", min_version="2.2")
    gr.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float2"]))
    for coord in ("float2", "uint2"):
        for fn in ("map_screen_to_physical_coordinates", "map_physical_to_screen_coordinates"):
            body = (f"    rasterization_rate_map_decoder map(*rrm);\n"
                    f"    sink(r + i * 64, map.{fn}({coord}(a0[i]), 0));")
            gr.add({"fn": fn, "coord": coord},
                   kernel_fn(f"v{len(gr.variants)}", body, ["float2"], [("constant rasterization_rate_map_data", "*rrm", "[[buffer(20)]]")]),
                   [fn])
    return [g, gi, gr]


# ---------------------------------------------------------------------------------------------
# Textures and samplers
# ---------------------------------------------------------------------------------------------

class Tex:
    def __init__(self, name, dim, array=False, ms=False, cube=False, depth=False, buffer=False, lod=True):
        self.name = name        # MSL template name
        self.dim = dim          # coordinate dimensions
        self.array = array
        self.ms = ms
        self.cube = cube
        self.depth = depth
        self.buffer = buffer
        self.lod = lod


TEXTURES = [
    Tex("texture1d", 1), Tex("texture1d_array", 1, array=True), Tex("texture2d", 2),
    Tex("texture2d_array", 2, array=True), Tex("texture3d", 3), Tex("texturecube", 3, cube=True),
    Tex("texturecube_array", 3, cube=True, array=True), Tex("texture2d_ms", 2, ms=True, lod=False),
    Tex("texture2d_ms_array", 2, ms=True, array=True, lod=False),
    Tex("texture_buffer", 1, buffer=True, lod=False),
    Tex("depth2d", 2, depth=True), Tex("depth2d_array", 2, depth=True, array=True),
    Tex("depthcube", 3, depth=True, cube=True), Tex("depthcube_array", 3, depth=True, cube=True, array=True),
    Tex("depth2d_ms", 2, depth=True, ms=True, lod=False),
    Tex("depth2d_ms_array", 2, depth=True, ms=True, array=True, lod=False),
]
COMPONENTS = ["float", "half", "int", "uint", "short", "ushort"]
ACCESS = ["access::sample", "access::read", "access::write", "access::read_write"]


def fvec(n):
    return "float" if n == 1 else f"float{n}"


def ivec(base, n):
    return base if n == 1 else f"{base}{n}"


def tex_methods(t):
    """(label, expression, reaches) for every method form of texture type ``t``.

    ``tex`` is the texture, ``s`` a sampler, ``c`` an input float, ``u`` an input uint.
    """
    out = []
    d = t.dim
    fc = f"{fvec(d)}(c)"
    arr = ", u" if t.array else ""
    off_t = "int2" if d == 2 and not t.cube else ("int3" if d == 3 and not t.cube else ("int" if d == 1 else None))
    off = f", {off_t}(1)" if off_t else ""
    grad = None
    if not t.ms and not t.buffer:
        grad = {1: None, 2: "gradient2d(float2(c), float2(c))", 3: "gradientcube(float3(c), float3(c))" if t.cube else "gradient3d(float3(c), float3(c))"}[d]
    if not t.ms and not t.buffer:
        base = f"tex.sample(s, {fc}{arr}"
        out.append(("sample", base + ")", "sample"))
        if off:
            out.append(("sample_offset", base + off + ")", "sample"))
        if t.lod and d > 1:
            out.append(("sample_bias", base + ", bias(c))", "sample"))
            out.append(("sample_level", base + ", level(c))", "sample"))
            out.append(("sample_min_lod_clamp", base + ", min_lod_clamp(c))", "sample"))
            out.append(("sample_bias_min_lod_clamp", base + ", bias(c), min_lod_clamp(c))", "sample"))
            if off:
                out.append(("sample_level_offset", base + ", level(c)" + off + ")", "sample"))
                out.append(("sample_bias_offset", base + ", bias(c)" + off + ")", "sample"))
            if grad:
                out.append(("sample_grad", base + f", {grad})", "sample_grad"))
                out.append(("sample_grad_min_lod_clamp", base + f", {grad}, min_lod_clamp(c))", "sample_grad"))
                if off:
                    out.append(("sample_grad_offset", base + f", {grad}" + off + ")", "sample_grad"))
        if t.depth:
            cbase = f"tex.sample_compare(s, {fc}{arr}, c"
            out.append(("sample_compare", cbase + ")", "sample_compare"))
            if off:
                out.append(("sample_compare_offset", cbase + off + ")", "sample_compare"))
            out.append(("sample_compare_level", cbase + ", level(c))", "sample_compare"))
            out.append(("sample_compare_bias", cbase + ", bias(c))", "sample_compare"))
            out.append(("sample_compare_min_lod_clamp", cbase + ", min_lod_clamp(c))", "sample_compare"))
            if grad:
                out.append(("sample_compare_grad", cbase + f", {grad})", "sample_compare_grad"))
                out.append(("sample_compare_grad_min_lod_clamp", cbase + f", {grad}, min_lod_clamp(c))", "sample_compare_grad"))
        if d >= 2 and not (d == 3 and not t.cube):
            gbase = f"tex.gather(s, {fc}{arr}"
            out.append(("gather", gbase + ")", "gather"))
            if off:
                out.append(("gather_offset", gbase + off + ")", "gather"))
            if not t.depth:
                for comp in ("x", "y", "z", "w"):
                    out.append((f"gather_{comp}", gbase + (off or "") + f", component::{comp})", "gather"))
            if t.depth:
                out.append(("gather_compare", f"tex.gather_compare(s, {fc}{arr}, c)", "gather_compare"))
                if off:
                    out.append(("gather_compare_offset", f"tex.gather_compare(s, {fc}{arr}, c{off})", "gather_compare"))
        if t.lod and d > 1:
            out.append(("calculate_clamped_lod", f"tex.calculate_clamped_lod(s, {fc})", "calculate_clamped_lod"))
            out.append(("calculate_unclamped_lod", f"tex.calculate_unclamped_lod(s, {fc})", "calculate_unclamped_lod"))
    # read
    for ct in ("uint", "ushort"):
        coord = f"{ivec(ct, d if not t.cube else 2)}(u)"
        extra = []
        if t.cube:
            extra.append(f"{ct}(u)")          # face
        if t.array:
            extra.append(f"{ct}(u)")
        if t.ms:
            extra.append(f"{ct}(u)")          # sample
        args = ", ".join([coord] + extra)
        out.append((f"read_{ct}", f"tex.read({args})", "read"))
        if t.lod:
            out.append((f"read_{ct}_lod", f"tex.read({args}, {ct}(u))", "read"))
    # queries
    out.append(("get_width", "tex.get_width()", "get_width"))
    if t.lod:
        out.append(("get_width_lod", "tex.get_width(u)", "get_width"))
    if d >= 2 and not t.buffer:
        out.append(("get_height", "tex.get_height()", "get_height"))
        if t.lod:
            out.append(("get_height_lod", "tex.get_height(u)", "get_height"))
    if d == 3 and not t.cube:
        out.append(("get_depth", "tex.get_depth()", "get_depth"))
        out.append(("get_depth_lod", "tex.get_depth(u)", "get_depth"))
    if t.array:
        out.append(("get_array_size", "tex.get_array_size()", "get_array_size"))
    if t.lod:
        out.append(("get_num_mip_levels", "tex.get_num_mip_levels()", "get_num_mip_levels"))
    if t.ms:
        out.append(("get_num_samples", "tex.get_num_samples()", "get_num_samples"))
    out.append(("is_null_texture", "is_null_texture(tex)", "is_null"))
    return out


def tex_writes(t, comp):
    out = []
    if t.ms or t.depth and t.ms:
        return out
    val = "c" if t.depth else f"vec<{comp}, 4>({comp}(c))"
    for ct in ("uint", "ushort"):
        coord = f"{ivec(ct, t.dim if not t.cube else 2)}(u)"
        extra = []
        if t.cube:
            extra.append(f"{ct}(u)")
        if t.array:
            extra.append(f"{ct}(u)")
        args = ", ".join([val, coord] + extra)
        out.append((f"write_{ct}", f"tex.write({args})", "write"))
        if t.lod:
            out.append((f"write_{ct}_lod", f"tex.write({args}, {ct}(u))", "write"))
    return out


TEX_ATOMICS = ["fetch_add", "fetch_sub", "fetch_and", "fetch_or", "fetch_xor", "fetch_min",
               "fetch_max", "exchange", "load", "store", "compare_exchange_weak", "min", "max"]


def texture_groups():
    groups = []
    for t in TEXTURES:
        comps = ["float", "half"] if t.depth else COMPONENTS
        for access in ACCESS:
            g = Group(f"tex_{t.name}_{access.split('::')[1]}", "texture")
            g.add({"harness": True}, fragment_fn("v0", f_sink("a0[i]"), ["float"]))
            for comp in comps:
                ttype = f"{t.name}<{comp}, {access}>"
                exprs = tex_methods(t) + [(lbl, e, rch) for lbl, e, rch in tex_writes(t, comp)]
                exprs.append(("fence", "tex.fence()", "fence"))
                for label, expr, reach in exprs:
                    for stage in ("fragment", "kernel"):
                        if stage == "kernel" and label.startswith(("sample", "gather", "calculate")) \
                                and "level" not in label and "grad" not in label:
                            continue  # implicit-derivative forms exist only in fragment functions
                        extra = [(ttype, "tex", "[[texture(0)]]"), ("sampler", "s", "[[sampler(0)]]")]
                        is_stmt = label.startswith(("write", "fence"))
                        body = (f"    float c = a0[i];\n    uint u = uint(a0[i]);\n"
                                + (f"    {expr};" if is_stmt else (f_sink(expr) if stage == "fragment" else k_sink(expr))))
                        build = fragment_fn if stage == "fragment" else kernel_fn
                        g.add({"texture": t.name, "component": comp, "access": access, "method": label, "stage": stage},
                              build(f"v{len(g.variants)}", body, ["float"], extra), [reach])
            groups.append(g)
        # Null textures and texture atomics (MSL 3.1).
        g = Group(f"tex_{t.name}_null_atomic", "texture")
        g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
        for comp in comps:
            g.add({"texture": t.name, "component": comp, "method": "null_texture"},
                  kernel_fn(f"v{len(g.variants)}", k_sink(f"is_null_texture({t.name}<{comp}>())"), ["float"]),
                  ["get_null", "is_null"])
        if not t.depth and not t.cube and not t.ms:
            for comp in ("int", "uint", "ulong"):
                ttype = f"{t.name}<{comp}, access::read_write>"
                for op in TEX_ATOMICS:
                    coord = f"{ivec('uint', t.dim)}(u)" if not t.buffer else "uint(u)"
                    arr = ", uint(u)" if t.array else ""
                    v = f"vec<{comp}, 4>({comp}(u))"
                    if op == "store":
                        stmt = f"    tex.atomic_store({v}, {coord}{arr});"
                    elif op == "load":
                        stmt = k_sink(f"tex.atomic_load({coord}{arr})")
                    elif op == "compare_exchange_weak":
                        stmt = (f"    vec<{comp}, 4> e = {v};\n"
                                + k_sink(f"tex.atomic_compare_exchange_weak({coord}{arr}, &e, {v})"))
                    elif op in ("min", "max"):
                        stmt = f"    tex.atomic_{op}({coord}{arr}, ulong4(u));"
                    else:
                        stmt = k_sink(f"tex.atomic_{op}({coord}{arr}, {v})")
                    body = f"    uint u = uint(a0[i]);\n{stmt}"
                    g.add({"texture": t.name, "component": comp, "method": f"atomic_{op}"},
                          kernel_fn(f"v{len(g.variants)}", body, ["float"], [(ttype, "tex", "[[texture(0)]]")]),
                          [f"atomic_{op}_explicit"])
        groups.append(g)
    # Sampler state: constexpr samplers with every enumerated property, and sampler arguments.
    gs = Group("samplers", "texture")
    gs.add({"harness": True}, fragment_fn("v0", f_sink("a0[i]"), ["float"]))
    props = ["coord::normalized", "coord::pixel", "address::repeat", "address::mirrored_repeat",
             "address::clamp_to_edge", "address::clamp_to_zero", "address::clamp_to_border",
             "address::mirrored_clamp_to_edge", "s_address::repeat", "t_address::mirrored_repeat",
             "r_address::clamp_to_edge", "border_color::transparent_black", "border_color::opaque_black",
             "border_color::opaque_white", "filter::nearest", "filter::linear", "mag_filter::linear",
             "min_filter::linear", "mip_filter::none", "mip_filter::nearest", "mip_filter::linear",
             "compare_func::never", "compare_func::less", "compare_func::less_equal",
             "compare_func::greater", "compare_func::greater_equal", "compare_func::equal",
             "compare_func::not_equal", "compare_func::always", "max_anisotropy(8)",
             "lod_clamp(1.0f, 4.0f)", "reduction::minimum", "reduction::maximum",
             "reduction::weighted_average"]
    for prop in props:
        name = f"v{len(gs.variants)}"
        gs.add({"sampler": prop},
               f"constexpr sampler smp_{name}({prop});\n"
               + fragment_fn(name, f_sink(f"tex.sample(smp_{name}, float2(a0[i]))"), ["float"],
                             [("texture2d<float>", "tex", "[[texture(0)]]")]), ["get_sampler"])
    groups.append(gs)
    return groups


# ---------------------------------------------------------------------------------------------
# Imageblocks, tile functions, raster order groups
# ---------------------------------------------------------------------------------------------

def imageblock_groups():
    g = Group("imageblocks", "imageblock", min_version="2.0")
    g.preamble = ("struct IB { half4 a [[color(0)]]; float b [[color(1)]]; };\n"
                  "struct EB { half4 a; float b; };\n"
                  "struct RT { half4 c0 [[color(0)]]; };\n"
                  "struct RT2 { half4 c0 [[color(0), raster_order_group(0)]]; };\n")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    fns = [
        ("tile_kernel_implicit",
         "kernel void {n}(imageblock<IB, imageblock_layout_implicit> ib, ushort2 lid [[thread_position_in_threadgroup]], device uchar *r [[buffer(30)]])\n"
         "{{\n    IB v = ib.read(lid);\n    v.b += 1.0f;\n    ib.write(v, lid);\n    sink(r, ib.get_width() + ib.get_height() + ib.get_num_samples());\n}}\n",
         ["imageblock_implicit_read", "imageblock_implicit_write", "get_imageblock_width", "get_imageblock_height", "get_imageblock_samples"]),
        ("tile_kernel_explicit",
         "kernel void {n}(imageblock<EB> ib, ushort2 lid [[thread_position_in_threadgroup]], device uchar *r [[buffer(30)]])\n"
         "{{\n    threadgroup_imageblock EB *p = ib.data(lid);\n    p->b += 1.0f;\n    sink(r, ib.get_width());\n}}\n",
         ["imageblock_explicit_data"]),
        ("fragment_imageblock_data",
         "fragment IB {n}(FragIn in [[stage_in]], IB ib [[imageblock_data]])\n{{\n    ib.b += 1.0f;\n    return ib;\n}}\n",
         []),
        ("raster_order_group",
         "fragment RT2 {n}(FragIn in [[stage_in]], RT2 d)\n{{\n    d.c0 += half4(1.0h);\n    return d;\n}}\n",
         []),
        ("framebuffer_fetch",
         "fragment half4 {n}(FragIn in [[stage_in]], half4 d [[color(0)]])\n{{\n    return d * 0.5h;\n}}\n",
         []),
        ("color_coverage_mask",
         "fragment half4 {n}(FragIn in [[stage_in]], uint m [[sample_mask]], imageblock<IB, imageblock_layout_implicit> ib)\n"
         "{{\n    return half4(ib.get_num_colors(ushort2(in.pos.xy), m));\n}}\n",
         ["get_imageblock_num_colors", "get_imageblock_color_coverage_mask"]),
        ("imageblock_slice_write",
         "kernel void {n}(imageblock<IB, imageblock_layout_implicit> ib, texture2d<half, access::write> t [[texture(0)]], ushort2 gid [[thread_position_in_grid]])\n"
         "{{\n    t.write(ib.slice(ib->a), gid);\n}}\n",
         []),
    ]
    for label, tmpl, reaches in fns:
        g.add({"fn": label}, tmpl.format(n=f"v{len(g.variants)}"), reaches)
    return [g]


IMAGEBLOCK_MEMBERS = ["half", "half2", "half4", "float", "float2", "float4", "int", "int2", "int4",
                     "uint", "uint2", "uint4", "short", "short2", "short4", "ushort", "ushort2", "ushort4",
                     "char4", "uchar4", "r8unorm<half>", "rg8unorm<half2>", "rgba8unorm<half4>",
                     "srgba8unorm<half4>", "rgba8snorm<half4>", "rgba8unorm<float4>", "r16unorm<float>",
                     "rg16snorm<float2>", "rgba16unorm<half4>", "rgb10a2<half4>", "rg11b10f<half3>",
                     "rgb9e5<half3>"]
SLICE_TEXTURES = {  # texture -> write arguments after the slice, by coordinate type
    "texture1d": lambda c: f"{c}(lid.x)", "texture1d_array": lambda c: f"{c}(lid.x), {c}(0)",
    "texture2d": lambda c: f"{c}2(lid)", "texture2d_array": lambda c: f"{c}2(lid), {c}(0)",
    "texture3d": lambda c: f"{c}3(lid.x, lid.y, 0)", "texturecube": lambda c: f"{c}2(lid), {c}(0)",
    "texturecube_array": lambda c: f"{c}2(lid), {c}(0), {c}(0)",
}


def imageblock_sweep_groups():
    """Implicit and explicit imageblocks over every member type: reads, writes, data-rate and
    coverage-mask forms, coverage queries, and slice writes to every texture type."""
    groups = []
    g = Group("imageblock_access", "imageblock", min_version="2.0")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    tile = ("kernel void {n}(imageblock<{ib}{layout}> ib, ushort2 lid [[thread_position_in_threadgroup]],"
            " device uchar *r [[buffer(30)]])\n{{\n{body}\n}}\n")
    for m in IMAGEBLOCK_MEMBERS:
        forms = [
            ("implicit_read_write", "imageblock_layout_implicit",
             "    I v = ib.read(lid);\n    ib.write(v, lid);", ["imageblock_implicit_read", "imageblock_implicit_write"]),
            ("implicit_rate", "imageblock_layout_implicit",
             "    I v = ib.read(lid, 0, imageblock_data_rate::color);\n    ib.write(v, lid, 1, imageblock_data_rate::sample);",
             ["imageblock_implicit_read", "imageblock_implicit_write"]),
            ("implicit_mask_write", "imageblock_layout_implicit",
             "    I v = ib.read(lid);\n    ib.write(v, lid, ushort(3));", ["imageblock_implicit_mask_write"]),
            ("coverage_queries", "imageblock_layout_implicit",
             "    sink(r, ushort(ib.get_num_colors(lid) + ib.get_color_coverage_mask(lid, 0)));\n"
             "    sink(r + 8, ushort(ib.get_width() + ib.get_height() + ib.get_num_samples()));",
             ["get_imageblock_num_colors", "get_imageblock_color_coverage_mask", "get_imageblock_width",
              "get_imageblock_height", "get_imageblock_samples"]),
            ("explicit_data", "imageblock_layout_explicit",
             "    threadgroup_imageblock E *p = ib.data(lid);\n    threadgroup_imageblock E *q = ib.data(lid, 0, imageblock_data_rate::sample);\n"
             "    q->a = p->a;", ["imageblock_explicit_data"]),
            ("explicit_mask_write", "imageblock_layout_explicit",
             "    E v = *ib.data(lid);\n    ib.write(v, lid, ushort(3));", ["imageblock_explicit_mask_write"]),
        ]
        for label, layout, body, reach in forms:
            name = f"v{len(g.variants)}"
            struct = (f"struct I_{name} {{ {m} a [[color(0)]]; }};\n" if "implicit" in layout
                      else f"struct E_{name} {{ {m} a; }};\n")
            ib = f"I_{name}" if "implicit" in layout else f"E_{name}"
            text = struct + tile.format(n=name, ib=ib, layout=", " + layout if "implicit" in layout else "",
                                        body=body.replace("I v", f"I_{name} v").replace("E v", f"E_{name} v")
                                        .replace("E *", f"E_{name} *"))
            g.add({"member": m, "form": label}, text, reach)
    groups.append(g)
    for tex, coord in SLICE_TEXTURES.items():
        gs = Group(f"imageblock_slice_{tex}", "imageblock", min_version="2.0")
        gs.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
        for m in IMAGEBLOCK_MEMBERS:
            for comp in ("half", "float", "short", "ushort", "int", "uint"):
                for ct in ("uint", "ushort"):
                    for layout in ("implicit", "explicit"):
                        name = f"v{len(gs.variants)}"
                        if layout == "implicit":
                            struct = f"struct I_{name} {{ {m} a [[color(0)]]; }};\n"
                            param = f"imageblock<I_{name}, imageblock_layout_implicit> ib"
                            slice_expr = f"ib.slice<{m}>(0)"
                        else:
                            struct = f"struct E_{name} {{ {m} a; }};\n"
                            param = f"imageblock<E_{name}> ib"
                            slice_expr = "ib.slice(ib.data(lid)->a)"
                        text = (struct + f"kernel void {name}({param}, {tex}<{comp}, access::write> t [[texture(0)]],"
                                f" ushort2 lid [[thread_position_in_threadgroup]])\n{{\n"
                                f"    t.write({slice_expr}, {coord(ct)});\n}}\n")
                        gs.add({"member": m, "texture": tex, "component": comp, "coord": ct, "layout": layout},
                               text, ["imageblock_implicit_data"] if layout == "implicit" else [])
        groups.append(gs)
    return groups


# ---------------------------------------------------------------------------------------------
# Command buffers, function tables, ray tracing, mesh, tessellation, function constants
# ---------------------------------------------------------------------------------------------

def indirect_groups():
    g = Group("indirect_commands", "command_buffer", min_version="2.1")
    g.preamble = "struct ICBArgs { command_buffer cmds [[id(0)]]; };\n"
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    render = [
        ("set_vertex_buffer", "cmd.set_vertex_buffer(r, 0);"), ("set_fragment_buffer", "cmd.set_fragment_buffer(r, 0);"),
        ("draw_primitives", "cmd.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);"),
        ("draw_indexed_primitives", "cmd.draw_indexed_primitives(primitive_type::triangle, 3, (device ushort *)r, 1, 0, 0);"),
        ("draw_patches", "cmd.draw_patches(3, 0, 1, (const device uint *)r, 1, 0, (const device MTLTriangleTessellationFactorsHalf *)r);"),
        ("draw_indexed_patches", "cmd.draw_indexed_patches(3, 0, 1, (const device uint *)r, (const device void *)r, 1, 0, (const device MTLTriangleTessellationFactorsHalf *)r);"),
        ("reset", "cmd.reset();"), ("set_render_pipeline_state", "cmd.set_render_pipeline_state(st->ps);"),
        ("set_object_buffer", "cmd.set_object_buffer(r, 0);"), ("set_mesh_buffer", "cmd.set_mesh_buffer(r, 0);"),
        ("set_object_threadgroup_memory_length", "cmd.set_object_threadgroup_memory_length(16, 0);"),
        ("draw_mesh_threadgroups", "cmd.draw_mesh_threadgroups(uint3(1), uint3(1), uint3(1));"),
        ("draw_mesh_threads", "cmd.draw_mesh_threads(uint3(1), uint3(1), uint3(1));"),
        ("set_barrier", "cmd.set_barrier();"), ("clear_barrier", "cmd.clear_barrier();"),
    ]
    def icb_variant(kind, label, stmt):
        # Pipeline-state handles exist from a later language version than command buffers, so
        # only the variants that set a pipeline state declare them.
        name = f"v{len(g.variants)}"
        extra = [("device ICBArgs", "*args", "[[buffer(10)]]")]
        prefix = ""
        if "pipeline_state" in label:
            prefix = (f"struct ICBState_{name} {{ render_pipeline_state ps [[id(0)]]; "
                      f"compute_pipeline_state cs [[id(1)]]; }};\n")
            extra.append((f"device ICBState_{name}", "*st", "[[buffer(11)]]"))
        body = f"    {kind}_command cmd(args->cmds, i);\n    {stmt}"
        extra.append(("constant uchar", "*c", "[[buffer(12)]]"))
        reach = label.replace("set_render_pipeline_state", "set_pipeline_state").replace(
            "set_compute_pipeline_state", "set_pipeline_state")
        g.add({"command": kind, "fn": label}, prefix + kernel_fn(name, body, ["float"], extra),
              [f"{reach.split(':')[0]}_{kind}_command"])

    # Every pointer argument in each address space and element type the encoder accepts.
    compute_pointers = []
    for space, ptr in (("device", "r"), ("constant", "c")):
        for t in ("ushort", "uint"):
            render.append((f"draw_indexed_primitives:{space}:{t}",
                           f"cmd.draw_indexed_primitives(primitive_type::triangle, 3, ({space} {t} *){ptr}, 1, 0, 0);"))
        for factors in ("MTLTriangleTessellationFactorsHalf", "MTLQuadTessellationFactorsHalf"):
            for space2, ptr2 in (("device", "r"), ("constant", "c")):
                render.append((f"draw_patches:{space}:{space2}:{factors}",
                               f"cmd.draw_patches(3, 0, 1, (const {space} uint *){ptr}, 1, 0, (const {space2} {factors} *){ptr2});"))
                render.append((f"draw_indexed_patches:{space}:{space2}:{factors}",
                               f"cmd.draw_indexed_patches(3, 0, 1, (const {space} uint *){ptr}, (const {space} void *){ptr}, 1, 0,"
                               f" (const {space2} {factors} *){ptr2});"))
        for fn in ("set_vertex_buffer", "set_fragment_buffer", "set_object_buffer", "set_mesh_buffer"):
            for t in ("uchar", "float", "float4"):
                render.append((f"{fn}:{space}:{t}", f"cmd.{fn}(({space} {t} *){ptr}, 0);"))
                render.append((f"{fn}:{space}:{t}:stride", f"cmd.{fn}(({space} {t} *){ptr}, 16, 0);"))
        for t in ("uchar", "float", "float4"):
            compute_pointers.append((f"set_kernel_buffer:{space}:{t}", f"cmd.set_kernel_buffer(({space} {t} *){ptr}, 0);"))
            compute_pointers.append((f"set_kernel_buffer:{space}:{t}:stride",
                                     f"cmd.set_kernel_buffer(({space} {t} *){ptr}, 16, 0);"))
    for label, stmt in render:
        icb_variant("render", label, stmt)
    compute = [
        ("set_kernel_buffer", "cmd.set_kernel_buffer(r, 0);"),
        ("concurrent_dispatch_threadgroups", "cmd.concurrent_dispatch_threadgroups(uint3(1), uint3(1));"),
        ("concurrent_dispatch_threads", "cmd.concurrent_dispatch_threads(uint3(1), uint3(1));"),
        ("set_threadgroup_memory_length", "cmd.set_threadgroup_memory_length(16, 0);"),
        ("set_stage_in_region", "cmd.set_stage_in_region(uint3(0), uint3(1));"),
        ("set_imageblock_size", "cmd.set_imageblock_size(ushort2(8, 8));"),
        ("set_compute_pipeline_state", "cmd.set_compute_pipeline_state(st->cs);"),
        ("reset", "cmd.reset();"), ("set_barrier", "cmd.set_barrier();"), ("clear_barrier", "cmd.clear_barrier();"),
    ]
    for label, stmt in compute + compute_pointers:
        icb_variant("compute", label, stmt)
    g.add({"fn": "copy_command"},
          kernel_fn(f"v{len(g.variants)}", "    render_command a(args->cmds, i), b(args->cmds, i + 1);\n    b.copy_command(a);\n"
                    "    compute_command c(args->cmds, i), d(args->cmds, i + 1);\n    d.copy_command(c);",
                    ["float"], [("device ICBArgs", "*args", "[[buffer(10)]]")]),
          ["copy_render_command", "copy_compute_command"])
    g.add({"fn": "size"}, kernel_fn(f"v{len(g.variants)}", k_sink("args->cmds.size()"), ["float"],
                                    [("device ICBArgs", "*args", "[[buffer(10)]]")]), ["get_size_command_buffer"])
    return [g]


def function_table_groups():
    g = Group("function_tables", "visible_function_table", min_version="2.3")
    g.preamble = ("[[visible]] int vf(int x) { return x + 1; }\n"
                  "[[visible]] float vg(float x, device float *p) { return x + p[0]; }\n")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["int"]))
    g.add({"fn": "call"}, kernel_fn("v1", k_sink("table[a0[i]](a0[i])"), ["int"],
                                    [("visible_function_table<int(int)>", "table", "[[buffer(10)]]")]),
          ["get_function_pointer_visible_function_table"])
    g.add({"fn": "size"}, kernel_fn("v2", k_sink("table.size()"), ["int"],
                                    [("visible_function_table<int(int)>", "table", "[[buffer(10)]]")]),
          ["get_size_visible_function_table"])
    g.add({"fn": "null"}, kernel_fn("v3", k_sink("is_null_visible_function_table(table)"), ["int"],
                                    [("visible_function_table<int(int)>", "table", "[[buffer(10)]]")]),
          ["is_null_visible_function_table"])
    g.add({"fn": "pointer"}, kernel_fn("v4", "    int (*fp)(int) = vf;\n" + k_sink("fp(a0[i])"), ["int"]), [])
    g.add({"fn": "null_table"}, kernel_fn("v5", k_sink("is_null_visible_function_table(visible_function_table<int(int)>())"), ["int"]),
          ["get_null_visible_function_table"])
    g.add({"fn": "stitchable"}, "[[stitchable]] float v6(float a, float b) { return a * b + 1.0f; }\n", [])
    g.add({"fn": "external"}, "extern float lib_fn(float x);\n" + kernel_fn("v7", k_sink("lib_fn(a0[i])"), ["float"]), [])
    return [g]


RT_PREAMBLE = "using namespace metal::raytracing;\n"
INTERSECTOR_TAGS = ["instancing", "triangle_data", "curve_data", "world_space_data", "primitive_motion",
                    "instance_motion", "extended_limits", "max_levels<3>"]
QUERY_TAGS = ["instancing", "triangle_data", "curve_data", "max_levels<3>", "extended_limits"]
ACCEL_TAGS = ("instancing", "primitive_motion", "instance_motion")
RT_QUERIES = [
    "next()", "abort()", "reset(rr, as, params)", "get_candidate_intersection_type()",
    "get_committed_intersection_type()", "get_candidate_triangle_distance()", "get_committed_distance()",
    "get_candidate_primitive_id()", "get_committed_primitive_id()", "get_candidate_geometry_id()",
    "get_committed_geometry_id()", "get_candidate_triangle_barycentric_coord()",
    "get_committed_triangle_barycentric_coord()", "is_candidate_triangle_front_facing()",
    "is_committed_triangle_front_facing()", "get_world_space_ray_origin()", "get_world_space_ray_direction()",
    "get_ray_min_distance()", "commit_triangle_intersection()", "commit_bounding_box_intersection(1.0f)",
    "is_candidate_non_opaque_bounding_box()", "get_candidate_ray_origin()", "get_candidate_ray_direction()",
    "get_committed_ray_origin()", "get_committed_ray_direction()", "get_candidate_instance_id()",
    "get_committed_instance_id()", "get_candidate_user_instance_id()", "get_committed_user_instance_id()",
    "get_candidate_instance_id(0)", "get_committed_instance_id(0)", "get_candidate_user_instance_id(0)",
    "get_committed_user_instance_id(0)", "get_candidate_object_to_world_transform()",
    "get_committed_object_to_world_transform()", "get_candidate_world_to_object_transform()",
    "get_committed_world_to_object_transform()", "get_candidate_primitive_data()",
    "get_committed_primitive_data()", "get_candidate_instance_count()", "get_committed_instance_count()",
    "get_candidate_curve_parameter()", "get_committed_curve_parameter()", "get_candidate_curve_distance()",
    "commit_curve_intersection()", "get_intersection_params()",
]
RT_STATEMENTS = ("next", "abort", "reset", "commit_triangle_intersection", "commit_bounding_box_intersection",
                 "commit_curve_intersection", "get_intersection_params")


def powerset(items):
    out = [[]]
    for item in items:
        out += [subset + [item] for subset in out]
    return out


def accel_types(tags, legacy):
    accel = [t for t in tags if t in ACCEL_TAGS]
    types = [f"acceleration_structure<{', '.join(accel)}>"]
    if legacy and not set(accel) - {"instancing"}:
        types.append("instance_acceleration_structure" if accel else "primitive_acceleration_structure")
    return types


def raytracing_groups():
    """Every tag combination of intersectors and intersection queries, with every call form;
    the compiler decides which combinations exist."""
    groups = []
    g = Group("raytracing_intersector", "raytracing", min_version="2.3", preamble=RT_PREAMBLE)
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    for tags in powerset(INTERSECTOR_TAGS):
        tl = ", ".join(tags)
        for accel in accel_types(tags, legacy=True):
            for mask in (False, True):
                for time in (False, True):
                    for table in (False, True):
                        args = "rr, as" + (", 0xffu" if mask else "") + (", 0.5f" if time else "") + (", ft" if table else "")
                        body = (f"    ray rr(float3(a0[i]), float3(0, 0, 1), 0.0f, 100.0f);\n"
                                f"    intersector<{tl}> it;\n"
                                f"    auto res = it.intersect({args});\n"
                                "    sink(r + i * 64, res.distance);\n    sink(r + i * 64 + 8, uint(res.type));\n")
                        extra = [(accel, "as", "[[buffer(10)]]")]
                        if table:
                            extra.append((f"intersection_function_table<{tl}>", "ft", "[[buffer(11)]]"))
                        g.add({"tags": tags, "accel": accel, "mask": mask, "time": time, "table": table},
                              kernel_fn(f"v{len(g.variants)}", body, ["float"], extra), ["intersect"])
    groups.append(g)
    gq = Group("raytracing_query", "raytracing", min_version="2.4", preamble=RT_PREAMBLE)
    gq.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    for tags in powerset(QUERY_TAGS):
        tl = ", ".join(tags)
        for accel in accel_types(tags, legacy=False):
            for mask in (False, True):
                ctor = "rr, as" + (", 0xffu" if mask else "") + ", params"
                for qexpr in RT_QUERIES:
                    method = qexpr.split("(")[0]
                    call = f"q.{qexpr}"
                    if method == "reset":
                        call = f"q.reset({ctor})"
                    if method == "get_intersection_params":
                        call = f"q.reset({ctor.replace('params', 'q.get_intersection_params()')})"
                    body = (f"    ray rr(float3(a0[i]), float3(0, 0, 1), 0.0f, 100.0f);\n    intersection_params params;\n"
                            f"    intersection_query<{tl}> q({ctor});\n"
                            + (f"    {call};" if method in RT_STATEMENTS else k_sink(call)))
                    gq.add({"tags": tags, "accel": accel, "mask": mask, "fn": qexpr},
                           kernel_fn(f"v{len(gq.variants)}", body, ["float"], [(accel, "as", "[[buffer(10)]]")]),
                           [method + "_intersection_query"])
    groups.append(gq)
    ga = Group("raytracing_objects", "raytracing", min_version="2.3", preamble=RT_PREAMBLE)
    ga.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    objs = [
        ("is_null_instance_acceleration_structure", k_sink("is_null_instance_acceleration_structure(ias)")),
        ("is_null_primitive_acceleration_structure", k_sink("is_null_primitive_acceleration_structure(pas)")),
        ("null_acceleration_structures", k_sink("is_null_instance_acceleration_structure(instance_acceleration_structure())")),
        ("is_null_intersection_function_table", k_sink("is_null_intersection_function_table(ft)")),
        ("ift_set_buffer", "    ft.set_buffer(r, 0);"),
        ("ift_set_visible_function_table", "    ft.set_visible_function_table(vt, 0);"),
        ("ift_size", k_sink("ft.size()")),
        ("ias_get_instance_count", k_sink("ias.get_instance_count()")),
        ("ias_get_primitive_acceleration_structure", k_sink("is_null_primitive_acceleration_structure(ias.get_primitive_acceleration_structure(0))")),
        ("resource_id", k_sink("pas.get_resource_id()")),
        ("get_acceleration_structure", k_sink("is_null_acceleration_structure(gas.get_acceleration_structure<>(0))")),
        ("get_acceleration_structure_instancing",
         k_sink("is_null_acceleration_structure(gas.get_acceleration_structure<instancing>(0))")),
        ("generic_get_instance_count", k_sink("gas.get_instance_count()")),
        ("is_null_acceleration_structure", k_sink("is_null_acceleration_structure(gas)")),
        ("ift_get_buffer", k_sink("ft.get_buffer<device float *>(0)[0]")),
        ("ift_get_visible_function_table", k_sink("is_null_visible_function_table(ft.get_visible_function_table<void(int)>(0))")),
        ("resource_id_instance", "    MTLResourceID rid = gas;\n" + k_sink("rid")),
        ("resource_id_primitive", "    MTLResourceID rid = pgas;\n" + k_sink("rid")),
    ]
    for label, body in objs:
        ga.add({"fn": label}, kernel_fn(f"v{len(ga.variants)}", body, ["float"],
                                        [("acceleration_structure<instancing>", "gas", "[[buffer(14)]]"),
                                         ("acceleration_structure<>", "pgas", "[[buffer(15)]]"),
                                         ("instance_acceleration_structure", "ias", "[[buffer(10)]]"),
                                         ("primitive_acceleration_structure", "pas", "[[buffer(11)]]"),
                                         ("intersection_function_table<triangle_data>", "ft", "[[buffer(12)]]"),
                                         ("visible_function_table<void(int)>", "vt", "[[buffer(13)]]")]),
               {"ift_set_buffer": ["set_buffer_intersection_function_table"],
                "ift_set_visible_function_table": ["set_visible_function_table_intersection_function_table"],
                "ift_size": ["get_size_intersection_function_table"],
                "ias_get_instance_count": ["get_instance_count_instance_acceleration_structure"],
                "generic_get_instance_count": ["get_instance_count_instance_acceleration_structure"],
                "ias_get_primitive_acceleration_structure": ["get_primitive_acceleration_structure_instance_acceleration_structure"],
                "get_acceleration_structure": ["get_primitive_acceleration_structure_instance_acceleration_structure"],
                "get_acceleration_structure_instancing": ["get_instance_acceleration_structure_instance_acceleration_structure"],
                "ift_get_buffer": ["get_buffer_intersection_function_table"],
                "ift_get_visible_function_table": ["get_visible_function_table_intersection_function_table"],
                "resource_id_instance": ["get_resource_id_instance_acceleration_structure"],
                "resource_id_primitive": ["get_resource_id_primitive_acceleration_structure"],
                }.get(label, [label]))
    groups.append(ga)
    gf = Group("intersection_functions", "raytracing", min_version="2.3", preamble=RT_PREAMBLE)
    gf.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    gf.add({"fn": "triangle"},
           "[[intersection(triangle, triangle_data, instancing)]] bool v1(float2 b [[barycentric_coord]], uint pid [[primitive_id]],"
           " uint gid [[geometry_id]], uint iid [[instance_id]], float t [[distance]], bool ff [[front_facing]])\n"
           "{\n    return b.x > 0.5f && pid != gid && iid != 0 && t > 0.0f && ff;\n}\n", [])
    gf.add({"fn": "bounding_box"},
           "struct BoxResult { bool accept [[accept_intersection]]; float distance [[distance]]; };\n"
           "[[intersection(bounding_box, instancing, world_space_data)]] BoxResult v2(float3 o [[origin]], float3 d [[direction]],"
           " float tmin [[min_distance]], float tmax [[max_distance]], float4x3 m [[object_to_world_transform]])\n"
           "{\n    return { o.x < d.x, tmin + tmax + m[0][0] };\n}\n", [])
    groups.append(gf)
    return groups


def mesh_groups():
    g = Group("mesh_shaders", "mesh", min_version="3.0")
    g.preamble = ("struct MV { float4 pos [[position]]; float2 uv; };\n"
                  "struct MP { float4 c [[flat]]; };\n"
                  "using M = metal::mesh<MV, MP, 64, 126, topology::triangle>;\n"
                  "struct Payload { uint n; };\n")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    g.add({"fn": "mesh"},
          "[[mesh]] void v1(M m, uint t [[thread_index_in_threadgroup]], const object_data Payload &p [[payload]])\n"
          "{\n    m.set_primitive_count(p.n);\n    MV v; v.pos = float4(t); v.uv = float2(t);\n    m.set_vertex(t, v);\n"
          "    m.set_index(t, uchar(t));\n    m.set_indices(t, uchar4(t));\n    MP q; q.c = float4(1);\n    m.set_primitive(t, q);\n}\n",
          ["set_primitive_count_mesh", "set_vertex_mesh", "set_index_mesh", "set_indices_mesh", "set_primitive_mesh"])
    g.add({"fn": "object"},
          "[[object]] void v2(object_data Payload &p [[payload]], mesh_grid_properties mgp, uint t [[thread_index_in_threadgroup]])\n"
          "{\n    p.n = t;\n    mgp.set_threadgroups_per_grid(uint3(1, 1, 1));\n}\n",
          ["set_threadgroups_per_grid_mesh_grid_properties"])
    return [g]


def tessellation_groups():
    g = Group("tessellation", "tessellation", min_version="1.2")
    g.preamble = ("struct CP { float4 p [[attribute(0)]]; };\n"
                  "struct PIn { patch_control_point<CP> cp; float4 f [[attribute(1)]]; };\n"
                  "struct POut { float4 pos [[position]]; };\n")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    for patch, n, coord in (("triangle", 3, "float3"), ("quad", 4, "float2")):
        for ptype in ("", f", {n}"):
            name = f"v{len(g.variants)}"
            g.add({"patch": patch, "count": bool(ptype)},
                  f"[[patch({patch}{ptype})]] vertex POut {name}(PIn in [[stage_in]], {coord} pc [[position_in_patch]], uint pid [[patch_id]])\n"
                  "{\n    POut o;\n    o.pos = in.cp[0].p + in.f + float4(in.cp.size()) + float4(pc.x) + float4(pid);\n    return o;\n}\n",
                  ["get_control_point", "get_num_patch_control_points"])
    return [g]


def function_constant_groups():
    g = Group("function_constants", "function_constant", min_version="1.2")
    g.preamble = ("constant bool fc_b [[function_constant(0)]];\n"
                  "constant int fc_i [[function_constant(1)]];\n"
                  "constant float fc_f [[function_constant(2)]];\n"
                  "constant half4 fc_h [[function_constant(3)]];\n"
                  "constant bool fc_derived = !fc_b;\n"
                  "constant int fc_i_or = is_function_constant_defined(fc_i) ? fc_i : 7;\n")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["float"]))
    g.add({"fn": "is_defined"}, kernel_fn("v1", k_sink("is_function_constant_defined(fc_f)"), ["float"]),
          ["is_function_constant_defined"])
    g.add({"fn": "values"}, kernel_fn("v2", "    if (fc_b) sink(r + i * 64, fc_i_or + int(fc_f) + int(fc_h.x));\n"
                                      "    if (fc_derived) sink(r + i * 64, a0[i]);", ["float"]), [])
    g.add({"fn": "argument"}, kernel_fn("v3", k_sink("a0[i]"), ["float"],
                                        [("device float", "*opt", "[[buffer(5), function_constant(fc_b)]]")]), [])
    # Every scalar and vector type as a function constant: as a value, through
    # is_function_constant_defined, and as the predicate of an optional argument.
    for s_, w in all_types():
        t = vt(s_, w)
        for use in ("value", "defined", "predicate"):
            name = f"v{len(g.variants)}"
            decl = f"constant {t} fc_{name} [[function_constant({len(g.variants) + 10})]];\n"
            if use == "value":
                text = decl + kernel_fn(name, k_sink(f"fc_{name}"), ["float"])
            elif use == "defined":
                text = decl + kernel_fn(name, k_sink(f"is_function_constant_defined(fc_{name})"), ["float"])
            else:
                text = decl + kernel_fn(name, k_sink("opt[i]"), ["float"],
                                        [("device float", "*opt", f"[[buffer(5), function_constant(fc_{name})]]")])
            g.add({"type": t, "use": use}, text, ["is_function_constant_defined"] if use == "defined" else [])
    return [g]


def uniform_groups():
    g = Group("uniform", "uniform", min_version="2.4")
    g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["int"]))
    for s, w in all_types():
        t = vt(s, w)
        g.add({"type": t}, kernel_fn(f"v{len(g.variants)}", f"    uniform<{t}> u = make_uniform(a0[0]);\n" + k_sink(f"{t}(u)"), [t]),
              ["is_uniform"])
    return [g]


FUZZ_SEED = 0x41495257  # fixed: the fuzz corpus is reproducible
FUZZ_GROUPS = 8
FUZZ_PER_GROUP = 50


def fuzz_groups():
    """Random kernels in the exact subset, from the grammar in fuzz/grammar.py."""
    import random
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fuzz"))
    import grammar
    rng = random.Random(FUZZ_SEED)
    groups = []
    for n in range(FUZZ_GROUPS):
        g = Group(f"fuzz_{n:03d}", "fuzz")
        g.add({"harness": True}, kernel_fn("v0", "    sink(r + i * 64, a0[i]);", ["uint"]))
        for _ in range(FUZZ_PER_GROUP):
            g.add({"fuzz_seed": FUZZ_SEED, "program": len(g.variants)}, grammar.program(rng, f"v{len(g.variants)}"))
        groups.append(g)
    return groups


def builder():
    groups = []
    for fn in (math_groups, conversion_groups, pack_groups, atomic_groups, sync_groups,
               graphics_groups, texture_groups, imageblock_groups, imageblock_sweep_groups, indirect_groups,
               function_table_groups, raytracing_groups, mesh_groups, tessellation_groups,
               function_constant_groups, uniform_groups, fuzz_groups):
        groups += fn()
    names = [g.name for g in groups]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate group names")
    return groups


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: generate.py OUT_DIR")
    out = sys.argv[1]
    os.makedirs(os.path.join(out, "msl"), exist_ok=True)
    manifest = {"versions": VERSIONS, "groups": []}
    for g in builder():
        path = os.path.join("msl", g.name + ".metal")
        with open(os.path.join(out, path), "w", encoding="utf-8") as fh:
            fh.write(PRELUDE)
            if g.preamble:
                fh.write("\n" + g.preamble)
            for index, v in enumerate(g.variants):
                fh.write(f"\n// variant {index}: {json.dumps(v['params'], sort_keys=True)}\n")
                fh.write(v["function"])
        manifest["groups"].append({
            "name": g.name, "chapter": g.chapter, "file": path, "min_version": g.min_version,
            "options": g.options,
            "variants": [{"params": v["params"], "reaches": v["reaches"]} for v in g.variants],
        })
    # Hand-written sources (stage interfaces, address spaces, argument buffers) use the same
    # variant markers and are compiled the same way.
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msl")
    for fname in sorted(os.listdir(here)):
        if not fname.endswith(".metal"):
            continue
        with open(os.path.join(here, fname), encoding="utf-8") as fh:
            text = fh.read()
        variants = []
        profiles = [""]
        for line in text.splitlines():
            if line.startswith("// profiles: "):
                profiles = [p_.strip() for p_ in line[len("// profiles: "):].split(";")]
                profiles = ["" if p_ == "default" else p_ for p_ in profiles]
            if line.startswith("// variant "):
                index, params = line[len("// variant "):].split(": ", 1)
                if int(index) != len(variants):
                    raise SystemExit(f"{fname}: variant {index} out of order")
                variants.append({"params": json.loads(params), "reaches": []})
        path = os.path.join("msl", "msl_" + fname)
        with open(os.path.join(out, path), "w", encoding="utf-8") as fh:
            fh.write(text)
        for profile in profiles:
            # A "// profiles:" line compiles the file once per option set (default first).
            suffix = "" if not profile else "__" + profile.lstrip("-").replace(" ", "_").replace("=", "_")
            manifest["groups"].append({
                "name": "msl_" + fname[:-len(".metal")] + suffix, "chapter": "interface", "file": path,
                "min_version": "1.1", "options": profile.split() if profile else [], "variants": variants,
            })
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    total = sum(len(g["variants"]) for g in manifest["groups"])
    print(f"groups={len(manifest['groups'])} variants={total}")


if __name__ == "__main__":
    main()
