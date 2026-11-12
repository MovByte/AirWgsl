"""Decompose AIR function names into family, dimension, type signature and modifiers.

An AIR intrinsic name is ``air.`` followed by dot-separated components. The first component
names the operation, possibly with the object it acts on folded in (``sample_texture_2d``,
``get_width_depth_cube_array``, ``next_intersection_query``); the rest are type tokens
(``v4f32``, ``i16``, ``u``, ``rte``...) and, for some operations, qualifiers (address space,
pixel format, ray-tracing tags). The decomposition is:

``family``
    The operation with ``fast_`` removed (the plan's grouping: ``sin`` for ``air.fast_sin.f32``
    and ``air.sin.f32``), or the leading components for namespaced operations (``wg.barrier``,
    ``atomic.add``).
``dimension``
    The object or qualifier the operation is specialized on: texture kind and dimension
    (``texture_2d_array``, ``depth_cube``), address space for atomics, pixel format for
    pack/unpack, ray-tracing tags, query and command object kinds. Empty when there is none.
``signature``
    The type tokens, dot-joined (``v4f32``, ``u.i32``, ``f.v4f16.f.v4f32``).
``modifiers``
    ``fast`` for ``fast_`` names; ``grad`` for the ``_grad`` texture forms.

The rules are written against the names found in the system corpus, the generated corpus and
the target compiler's strings; ``coverage.py`` lists every name the rules leave unusual so that a
new shape is seen rather than misfiled.
"""

import re

TYPE_TOKEN = re.compile(r"^(?:v\d+)?(?:i1|i8|i16|i32|i64|i128|f16|f32|f64|bf16)$")
POINTER_TOKEN = re.compile(r"^p\d+(?:i8|i16|i32|i64|f16|f32|v\d+f32|s\.[\w.]+)?$")
MARK_TOKEN = {"s", "u", "f", "rte", "rtz", "rtp", "rtn", "sat"}

TEXTURE_KINDS = r"(?:texture|depth)_(?:1d_array|1d|2d_ms_array|2d_ms|2d_array|2d|3d|cube_array|cube|buffer_1d)"
TEXTURE_HEAD = re.compile(rf"^(?P<op>.+?)_(?P<obj>{TEXTURE_KINDS})(?P<grad>_grad)?$")
OBJECT_SUFFIXES = (
    "intersection_query", "visible_function_table", "intersection_function_table",
    "instance_acceleration_structure", "primitive_acceleration_structure", "acceleration_structure",
    "render_command", "compute_command", "command_buffer", "mesh_grid_properties", "mesh",
    "render_pipeline_state", "compute_pipeline_state", "simdgroup_event", "wg_event", "sampler",
)
OBJECT_HEAD = re.compile(rf"^(?P<op>.+?)_(?P<obj>{'|'.join(OBJECT_SUFFIXES)})$")
NAMESPACED = {"wg", "simdgroup", "mem", "quadgroup"}


def is_type_token(token):
    return bool(TYPE_TOKEN.match(token) or POINTER_TOKEN.match(token) or token in MARK_TOKEN)


def decompose(name):
    """Returns (family, dimension, signature, modifiers) for an ``air.`` name, else None."""
    if not name.startswith("air."):
        return None
    parts = name[4:].split(".")
    head, tail = parts[0], parts[1:]
    modifiers = []
    if head.startswith("fast_"):
        head = head[5:]
        modifiers.append("fast")
    dimension = []
    if head == "atomic" and tail:
        # air.atomic.<space>.<op...>.<types>
        space = tail[0]
        ops = []
        rest = tail[1:]
        while rest and not is_type_token(rest[0]):
            ops.append(rest.pop(0))
        family = "atomic." + ".".join(ops) if ops else "atomic"
        dimension.append(space)
        tail = rest
    elif head in NAMESPACED and tail and not is_type_token(tail[0]):
        family = f"{head}.{tail[0]}"
        tail = tail[1:]
    elif head in ("pack", "unpack") and tail and not is_type_token(tail[0]):
        family = head
        dimension.append(tail[0])
        tail = tail[1:]
    else:
        m = TEXTURE_HEAD.match(head)
        if m:
            family = m.group("op")
            dimension.append(m.group("obj"))
            if m.group("grad"):
                modifiers.append("grad")
        else:
            m = OBJECT_HEAD.match(head)
            if m:
                family = m.group("op")
                dimension.append(m.group("obj"))
            else:
                family = head
    types = []
    for token in tail:
        if is_type_token(token):
            types.append(token)
        else:
            dimension.append(token)
    return family, ".".join(dimension), ".".join(types), tuple(modifiers)


def variant_key(name):
    """Hashable key of the (dimension, signature, modifiers) variant of a name."""
    d = decompose(name)
    if d is None:
        return None
    return d[1], d[2], d[3]
