"""Minimal deterministic TOML writer for the generated tables."""

import math


def _str(s):
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or o == 0x7F:
            out.append("\\u%04x" % o)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _key(k):
    if k and all(c.isalnum() or c in "-_" for c in k):
        return k
    return _str(k)


def value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("non-finite float in table")
        return repr(v)
    if isinstance(v, str):
        return _str(v)
    if isinstance(v, (list, tuple)):
        if not v:
            return "[]"
        if any(isinstance(x, dict) for x in v):
            return "[\n" + "".join("  " + value(x) + ",\n" for x in v) + "]"
        return "[" + ", ".join(value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join("%s = %s" % (_key(k), value(x)) for k, x in v.items()) + " }"
    raise TypeError("unsupported value %r" % (v,))


def table(name, d):
    lines = ["[%s]" % name]
    for k, v in d.items():
        lines.append("%s = %s" % (_key(k), value(v)))
    return "\n".join(lines) + "\n"


def array_item(name, d):
    lines = ["[[%s]]" % name]
    for k, v in d.items():
        if v is None:
            continue
        lines.append("%s = %s" % (_key(k), value(v)))
    return "\n".join(lines) + "\n"
