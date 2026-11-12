"""Grammar of random MSL kernels in the exact subset.

The exact subset is the part of MSL whose results both MSL and WGSL define bit for bit, so a
program's output can be compared exactly between real Metal, the IR interpreter and WebGPU:
integer arithmetic on 8-, 16-, 32- and 64-bit types, bit operations, comparisons, selects,
integer builtins with fully defined results, bit casts, thread-local arrays, bounded loops,
branches and switches, and device-buffer loads and stores at in-bounds indices.

Every production avoids behavior the C++ rules of MSL leave undefined or implementation-defined:

- addition, subtraction and multiplication run on unsigned types (signed values are bit cast to
  the unsigned type of the same width and back), so overflow wraps;
- shifts take an unsigned left operand and a count masked to the operand's width;
- division and remainder take an unsigned divisor forced odd and so nonzero, on unsigned
  operands (signed division would allow INT_MIN / -1);
- narrowing conversions go through unsigned types (unsigned narrowing is defined modulo 2^N),
  and signed values are produced from unsigned bits with ``as_type``;
- array indices are masked to the array length, buffer indices to the buffer length;
- loops have constant trip counts.

``program(rng, name)`` returns the source of one kernel named ``name``. The kernel reads
``in_bytes`` (at least 4096 bytes) and writes 64 bytes per thread to ``r``, matching the probe
harness of ``corpus/generate.py``.
"""

UNSIGNED = {"uchar": 8, "ushort": 16, "uint": 32, "ulong": 64}
SIGNED = {"char": "uchar", "short": "ushort", "int": "uint", "long": "ulong"}
SCALARS = list(UNSIGNED) + list(SIGNED)


def width(t):
    return UNSIGNED[t] if t in UNSIGNED else UNSIGNED[SIGNED[t]]


def unsigned_of(t):
    return t if t in UNSIGNED else SIGNED[t]


class Program:
    def __init__(self, rng):
        self.rng = rng
        self.vars = []          # (name, type)
        self.lines = []
        self.counter = 0
        self.depth = 0

    def fresh(self, prefix="t"):
        self.counter += 1
        return f"{prefix}{self.counter}"

    def emit(self, line):
        self.lines.append("    " * (self.depth + 1) + line)

    # Expressions produce (text, type). Every expression of type T is a well-defined value of T.
    def leaf(self, t):
        r = self.rng.random()
        same = [n for n, vt in self.vars if vt == t]
        if same and r < 0.5:
            return self.rng.choice(same)
        if r < 0.8:
            u = unsigned_of(t)
            off = self.rng.randrange(0, 4096 // 8 - 1) * 8
            load = f"*(device const {u} *)(in_bytes + (({off}u + i * 8u) & 4088u))"
            return load if t == u else f"as_type<{t}>({load})"
        value = self.rng.getrandbits(width(t))
        u = unsigned_of(t)
        lit = f"{u}({value}ul)" if width(t) == 64 else f"{u}({value}u)"
        return lit if t == u else f"as_type<{t}>({lit})"

    def expr(self, t, depth=0):
        if depth > 3 or self.rng.random() < 0.25:
            return self.leaf(t)
        u = unsigned_of(t)
        w = width(t)
        a = self.expr(t, depth + 1)
        b = self.expr(t, depth + 1)
        ua = a if t == u else f"as_type<{u}>({a})"
        ub = b if t == u else f"as_type<{u}>({b})"
        if w < 32:
            # 8- and 16-bit operands would be promoted to int, where a product or a shift can
            # overflow; compute in uint and narrow (unsigned narrowing is modulo 2^N).
            ua, ub = f"uint({ua})", f"uint({ub})"
        back = (lambda e: f"{u}({e})") if t == u else (lambda e: f"as_type<{t}>({u}({e}))")
        k = self.rng.randrange(17)
        if k == 0:
            return back(f"{ua} + {ub}")
        if k == 1:
            return back(f"{ua} - {ub}")
        if k == 2:
            return back(f"{ua} * {ub}")
        if k == 3:
            return back(f"{ua} & {ub}")
        if k == 4:
            return back(f"{ua} | {ub}")
        if k == 5:
            return back(f"{ua} ^ {ub}")
        if k == 6:
            return back(f"{ua} << ({ub} & {w - 1}u)")
        if k == 7:
            return back(f"{ua} >> ({ub} & {w - 1}u)")
        if k == 8:
            return back(f"{ua} / ({ub} | 1u)")
        if k == 9:
            return back(f"{ua} % ({ub} | 1u)")
        if k == 10:
            return back(f"~{ua}")
        if k == 11:
            c = self.cond(depth + 1)
            return f"(({c}) ? {a} : {b})"
        if k == 12:
            return f"{self.rng.choice(['min', 'max'])}({a}, {b})"
        if k == 13:
            fn = self.rng.choice(["popcount", "clz", "ctz", "reverse_bits"])
            ux = a if t == u else f"as_type<{u}>({a})"
            return back(f"{fn}({ux})")
        if k == 14:
            fn = self.rng.choice(["addsat", "subsat", "hadd", "rhadd", "absdiff", "mulhi"])
            r = f"{fn}({a}, {b})"
            return r if fn != "absdiff" else (r if t == u else f"as_type<{t}>({r})")
        if k == 15:
            ux = a if t == u else f"as_type<{u}>({a})"
            uy = b if t == u else f"as_type<{u}>({b})"
            return back(f"rotate({ux}, {uy})")
        # Conversion from another integer type: widen exactly, narrow through unsigned bits.
        src = self.rng.choice(SCALARS)
        e = self.expr(src, depth + 1)
        us = unsigned_of(src)
        bits = e if src == us else f"as_type<{us}>({e})"
        return back(f"{u}({bits})")

    def cond(self, depth):
        t = self.rng.choice(SCALARS)
        op = self.rng.choice(["<", "<=", ">", ">=", "==", "!="])
        return f"{self.expr(t, depth + 1)} {op} {self.expr(t, depth + 1)}"

    def statement(self, budget):
        k = self.rng.randrange(7)
        if k <= 2 or self.depth > 2:
            t = self.rng.choice(SCALARS)
            name = self.fresh()
            self.emit(f"{t} {name} = {self.expr(t)};")
            self.vars.append((name, t))
            return
        if k == 3 and self.vars:
            name, t = self.rng.choice(self.vars)
            self.emit(f"{name} = {self.expr(t)};")
            return
        if k == 4:
            self.emit(f"if ({self.cond(0)}) {{")
            self.block(budget // 2)
            self.emit("} else {")
            self.block(budget // 2)
            self.emit("}")
            return
        if k == 5:
            n = self.rng.randrange(1, 9)
            idx = self.fresh("k")
            self.emit(f"for (uint {idx} = 0; {idx} < {n}u; {idx}++) {{")
            self.vars.append((idx, "uint"))
            self.block(budget // 2)
            self.vars.remove((idx, "uint"))
            self.emit("}")
            return
        t = self.rng.choice(SCALARS)
        arr = self.fresh("a")
        self.emit(f"{t} {arr}[8];")
        self.emit(f"for (uint z = 0; z < 8u; z++) {arr}[z] = {self.expr(t)};")
        sel = self.expr("uint")
        self.emit(f"switch ({sel} & 3u) {{")
        for case in range(3):
            self.emit(f"case {case}u: {arr}[({self.expr('uint')}) & 7u] = {self.expr(t)}; break;")
        self.emit(f"default: {arr}[0] = {arr}[({self.expr('uint')}) & 7u]; break;")
        self.emit("}")
        name = self.fresh()
        self.emit(f"{t} {name} = {arr}[({self.expr('uint')}) & 7u];")
        self.vars.append((name, t))

    def block(self, budget):
        self.depth += 1
        saved = list(self.vars)
        for _ in range(max(1, budget)):
            self.statement(budget - 1)
        self.vars = saved
        self.depth -= 1


def program(rng, name):
    p = Program(rng)
    for _ in range(rng.randrange(4, 12)):
        p.statement(3)
    outs = []
    slot = 0
    for var, t in p.vars[-8:]:
        size = width(t) // 8
        if slot + size > 64:
            break
        outs.append(f"    *(device {t} *)(r + i * 64 + {slot}) = {var};")
        slot += 8
    body = "\n".join(p.lines + outs)
    return (f"kernel void {name}(device const uchar *in_bytes [[buffer(0)]], device uchar *r [[buffer(30)]],"
            f" uint i [[thread_position_in_grid]])\n{{\n{body}\n}}\n")
