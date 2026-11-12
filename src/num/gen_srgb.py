#!/usr/bin/env python3
"""Generate src/num/wgsl/srgb.wgsl: the sRGB conversion tables of MSL 8.7.7, computed exactly.

sRGB to linear (unpack): for each 8-bit code c, with x = c / 255,
    x <= 0.04045: x / 12.92
    otherwise:    ((x + 0.055) / 1.055)^2.4
as the binary32 and binary16 values nearest the exact result (ties cannot occur: the results are 0, 1 or not
dyadic). The power is an exact rational raised to 12/5: v^5 = q^12, so floor(v * 2^K) is the integer fifth root of
floor(q^12 * 2^5K), and whether that root is exact says whether v has bits beyond 2^-K.

Linear to sRGB (pack): MSL computes c = 12.92 x (x < 0.0031308) or 1.055 x^(1/2.4) - 0.055 (otherwise) after
clamping to [0, 1], then floor(255 c + 0.5). The result is the number of k in 1..255 with 255 c(x) + 0.5 >= k, so
thresholds[k - 1] is the smallest binary32 x with c(x) >= (2k - 1) / 510, found by exact comparison: on the linear
branch x >= (2k - 1) / (510 * 12.92); on the power branch x^5 >= ((400k + 5410) / 107610)^12. (The branches meet with
a small step at 0.0031308 that no threshold falls in, so the result is monotonic in x and a binary search over the
thresholds is exact; this script checks that.)

Usage: gen_srgb.py > src/num/wgsl/srgb.wgsl
"""

from fractions import Fraction
import struct
import sys


def iroot(n, k):
    """floor(n^(1/k)) for an integer n >= 0."""
    if n == 0:
        return 0
    x = 1 << ((n.bit_length() + k - 1) // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            break
        x = y
    while x ** k > n:
        x -= 1
    while (x + 1) ** k <= n:
        x += 1
    return x


def round_binary(fx, k, sticky, mant_bits, emin):
    """The binary floating-point value nearest fx * 2^-k (+ a positive amount below 2^-k when sticky), as (m, e)
    with value m * 2^e, m < 2^mant_bits, rounding to nearest even. emin is the exponent of the smallest subnormal."""
    if fx == 0 and not sticky:
        return 0, emin
    lead = fx.bit_length() - 1 - k  # exponent of the leading bit
    e = max(lead - (mant_bits - 1), emin)
    shift = e + k
    q = fx >> shift
    rem = fx - (q << shift)
    half = 1 << (shift - 1)
    if rem > half or (rem == half and (sticky or q & 1)):
        q += 1
    if q >> mant_bits:
        q >>= 1
        e += 1
    return q, e


def f32_bits(m, e):
    if m == 0:
        return 0
    # value m * 2^e with m < 2^24
    if m < (1 << 23):
        assert e == -149
        return m
    exp = e + 23 + 127
    assert 1 <= exp <= 254
    return (exp << 23) | (m & 0x7FFFFF)


def f16_bits(m, e):
    if m == 0:
        return 0
    if m < (1 << 10):
        assert e == -24
        return m
    exp = e + 10 + 15
    assert 1 <= exp <= 30
    return (exp << 10) | (m & 0x3FF)


K = 256


def srgb_to_linear_fixed(c):
    """floor(v * 2^K) and whether v has further bits, for code c."""
    x = Fraction(c, 255)
    if x <= Fraction(4045, 100000):
        v = x / Fraction(1292, 100)
        fx = (v.numerator << K) // v.denominator
        return fx, (v.numerator << K) % v.denominator != 0
    q = (x + Fraction(55, 1000)) / Fraction(1055, 1000)
    num = q.numerator ** 12 << (5 * K)
    den = q.denominator ** 12
    n = num // den
    r = iroot(n, 5)
    exact = r ** 5 * den == num
    return r, not exact


def f32_value(bits):
    return Fraction(struct.unpack("<f", struct.pack("<I", bits))[0])


def linear_to_srgb_ge(x, k):
    """Whether c(x) >= (2k - 1) / 510 for the binary32 value x in [0, 1]."""
    y = Fraction(2 * k - 1, 510)
    if x < Fraction(31308, 10000000):
        return Fraction(1292, 100) * x >= y
    if x == 0:
        return y <= 0
    q = Fraction(400 * k + 5410, 107610)
    return x ** 5 >= q ** 12


def main():
    to_f32 = []
    to_f16 = []
    for c in range(256):
        fx, sticky = srgb_to_linear_fixed(c)
        to_f32.append(f32_bits(*round_binary(fx, K, sticky, 24, -149)))
        to_f16.append(f16_bits(*round_binary(fx, K, sticky, 11, -24)))
    thresholds = []
    one = 0x3F800000
    for k in range(1, 256):
        lo, hi = 0, one
        assert linear_to_srgb_ge(f32_value(hi), k)
        while lo < hi:
            mid = (lo + hi) // 2
            if linear_to_srgb_ge(f32_value(mid), k):
                hi = mid
            else:
                lo = mid + 1
        thresholds.append(lo)
    # Monotonic thresholds, and no code produced on both sides of the branch step at 0.0031308.
    assert all(a < b for a, b in zip(thresholds, thresholds[1:]))
    step = 0x3B4D2E1B  # the largest binary32 value below 0.0031308 (checked below)
    assert f32_value(step) < Fraction(31308, 10000000) <= f32_value(step + 1)
    for k in range(1, 256):
        assert linear_to_srgb_ge(f32_value(step), k) == linear_to_srgb_ge(f32_value(step + 1), k)

    def table(name, doc, values, width=8):
        rows = []
        for i in range(0, len(values), width):
            rows.append("    " + ", ".join("0x%08xu" % v for v in values[i:i + width]) + ",")
        return "// %s\nconst %s = array<u32, %d>(\n%s\n);" % (doc, name, len(values), "\n".join(rows))

    out = [
        "// Generated by src/num/gen_srgb.py. Do not edit by hand.",
        "//",
        "// The sRGB conversions of MSL 8.7.7, exact (see gen_srgb.py).",
        "",
        table("nm_srgb_to_linear_f32", "sRGB code c to linear, the binary32 value nearest the exact result.", to_f32),
        "",
        table("nm_srgb_to_linear_f16", "sRGB code c to linear, the binary16 value nearest the exact result.", to_f16),
        "",
        table("nm_srgb_thresholds",
              "Linear to sRGB: entry k - 1 is the smallest binary32 value whose sRGB code is at least k.", thresholds),
    ]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
