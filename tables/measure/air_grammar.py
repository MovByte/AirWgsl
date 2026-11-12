#!/usr/bin/env python3
"""Measurement air-grammar: how the target's compiler builds and accepts air.* function names.

The AIR function tables of the iPadOS 21H461 compiler live in libGPUCompilerImpl.dylib
(extracted from the root's dyld shared cache by gen/frontend_names.py). They are LLVM
intrinsic tables: llvm::air::AIRFunctionInfo<version> is constructed from a sorted name
table, an IIT table (one 32-bit word per function) and an IIT long-encoding table, and
answers isOverloaded(ID) from a bitmap. The front end and the runtime compiler use one
AIRFunctionInfo, chosen by AIR version in AIRFunctionInfo::createAIRFunctionInfo, to name,
look up and validate air.* functions:

  getName(ID, Tys)         Names[ID-1], then "." + llvm_utils::getMangledTypeStr(T) for
                           each overloaded type T, in argument-number order.
  lookupName(Name)         Name must start with "air."; llvm_utils::lookupIntrinsicByName
                           finds the longest table name that equals Name or is followed
                           in it by "."; the ID is returned only when Name is that table
                           name and the function is not overloaded, or Name extends it and
                           the function is overloaded.
  isValidFunction(F)       lookupName(F's name), then the IIT descriptors of the ID are
                           matched against F's type by matchIntrinsicType, collecting the
                           overloaded types; F is valid only when F's name equals
                           getName(ID, those types).

The behavior of each routine was read from the image's code (disassembly of the image
whose SHA-256 is ANALYZED_IMAGE below; addresses are virtual addresses in the shared
cache). This script refuses any other image, re-reads every table and constant the
analysis relies on from the image itself (constructors, overload bitmaps, the version
dispatch, the mangler's literal strings, the "." separator, the IIT and argument-kind
jump tables), decodes every function's IIT entries with the decoding read from
getIntrinsicInfoTableEntries, and checks the decoding against the target's own
isOverloaded bitmap for every function of every table.

Outputs: the measurement record (default measure/air_grammar.toml next to this script)
and, with --cache, air-functions.json there: every table's names with their overload
slot kinds, which gen/gen_intrinsics.py reads.

Usage: air_grammar.py --cache DIR [--image PATH] [--out PATH]
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gen"))
sys.dont_write_bytecode = True
import tomlout  # noqa: E402

IMAGE = "/System/Library/PrivateFrameworks/GPUCompiler.framework/Libraries/libGPUCompilerImpl.dylib"
ANALYZED_IMAGE = "82e56263dfd0457cb57cee27b9ba243164d6dcb610802e0ff4174d18925a5d86"

SYM_CREATE = "__ZN4llvm3air15AIRFunctionInfo21createAIRFunctionInfoERNS0_14AIRTypeFactoryENS_12VersionTupleE"
SYM_GETNAME = "__ZNK4llvm3air15AIRFunctionInfo7getNameEjNS_8ArrayRefIPNS_4TypeEEE"
SYM_LOOKUPNAME = "__ZNK4llvm3air15AIRFunctionInfo10lookupNameENS_9StringRefE"
SYM_ISVALID = "__ZNK4llvm3air15AIRFunctionInfo15isValidFunctionERKNS_8FunctionERKNS_6TripleE"
SYM_MATCH = ("__ZNK4llvm3air15AIRFunctionInfo18matchIntrinsicTypeEPNS_4TypeERNS_8ArrayRefINS_9Intrinsic"
             "13IITDescriptorEEERNS_15SmallVectorImplIS3_EE")
SYM_IIT = ("__ZN10llvm_utils28getIntrinsicInfoTableEntriesEjRN4llvm15SmallVectorImplINS0_9Intrinsic"
           "13IITDescriptorEEENS0_8ArrayRefIjEENS6_IhEE")
SYM_MANGLE = "__ZN10llvm_utils17getMangledTypeStrEPN4llvm4TypeE"
SYM_LOOKUP = "__ZN10llvm_utils21lookupIntrinsicByNameEN4llvm8ArrayRefIPKcEENS0_9StringRefE"
SYM_1D8_ISVALID = "__ZNK4llvm3air18AIRFunctionInfo1d815isValidFunctionERKNS_8FunctionERKNS_6TripleE"
CTOR_RE = re.compile(r"^__ZN4llvm3air\d+AIRFunctionInfo(\d+)d(\d+)C2ERNS0_14AIRTypeFactoryE$")
OVERLOADED_RE = re.compile(r"^__ZNK4llvm3air\d+AIRFunctionInfo(\d+)d(\d+)12isOverloadedEj$")

# DecodeIITType (the static helper getIntrinsicInfoTableEntries calls), read from its
# 67-entry jump table and handlers. Each IIT code maps to the descriptor it appends:
# ("int", width), ("vec", width) followed by the element type, ("ptr", 0) with address
# space 0 or ("ptr", "byte") with the address space in the next byte, followed by the
# pointee, ("struct", n) followed by n element types, ("arg", "byte") whose next byte is
# (argument number << 3) | argument kind, or a plain type kind. Code 43 marks the next
# vector as scalable. Codes 54 and 55 append a pointer (address space 10, resp. 20) and
# then the empty struct (resp. i8).
IIT_CODES = {
    0: ("void",), 1: ("int", 1), 2: ("int", 8), 3: ("int", 16), 4: ("int", 32), 5: ("int", 64),
    6: ("half",), 7: ("float",), 8: ("double",), 9: ("vec", 2), 10: ("vec", 4), 11: ("vec", 8),
    12: ("vec", 16), 13: ("vec", 32), 14: ("ptr", 0), 15: ("arg", "byte"), 16: ("vec", 64),
    17: ("mmx",), 18: ("token",), 19: ("metadata",), 20: ("struct", 0), 21: ("struct", 2),
    22: ("struct", 3), 23: ("struct", 4), 24: ("struct", 5), 25: ("extend_arg", "byte"),
    26: ("trunc_arg", "byte"), 27: ("ptr", "byte"), 28: ("vec", 1), 29: ("vararg",),
    30: ("half_vec_arg", "byte"), 31: ("same_vec_width_arg", "byte"), 32: ("ptr_to_arg", "byte"),
    33: ("ptr_to_elt", "byte"), 34: ("vec_of_anyptrs_to_elt", "two bytes"), 35: ("int", 128),
    36: ("vec", 512), 37: ("vec", 1024), 38: ("struct", 6), 39: ("struct", 7), 40: ("struct", 8),
    41: ("fp128",), 42: ("vec_element", "byte"), 43: ("scalable",), 44: ("subdivide2_arg", "byte"),
    45: ("subdivide4_arg", "byte"), 46: ("vec_of_bitcasts_to_int", "byte"), 47: ("vec", 128),
    48: ("bfloat",), 49: ("struct", 9), 50: ("vec", 256), 51: ("amx",), 52: ("ppc_fp128",),
    53: ("vec", 3), 54: ("externref",), 55: ("funcref",), 56: ("anyptr_to_elt", "two bytes"),
}
for _n in range(10, 20):
    IIT_CODES[47 + _n] = ("struct", _n)
# Handler address of each code, as the jump table at IIT_JUMP_TABLE resolves it.
IIT_JUMP_TABLE = 0x1C459ECD0
IIT_JUMP_BASE = 0x1C44011EC

# Argument kinds (the low three bits of an IIT argument byte) as matchIntrinsicType checks
# a type that fills a new overload slot; the jump table at ARGKIND_JUMP_TABLE dispatches
# on them. Kinds 5 and 6 are not produced by the tables; 7 (match an earlier slot) is
# handled before the dispatch, by comparing with the type collected for that slot.
ARGKIND_JUMP_TABLE = 0x1C4468782
ARGKIND_JUMP_BASE = 0x1C4350D7C
ARG_KINDS = {
    0: ("any", 0x1C4350DA4, "returns no mismatch: any type fills the slot"),
    1: ("anyint", 0x1C4350D7C, "the type, or the element type of a vector, has type ID 13 (integer)"),
    2: ("anyfloat", 0x1C4350DB8, "the type, or the element type of a vector, has a type ID of at most 6 (half, bfloat, float, double, x86_fp80, fp128, ppc_fp128)"),
    3: ("anyvector", 0x1C4350DD8, "the type ID is 18 or 19 (fixed or scalable vector)"),
    4: ("anypointer", 0x1C4350DE8, "the type ID is 15 (pointer)"),
}
MATCH_TYPE = 7

# The productions of llvm_utils::getMangledTypeStr, by the literal each one writes.
MANGLER_LITERALS = ["p", "a", "f_", "vararg", "f", "v", "isVoid", "Metadata", "f16", "bf16", "f32",
                    "f64", "f80", "f128", "ppcf128", "x86mmx", "i"]


class Image:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        magic, _, _, _, ncmds, _, _, _ = struct.unpack_from("<8I", self.data, 0)
        if magic != 0xFEEDFACF:
            raise SystemExit("%s is not a 64-bit Mach-O image" % path)
        self.segs = []
        self.symbols = {}
        off = 32
        for _ in range(ncmds):
            cmd, size = struct.unpack_from("<2I", self.data, off)
            if cmd == 0x19:  # LC_SEGMENT_64
                vmaddr, vmsize, fileoff, filesize = struct.unpack_from("<4Q", self.data, off + 24)
                self.segs.append((vmaddr, filesize, fileoff))
            elif cmd == 0x2:  # LC_SYMTAB
                symoff, nsyms, stroff, _ = struct.unpack_from("<4I", self.data, off + 8)
                for i in range(nsyms):
                    strx, ntype, _, _, value = struct.unpack_from("<IBBHQ", self.data, symoff + 16 * i)
                    if ntype & 0x0E == 0x0E:  # N_SECT
                        end = self.data.index(b"\0", stroff + strx)
                        self.symbols[self.data[stroff + strx:end].decode("ascii", "replace")] = value
            off += size

    def off(self, va):
        for vmaddr, filesize, fileoff in self.segs:
            if vmaddr <= va < vmaddr + filesize:
                return fileoff + va - vmaddr
        raise SystemExit("address %#x is outside the image" % va)

    def u8(self, va):
        return self.data[self.off(va)]

    def u32(self, va):
        return struct.unpack_from("<I", self.data, self.off(va))[0]

    def u64(self, va):
        return struct.unpack_from("<Q", self.data, self.off(va))[0]

    def raw(self, va, n):
        o = self.off(va)
        return self.data[o:o + n]

    def cstr(self, va):
        o = self.off(va)
        return self.data[o:self.data.index(b"\0", o)].decode("ascii")

    def sym(self, name):
        if name not in self.symbols:
            raise SystemExit("symbol %s not in the image" % name)
        return self.symbols[name]


# ---------------------------------------------------------------- AArch64 decoding

def sext(v, bits):
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def adrp(w, pc):
    """(Rd, page) of an ADRP instruction, else None."""
    if (w & 0x9F000000) != 0x90000000:
        return None
    imm = sext((((w >> 5) & 0x7FFFF) << 2) | ((w >> 29) & 3), 21)
    return w & 31, (pc & ~0xFFF) + (imm << 12)


def add_imm(w):
    """(Rd, Rn, imm) of a 64-bit ADD (immediate), else None."""
    if (w & 0xFF800000) != 0x91000000:
        return None
    imm = (w >> 10) & 0xFFF
    if (w >> 22) & 1:
        imm <<= 12
    return w & 31, (w >> 5) & 31, imm


def movz(w):
    """(Rd, value, is64) of MOVZ, else None."""
    if (w & 0x7F800000) != 0x52800000:
        return None
    return w & 31, ((w >> 5) & 0xFFFF) << (16 * ((w >> 21) & 3)), bool(w >> 31)


def movk(w):
    """(Rd, imm16, shift) of MOVK, else None."""
    if (w & 0x7F800000) != 0x72800000:
        return None
    return w & 31, (w >> 5) & 0xFFFF, 16 * ((w >> 21) & 3)


def ror(x, r, size):
    return ((x >> r) | (x << (size - r))) & ((1 << size) - 1) if r else x


def orr_imm(w):
    """(Rd, Rn, value) of a 64-bit ORR (immediate), else None (DecodeBitMasks of the Arm ARM)."""
    if (w & 0xFF800000) != 0xB2000000:
        return None
    n, immr, imms = (w >> 22) & 1, (w >> 16) & 0x3F, (w >> 10) & 0x3F
    combined = (n << 6) | (~imms & 0x3F)
    length = combined.bit_length() - 1
    size = 1 << length
    s, r = imms & (size - 1), immr & (size - 1)
    elem = ror((1 << (s + 1)) - 1, r, size)
    value = 0
    for i in range(64 // size):
        value |= elem << (i * size)
    return w & 31, (w >> 5) & 31, value


def branch(w, pc):
    """("b"|"bl", target) of an unconditional immediate branch, else None."""
    if (w & 0x7C000000) != 0x14000000:
        return None
    return ("bl" if w >> 31 else "b"), pc + 4 * sext(w & 0x3FFFFFF, 26)


def tbz(w, pc):
    """(Rt, bit, target) of TBZ, else None."""
    if (w & 0x7F000000) != 0x36000000:
        return None
    return w & 31, ((w >> 31) << 5) | ((w >> 19) & 31), pc + 4 * sext((w >> 5) & 0x3FFF, 14)


def adr(w, pc):
    if (w & 0x9F000000) != 0x10000000:
        return None
    return w & 31, pc + sext((((w >> 5) & 0x7FFFF) << 2) | ((w >> 29) & 3), 21)


def page_refs(img, start, end):
    """Addresses formed by ADRP Xn + ADD Xn, Xn, #imm pairs in [start, end)."""
    out = []
    pages = {}
    for pc in range(start, end, 4):
        w = img.u32(pc)
        a = adrp(w, pc)
        if a:
            pages[a[0]] = a[1]
            continue
        d = add_imm(w)
        if d and d[1] in pages and d[0] == d[1]:
            out.append((pc, d[0], pages.pop(d[1]) + d[2]))
    return out


def function_end(img, start):
    return min(v for v in img.symbols.values() if v > start)


# ---------------------------------------------------------------- tables

def read_ctor(img, va):
    """(names, iit, long, counts) of an AIRFunctionInfo<version> constructor."""
    refs = {reg: addr for _, reg, addr in page_refs(img, va, va + 48)}
    counts = {}
    for pc in range(va, va + 48, 4):
        m = movz(img.u32(pc))
        if m and not m[2]:
            counts[m[0]] = m[1]
    if not all(r in refs for r in (2, 4, 6)) or not all(r in counts for r in (3, 5, 7)):
        raise SystemExit("constructor at %#x does not have the analyzed shape" % va)
    return refs[2], counts[3], refs[4], counts[5], refs[6], counts[7]


def read_bitmap(img, va):
    refs = page_refs(img, va, va + 16)
    if len(refs) != 1 or refs[0][1] != 9:
        raise SystemExit("isOverloaded at %#x does not have the analyzed shape" % va)
    return refs[0][2]


def decode_iit(infos, pos, out, last=None):
    code = infos[pos[0]]
    pos[0] += 1
    if code not in IIT_CODES:
        raise SystemExit("IIT code %d is outside the analyzed decoder" % code)
    d = IIT_CODES[code]
    kind = d[0]
    if kind == "scalable":
        decode_iit(infos, pos, out, code)
        return
    if kind in ("int", "struct"):
        out.append((kind, d[1]))
        if kind == "struct":
            for _ in range(d[1]):
                decode_iit(infos, pos, out)
        return
    if kind == "vec":
        out.append(("vec", d[1], last == 43))
        decode_iit(infos, pos, out)
        return
    if kind == "ptr":
        space = 0
        if d[1] == "byte":
            space = infos[pos[0]]
            pos[0] += 1
        out.append(("ptr", space))
        decode_iit(infos, pos, out)
        return
    if kind == "arg":
        # At the end of an inline-encoded entry the argument byte reads as 0 and the
        # position does not advance.
        info = 0
        if pos[0] < len(infos):
            info = infos[pos[0]]
            pos[0] += 1
        out.append(("arg", info >> 3, info & 7))
        return
    if len(d) > 1:
        raise SystemExit("IIT code %d (%s) occurs in a table; the matcher has no case for it" % (code, kind))
    out.append((kind,))


def iit_descriptors(iit, long_table, fid):
    value = iit[fid - 1]
    if value >> 31:
        infos, pos = long_table, [value & 0x7FFFFFFF]
    else:
        nibbles = []
        while True:
            nibbles.append(value & 0xF)
            value >>= 4
            if not value:
                break
        infos, pos = bytes(nibbles), [0]
    out = []
    decode_iit(infos, pos, out)
    while pos[0] != len(infos) and infos[pos[0]] != 0:
        decode_iit(infos, pos, out)
    return out


def slots_of(desc):
    """Overload slot kinds in argument-number order (the kind at each number's first use)."""
    seen = {}
    for d in desc:
        if d[0] == "arg" and d[1] not in seen:
            if d[1] != len(seen):
                raise SystemExit("argument %d is used before argument %d" % (d[1], len(seen)))
            if d[2] == MATCH_TYPE or d[2] not in ARG_KINDS:
                raise SystemExit("argument %d first appears with kind %d" % (d[1], d[2]))
            seen[d[1]] = d[2]
    return [ARG_KINDS[seen[i]][0] for i in range(len(seen))]


def read_dispatch(img, tables_by_names):
    """[(version, tag)] in the order createAIRFunctionInfo tests them, and the default tag."""
    start = img.sym(SYM_CREATE)
    end = function_end(img, start)
    x8 = None
    compares = []  # (version constant, pc of the call to operator<)
    tests = []     # (version constant, branch target when the version is not below it)
    for pc in range(start, end, 4):
        w = img.u32(pc)
        m = movz(w)
        if m and m[0] == 8 and m[2]:
            x8 = m[1]
        o = orr_imm(w)
        if o and o[0] == 8 and o[1] == 31:
            x8 = o[2]
        k = movk(w)
        if k and k[0] == 8 and x8 is not None:
            x8 = (x8 & ~(0xFFFF << k[2])) | (k[1] << k[2])
        b = branch(w, pc)
        if b and b[0] == "bl" and b[1] == VERSION_LESS:
            compares.append((x8, pc))
        t = tbz(w, pc)
        if t and t[0] == 0 and t[1] == 0:
            tests.append((compares[-1][0], t[2]))
            x8 = None
    if len(compares) != len(tests) + 1:
        raise SystemExit("createAIRFunctionInfo does not have the analyzed shape")

    def table_at(block):
        for _, reg, addr in page_refs(img, block, block + 96):
            if reg == 2:
                return tables_by_names[addr]
        raise SystemExit("no name table reference after %#x" % block)

    def version(value):
        if not (value >> 63):
            raise SystemExit("version constant %#x has no minor component" % value)
        return "%d.%d" % (value & 0xFFFFFFFF, (value >> 32) & 0x7FFFFFFF)

    out = [(version(value), table_at(target)) for value, target in tests]
    # The last comparison's result is not tested: every version below the last tested one
    # falls through to the block after it.
    return out, table_at(compares[-1][1] + 4)


# llvm::operator<(VersionTuple const&, VersionTuple const&) (no symbol in the image): compares
# major, then minor, subminor and build without their presence bits; returns 1 when the
# first operand is below the second.
VERSION_LESS = 0x1C40D1008


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=os.environ.get("TABLES_CACHE"))
    ap.add_argument("--image")
    ap.add_argument("--out", default=os.path.join(HERE, "air_grammar.toml"))
    a = ap.parse_args()
    if not a.cache:
        raise SystemExit("a cache directory is required (--cache or TABLES_CACHE)")
    image_path = a.image or os.path.join(a.cache, "frontend-images", os.path.basename(IMAGE))
    img = Image(image_path)
    if img.sha256 != ANALYZED_IMAGE:
        raise SystemExit("%s has SHA-256 %s; the analysis applies to %s only" % (image_path, img.sha256, ANALYZED_IMAGE))
    with open(os.path.join(a.cache, "frontend-names.json")) as f:
        cache_uuid = json.load(f)["cache_uuid"]

    # Version tables.
    tables = {}
    for name, va in img.symbols.items():
        m = CTOR_RE.match(name)
        if m:
            tag = "%s.%s" % (m.group(1), m.group(2))
            names_va, n_names, iit_va, n_iit, long_va, n_long = read_ctor(img, va)
            tables[tag] = {"ctor": va, "names_va": names_va, "iit_va": iit_va, "long_va": long_va,
                           "names": [img.cstr(img.u64(names_va + 8 * i) & 0xFFFFFFFFF) for i in range(n_names)],
                           "iit": [img.u32(iit_va + 4 * i) for i in range(n_iit)],
                           "long": img.raw(long_va, n_long)}
    for name, va in img.symbols.items():
        m = OVERLOADED_RE.match(name)
        if m:
            tables["%s.%s" % (m.group(1), m.group(2))]["bitmap_va"] = read_bitmap(img, va)
    order = sorted(tables, key=lambda t: tuple(int(x) for x in t.split(".")))
    for tag in order:
        t = tables[tag]
        names = t["names"]
        if names != sorted(names, key=lambda s: s.encode()):
            raise SystemExit("AIR %s name table is not sorted" % tag)
        if len(t["iit"]) != len(names) + 1:
            raise SystemExit("AIR %s: %d IIT words for %d names" % (tag, len(t["iit"]), len(names)))
        t["slots"] = {}
        overloaded = 0
        for i, n in enumerate(names):
            fid = i + 1
            slots = slots_of(iit_descriptors(t["iit"], t["long"], fid))
            bit = (img.u8(t["bitmap_va"] + (fid >> 3)) >> (fid & 7)) & 1
            if bool(slots) != bool(bit):
                raise SystemExit("AIR %s %s: decoded %d slots but isOverloaded = %d" % (tag, n, len(slots), bit))
            overloaded += bit
            t["slots"][n] = slots
        t["overloaded"] = overloaded
    by_names = {tables[t]["names_va"]: t for t in order}
    dispatch, default = read_dispatch(img, by_names)

    # getName: the separator literal and the call to the mangler.
    gn = img.sym(SYM_GETNAME)
    gn_end = function_end(img, gn)
    seps = [img.cstr(addr) for _, _, addr in page_refs(img, gn, gn_end)]
    calls = [branch(img.u32(pc), pc) for pc in range(gn, gn_end, 4)]
    if seps != ["."] or ("bl", img.sym(SYM_MANGLE)) not in calls:
        raise SystemExit("getName does not have the analyzed shape")

    # The mangler's literals.
    mg = img.sym(SYM_MANGLE)
    literals = [img.cstr(addr) for _, _, addr in page_refs(img, mg, function_end(img, mg))]
    literals = [s for s in literals if s]
    if sorted(set(literals)) != sorted(MANGLER_LITERALS):
        raise SystemExit("getMangledTypeStr literals %r differ from the analysis" % literals)

    # lookupName: the "air." test and the call to lookupIntrinsicByName.
    ln = img.sym(SYM_LOOKUPNAME)
    ln_words = [img.u32(pc) for pc in range(ln, function_end(img, ln), 4)]
    air_word = None
    for w in ln_words:
        m, k = movz(w), movk(w)
        if m and m[0] == 9:
            air_word = m[1]
        if k and k[0] == 9 and air_word is not None:
            air_word |= k[1] << k[2]
    if air_word is None or struct.pack("<I", air_word) != b"air.":
        raise SystemExit("lookupName does not test for the air. prefix as analyzed")
    if ("bl", img.sym(SYM_LOOKUP)) not in [branch(w, ln + 4 * i) for i, w in enumerate(ln_words)]:
        raise SystemExit("lookupName does not call lookupIntrinsicByName")

    # Jump tables the decoding was read from.
    iit_handlers = {code: IIT_JUMP_BASE + 4 * img.u8(IIT_JUMP_TABLE + code) for code in range(67)}
    kind_handlers = {k: ARGKIND_JUMP_BASE + 4 * img.u8(ARGKIND_JUMP_TABLE + k) for k in ARG_KINDS}
    for k, (_, handler, _) in ARG_KINDS.items():
        if kind_handlers[k] != handler:
            raise SystemExit("argument kind %d dispatches to %#x, not %#x" % (k, kind_handlers[k], handler))

    # Records.
    meta = {
        "id": "air-grammar",
        "script": "tables/measure/air_grammar.py",
        "image": IMAGE,
        "image_sha256": img.sha256,
        "cache": "iPadOS 17.7.11 (21H461) dyld shared cache %s" % cache_uuid,
        "method": "behavior read from the disassembly of the image (addresses are shared-cache virtual addresses); tables, constants and jump tables re-read from the image by this script",
    }
    cases = []

    def case(cid, **kv):
        cases.append({"id": cid, **kv})

    case("get-name",
         routine="llvm::air::AIRFunctionInfo::getName(unsigned, ArrayRef<Type*>) at %#x" % gn,
         rule="the name is Names[ID-1] followed, for each overloaded type T in argument-number order, by \".\" and llvm_utils::getMangledTypeStr(T)",
         separator=seps[0])
    case("lookup-name",
         routine="llvm::air::AIRFunctionInfo::lookupName(StringRef) at %#x; llvm_utils::lookupIntrinsicByName at %#x" % (ln, img.sym(SYM_LOOKUP)),
         rule="the name must begin with air.; successive binary searches over the sorted name table by dot-separated components find the longest table name that equals the name or is followed in it by a dot; the ID is returned when the name equals that table name and the function is not overloaded, or extends it and the function is overloaded; otherwise there is no function")
    case("is-valid-function",
         routine="llvm::air::AIRFunctionInfo::isValidFunction(Function const&, Triple const&) at %#x" % img.sym(SYM_ISVALID),
         rule="lookupName, then the IIT descriptors of the ID are matched against the return and parameter types (collecting the overloaded types), and the function's name must equal getName(ID, the collected types)",
         exception="AIRFunctionInfo1d8 overrides isValidFunction (at %#x) with a separate rule for clz and ctz under AIR 1.8 to 1.10; that rule is not decoded here" % img.sym(SYM_1D8_ISVALID))
    case("mangled-type",
         routine="llvm_utils::getMangledTypeStr(Type*) at %#x" % mg,
         rule="pointer: p, address space, pointee; array: a, element count, element; struct: its name; function: f_, return, parameters, vararg if variadic, f; fixed vector: v, element count, element; void: isVoid; metadata: Metadata; half f16; bfloat bf16; float f32; double f64; x86_fp80 f80; fp128 f128; ppc_fp128 ppcf128; x86_mmx x86mmx; integer: i and the bit width",
         literals=sorted(set(literals)))
    case("arg-kinds",
         routine="llvm::air::AIRFunctionInfo::matchIntrinsicType at %#x; argument-kind jump table at %#x" % (img.sym(SYM_MATCH), ARGKIND_JUMP_TABLE),
         kinds=["%d %s: %s" % (k, v[0], v[2]) for k, v in sorted(ARG_KINDS.items())],
         match="kind 7 (or any later use of an argument number) requires the type already collected for that number")
    case("iit-codes",
         routine="IIT decoder called by llvm_utils::getIntrinsicInfoTableEntries (at %#x); jump table at %#x" % (img.sym(SYM_IIT), IIT_JUMP_TABLE),
         entry="an IIT word with bit 31 set is an offset into the long-encoding table; otherwise its nibbles, low first, are the codes",
         codes=["%d %s at %#x" % (c, " ".join(str(x) for x in IIT_CODES[c]), iit_handlers[c]) for c in range(67)],
         air_types="a pointer whose address space is 0x80 or more followed by code 17 denotes the AIR type of kind (address space - 0x80) from AIRTypeFactory")
    case("version-dispatch",
         routine="llvm::air::AIRFunctionInfo::createAIRFunctionInfo(AIRTypeFactory&, VersionTuple) at %#x" % img.sym(SYM_CREATE),
         rule="the first table whose version the requested AIR version is not below, tested in this order; the last table otherwise",
         order=["%s: AIRFunctionInfo%s" % (v, t.replace(".", "d")) for v, t in dispatch] + ["below %s: AIRFunctionInfo%s" % (dispatch[-1][0], default.replace(".", "d"))])
    for tag in order:
        t = tables[tag]
        shapes = {}
        for s in t["slots"].values():
            key = ", ".join(s) if s else "none"
            shapes[key] = shapes.get(key, 0) + 1
        case("table-" + tag.replace(".", "d"),
             table="AIRFunctionInfo%s: names at %#x, IIT at %#x, long encoding at %#x" % (tag.replace(".", "d"), t["names_va"], t["iit_va"], t["long_va"]),
             functions=len(t["names"]), overloaded=t["overloaded"],
             slots=["%s: %d" % (k, v) for k, v in sorted(shapes.items())])

    parts = ["# Generated by tables/measure/air_grammar.py. Do not edit by hand.\n", tomlout.table("meta", meta)]
    for c in cases:
        parts.append(tomlout.array_item("case", c))
    with open(a.out, "w") as f:
        f.write("\n".join(parts))

    dump = {"image": IMAGE, "image_sha256": img.sha256, "cache_uuid": cache_uuid,
            "dispatch": [{"version": v, "table": t} for v, t in dispatch], "default": default,
            "tables": {tag: {"slots": tables[tag]["slots"]} for tag in order}}
    with open(os.path.join(a.cache, "air-functions.json"), "w") as f:
        json.dump(dump, f, indent=1)
        f.write("\n")
    print("wrote %s; %s" % (a.out, ", ".join("AIR %s: %d functions" % (t, len(tables[t]["names"])) for t in order)))


if __name__ == "__main__":
    main()
