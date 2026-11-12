#!/usr/bin/env python3
"""Generate src/bitcode/llvm_intrinsics.rs from LLVM's TableGen'erated intrinsic tables.

The bitcode reader needs, for every target-independent LLVM intrinsic, what LLVM 21's own reader
uses when it upgrades intrinsic declarations: the name, whether the intrinsic is overloaded, its
type signature descriptors (IIT tables) and its attributes. Those come from two files that an LLVM
21.1.8 build generates and installs:

    include/llvm/IR/IntrinsicEnums.inc   (intrinsic ids and names)
    include/llvm/IR/IntrinsicImpl.inc    (target table, overload bits, IIT tables, attributes)

Usage:

    gen-llvm-intrinsics.py LLVM_INCLUDE_DIR OUT.rs

LLVM_INCLUDE_DIR is the directory holding llvm/IR/ (for Homebrew: /opt/homebrew/opt/llvm@21/include).
The output records the LLVM version it was generated from and the sha256 of both inputs; anything
in the inputs this script does not understand is an error.
"""

import hashlib
import os
import re
import sys


def fail(msg):
    sys.exit("error: " + msg)


def section(text, macro):
    m = re.search(r"#ifdef " + macro + r"\n(.*?)\n#endif", text, re.S)
    if not m:
        fail("no section " + macro)
    return m.group(1)


def main():
    if len(sys.argv) != 3:
        fail("usage: gen-llvm-intrinsics.py LLVM_INCLUDE_DIR OUT.rs")
    inc = sys.argv[1]
    enums_path = os.path.join(inc, "llvm/IR/IntrinsicEnums.inc")
    impl_path = os.path.join(inc, "llvm/IR/IntrinsicImpl.inc")
    version_path = os.path.join(inc, "llvm/Config/llvm-config.h")
    enums = open(enums_path).read()
    impl = open(impl_path).read()
    version = re.search(r'#define LLVM_VERSION_STRING "([^"]+)"', open(version_path).read())
    if not version:
        fail("no LLVM_VERSION_STRING")
    version = version.group(1)

    # Target sub-tables: the generic intrinsics are the first entry.
    tdata = section(impl, "GET_INTRINSIC_TARGET_DATA")
    targets = re.findall(r'\{"([^"]*)", (\d+), (\d+)\}', tdata)
    if not targets or targets[0][0] != "":
        fail("unexpected target table")
    generic_count = int(targets[0][2])
    target_names = [t[0] for t in targets[1:]]

    # Names, in id order (id = position + 1).
    ev = section(enums, "GET_INTRINSIC_ENUM_VALUES")
    names = re.findall(r"^\s+\w+(?: = \d+)?,\s+// (llvm\.\S+)$", ev, re.M)
    if len(names) < generic_count:
        fail("fewer names than generic intrinsics")
    names = names[:generic_count]
    if names != sorted(names):
        fail("generic intrinsic names are not sorted")

    # Overload bits, indexed by id.
    ot = section(impl, "GET_INTRINSIC_OVERLOAD_TABLE")
    body = re.search(r"OTable\[\] = \{(.*?)\};", ot, re.S).group(1)
    obytes = []
    for entry in body.split(","):
        entry = entry.strip()
        if not entry:
            continue
        v = 0
        for bit in re.findall(r"\(1<<(\d+)\)", entry):
            v |= 1 << int(bit)
        if re.sub(r"\(1<<\d+\)|[0|\s]", "", entry):
            fail("unexpected overload table entry " + entry)
        obytes.append(v)

    def overloaded(i):
        return (obytes[i // 8] >> (i % 8)) & 1 == 1

    # IIT tables.
    gg = section(impl, "GET_INTRINSIC_GENERATOR_GLOBAL")
    iit = re.search(r"IIT_Table\[\] = \{(.*?)\};", gg, re.S).group(1)
    iit_vals = []
    for entry in iit.split(","):
        entry = entry.strip()
        if not entry:
            continue
        m = re.fullmatch(r"\(1U<<15\) \| (\d+)", entry)
        if m:
            iit_vals.append(0x8000 | int(m.group(1)))
        else:
            iit_vals.append(int(entry, 0))
    long_t = re.search(r"IIT_LongEncodingTable\[\] = \{(.*?)\};", gg, re.S).group(1)
    long_vals = []
    for entry in re.sub(r"/\*.*?\*/", "", long_t, flags=re.S).split(","):
        entry = entry.strip()
        if entry:
            long_vals.append(int(entry, 0))

    # Attributes.
    def between(start, end):
        a = impl.find(start)
        b = impl.find(end, a + len(start)) if a >= 0 else -1
        if a < 0 or b < 0:
            fail("cannot find %r .. %r" % (start, end))
        return impl[a:b]

    arg_fn = between("static AttributeSet getIntrinsicArgAttributeSet(", "static AttributeSet getIntrinsicFnAttributeSet(")
    fn_fn = between("static AttributeSet getIntrinsicFnAttributeSet(", "static constexpr uint16_t IntrinsicsToAttributesMap[]")
    at = between("static constexpr uint16_t IntrinsicsToAttributesMap[]", "#endif // GET_INTRINSIC_ATTRIBUTES")

    def parse_sets(text, is_fn):
        sets = {}
        for m in re.finditer(r"case (\d+):\s*return AttributeSet::get\(C, \{(.*?)\}\);", text, re.S):
            items = []
            for line in m.group(2).splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                a = re.fullmatch(r"Attribute::get\(C, Attribute::(\w+)\),", line)
                if a:
                    items.append("E(AttrKind::%s)" % a.group(1))
                    continue
                a = re.fullmatch(r"Attribute::getWithCaptureInfo\(C, CaptureInfo::none\(\)\),", line)
                if a:
                    items.append("CapturesNone")
                    continue
                a = re.fullmatch(r"Attribute::get\(C, Attribute::Alignment, (\d+)\),", line)
                if a:
                    items.append("Align(%s)" % a.group(1))
                    continue
                a = re.fullmatch(r"Attribute::get\(C, Attribute::Range, ConstantRange\(APInt\(BitWidth, (-?\d+), /\*isSigned=\*/true, /\*implicitTrunc=\*/true\), APInt\(BitWidth, (-?\d+), /\*isSigned=\*/true, /\*implicitTrunc=\*/true\)\)\),", line)
                if a:
                    items.append("Range(%s, %s)" % (a.group(1), a.group(2)))
                    continue
                a = re.fullmatch(r"Attribute::getWithMemoryEffects\(C, MemoryEffects::createFromIntValue\((\d+)\)\),", line)
                if a and is_fn:
                    items.append("Memory(%s)" % a.group(1))
                    continue
                fail("unknown attribute form: " + line)
            sets[int(m.group(1))] = items
        return sets

    arg_sets = parse_sets(arg_fn, False)
    fn_sets = parse_sets(fn_fn, True)
    amap = re.search(r"IntrinsicsToAttributesMap\[\] = \{(.*?)\};", at, re.S).group(1)
    amap_vals = re.findall(r"(\d+) << 8 \| (\d+), // (llvm\.\S+)", amap)
    get_attrs = at[at.find("AttributeList Intrinsic::getAttributes("):]
    lists = {}
    for m in re.finditer(r"case (\d+):\s*return AttributeList(?:\(\)|::get\(C, \{(.*?)\}\));", get_attrs, re.S):
        lid = int(m.group(1))
        entries = []
        has_fn = False
        for e in re.findall(r"\{([^{}]*?)\}", m.group(2) or ""):
            e = e.strip()
            f = re.fullmatch(r"AttributeList::FunctionIndex, getIntrinsicFnAttributeSet\(C, FnAttrID\)", e)
            if f:
                has_fn = True
                continue
            a = re.fullmatch(r"(\d+), getIntrinsicArgAttributeSet\(C, (\d+), FT->getContainedType\((\d+)\)\)", e)
            if not a or a.group(1) != a.group(3):
                fail("unknown attribute list entry: " + e)
            entries.append((int(a.group(1)), int(a.group(2))))
        lists[lid] = (entries, has_fn)

    for fid, lid, an in amap_vals[:generic_count]:
        if int(lid) not in lists:
            fail("attribute list %s of %s not found" % (lid, an))
        if int(fid) not in fn_sets and lists[int(lid)][1]:
            fail("function attribute set %s of %s not found" % (fid, an))
    for entries, _ in lists.values():
        for _, set_id in entries:
            if set_id not in arg_sets:
                fail("argument attribute set %d not found" % set_id)

    out = []
    w = out.append
    w("//! LLVM %s's target-independent intrinsics: names, overloading, type descriptors and attributes." % version)
    w("//!")
    w("//! Generated by `tools/gen-llvm-intrinsics.py` from the LLVM build's `IntrinsicEnums.inc` (sha256 %s)"
      % hashlib.sha256(enums.encode()).hexdigest())
    w("//! and `IntrinsicImpl.inc` (sha256 %s). Do not edit." % hashlib.sha256(impl.encode()).hexdigest())
    w("")
    w("use crate::ir::AttrKind;")
    w("")
    w("pub const LLVM_VERSION: &str = \"%s\";" % version)
    w("")
    w("/// Target name components of target-specific intrinsics (`llvm.<target>.*`).")
    w("pub const TARGETS: &[&str] = &[%s];" % ", ".join('"%s"' % t for t in target_names))
    w("")
    w("/// A parameter or return attribute of an intrinsic.")
    w("#[derive(Clone, Copy, Debug)]")
    w("pub enum ArgAttr {")
    w("    E(AttrKind),")
    w("    CapturesNone,")
    w("    Align(u64),")
    w("    /// `range(iN lo, hi)` with N the value's scalar width; bounds are sign-extended then truncated.")
    w("    Range(i64, i64),")
    w("}")
    w("")
    w("/// A function attribute of an intrinsic.")
    w("#[derive(Clone, Copy, Debug)]")
    w("pub enum FnAttr {")
    w("    E(AttrKind),")
    w("    Memory(u64),")
    w("}")
    w("")
    w("pub use ArgAttr::*;")
    w("pub use FnAttr::Memory;")
    w("")
    w("pub const ARG_SETS: &[&[ArgAttr]] = &[")
    for i in range(len(arg_sets)):
        w("    &[%s]," % ", ".join(arg_sets[i]))
    w("];")
    w("")
    w("pub const FN_SETS: &[&[FnAttr]] = &[")
    for i in range(len(fn_sets)):
        w("    &[%s]," % ", ".join(x if x.startswith("Memory") else "FnAttr::" + x for x in fn_sets[i]))
    w("];")
    w("")
    w("/// Attribute lists: (attribute index, argument set) pairs (index 0 is the return value, i+1 parameter i),")
    w("/// and whether the function attribute set applies.")
    w("pub const LISTS: &[(&[(u32, u16)], bool)] = &[")
    for i in range(max(lists) + 1):
        if i not in lists:
            w("    (&[], false),")
            continue
        entries, has_fn = lists[i]
        w("    (&[%s], %s)," % (", ".join("(%d, %d)" % e for e in entries), "true" if has_fn else "false"))
    w("];")
    w("")
    w("/// One intrinsic: name, overloaded, IIT table word, function attribute set, attribute list.")
    w("pub const GENERIC: &[(&str, bool, u16, u8, u8)] = &[")
    for i, n in enumerate(names):
        fid, lid, an = amap_vals[i]
        if an != n:
            fail("attribute map out of step at " + n)
        w('    ("%s", %s, 0x%04x, %s, %s),' % (n, "true" if overloaded(i + 1) else "false", iit_vals[i], fid, lid))
    w("];")
    w("")
    w("pub const IIT_LONG: &[u8] = &[")
    for i in range(0, len(long_vals), 24):
        w("    %s," % ", ".join(str(v) for v in long_vals[i:i + 24]))
    w("];")
    with open(sys.argv[2], "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("generic=%d targets=%d arg_sets=%d fn_sets=%d lists=%d long=%d"
          % (len(names), len(target_names), len(arg_sets), len(fn_sets), len(lists), len(long_vals)))


if __name__ == "__main__":
    main()
