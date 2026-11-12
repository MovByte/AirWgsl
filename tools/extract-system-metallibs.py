#!/usr/bin/env python3
"""Extract the AIR shaders shipped on an iOS root filesystem into a local corpus.

The input is a mounted or unpacked root filesystem made from the user's own IPSW. The output is a
directory of derived files that stays on the user's machine: it holds Apple's shader code and is
never committed. What the tool guarantees is the index: two runs on the same root produce
byte-identical ``index.tsv``, ``units.tsv`` and ``modules.tsv`` files.

Usage:

    extract-system-metallibs.py ROOT OUT [--jobs N] [--skip-cross-check] [--check-index FILE]
                                [--llvm-dis PATH] [--metal-nm PATH]

Every regular file under ROOT is read (symbolic links are not followed; ``private/var/tmp`` and
``private/tmp`` are skipped because they hold run-time scratch rather than shipped files). A file
is decoded by the magic at its start:

``mtlb``
    A metallib (``MTLB``): header, function list, header extension, bitcode section. Each function
    names one bitcode module by offset and size; several functions may share a module.
``fat``
    A big-endian universal file (``0xCAFEBABE``); each slice is decoded by these same rules.
``macho``
    A 64-bit little-endian Mach-O file (binary archives); each section whose contents start with
    ``MTLB`` is a metallib.
``lppm``
    A pipeline-library container (``LPPM``): a table of 32-byte keys with offsets and sizes of
    entries that tile the data region. Entries that are metallibs (the ``functions`` files) are
    decoded; other entries (pipeline descriptors, GPU binaries) hold no AIR.
``ar``
    A BSD archive (``!<arch>``; the compiler's ``.rtlib`` and ``.a`` runtimes): each bitcode
    member is one module; other members hold no AIR.
``bitcode``
    A file that is one bitcode module (wrapper ``0x0B17C0DE`` or raw ``BC 0xC0DE``).
``agx-rtlib``
    A ``.metallib`` file with none of these magics: the GPU driver's runtime libraries. A ``u32``
    entry count, then per entry a NUL-terminated symbol name and a ``u32`` file offset; at each
    offset a ``u32`` length and one module of that length. The entries must tile the file from the
    end of the table to the end of the file.

After that, every ``MTLB`` occurrence that is not inside a decoded library is tried as an embedded
metallib (the shared cache embeds some in dylib data); it counts only if its header, function list
and module references all decode. Every bitcode-wrapper occurrence that is not inside a decoded
module and whose header points at bitcode is disassembled: if its target triple is AIR the run
fails (AIR in a container this tool does not decode); otherwise it is listed in ``foreign.tsv``.
A ``.metallib`` file that does not decode, or any malformed structure, is an error: nothing is
skipped silently.

Output (OUT):

``index.tsv``
    One line per file that holds AIR, and per ``.metallib`` file even when its function list is
    empty, sorted by path (bytewise). Columns: ``path`` (relative to
    ROOT), ``sha256`` of the file, ``size`` in bytes, ``layout`` (its decoded structure),
    ``functions`` (function-list entries that name a module, summed over the file's metallibs and
    driver runtime tables; ``-`` when the file has no function directory, as for archives),
    ``modules`` (distinct AIR modules by content hash). ``#`` lines carry the format version and
    totals. No timestamps, host names or absolute paths.
``units.tsv``
    One line per decoded library (metallib, driver runtime table, archive, bitcode file): its
    location inside the file, sha256, size, entries with and without modules, header fields.
``modules.tsv``
    One line per entry naming a module: file, unit location, entry name, function type byte
    (metallibs), module sha256, offset and size in the file.
``foreign.tsv``
    Bitcode modules found outside any AIR container whose triple is not AIR.
``files/<path>``, ``units/<sha256>.metallib``, ``bitcode/<sha256>.bc``, ``ll/<sha256>.ll``
    Copies of the files, each metallib as a standalone file, each distinct module, and its
    ``llvm-dis`` output.
``llvm-dis.tsv``
    ``llvm-dis`` version and per-module status. A failure is an error.
``cross-check.tsv``
    Per metallib, the function names that ``metal-nm`` lists against the decoded function list.
    Any difference is an error (``--skip-cross-check`` turns the second method off).
"""

import argparse
import concurrent.futures
import hashlib
import mmap
import os
import shutil
import struct
import subprocess
import sys

INDEX_FORMAT = "airwgsl-system-corpus-index 1"
EXCLUDED_DIRS = ("private/var/tmp", "private/tmp")

MTLB_MAGIC = b"MTLB"
WRAPPER_MAGIC = b"\xde\xc0\x17\x0b"  # 0x0B17C0DE, little-endian
RAW_BITCODE_MAGIC = b"BC\xc0\xde"
FAT_MAGIC = b"\xca\xfe\xba\xbe"
FAT_MAGIC_64 = b"\xca\xfe\xba\xbf"
MACHO_MAGIC_64 = b"\xcf\xfa\xed\xfe"  # MH_MAGIC_64, little-endian
LPPM_MAGIC = b"LPPM"
AR_MAGIC = b"!<arch>\n"
LC_SEGMENT_64 = 0x19

# Names for the file-type byte as `metal-objdump --metallib --private-headers` prints them
# (METALLIB_EXECUTE, METALLIB_CI, METALLIB_DYLIB; bit 7 prints as "(STUB)").
MTLB_FILE_TYPES = {0: "execute", 1: "ci", 2: "dylib"}
MTLB_STUB = 0x80
NO_MODULE = 0xFFFFFFFFFFFFFFFF


class FormatError(Exception):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


class Module:
    __slots__ = ("offset", "size", "sha")

    def __init__(self, data, offset, size):
        self.offset = offset
        self.size = size
        self.sha = sha256(data[offset:offset + size])


class Entry:
    """A named entry of a library's directory; ``module`` is None for a stub entry."""

    __slots__ = ("name", "type", "module", "alias_of")

    def __init__(self, name, etype, module, alias_of=None):
        self.name = name
        self.type = etype
        self.module = module
        self.alias_of = alias_of


class Unit:
    """A decoded library: a metallib, a driver runtime table, an archive or a bitcode file."""

    def __init__(self, kind, location, offset, size, data):
        self.kind = kind
        self.location = location or "."
        self.offset = offset
        self.size = size
        self.sha = sha256(data[offset:offset + size])
        self.fields = {}
        self.entries = []
        self.stubs = []
        # True when the container has a function directory (metallib, driver runtime table).
        self.directory = kind in ("mtlb", "agx-rtlib")


def is_module_at(data, offset, limit):
    return data[offset:offset + 4] in (WRAPPER_MAGIC, RAW_BITCODE_MAGIC) and offset + 4 <= limit


def module_extent(data, offset, limit):
    """Size of the bitcode module starting at ``offset`` as its own header states it.

    A wrapper header (magic, version, offset, size, cputype; five u32) gives the bitcode offset
    and size, and the bitcode there must start with ``BC 0xC0DE``. Returns None when the header
    does not describe bitcode inside [offset, limit).
    """
    if data[offset:offset + 4] != WRAPPER_MAGIC or offset + 20 > limit:
        return None
    _magic, _version, bc_off, bc_size, _cpu = struct.unpack_from("<IIIII", data, offset)
    if bc_off < 20 or offset + bc_off + bc_size > limit:
        return None
    if data[offset + bc_off:offset + bc_off + 4] != RAW_BITCODE_MAGIC:
        return None
    return bc_off + bc_size


def decode_mtlb(data, base, limit, location):
    """Decode the metallib at data[base:limit].

    Header (88 bytes, little-endian): magic; u16 whose bit 15 marks a macOS target and whose low
    bits are the file major version; u16 minor; u16 bug-fix; u8 file type; u8 whose bit 7 marks a
    64-bit target and whose low bits are the platform; u16 platform major; u8 minor; u8 update;
    then u64 pairs: file size, function list (offset, size excluding the leading u32 count),
    public metadata, private metadata, bitcode (offsets relative to the metallib start).

    Function entries: u32 entry size, then tags (four bytes, u16 length, payload) up to ``ENDT``.
    An entry is named by ``NAME`` (NUL-terminated) or, for a specialized alias, by ``ALIS`` (the
    alias name and the original name, each NUL-terminated). ``OFFT`` holds three u64 offsets into
    the public metadata, private metadata and bitcode sections; ``MDSZ`` is the module size. In a
    stub library (bit 7 of the file type: the descriptor of a binary archive's GPU code) entries
    carry no ``MDSZ`` and a bitcode offset of all ones: they name no module.

    After the function list, a tag list (``UUID``, ``HDYN``, ``RLST``, ``SLST`` ...) ends in
    ``ENDT`` before the public metadata.
    """
    if limit - base < 88:
        raise FormatError(f"{location}: truncated metallib header")
    (magic, ver_major, ver_minor, ver_fix, ftype, platform, os_major, os_minor, os_update,
     file_size, fl_off, fl_size, pub_off, pub_size, priv_off, priv_size, bc_off,
     bc_size) = struct.unpack_from("<4sHHHBBHBBQQQQQQQQQ", data, base)
    if magic != MTLB_MAGIC:
        raise FormatError(f"{location}: not a metallib")
    if file_size < 88 or base + file_size > limit:
        raise FormatError(f"{location}: file size 0x{file_size:x} does not fit the container")
    for name, off, size in (("function list", fl_off, fl_size + 4), ("public metadata", pub_off, pub_size),
                            ("private metadata", priv_off, priv_size), ("bitcode", bc_off, bc_size)):
        if off < 88 or off + size > file_size:
            raise FormatError(f"{location}: {name} [0x{off:x}, +0x{size:x}) outside the metallib")
    unit = Unit("mtlb", location, base, file_size, data)
    unit.fields = {
        "file_version": f"{ver_major & 0x7fff}.{ver_minor}.{ver_fix}",
        "macos_target": (ver_major >> 15) & 1,
        "file_type": MTLB_FILE_TYPES.get(ftype & 0x7f, f"0x{ftype & 0x7f:02x}") + ("+stub" if ftype & MTLB_STUB else ""),
        "arch64": (platform >> 7) & 1,
        "platform": platform & 0x7f,
        "os_version": f"{os_major}.{os_minor}.{os_update}",
    }
    end = base + file_size
    p = base + fl_off
    count = u32(data, p)
    p += 4
    entries_start = p
    bc_base = base + bc_off
    bc_end = bc_base + bc_size
    modules = {}
    for index in range(count):
        if p + 4 > end:
            raise FormatError(f"{location}: function list overruns the metallib")
        entry_size = u32(data, p)
        q = p + 4
        tags = {}
        while True:
            if q + 4 > p + entry_size:
                raise FormatError(f"{location}: function {index}: tags overrun the entry")
            tag = data[q:q + 4]
            q += 4
            if tag == b"ENDT":
                break
            length = u16(data, q)
            q += 2
            if tag in tags:
                raise FormatError(f"{location}: function {index}: tag {tag!r} repeated")
            tags[tag] = bytes(data[q:q + length])
            q += length
        if q != p + entry_size:
            raise FormatError(f"{location}: function {index}: entry size {entry_size}, tags end at {q - p}")
        for required in (b"TYPE", b"OFFT"):
            if required not in tags:
                raise FormatError(f"{location}: function {index}: no {required.decode()} tag")
        alias_of = None
        if (b"NAME" in tags) == (b"ALIS" in tags):
            raise FormatError(f"{location}: function {index}: needs exactly one of NAME and ALIS")
        if b"NAME" in tags:
            raw = tags[b"NAME"]
            if not raw.endswith(b"\0") or raw.count(b"\0") != 1:
                raise FormatError(f"{location}: function {index}: malformed NAME")
            name = raw[:-1].decode("utf-8")
        else:
            parts = tags[b"ALIS"].split(b"\0")
            if len(parts) != 3 or parts[2] != b"":
                raise FormatError(f"{location}: function {index}: malformed ALIS")
            name, alias_of = parts[0].decode("utf-8"), parts[1].decode("utf-8")
        if len(tags[b"TYPE"]) != 1 or len(tags[b"OFFT"]) != 24:
            raise FormatError(f"{location}: function {name}: unexpected TYPE/OFFT length")
        etype = tags[b"TYPE"][0]
        _, _, module_off = struct.unpack("<QQQ", tags[b"OFFT"])
        if b"MDSZ" not in tags:
            if module_off != NO_MODULE or not ftype & MTLB_STUB:
                raise FormatError(f"{location}: function {name}: no MDSZ outside a stub library")
            unit.stubs.append(Entry(name, etype, None, alias_of))
            p += entry_size
            continue
        if len(tags[b"MDSZ"]) != 8:
            raise FormatError(f"{location}: function {name}: unexpected MDSZ length")
        module_size = u64(tags[b"MDSZ"], 0)
        start = bc_base + module_off
        if module_off > bc_size or start + module_size > bc_end:
            raise FormatError(f"{location}: function {name}: module outside the bitcode section")
        if not is_module_at(data, start, bc_end):
            raise FormatError(f"{location}: function {name}: no bitcode at 0x{start:x}")
        key = (start, module_size)
        if key not in modules:
            modules[key] = Module(data, start, module_size)
        unit.entries.append(Entry(name, etype, modules[key], alias_of))
        p += entry_size
    if p - entries_start != fl_size:
        raise FormatError(f"{location}: function entries span 0x{p - entries_start:x}, header says 0x{fl_size:x}")

    q = p
    extension = {}
    while True:
        if q + 4 > base + pub_off:
            raise FormatError(f"{location}: header extension runs into the public metadata")
        tag = data[q:q + 4]
        q += 4
        if tag == b"ENDT":
            break
        length = u16(data, q)
        q += 2
        extension[tag] = bytes(data[q:q + length])
        q += length
    if b"UUID" in extension:
        unit.fields["uuid"] = extension[b"UUID"].hex()
    unit.fields["extension_tags"] = ",".join(sorted(t.decode("latin-1") for t in extension)) or "-"

    # Modules must not overlap; a gap in the bitcode section may hold padding but not a module.
    cursor = bc_base
    for start, size in sorted(modules):
        if start < cursor:
            raise FormatError(f"{location}: overlapping modules at 0x{start:x}")
        if WRAPPER_MAGIC in data[cursor:start]:
            raise FormatError(f"{location}: unreferenced module in the bitcode section near 0x{cursor:x}")
        cursor = start + size
    if WRAPPER_MAGIC in data[cursor:bc_end]:
        raise FormatError(f"{location}: unreferenced module at the end of the bitcode section")
    unit.fields["bitcode_gap_bytes"] = bc_size - sum(size for _, size in modules)
    return unit


def decode_agx_rtlib(data, location):
    if len(data) < 4:
        raise FormatError(f"{location}: too short for a driver runtime table")
    count = u32(data, 0)
    p = 4
    table = []
    for _ in range(count):
        end = data.find(b"\0", p)
        if end < 0 or end + 5 > len(data):
            raise FormatError(f"{location}: truncated driver runtime table")
        table.append((bytes(data[p:end]).decode("utf-8"), u32(data, end + 1)))
        p = end + 5
    unit = Unit("agx-rtlib", location, 0, len(data), data)
    cursor = p
    for name, offset in table:
        if offset != cursor:
            raise FormatError(f"{location}: entry {name} at 0x{offset:x}, expected 0x{cursor:x}")
        size = u32(data, offset)
        start = offset + 4
        if start + size > len(data) or not is_module_at(data, start, start + size):
            raise FormatError(f"{location}: entry {name} is not a module inside the file")
        unit.entries.append(Entry(name, None, Module(data, start, size)))
        cursor = start + size
    if cursor != len(data):
        raise FormatError(f"{location}: {len(data) - cursor} bytes after the last entry")
    unit.fields = {"entries": count}
    return [unit], "agx-rtlib"


def decode_lppm(data, base, limit, location):
    """Pipeline-library container (``LPPM``).

    Header: ``LPPM``, three u32 (recorded, not interpreted: 1.1.0 for the ``functions`` files of
    pipeline libraries, other values for their per-GPU ``pipelines.*`` files and the driver's
    ``ds.*`` files), u64 file size, u64 table offset, u64 data offset. At the table offset: u64
    count, then per entry a 32-byte key, a u64 offset relative to the data offset and a u64 size.
    Sorted by offset the entries tile the data region. An entry that starts with ``MTLB`` is a
    metallib whose own file size must equal the entry size; other entries (pipeline descriptors,
    GPU binaries) hold no AIR and are counted.
    """
    if limit - base < 40:
        raise FormatError(f"{location}: truncated LPPM header")
    v0, v1, v2 = struct.unpack_from("<III", data, base + 4)
    file_size, table_off, data_off = struct.unpack_from("<QQQ", data, base + 16)
    if base + file_size != limit:
        raise FormatError(f"{location}: LPPM file size 0x{file_size:x} does not match the container")
    count = u64(data, base + table_off)
    if table_off + 8 + 48 * count != data_off:
        raise FormatError(f"{location}: LPPM table does not end at the data offset")
    units = []
    spans = []
    other = 0
    for i in range(count):
        e = base + table_off + 8 + 48 * i
        offset, size = struct.unpack_from("<QQ", data, e + 32)
        start = base + data_off + offset
        if start + size > limit:
            raise FormatError(f"{location}: LPPM entry {i} outside the file")
        spans.append((start, size))
        if data[start:start + 4] != MTLB_MAGIC:
            other += 1
            continue
        unit = decode_mtlb(data, start, start + size, f"{location}lppm[{i}]/")
        if unit.size != size:
            raise FormatError(f"{location}: LPPM entry {i}: metallib size 0x{unit.size:x}, entry size 0x{size:x}")
        unit.fields["lppm_key"] = bytes(data[e:e + 32]).hex()
        units.append(unit)
    cursor = base + data_off
    for start, size in sorted(spans):
        if start != cursor:
            raise FormatError(f"{location}: LPPM data region has a gap or overlap at 0x{cursor:x}")
        cursor = start + size
    if cursor != limit:
        raise FormatError(f"{location}: LPPM data region does not reach the end of the file")
    return units, f"lppm[{v0}.{v1}.{v2}:mtlb={len(units)},other={other}]"


def decode_ar(data, base, limit, location):
    """BSD archive. Member header: name[16] date[12] uid[6] gid[6] mode[8] size[10] "`\\n".

    A name ``#1/N`` means the real name is the first N bytes of the member data (NUL-padded) and
    counts in the size. Members start at even offsets. ``__.SYMDEF*`` members are the symbol
    table; a member that starts with a bitcode magic is one module; other members (object files
    of ordinary static libraries) hold no AIR and are counted.
    """
    unit = Unit("ar", location, base, limit - base, data)
    unit.directory = False
    p = base + 8
    symdefs = 0
    other = 0
    while p < limit:
        if p + 60 > limit:
            raise FormatError(f"{location}: truncated member header at 0x{p:x}")
        header = bytes(data[p:p + 60])
        if header[58:60] != b"`\n":
            raise FormatError(f"{location}: bad member header at 0x{p:x}")
        name = header[:16].decode("ascii").rstrip(" ")
        size = int(header[48:58].decode("ascii").strip())
        start = p + 60
        if name.startswith("#1/"):
            name_len = int(name[3:])
            name = bytes(data[start:start + name_len]).rstrip(b"\0").decode("utf-8")
            body, body_size = start + name_len, size - name_len
        elif name.startswith("/") or name.endswith("/"):
            raise FormatError(f"{location}: GNU archive member {name!r} (not seen on 21H461)")
        else:
            body, body_size = start, size
        if start + size > limit:
            raise FormatError(f"{location}: member {name} overruns the archive")
        if name.startswith("__.SYMDEF"):
            symdefs += 1
        elif is_module_at(data, body, body + body_size):
            unit.entries.append(Entry(name, None, Module(data, body, body_size)))
        else:
            other += 1
        p = start + size
        p += p & 1
    if p != limit:
        raise FormatError(f"{location}: archive does not end on a member boundary")
    unit.fields = {"members": len(unit.entries), "symbol_tables": symdefs, "other_members": other}
    return ([unit] if unit.entries else []), "ar"


def decode_bitcode_file(data, base, limit, location, rel):
    unit = Unit("bitcode", location, base, limit - base, data)
    unit.directory = False
    if data[base:base + 4] == WRAPPER_MAGIC and module_extent(data, base, limit) is None:
        raise FormatError(f"{location}: wrapper header does not describe bitcode in the file")
    unit.entries.append(Entry(os.path.basename(rel), None, Module(data, base, limit - base)))
    return [unit], "bitcode"


def decode_fat(data, base, limit, location):
    if data[base:base + 4] == FAT_MAGIC_64:
        raise FormatError(f"{location}: 64-bit fat header (not seen on 21H461; decoder not written)")
    count = struct.unpack_from(">I", data, base + 4)[0]
    units = []
    layouts = []
    for i in range(count):
        cputype, cpusub, offset, size, _align = struct.unpack_from(">IIIII", data, base + 8 + 20 * i)
        start = base + offset
        if start + size > limit:
            raise FormatError(f"{location}: slice {i} outside the file")
        sub, layout = decode_structure(data, start, start + size, f"{location}fat:{cputype:08x}/{cpusub:08x}/", "")
        if sub is None:
            raise FormatError(f"{location}: slice {i} has an unknown format")
        units += sub
        layouts.append(layout)
    return units, "fat[" + ",".join(layouts) + "]"


def decode_macho(data, base, limit, location):
    ncmds = u32(data, base + 16)
    p = base + 32
    units = []
    layouts = []
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, p)
        if cmd == LC_SEGMENT_64:
            nsects = u32(data, p + 64)
            for s in range(nsects):
                sp = p + 72 + 80 * s
                sectname = bytes(data[sp:sp + 16]).rstrip(b"\0").decode()
                segname = bytes(data[sp + 16:sp + 32]).rstrip(b"\0").decode()
                size = u64(data, sp + 40)
                offset = u32(data, sp + 48)
                if size == 0 or offset == 0:
                    continue
                start = base + offset
                if start + size > limit:
                    raise FormatError(f"{location}: section {segname},{sectname} outside the file")
                if data[start:start + 4] == MTLB_MAGIC:
                    units.append(decode_mtlb(data, start, start + size, f"{location}macho:{segname},{sectname}/"))
                    layouts.append(f"{segname},{sectname}:mtlb")
        p += cmdsize
    return units, "macho[" + ",".join(layouts) + "]"


def decode_structure(data, base, limit, location, rel):
    head = data[base:base + 8]
    if head[:4] == MTLB_MAGIC:
        return [decode_mtlb(data, base, limit, location)], "mtlb"
    if head[:4] in (FAT_MAGIC, FAT_MAGIC_64):
        return decode_fat(data, base, limit, location)
    if head[:4] == MACHO_MAGIC_64:
        return decode_macho(data, base, limit, location)
    if head[:4] == LPPM_MAGIC:
        return decode_lppm(data, base, limit, location)
    if head == AR_MAGIC:
        return decode_ar(data, base, limit, location)
    if head[:4] in (WRAPPER_MAGIC, RAW_BITCODE_MAGIC):
        return decode_bitcode_file(data, base, limit, location, rel)
    if base == 0 and rel.endswith(".metallib"):
        return decode_agx_rtlib(data, location)
    return None, None


def find_all(data, needle):
    out = []
    pos = data.find(needle)
    while pos >= 0:
        out.append(pos)
        pos = data.find(needle, pos + 1)
    return out


def contains(ranges, pos):
    return any(start <= pos < start + size for start, size in ranges)


def decode_file(data, rel):
    """Returns (units, layout, loose) where loose lists wrapper offsets outside decoded modules."""
    units, layout = decode_structure(data, 0, len(data), "", rel)
    if units is None:
        units, layout = [], ""
    unit_ranges = [(u.offset, u.size) for u in units]
    embedded = {"mtlb": 0, "stub": 0}
    for pos in find_all(data, MTLB_MAGIC):
        if contains(unit_ranges, pos):
            continue
        try:
            unit = decode_mtlb(data, pos, len(data), f"@{pos:x}/")
        except (FormatError, struct.error, UnicodeDecodeError, ValueError, IndexError):
            continue  # not a metallib header (for example the text "MTLBuffer")
        units.append(unit)
        embedded["mtlb" if unit.entries else "stub"] += 1
        unit_ranges.append((unit.offset, unit.size))
    if embedded["mtlb"] or embedded["stub"]:
        layout += ("+" if layout else "") + f"embedded[mtlb:{embedded['mtlb']},stub:{embedded['stub']}]"
    module_ranges = sorted({(e.module.offset, e.module.size) for u in units for e in u.entries})
    loose = []
    for pos in find_all(data, WRAPPER_MAGIC):
        if contains(module_ranges, pos):
            continue
        extent = module_extent(data, pos, len(data))
        if extent is not None:
            loose.append((pos, extent))
    return units, layout, loose


def walk(root):
    excluded = {os.path.join(root, d) for d in EXCLUDED_DIRS}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if os.path.join(dirpath, d) not in excluded)
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            yield path


def tsv(fields):
    out = []
    for field in fields:
        text = "" if field is None else str(field)
        if "\t" in text or "\n" in text:
            raise FormatError(f"field {text!r} contains a tab or newline")
        out.append(text)
    return "\t".join(out) + "\n"


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_once(path, blob):
    if not os.path.exists(path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, path)


def find_tool(name, explicit):
    if explicit:
        return explicit
    if name == "llvm-dis":
        candidate = "/opt/homebrew/opt/llvm@21/bin/llvm-dis"
        return candidate if os.path.exists(candidate) else shutil.which("llvm-dis")
    proc = subprocess.run(["xcrun", "-f", name], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def disassemble(llvm_dis, bc_path, ll_path):
    """Returns (status, message, triple)."""
    if not os.path.exists(ll_path):
        proc = subprocess.run([llvm_dis, "-o", ll_path + ".tmp", bc_path], capture_output=True, text=True)
        if proc.returncode != 0:
            return "fail", (proc.stderr.strip().splitlines() or ["(no message)"])[0], ""
        os.replace(ll_path + ".tmp", ll_path)
    triple = ""
    with open(ll_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("target triple = "):
                triple = line.split('"')[1]
                break
            if line.startswith(("define ", "declare ")):
                break
    return "ok", "", triple


def metal_nm_names(metal_nm, unit_path):
    proc = subprocess.run([metal_nm, unit_path], capture_output=True, text=True)
    if proc.returncode != 0:
        return None, (proc.stderr.strip().splitlines() or ["(no message)"])[0]
    names = []
    for line in proc.stdout.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[1] == "T":
            names.append(parts[2])
    return names, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", help="root filesystem to read (never written)")
    parser.add_argument("out", help="output directory (local; holds Apple's code)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    parser.add_argument("--skip-cross-check", action="store_true", help="do not compare with metal-nm")
    parser.add_argument("--llvm-dis", help="llvm-dis binary (default: Homebrew llvm@21, then PATH)")
    parser.add_argument("--metal-nm", help="metal-nm binary (default: xcrun -f metal-nm)")
    parser.add_argument("--check-index", help="compare the new index.tsv with this file; differences fail the run")
    args = parser.parse_args()

    root = os.path.realpath(args.root)
    out = os.path.abspath(args.out)
    if out == root or out.startswith(root + os.sep):
        sys.exit("error: OUT must not be inside ROOT")
    llvm_dis = find_tool("llvm-dis", args.llvm_dis)
    if not llvm_dis:
        sys.exit("error: llvm-dis not found (pass --llvm-dis)")
    for sub in ("files", "units", "bitcode", "ll", "foreign"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    errors = []
    records = []  # (rel, file_sha, size, layout, units)
    loose_modules = {}  # sha -> [(rel, offset, size)]
    module_blobs = {}
    scanned = 0
    for path in walk(root):
        rel = os.path.relpath(path, root)
        size = os.path.getsize(path)
        scanned += 1
        if size < 4:
            if rel.endswith(".metallib"):
                errors.append(f"{rel}: too short to be a metallib")
            continue
        with open(path, "rb") as fh:
            data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                head = data[:8]
                interesting = (rel.endswith(".metallib") or data.find(MTLB_MAGIC) >= 0
                               or data.find(WRAPPER_MAGIC) >= 0 or head[:4] in (RAW_BITCODE_MAGIC, LPPM_MAGIC)
                               or head == AR_MAGIC)
                if not interesting:
                    continue
                try:
                    units, layout, loose = decode_file(data, rel)
                except (FormatError, struct.error, UnicodeDecodeError, ValueError) as error:
                    errors.append(f"{rel}: {error}")
                    continue
                if rel.endswith(".metallib") and not units:
                    errors.append(f"{rel}: no decoder accepts this .metallib file")
                    continue
                for pos, extent in loose:
                    blob = bytes(data[pos:pos + extent])
                    sha = sha256(blob)
                    loose_modules.setdefault(sha, []).append((rel, pos, extent))
                    write_once(os.path.join(out, "foreign", sha + ".bc"), blob)
                # A .metallib is indexed even when it holds no AIR (a metallib with an empty
                # function list); any other file only when it holds at least one module.
                if not any(unit.entries for unit in units) and not rel.endswith(".metallib"):
                    continue
                for unit in units:
                    for entry in unit.entries:
                        m = entry.module
                        if m.sha not in module_blobs:
                            module_blobs[m.sha] = True
                            write_once(os.path.join(out, "bitcode", m.sha + ".bc"), data[m.offset:m.offset + m.size])
                    if unit.kind == "mtlb":
                        write_once(os.path.join(out, "units", unit.sha + ".metallib"),
                                   data[unit.offset:unit.offset + unit.size])
                file_sha = sha256(data)
                dest = os.path.join(out, "files", rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not (os.path.exists(dest) and os.path.getsize(dest) == size):
                    shutil.copyfile(path, dest)
                records.append((rel, file_sha, size, layout, units))
            finally:
                data.close()

    # Disassemble every distinct module; classify loose modules by triple.
    todo = sorted(set(module_blobs) | set(loose_modules))

    def job(sha):
        folder = "bitcode" if sha in module_blobs else "foreign"
        return disassemble(llvm_dis, os.path.join(out, folder, sha + ".bc"), os.path.join(out, "ll", sha + ".ll"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(zip(todo, pool.map(job, todo)))
    version = subprocess.run([llvm_dis, "--version"], capture_output=True, text=True).stdout
    vline = next((l.strip() for l in version.splitlines() if "version" in l.lower()), "unknown")
    failed = [sha for sha in todo if results[sha][0] != "ok"]
    write_atomic(os.path.join(out, "llvm-dis.tsv"),
                 f"# {vline}\n# modules={len(todo)} failed={len(failed)}\n# module_sha256\tstatus\ttriple\tmessage\n"
                 + "".join(tsv([sha, results[sha][0], results[sha][2], results[sha][1]]) for sha in todo))
    for sha in failed:
        errors.append(f"llvm-dis failed on module {sha}: {results[sha][1]}")

    foreign_lines = []
    for sha in sorted(loose_modules):
        triple = results[sha][2]
        if triple.startswith("air"):
            where = ", ".join(f"{rel}@0x{pos:x}" for rel, pos, _ in loose_modules[sha])
            errors.append(f"AIR module outside any decoded container: {where}")
        for rel, pos, extent in loose_modules[sha]:
            foreign_lines.append(tsv([rel, f"0x{pos:x}", extent, sha, triple]))
    write_atomic(os.path.join(out, "foreign.tsv"),
                 "# path\toffset\tsize\tmodule_sha256\ttriple\n" + "".join(sorted(foreign_lines)))

    index_lines = []
    unit_lines = []
    module_lines = []
    nfunctions = 0
    air_modules = set()
    nunits = 0
    for rel, file_sha, size, layout, units in sorted(records, key=lambda r: r[0].encode()):
        file_modules = set()
        functions = 0
        has_directory = False
        for unit in units:
            nunits += 1
            has_directory |= unit.directory
            if unit.directory:
                functions += len(unit.entries)
            for entry in unit.entries:
                triple = results[entry.module.sha][2]
                if not triple.startswith("air"):
                    errors.append(f"{rel} {unit.location} {entry.name}: module triple {triple!r} is not AIR")
                file_modules.add(entry.module.sha)
                module_lines.append(tsv([rel, unit.location, entry.name,
                                         "-" if entry.type is None else entry.type,
                                         entry.module.sha, entry.module.offset, entry.module.size]))
            unit_lines.append(tsv([rel, unit.location, unit.kind, unit.sha, unit.size, len(unit.entries),
                                   len(unit.stubs), len({e.module.sha for e in unit.entries})]
                                  + [f"{k}={v}" for k, v in sorted(unit.fields.items())]))
        air_modules |= file_modules
        nfunctions += functions
        index_lines.append(tsv([rel, file_sha, size, layout, functions if has_directory else "-", len(file_modules)]))

    header = (f"# {INDEX_FORMAT}\n"
              f"# excluded={','.join(EXCLUDED_DIRS)}\n"
              f"# files={len(index_lines)} libraries={nunits} functions={nfunctions} modules={len(air_modules)}\n"
              "# path\tsha256\tsize\tlayout\tfunctions\tmodules\n")
    index_text = header + "".join(index_lines)
    write_atomic(os.path.join(out, "index.tsv"), index_text)
    write_atomic(os.path.join(out, "units.tsv"),
                 f"# {INDEX_FORMAT}\n# path\tlocation\tkind\tsha256\tsize\tentries\tstubs\tmodules\tfields...\n"
                 + "".join(unit_lines))
    write_atomic(os.path.join(out, "modules.tsv"),
                 f"# {INDEX_FORMAT}\n# path\tlocation\tentry\ttype\tmodule_sha256\toffset\tsize\n"
                 + "".join(module_lines))

    if not args.skip_cross_check:
        metal_nm = find_tool("metal-nm", args.metal_nm)
        if not metal_nm:
            errors.append("metal-nm not found (pass --metal-nm, or --skip-cross-check)")
        else:
            libs = {}
            for rel, _, _, _, units in records:
                for unit in units:
                    if unit.kind == "mtlb" and unit.sha not in libs:
                        libs[unit.sha] = (rel, unit.location,
                                          sorted(e.name for e in unit.entries + unit.stubs))
            order = sorted(libs, key=lambda sha: (libs[sha][0].encode(), libs[sha][1]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                listed = dict(zip(order, pool.map(
                    lambda sha: metal_nm_names(metal_nm, os.path.join(out, "units", sha + ".metallib")), order)))
            lines = []
            for sha in order:
                rel, location, decoded = libs[sha]
                names, message = listed[sha]
                if names is None:
                    status = "error: " + message
                elif sorted(names) == decoded:
                    status = "equal"
                else:
                    status = f"differ: metal-nm {len(names)} names, decoded {len(decoded)}"
                if status != "equal":
                    errors.append(f"{rel} {location}: metal-nm cross-check {status}")
                lines.append(tsv([rel, location, sha, len(decoded), status]))
            write_atomic(os.path.join(out, "cross-check.tsv"),
                         "# path\tlocation\tunit_sha256\tnames\tmetal-nm\n" + "".join(lines))

    print(f"scanned={scanned} files={len(index_lines)} libraries={nunits} functions={nfunctions} "
          f"modules={len(air_modules)} foreign={len(loose_modules)} index_sha256={sha256(index_text.encode())}")
    if args.check_index:
        with open(args.check_index, encoding="utf-8") as fh:
            if fh.read() != index_text:
                errors.append(f"index differs from {args.check_index}")
            else:
                print(f"index equals {args.check_index}")
    for error in errors:
        print("error:", error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
