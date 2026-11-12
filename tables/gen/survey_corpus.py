#!/usr/bin/env python3
"""Survey the AIR modules embedded in a root's metallibs.

Every `.metallib` under the root is scanned for LLVM bitcode wrapper headers
(magic 0x0B17C0DE, then version, offset, size and CPU type as little-endian
u32). Each wrapped module is disassembled with `llvm-dis` and tallied:
target triples, `!air.language_version` values, declared callee names,
`air.*` metadata strings and named metadata. The summary is written as JSON
to the output path; nothing is written under the root.

Usage: survey_corpus.py --root ROOT --llvm-dis PATH --out SUMMARY.json [--jobs N]
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

WRAPPER_MAGIC = 0x0B17C0DE
BITCODE_MAGIC = b"BC\xc0\xde"

DECLARE_RE = re.compile(r'^declare\b[^@]*@("(?:[^"\\]|\\.)*"|[-a-zA-Z$._0-9]+)\(', re.M)
NAMED_MD_RE = re.compile(r"^!(air\.[-a-zA-Z$._0-9]+) = ", re.M)
MD_STRING_RE = re.compile(r'!"(air\.[^"]*)"')
TRIPLE_RE = re.compile(r'^target triple = "([^"]*)"', re.M)
LANGVER_RE = re.compile(r"^!air\.language_version = !\{!(\d+)\}", re.M)
MD_NODE_RE = r'^!{0} = !\{{!"([^"]*)", i32 (\d+), i32 (\d+), i32 (\d+)\}}'


def find_metallibs(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".metallib"):
                path = os.path.join(dirpath, name)
                if os.path.isfile(path) and not os.path.islink(path):
                    out.append(path)
    return out


def wrapped_modules(data):
    """Yield (file offset, module bytes) for every bitcode wrapper in data."""
    magic = struct.pack("<I", WRAPPER_MAGIC)
    pos = data.find(magic)
    while pos >= 0:
        if pos + 20 <= len(data):
            _, version, offset, size, _cpu = struct.unpack_from("<5I", data, pos)
            start = pos + offset
            end = start + size
            if version == 0 and offset == 20 and end <= len(data) and data[start:start + 4] == BITCODE_MAGIC:
                yield pos, data[pos:end]
                pos = data.find(magic, end)
                continue
        pos = data.find(magic, pos + 1)


def unquote(name):
    if name.startswith('"'):
        return bytes(name[1:-1], "utf-8").decode("unicode_escape")
    return name


def survey_module(args):
    llvm_dis, module = args
    proc = subprocess.run([llvm_dis, "-o", "-", "-"], input=module, capture_output=True)
    if proc.returncode != 0:
        return {"error": proc.stderr.decode("utf-8", "replace")[:400]}
    text = proc.stdout.decode("utf-8", "replace")
    triple = TRIPLE_RE.search(text)
    langver = None
    m = LANGVER_RE.search(text)
    if m:
        node = re.search(MD_NODE_RE.format(m.group(1)), text, re.M)
        if node:
            langver = "%s %s.%s.%s" % node.groups()
    return {
        "triple": triple.group(1) if triple else None,
        "language_version": langver,
        "declares": sorted({unquote(n) for n in DECLARE_RE.findall(text)}),
        "named_md": sorted(set(NAMED_MD_RE.findall(text))),
        "md_strings": sorted(set(MD_STRING_RE.findall(text))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--llvm-dis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    libs = find_metallibs(a.root)
    files = []
    jobs = []
    for path in libs:
        with open(path, "rb") as f:
            data = f.read()
        mods = list(wrapped_modules(data))
        files.append({
            "path": os.path.relpath(path, a.root),
            "sha256": hashlib.sha256(data).hexdigest(),
            "modules": len(mods),
        })
        for _, mod in mods:
            jobs.append((a.llvm_dis, mod))

    triples = collections.Counter()
    langvers = collections.Counter()
    declares = collections.Counter()
    named_md = collections.Counter()
    md_strings = collections.Counter()
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for i, r in enumerate(pool.map(survey_module, jobs, chunksize=16)):
            if "error" in r:
                errors.append({"module_index": i, "stderr": r["error"]})
                continue
            triples[r["triple"]] += 1
            langvers[r["language_version"]] += 1
            declares.update(r["declares"])
            named_md.update(r["named_md"])
            md_strings.update(r["md_strings"])

    summary = {
        "metallibs": len(files),
        "modules": len(jobs),
        "disassembly_failures": errors,
        "files": files,
        "triples": dict(sorted(triples.items(), key=lambda kv: str(kv[0]))),
        "language_versions": dict(sorted(langvers.items(), key=lambda kv: str(kv[0]))),
        "declared_callees": dict(sorted(declares.items())),
        "named_metadata": dict(sorted(named_md.items())),
        "metadata_strings": dict(sorted(md_strings.items())),
    }
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1, sort_keys=False)
        f.write("\n")
    print("metallibs=%d modules=%d failures=%d declared=%d" % (
        len(files), len(jobs), len(errors), len(declares)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
