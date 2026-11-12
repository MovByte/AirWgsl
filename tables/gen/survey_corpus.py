#!/usr/bin/env python3
"""Survey every AIR module of the target root, as the system corpus index lists them.

The modules come from the index that tools/extract-system-metallibs.py writes for the
root: every AIR container it decodes (metallibs, metallibs inside Mach-O binary archives
and universal files, pipeline-library `functions` files, metallibs embedded in the dyld
shared cache, the compiler's runtime archives and bitcode files, the GPU driver's runtime
tables). The index directory holds index.tsv (one row per file), modules.tsv (one row per
entry naming a module) and bitcode/<sha256>.bc (each distinct module). Every module file
is checked against its SHA-256, disassembled with `llvm-dis` and tallied: target triples,
`!air.language_version` values, declared callee names, `air.*` metadata strings and named
metadata. Counts are of distinct modules. The summary is written as JSON to the output
path; nothing is written under the index directory.

Usage: survey_corpus.py --index DIR --llvm-dis PATH --out SUMMARY.json [--jobs N]
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys

INDEX_FORMAT = "# airwgsl-system-corpus-index 1"

DECLARE_RE = re.compile(r'^declare\b[^@]*@("(?:[^"\\]|\\.)*"|[-a-zA-Z$._0-9]+)\(', re.M)
NAMED_MD_RE = re.compile(r"^!(air\.[-a-zA-Z$._0-9]+) = ", re.M)
MD_STRING_RE = re.compile(r'!"(air\.[^"]*)"')
TRIPLE_RE = re.compile(r'^target triple = "([^"]*)"', re.M)
LANGVER_RE = re.compile(r"^!air\.language_version = !\{!(\d+)\}", re.M)
MD_NODE_RE = r'^!{0} = !\{{!"([^"]*)", i32 (\d+), i32 (\d+), i32 (\d+)\}}'


def read_tsv(path):
    """Rows of an index file, whose first line names the index format."""
    rows = []
    with open(path, encoding="utf-8") as f:
        first = f.readline().rstrip("\n")
        if first != INDEX_FORMAT:
            raise SystemExit("%s: not a %s file" % (path, INDEX_FORMAT[2:]))
        for line in f:
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                rows.append(line.split("\t"))
    return rows


def unquote(name):
    if name.startswith('"'):
        return bytes(name[1:-1], "utf-8").decode("unicode_escape")
    return name


def survey_module(args):
    llvm_dis, path, sha = args
    with open(path, "rb") as f:
        module = f.read()
    if hashlib.sha256(module).hexdigest() != sha:
        return {"error": "content does not hash to %s" % sha}
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
    ap.add_argument("--index", required=True, help="output directory of tools/extract-system-metallibs.py")
    ap.add_argument("--llvm-dis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    index_path = os.path.join(a.index, "index.tsv")
    with open(index_path, "rb") as f:
        index_sha = hashlib.sha256(f.read()).hexdigest()
    layouts = {row[0]: row[3] for row in read_tsv(index_path)}
    # A layout names the outer container first: mtlb, lppm, macho, fat, embedded, ar,
    # bitcode or agx-rtlib.
    containers = collections.Counter(re.match(r"[a-z-]+", layout).group(0) for layout in layouts.values())
    entries = read_tsv(os.path.join(a.index, "modules.tsv"))
    modules = {}
    for row in entries:
        path, sha = row[0], row[4]
        if path not in layouts:
            raise SystemExit("modules.tsv names %s, which index.tsv does not list" % path)
        modules.setdefault(sha, set()).add(path)
    jobs = [(a.llvm_dis, os.path.join(a.index, "bitcode", sha + ".bc"), sha) for sha in sorted(modules)]
    for _, path, _ in jobs:
        if not os.path.exists(path):
            raise SystemExit("module file missing: %s" % path)

    triples = collections.Counter()
    langvers = collections.Counter()
    declares = collections.Counter()
    named_md = collections.Counter()
    md_strings = collections.Counter()
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for (_, _, sha), r in zip(jobs, pool.map(survey_module, jobs, chunksize=16)):
            if "error" in r:
                errors.append({"module": sha, "stderr": r["error"]})
                continue
            triples[r["triple"]] += 1
            langvers[r["language_version"]] += 1
            declares.update(r["declares"])
            named_md.update(r["named_md"])
            md_strings.update(r["md_strings"])

    summary = {
        "index_sha256": index_sha,
        "files": len(layouts),
        "containers": dict(sorted(containers.items())),
        "module_entries": len(entries),
        "modules": len(jobs),
        "disassembly_failures": errors,
        "triples": dict(sorted(triples.items(), key=lambda kv: str(kv[0]))),
        "language_versions": dict(sorted(langvers.items(), key=lambda kv: str(kv[0]))),
        "declared_callees": dict(sorted(declares.items())),
        "named_metadata": dict(sorted(named_md.items())),
        "metadata_strings": dict(sorted(md_strings.items())),
    }
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1, sort_keys=False)
        f.write("\n")
    print("files=%d modules=%d failures=%d declared=%d" % (
        len(layouts), len(jobs), len(errors), len(declares)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
