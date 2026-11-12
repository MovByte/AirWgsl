#!/usr/bin/env python3
"""Tally what the AIR modules of a corpus use, from their ``llvm-dis`` text.

Usage:

    survey.py MODULES_TSV LL_DIR OUT_JSON [--jobs N]

``MODULES_TSV`` is a ``modules.tsv`` written by ``tools/extract-system-metallibs.py`` (or by
``corpus/build.py`` for the generated corpus); every distinct ``module_sha256`` in it is read from
``LL_DIR/<sha256>.ll``. A missing ``.ll`` file is an error.

The JSON holds counts only, keyed by name:

``modules``, ``triples``, ``language_versions``, ``air_versions``
    Module counts per target triple and per ``!air.language_version`` / ``!air.version`` value.
``callees``
    Per declared function name: modules declaring it, call sites, and one declaration line.
``named_metadata``, ``metadata_strings``
    Modules containing each named metadata node and each ``air.*`` metadata string.
``opcodes``, ``fp_flags``, ``address_spaces``
    Instruction counts by opcode; floating-point instructions by fast-math flag set; uses of each
    ``addrspace(N)``.
``entry_points``
    Functions listed by each stage's named metadata (``air.vertex``, ``air.fragment``, ...).
"""

import argparse
import collections
import concurrent.futures
import json
import os
import re
import sys

DECLARE = re.compile(r'^declare .*?@("?)([^"(\s]+)\1\(')
DEFINE = re.compile(r'^define .*?@("?)([^"(\s]+)\1\(')
CALL = re.compile(r'\bcall\b[^@]*@("?)([^"(\s]+)\1\(')
NAMED_MD = re.compile(r"^!([A-Za-z_.][\w.]*) = !\{(.*)\}\s*$")
NUMBERED_MD = re.compile(r"^!(\d+) = (?:distinct )?!\{(.*)\}\s*$")
MD_STRING = re.compile(r'!"(air\.[^"]*)"')
ADDRSPACE = re.compile(r"addrspace\((\d+)\)")
INSTRUCTION = re.compile(r"^\s+(?:%[-\w.$\"]+ = )?(?:(?:tail|musttail|notail) )?([a-z][a-z0-9_]*)\b")
NOT_OPCODE = re.compile(r"^(?:i\d+|ptr|label|half|float|double|bfloat|void)$")
FP_OP = re.compile(r"=\s+(fadd|fsub|fmul|fdiv|frem|fneg|fcmp)\s+((?:(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc)\s+)*)")
REF = re.compile(r"!(\d+)")
INT = re.compile(r"i32 (-?\d+)")

STAGE_METADATA = ("air.vertex", "air.fragment", "air.kernel", "air.visible", "air.intersection",
                  "air.object", "air.mesh", "air.patch")


def survey_module(path):
    counts = {
        "triple": "",
        "named": set(),
        "strings": set(),
        "declares": {},
        "defines": 0,
        "calls": collections.Counter(),
        "opcodes": collections.Counter(),
        "fp_flags": collections.Counter(),
        "addrspaces": collections.Counter(),
    }
    numbered = {}
    named = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("target triple = "):
                counts["triple"] = line.split('"')[1]
                continue
            if line.startswith("!"):
                m = NUMBERED_MD.match(line)
                if m:
                    numbered[m.group(1)] = m.group(2)
                else:
                    m = NAMED_MD.match(line)
                    if m:
                        named[m.group(1)] = m.group(2)
                for s in MD_STRING.findall(line):
                    counts["strings"].add(s)
                continue
            if line.startswith("declare "):
                m = DECLARE.match(line)
                if m:
                    counts["declares"][m.group(2)] = line.split(" #")[0].strip()
                continue
            if line.startswith("define "):
                counts["defines"] += 1
            for space in ADDRSPACE.findall(line):
                counts["addrspaces"][space] += 1
            m = INSTRUCTION.match(line)
            if m and not NOT_OPCODE.match(m.group(1)):
                counts["opcodes"][m.group(1)] += 1
                for c in CALL.findall(line):
                    counts["calls"][c[1]] += 1
                f = FP_OP.search(line)
                if f:
                    flags = " ".join(sorted(f.group(2).split())) or "-"
                    counts["fp_flags"][f"{f.group(1)} {flags}"] += 1

    counts["named"] = set(named)

    def operands(body):
        return [numbered.get(ref, "") for ref in REF.findall(body)]

    def version(name):
        if name not in named:
            return "-"
        values = []
        for node in operands(named[name]):
            values.append(".".join(INT.findall(node)) or node)
        return ";".join(values)

    counts["language_version"] = version("air.language_version")
    counts["air_version"] = version("air.version")
    counts["entries"] = {stage: len(REF.findall(named[stage])) for stage in STAGE_METADATA if stage in named}
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("modules_tsv")
    parser.add_argument("ll_dir")
    parser.add_argument("out_json")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    args = parser.parse_args()

    shas = set()
    with open(args.modules_tsv, encoding="utf-8") as fh:
        header = None
        for line in fh:
            if line.startswith("# path"):
                header = line[2:].rstrip("\n").split("\t")
                continue
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            shas.add(fields[header.index("module_sha256")])
    paths = [os.path.join(args.ll_dir, sha + ".ll") for sha in sorted(shas)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"error: {len(missing)} modules have no .ll file, first {missing[0]}")

    total = {
        "modules": 0,
        "triples": collections.Counter(),
        "language_versions": collections.Counter(),
        "air_versions": collections.Counter(),
        "callees": {},
        "named_metadata": collections.Counter(),
        "metadata_strings": collections.Counter(),
        "opcodes": collections.Counter(),
        "fp_flags": collections.Counter(),
        "address_spaces": collections.Counter(),
        "entry_points": collections.Counter(),
        "defined_functions": 0,
    }
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for counts in pool.map(survey_module, paths, chunksize=32):
            total["modules"] += 1
            total["triples"][counts["triple"]] += 1
            total["language_versions"][counts["language_version"]] += 1
            total["air_versions"][counts["air_version"]] += 1
            total["defined_functions"] += counts["defines"]
            for name in counts["named"]:
                total["named_metadata"][name] += 1
            for s in counts["strings"]:
                total["metadata_strings"][s] += 1
            for name, decl in counts["declares"].items():
                entry = total["callees"].setdefault(name, {"modules": 0, "calls": 0, "declaration": decl})
                entry["modules"] += 1
            for name, n in counts["calls"].items():
                entry = total["callees"].setdefault(name, {"modules": 0, "calls": 0, "declaration": ""})
                entry["calls"] += n
            total["opcodes"].update(counts["opcodes"])
            total["fp_flags"].update(counts["fp_flags"])
            total["address_spaces"].update(counts["addrspaces"])
            for stage, n in counts["entries"].items():
                total["entry_points"][stage] += n

    # Calls to functions the module itself defines are not part of the callee vocabulary.
    total["callees"] = {name: v for name, v in total["callees"].items() if v["modules"]}
    out = {k: (dict(sorted(v.items())) if isinstance(v, (dict, collections.Counter)) else v)
           for k, v in total.items()}
    tmp = args.out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, args.out_json)
    print(f"modules={out['modules']} callees={len(out['callees'])} "
          f"air_callees={sum(1 for n in out['callees'] if n.startswith('air.'))}")


if __name__ == "__main__":
    main()
