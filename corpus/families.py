#!/usr/bin/env python3
"""Derive the AIR intrinsic family list from the target compiler's strings and the corpus survey.

Usage:

    families.py OUT_JSON --survey SYSTEM_SURVEY [--generated-survey GEN_SURVEY]
                (--root ROOT | --binaries DIR)

The plan derives the intrinsic table (package D2) from two sources that must agree: the
``air.*`` strings in the target's compiler binaries and the survey of the system corpus. This
script derives the same list, so that the coverage report has a family list while D2's
``tables/intrinsics.toml`` does not exist yet (``coverage.py`` prefers D2's table whenever it
exists).

Target binaries: ``MTLCompiler`` and the front-end libraries it links
(``libGPUCompiler.dylib``, ``libGPUCompilerImpl.dylib``, ``libGPUCompilerImplLazy.dylib``), plus
the GPU back end ``AGXCompilerCore`` recorded as a separate provenance column. With ``--root``
they are extracted from ROOT's shared cache with ``ipsw dyld extract`` into ``OUT_JSON``'s
directory; with ``--binaries`` they are read from DIR.

Classification of each target string (``air.`` followed by name characters):

``intrinsic``
    Declared as a function in the system or generated corpus.
``intrinsic-prefix``
    A prefix of such a name at a component boundary (the front end composes names from
    prefixes and type suffixes); it names no family by itself.
``metadata``
    Used as a metadata string or named metadata node in either corpus.
``unclassified``
    Neither. Listed with its decomposition; it is not dropped.

The family list is every family of an ``air.*`` function declared in the system corpus, plus
every family of a target compiler string classified ``intrinsic``, plus the family of an
``intrinsic-prefix`` string when it decomposes to the same family as the declared names it
begins (``air.atomic_fetch_add_explicit_texture_2d`` names ``atomic_fetch_add_explicit``;
``air.sample_texture`` names nothing). Target strings in the other classes are listed with
their decomposition for the coverage report. Families that only the generated
corpus declares are listed separately: they come from the newer compiler used to build the
generated corpus, not from the target.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

from airnames import decompose

COMPILER_BINARIES = {
    "MTLCompiler": "/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler",
    "libGPUCompiler.dylib": "/System/Library/PrivateFrameworks/GPUCompiler.framework/Libraries/libGPUCompiler.dylib",
    "libGPUCompilerImpl.dylib": "/System/Library/PrivateFrameworks/GPUCompiler.framework/Libraries/libGPUCompilerImpl.dylib",
    "libGPUCompilerImplLazy.dylib": "/System/Library/PrivateFrameworks/GPUCompiler.framework/Libraries/libGPUCompilerImplLazy.dylib",
}
BACKEND_BINARIES = {
    "AGXCompilerCore": "/System/Library/PrivateFrameworks/AGXCompilerCore.framework/AGXCompilerCore",
}
AIR_STRING = re.compile(rb"air\.[A-Za-z0-9_.]+")


def strings_of(path):
    with open(path, "rb") as fh:
        data = fh.read()
    out = set()
    for m in AIR_STRING.finditer(data):
        s = m.group(0).decode("ascii").rstrip(".")
        out.add(s)
    return out


def extract_binaries(root, dest):
    ipsw = shutil.which("ipsw")
    if not ipsw:
        sys.exit("error: ipsw not found (needed to extract the compiler from the shared cache; or pass --binaries)")
    cache = os.path.join(root, "System/Library/Caches/com.apple.dyld/dyld_shared_cache_arm64")
    os.makedirs(dest, exist_ok=True)
    for name, image in {**COMPILER_BINARIES, **BACKEND_BINARIES}.items():
        if os.path.exists(os.path.join(dest, name)):
            continue
        proc = subprocess.run([ipsw, "dyld", "extract", cache, image, "--output", dest],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(os.path.join(dest, name)):
            sys.exit(f"error: ipsw could not extract {image}: {proc.stderr.strip()[:300]}")


def load_survey(path):
    with open(path, encoding="utf-8") as fh:
        survey = json.load(fh)
    callees = {n for n in survey["callees"] if n.startswith("air.")}
    metadata = set(survey["metadata_strings"]) | {n for n in survey["named_metadata"] if n.startswith("air.")}
    return survey, callees, metadata


def is_component_prefix(prefix, names_sorted):
    """True when some name starts with ``prefix`` followed by '.' or '_' (or equals it)."""
    import bisect
    k = bisect.bisect_left(names_sorted, prefix)
    while k < len(names_sorted) and names_sorted[k].startswith(prefix):
        rest = names_sorted[k][len(prefix):]
        if rest == "" or rest[0] in "._":
            return True
        k += 1
    return False


def prefix_names_family(prefix, names_sorted):
    import bisect
    family = decompose(prefix)[0]
    k = bisect.bisect_left(names_sorted, prefix)
    while k < len(names_sorted) and names_sorted[k].startswith(prefix):
        rest = names_sorted[k][len(prefix):]
        if (rest == "" or rest[0] in "._") and decompose(names_sorted[k])[0] == family:
            return True
        k += 1
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_json")
    parser.add_argument("--survey", required=True, help="survey.json of the system corpus")
    parser.add_argument("--generated-survey", help="survey.json of the generated corpus")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root")
    group.add_argument("--binaries")
    args = parser.parse_args()

    bindir = args.binaries
    if args.root:
        bindir = os.path.join(os.path.dirname(os.path.abspath(args.out_json)), "target-bin")
        extract_binaries(args.root, bindir)

    target = {}
    for name in list(COMPILER_BINARIES) + list(BACKEND_BINARIES):
        path = os.path.join(bindir, name)
        if not os.path.exists(path):
            sys.exit(f"error: {path} missing")
        for s in strings_of(path):
            target.setdefault(s, set()).add(name)

    _, sys_callees, sys_meta = load_survey(args.survey)
    gen_callees, gen_meta = set(), set()
    if args.generated_survey:
        _, gen_callees, gen_meta = load_survey(args.generated_survey)
    declared = sorted(sys_callees | gen_callees)
    metadata = sys_meta | gen_meta

    strings = {}
    for s, owners in sorted(target.items()):
        compiler = sorted(o for o in owners if o in COMPILER_BINARIES)
        backend = sorted(o for o in owners if o in BACKEND_BINARIES)
        if s in sys_callees or s in gen_callees:
            kind = "intrinsic"
        elif s in metadata:
            kind = "metadata"
        elif is_component_prefix(s, declared):
            kind = "intrinsic-prefix"
        else:
            kind = "unclassified"
        strings[s] = {"kind": kind, "compiler": compiler, "backend": backend}

    families = {}

    def note(name, source):
        d = decompose(name)
        if d is None:
            return
        entry = families.setdefault(d[0], {"system": 0, "target": 0, "generated": 0, "examples": []})
        entry[source] += 1
        if len(entry["examples"]) < 3 and name not in entry["examples"]:
            entry["examples"].append(name)

    for name in sorted(sys_callees):
        note(name, "system")
    for s, info in strings.items():
        if not info["compiler"]:
            continue
        if info["kind"] == "intrinsic":
            note(s, "target")
        elif info["kind"] == "intrinsic-prefix" and prefix_names_family(s, declared):
            # The front end composes names from such stems; a stem names the family when it
            # decomposes to the same family as the declared names it begins.
            note(s, "target")
    for name in sorted(gen_callees):
        note(name, "generated")

    derived = sorted(f for f, e in families.items() if e["system"] or e["target"])
    generated_only = sorted(f for f, e in families.items() if not (e["system"] or e["target"]))
    out = {
        "method": "families of air.* functions declared in the system corpus, plus families of target "
                  "compiler strings classified as intrinsics",
        "families": derived,
        "generated_only_families": generated_only,
        "family_sources": dict(sorted(families.items())),
        "target_strings": strings,
        "counts": {
            "target_strings": len(strings),
            "intrinsic": sum(1 for s in strings.values() if s["kind"] == "intrinsic"),
            "intrinsic_prefix": sum(1 for s in strings.values() if s["kind"] == "intrinsic-prefix"),
            "metadata": sum(1 for s in strings.values() if s["kind"] == "metadata"),
            "unclassified": sum(1 for s in strings.values() if s["kind"] == "unclassified"),
            "families": len(derived),
        },
    }
    tmp = args.out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, args.out_json)
    print(" ".join(f"{k}={v}" for k, v in out["counts"].items()) + f" generated_only={len(generated_only)}")


if __name__ == "__main__":
    main()
