#!/usr/bin/env python3
"""Build both AIR corpora and the coverage report in one run.

Usage:

    pipeline.py ROOT OUT [--jobs N] [--skip-system] [--d2 TABLE]

Steps (each one is also a script that can be run on its own):

1. ``tools/extract-system-metallibs.py ROOT OUT/system``: the system corpus and its index.
2. ``corpus/survey.py`` over the system corpus: ``OUT/system/survey.json``.
3. ``corpus/generate.py OUT/gen``: the generated MSL sources.
4. ``corpus/build.py OUT/gen OUT/build``: AIR at every language version.
5. ``corpus/survey.py`` over the generated corpus: ``OUT/build/survey.json``.
6. ``corpus/families.py OUT/families.json --root ROOT``: the derived family list.
7. ``corpus/coverage.py OUT/report``: the coverage report (``coverage.md``, ``coverage.json``).

OUT holds Apple's code (the system corpus and the extracted compiler binaries) and must stay
local. ``--skip-system`` reuses an existing ``OUT/system`` (and its ``survey.json`` when present).
``--d2`` passes the intrinsic table to the coverage report (default: the report's own default,
``tables/intrinsics.toml`` of this repository).
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
HEADERS = "System/Library/PrivateFrameworks/GPUCompiler.framework/Libraries/lib/clang"


def run(args):
    print("+", " ".join(args), flush=True)
    proc = subprocess.run(["nice", "-n", "10", sys.executable] + args)
    if proc.returncode != 0:
        sys.exit(f"error: step failed ({proc.returncode}): {' '.join(args)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root")
    parser.add_argument("out")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    parser.add_argument("--skip-system", action="store_true")
    parser.add_argument("--d2", help="tables/intrinsics.toml to read (passed to coverage.py)")
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    jobs = ["--jobs", str(args.jobs)]
    system = os.path.join(out, "system")
    if not args.skip_system:
        run([os.path.join(TOOLS, "extract-system-metallibs.py"), args.root, system] + jobs)
    if not (args.skip_system and os.path.exists(os.path.join(system, "survey.json"))):
        run([os.path.join(HERE, "survey.py"), os.path.join(system, "modules.tsv"), os.path.join(system, "ll"),
             os.path.join(system, "survey.json")] + jobs)
    run([os.path.join(HERE, "generate.py"), os.path.join(out, "gen")])
    run([os.path.join(HERE, "build.py"), os.path.join(out, "gen"), os.path.join(out, "build")] + jobs)
    run([os.path.join(HERE, "survey.py"), os.path.join(out, "build", "modules.tsv"), os.path.join(out, "build", "ll"),
         os.path.join(out, "build", "survey.json")] + jobs)
    run([os.path.join(HERE, "families.py"), os.path.join(out, "families.json"), "--root", args.root,
         "--survey", os.path.join(system, "survey.json"),
         "--generated-survey", os.path.join(out, "build", "survey.json")])
    clang = os.path.join(args.root, HEADERS)
    versions = sorted(os.listdir(clang)) if os.path.isdir(clang) else []
    if len(versions) != 1:
        sys.exit(f"error: expected one compiler version under {clang}, found {versions}")
    run([os.path.join(HERE, "coverage.py"), os.path.join(out, "report"),
         "--build", os.path.join(out, "build"), "--gen", os.path.join(out, "gen"),
         "--system-survey", os.path.join(system, "survey.json"),
         "--generated-survey", os.path.join(out, "build", "survey.json"),
         "--families", os.path.join(out, "families.json"),
         "--target-headers", os.path.join(clang, versions[0], "include", "metal")]
        + (["--d2", args.d2] if args.d2 else []))


if __name__ == "__main__":
    main()
