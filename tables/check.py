#!/usr/bin/env python3
"""Check the AirWgsl specification tables.

Fails (exit status 1) when any table cell is empty, any row lacks a citation (open
questions instead need a reason), any citation does not verify against its pinned
source, or a generated table differs from what its generator produces now.

Usage: check.py --cache DIR
"""

import argparse
import difflib
import os
import subprocess
import sys
import tempfile
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(HERE, "gen"))
import citecheck  # noqa: E402
import specdoc  # noqa: E402

GENERATED = [
    ("numerics.toml", "gen_numerics.py"),
]

ROW_ARRAYS = {
    "numerics.toml": ["rule", "generic", "row", "question"],
}


def regenerate(cache, name, script):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, name)
        subprocess.run([sys.executable, "-B", "-I", os.path.join(HERE, "gen", script), "--cache", cache,
                        "--out", out], check=True, stdout=subprocess.DEVNULL)
        with open(out, encoding="utf-8") as f:
            return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=os.environ.get("TABLES_CACHE"))
    a = ap.parse_args()
    if not a.cache:
        raise SystemExit("a cache directory is required (--cache or TABLES_CACHE)")
    sources = specdoc.load_sources()
    chk = citecheck.Checker(sources, a.cache)

    for name, script in GENERATED:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            chk.fail(name, "missing")
            continue
        with open(path, encoding="utf-8") as f:
            have = f.read()
        want = regenerate(a.cache, name, script)
        if have != want:
            diff = "".join(list(difflib.unified_diff(have.splitlines(True), want.splitlines(True),
                                                     name, name + " (regenerated)"))[:40])
            chk.fail(name, "differs from its generator output:\n" + diff)

    for name, arrays in ROW_ARRAYS.items():
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            data = tomllib.load(f)
        chk.no_empty(name + ".meta", data.get("meta", {}))
        for arr in arrays:
            for i, row in enumerate(data.get(arr, [])):
                label = row.get("id") or row.get("name") or str(i)
                chk.row("%s %s[%s]" % (name, arr, label), row)

    for e in chk.errors:
        print("FAIL " + e)
    print("%d citations checked, %d failures" % (chk.checked, len(chk.errors)))
    return 1 if chk.errors else 0


if __name__ == "__main__":
    sys.exit(main())
