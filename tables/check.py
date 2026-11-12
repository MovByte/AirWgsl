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
    ("intrinsics.toml", "gen_intrinsics.py"),
]
# Generators whose inputs come from the target root (cached by gen/frontend_names.py and
# gen/survey_corpus.py); without those inputs in the cache they cannot be re-run.
TARGET_INPUTS = {"intrinsics.toml": ["frontend-names.json", "survey-root.json"]}

ROW_ARRAYS = {
    "numerics.toml": ["rule", "generic", "row", "question"],
    "intrinsics.toml": ["family", "question"],
}


def load_measures():
    """{measurement id: {case id: text}} from the committed measurement results."""
    out = {}
    mdir = os.path.join(HERE, "measure")
    for name in sorted(os.listdir(mdir)):
        if not name.endswith(".toml"):
            continue
        with open(os.path.join(mdir, name), "rb") as f:
            d = tomllib.load(f)
        cases = {}
        for c in d.get("case", []):
            lines = []
            for k, v in c.items():
                lines.append("%s: %s" % (k, " ".join(str(x) for x in v) if isinstance(v, list) else v))
            cases[c["id"]] = "\n".join(lines)
        out[d["meta"]["id"]] = cases
    return out


def check_names(chk, data):
    """Every name belongs to exactly one family by the families' own patterns."""
    import re
    fams = {f["id"]: [re.compile(p) for p in f["match"]] for f in data.get("family", [])}
    counts = {}
    for row in data.get("name", []):
        where = "intrinsics.toml name[%s]" % row.get("name")
        chk.no_empty(where, row)
        hits = [fid for fid, pats in fams.items() if any(p.fullmatch(row["name"]) for p in pats)]
        if hits != [row.get("family")]:
            chk.fail(where, "family %r but patterns match %r" % (row.get("family"), hits))
        if not row.get("sources"):
            chk.fail(where, "no source")
        counts[row.get("family")] = counts.get(row.get("family"), 0) + 1
    for f in data.get("family", []):
        if counts.get(f["id"], 0) != f.get("names"):
            chk.fail("intrinsics.toml family[%s]" % f["id"], "names = %r but %d rows" % (f.get("names"), counts.get(f["id"], 0)))


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
    ap.add_argument("--skip-target", action="store_true",
                    help="do not re-run generators that need target-root inputs (reported as skipped)")
    a = ap.parse_args()
    if not a.cache:
        raise SystemExit("a cache directory is required (--cache or TABLES_CACHE)")
    sources = specdoc.load_sources()
    chk = citecheck.Checker(sources, a.cache, measures=load_measures())
    skipped = []

    for name, script in GENERATED:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            chk.fail(name, "missing")
            continue
        with open(path, encoding="utf-8") as f:
            have = f.read()
        needs = TARGET_INPUTS.get(name, [])
        if needs and (a.skip_target or not all(os.path.exists(os.path.join(a.cache, n)) for n in needs)):
            if not a.skip_target:
                chk.fail(name, "target inputs %s are not in the cache (run gen/frontend_names.py and "
                               "gen/survey_corpus.py, or pass --skip-target)" % ", ".join(needs))
            skipped.append(name)
            continue
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
        if name == "intrinsics.toml":
            check_names(chk, data)
        for arr in arrays:
            for i, row in enumerate(data.get(arr, [])):
                label = row.get("id") or row.get("name") or str(i)
                chk.row("%s %s[%s]" % (name, arr, label), row)

    for e in chk.errors:
        print("FAIL " + e)
    for name in skipped:
        print("SKIPPED regeneration of %s" % name)
    print("%d citations checked, %d failures" % (chk.checked, len(chk.errors)))
    return 1 if chk.errors else 0


if __name__ == "__main__":
    sys.exit(main())
