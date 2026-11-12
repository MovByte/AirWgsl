#!/usr/bin/env python3
"""Check the AirWgsl specification tables.

Fails (exit status 1) when any table cell is empty, any row lacks a citation (open
questions instead need a reason), any citation does not verify against its pinned
source, a generated table differs from what its generator produces now, the air.* name
grammar of intrinsics.toml is inconsistent (a stem or name outside exactly one family, a
name of the system or generated corpus that the grammar refuses), or the mutation test
finds a near-miss name the grammar accepts (gen/grammar_mutations.py).

Usage: check.py --cache DIR
"""

import argparse
import bisect
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(HERE, "gen"))
import airgrammar  # noqa: E402
import citecheck  # noqa: E402
import grammar_mutations  # noqa: E402
import specdoc  # noqa: E402

GENERATED = [
    ("numerics.toml", "gen_numerics.py"),
    ("intrinsics.toml", "gen_intrinsics.py"),
]
# Generators whose inputs come from the target root and the corpora (cached by
# gen/frontend_names.py, gen/survey_corpus.py, measure/air_grammar.py and
# gen/generated_names.py); without those inputs in the cache they cannot be re-run.
TARGET_INPUTS = {"intrinsics.toml": ["frontend-names.json", "survey-root.json", "air-functions.json",
                                     "generated-names.json"]}

ROW_ARRAYS = {
    "numerics.toml": ["rule", "generic", "row", "question"],
    "intrinsics.toml": ["domain", "family", "question"],
}
NAME_KINDS = {"metadata", "prefix", "string"}
SLOT_KINDS = {"any", "anyint", "anyfloat", "anyvector", "anypointer"}


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


def check_grammar(chk, data, cache):
    """The grammar's rows agree with each other, and every corpus name is accepted."""
    start = len(chk.errors)
    fams = {f["id"]: f for f in data.get("family", [])}
    stem_re = {fid: [re.compile(p) for p in f.get("stem", [])] for fid, f in fams.items()}
    match_re = {fid: [re.compile(p) for p in f.get("match", [])] for fid, f in fams.items()}
    doms = {d["id"]: d for d in data.get("domain", [])}
    defaults = {}
    for d in doms.values():
        where = "intrinsics.toml domain[%s]" % d["id"]
        if d.get("tokens") != airgrammar.domain_regex(d):
            chk.fail(where, "tokens %r is not the domain's regular expression" % d.get("tokens"))
        for kind in d.get("default_for", []):
            if kind in defaults:
                chk.fail(where, "second default domain for %s" % kind)
            defaults[kind] = d["id"]
        for _, target in d.get("reclassify", []):
            if target not in fams:
                chk.fail(where, "reclassifies to unknown family %s" % target)
    if set(defaults) != SLOT_KINDS:
        chk.fail("intrinsics.toml domain", "slot kinds without a default domain: %s" % sorted(SLOT_KINDS - set(defaults)))
    for f in fams.values():
        for kind, dom in f.get("domains", {}).items():
            if kind not in SLOT_KINDS or dom not in doms:
                chk.fail("intrinsics.toml family[%s]" % f["id"], "unknown domain %s for %s" % (dom, kind))
    versions = data.get("grammar", {}).get("air_versions", [])
    counts = {fid: [0, 0] for fid in fams}
    stems = []
    for row in data.get("function", []):
        where = "intrinsics.toml function[%s]" % row.get("name")
        chk.no_empty(where, row)
        hits = [fid for fid, pats in stem_re.items() if any(p.fullmatch(row["name"]) for p in pats)]
        if hits != [row.get("family")]:
            chk.fail(where, "family %r but stem patterns match %r" % (row.get("family"), hits))
        if not set(row.get("slots", [])) <= SLOT_KINDS:
            chk.fail(where, "unknown slot kinds %s" % row.get("slots"))
        try:
            if not set(airgrammar.parse_air_range(row["air"], versions)) <= set(versions):
                raise ValueError(row["air"])
        except ValueError:
            chk.fail(where, "AIR versions %r are not among %s" % (row.get("air"), versions))
        counts.setdefault(row.get("family"), [0, 0])[0] += 1
        stems.append(row["name"])
    stems.sort(key=lambda x: x.encode())
    encoded = [x.encode() for x in stems]
    for row in data.get("name", []):
        where = "intrinsics.toml name[%s]" % row.get("name")
        chk.no_empty(where, row)
        hits = [fid for fid, pats in match_re.items() if any(p.fullmatch(row["name"]) for p in pats)]
        if hits != [row.get("family")]:
            chk.fail(where, "family %r but match patterns match %r" % (row.get("family"), hits))
        if not row.get("sources"):
            chk.fail(where, "no source")
        kind = row.get("kind")
        k = bisect.bisect_left(encoded, row["name"].encode())
        prefix = k < len(encoded) and encoded[k].startswith(row["name"].encode()) and encoded[k] != row["name"].encode()
        corpus = bool(set(row.get("sources", [])) - {"frontend"})
        want = "metadata" if corpus else ("prefix" if prefix else "string")
        if kind not in NAME_KINDS or kind != want:
            chk.fail(where, "kind %r, but its sources and the stems make it %r" % (kind, want))
        if row.get("family") == "name-prefixes" and kind != "prefix":
            chk.fail(where, "in name-prefixes but not a prefix of a stem")
        counts.setdefault(row.get("family"), [0, 0])[1] += 1
    for f in fams.values():
        nf, nn = counts.get(f["id"], [0, 0])
        if (f.get("functions") or 0) != nf or (f.get("names") or 0) != nn:
            chk.fail("intrinsics.toml family[%s]" % f["id"], "functions = %r, names = %r but %d and %d rows" % (
                f.get("functions"), f.get("names"), nf, nn))
        if bool(f.get("stem")) != bool(nf) or bool(f.get("match")) != bool(nn):
            chk.fail("intrinsics.toml family[%s]" % f["id"], "a pattern list classifies no row")
    if len(chk.errors) > start:
        return None

    grammar = airgrammar.Grammar(data)
    observed = {}
    report = []
    paths = {p: os.path.join(cache, p) for p in ("survey-root.json", "generated-names.json")}
    if all(os.path.exists(p) for p in paths.values()):
        with open(paths["survey-root.json"]) as f:
            survey = json.load(f)
        with open(paths["generated-names.json"]) as f:
            generated = json.load(f)
        functions = {"system": [n for n in survey["declared_callees"] if n.startswith("air.")],
                     "generated": list(generated["functions"])}
        metadata = {"system": [n for n in list(survey["metadata_strings"]) + list(survey["named_metadata"]) if n.startswith("air.")],
                    "generated": generated["metadata_strings"] + generated["named_metadata"]}
        for src, names in functions.items():
            for n in names:
                try:
                    grammar.function(n)
                except airgrammar.Refused as e:
                    chk.fail("%s corpus function %s" % (src, n), "refused: %s" % e)
            observed.setdefault(src, []).extend(names)
            report.append("%d %s corpus functions" % (len(names), src))
        for src, names in metadata.items():
            for n in names:
                try:
                    grammar.name(n)
                except airgrammar.Refused as e:
                    chk.fail("%s corpus metadata %s" % (src, n), "refused: %s" % e)
            report.append("%d %s corpus metadata names" % (len(names), src))
        report = ["classified " + ", ".join(report)]
    else:
        report = ["SKIPPED corpus names (survey-root.json and generated-names.json not in the cache)"]
    stats = grammar_mutations.run(data, observed.get("system", []) + observed.get("generated", []))
    for m, op, why in stats["disagree"]:
        chk.fail("mutation %s (%s)" % (m, op), why)
    if not stats["refused"]:
        chk.fail("mutation test", "no near-miss names")
    report.append("mutation test: %d mutants of %d names, %d near misses refused, %d valid names" % (
        stats["mutants"], stats["bases"], stats["refused"], stats["valid"]))
    return report


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
    grammar_report = []

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
                chk.fail(name, "target inputs %s are not in the cache (run gen/frontend_names.py, "
                               "gen/survey_corpus.py, measure/air_grammar.py and gen/generated_names.py, "
                               "or pass --skip-target)" % ", ".join(needs))
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
            grammar_report = check_grammar(chk, data, a.cache) or []
        for arr in arrays:
            for i, row in enumerate(data.get(arr, [])):
                label = row.get("id") or row.get("name") or str(i)
                chk.row("%s %s[%s]" % (name, arr, label), row)

    for e in chk.errors:
        print("FAIL " + e)
    for line in grammar_report:
        print(line)
    for name in skipped:
        print("SKIPPED regeneration of %s" % name)
    print("%d citations checked, %d failures" % (chk.checked, len(chk.errors)))
    return 1 if chk.errors else 0


if __name__ == "__main__":
    sys.exit(main())
