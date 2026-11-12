#!/usr/bin/env python3
"""Generate tables/intrinsics.toml: the closed set of air.* names and their handling.

Rows are the union of
  - the air.* identifiers in the target's compiler images (frontend_names.py output),
  - the air.* functions declared by the AIR modules of the target's metallibs, and the
    air.* metadata strings and named metadata of those modules (survey_corpus.py output),
  - the air.* functions this Mac's front end emits for the MSL probes (measure/air_probe.toml).
Each name is classified by the families of gen/families.toml; a name matching no family
or more than one stops generation. A probe name that is in neither target source is
listed as probe-only, so it never enters the accepted set unnoticed.

Usage: gen_intrinsics.py --cache DIR [--frontend JSON] [--survey JSON] [--out PATH]
"""

import argparse
import json
import os
import re
import sys
import tomllib

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tomlout  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)
DECISIONS = {"native", "numerics", "lowering", "emulation", "metadata", "refusal",
             "device-unsupported", "not-from-msl", "open"}


def load_families():
    with open(os.path.join(HERE, "families.toml"), "rb") as f:
        fams = tomllib.load(f)["family"]
    ids = set()
    for fam in fams:
        if fam["id"] in ids:
            raise SystemExit("duplicate family %s" % fam["id"])
        ids.add(fam["id"])
        if fam["decision"] not in DECISIONS:
            raise SystemExit("%s: unknown decision %r" % (fam["id"], fam["decision"]))
        fam["_re"] = [re.compile(p) for p in fam["match"]]
    return fams


def classify(fams, name):
    hits = [f["id"] for f in fams if any(r.fullmatch(name) for r in f["_re"])]
    return hits


def load_probe():
    with open(os.path.join(TABLES, "measure", "air_probe.toml"), "rb") as f:
        d = tomllib.load(f)
    cases = {}
    for c in d["case"]:
        cases[c["id"]] = sorted({n for n in c["fast"] + c["precise"] if n.startswith("air.")})
    return d["meta"], cases


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=os.environ.get("TABLES_CACHE"))
    ap.add_argument("--frontend")
    ap.add_argument("--survey")
    ap.add_argument("--out", default=os.path.join(TABLES, "intrinsics.toml"))
    ap.add_argument("--report", action="store_true", help="list unclassified names and exit")
    a = ap.parse_args()
    if not a.cache:
        raise SystemExit("a cache directory is required (--cache or TABLES_CACHE)")
    frontend_path = a.frontend or os.path.join(a.cache, "frontend-names.json")
    survey_path = a.survey or os.path.join(a.cache, "survey-root.json")
    with open(frontend_path) as f:
        frontend = json.load(f)
    with open(survey_path) as f:
        survey = json.load(f)
    fams = load_families()
    probe_meta, probe = load_probe()

    sources = {}
    def add(name, src):
        sources.setdefault(name, set()).add(src)
    for img in frontend["images"].values():
        for n in img["names"]:
            add(n, "frontend")
    for n in survey["declared_callees"]:
        if n.startswith("air."):
            add(n, "corpus-function")
    for n in survey["metadata_strings"]:
        if n.startswith("air."):
            add(n, "corpus-metadata")
    for n in survey["named_metadata"]:
        add(n, "corpus-named-metadata")
    for cid, names in probe.items():
        for n in names:
            add(n, "probe")

    unclassified, ambiguous = [], []
    rows = []
    for name in sorted(sources):
        hits = classify(fams, name)
        if not hits:
            unclassified.append(name)
            continue
        if len(hits) > 1:
            ambiguous.append((name, hits))
            continue
        src = sorted(sources[name])
        row = {"name": name, "family": hits[0], "sources": src}
        if src == ["probe"]:
            row["note"] = "emitted by this Mac's front end only; not in the target's compiler strings or corpus"
        uses = survey["declared_callees"].get(name) or survey["metadata_strings"].get(name) \
            or survey["named_metadata"].get(name)
        if uses:
            row["corpus_modules"] = uses
        rows.append(row)
    if a.report or unclassified or ambiguous:
        for n in unclassified:
            print("UNCLASSIFIED " + n)
        for n, h in ambiguous:
            print("AMBIGUOUS %s %s" % (n, h))
        print("%d names, %d unclassified, %d ambiguous" % (len(sources), len(unclassified), len(ambiguous)))
        if unclassified or ambiguous:
            sys.exit(1)
        return

    by_family = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r["name"])
    for fam in fams:
        for cid in fam.get("probe", []):
            if cid not in probe:
                raise SystemExit("%s cites unknown probe case %s" % (fam["id"], cid))
            if not any(any(rx.fullmatch(n) for rx in fam["_re"]) for n in probe[cid]):
                raise SystemExit("%s: probe case %s emitted no name of the family (%r)" % (fam["id"], cid, probe[cid]))
        if fam["id"] not in by_family:
            raise SystemExit("family %s matches no name" % fam["id"])

    other = sorted(n for n in survey["declared_callees"] if not n.startswith("air."))
    parts = [
        "# Generated by tables/gen/gen_intrinsics.py from the target's compiler strings\n"
        "# (gen/frontend_names.py), the target's metallib survey (gen/survey_corpus.py), the\n"
        "# measurement tables/measure/air_probe.toml, gen/families.toml and\n"
        "# gen/intrinsics_questions.toml. Do not edit by hand.\n",
        tomlout.table("meta", {
            "target": "iPadOS 17.7.11 (21H461) dyld shared cache %s" % frontend["cache_uuid"],
            "frontend_images": sorted(frontend["images"]),
            "extraction": frontend["ipsw"],
            "probe": "%s; %s; %s" % (probe_meta["compiler"], probe_meta["target"], probe_meta["host"]),
            "grammar": "Closed: a name outside [[name]] is a Refusal naming it (plan section 5.1 stage 4). Families give the handling; numerics decisions are in numerics.toml.",
            "families": len(fams),
            "names": len(rows),
        }),
        tomlout.table("survey", {
            "metallibs": survey["metallibs"],
            "modules": survey["modules"],
            "disassembly_failures": len(survey["disassembly_failures"]),
            "triples": ["%s: %d" % (k, v) for k, v in survey["triples"].items()],
            "language_versions": ["%s: %d" % (k if k != "null" else "none recorded", v)
                                  for k, v in survey["language_versions"].items()],
            "non_air_callees": other,
        }),
    ]
    for fam in fams:
        out = {k: v for k, v in fam.items() if not k.startswith("_")}
        out["names"] = len(by_family[fam["id"]])
        parts.append(tomlout.array_item("family", out))
    for r in rows:
        parts.append(tomlout.array_item("name", r))
    with open(os.path.join(HERE, "intrinsics_questions.toml"), "rb") as f:
        for q in tomllib.load(f).get("question", []):
            parts.append(tomlout.array_item("question", q))
    with open(a.out, "w") as f:
        f.write("\n".join(parts))
    print("wrote %s (%d names, %d families)" % (a.out, len(rows), len(fams)))


if __name__ == "__main__":
    main()
