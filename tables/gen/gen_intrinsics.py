#!/usr/bin/env python3
"""Generate tables/intrinsics.toml: the air.* name grammar and the handling of each name.

Function names: the stems are the names of the target compiler's AIR function tables,
with their overload slots and the AIR versions whose table lists them (measure/
air_grammar.py writes them to air-functions.json in the cache); a function name is a stem
followed by one type token per slot, from the slot's domain (gen/families.toml
[[domain]]). Every stem is classified by the families' stem patterns.

Other names: the air.* strings of the target's compiler images (gen/frontend_names.py)
that are no function name, and the metadata strings and named metadata of the system
corpus (gen/survey_corpus.py), the generated corpus (gen/generated_names.py) and the
probe measurement (measure/air_probe.toml). Every such name is classified by the
families' match patterns and listed with its kind: metadata (used as metadata by some
module), prefix (a proper prefix of a stem) or string (neither).

Every function name of those sources (the corpora's declared air.* functions, the probe's
calls, the images' complete names) must be accepted by the grammar; a name the grammar
refuses, or a stem or name that matches no family or more than one, stops generation.

Usage: gen_intrinsics.py --cache DIR [--out PATH]
"""

import argparse
import bisect
import json
import os
import re
import sys
import tomllib

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import airgrammar  # noqa: E402
import tomlout  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)
DECISIONS = {"native", "numerics", "lowering", "emulation", "metadata", "refusal",
             "device-unsupported", "not-from-msl", "open"}
SLOT_KINDS = ["any", "anyint", "anyfloat", "anyvector", "anypointer"]
INPUTS = ["frontend-names.json", "survey-root.json", "air-functions.json", "generated-names.json"]
METADATA_CASE = "module-metadata"


def load_families():
    with open(os.path.join(HERE, "families.toml"), "rb") as f:
        d = tomllib.load(f)
    fams, doms = d["family"], d["domain"]
    ids = set()
    for fam in fams:
        if fam["id"] in ids:
            raise SystemExit("duplicate family %s" % fam["id"])
        ids.add(fam["id"])
        if fam["decision"] not in DECISIONS:
            raise SystemExit("%s: unknown decision %r" % (fam["id"], fam["decision"]))
        if not fam.get("stem") and not fam.get("match"):
            raise SystemExit("%s: neither stem nor match patterns" % fam["id"])
        fam["_stem"] = [re.compile(p) for p in fam.get("stem", [])]
        fam["_match"] = [re.compile(p) for p in fam.get("match", [])]
    dom_ids = {d["id"] for d in doms}
    defaults = {}
    for d in doms:
        for kind in d.get("default_for", []):
            if kind not in SLOT_KINDS or kind in defaults:
                raise SystemExit("domain %s: default for %s is unknown or taken" % (d["id"], kind))
            defaults[kind] = d["id"]
        for _, target in d.get("reclassify", []):
            if target not in ids:
                raise SystemExit("domain %s reclassifies to unknown family %s" % (d["id"], target))
        if "spaces" in d and d["widths"] != [1]:
            raise SystemExit("domain %s: pointer domains take scalar pointees" % d["id"])
    if sorted(defaults) != sorted(SLOT_KINDS):
        raise SystemExit("slot kinds without a default domain: %s" % sorted(set(SLOT_KINDS) - set(defaults)))
    for fam in fams:
        for kind, dom in fam.get("domains", {}).items():
            if kind not in SLOT_KINDS or dom not in dom_ids:
                raise SystemExit("%s: domain %s for slot kind %s is unknown" % (fam["id"], dom, kind))
    return fams, doms


def unique(fams, key, name):
    hits = [f["id"] for f in fams if any(r.fullmatch(name) for r in f[key])]
    if len(hits) != 1:
        raise SystemExit("%s matches the %s patterns of %s" % (name, key[1:], hits or "no family"))
    return hits[0]


def load_probe():
    with open(os.path.join(TABLES, "measure", "air_probe.toml"), "rb") as f:
        d = tomllib.load(f)
    cases = {}
    for c in d["case"]:
        cases[c["id"]] = sorted({n for n in c["fast"] + c["precise"] if n.startswith("air.")})
    return d["meta"], cases


def air_range(tags, order):
    idx = sorted(order.index(t) for t in tags)
    if idx != list(range(idx[0], idx[-1] + 1)):
        raise SystemExit("AIR versions %s are not contiguous" % tags)
    return order[idx[0]] if len(idx) == 1 else "%s-%s" % (order[idx[0]], order[idx[-1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=os.environ.get("TABLES_CACHE"))
    ap.add_argument("--out", default=os.path.join(TABLES, "intrinsics.toml"))
    a = ap.parse_args()
    if not a.cache:
        raise SystemExit("a cache directory is required (--cache or TABLES_CACHE)")
    inputs = {}
    for name in INPUTS:
        with open(os.path.join(a.cache, name)) as f:
            inputs[name] = json.load(f)
    frontend, survey = inputs["frontend-names.json"], inputs["survey-root.json"]
    airfn, generated = inputs["air-functions.json"], inputs["generated-names.json"]
    fams, doms = load_families()
    probe_meta, probe = load_probe()

    # Stems: every table's names, with their slots and AIR versions.
    order = sorted(airfn["tables"], key=lambda t: tuple(int(x) for x in t.split(".")))
    stems = {}
    for tag in order:
        for name, slots in airfn["tables"][tag]["slots"].items():
            row = stems.setdefault(name, {"slots": slots, "air": []})
            if row["slots"] != slots:
                raise SystemExit("%s has slots %s in one table and %s in another" % (name, row["slots"], slots))
            row["air"].append(tag)
    stem_list = sorted(stems, key=lambda s: s.encode())
    functions = {}
    for name in stem_list:
        r = stems[name]
        functions[name] = {"name": name, "family": unique(fams, "_stem", name),
                           "slots": r["slots"] or None, "air": air_range(r["air"], order)}

    # The grammar, as a consumer reads it back from the table.
    domain_rows = []
    for d in doms:
        out = dict(d)
        out["tokens"] = airgrammar.domain_regex(d)
        domain_rows.append(out)
    grammar = airgrammar.Grammar({
        "grammar": {"air_versions": order}, "domain": [dict(d) for d in domain_rows], "family": fams,
        "function": [dict(r, slots=r["slots"] or []) for r in functions.values()],
    })

    # Every function name of the sources must be accepted; the other strings are names.
    evidence = {"system": {}, "generated": {}, "probe": {}, "frontend": {}}
    refused = []

    def accept(n, src, where):
        try:
            stem, _, _ = grammar.function(n)
        except airgrammar.Refused as e:
            refused.append((n, where, str(e)))
            return
        evidence[src].setdefault(stem, set()).add(n)

    for n in survey["declared_callees"]:
        if n.startswith("air."):
            accept(n, "system", "system corpus function")
    for n in generated["functions"]:
        accept(n, "generated", "generated corpus function")
    for cid, names in probe.items():
        if cid != METADATA_CASE:
            for n in names:
                accept(n, "probe", "probe case %s" % cid)
    metadata_sources = {}
    for n in survey["metadata_strings"]:
        if n.startswith("air."):
            metadata_sources.setdefault(n, set()).add("system")
    for n in survey["named_metadata"]:
        if n.startswith("air."):
            metadata_sources.setdefault(n, set()).add("system")
    for n in generated["metadata_strings"] + generated["named_metadata"]:
        metadata_sources.setdefault(n, set()).add("generated")
    for n in probe[METADATA_CASE]:
        metadata_sources.setdefault(n, set()).add("probe")
    # A metadata use is a name of its own, also when the string names a function (air.any
    # is the relational function and the depth qualifier).
    others = {}
    for n, srcs in metadata_sources.items():
        others.setdefault(n, set()).update(srcs)
    for img in frontend["images"].values():
        for n in img["names"]:
            if n in functions:
                continue
            try:
                grammar.function(n)
            except airgrammar.Refused:
                others.setdefault(n, set()).add("frontend")
                continue
            accept(n, "frontend", "front-end string")
    if refused:
        for n, where, why in refused:
            print("REFUSED %s (%s): %s" % (n, where, why))
        sys.exit(1)

    encoded = [s.encode() for s in stem_list]

    def is_prefix(n):
        k = bisect.bisect_left(encoded, n.encode())
        return k < len(encoded) and encoded[k].startswith(n.encode()) and encoded[k] != n.encode()

    names = []
    for n in sorted(others, key=lambda s: s.encode()):
        kind = "metadata" if n in metadata_sources else ("prefix" if is_prefix(n) else "string")
        row = {"name": n, "family": unique(fams, "_match", n), "kind": kind, "sources": sorted(others[n])}
        modules = survey["metadata_strings"].get(n) or survey["named_metadata"].get(n)
        if modules:
            row["corpus_modules"] = modules
        names.append(row)

    # Evidence per stem: the distinct names of each source with that stem.
    for name, row in functions.items():
        for src in ("system", "generated", "probe"):
            if name in evidence[src]:
                row[src] = len(evidence[src][name])

    # Probe cases cited by a family must emit a name of the family.
    name_family = {r["name"]: r["family"] for r in names}
    for fam in fams:
        for cid in fam.get("probe", []):
            if cid not in probe:
                raise SystemExit("%s cites unknown probe case %s" % (fam["id"], cid))
            if cid == METADATA_CASE:
                got = {name_family[n] for n in probe[cid]}
            else:
                got = {grammar.function(n)[2] for n in probe[cid]}
            if fam["id"] not in got:
                raise SystemExit("%s: probe case %s emitted no name of the family (%r)" % (fam["id"], cid, probe[cid]))
    functions_by_family, names_by_family = {}, {}
    for r in functions.values():
        functions_by_family[r["family"]] = functions_by_family.get(r["family"], 0) + 1
    for r in names:
        names_by_family[r["family"]] = names_by_family.get(r["family"], 0) + 1
    for fam in fams:
        if fam.get("stem") and not functions_by_family.get(fam["id"]):
            raise SystemExit("family %s: its stem patterns match no stem" % fam["id"])
        if fam.get("match") and not names_by_family.get(fam["id"]):
            raise SystemExit("family %s: its match patterns match no name" % fam["id"])

    def count(src):
        return sum(len(v) for v in evidence[src].values())

    other = sorted(n for n in survey["declared_callees"] if not n.startswith("air."))
    dispatch = ["%s: AIR %s table" % (d["version"], d["table"]) for d in airfn["dispatch"]]
    dispatch.append("below %s: AIR %s table" % (airfn["dispatch"][-1]["version"], airfn["default"]))
    parts = [
        "# Generated by tables/gen/gen_intrinsics.py from the target's AIR function tables\n"
        "# (measure/air_grammar.py), the target's compiler strings (gen/frontend_names.py), the\n"
        "# system corpus survey (gen/survey_corpus.py), the generated corpus\n"
        "# (gen/generated_names.py), the measurement tables/measure/air_probe.toml,\n"
        "# gen/families.toml and gen/intrinsics_questions.toml. Do not edit by hand.\n",
        tomlout.table("meta", {
            "target": "iPadOS 17.7.11 (21H461) dyld shared cache %s" % frontend["cache_uuid"],
            "frontend_images": sorted(frontend["images"]),
            "extraction": frontend["ipsw"],
            "probe": "%s; %s; %s" % (probe_meta["compiler"], probe_meta["target"], probe_meta["host"]),
            "families": len(fams),
            "functions": len(functions),
            "names": len(names),
        }),
        tomlout.table("grammar", {
            "rule": "An air.* function name is accepted as the target's compiler accepts it (measurement air-grammar): the module's AIR version selects a table (dispatch); llvm_utils::lookupIntrinsicByName finds a [[function]] of that table (its air field lists the version) whose name equals the function name or is followed in it by a dot; the function is overloaded exactly when the name extends it; and the name is the [[function]] name followed by one dot and type token per slot, in slot order. A token must be in the slot's [[domain]]: the domain the [[function]]'s family names for the slot kind in domains, else the kind's default domain. The name's family is the [[function]]'s, unless the domain of one of its tokens reclassifies that token's element type (the earliest listed reclassification wins). Any other air.* name is accepted only as a [[name]] row, whose family its match patterns give. Every other name is a Refusal naming it (plan section 5.1 stage 4).",
            "air_versions": order,
            "dispatch": dispatch,
            "image_sha256": airfn["image_sha256"],
            "unverified": "AIRFunctionInfo1d8 validates clz and ctz with a rule of its own for AIR 1.8 to 1.10, which is not decoded: such a module's clz and ctz calls are refused. Which table the target checks a module without an air.version record against is not established; the system corpus's modules without one all check against the AIR 2.6 table.",
            "system_functions": count("system"),
            "generated_functions": count("generated"),
            "generated_corpus": "%s; %s; %d accepted probes; results.json sha256 %s" % (
                generated["compiler"], generated["target"], generated["accepted_probes"], generated["results_sha256"]),
            "probe_functions": count("probe"),
            "frontend_functions": count("frontend"),
        }),
        tomlout.table("survey", {
            "index_sha256": survey["index_sha256"],
            "files": survey["files"],
            "containers": ["%s: %d" % kv for kv in survey["containers"].items()],
            "module_entries": survey["module_entries"],
            "modules": survey["modules"],
            "disassembly_failures": len(survey["disassembly_failures"]),
            "triples": ["%s: %d" % (k, v) for k, v in survey["triples"].items()],
            "language_versions": ["%s: %d" % (k if k != "null" else "none recorded", v)
                                  for k, v in survey["language_versions"].items()],
            "non_air_callees": other,
        }),
    ]
    for d in domain_rows:
        parts.append(tomlout.array_item("domain", d))
    for fam in fams:
        out = {k: v for k, v in fam.items() if not k.startswith("_")}
        out["functions"] = functions_by_family.get(fam["id"])
        out["names"] = names_by_family.get(fam["id"])
        parts.append(tomlout.array_item("family", out))
    for name in stem_list:
        parts.append(tomlout.array_item("function", functions[name]))
    for r in names:
        parts.append(tomlout.array_item("name", r))
    with open(os.path.join(HERE, "intrinsics_questions.toml"), "rb") as f:
        for q in tomllib.load(f).get("question", []):
            parts.append(tomlout.array_item("question", q))
    with open(a.out, "w") as f:
        f.write("\n".join(parts))
    print("wrote %s (%d functions, %d names, %d families)" % (a.out, len(functions), len(names), len(fams)))


if __name__ == "__main__":
    main()
