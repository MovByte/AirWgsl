#!/usr/bin/env python3
"""Coverage of the AIR intrinsic families by the generated corpus.

Usage:

    coverage.py OUT_DIR --build BUILD_DIR --gen GEN_DIR --system-survey SURVEY
                --generated-survey SURVEY --families FAMILIES_JSON
                [--d2 TABLE] [--target-headers DIR]

Inputs:

``--build``
    ``results.json`` of ``corpus/build.py``: per variant and language version, whether the
    compiler accepted it and which ``air.*`` functions it reaches.
``--gen``
    ``manifest.json`` of ``corpus/generate.py`` (the ``__metal_*`` entries each variant claims
    to reach).
``--system-survey``, ``--generated-survey``
    ``survey.json`` of the system and generated corpora (``corpus/survey.py``).
``--families``
    ``families.json`` of ``corpus/families.py``: the family list derived from the target
    compiler's strings and the system survey, and the classified target strings.
``--d2``
    Package D2's ``tables/intrinsics.toml`` (default: ``../tables/intrinsics.toml`` next to
    this directory). When it exists, its function families are the primary view and the
    derived list is the secondary view; when it does not, the derived list is the only view. A
    table that exists but does not have D2's ``[[family]]``/``[[name]]`` shape is an error.
``--target-headers``
    The target's MSL standard library headers. Every ``__metal_*`` function they call must be
    claimed by an accepted variant that emits AIR.

A *variant* of a family is one concrete AIR name decomposed by ``airnames.py`` into operation,
dimension, type signature and modifiers. Per family the report gives:

``accepted``
    Variants the compiler emitted for an accepted probe, with the language versions that
    accepted them: by measurement, the types and dimensions the compiler accepts for the
    family's MSL builtins. The generated corpus contains each of them.
``system``
    Variants the system corpus uses, and how many of them the generated corpus contains.
``listed``
    For the D2 view: names D2 lists for the family, by D2's source tags, and how many the
    generated corpus contains. For the derived view: target compiler strings in the family.

Names the generated corpus contains that D2's closed table does not list are reported: under
D2's grammar the translator would refuse them.

Writes ``OUT_DIR/coverage.json`` (complete data) and ``OUT_DIR/coverage.md`` (summary and every
uncovered item).
"""

import argparse
import collections
import hashlib
import json
import os
import re
import tomllib

from airnames import decompose


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_d2(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    data = tomllib.loads(raw.decode("utf-8"))
    if not isinstance(data.get("family"), list) or not isinstance(data.get("name"), list):
        raise SystemExit(f"error: {path} has no [[family]] and [[name]] arrays; the reader must be updated")
    families = {}
    for f in data["family"]:
        if "id" not in f or "match" not in f:
            raise SystemExit(f"error: {path}: family row without id or match: {f}")
        families[f["id"]] = {"kind": f.get("kind", ""), "decision": f.get("decision", ""),
                             "match": [re.compile(m) for m in f["match"]]}
    names = {}
    for n in data["name"]:
        if n.get("family") not in families:
            raise SystemExit(f"error: {path}: name {n.get('name')} names unknown family {n.get('family')}")
        names[n["name"]] = (n["family"], tuple(n.get("sources", ())))
    return families, names, hashlib.sha256(raw).hexdigest()


def variant_label(name, with_op):
    d = decompose(name)
    parts = ([d[0]] if with_op else []) + [p for p in (d[1], d[2], "+".join(d[3])) if p]
    return " ".join(parts) or "(none)"


def header_builtins(directory):
    calls = set()
    for fname in sorted(os.listdir(directory)):
        path = os.path.join(directory, fname)
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                calls |= set(re.findall(r"__metal_([A-Za-z0-9_]+)\s*\(", fh.read()))
    return calls


def view(family_ids, members, listed, gen_versions, sys_names, order, with_op):
    """Coverage of one family list. ``listed`` maps family -> {name: source tags}."""
    gen_names = set(gen_versions)
    per = {f: {"accepted": {}, "system": {}, "listed": {}} for f in family_ids}
    outside = {"generated": [], "system": []}
    for n in sorted(gen_names):
        ms = members(n)
        if not ms:
            outside["generated"].append(n)
        for f in ms:
            e = per[f]["accepted"].setdefault(variant_label(n, with_op), {"names": [], "versions": set()})
            e["names"].append(n)
            e["versions"] |= gen_versions[n]
    for n, modules in sorted(sys_names.items()):
        ms = members(n)
        if not ms:
            outside["system"].append(n)
        for f in ms:
            e = per[f]["system"].setdefault(variant_label(n, with_op), {"names": [], "modules": 0, "covered": False})
            e["names"].append(n)
            e["modules"] += modules
            e["covered"] = e["covered"] or n in gen_names
    for f in family_ids:
        for n, tags in sorted(listed.get(f, {}).items()):
            e = per[f]["listed"].setdefault(variant_label(n, with_op), {"names": [], "tags": set(), "covered": False})
            e["names"].append(n)
            e["tags"] |= set(tags)
            e["covered"] = e["covered"] or n in gen_names
    summary = []
    for f in family_ids:
        p = per[f]
        versions = set().union(*(e["versions"] for e in p["accepted"].values())) if p["accepted"] else set()
        by_tag = collections.defaultdict(lambda: [0, 0])
        for e in p["listed"].values():
            for t in e["tags"]:
                by_tag[t][1] += 1
                by_tag[t][0] += e["covered"]
        summary.append({
            "family": f, "accepted": len(p["accepted"]),
            "system": len(p["system"]), "system_covered": sum(e["covered"] for e in p["system"].values()),
            "listed": len(p["listed"]), "listed_covered": sum(e["covered"] for e in p["listed"].values()),
            "listed_by_source": {t: {"covered": c, "total": n} for t, (c, n) in sorted(by_tag.items())},
            "versions": sorted(versions, key=order.index),
        })
    detail = {}
    for f in family_ids:
        p = per[f]
        detail[f] = {
            "accepted": {k: {"names": v["names"], "versions": sorted(v["versions"], key=order.index)}
                         for k, v in sorted(p["accepted"].items())},
            "system": dict(sorted(p["system"].items())),
            "listed": {k: {"names": v["names"], "tags": sorted(v["tags"]), "covered": v["covered"]}
                       for k, v in sorted(p["listed"].items())},
        }
    return summary, detail, outside


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir")
    parser.add_argument("--build", required=True)
    parser.add_argument("--gen", required=True)
    parser.add_argument("--system-survey", required=True)
    parser.add_argument("--generated-survey", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--d2", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "tables", "intrinsics.toml"))
    parser.add_argument("--target-headers")
    args = parser.parse_args()

    results = load_json(os.path.join(args.build, "results.json"))
    manifest = load_json(os.path.join(args.gen, "manifest.json"))
    system = load_json(args.system_survey)
    generated = load_json(args.generated_survey)
    fam = load_json(args.families)
    order = list(results["versions"])

    gen_versions = collections.defaultdict(set)
    accepted_by_variant = collections.defaultdict(set)
    emitted_by_variant = collections.defaultdict(set)
    statuses = collections.Counter()
    for gname, entry in results["groups"].items():
        for version, res in entry["versions"].items():
            for index, rec in res["status"].items():
                statuses[rec["status"]] += 1
                if rec["status"] != "accepted":
                    continue
                accepted_by_variant[(gname, int(index))].add(version)
                for n in rec.get("intrinsics", []):
                    if n.startswith("air."):
                        gen_versions[n].add(version)
                        emitted_by_variant[(gname, int(index))].add(n)
    gen_names = set(gen_versions)
    sys_names = {n: v["modules"] for n, v in system["callees"].items() if n.startswith("air.")}

    views = {}
    # Derived view: the families of families.py, target strings as the listed names.
    derived_ids = list(fam["families"])
    derived_set = set(derived_ids)

    def derived_members(name):
        d = decompose(name)
        return [d[0]] if d and d[0] in derived_set else []

    derived_listed = collections.defaultdict(dict)
    for s, info in fam["target_strings"].items():
        if info["kind"] in ("intrinsic", "unclassified") and info["compiler"]:
            for f in derived_members(s):
                derived_listed[f][s] = ("target-" + info["kind"],)
    views["derived"] = {"source": f"derived list: {fam['method']}",
                        "result": view(derived_ids, derived_members, derived_listed, gen_versions, sys_names, order, False)}

    d2_unlisted = None
    if os.path.exists(args.d2):
        families, names, sha = load_d2(args.d2)
        function_ids = [f for f, info in families.items() if info["kind"] == "function"]

        def d2_members(name):
            if name in names:
                return [names[name][0]] if names[name][0] in function_ids else []
            return [f for f in function_ids if any(p.fullmatch(name) for p in families[f]["match"])]

        d2_listed = collections.defaultdict(dict)
        for n, (f, tags) in names.items():
            if f in function_ids:
                d2_listed[f][n] = tags
        views["d2"] = {"source": f"D2 table {args.d2} (sha256 {sha})", "families": families,
                       "result": view(function_ids, d2_members, d2_listed, gen_versions, sys_names, order, True)}
        d2_unlisted = sorted(n for n in gen_names if n not in names)
    primary = "d2" if "d2" in views else "derived"

    header_report = None
    if args.target_headers:
        claims = collections.defaultdict(list)
        for g in manifest["groups"]:
            for index, v in enumerate(g["variants"]):
                for reach in v["reaches"]:
                    claims[reach].append((g["name"], index))
        header_report = {}
        for b in sorted(header_builtins(args.target_headers)):
            keys = [k for k in claims if k == b or b.startswith(k + "_")]
            proven = [1 for k in keys for g, i in claims[k]
                      if accepted_by_variant.get((g, i)) and emitted_by_variant.get((g, i))]
            header_report[b] = {"claimed_by": sum(len(claims[k]) for k in keys), "proven": len(proven)}

    sys_meta = set(system["metadata_strings"])
    gen_meta = set(generated["metadata_strings"])
    sys_other = {n: v["modules"] for n, v in system["callees"].items() if not n.startswith("air.")}
    gen_other = {n for n in generated["callees"] if not n.startswith("air.")}

    def totals(summary):
        return {
            "families": len(summary),
            "families_with_accepted_variants": sum(1 for s in summary if s["accepted"]),
            "accepted_variants": sum(s["accepted"] for s in summary),
            "system_variants": sum(s["system"] for s in summary),
            "system_variants_covered": sum(s["system_covered"] for s in summary),
            "listed_variants": sum(s["listed"] for s in summary),
            "listed_variants_covered": sum(s["listed_covered"] for s in summary),
        }

    out = {
        "compiler": results.get("compiler"), "target": results.get("target"), "versions": order,
        "statuses": dict(statuses), "primary_view": primary,
        "generated_names": len(gen_names), "system_names": len(sys_names),
        "system_names_covered": sum(1 for n in sys_names if n in gen_names),
        "system_names_uncovered": {n: m for n, m in sorted(sys_names.items()) if n not in gen_names},
        "views": {k: {"source": v["source"], "totals": totals(v["result"][0]), "families": v["result"][0],
                      "detail": v["result"][1], "outside": v["result"][2]} for k, v in views.items()},
        "generated_names_not_in_d2": d2_unlisted,
        "metadata_strings_system": len(sys_meta), "metadata_strings_system_covered": len(sys_meta & gen_meta),
        "metadata_strings_uncovered": sorted(sys_meta - gen_meta),
        "llvm_intrinsics_system": sum(1 for n in sys_other if n.startswith("llvm.")),
        "llvm_intrinsics_system_covered": sum(1 for n in sys_other if n.startswith("llvm.") and n in gen_other),
        "llvm_intrinsics_uncovered": {n: m for n, m in sorted(sys_other.items()) if n.startswith("llvm.") and n not in gen_other},
        "other_system_callees": {n: m for n, m in sorted(sys_other.items()) if not n.startswith("llvm.")},
        "headers": header_report,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "coverage.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=sorted)
        fh.write("\n")

    lines = ["# AIR corpus coverage", "",
             f"- Compiler: {out['compiler']}; target `{out['target']}`; language versions {', '.join(order)}",
             f"- Probe outcomes over all versions: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())),
             f"- Distinct `air.*` names emitted by accepted probes: {len(gen_names)}",
             f"- System `air.*` names also emitted: {out['system_names_covered']} / {len(sys_names)}",
             f"- System metadata strings also in the generated corpus: {out['metadata_strings_system_covered']} / {len(sys_meta)}",
             f"- System `llvm.*` intrinsics also in the generated corpus: {out['llvm_intrinsics_system_covered']} / {out['llvm_intrinsics_system']}"]
    if d2_unlisted is not None:
        lines.append(f"- Generated `air.*` names absent from D2's closed name list: {len(d2_unlisted)}")
    for key in ([primary] + [k for k in views if k != primary]):
        v = out["views"][key]
        t = v["totals"]
        lines += ["", f"## {'Primary' if key == primary else 'Secondary'} view: {v['source']}", "",
                  f"- Families: {t['families']}; with accepted variants: {t['families_with_accepted_variants']}",
                  f"- Accepted variants: {t['accepted_variants']}",
                  f"- System variants covered: {t['system_variants_covered']} / {t['system_variants']}",
                  f"- Listed variants covered: {t['listed_variants_covered']} / {t['listed_variants']}", "",
                  "| family | decision | accepted | system | listed (by source) | versions |",
                  "| --- | --- | ---: | ---: | --- | --- |"]
        info = views[key].get("families", {})
        for s in v["families"]:
            src = "; ".join(f"{t} {c['covered']}/{c['total']}" for t, c in s["listed_by_source"].items())
            vr = f"{s['versions'][0]}-{s['versions'][-1]}" if s["versions"] else "-"
            dec = info.get(s["family"], {}).get("decision", "") if info else ""
            lines.append(f"| `{s['family']}` | {dec} | {s['accepted']} | {s['system_covered']}/{s['system']} | "
                         f"{s['listed_covered']}/{s['listed']} ({src}) | {vr} |")
        lines += ["", "Families with no accepted variant:", ""]
        for s in v["families"]:
            if not s["accepted"]:
                det = v["detail"][s["family"]]
                ex = sorted({n for e in list(det["system"].values()) + list(det["listed"].values()) for n in e["names"]})
                lines.append(f"- `{s['family']}`: {len(ex)} names, e.g. " + ", ".join(f"`{e}`" for e in ex[:5]))
        lines += ["", "Names outside every family of this view:", ""]
        for kind, names_ in v["outside"].items():
            lines.append(f"- {kind}: {len(names_)}" + (": " + ", ".join(f"`{n}`" for n in names_[:30]) if names_ else "")
                         + (" ..." if len(names_) > 30 else ""))
    lines += ["", "## System `air.*` names the generated corpus does not emit", ""]
    for n, m in out["system_names_uncovered"].items():
        lines.append(f"- `{n}` ({m} modules)")
    if d2_unlisted:
        lines += ["", "## Generated names D2's closed list does not contain", ""]
        lines.append(", ".join(f"`{n}`" for n in d2_unlisted))
    lines += ["", "## System metadata strings not in the generated corpus", ""]
    lines.append(", ".join(f"`{m}`" for m in out["metadata_strings_uncovered"]) or "(none)")
    lines += ["", "## System `llvm.*` intrinsics not in the generated corpus", ""]
    lines.append(", ".join(f"`{n}` ({m})" for n, m in out["llvm_intrinsics_uncovered"].items()) or "(none)")
    lines += ["", "## Other functions system modules declare but do not define", ""]
    other = out["other_system_callees"]
    lines.append(f"{len(other)} names" + (": " + ", ".join(f"`{n}`" for n in list(other)[:40]) if other else "")
                 + (" ..." if len(other) > 40 else ""))
    if header_report is not None:
        lines += ["", "## Target header builtins", "",
                  f"{sum(1 for x in header_report.values() if x['proven'])} of {len(header_report)} `__metal_*` functions "
                  "called by the target headers are claimed by an accepted variant that emits AIR.", ""]
        for b, x in sorted(header_report.items()):
            if not x["proven"]:
                lines.append(f"- `__metal_{b}` (claimed by {x['claimed_by']} variants, none accepted with AIR)")
    with open(os.path.join(args.out_dir, "coverage.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    p = out["views"][primary]["totals"]
    print(f"view={primary} " + " ".join(f"{k}={v}" for k, v in p.items())
          + f" system_names_covered={out['system_names_covered']}/{len(sys_names)}")


if __name__ == "__main__":
    main()
