#!/usr/bin/env python3
"""Collect the air.* names the generated AIR corpus emits.

The generated corpus (corpus/pipeline.py) compiles MSL probes for the target triple at
every language version the compiler accepts and records, per probe, whether the compiler
accepted it and which air.* functions the accepted probe calls (corpus/build.py writes
results.json), and surveys the modules' metadata (corpus/survey.py writes survey.json).
This script reads both and writes the names, with the language versions that emitted each
function, for gen/gen_intrinsics.py and check.py. Functions called from a module's static
initializers belong to the module and count as emitted by it.

Usage: generated_names.py --build DIR --out NAMES.json
"""

import argparse
import hashlib
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", required=True, help="output directory of corpus/build.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results_path = os.path.join(a.build, "results.json")
    with open(results_path, "rb") as f:
        raw = f.read()
    results = json.loads(raw)
    with open(os.path.join(a.build, "survey.json")) as f:
        survey = json.load(f)
    order = list(results["versions"])
    functions = {}
    accepted = 0
    for group in results["groups"].values():
        for version, res in group["versions"].items():
            for module in res.get("modules", []):
                for n in module.get("static_init_intrinsics", []):
                    if n.startswith("air."):
                        functions.setdefault(n, set()).add(version)
            for rec in res["status"].values():
                if rec["status"] != "accepted":
                    continue
                accepted += 1
                for n in rec.get("intrinsics", []):
                    if n.startswith("air."):
                        functions.setdefault(n, set()).add(version)
    out = {
        "results_sha256": hashlib.sha256(raw).hexdigest(),
        "compiler": results["compiler"],
        "target": results["target"],
        "language_versions": order,
        "accepted_probes": accepted,
        "modules": survey["modules"],
        "air_versions": survey["air_versions"],
        "functions": {n: sorted(v, key=order.index) for n, v in sorted(functions.items())},
        "metadata_strings": sorted(survey["metadata_strings"]),
        "named_metadata": sorted(n for n in survey["named_metadata"] if n.startswith("air.")),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("%d functions, %d metadata strings, %d named metadata from %d accepted probes" % (
        len(out["functions"]), len(out["metadata_strings"]), len(out["named_metadata"]), accepted))


if __name__ == "__main__":
    main()
