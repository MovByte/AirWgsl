#!/usr/bin/env python3
"""Compares argument activity real Metal reports with what reflection reports under each notion of "unused".

Usage: reflect_activity.py jobs JOBS.jsonl JOBS.tsv
       reflect_activity.py compare METAL.jsonl JOBS.tsv ACTIVITY.tsv

`jobs` turns the jobs examples/reflect_corpus.rs writes for tools/metal-reflection/record.m into the job list
examples/reflect_activity.rs reads: id, metallib, the stages' functions, the function constant values.

METAL.jsonl is tools/metal-reflection/record.m's output; JOBS.tsv and ACTIVITY.tsv are examples/reflect_activity.rs's
input and output. For every argument both sides list (matched by job, stage, name and index), and separately for
functions with and without function constants, it counts agreement of Metal's `active` with: the front end's
`air.arg_unused` alone ("none"), that plus the canonical function's unused arguments ("canonical"), that plus the
front end's function as it stands ("front"), and that plus the function specialized and simplified without inlining
("local").
"""

import collections
import json
import sys


def jobs(src, dst):
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            j = json.loads(line)
            stages = []
            if j["kind"] == "compute":
                stages.append("kernel:" + j["function"])
            else:
                v = j.get("vertex", {})
                if "function" in v and "source" not in v:
                    stages.append("vertex:" + v["function"])
                fr = j.get("fragment")
                if fr and "function" in fr:
                    stages.append("fragment:" + fr["function"])
            cs = ",".join("%d:%d:%s" % (c["index"], c["type"], c["hex"]) for c in j.get("constants", [])) or "-"
            out.write("\t".join([j["id"], j["library"], ",".join(stages), cs]) + "\n")


def compare(metal_path, jobs_path, activity_path):
    metal = {}
    with open(metal_path) as f:
        for line in f:
            j = json.loads(line)
            if j["status"] != "ok":
                continue
            for st in j["stages"]:
                for a in st["arguments"]:
                    metal[(j["id"], st["stage"], a["name"], a["index"])] = a["active"]
    constants = {}
    with open(jobs_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            constants[p[0]] = p[3] != "-"
    counts = collections.Counter()
    with open(activity_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[2] == "-":
                counts[("not reflected",)] += 1
                continue
            key = (p[0], p[1], p[2], int(p[3]))
            if key not in metal or "-" in (p[4], p[6], p[7]):
                counts[("not compared",)] += 1
                continue
            marked_used = p[4] == "1"
            kind = "constants" if constants[p[0]] else "plain"
            for name, col in (("none", 4), ("canonical", 5), ("front", 6), ("local", 7)):
                predicted = marked_used and p[col] == "1"
                counts[(name, kind, "agree" if predicted == metal[key] else "differ")] += 1
    for k, v in sorted(counts.items()):
        print(" ".join(k), v)


if sys.argv[1] == "jobs":
    jobs(sys.argv[2], sys.argv[3])
else:
    compare(sys.argv[2], sys.argv[3], sys.argv[4])
