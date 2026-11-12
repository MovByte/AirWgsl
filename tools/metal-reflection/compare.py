#!/usr/bin/env python3
"""Compares pipeline reflection built from AIR with real Metal's for the same pipelines.

Usage: compare.py OURS.jsonl METAL.jsonl [REPORT.tsv]

OURS.jsonl is written by examples/reflect_corpus.rs, METAL.jsonl by record.m from the jobs the same run wrote. Lines
are matched by id; a later line of OURS.jsonl replaces an earlier one with the same id (the `specialized` pass's
lines replace `pending` ones), and so does a later line of METAL.jsonl. A pipeline real Metal failed to create is
counted as `metal-error` whatever this crate did with it. A pipeline whose both sides are "ok" must have equal stages: every argument and every type entry,
field for field. The totals go to stdout, with the first mismatch of each kind; REPORT.tsv (optional) gets one line
per pipeline: id, outcome, detail. The exit status is non-zero when any pipeline mismatches.
"""

import collections
import json
import sys


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                out[j["id"]] = j
    return out


def first_difference(a, b, path=""):
    if type(a) is not type(b):
        return "%s: %r != %r" % (path, a, b)
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return "%s.%s: missing in ours (metal %r)" % (path, k, b[k])
            if k not in b:
                return "%s.%s: missing in metal (ours %r)" % (path, k, a[k])
            d = first_difference(a[k], b[k], "%s.%s" % (path, k))
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return "%s: %d entries in ours, %d in metal" % (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, "%s[%d]" % (path, i))
            if d:
                return d
        return None
    if a != b:
        return "%s: ours %r, metal %r" % (path, a, b)
    return None


def kind(detail):
    """A mismatch's kind: its path without indices, and the values for small enumerations."""
    import re
    return re.sub(r"\[\d+\]", "[]", detail.split(":")[0])


def main():
    ours = load(sys.argv[1])
    metal = load(sys.argv[2])
    report = open(sys.argv[3], "w") if len(sys.argv) > 3 else None
    counts = collections.Counter()
    examples = {}
    for id_, o in ours.items():
        status = o["status"]
        if id_ in metal and metal[id_]["status"] != "ok":
            outcome, detail = "metal-error", json.dumps(metal[id_].get("error"))
            key = "metal-error: %s" % metal[id_].get("error", {}).get("description", "")[:160]
        elif status in ("skip", "refused"):
            key = "%s: %s" % (status, (o.get("reason") or o.get("refusal", ""))[:160])
            outcome, detail = status, o.get("reason") or o.get("refusal", "")
        elif id_ not in metal:
            outcome, detail = "unrecorded", ""
            key = "unrecorded"
        else:
            d = first_difference(o["stages"], metal[id_]["stages"], "stages")
            if d:
                outcome, detail = "mismatch", d
                key = "mismatch: %s" % kind(d)
            else:
                outcome, detail = "equal", ""
                key = "equal"
        counts[key] += 1
        examples.setdefault(key, (id_, detail))
        if report:
            report.write("%s\t%s\t%s\n" % (id_, outcome, detail.replace("\t", " ").replace("\n", " ")))
    total = collections.Counter()
    for k, n in counts.items():
        total[k.split(":")[0]] += n
    print("pipelines=%d %s" % (len(ours), " ".join("%s=%d" % kv for kv in sorted(total.items()))))
    for k, n in counts.most_common():
        if k == "equal":
            continue
        print("%6d  %s" % (n, k))
        print("        e.g. %s %s" % (examples[k][0], examples[k][1][:300]))
    sys.exit(1 if total.get("mismatch") else 0)


if __name__ == "__main__":
    main()
