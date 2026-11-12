#!/usr/bin/env python3
"""Mutation test of the air.* name grammar of tables/intrinsics.toml.

Every function of the table (with each slot filled by the first token of its domain) and
every function name the corpora and the probe emit is mutated by operators that each break
one rule of the grammar: a type token too many or too few, a token on a function that is
not overloaded, a vector width, element type, address space or letter case that no domain
has, a token of another slot kind's domain or outside the family's narrower domain, a stem
that is no table function or only a prefix of one, a broken separator. Every listed
non-function name is mutated as well.

Each mutant is judged twice: by airgrammar (the table's grammar, with the binary-search
lookup and the domains' regular expressions) and by a reference written independently here
(the lookup as successive linear filters of the sorted table, and the domains as
enumerated token sets). They must agree on every mutant. A mutant the reference finds
valid is not a near miss (removing a .s marker can give another valid name) and is
counted apart; every other mutant must be refused.

Usage: grammar_mutations.py [--table PATH] [--names NAMES.json ...]
"""

import argparse
import json
import os
import sys
import tomllib

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import airgrammar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)


class Reference:
    def __init__(self, data, version):
        versions = data["grammar"]["air_versions"]
        self.functions = {r["name"]: r for r in data["function"]
                          if version in airgrammar.parse_air_range(r["air"], versions)}
        self.table = sorted(self.functions, key=lambda s: s.encode())
        self.domains = {d["id"]: set(airgrammar.domain_tokens(d)) for d in data["domain"]}
        self.default = {k: d["id"] for d in data["domain"] for k in d.get("default_for", [])}
        self.family_domains = {f["id"]: f.get("domains", {}) for f in data["family"]}
        self.names = {r["name"] for r in data["name"]}
        self._cache = {}

    def lookup(self, name):
        """The table name lookupIntrinsicByName returns, as successive filters by component."""
        windows = []
        end = 0
        while end < len(name):
            start = end
            dot = name.find(".", 0 if start == 0 else start + 1)
            end = len(name) if dot < 0 else dot
            windows.append((start, end))
        rows = self.table
        last = None
        prefix = ""
        for start, end in windows:
            if not rows:
                break
            last = rows[0]
            prefix = name[:end]
            if prefix in self._cache:
                rows = self._cache[prefix]
                continue
            rows = [s for s in rows if s[start:end] == name[start:end]]
            if start == 0 or end - start > 0:
                self._cache[prefix] = rows
        if rows:
            last = rows[0]
        if last is None:
            return None
        if name == last or name.startswith(last + "."):
            return last
        return None

    def valid(self, name):
        if not name.startswith("air."):
            return False
        stem = self.lookup(name)
        if stem is None:
            return False
        r = self.functions[stem]
        slots = r.get("slots", [])
        if name == stem:
            return not slots
        if not slots:
            return False
        tokens = name[len(stem) + 1:].split(".")
        if len(tokens) != len(slots):
            return False
        doms = self.family_domains[r["family"]]
        return all(t in self.domains[doms.get(k) or self.default[k]] for k, t in zip(slots, tokens))


def first_token(data, family, kind):
    fam = {f["id"]: f for f in data["family"]}[family]
    dom_id = fam.get("domains", {}).get(kind) or {k: d["id"] for d in data["domain"] for k in d.get("default_for", [])}[kind]
    dom = {d["id"]: d for d in data["domain"]}[dom_id]
    return sorted(airgrammar.domain_tokens(dom))[0]


OUTSIDE_TOKENS = ["v5f32", "v1f32", "v8f32", "v16i32", "v0i32", "f80", "f128", "i7", "i128", "bf32", "f8",
                  "F32", "V4F32", "p0i8", "p4i8", "p7f32", "p1v4f32", "p1f64", "a4f32", "s_struct.x", "isVoid", ""]
WRONG_KIND = {"any": ["p1i8", "v64f32"], "anyint": ["f32", "v2f16", "i1", "p1i32"],
              "anyfloat": ["i32", "v2i16", "bf16", "p1f32"], "anyvector": ["f32", "i32", "p1f32"],
              "anypointer": ["i8", "v4f32", "f32"]}
NARROW = ["v2f32", "v3i32", "v4i8", "v4i64", "v4bf16", "v64f32", "v4f32", "v2i1"]


def mutants(data, name, stem, tokens, slots):
    out = []
    if slots:
        out.append(("extra-token", name + "." + tokens[-1]))
        out.append(("missing-token", ".".join([stem] + tokens[:-1])))
        for i, kind in enumerate(slots):
            for bad in OUTSIDE_TOKENS + WRONG_KIND[kind] + NARROW:
                if bad != tokens[i]:
                    t = list(tokens)
                    t[i] = bad
                    out.append(("token", ".".join([stem] + t)))
        out.append(("separator", stem + "_" + ".".join(tokens)))
        out.append(("separator", stem + ".." + ".".join(tokens)))
        out.append(("separator", name + "."))
    else:
        out.append(("token-on-plain", name + ".i32"))
        out.append(("token-on-plain", name + ".v4f32"))
        out.append(("separator", name + "."))
    out.append(("stem", stem + "x" + name[len(stem):]))
    out.append(("stem", stem[:-1] + name[len(stem):]))
    if "." in stem[4:]:
        out.append(("stem", stem[:stem.rindex(".")] + name[len(stem):]))
    out.append(("stem", "air_" + name[4:]))
    out.append(("stem", "llvm." + name[4:]))
    out.append(("stem", name.upper()))
    return out


def run(data, extra_names=(), version=airgrammar.AIR_VERSION):
    grammar = airgrammar.Grammar(data)
    ref = Reference(data, version)
    bases = {}
    for r in data["function"]:
        if version not in airgrammar.parse_air_range(r["air"], data["grammar"]["air_versions"]):
            continue
        slots = r.get("slots", [])
        toks = [first_token(data, r["family"], k) for k in slots]
        bases[".".join([r["name"]] + toks)] = None
    for n in extra_names:
        bases[n] = None
    stats = {"bases": 0, "mutants": 0, "refused": 0, "valid": 0, "disagree": []}
    by_op = {}
    for name in bases:
        stem, tokens, _ = grammar.function(name, version)
        if not ref.valid(name):
            stats["disagree"].append((name, "base", "reference refuses a name the grammar accepts"))
            continue
        stats["bases"] += 1
        for op, m in mutants(data, name, stem, tokens, grammar.functions[stem].get("slots", [])):
            stats["mutants"] += 1
            expected = ref.valid(m)
            try:
                grammar.function(m, version)
                got = True
            except airgrammar.Refused:
                got = False
            if got != expected:
                stats["disagree"].append((m, op, "grammar %s, reference %s" % (got, expected)))
            if expected:
                stats["valid"] += 1
            else:
                stats["refused"] += 1
                by_op[op] = by_op.get(op, 0) + 1
    names = sorted(r["name"] for r in data["name"])
    for n in names:
        for m in (n + "_x", n + ".", n.upper(), n[:-1]):
            if m in ref.names:
                continue
            stats["mutants"] += 1
            try:
                grammar.name(m)
                stats["disagree"].append((m, "name", "a mutated name is listed"))
            except airgrammar.Refused:
                stats["refused"] += 1
                by_op["name"] = by_op.get("name", 0) + 1
    stats["by_operator"] = dict(sorted(by_op.items()))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default=os.path.join(TABLES, "intrinsics.toml"))
    ap.add_argument("--names", nargs="*", default=[], help="JSON files whose functions or declared_callees to mutate as well")
    a = ap.parse_args()
    with open(a.table, "rb") as f:
        data = tomllib.load(f)
    extra = []
    for p in a.names:
        with open(p) as f:
            d = json.load(f)
        extra += [n for n in d.get("functions", d.get("declared_callees", {})) if n.startswith("air.")]
    stats = run(data, extra)
    for m, op, why in stats["disagree"][:40]:
        print("DISAGREE %s (%s): %s" % (m, op, why))
    print("%d bases, %d mutants: %d refused near misses, %d valid names; by operator %s" % (
        stats["bases"], stats["mutants"], stats["refused"], stats["valid"], stats["by_operator"]))
    return 1 if stats["disagree"] else 0


if __name__ == "__main__":
    sys.exit(main())
