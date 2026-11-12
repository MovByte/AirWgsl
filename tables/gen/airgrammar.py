"""The air.* name grammar of tables/intrinsics.toml.

A function name is accepted as the target's compiler accepts it (measurement air-grammar):
llvm_utils::lookupIntrinsicByName finds a function of the AIR version's table (the stem),
the stem is overloaded exactly when the name extends it, and the name equals the stem
followed by one "." and type token per overload slot (AIRFunctionInfo::getName). Each
token must be in the slot's type domain: the domain the stem's family names for the slot
kind, else the kind's default domain. A token whose element type the domain reclassifies
moves the name to that family. Any other air.* name is accepted only when it is listed
as a [[name]] row.

Grammar.function(name) returns (stem, tokens, family) of a called function, and
Grammar.name(name) the (kind, family) of a name used otherwise, or they raise Refused with
the reason.
"""

import re

AIR_VERSION = "2.6"


class Refused(Exception):
    pass


def lookup(table, name):
    """Index into the sorted table of the stem of name, else -1: the binary searches of
    llvm_utils::lookupIntrinsicByName (measurement air-grammar, case lookup-name)."""
    key = name.encode()
    rows = [s.encode() for s in table] if table and isinstance(table[0], str) else table
    low, high = 0, len(rows)
    last = low
    cmp_end = 0
    while cmp_end < len(key) and high - low > 0:
        cmp_start = cmp_end
        dot = key.find(b".", 0 if cmp_start == 0 else cmp_start + 1)
        cmp_end = len(key) if dot < 0 else dot
        part = key[cmp_start:cmp_end]

        # strncmp over [cmp_start, cmp_end): a shorter table name compares at its end.
        def seg(i):
            return rows[i][cmp_start:cmp_end]

        last = low
        a, b = low, high
        while a < b:
            mid = (a + b) // 2
            if seg(mid) < part:
                a = mid + 1
            else:
                b = mid
        c, b = a, high
        while c < b:
            mid = (c + b) // 2
            if seg(mid) <= part:
                c = mid + 1
            else:
                b = mid
        low, high = a, c
    if high - low > 0:
        last = low
    if last == len(rows):
        return -1
    found = rows[last]
    if key == found or (key.startswith(found) and key[len(found):len(found) + 1] == b"."):
        return last
    return -1


def domain_tokens(dom):
    """Every token of a domain, as getMangledTypeStr writes the types it lists."""
    out = {}
    for e in dom["elements"]:
        if "spaces" in dom:
            for s in dom["spaces"]:
                out["p%d%s" % (s, e)] = e
        else:
            for w in dom["widths"]:
                out[("v%d" % w if w > 1 else "") + e] = e
    return out


def domain_regex(dom):
    """The regular expression of a domain's tokens, with the element type as group 1."""
    elements = "|".join(re.escape(e) for e in sorted(dom["elements"], key=lambda e: (-len(e), e)))
    if "spaces" in dom:
        return r"p(?:%s)(%s)" % ("|".join(str(s) for s in dom["spaces"]), elements)
    vec = [w for w in dom["widths"] if w > 1]
    prefix = ""
    if vec:
        prefix = "v(?:%s)" % "|".join(str(w) for w in vec)
        prefix = "(?:%s)?" % prefix if 1 in dom["widths"] else prefix
    return r"%s(%s)" % (prefix, elements)


def parse_air_range(text, versions):
    """The AIR versions of a [[function]] row's air field ("1.6-2.6" or "2.2")."""
    if "-" in text:
        first, last = text.split("-")
        return versions[versions.index(first):versions.index(last) + 1]
    return [text]


class Grammar:
    def __init__(self, data):
        """data: intrinsics.toml as tomllib loads it."""
        self.versions = data["grammar"]["air_versions"]
        self.domains = {d["id"]: d for d in data["domain"]}
        self.default = {}
        for d in data["domain"]:
            d["_re"] = re.compile(d.get("tokens") or domain_regex(d))
            for kind in d.get("default_for", []):
                self.default[kind] = d["id"]
        self.families = {f["id"]: f for f in data["family"]}
        self.functions = {r["name"]: r for r in data.get("function", [])}
        self.names = {r["name"]: r for r in data.get("name", [])}
        self.tables = {}
        for v in self.versions:
            self.tables[v] = sorted((n for n, r in self.functions.items()
                                     if v in parse_air_range(r["air"], self.versions)), key=lambda s: s.encode())
        self._encoded = {v: [s.encode() for s in t] for v, t in self.tables.items()}

    def slot_domain(self, family, kind):
        return self.domains[self.families[family].get("domains", {}).get(kind) or self.default[kind]]

    def function(self, name, version=AIR_VERSION):
        if not name.startswith("air."):
            raise Refused("does not begin with air.")
        table = self.tables[version]
        i = lookup(self._encoded[version], name)
        if i < 0:
            raise Refused("no function of the AIR %s table is a component prefix of it" % version)
        stem = table[i]
        row = self.functions[stem]
        slots = row.get("slots", [])
        if name == stem:
            if slots:
                raise Refused("%s is overloaded: its name needs %d type token(s)" % (stem, len(slots)))
            return stem, [], row["family"]
        if not slots:
            raise Refused("%s is not overloaded: its name takes no type tokens" % stem)
        tokens = name[len(stem) + 1:].split(".")
        if len(tokens) != len(slots):
            raise Refused("%s takes %d type token(s), the name has %d" % (stem, len(slots), len(tokens)))
        family = row["family"]
        moves = []
        for kind, tok in zip(slots, tokens):
            dom = self.slot_domain(row["family"], kind)
            m = dom["_re"].fullmatch(tok)
            if not m:
                raise Refused("%s is not in domain %s of the %s slot of %s" % (tok, dom["id"], kind, stem))
            for rank, (element, target) in enumerate(dom.get("reclassify", [])):
                if m.group(1) == element:
                    moves.append((rank, target))
        if moves:
            family = min(moves)[1]
        return stem, tokens, family

    def name(self, name):
        """(kind, family) of an air.* name used other than as a called function."""
        if name not in self.names:
            raise Refused("not a listed air.* name")
        r = self.names[name]
        return r["kind"], r["family"]
