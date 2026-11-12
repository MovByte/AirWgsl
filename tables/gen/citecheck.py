"""Citation and completeness checks shared by the table checkers.

A citation is an inline table in one of these forms:

  { src = ID, ref = REF, quote = TEXT }   TEXT occurs (whitespace-normalized) in the
                                          scope REF names in pinned source ID: an HTML
                                          anchor ("#id"), an MSL section number ("8.4"),
                                          or "" for a single documentation page.
  { src = ID, ref = REF, absent = TERM }  TERM does not occur in that scope.
  { sdk = PATH, quote = TEXT }            TEXT occurs in the SDK header PATH, pinned in
                                          the "sdk" source (or absent = TERM: it does not).
  { measure = ID, case = CASE, quote = TEXT }
                                          the committed result of measurement ID records
                                          CASE, and TEXT occurs in that case's record.
  { target = KIND, ... }                  a fact read from the target root's binaries or
                                          headers; checked only when the root is given.

Every row must carry at least one citation unless its status is "open", in which case
it must state a reason. No string anywhere in a table may be empty.
"""

import os
import re

import specdoc


class Checker:
    def __init__(self, sources, cache, measures=None, root=None):
        self.sources = sources
        self.cache = cache
        self.measures = measures or {}
        self.root = root
        self.errors = []
        self.checked = 0
        self.unverified = 0

    def fail(self, where, msg):
        self.errors.append("%s: %s" % (where, msg))

    def no_empty(self, where, value):
        if isinstance(value, str):
            if not value.strip():
                self.fail(where, "empty cell")
        elif isinstance(value, dict):
            for k, v in value.items():
                self.no_empty("%s.%s" % (where, k), v)
        elif isinstance(value, list):
            if not value:
                self.fail(where, "empty list")
            for i, v in enumerate(value):
                self.no_empty("%s[%d]" % (where, i), v)

    def cite(self, where, c):
        if not isinstance(c, dict):
            self.fail(where, "citation is not a table")
            return
        self.checked += 1
        if "sdk" in c:
            doc = specdoc.sdk_document("sdk", self.sources, c["sdk"])
            if "quote" in c:
                if not doc.quote_in(None, c["quote"]):
                    self.fail(where, "quote not found in SDK %s: %r" % (c["sdk"], c["quote"]))
            elif "absent" in c:
                if re.search(c["absent"], doc.text):
                    self.fail(where, "term %r is present in SDK %s" % (c["absent"], c["sdk"]))
            else:
                self.fail(where, "SDK citation without a quote or absent term")
            return
        if "measure" in c:
            m = self.measures.get(c["measure"])
            if m is None:
                self.fail(where, "unknown measurement %s" % c["measure"])
                return
            case = m.get(c.get("case", ""))
            if case is None:
                self.fail(where, "measurement %s has no case %r" % (c["measure"], c.get("case")))
                return
            if "quote" in c and specdoc.norm(c["quote"]) not in specdoc.norm(case):
                self.fail(where, "quote not in measurement %s case %s: %r" % (
                    c["measure"], c.get("case"), c["quote"]))
            return
        if "target" in c:
            if self.root is None:
                self.unverified += 1
                return
            path = os.path.join(self.root, c["file"])
            if not os.path.exists(path):
                self.fail(where, "target file missing: %s" % c["file"])
                return
            doc = specdoc.FileText(path)
            if "quote" in c and not doc.quote_in(None, c["quote"]):
                self.fail(where, "quote not found in target %s: %r" % (c["file"], c["quote"]))
            return
        src = c.get("src")
        if src not in self.sources:
            self.fail(where, "unknown source %r" % src)
            return
        kind = self.sources[src]["kind"]
        whole = kind == "appledoc" or self.sources[src].get("sections") is False
        if "ref" not in c and not whole:
            self.fail(where, "citation without ref")
            return
        doc = specdoc.document(src, self.sources, self.cache)
        ref = c.get("ref", "")
        if kind == "html":
            if not ref.startswith("#") or ref[1:] not in doc.anchors:
                self.fail(where, "anchor %s not in %s" % (ref, src))
                return
            key = ref[1:]
        elif kind == "pdf":
            if ref != "" and ref not in doc.sections:
                self.fail(where, "section %s not in %s" % (ref, src))
                return
            key = ref
        else:
            key = ref
        if "quote" in c:
            if not doc.quote_in(key, c["quote"]):
                self.fail(where, "quote not found in %s %s: %r" % (src, ref, c["quote"]))
        elif "absent" in c:
            if kind == "html":
                scope = specdoc.norm(doc.scope(key) or "")
            elif kind == "pdf":
                scope = specdoc.norm(doc.scope(key) or "")
            else:
                scope = specdoc.norm(doc.text)
            if re.search(c["absent"], scope):
                self.fail(where, "term %r is present in %s %s" % (c["absent"], src, ref))
        else:
            self.fail(where, "citation has neither quote nor absent")

    def row(self, where, row, cite_key="cite"):
        self.no_empty(where, row)
        status = row.get("status")
        cites = row.get(cite_key, [])
        if status == "open":
            if not row.get("reason"):
                self.fail(where, "open without a reason")
        elif not cites:
            self.fail(where, "no citation")
        for i, c in enumerate(cites):
            self.cite("%s.%s[%d]" % (where, cite_key, i), c)
