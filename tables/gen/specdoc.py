"""Pinned specification sources and their text, for table generation and checking.

A source is declared in `sources.toml` next to the tables with its URL, version,
date and SHA-256. `fetch` downloads it into a cache directory and refuses any
bytes whose hash differs from the pin. HTML sources (Bikeshed output: WebGPU,
WGSL, Khronos Data Format) are linearized into text with the offset of every
`id` anchor, every heading and every table; PDF sources are converted to text
page by page with PDFKit. A citation is checked by finding its quote, after
whitespace normalization, inside the scope its reference names.
"""

import hashlib
import html
import html.parser
import os
import re
import subprocess
import tomllib
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)

BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "caption", "dd", "details", "dialog", "div",
    "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
    "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary",
    "table", "tbody", "thead", "tfoot", "tr", "ul", "br",
}
SKIP_TAGS = {"script", "style"}
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def norm(s):
    """Whitespace-normalized text used for every quote comparison."""
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def load_sources(path=None):
    with open(path or os.path.join(TABLES, "sources.toml"), "rb") as f:
        return tomllib.load(f)["source"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Cloudflare inserts a per-request challenge script into some pages (the Khronos registry);
# a source marked strip = "cloudflare-challenge" is pinned and cached without it.
CF_CHALLENGE_RE = re.compile(rb"<script>\(function\(\)\{function c\(\)\{var b=a\.contentDocument.*?</script>", re.S)


def fetch(src_id, sources, cache):
    """Return the local path of a pinned source, downloading it if needed."""
    if not cache:
        raise SystemExit("a cache directory is required (--cache or TABLES_CACHE)")
    src = sources[src_id]
    os.makedirs(cache, exist_ok=True)
    ext = {"html": ".html", "pdf": ".pdf", "appledoc": ".json"}[src["kind"]]
    path = os.path.join(cache, src_id + "-" + src["sha256"][:16] + ext)
    if not os.path.exists(path):
        req = urllib.request.Request(src["url"], headers={"User-Agent": "spec-tables/1"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        if src.get("strip") == "cloudflare-challenge":
            data = CF_CHALLENGE_RE.sub(b"", data)
        got = hashlib.sha256(data).hexdigest()
        if got != src["sha256"]:
            raise SystemExit("%s: %s has sha256 %s, pinned %s" % (src_id, src["url"], got, src["sha256"]))
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    got = sha256_file(path)
    if got != src["sha256"]:
        raise SystemExit("%s: cached %s has sha256 %s, pinned %s" % (src_id, path, got, src["sha256"]))
    return path


class Table:
    def __init__(self, offset, anchor):
        self.offset = offset
        self.anchor = anchor
        self.rows = []

    def new_row(self):
        self.rows.append([])

    def new_cell(self, rowspan=1, colspan=1):
        if not self.rows:
            self.new_row()
        self.rows[-1].append({"text": "", "ids": [], "hrefs": [], "rowspan": rowspan, "colspan": colspan})

    def grid(self):
        """Rows with row and column spans expanded; each cell's text is normalized."""
        out = []
        pending = {}  # column -> (cell, remaining rows)
        for row in self.rows:
            line = []
            col = 0
            cells = list(row)
            while cells or col in pending:
                if col in pending:
                    cell, left = pending.pop(col)
                    line.append(cell)
                    if left > 1:
                        pending[col] = (cell, left - 1)
                    col += 1
                    continue
                cell = cells.pop(0)
                cell = dict(cell, text=norm(cell["text"]))
                for k in range(cell["colspan"]):
                    line.append(cell)
                    if cell["rowspan"] > 1:
                        pending[col] = (cell, cell["rowspan"] - 1)
                    col += 1
            out.append(line)
        return out

    def cell(self):
        if not self.rows or not self.rows[-1]:
            self.new_cell()
        return self.rows[-1][-1]


class _Parser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.length = 0
        self.anchors = {}
        self.headings = []
        self.tables = []
        self.table_stack = []
        self.idl = []
        self.skip = 0
        self.in_idl = 0
        self.idl_buf = []
        self.pre_depth = 0
        self.heading = None
        self.last_anchor = None

    def emit(self, s, cell=True):
        self.out.append(s)
        self.length += len(s)
        if self.table_stack and cell:
            t = self.table_stack[-1]
            if t.rows and t.rows[-1]:
                t.rows[-1][-1]["text"] += s
        if self.in_idl:
            self.idl_buf.append(s)
        if self.heading is not None:
            self.heading["title"] += s

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in SKIP_TAGS:
            self.skip += 1
            return
        if self.skip:
            return
        if tag == "tr" and self.table_stack:
            self.emit("\n", cell=False)
            self.table_stack[-1].new_row()
        elif tag in BLOCK_TAGS:
            self.emit("\n")
        if tag in ("td", "th"):
            self.emit(" | ", cell=False)
            if self.table_stack:
                self.table_stack[-1].new_cell(int(a.get("rowspan") or 1), int(a.get("colspan") or 1))
        if tag == "table":
            self.table_stack.append(Table(self.length, a.get("id") or self.last_anchor))
        if tag == "pre":
            self.pre_depth += 1
            if "idl" in (a.get("class") or "").split():
                self.in_idl += 1
                self.idl_buf = []
        if tag in HEADING_TAGS:
            # Bikeshed numbers chapters with h2, so the section depth is the tag level minus one.
            level = HEADING_TAGS[tag] - 1
            self.heading = {"offset": self.length, "tag_level": HEADING_TAGS[tag], "level": level,
                            "id": a.get("id"), "secno": a.get("data-level", ""), "title": ""}
        ident = a.get("id")
        if ident and ident not in self.anchors:
            self.anchors[ident] = self.length
            self.last_anchor = ident
            if self.table_stack:
                self.table_stack[-1].cell()["ids"].append(ident)
        if tag == "a" and self.table_stack and a.get("href", "").startswith("#"):
            self.table_stack[-1].cell()["hrefs"].append(a["href"][1:])

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip -= 1
            return
        if self.skip:
            return
        if tag == "table" and self.table_stack:
            self.tables.append(self.table_stack.pop())
        if tag == "pre":
            self.pre_depth -= 1
            if self.in_idl:
                self.in_idl -= 1
                self.idl.append("".join(self.idl_buf))
        if tag in HEADING_TAGS and self.heading is not None:
            self.heading["title"] = norm(self.heading["title"])
            self.headings.append(self.heading)
            self.heading = None
        if tag in BLOCK_TAGS:
            self.emit("\n")

    def handle_data(self, data):
        if self.skip:
            return
        self.emit(data)


class HtmlDoc:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        p = _Parser()
        p.feed(raw)
        p.close()
        self.text = "".join(p.out)
        self.anchors = p.anchors
        self.headings = sorted(p.headings, key=lambda h: h["offset"])
        self.tables = sorted(p.tables, key=lambda t: t.offset)
        self.idl = p.idl
        self._norm_cache = {}

    def heading_for(self, ident):
        for h in self.headings:
            if h["id"] == ident:
                return h
        return None

    def section_at(self, offset):
        """The innermost heading whose section contains offset."""
        cur = None
        for h in self.headings:
            if h["offset"] <= offset:
                cur = h
            else:
                break
        return cur

    def scope(self, ident):
        """Text of the section an anchor names (a heading) or lies in (any other id)."""
        if ident not in self.anchors:
            return None
        h = self.heading_for(ident)
        if h is None:
            h = self.section_at(self.anchors[ident])
            if h is None:
                return None
        start = h["offset"]
        end = len(self.text)
        for other in self.headings:
            if other["offset"] > start and other["level"] <= h["level"]:
                end = other["offset"]
                break
        if self.heading_for(ident) is None:
            # A non-heading anchor is scoped to its own subsection only.
            for other in self.headings:
                if other["offset"] > start:
                    end = min(end, other["offset"])
                    break
        return self.text[start:end]

    def quote_in(self, ident, quote):
        s = self.scope(ident)
        if s is None:
            return False
        key = (ident,)
        if key not in self._norm_cache:
            self._norm_cache[key] = norm(s)
        return norm(quote) in self._norm_cache[key]

    def section_label(self, ident):
        h = self.heading_for(ident) or self.section_at(self.anchors.get(ident, -1))
        if not h:
            return ""
        return h["title"]


FOOTER_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \| Copyright © \d{4} Apple Inc\. \| All Rights Reserved\.$|^Page \d+ of \d+$")
SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*) (\S.*)$")


class PdfDoc:
    """Text of the MSL specification PDF, with numbered sections located in the body."""

    def __init__(self, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            pages = f.read().split("\f")
        lines = []
        for page in pages:
            for line in page.split("\n"):
                if line.startswith("@@PAGE ") or FOOTER_RE.match(line.strip()):
                    continue
                lines.append(line)
        self.lines = lines
        # The table of contents lists every section with dot leaders; the body
        # heading is the first later line that starts with the same number and title.
        toc = []
        for line in lines:
            m = re.match(r"^(\d+(?:\.\d+)*) (.*?) ?\.{3,} ?\d+$", line)
            if m:
                toc.append((m.group(1), norm(m.group(2))))
        self.toc = toc
        self.sections = {}
        order = []
        idx = 0
        body_start = 0
        for i, line in enumerate(lines):
            if re.search(r"\.{3,} ?\d+$", line):
                body_start = i + 1
        idx = body_start
        for num, title in toc:
            first = title.split(" ")[0]
            for j in range(idx, len(lines)):
                m = SECTION_RE.match(lines[j])
                if m and m.group(1) == num and m.group(2).startswith(first):
                    self.sections[num] = j
                    order.append((j, num))
                    idx = j + 1
                    break
        order.sort()
        self.order = order
        self.titles = dict(toc)
        self._cache = {}

    def scope(self, secno):
        """Text from a section heading to the next heading at the same or a higher level.

        An empty section number names the whole document, for PDFs without numbered
        sections."""
        if secno == "":
            return "\n".join(self.lines)
        if secno not in self.sections:
            return None
        start = self.sections[secno]
        depth = secno.count(".")
        end = len(self.lines)
        for line_no, num in self.order:
            if line_no > start and num.count(".") <= depth:
                end = line_no
                break
        return "\n".join(self.lines[start:end])

    def quote_in(self, secno, quote):
        if secno not in self._cache:
            s = self.scope(secno)
            self._cache[secno] = norm(s) if s is not None else None
        s = self._cache[secno]
        return s is not None and norm(quote) in s

    def section_label(self, secno):
        return ("%s %s" % (secno, self.titles.get(secno, ""))).strip()


def pdf_text(pdf_path, cache):
    """Convert a PDF to text with PDFKit; the converter is built in the cache directory."""
    tool_src = os.path.join(HERE, "pdftext.swift")
    tool = os.path.join(cache, "pdftext-" + sha256_file(tool_src)[:12])
    if not os.path.exists(tool):
        subprocess.run(["xcrun", "swiftc", "-O", "-o", tool, tool_src], check=True,
                       cwd=cache)
    out = pdf_path + ".txt"
    if not os.path.exists(out):
        subprocess.run([tool, pdf_path, out + ".part"], check=True)
        os.replace(out + ".part", out)
    return out


def _apple_inline(items):
    out = []
    for it in items or []:
        t = it.get("type")
        if t == "text":
            out.append(it["text"])
        elif t == "codeVoice":
            out.append(it["code"])
        elif t == "reference":
            out.append(it.get("overridingTitle") or it.get("identifier", "").rsplit("/", 1)[-1])
        elif t == "image":
            pass
        else:
            out.append(_apple_inline(it.get("inlineContent")))
    return "".join(out)


def _apple_blocks(items):
    out = []
    for b in items or []:
        t = b.get("type")
        if t == "paragraph":
            out.append(_apple_inline(b.get("inlineContent")))
        elif t == "heading":
            out.append(b.get("text", ""))
        elif t == "codeListing":
            out.append("\n".join(b.get("code", [])))
        elif t in ("unorderedList", "orderedList"):
            for li in b.get("items", []):
                out.append(_apple_blocks(li.get("content")))
        elif t == "table":
            for row in b.get("rows", []):
                out.append(" | ".join(_apple_blocks(c) for c in row))
        elif t == "termList":
            for it in b.get("items", []):
                out.append(_apple_inline(it["term"].get("inlineContent")) + ": " +
                           _apple_blocks(it["definition"].get("content")))
        else:
            out.append(_apple_blocks(b.get("content")))
    return "\n".join(out)


class AppleDoc:
    """Text of one page of Apple's developer documentation (its published JSON form)."""

    def __init__(self, path):
        import json
        with open(path, "rb") as f:
            d = json.load(f)
        parts = [d.get("metadata", {}).get("title", ""), _apple_inline(d.get("abstract"))]
        for sec in d.get("primaryContentSections", []):
            if sec.get("kind") == "content":
                parts.append(_apple_blocks(sec.get("content")))
        # A type's page lists each member with its Objective-C name and abstract.
        for ref in d.get("references", {}).values():
            nav = ref.get("navigatorTitle") or []
            if ref.get("kind") == "symbol" and nav and "text" in nav[0]:
                parts.append("%s: %s" % (nav[0]["text"], _apple_inline(ref.get("abstract"))))
        self.text = "\n".join(parts)
        self._norm = norm(self.text)

    def quote_in(self, ref, quote):
        return norm(quote) in self._norm


class FileText:
    """A local text file (an SDK or target-root header) pinned by hash."""

    def __init__(self, path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            self.text = f.read()
        self._norm = norm(self.text)

    def quote_in(self, ref, quote):
        return norm(quote) in self._norm


def sdk_file(src, rel):
    """Path of a header in the SDK a source names, verified against its pin."""
    sdk = subprocess.run(["xcrun", "--sdk", src["sdk"], "--show-sdk-path"], check=True,
                         capture_output=True, text=True).stdout.strip()
    path = os.path.join(sdk, src.get("prefix", ""), rel)
    want = src["files"].get(rel)
    if want is None:
        raise SystemExit("%s is not pinned in source %s" % (rel, src["title"]))
    got = sha256_file(path)
    if got != want:
        raise SystemExit("%s has sha256 %s, pinned %s" % (path, got, want))
    return path


_DOCS = {}


def document(src_id, sources, cache):
    """Load (once) the parsed document of an HTML or PDF source."""
    if src_id in _DOCS:
        return _DOCS[src_id]
    kind = sources[src_id]["kind"]
    if kind == "sdk":
        raise SystemExit("an SDK source is opened per file with sdk_document")
    path = fetch(src_id, sources, cache)
    if kind == "html":
        doc = HtmlDoc(path)
    elif kind == "appledoc":
        doc = AppleDoc(path)
    else:
        doc = PdfDoc(pdf_text(path, cache))
    _DOCS[src_id] = doc
    return doc


def idl_text(doc):
    """All IDL definition blocks of an HTML specification, concatenated."""
    return "\n".join(html.unescape(b) for b in doc.idl)


def idl_enum(idl, name):
    m = re.search(r"\benum\s+%s\s*\{(.*?)\};" % re.escape(name), idl, re.S)
    if not m:
        raise SystemExit("IDL enum %s not found" % name)
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return re.findall(r'"([^"]*)"', body)


def idl_interface_attributes(idl, name):
    m = re.search(r"\binterface\s+%s\s*\{(.*?)\};" % re.escape(name), idl, re.S)
    if not m:
        raise SystemExit("IDL interface %s not found" % name)
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return re.findall(r"readonly\s+attribute\s+([A-Za-z0-9_ ]+?)\s+([A-Za-z0-9_]+)\s*;", body)


def sdk_document(src_id, sources, rel):
    key = (src_id, rel)
    if key not in _DOCS:
        _DOCS[key] = FileText(sdk_file(sources[src_id], rel))
    return _DOCS[key]
