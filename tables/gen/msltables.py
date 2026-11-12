"""Rows of the accuracy tables in section 8.4 of the MSL specification.

The PDF text of a table is a caption line ("Table 8.N. ..."), a column-header line that
repeats after each page break, and rows that start with the function label at the
beginning of a line and may wrap. A row is the text from its label to the next row's
label; the last row ends at the first following line that starts a paragraph about a
table ("Table ...") or the closing note ("Even though ..."). Labels are supplied in
table order, so a wrapped cell never starts a row.
"""

import re

import specdoc

HEADER = "Math function Minimum accuracy (ULP values)"
END_RE = re.compile(r"^(Table \d+\.\d+|Even though )")


def table_lines(doc, table):
    """Lines of one table, from the line after its caption to the next caption or note."""
    lines = doc.scope("8.4").split("\n")
    cap = "Table %s." % table
    start = None
    for i, line in enumerate(lines):
        if line.startswith(cap):
            start = i + 1
            break
    if start is None:
        raise SystemExit("MSL table %s not found in section 8.4" % table)
    out = []
    for line in lines[start:]:
        if END_RE.match(line):
            break
        out.append(line)
    return out


def rows(doc, table, labels):
    """{label: row text} for the labels of one table, in order."""
    lines = table_lines(doc, table)
    starts = []
    pos = 0
    for label in labels:
        pat = re.compile(r"^%s(\s|$)" % re.escape(label))
        for j in range(pos, len(lines)):
            if pat.match(lines[j]):
                starts.append(j)
                pos = j + 1
                break
        else:
            raise SystemExit("MSL table %s: row %r not found in order" % (table, label))
    out = {}
    for k, label in enumerate(labels):
        end = starts[k + 1] if k + 1 < len(labels) else len(lines)
        text = " ".join(lines[starts[k]:end])
        text = specdoc.norm(text)
        if text.endswith(HEADER):
            text = text[: -len(HEADER)].strip()
        if HEADER in text:
            raise SystemExit("MSL table %s: row %r spans a page header" % (table, label))
        out[label] = text
    return out
