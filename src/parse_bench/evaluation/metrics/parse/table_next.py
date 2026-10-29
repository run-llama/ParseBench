"""Table metric (``table_next``): header-keyed cell triples with severity floors.

Every data cell of a table becomes a triple (row-label path, column-header path, value). The
reference fixes the table structure: title rows, header rows and label columns are inferred once per reference table and
transferred to the output table it pairs with (the output's own inference is only one candidate
among the reference-derived readings; the reading whose cells fit the reference best wins).
Reference and output triples are matched one-to-one: only triples with compatible labels are
candidates, and a Hungarian assignment runs inside each connected component of that sparse graph.

Scores (each in [0, 1]):

* ``content``: F1 of matched triples, each weighted by label similarity x value similarity
  (1 equal, 0.95 letter case, 0.9 same number written differently, 0.5 unit lost, 0 any digit
  differs, else a character ratio).
* ``row_order``: share of reference rows inside the longest run kept in reference order.
* proportional = content - 0.3 x (1 - row_order).
* floor = 1 - severity cost. The first event of a class costs its full weight (critical 0.10,
  major 0.05, minor 0.01, minor total capped at 0.15); later events of the class cost
  ``min(1, K / reference cells)`` of it (K = 80), so a long table is not zeroed by a handful of
  repeated slips. Systematic quirks count once (a reworded label per table, a failing column,
  an inserted column, row labels wrapped over two rows where the reference spans one cell), and so do
  repeats of one structural cause (a rowspan value written in one row only, a whole table moved in the page
  order).
* ``table_next = min(proportional, floor)``.

Policy: row reorder, column swap, transposition, header flattening, table split/merge, number
format and case changes are errors (graded by severity); output format (Markdown vs HTML,
attributes, th vs td markup) is not. Output tables that share nothing with the reference are
ignored (references are not exhaustive).

test.json flags: ``max_top_title_rows`` caps the leading title rows stripped per table (default 1,
0 = none); ``allow_splitting_ambiguous_merged_tables`` splits side-by-side tables emitted as one
wide table (on both sides) without a merge penalty; ``trm_unsupported`` lets a cell also match by
row label and column position (the reference's column headers are not reliable keys) and counts a
header mismatch once per table; ``allow_merged_sub_rows`` also accepts one output row per first-column
span group of the reference, its sub-rows stacked as lines in the group's cells, at no cost.

Regexes follow JavaScript semantics: ``\\s`` is the ECMAScript white-space set, ``\\d``/``\\b``
are ASCII, ``$`` is the end of the input, strings sort by UTF-16 code units.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rapidfuzz.distance import OSA, LCSseq
from scipy.optimize import linear_sum_assignment

from parse_bench.schemas.evaluation import MetricValue

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K_SCALE = 80  # later events of a class weigh min(1, K_SCALE / reference cells)
WEIGHTS = {"critical": 0.10, "major": 0.05, "minor": 0.01}
MINOR_CAP = 0.15
ORDER_PENALTY = 0.3
BIG_COMPONENT = 400  # connected components larger than this are matched greedily
MIN_VALUES = 3  # with 1 to MIN_VALUES - 1 non-empty reference values, dash cells ("-", "n/a") count as values

# ---------------------------------------------------------------------------
# JavaScript compatibility
# ---------------------------------------------------------------------------

_JS_WS = "\t\n\x0b\x0c\r \xa0  -     　﻿"
_S = f"[{_JS_WS}]"
_NOT_W = "(?![A-Za-z0-9_])"  # JS `\b` after a word character
_JS_TRIM = " \t\n\x0b\x0c\r\xa0     　﻿" + "".join(map(chr, range(0x2000, 0x200B)))


def _trim(s: str) -> str:
    """``String.prototype.trim``."""
    return s.strip(_JS_TRIM)


def _u16(s: str) -> bytes:
    """Sort key matching JavaScript's default string order (UTF-16 code units)."""
    return s.encode("utf-16-be", "surrogatepass")


def _sorted_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if _u16(a) <= _u16(b) else (b, a)


def ratio(a: str, b: str) -> float:
    """2 x LCS / (|a| + |b|) over code points (rapidfuzz ``fuzz.ratio`` / 100); 1 for two empty strings."""
    n = len(a) + len(b)
    return 2 * LCSseq.similarity(a, b) / n if n else 1.0


def _u16_len(s: str) -> int:
    return len(s) + sum(1 for ch in s if ord(ch) > 0xFFFF)


def _num_key(v: float) -> str:
    return repr(0.0 if v == 0 else v)


# ---------------------------------------------------------------------------
# Parsing: HTML and Markdown pipe tables into positioned cells
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    r: int
    c: int
    rs: int
    cs: int
    text: str
    th: bool


@dataclass
class ParsedTable:
    kind: str  # "html" or "md"
    cells: list[Cell]
    before: str  # last non-empty line before the table
    pos: int
    end: int = -1  # offset just past the table in the source markdown


_RE_TABLE = re.compile(rf"<table{_NOT_W}[\s\S]*?(?:</table>|\Z)", re.I)  # an unclosed table runs to the end
_RE_TAG = re.compile(rf"<(/?)(thead|tbody|tfoot|tr|td|th){_NOT_W}([^>]*)>", re.I)
_RE_BR = re.compile(rf"<br{_S}*/?>", re.I)
_RE_ANY_TAG = re.compile(r"<[^>]+>")
# A <sup>/<sub> marker attaches to the text before it: pretty-printed "(42)\n<sup>\nab\n</sup>" reads
# "(42)ab", like "(42)<sup>ab</sup>" and the NFKC form of "(42)ᵃᵇ".
_RE_SUPSUB_WS = re.compile(rf"{_S}*(<su[pb]{_NOT_W}[^>]*>){_S}*|{_S}+(</su[pb]{_S}*>)", re.I)
_RE_WS = re.compile(f"{_S}+")
# a <sup> of digits is kept as superscript digits ("1,200<sup>1</sup>" reads "1,200¹"): NFKC compares it as before,
# and ``strip_footnotes`` still sees the marker
_RE_SUP_DIGITS = re.compile(rf"<sup{_NOT_W}[^>]*>{_S}*([0-9]{{1,2}}(?:,[0-9]{{1,2}})*){_S}*</sup{_S}*>", re.I)
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
# The reading ``score_table_next`` is scoring: whether <sup>/<sub> markers attach to the text before them
# (``parse_tables``) and whether labels equal once spaces are removed match (``key_sim``). Both default on.
_ATTACH_MARKS: ContextVar[bool] = ContextVar("table_next_attach_marks", default=True)
_SPACELESS_KEYS: ContextVar[bool] = ContextVar("table_next_spaceless_keys", default=True)
_RE_HEX_ENT = re.compile(r"&#x([0-9a-f]{1,8});", re.I)
_RE_DEC_ENT = re.compile(r"&#([0-9]{1,8});")
_RE_MD_SEP = re.compile(rf"{_S}*\|?{_S}*:?-{{3,}}[-|: ]*\Z")
_SPAN_RX = {k: re.compile(rf"{k}{_S}*={_S}*[\"']?([0-9]+)", re.I) for k in ("rowspan", "colspan")}


def _code_point(m: re.Match[str], n: int) -> str:
    return chr(n) if (0 <= n < 0xD800) or (0xDFFF < n <= 0x10FFFF) else m.group(0)


def decode_entities(s: str) -> str:
    s = _RE_HEX_ENT.sub(lambda m: _code_point(m, int(m.group(1), 16)), s)
    s = _RE_DEC_ENT.sub(lambda m: _code_point(m, int(m.group(1))), s)
    return (
        s.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")
    )


def _place(rows: list[list[dict[str, Any]]]) -> list[Cell]:
    occ: set[tuple[int, int]] = set()
    cells: list[Cell] = []
    for r, row in enumerate(rows):
        c = 0
        for x in row:
            while (r, c) in occ:
                c += 1
            for i in range(r, r + x["rs"]):
                for j in range(c, c + x["cs"]):
                    occ.add((i, j))
            cells.append(Cell(r=r, c=c, rs=x["rs"], cs=x["cs"], text=x["text"], th=x["th"]))
            c += x["cs"]
    return cells


_RE_BARE_SEP = re.compile(rf"{_S}*\|?{_S}*:?-+:?{_S}*(?:\|{_S}*:?-+:?{_S}*)+\|?{_S}*\Z")


def _md_cells(line: str) -> list[str]:
    return [_trim(x) for x in re.sub(r"\|\Z", "", re.sub(r"\A\|", "", _trim(line))).split("|")]


def _is_bare_pipe_table(lines: Sequence[str], i: int) -> bool:
    """A GFM pipe table written without leading pipes ("a | b" over "--- | ---"): a line with a pipe, then a
    delimiter row of as many cells."""
    return (
        i + 1 < len(lines)
        and "|" in lines[i]
        and not _trim(lines[i]).startswith("|")
        and bool(_RE_BARE_SEP.match(lines[i + 1]))
        and len(_md_cells(lines[i])) == len(_md_cells(lines[i + 1]))
    )


def _cap_spans(rows: list[list[dict[str, Any]]]) -> None:
    """Cap spans to the table's size: a rowspan ends at the last row (as a browser lays it out), and a colspan
    covers at most as many columns as the table has cells. An oversized span ("rowspan=1000 colspan=1000") would
    otherwise fill a grid of a million slots."""
    cells = max(1, sum(len(r) for r in rows))
    for r, row in enumerate(rows):
        for x in row:
            x["rs"] = min(x["rs"], len(rows) - r)
            x["cs"] = min(x["cs"], cells)


def _last_line(text: str) -> str:
    lines = [t for t in (_trim(line) for line in text.split("\n")) if t]
    return lines[-1] if lines else ""


def parse_tables(md: str | None, row_spans: bool = True) -> list[ParsedTable]:
    """Every HTML table and Markdown pipe table of ``md``, in document order (``row_spans=False`` reads every
    rowspan as 1: the reading of a span that was not meant to cover the rows below)."""
    md = md or ""
    tables: list[ParsedTable] = []
    for m in _RE_TABLE.finditer(md):
        html = m.group(0)
        rows: list[list[dict[str, Any]]] = []
        row: list[dict[str, Any]] | None = None
        open_tag: tuple[str, str, int] | None = None
        for t in _RE_TAG.finditer(html):
            close, tag = t.group(1) == "/", t.group(2).lower()
            if tag == "tr":
                if not close:
                    row = []
                    rows.append(row)
            elif tag in ("td", "th"):
                if not close:
                    open_tag = (tag, t.group(3), t.end())
                elif open_tag is not None:
                    raw, attrs = html[open_tag[2] : t.start()], open_tag[1]

                    def span(k: str, attrs: str = attrs) -> int:
                        x = _SPAN_RX[k].search(attrs)
                        return max(1, min(1000, int(x.group(1)))) if x else 1

                    if row is None:
                        row = []
                        rows.append(row)
                    if _ATTACH_MARKS.get():
                        raw = _RE_SUPSUB_WS.sub(lambda x: x.group(1) or x.group(2), raw)
                    raw = _RE_SUP_DIGITS.sub(lambda x: x.group(1).translate(_SUPERSCRIPT), raw)
                    text = _trim(_RE_WS.sub(" ", decode_entities(_RE_ANY_TAG.sub("", _RE_BR.sub(" ", raw)))))
                    rs = span("rowspan") if row_spans else 1
                    row.append({"rs": rs, "cs": span("colspan"), "text": text, "th": open_tag[0] == "th"})
                    open_tag = None
        _cap_spans(rows)
        tables.append(ParsedTable("html", _place(rows), _last_line(md[: m.start()]), m.start(), m.end()))
    masked = _RE_TABLE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), md)
    lines, orig = masked.split("\n"), md.split("\n")
    offs, off = [], 0
    for line in lines:
        offs.append(off)
        off += len(line) + 1
    i = 0
    while i < len(lines):
        bare = _is_bare_pipe_table(lines, i)
        if bare or _trim(lines[i]).startswith("|"):
            j = i
            while j < len(lines) and (
                _trim(lines[j]).startswith("|") or (bare and "|" in lines[j] and bool(_trim(lines[j])))
            ):
                j += 1
            body = [
                _md_cells(line)
                for line in lines[i:j]
                if not (_RE_MD_SEP.match(line) or (bare and _RE_BARE_SEP.match(line)))
            ]
            if j - i >= 2:
                rows_md = [[{"rs": 1, "cs": 1, "text": x, "th": k == 0} for x in r] for k, r in enumerate(body)]
                tables.append(
                    ParsedTable(
                        "md", _place(rows_md), _last_line("\n".join(orig[:i])), offs[i], offs[j - 1] + len(lines[j - 1])
                    )
                )
            i = j
        else:
            i += 1
    return sorted(tables, key=lambda t: t.pos)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

NX_EMPTY = frozenset({"", "-", "--", "---", "n/a", "na", "none", "nan"})
_PRE = [
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"<[^>]+>"), ""),
]
_POST = [
    (re.compile("[●•⚫▪∙]"), "•"),
    (re.compile("[✗✘✕✖⊗⮾☒❌]"), "✗"),
    (re.compile("[✓✔☑✅]"), "✓"),
    (re.compile("[‘’]"), "'"),
    (re.compile("[“”]"), '"'),
    (re.compile(rf"{_S}*(?:\.{_S}?){{3,}}{_S}*"), " "),
    (re.compile("…+"), " "),
    (re.compile("([$€£¥]) (?=[0-9(])"), r"\1"),
    (re.compile(rf"\A([-+]){_S}+(?=[$€£¥0-9])"), r"\1"),  # "– $75.5" is the signed number "-$75.5"
    (re.compile("([0-9]) %"), r"\1%"),
    (re.compile(f"{_S}+(?=[®™©])"), ""),
    (re.compile(f"{_S}*/{_S}*"), "/"),
    (re.compile(f"{_S}+"), " "),
]
_RE_END_PUNCT = re.compile(r"([^0-9])[.:]\Z")
_BASE = [
    (re.compile(r"\*\*|__|~~"), ""),
    (re.compile(r"(?<![\w*])\*(?=[^\s*])([^*]*?[^\s*])\*(?![\w*])"), r"\1"),
    (re.compile("[‐-―−]"), "-"),
    (re.compile(" "), " "),
    (re.compile("×"), "x"),
    (re.compile("[µμ]"), "u"),
    (re.compile(f"{_S}+"), " "),
]


def _pre(s: str) -> str:
    for rx, rep in _PRE:
        s = rx.sub(rep, s)
    return s


def _post(s: str) -> str:
    for rx, rep in _POST:
        s = rx.sub(rep, s)
    return _trim(_RE_END_PUNCT.sub(r"\1", _trim(s)))


def _base(s: str) -> str:
    s = unicodedata.normalize("NFKC", _pre(s))
    for rx, rep in _BASE:
        s = rx.sub(rep, s)
    return _trim(s)


_NORM: dict[str, str] = {}


def nx_norm(s: str) -> str:
    """Comparison form of a cell: links, tags, emphasis, typography, dot leaders and case removed."""
    v = _NORM.get(s)
    if v is None:
        v = _post(_base(s)).lower()
        if len(_NORM) > 200_000:
            _NORM.clear()
        _NORM[s] = v
    return v


def _surface(s: str) -> str:
    return _post(_base(s))


_RE_THOUSANDS = re.compile(rf"([0-9]),(?=[0-9]{{3}}{_NOT_W})")
# "1 200 000" and "1'200": thousands grouped by a space (NBSP and thin spaces read as one) or an apostrophe, one
# separator throughout, the first group 1 to 3 digits long
_RE_SPACE_GROUPS = re.compile(r"(?<![0-9.,])([0-9]{1,3})([ '])([0-9]{3}(?:\2[0-9]{3})*)(?![0-9])")
_RE_NUMS = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _ungroup(s: str) -> str:
    """Thousands separators removed from a normalized cell: commas, and spaces or apostrophes between groups."""
    s = _RE_SPACE_GROUPS.sub(lambda m: m.group(1) + m.group(3).replace(m.group(2), ""), s)
    return _RE_THOUSANDS.sub(r"\1", s)


def nx_digits(s: str) -> str:
    return "|".join(_num_key(float(x)) for x in _RE_NUMS.findall(_ungroup(nx_norm(s))))


@dataclass(frozen=True)
class Number:
    cmp: str
    v: float
    pct: str
    unit: str


_RE_CURRENCY = re.compile("[$€£¥]")
_RE_GROUP = re.compile(rf"([0-9])[, ](?=[0-9]{{3}}{_NOT_W})")
_RE_NUMBER = re.compile(rf"([<>]?){_S}*(-?[0-9]+(?:\.[0-9]+)?){_S}*(%?){_S}*(.*)\Z")


def nx_number(s: str) -> Number | None:
    t = _RE_GROUP.sub(r"\1", _ungroup(_RE_CURRENCY.sub("", nx_norm(s))))
    neg = False
    if re.match(r"\(.*\)\Z", t):
        neg, t = True, _trim(t[1:-1])
    m = _RE_NUMBER.match(t)
    if not m:
        return None
    return Number(m.group(1), float(m.group(2)) * (-1 if neg else 1), m.group(3), _trim(m.group(4)))


def is_numeric(s: str) -> bool:
    n = nx_number(s)
    return n is not None and not re.search(r"[a-z]{3,}.*[a-z]{3,}", n.unit)


def _is_year(s: str) -> bool:
    return bool(re.match(r"(19|20)[0-9]{2}\Z", _trim(s)))


_SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_RE_FOOTNOTE = re.compile(
    rf"(?<=[0-9%)])(?:[{_SUP}]{{1,2}}(?:,[{_SUP}]{{1,2}})*|\^\(?[0-9]{{1,2}}\)?|<sup>{_S}*[0-9]{{1,2}}{_S}*</sup>)(?![0-9])",
    re.I,
)
_RE_POWER_BASE = re.compile(r"(?:\A|[^0-9.,])10\Z")


def strip_footnotes(s: str) -> str:
    """A cell with the footnote digits glued to its numbers removed ("1,200¹", "1,200^1", "1,200<sup>1</sup>"
    read "1,200"). A marker after a bare "10" is a power of ten ("10⁶") and stays."""
    return _RE_FOOTNOTE.sub(lambda m: m.group(0) if _RE_POWER_BASE.search(s[: m.start()]) else "", s)


_NEGATIONS = frozenset({"not", "no", "non", "never"})


def _negated(a: str, b: str) -> bool:
    """Two normalized cells with the same words once negations are removed, negated an odd number of times
    apart: "approved" and "not approved" (or "non-approved") state opposite values."""
    wa, wb = (re.sub(r"\bnon-", "non ", x).split(" ") for x in (a, b))
    na, nb = (sum(1 for w in ws if w in _NEGATIONS) for ws in (wa, wb))
    return (na + nb) % 2 == 1 and [w for w in wa if w not in _NEGATIONS] == [w for w in wb if w not in _NEGATIONS]


def value_sim(a: str, b: str, key: str = "") -> float:
    """1 equal; 0.95 letter case only; 0.9 same number written differently; 0.5 unit lost; 0 any digit differs.

    A percent sign present on one side only is no loss when the cell's labels (``key``) already say percent."""
    na, nb = nx_norm(a), nx_norm(b)
    if na in NX_EMPTY and nb in NX_EMPTY:
        return 1.0
    if na == nb:
        return 1.0 if _surface(a) == _surface(b) else 0.95
    if nx_digits(na) != nx_digits(nb):
        fa, fb = strip_footnotes(a), strip_footnotes(b)
        if (fa, fb) != (a, b):  # footnote digits glued to a number on one side: the number is right
            return min(0.9, value_sim(fa, fb, key))
        return 0.0
    x, y = nx_number(na), nx_number(nb)
    if x and y:
        if x.v == y.v and x.cmp == y.cmp:
            if x.unit == y.unit and x.pct != y.pct and "%" in key:
                return 1.0
            return 0.9 if (x.unit == y.unit and x.pct == y.pct) else 0.5
        return 0.0
    if _negated(na, nb):
        return 0.0
    return ratio(na, nb)


def key_sim(a: str, b: str) -> float:
    """Label similarity: same words up to typos, digits agree; extra/missing words in order get 0.6.

    Labels equal once spaces are removed get 0.9: a line break dropped without a space
    ("total(net)" for "total (net)") is a word-boundary slip."""
    if a == b:
        return 1.0
    if nx_digits(a) != nx_digits(b):
        return 0.0
    if _SPACELESS_KEYS.get() and a.replace(" ", "") == b.replace(" ", ""):
        return 0.9
    wa, wb = a.split(" "), b.split(" ")
    if len(wa) != len(wb):
        if not a or not b:
            return 0.0
        short, long_ = (wa, wb) if len(wa) < len(wb) else (wb, wa)
        if 2 * len(short) < len(long_) and not any(_u16_len(w) >= 4 for w in short):
            return 0.0
        k = 0
        for w in long_:
            if k < len(short) and w == short[k]:
                k += 1
        return 0.6 if k == len(short) else 0.0
    p = 1.0
    for x, y in zip(wa, wb, strict=True):
        if x == y:
            continue
        r = ratio(x, y) if _u16_len(x) > 3 else 0.0
        if r < 0.8 and _u16_len(x) > 3 and OSA.distance(x, y) <= 1:
            r = 0.8  # one typo (a transposition included)
        if r < 0.8:
            return 0.0
        p *= r
    return p


# ---------------------------------------------------------------------------
# Table grid and structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    text: str
    th: bool
    id: int  # cell index; -1 for an empty filler slot


Grid = list[list[Slot]]


@dataclass
class TableGrid:
    g: Grid
    empty_rows: int
    empty_cols: int
    kind: str
    before: str
    transposed: bool = False


@dataclass(frozen=True)
class Schema:
    T: int  # title rows
    H: int  # header rows
    L: int  # label columns
    sec: bool = False  # section rows qualify the row labels below them


def _dense(cells: Sequence[Cell]) -> Grid:
    rr = max((x.r + x.rs for x in cells), default=0)
    cc = max((x.c + x.cs for x in cells), default=0)
    empty = Slot("", False, -1)
    g: Grid = [[empty] * cc for _ in range(rr)]
    for k, x in enumerate(cells):
        s = Slot(x.text, x.th, k)
        for i in range(x.r, x.r + x.rs):
            for j in range(x.c, x.c + x.cs):
                g[i][j] = s
    return g


def clean_grid(t: ParsedTable) -> TableGrid:
    """Dense grid with fully empty rows and columns removed (and counted)."""
    g = _dense(t.cells)
    rows0, cols0 = len(g), len(g[0]) if g else 0
    g = [r for r in g if any(x.text for x in r)]
    if not g:
        return TableGrid([], rows0, 0, t.kind, t.before)
    keep = [any(r[j].text for r in g) for j in range(len(g[0]))]
    g = [[x for x, k in zip(r, keep, strict=True) if k] for r in g]
    return TableGrid(g, rows0 - len(g), cols0 - len(g[0]), t.kind, t.before)


def drop_span_copy_rows(tg: TableGrid) -> TableGrid:
    """Rows holding nothing but copies of cells spanning from above (a header cell with rowspan over an otherwise
    empty row) carry no content of their own: the same table written without the span has no such row."""
    g = tg.g
    keep = [g[0]] + [
        r for i, r in enumerate(g[1:], 1) if any(s.text and s.id != g[i - 1][j].id for j, s in enumerate(r))
    ]
    if len(keep) == len(g):
        return tg
    return TableGrid(keep, tg.empty_rows, tg.empty_cols, tg.kind, tg.before, tg.transposed)


_COLUMN_INDEX = re.compile(rf"\A{_S}*\(([0-9]{{1,3}})\){_S}*\Z")


def drop_column_index_rows(tg: TableGrid) -> TableGrid:
    """Drop rows that only number the columns: "(1) | (2) | ... | (n)", one per cell, counting up by one. They
    are printed column references (formula notes cite them), not content, and many references leave them out; kept,
    they are read as the header row and push the real header into the body."""
    g = tg.g

    def is_index(r: Sequence[Slot]) -> bool:
        ns = []
        for j, x in enumerate(r):
            if not x.text:
                continue
            m = _COLUMN_INDEX.match(x.text)
            if not m or (j > 0 and x.id >= 0 and x.id == r[j - 1].id):
                return False
            ns.append(int(m.group(1)))
        return len(ns) >= 2 and all(b == a + 1 for a, b in zip(ns, ns[1:], strict=False))

    keep = [r for r in g if not is_index(r)]
    if len(keep) == len(g) or not keep:
        return tg
    return TableGrid(keep, tg.empty_rows, tg.empty_cols, tg.kind, tg.before, tg.transposed)


def span_labels(tg: TableGrid, schema: Schema) -> set[str]:
    """Normalized texts of the reference's row-label cells that span two or more body rows."""
    body = tg.g[schema.T + schema.H :]
    rows_of: dict[int, set[int]] = {}
    for i, r in enumerate(body):
        for x in r[: schema.L]:
            if x.text and x.id >= 0:
                rows_of.setdefault(x.id, set()).add(i)
    return {nx_norm(x.text) for r in body for x in r[: schema.L] if len(rows_of.get(x.id, ())) >= 2}


def join_wrapped_labels(tg: TableGrid, labels: set[str]) -> tuple[TableGrid, int]:
    """Join a label written over two rows ("Net sales +" above "other income") back into the one cell the reference
    spans over those rows, so both rows key on the full label. Returns the grid and the number of joins."""
    g = [list(r) for r in tg.g]
    joins = 0
    for j in range(len(g[0]) if g else 0):
        i = 0
        while i + 1 < len(g):
            a, b = g[i][j], g[i + 1][j]
            na, nb = nx_norm(a.text), nx_norm(b.text)
            whole = na + nb if na.endswith("-") else na + " " + nb
            if a.text and b.text and a.id != b.id and na not in labels and nb not in labels and whole in labels:
                g[i][j] = g[i + 1][j] = Slot(a.text + " " + b.text, a.th, a.id)
                joins += 1
                i += 2
            else:
                i += 1
    if not joins:
        return tg, 0
    return TableGrid(g, tg.empty_rows, tg.empty_cols, tg.kind, tg.before, tg.transposed), joins


def merge_label_columns(tg: TableGrid) -> TableGrid:
    """Join the first two columns into one cell per row ("1986 | Q3" becomes "1986 Q3"). A row label written in
    one cell on one side and in two on the other keys on the same words; joined, both tables have the same width
    and the value columns line up. A cell spanning both columns stays one cell; each pair of cells joins into one
    new cell (same ids, same new id, so row spans survive)."""
    g = tg.g
    nxt = 1 + max((x.id for r in g for x in r), default=0)
    joined: dict[tuple[int, int], int] = {}
    out: Grid = []
    for r in g:
        a, b = r[0], r[1]
        if a.id == b.id or not b.text:
            s = a
        elif not a.text:
            s = b
        else:
            k = joined.setdefault((a.id, b.id), nxt + len(joined))
            s = Slot(a.text + " " + b.text, a.th or b.th, k)
        out.append([s, *r[2:]])
    return TableGrid(out, tg.empty_rows, tg.empty_cols, tg.kind, tg.before, tg.transposed)


_CONNECTORS = frozenset({"to", "-", "through", "thru", "vs", "vs.", "and", "&", "or"})


def drop_header_only_columns(tg: TableGrid) -> TableGrid:
    """Drop columns whose only text is a connector word in their own header cell ("Credit | to | Debit" with "to"
    empty in every body row): the word belongs to the neighbors' header and the column holds no value. Kept, it
    shifts the value columns. Counted with the empty columns (one minor)."""
    g = tg.g
    h = 0
    while h < len(g) - 2 and all(x.th or not x.text for x in g[h]):
        h += 1
    if not h:
        return tg
    c = len(g[0])

    def own(i: int, j: int) -> bool:  # a header cell that does not span into a neighbor column
        x = g[i][j]
        return (j == 0 or g[i][j - 1].id != x.id) and (j == c - 1 or g[i][j + 1].id != x.id)

    drop = [
        j > 0
        and not any(r[j].text for r in g[h:])
        and any(own(i, j) and nx_norm(g[i][j].text) in _CONNECTORS for i in range(h))
        and all(not own(i, j) or not g[i][j].text for i in range(h) if nx_norm(g[i][j].text) not in _CONNECTORS)
        for j in range(c)
    ]
    if not any(drop) or all(drop[1:]):
        return tg
    g = [[x for x, d in zip(r, drop, strict=True) if not d] for r in g]
    return TableGrid(g, tg.empty_rows, tg.empty_cols + sum(drop), tg.kind, tg.before, tg.transposed)


_SERIAL = re.compile(r"\(?(?:[0-9]{1,3}|[ivxlc]{1,6}|[a-z])[.)]?\)?\Z", re.I)


def _is_serial_run(vals: Sequence[str]) -> bool:
    """Row serials in reading order: no serial repeats the one above, and each number steps 1 to 3 past the
    previous number or restarts at 1 (letters and roman numerals may sit between). Sizes such as 31, 31, 35
    or 19, 23, 27 are a key column, not serials."""
    keys = [re.sub(r"[^0-9a-z]", "", v.lower()) for v in vals]
    nums = [int(k) for k in keys if k.isdigit()]
    return all(a != b for a, b in zip(keys, keys[1:], strict=False)) and all(
        1 <= b - a <= 3 or b == 1 for a, b in zip(nums, nums[1:], strict=False)
    )


def join_serial_column(tg: TableGrid) -> TableGrid:
    """Join a leading column of row serials ("1.", "(2)", "iii", "b)") into the label cell beside it, so
    "(1) | Losses Paid" and "(1) Losses Paid" key the same row. The column qualifies when, from its first serial
    down, every own text is a serial (two or more, forming a run: ``_is_serial_run``) and the cells beside them
    are text. A serial spanning several rows prefixes only its first row's label; the serial column's header is
    kept only where the label's is empty."""
    g = tg.g
    if not g or len(g[0]) < 3:
        return tg
    i0 = next((i for i, r in enumerate(g) if _SERIAL.match(_trim(r[0].text))), len(g))
    rows = [i for i in range(i0, len(g)) if g[i][0].text and g[i][0].id != g[i][1].id]
    starts = [i for i in rows if i == 0 or g[i - 1][0].id != g[i][0].id]
    if (
        len(starts) < 2
        or not all(_SERIAL.match(_trim(g[i][0].text)) for i in rows)
        or not all(g[i][1].text and not is_numeric(g[i][1].text) for i in starts)
        or not _is_serial_run([g[i][0].text for i in starts])
    ):
        return tg
    new: dict[int, Slot] = {}
    for i in starts:
        x, y = g[i][0], g[i][1]
        new[y.id] = Slot(_trim(x.text) + " " + _trim(y.text), y.th, y.id)

    def lead(i: int, r: list[Slot]) -> Slot:
        return r[0] if i < i0 and not r[1].text else new.get(r[1].id, r[1])

    out = [[lead(i, r)] + r[2:] for i, r in enumerate(g)]
    return TableGrid(out, tg.empty_rows, tg.empty_cols, tg.kind, tg.before, tg.transposed)


def transpose_grid(tg: TableGrid) -> TableGrid:
    g = [list(col) for col in zip(*tg.g, strict=True)] if tg.g else []
    return TableGrid(g, tg.empty_cols, tg.empty_rows, tg.kind, tg.before, not tg.transposed)


def is_title_row(r: Sequence[Slot]) -> bool:
    """One text, spanning the row or alone in it."""
    if len(r) < 2 or not r[0].text:
        return False
    return all(x.id == r[0].id or x.text == r[0].text for x in r if x.text)


def is_section_row(r: Sequence[Slot], extent: int) -> bool:
    """A row without values: text only in the label columns, or in one cell spanning several columns."""
    if len(r) < 2:
        return False
    first_id: int | None = None
    n, first, last, one = 0, -1, -1, True
    for j, x in enumerate(r):
        if not x.text:
            continue
        if first_id is None:
            first_id, first = x.id, j
        elif x.id != first_id or x.id < 0:
            one = False
        last = j
        n += 1
    if first_id is None:
        return False
    if one and (first == 0 or n >= 2):
        return True
    return last < max(1, extent)


def _section_text(r: Sequence[Slot]) -> str:
    seen: set[int] = set()
    parts = []
    for x in r:
        if not x.text or (x.id >= 0 and x.id in seen):
            continue
        seen.add(x.id)
        parts.append(nx_norm(x.text))
    return " ".join(parts)


def path_of(slots: Iterable[Slot]) -> str:
    norms = [nx_norm(x.text) for x in slots]
    return " ".join(x for k, x in enumerate(norms) if x and (k == 0 or x != norms[k - 1]))


_ENUM = re.compile(r"\(?[0-9]{1,3}[.)]?\)?\Z")


def _is_enum_column(vals: Sequence[str]) -> bool:
    """Row numbers ("1", "2.", "(3)", increasing by 1 to 3) label rows."""
    if len(vals) < 2 or not all(_ENUM.match(v) for v in vals):
        return False
    n = [int(re.sub(r"[^0-9]", "", v)) for v in vals]
    return all(k == 0 or 1 <= x - n[k - 1] <= 3 for k, x in enumerate(n))  # consecutive, a few gaps allowed


def _label_extent(body: Grid, c: int) -> int:
    """Leading columns without numbers (row numbers allowed). A row of one cell spanning the full width (a
    section heading) is not read as a label value, so it reads like the same heading left in the label column."""
    lab = 0
    body = [r for r in body if not (r[0].id >= 0 and all(x.id == r[0].id for x in r))]
    while lab < c - 1:
        vals = [r[lab].text for r in body if r[lab].text]
        if all(not is_numeric(v) for v in vals) or _is_enum_column(vals):
            lab += 1
        else:
            break
    return lab


def _section_extent(body: Grid, c: int) -> int:
    """Columns that may hold a section label: the label extent when the table has number columns, else column 0."""
    e = _label_extent(body, c)
    return 1 if e >= c - 1 else max(1, e)


def _infer_labels(rows: Grid, h: int, c: int) -> tuple[int, bool]:
    body = rows[h:]
    lab, ext = _label_extent(body, c), _section_extent(body, c)
    if not lab and c > 1:
        lab = 1
    rv = [r for r in body if any(x.text for x in r)]

    def rk(n: int) -> int:
        return len({"|".join(nx_norm(x.text) for x in r[:n]) for r in rv})

    while lab < c - 1 and len(rv) > 1 and rk(lab) == 1:  # label columns that name no row take the next column too
        lab += 1
    data: list[tuple[str, list[Slot]]] = []
    cur = ""
    for r in rv:
        if is_section_row(r, ext):
            cur = _section_text(r)
        else:
            data.append((cur, r))

    def sk(n: int) -> int:
        return len({cs + "|" + "|".join(nx_norm(x.text) for x in r[:n]) for cs, r in data})

    def dk(n: int) -> int:
        return len({"|".join(nx_norm(x.text) for x in r[:n]) for _, r in data})

    lp = ls = 0
    sec = False
    for n in range(1, lab):  # shortest label prefix that makes row keys unique
        if rk(n) == len(rv):
            lp = n
            break
    if data and len(data) < len(rv):
        for n in range(1, lab + 1):
            if sk(n) == len(data):
                ls = n
                break
        if not ls and sk(lab) > dk(lab):
            ls = lab
    if ls and (ls < lp if lp else dk(lab) < 0.9 * len(data)):  # sections only when labels repeat a lot
        lab, sec = ls, True
    elif lp:
        lab = lp
    if not sec:  # low-cardinality key columns of lookup grids
        while (
            lab < min(c - 1, 4)
            and rk(lab) <= 0.7 * len(rv)
            and len({nx_norm(r[lab].text) for r in rv}) <= max(3, 0.5 * len(rv))
            and rk(lab + 1) > rk(lab)
        ):
            lab += 1
    return lab, sec


def infer_schema(g: Grid, max_title: int) -> Schema:
    """Title rows, header rows and label columns of a grid."""
    if not g:
        return Schema(0, 0, 0, False)
    rr, c = len(g), len(g[0])
    t = 0
    while t < max_title and rr - t > 1 and is_title_row(g[t]):
        t += 1
    rows = g[t:]
    th_h = 0
    if any(x.th for r in rows for x in r):
        while th_h < len(rows) and all(x.th or not x.text for x in rows[th_h]):
            th_h += 1
    k = 0  # an all-number th row is data
    while k < th_h and not (
        rows[k][0].text and all(is_numeric(x.text) and not _is_year(x.text) for x in rows[k] if x.text)
    ):
        k += 1
    th_h = k

    def headerish(s: str) -> bool:
        return not is_numeric(s) or _is_year(s)

    hh = 0  # without th markup: leading rows without numbers that label columns
    while (
        hh < len(rows)
        and all(headerish(x.text) for x in rows[hh][1:])
        and any(x.text and x.id != rows[hh][0].id for x in rows[hh][1:])
        and not is_section_row(rows[hh], 1)
    ):
        hh += 1
    if hh >= len(rows) - 1:
        hh = 0
    h = min(4, th_h or hh)
    lab, sec = _infer_labels(rows, h, c)
    return Schema(t, h, lab, sec)


@dataclass
class Triple:
    k: tuple[str, str]  # (row key, column key) sorted: orientation-free
    rk: str
    co: int  # value-column ordinal
    v: str
    ri: int
    cj: int
    raw_k: tuple[str, ...]
    gi: int = -1  # reference table index
    ti: int = -1  # output table index
    pos: int = 0


@dataclass
class Reading:
    triples: list[Triple]
    H: int  # informative header rows (constant rows excluded)
    title: str
    schema: Schema
    prefix: str
    label_head: str
    transposed: bool
    col_key: list[str] = field(default_factory=list)
    borrowed: bool = False  # headerless fragment read with the reference's column keys
    const_head: str = ""  # header rows spanning every value column (a parent level shared by all columns)


def _key(k: tuple[str, str]) -> str:
    return k[0] + "\u0001" + k[1]


def build_triples(
    tg: TableGrid,
    schema: Schema,
    ref_prefixes: Sequence[str] = (),
    borrow_keys: Sequence[str] | None = None,
    keep_marks: bool = False,
) -> Reading:
    """Triples of a grid read with ``schema``.

    ``ref_prefixes`` are parent header levels the reference drops; ``borrow_keys`` are the column keys of the
    paired reference table, used by a headerless fragment of the same width. ``keep_marks`` keeps cells holding
    an empty marker such as "-" or "n/a" (blank cells stay out).
    """
    t_, h, lab = schema.T, schema.H, schema.L
    g0 = tg.g
    if not g0:
        return Reading([], 0, "", schema, "", "", tg.transposed)
    c = len(g0[0])
    g = [list(r) for r in g0[t_:]]
    ext = _section_extent(g[h:], c)
    title = " ".join(r[0].text for r in g0[:t_])
    if tg.kind != "html":  # Markdown cannot express spans
        for i in range(min(h, len(g))):
            for j in range(lab + 1, c):
                if not g[i][j].text and g[i][j - 1].text:
                    g[i][j] = g[i][j - 1]
    for j in range(lab):  # row-label spans (not from section rows)
        for i in range(h + 1, len(g)):
            if (
                not g[i][j].text
                and g[i - 1][j].text
                and any(x.text for x in g[i][lab:])
                and not is_section_row(g0[t_ + i - 1], ext)
            ):
                g[i][j] = g[i - 1][j]
    head = g[:h]

    def is_const(r: Sequence[Slot]) -> bool:
        return c - lab >= 2 and len({nx_norm(x.text) for x in r[lab:]}) == 1

    const_rows = sum(1 for r in head if is_const(r))
    prefix = ""  # a parent level shared by every column carries no column identity
    if c - lab >= 2:
        cks = [p.split(" ") for p in (path_of(r[j] for r in head) for j in range(lab, c)) if p]
        if len(cks) >= 2:
            n = 0
            while all(len(a) > n + 1 and a[n] == cks[0][n] for a in cks):
                n += 1
            if n:
                prefix = " ".join(cks[0][:n]) + " "

    def strip(x: str) -> str:
        if prefix and x.startswith(prefix):
            return x[len(prefix) :]
        for p in ref_prefixes:
            if x.startswith(p):
                return x[len(p) :]
        return x

    borrowed = borrow_keys is not None and h == 0 and len(borrow_keys) == c
    col_key = (
        list(borrow_keys)
        if borrow_keys is not None and borrowed
        else [strip(path_of(r[j] for r in head)) for j in range(c)]
    )
    out: list[Triple] = []
    section = ""
    for i in range(h, len(g)):
        row = g[i]
        if is_section_row(row, ext):  # section labels hold no values
            if schema.sec:
                section = _section_text(row)
            continue
        rk = (section + " " if section else "") + path_of(row[:lab])
        raw = tuple(x.text for x in row[:lab] if x.text)
        for j in range(lab, c):
            v = row[j].text
            nv = nx_norm(v)
            if (nv in NX_EMPTY and not (keep_marks and nv)) or (
                j > 0 and row[j].id >= 0 and row[j].id == row[j - 1].id
            ):
                continue
            out.append(Triple(_sorted_pair(rk, col_key[j]), rk, j - lab, v, i, j, raw))
    informative = [r for r in head if not is_const(r)]
    label_head = " ".join(p for p in (path_of(r[j] for r in informative) for j in range(lab)) if p)
    const_head = " ".join(nx_norm(r[lab].text) for r in head if is_const(r))
    return Reading(out, h - const_rows, title, schema, prefix, label_head, tg.transposed, col_key, borrowed, const_head)


# ---------------------------------------------------------------------------
# Side-by-side tables emitted as one wide table
# ---------------------------------------------------------------------------


def _grid_cells(g: Grid) -> list[Cell]:
    cells: list[Cell] = []
    seen: dict[int, Cell] = {}
    for i, r in enumerate(g):
        for j, x in enumerate(r):
            if x.id < 0:
                cells.append(Cell(i, j, 1, 1, x.text, x.th))
                continue
            c = seen.get(x.id)
            if c is not None:
                c.rs = max(c.rs, i - c.r + 1)
                c.cs = max(c.cs, j - c.c + 1)
                continue
            n = Cell(i, j, 1, 1, x.text, x.th)
            seen[x.id] = n
            cells.append(n)
    return cells


def _slice(tg: TableGrid, lo: int, hi: int, first: bool) -> TableGrid:
    part = ParsedTable(tg.kind, _grid_cells([r[lo:hi] for r in tg.g]), tg.before if first else "", 0)
    return clean_grid(part)


def _non_empty(parts: list[TableGrid]) -> list[TableGrid] | None:
    p = [t for t in parts if t.g]
    return p if len(p) >= 2 else None


def split_side_by_side(tg: TableGrid, widths: Iterable[int]) -> list[TableGrid] | None:
    """Split a table whose header row repeats with the period of a reference width."""
    g = tg.g
    rr = len(g)
    c = len(g[0]) if rr else 0
    if rr < 2:
        return None
    for w in sorted(set(widths)):
        if w < 2 or c % w or c / w < 2:
            continue
        k = c // w
        for hr in range(min(4, rr)):
            row = [nx_norm(x.text) for x in g[hr]]
            if sum(1 for x in row[:w] if x) < 2:
                continue
            ok = all(row[b * w + j] == row[j] for b in range(1, k) for j in range(w))
            if ok:
                ok = not any(g[hr][b * w - 1].id >= 0 and g[hr][b * w - 1].id == g[hr][b * w].id for b in range(1, k))
            if not ok:
                continue
            return _non_empty([_slice(tg, b * w, (b + 1) * w, b == 0) for b in range(k)])
    return None


def split_adjacent(tg: TableGrid, refs: Sequence[TableGrid]) -> list[TableGrid] | None:
    """Reference tables that continue each other side by side (widths add up), emitted as one table."""
    g = tg.g
    c = len(g[0]) if g else 0
    if len(g) < 2:
        return None
    for a in range(len(refs)):
        total = 0
        for b in range(a, len(refs)):
            total += len(refs[b].g[0])
            if total > c:
                break
            if b == a or total != c:
                continue
            off, ok = 0, True
            for k in range(a, b + 1):
                w = len(refs[k].g[0])
                head = [nx_norm(x.text) for x in refs[k].g[0]]
                eq = sum(1 for j in range(w) if nx_norm(g[0][off + j].text) == head[j])
                if eq < 0.8 * w or (off and g[0][off - 1].id >= 0 and g[0][off - 1].id == g[0][off].id):
                    ok = False
                    break
                off += w
            if not ok:
                continue
            parts, off = [], 0
            for k in range(a, b + 1):
                w = len(refs[k].g[0])
                parts.append(_slice(tg, off, off + w, k == a))
                off += w
            return _non_empty(parts)
    return None


def split_stacked(tg: TableGrid, refs: Sequence[TableGrid]) -> list[TableGrid] | None:
    """Reference tables stacked one under the other and emitted as one table: a body row that repeats another
    reference table's header row (its cells first, the rest of the row empty) starts the next table."""
    g = tg.g
    heads = []
    for k, ref in enumerate(refs):
        h = [nx_norm(x.text) for x in ref.g[0]]
        if sum(1 for x in h if x) >= 2:
            heads.append((k, h))
    cuts = []
    for i in range(1, len(g)):
        row = [nx_norm(x.text) for x in g[i]]
        filled = [x for x in row if x]
        for _, h in heads:
            want = [x for x in h if x]
            if len(filled) == len(want) and filled == want and row[: len(h)] == h:
                cuts.append(i)
                break
    if not cuts:
        return None
    parts, lo = [], 0
    for hi in [*cuts, len(g)]:
        part = clean_grid(ParsedTable(tg.kind, _grid_cells(g[lo:hi]), tg.before if lo == 0 else "", 0))
        parts.append(TableGrid(part.g, 0, 0, part.kind, part.before))  # the padding belongs to the merge
        lo = hi
    return _non_empty(parts)


def _row_layout(r: Sequence[Slot]) -> tuple[tuple[int, bool], ...]:
    """Cells of a row as (columns spanned, has text)."""
    out: list[tuple[int, bool]] = []
    for j, x in enumerate(r):
        if j and x.id >= 0 and x.id == r[j - 1].id:
            out[-1] = (out[-1][0] + 1, out[-1][1])
        else:
            out.append((1, bool(x.text)))
    return tuple(out)


def _cut_rows(tg: TableGrid, cuts: Sequence[int]) -> list[TableGrid] | None:
    """The grid cut into stacked parts before each row index in ``cuts``."""
    parts, lo = [], 0
    for hi in [*cuts, len(tg.g)]:
        part = clean_grid(ParsedTable(tg.kind, _grid_cells(tg.g[lo:hi]), tg.before if lo == 0 else "", 0))
        parts.append(TableGrid(part.g, 0, 0, part.kind, part.before))
        lo = hi
    return _non_empty(parts) if cuts else None


def split_repeated_header_bands(tg: TableGrid) -> list[TableGrid] | None:
    """Sub-tables stacked in one table, each under its own copy of the header band.

    The band is the run of leading rows without numbers (at least two rows, one cell spanning two or more
    columns). A later run of rows without numbers with the same cell layout (texts may differ: other group
    names, sizes or periods) starts the next sub-table."""
    g = tg.g

    def plain(r: Sequence[Slot]) -> bool:
        return not any(is_numeric(x.text) for x in r[1:])

    h = 0
    while h < len(g) and plain(g[h]):
        h += 1
    band = [_row_layout(r) for r in g[:h]]
    if h < 2 or h >= len(g) or not any(w > 1 for r in band for w, _ in r):
        return None
    cuts, i = [], h + 1
    while i + h <= len(g):
        if all(plain(g[i + k]) and _row_layout(g[i + k]) == band[k] for k in range(h)):
            cuts.append(i)
            i += h + 1
        else:
            i += 1
    return _cut_rows(tg, cuts)


def unwrap_column_groups(tg: TableGrid, widths: Iterable[int]) -> tuple[TableGrid, int] | None:
    """One long list printed in ``k`` side-by-side column groups (an index of names and page numbers in four
    columns), emitted as one table ``k`` times as wide as a reference table that lists it top to bottom.

    Returns the table read group after group as one narrow table, and ``k``. The widest reference width that
    divides the table into at least two groups wins. A later group's leading rows that repeat the first group's
    header rows are dropped, and a cell spanning two groups rules the reading out."""
    g = tg.g
    c = len(g[0]) if g else 0
    ws = set(widths)
    if len(g) < 2 or c in ws:
        return None
    for w in sorted(ws, reverse=True):
        if w < 2 or c % w or c // w < 2:
            continue
        k = c // w
        if any(r[b * w - 1].id >= 0 and r[b * w - 1].id == r[b * w].id for r in g for b in range(1, k)):
            continue
        blocks = [[r[b * w : (b + 1) * w] for r in g] for b in range(k)]
        head = [[nx_norm(x.text) for x in r] for r in blocks[0]]
        rows = list(blocks[0])
        for blk in blocks[1:]:
            h = 0
            while h < min(4, len(blk)) and any(x.text for x in blk[h]) and [nx_norm(x.text) for x in blk[h]] == head[h]:
                h += 1
            rows.extend(blk[h:])
        part = clean_grid(ParsedTable(tg.kind, _grid_cells(rows), tg.before, 0))
        return (part, k) if part.g else None
    return None


def period_widths(tg: TableGrid) -> list[int]:
    """Repeating header periods of a table (no reference widths needed)."""
    g = tg.g
    c = len(g[0]) if g else 0
    out = []
    for p in range(2, c // 2 + 1):
        if c % p:
            continue
        for hr in range(min(3, len(g))):
            row = [nx_norm(x.text) for x in g[hr]]
            if sum(1 for x in row[:p] if x) < 2:
                continue
            if all(row[b * p + j] == row[j] for b in range(1, c // p) for j in range(p)):
                out.append(p)
                break
    return out


# ---------------------------------------------------------------------------
# Stacked sub-tables: one table with a mid-table header vs several tables
# ---------------------------------------------------------------------------


def split_header_bands(tg: TableGrid) -> list[TableGrid] | None:
    """Split a table before each mid-table header block: a run of all-``th`` rows that follows a data row.

    A page that stacks sub-tables in one box ("FY26" rows, then a "Comparison Period" band and its own
    column headers) reads either as one table or as one table per band. Applied to both sides, this puts
    the two readings on the same footing. Returns None when there is no band, or when the parts share a
    row label (the band then qualifies repeated labels, as section rows do, and must stay)."""
    g = tg.g

    def head(r: Sequence[Slot]) -> bool:
        return all(x.th or not x.text for x in r)

    cuts = [i for i in range(1, len(g)) if head(g[i]) and not head(g[i - 1])]
    if not cuts:
        return None
    parts, lo = [], 0
    for hi in [*cuts, len(g)]:
        parts.append(clean_grid(ParsedTable(tg.kind, _grid_cells(g[lo:hi]), tg.before if lo == 0 else "", 0)))
        lo = hi
    seen: set[str] = set()
    for p in parts:
        labels = {nx_norm(r[0].text) for r in p.g if not head(r) and r[0].text}
        if labels & seen:
            return None
        seen |= labels
    return _non_empty(parts)


def stitch_continuations(tables: Sequence[ParsedTable], md: str) -> tuple[list[ParsedTable], int, int]:
    """Join each output table to the previous one when both have the same width (empty columns included).

    The reverse reading of ``split_header_bands``: a page that draws one table as several boxes (split by
    an italic note or a second header block) is annotated as one table. Returns the tables, the joins with
    text between the two tables (the page breaks the table there: free) and the joins without (one major
    "table split" each, as a continuation read on its own costs)."""
    out: list[ParsedTable] = []
    free = glued = 0
    for t in tables:
        p = out[-1] if out else None
        w = max((x.c + x.cs for x in t.cells), default=0)
        if p is None or w < 2 or w != max((x.c + x.cs for x in p.cells), default=0):
            out.append(t)
            continue
        between = _trim(_RE_ANY_TAG.sub("", md[p.end : t.pos])) if p.end >= 0 else ""
        rows = max((x.r + x.rs for x in p.cells), default=0)
        cells = p.cells + [Cell(x.r + rows, x.c, x.rs, x.cs, x.text, x.th) for x in t.cells]
        out[-1] = ParsedTable(p.kind, cells, p.before, p.pos, t.end)
        if between:
            free += 1
        else:
            glued += 1
    return out, free, glued


def _n_tables(md: str | None) -> int:
    return sum(1 for t in parse_tables(md) if clean_grid(t).g)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class TableNextEvents:
    critical: int = 0
    major: int = 0
    minor: int = 0
    items: list[tuple[str, str, str, str]] = field(default_factory=list)  # (type, a, b, severity|info|detail)

    def push(self, sev: str | None, kind: str, a: object = "", b: object = "") -> None:
        if sev:
            setattr(self, sev, getattr(self, sev) + 1)
        self.items.append((kind, str(a), str(b), sev or "info"))


@dataclass
class TableNextResult:
    score: float
    proportional: float
    content: float
    row_order: float
    floor: float
    ref_cells: int
    tables_expected: int
    tables_output: int
    events: TableNextEvents

    def metadata(self, examples: int = 12) -> dict[str, Any]:
        types: dict[str, int] = {}
        for kind, _, _, sev in self.events.items:
            if sev != "detail":
                types[kind] = types.get(kind, 0) + 1
        shown = [
            {"type": k, "severity": s, "expected": a[:160], "output": b[:160]}
            for k, a, b, s in self.events.items
            if s in ("critical", "major", "minor")
        ][:examples]
        return {
            "events": {"critical": self.events.critical, "major": self.events.major, "minor": self.events.minor},
            "event_types": types,
            "event_examples": shown,
            "proportional": self.proportional,
            "floor": self.floor,
            "content": self.content,
            "row_order": self.row_order,
            "reference_cells": self.ref_cells,
            "tables_expected": self.tables_expected,
            "tables_output": self.tables_output,
        }


def _lis_len(a: Sequence[int]) -> int:
    t: list[int] = []
    for x in a:
        lo, hi = 0, len(t)
        while lo < hi:
            m = (lo + hi) >> 1
            if t[m] < x:
                lo = m + 1
            else:
                hi = m
        if lo == len(t):
            t.append(x)
        else:
            t[lo] = x
    return len(t)


def _moved_count(keys: Sequence[int], seq: Sequence[int], tables: bool) -> int:
    """Order events for reference rows (or columns) ``keys`` (table index x 10 000 + index) landing at ``seq``.

    A reference table whose rows all moved together (two tables emitted in swapped order) is one event: the
    tables are ordered by the median landing position of their rows (only when ``tables``; the row check
    charges a moved table, the column check does not charge it again), then each table's rows among themselves.
    Rows landing on the same output position (rows or columns merged into one) are in order here: the merge
    costs content, it reorders nothing.
    """

    def kept(a: Sequence[int]) -> int:  # longest non-decreasing run: equal positions keep their reference order
        return _lis_len([p * len(a) + i for i, p in enumerate(a)])

    by_t: dict[int, list[int]] = {}
    for k, p in zip(keys, seq, strict=True):
        by_t.setdefault(k // 10_000, []).append(p)
    meds = [sorted(v)[len(v) >> 1] for v in by_t.values()]
    return (len(meds) - kept(meds) if tables else 0) + sum(len(v) - kept(v) for v in by_t.values())


def _push_missing_rows(
    ev: TableNextEvents,
    G: Sequence[Triple],
    gts: Sequence[tuple[TableGrid, Schema, Reading]],
    matched: set[int],
    missing: Sequence[Sequence[int]],
) -> None:
    """One critical per missing reference row (``missing`` holds each row's indices into ``G``), except rows
    whose only values are one rowspan cell: one event for that cell, a major "spanned cell not repeated" when
    the output wrote the value in another row of the span, else one critical. A reference table none of whose
    rows came back is labelled "table missing" (info) and still costs its rows, so dropping a whole table never
    costs less than dropping part of it."""
    n_rows: dict[int, set[int]] = {}
    for x in G:
        n_rows.setdefault(x.gi, set()).add(x.ri)
    gone: dict[int, list[Sequence[int]]] = {}
    for idx in missing:
        gone.setdefault(G[idx[0]].gi, []).append(idx)

    def slot_id(i: int) -> int:
        tg, schema, _ = gts[G[i].gi]
        return tg.g[schema.T + G[i].ri][G[i].cj].id

    for gi, rows in gone.items():
        if len(rows) >= 2 and len(rows) == len(n_rows[gi]):
            ev.push(None, "table missing", gi, len(rows))
        spans: dict[frozenset[int], list[Sequence[int]]] = {}
        for idx in rows:
            spans.setdefault(frozenset(slot_id(i) for i in idx), []).append(idx)
        for ids, grp in spans.items():
            x0 = G[grp[0][0]]
            if len(grp) < 2 or -1 in ids:
                for idx in grp:
                    ev.push("critical", "row missing", " / ".join(G[idx[0]].k), "")
            elif any(i in matched and G[i].gi == gi and slot_id(i) in ids for i in range(len(G))):
                ev.push("major", "spanned cell not repeated", " / ".join(x0.k), len(grp))
            else:
                ev.push("critical", "row missing", " / ".join(x0.k), len(grp))


_RE_WORD = re.compile(r"[^\W_]+")


@dataclass
class _Read:
    tg: TableGrid
    tr: Reading
    hits: int


def _value_bag(tg: TableGrid) -> dict[str, int]:
    m: dict[str, int] = {}
    for r in tg.g:
        for x in r:
            v = nx_norm(x.text)
            if v not in NX_EMPTY:
                m[v] = m.get(v, 0) + 1
    return m


_RE_ROWSPAN = re.compile(rf"rowspan{_S}*={_S}*[\"']?0*(?:[2-9]|[1-9][0-9])", re.I)


_RE_DIGITS = re.compile(r"[0-9]+")


def _only_numbers_differ(a: str, b: str) -> bool:
    return (
        bool(nx_digits(a) and nx_digits(b))
        and nx_digits(a) != nx_digits(b)
        and _RE_DIGITS.sub("#", a) == _RE_DIGITS.sub("#", b)
    )


def _relabelled(a: str, b: str) -> bool:
    """Reference label ``a`` replaced by ``b``: words swapped one for one ("net sales" read "gross sales"), or no
    word kept. A label cut short, extended (a note appended, a section prefix lost) or merged with a cell keeps
    some words and changes the word count: the row is the same, and stays on the lenient once-per-table rule.
    So do the same words in another order, and a number used as the reference's row key (a row serial, or a
    value column read as the label)."""
    wa, wb = _RE_WORD.findall(a), _RE_WORD.findall(b)
    sa, sb = set(wa), set(wb)
    return bool(wa and wb) and not is_numeric(a) and sa != sb and (len(wa) == len(wb) or not sa & sb)


def _pct_key(x: Triple) -> str:
    return " ".join(x.k) + " " + " ".join(x.raw_k)


def _pair_rows_by_values(
    G: Sequence[Triple], P: Sequence[Triple], M: list[tuple[int, int, float, float, int]]
) -> set[tuple[int, int]]:
    """Pair a reference row and an output row that no cell matched but whose values agree column by column: the
    same row under a label that did not pair (reworded, merged with a note, keyed on a lost span). Without this
    the one row costs a missing row plus a repeated one; paired, it costs one "label partly differs"."""
    g_used, p_used = {m[0] for m in M}, {m[1] for m in M}
    g_rows: dict[tuple[int, int], list[int]] = {}
    p_rows: dict[tuple[int, int], list[int]] = {}
    for i, x in enumerate(G):
        g_rows.setdefault((x.gi, x.ri), []).append(i)
    for j, y in enumerate(P):
        p_rows.setdefault((y.ti, y.ri), []).append(j)
    open_g = [k for k, idx in g_rows.items() if not any(i in g_used for i in idx)]
    open_p = [k for k, idx in p_rows.items() if not any(j in p_used for j in idx)]
    if not open_g or not open_p:
        return set()
    g_rks = {nx_norm(x.rk) for x in G}
    # an output row under a label of the reference holds that row's values, not a relabelled row: values shifted
    # between rows stay wrong values
    open_p = [k for k in open_p if nx_norm(P[p_rows[k][0]].rk) not in g_rks]
    linked = {(G[m[0]].gi, P[m[1]].ti) for m in M}
    g_has = {gi for gi, _ in linked}
    cands: list[tuple[float, int, tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
    for gk in open_g:
        gidx = g_rows[gk]
        for pk in open_p:
            if gk[0] in g_has and (gk[0], pk[0]) not in linked:
                continue
            by_co: dict[int, list[int]] = {}
            for j in p_rows[pk]:
                by_co.setdefault(P[j].co, []).append(j)
            pairs = []
            for i in gidx:
                hit = next((j for j in by_co.get(G[i].co, ()) if value_sim(G[i].v, P[j].v, _pct_key(G[i])) == 1), None)
                if hit is not None:
                    pairs.append((i, hit))
            n = len(pairs)
            if n >= 2 and n >= 0.8 * len(gidx) and n >= 0.8 * len(p_rows[pk]):
                cands.append((-n / max(len(gidx), len(p_rows[pk])), abs(gk[1] - pk[1]), gk, pk, pairs))
    taken_g: set[tuple[int, int]] = set()
    taken_p: set[tuple[int, int]] = set()
    added: set[tuple[int, int]] = set()
    for _, _, gk, pk, pairs in sorted(cands, key=lambda c: (c[0], c[1], c[2], c[3])):
        if gk in taken_g or pk in taken_p:
            continue
        taken_g.add(gk)
        taken_p.add(pk)
        M.extend((i, j, 0.5, 0.5, 0) for i, j in pairs)
        added.update(pairs)
    return added


def merge_label_span_groups(expected_markdown: str | None) -> str | None:
    """The reference with each group of body rows under one first-column rowspan cell merged into one row, every
    column's cell texts joined in reading order: the reading of sub-rows the page stacks as lines inside one drawn
    cell. None when no reference table has such a group (only HTML tables carry row spans)."""
    md = expected_markdown or ""
    changed = False

    def merge(m: re.Match[str]) -> str:
        nonlocal changed
        cells = parse_tables(m.group(0))[0].cells
        nr = max((x.r + x.rs for x in cells), default=0)
        groups: dict[int, int] = {}  # first row -> row count
        for x in cells:
            if x.c == 0 and x.rs >= 2 and not x.th and x.r + x.rs <= nr:
                lo, hi = x.r, x.r + x.rs
                inside = [y for y in cells if y.r < hi and y.r + y.rs > lo]
                if all(lo <= y.r and y.r + y.rs <= hi and y.cs == 1 for y in inside):
                    groups[lo] = x.rs
        if not groups:
            return m.group(0)
        changed = True
        out, r = ["<table>"], 0
        while r < nr:
            n = groups.get(r, 1)
            row = sorted((y for y in cells if r <= y.r < r + n), key=lambda y: (y.c, y.r))
            if n == 1:
                tds = [(y.text, y.th, y.rs, y.cs) for y in row]
            else:
                by_col: dict[int, list[str]] = {}
                for y in row:
                    by_col.setdefault(y.c, []).append(y.text)
                tds = [(" ".join(t for t in by_col[c] if t), False, 1, 1) for c in sorted(by_col)]
            out.append("<tr>")
            for t, th, rs, cs in tds:
                tag = "th" if th else "td"
                out.append(f'<{tag} rowspan="{rs}" colspan="{cs}">{html.escape(t, quote=False)}</{tag}>')
            out.append("</tr>")
            r += n
        return "\n".join(out + ["</table>"])

    merged = _RE_TABLE.sub(merge, md)
    return merged if changed else None


def score_table_next(
    expected_markdown: str | None,
    output_markdown: str | None,
    *,
    max_top_title_rows: int | None = 1,
    allow_splitting_ambiguous_merged_tables: bool = False,
    trm_unsupported: bool = False,
    allow_merged_sub_rows: bool = False,
) -> TableNextResult:
    """Score the tables of ``output_markdown`` against the tables of ``expected_markdown``.

    Two grid normalizations are readings, scored only when they apply and kept only when they score higher:
    joining a leading column of row serials into the row label (``join_serial_column``) and keeping the rows that
    only number the columns (``drop_column_index_rows`` drops them by default). A side may print "(1)" as its own
    header row or inside the header text ("(1) Acc Yr 2014"), and "(1) | Losses Paid" or "(1) Losses Paid"; the
    reading where both sides agree scores highest, so a faithful output never loses points to either rule.
    See ``_score_readings`` for the other readings."""
    serials, index_rows, marks, spaceless = _normalizations_apply(expected_markdown, output_markdown)
    best: TableNextResult | None = None
    for attach in (True, False) if marks else (True,):
        for spaced in (True, False) if spaceless else (True,):
            for join in (False, True) if serials else (False,):
                for drop in (True, False) if index_rows else (True,):
                    tokens = (_ATTACH_MARKS.set(attach), _SPACELESS_KEYS.set(spaced))
                    try:
                        res = _score_readings(
                            expected_markdown,
                            output_markdown,
                            max_top_title_rows=max_top_title_rows,
                            allow_splitting_ambiguous_merged_tables=allow_splitting_ambiguous_merged_tables,
                            trm_unsupported=trm_unsupported,
                            allow_merged_sub_rows=allow_merged_sub_rows,
                            join_serials=join,
                            drop_index=drop,
                        )
                    finally:
                        _ATTACH_MARKS.reset(tokens[0])
                        _SPACELESS_KEYS.reset(tokens[1])
                    if best is None or res.score > best.score:
                        best = res
                    if best.score >= 1.0:
                        return best
    assert best is not None
    return best


def _normalizations_apply(expected_markdown: str | None, output_markdown: str | None) -> tuple[bool, bool, bool, bool]:
    """Which normalizations can change the reading: a row-serial column (``join_serial_column``), a column-index
    row (``drop_column_index_rows``), a <sup>/<sub> marker with spaces around it (``_RE_SUPSUB_WS``), and two cell
    texts equal once spaces are removed (``key_sim``)."""
    grids = [t for md in (expected_markdown, output_markdown) for t in (clean_grid(x) for x in parse_tables(md)) if t.g]
    serials = any(join_serial_column(drop_column_index_rows(t)) is not drop_column_index_rows(t) for t in grids)
    marks = any(
        m.group(0) != (m.group(1) or m.group(2))
        for md in (expected_markdown, output_markdown)
        for m in _RE_SUPSUB_WS.finditer(md or "")
    )
    by_stripped: dict[str, set[str]] = {}
    for tg in grids:
        for r in tg.g:
            for x in r:
                k = nx_norm(x.text)
                if " " in k:
                    by_stripped.setdefault(k.replace(" ", ""), set()).add(k)
    texts = {nx_norm(x.text) for tg in grids for r in tg.g for x in r if x.text and " " not in nx_norm(x.text)}
    spaceless = any(len(v) > 1 or k in texts for k, v in by_stripped.items())
    return serials, any(drop_column_index_rows(t) is not t for t in grids), marks, spaceless


def _score_readings(
    expected_markdown: str | None,
    output_markdown: str | None,
    *,
    max_top_title_rows: int | None,
    allow_splitting_ambiguous_merged_tables: bool,
    trm_unsupported: bool,
    allow_merged_sub_rows: bool,
    join_serials: bool,
    drop_index: bool,
) -> TableNextResult:
    """The best of the alternate readings of the pair, under one choice of the two grid normalizations.

    An output with row spans is also read with every rowspan as 1; when that reading scores higher, the spans
    were pushing cells into the wrong columns, which costs one major ("wrong row span") instead of the missing
    and repeated rows the shifted cells would otherwise produce.

    With ``allow_merged_sub_rows`` (a page that draws one cell per first-column span group and stacks the
    reference's sub-rows as lines inside it), the reference is also read with each group merged into one row
    (``merge_label_span_groups``), at no cost, and the higher score is kept.

    A row label may sit in two columns on one side and in one on the other ("1986 | Q3" vs "1986 Q3"), which
    offsets every value column by one. When the joined labels appear on the other side, each side's first two
    columns are also joined (``merge_label_columns``); that reading costs one major ("label columns merged").

    When the two sides hold a different number of tables, the band-split and the stitched segmentations are
    also scored (``segmentation``: see split_header_bands, stitch_continuations).

    An output table that wraps a reference-width list into column groups is also read group after group
    (``unwrap_column_groups``), one major per extra group."""

    def run(
        row_spans: bool, merge: str = "", segmentation: str = "", unwrap_groups: bool = False
    ) -> TableNextResult | None:
        return _score_table_next(
            expected_markdown,
            output_markdown,
            row_spans=row_spans,
            unwrap_groups=unwrap_groups,
            max_top_title_rows=max_top_title_rows,
            allow_splitting_ambiguous_merged_tables=allow_splitting_ambiguous_merged_tables,
            trm_unsupported=trm_unsupported,
            merge_labels=merge,
            segmentation=segmentation,
            join_serials=join_serials,
            drop_index=drop_index,
        )

    spans = [True, False] if output_markdown and _RE_ROWSPAN.search(output_markdown) else [True]
    res = run(True)
    assert res is not None
    best_rs = True  # the rowspan reading the wrapped-groups reading builds on
    for rs in spans[1:]:
        alt = run(rs)
        if alt is not None and alt.score > res.score:
            res, best_rs = alt, rs
    if res.score < 1:
        for rs in spans:
            for merge in ("ref", "out"):
                alt = run(rs, merge)
                if alt is not None and alt.score > res.score:
                    res = alt
    # stacked sub-tables segmented differently on the two sides: also score the band-split and the stitched
    # readings (see split_header_bands, stitch_continuations) and keep the best
    if res.score < 1 and _n_tables(expected_markdown) != _n_tables(output_markdown):
        for seg in ("bands", "stitch"):
            alt = run(True, segmentation=seg)
            if alt is not None and alt.score > res.score:
                res = alt
    if _has_wrapped_column_groups(expected_markdown, output_markdown, best_rs):
        wrapped = run(best_rs, unwrap_groups=True)
        if wrapped is not None and wrapped.score > res.score:
            res = wrapped
    merged = merge_label_span_groups(expected_markdown) if allow_merged_sub_rows and output_markdown else None
    if merged is not None and res.score < 1.0:
        alt = _score_table_next(
            merged,
            output_markdown,
            row_spans=True,
            max_top_title_rows=max_top_title_rows,
            allow_splitting_ambiguous_merged_tables=allow_splitting_ambiguous_merged_tables,
            trm_unsupported=trm_unsupported,
            merged_subrows=True,
            join_serials=join_serials,
            drop_index=drop_index,
        )
        if alt is not None and alt.score > res.score:
            res = alt
    return res


def _has_wrapped_column_groups(expected_markdown: str | None, output_markdown: str | None, row_spans: bool) -> bool:
    """Whether some output table reads as a reference-width list wrapped into column groups (cheap pre-check)."""
    widths = [len(t.g[0]) for t in (clean_grid(x) for x in parse_tables(expected_markdown)) if t.g]
    return any(
        unwrap_column_groups(drop_header_only_columns(drop_span_copy_rows(drop_column_index_rows(t))), widths)
        for t in (clean_grid(x) for x in parse_tables(output_markdown, row_spans))
        if t.g
    )


def _score_table_next(  # noqa: C901 - one scoring pass
    expected_markdown: str | None,
    output_markdown: str | None,
    *,
    row_spans: bool,
    unwrap_groups: bool = False,
    max_top_title_rows: int | None = 1,
    allow_splitting_ambiguous_merged_tables: bool = False,
    trm_unsupported: bool = False,
    merge_labels: str = "",
    segmentation: str = "",
    merged_subrows: bool = False,
    join_serials: bool = False,
    drop_index: bool = True,
) -> TableNextResult | None:
    """One reading of the pair. ``merge_labels`` ("ref" or "out") joins the first two columns of that side's
    tables whose two-cell row labels the other side writes in one cell; None when there is no such table.
    ``segmentation`` ("bands" or "stitch") splits both sides at mid-table header bands or joins consecutive
    output tables of the same width. ``merged_subrows`` marks a reference read with its first-column span
    groups merged into one row (``allow_merged_sub_rows``). ``join_serials`` and ``drop_index`` choose the grid
    normalizations (see ``score_table_next``)."""

    def index_rows(tg: TableGrid) -> TableGrid:
        return drop_column_index_rows(tg) if drop_index else tg

    max_title = max(0, max_top_title_rows) if isinstance(max_top_title_rows, int) else 1
    allow_split, trm = bool(allow_splitting_ambiguous_merged_tables), bool(trm_unsupported)
    ev = TableNextEvents()
    if not row_spans:
        ev.push("major", "wrong row span")
    if merged_subrows:
        ev.push(None, "sub-rows merged")

    g_grids = [
        drop_span_copy_rows(index_rows(t)) for t in (clean_grid(x) for x in parse_tables(expected_markdown)) if t.g
    ]
    o_parsed = parse_tables(output_markdown, row_spans)
    if segmentation == "stitch":
        o_parsed, free, glued = stitch_continuations(o_parsed, output_markdown or "")
        ev.push(None, "tables stitched", "", free + glued)
        for _ in range(glued):
            ev.push("major", "table split", "", "")
    o_grids = [
        drop_header_only_columns(drop_span_copy_rows(index_rows(t))) for t in (clean_grid(x) for x in o_parsed) if t.g
    ]
    if segmentation == "bands":
        g_grids = [p for t in g_grids for p in (split_header_bands(t) or [t])]
        o_grids = [p for t in o_grids for p in (split_header_bands(t) or [t])]
        ev.push(None, "tables split at header bands")
    # a reference table restating its header band mid-body holds stacked sub-tables: score them as separate
    # tables; an output kept as one table is cut where it repeats a band's first row (the same reading, no event)
    bands = [split_repeated_header_bands(t) or [t] for t in g_grids]
    if any(len(b) > 1 for b in bands):
        titles = {_section_text(p.g[0]) for b in bands for p in b[1:]}
        g_grids = [p for b in bands for p in b]
        o_grids = [
            p
            for t in o_grids
            for p in (_cut_rows(t, [i for i, r in enumerate(t.g) if i and _section_text(r) in titles]) or [t])
        ]
    if merge_labels:
        m_side = g_grids if merge_labels == "ref" else o_grids
        m_other = {nx_norm(x.text) for t in (o_grids if merge_labels == "ref" else g_grids) for r in t.g for x in r}

        def labels_joined(t: TableGrid) -> int:  # this table's two-cell labels the other side writes in one cell
            if not t.g or len(t.g[0]) < 3:
                return 0
            jl = {nx_norm(r[0].text + " " + r[1].text) for r in t.g if r[0].text and r[1].text and r[0].id != r[1].id}
            n = len(jl & m_other)
            return n if n >= max(1, 0.3 * len(jl)) else 0

        m_found = [labels_joined(t) for t in m_side]
        m_hit = [mi for mi, n in enumerate(m_found) if n]
        if sum(m_found) < 2:
            return None
        for mi in m_hit:
            m_side[mi] = merge_label_columns(m_side[mi])
        ev.push("major", "label columns merged", merge_labels, len(m_hit))
    split_info: list[tuple[int, int, str]] = []
    widths = [len(t.g[0]) for t in g_grids]
    if len(g_grids) > 1 or allow_split:
        n_out, new = len(o_grids), []
        for ti, t in enumerate(o_grids):
            if n_out >= len(g_grids) and not allow_split:
                new.append(t)
                continue
            kind = "side-by-side tables merged"
            parts = split_side_by_side(t, widths + period_widths(t) if allow_split else widths) or split_adjacent(
                t, g_grids
            )
            if not parts:
                kind, parts = "tables merged", split_stacked(t, g_grids)
            if not parts:
                new.append(t)
                continue
            split_info.append((ti, len(parts), kind))
            new.extend(parts)
        o_grids = new
        if allow_split and len(g_grids) < len(o_grids):  # a reference drawn as one wide table, output split
            o_widths = [len(t.g[0]) for t in o_grids]
            g_grids = [p for t in g_grids for p in (split_side_by_side(t, o_widths) or [t])]
    if unwrap_groups:  # one major per extra column group, as for a table split into that many parts
        unwrapped: list[TableGrid] = []
        for t in o_grids:
            u = unwrap_column_groups(t, widths)
            if u:
                for _ in range(1, u[1]):
                    ev.push("major", "column groups wrapped", len(unwrapped), u[1])
            unwrapped.append(u[0] if u else t)
        o_grids = unwrapped
    ref_cells = sum(len({x.id for r in t.g for x in r if x.text}) for t in g_grids)  # serial cells count too
    if join_serials:
        g_grids = [join_serial_column(t) for t in g_grids]
        o_grids = [join_serial_column(t) for t in o_grids]

    gts = []
    for tg in g_grids:
        schema = infer_schema(tg.g, max_title)
        gts.append((tg, schema, build_triples(tg, schema)))
    # a table of dashes with one or two values (a footer factor): its dash cells are scored too, so one stray
    # value does not decide the page; output tables are then read the same way
    marks = 0 < sum(len(t[2].triples) for t in gts) < MIN_VALUES
    if marks:
        gts = [(tg, schema, build_triples(tg, schema, keep_marks=True)) for tg, schema, _ in gts]
    G: list[Triple] = []
    for gi, (_, _, tr) in enumerate(gts):
        for x in tr.triples:
            G.append(Triple(x.k, x.rk, x.co, x.v, x.ri, x.cj, x.raw_k, gi=gi))
    g_keys: set[str] = {_key(x.k) for x in G}
    g_labels: dict[str, None] = dict.fromkeys(lab for x in G for lab in x.k)

    def result(
        score: float, prop: float, content: float, order: float, floor: float, ref_cells: int
    ) -> TableNextResult:
        return TableNextResult(score, prop, content, order, floor, ref_cells, len(gts), len(o_grids), ev)

    if not G:
        return result(1.0, 1.0, 1.0, 1.0, 1.0, 0)
    if not o_grids:
        ev.push("critical", "table missing")
        return result(0.0, 0.0, 0.0, 0.0, 1.0 - WEIGHTS["critical"], 0)

    # output: each table read with the reference structure that fits it best
    g_bags = [_value_bag(t[0]) for t in gts]

    def kv(x: Triple) -> str:
        return _key(x.k) + "\u0002" + nx_norm(x.v)

    kv_rows: dict[str, list[tuple[int, int]]] = {}
    for x in G:
        kv_rows.setdefault(kv(x), []).append((x.gi, x.ri))

    def fit(T: Sequence[Triple], gis: set[int] | None) -> int:
        """Per output row, the most cells (exact key and value) it shares with one reference row."""
        rows: dict[int, dict[tuple[int, int], int]] = {}
        for x in T:
            lst = kv_rows.get(kv(x))
            if not lst:
                continue
            m = rows.setdefault(x.ri, {})
            for r in dict.fromkeys(lst):
                if gis is None or r[0] in gis:
                    m[r] = m.get(r, 0) + 1
        return sum(max(m.values(), default=0) for m in rows.values())

    # a row label wrapped over two rows where the reference spans one cell: one minor per table
    wrapped = set().union(*(span_labels(t[0], t[1]) for t in gts))
    if wrapped:
        for ti, tg in enumerate(o_grids):
            o_grids[ti], joins = join_wrapped_labels(tg, wrapped)
            if joins:
                ev.push("minor", "label split across rows", ti, joins)

    reads: list[_Read] = []
    for tg in o_grids:
        bag = _value_bag(tg)
        aff = [sum(min(n, gb[v]) for v, n in bag.items() if v in gb) for gb in g_bags]
        ref_idx = [gi for a, gi in sorted(((a, gi) for gi, a in enumerate(aff) if a > 0), key=lambda p: (-p[0], p[1]))][
            :3
        ]
        refs = [gts[gi][1] for gi in ref_idx]
        prefixes = list(dict.fromkeys(gts[gi][2].prefix for gi in ref_idx if gts[gi][2].prefix))
        top = aff[ref_idx[0]] if ref_idx else 0
        pair = {gi for gi in ref_idx if aff[gi] >= 0.5 * top}

        def orient(
            otg: TableGrid,
            refs: list[Schema] = refs,
            prefixes: list[str] = prefixes,
            pair: set[int] = pair,
            ref_idx: list[int] = ref_idx,
        ) -> tuple[Reading, int]:
            own = infer_schema(otg.g, max_title)
            c, rr = len(otg.g[0]), len(otg.g)
            lead = 0
            while lead < rr - 1 and is_title_row(otg.g[lead]):
                lead += 1
            cand: dict[Schema, None] = {}

            def add(s: Schema) -> None:
                cand.setdefault(
                    Schema(min(s.T, rr - 1), max(0, min(4, s.H)), max(1 if c > 1 else 0, min(s.L, c - 1)), bool(s.sec)),
                    None,
                )

            for r in refs:
                add(r)
                add(Schema(min(r.T, lead), r.H, r.L, r.sec))
                add(Schema(r.T, r.H, own.L, r.sec))
                for h in (own.H, r.H - 1, r.H + 1):
                    add(Schema(r.T, h, r.L, r.sec))
                add(Schema(own.T, own.H, r.L, own.sec))
                add(Schema(own.T, r.H, r.L, r.sec))
            add(own)
            best: Reading | None = None
            bh = -1
            for s in cand:
                for pf in ([], prefixes) if prefixes else ([],):
                    tr = build_triples(otg, s, pf, keep_marks=marks)
                    h = fit(tr.triples, pair or None)
                    if h > bh:
                        best, bh = tr, h
            if own.H == 0:  # a headerless fragment (a table continued after a break) takes the reference's column keys
                for gi in ref_idx:
                    rg, rs_, rt = gts[gi]
                    if len(rg.g[0]) != c or not rs_.H:
                        continue
                    tr = build_triples(otg, Schema(0, 0, rs_.L, rs_.sec), (), rt.col_key, marks)
                    if not tr.triples:
                        continue
                    r0 = min(x.ri for x in tr.triples)
                    if not any(
                        x.ri == r0 and kv(x) in kv_rows for x in tr.triples
                    ):  # its first row must already be data
                        continue
                    h = fit(tr.triples, pair or None)
                    if h > bh:
                        best, bh = tr, h
            assert best is not None
            return best, bh

        a_tr, a_h = orient(tg)
        b_tr, b_h = orient(transpose_grid(tg))
        pick = a_tr
        if b_h > a_h:
            pick = b_tr
        elif b_h == a_h and b_h > 0:  # tie: shape closest to a reference table
            rr, c = len(tg.g), len(tg.g[0])

            def dist(r: int, cc: int) -> int:
                return min(abs(r - len(t[0].g)) + abs(cc - len(t[0].g[0])) for t in gts)

            if dist(c, rr) < dist(rr, c):
                pick = b_tr
        reads.append(_Read(tg, pick, fit(pick.triples, None)))

    # output tables sharing no key and few labels with the reference are tables it does not annotate
    used = list(range(len(reads)))
    if len(reads) > 1:
        keep = []
        for ti in used:
            T = reads[ti].tr.triples
            n_lab = sum(1 for x in T if any(k in g_labels for k in x.k))
            if reads[ti].hits > 0 or n_lab >= max(2, 0.5 * len(T)):
                keep.append(ti)
        if keep and len(keep) < len(used):
            ev.push(None, "unpaired output table ignored", "", len(used) - len(keep))
            used = keep
    PT = [reads[ti] for ti in used]
    P: list[Triple] = []
    for ti, rd in enumerate(PT):
        for x in rd.tr.triples:
            P.append(Triple(x.k, x.rk, x.co, x.v, x.ri, x.cj, x.raw_k, ti=ti, pos=ti * 10_000 + x.ri))
    if not P:
        ev.push("critical", "table missing")
        return result(0.0, 0.0, 0.0, 0.0, 1.0 - WEIGHTS["critical"], 0)
    scale = min(1.0, K_SCALE / max(1, ref_cells))

    # reworded labels: an output label absent from the reference carrying the values of a missing reference label
    alias: dict[str, str] = {}

    def vkey(v: str) -> str:
        n = nx_norm(v)
        x = nx_number(n)
        return f"#{x.cmp}{_num_key(x.v)}{x.pct}{x.unit}" if x else n

    p_by: dict[str, dict[str, list[str]]] = {}
    for y in P:
        for side, lab in enumerate(y.k):
            if lab in g_labels:
                continue
            p_by.setdefault(lab, {}).setdefault(vkey(y.v), []).append(y.k[1 - side])
    g_by: dict[str, list[tuple[str, str]]] = {}
    for x in G:
        for side, lab in enumerate(x.k):
            g_by.setdefault(lab, []).append((vkey(x.v), x.k[1 - side]))
    p_has = {lab for y in P for lab in y.k}
    taken: set[str] = set()
    reworded: set[int] = set()
    g_rks = {x.rk for x in G}
    alias_w: dict[str, float] = {}  # output row label -> content weight of its cells
    for gl in (lab for lab in g_labels if lab not in p_has):
        gc = g_by[gl]
        best: tuple[str, int] | None = None
        for pl, by_value in p_by.items():
            if pl in taken:
                continue
            h = 0
            for vk, other in gc:
                os_ = by_value.get(vk)
                if os_ and any(key_sim(other, o) > 0 for o in os_):
                    h += 1
            if h >= max(1, 0.6 * len(gc)) and (best is None or h > best[1]):
                best = (pl, h)
        if best is None:
            continue
        alias[gl] = best[0]
        taken.add(best[0])
        gi0 = next(x.gi for x in G if gl in x.k)
        sim = 1 - OSA.distance(gl, best[0]) / max(len(gl), len(best[0]), 1)  # a transposition is one typo
        if sim < 0.8 and gl in g_rks and _relabelled(gl, best[0]):
            alias_w[best[0]] = LCSseq.similarity(gl, best[0]) / max(1, len(gl))  # relabelled: label recall
        if _only_numbers_differ(gl, best[0]):  # same words, another year or number: the values are misattributed
            ev.push("critical", "label number changed", gl, best[0])
        elif gi0 not in reworded:
            reworded.add(gi0)
            ev.push("minor" if sim >= 0.8 else "major", "label reworded", gl, best[0])
        else:
            ev.push(None, "label reworded", gl, best[0])
    inv = {p: g for g, p in alias.items()}

    def canon(lab: str) -> str:
        return inv.get(lab, lab)

    # sparse candidate pairs: only triples with compatible labels can match
    pc = [(canon(y.k[0]), canon(y.k[1])) for y in P]
    p_index: dict[str, list[int]] = {}
    for j, k in enumerate(pc):
        p_index.setdefault(_key(_sorted_pair(*k)), []).append(j)
    p_labels = list(dict.fromkeys(lab for k in pc for lab in k))
    sim_cache: dict[str, list[tuple[str, float]]] = {}
    ks_cache: dict[tuple[str, str], float] = {}

    def similar(gl: str) -> list[tuple[str, float]]:
        r = sim_cache.get(gl)
        if r is None:
            r = [(pl, s) for pl in p_labels if (s := key_sim(gl, pl)) > 0]
            sim_cache[gl] = r
        return r

    def ksim(a: str, b: str) -> float:
        v = ks_cache.get((a, b))
        if v is None:
            v = ks_cache[(a, b)] = key_sim(a, b)
        return v

    p_row: dict[str, list[int]] = {}  # trm_unsupported: a cell may also match by row label and column position
    if trm:
        for j, y in enumerate(P):
            if not PT[y.ti].tr.transposed:
                p_row.setdefault(canon(y.rk), []).append(j)
    edges: list[tuple[int, int, float, float, int]] = []  # (i, j, S, label similarity, matched by position)
    for i, x in enumerate(G):
        a, b = x.k
        sa, sb = similar(a), similar(b)
        cand: dict[int, float] = {}
        for pa, _ in sa:
            for pb, _ in sb:
                for j in p_index.get(_key(_sorted_pair(pa, pb)), ()):
                    cand[j] = 0.0
        if trm:
            for pr, rs in similar(x.rk):
                for j in p_row.get(pr, ()):
                    if P[j].co == x.co and j not in cand:
                        cand[j] = rs
        for j, pos in cand.items():
            bk = pc[j]
            kn = max(ksim(a, bk[0]) * ksim(b, bk[1]), ksim(a, bk[1]) * ksim(b, bk[0]))
            ks = max(kn, pos)
            if not ks:
                continue
            sv = ks * value_sim(x.v, P[j].v, _pct_key(x))
            if sv > 0:
                edges.append((i, j, sv, ks, 1 if pos > kn else 0))

    # connected components; Hungarian inside each (greedy above BIG_COMPONENT)
    parent: dict[str, str] = {}

    def find(u: str) -> str:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for i, j, *_ in edges:
        ga, pb_ = f"g{i}", f"p{j}"
        parent.setdefault(ga, ga)
        parent.setdefault(pb_, pb_)
        ra, rb = find(ga), find(pb_)
        if ra != rb:
            parent[ra] = rb
    comps: dict[str, list[tuple[int, int, float, float, int]]] = {}
    for e in edges:
        comps.setdefault(find(f"g{e[0]}"), []).append(e)
    M: list[tuple[int, int, float, float, int]] = []
    for es in comps.values():
        if len(es) == 1:
            M.append(es[0])
            continue
        gi_l = sorted({e[0] for e in es})
        pj_l = sorted({e[1] for e in es})
        if max(len(gi_l), len(pj_l)) > BIG_COMPONENT:
            ug: set[int] = set()
            up: set[int] = set()
            for e in sorted(es, key=lambda e: (-e[2], e[0], e[1])):
                if e[0] not in ug and e[1] not in up:
                    ug.add(e[0])
                    up.add(e[1])
                    M.append(e)
            continue
        gix = {v: k for k, v in enumerate(gi_l)}
        pjx = {v: k for k, v in enumerate(pj_l)}
        S = np.zeros((len(gi_l), len(pj_l)))
        by_pair: dict[tuple[int, int], tuple[int, int, float, float, int]] = {}
        for e in es:
            S[gix[e[0]], pjx[e[1]]] = e[2]
            by_pair.setdefault((e[0], e[1]), e)
        # a tiny convex position tie-break keeps the optimum unique (same choice in any Hungarian implementation)
        d = (np.arange(len(gi_l)) / len(gi_l))[:, None] - (np.arange(len(pj_l)) / len(pj_l))[None, :]
        cost_m = 1 - S + 1e-10 * d * d
        for ra_, cb_ in zip(*linear_sum_assignment(cost_m), strict=True):
            if S[ra_, cb_] > 0:
                M.append(by_pair[(gi_l[ra_], pj_l[cb_])])
    by_values = _pair_rows_by_values(G, P, M)  # already charged once: kept out of the order checks
    M.sort(key=lambda m: m[0])
    tot = 0.0
    for mt in M:
        tot += mt[2] * alias_w.get(P[mt[1]].rk, 1.0)
    p_ = tot / len(P)
    r_ = tot / len(G)
    f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
    gm = {m[0]: m for m in M}
    p_matched = {m[1] for m in M}

    # reference cells: missing rows, wrong numbers (critical), wrong text, lost units;
    # a column failing in >= 3 rows counts once
    rows_by: dict[str, list[int]] = {}
    for i, x in enumerate(G):
        rows_by.setdefault(f"{x.gi}:{x.ri}", []).append(i)
    cell_ev: list[tuple[str, str, str, str, str, str]] = []  # (row, col, sev, type, a, b)
    pos_matched: set[int] = set()
    p_label_set = {lab for k in pc for lab in k}
    missing_rows: list[list[int]] = []
    for idx in rows_by.values():
        x0 = G[idx[0]]
        if all(i not in gm for i in idx) and x0.rk not in p_label_set:
            missing_rows.append(idx)
            continue
        rk0 = f"{x0.gi}:{x0.ri}"
        for i in idx:
            mm = gm.get(i)
            jm = mm[1] if mm else None
            sc = mm[2] if mm else 0.0
            v = G[i].v
            if mm and mm[4]:
                pos_matched.add(G[i].gi)
            if sc >= 0.999:
                continue
            pk_ = _pct_key(G[i])
            if mm and jm is not None and mm[4] and value_sim(v, P[jm].v, pk_) == 1:
                continue
            if mm and jm is not None and mm[3] < 0.999 and not mm[4] and value_sim(v, P[jm].v, pk_) == 1:
                typo = mm[3] >= 0.8 and (i, jm) not in by_values  # the same words up to a typo
                cell_ev.append(
                    (
                        rk0,
                        f"{G[i].gi}:labels",
                        "minor" if typo else "major",
                        "label typo" if typo else "label partly differs",
                        " / ".join(G[i].k),
                        " / ".join(P[jm].k),
                    )
                )
                continue
            pv = P[jm].v if jm is not None else ""
            vs = value_sim(v, pv, pk_) if jm is not None else 0.0
            col = f"{G[i].gi}:{G[i].cj}"
            if vs in (0.95, 0.9):
                cell_ev.append(
                    (rk0, col, "minor", "letter case changed" if vs == 0.95 else "number format changed", v, pv)
                )
            elif vs == 0.5:
                cell_ev.append((rk0, col, "major", "unit lost", v, pv))
            elif re.search(r"[0-9]", v) and (sc == 0 or nx_digits(v) != nx_digits(pv)):
                cell_ev.append((rk0, col, "critical", "wrong or misplaced number", " / ".join(G[i].k) + ": " + v, pv))
            elif sc >= 0.8:
                cell_ev.append((rk0, col, "minor", "small text difference", v, pv))
            else:
                cell_ev.append((rk0, col, "major", "wrong or misplaced text", v, pv))
    _push_missing_rows(ev, G, gts, set(gm), missing_rows)
    col_count: dict[str, int] = {}
    for ce in cell_ev:
        col_count[ce[1] + "|" + ce[2]] = col_count.get(ce[1] + "|" + ce[2], 0) + 1
    per_row: set[str] = set()
    for row_k, col, sev, kind, a_txt, b_txt in cell_ev:
        ev.items.append((kind, a_txt, b_txt, "detail"))
        if col_count[col + "|" + sev] >= 3:
            continue
        if row_k + "|" + sev not in per_row:
            per_row.add(row_k + "|" + sev)
            ev.push(sev, kind if sev == "minor" else "cell errors in row", row_k, "")
    for col_sev, n in col_count.items():
        if n >= 3:
            col, sev = col_sev.split("|")[:2]
            ev.push(sev, "systematic column error", col, n)
    for gi in sorted(pos_matched):
        ev.push("minor", "column headers differ", gi, "")
    seen_case: set[str] = set()
    for i, j, *_ in M:
        raw_a, raw_b = G[i].raw_k, P[j].raw_k
        ja, jb = ",".join(raw_a), ",".join(raw_b)
        if raw_a and raw_b and ja != jb and ja.lower() == jb.lower() and ja not in seen_case:
            seen_case.add(ja)
            ev.push("minor", "label case changed", " / ".join(raw_a), " / ".join(raw_b))

    # output cells nobody asked for: inserted column (major once), filled empty slots (minor), repeated/invented rows
    g_vals = {nx_norm(x.v) for x in G} | {nx_norm(s.text) for t in gts for r in t[0].g for s in r}
    # an unmatched output cell under the key of an unmatched reference cell is that cell's wrong value, already counted
    g_open = {_key(x.k) for i, x in enumerate(G) if i not in gm}
    extra: dict[int, list[Triple]] = {}
    for j, y in enumerate(P):
        if j not in p_matched and _key(_sorted_pair(*pc[j])) not in g_open:
            extra.setdefault(y.pos, []).append(y)
    cc: dict[str, int] = {}
    tot_c: dict[str, int] = {}
    for j, y in enumerate(P):
        c_ = f"{y.ti}:{y.cj}"
        tot_c[c_] = tot_c.get(c_, 0) + 1
        if j not in p_matched:
            cc[c_] = cc.get(c_, 0) + 1
    cols = [c_ for c_, n in cc.items() if n >= 3 and n >= 0.5 * tot_c[c_]]
    for c_ in cols:
        ev.push("major", "extra column", c_, cc[c_])
    col_set = set(cols)
    for pos_k, xs in list(extra.items()):
        rest = [y for y in xs if f"{y.ti}:{y.cj}" not in col_set]
        if rest:
            extra[pos_k] = rest
        else:
            del extra[pos_k]
    for xs in extra.values():
        known = [
            y
            for y in xs
            if all(canon(lab) in g_labels for lab in y.k)
            and _key(_sorted_pair(canon(y.k[0]), canon(y.k[1]))) not in g_keys
        ]
        if known:
            ev.push("minor", "cell filled where reference is empty", "", " | ".join(y.v for y in known))
        known_ids = {id(y) for y in known}
        xs = [y for y in xs if id(y) not in known_ids]
        if not xs:
            continue
        invented = any(nx_norm(y.v) not in g_vals for y in xs)
        ev.push(
            "critical" if invented else "major",
            "invented row" if invented else "repeated row",
            "",
            " | ".join(y.v for y in xs),
        )

    # structure: row order, column order, transposition, header rows, title, empty rows/cols, split or merged tables
    def order(sel: Callable[[Triple], int], pos_of: Callable[[Triple], int], rows: bool) -> tuple[int, int, int]:
        land: dict[int, list[int]] = {}
        for i, j, *_ in M:
            if (i, j) not in by_values:
                land.setdefault(sel(G[i]), []).append(pos_of(P[j]))
        keys = sorted(land)
        seq = [sorted(land[k])[len(land[k]) >> 1] for k in keys]
        ok = _lis_len(seq) if seq else 0
        return len(seq), ok, min(len(seq) - ok, _moved_count(keys, seq, rows))

    ro_n, ro_ok, ro_ev = order(lambda x: x.gi * 10_000 + x.ri, lambda y: y.pos, True)
    co_n, co_ok, co_ev = order(lambda x: x.gi * 10_000 + x.cj, lambda y: y.ti * 10_000 + y.cj, False)
    for _ in range(ro_ev):
        ev.push("major", "row out of order")
    for _ in range(co_ev):
        ev.push("major", "column out of order")
    used_t = sorted({P[m[1]].ti for m in M})
    for ti in used_t:
        pt = PT[ti].tr
        if pt.transposed:
            ev.push("critical", "table transposed", ti, "")
        gis = sorted({G[m[0]].gi for m in M if P[m[1]].ti == ti})
        for _ in range(1, len(gis)):
            ev.push("major", "tables merged", ti, len(gis))
        if pt.borrowed:  # a continuation fragment has no header of its own; alone, the header is missing
            if len(gis) == 1 and len({P[m[1]].ti for m in M if G[m[0]].gi == gis[0]}) == 1:
                ev.push("critical", "header rows missing", gts[gis[0]][2].H, 0)
        elif len(gis) == 1 and pt.H != gts[gis[0]][2].H:
            ev.push("major", "header rows differ", gts[gis[0]][2].H, pt.H)
        elif (
            len(gis) == 1
            and not pt.transposed
            and gts[gis[0]][2].const_head
            and not pt.const_head
            and gts[gis[0]][2].const_head in nx_norm(pt.prefix)
        ):  # the spanning header level written into every column header of one row
            ev.push("major", "header rows joined", gts[gis[0]][2].const_head, pt.prefix)
        elif len(gis) == 1 and not pt.transposed:  # wording over the row labels
            ga = re.sub(r":\Z", "", gts[gis[0]][2].label_head)
            ob = re.sub(r":\Z", "", pt.label_head)
            if ga != ob:
                mild = not ga or not ob or key_sim(ga, ob) > 0 or ratio(ga, ob) >= 0.8
                ev.push("minor" if mild else "major", "row-label header differs", ga, ob)
        g_meta = gts[G[next(m for m in M if P[m[1]].ti == ti)[0]].gi]
        extra_empty = max(0, PT[ti].tg.empty_rows - g_meta[0].empty_rows) + max(
            0, PT[ti].tg.empty_cols - g_meta[0].empty_cols
        )
        if extra_empty:
            ev.push("minor", "empty rows or columns added", "", extra_empty)
        g_title = g_meta[2].title
        if (
            g_title
            and nx_norm(pt.title) != nx_norm(g_title)
            and nx_norm(re.sub(rf"\A#+{_S}*|\*\*", "", PT[ti].tg.before)) != nx_norm(g_title)
        ):
            ev.push("major", "title missing", g_title, "")
    for gi in range(len(gts)):
        n_parts = len({P[m[1]].ti for m in M if G[m[0]].gi == gi})
        for _ in range(1, n_parts):
            ev.push("major", "table split", gi, n_parts)
    if not allow_split:
        for ti, n, kind in split_info:
            ev.push("major", kind, ti, n)

    # severity cost: the first event of a class costs its full weight, later ones min(1, K / reference cells) of it
    def cost(c_: str) -> float:
        n = getattr(ev, c_)
        return WEIGHTS[c_] * (1 + (n - 1) * scale) if n else 0.0

    floor = max(0.0, min(1.0, 1 - cost("critical") - cost("major") - min(MINOR_CAP, cost("minor"))))
    row_order = ro_ok / ro_n if ro_n else 1.0
    prop = max(0.0, f1 - ORDER_PENALTY * (1 - row_order))
    return result(min(prop, floor), prop, f1, row_order, floor, ref_cells)


# ---------------------------------------------------------------------------
# Evaluator entry point
# ---------------------------------------------------------------------------


def has_reference_tables(expected_markdown: str | None) -> bool:
    return any(clean_grid(t).g for t in parse_tables(expected_markdown))


def table_next_metrics(
    expected_markdown: str | None,
    output_markdown: str | None,
    *,
    max_top_title_rows: int | None = 1,
    allow_splitting_ambiguous_merged_tables: bool = False,
    trm_unsupported: bool = False,
    allow_merged_sub_rows: bool = False,
) -> list[MetricValue]:
    """``table_next`` headline plus its content and row-order diagnostics; empty when the reference has no table."""
    if not has_reference_tables(expected_markdown):
        return []
    res = score_table_next(
        expected_markdown,
        output_markdown,
        max_top_title_rows=max_top_title_rows,
        allow_splitting_ambiguous_merged_tables=allow_splitting_ambiguous_merged_tables,
        trm_unsupported=trm_unsupported,
        allow_merged_sub_rows=allow_merged_sub_rows,
    )
    meta = res.metadata()
    meta["flags"] = {
        "max_top_title_rows": max_top_title_rows,
        "allow_splitting_ambiguous_merged_tables": bool(allow_splitting_ambiguous_merged_tables),
        "trm_unsupported": bool(trm_unsupported),
        "allow_merged_sub_rows": bool(allow_merged_sub_rows),
    }
    return [
        MetricValue(metric_name="table_next", value=res.score, metadata=meta),
        MetricValue(metric_name="table_next_content", value=res.content),
        MetricValue(metric_name="table_next_row_order", value=res.row_order),
    ]
