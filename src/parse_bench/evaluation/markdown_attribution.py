"""Output-centric Markdown attribution (scorer version 2). Never calls the legacy layout scorer.

The final Markdown defines the inventory. Content-only reference correspondence
is resolved before comparing geometry. Unboxed and unresolved elements never
become zero-quality attributions; they remain visible in coverage and details.
The one exception is the box budget, always on: a resolved element claimed
with more distinct boxes than its reference's ``expected_bbox_count`` (default
1) scores 0 (status ``multi_box_penalized``).

The headline is the square-root-area-weighted quality
(``md_attribution_v1_sqrt_area_quality``), reported as ``micro_`` over the
dataset. Like the count-weighted quality it scores only what was assessed: a
reference nothing was assessed against is left out, not scored 0. Unlike it,
each assessed reference weighs sqrt(its page fraction): a paragraph at 40% of
the page weighs about 45 times a page number at 0.02%, not 2000 times (area) or
once (count). Recall is reported beside it, not mixed in:
``md_attribution_v1_sqrt_area_coverage`` is the share of the scorable sqrt-area
weight that was assessed.

Secondary views:

* the count-weighted quality (``md_attribution_v1_quality``): the mean IoU over
  every assessed generated element, each element counting once;
* the area-weighted quality (``md_attribution_v1_area_quality``): every scorable
  source reference weighted by the fraction of the page it covers, with a
  reference nothing was assessed against counted as 0. It answers "what share
  of the page's annotated area is attributed correctly".

Every rule is scored by scorer version 2 (the version 1 code path was removed;
a rule that still says ``version: 1``, or no version, is scored the same way).
Metric names keep the historical ``md_attribution_v1_`` prefix for continuity,
because dashboards and saved reports read them; the metadata field
``scorer_version`` says which scorer produced a value. Version 2 adds linking
passes after the per-element text match (reading-order anchors for duplicate texts and
pictures, many-to-one groups, relaxed links, a fuzzy text fallback),
alternative reference texts, a row-piece exemption from the box budget, IoU
after a 1 pt dilation, chart cells by role, page furniture absent from the
Markdown out of coverage, and diagnostics outside the headline (missed weight
by cause, kind agreement, reading-order tau, furniture through items, the
weight fuzzy links carry). docs/markdown_attribution_v1.md, "Scorer version 2".
Run-level intervals and comparisons are in markdown_attribution_report.py.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any

import numpy as np
from markdown_it import MarkdownIt
from rapidfuzz.fuzz import partial_ratio, ratio, token_set_ratio

from parse_bench.schemas.evaluation import MetricValue
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.test_cases.markdown_attribution_schema import (
    AttributionReference,
    AttributionRegion,
    MarkdownAttributionRule,
)

PREFIX = "md_attribution_v1_"
_MD = MarkdownIt("commonmark", {"html": True}).enable("table")
_PROTECTED = {"not", "no", "never", "without", "neither", "nor"}


# ---------------------------------------------------------------------------
# LaTeX. Two transcriptions of one formula often differ only in idiom: the
# order of a sub- and a superscript, \left( versus (, \quad versus \,, \text{},
# \geqslant versus \ge, an aligned block versus one $$ line per row. normalize()
# already drops the punctuation ($ & \\ { } ^ _), but the command names it keeps
# ("left", "quad", "mathrm", "geqslant") and the script order (C^5_{1-n} reads
# 5,1 against 1,5 in C_{1-n}^5) fail the similarity and the digit gate. Inside
# math only, the rewrites below map each such pair to one form. They drop
# presentation, never a symbol, digit or operator of the formula itself.
# ---------------------------------------------------------------------------

_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$|\\\[.+?\\\]|\\\(.+?\\\)", re.DOTALL)
# One script argument: a braced group (one nesting level), a command, or a character.
_SCRIPT = r"(?:\{(?:[^{}]|\{[^{}]*\})*\}|\\[A-Za-z]+|[^\s{}\\])"
_SUP_SUB = re.compile(rf"(\^{_SCRIPT})\s*(_{_SCRIPT})")
# Presentation-only commands: sizing, spacing, font and text wrappers, and the
# environments that lay out rows. Their arguments are kept.
_LATEX_DROP = re.compile(
    r"\\(?:[,;:!> ]|(?:left|right|[bB]igg?[lrm]?|quad|qquad|enspace|thinspace|(?:h|v)space\*?"
    r"|mathrm|mathit|mathbf|mathsf|mathtt|mathcal|mathbb|mathfrak|mathscr|boldsymbol|bm|operatorname\*?"
    r"|text(?:rm|bf|it|sf|tt|normal|up)?|displaystyle|textstyle|limits|nolimits"
    r"|(?:begin|end)\{(?:aligned|align\*?|gathered|gather\*?|split|equation\*?|multline\*?|array)\})(?![A-Za-z]))"
)
# Synonyms: one command for the same glyph.
_LATEX_SYNONYMS = {
    "geqslant": "ge",
    "geq": "ge",
    "leqslant": "le",
    "leq": "le",
    "neq": "ne",
    "Box": "square",
    "to": "rightarrow",
    "lbrace": "{",
    "rbrace": "}",
    "vert": "|",
    "lvert": "|",
    "rvert": "|",
    "dfrac": "frac",
    "tfrac": "frac",
    "ldots": "dots",
    "cdots": "dots",
}
_LATEX_COMMAND = re.compile(r"\\([A-Za-z]+)")


def _canonical_math(formula: str) -> str:
    """One LaTeX formula in canonical form (see the section comment)."""
    formula = re.sub(r"\\tag\*?\{([^{}]*)\}", r"(\1)", formula)
    formula = _LATEX_DROP.sub(" ", formula)
    formula = _LATEX_COMMAND.sub(lambda m: "\\" + _LATEX_SYNONYMS[m[1]] if m[1] in _LATEX_SYNONYMS else m[0], formula)
    # Subscript before superscript: x^a_b -> x_b^a.
    return _SUP_SUB.sub(r"\2\1", formula)


def canonical_latex(text: str) -> str:
    """``text`` with every $...$, $$...$$, \\(...\\) and \\[...\\] formula canonicalized."""
    if "\\" not in text and "^" not in text:
        return text
    return _MATH.sub(lambda m: _canonical_math(m[0]), text)


# Picture markers. Engines write an image's alt text with a type prefix
# ("![photo: The Wedding Feast](...)", "![icon: signature](...)"); the ground
# truth writes pictures as bracket markers ("[seal: ROYAL DENTAL COLLEGE]",
# "[logo: X]", a bare "[signature]"; annotation guidelines, Logos and Seals).
# Both read as their words: the prefix is dropped, a marker becomes its text, or
# its type word when it carries none ("[signature]" = "signature").
#
# Read that way, a seal's words can equal a heading's and a logo's words a
# seal's. Two guards keep that from creating a wrong match (score_markdown_
# attribution): a marker reference is matched only by a picture-side element
# (an image or a marker) and counts duplicates apart from ordinary text, and a
# logo, seal, stamp or signature does not match a different one of those.
_MARK_TYPES = r"logo|seal|stamp|signature|emblem|watermark|qr code|barcode"
_PICTURE_TYPES = (
    rf"{_MARK_TYPES}|photo|photograph|icon|drawing|chart|graph|diagram|illustration|image|picture|figure|map"
    r"|screenshot"
)
_ALT_PREFIX = re.compile(rf"!\[\s*(?:{_PICTURE_TYPES})\s*:\s*", re.IGNORECASE)
_MARKER = re.compile(rf"\[\s*({_PICTURE_TYPES})\s*(?::\s*([^\]]*))?\](?!\()", re.IGNORECASE)
# A whole block that is one image or one marker: the image's alt-text type, or the marker's type.
_WHOLE_IMAGE = re.compile(rf"\s*!\[\s*(?:({_PICTURE_TYPES})\s*:)?[^\]]*\]\([^)]*\)\s*", re.IGNORECASE)
_WHOLE_MARKER = re.compile(rf"\s*\[\s*({_PICTURE_TYPES})\s*(?::[^\]]*)?\]\s*", re.IGNORECASE)


def _read_picture_markers(text: str) -> str:
    if "[" not in text:
        return text
    text = _ALT_PREFIX.sub("![", text)
    return _MARKER.sub(lambda m: m[2] if m[2] and m[2].strip() else m[1], text)


@dataclass(frozen=True)
class _PictureSide:
    """Whether a block is one image or one marker, and its mark type if it names one."""

    is_picture: bool
    is_marker: bool
    mark: str | None


def _picture_side(text: str) -> _PictureSide:
    image = _WHOLE_IMAGE.fullmatch(text)
    marker = _WHOLE_MARKER.fullmatch(text)
    match = image or marker
    kind = match[1] if match else None
    mark = kind.lower() if kind and re.fullmatch(_MARK_TYPES, kind, re.IGNORECASE) else None
    return _PictureSide(match is not None, marker is not None, mark)


def _is_marker_reference(ref: AttributionReference) -> bool:
    return ref.kind == "Picture" and _picture_side(ref.text).is_marker


# A line-break hyphen whose line break became a space ("nei- ther", "Offi- cer"):
# a letter, "- ", then a lowercase letter. Joined before the gates, or the
# protected word "neither" goes missing. A suspended compound ("pre- and
# post-war", "short- or long-term") is followed by a conjunction and kept.
_HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])- +(?!(?:and|or|nor|to|und|oder|bzw|et|ou|y|o|e)\b)(?=[a-zß-ÿ])")


def normalize(text: str) -> str:
    """Nicode-aware, ordered visible text; no bag-of-words or math deletion."""
    text = canonical_latex(text)
    text = _read_picture_markers(text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = unescape(re.sub(r"<[^>]*>", " ", text))
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = _HYPHEN_BREAK.sub("", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


# A leading list ordinal: "1 the requesting entity ..." after normalize.
_ORDINAL = re.compile(r"\d{1,2} (?=[^\W\d_])")


def _same_digits(a: str, b: str) -> bool:
    """The numbers of two normalized texts agree, in order.

    One side may carry an extra leading list ordinal of one or two digits
    followed by a word: markdown_elements drops an ordered list's marker, so a
    reference that keeps the printed "1." reads "1 the requesting entity" and
    its element "the requesting entity". Any other difference in the numbers,
    including their order, fails.
    """
    da, db = re.findall(r"\d+", a), re.findall(r"\d+", b)
    if da == db:
        return True
    if len(da) == len(db) + 1 and _ORDINAL.match(a):
        return da[1:] == db
    if len(db) == len(da) + 1 and _ORDINAL.match(b):
        return db[1:] == da
    return False


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # Conservative source identity, not a semantic-equivalence oracle.
    if not _same_digits(a, b):
        return 0.0
    if Counter(t for t in a.split() if t in _PROTECTED) != Counter(t for t in b.split() if t in _PROTECTED):
        return 0.0
    if min(len(a), len(b)) < 20:
        return float(a == b)
    return ratio(a, b) / 100


# ---------------------------------------------------------------------------
# Table cells. A table is compared as a multiset of normalized cells, not as one
# string: whole-text similarity needs the same digit order, so it fails on cell
# order, on a colspan that one side repeats per column, and on small per-cell
# transcription differences, even when every cell is present.
# ---------------------------------------------------------------------------

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"), "-")
TABLE_CELL_RATIO = 0.8  # per-cell fuzzy floor (0-1), after exact matches are taken
PAIRING_TABLE_F1 = 0.9  # item <-> final-MD table pairing: same engine output, so near-identical
# Chart references (Picture references holding a chart table) are scored by cell
# role instead (chart_cell_f1).
# The scorer version every rule is scored with (see the module docstring).
SCORER_VERSION = 2


def _cell_key(raw: str) -> str:
    """Normalized cell text: formatting stripped, superscripts split off, dashes unified.

    A cell that normalizes to nothing but prints a symbol ("-", "—", "/") keeps the
    symbol, so placeholder cells still count one-to-one. Empty cells return "".
    """
    raw = re.sub(r"<sup[^>]*>(.*?)</sup>", r" \1 ", raw, flags=re.DOTALL | re.IGNORECASE)
    key = normalize(raw)
    if key:
        return key
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", raw)).translate(_DASHES).split())


class _HTMLCells(HTMLParser):
    """Rows of cell strings from HTML tables. A colspan is written once per spanned
    column (then collapsed back by _collapse, like a pipe table that repeats it);
    a rowspan is written once, in its first row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cell: list[str] | None = None
        self.span = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in {"td", "th"}:
            if not self.rows:
                self.rows.append([])
            self.cell = []
            try:
                self.span = max(1, min(50, int(dict(attrs).get("colspan") or 1)))
            except ValueError:
                self.span = 1
        elif self.cell is not None:
            self.cell.append(f"<{tag}>" if tag == "sup" else " ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None:
            self.rows[-1].extend(["".join(self.cell)] * self.span)
            self.cell = None
        elif tag == "sup" and self.cell is not None:
            self.cell.append("</sup>")

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)


def _collapse(rows: list[list[str]]) -> Counter[str]:
    """Cell multiset. Runs of identical adjacent cells in a row are one cell (a
    colspan expanded into repeats); empty cells are dropped. Vertical repeats are
    kept: a column of genuine "0" values must count once per row."""
    cells: Counter[str] = Counter()
    for row in rows:
        previous = None
        for raw in row:
            key = _cell_key(raw)
            if key and key != previous:
                cells[key] += 1
            previous = key
    return cells


def _table_rows(text: str) -> list[list[str]]:
    """Raw cell strings, row by row, of the HTML or pipe tables in ``text``."""
    if "<table" in text.lower():
        parser = _HTMLCells()
        parser.feed(text)
        return parser.rows
    rows: list[list[str]] = []
    row: list[str] | None = None
    for token in _MD.parse(text):
        if token.type == "tr_open":
            row = []
            rows.append(row)
        elif token.type == "inline" and row is not None:
            row.append(token.content)
        elif token.type == "tr_close":
            row = None
    return rows


def table_cells(text: str) -> Counter[str] | None:
    """Cells of the HTML or pipe tables in ``text``; None when it holds no table."""
    cells = _collapse(_table_rows(text))
    return cells if sum(cells.values()) >= 2 else None


# ---------------------------------------------------------------------------
# Charts. A chart reference (a Picture whose text is a table)
# holds two kinds of cell. Label cells, the header row and the first column,
# are printed words and axis ticks: they match exactly, like any table cell
# (version 1 gave every numeric cell, labels included, a flat 20% tolerance, so
# a pH of 10 matched 12). Value cells, the body, were read off the marks: a
# reading error is a share of the axis, not of the value, so a numeric value
# cell b matches a predicted a when
#     |a - b| <= max(CHART_VALUE_SHARE * |b|, CHART_RANGE_SHARE * (max V - min V))
# with V the numeric value cells of the reference. The roles come from the
# reference alone, so a prediction that transposes the chart (labels in the
# first row instead of the first column) meets the same label set.
#
# Optional mark: a reference may prefix estimated value cells with "≈". Once a
# chart uses the mark, only marked cells get the tolerance, and unmarked
# (printed) values must match exactly.
# ---------------------------------------------------------------------------

CHART_VALUE_SHARE = 0.02
CHART_RANGE_SHARE = 0.05
ESTIMATE_MARK = "\u2248"


@dataclass(frozen=True)
class _ChartCells:
    """A chart reference's cells: every cell (for exact matching), the label
    cells, and each value cell with whether it may use the range tolerance."""

    cells: Counter[str]
    labels: Counter[str]
    values: Counter[tuple[str, bool]]
    spread: float


def chart_cells(text: str) -> _ChartCells | None:
    """Cells of a chart reference with their roles; None when it holds no table."""
    rows = _table_rows(text)
    marked = any(raw.strip().startswith(ESTIMATE_MARK) for row in rows for raw in row)
    labels: Counter[str] = Counter()
    values: Counter[tuple[str, bool]] = Counter()
    for r, row in enumerate(rows):
        previous = None
        for c, raw in enumerate(row):
            estimated = raw.strip().startswith(ESTIMATE_MARK)
            key = _cell_key(raw.replace(ESTIMATE_MARK, "", 1) if estimated else raw)
            if key and key != previous:
                if r == 0 or c == 0:
                    labels[key] += 1
                else:
                    values[(key, estimated or not marked)] += 1
            previous = key
    cells = labels + Counter({key: n for (key, _), n in values.items()})
    if sum(cells.values()) < 2:
        return None
    numbers = [v for (key, _), n in values.items() if (v := _number(key)) is not None for _ in range(n)]
    return _ChartCells(cells, labels, values, max(numbers) - min(numbers) if numbers else 0.0)


def chart_cell_f1(pred: Counter[str], chart: _ChartCells) -> float:
    """cell_f1 against a chart reference, labels exact and values in range (see above)."""
    total = sum(pred.values()) + sum(chart.cells.values())
    if not total:
        return 0.0
    matched = float(sum((pred & chart.cells).values()))
    left_p = [k for k, n in (pred - chart.cells).items() for _ in range(n)]
    # Leftover reference cells with their tolerance: labels first leave the
    # values of a key both a label and a value to the tolerant pass.
    left_g: list[tuple[str, bool]] = []
    remaining = chart.cells - pred
    for key, n in remaining.items():
        as_value = [tolerant for (k, tolerant), m in chart.values.items() if k == key for _ in range(m)]
        taken_as_label = max(0, n - len(as_value))
        left_g.extend((key, False) for _ in range(taken_as_label))
        left_g.extend((key, tolerant) for tolerant in sorted(as_value, reverse=True)[: n - taken_as_label])
    if left_p and left_g:
        pairs = []
        for i, a in enumerate(left_p):
            ca = a.replace(" ", "")
            digits = re.findall(r"\d+", ca)
            value_a = _number(a)
            for j, (b, tolerant) in enumerate(left_g):
                if tolerant and value_a is not None and (value_b := _number(b)) is not None:
                    limit = max(CHART_VALUE_SHARE * abs(value_b), CHART_RANGE_SHARE * chart.spread)
                    if abs(value_a - value_b) <= limit:
                        pairs.append((1.0, i, j))
                    continue
                cb = b.replace(" ", "")
                if re.findall(r"\d+", cb) != digits:
                    continue
                score = 1.0 if ca == cb else ratio(ca, cb) / 100
                if score >= TABLE_CELL_RATIO:
                    pairs.append((score, i, j))
        used_p: set[int] = set()
        used_g: set[int] = set()
        for score, i, j in sorted(pairs, reverse=True):
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
                matched += score
    return 2 * matched / total


def _number(key: str) -> float | None:
    """The value of a normalized numeric cell key, else None.

    Keys come from ``normalize``, which drops punctuation: "1,000" is "1 000" and
    "12.5" (or "12,5") is "12 5". Trailing groups of exactly three digits read as
    thousands, and a last group of any other length reads as the decimal part. Sign
    and percent marks are gone by then, so only magnitudes are compared.
    """
    groups = key.split()
    if not groups or not all(g.isascii() and g.isdigit() for g in groups):
        return None
    if len(groups) > 1 and len(groups[-1]) != 3:
        return float("".join(groups[:-1]) + "." + groups[-1])
    return float("".join(groups))


def cell_f1(pred: Counter[str], gt: Counter[str]) -> float:
    """Order-independent cell-multiset F1 in [0, 1].

    Exact matches first, count-limited, so a value repeated three times on one
    side and twice on the other matches twice. Leftover cells then pair greedily
    by edit ratio >= TABLE_CELL_RATIO, compared without spaces and with the same
    digits, each earning its ratio. Chart references use chart_cell_f1.
    F1 = 2 x matched / (pred cells + GT cells).
    """
    total = sum(pred.values()) + sum(gt.values())
    if not total:
        return 0.0
    matched = float(sum((pred & gt).values()))
    left_p = [k for k, n in (pred - gt).items() for _ in range(n)]
    left_g = [k for k, n in (gt - pred).items() for _ in range(n)]
    if left_p and left_g:
        pairs = []
        for i, a in enumerate(left_p):
            ca = a.replace(" ", "")
            digits = re.findall(r"\d+", ca)
            for j, b in enumerate(left_g):
                cb = b.replace(" ", "")
                if re.findall(r"\d+", cb) != digits:
                    continue
                score = 1.0 if ca == cb else ratio(ca, cb) / 100
                if score >= TABLE_CELL_RATIO:
                    pairs.append((score, i, j))
        used_p: set[int] = set()
        used_g: set[int] = set()
        for score, i, j in sorted(pairs, reverse=True):
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
                matched += score
    return 2 * matched / total


def _table_scorer(ref: AttributionReference) -> Callable[[Counter[str]], float] | None:
    """Cell F1 of a predicted table against ``ref``; None when ``ref`` holds no table.

    Chart references (a Picture whose text is a table) are scored by cell role
    (chart_cell_f1); ordinary tables by cell_f1, with the same numbers required.
    """
    if ref.kind == "Picture":
        chart = chart_cells(ref.text)
        return None if chart is None else (lambda cells: chart_cell_f1(cells, chart))
    gt = table_cells(ref.text)
    return None if gt is None else (lambda cells: cell_f1(cells, gt))


def _table_assignment(
    elements: list[tuple[int, Counter[str]]], rule: MarkdownAttributionRule
) -> dict[int, tuple[AttributionReference, float]]:
    """One-to-one table element -> table reference assignment by cell F1.

    Candidates are references and spans whose text parses to cells. Pairs at or
    above ``table_match_threshold`` are taken greedily by F1 (ties by document
    order), so two predicted tables never claim the same reference.
    """
    refs = [r for ref in sorted(rule.references, key=lambda r: r.order) for r in (ref, *ref.spans)]
    targets = [(ref, f1) for ref in refs if (f1 := _table_scorer(ref)) is not None]
    scored = sorted(
        (
            (score, -index, -t, ref)
            for index, cells in elements
            for t, (ref, f1) in enumerate(targets)
            if (score := f1(cells)) >= rule.table_match_threshold
        ),
        key=lambda v: v[:3],
        reverse=True,
    )
    result: dict[int, tuple[AttributionReference, float]] = {}
    taken: set[str] = set()
    for score, neg_index, _, ref in scored:
        if -neg_index not in result and ref.id not in taken:
            result[-neg_index] = (ref, score)
            taken.add(ref.id)
    return result


@dataclass
class Element:
    text: str
    kind: str
    start_line: int
    end_line: int


def markdown_elements(markdown: str) -> list[Element]:
    """Leaf paragraphs/headings/list paragraphs, whole tables, HTML blocks, code.

    Container nodes are not counted twice. Provider items do not decide the
    inventory. Raw HTML is one opaque block; its limitation is reported in docs.
    """
    lines = markdown.splitlines(keepends=True)
    result: list[Element] = []
    covered_until = -1
    tokens = _MD.parse(markdown)
    for token_index, token in enumerate(tokens):
        if not token.map or token.map[0] < covered_until:
            continue
        if token.type not in {"paragraph_open", "heading_open", "table_open", "html_block", "fence", "code_block"}:
            continue
        start, end = token.map
        text = "".join(lines[start:end]).strip()
        if not text or not re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip():
            continue
        kind = token.type.removesuffix("_open")
        if kind in {"paragraph", "heading"} and token_index + 1 < len(tokens):
            # The inline payload excludes list markers/heading delimiters while
            # retaining actual content; source line offsets still point at final MD.
            inline = tokens[token_index + 1]
            if inline.type == "inline":
                text = inline.content
        if re.match(r"\s*!\[", text):
            kind = "image"
        elif "<table" in text.lower():
            kind = "table"
        result.append(Element(text, kind, start, end))
        covered_until = end
    return result


def _area(regions: list[AttributionRegion]) -> float:
    """Exact rectangle union area, with each page a separate coordinate plane."""
    total = 0.0
    for page in {r.page for r in regions}:
        boxes = [r.bbox for r in regions if r.page == page]
        xs = sorted({x for b in boxes for x in (b[0], b[0] + b[2])})
        for left, right in zip(xs, xs[1:], strict=False):
            spans = sorted((b[1], b[1] + b[3]) for b in boxes if b[0] < right and b[0] + b[2] > left)
            top = bottom = 0.0
            height = 0.0
            for lo, hi in spans:
                if lo > bottom:
                    height += bottom - top
                    top, bottom = lo, hi
                else:
                    bottom = max(bottom, hi)
            total += (right - left) * (height + bottom - top)
    return total


def _clamped_area(regions: list[AttributionRegion]) -> float:
    """Nion area in page fractions after clipping every box to the [0, 1] page."""
    clipped = []
    for region in regions:
        x, y, w, h = region.bbox
        left, top = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        right, bottom = min(max(x + w, 0.0), 1.0), min(max(y + h, 0.0), 1.0)
        if right > left and bottom > top:
            clipped.append(AttributionRegion(page=region.page, bbox=[left, top, right - left, bottom - top]))
    return _area(clipped)


def geometry(pred: list[AttributionRegion], target: list[AttributionRegion]) -> tuple[float, float, float]:
    intersections = []
    for p in pred:
        for r in target:
            if p.page != r.page:
                continue
            x = max(p.bbox[0], r.bbox[0])
            y = max(p.bbox[1], r.bbox[1])
            w = min(p.bbox[0] + p.bbox[2], r.bbox[0] + r.bbox[2]) - x
            h = min(p.bbox[1] + p.bbox[3], r.bbox[1] + r.bbox[3]) - y
            if w > 0 and h > 0:
                intersections.append(AttributionRegion(page=p.page, bbox=[x, y, w, h]))
    inter, pa, ta = _area(intersections), _area(pred), _area(target)
    return (
        min(1.0, max(0.0, inter / (pa + ta - inter))),
        min(1.0, max(0.0, inter / ta)),
        min(1.0, max(0.0, inter / pa)),
    )


def _is_stack(regions: list[AttributionRegion]) -> bool:
    """Each box sits just below the previous one (gap -0.01 to 0.04 of the page
    height) and shares at least half the narrower width with it: one column."""
    for first, second in zip(regions, regions[1:], strict=False):
        a, b = first.bbox, second.bbox
        x_overlap = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        gap = b[1] - (a[1] + a[3])
        if x_overlap < 0.5 * min(a[2], b[2]) or not -0.01 <= gap <= 0.04:
            return False
    return True


# A row neighbour may overlap its left neighbour horizontally by this much (page
# fraction) and still count as beside it, not on top of it: refitted ink boxes
# of adjacent form labels and fields can touch.
ROW_X_OVERLAP = 0.01


def _row_order(regions: list[AttributionRegion]) -> list[int] | None:
    """Left-to-right order of side-by-side boxes on one row, else None.

    Taken left to right, every box shares at least half the smaller height with
    the one before it and starts where that one ends (within ROW_X_OVERLAP).
    This is a form row ("City: | State: | ZIP Code:"), a header or footer band
    written as one line, or a margin note beside the paragraph it annotates
    (the note's short height lies within the paragraph's).
    """
    order = sorted(range(len(regions)), key=lambda i: regions[i].bbox[0])
    for i, j in zip(order, order[1:], strict=False):
        a, b = regions[i].bbox, regions[j].bbox
        y_overlap = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        if y_overlap < 0.5 * min(a[3], b[3]) or b[0] < a[0] + a[2] - ROW_X_OVERLAP:
            return None
    return order


# Row pieces. An engine that boxes one line of a form or a
# header band as several side-by-side boxes has claimed one element, not
# several: the pieces sit on one baseline and do not overlap horizontally. Two
# boxes are row pieces when they share at least half the smaller height and
# overlap horizontally by at most a tenth of the narrower width. Stacked halves
# of a paragraph, a wrapper box plus a line box, or nested boxes are not row
# pieces, and still count one box each against the budget.
ROW_PIECE_MIN_Y_OVERLAP = 0.5
ROW_PIECE_MAX_X_OVERLAP = 0.1


def _row_pieces(a: AttributionRegion, b: AttributionRegion) -> bool:
    (ax, ay, aw, ah), (bx, by, bw, bh) = a.bbox, b.bbox
    y_overlap = min(ay + ah, by + bh) - max(ay, by)
    x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
    return (
        a.page == b.page
        and y_overlap >= ROW_PIECE_MIN_Y_OVERLAP * min(ah, bh)
        and x_overlap <= ROW_PIECE_MAX_X_OVERLAP * min(aw, bw)
    )


def _box_count(regions: list[AttributionRegion]) -> int:
    """Boxes counted against the budget, with each row of row pieces as one.

    Boxes are joined into connected groups by the row-piece test; a group counts
    as one box only if every pair in it passes the test (so a chain cannot join
    two boxes that overlap each other), otherwise each of its boxes counts.
    """
    n = len(regions)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _row_pieces(regions[i], regions[j]):
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    count = 0
    for members in groups.values():
        pairs = ((i, j) for i in members for j in members if i < j)
        count += 1 if all(_row_pieces(regions[i], regions[j]) for i, j in pairs) else len(members)
    return count


# IoU after dilating both boxes by DILATION_PT points on every
# side (clipped to the page). On a six-point line number one point of padding
# halves the raw IoU, so the raw score mostly measures the box convention (ink
# fitted or padded); a point of slack on both sides hardly moves a large box.
DILATION_PT = 1.0
# Page size in points when the output does not give one (US Letter).
DEFAULT_PAGE_PT = (612.0, 792.0)


def _dilate(regions: list[AttributionRegion], sizes: Mapping[int, tuple[float, float]]) -> list[AttributionRegion]:
    result = []
    for region in regions:
        width, height = sizes.get(region.page, DEFAULT_PAGE_PT)
        dx, dy = DILATION_PT / width, DILATION_PT / height
        x, y, w, h = region.bbox
        left, top = max(0.0, x - dx), max(0.0, y - dy)
        right, bottom = min(1.0, x + w + dx), min(1.0, y + h + dy)
        result.append(AttributionRegion.model_construct(page=region.page, bbox=[left, top, right - left, bottom - top]))
    return result


def _page_sizes(output: ParseOutput) -> dict[int, tuple[float, float]]:
    """Page width and height in points, by page number, where the output gives them."""
    return {
        page.page_number: (float(page.width), float(page.height))
        for page in output.layout_pages
        if page.width and page.height and page.width > 0 and page.height > 0
    }


@dataclass
class _Merge:
    """An admissible merge of GT-consecutive references.

    ``text`` is what one Markdown block holding all of them reads like: GT order
    for a stack, left-to-right order for a row (a margin note written before
    its paragraph). ``target`` is what the element's box is compared with, and
    ``budget`` how many distinct boxes it may be claimed with (the box budget).
    """

    text: str
    target: list[AttributionRegion]
    budget: int


def _merge(refs: list[AttributionReference]) -> _Merge | None:
    """Only ordinary single-box references on one page merge, as a stack or a row.

    A stack (lines of one column) is one block on the page: it is compared with
    its envelope, which spans the lines and the leading between them, and gets
    one box. A row is several blocks that the Markdown happens to write on one
    line: it is compared with the union of the references' own boxes, not their
    envelope (the envelope of "Name: ____ Date: ____" is mostly blank fill-in
    space that no reference covers), and may be claimed with one box per
    reference. One box over the whole row is still scored, and the union IoU
    charges the blank space it covers.
    """
    if len(refs) == 1:
        return _Merge(refs[0].text, refs[0].regions, refs[0].expected_bbox_count)
    if any(len(r.regions) != 1 or r.exception for r in refs):
        return None
    regions = [r.regions[0] for r in refs]
    if len({r.page for r in regions}) != 1:
        return None
    if _is_stack(regions):
        x = min(r.bbox[0] for r in regions)
        y = min(r.bbox[1] for r in regions)
        envelope = AttributionRegion(
            page=regions[0].page,
            bbox=[
                x,
                y,
                max(r.bbox[0] + r.bbox[2] for r in regions) - x,
                max(r.bbox[1] + r.bbox[3] for r in regions) - y,
            ],
        )
        return _Merge(" ".join(r.text for r in refs), [envelope], 1)
    order = _row_order(regions)
    if order is None:
        return None
    return _Merge(" ".join(refs[i].text for i in order), regions, len(refs))


def _candidates(rule: MarkdownAttributionRule) -> list[tuple[str, list[AttributionReference]]]:
    refs = sorted(rule.references, key=lambda r: r.order)
    result = [(normalize(ref.text), [ref]) for ref in refs if normalize(ref.text)]
    result.extend((normalize(span.text), [span]) for ref in refs for span in ref.spans if normalize(span.text))
    for size in range(2, rule.max_merge_elements + 1):
        for i in range(len(refs) - size + 1):
            group = refs[i : i + size]
            if all(normalize(r.text) for r in group) and (merge := _merge(group)):
                result.append((normalize(merge.text), group))
    return result


def _alt_candidates(rule: MarkdownAttributionRule) -> list[tuple[str, list[AttributionReference]]]:
    """One candidate per alternative text of a reference or span.

    Alternatives only extend what an element may match; they never make a
    reference a duplicate of another (_area_weighting counts primary texts).
    """
    refs = sorted(rule.references, key=lambda r: r.order)
    return [
        (content, [r])
        for ref in refs
        for r in (ref, *ref.spans)
        for alt in r.alt_texts
        if (content := normalize(alt)) and content != normalize(r.text)
    ]


def _family(refs: list[AttributionReference], rule: MarkdownAttributionRule) -> set[str]:
    """Reference ids of a candidate, with a span standing for its parent too."""
    parents = {span.id: ref.id for ref in rule.references for span in ref.spans}
    return {r.id for r in refs} | {parents[r.id] for r in refs if r.id in parents}


def _correspond(
    text: str, candidates: list[tuple[str, list[AttributionReference]]], rule: MarkdownAttributionRule
) -> tuple[list[AttributionReference] | None, str, float]:
    exact = [refs for content, refs in candidates if text and text == content]
    # The text and an alternative of one reference are one identity.
    exact = list({tuple(r.id for r in refs): refs for refs in exact}.values())
    if exact:
        # A unique full-content identity beats a fuzzy superset (e.g. the same
        # long paragraph plus a short neighboring heading). Repeated identities
        # remain ambiguous; the submitted box must not resolve them.
        if len(exact) == 1:
            return exact[0], "resolved", 1.0
        return None, "ambiguous_source", 1.0
    ranked = sorted(
        ((similarity(text, content), refs) for content, refs in candidates), key=lambda v: v[0], reverse=True
    )
    if not ranked or ranked[0][0] < rule.text_match_threshold:
        # Do not compare a split output to the enclosing whole-paragraph box.
        partial = bool(text) and any(text in content and text != content for content, _ in candidates)
        return None, "needs_span_geometry" if partial else "unresolved_content", 0.0
    best, refs = ranked[0]
    # A piece of a longer reference can still clear the threshold (a column
    # continuation matched to its annotated span, a label missing a short
    # trailing word). It resolves only if every candidate that contains it
    # shares a reference with the best match; a piece that also occurs inside
    # an unrelated reference (a repeated form label) is left to span geometry.
    family = _family(refs, rule)
    if any(text in content and text != content and not family & _family(other, rule) for content, other in candidates):
        return None, "needs_span_geometry", best
    # A runner-up within the margin makes the match ambiguous unless it is the
    # best candidate plus or minus neighbours: "paragraph" at 0.99 and
    # "paragraph + short heading" at 0.96 both name the paragraph, and the
    # better score says whether the heading is in the element.
    ids = {r.id for r in refs}
    for score, other in ranked[1:]:
        if best - score > rule.ambiguity_margin:
            break
        other_ids = {r.id for r in other}
        # Another text of the same reference (an alternative) is not a rival.
        if not (ids < other_ids or other_ids < ids or ids == other_ids):
            return None, "ambiguous_source", best
    return refs, "resolved", best


def _picture_filter(
    element: Element, candidates: list[tuple[str, list[AttributionReference]]]
) -> list[tuple[str, list[AttributionReference]]]:
    """Candidates an element may match on the picture side (see _read_picture_markers).

    An image or a marker matches only candidates holding a Picture reference,
    and not a marker of another mark type (a logo is not the seal that prints
    the same words). Any other element does not match a marker reference.
    """
    side = _picture_side(element.text)
    if element.kind != "image" and not side.is_picture:
        return [(c, refs) for c, refs in candidates if not any(_is_marker_reference(r) for r in refs)]
    result = []
    for content, refs in candidates:
        if not any(r.kind == "Picture" for r in refs):
            continue
        marks = {_picture_side(r.text).mark for r in refs if r.kind == "Picture"} - {None}
        if side.mark and marks and side.mark not in marks:
            continue
        result.append((content, refs))
    return result


def _item_regions(page: Any, item: Any) -> tuple[list[AttributionRegion], bool, bool]:
    """A provider item's boxes as page-normalized regions: (regions, attempted, invalid)."""
    segments = item.layout_segments or ([item.bbox] if item.bbox else [])
    regions: list[AttributionRegion] = []
    invalid = False
    for segment in segments:
        box = [segment.x, segment.y, segment.w, segment.h]
        # Same convention as normalized layout IR: page-normalized or pixels.
        if any(abs(v) > 1 for v in box):
            if not page.width or not page.height or page.width <= 0 or page.height <= 0:
                invalid = True
                continue
            box = [box[0] / page.width, box[1] / page.height, box[2] / page.width, box[3] / page.height]
        try:
            if segment.r is not None and not math.isclose(segment.r % 360, 0):
                raise ValueError("rotated boxes need a polygon target")
            region = AttributionRegion(page=page.page_number, bbox=box)
            if region not in regions:
                regions.append(region)
        except ValueError:
            invalid = True
    return regions, bool(segments), invalid


def _claims(output: ParseOutput) -> list[dict[str, Any]]:
    claims = []
    for page in output.layout_pages:
        for item in page.items:
            regions, attempted, invalid = _item_regions(page, item)
            raw = item.md or item.html or item.value
            pieces = markdown_elements(raw)
            # Multiple final blocks under one provider item are still distinct MD
            # elements. No per-block segmentation means the item's claim is broad.
            for piece in pieces:
                claims.append(
                    {
                        "text": normalize(piece.text),
                        "raw_text": piece.text,
                        "regions": regions,
                        "attempted": attempted,
                        "invalid": invalid,
                        "used": False,
                        "kind": piece.kind,
                    }
                )
    return claims


# Longest reference text kept in the per-example output. The report shows a
# snippet next to each box; the full text stays in the dataset's test.json.
DISPLAY_TEXT_CHARS = 240


def _snippet(text: str) -> str:
    return text if len(text) <= DISPLAY_TEXT_CHARS else text[: DISPLAY_TEXT_CHARS - 1] + "\u2026"


def _reference_display(ref: AttributionReference) -> dict[str, Any]:
    """What the report needs to draw a reference: its text snippet, regions and box budget."""
    return {
        "text": _snippet(ref.text),
        "order": ref.order,
        "expected_bbox_count": ref.expected_bbox_count,
        "regions": [region.model_dump() for region in ref.regions],
    }


# ---------------------------------------------------------------------------
# Why a reference was missed, and page furniture.
#
# A missed reference is "absent" when its content is not in the Markdown at
# all, and "unlinked" when it is there but no element resolved to it. Absent
# weight measures the transcription; unlinked weight is the scorer's own error
# budget (a scorer change that raises it is a scorer regression).
#
# Page furniture (running heads and footers, folios) that the Markdown leaves
# out is out of both the headline and coverage by default ("not_requested"),
# whatever its optional flag: the product drops the bands by design. Furniture
# the Markdown does write is scored as usual: assessed when an element links
# it, else missed (unlinked), or optional when it is flagged optional.
# Separately, the layout items an engine typed as page header or footer are
# matched to the furniture references by text, which measures
# whether the engine boxed the furniture it kept as items (furniture_*).
# ---------------------------------------------------------------------------

FURNITURE_KINDS = frozenset({"Page-header", "Page-footer", "Page-number"})
_FURNITURE_ITEM_TYPES = frozenset({"pageheader", "pagefooter", "pagenumber", "header", "footer"})
ABSENT_PARTIAL_RATIO = 90  # a text of ABSENT_MIN_CHARS+ is present when it nearly occurs in the Markdown
ABSENT_MIN_CHARS = 8
ABSENT_TABLE_CELL_RECALL = 0.5  # a table is present when half its cells occur in the Markdown


@dataclass
class _MarkdownText:
    """The whole Markdown, normalized, for the absent / unlinked test."""

    text: str
    cells: Counter[str]
    images: int  # image elements no reference was assessed against

    def contains(self, ref: AttributionReference) -> bool:
        if ref.kind == "Picture" and not normalize(ref.text):
            # A picture has no words, so this is a page-level test: present
            # (unlinked) when any image element on the page went unscored,
            # even one that belongs to a different picture.
            return self.images > 0
        for raw in (ref.text, *ref.alt_texts):
            cells = table_cells(raw)
            if cells is not None:
                present = sum(n for key, n in cells.items() if key in self.cells or f" {key} " in f" {self.text} ")
                if present >= ABSENT_TABLE_CELL_RECALL * sum(cells.values()):
                    return True
                continue
            text = normalize(raw)
            if not text:
                continue
            if f" {text} " in f" {self.text} ":
                return True
            if len(text) >= ABSENT_MIN_CHARS and partial_ratio(text, self.text) >= ABSENT_PARTIAL_RATIO:
                return True
        return False


def _furniture_score(
    output: ParseOutput, rule: MarkdownAttributionRule, sizes: Mapping[int, tuple[float, float]]
) -> dict[str, Any]:
    """Furniture references against the layout items typed as page header/footer.

    Each furniture reference with text is matched one-to-one to the item whose
    text is most similar (at least ``text_match_threshold``, exact under 20
    characters), and scored like an element: dilated IoU, 0 over its box budget.
    Weighted by sqrt(area) like the headline: quality over matched references,
    coverage = matched / all furniture references with text.
    """
    refs = [r for r in rule.references if r.kind in FURNITURE_KINDS and normalize(r.text)]
    items = []
    for page in output.layout_pages:
        for item in page.items:
            kind = re.sub(r"[-_ ]", "", (item.type or "").lower())
            if kind in _FURNITURE_ITEM_TYPES:
                regions, attempted, invalid = _item_regions(page, item)
                if attempted and not invalid and regions:
                    items.append((normalize(item.md or item.html or item.value or ""), regions))
    pairs = sorted(
        (
            (score, -i, -j)
            for i, ref in enumerate(refs)
            for j, (text, _) in enumerate(items)
            if (score := similarity(text, normalize(ref.text))) >= rule.text_match_threshold
        ),
        reverse=True,
    )
    matched: dict[int, int] = {}
    for _, i, j in pairs:
        if -i not in matched and -j not in matched.values():
            matched[-i] = -j
    score = total = scorable = 0.0
    rows = []
    for i, ref in enumerate(refs):
        weight = math.sqrt(_clamped_area(ref.regions))
        scorable += weight
        quality = None
        if i in matched:
            regions = items[matched[i]][1]
            quality = geometry(_dilate(regions, sizes), _dilate(ref.regions, sizes))[0]
            if _box_count(regions) > ref.expected_bbox_count:
                quality = 0.0
            score += quality * weight
            total += weight
        rows.append({"reference_id": ref.id, "quality": quality})
    return {
        "score": score,
        "total": total,
        "scorable": scorable,
        "quality": score / total if total > 0 else None,
        "coverage": total / scorable if scorable > 0 else None,
        "references": rows,
    }


def _area_weighting(
    rule: MarkdownAttributionRule,
    rows: list[dict[str, Any]],
    candidates: list[tuple[str, list[AttributionReference]]],
    markdown: _MarkdownText,
) -> dict[str, Any]:
    """Reference-side, area-weighted quality: sum(quality * area) / sum(area).

    Weight = union area of the reference's regions as a fraction of the page
    (clipped to [0, 1]; a multi-column paragraph's pieces are unioned, not
    summed). The quality of a reference is the best per-element quality among
    assessed elements resolved to it. A merged element passes its quality to
    every reference it spans. An annotated span credits its parent in
    proportion to its area: the parent gets the larger of its own best quality
    and sum(span quality x span area) / parent area (capped at 1), so a column
    paragraph output as two pieces, each matched to its span, earns what the
    two pieces earn on their own areas, and one piece alone earns only its
    share. Each reference ends in exactly one of five states, decided in this
    order:

    * ``unscorable``: nothing can ever link it: empty normalized text and not a
      ``Picture`` (for example an empty Table or Text box). Out of the weights.
      Duplicate texts and text-less pictures are not unscorable: the
      reading-order anchors can link them (_anchor_pass), so they end in one
      of the states below like any other reference.
    * ``assessed``: at least one element (or annotated span) was scored
      against it; weighted by its quality (``multi_box_penalized`` elements
      contribute their 0). An ``optional``, ``decorative`` or furniture
      reference that an element resolves to is assessed like any other.
    * ``not_requested``: nothing assessed, the reference is page furniture
      (Page-header, Page-footer, Page-number) and its content is absent from
      the Markdown. Out of the weights, and checked before the flags, so it
      overrides ``optional``: the product drops these bands by design.
    * ``optional``: nothing assessed and the reference is flagged ``optional``
      (page furniture the Markdown may leave out) or ``decorative`` (a picture
      that is background art, an ornament or a scan speck). Out of the
      weights. ``decorative`` has no state of its own: it gives ``optional``.
    * ``missed``: every other reference nothing was assessed against (not
      output, unboxed, unresolved text, invalid geometry, an unpaired
      picture or duplicate): quality 0 in the area-weighted view, and in the
      sqrt-area coverage denominator only. This includes furniture whose
      content is in the Markdown but that no element linked, unless it is
      flagged ``optional``.

    Each missed reference also gets a ``missed_reason``: ``absent``
    (its content is not in the Markdown) or ``unlinked`` (it is, but no element
    resolved to it); see _MarkdownText. A text-less picture has no words to
    look for, so the test is page-level: it is ``unlinked`` whenever any image
    element on the page was left unscored, and ``absent`` only when every
    image element was scored.
    """
    # Marker references count duplicates apart from ordinary text: only
    # picture-side elements can match them (_picture_filter).
    content_counts = Counter((content, any(_is_marker_reference(r) for r in refs)) for content, refs in candidates)
    # Every resolved row is assessed (the box budget scores instead of skipping),
    # so a reference id on a row always comes with a quality.
    best: dict[str, float] = {}
    for row in rows:
        for ref_id in row["reference_ids"]:
            if row["quality"] is not None:
                best[ref_id] = max(best.get(ref_id, 0.0), row["quality"])
    references: list[dict[str, Any]] = []
    areas = dict.fromkeys(("assessed", "missed", "unscorable", "optional", "not_requested"), 0.0)
    missed_sqrt = {"absent": 0.0, "unlinked": 0.0}
    # References the fuzzy fallback credits (a span stands for its
    # parent). The fallback only links references nothing else links, so these
    # are credited by fuzzy links alone.
    fuzzy_ids = {r for row in rows if row.get("match") == "fuzzy" for r in row["reference_ids"]}
    fuzzy_ids |= {ref.id for ref in rule.references for span in ref.spans if span.id in fuzzy_ids}
    sqrt_fuzzy = 0.0
    # sqrt_*: the same references and qualities weighted by sqrt(area), over
    # assessed references only (missed ones enter only the sqrt coverage).
    total = score = sqrt_total = sqrt_score = sqrt_scorable = 0.0
    for ref in sorted(rule.references, key=lambda r: r.order):
        area = _clamped_area(ref.regions)
        text = normalize(ref.text)
        quality: float | None = None
        credited = [best[ref.id]] if ref.id in best else []
        spans = [span for span in ref.spans if span.id in best]
        if spans and area > 0:
            credited.append(min(1.0, sum(best[s.id] * _clamped_area(s.regions) for s in spans) / area))
        reason = None
        duplicate = bool(text) and content_counts[(text, _is_marker_reference(ref))] > 1
        # Duplicate texts and text-less pictures (decorative ones included)
        # are linkable: the reading-order anchors can link them (_anchor_pass).
        # Only a text-less reference that is not a Picture is unscorable.
        linkable = duplicate or (not text and ref.kind == "Picture")
        if (not text or duplicate) and not linkable:
            status = "unscorable"
        elif credited:
            status, quality = "assessed", max(credited)
        else:
            reason = "unlinked" if markdown.contains(ref) else "absent"
            if reason == "absent" and ref.kind in FURNITURE_KINDS:
                status, reason = "not_requested", None
            elif ref.optional or ref.decorative:
                status, reason = "optional", None
            else:
                status, quality = "missed", 0.0
                missed_sqrt[reason] += math.sqrt(area)
        areas[status] += area
        if quality is not None:
            total += area
            score += quality * area
            sqrt_scorable += math.sqrt(area)
            if status == "assessed":
                sqrt_total += math.sqrt(area)
                sqrt_score += quality * math.sqrt(area)
                if ref.id in fuzzy_ids:
                    sqrt_fuzzy += math.sqrt(area)
        references.append(
            {
                "reference_id": ref.id,
                "kind": ref.kind,
                "area": area,
                "quality": quality,
                "status": status,
                # Display-only fields for the report's page preview; no metric reads them.
                **_reference_display(ref),
                "spans": [_reference_display(span) | {"reference_id": span.id} for span in ref.spans],
            }
        )
        if reason:
            references[-1]["missed_reason"] = reason
        if status == "assessed" and ref.id in fuzzy_ids:
            references[-1]["fuzzy"] = True
    result: dict[str, Any] = {
        "area_total": total,
        "area_score": score,
        # None when the page has no scorable reference area: excluded, not 0 or 1.
        "area_quality": score / total if total > 0 else None,
        "area_by_status": areas,
        "sqrt_area_total": sqrt_total,
        "sqrt_area_score": sqrt_score,
        "sqrt_area_scorable": sqrt_scorable,
        # None when nothing with area was assessed (e.g. every reference missed).
        "sqrt_area_quality": sqrt_score / sqrt_total if sqrt_total > 0 else None,
        # Positive exactly when area_total is, so the same pages are excluded.
        "sqrt_area_coverage": sqrt_total / sqrt_scorable if sqrt_scorable > 0 else None,
        "references": references,
    }
    # The headline's sums at weight exponents 0 (count) and 1 (area), for the
    # comparison check that the sign of a difference does not hang on the
    # frozen exponent 0.5 (markdown_attribution_report.compare).
    for p in (0, 1):
        assessed = [r for r in references if r["status"] == "assessed" and r["area"] > 0]
        result[f"p{p}_score"] = sum(r["quality"] * r["area"] ** p for r in assessed)
        result[f"p{p}_total"] = sum(r["area"] ** p for r in assessed)
    # Missed sqrt(area) weight by reason; with sqrt_area_total they sum to
    # sqrt_area_scorable.
    result["sqrt_area_missed_absent"] = missed_sqrt["absent"]
    # Assessed sqrt(area) weight credited by fuzzy links (fuzzy_weight_share).
    result["sqrt_area_fuzzy"] = sqrt_fuzzy
    result["sqrt_area_missed_unlinked"] = missed_sqrt["unlinked"]
    return result


def _score_claim(
    row: dict[str, Any],
    claim_regions: list[AttributionRegion],
    refs: list[AttributionReference],
    sizes: Mapping[int, tuple[float, float]],
) -> None:
    """Score one element's claim against the references it resolved to (in place)."""
    merge = _merge(refs)
    assert merge is not None  # candidates only include admissible merges
    regions = merge.target
    row["reference_ids"] = [ref.id for ref in refs]
    # Box budget: distinct boxes (exact duplicates were dropped in _claims) may
    # not exceed the reference's expected_bbox_count. A stack merge is one
    # block, so it gets 1; a row merge gets one per reference (see _merge).
    # Fewer boxes than expected is scored normally: a single box over a
    # two-column paragraph also covers the gutter, and the union IoU charges
    # that extra area.
    budget = merge.budget
    raw_q = geometry(claim_regions, regions)[0]
    q, coverage, tightness = geometry(_dilate(claim_regions, sizes), _dilate(regions, sizes))
    row.update(
        expected_bbox_count=budget,
        quality=q,
        source_extent_coverage=coverage,
        box_tightness=tightness,
        target_regions=[r.model_dump() for r in regions],
        raw_quality=raw_q,
    )
    boxes = len(claim_regions)
    if boxes > budget and _box_count(claim_regions) <= budget:
        # Side-by-side pieces of one line: scored on their union.
        row.update(row_pieces=boxes, contract_valid=True)
    elif boxes > budget:
        # Stays contract-invalid, but is assessed at 0. The union IoU is
        # kept only as a diagnostic of what the boxes would have scored.
        row.update(status="multi_box_penalized", quality=0.0, unpenalized_quality=q)
    else:
        row["contract_valid"] = True


def _score_group(
    members: list[tuple[dict[str, Any], list[AttributionRegion]]],
    target: list[AttributionRegion],
    ref_ids: list[str],
    sizes: Mapping[int, tuple[float, float]],
    status: str,
    match: str,
) -> None:
    """Several elements that together stand for one target (in place).

    Quality = dilated IoU of the union of the members' boxes with the target;
    every member gets it and the target's ids, so the reference is credited
    once. The Markdown split the content, not the engine, so the group is not
    penalized for having several boxes; each member keeps its own one-box
    budget (row pieces count as one), and one member over it scores the group 0.
    """
    union = [r for _, regions in members for r in regions]
    raw_q = geometry(union, target)[0]
    q, coverage, tightness = geometry(_dilate(union, sizes), _dilate(target, sizes))
    over = [row for row, regions in members if _box_count(regions) > 1]
    for row, regions in members:
        row.update(
            status="multi_box_penalized" if over else status,
            match=match,
            reference_ids=list(ref_ids),
            quality=0.0 if over else q,
            raw_quality=raw_q,
            source_extent_coverage=coverage,
            box_tightness=tightness,
            expected_bbox_count=1,
            group_size=len(members),
            target_regions=[r.model_dump() for r in target],
            contract_valid=not over and _box_count(regions) <= 1,
        )
        if over:
            row["unpenalized_quality"] = q


def _pick_claim(
    matches: list[dict[str, Any]], claims: list[dict[str, Any]], elements: list[Element], index: int
) -> dict[str, Any]:
    """Which of several provider items with the element's text supplies its boxes.

    Taking the first unused one in item order would make two elements with
    the same text swap boxes when the Markdown orders them differently from
    the items (a sidebar entry and the title it repeats). This prefers the
    item whose raw text is the element's, then the one whose neighbouring
    items carry the element's neighbouring Markdown texts; ties keep item
    order. Content only, never geometry.
    """
    position = {id(c): n for n, c in enumerate(claims)}
    before = normalize(elements[index - 1].text) if index > 0 else None
    after = normalize(elements[index + 1].text) if index + 1 < len(elements) else None

    def rank(claim: dict[str, Any]) -> tuple[bool, int]:
        n = position[id(claim)]
        context = int(n > 0 and claims[n - 1]["text"] == before) + int(
            n + 1 < len(claims) and claims[n + 1]["text"] == after
        )
        return claim["raw_text"] == elements[index].text, context

    return max(matches, key=rank)


# ---------------------------------------------------------------------------
# Reading-order anchors. An element that text cannot link sits
# between two elements that text did link (exact or near text, or table
# cells: passes 1-3). The
# nearest one before it has GT order lo (the largest order of its references),
# the nearest one after it GT order hi (the smallest). The references it can
# stand for are those with lo < order < hi: its window. No box is used, so a
# submitted box can never choose the reference it is then scored against.
#
# The window settles what text cannot:
#
# * duplicate texts: an element whose text equals several references keeps the
#   ones inside its window, and resolves when one remains;
# * pictures: image elements pair with the empty-text Picture references
#   inside their window, the ``decorative`` ones included. Consecutive images
#   between the same two anchors and the pictures in that window pair in
#   order when the counts are equal; otherwise the images stay
#   ``ambiguous_picture``. (Unequal counts are left alone: on the v0.3 audit,
#   one image for the union of several pictures, or several images grouped
#   onto one, linked distinct pictures wrongly.) A decorative picture paired
#   and boxed is scored; left out, it is out of the weights (``optional``).
#
# Anchors come only from text links, never from links the window itself made.
# ---------------------------------------------------------------------------

TEXT_MATCHES = frozenset({"text", "table_cells"})


def _order_of(rule: MarkdownAttributionRule) -> dict[str, int]:
    """GT order by reference id; a span takes its parent's order."""
    order = {}
    for ref in rule.references:
        order[ref.id] = ref.order
        for span in ref.spans:
            order[span.id] = ref.order
    return order


def _windows(rows: list[dict[str, Any]], order: Mapping[str, int]) -> list[tuple[float, float]]:
    """(lo, hi) per row from the nearest text-linked rows before and after it."""
    lo, hi = [-math.inf] * len(rows), [math.inf] * len(rows)
    last = -math.inf
    for i, row in enumerate(rows):
        lo[i] = last
        if row.get("match") in TEXT_MATCHES and row["reference_ids"]:
            last = max(order[r] for r in row["reference_ids"])
    nxt = math.inf
    for i in range(len(rows) - 1, -1, -1):
        hi[i] = nxt
        if rows[i].get("match") in TEXT_MATCHES and rows[i]["reference_ids"]:
            nxt = min(order[r] for r in rows[i]["reference_ids"])
    return list(zip(lo, hi, strict=True))


def _linked_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {r for row in rows if row["quality"] is not None or row.get("match") for r in row["reference_ids"]}


def _anchor_pass(
    rows: list[dict[str, Any]],
    claim_regions: Mapping[int, list[AttributionRegion]],
    duplicates: Mapping[int, list[list[AttributionReference]]],
    rule: MarkdownAttributionRule,
    sizes: Mapping[int, tuple[float, float]],
) -> None:
    order = _order_of(rule)
    windows = _windows(rows, order)
    # Duplicate texts.
    for index, groups in duplicates.items():
        lo, hi = windows[index]
        inside = [refs for refs in groups if all(lo < order[r.id] < hi for r in refs)]
        if len(inside) == 1:
            rows[index].update(status="resolved", match="anchor")
            _score_claim(rows[index], claim_regions[index], inside[0], sizes)
    # Pictures.
    taken = _linked_ids(rows)
    pictures = sorted(
        (ref for ref in rule.references if ref.kind == "Picture" and not normalize(ref.text) and ref.id not in taken),
        key=lambda r: r.order,
    )
    runs: dict[tuple[float, float], list[int]] = {}
    for index, row in enumerate(rows):
        if row["kind"] == "image" and not row["reference_ids"] and row["status"] != "invalid_geometry":
            runs.setdefault(windows[index], []).append(index)
    for (lo, hi), images in runs.items():
        window = [ref for ref in pictures if lo < ref.order < hi]
        if not window:
            continue
        if len(images) != len(window):
            for index in images:
                rows[index]["status"] = "ambiguous_picture"
            continue
        for index, ref in zip(images, window, strict=True):
            if index in claim_regions:
                rows[index].update(status="resolved", match="picture_anchor")
                _score_claim(rows[index], claim_regions[index], [ref], sizes)
            else:
                rows[index].update(match="picture_anchor", reference_ids=[ref.id])


# ---------------------------------------------------------------------------
# Many-to-one groups. The Markdown can split one reference into
# several elements (a grid written as a small table plus paragraphs, a court
# name written as a heading and a paragraph); none of them matches the
# reference alone. After per-element linking, each maximal run of consecutive
# boxed elements that are still unresolved is searched, for every reference
# nothing links yet, for the sub-run (2 to GROUP_MAX_ELEMENTS elements) whose
# joined text matches the reference's text best (similarity >=
# text_match_threshold), or, for a table reference, whose cells do (cell F1 >=
# table_match_threshold, each paragraph one cell); ties go to the shorter run.
# (The best score, not the shortest run over the threshold: for a table the
# first half of its cells can already clear the threshold.) The reference must lie in
# the sub-run's reading-order window. The sub-run resolves as one group
# (status resolved_group, match group), scored on the union of its boxes.
# ---------------------------------------------------------------------------

GROUP_MAX_ELEMENTS = 60
GROUPABLE = frozenset({"unresolved_content", "needs_span_geometry"})


def _runs(rows: list[dict[str, Any]], claim_regions: Mapping[int, list[AttributionRegion]]) -> list[list[int]]:
    """Maximal runs of consecutive unresolved elements with at least one boxed.

    An unboxed element inside a run still carries part of the reference's
    text, so it joins the run (it adds no box); invalid geometry and images
    end a run.
    """
    runs: list[list[int]] = [[]]
    for index, row in enumerate(rows):
        boxed = index in claim_regions and row["status"] in GROUPABLE
        unboxed = row["status"] == "unboxed"
        if (boxed or unboxed) and row["kind"] != "image" and not row["reference_ids"]:
            runs[-1].append(index)
        elif runs[-1]:
            runs.append([])
    return [run for run in runs if len(run) >= 2 and any(i in claim_regions for i in run)]


def _group_pass(
    rows: list[dict[str, Any]],
    claim_regions: Mapping[int, list[AttributionRegion]],
    elements: list[Element],
    rule: MarkdownAttributionRule,
    sizes: Mapping[int, tuple[float, float]],
) -> None:
    order = _order_of(rule)
    windows = _windows(rows, order)
    linked = _linked_ids(rows)
    missed = [
        ref
        for ref in sorted(rule.references, key=lambda r: r.order)
        if normalize(ref.text)
        and not _is_marker_reference(ref)
        and ref.id not in linked
        and not any(span.id in linked for span in ref.spans)
    ]
    texts = {i: normalize(e.text) for i, e in enumerate(elements)}
    runs = _runs(rows, claim_regions)
    for ref in missed:
        table = _table_scorer(ref)
        targets = [normalize(t) for t in (ref.text, *ref.alt_texts)]
        ref_cells = table_cells(ref.text) if table is not None else None
        gt_cells = sum(ref_cells.values()) if ref_cells is not None else 0
        threshold = rule.table_match_threshold if table is not None else rule.text_match_threshold
        best: tuple[float, int, int, int] | None = None  # (-score, length, run, start): best score, then shortest
        for r, run in enumerate(runs):
            for start in range(len(run) - 1):
                if run[start] not in claim_regions or not windows[run[start]][0] < ref.order < windows[run[start]][1]:
                    continue
                joined: list[str] = []
                cells: Counter[str] = Counter()
                for end in range(start, min(len(run), start + GROUP_MAX_ELEMENTS)):
                    index = run[end]
                    if table is not None:
                        element_cells = table_cells(elements[index].text)
                        cells += (
                            element_cells if element_cells is not None else Counter([_cell_key(elements[index].text)])
                        )
                        predicted = sum(cells.values())
                        # F1 cannot reach the threshold once the prediction is this much larger.
                        if gt_cells and 2 * gt_cells / (predicted + gt_cells) < threshold:
                            break
                    joined.append(texts[index])
                    if end == start:
                        continue
                    if table is not None:
                        score = table(cells)
                    else:
                        text = " ".join(joined)
                        if len(text) > 1.2 * max(len(t) for t in targets):
                            break
                        score = max(similarity(text, t) for t in targets)
                    candidate = (-score, end - start + 1, r, start)
                    if score >= threshold and (best is None or candidate < best):
                        best = candidate
        if best is None:
            continue
        _, length, r, start = best
        members = runs[r][start : start + length]
        boxed = [i for i in members if i in claim_regions]
        if not boxed:
            continue
        _score_group(
            [(rows[i], claim_regions[i]) for i in boxed], ref.regions, [ref.id], sizes, "resolved_group", "group"
        )
        for i in members:
            if i not in claim_regions:
                rows[i].update(match="group", reference_ids=[ref.id], group_size=len(boxed))
        rows_left = runs[r][:start], runs[r][start + length :]
        runs[r : r + 1] = [part for part in rows_left if len(part) >= 2] or [[]]


# ---------------------------------------------------------------------------
# Relaxed links. A boxed element whose text matched nothing
# (unresolved_content) after the passes above shortlists the references
# nothing links yet, inside its reading-order window, whose text is close in a
# looser sense: token_set_ratio >= RELAXED_TEXT_RATIO (0-100) with the same
# numbers (the digit gate) and comparable lengths, or, for a table element and a table reference,
# cell F1 >= RELAXED_TABLE_F1. It resolves when exactly one is shortlisted.
# Never by geometry: a box that picks its own reference always looks right.
# Such rows say match "relaxed", are counted (relaxed_links) and never serve as
# anchors. Stricter than the first design, which resolved a single candidate
# even outside the window and let a table relax to a text reference: on the
# v0.3 audit those were the wrong links. A split piece of a paragraph
# (needs_span_geometry) is not relaxed-linked: scored against the whole
# paragraph's box it would be charged for the part it does not claim.
RELAXED_TEXT_RATIO = 60
# token_set_ratio scores a subset at 100, so an element that merges a heading
# with its price, or a whole form row with one label, would pass: the shorter
# text must be at least half the longer.
RELAXED_MIN_LENGTH_RATIO = 0.5
RELAXED_TABLE_F1 = 0.3
# No containment gate here, unlike the fuzzy fallback below. On the v0.3
# rewrite stress test a 0.85 gate removed 216 of 237 wrong relaxed links but
# also 13,487 of 19,295 right ones (a typo breaks three character 3-grams), and
# on the real outputs it dropped correct links such as "1 Green Bonds" to
# "Green Bonds". Every threshold from 0.3 to 0.85 lost 30 to 60 right links
# per wrong one it removed. Most of the wrong ones are extent errors (an
# element holding two references linked to the larger one), not other content.
#
# Known limits, kept on purpose for now:
#
# * No negation gate. The shortlist checks numbers and word overlap but not
#   the words of _PROTECTED (not, no, never, ...), unlike similarity(). That is
#   arguably right for attribution, which asks where text is printed, not
#   whether it was transcribed faithfully.
# * A merged element can relaxed-link to one of its references. token_set_ratio
#   scores a subset at 100 and the length gate only needs the reference to be
#   half the element, so an element holding reference A plus a shorter
#   neighbour links to A alone. The low IoU charges the large box, but A counts
#   as assessed (raising coverage) and the neighbours stay missed (unlinked).
#   Example: on markdown_block_layout v0.3 layout_0012_f944_p2 (cost_effective,
#   new engine), one element holding a "Yes. Designee's name" label, a PIN
#   line and a "No" checkbox links to the PIN line at IoU 0.241.


def _relaxed_pass(
    rows: list[dict[str, Any]],
    claim_regions: Mapping[int, list[AttributionRegion]],
    elements: list[Element],
    rule: MarkdownAttributionRule,
    sizes: Mapping[int, tuple[float, float]],
) -> None:
    order = _order_of(rule)
    windows = _windows(rows, order)
    linked = _linked_ids(rows)
    missed = [
        ref
        for ref in rule.references
        if normalize(ref.text) and ref.id not in linked and not any(span.id in linked for span in ref.spans)
    ]
    for index, row in enumerate(rows):
        if index not in claim_regions or row["status"] != "unresolved_content" or row["reference_ids"]:
            continue
        element = elements[index]
        side = _picture_side(element.text)
        picture_side = element.kind == "image" or side.is_picture
        text = normalize(element.text)
        cells = table_cells(element.text) if element.kind == "table" else None
        lo, hi = windows[index]
        shortlist = []
        for ref in missed:
            if picture_side != (ref.kind == "Picture") or (_is_marker_reference(ref) and not picture_side):
                continue
            if not lo < order[ref.id] < hi:
                continue
            table = _table_scorer(ref)
            if (cells is None) != (table is None):
                continue  # a table relaxes only to a table reference, and text only to text
            if cells is not None and table is not None:
                if table(cells) >= RELAXED_TABLE_F1:
                    shortlist.append(ref)
                continue
            for alt in (ref.text, *ref.alt_texts):
                target = normalize(alt)
                if (
                    min(len(text), len(target)) >= RELAXED_MIN_LENGTH_RATIO * max(len(text), len(target))
                    and _same_digits(text, target)
                    and token_set_ratio(text, target) >= RELAXED_TEXT_RATIO
                ):
                    shortlist.append(ref)
                    break
        if len(shortlist) == 1:
            row.update(status="resolved", match="relaxed")
            _score_claim(row, claim_regions[index], shortlist, sizes)


# ---------------------------------------------------------------------------
# Fuzzy fallback. Runs after every other pass and only adds links.
# It pairs the boxed elements still unresolved (unresolved_content, or
# ambiguous_source short of an exact duplicate, which the anchor pass owns)
# with the references nothing links yet, so a transcription difference that
# every exact or token test misses (a thousands separator, dropped
# punctuation, two swapped words) still reaches its reference.
#
# Similarity: cosine of TF-IDF vectors of character 3-grams taken inside word
# boundaries (each word padded with a space, as scikit-learn's char_wb). The
# IDF is fitted per page on every element text and reference text of that
# page, so n-grams shared by every block on the page count for little. A pair
# is accepted only if all of these hold:
#
# * kind: a table element pairs only with a table reference, a text element
#   only with a non-table one; pictures, images and marker references stay on
#   the picture path (_anchor_pass);
# * reading order: every reference of the target lies inside the element's
#   window (_windows, text anchors only);
# * one-to-one: the Hungarian assignment over the page's gated similarity
#   matrix picks at most one target per element and one element per target;
# * cosine >= FUZZY_MIN_SIMILARITY, and FUZZY_MIN_MARGIN over the best rival
#   (another target of the element that shares no reference with this one, or
#   another element of the target), so a near-tie is left alone;
# * containment >= FUZZY_MIN_CONTAINMENT: each text holds that share of the
#   other's character 3-grams. The cosine alone accepts an element that holds
#   the target plus a neighbouring annotated block, and scores its box against
#   the wrong extent; containment is the gate that refuses it;
# * text pairs only: the shorter text is at least FUZZY_MIN_LENGTH_RATIO of the
#   longer. Containment compares 3-gram sets, so repeated content passes it:
#   one sentence would meet the merge of two near-identical references (two
#   form rows that differ in a date). Tables are exempt: they never merge, and
#   a pipe table that repeats a spanning header cell per column is much longer
#   than the same table in HTML.
#
# There is no digit gate: number formatting ("1,234.5" vs "1234.5") is the
# main thing the fallback exists for, and on short texts the digits' own
# 3-grams already weigh enough in the containment to keep "Table 1" from
# "Table 2".
#
# Targets are single references, annotated spans (a paragraph continued in
# another column meets its span), and admissible merges (_merge) of 2 to
# max_merge_elements free references consecutive in GT order. Accepted rows
# get status fuzzy_resolved and match "fuzzy", are scored like any resolved
# element (dilated IoU, box budget) and never serve as anchors ("fuzzy" is not
# in TEXT_MATCHES). No predicted geometry is read. Thresholds were chosen on
# the v0.3 ce arms and checked on the agentic arms with a rewrite stress test
# (docs/markdown_attribution_v1.md, "Scorer version 2", item 6).
# ---------------------------------------------------------------------------

FUZZY_MIN_SIMILARITY = 0.7
FUZZY_MIN_MARGIN = 0.1
FUZZY_MIN_CONTAINMENT = 0.85
FUZZY_MIN_LENGTH_RATIO = 0.6
FUZZY_NGRAM = 3
FUZZY_STATUSES = frozenset({"unresolved_content", "ambiguous_source"})


def _grams(text: str) -> set[str]:
    padded = f" {text} "
    return {padded[i : i + FUZZY_NGRAM] for i in range(len(padded) - FUZZY_NGRAM + 1)}


def containment(a: str, b: str) -> float:
    """min(share of a's character 3-grams found in b, share of b's found in a).

    Low when one side holds content the other lacks: "green bonds" finds all
    its 3-grams in "1 green bonds", but only 11 of the 13 of "1 green bonds"
    are in "green bonds" (0.85, just under FUZZY_MIN_CONTAINMENT); a paragraph
    plus its heading is far under it against the paragraph alone. Inputs are
    normalized texts (spaces padded at both ends); 0 when either is empty.
    """
    ga, gb = _grams(a), _grams(b)
    if not a or not b or not ga or not gb:
        return 0.0
    shared = len(ga & gb)
    return min(shared / len(ga), shared / len(gb))


def _word_grams(text: str) -> Counter[str]:
    """Character n-grams inside word boundaries, each word padded with a space.

    The same n-grams as scikit-learn's ``char_wb`` analyzer: a word shorter
    than n contributes itself once, padded.
    """
    grams: Counter[str] = Counter()
    n = FUZZY_NGRAM
    for word in text.split():
        padded = f" {word} "
        if len(padded) <= n:
            grams[padded] += 1
            continue
        for i in range(len(padded) - n + 1):
            grams[padded[i : i + n]] += 1
    return grams


def tfidf_cosine(corpus: list[str], left: list[str], right: list[str]) -> np.ndarray:
    """Cosine of TF-IDF vectors of char_wb 3-grams, IDF fitted on ``corpus``.

    TF is sublinear (1 + log count), IDF smoothed (log((1 + n) / (1 + df)) + 1),
    vectors L2-normalized: the scikit-learn TfidfVectorizer defaults with
    ``sublinear_tf=True``, without the dependency. N-grams absent from the
    corpus are dropped, as a fitted vectorizer drops them. Returns a
    len(left) x len(right) matrix; a text with no known n-gram has cosine 0
    with everything. Deterministic: the vocabulary is sorted.
    """
    df: Counter[str] = Counter()
    for text in corpus:
        df.update(set(_word_grams(text)))
    vocabulary = {gram: i for i, gram in enumerate(sorted(df))}
    idf = np.array([math.log((1 + len(corpus)) / (1 + df[g])) + 1 for g in sorted(df)])

    def vectors(texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(vocabulary)))
        for row, text in enumerate(texts):
            for gram, count in _word_grams(text).items():
                if gram in vocabulary:
                    matrix[row, vocabulary[gram]] = (1 + math.log(count)) * idf[vocabulary[gram]]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        unit: np.ndarray = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        return unit

    if not left or not right:
        return np.zeros((len(left), len(right)))
    cosine: np.ndarray = vectors(left) @ vectors(right).T
    return cosine


def _fuzzy_assign(similarity: np.ndarray, rivals: Callable[[int, int], bool]) -> list[tuple[int, int, float, float]]:
    """(element, target, similarity, margin) pairs the fallback accepts.

    Hungarian assignment on the gated matrix (a refused pair is 0), then the
    threshold and the margin. ``rivals(j, k)``: targets j and k compete (they
    share no reference; a span and its parent, or two overlapping merges, are
    one content and not rivals of each other).
    """
    if similarity.size == 0 or similarity.max() <= 0:
        return []
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-similarity)
    accepted = []
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        value = float(similarity[i, j])
        if value < FUZZY_MIN_SIMILARITY:
            continue
        row_rival = max((similarity[i, k] for k in range(similarity.shape[1]) if k != j and rivals(j, k)), default=0.0)
        col_rival = max((similarity[k, j] for k in range(similarity.shape[0]) if k != i), default=0.0)
        margin = value - float(max(row_rival, col_rival))
        if margin >= FUZZY_MIN_MARGIN:
            accepted.append((i, j, value, margin))
    return accepted


def _fuzzy_pass(
    rows: list[dict[str, Any]],
    claim_regions: Mapping[int, list[AttributionRegion]],
    elements: list[Element],
    rule: MarkdownAttributionRule,
    sizes: Mapping[int, tuple[float, float]],
) -> None:
    order = _order_of(rule)
    windows = _windows(rows, order)
    linked = _linked_ids(rows)
    candidates = []
    for index, row in enumerate(rows):
        if index not in claim_regions or row["reference_ids"] or row["status"] not in FUZZY_STATUSES:
            continue
        if row["status"] == "ambiguous_source" and row.get("text_similarity") == 1.0:
            continue  # an exact duplicate: the anchor pass owns it
        element = elements[index]
        if element.kind == "image" or _picture_side(element.text).is_picture or not normalize(element.text):
            continue
        candidates.append(index)
    if not candidates:
        return
    references = sorted(rule.references, key=lambda r: r.order)
    free = [
        ref
        for ref in references
        if normalize(ref.text)
        and ref.kind != "Picture"
        and not _is_marker_reference(ref)
        and ref.id not in linked
        and not any(span.id in linked for span in ref.spans)
    ]
    if not free:
        return
    # Targets: (references, their texts, is a table).
    targets: list[tuple[list[AttributionReference], list[str], bool]] = []
    for ref in free:
        for r in (ref, *ref.spans):
            readings = [t for t in dict.fromkeys(normalize(x) for x in (r.text, *r.alt_texts)) if t]
            if readings:
                targets.append(([r], readings, table_cells(r.text) is not None))
    position = {ref.id: n for n, ref in enumerate(references)}
    for size in range(2, rule.max_merge_elements + 1):
        for start in range(len(free) - size + 1):
            group = free[start : start + size]
            if position[group[-1].id] - position[group[0].id] != size - 1:
                continue  # not consecutive in GT order: a linked reference sits between them
            if any(table_cells(r.text) is not None for r in group):
                continue
            merge = _merge(group)
            if merge is not None and (text := normalize(merge.text)):
                targets.append((group, [text], False))
    texts = {i: normalize(elements[i].text) for i in candidates}
    is_table = {i: elements[i].kind == "table" and table_cells(elements[i].text) is not None for i in candidates}
    corpus = [t for t in (normalize(e.text) for e in elements) if t] + [
        t for ref in rule.references for x in (ref.text, *ref.alt_texts) if (t := normalize(x))
    ]
    flat = [t for _, target_texts, _ in targets for t in target_texts]
    cosine = tfidf_cosine(corpus, [texts[i] for i in candidates], flat)
    offsets = np.cumsum([0] + [len(target_texts) for _, target_texts, _ in targets])
    gated = np.zeros((len(candidates), len(targets)))
    for a, index in enumerate(candidates):
        lo, hi = windows[index]
        for j, (refs, target_texts, table) in enumerate(targets):
            if table != is_table[index] or not all(lo < order[r.id] < hi for r in refs):
                continue
            best = [
                cosine[a, offsets[j] + n]
                for n, t in enumerate(target_texts)
                if containment(texts[index], t) >= FUZZY_MIN_CONTAINMENT
                and (table or min(len(t), len(texts[index])) >= FUZZY_MIN_LENGTH_RATIO * max(len(t), len(texts[index])))
            ]
            if best:
                gated[a, j] = max(best)
    families = [frozenset(_family(refs, rule)) for refs, _, _ in targets]
    accepted = _fuzzy_assign(gated, lambda j, k: not families[j] & families[k])
    taken: set[str] = set()
    # Overlapping merges: the higher similarity wins (stable, so ties keep row order).
    for a, j, value, margin in sorted(accepted, key=lambda v: -v[2]):
        if families[j] & taken:
            continue
        index = candidates[a]
        rows[index].update(status="fuzzy_resolved", match="fuzzy", fuzzy_similarity=value, fuzzy_margin=margin)
        _score_claim(rows[index], claim_regions[index], targets[j][0], sizes)
        taken |= families[j]


# ---------------------------------------------------------------------------
# Diagnostics outside the headline.
#
# Kind agreement: for each element linked to a single reference (groups
# excluded), does the Markdown write it as the kind the reference is? The
# reference's kind maps to heading (Section-header, Title), table (Table),
# image (a Picture without a table), formula (Formula), or paragraph (Text,
# Caption, Footnote, List-item, page furniture); a chart (a Picture holding a
# table) agrees with a table or an image. Form, Key-Value Region and Code carry
# no kind signal and are left out. The Markdown kind is the element's: a $$
# block is a formula, a bold-only line is a paragraph (a reader of the
# Markdown sees a paragraph). Weighted by sqrt(area) like the headline, with a
# confusion matrix (reference class -> Markdown class) and per-class counts.
# ---------------------------------------------------------------------------

KIND_CLASSES = ("heading", "paragraph", "table", "image", "formula")
_KIND_OF_REFERENCE = {
    "Section-header": "heading",
    "Title": "heading",
    "Table": "table",
    "Formula": "formula",
    "Text": "paragraph",
    "Caption": "paragraph",
    "Footnote": "paragraph",
    "List-item": "paragraph",
    "Page-header": "paragraph",
    "Page-footer": "paragraph",
    "Page-number": "paragraph",
}


def _reference_class(ref: AttributionReference) -> str | None:
    if ref.kind == "Picture":
        return "chart" if table_cells(ref.text) is not None else "image"
    return _KIND_OF_REFERENCE.get(ref.kind)


def _markdown_class(row: Mapping[str, Any]) -> str:
    if row["kind"] in {"heading", "table", "image"}:
        return str(row["kind"])
    if row["text"].lstrip().startswith("$$"):
        return "formula"
    return "paragraph"


def _kind_agreement(rows: list[dict[str, Any]], rule: MarkdownAttributionRule) -> dict[str, Any]:
    by_id = {r.id: r for ref in rule.references for r in (ref, *ref.spans)}
    agree = total = 0.0
    confusion: Counter[str] = Counter()
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in KIND_CLASSES}
    for row in rows:
        if row["quality"] is None or len(row["reference_ids"]) != 1 or row.get("match") == "group":
            continue
        ref = by_id[row["reference_ids"][0]]
        expected = _reference_class(ref)
        if expected is None:
            continue
        written = _markdown_class(row)
        if expected == "chart":
            expected = written if written in {"table", "image"} else "table"
        weight = math.sqrt(_clamped_area(ref.regions))
        total += weight
        agree += weight * (written == expected)
        confusion[f"{expected}->{written}"] += 1
        if written == expected:
            counts[expected]["tp"] += 1
        else:
            counts[expected]["fn"] += 1
            counts[written]["fp"] += 1
    return {"score": agree, "total": total, "confusion": dict(confusion), "classes": counts}


# Reading order: Kendall tau-b between the Markdown position of each assessed
# element and the GT order of its reference (the smallest order for a merge),
# with ties (a group's members share one order). 1 means the Markdown keeps the
# GT order, -1 reverses it; it catches long-range swaps such as two columns
# written in the wrong order, which adjacent-pair accuracy misses. Emitted when
# at least ORDER_MIN_PAIRS pairs are comparable. It depends on linking: a page
# with low coverage has few pairs.
ORDER_MIN_PAIRS = 3


def _order_tau(rows: list[dict[str, Any]], rule: MarkdownAttributionRule) -> dict[str, Any] | None:
    order = _order_of(rule)
    ranks = [min(order[r] for r in row["reference_ids"]) for row in rows if row["quality"] is not None]
    concordant = discordant = ties = 0
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            if ranks[i] < ranks[j]:
                concordant += 1
            elif ranks[i] > ranks[j]:
                discordant += 1
            else:
                ties += 1
    comparable = concordant + discordant
    if comparable < ORDER_MIN_PAIRS:
        return None
    pairs = comparable + ties
    # tau-b: the Markdown positions never tie, the GT orders may.
    tau = (concordant - discordant) / math.sqrt(pairs * comparable)
    return {"tau": tau, "concordant": concordant, "discordant": discordant, "ties": ties}


def score_markdown_attribution(output: ParseOutput, rule: MarkdownAttributionRule) -> dict[str, Any]:
    sizes = _page_sizes(output)
    elements = markdown_elements(output.markdown)
    claims = _claims(output)
    candidates = _candidates(rule)
    # What elements may match: the texts plus the alternative texts. The
    # duplicate test in _area_weighting still sees the primary texts only.
    matchable = candidates + _alt_candidates(rule)
    cells = {i: c for i, e in enumerate(elements) if e.kind == "table" and (c := table_cells(e.text)) is not None}
    tables = _table_assignment(list(cells.items()), rule)
    table_ref_ids = {ref.id for ref, _ in tables.values()}
    rows = []
    # For the linking passes: the valid boxes of each boxed element, and the exact
    # candidates of an element whose text equals several references.
    claim_regions: dict[int, list[AttributionRegion]] = {}
    duplicates: dict[int, list[list[AttributionReference]]] = {}
    for index, element in enumerate(elements):
        text = normalize(element.text)
        # Match final MD to provider claims by content occurrence, never geometry.
        matches = [
            c
            for c in claims
            if not c["used"] and ((text and c["text"] == text) or (not text and c["raw_text"] == element.text))
        ]
        if not matches and index in cells:
            # A provider table item can carry a pipe table (colspans repeated)
            # while the final Markdown holds the same table as HTML: pair by cells.
            best = max(
                (
                    (cell_f1(cells[index], item_cells), -n, c)
                    for n, c in enumerate(claims)
                    if not c["used"] and c["kind"] == "table" and (item_cells := table_cells(c["raw_text"])) is not None
                ),
                default=None,
                key=lambda v: v[:2],
            )
            if best and best[0] >= PAIRING_TABLE_F1:
                matches = [best[2]]
        if len(matches) > 1:
            matches = [_pick_claim(matches, claims, elements, index)]
        claim = matches[0] if matches else None
        if claim:
            claim["used"] = True
        row: dict[str, Any] = {
            "element_id": f"md-{index + 1}",
            "kind": element.kind,
            "text": element.text,
            "lines": [element.start_line, element.end_line],
            "boxed": bool(claim and claim["attempted"]),
            "quality": None,
            "status": "unboxed",
            "reference_ids": [],
        }
        row["contract_valid"] = bool(
            claim and claim["attempted"] and not claim["invalid"] and len(claim["regions"]) == 1
        )
        if not row["boxed"]:
            rows.append(row)
            continue
        assert claim is not None  # boxed requires an attempted provider claim
        row["predicted_regions"] = [r.model_dump() for r in claim["regions"]]
        if claim["invalid"]:
            row["status"] = "invalid_geometry"
            rows.append(row)
            continue
        claim_regions[index] = claim["regions"]
        if index in tables:
            ref, f1 = tables[index]
            refs: list[AttributionReference] | None = [ref]
            status, confidence = "resolved", f1
            row["match"] = "table_cells"
        else:
            # Non-tables, and tables no reference reaches by cells: whole-text path,
            # minus references another table already took.
            usable = matchable
            if index in cells:
                usable = [(c, r) for c, r in matchable if not any(x.id in table_ref_ids for x in r)]
            usable = _picture_filter(element, usable)
            refs, status, confidence = _correspond(text, usable, rule)
            if refs is not None:
                row["match"] = "text"
            elif status == "ambiguous_source" and confidence == 1.0:
                duplicates[index] = list({tuple(r.id for r in g): g for c, g in usable if c == text}.values())
        row.update(status=status, text_similarity=confidence)
        if refs is not None:
            _score_claim(row, claim["regions"], refs, sizes)
        rows.append(row)
    _anchor_pass(rows, claim_regions, duplicates, rule, sizes)
    _group_pass(rows, claim_regions, elements, rule, sizes)
    _relaxed_pass(rows, claim_regions, elements, rule, sizes)
    _fuzzy_pass(rows, claim_regions, elements, rule, sizes)
    boxed = sum(row["boxed"] for row in rows)
    assessed = [row for row in rows if row["quality"] is not None]
    parameters: dict[str, Any] = {
        "text_match_threshold": rule.text_match_threshold,
        "ambiguity_margin": rule.ambiguity_margin,
        "max_merge_elements": rule.max_merge_elements,
        "table_match_threshold": rule.table_match_threshold,
    }
    markdown_text = _MarkdownText(
        text=normalize(output.markdown),
        cells=sum(cells.values(), Counter()),
        images=sum(row["kind"] == "image" and row["quality"] is None for row in rows),
    )
    result = {
        "version": SCORER_VERSION,
        "parameters": parameters,
        "multi_box_penalized": sum(row["status"] == "multi_box_penalized" for row in rows),
        "generated": len(rows),
        "boxed": boxed,
        "assessed": len(assessed),
        "valid_boxed": sum(row["contract_valid"] for row in rows),
        "quality": sum(row["quality"] for row in assessed) / len(assessed) if assessed else None,
        "box_coverage": boxed / len(rows) if rows else None,
        "valid_contract_coverage": sum(row["contract_valid"] for row in rows) / len(rows) if rows else None,
        "assessment_coverage": len(assessed) / boxed if boxed else None,
        "unlinked_claims": sum(c["attempted"] and not c["used"] for c in claims),
        # Provider claims no final-Markdown element matched: the boxes behind
        # unlinked_claims, kept so the report can draw them.
        "unlinked": [
            {
                "kind": c["kind"],
                "text": _snippet(c["raw_text"]),
                "invalid": c["invalid"],
                "regions": [r.model_dump() for r in c["regions"]],
            }
            for c in claims
            if c["attempted"] and not c["used"]
        ],
        "statuses": dict(Counter(row["status"] for row in rows)),
        "elements": rows,
        **_area_weighting(rule, rows, candidates, markdown_text),
    }
    result["furniture"] = _furniture_score(output, rule, sizes)
    result["relaxed_links"] = sum(row.get("match") == "relaxed" for row in rows)
    result["fuzzy_links"] = sum(row.get("match") == "fuzzy" for row in rows)
    # Share of the assessed sqrt(area) weight that only the fuzzy fallback
    # linked; None when nothing was assessed.
    total = result["sqrt_area_total"]
    result["fuzzy_weight_share"] = result["sqrt_area_fuzzy"] / total if total > 0 else None
    result["kind_agreement"] = _kind_agreement(rows, rule)
    result["order_tau"] = _order_tau(rows, rule)
    return result


def compute_markdown_attribution_metrics(
    output: ParseOutput, rules: list[MarkdownAttributionRule]
) -> list[MetricValue]:
    if not rules:
        return []
    if len(rules) != 1:
        raise ValueError("exactly one document-scoped markdown_attribution rule is allowed")
    result = score_markdown_attribution(output, rules[0])
    metrics = []
    if result["sqrt_area_quality"] is not None:
        # Headline. micro_ divides summed quality x sqrt(area) by summed
        # sqrt(area) over every assessed reference of every page; avg_ is the
        # page mean. Missed references are left out, as in the count-weighted
        # quality; they show in sqrt_area_coverage.
        metrics.append(
            MetricValue(
                metric_name=PREFIX + "sqrt_area_quality",
                value=result["sqrt_area_quality"],
                metadata={
                    "passed": result["sqrt_area_score"],
                    "total": result["sqrt_area_total"],
                    "numerator_label": "score",
                    "weighting": "sqrt_area",
                    "scorer_version": result["version"],
                    **{k: result[k] for k in ("p0_score", "p0_total", "p1_score", "p1_total") if k in result},
                },
            )
        )
    if result["sqrt_area_coverage"] is not None:
        # Recall beside the headline: assessed sqrt(area) / scorable sqrt(area).
        metrics.append(
            MetricValue(
                metric_name=PREFIX + "sqrt_area_coverage",
                value=result["sqrt_area_coverage"],
                metadata={
                    "passed": result["sqrt_area_total"],
                    "total": result["sqrt_area_scorable"],
                    "numerator_label": "score",
                    "weighting": "sqrt_area",
                    "scorer_version": result["version"],
                },
            )
        )
    if result["area_quality"] is not None:
        # Secondary. avg_ over cases is the page mean (each page counts once);
        # micro_ divides summed quality-area by summed area (area-weighted micro).
        metrics.append(
            MetricValue(
                metric_name=PREFIX + "area_quality",
                value=result["area_quality"],
                metadata={
                    "passed": result["area_score"],
                    "total": result["area_total"],
                    "numerator_label": "score",
                    "weighting": "area",
                    "area_by_status": result["area_by_status"],
                    "scorer_version": result["version"],
                },
            )
        )
    metrics.append(
        MetricValue(
            metric_name=PREFIX + "generated_elements",
            value=float(result["generated"]),
            metadata={**result, "count": result["generated"]},
        )
    )
    for name, numerator, denominator in (
        ("quality", sum(r["quality"] for r in result["elements"] if r["quality"] is not None), result["assessed"]),
        ("box_coverage", result["boxed"], result["generated"]),
        ("valid_contract_coverage", result["valid_boxed"], result["generated"]),
        ("assessment_coverage", result["assessed"], result["boxed"]),
    ):
        # Absent means N/A. Never emit an artificial perfect/zero quality score.
        if denominator:
            metrics.append(
                MetricValue(
                    metric_name=PREFIX + name,
                    value=numerator / denominator,
                    metadata={
                        "passed": numerator,
                        "total": denominator,
                        "numerator_label": "score" if name == "quality" else "passed",
                        "scorer_version": result["version"],
                    },
                )
            )
    metrics.extend(_diagnostic_metrics(result))
    for name in ("boxed", "assessed", "unlinked_claims", "multi_box_penalized"):
        metrics.append(
            MetricValue(
                metric_name=PREFIX + name + "_count", value=float(result[name]), metadata={"count": result[name]}
            )
        )
    # Pages left out of the area-weighted quality and sqrt coverage: no scorable area.
    zero_area = int(result["area_quality"] is None)
    metrics.append(
        MetricValue(
            metric_name=PREFIX + "zero_area_excluded_count", value=float(zero_area), metadata={"count": zero_area}
        )
    )
    return metrics


def _ratio_metric(name: str, numerator: float, denominator: float, **extra: Any) -> MetricValue:
    return MetricValue(
        metric_name=PREFIX + name,
        value=numerator / denominator,
        metadata={
            "passed": numerator,
            "total": denominator,
            "numerator_label": "score",
            "scorer_version": SCORER_VERSION,
            **extra,
        },
    )


def _diagnostic_metrics(result: Mapping[str, Any]) -> list[MetricValue]:
    """Diagnostics outside the headline (added by scorer version 2)."""
    metrics = []
    scorable = result["sqrt_area_scorable"]
    if scorable > 0:
        # Missed weight by reason, as a share of the scorable sqrt(area) weight.
        metrics.append(_ratio_metric("missed_absent_share", result["sqrt_area_missed_absent"], scorable))
        metrics.append(_ratio_metric("missed_unlinked_share", result["sqrt_area_missed_unlinked"], scorable))
    if (tau := result["order_tau"]) is not None:
        # Page value; avg_ over pages is the mean tau. Not a ratio: no passed/total.
        metrics.append(
            MetricValue(
                metric_name=PREFIX + "order_tau",
                value=tau["tau"],
                metadata={k: tau[k] for k in ("concordant", "discordant", "ties")} | {"scorer_version": SCORER_VERSION},
            )
        )
    kinds = result["kind_agreement"]
    if kinds["total"] > 0:
        # Kind agreement, sqrt(area)-weighted; the confusion matrix counts pairs.
        metrics.append(_ratio_metric("kind_agreement", kinds["score"], kinds["total"], confusion=kinds["confusion"]))
    for kind, counts in kinds["classes"].items():
        if any(counts.values()):
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            # tp/fp/fn metadata: the runner pools them into micro_..._kind_<class>_f1.
            metrics.append(
                MetricValue(
                    metric_name=PREFIX + f"kind_{kind}_f1",
                    value=f1,
                    metadata={"tp": tp, "fp": fp, "fn": fn, "scorer_version": SCORER_VERSION},
                )
            )
    if result["sqrt_area_total"] > 0:
        # How much of the headline's weight rests on fuzzy links: a jump between
        # runs means the fallback, not exact text, decides more of the score.
        metrics.append(
            _ratio_metric(
                "fuzzy_weight_share",
                result["sqrt_area_fuzzy"],
                result["sqrt_area_total"],
                fuzzy_links=result["fuzzy_links"],
                relaxed_links=result["relaxed_links"],
            )
        )
    for name in ("fuzzy_links", "relaxed_links"):
        metrics.append(
            MetricValue(
                metric_name=PREFIX + name + "_count",
                value=float(result[name]),
                metadata={"count": result[name], "scorer_version": SCORER_VERSION},
            )
        )
    furniture = result["furniture"]
    if furniture["total"] > 0:
        metrics.append(_ratio_metric("furniture_quality", furniture["score"], furniture["total"]))
    if furniture["scorable"] > 0:
        metrics.append(_ratio_metric("furniture_coverage", furniture["total"], furniture["scorable"]))
    return metrics


# Report labels for the attribution headline block, in display order. The
# pooled sqrt-area-weighted quality over assessed references (H) is the
# headline, with its coverage (C) beside it: read them as a pair. The
# attribution F = 2HC / (H + C) of the pooled H and C is the one combined
# number (derived_aggregates); the page mean shows whether the pooled value
# holds page by page. The count- and area-weighted views are secondary.
HEADLINE_ROWS = (
    (
        "micro_" + PREFIX + "sqrt_area_quality",
        "Md Attribution Sqrt-Area-Weighted Quality H (Micro, Assessed Only, Headline)",
    ),
    ("micro_" + PREFIX + "sqrt_area_coverage", "Md Attribution Sqrt-Area Coverage C (Micro, Assessed / Scorable)"),
    ("micro_" + PREFIX + "sqrt_area_f", "Md Attribution F (Combined 2HC/(H+C) of Micro H and C)"),
    ("avg_" + PREFIX + "sqrt_area_quality", "Md Attribution Sqrt-Area-Weighted Quality H (Page Mean, Secondary)"),
    ("micro_" + PREFIX + "quality", "Md Attribution Count-Weighted Quality (Micro Element-Weighted, Secondary)"),
    ("avg_" + PREFIX + "quality", "Md Attribution Count-Weighted Quality (Page Mean, Secondary)"),
    ("avg_" + PREFIX + "area_quality", "Md Attribution Area-Weighted Quality (Page Mean, Missed Count 0, Secondary)"),
    ("micro_" + PREFIX + "area_quality", "Md Attribution Area-Weighted Quality (Micro, Missed Count 0, Secondary)"),
    ("total_" + PREFIX + "zero_area_excluded_count", "Md Attribution Pages Excluded (Zero Scorable Area)"),
    # Diagnostics outside the headline, added by scorer version 2 (absent from
    # runs scored with version 1).
    ("micro_" + PREFIX + "missed_absent_share", "Md Attribution Missed Weight, Absent From The Markdown (Micro, v2)"),
    (
        "micro_" + PREFIX + "missed_unlinked_share",
        "Md Attribution Missed Weight, In The Markdown But Unlinked (Micro, v2)",
    ),
    ("micro_" + PREFIX + "kind_agreement", "Md Attribution Kind Agreement (Micro Sqrt-Area, v2)"),
    ("macro_" + PREFIX + "kind_f1", "Md Attribution Kind Macro-F1 (v2)"),
    ("avg_" + PREFIX + "order_tau", "Md Attribution Reading-Order Tau-b (Page Mean, v2)"),
    ("micro_" + PREFIX + "furniture_quality", "Md Attribution Furniture Quality Through Items (Micro, v2)"),
    ("micro_" + PREFIX + "furniture_coverage", "Md Attribution Furniture Coverage Through Items (Micro, v2)"),
    (
        "micro_" + PREFIX + "fuzzy_weight_share",
        "Md Attribution Weight Linked By The Fuzzy Fallback (Micro Share Of Assessed, v2 Diagnostic)",
    ),
    ("total_" + PREFIX + "fuzzy_links_count", "Md Attribution Fuzzy Links (Total, v2 Diagnostic)"),
    ("total_" + PREFIX + "relaxed_links_count", "Md Attribution Relaxed Links (Total, v2 Diagnostic)"),
)


def derived_aggregates(aggregate_metrics: Mapping[str, float]) -> dict[str, float]:
    """Run-level values computed from pooled aggregates, not from page values.

    ``micro_..._sqrt_area_f``: the attribution F of the pooled headline H and
    coverage C. A mean of page Fs would let pages with little content swing it.
    ``macro_..._kind_f1``: the mean of the pooled per-class kind F1s.
    """
    derived: dict[str, float] = {}
    h = aggregate_metrics.get("micro_" + PREFIX + "sqrt_area_quality")
    c = aggregate_metrics.get("micro_" + PREFIX + "sqrt_area_coverage")
    if h is not None and c is not None:
        derived["micro_" + PREFIX + "sqrt_area_f"] = 2 * h * c / (h + c) if h + c > 0 else 0.0
    # Macro-F1 of the pooled per-class kind F1s.
    class_f1 = [v for k in KIND_CLASSES if (v := aggregate_metrics.get(f"micro_{PREFIX}kind_{k}_f1")) is not None]
    if class_f1:
        derived["macro_" + PREFIX + "kind_f1"] = sum(class_f1) / len(class_f1)
    return derived


def headline_rows(aggregate_metrics: Mapping[str, float]) -> list[tuple[str, float]]:
    """Headline rows present in an evaluation summary; empty without the rule."""
    if "avg_" + PREFIX + "generated_elements" not in aggregate_metrics:
        return []
    values = {**derived_aggregates(aggregate_metrics), **aggregate_metrics}
    return [(label, values[key]) for key, label in HEADLINE_ROWS if key in values]
