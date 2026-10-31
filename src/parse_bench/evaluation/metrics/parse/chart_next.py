"""Chart data metric (``chart_next``): unit-aware chart data matching with an error budget.

Every table on the page is a candidate; a content adapter also reads JSON blocks, ``- Category: value`` bullet
lists and one-table-per-series layouts when the page holds no parseable table.

Two reference modes:

* **Full tables** (metadata ``chart_tables``): each chart is a list of categories and series with string values
  (``""`` = unknown, skipped). Every output grid is read several ways (categories down the rows or across the
  header, the top header level melted into the rows, a long-format label column read as the series), with label
  chains: a row's label cells plus the section row above it, a column's header levels. Labels are normalized
  (``1Q23`` = ``Q1 2023``, ``FY2011`` = ``2011``, ``Jan-23`` = ``January 2023``, footnote markers, unit and axis
  parentheticals dropped), a label binds a chain element, the joined chain, or all of the chain's words; typos
  within OSA similarity 0.8 bind as minor errors. Rows and columns are assigned to categories and series with
  the Hungarian method on label binding, value agreement and a panel bonus (a chain element naming the chart:
  merged small multiples). Reworded labels are detected from their values only when the names are related
  (shared word, prefix, acronym) or nearly every value agrees; colour or marker names are a lost legend. Values
  are converted from cell suffixes, header units, series-name units and notes (a stated unit that contradicts
  the reference is a wrong value; percentage points under a ``%`` reference agree; prose and notes only supply
  scales). Tables are read as browsers render them: a rowspan stops at the end of its ``<thead>`` / ``<tbody>``
  and a ``colspan`` copy of a value is one cell. Bare day numbers on a daily axis take the month printed above
  them (``Jan. 2``, ``5``, ``6``). A range (``40–45``) that contains the
  reference value (with tolerance) scores as within tolerance and costs one minor "range instead of value" per
  chart; a range that misses it is scored by its midpoint. ``16`` for ``2016`` binds as a truncated category
  (major, per category); ``'16`` and ``’16`` are exact.
  Charts are placed greedily on the grid that fits best; a grid can hold several charts (each chart's cells are
  excluded for the others), and cells of a grid that belong to no matched row or column are never charged
  unless the grid otherwise fits the chart well (another table on the page is not this chart's precision). For
  a single-series chart, each extra value column of its grid (a duplicate, another series) is invented data (one
  major per column) unless it belongs to another reference chart, is a derived total or is the same series in
  another unit. Extra categories of an unlabeled line, area or scatter chart are free when they fall between
  the reference categories (denser sampling) and invented otherwise.
  A lost legend (output series named by colour, marker or placeholder, or unnamed) binds the reference series
  whose values it carries (60% of the points) and costs one "legend lost" per chart: critical with two or more
  series (the values cannot be attributed), major for a single series under a colour or ``Series 1`` name. An
  unrelated real name binds only when nearly every value agrees ("series renamed", major). When no series of a
  chart binds by name (lines named after their on-chart callouts), named columns (letters, not a category) bind
  the series whose values they clearly follow (mean value score >= 0.5, ahead of every other pairing by 0.25), only
  when every series of the chart binds this way ("series renamed", major, per series).
* **Sparse rules** (``chart_data_point``): each rule is one point with its labels. Labels bind on per-cell
  evidence (row cells, column-header chain, the table's context lines), rules are assigned to cells with the
  Hungarian method, and there is no contradiction penalty (sparse labels are not a full key).

Value tolerance (full mode): labeled charts 1% relative (an output rounded to fewer printed decimals is exact);
unlabeled charts ``max(1% * |t|, 2% of the chart's value range)``, with 0 in the range when all values share a
sign (bar baseline). Up to 3x the tolerance is a "value off" (major), beyond it a "wrong value" (critical).

Headline: ``min(proportional, floor)``. The proportional score is the point F1 (full mode) or point recall
(sparse mode) weighted by label binding and value closeness; the floor subtracts a fixed budget per error:
critical 0.10, major 0.05, minor 0.01 (minor total capped at 0.15). Sparse mode charges every occurrence.
Full mode: the first event of each severity on the page costs its full weight (one wrong value always costs
at least 10 points); later per-cell events of that severity (wrong value, value off, point missing, label typo,
repeated point, range instead of value, reworded or truncated category) cost weight x ``min(1, 10 / cells on
the page)`` (``FLOOR_SCOPE = "chart"`` scales by the chart's cells instead), so a page's floor does not depend
on how a figure is split into reference charts. Structural events are charged once and unscaled: chart or
series missing, legend lost with several series (critical); category missing, invented series or category,
wrong or missing title, derived total, single-series legend lost, series renamed (major); reworded series,
category axis name (minor). Omissions (missing points, categories, series) never cost a chart more than a
missing chart (0.10).

Regexes follow JavaScript semantics: ``\\s`` is the ECMAScript white-space set and
``\\d``/``\\b`` are ASCII.
"""

from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from parse_bench.schemas.evaluation import MetricValue

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FLOOR = {"critical": 0.10, "major": 0.05, "minor": 0.01, "minor_cap": 0.15}
DEFAULT_RULE_TOL = 0.01  # chart_data_point default relative tolerance
LABELED_TOL = 0.01  # full tables: relative tolerance when data labels are printed
UNLABELED_TOL = 0.01  # ... and when values are read from the axis (relative part; RANGE_TOL usually dominates)
RANGE_TOL = 0.02  # unlabeled charts: absolute tolerance as a share of the chart's value range (bar baseline 0 included)
OFF_FACTOR = 3.0  # an error up to 3x the tolerance is "value off" (major), beyond it a "wrong value" (critical)
CELL_EVENT_REF = 10  # floors charge per-cell events as if the chart had at most this many cells
FLOOR_SCOPE = "page"  # per-cell scaling counts the cells of the chart ("chart") or of the whole page ("page")
CONTINUOUS_TYPES = ("line", "area", "scatter")  # unlabeled: extra x positions are a denser sampling, not invented
SCALES = (100, 0.01, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)  # undeclared rescalings tried by the lenient lane
CTX_LINES = 4  # sparse mode: non-bullet context lines kept before a table
STOP = frozenset("of the and in for to a an by on at or per vs".split())

# ---------------------------------------------------------------------------
# JavaScript regex compatibility
# ---------------------------------------------------------------------------

_WS = "\t\n\x0b\x0c\r \xa0  -     　﻿"
_S = f"[{_WS}]"
_TRIM = " \t\n\x0b\x0c\r\xa0     　﻿" + "".join(map(chr, range(0x2000, 0x200B)))
_DOT = "[^\n\r  ]"  # JS `.`
_A = re.ASCII
_I = re.IGNORECASE


def _js_trim(s: str) -> str:
    return s.strip(_TRIM)


def _js_str(v: Any) -> str:
    """``String(v)`` for JSON-like values."""
    if isinstance(v, str):
        return v
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if v.is_integer() and abs(v) < 1e21:
            return str(int(v))
        return repr(v)
    if isinstance(v, list):
        return ",".join("" if x is None else _js_str(x) for x in v)
    if isinstance(v, dict):
        return "[object Object]"
    return str(v)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _has_digit(s: str) -> bool:
    return bool(_DIGIT.search(s))


_DIGIT = re.compile("[0-9]")

# ---------------------------------------------------------------------------
# Normalization and number reading
# ---------------------------------------------------------------------------

_COMBINING = re.compile("[̀-ͯ]")
_BR = re.compile(rf"<br{_S}*/?>")
_SUP = re.compile(r"<sup>.*?</sup>", re.DOTALL)
_EMPH = re.compile(r"\*\*|__|\*")
_WS_RUN = re.compile(f"{_S}+")


def pb_norm(s: Any) -> str:
    """Label normalization: no accents, lower case, no emphasis, single spaces."""
    t = _COMBINING.sub("", unicodedata.normalize("NFD", str(s))).lower()
    t = _SUP.sub("", _BR.sub(" ", t))
    t = _EMPH.sub("", t).replace("_", " ")
    return _js_trim(_WS_RUN.sub(" ", t))


MULT = {"trillion": 1e12, "trn": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "g": 1e9}
MULT |= {"million": 1e6, "mln": 1e6, "m": 1e6, "k": 1e3}
_NUM = re.compile(r"([-+]?[0-9]*\.?[0-9]+)(trillion|trn|t|billion|bn|b|g|million|mln|m|k)?\Z", _I)
_PAREN_ALL = re.compile(f"\\({_DOT}*\\)\\Z")
_COMMA_DEC = re.compile(",(?! )")


def num_cands(raw: Any) -> list[float]:
    """Numeric readings of a cell: thousands separators, decimal comma, accounting negatives, k/m/bn suffixes."""
    s = pb_norm(raw)
    if s[:1] in ("~", "≈"):
        s = s[1:]
    s = _js_trim(re.sub("[$€£¥]", "", s.replace("−", "-")))
    neg = False
    if s.startswith("(") and _PAREN_ALL.match(s):
        # "(5)" is an accounting -5; "(-11)" carries its own sign, so its parentheses only wrap it
        inner = _js_trim(s[1:-1])
        neg, s = inner[:1] not in ("-", "+"), inner
    if s.endswith("-"):
        neg, s = not neg, s[:-1]
    s = _js_trim(s.replace("%", ""))
    variants = [_WS_RUN.sub("", s.replace(",", ""))]
    if _COMMA_DEC.search(s):
        variants.append(_WS_RUN.sub("", s.replace(".", "").replace(",", ".")))
    out = []
    for v in variants:
        m = _NUM.match(v)
        if m:
            x = float(m.group(1)) * (MULT[m.group(2).lower()] if m.group(2) else 1)
            out.append(-x if neg else x)
    return out


_PP = re.compile("percentage points?")
_PCT = re.compile("%|percent")
_FRAC = re.compile(r"\bshare\b|fraction|proportion", _A)
# currency codes as words, optionally with a scale suffix ("EURm", "USDbn"): "eur" in "Europe" or "Neuro" is none
_CUR = re.compile(r"(?<![a-z])(?:usd|eur(?:os?)?)(?:m|mn|bn|b|k)?(?![a-z])|\$|dollar|€|£", _A)
_BN = re.compile(r"\bbn\b|billion", _A)
_MN = re.compile(r"million|\bmn\b|\bm\b", _A)
_TH = re.compile(r"thousand|\bk\b|\('000\)", _A)


def unit_desc(t: Any) -> dict[str, Any] | None:
    """Unit stated in a header, note or reference: ``kind`` (pct, pp, frac, cur) and/or ``scale``."""
    t = str(t or "").lower()
    if not _js_trim(t):
        return None
    d: dict[str, Any] = {}
    if _PP.search(t):
        d["kind"] = "pp"
    elif _PCT.search(t):
        d["kind"] = "pct"
    elif _FRAC.search(t):
        d["kind"] = "frac"
    elif _CUR.search(t):
        d["kind"] = "cur"
    sc = 1e9 if _BN.search(t) else 1e6 if _MN.search(t) else 1e3 if _TH.search(t) else None
    if sc:
        d["scale"] = sc
    return d or None


_PAREN = re.compile(f"\\(({_DOT}*?)\\)")
_Q1 = re.compile(rf"q([1-4]){_S}*'?([0-9]{{2}}|[0-9]{{4}})\Z")
_Q2 = re.compile(rf"([1-4])q{_S}*'?([0-9]{{2}}|[0-9]{{4}})\Z")
_Q3 = re.compile(rf"([0-9]{{4}}){_S}*q([1-4])\Z")
_YY = re.compile(r"'([0-9]{2})\Z")


def nx_label(x: Any) -> str:
    """Label key: NFKC, lower case, unit parentheticals dropped, quarters as ``2023q1``, ``'23`` as 2023."""
    t = unicodedata.normalize("NFKC", str(x)).lower()
    t = _PAREN.sub(lambda m: "" if unit_desc(m.group(1)) else m.group(0), t)
    t = _js_trim(_WS_RUN.sub(" ", t.replace("**", "")))
    m = _Q1.match(t) or _Q2.match(t)
    if m:
        t = f"{'20' + m.group(2) if len(m.group(2)) == 2 else m.group(2)}q{m.group(1)}"
    elif m := _Q3.match(t):
        t = f"{m.group(1)}q{m.group(2)}"
    elif m := _YY.match(t):
        t = "20" + m.group(1)
    return t[:-1] if t.endswith("s") else t


def osa(a: str, b: str) -> int:
    """Edit distance counting a swap of neighbours as one edit."""
    d = [[i] + [0] * len(b) for i in range(len(a) + 1)]
    d[0] = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


_DIGITS = re.compile("[0-9]+")


def nx_label_sim(a: str, b: str) -> float:
    """Typo tolerance: 1 - OSA / length when >= 0.8 and the digit runs agree, else 0."""
    if a == b:
        return 1.0
    if _DIGITS.findall(a) != _DIGITS.findall(b):
        return 0.0
    r = 1 - osa(a, b) / max(len(a), len(b), 1)
    return r if r >= 0.8 else 0.0


def hungarian(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Minimum-cost assignment (rectangular), O(n^2 m), with deterministic tie-breaking."""
    if not cost or not cost[0]:
        return []
    n, m = len(cost), len(cost[0])
    tr = n > m
    a = [list(col) for col in zip(*cost, strict=True)] if tr else cost
    rows, cols = len(a), len(a[0])
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        mn = [math.inf] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0, d, j1 = p[j0], math.inf, 0
            ai = a[i0 - 1]
            for j in range(1, cols + 1):
                if not used[j]:
                    c = ai[j - 1] - u[i0] - v[j]
                    if c < mn[j]:
                        mn[j], way[j] = c, j0
                    if mn[j] < d:
                        d, j1 = mn[j], j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += d
                    v[j] -= d
                else:
                    mn[j] -= d
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if not j0:
                break
    return [((j - 1, p[j] - 1) if tr else (p[j] - 1, j - 1)) for j in range(1, cols + 1) if p[j]]


# ---------------------------------------------------------------------------
# Label binding (label vs one evidence string)
# ---------------------------------------------------------------------------

_TOK = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")
_NOT_ALNUM = re.compile("[^a-z0-9]")
_PARENS = re.compile("[()]")
_PAREN_LAZY = re.compile(f"\\({_DOT}*?\\)")
_TWO_DIGITS = re.compile(r"[0-9]{2}\Z")
_YEAR = re.compile(r"(19|20)[0-9]{2}\Z")
_LETTER = re.compile("[a-z]")


def _toks(s: str) -> list[str]:
    return _TOK.findall(s)


def _alnum(s: str) -> str:
    return _NOT_ALNUM.sub("", s)


def _sub(a: Sequence[str], b: Sequence[str]) -> bool:
    """``a`` is a contiguous run inside ``b``."""
    if not a or len(a) > len(b):
        return False
    return any(list(b[i : i + len(a)]) == list(a) for i in range(len(b) - len(a) + 1))


def _bind_forms(a: str, b: str) -> float:
    if a == b or (_alnum(a) and _alnum(a) == _alnum(b)):
        return 1.0
    if _TWO_DIGITS.match(a) and _YEAR.match(b) and b.endswith(a):
        return 1.0
    ta, tb = _toks(a), _toks(b)
    if ta and _sub(ta, tb) and (_LETTER.search(a) or len(ta) > 1 or _YEAR.match(a)):
        return 0.99  # the label is a token run inside a longer cell: binds, but is not an exact key
    if tb and len(_alnum(b)) >= 3 and _LETTER.search(b) and _sub(tb, ta):
        return 0.9  # the evidence is a truncation of the label
    return nx_label_sim(a, b)


def _strip_parens(s: str) -> str:
    return _js_trim(_WS_RUN.sub(" ", _PARENS.sub(" ", s)))


@lru_cache(maxsize=200_000)
def bind(label: str, ev: str) -> float:
    """Label -> evidence similarity: 1 exact, 0.99 token-contained, 0.9 truncated, OSA >= 0.8 typo, else 0.

    Both the normalized form (unit parentheticals dropped) and the raw form are compared, so
    ``SIDS (Yes %)`` still binds ``Yes``.
    """
    l0, e0 = pb_norm(label), pb_norm(ev)
    if not l0 or not e0:
        return 0.0
    return max(
        _bind_forms(nx_label(l0) or l0, nx_label(e0) or e0),
        _bind_forms(_strip_parens(l0), _strip_parens(e0)),
    )


def _content_toks(label: str, drop_parens: bool = True) -> list[str]:
    t = pb_norm(label)
    if drop_parens:
        t = _PAREN_LAZY.sub(" ", t)
    return [x for x in _toks(t) if x not in STOP]


def ctx_bind(label: str, line: str) -> bool:
    """Title/caption binding: the label's content words all occur in one context line (order-free)."""
    lt = _content_toks(label)
    nl = pb_norm(line)
    xt = set(_toks(nl))
    if lt and all(t in xt for t in lt):
        return True
    al = _alnum("".join(lt))
    return len(al) >= 4 and al in _alnum(nl)  # "FY14 Q2 Forecast" ~ "FY 14 Q2 Forecast Total"


# ---------------------------------------------------------------------------
# Cell readings
# ---------------------------------------------------------------------------

_RANGE = re.compile(rf"{_S}*[~≈]?\(?{_S}*(-?[0-9.,]+){_S}*%?{_S}*(?:–|-|to){_S}*(-?[0-9.,]+){_S}*\)?{_S}*(%?){_S}*\Z")
_SUFFIX = re.compile(rf"{_S}*(trillion|billion|million|thousand|trn|bn|mn|mln|k|m|b|t)\.?\Z", _I)
_TAG = re.compile("<[^>]+>")
_NUM_TOKEN = re.compile(rf"[-+−]?\$?[0-9](?:[0-9,]*[0-9])?(?:\.[0-9]+)?{_S}*%?")
_RULE_NUM = re.compile(rf"[-+−]?\$?[0-9][0-9,]*(?:\.[0-9]+)?{_S}*%?")


@dataclass
class Reading:
    vals: list[float]
    range: bool = False
    pct: bool = False
    compound: bool = False


def interps(raw: str) -> Reading:
    """Strict "as written" readings of one cell; a range reads as its midpoint."""
    s = _js_trim(str(raw))
    rng = _RANGE.match(s)
    if rng:
        a, b = num_cands(rng.group(1)), num_cands(rng.group(2))
        if a and b:
            return Reading([(a[0] + b[0]) / 2], range=True, pct=bool(rng.group(3)))
    out = dict.fromkeys(num_cands(s))
    out |= dict.fromkeys(num_cands(_SUFFIX.sub("", s, count=1)))
    toks = _NUM_TOKEN.findall(_TAG.sub(" ", s))
    compound = False
    if not out and len(toks) == 1:
        out |= dict.fromkeys(num_cands(toks[0]))
    elif not out and len(toks) > 1:
        compound = True
        for x in toks:
            out |= dict.fromkeys(num_cands(x))
    return Reading(list(out), pct="%" in s, compound=compound)


def vscore(t: float, p: float, tol: float, abs_tol: float = 0.0) -> float:
    """Value closeness: 1 within tolerance (relative to max(|t|, |p|)), linear to 0 at 3x."""
    if t == 0:
        allowed = max(tol, abs_tol, 1e-9)
    else:
        allowed = max(tol * max(abs(t), abs(p)), abs_tol, 1e-9)
    e = abs(p - t)
    if _sign(p) != _sign(t) and abs(t) > allowed and abs(p) > allowed:
        return 0.0
    return 1.0 if e <= allowed else max(0.0, 1 - (e - allowed) / (2 * allowed))


def rule_number(value: str) -> float | None:
    """Reference number of a rule value; a compound ``76 (39.4%)`` reads as its first number."""
    if not _has_digit(value):
        return None
    c = num_cands(value)
    if c:
        return c[0]
    m = _RULE_NUM.search(value)
    c = num_cands(m.group(0) if m else "")
    return c[0] if c else None


# ---------------------------------------------------------------------------
# Content adapter: tables, JSON, bullet lists, one-table-per-series
# ---------------------------------------------------------------------------

_HTML_TABLE = re.compile(r"<table.*?</table>", _I | re.DOTALL)
_HTML_ROW = re.compile(r"<tr.*?</tr>", _I | re.DOTALL)
_HTML_CELL = re.compile(r"<(t[hd])[^>]*>(.*?)</t[hd]>", _I | re.DOTALL)
_BR_I = re.compile(rf"<br{_S}*/?>", _I)
_SEP_ROW = re.compile(rf"{_S}*\|?{_S}*:?-{{3,}}[-|: ]*\Z")
_HEADING = re.compile(rf"{_S}*#")
_BOLD_LINE = re.compile(rf"{_S}*\*\*{_DOT}+\*\*{_S}*\Z")
_HASHES = re.compile(rf"{_S}*#+{_S}*")


@dataclass
class _LabTable:
    rows: list[list[str]]
    header_rows: int
    context: list[str]
    pos: int


def extract_tables(md: str) -> list[_LabTable]:
    """Markdown pipe tables and HTML tables with the few lines before each."""
    tables: list[_LabTable] = []
    lines = md.split("\n")
    for m in _HTML_TABLE.finditer(md):
        rows, th = [], []
        for r in _HTML_ROW.finditer(m.group(0)):
            cells, ths = [], []
            for c in _HTML_CELL.finditer(r.group(0)):
                cells.append(_js_trim(_TAG.sub("", _BR_I.sub(" ", c.group(2)))))
                ths.append(c.group(1).lower() == "th")
            rows.append(cells)
            th.append(ths)
        before = [ln for ln in md[: m.start()].split("\n") if _js_trim(ln)][-3:]
        hdr = 0
        while hdr < len(rows) and th[hdr] and all(th[hdr]):
            hdr += 1
        tables.append(_LabTable(rows, hdr, before, m.start()))
    stripped = _HTML_TABLE.sub(lambda m: re.sub("[^\n]", " ", m.group(0)), md).split("\n")
    offs, off = [], 0
    for ln in stripped:
        offs.append(off)
        off += len(ln) + 1
    i = 0
    while i < len(stripped):
        if "|" not in stripped[i]:
            i += 1
            continue
        j = i
        while j < len(stripped) and "|" in stripped[j]:
            j += 1
        if j - i >= 2:
            rows = []
            for ln in stripped[i:j]:
                if _SEP_ROW.match(ln):
                    continue
                t = _js_trim(ln)
                t = t[1:] if t.startswith("|") else t
                t = t[:-1] if t.endswith("|") else t
                rows.append([_js_trim(x) for x in t.split("|")])
            ctx = [ln for ln in lines[max(0, i - 5) : i] if _js_trim(ln)]
            tables.append(_LabTable(rows, 1, ctx, offs[i]))
        i = j
    return sorted(tables, key=lambda t: t.pos)


def heading_of(context: Sequence[str]) -> str | None:
    for ln in reversed(context):
        if _HEADING.match(ln) or _BOLD_LINE.match(ln):
            return _js_trim(_HASHES.sub("", ln, count=1).replace("**", ""))
    return None


_FENCE = re.compile(r"```(?:json)?" + _S + r"*(.*?)```", re.DOTALL)
_BULLET = re.compile(rf"{_S}*[-*]{_S}+[^:|\n]+:{_S}")
_BULLET_PARTS = re.compile(rf"{_S}*[-*]{_S}+([^:]+):{_S}*({_DOT}*)\Z")
_COMMA_SPLIT = re.compile(f",{_S}*")
_NAME_VAL = re.compile(rf"({_DOT}*?){_S}+([^{_WS}]+)\Z")
_INDEX_KEY = re.compile(r"(0|[1-9][0-9]*)\Z")


def _js_keys(o: Any) -> list[str]:
    """``Object.keys``: integer-like keys first in ascending order, then insertion order."""
    if isinstance(o, dict):
        keys = [str(k) for k in o]
    elif isinstance(o, list | str):
        return [str(i) for i in range(len(o))]
    else:
        return []
    idx = sorted((k for k in keys if _INDEX_KEY.match(k) and int(k) < 2**32 - 1), key=int)
    return idx + [k for k in keys if k not in set(idx)]


def _js_get(o: Any, k: str) -> Any:
    if isinstance(o, dict):
        return o.get(k, _MISSING)
    if isinstance(o, list | str):
        return o[int(k)] if _INDEX_KEY.match(k) and int(k) < len(o) else _MISSING
    return _MISSING


_MISSING = object()


def _json_grid(src: str) -> list[list[str]] | None:
    try:
        j = json.loads(src)
    except (ValueError, RecursionError):
        return None
    if isinstance(j, list):
        arr = j
    elif isinstance(j, dict):
        arr = next((x for x in j.values() if isinstance(x, list)), [])
    else:
        arr = None
    if not arr or not (arr[0] is None or isinstance(arr[0], dict | list)):
        return None
    if any(o is None for o in arr):
        return None  # a null item has no keys: not a table
    keys = list(dict.fromkeys(k for o in arr for k in _js_keys(o)))
    grid = [keys]
    for o in arr:
        row = []
        for k in keys:
            v = _js_get(o, k)
            row.append("" if v is _MISSING else _js_str(v))
        grid.append(row)
    return grid


class _Pairs(list):
    """A JSON object kept as its ``(key, value)`` pairs, so repeated keys ("15yr" twice) all survive."""


_JSON_OBJ_LINE = re.compile(r"^[ \t]*\{", re.M)


def _scalar_map(o: Any) -> bool:
    """A non-empty JSON object of scalars, at least one holding a digit: one series, ``{category: value}``."""
    return (
        isinstance(o, _Pairs)
        and bool(o)
        and not any(isinstance(x, list) for _, x in o)
        and any(_has_digit(_js_str(x)) for _, x in o)
    )


def _series_grids(o: _Pairs, path: tuple[str, ...] = ()) -> list[tuple[list[list[str]], list[str]]]:
    """The series objects inside one JSON object, each as a grid (categories down the rows, one column per series)
    with the names it sits under (panel keys, ``title`` members). A series object is ``{series: {category: value}}``,
    or ``{category: value}`` below the top level (named by its key). A repeated category key pairs up by occurrence."""
    titles = [x for k, x in o if k == "title" and isinstance(x, str)][:1]
    names = [*path, *titles]
    if len(o) and all(_scalar_map(s) for _, s in o):
        series = list(o)
    elif path and _scalar_map(o):
        series, names = [(path[-1], o)], [*path[:-1], *titles]
    else:
        subs = [(k, x) for k, x in o if isinstance(x, _Pairs)]
        subs += [
            (k, y)
            for k, x in o
            if isinstance(x, list) and not isinstance(x, _Pairs)
            for y in x
            if isinstance(y, _Pairs)
        ]
        return [g for k, x in subs for g in _series_grids(x, (*names, k))]
    cats: dict[tuple[str, int], dict[int, str]] = {}
    for j, (_, s) in enumerate(series):
        seen: dict[str, int] = {}
        for k, x in s:
            seen[k] = seen.get(k, 0) + 1
            cats.setdefault((k, seen[k]), {})[j] = "" if x is None else _js_str(x)
    head = ["", *(str(name) for name, _ in series)]
    return [([head, *([k, *(row.get(j, "") for j in range(len(series)))] for (k, _), row in cats.items())], names)]


def _series_json_grids(md: str) -> list[tuple[list[list[str]], list[str], str | None]]:
    """Every series object in a JSON object starting a line (fenced or not), read as a grid, with its context (the
    5 lines above the JSON as for tables, the names it sits under, 3 lines below the JSON) and heading."""
    out: list[tuple[list[list[str]], list[str], str | None]] = []
    dec, end = json.JSONDecoder(object_pairs_hook=_Pairs), 0
    for m in _JSON_OBJ_LINE.finditer(md):
        if m.start() < end:
            continue  # inside an object already read
        try:
            o, end = dec.raw_decode(md, m.end() - 1)
        except (ValueError, RecursionError):
            continue
        found = _series_grids(o)
        if found:
            before = [_js_trim(x) for x in md[: m.start()].split("\n")[-6:] if _js_trim(x) and x.strip() != "```"]
            after = [_js_trim(x) for x in md[end:].split("\n") if _js_trim(x) and x.strip() != "```"][:3]
            out += [(g, [*before, *names, *after], heading_of(before)) for g, names in found]
    return out


def data_grids(md: str, include_tables: bool = True, series_json: bool = True) -> list[list[list[str]]]:
    """Every data representation on the page as a grid (header row first): JSON, bullets, per-series tables.
    ``series_json``: include the ``{series: {category: value}}`` objects (output_grids reads them with context)."""
    grids: list[list[list[str]]] = []
    fence = _FENCE.search(md)
    t = _js_trim(md)
    src = fence.group(1) if fence else (t if t.startswith(("{", "[")) else None)
    if src is not None:
        g = _json_grid(src)
        if g:
            grids.append(g)
    if series_json:
        grids += [g for g, _, _ in _series_json_grids(md)]
    bullets = [ln for ln in md.split("\n") if _BULLET.match(ln)]
    if len(bullets) >= 2:
        rows: list[list[Any]] = []
        for ln in bullets:
            m = _BULLET_PARTS.match(ln)
            if not m:
                continue
            parts: list[Any] = [_js_trim(m.group(1))]
            for x in _COMMA_SPLIT.split(m.group(2)):
                nv = _NAME_VAL.match(x)
                parts.append([_js_trim(nv.group(1)), nv.group(2)] if nv else [None, _js_trim(x)])
            rows.append(parts)
        if rows:
            names = [x[0] if x[0] is not None else f"Value {k + 1}" for k, x in enumerate(rows[0][1:])]
            grids.append([["", *names], *([r[0], *(x[1] for x in r[1:])] for r in rows)])
    tables = extract_tables(md)
    if len(tables) >= 2 and all(tb.rows and len(tb.rows[0]) == 2 for tb in tables):
        cats = list(dict.fromkeys(r[0] for tb in tables for r in tb.rows[1:]))
        names = [heading_of(tb.context) or tb.rows[0][1] for tb in tables]
        merged = [[tables[0].rows[0][0], *names]]
        for c in cats:
            vals = []
            for tb in tables:
                hit = next((r for r in tb.rows if r[0] == c), [])
                vals.append(hit[1] if len(hit) > 1 else "")
            merged.append([c, *vals])
        grids.append(merged)
    elif tables and include_tables:
        grids.append(tables[0].rows)
    return grids


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ChartNextEvents:
    critical: int = 0
    major: int = 0
    minor: int = 0
    examples: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, severity: str, kind: str, reference: str = "", output: str = "", n: int = 1) -> None:
        setattr(self, severity, getattr(self, severity) + n)
        self.examples.append((kind, reference, output))

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kind, _, _ in self.examples:
            out[kind] = out.get(kind, 0) + 1
        return out


def next_floor(ev: ChartNextEvents) -> float:
    return _clamp01(
        1
        - FLOOR["critical"] * ev.critical
        - FLOOR["major"] * ev.major
        - min(FLOOR["minor_cap"], FLOOR["minor"] * ev.minor)
    )


@dataclass
class ChartNextResult:
    score: float
    proportional: float
    floor: float
    mode: str  # "tables" (full chart_tables) or "rules" (sparse chart_data_point)
    recall: float  # binding-weighted value score per reference point
    precision: float | None  # ... per output point (tables mode)
    values: float | None  # mean value score over matched points
    labels: float | None  # mean label binding over matched points
    strict_pass: float  # share of reference points matched with an exact-tolerance value
    lenient: float | None  # recall when undeclared rescaling (x100, x1e3, ...) is forgiven (rules mode)
    title: float | None  # share of charts whose title is on the page (tables mode)
    scale_only: int  # wrong values that a rescaling would fix
    n_points: int
    n_cells: int
    n_charts: int
    charts_found: int
    events: ChartNextEvents
    points: list[dict[str, Any]] = field(default_factory=list)

    def metadata(self, max_examples: int = 20) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "proportional": self.proportional,
            "floor": self.floor,
            "recall": self.recall,
            "precision": self.precision,
            "values": self.values,
            "labels": self.labels,
            "strict_pass": self.strict_pass,
            "lenient": self.lenient,
            "title": self.title,
            "scale_only": self.scale_only,
            "n_points": self.n_points,
            "n_cells": self.n_cells,
            "n_charts": self.n_charts,
            "charts_found": self.charts_found,
            "events": {"critical": self.events.critical, "major": self.events.major, "minor": self.events.minor},
            "event_kinds": self.events.kinds(),
            "event_examples": [
                {"kind": k, "reference": r[:160], "output": h[:160]} for k, r, h in self.events.examples[:max_examples]
            ],
        }


# ---------------------------------------------------------------------------
# Sparse rules mode
# ---------------------------------------------------------------------------


@dataclass
class _Cell:
    ti: int
    r: int
    c: int
    t: str
    row_ev: list[str]
    col_ev: list[str]
    ctx_lines: list[str]
    unit: dict[str, Any] | None
    reading: Reading

    @property
    def ev(self) -> list[str]:
        return self.row_ev + self.col_ev


@dataclass
class _Table:
    ctx: str
    cap: str
    rowlabels: list[str]
    header: list[str]
    cells: list[dict[str, Any]]
    row0: list[str] | None = None
    has_th: bool = False


def rule_tables(md: str) -> list[_Table]:
    """Every value cell of every output table with its label scope (row vs column-header evidence)."""
    import numpy as np

    from parse_bench.evaluation.metrics.parse.rules_chart import ChartDataPointRule, parse_chart_tables
    from parse_bench.evaluation.metrics.parse.utils import normalize_text

    probe = ChartDataPointRule({"type": "chart_data_point", "labels": ["x"], "value": "1"})
    try:
        parsed = parse_chart_tables(md)
    except Exception:  # noqa: BLE001 - a malformed table means no table
        parsed = []
    out: list[_Table] = []
    for td in parsed:
        a = td.data
        if not isinstance(a, np.ndarray) or a.ndim != 2 or 0 in a.shape:
            continue
        rows, cols = a.shape
        cells = []
        for r in range(rows):
            for c in range(cols):
                txt = str(a[r, c]).strip()
                if (
                    not re.search(r"\d", txt)
                    or (r, c) in td.header_cells
                    or (r in td.header_rows and not td.header_cells)
                ):
                    continue
                try:
                    scope = probe._candidate_data_scope(a, r, c, td)
                except Exception:  # noqa: BLE001
                    scope = []
                rowev = {normalize_text(str(a[r, k])) for k in range(cols) if k != c}
                cells.append(
                    {
                        "r": r,
                        "c": c,
                        "t": txt,
                        "row": [s for s in scope if s in rowev],
                        "col": [s for s in scope if s not in rowev],
                    }
                )
        out.append(
            _Table(
                ctx=td.context_before[-600:],
                cap=td.caption or "",
                rowlabels=[normalize_text(str(a[r, 0])) for r in range(rows)],
                header=[normalize_text(str(a[h, k])) for h in sorted(td.header_rows) for k in range(cols)][:200],
                cells=cells,
                row0=[normalize_text(str(a[0, k])) for k in range(cols)],
                has_th=bool(td.header_cells or td.header_rows),
            )
        )
    return out


def _fallback_tables(md: str) -> list[_Table]:
    """No parseable table: read JSON / bullet / per-series grids instead."""
    out = []
    for g in data_grids(md):
        if not g or len(g) < 2:
            continue
        cells = []
        for i, row in enumerate(g[1:]):
            for c, x in enumerate(row):
                if c == 0 or not _has_digit(str(x)):
                    continue
                rowev = [str(y) for k, y in enumerate(row) if k != c and not _has_digit(str(y))] + [str(row[0])]
                head = g[0][c] if c < len(g[0]) and g[0][c] is not None else ""
                cells.append(
                    {"r": i + 1, "c": c, "t": str(x), "row": [pb_norm(y) for y in rowev], "col": [pb_norm(head)]}
                )
        out.append(
            _Table(
                ctx="",
                cap="",
                rowlabels=[pb_norm(str(r[0]) if r and r[0] is not None else "") for r in g],
                header=[pb_norm(str(x)) for x in g[0]],
                cells=cells,
            )
        )
    return out


_BULLET_LINE = re.compile(rf"[-*]{_S}")
_NOTE_LINE = re.compile(r"\b(in|units?|values?|scale)\b|\(.*\)", _I | _A)
_UNIT_PAREN = re.compile(f"\\(({_DOT}*?)\\)")
_TITLE_LINE = re.compile(r"#|<caption|<h[1-6]", _I)


def _col_unit_text(h: str) -> str:
    m = _UNIT_PAREN.search(h)
    if m and m.group(1):
        return m.group(1)
    return "%" if _PCT.search(h) else ""


def _rule_cells(tables: list[_Table]) -> list[_Cell]:
    cells: list[_Cell] = []
    for ti, tb in enumerate(tables):
        ctx_all = [_js_trim(x) for x in (tb.ctx + "\n" + (tb.cap or "")).split("\n")]
        ctx_lines = [x for x in ctx_all if x and not _BULLET_LINE.match(x)][-CTX_LINES:]  # legend bullets stay out
        note_unit = unit_desc(" ".join(ln for ln in ctx_lines if _NOTE_LINE.search(ln)))
        for c in tb.cells:
            col = list(c["col"])
            # all-<td> table: the first row is the column header
            if not col and tb.row0 is not None and c["r"] > 0 and c["c"] < len(tb.row0) and tb.row0[c["c"]]:
                if not tb.has_th:
                    col = [tb.row0[c["c"]]]
                    c["col"] = col
            rl = tb.rowlabels[c["r"]] if c["r"] < len(tb.rowlabels) else ""
            col_unit = unit_desc(" ".join(_col_unit_text(h) for h in [*col, rl or ""]))
            cells.append(
                _Cell(
                    ti, c["r"], c["c"], c["t"], list(c["row"]), col, ctx_lines, col_unit or note_unit, interps(c["t"])
                )
            )
    return cells


@dataclass
class ChartRule:
    labels: list[str]
    value: str
    tol: float = DEFAULT_RULE_TOL
    estimate: bool = False  # tagged ``need_estimate``: the value is read off the axis, not printed

    @property
    def num(self) -> float | None:
        return rule_number(self.value)


def chart_rules(rules: Iterable[Any]) -> list[ChartRule]:
    """``chart_data_point`` rules (dicts or rule models) as ``ChartRule``."""
    out = []
    for r in rules or ():
        get = r.get if isinstance(r, Mapping) else (lambda k, _r=r: getattr(_r, k, None))
        if get("type") != "chart_data_point":
            continue
        tol = get("relative_tolerance")
        out.append(
            ChartRule(
                labels=[_js_str(x) for x in (get("labels") or [])],
                value=_js_str(get("value")),
                tol=DEFAULT_RULE_TOL if tol is None else float(tol),
                estimate="need_estimate" in (get("tags") or ()),
            )
        )
    return out


def score_chart_rules(md: str, rules: Sequence[ChartRule], tables: list[_Table] | None = None) -> ChartNextResult:
    """Sparse mode: score ``chart_data_point`` rules against every table on the page."""
    tables = rule_tables(md) if tables is None else tables
    if not tables:
        tables = _fallback_tables(md)
    cells = _rule_cells(tables)
    ev = ChartNextEvents()
    nums = [r.num for r in rules]
    table_ev = []
    for ti, tb in enumerate(tables):
        table_ev.append(
            list(dict.fromkeys([*tb.rowlabels, *tb.header, *(x for c in cells if c.ti == ti for x in c.ev)]))
        )
    table_tok = [{t for x in e for t in _toks(pb_norm(x))} for e in table_ev]
    in_table: dict[tuple[str, int], bool] = {}

    def label_in_table(lab: str, ti: int) -> bool:
        key = (lab, ti)
        if key not in in_table:
            in_table[key] = any(bind(lab, x) > 0 for x in table_ev[ti])
        return in_table[key]

    n_r, n_c = len(rules), len(cells)
    K = [[0.0] * n_c for _ in range(n_r)]
    V = [[0.0] * n_c for _ in range(n_r)]
    VL = [[0.0] * n_c for _ in range(n_r)]
    for i, rule in enumerate(rules):
        for j, c in enumerate(cells):
            k = 1.0
            for lab in rule.labels:
                # label_in_table is only consulted on the context-line fallback; evaluate lazily
                b = _label_binds_lazy(lab, c, table_tok[c.ti], lambda lab=lab, ti=c.ti: label_in_table(lab, ti))
                k = min(k, b)
                if not k:
                    break
            K[i][j] = k
            if not k:
                continue
            num = nums[i]
            if num is None:
                V[i][j] = VL[i][j] = 1.0 if pb_norm(c.t) == pb_norm(rule.value) else 0.0
                continue
            u = c.unit or {}
            vals = c.reading.vals
            cands = list(vals)
            if u.get("scale"):
                cands += [v * u["scale"] for v in vals] + [v / u["scale"] for v in vals]
            if u.get("kind") == "frac":
                cands += [v * 100 for v in vals] + [v / 100 for v in vals]
            V[i][j] = max([0.0, *(vscore(num, p, rule.tol) for p in cands)])
            VL[i][j] = max([V[i][j], *(vscore(num, p * s, rule.tol) for p in vals for s in SCALES)])
    S = [[K[i][j] * V[i][j] for j in range(n_c)] for i in range(n_r)]
    pairs: list[tuple[int, int]] = []
    if n_r and n_c:
        cost = [[1 - (0.5 * K[i][j] + 0.5 * S[i][j]) if K[i][j] else 1.01 for j in range(n_c)] for i in range(n_r)]
        pairs = [(i, j) for i, j in hungarian(cost) if K[i][j] > 0]
    tm = dict(pairs)
    tot = tot_l = 0.0
    passed = scale_only = 0
    points: list[dict[str, Any]] = []
    vals_m, labs_m = [], []
    for i, rule in enumerate(rules):
        lab = " / ".join(rule.labels) + " = " + rule.value
        jm = tm.get(i)
        if jm is None:
            points.append({"rule": i, "K": 0.0, "V": 0.0, "cell": "", "unit": ""})
            ev.add("major", "missing", lab)
            continue
        j = jm
        c, v = cells[j], V[i][j]
        scale_hint = VL[i][j] >= 0.5 and v < 0.5
        points.append({"rule": i, "K": K[i][j], "V": v, "cell": c.t[:40], "scale": scale_hint, "table": c.ti})
        tot += S[i][j]
        tot_l += K[i][j] * VL[i][j]
        vals_m.append(v)
        labs_m.append(K[i][j])
        if v == 1:
            passed += 1
        if K[i][j] < 0.95:
            ev.add("minor", "label typo", lab, "|".join(c.row_ev + c.col_ev))
        if c.reading.range:
            ev.add("minor", "range", lab, c.t)
        if v < 0.5:
            scale_only += scale_hint
            ev.add("critical", "wrong value (scale)" if scale_hint else "wrong value", lab, c.t)
        elif v < 1:
            ev.add("major", "value off", lab, c.t)
    # a rule's row repeated inside one table (duplicated rows, not a long-format table)
    for tb in tables:
        body = [x for x in tb.rowlabels[1:] if x]
        seen: dict[str, int] = {}
        for x in body:
            seen[x] = seen.get(x, 0) + 1
        if len(set(body)) / max(len(body), 1) < 0.8:
            continue
        for x, n in seen.items():
            if n <= 1:
                continue
            if any(bind(lab, x) == 1 for r in rules for lab in r.labels) and any(
                c["r"] < len(tb.rowlabels) and tb.rowlabels[c["r"]] == x for c in tb.cells
            ):
                ev.add("major", "repeated row", x, str(n))
    n = max(n_r, 1)
    recall = tot / n
    floor = next_floor(ev)
    return ChartNextResult(
        score=min(recall, floor),
        proportional=recall,
        floor=floor,
        mode="rules",
        recall=recall,
        precision=None,
        values=sum(vals_m) / len(vals_m) if vals_m else None,
        labels=sum(labs_m) / len(labs_m) if labs_m else None,
        strict_pass=passed / n,
        lenient=tot_l / n,
        title=None,
        scale_only=scale_only,
        n_points=n_r,
        n_cells=n_c,
        n_charts=0,
        charts_found=len(tables),
        events=ev,
        points=points,
    )


def _label_binds_lazy(label: str, c: _Cell, table_tok: set[str], in_table: Any) -> float:
    b = max([0.0, *(bind(label, x) for x in c.ev)])
    if b:
        return b
    lt = _content_toks(label, drop_parens=False)
    if len(lt) > 1:  # "Fixed 2023" over the header chain "Fixed broadband speed" > "2023"
        for chain in (c.col_ev, c.row_ev):
            xt = {t for x in chain for t in _toks(pb_norm(x))}
            if all(t in xt for t in lt):
                return 0.99
    lt = _content_toks(label)
    if len(lt) > 1:  # "Women: 2024" = panel heading "Women" + column "2024"
        ev_in = [et for et in (_toks(pb_norm(x)) for x in c.col_ev + c.row_ev) if et and all(t in lt for t in et)]
        if ev_in:
            ctx_t = {t for x in c.ctx_lines for t in _toks(pb_norm(x))}
            if all(any(t in et for et in ev_in) or (t in ctx_t and t not in table_tok) for t in lt):
                return 0.99
    # a title/caption line names the label, and the label is not a cell of the table (or the line is a heading)
    if any(ctx_bind(label, x) for x in c.ctx_lines) and (
        not in_table() or any(_TITLE_LINE.match(x) and ctx_bind(label, x) for x in c.ctx_lines)
    ):
        return 0.99
    return 0.0


# ---------------------------------------------------------------------------
# Full tables mode (several charts per page, every output grid)
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(rf"(series|data|value|column|col){_S}*[0-9]*\Z", _I)
_DERIVED = re.compile(r"(total|sum|average|avg|overall|all)\Z")
_DECIMAL_COMMA = re.compile(r"[(-]?\$?[0-9]+,[0-9]{1,2}\)?%?\Z")
_SUFFIXED = re.compile(r"[0-9]\s*(k|m|bn|b|t|million|billion|thousand)\b", _I | _A)
_CELL_RANGE = re.compile(rf"{_S}*\(?(-?[0-9.,]+){_S}*%?{_S}*[–-]{_S}*(-?[0-9.,]+)\)?{_S}*(%?){_S}*\Z")
_TITLE_PREFIX = re.compile(rf"(figure|fig\.?|chart|exhibit|graph|table){_S}*[0-9a-z]*[.:]?{_S}+", _I)

# per-cell events: their floor charge scales with the chart size (CELL_EVENT_REF cells = one unit)
CELL_EVENTS = frozenset(
    {
        "wrong value",
        "value off",
        "point missing",
        "label typo",
        "repeated point",
        "range instead of value",  # once per chart
        "category label truncated",  # once per category
        "category label reworded",  # one per category: a label-level error like a typo, so it scales too
    }
)
OMISSION_EVENTS = frozenset({"point missing", "category missing", "series missing"})


@dataclass
class ChartSpec:
    """One reference chart of ``chart_tables`` (values are strings; ``""`` = unknown, skipped)."""

    title: str = ""
    chart_type: str = ""
    x_label: str = ""
    y_label: str = ""
    unit: str = ""
    scale: Any = None
    data_labels_shown: bool = True
    categories: list[str] = field(default_factory=list)
    series: list[tuple[str, list[str]]] = field(default_factory=list)
    relative_tolerance: float | None = None
    optional: frozenset[int] = frozenset()  # series flagged ``optional`` (derived, e.g. the 100 - Yes of a Yes/No bar)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ChartSpec:
        series: list[tuple[str, list[str]]] = []
        optional: set[int] = set()
        for s in d.get("series") or ():
            vals = s.get("values") or []
            if s.get("optional"):
                optional.add(len(series))
            series.append((_js_str(s.get("name") or ""), ["" if v is None else _js_str(v) for v in vals]))
        tol = d.get("relative_tolerance")
        return cls(
            title=_js_str(d.get("title") or ""),
            chart_type=_js_str(d.get("chart_type") or ""),
            x_label=_js_str(d.get("x_label") or ""),
            y_label=_js_str(d.get("y_label") or ""),
            unit=_js_str(d.get("unit") or ""),
            scale=d.get("scale"),
            data_labels_shown=bool(d.get("data_labels_shown", True)),
            categories=[_js_str(c) for c in d.get("categories") or ()],
            series=series,
            relative_tolerance=None if tol is None else float(tol),
            optional=frozenset(optional),
        )

    @property
    def tol(self) -> float:
        if self.relative_tolerance is not None:
            return self.relative_tolerance
        return LABELED_TOL if self.data_labels_shown else UNLABELED_TOL

    @property
    def continuous(self) -> bool:
        """Values read from a continuous x axis: extra x positions in the output are a denser sampling."""
        return not self.data_labels_shown and any(k in self.chart_type.lower() for k in CONTINUOUS_TYPES)

    def unit_parts(self) -> list[dict[str, Any]]:
        """The units of a dual-axis chart (``% ; USD (2024)/bbl``, ``# of transactions; $ in bn``)."""
        return [u for u in (unit_desc(x) for x in _UNIT_SPLIT.split(self.unit)) if u]

    def ref_unit(self) -> dict[str, Any]:
        """Reference unit: ``kind`` pct when the values are percentages, ``scale`` the stated multiplier.
        A chart stating two different units (one per axis) has no chart-level unit: the series names decide."""
        parts = self.unit_parts()
        if len(parts) > 1 and any(p != parts[0] for p in parts):
            return {}
        u = dict(unit_desc(self.unit) or {})
        if isinstance(self.scale, int | float) and not isinstance(self.scale, bool) and self.scale > 0:
            u["scale"] = float(self.scale)
        elif self.scale:
            u.update({k: v for k, v in (unit_desc(self.scale) or {}).items() if k == "scale"})
        return u


_UNIT_SPLIT = re.compile(rf"{_S}*;{_S}*|{_S}+and{_S}+|{_S}+/{_S}+")


# --- label normalization (full mode only; sparse mode keeps ``bind``/``nx_label``) ---------------------------

_FOOTNOTE_CHARS = re.compile("[†‡§¶⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁽⁾]+")
_BRACKET_NOTE = re.compile(r"\[[a-z0-9]{1,3}\]")
_FY = re.compile(rf"\b(?:fy|cy){_S}*'?([0-9]{{4}}|[0-9]{{2}})\b", _A)
_YEAR_MARK = re.compile(r"\b((?:19|20)[0-9]{2})[efpa]\b", _A)
_MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_MON_NAME = "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\\.?"
_MON_Y = re.compile(rf"{_MON_NAME}[{_WS}\-/,']*'?([0-9]{{4}}|[0-9]{{2}})\Z")
_Y_MON = re.compile(rf"([0-9]{{4}})[{_WS}\-/,:]*{_MON_NAME}\Z")
_FQ1 = re.compile(rf"([qh])([1-4])[{_WS}\-/']*'?([0-9]{{4}}|[0-9]{{2}})\Z")
_FQ2 = re.compile(rf"([1-4])([qh])[{_WS}\-/']*'?([0-9]{{4}}|[0-9]{{2}})\Z")
_FQ3 = re.compile(rf"([0-9]{{4}})[{_WS}\-/:]*([qh])([1-4])\Z")
_AXIS_NOTE = re.compile(
    r"\((?:right|left|rhs|lhs|r\.h\.s\.|l\.h\.s\.)(?:[ -]hand)?(?: (?:side|axis|scale))*\)"
    r"|\b(?:right|left)[ -]hand axis\b|\b(?:right|left) axis\b",
    _A,
)
_YOY = re.compile(r"\by[/-]?o[/-]?y\b", _A)
_PAREN_LEAD = re.compile(rf"\({_S}*[,;]+{_S}*")
_EMPTY_PAREN = re.compile(rf"\({_S}*\)")
_YY_MARK = re.compile(r"'([0-9]{2})[efp]?\Z")
_APOSTROPHES = str.maketrans("’‘ʼ′", "''''")  # "’20" is "'20": typeset year marks
_MON_ONLY = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\Z")
_OUTER_PARENS = re.compile(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)")


def _year(y: str) -> str:
    return y if len(y) == 4 else ("20" + y if int(y) < 50 else "19" + y)


@lru_cache(maxsize=200_000)
def full_label(x: str) -> str:
    """Full-mode label key: ``nx_label`` plus FY prefixes, months, half-years, footnote markers and nested units.

    ``FY2011`` = ``2011``, ``Q1-23`` = ``1Q 2023`` = ``2023 Q1``, ``Jan-23`` = ``January 2023`` = ``2023 Jan``,
    ``Poland*`` / ``Poland¹`` / ``Poland [a]`` = ``Poland``, ``Investment (USD (2023, MER))`` = ``Investment``.
    """
    t = _FOOTNOTE_CHARS.sub("", str(x).replace("~~", "").translate(_APOSTROPHES))
    t = _YOY.sub("year over year", _AXIS_NOTE.sub(" ", pb_norm(t)))
    t = _EMPTY_PAREN.sub("", _PAREN_LEAD.sub("(", t))
    t = _js_trim(_BRACKET_NOTE.sub(" ", t))
    t = _OUTER_PARENS.sub(lambda m: " " if unit_desc(m.group(1)) else m.group(0), t)
    t = _js_trim(_WS_RUN.sub(" ", t))
    t = _FY.sub(lambda m: _year(m.group(1)), t)
    t = _YEAR_MARK.sub(lambda m: m.group(1), t)
    if m := _FQ1.match(t):
        t = f"{_year(m.group(3))}{m.group(1)}{m.group(2)}"
    elif m := _FQ2.match(t):
        t = f"{_year(m.group(3))}{m.group(2)}{m.group(1)}"
    elif m := _FQ3.match(t):
        t = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    elif m := _MON_Y.match(t):
        t = f"{_year(m.group(2))}m{_MONTHS[m.group(1)]:02d}"
    elif m := _Y_MON.match(t):
        t = f"{m.group(1)}m{_MONTHS[m.group(2)]:02d}"
    elif m := _YY_MARK.match(t):
        t = "20" + m.group(1)
    elif m := _MON_ONLY.match(t):
        t = f"m{_MONTHS[m.group(1)]:02d}"
    return t[:-1] if t.endswith("s") and len(t) > 2 else t


_LEGEND_WORDS = frozenset(
    "series line lines bar bars column area dot dots point points marker markers symbol shape dashed dotted solid "
    "dash triangle diamond circle square cross star light dark darker lighter pale bright deep medium "
    "blue red green yellow orange purple violet pink grey gray black white brown teal cyan magenta navy olive "
    "maroon gold beige turquoise lime upper lower middle top bottom left right first second third fourth "
    "unlabeled unlabelled unknown legend color colour shaded filled hollow outline value values data".split()
)


_SERIES_N = re.compile(rf"(series|column|col){_S}*[0-9]+\Z", _I)


def _marker_name(label: str) -> bool:
    """A colour, marker or ``Series 1`` name (generic headers such as ``Value`` are fine for one series)."""
    toks = [t for t in _content_toks(label) if not t.isdigit()]
    generic = {"value", "values", "data"}
    return bool(_SERIES_N.match(full_label(label))) or (
        bool(toks) and all(t in _LEGEND_WORDS and t not in generic for t in toks)
    )


def _lost_legend(label: str) -> bool:
    """A colour, marker or placeholder name in place of the legend (``Dark Blue Line``, ``Series 2``)."""
    toks = [t for t in _content_toks(label) if not t.isdigit()]
    return bool(_PLACEHOLDER.match(full_label(label))) or (bool(toks) and all(t in _LEGEND_WORDS for t in toks))


def _initials(toks: Sequence[str]) -> str:
    return "".join(t[0] for t in toks if t)


def _related(ref: str, out: str) -> bool:
    """A reworded label still names the same thing: a shared word, a prefix (``ITA``/``Italy``) or an acronym
    (``EU``/``European Union``). Colour or marker names (``Dark Blue Line``) and other categories are not."""
    a, b = _content_toks(ref), _content_toks(out)
    if not a or not b or _PLACEHOLDER.match(full_label(out)):
        return False
    if set(a) & set(b):
        return True
    sa, sb = _alnum("".join(a)), _alnum("".join(b))
    short, long_ = sorted((sa, sb), key=len)
    if not _LETTER.search(short) or short.isdigit():
        return False
    if len(short) >= 3 and long_.startswith(short):
        return True
    long_toks = b if len(sb) >= len(sa) else a
    ini = _initials(long_toks)
    return len(short) >= 2 and len(ini) >= 2 and (short.startswith(ini) or ini.startswith(short))


def _raw_label(x: str) -> str:
    """Label with parentheses kept as words (``SIDS (Yes %)`` stays distinct from ``SIDS (No %)``)."""
    return _strip_parens(pb_norm(_FOOTNOTE_CHARS.sub("", str(x).replace("~~", ""))))


@lru_cache(maxsize=400_000)
def label_bind(label: str, ev: str) -> float:
    """Label -> one evidence string: 1 exact, 0.995 equal once units are dropped, 0.97 token run inside the
    evidence, 0.9 truncated evidence, OSA >= 0.8 typo, else 0."""
    ra, rb = _raw_label(label), _raw_label(ev)
    if not ra or not rb:
        return 0.0
    if ra == rb or (_alnum(ra) and _alnum(ra) == _alnum(rb)):
        return 1.0
    a, b = full_label(label) or ra, full_label(ev) or rb
    if a == b or (_alnum(a) and _alnum(a) == _alnum(b)):
        return 0.995
    ca, cb = _content_toks(a), _content_toks(b)
    if ca and ca == cb:
        return 0.97  # the same words but for "of", "in", "the" ... ("Score in 2018" = "Score 2018")
    best = 0.0
    for x, y in ((a, b), (ra, rb)):
        if _TWO_DIGITS.match(x) and _YEAR.match(y) and y.endswith(x):
            return 1.0
        if _TWO_DIGITS.match(x) and (ty0 := _toks(y)[:1]) and _YEAR.match(ty0[0]) and ty0[0].endswith(x):
            best = max(best, 0.97)  # "19" for "2019 (Start)": the year of a finer x position
        if _year_cut(x, y):
            return 0.9  # "2016" printed as "16": binds, as a truncated category label
        tx, ty = _toks(x), _toks(y)
        if tx and _sub(tx, ty) and (_LETTER.search(x) or len(tx) > 1 or _YEAR.match(x)):
            best = max(best, 0.97)  # the label is a token run inside a longer cell (a panel or group prefix)
        elif ty and len(_alnum(y)) >= 3 and _LETTER.search(y) and tx[: len(ty)] == ty and 2 * len(ty) >= len(tx):
            best = max(best, 0.9)  # the evidence is a truncation of the label
        if abs(len(x) - len(y)) <= 0.2 * max(len(x), len(y)):  # else the typo similarity is below 0.8
            best = max(best, nx_label_sim(x, y))
    return best


def _year_cut(ref: str, out: str) -> bool:
    """A four-digit year cut to its last two digits (``2016`` -> ``16``)."""
    return bool(_YEAR.match(ref)) and bool(_TWO_DIGITS.match(out)) and len(out) == 2 and ref.endswith(out)


def _singular(toks: Sequence[str]) -> list[str]:
    return [t[:-1] if t.endswith("s") and len(t) > 2 else t for t in toks]


def _words_in(label: str, chain: Sequence[str]) -> bool:
    """Every word (two or more) of the label is in the point's row and column labels."""
    lt = _singular(_content_toks(full_label(label)))
    ct = {t for e in chain for t in _singular(_content_toks(full_label(e)))}
    return len(lt) >= 2 and all(t in ct for t in lt)


def chain_bind(label: str, chain: Sequence[str]) -> float:
    """Label -> a header chain (row label cells, column header levels): best element, or the joined chain."""
    if not chain:
        return 0.0
    best = max(label_bind(label, e) for e in chain)
    if len(chain) > 1 and best < 1:
        # the joined chain, either way round ("Canada Dec 2023" over the row "Dec 2023" > panel "Canada")
        best = max(best, label_bind(label, " ".join(chain)), label_bind(label, " ".join(reversed(chain))))
        if best < 0.97:  # "Summer CAGR (%)" over the chain "CAGR (%)" > "Summer": every word, in any order
            # (singular words: full_label drops a final "s" of the whole label only, "ETS 2012" over "2012" > "ETS")
            lt = _singular(_content_toks(full_label(label)))
            ct = {t for e in chain for t in _singular(_content_toks(full_label(e)))}
            if lt and (len(lt) > 1 or _LETTER.search(lt[0])) and all(t in ct for t in lt):
                best = 0.97
    return best


def _panel_bound(title: str, chain: Sequence[str], skip: Sequence[str]) -> bool:
    """A chain element (other than the ones naming the point) names the chart: a panel of a merged table."""
    if not _js_trim(title):
        return False
    tt = set(_content_toks(title))
    for e in chain:
        if label_bind(title, e) >= 0.9:
            return True
        if any(label_bind(s, e) >= 0.995 for s in skip if s):
            continue
        et = _content_toks(e)
        if et and _LETTER.search(" ".join(et)) and all(t in tt for t in et):
            return True
    return False


# --- output grids -------------------------------------------------------------------------------------------------

_PCT_RANGE_LABEL = re.compile(r"[-+−–]?[0-9]+%|[0-9]+(?:\.[0-9]+)?\s*[-–]\s*[0-9]+(?:\.[0-9]+)?%?")
_LABEL_HEADER = re.compile(
    r"\b(year|years|period|date|time|quarter|month|decile|quintile|percentile|rank|age|cohort|wave|round)\b", _I
)
_INT_CELL = re.compile(r"[0-9]{1,4}\Z")
_TENOR = re.compile(
    r"[0-9]+(?:\.[0-9]+)?\s*(?:y|yrs?|years?|mos?|months?|wks?|weeks?|days?)\Z", _I
)  # "10yr": a maturity
_KEY_NUM = re.compile(
    r"[0-9]{1,4}(?:\s*[-–/]\s*[0-9]{1,4})?(?:\s*[A-Za-z][0-9]?)?\Z"
)  # "2011-12", "96-99", "2024H1", "2029 P"
_LONG_KEY_HEADER = re.compile(r"series|categor(?:y|ies)|group|panel|chart|label|legend|x", _I)
_NOTE_TEXT = re.compile(r"(notes?|sources?|\*)\b", _I)


@dataclass
class _Grid:
    rows: list[list[str]]  # body rows, full width
    hchain: list[list[str]]  # per column: the header levels above it
    nlab: int  # leading label columns
    context: list[str]  # lines around the grid (before, caption, after): titles, units and notes
    heading: str | None = None
    rchain: list[list[str]] = field(default_factory=list)  # per body row: section row + label cells
    twins: frozenset[int] = frozenset()  # columns split off a compound "N (P%)" column: its second number
    twin_rows: frozenset[int] = frozenset()  # body rows split off a compound row
    ambiguous: bool = False  # an integer column headed like a label ("Start year") was read as values
    src: dict[tuple[int, int], tuple[int, int, int]] = field(default_factory=dict)  # joined grid: cell -> source cell
    tables: frozenset[int] = frozenset()  # the output tables (or JSON / list blocks) this grid reads

    @property
    def header(self) -> list[str]:
        return [" ".join(h) for h in self.hchain]

    @property
    def body(self) -> list[list[str]]:
        return self.rows


def _dedupe(xs: Iterable[str]) -> list[str]:
    return [x for x in dict.fromkeys(_js_trim(x) for x in xs) if x]


def _numeric_reading(v: str) -> bool:
    """A value cell for label-column detection: a number, a range, a compound "N (P%)" cell or an approximate
    number ("approximately 22", "22 (est.)"); a bound ("Above 60%") reads as a label here."""
    lead = _APPROX_LEAD.match(_js_trim(v))  # only approximation words: "Above 60%" is a label
    u = _js_trim(_APPROX_TAIL.sub("", _js_trim(v)[lead.end() :] if lead else _js_trim(v)))
    return _is_num(v) or _is_num(u) or bool(_COMPOUND_CELL.match(v))


def _is_num(s: str) -> bool:
    return bool(_NUMERIC_CELL.match(s.strip()) or _CELL_RANGE.match(s))


_YEAR_HEADER = re.compile(r"(?:fiscal\s+|calendar\s+)?years?|fy|period", _I)  # "Q2 | 2020": the year half of a key
_PERIOD_LABEL = re.compile(r"[1-4]\s*[qh]|[qh]\s*[1-4]", _I)  # "1Q", "H2": a period, not a number
_COMPOUND_CELL = re.compile(
    rf"(?P<a>[-+−]?[$€£¥]?[0-9][0-9,.]*{_S}*%?)(?:{_S}*[A-Za-z][A-Za-z ]{{0,20}}?)?{_S}*"
    rf"(?:\({_S}*(?P<b>[-+−]?[0-9][0-9,.]*{_S}*%?){_S}*\)|,{_S}+(?P<c>[-+−]?[0-9][0-9,.]*{_S}*%))"
    rf"{_S}*\Z"
)  # "32 (16.6%)", "135 countries, 70%": a count and its share ("18,7%" is a decimal comma, not two)
_NUMERIC_CELL = re.compile(r"[~≈<>]?\s*[-+−–(]?[$€£¥]?\d[\d,.\s]*%?\)?[A-Za-z]{0,3}\*?\Z")
_EMPTY_CELL = re.compile(r"[-–—.…]+|n/?a|n\.a\.|none|nil|×|x\Z", _I)


_MONTH_WORD = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:uary|ruary|ch|il|e|y|ust|t|tember|ober|ember)?\.?"
_MON_DAY = re.compile(rf"{_MONTH_WORD}{_S}*([0-9]{{1,2}})\Z", _I)
_MON_HEAD = re.compile(rf"\b{_MONTH_WORD}(?![a-z]){_S}*(?:19|20)[0-9]{{2}}\b", _I)  # "Date (Jan. 2026)"
_DAY = re.compile(r"[0-9]{1,2}\Z")


def _date_days(rows: list[list[str]], hdr: Sequence[int], body_idx: Sequence[int]) -> None:
    """Read bare day numbers in the label column as dates of the month above them, in place.

    A daily axis prints the month once: ``Jan. 2``, ``5``, ``6`` ..., or ``5``, ``6`` under a ``Date (Jan. 2026)``
    header (a month with its year). Each bare day takes the last month named before it. Nothing is rewritten
    unless every label is a month-day or a day, the days rise within each month and a month is named above the
    first bare day.
    """
    head = " ".join(rows[h][0] for h in hdr)
    month = m.group(1).lower() if (m := _MON_HEAD.search(head)) else None
    days: list[tuple[int, str]] = []
    prev = 0
    for r in body_idx:
        x = _js_trim(rows[r][0])
        if not x:
            continue
        if m := _MON_DAY.match(x):
            month, prev = m.group(1).lower(), int(m.group(2))
        elif _DAY.match(x) and month and prev < int(x) <= 31:
            prev = int(x)
            days.append((r, f"{month.capitalize()}. {x}"))
        else:
            return
    if len(days) >= 2:
        for r, x in days:
            rows[r][0] = x


def _fill_group_labels(rows: list[list[str]], body_idx: Sequence[int]) -> None:
    """Copy a group label written once down the blank label cells below it, in place.

    ``Low-income countries | D1 | 45`` followed by ``| D2 | 39``: the blank first cell belongs to the group above.
    Only column 0, only rows whose second cell is a text label followed by values, and only when no group label
    repeats (a table that writes every label in full is left as it is). At least two cells must be filled.
    """
    named = [rows[r][0] for r in body_idx if rows[r][0]]
    if not named or len(set(named)) < len(named):
        return
    fills: list[tuple[int, str]] = []
    group = ""
    for r in body_idx:
        row = rows[r]
        if row[0]:
            group = row[0]
        elif group and len(row) > 2 and row[1] and not _numeric_reading(row[1]) and any(row[2:]):
            fills.append((r, group))
    if len(fills) >= 2:
        for r, x in fills:
            rows[r][0] = x


def _make_grid(
    rows: list[list[str]],
    hdr: list[int],
    th: set[tuple[int, int]],
    drop: set[int],
    context: list[str],
    heading: str | None,
    split_rows: bool = False,
    keyed_unique: bool = False,
    before: Sequence[str] = (),
) -> _Grid | None:
    """``split_rows``: compound ``N (P%)`` cells are split into twin rows (series down the rows) instead of
    twin columns. ``before``: the lines above the table (a legend line names the parts of stacked cells)."""
    width = max((len(r) for r in rows), default=0)
    rows = [[_js_trim(str(x)) for x in r] + [""] * (width - len(r)) for r in rows]
    if width < 2 or len(rows) < 2:
        return None
    body_idx = [r for r in range(len(rows)) if r not in hdr and r not in drop]
    _date_days(rows, hdr, body_idx)
    _fill_group_labels(rows, body_idx)
    if split_rows:
        twins: set[int] = set()
        rows, body_idx, twin_rows = _split_compound_rows(rows, body_idx)
    else:
        rows, th, twins = _split_compound(rows, body_idx, th)
        twin_rows = set()
    rows, th, stack_names, shift = _split_stacked(rows, body_idx, th, before)
    rows = [[_js_trim(_STACK_SEP.sub(" ", x)) for x in r] for r in rows]  # other line breaks read as spaces
    twins = {t + shift[t] for t in twins}
    width = len(rows[0])
    hchain = [_dedupe(rows[h][j] for h in hdr) for j in range(width)]
    for j, name in stack_names.items():
        hchain[j] = [*hchain[j], name]
    for j in twins:
        if any("%" in rows[r][j] for r in body_idx):
            hchain[j] = [*hchain[j], "%"]
    # label columns: column 0, then <th> or mostly non-numeric columns, or integer columns headed year/decile/...
    nlab = 1
    ambiguous = False
    col0 = [rows[r][0] for r in body_idx if rows[r][0] and not _EMPTY_CELL.fullmatch(rows[r][0])]
    head0 = " ".join(hchain[0]) if hchain else ""
    if (
        col0
        and len(body_idx) <= 2
        and head0
        and all(
            _is_num(v)
            and not _YEAR.match(v.strip())
            and not _PCT_RANGE_LABEL.fullmatch(v.strip())
            and not _PERIOD_LABEL.fullmatch(v.strip())
            for v in col0
        )
        and not _LABEL_HEADER.search(head0)
    ):
        nlab = 0  # no label column: one row of values under "EU 2011 | US 2011 | ..." headers
    for j in range(1, width - 1) if nlab else ():
        vals = [(r, rows[r][j]) for r in body_idx if rows[r][j] and not _EMPTY_CELL.fullmatch(rows[r][j])]
        if not vals:
            break
        th_share = sum(1 for r, _ in vals if (r, j) in th) / len(vals)
        text_share = sum(1 for _, v in vals if not _numeric_reading(v)) / len(vals)
        head = " ".join(hchain[j])
        prev = [tuple(rows[r][:j]) for r in body_idx if any(rows[r][:j])]
        # a "Year" / "Quarter" column of integers next to repeated row labels (long format); next to unique
        # labels an integer column headed "25th percentile", "<30 years" or "Start year" holds values
        label_like = bool(
            _LABEL_HEADER.search(_PAREN_LAZY.sub(" ", head.lower()))
            and len(_content_toks(head, drop_parens=False)) <= 2
            and all(_INT_CELL.match(v) for _, v in vals)
        ) or all(_TENOR.match(v) for _, v in vals)
        codes = (
            len(vals) >= 5
            and len({v for _, v in vals}) <= 3
            and all(_INT_CELL.match(v) and int(v) < 100 for _, v in vals)
        )
        keyed = label_like and (
            keyed_unique
            or codes  # "Round | 1, 2, 3": a group code, not a value
            or len(set(prev)) < len(prev)
            or (_YEAR_HEADER.fullmatch(_js_trim(head)) is not None and all(_YEAR.match(v) for _, v in vals))
        )
        # long format: a key column of labels, some of them years ("Initiated", ..., "2017"), or a "Series" /
        # "Category" key of years, next to repeated row labels (one stray word in a value column is not a key)
        long_key = (
            len(set(prev)) < len(prev)
            and all(_KEY_NUM.match(v) or not _numeric_reading(v) for _, v in vals)
            and (text_share >= 0.25 or _LONG_KEY_HEADER.fullmatch(_js_trim(head)) is not None)
        )
        keyed = keyed or long_key
        if label_like and not keyed and th_share < 0.6 and text_share < 0.6:
            ambiguous = True  # both readings are offered (output_grids adds the label reading)
        # "Germany | 1", "Germany | 2": an integer column that tells repeated row labels apart is a label column
        splits = (
            not keyed
            and len(vals) == len(prev) >= 4
            and all(_INT_CELL.match(v) for _, v in vals)
            and len(set(prev)) <= 0.5 * len(prev)
            and len({(*rows[r][:j], v) for r, v in vals}) == len(vals)
        )
        if th_share >= 0.6 or text_share >= 0.6 or keyed or splits:
            nlab = j + 1
        else:
            break
    body, rchain, kept = [], [], []
    section: str | None = None
    for r in body_idx:
        row = rows[r]
        filled = [x for x in row if x]
        if not filled:
            continue
        values = [x for x in row[nlab:] if x]
        if (len(set(filled)) == 1 and len(filled) > 1) or (len(filled) == 1 and not values):
            text = filled[0]
            if len(text) > 60 or _NOTE_TEXT.match(text):
                continue  # a note or source line spanning the table
            if not _is_num(text):
                section = text  # a section row ("Crude steel") labels the rows below it
                continue
        body.append(row)
        kept.append(r)
        rchain.append(_dedupe([*([section] if section else []), *row[:nlab]]))
    if not body:
        return None
    twin_body = frozenset(k for k, r in enumerate(kept) if r in twin_rows)
    return _Grid(body, hchain, nlab, context, heading, rchain, frozenset(twins), twin_body, ambiguous)


_BR_MARK = "\u2063"  # a line break inside a table cell, kept through table parsing (which joins it with a space)
_STACK_SEP = re.compile(rf"{_S}*(?:{_BR_MARK}|<br{_S}*/?>){_S}*", _I)
_LEGEND_BOLD = re.compile(r"\*\*(.+?)\*\*")
_LEGEND_SEP = re.compile(rf"{_S}*(?:[|·•;]|{_S}{{2,}}){_S}*")


def _stack_parts(x: str) -> list[str] | None:
    """``2<br>6``: one number per series, stacked in one cell on separate lines."""
    parts = [p for p in _STACK_SEP.split(x) if p] if x and _STACK_SEP.search(x) else []
    if len(parts) < 2 or not all(_is_num(p) or _EMPTY_CELL.fullmatch(p) for p in parts):
        return None
    return parts


def _legend_names(before: Sequence[str], k: int) -> list[str] | None:
    """The k series names of a legend line written above the table (``**2024** **2022**``, ``2024 | 2022``)."""
    for ln in reversed(list(before)[-4:]):
        t = _js_trim(_HASHES.sub("", ln, count=1))
        if not t or len(t) > 120:
            continue
        for parts in (
            _LEGEND_BOLD.findall(t),
            _LEGEND_SEP.split(t.replace("**", "")),
            t.replace("**", "").split(),
        ):
            parts = [_js_trim(p) for p in parts if _js_trim(p)]
            if len(parts) == k and len(set(parts)) == k:
                return parts
    return None


def _split_stacked(
    rows: list[list[str]], body_idx: list[int], th: set[tuple[int, int]], before: Sequence[str]
) -> tuple[list[list[str]], set[tuple[int, int]], dict[int, str], list[int]]:
    """Columns whose cells stack one number per series (``2<br>6``) become one column per series, named by the
    legend line above the table, else ``Series 1``, ``Series 2`` (a lost legend). Also returns, per old column,
    how far it moved right."""
    width = len(rows[0]) if rows else 0
    split: dict[int, int] = {}
    for j in range(1, width):
        cells = [rows[r][j] for r in body_idx if rows[r][j] and not _EMPTY_CELL.fullmatch(rows[r][j])]
        ks = [len(p) for p in map(_stack_parts, cells) if p]
        if len(ks) >= 2 and len(ks) >= 0.6 * len(cells) and len(set(ks)) == 1:
            split[j] = ks[0]
    shift = [sum(split[s_] - 1 for s_ in split if s_ < j) for j in range(width)]
    if not split:
        return rows, th, {}, shift
    names = _legend_names(before, max(split.values())) or [f"Series {i + 1}" for i in range(max(split.values()))]
    body = set(body_idx)
    out_rows: list[list[str]] = []
    for r, row in enumerate(rows):
        new: list[str] = []
        for j, x in enumerate(row):
            n = split.get(j)
            if n is None:
                new.append(x)
            elif r not in body:
                new += [x] * n  # the header spans the series columns
            else:
                parts = _stack_parts(x) or [x]
                new += [("" if _EMPTY_CELL.fullmatch(p) else p) for p in parts[:n]] + [""] * (n - len(parts))
        out_rows.append(new)
    th2 = {(r, c + shift[c] + d) for r, c in th if c < width for d in range(split.get(c, 1))}
    named = {j + shift[j] + d: names[d] for j, n in split.items() for d in range(n)}
    return out_rows, th2, named, shift


def _split_compound_rows(rows: list[list[str]], body_idx: list[int]) -> tuple[list[list[str]], list[int], set[int]]:
    """A body row of ``127 countries, 66%`` cells (series down the rows) becomes the count row and, below it, a
    twin row ``<label> (%)`` holding the shares."""
    out: list[list[str]] = []
    new_body: list[int] = []
    twins: set[int] = set()
    body = set(body_idx)
    for r, row in enumerate(rows):
        cells = [x for x in row[1:] if x and not _EMPTY_CELL.fullmatch(x)]
        hits = [m for m in (_COMPOUND_CELL.match(x) for x in cells) if m]
        is_body = r in body
        if is_body and len(hits) >= 2 and len(hits) >= 0.6 * len(cells):
            first, second = [row[0]], [f"{row[0]} (%)" if row[0] else "(%)"]
            for x in row[1:]:
                m = _COMPOUND_CELL.match(x)
                first.append(_js_trim(m.group("a")) if m else x)
                second.append(_js_trim(m.group("b") or m.group("c") or "") if m else "")
            new_body.append(len(out))
            out.append(first)
            new_body.append(len(out))
            twins.add(len(out))
            out.append(second)
            continue
        if is_body:
            new_body.append(len(out))
        out.append(row)
    if not twins:
        return rows, body_idx, set()
    return out, new_body, twins


def _split_compound(
    rows: list[list[str]], body_idx: list[int], th: set[tuple[int, int]]
) -> tuple[list[list[str]], set[tuple[int, int]], set[int]]:
    """Columns of ``32 (16.6%)`` / ``135 countries, 70%`` cells become two columns: the count and, next to it, a
    twin holding the share (a chart may plot either or both)."""
    width = len(rows[0]) if rows else 0
    split = []
    for j in range(1, width):
        cells = [rows[r][j] for r in body_idx if rows[r][j] and not _EMPTY_CELL.fullmatch(rows[r][j])]
        hits = [m for m in (_COMPOUND_CELL.match(x) for x in cells) if m]
        if len(hits) >= 2 and len(hits) >= 0.6 * len(cells):
            split.append(j)
    if not split:
        return rows, th, set()
    body = set(body_idx)
    out_rows = []
    for r, row in enumerate(rows):
        new: list[str] = []
        for j, x in enumerate(row):
            if j not in split:
                new.append(x)
                continue
            m = _COMPOUND_CELL.match(x) if r in body else None
            if m:
                new += [_js_trim(m.group("a")), _js_trim(m.group("b") or m.group("c") or "")]
            else:
                new += [x, x if r not in body else ""]
        out_rows.append(new)
    shift = [sum(1 for s_ in split if s_ < j) for j in range(width)]
    th2 = {(r, c + shift[c]) for r, c in th if c < width} | {(r, c + shift[c] + 1) for r, c in th if c in split}
    twins = {j + shift[j] + 1 for j in split}
    return out_rows, th2, twins


def _sibling_grids(primary: Sequence[tuple[int, _Grid]]) -> list[_Grid]:
    """Consecutive tables with the same columns and (mostly) the same row labels, each under its own heading (one
    table per series, a column per chart), joined side by side on their row labels: each heading becomes the top
    header level of its columns. ``primary``: (index in the page's grids, grid) per table."""

    def keys(g: _Grid) -> list[tuple[str, ...]]:
        return [tuple(_raw_label(x) for x in ch) for ch in g.rchain]

    def same_shape(a: _Grid, b: _Grid) -> bool:
        ka, kb = set(keys(a)), set(keys(b))
        return (
            a.nlab == b.nlab
            and a.hchain[a.nlab :] == b.hchain[b.nlab :]
            and len(ka & kb) >= 0.8 * max(len(ka), len(kb))
        )

    out: list[_Grid] = []
    k = 0
    while k < len(primary):
        run = [primary[k][1]]
        while k + len(run) < len(primary) and same_shape(run[0], primary[k + len(run)][1]):
            run.append(primary[k + len(run)][1])
        heads = [g.heading for g in run]
        if len(run) >= 2 and all(heads) and len(set(heads)) == len(heads):
            nlab = run[0].nlab
            order: dict[tuple[str, ...], int] = {}
            labels: list[tuple[list[str], list[str]]] = []  # per joined row: (label cells, rchain)
            index: list[dict[tuple[str, ...], int]] = []  # per grid: its row key -> its row
            for g in run:
                idx: dict[tuple[str, ...], int] = {}
                for r, key in enumerate(keys(g)):
                    idx.setdefault(key, r)
                    if key not in order:
                        order[key] = len(order)
                        labels.append((g.rows[r][:nlab], list(g.rchain[r])))
                index.append(idx)
            rows: list[list[str]] = []
            src: dict[tuple[int, int], tuple[int, int, int]] = {}
            for jr, key in enumerate(order):
                row = list(labels[jr][0])
                for m, g in enumerate(run):
                    sr = index[m].get(key)
                    for c in range(nlab, len(g.hchain)):
                        if sr is not None:
                            src[(jr, len(row))] = (primary[k + m][0], sr, c)
                        row.append(g.rows[sr][c] if sr is not None and c < len(g.rows[sr]) else "")
                rows.append(row)
            hchain: list[list[str]] = [
                *run[0].hchain[:nlab],
                *([h or "", *c] for g, h in zip(run, heads, strict=True) for c in g.hchain[nlab:]),
            ]
            ctx = [x for g in run for x in g.context]
            out.append(_Grid(rows, hchain, nlab, ctx, None, [lab[1] for lab in labels], src=src))
        k += len(run)
    return out


def _panel_grids(primary: Sequence[tuple[int, _Grid]]) -> list[_Grid]:
    """Charts side by side in one table, each a group of columns under its own top header with its own label
    column (``Chart A: Category | Series`` next to ``Chart B: Year | Value``): one grid per group, its first column
    the labels. ``primary``: (index in the page's grids, grid) per table."""
    out: list[_Grid] = []
    for j, g in primary:
        width = len(g.hchain)
        if width < 4 or not all(len(h) >= 2 for h in g.hchain):
            continue
        groups: list[list[int]] = []
        for c in range(width):
            if groups and g.hchain[groups[-1][0]][0] == g.hchain[c][0]:
                groups[-1].append(c)
            else:
                groups.append([c])
        tops = [g.hchain[cols[0]][0] for cols in groups]
        if len(groups) < 2 or len(set(tops)) < len(tops) or any(len(cols) < 2 for cols in groups):
            continue
        for cols, top in zip(groups, tops, strict=True):
            labels = [g.rows[r][cols[0]] for r in range(len(g.rows)) if g.rows[r][cols[0]]]
            values = sum(1 for x in labels if _numeric_reading(x) and not _KEY_NUM.match(x))
            if not labels or values > 0.5 * len(labels):
                continue  # the group does not start with a label column
            rows: list[list[str]] = []
            rchain: list[list[str]] = []
            src: dict[tuple[int, int], tuple[int, int, int]] = {}
            for r, row in enumerate(g.rows):
                if not row[cols[0]] or not any(row[c] for c in cols[1:]):
                    continue
                for k, c in enumerate(cols):
                    src[(len(rows), k)] = (j, r, c)
                rows.append([row[c] for c in cols])
                rchain.append([row[cols[0]]])
            if rows:
                hchain = [list(g.hchain[c][1:]) for c in cols]
                out.append(_Grid(rows, hchain, 1, [*g.context, top], top, rchain, src=src))
    return out


def _add_table(
    grids: list[_Grid],
    primary: list[tuple[int, _Grid]],
    rows: list[list[str]],
    th: set[tuple[int, int]],
    thead_rows: set[int],
    tfoot_rows: set[int],
    context: list[str],
    heading: str | None,
    before: list[str],
) -> None:
    """The readings of one table (or one <thead> section of a table): its grid, the grid with compound cells split
    into rows, and the label reading of an ambiguous integer column."""
    n = len(rows)
    if n == 0:
        return
    w = len(rows[0])
    if thead_rows:
        hdr = sorted(thead_rows)
    elif th:
        # leading rows made only of <th> cells; a body row whose label cell is a <th> is a row header
        hdr = []
        for r in range(n):
            if all((r, c) in th for c in range(w) if rows[r][c]) and any(rows[r]):
                hdr.append(r)
            else:
                break
    else:
        hdr = [0]
    hdr = [h for k, h in enumerate(hdr) if h == k] or [0]
    if len(hdr) >= n:
        hdr = hdr[: max(1, n - 1)]
    # the real column header written as the first body row (under a "[tsv]" placeholder or a spanning
    # group row): every value cell of the row is a label and the row below holds values
    k = len(hdr)
    if k + 1 < n:
        texts = [x for x in rows[k][1:] if x]
        if (
            len(texts) >= 2
            and len(set(texts)) >= 2
            and not any(_value_cell(x) for x in texts)
            and any(x and _value_cell(x) for x in rows[k + 1][1:])
        ):
            hdr = [*hdr, k]
    g = _make_grid(rows, hdr, th, tfoot_rows, context, heading, before=before)
    if not g:
        return
    primary.append((len(grids), g))
    grids.append(g)
    g2 = _make_grid(rows, hdr, th, tfoot_rows, context, heading, split_rows=True, before=before)
    if g2 and g2.twin_rows:
        grids.append(g2)  # the same table with its compound cells split the other way
    if g.ambiguous:  # "Country | Round | Value" or "Category | Start year | ...": also as a label column
        g3 = _make_grid(rows, hdr, th, tfoot_rows, context, heading, keyed_unique=True, before=before)
        if g3:
            grids.append(g3)


_ROW_GROUP = re.compile(r"(<(thead|tbody|tfoot)\b[^>]*>)(.*?)(</\2\s*>)", _I | re.S)
_TR_OPEN = re.compile(r"<tr\b", _I)
_ROWSPAN_ATTR = re.compile(r"""(<t[hd]\b[^>]*?\browspan\s*=\s*["']?)(\d+)""", _I)


def _clip_rowspans(md: str) -> str:
    """Clip each ``rowspan`` at the end of its ``<thead>`` / ``<tbody>`` / ``<tfoot>``, as browsers render it.

    ``<th rowspan="2">Currency</th>`` in a one-row ``<thead>`` covers the header row only; read as written it
    would also fill the first body cell and shift that row's values one column right.
    """

    def group(m: re.Match[str]) -> str:
        starts = [t.start() for t in _TR_OPEN.finditer(m.group(3))]

        def clip(c: re.Match[str]) -> str:
            row = sum(1 for x in starts if x <= c.start()) - 1  # the row this cell sits in
            left = len(starts) - max(row, 0)
            return c.group(1) + str(min(int(c.group(2)), max(left, 1)))

        return m.group(1) + _ROWSPAN_ATTR.sub(clip, m.group(3)) + m.group(4)

    return _ROW_GROUP.sub(group, md) if "rowspan" in md.lower() else md


def _twin_free_columns(rows: Sequence[Sequence[str]], spanned: set[tuple[int, int]]) -> list[int]:
    """Columns to keep: each one except a column that is, in every row, a ``colspan`` copy of the cell on its
    left (``spanned`` holds the positions a span filled). Kept, the copy would score every value twice."""
    width = len(rows[0]) if rows else 0
    return [
        j
        for j in range(width)
        if j == 0 or any((r, j) not in spanned or row[j] != row[j - 1] for r, row in enumerate(rows))
    ]


def output_grids(md: str) -> list[_Grid]:
    """Every candidate data grid on the page: each table, plus JSON / bullet / per-series-merged grids."""
    from parse_bench.evaluation.metrics.parse.rules_chart import parse_chart_tables

    grids: list[_Grid] = []
    primary: list[tuple[int, _Grid]] = []  # one reading per table, in page order
    try:
        parsed = parse_chart_tables(_BR_I.sub(_BR_MARK, _clip_rowspans(md)))
    except Exception:  # noqa: BLE001 - a malformed table means no table
        parsed = []
    for td in parsed:
        a = td.data
        if getattr(a, "ndim", 0) != 2 or 0 in a.shape:
            continue
        all_rows = [[html.unescape(str(x)).replace("~~", "").strip() for x in row] for row in a.tolist()]
        keep = _twin_free_columns(all_rows, set(td.spanned_cells))
        col = {j: k for k, j in enumerate(keep)}
        all_rows = [[row[j] for j in keep] for row in all_rows]
        header_cells = {(r, col[c]) for r, c in td.header_cells if c in col}
        before = [_js_trim(x.replace(_BR_MARK, " ")) for x in (td.context_before or "").split("\n") if _js_trim(x)]
        after = [_js_trim(x.replace(_BR_MARK, " ")) for x in (td.context_after or "").split("\n") if _js_trim(x)][:3]
        cap = [td.caption.replace(_BR_MARK, " ")] if td.caption else []
        # a table of several <thead> sections (one chart each, stacked in one <table>): each section a table
        thead = sorted(td.thead_rows)
        starts = [0, *(r for k, r in enumerate(thead) if r > 0 and (k == 0 or thead[k - 1] != r - 1))]
        bounds = list(zip(starts, [*starts[1:], len(all_rows)], strict=True))
        for lo, hi in bounds:
            rows = all_rows[lo:hi]
            th = {(r - lo, c) for r, c in header_cells if lo <= r < hi}
            sec_head = {r - lo for r in thead if lo <= r < hi}
            tfoot = {r - lo for r in td.tfoot_rows if lo <= r < hi}
            heading = td.caption or heading_of(before)
            ctx = [*before, *cap, *after]
            if len(bounds) > 1 and sec_head:
                # the section's own title: a spanning header row above its column headers
                top = [x for x in dict.fromkeys(rows[min(sec_head)]) if x]
                if len(top) == 1 and lo > 0:
                    heading = top[0]
                ctx = [*ctx, *top[:1]]
            n0 = len(grids)
            _add_table(grids, primary, rows, th, sec_head, tfoot, ctx, heading, before)
            for reading in grids[n0:]:
                reading.tables = frozenset({n0})
    for joined in [*_sibling_grids(primary), *_panel_grids(primary)]:
        joined.tables = frozenset(t for j, _, _ in joined.src.values() for t in grids[j].tables)
        grids.append(joined)
    raws: list[tuple[list[list[str]], list[str], str | None]] = [
        (raw, [], None) for raw in data_grids(md, include_tables=not grids, series_json=False)
    ]
    for raw, ctx, heading in [*raws, *_series_json_grids(md)]:
        if len(raw) >= 2:
            g = _make_grid([[html.unescape(str(x)) for x in r] for r in raw], [0], set(), set(), ctx, heading)
            if g:
                g.tables = frozenset({len(grids)})
                grids.append(g)
    return grids


# --- values -------------------------------------------------------------------------------------------------------


def nx_value(
    raw: str,
    col_unit: dict[str, Any] | None,
    note_unit: dict[str, Any] | None,
    ref: dict[str, Any],
    note_only: bool = False,
) -> list[float]:
    """Cell value(s) in the reference unit; ``[nan]`` when the stated unit contradicts the reference.

    The unit comes from the cell suffix (``412.5M``), the column header (``Revenue (USD bn)``) or a note
    (``Values in millions``). A plain full-digit number is also read in the reference scale
    (``412,500,000`` = 412.5 when the reference is in millions). Only the cell and its own header can
    contradict the reference: prose and notes around a chart mention percentages and currencies freely, so a
    note only ever supplies a scale.
    """
    raw = _DASH_MINUS.sub(r"\1-", _unhedge(str(raw))[0])  # an en/em dash before a number is a minus sign
    if not raw or not _has_digit(raw):
        return []
    cands = num_cands(raw)
    if not cands:
        r = interps(raw)
        cands = r.vals[:1]
    if not cands:
        return []
    c = cands[-1] if _DECIMAL_COMMA.match(raw) else cands[0]  # 412,5 is a decimal comma
    if c < 0 and raw.startswith("(") and raw.endswith(")") and not ref.get("negatives"):
        # "(0.47)" is an accounting negative in a table, but charts print comparison values in parentheses too
        return [v for x in (c, -c) for v in nx_value(str(x), col_unit, note_unit, {**ref, "negatives": 1}, note_only)]
    pct = "%" in raw
    suffixed = bool(_SUFFIXED.search(raw))
    if col_unit:
        u = col_unit
    elif note_unit and note_unit.get("scale") and note_unit.get("kind") in (None, "cur"):
        u = note_unit
        note_only = True
    else:
        u = {}
    if note_only and (ref.get("kind") == "pct" or (u.get("kind") and u.get("kind") != "cur")):
        u = {}
    if not ref:  # the reference states no unit: nothing to contradict
        bare = num_cands(_SUFFIX.sub("", raw, count=1)) if suffixed else []
        return [c, *bare[:1]]
    nan = [math.nan]
    if ref.get("kind") == "pp":  # percentage points are often printed with a % sign
        return [c] if u.get("kind") in (None, "pp", "pct") else nan
    if u.get("kind") == "pp" and ref.get("kind") == "pct":
        return [c]  # a change in a share: "percentage points" over a column, "%" on the chart's axis
    if u.get("kind") == "pp" or (ref.get("kind") == "pct" and u.get("kind") == "cur"):
        return nan
    if ref.get("kind") == "pct":
        if pct:
            return [c]
        if u.get("kind") == "frac":  # "Share of ..." headers hold percentages as often as fractions
            return [c * 100, c] if abs(c) <= 1 else [c]
        if u.get("kind") and u.get("kind") != "pct":
            return nan
        return [c]
    if pct or (u.get("kind") and u.get("kind") != "cur"):
        return nan
    scale = ref.get("scale") or 1
    if suffixed:
        return [c / scale]
    if u.get("scale"):
        return [c * u["scale"] / scale]
    if scale > 1 and abs(c) >= 1000 and float(c).is_integer():
        return [c, c / scale]
    return [c]


_DASH_MINUS = re.compile(r"^([~≈]?\s*)[–—](?=\s*[$€£¥]?\d)")
_SIGNED_PAREN = re.compile(r"\(\s*[-+−–]\s*[$€£¥]?\d")
# hedges around a value read off a chart: an approximate reading is scored as written (no charge); a bound
# ("<1%", "less than 5") contains every value between 0 and the bound (scored as a range: one minor per chart)
_APPROX_LEAD = re.compile(
    rf"(?:approximately|approx\.?|about|around|roughly|nearly|almost|circa|ca\.|c\.|est\.|estimated|"
    rf"some|~|≈|±){_S}*(?=[-+−–—(]?[$€£¥]?{_S}*[0-9.])",
    _I,
)
_APPROX_TAIL = re.compile(rf"{_S}*(?:\({_S}*(?:approx\.?|est\.?|estimated?|e){_S}*\)|approx\.?|est\.?){_S}*\Z", _I)
_BOUND_LEAD = re.compile(
    rf"(?:<=?|≤|less{_S}+than|under|below|up{_S}+to|>=?|≥|more{_S}+than|over|above|at{_S}+least){_S}*"
    rf"(?=[-+−(]?[$€£¥]?{_S}*[0-9.])",
    _I,
)
_UPPER_BOUND = re.compile(rf"(?:<=?|≤|less{_S}+than|under|below|up{_S}+to)", _I)


def _unhedge(raw: str) -> tuple[str, str]:
    """Cell text without a leading/trailing hedge, and the bound it states: ``"<"``, ``">"`` or ``""``.

    ``Approximately £22M`` -> ``£22M``; ``~1M`` -> ``1M``; ``<1%`` -> (``1%``, ``"<"``).
    """
    s = _js_trim(raw)
    lead = _APPROX_LEAD.match(s)
    s2 = _js_trim(_APPROX_TAIL.sub("", s[lead.end() :] if lead else s))
    m = _BOUND_LEAD.match(s2)
    if m:
        return _js_trim(s2[m.end() :]), "<" if _UPPER_BOUND.match(m.group(0)) else ">"
    return s2 or s, ""


_DECIMALS = re.compile(r"\.([0-9]+)")


def _decimals(s: str) -> int:
    m = _DECIMALS.search(s.replace(",", ""))
    return len(m.group(1)) if m else 0


@dataclass
class _Tol:
    """Value tolerance of one chart: printed labels are exact up to rounding; axis readings use the range."""

    rel: float
    abs_: float
    labeled: bool

    def allowed(self, t: float, p: float) -> float:
        if self.labeled:
            return max(self.rel * max(abs(t), abs(p)), 1e-9)
        return max(self.rel * abs(t), self.abs_, 1e-9)

    def score(self, t: float, p: float, t_text: str = "", p_text: str = "") -> tuple[float, float]:
        """(value score, error / allowed): 1 within, linear to 0 at OFF_FACTOR x the allowed error."""
        a = self.allowed(t, p)
        e = abs(p - t)
        if self.labeled and t_text and p_text and e > a:
            d = _decimals(p_text)
            if d < _decimals(t_text) and abs(round(t, d) - p) <= 0.5 * 10**-d + 1e-9:
                return 1.0, 0.0  # the printed number rounded to the output's precision
        if _sign(p) != _sign(t) and abs(t) > a and abs(p) > a:
            return 0.0, math.inf
        if e <= a:
            return 1.0, e / a
        return max(0.0, 1 - (e - a) / ((OFF_FACTOR - 1) * a)), e / a


def chart_tol(chart: ChartSpec, nums: Sequence[float]) -> _Tol:
    if chart.data_labels_shown:
        return _Tol(chart.tol, 0.0, True)
    vals = [*nums, 0.0] if nums and (min(nums) >= 0 or max(nums) <= 0) else list(nums)
    rng = (max(vals) - min(vals)) if vals else 0.0
    return _Tol(chart.tol, RANGE_TOL * rng, False)


# --- one chart against one grid -----------------------------------------------------------------------------------


@dataclass
class _Fit:
    tot: float
    n_ref: int
    n_out: int
    events: ChartNextEvents
    values: list[float]
    labels: list[float]
    passed: int
    title_ok: bool
    penalty: float = 0.0  # critical + major floor charge of this chart
    penalty_minor: float = 0.0
    cell_n: dict[str, int] = field(default_factory=dict)  # per-cell events by severity (charged scaled)
    struct_n: dict[str, int] = field(default_factory=dict)  # structural events by severity (charged in full)
    unit: float = 1.0  # per-cell scale of this chart: min(1, CELL_EVENT_REF / cells)
    claimed: set[tuple[int, int]] = field(default_factory=set)  # grid cells (row, col) read as this chart's
    panel: float = 0.0  # share of matched cells whose row/column chain names the chart (merged panels)
    bound: int = 0  # matched reference points

    @property
    def prop(self) -> float:
        pr = self.tot / self.n_out if self.n_out else 0.0
        rc = self.tot / self.n_ref if self.n_ref else 0.0
        return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

    @property
    def score(self) -> float:
        main, minor = _first_event_full([self])
        return min(self.prop, _clamp01(1 - main - min(FLOOR["minor_cap"], minor)))

    @property
    def key(self) -> float:
        return self.prop + 0.01 * self.title_ok + 0.02 * self.panel


@dataclass
class _Pen:
    main: float = 0.0  # critical + major charge
    minor: float = 0.0
    omit: float = 0.0  # share of the charge that is omissions (capped at a missing chart)
    cell_n: dict[str, int] = field(default_factory=dict)
    struct_n: dict[str, int] = field(default_factory=dict)


def _charge(ev: ChartNextEvents, pen: _Pen, sev: str, kind: str, ref: str, out: str, unit: float) -> None:
    ev.add(sev, kind, ref, out)
    cell = kind in CELL_EVENTS
    w = FLOOR[sev] * (unit if cell else 1.0)
    counts = pen.cell_n if cell else pen.struct_n
    counts[sev] = counts.get(sev, 0) + 1
    if sev == "minor":
        pen.minor += w
    else:
        pen.main += w
    if kind in OMISSION_EVENTS:
        pen.omit += w


def _first_event_full(fits: Sequence[_Fit], extra_struct: Mapping[str, int] | None = None) -> tuple[float, float]:
    """Page charge (critical + major, minor): the first event of each severity costs its full weight, later
    per-cell events of that severity are scaled (one wrong value always costs at least 10 points)."""
    main = sum(f.penalty for f in fits)
    minor = sum(f.penalty_minor for f in fits)
    for sev in ("critical", "major", "minor"):
        struct = sum(f.struct_n.get(sev, 0) for f in fits) + (extra_struct or {}).get(sev, 0)
        cells = [f for f in fits if f.cell_n.get(sev)]
        if struct or not cells:
            continue  # the first event of the class is structural (already full) or there is none
        bump = FLOOR[sev] * (1 - cells[0].unit)
        if sev == "minor":
            minor += bump
        else:
            main += bump
    return main, minor


def _ref_points(chart: ChartSpec) -> tuple[list[_Point], set[tuple[str, str]]]:
    single = len(chart.series) == 1
    pts, unknown = [], set()
    for name, vals in chart.series:
        for i, cat in enumerate(chart.categories):
            raw = vals[i] if i < len(vals) else ""
            key = (_key(cat), "" if single else _key(name))
            if not _js_trim(raw) or _EMPTY_CELL.fullmatch(_js_trim(raw)):
                unknown.add(key)
                continue
            num = rule_number(raw)
            pts.append(_Point(key[0], key[1], num, text=pb_norm(raw), raw_c=cat, raw_r=name))
    return pts, unknown


@dataclass
class _Point:
    c: str  # normalized category label
    r: str  # normalized series label ("" on single-series charts)
    v: float | None  # number (None: text value or unreadable cell)
    text: str = ""
    range: bool = False
    raw_c: str = ""
    raw_r: str = ""


def _key(x: Any) -> str:
    return nx_label(pb_norm(x))


def _title_found(title: str, lines: Sequence[str]) -> bool:
    return any(_title_in_line(title, ln) for ln in lines)


_TITLE_NUMBERING = re.compile(
    rf"(?:figure|fig\.?|chart|exhibit|graph|table|panel){_S}*[0-9]+[a-z]?(?:[.\-][0-9a-z]+)*{_S}*[.:–—-]?{_S}*", _I
)


def _title_core(s: str) -> str:
    """A title without its ``Figure 4.1 –`` / ``Chart 3:`` numbering (empty when nothing else is left)."""
    return _js_trim(_TITLE_NUMBERING.sub("", s, count=1)) if _TITLE_NUMBERING.match(s) else s


def _close_keys(a: str, b: str) -> bool:
    longest = max(len(a), len(b), 1)
    # the edit distance is at least the length difference: skip the O(n m) distance on long prose lines
    return bool(a) and bool(b) and abs(len(a) - len(b)) <= 0.08 * longest and 1 - osa(a, b) / longest >= 0.92


_PANEL_TITLE = re.compile(rf"({_DOT}{{8,}}?){_S}+[–—-]{_S}+([^–—]{{1,60}})\Z")


@lru_cache(maxsize=100_000)
def _title_in_line(title: str, ln: str) -> bool:
    """The chart title is on this line: compared with and without the ``Figure 4.1 –`` numbering on either side
    (the line ``Figure 4.1 – Latest Fundraisings`` names the title ``Figure 4.1 – Latest Fundraisings``)."""
    t = _js_trim(_HASHES.sub("", ln, count=1).replace("**", ""))
    t = _js_trim(_TAG.sub(" ", t))
    core_t = _TITLE_PREFIX.sub("", t, count=1) if _TITLE_PREFIX.match(t) else t
    if not _key(core_t):
        return False
    title_core = _title_core(_js_trim(title)) or title
    wants = {_key(title), _key(title_core)}
    lines = {_key(t), _key(core_t), _key(_title_core(t))}
    if any(_close_keys(k, w) for k in lines for w in wants):
        return True
    if ctx_bind(title, core_t):
        return True
    # numbering on both sides, or a numbering that differs or is absent: the words of the title open the line
    # (a caption may go on with a subtitle; prose such as "Figure 4.1 shows the latest fundraisings ..." spreads
    # the words over a sentence)
    head = t[: int(1.3 * len(title)) + 10]
    return ctx_bind(title, head) or (
        title_core != title and len(_content_toks(title_core)) >= 2 and ctx_bind(title_core, head)
    )


def _assign(score: list[list[float]]) -> list[tuple[int, int]]:
    """Maximum-score one-to-one assignment over positive scores."""
    if not score or not score[0]:
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    m = np.array(score, dtype=float)
    rows, cols = linear_sum_assignment(-np.where(m > 0, m, -1.0))
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if m[i, j] > 0]


_UNIT_WORDS = re.compile(
    r"trillion|billion|million|thousand|percent|per cent|usd|eur|gbp|bn|mn|mln|pp|pts?|times|x\b", _I | _A
)
_LETTERS = re.compile("[^\\W\\d_]")


def _value_cell(x: str) -> bool:
    """A cell holding a number (``Covid-19 Pandemic`` is an annotation, ``$412.5 million`` a value,
    ``Approximately £22M`` a hedged value)."""
    x = _unhedge(_TAG.sub("", x))[0]
    if _COMPOUND_CELL.match(x):
        return True
    return _has_digit(x) and len(_LETTERS.findall(_UNIT_WORDS.sub("", x))) <= 4


_PCT_LABEL = re.compile(rf"[0-9][0-9.,]*{_S}*%(?:{_S}*[–-]{_S}*[0-9][0-9.,]*{_S}*%)?")


def _unit_of(chain: Sequence[str]) -> dict[str, Any] | None:
    """Unit stated by a header chain; ``Top 5%`` or ``20%–40%`` are labels, not percentages."""
    for e in reversed(chain):  # the leaf header is the most specific
        e = _PCT_LABEL.sub(" ", e)
        u = _label_unit(e) or unit_desc("%" if _PCT.search(e) else "")
        if u:
            return u
    return None


def _label_unit(label: str) -> dict[str, Any] | None:
    """Unit in a label's parenthetical (``Total (£m)`` -> currency, millions), else a percentage named in the
    label itself (``Insured losses as a percentage of ...``); None when the label states none."""
    for m in _PAREN.finditer(str(label)):
        u = unit_desc(_PCT_LABEL.sub(" ", m.group(1)))
        if u:
            return u
    bare = _PCT_LABEL.sub(" ", _PAREN.sub(" ", str(label)).lower())
    u = unit_desc(bare) or {}
    return {"kind": u["kind"]} if u.get("kind") in ("pct", "pp") else None


_TRENDLINE = re.compile(
    rf"(?:linear|poly\.?|polynomial|expon\.?|exponential|log\.?|logarithmic|power|"
    rf"[0-9]+{_S}*per\.?{_S}*mov\.?{_S}*avg\.?|moving{_S}+average){_S}*\(({_DOT}+)\){_S}*\Z",
    _I,
)


def _trendline_of(name: str) -> str | None:
    """The series an Excel trendline is drawn over: ``Linear (global average score)`` -> ``global average score``."""
    m = _TRENDLINE.match(_js_trim(str(name)))
    return _js_trim(m.group(1)) if m else None


def _distinct_words(names: Sequence[str]) -> list[list[str]]:
    """Per series name, the words that tell it apart (``real`` in ``Real minimum wage`` / ``Nominal minimum
    wage``); empty when the names share no word or a name has nothing of its own."""
    toks = [_singular(_content_toks(full_label(n))) for n in names]
    if len(toks) < 2 or not all(toks):
        return [[] for _ in names]
    common = set.intersection(*(set(t) for t in toks))
    if not common:
        return [[] for _ in names]
    out = [[t for t in ts if t not in common] for ts in toks]
    return [d if d and sum(1 for o in out if o == d) == 1 else [] for d in out]


_GENERIC_CORNER = frozenset(
    "country countries economy economies region regions area areas state states market markets category categories "
    "year years date dates period periods time month months quarter quarters week day group groups item items label "
    "labels name names type types variable variables indicator indicators measure measures metric metrics sector "
    "sectors segment segments industry industries series class classes characteristic characteristics "
    "component components dimension dimensions".split()
)
_VALUE_AXIS = re.compile(
    r"%|\bp\.?p\.?\b|\bpercent|\bshare\b|\bindex\b|\bscore\b|\brate\b|\bnumber\b|\bcount\b", _I | _A
)


def _generic_corner(corner: str) -> bool:
    """A column header that only says what kind of label the rows hold (``Country``, ``Year``, ``Category``):
    parsers write one when the chart has no axis title, so it never contradicts the reference's axis name."""
    toks = _singular(_content_toks(corner))
    return not toks or all(t in _GENERIC_CORNER or t + "s" in _GENERIC_CORNER for t in toks)


_NEUTRAL_CORNER = frozenset(
    "category categories label labels item items name names group groups type types variable variables indicator "
    "indicators measure measures metric metrics series characteristic characteristics component components "
    "dimension dimensions class classes".split()
)


def _neutral_corner(corner: str) -> bool:
    """A placeholder header (``Category``, ``Label``) that names no kind of category; ``Month`` or ``Region``
    does, and a different one than the reference axis (``Month``) is a mislabeled axis."""
    toks = _singular(_content_toks(corner))
    return not toks or all(t in _NEUTRAL_CORNER or t + "s" in _NEUTRAL_CORNER for t in toks)


def _value_axis_name(xl: str) -> bool:
    """The reference ``x_label`` names the value axis (a horizontal bar chart's ``Score in 2025``, ``Mt CO2``,
    ``%``): a unit, a share, an index or a score is not the category axis the output's corner names."""
    return bool(unit_desc(xl) or _label_unit(xl) or _VALUE_AXIS.search(xl) or _UNIT_WORDS.search(xl))


_LABEL_PARTS = re.compile(rf":{_S}+|{_S}+[-–—]{_S}+")


def _unique_parts(labels: Sequence[str]) -> list[list[str]]:
    """Per label, the parts of a ``Code: Name`` / ``Group - Item`` label that no other label shares
    (``AR`` and ``ARGENTINA`` for ``AR: ARGENTINA``; not ``2020`` for ``Women - 2020`` next to ``Men - 2020``)."""
    parts = [[full_label(x) for x in _LABEL_PARTS.split(str(lab)) if _js_trim(x)] for lab in labels]
    parts = [p if len(p) >= 2 else [] for p in parts]
    count: dict[str, int] = {}
    for p in parts:
        for x in set(p):
            count[x] = count.get(x, 0) + 1
    return [[x for x in p if count[x] == 1 and x] for p in parts]


def _part_bind(parts: Sequence[str], chain: Sequence[str]) -> float:
    """A label written as one of its unique parts (``AR`` for ``AR: ARGENTINA``): bound, not reworded."""
    return 0.97 if parts and any(full_label(e) in parts for e in chain) else 0.0


def _generic_header(e: str) -> bool:
    """A placeholder column name (``Value``, ``Data``, ``Series 1``) that names no series."""
    return bool(_PLACEHOLDER.match(full_label(e)))


@dataclass
class _ChartRef:
    """A reference chart prepared for matching."""

    spec: ChartSpec
    cats: list[str]
    names: list[str]
    single: bool
    known: dict[tuple[int, int], tuple[float | None, str]]  # (category, series) -> (number, text)
    tol: _Tol
    ref_unit: dict[str, Any]
    others: tuple[str, ...] = ()  # titles of the other charts on the page (their panels are not invented here)
    page_cells: int = 0  # known reference cells of the whole page
    series_units: list[dict[str, Any] | None] = field(default_factory=list)
    cat_units: list[dict[str, Any] | None] = field(default_factory=list)
    optional: frozenset[int] = frozenset()  # trendline series (``Linear (X)``): scored when given, never required
    distinct: list[list[str]] = field(default_factory=list)  # per series: its words not shared by every series
    pct_points: frozenset[tuple[int, int]] = frozenset()  # reference values printed with a % sign
    cat_parts: list[list[str]] = field(default_factory=list)  # per category: its unique "Code: Name" parts
    virtual: int | None = None  # the series standing for merged colour groups (see ``merged``)
    axis_read: frozenset[tuple[int, int]] = frozenset()  # labeled chart: points whose value is read off the axis
    tol_axis: _Tol | None = None  # ... and their (unlabeled) tolerance
    mixed_signs: bool = False  # the reference holds negative and positive values
    other_labels: tuple[str, ...] = ()  # categories and series names of the other charts on the page
    max_abs: float = 0.0  # the largest reference magnitude

    def tol_at(self, ci: int, si: int) -> _Tol:
        """A partially labeled chart: printed labels are exact, the other points are read from the axis."""
        return self.tol_axis if self.tol_axis is not None and (ci, si) in self.axis_read else self.tol

    merged: _ChartRef | None = None  # the chart with its colour-group series read as one series

    @property
    def n_required(self) -> int:
        """Reference points the output must hold (trendline series are optional)."""
        return sum(1 for _, si in self.known if si not in self.optional)

    def point_unit(self, ci: int, si: int) -> dict[str, Any]:
        """A unit stated in a series or category name (``Fixed pay (%)``, ``Total (£m)``) beats the chart unit;
        a reference value printed with ``%`` (a ``+6%`` change series in a €bn chart) is a percentage."""
        chart_u = self.ref_unit
        if not self.series_units[si] and any(self.series_units):
            # other series carry their own unit ("Earnings per share (EUR)"): the chart unit is theirs, and a
            # series without one ("Payout ratio") may be in another unit (a combo chart's second axis)
            chart_u = {}
        u = self.series_units[si] or self.cat_units[ci] or chart_u
        if (ci, si) in self.pct_points and u.get("kind") != "pct":
            return _PCT_UNIT
        return u

    @classmethod
    def of(cls, spec: ChartSpec, virtual: int | None = None) -> _ChartRef:
        known = {}
        for si, (_, vals) in enumerate(spec.series):
            for ci in range(len(spec.categories)):
                raw = vals[ci] if ci < len(vals) else ""
                if _js_trim(raw) and not _EMPTY_CELL.fullmatch(_js_trim(raw)):  # "N/A", "-": no data
                    known[(ci, si)] = (rule_number(raw), raw)
        nums = [v for v, _ in known.values() if v is not None]
        trend = [si for si, (n, _) in enumerate(spec.series) if _trendline_of(n)]
        # trendlines and series flagged optional are scored when given, never required (not every series of a chart)
        optional = frozenset(trend) | spec.optional
        if len(optional) >= len(spec.series):
            optional = frozenset()
        return cls(
            spec,
            list(spec.categories),
            [n for n, _ in spec.series],
            len(spec.series) == 1,
            known,
            chart_tol(spec, nums),
            spec.ref_unit(),
            series_units=[_with_chart_scale(_label_unit(n), spec) for n, _ in spec.series],
            pct_points=frozenset(p for p, (_, raw) in known.items() if "%" in raw),
            cat_units=[_label_unit(c) for c in spec.categories],
            optional=optional,
            distinct=_distinct_words([n for n, _ in spec.series]),
            cat_parts=_unique_parts(spec.categories),
            virtual=virtual,
            merged=None if virtual is not None else _merged_groups(spec),
            mixed_signs=any(v < 0 for v in nums) and any(v > 0 for v in nums),
            max_abs=max((abs(v) for v in nums), default=0.0),
        )


_PCT_UNIT: dict[str, Any] = {"kind": "pct"}


def _with_chart_scale(u: dict[str, Any] | None, spec: ChartSpec) -> dict[str, Any] | None:
    """A series unit without a scale (``Deal Value ($)``) takes the scale of the chart unit of the same kind
    (``$ in bn``)."""
    if not u or u.get("scale"):
        return u
    for part in spec.unit_parts():
        if part.get("kind") == u.get("kind") and part.get("scale"):
            return {**u, "scale": part["scale"]}
    return u


def _colour_groups(spec: ChartSpec) -> list[int]:
    """Series that only colour-code the bars: two or more series, each holding some categories and no two the
    same one (``Emerging innovators`` / ``Strong innovators`` ... one bar per country), covering most categories."""
    n = len(spec.categories)
    sup = []
    for si, (_, vals) in enumerate(spec.series):
        cats = {ci for ci in range(n) if ci < len(vals) and _js_trim(vals[ci])}
        if cats and len(cats) < n:
            sup.append((si, cats))
    if len(sup) < 2:
        return []
    seen: set[int] = set()
    for _, cats in sup:
        if cats & seen:
            return []
        seen |= cats
    return [si for si, _ in sup] if len(seen) >= 0.6 * n else []


def _merged_groups(spec: ChartSpec) -> _ChartRef | None:
    """The chart with its colour-group series merged into one (a parser that keeps every bar in one column)."""
    group = _colour_groups(spec)
    if not group:
        return None
    vals = [""] * len(spec.categories)
    for si in group:
        for ci, v in enumerate(spec.series[si][1][: len(vals)]):
            if _js_trim(v):
                vals[ci] = v
    rest = [s for si, s in enumerate(spec.series) if si not in group]
    name = " / ".join(spec.series[si][0] for si in group)
    merged = ChartSpec(**{**spec.__dict__, "series": [*rest, (name, vals)], "optional": frozenset()})
    return _ChartRef.of(merged, virtual=len(rest))


def _mark_axis_read(ch: _ChartRef, rules: Sequence[ChartRule]) -> None:
    """A chart flagged as labeled whose sparse rules say some of its points are estimated from the axis
    (``need_estimate``) is partially labeled: those points, and every point no exact (printed) rule confirms,
    get the unlabeled tolerance; points an exact rule confirms keep the 1% label tolerance."""
    if not ch.spec.data_labels_shown or not rules:
        return
    exact: set[tuple[int, int]] = set()
    estimated = False
    for r in rules:
        num = r.num
        if num is None:
            continue
        for (ci, si), (v, _) in ch.known.items():
            if v is None or vscore(v, num, max(r.tol, 0.01)) < 1:
                continue
            labs = [ch.cats[ci], ch.names[si], ch.spec.title]
            if all(any(max(label_bind(lab, x), bind(lab, x)) >= 0.9 for x in labs if _js_trim(x)) for lab in r.labels):
                if r.estimate:
                    estimated = True
                else:
                    exact.add((ci, si))
    if estimated:
        nums = [v for v, _ in ch.known.values() if v is not None]
        ch.tol_axis = chart_tol(ChartSpec(**{**ch.spec.__dict__, "data_labels_shown": False}), nums)
        ch.axis_read = frozenset(p for p in ch.known if p not in exact)


def _series_bind(ch: _ChartRef, si: int, chain: Sequence[str]) -> float:
    """Series name -> a column's header chain. A trendline also binds through the series it is drawn over
    (``Linear (global average score)`` ~ ``Global average score``); a series named like its chart (a panel's own
    series next to its trendlines) binds a placeholder ``Value`` column."""
    name = ch.names[si]
    best = chain_bind(name, chain)
    if best < 0.97 and ch.distinct[si]:
        # the legend's distinguishing words: "Real" for "Real minimum wage" next to "Nominal minimum wage"
        if any(_singular(_content_toks(full_label(e))) == ch.distinct[si] for e in chain):
            best = 0.97
        elif _js_trim(ch.spec.title):
            # ... after the chart's own title: "A. Literacy Before" for "Before accounting for demographics"
            tt = set(_singular(_content_toks(full_label(ch.spec.title))))
            for e in chain:
                et = _singular(_content_toks(full_label(e)))
                rest = [t for t in et if t not in tt]
                if rest and len(rest) < len(et) and rest == ch.distinct[si]:
                    best = max(best, 0.9)
    core = _trendline_of(name) if si in ch.optional else None
    others = [n for k, n in enumerate(ch.names) if k != si]
    if core and not any(chain_bind(n, chain) >= 0.995 for n in others):
        # through its series' name only when the column is not named like that series itself (bind by name
        # first: a bar column holding the trendline's values is a swap, not a trendline)
        best = max(best, 0.98 * chain_bind(core, chain))
    if si not in ch.optional and best < 0.995 and any(_trendline_of(e) for e in chain):
        best = 0.0  # "Linear (X)" is the trendline's column, not series X's
    title = ch.spec.title
    if best < 0.995 and chain and _generic_header(chain[-1]) and _js_trim(title) and label_bind(name, title) >= 0.995:
        best = 0.995
    return best


RENAMED_AGREE = 0.5  # legend replaced: least mean value score of a column bound to a series by its values
RENAMED_MARGIN = 0.25  # ... and its lead over every other pairing of that column or that series


def _bind_by_values(
    ch: _ChartRef, free_b: Sequence[int], amap: Mapping[int, int], agree_b: Callable[[int, int, dict[int, int]], float]
) -> list[tuple[int, int]]:
    """Series of a chart none of whose legend names bind (the output names its lines after the callouts printed
    on the chart): named columns -> series by value agreement, Hungarian on the mean value score of each
    (column, series) pair. A pair binds when its agreement is at least ``RENAMED_AGREE`` and leads every other
    pairing of the same column and of the same series by ``RENAMED_MARGIN``, so close or crossing series stay
    unbound. All or nothing: every series must bind (a replaced legend renames them all). Needs two or more
    series, none of them optional: one series has no alternative to beat. ``free_b`` holds named columns only."""
    if len(ch.names) < 2 or ch.optional or ch.virtual is not None or len(free_b) < len(ch.names):
        return []
    names = range(len(ch.names))
    am = dict(amap)
    agree = {(b, si): agree_b(b, si, am) for b in free_b for si in names}
    pairs = _assign([[agree[(b, si)] for si in names] for b in free_b])
    out = []
    for i, j in pairs:
        b, si = free_b[i], names[j]
        s = agree[(b, si)]
        rivals = [agree[(b, sj)] for sj in names if sj != si] + [agree[(b2, si)] for b2 in free_b if b2 != b]
        if s >= RENAMED_AGREE and all(s >= r + RENAMED_MARGIN for r in rivals):
            out.append((b, si))
    return out if len(out) == len(ch.names) else []


@dataclass
class _View:
    """One reading of a grid: category-side units (A) and series-side units (B) with their label chains."""

    a_chain: dict[int, list[str]]
    b_chain: dict[int, list[str]]
    cells: dict[tuple[int, int], tuple[str, tuple[int, int]]]  # (a, b) -> (text, grid position)
    cellwise: bool = False  # every cell its own category (labels split over the row and the column)


_HEAD_TAIL = re.compile(
    rf"(.*[^{_WS}]){_S}+((?:19|20)[0-9]{{2}}(?:{_S}*[qh][1-4])?|[qh][1-4]{_S}*'?[0-9]{{2,4}})\Z", _I
)
_NOTE_COLUMN = re.compile(r"(notes?|comments?|remarks?|sources?|annotations?|events?)\b", _I)
_HEAD_SEP = re.compile(rf"{_S}+[-–—:/|]{_S}+|:{_S}+")


def _views(g: _Grid, exclude: set[tuple[int, int]]) -> list[_View]:
    """Categories down the rows, across the header, and (multi-level headers) the top header level melted
    into the rows: ``Women - Researchers`` read as row ``EU-27 Women``, series ``Researchers``."""
    vcols = [j for j in range(g.nlab, len(g.hchain)) if not _NOTE_COLUMN.match(" ".join(g.hchain[j]))]
    pos = [
        (r, j, row[j])
        for r, row in enumerate(g.rows)
        for j in vcols
        if j < len(row) and row[j] and _value_cell(row[j]) and (r, j) not in exclude
    ]
    rows_ = {r: g.rchain[r] for r in range(len(g.rows))}
    cols_ = {j: g.hchain[j] for j in vcols}
    views = [
        _View(rows_, cols_, {(r, j): (x, (r, j)) for r, j, x in pos}),
        _View(cols_, rows_, {(j, r): (x, (r, j)) for r, j, x in pos}),
    ]
    splits: list[dict[int, tuple[str, list[str]]]] = []
    split: dict[int, tuple[str, list[str]]] = {}
    for j in vcols:
        h = g.hchain[j]
        if len(h) >= 2:
            split[j] = (h[0], h[1:])
        elif h and (parts := _HEAD_SEP.split(h[0], maxsplit=1)) and len(parts) == 2 and all(map(_js_trim, parts)):
            split[j] = (_js_trim(parts[0]), [_js_trim(parts[1])])
        elif h and (m := _HEAD_TAIL.match(h[0])):
            split[j] = (_js_trim(m.group(1)), [m.group(2)])  # "EU 2011": series "EU", category 2011
    splits.append(split)
    if vcols and all(len(g.hchain[j]) >= 3 for j in vcols):
        # three header levels ("Posted wage growth and Inflation" > "Canada" > "Inflation"): the middle level
        # can be the one that belongs to the rows ("Canada Dec 2023")
        splits.append({j: (g.hchain[j][1], [g.hchain[j][0], *g.hchain[j][2:]]) for j in vcols})
    for split in splits:
        prefixes = list(dict.fromkeys(p for p, _ in split.values()))
        suffixes = list(dict.fromkeys(tuple(x) for _, x in split.values()))
        if not (len(split) == len(vcols) and len(prefixes) >= 2):
            continue
        # the top header level read as part of the row label (and, for "EU 2011", the tail as the category)
        for top_to_rows in (True, False):
            a_chain: dict[int, list[str]] = {}
            b_chain: dict[int, list[str]] = {}
            cells = {}
            for r, j, x in pos:
                pre, suf = split[j]
                if top_to_rows:
                    a = r * len(prefixes) + prefixes.index(pre)
                    a_chain[a], b = [*g.rchain[r], pre], suffixes.index(tuple(suf))
                    b_chain[b] = list(suf)
                else:
                    a = r * len(suffixes) + suffixes.index(tuple(suf))
                    a_chain[a], b = [*g.rchain[r], *suf], prefixes.index(pre)
                    b_chain[b] = [pre]
                cells[(a, b)] = (x, (r, j))
            if top_to_rows or len(suffixes) >= 2:
                views.append(_View(a_chain, b_chain, cells))
    # one series whose categories are split over the row and the column ("Green new occupations - Total" at row
    # "Total", column "Green new occupations"): every cell its own category
    if len(vcols) >= 2 and len({r for r, _, _ in pos}) >= 2:
        views.append(
            _View(
                {k: [*g.rchain[r], *g.hchain[j]] for k, (r, j, _) in enumerate(pos)},
                {0: []},
                {(k, 0): (x, (r, j)) for k, (r, j, x) in enumerate(pos)},
                cellwise=True,
            )
        )
    # long format (several label columns, e.g. Panel | Currency | Value): one label column read as the series
    for q in range(min(g.nlab, 4)) if g.nlab >= 2 else ():
        if len({g.rows[r][q] for r, _, _ in pos if g.rows[r][q]}) < 2:
            continue
        a_ids: dict[tuple[str, ...], int] = {}
        b_ids: dict[tuple[str, int], int] = {}
        pa: dict[int, list[str]] = {}
        pb: dict[int, list[str]] = {}
        cells = {}
        for r, j, x in pos:
            key_b = g.rows[r][q]
            if not key_b:
                continue
            chain_a = [e for e in g.rchain[r] if e != key_b]
            a = a_ids.setdefault(tuple(chain_a), len(a_ids))
            b = b_ids.setdefault((key_b, j), len(b_ids))
            pa[a] = chain_a
            pb[b] = [key_b, *g.hchain[j]]
            cells.setdefault((a, b), (x, (r, j)))
        views.append(_View(pa, pb, cells))
    return views


_YEAR_POS = re.compile(r"((?:19|20)[0-9]{2})(?:([qhm])([0-9]{1,2}))?\Z")
_NUM_POS = re.compile(r"-?[0-9]+(?:\.[0-9]+)?\Z")


def _cat_pos(label: str) -> float | None:
    """Position of an x label on a continuous axis: a year (quarter, half, month) or a plain number."""
    t = full_label(label).replace(" ", "")
    if m := _YEAR_POS.match(t):
        per = {"q": 4, "h": 2, "m": 12}.get(m.group(2) or "", 1)
        return int(m.group(1)) + ((int(m.group(3)) - 1) / per if m.group(3) else 0.0)
    return float(t) if _NUM_POS.match(t) else None


def _unit_variant(chain: Sequence[str], bound: Sequence[str]) -> bool:
    """The same series in another unit (``current dollars ($/lb)`` next to ``current dollars ($/kg)``)."""
    a, b = " ".join(chain), " ".join(bound)
    return full_label(a) == full_label(b) and _raw_label(a) != _raw_label(b)


def _fit_view(  # noqa: C901, PLR0912, PLR0915 - one pass over the matching steps
    ch: _ChartRef, g: _Grid, view: _View, page_lines: Sequence[str]
) -> _Fit:
    spec = ch.spec
    ev = ChartNextEvents()
    pen = _Pen()
    n_cells = max(ch.page_cells if FLOOR_SCOPE == "page" else ch.n_required, 1)
    unit = min(1.0, CELL_EVENT_REF / n_cells)
    # units: A = category side, B = series side; cells keyed (a, b) with their grid position
    a_chain, b_chain, cells = view.a_chain, view.b_chain, view.cells
    A = sorted({a for a, _ in cells})
    B = sorted({b for _, b in cells})
    title = spec.title
    title_ok = True
    if _js_trim(title):
        # a short panel title ("Europe", "Income") must sit next to its table; a long figure title anywhere
        long_title = len(_content_toks(title)) >= 3
        title_ok = _title_found(title, [*g.context, *(page_lines if long_title else ())])
    if not cells:
        return _Fit(0.0, ch.n_required, 0, ev, [], [], 0, title_ok)

    note_unit = unit_desc(" ".join(ln for ln in g.context if _NOTE_LINE.search(ln)))
    vals: dict[tuple[int, int, int], list[float]] = {}
    bounds: dict[tuple[int, int, int], tuple[float, float]] = {}
    range_hit: set[tuple[int, int]] = set()  # matched cells whose range contains the reference value
    upper: set[tuple[int, int, int]] = set()  # cells stating an upper bound ("<1%"): ranges from 0
    # a grid writing "(-11)" wraps every value in parentheses ("Finland (32)"): there "(32)" is +32
    signed_parens = any(_SIGNED_PAREN.match(x) for row in g.rows for x in row)

    def cell_vals(a: int, b: int, ref_unit: dict[str, Any]) -> list[float]:
        key = (a, b, id(ref_unit))
        if key not in vals:
            x, bound = _unhedge(cells[(a, b)][0])
            col_unit = _unit_of(b_chain[b]) or _unit_of(a_chain[a])
            rng = _CELL_RANGE.match(x)
            if bound:
                # "<1%" contains every value between 0 and 1 (a range); ">5" is scored as its number
                vals[key] = nx_value(x, col_unit, note_unit, ref_unit)
                p = vals[key][0] if vals[key] else math.nan
                if bound == "<" and not math.isnan(p) and p > 0:
                    bounds[key] = (0.0, p)
                    upper.add(key)
            elif rng:
                lo = nx_value(rng.group(1) + rng.group(3), col_unit, note_unit, ref_unit)
                hi = nx_value(rng.group(2) + rng.group(3), col_unit, note_unit, ref_unit)
                vals[key] = [(lo[0] + hi[0]) / 2] if lo and hi else []
                if lo and hi and not math.isnan(lo[0]) and not math.isnan(hi[0]):
                    bounds[key] = (min(lo[0], hi[0]), max(lo[0], hi[0]))
            else:
                vals[key] = nx_value(x, col_unit, note_unit, ref_unit)
                if ch.mixed_signs and not signed_parens and _PAREN_ALL.match(_js_trim(x)):
                    # "(5)" reads as 5 or -5 (a comparison label or an accounting negative); on a chart with
                    # values of both signs it is the negative only, so parentheses cannot hedge the sign
                    vals[key] = [v for v in vals[key] if not v > 0]
        return vals[key]

    def vscore_cell(ci: int, si: int, a: int, b: int) -> tuple[float, float]:
        num, text = ch.known[(ci, si)]
        x = cells[(a, b)][0]
        if num is None:
            return (1.0, 0.0) if pb_norm(x) == pb_norm(text) else (0.0, math.inf)
        best = (0.0, math.inf)
        pu = ch.point_unit(ci, si)
        rb = bounds.get((a, b, id(pu))) if cell_vals(a, b, pu) else None
        tol = ch.tol_at(ci, si)
        if (a, b, id(pu)) in upper and any(tol.score(num, p)[0] == 1 for p in vals[(a, b, id(pu))]):
            rb = None  # "<1%" for 1%: the bound is the value
        a_ = tol.allowed(num, num)
        # a range no wider than the tolerance, or a "<1%" bound under 2% of the chart's largest value (a slice
        # too small to print), that contains the value (one minor for the chart); a wider range or bound is
        # scored by its midpoint or its bound, like a single reading
        narrow = rb is not None and (
            rb[1] - rb[0] <= 2 * a_ or ((a, b, id(pu)) in upper and rb[1] <= RANGE_TOL * ch.max_abs)
        )
        if rb and narrow and rb[0] - a_ <= num <= rb[1] + a_:
            return (1.0, 0.0)
        for p in cell_vals(a, b, pu):
            if not math.isnan(p):
                s = tol.score(num, p, text, x)
                if s[0] > best[0] or (s[0] == best[0] and s[1] < best[1]):
                    best = s
        return best

    # label binding
    KA = {
        a: [max(chain_bind(c, a_chain[a]), _part_bind(ch.cat_parts[ci], a_chain[a])) for ci, c in enumerate(ch.cats)]
        for a in A
    }
    if ch.single:
        lab = [ch.names[0], title]
        # an unnamed series of an untitled chart has no label to bind: its values place it (assign_b)
        KB = {b: [max((chain_bind(x, b_chain[b]) for x in lab if _js_trim(x)), default=0.0)] for b in B}
    else:
        KB = {b: [_series_bind(ch, si, b_chain[b]) for si in range(len(ch.names))] for b in B}
    PA = {a: _panel_bound(title, a_chain[a], ch.cats) for a in A}
    PB = {b: _panel_bound(title, b_chain[b], ch.names) for b in B}

    def agree_a(a: int, ci: int, bmap: dict[int, int]) -> float:
        pts = [(si, b) for b, si in bmap.items() if (ci, si) in ch.known and (a, b) in cells]
        n = sum(1 for si in range(len(ch.names)) if (ci, si) in ch.known)
        return sum(vscore_cell(ci, si, a, b)[0] for si, b in pts) / n if n else 0.0

    def agree_b(b: int, si: int, amap: dict[int, int]) -> float:
        pts = [(ci, a) for a, ci in amap.items() if (ci, si) in ch.known and (a, b) in cells]
        n = sum(1 for ci in range(len(ch.cats)) if (ci, si) in ch.known)
        return sum(vscore_cell(ci, si, a, b)[0] for ci, a in pts) / n if n else 0.0

    def assign_a(bmap: dict[int, int] | None) -> dict[int, int]:
        sc = [
            [
                (KA[a][ci] + 0.05 * PA[a] + (0.3 * agree_a(a, ci, bmap) if bmap else 0.0)) if KA[a][ci] else 0.0
                for ci in range(len(ch.cats))
            ]
            for a in A
        ]
        return {A[i]: ci for i, ci in _assign(sc)}

    def assign_b(amap: dict[int, int]) -> dict[int, int]:
        if ch.single:
            scored = [(KB[b][0] + 0.05 * PB[b] + 0.3 * agree_b(b, 0, amap), -k, b) for k, b in enumerate(B)]
            best = max(scored)
            return {best[2]: 0} if best[0] > 0 or len(B) == 1 else {}
        sc = [
            [
                (KB[b][si] + 0.05 * PB[b] + 0.3 * agree_b(b, si, amap)) if KB[b][si] else 0.0
                for si in range(len(ch.names))
            ]
            for b in B
        ]
        return {B[i]: si for i, si in _assign(sc)}

    amap = assign_a(None)
    bmap = assign_b(amap)
    amap = assign_a(bmap)
    bmap = assign_b(amap)
    alias_a: set[int] = set()
    alias_b: set[int] = set()

    # reworded labels: an unbound unit carrying an unbound reference label's values
    def derived(chain: Sequence[str]) -> bool:
        return bool(chain) and bool(_DERIVED.match(full_label(chain[-1])))

    if not ch.single:
        legend_lost = False
        free_b = [b for b in B if b not in bmap and not derived(b_chain[b])]
        for si in range(len(ch.names)):
            if si in bmap.values():
                continue
            n = sum(1 for ci in range(len(ch.cats)) if (ci, si) in ch.known)
            best = None
            for b in free_b:
                if b in bmap:
                    continue
                lost = not b_chain[b] or _lost_legend(b_chain[b][-1])
                related = not lost and any(_related(ch.names[si], e) for e in b_chain[b])
                hits = sum(
                    1
                    for a, ci in amap.items()
                    if (ci, si) in ch.known and (a, b) in cells and vscore_cell(ci, si, a, b)[0] == 1
                )
                # a related name ("North" / "Northern region") or a lost legend ("Dark Blue Line", "Series 2")
                # needs 60% of the values; an unrelated name ("Inflation" for "Average Annual CPI") nearly all;
                # a trendline's name is the spreadsheet's, not the chart's: its values are enough
                need = 0.6 * n if related or lost or si in ch.optional or si == ch.virtual else max(3, 0.85 * n)
                rank_ = 2 if related else 1 if lost else 0
                if hits and hits >= need and (best is None or (rank_, hits) > best[1:]):
                    best = (b, rank_, hits)
            if best:
                bmap[best[0]] = si
                alias_b.add(best[0])
                if si in ch.optional or si == ch.virtual:
                    continue  # an optional trendline under another name: harmless; merged groups: charged below
                if best[1] == 1:
                    if not legend_lost:  # one charge per chart: the reader cannot tell the series apart
                        legend_lost = True
                        lab_out = " / ".join(b_chain[best[0]])
                        _charge(ev, pen, "critical", "legend lost", ch.names[si], lab_out, unit)
                    continue
                kind, sev = ("series name reworded", "minor") if best[1] == 2 else ("series renamed", "major")
                _charge(ev, pen, sev, kind, ch.names[si], " / ".join(b_chain[best[0]]), unit)
        if not bmap and amap:
            # named columns: a header with letters that is not one of the chart's categories (a transposed reading)
            named = [
                b
                for b in free_b
                if b_chain[b]
                and _LETTER.search(pb_norm(b_chain[b][-1]))
                and not _lost_legend(b_chain[b][-1])
                and not any(label_bind(c, b_chain[b][-1]) >= 0.95 for c in ch.cats)
            ]
            for b, si in _bind_by_values(ch, named, amap, agree_b):
                bmap[b] = si
                alias_b.add(b)
                _charge(ev, pen, "major", "series renamed", ch.names[si], " / ".join(b_chain[b]), unit)
    elif bmap:
        b0 = next(iter(bmap))
        if b_chain[b0] and _marker_name(b_chain[b0][-1]) and not KB[b0][0]:
            # a single series under a colour or "Series 1" name: the reader still has the values (major)
            _charge(ev, pen, "major", "legend lost", ch.names[0], " / ".join(b_chain[b0]), unit)
    if ch.virtual is not None and ch.virtual in bmap.values():
        # every bar is there, in one column: only the colour coding of the groups is lost
        _charge(ev, pen, "minor", "colour groups merged", ch.names[ch.virtual], "", unit)
    free_a = [a for a in A if a not in amap and not derived(a_chain[a])]
    rank = {a: k for k, a in enumerate(A)}
    for ci in range(len(ch.cats)):
        if ci in amap.values():
            continue
        n = sum(1 for si in range(len(ch.names)) if (ci, si) in ch.known)
        best_a: tuple[int, int] | None = None
        for a in free_a:
            if a in amap or (ch.single and abs(rank[a] - ci) > 1):
                continue
            if not any(_related(ch.cats[ci], e) for e in a_chain[a]):
                continue
            hits = sum(
                1
                for b, si in bmap.items()
                if (ci, si) in ch.known and (a, b) in cells and vscore_cell(ci, si, a, b)[0] == 1
            )
            if hits >= max(1, 0.6 * n) and (best_a is None or hits > best_a[1]):
                best_a = (a, hits)
        if best_a:
            amap[best_a[0]] = ci
            alias_a.add(best_a[0])
            _charge(ev, pen, "major", "category label reworded", ch.cats[ci], " / ".join(a_chain[best_a[0]]), unit)

    # title and category-axis name
    panel_cells = 0
    if _js_trim(title) and not title_ok:
        title_ok = any(PA[a] for a in amap) or any(PB[b] for b in bmap)
        m = _PANEL_TITLE.match(_js_trim(title))
        if not title_ok and m and len(_content_toks(_title_core(m.group(1)))) >= 2:  # not "Figure 4.1 – ..."
            # "<figure title> – Brazil": the figure title on the page, the panel name on the table (its
            # heading, a header or a row label of the matched units)
            fig, panel = m.group(1), m.group(2)
            labels_ = [*(e for a in amap for e in a_chain[a]), *(e for b in bmap for e in b_chain[b]), *g.context]
            title_ok = _title_found(fig, [*g.context, *page_lines]) and any(
                label_bind(panel, e) >= 0.97 or _title_in_line(panel, e) for e in labels_
            )
        if not title_ok:
            _charge(ev, pen, "major", "wrong title" if g.heading else "title missing", title, g.heading or "", unit)
    corner = " ".join(g.hchain[0]) if g.hchain else ""
    if g.nlab >= 2 and amap:
        # several label columns (Chart | Category | ...): the axis name is the header of the categories' column
        col_hits = [
            sum(
                1
                for a, ci in amap.items()
                for r in {cells[(a, b)][1][0] for b in B if (a, b) in cells}
                if label_bind(ch.cats[ci], g.rows[r][q]) >= 0.97
            )
            for q in range(g.nlab)
        ]
        if max(col_hits):
            corner = " ".join(g.hchain[col_hits.index(max(col_hits))])
    xl = spec.x_label

    def _cats_down_rows() -> bool:
        """The categories are the grid's rows, so the corner header names the category axis (in a transposed
        table it names the series: "Region" over regions is right)."""
        rows_of = [{cells[(a, b)][1][0] for b in B if (a, b) in cells} for a in amap]
        return bool(rows_of) and all(len(r) == 1 for r in rows_of) and len({min(r) for r in rows_of}) == len(rows_of)

    if (
        corner
        and _js_trim(xl)
        and not label_bind(xl, corner)
        and not bind(xl, corner)
        and not _value_axis_name(xl)
        and (
            not _generic_corner(corner)
            or (not _neutral_corner(corner) and _generic_corner(xl) and not _neutral_corner(xl) and _cats_down_rows())
        )
        and not _related(xl, corner)
    ):
        _charge(ev, pen, "minor", "category axis mislabeled", xl, corner, unit)

    # points
    inv_a = {ci: a for a, ci in amap.items()}
    inv_b = {si: b for b, si in bmap.items()}
    tot = 0.0
    values: list[float] = []
    labels: list[float] = []
    passed = bound = 0
    claimed: set[tuple[int, int]] = set()
    missing: list[tuple[int, int]] = []
    opt_bound = 0  # matched points of optional (trendline) series: scored, so they count as reference points
    cut_a: set[int] = set()
    typo_seen: set[tuple[str, int]] = set()
    unit_pos: dict[tuple[str, int], tuple[set[int], set[int]]] = {}
    for (ua, ub), (_, (gr, gc)) in cells.items():
        for key in (("a", ua), ("b", ub)):
            rs, cs = unit_pos.setdefault(key, (set(), set()))
            rs.add(gr)
            cs.add(gc)

    def blank_zero(ci: int, si: int, a: int, b: int) -> bool:
        """A reference 0 left blank (or ``-``) at the crossing of its bound row and column: a zero-height bar or
        segment is not drawn, so an empty cell is the same reading."""
        num = ch.known[(ci, si)][0]
        if num != 0 or ("a", a) not in unit_pos or ("b", b) not in unit_pos:
            return False
        (ra, ca), (rb, cb) = unit_pos[("a", a)], unit_pos[("b", b)]
        if len(ra) == 1 and len(cb) == 1:
            gr, gc = next(iter(ra)), next(iter(cb))
        elif len(ca) == 1 and len(rb) == 1:
            gr, gc = next(iter(rb)), next(iter(ca))
        else:
            return False
        x = g.rows[gr][gc] if gr < len(g.rows) and gc < len(g.rows[gr]) else None
        return x is not None and (not x or bool(_EMPTY_CELL.fullmatch(x)))

    for (ci, si), (_, text) in ch.known.items():
        ma, mb = inv_a.get(ci), inv_b.get(si)
        if ma is not None and mb is not None and (ma, mb) not in cells and blank_zero(ci, si, ma, mb):
            tot += 1.0
            values.append(1.0)
            labels.append(1.0)
            passed += 1
            bound += 1
            opt_bound += si in ch.optional
            continue
        if ma is None or mb is None or (ma, mb) not in cells:
            if si not in ch.optional:  # a trendline left out costs nothing
                missing.append((ci, si))
            continue
        a, b = ma, mb
        opt_bound += si in ch.optional
        ka = 1.0 if a in alias_a or KA[a][ci] >= 0.95 else KA[a][ci]  # unit / date forms and panel prefixes bind
        kb = 1.0 if ch.single or b in alias_b or KB[b][si] >= 0.95 or si in ch.optional else KB[b][si]
        if ka < 0.95 and _words_in(ch.cats[ci], [*a_chain[a], *b_chain[b]]):
            ka = 1.0  # "Capacity in 2021" split over the row "2021" and the column "Capacity in"
        if kb < 0.95 and _words_in(ch.names[si], [*a_chain[a], *b_chain[b]]):
            kb = 1.0
        k = ka * kb
        v, err = vscore_cell(ci, si, a, b)
        claimed.add(cells[(a, b)][1])
        bound += 1
        panel_cells += PA[a] or PB[b]
        tot += k * v
        values.append(v)
        labels.append(k)
        passed += v == 1
        ref_lab = f"{ch.cats[ci]} / {ch.names[si]}"
        out_lab = f"{' / '.join(a_chain[a])} / {' / '.join(b_chain[b])}"
        if a not in alias_a and KA[a][ci] < 0.95 and any(_year_cut(ch.cats[ci], _raw_label(e)) for e in a_chain[a]):
            if a not in cut_a:
                cut_a.add(a)
                _charge(ev, pen, "major", "category label truncated", ch.cats[ci], " / ".join(a_chain[a]), unit)
        elif k < 0.95:
            # one typo per reference label: a misspelt series name is one error, not one per point
            for side, uid, kk in (("a", ci, ka), ("b", si, kb)):
                if kk < 0.95 and (side, uid) not in typo_seen:
                    typo_seen.add((side, uid))
                    _charge(ev, pen, "minor", "label typo", ref_lab, out_lab, unit)
        pu = ch.point_unit(ci, si)
        if (
            v == 1
            and (a, b, id(pu)) in bounds
            and not (
                (a, b, id(pu)) in upper
                and any(ch.tol_at(ci, si).score(ch.known[(ci, si)][0] or 0.0, p)[0] == 1 for p in vals[(a, b, id(pu))])
            )
        ):
            range_hit.add((a, b))
        if err > OFF_FACTOR:
            _charge(ev, pen, "critical", "wrong value", f"{ref_lab}: {text}", cells[(a, b)][0], unit)
        elif v < 1:
            _charge(ev, pen, "major", "value off", f"{ref_lab}: {text}", cells[(a, b)][0], unit)
    if range_hit:
        _charge(ev, pen, "minor", "range instead of value", "", f"{len(range_hit)} cells", unit)
    # cells of the matched rows x columns at reference-unknown positions are neither scored nor charged
    for a in amap:
        for b in bmap:
            if (a, b) in cells and (amap[a], bmap[b]) not in ch.known:
                claimed.add(cells[(a, b)][1])
    # omissions: a whole series (critical) or category (major) once, else per point
    gone: set[tuple[int, int]] = set()
    if not ch.single:
        for si in dict.fromkeys(si for _, si in missing):
            idx = [p for p in ch.known if p[1] == si]  # (si is never optional: its points are not missing)
            if si not in inv_b:
                _charge(ev, pen, "critical", "series missing", ch.names[si], "", unit)
                gone.update(idx)
        for ci in dict.fromkeys(ci for ci, si in missing if (ci, si) not in gone):
            idx = [p for p in ch.known if p[0] == ci and p not in gone and p[1] not in ch.optional]
            if len(idx) > 1 and ci not in inv_a:
                _charge(ev, pen, "major", "category missing", ch.cats[ci], "", unit)
                gone.update(idx)
    for ci, si in missing:
        if (ci, si) not in gone:
            _charge(ev, pen, "major", "point missing", f"{ch.cats[ci]} / {ch.names[si]}", "", unit)

    # unmatched output rows / columns inside the grid: derived totals, duplicates, invented labels
    n_out = bound
    a_units = [a for a in A if not derived(a_chain[a])]
    b_units = [b for b in B if not derived(b_chain[b])]
    well = (
        bool(amap) and bool(bmap) and len(amap) >= 0.5 * len(a_units) and (ch.single or len(bmap) >= 0.5 * len(b_units))
    )
    bound_keys_a = {_raw_label(" ".join(a_chain[a])) for a in amap}
    bound_keys_b = {_raw_label(" ".join(b_chain[b])) for b in bmap}
    seen_extra: set[str] = set()
    sides = (("a", A, amap, a_chain, bound_keys_a), ("b", B, bmap, b_chain, bound_keys_b))
    single_well = ch.single and bool(amap) and bool(bmap) and len(amap) >= 0.5 * len(a_units)
    cat_pos = [p for p in (_cat_pos(c) for c in ch.cats) if p is not None]
    ref_span = (min(cat_pos), max(cat_pos)) if cat_pos and len(cat_pos) == len(ch.cats) else None
    bound_rank = sorted(rank[a] for a in amap)

    def inside(u: int) -> bool:
        """An extra x position between reference categories is a denser sampling of a continuous axis."""
        p = _cat_pos(" ".join(a_chain[u]))
        if ref_span is not None and p is not None:
            return ref_span[0] <= p <= ref_span[1]
        return bool(bound_rank) and bound_rank[0] < rank[u] < bound_rank[-1]

    # the table is presented as the chart (its title next to it) and holds exactly the chart's series: rows
    # appended before or after the chart's categories are invented, whatever share of the rows the chart binds
    as_chart = (
        not view.cellwise
        and bool(amap)
        and (ch.single or len(bmap) >= len(b_units) > 0)
        and bool(_js_trim(title))
        and _title_found(title, g.context)
    )

    def appended(u: int) -> bool:
        """A row before or after the bound categories (a continuous axis may be sampled beyond its ticks), with no
        label of another chart on the page."""
        if not as_chart or spec.continuous or not bound_rank or bound_rank[0] <= rank[u] <= bound_rank[-1]:
            return False
        return not any(label_bind(t, e) >= 0.95 for t in ch.other_labels for e in a_chain[u])

    def _repeats(u: int, side: str, ucells: list[tuple[int, int]]) -> bool:
        """The unit repeats a bound unit's label in the same unit (a "% change" column under the same spanning
        header as the values is another measure, not a repeated point)."""
        cmap_, chains_ = (amap, a_chain) if side == "a" else (bmap, b_chain)
        key = _raw_label(" ".join(chains_[u]))

        def pct_share(cs: Sequence[tuple[int, int]]) -> float:
            return sum("%" in cells[c][0] for c in cs) / len(cs) if cs else 0.0

        mine = pct_share(ucells)
        for t in cmap_:
            if _raw_label(" ".join(chains_[t])) != key:
                continue
            tcells = [c for c in cells if (c[0] if side == "a" else c[1]) == t]
            if abs(pct_share(tcells) - mine) < 0.5:
                return True
        return False

    for side, units, cmap, chains, keys in sides:
        if side == "b" and ch.single and not single_well:
            continue
        other = bmap if side == "a" else amap
        for u in units:
            if u in cmap:
                continue
            ucells = [((u, o) if side == "a" else (o, u)) for o in other]
            ucells = [c for c in ucells if c in cells]
            if not ucells:
                continue
            label = " / ".join(chains[u])
            if all(cells[c][1][1] in g.twins or cells[c][1][0] in g.twin_rows for c in ucells) and all(
                "%" in cells[c][0] for c in ucells
            ):
                continue  # the share half of a "N (P%)" column: the chart plotted the other half
            if derived(chains[u]):
                if label not in seen_extra:
                    seen_extra.add(label)
                    _charge(ev, pen, "major", "derived total", "", label, unit)
                continue
            if chains[u] and _trendline_of(chains[u][-1]):
                continue  # a "Linear (X)" trendline is derived from the data, never invented
            if any(chain_bind(ch.names[si], chains[u]) >= 0.97 for si in ch.optional):
                continue  # an optional series (a reference line, a 100 - Yes share) written across the other axis
            if side == "b" and ch.single:
                # an extra value column next to a single series: invented data, once per column (a column of
                # another reference chart is excluded before this read; the same series in another unit is free)
                if any(label_bind(t, e) >= 0.9 for t in ch.others for e in chains[u]):
                    continue
                if any(_unit_variant(chains[u], b_chain[bb]) for bb in bmap):
                    continue
                _charge(ev, pen, "major", "invented series or category", "", label or "(unnamed column)", unit)
                n_out += len(ucells)
                continue
            if _raw_label(" ".join(chains[u])) in keys and _repeats(u, side, ucells):  # the same row twice
                for _ in ucells:
                    _charge(ev, pen, "major", "repeated point", "", label, unit)
                n_out += len(ucells)
                continue
            if (not well and not (side == "a" and appended(u))) or (side == "a" and spec.continuous and inside(u)):
                continue
            if any(label_bind(t, e) >= 0.9 for t in ch.others for e in chains[u]):
                continue  # a panel of another chart merged into this grid
            trend_lines = any(_trendline_of(ch.names[si]) for si in ch.optional)
            if side == "b" and trend_lines and len(ucells) > 1 and len({cells[c][0] for c in ucells}) == 1:
                continue  # a flat average line next to a chart drawn with trendlines: derived, harmless
            if any(
                max(chain_bind(core, chains[u]), max((label_bind(e, core) for e in chains[u]), default=0.0)) >= 0.97
                for si in ch.optional
                if (core := _trendline_of(ch.names[si]))
            ):
                continue  # a trendline's values (optional) in a row or column of a transposed table
            if label not in seen_extra:
                seen_extra.add(label)
                _charge(ev, pen, "major", "invented series or category", "", label, unit)
            n_out += len(ucells)
    # omissions never cost more than a missing chart
    if pen.omit > FLOOR["critical"]:
        pen.main -= pen.omit - FLOOR["critical"]
    return _Fit(
        tot,
        ch.n_required + opt_bound,
        n_out,
        ev,
        values,
        labels,
        passed,
        title_ok,
        penalty=pen.main,
        penalty_minor=pen.minor,
        cell_n=pen.cell_n,
        struct_n=pen.struct_n,
        unit=unit,
        claimed=claimed,
        panel=panel_cells / bound if bound else 0.0,
        bound=bound,
    )


def _fit_chart(ch: _ChartRef, g: _Grid, page_lines: Sequence[str], exclude: set[tuple[int, int]] | None = None) -> _Fit:
    """Best orientation of one grid for one chart (categories down the rows, or across the header)."""
    views = _views(g, exclude or set())
    # the cell-by-cell reading only for one series whose categories join two labels ("Occupation - Total")
    joined = ch.single and sum(1 for c in ch.cats if len(_HEAD_SEP.split(c, maxsplit=1)) == 2) >= 0.5 * len(ch.cats)
    views = [v for v in views if joined or not v.cellwise]
    fits = [_fit_view(c, g, v, page_lines) for c in (ch, ch.merged) if c is not None for v in views]
    return max(fits, key=lambda f: (f.key, f.bound, -f.penalty))


_RELABELED = frozenset(
    {"series renamed", "series name reworded", "legend lost", "category label reworded", "category label truncated"}
)


def _restates(ch: _ChartRef, f: _Fit) -> bool:
    """The grid holds every point of the chart under the chart's own category and series names (a second
    version of the chart, not another measure that shares its caption)."""
    return (
        f.bound >= ch.n_required > 0
        and all(k >= 0.95 for k in f.labels)
        and not any(kind in _RELABELED for kind, _, _ in f.events.examples)
    )


def _copy_charges(f: _Fit, placed: _Fit, tol: _Tol) -> _Fit:
    """A second version of a placed chart: a point it states wrongly where the placed version is right is a
    contradiction, charged as a value error and counted as an output cell with no credit (precision).

    A point the placed version already gets wrong is charged once, there; a copy cell that writes the reference
    number itself, within tolerance (its header's unit read differently), is no contradiction; what the copy
    leaves out or adds is not charged."""
    ev = ChartNextEvents()
    pen = _Pen()
    bad = {ref for kind, ref, _ in placed.events.examples if kind in ("wrong value", "value off")}
    n = 0
    for kind, ref, out in f.events.examples:
        if kind not in ("wrong value", "value off") or ref in bad:
            continue
        t = rule_number(ref.rsplit(": ", 1)[-1])
        if t is not None and any(tol.score(t, p)[0] == 1 for p in num_cands(_unhedge(out)[0])):
            continue
        _charge(ev, pen, "critical" if kind == "wrong value" else "major", kind, ref, out, f.unit)
        n += 1
    return _Fit(
        0.0,
        0,
        n,
        ev,
        [],
        [],
        0,
        True,
        penalty=pen.main,
        penalty_minor=pen.minor,
        cell_n=pen.cell_n,
        struct_n=pen.struct_n,
        unit=f.unit,
    )


def _foreign_cells(refs: Sequence[_ChartRef], g: _Grid) -> dict[int, set[tuple[int, int]]]:
    """Per chart: the grid cells that sit in another chart's panel of a merged table.

    A row or column labeled with a chart's title (``Chart | Category | Series | Value``, a rowspan panel cell, a
    column per chart) belongs to that chart. Its cells are left out when another chart reads the grid, unless the
    label is also one of that chart's own categories or series (a pie titled ``1969`` next to a line chart with a
    ``1969`` category)."""
    titles = [_js_trim(r.spec.title) for r in refs]
    if len({t for t in titles if t}) < 2:
        return {}
    cache: dict[str, frozenset[int]] = {}

    def owners(e: str) -> frozenset[int]:
        if e not in cache:
            # a title inside a longer label ("SIDS" in "SIDS (Yes %)") owns it; a number title ("1979") only when
            # it is the whole label (not the base year in "GDP (DA 1979)")
            sc = [
                s
                if (s := label_bind(t, e) if t else 0.0) >= 0.995
                or (s >= 0.97 and _LETTERS.search(t) and set(_content_toks(t, False)) <= set(_content_toks(e, False)))
                else 0.0
                for t in titles
            ]
            top = max(sc)
            cache[e] = frozenset(k for k, s in enumerate(sc) if s == top) if top >= 0.97 else frozenset()
        return cache[e]

    def own(chain: Sequence[str]) -> dict[str, frozenset[int]]:
        return {e: o for e in chain if (o := owners(e))}

    rows = [own(ch) for ch in g.rchain]
    cols = [own(ch) for ch in g.hchain]

    def panels(tags: list[dict[str, frozenset[int]]]) -> bool:
        # the axis is split into panels only when two or more charts label its units; one title over every
        # column (or row) is a caption, not a panel
        named = {k for t in tags for o in t.values() if len(o) == 1 for k in o}
        return len(named) >= 2

    if not panels(rows):
        rows = [{} for _ in rows]
    if not panels(cols):
        cols = [{} for _ in cols]
    out: dict[int, set[tuple[int, int]]] = {}
    for i, ref in enumerate(refs):
        labels = [*ref.cats, *ref.names]

        def foreign(tags: dict[str, frozenset[int]], ref_labels: Sequence[str] = labels, me: int = i) -> bool | None:
            """True: another chart's panel; False: this chart's; None: no panel label."""
            hit = None
            for e, o in tags.items():
                if me in o:
                    return False
                if not any(label_bind(lab, e) >= 0.995 for lab in ref_labels if _js_trim(lab)):
                    hit = True
            return hit

        rf = [foreign(t) for t in rows]
        cf = [foreign(t) for t in cols]
        bad = {
            (r, j)
            for r in range(len(g.rows))
            for j in range(len(g.hchain))
            if (rf[r] or cf[j]) and rf[r] is not False and cf[j] is not False
        }
        if bad:
            out[i] = bad
    return out


def score_chart_tables(
    md: str, charts: Sequence[ChartSpec | Mapping[str, Any]], rules: Sequence[ChartRule] = ()
) -> ChartNextResult:
    """Full mode: every reference chart against the output grid that fits it best.

    Charts are placed greedily, best fit first. A grid can hold several charts (merged panels, a table with
    one column per single-series chart); the cells one chart reads are excluded for the next, and every
    chart is finally re-read with the cells of the others excluded, so a panel is never charged as invented
    in its neighbour.
    """
    specs = [c if isinstance(c, ChartSpec) else ChartSpec.from_dict(c) for c in charts]
    refs = [_ChartRef.of(c) for c in specs if _ref_points(c)[0]]
    for i, r in enumerate(refs):
        r.others = tuple(o.spec.title for k, o in enumerate(refs) if k != i and _js_trim(o.spec.title))
        r.page_cells = sum(o.n_required for o in refs)
        r.other_labels = tuple(x for k, o in enumerate(refs) if k != i for x in (*o.cats, *o.names) if _js_trim(x))
        _mark_axis_read(r, rules)
        if r.merged is not None:
            r.merged.others, r.merged.page_cells, r.merged.other_labels = r.others, r.page_cells, r.other_labels
            _mark_axis_read(r.merged, rules)
    grids = output_grids(md)
    page_lines = [_js_trim(x) for x in md.split("\n") if _js_trim(x) and "|" not in x and "<t" not in x]
    claimed: dict[int, dict[int, set[tuple[int, int]]]] = {j: {} for j in range(len(grids))}
    foreign = [_foreign_cells(refs, g) for g in grids]
    # a joined grid copies the cells of its tables: a cell one chart reads in either is taken for the others
    links: dict[int, list[tuple[tuple[int, int], int, tuple[int, int]]]] = {j: [] for j in range(len(grids))}
    for j, g in enumerate(grids):
        for pos, (j2, r2, c2) in g.src.items():
            links[j].append((pos, j2, (r2, c2)))
            links[j2].append(((r2, c2), j, pos))

    def others(j: int, i: int) -> set[tuple[int, int]]:
        out = {c for k, s in claimed[j].items() if k != i for c in s} | foreign[j].get(i, set())
        if links[j]:
            taken = {j2: {c for k, s in claimed[j2].items() if k != i for c in s} for j2 in {x[1] for x in links[j]}}
            out |= {pos for pos, j2, pos2 in links[j] if pos2 in taken[j2]}
        return out

    cache: dict[tuple[int, int, int], _Fit] = {}

    def fit(i: int, j: int) -> _Fit:
        ex = others(j, i)
        key = (i, j, hash(frozenset(ex)))
        if key not in cache:
            cache[key] = _fit_chart(refs[i], grids[j], page_lines, ex)
        return cache[key]

    place: dict[int, int] = {}
    remaining = set(range(len(refs)))
    while remaining and grids:
        best = None
        for i in remaining:
            for j in range(len(grids)):
                f = fit(i, j)
                if not f.bound:
                    continue
                k = (f.key, f.bound)
                if best is None or k > best[0]:
                    best = (k, i, j)
        if best is None:
            break
        _, i, j = best
        place[i] = j
        claimed[j][i] = fit(i, j).claimed
        remaining.discard(i)
    # re-read each placed chart with the cells of its grid neighbours excluded (stable after two rounds)
    for _ in range(2):
        for i, j in place.items():
            claimed[j][i] = fit(i, j).claimed

    # a leftover table under a placed chart's title that reads that chart again is a second version of it: its
    # values are scored too (precision, and one charge per wrong cell), so a wrong copy beside a right one costs
    used = {t for j in place.values() for t in grids[j].tables}
    copies: list[_Fit] = []
    for j, g in enumerate(grids):
        if g.tables & used:
            continue
        # the placed chart the grid reads best (a table of a neighbour chart is that chart's copy)
        best_copy: tuple[float, int, _Fit] | None = None
        for i in place:
            f = _fit_chart(refs[i], g, page_lines, foreign[j].get(i, set()))
            if f.bound and (best_copy is None or f.key > best_copy[0]):
                best_copy = (f.key, i, f)
        if best_copy is None:
            continue
        _, i, f = best_copy
        title = refs[i].spec.title
        if _js_trim(title) and _title_found(title, g.context) and _restates(refs[i], f):
            copies.append(_copy_charges(f, fit(i, place[i]), refs[i].tol))
            used |= g.tables

    ev = ChartNextEvents()
    tot = 0.0
    n_ref = n_out = passed = 0
    values: list[float] = []
    labels: list[float] = []
    titles = 0
    placed: list[_Fit] = []
    n_missing = 0
    n_cells = 0
    for i, ch in enumerate(refs):
        jp = place.get(i)
        if jp is None:
            n_ref += ch.n_required
            ev.add("critical", "chart missing", ch.spec.title or f"chart {i + 1}", "")
            n_missing += 1
            titles += _js_trim(ch.spec.title) == "" or _title_found(ch.spec.title, page_lines)
            continue
        j = jp
        f = fit(i, j)
        n_ref += f.n_ref
        tot += f.tot
        n_out += f.n_out
        passed += f.passed
        values += f.values
        labels += f.labels
        titles += f.title_ok
        n_cells += len(f.values)
        placed.append(f)
        for sev in ("critical", "major", "minor"):
            setattr(ev, sev, getattr(ev, sev) + getattr(f.events, sev))
        ev.examples += f.events.examples
    tot_out = tot
    for f in copies:
        tot_out += f.tot
        n_out += f.n_out
        placed.append(f)
        for sev in ("critical", "major", "minor"):
            setattr(ev, sev, getattr(ev, sev) + getattr(f.events, sev))
        ev.examples += f.events.examples
    pr = tot_out / n_out if n_out else 0.0
    rc = tot / n_ref if n_ref else 0.0
    prop = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    penalty, penalty_minor = _first_event_full(placed, {"critical": n_missing})
    penalty += FLOOR["critical"] * n_missing
    floor = _clamp01(1 - penalty - min(FLOOR["minor_cap"], penalty_minor))
    return ChartNextResult(
        score=min(prop, floor),
        proportional=prop,
        floor=floor,
        mode="tables",
        recall=rc,
        precision=pr if n_out else None,
        values=sum(values) / len(values) if values else None,
        labels=sum(labels) / len(labels) if labels else None,
        strict_pass=passed / n_ref if n_ref else 0.0,
        lenient=None,
        title=titles / len(refs) if refs else None,
        scale_only=0,
        n_points=n_ref,
        n_cells=n_cells,
        n_charts=len(refs),
        charts_found=len(place),
        events=ev,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def chart_tables_of(metadata: Mapping[str, Any] | None) -> list[ChartSpec]:
    """Reference charts from test-case metadata ``chart_tables`` (empty when absent)."""
    raw = (metadata or {}).get("chart_tables") or []
    if isinstance(raw, Mapping):
        raw = [raw]
    return [ChartSpec.from_dict(c) for c in raw if isinstance(c, Mapping)]


def score_chart_next(
    md: str,
    chart_tables: Sequence[ChartSpec | Mapping[str, Any]] | None = None,
    rules: Iterable[Any] | None = None,
) -> ChartNextResult | None:
    """Score parser output against full ``chart_tables`` when given, else the sparse ``chart_data_point`` rules.

    Returns ``None`` when the page has neither (nothing to score).
    """
    if chart_tables:
        res = score_chart_tables(md, chart_tables, chart_rules(rules or ()))
        if res.n_points:
            return res
    sparse = chart_rules(rules or ())
    if not sparse:
        return None
    return score_chart_rules(md, sparse)


LANES = ("recall", "precision", "values", "labels", "strict_pass", "title")


def chart_next_metrics(
    md: str,
    chart_tables: Sequence[ChartSpec | Mapping[str, Any]] | None = None,
    rules: Iterable[Any] | None = None,
) -> list[MetricValue]:
    """``chart_next`` headline plus its lane diagnostics as metric values (empty when nothing to score)."""
    res = score_chart_next(md, chart_tables, rules)
    if res is None:
        return []
    metrics = [MetricValue(metric_name="chart_next", value=res.score, metadata=res.metadata())]
    for lane in LANES:
        value = getattr(res, lane)
        if value is not None:
            metrics.append(MetricValue(metric_name=f"chart_next_{lane}", value=float(value)))
    return metrics
