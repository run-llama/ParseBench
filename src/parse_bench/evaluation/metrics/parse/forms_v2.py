"""Form field metric (``forms_v2``): form fields read format-neutrally, scored with severity floors.

Reference: the form's FormNode tree (``metadata.forms.gt``), flattened in document order into scored items
(text fields, checkboxes / select options, signatures) and anchors (section captions, select captions,
table column captions and row headers, static text). A case without a tree is scored from its
``form_field`` rules (a sparse, spot reference: label, value or box state).

Output: the page Markdown, whatever its syntax. Tables (pipe or HTML) become a grid of cells, every other
line one cell; markup, links and tags are dropped; ``[x]``, ``[ ]``, ``(X)``, ``( )``, ``☒``, ``☐``, ``●``,
``:selected:``, ``<input type=checkbox checked>`` and their variants become a checked or unchecked box token. Words
become case- and accent-folded tokens.

Matching is answer-independent. Every reference caption (field label, option label, anchor) is located as
a phrase of output tokens (exact, or at least 60% of its token weight in order; one- and two-letter words
weigh a quarter), then one occurrence per caption is chosen by a max-weight chain in reading order
(repeated labels, "Yes" / "No", go to their occurrences in order), and captions the chain leaves out take
their closest unused exact occurrence (reordered blocks). A value is then READ where the output put it:
the rest of the caption's cell (a printed hint or label tail before a colon is skipped), else the next cell
of a table row, else the cell below, else the next line; reading stops at the next located caption or box. A
caption opening a table cell whose rest is a printed instruction also reads the next non-empty cell of its row.
A value written before its caption (the line above, or inside a bracket the caption closes) is taken only when
it matches.
A box is the glyph before its option label (after it, when the cell writes ``Label [ ]``), a state word
after the label, or the option listed after its select caption. Table cells are read through the grid the
output table forms: columns by their located captions, rows by located row headers or an order-preserving
alignment of row contents (order-free when that matches far better: rows reordered, a minor event). The value is
compared only after it was read from its own slot, so a value swapped into another field is wrong, not found. Values
compare type-aware: amounts as numbers when the reference writes separators, dates as (year, month, day), also with
month names.

Every scored item is one atom; each failure is one event with a severity class:

* critical (0.10 floor, atom loss 1.0): a wrong value (any digit changed, an amount that lost its decimal
  point or was cut short, or text not recognisably the same), a value that belongs to another field, a value
  invented in a blank field, a box state flipped (one per single-select group, every option ticked included),
  a signature claimed on an unsigned line.
* major (0.05, 0.5): a filled field missing or its value dropped (or left without its label, or replaced by its
  caption or an ``[illegible]`` marker), a value truncated, an amount rounded, a checked box whose state is lost
  or that is missing, a single choice shown with no box ticked, a signature lost.
* minor (0.01 each, capped at 0.15; 0.2): a typo in a text value, extra text in a value, a blank field
  omitted, an unchecked option omitted or its box dropped, a garbled caption, a form table not kept as a
  grid or its rows reordered.
* harmless: syntax (key: value, table, HTML, list, JSON-ish), bold/underline, caption punctuation and case,
  printed ids, blanks written as ``___`` / ``N/A`` / ``[blank]``, box glyph variants and state words, amount
  formatting (currency, grouping, decimal style, trailing zeros), date formats and month names, field order,
  unchecked options omitted from a single-select whose choice is shown.

Headline: ``min(proportional, floor)``. Proportional = 1 - sum(atom losses) / (atoms + PRIOR_ATOMS). Floor =
1 - 0.10 * fix(critical) - 0.05 * fix(major) - min(0.15, 0.01 * minor), the first event in full and later
ones scaled by min(1, FLOOR_K / atoms). A minor or major event kind that hits every instance of its kind
(3+) is a systematic quirk, charged once. Coverage is the better of the share of reference items located and
the share of content items (a value, a signature, a ticked box) located; blank table cells not located while a
caption of their table is leave the count. Below one half the score is capped at 2 x coverage. Lanes:
``filled`` (filled text fields and signed lines), ``empty`` (blank fields and unsigned lines), ``checkboxes``.

Regexes use explicit character classes so Python ``re`` and JavaScript (``u`` flag) behave the same.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from parse_bench.schemas.evaluation import MetricValue

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FLOOR = {"critical": 0.10, "major": 0.05, "minor": 0.01, "minor_cap": 0.15}
LOSS = {"critical": 1.0, "major": 0.5, "minor": 0.2, "ok": 0.0}
PRIOR_ATOMS = 3
FLOOR_K = 10
SYSTEMATIC_MIN = 3
COVERAGE_FREE = 0.5
LOCATE_Q = 0.6  # share of a caption's token weight that must be found, in order, to locate it
CLEAN_Q = 0.85  # below this (but located) the caption is garbled: a minor event
WEAK_WEIGHT = 0.25  # one- and two-letter words (de, of, s, no)
WINDOW_SLACK = 2  # a caption occurrence may have this many extra tokens inside it
PROSE_WORDS = 4  # a blank field read as this many words without digits: printed text (minor)
TYPO_EDITS = 0.15  # text values within this share of edits (at least one) are a typo
TRUNC_RATIO = 0.5  # a number read as a prefix of at least this share of its digits is truncated, not wrong
GLYPH_GAP = 4  # tokens allowed between a box and its option label
CONT_LINES = 2  # continuation lines read for a multi-line value
MAX_LABEL_TAIL = 12  # tokens of printed label tail skipped before a colon
PROSE_CELL_WORDS = 12  # a caption cell's rest this long, mostly words, is a printed instruction ...
PROSE_CELL_DIGITS = 0.25  # ... when at most this share of its tokens hold digits
PASS2_MIN_CHARS = 4  # a caption re-located out of order must have this many characters
PASS2_Q = 0.98  # and be found (nearly) exactly
CAPTION_BONUS = 0.05  # chain preference for caption-like occurrences (cell start, colon after)
SHORT_CAPTION = 2.0  # captions lighter than this (one or two words) ...
PROSE_TOKENS = 12  # ... found inside a line longer than this, not caption-like ...
PROSE_FACTOR = 0.3  # ... weigh this much in the chain
FUZZY_TOKEN = 0.5  # credit of a caption word found with one misread letter
REORDER_GAIN = 2.0  # a form table's rows are taken as reordered when matching them out of order gains this much
REORDER_RATIO = 1.5  # ... and this share over the order-preserving match
CORE_Q = 0.99  # weight of an occurrence that matches the caption without its parenthesised hint

CHECKED = "\u2612"
UNCHECKED = "\u2610"
CHECKED_CHARS = "☑☒✓✔✅■●◉⦿⊠⊗✗✘▣◼⬛"
UNCHECKED_CHARS = "☐□○◯⬜▢◻❏❐❑❒"
SEV_ORDER = {"ok": 0, "minor": 1, "major": 2, "critical": 3}
BLANK_WORDS = {
    "na",
    "n",
    "blank",
    "empty",
    "none",
    "null",
    "nil",
    "notstated",
    "notprovided",
    "notapplicable",
    "notfilled",
    "unfilled",
    "nan",
}
HINT_WORDS = {
    "mm",
    "dd",
    "yy",
    "yyyy",
    "mmddyyyy",
    "ddmmyyyy",
    "mmddyy",
    "month",
    "day",
    "year",
    "jj",
    "aaaa",
    "ft",
    "in",
    "lbs",
    "lb",
    "kg",
    "cm",
    "am",
    "pm",
}
CHECK_WORDS = {"x", "checked", "selected", "true", "ticked", "marked"}
UNCHECK_WORDS = {
    "unchecked",
    "unselected",
    "false",
    "notchecked",
    "notselected",
    "unticked",
    "notticked",
    "unmarked",
    "notmarked",
}
YES_WORDS = {"yes", "y", "si", "oui", "ja"}
NO_WORDS = {"no", "non", "nein"}
# a value replaced by an "unreadable" marker: the parser abstains (a lost value), it does not misread
ILLEGIBLE_WORDS = {"illegible", "unreadable", "unclear", "indecipherable", "handwritten", "handwriting"}
ILLEGIBLE_FILLER = {"text", "value", "entry", "field", "word", "words"}
# month names (English, Spanish, French, German; folded) for dates written with words
_MONTH_NAMES = (
    "january february march april may june july august september october november december",
    "jan feb mar apr may jun jul aug sep oct nov dec",
    "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre",
    "janvier fevrier mars avril mai juin juillet aout septembre octobre novembre decembre",
    "januar februar marz april mai juni juli august september oktober november dezember",
)
MONTHS = {m: i + 1 for names in _MONTH_NAMES for i, m in enumerate(names.split())}
MONTHS.update({"sept": 9, "setiembre": 9, "maerz": 3})
DATE_FILLER = {"de", "of", "the", "le", "du", "del"}

_CJK = re.compile("[\u3005-\u3007\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u0e00-\u0e7f]")
_ENTITIES = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": " ",
    "ndash": "–",
    "mdash": "—",
    "hellip": "…",
    "rsquo": "’",
    "lsquo": "‘",
    "ldquo": "“",
    "rdquo": "”",
    "bull": "•",
    "middot": "·",
    "times": "×",
    "deg": "°",
    "check": "✓",
    "cross": "✗",
    "squ": "□",
    "square": "□",
    "ballot": "☐",
}

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _is_word(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LNM"


def _key_chars(ch: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", ch) if unicodedata.category(c) != "Mn").lower()


@dataclass
class Tok:
    key: str
    start: int  # character span in its cell's text
    end: int
    box: int = 0  # 1 checked, 2 unchecked


def tokenize(text: str) -> list[Tok]:
    out: list[Tok] = []
    key: list[str] = []
    start = 0

    def flush(end: int) -> None:
        if key:
            out.append(Tok("".join(key), start, end))
            key.clear()

    for ci, ch in enumerate(text):
        if ch in CHECKED_CHARS or ch in UNCHECKED_CHARS:
            flush(ci)
            out.append(Tok(CHECKED if ch in CHECKED_CHARS else UNCHECKED, ci, ci + 1, 1 if ch in CHECKED_CHARS else 2))
        elif _CJK.match(ch):
            flush(ci)
            k = _key_chars(ch)
            if k:
                out.append(Tok(k, ci, ci + 1))
        elif _is_word(ch):
            kc = _key_chars(ch)
            if kc and not key:
                start = ci
            key.extend(kc)
        else:
            flush(ci)
    flush(len(text))
    return out


def word_keys(text: str) -> list[str]:
    return [t.key for t in tokenize(clean_text(text)) if not t.box]


def _isdigits(s: str) -> bool:
    return bool(s) and all(ch in "0123456789" for ch in s)


def _weight(key: str) -> float:
    if len(key) <= 2 and not _isdigits(key) and not _CJK.match(key):
        return WEAK_WEIGHT
    return 1.0


# ---------------------------------------------------------------------------
# Markup cleaning (one cell of output, or one caption of the reference)
# ---------------------------------------------------------------------------


def _rx(p: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(p, flags)


# whitespace exactly as JavaScript's \s (Python's str.strip / str.split also take \x1c-\x1f and \x85)
_WS = "\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_WS_EDGES = _rx("^[" + _WS + "]+|[" + _WS + "]+$")
_WS_RUN = _rx("[" + _WS + "]+")


def _strip(text: str) -> str:
    return _WS_EDGES.sub("", text)


_ENTITY = _rx(r"&(#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,7});")
_BR = _rx(r"<br[ \t]*/?>", re.IGNORECASE)
_INPUT = _rx(r"<input(?:[ \t\n][^<>]*)?>", re.IGNORECASE)
_ESCAPE = _rx(r"\\([!-/:-@\[-`{-~])")
_IMAGE = _rx(r"!\[([^\]\n]*)\]\([^)\n]*\)")
_LINK = _rx(r"\[([^\]\n]*)\]\((?:[^()\n]|\([^()\n]*\))*\)")
_BOX_CHECKED = _rx(r"\[[xX✓✔●■✗✘☒☑*][ \t]?\]|\([xX●•✓✔][ \t]?\)|:selected:")
# "( )" is an empty radio button, unless three digits follow (a printed area-code bracket: "( ) 503 555-0175")
_BOX_UNCHECKED = _rx(r"\[[ \t\u00a0]?\]|\([ \t\u00a0]\)(?![ \t*_:]*(?:<[^<>]*>[ \t*_:]*)*[0-9]{3})|:unselected:")
_AUTOLINK = _rx(r"<((?:[a-z][a-z0-9+.-]*:|www\.)[^<> \t\n]+|[^<> \t\n@]+@[^<> \t\n@]+\.[^<> \t\n@]+)>", re.IGNORECASE)
_TAG = _rx(r"</?[A-Za-z][^<>]*>")
_HTML_COMMENT = _rx(r"<!--[\s\S]*?-->")
_FILL = _rx(r"_{3,}|\.{4,}|-{3,}|…{2,}")
_PAREN = _rx(r"\((?:[^()]|\([^()]*\))*\)|\[[^\[\]]*\]")
_PAREN_ALL = _rx(r"^(?:\((?:[^()]|\([^()]*\))*\)|\[[^\[\]]*\])(?![\s\S])")
_LEAD_HINT = _rx(r"[ \t:*_]*(\((?:[^()]|\([^()]*\))*\))")
# a superscript number written as a tag keeps its superscript form, so a footnote mark stays recognisable
_SUP_NUM = _rx(r"<sup(?:[ \t\n][^<>]*)?>[ \t]*([0-9]{1,3})[ \t]*</sup[ \t]*>", re.IGNORECASE)
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _entity(m: re.Match[str]) -> str:
    body = m.group(1)
    if body.startswith("#"):
        try:
            code = int(body[2:], 16) if body[1:2] in ("x", "X") else int(body[1:])
            return chr(code) if 0 < code < 0x110000 else ""
        except ValueError:
            return m.group(0)
    return _ENTITIES.get(body.lower(), m.group(0))


def _input_box(m: re.Match[str]) -> str:
    tag = m.group(0).lower()
    if "checkbox" not in tag and "radio" not in tag:
        return " "
    return " " + (CHECKED if re.search(r"[ \t\n]checked", tag) else UNCHECKED) + " "


def clean_text(text: str) -> str:
    """Markup-free text of one cell: entities, escapes, links, images, box notations, tags."""
    s = _ENTITY.sub(_entity, text)
    s = _BR.sub(" ", s)
    s = _INPUT.sub(_input_box, s)
    s = _ESCAPE.sub(lambda m: m.group(1), s)
    s = _IMAGE.sub(lambda m: " " + m.group(1) + " ", s)
    s = _LINK.sub(lambda m: m.group(1), s)
    s = _BOX_CHECKED.sub(" " + CHECKED + " ", s)
    s = _BOX_UNCHECKED.sub(" " + UNCHECKED + " ", s)
    s = _AUTOLINK.sub(lambda m: " " + m.group(1) + " ", s)
    s = _SUP_NUM.sub(lambda m: " " + m.group(1).translate(_SUPERSCRIPT) + " ", s)
    s = _TAG.sub(" ", s)
    return s


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    text: str
    line: int
    table: int = -1
    row: int = 0
    col: int = 0
    fill: bool = False  # a fill-in line or cell with no words (____)
    rowspanned: bool = False  # covered by the rowspan of an HTML cell above (no text of its own)
    colspanned: bool = False  # a slot a colspan cell to its left covers
    first: int = 0  # first token index
    last: int = -1  # last token index (inclusive); last < first = no tokens


@dataclass
class Out:
    toks: list[Tok]
    tcell: list[int]  # cell of each token
    cells: list[Cell]
    grids: list[list[list[int]]]  # per output table: rows of cell indices


_HTML_TABLE = _rx(r"<table(?:[ \t\n][^>]*)?>([\s\S]*?)</table[ \t]*>", re.IGNORECASE)
_TR = _rx(r"<tr(?:[ \t\n][^>]*)?>", re.IGNORECASE)
_TD = _rx(
    r"<(td|th)((?:[ \t\n][^>]*)?)>([\s\S]*?)(?=<td(?:[ \t\n][^>]*)?>|<th(?:[ \t\n][^>]*)?>|</tr|(?![\s\S]))",
    re.IGNORECASE,
)
_TD_CLOSE = _rx(r"</t[dh][ \t]*>", re.IGNORECASE)
_COLSPAN = _rx(r"colspan[ \t]*=[ \t]*[\"']?([0-9]+)", re.IGNORECASE)
_ROWSPAN = _rx(r"rowspan[ \t]*=[ \t]*[\"']?([0-9]+)", re.IGNORECASE)
_PIPE_ROW = _rx(r"^[ \t]*\|.*\|[ \t]*$")
_PIPE_SEP = _rx(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$")
_PIPE_SPLIT = _rx(r"(?<!\\)\|")
_TABLE_MARK = _rx("^\u0000T([0-9]+)\u0000$")
_ROWSPANNED = "\u0000R"  # placeholder text of a slot covered by a rowspan from above
_COLSPANNED = "\u0000S"  # text of a slot covered by a colspan cell to its left


def _html_rows(inner: str) -> list[list[str]]:
    """Rows of cell texts; a spanned cell keeps its text in its first slot and leaves the others empty, so the
    cells of later rows stay under their own columns. Slots a colspan covers hold ``_COLSPANNED``, slots
    a rowspan from above covers hold ``_ROWSPANNED``."""
    rows: list[list[str]] = []
    pending: dict[int, int] = {}  # column -> rows a rowspan above still covers
    for part in _TR.split(inner)[1:]:
        row: list[str] = []
        spent: set[int] = set()

        def fill_spanned(row: list[str], spent: set[int]) -> None:
            while pending.get(len(row), 0) > 0:
                pending[len(row)] -= 1
                spent.add(len(row))
                row.append(_ROWSPANNED)

        for m in _TD.finditer(part):
            fill_spanned(row, spent)
            text = _TD_CLOSE.sub(" ", m.group(3))
            cspan = _COLSPAN.search(m.group(2) or "")
            rspan = _ROWSPAN.search(m.group(2) or "")
            width = min(20, int(cspan.group(1))) if cspan else 1
            for k in range(width):
                if rspan and int(rspan.group(1)) > 1:
                    pending[len(row)] = min(50, int(rspan.group(1))) - 1
                    spent.add(len(row))
                row.append(text if k == 0 else _COLSPANNED)
        fill_spanned(row, spent)
        for col in pending:
            if col not in spent and pending[col] > 0:
                pending[col] -= 1  # a short or overlapping row still spends the span above it
        if row:
            rows.append(row)
    return rows


def read_output(md: str) -> Out:
    """The output Markdown as cells (table cells, other lines) and folded tokens."""
    md = _HTML_COMMENT.sub(" ", (md or "").replace("\r\n", "\n").replace("\r", "\n"))
    html_tables: list[list[list[str]]] = []

    def stash(m: re.Match[str]) -> str:
        html_tables.append(_html_rows(m.group(1)))
        return "\n\u0000T" + str(len(html_tables) - 1) + "\u0000\n"

    md = _HTML_TABLE.sub(stash, md)
    md = _BR.sub("\n", md)
    cells: list[Cell] = []
    grids: list[list[list[int]]] = []
    line_no = 0

    def add_grid(rows: list[list[str]]) -> None:
        nonlocal line_no
        tid = len(grids)
        grid: list[list[int]] = []
        for r, row in enumerate(rows):
            idxs = []
            for c, text in enumerate(row):
                idxs.append(len(cells))
                rowspanned, colspanned = text == _ROWSPANNED, text == _COLSPANNED
                cells.append(
                    Cell(
                        text="" if rowspanned or colspanned else text,
                        line=line_no,
                        table=tid,
                        row=r,
                        col=c,
                        rowspanned=rowspanned,
                        colspanned=colspanned,
                    )
                )
            grid.append(idxs)
            line_no += 1
        grids.append(grid)

    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        tm = _TABLE_MARK.match(line)
        if tm:
            add_grid(html_tables[int(tm.group(1))])
            i += 1
            continue
        if _PIPE_ROW.match(line):
            rows: list[list[str]] = []
            while i < len(lines) and (_PIPE_ROW.match(lines[i]) or _PIPE_SEP.match(lines[i])):
                if not _PIPE_SEP.match(lines[i]):
                    parts = _PIPE_SPLIT.split(_strip(lines[i]))
                    if parts and _strip(parts[0]) == "":
                        parts = parts[1:]
                    if parts and _strip(parts[-1]) == "":
                        parts = parts[:-1]
                    rows.append(parts)
                i += 1
            add_grid(rows)
            continue
        cells.append(Cell(text=line, line=line_no))
        line_no += 1
        i += 1
    toks: list[Tok] = []
    tcell: list[int] = []
    for ci, cell in enumerate(cells):
        cell.text = clean_text(cell.text)
        ts = tokenize(cell.text)
        cell.first = len(toks)
        cell.last = len(toks) + len(ts) - 1
        if not any(not t.box for t in ts) and _FILL.search(cell.text):
            cell.fill = True
        for t in ts:
            toks.append(t)
            tcell.append(ci)
    return Out(toks=toks, tcell=tcell, cells=cells, grids=grids)


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------


@dataclass
class Item:
    kind: str  # text | box | sig | anchor
    label: str = ""
    value: str | None = None  # expected text (None = blank)
    state: bool | None = None  # expected box state / signed
    pid: str = ""  # printed id
    group: int = -1  # select group (options and the group's caption anchor)
    single: bool = False  # option of a single-select group
    table: int = -1
    row: int = -1
    col: int = -1
    page: int = 1
    ctx: str = ""  # section path, for explanations
    role: str = ""  # anchor role: section | group | caption | rowhead | static | context
    phrase: bool = True  # whether the label is located as a phrase (table cells inherit captions: no)


@dataclass
class RefTable:
    ncols: int
    nrows: int
    captions: list[int]  # item index of each column caption anchor (-1 = no caption)
    rowheads: list[int]  # item index of each row header anchor (-1 = none)


@dataclass
class Reference:
    items: list[Item]
    tables: list[RefTable]
    source: str  # tree | rules
    unscorable: int = 0


def _is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return not word_keys(v) and not v.strip(" \t\n_.-–—/")
    return False


def _value_text(v: Any) -> str | None:
    if _is_empty_value(v):
        return None
    if isinstance(v, bool):
        return "Yes" if v else "No"
    return str(v)


def reference_from_tree(pages: Sequence[Any], excluded_pages: Sequence[Any] = ()) -> Reference:
    """Scored items and anchors of a FormNode tree, in document order."""
    items: list[Item] = []
    tables: list[RefTable] = []
    excluded = {int(p) for p in excluded_pages if isinstance(p, int) or (isinstance(p, str) and p.isdigit())}
    uns = [0]
    group_no = [0]

    def anchor(label: Any, page: int, ctx: str, role: str, **kw: Any) -> int:
        text = _strip(str(label)) if label is not None else ""
        if not word_keys(text):
            return -1
        items.append(Item(kind="anchor", label=text, page=page, ctx=ctx, role=role, **kw))
        return len(items) - 1

    def node_value(node: dict[str, Any]) -> Any:
        return None if node.get("isEmpty") is True else node.get("value")

    def add_field(node: dict[str, Any], page: int, ctx: str, inherited: str = "", **kw: Any) -> None:
        kind = node.get("field") or "text"
        own = _strip(str(node.get("label"))) if node.get("label") is not None else ""
        label = own or inherited
        pid = "" if node.get("id") is None else str(node.get("id"))
        if kind in ("single_select", "multi_select"):
            group_no[0] += 1
            gid = group_no[0]
            anchor(own, page, ctx, "group", group=gid, pid=pid, **kw)
            for vi in node.get("valueItems") or []:
                if not isinstance(vi, dict):
                    continue
                members = vi.get("items") if vi.get("type") in ("section", "option_group") else [vi]
                for m in members or []:
                    if not isinstance(m, dict):
                        continue
                    mlabel = m.get("label") if m.get("label") is not None else vi.get("label")
                    mm = dict(m, label=mlabel)
                    if (m.get("field") or "checkbox") == "checkbox" or m.get("type") == "option":
                        add_box(mm, page, ctx, gid, kind == "single_select", **kw)
                    else:
                        add_field(mm, page, ctx, **kw)
            return
        if kind == "checkbox":
            add_box(node, page, ctx, -1, False, inherited=inherited, **kw)
            return
        has_phrase = bool(own) and bool(word_keys(own))
        if not has_phrase and kw.get("table", -1) < 0:
            uns[0] += 1
            return
        v = node_value(node)
        if kind == "signature":
            signed = bool(v) and not _is_empty_value(v) if not isinstance(v, bool) else v
            items.append(
                Item(kind="sig", label=label, state=signed, pid=pid, page=page, ctx=ctx, phrase=has_phrase, **kw)
            )
            return
        items.append(
            Item(kind="text", label=label, value=_value_text(v), pid=pid, page=page, ctx=ctx, phrase=has_phrase, **kw)
        )

    def add_box(
        node: dict[str, Any], page: int, ctx: str, gid: int, single: bool, inherited: str = "", **kw: Any
    ) -> None:
        own = _strip(str(node.get("label"))) if node.get("label") is not None else ""
        v = node.get("value")
        if not isinstance(v, bool) or (not word_keys(own) and kw.get("table", -1) < 0):
            uns[0] += 1
            return
        items.append(
            Item(
                kind="box",
                label=own or inherited,
                state=v,
                pid="" if node.get("id") is None else str(node.get("id")),
                group=gid,
                single=single,
                page=page,
                ctx=ctx,
                phrase=bool(own) and bool(word_keys(own)),
                **kw,
            )
        )

    def walk(nodes: Any, page: int, ctx: str) -> None:
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            t = node.get("type")
            if t == "section":
                label = " ".join(str(x) for x in (node.get("id"), node.get("label")) if x is not None)
                anchor(label, page, ctx, "section")
                sub = (ctx + " > " if ctx else "") + _strip(str(node.get("label") or node.get("id") or ""))
                walk(node.get("items"), page, sub if sub.strip(" >") else ctx)
            elif t == "static_text":
                anchor(node.get("text") or node.get("label") or node.get("value"), page, ctx, "static")
            elif t == "table":
                walk_table(node, page, ctx)
            elif t == "field":
                add_field(node, page, ctx)

    def walk_table(node: dict[str, Any], page: int, ctx: str) -> None:
        tid = len(tables)
        caption = " ".join(str(x) for x in (node.get("id"), node.get("label")) if x is not None)
        anchor(caption, page, ctx, "section")
        cols = [str(c) if isinstance(c, str) else "" for c in (node.get("columns") or [])]
        rows = [r for r in (node.get("rows") or []) if isinstance(r, list)]
        ncols = max([len(cols)] + [len(r) for r in rows]) if (cols or rows) else 0
        rt = RefTable(ncols=ncols, nrows=len(rows), captions=[], rowheads=[])
        tables.append(rt)
        for c in range(ncols):
            rt.captions.append(anchor(cols[c] if c < len(cols) else "", page, ctx, "caption", table=tid, col=c))
        # column 0 holds row headers when it is all text and its caption is empty, or every row's
        # first cell is printed text while other cells are fill-ins
        head = bool(rows) and all(isinstance(r[0], str) for r in rows if r) and (not cols or not _strip(cols[0]))
        for r, row in enumerate(rows):
            rh = anchor(row[0], page, ctx, "rowhead", table=tid, row=r, col=0) if head and row else -1
            rt.rowheads.append(rh)
            for c, cell in enumerate(row):
                if head and c == 0:
                    continue
                inherited = _strip(cols[c]) if c < len(cols) else ""
                if head and row and isinstance(row[0], str):
                    inherited = (_strip(str(row[0])) + " / " + inherited).strip(" /")
                pos: dict[str, Any] = {"table": tid, "row": r, "col": c}
                if isinstance(cell, dict):
                    for m in cell.get("items") or []:
                        if isinstance(m, dict) and m.get("type") == "field":
                            add_field(m, page, ctx, inherited=inherited, **pos)
                        elif isinstance(m, dict) and m.get("type") == "static_text":
                            anchor(m.get("text") or m.get("label"), page, ctx, "static", **pos)
                elif cell is None or isinstance(cell, (str, int, float)):
                    if not inherited:
                        uns[0] += 1
                        continue
                    items.append(
                        Item(
                            kind="text",
                            label=inherited,
                            value=_value_text(cell),
                            page=page,
                            ctx=ctx,
                            phrase=False,
                            **pos,
                        )
                    )

    for pg in pages or []:
        if not isinstance(pg, dict):
            continue
        page = int(pg.get("page_number") or 1)
        if page in excluded:
            continue
        walk(pg.get("items"), page, "")
    # a caption-less item is read from its table cell: only when it is the cell's only scored item
    per_cell: dict[tuple[int, int, int], int] = {}
    for it in items:
        if it.kind != "anchor" and it.table >= 0:
            key = (it.table, it.row, it.col)
            per_cell[key] = per_cell.get(key, 0) + 1
    kept: list[Item] = []
    remap: dict[int, int] = {}
    for x, it in enumerate(items):
        if it.kind != "anchor" and it.table >= 0 and not it.phrase and per_cell[(it.table, it.row, it.col)] > 1:
            uns[0] += 1
            continue
        remap[x] = len(kept)
        kept.append(it)
    for rt in tables:
        rt.captions = [remap.get(x, -1) if x >= 0 else -1 for x in rt.captions]
        rt.rowheads = [remap.get(x, -1) if x >= 0 else -1 for x in rt.rowheads]
    items = kept
    return Reference(items=items, tables=tables, source="tree", unscorable=uns[0])


def reference_from_rules(rules: Sequence[Any] | None) -> Reference | None:
    """A sparse reference from ``form_field`` rules (label or [context..., label], value, value_type)."""
    items: list[Item] = []
    last_ctx: list[str] = []
    for rule in rules or []:
        get = (lambda k, r=rule: r.get(k)) if isinstance(rule, dict) else (lambda k, r=rule: getattr(r, k, None))
        if get("type") != "form_field":
            continue
        label = get("label")
        parts = [_strip(str(p)) for p in (label if isinstance(label, list) else [label]) if p is not None]
        parts = [p for p in parts if p]
        if not parts:
            continue
        page = int(get("page") or 1)
        ctx = parts[:-1]
        if ctx != last_ctx:
            for c in ctx:
                if c not in last_ctx and word_keys(c):
                    items.append(Item(kind="anchor", label=c, page=page, role="context"))
            last_ctx = ctx
        vt = get("value_type") or "text"
        v = get("value")
        if vt == "checkbox":
            if isinstance(v, bool):
                items.append(Item(kind="box", label=parts[-1], state=v, page=page, ctx=" > ".join(ctx)))
        elif vt == "signature":
            items.append(Item(kind="sig", label=parts[-1], state=bool(v), page=page, ctx=" > ".join(ctx)))
        else:
            items.append(Item(kind="text", label=parts[-1], value=_value_text(v), page=page, ctx=" > ".join(ctx)))
    if not any(it.kind != "anchor" for it in items):
        return None
    return Reference(items=items, tables=[], source="rules")


def forms_reference(metadata: Any, rules: Sequence[Any] | None = None) -> Reference | None:
    """The forms_v2 reference of a test case: its FormNode tree, else its form_field rules."""
    meta = metadata if isinstance(metadata, dict) else {}
    forms: dict[str, Any] = meta["forms"] if isinstance(meta.get("forms"), dict) else {}
    gt = forms.get("gt")
    if isinstance(gt, dict) and isinstance(gt.get("pages"), list) and gt["pages"]:
        fb: dict[str, Any] = meta["form_bench"] if isinstance(meta.get("form_bench"), dict) else {}
        excluded = list(forms.get("review_excluded_pages") or []) + list(fb.get("review_excluded_pages") or [])
        ref = reference_from_tree(gt["pages"], excluded)
        if any(it.kind != "anchor" for it in ref.items):
            return ref
        return None
    return reference_from_rules(rules)


# ---------------------------------------------------------------------------
# Locating captions
# ---------------------------------------------------------------------------


@dataclass
class Occ:
    item: int
    start: int
    end: int
    q: float


_SEGMENT = _rx(r"[ \t]+[-–—][ \t]+|:[ \t]+")


def _phrase(label: str) -> tuple[list[str], list[list[str]]]:
    """Caption tokens, and shorter forms a parser may print: without parenthesised hints, the last segment
    of a ``Context - Caption`` label."""
    full = word_keys(label)
    alts: list[list[str]] = []
    for text in (_PAREN.sub(" ", clean_text(label)), _SEGMENT.split(label)[-1]):
        keys = word_keys(text)
        if keys and keys != full and keys not in alts and sum(_weight(k) for k in keys) >= 1.0:
            alts.append(keys)
    return full, alts


def _one_edit(a: str, b: str) -> bool:
    """Whether two different strings are one insertion, deletion or substitution apart."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    if len(a) == len(b):
        return a[i + 1 :] == b[i + 1 :]
    return a[i:] == b[i + 1 :]


def _tok_sim(a: str, b: str) -> float:
    """1 for equal words, FUZZY_TOKEN for long words one edit apart (a misread letter), else 0."""
    if a == b:
        return 1.0
    if len(a) >= 5 and len(b) >= 5 and _one_edit(a, b):
        return FUZZY_TOKEN
    return 0.0


def _window_match(p: Sequence[str], w: Sequence[float], h: Sequence[str]) -> tuple[float, int, int]:
    """Weighted in-order match of phrase ``p`` against window ``h``: (weight, first, last) matched index."""
    n, m = len(p), len(h)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            a = best[i + 1][j]
            b = best[i][j + 1]
            c = best[i + 1][j + 1] + w[i] * _tok_sim(p[i], h[j]) if _tok_sim(p[i], h[j]) > 0 else -1.0
            best[i][j] = max(a, b, c)
    i = j = 0
    first = last = -1
    while i < n and j < m:
        if _tok_sim(p[i], h[j]) > 0 and best[i][j] == best[i + 1][j + 1] + w[i] * _tok_sim(p[i], h[j]):
            if first < 0:
                first = j
            last = j
            i += 1
            j += 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return best[0][0], first, last


def _window_bound(out: Out, cell: int) -> int:
    """End (exclusive) of the tokens a caption starting in ``cell`` may span: its cell, or for a text line
    the lines that follow it (captions wrap)."""
    c = out.cells[cell]
    if c.table >= 0:
        return c.last + 1
    end = c.last + 1
    nxt = cell + 1
    while nxt < len(out.cells) and out.cells[nxt].table < 0:
        end = out.cells[nxt].last + 1
        nxt += 1
    return end


def _starts_cell(out: Out, j: int) -> bool:
    c = out.cells[out.tcell[j]]
    return all(out.toks[t].box for t in range(c.first, j))


def _candidates(
    item_idx: int, phrase: Sequence[str], out: Out, index: dict[str, list[int]], at_start: bool = False
) -> list[Occ]:
    """Occurrences of a caption; ``at_start``: only where a cell or line starts (table row headers)."""
    if not phrase:
        return []
    w = [_weight(k) for k in phrase]
    total = sum(w)
    strong = [a for a in range(len(phrase)) if w[a] == 1.0] or list(range(len(phrase)))
    starts: list[int] = []
    seen: set[int] = set()
    for a in strong:
        for j in index.get(phrase[a], []):
            s = j - a
            cell = out.tcell[j]
            s = max(s, out.cells[cell].first)
            if s not in seen:
                seen.add(s)
                starts.append(s)
    starts.sort()
    cands: list[Occ] = []
    for s in starts:
        e = min(s + len(phrase) + WINDOW_SLACK, _window_bound(out, out.tcell[s]))
        window = [out.toks[x].key if not out.toks[x].box else "\u0001" for x in range(s, e)]
        got, first, last = _window_match(phrase, w, window)
        q = got / total if total else 0.0
        if at_start and first >= 0 and not _starts_cell(out, s + first):
            continue
        if q >= LOCATE_Q - 1e-9 and first >= 0:
            cands.append(Occ(item_idx, s + first, s + last, q))
    # keep the best of overlapping windows
    cands.sort(key=lambda o: (-o.q, o.end - o.start, o.start))
    kept: list[Occ] = []
    for o in cands:
        if all(o.end < k.start or o.start > k.end for k in kept):
            kept.append(o)
    kept.sort(key=lambda o: (o.start, o.end))
    return kept


_AFTER_CAPTION = _rx(r"^[ \t\n*_]*(?::|(?![\s\S]))")


def _caption_bonus(out: Out, o: Occ) -> float:
    """Chain preference for an occurrence that looks like a caption: it starts its cell or line, and it is
    followed by a colon or ends the cell (``Fecha: …`` rather than ``lugar, fecha y firma``)."""
    c = out.cells[out.tcell[o.end]]
    bonus = CAPTION_BONUS if _starts_cell(out, o.start) else 0.0
    tail = c.text[out.toks[o.end].end :]
    if _AFTER_CAPTION.match(tail) or (o.end + 1 <= c.last and out.toks[o.end + 1].box):
        bonus += CAPTION_BONUS
    return bonus


def _chain_weight(out: Out, o: Occ, caption_weight: float) -> float:
    """A short caption found inside running text (a long line, not at its start, no colon after) is likely
    a word of the prose, not the caption: it counts for less in the chain."""
    bonus = _caption_bonus(out, o)
    c = out.cells[out.tcell[o.start]]
    if bonus == 0.0 and caption_weight < SHORT_CAPTION and c.last - c.first + 1 > PROSE_TOKENS:
        return o.q * PROSE_FACTOR
    return o.q + bonus


class _Fenwick:
    """Prefix maximum of (value, -point id) over positions."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.val = [-1.0] * (n + 1)
        self.pid = [-1] * (n + 1)

    @staticmethod
    def _better(v1: float, p1: int, v2: float, p2: int) -> bool:
        if p2 < 0:
            return p1 >= 0
        if p1 < 0:
            return False
        if abs(v1 - v2) > 1e-9:
            return v1 > v2
        return p1 < p2

    def update(self, pos: int, v: float, p: int) -> None:
        i = pos + 1
        while i <= self.n:
            if self._better(v, p, self.val[i], self.pid[i]):
                self.val[i] = v
                self.pid[i] = p
            i += i & -i

    def query(self, pos: int) -> tuple[float, int]:
        """Best over positions 0..pos (pos < 0 = none)."""
        bv, bp = 0.0, -1
        i = pos + 1
        while i > 0:
            if self._better(self.val[i], self.pid[i], bv, bp):
                bv, bp = self.val[i], self.pid[i]
            i -= i & -i
        return bv, bp


def locate(ref: Reference, out: Out) -> tuple[list[Occ | None], list[bool]]:
    """One output occurrence per reference caption (or None), and the tokens they cover."""
    index: dict[str, list[int]] = {}
    for j, t in enumerate(out.toks):
        if not t.box:
            index.setdefault(t.key, []).append(j)
    per_item: list[list[Occ]] = []
    for x, it in enumerate(ref.items):
        if not it.phrase or not it.label:
            per_item.append([])
            continue
        full, alts = _phrase(it.label)
        at_start = it.role == "rowhead" or (it.kind == "anchor" and all(_isdigits(k) for k in full))
        cands = _candidates(x, full, out, index, at_start)
        for alt in alts:
            for o in _candidates(x, alt, out, index, at_start):
                o.q *= CORE_Q  # a shorter form of the caption: a full-caption occurrence wins
                same = [k for k in cands if k.start == o.start and k.end == o.end]
                if not same:
                    cands.append(o)
                elif o.q > same[0].q:
                    same[0].q = o.q
        if alts:
            cands.sort(key=lambda o: (o.start, o.end))
        per_item.append(cands)
    weights = [sum(_weight(k) for k in word_keys(it.label)) if it.phrase else 0.0 for it in ref.items]
    # max-weight chain in reading order
    points: list[Occ] = []
    m = len(out.toks)
    fw = _Fenwick(max(1, m))
    dp: list[float] = []
    prev: list[int] = []
    for cands in per_item:
        batch = []
        for o in cands:
            bv, bp = fw.query(o.start - 1)
            pid = len(points)
            points.append(o)
            dp.append(bv + _chain_weight(out, o, weights[o.item]))
            prev.append(bp)
            batch.append(pid)
        for pid in batch:
            fw.update(points[pid].end, dp[pid], pid)
    best_v, best_p = fw.query(m - 1)
    occ: list[Occ | None] = [None] * len(ref.items)
    used = [False] * m
    p = best_p
    while p >= 0:
        o = points[p]
        occ[o.item] = o
        for x in range(o.start, o.end + 1):
            used[x] = True
        p = prev[p]
    # captions left out of the chain: their closest unused exact occurrence
    last_end = -1
    for x, cands in enumerate(per_item):
        if occ[x] is not None:
            last_end = occ[x].end  # type: ignore[union-attr]
            continue
        short = sum(len(k) for k in word_keys(ref.items[x].label)) < PASS2_MIN_CHARS
        best: Occ | None = None
        best_key = (0.0, 0)
        for o in cands:
            if o.q < PASS2_Q or any(used[y] for y in range(o.start, o.end + 1)):
                continue
            if short and _caption_bonus(out, o) < 2 * CAPTION_BONUS - 1e-9:
                continue  # a short caption (12a, Fax) only where it stands alone as a caption
            key = (-_caption_bonus(out, o), abs(o.start - last_end))  # caption-like first, then the closest
            if best is None or key < best_key:
                best, best_key = o, key
        if best is not None:
            occ[x] = best
            for y in range(best.start, best.end + 1):
                used[y] = True
    _repair_sections(ref, out, per_item, occ, used)
    return occ, used


def _repair_sections(
    ref: Reference, out: Out, per_item: list[list[Occ]], occ: list[Occ | None], used: list[bool]
) -> None:
    """Captions repeated under different sections (Code under 12a, 12b …; Name under Applicant and
    Co-applicant) follow their section when the output moved it: a caption placed outside its section's
    output region is released, then takes an unused exact occurrence inside that region (else it keeps its
    old place when that is still free)."""
    items = ref.items
    heads = [x for x, it in enumerate(items) if it.kind == "anchor" and it.role == "section" and occ[x] is not None]
    starts = sorted(occ[x].start for x in heads)  # type: ignore[union-attr]
    moves: list[tuple[int, int, int, Occ | None]] = []
    for a in heads:
        oa = occ[a]
        assert oa is not None
        lo = oa.end + 1
        hi = next((st for st in starts if st > oa.end), len(out.toks))
        for x in range(a + 1, len(items)):
            if items[x].kind == "anchor" and items[x].role == "section":
                break
            o = occ[x]
            if o is not None and lo <= o.start < hi:
                continue
            if not any(c.q >= PASS2_Q and lo <= c.start < hi for c in per_item[x]):
                continue
            moves.append((x, lo, hi, o))
            if o is not None:
                for y in range(o.start, o.end + 1):
                    used[y] = False
                occ[x] = None
    for x, lo, hi, old in moves:
        for c in per_item[x]:
            if c.q >= PASS2_Q and lo <= c.start < hi and not any(used[y] for y in range(c.start, c.end + 1)):
                occ[x] = c
                break
        if occ[x] is None and old is not None and not any(used[y] for y in range(old.start, old.end + 1)):
            occ[x] = old
        o = occ[x]
        if o is not None:
            for y in range(o.start, o.end + 1):
                used[y] = True


# ---------------------------------------------------------------------------
# Reading values
# ---------------------------------------------------------------------------


@dataclass
class Reading:
    toks: list[int]
    text: str
    weak: bool = False  # an adjacent-line reading: a blank reading is also allowed
    before: bool = False  # written before the caption: taken only when it matches the value (ok or minor)


def _span_text(out: Out, toks: Sequence[int]) -> str:
    if not toks:
        return ""
    parts: list[str] = []
    cur_cell = -1
    seg_start = 0
    seg_end = 0
    for t in toks:
        c = out.tcell[t]
        if c != cur_cell:
            if cur_cell >= 0:
                parts.append(out.cells[cur_cell].text[seg_start:seg_end])
            cur_cell, seg_start = c, out.toks[t].start
        seg_end = out.toks[t].end
    parts.append(out.cells[cur_cell].text[seg_start:seg_end])
    return " / ".join(_strip(p) for p in parts)


def _run(out: Out, start: int, stop: list[bool], cell: int) -> list[int]:
    """Tokens from ``start`` in ``cell`` up to the first stop token."""
    run: list[int] = []
    k = start
    while k < len(out.toks) and out.tcell[k] == cell and not stop[k]:
        run.append(k)
        k += 1
    return run


def _skip_label_tail(out: Out, cell: int, after: int, run: list[int]) -> list[int]:
    """Drop a printed label tail before a colon (no digits, or bracketed) and a leading bracketed hint."""
    if not run:
        return run
    text = out.cells[cell].text
    seg = text[after : out.toks[run[-1]].end]
    colon = seg.find(":")
    if colon > 0:
        cut = after + colon
        head = [t for t in run if out.toks[t].end <= cut]
        head_text = _strip(text[after:cut])
        if (
            head
            and len(head) <= MAX_LABEL_TAIL
            and len(head) < len(run)
            and (not _digits(head_text) or _PAREN_ALL.match(head_text))
        ):
            return [t for t in run if out.toks[t].start > cut]
    lead = text[after:]
    m = _LEAD_HINT.match(lead)
    if m and (not _digits(m.group(1)) or len(word_keys(m.group(1))) >= 3) and "signed" not in m.group(1).lower():
        cut = after + m.end()
        return [t for t in run if out.toks[t].start >= cut]
    return run


_SUPERSCRIPT_DIGITS = frozenset("⁰¹²³⁴⁵⁶⁷⁸⁹")


def _footnote_defined(out: Out, cell: int, key: str) -> bool:
    """A later non-table line opens with ``key`` as a footnote number (``7 Where ...``, ``[^7]: ...``, or ``7`` alone
    above its text); a numbered item (``7.``, ``7)``) does not count."""
    for c in range(cell + 1, len(out.cells)):
        nc = out.cells[c]
        if nc.table >= 0 or nc.last < nc.first or out.toks[nc.first].key != key:
            continue
        t = out.toks[nc.first]
        if any(_is_word(ch) for ch in nc.text[: t.start]) or nc.text[t.end : t.end + 1] in (".", ")"):
            continue
        if nc.last > nc.first or _next_cell(out, c) >= 0:
            return True
    return False


def _drop_footnote_mark(out: Out, cell: int, run: list[int]) -> list[int]:
    """Drop a footnote number right after a caption (``estimate:<sup>7</sup>``, ``estimate:⁷``, ``sentence?8``):
    superscript or glued to the caption, and defined as a footnote further down the output."""
    if not run or run[0] == 0 or out.tcell[run[0] - 1] != cell:
        return run
    t = out.toks[run[0]]
    if not _isdigits(t.key) or len(t.key) > 3:
        return run
    text = out.cells[cell].text
    sup = all(ch in _SUPERSCRIPT_DIGITS for ch in text[t.start : t.end])
    glued = not _WS_RUN.search(text[out.toks[run[0] - 1].end : t.start])
    if (sup or glued) and _footnote_defined(out, cell, t.key):
        return run[1:]
    return run


def _next_cell(out: Out, cell: int) -> int:
    """The next non-table line with tokens or a fill marker (-1 = none)."""
    c = cell + 1
    while c < len(out.cells):
        nc = out.cells[c]
        if nc.table >= 0:
            return -1
        if nc.last >= nc.first or nc.fill:
            return c
        c += 1
    return -1


def _grid_cell(out: Out, table: int, row: int, col: int) -> int:
    grid = out.grids[table]
    if 0 <= row < len(grid) and 0 <= col < len(grid[row]):
        return grid[row][col]
    return -1


def _cell_tokens(out: Out, cell: int) -> list[int]:
    """Every word token of a cell the table grid gives to one reference item."""
    c = out.cells[cell]
    return [t for t in range(c.first, c.last + 1) if not out.toks[t].box]


_SUBFIELD_SEP = _rx(r"[/\-]")


def _split_subfields(out: Out, cell: int, toks: Sequence[int], n: int) -> list[list[int]] | None:
    """A cell the output wrote for ``n`` reference sub-fields sharing one table cell (``DD`` ``MM`` ``YY`` under
    one ``Date of birth`` column), split into one token list per sub-field, in order. Parts are separated by
    ``/`` or ``-`` (``A / D / 2``, ``/ 2 /`` keeps the blank DD and YY); without separators, each word is a part.
    None when the part count is not ``n``: the cell is then read whole."""
    text = out.cells[cell].text
    seps = [m.start() for m in _SUBFIELD_SEP.finditer(text)]
    if len(seps) == n - 1:
        parts: list[list[int]] = [[] for _ in range(n)]
        for t in toks:
            parts[sum(1 for s in seps if s < out.toks[t].start)].append(t)
        return parts
    if not seps and len(toks) == n:
        return [[t] for t in toks]
    return None


def _cell_run(out: Out, cell: int, stop: list[bool], ids_at: dict[int, str] | None = None) -> tuple[list[int], bool]:
    """A whole cell read as a value: (tokens up to the first stop, whether it starts with a stop). A printed id
    right before a caption belongs to it, so a cell holding only that id starts with a stop: in the caption's
    own cell (``2b Unrecap. Sec. 1250 gain``), or alone in the cell before it when it is that caption's printed
    id (``ids_at``: ``| 15 | Credits |``)."""
    c = out.cells[cell]
    if c.last < c.first:
        return [], False
    if stop[c.first]:
        return [], True
    run = _run(out, c.first, stop, cell)
    nxt = run[-1] + 1
    if nxt < len(out.toks) and stop[nxt] and _is_id_run(out, cell, run):
        if out.tcell[nxt] == cell:
            return [], True
        nc = out.cells[out.tcell[nxt]]
        alone = _strip(c.text) == c.text[out.toks[run[0]].start : out.toks[run[-1]].end]
        code = "".join(out.toks[t].key for t in run)
        own_id = bool(ids_at) and ids_at.get(nxt) == code  # type: ignore[union-attr]
        if alone and own_id and out.tcell[nxt] == cell + 1 and nc.table == c.table >= 0 and nc.row == c.row:
            return [], True
    return run, False


def _right_of_prose(
    out: Out,
    occ: Occ,
    run: list[int],
    stop: list[bool],
    owned: set[int],
    neighbours: tuple[bool, bool],
    ids_at: dict[int, str] | None,
) -> list[Reading]:
    """The next cell of the row as a further reading when the caption opens a table cell whose rest (``run``) is a
    printed instruction: ``| 4. Pension income - If you expect ... enter the lesser amount | 1,200 |``. The rest is
    prose when it has at least PROSE_CELL_WORDS tokens and at most PROSE_CELL_DIGITS of them hold digits (the
    instruction may quote amounts). An empty next cell reads as a blank field. Nothing is added when the next cell
    belongs to another reference item or the reference has a field right of this caption."""
    cell = out.tcell[occ.end]
    c = out.cells[cell]
    if c.table < 0 or neighbours[0] or len(run) < PROSE_CELL_WORDS:
        return []
    lead = [t for t in range(c.first, occ.start) if not out.toks[t].box]
    if len(lead) > 2 or not all(_LINE_NO.match(out.toks[t].key) for t in lead):
        return []
    if sum(1 for t in run if _digits(out.toks[t].key)) > PROSE_CELL_DIGITS * len(run):
        return []
    # empty slots right of the caption cell may be its colspan: the value is the next cell with words
    col = c.col + 1
    right = _grid_cell(out, c.table, c.row, col)
    while right >= 0 and out.cells[right].last < out.cells[right].first and right not in owned:
        col += 1
        nxt = _grid_cell(out, c.table, c.row, col)
        if nxt < 0:
            return [Reading([], "")]
        right = nxt
    if right < 0 or right in owned:
        return []
    toks, starts_stop = _cell_run(out, right, stop, ids_at)
    if starts_stop:
        return []
    return [Reading(toks, _span_text(out, toks))]


# a printed box code or line number: 10400, 14, 4a, C, or a dotted id such as 1.3 or 2.10
_BOX_CODE = _rx(r"^(?:[0-9]{1,5}[a-z]?|[a-z]|[0-9]{1,2}(?:\.[0-9]{1,2}){1,2})(?![\s\S])", re.IGNORECASE)


def _is_id_run(out: Out, cell: int, run: list[int]) -> bool:
    """Whether ``run`` is one printed id: a line number (``15``, ``4a``) or a dotted id (``1.3``)."""
    if len(run) == 1:
        return bool(_LINE_NO.match(out.toks[run[0]].key))
    text = out.cells[cell].text[out.toks[run[0]].start : out.toks[run[-1]].end]
    return len(run) <= 3 and "." in text and bool(_BOX_CODE.match(text))


def _code_cell(out: Out, cell: int) -> bool:
    """Whether a table cell holds only printed codes: one or two of ``10400``, ``14``, ``C``, ``1.3``."""
    c = out.cells[cell]
    parts = _strip(c.text).split()
    return c.last >= c.first and 1 <= len(parts) <= 2 and all(_BOX_CODE.match(p) for p in parts)


def _code_column(out: Out, table: int, col: int, stop: list[bool], ids: set[str]) -> str:
    """How a column of printed codes sits in an output table: "" (not one), "boundary" (it opens another
    form column: a caption starts a cell right of it) or "skip" (the codes precede their row's value).
    A code column has at least 3 code cells, and one of its codes is a printed id of the reference."""
    grid = out.grids[table]
    codes = {grid[r][col] for r in range(len(grid)) if col < len(grid[r]) and _code_cell(out, grid[r][col])}
    keys = {"".join(out.toks[t].key for t in range(out.cells[k].first, out.cells[k].last + 1)) for k in codes}
    if len(codes) < 3 or not keys & ids:
        return ""
    for r in range(len(grid)):
        nxt = _grid_cell(out, table, r, col + 1)
        if nxt >= 0 and out.cells[nxt].last >= out.cells[nxt].first:
            first = out.cells[nxt].first
            if stop[first] and not out.toks[first].box:
                return "boundary"
    return "skip"


def _header_row(out: Out, c: Cell, stop: list[bool]) -> bool:
    """Whether the table row of cell ``c`` is a header row: at least two of its cells, and more than half of
    those with words, open with a located caption."""
    filled = [x for x in out.grids[c.table][c.row] if out.cells[x].last >= out.cells[x].first]
    caps = sum(1 for x in filled if stop[out.cells[x].first] and not out.toks[out.cells[x].first].box)
    return caps >= 2 and 2 * caps > len(filled)


# a currency or percent sign or a fill line after a caption in its table cell: that cell is the value slot
_SLOT_MARK = _rx(r"[$€£¥%]|_{3,}|\.{4,}")


def read_slot(
    out: Out,
    occ: Occ,
    stop: list[bool],
    owned: set[int] | None = None,
    neighbours: tuple[bool, bool] = (False, False),
    ids_at: dict[int, str] | None = None,
) -> list[Reading]:
    """Candidate readings of the value that belongs to the caption at ``occ`` (``owned``: output cells the
    table grid gives to other reference items, never read as this caption's neighbour; ``neighbours``: the
    reference itself has another field right of / below this caption's cell; ``ids_at``: printed id of the
    caption located at a token, which is not part of the value before it: ``Name: Ana  **b Date**: ...``)."""
    owned = owned or set()
    cell = out.tcell[occ.end]
    c = out.cells[cell]
    after = out.toks[occ.end].end
    run = _run(out, occ.end + 1, stop, cell)
    run = _skip_label_tail(out, cell, after, run)
    run = _drop_footnote_mark(out, cell, run)
    if run and ids_at and out.toks[run[-1]].key == ids_at.get(run[-1] + 1):
        run = run[:-1]
    if run:
        cands = [Reading(run, _span_text(out, run))]
        cands += _right_of_prose(out, occ, run, stop, owned, neighbours, ids_at)
        last = run[-1]
        if c.table < 0 and last == c.last:
            ext = list(run)
            nxt = cell
            for _ in range(CONT_LINES):
                nxt = _next_cell(out, nxt)
                if nxt < 0:
                    break
                nc = out.cells[nxt]
                span = list(range(nc.first, nc.last + 1))
                if not span or any(stop[t] for t in span):
                    break
                ext = ext + span
                cands.append(Reading(ext, _span_text(out, ext)))
        return cands
    k = occ.end + 1
    if k <= c.last and stop[k]:
        return [Reading([], "")]
    # the caption opens its cell (boxes and up to two printed ids such as "12a" may precede it)
    lead = [t for t in range(c.first, occ.start) if not out.toks[t].box]
    caption_cell = len(lead) <= 2 and all(_LINE_NO.match(out.toks[t].key) for t in lead)
    if c.table >= 0 and not caption_cell:
        return [Reading([], "")]  # a caption inside a cell with other content: its value is in that cell
    if c.table >= 0:
        right = _grid_cell(out, c.table, c.row, c.col + 1)
        # a caption spanning several columns: its value is in the cell past the span, a weaker claim
        spanned = right >= 0 and out.cells[right].colspanned
        while right >= 0 and out.cells[right].colspanned:
            right = _grid_cell(out, c.table, c.row, out.cells[right].col + 1)
        if right in owned or neighbours[0]:
            return [Reading([], "")]
        below = _grid_cell(out, c.table, c.row + 1, c.col)
        if below >= 0 and out.cells[below].rowspanned and _header_row(out, c, stop):
            # a caption spanning header rows heads its column: its value is the first cell under the span
            r = c.row + 1
            while below >= 0 and out.cells[below].rowspanned:
                r += 1
                below = _grid_cell(out, c.table, r, c.col)
            btoks = _cell_run(out, below, stop)[0] if below >= 0 and below not in owned and not neighbours[1] else []
            return [Reading(btoks, _span_text(out, btoks))]
        # "1b Qualified dividends $" | "Form 1099-DIV": the slot is in the caption's cell, the neighbour is optional
        slot_here = bool(_SLOT_MARK.search(c.text[after:]))
        if right >= 0:
            toks, starts_stop = _cell_run(out, right, stop, ids_at)
            rcol = out.cells[right].col
            code_col = _code_column(out, c.table, rcol, stop, set((ids_at or {}).values())) if toks else ""
            if code_col == "boundary" and _code_cell(out, right):
                toks, starts_stop = [], True  # the code opens another form column: the value is below
            elif code_col == "skip" and _code_cell(out, right):
                # the code is printed before the value box (| caption | 10400 | 100,379 | 2 |): read past it
                after_code = _grid_cell(out, c.table, c.row, rcol + 1)
                toks = _cell_run(out, after_code, stop, ids_at)[0] if after_code >= 0 else []
                return [Reading(toks, _span_text(out, toks), weak=slot_here)]
            if toks:
                cands = [Reading(toks, _span_text(out, toks), weak=slot_here or spanned)]
                below = _grid_cell(out, c.table, c.row + 1, c.col)
                if spanned and below >= 0 and below not in owned and not neighbours[1]:
                    btoks, _ = _cell_run(out, below, stop)
                    if btoks:
                        cands.append(Reading(btoks, _span_text(out, btoks), weak=True))  # value under the span
                return cands
            if not starts_stop and out.cells[right].last < out.cells[right].first:
                below = _grid_cell(out, c.table, c.row + 1, c.col)
                cands = [Reading([], "")]
                if below >= 0:
                    btoks, _ = _cell_run(out, below, stop)
                    if btoks:
                        cands.append(Reading(btoks, _span_text(out, btoks), weak=True))
                return cands
        below = _grid_cell(out, c.table, c.row + 1, c.col)
        if below >= 0 and below not in owned and not neighbours[1]:
            btoks, _ = _cell_run(out, below, stop)
            if btoks:
                # a header-row caption owns the cell below; deeper in a layout table it may be printed text, and
                # under a caption spanning to the row end it may be the value written above the next caption
                return [Reading(btoks, _span_text(out, btoks), weak=c.row > 0 or spanned)]
        return [Reading([], "")]
    nxt = _next_cell(out, cell)
    cands = [Reading([], "")]
    if nxt >= 0 and not out.cells[nxt].fill:
        toks, _ = _cell_run(out, nxt, stop)
        if toks:
            cands.append(Reading(toks, _span_text(out, toks), weak=True))
    return cands


_CLOSES_BRACKET = _rx(r"^[ \t*_.…]*\)")


def read_before(out: Out, occ: Occ, stop: list[bool], claimed: set[int]) -> Reading | None:
    """A value written before its caption, as printed on forms whose caption sits under or after the blank:
    the line above a caption that opens its line (``Example Holdings Ltd`` / ``Partie demanderesse``), or the
    words inside a bracket the caption closes (``( AB Regional Division)``). The line above must hold no
    caption or box and must not be the next-line slot of another caption (``claimed``)."""
    cell = out.tcell[occ.start]
    c = out.cells[cell]
    if c.table >= 0:
        return None
    lead = [t for t in range(c.first, occ.start) if not out.toks[t].box]
    if lead:
        text = c.text
        opened = text.rfind("(", 0, out.toks[occ.start].start)
        closes = _CLOSES_BRACKET.match(text[out.toks[occ.end].end :])
        toks = [t for t in range(c.first, occ.start) if out.toks[t].start > opened]
        if opened < 0 or not closes or ")" in text[opened : out.toks[occ.start].start] or not toks:
            return None
        if any(stop[t] for t in toks):
            return None
        return Reading(toks, _span_text(out, toks), before=True)
    prev = cell - 1
    while prev >= 0 and out.cells[prev].table < 0 and out.cells[prev].last < out.cells[prev].first:
        if out.cells[prev].fill:
            return None
        prev -= 1
    if prev < 0 or prev in claimed or out.cells[prev].table >= 0:
        return None
    pc = out.cells[prev]
    toks = list(range(pc.first, pc.last + 1))
    if any(stop[t] for t in toks):
        return None
    return Reading(toks, _span_text(out, toks), before=True)


# ---------------------------------------------------------------------------
# Comparing values
# ---------------------------------------------------------------------------


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _ratio(a: str, b: str) -> float:
    n = max(len(a), len(b))
    return 1.0 - _lev(a, b) / n if n else 1.0


def _typo(a: str, b: str) -> bool:
    """A misread letter or two: at most max(1, 15%) edits."""
    return _lev(a, b) <= max(1, int(TYPO_EDITS * max(len(a), len(b))))


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch in "0123456789")


def _date_parts(keys: Sequence[str]) -> list[tuple[int, int, int]] | None:
    """Readings (year, month, day) of a date written with three numbers, or two and a month name."""
    keys = [k for k in keys if k not in DATE_FILLER]
    words = [k for k in keys if not _isdigits(k)]
    if len(keys) == 3 and len(words) == 1 and words[0] in MONTHS:
        p, q = (int(k) for k in keys if _isdigits(k))
        mo = MONTHS[words[0]]
        named = []
        for y, d in ((p, q), (q, p)):
            if y >= 32 and 1 <= d <= 31:
                named.append((y if y >= 100 else y + 1900, mo, d))
        return named or None
    nums = [k for k in keys if _isdigits(k)]
    if len(nums) != 3 or len(keys) != 3:
        return None
    a, b, c = (int(x) for x in nums)
    out = []
    for y, mo, d in ((a, b, c), (c, a, b), (c, b, a)):
        if y < 100:
            y += 2000 if y < 50 else 1900
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append((y, mo, d))
    return out


def _is_blank(keys: Sequence[str]) -> bool:
    """No value: no words, a blank word (N/A, none, blank), format hints and units, printed cents (00)."""
    if not keys:
        return True
    compact = "".join(keys)
    return compact in BLANK_WORDS or compact in ("00", "000") or all(k in HINT_WORDS for k in keys)


_CURRENCY = _rx(r"[$€£¥]|\b(?:USD|EUR|GBP|CAD|AUD|CHF|JPY)\b", re.IGNORECASE)
_AMOUNT = _rx(r"^(\()?([-−])?[ \t]*([0-9][0-9.,']*)(\))?$")


def _amount(text: str) -> str | None:
    """Canonical number of a value that is only an amount (currency, sign, grouping and decimal style dropped):
    ``$ 1,234.50`` -> ``1234.5``, ``1.234,50 EUR`` -> ``1234.5``, ``(12.00)`` -> ``12``. None otherwise. The sign is
    dropped because a read value's text starts at its first word (``-`` and ``(`` fall outside it)."""
    m = _AMOUNT.match(_strip(_CURRENCY.sub("", text)))
    if not m or (m.group(1) is None) != (m.group(4) is None):
        return None
    body = m.group(3).replace("'", "")  # 1'237.50 (Swiss grouping)
    if body[-1] in ".,":
        return None
    seps = [i for i, ch in enumerate(body) if ch in ".,"]
    frac = ""
    if seps:
        last = seps[-1]
        tail = body[last + 1 :]
        mixed = "," in body and "." in body
        if mixed or (body.count(body[last]) == 1 and len(tail) != 3):
            if mixed and body.count(body[last]) > 1:
                return None
            frac = tail
            body = body[:last]
        body = body.replace(",", "").replace(".", "")
    if not body:
        return None
    frac = frac.rstrip("0")
    return body + ("." + frac if frac else "")


def _decimal_kept(expected: str, got: str) -> bool:
    """Same digits, and a separator where the expected amount has its decimal mark (``47,843,41`` for
    ``47,843.41``): the decimal mark is written in another style, not lost."""
    e = _strip(_CURRENCY.sub("", expected))
    g = _strip(_CURRENCY.sub("", got))
    ae = _amount(expected)
    if ae is None or "." not in ae or _digits(e) != _digits(g):
        return False
    seps = [i for i, ch in enumerate(e) if ch in ".,"]
    tail = len(e) - seps[-1] - 1 if seps else 0
    return tail > 0 and len(g) > tail and g[-tail - 1] in ".,"


def _illegible(keys: Sequence[str]) -> bool:
    return (
        bool(keys)
        and all(k in ILLEGIBLE_WORDS or k in ILLEGIBLE_FILLER for k in keys)
        and any(k in ILLEGIBLE_WORDS for k in keys)
    )


@dataclass
class Verdict:
    cls: str  # ok | minor | major | critical
    kind: str


def compare_value(expected: str | None, keys: Sequence[str], others: dict[str, str], got: str = "") -> Verdict:
    """Severity of reading ``keys`` (raw text ``got``) where ``expected`` is due (``others``: compact values of
    other fields)."""
    exp = word_keys(expected) if expected is not None else []
    hint_value = len(exp) == 1 and exp[0] in HINT_WORDS  # "DD" alone is a value, not a format hint
    blank_exp = _is_blank(exp) and not hint_value
    blank_got = _is_blank(keys) and not (hint_value and list(keys) == exp)
    if _illegible(keys):
        # the parser marked the value unreadable: nothing false is claimed, the value is lost
        if blank_exp:
            return Verdict("minor", "illegible mark in a blank field")
        return Verdict("major", "value marked illegible")
    if blank_exp:
        if blank_got:
            return Verdict("ok", "")
        compact = "".join(keys)
        if len(compact) >= 2 and compact in others:
            return Verdict("critical", "value of another field")
        if len(keys) >= PROSE_WORDS and not _digits(compact):
            # a run of words without digits in a blank field reads as printed instructions, not a fill
            return Verdict("minor", "printed text in a blank field")
        return Verdict("critical", "invented value")
    if blank_got:
        return Verdict("major", "value lost")
    ae, ag = _amount(expected or ""), _amount(got)
    if ae is not None and ag is not None and not _isdigits(_strip(expected or "")):
        # amounts compare as numbers: grouping, decimal style, currency and trailing zeros are format (a reference
        # written as plain digits, an ID or an amount without its separators, keeps the comparison below)
        if ae == ag or _decimal_kept(expected or "", got):
            return Verdict("ok", "")
        if "." in ae and "." not in ag and ae.split(".")[0] == ag:
            return Verdict("major", "amount rounded")
        return Verdict("critical", "wrong value")
    ce, cr = "".join(exp), "".join(keys)
    if ce == cr:
        return Verdict("ok", "")
    de, dr = _date_parts(exp), _date_parts(keys)
    if de and dr and set(de) & set(dr):
        return Verdict("ok", "")
    if cr.startswith(ce) or cr.endswith(ce):
        return Verdict("minor", "extra text in value")
    if len(cr) >= 2 and cr in others and others[cr] != ce:
        return Verdict("critical", "value of another field")
    prefix = ce.startswith(cr)
    if _digits(ce) != _digits(cr):
        # a number cut short is a different number unless most of it is there
        if prefix and len(cr) >= TRUNC_RATIO * len(ce):
            return Verdict("minor", "value typo") if _typo(ce, cr) else Verdict("major", "value truncated")
        return Verdict("critical", "wrong value")
    if _typo(ce, cr):
        return Verdict("minor", "value typo")
    if prefix or keys == exp[: len(keys)]:
        return Verdict("major", "value truncated")
    return Verdict("critical", "wrong value")


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------


def _box_elements(out: Out, occ: list[Occ | None], ref: Reference) -> dict[int, list[tuple[int, int, int]]]:
    """Per cell: (position, 0 = glyph / 1 = option label, item or -1), in reading order."""
    el: dict[int, list[tuple[int, int, int]]] = {}
    for j, t in enumerate(out.toks):
        if t.box:
            el.setdefault(out.tcell[j], []).append((j, 0, -1))
    for x, o in enumerate(occ):
        if o is not None and ref.items[x].kind == "box":
            el.setdefault(out.tcell[o.start], []).append((o.start, 1, x))
    for v in el.values():
        v.sort()
    return el


def _only_glyphs(out: Out, cell: int) -> list[int]:
    c = out.cells[cell]
    if c.last < c.first:
        return []
    span = list(range(c.first, c.last + 1))
    return span if all(out.toks[t].box for t in span) else []


def find_glyph(
    out: Out, o: Occ, el: dict[int, list[tuple[int, int, int]]], claimed: list[bool], used: list[bool]
) -> int:
    """The box token that belongs to the option label at ``o`` (-1 = none)."""
    cell = out.tcell[o.start]
    elems = el.get(cell, [])
    trailing = bool(elems) and elems[0][1] == 1 and elems[-1][1] == 0

    def scan(step: int) -> int:
        k = o.start - 1 if step < 0 else o.end + 1
        gap = 0
        while out.cells[cell].first <= k <= out.cells[cell].last:
            t = out.toks[k]
            if t.box:
                return k if not claimed[k] else -1
            if used[k]:
                return -1
            gap += 1
            if gap > GLYPH_GAP:
                return -1
            k += step
        return -1

    g = scan(1 if trailing else -1)  # one direction per cell: [x] Yes [ ] No, or Yes [x] No [ ]
    if g >= 0:
        return g
    # a box alone in the previous / next line or table cell
    c = out.cells[cell]
    before_free = all(not used[t] and not out.toks[t].box for t in range(c.first, o.start))
    after_free = all(not used[t] and not out.toks[t].box for t in range(o.end + 1, c.last + 1))
    neighbours: list[tuple[int, bool]] = []
    if c.table >= 0:
        neighbours = [
            (_grid_cell(out, c.table, c.row, c.col - 1), True),
            (_grid_cell(out, c.table, c.row, c.col + 1), False),
        ]
    else:
        p = cell - 1
        while p >= 0 and out.cells[p].table < 0 and out.cells[p].last < out.cells[p].first:
            p -= 1
        nxt = cell + 1
        while nxt < len(out.cells) and out.cells[nxt].table < 0 and out.cells[nxt].last < out.cells[nxt].first:
            nxt += 1
        neighbours = [
            (p if p >= 0 and out.cells[p].table < 0 else -1, True),
            (nxt if nxt < len(out.cells) and out.cells[nxt].table < 0 else -1, False),
        ]
    for nb, is_before in neighbours:
        if nb < 0 or (is_before and not before_free) or (not is_before and not after_free):
            continue
        gl = _only_glyphs(out, nb)
        if gl:
            g = gl[-1] if is_before else gl[0]
            if not claimed[g]:
                return g
    return -1


def _state_word(out: Out, o: Occ, stop: list[bool], standalone: bool) -> bool | None:
    cell = out.tcell[o.end]
    run = _run(out, o.end + 1, stop, cell)
    run = _skip_label_tail(out, cell, out.toks[o.end].end, run)
    if not run:
        return None
    keys = [out.toks[t].key for t in run[:2]]
    # Yes / No after an option needs a colon (``Single: Yes``); bare, it may be the next option's label
    text = out.cells[cell].text
    yes_no = standalone or ":" in text[out.toks[o.end].end : out.toks[run[0]].start]
    for compact in ("".join(keys), keys[0]):  # "not checked" before "x" / "true"
        if compact in CHECK_WORDS or (yes_no and compact in YES_WORDS):
            return True
        if compact in UNCHECK_WORDS or (yes_no and compact in NO_WORDS):
            return False
    return None


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _cell_keys(out: Out, cell: int) -> list[str]:
    return [out.toks[t].key for t in _cell_tokens(out, cell)]


def _dropped_blank_columns(
    ref: Reference,
    tid: int,
    out: Out,
    ot: int,
    rowmap: dict[int, int],
    colmap: dict[int, int],
    head_col: int,
    width: int,
) -> set[int]:
    """Reference data columns of a row-header table that the output left out: columns blank in every row (a
    spacer between a row header and its codes), taken only when the output grid is narrower by exactly their
    number and the output cells where they would sit hold words in at least half of the rows (the remaining
    columns shifted left). Their blank fields read as blank."""
    rt = ref.tables[tid]
    missing = rt.ncols - (width - head_col)
    if missing <= 0 or not rowmap:
        return set()
    filled = {it.col for it in ref.items if it.table == tid and not (it.kind == "text" and it.value is None)}
    blank = [c for c in range(1, rt.ncols) if c not in colmap and c not in filled]
    if len(blank) != missing:
        return set()
    for c in blank:
        cells = [_grid_cell(out, ot, orow, head_col + c) for orow in rowmap.values()]
        worded = sum(1 for cell in cells if cell >= 0 and _cell_keys(out, cell))
        if 2 * worded < len(cells):
            return set()
    return set(blank)


def map_table(
    ref: Reference, tid: int, out: Out, occ: list[Occ | None], stop: list[bool]
) -> tuple[int, dict[int, int], dict[int, int], bool]:
    """(output table, ref col -> output col, ref row -> output row, rows out of order) for a reference table."""
    rt = ref.tables[tid]
    votes: dict[int, int] = {}
    for x in rt.captions + rt.rowheads:
        if x >= 0 and occ[x] is not None:
            vc = out.cells[out.tcell[occ[x].start]]  # type: ignore[union-attr]
            if vc.table >= 0:
                votes[vc.table] = votes.get(vc.table, 0) + 1
    if not votes:
        return -1, {}, {}, False
    ot = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    colmap: dict[int, int] = {}
    header = -1
    for c, x in enumerate(rt.captions):
        if x >= 0 and occ[x] is not None:
            oc = out.cells[out.tcell[occ[x].start]]  # type: ignore[union-attr]
            if oc.table == ot:
                colmap[c] = oc.col
                header = max(header, oc.row)
    rowmap: dict[int, int] = {}
    head_cols: dict[int, int] = {}
    for r, x in enumerate(rt.rowheads):
        if x >= 0 and occ[x] is not None:
            oc = out.cells[out.tcell[occ[x].start]]  # type: ignore[union-attr]
            if oc.table == ot:
                rowmap[r] = oc.row
                head_cols[oc.col] = head_cols.get(oc.col, 0) + 1
    # the column most row headers sit in (section titles written as a rowspan column left of them are fewer)
    head_col = max(head_cols, key=lambda col: (head_cols[col], -col)) if head_cols else -1
    grid = out.grids[ot]
    width = max((len(r) for r in grid), default=0)
    if head_col >= 0:
        # row-header tables: data columns follow the header column in order (-1: a blank column left out)
        dropped = _dropped_blank_columns(ref, tid, out, ot, rowmap, colmap, head_col, width)
        pos = head_col
        for c in range(1, rt.ncols):
            if c in dropped:
                colmap[c] = -1
                continue
            pos += 1
            if c not in colmap and pos < width:
                colmap[c] = pos
    elif len(colmap) < rt.ncols and rt.ncols == width:
        for c in range(rt.ncols):
            colmap.setdefault(c, c)
    if not colmap:
        return -1, {}, {}, False
    reordered = False
    if not rowmap:
        # order-preserving alignment of reference rows to output rows below the header
        ref_rows: dict[int, dict[int, list[str]]] = {}
        for it in ref.items:
            if it.table == tid and it.kind == "text" and it.row >= 0:
                ref_rows.setdefault(it.row, {})[it.col] = word_keys(it.value) if it.value else []
        orows = list(range(header + 1, len(grid)))
        nr = rt.nrows
        sims = [[0.0] * len(orows) for _ in range(nr)]
        for r in range(nr):
            cells = ref_rows.get(r, {})
            for b, orow in enumerate(orows):
                s = 0.0
                for c, keys in cells.items():
                    ocol = colmap.get(c)
                    if ocol is None:
                        continue
                    cell = _grid_cell(out, ot, orow, ocol)
                    got = _cell_keys(out, cell) if cell >= 0 else []
                    ce, cg = "".join(keys), "".join(got)
                    if keys and ce == cg:
                        s += 1.0
                    elif not keys and _is_blank(got):
                        s += 0.25
                    elif keys and got and _ratio(ce, cg) >= 0.5:
                        s += 0.5 * _ratio(ce, cg)  # a misread cell still ties its row
                sims[r][b] = s
        best = [[0.0] * (len(orows) + 1) for _ in range(nr + 1)]
        for r in range(nr - 1, -1, -1):
            for b in range(len(orows) - 1, -1, -1):
                take = best[r + 1][b + 1] + sims[r][b] if sims[r][b] > 0 else -1.0
                best[r][b] = max(best[r + 1][b], best[r][b + 1], take)
        r = b = 0
        while r < nr and b < len(orows):
            if sims[r][b] > 0 and best[r][b] == best[r + 1][b + 1] + sims[r][b]:
                rowmap[r] = orows[b]
                r += 1
                b += 1
            elif best[r + 1][b] >= best[r][b + 1]:
                r += 1
            else:
                b += 1
        # rows written in another order: a best-match assignment far better than the order-preserving one
        w_order = sum(sims[r][orows.index(b)] for r, b in rowmap.items())
        greedy: dict[int, int] = {}
        taken: set[int] = set()
        for _, r, b in sorted((-sims[r][b], r, b) for r in range(nr) for b in range(len(orows)) if sims[r][b] > 0):
            if r not in greedy and b not in taken:
                greedy[r] = b
                taken.add(b)
        w_greedy = sum(sims[r][b] for r, b in greedy.items())
        if w_greedy > max(w_order + REORDER_GAIN, REORDER_RATIO * w_order) + 1e-9:
            rowmap = {r: orows[b] for r, b in greedy.items()}
            seq = [rowmap[r] for r in sorted(rowmap)]
            reordered = any(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    return ot, colmap, rowmap, reordered


def stacked_cells(ref: Reference, tid: int, out: Out) -> dict[tuple[int, int], int]:
    """(ref row, ref col) -> output value cell for a reference table the output wrote as repeated blocks, one
    per row, each repeating the column captions with the values under them (``8. Dependent's SSN`` | ... then a
    value row, then ``9. Dependent's SSN`` ...). A column caption printed once per reference row (a printed id
    such as ``8.`` may precede it) gives its r-th occurrence to row r; the value is the cell below it. Columns
    whose caption count differs from the row count, or with an occurrence outside an output table, are left out:
    caption-per-line blocks (``Date: <empty>``) are read by the caption chain."""
    rt = ref.tables[tid]
    found: dict[tuple[int, int], int] = {}
    if rt.nrows < 2:
        return found
    col_forms: dict[int, list[str]] = {}
    for c, x in enumerate(rt.captions):
        if x >= 0:
            full, alts = _phrase(ref.items[x].label)
            col_forms[c] = ["".join(k) for k in [full, *alts] if k]
    every_form = {f for fs in col_forms.values() for f in fs}
    for c, forms in col_forms.items():
        hits: list[int] = []
        for ci in range(len(out.cells)):
            keys = _cell_keys(out, ci)
            for _ in range(2):
                if len(keys) > 1 and _LINE_NO.match(keys[0]):
                    keys = keys[1:]
            if keys and "".join(keys) in forms:
                hits.append(ci)
        if len(hits) != rt.nrows or any(out.cells[ci].table < 0 for ci in hits):
            continue
        vals = [_grid_cell(out, out.cells[ci].table, out.cells[ci].row + 1, out.cells[ci].col) for ci in hits]
        if any(v >= 0 and "".join(_cell_keys(out, v)).startswith(f) for v in vals for f in every_form):
            continue  # a caption (with or without its value) under the caption: not a caption-over-value layout
        first_row = out.grids[out.cells[hits[0]].table][0]
        opens_with_caption = any("".join(_cell_keys(out, ci)).startswith(f) for ci in first_row for f in every_form)
        ends_on_caption = vals[-1] < 0 and out.cells[hits[-1]].table == out.cells[hits[-2]].table
        if ends_on_caption or not opens_with_caption:
            continue  # a grid that opens with values, or closes on its captions, writes values above its captions
        for r, val in enumerate(vals):
            if val >= 0:
                found[(r, c)] = val
    return found


def _caption_spans(forms: dict[int, list[list[str]]], out: Out, cell: int) -> list[tuple[int, int, int]]:
    """(column, first, end) of each column caption (``forms``: column -> caption key forms) written in an output
    cell, as positions in the cell's word tokens, when the cell opens with one (up to two printed ids such as
    ``9.`` may precede it); [] for a cell that does not open with a caption."""
    keys = _cell_keys(out, cell)
    lead = 0
    while lead < 2 and len(keys) > lead + 1 and _LINE_NO.match(keys[lead]):
        lead += 1
    spans: list[tuple[int, int, int]] = []
    for c, fs in forms.items():
        for i in range(len(keys)):
            n = max((len(f) for f in fs if keys[i : i + len(f)] == f), default=0)
            if n:
                spans.append((c, i, i + n))
                break
    spans.sort(key=lambda s: (s[1], -s[2]))
    return spans if spans and spans[0][1] <= lead else []


@dataclass
class RowBlocks:
    """Output rows of each reference row in a grid that repeats the column captions in every row's block."""

    forms: dict[int, list[list[str]]]  # column -> caption key forms
    blocks: dict[int, tuple[int, int]]  # ref row -> (first output row, end row)
    heads: dict[int, tuple[int, int]]  # column -> (row offset in a block, output column) of a stacked header


def row_blocks(
    ref: Reference,
    tid: int,
    out: Out,
    occ: list[Occ | None],
    ot: int,
    colmap: dict[int, int],
    rowmap: dict[int, int],
) -> RowBlocks:
    """Blocks of output rows, one per reference row, in a grid that writes each reference row on several lines.
    Either the captions repeat in every block (``9. Dependent's SSN`` | ``Dependent's date of birth`` over their
    values, then ``Dependent's first name`` | ... over theirs), or the header itself is stacked: two captions
    located in the same output column on different header rows (``Box b - EIN`` over ``Box 15 - Employer's ID``),
    each data row then has one line per header row. A block starts at the caption row printed with the reference
    row's id (``9.``); without such ids, at the row's mapped output row when two of its cells open with captions or
    the header is stacked. It ends where the next block starts."""
    rt = ref.tables[tid]
    forms: dict[int, list[list[str]]] = {}
    for c, x in enumerate(rt.captions):
        if x >= 0:
            full, alts = _phrase(ref.items[x].label)
            forms[c] = [f for f in [full, *alts] if f]
    grid = out.grids[ot]
    starts: dict[int, int] = {}
    ids = {r: word_keys(ref.items[x].label) for r, x in enumerate(rt.rowheads) if x >= 0}
    for orow, cells in enumerate(grid):
        for cell in cells:
            keys = _cell_keys(out, cell)
            for r, hk in ids.items():
                if r not in starts and hk and keys[: len(hk)] == hk and _caption_spans(forms, out, cell):
                    starts[r] = orow
    caps: dict[int, tuple[int, int]] = {}  # column -> (output row, output column) of its located caption
    for c, x in enumerate(rt.captions):
        o = occ[x] if x >= 0 else None
        if o is not None and out.cells[out.tcell[o.start]].table == ot:
            caps[c] = (out.cells[out.tcell[o.start]].row, out.cells[out.tcell[o.start]].col)
    stacked = any(a[1] == b[1] and a[0] != b[0] for a in caps.values() for b in caps.values())
    if not starts:
        for r, orow in rowmap.items():
            if stacked or sum(1 for cell in grid[orow] if _caption_spans(forms, out, cell)) >= 2:
                starts[r] = orow
    rows = sorted(set(starts.values()))
    blocks = {r: (s, next((n for n in rows if n > s), len(grid))) for r, s in starts.items()}
    top = min((rc[0] for rc in caps.values()), default=0)
    heads = {c: (rc[0] - top, rc[1]) for c, rc in caps.items()} if stacked else {}
    return RowBlocks(forms, blocks, heads)


def block_reading(rb: RowBlocks, out: Out, ot: int, row: int, col: int, mapped: int | None = None) -> Reading | None:
    """Value of reference cell (row, col) inside its row's block: the rest of the cell holding the column's caption
    (up to the next caption), else the cell below it in the block; when the block does not print the caption, the
    cell at the caption's place in a stacked header, else the cell the table mapping gives (``mapped``, -1 = outside
    the grid; a caption there reads as no value). None when none exists."""
    start, end = rb.blocks[row]
    grid = out.grids[ot]
    for orow in range(start, end):
        for ocol, cell in enumerate(grid[orow]):
            spans = _caption_spans(rb.forms, out, cell)
            own = [s for s in spans if s[0] == col]
            if not own:
                continue
            words = _cell_tokens(out, cell)
            stop = min((s[1] for s in spans if s[1] >= own[0][2]), default=len(words))
            toks = words[own[0][2] : stop]
            if not toks and orow + 1 < end:
                below = _grid_cell(out, ot, orow + 1, ocol)
                if below >= 0 and not _caption_spans(rb.forms, out, below):
                    toks = _cell_tokens(out, below)
                    # captions merged in one cell share the value cell below: it may belong to the other one
                    return Reading(toks, _span_text(out, toks), weak=len(spans) > 1)
            return Reading(toks, _span_text(out, toks))
    if col in rb.heads and start + rb.heads[col][0] < end:
        cell = _grid_cell(out, ot, start + rb.heads[col][0], rb.heads[col][1])
        if cell >= 0 and not _caption_spans(rb.forms, out, cell):
            toks = _cell_tokens(out, cell)
            return Reading(toks, _span_text(out, toks))
    if mapped is not None:
        toks = [] if mapped < 0 or _caption_spans(rb.forms, out, mapped) else _cell_tokens(out, mapped)
        return Reading(toks, _span_text(out, toks))
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class Event:
    kind: str
    cls: str
    lane: str
    label: str
    expected: str
    got: str
    group: str
    item: int = -1
    charged: bool = True


@dataclass
class FormsResult:
    score: float
    proportional: float
    floor: float
    coverage: float
    lanes: dict[str, float | None]
    atoms: int
    located: int
    events: list[Event]
    source: str
    unscorable: int = 0
    states: list[dict[str, Any]] = field(default_factory=list)
    extras: int = 0  # field lines the output wrote that the reference does not have (tree references only)
    ids_found: int = 0  # located fields whose printed id came back with the caption
    ids_total: int = 0  # located fields that have a printed id

    def counts(self) -> dict[str, int]:
        c = {"critical": 0, "major": 0, "minor": 0}
        for e in self.events:
            if e.charged and e.cls in c:
                c[e.cls] += 1
        return c

    def metadata(self, max_examples: int = 25) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for e in self.events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        ordered = sorted(self.events, key=lambda e: (-SEV_ORDER.get(e.cls, 0), not e.charged))
        return {
            "proportional": self.proportional,
            "floor": self.floor,
            "coverage": self.coverage,
            "atoms": self.atoms,
            "located": self.located,
            "lanes": self.lanes,
            "events": self.counts(),
            "event_kinds": kinds,
            "uncharged_repeats": sum(1 for e in self.events if not e.charged),
            "event_examples": [
                {
                    "kind": e.kind,
                    "severity": e.cls,
                    "field": e.label[:120],
                    "expected": e.expected[:120],
                    "output": e.got[:160],
                    "charged": e.charged,
                }
                for e in ordered[:max_examples]
            ],
            "reference": self.source,
            "unscorable_reference_items": self.unscorable,
            "extra_fields": self.extras,
            "printed_ids": {"found": self.ids_found, "with_id": self.ids_total},
        }


def _lane(it: Item) -> str:
    if it.kind == "box":
        return "checkboxes"
    if it.kind == "sig":
        return "signatures"
    return "filled" if it.value is not None else "empty"


def _show(it: Item) -> str:
    if it.kind == "box":
        return "[x]" if it.state else "[ ]"
    if it.kind == "sig":
        return "signed" if it.state else "unsigned"
    return it.value if it.value is not None else "(blank)"


def score_reference(ref: Reference, out: Out) -> FormsResult:  # noqa: C901 - mirrored in JavaScript
    items = ref.items
    occ, used = locate(ref, out)
    stop = [used[j] or bool(out.toks[j].box) for j in range(len(out.toks))]
    others: dict[str, str] = {}
    for it in items:
        if it.kind == "text" and it.value is not None:
            ce = "".join(word_keys(it.value))
            if len(ce) >= 2:
                others.setdefault(ce, ce)
    captions = {c for c in ("".join(word_keys(it.label)) for it in items if it.label) if len(c) >= 2}
    printed = _printed_words(ref)
    events: list[Event] = []
    loss: dict[int, float] = {}
    located: set[int] = set()
    states: list[dict[str, Any]] = []

    def ev(x: int, kind: str, cls: str, got: str, group: str = "") -> None:
        it = items[x]
        events.append(Event(kind, cls, _lane(it), it.label, _show(it), got, group or kind, x))
        loss[x] = max(loss.get(x, 0.0), LOSS[cls])

    # ---- tables: output grid per reference table ----
    tmaps = [map_table(ref, t, out, occ, stop) for t in range(len(ref.tables))]
    for t, tm in enumerate(tmaps):
        if tm[3]:
            rt = ref.tables[t]
            x0 = next((x for x in rt.captions if x >= 0), -1)
            events.append(
                Event(
                    "form table rows reordered",
                    "minor",
                    "filled",
                    ref.items[x0].label if x0 >= 0 else "table",
                    "table",
                    "",
                    "table order",
                )
            )
    table_flat: set[int] = set()
    ref_cells = {(it.table, it.row, it.col) for it in items if it.kind != "anchor" and it.table >= 0}
    ref_neighbours = {
        x: ((it.table, it.row, it.col + 1) in ref_cells, (it.table, it.row + 1, it.col) in ref_cells)
        for x, it in enumerate(items)
        if it.kind != "anchor" and it.table >= 0 and it.phrase
    }
    owned: set[int] = set()
    for it in items:
        if it.kind != "anchor" and it.table >= 0 and not it.phrase:
            ot, colmap, rowmap, _ = tmaps[it.table]
            if ot >= 0 and it.col in colmap and it.row in rowmap and colmap[it.col] >= 0:
                owned.add(_grid_cell(out, ot, rowmap[it.row], colmap[it.col]))

    _rebind_table_captions(ref, out, occ, tmaps)
    ids_at = {o.start: "".join(word_keys(items[x].pid)) for x, o in enumerate(occ) if o is not None and items[x].pid}
    for x, o in enumerate(occ):
        if o is None or items[x].pid:
            continue
        lead = word_keys(items[x].label)[:1]
        if lead and _LINE_NO.match(lead[0]) and out.toks[o.start].key != lead[0]:
            ids_at.setdefault(o.start, lead[0])  # a section printed as "14 Self-employment earnings"
    # next-line slots of captions that end their line: not read as a value written above another caption
    next_line_slots = {
        _next_cell(out, out.tcell[o.end])
        for x, o in enumerate(occ)
        if o is not None and items[x].kind in ("text", "sig") and o.end == out.cells[out.tcell[o.end]].last
    }

    # ---- text fields and signatures ----
    stacked: dict[int, dict[tuple[int, int], int]] = {}  # reference table -> repeated-block cells, built on demand
    subfields: dict[tuple[int, int, int], list[int]] = {}  # (table, row, col) -> text items sharing that cell
    for x, it in enumerate(items):
        if it.kind == "text" and it.table >= 0:
            subfields.setdefault((it.table, it.row, it.col), []).append(x)
    for (stab, _, scol), sibs in subfields.items():
        # sub-field captions found only in their column's caption cell (``Date of birth (DD/MM/YY)``) are hints
        heads = ref.tables[stab].captions
        head = heads[scol] if scol < len(heads) else -1
        ho = occ[head] if len(sibs) > 1 and head >= 0 else None
        for y in sibs:
            if ho is not None and (oy := occ[y]) is not None and out.tcell[oy.start] == out.tcell[ho.start]:
                occ[y] = None
    blocks: dict[int, RowBlocks] = {}  # reference table -> caption/value row blocks, built on demand
    for x, it in enumerate(items):
        if it.kind not in ("text", "sig"):
            continue
        readings: list[Reading] = []
        o = occ[x]
        if o is not None:
            located.add(x)
            readings = read_slot(out, o, stop, owned, ref_neighbours.get(x, (False, False)), ids_at)
            # a filled field only: a blank field never takes its value from before its caption
            above = read_before(out, o, stop, next_line_slots) if it.value is not None else None
            if above is not None:
                readings.append(above)
            if o.q < CLEAN_Q - 1e-9:
                ev(x, "caption garbled", "minor", _span_text(out, list(range(o.start, o.end + 1))))
        elif it.table >= 0:
            ot, colmap, rowmap, _ = tmaps[it.table]
            if it.table not in stacked:
                stacked[it.table] = stacked_cells(ref, it.table, out)
            if ot >= 0 and it.table not in blocks:
                blocks[it.table] = row_blocks(ref, it.table, out, occ, ot, colmap, rowmap)
            if ot >= 0 and it.row in blocks[it.table].blocks:
                # captions and values alternate within each reference row's block of output rows
                located.add(x)
                mapped = (
                    _grid_cell(out, ot, rowmap[it.row], colmap[it.col])
                    if it.col in colmap and it.row in rowmap
                    else None
                )
                blk = block_reading(blocks[it.table], out, ot, it.row, it.col, mapped)
                readings = [blk] if blk is not None else []
            elif ot >= 0 and it.col in colmap and it.row in rowmap:
                located.add(x)
                cell = _grid_cell(out, ot, rowmap[it.row], colmap[it.col])
                toks = _cell_tokens(out, cell) if cell >= 0 else []
                sibs = subfields.get((it.table, it.row, it.col), [])
                if toks and len(sibs) > 1 and all(occ[y] is None for y in sibs):
                    parts = _split_subfields(out, cell, toks, len(sibs))
                    toks = parts[sibs.index(x)] if parts else toks
                readings = [Reading(toks, _span_text(out, toks))]
            elif (it.row, it.col) in stacked[it.table]:
                located.add(x)
                toks = _cell_tokens(out, stacked[it.table][(it.row, it.col)])
                readings = [Reading(toks, _span_text(out, toks))]
            elif ot < 0:
                table_flat.add(it.table)
        if o is None and not readings:
            if it.kind == "sig":
                if it.state:
                    ev(x, "signature field missing", "major", "")
                else:
                    ev(x, "empty field omitted", "minor", "")
                continue
            if it.value is None:
                if it.table >= 0 and it.table in table_flat:
                    continue  # blank cell of a table the output did not keep as a grid: not checkable
                ev(x, "empty field omitted", "minor", "")
                continue
            if it.table >= 0 and it.table in table_flat:
                exp = word_keys(it.value)
                if _contains(out, exp) or _in_row_cells(out, ref, x, exp):
                    loss.setdefault(x, 0.0)
                    located.add(x)
                    continue
            exp = word_keys(it.value)
            ev(x, "value without its caption" if _contains(out, exp) else "field missing", "major", "")
            continue
        # most favourable reading among the candidates
        best: tuple[int, Verdict, Reading] | None = None
        variants: list[tuple[Reading, list[str], str]] = []
        ids = _printed_ids(it, out, o)
        for rd in readings:
            words = [t for t in rd.toks if not out.toks[t].box]
            keys = [out.toks[t].key for t in words]
            variants.append((rd, keys, rd.text))
            if keys and keys[0] in ids:
                # the printed line number repeated before the value
                variants.append((rd, keys[1:], _span_text(out, words[1:])))
        blank_due = it.value is None if it.kind == "text" else not it.state
        own_caption = "".join(word_keys(it.label))
        for rd, keys, got_text in variants:
            if blank_due and "".join(keys) in captions:
                keys = []  # a blank slot read as a printed caption (a repeated block the reference lists fewer times)
            elif it.kind == "text" and keys and "".join(keys) == own_caption != "".join(word_keys(it.value or "")):
                keys = []  # the caption repeated in its own value slot: no value
            if it.kind == "sig":
                has = (bool(keys) and not _is_blank(keys)) or any(out.toks[t].box == 1 for t in rd.toks)
                if it.state:
                    v = Verdict("ok", "") if has else Verdict("major", "signature lost")
                else:
                    v = Verdict("critical", "signature invented") if has else Verdict("ok", "")
            else:
                v = compare_value(it.value, keys, others, got_text if keys else "")
                if v.kind == "invented value" and _printed_reading(keys, printed):
                    v = Verdict("minor", "printed text in a blank field")
            if rd.before and SEV_ORDER[v.cls] > SEV_ORDER["minor"]:
                continue
            rank = SEV_ORDER[v.cls]
            if best is None or rank < best[0]:
                best = (rank, v, rd)
            if rd.weak and v.cls != "ok":
                # an adjacent line may not belong to this field: reading it blank is allowed
                vb = compare_value(it.value, [], others) if it.kind == "text" else v
                if it.kind == "sig":
                    vb = Verdict("major", "signature lost") if it.state else Verdict("ok", "")
                if SEV_ORDER[vb.cls] < best[0]:
                    best = (SEV_ORDER[vb.cls], vb, Reading([], ""))
        assert best is not None
        _, v, rd = best
        states.append({"item": x, "reading": rd.text, "verdict": v.cls, "kind": v.kind})
        if v.cls == "ok":
            loss.setdefault(x, 0.0)
        else:
            ev(x, v.kind, v.cls, rd.text)
    for t in sorted(table_flat):
        rt = ref.tables[t]
        filled = [x for x, it in enumerate(items) if it.table == t and it.kind == "text" and it.value is not None]
        if filled and any(x >= 0 for x in rt.captions + rt.rowheads):
            x0 = rt.captions[0] if rt.captions and rt.captions[0] >= 0 else filled[0]
            events.append(
                Event("form table not kept as a grid", "minor", "filled", items[x0].label, "table", "", "table flat")
            )

    # ---- boxes ----
    el = _box_elements(out, occ, ref)
    claimed = [False] * len(out.toks)
    got_state: dict[int, bool | None] = {}
    for x, it in enumerate(items):
        if it.kind != "box":
            continue
        o = occ[x]
        st: bool | None = None
        if o is not None:
            located.add(x)
            g = find_glyph(out, o, el, claimed, used)
            if g >= 0:
                claimed[g] = True
                st = out.toks[g].box == 1
            else:
                st = _state_word(out, o, stop, it.group < 0)
        elif it.table >= 0:
            ot, colmap, rowmap, _ = tmaps[it.table]
            if ot >= 0 and it.col in colmap and it.row in rowmap:
                cell = _grid_cell(out, ot, rowmap[it.row], colmap[it.col])
                same_cell = [
                    y
                    for y, z in enumerate(items)
                    if z.kind == "box" and z.table == it.table and z.row == it.row and z.col == it.col
                ]
                if cell >= 0 and len(same_cell) == 1:
                    located.add(x)
                    c = out.cells[cell]
                    for t in range(c.first, c.last + 1):
                        if out.toks[t].box and not claimed[t]:
                            claimed[t] = True
                            st = out.toks[t].box == 1
                            break
        got_state[x] = st
    # options listed after their select caption (no glyphs there): listed = checked
    groups: dict[int, list[int]] = {}
    for x, it in enumerate(items):
        if it.kind == "box" and it.group >= 0:
            groups.setdefault(it.group, []).append(x)
    for gid, opts in groups.items():
        cap = next(
            (x for x, it in enumerate(items) if it.kind == "anchor" and it.role == "group" and it.group == gid), -1
        )
        if cap < 0 or occ[cap] is None:
            continue
        co = occ[cap]
        assert co is not None
        cell = out.tcell[co.end]
        c = out.cells[cell]
        if any(out.toks[t].box for t in range(c.first, c.last + 1)):
            continue
        listed = [
            x
            for x in opts
            if occ[x] is not None and out.tcell[occ[x].start] == cell and occ[x].start > co.end  # type: ignore[union-attr]
        ]
        if listed and len(listed) < len(opts):
            for x in opts:
                if got_state.get(x) is None:
                    got_state[x] = x in listed
                    located.add(x)
    for opts in groups.values():
        if any(got_state.get(x) is True for x in opts):
            for x in opts:
                if got_state.get(x) is None and occ[x] is None and items[x].single:
                    got_state[x] = False  # a single choice shown implies the other options are not chosen
                    located.add(x)
    flipped_groups: set[int] = set()
    # a single choice shown with every box empty: the answer is lost, no other option is claimed
    cleared = {
        gid
        for gid, opts in groups.items()
        if any(items[x].single for x in opts)
        and not any(got_state.get(x) is True for x in opts)
        and any(got_state.get(x) is False and items[x].state for x in opts)
    }
    for x, it in enumerate(items):
        if it.kind != "box":
            continue
        st = got_state.get(x)
        if st is None:
            if occ[x] is not None:
                ev(
                    x,
                    "checked state lost" if it.state else "unchecked state lost",
                    "major" if it.state else "minor",
                    "",
                )
            else:
                ev(
                    x,
                    "checked option missing" if it.state else "unchecked option omitted",
                    "major" if it.state else "minor",
                    "",
                )
            continue
        states.append({"item": x, "reading": "[x]" if st else "[ ]", "verdict": "ok" if st == it.state else "critical"})
        if st == it.state:
            loss.setdefault(x, 0.0)
            continue
        if it.single and it.group in cleared:
            ev(x, "choice lost (no option ticked)", "major", "[ ]")
            continue
        if it.single and it.group >= 0:
            group_key = f"select {it.group}"
            ev(x, "selected option changed", "critical", "[x]" if st else "[ ]", group_key)
            if it.group in flipped_groups:
                events[-1].charged = False
            flipped_groups.add(it.group)
        else:
            ev(x, "box state flipped", "critical", "[x]" if st else "[ ]")

    # ---- printed ids ----
    # A field's printed number (1a, 7, Box 11) must come back with its caption. Dropped or wrong is a
    # small error on the floor, never a loss on the atom: the value itself is scored above, and the
    # downstream consumers match fields by these numbers.
    with_pid = [
        x
        for x, it in enumerate(items)
        if it.kind != "anchor" and it.pid and occ[x] is not None and "".join(word_keys(it.pid))
    ]
    ids_found = 0
    for x in with_pid:
        it = items[x]
        o = occ[x]
        assert o is not None
        want = "".join(word_keys(it.pid))
        if _id_near_caption(out, o, want):
            ids_found += 1
            continue
        prev = _prev_key_same_cell(out, o)
        wrong = bool(prev) and bool(_LINE_NO.match(prev)) and prev != want
        kind = "printed id wrong" if wrong else "printed id dropped"
        events.append(Event(kind, "minor", "ids", it.label, it.pid, prev if wrong else "", kind, x))

    # ---- extra fields ----
    # Fields the output invents: page furniture written as a field (OMB No.: 1545-0112) or a field that is
    # not on the form. Only an exhaustive reference (the tree) can tell; a spot reference lists some
    # fields and cannot. A small error on the floor, no atom loss.
    extras = _extra_fields(out, used, captions, others) if ref.source == "tree" else []
    for key_text, val_text in extras:
        events.append(Event("extra field", "minor", "extra", key_text, "", val_text, "extra field", -1))

    # ---- aggregation ----
    scored = [x for x, it in enumerate(items) if it.kind != "anchor"]
    n_atoms = len(scored)
    for x in scored:
        loss.setdefault(x, 0.0)
    by_group: dict[str, list[Event]] = {}
    for e in events:
        by_group.setdefault(e.group, []).append(e)
    instances = _instances(items)
    for grp, es in by_group.items():
        if es[0].cls != "critical" and len(es) >= SYSTEMATIC_MIN and len(es) >= instances.get(grp, 10**9):
            for e in es[1:]:
                e.charged = False
            es[0].kind += f" (systematic, {len(es)}x)"
    cnt = {"critical": 0, "major": 0, "minor": 0}
    for e in events:
        if e.charged:
            cnt[e.cls] += 1
    scale = min(1.0, FLOOR_K / max(1, n_atoms))

    def fix(c: int) -> float:
        return min(c, 1) + max(0, c - 1) * scale

    floor = _clamp01(
        1
        - FLOOR["critical"] * fix(cnt["critical"])
        - FLOOR["major"] * fix(cnt["major"])
        - min(FLOOR["minor_cap"], FLOOR["minor"] * cnt["minor"])
    )
    total_loss = sum(loss[x] for x in scored)
    proportional = _clamp01(1 - total_loss / (n_atoms + PRIOR_ATOMS))
    n_located = sum(1 for x in scored if x in located)
    # Coverage is the better of all fields located and content fields (a value, a signature, a ticked box)
    # located. Blank fields and unticked boxes are minor events above; a form whose filled fields all came back
    # is not a lost form because its blank rows were dropped. An empty output still scores 0.
    content = [x for x in scored if (items[x].value is not None if items[x].kind == "text" else items[x].state)]
    n_content_located = sum(1 for x in content if x in located)
    # Blank table cells not located while their table's captions are (rows written as lines, header-only or
    # with fewer blank rows) leave the count: they cost a minor event at most and hold nothing to lose.
    n_checkable = n_atoms - sum(1 for x in scored if x not in located and _blank_cell_of_found_table(ref, occ, x))
    coverage = n_located / n_checkable if n_checkable else 1.0
    if content:
        coverage = max(coverage, n_content_located / len(content))
    score = min(proportional, floor)
    if coverage < COVERAGE_FREE:
        score = min(score, coverage / COVERAGE_FREE)
        events.append(
            Event(
                f"form fields missing ({n_checkable - n_located} of {n_checkable} not found)",
                "major",
                "coverage",
                "",
                "",
                "",
                "coverage",
                -1,
                False,
            )
        )
    lanes: dict[str, float | None] = {}
    for lane in ("filled", "empty", "checkboxes", "signatures"):
        lane_loss = [loss[x] for x in scored if _lane(items[x]) == lane]
        lanes[lane] = _clamp01(1 - sum(lane_loss) / len(lane_loss)) if lane_loss else None
    # share of located fields whose printed id came back; share of the output's fields that are on the form
    lanes["ids"] = ids_found / len(with_pid) if with_pid else None
    lanes["extra"] = (
        _clamp01(1 - len(extras) / (n_located + len(extras)))
        if ref.source == "tree" and (n_located + len(extras)) > 0
        else None
    )
    return FormsResult(
        score=score,
        proportional=proportional,
        floor=floor,
        coverage=coverage,
        lanes=lanes,
        atoms=n_atoms,
        located=n_located,
        events=events,
        source=ref.source,
        unscorable=ref.unscorable,
        states=states,
        extras=len(extras),
        ids_found=ids_found,
        ids_total=len(with_pid),
    )


_LINE_NO = re.compile(r"^(?:[0-9]{1,3}[a-z]?|[a-z])(?![\s\S])")


def _blank_cell_of_found_table(ref: Reference, occ: list[Occ | None], x: int) -> bool:
    """Whether item ``x`` is a blank text or unsigned signature cell of a table with a caption located."""
    it = ref.items[x]
    if it.table < 0 or (it.value is not None if it.kind == "text" else it.kind != "sig" or it.state):
        return False
    rt = ref.tables[it.table]
    return any(c >= 0 and occ[c] is not None for c in rt.captions + rt.rowheads)


def _id_near_caption(out: Out, o: Occ, want: str) -> bool:
    """Whether the field's printed id is written with its caption: just before it in the same cell
    (``- [1a] **Wages**``, ``**1a Wages**``, ``1.10 TIN``), baked onto the first caption word, or right after it."""
    cell = out.tcell[o.start]
    for j in range(o.start - 1, max(out.cells[cell].first, o.start - 4) - 1, -1):
        if out.toks[j].key == want or "".join(t.key for t in out.toks[j : o.start]) == want:
            return True
    first = out.toks[o.start].key
    if len(first) > len(want) and first.startswith(want):
        return True
    j = o.end + 1
    return j < len(out.toks) and out.tcell[j] == cell and out.toks[j].key == want


def _prev_key_same_cell(out: Out, o: Occ) -> str:
    if o.start > 0 and out.tcell[o.start - 1] == out.tcell[o.start] and not out.toks[o.start - 1].box:
        return out.toks[o.start - 1].key
    return ""


# A field line: an optional bullet, an optional [id], a bold or plain key, a colon, a value. Only a
# bulleted or bold key counts as a field claim; ``Note: see instructions`` in prose does not.
_FIELD_LINE = _rx(
    r"^[ \t]*(?P<bullet>[-*•][ \t]+)?(?:\[[^\]\n]*\][ \t]*)?(?P<bold>\*\*|__)?"
    r"(?P<key>[^:*_\n]{1,80}?)(?:\*\*|__)?[ \t]*:[ \t]*(?P<val>[^\n]*?)[ \t]*$"
)
EXTRA_MAX_KEY_WORDS = 6
EXTRA_MAX_VAL_WORDS = 12


def _extra_fields(out: Out, used: Sequence[bool], captions: set[str], others: dict[str, str]) -> list[tuple[str, str]]:
    """Field lines the output writes that no reference field accounts for: page furniture written as a
    field (``**OMB No.**: 1545-0112``) or an invented field. Only ``key: value`` lines outside tables with
    a bullet or a bold key, a short key and a short non-empty value count. A line that holds a located
    caption, a reference caption or value, or a fragment of a caption is never an extra."""
    found: list[tuple[str, str]] = []
    for c in out.cells:
        if c.table >= 0 or c.last < c.first:
            continue
        if any(used[t] for t in range(c.first, c.last + 1)):
            continue
        m = _FIELD_LINE.match(c.text)
        if not m or not (m.group("bullet") or m.group("bold")):
            continue
        key_keys = word_keys(m.group("key"))
        val_keys = word_keys(m.group("val"))
        if not key_keys or len(key_keys) > EXTRA_MAX_KEY_WORDS or all(_isdigits(k) for k in key_keys):
            continue
        if not val_keys or len(val_keys) > EXTRA_MAX_VAL_WORDS:
            continue
        kk = "".join(key_keys)
        if kk in captions or kk in others:
            continue
        if any(len(kk) >= 3 and (kk in cap or cap in kk) for cap in captions):
            continue
        found.append((m.group("key").strip(), m.group("val").strip()))
    return found


def _printed_ids(it: Item, out: Out, o: Occ | None) -> set[str]:
    """The field's printed id, and a line number printed just before its caption (``**38** Amount …``)."""
    ids = {"".join(word_keys(it.pid))} if it.pid else set()
    if o is not None and o.start > 0 and out.tcell[o.start - 1] == out.tcell[o.start]:
        k = out.toks[o.start - 1].key
        if _LINE_NO.match(k):
            ids.add(k)
    ids.discard("")
    return ids


_SUB_LINE = re.compile(r"^(?:[0-9]{1,3}[a-z]|[a-z])$")


def _printed_words(ref: Reference) -> set[str]:
    """Words printed on the form as the reference records them: captions, anchors, sections and printed ids.
    Expected values are left out."""
    words: set[str] = set()
    for it in ref.items:
        words.update(word_keys(it.label))
        words.update(word_keys(it.ctx))
        words.update(word_keys(it.pid))
    return words


def _printed_reading(keys: Sequence[str], printed: set[str]) -> bool:
    """A blank field read as printed text: every word is printed on the form, a sub-line label (``2a``, ``b``)
    or printed cents (``0``, ``00``), and at least one word is not a bare number (a number alone may be an
    invented amount). ``Tax-exempt interest`` read as ``2a b``, ``Net refund`` read as ``This is your net
    refund. 53``."""
    return any(not _isdigits(k) for k in keys) and all(
        k in printed or _SUB_LINE.match(k) or not k.strip("0") for k in keys
    )


def _contains(out: Out, keys: Sequence[str]) -> bool:
    """Whether the folded value occurs as consecutive output tokens (compact comparison)."""
    target = "".join(keys)
    if len(target) < 2:
        return False
    words = [t.key for t in out.toks if not t.box]
    for i in range(len(words)):
        acc = ""
        j = i
        while j < len(words) and len(acc) < len(target):
            acc += words[j]
            j += 1
        if acc == target:
            return True
    return False


def _rebind_table_captions(
    ref: Reference, out: Out, occ: list[Occ | None], tmaps: list[tuple[int, dict[int, int], dict[int, int], bool]]
) -> None:
    """Captions repeated inside form table cells (``EACH OCCURRENCE`` in every coverage row) are bound to their
    occurrences in reading order; when the table's rows were matched to other output rows, swap two such
    occurrences so each caption is read in the output cell of its own row."""
    items = ref.items

    def target(x: int) -> int:
        it = items[x]
        ot, colmap, rowmap, _ = tmaps[it.table]
        if ot < 0 or it.col not in colmap or it.row not in rowmap:
            return -1
        return _grid_cell(out, ot, rowmap[it.row], colmap[it.col])

    keys = {x: "".join(word_keys(it.label)) for x, it in enumerate(items) if it.table >= 0 and it.phrase}
    for x in sorted(keys):
        o = occ[x]
        tx = target(x)
        if o is None or tx < 0 or out.tcell[o.start] == tx:
            continue
        for y in sorted(keys):
            oy = occ[y]
            if y == x or oy is None or keys[y] != keys[x] or out.tcell[oy.start] != tx:
                continue
            ty = target(y)
            if ty < 0 or ty == out.tcell[o.start]:
                occ[x], occ[y] = Occ(x, oy.start, oy.end, oy.q), Occ(y, o.start, o.end, o.q)
                break


def _in_row_cells(out: Out, ref: Reference, x: int, keys: Sequence[str]) -> bool:
    """A value too short to search the whole page for (``7``, ``A``), found as a word of an output line that
    holds another value of its reference table row (a form table flattened to lines)."""
    target = "".join(keys)
    if not target or len(target) >= 2:
        return False
    it = ref.items[x]
    cells: set[int] = set()
    for z in ref.items:
        if z is it or z.table != it.table or z.row != it.row or z.kind != "text" or z.value is None:
            continue
        cells |= _cells_with(out, word_keys(z.value))
    return any(out.toks[t].key == target for c in cells for t in _cell_tokens(out, c))


def _cells_with(out: Out, keys: Sequence[str]) -> set[int]:
    """Output cells where the folded value starts (compact comparison, at least two characters)."""
    target = "".join(keys)
    found: set[int] = set()
    if len(target) < 2:
        return found
    idx = [j for j, t in enumerate(out.toks) if not t.box]
    for i in range(len(idx)):
        acc = ""
        j = i
        while j < len(idx) and len(acc) < len(target):
            acc += out.toks[idx[j]].key
            j += 1
        if acc == target:
            found.add(out.tcell[idx[i]])
    return found


def _instances(items: Sequence[Item]) -> dict[str, int]:
    """How many reference items a systematic event kind must hit."""
    filled = sum(1 for it in items if it.kind == "text" and it.value is not None)
    blank = sum(
        1 for it in items if it.kind in ("text", "sig") and (it.value is None if it.kind == "text" else not it.state)
    )
    checked = sum(1 for it in items if it.kind == "box" and it.state)
    unchecked = sum(1 for it in items if it.kind == "box" and not it.state)
    signed = sum(1 for it in items if it.kind == "sig" and it.state)
    return {
        "empty field omitted": blank,
        "value lost": filled,
        "field missing": filled,
        "value without its caption": filled,
        "value typo": filled,
        "extra text in value": filled,
        "value truncated": filled,
        "checked state lost": checked,
        "checked option missing": checked,
        "unchecked state lost": unchecked,
        "unchecked option omitted": unchecked,
        "signature lost": signed,
        "caption garbled": len([it for it in items if it.kind != "anchor"]),
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_forms(ref: Reference, output_markdown: str) -> FormsResult:
    """Score an output document's Markdown against its forms reference."""
    return score_reference(ref, read_output(output_markdown))


def forms_v2_applies(ref: Reference | None) -> bool:
    return ref is not None and any(it.kind != "anchor" for it in ref.items)


def forms_v2_metrics(ref: Reference, output_markdown: str) -> list[MetricValue]:
    """``forms_v2`` headline plus its lane diagnostics as metric values: ``filled``, ``empty``,
    ``checkboxes``, ``signatures``, ``ids`` (printed ids read back) and ``extra`` (fields on the form
    among the fields written), each omitted when the page has nothing of that kind."""
    res = score_forms(ref, output_markdown)
    metrics = [MetricValue(metric_name="forms_v2", value=res.score, metadata=res.metadata())]
    for lane, value in res.lanes.items():
        if value is not None:
            metrics.append(MetricValue(metric_name=f"forms_v2_{lane}", value=float(value)))
    return metrics


# ---------------------------------------------------------------------------
# Rendering a reference (oracle check)
# ---------------------------------------------------------------------------


def _md_escape(text: str) -> str:
    return " ".join(t for t in _WS_RUN.split(text) if t).replace("|", "\\|").replace("*", "\\*")


def render_reference(ref: Reference) -> str:
    """The reference written as a plain key-value page: the output a perfect parser could give."""
    items = ref.items
    lines: list[str] = []
    done_tables: set[int] = set()
    for it in items:
        if it.table >= 0:
            if it.table in done_tables:
                continue
            done_tables.add(it.table)
            lines.extend(_render_table(ref, it.table))
            continue
        label = _md_escape(it.label)
        pid = (it.pid + " ") if it.pid else ""
        if it.kind == "anchor":
            if it.role == "section":
                lines.append("")
                lines.append("## " + label)
            elif it.role == "group":
                lines.append("- **" + pid + label + "**")
            else:
                lines.append(label)
        elif it.kind == "box":
            lines.append(("  " if it.group >= 0 else "") + "- " + ("[x] " if it.state else "[ ] ") + pid + label)
        elif it.kind == "sig":
            lines.append("- **" + pid + label + "**: " + ("[signed]" if it.state else ""))
        else:
            lines.append("- **" + pid + label + "**: " + (_md_escape(it.value).replace("\n", " ") if it.value else ""))
    return _strip("\n".join(lines)) + "\n"


def _render_table(ref: Reference, tid: int) -> list[str]:
    rt = ref.tables[tid]
    items = ref.items
    cells: dict[tuple[int, int], list[str]] = {}
    for it in items:
        if it.table != tid or it.kind == "anchor" and it.role not in ("rowhead", "static"):
            continue
        key = (it.row, it.col)
        if it.kind == "anchor":
            cells.setdefault(key, []).append(_md_escape(it.label))
        elif it.kind == "box":
            cells.setdefault(key, []).append(
                ("[x] " if it.state else "[ ] ") + (_md_escape(it.label) if it.phrase else "")
            )
        elif it.kind == "sig":
            cells.setdefault(key, []).append("[signed]" if it.state else "")
        else:
            own = (_md_escape(it.label) + ": ") if it.phrase else ""
            cells.setdefault(key, []).append(own + (_md_escape(it.value).replace("\n", " ") if it.value else ""))
    head = ["" if x < 0 else _md_escape(items[x].label) for x in rt.captions]
    ncols = max(rt.ncols, 1)
    head = head + [""] * (ncols - len(head))
    out = ["", "| " + " | ".join(head) + " |", "|" + "---|" * ncols]
    for r in range(rt.nrows):
        row = [_strip(" ".join(cells.get((r, c), []))) for c in range(ncols)]
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out
