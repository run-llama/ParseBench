"""``repetition_overshoot``: the output repeats nothing more than the page holds.

For pages that tempt a parser to repeat itself (leaders, blanks, grids, repeated rows). Only overshoot fails:
writing a leader in another character, splitting, merging or dropping a repetition is never penalised, because a
page's text layer bounds a repetition only loosely (symbol fonts, column reading order, values drawn in charts).
Three budgets, after the output's markdown syntax is removed:

- fill: dot leaders, rules and blanks (any mix of ``FILL``), in characters. One run may be at most
  ``factor * longest + fill_slack``, one line of the page (``line_cap``), or ``factor * block + fill_slack`` where
  ``block`` is the page's consecutive leader lines merged into one; all of it at most ``factor * total + fill_slack``
  or ``line_cap`` per line of the page.
- units: a run of one character, a 2-8 character unit or a token, at most ``factor * n + unit_slack`` where ``n`` is how
  often the unit occurs anywhere in ``reference_text``.
- lines: a line repeated in the output, at most ``factor * n + line_slack`` where ``n`` counts the line's rarest word in
  ``reference_text`` (the text layer may read a repeated table row column by column, so the line itself is not counted),
  or the line's own count in ``reference_text`` for a line with no word (emoji, symbols) and whenever the reference is
  exact (``skip_units_absent_from_reference`` false: scans and generated pages); on an exact page also how often the
  line occurs inside the reference's lines (a row written as shorter lines, or one table cell per line).

Tables: a whole table written out more than ``factor`` times as often as the page holds it (at least once) is a loop,
and so is an html row repeated past the line budget. A value repeated across a table's complete, distinct rows, one
cell per line (a chart transcribed point by point), is data and is not budgeted as a line.

Units and lines are counted after folding look-alike characters (``O``/``0``, ``I``/``l``/``1``, ``$``/``5``), so a
page read with one for the other is not a loop.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, cast

from parse_bench.evaluation.metrics.parse.rules_base import ParseTestRule
from parse_bench.test_cases.parse_rule_schemas import ParseRepetitionOvershootRule

FILL = ".·•…_\\-–—=*═─~・▪∙⋅●◦"
_FILL_RUN = re.compile(f"[{FILL}](?:[ \t]*[{FILL}]){{9,}}")
_FILL_BLOCK = re.compile(f"[{FILL}](?:\\s*[{FILL}]){{9,}}")  # leader lines merged into one
_CHAR_RUN = re.compile(r"([^\s" + FILL + r"])\1{9,}")
_UNIT_RUN = re.compile(r"(\S.{1,7}?)\1{5,}")
_TOKEN_RUN = re.compile(r"(?<!\S)(\S+)(?:[ \t]+\1){5,}(?!\S)")
_WORD = re.compile(r"[^\W_]{2,}")
_LOOKALIKE = str.maketrans("OoIilǀ$", "0011115")
_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_HTML_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_HTML_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.S | re.I)
_MD_TABLE = re.compile(r"(?:^[ \t]*\|.*\|[ \t]*\n?){3,}", re.M)


def _text(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def _norm(text: str) -> str:
    return _text(text).lower().translate(_LOOKALIKE)


def tables(md: str) -> tuple[Counter[str], Counter[str], dict[str, set[str]], dict[str, str]]:
    """The output's tables (html and markdown, normalized text -> copies), its html rows (text -> copies), the cells
    of each complete html row, and each row as written."""
    found = [_norm(t) for t in _HTML_TABLE.findall(md)] + [_norm(t.replace("|", " ")) for t in _MD_TABLE.findall(md)]
    rows: Counter[str] = Counter()
    cells: dict[str, set[str]] = {}
    written: dict[str, str] = {}
    for r in _HTML_ROW.findall(md):
        text = _norm(r)
        rows[text] += 1
        cells[text] = {_norm(c) for c in _HTML_CELL.findall(r)}
        written.setdefault(text, _text(r))
    return Counter(t for t in found if len(t.split()) >= 20), rows, cells, written


def strip_markup(text: str) -> str:
    """Markdown and html syntax that repeats by construction (table separators, grids, rules, bold markers)."""
    text = text.replace("…", "...").replace(" ", " ").replace("\\", "")
    text = text.replace("&#x26;", "&").replace("&amp;", "&").replace("\ufe0f", "").replace("\ufe0e", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    # table separators, grids and horizontal rules; a longer line of them is a loop, not markup
    text = re.sub(r"(?m)^(?=.{0,250}$)(?:[ \t|+:\-=~]*[-=][ \t|+:\-=~]*|\s*(?:[*_]\s*){3,})$", " ", text)
    text = re.sub(r"~~([^~\n]+?)~~", r"\1", text)
    text = re.sub(
        r"\*\*([^\s*][^*\n]*?)\*\*|__([^\s_][^_\n]*?)__", lambda m: m.group(1) or m.group(2), text
    )  # bold, not a run
    return text.replace("|", " ")


def fill_runs(text: str) -> list[int]:
    """Length in characters (spaces excluded) of every fill run of 10 or more. A spaced leader (``. . . .``) is one
    run; leaders of 10 or more characters separated by a space (answer lines written on one line) are one run each."""
    runs: list[int] = []
    for m in _FILL_RUN.finditer(text):
        pieces = [len(p) for p in m.group(0).split()]
        if len(pieces) > 1 and max(pieces) >= 10:
            runs.extend(p for p in pieces if p >= 10)
        else:
            runs.append(sum(pieces))
    return runs


def _runs(text: str) -> dict[tuple[str, str], int]:
    """(kind, unit) -> longest run in ``text``: kind is ``u`` for characters, units and tokens, ``l`` for lines."""
    out: dict[tuple[str, str], int] = {}

    def put(key: tuple[str, str], n: int) -> None:
        out[key] = max(out.get(key, 0), n)

    for m in _CHAR_RUN.finditer(text):
        put(("u", m.group(1)), len(m.group(0)))
    for m in _UNIT_RUN.finditer(text):
        if m.group(1).strip().strip(FILL):
            put(("u", m.group(1).strip()), len(m.group(0)) // len(m.group(1)))
    for m in _TOKEN_RUN.finditer(text):
        if m.group(1).strip(FILL):
            put(("u", m.group(1)), len(m.group(0).split()))
    lines = Counter(" ".join(line.split()).lower() for line in text.splitlines())
    for line, n in lines.items():
        if n >= 4 and line.strip(FILL + " "):
            put(("l", line), n)
    return out


class RepetitionOvershootRule(ParseTestRule):
    """Fail when anything in the output is repeated past the page's own budget for it."""

    def __init__(self, rule_data: ParseRepetitionOvershootRule | dict[str, Any]):
        super().__init__(rule_data)
        if not isinstance(self._rule_data, ParseRepetitionOvershootRule):
            raise TypeError("repetition_overshoot requires ParseRepetitionOvershootRule")

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        rule = cast(ParseRepetitionOvershootRule, self._rule_data)
        exact = not rule.skip_units_absent_from_reference  # the reference is the whole page (generated) or empty (scan)
        out = strip_markup(md_content)
        ref = strip_markup(rule.reference_text)
        folded = out.translate(_LOOKALIKE)
        out_flat = " ".join(folded.split()).lower()
        flat = " ".join(ref.translate(_LOOKALIKE).split()).lower()
        words = Counter(_WORD.findall(flat))
        ref_lines = Counter(" ".join(line.split()).lower() for line in ref.translate(_LOOKALIKE).splitlines())
        overs: list[tuple[str, int, int, int]] = []  # (what, page budget, output, limit)
        copies, rows, row_cells, written = tables(md_content)
        lowered = folded.lower()

        def shown(unit: str) -> str:  # the unit as the output wrote it, before folding
            if len(lowered) != len(out):
                return unit
            m = re.search(r"\s+".join(map(re.escape, unit.split())), lowered) if unit.strip() else None
            return " ".join(out[m.start() : m.end()].split()) if m else unit

        def line_page(line: str) -> int:
            """How often the page holds a line: its rarest word on a text layer (which may read a repeated table row
            column by column), else the line itself."""
            if _WORD.search(line) and not exact:
                seen = [words[w] for w in _WORD.findall(line) if words[w]]  # a misread word (Il / II) is no evidence
                return min(seen) if seen else 0
            page = ref_lines[line]
            if exact:  # or the page's rows written as shorter lines, one table cell per line
                page = max(page, flat.count(line))
            return page

        def check(what: str, page: int, output: int, limit: int) -> None:
            if output > limit:
                overs.append((what, page, output, limit))

        if rule.max_chars is not None:
            n = sum(1 for c in out if c.isalnum())
            check("alphanumeric characters", 0, n, rule.max_chars)
        runs = fill_runs(out)
        longest = max(rule.page_fill_runs, default=0)
        # a text layer undercounts leaders, so there one line of the page is always allowed; an exact reference is not
        cap = 0 if exact else rule.line_cap
        block = max((sum(1 for c in m.group(0) if not c.isspace()) for m in _FILL_BLOCK.finditer(ref)), default=0)
        # or the page's block of leader lines (a form's answer box) merged into one run
        limit = max(int(rule.factor * longest) + rule.fill_slack, cap, int(rule.factor * block) + rule.fill_slack)
        check("fill run", longest, max(runs, default=0), limit)
        total = sum(rule.page_fill_runs)
        check(
            "fill in all",
            total,
            sum(runs),
            max(int(rule.factor * total) + rule.fill_slack, cap * rule.page_lines),
        )
        # a whole table written out again: the page holds it as often as its text occurs there (at least once)
        for table, n in copies.items():
            page = max(flat.count(table), 1)
            check(f"table {table[:30]!r}", page, n, int(rule.factor * page))
        # a table row written out again, budgeted like a repeated line
        looped = set()
        for row, n in rows.items():
            page = line_page(row)
            if n >= 4 and len(row.split()) > 1 and (page or exact) and n > int(rule.factor * page) + rule.line_slack:
                looped.add(row)
                check(repr(written[row][:40]), page, n, int(rule.factor * page) + rule.line_slack)
        # one table cell per line: a value repeated across a table's distinct, complete rows is data (a chart's
        # points), unless a table or one of its rows is itself a loop
        cells = set()
        if not any(n > 1 for n in copies.values()):
            cells = {c for row, cs in row_cells.items() if row not in looped for c in cs}
        for (kind, unit), n in _runs(folded).items():
            if kind == "l" and not exact and unit in cells:
                continue
            if kind == "l":
                page = line_page(unit)
                slack = rule.line_slack
            else:
                key = " ".join(unit.split()).lower()
                page = flat.count(key)
                slack = rule.unit_slack
                # every copy in the output counts, not only the longest run; on a generated page a unit it lacks is a
                # misread (a barcode read as ■I), so there only its run counts
                if exact and (page or not flat):
                    n = max(n, out_flat.count(key))
            if page == 0 and not exact:
                continue  # values read off a chart or an image, which the text layer cannot count
            check(repr(shown(unit)[:40]), page, n, int(rule.factor * page) + slack)

        if not overs:
            return True, ""
        what, page, output, limit = max(overs, key=lambda o: o[2] / (o[3] + 1))
        return False, f"Repeated more than the page holds: {what} x{output} (page {page}, limit {limit})"
