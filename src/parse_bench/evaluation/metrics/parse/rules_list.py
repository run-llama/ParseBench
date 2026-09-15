"""List-item nesting level test rule (CommonMark list semantics).

Motivation: parser markdown sometimes flattens nested bullets, and every text
rule normalizes whitespace away before matching, so indentation is invisible
to them. Heading levels are scored (``is_title``/``title_hierarchy_percent``)
but list levels were not.

Levels are computed with CommonMark list-item semantics
(https://spec.commonmark.org/0.31.2/#list-items): there is no fixed number of
spaces that nests an item - a nested item's marker must sit at or beyond the
column of the first content character of the item it nests under ("the
position of the text after the list marker determines how much indentation is
needed in subsequent blocks in the list item"). Rather than re-implementing
that arithmetic, extraction delegates to ``markdown-it-py``'s ``commonmark``
preset, which is a CommonMark reference implementation.

Two deliberate widenings of strict CommonMark, mirroring how the formatting
rules tolerate equivalent HTML markup:

- A line whose first non-space character is a typographic bullet glyph
  (``•``, ``◦``, ``–``, ...) followed by whitespace is treated as a ``-``
  bullet at the same column. Faithful parses of PDFs often transcribe the
  visible glyph, and every glyph handled is one character wide, so the
  CommonMark column arithmetic is unchanged.
- HTML ``<ul>/<ol>/<li>`` nesting is accepted as equivalent, with the level of
  an ``<li>`` being the number of open list containers around it.

Indentation inside tables is deliberately NOT measured: ``<table>`` spans and
pipe-table rows are masked out before extraction, so a bullet drawn inside a
cell can neither satisfy nor fail a ``list_level`` rule.
"""

import re
from functools import lru_cache
from html import unescape
from typing import cast

from markdown_it import MarkdownIt

from parse_bench.evaluation.metrics.parse.rules_base import (
    ParseTestRule,
    RuleNotApplicable,
    is_degenerate_marker_text,
)
from parse_bench.evaluation.metrics.parse.test_types import TestType
from parse_bench.evaluation.metrics.parse.utils import normalize_text
from parse_bench.test_cases.parse_rule_schemas import ParseListLevelRule

# One-character typographic bullets a faithful PDF transcription may keep in
# place of a markdown marker. Each maps onto "-" at the same column, so the
# CommonMark indentation rules apply unchanged.
_BULLET_GLYPH_LINE = re.compile(r"^([ \t]*)([•◦▪▸‣∙·–—])([ \t])")

_TABLE_SPAN = re.compile(r"<table\b.*?(?:</table\s*>|\Z)", re.IGNORECASE | re.DOTALL)
# A pipe-table row: the cheap GFM signature. Cell text is table content, which
# this rule must not measure.
_PIPE_ROW = re.compile(r"^\s{0,3}\|.*\|\s*$")

_HTML_LIST_TAG = re.compile(r"<\s*(/?)\s*(ul|ol|li)\b[^>]*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def _mask_tables(md_content: str) -> str:
    """Blank out table regions while preserving line structure elsewhere."""

    def blank_span(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    masked = _TABLE_SPAN.sub(blank_span, md_content)
    return "\n".join("" if _PIPE_ROW.match(line) else line for line in masked.split("\n"))


def _canonicalize_bullet_glyphs(md_content: str) -> str:
    return "\n".join(
        _BULLET_GLYPH_LINE.sub(lambda match: f"{match.group(1)}-{match.group(3)}", line)
        for line in md_content.split("\n")
    )


def _markdown_list_items(md_content: str) -> list[tuple[str, int]]:
    """(normalized own text, 1-based level) for every CommonMark list item.

    An item's own text is its inline content outside any nested descendant
    list, so a parent matches its own sentence and never its children's.
    """

    tokens = MarkdownIt("commonmark").parse(md_content)
    items: list[tuple[str, int]] = []
    list_depth = 0
    open_items: list[tuple[int, list[str]]] = []
    for token in tokens:
        if token.type in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
        elif token.type in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
        elif token.type == "list_item_open":
            open_items.append((list_depth, []))
        elif token.type == "list_item_close":
            level, chunks = open_items.pop()
            items.append((normalize_text(" ".join(chunks)).strip(), level))
        elif token.type == "inline" and open_items and list_depth == open_items[-1][0]:
            open_items[-1][1].append(token.content)
    return items


def _html_chunk_text(chunks: list[str]) -> str:
    return normalize_text(unescape(_HTML_TAG.sub(" ", " ".join(chunks)))).strip()


def _html_list_items(md_content: str) -> list[tuple[str, int]]:
    """(normalized own text, 1-based level) for every HTML ``<li>``.

    A tag-stack scan rather than a full HTML parse: the level of an ``<li>``
    is the number of ``<ul>/<ol>`` containers open around it, and unclosed
    sibling ``<li>`` tags close implicitly, as in HTML.
    """

    items: list[tuple[str, int]] = []
    container_depth = 0
    open_items: list[tuple[int, list[str]]] = []
    position = 0
    for match in _HTML_LIST_TAG.finditer(md_content):
        text_between = md_content[position : match.start()]
        position = match.end()
        if open_items and container_depth == open_items[-1][0]:
            open_items[-1][1].append(text_between)
        closing = match.group(1) == "/"
        tag = match.group(2).lower()
        if tag in ("ul", "ol"):
            if not closing:
                container_depth += 1
            else:
                while open_items and open_items[-1][0] >= container_depth:
                    level, chunks = open_items.pop()
                    items.append((_html_chunk_text(chunks), level))
                container_depth = max(0, container_depth - 1)
        elif closing:
            if open_items and open_items[-1][0] == container_depth:
                level, chunks = open_items.pop()
                items.append((_html_chunk_text(chunks), level))
        elif container_depth > 0:
            if open_items and open_items[-1][0] == container_depth:
                level, chunks = open_items.pop()
                items.append((_html_chunk_text(chunks), level))
            open_items.append((container_depth, []))
    while open_items:
        level, chunks = open_items.pop()
        items.append((_html_chunk_text(chunks), level))
    return items


@lru_cache(maxsize=16)
def extract_list_items(md_content: str) -> tuple[tuple[str, int], ...]:
    """Every list item in the output as (normalized own text, 1-based level).

    Cached because a document carries one ``list_level`` rule per item and the
    extraction cost is per document, not per rule.
    """

    prepared = _canonicalize_bullet_glyphs(_mask_tables(md_content))
    return tuple(_markdown_list_items(prepared) + _html_list_items(prepared))


class ListLevelRule(ParseTestRule):
    """Verify that text appears as a list item at the expected nesting level.

    Passes when any markdown or HTML list item whose own text contains the
    rule text sits at exactly ``level`` (1 = top level). An item rendered as a
    plain paragraph fails - the list identity itself was lost - and an item
    whose indentation does not satisfy the CommonMark nesting rules counts at
    the level CommonMark actually assigns it, which is precisely the defect
    this rule exists to catch.
    """

    def __init__(self, rule_data: ParseListLevelRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseListLevelRule, self._rule_data)

        if self.type != TestType.LIST_LEVEL.value:
            raise ValueError(f"Invalid type for ListLevelRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

        if rule_data.level is None or rule_data.level < 1:
            raise ValueError("list_level requires an integer level >= 1")
        self.level = rule_data.level

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        if is_degenerate_marker_text(self.text):
            raise RuleNotApplicable(f"list_level text {self.text!r} is markdown markers only")
        query = normalize_text(self.text).strip()
        if not query:
            raise RuleNotApplicable(f"list_level text {self.text!r} is empty after normalization")

        # Whitespace-insensitive containment: spacing is not list structure,
        # and inline styling around a number routinely leaves a stray space
        # ("ときは、**15** 日" parses back with one more space than the source).
        compact_query = _WHITESPACE.sub("", query)
        matched_levels = sorted(
            {
                level
                for text, level in extract_list_items(md_content)
                if query in text or compact_query in _WHITESPACE.sub("", text)
            }
        )
        if self.level in matched_levels:
            return True, ""
        if not matched_levels:
            return (
                False,
                (
                    f"Expected '{self.text[:60]}' to be a list item at nesting level {self.level}, "
                    "but it does not appear as a markdown or HTML list item"
                ),
            )
        return (
            False,
            (
                f"Expected '{self.text[:60]}' at list nesting level {self.level}, "
                f"but it appears at level(s) {matched_levels}"
            ),
        )
