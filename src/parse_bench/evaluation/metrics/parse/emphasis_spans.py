"""Delimiter-run pairing for the markdown emphasis detectors.

The ``**`` / ``*`` / ``_`` / ``~~`` arms of ``FormattingRule`` used to be
regexes guarded by CommonMark flanking rules. Flanking is a *local* test, and
two requirements proved locally indistinguishable: whitespace-padded emphasis
(``Grand ** Total ** shown below.``) must count as emphasis, while the same
characters seen between two padded spans (``** Alpha ** plain ** Beta **``)
must not let ``plain`` score. Around a single delimiter run those contexts are
identical - only *which run pairs with which* separates them - so this module
resolves emphasis the way a parser does: tokenize delimiter runs, pair openers
with closers, and let the caller search query text inside the paired spans.

Pairing happens in two passes over each kind's runs:

1. **Strict pass** - runs are classified with CommonMark's flanking rules
   (including the punctuation nuances and the no-intraword restriction for
   ``_``) and paired on a stack, nearest-opener-first. This resolves every
   tightly-delimited span exactly like a markdown parser would, and keeps a
   literal run such as the ``**`` in ``2**3`` from stealing an opener.
2. **Padded pass** - the runs left unpaired keep any strict capability they
   had and additionally may open or close across *inner* padding, provided the
   padded side's immediate neighbor really is whitespace. This is the
   evaluator's one deliberate departure from CommonMark: parser output often
   pads just inside the markers (``** Total **``, ``**Total **:``) while still
   meaning emphasis, so padded shapes score exactly like tight ones. A padded
   pairing whose content range would swallow a strict-pass span is discarded:
   two padded runs bracketing a real span (``** foo **bold** bar **``) are
   that span's neighbors, not its padding, and letting them pair would score
   the plain text around it.

Both passes drop stack openers once a blank line intervenes: markdown emphasis
cannot cross a paragraph break, whatever the line endings.

Padding may wrap a single line break for ``**`` and ``~~`` (a closer on its own
line, ``**Total\\n**``, still closes) but stays horizontal for ``*`` and ``_``:
a line-leading ``*`` is usually a bullet, and letting it reach across the line
break would pair one list item's marker with the next and italicize the item.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

# A paragraph break with LF or CRLF endings, possibly with trailing whitespace
# on the blank line.
_BLANK_LINE_RE = re.compile(r"\r?\n[ \t]*\r?\n")

# ASCII punctuation, per CommonMark's flanking definitions. Parser output is
# overwhelmingly ASCII; Unicode punctuation is treated as an ordinary letter.
_PUNCT = frozenset(string.punctuation)


@dataclass(frozen=True)
class _KindSpec:
    char: str
    exact_length: int | None  # run must be exactly this long (None: 2 or more)
    underscore_rule: bool  # apply CommonMark's no-intraword restriction
    padding_may_break_line: bool


_KIND_SPECS = {
    "bold": _KindSpec("*", None, False, True),
    "italic_star": _KindSpec("*", 1, False, False),
    "italic_under": _KindSpec("_", 1, True, False),
    "strikeout": _KindSpec("~", None, False, True),
}


@dataclass(frozen=True)
class _Run:
    start: int
    end: int


def _find_runs(md: str, spec: _KindSpec) -> list[_Run]:
    runs = []
    for match in re.finditer(re.escape(spec.char) + "+", md):
        length = match.end() - match.start()
        if spec.exact_length is not None:
            if length != spec.exact_length:
                continue
        elif length < 2:
            continue
        runs.append(_Run(match.start(), match.end()))
    return runs


def _is_ws(char: str | None) -> bool:
    # String edges count as whitespace, as in CommonMark.
    return char is None or char.isspace()


def _is_punct(char: str | None) -> bool:
    return char is not None and char in _PUNCT


def _strict_caps(md: str, run: _Run, spec: _KindSpec) -> tuple[bool, bool]:
    """(can_open, can_close) under CommonMark flanking rules."""
    prev = md[run.start - 1] if run.start > 0 else None
    nxt = md[run.end] if run.end < len(md) else None

    left_flanking = not _is_ws(nxt) and (not _is_punct(nxt) or _is_ws(prev) or _is_punct(prev))
    right_flanking = not _is_ws(prev) and (not _is_punct(prev) or _is_ws(nxt) or _is_punct(nxt))

    if not spec.underscore_rule:
        return left_flanking, right_flanking
    # Underscores do not open or close intraword emphasis (snake_case).
    can_open = left_flanking and (not right_flanking or _is_punct(prev))
    can_close = right_flanking and (not left_flanking or _is_punct(nxt))
    return can_open, can_close


def _padded_after(md: str, pos: int, may_break_line: bool) -> bool:
    """Non-whitespace follows *pos* after inner padding (never a blank line)."""
    i = pos
    while i < len(md) and md[i] in " \t":
        i += 1
    if may_break_line and i < len(md) and md[i] in "\r\n":
        if md[i] == "\r":
            i += 1
        if i < len(md) and md[i] == "\n":
            i += 1
        while i < len(md) and md[i] in " \t":
            i += 1
    return i < len(md) and not md[i].isspace()


def _padded_before(md: str, pos: int, may_break_line: bool) -> bool:
    """Non-whitespace precedes *pos* before inner padding (never a blank line)."""
    i = pos - 1
    while i >= 0 and md[i] in " \t":
        i -= 1
    if may_break_line and i >= 0 and md[i] == "\n":
        i -= 1
        if i >= 0 and md[i] == "\r":
            i -= 1
        while i >= 0 and md[i] in " \t":
            i -= 1
    return i >= 0 and not md[i].isspace()


def _padded_caps(md: str, run: _Run, spec: _KindSpec, strict: tuple[bool, bool]) -> tuple[bool, bool]:
    """Strict capabilities carried forward, plus genuinely padded sides.

    A padded capability requires actual padding - the immediate neighbor must
    be whitespace. Without that gate this pass would re-grant capabilities the
    strict rules deliberately withheld (an intraword ``_``, for example).
    """
    strict_open, strict_close = strict
    prev = md[run.start - 1] if run.start > 0 else None
    nxt = md[run.end] if run.end < len(md) else None
    can_open = strict_open or (
        nxt is not None and nxt.isspace() and _padded_after(md, run.end, spec.padding_may_break_line)
    )
    can_close = strict_close or (
        prev is not None and prev.isspace() and _padded_before(md, run.start, spec.padding_may_break_line)
    )
    return can_open, can_close


def _pair(
    md: str,
    runs: list[_Run],
    caps: dict[int, tuple[bool, bool]],
    blank_line_starts: list[int],
) -> tuple[list[tuple[int, int]], list[_Run]]:
    """Stack-pair *runs*, closing against the nearest open run first.

    A run that can both open and close acts as a closer whenever an opener is
    on the stack, as in CommonMark. Returns the paired spans as (start, end)
    content ranges plus the runs left unpaired.
    """
    spans: list[tuple[int, int]] = []
    stack: list[_Run] = []
    paired: set[int] = set()
    for run in runs:
        # Emphasis cannot cross a paragraph break: openers left of one are dead.
        # Openers sit in stack order, so one blank line strands all of them.
        if stack and _has_blank_between(blank_line_starts, stack[-1].end, run.start):
            stack.clear()
        can_open, can_close = caps[run.start]
        if can_close and stack:
            opener = stack.pop()
            spans.append((opener.end, run.start))
            paired.add(opener.start)
            paired.add(run.start)
        elif can_open:
            stack.append(run)
    unpaired = [run for run in runs if run.start not in paired]
    return spans, unpaired


def _has_blank_between(blank_line_starts: list[int], lo: int, hi: int) -> bool:
    import bisect

    idx = bisect.bisect_left(blank_line_starts, lo)
    return idx < len(blank_line_starts) and blank_line_starts[idx] < hi


def emphasis_spans(md: str, kind: str) -> list[tuple[int, int]]:
    """Content ranges of every *kind* emphasis span in *md*, in order."""
    spec = _KIND_SPECS[kind]
    runs = _find_runs(md, spec)
    if len(runs) < 2:
        return []
    blank_line_starts = [match.start() for match in _BLANK_LINE_RE.finditer(md)]

    strict = {run.start: _strict_caps(md, run, spec) for run in runs}
    spans, leftover = _pair(md, runs, strict, blank_line_starts)
    padded = {run.start: _padded_caps(md, run, spec, strict[run.start]) for run in leftover}
    more, _ = _pair(md, leftover, padded, blank_line_starts)
    # Two padded runs bracketing a strict span pair into a wrapper that would
    # score the plain text around that span. Bracketing is not padding: drop
    # any padded span that contains a strict one.
    more = [padded_span for padded_span in more if not _contains_any(padded_span, spans)]
    return sorted(spans + more)


def _contains_any(outer: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(outer[0] <= start and end <= outer[1] for start, end in spans)


class EmphasisSpanMatcher:
    """Searches for a query pattern inside paired emphasis spans.

    Duck-types the one method ``FormattingRule.run`` uses on its compiled
    patterns, so a matcher can sit in the same detector list as the tag-pair
    and heading regexes.
    """

    def __init__(self, kinds: tuple[str, ...], query: str):
        self._kinds = kinds
        self._query = re.compile(query, re.IGNORECASE)

    def search(self, content: str) -> bool:
        return any(
            self._query.search(content[start:end])
            for kind in self._kinds
            for start, end in emphasis_spans(content, kind)
        )
