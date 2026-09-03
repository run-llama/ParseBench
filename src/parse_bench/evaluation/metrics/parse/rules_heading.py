"""Page-complete Markdown heading structure evaluation.

The older title rules answer whether selected labels look title-like or preserve
relative nesting.  This rule instead compares the complete ordered heading list
for one page, including absolute Markdown levels and false-positive headings.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html import unescape
from typing import cast

from rapidfuzz import fuzz

from parse_bench.evaluation.metrics.parse.rules_base import ParseTestRule
from parse_bench.evaluation.metrics.parse.test_types import TestType
from parse_bench.test_cases.parse_rule_schemas import ParseHeadingStructureRule

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$", re.MULTILINE)
_HTML_HEADING = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_FENCED_CODE = re.compile(r"^ {0,3}(`{3,}|~{3,}).*?^ {0,3}\1[ \t]*$", re.MULTILINE | re.DOTALL)
_HTML_TABLE = re.compile(r"<table\b.*?(?:</table\s*>|\Z)", re.IGNORECASE | re.DOTALL)
_PIPE_TABLE_ROW = re.compile(r"^\s{0,3}\|.*\|\s*$", re.MULTILINE)
_INLINE_MARKUP = re.compile(r"(?:\*\*|__|~~|`)")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_MATCH_THRESHOLD = 0.82


@dataclass(frozen=True)
class _Heading:
    text: str
    level: int
    offset: int

    @property
    def normalized(self) -> str:
        return _normalize_heading(self.text)


def _blank_preserving_offsets(match: re.Match[str]) -> str:
    """Hide excluded Markdown regions without changing later match offsets."""

    return "".join("\n" if char == "\n" else " " for char in match.group(0))


def _visible_text(text: str) -> str:
    text = _HTML_TAG.sub(" ", unescape(text))
    text = _INLINE_MARKUP.sub("", text)
    # Markdown links retain their visible label but not their destination.
    text = re.sub(r"!?(?:\[([^]]*)\])\([^)]*\)", r"\1", text)
    return _WHITESPACE.sub(" ", text).strip()


def _normalize_heading(text: str) -> str:
    text = unicodedata.normalize("NFKC", _visible_text(text)).casefold()
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", text)).strip()


def _extract_headings(markdown: str) -> list[_Heading]:
    """Extract only explicit ATX/HTML headings, in document order.

    Standalone bold lines are intentionally excluded: the benchmark measures
    whether the parser emitted a Markdown heading, not whether text merely has
    visual emphasis. Headings inside code fences or tables are excluded too.
    """

    masked = _FENCED_CODE.sub(_blank_preserving_offsets, markdown)
    masked = _HTML_TABLE.sub(_blank_preserving_offsets, masked)
    masked = _PIPE_TABLE_ROW.sub(_blank_preserving_offsets, masked)
    found: list[_Heading] = []

    for match in _ATX_HEADING.finditer(masked):
        raw = (match.group(2) or "").strip()
        raw = re.sub(r"[ \t]+#+[ \t]*$", "", raw).strip()
        text = _visible_text(raw)
        if _normalize_heading(text):
            found.append(_Heading(text=text, level=len(match.group(1)), offset=match.start()))

    for match in _HTML_HEADING.finditer(masked):
        text = _visible_text(match.group(2))
        if _normalize_heading(text):
            found.append(_Heading(text=text, level=int(match.group(1)), offset=match.start()))

    found.sort(key=lambda heading: heading.offset)
    return found


def _similarity(expected: _Heading, actual: _Heading) -> float:
    if not expected.normalized or not actual.normalized:
        return 0.0
    return fuzz.ratio(expected.normalized, actual.normalized) / 100.0


def _ordered_matches(expected: list[_Heading], actual: list[_Heading]) -> list[tuple[int, int, float]]:
    """Return the maximum-similarity monotonic one-to-one assignment.

    A sequence alignment is deliberate here. It prevents duplicate heading text
    from matching the wrong occurrence and makes reordered headings observable
    as a miss plus an extra rather than silently granting full credit.
    """

    rows = len(expected) + 1
    cols = len(actual) + 1
    scores = [[0.0] * cols for _ in range(rows)]
    actions = [[""] * cols for _ in range(rows)]

    for i in range(1, rows):
        for j in range(1, cols):
            choices = [(scores[i - 1][j], "skip_expected"), (scores[i][j - 1], "skip_actual")]
            similarity = _similarity(expected[i - 1], actual[j - 1])
            if similarity >= _MATCH_THRESHOLD:
                choices.append((scores[i - 1][j - 1] + similarity, "match"))
            scores[i][j], actions[i][j] = max(choices, key=lambda choice: (choice[0], choice[1] == "match"))

    matches: list[tuple[int, int, float]] = []
    i, j = len(expected), len(actual)
    while i and j:
        action = actions[i][j]
        if action == "match":
            matches.append((i - 1, j - 1, _similarity(expected[i - 1], actual[j - 1])))
            i -= 1
            j -= 1
        elif action == "skip_expected":
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches


class HeadingStructureRule(ParseTestRule):
    """Score the complete ordered set of page headings and absolute levels."""

    def __init__(self, rule_data: ParseHeadingStructureRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseHeadingStructureRule, self._rule_data)
        if self.type != TestType.HEADING_STRUCTURE.value:
            raise ValueError(f"Invalid type for HeadingStructureRule: {self.type}")
        self.expected = [
            _Heading(text=heading.text, level=heading.level, offset=index)
            for index, heading in enumerate(rule_data.headings)
        ]

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        actual = _extract_headings(md_content)
        matches = _ordered_matches(self.expected, actual)
        expected_matched = {expected_index for expected_index, _, _ in matches}
        actual_matched = {actual_index for _, actual_index, _ in matches}

        match_details: list[dict[str, object]] = []
        quality_sum = 0.0
        distances: list[int] = []
        for expected_index, actual_index, similarity in matches:
            expected = self.expected[expected_index]
            observed = actual[actual_index]
            distance = abs(expected.level - observed.level)
            # One wrong nesting level receives half credit; larger errors decay
            # further. Misses receive zero by construction.
            level_quality = 1.0 / (1.0 + distance)
            quality_sum += similarity * level_quality
            distances.append(distance)
            match_details.append(
                {
                    "expected_index": expected_index + 1,
                    "actual_index": actual_index + 1,
                    "expected_text": expected.text,
                    "actual_text": observed.text,
                    "expected_level": expected.level,
                    "actual_level": observed.level,
                    "text_similarity": round(similarity, 4),
                    "level_distance": distance,
                }
            )

        if not self.expected:
            score = 1.0 if not actual else 0.0
        elif not actual:
            score = 0.0
        else:
            quality_recall = quality_sum / len(self.expected)
            quality_precision = quality_sum / len(actual)
            score = (
                2.0 * quality_precision * quality_recall / (quality_precision + quality_recall)
                if quality_precision + quality_recall
                else 0.0
            )

        missed = [
            {"index": index + 1, "text": heading.text, "level": heading.level}
            for index, heading in enumerate(self.expected)
            if index not in expected_matched
        ]
        extra = [
            {"index": index + 1, "text": heading.text, "level": heading.level}
            for index, heading in enumerate(actual)
            if index not in actual_matched
        ]
        exact_levels = sum(detail["level_distance"] == 0 for detail in match_details)
        mean_distance = sum(distances) / len(distances) if distances else None
        self.result_details = {
            "expected_count": len(self.expected),
            "actual_count": len(actual),
            "matched_count": len(matches),
            "missed_count": len(missed),
            "extra_count": len(extra),
            "exact_level_count": exact_levels,
            "mean_level_distance": round(mean_distance, 4) if mean_distance is not None else None,
            "expected": [{"index": i + 1, "text": h.text, "level": h.level} for i, h in enumerate(self.expected)],
            "actual": [{"index": i + 1, "text": h.text, "level": h.level} for i, h in enumerate(actual)],
            "matches": match_details,
            "missed": missed,
            "extra": extra,
        }
        explanation = (
            f"Heading structure score={score:.3f}; expected={len(self.expected)}, actual={len(actual)}, "
            f"matched={len(matches)}, missed={len(missed)}, extra={len(extra)}, "
            f"exact_levels={exact_levels}/{len(matches)}, "
            f"mean_level_distance={mean_distance:.2f}"
            if mean_distance is not None
            else f"Heading structure score={score:.3f}; expected={len(self.expected)}, actual={len(actual)}, "
            f"matched=0, missed={len(missed)}, extra={len(extra)}, exact_levels=0/0, mean_level_distance=n/a"
        )
        return score == 1.0, explanation, score
