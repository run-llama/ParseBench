"""Rule-mode heading lists bind each heading to an output heading line of the same text before body text."""

from __future__ import annotations

from types import SimpleNamespace

from parse_bench.evaluation.metrics.parse.formatting_v2 import rules_reference, score_formatting

HEADINGS = SimpleNamespace(
    type="heading_structure",
    headings=[{"text": "Results", "level": 2}, {"text": "Free Cash Flow", "level": 3}],
)
BODY = (
    "Free cash flow is defined as cash from operations minus capital expenditures.\n\n"
    "| Item | Free Cash Flow |\n|---|---|\n| Q3 | 4.9 |"
)


def _kinds(out: str) -> list[str]:
    ref, scope = rules_reference([HEADINGS])
    return [e.kind for e in score_formatting(ref, out, scope).events]


def test_heading_whose_words_recur_in_body_text_is_matched_to_the_heading_line():
    assert _kinds(f"## Results\n\n{BODY}\n\n### Free Cash Flow\n\nMore text.") == []


def test_heading_present_only_as_body_text_is_still_lost():
    assert _kinds(f"## Results\n\n{BODY}\n\nFree Cash Flow\n\nMore text.") == ["heading lost"]
