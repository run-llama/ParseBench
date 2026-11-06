"""forms_v2: a footnote number printed right after a caption is not the field's value."""

from __future__ import annotations

from typing import Any

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, render_reference, score_forms

TREE: list[dict[str, Any]] = [
    {
        "page_number": 1,
        "items": [
            {"type": "field", "field": "text", "label": "Joint time estimate", "isEmpty": True},
            {"type": "field", "field": "text", "label": "Hearing location", "value": "Brisbane"},
        ],
    }
]
REF = reference_from_tree(TREE)
FOOTNOTE = "7 Where the parties cannot agree, each party's time estimate must be provided."


def kinds(md: str) -> list[str]:
    return [e.kind for e in score_forms(REF, md).events if e.charged]


def test_reference_scores_one():
    assert score_forms(REF, render_reference(REF)).score == 1.0


@pytest.mark.parametrize(
    "line, note",
    [
        ("Joint time estimate:<sup>7</sup>", "<sup>7</sup> Where the parties cannot agree."),
        ("Joint time estimate:⁷", FOOTNOTE),
        ("Joint time estimate:7", "7\n\nWhere the parties cannot agree."),
        ("<table><tr><td><strong>Joint time estimate:</strong><sup>7</sup></td><td></td></tr></table>", FOOTNOTE),
    ],
)
def test_footnote_mark_after_caption_is_not_a_value(line: str, note: str):
    md = f"{line}\n\nHearing location: Brisbane\n\n---\n\n{note}"
    assert kinds(md) == []


def test_footnote_mark_before_a_value_keeps_the_value():
    md = f"Joint time estimate:<sup>7</sup>\n\nHearing location:<sup>7</sup> Sydney\n\n{FOOTNOTE}"
    assert kinds(md) == ["wrong value"]


@pytest.mark.parametrize(
    "line, note",
    [
        ("Joint time estimate: 7", FOOTNOTE),  # plain number after a space: a value
        ("Joint time estimate:<sup>7</sup>", ""),  # no footnote 7 in the output
        ("Joint time estimate:7", "7. Where the parties cannot agree."),  # a numbered item, not a footnote
    ],
)
def test_number_without_footnote_evidence_is_still_invented(line: str, note: str):
    md = f"{line}\n\nHearing location: Brisbane\n\n{note}"
    assert kinds(md) == ["invented value"]
