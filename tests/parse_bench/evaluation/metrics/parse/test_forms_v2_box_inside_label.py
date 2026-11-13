"""forms_v2: a checkbox written inside its own option label."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms

LABEL = "I consent to receive notices by email at the address given above"
CONSENT = reference_from_tree(
    [{"page_number": 1, "items": [{"type": "field", "field": "checkbox", "label": LABEL, "value": True}]}]
)


def _charged(md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(CONSENT, md).events if e.charged]


def test_a_box_at_the_line_break_of_its_wrapped_label_is_that_options_box():
    assert _charged("I consent to receive notices by email\n[x] at the address given above\n") == []


def test_an_unticked_box_inside_the_label_is_still_charged():
    assert _charged("I consent to receive notices by email\n[ ] at the address given above\n") != []
