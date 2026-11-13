"""forms_v2: the printed century of a year slot ("ending [___], 20[___]") is form text, not a value."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms

PERIOD = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "field", "field": "text", "label": "tax year beginning", "isEmpty": True},
                {"type": "field", "field": "text", "label": "ending", "isEmpty": True},
            ],
        }
    ]
)


def _ending(md: str) -> list[tuple[str, str]]:
    return [(e.cls, e.kind) for e in score_forms(PERIOD, md).events if e.charged and e.label == "ending"]


def test_a_printed_century_before_a_blank_year_slot_is_not_an_invented_value():
    md = "**tax year beginning:** [___], 2025, **ending:** [___], 20[___]. See separate instructions.\n"
    assert ("critical", "invented value") not in _ending(md)


def test_a_filled_year_after_the_century_is_still_an_invented_value():
    md = "**tax year beginning:** [___], 2025, **ending:** [___], 2026. See separate instructions.\n"
    assert ("critical", "invented value") in _ending(md)


def test_a_printed_century_ending_the_line_is_not_an_invented_value():
    assert ("critical", "invented value") not in _ending(
        "**tax year beginning:** [___], 2025, **ending:** [___], 20[___]\n"
    )


AREA = reference_from_tree(
    [{"page_number": 1, "items": [{"type": "field", "field": "text", "label": "Area", "isEmpty": True}]}]
)


def test_a_number_before_a_unit_blank_is_still_an_invented_value():
    events = score_forms(AREA, "**Area:** [___] 20[___] SQ FT\n").events
    assert [(e.cls, e.kind) for e in events if e.charged] == [("critical", "invented value")]


def test_a_number_after_a_coded_blank_is_still_an_invented_value():
    events = score_forms(AREA, "**Area:** F22: [___] 20\n").events
    assert [(e.cls, e.kind) for e in events if e.charged] == [("critical", "invented value")]


def test_a_year_suffix_on_the_next_line_completes_the_century():
    ref = reference_from_tree(
        [{"page_number": 1, "items": [{"type": "field", "field": "text", "label": "ending", "value": "2026"}]}]
    )
    assert [e for e in score_forms(ref, "**ending:** [___], 20[___]\n26\n").events if e.charged] == []


def test_a_page_number_after_a_blank_line_does_not_complete_the_century():
    ref = reference_from_tree(
        [{"page_number": 1, "items": [{"type": "field", "field": "text", "label": "ending", "value": "2026"}]}]
    )
    assert any(e.charged for e in score_forms(ref, "**ending:** [___], 20[___]\n\n26\n").events)


def test_a_numbered_caption_on_the_next_line_does_not_complete_the_century():
    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {"type": "field", "field": "text", "label": "ending", "value": "2026"},
                    {"type": "field", "field": "text", "label": "26", "value": "Ada Moss"},
                ],
            }
        ]
    )
    md = "**ending:** [___], 20[___]\n26\nAda Moss\n"
    assert ("ending", "2026") in [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]
