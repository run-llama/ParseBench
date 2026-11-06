"""forms_v2: one value written between the parts of its caption across lines."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _ref(label: str, value: str):
    return reference_from_tree(
        [{"page_number": 1, "items": [{"type": "field", "field": "text", "label": label, "value": value}]}]
    )


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]


BIRTH = _ref("Birth date Year Month Day", "1997-08-22")
PHONE = _ref("Area code Phone (other)", "(842) 555-8508")


def test_date_parts_between_unit_captions_on_separate_lines_are_one_value():
    assert _charged(BIRTH, "Birth date\nYear\n1997\nMonth\n08\nDay\n22\n") == []


def test_a_wrong_last_date_part_on_the_next_line_is_still_charged():
    assert _charged(BIRTH, "Birth date\nYear\n1997\nMonth\n08\nDay\n23\n") != []


def test_area_code_and_number_around_the_phone_caption_are_one_value():
    assert _charged(PHONE, "Area code\n(842)\n\nPhone (other)\n555-8508\n") == []
