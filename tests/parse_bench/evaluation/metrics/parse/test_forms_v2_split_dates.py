"""forms_v2: a date written as day / month / year part fields on one side and as one field on the other."""

from __future__ import annotations

from typing import Any

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import FormsResult, reference_from_tree, score_forms


def _field(label: str | None, value: str | None) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "field", "field": "text", "label": label}
    return node | ({"isEmpty": True} if value is None else {"value": value})


def _ref(*items: dict[str, Any]) -> Any:
    name = _field("Name", "Ana Lopez")
    return reference_from_tree([{"page_number": 1, "items": [name, *items]}])


def _parts_ref(caption: str, labels: tuple[str, ...], values: tuple[str | None, ...]) -> Any:
    parts = [_field(lab, v) for lab, v in zip(labels, values, strict=True)]
    return _ref({"type": "section", "label": caption, "items": parts})


def _charged(res: FormsResult) -> list[tuple[str, str]]:
    return [(e.kind, e.label) for e in res.events if e.charged]


@pytest.mark.parametrize(
    ("labels", "md_date"),
    [
        (("DD", "MM", "YYYY"), "12/03/1990"),
        (("jour", "mois", "année"), "12.03.1990"),
        (("Tag", "Monat", "Jahr"), "12-03-90"),
        (("día", "mes", "año"), "12 / 3 / 1990"),
        (("日", "月", "年"), "12 03 1990"),
        (("d d", "m m", "y y y y"), "12/03/1990"),
    ],
)
def test_reference_parts_match_a_date_written_in_one_field(labels: tuple[str, ...], md_date: str) -> None:
    """Bug: the parts were charged "value lost" / "field missing" when the output wrote the date in one field."""
    ref = _parts_ref("Date of birth", labels, ("12", "03", "1990"))
    res = score_forms(ref, f"- **Name**: Ana Lopez\n- **Date of birth**: {md_date}\n")
    assert _charged(res) == []
    assert res.score == 1.0


def test_reference_parts_read_together_from_a_hinted_caption() -> None:
    """Bug: ``Date of Birth (Month/Day/Year): 08 / 13 / 1990`` gave Month and Day "value lost" and Year
    "extra text in value": the part captions sit in the hint and the last one read the whole date."""
    ref = _parts_ref("Date of Birth (Month/Day/Year)", ("Month", "Day", "Year"), ("08", "13", "1990"))
    res = score_forms(ref, "- **Name**: Ana Lopez\nDate of Birth (Month/Day/Year): 08 / 13 / 1990\n")
    assert _charged(res) == []


def test_reference_combined_date_matches_output_part_lines() -> None:
    """Bug: the date field read its first part line as a wrong value, and every part line was an extra field."""
    ref = _ref(_field("Date of birth", "12/03/1990"))
    md = "- **Name**: Ana Lopez\n- **Date of birth**\n  - **dd**: 12\n  - **mm**: 03\n  - **jjjj**: 1990\n"
    res = score_forms(ref, md)
    assert _charged(res) == []
    assert res.extras == 0


def test_unlabelled_date_boxes_are_not_extra_fields() -> None:
    """Bug: a reference date kept as unlabelled digit boxes made each labelled output box an extra field."""
    boxes = [_field(None, v) for v in ("2", "3", "0", "9", "2", "0", "2", "5")]
    ref = _ref({"type": "section", "label": "Authentication date", "items": boxes})
    lines = [("dd", "2"), ("dd", "3"), ("mm", "0"), ("mm", "9"), ("yyyy", "2"), ("yyyy", "0"), ("yyyy", "2")]
    md = "- **Name**: Ana Lopez\n- **Authentication date**\n" + "".join(f"  - **{k}**: {v}\n" for k, v in lines)
    assert score_forms(ref, md + "  - **yyyy**: 5\n").extras == 0
    assert score_forms(ref, md + "  - **yyyy**: 6\n").extras == 8  # one box misread: the lines stay extra


@pytest.mark.parametrize(
    ("ref", "md"),
    [
        (_parts_ref("Date of death", ("DD", "MM", "YYYY"), (None, None, None)), "**Date of death**: [___]\n"),
        (_ref(_field("Date of death", None)), "- **Date of death**\n  - **dd**: [___]\n  - **mm**: [___]\n"),
    ],
    ids=["reference_parts", "reference_one_field"],
)
def test_blank_date_matches_blank_other_shape(ref: Any, md: str) -> None:
    """Bug: blank parts not written as their own lines were "empty field omitted" under a blank date field."""
    res = score_forms(ref, "- **Name**: Ana Lopez\n" + md)
    assert _charged(res) == []


def test_lone_day_field_is_not_merged_into_a_date() -> None:
    """A ``Day`` field with no month or year beside it is no date part: its value is still read from its slot."""
    ref = _ref({"type": "section", "label": "Shift", "items": [_field("Day", "14"), _field("Hours", "8")]})
    res = score_forms(ref, "- **Name**: Ana Lopez\n- **Shift**: 14\n- **Hours**: 8\n")
    assert ("value without its caption", "Day") in _charged(res)


@pytest.mark.parametrize(
    ("md_date", "printed"),
    [("12/04/1990", "Date of birth"), ("03/12/1990", "Date of birth (DD/MM/YYYY)")],
    ids=["wrong_month", "day_and_month_swapped"],
)
def test_wrong_digits_in_the_one_field_are_still_charged(md_date: str, printed: str) -> None:
    """A different date written in one field must not pass as the parts: a changed digit, or day and month
    swapped where both read as a valid date."""
    ref = _parts_ref(printed, ("DD", "MM", "YYYY"), ("12", "03", "1990"))
    res = score_forms(ref, f"- **Name**: Ana Lopez\n- **{printed}**: {md_date}\n")
    assert res.score < 1.0
    assert {lab for _, lab in _charged(res)} & {"DD", "MM", "YYYY"}


def test_wrong_digits_in_part_lines_are_still_charged() -> None:
    ref = _ref(_field("Date of birth", "12/03/1990"))
    md = "- **Name**: Ana Lopez\n- **Date of birth**\n  - **dd**: 12\n  - **mm**: 04\n  - **yyyy**: 1990\n"
    res = score_forms(ref, md)
    assert res.extras == 3  # the part lines stay extra fields
    assert any(lab == "Date of birth" for _, lab in _charged(res))
