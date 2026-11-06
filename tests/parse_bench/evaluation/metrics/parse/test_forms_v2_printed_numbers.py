"""forms_v2: printed line numbers, footnote marks and bullets beside a caption are not values or marks."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _field(label: str, value: str | None) -> dict[str, Any]:
    if value is None:
        return {"type": "field", "field": "text", "label": label, "isEmpty": True}
    return {"type": "field", "field": "text", "label": label, "value": value}


def _ref(*nodes: dict[str, Any]):
    return reference_from_tree([{"page_number": 1, "items": list(nodes)}])


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.kind, e.got) for e in score_forms(ref, md).events if e.charged]


LINES = _ref(_field("Other deductions from the schedule", "4,821"), _field("Total deductions", "9,000"))


def test_line_number_repeated_at_the_end_of_a_row_label_cell_is_not_the_value():
    md = (
        "<table><tr><td>15 Other deductions from the schedule 15</td><td></td><td>4,821</td></tr>"
        "<tr><td>16 Total deductions 16</td><td></td><td>9,000</td></tr></table>\n"
    )
    assert _charged(LINES, md) == []
    assert _charged(LINES, md.replace("9,000", "9,100")) != []


def test_line_number_in_its_own_column_before_the_value_is_skipped():
    md = (
        "<table><tr><td>14. Some other line</td><td>14.</td><td></td></tr>"
        "<tr><td>15. Other deductions from the schedule</td><td>15.</td><td>4,821</td></tr>"
        "<tr><td>16. Total deductions</td><td>16.</td><td>9,000</td></tr></table>\n"
    )
    assert _charged(LINES, md) == []
    assert _charged(LINES, md.replace("4,821", "4,812")) != []


def test_value_on_the_line_below_a_numbered_caption_is_read_even_if_it_equals_the_number():
    ref = _ref(_field("P/S", "1"))
    assert _charged(ref, "<table><tr><td>1. P/S<br>1</td><td>Payer</td></tr></table>\n") == []


NOTES = _ref(_field("Former name(s)", None), _field("Contact details", None))
NOTE_1 = "1 Former name(s)\nPlease give any previous names.\n"


def test_spaced_footnote_number_whose_note_repeats_the_caption_is_not_a_value():
    md = f"Former name(s) 1\n\nContact details: 2\n\n{NOTE_1}\n2 Contact details\nA phone number.\n"
    assert _charged(NOTES, md) == []


def test_spaced_number_whose_note_does_not_repeat_the_caption_stays_invented():
    md = f"Former name(s) 1\n\nContact details: 7\n\n{NOTE_1}\n7 Where a phone number is unknown.\n"
    assert [g for _, g in _charged(NOTES, md)] == ["7"]


def _yes(checked: bool) -> dict[str, Any]:
    box = {"type": "field", "field": "checkbox", "label": "Yes", "value": checked}
    return {"type": "field", "field": "single_select", "label": "Child tax credit", "valueItems": [box]}


def test_bullet_after_a_box_glyph_is_not_a_second_box():
    ref = _ref(_yes(False))
    assert _charged(ref, "Child tax credit: ☐ ● Yes\n") == []
    assert _charged(ref, "Child tax credit: ☒ ● Yes\n") != []


def test_next_captions_printed_number_after_a_blank_marker_is_not_a_value():
    ref = _ref(_field("Date (day month year)", None), _field("Signature of the officer", None))
    md = "**Date (day month year):** [___]  **1 Signature of the officer:** [___]\n"
    assert _charged(ref, md) == []
    assert _charged(ref, md.replace("[___]  **1", "[___] 03/05 **1")) != []


def test_printed_year_after_the_blanks_of_a_date_template_is_printed_text_not_an_invented_value():
    ref = _ref(_field("Date of death", None))
    assert [k for k, _ in _charged(ref, "Date of death [___] / [___] /2025\n")] == ["printed text in a blank field"]
    assert [k for k, _ in _charged(ref, "Date of death 20/11/1993\n")] == ["invented value"]
