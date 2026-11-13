"""forms_v2 round 3 review fixes: shapes where a round 3 rule hid a wrong value or lost a right one."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _f(value: str | None, label: str | None = None) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "field", "field": "text"}
    if label:
        node["label"] = label
    node.update({"isEmpty": True} if value is None else {"value": value})
    return node


def _cell(value: str | None) -> dict[str, Any]:
    return {"items": [_f(value)]}


def _ref(*nodes: dict[str, Any]):
    return reference_from_tree([{"page_number": 1, "items": list(nodes)}])


def _table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"type": "table", "columns": columns, "rows": rows}


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]


def _serious(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged and e.cls != "minor"]


def test_f5_values_equal_to_the_line_numbers_are_read_as_values():
    ref = _ref(_f("14", "Alpha amount"), _f("15", "Beta amount"), _f("16", "Gamma amount"))
    md = "<table><tr><td>14 Alpha amount</td><td>14</td></tr><tr><td>15 Beta amount</td><td>15</td></tr>"
    md += "<tr><td>16 Gamma amount</td><td>16</td></tr></table>"
    assert _serious(ref, md) == []


def test_f6_number_after_a_blank_marker_before_the_next_caption_is_the_value():
    ref = _ref(_f("35", "Age"), _f("Ana", "Name"))
    assert _serious(ref, "Age: [___] 35 Name: Ana\n") == []


def test_f6_number_after_a_blank_marker_is_invented_for_a_blank_field():
    ref = _ref(_f(None, "Age"), _f("Ana", "Name"))
    assert _charged(ref, "Age: [___] 35 Name: Ana\n") != []


def test_f9_wrong_data_cell_is_charged_even_when_the_header_repeats_the_value():
    ref = _ref(_table(["Owner name", "Share"], [[_cell("Ada"), _cell("30")]]))
    md = "<table><thead><tr><th>Owner name Ada</th><th>Share</th></tr></thead>"
    md += "<tbody><tr><td>Bo</td><td>30</td></tr></tbody></table>"
    assert ("Owner name", "Ada") in _charged(ref, md)


def test_f10_header_row_is_not_read_as_the_first_record():
    ref = _ref(_table(["Name", "Age"], [[_cell("Ada"), _cell("30")], [_cell("Bo"), _cell("40")]]))
    md = "<table><tr><td>Name</td><td>Age</td></tr><tr><td>Name Ada</td><td>Age 30</td></tr>"
    md += "<tr><td>Bo</td><td>40</td></tr></table>"
    assert [e for e in score_forms(ref, md).events if e.charged and e.cls == "critical"] == []


def test_f11_wrong_grid_value_is_charged_when_two_captions_sit_above_the_grid():
    ref = _ref(_table(["Name", "Number", "Balance"], [[_cell("Alice"), _cell("12345"), _cell("900")]]))
    md = "Number\n\nBalance\n\n| Name | No. | Value |\n|---|---|---|\n| Alice | 98765 | 900 |\n\nRef 12345 noted.\n"
    assert ("Number", "12345") in _charged(ref, md)


def test_f12_a_notes_column_is_not_the_second_sections_value():
    sec = lambda v: {"type": "section", "items": [_f(v, "Name")]}  # noqa: E731
    ref = _ref({"type": "section", "label": "Tenants", "items": [sec("Ann"), sec("Ben")]})
    md = "Tenants\n\n| Field | Value | Notes |\n|---|---|---|\n| Name | Ann | Ben |\n"
    assert ("Name", "Ben") in _charged(ref, md)


def test_f13_two_value_lines_do_not_credit_three_fields():
    ref = _ref(_f("500", "First"), _f("500", "Second"), _f("500", "Third"))
    assert _serious(ref, "First\n\n500\n\nSecond\n\n500\n\nThird\n") != []


def test_f14_formatted_amount_above_its_caption_releases_the_next_line():
    ref = _ref(_f("$1,200.00", "Amount paid"), _f("42", "Amount due"))
    assert _serious(ref, "1200\n\nAmount paid\n\n42\n\nAmount due\n") == []


def test_f15_another_options_box_inside_a_wrapped_label_is_not_its_box():
    label = "I consent to receive notices by email at the address given above"
    ref = _ref({"type": "field", "field": "checkbox", "label": label, "value": True})
    md = "I consent to receive notices by email\n[x] Marketing\nat the address given above\n"
    assert _charged(ref, md) != []


def test_lettered_person_columns_each_read_their_section():
    sec = lambda v: {"type": "section", "items": [_f(v, "Name")]}  # noqa: E731
    ref = _ref({"type": "section", "label": "Tenants", "items": [sec("Ann"), sec("Ben")]})
    md = "Tenants\n\n| Field | Person A | Person B |\n|---|---|---|\n| Name | Ann | {} |\n"
    assert _charged(ref, md.format("Ben")) == []
    assert ("Name", "Ben") in _charged(ref, md.format("Bo"))


def test_a_later_record_is_read_when_the_first_record_is_missing():
    ref = _ref(_table(["Name", "Age"], [[_cell("Ada"), _cell("30")], [_cell("Bo"), _cell("40")]]))
    md = "<table><tr><td>Name</td><td>Age</td></tr><tr><td>Name Bo</td><td>Age {}</td></tr></table>"
    assert sorted(_charged(ref, md.format("40"))) == [("Age", "30"), ("Name", "Ada")]
    assert ("Age", "40") in _charged(ref, md.format("41"))


def test_a_copy_holding_values_of_two_records_opens_neither():
    ref = _ref(_table(["Name", "Age"], [[_cell("Ada"), _cell("30")], [_cell("Bo"), _cell("40")]]))
    md = "<table><tr><td>Name</td><td>Age</td></tr><tr><td>Name Bo</td><td>Age 30</td></tr></table>"
    assert ("Name", "Bo") not in [(e.label, e.got) for e in score_forms(ref, md).events if e.charged]
