"""forms_v2: printed box codes and line numbers between a caption and its value are not values."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, render_reference, score_forms


def _page(*items: dict[str, Any]) -> Any:
    return reference_from_tree([{"page_number": 1, "items": list(items)}])


def _text(label: str, value: str | None = None, pid: str = "") -> dict[str, Any]:
    field: dict[str, Any] = {"type": "field", "field": "text", "label": label}
    if pid:
        field["id"] = pid
    if value is None:
        field["isEmpty"] = True
    else:
        field["value"] = value
    return field


def _critical(ref: Any, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.kind) for e in score_forms(ref, md).events if e.cls == "critical"]


# A return whose rows print a 5-digit box code before the amount box and a line number after it.
CODES_REF = _page(
    _text("Employment income", None, "1"),
    _text("Commissions included", "283,621", "10120"),
    _text("Other employment income", "100,379", "2"),
    _text("Old age security pension", None, "3"),
    _text("CPP or QPP benefits", "71,409", "4"),
)


def _codes_table(other_income: str) -> str:
    rows = [
        ("Employment income", "10100", "", "1"),
        ("Commissions included", "10120", "283,621", ""),
        ("Other employment income", "10400", other_income, "2"),
        ("Old age security pension", "11300", "+", "3"),
        ("CPP or QPP benefits", "11400", "+ 71,409", "4"),
    ]
    return "<table>" + "".join(f"<tr><td>{'</td><td>'.join(r)}</td></tr>" for r in rows) + "</table>"


def test_box_code_column_before_the_amount_is_read_past():
    res = score_forms(CODES_REF, _codes_table("+ 100,379"))
    assert [e.kind for e in res.events if e.lane != "ids"] == []


def test_wrong_amount_after_a_box_code_is_still_wrong():
    assert _critical(CODES_REF, _codes_table("+ 900,000")) == [("Other employment income", "wrong value")]


# A two-column layout written as one table: the right half's box numbers and code letters sit in column 3.
TWO_COLUMN_REF = _page(
    _text("Ordinary business income", "338,937.94", "1"),
    {"type": "section", "id": "14", "label": "Self-employment earnings"},
    _text("Net rental income", None, "2"),
    {"type": "section", "id": "15", "label": "Credits"},
    _text("Interest income", None, "5"),
    {"type": "section", "id": "17", "label": "Alternative minimum tax items"},
    _text("Royalties", None, "7"),
)


def _two_column_table(royalties: str) -> str:
    rows = [
        ("1", "Ordinary business income", "14", "Self-employment earnings"),
        ("", "338,937.94", "C", "129,083.41"),
        ("2", "Net rental income", "15", "Credits"),
        ("5", "Interest income", "17", "Alternative minimum tax items"),
        ("", "", "C", "14,376.56"),
        ("7", "Royalties", royalties, "10,932.53"),
    ]
    return "<table>" + "".join(f"<tr><td>{'</td><td>'.join(r)}</td></tr>" for r in rows) + "</table>"


def test_codes_of_the_next_form_column_are_not_values():
    assert _critical(TWO_COLUMN_REF, _two_column_table("A")) == []


def test_value_invented_next_to_a_caption_is_still_critical():
    assert _critical(TWO_COLUMN_REF, _two_column_table("5,000")) == [("Royalties", "invented value")]


DOTTED_REF = _page(
    _text("Country of residence", None, "1.8"),
    _text("Country code", "81", "1.9"),
    _text("Tax id. no. in country of residence (TIN)", None, "1.10"),
)


def test_dotted_id_opening_the_next_caption_is_not_a_value():
    md = (
        "<table><tr><td>1.8 Country of residence</td><td>1.9 Country code</td>"
        "<td>1.10 Tax id. no. in country of residence (TIN)</td></tr>"
        "<tr><td></td><td>81</td><td></td></tr></table>"
    )
    res = score_forms(DOTTED_REF, md)
    assert [e.kind for e in res.events] == []
    assert score_forms(DOTTED_REF, render_reference(DOTTED_REF)).score == 1.0
