"""forms_v2: reference sub-fields sharing one table cell (DD / MM / YY under one date column)."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, render_reference, score_forms


def _date(dd: str | None, mm: str | None, yy: str | None) -> dict[str, Any]:
    def sub(label: str, value: str | None) -> dict[str, Any]:
        if value is None:
            return {"type": "field", "field": "text", "label": label, "isEmpty": True}
        return {"type": "field", "field": "text", "label": label, "value": value}

    return {"items": [sub("DD", dd), sub("MM", mm), sub("YY", yy)]}


def _name(value: str) -> dict[str, Any]:
    return {"items": [{"type": "field", "field": "text", "value": value}]}


REF = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "table",
                    "label": "Children",
                    "columns": [
                        "Full name",
                        "Date of birth (DD/MM/YY)",
                        "Partnership status (e.g. single, married, partner/de facto, etc.)",
                        "Country of residence",
                    ],
                    "rows": [
                        [_name("Ada Lovelace"), _date("03", "11", "90"), _name("Married"), _name("France")],
                        [_name("Leo Marsh"), _date(None, "2", None), _name("Single"), _name("Sweden")],
                        [_name("Ivy Chen"), _date("X", None, "F"), _name("Partner"), _name("Ireland")],
                    ],
                }
            ],
        }
    ]
)

HEAD = (
    "| Full name | Date of birth (DD/MM/YY) | Partnership status (e.g. single, married, partner/de facto, etc.)"
    " | Country of residence |\n|---|---|---|---|\n"
)


def test_reference_renders_to_one():
    assert score_forms(REF, render_reference(REF)).score == 1.0


def test_date_subfields_written_in_one_cell_are_read_by_position():
    md = (
        HEAD + "| Ada Lovelace | 03 / 11 / 90 | Married | France |\n"
        "| Leo Marsh | / 2 / | Single | Sweden |\n| Ivy Chen | X / / F | Partner | Ireland |\n"
    )
    res = score_forms(REF, md)
    assert [e.kind for e in res.events if e.charged] == []
    assert res.score == 1.0


def test_date_subfields_in_the_wrong_position_are_critical():
    md = (
        HEAD + "| Ada Lovelace | 11 / 03 / 90 | Married | France |\n"
        "| Leo Marsh | 2 / / | Single | Sweden |\n| Ivy Chen | X / / F | Partner | Ireland |\n"
    )
    critical = [(e.label, e.got) for e in score_forms(REF, md).events if e.charged and e.cls == "critical"]
    assert ("DD", "11") in critical and ("MM", "03") in critical  # Ada's day and month swapped
    assert ("DD", "2") in critical  # Leo's month written in the day slot
