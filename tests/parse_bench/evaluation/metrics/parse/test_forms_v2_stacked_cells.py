"""forms_v2: a value written in the same table cell as its column caption (``Caption<br>Value``)."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _f(value: str | None) -> dict[str, Any]:
    return {"items": [{"type": "field", "field": "text", **({"isEmpty": True} if value is None else {"value": value})}]}


PARTNERS = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "table",
                    "columns": ["Partner name", "Address", "Province", "Share"],
                    "rows": [
                        [_f(None), _f("12 Elm St"), _f("ON"), _f("40.5")],
                        [_f(None), _f("9 Oak Ave"), _f(None), _f("59.5")],
                    ],
                }
            ],
        }
    ]
)


def _md(first: str) -> str:
    rows = [
        '<tr><td colspan="3">Partner name</td></tr>',
        f"<tr><td>Address<br />{first}</td><td>Province<br />ON</td><td>Share<br />40.5 %</td></tr>",
        '<tr><td colspan="3">Partner name</td></tr>',
        "<tr><td>Address<br />9 Oak Ave</td><td>Province</td><td>Share<br />59.5 %</td></tr>",
    ]
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def _charged(md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(PARTNERS, md).events if e.charged]


def test_first_record_value_under_its_caption_in_the_same_cell_is_read():
    assert _charged(_md("12 Elm St")) == []


def test_wrong_first_record_value_in_a_caption_cell_is_still_charged():
    assert ("Address", "12 Elm St") in _charged(_md("77 Pine Rd"))


CHILDREN = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "table",
                    "columns": ["ID number", "Amount", "ID number", "Amount"],
                    "rows": [[_f(None)] * 4, [_f("5,100.20"), _f(None), _f("7,300"), _f(None)]],
                }
            ],
        }
    ]
)


def _children(a: str, b: str) -> list[tuple[str, str]]:
    head = "<thead><tr><th>ID number</th><th>Amount</th><th>ID number</th><th>Amount</th></tr></thead>"
    body = f"<tr><td>ID number<br />{a}</td><td>Amount</td><td>ID number<br />{b}</td><td>Amount</td></tr>"
    md = f"<table>{head}<tbody>{body}</tbody></table>"
    return [(e.label, e.expected) for e in score_forms(CHILDREN, md).events if e.charged]


def test_a_caption_printed_for_two_columns_gives_each_column_its_own_copy():
    assert _children("5,100.20", "7,300") == []


def test_values_swapped_between_two_columns_with_the_same_caption_are_charged():
    assert ("ID number", "5,100.20") in _children("7,300", "5,100.20")


OWNER = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "table",
                    "columns": ["Owner name", "Role", "Share"],
                    "rows": [[_f("Ada Brook"), _f(None), _f("30")]],
                },
                {"type": "field", "field": "text", "label": "Next question", "isEmpty": True},
            ],
        }
    ]
)


def _owner(name: str) -> list[tuple[str, str]]:
    row = f"<tr><td>Owner name<br>{name}</td><td>Role</td><td>Share<br>30</td></tr>"
    md = f"<table><tbody>{row}<tr><td colspan='3'>Next question</td></tr></tbody></table>"
    return [(e.label, e.expected) for e in score_forms(OWNER, md).events if e.charged]


def test_a_one_row_table_written_in_its_caption_cells_is_read_there():
    assert _owner("Ada Brook") == []


def test_a_wrong_value_in_a_one_row_caption_cell_is_charged():
    assert ("Owner name", "Ada Brook") in _owner("Bo Lind")
