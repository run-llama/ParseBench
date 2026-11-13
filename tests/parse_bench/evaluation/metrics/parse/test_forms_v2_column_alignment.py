"""forms_v2: a row-header table whose output drops a blank spacer column, and a caption in a colspan cell."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, render_reference, score_forms


def _value(v: str | None) -> dict[str, Any]:
    field: dict[str, Any] = {"type": "field", "field": "text"}
    if v is None:
        field["isEmpty"] = True
    else:
        field["value"] = v
    return {"items": [field]}


# row header | blank spacer | printed code | amount, then a total below the table
_REF = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "table",
                    "columns": ["", "", "", ""],
                    "rows": [
                        ["Ramp", None, "914.145.2", _value("361,760")],
                        ["Stair lift", None, "914.145.3", _value("78,777.76")],
                        ["Building works", None, "914.145.4", _value(None)],
                        ["Repairs", None, "914.145.99", _value("193,020")],
                    ],
                },
                {"type": "field", "field": "text", "label": "Total in CHF", "value": "23,479.01"},
            ],
        }
    ]
)
_ROWS = [("Ramp", "914.145.2", "361,760"), ("Stair lift", "914.145.3", "78,777.76")]
_ROWS += [("Building works", "914.145.4", ""), ("Repairs", "914.145.99", "193,020")]


def _table(cells: list[list[str]], total: str) -> str:
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells)
    return f"<table>{body}{total}</table>"


def _charged(md: str) -> list[str]:
    return [e.kind for e in score_forms(_REF, md).events if e.charged]


def test_reference_scores_itself_perfectly():
    assert score_forms(_REF, render_reference(_REF)).score == 1.0


def test_dropped_blank_spacer_column_keeps_values_under_their_columns():
    total = '<tr><td colspan="2"><b>Total in CHF</b></td><td>23,479.01</td></tr>'
    md = _table([[h, code, v] for h, code, v in _ROWS], total)
    assert _charged(md) == []


def test_kept_spacer_column_still_scores_perfectly():
    total = '<tr><td colspan="3"><b>Total in CHF</b></td><td>23,479.01</td></tr>'
    md = _table([[h, "", code, v] for h, code, v in _ROWS], total)
    assert _charged(md) == []


def test_dropped_amount_column_is_not_realigned():
    # the spacer is kept and the amounts are gone: the spacer slot stays where it is, amounts read missing
    total = '<tr><td colspan="2"><b>Total in CHF</b></td><td>23,479.01</td></tr>'
    md = _table([[h, "", code] for h, code, _ in _ROWS], total)
    kinds = _charged(md)
    assert kinds.count("field missing") == 3
    assert "wrong value" not in kinds and "value of another field" not in kinds


def test_empty_cell_after_a_colspan_caption_is_still_blank():
    total = '<tr><td colspan="2"><b>Total in CHF</b></td><td></td></tr>'
    md = _table([[h, code, v] for h, code, v in _ROWS], total)
    assert _charged(md) == ["value lost"]


def test_colspan_caption_with_printed_text_beside_reads_the_value_below():
    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [{"type": "field", "field": "text", "label": "Total dividends", "value": "14.53"}],
            }
        ]
    )
    md = (
        '<table><tr><td colspan="2">1a Total dividends</td><td>OMB No. 1545-0110</td></tr>'
        '<tr><td colspan="2">$ 14.53</td><td>Form 1099</td></tr></table>'
    )
    assert [e.kind for e in score_forms(ref, md).events if e.charged] == []


def _blank_name_ref() -> Any:
    return reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {"type": "field", "field": "text", "label": "1.1 Official name", "isEmpty": True},
                    {"type": "field", "field": "text", "label": "1.2 Tax number", "value": "Canada"},
                    {"type": "field", "field": "text", "label": "1.3 End date", "value": "2001-05-01"},
                ],
            }
        ]
    )


# the output writes each value above its caption; the first caption spans the whole row and is blank
_VALUES_ABOVE = (
    '<table><tr><td colspan="2">1.1 Official name</td></tr>'
    "<tr><td><em>Canada</em></td><td><em>2001-05-01</em></td></tr>"
    "<tr><td>1.2 Tax number</td><td>1.3 End date</td></tr></table>"
)


def test_blank_caption_spanning_the_row_does_not_take_the_next_captions_value():
    kinds = [e.kind for e in score_forms(_blank_name_ref(), _VALUES_ABOVE).events if e.charged]
    assert "value of another field" not in kinds and "invented value" not in kinds


def test_filled_caption_spanning_the_row_still_reads_the_cell_below():
    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [{"type": "field", "field": "text", "label": "1.1 Official name", "value": "Ada"}],
            }
        ]
    )
    md = '<table><tr><td colspan="2">1.1 Official name</td></tr><tr><td>Ada</td><td></td></tr></table>'
    assert [e.kind for e in score_forms(ref, md).events if e.charged] == []
