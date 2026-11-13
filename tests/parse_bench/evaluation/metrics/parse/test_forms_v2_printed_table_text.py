"""forms_v2: printed text in GT table cells is an anchor, never an expected filled value."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _f(value: str | None, label: str | None = None) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "field", "field": "text"}
    if label:
        node["label"] = label
    node.update({"isEmpty": True} if value is None else {"value": value})
    return {"items": [node]}


def _ref(table: dict[str, Any]):
    return reference_from_tree([{"page_number": 1, "items": [{"type": "table", **table}]}])


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]


DOCS = _ref(
    {
        "columns": ["Documents", "Submitted", "Number submitted"],
        "rows": [
            ["Alpha form (Mandatory)", _f("Yes"), _f("2")],
            ["Beta deed (Mandatory)", _f("No"), _f("1")],
            [_f("Gamma letter"), _f("Yes"), _f("3")],
        ],
    }
)


def test_printed_row_names_under_a_column_caption_are_not_expected_values():
    md = "Documents\n\nAlpha form (Mandatory)\n\nBeta deed (Mandatory)\n\nGamma letter\n\n"
    md += "| Submitted | Number submitted |\n|---|---|\n| Yes | 2 |\n| No | 1 |\n| Yes | 3 |\n"
    assert not [x for x in _charged(DOCS, md) if "Alpha" in x[1] or "Beta" in x[1]]


def test_a_filled_cell_beside_a_printed_row_name_is_still_scored():
    md = "| Documents | Submitted | Number submitted |\n|---|---|---|\n| Alpha form (Mandatory) | Yes | 9 |\n"
    md += "| Beta deed (Mandatory) | No | 1 |\n| Gamma letter | Yes | 3 |\n"
    assert _charged(DOCS, md) != []
    ok = md.replace("| 9 |", "| 2 |")
    assert _charged(DOCS, ok) == []


TOTALS = _ref(
    {
        "columns": ["Currency", "Class of shares", "Number of shares", "Total"],
        "rows": [
            ["Currency table A", None, None, None],
            [_f("GBP"), _f("Ordinary"), _f("100"), None],
            [_f("EUR"), _f("Preference"), _f("50"), None],
            [None, "Totals", _f("100"), _f("1,000")],
        ],
    }
)


def test_printed_totals_row_label_and_its_span_are_not_fields():
    md = (
        "Currency table A\n\n| Currency | Class of shares | Number of shares | Total |\n|---|---|---|---|\n"
        "| GBP | Ordinary | 100 | |\n| EUR | Preference | 50 | |\n| Totals | | 100 | 1,000 |\n"
    )
    assert _charged(TOTALS, md) == []


def test_bare_values_in_a_table_without_fill_in_fields_stay_scored():
    ref = _ref({"columns": ["State tax withheld"], "rows": [["1,451.45"], ["592.55"]]})
    md = "| State tax withheld |\n|---|\n| 1,451.45 |\n| 592.55 |\n"
    assert _charged(ref, md) == []
    assert _charged(ref, md.replace("592.55", "529.55")) != []


def test_printed_day_numbers_head_the_rows_of_a_fill_in_grid():
    ref = _ref({"columns": ["Day", "Hours"], "rows": [["1", _f("8")], ["2", _f("6")]]})
    md = "| Day | Hours |\n|---|---|\n| 1 | 8 |\n| 2 | 6 |\n"
    assert _charged(ref, md) == []
    assert _charged(ref, md.replace("| 6 |", "| 7 |")) != []
