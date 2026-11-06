"""forms_v2: repeated blocks of one form printed side by side in one output table."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]


def _statement(letter: str, state_id: str | None, wages: str | None) -> dict[str, Any]:
    def text(label: str, value: str | None) -> dict[str, Any]:
        return {"type": "field", "field": "text", "label": label, **({"value": value} if value else {"isEmpty": True})}

    items = [text("EMPLOYER STATE WITHHOLDING ID", state_id), text("STATE WAGES / INCOME", wages)]
    return {"type": "section", "label": f"(INCOME STATEMENT {letter})", "items": items}


STATEMENTS = reference_from_tree(
    [{"page_number": 1, "items": [_statement("D", None, "1,250.00"), _statement("E", "Ada Moss", None)]}]
)
STATEMENTS_MD = (
    '<table><tr><td colspan="2">(INCOME STATEMENT D)</td><td colspan="2">(INCOME STATEMENT E)</td></tr>'
    '<tr><td colspan="2">3. EMPLOYER STATE WITHHOLDING ID</td>'
    '<td colspan="2">3. EMPLOYER STATE WITHHOLDING ID {id}</td></tr>'
    '<tr><td colspan="2">4. STATE WAGES / INCOME {wages}</td><td colspan="2">4. STATE WAGES / INCOME</td></tr>'
    "</table>"
)


def test_each_block_reads_the_column_under_its_own_heading():
    assert _charged(STATEMENTS, STATEMENTS_MD.format(id="Ada Moss", wages="1,250.00")) == []


def test_a_wrong_value_under_a_block_heading_is_still_charged():
    charged = _charged(STATEMENTS, STATEMENTS_MD.format(id="Ada Moss", wages="1,520.00"))
    assert ("STATE WAGES / INCOME", "1,250.00") in charged


def _f(value: str | None) -> dict[str, Any]:
    return {"items": [{"type": "field", "field": "text", **({"isEmpty": True} if value is None else {"value": value})}]}


HOURS = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "table",
                    "columns": ["Day", "Hours", "Minutes"],
                    "rows": [[str(d), _f(h), _f(m)] for d, h, m in ((1, "07", "15"), (2, "08", None), (3, "06", "45"))]
                    + [["4", _f(None), _f("30")]],
                }
            ],
        }
    ]
)


def _hours_md(day3_hours: str) -> str:
    return (
        "| Day | Hours | Minutes | Day | Hours | Minutes |\n|---|---|---|---|---|---|\n"
        f"| 1 | 07 | 15 | 3 | {day3_hours} | 45 |\n| 2 | 08 | | 4 | | 30 |\n"
    )


def test_a_table_printed_as_side_by_side_blocks_reads_each_row_in_its_block():
    assert _charged(HOURS, _hours_md("06")) == []


def test_a_wrong_value_in_a_side_by_side_block_is_still_charged():
    assert ("3 / Hours", "06") in _charged(HOURS, _hours_md("09"))


def _currency(letter: str, rows: list[tuple[str | None, str | None]]) -> dict[str, Any]:
    table = {"type": "table", "columns": ["Currency", "Number of shares"], "rows": [[_f(a), _f(b)] for a, b in rows]}
    return {"type": "section", "label": f"Currency table {letter}", "items": [table]}


CURRENCIES = reference_from_tree(
    [{"page_number": 1, "items": [_currency("A", [("GBP", "1,200")]), _currency("B", [("EUR", "3,400")])]}]
)


def _currencies_md(eur_shares: str) -> str:
    return (
        "| Currency | Number of shares |\n|---|---|\n| One table per currency | |\n\n"
        "| Currency table A | |\n|---|---|\n| GBP | 1,200 |\n\n"
        f"| Currency table B | |\n|---|---|\n| EUR | {eur_shares} |\n"
    )


def test_caption_less_sub_tables_under_one_header_table_read_by_their_label():
    assert _charged(CURRENCIES, _currencies_md("3,400")) == []


def test_a_wrong_value_in_a_labelled_sub_table_is_still_charged():
    assert ("Number of shares", "3,400") in _charged(CURRENCIES, _currencies_md("4,300"))


def _hours_table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"type": "table", "columns": columns, "rows": rows}


ROSTER = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                _hours_table(["Day", "Hours"], [["1", _f("07")]]),
                _hours_table(["Day", "Hours", "Day", "Hours"], [["2", _f("08"), "3", _f("06")]]),
                _hours_table(["Day", "Hours", "Day", "Hours"], [["4", _f("05"), "5", _f("04")]]),
            ],
        }
    ]
)


def test_a_header_the_reference_prints_whole_is_not_split_into_blocks():
    md = (
        "| Day | Hours |\n|---|---|\n| 1 | 07 |\n\n"
        "| Day | Hours | Day | Hours |\n|---|---|---|---|\n| 2 | 08 | 3 | 06 |\n\n"
        "| Day | Hours | Day | Hours |\n|---|---|---|---|\n| 4 | 05 | 5 | 04 |\n"
    )
    assert _charged(ROSTER, md) == []


NAMED = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "section",
                    "label": "Currency table B",
                    "items": [
                        {
                            "type": "table",
                            "columns": ["Currency", "Number of shares"],
                            "rows": [[c, _f(n)] for c, n in (("GBP", "1,200"), ("EUR", "3,400"), ("USD", "5,600"))],
                        }
                    ],
                }
            ],
        }
    ]
)


def test_a_label_only_sub_table_does_not_replace_the_grid_holding_the_rows():
    md = (
        "| Currency | Number of shares |\n|---|---|\n| One table per currency | |\n\n"
        "| Currency table B | |\n|---|---|\n\n"
        "| GBP | 1,200 |\n|---|---|\n| EUR | 3,400 |\n| USD | 5,600 |\n"
    )
    assert _charged(NAMED, md) == []


def test_a_row_header_in_the_caption_grid_without_its_values_keeps_the_labelled_sub_table():
    md = (
        "| Currency | Number of shares |\n|---|---|\n| GBP | See below |\n\n"
        "| Currency table B | |\n|---|---|\n| GB | 1,200 |\n| EU | 3,400 |\n| US | 5,600 |\n"
    )
    assert [x for x in _charged(NAMED, md) if x[0].endswith("Number of shares")] == []
