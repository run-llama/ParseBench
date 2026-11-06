"""forms_v2: which output table a reference table maps to."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _f(value: str | None) -> dict[str, Any]:
    return {"items": [{"type": "field", "field": "text", **({"isEmpty": True} if value is None else {"value": value})}]}


ACCOUNTS = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "table",
                    "columns": ["Bank name", "Account no.", "Market value"],
                    "rows": [[_f("North Bank"), _f("300239"), _f(None)], [_f(None), _f("170500"), _f(None)]],
                }
            ],
        }
    ]
)

PART3 = "| Item | Market value |\n|---|---|\n| Real property | 925,006 |\n| Cars | 46,196 |\n\n"


def _charged(first_account: str) -> list[tuple[str, str]]:
    flat = f"Bank name\nNorth Bank\n\nAccount no.\n{first_account}\n\nBank name\n\nAccount no.\n170500\n"
    return [(e.label, e.expected) for e in score_forms(ACCOUNTS, PART3 + flat).events if e.charged and e.cls != "minor"]


def test_a_flat_table_is_not_mapped_onto_another_grid_that_shares_one_caption():
    assert _charged("300239") == []


def test_a_wrong_value_in_the_flat_table_is_still_charged():
    assert ("Account no.", "300239") in _charged("300293")
