from __future__ import annotations

from parse_bench.evaluation.metrics.parse.test_rules import create_test_rule

GRID = {
    "rows": [{"cells": [{"text": h} for h in ("Zip", "BI", "PD")]}]
    + [{"cells": [{"text": str(80000 + i)}, {"text": f"1.{i:03d}"}, {"text": f"0.{i:03d}"}]} for i in range(20)]
}


def _table(rows: list[list[str]]) -> str:
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table>{body}</table>"


def _rows() -> list[list[str]]:
    return [[c["text"] for c in row["cells"]] for row in GRID["rows"]]


def _rule() -> object:
    return create_test_rule(
        {
            "type": "tables_values",
            "table_variations": [GRID],
            "table_match_threshold": 0.9,
            "table_values_match_threshold": 1.0,
        }
    )


def test_one_wrong_cell_fails_with_a_message_and_no_third_value() -> None:
    """Bug caught: a string returned as a third item is read as a score by the metric, and the whole page loses
    its rule results instead of failing this rule."""
    rows = _rows()
    rows[7][1] = "9.999"
    result = _rule().run(_table(rows))  # type: ignore[attr-defined]
    assert len(result) == 2
    assert result[0] is False
    assert "cell exact match 62/63" in result[1]


def test_exact_table_still_passes() -> None:
    result = _rule().run(_table(_rows()))  # type: ignore[attr-defined]
    assert result[0] is True and len(result) == 2
