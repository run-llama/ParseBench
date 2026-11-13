"""table_next: a section heading spanning the full row reads like the same heading left in the label column."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.table_next import score_table_next

# row number, item, count, amount; "Heritage" and "Mail" rows hold no values
ROWS = [
    ["26", "Sports", "", ""],
    ["27", "Health", "1", "6412.00"],
    ["28", "Heritage", "", ""],
    ["29", "Family & Child", "1", "5735.00"],
    None,
    ["30", "Broadcasting", "1", "1502.45"],
    ["31", "Mail", "", ""],
    ["32", "Electronics", "1", "22871.00"],
    ["33", "Telecom", "2", "700.00"],
]


def table(section: str) -> str:
    out = ["<table>"]
    for r in ROWS:
        out.append(section if r is None else "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return "\n".join([*out, "</table>"])


REF = table('<tr><td colspan="4"><b>COMMUNICATION</b></td></tr>')
LABEL_COLUMN = table("<tr><td></td><td><b>COMMUNICATION</b></td><td></td><td></td></tr>")


def test_section_row_in_label_column_matches_full_width_span():
    assert score_table_next(REF, REF).score == pytest.approx(1.0)
    assert score_table_next(REF, LABEL_COLUMN).score == pytest.approx(1.0)


def test_wrong_value_under_label_column_section_still_costs():
    wrong = LABEL_COLUMN.replace("22871.00", "22870.00")
    assert score_table_next(REF, wrong).score < 0.99
