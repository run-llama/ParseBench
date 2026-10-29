"""table_next on reference tables whose data cells are almost all dashes."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import score_table_next

YEARS = [str(y) for y in range(2005, 2025)]


def table(rows: list[list[str]]) -> str:
    head = "<tr><th>Year</th><th>Catastrophe</th><th>Ex-Cat</th><th>Total</th></tr>"
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


DASH_ROWS = [[y, "-", "-", "-"] for y in YEARS]
FACTOR_ROW = ["", "", "Selected factor:", "1.000"]


def test_dash_table_with_footer_line_moved_below_is_not_scored_as_missing() -> None:
    ref = table([*DASH_ROWS, FACTOR_ROW])
    out = table(DASH_ROWS) + "\n\nSelected factor: 1.000"

    res = score_table_next(ref, out)

    assert res.score >= 0.8
    assert score_table_next(ref, ref).score == 1.0


def test_numbers_written_into_dash_cells_score_low() -> None:
    ref = table([*DASH_ROWS, FACTOR_ROW])
    invented = table([*([y, "12", "34", "46"] for y in YEARS), FACTOR_ROW])

    assert score_table_next(ref, invented).score < 0.2
