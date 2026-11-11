"""table_next on dashes: reference tables whose data cells are almost all dashes, and blank cells drawn as dashes."""

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


BALANCE = [
    ["", "2025", "2024"],
    ["Assets", "", ""],
    ["Cash", "120", "95"],
    ["Other", "", "7"],
    ["Liabilities", "", ""],
    ["Debt", "80", "60"],
    ["Other", "11", "9"],
]


def _plain(rows: list[list[str]]) -> str:
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


def test_blank_cells_drawn_as_dashes_cost_nothing() -> None:
    # a dash in the corner read as a row-label header, and dashes in the "Assets" row made it a data row,
    # so the two "Other" rows lost their section and the output scored about 0.85
    ref = _plain(BALANCE)
    out = _plain([[c or "\u2014" for c in r] for r in BALANCE])

    assert score_table_next(ref, out).score == 1.0


def test_dash_written_over_a_value_still_costs() -> None:
    dashed = [[c or "\u2014" for c in r] for r in BALANCE]
    dashed[2][1] = "\u2014"  # Cash 2025: 120 on the page

    assert score_table_next(_plain(BALANCE), _plain(dashed)).score < 0.95


def test_table_whose_values_are_all_empty_markers_is_scored() -> None:
    # no value cell, so the page scored 1.0 for any output, a missing table included
    head = "<tr><th>Year ending</th><th>Installment fees</th></tr>"
    rows = [[f"12/31/{y}", "None"] for y in range(2008, 2012)]
    ref = _plain(rows).replace("<table>", "<table>" + head)

    assert score_table_next(ref, ref).score == 1.0
    assert score_table_next(ref, "No table here.").score == 0.0
    assert score_table_next(ref, _plain(rows[:2]).replace("<table>", "<table>" + head)).score < 0.9
