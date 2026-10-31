"""table_next: a row label in two cells on one side and in one cell on the other ("1.7 | US governments")."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import TableNextResult, score_table_next

HEAD = ["One year or less", "Over 1 yr through 5 yrs", "Over 5 yrs through 10 yrs"]
ROWS = [
    ["1.7", "US governments", "641,278", "1,023,519", "358,912"],
    ["2.7", "All other governments", "0", "0", "0"],
    ["3.7", "States and territories", "1,927", "0", "583"],
    ["4.7", "Political subdivisions", "69,845", "179,264", "21,406"],
    ["5.7", "Special revenue", "615,839", "2,157,693", "1,384,251"],
    ["6.7", "Public utilities", "24,987", "188,302", "193,615"],
]


def table(head: list[str], rows: list[list[str]]) -> str:
    h = "<tr>" + "".join(f"<th>{x}</th>" for x in head) + "</tr>"
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table><thead>{h}</thead><tbody>{b}</tbody></table>"


def split(rows: list[list[str]]) -> str:
    return table(["Line number", "", *HEAD], rows)


def joined(rows: list[list[str]]) -> str:
    return table(["Line number", *HEAD], [[r[0] + " " + r[1], *r[2:]] for r in rows])


def kinds(res: TableNextResult) -> set[str]:
    return {k for k, *_ in res.events.items}


def test_output_joining_the_two_label_cells_costs_one_major() -> None:
    res = score_table_next(split(ROWS), joined(ROWS))
    assert kinds(res) == {"label columns merged"}
    assert res.score == 0.95


def test_output_splitting_a_one_cell_label_costs_one_major() -> None:
    res = score_table_next(joined(ROWS), split(ROWS))
    assert kinds(res) == {"label columns merged"}
    assert res.score == 0.95


def test_references_score_one_against_themselves() -> None:
    assert score_table_next(split(ROWS), split(ROWS)).score == 1.0
    assert score_table_next(joined(ROWS), joined(ROWS)).score == 1.0


def test_joined_labels_with_wrong_values_stay_low() -> None:
    wrong = [[*r[:2], *("9" + v for v in r[2:])] for r in ROWS]
    assert score_table_next(split(ROWS), joined(wrong)).score < 0.1


def test_dropped_line_numbers_are_not_read_as_joined() -> None:
    dropped = table(["Category", *HEAD], [r[1:] for r in ROWS])
    assert "label columns merged" not in kinds(score_table_next(split(ROWS), dropped))


def test_page_with_a_one_column_table_still_reads_joined_labels() -> None:
    ref = (
        "| Year | Q | Sales | Cost |\n|---|---|---|---|\n| 1986 | Q3 | 10 | 4 |\n| 1986 | Q4 | 12 | 5 |\n"
        "| 1987 | Q1 | 14 | 6 |\n\n| Note |\n|---|\n| a |\n| b |\n"
    )
    out = (
        "| Year | Sales | Cost |\n|---|---|---|\n| 1986 Q3 | 10 | 4 |\n| 1986 Q4 | 12 | 5 |\n"
        "| 1987 Q1 | 14 | 9 |\n\n| Note |\n|---|\n| a |\n| b |\n"
    )
    assert 0.5 < score_table_next(ref, out).score < 1.0
