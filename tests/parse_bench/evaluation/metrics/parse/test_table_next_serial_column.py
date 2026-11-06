"""table_next: a row serial in its own column keys the row like the serial written into the label cell."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import score_table_next

HEAD = ["Item", "2011", "2012"]
ROWS = [
    ("(1)", "Losses Paid", "431,206", "398,537"),
    ("(2)", "Changes in Unpaid Losses", "26,815", "119,764"),
    ("(3)", "Losses Incurred (1) + (2)", "458,021", "518,301"),
    ("(4)", "Allocated Loss Adjustment Expenses", "172,946", "184,203"),
]


def table(rows: list[list[str]], head: list[str]) -> str:
    out = ["<table>", "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"]
    out += ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]
    return "\n".join(out + ["</table>"])


def merged(rows: list[tuple[str, ...]] = ROWS) -> str:
    return table([[f"{n} {lab}", a, b] for n, lab, a, b in rows], HEAD)


def split(rows: list[tuple[str, ...]] = ROWS) -> str:
    return table([list(r) for r in rows], ["", *HEAD])


def test_serial_column_and_serial_in_label_score_the_same() -> None:
    assert score_table_next(merged(), split()).score == 1.0
    assert score_table_next(split(), merged()).score == 1.0
    assert score_table_next(split(), split()).score == 1.0


def test_spanning_serial_prefixes_only_its_first_row() -> None:
    ref = (
        "<table><tr><th colspan='2'>Name of the Associate</th><th>Value</th></tr>"
        "<tr><td>1.</td><td>Balance Sheet Date</td><td>March 31, 2025</td></tr>"
        "<tr><td rowspan='3'>2.</td><td>Shares held</td><td></td></tr>"
        "<tr><td>Number</td><td>10,000</td></tr>"
        "<tr><td>Extent of Holding %</td><td>50%</td></tr></table>"
    )
    out = (
        "<table><tr><th>Name of the Associate</th><th>Value</th></tr>"
        "<tr><td>1. Balance Sheet Date</td><td>March 31, 2025</td></tr>"
        "<tr><td>2. Shares held</td><td></td></tr>"
        "<tr><td>Number</td><td>10,000</td></tr>"
        "<tr><td>Extent of Holding %</td><td>50%</td></tr></table>"
    )
    assert score_table_next(ref, out).score == 1.0


def test_values_moved_to_another_serial_row_still_cost() -> None:
    swapped = [ROWS[1][:2] + ROWS[0][2:], ROWS[0][:2] + ROWS[1][2:], *ROWS[2:]]
    assert score_table_next(merged(), split(swapped)).score < 0.9
