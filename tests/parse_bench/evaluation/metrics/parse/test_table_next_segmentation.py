"""table_next: stacked sub-tables segmented differently in the reference and in the output."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.table_next import score_table_next

HEAD = ["", "Q1", "Q2", "Full Year"]


def block(title: str, rows: list[list[str]]) -> list[str]:
    """A titled sub-table: a th band spanning every column, a th header row, then data rows."""
    out = [
        f'<tr><th colspan="{len(HEAD)}">{title}</th></tr>',
        "<tr>" + "".join(f"<th>{c}</th>" for c in HEAD) + "</tr>",
    ]
    return out + ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]


CURRENT = [["FY26", "May 02", "Aug 01", "Jan 30"], ["Base: FY25", "May 03", "Aug 02", "Jan 31"]]
PRIOR = [["FY25", "Apr 26", "Jul 26", "Jan 31"], ["Base: FY24", "Apr 28", "Jul 28", "Feb 02"]]


def table(*rows: list[str]) -> str:
    return "<table>\n" + "\n".join(r for rs in rows for r in rs) + "\n</table>"


def test_band_inside_one_table_matches_the_band_starting_a_table():
    as_two = table(block("FY26 Sales", CURRENT)) + "\n\n" + table(block("Comparison Period: FY25 Sales", PRIOR))
    as_one = table(block("FY26 Sales", CURRENT), block("Comparison Period: FY25 Sales", PRIOR))
    assert score_table_next(as_two, as_one).score == pytest.approx(1.0)
    assert score_table_next(as_one, as_two).score == pytest.approx(1.0)
    wrong = as_one.replace("Apr 28", "Apr 29")
    assert score_table_next(as_two, wrong).score <= 0.9  # a wrong date is still a critical


LINES_HEAD = ["Line", "Description", "Prior Year", "Recent Year", "Projected"]
LINES = [
    ["(1)", "Written Premium", "20,144", "22,835", ""],
    ["(2)", "Earned Premium", "19,870", "22,091", ""],
    ["(3)", "Trend Factor", "", "0.981", "-0.8%"],
    ["(4)", "Expense Ratio", "", "", "0.6%"],
    ["(5)", "Federal Tax Rate", "", "", "29.1%"],
    ["(6)", "Projected Yield", "", "", "7.3%"],
]


def rows_html(rows: list[list[str]], head: int = 0) -> str:
    tags = ["th" if i < head else "td" for i in range(len(rows))]
    return "\n".join("<tr>" + "".join(f"<{t}>{c}</{t}>" for c in r) + "</tr>" for t, r in zip(tags, rows, strict=True))


def test_table_continued_after_a_note_is_read_as_one_table():
    ref = f"<table>\n{rows_html([LINES_HEAD, *LINES], head=1)}\n</table>"
    first = f"<table>\n{rows_html([LINES_HEAD, *LINES[:3]], head=1)}\n</table>"
    rest = f"<table>\n{rows_html(LINES[3:])}\n</table>"  # headerless, its two year columns empty
    assert score_table_next(ref, first + "\n<p><i>Lines 4-6 link to page 3.</i></p>\n" + rest).score == 1.0
    res = score_table_next(ref, first + "\n\n" + rest)  # nothing on the page between: still one split
    assert res.score == pytest.approx(0.95)
    shifted = rest.replace("<td></td><td>0.6%</td>", "<td>0.6%</td><td></td>")
    assert score_table_next(ref, first + "\n<p>Note.</p>\n" + shifted).score <= 0.9  # a value in the wrong column
