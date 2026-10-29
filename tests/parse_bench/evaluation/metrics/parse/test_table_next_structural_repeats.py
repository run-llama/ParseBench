"""table_next: repeated floor events that share one structural cause are charged once."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.table_next import score_table_next

COUNTIES = ["Kern", "Orange", "Riverside", "San Diego", "Ventura"]
REGIONS = [["North", "1,284", "1,352"], ["Europe", "846", "901"], ["Asia", "512", "618"], ["Africa", "64", "88"]]
STAFF = [["Toronto", "412"], ["Berlin", "305"], ["Paris", "211"], ["Lima", "98"]]


def html(rows: list[list[str]]) -> str:
    """One HTML table; the first row is the header."""
    tags = ["th"] + ["td"] * (len(rows) - 1)
    body = "".join("<tr>" + "".join(f"<{t}>{c}</{t}>" for c in r) + "</tr>" for t, r in zip(tags, rows, strict=True))
    return f"<table>{body}</table>"


def spanned_factor(values: list[str]) -> str:
    """Location / Factor table; the first five rows share one factor cell when ``values`` has one entry."""
    rows = ["<tr><th>Location</th><th>Factor</th></tr>"]
    for i, county in enumerate(COUNTIES):
        if len(values) == 1:
            cell = f'<td rowspan="{len(COUNTIES)}">{values[0]}</td>' if i == 0 else ""
        else:
            cell = f"<td>{values[i]}</td>"
        rows.append(f"<tr><td>{county}</td>{cell}</tr>")
    rows.append("<tr><td>All Other</td><td>1.000</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def events(res: object) -> dict[str, int]:
    out: dict[str, int] = {}
    for kind, _, _, sev in res.events.items:  # type: ignore[attr-defined]
        if sev not in ("detail", "info"):
            out[kind] = out.get(kind, 0) + 1
    return out


REF = html([["Region", "2023", "2024"], *REGIONS]) + "\n\n" + spanned_factor(["1.152"])
REF2 = html([["Region", "2023", "2024"], *REGIONS]) + "\n\n" + html([["Office", "Staff"], *STAFF])


def test_value_of_a_rowspan_written_once_is_one_major():
    out = html([["Region", "2023", "2024"], *REGIONS]) + "\n\n" + spanned_factor(["", "", "1.152", "", ""])
    res = score_table_next(REF, out)
    assert events(res) == {"spanned cell not repeated": 1}
    assert res.floor == pytest.approx(0.95) and res.score == pytest.approx(res.proportional)


def test_value_of_a_rowspan_lost_everywhere_is_one_critical():
    out = html([["Region", "2023", "2024"], *REGIONS]) + "\n\n" + spanned_factor(["", "", "", "", ""])
    res = score_table_next(REF, out)
    assert events(res) == {"row missing": 1} and res.floor == pytest.approx(0.9)


def test_dropping_a_whole_table_never_beats_dropping_part_of_it():
    as_text = html([["Region", "2023", "2024"], *REGIONS]) + "\n\n" + "\n".join(f"{a}: {b}" for a, b in STAFF)
    half = html([["Region", "2023", "2024"], *REGIONS]) + "\n\n" + html([["Office", "Staff"], *STAFF[:2]])
    whole, part = score_table_next(REF2, as_text), score_table_next(REF2, half)
    assert ("table missing", "1", "4", "info") in whole.events.items
    assert events(whole) == {"row missing": 4} and events(part) == {"row missing": 2}
    assert whole.score <= part.score and whole.floor <= part.floor


def test_two_tables_in_swapped_order_is_one_order_event():
    out = html([["Office", "Staff"], *STAFF]) + "\n\n" + html([["Region", "2023", "2024"], *REGIONS])
    res = score_table_next(REF2, out)
    assert events(res) == {"row out of order": 1} and res.floor == pytest.approx(0.95)


def test_independent_errors_still_count_one_by_one():
    # distinct rows dropped, distinct spanned values changed and rows reordered inside a table are not collapsed
    dropped = html([["Region", "2023", "2024"], REGIONS[0], REGIONS[3]]) + "\n\n" + html([["Office", "Staff"], *STAFF])
    assert events(score_table_next(REF2, dropped)) == {"row missing": 2}
    moved = html([["Region", "2023", "2024"], REGIONS[3], REGIONS[2], REGIONS[1], REGIONS[0]])
    moved += "\n\n" + html([["Office", "Staff"], *STAFF])
    assert events(score_table_next(REF2, moved))["row out of order"] == 3
    out = html([["Region", "2023", "2024"], *REGIONS]) + "\n\n" + spanned_factor(["", "", "1.152", "", ""])
    worse = out.replace("1.152", "1.125")  # the one value written is wrong too: not a span-only loss
    assert score_table_next(REF, worse).score < score_table_next(REF, out).score
