"""table_next: sub-rows under a first-column rowspan, written as stacked lines in one output row."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import score_table_next

REF = """<table>
<tr><th>Code</th><th>Plan</th><th>Form</th><th>Charge</th></tr>
<tr><td rowspan="3">116.4</td><td rowspan="3">Single parcel</td><td>Owner:</td><td></td></tr>
<tr><td>Extended</td><td>20%</td></tr>
<tr><td>Standard</td><td>10%</td></tr>
<tr><td rowspan="3">116.5</td><td rowspan="3">Housing unit</td><td>Owner:</td><td></td></tr>
<tr><td>Extended</td><td>$50</td></tr>
<tr><td>Standard</td><td>$75</td></tr>
<tr><td rowspan="3">116.6</td><td rowspan="3">Multiple parcels</td><td>Lender:</td><td></td></tr>
<tr><td>Extended</td><td>N/C</td></tr>
<tr><td>Standard</td><td>30%</td></tr>
</table>"""


def stacked(charges: tuple[str, str, str]) -> str:
    rows = [("116.4", "Single parcel", "Owner:", charges[0]), ("116.5", "Housing unit", "Owner:", charges[1])]
    rows.append(("116.6", "Multiple parcels", "Lender:", charges[2]))
    body = "".join(
        f"<tr><td>{c}</td><td>{p}</td><td>{h}<br>Extended<br>Standard</td><td>{v}</td></tr>" for c, p, h, v in rows
    )
    return f"<table><tr><th>Code</th><th>Plan</th><th>Form</th><th>Charge</th></tr>{body}</table>"


def test_flagged_page_accepts_sub_rows_stacked_as_lines_in_one_row():
    out = stacked(("20%<br>10%", "$50<br>$75", "N/C<br>30%"))
    assert score_table_next(REF, out, allow_merged_sub_rows=True).score == 1.0


def test_unflagged_page_keeps_sub_rows_as_rows():
    out = stacked(("20%<br>10%", "$50<br>$75", "N/C<br>30%"))
    assert score_table_next(REF, out).score < 0.5


def test_stacked_lines_out_of_order_stay_wrong():
    # every Charge cell lists its lines out of order: only the Plan and Form cells count
    out = stacked(("10%<br>20%", "$75<br>$50", "30%<br>N/C"))
    assert score_table_next(REF, out, allow_merged_sub_rows=True).score < 0.75


def test_flagged_page_still_scores_sub_rows_written_as_rows_in_full():
    assert score_table_next(REF, REF, allow_merged_sub_rows=True).score == 1.0
