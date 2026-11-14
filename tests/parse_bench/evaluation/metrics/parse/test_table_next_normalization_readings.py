"""table_next: the row-serial join and the column-index-row drop are readings that never lower a faithful output."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import score_table_next

ROWS = [("(A)", "Written premium", "$89,074"), ("(B)", "Base rate change", "$1,250"), ("(C)", "Total change", "$2,500")]
ROWS2 = [("(D)", "Earned premium", "$125,210"), ("(E)", "Premium impact", "$3,100")]


def _serial_table(section_colspan: int) -> str:
    """Serial column, then a section row written in the serial column alone or spanning serial and label."""
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in ROWS)
    if section_colspan == 2:
        section = "<tr><td colspan='2'>Proposed changes</td><td>Dollar amount</td></tr>"
    else:
        section = "<tr><td>Proposed changes</td><td></td><td>Dollar amount</td></tr>"
    body2 = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in ROWS2)
    return f"<table>{body}{section}{body2}</table>"


def test_serial_column_joined_on_one_side_only_keeps_a_faithful_output_at_one() -> None:
    # the section row spans serial and label in the output only, so only the output's serials qualify for the join
    assert score_table_next(_serial_table(1), _serial_table(2)).score == 1.0


def test_serial_reading_still_charges_a_wrong_value() -> None:
    wrong = _serial_table(2).replace("$1,250", "$1,520")
    assert score_table_next(_serial_table(1), wrong).score < 1.0


YEARS = ("2014", "2015", "2016")
BODY = (
    "<tr><td>Bodily injury</td><td>6,366</td><td>7,277</td><td>8,142</td></tr>"
    "<tr><td>Property damage</td><td>7,252</td><td>8,317</td><td>9,295</td></tr>"
    "<tr><td>Collision</td><td>9,069</td><td>11,242</td><td>13,795</td></tr>"
)
HEAD_IN_TEXT = (
    "<table><tr><td></td>"
    + "".join(f"<th>({k}) Acc Yr 12/31/{y}</th>" for k, y in enumerate(YEARS, 1))
    + "</tr>"
    + BODY
    + "</table>"
)
HEAD_AS_ROWS = (
    "<table><tr><th></th><th>(1)</th><th>(2)</th><th>(3)</th></tr>"
    "<tr><th></th><th>Acc Yr</th><th>Acc Yr</th><th>Acc Yr</th></tr>"
    "<tr><th></th>" + "".join(f"<th>12/31/{y}</th>" for y in YEARS) + "</tr>" + BODY + "</table>"
)


def test_column_index_written_inside_the_reference_header_matches_an_index_row() -> None:
    # the output prints the column numbers as their own header row: dropping that row alone loses every header
    assert score_table_next(HEAD_IN_TEXT, HEAD_AS_ROWS).score > 0.9


def test_index_row_reading_still_charges_a_wrong_value() -> None:
    assert score_table_next(HEAD_IN_TEXT, HEAD_AS_ROWS.replace("7,277", "7,727")).score < 0.95


MARKED = [("Offshore Asia", "0.55%"), ("Malaysia", "0.51%"), ("Poland", "0.34%"), ("Brazil", "0.31%")]


def _marked_table(cell: str) -> str:
    rows = "".join(f"<tr><td>{cell.format(a)}</td><td>{b}</td></tr>" for a, b in MARKED)
    return f"<table><tr><th>Country</th><th>Effect</th></tr>{rows}</table>"


def test_superscript_marker_written_as_plain_text_after_a_space_scores_one() -> None:
    # the reference marks "Poland <sup>#</sup>", the output drops the tag and keeps "Poland #"
    assert score_table_next(_marked_table("{} <sup>#</sup>"), _marked_table("{} #")).score == 1.0


def test_marker_reading_still_charges_a_wrong_value() -> None:
    wrong = _marked_table("{} #").replace("0.34%", "0.43%")
    assert score_table_next(_marked_table("{} <sup>#</sup>"), wrong).score < 1.0
