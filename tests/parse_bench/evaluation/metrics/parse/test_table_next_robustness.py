"""table_next against outputs built to game or break the scorer (synthetic audit cases)."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import clean_grid, parse_tables, score_table_next


def md(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


HEAD = ["Region", "2023", "2024"]
ROWS = [
    ["North America", "1,200", "1,900"],
    ["Western Europe", "1,237", "1,941"],
    ["Japan", "1,274", "1,982"],
    ["China", "1,311", "2,023"],
]
REF = md(HEAD, ROWS)


def test_oversized_spans_are_capped_to_the_table() -> None:
    out = "<table>" + "<tr><td rowspan=1000 colspan=1000>x</td></tr>" * 5 + "</table>"

    g = clean_grid(parse_tables(out)[0]).g

    assert len(g) <= 5 and len(g[0]) <= 25  # five rows; five cells of at most five columns each


def test_title_cell_spanning_the_header_keeps_its_width() -> None:
    ref = (
        '<table><tr><th colspan="3">Sales</th></tr><tr><th>Region</th><th>2023</th><th>2024</th></tr>'
        + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in ROWS)
        + "</table>"
    )

    assert len(clean_grid(parse_tables(ref)[0]).g[0]) == 3
    assert score_table_next(ref, ref).score == 1.0


def test_pipe_table_without_outer_pipes_is_read() -> None:
    bare = "\n".join(line.strip("|").strip() for line in REF.splitlines())

    assert score_table_next(REF, bare).score == 1.0


def test_prose_line_with_a_pipe_is_not_a_table() -> None:
    out = "Revenue | Cost of sales\nboth rose in 2024.\n\n" + REF

    assert len(parse_tables(out)) == 1
    assert score_table_next(REF, out).score == 1.0


def regroup(sep: str, rows: list[list[str]] = ROWS) -> str:
    return md(HEAD, [[r[0], *(v.replace(",", sep) for v in r[1:])] for r in rows])


def test_space_and_apostrophe_thousands_read_like_a_dropped_comma() -> None:
    plain = score_table_next(REF, regroup("")).score  # "1200" for "1,200": number format changed

    for sep in (" ", "\xa0", " ", " ", "'"):
        assert score_table_next(REF, regroup(sep)).score == plain > 0.85


def test_space_grouped_number_with_a_wrong_digit_is_still_wrong() -> None:
    rows = [[*r] for r in ROWS]
    rows[0][1] = "1,300"

    assert score_table_next(REF, regroup(" ", rows)).score < 0.85


def html(head: list[str], rows: list[list[str]]) -> str:
    cells = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return "<table><tr>" + "".join(f"<th>{c}</th>" for c in head) + f"</tr>{cells}</table>"


def test_footnote_digit_glued_to_a_number_is_a_format_slip() -> None:
    for out in (
        REF.replace("1,200", "1,200¹"),
        REF.replace("1,200", "1,200^1"),
        REF.replace("1,200", "1,200<sup>1</sup>"),
        html(HEAD, [[r[0], r[1].replace("1,200", "1,200<sup>1</sup>"), r[2]] for r in ROWS]),
    ):
        assert score_table_next(REF, out).score >= 0.95


def test_power_of_ten_exponent_is_not_a_footnote() -> None:
    ref = md(["Item", "Factor"], [["A", "10⁶"], ["B", "10³"], ["C", "10⁹"]])
    out = md(["Item", "Factor"], [["A", "10"], ["B", "10³"], ["C", "10⁹"]])

    assert score_table_next(ref, out).score < 0.9
    assert score_table_next(ref, ref).score == 1.0


def test_reference_with_footnote_markers_matches_itself() -> None:
    ref = html(HEAD, [[r[0], r[1] + "<sup>1</sup>", r[2] + "<sup>2</sup>"] for r in ROWS])

    assert score_table_next(ref, ref).score == 1.0


STATUS = md(["Item", "Status"], [["A", "Approved"], ["B", "Pending"], ["C", "Approved"], ["D", "Rejected"]])


def test_negated_text_cell_is_a_wrong_value() -> None:
    res = score_table_next(STATUS, STATUS.replace("| A | Approved", "| A | Not approved"))

    assert res.score <= 0.8
    assert any(k == "wrong or misplaced text" for k, *_ in res.events.items)


def test_text_cell_typo_stays_a_small_difference() -> None:
    assert score_table_next(STATUS, STATUS.replace("| A | Approved", "| A | Aproved")).score > 0.9


def test_garbage_row_labels_lose_the_rows() -> None:
    out = md(HEAD, [[f"Zq{i}x", *r[1:]] for i, r in enumerate(ROWS)])

    assert score_table_next(REF, out).score < 0.5


def test_relabelled_row_costs_more_than_a_label_typo() -> None:
    ref = md(["Item", "2023"], [["Net sales", "100"], ["Cost", "60"], ["Net income", "40"]])

    relabelled = score_table_next(ref, ref.replace("Net sales", "Gross sales")).score
    typo = score_table_next(ref, ref.replace("Net sales", "Net slaes")).score

    assert relabelled < 0.95 <= typo


def test_row_label_with_a_note_appended_keeps_its_cells() -> None:
    out = md(HEAD, [*ROWS[:2], ["Japan (Note 7)", *ROWS[2][1:]], *ROWS[3:]])

    assert score_table_next(REF, out).score >= 0.95


def test_row_label_words_reordered_keep_their_cells() -> None:
    out = md(HEAD, [["America North", *ROWS[0][1:]], *ROWS[1:]])

    assert score_table_next(REF, out).score >= 0.95
