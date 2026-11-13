"""table_next: cells, titles and HTML shapes that carry the reference content but used to lose points."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import score_table_next, value_sim


def _kinds(res) -> list[str]:  # noqa: ANN001 - TableNextResult
    return [e[0] for e in res.events.items if e[-1] != "detail"]


def _trs(rows: list[list[str]], tag: str = "td") -> str:
    return "".join("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in r) + "</tr>" for r in rows)


def _band(text: str, width: int) -> str:
    return f'<tr><th colspan="{width}">{text}</th></tr>'


def _html(rows: list[list[str]], head: list[str] | None = None, pre: str = "") -> str:
    return f"<table>{pre}{_trs([head], 'th') if head else ''}{_trs(rows)}</table>"


def test_currency_symbol_only_cells_are_empty() -> None:
    ref = "| Item | 2024 |\n|---|---|\n| Revenue | $1,234 |\n| Cost | $ - |\n| Net | $500 |\n| Tax | $12 |"
    out = (
        "| Item | | 2024 |\n|---|---|---|\n| Revenue | $ | 1,234 |\n| Cost | $ | - |\n| Net | $ | 500 |\n"
        "| Tax | $ | 12 |"
    )
    res = score_table_next(ref, out)
    assert "extra column" not in _kinds(res) and "invented row" not in _kinds(res)
    assert res.score >= 0.9  # only the "$" moved out of the amount cells
    assert value_sim("$ -", "-") == 1.0


def test_output_caption_counts_as_the_title() -> None:
    rows = [["a", "1"], ["b", "2"], ["c", "3"]]
    ref = _html(rows, ["Name", "Value"], pre=_band("Quarterly Acquisition Activity", 2))
    out = _html(rows, ["Name", "Value"], pre="<caption>Quarterly Acquisition Activity</caption>")
    res = score_table_next(ref, out)
    assert "title missing" not in _kinds(res)
    assert res.score == 1.0


def test_checkbox_glyphs_equal_task_marks() -> None:
    ref = "| Option | A | B |\n|---|---|---|\n| One | ● | ○ |\n| Two | ○ | ● |\n| Three | ● | ● |\n| Four | ○ | ○ |"
    out = (
        "| Option | A | B |\n|---|---|---|\n| One | [x] | [ ] |\n| Two | [ ] | [x] |\n| Three | [x] | [x] |\n"
        "| Four | [ ] | [ ] |"
    )
    assert score_table_next(ref, out).score >= 0.9
    assert value_sim("○", "[x]") == 0.0  # checked and unchecked stay different


def _marks(marks: tuple[str, str]) -> str:
    """An availability grid: product rows, state columns, ``marks[0]`` where available and ``marks[1]`` elsewhere."""
    rows = [[f"Plan {i}"] + [marks[(i + j) % 3 == 0] for j in range(4)] for i in range(1, 7)]
    return _html(rows, ["Product", "AL", "AK", "AZ", "AR"])


def test_checked_box_glyph_reads_as_a_cross() -> None:
    # LTCprodkit page1 shape: the reference marks cells with "⊗", the output with "☒"; both are a cross
    assert score_table_next(_marks(("⊗", "")), _marks(("☒", ""))).score == 1.0
    # a yes / no table written with boxes: "☑" is "✓" and "☒" is "✗"
    assert score_table_next(_marks(("✗", "✓")), _marks(("☒", "☑"))).score == 1.0
    assert value_sim("☒", "✓") == 0.0  # a wrong cross against a reference check is wrong


def test_task_mark_against_a_letter_x_keeps_partial_credit() -> None:
    # dense_table_380 shape: the reference marks checked cells with "X", the output with "[x]"
    assert value_sim("[x]", "X") == 0.5
    assert 0.0 < score_table_next(_marks(("X", "")), _marks(("[x]", ""))).score < 1.0


def test_unit_after_a_number_ignores_spaces_and_zero_padding() -> None:
    assert value_sim("2/10 (20.0%)", "2/10(20.0%)") == 0.9
    assert value_sim("05/01/2019", "5/1/2019") == 0.9
    assert value_sim("2/01 (20%)", "2/1(20%)") == 0.9
    assert value_sim("2/1. 05", "2/1 .5") == 0.5
    assert value_sim("1.2.05", "1.2.5") == 0.5  # a zero after "." is a digit of a decimal or version
    assert value_sim("3 years or more", "3 years or less") == 0.5


def test_increment_sign_equals_greek_delta() -> None:
    rows = [["Revenue ∆ in %", "5", "6"], ["Margin ∆ in %", "7", "8"], ["Cost", "9", "10"]]
    ref = _html(rows, ["Line", "Q1", "Q2"])
    out = _html([[r[0].replace("∆", "Δ"), *r[1:]] for r in rows], ["Line", "Q1", "Q2"])
    res = score_table_next(ref, out)
    assert "label reworded" not in _kinds(res)
    assert res.score == 1.0


def test_output_title_band_of_several_rows_is_not_the_header() -> None:
    top = '<tr><th rowspan="2">Region</th><th colspan="3">Written</th><th colspan="3">Earned</th></tr>'
    head = top + _trs([["2021", "2022", "2023", "2021", "2022", "2023"]], "th")
    rows = [[f"Region {i}"] + [str(10 * i + j) for j in range(6)] for i in range(1, 7)]
    titles = ["Exhibit 12", "Example Holdings Inc.", "Premium by Region", "(in thousands)"]
    ref = "\n\n".join(titles) + f"\n\n<table><thead>{head}</thead>{_trs(rows)}</table>"
    out = f"<table><thead>{''.join(_band(t, 7) for t in titles)}{head}</thead>{_trs(rows)}</table>"
    assert score_table_next(ref, out).score == 1.0


def test_headerless_continuation_after_a_heading_is_not_missing_a_header() -> None:
    first = [[f"FACTOR_{i}", "", "", "", f"{i}.5%"] for i in range(1, 9)]
    second = [[f"RATE_{i}", f"{i}.1", f"{i}.2", f"{i}.3", f"{i}.4"] for i in range(1, 6)]
    ref = "<table>" + _trs(first) + _band("Alternate Calculation", 5) + _trs(second) + "</table>"
    out = _html([[r[0], r[4]] for r in first]) + "\n\n## ***Alternate Calculation***\n\n" + _html(second)
    res = score_table_next(ref, out, max_top_title_rows=0, trm_unsupported=True)
    assert "header rows missing" not in _kinds(res)
    assert res.score == 1.0


def test_html_with_omitted_end_tags_reads_like_closed_tags() -> None:
    ref = _html([["a", "1"], ["b", "2"]], ["k", "v"])
    assert score_table_next(ref, "<table><tr><th>k<th>v<tr><td>a<td>1<tr><td>b<td>2</table>").score == 1.0


def test_table_nested_in_a_cell_stays_in_its_outer_table() -> None:
    ref = _html([["a", "1"], ["b", "2"]], ["k", "v"])
    nested = (
        "<table><tr><th>k</th><th>v</th></tr><tr><td>a</td><td><table><tr><td>1</td></tr></table></td></tr>"
        "<tr><td>b</td><td>2</td></tr></table>"
    )
    assert score_table_next(ref, nested).score == 1.0


def test_unclosed_table_does_not_swallow_the_tables_after_it() -> None:
    # BRWS page1123 shape, tags open/close O C O O C: the second table's "</table>" is missing, and the third
    # table opens outside any cell, so it is a sibling
    tables = [
        _html([[f"{k}{i}", str(10 * n + i)] for i in range(1, 5)], ["Item", f"Q{n}"]) for n, k in enumerate("abc")
    ]
    ref = "\n\n".join(tables)
    out = "\n\n".join([tables[0], tables[1].removesuffix("</table>"), tables[2]])
    assert score_table_next(ref, out).score == 1.0


# Near misses: shapes that look like the patterns above and must keep their old reading.


def test_currency_with_a_value_is_not_empty() -> None:
    for cell in ("$0", "$ 0", "US$", "$ 1", "($)"):
        assert value_sim(cell, "") == 0.0, cell
    assert value_sim("€ n/a", "-") == 1.0


def test_checkbox_map_is_whole_cell_only_and_keeps_yes_no_marks_apart() -> None:
    assert value_sim("✓", "✗") == 0.0  # yes / no tables
    assert value_sim("[x]", "[ ]") == 0.0  # checked and unchecked task marks stay apart
    assert value_sim("● Yes", "[x] Yes") < 0.9  # a mark inside text is not folded
    assert value_sim("[ x ]", "[x]") < 0.9


def test_zero_padding_stays_significant_after_decimal_point_or_colon() -> None:
    assert value_sim("10:05", "10:5") == 0.5  # 10:5 reads 10:50
    assert value_sim("1.05", "1.5") == 0.0
    assert value_sim("5 mg", "5 g") == 0.5


def test_caption_inside_a_cell_is_not_a_title() -> None:
    rows = [["a", "1"], ["b", "2"], ["c", "3"]]
    ref = _html(rows, ["Name", "Value"], pre=_band("Quarterly Activity", 2))
    out = (
        "<table><tr><th>Name</th><th>Value</th></tr>"
        + _trs(rows)[:-5]
        + "<caption>Quarterly Activity</caption></tr></table>"
    )
    assert "title missing" in _kinds(score_table_next(ref, out))


def test_headerless_fragment_is_still_charged_when_its_reference_part_has_a_header() -> None:
    first = [[f"F{i}", f"{i}.5"] for i in range(1, 6)]
    second = [[f"R{i}", f"{i}.1", f"{i}.2"] for i in range(1, 6)]
    ref = _html(first, ["Factor", "Value"]) + "\n\n## Alt\n\n" + _html(second, ["Rate", "A", "B"])
    out = _html(first, ["Factor", "Value"]) + "\n\n## Alt\n\n" + _html(second)
    assert "header rows missing" in _kinds(score_table_next(ref, out))


def test_full_width_section_rows_after_the_header_stay_section_rows() -> None:
    body = _band("Current assets", 3) + _trs([["Cash", "10", "11"], ["Receivables", "12", "13"]])
    body += _band("Non-current assets", 3) + _trs([["Plant", "20", "21"], ["Goodwill", "22", "23"]])
    ref = "<table>" + _band("Balance Sheet", 3) + _trs([["Item", "2023", "2024"]], "th") + body + "</table>"
    assert score_table_next(ref, ref).score == 1.0


def test_stray_close_tag_and_lookalike_tag_do_not_open_tables() -> None:
    ref = _html([["a", "1"], ["b", "2"]], ["k", "v"])
    assert score_table_next(ref, "</table>" + ref).score == 1.0
    assert score_table_next(ref, "<tablex><tr><td>a</td><td>1</td></tr></tablex>").score == 0.0


def test_mapped_mark_against_text_keeps_the_mapped_reading() -> None:
    assert value_sim("●", "✓ yes") == value_sim("✓", "✓ yes") > 0.0
    assert value_sim("●", "✓") == value_sim("[ ]", "☐") == 0.95  # same state, different glyph
    assert value_sim("[x]", "X") == 0.5


def test_protocol_bracketed_mark_words_read_as_the_mark_they_name() -> None:
    # the protocol transcribes a value mark as "[yes]" / "[icon: bullet]"; it scored 0.6 against "●" before
    assert value_sim("●", "[yes]") == value_sim("●", "[icon: bullet]") == value_sim("✓", "[Checked]") == 0.95
    assert value_sim("☐", "[unchecked]") == value_sim("✗", "[no]") == 0.95
    assert value_sim("●", "[no]") < 0.95 and value_sim("✓", "[unchecked]") < 0.95  # opposite states stay apart
    assert value_sim("[yes]", "[yes]") == 1.0 and value_sim("Yes", "[yes]") < 0.95  # plain words are not marks


def test_sparse_table_leaves_bare_currency_cells_out() -> None:
    # two amounts make the reference sparse, so its dash cells are scored; a "$" column is not a dash
    ref = "| Item | | 2024 |\n|---|---|---|\n| Revenue | $ | 1,234 |\n| Net | $ | 500 |"
    res = score_table_next(ref, ref.replace("| $ |", "| |"))
    assert "cell errors in row" not in _kinds(res)
    assert res.score >= 0.99
    dash = "| Item | 2024 |\n|---|---|\n| Revenue | $ - |\n| Net | 500 |"
    assert score_table_next(dash, dash.replace("| $ - |", "| |")).score < 0.9  # "$ -" is still a dash mark


def test_numbered_prose_difference_does_not_claim_unit_loss() -> None:
    ref = _html([["Item", "Details"], ["A", "1. Review stakeholder positions"]])
    out = _html([["Item", "Details"], ["A", "1. Review stake-holder positions"]])
    result = score_table_next(ref, out)
    details = [event[0] for event in result.events.items if event[-1] == "detail"]
    assert "unit lost" not in details
    assert "value partly differs" in details
    assert result.score < 1


def test_actual_unit_loss_retains_partial_credit_and_major_penalty() -> None:
    ref = _html([["Item", "Dose"], ["A", "12 mg"]])
    out = _html([["Item", "Dose"], ["A", "12"]])
    result = score_table_next(ref, out)
    assert value_sim("12 mg", "12") == 0.5
    assert result.score < 1
    assert any(event[-1] == "major" for event in result.events.items)
