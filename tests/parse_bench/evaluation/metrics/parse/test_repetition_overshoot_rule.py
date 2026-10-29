from __future__ import annotations

from parse_bench.evaluation.metrics.parse.rules_repetition import RepetitionOvershootRule


def _rule(**payload: object) -> RepetitionOvershootRule:
    return RepetitionOvershootRule({"type": "repetition_overshoot", **payload})


def test_leader_merged_across_lines_fails_only_past_one_page_line() -> None:
    """Bug caught: budgeting a leader by its text-layer length flags faithful output (symbol fonts undercount)."""
    rule = _rule(reference_text="Cuba. " + "." * 59, page_fill_runs=[59, 60, 56], page_lines=3)
    assert rule.run("Cuba. " + "." * 190)[0]
    ok, msg = rule.run("Cuba " + "." * 395)
    assert not ok and msg.startswith("Repeated more than the page holds: fill run x395 (page 60")


def test_markdown_table_syntax_is_not_counted_as_repetition() -> None:
    """Bug caught: `| --- | :-: |` separator rows and `+-----+` grids read as dash loops on every table page."""
    table = "| a | b |\n| :-: | --- |\n" + "".join(f"| {i} | x |\n" for i in range(3))
    grid = "+" + "-" * 60 + "+\n"
    assert _rule(reference_text="a b 0 1 2 x").run(table + grid * 30)[0]


def test_unit_missing_from_reference_is_skipped_on_text_pages_but_budgeted_on_scans() -> None:
    """Bug caught: chart values absent from the text layer fail text pages; a scan loop passes because nothing is
    budgeted."""
    loop = " ".join(["ಎಲ್ಲಾ"] * 30)
    assert _rule(reference_text="Figure 8").run(loop)[0]
    assert not _rule(reference_text="", skip_units_absent_from_reference=False).run(loop)[0]


def test_repeated_row_is_budgeted_by_its_rarest_word_on_the_page() -> None:
    """Bug caught: the text layer reads a repeated table row column by column, so the exact line is not on the page."""
    row = "20129IL0340067 ILS001 2024 Health Alliance"
    columns = "\n".join(["20129IL0340067"] * 30 + ["ILS001"] * 30 + ["2024 Health Alliance"] * 30)
    rule = _rule(reference_text=columns)
    assert rule.run("\n".join([row] * 30))[0]
    assert not rule.run("\n".join([row] * 120))[0]


def test_emoji_row_loop_is_caught_and_variation_selectors_do_not_count() -> None:
    """Bug caught: rows with no word were never budgeted, so an emoji grid looped for 200 rows passed; and a faithful
    `🕷️` (with U+FE0F) looked like a unit the page never had."""
    page = "\n".join(["👻" * 12] * 16 + ["🕷" * 12] * 2)
    rule = _rule(reference_text=page, skip_units_absent_from_reference=False)
    assert rule.run("\n".join(["👻" * 12] * 16 + ["🕷️" * 12] * 2))[0]
    ok, msg = rule.run("\n".join(["👻" * 12] * 200))
    assert not ok and msg.startswith("Repeated more than the page holds")


def test_underscore_blank_loop_is_not_erased_as_bold_markup() -> None:
    """Bug caught: stripping every `__` as a bold marker deleted underscore runs, so a looping fill-in blank passed."""
    rule = _rule(
        reference_text="Name: " + "_" * 30, page_fill_runs=[30], page_lines=1, skip_units_absent_from_reference=False
    )
    assert rule.run("Name: __Jane__ " + "_" * 30)[0]
    assert not rule.run("Name: " + "_" * 1200)[0]
    assert not rule.run("Name: " + "_" * 30 + "\n" + "_" * 1200)[0]  # a run on its own line, not only after a label


def test_generated_grid_written_one_table_cell_per_line_passes() -> None:
    """Bug caught: an html table puts every emoji on its own line, which the exact page never has as a line."""
    grid = "\n".join(["👻" * 12] * 16)
    rule = _rule(reference_text=grid, page_fill_runs=[], page_lines=16, skip_units_absent_from_reference=False)
    cells = "<table>\n" + "".join("<tr>\n" + "<td>👻</td>\n" * 12 + "</tr>\n" for _ in range(16)) + "</table>"
    assert rule.run(cells)[0]
    assert not rule.run(grid + ("\n" + "👻" * 12) * 150)[0]


def test_lookalike_misreads_and_other_leader_dots_are_not_loops() -> None:
    """Bug caught: O read as 0, ǀ/l read as I, $0.00 read as 50.00 and a dotted page written with ▪ all failed."""
    codes = "\n".join(["O0O0OO00O0OOO0O00OO0  mmm"] * 18)
    rule = _rule(reference_text=codes, page_fill_runs=[], page_lines=18, skip_units_absent_from_reference=False)
    assert rule.run("\n".join(["00000000000000000000 mmm"] * 18))[0]
    assert _rule(reference_text=" ".join(["$0.00"] * 30)).run(" ".join(["50.00"] * 40))[0]
    dots = "\n".join(["." * 92] * 10)
    rule = _rule(reference_text=dots, page_fill_runs=[92] * 10, page_lines=10, skip_units_absent_from_reference=False)
    assert rule.run("\n\n".join([" ".join(["▪"] * 92)] * 10))[0]
    assert not rule.run(dots + "." * 2000)[0]


def test_answer_box_lines_merged_into_one_leader_pass_up_to_the_box() -> None:
    """Bug caught: a form's 4-line answer box written as one leader failed against the one-line budget; merged, the
    box gets the same 2x budget as one leader."""
    box = "Q1 Why?\n" + "\n\n".join(["." * 130] * 4)
    rule = _rule(reference_text=box, page_fill_runs=[130] * 4, page_lines=5)
    assert rule.run("Q1 Why? " + "." * 1000)[0]
    ok, msg = rule.run("Q1 Why? " + "." * 1200)
    assert not ok and "fill run x1200" in msg


def test_misread_barcode_on_a_generated_page_is_budgeted_by_its_run() -> None:
    """Bug caught: a barcode read as ■I on every row summed to a loop although each row has the page's length."""
    page = "\n".join(["▍█▌▍▌▐▌▐▐█▍██▌▍▌▌█▍▍▍██▌▍▍▌▌▍█", "0 26115 12949 0"] * 12)
    rule = _rule(reference_text=page, page_fill_runs=[], page_lines=24, skip_units_absent_from_reference=False)
    assert rule.run("\n".join(["■I" * 15 + "■", "0 26115 12949 0"] * 12))[0]
    assert not rule.run("■I" * 200)[0]


def test_answer_lines_written_on_one_line_are_separate_leaders() -> None:
    """Bug caught: four 90-dash answer lines joined by spaces read as one 360-dash run on a page whose text layer
    splits the box."""
    rule = _rule(reference_text="Comments: " + "-" * 61 + "\nName " + "-" * 40, page_fill_runs=[61, 40], page_lines=2)
    assert rule.run("Comments: " + " ".join(["-" * 90] * 4))[0]
    assert not rule.run("Comments: " + "-" * 400)[0]
    assert not rule.run("Comments: " + " ".join(["-"] * 400))[0]  # a spaced leader is still one run


def _html_table(rows: list[list[str]]) -> str:
    return (
        "<table>\n" + "".join("<tr>\n" + "".join(f"<td>{c}</td>\n" for c in r) + "</tr>\n" for r in rows) + "</table>"
    )


def test_chart_points_transcribed_one_cell_per_line_are_data_but_a_copied_table_or_row_is_a_loop() -> None:
    """Bug caught: a scatter plot read into a 150-row table put its class label on 50 lines, past the legend's budget;
    the fix must not let a table written out six times, or one row repeated 130 times, through."""
    rule = _rule(reference_text="setosa versicolor virginica\nsepal length sepal width")
    points = [
        [f"{4 + i / 10:.1f}", f"{2 + (i % 13) / 10:.1f}", ("setosa", "versicolor", "virginica")[i // 50]]
        for i in range(150)
    ]
    assert rule.run(_html_table(points))[0]
    section = _html_table([[f"4.{i} Loss Ratio", "87.83%", *["#DIV/0!"] * 17] for i in range(18)])
    assert _rule(reference_text="Loss Ratio 85.12% #DIV/0!").run(section)[0]
    assert not _rule(reference_text="Loss Ratio 85.12% #DIV/0!").run("\n\n".join([section] * 6))[0]
    ok, msg = rule.run(_html_table(points[:5] + [["5.0", "3.4", "versicolor recourse"]] * 130))
    assert not ok and "versicolor recourse" in msg
