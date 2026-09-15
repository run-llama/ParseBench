from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.evaluation.metrics.parse.rules_table import TableMarkerCellsRule


def _run(markdown: str, **overrides: object) -> tuple[bool, str]:
    rule = create_test_rule({"type": "table_marker_cells", **overrides})
    assert isinstance(rule, TableMarkerCellsRule)
    return rule.run(markdown)


def test_repeated_ocr_glyphs_pass_across_rows_and_columns() -> None:
    markdown = """
<table>
  <tr><th>Skill</th><th>A</th><th>B</th></tr>
  <tr><td>Finance</td><td>Ө</td><td></td></tr>
  <tr><td>Operations</td><td></td><td>Ө</td></tr>
</table>
"""

    assert _run(markdown, min_distinct_columns=2) == (True, "")


def test_known_markers_pass_in_one_status_column() -> None:
    markdown = """
| Goal | Status |
| --- | --- |
| First | ✓ |
| Second | ✓ |
"""

    assert _run(markdown) == (True, "")


def test_html_images_inside_table_cells_pass() -> None:
    markdown = """
<table>
  <tr><th>Goal</th><th>Status</th></tr>
  <tr><td>First</td><td><img src="first.png" alt="green"></td></tr>
  <tr><td>Second</td><td><p><img src="second.png"></p></td></tr>
</table>
"""

    assert _run(markdown) == (True, "")


def test_html_images_with_labels_still_count_as_icon_cells() -> None:
    markdown = """
<table>
  <tr><th>Crop</th><th>Product</th></tr>
  <tr><td><img src="corn.png"> Corn</td><td>A</td></tr>
  <tr><td><img src="soy.png"> Soybeans</td><td>B</td></tr>
</table>
"""

    assert _run(markdown) == (True, "")


def test_semantic_marker_labels_pass() -> None:
    markdown = """
| Skill | Alice | Bob |
| --- | --- | --- |
| Finance | Orange Square | Grey Square |
| Operations | Grey Square | Orange Square |
"""

    assert _run(markdown, min_distinct_columns=2) == (True, "")


def test_bracketed_icon_labels_pass_with_surrounding_text() -> None:
    markdown = """
| Crop | Product |
| --- | --- |
| [icon: Corn] Corn | A |
| [icon: Soybeans] Soybeans | B |
"""

    assert _run(markdown) == (True, "")


def test_leading_symbol_with_value_passes() -> None:
    markdown = """
| Company | Change |
| --- | --- |
| First | ▲ 512.40 |
| Second | ▲ 125.00 |
"""

    assert _run(markdown) == (True, "")


def test_word_alias_embedded_in_prose_does_not_pass() -> None:
    markdown = """
| Goal | Comment |
| --- | --- |
| First | selected for review |
| Second | selected by committee |
"""

    assert _run(markdown)[0] is False


def test_missing_value_dashes_are_not_markers_by_default() -> None:
    markdown = """
| Metric | 2024 | 2025 |
| --- | --- | --- |
| Revenue | - | 10 |
| Profit | - | 2 |
"""

    assert _run(markdown)[0] is False


def test_markers_in_one_column_fail_when_two_are_required() -> None:
    markdown = """
| Goal | Status |
| --- | --- |
| First | ✓ |
| Second | ✓ |
"""

    passed, message = _run(markdown, min_distinct_columns=2)
    assert passed is False
    assert "only 1 table column(s)" in message

    rule = create_test_rule({"type": "table_marker_cells", "min_distinct_columns": 2})
    rule.run(markdown)
    assert rule.result_details["tables_inspected"] == 1
    assert rule.result_details["best_table"]["marker_cells"] == 2
    assert rule.result_details["best_table"]["recognized_tokens"] == {"✓": 2}
    assert "at least 2 are required" in rule.result_details["diagnosis"]


def test_repeated_ascii_abbreviation_is_not_a_marker() -> None:
    markdown = """
| Metric | 2024 |
| --- | --- |
| Revenue | M |
| Profit | M |
"""

    assert _run(markdown)[0] is False


def test_custom_marker_alias_passes() -> None:
    markdown = """
| Goal | Status |
| --- | --- |
| First | done-icon |
| Second | done-icon |
"""

    assert _run(markdown, marker_aliases=["done-icon"])[0] is True


def test_custom_marker_alias_lists_pass() -> None:
    markdown = """
| Impact | Low | High |
| --- | --- | --- |
| Very high | | H, D, E |
| High | | B, C |
| Medium | F | I |
"""

    assert _run(markdown, marker_aliases=list("abcdefghi"))[0] is True


def test_bracket_wrapped_aliases_pass() -> None:
    markdown = """
| Skill | Alice | Bob |
| --- | --- | --- |
| Finance | [yes] | |
| Operations | | [yes] |
"""

    assert _run(markdown, min_distinct_columns=2)[0] is True


def test_marker_text_outside_a_table_fails() -> None:
    rule = create_test_rule({"type": "table_marker_cells"})
    assert rule.run("First ✓\n\nSecond ✓") == (
        False,
        "No tables found; icon-valued cells could not be evaluated",
    )
    assert rule.result_details == {
        "requirement": ">=2 marker cells across >=2 rows and >=1 columns",
        "tables_inspected": 0,
        "diagnosis": "The parser emitted no recognizable Markdown or HTML table.",
    }
