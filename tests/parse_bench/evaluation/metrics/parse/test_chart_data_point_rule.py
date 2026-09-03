from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.rules_chart import (
    ChartDataPointRule,
    extract_numeric_parts,
)
from parse_bench.evaluation.metrics.parse.table_parsing import (
    parse_html_tables,
)

# ---------------------------------------------------------------------------
# extract_numeric_parts
# ---------------------------------------------------------------------------


class TestExtractNumericParts:
    def test_composite_with_parenthetical_percentage(self):
        assert extract_numeric_parts("25 (13.0%)") == ["25", "13.0%"]

    def test_composite_comma_separated(self):
        assert extract_numeric_parts("25, 13.0%") == ["25", "13.0%"]

    def test_single_number(self):
        assert extract_numeric_parts("25") == ["25"]

    def test_single_percentage(self):
        assert extract_numeric_parts("13.0%") == ["13.0%"]

    def test_no_numbers(self):
        assert extract_numeric_parts("hello world") == []

    def test_negative_number(self):
        assert extract_numeric_parts("-5 (2.3%)") == ["-5", "2.3%"]

    def test_thousands_separator(self):
        assert extract_numeric_parts("1,234 (56.7%)") == ["1,234", "56.7%"]


# ---------------------------------------------------------------------------
# parse_html_tables – colspan/rowspan grid alignment
# ---------------------------------------------------------------------------

EGDI_HTML = """\
<table>
  <thead>
    <tr>
      <th rowspan="2">Year</th>
      <th colspan="2">Very high EGDI</th>
      <th colspan="2">High EGDI</th>
      <th colspan="2">Middle EGDI</th>
      <th colspan="2">Low EGDI</th>
    </tr>
    <tr>
      <th>Number</th><th>Percentage</th>
      <th>Number</th><th>Percentage</th>
      <th>Number</th><th>Percentage</th>
      <th>Number</th><th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>2014</td><td>25</td><td>13.0%</td><td>62</td><td>32.1%</td><td>74</td><td>38.3%</td><td>32</td><td>16.6%</td></tr>
    <tr><td>2016</td><td>29</td><td>15.0%</td><td>75</td><td>38.9%</td><td>57</td><td>29.5%</td><td>32</td><td>16.6%</td></tr>
    <tr><td>2018</td><td>40</td><td>20.7%</td><td>73</td><td>37.8%</td><td>65</td><td>33.7%</td><td>15</td><td>7.8%</td></tr>
    <tr><td>2022</td><td>60</td><td>31.1%</td><td>73</td><td>37.8%</td><td>53</td><td>27.5%</td><td>7</td><td>3.6%</td></tr>
    <tr><td>2024</td><td>76</td><td>39.4%</td><td>62</td><td>32.1%</td><td>44</td><td>22.8%</td><td>11</td><td>5.7%</td></tr>
  </tbody>
</table>
"""


class TestParseHtmlTablesColspan:
    """Verify that colspan/rowspan tables produce a correctly aligned grid."""

    def test_grid_dimensions(self):
        tables = parse_html_tables(EGDI_HTML)
        assert len(tables) == 1
        arr = tables[0].data
        # 2 header rows + 5 data rows = 7 rows, 9 columns (Year + 4×2)
        assert arr.shape == (7, 9)

    def test_header_row0_spans_correctly(self):
        """Row 0 should have 'Very high EGDI' in both col 1 and col 2."""
        arr = parse_html_tables(EGDI_HTML)[0].data
        assert arr[0, 0] == "Year"
        assert arr[0, 1] == "Very high EGDI"
        assert arr[0, 2] == "Very high EGDI"  # colspan=2 fills both
        assert arr[0, 3] == "High EGDI"
        assert arr[0, 4] == "High EGDI"

    def test_header_row1_sub_headers(self):
        """Row 1 col 0 should be 'Year' (from rowspan=2), rest are sub-headers."""
        arr = parse_html_tables(EGDI_HTML)[0].data
        assert arr[1, 0] == "Year"  # rowspan=2 from row 0
        assert arr[1, 1] == "Number"
        assert arr[1, 2] == "Percentage"
        assert arr[1, 3] == "Number"

    def test_data_row_alignment(self):
        """Data rows should align with the grid columns."""
        arr = parse_html_tables(EGDI_HTML)[0].data
        assert arr[2, 0] == "2014"
        assert arr[2, 1] == "25"
        assert arr[2, 2] == "13.0%"
        assert arr[2, 3] == "62"

    def test_col_headers_include_parent_and_sub(self):
        """col_headers for column 1 should include both 'Very high EGDI' and 'Number'."""
        td = parse_html_tables(EGDI_HTML)[0]
        headers_col1 = td.col_headers.get(1, [])
        header_texts = [text for _, text in headers_col1]
        assert "Very high EGDI" in header_texts
        assert "Number" in header_texts


# ---------------------------------------------------------------------------
# ChartDataPointRule – composite value + multi-row header integration
# ---------------------------------------------------------------------------


class TestChartDataPointRuleComposite:
    """End-to-end tests for composite value matching against colspan tables."""

    def test_composite_value_found(self):
        rule = ChartDataPointRule(
            {
                "type": "chart_data_point",
                "value": "25 (13.0%)",
                "labels": ["Very high EGDI", "2014"],
                "normalize_numbers": True,
            }
        )
        passed, msg, score = rule.run(EGDI_HTML)
        assert passed, f"Expected pass but got: {msg}"
        assert score == 1.0

    def test_composite_value_middle_egdi(self):
        rule = ChartDataPointRule(
            {
                "type": "chart_data_point",
                "value": "74 (38.3%)",
                "labels": ["Middle EGDI", "2014"],
                "normalize_numbers": True,
            }
        )
        passed, msg, score = rule.run(EGDI_HTML)
        assert passed, f"Expected pass but got: {msg}"

    def test_composite_value_high_egdi_2018(self):
        rule = ChartDataPointRule(
            {
                "type": "chart_data_point",
                "value": "73 (37.8%)",
                "labels": ["High EGDI", "2018"],
                "normalize_numbers": True,
            }
        )
        passed, msg, score = rule.run(EGDI_HTML)
        assert passed, f"Expected pass but got: {msg}"

    def test_composite_value_very_high_2024(self):
        rule = ChartDataPointRule(
            {
                "type": "chart_data_point",
                "value": "76 (39.4%)",
                "labels": ["Very high EGDI", "2024"],
                "normalize_numbers": True,
            }
        )
        passed, msg, score = rule.run(EGDI_HTML)
        assert passed, f"Expected pass but got: {msg}"

    def test_composite_value_low_egdi_2022(self):
        rule = ChartDataPointRule(
            {
                "type": "chart_data_point",
                "value": "7 (3.6%)",
                "labels": ["Low EGDI", "2022"],
                "normalize_numbers": True,
            }
        )
        passed, msg, score = rule.run(EGDI_HTML)
        assert passed, f"Expected pass but got: {msg}"

    def test_wrong_label_fails(self):
        rule = ChartDataPointRule(
            {
                "type": "chart_data_point",
                "value": "25 (13.0%)",
                "labels": ["Low EGDI", "2014"],
                "normalize_numbers": True,
            }
        )
        passed, msg, score = rule.run(EGDI_HTML)
        assert not passed

    def test_simple_value_still_works(self):
        """Non-composite values should still match directly."""
        simple_html = """\
        <table>
          <thead><tr><th>Category</th><th>Value</th></tr></thead>
          <tbody><tr><td>Alpha</td><td>42</td></tr></tbody>
        </table>
        """
        rule = ChartDataPointRule(
            {
                "type": "chart_data_point",
                "value": "42",
                "labels": ["Alpha"],
                "normalize_numbers": True,
            }
        )
        passed, msg, score = rule.run(simple_html)
        assert passed, f"Expected pass but got: {msg}"


# ---------------------------------------------------------------------------
# ChartDataPointRule – candidate-local label association
# ---------------------------------------------------------------------------


def _chart_rule(value: str, labels: list[str], **overrides: object) -> ChartDataPointRule:
    return ChartDataPointRule({"type": "chart_data_point", "value": value, "labels": labels, **overrides})


TDR_SHORT_YEAR_HTML = """\
<table>
  <caption>Profit shares, 2003–2023</caption>
  <thead><tr><th>series</th><th>time</th><th>Percentage of GDP</th></tr></thead>
  <tbody>
    <tr><td>Developed countries</td><td>2008</td><td>35.4</td></tr>
    <tr><td>Developing countries</td><td>2008</td><td>43.9</td></tr>
    <tr><td>Developed countries</td><td>2013</td><td>36.2</td></tr>
    <tr><td>Developing countries</td><td>2013</td><td>42.8</td></tr>
    <tr><td>Developed countries</td><td>2018</td><td>36.6</td></tr>
    <tr><td>Developing countries</td><td>2018</td><td>42.8</td></tr>
    <tr><td>Developed countries</td><td>2023</td><td>37.5</td></tr>
    <tr><td>Developing countries</td><td>2009</td><td>42</td></tr>
  </tbody>
</table>
"""


class TestChartDataPointRuleCandidateScope:
    @pytest.mark.parametrize(
        ("rule_id", "value", "labels"),
        [
            ("6e8111febcee9313", "35.4", ["08", "Developed countries"]),
            ("47ae484dae89bd02", "43.9", ["08", "Developing countries"]),
            ("3b541111e18251ba", "36.2", ["13", "Developed countries"]),
            ("7edf15f4a202c4bc", "42.8", ["13", "Developing countries"]),
            ("cb54981ef59399df", "36.6", ["18", "Developed countries"]),
            ("c2fcad4000b813ca", "42.8", ["18", "Developing countries"]),
            ("be7daa129968d42a", "37.5", ["23", "Developed countries"]),
            ("1ba4437111b8bcf6", "42", ["09", "Developing countries"]),
        ],
        ids=[
            "6e8111febcee9313",
            "47ae484dae89bd02",
            "3b541111e18251ba",
            "7edf15f4a202c4bc",
            "cb54981ef59399df",
            "c2fcad4000b813ca",
            "be7daa129968d42a",
            "1ba4437111b8bcf6",
        ],
    )
    def test_two_digit_year_suffix_matches_candidate_local_four_digit_year(
        self,
        rule_id: str,
        value: str,
        labels: list[str],
    ) -> None:
        passed, message, score = _chart_rule(value, labels).run(TDR_SHORT_YEAR_HTML)

        assert passed, f"{rule_id}: {message}"
        assert score == 1.0
        assert "candidate-local scope" in message

    @pytest.mark.parametrize("candidate_year", ["1908", "20090", "2009.0"])
    def test_two_digit_year_suffix_rejects_non_equivalent_numeric_tokens(self, candidate_year: str) -> None:
        table = f"""\
<table>
  <thead><tr><th>Series</th><th>Year</th><th>Value</th></tr></thead>
  <tbody><tr><td>Target</td><td>{candidate_year}</td><td>42</td></tr></tbody>
</table>
"""

        passed, message, score = _chart_rule("42", ["Target", "09"]).run(table)

        assert not passed, message
        assert score == 0.0

    def test_two_digit_year_suffix_does_not_borrow_caption_or_unrelated_body_row(self) -> None:
        table = """\
<p>Unrelated discussion of results from 2009.</p>
<table>
  <caption>Archive for 2009</caption>
  <thead><tr><th>Series</th><th>Year</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Other</td><td>2009</td><td>17</td></tr>
    <tr><td>Target</td><td>2010</td><td>42</td></tr>
  </tbody>
</table>
"""

        passed, message, score = _chart_rule("42", ["Target", "09"]).run(table)

        assert not passed, message
        assert score == 0.0

    def test_eu27_collision_does_not_borrow_another_value_row(self) -> None:
        table = """\
| Country | Year | Value |
| --- | --- | --- |
| EU27 | 2014 | 24.425 |
| Greece | 2022 | 17 |
| Germany | 2022 | 17 |
"""
        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert not passed
        assert score == 0.0
        assert message.startswith("Value found but labels not associated:")
        assert "found with all labels" not in message

    def test_full_length_wrong_year_collision_fails(self) -> None:
        table = """\
| Country | Year | Value |
| --- | --- | --- |
| Hungary | 2014 | 29 |
| Poland | 2022 | 11 |
"""
        passed, _, score = _chart_rule("29", ["Hungary", "2022"]).run(table)

        assert not passed
        assert score == 0.0

    def test_exact_target_reports_the_target_candidate(self) -> None:
        table = """\
| Country | Year | Value |
| --- | --- | --- |
| EU27 | 2022 | 17 |
| Greece | 2022 | 17 |
"""
        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message
        assert score == 1.0
        assert "candidate-local scope at (1, 2)" in message

    def test_column_oriented_table_uses_candidate_row_and_column_header(self) -> None:
        table = """\
| Year | EU27 | Greece |
| --- | --- | --- |
| 2022 | 17 | 19 |
"""
        passed, message, _ = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message

    def test_transposed_series_by_year_table_passes(self) -> None:
        table = """\
| Series | 2021 | 2022 |
| --- | --- | --- |
| EU27 | 16 | 17 |
"""
        passed, message, _ = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message

    def test_horizontal_records_table_passes(self) -> None:
        table = """\
| Country | 2021 | 2022 |
| --- | --- | --- |
| EU27 | 16 | 17 |
| Greece | 15 | 18 |
"""
        passed, message, _ = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message

    def test_rowspan_row_header_and_column_header_pass(self) -> None:
        table = """\
<table>
  <thead><tr><th>Country</th><th>Year</th><th>Value</th></tr></thead>
  <tbody>
    <tr><th rowspan="2">EU27</th><td>2022</td><td>17</td></tr>
    <tr><td>2023</td><td>18</td></tr>
  </tbody>
</table>
"""
        passed, message, _ = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message

    def test_mixed_cell_thead_keeps_every_authored_header_level(self) -> None:
        table = """\
<table>
  <thead><tr><td>2022</td></tr><tr><th>Q1</th></tr></thead>
  <tbody><tr><td>17</td></tr></tbody>
</table>
"""

        passed, message, score = _chart_rule("17", ["2022", "Q1"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_top_level_th_header_row_without_thead_supplies_column_label(self) -> None:
        table = "<table><tr><th>Country</th><th>2022</th></tr><tr><td>EU27</td><td>17</td></tr></table>"

        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_leading_td_title_does_not_hide_top_level_header(self) -> None:
        table = """\
<table>
  <tr><td colspan="2">Population</td></tr>
  <tr><th>Country</th><th>2022</th></tr>
  <tr><td>EU27</td><td>17</td></tr>
</table>
"""

        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_leading_empty_row_does_not_hide_top_level_header(self) -> None:
        table = (
            "<table><tr><td></td><td></td></tr><tr><th>Country</th><th>2022</th></tr>"
            "<tr><td>EU27</td><td>17</td></tr></table>"
        )

        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_identical_tbody_row_does_not_shadow_top_level_header_provenance(self) -> None:
        table = """\
<table>
  <tr><th>2022</th></tr>
  <tbody><tr><th>2022</th></tr><tr><td>17</td></tr></tbody>
</table>
"""

        passed, message, score = _chart_rule("17", ["2022"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_lxml_recovered_top_level_th_header_row_supplies_column_label(self) -> None:
        table = "<table><tr><th>Country<th>2022</tr><tr><td>EU27<td>17</tr></table>"

        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_top_level_mixed_header_accepts_empty_corner_td(self) -> None:
        table = "<table><tr><td></td><th>2022</th></tr><tr><th>EU27</th><td>17</td></tr></table>"

        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_top_level_mixed_header_rejects_nonempty_corner_td(self) -> None:
        table = "<table><tr><td>Country</td><th>2022</th></tr><tr><td>EU27</td><td>17</td></tr></table>"

        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert not passed
        assert score == 0.0
        assert message.startswith("Value found but labels not associated:")

    def test_tbody_th_does_not_become_a_column_label_for_later_value(self) -> None:
        table = """\
<table><tbody>
  <tr><th>Other</th><td>2022</td></tr>
  <tr><td>Target</td><td>17</td></tr>
</tbody></table>
"""
        passed, message, score = _chart_rule("17", ["Target", "2022"]).run(table)

        assert not passed
        assert score == 0.0
        assert message.startswith("Value found but labels not associated:")

    def test_tbody_th_only_row_does_not_become_an_implicit_header_block(self) -> None:
        table = "<table><tbody><tr><th>Country</th><th>2022</th></tr><tr><td>EU27</td><td>17</td></tr></tbody></table>"

        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert not passed
        assert score == 0.0
        assert message.startswith("Value found but labels not associated:")

    def test_tfoot_th_does_not_become_an_implicit_header_block(self) -> None:
        table = "<table><tfoot><tr><th>2022</th></tr></tfoot><tbody><tr><td>17</td></tr></tbody></table>"

        passed, message, score = _chart_rule("17", ["2022"]).run(table)

        assert not passed
        assert score == 0.0
        assert message.startswith("Value found but labels not associated:")

    def test_scope_row_does_not_become_an_implicit_column_header(self) -> None:
        table = '<table><tr><th scope="row">2022</th><td></td></tr><tr><td>Target</td><td>17</td></tr></table>'

        passed, _, score = _chart_rule("17", ["Target", "2022"]).run(table)

        assert not passed
        assert score == 0.0

    def test_scope_row_in_thead_does_not_become_a_column_header(self) -> None:
        table = (
            '<table><thead><tr><th scope="row">Wrong</th><td>Series</td></tr></thead>'
            "<tbody><tr><td>17</td><td>Target</td></tr></tbody></table>"
        )

        passed, _, score = _chart_rule("17", ["Wrong", "Target"]).run(table)

        assert not passed
        assert score == 0.0

    def test_scope_row_in_thead_keeps_adjacent_td_column_header(self) -> None:
        table = (
            '<table><thead><tr><th scope="row">Row label</th><td>2022</td></tr></thead>'
            "<tbody><tr><td>Target</td><td>17</td></tr></tbody></table>"
        )

        passed, message, score = _chart_rule("17", ["Target", "2022"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_value_before_a_later_thead_fails_without_crashing(self) -> None:
        table = (
            "<table><tbody><tr><td>Target</td><td>17</td></tr></tbody>"
            "<thead><tr><th>Series</th><th>2022</th></tr></thead></table>"
        )

        passed, _, score = _chart_rule("17", ["Target", "2022"]).run(table)

        assert not passed
        assert score == 0.0

    def test_scope_col_in_tbody_is_an_authored_column_header(self) -> None:
        table = (
            '<table><tbody><tr><th scope="col">Series</th><th scope="col">2022</th></tr>'
            "<tr><td>Target</td><td>17</td></tr></tbody></table>"
        )

        passed, message, score = _chart_rule("17", ["Target", "2022"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_scope_colgroup_expands_to_the_authored_column_group(self) -> None:
        table = """\
<table>
  <colgroup span="2"></colgroup>
  <thead><tr><th scope="colgroup">Group A</th><th>Year</th></tr></thead>
  <tbody><tr><td>18</td><td>17</td></tr></tbody>
</table>
"""

        passed, message, score = _chart_rule("17", ["Group A", "Year"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_scope_rowgroup_expands_to_later_rows_in_the_authored_group(self) -> None:
        table = """\
<table><tbody>
  <tr><th scope="rowgroup">Group A</th><td>2022</td><td>17</td></tr>
  <tr><td>2023</td><td>18</td></tr>
</tbody></table>
"""

        passed, message, score = _chart_rule("18", ["Group A", "2023"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_scoped_tfoot_header_does_not_leak_to_body_candidate(self) -> None:
        table = '<table><tfoot><tr><th scope="col">2022</th></tr></tfoot><tbody><tr><td>17</td></tr></tbody></table>'

        passed, message, score = _chart_rule("17", ["2022"]).run(table)

        assert not passed
        assert score == 0.0
        assert message.startswith("Value found but labels not associated:")

    def test_all_td_table_preserves_a_unique_vertical_header_association(self) -> None:
        table = "<table><tr><td>Year</td><td>2022</td></tr><tr><td>Target</td><td>17</td></tr></table>"

        passed, message, score = _chart_rule("17", ["Target", "2022"]).run(table)

        assert passed, message
        assert score == 1.0
        assert "unique all-td row/column scope" in message

    def test_all_td_table_rejects_an_ambiguous_vertical_header_association(self) -> None:
        table = (
            "<table><tr><td>Year</td><td>2022</td><td>2022</td></tr>"
            "<tr><td>Target</td><td>17</td><td>17</td></tr></table>"
        )

        passed, message, score = _chart_rule("17", ["Target", "2022"]).run(table)

        assert not passed
        assert score == 0.0
        assert message.startswith("Value found but labels not associated:")

    def test_later_thead_replaces_stale_header_block(self) -> None:
        table = """\
<table>
  <thead><tr><th>Series</th><th>2014</th></tr></thead>
  <tbody><tr><td>Old</td><td>17</td></tr></tbody>
  <thead><tr><th>Series</th><th>2022</th></tr></thead>
  <tbody><tr><td>Target</td><td>18</td></tr></tbody>
</table>
"""

        current_passed, current_message, _ = _chart_rule("18", ["Target", "2022"]).run(table)
        stale_passed, _, stale_score = _chart_rule("18", ["Target", "2014"]).run(table)

        assert current_passed, current_message
        assert not stale_passed
        assert stale_score == 0.0

    def test_tfoot_rowspan_cannot_leak_to_a_body_candidate(self) -> None:
        table = (
            '<table><tfoot><tr><th rowspan="2">2022</th><td>Footer</td></tr></tfoot>'
            "<tbody><tr><td>Target</td><td>17</td></tr></tbody></table>"
        )

        passed, _, score = _chart_rule("17", ["Target", "2022"]).run(table)

        assert not passed
        assert score == 0.0

    def test_tbody_th_in_value_column_does_not_leak_to_later_value(self) -> None:
        table = """\
<table><tbody>
  <tr><td>EU27</td><td>2014</td><th>EU27</th></tr>
  <tr><td>Greece</td><td>2022</td><td>17</td></tr>
</tbody></table>
"""
        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert not passed
        assert score == 0.0
        assert message.startswith("Value found but labels not associated:")

    def test_context_title_augments_a_coherent_candidate_scope(self) -> None:
        table = """\
## Chart identity: EU27 outcomes

| Country | Value |
| --- | --- |
| Greece | 17 |
"""
        passed, message, _ = _chart_rule("17", ["Greece", "EU27 outcomes"]).run(table)

        assert passed, message
        assert "title labels ['eu27 outcomes'] in context" in message

    def test_generic_formatted_context_cannot_repair_data_labels_from_different_rows(self) -> None:
        table = """\
**EU27 in 2022**

| Country | Year | Value |
| --- | --- | --- |
| EU27 | 2014 | 17 |
| Greece | 2022 | 19 |
"""
        passed, _, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert not passed
        assert score == 0.0

    def test_heading_identity_remains_valid_when_label_also_occurs_in_table(self) -> None:
        table = """\
## Germany overview

| Country | Year | Value |
| --- | --- | --- |
| France | 2022 | 17 |
| Germany | 2021 | 18 |
"""

        passed, message, score = _chart_rule("17", ["France", "2022", "Germany"]).run(table)

        assert passed, message
        assert score == 1.0
        assert "title labels ['germany'] in context" in message

    def test_caption_identity_remains_valid_when_label_also_occurs_in_table(self) -> None:
        table = """\
<table>
  <caption>Earthquakes overview</caption>
  <thead><tr><th>Series</th><th>Value</th></tr></thead>
  <tbody><tr><td>Floods</td><td>17</td></tr><tr><td>Earthquakes</td><td>18</td></tr></tbody>
</table>
"""

        passed, message, score = _chart_rule("17", ["Floods", "Earthquakes"]).run(table)

        assert passed, message
        assert score == 1.0
        assert "title labels ['earthquakes'] in context" in message

    def test_punctuation_and_short_exact_labels_remain_valid(self) -> None:
        table = """\
| Series | Value |
| --- | --- |
| U.S. | 7 |
| EU-27 | 17 |
"""
        us_passed, us_message, _ = _chart_rule("7", ["US"]).run(table)
        eu_passed, eu_message, _ = _chart_rule("17", ["EU 27"]).run(table)
        combined_us_table = """\
| Series | Value |
| --- | --- |
| U.S. total | 8 |
"""
        combined_us_passed, combined_us_message, _ = _chart_rule("8", ["US"]).run(combined_us_table)
        hyphenated_q1_table = """\
| Quarter | Value |
| --- | --- |
| Q1-2024 | 9 |
"""
        slash_delimited_q1_table = hyphenated_q1_table.replace("Q1-2024", "Q1/2024").replace("9", "10")
        hyphenated_q1_passed, hyphenated_q1_message, _ = _chart_rule("9", ["Q1"]).run(hyphenated_q1_table)
        slash_delimited_q1_passed, slash_delimited_q1_message, _ = _chart_rule("10", ["Q1"]).run(
            slash_delimited_q1_table
        )

        assert us_passed, us_message
        assert eu_passed, eu_message
        assert combined_us_passed, combined_us_message
        assert hyphenated_q1_passed, hyphenated_q1_message
        assert slash_delimited_q1_passed, slash_delimited_q1_message
        assert not _chart_rule("7", ["EU27"])._label_matches("EU27", "7")
        assert not _chart_rule("27", ["EU27"])._label_matches("EU27", "27")

    def test_short_label_as_a_distinct_combined_cell_token_remains_valid(self) -> None:
        table = """\
| Quarter | Value |
| --- | --- |
| Q1 2024 | 4.2 |
"""

        passed, message, score = _chart_rule("4.2", ["Q1"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_short_delimited_label_matches_a_longer_candidate_local_cell(self) -> None:
        table = """\
| Series | Value |
| --- | --- |
| N/A total | 4.2 |
"""

        passed, message, score = _chart_rule("4.2", ["N/A"]).run(table)

        assert passed, message
        assert score == 1.0

    def test_semantically_different_label_fails(self) -> None:
        table = """\
| Degree | Value |
| --- | --- |
| Bachelor's degree | 42 |
"""
        passed, _, score = _chart_rule("42", ["Master's degree"]).run(table)

        assert not passed
        assert score == 0.0

    def test_three_part_composite_value_uses_the_first_part_as_anchor(self) -> None:
        table = """\
| Country | Year | Count | Share | Rate |
| --- | --- | --- | --- | --- |
| EU27 | 2022 | 25 | 13.0% | 40 |
"""
        passed, message, _ = _chart_rule("25 (13.0%) 40", ["EU27", "2022"], normalize_numbers=True).run(table)

        assert passed, message
        assert "(1, 2)" in message

    def test_numeric_tolerance_keeps_candidate_association(self) -> None:
        within = """\
| Country | Year | Value |
| --- | --- | --- |
| EU27 | 2022 | 101 |
"""
        outside = within.replace("101", "103")
        rule = _chart_rule("100", ["EU27", "2022"], normalize_numbers=True, relative_tolerance=0.02)

        assert rule.run(within)[0]
        assert not rule.run(outside)[0]

    def test_later_correct_duplicate_value_is_selected(self) -> None:
        table = """\
| Country | Year | Value |
| --- | --- | --- |
| EU27 | 2014 | 17 |
| Greece | 2022 | 17 |
| EU27 | 2022 | 17 |
"""
        passed, message, score = _chart_rule("17", ["EU27", "2022"]).run(table)

        assert passed, message
        assert score == 1.0
        assert "candidate-local scope at (3, 2)" in message
