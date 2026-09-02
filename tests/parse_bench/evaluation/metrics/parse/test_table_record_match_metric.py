"""Tests for table_record_match metric."""

import pytest

from parse_bench.evaluation.metrics.parse.table_extraction import (
    extract_html_tables,
)
from parse_bench.evaluation.metrics.parse.table_parsing import (
    parse_html_tables,
)
from parse_bench.evaluation.metrics.parse.table_record_match_metric import (
    HeaderInfo,
    TableRecordMatchMetric,
    _align_columns_header_core,
    _demote_pred_headers,
    _is_all_empty_record,
    _normalize_sub_sup_for_table,
    _normalize_trm_cell_text,
    _resolve_header_row_values,
    _try_recover_pred_header,
    _try_section_header_promotion,
    align_columns,
    build_record_details,
    build_records,
    cell_score,
    extract_header_info,
    match_records,
    normalize_table,
    table_to_records,
)
from parse_bench.evaluation.metrics.parse.table_splitting import (
    _detect_period_candidates,
    enumerate_split_options,
)
from parse_bench.evaluation.metrics.parse.table_splitting import (
    build_sub_table as _build_sub_table,
)
from parse_bench.evaluation.metrics.parse.utils import _normalize_table_boolean_marker, normalize_text


def _norm(s: str) -> str:
    """Pre-normalize a string the same way normalize_table does per cell."""
    return _normalize_trm_cell_text(normalize_text(_normalize_table_boolean_marker(s)))


def _html(table_body: str) -> str:
    """Wrap table rows in a full HTML table."""
    return f"<table>{table_body}</table>"


def _simple_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a simple HTML table from headers and row data."""
    hdr = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return _html(f"<thead>{hdr}</thead><tbody>{body}</tbody>")


@pytest.fixture
def metric() -> TableRecordMatchMetric:
    return TableRecordMatchMetric()


# ---- 1. Identical tables → 1.0 ----
def test_identical_tables(metric: TableRecordMatchMetric) -> None:
    table = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
    result = metric.compute(expected=table, actual=table)
    assert len(result) == 2
    assert result[0].value == pytest.approx(1.0)
    # Perfect match rate
    assert result[1].metric_name == "table_record_match_perfect"
    assert result[1].value == pytest.approx(1.0)


# ---- 2. Row-shuffled → 1.0 ----
def test_row_shuffled(metric: TableRecordMatchMetric) -> None:
    gt = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"], ["Carol", "35"]])
    pred = _simple_table(["Name", "Age"], [["Carol", "35"], ["Alice", "30"], ["Bob", "25"]])
    result = metric.compute(expected=gt, actual=pred)
    assert result[0].value == pytest.approx(1.0)


# ---- 3. Column-shuffled → 1.0 ----
def test_column_shuffled(metric: TableRecordMatchMetric) -> None:
    gt = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
    pred = _simple_table(["Age", "Name"], [["30", "Alice"], ["25", "Bob"]])
    result = metric.compute(expected=gt, actual=pred)
    assert result[0].value == pytest.approx(1.0)


# ---- 4. Row + column shuffled → 1.0 ----
def test_row_and_column_shuffled(metric: TableRecordMatchMetric) -> None:
    gt = _simple_table(
        ["Name", "Age", "City"],
        [
            ["Alice", "30", "NYC"],
            ["Bob", "25", "LA"],
        ],
    )
    pred = _simple_table(
        ["City", "Name", "Age"],
        [
            ["LA", "Bob", "25"],
            ["NYC", "Alice", "30"],
        ],
    )
    result = metric.compute(expected=gt, actual=pred)
    assert result[0].value == pytest.approx(1.0)


# ---- 5. Missing row → proportional penalty ----
def test_missing_row(metric: TableRecordMatchMetric) -> None:
    gt = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"], ["Carol", "35"]])
    pred = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
    result = metric.compute(expected=gt, actual=pred)
    # 2 of 3 rows match → ~0.67
    assert 0.55 <= result[0].value <= 0.75


# ---- 6. Extra row → proportional penalty ----
def test_extra_row(metric: TableRecordMatchMetric) -> None:
    gt = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
    pred = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"], ["Carol", "35"]])
    result = metric.compute(expected=gt, actual=pred)
    # 2 matched / max(2, 3) = 0.67
    assert 0.55 <= result[0].value <= 0.75


# ---- 7. Wrong cell value → strict penalty ----
def test_wrong_cell_strict(metric: TableRecordMatchMetric) -> None:
    gt = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
    pred = _simple_table(["Name", "Age"], [["Alice", "31"], ["Bob", "25"]])
    result = metric.compute(expected=gt, actual=pred)
    # One cell wrong out of 4 total → 3/4 cells correct in matched row → 0.75 avg per row
    # Row 1: 1 of 2 cells match → 0.5. Row 2: 2 of 2 → 1.0. Avg = 0.75
    assert result[0].value == pytest.approx(0.75)


# ---- 8. Headerless tables → content-based alignment ----
def test_headerless_tables(metric: TableRecordMatchMetric) -> None:
    gt = _html("<tr><td>Alice</td><td>30</td></tr><tr><td>Bob</td><td>25</td></tr>")
    pred = _html("<tr><td>Alice</td><td>30</td></tr><tr><td>Bob</td><td>25</td></tr>")
    result = metric.compute(expected=gt, actual=pred)
    assert result[0].value == pytest.approx(1.0)


# ---- 9. Dual-axis headers (row + column) → matched via Hungarian ----
def test_dual_axis_headers(metric: TableRecordMatchMetric) -> None:
    gt = _html(
        "<thead><tr><th></th><th>Q1</th><th>Q2</th></tr></thead>"
        "<tbody>"
        "<tr><th>Revenue</th><td>100</td><td>200</td></tr>"
        "<tr><th>Cost</th><td>50</td><td>80</td></tr>"
        "</tbody>"
    )
    # Rows swapped
    pred = _html(
        "<thead><tr><th></th><th>Q1</th><th>Q2</th></tr></thead>"
        "<tbody>"
        "<tr><th>Cost</th><td>50</td><td>80</td></tr>"
        "<tr><th>Revenue</th><td>100</td><td>200</td></tr>"
        "</tbody>"
    )
    result = metric.compute(expected=gt, actual=pred)
    assert result[0].value == pytest.approx(1.0)


# ---- 10. Multi-level headers → keys flatten with space separator ----
def test_multi_level_headers(metric: TableRecordMatchMetric) -> None:
    gt = _html(
        "<thead>"
        '<tr><th colspan="2">Sales</th></tr>'
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead>"
        "<tbody>"
        "<tr><td>100</td><td>200</td></tr>"
        "</tbody>"
    )
    pred = _html(
        "<thead>"
        '<tr><th colspan="2">Sales</th></tr>'
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead>"
        "<tbody>"
        "<tr><td>100</td><td>200</td></tr>"
        "</tbody>"
    )
    result = metric.compute(expected=gt, actual=pred)
    assert result[0].value == pytest.approx(1.0)

    # Verify that the spanning "Sales" title row is excluded from keys,
    # leaving just the column-discriminating "Q1", "Q2"
    tables = parse_html_tables(gt)
    keys, records, _ = table_to_records(tables[0])
    assert keys == ["Q1", "Q2"], f"Expected title row excluded, got: {keys}"


# ---- 11. Section headers → now regular data rows (no special treatment) ----
def test_section_headers(metric: TableRecordMatchMetric) -> None:
    gt = _html(
        "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
        "<tbody>"
        "<tr><td>A</td><td>10</td></tr>"
        '<tr><th colspan="2">Section B</th></tr>'
        "<tr><td>B1</td><td>20</td></tr>"
        "</tbody>"
    )
    pred = _html(
        "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
        "<tbody>"
        "<tr><td>A</td><td>10</td></tr>"
        '<tr><th colspan="2">Section B</th></tr>'
        "<tr><td>B1</td><td>20</td></tr>"
        "</tbody>"
    )
    result = metric.compute(expected=gt, actual=pred)
    assert result[0].value == pytest.approx(1.0)

    # Section rows are now regular records (no _section field)
    tables = parse_html_tables(gt)
    keys, records, _ = table_to_records(tables[0])
    assert len(records) == 3  # A, Section B, B1
    assert all("_section" not in r for r in records)


# ---- 12. Empty table → 0.0 ----
def test_empty_table(metric: TableRecordMatchMetric) -> None:
    gt = _simple_table(["A", "B"], [["x", "y"]])
    pred = _html("<thead><tr><th>A</th><th>B</th></tr></thead><tbody></tbody>")
    result = metric.compute(expected=gt, actual=pred)
    assert result[0].value == pytest.approx(0.0)


# ---- 13. Tables with numeric values → strict matching ----
def test_numeric_strict(metric: TableRecordMatchMetric) -> None:
    gt = _simple_table(["Metric", "Value"], [["Revenue", "1000000"]])
    pred = _simple_table(["Metric", "Value"], [["Revenue", "1000001"]])
    result = metric.compute(expected=gt, actual=pred)
    # "1000000" != "1000001" → 1 of 2 cells wrong → score = 0.5
    assert result[0].value == pytest.approx(0.5)


# ---- 14. Multiple tables per document → scores averaged ----
def test_multiple_tables_averaged(metric: TableRecordMatchMetric) -> None:
    t1_gt = _simple_table(["A"], [["1"]])
    t2_gt = _simple_table(["B"], [["2"]])
    t1_pred = _simple_table(["A"], [["1"]])  # Perfect match
    t2_pred = _simple_table(["B"], [["999"]])  # Bad match

    expected = t1_gt + t2_gt
    actual = t1_pred + t2_pred
    result = metric.compute(expected=expected, actual=actual)
    # Average of ~1.0 and a low score
    assert 0.2 < result[0].value < 0.9


# ---- 15. Mismatched table counts → unmatched GT tables score 0.0 ----
def test_mismatched_table_counts(metric: TableRecordMatchMetric) -> None:
    t1 = _simple_table(["A"], [["1"]])
    t2 = _simple_table(["B"], [["2"]])
    t3 = _simple_table(["C"], [["3"]])

    expected = t1 + t2 + t3  # 3 GT tables
    actual = t1 + t2  # 2 predicted tables

    result = metric.compute(expected=expected, actual=actual)
    # Two tables score ~1.0 each, third scores 0.0 → avg ~0.67
    assert 0.55 <= result[0].value <= 0.75
    assert result[0].metadata["n_gt_tables"] == 3
    assert result[0].metadata["n_pred_tables"] == 2


# ---- 15b. Extra predicted tables → no penalty ----
def test_extra_predicted_tables_no_penalty(metric: TableRecordMatchMetric) -> None:
    t1 = _simple_table(["A"], [["1"]])
    t2 = _simple_table(["B"], [["2"]])
    t3 = _simple_table(["C"], [["3"]])

    expected = t1 + t2  # 2 GT tables
    actual = t1 + t2 + t3  # 3 predicted tables (1 extra)

    result = metric.compute(expected=expected, actual=actual)
    # Both GT tables matched perfectly, extra pred table ignored → 1.0
    assert result[0].value == pytest.approx(1.0)
    assert result[0].metadata["n_gt_tables"] == 2
    assert result[0].metadata["n_pred_tables"] == 3


# ---- 16. Duplicate column headers → disambiguated with _0, _1 suffixes ----
def test_duplicate_column_headers(metric: TableRecordMatchMetric) -> None:
    gt = _html("<thead><tr><th>Value</th><th>Value</th></tr></thead><tbody><tr><td>A</td><td>B</td></tr></tbody>")
    tables = parse_html_tables(gt)
    keys, records, _ = table_to_records(tables[0])
    assert len(keys) == 2
    assert keys[0] != keys[1], f"Keys should be disambiguated, got: {keys}"
    assert "Value" in keys[0] and "Value" in keys[1]


# ---- 17. No tables in actual → 0.0 ----
def test_no_tables_in_actual(metric: TableRecordMatchMetric) -> None:
    gt = _simple_table(["A", "B"], [["1", "2"]])
    result = metric.compute(expected=gt, actual="No tables here")
    assert len(result) == 2
    assert result[0].value == 0.0
    assert result[0].metadata["tables_predicted"] is False
    assert result[1].metric_name == "table_record_match_perfect"
    assert result[1].value == 0.0


# ---- Additional: cell_score unit tests ----
def test_cell_score_exact_match() -> None:
    assert cell_score("hello", "hello") == 1.0


def test_cell_score_empty_both() -> None:
    assert cell_score("", "") == 1.0
    assert cell_score("nan", "NaN") == 1.0


def test_cell_score_one_empty() -> None:
    assert cell_score("hello", "") == 0.0
    assert cell_score("", "hello") == 0.0


def test_cell_score_numeric_strict() -> None:
    assert cell_score("100", "101") == 0.0
    assert cell_score("100", "100") == 1.0


def test_cell_score_text_strict() -> None:
    assert cell_score("Revenue", "Revnue") == 0.0
    assert cell_score("Revenue", "Revenue") == 1.0


# ---- No GT tables → empty list ----
def test_no_gt_tables(metric: TableRecordMatchMetric) -> None:
    result = metric.compute(expected="No tables", actual=_simple_table(["A"], [["1"]]))
    assert result == []


# ===========================================================================
# Bug-fix regression tests (Notion triage doc fixes)
# ===========================================================================


# ---- Fix 1: <br> tag boundaries preserve spaces in parse_html_tables ----


class TestBrTagSpacePreservation:
    """Fix 1: parse_html_tables must not merge words across <br> boundaries.

    Before the fix, `cell.get_text(strip=True)` would concatenate text nodes
    without separators, so `<td>Name and<br>location</td>` became
    "Name andlocation" instead of "Name and location".
    """

    def test_br_preserves_space_between_words(self) -> None:
        """<br> between words should produce a space, not merge them."""
        html = _html("<thead><tr><th>Col</th></tr></thead><tbody><tr><td>Name and<br>location</td></tr></tbody>")
        tables = parse_html_tables(html)
        cell_value = tables[0].data[1][0]  # first data row, first col
        assert "and location" in cell_value or "and\nlocation" in cell_value
        assert "andlocation" not in cell_value

    def test_br_in_header_preserves_space(self) -> None:
        """<br> inside header cells should also preserve word boundaries."""
        html = _html("<thead><tr><th>Earned Car<br>Years</th></tr></thead><tbody><tr><td>100</td></tr></tbody>")
        tables = parse_html_tables(html)
        header_value = tables[0].data[0][0]  # header row
        assert "Car Years" in header_value or "Car\nYears" in header_value
        assert "CarYears" not in header_value

    def test_br_does_not_affect_cells_without_br(self) -> None:
        """Normal cells without <br> should be unaffected by the fix."""
        html = _html("<thead><tr><th>A</th></tr></thead><tbody><tr><td>hello world</td></tr></tbody>")
        tables = parse_html_tables(html)
        assert tables[0].data[1][0] == "hello world"

    def test_multiple_br_tags(self) -> None:
        """Multiple <br> tags should each produce a space."""
        html = _html("<thead><tr><th>Col</th></tr></thead><tbody><tr><td>A<br>B<br>C</td></tr></tbody>")
        tables = parse_html_tables(html)
        value = tables[0].data[1][0]
        # Should have spaces (or newlines) between A, B, C — never merged
        assert "AB" not in value
        assert "BC" not in value

    def test_end_to_end_br_does_not_penalize_score(self, metric: TableRecordMatchMetric) -> None:
        """A GT with <br> and a prediction without should still match."""
        gt = _html("<thead><tr><th>Category</th></tr></thead><tbody><tr><td>Contracts,<br>Employment</td></tr></tbody>")
        pred = _html("<thead><tr><th>Category</th></tr></thead><tbody><tr><td>Contracts, Employment</td></tr></tbody>")
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)


# ---- Fix 2: Whitespace around currency/punctuation in cell_score ----


class TestCellTextNormalization:
    """Fix 2: cell_score must treat "$ 5,061" and "$5,061" as equal.

    _normalize_trm_cell_text() runs inside cell_score only (not shared
    normalize_text). It handles whitespace around $/%/parens and
    thousands-separator commas in numeric contexts.
    """

    # -- whitespace around punctuation --

    def test_dollar_space(self) -> None:
        """Space after $ should be ignored: '$ 5,061' == '$5,061'."""
        assert cell_score(_norm("$ 5,061"), _norm("$5,061")) == 1.0

    def test_dollar_multiple_spaces(self) -> None:
        """Multiple spaces after $ should also be collapsed."""
        assert cell_score(_norm("$  315"), _norm("$315")) == 1.0

    def test_percent_space(self) -> None:
        """Space before % should be ignored: '50 %' == '50%'."""
        assert cell_score(_norm("50 %"), _norm("50%")) == 1.0

    def test_open_paren_space(self) -> None:
        """Space before ( should be ignored: 'word (n)' == 'word(n)'."""
        assert cell_score(_norm("word (n)"), _norm("word(n)")) == 1.0

    def test_close_paren_space(self) -> None:
        """Space before ) should be ignored: '(n )' == '(n)'."""
        assert cell_score(_norm("(n )"), _norm("(n)")) == 1.0

    def test_combined_currency_and_paren(self) -> None:
        """Combined patterns: '$ (5,061)' == '$(5,061)'."""
        assert cell_score(_norm("$ (5,061)"), _norm("$(5,061)")) == 1.0

    # -- thousands-separator comma removal --

    def test_dollar_with_comma(self) -> None:
        """'$5,061' == '$5061' — comma is a thousands separator."""
        assert cell_score(_norm("$5,061"), _norm("$5061")) == 1.0

    def test_plain_number_with_comma(self) -> None:
        """'1,000' == '1000' — plain numeric value."""
        assert cell_score(_norm("1,000"), _norm("1000")) == 1.0

    def test_large_number_multiple_commas(self) -> None:
        """'1,000,000' == '1000000' — multiple thousands separators."""
        assert cell_score(_norm("1,000,000"), _norm("1000000")) == 1.0

    def test_percentage_with_comma(self) -> None:
        """'1,234%' == '1234%' — percentage with comma."""
        assert cell_score(_norm("1,234%"), _norm("1234%")) == 1.0

    def test_negative_number_with_comma(self) -> None:
        """'-1,500' == '-1500'."""
        assert cell_score(_norm("-1,500"), _norm("-1500")) == 1.0

    def test_parenthetical_negative_with_comma(self) -> None:
        """'(1,500)' == '(1500)' — accounting-style negatives."""
        assert cell_score(_norm("(1,500)"), _norm("(1500)")) == 1.0

    def test_comma_in_text_preserved(self) -> None:
        """'Smith, John' must NOT match 'Smith John' — not a number."""
        assert cell_score(_norm("Smith, John"), _norm("Smith John")) == 0.0

    def test_comma_between_words_preserved(self) -> None:
        """'Revenue, net' must NOT match 'Revenue net'."""
        assert cell_score(_norm("Revenue, net"), _norm("Revenue net")) == 0.0

    def test_genuine_difference_still_fails(self) -> None:
        """Actual value differences must still score 0.0."""
        assert cell_score(_norm("$5,061"), _norm("$5,062")) == 0.0

    # -- helper function direct tests --

    def test_normalize_trm_cell_text_whitespace(self) -> None:
        """Direct test of whitespace rules."""
        assert _normalize_trm_cell_text("$ 315") == "$315"
        assert _normalize_trm_cell_text("50 %") == "50%"
        assert _normalize_trm_cell_text("word (n)") == "word(n)"
        assert _normalize_trm_cell_text("(n )") == "(n)"

    def test_normalize_trm_cell_text_commas(self) -> None:
        """Direct test of comma removal rules."""
        assert _normalize_trm_cell_text("$5,061") == "$5061"
        assert _normalize_trm_cell_text("1,000,000") == "1000000"
        assert _normalize_trm_cell_text("smith, john") == "smith, john"

    def test_normalize_trm_cell_text_registered_symbol(self) -> None:
        """Whitespace before ® is stripped: 'Apple ® Inc' → 'Apple® Inc'."""
        assert _normalize_trm_cell_text("Apple ® Inc") == "Apple® Inc"
        assert _normalize_trm_cell_text("Acme  ®") == "Acme®"
        # No space before → unchanged
        assert _normalize_trm_cell_text("Apple® Inc") == "Apple® Inc"

    def test_normalize_trm_cell_text_boolean_markers(self) -> None:
        """Visual boolean markers match bracketed textual booleans."""
        assert _norm("✓") == _norm("[yes]") == "yes"
        assert _norm("✔") == _norm("yes") == "yes"
        assert _norm("X") == _norm("[yes]") == "yes"
        assert _norm("x") == _norm("[yes]") == "yes"
        assert _norm("●") == _norm("[yes]") == "yes"
        assert _norm("✗") == _norm("[no]") == "no"
        assert _norm("✘") == _norm("no") == "no"
        assert _norm("○") == _norm("[no]") == "no"

    # -- end-to-end table test --

    def test_end_to_end_dollar_space_and_comma_table(self, metric: TableRecordMatchMetric) -> None:
        """Full table where GT has '$ 1,200' and pred has '$1200'."""
        gt = _simple_table(["Item", "Cost"], [["Widget", "$ 315"], ["Gadget", "$ 1,200"]])
        pred = _simple_table(["Item", "Cost"], [["Widget", "$315"], ["Gadget", "$1200"]])
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_end_to_end_teds_boolean_glyph_table(self, metric: TableRecordMatchMetric) -> None:
        """Check/cross glyphs should match [yes]/[no] in boolean matrix cells."""
        gt = _html(
            """
<tr><th>Dataset</th><th>TD</th><th>TSR</th><th>TR</th><th># tables</th></tr>
<tr><td>Marmot [6]</td><td>✓</td><td>✗</td><td>✗</td><td>958</td></tr>
<tr><td>ICDAR2013 [2]</td><td>✓</td><td>✓</td><td>✓</td><td>156</td></tr>
<tr><td>SciTSR³</td><td>✗</td><td>✓</td><td>✓</td><td>15k</td></tr>
"""
        )
        pred = _html(
            """
<thead><tr><th>Dataset</th><th>TD</th><th>TSR</th><th>TR</th><th># tables</th></tr></thead>
<tbody>
<tr><td>Marmot [6]</td><td>[yes]</td><td>[no]</td><td>[no]</td><td>958</td></tr>
<tr><td>ICDAR2013 [2]</td><td>[yes]</td><td>[yes]</td><td>[yes]</td><td>156</td></tr>
<tr><td>SciTSR<sup>3</sup></td><td>[no]</td><td>[yes]</td><td>[yes]</td><td>15k</td></tr>
</tbody>
"""
        )

        result = metric.compute(expected=gt, actual=pred)

        assert result[0].value == pytest.approx(1.0)

    def test_end_to_end_hospital_dot_table(self, metric: TableRecordMatchMetric) -> None:
        """Filled dot availability markers should match [yes] cells."""
        gt = _html(
            """
<tr><th>Hospital</th><th>County</th><th>Trio HMO</th><th>Local Access+ / SaveNet</th><th>Flag</th></tr>
<tr><td>Alameda Hospital</td><td>Alameda</td><td>●</td><td>●</td><td>X</td></tr>
<tr><td>Eden Medical Center</td><td>Alameda</td><td>●</td><td></td><td></td></tr>
"""
        )
        pred = _html(
            """
<thead>
<tr><th>Hospital</th><th>County</th><th>Trio HMO</th><th>Local Access+ /<br>SaveNet</th><th>Flag</th></tr>
</thead>
<tbody>
<tr><td>Alameda Hospital</td><td>Alameda</td><td>[yes]</td><td>[yes]</td><td>[yes]</td></tr>
<tr><td>Eden Medical Center</td><td>Alameda</td><td>[yes]</td><td></td><td></td></tr>
</tbody>
"""
        )

        result = metric.compute(expected=gt, actual=pred)

        assert result[0].value == pytest.approx(1.0)


# ---- Fix 3: Superscript digit stripping in normalize_text ----


class TestSuperscriptDigitStripping:
    """Fix 3: Unicode superscript digits (¹²³ etc.) should be stripped.

    These codepoints (U+00B9, U+00B2, U+00B3, U+2070, U+2074–U+2079) are
    typically footnote markers. NFD decomposition does NOT decompose them.
    We strip them (not convert to regular digits) to avoid changing values
    like "84.1¹" → "84.11". This is consistent with <sup> tag removal.
    """

    def test_superscript_one_stripped(self) -> None:
        """U+00B9 (¹) should be stripped."""
        assert normalize_text("84.1\u00b9") == normalize_text("84.1")

    def test_superscript_two_stripped(self) -> None:
        """U+00B2 (²) should be stripped."""
        assert normalize_text("100\u00b2") == normalize_text("100")

    def test_superscript_three_stripped(self) -> None:
        """U+00B3 (³) should be stripped."""
        assert normalize_text("value\u00b3") == normalize_text("value")

    def test_superscript_higher_digits_stripped(self) -> None:
        """U+2074–U+2079 (⁴⁵⁶⁷⁸⁹) should be stripped."""
        for cp in ["\u2074", "\u2075", "\u2076", "\u2077", "\u2078", "\u2079"]:
            assert normalize_text(f"test{cp}") == normalize_text("test"), f"Superscript {cp!r} was not stripped"

    def test_superscript_zero_stripped(self) -> None:
        """U+2070 (⁰) should be stripped."""
        assert normalize_text("x\u2070") == normalize_text("x")

    def test_multiple_superscripts_stripped(self) -> None:
        """Multiple consecutive superscript digits should all be stripped."""
        assert normalize_text("note\u00b9\u00b2") == normalize_text("note")

    def test_superscript_not_converted_to_digit(self) -> None:
        """Stripping must NOT convert ¹ to 1 (would change numeric values)."""
        result = normalize_text("84.1\u00b9")
        assert "84.11" not in result
        assert "84.1" in result

    def test_cell_score_with_superscript(self) -> None:
        """cell_score should match values that differ only by superscript."""
        assert cell_score(_norm("84.1\u00b9"), _norm("84.1")) == 1.0
        assert cell_score(_norm("Total\u00b2"), _norm("Total")) == 1.0

    def test_regular_digits_unaffected(self) -> None:
        """Regular digits 0-9 must NOT be stripped."""
        assert "123" in normalize_text("123")
        assert normalize_text("84.1") == normalize_text("84.1")

    def test_html_sup_vs_unicode_superscript_header_match(self) -> None:
        """<sup>1</sup> in header must normalize the same as unicode ¹.

        When one table uses <th>Name<sup>1</sup></th> and the other uses
        <th>Name¹</th>, both should produce the same header text so that
        column alignment succeeds.
        """
        gt_html = (
            "<table><thead><tr>"
            "<th>Number</th><th>Name<sup>1</sup></th><th>Description</th>"
            "</tr></thead><tbody>"
            "<tr><td>1</td><td>Beet</td><td>Leafy</td></tr>"
            "</tbody></table>"
        )
        pred_html = (
            "<table><thead><tr>"
            "<th>Number</th><th>Name\u00b9</th><th>Description</th>"
            "</tr></thead><tbody>"
            "<tr><td>1</td><td>Beet</td><td>Leafy</td></tr>"
            "</tbody></table>"
        )
        gt_tables = parse_html_tables(gt_html)
        pred_tables = parse_html_tables(pred_html)
        assert len(gt_tables) == 1 and len(pred_tables) == 1
        gt_keys, gt_records, _ = table_to_records(normalize_table(gt_tables[0]))
        pred_keys, pred_records, _ = table_to_records(normalize_table(pred_tables[0]))
        # Headers should match — both should normalize "Name" the same way
        assert gt_keys == pred_keys
        # Records should therefore match perfectly
        assert gt_records == pred_records


# ===========================================================================
# Phase 1: Nested table extraction and parsing
# ===========================================================================


class TestNestedTableExtraction:
    """Phase 1: Nested tables should not break extraction or parsing."""

    def test_extract_html_tables_nested(self) -> None:
        """Nested table inside a cell should not truncate the outer table."""
        html = (
            "<table><tr><td>A</td><td>"
            "<table><tr><td>Nested</td></tr></table>"
            "</td></tr><tr><td>B</td><td>C</td></tr></table>"
        )
        tables = extract_html_tables(html)
        assert len(tables) == 1  # One top-level table, not two
        assert "B" in tables[0]  # Second row NOT truncated

    def test_extract_html_tables_nested_preserves_outer(self) -> None:
        """Outer table should contain the full HTML including nested table."""
        html = (
            "<table><tr><td>Row1</td></tr>"
            "<tr><td><table><tr><td>Inner</td></tr></table></td></tr>"
            "<tr><td>Row3</td></tr></table>"
        )
        tables = extract_html_tables(html)
        assert len(tables) == 1
        assert "Row3" in tables[0]

    def test_extract_multiple_top_level_with_nested(self) -> None:
        """Multiple top-level tables, one with a nested table."""
        html = "<table><tr><td>T1</td></tr></table><table><tr><td><table><tr><td>N</td></tr></table></td></tr></table>"
        tables = extract_html_tables(html)
        assert len(tables) == 2

    def test_extract_does_not_match_table_prefix(self) -> None:
        """<tabledata> or similar tags should not be matched as <table>."""
        html = "<tabledata>foo</tabledata><table><tr><td>Real</td></tr></table>"
        tables = extract_html_tables(html)
        assert len(tables) == 1
        assert "Real" in tables[0]

    def test_parse_html_tables_nested_no_row_leak(self) -> None:
        """Nested table rows must not appear in the outer table's grid."""
        html = (
            "<table>"
            "<tr><td>Cell A</td><td>"
            "<table><tr><td>Nested1</td></tr><tr><td>Nested2</td></tr></table>"
            "</td></tr>"
            "<tr><td>Cell B</td><td>Cell C</td></tr>"
            "</table>"
        )
        tables = parse_html_tables(html)
        # Should be 1 top-level table (nested table becomes cell text)
        assert len(tables) == 1
        assert tables[0].data.shape == (2, 2)  # 2 rows, 2 cols
        assert tables[0].data[0, 0] == "Cell A"
        assert tables[0].data[1, 0] == "Cell B"

    def test_nested_table_becomes_cell_text(self) -> None:
        """Nested table content should appear as text in the containing cell."""
        html = "<table><tr><td>X</td><td><table><tr><td>A</td></tr><tr><td>B</td></tr></table></td></tr></table>"
        tables = parse_html_tables(html)
        assert len(tables) == 1
        # The cell with the nested table should contain flattened text
        cell_val = tables[0].data[0, 1]
        assert "A" in cell_val
        assert "B" in cell_val

    def test_end_to_end_nested_table_scores(self, metric: TableRecordMatchMetric) -> None:
        """A table with nested content should still score well against flat equivalent."""
        gt = _simple_table(["Name", "Details"], [["Alice", "Note A Note B"], ["Bob", "Note C"]])
        pred = (
            "<table><thead><tr><th>Name</th><th>Details</th></tr></thead>"
            "<tbody>"
            "<tr><td>Alice</td><td>"
            "<table><tr><td>Note A</td></tr><tr><td>Note B</td></tr></table>"
            "</td></tr>"
            "<tr><td>Bob</td><td>Note C</td></tr>"
            "</tbody></table>"
        )
        result = metric.compute(expected=gt, actual=pred)
        # Should not crash; should find 1 table on each side
        assert len(result) == 2
        assert result[0].metadata["n_pred_tables"] == 1


# ===========================================================================
# Phase 2: <td> title row detection
# ===========================================================================


class TestTdTitleRows:
    """Phase 2: <td> title rows should be excluded like <th> title rows."""

    def test_td_title_row_excluded_from_records(self) -> None:
        """A <td> row spanning full width with uniform text is a title, not data."""
        html = _html(
            '<tr><td colspan="2">Table Title</td></tr>'
            "<thead><tr><th>A</th><th>B</th></tr></thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # Title row should not appear as a record
        assert len(records) == 1
        assert records[0] != {"A": "Table Title", "B": "Table Title"}

    def test_td_title_row_not_in_keys(self) -> None:
        """<td> title row text should not appear in column keys."""
        html = _html(
            '<tr><td colspan="2">Sales Data</td></tr>'
            "<thead><tr><th>Q1</th><th>Q2</th></tr></thead>"
            "<tbody><tr><td>100</td><td>200</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        assert keys == ["Q1", "Q2"]

    def test_td_title_matches_th_title(self, metric: TableRecordMatchMetric) -> None:
        """GT with <th> title and pred with <td> title should score 1.0."""
        gt = _html(
            "<thead>"
            '<tr><th colspan="2">Title</th></tr>'
            "<tr><th>A</th><th>B</th></tr>"
            "</thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
        )
        pred = _html(
            '<tr><td colspan="2">Title</td></tr>'
            "<thead><tr><th>A</th><th>B</th></tr></thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_td_title_vs_no_title(self, metric: TableRecordMatchMetric) -> None:
        """Pred with <td> title row vs GT without title should still score 1.0."""
        gt = _simple_table(["A", "B"], [["1", "2"]])
        pred = _html(
            '<tr><td colspan="2">Some Title</td></tr>'
            "<thead><tr><th>A</th><th>B</th></tr></thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_multiple_td_title_rows(self) -> None:
        """Multiple consecutive <td> title rows should all be excluded."""
        html = _html(
            '<tr><td colspan="2">Title Line 1</td></tr>'
            '<tr><td colspan="2">Title Line 2</td></tr>'
            "<thead><tr><th>X</th><th>Y</th></tr></thead>"
            "<tbody><tr><td>a</td><td>b</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        assert len(records) == 1
        assert keys == ["X", "Y"]

    def test_non_uniform_td_row_is_not_title(self) -> None:
        """A <td> row with different values in each column is data, not a title."""
        html = _html(
            "<tr><td>Cat A</td><td>Cat B</td></tr>"
            "<thead><tr><th>X</th><th>Y</th></tr></thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # "Cat A" / "Cat B" is a data row, not a title — it must appear in records
        cat_values = [r for r in records if "Cat A" in r.values() or "Cat B" in r.values()]
        assert len(cat_values) > 0, "Non-uniform <td> row should be data, not a title"

    def test_headerless_table_uniform_row_not_title(self) -> None:
        """In a fully headerless table where ALL rows are uniform, none are titles.

        If every row has uniform text across columns, the entire table
        is just data — we can't strip all rows as titles.
        """
        html = _html("<tr><td>Yes</td><td>Yes</td></tr><tr><td>No</td><td>No</td></tr>")
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        assert len(records) == 2  # Both rows are data, not titles

    def test_headerless_table_with_td_title(self) -> None:
        """A headerless table with a <td colspan> title row followed by data rows.

        The title row should be stripped even without any <th> cells.
        """
        html = _html(
            '<tr><td colspan="2">Report Title</td></tr>'
            "<tr><td>Alice</td><td>30</td></tr>"
            "<tr><td>Bob</td><td>25</td></tr>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        assert len(records) == 2  # Title row excluded, 2 data rows remain
        # Title text should not appear in records
        for r in records:
            assert "Report Title" not in r.values()

    def test_single_column_td_row_not_title(self) -> None:
        """In a 1-column table, a leading <td> row must NOT be treated as a title.

        "Uniform text across all columns" is trivially true for 1 column,
        so the n_cols > 1 guard must prevent false positives.
        """
        html = _html("<tr><td>Row 1</td></tr><tr><td>Row 2</td></tr>")
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        assert len(records) == 2  # Both rows are data, not titles


# ===========================================================================
# Phase 3: Section header removal
# ===========================================================================


class TestSectionHeaderRemoval:
    """Phase 3: Section headers are regular data rows — no special treatment."""

    def test_th_section_row_becomes_record(self) -> None:
        """A mid-table <th> row should be emitted as a regular record."""
        html = _html(
            "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>A</td><td>10</td></tr>"
            '<tr><th colspan="2">Section B</th></tr>'
            "<tr><td>B1</td><td>20</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # 3 records: A/10, Section B/Section B, B1/20
        assert len(records) == 3

    def test_th_vs_td_section_row_match(self, metric: TableRecordMatchMetric) -> None:
        """GT with <th> section row and pred with <td> equivalent should score 1.0."""
        gt = _html(
            "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>A</td><td>10</td></tr>"
            '<tr><th colspan="2">Section B</th></tr>'
            "<tr><td>B1</td><td>20</td></tr>"
            "</tbody>"
        )
        pred = _html(
            "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>A</td><td>10</td></tr>"
            '<tr><td colspan="2">Section B</td></tr>'
            "<tr><td>B1</td><td>20</td></tr>"
            "</tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_no_section_field_in_records(self) -> None:
        """Records must not contain _section metadata field."""
        html = _html(
            "<thead><tr><th>A</th><th>B</th></tr></thead>"
            "<tbody>"
            "<tr><td>1</td><td>2</td></tr>"
            '<tr><th colspan="2">Section</th></tr>'
            "<tr><td>3</td><td>4</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        for r in records:
            assert "_section" not in r

    def test_section_header_both_sides(self, metric: TableRecordMatchMetric) -> None:
        """Both sides with same section headers should score 1.0."""
        table = _html(
            "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>A</td><td>10</td></tr>"
            '<tr><th colspan="2">Group</th></tr>'
            "<tr><td>B</td><td>20</td></tr>"
            "</tbody>"
        )
        result = TableRecordMatchMetric().compute(expected=table, actual=table)
        assert result[0].value == pytest.approx(1.0)

    def test_missing_section_row_penalizes_proportionally(self, metric: TableRecordMatchMetric) -> None:
        """If GT has a section row and pred omits it, penalty is proportional (1 missing row)."""
        gt = _html(
            "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>A</td><td>10</td></tr>"
            '<tr><th colspan="2">Section B</th></tr>'
            "<tr><td>B1</td><td>20</td></tr>"
            "</tbody>"
        )
        pred = _html(
            "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>A</td><td>10</td></tr>"
            "<tr><td>B1</td><td>20</td></tr>"
            "</tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        # 2 of 3 GT records matched → ~0.67, not 1.0 (section row is now a real record)
        assert 0.55 <= result[0].value <= 0.75


# ===========================================================================
# Combined integration test (all three normalizations)
# ===========================================================================


class TestCombinedNormalizations:
    """All three normalizations together in realistic scenarios."""

    def test_combined_nested_td_title_and_section(self, metric: TableRecordMatchMetric) -> None:
        """All three normalizations in one table: nested table + <td> title + <th> section."""
        gt = _html(
            "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>A</td><td>10</td></tr>"
            "<tr><td>B</td><td>20</td></tr>"
            "</tbody>"
        )
        pred = (
            "<table>"
            '<tr><td colspan="2">Report Title</td></tr>'
            "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>A</td><td>10</td></tr>"
            '<tr><th colspan="2">Section X</th></tr>'
            "<tr><td>B</td><td>"
            "<table><tr><td>20</td></tr></table>"
            "</td></tr>"
            "</tbody>"
            "</table>"
        )
        result = metric.compute(expected=gt, actual=pred)
        # Should find 1 table on each side (nested table is cell text, not a separate table)
        assert result[0].metadata["n_gt_tables"] == 1
        assert result[0].metadata["n_pred_tables"] == 1
        # <td> title row excluded, section row is a record, nested table content is cell text
        # GT has 2 data records; pred has 3 (A, Section X, B) — section row is extra
        # Score should reflect 2 matched out of 3, not crash or score 0
        assert result[0].value > 0.5


# ===========================================================================
# Row-key removal tests
# ===========================================================================


class TestNoRowKey:
    """Records should not contain _row_key metadata."""

    def test_dual_axis_no_row_key(self) -> None:
        """Dual-axis table records should NOT have _row_key field."""
        html = _html(
            "<thead><tr><th></th><th>Q1</th><th>Q2</th></tr></thead>"
            "<tbody>"
            "<tr><th>Revenue</th><td>100</td><td>200</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        for r in records:
            assert "_row_key" not in r

    def test_dual_axis_row_header_in_data(self) -> None:
        """Row header <th> values should appear as regular data columns."""
        html = _html(
            "<thead><tr><th></th><th>Q1</th><th>Q2</th></tr></thead>"
            "<tbody>"
            "<tr><th>Revenue</th><td>100</td><td>200</td></tr>"
            "<tr><th>Cost</th><td>50</td><td>80</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # All columns should be in keys (including the row-header column)
        assert len(keys) == 3
        # Revenue/Cost should appear as values in records
        all_values = [v for r in records for v in r.values()]
        assert "Revenue" in all_values
        assert "Cost" in all_values


# ===========================================================================
# Partial-span title detection tests
# ===========================================================================


class TestPartialSpanTitle:
    """Title detection for headers that span most but not all columns."""

    def test_title_with_empty_corner(self) -> None:
        """A spanning header missing the leftmost corner cell is still a title."""
        html = _html(
            "<thead>"
            '<tr><th></th><th colspan="2">Sales</th></tr>'
            "<tr><th>Region</th><th>Q1</th><th>Q2</th></tr>"
            "</thead>"
            "<tbody><tr><td>East</td><td>100</td><td>200</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # "Sales" should be excluded as a title — it spans all columns
        # that Q1/Q2 span but misses the empty corner
        assert "Sales" not in " ".join(keys)
        assert "Q1" in keys or any("Q1" in k for k in keys)

    def test_title_below_leaf_headers(self) -> None:
        """A spanning header BELOW the leaf column headers is still a title."""
        html = _html(
            "<thead>"
            "<tr><th>Q1</th><th>Q2</th></tr>"
            '<tr><th colspan="2">Sales</th></tr>'
            "</thead>"
            "<tbody><tr><td>100</td><td>200</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # "Sales" spans all columns as a redundant leaf → title, excluded
        assert keys == ["Q1", "Q2"]

    def test_title_below_with_empty_corner(self) -> None:
        """Spanning header below leaf headers, with an empty corner cell."""
        html = _html(
            "<thead>"
            "<tr><th>Region</th><th>Q1</th><th>Q2</th></tr>"
            '<tr><th></th><th colspan="2">Sales</th></tr>'
            "</thead>"
            "<tbody><tr><td>East</td><td>100</td><td>200</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # "Sales" covers all columns that have headers above (Q1, Q2)
        # minus the edge (Region) → title, excluded
        assert "Sales" not in " ".join(keys)
        assert "Region" in keys

    def test_title_below_vs_above_match(self, metric: TableRecordMatchMetric) -> None:
        """GT and pred both with title below (trailing) should score 1.0."""
        gt = _html(
            "<thead>"
            "<tr><th>Q1</th><th>Q2</th></tr>"
            '<tr><th colspan="2">Sales</th></tr>'
            "</thead>"
            "<tbody><tr><td>100</td><td>200</td></tr></tbody>"
        )
        pred = _html(
            "<thead>"
            "<tr><th>Q1</th><th>Q2</th></tr>"
            '<tr><th colspan="2">Sales</th></tr>'
            "</thead>"
            "<tbody><tr><td>100</td><td>200</td></tr></tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_group_header_not_title(self) -> None:
        """A header that spans only SOME data columns is a group header, not a title."""
        html = _html(
            "<thead>"
            '<tr><th colspan="2">Group A</th><th>Other</th></tr>'
            "<tr><th>X</th><th>Y</th><th>Z</th></tr>"
            "</thead>"
            "<tbody><tr><td>1</td><td>2</td><td>3</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # "Group A" spans only 2 of 3 data columns — NOT a title
        assert any("Group A" in k for k in keys)

    def test_trailing_th_title_becomes_data_record(self) -> None:
        """A <th> title row at the END of a multi-row header block becomes a data record."""
        html = _html(
            "<tr><th>Disorder</th><th>Medical History</th><th>Physical Exam</th></tr>"
            '<tr><th colspan="3">SPINAL DISORDERS</th></tr>'
            "<tr><td>Fracture</td><td>Major trauma</td><td>Percussion</td></tr>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # Keys should come from the real header row only
        assert keys == ["Disorder", "Medical History", "Physical Exam"]
        # SPINAL DISORDERS should be emitted as a data record
        assert len(records) == 2
        assert records[0] == {
            "Disorder": "SPINAL DISORDERS",
            "Medical History": "SPINAL DISORDERS",
            "Physical Exam": "SPINAL DISORDERS",
        }
        assert records[1] == {
            "Disorder": "Fracture",
            "Medical History": "Major trauma",
            "Physical Exam": "Percussion",
        }

    def test_trailing_th_title_matches_td_equivalent(self) -> None:
        """<th> title at end of header block should produce same records as <td> equivalent."""
        th_html = _html(
            '<tr><th>A</th><th>B</th></tr><tr><th colspan="2">Section</th></tr><tr><td>1</td><td>2</td></tr>'
        )
        td_html = _html(
            '<tr><th>A</th><th>B</th></tr><tr><td colspan="2">Section</td></tr><tr><td>1</td><td>2</td></tr>'
        )
        th_tables = parse_html_tables(th_html)
        td_tables = parse_html_tables(td_html)
        th_keys, th_records, _ = table_to_records(th_tables[0])
        td_keys, td_records, _ = table_to_records(td_tables[0])
        assert th_keys == td_keys
        assert th_records == td_records

    def test_leading_th_title_still_skipped(self) -> None:
        """A <th> title row at the START of a multi-row header block stays skipped."""
        html = _html(
            "<thead>"
            '<tr><th colspan="2">Title</th></tr>'
            "<tr><th>A</th><th>B</th></tr>"
            "</thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        assert keys == ["A", "B"]
        # Leading title row should NOT become a data record
        assert len(records) == 1
        assert records[0] == {"A": "1", "B": "2"}

    def test_two_th_title_rows_at_top_no_double_emit(self) -> None:
        """Two consecutive single-value <th> rows at the top.

        Locks in the expected behavior for this shape:
          1. Strip the top <th> row as the table title.
          2. After stripping, only one header row remains, so the
             section-header carve-out (which fires only inside a
             multi-row header block) does NOT engage.
          3. The remaining <th> row stays as the column header and is
             not also emitted as data — exactly two data records (A, B).
        """
        from parse_bench.evaluation.metrics.parse.table_extraction import (
            ExtractedTable,
        )
        from parse_bench.evaluation.metrics.parse.table_record_match_metric import (
            extract_header_info_from_hints,
        )
        from parse_bench.evaluation.metrics.parse.table_title_stripping import (
            strip_title_rows,
        )

        html = _html(
            "<tbody>"
            '<tr><th colspan="2">NAFOCG Tier Codes:</th></tr>'
            '<tr><th colspan="2">NAFOCG Tie/AF_OCG Ratio</th></tr>'
            "<tr><td>A</td><td>&lt; 1</td></tr>"
            "<tr><td>B</td><td>= 1</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        et = strip_title_rows(ExtractedTable(raw_html="", table_data=tables[0]))
        header = extract_header_info_from_hints(et.table_data, et.header_hints)
        records = build_records(et.table_data, header)

        # Top title stripped — recorded in stripped_titles, not in column keys
        assert any("nafocg tier codes" in t.lower() for t in header.stripped_titles)
        assert not any("Tier Codes" in k for k in header.keys)

        # Exactly two data records (A and B) — the remaining <th> row is
        # the column header, not a section header, and must not be
        # double-classified as data.
        assert len(records) == 2
        data_values = [tuple(r.values()) for r in records]
        assert ("A", "< 1") in data_values
        assert ("B", "= 1") in data_values
        # Section-header text must not also appear as a data record
        assert not any(any("Tie/AF_OCG Ratio" in str(v) for v in r.values()) for r in records)

    def test_two_th_title_rows_e2e_no_extra_prediction(self, metric: TableRecordMatchMetric) -> None:
        """End-to-end regression for the BRWS-134565917 bug.

        Pred has two consecutive single-value <th> rows at the top
        (title + section header). GT contains the same content but with
        plain <td> cells (no <th> markup), so it parses with synthetic
        column keys and the section-header text as the first data row.

        Pre-fix: the pred-side header detector kept the title <th> row as
        a header AND re-emitted the second <th> row as data via the
        trailing-th-title carve-out, producing a spurious extra predicted
        record (no GT match) that dragged TRM below ~0.65.

        Post-fix: pred's title row is stripped, the remaining <th> row is
        the column header, and there is no spurious extra record.
        """
        pred = _html(
            "<tbody>"
            '<tr><th colspan="2">NAFOCG Tier Codes:</th></tr>'
            '<tr><th colspan="2">NAFOCG Tie/AF_OCG Ratio</th></tr>'
            "<tr><td>A</td><td>&lt; 1</td></tr>"
            "<tr><td>B</td><td>= 1</td></tr>"
            "</tbody>"
        )
        gt = _html(
            "<tbody>"
            "<tr><td>NAFOCG Tier Codes:</td><td>NAFOCG Tie/AF_OCG Ratio</td></tr>"
            "<tr><td>A</td><td>&lt; 1</td></tr>"
            "<tr><td>B</td><td>= 1</td></tr>"
            "</tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        # Pre-fix this scored ~0.625 (spurious Pred#1 extra prediction).
        # Post-fix it scores ~0.833 — the extra prediction is gone.
        assert result[0].metric_name == "table_record_match"
        assert result[0].value > 0.75

    def test_leading_all_empty_row_stripped(self, metric: TableRecordMatchMetric) -> None:
        """Leading all-empty rows are physically removed by the strip stage.

        Before this fix, a leading ``<tr><td></td><td></td></tr>`` caused
        ``find_col_header_rows`` to break at row 0 (not a ``<th>`` row),
        losing the real ``<th>`` header block behind it. Column keys fell
        back to synthetic ``col_0`` / ``col_1``, and comparing against a
        GT with real headers dropped TRM to 0.0.
        """
        from parse_bench.evaluation.metrics.parse.table_extraction import (
            ExtractedTable,
        )
        from parse_bench.evaluation.metrics.parse.table_title_stripping import (
            strip_title_rows,
        )

        html_with_empty = _html(
            "<tbody>"
            "<tr><td></td><td></td></tr>"
            "<tr><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td></tr>"
            "<tr><td>3</td><td>4</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html_with_empty)
        et = strip_title_rows(ExtractedTable(raw_html="", table_data=tables[0]))
        # Leading empty row is gone: 1 header + 2 data = 3 rows.
        assert et.table_data.data.shape[0] == 3
        assert et.header_hints is not None
        # Real header row is preserved (row index 0 in the trimmed table).
        assert et.header_hints.col_header_rows == frozenset({0})

        # End-to-end: GT clean, pred with a leading empty row → perfect.
        gt = _html("<tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr>")
        result = metric.compute(expected=gt, actual=html_with_empty)
        assert result[0].value == pytest.approx(1.0)

    def test_stripped_titles_not_double_normalized(self) -> None:
        """Regression: stripped_titles must be passed through verbatim from hints.

        ``strip_title_rows`` already collects ``stripped_titles`` from a
        normalized copy of the table, so the strings stored in
        ``HeaderHints.stripped_titles`` are single-normalized. Re-applying
        the per-cell normalization chain in ``extract_header_info_from_hints``
        would double-normalize them — and ``_normalize_trm_cell_text``'s
        comma-stripping rule is not idempotent (``"1,2,3" → "12,3" → "123"``),
        so a doubly-normalized title would fail to prefix-match the
        single-normalized column keys it is compared against.
        """
        from parse_bench.evaluation.metrics.parse.table_record_match_metric import (
            extract_header_info_from_hints,
        )
        from parse_bench.evaluation.metrics.parse.table_title_stripping import (
            HeaderHints,
        )

        # Build a minimal trimmed table; the contents of the table aren't
        # what's under test — only the title round-trip.
        tables = parse_html_tables(_html("<tr><td>x</td><td>y</td></tr>"))
        hints = HeaderHints(
            col_header_rows=frozenset(),
            th_title_rows=frozenset(),
            stripped_titles=frozenset({"foo 12,3"}),  # already single-normalized
        )
        header = extract_header_info_from_hints(tables[0], hints)
        # Must be preserved verbatim — NOT double-normalized to "foo 123".
        assert header.stripped_titles == {"foo 12,3"}


# ===========================================================================
# Multi-level key separator tests
# ===========================================================================


class TestKeySpaceSeparator:
    """Multi-level headers use space separator, matching concatenated forms."""

    def test_space_separator_key_format(self) -> None:
        """Keys from multi-level headers should use space, not ' > '."""
        html = _html(
            "<thead>"
            '<tr><th colspan="2">Group A</th><th>Other</th></tr>'
            "<tr><th>X</th><th>Y</th><th>Z</th></tr>"
            "</thead>"
            "<tbody><tr><td>1</td><td>2</td><td>3</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # Should be "Group A X", "Group A Y", "Z" — not "Group A > X"
        assert " > " not in " | ".join(keys)
        assert "Group A X" in keys
        assert "Group A Y" in keys

    def test_concatenated_vs_multilevel_match(self, metric: TableRecordMatchMetric) -> None:
        """'July €000s' in one cell should match 'July' + '€000s' in two header rows."""
        gt = _html(
            "<thead>"
            '<tr><th colspan="2">Period</th><th>Other</th></tr>'
            "<tr><th>July</th><th>Aug</th><th>Total</th></tr>"
            "</thead>"
            "<tbody><tr><td>100</td><td>200</td><td>300</td></tr></tbody>"
        )
        # Pred has the same structure — keys should match via space join
        pred = _html(
            "<thead>"
            '<tr><th colspan="2">Period</th><th>Other</th></tr>'
            "<tr><th>July</th><th>Aug</th><th>Total</th></tr>"
            "</thead>"
            "<tbody><tr><td>100</td><td>200</td><td>300</td></tr></tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Record details diagnostics
# ---------------------------------------------------------------------------


class TestRecordDetails:
    """Tests for build_record_details and its integration into compute()."""

    def test_build_record_details_matched_pair(self) -> None:
        """Matched records produce per-cell expected/actual/score entries."""
        gt_records = [{"Name": "Alice", "Age": "30"}]
        pred_records = [{"Name": "Alice", "Age": "30"}]
        col_mapping = {"Name": "Name", "Age": "Age"}
        matches = [(0, 0, 1.0)]

        details = build_record_details(gt_records, pred_records, col_mapping, matches)

        assert len(details) == 1
        d = details[0]
        assert d["type"] == "matched"
        assert d["gt_index"] == 0
        assert d["pred_index"] == 0
        assert d["score"] == pytest.approx(1.0)
        assert len(d["cells"]) == 2
        for cell in d["cells"]:
            assert cell["expected"] == cell["actual"]
            assert cell["score"] == pytest.approx(1.0)

    def test_build_record_details_wrong_cell(self) -> None:
        """A mismatched cell should score 0.0 while correct cells score 1.0."""
        gt_records = [{"Name": "Alice", "Age": "30"}]
        pred_records = [{"Name": "Alice", "Age": "99"}]
        col_mapping = {"Name": "Name", "Age": "Age"}
        matches = [(0, 0, 0.5)]

        details = build_record_details(gt_records, pred_records, col_mapping, matches)

        cells_by_col = {c["column"]: c for c in details[0]["cells"]}
        assert cells_by_col["Name"]["score"] == pytest.approx(1.0)
        assert cells_by_col["Age"]["expected"] == "30"
        assert cells_by_col["Age"]["actual"] == "99"
        assert cells_by_col["Age"]["score"] == pytest.approx(0.0)

    def test_build_record_details_unmatched_gt(self) -> None:
        """GT records with no pred match appear as unmatched_gt."""
        gt_records = [{"X": "1"}, {"X": "2"}]
        pred_records = [{"X": "1"}]
        col_mapping = {"X": "X"}
        matches = [(0, 0, 1.0)]

        details = build_record_details(gt_records, pred_records, col_mapping, matches)

        unmatched = [d for d in details if d["type"] == "unmatched_gt"]
        assert len(unmatched) == 1
        assert unmatched[0]["gt_index"] == 1
        assert unmatched[0]["pred_index"] is None
        assert unmatched[0]["score"] == 0.0

    def test_build_record_details_unmatched_pred(self) -> None:
        """Extra pred records appear as unmatched_pred."""
        gt_records = [{"X": "1"}]
        pred_records = [{"X": "1"}, {"X": "extra"}]
        col_mapping = {"X": "X"}
        matches = [(0, 0, 1.0)]

        details = build_record_details(gt_records, pred_records, col_mapping, matches)

        unmatched = [d for d in details if d["type"] == "unmatched_pred"]
        assert len(unmatched) == 1
        assert unmatched[0]["gt_index"] is None
        assert unmatched[0]["pred_index"] == 1

    def test_build_record_details_zero_columns(self) -> None:
        """Cells in zero_columns should report score 0.0, not the raw cell_score."""
        gt_records = [{"Name": "Alice", "Age": "30"}]
        pred_records = [{"Name": "Alice", "Age": "30"}]
        col_mapping = {"Name": "Name", "Age": "Age"}
        matches = [(0, 0, 0.5)]  # pair_score reflects the zero-column penalty

        details = build_record_details(
            gt_records,
            pred_records,
            col_mapping,
            matches,
            zero_columns={"Age"},
        )

        cells_by_col = {c["column"]: c for c in details[0]["cells"]}
        # Name is not penalized — values match so score should be 1.0
        assert cells_by_col["Name"]["score"] == pytest.approx(1.0)
        # Age is in zero_columns — detail score must be 0.0 even though values match
        assert cells_by_col["Age"]["score"] == pytest.approx(0.0)

    def test_record_details_in_compute_output(self, metric: TableRecordMatchMetric) -> None:
        """record_details should appear in per_table_details from compute()."""
        gt = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        pred = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "99"]])
        result = metric.compute(expected=gt, actual=pred)

        per_table = result[0].metadata["per_table_details"]
        assert len(per_table) == 1
        assert "record_details" in per_table[0]
        record_details = per_table[0]["record_details"]
        assert len(record_details) == 2  # two matched records
        for rd in record_details:
            assert rd["type"] == "matched"
            assert "cells" in rd
            assert len(rd["cells"]) == 2  # Name and Age columns


# ---- Incomplete table (no closing </table>) should still score > 0 ----
def test_incomplete_predicted_table_scores_nonzero(metric: TableRecordMatchMetric) -> None:
    """A predicted table missing </table> should still be scored, not silently dropped."""
    expected = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
    # Identical content but missing closing tag
    actual = (
        "<table><thead><tr><th>Name</th><th>Age</th></tr></thead>"
        "<tbody><tr><td>Alice</td><td>30</td></tr>"
        "<tr><td>Bob</td><td>25</td></tr></tbody>"
    )
    result = metric.compute(expected=expected, actual=actual)
    assert len(result) >= 1
    assert result[0].value > 0.0


# ---------------------------------------------------------------------------
# Unmatched column penalty
# ---------------------------------------------------------------------------


class TestBidirectionalUnmatchedColumnPenalty:
    """Unmatched GT columns must penalize the score, not be silently ignored."""

    def test_record_similarity_penalizes_unmatched_columns(self) -> None:
        """_record_similarity (via match_records) must score 0 for unmatched GT columns."""
        gt_keys = ["Name", "Age", "City"]
        pred_keys = ["Name"]  # only 1 of 3 columns

        gt_records = [{"Name": "Alice", "Age": "30", "City": "NYC"}]
        pred_records = [{"Name": "Alice"}]

        col_mapping, _ = align_columns(gt_keys, pred_keys)
        # Only "Name" should map
        assert len(col_mapping) == 1

        _, score = match_records(gt_records, pred_records, col_mapping, gt_keys=gt_keys)
        # With 1/3 columns matched and that cell correct: ~0.33, NOT 1.0
        assert score < 0.5
        assert score == pytest.approx(1.0 / 3.0)

    def test_multilevel_header_title_stripping_aligns(self, metric: TableRecordMatchMetric) -> None:
        """Multi-level GT headers vs flat pred headers should align via title fallback.

        GT has hierarchical keys like 'Group A', 'Group B', 'Group C' because
        'Group' is kept as part of the multi-level header.  Pred strips 'Group'
        as a title row, producing keys 'A', 'B', 'C'.  The prefix-stripping
        fallback should detect that 'Group' was a stripped title in pred, use
        it to strip the prefix from GT keys, and align all columns.
        """
        gt = _html(
            "<tbody>"
            '<tr><th rowspan="2">Q</th><th colspan="3">Group</th></tr>'
            "<tr><th>A</th><th>B</th><th>C</th></tr>"
            "<tr><td>Q1</td><td>1</td><td>2</td><td>3</td></tr>"
            "<tr><td>Q2</td><td>4</td><td>5</td><td>6</td></tr>"
            "</tbody>"
        )
        # Pred: title row makes "Group" a title, so keys become just "A","B","C"
        pred = _html(
            "<tbody>"
            '<tr><th colspan="4">Group</th></tr>'
            "<tr><th>Q</th><th>A</th><th>B</th><th>C</th></tr>"
            "<tr><td>Q1</td><td>1</td><td>2</td><td>3</td></tr>"
            "<tr><td>Q2</td><td>4</td><td>5</td><td>6</td></tr>"
            "</tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        # Title-aware prefix stripping should align all columns → 1.0
        assert result[0].value == pytest.approx(1.0)

    def test_unmatched_columns_penalize_without_title(self, metric: TableRecordMatchMetric) -> None:
        """When column mismatch is NOT due to title stripping, penalty still applies."""
        gt = _simple_table(
            ["Name", "Age", "City"],
            [["Alice", "30", "NYC"], ["Bob", "25", "LA"]],
        )
        pred = _simple_table(
            ["Name", "X", "Y"],
            [["Alice", "30", "NYC"], ["Bob", "25", "LA"]],
        )
        result = metric.compute(expected=gt, actual=pred)
        # "Name" matches, but "Age"↔"X" and "City"↔"Y" are below threshold
        # so only 1/3 columns map → score < 1.0
        assert result[0].value < 1.0

    def test_extra_pred_columns_penalize(self, metric: TableRecordMatchMetric) -> None:
        """Extra pred columns that don't map to GT should penalize the score."""
        gt = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        pred = _simple_table(
            ["Name", "Age", "Extra1", "Extra2"],
            [["Alice", "30", "x", "y"], ["Bob", "25", "a", "b"]],
        )
        result = metric.compute(expected=gt, actual=pred)
        # 2 of 4 columns match → record score ~0.5, not 1.0
        assert result[0].value < 1.0
        assert result[0].value == pytest.approx(0.5)

    def test_partial_column_match_gives_credit(self, metric: TableRecordMatchMetric) -> None:
        """Matched columns should still contribute to the score."""
        gt = _simple_table(
            ["Name", "Age", "City", "Country"],
            [["Alice", "30", "NYC", "US"]],
        )
        # Pred matches 2 of 4 GT columns
        pred = _simple_table(
            ["Name", "City"],
            [["Alice", "NYC"]],
        )
        result = metric.compute(expected=gt, actual=pred)
        # 2/4 columns matched with correct values → 0.5
        assert result[0].value == pytest.approx(0.5)

    def test_all_columns_matched_still_scores_perfectly(self, metric: TableRecordMatchMetric) -> None:
        """When all columns align, score should still be 1.0 (no false penalty)."""
        table = _simple_table(
            ["Name", "Age", "City"],
            [["Alice", "30", "NYC"], ["Bob", "25", "LA"]],
        )
        result = metric.compute(expected=table, actual=table)
        assert result[0].value == pytest.approx(1.0)


# ===========================================================================
# Phase 1 v2: Header normalization & strict threshold
# ===========================================================================


class TestHeaderNormalization:
    """Phase 1 v2: col_headers are normalized; threshold is 0.9."""

    def test_header_whitespace_accent_normalized(self, metric: TableRecordMatchMetric) -> None:
        """Headers with whitespace/accent differences should match after normalization."""
        gt = _simple_table(["Résumé", "Naïve"], [["a", "b"]])
        pred = _simple_table(["Resume", "Naive"], [["a", "b"]])
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_threshold_rejects_fuzzy_below_90(self, metric: TableRecordMatchMetric) -> None:
        """Column headers below 0.9 similarity should not match."""
        gt = _simple_table(["Revenue", "Cost"], [["100", "50"]])
        pred = _simple_table(["Rev", "Cst"], [["100", "50"]])
        result = metric.compute(expected=gt, actual=pred)
        # "Revenue" vs "Rev" is ~55% similarity, should NOT match
        assert result[0].value < 1.0

    def test_empty_header_vs_nonempty_no_match(self) -> None:
        """Empty header vs non-empty header should produce 0.0 similarity."""
        mapping, _ = _align_columns_header_core(
            ["Name", "Age"],
            ["Name", "col_1"],
            pred_synthetic=frozenset({"col_1"}),
        )
        # "Age" (real) vs "col_1" (synthetic) should not match
        assert "Age" not in mapping or mapping.get("Age") != "col_1"


# ===========================================================================
# Phase 2 v2: Header block & title refinements
# ===========================================================================


class TestFlatBottomHeader:
    """Phase 2 v2: Flat-bottom header enforcement."""

    def test_bottom_row_non_th_excluded(self) -> None:
        """Bottom header row with non-<th> cells should be excluded from header block."""
        # Row 0 is all <th>, row 1 has mix of <th> and <td>
        html = _html("<thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody>")
        tables = parse_html_tables(html)
        keys, records, _ = table_to_records(tables[0])
        # Normal case: header row kept
        assert keys == ["A", "B"]
        assert len(records) == 1

    def test_single_spanning_header_not_stripped_as_title(self) -> None:
        """A single spanning header row should NOT be stripped as title."""
        html = _html(
            '<thead><tr><th colspan="2">Category</th></tr></thead><tbody><tr><td>A</td><td>B</td></tr></tbody>'
        )
        tables = parse_html_tables(html)
        header = extract_header_info(tables[0])
        # Should NOT be stripped — it's the only header row
        assert header.col_header_rows, "Single header row should not be stripped"
        assert not header.th_title_rows, "Only header row should not be detected as title"

    def test_mixed_th_td_bottom_row_excluded(self) -> None:
        """Bottom header row with <td> at a column that has <th> in an earlier row is excluded.

        Row 0: <th>A</th> <td>B</td>  — mixed, but first <th> row so included
        Row 1: <td>X</td> <th>Y</th>  — col 0 is <td>, even though col 0 had <th> at row 0

        Row 1 is excluded because (1, 0) is not a <th>-originated cell.
        """
        html = _html("<tr><th>A</th><td>B</td></tr><tr><td>X</td><th>Y</th></tr><tr><td>1</td><td>2</td></tr>")
        tables = parse_html_tables(html)
        header = extract_header_info(tables[0])
        # Row 1 has a <td> at col 0 — it should NOT be in the header block
        assert 1 not in header.col_header_rows
        # Row 0 may or may not be a header (it has mixed <th>/<td>),
        # but row 1's <td>X must not sneak in as a header cell
        assert "X" not in " ".join(header.keys)

    def test_colspan_th_row_kept_as_header(self) -> None:
        """A row where <th colspan> covers all columns should stay in the header block.

        Row 0: <th colspan="3">Group</th>  — expands to 3 grid cells, all from <th>
        Row 1: <th>A</th><th>B</th><th>C</th>  — all <th>

        The grid has non-empty values at all 3 columns of row 0, but they
        all originate from the same <th> element.  The partial-<th> guard
        must not reject this row.
        """
        html = _html(
            "<thead>"
            '<tr><th colspan="3">Group</th></tr>'
            "<tr><th>A</th><th>B</th><th>C</th></tr>"
            "</thead>"
            "<tbody><tr><td>1</td><td>2</td><td>3</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        header = extract_header_info(tables[0])
        assert 0 in header.col_header_rows, "colspan'd <th> row should be in header block"
        assert 1 in header.col_header_rows, "leaf <th> row should be in header block"

    def test_partial_th_row_with_empty_td_is_a_section_row(self) -> None:
        """A partial-<th> row in a table with a <th> row-header column is data.

        Row 0: <th></th><th>Q1</th><th>Q2</th>  — all <th>, header
        Row 1: <th>Revenue</th><td></td><td></td>  — <td> cells are empty
        Row 2: <th>Cost</th><td>50</td><td>80</td>  — a data row

        Row 2 proves column 0 carries <th> *row* headers, so row 1's <th> is
        a row header too and its empty <td> cells make it a section label, not
        another level of column header. This is the balance-sheet shape — a
        prediction spelling row labels as <th> used to fold "Revenue" into the
        label column's key and unmatch that column against a ground truth that
        spells the same rows with <td>.
        """
        html = _html(
            "<thead><tr><th></th><th>Q1</th><th>Q2</th></tr></thead>"
            "<tbody>"
            "<tr><th>Revenue</th><td></td><td></td></tr>"
            "<tr><th>Cost</th><td>50</td><td>80</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        header = extract_header_info(tables[0])
        assert header.col_header_rows == {0}, "Section row must not extend the header block"
        assert "Revenue" not in header.keys

    def test_partial_th_row_with_empty_td_section_row_nonempty_corner(self) -> None:
        """Same as above but row 0 has a non-empty corner cell instead of empty.

        Row 0: <th>Q0</th><th>Q1</th><th>Q2</th>  — all <th>, header
        Row 1: <th>Revenue</th><td></td><td></td>  — a section label

        The keys stay the row-0 header; "Revenue" does not join "Q0".
        """
        html = _html(
            "<thead><tr><th>Q0</th><th>Q1</th><th>Q2</th></tr></thead>"
            "<tbody>"
            "<tr><th>Revenue</th><td></td><td></td></tr>"
            "<tr><th>Cost</th><td>50</td><td>80</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        header = extract_header_info(tables[0])
        assert header.col_header_rows == {0}, "Section row must not extend the header block"
        assert header.keys == ["Q0", "Q1", "Q2"]

    def test_dual_axis_data_row_excluded_from_headers(self) -> None:
        """A dual-axis row (<th> row header + <td> data) must NOT be in the header block.

        Row 0: <th></th><th>Q1</th><th>Q2</th>  — all <th>, header
        Row 1: <th>Revenue</th><td>100</td><td>200</td>  — <th> is row label, <td> is data

        Row 1 has non-empty <td> cells that are NOT from <th> elements,
        so it's a data row despite containing a <th>.
        """
        html = _html(
            "<thead><tr><th></th><th>Q1</th><th>Q2</th></tr></thead>"
            "<tbody>"
            "<tr><th>Revenue</th><td>100</td><td>200</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        header = extract_header_info(tables[0])
        assert 0 in header.col_header_rows, "Full-<th> row should be header"
        assert 1 not in header.col_header_rows, "Dual-axis data row should NOT be header"


class TestSectionHeaderPromotion:
    """Phase 2 v2: Section header promotion try-both logic."""

    def test_section_header_promoted_when_improves_alignment(self, metric: TableRecordMatchMetric) -> None:
        """Section header below main header should be promoted when it improves alignment."""
        # GT has the section value as part of header (2-level)
        gt = _html(
            "<thead>"
            "<tr><th>Item</th><th>Value</th></tr>"
            "</thead>"
            "<tbody>"
            '<tr><td colspan="2">Group A</td></tr>'
            "<tr><td>X</td><td>10</td></tr>"
            "</tbody>"
        )
        # Same structure — both sides have identical data
        pred = _html(
            "<thead>"
            "<tr><th>Item</th><th>Value</th></tr>"
            "</thead>"
            "<tbody>"
            '<tr><td colspan="2">Group A</td></tr>'
            "<tr><td>X</td><td>10</td></tr>"
            "</tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)


# ===========================================================================
# Phase 3 v2: Header recovery & demotion
# ===========================================================================


class TestHeaderRecovery:
    """Phase 3 v2: Recover pred headers from first data row."""

    def test_pred_without_th_first_row_matches_gt(self, metric: TableRecordMatchMetric) -> None:
        """Pred without <th> tags but first row matches GT headers → recovered."""
        gt = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        pred = _html(
            "<tr><td>Name</td><td>Age</td></tr><tr><td>Alice</td><td>30</td></tr><tr><td>Bob</td><td>25</td></tr>"
        )
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_pred_without_headers_no_match_no_recovery(self, metric: TableRecordMatchMetric) -> None:
        """Pred without headers, first row doesn't match GT → no recovery."""
        gt = _simple_table(["Name", "Age"], [["Alice", "30"]])
        pred = _html("<tr><td>Foo</td><td>Bar</td></tr><tr><td>Alice</td><td>30</td></tr>")
        result = metric.compute(expected=gt, actual=pred)
        # No recovery — columns don't align, score should be low
        assert result[0].value < 1.0


class TestHeaderDemotion:
    """Phase 3 v2: Demote pred headers when GT has no headers."""

    def test_gt_no_headers_pred_with_headers_demoted(self, metric: TableRecordMatchMetric) -> None:
        """Pred with <th> tags but GT has no headers → pred headers demoted to data."""
        gt = _html("<tr><td>Alice</td><td>30</td></tr><tr><td>Bob</td><td>25</td></tr>")
        pred = _simple_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        result = metric.compute(expected=gt, actual=pred)
        # After demotion, pred has 3 rows (header demoted to data) vs 2 GT rows
        # Score should reflect mismatch but not crash
        assert result[0].value < 1.0

    def test_gt_no_headers_pred_with_td_title_and_headers_demoted(self, metric: TableRecordMatchMetric) -> None:
        """Pred with <td> title + <th> headers, GT has no headers → title stripped, header demoted.

        GT: 3 data rows (no headers, no titles)
        Pred: 1 <td> title row + 1 <th> header row + 3 data rows

        The upstream strip stage physically removes the <td> title row,
        so it never reaches the demotion stage. The <th> header row is
        then demoted to data because GT has no real headers. Net: pred
        has 4 data rows (demoted header + 3 originally-data rows).
        """
        gt = _html(
            "<tr><td>Alice</td><td>30</td></tr><tr><td>Bob</td><td>25</td></tr><tr><td>Carol</td><td>35</td></tr>"
        )
        pred = _html(
            '<tr><td colspan="2">People</td></tr>'
            "<thead><tr><th>Name</th><th>Age</th></tr></thead>"
            "<tbody>"
            "<tr><td>Alice</td><td>30</td></tr>"
            "<tr><td>Bob</td><td>25</td></tr>"
            "<tr><td>Carol</td><td>35</td></tr>"
            "</tbody>"
        )
        result = metric.compute(expected=gt, actual=pred)
        # Title stripped + <th> row demoted: 1 demoted header + 3 data = 4 records.
        n_pred = result[0].metadata["per_table_details"][0].get("n_pred_records", 0)
        assert n_pred == 4, f"Expected 4 pred records after strip+demote, got {n_pred}"

    def test_gt_no_headers_pred_no_headers_positional(self, metric: TableRecordMatchMetric) -> None:
        """Both without headers → positional alignment via synthetic keys."""
        gt = _html("<tr><td>Alice</td><td>30</td></tr><tr><td>Bob</td><td>25</td></tr>")
        pred = _html("<tr><td>Alice</td><td>30</td></tr><tr><td>Bob</td><td>25</td></tr>")
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)


# ===========================================================================
# Phase 4 v2: Empty header penalty, hard zero gate, no content alignment
# ===========================================================================


class TestEmptyHeaderPenalty:
    """Phase 4 v2: Empty pred header for real GT header → column scores 0."""

    def test_empty_pred_header_zeros_column(self, metric: TableRecordMatchMetric) -> None:
        """GT column with real header matched to empty pred header → entire column scores 0."""
        gt = _simple_table(["Name", "Age"], [["Alice", "30"]])
        # Pred has same data but empty header for second column
        pred = _html("<thead><tr><th>Name</th><th></th></tr></thead><tbody><tr><td>Alice</td><td>30</td></tr></tbody>")
        result = metric.compute(expected=gt, actual=pred)
        # Union denominator: "Name" matches (1.0), GT "Age" unmatched, pred ""
        # unmatched → union size = 3, score = 1/3.
        assert result[0].value == pytest.approx(1 / 3)

    def test_no_column_matches_returns_zero(self, metric: TableRecordMatchMetric) -> None:
        """No column matches → score exactly 0.0."""
        gt = _simple_table(["Alpha", "Beta"], [["1", "2"]])
        pred = _simple_table(["Gamma", "Delta"], [["1", "2"]])
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(0.0)

    def test_headerless_vs_headerless_positional(self, metric: TableRecordMatchMetric) -> None:
        """Headerless-vs-headerless tables align positionally via synthetic keys."""
        gt = _html("<tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr>")
        pred = _html("<tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr>")
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_all_columns_zeroed_scores_zero(self, metric: TableRecordMatchMetric) -> None:
        """When ALL GT columns have real headers but ALL pred headers are empty, score is 0."""
        gt = _simple_table(["Name", "Age"], [["Alice", "30"]])
        pred = _html("<thead><tr><th></th><th></th></tr></thead><tbody><tr><td>Alice</td><td>30</td></tr></tbody>")
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(0.0)

    def test_no_column_matches_detail_has_reason(self, metric: TableRecordMatchMetric) -> None:
        """Hard gate returns detail dict with 'reason': 'no column matches'."""
        gt = _simple_table(["Alpha", "Beta"], [["1", "2"]])
        pred = _simple_table(["Gamma", "Delta"], [["1", "2"]])
        result = metric.compute(expected=gt, actual=pred)
        detail = result[0].metadata["per_table_details"][0]
        assert detail["reason"] == "no column matches"

    def test_no_column_matches_appears_in_detail_strings(self, metric: TableRecordMatchMetric) -> None:
        """Hard gate (no column matches) should show reason, columns, and record count."""
        gt = _simple_table(["Alpha", "Beta"], [["1", "2"]])
        pred = _simple_table(["Gamma", "Delta"], [["1", "2"]])
        result = metric.compute(expected=gt, actual=pred)
        details = result[0].details or []
        matching = [line for line in details if "no column matches" in line]
        assert matching
        assert "alpha" in matching[0].lower()
        assert "beta" in matching[0].lower()
        assert "1 record" in matching[0]

    def test_unmatched_table_appears_in_detail_strings(self, metric: TableRecordMatchMetric) -> None:
        """A GT table with no pred pairing should show columns and record count."""
        t1 = _simple_table(["A"], [["1"]])
        t2 = _simple_table(["B"], [["2"]])
        t3 = _simple_table(["C"], [["3"]])
        result = metric.compute(expected=t1 + t2 + t3, actual=t1 + t2)
        details = result[0].details or []
        matching = [line for line in details if "unmatched" in line]
        assert matching
        # Should include the GT table's column name and record count
        assert "1 record" in matching[0]


# ===========================================================================
# Non-happy-path unit tests for helpers
# ===========================================================================


class TestSectionHeaderPromotionEdgeCases:
    """Edge cases for _try_section_header_promotion."""

    def test_no_header_rows_returns_none(self) -> None:
        """No header rows → None (nothing to promote below)."""
        header = HeaderInfo(
            keys=["col_0", "col_1"],
            synthetic_keys=frozenset({"col_0", "col_1"}),
            col_header_rows=set(),
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        html = _html("<tr><td>A</td><td>B</td></tr>")
        table = parse_html_tables(html)[0]
        assert _try_section_header_promotion(table, header) is None

    def test_header_block_is_last_row_returns_none(self) -> None:
        """Header block is the last row → no candidate row below."""
        html = _html("<thead><tr><th>A</th><th>B</th></tr></thead>")
        table = parse_html_tables(html)[0]
        header = extract_header_info(table)
        assert _try_section_header_promotion(table, header) is None

    def test_candidate_row_not_section_like_returns_none(self) -> None:
        """Candidate row has multiple distinct values → not a section header."""
        html = _html(
            "<thead><tr><th>A</th><th>B</th></tr></thead>"
            "<tbody>"
            "<tr><td>X</td><td>Y</td></tr>"
            "<tr><td>1</td><td>2</td></tr>"
            "</tbody>"
        )
        table = parse_html_tables(html)[0]
        header = extract_header_info(table)
        assert _try_section_header_promotion(table, header) is None


class TestHeaderRecoveryEdgeCases:
    """Edge cases for _try_recover_pred_header."""

    def test_gt_all_synthetic_no_recovery(self) -> None:
        """GT has all synthetic keys → no recovery attempted."""
        gt_header = HeaderInfo(
            keys=["col_0", "col_1"],
            synthetic_keys=frozenset({"col_0", "col_1"}),
            col_header_rows=set(),
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        pred_header = HeaderInfo(
            keys=["col_0", "col_1"],
            synthetic_keys=frozenset({"col_0", "col_1"}),
            col_header_rows=set(),
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        html = _html("<tr><td>Name</td><td>Age</td></tr>")
        pred_table = parse_html_tables(html)[0]
        assert _try_recover_pred_header(gt_header, pred_table, pred_header) is None

    def test_pred_already_has_real_headers_no_recovery(self) -> None:
        """Pred already has real headers → no recovery attempted."""
        gt_header = HeaderInfo(
            keys=["Name", "Age"],
            synthetic_keys=frozenset(),
            col_header_rows={0},
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        pred_header = HeaderInfo(
            keys=["Name", "Age"],
            synthetic_keys=frozenset(),
            col_header_rows={0},
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        html = _simple_table(["Name", "Age"], [["Alice", "30"]])
        pred_table = parse_html_tables(html)[0]
        assert _try_recover_pred_header(gt_header, pred_table, pred_header) is None

    def test_pred_no_data_rows_no_recovery(self) -> None:
        """Pred has no data rows → no recovery (nothing to promote)."""
        gt_header = HeaderInfo(
            keys=["Name", "Age"],
            synthetic_keys=frozenset(),
            col_header_rows={0},
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        html = _html("<thead><tr><th>X</th><th>Y</th></tr></thead>")
        pred_table = parse_html_tables(html)[0]
        pred_header = extract_header_info(pred_table)
        # All rows are header rows — no data rows to recover from
        assert _try_recover_pred_header(gt_header, pred_table, pred_header) is None


class TestHeaderDemotionEdgeCases:
    """Edge cases for _demote_pred_headers."""

    def test_gt_has_real_headers_no_demotion(self) -> None:
        """GT has real headers → no demotion needed."""
        gt_header = HeaderInfo(
            keys=["Name", "Age"],
            synthetic_keys=frozenset(),
            col_header_rows={0},
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        pred_header = HeaderInfo(
            keys=["Name", "Age"],
            synthetic_keys=frozenset(),
            col_header_rows={0},
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        html = _simple_table(["Name", "Age"], [["Alice", "30"]])
        pred_table = parse_html_tables(html)[0]
        result = _demote_pred_headers(gt_header, pred_table, pred_header)
        assert result is pred_header  # Unchanged

    def test_pred_all_synthetic_no_demotion(self) -> None:
        """Pred already has all synthetic keys → nothing to demote."""
        gt_header = HeaderInfo(
            keys=["col_0", "col_1"],
            synthetic_keys=frozenset({"col_0", "col_1"}),
            col_header_rows=set(),
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        pred_header = HeaderInfo(
            keys=["col_0", "col_1"],
            synthetic_keys=frozenset({"col_0", "col_1"}),
            col_header_rows=set(),
            th_title_rows=set(),
            td_title_rows=set(),
            stripped_titles=set(),
        )
        html = _html("<tr><td>A</td><td>B</td></tr>")
        pred_table = parse_html_tables(html)[0]
        result = _demote_pred_headers(gt_header, pred_table, pred_header)
        assert result is pred_header  # Unchanged


class TestAlignColumnsHeaderCoreEdgeCases:
    """Edge cases for _align_columns_header_core empty-vs-nonempty guard."""

    def test_synthetic_gt_vs_real_pred_no_match(self) -> None:
        """Synthetic GT key vs real pred key → 0 similarity (no match)."""
        mapping, _ = _align_columns_header_core(["col_0"], ["Name"], gt_synthetic=frozenset({"col_0"}))
        assert not mapping

    def test_synthetic_pred_vs_real_gt_no_match(self) -> None:
        """Real GT key vs synthetic pred key → 0 similarity (no match)."""
        mapping, _ = _align_columns_header_core(["Name"], ["col_0"], pred_synthetic=frozenset({"col_0"}))
        assert not mapping

    def test_both_real_headers_match(self) -> None:
        """Both sides have real (non-synthetic) headers → normal fuzzy match."""
        mapping, score = _align_columns_header_core(["Name", "Age"], ["Name", "Age"])
        assert len(mapping) == 2
        assert mapping == {"Name": "Name", "Age": "Age"}
        assert score == pytest.approx(1.0)

    def test_dash_separator_in_collapsed_hierarchy_matches(self) -> None:
        """'A B' (GT, joined parent/child) matches 'A - B' (pred uses dash join)."""
        mapping, score = _align_columns_header_core(["Group A B"], ["Group A - B"])
        assert mapping == {"Group A B": "Group A - B"}
        assert score >= 0.9

    def test_short_dash_separator_in_collapsed_hierarchy_matches(self) -> None:
        """Short keys: 'A B' vs 'A - B'. Without dash normalization fuzz=75%."""
        mapping, score = _align_columns_header_core(["A B"], ["A - B"])
        assert mapping == {"A B": "A - B"}
        assert score >= 0.9

    def test_en_dash_and_em_dash_separator_matches(self) -> None:
        """En-dash and em-dash variants are also accepted as join separators."""
        mapping_en, _ = _align_columns_header_core(["Group A B"], ["Group A \u2013 B"])
        assert mapping_en == {"Group A B": "Group A \u2013 B"}
        mapping_em, _ = _align_columns_header_core(["Group A B"], ["Group A \u2014 B"])
        assert mapping_em == {"Group A B": "Group A \u2014 B"}

    def test_dash_in_data_value_not_collapsed(self) -> None:
        """Header normalization doesn't break unrelated dashes (sanity check)."""
        # Pre-fix sanity: completely different keys still don't match.
        mapping, _ = _align_columns_header_core(["Range"], ["Date"])
        assert not mapping

    def test_both_synthetic_keys_match(self) -> None:
        """Both sides synthetic → positional alignment via matching key names.

        When both sides are synthetic, both gk_empty and pk_empty are True,
        so the empty-vs-nonempty guard doesn't fire. fuzz.ratio("col_0",
        "col_0") gives 1.0, producing a positional match. This is intended:
        headerless tables should align by column position, not by content.
        """
        syn = frozenset({"col_0", "col_1"})
        mapping, score = _align_columns_header_core(
            ["col_0", "col_1"], ["col_0", "col_1"], gt_synthetic=syn, pred_synthetic=syn
        )
        assert len(mapping) == 2
        assert score == pytest.approx(1.0)

    def test_real_key_named_col_not_treated_as_synthetic(self) -> None:
        """A real header literally named 'col_0' should NOT be treated as synthetic.

        This is the key reason for using a set rather than string-prefix matching.
        """
        mapping, score = _align_columns_header_core(
            ["col_0"],
            ["col_0"],
            gt_synthetic=frozenset(),  # NOT synthetic despite the name
            pred_synthetic=frozenset(),
        )
        assert len(mapping) == 1
        assert score == pytest.approx(1.0)

    def test_pred_synthetic_col_0_vs_gt_real_col_0_no_match(self) -> None:
        """Pred has synthetic 'col_0', GT has real header named 'col_0' → no match.

        The strings are identical, but the synthetic-vs-real asymmetry should
        block the match.  With string-prefix matching this would incorrectly
        match (both start with 'col_').
        """
        mapping, _ = _align_columns_header_core(
            ["col_0"],
            ["col_0"],
            gt_synthetic=frozenset(),  # real
            pred_synthetic=frozenset({"col_0"}),  # synthetic
        )
        assert not mapping

    def test_gt_synthetic_col_0_vs_pred_real_col_0_no_match(self) -> None:
        """Mirror: GT synthetic 'col_0' vs pred real 'col_0' → no match."""
        mapping, _ = _align_columns_header_core(
            ["col_0"],
            ["col_0"],
            gt_synthetic=frozenset({"col_0"}),  # synthetic
            pred_synthetic=frozenset(),  # real
        )
        assert not mapping

    def test_mixed_real_and_synthetic_partial_match(self) -> None:
        """Mix of real and synthetic keys — only real-to-real pairs should match."""
        mapping, _ = _align_columns_header_core(
            ["Name", "col_1"],
            ["Name", "col_1"],
            gt_synthetic=frozenset({"col_1"}),
            pred_synthetic=frozenset({"col_1"}),
        )
        assert mapping == {"Name": "Name", "col_1": "col_1"}

    def test_mixed_real_synthetic_cross_no_match(self) -> None:
        """Real key on one side, synthetic on the other — even with same name, no match."""
        mapping, _ = _align_columns_header_core(
            ["Name", "col_1"],
            ["Name", "col_1"],
            gt_synthetic=frozenset(),  # both real on GT side
            pred_synthetic=frozenset({"col_1"}),  # col_1 synthetic on pred side
        )
        # "Name" ↔ "Name" matches (both real)
        # "col_1" GT (real) vs "col_1" pred (synthetic) → blocked
        assert len(mapping) == 1
        assert "Name" in mapping


class TestTitleStripGuardEdgeCases:
    """Edge cases for _detect_th_title_rows: at most one title stripped from top."""

    def test_two_title_rows_both_spanning_strips_only_top(self) -> None:
        """Two header rows that are both spanning titles → only topmost stripped."""
        html = _html(
            "<thead>"
            '<tr><th colspan="2">Title A</th></tr>'
            '<tr><th colspan="2">Title B</th></tr>'
            "</thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        header = extract_header_info(tables[0])
        # Only the topmost title row is stripped; row 1 remains as a header
        assert header.th_title_rows == {0}
        assert 1 in header.col_header_rows

    def test_three_deep_titles_one_real_header_strips_top_only(self) -> None:
        """Three header rows: two spanning titles + one real → only topmost title stripped.

        Row 0: <th colspan="3">Title A</th>  — spanning title (stripped)
        Row 1: <th colspan="3">Title B</th>  — spanning title (kept — max 1 strip)
        Row 2: <th>X</th><th>Y</th><th>Z</th>  — real leaf header

        Only row 0 is stripped. Row 1's "Title B" becomes part of keys.
        """
        html = _html(
            "<thead>"
            '<tr><th colspan="3">Title A</th></tr>'
            '<tr><th colspan="3">Title B</th></tr>'
            "<tr><th>X</th><th>Y</th><th>Z</th></tr>"
            "</thead>"
            "<tbody><tr><td>1</td><td>2</td><td>3</td></tr></tbody>"
        )
        tables = parse_html_tables(html)
        header = extract_header_info(tables[0])
        assert header.th_title_rows == {0}
        assert header.col_header_rows == {0, 1, 2}
        # Row 1 "Title B" is now part of keys (joined with row 2 leaf headers)
        assert any("Title B" in k for k in header.keys) or any("title b" in k.lower() for k in header.keys)


class TestBottomOfBlockTitleStrip:
    """Title rows at the bottom of the header block should be stripped."""

    def test_bottom_units_subheader_stripped(self) -> None:
        """A spanning <th> units row at the bottom of the header block is stripped.

        This reproduces a real-world pattern (e.g. AMZN 10-K tables) where GT
        has a <td colspan> units row (not a header) but pred has it as <th colspan>
        (a header).  Without stripping, the units text pollutes every column key
        and column alignment fails completely.

        GT structure:
          Row 0: <th rowspan="2"></th><th colspan="5">Year Ended December 31,</th>  — title
          Row 1: <th>2005</th>...<th>2001</th>  — real leaf header
          Row 2: <td colspan="5">(in millions)</td>  — data row (units)

        Pred structure (same content, but units row is <th>):
          Row 0: <th colspan="6">Year Ended December 31,</th>  — title
          Row 1: <th></th><th>2005</th>...<th>2001</th>  — real leaf header
          Row 2: <th colspan="6">(in millions)</th>  — title at bottom of block
        """
        gt_html = _html(
            "<thead>"
            '<tr><th rowspan="2"></th><th colspan="5">Year Ended December 31,</th></tr>'
            "<tr><th>2005</th><th>2004</th><th>2003</th><th>2002</th><th>2001</th></tr>"
            "</thead>"
            "<tbody>"
            '<tr><td></td><td colspan="5">(in millions)</td></tr>'
            "<tr><td>Net sales</td><td>$8,490</td><td>$6,921</td><td>$5,264</td><td>$3,933</td><td>$3,122</td></tr>"
            "</tbody>"
        )
        pred_html = _html(
            "<thead>"
            '<tr><th colspan="6">Year Ended December 31,</th></tr>'
            "<tr><th> </th><th>2005</th><th>2004</th><th>2003</th><th>2002</th><th>2001</th></tr>"
            '<tr><th colspan="6">(in millions)</th></tr>'
            "</thead>"
            "<tbody>"
            "<tr><td>Net sales</td><td>$8,490</td><td>$6,921</td><td>$5,264</td><td>$3,933</td><td>$3,122</td></tr>"
            "</tbody>"
        )

        # Pred should strip both top title (row 0) and bottom title (row 2)
        pred_tables = parse_html_tables(pred_html)
        pred_header = extract_header_info(normalize_table(pred_tables[0]))
        assert 0 in pred_header.th_title_rows, "top title row should be stripped"
        assert 2 in pred_header.th_title_rows, "bottom units title row should be stripped"

        # Column alignment should succeed
        gt_tables = parse_html_tables(gt_html)
        gt_header = extract_header_info(normalize_table(gt_tables[0]))
        col_mapping, score = align_columns(
            gt_header.keys,
            pred_header.keys,
            gt_synthetic=gt_header.synthetic_keys,
            pred_synthetic=pred_header.synthetic_keys,
        )
        # All 6 columns should align
        assert len(col_mapping) == 6
        assert score > 0.9

        # End-to-end: metric should score > 0
        metric = TableRecordMatchMetric()
        results = metric.compute(gt_html, pred_html)
        assert results[0].value > 0.9


class TestMidTableSectionHeaderAsData:
    """Mid-table section header rows are data, not headers."""

    def test_mid_table_th_section_is_data_row(self) -> None:
        """A <th> row in the middle of the table body is a data record, not a header."""
        html = _html(
            "<thead><tr><th>Item</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>A</td><td>10</td></tr>"
            '<tr><th colspan="2">Section X</th></tr>'
            "<tr><td>B</td><td>20</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        header = extract_header_info(tables[0])
        # The section row (row 2 in grid) should NOT be in col_header_rows
        assert header.col_header_rows == {0}
        # It should appear as a data record
        records = build_records(tables[0], header)
        section_records = [r for r in records if "Section X" in r.values()]
        assert len(section_records) == 1


# ===========================================================================
# Phase 5: Side-by-Side Table Splitting Tests
# ===========================================================================


class TestResolveHeaderRowValues:
    """Tests for _resolve_header_row_values helper."""

    def test_single_header_row(self) -> None:
        html = _html(
            "<tr><th>A</th><th>B</th><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        row_values = _resolve_header_row_values(table, header)
        assert len(row_values) == 1
        assert row_values[0] == [_norm("A"), _norm("B"), _norm("A"), _norm("B")]

    def test_multi_level_header(self) -> None:
        html = _html(
            '<tr><th colspan="2">Group 1</th><th colspan="2">Group 2</th></tr>'
            "<tr><th>X</th><th>Y</th><th>X</th><th>Y</th></tr>"
            "<tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        row_values = _resolve_header_row_values(table, header)
        assert len(row_values) == 2
        # Row 0: colspan spans, so both cols get "Group 1" / "Group 2"
        assert row_values[0] == [_norm("Group 1"), _norm("Group 1"), _norm("Group 2"), _norm("Group 2")]
        assert row_values[1] == [_norm("X"), _norm("Y"), _norm("X"), _norm("Y")]


class TestDetectRepeatingHeaderPeriod:
    """Tests for _detect_period_candidates."""

    def test_simple_repeating_2col_pattern(self) -> None:
        """4 columns with headers [A, B, A, B] → period 2."""
        html = _html(
            "<tr><th>A</th><th>B</th><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        assert _detect_period_candidates(table, header) == [(2, 1)]

    def test_repeating_3col_pattern(self) -> None:
        """6 columns with headers [A, B, C, A, B, C] → period 3."""
        html = _html(
            "<tr><th>A</th><th>B</th><th>C</th><th>A</th><th>B</th><th>C</th></tr>"
            "<tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        assert _detect_period_candidates(table, header) == [(3, 1)]

    def test_returns_all_valid_periods(self) -> None:
        """12 columns repeating every 2 — exposes periods 2, 4, 6 (no GT filter)."""
        html = _html(
            "<tr><th>A</th><th>B</th><th>A</th><th>B</th>"
            "<th>A</th><th>B</th><th>A</th><th>B</th>"
            "<th>A</th><th>B</th><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td><td>3</td><td>4</td>"
            "<td>5</td><td>6</td><td>7</td><td>8</td>"
            "<td>9</td><td>10</td><td>11</td><td>12</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        candidates = _detect_period_candidates(table, header)
        periods = sorted(p for p, _ in candidates)
        assert periods == [2, 4, 6]

    def test_no_repeating_pattern_returns_empty(self) -> None:
        html = _html(
            "<tr><th>A</th><th>B</th><th>C</th><th>D</th></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        assert _detect_period_candidates(table, header) == []

    def test_varying_title_row_doesnt_block(self) -> None:
        """Group/title row varies but leaf row repeats → period still detected."""
        html = _html(
            '<tr><th colspan="2">Title A</th><th colspan="2">Title B</th></tr>'
            "<tr><th>X</th><th>Y</th><th>X</th><th>Y</th></tr>"
            "<tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        assert _detect_period_candidates(table, header) == [(2, 1)]

    def test_no_header_rows_returns_empty(self) -> None:
        html = _html(
            "<tr><td>A</td><td>B</td><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        assert _detect_period_candidates(table, header) == []

    def test_single_column_returns_empty(self) -> None:
        html = _html("<tr><th>A</th></tr><tr><td>1</td></tr>")
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        assert _detect_period_candidates(table, header) == []


class TestBuildSubTable:
    """Tests for _build_sub_table."""

    def test_basic_column_split(self) -> None:
        html = _html(
            "<tr><th>A</th><th>B</th><th>C</th><th>D</th></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        sub = _build_sub_table(tables[0], 0, 2)
        assert sub.data.shape == (2, 2)
        assert str(sub.data[1, 0]) == "1"
        assert str(sub.data[1, 1]) == "2"

    def test_header_cells_remapped(self) -> None:
        html = _html(
            "<tr><th>A</th><th>B</th><th>C</th><th>D</th></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        sub = _build_sub_table(tables[0], 2, 4)
        # Header cells should be remapped: (0,2) → (0,0), (0,3) → (0,1)
        assert (0, 0) in sub.header_cells
        assert (0, 1) in sub.header_cells

    def test_col_headers_remapped(self) -> None:
        html = _html(
            "<tr><th>A</th><th>B</th><th>C</th><th>D</th></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        sub = _build_sub_table(tables[0], 2, 4)
        # col_headers keys should be 0, 1 (remapped from 2, 3)
        assert 0 in sub.col_headers
        assert 1 in sub.col_headers


class TestEnumerateSplitOptions:
    """Tests for enumerate_split_options."""

    def test_includes_split_options(self) -> None:
        """6-col table with period 2 headers → no-split + period-2 (3 segs)."""
        html = _html(
            "<tr><th>A</th><th>B</th><th>A</th><th>B</th><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        options = enumerate_split_options(table)
        # No-split sentinel + at least one real split
        assert options[0].sub_tables is None
        assert options[0].n_segments == 1
        split_opts = [o for o in options if o.sub_tables is not None]
        assert any(o.n_segments == 3 and o.period == 2 for o in split_opts)
        for o in split_opts:
            if o.period == 2:
                assert o.sub_tables is not None
                assert all(st.data.shape[1] == 2 for st in o.sub_tables)

    def test_no_period_returns_only_no_split(self) -> None:
        html = _html(
            "<tr><th>A</th><th>B</th><th>C</th><th>D</th></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        options = enumerate_split_options(table)
        assert len(options) == 1
        assert options[0].sub_tables is None


# ===========================================================================
# Empty Row Filtering
# ===========================================================================


class TestEmptyRowFiltering:
    """Tests for filtering all-empty records from build_records()."""

    def test_is_all_empty_record_true(self) -> None:
        assert _is_all_empty_record({"a": "", "b": "", "c": ""})
        assert _is_all_empty_record({"a": "  ", "b": " "})

    def test_is_all_empty_record_false(self) -> None:
        assert not _is_all_empty_record({"a": "x", "b": ""})
        assert not _is_all_empty_record({"a": "-", "b": ""})
        assert not _is_all_empty_record({"a": "n/a", "b": ""})

    def test_empty_rows_filtered_from_records(self) -> None:
        """All-empty rows are excluded from build_records() output."""
        html = _html(
            "<tr><th>A</th><th>B</th></tr>"
            "<tr><td>1</td><td>2</td></tr>"
            "<tr><td></td><td></td></tr>"
            "<tr><td>3</td><td>4</td></tr>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        records = build_records(table, header)
        assert len(records) == 2
        assert records[0]["a"] == "1"
        assert records[1]["a"] == "3"

    def test_sentinel_rows_not_filtered(self) -> None:
        """Rows with sentinel values like '-' or 'n/a' are NOT filtered."""
        html = _html("<tr><th>A</th><th>B</th></tr><tr><td>-</td><td>n/a</td></tr>")
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        header = extract_header_info(table)
        records = build_records(table, header)
        assert len(records) == 1

    def test_empty_rows_dont_penalize_score(self, metric: TableRecordMatchMetric) -> None:
        """GT with trailing empty rows and pred without should still score 1.0."""
        gt = _html(
            "<tr><th>Name</th><th>Age</th></tr>"
            "<tr><td>Alice</td><td>30</td></tr>"
            "<tr><td></td><td></td></tr>"
            "<tr><td></td><td></td></tr>"
        )
        pred = _simple_table(["Name", "Age"], [["Alice", "30"]])
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_empty_rows_in_pred_dont_penalize(self, metric: TableRecordMatchMetric) -> None:
        """Pred with trailing empty rows should still score 1.0."""
        gt = _simple_table(["Name", "Age"], [["Alice", "30"]])
        pred = _html("<tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr><tr><td></td><td></td></tr>")
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    @pytest.fixture
    def metric(self) -> TableRecordMatchMetric:
        return TableRecordMatchMetric()


# ===========================================================================
# Subscript/Superscript Preservation
# ===========================================================================


class TestSubSupPreservation:
    """Tests for preserving subscript/superscript content in table normalization."""

    def test_normalize_sub_sup_html_tags(self) -> None:
        assert _normalize_sub_sup_for_table("H<sub>2</sub>O") == "H2O"
        assert _normalize_sub_sup_for_table("x<sup>2</sup>") == "x2"
        assert _normalize_sub_sup_for_table("10<sup>-3</sup>") == "10-3"

    def test_normalize_sub_sup_preserves_non_digit_content(self) -> None:
        """Non-digit content in sup/sub tags is preserved (e.g. ordinals)."""
        assert _normalize_sub_sup_for_table("1<sup>st</sup>") == "1st"
        assert _normalize_sub_sup_for_table("2<sup>nd</sup>") == "2nd"

    def test_normalize_sub_sup_unicode_digits(self) -> None:
        assert _normalize_sub_sup_for_table("H₂O") == "H2O"
        assert _normalize_sub_sup_for_table("CO₂") == "CO2"
        assert _normalize_sub_sup_for_table("x²") == "x2"
        assert _normalize_sub_sup_for_table("10⁻³") == "10-3"

    def test_normalize_sub_sup_unicode_letters(self) -> None:
        assert _normalize_sub_sup_for_table("aⁿ") == "an"
        assert _normalize_sub_sup_for_table("xᵢ") == "xi"

    def test_normalize_sub_sup_no_tags(self) -> None:
        """Plain text without sub/sup passes through unchanged."""
        assert _normalize_sub_sup_for_table("hello world") == "hello world"
        assert _normalize_sub_sup_for_table("123") == "123"

    def test_end_to_end_unicode_sub_matches_html_sub(self, metric: TableRecordMatchMetric) -> None:
        """GT with Unicode subscript and pred with HTML <sub> should score 1.0."""
        gt = _simple_table(["Formula"], [["H₂O"], ["CO₂"]])
        pred = _simple_table(["Formula"], [["H<sub>2</sub>O"], ["CO<sub>2</sub>"]])
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_end_to_end_unicode_superscript_symmetric(self, metric: TableRecordMatchMetric) -> None:
        """Both sides with identical Unicode superscripts score 1.0."""
        gt = _simple_table(["Value"], [["x²"], ["10⁻³"]])
        pred = _simple_table(["Value"], [["x²"], ["10⁻³"]])
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_end_to_end_plain_text_matches_unicode_sup(self, metric: TableRecordMatchMetric) -> None:
        """Plain ASCII text matches Unicode superscript after normalization."""
        gt = _simple_table(["Value"], [["x2"], ["a3"]])
        pred = _simple_table(["Value"], [["x²"], ["a³"]])
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_end_to_end_unicode_subscript_symmetric(self, metric: TableRecordMatchMetric) -> None:
        """Both sides with identical Unicode subscripts score 1.0."""
        gt = _simple_table(["Value"], [["H₂O"], ["CO₂"]])
        pred = _simple_table(["Value"], [["H₂O"], ["CO₂"]])
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_end_to_end_plain_text_matches_unicode_sub(self, metric: TableRecordMatchMetric) -> None:
        """Plain ASCII text matches Unicode subscript after normalization."""
        gt = _simple_table(["Value"], [["H2O"], ["CO2"]])
        pred = _simple_table(["Value"], [["H₂O"], ["CO₂"]])
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    @pytest.fixture
    def metric(self) -> TableRecordMatchMetric:
        return TableRecordMatchMetric()


class TestUnmatchedColumnPenaltyUnionDenominator:
    """Both unmatched GT columns and extra pred columns must penalize the score.

    With the union-size denominator, _record_similarity divides matched
    cell scores by ``n_matched + (n_gt - n_matched) + (n_pred - n_matched)``.
    A perfectly-matched single column shared between GT and pred should
    therefore drop below 1.0 whenever either side carries extra columns.
    """

    def test_extra_pred_column_penalizes(self) -> None:
        gt = _simple_table(["Name"], [["Alice"]])
        pred = _simple_table(["Name", "Age"], [["Alice", "30"]])
        metric = TableRecordMatchMetric()
        result = metric.compute(expected=gt, actual=pred)
        # 1 matched + 0 extra GT + 1 extra pred = 2 → 1/2
        assert result[0].value == pytest.approx(0.5)

    def test_extra_gt_column_penalizes(self) -> None:
        gt = _simple_table(["Name", "Age"], [["Alice", "30"]])
        pred = _simple_table(["Name"], [["Alice"]])
        metric = TableRecordMatchMetric()
        result = metric.compute(expected=gt, actual=pred)
        # 1 matched + 1 unmatched GT + 0 extra pred = 2 → 1/2
        assert result[0].value == pytest.approx(0.5)

    def test_extra_columns_on_both_sides_penalize(self) -> None:
        gt = _simple_table(["Name", "Age"], [["Alice", "30"]])
        pred = _simple_table(["Name", "City"], [["Alice", "NYC"]])
        metric = TableRecordMatchMetric()
        result = metric.compute(expected=gt, actual=pred)
        # 1 matched ("Name") + 1 unmatched GT ("Age") + 1 extra pred ("City")
        # = 3 → 1/3
        assert result[0].value == pytest.approx(1 / 3)


class TestEmptyColumnDropping:
    """normalize_table drops columns that are entirely empty.

    A column counts as "entirely empty" only when every data cell AND
    every header entry is the literal empty string after normalization.
    Sentinel values like "-" or "n/a" count as real content and do not
    qualify a column for removal.
    """

    def test_empty_gt_column_dropped(self) -> None:
        gt = _html("<thead><tr><th>Name</th><th></th></tr></thead><tbody><tr><td>Alice</td><td></td></tr></tbody>")
        pred = _simple_table(["Name"], [["Alice"]])
        metric = TableRecordMatchMetric()
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_empty_pred_column_dropped(self) -> None:
        gt = _simple_table(["Name"], [["Alice"]])
        pred = _html("<thead><tr><th>Name</th><th></th></tr></thead><tbody><tr><td>Alice</td><td></td></tr></tbody>")
        metric = TableRecordMatchMetric()
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_column_with_header_not_dropped(self) -> None:
        """Header text present but all data cells empty → column kept."""
        gt = _html("<thead><tr><th>Name</th><th>Age</th></tr></thead><tbody><tr><td>Alice</td><td></td></tr></tbody>")
        pred = _html("<thead><tr><th>Name</th><th>Age</th></tr></thead><tbody><tr><td>Alice</td><td></td></tr></tbody>")
        metric = TableRecordMatchMetric()
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_column_with_data_empty_header_not_dropped(self) -> None:
        """Data cells present but header empty → column kept."""
        gt = _html("<thead><tr><th>Name</th><th></th></tr></thead><tbody><tr><td>Alice</td><td>extra</td></tr></tbody>")
        pred = _html(
            "<thead><tr><th>Name</th><th></th></tr></thead><tbody><tr><td>Alice</td><td>extra</td></tr></tbody>"
        )
        metric = TableRecordMatchMetric()
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)

    def test_column_of_only_dash_sentinels_not_dropped(self) -> None:
        """A column whose data is only ``-``/``n/a`` sentinels must survive.

        ``_normalize_trm_cell_text`` doesn't touch sentinels and
        ``_col_is_empty`` only drops cells that normalize to the literal
        empty string — so a column where every data cell is ``-`` must
        still be present after ``normalize_table``. If a regression were
        to extend the empty check to include sentinels, this would
        silently erase real columns from the comparison.
        """
        html = _html(
            "<thead><tr><th>Name</th><th>Status</th></tr></thead>"
            "<tbody>"
            "<tr><td>Alice</td><td>-</td></tr>"
            "<tr><td>Bob</td><td>n/a</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        # Both columns must survive
        assert table.data.shape[1] == 2
        header = extract_header_info(table)
        assert "status" in header.keys  # normalized to lowercase

    def test_row_headers_and_header_cells_remap_after_dropping_middle_column(self) -> None:
        """Dropping an empty middle column must remap row_headers AND header_cells.

        Build a 3-col table where column index 1 is entirely empty, with
        a row-header ``<th>`` in col 0 and a header row in row 0. After
        ``normalize_table`` drops col 1, the surviving ``header_cells``
        and ``row_headers`` entries must reference the *new* column
        indices (0 and 1), never the old index 2.
        """
        html = _html(
            "<thead><tr><th>Metric</th><th></th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><th>Revenue</th><td></td><td>100</td></tr>"
            "<tr><th>Cost</th><td></td><td>50</td></tr>"
            "</tbody>"
        )
        tables = parse_html_tables(html)
        table = normalize_table(tables[0])
        # Middle column should be dropped
        assert table.data.shape[1] == 2
        # Remaining columns' header_cells must be remapped to {0, 1}.
        # (row 0 headers + each row-header <th> at col 0)
        col_indices = {c for (_r, c) in table.header_cells}
        assert col_indices <= {0, 1}, f"header_cells not remapped: {table.header_cells}"
        # row_headers entries must also be remapped to col 0
        for _row, entries in table.row_headers.items():
            for c, _text in entries:
                assert c in {0, 1}, f"row_headers col index not remapped: {c}"


class TestFootnoteParenPreservation:
    """Footnote markers like (1) inside <sup> must not lose their parens.

    Regression: ``_sup_sub_to_unicode`` in table_parsing maps each char inside
    <sup> through ``_ASCII_TO_SUPERSCRIPT.get(c, "")``, which silently drops
    any non-digit (including parens). So ``<sup>(2)</sup>`` becomes ``²`` and
    later normalizes to ``2``, gluing onto the preceding number. A bare-paren
    prediction ``(2)`` survives normalization unchanged, so two semantically
    equivalent representations score 0 against each other.
    """

    def test_sup_wrapped_paren_matches_bare_paren(self) -> None:
        gt = (
            "<table><tbody>"
            "<tr><th>Owner</th><th>Shares</th></tr>"
            "<tr><td>Vanguard</td><td>1,261,261,357<sup>(2)</sup></td></tr>"
            "</tbody></table>"
        )
        pred = (
            "<table><tbody>"
            "<tr><th>Owner</th><th>Shares</th></tr>"
            "<tr><td>Vanguard</td><td>1,261,261,357(2)</td></tr>"
            "</tbody></table>"
        )
        metric = TableRecordMatchMetric()
        result = metric.compute(expected=gt, actual=pred)
        assert result[0].value == pytest.approx(1.0)
