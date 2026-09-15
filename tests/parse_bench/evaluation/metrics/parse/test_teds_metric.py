"""Tests for the TEDS (Tree Edit Distance based Similarity) metric.

Includes tests for:
- Low-level TEDS.evaluate() with all three variants (content, struct, struct_bool)
- High-level TEDSMetric.compute() with variant selection
- Identical tables score 1.0 across all variants
- Content changes affect content score but not struct scores
- Empty-cell mismatches affect struct_bool but not struct
- Structural differences (extra rows/cols, colspan) affect all variants
- Edge cases: empty strings, no tables, malformed HTML
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from parse_bench.evaluation.metrics.parse.teds_metric import (
    ALL_TEDS_VARIANTS,
    TEDS,
    TEDS_CONTENT,
    TEDS_STRUCT,
    TEDS_STRUCT_BOOL,
    TEDSMetric,
)

# =============================================================================
# Test tables
# =============================================================================

# Simple 2x2 table
SIMPLE_2X2 = """<table>
<tr><td>A</td><td>B</td></tr>
<tr><td>C</td><td>D</td></tr>
</table>"""

# Same structure, completely different content
SIMPLE_2X2_DIFF_CONTENT = """<table>
<tr><td>X</td><td>Y</td></tr>
<tr><td>Z</td><td>W</td></tr>
</table>"""

# Same structure, one cell empty
SIMPLE_2X2_ONE_EMPTY = """<table>
<tr><td>A</td><td>B</td></tr>
<tr><td>C</td><td></td></tr>
</table>"""

# Same structure, all cells empty
SIMPLE_2X2_ALL_EMPTY = """<table>
<tr><td></td><td></td></tr>
<tr><td></td><td></td></tr>
</table>"""

# 3x3 table (different dimensions)
SIMPLE_3X3 = """<table>
<tr><td>A</td><td>B</td><td>C</td></tr>
<tr><td>D</td><td>E</td><td>F</td></tr>
<tr><td>G</td><td>H</td><td>I</td></tr>
</table>"""

# Table with colspan
TABLE_WITH_COLSPAN = """<table>
<tr><td colspan="2">Header</td></tr>
<tr><td>A</td><td>B</td></tr>
</table>"""

# Same content without colspan (different structure)
TABLE_NO_COLSPAN = """<table>
<tr><td>Header</td><td>Header</td></tr>
<tr><td>A</td><td>B</td></tr>
</table>"""

# Table with headers (th tags)
TABLE_WITH_HEADERS = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Alpha</td><td>100</td></tr>
</table>"""

# Same content using td instead of th
TABLE_WITHOUT_HEADERS = """<table>
<tr><td>Name</td><td>Value</td></tr>
<tr><td>Alpha</td><td>100</td></tr>
</table>"""

# Table with extra row
TABLE_3_ROWS = """<table>
<tr><td>A</td><td>B</td></tr>
<tr><td>C</td><td>D</td></tr>
<tr><td>E</td><td>F</td></tr>
</table>"""


# =============================================================================
# Tests for TEDS.evaluate() — low-level, all variants
# =============================================================================


class TestTEDSEvaluateIdentical:
    """Identical tables should score 1.0 across all variants."""

    def test_identical_simple(self):
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, gt_n, pred_n = teds.evaluate(SIMPLE_2X2, SIMPLE_2X2)
        assert scores[TEDS_CONTENT] == 1.0
        assert scores[TEDS_STRUCT] == 1.0
        assert scores[TEDS_STRUCT_BOOL] == 1.0
        assert gt_n == pred_n

    def test_identical_with_colspan(self):
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(TABLE_WITH_COLSPAN, TABLE_WITH_COLSPAN)
        assert scores[TEDS_CONTENT] == 1.0
        assert scores[TEDS_STRUCT] == 1.0
        assert scores[TEDS_STRUCT_BOOL] == 1.0

    def test_identical_all_empty(self):
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2_ALL_EMPTY, SIMPLE_2X2_ALL_EMPTY)
        assert scores[TEDS_CONTENT] == 1.0
        assert scores[TEDS_STRUCT] == 1.0
        assert scores[TEDS_STRUCT_BOOL] == 1.0


class TestTEDSEvaluateContentDifferences:
    """Content differences should only affect TEDS content score."""

    def test_different_content_same_structure(self):
        """Completely different text: struct variants should still be 1.0."""
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2, SIMPLE_2X2_DIFF_CONTENT)
        assert scores[TEDS_CONTENT] < 1.0
        assert scores[TEDS_STRUCT] == 1.0
        assert scores[TEDS_STRUCT_BOOL] == 1.0

    def test_one_cell_empty_vs_filled(self):
        """One cell empty in pred, filled in GT:
        - content: penalized (Levenshtein)
        - struct: not penalized (ignores content)
        - struct_bool: penalized (empty vs non-empty mismatch)
        """
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2, SIMPLE_2X2_ONE_EMPTY)
        assert scores[TEDS_CONTENT] < 1.0
        assert scores[TEDS_STRUCT] == 1.0
        assert scores[TEDS_STRUCT_BOOL] < 1.0

    def test_all_empty_vs_filled(self):
        """All cells empty vs all filled:
        - content: heavily penalized
        - struct: still 1.0
        - struct_bool: penalized (all mismatches)
        """
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2, SIMPLE_2X2_ALL_EMPTY)
        assert scores[TEDS_CONTENT] < 1.0
        assert scores[TEDS_STRUCT] == 1.0
        assert scores[TEDS_STRUCT_BOOL] < 1.0

    def test_both_empty_tables(self):
        """Both tables have empty cells — no content mismatch."""
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2_ALL_EMPTY, SIMPLE_2X2_ALL_EMPTY)
        assert scores[TEDS_CONTENT] == 1.0
        assert scores[TEDS_STRUCT] == 1.0
        assert scores[TEDS_STRUCT_BOOL] == 1.0


class TestTEDSEvaluateStructuralDifferences:
    """Structural differences should penalize all variants."""

    def test_different_dimensions(self):
        """2x2 vs 3x3: different number of rows/cols penalizes all variants."""
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2, SIMPLE_3X3)
        assert scores[TEDS_CONTENT] < 1.0
        assert scores[TEDS_STRUCT] < 1.0
        assert scores[TEDS_STRUCT_BOOL] < 1.0

    def test_extra_row(self):
        """2x2 vs 3x2: extra row penalizes all variants."""
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2, TABLE_3_ROWS)
        assert scores[TEDS_CONTENT] < 1.0
        assert scores[TEDS_STRUCT] < 1.0
        assert scores[TEDS_STRUCT_BOOL] < 1.0

    def test_colspan_mismatch(self):
        """Table with colspan vs same content without — structural difference."""
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(TABLE_WITH_COLSPAN, TABLE_NO_COLSPAN)
        assert scores[TEDS_CONTENT] < 1.0
        assert scores[TEDS_STRUCT] < 1.0
        assert scores[TEDS_STRUCT_BOOL] < 1.0

    def test_th_vs_td(self):
        """th tags vs td tags: tag mismatch penalizes all variants."""
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(TABLE_WITH_HEADERS, TABLE_WITHOUT_HEADERS)
        assert scores[TEDS_CONTENT] < 1.0
        assert scores[TEDS_STRUCT] < 1.0
        assert scores[TEDS_STRUCT_BOOL] < 1.0


class TestTEDSEvaluateScoreOrdering:
    """Struct scores should be >= content scores when structure is the same."""

    def test_struct_gte_content_for_content_diff(self):
        """With only content differences, struct >= content."""
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2, SIMPLE_2X2_DIFF_CONTENT)
        assert scores[TEDS_STRUCT] >= scores[TEDS_CONTENT]
        assert scores[TEDS_STRUCT_BOOL] >= scores[TEDS_CONTENT]

    def test_struct_bool_between_struct_and_content(self):
        """With empty cell mismatch, struct_bool should be between struct and content."""
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2, SIMPLE_2X2_ONE_EMPTY)
        assert scores[TEDS_STRUCT] >= scores[TEDS_STRUCT_BOOL]
        assert scores[TEDS_STRUCT_BOOL] >= scores[TEDS_CONTENT]


class TestTEDSEvaluateEdgeCases:
    """Edge cases: empty inputs, no tables, malformed HTML."""

    def test_empty_pred(self):
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, gt_n, pred_n = teds.evaluate("", SIMPLE_2X2)
        for v in ALL_TEDS_VARIANTS:
            assert scores[v] == 0.0

    def test_empty_true(self):
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate(SIMPLE_2X2, "")
        for v in ALL_TEDS_VARIANTS:
            assert scores[v] == 0.0

    def test_both_empty(self):
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate("", "")
        for v in ALL_TEDS_VARIANTS:
            assert scores[v] == 0.0

    def test_no_table_element(self):
        teds = TEDS(variants=ALL_TEDS_VARIANTS)
        scores, _, _ = teds.evaluate("<div>hello</div>", SIMPLE_2X2)
        for v in ALL_TEDS_VARIANTS:
            assert scores[v] == 0.0


class TestTEDSVariantSelection:
    """Verify that only requested variants are computed."""

    def test_single_variant(self):
        teds = TEDS(variants={TEDS_STRUCT})
        scores, _, _ = teds.evaluate(SIMPLE_2X2, SIMPLE_2X2_DIFF_CONTENT)
        assert TEDS_STRUCT in scores
        assert TEDS_CONTENT not in scores
        assert TEDS_STRUCT_BOOL not in scores

    def test_two_variants(self):
        teds = TEDS(variants={TEDS_CONTENT, TEDS_STRUCT_BOOL})
        scores, _, _ = teds.evaluate(SIMPLE_2X2, SIMPLE_2X2)
        assert TEDS_CONTENT in scores
        assert TEDS_STRUCT_BOOL in scores
        assert TEDS_STRUCT not in scores


# =============================================================================
# Tests for TEDSMetric.compute() — high-level, returns list[MetricValue]
# =============================================================================


class TestTEDSMetricDefaults:
    """Default TEDSMetric should compute all variants."""

    def test_default_returns_all_variants(self):
        metric = TEDSMetric()
        results = metric.compute(expected=SIMPLE_2X2, actual=SIMPLE_2X2)
        names = {r.metric_name for r in results}
        assert names == set(ALL_TEDS_VARIANTS)
        for r in results:
            assert r.value == 1.0

    def test_content_only_variant(self):
        metric = TEDSMetric(variants={TEDS_CONTENT})
        results = metric.compute(expected=SIMPLE_2X2, actual=SIMPLE_2X2_DIFF_CONTENT)
        assert len(results) == 1
        assert results[0].metric_name == TEDS_CONTENT


class TestTEDSMetricAllVariants:
    """TEDSMetric with all variants."""

    def test_all_variants_returns_three_metrics(self):
        metric = TEDSMetric(variants=ALL_TEDS_VARIANTS)
        results = metric.compute(expected=SIMPLE_2X2, actual=SIMPLE_2X2)
        names = {r.metric_name for r in results}
        assert names == set(ALL_TEDS_VARIANTS)
        for r in results:
            assert r.value == 1.0

    def test_content_diff_scores(self):
        """Different content: struct variants 1.0, content < 1.0."""
        metric = TEDSMetric(variants=ALL_TEDS_VARIANTS)
        results = metric.compute(expected=SIMPLE_2X2, actual=SIMPLE_2X2_DIFF_CONTENT)
        by_name = {r.metric_name: r for r in results}
        assert by_name[TEDS_CONTENT].value < 1.0
        assert by_name[TEDS_STRUCT].value == 1.0
        assert by_name[TEDS_STRUCT_BOOL].value == 1.0

    def test_empty_cell_mismatch(self):
        """One empty cell: struct still 1.0, struct_bool < 1.0."""
        metric = TEDSMetric(variants=ALL_TEDS_VARIANTS)
        results = metric.compute(expected=SIMPLE_2X2, actual=SIMPLE_2X2_ONE_EMPTY)
        by_name = {r.metric_name: r for r in results}
        assert by_name[TEDS_STRUCT].value == 1.0
        assert by_name[TEDS_STRUCT_BOOL].value < 1.0
        assert by_name[TEDS_CONTENT].value < 1.0


class TestTEDSMetricMetadata:
    """Verify metadata fields in returned MetricValues."""

    def test_metadata_fields_present(self):
        metric = TEDSMetric(variants={TEDS_CONTENT})
        results = metric.compute(expected=SIMPLE_2X2, actual=SIMPLE_2X2)
        r = results[0]
        assert r.metadata["tables_predicted"] is True
        assert r.metadata["tables_found_expected"] == 1
        assert r.metadata["tables_found_actual"] == 1
        assert r.metadata["tables_matched"] == 1
        assert len(r.metadata["per_table_scores"]) == 1
        assert len(r.metadata["per_table_details"]) == 1

    def test_per_table_details_use_score_key(self):
        metric = TEDSMetric(variants={TEDS_STRUCT})
        results = metric.compute(expected=SIMPLE_2X2, actual=SIMPLE_2X2)
        detail = results[0].metadata["per_table_details"][0]
        assert "score" in detail
        assert detail["score"] == 1.0
        assert "gt_nodes" in detail
        assert "pred_nodes" in detail

    def test_no_tables_in_expected(self):
        metric = TEDSMetric(variants=ALL_TEDS_VARIANTS)
        results = metric.compute(expected="no tables here", actual=SIMPLE_2X2)
        assert len(results) == 3
        for r in results:
            assert r.value == 0.0
            assert "note" in r.metadata

    def test_no_tables_in_actual(self):
        metric = TEDSMetric(variants=ALL_TEDS_VARIANTS)
        results = metric.compute(expected=SIMPLE_2X2, actual="no tables here")
        assert len(results) == 3
        for r in results:
            assert r.value == 0.0
            assert r.metadata["tables_matched"] == 0


class TestTEDSMetricMultipleTables:
    """Test Hungarian matching with multiple tables."""

    def test_two_tables_in_expected(self):
        two_tables = SIMPLE_2X2 + SIMPLE_3X3
        metric = TEDSMetric(variants={TEDS_CONTENT})
        results = metric.compute(expected=two_tables, actual=two_tables)
        assert len(results) == 1
        assert results[0].value == 1.0
        assert results[0].metadata["tables_found_expected"] == 2
        assert results[0].metadata["tables_found_actual"] == 2
        assert results[0].metadata["tables_matched"] == 2

    def test_fewer_actual_than_expected(self):
        two_tables = SIMPLE_2X2 + SIMPLE_3X3
        metric = TEDSMetric(variants={TEDS_CONTENT})
        results = metric.compute(expected=two_tables, actual=SIMPLE_2X2)
        assert len(results) == 1
        # One table matched at 1.0, one unmatched at 0.0 → average 0.5
        assert results[0].value == pytest.approx(0.5)


# =============================================================================
# Page-constrained matching (same-page table pairing)
# =============================================================================


def _matched(result):
    """Set of (gt_idx, pred_idx) matched pairs from a TEDS MetricValue."""
    return {
        (d["gt_table_index"], d["pred_table_index"])
        for d in result.metadata["per_table_details"]
        if d.get("pred_table_index") is not None
    }


class TestTEDSPageBlocking:
    def test_prefers_same_page_over_better_cross_page_match(self):
        metric = TEDSMetric(variants={TEDS_CONTENT})
        # GT: A@1, B@2.  Pred: B@1, A@2.  (A and B share structure, differ in content.)
        expected = SIMPLE_2X2 + SIMPLE_2X2_DIFF_CONTENT
        actual = SIMPLE_2X2_DIFF_CONTENT + SIMPLE_2X2

        global_res = metric.compute(expected=expected, actual=actual)[0]
        assert global_res.value == pytest.approx(1.0)
        assert _matched(global_res) == {(0, 1), (1, 0)}

        blocked = metric.compute(expected=expected, actual=actual, expected_pages=[1, 2], actual_pages=[1, 2])[0]
        assert _matched(blocked) == {(0, 0), (1, 1)}
        assert blocked.value < 1.0
        for d in blocked.metadata["per_table_details"]:
            if d.get("pred_table_index") is not None:
                assert d["gt_page"] == d["pred_page"]

    def test_unmatched_gt_page_scores_zero(self):
        metric = TEDSMetric(variants={TEDS_CONTENT})
        expected = SIMPLE_2X2 + SIMPLE_2X2_DIFF_CONTENT  # A@1, B@2
        actual = SIMPLE_2X2  # only page 1
        res = metric.compute(expected=expected, actual=actual, expected_pages=[1, 2], actual_pages=[1])[0]
        assert _matched(res) == {(0, 0)}
        assert res.value == pytest.approx(0.5)  # 1.0 matched + 0.0 unmatched page 2

    def test_noop_when_all_one_page(self):
        metric = TEDSMetric(variants={TEDS_CONTENT})
        expected = SIMPLE_2X2 + SIMPLE_2X2_DIFF_CONTENT
        actual = SIMPLE_2X2 + SIMPLE_2X2_DIFF_CONTENT
        plain = metric.compute(expected=expected, actual=actual)[0]
        paged = metric.compute(expected=expected, actual=actual, expected_pages=[1, 1], actual_pages=[1, 1])[0]
        assert plain.value == paged.value
        assert _matched(plain) == _matched(paged)

    def test_length_mismatch_falls_back_to_global(self):
        metric = TEDSMetric(variants={TEDS_CONTENT})
        expected = SIMPLE_2X2 + SIMPLE_2X2_DIFF_CONTENT
        actual = SIMPLE_2X2_DIFF_CONTENT + SIMPLE_2X2
        res = metric.compute(expected=expected, actual=actual, expected_pages=[1], actual_pages=[1, 2])[0]
        assert res.value == pytest.approx(1.0)
        assert _matched(res) == {(0, 1), (1, 0)}
