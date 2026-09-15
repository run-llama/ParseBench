"""Tests for the GriTS (Grid Table Similarity) metric.

Includes sanity checks with hand-crafted tables to verify:
- Identical tables score 1.0
- Spanning cells (rowspan/colspan) are handled correctly
- Missing/extra rows and columns are penalized
- Empty and malformed inputs are handled gracefully
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from parse_bench.evaluation.metrics.parse.grits_metric import (  # noqa: E402
    GriTSMetric,
    _bbox_iou,
    _compute_fscore,
    _lcs_similarity,
    cells_to_grid,
    grits_from_html,
    html_to_cells,
)
from parse_bench.evaluation.metrics.parse.table_extraction import (  # noqa: E402
    extract_html_tables,
    extract_table_pairs,
)


def _compute(metric: GriTSMetric, expected: str, actual: str):
    """Helper: extract tables from markdown and run GriTSMetric.compute."""
    expected_tables, actual_tables, _ = extract_table_pairs(expected, actual)
    return metric.compute(expected_tables, actual_tables)


# =============================================================================
# Test tables
# =============================================================================

# Simple 2x2 table
SIMPLE_2X2 = """<table>
<tr><td>A</td><td>B</td></tr>
<tr><td>C</td><td>D</td></tr>
</table>"""

# Same structure, different content
SIMPLE_2X2_DIFF_CONTENT = """<table>
<tr><td>X</td><td>Y</td></tr>
<tr><td>Z</td><td>W</td></tr>
</table>"""

# 2x2 with one cell changed
SIMPLE_2X2_ONE_CHANGED = """<table>
<tr><td>A</td><td>B</td></tr>
<tr><td>C</td><td>CHANGED</td></tr>
</table>"""

# 3x3 table (different dimensions than 2x2)
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

# Same structure as colspan table but without spanning
TABLE_NO_COLSPAN = """<table>
<tr><td>Header</td><td>Header</td></tr>
<tr><td>A</td><td>B</td></tr>
</table>"""

# Table with rowspan
TABLE_WITH_ROWSPAN = """<table>
<tr><td rowspan="2">Label</td><td>Val1</td></tr>
<tr><td>Val2</td></tr>
</table>"""

# Table with both colspan and rowspan
TABLE_COMPLEX_SPAN = """<table>
<tr><td colspan="2">Title</td><td>Col3</td></tr>
<tr><td rowspan="2">R1</td><td>A</td><td>B</td></tr>
<tr><td>C</td><td>D</td></tr>
</table>"""

# Table with header tags
TABLE_WITH_HEADERS = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>"""

# Same structure and content as TABLE_WITH_HEADERS but using td instead of th
TABLE_WITHOUT_HEADERS = """<table>
<tr><td>Name</td><td>Value</td></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>"""

# Table with extra row appended
TABLE_EXTRA_ROW = """<table>
<tr><td>A</td><td>B</td></tr>
<tr><td>C</td><td>D</td></tr>
<tr><td>E</td><td>F</td></tr>
</table>"""

# Table with extra column
TABLE_EXTRA_COL = """<table>
<tr><td>A</td><td>B</td><td>X</td></tr>
<tr><td>C</td><td>D</td><td>Y</td></tr>
</table>"""

# Realistic data table (financial report style)
FINANCIAL_TABLE_GT = """<table>
<tr><th>Item</th><th>Q1</th><th>Q2</th><th>Q3</th></tr>
<tr><td>Revenue</td><td>$1.2M</td><td>$1.5M</td><td>$1.8M</td></tr>
<tr><td>Expenses</td><td>$0.8M</td><td>$0.9M</td><td>$1.0M</td></tr>
<tr><td>Profit</td><td>$0.4M</td><td>$0.6M</td><td>$0.8M</td></tr>
</table>"""

# Same financial table with OCR-like errors in some cells
FINANCIAL_TABLE_OCR = """<table>
<tr><th>Item</th><th>Q1</th><th>Q2</th><th>Q3</th></tr>
<tr><td>Revenue</td><td>$1.2M</td><td>$1.5M</td><td>$1.8M</td></tr>
<tr><td>Expenses</td><td>$0.8M</td><td>$O.9M</td><td>$1.0M</td></tr>
<tr><td>Profit</td><td>$0.4M</td><td>$0.6M</td><td>$O.8M</td></tr>
</table>"""


# Table with gaps — large colspan that doesn't tile cleanly with other rows.
# This triggers the inhomogeneous numpy array bug when dtype=object is not used.
TABLE_WITH_GAPS = """<table>
<tr><td colspan="3">Wide Header</td></tr>
<tr><td>A</td><td>B</td><td>C</td></tr>
<tr><td colspan="2">Merged</td><td>D</td></tr>
</table>"""

# Tables with unoccupied grid cells (gaps).
# Row 1 has only 1 cell but the grid is 2 columns wide, leaving (1,1) empty.
TABLE_WITH_UNOCCUPIED_CELLS = """<table>
<tr><td>A</td><td>B</td></tr>
<tr><td>C</td></tr>
</table>"""

# Different gap structure for cross-table comparison
TABLE_WITH_UNOCCUPIED_CELLS_2 = """<table>
<tr><td>X</td><td>Y</td></tr>
<tr><td>Z</td></tr>
</table>"""

# A table mimicking a complex invoice: many rows, irregular spanning
TABLE_INVOICE_COMPLEX = """<table>
<tr><th colspan="4">Invoice #12345</th></tr>
<tr><th>Item</th><th>Qty</th><th>Price</th><th>Total</th></tr>
<tr><td>Widget A</td><td>10</td><td>$5.00</td><td>$50.00</td></tr>
<tr><td>Widget B</td><td>5</td><td>$12.00</td><td>$60.00</td></tr>
<tr><td colspan="3">Subtotal</td><td>$110.00</td></tr>
<tr><td colspan="3">Tax (10%)</td><td>$11.00</td></tr>
<tr><td colspan="3">Total</td><td>$121.00</td></tr>
</table>"""


# =============================================================================
# Low-level utility tests
# =============================================================================


class TestBboxIou:
    def test_identical_boxes(self):
        assert _bbox_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0

    def test_no_overlap(self):
        assert _bbox_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0

    def test_partial_overlap(self):
        iou = _bbox_iou([0, 0, 2, 2], [1, 1, 3, 3])
        # intersection = 1x1 = 1, bbox union = [0,0,3,3] = 9
        assert abs(iou - 1.0 / 9.0) < 1e-9

    def test_empty_input(self):
        assert _bbox_iou([], [0, 0, 1, 1]) == 0.0
        assert _bbox_iou([0, 0, 1, 1], []) == 0.0


class TestLcsSimilarity:
    def test_identical_strings(self):
        assert _lcs_similarity("hello", "hello") == 1.0

    def test_empty_strings(self):
        assert _lcs_similarity("", "") == 1.0

    def test_one_empty(self):
        assert _lcs_similarity("hello", "") == 0.0

    def test_similar_strings(self):
        sim = _lcs_similarity("$0.9M", "$O.9M")  # OCR-like error
        assert 0.5 < sim < 1.0

    def test_completely_different(self):
        sim = _lcs_similarity("abc", "xyz")
        assert sim == 0.0


class TestComputeFscore:
    def test_perfect(self):
        f, p, r = _compute_fscore(10, 10, 10)
        assert f == 1.0
        assert p == 1.0
        assert r == 1.0

    def test_no_predictions(self):
        f, p, r = _compute_fscore(0, 10, 0)
        # precision=1 (no predictions), recall=0
        assert p == 1.0
        assert r == 0.0
        assert f == 0.0

    def test_no_ground_truth(self):
        f, p, r = _compute_fscore(0, 0, 10)
        # precision=0, recall=1 (no ground truth)
        assert p == 0.0
        assert r == 1.0
        assert f == 0.0


# =============================================================================
# HTML parsing tests
# =============================================================================


class TestHtmlToCells:
    def test_simple_table(self):
        cells = html_to_cells(SIMPLE_2X2)
        assert cells is not None
        assert len(cells) == 4
        texts = [c["cell_text"].strip() for c in cells]
        assert set(texts) == {"A", "B", "C", "D"}

    def test_colspan(self):
        cells = html_to_cells(TABLE_WITH_COLSPAN)
        assert cells is not None
        # "Header" spans 2 columns, then A and B
        header_cell = [c for c in cells if "Header" in c["cell_text"]][0]
        assert header_cell["column_nums"] == [0, 1]
        assert header_cell["row_nums"] == [0]

    def test_rowspan(self):
        cells = html_to_cells(TABLE_WITH_ROWSPAN)
        assert cells is not None
        label_cell = [c for c in cells if "Label" in c["cell_text"]][0]
        assert label_cell["row_nums"] == [0, 1]
        assert label_cell["column_nums"] == [0]

    def test_invalid_html(self):
        result = html_to_cells("not a table at all")
        assert result is None

    def test_header_detection(self):
        cells = html_to_cells(TABLE_WITH_HEADERS)
        assert cells is not None
        header_cells = [c for c in cells if c["is_column_header"]]
        assert len(header_cells) == 2
        texts = {c["cell_text"].strip() for c in header_cells}
        assert texts == {"Name", "Value"}


class TestCellsToGrid:
    def test_simple_grid(self):
        cells = html_to_cells(SIMPLE_2X2)
        grid = cells_to_grid(cells)
        assert len(grid) == 2
        assert len(grid[0]) == 2
        assert grid[0][0].strip() == "A"
        assert grid[1][1].strip() == "D"


class TestExtractHtmlTables:
    def test_single_table(self):
        tables = extract_html_tables(SIMPLE_2X2)
        assert len(tables) == 1

    def test_multiple_tables(self):
        content = f"Some text\n{SIMPLE_2X2}\nMore text\n{SIMPLE_3X3}\nEnd"
        tables = extract_html_tables(content)
        assert len(tables) == 2

    def test_no_tables(self):
        assert extract_html_tables("just plain text") == []

    def test_empty_content(self):
        assert extract_html_tables("") == []


# =============================================================================
# GriTS core algorithm tests
# =============================================================================


class TestGritsFromHtml:
    def test_identical_tables(self):
        """Identical tables should score 1.0 for content."""
        result = grits_from_html(SIMPLE_2X2, SIMPLE_2X2)
        assert result is not None
        assert result["grits_con"] == pytest.approx(1.0)

    def test_identical_with_spans(self):
        """Identical tables with spanning cells should also score 1.0."""
        result = grits_from_html(TABLE_WITH_COLSPAN, TABLE_WITH_COLSPAN)
        assert result is not None
        assert result["grits_con"] == pytest.approx(1.0)

    def test_identical_complex_span(self):
        """Complex spanning table compared to itself should be 1.0."""
        result = grits_from_html(TABLE_COMPLEX_SPAN, TABLE_COMPLEX_SPAN)
        assert result is not None
        assert result["grits_con"] == pytest.approx(1.0)

    def test_same_structure_different_content(self):
        """Same topology, completely different content → GriTS_Con = 0.0."""
        result = grits_from_html(SIMPLE_2X2, SIMPLE_2X2_DIFF_CONTENT)
        assert result is not None
        assert result["grits_con"] == pytest.approx(0.0)

    def test_one_cell_changed(self):
        """One cell changed out of four → high but not perfect content."""
        result = grits_from_html(SIMPLE_2X2, SIMPLE_2X2_ONE_CHANGED)
        assert result is not None
        # 3 out of 4 cells match perfectly, the fourth has partial similarity
        assert 0.7 < result["grits_con"] < 1.0

    def test_different_dimensions(self):
        """2x2 vs 3x3: some content can still be matched."""
        result = grits_from_html(SIMPLE_2X2, SIMPLE_3X3)
        assert result is not None
        assert 0.0 < result["grits_con"] <= 1.0

    def test_extra_row(self):
        """Original 2x2 vs 3x2 (extra row)."""
        result = grits_from_html(SIMPLE_2X2, TABLE_EXTRA_ROW)
        assert result is not None
        assert result["grits_con"] > 0.5

    def test_extra_column(self):
        """Original 2x2 vs 2x3 (extra column)."""
        result = grits_from_html(SIMPLE_2X2, TABLE_EXTRA_COL)
        assert result is not None
        assert result["grits_con"] > 0.5

    def test_th_vs_td(self):
        """Headers (th) vs data cells (td) with same content."""
        result = grits_from_html(TABLE_WITH_HEADERS, TABLE_WITHOUT_HEADERS)
        assert result is not None
        assert result["grits_con"] == pytest.approx(1.0)

    def test_financial_table_with_ocr_errors(self):
        """Realistic financial table with minor OCR errors."""
        result = grits_from_html(FINANCIAL_TABLE_GT, FINANCIAL_TABLE_OCR)
        assert result is not None
        # 14 out of 16 cells match perfectly; 2 have 0->O substitution
        assert result["grits_con"] > 0.9

    def test_precision_recall(self):
        """Verify precision and recall are reported and make sense."""
        result = grits_from_html(SIMPLE_2X2, TABLE_EXTRA_ROW)
        assert result is not None
        # Pred has more cells than GT, so recall should be >= precision
        assert result["grits_recall_con"] >= result["grits_precision_con"]

    def test_symmetry(self):
        """GriTS(A, B) F-score should equal GriTS(B, A)."""
        result_ab = grits_from_html(SIMPLE_2X2, TABLE_EXTRA_ROW)
        result_ba = grits_from_html(TABLE_EXTRA_ROW, SIMPLE_2X2)
        assert result_ab is not None and result_ba is not None
        assert result_ab["grits_con"] == pytest.approx(result_ba["grits_con"])

    def test_invalid_html_returns_none(self):
        """Invalid HTML should return None."""
        result = grits_from_html("not html", SIMPLE_2X2)
        assert result is None

    def test_table_with_gaps_no_crash(self):
        """Tables with irregular spanning should not crash with numpy errors."""
        result = grits_from_html(TABLE_WITH_GAPS, TABLE_WITH_GAPS)
        assert result is not None
        assert result["grits_con"] == pytest.approx(1.0)

    def test_complex_invoice_table(self):
        """Complex invoice table with mixed spanning should work."""
        result = grits_from_html(TABLE_INVOICE_COMPLEX, TABLE_INVOICE_COMPLEX)
        assert result is not None
        assert result["grits_con"] == pytest.approx(1.0)

    def test_complex_invoice_vs_simple(self):
        """Cross-comparison between complex and simple tables."""
        result = grits_from_html(TABLE_INVOICE_COMPLEX, FINANCIAL_TABLE_GT)
        assert result is not None
        assert 0.0 <= result["grits_con"] <= 1.0


# =============================================================================
# GriTSMetric class tests (Metric interface)
# =============================================================================


def _find_metric(results, name):
    """Helper: find a MetricValue by name in a list."""
    for r in results:
        if r.metric_name == name:
            return r
    raise AssertionError(f"Metric '{name}' not found in {[r.metric_name for r in results]}")


class TestGriTSMetric:
    def setup_method(self):
        self.metric = GriTSMetric()

    def test_boolean_marker_content_equivalence(self):
        """Check/cross/dot markers should match textual boolean cells."""
        expected = """<table>
<tr><th>Dataset</th><th>TD</th><th>TR</th><th>Network</th><th>Flag</th></tr>
<tr><td>Marmot</td><td>✓</td><td>✗</td><td>●</td><td>X</td></tr>
</table>"""
        actual = """<table>
<tr><th>Dataset</th><th>TD</th><th>TR</th><th>Network</th><th>Flag</th></tr>
<tr><td>Marmot</td><td>[yes]</td><td>[no]</td><td>[yes]</td><td>[yes]</td></tr>
</table>"""

        result = grits_from_html(expected, actual)

        assert result is not None
        assert result["grits_con"] == pytest.approx(1.0)

    def test_name(self):
        assert self.metric.name == "grits"

    def test_returns_one_metric(self):
        """compute() returns a single grits_con MetricValue."""
        results = _compute(self.metric, SIMPLE_2X2, SIMPLE_2X2)
        assert isinstance(results, list)
        assert len(results) == 1
        names = {r.metric_name for r in results}
        assert names == {"grits_con"}

    def test_identical_tables(self):
        results = _compute(self.metric, SIMPLE_2X2, SIMPLE_2X2)
        con = _find_metric(results, "grits_con")
        assert con.value == pytest.approx(1.0)

    def test_no_tables_in_expected(self):
        results = _compute(self.metric, "no tables here", SIMPLE_2X2)
        con = _find_metric(results, "grits_con")
        assert con.value == 0.0
        assert con.metadata["tables_found_expected"] == 0

    def test_no_tables_in_actual(self):
        results = _compute(self.metric, SIMPLE_2X2, "no tables here")
        con = _find_metric(results, "grits_con")
        assert con.value == 0.0
        assert con.metadata["tables_found_actual"] == 0

    def test_multiple_tables_matching(self):
        """Multiple tables should be matched optimally via Hungarian algorithm."""
        expected = f"{SIMPLE_2X2}\n{SIMPLE_3X3}"
        actual = f"{SIMPLE_3X3}\n{SIMPLE_2X2}"  # Reversed order
        results = _compute(self.metric, expected, actual)
        con = _find_metric(results, "grits_con")
        assert con.value == pytest.approx(1.0)
        assert con.metadata["tables_matched"] == 2

    def test_metadata_has_per_table_details(self):
        results = _compute(self.metric, SIMPLE_2X2, SIMPLE_2X2)
        con = _find_metric(results, "grits_con")
        assert "per_table_details" in con.metadata
        details = con.metadata["per_table_details"]
        assert len(details) == 1
        assert details[0]["grits_con"] == pytest.approx(1.0)

    def test_value_range(self):
        """All scores should be in [0, 1]."""
        for actual in [SIMPLE_2X2, SIMPLE_2X2_DIFF_CONTENT, TABLE_EXTRA_ROW, SIMPLE_3X3]:
            results = _compute(self.metric, SIMPLE_2X2, actual)
            con = _find_metric(results, "grits_con")
            assert 0.0 <= con.value <= 1.0


# =============================================================================
# Reference implementation comparison tests
#
# These compare our implementation against the Microsoft table-transformer
# reference. The vendored file (_vendor_grits_reference.py) should be
# removed before deploying.
# =============================================================================

# Import reference implementation (from microsoft/table-transformer)
from parse_bench.evaluation.metrics.parse._vendor_grits_reference import (  # noqa: E402
    grits_from_html as ref_grits_from_html,
)
from parse_bench.evaluation.metrics.parse._vendor_grits_reference import (  # noqa: E402
    iou as ref_iou,
)
from parse_bench.evaluation.metrics.parse._vendor_grits_reference import (  # noqa: E402
    lcs_similarity as ref_lcs_similarity,
)

# All table pairs to test against reference
_REFERENCE_TEST_PAIRS = [
    ("identical_2x2", SIMPLE_2X2, SIMPLE_2X2),
    ("diff_content_2x2", SIMPLE_2X2, SIMPLE_2X2_DIFF_CONTENT),
    ("one_changed_2x2", SIMPLE_2X2, SIMPLE_2X2_ONE_CHANGED),
    ("2x2_vs_3x3", SIMPLE_2X2, SIMPLE_3X3),
    ("colspan", TABLE_WITH_COLSPAN, TABLE_WITH_COLSPAN),
    ("colspan_vs_no_colspan", TABLE_WITH_COLSPAN, TABLE_NO_COLSPAN),
    ("rowspan", TABLE_WITH_ROWSPAN, TABLE_WITH_ROWSPAN),
    ("complex_span", TABLE_COMPLEX_SPAN, TABLE_COMPLEX_SPAN),
    ("extra_row", SIMPLE_2X2, TABLE_EXTRA_ROW),
    ("extra_col", SIMPLE_2X2, TABLE_EXTRA_COL),
    ("headers_th_vs_td", TABLE_WITH_HEADERS, TABLE_WITHOUT_HEADERS),
    ("financial_ocr", FINANCIAL_TABLE_GT, FINANCIAL_TABLE_OCR),
    ("reversed_extra_row", TABLE_EXTRA_ROW, SIMPLE_2X2),
]


class TestReferenceComparison:
    """Compare our GriTS implementation against the Microsoft reference.

    Every test pair should produce identical scores (within floating point
    tolerance) for grits_top, grits_con, and their precision/recall variants.
    """

    @pytest.mark.parametrize(
        "name,true_html,pred_html",
        _REFERENCE_TEST_PAIRS,
        ids=[p[0] for p in _REFERENCE_TEST_PAIRS],
    )
    def test_matches_reference(self, name, true_html, pred_html):
        ours = grits_from_html(true_html, pred_html)
        ref = ref_grits_from_html(true_html, pred_html)

        assert ours is not None, f"Our implementation returned None for {name}"
        assert ref is not None, f"Reference implementation returned None for {name}"

        for key in [
            "grits_con",
            "grits_precision_con",
            "grits_recall_con",
            "grits_con_upper_bound",
        ]:
            assert ours[key] == pytest.approx(ref[key], abs=1e-9), (
                f"{name}: {key} mismatch: ours={ours[key]}, ref={ref[key]}"
            )


class TestReferenceTimingComparison:
    """Timing comparison between our implementation and the reference.

    Not a correctness test — just prints timing info for manual inspection.
    Uses a larger table to make timing differences visible.
    """

    # 6x4 table with mixed content for realistic timing
    TIMING_TABLE_GT = """<table>
<tr><th>Region</th><th>Q1</th><th>Q2</th><th>Q3</th></tr>
<tr><td>North America</td><td>$12.3M</td><td>$14.1M</td><td>$15.8M</td></tr>
<tr><td>Europe</td><td>$8.7M</td><td>$9.2M</td><td>$10.1M</td></tr>
<tr><td>Asia Pacific</td><td>$6.4M</td><td>$7.8M</td><td>$8.9M</td></tr>
<tr><td>Latin America</td><td>$2.1M</td><td>$2.5M</td><td>$3.0M</td></tr>
<tr><td colspan="3">Total</td><td>$55.9M</td></tr>
</table>"""

    TIMING_TABLE_PRED = """<table>
<tr><th>Region</th><th>Q1</th><th>Q2</th><th>Q3</th></tr>
<tr><td>North America</td><td>$12.3M</td><td>$14.1M</td><td>$15.8M</td></tr>
<tr><td>Europe</td><td>$8.7M</td><td>$9.2M</td><td>$1O.1M</td></tr>
<tr><td>Asia Pacific</td><td>$6.4M</td><td>$7.8M</td><td>$8.9M</td></tr>
<tr><td>Latin America</td><td>$2.1M</td><td>$2.5M</td><td>$3.OM</td></tr>
<tr><td colspan="3">Total</td><td>$55.9M</td></tr>
</table>"""

    def test_timing_comparison(self):
        """Compare wall-clock time of both implementations.

        Runs each implementation multiple times and reports median timing.
        This test always passes — it's purely informational.
        """
        import time

        n_iterations = 20

        # Warm up both implementations
        grits_from_html(self.TIMING_TABLE_GT, self.TIMING_TABLE_PRED)
        ref_grits_from_html(self.TIMING_TABLE_GT, self.TIMING_TABLE_PRED)

        # Time our implementation
        ours_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            grits_from_html(self.TIMING_TABLE_GT, self.TIMING_TABLE_PRED)
            ours_times.append(time.perf_counter() - start)

        # Time reference implementation
        ref_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            ref_grits_from_html(self.TIMING_TABLE_GT, self.TIMING_TABLE_PRED)
            ref_times.append(time.perf_counter() - start)

        ours_median = sorted(ours_times)[n_iterations // 2]
        ref_median = sorted(ref_times)[n_iterations // 2]

        print(f"\n  Timing comparison (median of {n_iterations} runs, 6x4 table):")
        print(f"    Our implementation: {ours_median * 1000:.2f} ms")
        print(f"    Reference (MS):     {ref_median * 1000:.2f} ms")
        print(f"    Ratio (ours/ref):   {ours_median / ref_median:.2f}x")


# =============================================================================
# Unoccupied grid cell tests (crash regression)
#
# Tables with gaps (grid positions not covered by any <td>/<th>) produce
# scalar 0 at those positions. Before the fix, the reference implementation
# crashed: iou() called Rect(0) → TypeError, lcs_similarity() called
# len(0) → TypeError. Both now gracefully return 0.0 for those cells.
# =============================================================================


class TestUnoccupiedGridCells:
    """Verify both implementations handle tables with unoccupied grid cells.

    Before the fix, the reference's iou(0, ...) → Rect(0) → TypeError
    crashed the entire grits_from_html call, scoring both GriTS_Top and
    GriTS_Con as 0.0 for the whole table pair. Our implementation was
    already safe. After the fix, both survive and produce real scores.
    """

    def test_ref_iou_scalar_zero(self):
        """Reference iou() should handle scalar inputs, not crash.

        Before fix: Rect(0) → TypeError
        After fix:
          - Both unoccupied (0 vs 0) → 1.0 (structural agreement)
          - One occupied, one not → 0.0 (mismatch)
        """
        assert ref_iou(0, [0, 0, 1, 1]) == 0.0
        assert ref_iou([0, 0, 1, 1], 0) == 0.0
        assert ref_iou(0, 0) == 1.0  # both unoccupied = match
        assert ref_iou(0.0, 0.0) == 1.0

    def test_ref_lcs_similarity_scalar_zero(self):
        """Reference lcs_similarity() should handle non-string inputs, not crash.

        Before fix: len(0) → TypeError
        After fix:  converts to str, computes similarity normally
        """
        assert ref_lcs_similarity(0, 0) == 1.0  # "0" vs "0"
        assert ref_lcs_similarity("hello", 0) == ref_lcs_similarity("hello", "0")
        assert ref_lcs_similarity(0, "hello") == ref_lcs_similarity("0", "hello")

    def test_our_impl_identical_tables_with_gaps(self):
        """Identical tables with gaps should score 1.0 for content."""
        result = grits_from_html(TABLE_WITH_UNOCCUPIED_CELLS, TABLE_WITH_UNOCCUPIED_CELLS)
        assert result is not None
        assert result["grits_con"] == pytest.approx(1.0)

    def test_ref_impl_identical_tables_with_gaps(self):
        """Reference implementation now also survives tables with gaps.

        Before fix: Rect(0) → TypeError → crashed → scored as 0.0
        After fix:  identical tables score 1.0
        """
        result = ref_grits_from_html(TABLE_WITH_UNOCCUPIED_CELLS, TABLE_WITH_UNOCCUPIED_CELLS)
        assert result is not None
        assert result["grits_top"] == pytest.approx(1.0)
        assert result["grits_con"] == pytest.approx(1.0)

    def test_cross_table_gap_vs_no_gap(self):
        """Table with gap vs table without — scores should be real, not crash."""
        ours = grits_from_html(TABLE_WITH_UNOCCUPIED_CELLS, SIMPLE_2X2)
        ref = ref_grits_from_html(TABLE_WITH_UNOCCUPIED_CELLS, SIMPLE_2X2)
        assert ours is not None
        assert ref is not None
        # GriTS_Con should match (both use LCS, same logic now)
        assert ours["grits_con"] == pytest.approx(ref["grits_con"], abs=1e-9)

    def test_gap_vs_gap_different_content(self):
        """Two tables with same gap structure but different content."""
        ours = grits_from_html(TABLE_WITH_UNOCCUPIED_CELLS, TABLE_WITH_UNOCCUPIED_CELLS_2)
        ref = ref_grits_from_html(TABLE_WITH_UNOCCUPIED_CELLS, TABLE_WITH_UNOCCUPIED_CELLS_2)
        assert ours is not None
        assert ref is not None
        # Different content but gap cells give partial credit → between 0 and 1
        assert 0.0 < ours["grits_con"] < 1.0
        # Both implementations agree on content score
        assert ours["grits_con"] == pytest.approx(ref["grits_con"], abs=1e-9)


# =============================================================================
# extract_html_tables — incomplete table handling
# =============================================================================


def test_extract_html_tables_incomplete_table():
    """An incomplete table (no closing </table>) should still be extracted."""
    html = "<table><tr><td>A</td><td>B</td></tr>"
    tables = extract_html_tables(html)
    assert len(tables) == 1
    assert "<td>A</td>" in tables[0]


def test_extract_html_tables_incomplete_after_complete():
    """A complete table followed by an incomplete one should yield 2 tables."""
    html = "<table><tr><td>1</td></tr></table><table><tr><td>2</td></tr>"
    tables = extract_html_tables(html)
    assert len(tables) == 2


# =============================================================================
# Page-constrained matching (same-page table pairing)
# =============================================================================

# Two same-structure 2x2 tables with disjoint content. Cross-pairing identical
# content scores 1.0; pairing alpha against beta scores low.
_T_ALPHA = "<table><tr><td>a1</td><td>a2</td></tr><tr><td>a3</td><td>a4</td></tr></table>"
_T_BETA = "<table><tr><td>b1</td><td>b2</td></tr><tr><td>b3</td><td>b4</td></tr></table>"


def _pairing(grits_results):
    return list(grits_results[0].metadata["pairing"])


def test_page_blocking_prefers_same_page_over_better_cross_page_match():
    metric = GriTSMetric()
    # GT: alpha@page1, beta@page2.  Pred: beta@page1, alpha@page2.
    expected = _T_ALPHA + _T_BETA
    actual = _T_BETA + _T_ALPHA
    exp_tables, act_tables, _ = extract_table_pairs(expected, actual)

    # Global matching cross-pairs the identical content -> perfect score.
    global_res = metric.compute(exp_tables, act_tables)
    assert global_res[0].value == 1.0
    assert set(_pairing(global_res)) == {(0, 1), (1, 0)}

    # Page-blocked: alpha@1 must pair with beta@1, beta@2 with alpha@2 -> low.
    blocked_res = metric.compute(exp_tables, act_tables, expected_pages=[1, 2], actual_pages=[1, 2])
    assert set(_pairing(blocked_res)) == {(0, 0), (1, 1)}
    assert blocked_res[0].value < 1.0
    # per_table_details carry the page labels and never cross pages.
    for d in blocked_res[0].metadata["per_table_details"]:
        if d.get("pred_table_index") is not None:
            assert d["gt_page"] == d["pred_page"]


def test_page_blocking_unmatched_gt_page_scores_zero():
    metric = GriTSMetric()
    # GT alpha@1, beta@2; prediction only on page 1.
    expected = _T_ALPHA + _T_BETA
    actual = _T_ALPHA
    exp_tables, act_tables, _ = extract_table_pairs(expected, actual)
    res = metric.compute(exp_tables, act_tables, expected_pages=[1, 2], actual_pages=[1])
    pairing = _pairing(res)
    assert (0, 0) in pairing
    assert (1, None) in pairing
    # avg over 2 GT tables: 1.0 (matched) + 0.0 (unmatched page 2) = 0.5
    assert res[0].value == 0.5


def test_page_blocking_pred_only_page_is_ignored():
    metric = GriTSMetric()
    # GT alpha@1 only; predictions alpha@1 and beta@3.
    expected = _T_ALPHA
    actual = _T_ALPHA + _T_BETA
    exp_tables, act_tables, _ = extract_table_pairs(expected, actual)
    res = metric.compute(exp_tables, act_tables, expected_pages=[1], actual_pages=[1, 3])
    assert res[0].value == 1.0  # the lone GT pairs cleanly; extra pred ignored
    assert _pairing(res) == [(0, 0)]


def test_page_blocking_noop_when_all_one_page():
    metric = GriTSMetric()
    expected = _T_ALPHA + _T_BETA
    actual = _T_ALPHA + _T_BETA
    exp_tables, act_tables, _ = extract_table_pairs(expected, actual)
    plain = metric.compute(exp_tables, act_tables)
    paged = metric.compute(exp_tables, act_tables, expected_pages=[1, 1], actual_pages=[1, 1])
    assert plain[0].value == paged[0].value
    assert _pairing(plain) == _pairing(paged)


def test_page_blocking_length_mismatch_falls_back_to_global():
    metric = GriTSMetric()
    expected = _T_ALPHA + _T_BETA
    actual = _T_BETA + _T_ALPHA
    exp_tables, act_tables, _ = extract_table_pairs(expected, actual)
    # expected_pages too short -> page blocking disabled -> global cross-pairing.
    res = metric.compute(exp_tables, act_tables, expected_pages=[1], actual_pages=[1, 2])
    assert res[0].value == 1.0
    assert set(_pairing(res)) == {(0, 1), (1, 0)}
