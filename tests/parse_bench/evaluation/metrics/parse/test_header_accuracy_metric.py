"""Tests for the header accuracy metric."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from parse_bench.evaluation.metrics.parse.header_accuracy_metric import (
    SUBMETRIC_KEYS,
    HeaderAccuracyMetric,
    HeaderAccuracyMetricGenerous,
    HeaderBlock,
    HeaderCell,
    _apply_generous_header_normalization,
    _block_edge_distance,
    _block_edge_vector,
    _block_extent_iou,
    _build_text_lookup,
    _detailed_header_composite_for_table_pair,
    _direction_similarity,
    _find_contiguous_groups,
    _find_header_blocks,
    _header_block_relative_position_score,
    _header_cell_count_score,
    _header_content_bag_score,
    _header_data_alignment_score,
    _header_grits_score,
    _header_hierarchy_depth,
    _header_hierarchy_depth_score,
    _header_perfect_score,
    _is_bottom_left_block,
    _match_blocks,
    _parse_header_cells,
    _promote_bottom_left_to_header,
    _promote_top_row_to_header,
    compute_header_composite_for_table_pair,
)

# =============================================================================
# Test tables
# =============================================================================

# Simple table with th headers
TABLE_SIMPLE_HEADERS = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>"""

# Same structure but td instead of th (no headers)
TABLE_NO_HEADERS = """<table>
<tr><td>Name</td><td>Value</td></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>"""

# Same headers, different body
TABLE_SAME_HEADERS_DIFF_BODY = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Gamma</td><td>300</td></tr>
<tr><td>Delta</td><td>400</td></tr>
</table>"""

# Different header text, same structure
TABLE_DIFF_HEADER_TEXT = """<table>
<tr><th>Label</th><th>Amount</th></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>"""

# Extra header column
TABLE_EXTRA_HEADER_COL = """<table>
<tr><th>Name</th><th>Value</th><th>Unit</th></tr>
<tr><td>Alpha</td><td>100</td><td>kg</td></tr>
</table>"""

# Multi-row header with colspan
TABLE_MULTI_ROW_HEADER = """<table>
<tr><th colspan="2">Financial Summary</th></tr>
<tr><th>Item</th><th>Amount</th></tr>
<tr><td>Revenue</td><td>$1M</td></tr>
</table>"""

# Same multi-row header
TABLE_MULTI_ROW_HEADER_SAME = """<table>
<tr><th colspan="2">Financial Summary</th></tr>
<tr><th>Item</th><th>Amount</th></tr>
<tr><td>Expenses</td><td>$0.5M</td></tr>
</table>"""

# Multi-row header with wrong colspan
TABLE_MULTI_ROW_HEADER_WRONG_SPAN = """<table>
<tr><th>Financial Summary</th><th></th></tr>
<tr><th>Item</th><th>Amount</th></tr>
<tr><td>Revenue</td><td>$1M</td></tr>
</table>"""

# Table with row headers (th in first column)
TABLE_ROW_HEADERS = """<table>
<tr><th>Category</th><td>Q1</td><td>Q2</td></tr>
<tr><th>Revenue</th><td>$1M</td><td>$2M</td></tr>
<tr><th>Cost</th><td>$0.5M</td><td>$0.8M</td></tr>
</table>"""

# Table with both row and column headers (two blocks)
TABLE_TWO_HEADER_BLOCKS = """<table>
<tr><th></th><th>Q1</th><th>Q2</th></tr>
<tr><th>Revenue</th><td>$1M</td><td>$2M</td></tr>
<tr><th>Cost</th><td>$0.5M</td><td>$0.8M</td></tr>
</table>"""

# Table with thead
TABLE_WITH_THEAD = """<table>
<thead>
<tr><td>Name</td><td>Value</td></tr>
</thead>
<tbody>
<tr><td>Alpha</td><td>100</td></tr>
</tbody>
</table>"""

# Table with formatting in headers
TABLE_FORMATTED_HEADERS = """<table>
<tr><th><b>Name</b></th><th><i>Value</i></th></tr>
<tr><td>Alpha</td><td>100</td></tr>
</table>"""

# Table with strikethrough in headers
TABLE_STRIKETHROUGH_HEADERS = """<table>
<tr><th><s>Name</s></th><th><del>Value</del></th></tr>
<tr><td>Alpha</td><td>100</td></tr>
</table>"""


# --- Tables for header_data_alignment tests ---

# 3-column table with unique data (forces deterministic GriTS alignment)
TABLE_3COL = """<table>
<tr><th>Name</th><th>Age</th><th>City</th></tr>
<tr><td>Alice</td><td>30</td><td>NYC</td></tr>
<tr><td>Bob</td><td>25</td><td>LA</td></tr>
</table>"""

# Extra empty <td> in header row — pushes header cols 1,2 to positions 2,3
TABLE_3COL_EXTRA_TD = """<table>
<tr><th>Name</th><td></td><th>Age</th><th>City</th></tr>
<tr><td>Alice</td><td>30</td><td>NYC</td><td></td></tr>
<tr><td>Bob</td><td>25</td><td>LA</td><td></td></tr>
</table>"""

# Same layout but the extra cell is a <th> instead of <td>
TABLE_3COL_EXTRA_TH = """<table>
<tr><th>Name</th><th></th><th>Age</th><th>City</th></tr>
<tr><td>Alice</td><td>30</td><td>NYC</td><td></td></tr>
<tr><td>Bob</td><td>25</td><td>LA</td><td></td></tr>
</table>"""

# Same content as TABLE_3COL but all headers are <td> (no th at all)
TABLE_3COL_NO_TH = """<table>
<tr><td>Name</td><td>Age</td><td>City</td></tr>
<tr><td>Alice</td><td>30</td><td>NYC</td></tr>
<tr><td>Bob</td><td>25</td><td>LA</td></tr>
</table>"""

# Two extra empty cells at different positions in the header row:
# one after Name and one after Age.  Data rows also have 5 columns
# (empty cells at positions 1 and 3) so the table is rectangular.
#
# GT grid (3 cols):         Pred grid (5 cols):
#   Name  Age  City           Name  ""  Age  ""  City
#   Alice 30   NYC            Alice 30  NYC  ""  ""
#   Bob   25   LA             Bob   25  LA   ""  ""
#
# GriTS col alignment (data-dominated):
#   col 0→0 ("alice"↔"alice", "bob"↔"bob", "name"↔"name")
#   col 1→1 ("30"↔"30", "25"↔"25" dominate over "age"↔"")
#   col 2→2 ("nyc"↔"nyc", "la"↔"la" dominate over "city"↔"age")
#
# Header text check at mapped positions:
#   "name" at (0,0) → pred (0,0) = "name" → match
#   "age"  at (0,1) → pred (0,1) = ""     → mismatch
#   "city" at (0,2) → pred (0,2) = "age"  → mismatch
# Score = 1/3
TABLE_3COL_TWO_EXTRA_SPREAD = """<table>
<tr><th>Name</th><td></td><th>Age</th><td></td><th>City</th></tr>
<tr><td>Alice</td><td>30</td><td>NYC</td><td></td><td></td></tr>
<tr><td>Bob</td><td>25</td><td>LA</td><td></td><td></td></tr>
</table>"""

# Two extra empty cells both inserted before the first header.
# Data rows also have 5 columns (empty cells at positions 0 and 1).
#
# GT grid (3 cols):         Pred grid (5 cols):
#   Name  Age  City           ""  ""  Name  Age  City
#   Alice 30   NYC            ""  ""  Alice 30   NYC
#   Bob   25   LA             ""  ""  Bob   25   LA
#
# GriTS col alignment (data-dominated):
#   col 0→2 ("alice"↔"alice", "bob"↔"bob", "name"↔"name")
#   col 1→3 ("30"↔"30", "25"↔"25", "age"↔"age")
#   col 2→4 ("nyc"↔"nyc", "la"↔"la", "city"↔"city")
#
# Header text check: all 3 headers map to correct text → score = 1.0
TABLE_3COL_TWO_EXTRA_LEADING = """<table>
<tr><td></td><td></td><th>Name</th><th>Age</th><th>City</th></tr>
<tr><td></td><td></td><td>Alice</td><td>30</td><td>NYC</td></tr>
<tr><td></td><td></td><td>Bob</td><td>25</td><td>LA</td></tr>
</table>"""

# Two extra empty cells both at the end of each row.
# The extra cells don't displace any headers.
#
# GriTS col alignment:
#   col 0→0, col 1→1, col 2→2  (content matches perfectly in first 3 cols)
#
# All headers at mapped positions have correct text → score = 1.0
TABLE_3COL_TWO_EXTRA_TRAILING = """<table>
<tr><th>Name</th><th>Age</th><th>City</th><td></td><td></td></tr>
<tr><td>Alice</td><td>30</td><td>NYC</td><td></td><td></td></tr>
<tr><td>Bob</td><td>25</td><td>LA</td><td></td><td></td></tr>
</table>"""

# Extra empty row inserted before the header row.
# Data content is unchanged so column alignment is perfect.
#
# GriTS row alignment:
#   row 0→1 (header content "name"↔"name", "age"↔"age", "city"↔"city"
#            outweigh "name"↔"", "age"↔"", "city"↔"")
#   row 1→2, row 2→3
#
# Headers at (0,0),(0,1),(0,2) map to (1,0),(1,1),(1,2) in pred,
# which have "name","age","city" → all match → score = 1.0
TABLE_3COL_EXTRA_ROW = """<table>
<tr><td></td><td></td><td></td></tr>
<tr><th>Name</th><th>Age</th><th>City</th></tr>
<tr><td>Alice</td><td>30</td><td>NYC</td></tr>
<tr><td>Bob</td><td>25</td><td>LA</td></tr>
</table>"""


# =============================================================================
# Header cell parsing tests
# =============================================================================


class TestParseHeaderCells:
    def test_simple_th_headers(self):
        cells, rows, cols = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        assert len(cells) == 2
        texts = {c.text for c in cells}
        assert "name" in texts  # normalized to lowercase
        assert "value" in texts

    def test_no_headers(self):
        cells, rows, cols = _parse_header_cells(TABLE_NO_HEADERS)
        assert len(cells) == 0

    def test_thead_marks_headers(self):
        cells, rows, cols = _parse_header_cells(TABLE_WITH_THEAD)
        assert len(cells) == 2
        texts = {c.text for c in cells}
        assert "name" in texts
        assert "value" in texts

    def test_colspan_header(self):
        cells, rows, cols = _parse_header_cells(TABLE_MULTI_ROW_HEADER)
        spanning = [c for c in cells if c.colspan == 2]
        assert len(spanning) == 1
        assert "financial summary" in spanning[0].text

    def test_row_headers(self):
        cells, rows, cols = _parse_header_cells(TABLE_ROW_HEADERS)
        assert len(cells) == 3  # Category, Revenue, Cost
        assert rows == 3
        assert cols == 3

    def test_formatting_stripped(self):
        """Bold/italic/strikethrough formatting should be stripped."""
        plain_cells, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        bold_cells, _, _ = _parse_header_cells(TABLE_FORMATTED_HEADERS)
        strike_cells, _, _ = _parse_header_cells(TABLE_STRIKETHROUGH_HEADERS)

        plain_texts = sorted(c.text for c in plain_cells)
        bold_texts = sorted(c.text for c in bold_cells)
        strike_texts = sorted(c.text for c in strike_cells)

        assert plain_texts == bold_texts
        assert plain_texts == strike_texts


# =============================================================================
# Header block detection tests
# =============================================================================


class TestFindHeaderBlocks:
    def test_single_header_row(self):
        cells, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        blocks = _find_header_blocks(cells)
        assert len(blocks) == 1
        assert len(blocks[0].cells) == 2

    def test_multi_row_header(self):
        cells, _, _ = _parse_header_cells(TABLE_MULTI_ROW_HEADER)
        blocks = _find_header_blocks(cells)
        # Colspan cell in row 0 + two cells in row 1 = one contiguous block
        assert len(blocks) == 1
        assert len(blocks[0].cells) == 3

    def test_two_header_blocks(self):
        cells, _, _ = _parse_header_cells(TABLE_TWO_HEADER_BLOCKS)
        blocks = _find_header_blocks(cells)
        # The corner th, top row ths, and left column ths are all adjacent
        # so they should form one block
        assert len(blocks) == 1

    def test_no_headers(self):
        cells, _, _ = _parse_header_cells(TABLE_NO_HEADERS)
        blocks = _find_header_blocks(cells)
        assert len(blocks) == 0


# =============================================================================
# Block matching tests
# =============================================================================


class TestMatchBlocks:
    def test_identical_blocks(self):
        cells, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        blocks = _find_header_blocks(cells)
        gt_to_pred, grits_scores = _match_blocks(blocks, blocks)
        assert len(gt_to_pred) == 1
        assert gt_to_pred[0] == 0
        assert grits_scores[(0, 0)] == pytest.approx(1.0)

    def test_empty_blocks(self):
        gt_to_pred, grits_scores = _match_blocks([], [])
        assert gt_to_pred == {}
        assert grits_scores == {}

    def test_no_pred_blocks(self):
        cells, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        blocks = _find_header_blocks(cells)
        gt_to_pred, grits_scores = _match_blocks(blocks, [])
        assert gt_to_pred == {}

    def test_extent_iou_tiebreak_same_content(self):
        """When two pred blocks have identical content, prefer positional match."""
        # GT: block at top-left, block at bottom-right
        gt_block_top = HeaderBlock(
            cells=[HeaderCell(text="x", row=0, col=0, rowspan=1, colspan=1)],
            min_row=0,
            max_row=1,
            min_col=0,
            max_col=1,
        )
        gt_block_bot = HeaderBlock(
            cells=[HeaderCell(text="x", row=5, col=5, rowspan=1, colspan=1)],
            min_row=5,
            max_row=6,
            min_col=5,
            max_col=6,
        )
        # Pred: same content "x" at both positions (GriTS-Con ties)
        pred_block_top = HeaderBlock(
            cells=[HeaderCell(text="x", row=0, col=0, rowspan=1, colspan=1)],
            min_row=0,
            max_row=1,
            min_col=0,
            max_col=1,
        )
        pred_block_bot = HeaderBlock(
            cells=[HeaderCell(text="x", row=5, col=5, rowspan=1, colspan=1)],
            min_row=5,
            max_row=6,
            min_col=5,
            max_col=6,
        )
        gt_to_pred, _ = _match_blocks(
            [gt_block_top, gt_block_bot],
            [pred_block_top, pred_block_bot],
            gt_rows=6,
            gt_cols=6,
            pred_rows=6,
            pred_cols=6,
        )
        # Tiebreaker should match top↔top (0↔0) and bot↔bot (1↔1)
        assert gt_to_pred[0] == 0
        assert gt_to_pred[1] == 1

    def test_content_overrides_position(self):
        """GriTS-Con should still win over position when content differs."""
        # GT: block A at top, block B at bottom
        gt_block_top = HeaderBlock(
            cells=[HeaderCell(text="alpha", row=0, col=0, rowspan=1, colspan=1)],
            min_row=0,
            max_row=1,
            min_col=0,
            max_col=1,
        )
        gt_block_bot = HeaderBlock(
            cells=[HeaderCell(text="beta", row=5, col=0, rowspan=1, colspan=1)],
            min_row=5,
            max_row=6,
            min_col=0,
            max_col=1,
        )
        # Pred: "beta" at top position, "alpha" at bottom — content swapped
        pred_block_top = HeaderBlock(
            cells=[HeaderCell(text="beta", row=0, col=0, rowspan=1, colspan=1)],
            min_row=0,
            max_row=1,
            min_col=0,
            max_col=1,
        )
        pred_block_bot = HeaderBlock(
            cells=[HeaderCell(text="alpha", row=5, col=0, rowspan=1, colspan=1)],
            min_row=5,
            max_row=6,
            min_col=0,
            max_col=1,
        )
        gt_to_pred, _ = _match_blocks(
            [gt_block_top, gt_block_bot],
            [pred_block_top, pred_block_bot],
            gt_rows=6,
            gt_cols=1,
            pred_rows=6,
            pred_cols=1,
        )
        # Content match should win: "alpha" GT top (0) ↔ "alpha" pred bot (1)
        assert gt_to_pred[0] == 1
        assert gt_to_pred[1] == 0


class TestBlockExtentIou:
    def test_identical_blocks(self):
        b = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=3)
        assert _block_extent_iou(b, b, 4, 4, 4, 4) == pytest.approx(1.0)

    def test_non_overlapping_blocks(self):
        b1 = HeaderBlock(min_row=0, max_row=1, min_col=0, max_col=1)
        b2 = HeaderBlock(min_row=3, max_row=4, min_col=3, max_col=4)
        assert _block_extent_iou(b1, b2, 4, 4, 4, 4) == pytest.approx(0.0)

    def test_partial_overlap(self):
        b1 = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=2)
        b2 = HeaderBlock(min_row=1, max_row=3, min_col=1, max_col=3)
        iou = _block_extent_iou(b1, b2, 4, 4, 4, 4)
        # Intersection: [0.25,0.25]-[0.5,0.5] = 0.0625
        # Each area: 0.25, union: 0.5 - 0.0625 = 0.4375
        assert 0.0 < iou < 0.5


# =============================================================================
# Submetric tests
# =============================================================================


class TestHeaderCellCountScore:
    def test_identical(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        assert _header_cell_count_score(gt, gt) == 1.0

    def test_no_headers_both(self):
        assert _header_cell_count_score([], []) == 1.0

    def test_missing_all(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        assert _header_cell_count_score(gt, []) == 0.0

    def test_extra_header(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)  # 2 headers
        pred, _, _ = _parse_header_cells(TABLE_EXTRA_HEADER_COL)  # 3 headers
        score = _header_cell_count_score(gt, pred)
        assert score == pytest.approx(2.0 / 3.0)


class TestHeaderContentBagScore:
    def test_identical(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        assert _header_content_bag_score(gt, gt) == 1.0

    def test_completely_different(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        pred, _, _ = _parse_header_cells(TABLE_DIFF_HEADER_TEXT)
        assert _header_content_bag_score(gt, pred) == 0.0

    def test_partial_match(self):
        gt, _, _ = _parse_header_cells(TABLE_EXTRA_HEADER_COL)  # Name, Value, Unit
        pred, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)  # Name, Value
        # GT has 3 cells, pred matches 2 of them
        score = _header_content_bag_score(gt, pred)
        assert score == pytest.approx(2.0 / 3.0)

    def test_formatting_normalized(self):
        """Bold/italic headers should match plain headers exactly."""
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        pred, _, _ = _parse_header_cells(TABLE_FORMATTED_HEADERS)
        assert _header_content_bag_score(gt, pred) == 1.0

    def test_strikethrough_normalized(self):
        """Strikethrough headers should match plain headers exactly."""
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        pred, _, _ = _parse_header_cells(TABLE_STRIKETHROUGH_HEADERS)
        assert _header_content_bag_score(gt, pred) == 1.0


class TestPerfectHeaderScore:
    def test_identical(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        assert _header_perfect_score(gt, gt) == 1.0

    def test_different_count(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        pred, _, _ = _parse_header_cells(TABLE_EXTRA_HEADER_COL)
        assert _header_perfect_score(gt, pred) == 0.0

    def test_same_structure_diff_text(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        pred, _, _ = _parse_header_cells(TABLE_DIFF_HEADER_TEXT)
        assert _header_perfect_score(gt, pred) == 0.0

    def test_same_headers_diff_body(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        pred, _, _ = _parse_header_cells(TABLE_SAME_HEADERS_DIFF_BODY)
        assert _header_perfect_score(gt, pred) == 1.0

    def test_wrong_colspan(self):
        gt, _, _ = _parse_header_cells(TABLE_MULTI_ROW_HEADER)
        pred, _, _ = _parse_header_cells(TABLE_MULTI_ROW_HEADER_WRONG_SPAN)
        assert _header_perfect_score(gt, pred) == 0.0


class TestHeaderBlockGritsScore:
    def test_identical(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        gt_blocks = _find_header_blocks(gt)
        gt_to_pred, grits_scores = _match_blocks(gt_blocks, gt_blocks)
        score = _header_grits_score(gt_blocks, gt_blocks, gt_to_pred, grits_scores)
        assert score == pytest.approx(1.0)

    def test_no_blocks(self):
        assert _header_grits_score([], [], {}, {}) == 1.0

    def test_missing_blocks(self):
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        gt_blocks = _find_header_blocks(gt)
        assert _header_grits_score(gt_blocks, [], {}, {}) == 0.0


class TestBlockEdgeDistance:
    def test_non_overlapping_horizontal(self):
        """Blocks separated horizontally: distance is column gap."""
        a = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=2)
        b = HeaderBlock(min_row=0, max_row=2, min_col=5, max_col=7)
        assert _block_edge_distance(a, b) == pytest.approx(3.0)

    def test_non_overlapping_vertical(self):
        """Blocks separated vertically: distance is row gap."""
        a = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=2)
        b = HeaderBlock(min_row=6, max_row=8, min_col=0, max_col=2)
        assert _block_edge_distance(a, b) == pytest.approx(4.0)

    def test_non_overlapping_diagonal(self):
        """Blocks separated diagonally: Euclidean distance of gaps."""
        a = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=2)
        b = HeaderBlock(min_row=5, max_row=7, min_col=5, max_col=7)
        # Gap: dr=3, dc=3
        assert _block_edge_distance(a, b) == pytest.approx((3**2 + 3**2) ** 0.5)

    def test_adjacent_blocks(self):
        """Blocks sharing an edge have distance 0."""
        a = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=3)
        b = HeaderBlock(min_row=2, max_row=4, min_col=0, max_col=3)
        assert _block_edge_distance(a, b) == pytest.approx(0.0)

    def test_overlapping_blocks(self):
        """Overlapping blocks have distance 0."""
        a = HeaderBlock(min_row=0, max_row=3, min_col=0, max_col=3)
        b = HeaderBlock(min_row=1, max_row=4, min_col=1, max_col=4)
        assert _block_edge_distance(a, b) == pytest.approx(0.0)

    def test_same_block(self):
        """Same block has distance 0."""
        a = HeaderBlock(min_row=2, max_row=5, min_col=2, max_col=5)
        assert _block_edge_distance(a, a) == pytest.approx(0.0)


class TestBlockEdgeVector:
    def test_b_below_and_right(self):
        a = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=2)
        b = HeaderBlock(min_row=5, max_row=7, min_col=5, max_col=7)
        dr, dc = _block_edge_vector(a, b)
        assert dr == pytest.approx(3.0)  # b is below a
        assert dc == pytest.approx(3.0)  # b is right of a

    def test_b_above_and_left(self):
        a = HeaderBlock(min_row=5, max_row=7, min_col=5, max_col=7)
        b = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=2)
        dr, dc = _block_edge_vector(a, b)
        assert dr == pytest.approx(-3.0)  # b is above a
        assert dc == pytest.approx(-3.0)  # b is left of a

    def test_overlapping_in_row(self):
        """Blocks overlapping in rows → dr=0, only column gap."""
        a = HeaderBlock(min_row=0, max_row=3, min_col=0, max_col=2)
        b = HeaderBlock(min_row=1, max_row=4, min_col=5, max_col=7)
        dr, dc = _block_edge_vector(a, b)
        assert dr == pytest.approx(0.0)  # overlapping rows
        assert dc == pytest.approx(3.0)

    def test_adjacent(self):
        """Adjacent blocks have zero vector."""
        a = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=3)
        b = HeaderBlock(min_row=2, max_row=4, min_col=0, max_col=3)
        dr, dc = _block_edge_vector(a, b)
        assert dr == pytest.approx(0.0)
        assert dc == pytest.approx(0.0)

    def test_vector_magnitude_matches_distance(self):
        """Edge vector magnitude should equal edge distance."""
        a = HeaderBlock(min_row=0, max_row=2, min_col=0, max_col=2)
        b = HeaderBlock(min_row=5, max_row=7, min_col=5, max_col=7)
        dr, dc = _block_edge_vector(a, b)
        assert (dr**2 + dc**2) ** 0.5 == pytest.approx(_block_edge_distance(a, b))


class TestDirectionSimilarity:
    def test_parallel_vectors(self):
        """Vectors in same direction → 1.0."""
        assert _direction_similarity(1.0, 0.0, 2.0, 0.0) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Vectors in opposite directions → 0.0."""
        assert _direction_similarity(1.0, 0.0, -1.0, 0.0) == pytest.approx(0.0)

    def test_perpendicular_vectors(self):
        """Perpendicular vectors → 0.5."""
        assert _direction_similarity(1.0, 0.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_zero_gt_vector(self):
        """Zero GT vector → 1.0 (direction irrelevant)."""
        assert _direction_similarity(0.0, 0.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_zero_pred_vector(self):
        """Zero pred vector → 1.0 (direction irrelevant)."""
        assert _direction_similarity(1.0, 1.0, 0.0, 0.0) == pytest.approx(1.0)

    def test_both_zero(self):
        """Both zero → 1.0."""
        assert _direction_similarity(0.0, 0.0, 0.0, 0.0) == pytest.approx(1.0)

    def test_diagonal_same_direction(self):
        """Same diagonal direction → 1.0."""
        assert _direction_similarity(3.0, 4.0, 6.0, 8.0) == pytest.approx(1.0)

    def test_diagonal_opposite(self):
        """Opposite diagonal → 0.0."""
        assert _direction_similarity(3.0, 4.0, -3.0, -4.0) == pytest.approx(0.0)


class TestHeaderBlockRelativePositionScore:
    def test_single_block(self):
        """Single block should return all 1.0."""
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        gt_blocks = _find_header_blocks(gt)
        gt_to_pred, _ = _match_blocks(gt_blocks, gt_blocks)
        prox, dirn = _header_block_relative_position_score(gt_blocks, gt_blocks, gt_to_pred, 3, 2, 3, 2)
        assert prox == 1.0
        assert dirn == 1.0

    def test_identical_multi_block(self):
        """Identical multi-block table should score 1.0."""
        gt, rows, cols = _parse_header_cells(TABLE_TWO_HEADER_BLOCKS)
        gt_blocks = _find_header_blocks(gt)
        gt_to_pred, _ = _match_blocks(gt_blocks, gt_blocks)
        prox, dirn = _header_block_relative_position_score(gt_blocks, gt_blocks, gt_to_pred, rows, cols, rows, cols)
        assert prox == pytest.approx(1.0)
        assert dirn == pytest.approx(1.0)

    def test_no_matched_blocks(self):
        """GT has blocks but pred has none — block count mismatch returns 0."""
        gt, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        gt_blocks = _find_header_blocks(gt)
        prox, dirn = _header_block_relative_position_score(gt_blocks, [], {}, 3, 2, 3, 2)
        assert prox == 0.0

    def test_swapped_direction_penalised(self):
        """Blocks with swapped direction should get low direction score."""
        # GT: block A at top-left, block B at bottom-right
        gt_a = HeaderBlock(
            cells=[HeaderCell(text="a", row=0, col=0, rowspan=1, colspan=1)],
            min_row=0,
            max_row=1,
            min_col=0,
            max_col=1,
        )
        gt_b = HeaderBlock(
            cells=[HeaderCell(text="b", row=5, col=5, rowspan=1, colspan=1)],
            min_row=5,
            max_row=6,
            min_col=5,
            max_col=6,
        )
        # Pred: block A at bottom-right, block B at top-left (swapped)
        pred_a = HeaderBlock(
            cells=[HeaderCell(text="a", row=5, col=5, rowspan=1, colspan=1)],
            min_row=5,
            max_row=6,
            min_col=5,
            max_col=6,
        )
        pred_b = HeaderBlock(
            cells=[HeaderCell(text="b", row=0, col=0, rowspan=1, colspan=1)],
            min_row=0,
            max_row=1,
            min_col=0,
            max_col=1,
        )
        prox, dirn = _header_block_relative_position_score(
            [gt_a, gt_b],
            [pred_a, pred_b],
            {0: 0, 1: 1},
            6,
            6,
            6,
            6,
        )
        # Direction should be 0.0 (opposite), proximity ~1.0 (same distance)
        assert dirn == pytest.approx(0.0)
        assert prox == pytest.approx(1.0)


# =============================================================================
# Composite score tests
# =============================================================================


class TestComputeHeaderAccuracyForTablePair:
    def test_identical_tables(self):
        scores = compute_header_composite_for_table_pair(TABLE_SIMPLE_HEADERS, TABLE_SIMPLE_HEADERS)
        assert scores["header_composite_v3"] == pytest.approx(1.0)
        assert scores["header_perfect"] == 1.0
        assert scores["header_content_bag"] == 1.0
        assert scores["header_cell_count"] == 1.0
        assert scores["header_block_proximity"] == pytest.approx(1.0)
        assert scores["header_block_relative_direction"] == pytest.approx(1.0)

    def test_no_headers_both(self):
        scores = compute_header_composite_for_table_pair(TABLE_NO_HEADERS, TABLE_NO_HEADERS)
        assert scores["header_composite_v3"] == pytest.approx(1.0)

    def test_headers_vs_no_headers(self):
        scores = compute_header_composite_for_table_pair(TABLE_SIMPLE_HEADERS, TABLE_NO_HEADERS)
        assert scores["header_cell_count"] == 0.0
        assert scores["header_content_bag"] == 0.0
        assert scores["header_perfect"] == 0.0
        assert scores["header_composite_v3"] < 0.5

    def test_same_headers_diff_body(self):
        scores = compute_header_composite_for_table_pair(TABLE_SIMPLE_HEADERS, TABLE_SAME_HEADERS_DIFF_BODY)
        assert scores["header_perfect"] == 1.0
        assert scores["header_content_bag"] == 1.0
        assert scores["header_composite_v3"] == pytest.approx(1.0)

    def test_multi_row_header_identical(self):
        scores = compute_header_composite_for_table_pair(TABLE_MULTI_ROW_HEADER, TABLE_MULTI_ROW_HEADER_SAME)
        assert scores["header_perfect"] == 1.0
        assert scores["header_composite_v3"] == pytest.approx(1.0)

    def test_has_all_submetrics(self):
        scores = compute_header_composite_for_table_pair(TABLE_SIMPLE_HEADERS, TABLE_SIMPLE_HEADERS)
        for key in SUBMETRIC_KEYS:
            assert key in scores, f"Missing submetric: {key}"


# =============================================================================
# Metric class tests
# =============================================================================


def _find_metric(results, name):
    for r in results:
        if r.metric_name == name:
            return r
    raise AssertionError(f"Metric '{name}' not found in {[r.metric_name for r in results]}")


class TestHeaderAccuracyMetric:
    def setup_method(self):
        self.metric = HeaderAccuracyMetric()

    def test_name(self):
        assert self.metric.name == "header_composite_v3"

    def test_identical_tables(self):
        results = self.metric.compute(expected=TABLE_SIMPLE_HEADERS, actual=TABLE_SIMPLE_HEADERS)
        ha = _find_metric(results, "header_composite_v3")
        assert ha.value == pytest.approx(1.0)

    def test_no_tables_in_expected(self):
        results = self.metric.compute(expected="no tables", actual=TABLE_SIMPLE_HEADERS)
        ha = _find_metric(results, "header_composite_v3")
        assert ha.value == 0.0

    def test_no_tables_in_actual(self):
        results = self.metric.compute(expected=TABLE_SIMPLE_HEADERS, actual="no tables")
        ha = _find_metric(results, "header_composite_v3")
        assert ha.value == 0.0

    def test_multiple_tables(self):
        expected = f"{TABLE_SIMPLE_HEADERS}\n{TABLE_MULTI_ROW_HEADER}"
        actual = f"{TABLE_MULTI_ROW_HEADER}\n{TABLE_SIMPLE_HEADERS}"
        results = self.metric.compute(expected=expected, actual=actual)
        ha = _find_metric(results, "header_composite_v3")
        # Hungarian matching should pair them correctly
        assert ha.value == pytest.approx(1.0)

    def test_returns_submetrics(self):
        results = self.metric.compute(expected=TABLE_SIMPLE_HEADERS, actual=TABLE_SIMPLE_HEADERS)
        names = {r.metric_name for r in results}
        for key in SUBMETRIC_KEYS:
            assert key in names, f"Missing submetric result: {key}"

    def test_value_range(self):
        for actual in [TABLE_SIMPLE_HEADERS, TABLE_NO_HEADERS, TABLE_DIFF_HEADER_TEXT]:
            results = self.metric.compute(expected=TABLE_SIMPLE_HEADERS, actual=actual)
            for r in results:
                assert 0.0 <= r.value <= 1.0, f"{r.metric_name} out of range: {r.value}"

    def test_table_pairs_parameter(self):
        """When table_pairs is provided, use those pairs directly."""
        pairs = [(TABLE_SIMPLE_HEADERS, TABLE_SIMPLE_HEADERS)]
        results = self.metric.compute(expected="ignored", actual="ignored", table_pairs=pairs)
        ha = _find_metric(results, "header_composite_v3")
        assert ha.value == pytest.approx(1.0)

    def test_table_pairs_with_unmatched(self):
        """Unmatched GT tables (empty pred) should score 0."""
        pairs = [(TABLE_SIMPLE_HEADERS, "")]
        results = self.metric.compute(expected="ignored", actual="ignored", table_pairs=pairs)
        ha = _find_metric(results, "header_composite_v3")
        assert ha.value == 0.0


# =============================================================================
# Block detail format tests
# =============================================================================


class TestBlockDetailFormat:
    def test_block_details_show_unique_cells(self):
        """Block details should show unique cell texts when available."""
        _, details = _detailed_header_composite_for_table_pair(TABLE_SIMPLE_HEADERS, TABLE_SIMPLE_HEADERS)
        grits_lines = details["header_grits"]
        detail_text = "\n".join(grits_lines)
        assert "expected [" in detail_text
        assert "predicted [" in detail_text

    def test_block_details_fallback_to_position(self):
        """Block details should fall back to row/col ranges when cells have no text."""
        # Build two tables with empty header text — triggers positional fallback
        gt_html = "<table><tr><th></th><th></th></tr><tr><td>A</td><td>B</td></tr></table>"
        pred_html = "<table><tr><th></th><th></th></tr><tr><td>C</td><td>D</td></tr></table>"
        _, details = _detailed_header_composite_for_table_pair(gt_html, pred_html)
        grits_lines = details["header_grits"]
        detail_text = "\n".join(grits_lines)
        # Fallback path should show positional info instead of cell texts
        assert "rows [" in detail_text
        assert "cols [" in detail_text
        assert "expected [" not in detail_text


# =============================================================================
# Extra header penalty tests (block_relative_position)
# =============================================================================


class TestRelativePositionExtraHeaderPenalty:
    """block_relative_position should penalise extra predicted header blocks."""

    def _make_block(self, text: str, row: int, col: int) -> HeaderBlock:
        cell = HeaderCell(text=text, row=row, col=col, rowspan=1, colspan=1)
        return HeaderBlock(
            cells=[cell],
            min_row=row,
            max_row=row + 1,
            min_col=col,
            max_col=col + 1,
        )

    def test_identical_two_blocks(self):
        """Two identical blocks → 1.0."""
        gt = [self._make_block("a", 0, 0), self._make_block("b", 5, 5)]
        prox, dirn = _header_block_relative_position_score(gt, gt, {0: 0, 1: 1}, 6, 6, 6, 6)
        assert prox == pytest.approx(1.0)
        assert dirn == pytest.approx(1.0)

    def test_extra_pred_blocks_penalise(self):
        """Extra pred blocks lower score via denominator."""
        gt = [self._make_block("a", 0, 0), self._make_block("b", 5, 5)]
        pred = [
            self._make_block("a", 0, 0),
            self._make_block("b", 5, 5),
            self._make_block("c", 10, 10),
        ]
        prox, dirn = _header_block_relative_position_score(gt, pred, {0: 0, 1: 1}, 11, 11, 11, 11)
        # 1 matched pair scores 1.0 each, denominator = max(1, 3) = 3
        assert prox == pytest.approx(1.0 / 3.0)
        assert dirn == pytest.approx(1.0 / 3.0)

    def test_one_gt_two_pred_blocks(self):
        """1 GT block, 2 pred blocks → 0 matched pairs / 1 total → score=0."""
        gt = [self._make_block("a", 0, 0)]
        pred = [self._make_block("a", 0, 0), self._make_block("x", 5, 5)]
        prox, dirn = _header_block_relative_position_score(gt, pred, {0: 0}, 6, 6, 6, 6)
        # gt_pairs=0, pred_pairs=1, total=1, 0 matched pairs → 0/1
        assert prox == pytest.approx(0.0)

    def test_no_extra_blocks_perfect(self):
        """Same number of blocks, identical positions → 1.0."""
        gt = [
            self._make_block("a", 0, 0),
            self._make_block("b", 0, 5),
            self._make_block("c", 5, 0),
        ]
        prox, dirn = _header_block_relative_position_score(gt, gt, {0: 0, 1: 1, 2: 2}, 6, 6, 6, 6)
        assert prox == pytest.approx(1.0)
        assert dirn == pytest.approx(1.0)


# =============================================================================
# Header hierarchy depth tests
# =============================================================================


class TestHeaderHierarchyDepth:
    def test_no_headers(self):
        assert _header_hierarchy_depth([]) == 0

    def test_single_row_no_colspan(self):
        """Single row of simple headers → depth 1."""
        cells = [
            HeaderCell(text="a", row=0, col=0, rowspan=1, colspan=1),
            HeaderCell(text="b", row=0, col=1, rowspan=1, colspan=1),
        ]
        assert _header_hierarchy_depth(cells) == 1

    def test_two_level_hierarchy(self):
        """Spanning header + leaf headers → depth 2."""
        cells = [
            HeaderCell(text="group", row=0, col=0, rowspan=1, colspan=2),
            HeaderCell(text="a", row=1, col=0, rowspan=1, colspan=1),
            HeaderCell(text="b", row=1, col=1, rowspan=1, colspan=1),
        ]
        assert _header_hierarchy_depth(cells) == 2

    def test_three_level_hierarchy(self):
        """Three levels of nesting → depth 3."""
        cells = [
            HeaderCell(text="top", row=0, col=0, rowspan=1, colspan=4),
            HeaderCell(text="mid1", row=1, col=0, rowspan=1, colspan=2),
            HeaderCell(text="mid2", row=1, col=2, rowspan=1, colspan=2),
            HeaderCell(text="a", row=2, col=0, rowspan=1, colspan=1),
            HeaderCell(text="b", row=2, col=1, rowspan=1, colspan=1),
            HeaderCell(text="c", row=2, col=2, rowspan=1, colspan=1),
            HeaderCell(text="d", row=2, col=3, rowspan=1, colspan=1),
        ]
        assert _header_hierarchy_depth(cells) == 3

    def test_rowspan_does_not_add_depth(self):
        """A cell with rowspan=2 + colspan=2 spanning two rows, with leaf cells
        in the rows it spans, should still be depth 2 (not 3).

        row 0: <th rowspan="2" colspan="2">A</th> <th colspan="2">B</th>
        row 1:                                     <th>C</th> <th>D</th>
        row 2: <th>E</th> <th>F</th>
        """
        cells = [
            HeaderCell(text="A", row=0, col=0, rowspan=2, colspan=2),
            HeaderCell(text="B", row=0, col=2, rowspan=1, colspan=2),
            HeaderCell(text="C", row=1, col=2, rowspan=1, colspan=1),
            HeaderCell(text="D", row=1, col=3, rowspan=1, colspan=1),
            HeaderCell(text="E", row=2, col=0, rowspan=1, colspan=1),
            HeaderCell(text="F", row=2, col=1, rowspan=1, colspan=1),
        ]
        assert _header_hierarchy_depth(cells) == 2

    def test_from_html_two_level(self):
        """Parse actual HTML and check depth."""
        cells, _, _ = _parse_header_cells(TABLE_MULTI_ROW_HEADER)
        assert _header_hierarchy_depth(cells) == 2

    def test_from_html_single_row(self):
        """Simple single-row header → depth 1."""
        cells, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        assert _header_hierarchy_depth(cells) == 1


class TestHeaderHierarchyDepthScore:
    def test_both_zero(self):
        assert _header_hierarchy_depth_score([], []) == 1.0

    def test_same_depth(self):
        cells = [
            HeaderCell(text="group", row=0, col=0, rowspan=1, colspan=2),
            HeaderCell(text="a", row=1, col=0, rowspan=1, colspan=1),
            HeaderCell(text="b", row=1, col=1, rowspan=1, colspan=1),
        ]
        assert _header_hierarchy_depth_score(cells, cells) == 1.0

    def test_depth_2_vs_1(self):
        gt = [
            HeaderCell(text="group", row=0, col=0, rowspan=1, colspan=2),
            HeaderCell(text="a", row=1, col=0, rowspan=1, colspan=1),
            HeaderCell(text="b", row=1, col=1, rowspan=1, colspan=1),
        ]
        pred = [
            HeaderCell(text="a", row=0, col=0, rowspan=1, colspan=1),
            HeaderCell(text="b", row=0, col=1, rowspan=1, colspan=1),
        ]
        # min(2,1)/max(2,1) = 0.5
        assert _header_hierarchy_depth_score(gt, pred) == pytest.approx(0.5)

    def test_gt_zero_pred_nonzero(self):
        pred = [HeaderCell(text="a", row=0, col=0, rowspan=1, colspan=1)]
        assert _header_hierarchy_depth_score([], pred) == 0.0

    def test_gt_nonzero_pred_zero(self):
        gt = [HeaderCell(text="a", row=0, col=0, rowspan=1, colspan=1)]
        assert _header_hierarchy_depth_score(gt, []) == 0.0

    def test_depth_3_vs_1(self):
        gt = [
            HeaderCell(text="top", row=0, col=0, rowspan=1, colspan=4),
            HeaderCell(text="mid", row=1, col=0, rowspan=1, colspan=2),
            HeaderCell(text="mid2", row=1, col=2, rowspan=1, colspan=2),
            HeaderCell(text="a", row=2, col=0, rowspan=1, colspan=1),
            HeaderCell(text="b", row=2, col=1, rowspan=1, colspan=1),
            HeaderCell(text="c", row=2, col=2, rowspan=1, colspan=1),
            HeaderCell(text="d", row=2, col=3, rowspan=1, colspan=1),
        ]
        pred = [
            HeaderCell(text="a", row=0, col=0, rowspan=1, colspan=1),
            HeaderCell(text="b", row=0, col=1, rowspan=1, colspan=1),
        ]
        # min(3,1)/max(3,1) = 1/3
        assert _header_hierarchy_depth_score(gt, pred) == pytest.approx(1.0 / 3.0)


# =============================================================================
# Malformed HTML parsing tests (lxml robustness)
# =============================================================================


TABLE_MALFORMED_TH_TD_MISMATCH = """<table>
<tr><th>Name</th><th>Value</td></tr>
<tr><td>Alpha</td><td>100</td></tr>
</table>"""


class TestMalformedHtmlParsing:
    def test_mismatched_th_td_tags(self):
        """lxml should handle <th>...</td> mismatches gracefully."""
        cells, rows, cols = _parse_header_cells(TABLE_MALFORMED_TH_TD_MISMATCH)
        texts = {c.text for c in cells}
        assert "name" in texts
        assert "value" in texts


# =============================================================================
# Header data alignment tests
# =============================================================================


class TestHeaderDataAlignment:
    """Tests for the header_data_alignment submetric."""

    def test_perfect_alignment(self):
        """Identical tables -> GriTS aligns perfectly -> score 1.0."""
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        result = grits_from_html(TABLE_SIMPLE_HEADERS, TABLE_SIMPLE_HEADERS)
        assert result is not None
        row_map = result["_con_row_alignment"]
        col_map = result["_con_col_alignment"]

        gt_cells, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        pred_text_lookup = _build_text_lookup(TABLE_SIMPLE_HEADERS)
        score = _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)
        assert score == 1.0

    def test_extra_td_in_header_row(self):
        """Extra <td> in header row displaces headers -> score 1/3."""
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        result = grits_from_html(TABLE_3COL, TABLE_3COL_EXTRA_TD)
        assert result is not None
        row_map = result["_con_row_alignment"]
        col_map = result["_con_col_alignment"]

        gt_cells, _, _ = _parse_header_cells(TABLE_3COL)
        pred_text_lookup = _build_text_lookup(TABLE_3COL_EXTRA_TD)
        score = _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)
        assert score == pytest.approx(1 / 3, abs=0.01)

    def test_extra_th_in_header_row(self):
        """Extra <th> in header row -- same displacement, same score as extra <td>."""
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        result = grits_from_html(TABLE_3COL, TABLE_3COL_EXTRA_TH)
        assert result is not None
        row_map = result["_con_row_alignment"]
        col_map = result["_con_col_alignment"]

        gt_cells, _, _ = _parse_header_cells(TABLE_3COL)
        pred_text_lookup = _build_text_lookup(TABLE_3COL_EXTRA_TH)
        score = _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)
        assert score == pytest.approx(1 / 3, abs=0.01)

    def test_headers_converted_to_td(self):
        """All <th> converted to <td> but text/positions unchanged -> 1.0."""
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        result = grits_from_html(TABLE_3COL, TABLE_3COL_NO_TH)
        assert result is not None
        row_map = result["_con_row_alignment"]
        col_map = result["_con_col_alignment"]

        gt_cells, _, _ = _parse_header_cells(TABLE_3COL)
        pred_text_lookup = _build_text_lookup(TABLE_3COL_NO_TH)
        score = _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)
        assert score == 1.0

    def test_no_gt_headers(self):
        """No GT headers -> 1.0 (vacuously correct)."""
        score = _header_data_alignment_score([], {}, {0: 0}, {0: 0})
        assert score == 1.0

    def test_empty_alignment_maps(self):
        """Empty alignment maps -> 0.0."""
        gt_cells, _, _ = _parse_header_cells(TABLE_SIMPLE_HEADERS)
        pred_text_lookup = _build_text_lookup(TABLE_SIMPLE_HEADERS)
        score = _header_data_alignment_score(gt_cells, pred_text_lookup, {}, {})
        assert score == 0.0

    def test_grits_emits_alignment_keys(self):
        """grits_from_html result contains _con_row/col_alignment keys."""
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        result = grits_from_html(TABLE_SIMPLE_HEADERS, TABLE_SIMPLE_HEADERS)
        assert result is not None
        assert "_con_row_alignment" in result
        assert "_con_col_alignment" in result
        assert isinstance(result["_con_row_alignment"], dict)
        assert isinstance(result["_con_col_alignment"], dict)

    def test_end_to_end_with_grits_alignment(self):
        """End-to-end: GriTS alignment -> _detailed_header_composite_for_table_pair."""
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        gt = TABLE_SIMPLE_HEADERS
        pred = TABLE_SAME_HEADERS_DIFF_BODY
        grits_result = grits_from_html(gt, pred)
        assert grits_result is not None

        scores, details = _detailed_header_composite_for_table_pair(
            gt,
            pred,
            row_map=grits_result["_con_row_alignment"],
            col_map=grits_result["_con_col_alignment"],
        )
        assert "header_data_alignment" in scores
        assert scores["header_data_alignment"] == 1.0
        assert "grits" in details["header_data_alignment"][0].lower()

    def test_fallback_path_emits_standalone_detail(self):
        """Without GriTS alignment, fallback computes its own and labels it."""
        scores, details = _detailed_header_composite_for_table_pair(
            TABLE_SIMPLE_HEADERS,
            TABLE_SIMPLE_HEADERS,
            row_map=None,
            col_map=None,
        )
        assert "header_data_alignment" in scores
        assert scores["header_data_alignment"] == 1.0
        assert "standalone" in details["header_data_alignment"][0].lower()

    def test_metric_class_alignment_source_metadata(self):
        """HeaderAccuracyMetric.compute() records alignment_source in metadata."""
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        gt = TABLE_SIMPLE_HEADERS
        grits_result = grits_from_html(gt, gt)
        assert grits_result is not None
        row_map = grits_result["_con_row_alignment"]
        col_map = grits_result["_con_col_alignment"]

        metric = HeaderAccuracyMetric()

        # With GriTS alignment
        results = metric.compute(
            expected=gt,
            actual=gt,
            table_pairs=[(gt, gt)],
            table_alignments=[(row_map, col_map)],
        )
        alignment_mv = [r for r in results if r.metric_name == "header_data_alignment"]
        assert len(alignment_mv) == 1
        assert alignment_mv[0].value == 1.0
        composite_mv = [r for r in results if r.metric_name == "header_composite_v3"]
        assert composite_mv[0].metadata.get("alignment_source") == "grits"

        # Without alignment -> fallback
        results_fb = metric.compute(
            expected=gt,
            actual=gt,
            table_pairs=[(gt, gt)],
            table_alignments=None,
        )
        composite_fb = [r for r in results_fb if r.metric_name == "header_composite_v3"]
        assert composite_fb[0].metadata.get("alignment_source") == "fallback"

    def test_in_composite_keys(self):
        """header_data_alignment is included in _COMPOSITE_KEYS."""
        from parse_bench.evaluation.metrics.parse.header_accuracy_metric import (
            _COMPOSITE_KEYS,
        )

        assert "header_data_alignment" in _COMPOSITE_KEYS

    def test_two_extra_cells_spread_across_header(self):
        """Two extra empty cells at different points in the header row.

        Extra cells after "Name" and after "Age" displace later headers.
        GriTS col alignment is data-dominated (2 data rows vs 1 header row),
        so col 0->0, col 1->1, col 2->2.

        Header text at mapped positions:
          "name" at (0,0) -> pred (0,0) = "name" -> match
          "age"  at (0,1) -> pred (0,1) = ""     -> mismatch
          "city" at (0,2) -> pred (0,2) = "age"  -> mismatch
        Score = 1/3.
        """
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        result = grits_from_html(TABLE_3COL, TABLE_3COL_TWO_EXTRA_SPREAD)
        assert result is not None
        row_map = result["_con_row_alignment"]
        col_map = result["_con_col_alignment"]

        gt_cells, _, _ = _parse_header_cells(TABLE_3COL)
        pred_text_lookup = _build_text_lookup(TABLE_3COL_TWO_EXTRA_SPREAD)
        score = _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)
        assert score == pytest.approx(1 / 3, abs=0.01)

    def test_two_extra_cells_leading(self):
        """Two extra empty cells before the first header.

        All data and header content shifts right uniformly, so GriTS
        aligns col 0->2, col 1->3, col 2->4. All header text matches
        at the mapped positions -> score = 1.0.
        """
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        result = grits_from_html(TABLE_3COL, TABLE_3COL_TWO_EXTRA_LEADING)
        assert result is not None
        row_map = result["_con_row_alignment"]
        col_map = result["_con_col_alignment"]

        gt_cells, _, _ = _parse_header_cells(TABLE_3COL)
        pred_text_lookup = _build_text_lookup(TABLE_3COL_TWO_EXTRA_LEADING)
        score = _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)
        assert score == 1.0

    def test_two_extra_cells_trailing(self):
        """Two extra empty cells at the end of each row.

        No displacement of existing content, so GriTS aligns
        col 0->0, col 1->1, col 2->2. All headers match -> score = 1.0.
        """
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        result = grits_from_html(TABLE_3COL, TABLE_3COL_TWO_EXTRA_TRAILING)
        assert result is not None
        row_map = result["_con_row_alignment"]
        col_map = result["_con_col_alignment"]

        gt_cells, _, _ = _parse_header_cells(TABLE_3COL)
        pred_text_lookup = _build_text_lookup(TABLE_3COL_TWO_EXTRA_TRAILING)
        score = _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)
        assert score == 1.0

    def test_extra_row_before_header(self):
        """Extra empty row before header row shifts headers down.

        GriTS row alignment maps row 0->1 (content match outweighs
        the empty row), so headers at mapped positions still match.
        Score = 1.0.
        """
        from parse_bench.evaluation.metrics.parse.grits_metric import (
            grits_from_html,
        )

        result = grits_from_html(TABLE_3COL, TABLE_3COL_EXTRA_ROW)
        assert result is not None
        row_map = result["_con_row_alignment"]
        col_map = result["_con_col_alignment"]

        gt_cells, _, _ = _parse_header_cells(TABLE_3COL)
        pred_text_lookup = _build_text_lookup(TABLE_3COL_EXTRA_ROW)
        score = _header_data_alignment_score(gt_cells, pred_text_lookup, row_map, col_map)
        assert score == 1.0


# =============================================================================
# Tests for generous header normalization
# =============================================================================


class TestPromoteTopRowToHeader:
    """Tests for _promote_top_row_to_header."""

    def test_basic(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        result = _promote_top_row_to_header(html)
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(result, "lxml")
        rows = soup.find_all("tr")
        assert all(c.name == "th" for c in rows[0].find_all(["th", "td"]))
        assert all(c.name == "td" for c in rows[1].find_all(["th", "td"]))

    def test_no_rows(self):
        html = "<table></table>"
        result = _promote_top_row_to_header(html)
        assert "table" in result

    def test_already_th(self):
        html = "<table><tr><th>A</th></tr></table>"
        result = _promote_top_row_to_header(html)
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(result, "lxml")
        rows = soup.find_all("tr")
        assert all(c.name == "th" for c in rows[0].find_all(["th", "td"]))


class TestApplyGenerousNormalization:
    """Tests for _apply_generous_header_normalization."""

    def test_gt_no_headers(self):
        """When GT has no headers, pred is returned unchanged."""
        gt = "<table><tr><td>A</td></tr></table>"
        pred = "<table><tr><td>X</td></tr></table>"
        result = _apply_generous_header_normalization(gt, pred)
        assert result == pred

    def test_pred_has_headers(self):
        """When pred already has headers, pred is returned unchanged."""
        gt = TABLE_SIMPLE_HEADERS
        pred = TABLE_SIMPLE_HEADERS
        result = _apply_generous_header_normalization(gt, pred)
        assert result == pred

    def test_promotes(self):
        """When GT has headers and pred has none, top row is promoted."""
        gt = TABLE_SIMPLE_HEADERS
        pred = TABLE_NO_HEADERS
        result = _apply_generous_header_normalization(gt, pred)
        cells, _, _ = _parse_header_cells(result)
        assert len(cells) > 0

    def test_both_no_headers(self):
        """When both GT and pred have no headers, pred is returned unchanged."""
        gt = TABLE_NO_HEADERS
        pred = TABLE_NO_HEADERS
        result = _apply_generous_header_normalization(gt, pred)
        assert result == pred

    def test_gt_no_headers_pred_has_headers(self):
        """When GT has no headers but pred does, pred is returned unchanged."""
        gt = TABLE_NO_HEADERS
        pred = TABLE_SIMPLE_HEADERS
        result = _apply_generous_header_normalization(gt, pred)
        assert result == pred


class TestHeaderAccuracyMetricGenerous:
    """Tests for HeaderAccuracyMetricGenerous."""

    def test_returns_single_composite(self):
        """compute() returns exactly one MetricValue named header_composite_v3_generous."""
        metric = HeaderAccuracyMetricGenerous()
        results = metric.compute(
            expected=TABLE_SIMPLE_HEADERS,
            actual=TABLE_SIMPLE_HEADERS,
            table_pairs=[(TABLE_SIMPLE_HEADERS, TABLE_SIMPLE_HEADERS)],
        )
        assert len(results) == 1
        assert results[0].metric_name == "exp_header_composite_v3_generous"

    def test_equals_base_when_pred_has_headers(self):
        """When pred has headers, generous == base (normalization is a no-op)."""
        base = HeaderAccuracyMetric()
        generous = HeaderAccuracyMetricGenerous()
        pairs = [(TABLE_SIMPLE_HEADERS, TABLE_SIMPLE_HEADERS)]
        base_results = base.compute(
            expected=TABLE_SIMPLE_HEADERS,
            actual=TABLE_SIMPLE_HEADERS,
            table_pairs=pairs,
        )
        gen_results = generous.compute(
            expected=TABLE_SIMPLE_HEADERS,
            actual=TABLE_SIMPLE_HEADERS,
            table_pairs=pairs,
        )
        base_composite = next(r for r in base_results if r.metric_name == "header_composite_v3")
        gen_composite = gen_results[0]
        assert gen_composite.value == pytest.approx(base_composite.value)

    def test_generous_gte_base_when_pred_missing_headers(self):
        """When pred has no headers but GT does, generous >= base."""
        base = HeaderAccuracyMetric()
        generous = HeaderAccuracyMetricGenerous()
        pairs = [(TABLE_SIMPLE_HEADERS, TABLE_NO_HEADERS)]
        base_results = base.compute(
            expected=TABLE_SIMPLE_HEADERS,
            actual=TABLE_NO_HEADERS,
            table_pairs=pairs,
        )
        gen_results = generous.compute(
            expected=TABLE_SIMPLE_HEADERS,
            actual=TABLE_NO_HEADERS,
            table_pairs=pairs,
        )
        base_composite = next(r for r in base_results if r.metric_name == "header_composite_v3")
        gen_composite = gen_results[0]
        assert gen_composite.value >= base_composite.value

    def test_end_to_end_top_and_bottom_left_headers(self):
        """Table with top headers + bottom-left row headers in GT, pred has only <td>.

        Generous score should be higher than base score because both top-row
        and bottom-left promotions fire.
        """
        gt = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><th>Alpha</th><td>100</td></tr>
<tr><th>Beta</th><td>200</td></tr>
</table>"""
        pred = """<table>
<tr><td>Name</td><td>Value</td></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>"""
        base = HeaderAccuracyMetric()
        generous = HeaderAccuracyMetricGenerous()
        pairs = [(gt, pred)]
        base_results = base.compute(expected=gt, actual=pred, table_pairs=pairs)
        gen_results = generous.compute(expected=gt, actual=pred, table_pairs=pairs)
        base_composite = next(r for r in base_results if r.metric_name == "header_composite_v3")
        gen_composite = gen_results[0]
        assert gen_composite.value > base_composite.value


# =============================================================================
# Tests for _is_bottom_left_block
# =============================================================================


class TestIsBottomLeftBlock:
    def test_rectangular_bottom_left(self):
        """Rectangular block at col 0, rows 2–4 in a 4-row table → True."""
        block = HeaderBlock(
            cells=[
                HeaderCell(text="a", row=2, col=0, rowspan=1, colspan=1),
                HeaderCell(text="b", row=3, col=0, rowspan=1, colspan=1),
            ],
            min_row=2,
            max_row=4,
            min_col=0,
            max_col=1,
        )
        assert _is_bottom_left_block(block, num_rows=4) is True

    def test_touches_top_row(self):
        """Block at col 0, rows 0–2 (touches top row) → False."""
        block = HeaderBlock(
            cells=[
                HeaderCell(text="a", row=0, col=0, rowspan=1, colspan=1),
                HeaderCell(text="b", row=1, col=0, rowspan=1, colspan=1),
            ],
            min_row=0,
            max_row=2,
            min_col=0,
            max_col=1,
        )
        assert _is_bottom_left_block(block, num_rows=2) is False

    def test_not_leftmost_col(self):
        """Block at col 1, rows 2–4 (not leftmost col) → False."""
        block = HeaderBlock(
            cells=[
                HeaderCell(text="a", row=2, col=1, rowspan=1, colspan=1),
                HeaderCell(text="b", row=3, col=1, rowspan=1, colspan=1),
            ],
            min_row=2,
            max_row=4,
            min_col=1,
            max_col=2,
        )
        assert _is_bottom_left_block(block, num_rows=4) is False

    def test_does_not_reach_bottom_row(self):
        """Block at col 0, rows 2–3 in a 4-row table (doesn't reach bottom) → False."""
        block = HeaderBlock(
            cells=[
                HeaderCell(text="a", row=2, col=0, rowspan=1, colspan=1),
            ],
            min_row=2,
            max_row=3,
            min_col=0,
            max_col=1,
        )
        assert _is_bottom_left_block(block, num_rows=4) is False

    def test_l_shaped_not_rectangular(self):
        """L-shaped block at col 0 rows 2–4 + col 1 row 3 only → False."""
        block = HeaderBlock(
            cells=[
                HeaderCell(text="a", row=2, col=0, rowspan=1, colspan=1),
                HeaderCell(text="b", row=3, col=0, rowspan=1, colspan=1),
                HeaderCell(text="c", row=3, col=1, rowspan=1, colspan=1),
            ],
            min_row=2,
            max_row=4,
            min_col=0,
            max_col=2,
        )
        # Expected area = 2*2=4, but only 3 positions occupied → not rectangular
        assert _is_bottom_left_block(block, num_rows=4) is False

    def test_rectangular_two_columns(self):
        """Rectangular block spanning 2 columns, rows 2–4 in a 4-row table → True."""
        block = HeaderBlock(
            cells=[
                HeaderCell(text="a", row=2, col=0, rowspan=1, colspan=1),
                HeaderCell(text="b", row=2, col=1, rowspan=1, colspan=1),
                HeaderCell(text="c", row=3, col=0, rowspan=1, colspan=1),
                HeaderCell(text="d", row=3, col=1, rowspan=1, colspan=1),
            ],
            min_row=2,
            max_row=4,
            min_col=0,
            max_col=2,
        )
        assert _is_bottom_left_block(block, num_rows=4) is True


# =============================================================================
# Tests for _find_contiguous_groups
# =============================================================================


class TestFindContiguousGroups:
    def test_single_position(self):
        groups = _find_contiguous_groups({(0, 0)})
        assert len(groups) == 1
        assert groups[0] == {(0, 0)}

    def test_two_adjacent(self):
        groups = _find_contiguous_groups({(0, 0), (0, 1)})
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_two_non_adjacent(self):
        groups = _find_contiguous_groups({(0, 0), (5, 5)})
        assert len(groups) == 2
        assert all(len(g) == 1 for g in groups)

    def test_l_shaped(self):
        groups = _find_contiguous_groups({(0, 0), (1, 0), (1, 1)})
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_empty(self):
        groups = _find_contiguous_groups(set())
        assert len(groups) == 0


# =============================================================================
# Tests for _promote_bottom_left_to_header
# =============================================================================


class TestPromoteBottomLeftToHeader:
    def test_matching_text_promoted(self):
        """GT has bottom-left header block with matching pred text → promoted.

        The GT must have a gap row between top headers and bottom-left block
        so they form separate blocks (otherwise they merge into one block
        touching the top row).
        """
        gt = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><th>Alpha</th><td>100</td></tr>
<tr><th>Beta</th><td>200</td></tr>
</table>"""
        pred = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        cells, _, _ = _parse_header_cells(result)
        # Should have promoted Alpha and Beta to <th>
        texts = {c.text for c in cells}
        assert "alpha" in texts
        assert "beta" in texts

    def test_low_similarity_no_promotion(self):
        """GT has bottom-left header block, pred text very different → no promotion."""
        gt = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><th>Alpha</th><td>100</td></tr>
<tr><th>Beta</th><td>200</td></tr>
</table>"""
        pred = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><td>XXXXX</td><td>100</td></tr>
<tr><td>YYYYY</td><td>200</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        cells, _, _ = _parse_header_cells(result)
        # Only top-row headers should exist, no promotion happened
        texts = {c.text for c in cells}
        assert "xxxxx" not in texts
        assert "yyyyy" not in texts

    def test_threshold_boundary_below(self):
        """Similarity just below 0.8 → no promotion."""
        # "abcde" vs "abcXY" → LCS="abc"=3, sim=2*3/(5+5)=0.6 < 0.8
        gt = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><th>abcde</th><td>100</td></tr>
</table>"""
        pred = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><td>abcXY</td><td>100</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        cells, _, _ = _parse_header_cells(result)
        texts = {c.text for c in cells}
        assert "abcxy" not in texts

    def test_threshold_boundary_above(self):
        """Similarity >= 0.8 → promotion happens."""
        # "abcde" vs "abcdf" → LCS="abcd"=4, sim=2*4/(5+5)=0.8 >= 0.8
        gt = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><th>abcde</th><td>100</td></tr>
</table>"""
        pred = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><td>abcdf</td><td>100</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        cells, _, _ = _parse_header_cells(result)
        texts = {c.text for c in cells}
        assert "abcdf" in texts

    def test_pred_already_has_bl_headers(self):
        """GT has bottom-left block, pred already has <th> there → no change."""
        gt = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><th>Alpha</th><td>100</td></tr>
<tr><th>Beta</th><td>200</td></tr>
</table>"""
        pred = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><th>Alpha</th><td>100</td></tr>
<tr><th>Beta</th><td>200</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        assert result == pred

    def test_no_bottom_left_block_in_gt(self):
        """GT has no bottom-left block → no promotion."""
        gt = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Alpha</td><td>100</td></tr>
</table>"""
        pred = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Alpha</td><td>100</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        assert result == pred

    def test_contiguity_selects_group_with_pred_bottom_left(self):
        """Matching cells in two non-contiguous groups; only the group
        containing (pred_bottom_row, 0) gets promoted."""
        # GT: 5 rows, bottom-left block spans rows 2-4 col 0
        gt = """<table>
<tr><th>H1</th><th>H2</th></tr>
<tr><td>data</td><td>data</td></tr>
<tr><th>Alpha</th><td>100</td></tr>
<tr><td>data</td><td>data</td></tr>
<tr><th>Beta</th><td>200</td></tr>
</table>"""
        # Pred: same layout, all <td>
        pred = """<table>
<tr><th>H1</th><th>H2</th></tr>
<tr><td>data</td><td>data</td></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>data</td><td>data</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        cells, _, _ = _parse_header_cells(result)
        texts = {c.text for c in cells}
        # "beta" should be promoted (bottom-left of pred)
        assert "beta" in texts

    def test_pred_bottom_left_no_match_no_promotion(self):
        """GT has bottom-left block but pred cell at (pred_bottom_row, 0)
        doesn't match → no promotion even if other cells match."""
        gt = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><th>Alpha</th><td>100</td></tr>
<tr><th>Beta</th><td>200</td></tr>
</table>"""
        pred = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>ZZZZZ</td><td>200</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        cells, _, _ = _parse_header_cells(result)
        texts = {c.text for c in cells}
        # Alpha matches but Beta/ZZZZZ don't; since (2,0) doesn't match,
        # and the contiguous group with (2,0) doesn't exist, nothing promoted
        assert "alpha" not in texts

    def test_truncated_pred(self):
        """GT has 5 rows, pred has 3 rows. GT bottom-left block spans rows 3–4.
        Pred has matching text at rows 1–2 col 0. The contiguous group containing
        (2, 0) (pred's bottom row) gets promoted."""
        gt = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>x</td><td>1</td></tr>
<tr><td>y</td><td>2</td></tr>
<tr><th>Alpha</th><td>100</td></tr>
<tr><th>Beta</th><td>200</td></tr>
</table>"""
        # Pred is truncated - only 3 rows, with Alpha/Beta at rows 1-2
        pred = """<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Alpha</td><td>100</td></tr>
<tr><td>Beta</td><td>200</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        # The GT bottom-left block is at rows 3-4 col 0, but pred rows 1-2 col 0
        # have matching text. However, positions (3,0) and (4,0) don't exist in pred,
        # so pred_text_lookup won't match them. This test documents the behavior:
        # since the GT block positions don't map to pred positions, no promotion.
        cells, _, _ = _parse_header_cells(result)
        # No promotion expected because the GT block's positions (3,0),(4,0)
        # don't exist in pred's text lookup
        texts = {c.text for c in cells}
        assert "name" in texts  # top-row headers still there

    def test_wider_pred_match(self):
        """GT has rectangular bottom-left block (rows 2–3, col 0).
        Pred has matching text at rows 2–3 cols 0–1 (wider than GT block).
        Contiguous group contains (pred_bottom_row, 0) → all matching cells promoted."""
        gt = """<table>
<tr><th>Name</th><th>Value</th><th>Extra</th></tr>
<tr><td>x</td><td>1</td><td>a</td></tr>
<tr><th>Alpha</th><td>100</td><td>b</td></tr>
<tr><th>Beta</th><td>200</td><td>c</td></tr>
</table>"""
        pred = """<table>
<tr><th>Name</th><th>Value</th><th>Extra</th></tr>
<tr><td>x</td><td>1</td><td>a</td></tr>
<tr><td>Alpha</td><td>100</td><td>b</td></tr>
<tr><td>Beta</td><td>200</td><td>c</td></tr>
</table>"""
        result = _promote_bottom_left_to_header(gt, pred)
        cells, _, _ = _parse_header_cells(result)
        texts = {c.text for c in cells}
        assert "alpha" in texts
        assert "beta" in texts
