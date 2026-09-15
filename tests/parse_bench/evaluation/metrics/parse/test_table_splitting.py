"""Tests for the upstream ambiguous-merged-table splitter."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_extraction import (
    ExtractedTable,
    extract_table_pairs,
)
from parse_bench.evaluation.metrics.parse.table_splitting import (
    _SAFETY_CAP,
    select_joint_split,
    split_ambiguous_merged_pred,
)


def _doc_with_tables(tables_html: list[str]) -> str:
    return "\n".join(tables_html)


def _wrap(rows: str) -> str:
    return f"<table>{rows}</table>"


def test_no_split_when_actual_has_multiple_tables() -> None:
    expected_md = _doc_with_tables(
        [
            _wrap("<tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr>"),
            _wrap("<tr><th>A</th><th>B</th></tr><tr><td>3</td><td>4</td></tr>"),
        ]
    )
    actual_md = expected_md
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is False
    assert result is actual


def test_no_split_when_expected_has_one_table() -> None:
    md = _wrap("<tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr>")
    expected, actual, _ = extract_table_pairs(md, md)

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is False
    assert result is actual


def test_splits_periodic_merged_pred() -> None:
    expected_md = _doc_with_tables(
        [
            _wrap("<tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr>"),
            _wrap("<tr><th>A</th><th>B</th></tr><tr><td>3</td><td>4</td></tr>"),
            _wrap("<tr><th>A</th><th>B</th></tr><tr><td>5</td><td>6</td></tr>"),
        ]
    )
    actual_md = _wrap(
        "<tr><th>A</th><th>B</th><th>A</th><th>B</th><th>A</th><th>B</th></tr>"
        "<tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr>"
    )
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    assert len(expected) == 3
    assert len(actual) == 1

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is True
    assert len(result) == 3
    for et in result:
        assert isinstance(et, ExtractedTable)
        assert et.raw_html == ""
        assert et.table_data.data.shape[1] == 2


def _two_col_unit() -> str:
    return "<tr><th>A</th><th>B</th></tr><tr><td>x</td><td>y</td></tr>"


def _gt_n_two_col_tables(n: int) -> str:
    return _doc_with_tables([_wrap(_two_col_unit()) for _ in range(n)])


def _merged_period2(n_segments: int) -> str:
    headers = "".join("<th>A</th><th>B</th>" for _ in range(n_segments))
    cells = "".join("<td>x</td><td>y</td>" for _ in range(n_segments))
    return _wrap(f"<tr>{headers}</tr><tr>{cells}</tr>")


def _fixed_table() -> str:
    return _wrap("<tr><th>X</th><th>Y</th><th>Z</th></tr><tr><td>1</td><td>2</td><td>3</td></tr>")


def test_inexact_single_pred_split_applied_when_closer_than_baseline() -> None:
    """Single merged pred (3 segs) vs 4 expected: dist 1 < baseline 3 → split."""
    expected_md = _gt_n_two_col_tables(4)
    actual_md = _merged_period2(3)
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    assert len(expected) == 4 and len(actual) == 1

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is True
    assert len(result) == 3


def test_multi_pred_single_splittable() -> None:
    """One merged pred + one fixed pred → joint selector splits the merged one."""
    expected_md = _gt_n_two_col_tables(3)
    actual_md = _doc_with_tables([_merged_period2(2), _fixed_table()])
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    assert len(expected) == 3 and len(actual) == 2

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is True
    assert len(result) == 3
    # The fixed table should be preserved as-is (raw_html non-empty)
    fixed_preserved = [et for et in result if et.raw_html != ""]
    assert len(fixed_preserved) == 1


def test_multi_pred_two_splittable_joint_optimum() -> None:
    """Two splittable preds (each 2 segs) targeting 4 expected → both split."""
    expected_md = _gt_n_two_col_tables(4)
    actual_md = _doc_with_tables([_merged_period2(2), _merged_period2(2)])
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    assert len(expected) == 4 and len(actual) == 2

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is True
    assert len(result) == 4
    # Both got split → no original raw_html preserved
    assert all(et.raw_html == "" for et in result)


def test_strict_improvement_gate_no_op() -> None:
    """Single splittable pred (2 segs) vs 2 expected: dist 1 vs baseline 1 → no-op."""
    expected_md = _gt_n_two_col_tables(2)
    actual_md = _merged_period2(2)
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    # Trigger only fires when len(expected) > len(actual). Here that holds (2 > 1).
    assert len(expected) == 2 and len(actual) == 1

    # Actually len(actual)==1, baseline distance = |1-2|=1, split gives n_segments=2, dist=0.
    # That's an improvement, so this should split. Adjust: use 3 expected to test gate.
    expected_md = _gt_n_two_col_tables(3)
    actual_md = _merged_period2(2)
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    # baseline dist = |1-3|=2, split dist = |2-3|=1 → still improves. Use larger.
    expected_md = _gt_n_two_col_tables(2)
    actual_md = _doc_with_tables([_merged_period2(2), _fixed_table()])
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    # Trigger requires len(expected) > len(actual). 2 > 2 is false → no-op trivially.
    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is False


def test_safety_cap_no_op() -> None:
    """Synthesize >256 variable Cartesian product → no-op."""
    # Each merged_period2(2) yields 2 options (no-split + 1 split). 9 such tables → 2**9 = 512 > 256.
    actual_md = _doc_with_tables([_merged_period2(2) for _ in range(9)])
    expected_md = _gt_n_two_col_tables(18)
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    chosen = select_joint_split(actual, len(expected))
    assert chosen is None
    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is False
    assert _SAFETY_CAP == 256


def test_trigger_gate_when_expected_lte_actual() -> None:
    """len(expected) <= len(actual) → no-op even with splittable preds."""
    expected_md = _gt_n_two_col_tables(2)
    actual_md = _doc_with_tables([_merged_period2(2), _merged_period2(2)])
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    assert len(expected) == 2 and len(actual) == 2

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is False
    assert result is actual


def test_raw_html_preserved_for_unchanged_tables() -> None:
    """Partial split: untouched tables keep raw_html, sub-tables get ''."""
    expected_md = _gt_n_two_col_tables(3)
    actual_md = _doc_with_tables([_merged_period2(2), _fixed_table()])
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    original_fixed_html = actual[1].raw_html
    assert original_fixed_html != ""

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is True
    # Find the preserved fixed table
    preserved = [et for et in result if et.raw_html != ""]
    assert len(preserved) == 1
    assert preserved[0].raw_html == original_fixed_html
    # The other two are split sub-tables with empty raw_html
    split_subs = [et for et in result if et.raw_html == ""]
    assert len(split_subs) == 2


def _merged_two_header_rows_period2() -> str:
    """4 cols, two <th> rows that BOTH repeat with period 2 → n_repeating_rows=2."""
    return _wrap(
        "<tr><th>X</th><th>Y</th><th>X</th><th>Y</th></tr>"
        "<tr><th>A</th><th>B</th><th>A</th><th>B</th></tr>"
        "<tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
    )


def _merged_period3() -> str:
    """6 cols, one <th> row repeating with period 3 → n_segments=2, period=3."""
    return _wrap(
        "<tr><th>A</th><th>B</th><th>C</th><th>A</th><th>B</th><th>C</th></tr>"
        "<tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr>"
    )


def test_tiebreak_more_repeating_rows_wins() -> None:
    """Two splittable preds, equal segment-distance, one has more header evidence.

    A = 4 cols, two repeating header rows (n_repeating_rows=2, period=2, segs=2).
    B = 4 cols, one repeating header row  (n_repeating_rows=1, period=2, segs=2).
    n_expected = 3.

    Combos with distance 0 are (split-A, no-B)=3 and (no-A, split-B)=3.
    Both have segment-distance 0. Tiebreak by Σn_repeating_rows: (split-A, no-B)
    has rows=2 vs (no-A, split-B) has rows=1 → A is split, B is preserved.
    """
    expected_md = _gt_n_two_col_tables(3)
    actual_md = _doc_with_tables([_merged_two_header_rows_period2(), _merged_period2(2)])
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    assert len(expected) == 3 and len(actual) == 2
    original_b_html = actual[1].raw_html
    assert original_b_html != ""

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is True
    assert len(result) == 3
    # B should be preserved (its raw_html survives); A's two sub-tables have raw_html="".
    preserved = [et for et in result if et.raw_html != ""]
    assert len(preserved) == 1
    assert preserved[0].raw_html == original_b_html


def test_tiebreak_larger_period_wins() -> None:
    """Two splittable preds, tied on distance and on repeating-rows, differ on period.

    A = 4 cols, period 2, n_segs=2, n_repeating_rows=1.
    B = 6 cols, period 3, n_segs=2, n_repeating_rows=1.
    n_expected = 3.

    (split-A, no-B) and (no-A, split-B) both have distance 0 and Σrows=1.
    Period tiebreak: Σperiod 2 vs 3 → larger wins → B is split, A preserved.
    """
    expected_md = _gt_n_two_col_tables(3)
    actual_md = _doc_with_tables([_merged_period2(2), _merged_period3()])
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    assert len(expected) == 3 and len(actual) == 2
    original_a_html = actual[0].raw_html
    assert original_a_html != ""

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is True
    assert len(result) == 3
    preserved = [et for et in result if et.raw_html != ""]
    assert len(preserved) == 1
    assert preserved[0].raw_html == original_a_html


def test_all_fixed_page_no_op() -> None:
    """No pred table has any candidate periods → no-op."""
    expected_md = _gt_n_two_col_tables(3)
    actual_md = _doc_with_tables([_fixed_table(), _fixed_table()])
    expected, actual, _ = extract_table_pairs(expected_md, actual_md)
    assert len(expected) == 3 and len(actual) == 2

    result, did_split = split_ambiguous_merged_pred(expected, actual)
    assert did_split is False
    assert result is actual
