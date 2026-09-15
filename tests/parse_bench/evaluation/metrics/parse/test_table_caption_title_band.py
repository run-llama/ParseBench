"""Tests for ``<caption>`` / title-band equivalence in GriTS.

``<caption>`` is the element HTML provides for a table's title, and a
prediction that uses it is doing the right thing. Ground truth spells the same
title three other ways: as a full-width header band (a single ``colspan`` cell
across row 0), as body text above the table, or not at all.

Only the header band cost anything. The caption is not a ``<td>``/``<th>``, so
it never enters either grid; against GT body text or no title at all the two
grids already matched. Against a GT band, the band was an unmatched extra GT
row and GriTS charged the prediction for it.

The band is now dropped when the other side's caption says the same thing,
checked in both directions and gated on text equality under the ordinary cell
normalization. A band that says something the caption does not is left alone
and still scores, so this cannot silently forgive a lost title row.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.grits_metric import GriTSMetric
from parse_bench.evaluation.metrics.parse.table_extraction import extract_table_pairs

_BODY = "<tbody><tr><td>Cash</td><td>10</td></tr><tr><td>Debt</td><td>5</td></tr></tbody>"
_HEADER = "<tr><th>Item</th><th>2025</th></tr>"

GT_BAND = f'<table><thead><tr><th colspan="2">Balance Sheet</th></tr>{_HEADER}</thead>{_BODY}</table>'
GT_BODY_TEXT = f"Balance Sheet\n\n<table><thead>{_HEADER}</thead>{_BODY}</table>"
GT_NO_TITLE = f"<table><thead>{_HEADER}</thead>{_BODY}</table>"
PRED_CAPTION = f"<table><caption>Balance Sheet</caption><thead>{_HEADER}</thead>{_BODY}</table>"


def _grits(expected: str, actual: str) -> float:
    exp, act, _ = extract_table_pairs(expected, actual)
    values = GriTSMetric().compute(exp, act)
    return float(next(v.value for v in values if v.metric_name == "grits_con"))


class TestCaptionIsNotPenalized:
    def test_against_a_ground_truth_header_band(self) -> None:
        assert _grits(GT_BAND, PRED_CAPTION) == 1.0

    def test_against_ground_truth_body_text(self) -> None:
        assert _grits(GT_BODY_TEXT, PRED_CAPTION) == 1.0

    def test_against_a_ground_truth_with_no_title(self) -> None:
        assert _grits(GT_NO_TITLE, PRED_CAPTION) == 1.0

    def test_symmetric_when_the_ground_truth_carries_the_caption(self) -> None:
        assert _grits(PRED_CAPTION, GT_BAND) == 1.0


class TestUnchangedCases:
    def test_identical_banded_tables(self) -> None:
        assert _grits(GT_BAND, GT_BAND) == 1.0

    def test_identical_captioned_tables(self) -> None:
        assert _grits(PRED_CAPTION, PRED_CAPTION) == 1.0

    def test_a_prediction_that_simply_loses_the_band_still_pays(self) -> None:
        # No caption anywhere: the missing band is a real content loss.
        assert _grits(GT_BAND, GT_NO_TITLE) < 1.0


class TestTheGuardHolds:
    def test_a_different_caption_does_not_erase_the_band(self) -> None:
        # The caption names another table; the GT band is real content the
        # prediction did not reproduce, and must still be charged.
        pred = PRED_CAPTION.replace("<caption>Balance Sheet</caption>", "<caption>Income Statement</caption>")
        assert _grits(GT_BAND, pred) < 1.0

    def test_cell_errors_are_still_scored_under_the_equivalence(self) -> None:
        pred = PRED_CAPTION.replace("<td>10</td>", "<td>99</td>")
        assert _grits(GT_BAND, pred) < 1.0

    def test_a_non_uniform_first_row_is_not_a_band(self) -> None:
        # Row 0 is a genuine two-cell header row, not a colspan title, so the
        # caption must not delete it even though its first cell matches.
        gt = f"<table><thead><tr><th>Balance Sheet</th><th>2025</th></tr>{_HEADER}</thead>{_BODY}</table>"
        assert _grits(gt, PRED_CAPTION) < 1.0


def test_caption_survives_the_title_stripping_stage() -> None:
    """The equivalence needs the caption to reach the metric."""
    from parse_bench.evaluation.metrics.parse.table_title_stripping import strip_title_rows

    _exp, act, _counts = extract_table_pairs(GT_BAND, PRED_CAPTION)
    stripped = strip_title_rows(act[0])
    assert stripped.table_data.caption == "Balance Sheet"
