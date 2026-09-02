"""Behaviour-preservation tests for the paged/non-paged classification path."""

from __future__ import annotations

import numpy as np
import pytest

from parse_bench.evaluation.metrics.layoutdet.classification_utils import (
    compute_map_at_thresholds,
    compute_per_class_metrics,
    match_prediction_dicts_to_gt,
)

_GT = [
    {"bbox": [0.0, 0.0, 0.5, 0.5], "class_name": "Text"},
    {"bbox": [0.5, 0.5, 1.0, 1.0], "class_name": "Table"},
]
_PRED = [
    {"bbox": [0.0, 0.0, 0.5, 0.5], "class_name": "Text", "score": 0.9},
    {"bbox": [0.5, 0.5, 1.0, 1.0], "class_name": "Table", "score": 0.8},
    {"bbox": [0.0, 0.5, 0.2, 0.7], "class_name": "Text", "score": 0.3},  # FP
]


def _with_page(entries: list[dict], page: str) -> list[dict]:
    return [{**entry, "example_id": page} for entry in entries]


def test_non_paged_and_single_page_inputs_give_identical_metrics() -> None:
    non_paged = compute_per_class_metrics(_PRED, _GT, ["Text", "Table"])
    paged = compute_per_class_metrics(_with_page(_PRED, "p1"), _with_page(_GT, "p1"), ["Text", "Table"])
    assert non_paged == paged
    assert non_paged["Text"]["precision"] == pytest.approx(0.5)
    assert non_paged["Text"]["recall"] == pytest.approx(1.0)
    assert non_paged["Text"]["support"] == 1
    assert non_paged["Table"]["f1"] == pytest.approx(1.0)


def test_class_without_gt_or_predictions_is_zero_filled() -> None:
    metrics = compute_per_class_metrics(_PRED, _GT, ["Text", "Table", "Picture"])
    assert metrics["Picture"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "support": 0}


def test_custom_page_key_separates_pages() -> None:
    preds = [{**_PRED[0], "page_id": "a"}]
    gts = [{**_GT[0], "page_id": "b"}]  # same box, different page: no match
    metrics = compute_per_class_metrics(preds, gts, ["Text"], page_key="page_id")
    assert metrics["Text"]["recall"] == 0.0
    metrics_same = compute_per_class_metrics(preds, [{**_GT[0], "page_id": "a"}], ["Text"], page_key="page_id")
    assert metrics_same["Text"]["recall"] == 1.0


def test_match_prediction_dicts_to_gt_defaults_to_axis_aligned_iou() -> None:
    y_true, y_scores = match_prediction_dicts_to_gt(
        [p for p in _PRED if p["class_name"] == "Text"],
        [g for g in _GT if g["class_name"] == "Text"],
        iou_threshold=0.5,
    )
    assert y_true.tolist() == [1.0, 0.0]
    assert y_scores.tolist() == [0.9, 0.3]


def test_match_prediction_dicts_to_gt_handles_empty_inputs() -> None:
    y_true, y_scores = match_prediction_dicts_to_gt([], _GT, iou_threshold=0.5)
    assert y_true.size == 0 and y_scores.size == 0
    y_true, y_scores = match_prediction_dicts_to_gt(_PRED[:1], [], iou_threshold=0.5)
    assert y_true.tolist() == [0.0]
    assert y_scores.tolist() == [0.9]


def test_match_prediction_dicts_to_gt_custom_overlap_fn() -> None:
    def always_overlap(pred: dict, gt: dict) -> float:
        return 1.0

    y_true, _ = match_prediction_dicts_to_gt(
        [{"bbox": [0.0, 0.0, 0.1, 0.1], "class_name": "Text", "score": 0.5}],
        [{"bbox": [0.9, 0.9, 1.0, 1.0], "class_name": "Text"}],
        iou_threshold=0.5,
        overlap_fn=always_overlap,
    )
    assert y_true.tolist() == [1.0]


def test_missing_score_defaults_to_zero() -> None:
    y_true, y_scores = match_prediction_dicts_to_gt(
        [{"bbox": [0.0, 0.0, 0.5, 0.5], "class_name": "Text"}],
        [_GT[0]],
        iou_threshold=0.5,
    )
    assert y_true.tolist() == [1.0]
    assert np.array_equal(y_scores, np.array([0.0]))


def test_compute_map_at_thresholds_accepts_page_key_and_overlap_fn() -> None:
    default = compute_map_at_thresholds(_PRED, _GT, ["Text", "Table"])
    explicit = compute_map_at_thresholds(_PRED, _GT, ["Text", "Table"], overlap_fn=None, page_key="example_id")
    assert default == explicit
    assert default["AP50"] == pytest.approx(1.0)
