"""Rotation must survive the real attribution parse and scoring boundary."""

from dataclasses import replace

import numpy as np
import pytest

from parse_bench.evaluation.metrics.attribution.core import (
    compute_attribution_metrics,
    compute_per_class_lap_by_gt,
    compute_reading_order,
    parse_gt_elements,
    parse_pred_blocks,
)
from parse_bench.evaluation.metrics.attribution.geometry import compute_overlap_matrix


def _pair(angle=30, prediction_angle=30, *, attributes=None):
    gt = parse_gt_elements(
        [
            {
                "type": "layout",
                "bbox": [0.2, 0.45, 0.6, 0.05],
                "r": angle,
                "canonical_class": "Text",
                "content": {"type": "text", "text": "alpha alpha beta"},
                "attributes": attributes or {},
            }
        ]
    )
    box = {"x": 20, "y": 45, "w": 60, "h": 5, "r": prediction_angle, "label": "text"}
    pred = parse_pred_blocks(
        [
            {
                "type": "text",
                "value": "alpha alpha beta",
                "bBox": box,
                "layoutAwareBbox": [box],
            }
        ],
        "",
        100,
        100,
    )
    return gt, pred


@pytest.mark.parametrize("angle", [None, 0, -30, 30, -90, 90, 180])
def test_rotation_survives_both_parsers(angle):
    gt, pred = _pair(angle, angle)
    assert gt[0].r == angle
    assert pred[0].r == angle
    assert (pred[0].page_width, pred[0].page_height) == (100, 100)
    assert compute_attribution_metrics(gt, pred).af1 == 1


@pytest.mark.parametrize("angle", [-30, 30, -90, 90, 180])
def test_perpendicular_identical_text_is_not_grounded(angle):
    gt, pred = _pair(angle, angle + 90)
    result = compute_attribution_metrics(gt, pred)
    assert result.af1 == result.lap == result.lar == result.grounding_accuracy == 0
    assert result.unmatched_gt_elements == result.unmatched_pred_blocks == 1
    assert result.spatial_fp_blocks == 1
    assert result.supported_claim_count == 0
    assert result.per_class_lar == {"Text": 0}
    assert compute_per_class_lap_by_gt(gt, pred) == {}


def test_rotated_duplicate_ownership_and_explicit_filtering():
    gt, pred = _pair()
    wrong = replace(pred[0], r=120, order_index=0)
    right = replace(pred[0], order_index=1)
    duplicate = replace(right, order_index=2)
    result = compute_attribution_metrics(gt, [wrong, right, duplicate])
    assert result.num_scored_pred_blocks == 3
    assert result.spatial_fp_blocks == 1
    assert result.supported_tp_blocks == result.redundant_fp_blocks == 1
    assert result.duplicate_supported_token_instances == 3
    assert result.grounding_accuracy == 1
    explicit = [replace(gt[0], attributes={"explicit": "true"})]
    explicit_result = compute_attribution_metrics(explicit, [wrong, right])
    assert explicit_result.num_scored_pred_blocks == 1
    assert explicit_result.spatial_fp_blocks == 1
    assert explicit_result.lap == 0
    ignored = [replace(gt[0], attributes={"ignore": "true"})]
    assert compute_attribution_metrics(ignored, [right]).num_gt_elements == 0


def test_reading_order_uses_rotated_eligibility():
    gt, pred = _pair(90, 90)
    gt[0] = replace(gt[0], bbox_coco=[0.1, 0.45, 0.2, 0.05], bbox_xyxy=[0.1, 0.45, 0.3, 0.5])
    pred[0] = replace(pred[0], bbox_xyxy=gt[0].bbox_xyxy)
    second_gt = replace(gt[0], bbox_coco=[0.6, 0.45, 0.2, 0.05], bbox_xyxy=[0.6, 0.45, 0.8, 0.5], ro_index=1)
    second_pred = replace(pred[0], bbox_xyxy=second_gt.bbox_xyxy, order_index=1)
    wrong_first = replace(second_pred, r=0, order_index=-1)
    assert compute_reading_order(gt + [second_gt], pred + [second_pred, wrong_first]) == (1, 1)


@pytest.mark.parametrize("angle", [0, 30, -30, 90, -90, 180])
def test_overlap_retains_merge_split_metric(angle):
    small = np.array([[0.35, 0.475, 0.65, 0.5]])
    large = np.array([[0.2, 0.45, 0.8, 0.525]])
    assert compute_overlap_matrix(small, large, gt_angles=[angle], pred_angles=[angle])[0, 0] == pytest.approx(1)
    assert compute_overlap_matrix(large, small, gt_angles=[angle], pred_angles=[angle])[0, 0] == pytest.approx(1)


def test_nonsquare_geometry_negative_origins_and_perpendicular_ratio():
    horizontal = np.array([[0.4, 0.4, 0.6, 0.5]])
    vertical = np.array([[0.475, 0.25, 0.525, 0.65]])
    assert compute_overlap_matrix(
        horizontal, vertical, gt_angles=[90], pred_angles=[0], page_width=200, page_height=100
    )[0, 0] == pytest.approx(1)
    thin = np.array([[0.2, 0.45, 0.8, 0.5]])
    assert compute_overlap_matrix(thin, thin, gt_angles=[0], pred_angles=[90])[0, 0] == pytest.approx(1 / 12)
    # Literal x is negative; its quarter-turn footprint is [0,.1,.1,.5].
    edge = np.array([[-0.15, 0.25, 0.25, 0.35]])
    drawn = np.array([[0, 0.1, 0.1, 0.5]])
    assert compute_overlap_matrix(edge, drawn, gt_angles=[90], pred_angles=[0])[0, 0] == pytest.approx(1)


def test_clockwise_sign_controls_membership():
    line = np.array([[0.2, 0.475, 0.8, 0.525]])
    # At +30 degrees its right end lies below the center, not above it.
    point = np.array([[0.72, 0.625, 0.73, 0.635]])
    assert compute_overlap_matrix(line, point, gt_angles=[30], pred_angles=[0])[0, 0] == pytest.approx(1)
    assert compute_overlap_matrix(line, point, gt_angles=[-30], pred_angles=[0])[0, 0] == 0


@pytest.mark.parametrize("angle", [None, 0, 30])
def test_empty_degenerate_and_disjoint_geometry(angle):
    empty = np.empty((0, 4))
    box = np.array([[0.1, 0.1, 0.2, 0.2]])
    assert compute_overlap_matrix(empty, box, gt_angles=[], pred_angles=[angle]).shape == (0, 1)
    assert compute_overlap_matrix(box, empty, gt_angles=[angle], pred_angles=[]).shape == (1, 0)
    assert compute_overlap_matrix(empty, empty).shape == (0, 0)
    for other in ([[0.8, 0.8, 0.9, 0.9]], [[0.1, 0.1, 0.1, 0.2]]):
        assert compute_overlap_matrix(box, np.array(other), gt_angles=[angle], pred_angles=[angle])[0, 0] == 0


@pytest.mark.parametrize("width,height", [(0, 100), (-1, 100), (100, float("nan")), (float("inf"), 100)])
def test_invalid_page_dimensions_fail_explicitly(width, height):
    with pytest.raises(ValueError, match="page dimensions"):
        parse_pred_blocks([], "", width, height)


def test_touching_upright_boxes_have_no_overlap():
    assert compute_overlap_matrix(np.array([[0.1, 0.1, 0.2, 0.2]]), np.array([[0.2, 0.1, 0.3, 0.2]]))[0, 0] == 0


def test_coarse_box_retains_rotation_only_when_scope_allows():
    box = {"x": 20, "y": 45, "w": 60, "h": 5, "r": -30}
    items = [{"type": "text", "value": "alpha", "bBox": box}]
    assert parse_pred_blocks(items, "", 100, 100)[0].r == -30
    assert parse_pred_blocks(items, "", 100, 100, require_layout_aware_segments=True) == []


def test_attribution_refuses_mixed_and_invalid_page_dimensions():
    from parse_bench.evaluation.metrics.attribution.core import compute_attribution_overlap

    gt, pred = _pair()
    with pytest.raises(ValueError, match="consistent page dimensions"):
        compute_attribution_overlap(gt, pred + [replace(pred[0], page_width=200)])
    with pytest.raises(ValueError, match="finite positive page dimensions"):
        compute_attribution_overlap(gt, [replace(pred[0], page_height=float("nan"))])


def test_degenerate_rotated_source_has_zero_overlap():
    empty = np.array([[0.1, 0.1, 0.1, 0.2]])
    box = np.array([[0.1, 0.1, 0.2, 0.2]])
    assert compute_overlap_matrix(empty, box, gt_angles=[30], pred_angles=[30])[0, 0] == 0
