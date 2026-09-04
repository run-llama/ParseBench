from __future__ import annotations

import numpy as np
import pytest

from parse_bench.evaluation.metrics.layoutdet.iou import (
    compute_iou,
    compute_iou_matrix,
    compute_rotated_ioa_matrix,
    compute_rotated_iou,
    compute_rotated_iou_matrix,
    convex_polygon_intersection,
    polygon_area,
    xyxy_to_rotated_polygon,
)
from parse_bench.geometry.rotated_bbox import xywh_r_to_polygon


def test_identical_rotated_rectangles_have_iou_one() -> None:
    box = [0.1, 0.2, 0.4, 0.5]
    iou = compute_rotated_iou(box, 45.0, box, 45.0, gt_angle_present=True)
    assert iou == pytest.approx(1.0)


def test_perpendicular_narrow_rectangles_have_low_iou() -> None:
    gt_box = [0.2, 0.2, 0.8, 0.3]
    pred_box = [0.2, 0.2, 0.8, 0.3]
    iou = compute_rotated_iou(gt_box, 0.0, pred_box, 90.0, gt_angle_present=True)
    assert iou < 0.2


def test_missing_gt_angle_matches_legacy_aabb_iou() -> None:
    box1 = [0.1, 0.1, 0.4, 0.4]
    box2 = [0.2, 0.2, 0.5, 0.5]
    rotated = compute_rotated_iou(box1, None, box2, 90.0, gt_angle_present=False)
    assert rotated == pytest.approx(compute_iou(box1, box2))


def test_missing_pred_angle_uses_zero_degrees() -> None:
    box = [0.1, 0.1, 0.4, 0.2]
    with_zero = compute_rotated_iou(box, 90.0, box, None, gt_angle_present=True)
    explicit_zero = compute_rotated_iou(box, 90.0, box, 0.0, gt_angle_present=True)
    assert with_zero == pytest.approx(explicit_zero)


def test_rotated_iou_matrix_falls_back_to_aabb_when_gt_has_no_angle() -> None:
    gt_boxes = np.array([[0.0, 0.0, 0.5, 0.5]], dtype=float)
    pred_boxes = np.array([[0.25, 0.25, 0.75, 0.75]], dtype=float)
    matrix = compute_rotated_iou_matrix(gt_boxes, pred_boxes, [None], [90.0])
    expected = compute_iou_matrix(gt_boxes, pred_boxes)
    assert matrix == pytest.approx(expected)


def test_empty_rotated_iou_matrix_shape() -> None:
    matrix = compute_rotated_iou_matrix(np.zeros((0, 4)), np.zeros((0, 4)), [], [])
    assert matrix.shape == (0, 0)


def test_polygon_area_and_intersection_for_axis_aligned_box() -> None:
    polygon = xyxy_to_rotated_polygon([0.0, 0.0, 0.2, 0.1], 0.0)
    assert polygon_area(polygon) == pytest.approx(0.02)
    intersection = convex_polygon_intersection(polygon, polygon)
    assert polygon_area(intersection) == pytest.approx(0.02)


def test_xyxy_to_rotated_polygon_uses_svg_page_clockwise_helper() -> None:
    polygon = xyxy_to_rotated_polygon([0.25, 0.2, 0.45, 0.3], 90.0, page_width=2.0, page_height=1.0)
    expected = xywh_r_to_polygon(0.25, 0.2, 0.2, 0.1, 90.0, page_width=2.0, page_height=1.0)

    assert polygon == pytest.approx(expected)


def test_rotated_iou_uses_page_dimensions_for_normalized_boxes() -> None:
    gt_box = [0.4, 0.4, 0.6, 0.5]
    page_aware_axis_match = [0.475, 0.25, 0.525, 0.65]

    page_aware = compute_rotated_iou(
        gt_box,
        90.0,
        page_aware_axis_match,
        None,
        gt_angle_present=True,
        page_width=2.0,
        page_height=1.0,
    )
    unit_square = compute_rotated_iou(
        gt_box,
        90.0,
        page_aware_axis_match,
        None,
        gt_angle_present=True,
        page_width=1.0,
        page_height=1.0,
    )

    assert page_aware == pytest.approx(1.0)
    assert unit_square < 0.5


def test_rotated_ioa_matrix_uses_literal_rotated_source_area() -> None:
    gt_boxes = np.array([[0.4, 0.4, 0.6, 0.5]], dtype=float)
    pred_boxes = np.array([[0.475, 0.25, 0.525, 0.65]], dtype=float)

    matrix = compute_rotated_ioa_matrix(
        gt_boxes,
        pred_boxes,
        [90.0],
        [None],
        page_width=2.0,
        page_height=1.0,
    )

    assert matrix[0, 0] == pytest.approx(1.0)
