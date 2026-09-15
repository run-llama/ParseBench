"""IoU utilities for layout detection.

Axis-aligned IoU / IoA for COCO and xyxy boxes, plus the rotated polygon
IoU / IoA used by layout scoring on datasets whose ground truth carries an
``r`` rotation (see :mod:`parse_bench.geometry.rotated_bbox`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from parse_bench.geometry.rotated_bbox import xywh_r_to_polygon

__all__ = [
    "coco_to_xyxy",
    "compute_iou",
    "compute_iou_matrix",
    "compute_rotated_ioa_matrix",
    "compute_rotated_iou",
    "compute_rotated_iou_matrix",
    "convex_polygon_intersection",
    "polygon_area",
    "xyxy_to_rotated_polygon",
]


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """
    Compute IoU between two boxes in xyxy format [x1, y1, x2, y2].

    :param box1: First bounding box [x1, y1, x2, y2]
    :param box2: Second bounding box [x1, y1, x2, y2]
    :return: IoU value between 0.0 and 1.0
    """
    # Determine intersection coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute intersection area
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def compute_iou_matrix(
    boxes1: np.ndarray,  # shape (N, 4)
    boxes2: np.ndarray,  # shape (M, 4)
) -> np.ndarray:  # shape (N, M)
    """
    Compute pairwise IoU matrix between two sets of boxes (vectorized).

    Both box sets should be in xyxy format [x1, y1, x2, y2].

    :param boxes1: Array of shape (N, 4) with N bounding boxes
    :param boxes2: Array of shape (M, 4) with M bounding boxes
    :return: IoU matrix of shape (N, M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection
    # boxes1[:, None, :2] has shape (N, 1, 2), boxes2[None, :, :2] has shape (1, M, 2)
    # Result has shape (N, M, 2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # left-top
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # right-bottom
    wh = np.clip(rb - lt, 0, None)  # width-height, clipped to non-negative
    intersection = wh[:, :, 0] * wh[:, :, 1]

    # Compute union
    # area1[:, None] has shape (N, 1), area2[None, :] has shape (1, M)
    union = area1[:, None] + area2[None, :] - intersection

    return intersection / np.clip(union, 1e-10, None)  # type: ignore[no-any-return]


def coco_to_xyxy(bbox: list[float]) -> list[float]:
    """
    Convert COCO format bbox [x, y, width, height] to xyxy format [x1, y1, x2, y2].

    :param bbox: Bounding box in COCO format [x, y, width, height]
    :return: Bounding box in xyxy format [x1, y1, x2, y2]
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_rotated_polygon(
    box_xyxy: list[float],
    angle_degrees: float,
    *,
    page_width: float = 1.0,
    page_height: float = 1.0,
) -> list[tuple[float, float]]:
    """Convert a normalized xyxy box and page/SVG angle into four polygon corners."""
    x1, y1, x2, y2 = box_xyxy
    return xywh_r_to_polygon(
        x1,
        y1,
        x2 - x1,
        y2 - y1,
        angle_degrees,
        page_width=page_width,
        page_height=page_height,
        normalized=True,
    )


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Return absolute shoelace area for a simple polygon."""
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _polygon_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segment_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) <= 1e-12:
        return p2

    det_p = x1 * y2 - y1 * x2
    det_q = x3 * y4 - y3 * x4
    intersection_x = (det_p * (x3 - x4) - (x1 - x2) * det_q) / denominator
    intersection_y = (det_p * (y3 - y4) - (y1 - y2) * det_q) / denominator
    return (intersection_x, intersection_y)


def convex_polygon_intersection(
    subject: list[tuple[float, float]],
    clip: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Clip a convex subject polygon against a convex clip polygon."""
    output = list(subject)
    if len(output) < 3:
        return []

    for clip_index in range(len(clip)):
        if not output:
            return []

        clip_start = clip[clip_index]
        clip_end = clip[(clip_index + 1) % len(clip)]
        input_list = output
        output = []

        if not input_list:
            continue

        previous_point = input_list[-1]
        for current_point in input_list:
            current_inside = _cross(clip_start, clip_end, current_point) >= 0.0
            previous_inside = _cross(clip_start, clip_end, previous_point) >= 0.0

            if current_inside:
                if not previous_inside:
                    output.append(_segment_intersection(previous_point, current_point, clip_start, clip_end))
                output.append(current_point)
            elif previous_inside:
                output.append(_segment_intersection(previous_point, current_point, clip_start, clip_end))

            previous_point = current_point

    return output


def compute_rotated_iou(
    box1: list[float],
    angle1: float | None,
    box2: list[float],
    angle2: float | None,
    *,
    gt_angle_present: bool,
    page_width: float = 1.0,
    page_height: float = 1.0,
    force_rotated: bool = False,
) -> float:
    """Compute IoU between two boxes, using polygon IoU only when GT rotation is present."""
    if not gt_angle_present and not force_rotated:
        return compute_iou(box1, box2)

    pred_angle = 0.0 if angle2 is None else angle2
    gt_angle = 0.0 if angle1 is None else angle1
    polygon1 = xyxy_to_rotated_polygon(box1, gt_angle, page_width=page_width, page_height=page_height)
    polygon2 = xyxy_to_rotated_polygon(box2, pred_angle, page_width=page_width, page_height=page_height)
    intersection_area = _polygon_overlap_area(polygon1, polygon2)
    if intersection_area <= 0.0:
        return 0.0

    union_area = polygon_area(polygon1) + polygon_area(polygon2) - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def compute_rotated_iou_matrix(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    gt_angles: list[float | None] | None,
    pred_angles: list[float | None] | None,
    *,
    page_width: float = 1.0,
    page_height: float = 1.0,
    page_widths: Sequence[float] | None = None,
    page_heights: Sequence[float] | None = None,
    force_rotated: bool = False,
) -> np.ndarray:
    """Compute pairwise IoU matrix with rotated polygon IoU when GT carries rotation."""
    gt_count = len(gt_boxes)
    pred_count = len(pred_boxes)
    if gt_count == 0 or pred_count == 0:
        return np.zeros((gt_count, pred_count))

    gt_angle_values = gt_angles or [None] * gt_count
    pred_angle_values = pred_angles or [None] * pred_count
    matrix = np.zeros((gt_count, pred_count), dtype=float)
    pred_polygon_cache: dict[
        tuple[float, float],
        tuple[
            list[list[tuple[float, float]]],
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    def pred_polygons_for_dimensions(
        width: float,
        height: float,
    ) -> tuple[list[list[tuple[float, float]]], np.ndarray, np.ndarray]:
        key = (width, height)
        cached = pred_polygon_cache.get(key)
        if cached is not None:
            return cached

        polygons: list[list[tuple[float, float]]] = []
        bounds: list[tuple[float, float, float, float]] = []
        areas: list[float] = []
        for pred_index in range(pred_count):
            pred_angle = pred_angle_values[pred_index] if pred_index < len(pred_angle_values) else None
            polygon = xyxy_to_rotated_polygon(
                pred_boxes[pred_index].tolist(),
                0.0 if pred_angle is None else pred_angle,
                page_width=width,
                page_height=height,
            )
            polygons.append(polygon)
            bounds.append(_polygon_bounds(polygon))
            areas.append(polygon_area(polygon))

        cached = (
            polygons,
            np.array(bounds, dtype=float),
            np.array(areas, dtype=float),
        )
        pred_polygon_cache[key] = cached
        return cached

    for gt_index in range(gt_count):
        gt_box = gt_boxes[gt_index].tolist()
        gt_angle = gt_angle_values[gt_index] if gt_index < len(gt_angle_values) else None
        gt_angle_present = gt_angle is not None

        if not gt_angle_present and not force_rotated:
            row = compute_iou_matrix(
                gt_boxes[gt_index : gt_index + 1],
                pred_boxes,
            )
            matrix[gt_index, :] = row[0]
            continue

        row_page_width = _dimension_for_index(page_widths, gt_index, page_width)
        row_page_height = _dimension_for_index(page_heights, gt_index, page_height)
        gt_polygon = xyxy_to_rotated_polygon(
            gt_box,
            0.0 if gt_angle is None else gt_angle,
            page_width=row_page_width,
            page_height=row_page_height,
        )
        gt_area = polygon_area(gt_polygon)
        if gt_area <= 0.0:
            continue
        gt_min_x, gt_min_y, gt_max_x, gt_max_y = _polygon_bounds(gt_polygon)
        pred_polygons, pred_bounds, pred_areas = pred_polygons_for_dimensions(row_page_width, row_page_height)
        candidate_indices = np.flatnonzero(
            (pred_bounds[:, 2] >= gt_min_x)
            & (pred_bounds[:, 0] <= gt_max_x)
            & (pred_bounds[:, 3] >= gt_min_y)
            & (pred_bounds[:, 1] <= gt_max_y)
        )

        for pred_index in candidate_indices:
            intersection_area = _polygon_overlap_area(gt_polygon, pred_polygons[pred_index])
            if intersection_area <= 0.0:
                continue

            union_area = gt_area + pred_areas[pred_index] - intersection_area
            if union_area <= 0.0:
                continue
            matrix[gt_index, pred_index] = intersection_area / union_area

    return matrix


def compute_rotated_ioa_matrix(
    source_boxes: np.ndarray,
    target_boxes: np.ndarray,
    source_angles: list[float | None] | None,
    target_angles: list[float | None] | None,
    *,
    page_width: float = 1.0,
    page_height: float = 1.0,
    page_widths: Sequence[float] | None = None,
    page_heights: Sequence[float] | None = None,
) -> np.ndarray:
    """Compute pairwise IoA using literal rotated boxes when either side carries rotation.

    IoA is source-area anchored: result[i, j] is
    area(intersection(source_i, target_j)) / area(source_i).
    """
    source_count = len(source_boxes)
    target_count = len(target_boxes)
    if source_count == 0 or target_count == 0:
        return np.zeros((source_count, target_count))

    source_angle_values = source_angles or [None] * source_count
    target_angle_values = target_angles or [None] * target_count
    if not any(angle is not None for angle in source_angle_values) and not any(
        angle is not None for angle in target_angle_values
    ):
        return _compute_axis_aligned_ioa_matrix(source_boxes, target_boxes)

    matrix = np.zeros((source_count, target_count), dtype=float)
    for source_index in range(source_count):
        source_box = source_boxes[source_index].tolist()
        source_angle = source_angle_values[source_index] if source_index < len(source_angle_values) else None
        width = _dimension_for_index(page_widths, source_index, page_width)
        height = _dimension_for_index(page_heights, source_index, page_height)
        source_polygon = xyxy_to_rotated_polygon(
            source_box,
            0.0 if source_angle is None else source_angle,
            page_width=width,
            page_height=height,
        )
        source_area = polygon_area(source_polygon)
        if source_area <= 0.0:
            continue

        for target_index in range(target_count):
            target_box = target_boxes[target_index].tolist()
            target_angle = target_angle_values[target_index] if target_index < len(target_angle_values) else None
            target_polygon = xyxy_to_rotated_polygon(
                target_box,
                0.0 if target_angle is None else target_angle,
                page_width=width,
                page_height=height,
            )
            matrix[source_index, target_index] = _polygon_overlap_area(source_polygon, target_polygon) / source_area

    return matrix


def _polygon_overlap_area(
    polygon1: list[tuple[float, float]],
    polygon2: list[tuple[float, float]],
) -> float:
    return polygon_area(convex_polygon_intersection(polygon1, polygon2))


def _compute_axis_aligned_ioa_matrix(source_boxes: np.ndarray, target_boxes: np.ndarray) -> np.ndarray:
    source_boxes = np.asarray(source_boxes, dtype=float)
    target_boxes = np.asarray(target_boxes, dtype=float)
    source_areas = (source_boxes[:, 2] - source_boxes[:, 0]) * (source_boxes[:, 3] - source_boxes[:, 1])
    lt = np.maximum(source_boxes[:, None, :2], target_boxes[None, :, :2])
    rb = np.minimum(source_boxes[:, None, 2:], target_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    return intersection / np.clip(source_areas[:, None], 1e-10, None)  # type: ignore[no-any-return]


def _dimension_for_index(values: Sequence[float] | None, index: int, default: float) -> float:
    if values is None or index >= len(values):
        return default
    value = float(values[index])
    return value if value > 0 else default
