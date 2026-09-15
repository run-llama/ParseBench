from __future__ import annotations

import pytest

from parse_bench.geometry.rotated_bbox import (
    LiteralRotatedBox,
    polygon_angle_degrees,
    polygon_to_literal_xywh_r,
    rotated_rect_contains_point,
    xywh_r_to_polygon,
)


@pytest.mark.parametrize(
    ("polygon", "expected"),
    [
        ([[10, 20], [50, 20], [50, 30], [10, 30]], LiteralRotatedBox(0.1, 0.1, 0.4, 0.05, 0.0)),
        ([50, 20, 50, 60, 40, 60, 40, 20], LiteralRotatedBox(0.25, 0.175, 0.4, 0.05, 90.0)),
        ([[50, 60], [50, 20], [60, 20], [60, 60]], LiteralRotatedBox(0.35, 0.175, 0.4, 0.05, -90.0)),
    ],
)
def test_polygon_to_literal_xywh_r_preserves_literal_dimensions(
    polygon: list[float] | list[list[float]],
    expected: LiteralRotatedBox,
) -> None:
    actual = polygon_to_literal_xywh_r(polygon, page_width=100, page_height=200)

    assert actual is not None
    assert actual.x == pytest.approx(expected.x)
    assert actual.y == pytest.approx(expected.y)
    assert actual.w == pytest.approx(expected.w)
    assert actual.h == pytest.approx(expected.h)
    assert actual.r == pytest.approx(expected.r)


@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 135.0, -90.0])
def test_literal_box_round_trips_through_polygon(angle: float) -> None:
    polygon = xywh_r_to_polygon(
        0.2 * 612,
        0.3 * 792,
        0.25 * 612,
        0.05 * 792,
        angle,
        normalized=False,
    )

    actual = polygon_to_literal_xywh_r(polygon, page_width=612, page_height=792)

    assert actual is not None
    assert actual.x == pytest.approx(0.2, abs=1e-12)
    assert actual.y == pytest.approx(0.3, abs=1e-12)
    assert actual.w == pytest.approx(0.25, abs=1e-12)
    assert actual.h == pytest.approx(0.05, abs=1e-12)
    assert actual.r == pytest.approx(angle)


def test_xywh_r_to_polygon_uses_page_units_before_rotation() -> None:
    polygon = xywh_r_to_polygon(
        0.25,
        0.25,
        0.5,
        0.1,
        90.0,
        page_width=100,
        page_height=200,
    )

    assert polygon == pytest.approx(
        [
            (0.6, 0.175),
            (0.6, 0.425),
            (0.4, 0.425),
            (0.4, 0.175),
        ]
    )


def test_polygon_angle_degrees_accepts_flat_and_nested_polygons() -> None:
    assert polygon_angle_degrees([0, 0, 10, 10, 10, 20, 0, 10]) == pytest.approx(45.0)
    assert polygon_angle_degrees([[0, 0], [10, 10], [10, 20], [0, 10]]) == pytest.approx(45.0)


@pytest.mark.parametrize(
    "polygon",
    [
        None,
        [],
        [0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, float("nan"), 0, 1],
        [[0, 0], [1, 0], [1, "bad"], [0, 1]],
    ],
)
def test_polygon_to_literal_xywh_r_rejects_degenerate_polygons(polygon: object) -> None:
    assert polygon_to_literal_xywh_r(polygon, page_width=100, page_height=100) is None


def test_rotated_rect_contains_point_uses_literal_rotated_frame() -> None:
    box = [0.45, 0.4, 0.1, 0.2]

    assert rotated_rect_contains_point(box, 90.0, (0.5, 0.5), page_width=100, page_height=100)
    assert rotated_rect_contains_point(box, 90.0, (0.4, 0.5), page_width=100, page_height=100)
    assert not rotated_rect_contains_point(box, 90.0, (0.5, 0.25), page_width=100, page_height=100)
