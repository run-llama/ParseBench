"""Dataset preparation for conservative word-height outward margins (no scoring-time inference)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from parse_bench.test_cases.schema import LayoutBboxTolerance, LayoutTestRule


def word_outward_tolerances(words: Sequence[LayoutTestRule], size: tuple[float, float]) -> list[LayoutBboxTolerance]:
    """Return 10%-of-height margins, capped at the page and half the neighbor gap.

    Input is all word rules on one page, including ignored/picture children, in
    normalized xywh coordinates. Rotation and existing overlap receive zero.
    Punctuation uses the nearest aligned alphanumeric word's height, when present.
    """
    pw, ph = size
    if pw <= 0 or ph <= 0 or not np.isfinite(size).all():
        raise ValueError("Page dimensions must be positive and finite")
    boxes = np.array(
        [[r.bbox[0] * pw, r.bbox[1] * ph, (r.bbox[0] + r.bbox[2]) * pw, (r.bbox[1] + r.bbox[3]) * ph] for r in words],
        dtype=float,
    ).reshape(-1, 4)
    heights = boxes[:, 3] - boxes[:, 1]
    ref_heights = heights.copy()
    alpha = np.array(
        [
            any(c.isalnum() for c in (r.content.text or "")) if r.content and r.content.type == "text" else False
            for r in words
        ],
        dtype=bool,
    )
    rotated = np.array([bool(r.r) for r in words], dtype=bool)
    for i in np.flatnonzero(~alpha):
        # Only nearby words on the same printed baseline are useful for a dash/dot.
        cy = (boxes[i, 1] + boxes[i, 3]) / 2
        dx = np.maximum(0, np.maximum(boxes[:, 0] - boxes[i, 2], boxes[i, 0] - boxes[:, 2]))
        choices = np.flatnonzero(alpha & ~rotated & (boxes[:, 1] <= cy) & (boxes[:, 3] >= cy) & (dx <= 3 * heights))
        if len(choices):
            closest = choices[np.argmin(dx[choices])]
            ref_heights[i] = heights[closest]
    tolerance = 0.1 * np.maximum(0, ref_heights)
    outward = np.repeat(tolerance[:, None], 4, axis=1)
    outward = np.maximum(0, np.minimum(outward, np.c_[boxes[:, 0], boxes[:, 1], pw - boxes[:, 2], ph - boxes[:, 3]]))
    overlap = np.zeros(len(words), dtype=bool)
    for i, box in enumerate(boxes):
        dx = np.maximum(boxes[:, 0] - box[2], box[0] - boxes[:, 2])
        dy = np.maximum(boxes[:, 1] - box[3], box[1] - boxes[:, 3])
        intersects = (dx < 0) & (dy < 0)
        intersects[i] = False
        if intersects.any():
            overlap[i] = True
            overlap[intersects] = True
        candidates = np.flatnonzero(
            (np.arange(len(words)) > i)
            & (dx < tolerance[i] + tolerance)
            & (dy < tolerance[i] + tolerance)
            & ~intersects
        )
        for j in candidates:
            if dx[j] >= dy[j]:
                a, b, gap = (2, 0, dx[j]) if box[2] <= boxes[j, 0] else (0, 2, dx[j])
            else:
                a, b, gap = (3, 1, dy[j]) if box[3] <= boxes[j, 1] else (1, 3, dy[j])
            outward[i, a] = min(outward[i, a], max(0, gap / 2))
            outward[j, b] = min(outward[j, b], max(0, gap / 2))
    outward[overlap | rotated | (heights <= 0)] = 0
    return [
        LayoutBboxTolerance.model_validate(
            {
                "version": "outward-word-height-10-v1",
                "outward_normalized": (row / [pw, ph, pw, ph]).tolist(),
            }
        )
        for row in outward
    ]
