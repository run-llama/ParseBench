"""Bbox geometry for the grounded family: pixel {x,y,w,h} -> normalized xyxy + IoU.

Both trees declare their own pixel space per page (`page_width`/`page_height`,
the parse output's `items[]` space), and the two sides may render at
different DPIs, so boxes are compared only after each side is normalized by its OWN
declared dims. Degenerate (zero-area) boxes score 0.0.
"""

from typing import Any

Box = tuple[float, float, float, float]  # normalized x0, y0, x1, y1


def normalize_node_bboxes(raw: Any, page_width: float | None, page_height: float | None) -> tuple[Box, ...]:
    """Pixel `{x, y, w, h}` dicts -> normalized xyxy boxes.

    Returns () when `raw` is not a list or the page dims are missing/zero — a bbox
    claim without a declared page space is ungradeable, not an error. Malformed
    entries (non-dict, missing/non-numeric coords) are skipped individually.
    """
    if not isinstance(raw, list) or not page_width or not page_height:
        return ()
    out: list[Box] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            x, y, w, h = (float(entry[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            continue
        out.append((x / page_width, y / page_height, (x + w) / page_width, (y + h) / page_height))
    return tuple(out)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_pair_iou(gt: tuple[Box, ...], pred: tuple[Box, ...]) -> float:
    """Max IoU over the gt x pred cross product; 0.0 when either side is empty."""
    return max((iou(g, p) for g in gt for p in pred), default=0.0)
