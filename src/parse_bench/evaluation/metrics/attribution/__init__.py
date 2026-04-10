"""Layout attribution metrics for evaluating text-to-region correctness."""

from parse_bench.evaluation.metrics.attribution.core import (
    compute_af1,
    compute_attribution_metrics,
    compute_grounding_accuracy,
    compute_lap,
    compute_lar,
)
from parse_bench.evaluation.metrics.attribution.geometry import (
    coco_to_xyxy,
    compute_ioa,
    compute_ioa_matrix,
    normalize_bbox_to_unit,
)
from parse_bench.evaluation.metrics.attribution.text_utils import (
    extract_text_from_html,
    normalize_attribution_text,
    tokenize,
)

__all__ = [
    "normalize_attribution_text",
    "tokenize",
    "extract_text_from_html",
    "compute_ioa",
    "compute_ioa_matrix",
    "normalize_bbox_to_unit",
    "coco_to_xyxy",
    "compute_lap",
    "compute_lar",
    "compute_af1",
    "compute_grounding_accuracy",
    "compute_attribution_metrics",
]
