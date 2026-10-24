"""LayoutDetectionEvaluator on layout rules that carry no ro_index."""

from datetime import datetime

import pytest

from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
from parse_bench.evaluation.metrics.attribution.core import parse_gt_elements
from parse_bench.inference.layout_extraction import extract_all_layouts_from_llamaparse_output
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import LayoutDetectionTestCase

_LINES = [("alpha beta", 10), ("gamma delta", 40)]


def _inference():
    items = [
        {
            "type": "text",
            "value": text,
            "layoutAwareBbox": [
                {
                    "x": 10,
                    "y": y,
                    "w": 80,
                    "h": 10,
                    "label": "text",
                    "confidence": 1,
                    "start_index": 0,
                    "end_index": len(text),
                }
            ],
        }
        for text, y in _LINES
    ]
    output = extract_all_layouts_from_llamaparse_output({"pages": [{"width": 100, "height": 100, "items": items}]})
    return InferenceResult(
        request=InferenceRequest(
            example_id="probe", source_file_path="/tmp/probe.pdf", product_type=ProductType.LAYOUT_DETECTION
        ),
        pipeline_name="llamaparse",
        product_type=ProductType.LAYOUT_DETECTION,
        raw_output={},
        output=output,
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
        latency_in_ms=0,
    )


@pytest.mark.parametrize("ro_indices", [(None, None), (0, None)], ids=["all_missing", "mixed"])
def test_rules_without_ro_index_do_not_crash_reading_order(ro_indices):
    """Bug: rule.model_dump() keeps ro_index=None, so compute_reading_order's sort raises TypeError."""
    rules = []
    for (text, y), ro_index in zip(_LINES, ro_indices, strict=True):
        rule = {
            "type": "layout",
            "page": 1,
            "bbox": [0.1, y / 100, 0.8, 0.1],
            "canonical_class": "Text",
            "content": {"type": "text", "text": text},
        }
        if ro_index is not None:
            rule["ro_index"] = ro_index
        rules.append(rule)
    case = LayoutDetectionTestCase(test_id="probe", group="test", file_path="/tmp/probe.pdf", test_rules=rules)

    evaluation = LayoutDetectionEvaluator().evaluate(_inference(), case)

    metrics = {m.metric_name: m.value for m in evaluation.metrics}
    assert metrics["af1"] == 1.0
    assert metrics["layout_attribution_pass_rate"] == 1.0


def test_parse_gt_elements_keeps_given_ro_index_and_zeroes_missing_or_null():
    """Bug: a fix that defaults every ro_index to 0 would silently drop the reading order rules do carry."""
    base = {"type": "layout", "bbox": [0.1, 0.1, 0.8, 0.1], "content": {"type": "text", "text": "x"}}
    elements = parse_gt_elements([{**base, "ro_index": 3}, {**base, "ro_index": None}, base])

    assert [e.ro_index for e in elements] == [3, 0, 0]
