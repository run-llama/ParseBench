"""Exercise real normalization emitters, projections and attribution adapters."""

from datetime import datetime

import pytest

from parse_bench.evaluation.layout_adapters.adapters import (
    DoclingParseLayoutAdapter,
    LlamaParseLayoutAdapter,
    Qwen3VLLayoutAdapter,
    _resolve_llamaparse_pages,
)
from parse_bench.evaluation.metrics.attribution.core import compute_attribution_metrics, parse_gt_elements
from parse_bench.inference.layout_extraction import extract_all_layouts_from_llamaparse_output
from parse_bench.inference.providers.parse.llamaparse_v2_normalization import (
    build_pages_from_cli2_raw_payload,
    build_pages_from_sdk_response_payload,
    layout_pages_to_legacy_pages_payload,
)
from parse_bench.layout_projection import project_to_canonical_predictions, project_to_core_predictions
from parse_bench.schemas.layout_detection_output import LayoutOutput
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _result(raw=None, pages=None, output=None):
    return InferenceResult(
        request=InferenceRequest(example_id="probe", source_file_path="/tmp/probe.pdf", product_type=ProductType.PARSE),
        pipeline_name="llamaparse",
        product_type=ProductType.PARSE,
        raw_output=raw or {},
        output=output
        or ParseOutput(example_id="probe", pipeline_name="llamaparse", markdown="alpha", layout_pages=pages or []),
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
        latency_in_ms=0,
    )


def _box(angle, *, normalized=False):
    scale = 100 if normalized else 1
    return dict(
        x=20 / scale,
        y=45 / scale,
        w=60 / scale,
        h=5 / scale,
        label="text",
        confidence=1,
        start_index=0,
        end_index=4,
        **({} if angle is None else {"r": angle}),
    )


@pytest.mark.parametrize("confidence,expected", [("omitted", 0), (None, 0), (0, 0), (0.87, 0.87)])
def test_nullable_confidence_roundtrip(confidence, expected):
    box = _box(30)
    box.pop("confidence")
    if confidence != "omitted":
        box["confidence"] = confidence
    output = extract_all_layouts_from_llamaparse_output(
        {
            "pages": [
                {"width": 100, "height": 100, "items": [{"type": "text", "value": "alpha", "layoutAwareBbox": [box]}]}
            ]
        }
    )
    restored = LayoutOutput.model_validate_json(output.model_dump_json())
    assert restored.predictions[0].score == expected
    assert restored.layout_pages[0].items[0].layout_segments[0].confidence == expected


def test_invalid_confidence_remains_an_error():
    box = {**_box(30), "confidence": "bad"}
    with pytest.raises(ValueError):
        extract_all_layouts_from_llamaparse_output(
            {
                "pages": [
                    {
                        "width": 100,
                        "height": 100,
                        "items": [{"type": "text", "value": "alpha", "layoutAwareBbox": [box]}],
                    }
                ]
            }
        )


@pytest.mark.parametrize("angle", [None, 0, 30, -30, 90, -90, 180])
@pytest.mark.parametrize("route", ["raw", "cli2", "sdk", "ir", "generic", "docling"])
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("kind", ["text", "table"])
def test_full_rotation_routes(route, angle, count, kind):
    box = {**_box(angle), "label": kind}
    segment = LayoutSegmentIR(**box)
    ir_page = ParseLayoutPageIR(
        page_number=1,
        width=100,
        height=100,
        items=[LayoutItemIR(type=kind, value="alpha", bbox=segment, layout_segments=[segment] * count)],
    )
    legacy_box = {**box, "startIndex": 0, "endIndex": 4}
    raw_pages = [
        {
            "page": 1,
            "width": 100,
            "height": 100,
            "items": [{"type": kind, "value": "alpha", "bBox": legacy_box, "layoutAwareBbox": [legacy_box] * count}],
        }
    ]
    adapter = LlamaParseLayoutAdapter()
    if route == "raw":
        result = _result(raw={"pages": raw_pages})
    elif route in ("cli2", "sdk"):
        # A real nested SDK container exercises the recursive flattening arm.
        item = {"type": kind, "md": "alpha", "value": "alpha", "bbox": [box] * count}
        if kind == "table":
            item.update(csv="alpha", html="alpha", rows=[["alpha"]])
        page = {
            "page_number": 1,
            "page_width": 100,
            "page_height": 100,
            "success": True,
            "items": [{"type": "list", "ordered": False, "md": "alpha", "items": [item]}] if kind == "text" else [item],
        }
        raw = (
            {"v2_items": {"pages": [page]}}
            if route == "cli2"
            else {"job": {"id": "probe", "project_id": "probe", "status": "COMPLETED"}, "items": {"pages": [page]}}
        )
        emitter = build_pages_from_cli2_raw_payload if route == "cli2" else build_pages_from_sdk_response_payload
        emitted = emitter(raw_payload=raw, output_tables_as_markdown=False)[0]["items"][0]
        assert emitted["bBox"].get("r") == angle
        assert [s.get("r") for s in emitted["layoutAwareBbox"]] == [angle] * count
        result = _result(raw=raw)
    elif route == "ir":
        emitted = layout_pages_to_legacy_pages_payload([ir_page])[0]["items"][0]
        assert emitted["bBox"].get("r") == angle
        assert [s.get("r") for s in emitted["layoutAwareBbox"]] == [angle] * count
        result = _result(pages=[ir_page])
    elif route == "generic":
        output = extract_all_layouts_from_llamaparse_output({"pages": raw_pages})
        output = LayoutOutput.model_validate_json(output.model_dump_json())
        assert output.layout_pages[0].items[0].bbox.r == angle
        assert [p.r for p in project_to_canonical_predictions(output)] == [angle] * count
        assert [p.r for p in project_to_core_predictions(output)] == [angle] * count
        result = _result(output=output)
    else:
        normalized = LayoutSegmentIR(**_box(angle, normalized=True))
        ir_page.items[0].bbox = normalized
        ir_page.items[0].layout_segments = [normalized] * count
        result = _result(pages=[ir_page])
        adapter = DoclingParseLayoutAdapter()
    layout = adapter.to_layout_output(result)
    blocks = adapter.to_attribution_blocks(layout, page_number=1)
    assert [b.r for b in blocks] == [angle] * count
    assert [b.text for b in blocks] == ["alpha"] * count
    assert all((b.page_width, b.page_height) == (100, 100) for b in blocks)
    gt = parse_gt_elements(
        [
            {
                "type": "layout",
                "bbox": [0.2, 0.45, 0.6, 0.05],
                "r": angle,
                "canonical_class": "Text",
                "content": {"type": "text", "text": "alpha"},
            }
        ]
    )
    assert compute_attribution_metrics(gt, blocks).af1 == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_ir_fallback_retains_rotation_without_expanding_attribution_scope(enabled):
    segment = LayoutSegmentIR(**_box(-30))
    page = ParseLayoutPageIR(
        page_number=1, width=100, height=100, items=[LayoutItemIR(type="text", value="alpha", bbox=segment)]
    )
    emitted = layout_pages_to_legacy_pages_payload([page], include_bbox_segment_fallback=enabled)[0]["items"][0]
    assert emitted["bBox"]["r"] == -30
    assert bool(emitted.get("layoutAwareBbox")) is enabled
    if enabled:
        assert emitted["layoutAwareBbox"][0]["r"] == -30
    result = _result(pages=[page])
    assert not _resolve_llamaparse_pages(result, include_bbox_segment_fallback=False)[0]["items"][0].get(
        "layoutAwareBbox"
    )
    adapter = LlamaParseLayoutAdapter()
    assert adapter.to_attribution_blocks(adapter.to_layout_output(result), page_number=1) == []


@pytest.mark.parametrize("reverse", [False, True])
def test_generic_pages_keep_their_own_aspect_ratio(reverse):
    pages = [
        ParseLayoutPageIR(
            page_number=i,
            width=w,
            height=h,
            items=[
                LayoutItemIR(
                    type="text",
                    value="alpha",
                    layout_segments=[LayoutSegmentIR(x=0.4, y=0.4, w=0.2, h=0.1, r=90, confidence=1, label="text")],
                )
            ],
        )
        for i, w, h in [(1, 200, 100), (2, 100, 200)]
    ]
    if reverse:
        pages.reverse()
    adapter = Qwen3VLLayoutAdapter()
    output = adapter.to_layout_output(_result(pages=pages))
    for i, w, h in [(1, 200, 100), (2, 100, 200)]:
        block = adapter.to_attribution_blocks(output, page_number=i)[0]
        assert block.bbox_xyxy == pytest.approx([0.4, 0.4, 0.6, 0.5])
        assert (block.page_width, block.page_height) == (w, h)
        gt_box = [0.475, 0.25, 0.05, 0.4] if i == 1 else [0.3, 0.425, 0.4, 0.05]
        gt = parse_gt_elements(
            [
                {
                    "type": "layout",
                    "bbox": gt_box,
                    "r": 0,
                    "canonical_class": "Text",
                    "content": {"type": "text", "text": "alpha"},
                }
            ]
        )
        assert compute_attribution_metrics(gt, [block]).af1 == 1


@pytest.mark.parametrize("angle,expected", [(30, 1), (120, 0)])
def test_shared_evaluator_rotated_diagnostics_agree_with_scores(angle, expected):
    from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
    from parse_bench.test_cases.schema import LayoutDetectionTestCase

    box = _box(angle)
    output = extract_all_layouts_from_llamaparse_output(
        {
            "pages": [
                {"width": 100, "height": 100, "items": [{"type": "text", "value": "alpha", "layoutAwareBbox": [box]}]}
            ]
        }
    )
    inference = _result(output=output).model_copy(update={"product_type": ProductType.LAYOUT_DETECTION})
    case = LayoutDetectionTestCase(
        test_id="probe",
        group="test",
        file_path="/tmp/probe.pdf",
        test_rules=[
            {
                "type": "layout",
                "page": 1,
                "bbox": [0.2, 0.45, 0.6, 0.05],
                "r": 30,
                "canonical_class": "Text",
                "content": {"type": "text", "text": "alpha"},
            }
        ],
    )
    evaluation = LayoutDetectionEvaluator().evaluate(inference, case)
    metrics = {m.metric_name: m for m in evaluation.metrics}
    assert metrics["af1"].value == expected
    assert metrics["layout_attribution_pass_rate"].value == expected
    rules = metrics["layout_element_rule_pass_rate"].metadata["rule_results"]
    assert rules[0]["attribution_pass"] is bool(expected)


def test_multisize_raw_pages_retain_geometry_after_serialization():
    pages = []
    for number, width, height in [(1, 200, 100), (2, 100, 200)]:
        box = {"x": 0.4 * width, "y": 0.4 * height, "w": 0.2 * width, "h": 0.1 * height, "r": 90, "label": "text"}
        pages.append(
            {
                "page": number,
                "width": width,
                "height": height,
                "items": [{"type": "text", "value": "alpha", "layoutAwareBbox": [box]}],
            }
        )
    output = extract_all_layouts_from_llamaparse_output({"pages": pages})
    output = LayoutOutput.model_validate_json(output.model_dump_json())
    adapter = LlamaParseLayoutAdapter()
    converted = adapter.to_layout_output(_result(output=output))
    for page, width, height in [(1, 200, 100), (2, 100, 200)]:
        block = adapter.to_attribution_blocks(converted, page_number=page)[0]
        assert (block.page_width, block.page_height) == (width, height)
        assert block.bbox_xyxy == pytest.approx([0.4, 0.4, 0.6, 0.5])


@pytest.mark.parametrize("width,height", [(0, 100), (100, 0), (-1, 100)])
def test_adapter_refuses_invalid_page_geometry(width, height):
    box = {**_box(30), "startIndex": 0, "endIndex": 4}
    result = _result(
        raw={
            "pages": [
                {
                    "width": width,
                    "height": height,
                    "items": [{"type": "text", "value": "alpha", "bBox": box, "layoutAwareBbox": [box]}],
                }
            ]
        }
    )
    adapter = LlamaParseLayoutAdapter()
    with pytest.raises(ValueError, match="page dimensions"):
        adapter.to_attribution_blocks(adapter.to_layout_output(result), page_number=1)


@pytest.mark.parametrize("use_items", [False, True])
def test_scored_label_projection_keeps_rotation_and_page_dimensions(use_items):
    from parse_bench.evaluation.layout_label_mappers.projection import project_layout_predictions

    box = {**_box(-30), "label": "text"}
    output = extract_all_layouts_from_llamaparse_output(
        {
            "pages": [
                {"page": 1, "width": 200, "height": 100, "items": []},
                {
                    "page": 2,
                    "width": 100,
                    "height": 200,
                    "items": [{"type": "text", "value": "alpha", "layoutAwareBbox": [box]}],
                },
            ]
        }
    )
    if not use_items:
        output.layout_pages = [page.model_copy(update={"items": []}) for page in output.layout_pages]
    projected = project_layout_predictions(_result(output=output), output)
    assert len(projected) == 1
    assert projected[0]["bbox"] == pytest.approx([0.2, 0.225, 0.8, 0.25])
    assert projected[0]["r"] == -30


@pytest.mark.parametrize("adapter_class", [DoclingParseLayoutAdapter, Qwen3VLLayoutAdapter])
def test_normalized_ir_zero_confidence_stays_zero(adapter_class):
    segment = LayoutSegmentIR(**{**_box(30, normalized=True), "confidence": 0})
    page = ParseLayoutPageIR(
        page_number=1,
        width=100,
        height=200,
        items=[LayoutItemIR(type="text", value="alpha", bbox=segment, layout_segments=[segment])],
    )
    assert adapter_class().to_layout_output(_result(pages=[page])).predictions[0].score == 0
