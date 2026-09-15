"""Regression tests for ``project_layout_predictions`` output shape."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from parse_bench.evaluation.layout_label_mappers.projection import project_layout_predictions
from parse_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
)
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _make_inference_result(layout_output: LayoutOutput) -> InferenceResult:
    request = InferenceRequest(
        example_id=layout_output.example_id,
        source_file_path="/tmp/fake.png",
        product_type=ProductType.PARSE,
    )
    now = datetime.now(UTC)
    return InferenceResult(
        request=request,
        pipeline_name=layout_output.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output={},
        output=layout_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )


def test_legacy_predictions_branch_projects_unchanged() -> None:
    """Flat ``predictions`` project through the llamaparse mapper chain and
    carry a copied ``attributes`` dict."""
    prediction = LayoutPrediction(
        bbox=[100.0, 200.0, 300.0, 400.0],
        score=0.9,
        label="text",  # LlamaParse V2 string label -> Canonical "Text"
        page=None,
        attributes={"scope": "region"},
        provider_metadata={"order_index": 3},
    )
    layout_output = LayoutOutput(
        example_id="doc1/page-1",
        pipeline_name="llamaparse",
        model=LayoutDetectionModel.LLAMAPARSE,
        image_width=1000,
        image_height=1000,
        predictions=[prediction],
    )
    result = _make_inference_result(layout_output)
    projected = project_layout_predictions(result, layout_output, evaluation_view="core")

    assert len(projected) == 1
    entry = projected[0]
    assert entry["bbox"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert entry["score"] == pytest.approx(0.9)
    assert isinstance(entry["score"], float)
    assert entry["class_name"] == "Text"
    assert entry["order_index"] == 3
    assert entry["attributes"] == {"scope": "region"}
    # Copied, not aliased: mutating the projection must not leak upstream.
    entry["attributes"]["scope"] = "mark"
    assert prediction.attributes == {"scope": "region"}


def test_projection_emits_empty_attributes_dict_when_absent() -> None:
    layout_output = LayoutOutput(
        example_id="doc1/page-1",
        pipeline_name="llamaparse",
        model=LayoutDetectionModel.LLAMAPARSE,
        image_width=1000,
        image_height=1000,
        predictions=[LayoutPrediction(bbox=[0.0, 0.0, 10.0, 10.0], score=0.5, label="text")],
    )
    result = _make_inference_result(layout_output)
    projected = project_layout_predictions(result, layout_output, evaluation_view="core")
    assert projected[0]["attributes"] == {}
