"""A merged anyformat block is scored whole and by its detections; an unmerged one once."""

from __future__ import annotations

from datetime import datetime

import pytest

from parse_bench.evaluation.layout_adapters.adapters import AnyformatLayoutAdapter
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutRegionIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _result(items: list[LayoutItemIR]) -> InferenceResult:
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(
            example_id="doc-1", source_file_path="/tmp/doc-1.pdf", product_type=ProductType.PARSE
        ),
        pipeline_name="anyformat_standard",
        product_type=ProductType.PARSE,
        raw_output={"results": {}},
        output=ParseOutput(
            task_type="parse",
            example_id="doc-1",
            pipeline_name="anyformat_standard",
            pages=[PageIR(page_index=0, markdown="Alpha\n\nBeta")],
            layout_pages=[
                ParseLayoutPageIR(page_number=1, width=1000.0, height=1000.0, md="Alpha\n\nBeta", items=items)
            ],
            markdown="Alpha\n\nBeta",
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _region(y: float, text: str) -> LayoutRegionIR:
    return LayoutRegionIR(
        type="text", text=text, bbox=LayoutSegmentIR(x=0.1, y=y, w=0.2, h=0.05, label="text", confidence=1.0)
    )


def test_a_block_of_two_detections_is_scored_at_both_levels():
    item = LayoutItemIR(
        type="text",
        value="Alpha Beta",
        bbox=LayoutSegmentIR(x=0.1, y=0.1, w=0.2, h=0.15, label="text", confidence=1.0),
        regions=[_region(0.1, "Alpha"), _region(0.2, "Beta")],
    )

    predictions = AnyformatLayoutAdapter().to_layout_output(_result([item])).predictions

    assert [p.content.text for p in predictions] == ["Alpha Beta", "Alpha", "Beta"]
    assert predictions[0].bbox == pytest.approx([100.0, 100.0, 300.0, 250.0])


def test_a_block_of_one_detection_is_scored_once():
    item = LayoutItemIR(
        type="text",
        value="Alpha",
        bbox=LayoutSegmentIR(x=0.1, y=0.1, w=0.2, h=0.05, label="text", confidence=1.0),
        regions=[_region(0.1, "Alpha")],
    )

    [prediction] = AnyformatLayoutAdapter().to_layout_output(_result([item])).predictions

    assert prediction.bbox == pytest.approx([100.0, 100.0, 300.0, 150.0])
    assert prediction.content.text == "Alpha"
