"""Phase 1 dual-read regression tests for ``project_layout_predictions``.

Exercises both branches of the migration:
  * Legacy branch — ``layout_output.predictions`` populated, empty
    ``layout_pages``. Behavior unchanged from pre-migration.
  * New branch — ``layout_output.layout_pages[*].items`` populated,
    empty ``predictions``. Projection iterates items, coerces
    ``item.type`` to ``CanonicalLabel``, and emits the same projected
    dict shape as the legacy branch so downstream matching code is
    unaffected.

These tests are Phase-1-specific scaffolding and will be simplified
once Phase 3 drops the legacy ``predictions`` path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from parse_bench.evaluation.layout_label_mappers.projection import (
    project_layout_predictions,
)
from parse_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
)
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
)
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


def _make_pipeline_spec() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="docling_layout_old",
        provider_name="docling_layout",
        product_type=ProductType.PARSE,
        config={},
    )


def test_legacy_predictions_branch_projects_unchanged() -> None:
    """When ``layout_pages`` is empty, fall back to the flat predictions
    list and produce the same projected dicts as pre-migration. Exercises
    the llamaparse mapper chain (still active for pre-Phase-2 providers).
    """
    layout_output = LayoutOutput(
        example_id="doc1/page-1",
        pipeline_name="llamaparse",
        model=LayoutDetectionModel.LLAMAPARSE,
        image_width=1000,
        image_height=1000,
        predictions=[
            LayoutPrediction(
                bbox=[100.0, 200.0, 300.0, 400.0],
                score=0.9,
                label="text",  # LlamaParse V2 string label -> Canonical "Text"
                page=None,
                attributes={},
                provider_metadata={},
            ),
        ],
    )
    result = _make_inference_result(layout_output)
    projected = project_layout_predictions(result, layout_output, evaluation_view="core")

    assert len(projected) == 1
    entry = projected[0]
    # Bbox normalized to [0,1]
    assert entry["bbox"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert entry["score"] == pytest.approx(0.9)
    # Text maps to basic "Text" in the default basic ontology
    assert entry["class_name"] == "Text"
    assert entry["attributes"] == {}


def test_new_layout_pages_branch_is_preferred_when_items_present() -> None:
    """When ``layout_pages`` has any items, iterate them instead of
    ``predictions``. ``item.type`` is treated as a canonical label."""
    page = ParseLayoutPageIR(
        page_number=1,
        width=1000.0,
        height=1000.0,
        items=[
            LayoutItemIR(
                type="Text",  # Canonical17 string
                layout_segments=[
                    LayoutSegmentIR(x=100.0, y=200.0, w=200.0, h=200.0, confidence=0.9),
                ],
                score=0.9,
                attributes={},
            ),
        ],
    )
    layout_output = LayoutOutput(
        example_id="doc1/page-1",
        pipeline_name="docling_layout_old",
        model=LayoutDetectionModel.DOCLING_LAYOUT_OLD,
        image_width=1000,
        image_height=1000,
        layout_pages=[page],
        # Even if ``predictions`` were populated, the new branch would win.
        predictions=[
            LayoutPrediction(
                bbox=[999.0, 999.0, 1000.0, 1000.0],
                score=0.01,
                label="9",
                provider_metadata={},
            ),
        ],
    )
    result = _make_inference_result(layout_output)
    projected = project_layout_predictions(result, layout_output, evaluation_view="core")

    assert len(projected) == 1  # ignored the legacy prediction
    entry = projected[0]
    # xywh → xyxy, then normalized by image dims
    assert entry["bbox"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert entry["score"] == pytest.approx(0.9)
    assert entry["class_name"] == "Text"
    assert entry["page"] == 1
    assert entry["attributes"] == {}


def test_new_branch_preserves_mark_scope_attributes() -> None:
    """The new branch must copy ``item.attributes`` through so the
    evaluator's mark-scope dispatch (PR #689) continues to work."""
    page = ParseLayoutPageIR(
        page_number=1,
        width=1000.0,
        height=1000.0,
        items=[
            LayoutItemIR(
                type="Checkbox-Selected",
                layout_segments=[
                    LayoutSegmentIR(x=10.0, y=10.0, w=20.0, h=20.0, confidence=0.8),
                ],
                score=0.8,
                attributes={"scope": "mark"},
            ),
        ],
    )
    layout_output = LayoutOutput(
        example_id="doc1/page-1",
        pipeline_name="checkbox_detector_yolov8_1280",
        model=LayoutDetectionModel.CHECKBOX_DETECTOR_YOLOV8,
        image_width=1000,
        image_height=1000,
        layout_pages=[page],
    )
    result = _make_inference_result(layout_output)
    # canonical view keeps Checkbox-Selected (dropped by Core11 filter)
    projected = project_layout_predictions(
        result, layout_output, evaluation_view="canonical", target_ontology="canonical"
    )

    assert len(projected) == 1
    entry = projected[0]
    assert entry["attributes"] == {"scope": "mark"}
    assert entry["class_name"] == "Checkbox-Selected"


def test_new_branch_skips_unknown_item_type() -> None:
    """An item whose ``type`` is not a Canonical17 label is skipped,
    matching the legacy mapper's ``None``-return behavior."""
    page = ParseLayoutPageIR(
        page_number=1,
        width=1000.0,
        height=1000.0,
        items=[
            LayoutItemIR(
                type="not-a-canonical-label",
                layout_segments=[
                    LayoutSegmentIR(x=0.0, y=0.0, w=10.0, h=10.0, confidence=0.5),
                ],
                score=0.5,
            ),
            LayoutItemIR(
                type="Text",
                layout_segments=[
                    LayoutSegmentIR(x=10.0, y=10.0, w=90.0, h=90.0, confidence=0.9),
                ],
                score=0.9,
            ),
        ],
    )
    layout_output = LayoutOutput(
        example_id="doc1/page-1",
        pipeline_name="docling_layout_old",
        model=LayoutDetectionModel.DOCLING_LAYOUT_OLD,
        image_width=1000,
        image_height=1000,
        layout_pages=[page],
    )
    result = _make_inference_result(layout_output)
    projected = project_layout_predictions(result, layout_output, evaluation_view="core")

    assert len(projected) == 1
    assert projected[0]["class_name"] == "Text"


def test_new_branch_coerces_none_score_to_zero() -> None:
    """``LayoutItemIR.score`` is ``float | None`` but downstream metric
    code assumes ``float``. Projection must coerce ``None`` to a float
    so ``np.array(..., dtype=float)`` and ``float(p["score"])`` don't
    explode on parse-origin items that legitimately carry no detector
    confidence.
    """
    page = ParseLayoutPageIR(
        page_number=1,
        width=1000.0,
        height=1000.0,
        items=[
            LayoutItemIR(
                type="Text",
                layout_segments=[LayoutSegmentIR(x=0.0, y=0.0, w=100.0, h=100.0)],
                score=None,  # parse-origin item: no detector confidence
            ),
        ],
    )
    layout_output = LayoutOutput(
        example_id="doc1",
        pipeline_name="docling_layout_old",
        model=LayoutDetectionModel.DOCLING_LAYOUT_OLD,
        image_width=1000,
        image_height=1000,
        layout_pages=[page],
    )
    result = _make_inference_result(layout_output)
    projected = project_layout_predictions(result, layout_output, evaluation_view="core")

    assert len(projected) == 1
    # Crucially, score is a float — not None.
    assert isinstance(projected[0]["score"], float)
    assert projected[0]["score"] == 0.0


def test_new_branch_honors_page_filter() -> None:
    pages = [
        ParseLayoutPageIR(
            page_number=1,
            width=1000.0,
            height=1000.0,
            items=[
                LayoutItemIR(
                    type="Text",
                    layout_segments=[LayoutSegmentIR(x=0.0, y=0.0, w=10.0, h=10.0)],
                    score=0.9,
                ),
            ],
        ),
        ParseLayoutPageIR(
            page_number=2,
            width=1000.0,
            height=1000.0,
            items=[
                LayoutItemIR(
                    type="Text",
                    layout_segments=[LayoutSegmentIR(x=50.0, y=50.0, w=10.0, h=10.0)],
                    score=0.8,
                ),
            ],
        ),
    ]
    layout_output = LayoutOutput(
        example_id="doc1",
        pipeline_name="docling_layout_old",
        model=LayoutDetectionModel.DOCLING_LAYOUT_OLD,
        image_width=1000,
        image_height=1000,
        layout_pages=pages,
    )
    result = _make_inference_result(layout_output)
    projected = project_layout_predictions(result, layout_output, evaluation_view="core", page_filter=2)

    assert len(projected) == 1
    assert projected[0]["page"] == 2
    assert projected[0]["score"] == pytest.approx(0.8)


def test_layout_output_can_carry_parse_shape_without_task_type_confusion() -> None:
    """LayoutOutput declares the same fields as ParseOutput but remains
    a distinct type. ``isinstance`` dispatch continues to distinguish
    them — this is load-bearing for evaluator runner dispatch.
    """
    from parse_bench.schemas.parse_output import ParseOutput

    layout_output = LayoutOutput(
        example_id="doc1/page-1",
        pipeline_name="docling_layout_old",
        model=LayoutDetectionModel.DOCLING_LAYOUT_OLD,
        image_width=1000,
        image_height=1000,
    )
    assert not isinstance(layout_output, ParseOutput)
    assert layout_output.task_type == "layout_detection"
    # Fields inherited-in-shape (not type) from ParseOutput are all present.
    assert layout_output.pages == []
    assert layout_output.layout_pages == []
    assert layout_output.grounded_pages == []
    assert layout_output.markdown == ""
    assert layout_output.job_id is None
