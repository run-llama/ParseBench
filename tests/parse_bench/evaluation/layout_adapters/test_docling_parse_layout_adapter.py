from __future__ import annotations

from datetime import datetime

from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers.projection import project_layout_predictions
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _make_docling_parse_inference_result(*, layout_pages: list[ParseLayoutPageIR], markdown: str) -> InferenceResult:
    now = datetime.now()
    output = ParseOutput(
        task_type="parse",
        example_id="doc-1",
        pipeline_name="docling_parse",
        pages=[PageIR(page_index=0, markdown=markdown)],
        layout_pages=layout_pages,
        markdown=markdown,
    )
    return InferenceResult(
        request=InferenceRequest(
            example_id="doc-1",
            source_file_path="/tmp/doc-1.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline_name="docling_parse",
        product_type=ProductType.PARSE,
        raw_output={"docling_document": {"schema_name": "DoclingDocument"}},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_docling_parse_projection_maps_reference_and_document_index_distinctly() -> None:
    inference_result = _make_docling_parse_inference_result(
        markdown="References\n\n<table><tr><td>Contents</td></tr></table>",
        layout_pages=[
            ParseLayoutPageIR(
                page_number=1,
                width=1000.0,
                height=1000.0,
                md="References\n\n<table><tr><td>Contents</td></tr></table>",
                items=[
                    LayoutItemIR(
                        type="text",
                        value="References entry",
                        bbox=LayoutSegmentIR(x=0.1, y=0.1, w=0.3, h=0.05, label="reference"),
                        layout_segments=[
                            LayoutSegmentIR(
                                x=0.1,
                                y=0.1,
                                w=0.3,
                                h=0.05,
                                label="reference",
                                start_index=0,
                                end_index=9,
                            )
                        ],
                    ),
                    LayoutItemIR(
                        type="table",
                        value="<table><tr><td>Contents</td></tr></table>",
                        bbox=LayoutSegmentIR(x=0.1, y=0.2, w=0.4, h=0.2, label="document_index"),
                        layout_segments=[
                            LayoutSegmentIR(
                                x=0.1,
                                y=0.2,
                                w=0.4,
                                h=0.2,
                                label="document_index",
                            )
                        ],
                    ),
                ],
            )
        ],
    )

    adapter = create_layout_adapter_for_result(inference_result)
    layout_output = adapter.to_layout_output(inference_result)
    projected = project_layout_predictions(
        inference_result,
        layout_output,
        evaluation_view="canonical",
        target_ontology="canonical",
    )

    assert layout_output.model.value == "docling_parse_layout"
    assert [prediction["class_name"] for prediction in projected] == ["Text", "Document Index"]


def test_docling_parse_adapter_preserves_spans_and_table_text_for_attribution() -> None:
    inference_result = _make_docling_parse_inference_result(
        markdown="AlphaBeta\n\n<table><tr><td>A</td></tr></table>",
        layout_pages=[
            ParseLayoutPageIR(
                page_number=1,
                width=1000.0,
                height=1000.0,
                md="AlphaBeta\n\n<table><tr><td>A</td></tr></table>",
                items=[
                    LayoutItemIR(
                        type="text",
                        value="AlphaBeta",
                        bbox=LayoutSegmentIR(x=0.1, y=0.1, w=0.2, h=0.05, label="text"),
                        layout_segments=[
                            LayoutSegmentIR(
                                x=0.1,
                                y=0.1,
                                w=0.2,
                                h=0.05,
                                label="text",
                                start_index=0,
                                end_index=4,
                            )
                        ],
                    ),
                    LayoutItemIR(
                        type="table",
                        value="<table><tr><td>A</td></tr></table>",
                        bbox=LayoutSegmentIR(x=0.1, y=0.2, w=0.3, h=0.2, label="table"),
                        layout_segments=[
                            LayoutSegmentIR(
                                x=0.1,
                                y=0.2,
                                w=0.3,
                                h=0.2,
                                label="table",
                            )
                        ],
                    ),
                ],
            )
        ],
    )

    adapter = create_layout_adapter_for_result(inference_result)
    layout_output = adapter.to_layout_output(inference_result)
    blocks = adapter.to_attribution_blocks(layout_output, page_number=1)

    assert len(layout_output.predictions) == 2
    assert layout_output.predictions[1].content is not None
    assert layout_output.predictions[1].content.type == "table"
    assert [block.text for block in blocks] == ["Alpha", "A"]
    assert [block.label for block in blocks] == ["text", "table"]


def test_docling_parse_adapter_table_type_check_is_case_insensitive() -> None:
    inference_result = _make_docling_parse_inference_result(
        markdown="<table><tr><td>Cell</td></tr></table>",
        layout_pages=[
            ParseLayoutPageIR(
                page_number=1,
                width=1000.0,
                height=1000.0,
                md="<table><tr><td>Cell</td></tr></table>",
                items=[
                    LayoutItemIR(
                        type="Table ",
                        value="<table><tr><td>Cell</td></tr></table>",
                        bbox=LayoutSegmentIR(x=0.1, y=0.2, w=0.3, h=0.2, label="table"),
                        layout_segments=[LayoutSegmentIR(x=0.1, y=0.2, w=0.3, h=0.2, label="table")],
                    ),
                ],
            )
        ],
    )

    adapter = create_layout_adapter_for_result(inference_result)
    layout_output = adapter.to_layout_output(inference_result)
    blocks = adapter.to_attribution_blocks(layout_output, page_number=1)

    assert [block.text for block in blocks] == ["Cell"]


def test_docling_parse_adapter_skips_bbox_only_items_for_attribution() -> None:
    inference_result = _make_docling_parse_inference_result(
        markdown="Container fallback",
        layout_pages=[
            ParseLayoutPageIR(
                page_number=1,
                width=1000.0,
                height=1000.0,
                md="Container fallback",
                items=[
                    LayoutItemIR(
                        type="text",
                        value="Container fallback",
                        bbox=LayoutSegmentIR(x=0.1, y=0.1, w=0.2, h=0.05, label="text"),
                        layout_segments=[],
                    )
                ],
            )
        ],
    )

    adapter = create_layout_adapter_for_result(inference_result)
    layout_output = adapter.to_layout_output(inference_result)
    blocks = adapter.to_attribution_blocks(layout_output, page_number=1)

    assert len(layout_output.predictions) == 1
    assert blocks == []
