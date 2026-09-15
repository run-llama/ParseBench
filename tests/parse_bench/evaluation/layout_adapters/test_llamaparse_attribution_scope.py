"""LlamaParse attribution claims are restricted to layout-aware segments."""

from __future__ import annotations

from datetime import datetime

from parse_bench.evaluation.layout_adapters.adapters import (
    LlamaParseLayoutAdapter,
    _resolve_llamaparse_pages,
)
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _make_llamaparse_result(layout_pages: list[ParseLayoutPageIR]) -> InferenceResult:
    now = datetime.now()
    output = ParseOutput(
        task_type="parse",
        example_id="doc-1",
        pipeline_name="llamaparse",
        pages=[PageIR(page_index=0, markdown="Alpha\n\nBeta")],
        layout_pages=layout_pages,
        markdown="Alpha\n\nBeta",
    )
    return InferenceResult(
        request=InferenceRequest(
            example_id="doc-1",
            source_file_path="/tmp/doc-1.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline_name="llamaparse",
        product_type=ProductType.PARSE,
        raw_output={},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _pages() -> list[ParseLayoutPageIR]:
    return [
        ParseLayoutPageIR(
            page_number=1,
            width=1000.0,
            height=1000.0,
            md="Alpha\n\nBeta",
            items=[
                LayoutItemIR(
                    type="text",
                    value="Alpha",
                    bbox=LayoutSegmentIR(x=100.0, y=100.0, w=200.0, h=50.0, label="text", confidence=1.0),
                    layout_segments=[
                        LayoutSegmentIR(
                            x=100.0,
                            y=100.0,
                            w=200.0,
                            h=50.0,
                            label="text",
                            confidence=1.0,
                            start_index=0,
                            end_index=4,
                        )
                    ],
                ),
                # Coarse item bbox only: no layout-aware segments.
                LayoutItemIR(
                    type="text",
                    value="Beta",
                    bbox=LayoutSegmentIR(x=100.0, y=300.0, w=200.0, h=50.0, label="text", confidence=1.0),
                    layout_segments=[],
                ),
            ],
        )
    ]


def test_resolve_pages_without_bbox_fallback_drops_synthesized_segments() -> None:
    result = _make_llamaparse_result(_pages())

    with_fallback = _resolve_llamaparse_pages(result)
    assert [len(item.get("layoutAwareBbox") or []) for item in with_fallback[0]["items"]] == [1, 1]

    without_fallback = _resolve_llamaparse_pages(result, include_bbox_segment_fallback=False)
    items = without_fallback[0]["items"]
    assert "layoutAwareBbox" in items[0]
    assert "layoutAwareBbox" not in items[1]
    # The coarse bbox itself is untouched so layout detection still sees it.
    assert items[1]["bBox"]["x"] == 100.0


def test_llamaparse_attribution_blocks_only_from_layout_aware_segments() -> None:
    result = _make_llamaparse_result(_pages())
    adapter = LlamaParseLayoutAdapter()
    layout_output = adapter.to_layout_output(result)

    # Layout detection keeps both items (bbox fallback still applies there).
    assert len(layout_output.predictions) == 2

    blocks = adapter.to_attribution_blocks(layout_output, page_number=1)
    assert [block.text for block in blocks] == ["Alpha"]
