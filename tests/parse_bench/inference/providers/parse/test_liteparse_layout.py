"""LiteParse ``--extract-blocks`` output -> layout_pages -> LayoutOutput."""

from datetime import datetime

from parse_bench.evaluation.layout_adapters.adapters import LiteParseLayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.inference.providers.parse.liteparse import _block_text, _build_layout_pages
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult

_PAGE = {
    "page_index": 0,
    "width": 612.0,
    "height": 792.0,
    "text": "Title\nbody",
    "blocks": [
        {
            "kind": "heading",
            "level": 1,
            "text": "Title",
            "bbox": {"x": 61.2, "y": 79.2, "width": 306.0, "height": 15.84},
        },
        {"kind": "paragraph", "text": "body", "bbox": {"x": 61.2, "y": 158.4, "width": 489.6, "height": 79.2}},
        {"kind": "rule", "bbox": {"x": 0, "y": 300, "width": 612, "height": 0.5}},
        {
            "kind": "table",
            "header": [{"text": "a"}, {"text": "b"}],
            "rows": [[{"text": "1"}, {"text": "2"}]],
            "bbox": {"x": 61.2, "y": 396.0, "width": 489.6, "height": 158.4},
        },
        {"kind": "figure", "id": "img1", "bbox": {"x": 0, "y": 600, "width": 0, "height": 10}},
    ],
}


def test_build_layout_pages_normalizes_and_filters() -> None:
    pages = _build_layout_pages([_PAGE])
    assert len(pages) == 1
    page = pages[0]
    assert page.page_number == 1 and page.width == 612.0 and page.height == 792.0
    # rule kind and zero-width figure are dropped
    assert [item.type for item in page.items] == ["Section-header", "Text", "Table"]
    heading = page.items[0]
    assert heading.bbox is not None
    assert abs(heading.bbox.x - 0.1) < 1e-9 and abs(heading.bbox.y - 0.1) < 1e-9
    assert abs(heading.bbox.w - 0.5) < 1e-9 and abs(heading.bbox.h - 0.02) < 1e-9
    assert heading.layout_segments[0].label == "Section-header"
    assert page.items[2].value == "a | b\n1 | 2"


def test_build_layout_pages_skips_pages_without_blocks_or_size() -> None:
    assert _build_layout_pages([{"page_index": 0, "width": 612, "height": 792, "blocks": []}]) == []
    assert _build_layout_pages([{"page_index": 0, "blocks": _PAGE["blocks"]}]) == []


def test_block_text_variants() -> None:
    assert _block_text({"kind": "paragraph", "text": "hi"}) == "hi"
    assert _block_text({"kind": "code", "lines": ["x = 1", "y = 2"]}) == "x = 1\ny = 2"
    assert _block_text({"kind": "figure"}) == ""


def _result() -> InferenceResult:
    now = datetime.now()
    output = ParseOutput(
        task_type="parse",
        example_id="layout/doc",
        pipeline_name="liteparse_markdown",
        markdown="Title\n\nbody",
        layout_pages=_build_layout_pages([_PAGE]),
    )
    return InferenceResult(
        request=InferenceRequest(example_id="layout/doc", source_file_path="doc.pdf", product_type="parse"),
        pipeline_name="liteparse_markdown",
        product_type="parse",
        raw_output={"pages": [_PAGE], "output_format": "markdown", "text": ""},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_adapter_resolves_and_scales_to_points() -> None:
    result = _result()
    adapter = create_layout_adapter_for_result(result)
    assert isinstance(adapter, LiteParseLayoutAdapter)
    layout = adapter.to_layout_output(result)
    assert layout.model is LayoutDetectionModel.LITEPARSE_LAYOUT
    assert (layout.image_width, layout.image_height) == (612, 792)
    assert [p.label for p in layout.predictions] == ["Section-header", "Text", "Table"]
    x1, y1, x2, y2 = layout.predictions[0].bbox
    assert (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)) == (61.2, 79.2, 367.2, 95.04)
    assert layout.predictions[2].content is not None
    assert adapter.to_layout_output(result, page_filter=2).predictions == []
