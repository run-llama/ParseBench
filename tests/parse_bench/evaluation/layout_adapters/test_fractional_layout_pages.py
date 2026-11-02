"""Layout adapters read layout pages whose boxes are 0-1 fractions of the page.

``LayoutSegmentIR`` allows boxes in page units or normalized to [0, 1]. Several
normalizers store fractions; adapters that assumed page units read them as boxes
a few units wide in the page's corner and localized nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from parse_bench.evaluation.layout_adapters.adapters import (
    LlamaParseLayoutAdapter,
    NutrientDwsLayoutAdapter,
    _looks_like_llamaparse_pages,
    layout_pages_in_page_units,
)
from parse_bench.evaluation.layout_label_mappers.base import MappingContext
from parse_bench.evaluation.layout_label_mappers.mappers import IndexedLayoutModelMapper, UpstageLabelMapper
from parse_bench.layout_label_mapping import UnknownRawLayoutLabelError
from parse_bench.schemas.layout_detection_output import (
    QWEN3VL_STR_TO_LABEL,
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
)
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _seg(x: float, y: float, w: float, h: float, label: str = "Text") -> LayoutSegmentIR:
    return LayoutSegmentIR(x=x, y=y, w=w, h=h, label=label)


def _page(segments: list[LayoutSegmentIR], width: float = 1000.0, height: float = 2000.0) -> ParseLayoutPageIR:
    items = [LayoutItemIR(type="text", value=f"block {i}", bbox=s, layout_segments=[s]) for i, s in enumerate(segments)]
    return ParseLayoutPageIR(page_number=1, width=width, height=height, items=items)


def _result(pages: list[ParseLayoutPageIR], pipeline_name: str, raw_output: dict[str, Any] | None = None):
    now = datetime.now()
    output = ParseOutput(
        task_type="parse",
        example_id="layout/doc",
        pipeline_name=pipeline_name,
        pages=[PageIR(page_index=0, markdown="block 0")],
        layout_pages=pages,
        markdown="block 0",
    )
    return InferenceResult(
        request=InferenceRequest(example_id="layout/doc", source_file_path="", product_type=ProductType.PARSE),
        pipeline_name=pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw_output or {},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_fractional_page_is_scaled_to_page_units() -> None:
    page = _page([_seg(0.1, 0.2, 0.3, 0.05), _seg(0.5, 0.5, 0.5, 0.5)])
    [scaled] = layout_pages_in_page_units([page])
    box = scaled.items[0].bbox
    assert (box.x, box.y, box.w, box.h) == pytest.approx((100.0, 400.0, 300.0, 100.0))
    assert scaled.items[0].layout_segments[0].x == pytest.approx(100.0)
    assert scaled.items[0].value == "block 0"  # content untouched


def test_page_unit_boxes_are_left_alone() -> None:
    page = _page([_seg(8.15, 24.8, 238.38, 40.54)], width=612, height=792)
    assert layout_pages_in_page_units([page]) == [page]


def test_page_without_a_real_size_is_left_alone() -> None:
    page = _page([_seg(0.1, 0.1, 0.2, 0.2)], width=1, height=1)
    assert layout_pages_in_page_units([page]) == [page]


def test_llamaparse_adapter_localizes_fractional_pages_with_raw_image_size() -> None:
    # Surya-style record: fractional boxes plus the page size in raw_output.
    page = _page([_seg(0.01, 0.028, 0.021, 0.055, "Page-header")], width=1632, height=2112)
    result = _result([page], "surya2_sdk", raw_output={"image_width": 1632, "image_height": 2112, "blocks": []})
    layout = LlamaParseLayoutAdapter().to_layout_output(result)
    assert layout.image_width == 1632
    x1, y1, x2, y2 = layout.predictions[0].bbox
    assert (x1, y1) == pytest.approx((16.32, 59.136))
    assert x2 - x1 == pytest.approx(0.021 * 1632)


def test_llamaparse_adapter_ignores_another_providers_native_pages() -> None:
    # Mistral / GLM keep their own ``pages`` in raw_output; layout_pages must win.
    page = _page([_seg(0.1, 0.1, 0.2, 0.05)], width=791, height=1023)
    native = {"pages": [{"index": 0, "markdown": "x", "items": [{"bbox": [1, 2, 3, 4], "label": "text", "text": "x"}]}]}
    layout = LlamaParseLayoutAdapter().to_layout_output(_result([page], "mistral_ocr_4", raw_output=native))
    assert len(layout.predictions) == 1


def test_looks_like_llamaparse_pages() -> None:
    assert _looks_like_llamaparse_pages([{"items": [{"bBox": {"x": 1}}]}])
    assert _looks_like_llamaparse_pages([{"items": [{"layoutAwareBbox": []}]}])
    assert _looks_like_llamaparse_pages([{"items": []}])
    assert not _looks_like_llamaparse_pages([{"items": [{"bbox": [1, 2, 3, 4], "label": "text"}]}])
    assert not _looks_like_llamaparse_pages([{"markdown": "x", "images": []}])


def test_nutrient_adapter_reads_fractional_boxes() -> None:
    page = _page([_seg(0.0136, 0.0318, 0.0142, 0.0501, "Page-header")], width=1700, height=2200)
    layout = NutrientDwsLayoutAdapter().to_layout_output(_result([page], "nutrient_dws_agentic"))
    x1, y1, x2, y2 = layout.predictions[0].bbox
    assert (x1, y1) == pytest.approx((0.0136 * 1700, 0.0318 * 2200))


def _context(model: LayoutDetectionModel) -> MappingContext:
    return MappingContext(
        provider_name=None,
        pipeline_name="p",
        model=model,
        raw_output={},
        layout_output=LayoutOutput(
            task_type="layout_detection",
            example_id="x",
            pipeline_name="p",
            model=model,
            image_width=1,
            image_height=1,
            predictions=[],
        ),
    )


def _pred(label: str) -> LayoutPrediction:
    return LayoutPrediction(bbox=[0, 0, 1, 1], score=1.0, label=label, page=1)


def test_upstage_mapper_accepts_canonical_labels() -> None:
    mapper, ctx = UpstageLabelMapper(), _context(LayoutDetectionModel.UPSTAGE_LAYOUT)
    assert mapper.to_canonical("paragraph", _pred("paragraph"), ctx) == CanonicalLabel.TEXT
    assert mapper.to_canonical("Page-header", _pred("Page-header"), ctx) == CanonicalLabel.PAGE_HEADER
    assert mapper.to_canonical("Section-header", _pred("Section-header"), ctx) == CanonicalLabel.SECTION_HEADER
    with pytest.raises(UnknownRawLayoutLabelError):
        mapper.to_canonical("banner", _pred("banner"), ctx)


def test_openai_compatible_vlm_layout_uses_qwen3vl_indices() -> None:
    ctx = _context(LayoutDetectionModel.OPENAI_COMPATIBLE_VLM_LAYOUT)
    index = str(int(QWEN3VL_STR_TO_LABEL["page_header"]))
    assert IndexedLayoutModelMapper().to_canonical(index, _pred(index), ctx) == CanonicalLabel.PAGE_HEADER
