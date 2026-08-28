from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

PROVIDERS = [
    ("google", "GoogleProvider"),
    ("openai", "OpenAIProvider"),
    ("anthropic", "AnthropicProvider"),
    ("amazon_nova", "AmazonNovaProvider"),
    ("dots_ocr", "DotsOcrParseProvider"),
]

TEXT_PAGE_PROVIDERS = [
    ("pymupdf4llm", "PyMuPDF4LLMProvider"),
    ("tesseract", "TesseractProvider"),
]


def _raw_result(module_name: str, pages: list[dict[str, Any]]) -> RawInferenceResult:
    now = datetime.now()
    pipeline = PipelineSpec(
        pipeline_name=f"{module_name}_test",
        provider_name=module_name,
        product_type=ProductType.PARSE,
    )
    request = InferenceRequest(
        example_id="document",
        source_file_path=str(Path("document.pdf")),
        product_type=ProductType.PARSE,
    )
    raw_output: dict[str, Any] = {"pages": pages, "mode": "parse_with_layout", "bbox_scale": 1000}
    if module_name == "dots_ocr":
        raw_output["prompt_mode"] = "prompt_layout_all_en_v1_5"
    return RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _page(module_name: str, page_index: int, text: str) -> dict[str, Any]:
    if module_name == "dots_ocr":
        layout_items = [{"category": "Text", "bbox": [0, 0, 50, 50], "text": text}] if text else []
        return {
            "page_index": page_index,
            "width": 100,
            "height": 100,
            "markdown": text,
            "layout_items": layout_items,
        }
    items = [{"label": "Text", "bbox": [0, 0, 500, 500], "text": text}] if text else []
    return {"page_index": page_index, "width": 100, "height": 100, "items": items}


@pytest.mark.parametrize(("module_name", "class_name"), PROVIDERS)
def test_normalization_sorts_once_and_preserves_explicit_blank_layout_page(
    module_name: str,
    class_name: str,
) -> None:
    provider_class = getattr(
        importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}"), class_name
    )
    provider = object.__new__(provider_class)
    raw = _raw_result(
        module_name,
        [_page(module_name, 2, "third"), _page(module_name, 1, ""), _page(module_name, 0, "first")],
    )

    result = provider.normalize(raw)

    assert [(page.page_index, page.markdown) for page in result.output.pages] == [
        (0, "first"),
        (1, ""),
        (2, "third"),
    ]
    assert result.output.markdown == "first\n\n\n\nthird"
    assert [page.page_number for page in result.output.layout_pages] == [1, 2, 3]
    assert result.output.layout_pages[1].items == []


@pytest.mark.parametrize(("module_name", "class_name"), PROVIDERS)
def test_normalization_rejects_duplicate_raw_page_indices(module_name: str, class_name: str) -> None:
    provider_class = getattr(
        importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}"), class_name
    )
    provider = object.__new__(provider_class)
    raw = _raw_result(module_name, [_page(module_name, 0, "first"), _page(module_name, 0, "duplicate")])

    with pytest.raises(ProviderPermanentError, match="unique, contiguous, and zero-based"):
        provider.normalize(raw)


@pytest.mark.parametrize(("module_name", "class_name"), TEXT_PAGE_PROVIDERS)
def test_text_normalization_sorts_raw_pages_before_all_views(module_name: str, class_name: str) -> None:
    provider_class = getattr(
        importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}"), class_name
    )
    provider = object.__new__(provider_class)
    raw = _raw_result(
        module_name,
        [
            {"page_index": 1, "text": "second", "width": 100, "height": 100, "blocks": []},
            {"page_index": 0, "text": "first", "width": 100, "height": 100, "blocks": []},
        ],
    )

    result = provider.normalize(raw)

    assert [(page.page_index, page.markdown) for page in result.output.pages] == [
        (0, "first"),
        (1, "second"),
    ]
    assert result.output.markdown == "first\n\nsecond"


@pytest.mark.parametrize(("module_name", "class_name"), TEXT_PAGE_PROVIDERS)
def test_text_normalization_rejects_duplicate_raw_page_indices(module_name: str, class_name: str) -> None:
    provider_class = getattr(
        importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}"), class_name
    )
    provider = object.__new__(provider_class)
    raw = _raw_result(
        module_name,
        [
            {"page_index": 0, "text": "first"},
            {"page_index": 0, "text": "duplicate"},
        ],
    )

    with pytest.raises(ProviderPermanentError, match="unique, contiguous, and zero-based"):
        provider.normalize(raw)
