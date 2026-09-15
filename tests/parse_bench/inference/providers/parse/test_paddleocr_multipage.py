"""Regression coverage for PaddleOCR multi-page PDF handling."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from parse_bench.inference.providers.parse.paddleocr import PaddleOCRProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="paddleocr",
        provider_name="paddleocr",
        product_type=ProductType.PARSE,
        config={},
    )


def _request(source_file: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="paddleocr-multipage",
        source_file_path=str(source_file),
        product_type=ProductType.PARSE,
    )


def test_paddleocr_runs_pdf_pages_in_order_and_remaps_layout_page_numbers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\n")
    provider = PaddleOCRProvider(
        "paddleocr",
        {"server_url": "https://example.invalid", "api_format": "simple"},
    )
    calls: list[bytes] = []

    monkeypatch.setattr(provider, "_pdf_to_images", lambda _path: [b"first", b"second"])

    async def fake_run_inference_async(page: bytes) -> dict[str, Any]:
        calls.append(page)
        number = len(calls)
        return {
            "markdown": f"page {number}",
            "layout_pages": [
                {
                    "page_number": 1,
                    "width": 100 * number,
                    "height": 200 * number,
                    "items": [
                        {
                            "bbox": [0, 0, 50, 50],
                            "label": "text",
                            "text": f"page {number}",
                            "score": 0.9,
                        }
                    ],
                }
            ],
            "_config": {"page": number},
        }

    monkeypatch.setattr(provider, "_run_inference_async", fake_run_inference_async)

    raw = provider.run_inference(_pipeline(), _request(source_pdf))
    normalized = provider.normalize(raw)

    assert calls == [b"first", b"second"]
    assert raw.raw_output["markdown"] == "page 1"
    assert [page["markdown"] for page in raw.raw_output["page_results"]] == ["page 1", "page 2"]
    assert normalized.output.markdown == "page 1\n\npage 2"
    assert [page.page_number for page in normalized.output.layout_pages] == [1, 2]
    assert [(page.width, page.height) for page in normalized.output.layout_pages] == [(100.0, 200.0), (200.0, 400.0)]
    # Pixel bboxes are normalized by each page's own image dimensions.
    assert [[item.bbox.label for item in page.items] for page in normalized.output.layout_pages] == [["Text"], ["Text"]]
    first_bbox = normalized.output.layout_pages[0].items[0].bbox
    second_bbox = normalized.output.layout_pages[1].items[0].bbox
    assert (first_bbox.x, first_bbox.y, first_bbox.w, first_bbox.h) == (0.0, 0.0, 0.5, 0.25)
    assert (second_bbox.x, second_bbox.y, second_bbox.w, second_bbox.h) == (0.0, 0.0, 0.25, 0.125)

    calls.clear()
    single_page = asyncio.run(provider._run_inference_pages_async([b"only"]))
    assert calls == [b"only"]
    assert single_page == {
        "markdown": "page 1",
        "layout_pages": [
            {
                "page_number": 1,
                "width": 100,
                "height": 200,
                "items": [
                    {
                        "bbox": [0, 0, 50, 50],
                        "label": "text",
                        "text": "page 1",
                        "score": 0.9,
                    }
                ],
            }
        ],
        "_config": {"page": 1},
    }
