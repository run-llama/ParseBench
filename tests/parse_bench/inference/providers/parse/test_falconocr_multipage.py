"""Regression coverage for Falcon-OCR multi-page PDF handling."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from parse_bench.inference.providers.parse.falconocr import FalconOcrProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(pipeline_name="falconocr", provider_name="falconocr", product_type=ProductType.PARSE, config={})


def _request(source_file: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="falconocr-multipage", source_file_path=str(source_file), product_type=ProductType.PARSE
    )


def test_falconocr_runs_pdf_pages_in_order_and_normalizes_each_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\n")
    provider = FalconOcrProvider("falconocr", {"server_url": "https://example.invalid"})
    calls: list[bytes] = []

    monkeypatch.setattr(provider, "_pdf_to_images", lambda _path: [b"first", b"second"])

    async def fake_run_inference_async(page: bytes) -> dict[str, Any]:
        calls.append(page)
        number = len(calls)
        return {
            "markdown": f"<table><td colspan={number}>page {number}</td></table>",
            "regions": [{"category": "text", "bbox": [10, 20, 90, 180], "score": 0.9, "text": "page"}],
            "image_width": number * 100,
            "image_height": number * 200,
            "_config": {"page": number},
        }

    monkeypatch.setattr(provider, "_run_inference_async", fake_run_inference_async)

    raw = provider.run_inference(_pipeline(), _request(source_pdf))
    normalized = provider.normalize(raw)

    assert calls == [b"first", b"second"]
    assert raw.raw_output["markdown"] == "<table><td colspan=1>page 1</td></table>"
    assert [page["markdown"] for page in raw.raw_output["page_results"]] == [
        "<table><td colspan=1>page 1</td></table>",
        "<table><td colspan=2>page 2</td></table>",
    ]
    assert normalized.output.markdown == (
        '<table><td colspan="1">page 1</td></table>\n\n<table><td colspan="2">page 2</td></table>'
    )
    assert [page.page_number for page in normalized.output.layout_pages] == [1, 2]
    assert [(page.width, page.height) for page in normalized.output.layout_pages] == [(100.0, 200.0), (200.0, 400.0)]

    calls.clear()
    single_page = asyncio.run(provider._run_inference_pages_async([b"only"]))
    assert calls == [b"only"]
    assert single_page == {
        "markdown": "<table><td colspan=1>page 1</td></table>",
        "regions": [{"category": "text", "bbox": [10, 20, 90, 180], "score": 0.9, "text": "page"}],
        "image_width": 100,
        "image_height": 200,
        "_config": {"page": 1},
    }
