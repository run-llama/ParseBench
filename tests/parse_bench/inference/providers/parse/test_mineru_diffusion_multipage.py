"""Regression coverage for MinerU-Diffusion multi-page PDF handling."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from parse_bench.inference.providers.parse.mineru_diffusion import MinerUDiffusionProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="mineru_diffusion",
        provider_name="mineru_diffusion",
        product_type=ProductType.PARSE,
        config={},
    )


def _request(source_file: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="mineru-diffusion-multipage",
        source_file_path=str(source_file),
        product_type=ProductType.PARSE,
    )


def test_mineru_diffusion_runs_pdf_pages_in_order_and_normalizes_each_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\n")
    provider = MinerUDiffusionProvider("mineru_diffusion", {"server_url": "https://example.invalid"})
    calls: list[bytes] = []

    monkeypatch.setattr(provider, "_pdf_to_images", lambda _path: [b"first", b"second"])

    async def fake_run_inference_async(page: bytes) -> dict[str, Any]:
        calls.append(page)
        number = len(calls)
        return {
            "markdown": f"page {number}",
            "blocks": [
                {
                    "type": "text",
                    "bbox": [0, 0, 1, 1],
                    "content": f"page {number}",
                    "rendered": f"page {number}",
                }
            ],
            "image_width": 100 * number,
            "image_height": 200 * number,
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
    assert [(page.width, page.height) for page in normalized.output.layout_pages] == [
        (100.0, 200.0),
        (200.0, 400.0),
    ]

    calls.clear()
    single_page = asyncio.run(provider._run_inference_pages_async([b"only"]))
    assert calls == [b"only"]
    assert single_page == {
        "markdown": "page 1",
        "blocks": [
            {
                "type": "text",
                "bbox": [0, 0, 1, 1],
                "content": "page 1",
                "rendered": "page 1",
            }
        ],
        "image_width": 100,
        "image_height": 200,
        "_config": {"page": 1},
    }
