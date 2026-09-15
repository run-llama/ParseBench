"""Regression coverage for Nemotron Omni multi-page PDF handling."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from parse_bench.inference.providers.parse.nemotron_omni import NemotronOmniProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="nemotron", provider_name="nemotron_omni", product_type=ProductType.PARSE, config={}
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="nemotron-multipage", source_file_path=str(source), product_type=ProductType.PARSE
    )


@pytest.mark.parametrize("prompt_mode", ["parse", "layout"])
def test_nemotron_omni_preserves_all_pdf_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_mode: str
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    provider = NemotronOmniProvider("nemotron", {"server_url": "https://example.invalid", "prompt_mode": prompt_mode})
    calls: list[bytes] = []
    monkeypatch.setattr(
        provider, "_pdf_to_images_with_size", lambda _path: [(b"first", 100, 200), (b"second", 300, 400)]
    )

    async def fake_run(image: bytes, width: int, height: int) -> dict[str, Any]:
        calls.append(image)
        number = len(calls)
        if prompt_mode == "layout":
            return {
                "prompt_mode": "layout",
                "layout_items": [{"bbox": [0, 0, 10, 10], "label": "Text", "text": f"page {number}"}],
                "image_width": width,
                "image_height": height,
            }
        return {"prompt_mode": "parse", "markdown": f"page {number}"}

    monkeypatch.setattr(provider, "_run_inference_async", fake_run)
    raw = provider.run_inference(_pipeline(), _request(source))
    normalized = provider.normalize(raw)

    assert calls == [b"first", b"second"]
    assert len(raw.raw_output["page_results"]) == 2
    assert normalized.output.markdown == "page 1\n\npage 2"
    if prompt_mode == "layout":
        assert [page.page_number for page in normalized.output.layout_pages] == [1, 2]
        assert [(page.width, page.height) for page in normalized.output.layout_pages] == [
            (100.0, 200.0),
            (300.0, 400.0),
        ]

    calls.clear()
    single = asyncio.run(provider._run_inference_pages_async([(b"only", 10, 20)]))
    assert calls == [b"only"]
    assert "page_results" not in single
