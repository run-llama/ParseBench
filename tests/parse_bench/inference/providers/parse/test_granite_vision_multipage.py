"""Regression coverage for Granite Vision multi-page PDF handling."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from parse_bench.inference.providers.parse.granite_vision import GraniteVisionProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="granite_vision", provider_name="granite_vision", product_type=ProductType.PARSE, config={}
    )


def _request(source_file: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="granite-vision-multipage", source_file_path=str(source_file), product_type=ProductType.PARSE
    )


@pytest.mark.parametrize("api_format", ["simple", "openai"])
def test_granite_vision_runs_pdf_pages_in_order_and_normalizes_each_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_format: str
) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\n")
    provider = GraniteVisionProvider(
        "granite_vision", {"server_url": "https://example.invalid", "api_format": api_format}
    )
    calls: list[bytes] = []

    monkeypatch.setattr(provider, "_pdf_to_images", lambda _path: [b"first", b"second"])

    async def fake_run_inference_async(page: bytes) -> dict[str, Any]:
        calls.append(page)
        number = len(calls)
        return {
            "markdown": f"<table><td colspan={number}>page {number}</td></table>",
            "_config": {"api_format": api_format, "page": number},
        }

    monkeypatch.setattr(provider, "_run_inference_async", fake_run_inference_async)

    raw = provider.run_inference(_pipeline(), _request(source_pdf))
    normalized = provider.normalize(raw)

    assert calls == [b"first", b"second"]
    assert raw.raw_output["markdown"] == "<table><td colspan=1>page 1</td></table>"
    assert raw.raw_output["page_results"] == [
        {"markdown": "<table><td colspan=1>page 1</td></table>", "_config": {"api_format": api_format, "page": 1}},
        {"markdown": "<table><td colspan=2>page 2</td></table>", "_config": {"api_format": api_format, "page": 2}},
    ]
    assert normalized.output.markdown == (
        '<table><td colspan="1">page 1</td></table>\n\n<table><td colspan="2">page 2</td></table>'
    )

    calls.clear()
    single_page = asyncio.run(provider._run_inference_pages_async([b"only"]))
    assert calls == [b"only"]
    assert single_page == {
        "markdown": "<table><td colspan=1>page 1</td></table>",
        "_config": {"api_format": api_format, "page": 1},
    }
