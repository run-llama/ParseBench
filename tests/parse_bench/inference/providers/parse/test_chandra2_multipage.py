"""Regression coverage for Chandra OCR 2 multi-page PDF handling."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.chandra2 import Chandra2Provider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(pipeline_name="chandra2", provider_name="chandra2", product_type=ProductType.PARSE, config={})


def _request(source_file: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="chandra2-multipage", source_file_path=str(source_file), product_type=ProductType.PARSE
    )


def test_chandra2_runs_pdf_pages_in_order_and_normalizes_each_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\n")
    provider = Chandra2Provider("chandra2", {"server_url": "https://example.invalid"})
    calls: list[bytes] = []

    monkeypatch.setattr(provider, "_pdf_to_images", lambda _path: [b"first", b"second"])

    async def fake_run_inference_async(page: bytes) -> dict[str, Any]:
        calls.append(page)
        number = len(calls)
        return {
            "markdown": (
                f'<div data-bbox="0 0 500 {number * 100}" data-label="Text">'
                f"<table><td colspan={number}>page {number}</td></table></div>"
            ),
            "_source": "vllm",
            "_config": {"page": number},
        }

    monkeypatch.setattr(provider, "_run_inference_async", fake_run_inference_async)

    raw = provider.run_inference(_pipeline(), _request(source_pdf))
    normalized = provider.normalize(raw)

    assert calls == [b"first", b"second"]
    assert len(raw.raw_output["page_results"]) == 2
    assert raw.raw_output["markdown"] == raw.raw_output["page_results"][0]["markdown"]
    assert normalized.output.markdown == (
        '<table><td colspan="1">page 1</td></table>\n\n<table><td colspan="2">page 2</td></table>'
    )
    assert [page.page_number for page in normalized.output.layout_pages] == [1, 2]
    assert [len(page.items) for page in normalized.output.layout_pages] == [1, 1]

    calls.clear()
    single_page = asyncio.run(provider._run_inference_pages_async([b"only"]))
    assert calls == [b"only"]
    assert "page_results" not in single_page
    assert single_page["_config"] == {"page": 1}


def test_chandra2_prompt_appendix_extends_task_prompt() -> None:
    provider = Chandra2Provider(
        "chandra2", {"server_url": "https://example.invalid", "prompt_appendix": "Keep footnotes."}
    )
    assert provider._primary_prompt.endswith("\n\nKeep footnotes.")

    with pytest.raises(ProviderConfigError):
        Chandra2Provider(
            "chandra2",
            {"server_url": "https://example.invalid", "api_format": "simple", "prompt_appendix": "x"},
        )
