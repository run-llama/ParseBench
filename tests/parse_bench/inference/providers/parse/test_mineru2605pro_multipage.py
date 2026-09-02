"""Regression coverage for MinerU2605ProProvider multi-page PDF handling."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from parse_bench.inference.providers.parse.mineru2605pro import MinerU2605ProProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="mineru2605pro", provider_name="mineru2605pro", product_type=ProductType.PARSE, config={}
    )


def _request(source_file: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="mineru2605pro-multipage", source_file_path=str(source_file), product_type=ProductType.PARSE
    )


def _page_payload(number: int) -> dict[str, Any]:
    return {
        "markdown": f"<table><tr><td>page {number}</td></tr></table>",
        "blocks": [{"type": "table", "bbox": [0.1, 0.1, 0.9, 0.5], "angle": 0, "content": f"page {number}"}],
        "image_width": 100 * number,
        "image_height": 200 * number,
        "status": "success",
    }


def test_mineru2605pro_runs_pdf_pages_in_order_and_normalizes_each_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\n")
    provider = MinerU2605ProProvider("mineru2605pro", {"server_url": "https://example.invalid"})
    calls: list[bytes] = []

    monkeypatch.setattr(provider, "_pdf_to_images", lambda _path: [b"first", b"second"])

    async def fake_run_inference_async(page: bytes) -> dict[str, Any]:
        calls.append(page)
        return _page_payload(len(calls))

    monkeypatch.setattr(provider, "_run_inference_async", fake_run_inference_async)

    raw = provider.run_inference(_pipeline(), _request(source_pdf))
    normalized = provider.normalize(raw)

    assert calls == [b"first", b"second"]
    assert raw.raw_output["markdown"] == _page_payload(1)["markdown"]
    assert [p["image_width"] for p in raw.raw_output["page_results"]] == [100, 200]

    markdown = normalized.output.markdown
    assert "page 1" in markdown and "page 2" in markdown
    assert markdown.index("page 1") < markdown.index("page 2")
    assert [lp.page_number for lp in normalized.output.layout_pages] == [1, 2]
    assert [(lp.width, lp.height) for lp in normalized.output.layout_pages] == [(100.0, 200.0), (200.0, 400.0)]
    assert all(len(lp.items) == 1 for lp in normalized.output.layout_pages)

    calls.clear()
    single_page = asyncio.run(provider._run_inference_pages_async([b"only"]))
    assert calls == [b"only"]
    assert single_page == _page_payload(1)
    assert "page_results" not in single_page


def test_mineru2605pro_legacy_single_page_raw_output_still_normalizes(tmp_path: Path) -> None:
    provider = MinerU2605ProProvider("mineru2605pro", {"server_url": "https://example.invalid"})
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.7\n")

    now = datetime.now()
    raw = RawInferenceResult(
        request=_request(source_pdf),
        pipeline=_pipeline(),
        pipeline_name="mineru2605pro",
        product_type=ProductType.PARSE,
        raw_output=_page_payload(1),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    normalized = provider.normalize(raw)
    assert "page 1" in normalized.output.markdown
    assert [lp.page_number for lp in normalized.output.layout_pages] == [1]
