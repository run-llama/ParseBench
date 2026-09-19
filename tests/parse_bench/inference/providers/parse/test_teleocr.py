"""Focused coverage for the TeleOCR public provider integration."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
from parse_bench.evaluation.layout_adapters.adapters import TeleOCRLayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.teleocr import TeleOCRProvider
from parse_bench.layout_projection import project_to_canonical_predictions
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType


class _Response:
    def __init__(
        self,
        status: int,
        *,
        payload: Any = None,
        json_error: Exception | None = None,
        text: str = "request failed",
    ) -> None:
        self.status = status
        self._payload = payload
        self._json_error = json_error
        self._text = text

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def text(self) -> str:
        return self._text

    async def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class _Session:
    def __init__(self, response: _Response | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    def post(self, *_args: object, **_kwargs: object) -> _Response:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _provider() -> TeleOCRProvider:
    return TeleOCRProvider("teleocr", {"server_url": "https://teleocr.example.invalid/predict"})


def _request(source_file: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="teleocr-errors",
        source_file_path=str(source_file),
        product_type=ProductType.PARSE,
    )


def test_teleocr_requires_an_explicit_server_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEOCR_SERVER_URL", raising=False)
    with pytest.raises(ProviderConfigError, match="TELEOCR_SERVER_URL"):
        TeleOCRProvider("teleocr", {})

    monkeypatch.setenv("TELEOCR_SERVER_URL", "https://teleocr.example.invalid/predict")
    provider = TeleOCRProvider("teleocr", {})
    assert provider._config_snapshot()["server_url"] == "https://teleocr.example.invalid/predict"


@pytest.mark.parametrize(
    "status,error_type",
    [
        (400, ProviderPermanentError),
        (404, ProviderPermanentError),
        (408, ProviderTransientError),
        (429, ProviderRateLimitError),
        (500, ProviderTransientError),
        (599, ProviderTransientError),
    ],
)
def test_teleocr_http_statuses_use_provider_error_taxonomy(status: int, error_type: type[Exception]) -> None:
    with pytest.raises(error_type, match=f"HTTP {status}"):
        asyncio.run(_provider()._call_api(_Session(_Response(status)), "encoded"))  # type: ignore[arg-type]


def test_teleocr_invalid_json_is_transient_at_http_boundary() -> None:
    invalid_json = json.JSONDecodeError("invalid", "not-json", 0)
    response = _Response(200, json_error=invalid_json)

    with pytest.raises(ProviderTransientError, match="invalid JSON"):
        asyncio.run(_provider()._call_api(_Session(response), "encoded"))  # type: ignore[arg-type]


def test_teleocr_rejects_response_without_markdown_or_blocks() -> None:
    response = _Response(200, payload={"status": "success", "markdown": "", "blocks": []})

    with pytest.raises(ProviderPermanentError, match="neither markdown nor valid blocks"):
        asyncio.run(_provider()._call_api(_Session(response), "encoded"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("timed out"),
        aiohttp.ClientConnectionError("connection closed"),
        aiohttp.ClientConnectorError(
            SimpleNamespace(host="teleocr.example.invalid", port=443, ssl=True),
            OSError("DNS lookup failed"),
        ),
    ],
)
def test_teleocr_transport_failures_are_transient_at_http_boundary(failure: Exception) -> None:
    with pytest.raises(ProviderTransientError):
        asyncio.run(_provider()._call_api(_Session(error=failure), "encoded"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "failure,error_type",
    [
        (ProviderPermanentError("bad request"), ProviderPermanentError),
        (ProviderRateLimitError("rate limited"), ProviderRateLimitError),
        (ProviderTransientError("retry"), ProviderTransientError),
    ],
)
def test_teleocr_run_inference_propagates_provider_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    error_type: type[Exception],
) -> None:
    source = tmp_path / "page.png"
    source.write_bytes(b"image")
    provider = _provider()

    async def fail(_pages: list[bytes]) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr(provider, "_run_inference_pages_async", fail)
    with pytest.raises(error_type):
        provider.run_inference(get_pipeline("teleocr_vllm"), _request(source))


@pytest.mark.parametrize(
    "failure,message",
    [
        (TimeoutError("timed out"), "timed out"),
        (aiohttp.ClientConnectionError("connection closed"), "request failed"),
        (json.JSONDecodeError("invalid", "not-json", 0), "invalid JSON"),
    ],
)
def test_teleocr_run_inference_converts_retryable_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    message: str,
) -> None:
    source = tmp_path / "page.png"
    source.write_bytes(b"image")
    provider = _provider()

    async def fail(_pages: list[bytes]) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr(provider, "_run_inference_pages_async", fail)
    with pytest.raises(ProviderTransientError, match=message):
        provider.run_inference(get_pipeline("teleocr_vllm"), _request(source))


def test_teleocr_block_only_response_reaches_layout_adapter() -> None:
    pipeline = get_pipeline("teleocr_vllm")
    provider = _provider()
    payload = {
        "status": "success",
        "markdown": "",
        "blocks": [{"type": "image", "bbox": [0.1, 0.2, 0.9, 0.8], "content": ""}],
        "image_width": 1000,
        "image_height": 2000,
    }
    accepted = asyncio.run(provider._call_api(_Session(_Response(200, payload=payload)), "encoded"))  # type: ignore[arg-type]
    now = datetime.now()
    raw = RawInferenceResult(
        request=InferenceRequest(
            example_id="teleocr-block-only",
            source_file_path="document.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=accepted,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )

    result = provider.normalize(raw)
    assert result.output.markdown == ""
    assert len(result.output.layout_pages) == 1
    adapter = create_layout_adapter_for_result(result)
    assert isinstance(adapter, TeleOCRLayoutAdapter)
    layout = adapter.to_layout_output(result)
    assert [prediction.label for prediction in layout.predictions] == ["Picture"]
    assert layout.predictions[0].bbox == [100.0, 200.0, 900.0, 800.0]


def test_teleocr_preserves_server_markdown_and_normalizes_block_layout() -> None:
    pipeline = get_pipeline("teleocr_vllm")
    provider = _provider()
    server_markdown = (
        "```python\n"
        "print('<table><tr><td raw=1>literal</td></tr></table>')\n"
        "```\n\n"
        "![chart](chart.png)\n\n"
        "<table><tr><td colspan=2>Header</td></tr><tr><td>Value</td></tr></table>"
    )
    now = datetime.now()
    raw = RawInferenceResult(
        request=InferenceRequest(
            example_id="teleocr-layout",
            source_file_path="document.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output={
            "markdown": server_markdown,
            "blocks": [
                {"type": "title", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "Report"},
                {
                    "type": "table",
                    "bbox": [0.1, 0.3, 0.9, 0.6],
                    "content": "<table><tr><td colspan=2>Header</td></tr><tr><td>Value",
                },
                {"type": "page_number", "bbox": [0.45, 0.92, 0.55, 0.97], "content": "1"},
            ],
            "image_width": 1000,
            "image_height": 2000,
        },
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )

    result = provider.normalize(raw)
    assert "```python\nprint('<table><tr><td raw=1>literal</td></tr></table>')\n```" in result.output.markdown
    assert "![chart](chart.png)" in result.output.markdown
    assert '<th colspan="2">Header</th>' in result.output.markdown
    assert result.output.markdown.count("Header") == 1
    assert "# Report" not in result.output.markdown
    assert [item.layout_segments[0].label for item in result.output.layout_pages[0].items] == [
        "Title",
        "Table",
        "Page-footer",
    ]
    assert '<th colspan="2">Header</th>' in result.output.layout_pages[0].items[1].value

    adapter = create_layout_adapter_for_result(result)
    assert isinstance(adapter, TeleOCRLayoutAdapter)
    layout = adapter.to_layout_output(result)
    assert layout.model is LayoutDetectionModel.TELEOCR_LAYOUT
    assert [prediction.label for prediction in layout.predictions] == ["Title", "Table", "Page-footer"]
    assert layout.predictions[0].bbox == [100.0, 100.0, 900.0, 200.0]


def test_teleocr_mixed_page_sizes_use_common_frame_and_project_code_labels() -> None:
    pipeline = get_pipeline("teleocr_vllm")
    provider = _provider()
    page_one_markdown = "```python\nprint('page one')\n```"
    page_two_markdown = "Algorithm result"
    page_results = [
        {
            "markdown": page_one_markdown,
            "blocks": [{"type": "code", "bbox": [0.1, 0.2, 0.4, 0.5], "content": "print('page one')"}],
            "image_width": 1000,
            "image_height": 2000,
        },
        {
            "markdown": page_two_markdown,
            "blocks": [{"type": "algorithm", "bbox": [0.25, 0.1, 0.75, 0.9], "content": "result"}],
            "image_width": 2000,
            "image_height": 1000,
        },
    ]
    now = datetime.now()
    raw = RawInferenceResult(
        request=InferenceRequest(
            example_id="teleocr-mixed-pages",
            source_file_path="document.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output={**page_results[0], "page_results": page_results},
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )

    result = provider.normalize(raw)
    assert result.output.markdown == f"{page_one_markdown}\n\n{page_two_markdown}"
    assert [(page.width, page.height) for page in result.output.layout_pages] == [
        (1000.0, 2000.0),
        (2000.0, 1000.0),
    ]
    assert [page.items[0].layout_segments[0].label for page in result.output.layout_pages] == ["Code", "Code"]

    adapter = create_layout_adapter_for_result(result)
    assert isinstance(adapter, TeleOCRLayoutAdapter)
    layout = adapter.to_layout_output(result)
    assert (layout.image_width, layout.image_height) == (1000, 1000)
    assert [prediction.bbox for prediction in layout.predictions] == [
        [100.0, 200.0, 400.0, 500.0],
        [250.0, 100.0, 750.0, 900.0],
    ]
    assert [prediction.label for prediction in layout.predictions] == ["Code", "Code"]

    canonical = project_to_canonical_predictions(layout)
    assert [prediction.canonical_class for prediction in canonical] == [CanonicalLabel.CODE, CanonicalLabel.CODE]

    evaluator = LayoutDetectionEvaluator(evaluation_view="canonical", default_ontology="canonical")
    projected = evaluator._extract_predictions(result, layout, target_ontology="canonical")
    assert projected[0]["bbox"] == pytest.approx([0.1, 0.2, 0.4, 0.5])
    assert projected[1]["bbox"] == pytest.approx([0.25, 0.1, 0.75, 0.9])
    assert [prediction["class_name"] for prediction in projected] == ["Code", "Code"]
    assert [prediction["page"] for prediction in projected] == [1, 2]

    page_two_layout = adapter.to_layout_output(result, page_filter=2)
    assert [prediction.page for prediction in page_two_layout.predictions] == [2]
    page_two_projected = evaluator._extract_predictions(
        result,
        page_two_layout,
        target_ontology="canonical",
        page_filter=2,
    )
    assert len(page_two_projected) == 1
    assert page_two_projected[0]["bbox"] == pytest.approx([0.25, 0.1, 0.75, 0.9])
    assert page_two_projected[0]["class_name"] == "Code"


def test_teleocr_pipeline_has_no_endpoint_default() -> None:
    pipeline = get_pipeline("teleocr_vllm")
    assert pipeline.provider_name == "teleocr"
    assert pipeline.product_type == ProductType.PARSE
    assert pipeline.config["server_url"] == ""
