"""Focused tests for the shared WeVisDoc provider."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.wevisdoc import WeVisDocProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        body: dict[str, Any] | None = None,
        text: str = "",
        json_error: Exception | None = None,
    ) -> None:
        self.status = status
        self._body = (
            body
            if body is not None
            else {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "converted markdown"},
                    }
                ]
            }
        )
        self._text = text
        self._json_error = json_error

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        if self._json_error is not None:
            raise self._json_error
        return self._body

    async def text(self) -> str:
        return self._text


class _FakeSession:
    def __init__(
        self,
        response: _FakeResponse | None = None,
        post_error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response = response or _FakeResponse()
        self._post_error = post_error

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self._post_error is not None:
            raise self._post_error
        return self._response


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="wevisdoc",
        provider_name="wevisdoc",
        product_type=ProductType.PARSE,
        config={},
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="wevisdoc-test",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def _raw_result(raw_output: dict[str, Any]) -> RawInferenceResult:
    now = datetime.now()
    return RawInferenceResult(
        request=InferenceRequest(
            example_id="wevisdoc-normalize",
            source_file_path="/tmp/wevisdoc.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline=_pipeline(),
        pipeline_name="wevisdoc",
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


@pytest.mark.parametrize(
    ("pipeline_name", "server_url_env", "served_model_name"),
    [
        ("wevisdoc_2b_vllm", "WEVISDOC_2B_SERVER_URL", "wevisdoc-2b"),
        ("wevisdoc_4b_vllm", "WEVISDOC_4B_SERVER_URL", "wevisdoc-4b"),
    ],
)
def test_pipeline_endpoint_env_selects_request_url_and_model(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_name: str,
    server_url_env: str,
    served_model_name: str,
) -> None:
    endpoint = f"https://{served_model_name}.example.invalid/"
    monkeypatch.setenv(server_url_env, endpoint)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    spec = get_pipeline(pipeline_name)
    provider = WeVisDocProvider(spec.provider_name, spec.config)
    session = _FakeSession()

    assert asyncio.run(provider._call_openai_api(session, "encoded-image")) == "converted markdown"

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == f"{endpoint}v1/chat/completions"
    assert call["headers"] == {"Content-Type": "application/json"}
    payload = call["json"]
    assert payload["model"] == served_model_name
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 8192
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    user_content = payload["messages"][1]["content"]
    assert user_content[0]["image_url"]["url"] == "data:image/png;base64,encoded-image"
    assert user_content[1] == {"type": "text", "text": "Convert this document image to Markdown."}


def test_direct_provider_uses_fallback_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEVISDOC_SERVER_URL", "https://fallback.example.invalid")

    provider = WeVisDocProvider("wevisdoc", {})

    assert provider._server_url == "https://fallback.example.invalid"
    assert provider._served_model_name == "wevisdoc-2b"


def test_direct_provider_requires_endpoint_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEVISDOC_SERVER_URL", raising=False)

    with pytest.raises(ProviderConfigError, match="WEVISDOC_SERVER_URL"):
        WeVisDocProvider("wevisdoc", {})


def test_finish_reason_stop_accepts_complete_content() -> None:
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    response = _FakeResponse(
        body={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "complete markdown"},
                }
            ]
        }
    )

    result = asyncio.run(provider._call_openai_api(_FakeSession(response), "encoded-image"))

    assert result == "complete markdown"


def test_finish_reason_length_rejects_truncated_content() -> None:
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    response = _FakeResponse(
        body={
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "truncated markdown"},
                }
            ]
        }
    )

    with pytest.raises(ProviderPermanentError, match="finish_reason='length'"):
        asyncio.run(provider._call_openai_api(_FakeSession(response), "encoded-image"))


def test_other_finish_reason_rejects_incomplete_content() -> None:
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    response = _FakeResponse(
        body={
            "choices": [
                {
                    "finish_reason": "content_filter",
                    "message": {"content": "filtered markdown"},
                }
            ]
        }
    )

    with pytest.raises(ProviderPermanentError, match="finish_reason='content_filter'"):
        asyncio.run(provider._call_openai_api(_FakeSession(response), "encoded-image"))


@pytest.mark.parametrize(
    "choice",
    [
        {"message": {"content": "markdown"}},
        {"finish_reason": None, "message": {"content": "markdown"}},
    ],
)
def test_missing_finish_reason_rejects_response(choice: dict[str, Any]) -> None:
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    response = _FakeResponse(body={"choices": [choice]})

    with pytest.raises(ProviderPermanentError, match="missing finish_reason"):
        asyncio.run(provider._call_openai_api(_FakeSession(response), "encoded-image"))


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (429, ProviderRateLimitError),
        (500, ProviderTransientError),
        (503, ProviderTransientError),
        (599, ProviderTransientError),
        (400, ProviderPermanentError),
        (499, ProviderPermanentError),
    ],
)
def test_http_status_classification(status: int, error_type: type[Exception]) -> None:
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    session = _FakeSession(_FakeResponse(status=status, text="server response"))

    with pytest.raises(error_type, match=f"HTTP {status}"):
        asyncio.run(provider._call_openai_api(session, "encoded-image"))


def test_http_408_is_transient() -> None:
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    session = _FakeSession(_FakeResponse(status=408, text="request timeout"))

    with pytest.raises(ProviderTransientError, match="HTTP 408"):
        asyncio.run(provider._call_openai_api(session, "encoded-image"))


def test_invalid_json_is_transient() -> None:
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    session = _FakeSession(_FakeResponse(json_error=ValueError("not json")))

    with pytest.raises(ProviderTransientError, match="Invalid JSON response"):
        asyncio.run(provider._call_openai_api(session, "encoded-image"))


def test_aiohttp_transport_error_is_transient() -> None:
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    session = _FakeSession(post_error=aiohttp.ClientConnectionError("connection reset"))

    with pytest.raises(ProviderTransientError, match="Transport error"):
        asyncio.run(provider._call_openai_api(session, "encoded-image"))


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError(),
        aiohttp.ClientConnectionError("connection reset"),
        json.JSONDecodeError("invalid", "not json", 0),
    ],
)
def test_run_inference_raises_retryable_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})

    async def fail(_pages: list[bytes]) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr(provider, "_run_inference_pages_async", fail)

    with pytest.raises(ProviderTransientError):
        provider.run_inference(_pipeline(), _request(source))


def test_run_inference_preserves_permanent_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    failure = ProviderPermanentError("invalid request")

    async def fail(_pages: list[bytes]) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr(provider, "_run_inference_pages_async", fail)

    with pytest.raises(ProviderPermanentError) as exc_info:
        provider.run_inference(_pipeline(), _request(source))
    assert exc_info.value is failure


def test_run_inference_preserves_rate_limit_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    failure = ProviderRateLimitError("rate limited")

    async def fail(_pages: list[bytes]) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr(provider, "_run_inference_pages_async", fail)

    with pytest.raises(ProviderRateLimitError) as exc_info:
        provider.run_inference(_pipeline(), _request(source))
    assert exc_info.value is failure


def test_multipage_pdf_runs_and_normalizes_pages_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    calls: list[bytes] = []
    monkeypatch.setattr(provider, "_pdf_to_images", lambda _path: [b"first", b"second"])

    async def fake_run(page: bytes) -> dict[str, Any]:
        calls.append(page)
        return {"markdown": f"page {len(calls)}"}

    monkeypatch.setattr(provider, "_run_inference_async", fake_run)

    raw = provider.run_inference(_pipeline(), _request(source))
    normalized = provider.normalize(raw)

    assert calls == [b"first", b"second"]
    assert [page["markdown"] for page in raw.raw_output["page_results"]] == ["page 1", "page 2"]
    assert normalized.output.markdown == "page 1\n\npage 2"


def test_normalize_converts_pipe_tables_and_repairs_html() -> None:
    provider = WeVisDocProvider("wevisdoc", {"server_url": "https://example.invalid"})
    raw = _raw_result(
        {
            "page_results": [
                {"markdown": "| Column |\n| --- |\n| alpha |"},
                {"markdown": "<table><tr><td colspan=2>omega"},
            ]
        }
    )

    markdown = provider.normalize(raw).output.markdown

    assert "| Column |" not in markdown
    assert "<table>" in markdown
    assert "alpha" in markdown
    assert '<td colspan="2">omega</td></tr></table>' in markdown
    assert markdown.index("alpha") < markdown.index("omega")
