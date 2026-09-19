"""Focused coverage for the OvisOCR2 parse provider."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.ovisocr2 import OCR_PROMPT, OvisOcr2Provider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="ovisocr2_vllm",
        provider_name="ovisocr2",
        product_type=ProductType.PARSE,
        config={},
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="ovisocr2-test",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        json_result: Any = None,
        json_error: Exception | None = None,
        response_text: str = "",
    ) -> None:
        self.status = status
        self._json_result = json_result
        self._json_error = json_error
        self._response_text = response_text

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        if self._json_error is not None:
            raise self._json_error
        if self._json_result is not None:
            return self._json_result
        return {"choices": [{"message": {"content": "parsed markdown"}}]}

    async def text(self) -> str:
        return self._response_text


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


def test_ovisocr2_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OVISOCR2_SERVER_URL", raising=False)

    with pytest.raises(ProviderConfigError, match="OVISOCR2_SERVER_URL"):
        OvisOcr2Provider("ovisocr2")


def test_ovisocr2_uses_endpoint_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVISOCR2_SERVER_URL", "https://example.invalid/from-env")

    provider = OvisOcr2Provider("ovisocr2")

    assert provider._server_url == "https://example.invalid/from-env"


def test_ovisocr2_sends_openai_compatible_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVISOCR2_TEST_API_KEY", "test-key")
    provider = OvisOcr2Provider(
        "ovisocr2",
        {
            "server_url": "https://example.invalid/root/",
            "served_model_name": "custom-ovisocr2",
            "api_key_env": "OVISOCR2_TEST_API_KEY",
            "temperature": 0.25,
            "max_tokens": 321,
            "timeout": 45,
        },
    )
    session = _FakeSession()

    assert asyncio.run(provider._call_openai_api(session, "encoded-image")) == "parsed markdown"

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://example.invalid/root/v1/chat/completions"
    assert call["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-key",
    }
    assert call["timeout"].total == 45
    assert call["json"] == {
        "model": "custom-ovisocr2",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,encoded-image"},
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        "temperature": 0.25,
        "max_tokens": 321,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {
            "images_kwargs": {
                "min_pixels": 200704,
                "max_pixels": 8294400,
            }
        },
    }


def test_ovisocr2_classifies_rate_limit_response() -> None:
    provider = OvisOcr2Provider("ovisocr2", {"server_url": "https://example.invalid"})
    session = _FakeSession(_FakeResponse(status=429, response_text="slow down"))

    with pytest.raises(ProviderRateLimitError, match="HTTP 429: slow down"):
        asyncio.run(provider._call_openai_api(session, "encoded-image"))


@pytest.mark.parametrize("status", [408, 500, 501, 503, 599])
def test_ovisocr2_classifies_retryable_http_errors_as_transient(status: int) -> None:
    provider = OvisOcr2Provider("ovisocr2", {"server_url": "https://example.invalid"})
    session = _FakeSession(_FakeResponse(status=status, response_text="server failed"))

    with pytest.raises(ProviderTransientError, match=rf"HTTP {status}: server failed"):
        asyncio.run(provider._call_openai_api(session, "encoded-image"))


@pytest.mark.parametrize("status", [400, 401, 422, 499])
def test_ovisocr2_classifies_other_client_errors_as_permanent(status: int) -> None:
    provider = OvisOcr2Provider("ovisocr2", {"server_url": "https://example.invalid"})
    session = _FakeSession(_FakeResponse(status=status, response_text="bad request"))

    with pytest.raises(ProviderPermanentError, match=rf"HTTP {status}: bad request"):
        asyncio.run(provider._call_openai_api(session, "encoded-image"))


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        aiohttp.ClientConnectionError("connection dropped"),
    ],
    ids=["timeout", "aiohttp-transport"],
)
def test_ovisocr2_classifies_request_failures_as_transient(error: Exception) -> None:
    provider = OvisOcr2Provider("ovisocr2", {"server_url": "https://example.invalid"})
    session = _FakeSession(post_error=error)

    with pytest.raises(ProviderTransientError):
        asyncio.run(provider._call_openai_api(session, "encoded-image"))


def test_ovisocr2_classifies_invalid_json_as_transient() -> None:
    provider = OvisOcr2Provider("ovisocr2", {"server_url": "https://example.invalid"})
    invalid_json = json.JSONDecodeError("invalid", "not-json", 0)
    session = _FakeSession(_FakeResponse(json_error=invalid_json))

    with pytest.raises(ProviderTransientError, match="Invalid JSON response"):
        asyncio.run(provider._call_openai_api(session, "encoded-image"))


@pytest.mark.parametrize(
    "error_type",
    [ProviderTransientError, ProviderRateLimitError, ProviderPermanentError],
)
def test_ovisocr2_run_inference_does_not_turn_provider_errors_into_blank_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    provider = OvisOcr2Provider("ovisocr2", {"server_url": "https://example.invalid"})

    async def fail(_pages: list[bytes]) -> dict[str, Any]:
        raise error_type("request failed")

    monkeypatch.setattr(provider, "_run_inference_pages_async", fail)

    with pytest.raises(error_type, match="request failed"):
        provider.run_inference(_pipeline(), _request(source))


def test_ovisocr2_preserves_multipage_output_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    provider = OvisOcr2Provider("ovisocr2", {"server_url": "https://example.invalid"})
    calls: list[bytes] = []
    monkeypatch.setattr(provider, "_pdf_to_images", lambda _path: [b"first", b"second"])

    async def fake_run(page: bytes) -> dict[str, Any]:
        calls.append(page)
        number = len(calls)
        return {
            "markdown": f"page {number}",
            "_config": {"filter_imgtags": True},
        }

    monkeypatch.setattr(provider, "_run_inference_async", fake_run)

    raw = provider.run_inference(_pipeline(), _request(source))
    normalized = provider.normalize(raw)

    assert calls == [b"first", b"second"]
    assert raw.raw_output["markdown"] == "page 1"
    assert [page["markdown"] for page in raw.raw_output["page_results"]] == ["page 1", "page 2"]
    assert normalized.output.markdown == "page 1\n\npage 2"

    calls.clear()
    single_page = asyncio.run(provider._run_inference_pages_async([b"only"]))
    assert calls == [b"only"]
    assert "page_results" not in single_page
    assert single_page["markdown"] == "page 1"


def test_ovisocr2_normalization_applies_model_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    provider = OvisOcr2Provider("ovisocr2", {"server_url": "https://example.invalid"})

    async def fake_run(_page: bytes) -> dict[str, Any]:
        return {
            "markdown": (
                "Before\n\n"
                '<img src="images/bbox_1_2_3_4.jpg" />\n\n'
                "| Name | Value |\n"
                "| --- | --- |\n"
                "| Alpha | 1 |\n\n"
                "<span data-count=2>After</span>"
            ),
            "_config": {"filter_imgtags": True},
        }

    monkeypatch.setattr(provider, "_run_inference_async", fake_run)

    normalized = provider.normalize(provider.run_inference(_pipeline(), _request(source)))
    markdown = normalized.output.markdown

    assert "bbox_1_2_3_4" not in markdown
    assert "<table>" in markdown
    assert "<td>Alpha</td>" in markdown
    assert '<span data-count="2">After</span>' in markdown
    assert (
        provider._clean_truncated_repeats(
            "prefix abcabcabcabcabc",
            min_text_len=1,
            max_period=3,
            min_repeat_chars=9,
            min_repeat_times=3,
        )
        == "prefix abc"
    )
