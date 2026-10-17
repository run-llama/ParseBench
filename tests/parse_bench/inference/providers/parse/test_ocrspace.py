"""ocrspace provider: response parsing, error taxonomy, config validation and normalize."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.ocrspace import OCRSpaceProvider, _is_rate_limit_message
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

_OK_PAYLOAD = {
    "ParsedResults": [
        {
            "ParsedText": "Hello world\r\n",
            "TextOverlay": {
                "Lines": [
                    {
                        "Words": [
                            {"WordText": "Hello", "Left": 10, "Top": 20, "Width": 40, "Height": 12},
                            {"WordText": "world", "Left": 55, "Top": 20, "Width": 40, "Height": 12},
                        ]
                    }
                ],
                "HasOverlay": True,
            },
        },
        {"ParsedText": "Page two\r\n", "TextOverlay": {"Lines": [], "HasOverlay": False}},
    ],
    "OCRExitCode": 1,
    "IsErroredOnProcessing": False,
}


def _provider(**cfg):
    return OCRSpaceProvider("ocrspace", {"api_key": "test-key", **cfg})


def _request(tmp_path: Path, name: str = "doc.png", content: bytes = b"x" * 10):
    p = tmp_path / name
    p.write_bytes(content)
    return InferenceRequest(
        example_id="ex1",
        source_file_path=str(p),
        product_type=ProductType.PARSE,
    )


def _pipeline():
    return PipelineSpec(pipeline_name="ocrspace_default", provider_name="ocrspace", product_type=ProductType.PARSE)


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _run_with_payload(tmp_path, payload, status_code=200, text=""):
    provider = _provider()
    with patch("requests.post", return_value=_Resp(status_code, payload, text)) as mock_post:
        result = provider.run_inference(_pipeline(), _request(tmp_path))
    return result, mock_post


def test_parses_text_and_word_boxes(tmp_path):
    result, mock_post = _run_with_payload(tmp_path, _OK_PAYLOAD)
    pages = result.raw_output["pages"]
    assert result.raw_output["num_pages"] == 2
    assert pages[0]["text"] == "Hello world\r\n"
    assert [w["text"] for w in pages[0]["words"]] == ["Hello", "world"]
    assert pages[0]["words"][0]["left"] == 10
    assert pages[1]["words"] == []
    assert result.raw_output["cost_usd"] == 0.0
    # apikey header is sent, file is multipart
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["apikey"] == "test-key"
    assert kwargs["data"]["OCREngine"] == "2"


def test_normalize_builds_pages_and_markdown(tmp_path):
    provider = _provider()
    raw, _ = _run_with_payload(tmp_path, _OK_PAYLOAD)
    out = provider.normalize(raw)
    assert [p.page_index for p in out.output.pages] == [0, 1]
    assert out.output.pages[0].markdown == "Hello world\r\n"
    assert "Page two" in out.output.markdown


def test_api_error_message_raises_permanent(tmp_path):
    with pytest.raises(ProviderPermanentError):
        _run_with_payload(tmp_path, {"OCRExitCode": 2, "ErrorMessage": "E101: file too big"})


def test_rate_limit_message_raises_rate_limit(tmp_path):
    assert _is_rate_limit_message("E207: rate limit exceeded")
    with pytest.raises(ProviderRateLimitError):
        _run_with_payload(tmp_path, {"OCRExitCode": 2, "ErrorMessage": "E207: rate limit exceeded"})


def test_http_429_raises_rate_limit(tmp_path):
    with pytest.raises(ProviderRateLimitError):
        _run_with_payload(tmp_path, None, status_code=429, text="too many")


def test_http_500_raises_transient(tmp_path):
    with pytest.raises(ProviderTransientError):
        _run_with_payload(tmp_path, None, status_code=500, text="oops")


def test_http_400_raises_permanent(tmp_path):
    with pytest.raises(ProviderPermanentError):
        _run_with_payload(tmp_path, None, status_code=400, text="bad request")


def test_connection_error_raises_transient(tmp_path):
    import requests

    provider = _provider()
    with patch("requests.post", side_effect=requests.ConnectionError("dns fail")):
        with pytest.raises(ProviderTransientError):
            provider.run_inference(_pipeline(), _request(tmp_path))


def test_invalid_engine_rejected():
    with pytest.raises(ProviderConfigError):
        _provider(ocr_engine=4)


def test_oversize_file_rejected(tmp_path):
    provider = _provider(max_file_size_mb=0.000001)
    with pytest.raises(ProviderPermanentError, match="over the"):
        provider.run_inference(_pipeline(), _request(tmp_path))


def test_unsupported_extension_rejected(tmp_path):
    provider = _provider()
    with pytest.raises(ProviderPermanentError, match="only supports"):
        provider.run_inference(_pipeline(), _request(tmp_path, name="doc.txt"))


def test_missing_file_rejected(tmp_path):
    provider = _provider()
    req = InferenceRequest(
        example_id="ex1",
        source_file_path=str(tmp_path / "nope.png"),
        product_type=ProductType.PARSE,
    )
    with pytest.raises(ProviderPermanentError, match="not found"):
        provider.run_inference(_pipeline(), req)


def test_demo_key_fallback(monkeypatch):
    monkeypatch.delenv("OCRSPACE_API_KEY", raising=False)
    p = OCRSpaceProvider("ocrspace", {})
    assert p._api_key == "helloworld"
