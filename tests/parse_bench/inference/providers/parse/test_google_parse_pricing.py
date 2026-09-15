"""Google Gemini parse provider: pricing tables, thinking-token guard, and
empty/RECITATION response handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from parse_bench.inference.providers.base import ProviderTransientError
from parse_bench.inference.providers.parse import google as google_module
from parse_bench.inference.providers.parse.google import (
    _RECITATION_RETRY_PROMPT,
    GoogleProvider,
    _is_rate_limited_gemini_reason,
    _is_recitation_gemini_reason,
    _is_retryable_empty_gemini_reason,
    _request_timeout_ms,
)


def _provider_for_model(model: str) -> GoogleProvider:
    provider = object.__new__(GoogleProvider)
    provider._model = model
    return provider


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3.7-flash", (0.75, 3.75)),
        ("gemini-3.8-flash", (0.75, 3.75)),
        ("gemini-3.5-flash", (1.50, 9.00)),
        ("gemini-3.5-flash-preview-2026-05-01", (1.50, 9.00)),
        ("gemini-3.5-flash-lite", (0.30, 2.50)),
        ("gemini-2.0-flash-lite", (0.075, 0.30)),
        ("gemini-1.5-flash-8b", (0.0375, 0.15)),
        ("gemini-3-flash-preview", (0.50, 3.00)),
        ("gemini-3.1-flash-lite-preview", (0.25, 1.50)),
        ("gemini-2.5-flash", (0.30, 2.50)),
        ("gemini-2.5-flash-lite", (0.10, 0.40)),
    ],
)
def test_get_pricing(model: str, expected: tuple[float, float]) -> None:
    assert _provider_for_model(model)._get_pricing() == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3.5-flash", (0.15, 1.00)),
        ("gemini-3.5-flash-lite", (0.03, 1.00)),
        ("gemini-3.6-flash", (0.15, 1.00)),
        ("gemini-2.5-flash", (0.03, 1.00)),
    ],
)
def test_get_context_cache_pricing(model: str, expected: tuple[float, float]) -> None:
    assert _provider_for_model(model)._get_context_cache_pricing() == expected


def test_cached_tokens_billed_at_cache_rate_for_gemini_3_5_flash() -> None:
    provider = _provider_for_model("gemini-3.5-flash")
    breakdown = provider._usage_cost_breakdown(
        {"input_tokens": 1_000_000, "cached_content_tokens": 1_000_000, "output_tokens": 0}
    )
    assert breakdown["cached_input_cost_usd"] == pytest.approx(0.15)
    assert breakdown["input_cost_usd"] == 0.0


def test_thinking_tokens_not_double_counted_when_included_in_output() -> None:
    provider = _provider_for_model("gemini-2.5-flash")
    usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 1_000_000, "thinking_tokens": 500_000}
    separate = provider._usage_cost_breakdown(usage)
    assert separate["output_and_thinking_cost_usd"] == pytest.approx(2.50 * 1.5)

    folded = provider._usage_cost_breakdown({**usage, "thinking_tokens_in_output_tokens": True})
    assert folded["thinking_cost_usd"] == 0.0
    assert folded["output_and_thinking_cost_usd"] == pytest.approx(2.50)
    assert folded["cost_usd"] == pytest.approx(2.50)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, None), (0, None), ("abc", None), (0.0005, 1000), (120, 120_000), ("30", 30_000)],
)
def test_request_timeout_ms(seconds: Any, expected: int | None) -> None:
    assert _request_timeout_ms(seconds) == expected


@pytest.mark.parametrize(
    "reason",
    [
        "finish_reason=FinishReason.RECITATION",
        "no candidates returned",
        "no candidates (prompt blocked: SAFETY)",
        "candidate has no content",
        "candidate content has no parts",
        "empty text in response",
    ],
)
def test_retryable_empty_reasons(reason: str) -> None:
    assert _is_retryable_empty_gemini_reason(reason)


def test_recitation_and_rate_limit_classifiers() -> None:
    assert _is_recitation_gemini_reason("finish_reason=FinishReason.RECITATION")
    assert not _is_recitation_gemini_reason("finish_reason=FinishReason.MAX_TOKENS")
    assert _is_rate_limited_gemini_reason("429 RESOURCE_EXHAUSTED")
    assert not _is_rate_limited_gemini_reason("empty text in response")


# --- response handling -------------------------------------------------------


class _FakePart:
    def __init__(self, text: str | None):
        self.text = text


class _FakeTypes:
    class Part:
        @staticmethod
        def from_text(text: str) -> _FakePart:
            return _FakePart(text)


def _response(text: str | None, finish_reason: str | None = None) -> Any:
    if text is None and finish_reason is None:
        return SimpleNamespace(candidates=[], prompt_feedback=None, usage_metadata=None)
    content = SimpleNamespace(parts=[_FakePart(text)])
    candidate = SimpleNamespace(content=content, finish_reason=finish_reason)
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


class _FakeClient:
    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.calls: list[list[Any]] = []

    @property
    def models(self) -> Any:
        return self

    def generate_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
        self.calls.append([p.text for p in contents[0].parts])
        return self._responses.pop(0)


def _provider_with_client(responses: list[Any]) -> tuple[GoogleProvider, _FakeClient]:
    provider = _provider_for_model("gemini-2.5-flash")
    client = _FakeClient(responses)
    provider._client = client
    provider._types = _FakeTypes
    return provider, client


def _contents() -> list[Any]:
    return [SimpleNamespace(role="user", parts=[_FakePart("<image>"), _FakePart("prompt")])]


def test_first_attempt_success_does_not_retry() -> None:
    provider, client = _provider_with_client([_response("# ok")])
    text, _usage, summary = provider._generate_with_empty_retry(_contents(), None)
    assert text == "# ok"
    assert summary == ""
    assert len(client.calls) == 1


def test_recitation_retry_appends_anti_recitation_prompt() -> None:
    provider, client = _provider_with_client([_response(None, "FinishReason.RECITATION"), _response("# transcribed")])
    text, _usage, _summary = provider._generate_with_empty_retry(_contents(), None)
    assert text == "# transcribed"
    assert len(client.calls) == 2
    assert client.calls[0] == ["<image>", "prompt"]
    assert client.calls[1] == ["<image>", "prompt", _RECITATION_RETRY_PROMPT]


def test_plain_empty_retry_keeps_prompt_unchanged() -> None:
    provider, client = _provider_with_client([_response(None), _response("# second")])
    text, _usage, _summary = provider._generate_with_empty_retry(_contents(), None)
    assert text == "# second"
    assert client.calls[1] == ["<image>", "prompt"]


def test_two_empty_attempts_escalate_to_transient_error() -> None:
    provider, _client = _provider_with_client([_response(None), _response(None, "FinishReason.RECITATION")])
    text, _usage, summary = provider._generate_with_empty_retry(_contents(), None)
    assert text is None
    assert "no candidates" in summary and "RECITATION" in summary
    with pytest.raises(ProviderTransientError, match="no usable output"):
        provider._raise_if_retryable_empty(summary)


def test_non_retryable_summary_does_not_raise() -> None:
    GoogleProvider._raise_if_retryable_empty("1st=unknown, 2nd=unknown")


def test_parse_image_raises_transient_after_two_empty_responses() -> None:
    provider, _client = _provider_with_client([_response(None), _response(None)])
    provider._max_tokens = 10
    provider._thinking_level = None
    provider._image_to_bytes = lambda image: b"jpeg"  # type: ignore[method-assign]

    class _Types(_FakeTypes):
        class Part(_FakeTypes.Part):
            @staticmethod
            def from_bytes(data: bytes, mime_type: str) -> _FakePart:
                return _FakePart("<image>")

        class GenerateContentConfig:
            def __init__(self, **kwargs: Any):
                self.kwargs = kwargs

        class Content(SimpleNamespace):
            pass

    provider._types = _Types
    with pytest.raises(ProviderTransientError):
        provider._parse_image(object())  # type: ignore[arg-type]


def test_module_exposes_no_specialist_paths() -> None:
    assert not hasattr(google_module, "_SPECIALIST_REPLAY_PRICING_PER_M")
