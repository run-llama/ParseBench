"""Meta Muse parse provider: Responses routing, endpoint, pricing and error classification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderConfigError, ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse._layout_utils import SYSTEM_PROMPT_LAYOUT, USER_PROMPT_LAYOUT
from parse_bench.inference.providers.parse.meta import MetaProvider


def _provider(monkeypatch: pytest.MonkeyPatch, create, **config) -> MetaProvider:  # type: ignore[no-untyped-def]
    def chat_create(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Meta pages go through the Responses API")

    monkeypatch.setenv("META_API_KEY", "meta-test")
    provider = MetaProvider(
        "meta",
        {"model": "muse-ember-1.0", "max_tokens": 32768, "reasoning_effort": "low", **config},
    )
    provider._client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
    )
    return provider


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=text,
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            output_tokens_details=SimpleNamespace(reasoning_tokens=200),
            input_tokens_details=SimpleNamespace(cached_tokens=100),
        ),
    )


def test_client_targets_meta_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", "meta-test")
    provider = MetaProvider("meta", {"model": "muse-ember-1.0"})

    assert str(provider._client.base_url).rstrip("/") == "https://api.meta.ai/v1"
    assert provider._mode == "parse_with_layout_file"


def test_pdf_page_goes_through_responses_with_shared_layout_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def create(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return _response('<div data-bbox="[0,0,500,100]" data-label="Text">Hi</div>')

    provider = _provider(monkeypatch, create, mode="parse_with_layout_file")

    items, _text, usage = provider._parse_pdf_page_with_layout(b"%PDF-1.4")

    call = calls[0]
    assert call["model"] == "muse-ember-1.0"
    assert call["reasoning"] == {"effort": "low"}
    assert call["max_output_tokens"] == 32768
    assert call["input"][0]["content"] == SYSTEM_PROMPT_LAYOUT
    assert call["input"][1]["content"][0]["type"] == "input_file"
    assert call["input"][1]["content"][1]["text"] == USER_PROMPT_LAYOUT
    assert items[0]["bbox"] == [0, 0, 500, 100]
    assert usage["output_tokens"] == 300
    assert usage["thinking_tokens"] == 200


def test_image_mode_also_uses_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def create(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return _response("<text>Hi</text>")

    provider = _provider(monkeypatch, create, mode="parse_with_layout")

    provider._parse_image_with_layout(Image.new("RGB", (8, 8), "white"))

    assert calls[0]["input"][1]["content"][0]["type"] == "input_image"


def test_cost_uses_meta_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch, lambda **kwargs: None)
    usage = {"input_tokens": 1_000_000, "cached_tokens": 500_000, "output_tokens": 100_000, "thinking_tokens": 100_000}

    # 500k fresh input at 0.10, 500k cached at 0.004, 200k output (reasoning included) at 0.50.
    assert provider._estimate_cost_usd(usage) == pytest.approx(0.05 + 0.002 + 0.10)


def test_config_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", "meta-test")
    with pytest.raises(ProviderConfigError, match="Invalid mode"):
        MetaProvider("meta", {"model": "muse-ember-1.0", "mode": "image"})
    with pytest.raises(ProviderConfigError, match="No Meta pricing"):
        MetaProvider("meta", {"model": "muse-unknown"})
    monkeypatch.delenv("META_API_KEY")
    with pytest.raises(ProviderConfigError, match="META_API_KEY"):
        MetaProvider("meta", {"model": "muse-ember-1.0"})


def test_error_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch, lambda **kwargs: None)

    class _Error(Exception):
        def __init__(self, message: str, status_code: int):
            super().__init__(message)
            self.status_code = status_code

    with pytest.raises(ProviderTransientError):
        provider._raise_openai_error(_Error("server error", 503))
    with pytest.raises(ProviderTransientError):
        provider._raise_openai_error(_Error("429 rate limit", 429))
    with pytest.raises(ProviderPermanentError):
        provider._raise_openai_error(_Error("invalid request", 400))
