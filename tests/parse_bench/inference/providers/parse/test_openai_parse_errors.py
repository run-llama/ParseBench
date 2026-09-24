"""OpenAI parse provider: pricing table, error classification and Responses API routing."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderConfigError, ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse.openai import OpenAIProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


class _OpenAIError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _provider_for_model(model: str) -> OpenAIProvider:
    provider = object.__new__(OpenAIProvider)
    provider._model = model
    return provider


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5-mini", (0.25, 2.00)),
        ("gpt-5-nano", (0.05, 0.40)),
        ("gpt-5", (1.25, 10.00)),
        ("gpt-5.4-mini", (0.75, 4.50)),
        ("gpt-5.6-sol", (4.0, 20.0)),
        ("gpt-5.6-terra", (2.0, 12.0)),
        ("gpt-5.6-luna", (0.2, 1.2)),
        ("gpt-6-astra", (10.0, 50.0)),
        ("gpt-6-sol", (2.0, 10.0)),
        ("gpt-6-luna", (0.1, 0.5)),
        ("gpt-4.5-preview", (75.0, 150.0)),
    ],
)
def test_standard_short_context_pricing(model: str, expected: tuple[float, float]) -> None:
    assert _provider_for_model(model)._get_pricing() == expected


def test_usage_splits_reasoning_out_of_completion_tokens_and_reads_cached_input() -> None:
    """Chat Completions counts reasoning inside completion_tokens (total = prompt + completion)."""
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=2689,
            completion_tokens=3600,
            total_tokens=6289,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=1024),
            prompt_tokens_details=SimpleNamespace(cached_tokens=1792, cache_write_tokens=512),
        )
    )

    usage = OpenAIProvider._extract_usage(response)

    assert usage == {
        "input_tokens": 2689,
        "cached_tokens": 1792,
        "cache_write_tokens": 512,
        "output_tokens": 2576,
        "thinking_tokens": 1024,
        "total_tokens": 6289,
    }
    assert usage["output_tokens"] + usage["thinking_tokens"] == 3600


def test_cost_bills_cached_and_cache_write_input_at_their_own_rates() -> None:
    usage = {
        "input_tokens": 20_000,
        "cached_tokens": 10_000,
        "cache_write_tokens": 8_000,
        "output_tokens": 1_500,
        "thinking_tokens": 500,
    }

    # gpt-6-sol: 2K fresh x $2.00 + 10K cached x $0.20 + 8K writes x $2.50 + 2K out x $10.00.
    sol = _provider_for_model("gpt-6-sol")
    assert sol._estimate_cost_usd(usage) == pytest.approx(0.004 + 0.002 + 0.020 + 0.020)

    # gpt-5.6-terra: 2K fresh x $2.00 + 10K cached x $0.20 + 8K writes x $2.50 + 2K out x $12.00.
    terra = _provider_for_model("gpt-5.6-terra")
    assert terra._estimate_cost_usd(usage) == pytest.approx(0.004 + 0.002 + 0.020 + 0.024)

    # An unlisted model bills cached and written input at its full input rate, as before.
    gpt55 = _provider_for_model("gpt-5.5")
    assert gpt55._estimate_cost_usd(usage) == pytest.approx(20_000 * 5.00 / 1e6 + 2_000 * 30.00 / 1e6)


def test_gpt5_mini_does_not_inherit_gpt5_rate() -> None:
    """Longest-prefix match keeps gpt-5-mini on its own (cheaper) rate."""
    assert _provider_for_model("gpt-5-mini-2026-01-01")._get_pricing() == (0.25, 2.00)


def test_gpt56_intermittent_permission_401_is_retryable() -> None:
    provider = _provider_for_model("gpt-5.6-sol")
    error = _OpenAIError("You have insufficient permissions for this operation", 401)

    with pytest.raises(ProviderTransientError, match="retryable"):
        provider._raise_openai_error(error)


def test_gpt56_permission_401_without_status_code_is_retryable() -> None:
    provider = _provider_for_model("gpt-5.6-terra")
    error = Exception("Error code: 401 - You have insufficient permissions for this operation")

    with pytest.raises(ProviderTransientError):
        provider._raise_openai_error(error)


def test_genuine_bad_key_401_is_permanent() -> None:
    provider = _provider_for_model("gpt-5.6-terra")
    error = _OpenAIError("Incorrect API key provided", 401)

    with pytest.raises(ProviderPermanentError, match="Incorrect API key"):
        provider._raise_openai_error(error)


def test_permission_error_is_not_retried_for_other_models() -> None:
    provider = _provider_for_model("gpt-5.5")
    error = _OpenAIError("You have insufficient permissions for this operation", 401)

    with pytest.raises(ProviderPermanentError, match="insufficient permissions"):
        provider._raise_openai_error(error)


def test_rate_limit_is_transient() -> None:
    with pytest.raises(ProviderTransientError, match="Rate limited"):
        _provider_for_model("gpt-5.5")._raise_openai_error(Exception("429 rate_limit_exceeded"))


def _responses_provider(monkeypatch: pytest.MonkeyPatch, create) -> OpenAIProvider:  # type: ignore[no-untyped-def]
    """A GPT-6 Luna effort-max provider whose Chat Completions call fails."""

    def chat_create(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Chat Completions rejects effort max")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider(
        "openai",
        {
            "model": "gpt-6-luna",
            "max_tokens": 32768,
            "mode": "parse_with_layout_file",
            "reasoning_effort": "max",
            "api": "responses",
        },
    )
    provider._client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
    )
    return provider


def test_responses_api_page_call_sends_effort_max_and_splits_reasoning_out_of_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat Completions rejects effort "max"; api="responses" routes the page through the Responses API."""
    calls: list[dict] = []

    def create(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return SimpleNamespace(
            output_text="<text>Hello</text>",
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=900,
                total_tokens=1900,
                output_tokens_details=SimpleNamespace(reasoning_tokens=600),
                input_tokens_details=SimpleNamespace(cached_tokens=200),
            ),
        )

    provider = _responses_provider(monkeypatch, create)

    _items, text, usage = provider._parse_pdf_page_with_layout(b"%PDF-1.4")

    assert calls[0]["reasoning"] == {"effort": "max"}
    assert calls[0]["max_output_tokens"] == 32768
    assert "reasoning_effort" not in calls[0]
    assert calls[0]["input"][1]["content"][0]["type"] == "input_file"
    assert text == "<text>Hello</text>"
    assert usage == {
        "input_tokens": 1000,
        "cached_tokens": 200,
        "cache_write_tokens": 0,
        "output_tokens": 300,
        "thinking_tokens": 600,
        "total_tokens": 1900,
    }


def test_responses_api_image_page_call_sends_effort_max(monkeypatch: pytest.MonkeyPatch) -> None:
    """PNG/JPG test cases take the image layout path in parse_with_layout_file mode; max must still use Responses."""
    calls: list[dict] = []

    def create(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return SimpleNamespace(
            output_text="<text>Hi</text>",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    provider = _responses_provider(monkeypatch, create)

    _items, text, _usage = provider._parse_image_with_layout(Image.new("RGB", (8, 8), "white"))

    assert calls[0]["reasoning"] == {"effort": "max"}
    assert calls[0]["input"][1]["content"][0]["type"] == "input_image"
    assert text == "<text>Hi</text>"


def test_responses_api_is_refused_outside_parse_with_layout_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ProviderConfigError, match="parse_with_layout_file"):
        OpenAIProvider("openai", {"model": "gpt-6-luna", "mode": "image", "api": "responses"})


def test_unknown_api_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ProviderConfigError, match="Invalid api"):
        OpenAIProvider("openai", {"model": "gpt-6-luna", "mode": "parse_with_layout_file", "api": "batch"})


@pytest.mark.parametrize(
    ("module", "cls", "env_var"),
    [
        ("deepseek", "DeepSeekParseProvider", "DEEPSEEK_API_KEY"),
        ("glm_zai", "GLMZaiParseProvider", "GLM_ZAI_API_KEY"),
    ],
)
def test_subclasses_that_skip_openai_init_still_run_inference(
    module: str, cls: str, env_var: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """These providers skip OpenAIProvider.__init__, so attributes run_inference reads need class defaults."""
    monkeypatch.setenv(env_var, "sk-test")
    provider = getattr(importlib.import_module(f"parse_bench.inference.providers.parse.{module}"), cls)(module, {})
    usage = {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0, "total_tokens": 2}
    provider._parse_image = lambda image: ("text", usage)
    provider._parse_image_with_layout = lambda image: ([], "text", usage)
    png = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(png)

    result = provider.run_inference(
        PipelineSpec(pipeline_name="p", provider_name=module, product_type=ProductType.PARSE, config={}),
        InferenceRequest(example_id="page", source_file_path=str(png), product_type=ProductType.PARSE),
    )

    assert "api" not in result.raw_output["config"]
