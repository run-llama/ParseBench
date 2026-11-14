"""Unit tests for Anthropic provider pricing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from parse_bench.inference.providers.parse import anthropic
from parse_bench.inference.providers.parse.anthropic import AnthropicProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _provider_for_model(model: str) -> AnthropicProvider:
    provider = object.__new__(AnthropicProvider)
    provider._model = model
    return provider


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Longest prefix wins: the x.5 ids must not fall through to the x entry.
        ("claude-opus-5-5", (4.00, 20.00, 0.20, 5.00)),
        ("claude-opus-5", (5.00, 25.00, 0.50, 6.25)),
        ("claude-fable-5-1", (10.00, 50.00, 0.25, 12.50)),
        ("claude-fable-5", (10.00, 50.00, 1.00, 12.50)),
        ("claude-sonnet-5", (2.00, 10.00, 0.20, 2.50)),
        # Dated snapshot ids resolve to their family entry.
        ("claude-haiku-4-5-20251001", (1.00, 5.00, 0.10, 1.25)),
        ("claude-unknown-model", (0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_get_pricing_resolves_longest_prefix(model: str, expected: tuple[float, float, float, float]) -> None:
    assert _provider_for_model(model)._get_pricing() == expected


# --- cache-aware cost --------------------------------------------------------


def test_cache_aware_cost_uses_listed_cache_rates() -> None:
    # Opus 5.5 listed rates: $4 in, $20 out, $0.20 cache read, $5 5m cache write.
    usage = {"input": 1_000_000, "output": 1_000_000, "cache_read": 1_000_000, "cache_write": 1_000_000}
    cost = anthropic.anthropic_cache_aware_cost_usd(usage, 4.0, 20.0, 0.20, 5.0)
    assert cost == pytest.approx(4.0 + 20.0 + 0.20 + 5.0)


def test_cache_aware_cost_matches_flat_formula_without_cache_tokens() -> None:
    usage = {"input": 2_000_000, "output": 500_000}
    assert anthropic.anthropic_cache_aware_cost_usd(usage, 1.0, 5.0, 0.10, 1.25) == pytest.approx(2.0 + 2.5)


def test_extract_usage_reads_cache_tokens() -> None:
    class Usage:
        input_tokens = 100
        output_tokens = 20
        cache_read_input_tokens = 300
        cache_creation_input_tokens = 50

    class Response:
        usage = Usage()
        content: list = []

    usage = AnthropicProvider._extract_usage(Response())
    assert usage["cache_read_tokens"] == 300
    assert usage["cache_write_tokens"] == 50
    assert usage["total_tokens"] == 470


def test_cache_tokens_reach_evaluation_stats() -> None:
    from types import SimpleNamespace

    from parse_bench.evaluation.stats import build_operational_stats

    result = SimpleNamespace(latency_in_ms=None, raw_output={"cache_read_tokens": 300, "cache_write_tokens": 50})
    stats = {stat.name: stat.value for stat in build_operational_stats(result)}  # type: ignore[arg-type]
    assert stats["cache_read_tokens"] == 300
    assert stats["cache_write_tokens"] == 50


@pytest.mark.parametrize(
    ("page_tokens", "expected_cost"),
    [
        ([(50_000, 10_000, 49_000, 1_000)], 0.010615),
        ([(50_001, 10_000, 49_000, 1_000)], 0.0530755),
        ([(60_000, 1, 0, 0), (60_000, 1, 0, 0)], 0.012001),
    ],
    ids=["100k_including_cache", "over_100k_including_cache", "separate_requests_under_100k"],
)
def test_haiku_5_5_cost_uses_each_requests_prompt_length_including_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    page_tokens: list[tuple[int, int, int, int]],
    expected_cost: float,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    provider = _provider_for_model("claude-haiku-5-5")
    provider._mode = "parse_with_layout_file"
    provider._dpi = 150
    provider._max_tokens = 32768
    provider._bbox_scale = 1000
    usages = iter(
        dict(zip(("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"), tokens, strict=True))
        for tokens in page_tokens
    )
    monkeypatch.setattr(anthropic, "split_pdf_to_pages", lambda _: [(b"pdf", 100, 100)] * len(page_tokens))
    monkeypatch.setattr(provider, "_parse_pdf_page_with_layout", lambda _: ([], "", next(usages)))
    pipeline = PipelineSpec(
        pipeline_name="test_haiku_cost", provider_name="anthropic", product_type=ProductType.PARSE, config={}
    )
    request = InferenceRequest(example_id="document", source_file_path=str(source), product_type=ProductType.PARSE)

    raw = provider.run_inference(pipeline, request)

    assert raw.raw_output["cost_usd"] == pytest.approx(expected_cost)
    assert raw.raw_output["cost_per_page_usd"] == pytest.approx(expected_cost / len(page_tokens))


def test_haiku_5_5_omits_rejected_temperature_without_explicit_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def create(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return SimpleNamespace(content=[], usage=None)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider("anthropic", {"model": "claude-haiku-5-5", "mode": "parse_with_layout_file"})
    provider._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))

    provider._parse_pdf_page_with_layout(b"%PDF-1.4")

    assert "temperature" not in calls[0]
