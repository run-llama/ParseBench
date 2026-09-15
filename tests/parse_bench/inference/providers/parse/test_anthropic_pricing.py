"""Unit tests for Anthropic provider pricing, including the Sonnet 5
introductory-rate transition."""

from __future__ import annotations

from datetime import date

import pytest

from parse_bench.inference.providers.parse import anthropic
from parse_bench.inference.providers.parse.anthropic import AnthropicProvider


def _provider_for_model(model: str) -> AnthropicProvider:
    provider = object.__new__(AnthropicProvider)
    provider._model = model
    return provider


def test_sonnet_5_uses_introductory_pricing_through_august_2026(monkeypatch: pytest.MonkeyPatch) -> None:
    class IntroDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2026, 8, 31)

    monkeypatch.setattr(anthropic, "date", IntroDate)

    assert _provider_for_model("claude-sonnet-5")._get_pricing() == (2.00, 10.00)


def test_sonnet_5_uses_standard_pricing_after_intro_period(monkeypatch: pytest.MonkeyPatch) -> None:
    class StandardDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2026, 9, 1)

    monkeypatch.setattr(anthropic, "date", StandardDate)

    assert _provider_for_model("claude-sonnet-5")._get_pricing() == (3.00, 15.00)


# --- cache-aware cost --------------------------------------------------------


def test_cache_aware_cost_applies_write_and_read_multipliers() -> None:
    usage = {"input": 1_000_000, "output": 1_000_000, "cache_read": 1_000_000, "cache_write": 1_000_000}
    cost = anthropic.anthropic_cache_aware_cost_usd(usage, 3.0, 15.0)
    # input 3.0 + cache write 1.25*3.0 + cache read 0.1*3.0 + output 15.0
    assert cost == pytest.approx(3.0 + 3.75 + 0.30 + 15.0)


def test_cache_aware_cost_matches_flat_formula_without_cache_tokens() -> None:
    usage = {"input": 2_000_000, "output": 500_000}
    assert anthropic.anthropic_cache_aware_cost_usd(usage, 1.0, 5.0) == pytest.approx(2.0 + 2.5)


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
