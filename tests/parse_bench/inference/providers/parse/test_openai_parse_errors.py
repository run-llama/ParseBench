"""OpenAI parse provider: pricing table and error classification."""

from __future__ import annotations

import pytest

from parse_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse.openai import OpenAIProvider


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
        ("gpt-4.5-preview", (75.0, 150.0)),
    ],
)
def test_standard_short_context_pricing(model: str, expected: tuple[float, float]) -> None:
    assert _provider_for_model(model)._get_pricing() == expected


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
