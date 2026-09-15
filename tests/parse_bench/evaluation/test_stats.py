"""Operational stat extraction: timing validity guards and cost/credit keys."""

from __future__ import annotations

from types import SimpleNamespace

from parse_bench.evaluation.stats import (
    _RAW_OUTPUT_STATS,
    build_operational_stats,
    is_valid_timing_stat,
)


def _names_and_values(result: object) -> list[tuple[str, float]]:
    return [(stat.name, stat.value) for stat in build_operational_stats(result)]  # type: ignore[arg-type]


def test_positive_latency_emits_latency_and_per_page_rate() -> None:
    result = SimpleNamespace(latency_in_ms=125, raw_output={"num_pages": 5})
    assert _names_and_values(result) == [("latency_ms", 125.0), ("latency_ms_per_page", 25.0)]


def test_zero_latency_emits_latency_without_per_page_rate() -> None:
    """Zero is a real measurement (an explicit no-op) but has no meaningful rate."""
    result = SimpleNamespace(latency_in_ms=0, raw_output={"num_pages": 5})
    assert _names_and_values(result) == [("latency_ms", 0.0)]


def test_invalid_latency_values_are_dropped() -> None:
    for bad in (None, -1, float("nan"), float("inf"), True):
        result = SimpleNamespace(latency_in_ms=bad, raw_output={"num_pages": 5})
        assert _names_and_values(result) == [], repr(bad)


def test_per_page_rate_requires_a_positive_numeric_page_count() -> None:
    for pages in (0, -2, None, True, "3"):
        result = SimpleNamespace(latency_in_ms=100, raw_output={"num_pages": pages})
        assert _names_and_values(result) == [("latency_ms", 100.0)], repr(pages)


def test_is_valid_timing_stat_rejects_bools_negatives_and_non_finite() -> None:
    assert is_valid_timing_stat("latency_ms", 0)
    assert is_valid_timing_stat("latency_ms", 12.5)
    assert not is_valid_timing_stat("latency_ms", True)
    assert not is_valid_timing_stat("latency_ms", -0.1)
    assert not is_valid_timing_stat("latency_ms", float("nan"))
    assert not is_valid_timing_stat("latency_ms", "12")


def test_cost_credit_and_tool_keys_are_picked_up_from_raw_output() -> None:
    raw = {
        "num_pages": 2,
        "cost_usd": 1.5,
        "cache_write_cost_usd": 0.25,
        "container_cost_usd": 0.1,
        "container_time_cost_usd": 0.05,
        "tool_surcharge_usd": 0.2,
        "token_cost_usd": 0.9,
        "parse_credits": 6,
        "extract_credits": 4,
        "total_credits": 10,
        "credits_per_page": 5,
        "num_pages_billed": 2,
        "cache_write_tokens": 300,
        "num_tool_calls": 3,
        "num_containers": 1,
        "num_regions": 4,
    }
    result = SimpleNamespace(latency_in_ms=10, raw_output=raw)
    stats = {stat.name: (stat.value, stat.unit) for stat in build_operational_stats(result)}  # type: ignore[arg-type]

    for key, value in raw.items():
        if key == "num_pages":
            continue
        assert stats[key][0] == float(value), key
    assert stats["parse_credits"][1] == "credits"
    assert stats["credits_per_page"][1] == "credits/page"
    assert stats["num_pages_billed"][1] == "pages"
    assert stats["num_tool_calls"][1] == "calls"
    assert stats["cache_write_tokens"][1] == "tokens"
    assert stats["tool_surcharge_usd"][1] == "$"


def test_raw_output_stat_keys_are_unique() -> None:
    keys = [key for key, _unit in _RAW_OUTPUT_STATS]
    assert len(keys) == len(set(keys))
