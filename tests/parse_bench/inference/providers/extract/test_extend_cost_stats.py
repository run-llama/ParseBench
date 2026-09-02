"""Unit tests for ExtendProvider cost-stat surfacing (``_attach_cost_stats``)."""

from __future__ import annotations

from typing import Any

from parse_bench.inference.providers.extract.extend import _attach_cost_stats


def _raw_output(credits: float | None = 30.0, page_count: int | None = 10) -> dict[str, Any]:
    processor_run: dict[str, Any] = {"id": "dpr_123", "status": "PROCESSED"}
    if credits is not None:
        processor_run["usage"] = {"credits": credits}
    if page_count is not None:
        processor_run["files"] = [{"id": "file_abc", "metadata": {"page_count": page_count}}]
    return {"success": True, "processor_run": processor_run}


def test_surfaces_measured_extraction_credits_and_estimated_parse_credits() -> None:
    raw = _raw_output(credits=30.0, page_count=10)
    _attach_cost_stats(raw, {})
    assert raw["credits_used"] == 30.0
    assert raw["num_pages"] == 10
    # Performance parser default: 2 credits/page
    assert raw["parse_credits"] == 20.0
    assert raw["total_credits"] == 50.0
    assert raw["credits_per_page"] == 5.0
    # USD populates from the hardcoded PAYG list rate ($0.0125/credit) so the
    # dashboard cost columns render.
    assert raw["cost_usd"] == 50.0 * 0.0125
    assert raw["cost_per_page_usd"] == raw["cost_usd"] / 10


def test_prefers_authoritative_total_credits_when_present() -> None:
    raw = _raw_output(credits=30.0, page_count=10)
    raw["processor_run"]["usage"]["totalCredits"] = 80.0
    _attach_cost_stats(raw, {})
    assert raw["credits_used"] == 30.0
    assert raw["parse_credits"] == 50.0
    assert raw["total_credits"] == 80.0
    assert raw["credits_per_page"] == 8.0
    assert raw["cost_per_page_usd"] == 0.10


def test_parse_credits_per_page_override() -> None:
    raw = _raw_output(credits=7.0, page_count=10)
    _attach_cost_stats(raw, {"parse_credits_per_page": 0.5})
    assert raw["parse_credits"] == 5.0
    assert raw["total_credits"] == 12.0


def test_missing_usage_only_sets_num_pages() -> None:
    """Runs created before 2025-10-07 / legacy billing return no usage block."""
    raw = _raw_output(credits=None, page_count=10)
    _attach_cost_stats(raw, {})
    assert raw["num_pages"] == 10
    assert "credits_used" not in raw
    assert "total_credits" not in raw


def test_missing_files_metadata_keeps_extraction_credits() -> None:
    raw = _raw_output(credits=30.0, page_count=None)
    _attach_cost_stats(raw, {})
    assert "num_pages" not in raw
    assert raw["credits_used"] == 30.0
    assert raw["parse_credits"] == 0.0
    assert raw["total_credits"] == 30.0
