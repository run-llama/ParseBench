"""Tests for usage metadata compatibility fields in LlamaParse raw output."""

from __future__ import annotations

from datetime import datetime

import pytest

import parse_bench.inference.providers.parse.llamaparse as llamaparse_module
from parse_bench.inference.providers.parse.llamaparse import LlamaParseProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType


@pytest.fixture(autouse=True)
def _clean_llama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LLAMA_CLOUD_API_KEY", "LLAMA_CLOUD_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llamaparse_module, "_HAS_V2_SDK", True)


def test_attach_usage_metadata_adds_cost_fields_from_page_count() -> None:
    provider = LlamaParseProvider(
        provider_name="llamaparse",
        base_config={"api_key": "test-key", "tier": "agentic"},
    )

    raw_output = {
        "items": {"pages": [{"page_number": 1}, {"page_number": 2}]},
        "job": {"id": "pjb-123"},
    }

    enriched = provider._attach_usage_metadata(raw_output)

    assert enriched["num_pages"] == 2
    assert enriched["credits_used"] == 20
    assert enriched["cost_usd"] == 0.025
    assert enriched["cost_per_page_usd"] == 0.0125
    assert enriched["job_id"] == "pjb-123"


def test_attach_usage_metadata_bills_cost_optimized_pages_at_cost_effective_rate() -> None:
    provider = LlamaParseProvider(
        provider_name="llamaparse",
        base_config={"api_key": "test-key", "tier": "agentic"},
    )

    raw_output = {
        "items": {
            "pages": [
                {"page_number": 1},
                {"page_number": 2, "cost_optimized": True},
                {"page_number": 3, "cost_optimized": True},
                {"page_number": 4},
            ]
        },
        "job": {"id": "pjb-mixed"},
    }

    enriched = provider._attach_usage_metadata(raw_output)

    # 2 full agentic pages * 10 credits + 2 cost-optimized pages * 3 credits = 26
    assert enriched["num_pages"] == 4
    assert enriched["credits_used"] == 26
    assert enriched["cost_usd"] == 26 * 0.00125
    assert enriched["cost_per_page_usd"] == (26 * 0.00125) / 4
    assert enriched["cost_optimized_pages"] == 2


def test_attach_usage_metadata_reads_cost_optimized_from_metadata_pages() -> None:
    """Cost-optimizer billing flags live on metadata.pages, not items.pages."""
    provider = LlamaParseProvider(
        provider_name="llamaparse",
        base_config={"api_key": "test-key", "tier": "agentic"},
    )

    raw_output = {
        "items": {
            "pages": [
                {"page_number": 1, "md": "a"},
                {"page_number": 2, "md": "b"},
                {"page_number": 3, "md": "c"},
            ]
        },
        "metadata": {
            "pages": [
                {"page_number": 1, "cost_optimized": False},
                {"page_number": 2, "cost_optimized": True},
                {"page_number": 3, "cost_optimized": True},
            ]
        },
        "job": {"id": "pjb-meta"},
    }

    enriched = provider._attach_usage_metadata(raw_output)

    # 1 agentic page * 10 + 2 cost-optimized pages * 3 = 16 credits
    assert enriched["num_pages"] == 3
    assert enriched["credits_used"] == 16
    assert enriched["cost_optimized_pages"] == 2
    assert enriched["cost_usd"] == 16 * 0.00125


def test_attach_usage_metadata_no_cost_optimized_flag_unchanged() -> None:
    provider = LlamaParseProvider(
        provider_name="llamaparse",
        base_config={"api_key": "test-key", "tier": "agentic_plus"},
    )

    raw_output = {
        "items": {"pages": [{"page_number": 1}, {"page_number": 2}]},
        "job": {"id": "pjb-noflag"},
    }

    enriched = provider._attach_usage_metadata(raw_output)

    assert enriched["credits_used"] == 90
    assert "cost_optimized_pages" not in enriched


def test_attach_usage_metadata_recomputes_stale_cost_fields() -> None:
    """Stale cost fields (e.g. from runs before the cost-optimizer fix) must be
    overwritten so `bench inference renormalize` heals them. Only `num_pages`
    and `job_id` are still preserved when already present."""
    provider = LlamaParseProvider(
        provider_name="llamaparse",
        base_config={"api_key": "test-key", "tier": "agentic"},
    )

    raw_output = {
        "metadata": {
            "pages": [
                {"page_number": 1, "cost_optimized": True},
                {"page_number": 2, "cost_optimized": False},
            ]
        },
        "job": {"id": "pjb-keep"},
        "num_pages": 2,
        "credits_used": 20,  # stale: old code billed both pages at agentic rate
        "cost_usd": 0.025,
        "cost_per_page_usd": 0.0125,
        "job_id": "preexisting-id",
    }

    enriched = provider._attach_usage_metadata(raw_output)

    # num_pages and job_id are still preserved (not derived from tier)
    assert enriched["num_pages"] == 2
    assert enriched["job_id"] == "preexisting-id"
    # 1 agentic * 10 + 1 cost-optimized * 3 = 13 credits — stale values overwritten
    assert enriched["credits_used"] == 13
    assert enriched["cost_usd"] == 13 * 0.00125
    assert enriched["cost_per_page_usd"] == (13 * 0.00125) / 2
    assert enriched["cost_optimized_pages"] == 1


def test_attach_usage_metadata_is_idempotent() -> None:
    provider = LlamaParseProvider(
        provider_name="llamaparse",
        base_config={"api_key": "test-key", "tier": "agentic"},
    )
    raw_output = {
        "metadata": {"pages": [{"page_number": 1, "cost_optimized": True}, {"page_number": 2}]},
        "job": {"id": "pjb-idem"},
    }

    once = provider._attach_usage_metadata(raw_output)
    twice = provider._attach_usage_metadata(once)

    assert twice == once


def test_attach_usage_metadata_unknown_tier_keeps_num_pages_without_cost() -> None:
    provider = LlamaParseProvider(
        provider_name="llamaparse",
        base_config={"api_key": "test-key", "tier": "unknown_tier"},
    )

    raw_output = {
        "metadata": {"pages": [{"page_number": 1}, {"page_number": 2}, {"page_number": 3}]},
        "job": {"id": "pjb-xyz"},
    }

    enriched = provider._attach_usage_metadata(raw_output)

    assert enriched["num_pages"] == 3
    assert "credits_used" not in enriched
    assert "cost_usd" not in enriched
    assert "cost_per_page_usd" not in enriched
    assert enriched["job_id"] == "pjb-xyz"


def test_normalize_backfills_usage_metadata_into_result_raw_output() -> None:
    provider = LlamaParseProvider(
        provider_name="llamaparse",
        base_config={"api_key": "test-key", "tier": "cost_effective"},
    )

    request = InferenceRequest(
        example_id="sample",
        source_file_path="/tmp/sample.pdf",
        product_type=ProductType.PARSE,
    )
    pipeline = PipelineSpec(
        pipeline_name="llamaparse_cost_effective",
        provider_name="llamaparse",
        product_type=ProductType.PARSE,
        config={"tier": "cost_effective", "version": "latest"},
    )

    raw = RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output={
            "pages": [
                {"page": 1, "md": "first", "text": "first", "items": [], "width": 100, "height": 100},
                {"page": 2, "md": "second", "text": "second", "items": [], "width": 100, "height": 100},
                {"page": 3, "md": "third", "text": "third", "items": [], "width": 100, "height": 100},
            ],
            "job": {"id": "pjb-normalize"},
        },
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=10,
    )

    normalized = provider.normalize(raw)

    assert normalized.raw_output["num_pages"] == 3
    assert normalized.raw_output["credits_used"] == 9
    assert normalized.raw_output["cost_usd"] == 0.01125
    assert normalized.raw_output["cost_per_page_usd"] == 0.00375
    assert normalized.raw_output["job_id"] == "pjb-normalize"
