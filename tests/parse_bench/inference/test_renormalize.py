"""Tests for re-normalizing saved inference outputs."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from parse_bench.inference import renormalize as renormalize_mod
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType


def test_renormalize_skips_error_debug_sidecars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = PipelineSpec(
        pipeline_name="test_parse_pipeline",
        provider_name="test_provider",
        product_type=ProductType.PARSE,
    )
    request = InferenceRequest(
        example_id="doc1",
        source_file_path="doc1.pdf",
        product_type=ProductType.PARSE,
    )
    document_url = "https://docs.example/doc?token=document-value#section"
    raw_result = RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output={"markdown": document_url, "nullable": None, "token": "document-value"},
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
        latency_in_ms=1,
    )

    (tmp_path / "_metadata.json").write_text(json.dumps({"pipeline_name": pipeline.pipeline_name}))
    (tmp_path / "doc1.raw.json").write_text(raw_result.model_dump_json())
    (tmp_path / "doc2.error.raw.json").write_text(json.dumps({"debug": "not a RawInferenceResult"}))

    class FakeProvider:
        def recompute_cost(self, raw_output: dict) -> None:
            return None

        def normalize(self, raw: RawInferenceResult) -> InferenceResult:
            return InferenceResult(
                request=raw.request,
                pipeline_name=raw.pipeline_name,
                product_type=raw.product_type,
                raw_output=raw.raw_output,
                output=ParseOutput(
                    example_id=raw.request.example_id,
                    pipeline_name=raw.pipeline_name,
                    pages=[],
                    markdown=raw.raw_output["markdown"],
                ),
                started_at=raw.started_at,
                completed_at=raw.completed_at,
                latency_in_ms=raw.latency_in_ms,
            )

    monkeypatch.setattr(renormalize_mod, "get_pipeline", lambda name: pipeline)
    monkeypatch.setattr(renormalize_mod, "create_provider", lambda spec: FakeProvider())

    assert renormalize_mod.renormalize_results(tmp_path, force=True) == 0
    assert (tmp_path / "doc1.result.json").exists()
    persisted = json.loads((tmp_path / "doc1.result.json").read_text())
    assert persisted["raw_output"] == {
        "markdown": document_url,
        "nullable": None,
        "token": "document-value",
    }
    assert persisted["output"]["markdown"] == document_url
    assert not (tmp_path / "doc2.error.result.json").exists()


def test_renormalize_calls_recompute_cost_generically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The re-pricing seam is invoked for whatever provider renormalize resolves.

    This is the whole point of the generic hook: the runner never names a provider,
    so a new provider that overrides recompute_cost gets re-priced on renormalize
    with zero changes here.
    """
    pipeline = PipelineSpec(
        pipeline_name="test_extract_pipeline",
        provider_name="test_provider",
        product_type=ProductType.PARSE,
    )
    request = InferenceRequest(example_id="doc1", source_file_path="doc1.pdf", product_type=ProductType.PARSE)
    raw_result = RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output={"markdown": "hi", "cost_usd": 0.0},
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
        latency_in_ms=1,
    )
    (tmp_path / "_metadata.json").write_text(json.dumps({"pipeline_name": pipeline.pipeline_name}))
    (tmp_path / "doc1.raw.json").write_text(raw_result.model_dump_json())

    calls: list[dict] = []

    class RepricingProvider:
        def recompute_cost(self, raw_output: dict) -> None:
            calls.append(raw_output)
            raw_output["cost_usd"] = 0.42  # a corrected rate

        def normalize(self, raw: RawInferenceResult) -> InferenceResult:
            return InferenceResult(
                request=raw.request,
                pipeline_name=raw.pipeline_name,
                product_type=raw.product_type,
                raw_output=raw.raw_output,
                output=ParseOutput(
                    example_id=raw.request.example_id,
                    pipeline_name=raw.pipeline_name,
                    pages=[],
                    markdown=raw.raw_output["markdown"],
                ),
                started_at=raw.started_at,
                completed_at=raw.completed_at,
                latency_in_ms=raw.latency_in_ms,
            )

    monkeypatch.setattr(renormalize_mod, "get_pipeline", lambda name: pipeline)
    monkeypatch.setattr(renormalize_mod, "create_provider", lambda spec: RepricingProvider())

    assert renormalize_mod.renormalize_results(tmp_path, force=True) == 0
    assert len(calls) == 1  # invoked once, generically, before normalize
    persisted = json.loads((tmp_path / "doc1.result.json").read_text())
    assert persisted["raw_output"]["cost_usd"] == 0.42


def test_base_provider_recompute_cost_is_a_noop() -> None:
    """A provider that cannot re-price keeps whatever cost it recorded."""
    from parse_bench.inference.providers.base import Provider

    class _BareProvider(Provider):
        def run_inference(self, pipeline, request):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def normalize(self, raw_result):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    raw_output = {"cost_usd": 1.23}
    provider = _BareProvider("bare")
    provider.recompute_cost(raw_output)
    assert raw_output == {"cost_usd": 1.23}
    # The other base no-ops the runner relies on are declared on Provider too.
    assert provider.cancel("doc1") is False
    assert provider.consume_active_job_id("doc1") is None
