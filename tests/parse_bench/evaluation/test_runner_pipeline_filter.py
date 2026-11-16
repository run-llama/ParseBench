"""Pipeline selection follows result metadata rather than directory substrings."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from parse_bench.evaluation import runner as runner_mod
from parse_bench.evaluation.runner import EvaluationRunner
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _write_result(path: Path, pipeline: str) -> None:
    now = datetime.now(UTC)
    result = InferenceResult(
        request=InferenceRequest(example_id="g/doc", source_file_path="doc.pdf", product_type=ProductType.PARSE),
        pipeline_name=pipeline,
        product_type=ProductType.PARSE,
        output=ParseOutput(example_id="g/doc", pipeline_name=pipeline, markdown="Revenue rose by 20 percent."),
        raw_output={},
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(), encoding="utf-8")


@pytest.mark.parametrize("layout", ["standard", "renamed", "flat"])
@pytest.mark.parametrize("selection", ["alpha", "alpha_plus", None], ids=["short", "long", "all"])
def test_pipeline_filter_uses_exact_result_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, selection: str | None
) -> None:
    cases = tmp_path / "cases" / "g"
    cases.mkdir(parents=True)
    (cases / "doc.pdf").write_bytes(b"%PDF-1.4 dummy")
    (cases / "doc.test.json").write_text(
        json.dumps({"expected_markdown": "Revenue rose by 20 percent."}), encoding="utf-8"
    )
    outputs = tmp_path / "outputs"
    for index, pipeline in enumerate(["alpha", "alpha_plus"]):
        folder = pipeline if layout == "standard" else f"renamed_{index}"
        path = outputs / folder / "g" / "doc.result.json"
        if layout == "flat":
            path = outputs / "g" / f"doc_{index}.result.json"
        _write_result(path, pipeline)
    monkeypatch.setattr(runner_mod, "ProcessPoolExecutor", ThreadPoolExecutor)
    summary = EvaluationRunner(outputs, tmp_path / "cases", next_metrics=True, next_only=True).run_evaluation(
        pipeline_name=selection, group="g", use_rich=False, max_workers=1
    )

    expected = {selection} if selection else {"alpha", "alpha_plus"}
    assert summary.successful == len(expected)
    assert {result.pipeline_name for result in summary.per_example_results if result.success} == expected
    assert summary.aggregate_metrics["avg_text_next"] == pytest.approx(1.0)


@pytest.mark.parametrize("ancestor", ["alpha", "runs_alpha"])
def test_pipeline_filter_ignores_ancestor_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ancestor: str) -> None:
    cases = tmp_path / "cases" / "g"
    cases.mkdir(parents=True)
    (cases / "doc.pdf").write_bytes(b"%PDF-1.4 dummy")
    (cases / "doc.test.json").write_text(
        json.dumps({"expected_markdown": "Revenue rose by 20 percent."}), encoding="utf-8"
    )
    outputs = tmp_path / ancestor / "outputs"
    _write_result(outputs / "beta" / "g" / "doc.result.json", "beta")
    monkeypatch.setattr(runner_mod, "ProcessPoolExecutor", ThreadPoolExecutor)
    summary = EvaluationRunner(outputs, tmp_path / "cases", next_metrics=True, next_only=True).run_evaluation(
        pipeline_name="alpha", use_rich=False, max_workers=1
    )
    # Keep the runner's existing missing-output policy: score the requested
    # pipeline's absent page as blank instead of scoring beta's perfect output.
    assert {result.pipeline_name for result in summary.per_example_results} == {"alpha"}
    assert summary.aggregate_metrics["avg_text_next"] == 0.0
