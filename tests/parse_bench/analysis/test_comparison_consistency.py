"""The two comparison entry points must agree on the primary-metric chain.

``comparison.py`` (Pydantic-backed, used by the CLI) and ``comparison_core.py``
(dependency-free, used by the dashboard) each carry a per-product-type
candidate list; a drift between them would make the CLI and the dashboard
crown different winners for the same pair of runs.
"""

from __future__ import annotations

from parse_bench.analysis.comparison import PipelineComparison
from parse_bench.analysis.comparison_core import COMPARISON_METRIC_CANDIDATES
from parse_bench.schemas.evaluation import EvaluationResult, EvaluationSummary, MetricValue


def test_candidate_chains_are_identical() -> None:
    assert PipelineComparison.METRIC_CANDIDATES == COMPARISON_METRIC_CANDIDATES


def test_parse_chain_starts_with_rule_pass_rate_and_excludes_text_score() -> None:
    parse_chain = COMPARISON_METRIC_CANDIDATES["parse"]
    assert parse_chain[0] == "rule_pass_rate"
    assert "grits_trm_composite" in parse_chain
    assert "mAP@[.50:.95]" in parse_chain
    assert "normalized_text_score" not in parse_chain
    assert "form_composite" not in parse_chain


def _result(test_id: str, metrics: dict[str, float]) -> EvaluationResult:
    return EvaluationResult(
        test_id=test_id,
        example_id=test_id,
        pipeline_name="p",
        product_type="parse",
        success=True,
        metrics=[MetricValue(metric_name=k, value=v) for k, v in metrics.items()],
    )


def _summary(results: list[EvaluationResult]) -> EvaluationSummary:
    return EvaluationSummary(
        total_examples=len(results),
        successful=len(results),
        failed=0,
        skipped=0,
        aggregate_metrics={},
        per_example_results=results,
    )


def test_pipeline_comparison_falls_through_the_chain_per_example() -> None:
    comparison = PipelineComparison.__new__(PipelineComparison)
    layout_only = _result("a", {"mAP@[.50:.95]": 0.4, "normalized_text_score": 0.9})
    rules = _result("b", {"rule_pass_rate": 0.7, "mAP@[.50:.95]": 0.1})

    assert comparison._get_comparison_metric(layout_only, "parse") == 0.4
    assert comparison._get_comparison_metric(rules, "parse") == 0.7
    assert comparison._get_comparison_metric(_result("c", {"normalized_text_score": 0.5}), "parse") is None


def test_pipeline_comparison_label_matches_the_metric_actually_emitted() -> None:
    comparison = PipelineComparison.__new__(PipelineComparison)

    layout_only = _summary([_result("a", {"mAP@[.50:.95]": 0.4})])
    assert comparison._resolve_comparison_metric_name(layout_only, "parse") == "mAP@[.50:.95]"

    mixed = _summary([_result("a", {"mAP@[.50:.95]": 0.4}), _result("b", {"rule_pass_rate": 0.7})])
    assert comparison._resolve_comparison_metric_name(mixed, "parse") == "rule_pass_rate"

    # Empty summaries still get a sane label.
    assert comparison._resolve_comparison_metric_name(_summary([]), "parse") == "rule_pass_rate"
    assert comparison._resolve_comparison_metric_name(_summary([]), "layout_detection") == "mAP@[.50:.95]"
    assert comparison._resolve_comparison_metric_name(_summary([]), "unknown") == "accuracy"
