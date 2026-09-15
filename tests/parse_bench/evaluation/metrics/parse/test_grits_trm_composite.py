from __future__ import annotations

from parse_bench.evaluation.evaluators.parse import ParseEvaluator
from parse_bench.schemas.evaluation import MetricValue


def _mv(name: str, value: float) -> MetricValue:
    return MetricValue(metric_name=name, value=value)


def _composite(metrics: list[MetricValue]) -> MetricValue | None:
    out = ParseEvaluator._compute_grits_trm_composite(metrics, trm_unsupported=False)
    return out[0] if out else None


def test_composite_average_when_both_present() -> None:
    metrics = [_mv("grits_con", 0.8), _mv("table_record_match", 0.6)]
    result = ParseEvaluator._compute_grits_trm_composite(metrics, trm_unsupported=False)
    assert len(result) == 1
    assert result[0].metric_name == "grits_trm_composite"
    assert abs(result[0].value - 0.7) < 1e-9
    assert result[0].metadata["fallback"] is None


def test_composite_falls_back_to_grits_when_trm_unsupported() -> None:
    metrics = [_mv("grits_con", 0.42), _mv("table_record_match", 0.99)]
    result = ParseEvaluator._compute_grits_trm_composite(metrics, trm_unsupported=True)
    assert len(result) == 1
    assert result[0].value == 0.42
    assert result[0].metadata["fallback"] == "grits_only"
    assert result[0].metadata["reason"] == "trm_unsupported"


def test_composite_falls_back_when_trm_missing() -> None:
    metrics = [_mv("grits_con", 0.42)]
    result = ParseEvaluator._compute_grits_trm_composite(metrics, trm_unsupported=False)
    assert len(result) == 1
    assert result[0].value == 0.42
    assert result[0].metadata["reason"] == "trm_missing"


def test_composite_empty_when_grits_missing() -> None:
    metrics = [_mv("table_record_match", 0.6)]
    result = ParseEvaluator._compute_grits_trm_composite(metrics, trm_unsupported=False)
    assert result == []


def test_composite_no_actual_tables_zero_case() -> None:
    # Both inputs zero (no-actual-tables fallback) → composite is derived 0.0
    metrics = [_mv("grits_con", 0.0), _mv("table_record_match", 0.0)]
    result = ParseEvaluator._compute_grits_trm_composite(metrics, trm_unsupported=False)
    assert len(result) == 1
    assert result[0].value == 0.0
    assert result[0].metadata["fallback"] is None
