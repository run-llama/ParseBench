"""``MetricValue`` and ``RunStat`` round-trip fields they do not declare.

A harness built on parse-bench decorates metric rows with provenance (stage,
checkpoint, cost source). Those fields must survive ``model_dump`` /
``model_validate`` unchanged instead of being silently dropped.
"""

from parse_bench.schemas.evaluation import EvaluationResult, MetricValue, RunStat


def test_metric_value_keeps_extra_fields() -> None:
    row = MetricValue.model_validate({"metric_name": "m", "value": 0.5, "stage": "parse", "checkpoint": 3})
    assert row.model_dump()["stage"] == "parse"
    assert MetricValue.model_validate(row.model_dump()).model_dump()["checkpoint"] == 3


def test_run_stat_keeps_extra_fields() -> None:
    stat = RunStat.model_validate({"name": "latency_ms", "value": 1.0, "unit": "ms", "source": "provider"})
    assert stat.model_dump()["source"] == "provider"


def test_extra_fields_survive_inside_evaluation_result() -> None:
    result = EvaluationResult.model_validate(
        {
            "test_id": "t",
            "example_id": "e",
            "pipeline_name": "p",
            "product_type": "parse",
            "success": True,
            "metrics": [{"metric_name": "m", "value": 1.0, "stage": "s"}],
            "stats": [{"name": "n", "value": 1.0, "unit": "u", "source": "x"}],
        }
    )
    dumped = result.model_dump()
    assert dumped["metrics"][0]["stage"] == "s"
    assert dumped["stats"][0]["source"] == "x"
