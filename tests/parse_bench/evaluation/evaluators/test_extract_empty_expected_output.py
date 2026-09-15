"""An empty-dict ``expected_output`` must still be scored, not skipped."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from parse_bench.evaluation.evaluators.extract import ExtractEvaluator
from parse_bench.schemas.extract_output import ExtractOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import ExtractTestCase


def _result(extracted: dict) -> InferenceResult:
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(
            example_id="group/doc",
            source_file_path="doc.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="extract",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(example_id="group/doc", pipeline_name="extract", extracted_data=extracted),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _case(expected_output: dict | None) -> ExtractTestCase:
    return ExtractTestCase(
        test_id="group/doc",
        group="group",
        file_path=Path("doc.pdf"),
        schema={"type": "object", "properties": {}},
        expected_output=expected_output,
    )


def test_empty_dict_expected_output_emits_accuracy_metric() -> None:
    evaluator = ExtractEvaluator(enable_rule_based=False)
    test_case = _case({})
    assert evaluator.can_evaluate(_result({}), test_case)

    evaluation = evaluator.evaluate(_result({}), test_case)
    names = [metric.metric_name for metric in evaluation.metrics]
    assert "accuracy" in names
    accuracy = next(metric for metric in evaluation.metrics if metric.metric_name == "accuracy")
    assert accuracy.value == 1.0


def test_non_empty_expected_output_still_scored() -> None:
    evaluator = ExtractEvaluator(enable_rule_based=False)
    evaluation = evaluator.evaluate(_result({"name": "Ada"}), _case({"name": "Ada"}))
    accuracy = next(metric for metric in evaluation.metrics if metric.metric_name == "accuracy")
    assert accuracy.value == 1.0
