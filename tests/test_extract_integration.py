from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from parse_bench.evaluation.evaluators.extract import ExtractEvaluator
from parse_bench.evaluation.evaluators.parse import ParseEvaluator
from parse_bench.evaluation.runner import EvaluationRunner, _evaluate_single_worker
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.schemas.evaluation import EvaluationResult, MetricValue
from parse_bench.schemas.extract_output import ExtractOutput, FieldCitation
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases import filter_verified_test_rules, load_test_cases
from parse_bench.test_cases.schema import ExtractTestCase, ParseTestCase


def _extract_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "invoice": {
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "date": {"type": "string"},
                },
            }
        },
    }


def test_extract_sidecar_loader_ignores_companion_jsonl(tmp_path: Path) -> None:
    pdf_path = tmp_path / "payroll_7.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    (tmp_path / "payroll_7.v2.raw_words.jsonl").write_text('{"word":"ignored"}\n', encoding="utf-8")
    (tmp_path / "payroll_7.test.json").write_text(
        json.dumps(
            {
                "data_schema": _extract_schema(),
                "expected_output": {"invoice": {"number": "INV-001"}},
                "test_rules": [
                    {
                        "type": "extract_field",
                        "field_path": "invoice.number",
                        "expected_value": "INV-001",
                        "bboxes": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_test_cases(tmp_path, product_type="extract")

    assert len(cases) == 1
    assert isinstance(cases[0], ExtractTestCase)
    assert cases[0].test_id == f"{tmp_path.name}/payroll_7"
    assert cases[0].get_extract_field_rules()[0].field_path == "invoice.number"


def test_extract_evaluator_emits_native_extract_metrics_only(tmp_path: Path) -> None:
    case = ExtractTestCase(
        test_id="docs/payroll_7",
        group="docs",
        file_path=tmp_path / "payroll_7.pdf",
        schema=_extract_schema(),
        expected_output={"invoice": {"number": "INV-001", "date": "2026-05-01"}},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "invoice.number",
                "expected_value": "INV-001",
                "bboxes": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1]}],
            },
            {
                "type": "extract_field",
                "field_path": "invoice.date",
                "expected_value": "2026-05-01",
                "bboxes": [{"page": 1, "bbox": [0.5, 0.2, 0.2, 0.1]}],
            },
        ],
    )
    now = datetime.now()
    result = InferenceResult(
        request=InferenceRequest(
            example_id="docs/payroll_7",
            source_file_path=str(case.file_path),
            product_type=ProductType.EXTRACT,
            schema_override=case.data_schema,
        ),
        pipeline_name="extend_extract",
        product_type=ProductType.EXTRACT,
        raw_output={"job_id": "ext-123", "parse_config_id": "cfg-123"},
        output=ExtractOutput(
            example_id="docs/payroll_7",
            pipeline_name="extend_extract",
            extracted_data={"invoice": {"number": "INV-001", "date": "May 1, 2026"}},
            field_citations=[
                FieldCitation(field_path="invoice.number", page=1, bbox=[0.1, 0.2, 0.3, 0.1]),
                FieldCitation(field_path="invoice.date", page=1, bbox=[0.5, 0.2, 0.2, 0.1]),
            ],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )

    evaluated = ExtractEvaluator().evaluate(result, case)
    by_name = {metric.metric_name: metric for metric in evaluated.metrics}

    for metric_name in (
        "extract_value_precision",
        "extract_value_recall",
        "extract_value_f1",
        "extract_value_pass_rate",
        "extract_bbox_iou",
        "extract_bbox_recall",
        "extract_localization_pass_rate",
        "extract_attribution_pass_rate",
        "extract_element_pass_rate",
    ):
        assert by_name[metric_name].value == pytest.approx(1.0)

    for metric_name in (
        "extract_value_pass_rate",
        "extract_localization_pass_rate",
        "extract_attribution_pass_rate",
        "extract_element_pass_rate",
    ):
        assert by_name[metric_name].metadata["passed"] == 2
        assert by_name[metric_name].metadata["total"] == 2

    assert by_name["extract_bbox_iou"].metadata["score_sum"] == pytest.approx(2.0)
    assert by_name["extract_bbox_iou"].metadata["score_count"] == 2
    assert by_name["extract_bbox_recall"].metadata["score_sum"] == pytest.approx(2.0)
    assert by_name["extract_bbox_recall"].metadata["score_count"] == 2

    assert "extract_field_value_pass_rate" not in by_name
    assert "extract_field_localization_pass_rate" not in by_name
    assert "extract_field_attribution_pass_rate" not in by_name
    assert "extract_field_element_pass_rate" not in by_name
    assert evaluated.job_id == "ext-123"


def test_verified_only_filter_removes_unverified_rules_generically(tmp_path: Path) -> None:
    case = ParseTestCase(
        test_id="docs/payroll_7",
        group="docs",
        file_path=tmp_path / "payroll_7.pdf",
        test_rules=[
            {"type": "present", "text": "keep me"},
            {"type": "present", "text": "drop me", "verified": False},
        ],
    )

    filtered = filter_verified_test_rules(case)

    assert filtered.test_rules is not None
    assert len(filtered.test_rules) == 1
    assert filtered.test_rules[0].get("text") == "keep me"


def test_extract_evaluator_scores_filtered_verified_rules(tmp_path: Path) -> None:
    case = ExtractTestCase(
        test_id="docs/payroll_7",
        group="docs",
        file_path=tmp_path / "payroll_7.pdf",
        schema=_extract_schema(),
        expected_output={"invoice": {"number": "INV-001", "date": "MISSING-VALUE"}},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "invoice.number",
                "expected_value": "INV-001",
                "bboxes": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1]}],
                "verified": True,
            },
            {
                "type": "extract_field",
                "field_path": "invoice.date",
                "expected_value": "MISSING-VALUE",
                "bboxes": [{"page": 1, "bbox": [0.5, 0.2, 0.2, 0.1]}],
                "verified": False,
            },
        ],
    )
    now = datetime.now()
    result = InferenceResult(
        request=InferenceRequest(
            example_id="docs/payroll_7",
            source_file_path=str(case.file_path),
            product_type=ProductType.EXTRACT,
            schema_override=case.data_schema,
        ),
        pipeline_name="extend_extract",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id="docs/payroll_7",
            pipeline_name="extend_extract",
            extracted_data={"invoice": {"number": "INV-001", "date": "2026-05-01"}},
            field_citations=[
                FieldCitation(field_path="invoice.number", page=1, bbox=[0.1, 0.2, 0.3, 0.1]),
            ],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )

    default_by_name = {metric.metric_name: metric for metric in ExtractEvaluator().evaluate(result, case).metrics}
    verified_case = filter_verified_test_rules(case)
    verified_by_name = {
        metric.metric_name: metric for metric in ExtractEvaluator().evaluate(result, verified_case).metrics
    }

    assert default_by_name["extract_value_pass_rate"].metadata["total"] == 2
    assert default_by_name["extract_value_pass_rate"].metadata["passed"] == 1
    assert default_by_name["extract_bbox_iou"].metadata["score_count"] == 2
    assert "field_accuracy[invoice.date]" in default_by_name

    assert len(verified_case.test_rules or []) == 1
    assert verified_by_name["extract_value_pass_rate"].metadata["total"] == 1
    assert verified_by_name["extract_value_pass_rate"].metadata["passed"] == 1
    assert verified_by_name["extract_bbox_iou"].metadata["score_count"] == 1
    assert "field_accuracy[invoice.date]" not in verified_by_name


def test_extract_avg_micro_aggregation() -> None:
    runner = EvaluationRunner(output_dir=Path("/tmp/unused"))
    results = [
        EvaluationResult(
            test_id="a",
            example_id="a",
            pipeline_name="p",
            product_type="extract",
            success=True,
            metrics=[
                MetricValue(
                    metric_name="extract_element_pass_rate",
                    value=0.5,
                    metadata={"passed": 1, "total": 2, "tp": 1, "fp": 1, "fn": 0},
                )
            ],
        ),
        EvaluationResult(
            test_id="b",
            example_id="b",
            pipeline_name="p",
            product_type="extract",
            success=True,
            metrics=[
                MetricValue(
                    metric_name="extract_element_pass_rate",
                    value=1.0,
                    metadata={"passed": 3, "total": 3, "tp": 3, "fp": 0, "fn": 0},
                )
            ],
        ),
    ]

    aggregate = runner._aggregate_metrics(results)

    assert aggregate["avg_extract_element_pass_rate"] == 0.75
    assert aggregate["micro_extract_element_pass_rate"] == 0.8
    assert "macro_extract_element_pass_rate" not in aggregate
    assert aggregate["total_extract_element_pass_rate_passed"] == 4.0
    assert aggregate["total_extract_element_pass_rate_evaluated"] == 5.0
    assert aggregate["total_extract_element_pass_rate_tp"] == 4.0
    assert aggregate["total_extract_element_pass_rate_fp"] == 1.0
    assert aggregate["total_extract_element_pass_rate_fn"] == 0.0


def test_requested_extract_pipelines_registered() -> None:
    extend_pipeline = get_pipeline("extend_extract")

    assert isinstance(extend_pipeline, PipelineSpec)
    assert extend_pipeline.product_type == ProductType.EXTRACT
    assert extend_pipeline.provider_name == "extend"
    assert extend_pipeline.config["advancedOptions"]["citationsEnabled"] is True


def test_parse_evaluator_scores_extract_field_grounding_rules(tmp_path: Path) -> None:
    case = ExtractTestCase(
        test_id="docs/payroll_7",
        group="docs",
        file_path=tmp_path / "payroll_7.pdf",
        schema=_extract_schema(),
        expected_output={"invoice": {"number": "INV-001"}},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "invoice.number",
                "expected_value": "INV-001",
                "bboxes": [{"page": 1, "bbox": [0.1, 0.2, 0.1, 0.05]}],
            }
        ],
    )
    now = datetime.now()
    result = InferenceResult(
        request=InferenceRequest(
            example_id="docs/payroll_7",
            source_file_path=str(case.file_path),
            product_type=ProductType.PARSE,
        ),
        pipeline_name="llamaparse_agentic",
        product_type=ProductType.PARSE,
        raw_output={
            "v2_grounded_items": [
                {
                    "page_number": 1,
                    "page_width": 1000,
                    "page_height": 1000,
                    "items": [
                        {
                            "md": "Invoice INV-001",
                            "grounding": {
                                "source": "md",
                                "lines": [
                                    {
                                        "span": [8, 15],
                                        "bbox": {"x": 100, "y": 200, "w": 100, "h": 50},
                                        "words": [
                                            {
                                                "span": [8, 15],
                                                "bbox": {"x": 100, "y": 200, "w": 100, "h": 50},
                                            }
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        },
        output=ParseOutput(
            example_id="docs/payroll_7",
            pipeline_name="llamaparse_agentic",
            markdown="Invoice INV-001",
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )

    evaluated = ParseEvaluator().evaluate(result, case)
    by_name = {metric.metric_name: metric for metric in evaluated.metrics}

    assert by_name["parse_field_localization_pass_rate"].value == 1.0
    assert by_name["parse_field_attribution_pass_rate"].value == 1.0
    assert by_name["parse_field_element_pass_rate"].value == 1.0
    assert by_name["parse_field_iou"].value == 1.0
    assert by_name["parse_field_iou"].metadata["score_sum"] == pytest.approx(1.0)
    assert by_name["parse_field_iou"].metadata["score_count"] == 1
    assert by_name["parse_field_bbox_recall"].value == pytest.approx(1.0)
    assert by_name["parse_field_bbox_recall"].metadata["score_sum"] == pytest.approx(1.0)
    assert by_name["parse_field_bbox_recall"].metadata["score_count"] == 1
    assert by_name["parse_field_gt_count"].value == 1.0
    assert "extract_field_localization_pass_rate" not in by_name
    assert "extract_field_attribution_pass_rate" not in by_name
    assert "extract_field_element_pass_rate" not in by_name


def test_parse_evaluator_scores_filtered_verified_rules(tmp_path: Path) -> None:
    case = ExtractTestCase(
        test_id="docs/payroll_7",
        group="docs",
        file_path=tmp_path / "payroll_7.pdf",
        schema=_extract_schema(),
        expected_output={"invoice": {"number": "INV-001", "date": "MISSING-VALUE"}},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "invoice.number",
                "expected_value": "INV-001",
                "bboxes": [{"page": 1, "bbox": [0.1, 0.2, 0.1, 0.05]}],
                "verified": True,
            },
            {
                "type": "extract_field",
                "field_path": "invoice.date",
                "expected_value": "MISSING-VALUE",
                "bboxes": [{"page": 1, "bbox": [0.5, 0.2, 0.1, 0.05]}],
                "verified": False,
            },
        ],
    )
    now = datetime.now()
    result = InferenceResult(
        request=InferenceRequest(
            example_id="docs/payroll_7",
            source_file_path=str(case.file_path),
            product_type=ProductType.PARSE,
        ),
        pipeline_name="llamaparse_agentic",
        product_type=ProductType.PARSE,
        raw_output={
            "v2_grounded_items": [
                {
                    "page_number": 1,
                    "page_width": 1000,
                    "page_height": 1000,
                    "items": [
                        {
                            "md": "Invoice INV-001",
                            "grounding": {
                                "source": "md",
                                "lines": [
                                    {
                                        "span": [8, 15],
                                        "bbox": {"x": 100, "y": 200, "w": 100, "h": 50},
                                        "words": [
                                            {
                                                "span": [8, 15],
                                                "bbox": {"x": 100, "y": 200, "w": 100, "h": 50},
                                            }
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        },
        output=ParseOutput(
            example_id="docs/payroll_7",
            pipeline_name="llamaparse_agentic",
            markdown="Invoice INV-001",
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )

    default_by_name = {metric.metric_name: metric for metric in ParseEvaluator().evaluate(result, case).metrics}
    verified_case = filter_verified_test_rules(case)
    verified_by_name = {
        metric.metric_name: metric for metric in ParseEvaluator().evaluate(result, verified_case).metrics
    }

    assert default_by_name["parse_field_localization_pass_rate"].metadata["total"] == 2
    assert default_by_name["parse_field_iou"].metadata["score_count"] == 2

    assert len(verified_case.test_rules or []) == 1
    assert verified_by_name["parse_field_localization_pass_rate"].metadata["total"] == 1
    assert verified_by_name["parse_field_iou"].metadata["score_count"] == 1


def test_parallel_worker_respects_verified_only_flag(tmp_path: Path) -> None:
    case = ExtractTestCase(
        test_id="docs/payroll_7",
        group="docs",
        file_path=tmp_path / "payroll_7.pdf",
        schema=_extract_schema(),
        expected_output={"invoice": {"number": "INV-001", "date": "MISSING-VALUE"}},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "invoice.number",
                "expected_value": "INV-001",
                "bboxes": [{"page": 1, "bbox": [0.1, 0.2, 0.3, 0.1]}],
                "verified": True,
            },
            {
                "type": "extract_field",
                "field_path": "invoice.date",
                "expected_value": "MISSING-VALUE",
                "bboxes": [{"page": 1, "bbox": [0.5, 0.2, 0.2, 0.1]}],
                "verified": False,
            },
        ],
    )
    now = datetime.now()
    result = InferenceResult(
        request=InferenceRequest(
            example_id="docs/payroll_7",
            source_file_path=str(case.file_path),
            product_type=ProductType.EXTRACT,
            schema_override=case.data_schema,
        ),
        pipeline_name="extend_extract",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id="docs/payroll_7",
            pipeline_name="extend_extract",
            extracted_data={"invoice": {"number": "INV-001", "date": "2026-05-01"}},
            field_citations=[
                FieldCitation(field_path="invoice.number", page=1, bbox=[0.1, 0.2, 0.3, 0.1]),
            ],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )

    worker_result = _evaluate_single_worker(
        result.model_dump(),
        case.model_dump(),
        "extract",
        False,
        "extract",
        verified_only=True,
    )
    evaluated = EvaluationResult.model_validate(worker_result)
    by_name = {metric.metric_name: metric for metric in evaluated.metrics}

    assert by_name["extract_value_pass_rate"].metadata["total"] == 1
