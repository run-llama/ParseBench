"""Documents with mixed layout and parse rules ("multi_task") score the same in a pool worker as in the main process."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from parse_bench.evaluation.runner import EvaluationRunner, _evaluate_single_worker, _next_only_result
from parse_bench.schemas.evaluation import EvaluationResult
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel, LayoutOutput
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import LayoutDetectionTestCase, ParseTestCase


def _case_and_result() -> tuple[LayoutDetectionTestCase, InferenceResult]:
    case = LayoutDetectionTestCase.model_validate(
        {
            "test_id": "layout/page",
            "group": "layout",
            "file_path": "page.pdf",
            "test_rules": [
                {"type": "layout", "page": 1, "bbox": [0.1, 0.1, 0.5, 0.1], "canonical_class": "Text", "id": "r0"},
                {"type": "present", "text": "Quarterly revenue", "id": "p0"},
            ],
        }
    )
    now = datetime.now()
    result = InferenceResult(
        request=InferenceRequest(example_id="layout/page", source_file_path="page.pdf", product_type=ProductType.PARSE),
        pipeline_name="test",
        product_type=ProductType.PARSE,
        raw_output={},
        output=ParseOutput(
            example_id="layout/page",
            pipeline_name="test",
            markdown="Quarterly revenue rose.",
            pages=[PageIR(page_index=0, markdown="Quarterly revenue rose.")],
            layout_pages=[ParseLayoutPageIR(page_number=1, items=[])],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    return case, result


def _comparable(result: EvaluationResult) -> dict:
    d = result.model_dump(mode="json")
    d.pop("evaluated_at", None)
    return d


@pytest.mark.parametrize("next_metrics", [False, True])
def test_worker_scores_multi_task_like_the_main_process(next_metrics: bool) -> None:
    case, result = _case_and_result()
    serial = EvaluationRunner(output_dir=Path("."), next_metrics=next_metrics)._evaluate_single(
        result, case, None, "multi_task"
    )
    worker = EvaluationResult.model_validate(
        _evaluate_single_worker(
            result.model_dump(), case.model_dump(), "layout_detection", "multi_task", "parse", next_metrics=next_metrics
        )
    )
    assert "not supported in parallel mode" not in (worker.error or "")
    assert _comparable(worker) == _comparable(serial)


def _text_case_and_result() -> tuple[ParseTestCase, InferenceResult]:
    case = ParseTestCase.model_validate(
        {
            "test_id": "text/page",
            "group": "text",
            "file_path": "page.pdf",
            "expected_markdown": "Quarterly revenue rose 12% to $4.2 million.",
            "test_rules": [{"type": "present", "text": "Quarterly revenue", "id": "p0"}],
        }
    )
    now = datetime.now()
    result = InferenceResult(
        request=InferenceRequest(example_id="text/page", source_file_path="page.pdf", product_type=ProductType.PARSE),
        pipeline_name="test",
        product_type=ProductType.PARSE,
        raw_output={},
        output=ParseOutput(
            example_id="text/page",
            pipeline_name="test",
            markdown="Quarterly revenue rose 12% to $4.7 million.",
            pages=[PageIR(page_index=0, markdown="Quarterly revenue rose 12% to $4.7 million.")],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    return case, result


def test_next_only_keeps_the_extended_metrics_and_drops_v1() -> None:
    case, result = _text_case_and_result()
    full_runner = EvaluationRunner(output_dir=Path("."), next_metrics=True)
    full = full_runner._evaluate_single(result, case, full_runner._evaluators["parse"], False)
    only = EvaluationRunner(output_dir=Path("."), next_only=True)._evaluate_single(result, case, None, False)
    family = ("text_next", "formatting_v2", "table_next", "chart_next", "layout_grounding", "forms_v2", "form_kv")
    expected = {m.metric_name: m.value for m in full.metrics if m.metric_name.startswith(family)}
    assert "text_next" in expected
    assert any(not m.metric_name.startswith(family) for m in full.metrics), "the full run also has v1 metrics"
    assert {m.metric_name: m.value for m in only.metrics} == expected
    assert only.success and full.success


def test_next_only_records_the_v1_product_type_for_cross_evaluated_layout() -> None:
    case, result = _case_and_result()
    cross = _next_only_result(result, case, True)
    multi = _next_only_result(result, case, "multi_task")
    assert (cross.product_type, multi.product_type) == ("layout_detection", "parse")


def test_next_only_fails_a_document_no_extended_metric_applies_to() -> None:
    case, result = _case_and_result()
    layout_only = LayoutOutput(
        example_id="layout/page",
        pipeline_name="test",
        model=LayoutDetectionModel.YOLO_DOCLAYNET,
        image_width=100,
        image_height=100,
    )
    no_markdown = result.model_copy(update={"output": layout_only})
    scored = _next_only_result(no_markdown, case, True)
    assert not scored.success and scored.metrics == [] and "No extended metric" in (scored.error or "")


def test_worker_honours_next_only() -> None:
    case, result = _case_and_result()
    serial = EvaluationRunner(output_dir=Path("."), next_only=True)._evaluate_single(result, case, None, "multi_task")
    worker = EvaluationResult.model_validate(
        _evaluate_single_worker(
            result.model_dump(), case.model_dump(), "layout_detection", "multi_task", "parse", next_only=True
        )
    )
    assert _comparable(worker) == _comparable(serial)
