"""Documents with mixed layout and parse rules ("multi_task") score the same in a pool worker as in the main process."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from parse_bench.evaluation.runner import EvaluationRunner, _evaluate_single_worker
from parse_bench.schemas.evaluation import EvaluationResult
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import LayoutDetectionTestCase


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
