"""Missing outputs use the same aggregation scope as successful layout evaluations."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from parse_bench.evaluation import runner as runner_mod
from parse_bench.evaluation.runner import EvaluationRunner
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


@pytest.mark.parametrize("next_only", [False, True])
@pytest.mark.parametrize(
    ("mixed", "multi_task", "product_filter", "expected_product"),
    [
        (False, True, None, "layout_detection"),
        (True, True, None, "parse"),
        (True, False, None, "layout_detection"),
        (True, True, "layout_detection", "layout_detection"),
        (False, True, "parse", "layout_detection"),
        (True, True, "parse", "parse"),
    ],
)
def test_missing_layout_output_counts_zero_in_extended_average(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    next_only: bool,
    mixed: bool,
    multi_task: bool,
    product_filter: str | None,
    expected_product: str,
) -> None:
    text = "Quarterly revenue rose in every region."
    rules = [
        {
            "type": "layout",
            "id": "r1",
            "page": 1,
            "bbox": [0.1, 0.1, 0.5, 0.1],
            "canonical_class": "Text",
            "content": {"type": "text", "text": text},
        }
    ]
    if mixed:
        rules.append({"type": "present", "text": "Quarterly revenue"})
    cases = tmp_path / "cases" / "layout"
    results = tmp_path / "results" / "test-pipe" / "layout"
    cases.mkdir(parents=True)
    results.mkdir(parents=True)
    for name in ("good", "missing"):
        (cases / f"{name}.pdf").write_bytes(b"%PDF-1.4\n")
        (cases / f"{name}.test.json").write_text(json.dumps({"test_rules": rules}), encoding="utf-8")

    segment = LayoutSegmentIR(
        x=0.1, y=0.1, w=0.5, h=0.1, label="text", confidence=1.0, start_index=0, end_index=len(text) - 1
    )
    output = ParseOutput(
        example_id="layout/good",
        pipeline_name="test-pipe",
        markdown=text,
        pages=[PageIR(page_index=0, markdown=text)],
        layout_pages=[
            ParseLayoutPageIR(
                page_number=1,
                width=1,
                height=1,
                items=[LayoutItemIR(type="Text", value=text, md=text, bbox=segment, layout_segments=[segment])],
            )
        ],
    )
    now = datetime.now(UTC)
    inference = InferenceResult(
        request=InferenceRequest(
            example_id="layout/good", source_file_path=str(cases / "good.pdf"), product_type=ProductType.PARSE
        ),
        pipeline_name="test-pipe",
        product_type=ProductType.PARSE,
        raw_output={},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    (results / "good.result.json").write_text(inference.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(runner_mod, "ProcessPoolExecutor", ThreadPoolExecutor)
    summary = EvaluationRunner(
        tmp_path / "results", tmp_path / "cases", multi_task=multi_task, next_metrics=True, next_only=next_only
    ).run_evaluation(product_type=product_filter, max_workers=1, use_rich=False)

    by_id = {result.test_id: result for result in summary.per_example_results}
    assert summary.successful == summary.failed == 1
    assert summary.total_examples == 2
    assert by_id["layout/good"].success
    assert by_id["layout/good"].product_type == by_id["layout/missing"].product_type == expected_product
    assert summary.aggregate_metrics["avg_layout_grounding_pass_rate"] == pytest.approx(0.5)
