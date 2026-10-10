"""One inference result must evaluate both text rule categories."""

import json
from datetime import datetime
from pathlib import Path

from parse_bench.evaluation.runner import EvaluationRunner
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def test_shared_text_inference_evaluates_both_rule_groups(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground_truth"
    output_dir = tmp_path / "output"
    document = ground_truth / "docs" / "text" / "sample.pdf"
    document.parent.mkdir(parents=True)
    document.write_bytes(b"%PDF-1.4\n%%EOF\n")

    cases = (
        ("text_content", "missing_specific_word", {"word": "ALPHA"}),
        ("text_formatting", "is_bold", {"text": "Bold"}),
    )
    for category, rule_type, rule in cases:
        row = {
            "pdf": "docs/text/sample.pdf",
            "category": category,
            "id": f"sample_{rule_type}_0",
            "type": rule_type,
            "rule": json.dumps(rule),
            "page": None,
            "tags": ["shared-inference"],
        }
        (ground_truth / f"{category}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    result_dir = output_dir / "text"
    result_dir.mkdir(parents=True)
    now = datetime(2026, 1, 1)
    inference = InferenceResult(
        request=InferenceRequest(
            example_id="text/sample",
            source_file_path=str(document),
            product_type=ProductType.PARSE,
        ),
        pipeline_name="test-pipeline",
        product_type=ProductType.PARSE,
        raw_output={},
        output=ParseOutput(example_id="text/sample", pipeline_name="test-pipeline", markdown="ALPHA **Bold**"),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    (result_dir / "sample.result.json").write_text(inference.model_dump_json(), encoding="utf-8")

    summary = EvaluationRunner(output_dir=output_dir, test_cases_dir=ground_truth).run_evaluation(
        product_type="parse", use_rich=False, max_workers=1
    )

    by_id = {result.test_id: result for result in summary.per_example_results}
    assert summary.total_examples == 2
    assert set(by_id) == {"text_content/sample", "text_formatting/sample"}
    assert all(result.success for result in by_id.values())
    assert "text_content" in by_id["text_content/sample"].tags
    assert "text_formatting" in by_id["text_formatting/sample"].tags
    assert summary.aggregate_metrics["total_rule_pass_rate_evaluated"] == 2
