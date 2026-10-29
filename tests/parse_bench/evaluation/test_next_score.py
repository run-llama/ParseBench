"""`evaluation run --next`: per-document wiring, the combined score and the default path left unchanged."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from parse_bench.evaluation.cli import NEXT_SCORE_FILE, EvaluationCLI
from parse_bench.evaluation.metrics.parse.table_next import table_next_metrics
from parse_bench.evaluation.next_score import (
    NEXT_DIMENSIONS,
    NEXT_SUBSETS,
    combine_next_score,
    parse_next_option,
)
from parse_bench.evaluation.runner import EvaluationRunner
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType

TABLE = "<table><tr><th>Region</th><th>2024</th></tr><tr><td>North</td><td>120</td></tr></table>"
WRONG_TABLE = TABLE.replace("120", "910")
PROSE = "Revenue grew in every region during the year, led by the northern sales team."
EXTENDED = ("text_next", "formatting_v2", "table_next", "chart_next", "forms_v2", "form_kv_", "layout_grounding")


def _write_case(root: Path, group: str, stem: str, expected: str, output: str) -> None:
    """One sidecar test case and its saved inference result."""
    cases, results = root / "cases" / group, root / "results" / group
    cases.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (cases / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n")
    (cases / f"{stem}.test.json").write_text(
        json.dumps({"expected_markdown": expected, "test_rules": [{"type": "present", "text": "North"}]})
    )
    now = datetime(2026, 1, 1)
    result = InferenceResult(
        request=InferenceRequest(
            example_id=f"{group}/{stem}", source_file_path=str(cases / f"{stem}.pdf"), product_type=ProductType.PARSE
        ),
        pipeline_name="p",
        product_type=ProductType.PARSE,
        raw_output={},
        output=ParseOutput(
            example_id=f"{group}/{stem}",
            pipeline_name="p",
            markdown=output,
            pages=[PageIR(page_index=0, markdown=output)],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )
    (results / f"{stem}.result.json").write_text(result.model_dump_json())


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _write_case(tmp_path, "table", "right", TABLE, TABLE)
    _write_case(tmp_path, "table", "wrong", TABLE, WRONG_TABLE)
    _write_case(tmp_path, "text_content", "prose", PROSE, PROSE)
    return tmp_path


def _run(root: Path, next_metrics: bool) -> dict[str, dict[str, float]]:
    runner = EvaluationRunner(root / "results", root / "cases", next_metrics=next_metrics)
    summary = runner.run_evaluation(max_workers=1, use_rich=False)
    return {r.test_id: {m.metric_name: m.value for m in r.metrics} for r in summary.per_example_results}


def test_default_run_has_no_extended_metrics_and_next_only_adds_rows(dataset: Path) -> None:
    default, extended = _run(dataset, False), _run(dataset, True)
    assert not any(name.startswith(EXTENDED) for doc in default.values() for name in doc)
    for test_id, metrics in default.items():
        added = {k: v for k, v in extended[test_id].items() if k not in metrics}
        assert {k: extended[test_id][k] for k in metrics} == metrics
        assert added and all(name.startswith(EXTENDED) for name in added)
    assert extended["table/right"]["table_next"] == pytest.approx(1.0)
    assert extended["table/wrong"]["table_next"] < 1.0
    assert extended["text_content/prose"]["text_next"] == pytest.approx(1.0)


def _all_subsets(value: float) -> dict[str, dict[str, float]]:
    metrics = {f"avg_{s.metric}" for slots in NEXT_DIMENSIONS.values() for s in slots}
    return {subset: dict.fromkeys(metrics, value) for subset in NEXT_SUBSETS}


def test_combined_score_weights_each_slot() -> None:
    subsets = _all_subsets(1.0)
    subsets["text"]["avg_formatting_v2"] = 0.2  # 3/4 of formatting
    subsets["strikeout"]["avg_formatting_v2"] = 0.0  # 1/16 of formatting
    subsets["tables"]["avg_table_next"] = 0.6  # 1/2 of tables
    subsets["fillable_forms"]["avg_forms_v2"] = 0.2  # 1/8 of forms
    combined = combine_next_score(subsets)
    dims = {name: d["score"] for name, d in combined["dimensions"].items()}
    assert dims["formatting"] == pytest.approx(0.75 * 0.2 + 0.125 + 0.0625 * 0.0 + 0.0625)
    assert dims["tables"] == pytest.approx(0.5 * 0.6 + 0.5)
    assert dims["forms"] == pytest.approx(0.875 + 0.125 * 0.2)
    assert dims["text"] == dims["charts"] == dims["layout"] == pytest.approx(1.0)
    assert combined["next_score"] == pytest.approx(sum(dims.values()) / 6)
    assert combined["missing_dimensions"] == []


def test_a_dimension_with_an_unscored_subset_is_missing_not_zero() -> None:
    subsets = _all_subsets(0.8)
    del subsets["dense_tables"]
    subsets["layout"] = {}  # evaluated, but nothing scored the layout metric
    combined = combine_next_score(subsets)
    assert combined["dimensions"]["tables"]["score"] is None
    assert combined["dimensions"]["tables"]["missing_subsets"] == ["dense_tables"]
    assert combined["dimensions"]["layout"]["score"] is None
    assert combined["missing_dimensions"] == ["tables", "layout"]
    assert combined["next_score"] is None
    assert combined["dimensions"]["text"]["score"] == pytest.approx(0.8)


def test_next_option_forms(tmp_path: Path) -> None:
    assert parse_next_option(True, tmp_path) == (None, [])
    assert parse_next_option("tables", tmp_path) == ("tables", [])
    _, sources = parse_next_option(f"tables=out/t, charts={tmp_path / 'c'}::cases/c", tmp_path)
    assert [(s.subset, s.output_dir, s.test_cases_dir) for s in sources] == [
        ("tables", tmp_path / "out/t", None),
        ("charts", tmp_path / "c", tmp_path / "cases/c"),
    ]
    _, sources = parse_next_option({"layout": {"output_dir": "l", "test_cases_dir": "d"}}, tmp_path)
    assert (sources[0].output_dir, sources[0].test_cases_dir) == (tmp_path / "l", tmp_path / "d")
    for bad in ("table", "tables=a,tables=b", "tables=", "tables=a,charts"):
        with pytest.raises(ValueError):
            parse_next_option(bad, tmp_path)


def test_cli_scores_each_subset_directory_and_reports_the_rest_missing(dataset: Path) -> None:
    code = EvaluationCLI().run(
        dataset,
        next=f"tables=results/table::{dataset / 'cases' / 'table'},text=results/text_content::cases/text_content",
        export_csv=False,
        export_rule_csv=False,
        export_markdown=False,
        export_html=False,
        max_workers=1,
    )
    assert code == 0
    combined = json.loads((dataset / NEXT_SCORE_FILE).read_text())
    tables = {s["subset"]: s for s in combined["dimensions"]["tables"]["slots"]}
    wrong = table_next_metrics(TABLE, WRONG_TABLE)[0].value
    assert wrong < 1.0
    assert tables["tables"]["score"] == pytest.approx(0.5 * (1.0 + wrong))
    assert tables["tables"]["documents"] == 2
    assert combined["dimensions"]["text"]["score"] == pytest.approx(1.0)
    assert combined["dimensions"]["tables"]["missing_subsets"] == [
        "complex_table_headers",
        "financial_tables",
        "insurance_filing_tables",
        "dense_tables",
    ]
    assert combined["next_score"] is None


def test_layout_document_without_layout_pages_scores_zero_only_with_next(tmp_path: Path) -> None:
    _write_case(tmp_path, "layout", "page", PROSE, PROSE)
    rule = {
        "type": "layout",
        "id": "r1",
        "page": 1,
        "bbox": [0.1, 0.1, 0.5, 0.1],
        "canonical_class": "Text",
        "content": {"type": "text", "text": PROSE},
    }
    (tmp_path / "cases" / "layout" / "page.test.json").write_text(json.dumps({"test_rules": [rule]}))

    def evaluate(next_metrics: bool) -> Any:
        runner = EvaluationRunner(tmp_path / "results", tmp_path / "cases", next_metrics=next_metrics)
        (result,) = runner.run_evaluation(max_workers=1, use_rich=False).per_example_results
        return result

    default = evaluate(False)
    assert not default.success and not default.metrics
    scored = evaluate(True)
    assert scored.success
    (metric,) = [m for m in scored.metrics if m.metric_name == "layout_grounding_pass_rate"]
    assert metric.value == 0.0 and metric.metadata["no_predictions"] is True
