"""A detailed-HTML rendering failure must never fail the whole evaluation command."""

from __future__ import annotations

from pathlib import Path

import pytest

from parse_bench.evaluation import cli as cli_mod
from parse_bench.schemas.evaluation import EvaluationResult, EvaluationSummary, MetricValue


def _write_summary(evaluation_dir: Path) -> None:
    summary = EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        aggregate_metrics={"avg_rule_pass_rate": 0.5},
        per_example_results=[
            EvaluationResult(
                test_id="g/doc",
                example_id="g/doc",
                pipeline_name="p",
                product_type="parse",
                success=True,
                metrics=[MetricValue(metric_name="rule_pass_rate", value=0.5)],
            )
        ],
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "_evaluation_report.json").write_text(summary.model_dump_json(indent=2))


def test_regenerate_report_survives_detailed_html_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_summary(tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> Path:
        raise RecursionError("markdown2 blew up on pathological output")

    monkeypatch.setattr(cli_mod, "generate_detailed_html_report", _boom)

    exit_code = cli_mod.EvaluationCLI().regenerate_report(
        evaluation_dir=tmp_path,
        export_csv=False,
        export_rule_csv=False,
        export_markdown=False,
        export_html=True,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Skipped detailed HTML report (non-fatal)" in captured.err
    # The summary HTML report was still written.
    assert any(p.suffix == ".html" for p in tmp_path.iterdir())
