"""Detailed HTML report: the embedded viewer script."""

from __future__ import annotations

import re
from pathlib import Path

from parse_bench.analysis.detailed_report import generate_detailed_html_report
from parse_bench.schemas.evaluation import EvaluationResult, EvaluationSummary, MetricValue


def _summary() -> EvaluationSummary:
    return EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        aggregate_metrics={"avg_rule_pass_rate": 0.5, "min_rule_pass_rate": 0.5, "max_rule_pass_rate": 0.5},
        per_example_results=[
            EvaluationResult(
                test_id="group/doc with spaces",
                example_id="group/doc with spaces",
                pipeline_name="p",
                product_type="parse",
                success=True,
                metrics=[
                    MetricValue(
                        metric_name="rule_pass_rate",
                        value=0.5,
                        metadata={
                            "passed": 1,
                            "total": 2,
                            "rule_results": [
                                {"type": "text_presence", "id": "r1", "passed": True, "message": "ok"},
                                {"type": "text_presence", "id": "r2", "passed": False, "message": "missing"},
                            ],
                        },
                    )
                ],
            )
        ],
    )


def _render(tmp_path: Path, **kwargs: object) -> str:
    report_dir = tmp_path / "report"
    report_dir.mkdir(exist_ok=True)
    path = generate_detailed_html_report(_summary(), report_dir, test_cases_dir=tmp_path / "cases", **kwargs)  # type: ignore[arg-type]
    return path.read_text(encoding="utf-8")


def test_pdf_url_builder_encodes_root_relative_bases(tmp_path: Path) -> None:
    """Only ``http(s)://`` bases used to be percent-encoded; a root-relative
    ``/files/...`` base with spaces or unicode in the test id produced a broken
    PDF URL. Both forms now go through ``buildPdfDocumentUrl``."""
    html = _render(tmp_path, pdf_base_url="/files/data")

    assert "function buildPdfDocumentUrl(baseUrl, testId)" in html
    builder = html[html.index("function buildPdfDocumentUrl") :]
    builder = builder[: builder.index("window.loadPdf")]
    assert re.search(r"/\^https\?:\\/\\//i\.test\(baseUrl\) \|\| baseUrl\.charAt\(0\) === '/'", builder)
    assert "encodeURIComponent" in builder
    # loadPdf delegates instead of carrying its own copy of the logic.
    load_pdf = html[html.index("window.loadPdf = function") :]
    load_pdf = load_pdf[: load_pdf.index("function doLoadPdf")]
    assert "buildPdfDocumentUrl(baseUrl, testId)" in load_pdf
    assert "encodeURIComponent" not in load_pdf


def test_rule_results_panel_reports_failed_count_and_auto_expands(tmp_path: Path) -> None:
    """The panel used to render collapsed with a bare rule count; readers open
    an example to see the failures, so it now says ``N failed / M rules`` and
    starts expanded whenever anything failed."""
    html = _render(tmp_path)

    assert "failedRules" in html
    assert "' failed</span> / ' + totalRules + ' rules'" in html
    assert "var ruleOpenClass = failedRules > 0 ? ' open' : '';" in html
    assert "'<div class=\"detail-collapsible-body' + ruleOpenClass + '\">'" in html
    # Both the boolean and the tri-state shapes count as a failure.
    assert "metricRules[fri].status === 'fail' || metricRules[fri].passed === false" in html
