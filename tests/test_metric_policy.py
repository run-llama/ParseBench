"""Policy interchange must retain scorer penalties and never mutate paper reports."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from parse_bench.evaluation.metric_policy import PAPER, MetricPolicy, score_metric
from parse_bench.evaluation.metrics.parse.chart_next import score_chart_next
from parse_bench.evaluation.metrics.parse.formatting_v2 import score_formatting
from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms
from parse_bench.evaluation.metrics.parse.table_next import score_table_next
from parse_bench.evaluation.metrics.parse.text_next import score_text_next
from parse_bench.evaluation.rescore import rescore_reports


def policy_file(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("metric", ["text_next", "formatting_v2", "table_next", "chart_next", "forms_v2"])
def test_paper_policy_reproduces_real_scorer_after_json_roundtrip(metric: str, tmp_path: Path) -> None:
    """Catches loss of event severity, systematic counts, or chart omission groups."""
    table = "| Region | Revenue |\n|---|---|\n" + "\n".join(f"| Area {i} | {100 + i} |" for i in range(30))
    if metric == "text_next":
        reference = "The balance is 1234 dollars. " * 25
        result = score_text_next(reference, reference.replace("1234", "4321"))
    elif metric == "formatting_v2":
        reference = "\n\n".join(f"## Section {i}\n\nThe **amount** is {i}." for i in range(10))
        result = score_formatting(reference, reference.replace("**", ""))
    elif metric == "table_next":
        result = score_table_next(table, table.replace("| 10", "| 90"))
    elif metric == "chart_next":
        chart = {
            "title": "Revenue",
            "chart_type": "bar",
            "categories": [f"Area {i}" for i in range(30)],
            "series": [{"name": "Revenue", "values": [str(100 + i) for i in range(30)]}],
        }
        result = score_chart_next(table.replace("| 10", "| 90"), [chart, {**chart, "title": "Other"}])
        assert result is not None
    else:
        reference = reference_from_tree(
            [
                {
                    "page_number": 1,
                    "items": [
                        {"type": "field", "field": "text", "label": f"Field {i}", "value": str(100 + i)}
                        for i in range(15)
                    ],
                }
            ]
        )
        result = score_forms(reference, "\n".join(f"Field {i}: {900 + i}" for i in range(15)))
    metadata = json.loads(json.dumps(result.metadata()))
    policy = MetricPolicy.load(policy_file(tmp_path, PAPER))
    assert score_metric(metric, metadata, policy) == pytest.approx(result.score, abs=1e-12)
    metadata["proportional"] = 1
    cap = min(result.floor, metadata.get("coverage", 1) / 0.5)
    assert score_metric(metric, metadata, policy) == pytest.approx(cap, abs=1e-12)


def test_reclassification_uses_new_class_totals_and_grouped_repeats(tmp_path: Path) -> None:
    data = copy.deepcopy(PAPER)
    data["error_classes"]["fmt"]["all minor errors"] = "critical"
    policy = MetricPolicy.load(policy_file(tmp_path, data))
    meta = {
        "proportional": 1,
        "coverage": 1,
        "atoms": 20,
        "policy_inputs": [
            {"label": "all minor errors", "severity": "minor", "count": 2},
            {"label": "all minor errors", "severity": "minor", "count": 1, "grouped": True},
        ],
    }
    assert score_metric("formatting_v2", meta, policy) == pytest.approx(0.8)
    policy.error_classes["fmt"]["all minor errors"] = "ignore"
    assert score_metric("formatting_v2", meta, policy) == 1


@pytest.mark.parametrize("failure", [None, "Provider returned empty output", "Worker error: scorer crashed"])
def test_cli_rescore_preserves_failure_denominator_and_source(tmp_path: Path, failure: str | None) -> None:
    data = copy.deepcopy(PAPER)
    data["dimension_weights"] = {key: int(key == "text") for key in data["dimension_weights"]}
    data["weights"]["critical"] = 0.2
    policy = policy_file(tmp_path, data)
    metric = score_text_next("The amount is 123 dollars.", "The amount is 456 dollars.")
    report = {
        "per_example_results": [
            {
                "test_id": "text/page",
                "example_id": "page",
                "product_type": "parse",
                "success": True,
                "pipeline_name": "parser",
                "metrics": [{"metric_name": "text_next", "value": metric.score, "metadata": metric.metadata()}],
            }
        ]
    }
    if failure:
        report["per_example_results"].append(
            {
                "test_id": "text/failed",
                "example_id": "failed",
                "pipeline_name": "parser",
                "product_type": "parse",
                "success": False,
                "error": failure,
                "metrics": [],
            }
        )
    source = tmp_path / "_evaluation_report.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    before = source.read_bytes()
    result = rescore_reports(tmp_path, policy)
    denominator = 2 if failure and not failure.startswith("Worker error:") else 1
    assert result["policy_score"] == pytest.approx(min(metric.proportional, 0.8) / denominator)
    assert source.read_bytes() == before
    assert json.loads((tmp_path / "_policy_score.json").read_text(encoding="utf-8"))["policy"] == data


@pytest.mark.parametrize("change", ["nan", "zero", "version", "unknown", "order"])
def test_invalid_policy_is_rejected_before_output(change: str, tmp_path: Path) -> None:
    data = copy.deepcopy(PAPER)
    if change == "nan":
        data["weights"]["major"] = float("nan")
    elif change == "zero":
        data["dimension_weights"] = dict.fromkeys(data["dimension_weights"], 0)
    elif change == "version":
        data["version"] = 2
    elif change == "unknown":
        data["error_classes"]["tables"]["typo in kind"] = "minor"
    else:
        data["error_classes"]["text"]["moved block (reading order)"] = "critical"
    with pytest.raises(ValueError):
        MetricPolicy.load(policy_file(tmp_path, data))


def test_old_truncated_metadata_fails_instead_of_guessing_severity(tmp_path: Path) -> None:
    policy = MetricPolicy.load(policy_file(tmp_path, PAPER))
    with pytest.raises(ValueError, match="rerun evaluation"):
        score_metric("table_next", {"proportional": 1, "events": {"major": 12}}, policy)


def test_reclassified_form_coverage_event_is_retained(tmp_path: Path) -> None:
    """Coverage diagnostics are ignored by default but can be charged by an exported policy."""
    reference = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {"type": "field", "field": "text", "label": f"Field {i}", "isEmpty": True} for i in range(20)
                ],
            }
        ]
    )
    result = score_forms(reference, "\n".join(f"Field {i}: ______" for i in range(9)))
    data = copy.deepcopy(PAPER)
    data["error_classes"]["forms"] = dict.fromkeys(data["error_classes"]["forms"], "ignore")
    data["error_classes"]["forms"]["form fields missing (under half found; coverage cap applies)"] = "critical"
    data["weights"]["critical"] = 0.3
    policy = MetricPolicy.load(policy_file(tmp_path, data))
    assert score_metric("forms_v2", result.metadata(), policy) == pytest.approx(0.7)
