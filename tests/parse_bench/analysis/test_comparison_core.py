"""Tests for analysis.comparison_core helpers."""

from __future__ import annotations

import json
from pathlib import Path

from parse_bench.analysis.comparison_core import (
    compare_pipelines,
    get_predictions_from_inference,
)


def test_get_predictions_from_inference_returns_xyxy_bbox() -> None:
    """Bboxes in ``output.layout_pages`` are xywh pixel; the helper must return
    xyxy so the dashboard overlay (which consumes ``bbox[2]-bbox[0]``) and
    ``comparison.py::_get_predictions`` agree."""
    inference = {
        "output": {
            "layout_pages": [
                {
                    "page_number": 1,
                    "items": [
                        {
                            "type": "Title",
                            "score": 0.9,
                            "bbox": {
                                "x": 10.0,
                                "y": 20.0,
                                "w": 30.0,
                                "h": 40.0,
                                "label": "Title",
                            },
                        }
                    ],
                }
            ]
        }
    }

    predictions = get_predictions_from_inference(inference)
    assert predictions is not None
    assert len(predictions) == 1
    pred = predictions[0]
    # xyxy: [x, y, x + w, y + h] = [10, 20, 40, 60]
    assert pred["bbox"] == [10.0, 20.0, 40.0, 60.0]
    assert pred["class"] == "Title"
    assert pred["score"] == 0.9


def test_get_predictions_from_inference_handles_missing_bbox() -> None:
    """Items without a ``bbox`` are skipped."""
    inference = {
        "output": {
            "layout_pages": [
                {
                    "page_number": 1,
                    "items": [
                        {"type": "Text", "bbox": None},
                        {"type": "Text"},
                    ],
                }
            ]
        }
    }
    assert get_predictions_from_inference(inference) is None


def test_get_predictions_from_inference_returns_none_when_empty() -> None:
    """No layout_pages -> None (not an empty list)."""
    assert get_predictions_from_inference(None) is None
    assert get_predictions_from_inference({}) is None
    assert get_predictions_from_inference({"output": {}}) is None
    assert get_predictions_from_inference({"output": {"layout_pages": []}}) is None


def test_get_predictions_from_inference_falls_back_to_item_type() -> None:
    """When bbox has no ``label``, ``class`` falls back to ``item.type``."""
    inference = {
        "output": {
            "layout_pages": [
                {
                    "page_number": 1,
                    "items": [
                        {
                            "type": "Picture",
                            "bbox": {"x": 0.0, "y": 0.0, "w": 5.0, "h": 5.0},
                        }
                    ],
                }
            ]
        }
    }
    predictions = get_predictions_from_inference(inference)
    assert predictions is not None
    assert predictions[0]["class"] == "Picture"
    assert predictions[0]["bbox"] == [0.0, 0.0, 5.0, 5.0]


def _write_eval_report(
    pipeline_dir: Path,
    pipeline_name: str,
    per_example: list[dict],
) -> None:
    """Write a minimal ``_evaluation_report.json`` to ``pipeline_dir``."""
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "total_examples": len(per_example),
        "successful": len(per_example),
        "failed": 0,
        "skipped": 0,
        "aggregate_metrics": {},
        "per_example_results": [
            {
                "test_id": ex["test_id"],
                "example_id": ex["test_id"],
                "pipeline_name": pipeline_name,
                "product_type": "parse",
                "success": True,
                "metrics": ex["metrics"],
                "error": None,
                "stats": [],
                "tags": [],
            }
            for ex in per_example
        ],
    }
    (pipeline_dir / "_evaluation_report.json").write_text(json.dumps(report))


def test_comparison_metric_label_reflects_picked_metric_not_first_candidate(
    tmp_path: Path,
) -> None:
    """For a layout-only parse run, only ``mAP@[.50:.95]`` is emitted; the
    dashboard label must reflect that instead of the first candidate."""
    path_a = tmp_path / "pipeline_a"
    path_b = tmp_path / "pipeline_b"
    _write_eval_report(
        path_a,
        "pipeline_a",
        [{"test_id": "grp/doc1", "metrics": [{"metric_name": "mAP@[.50:.95]", "value": 0.42}]}],
    )
    _write_eval_report(
        path_b,
        "pipeline_b",
        [{"test_id": "grp/doc1", "metrics": [{"metric_name": "mAP@[.50:.95]", "value": 0.55}]}],
    )

    result = compare_pipelines(path_a, path_b)

    assert result["comparison_metric"] == "mAP@[.50:.95]"
    assert result["stats"]["comparison_metric"] == "mAP@[.50:.95]"
    # And the actual values must come from the picked metric, so category
    # reflects the mAP comparison.
    assert result["matched_results"][0]["category"] == "b_better"


def test_comparison_metric_uses_table_metric_for_table_only_parse_runs(
    tmp_path: Path,
) -> None:
    """Table-only parse reports emit ``grits_trm_composite`` but no
    ``rule_pass_rate``; the dashboard should not classify every row as N/A."""
    path_a = tmp_path / "pipeline_a"
    path_b = tmp_path / "pipeline_b"
    _write_eval_report(
        path_a,
        "pipeline_a",
        [{"test_id": "grp/table1", "metrics": [{"metric_name": "grits_trm_composite", "value": 0.99}]}],
    )
    _write_eval_report(
        path_b,
        "pipeline_b",
        [{"test_id": "grp/table1", "metrics": [{"metric_name": "grits_trm_composite", "value": 0.25}]}],
    )

    result = compare_pipelines(path_a, path_b)

    assert result["comparison_metric"] == "grits_trm_composite"
    assert result["stats"]["comparison_metric"] == "grits_trm_composite"
    assert result["matched_results"][0]["category"] == "a_better"


def test_comparison_metric_prefers_rule_pass_rate_over_normalized_text_score(
    tmp_path: Path,
) -> None:
    """``normalized_text_score`` is a secondary signal; when both are emitted
    the rule-based score decides the comparison."""
    path_a = tmp_path / "pipeline_a"
    path_b = tmp_path / "pipeline_b"
    _write_eval_report(
        path_a,
        "pipeline_a",
        [
            {
                "test_id": "grp/doc1",
                "metrics": [
                    {"metric_name": "normalized_text_score", "value": 0.9},
                    {"metric_name": "rule_pass_rate", "value": 0.2},
                ],
            }
        ],
    )
    _write_eval_report(
        path_b,
        "pipeline_b",
        [
            {
                "test_id": "grp/doc1",
                "metrics": [
                    {"metric_name": "normalized_text_score", "value": 0.1},
                    {"metric_name": "rule_pass_rate", "value": 0.8},
                ],
            }
        ],
    )

    result = compare_pipelines(path_a, path_b)

    assert result["comparison_metric"] == "rule_pass_rate"
    assert result["matched_results"][0]["category"] == "b_better"


def test_comparison_metric_label_prefers_higher_priority_when_mixed(
    tmp_path: Path,
) -> None:
    """When examples emit different candidates (one has ``rule_pass_rate``,
    another only ``mAP@[.50:.95]``), the label picks the highest-priority
    candidate actually seen — matching ``comparison.py::_resolve_comparison_metric_name``."""
    path_a = tmp_path / "pipeline_a"
    path_b = tmp_path / "pipeline_b"
    _write_eval_report(
        path_a,
        "pipeline_a",
        [
            {"test_id": "grp/doc_rules", "metrics": [{"metric_name": "rule_pass_rate", "value": 0.8}]},
            {"test_id": "grp/doc_layout", "metrics": [{"metric_name": "mAP@[.50:.95]", "value": 0.3}]},
        ],
    )
    _write_eval_report(
        path_b,
        "pipeline_b",
        [
            {"test_id": "grp/doc_rules", "metrics": [{"metric_name": "rule_pass_rate", "value": 0.6}]},
            {"test_id": "grp/doc_layout", "metrics": [{"metric_name": "mAP@[.50:.95]", "value": 0.4}]},
        ],
    )

    result = compare_pipelines(path_a, path_b)

    # rule_pass_rate ranks ahead of mAP@[.50:.95] in the parse candidate
    # chain, so it wins even though some examples only emit mAP.
    assert result["comparison_metric"] == "rule_pass_rate"
    assert result["stats"]["comparison_metric"] == "rule_pass_rate"
    # Every example is still compared on whatever it emitted -- no "no data" bucket.
    categories = {r["test_id"]: r["category"] for r in result["matched_results"]}
    assert categories == {"grp/doc_rules": "a_better", "grp/doc_layout": "b_better"}
