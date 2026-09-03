"""
Lightweight comparison module for evaluating two pipeline results.

This module has NO dependencies on Pydantic or other parse_bench modules,
making it suitable for use in the dashboard deployment where heavy deps aren't installed.
"""

import json
import re
from pathlib import Path
from typing import Any

# Ordered metric-name candidates per product type. The first name present
# in an evaluation result wins. Parse carries a fallback chain because
# layout-only parse runs (test cases with only ``LayoutTestRule`` entries)
# emit table-only metrics such as ``grits_trm_composite`` or layout-only
# metrics such as ``mAP@[.50:.95]`` instead of ``rule_pass_rate``.
#
# MUST stay in sync with ``comparison.py::PipelineComparison.METRIC_CANDIDATES``
# — enforced by ``tests/.../test_comparison_consistency.py``. The canonical
# parse metric is ``rule_pass_rate`` (pass/fail rule semantics from
# ``ParseEvaluator``). ``grits_trm_composite`` is the primary table-only parse
# metric. ``normalized_text_score`` is a secondary text-similarity signal and
# is intentionally NOT in the candidate list — when both are emitted for the
# same run, we pick the definitive rule-based score.
COMPARISON_METRIC_CANDIDATES: dict[str, tuple[str, ...]] = {
    "extract": ("accuracy",),
    "parse": ("rule_pass_rate", "grits_trm_composite", "mAP@[.50:.95]"),
    "layout_detection": ("mAP@[.50:.95]",),
}
_DEFAULT_COMPARISON_METRIC = "accuracy"


def load_evaluation_report(pipeline_path: Path) -> dict | None:
    """Load evaluation report JSON from a pipeline directory."""
    report_file = pipeline_path / "_evaluation_report.json"
    if not report_file.exists():
        return None
    try:
        with open(report_file) as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        return None


def load_inference_result(pipeline_path: Path, test_id: str) -> dict | None:
    """Load inference result for a specific test_id."""
    # Result files are stored as: <group>/<filename>.result.json
    parts = test_id.split("/")
    if len(parts) == 2:
        group, filename = parts
        result_path = pipeline_path / group / f"{filename}.result.json"
    else:
        result_path = pipeline_path / f"{test_id}.result.json"

    if not result_path.exists():
        # Fallback: search recursively
        for result_file in pipeline_path.rglob(f"*{test_id}*.result.json"):
            result_path = result_file
            break
        else:
            return None

    try:
        with open(result_path) as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        return None


def get_metric_value(metrics_list: list, metric_name: str) -> float | None:
    """Extract a specific metric value from a metrics list."""
    for metric in metrics_list:
        if metric.get("metric_name") == metric_name:
            return metric.get("value")  # type: ignore[no-any-return]
    return None


def get_directory_suffix(pipeline_dir: Path) -> str:
    """
    Extract a distinguishing suffix from the pipeline directory path.

    Looks for run IDs, dates, or other identifying info in parent directories.
    """
    parent_name = pipeline_dir.parent.name

    # Try to extract a run ID pattern (e.g., run-21391181794). Matrix-leg
    # dirs append a dataset suffix (run-<id>-<dataset-slug>) — keep it, or
    # two legs of the same parent run would get identical labels.
    run_id_match = re.search(r"run-(\d+(?:-[A-Za-z0-9._-]+)?)", parent_name)
    if run_id_match:
        return f"run-{run_id_match.group(1)}"

    # Try to extract a date pattern (e.g., 2025-01-27)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", parent_name)
    if date_match:
        return date_match.group(1)

    # Fall back to the parent directory name
    if parent_name and parent_name != "output":
        return parent_name

    # Last resort: use the full parent path's last 2 components
    parts = pipeline_dir.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])

    return str(pipeline_dir)


def get_predictions_from_inference(inference: dict | None) -> list[dict] | None:
    """Extract layout predictions from an inference result as list of dicts.

    Reads ``output.layout_pages[*].items``: each item's ``bbox`` is xywh pixel,
    ``label`` (or falling back to ``item.type``) is the class name, and
    ``score`` is detector confidence. Returns bboxes in ``[x1, y1, x2, y2]``
    (xyxy) to match ``comparison.py::_get_predictions`` and the dashboard
    overlay renderer, both of which consume xyxy. Returns ``None`` when no
    layout items are present.
    """
    if not inference:
        return None
    output = inference.get("output")
    if not output:
        return None
    layout_pages = output.get("layout_pages") or []
    predictions: list[dict] = []
    for page in layout_pages:
        for item in page.get("items", []):
            bbox = item.get("bbox")
            if not bbox:
                continue
            x = float(bbox.get("x") or 0)
            y = float(bbox.get("y") or 0)
            w = float(bbox.get("w") or 0)
            h = float(bbox.get("h") or 0)
            predictions.append(
                {
                    "bbox": [x, y, x + w, y + h],
                    "class": bbox.get("label") or item.get("type"),
                    "score": item.get("score"),
                }
            )
    return predictions or None


def compare_pipelines(
    path_a: Path,
    path_b: Path,
    test_cases_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Compare results from two pipeline directories.

    Args:
        path_a: Directory containing pipeline A evaluation results
        path_b: Directory containing pipeline B evaluation results
        test_cases_dir: Optional directory containing test cases (for input file paths)

    Returns:
        Dictionary with comparison data including:
        - matched_results: List of per-example comparisons
        - pipeline_a_only: Results only in pipeline A
        - pipeline_b_only: Results only in pipeline B
        - stats: Summary statistics
        - product_type: The detected product type
        - comparison_metric: The metric used for comparison
    """
    path_a = Path(path_a)
    path_b = Path(path_b)

    # Load evaluation reports
    report_a = load_evaluation_report(path_a)
    report_b = load_evaluation_report(path_b)

    if not report_a or not report_b:
        raise ValueError(
            "Could not load evaluation reports. Make sure both directories contain _evaluation_report.json files."
        )

    # Extract per-example results
    results_a = {r["test_id"]: r for r in report_a.get("per_example_results", [])}
    results_b = {r["test_id"]: r for r in report_b.get("per_example_results", [])}

    # Detect product type from first result
    product_type = "extract"
    if results_a:
        first_result = next(iter(results_a.values()))
        product_type = first_result.get("product_type", "extract").lower()

    metric_candidates = COMPARISON_METRIC_CANDIDATES.get(product_type, (_DEFAULT_COMPARISON_METRIC,))

    def _pick_metric(metrics_list: list) -> float | None:
        """Return the first candidate metric value present in ``metrics_list``."""
        by_name = {m.get("metric_name"): m.get("value") for m in metrics_list}
        for name in metric_candidates:
            if name in by_name:
                return by_name[name]  # type: ignore[no-any-return]
        return None

    # Resolve the label for the comparison metric to whichever candidate was
    # actually emitted across examples. Mirrors
    # ``comparison.py::_resolve_comparison_metric_name``: scan all emitted
    # metric names, pick the highest-priority candidate present, fall back
    # to the first candidate for empty/no-match runs so an empty summary
    # still gets a sane label. Without this, layout-only parse runs mislabel
    # their ``mAP@[.50:.95]`` values as ``rule_pass_rate``.
    emitted_names = {
        m.get("metric_name") for result in (*results_a.values(), *results_b.values()) for m in result.get("metrics", [])
    }
    comparison_metric = next(
        (name for name in metric_candidates if name in emitted_names),
        metric_candidates[0],
    )

    # Compare matched results
    matched_results: list[dict[str, Any]] = []
    pipeline_a_only: list[str] = []
    pipeline_b_only: list[str] = []

    all_test_ids = set(results_a.keys()) | set(results_b.keys())

    for test_id in all_test_ids:
        result_a = results_a.get(test_id)
        result_b = results_b.get(test_id)

        if result_a and result_b:
            # Both have results - compare using the first fallback candidate
            # that is emitted by this example (layout-only parse runs fall
            # through to ``mAP@[.50:.95]``).
            metrics_a = result_a.get("metrics", [])
            metrics_b = result_b.get("metrics", [])

            metric_a = _pick_metric(metrics_a)
            metric_b = _pick_metric(metrics_b)

            # Load inference results for output data
            inference_a = load_inference_result(path_a, test_id)
            inference_b = load_inference_result(path_b, test_id)

            # Extract input file path from inference results
            input_file_a = inference_a.get("request", {}).get("source_file_path") if inference_a else None
            input_file_b = inference_b.get("request", {}).get("source_file_path") if inference_b else None

            comparison: dict[str, Any] = {
                "test_id": test_id,
                "input_file": input_file_a or input_file_b,
                "pipeline_a": {
                    "pipeline_name": result_a.get("pipeline_name", "Pipeline A"),
                    "metric_value": metric_a,
                    "success": result_a.get("success", False),
                    "error": result_a.get("error"),
                    "all_metrics": metrics_a,
                    "all_stats": result_a.get("stats", []),
                },
                "pipeline_b": {
                    "pipeline_name": result_b.get("pipeline_name", "Pipeline B"),
                    "metric_value": metric_b,
                    "success": result_b.get("success", False),
                    "error": result_b.get("error"),
                    "all_metrics": metrics_b,
                    "all_stats": result_b.get("stats", []),
                },
            }

            # Add product-type-specific output data
            if product_type == "layout_detection":
                comparison["pipeline_a"]["predictions"] = get_predictions_from_inference(inference_a)
                comparison["pipeline_b"]["predictions"] = get_predictions_from_inference(inference_b)
                # GT annotations would need test case loading which we skip for now
                comparison["gt_annotations"] = None
            elif product_type == "extract":
                output_a = inference_a.get("output", {}) if inference_a else {}
                output_b = inference_b.get("output", {}) if inference_b else {}
                comparison["pipeline_a"]["output"] = output_a.get("extracted_data")
                comparison["pipeline_b"]["output"] = output_b.get("extracted_data")
            elif product_type == "parse":
                output_a = inference_a.get("output", {}) if inference_a else {}
                output_b = inference_b.get("output", {}) if inference_b else {}
                comparison["pipeline_a"]["output"] = output_a.get("markdown")
                comparison["pipeline_b"]["output"] = output_b.get("markdown")
                # Surface layout predictions (if any) so the dashboard can
                # render the overlay view for layout-bearing parse runs.
                predictions_a = get_predictions_from_inference(inference_a)
                predictions_b = get_predictions_from_inference(inference_b)
                if predictions_a or predictions_b:
                    comparison["pipeline_a"]["predictions"] = predictions_a
                    comparison["pipeline_b"]["predictions"] = predictions_b
                    comparison["gt_annotations"] = None

            # Determine comparison category
            if metric_a is not None and metric_b is not None:
                if metric_a > metric_b:
                    comparison["category"] = "a_better"
                elif metric_b > metric_a:
                    comparison["category"] = "b_better"
                else:
                    comparison["category"] = "tie"
            elif metric_a is None and metric_b is None:
                comparison["category"] = "both_bad"
            elif metric_a is None:
                comparison["category"] = "b_better"
            else:
                comparison["category"] = "a_better"

            matched_results.append(comparison)
        elif result_a:
            pipeline_a_only.append(test_id)
        elif result_b:
            pipeline_b_only.append(test_id)

    # Get pipeline names from results
    pipeline_a_name = "Pipeline A"
    pipeline_b_name = "Pipeline B"
    if results_a:
        first_a = next(iter(results_a.values()))
        pipeline_a_name = first_a.get("pipeline_name", path_a.name)
    if results_b:
        first_b = next(iter(results_b.values()))
        pipeline_b_name = first_b.get("pipeline_name", path_b.name)

    # Disambiguate if same name
    if pipeline_a_name == pipeline_b_name:
        suffix_a = get_directory_suffix(path_a)
        suffix_b = get_directory_suffix(path_b)

        if suffix_a != suffix_b:
            pipeline_a_name = f"{pipeline_a_name} ({suffix_a})"
            pipeline_b_name = f"{pipeline_b_name} ({suffix_b})"
        else:
            pipeline_a_name = f"{pipeline_a_name} (A)"
            pipeline_b_name = f"{pipeline_b_name} (B)"

    # Calculate statistics
    stats = {
        "total_matched": len(matched_results),
        "a_better": sum(1 for r in matched_results if r["category"] == "a_better"),
        "b_better": sum(1 for r in matched_results if r["category"] == "b_better"),
        "tie": sum(1 for r in matched_results if r["category"] == "tie"),
        "both_bad": sum(1 for r in matched_results if r["category"] == "both_bad"),
        "pipeline_a_only": len(pipeline_a_only),
        "pipeline_b_only": len(pipeline_b_only),
        "pipeline_a_name": pipeline_a_name,
        "pipeline_b_name": pipeline_b_name,
        "product_type": product_type,
        "comparison_metric": comparison_metric,
    }

    return {
        "matched_results": matched_results,
        "pipeline_a_only": pipeline_a_only,
        "pipeline_b_only": pipeline_b_only,
        "stats": stats,
        "product_type": product_type,
        "comparison_metric": comparison_metric,
        "original_base_path": str(test_cases_dir) if test_cases_dir else "",
    }
