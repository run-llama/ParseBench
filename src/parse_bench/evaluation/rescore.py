"""Rescore saved evaluation reports without inference or changes to paper scores."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from parse_bench.evaluation.metric_policy import METRIC_KEYS, MetricPolicy, score_metric
from parse_bench.evaluation.next_score import (
    QUESTION_MARK,
    combine_next_score,
    group_subset,
    parse_next_option,
)
from parse_bench.evaluation.runner import EvaluationRunner
from parse_bench.schemas.evaluation import EvaluationResult

POLICY_SCORE_FILE = "_policy_score.json"
EVALUATION_REPORT_FILE = "_evaluation_report.json"


def rescore_reports(
    output_dir: str | Path,
    policy: str | Path,
    next: bool | str | list[str] | tuple[str, ...] | dict[str, Any] = True,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Apply a Metric Lab policy to one report or --next subset-directory mapping.

    Requires evaluation reports from ``evaluation run --next --next_only``. Reports
    predating sufficient event counts must be regenerated from saved inference outputs.
    All scores in the output are 0–1, including the weighted overall score.
    """
    chosen = MetricPolicy.load(policy)
    base = Path(output_dir)
    subset, sources = parse_next_option(next, base)
    paths: list[tuple[str | None, Path]] = [(s.subset, s.output_dir) for s in sources] if sources else [(subset, base)]
    results_by_subset: dict[str, list[EvaluationResult]] = defaultdict(list)
    documents: dict[str, int] = defaultdict(int)
    pages: list[dict[str, Any]] = []
    pipelines: set[str] = set()
    for explicit_subset, directory in paths:
        path = directory / EVALUATION_REPORT_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        for result in data["per_example_results"]:
            test_id = result["test_id"]
            if QUESTION_MARK in test_id:
                continue
            name = explicit_subset or group_subset(test_id)
            if name is None:
                raise ValueError(f"{test_id}: unknown subset; specify --next=SUBSET or a subset-directory mapping")
            pipelines.add(result["pipeline_name"])
            if len(pipelines) > 1:
                raise ValueError("rescore expects one pipeline; use separate reports for different pipelines")
            scored = EvaluationResult.model_validate(result)
            results_by_subset[name].append(scored)
            documents[name] += 1
            scores: dict[str, float] = {}
            for metric in result["metrics"] if scored.success else []:
                key = metric["metric_name"]
                if key in METRIC_KEYS:
                    try:
                        score = score_metric(key, metric["metadata"], chosen)
                    except (ValueError, KeyError) as exc:
                        raise ValueError(f"{path}: {test_id}: {exc}") from exc
                elif key == "layout_grounding_pass_rate":
                    score = metric["value"]
                else:
                    continue
                if not 0 <= score <= 1:
                    raise ValueError(f"{test_id}: {key}: invalid score {score}")
                scores[key] = score
                for value in scored.metrics:
                    if value.metric_name == key:
                        value.value = score
            if scores:
                pages.append({"test_id": test_id, "subset": name, "metrics": scores})
    if not pages:
        raise ValueError("No extended scores found; run evaluation with --next --next_only first")
    # Reuse evaluation aggregation so provider failures still contribute zero and
    # harness failures/skips retain the same denominator treatment as paper scores.
    means = {name: EvaluationRunner.aggregate_metrics(results) for name, results in results_by_subset.items()}
    combined = combine_next_score(means, documents)
    weights = {("formatting" if name == "fmt" else name): weight for name, weight in chosen.dimension_weights.items()}
    missing = [name for name, dim in combined["dimensions"].items() if weights[name] > 0 and dim["score"] is None]
    overall = (
        None
        if missing
        else sum(weights[name] * dim["score"] for name, dim in combined["dimensions"].items() if weights[name] > 0)
        / sum(weights.values())
    )
    report = {
        "policy": chosen.model_dump(),
        "pipeline": sorted(pipelines)[0],
        "policy_score": overall,
        "missing_dimensions": missing,
        "dimensions": combined["dimensions"],
        "per_example_results": pages,
    }
    destination = (Path(report_dir) if report_dir else base) / POLICY_SCORE_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
