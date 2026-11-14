"""Test helper: the extended metrics of one document, as ``evaluation run --next`` appends them."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from parse_bench.evaluation.metrics.parse.next_metrics import compute_next_metrics


class NextScorers:
    """Evaluator stand-in whose ``evaluate`` returns the extended metrics only."""

    def __init__(self, **_options: Any) -> None:
        pass

    def evaluate(self, inference_result: Any, test_case: Any) -> SimpleNamespace:
        return SimpleNamespace(metrics=compute_next_metrics(inference_result, test_case))
