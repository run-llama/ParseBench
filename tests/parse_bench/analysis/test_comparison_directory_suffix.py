"""Tests for comparison-report distinguishing labels on suffixed run dirs.

CI composes matrix-leg results dirs as ``run-<gh_run_id>-<dataset-slug>``.
The directory-suffix label must keep the dataset discriminator — two legs of
the same parent run would otherwise get identical "distinguishing" labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parse_bench.analysis.comparison import PipelineComparison
from parse_bench.analysis.comparison_core import get_directory_suffix

CASES = [
    # Plain run dir — unchanged
    ("output/run-21391181794/llamaparse_agentic", "run-21391181794"),
    # Embedded run id with surrounding text — unchanged
    ("output/financial_tables_run-21391181794/llamaparse_agentic", "run-21391181794"),
    # Matrix-leg run dir — must keep the dataset discriminator
    (
        "results/2026-06-13/run-27055588807-tables_extended-v0.8/llamaparse_agentic",
        "run-27055588807-tables_extended-v0.8",
    ),
    (
        "results/2026-06-13/run-27055588807-charts_extended-v1.0/llamaparse_agentic",
        "run-27055588807-charts_extended-v1.0",
    ),
    # Date fallback — unchanged
    ("output/2025-01-27/llamaparse_agentic", "2025-01-27"),
    # Parent-name fallback — unchanged
    ("output/experiment_v2/llamaparse_agentic", "experiment_v2"),
]


@pytest.mark.parametrize(("pipeline_dir", "expected"), CASES)
def test_core_get_directory_suffix(pipeline_dir: str, expected: str) -> None:
    assert get_directory_suffix(Path(pipeline_dir)) == expected


@pytest.mark.parametrize(("pipeline_dir", "expected"), CASES)
def test_report_get_directory_suffix(pipeline_dir: str, expected: str) -> None:
    comparison = PipelineComparison.__new__(PipelineComparison)  # suffix helper needs no init state
    assert comparison._get_directory_suffix(Path(pipeline_dir)) == expected


def test_sibling_matrix_legs_get_distinct_labels() -> None:
    a = Path("results/2026-06-13/run-27055588807-tables_extended-v0.8/llamaparse_agentic")
    b = Path("results/2026-06-13/run-27055588807-charts_extended-v1.0/llamaparse_agentic")
    assert get_directory_suffix(a) != get_directory_suffix(b)
