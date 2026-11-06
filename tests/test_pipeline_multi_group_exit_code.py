"""Exit-code tests for ``parse-bench run`` with several evaluation categories.

``_run_multi_group_evaluation`` evaluates each category in turn. A failed or
interrupted category evaluation must show up in the exit code: a run whose
evaluation failed returns non-zero, and Ctrl-C (``EvaluationCLI.run`` returns
130) stops the run instead of moving on to the next category.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from parse_bench.pipeline.cli import PipelineCLI

_GROUPS = ["chart", "layout", "table"]


def _run(tmp_path: Path, eval_exit_codes: list[int]) -> tuple[int, MagicMock]:
    cli = PipelineCLI()
    with (
        patch("parse_bench.pipeline.cli._discover_groups", return_value=_GROUPS),
        patch("parse_bench.pipeline.cli.EvaluationCLI") as mock_eval_cls,
        patch("parse_bench.pipeline.cli.AnalysisCLI") as mock_analysis_cls,
        patch(
            "parse_bench.pipeline.cli.generate_aggregation_report",
            return_value=tmp_path / "_evaluation_report_dashboard.html",
        ),
        patch(
            "parse_bench.pipeline.cli.generate_leaderboard_report",
            return_value=tmp_path / "leaderboard.html",
        ),
    ):
        mock_eval = mock_eval_cls.return_value
        mock_eval.run.side_effect = eval_exit_codes
        mock_analysis_cls.return_value.generate_report.return_value = 0
        rc = cli._run_multi_group_evaluation(
            pipeline_output_dir=tmp_path / "dummy",
            input_path=tmp_path / "data",
            pipeline_name="dummy",
            verbose=False,
            force=False,
            open_report=False,
        )
    return rc, mock_eval


def test_all_groups_succeed_returns_zero(tmp_path: Path) -> None:
    rc, mock_eval = _run(tmp_path, [0, 0, 0])

    assert rc == 0
    assert mock_eval.run.call_count == len(_GROUPS)


@pytest.mark.parametrize("failing_index", [0, 1, 2])
def test_failed_group_evaluation_returns_non_zero(tmp_path: Path, failing_index: int) -> None:
    codes = [0, 0, 0]
    codes[failing_index] = 1
    rc, mock_eval = _run(tmp_path, codes)

    assert rc == 1
    # The remaining categories are still evaluated.
    assert mock_eval.run.call_count == len(_GROUPS)


def test_interrupted_group_evaluation_stops_the_run(tmp_path: Path) -> None:
    rc, mock_eval = _run(tmp_path, [0, 130, 0])

    assert rc == 130
    assert [c.kwargs["group"] for c in mock_eval.run.call_args_list] == ["chart", "layout"]
