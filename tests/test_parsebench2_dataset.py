"""``--dataset parsebench2``: the ParseBench 2 subsets, their readiness check and the per-subset run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from parse_bench.data.download import (
    PARSEBENCH2_DATA_DIR,
    PARSEBENCH2_SUBSETS,
    dataset_spec,
    default_data_dir,
    is_dataset_ready,
)
from parse_bench.evaluation.next_score import NEXT_SUBSETS
from parse_bench.pipeline.cli import PipelineCLI


def _write_cases(root: Path, subsets: dict[str, str]) -> None:
    for i, folder in enumerate(subsets.values()):
        case_dir = root / folder / "nested"
        case_dir.mkdir(parents=True)
        (case_dir / f"case{i}.test.json").write_text("{}")


def test_subsets_are_the_next_subsets() -> None:
    assert tuple(PARSEBENCH2_SUBSETS) == NEXT_SUBSETS


def test_default_dir_is_outside_v1_data_dir() -> None:
    # Under ./data, its sidecar test.json files would make the v1 loader read ./data as a sidecar dataset.
    assert default_data_dir(dataset="parsebench2") == PARSEBENCH2_DATA_DIR
    assert Path("./data") not in PARSEBENCH2_DATA_DIR.parents


def test_ready_needs_a_test_json_in_every_subset(tmp_path: Path) -> None:
    assert not is_dataset_ready(tmp_path, "parsebench2")
    _write_cases(tmp_path, dict(list(PARSEBENCH2_SUBSETS.items())[:-1]))
    assert not is_dataset_ready(tmp_path, "parsebench2")
    _write_cases(tmp_path, dict(list(PARSEBENCH2_SUBSETS.items())[-1:]))
    assert is_dataset_ready(tmp_path, "parsebench2")


def test_run_scores_every_subset_with_results(tmp_path: Path) -> None:
    """``--group`` limits inference; scoring covers each subset that has results, including earlier runs."""
    data_dir, output = tmp_path / "data", tmp_path / "output"
    _write_cases(data_dir, PARSEBENCH2_SUBSETS)
    earlier = output / "parsebench2" / "tables" / "dummy"
    earlier.mkdir(parents=True)
    (earlier / "a.result.json").write_text("{}")

    def fake_inference(**kwargs: object) -> int:
        results = Path(str(kwargs["output_dir"])) / "dummy"
        results.mkdir(parents=True)
        (results / "b.result.json").write_text("{}")
        return 0

    cli = PipelineCLI()
    with (
        patch("parse_bench.pipeline.cli.InferenceCLI") as mock_inf_cls,
        patch("parse_bench.pipeline.cli.EvaluationCLI") as mock_eval_cls,
    ):
        mock_inf_cls.return_value.run.side_effect = fake_inference
        mock_eval_cls.return_value.run.return_value = 0
        rc = cli.run(
            pipeline="dummy",
            dataset="parsebench2",
            input_dir=data_dir,
            output_dir=output,
            group="charts",
            open_report=False,
        )

    assert rc == 0
    inf_kwargs = mock_inf_cls.return_value.run.call_args.kwargs
    assert inf_kwargs["input_dir"] == data_dir / PARSEBENCH2_SUBSETS["charts"]
    assert inf_kwargs["output_dir"] == output / "parsebench2" / "charts"
    next_arg = mock_eval_cls.return_value.run.call_args.kwargs["next"]
    assert set(next_arg) == {"charts", "tables"}
    assert next_arg["tables"]["test_cases_dir"] == str((data_dir / PARSEBENCH2_SUBSETS["tables"]).resolve())


def test_unknown_subset_and_test_flag_are_rejected() -> None:
    cli = PipelineCLI()
    assert cli.run(pipeline="dummy", dataset="parsebench2", group="nope") == 1
    assert cli.run(pipeline="dummy", dataset="parsebench2", test=True) == 1
    assert dataset_spec("parsebench2").subsets
