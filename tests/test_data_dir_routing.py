"""Regression tests for the ``--test`` flag's data-directory routing.

Background: ``parse-bench run <pipeline> --test`` and ``parse-bench download
--test`` previously read/wrote the same ``./data`` location as the full
dataset, so when ``./data`` already had the full dataset present, ``--test``
was silently ignored and the runner processed all 2,078 examples instead of
the advertised 3-files-per-category subset.

Fix: route ``--test`` to ``./data/test`` by default. Both datasets now coexist.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from parse_bench.data.cli import DataCLI
from parse_bench.data.download import (
    DEFAULT_DATA_DIR,
    DEFAULT_TEST_DATA_DIR,
    default_data_dir,
)


class TestDefaultDataDir:
    def test_full_dataset_default(self) -> None:
        assert default_data_dir(False) == Path("./data")
        assert default_data_dir(False) == DEFAULT_DATA_DIR

    def test_test_dataset_default(self) -> None:
        assert default_data_dir(True) == Path("./data/test")
        assert default_data_dir(True) == DEFAULT_TEST_DATA_DIR

    def test_full_and_test_paths_diverge(self) -> None:
        # The whole point of the fix: the two paths cannot collide.
        assert default_data_dir(False) != default_data_dir(True)


class TestDownloadRouting:
    def test_download_test_routes_to_test_subdir(self) -> None:
        cli = DataCLI()
        with patch("parse_bench.data.cli.download_dataset") as mock_dl:
            cli.download(test=True)
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["data_dir"] == DEFAULT_TEST_DATA_DIR
        assert kwargs["test"] is True

    def test_download_default_routes_to_full_dir(self) -> None:
        cli = DataCLI()
        with patch("parse_bench.data.cli.download_dataset") as mock_dl:
            cli.download()
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["data_dir"] == DEFAULT_DATA_DIR
        assert kwargs["test"] is False

    def test_explicit_data_dir_is_respected(self, tmp_path: Path) -> None:
        # Explicit --data_dir overrides the --test default.
        cli = DataCLI()
        explicit = tmp_path / "elsewhere"
        with patch("parse_bench.data.cli.download_dataset") as mock_dl:
            cli.download(data_dir=str(explicit), test=True)
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["data_dir"] == explicit


class TestStatusRouting:
    def test_status_test_flag_checks_test_subdir(self, tmp_path: Path) -> None:
        # Use a clean cwd so the default ./data/test resolves under tmp_path.
        cli = DataCLI()
        with patch("parse_bench.data.cli.is_dataset_ready", return_value=False) as mock_ready, \
             patch("parse_bench.data.cli.Path.cwd", return_value=tmp_path):
            rc = cli.status(test=True)
        # Status returns 1 when not ready; we only care about which path it checked.
        assert rc == 1
        checked_path = mock_ready.call_args.args[0]
        assert checked_path == tmp_path / DEFAULT_TEST_DATA_DIR


@pytest.mark.parametrize(
    "test_flag,expected_relative",
    [(False, Path("./data")), (True, Path("./data/test"))],
)
def test_pipeline_run_input_dir_routing(test_flag: bool, expected_relative: Path) -> None:
    """The pipeline runner must default ``input_dir`` based on ``--test``.

    We mock the heavy machinery (download, inference, evaluation, analysis)
    and only inspect the ``input_dir`` that gets propagated to
    ``InferenceCLI.run`` — that is the one parameter the bug was dropping.
    """
    from parse_bench.pipeline.cli import PipelineCLI

    cli = PipelineCLI()
    with patch("parse_bench.pipeline.cli.is_dataset_ready", return_value=True), \
         patch("parse_bench.pipeline.cli.InferenceCLI") as mock_inf_cls, \
         patch.object(cli, "_run_multi_group_evaluation", return_value=0):
        mock_inf = mock_inf_cls.return_value
        mock_inf.run.return_value = 0
        rc = cli.run(pipeline="dummy", test=test_flag)

    assert rc == 0
    # InferenceCLI.run receives the resolved input_dir; assert routing works.
    inf_kwargs = mock_inf.run.call_args.kwargs
    assert inf_kwargs["input_dir"] == expected_relative


@pytest.mark.parametrize(
    "test_flag,expected_relative",
    [(False, Path("./data")), (True, Path("./data/test"))],
)
def test_pipeline_run_auto_download_routing(test_flag: bool, expected_relative: Path) -> None:
    """When the dataset isn't on disk, ``pipeline.run`` must auto-download to
    the test-routed path — not silently to ``./data``.

    This is the second half of the original bug: even after fixing the
    ``input_dir`` default, a wrong download target would re-introduce the
    overlay/masking problem on a fresh machine.
    """
    from parse_bench.pipeline.cli import PipelineCLI

    cli = PipelineCLI()
    with patch("parse_bench.pipeline.cli.is_dataset_ready", return_value=False), \
         patch("parse_bench.pipeline.cli.download_dataset") as mock_dl, \
         patch("parse_bench.pipeline.cli.InferenceCLI") as mock_inf_cls, \
         patch.object(cli, "_run_multi_group_evaluation", return_value=0):
        mock_inf = mock_inf_cls.return_value
        mock_inf.run.return_value = 0
        rc = cli.run(pipeline="dummy", test=test_flag)

    assert rc == 0
    mock_dl.assert_called_once()
    dl_kwargs = mock_dl.call_args.kwargs
    assert dl_kwargs["data_dir"] == expected_relative
    assert dl_kwargs["test"] is test_flag
