"""``--dataset org/name`` routing and ``parse-bench repetition make`` on a user's own PDFs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from parse_bench.data.cli import DataCLI
from parse_bench.data.download import download_dataset, is_dataset_ready
from parse_bench.data.repetition import make_tests
from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.test_cases.loader import load_test_cases

pymupdf = pytest.importorskip("pymupdf")


def test_download_dataset_repo_id_fetches_that_repo_into_its_own_dir() -> None:
    """Bug caught: `--dataset org/name` downloading ParseBench, or into ./data where it would mask the main dataset."""
    with (
        patch("huggingface_hub.snapshot_download") as snap,
        patch("parse_bench.data.download.is_dataset_ready", side_effect=[False, True]),
    ):
        DataCLI().download(dataset="acme/MyBench")
    kwargs = snap.call_args.kwargs
    assert kwargs["repo_id"] == "acme/MyBench"
    assert Path(kwargs["local_dir"]) == Path("./data/mybench")
    assert kwargs["revision"] is None


def test_repo_id_has_no_test_subset_and_needs_rows_and_docs(tmp_path: Path) -> None:
    """Bug caught: `--test --dataset org/name` silently downloading the ParseBench test branch, a half-downloaded
    copy (rows without docs) passing the readiness check, or a name that is not a repo id taken for one."""
    with pytest.raises(ValueError, match="no test subset"):
        download_dataset(data_dir=tmp_path, test=True, dataset="acme/MyBench")
    with pytest.raises(ValueError, match="Unknown dataset"):
        is_dataset_ready(tmp_path, "mybench")
    (tmp_path / "rows.jsonl").write_text("")
    assert not is_dataset_ready(tmp_path, "acme/MyBench")
    (tmp_path / "docs" / "rows").mkdir(parents=True)
    (tmp_path / "docs" / "rows" / "a.pdf").write_bytes(b"%PDF")
    assert is_dataset_ready(tmp_path, "acme/MyBench")


def _pdf(path: Path, lines: list[str] | None) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    if lines:
        for i, line in enumerate(lines):
            page.insert_text((40, 60 + 16 * i), line, fontsize=9)
    else:  # image only: a scan
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
        pix.clear_with(200)
        page.insert_image(page.rect, pixmap=pix)
    doc.save(path)


def test_make_tests_budgets_a_users_pdf_from_its_text_layer(tmp_path: Path) -> None:
    """Bug caught: a generated test that fails the document's own text (budgets read wrong from the text layer), or
    that lets a runaway leader through; a scan tested by default although it has nothing to budget from."""
    _pdf(tmp_path / "toc.pdf", ["Contents", "Methods " + "." * 60 + " 4", "Results " + "." * 60 + " 9"])
    _pdf(tmp_path / "scan.pdf", None)
    done = make_tests(tmp_path)
    assert [Path(p).name for p in done["written"]] == ["toc.pdf"]
    assert [Path(p).name for p in done["skipped"]] == ["scan.pdf"]
    case = next(c for c in load_test_cases(tmp_path) if c.test_id.endswith("/toc"))
    rule = create_test_rule(json.loads((tmp_path / "toc.test.json").read_text())["test_rules"][0])
    assert case.test_rules[0].type == "repetition_overshoot"
    assert rule.run("Contents\nMethods ........ 4\nResults ........ 9")[0]
    assert not rule.run("Contents\nMethods " + "." * 5000)[0]
    assert [Path(p).name for p in make_tests(tmp_path, scans=True)["written"]] == ["scan.pdf"]
