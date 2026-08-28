from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest


def _load_checker() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts/check_static_quality_delta.py"
    spec = importlib.util.spec_from_file_location("check_static_quality_delta", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["tool"], returncode, stdout, stderr)


@pytest.mark.parametrize(
    ("function_name", "returncode", "stdout", "stderr", "message"),
    [
        ("_ruff_diagnostics", 2, "", "invalid config", "Ruff check failed"),
        ("_ruff_diagnostics", 0, "[]", "warning", "Ruff check wrote to stderr"),
        ("_ruff_diagnostics", 0, "not json", "", "malformed JSON"),
        ("_ruff_diagnostics", 0, "{}", "", "not a list"),
        ("_mypy_diagnostics", 2, "", "invalid config", "mypy failed"),
        ("_mypy_diagnostics", 0, "", "warning", "mypy wrote to stderr"),
        ("_mypy_diagnostics", 1, "mypy internal surprise", "", "unrecognized output"),
    ],
)
def test_diagnostic_tools_fail_closed(
    function_name: str,
    returncode: int,
    stdout: str,
    stderr: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sample.py").write_text("value = 1\n")
    monkeypatch.setattr(checker, "_run", lambda *args, **kwargs: _result(returncode, stdout, stderr))

    with pytest.raises(checker.StaticQualityError, match=message):
        getattr(checker, function_name)(tmp_path, ["sample.py"], "tool")


def test_ruff_rejects_malformed_diagnostic_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "sample.py").write_text("value = 1\n")
    monkeypatch.setattr(checker, "_run", lambda *args, **kwargs: _result(1, '[{"filename": "sample.py"}]'))

    with pytest.raises(checker.StaticQualityError, match="invalid schema"):
        checker._ruff_diagnostics(tmp_path, ["sample.py"], "ruff")


def test_mypy_parses_errors_and_rejects_exit_output_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "sample.py").write_text("value: str = 1\n")
    output = (
        "sample.py:1: error: Incompatible types in assignment  [assignment]\n"
        'sample.py:1: note: expression has type "int"\n'
        "Found 1 error in 1 file (checked 1 source file)\n"
    )
    monkeypatch.setattr(checker, "_run", lambda *args, **kwargs: _result(1, output))

    diagnostics = checker._mypy_diagnostics(tmp_path, ["sample.py"], "mypy")
    assert [(item.path, item.row, item.code, item.source) for item in diagnostics] == [
        ("sample.py", 1, "assignment", "value: str = 1")
    ]

    monkeypatch.setattr(
        checker, "_run", lambda *args, **kwargs: _result(1, "Success: no issues found in 1 source file\n")
    )
    with pytest.raises(checker.StaticQualityError, match="without parseable errors"):
        checker._mypy_diagnostics(tmp_path, ["sample.py"], "mypy")


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (_result(2, stderr="formatter crashed"), "Ruff formatter failed"),
        (_result(0, stderr="config warning"), "Ruff formatter wrote to stderr"),
        (_result(0, stdout="unexpected formatter output"), "malformed diff output"),
    ],
)
def test_formatter_failures_are_terminal(
    result: subprocess.CompletedProcess[str],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checker, "_run", lambda *args, **kwargs: result)
    with pytest.raises(checker.StaticQualityError, match=message):
        checker._format_hunks(tmp_path, ["sample.py"], "ruff")


def test_formatter_hunks_use_current_tree_line_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formatter_diff = """--- sample.py
+++ sample.py
@@ -2,4 +2,7 @@
-old
+new
"""
    monkeypatch.setattr(checker, "_run", lambda *args, **kwargs: _result(0, formatter_diff))

    assert checker._format_hunks(tmp_path, ["sample.py"], "ruff") == [("sample.py", range(2, 9))]


def test_baseline_comparison_tracks_source_across_line_moves() -> None:
    baseline = checker.Diagnostic("sample.py", "E001", "problem", "bad()", 2)
    moved = checker.Diagnostic("sample.py", "E001", "problem", "bad()", 20)
    changed_source = checker.Diagnostic("sample.py", "E001", "problem", "worse()", 20)

    assert checker._new_diagnostics(Counter([moved]), Counter([baseline])) == Counter()
    assert checker._new_diagnostics(Counter([changed_source]), Counter([baseline])) == Counter([changed_source])
    assert checker._on_changed_lines([moved], {"sample.py": [range(20, 21)]}) == Counter([moved])


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_divergent_baseline_uses_merge_base_for_diff_and_archive(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Tests")
    (repo / "sample.py").write_text("base = 1\n")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")

    _git(repo, "branch", "baseline")
    _git(repo, "checkout", "-q", "baseline")
    (repo / "sample.py").write_text("baseline = 2\n")
    _git(repo, "commit", "-qam", "baseline work")

    _git(repo, "checkout", "-q", "-b", "feature", base)
    (repo / "sample.py").write_text("feature = 3\n")
    _git(repo, "commit", "-qam", "feature work")

    merge_base = checker._merge_base(repo, "baseline")
    assert merge_base == base
    assert checker._changed_python_files(repo, merge_base) == ["sample.py"]

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    checker._extract_baseline(repo, merge_base, extracted)
    assert (extracted / "sample.py").read_text() == "base = 1\n"
    assert checker._added_line_ranges(repo, merge_base, ["sample.py"]) == {"sample.py": [range(1, 2)]}


def test_github_actions_gate_runs_checker_against_fetched_origin_main() -> None:
    workflow = Path(__file__).parents[1] / ".github/workflows/static-quality-delta.yml"
    content = workflow.read_text()

    assert "fetch-depth: 0" in content
    assert "git fetch --no-tags origin main:refs/remotes/origin/main" in content
    assert "scripts/check_static_quality_delta.py --baseline origin/main" in content
