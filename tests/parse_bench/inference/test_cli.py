"""Tests for the inference CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

import parse_bench.inference.cli as inference_cli_module
from parse_bench.inference.cli import InferenceCLI, _print_unrecognized_extension_hint
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.product import ProductType


def test_unrecognized_extension_hint_lists_skipped_suffixes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "scan.heic").write_bytes(b"II*\x00")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "notes.TXT").write_text("x")
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "doc.test.json").write_text("{}")

    _print_unrecognized_extension_hint(tmp_path)

    err = capsys.readouterr().err
    assert "Skipped files with unrecognized extensions: .heic, .txt." in err
    assert ".pdf" in err  # the supported list is spelled out


def test_unrecognized_extension_hint_is_silent_when_nothing_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    _print_unrecognized_extension_hint(tmp_path)
    assert capsys.readouterr().err == ""


def test_run_prints_extension_hint_for_unsupported_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "scan.heic").write_bytes(b"II*\x00")
    pipeline = PipelineSpec(pipeline_name="fake", provider_name="fake", product_type=ProductType.PARSE)
    monkeypatch.setattr(inference_cli_module, "get_pipeline", lambda name: pipeline)

    exit_code = InferenceCLI().run(
        pipeline="fake",
        input_dir=corpus,
        output_dir=tmp_path / "out",
        force_exit_on_completion=False,
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "No test cases found" in err
    assert "Skipped files with unrecognized extensions: .heic." in err


def test_run_passes_per_file_timeout_none_through_to_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI default is None so the pipeline's per_file_timeout (then the global default) applies."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    pipeline = PipelineSpec(pipeline_name="fake", provider_name="fake", product_type=ProductType.PARSE)
    monkeypatch.setattr(inference_cli_module, "get_pipeline", lambda name: pipeline)
    monkeypatch.setattr(inference_cli_module, "create_provider", lambda spec: object())

    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def _run_test_cases_sync(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("stop")

    monkeypatch.setattr(inference_cli_module, "InferenceRunner", _FakeRunner)

    exit_code = InferenceCLI().run(
        pipeline="fake",
        input_dir=corpus,
        output_dir=tmp_path / "out",
        max_concurrent=1,
        force_exit_on_completion=False,
    )

    assert exit_code == 1  # the fake runner raised
    assert "per_file_timeout" in captured
    assert captured["per_file_timeout"] is None
