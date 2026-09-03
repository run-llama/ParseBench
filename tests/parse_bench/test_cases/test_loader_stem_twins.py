"""Stem-twin dedupe coverage.

A dataset directory can carry derived artifacts next to the input file with the
same stem (`<sha>.pdf` + `<sha>.png`). Both used to load as test cases with the
same test_id, so every downstream layer keyed on test_id — result files,
evaluation — raced on which twin's parse won. Discovery must emit exactly one
test case per stem, preferring the canonical input format.

The upstream version of this test used spreadsheet twins; ParseBench does not
list spreadsheet extensions in ``SUPPORTED_EXTENSIONS``, so the twins here are
image files, which are supported inputs and rank below documents in
``_STEM_TWIN_PRIORITY``.
"""

from __future__ import annotations

import json
from pathlib import Path

from parse_bench.test_cases.loader import load_test_cases

_TEST_JSON = json.dumps({"test_rules": [{"type": "present", "text": "hello"}]})


def _grouped_twin_dataset(tmp_path: Path) -> Path:
    group = tmp_path / "financial_reports"
    group.mkdir()
    (group / "doc.pdf").write_bytes(b"%PDF-1.3 input")
    (group / "doc.png").write_bytes(b"\x89PNG derived artifact")
    (group / "doc.test.json").write_text(_TEST_JSON, encoding="utf-8")
    return group


def test_pdf_wins_over_sibling_png_with_shared_test_json(tmp_path: Path, capsys) -> None:
    group = _grouped_twin_dataset(tmp_path)

    cases = load_test_cases(tmp_path, product_type="PARSE")

    assert [c.test_id for c in cases] == ["financial_reports/doc"]
    assert cases[0].file_path == (group / "doc.pdf").resolve()
    assert "share stem 'doc'" in capsys.readouterr().out


def test_twin_priority_is_not_iteration_order(tmp_path: Path) -> None:
    # `.docx` sorts before `.pdf` alphabetically; the canonical input must still win.
    group = tmp_path / "g"
    group.mkdir()
    (group / "doc.docx").write_bytes(b"PK")
    (group / "doc.pdf").write_bytes(b"%PDF-1.3")
    (group / "doc.test.json").write_text(_TEST_JSON, encoding="utf-8")

    cases = load_test_cases(tmp_path, product_type="PARSE")

    assert len(cases) == 1
    assert cases[0].file_path.suffix == ".pdf"


def test_flat_directory_twins_are_also_deduped(tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.3")
    (tmp_path / "doc.png").write_bytes(b"\x89PNG")
    (tmp_path / "doc.test.json").write_text(_TEST_JSON, encoding="utf-8")

    cases = load_test_cases(tmp_path, product_type="PARSE")

    assert len(cases) == 1
    assert cases[0].file_path.suffix == ".pdf"


def test_image_without_twin_still_loads(tmp_path: Path) -> None:
    # Image-only documents must be untouched by the dedupe.
    group = tmp_path / "scans"
    group.mkdir()
    (group / "scan.png").write_bytes(b"\x89PNG real input")
    (group / "scan.test.json").write_text(_TEST_JSON, encoding="utf-8")

    cases = load_test_cases(tmp_path, product_type="PARSE")

    assert len(cases) == 1
    assert cases[0].file_path.suffix == ".png"


def test_unsupported_sibling_is_ignored_without_warning(tmp_path: Path, capsys) -> None:
    # Spreadsheet siblings are not supported inputs here, so they never enter the
    # candidate list and never trigger the twin warning.
    group = tmp_path / "g"
    group.mkdir()
    (group / "doc.pdf").write_bytes(b"%PDF-1.3")
    (group / "doc.xlsx").write_bytes(b"PK derived artifact")
    (group / "doc.test.json").write_text(_TEST_JSON, encoding="utf-8")

    cases = load_test_cases(tmp_path, product_type="PARSE")

    assert [c.file_path.suffix for c in cases] == [".pdf"]
    assert "share stem" not in capsys.readouterr().out


def test_artifact_dirs_do_not_flip_flat_dataset_into_groups(tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.3")
    (tmp_path / "doc.test.json").write_text(_TEST_JSON, encoding="utf-8")
    (tmp_path / "doc.pdf.images").mkdir()
    (tmp_path / "doc.parse").mkdir()

    cases = load_test_cases(tmp_path, product_type="PARSE")

    assert [c.test_id for c in cases] == [f"{tmp_path.name}/doc"]


def test_distinct_stems_are_all_kept(tmp_path: Path) -> None:
    group = tmp_path / "g"
    group.mkdir()
    for stem in ("a", "b"):
        (group / f"{stem}.pdf").write_bytes(b"%PDF-1.3")
        (group / f"{stem}.test.json").write_text(_TEST_JSON, encoding="utf-8")

    cases = load_test_cases(tmp_path, product_type="PARSE")

    assert sorted(c.test_id for c in cases) == ["g/a", "g/b"]
