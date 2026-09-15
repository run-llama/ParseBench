"""Which ``*.result.json`` files the runner treats as benchmark examples."""

from __future__ import annotations

from pathlib import Path

from parse_bench.evaluation.runner import EvaluationRunner


def test_excludes_inline_image_crop_parse_artifacts(tmp_path: Path) -> None:
    """Inline-image crop parsing writes its own normalized artifacts to
    ``<document>.images/``; they are dependencies of the parent Parse result,
    not examples, and must never be scored."""
    runner = EvaluationRunner(output_dir=tmp_path)
    expected = tmp_path / "group" / "document.result.json"
    crop = tmp_path / "document.pdf.images" / "page_1_image_1_v2.result.json"
    nested_crop = tmp_path / "group" / "other.pdf.images" / "nested" / "page_2_image_1.result.json"
    for path in (expected, crop, nested_crop):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    assert runner._find_result_files(tmp_path) == [expected]


def test_images_suffix_only_matches_directory_components(tmp_path: Path) -> None:
    """A file whose *own* name mentions images is still a result; only a
    ``*.images`` directory on the path excludes it."""
    runner = EvaluationRunner(output_dir=tmp_path)
    keep = tmp_path / "group" / "images.result.json"
    keep.parent.mkdir(parents=True)
    keep.write_text("{}")

    assert runner._find_result_files(tmp_path) == [keep]
