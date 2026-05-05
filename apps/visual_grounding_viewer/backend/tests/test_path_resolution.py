from __future__ import annotations

import json
from pathlib import Path

from backend.path_resolution import (
    candidate_test_case_roots,
    map_host_path_to_files_url,
    map_files_url_to_host_path,
    normalize_user_path_input,
    parse_metadata_test_cases_dir,
    resolve_existing_test_case_root,
)


def test_parse_metadata_test_cases_dir(tmp_path: Path) -> None:
    metadata_path = tmp_path / "_metadata.json"
    metadata_path.write_text(
        json.dumps({"test_cases_dir": "/datasets/bench-data/data/visual_grounding/v1.3"}),
        encoding="utf-8",
    )

    parsed = parse_metadata_test_cases_dir(metadata_path)
    assert parsed == "/datasets/bench-data/data/visual_grounding/v1.3"


def test_candidate_test_case_roots_remaps_ci_path_from_results_anchor(tmp_path: Path) -> None:
    results_root = tmp_path / "shared-data" / "bench-data" / "results" / "2026-02-26" / "run123" / "candidate"
    results_root.mkdir(parents=True)

    expected_mapped = tmp_path / "shared-data" / "bench-data" / "data" / "visual_grounding" / "v1.3"
    expected_mapped.mkdir(parents=True)

    candidates = candidate_test_case_roots(
        "/datasets/bench-data/data/visual_grounding/v1.3",
        results_root=results_root,
    )

    assert any(candidate.resolve(strict=False) == expected_mapped.resolve(strict=False) for candidate in candidates)

    resolved = resolve_existing_test_case_root(candidates)
    assert resolved == expected_mapped.resolve(strict=True)


def test_candidate_test_case_roots_prefers_explicit_hint_first(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()

    explicit_root = tmp_path / "explicit-test-cases"
    explicit_root.mkdir()

    candidates = candidate_test_case_roots(
        "/datasets/bench-data/data/visual_grounding/v1.3",
        results_root=results_root,
        explicit_hint=str(explicit_root),
    )

    assert candidates
    assert candidates[0].resolve(strict=False) == explicit_root.resolve(strict=True)


def test_map_files_url_to_host_path(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "shared-data"
    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT", str(shared_root))

    mapped = map_files_url_to_host_path(
        "http://localhost/files/bench-data/results/2026-02-26/run123/candidate_pipeline"
    )

    assert mapped is not None
    expected = shared_root / "bench-data" / "results" / "2026-02-26" / "run123" / "candidate_pipeline"
    assert mapped.resolve(strict=False) == expected.resolve(strict=False)


def test_map_files_url_to_host_path_blocks_path_traversal(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "shared-data"
    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT", str(shared_root))

    mapped = map_files_url_to_host_path("http://localhost/files/../../etc/passwd")
    assert mapped is None


def test_map_host_path_to_files_url(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "shared-data"
    source_path = shared_root / "bench-data" / "data" / "visual grounding" / "doc 1.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT", str(shared_root))
    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_BASE_URL", "http://localhost")

    mapped = map_host_path_to_files_url(source_path)

    assert mapped == "http://localhost/files/bench-data/data/visual%20grounding/doc%201.pdf"


def test_map_host_path_to_files_url_returns_none_outside_shared_root(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "shared-data"
    source_path = tmp_path / "outside" / "doc.pdf"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT", str(shared_root))

    assert map_host_path_to_files_url(source_path) is None


def test_normalize_user_path_input_maps_files_url(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "shared-data"
    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT", str(shared_root))

    normalized, note = normalize_user_path_input(
        "http://localhost/files/bench-data/results/2026-02-26/run123/candidate_pipeline",
        label="Results path",
    )

    assert normalized == str(
        (shared_root / "bench-data" / "results" / "2026-02-26" / "run123" / "candidate_pipeline").resolve(strict=False)
    )
    assert note is not None
    assert "mapped files URL" in note
