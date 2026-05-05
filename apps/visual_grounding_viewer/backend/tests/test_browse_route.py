from __future__ import annotations

import os
from pathlib import Path

from backend.routes.browse import browse_directory


def test_browse_lists_directories_only(monkeypatch, tmp_path: Path) -> None:
    browse_root = tmp_path / "browse-root"
    browse_root.mkdir()
    (browse_root / "alpha").mkdir()
    (browse_root / "beta").mkdir()
    (browse_root / "file.txt").write_text("x", encoding="utf-8")
    os.utime(browse_root / "alpha", ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    os.utime(browse_root / "beta", ns=(1_700_000_100_000_000_000, 1_700_000_100_000_000_000))

    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_BROWSE_ROOTS", str(browse_root))
    payload = browse_directory()

    assert payload.current == str(browse_root.resolve(strict=True))
    assert payload.parent is None
    assert [item.name for item in payload.items] == ["beta", "alpha"]
    assert payload.items[0].last_modified_ms > payload.items[1].last_modified_ms


def test_browse_restricts_paths_outside_allowed_roots(monkeypatch, tmp_path: Path) -> None:
    browse_root = tmp_path / "browse-root"
    browse_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_BROWSE_ROOTS", str(browse_root))
    payload = browse_directory(path=str(outside))

    assert payload.current == str(browse_root.resolve(strict=True))


def test_browse_accepts_files_url_path(monkeypatch, tmp_path: Path) -> None:
    browse_root = tmp_path / "shared-data"
    target = browse_root / "bench-data" / "results"
    target.mkdir(parents=True)

    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_BROWSE_ROOTS", str(browse_root))
    monkeypatch.setenv("VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT", str(browse_root))

    payload = browse_directory(path="http://localhost/files/bench-data/results")
    assert payload.current == str(target.resolve(strict=True))
