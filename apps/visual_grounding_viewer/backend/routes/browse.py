from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter

from ..models import BrowseItem, BrowseResponse
from ..path_resolution import normalize_user_path_input

router = APIRouter(prefix="/api", tags=["browse"])

_BROWSE_ROOTS_ENV = "VISUAL_GROUNDING_VIEWER_BROWSE_ROOTS"
_DEFAULT_BROWSE_ROOTS = (
    Path.home(),
    Path("/home"),
    Path("/Users"),
    Path("/mnt"),
    Path("/tmp"),
)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _allowed_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve(strict=True)
        except Exception:
            return
        if not resolved.is_dir() or resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    raw_roots = os.getenv(_BROWSE_ROOTS_ENV, "")
    normalized_roots = raw_roots.replace(";", ",").replace(os.pathsep, ",")
    for raw_root in normalized_roots.split(","):
        root_value = raw_root.strip()
        if root_value:
            add(Path(root_value))

    if roots:
        return roots

    for default_root in _DEFAULT_BROWSE_ROOTS:
        add(default_root)

    if roots:
        return roots

    fallback = Path("/").resolve(strict=True)
    return [fallback]


def _is_allowed(path: Path, allowed_roots: list[Path]) -> bool:
    return any(_is_within(path, root) for root in allowed_roots)


def _resolve_current_dir(path: str | None, allowed_roots: list[Path]) -> Path:
    default_root = allowed_roots[0]
    normalized_input, _ = normalize_user_path_input(path, label="Browse path")
    if not normalized_input:
        return default_root

    requested = Path(normalized_input).expanduser()
    if not requested.is_absolute():
        requested = default_root / requested

    try:
        resolved = requested.resolve(strict=True)
    except Exception:
        return default_root

    if not resolved.is_dir():
        return default_root
    if not _is_allowed(resolved, allowed_roots):
        return default_root

    return resolved


def _path_mtime_ms(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns // 1_000_000
    except OSError:
        return 0


@router.get("/browse", response_model=BrowseResponse)
def browse_directory(path: str | None = None) -> BrowseResponse:
    allowed_roots = _allowed_roots()
    current_dir = _resolve_current_dir(path, allowed_roots)

    parent = current_dir.parent
    parent_path = str(parent) if parent != current_dir and _is_allowed(parent, allowed_roots) else None

    items: list[BrowseItem] = []
    try:
        children = sorted(current_dir.iterdir(), key=lambda item: (-_path_mtime_ms(item), item.name.lower()))
    except PermissionError:
        children = []

    for child in children:
        if not child.is_dir() or child.name.startswith("."):
            continue
        normalized_child = child.resolve(strict=False)
        if not _is_allowed(normalized_child, allowed_roots):
            continue
        items.append(
            BrowseItem(
                name=child.name,
                path=str(normalized_child),
                last_modified_ms=_path_mtime_ms(child),
            )
        )

    return BrowseResponse(current=str(current_dir), parent=parent_path, items=items)
