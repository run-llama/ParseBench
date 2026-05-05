from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

_METADATA_FILENAME = "_metadata.json"
_BENCH_ANCHORS = ("parsebench-data", "bench-data")
_BASE_HINTS_ENV = "VISUAL_GROUNDING_VIEWER_TEST_CASE_BASE_HINTS"
_FILES_URL_ROOT_ENV = "VISUAL_GROUNDING_VIEWER_FILES_URL_ROOT"
_FILES_URL_HOSTS_ENV = "VISUAL_GROUNDING_VIEWER_FILES_URL_HOSTS"
_FILES_URL_BASE_URL_ENV = "VISUAL_GROUNDING_VIEWER_FILES_URL_BASE_URL"
_DEFAULT_FILES_URL_ROOT = ""
_DEFAULT_FILES_URL_HOSTS = ("localhost", "127.0.0.1")
_DEFAULT_FILES_URL_BASE_URL = "http://localhost"
_FILES_URL_PREFIX = "/files/"


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _files_url_root() -> Path | None:
    root = os.getenv(_FILES_URL_ROOT_ENV, _DEFAULT_FILES_URL_ROOT).strip()
    if not root:
        return None
    return Path(root).expanduser()


def _files_url_allowed_hosts() -> set[str]:
    raw_hosts = os.getenv(_FILES_URL_HOSTS_ENV, "")
    normalized_hosts = raw_hosts.replace(";", ",").replace(os.pathsep, ",")
    hosts = {host.strip().lower() for host in normalized_hosts.split(",") if host.strip()}
    if hosts:
        return hosts
    return set(_DEFAULT_FILES_URL_HOSTS)


def _files_url_base_url() -> str:
    return os.getenv(_FILES_URL_BASE_URL_ENV, _DEFAULT_FILES_URL_BASE_URL).strip() or _DEFAULT_FILES_URL_BASE_URL


def map_files_url_to_host_path(raw_path: str) -> Path | None:
    parsed = urlparse(raw_path.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    host = (parsed.hostname or "").lower()
    allowed_hosts = _files_url_allowed_hosts()
    if "*" not in allowed_hosts and host not in allowed_hosts:
        return None

    if not parsed.path.startswith(_FILES_URL_PREFIX):
        return None

    relative = unquote(parsed.path[len(_FILES_URL_PREFIX) :]).strip("/")
    if not relative:
        return None

    root = _files_url_root()
    if root is None:
        return None
    root_resolved = root.resolve(strict=False)
    candidate = (root / relative).resolve(strict=False)
    if not _is_within(candidate, root_resolved):
        return None
    return candidate


def map_host_path_to_files_url(raw_path: Path) -> str | None:
    root = _files_url_root()
    if root is None:
        return None
    base_url = _files_url_base_url().rstrip("/")
    if not base_url:
        return None

    root_resolved = root.resolve(strict=False)
    candidate = raw_path.expanduser().resolve(strict=False)

    if not _is_within(candidate, root_resolved):
        return None

    relative = candidate.relative_to(root_resolved)
    relative_url = quote(relative.as_posix(), safe="/")
    return f"{base_url}{_FILES_URL_PREFIX}{relative_url}"


def normalize_user_path_input(raw_path: str | None, *, label: str) -> tuple[str | None, str | None]:
    if raw_path is None:
        return None, None

    trimmed = raw_path.strip()
    if not trimmed:
        return "", None

    mapped = map_files_url_to_host_path(trimmed)
    if mapped is None:
        return trimmed, None

    return str(mapped), f"{label}: mapped files URL '{trimmed}' to '{mapped}'."


def discover_metadata_files(results_root: Path) -> list[Path]:
    metadata_files: list[Path] = []
    for candidate in results_root.rglob(_METADATA_FILENAME):
        if candidate.is_file():
            metadata_files.append(candidate)
    return sorted(metadata_files)


def parse_metadata_test_cases_dir(metadata_path: Path) -> str | None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    raw_value = payload.get("test_cases_dir")
    if isinstance(raw_value, str):
        trimmed = raw_value.strip()
        return trimmed or None
    return None


def infer_bench_anchor_bases(results_root: Path) -> list[Path]:
    bases: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        normalized = path.expanduser()
        try:
            resolved = normalized.resolve(strict=False)
        except RuntimeError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        bases.append(resolved)

    resolved_root = results_root.expanduser().resolve(strict=False)
    parts = resolved_root.parts

    for anchor in _BENCH_ANCHORS:
        for idx, part in enumerate(parts):
            if part == anchor:
                add(Path(*parts[: idx + 1]))

    for idx, part in enumerate(parts):
        if part == "results" and idx > 0:
            add(Path(*parts[:idx]))

    raw_hints = os.getenv(_BASE_HINTS_ENV, "")
    normalized_hints = raw_hints.replace(";", ",").replace(os.pathsep, ",")
    for raw_hint in normalized_hints.split(","):
        hint = raw_hint.strip()
        if hint:
            add(Path(hint))

    return bases


def candidate_test_case_roots(
    raw_path: str,
    *,
    results_root: Path,
    metadata_path: Path | None = None,
    explicit_hint: str | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        expanded = path.expanduser()
        try:
            normalized = expanded.resolve(strict=False)
        except RuntimeError:
            return
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(expanded)

    if explicit_hint:
        explicit = Path(explicit_hint.strip()).expanduser()
        if explicit.is_absolute():
            add(explicit)
        else:
            add((results_root / explicit).resolve(strict=False))

    raw_candidate = Path(raw_path).expanduser()
    if raw_candidate.is_absolute():
        add(raw_candidate)
    elif metadata_path is not None:
        add((metadata_path.parent / raw_candidate).resolve(strict=False))
    else:
        add((results_root / raw_candidate).resolve(strict=False))

    absolute_raw = raw_candidate if raw_candidate.is_absolute() else None
    if absolute_raw is None:
        return candidates

    for anchor in _BENCH_ANCHORS:
        raw_parts = absolute_raw.parts
        if anchor not in raw_parts:
            continue

        anchor_index = raw_parts.index(anchor)
        suffix = Path(*raw_parts[anchor_index + 1 :])
        for base in infer_bench_anchor_bases(results_root):
            if base.name == anchor:
                add(base / suffix)
            else:
                add(base / anchor / suffix)

    return candidates


def resolve_existing_test_case_root(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except Exception:
            continue
        if resolved.is_dir():
            return resolved
    return None
