from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .constants import ARTIFACT_SUFFIXES, MAX_PAGE_SIZE, SOURCE_EXTENSIONS
from .models import ArtifactFlags, FolderNode, IndexCounts, IndexResponse, VisualizableDocument
from .path_resolution import (
    candidate_test_case_roots,
    discover_metadata_files,
    normalize_user_path_input,
    parse_metadata_test_cases_dir,
    resolve_existing_test_case_root,
)


@dataclass
class ArtifactGroup:
    relative_dir: str
    canonical_stem: str
    source_files: list[Path] = field(default_factory=list)
    raw_files: list[Path] = field(default_factory=list)
    result_files: list[Path] = field(default_factory=list)
    v2_items_files: list[Path] = field(default_factory=list)


@dataclass
class IndexedDocumentInternal:
    doc_id: str
    base_name: str
    relative_dir: str
    source_kind: Literal["pdf", "image"]
    source_ext: str
    last_modified_ms: int
    source_path: Path
    raw_path: Path | None
    result_path: Path | None
    v2_items_path: Path | None
    markdown_path: Path | None
    markdown_json_path: Path | None
    artifact_flags: ArtifactFlags
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    test_case_path: Path | None = None


@dataclass
class IndexBuildResult:
    response: IndexResponse
    docs_by_id: dict[str, IndexedDocumentInternal]


@dataclass
class CacheEntry:
    root_path: Path
    snapshot: tuple[int, int]
    full_response: IndexResponse
    docs_by_id: dict[str, IndexedDocumentInternal]


@dataclass
class SourceIndex:
    root_path: Path
    by_key: dict[tuple[str, str], list[tuple[Path, str]]] = field(default_factory=dict)
    by_stem: dict[str, list[tuple[Path, str]]] = field(default_factory=dict)


@dataclass
class MetadataContext:
    metadata_path: Path
    metadata_dir: Path
    raw_test_cases_dir: str
    resolved_test_cases_root: Path | None


_CACHE: dict[str, CacheEntry] = {}


def _canonicalize_stem(stem: str) -> str:
    normalized = stem
    while ".pdf_" in normalized:
        normalized = normalized.replace(".pdf_", "_")
    if normalized.endswith(".pdf"):
        normalized = normalized[: -len(".pdf")]
    return normalized


def _detect_artifact(path: Path) -> tuple[str, str] | None:
    name = path.name
    for artifact, suffix in ARTIFACT_SUFFIXES.items():
        if name.endswith(suffix):
            stem = _canonicalize_stem(name[: -len(suffix)])
            return artifact, stem

    ext = path.suffix.lower()
    if ext in SOURCE_EXTENSIONS:
        stem = _canonicalize_stem(name[: -len(ext)])
        return "source", stem

    return None


def _hash_doc_id(relative_dir: str, canonical_stem: str) -> str:
    raw = f"{relative_dir}::{canonical_stem}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _find_nearest_evaluation_report_path(artifact_path: Path | None) -> Path | None:
    if artifact_path is None or not artifact_path.is_file():
        return None

    current = artifact_path.parent
    while True:
        candidate = current / "_evaluation_report.json"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


def _resolve_report_example_id(artifact_path: Path, report_path: Path) -> str | None:
    try:
        relative = artifact_path.relative_to(report_path.parent)
    except ValueError:
        return None

    relative_name = str(relative)
    for suffix in (".result.json", ".raw.json", ".v2.items.json"):
        if relative_name.endswith(suffix):
            return relative_name[: -len(suffix)]
    return relative_name


def _load_report_metric_index(report_path: Path) -> dict[str, dict[str, float]]:
    payload = _load_json(report_path)
    if not payload:
        return {}

    per_example_results = payload.get("per_example_results")
    if not isinstance(per_example_results, list):
        return {}

    by_example: dict[str, dict[str, float]] = {}
    for example in per_example_results:
        if not isinstance(example, dict):
            continue
        metrics_payload = example.get("metrics")
        if not isinstance(metrics_payload, list):
            continue

        metrics: dict[str, float] = {}
        for metric in metrics_payload:
            if not isinstance(metric, dict):
                continue
            metric_name = metric.get("metric_name")
            metric_value = metric.get("value")
            if not isinstance(metric_name, str):
                continue
            if not isinstance(metric_value, (int, float)):
                continue
            metrics[metric_name] = float(metric_value)

        for key_name in ("example_id", "test_id"):
            example_key = example.get(key_name)
            if isinstance(example_key, str) and example_key and example_key not in by_example:
                by_example[example_key] = metrics

    return by_example


def _raw_output_has_grounding_payload(raw_output: dict) -> bool:
    v2_items = raw_output.get("v2_items")
    if not isinstance(v2_items, dict):
        v2_items = None

    if v2_items is not None:
        pages = v2_items.get("pages")
        if isinstance(pages, list):
            return True

    items = raw_output.get("items")
    if isinstance(items, dict):
        pages = items.get("pages")
        if isinstance(pages, list):
            return True

    for grounded_key in ("v2_grounded_items", "grounded_items"):
        grounded_pages = raw_output.get(grounded_key)
        if isinstance(grounded_pages, list) and grounded_pages:
            return True

    parse_raw_output = raw_output.get("parse_raw_output")
    if isinstance(parse_raw_output, dict) and _raw_output_has_grounding_payload(parse_raw_output):
        return True

    return False


def _has_grounding_payload(path: Path) -> bool:
    payload = _load_json(path)
    if not payload:
        return False

    output = payload.get("output")
    if isinstance(output, dict):
        layout_pages = output.get("layout_pages")
        if isinstance(layout_pages, list) and layout_pages:
            return True

        field_citations = output.get("field_citations")
        if isinstance(field_citations, list) and field_citations:
            return True

    raw_output = payload.get("raw_output")
    if not isinstance(raw_output, dict):
        return False

    if _raw_output_has_grounding_payload(raw_output):
        return True

    return False


def _select_single(paths: list[Path], label: str, warnings: list[str]) -> Path | None:
    if not paths:
        return None

    if len(paths) > 1:
        ordered = sorted(paths, key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
        warnings.append(f"Multiple {label} files found; selected newest: {ordered[0]}")
        return ordered[0]

    return paths[0]


def _resolve_source_path(path: Path, warnings: list[str]) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        warnings.append(f"Broken source symlink or missing source file: {path}")
        return None

    if not resolved.is_file():
        warnings.append(f"Source is not a file: {path}")
        return None

    return resolved


def _resolve_test_case_json_path(source_path: Path, base_name: str) -> Path | None:
    candidate = source_path.parent / f"{base_name}.test.json"
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None
    return resolved if resolved.is_file() else None


def _select_source_candidate(
    source_candidates: list[tuple[Path, str]], warnings: list[str], group_label: str
) -> tuple[Path, str] | None:
    if not source_candidates:
        return None

    if len(source_candidates) == 1:
        return source_candidates[0]

    pdf_candidates = [candidate for candidate in source_candidates if candidate[1] == "pdf"]
    image_candidates = [candidate for candidate in source_candidates if candidate[1] == "image"]

    if len(pdf_candidates) == 1:
        warnings.append(f"Both PDF and image sources found for {group_label}; preferring PDF source.")
        return pdf_candidates[0]

    if len(pdf_candidates) > 1:
        return None

    if len(image_candidates) == 1:
        warnings.append(f"Multiple image-like source entries found for {group_label}; using first.")
        return image_candidates[0]

    return None


def _build_folder_tree(relative_dirs: list[str]) -> FolderNode:
    nodes: dict[str, dict] = {".": {"name": ".", "path": ".", "children": {}, "document_count": 0}}

    for rel_dir in relative_dirs:
        parts = [part for part in rel_dir.split("/") if part and part != "."]
        current_path = "."

        for part in parts:
            parent = nodes[current_path]
            next_path = part if current_path == "." else f"{current_path}/{part}"
            if next_path not in nodes:
                nodes[next_path] = {
                    "name": part,
                    "path": next_path,
                    "children": {},
                    "document_count": 0,
                }
                parent["children"][part] = next_path
            current_path = next_path

        nodes[current_path]["document_count"] += 1

    def build(path: str) -> FolderNode:
        node_data = nodes[path]
        children_nodes = [build(nodes[path]["children"][key]) for key in sorted(node_data["children"])]
        total = node_data["document_count"] + sum(child.total_document_count for child in children_nodes)
        return FolderNode(
            name=node_data["name"],
            path=node_data["path"],
            document_count=node_data["document_count"],
            total_document_count=total,
            children=children_nodes,
        )

    return build(".")


def _paginate_documents(
    documents: list[VisualizableDocument], page: int, page_size: int
) -> tuple[list[VisualizableDocument], bool]:
    safe_size = max(1, min(page_size, MAX_PAGE_SIZE))
    start = (page - 1) * safe_size
    end = start + safe_size
    return documents[start:end], end < len(documents)


def _build_snapshot(root_path: Path) -> tuple[int, int]:
    count = 0
    max_mtime_ns = 0
    for file_path in root_path.rglob("*"):
        if not file_path.is_file() and not file_path.is_symlink():
            continue
        count += 1
        mtime_ns = file_path.lstat().st_mtime_ns
        max_mtime_ns = max(max_mtime_ns, mtime_ns)
    return count, max_mtime_ns


def _path_mtime_ms(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        return path.stat().st_mtime_ns // 1_000_000
    except OSError:
        return 0


def _latest_mtime_ms(*paths: Path | None) -> int:
    return max((_path_mtime_ms(path) for path in paths), default=0)


def _add_source_entry(
    source_index: SourceIndex,
    relative_dir: str,
    canonical_stem: str,
    candidate: tuple[Path, str],
) -> None:
    source_index.by_key.setdefault((relative_dir, canonical_stem), []).append(candidate)
    source_index.by_stem.setdefault(canonical_stem, []).append(candidate)


def _build_source_index(root_path: Path) -> SourceIndex:
    source_index = SourceIndex(root_path=root_path)

    for file_path in root_path.rglob("*"):
        if not file_path.is_file() and not file_path.is_symlink():
            continue

        detected = _detect_artifact(file_path)
        if detected is None:
            continue

        artifact_type, canonical_stem = detected
        if artifact_type != "source":
            continue

        ext = file_path.suffix.lower()
        source_kind = SOURCE_EXTENSIONS.get(ext)
        if source_kind is None:
            continue

        relative_dir = str(file_path.parent.relative_to(root_path))
        if relative_dir == "":
            relative_dir = "."

        _add_source_entry(source_index, relative_dir, canonical_stem, (file_path, source_kind))

    return source_index


def _lookup_source_candidates(
    source_index: SourceIndex,
    relative_dir: str,
    canonical_stem: str,
    warnings: list[str],
    group_label: str,
    source_label: str,
) -> list[tuple[Path, str]]:
    exact = source_index.by_key.get((relative_dir, canonical_stem), [])
    if exact:
        return exact

    stem_matches = source_index.by_stem.get(canonical_stem, [])
    if len(stem_matches) == 1:
        warnings.append(f"No exact path match for {group_label}; using unique stem match from {source_label}.")
        return stem_matches

    if len(stem_matches) > 1:
        warnings.append(
            f"No exact path match for {group_label}; found {len(stem_matches)} stem matches in {source_label}."
        )

    return []


def _discover_metadata_contexts(resolved_root: Path, warnings: list[str]) -> dict[Path, list[MetadataContext]]:
    contexts_by_dir: dict[Path, list[MetadataContext]] = {}

    for metadata_path in discover_metadata_files(resolved_root):
        raw_test_cases_dir = parse_metadata_test_cases_dir(metadata_path)
        if raw_test_cases_dir is None:
            continue

        candidates = candidate_test_case_roots(
            raw_test_cases_dir,
            results_root=resolved_root,
            metadata_path=metadata_path,
        )
        resolved_test_cases_root = resolve_existing_test_case_root(candidates)

        context = MetadataContext(
            metadata_path=metadata_path,
            metadata_dir=metadata_path.parent.resolve(strict=False),
            raw_test_cases_dir=raw_test_cases_dir,
            resolved_test_cases_root=resolved_test_cases_root,
        )
        contexts_by_dir.setdefault(context.metadata_dir, []).append(context)

        if resolved_test_cases_root is None:
            warnings.append(
                "Could not resolve metadata test_cases_dir "
                f"'{raw_test_cases_dir}' from {metadata_path}. "
                "Provide Test cases path manually."
            )

    return contexts_by_dir


def _ordered_metadata_contexts_for_group(
    contexts_by_dir: dict[Path, list[MetadataContext]],
    group_relative_dir: str,
    resolved_root: Path,
) -> list[MetadataContext]:
    group_dir = (
        resolved_root if group_relative_dir == "." else (resolved_root / group_relative_dir).resolve(strict=False)
    )
    ordered: list[MetadataContext] = []

    current = group_dir
    while True:
        ordered.extend(contexts_by_dir.get(current, []))
        if current == resolved_root:
            break
        if resolved_root not in current.parents:
            break
        current = current.parent

    return ordered


def _contains_metadata_file(root_path: Path) -> bool:
    for metadata_path in root_path.rglob("_metadata.json"):
        if metadata_path.is_file():
            return True
    return False


def _normalize_optional_path(path: str | None) -> str:
    if path is None:
        return ""
    trimmed = path.strip()
    if not trimmed:
        return ""
    return str(Path(trimmed).expanduser().resolve(strict=False))


def build_index(
    root_path: str,
    page: int,
    page_size: int,
    test_cases_path: str | None = None,
) -> IndexBuildResult:
    normalized_root_input, root_input_note = normalize_user_path_input(
        root_path,
        label="Results path",
    )
    resolved_root = Path(normalized_root_input or root_path).expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ValueError(f"Invalid root_path: {root_path}")

    normalized_test_cases_input, test_cases_input_note = normalize_user_path_input(
        test_cases_path,
        label="Test cases path",
    )
    normalized_test_cases_path = _normalize_optional_path(normalized_test_cases_input)
    cache_enabled = normalized_test_cases_path == "" and not _contains_metadata_file(resolved_root)

    cache_key = f"{resolved_root}::{normalized_test_cases_path}"
    snapshot = _build_snapshot(resolved_root)
    cache_entry = _CACHE.get(cache_key)

    if cache_enabled and cache_entry and cache_entry.snapshot == snapshot:
        docs_page, has_more = _paginate_documents(cache_entry.full_response.documents, page, page_size)
        cached_warnings = list(cache_entry.full_response.warnings)
        if root_input_note and root_input_note not in cached_warnings:
            cached_warnings.insert(0, root_input_note)
        if test_cases_input_note and test_cases_input_note not in cached_warnings:
            cached_warnings.insert(0, test_cases_input_note)
        response = cache_entry.full_response.model_copy(
            update={
                "root_path": root_path,
                "resolved_root_path": str(resolved_root),
                "documents": docs_page,
                "page": page,
                "page_size": page_size,
                "has_more": has_more,
                "warnings": cached_warnings,
            }
        )
        return IndexBuildResult(response=response, docs_by_id=cache_entry.docs_by_id)

    warnings: list[str] = []
    if root_input_note:
        warnings.append(root_input_note)
    if test_cases_input_note:
        warnings.append(test_cases_input_note)
    groups: dict[tuple[str, str], ArtifactGroup] = {}

    for file_path in resolved_root.rglob("*"):
        if not file_path.is_file() and not file_path.is_symlink():
            continue

        detected = _detect_artifact(file_path)
        if not detected:
            continue

        artifact_type, canonical_stem = detected
        relative_dir = str(file_path.parent.relative_to(resolved_root))
        if relative_dir == "":
            relative_dir = "."

        group_key = (relative_dir, canonical_stem)
        group = groups.get(group_key)
        if group is None:
            group = ArtifactGroup(relative_dir=relative_dir, canonical_stem=canonical_stem)
            groups[group_key] = group

        if artifact_type == "source":
            group.source_files.append(file_path)
        elif artifact_type == "raw":
            group.raw_files.append(file_path)
        elif artifact_type == "result":
            group.result_files.append(file_path)
        elif artifact_type == "v2_items":
            group.v2_items_files.append(file_path)

    source_index_cache: dict[Path, SourceIndex] = {}

    explicit_source_index: SourceIndex | None = None
    trimmed_test_cases_path = (normalized_test_cases_input or "").strip()
    if trimmed_test_cases_path:
        explicit_candidates = candidate_test_case_roots(
            trimmed_test_cases_path,
            results_root=resolved_root,
            explicit_hint=trimmed_test_cases_path,
        )
        explicit_resolved = resolve_existing_test_case_root(explicit_candidates)
        if explicit_resolved is None:
            warnings.append(f"Test cases path '{trimmed_test_cases_path}' is invalid or inaccessible.")
        else:
            explicit_source_index = _build_source_index(explicit_resolved)
            source_index_cache[explicit_resolved] = explicit_source_index
            warnings.append(f"Using test cases path override: {explicit_resolved}")

    metadata_contexts_by_dir = _discover_metadata_contexts(resolved_root, warnings)
    report_metrics_cache: dict[Path, dict[str, dict[str, float]]] = {}

    docs_internal: list[IndexedDocumentInternal] = []
    skipped = 0

    for group in groups.values():
        group_label = f"{group.relative_dir}/{group.canonical_stem}"
        has_artifact_payload = bool(group.v2_items_files or group.raw_files or group.result_files)

        source_candidates: list[tuple[Path, str]] = []
        for source_file in group.source_files:
            ext = source_file.suffix.lower()
            source_kind = SOURCE_EXTENSIONS.get(ext)
            if source_kind:
                source_candidates.append((source_file, source_kind))

        source_origin = "results"

        if not source_candidates and has_artifact_payload and explicit_source_index is not None:
            explicit_matches = _lookup_source_candidates(
                explicit_source_index,
                group.relative_dir,
                group.canonical_stem,
                warnings,
                group_label,
                f"test_cases_path({explicit_source_index.root_path})",
            )
            if explicit_matches:
                source_candidates = explicit_matches
                source_origin = "test_cases_override"

        if not source_candidates and has_artifact_payload:
            for context in _ordered_metadata_contexts_for_group(
                metadata_contexts_by_dir,
                group.relative_dir,
                resolved_root,
            ):
                if context.resolved_test_cases_root is None:
                    continue

                metadata_root = context.resolved_test_cases_root
                source_index = source_index_cache.get(metadata_root)
                if source_index is None:
                    source_index = _build_source_index(metadata_root)
                    source_index_cache[metadata_root] = source_index

                metadata_matches = _lookup_source_candidates(
                    source_index,
                    group.relative_dir,
                    group.canonical_stem,
                    warnings,
                    group_label,
                    f"metadata({context.metadata_path})",
                )
                if metadata_matches:
                    source_candidates = metadata_matches
                    source_origin = "metadata"
                    break

        if not source_candidates:
            if has_artifact_payload:
                skipped += 1
                warnings.append(
                    f"Skipped {group_label}: no matching source file found. "
                    "If this is a results-only folder, provide Test cases path manually."
                )
            continue

        selected_source = _select_source_candidate(source_candidates, warnings, group_label)
        if selected_source is None:
            skipped += 1
            warnings.append(f"Skipped ambiguous source group {group_label}: {len(source_candidates)} source files")
            continue

        source_file, source_kind = selected_source
        source_resolved = _resolve_source_path(source_file, warnings)
        if source_resolved is None:
            skipped += 1
            continue

        if source_origin != "results":
            warnings.append(f"Resolved source for {group_label} via {source_origin}: {source_resolved}")

        test_case_path = _resolve_test_case_json_path(source_resolved, group.canonical_stem)

        if test_case_path is None and explicit_source_index is not None:
            explicit_matches = _lookup_source_candidates(
                explicit_source_index,
                group.relative_dir,
                group.canonical_stem,
                warnings=[],
                group_label=group_label,
                source_label=f"test_cases_path({explicit_source_index.root_path})",
            )
            explicit_selected = _select_source_candidate(explicit_matches, [], group_label)
            if explicit_selected is not None:
                explicit_source_resolved = _resolve_source_path(explicit_selected[0], warnings=[])
                if explicit_source_resolved is not None:
                    test_case_path = _resolve_test_case_json_path(explicit_source_resolved, group.canonical_stem)

        if test_case_path is None:
            for context in _ordered_metadata_contexts_for_group(
                metadata_contexts_by_dir,
                group.relative_dir,
                resolved_root,
            ):
                if context.resolved_test_cases_root is None:
                    continue

                metadata_root = context.resolved_test_cases_root
                source_index = source_index_cache.get(metadata_root)
                if source_index is None:
                    source_index = _build_source_index(metadata_root)
                    source_index_cache[metadata_root] = source_index

                metadata_matches = _lookup_source_candidates(
                    source_index,
                    group.relative_dir,
                    group.canonical_stem,
                    warnings=[],
                    group_label=group_label,
                    source_label=f"metadata({context.metadata_path})",
                )
                metadata_selected = _select_source_candidate(metadata_matches, [], group_label)
                if metadata_selected is None:
                    continue

                metadata_source_resolved = _resolve_source_path(metadata_selected[0], warnings=[])
                if metadata_source_resolved is None:
                    continue

                test_case_path = _resolve_test_case_json_path(metadata_source_resolved, group.canonical_stem)
                if test_case_path is not None:
                    break

        selected_v2 = _select_single(group.v2_items_files, "v2.items", warnings)
        selected_raw = _select_single(group.raw_files, "raw", warnings)
        selected_result = _select_single(group.result_files, "result", warnings)

        has_v2_file = selected_v2 is not None
        has_raw_file = selected_raw is not None
        has_result_file = selected_result is not None

        has_grounding_payload = has_v2_file
        if not has_grounding_payload and selected_raw is not None:
            has_grounding_payload = _has_grounding_payload(selected_raw)
        if not has_grounding_payload and selected_result is not None:
            has_grounding_payload = _has_grounding_payload(selected_result)

        if not has_grounding_payload:
            skipped += 1
            continue

        source_ext = source_file.suffix.lower()
        doc_id = _hash_doc_id(group.relative_dir, group.canonical_stem)
        markdown_path = source_file.parent / f"{group.canonical_stem}.md"
        if not markdown_path.is_file():
            markdown_path = None
        markdown_json_path = source_file.parent / f"{group.canonical_stem}.v2.md.json"
        if not markdown_json_path.is_file():
            markdown_json_path = None
        artifact_flags = ArtifactFlags(
            has_v2_items_file=has_v2_file,
            has_raw_file=has_raw_file,
            has_result_file=has_result_file,
            has_v2_items_payload=has_grounding_payload,
        )
        evaluation_metrics: dict[str, float] = {}
        metric_lookup_artifact = selected_result or selected_raw or selected_v2
        report_path = _find_nearest_evaluation_report_path(metric_lookup_artifact)
        if report_path is not None and metric_lookup_artifact is not None:
            report_metric_index = report_metrics_cache.get(report_path)
            if report_metric_index is None:
                report_metric_index = _load_report_metric_index(report_path)
                report_metrics_cache[report_path] = report_metric_index

            example_id = _resolve_report_example_id(metric_lookup_artifact, report_path)
            if example_id is None or example_id not in report_metric_index:
                metric_payload = _load_json(metric_lookup_artifact)
                request = metric_payload.get("request") if isinstance(metric_payload, dict) else None
                request_example_id = request.get("example_id") if isinstance(request, dict) else None
                if isinstance(request_example_id, str):
                    example_id = request_example_id

            if example_id is not None:
                evaluation_metrics = dict(report_metric_index.get(example_id, {}))

        last_modified_ms = _latest_mtime_ms(
            source_resolved,
            selected_v2,
            selected_raw,
            selected_result,
            markdown_path,
            markdown_json_path,
        )

        docs_internal.append(
            IndexedDocumentInternal(
                doc_id=doc_id,
                base_name=group.canonical_stem,
                relative_dir=group.relative_dir,
                source_kind="pdf" if source_kind == "pdf" else "image",
                source_ext=source_ext,
                last_modified_ms=last_modified_ms,
                source_path=source_resolved,
                test_case_path=test_case_path,
                raw_path=selected_raw,
                result_path=selected_result,
                v2_items_path=selected_v2,
                markdown_path=markdown_path,
                markdown_json_path=markdown_json_path,
                artifact_flags=artifact_flags,
                evaluation_metrics=evaluation_metrics,
            )
        )

    docs_internal.sort(key=lambda d: (-d.last_modified_ms, d.relative_dir, d.base_name.lower()))

    documents = [
        VisualizableDocument(
            doc_id=doc.doc_id,
            base_name=doc.base_name,
            relative_dir=doc.relative_dir,
            source_kind=doc.source_kind,
            source_ext=doc.source_ext,
            last_modified_ms=doc.last_modified_ms,
            artifact_flags=doc.artifact_flags,
            evaluation_metrics=doc.evaluation_metrics,
        )
        for doc in docs_internal
    ]

    tree = _build_folder_tree([doc.relative_dir for doc in docs_internal])
    docs_page, has_more = _paginate_documents(documents, page, page_size)

    full_response = IndexResponse(
        session_id="",
        root_path=root_path,
        resolved_root_path=str(resolved_root),
        tree=tree,
        documents=documents,
        document_total=len(documents),
        page=1,
        page_size=len(documents) if documents else page_size,
        has_more=False,
        counts=IndexCounts(
            visualizable=len(documents),
            skipped=skipped,
            warnings=len(warnings),
        ),
        warnings=warnings,
    )

    docs_by_id = {doc.doc_id: doc for doc in docs_internal}
    if cache_enabled:
        _CACHE[cache_key] = CacheEntry(
            root_path=resolved_root,
            snapshot=snapshot,
            full_response=full_response,
            docs_by_id=docs_by_id,
        )

    page_response = full_response.model_copy(
        update={
            "documents": docs_page,
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
        }
    )

    return IndexBuildResult(response=page_response, docs_by_id=docs_by_id)
