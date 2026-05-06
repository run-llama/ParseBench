"""Helpers for normalizing provider field citation bboxes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parse_bench.schemas.extract_output import FieldCitation
else:
    FieldCitation = Any

_STRUCTURAL_KEYS = {
    "citation",
    "citations",
    "document_metadata",
    "field_metadata",
    "fields",
    "metadata",
    "page_metadata",
    "properties",
    "row_metadata",
}


def _field_citation_cls() -> type[Any]:
    from parse_bench.schemas.extract_output import FieldCitation as _FieldCitation

    return _FieldCitation


def extract_extend_field_citations(raw_output: Mapping[str, Any]) -> list[FieldCitation]:
    """Extract citations from Extend AI processor-run metadata."""
    output = _as_mapping(_as_mapping(raw_output.get("processor_run")).get("output"))
    metadata = _as_mapping(output.get("metadata"))
    return _dedupe(_collect_field_map(metadata, source="extend"))


def extract_llamaextract_field_citations(metadata: Any, *, source: str) -> list[FieldCitation]:
    """Extract citations from LlamaExtract metadata in known and fallback shapes."""
    metadata_map = _as_mapping(metadata)
    if not metadata_map:
        return []

    citations: list[FieldCitation] = []

    for key in ("field_metadata", "document_metadata", "fields"):
        citations.extend(_collect_field_map(_as_mapping(metadata_map.get(key)), source=source))

    for key in ("page_metadata", "row_metadata"):
        entries = metadata_map.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
            continue
        for entry in entries:
            entry_map = _as_mapping(entry)
            default_page = _extract_page(entry_map)
            default_dimensions = _extract_dimensions(entry_map)
            for field_key in ("field_metadata", "document_metadata", "fields"):
                citations.extend(
                    _collect_field_map(
                        _as_mapping(entry_map.get(field_key)),
                        source=source,
                        default_page=default_page,
                        default_dimensions=default_dimensions,
                    )
                )

    citations.extend(_collect_recursive(node=metadata_map, source=source, path=[]))
    return _dedupe(citations)


def _collect_field_map(
    field_map: Mapping[str, Any],
    *,
    source: str,
    default_page: int | None = None,
    default_dimensions: tuple[float, float] | None = None,
) -> list[FieldCitation]:
    citations: list[FieldCitation] = []
    for field_path, node in field_map.items():
        if field_path.startswith("_"):
            continue
        citations.extend(
            _collect_node_citations(
                field_path=field_path,
                node=node,
                source=source,
                default_page=default_page,
                default_dimensions=default_dimensions,
            )
        )
    return citations


def _collect_node_citations(
    *,
    field_path: str,
    node: Any,
    source: str,
    default_page: int | None,
    default_dimensions: tuple[float, float] | None,
) -> list[FieldCitation]:
    node_map = _as_mapping(node)
    if not node_map:
        return []

    page = _extract_page(node_map) or default_page
    dimensions = _extract_dimensions(node_map) or default_dimensions
    citations: list[FieldCitation] = []
    for citation in _iter_citation_entries(node_map):
        citations.extend(
            _normalize_citation(
                field_path=field_path,
                citation=citation,
                source=source,
                default_page=page,
                default_dimensions=dimensions,
            )
        )
    return citations


def _iter_citation_entries(node: Mapping[str, Any]) -> list[Any]:
    """Iterate citation entries supporting both plural `citations` and singular `citation` keys."""
    entries: list[Any] = []
    for key in ("citations", "citation"):
        for entry in _as_sequence(node.get(key)):
            entries.append(entry)
    return entries


def _collect_recursive(*, node: Any, source: str, path: list[str]) -> list[FieldCitation]:
    node_map = _as_mapping(node)
    if not node_map:
        return []

    citations: list[FieldCitation] = []
    explicit_path = _extract_field_path(node_map)
    field_path = explicit_path or _format_field_path(path)
    if field_path:
        for citation in _iter_citation_entries(node_map):
            citations.extend(
                _normalize_citation(
                    field_path=field_path,
                    citation=citation,
                    source=source,
                    default_page=_extract_page(node_map),
                    default_dimensions=_extract_dimensions(node_map),
                )
            )

    for key, value in node_map.items():
        if key in ("citations", "citation"):
            continue
        next_path = path if key in _STRUCTURAL_KEYS else [*path, key]
        if isinstance(value, Mapping):
            citations.extend(_collect_recursive(node=value, source=source, path=next_path))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                item_path = next_path if key in _STRUCTURAL_KEYS else [*next_path, f"[{index}]"]
                citations.extend(_collect_recursive(node=item, source=source, path=item_path))

    return citations


def _format_field_path(path: list[str]) -> str:
    """Render path tokens so list-index tokens (`[N]`) attach to the prior key without a dot.

    GT field paths use bracket notation (`employees[0].basic_salary`). We collect tokens during
    the recursive walk and convert any leading-bracket tokens into bracket-joined segments so
    predictions match GT field path scope.
    """
    rendered = ""
    for token in path:
        if token.startswith("[") and token.endswith("]"):
            rendered += token
        elif rendered:
            rendered += "." + token
        else:
            rendered = token
    return rendered


def _normalize_citation(
    *,
    field_path: str,
    citation: Any,
    source: str,
    default_page: int | None,
    default_dimensions: tuple[float, float] | None,
) -> list[FieldCitation]:
    citation_map = _as_mapping(citation)
    if not citation_map:
        return []

    page = _extract_page(citation_map) or default_page or 1
    dimensions = _extract_dimensions(citation_map) or default_dimensions
    polygon = _extract_polygon(citation_map)
    reference_text = _extract_reference_text(citation_map)
    confidence = _extract_confidence(citation_map)
    metadata = _compact_metadata(citation_map)

    plural_bboxes = _extract_bbox_list(citation_map)
    if plural_bboxes:
        normalized_polygon = _normalize_polygon(polygon, dimensions) if polygon is not None else None
        results: list[FieldCitation] = []
        for entry_bbox in plural_bboxes:
            normalized_bbox = _normalize_bbox(entry_bbox, dimensions)
            if normalized_bbox is None:
                continue
            results.append(
                _field_citation_cls()(
                    field_path=field_path,
                    page=page,
                    bbox=normalized_bbox,
                    polygon=normalized_polygon,
                    reference_text=reference_text,
                    confidence=confidence,
                    source=source,
                    metadata=metadata,
                )
            )
        if results:
            return results
        return [
            _field_citation_cls()(
                field_path=field_path,
                page=page,
                bbox=None,
                polygon=normalized_polygon,
                reference_text=reference_text,
                confidence=confidence,
                source=source,
                metadata=metadata,
            )
        ]

    raw_bbox = _bbox_from_polygon(polygon) if polygon is not None else _extract_bbox(citation_map)
    normalized_bbox = _normalize_bbox(raw_bbox, dimensions)
    normalized_polygon = _normalize_polygon(polygon, dimensions) if polygon is not None else None
    return [
        _field_citation_cls()(
            field_path=field_path,
            page=page,
            bbox=normalized_bbox,
            polygon=normalized_polygon,
            reference_text=reference_text,
            confidence=confidence,
            source=source,
            metadata=metadata,
        )
    ]


def _extract_bbox_list(node: Mapping[str, Any]) -> list[list[float]] | None:
    """Extract a plural list of bboxes if `bounding_boxes` is present.

    Each entry can be either a 4-element [x, y, w, h] sequence or a mapping with
    x/y/w/h or x1/y1/x2/y2 keys.
    """
    raw = node.get("bounding_boxes")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    if not raw:
        return None
    bboxes: list[list[float]] = []
    for entry in raw:
        bbox: list[float] | None = None
        if isinstance(entry, Mapping):
            bbox = _bbox_from_mapping(entry)
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
            bbox = _bbox_from_sequence(entry)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes or None


def _extract_field_path(node: Mapping[str, Any]) -> str | None:
    for key in ("field_path", "fieldPath", "path", "field", "name", "key"):
        value = node.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_page(node: Mapping[str, Any]) -> int | None:
    for key in ("page", "page_number", "pageNumber"):
        value = _coerce_int(node.get(key))
        if value is not None and value >= 1:
            return value
    for key in ("page_index", "pageIndex"):
        value = _coerce_int(node.get(key))
        if value is not None and value >= 0:
            return value + 1
    return None


def _extract_dimensions(node: Mapping[str, Any]) -> tuple[float, float] | None:
    width = _coerce_float(_first_present(node, ("page_width", "pageWidth", "width", "image_width", "imageWidth")))
    height = _coerce_float(_first_present(node, ("page_height", "pageHeight", "height", "image_height", "imageHeight")))
    if width is not None and height is not None and width > 0 and height > 0:
        return width, height

    for key in ("page_dimensions", "pageDimensions", "page_size", "pageSize", "dimensions", "image_size", "imageSize"):
        size = _as_mapping(node.get(key))
        width = _coerce_float(_first_present(size, ("width", "w")))
        height = _coerce_float(_first_present(size, ("height", "h")))
        if width is not None and height is not None and width > 0 and height > 0:
            return width, height
    return None


def _extract_bbox(node: Mapping[str, Any]) -> list[float] | None:
    for key in ("bbox", "bounding_box", "boundingBox", "box"):
        bbox = node.get(key)
        bbox_from_dict = _bbox_from_mapping(_as_mapping(bbox))
        if bbox_from_dict is not None:
            return bbox_from_dict
        bbox_from_sequence = _bbox_from_sequence(bbox)
        if bbox_from_sequence is not None:
            return bbox_from_sequence

    bbox_from_dict = _bbox_from_mapping(node)
    if bbox_from_dict is not None:
        return bbox_from_dict
    return None


def _bbox_from_mapping(node: Mapping[str, Any]) -> list[float] | None:
    if not node:
        return None

    x = _coerce_float(_first_present(node, ("x", "left")))
    y = _coerce_float(_first_present(node, ("y", "top")))
    width = _coerce_float(_first_present(node, ("w", "width")))
    height = _coerce_float(_first_present(node, ("h", "height")))
    if x is not None and y is not None and width is not None and height is not None:
        return [x, y, width, height]

    x1 = _coerce_float(_first_present(node, ("x1", "left")))
    y1 = _coerce_float(_first_present(node, ("y1", "top")))
    x2 = _coerce_float(_first_present(node, ("x2", "right")))
    y2 = _coerce_float(_first_present(node, ("y2", "bottom")))
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        return [x1, y1, x2 - x1, y2 - y1]
    return None


def _bbox_from_sequence(raw: Any) -> list[float] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 4:
        return None
    values = [_coerce_float(value) for value in raw]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values if value is not None]


def _extract_polygon(node: Mapping[str, Any]) -> list[list[float]] | None:
    for key in ("polygon", "bounding_polygon", "boundingPolygon", "points", "vertices"):
        polygon = _polygon_from_raw(node.get(key))
        if polygon is not None:
            return polygon
    return None


def _polygon_from_raw(raw: Any) -> list[list[float]] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    if not raw:
        return None

    points: list[list[float]] = []
    if all(isinstance(point, Mapping) for point in raw):
        for point in raw:
            point_map = _as_mapping(point)
            x = _coerce_float(point_map.get("x"))
            y = _coerce_float(point_map.get("y"))
            if x is None or y is None:
                return None
            points.append([x, y])
    elif all(isinstance(point, Sequence) and not isinstance(point, (str, bytes, bytearray)) for point in raw):
        for point in raw:
            if len(point) < 2:
                return None
            x = _coerce_float(point[0])
            y = _coerce_float(point[1])
            if x is None or y is None:
                return None
            points.append([x, y])
    else:
        values = [_coerce_float(value) for value in raw]
        if len(values) % 2 != 0 or any(value is None for value in values):
            return None
        numeric_values = [float(value) for value in values if value is not None]
        points = [[numeric_values[index], numeric_values[index + 1]] for index in range(0, len(numeric_values), 2)]

    return points if len(points) >= 2 else None


def _bbox_from_polygon(polygon: list[list[float]] | None) -> list[float] | None:
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left = min(xs)
    top = min(ys)
    return [left, top, max(xs) - left, max(ys) - top]


def _normalize_bbox(raw_bbox: list[float] | None, dimensions: tuple[float, float] | None) -> list[float] | None:
    if raw_bbox is None or len(raw_bbox) != 4:
        return None
    x, y, width, height = raw_bbox
    if width <= 0 or height <= 0:
        return None

    if _looks_normalized(raw_bbox):
        normalized = raw_bbox
    elif dimensions is not None:
        page_width, page_height = dimensions
        normalized = [x / page_width, y / page_height, width / page_width, height / page_height]
    else:
        return None

    if not _looks_normalized(normalized):
        return None
    return [round(value, 8) for value in normalized]


def _normalize_polygon(
    polygon: list[list[float]] | None,
    dimensions: tuple[float, float] | None,
) -> list[list[float]] | None:
    if polygon is None:
        return None
    flat = [coordinate for point in polygon for coordinate in point]
    if all(0 <= value <= 1 for value in flat):
        return [[round(point[0], 8), round(point[1], 8)] for point in polygon]
    if dimensions is None:
        return None
    page_width, page_height = dimensions
    normalized = [[point[0] / page_width, point[1] / page_height] for point in polygon]
    if not all(0 <= value <= 1 for point in normalized for value in point):
        return None
    return [[round(point[0], 8), round(point[1], 8)] for point in normalized]


def _looks_normalized(bbox: list[float]) -> bool:
    x, y, width, height = bbox
    return (
        0 <= x <= 1
        and 0 <= y <= 1
        and 0 < width <= 1
        and 0 < height <= 1
        and x + width <= 1.000001
        and y + height <= 1.000001
    )


def _extract_reference_text(node: Mapping[str, Any]) -> str | None:
    value = _first_present(
        node, ("reference_text", "referenceText", "matching_text", "matchingText", "text", "content", "value")
    )
    if isinstance(value, str):
        return value
    return None


def _extract_confidence(node: Mapping[str, Any]) -> float | None:
    confidence = _coerce_float(_first_present(node, ("confidence", "score", "probability")))
    if confidence is None:
        return None
    return confidence


def _compact_metadata(node: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = {
        key: value
        for key, value in node.items()
        if key
        not in {
            "bbox",
            "bounding_box",
            "boundingBox",
            "box",
            "bounding_boxes",
            "polygon",
            "bounding_polygon",
            "boundingPolygon",
            "points",
            "vertices",
        }
    }
    return dict(metadata) if metadata else None


def _dedupe(citations: list[FieldCitation]) -> list[FieldCitation]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[FieldCitation] = []
    for citation in citations:
        key = (
            citation.field_path,
            citation.page,
            tuple(citation.bbox) if citation.bbox is not None else None,
            citation.reference_text,
            citation.source,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _first_present(node: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in node:
            return node[key]
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
    return None
