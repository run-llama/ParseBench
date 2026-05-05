from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import fitz
from PIL import Image

from .gt_rules import load_page_gt_rules
from .indexer import IndexedDocumentInternal
from .models import (
    DocumentResponse,
    GroundingBbox,
    GroundingGranularLayer,
    GroundingGranularUnit,
    GroundingItem,
    GroundingPage,
)
from .path_resolution import map_host_path_to_files_url


@dataclass(slots=True)
class _GranularPayloadUnit:
    text: str
    bbox: dict[str, float]
    order_index: int
    unit_id: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    row_span: int | None = None
    column_span: int | None = None


@dataclass(slots=True)
class _GranularPayloadPage:
    page_number: int
    lines: list[_GranularPayloadUnit]
    words: list[_GranularPayloadUnit]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _extract_grounding_payload_from_raw_output(raw_output: Any) -> dict[str, Any] | None:
    if not isinstance(raw_output, dict):
        return None

    v2_items = raw_output.get("v2_items")
    v2_grounded_items = raw_output.get("v2_grounded_items")
    if isinstance(v2_items, dict) and isinstance(v2_items.get("pages"), list) and isinstance(v2_grounded_items, list):
        return _merge_llamaparse_items_payload(v2_items, v2_grounded_items)

    if isinstance(v2_items, dict) and isinstance(v2_items.get("pages"), list):
        return v2_items

    items = raw_output.get("items")
    if isinstance(items, dict) and isinstance(items.get("pages"), list):
        return items

    if isinstance(v2_grounded_items, list):
        return {"pages": v2_grounded_items}

    grounded_items = raw_output.get("grounded_items")
    if isinstance(grounded_items, list):
        return {"pages": grounded_items}

    parse_raw_output = raw_output.get("parse_raw_output")
    nested_payload = _extract_grounding_payload_from_raw_output(parse_raw_output)
    if nested_payload is not None:
        return nested_payload

    return None


def _merge_llamaparse_items_payload(
    display_payload: dict[str, Any],
    grounded_pages: list[Any],
) -> dict[str, Any]:
    raw_pages = display_payload.get("pages")
    if not isinstance(raw_pages, list):
        return display_payload

    merged_pages: list[dict[str, Any]] = []
    for page_index, display_page_entry in enumerate(raw_pages):
        if not isinstance(display_page_entry, dict):
            continue
        grounded_page_entry = grounded_pages[page_index] if page_index < len(grounded_pages) else None
        grounded_page = grounded_page_entry if isinstance(grounded_page_entry, dict) else None

        merged_page = dict(display_page_entry)
        if grounded_page is not None:
            for key, value in grounded_page.items():
                if key == "items":
                    continue
                if key not in merged_page:
                    merged_page[key] = value

        display_items = display_page_entry.get("items")
        grounded_items = grounded_page.get("items") if grounded_page is not None else None
        if isinstance(display_items, list) and isinstance(grounded_items, list):
            merged_page["items"] = _merge_llamaparse_item_list(display_items, grounded_items)

        merged_pages.append(merged_page)

    return {"pages": merged_pages}


def _merge_llamaparse_item_list(
    display_items: list[Any],
    grounded_items: list[Any],
) -> list[dict[str, Any]]:
    merged_items: list[dict[str, Any]] = []
    for item_index, display_item_entry in enumerate(display_items):
        if not isinstance(display_item_entry, dict):
            continue
        grounded_item_entry = grounded_items[item_index] if item_index < len(grounded_items) else None
        grounded_item = grounded_item_entry if isinstance(grounded_item_entry, dict) else None

        merged_item = dict(display_item_entry)
        if grounded_item is not None:
            for key, value in grounded_item.items():
                if key == "items":
                    continue
                if key == "grounding" or key not in merged_item:
                    merged_item[key] = value

        display_children = display_item_entry.get("items")
        grounded_children = grounded_item.get("items") if grounded_item is not None else None
        if isinstance(display_children, list) and isinstance(grounded_children, list):
            merged_item["items"] = _merge_llamaparse_item_list(display_children, grounded_children)

        merged_items.append(merged_item)

    return merged_items


def _extract_llamaparse_grounded_items_by_page(raw_payload: dict[str, Any] | None) -> dict[int, list[dict[str, Any]]]:
    if not isinstance(raw_payload, dict):
        return {}

    raw_output = raw_payload.get("raw_output")
    if not isinstance(raw_output, dict):
        return {}

    grounded_pages = raw_output.get("v2_grounded_items")
    if not isinstance(grounded_pages, list):
        return {}

    by_page: dict[int, list[dict[str, Any]]] = {}
    for page_index, page_entry in enumerate(grounded_pages):
        if not isinstance(page_entry, dict):
            continue
        page_number = _as_int(page_entry.get("page_number"), fallback=page_index + 1)
        items = page_entry.get("items")
        if not isinstance(items, list):
            continue
        flattened: list[dict[str, Any]] = []
        _flatten_grounded_items(items, flattened)
        by_page[page_number] = flattened
    return by_page


def _flatten_grounded_items(raw_items: list[Any], out_items: list[dict[str, Any]]) -> None:
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        out_items.append(raw_item)
        nested = raw_item.get("items")
        if isinstance(nested, list):
            _flatten_grounded_items(nested, out_items)


def _normalize_item_match_text(value: str) -> str:
    normalized = html.unescape(value)
    normalized = re.sub(r"<\s*br\s*/?\s*>", "\n", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", normalized)
    normalized = re.sub(r"\[([^\]]+)\]\([^)]*\)", r" \1 ", normalized)
    normalized = re.sub(r"[*_~`#>|-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def _score_grounded_item_match(raw_item: dict[str, Any], candidate: dict[str, Any]) -> float:
    raw_type = str(raw_item.get("type") or "")
    candidate_type = str(candidate.get("type") or "")
    raw_text = _normalize_item_match_text(_extract_md(raw_item))
    candidate_text = _normalize_item_match_text(_extract_md(candidate))
    if not raw_text or not candidate_text:
        return 0.0
    if raw_type == candidate_type and raw_text == candidate_text:
        return 1.0
    if raw_type == candidate_type and candidate_text.startswith(raw_text):
        return 0.92
    if raw_type == candidate_type and raw_text in candidate_text:
        return 0.88
    if raw_text == candidate_text:
        return 0.85
    if raw_text in candidate_text or candidate_text in raw_text:
        return 0.72
    raw_tokens = set(raw_text.split())
    candidate_tokens = set(candidate_text.split())
    if not raw_tokens or not candidate_tokens:
        return 0.0
    overlap = len(raw_tokens & candidate_tokens) / max(1, min(len(raw_tokens), len(candidate_tokens)))
    type_bonus = 0.1 if raw_type == candidate_type else 0.0
    return overlap + type_bonus


def _match_grounded_item_override(
    raw_item: dict[str, Any],
    override_candidates: list[dict[str, Any]] | None,
    override_cursor: list[int] | None,
) -> dict[str, Any] | None:
    if not override_candidates or override_cursor is None:
        return None

    best_index = -1
    best_score = 0.0
    start_index = override_cursor[0]
    look_ahead = 12
    upper_bound = min(len(override_candidates), start_index + look_ahead)
    for candidate_index in range(start_index, upper_bound):
        candidate = override_candidates[candidate_index]
        score = _score_grounded_item_match(raw_item, candidate)
        if score > best_score:
            best_score = score
            best_index = candidate_index

    if best_index < 0 or best_score < 0.45:
        return None

    override_cursor[0] = best_index + 1
    return override_candidates[best_index]


def _extract_grounding_payload_from_output(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, dict):
        return None

    layout_pages = output.get("layout_pages")
    if isinstance(layout_pages, list) and layout_pages:
        return {"pages": layout_pages}

    field_citations = output.get("field_citations")
    if isinstance(field_citations, list) and field_citations:
        return {"pages": []}

    return None


def _item_has_display_content(item: dict[str, Any]) -> bool:
    for key in ("md", "markdown", "html", "value"):
        candidate = item.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return True
    return False


def _layout_payload_has_complete_table_content(payload: dict[str, Any]) -> bool:
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        return False

    def walk(items: list[Any]) -> bool:
        for raw_item in items:
            if not isinstance(raw_item, dict):
                continue
            if str(raw_item.get("type") or "") == "table" and not _item_has_display_content(raw_item):
                return False
            nested = raw_item.get("items")
            if isinstance(nested, list) and not walk(nested):
                return False
        return True

    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue
        page_items = raw_page.get("items")
        if isinstance(page_items, list) and not walk(page_items):
            return False

    return True


def _extract_page_markdown_payload(raw_output: Any) -> dict[int, str]:
    if not isinstance(raw_output, dict):
        return {}

    payload_candidates: list[Any] = [raw_output.get("v2_md"), raw_output.get("markdown")]

    for candidate in payload_candidates:
        page_markdown = _extract_page_markdown_from_pages_payload(candidate)
        if page_markdown:
            return page_markdown

    return {}


def _extract_page_markdown_from_output(output: Any) -> dict[int, str]:
    if not isinstance(output, dict):
        return {}

    payload_candidates: list[dict[str, Any]] = []

    layout_pages = output.get("layout_pages")
    if isinstance(layout_pages, list):
        payload_candidates.append({"pages": layout_pages})

    pages = output.get("pages")
    if isinstance(pages, list):
        payload_candidates.append({"pages": pages})

    for candidate in payload_candidates:
        page_markdown = _extract_page_markdown_from_pages_payload(candidate)
        if page_markdown:
            return page_markdown

    return {}


def _extract_page_markdown_from_pages_payload(payload: Any) -> dict[int, str]:
    if not isinstance(payload, dict):
        return {}

    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        return {}

    page_markdown: dict[int, str] = {}
    for page_pos, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, dict):
            continue

        markdown: str | None = None
        for key in ("markdown", "md", "text"):
            candidate = raw_page.get(key)
            if isinstance(candidate, str) and candidate.strip():
                markdown = candidate
                break

        if markdown is None:
            continue

        page_number = _as_int(
            raw_page.get("page_number") or raw_page.get("page"),
            fallback=_as_int(raw_page.get("page_index"), fallback=page_pos) + 1,
        )
        page_markdown[page_number] = markdown

    return page_markdown


def _extract_document_markdown_payload(raw_output: Any) -> str | None:
    if not isinstance(raw_output, dict):
        return None

    for key in ("markdown_full", "markdown"):
        candidate = raw_output.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    return None


def _payload_pipeline_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("pipeline_name") or "").strip()


def _payload_raw_output(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    raw_output = payload.get("raw_output")
    if isinstance(raw_output, dict):
        return raw_output
    return None


def _looks_like_textract_payload(raw_output: dict[str, Any]) -> bool:
    textract_response = raw_output.get("textract_response")
    return isinstance(textract_response, dict) and isinstance(textract_response.get("Blocks"), list)


def _looks_like_azure_payload(raw_output: dict[str, Any]) -> bool:
    raw_pages = raw_output.get("pages")
    if not isinstance(raw_pages, list):
        return False
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue
        if isinstance(raw_page.get("lines"), list) or isinstance(raw_page.get("words"), list):
            return True
    return False


def _looks_like_llamaparse_payload(raw_output: dict[str, Any], pipeline_name: str) -> bool:
    if isinstance(raw_output.get("v2_grounded_items"), list) or isinstance(raw_output.get("grounded_items"), list):
        return True
    lowered = pipeline_name.lower()
    return any(token in lowered for token in ("llamaparse", "agentic", "ours_"))


def _infer_granular_provider_kind(payload: Any) -> Literal["llamaparse", "textract", "azure"] | None:
    raw_output = _payload_raw_output(payload)
    if raw_output is None:
        return None

    pipeline_name = _payload_pipeline_name(payload)
    if _looks_like_textract_payload(raw_output):
        return "textract"
    if _looks_like_azure_payload(raw_output):
        return "azure"
    if _looks_like_llamaparse_payload(raw_output, pipeline_name):
        return "llamaparse"
    return None


def _granular_bbox_to_page(
    bbox: Any,
    *,
    page_width: float,
    page_height: float,
) -> GroundingBbox | None:
    if not hasattr(bbox, "x") and not isinstance(bbox, dict):
        return None

    if isinstance(bbox, dict):
        x = bbox.get("x")
        y = bbox.get("y")
        w = bbox.get("w")
        h = bbox.get("h")
    else:
        x = getattr(bbox, "x", None)
        y = getattr(bbox, "y", None)
        w = getattr(bbox, "w", None)
        h = getattr(bbox, "h", None)

    if any(value is None for value in (x, y, w, h)):
        return None

    normalized = GroundingBbox(x=_as_float(x), y=_as_float(y), w=_as_float(w), h=_as_float(h))
    if _bbox_looks_normalized(normalized):
        return _scale_bbox_to_page(normalized, page_width, page_height)
    return normalized


def _collect_bbox_payloads(raw_bboxes: Any) -> list[dict[str, Any]]:
    if isinstance(raw_bboxes, dict):
        if all(key in raw_bboxes for key in ("x", "y", "w", "h")):
            return [raw_bboxes]
        return []

    if not isinstance(raw_bboxes, list):
        return []

    candidates: list[dict[str, Any]] = []
    for raw_bbox in raw_bboxes:
        if isinstance(raw_bbox, dict) and all(key in raw_bbox for key in ("x", "y", "w", "h")):
            candidates.append(raw_bbox)

    return candidates


def _merge_bbox_payloads(raw_bboxes: Any) -> dict[str, Any] | None:
    candidates = _collect_bbox_payloads(raw_bboxes)
    if not candidates:
        return None

    min_x = min(_as_float(candidate.get("x")) for candidate in candidates)
    min_y = min(_as_float(candidate.get("y")) for candidate in candidates)
    max_x = max(_as_float(candidate.get("x")) + _as_float(candidate.get("w")) for candidate in candidates)
    max_y = max(_as_float(candidate.get("y")) + _as_float(candidate.get("h")) for candidate in candidates)
    return {"x": min_x, "y": min_y, "w": max(0.0, max_x - min_x), "h": max(0.0, max_y - min_y)}


def _normalize_bbox_payloads_to_page(
    raw_bboxes: Any,
    *,
    page_width: float,
    page_height: float,
) -> list[GroundingBbox]:
    normalized_bboxes: list[GroundingBbox] = []
    for raw_bbox in _collect_bbox_payloads(raw_bboxes):
        normalized_bbox = _normalize_bbox(raw_bbox)
        if normalized_bbox is None:
            continue
        normalized_bboxes.append(
            _scale_bbox_to_page(normalized_bbox, page_width, page_height)
            if _bbox_looks_normalized(normalized_bbox)
            else normalized_bbox
        )
    return normalized_bboxes


def _merge_grounding_bboxes(bboxes: list[GroundingBbox]) -> GroundingBbox | None:
    if not bboxes:
        return None

    min_x = min(bbox.x for bbox in bboxes)
    min_y = min(bbox.y for bbox in bboxes)
    max_x = max(bbox.x + bbox.w for bbox in bboxes)
    max_y = max(bbox.y + bbox.h for bbox in bboxes)
    return GroundingBbox(x=min_x, y=min_y, w=max(0.0, max_x - min_x), h=max(0.0, max_y - min_y))


def _coerce_cell_text(source_cell: Any) -> str:
    if isinstance(source_cell, str):
        return source_cell
    if isinstance(source_cell, dict):
        for key in ("value", "md", "text", "html"):
            candidate = source_cell.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return ""


def _extract_llamaparse_cell_layers(
    raw_output: dict[str, Any],
    *,
    page_dimensions: dict[int, tuple[float, float]],
) -> dict[int, list[GroundingGranularUnit]]:
    grounded_pages = raw_output.get("v2_grounded_items", raw_output.get("grounded_items"))
    if not isinstance(grounded_pages, list):
        return {}

    pages: dict[int, list[GroundingGranularUnit]] = {}
    for page_payload in grounded_pages:
        if not isinstance(page_payload, dict) or page_payload.get("success") is False:
            continue

        page_number = _as_int(page_payload.get("page_number"), fallback=0)
        if page_number <= 0:
            continue

        raw_items = page_payload.get("items")
        if not isinstance(raw_items, list):
            continue

        page_units = pages.setdefault(page_number, [])
        page_width, page_height = page_dimensions.get(page_number, (1.0, 1.0))
        stack: list[tuple[int, dict[str, Any], str]] = []
        for item_index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                continue
            stack.append((item_index, raw_item, f"v2_grounded_items[{page_number}].items[{item_index}]"))

        while stack:
            item_index, raw_item, item_source_path = stack.pop()
            nested_items = raw_item.get("items")
            if isinstance(nested_items, list):
                for nested_index, nested_item in enumerate(nested_items):
                    if isinstance(nested_item, dict):
                        stack.append(
                            (
                                item_index,
                                nested_item,
                                f"{item_source_path}.items[{nested_index}]",
                            )
                        )

            grounding = raw_item.get("grounding")
            if not isinstance(grounding, dict):
                continue

            source_rows = raw_item.get("rows")
            grounded_rows = grounding.get("rows")
            if not isinstance(source_rows, list) or not isinstance(grounded_rows, list):
                continue

            for row_index, (source_row, grounded_row) in enumerate(zip(source_rows, grounded_rows, strict=False)):
                if not isinstance(source_row, list) or not isinstance(grounded_row, list):
                    continue

                for column_index, (source_cell, grounded_cell) in enumerate(
                    zip(source_row, grounded_row, strict=False)
                ):
                    if not isinstance(grounded_cell, dict):
                        continue

                    cell_bboxes = _normalize_bbox_payloads_to_page(
                        grounded_cell.get("bbox"),
                        page_width=page_width,
                        page_height=page_height,
                    )
                    if not cell_bboxes:
                        cell_lines = grounded_cell.get("lines")
                        if isinstance(cell_lines, list):
                            cell_bboxes = _normalize_bbox_payloads_to_page(
                                [line.get("bbox") for line in cell_lines if isinstance(line, dict)],
                                page_width=page_width,
                                page_height=page_height,
                            )
                    if not cell_bboxes:
                        continue

                    bbox = _merge_grounding_bboxes(cell_bboxes)
                    if bbox is None:
                        continue

                    row_span = grounded_cell.get("row_span")
                    column_span = grounded_cell.get("column_span")
                    page_units.append(
                        GroundingGranularUnit(
                            unit_id=f"p{page_number}-table-{item_index}-cell-{row_index}-{column_index}",
                            granularity="cell",
                            order_index=len(page_units),
                            text=_coerce_cell_text(source_cell),
                            bbox=bbox,
                            bboxes=cell_bboxes,
                            row_index=row_index,
                            column_index=column_index,
                            row_span=_as_int(row_span, fallback=1) if row_span is not None else None,
                            column_span=_as_int(column_span, fallback=1) if column_span is not None else None,
                            source_path=f"{item_source_path}.grounding.rows[{row_index}][{column_index}]",
                            provider="llamaparse",
                        )
                    )

    return pages


def _extract_textract_cell_text(
    block: dict[str, Any],
    *,
    block_by_id: dict[str, dict[str, Any]],
) -> str:
    relationships = block.get("Relationships")
    if not isinstance(relationships, list):
        return ""

    child_ids: list[str] = []
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        if relationship.get("Type") != "CHILD":
            continue
        ids = relationship.get("Ids")
        if isinstance(ids, list):
            child_ids.extend(str(child_id) for child_id in ids)

    texts: list[str] = []
    for child_id in child_ids:
        child_block = block_by_id.get(child_id)
        if not isinstance(child_block, dict):
            continue
        child_type = str(child_block.get("BlockType") or "")
        if child_type == "WORD":
            text = str(child_block.get("Text") or "").strip()
            if text:
                texts.append(text)
        elif child_type == "SELECTION_ELEMENT" and child_block.get("SelectionStatus") == "SELECTED":
            texts.append("[x]")

    return " ".join(texts)


def _coerce_textract_cell_index(value: Any) -> int | None:
    if value is None:
        return None
    return max(_as_int(value, fallback=1) - 1, 0)


def _extract_textract_cell_layers(
    textract_response: dict[str, Any],
    *,
    page_dimensions: dict[int, tuple[float, float]],
) -> dict[int, list[GroundingGranularUnit]]:
    blocks = textract_response.get("Blocks")
    if not isinstance(blocks, list):
        return {}

    pages: dict[int, list[GroundingGranularUnit]] = {}
    block_by_id = {
        str(block.get("Id")): block for block in blocks if isinstance(block, dict) and block.get("Id") is not None
    }
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict) or str(block.get("BlockType") or "") != "CELL":
            continue

        geometry = block.get("Geometry")
        bbox_payload = geometry.get("BoundingBox") if isinstance(geometry, dict) else None
        if not isinstance(bbox_payload, dict):
            continue

        normalized_bbox = _normalize_bbox(
            {
                "x": bbox_payload.get("Left"),
                "y": bbox_payload.get("Top"),
                "w": bbox_payload.get("Width"),
                "h": bbox_payload.get("Height"),
            }
        )
        if normalized_bbox is None:
            continue

        page_number = _as_int(block.get("Page"), fallback=1)
        page_width, page_height = page_dimensions.get(page_number, (1.0, 1.0))
        bbox = (
            _scale_bbox_to_page(normalized_bbox, page_width, page_height)
            if _bbox_looks_normalized(normalized_bbox)
            else normalized_bbox
        )
        page_units = pages.setdefault(page_number, [])
        row_index = block.get("RowIndex")
        column_index = block.get("ColumnIndex")
        row_span = block.get("RowSpan")
        column_span = block.get("ColumnSpan")
        page_units.append(
            GroundingGranularUnit(
                unit_id=str(block.get("Id") or f"p{page_number}-cell-{block_index}"),
                granularity="cell",
                order_index=block_index,
                text=_extract_textract_cell_text(block, block_by_id=block_by_id),
                bbox=bbox,
                bboxes=[bbox],
                row_index=_coerce_textract_cell_index(row_index),
                column_index=_coerce_textract_cell_index(column_index),
                row_span=_as_int(row_span, fallback=1) if row_span is not None else None,
                column_span=_as_int(column_span, fallback=1) if column_span is not None else None,
                source_path=f"Blocks[{block_index}]",
                provider="textract",
            )
        )

    return pages


def _build_llamaparse_granular_pages(raw_output: dict[str, Any]) -> list[_GranularPayloadPage]:
    grounded_pages = raw_output.get("v2_grounded_items", raw_output.get("grounded_items"))
    if not isinstance(grounded_pages, list):
        return []

    pages: list[_GranularPayloadPage] = []
    for page_payload in grounded_pages:
        if not isinstance(page_payload, dict) or page_payload.get("success") is False:
            continue

        page_number = _as_int(page_payload.get("page_number"), fallback=0)
        page_width = _as_float(page_payload.get("page_width"), fallback=0.0)
        page_height = _as_float(page_payload.get("page_height"), fallback=0.0)
        if page_number <= 0 or page_width <= 0 or page_height <= 0:
            continue

        raw_items = page_payload.get("items")
        if not isinstance(raw_items, list):
            continue

        line_units: list[_GranularPayloadUnit] = []
        word_units: list[_GranularPayloadUnit] = []
        for order_index, line_context in enumerate(_iter_llamaparse_line_contexts(raw_items)):
            line_text = str(line_context.get("text") or "")
            line_bbox = line_context.get("bbox")
            if not line_text or not isinstance(line_bbox, dict):
                continue

            normalized_line_bbox = _normalize_grounded_bbox(
                line_bbox,
                page_width=page_width,
                page_height=page_height,
            )
            if normalized_line_bbox is None:
                continue

            line_units.append(
                _GranularPayloadUnit(
                    text=line_text,
                    bbox=normalized_line_bbox,
                    order_index=order_index,
                )
            )
            word_units.extend(
                _build_llamaparse_word_units(
                    line_context,
                    page_width=page_width,
                    page_height=page_height,
                    order_index=order_index,
                )
            )

        deduped_lines = _dedupe_granular_units(line_units)
        deduped_words = _dedupe_granular_units(word_units)
        if not deduped_lines and not deduped_words:
            continue

        pages.append(
            _GranularPayloadPage(
                page_number=page_number,
                lines=deduped_lines,
                words=deduped_words,
            )
        )

    return pages


def _iter_llamaparse_line_contexts(raw_nodes: list[Any]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for raw_node in raw_nodes:
        contexts.extend(_collect_llamaparse_line_contexts(raw_node))
    return contexts


def _collect_llamaparse_line_contexts(raw_node: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_node, dict):
        return []

    contexts: list[dict[str, Any]] = []
    grounding = raw_node.get("grounding")
    if isinstance(grounding, dict):
        source_text = _resolve_llamaparse_grounding_source_text(raw_node, grounding)
        raw_lines = grounding.get("lines")
        if isinstance(raw_lines, list):
            contexts.extend(_build_llamaparse_line_context_entries(source_text, raw_lines))

        raw_rows = grounding.get("rows")
        source_rows = raw_node.get("rows")
        if isinstance(raw_rows, list) and isinstance(source_rows, list):
            contexts.extend(_build_llamaparse_table_line_context_entries(source_rows, raw_rows))

    child_items = raw_node.get("items")
    if isinstance(child_items, list):
        for child in child_items:
            contexts.extend(_collect_llamaparse_line_contexts(child))

    return contexts


def _build_llamaparse_line_context_entries(source_text: str, raw_lines: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            continue

        line_span = _coerce_span(raw_line.get("span"))
        line_bbox = raw_line.get("bbox")
        if line_span is None or not isinstance(line_bbox, dict):
            continue

        line_text = _normalize_llamaparse_grounded_text(_slice_span_text(source_text, line_span))
        if not line_text:
            continue

        entries.append(
            {
                "text": line_text,
                "bbox": line_bbox,
                "line_span": line_span,
                "raw_words": raw_line.get("words") if isinstance(raw_line.get("words"), list) else [],
                "source_text": source_text,
            }
        )

    return entries


def _build_llamaparse_table_line_context_entries(
    source_rows: list[Any],
    raw_rows: list[Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source_row, grounding_row in zip(source_rows, raw_rows, strict=False):
        if not isinstance(source_row, list) or not isinstance(grounding_row, list):
            continue
        for source_cell, grounding_cell in zip(source_row, grounding_row, strict=False):
            if not isinstance(grounding_cell, dict):
                continue

            cell_text = _coerce_cell_text(source_cell)
            if not cell_text:
                continue

            cell_lines = grounding_cell.get("lines")
            if isinstance(cell_lines, list):
                entries.extend(_build_llamaparse_line_context_entries(cell_text, cell_lines))

    return entries


def _resolve_llamaparse_grounding_source_text(raw_node: dict[str, Any], grounding: dict[str, Any]) -> str:
    source_name = grounding.get("source")
    if source_name == "caption":
        source_text = raw_node.get("caption")
    elif source_name == "value":
        source_text = raw_node.get("value")
    else:
        source_text = raw_node.get("md")

    if isinstance(source_text, str) and source_text:
        return source_text

    for candidate_key in ("value", "md", "caption", "html"):
        candidate = raw_node.get(candidate_key)
        if isinstance(candidate, str) and candidate:
            return candidate

    return ""


def _build_llamaparse_word_units(
    line_context: dict[str, Any],
    *,
    page_width: float,
    page_height: float,
    order_index: int,
) -> list[_GranularPayloadUnit]:
    source_text = str(line_context.get("source_text") or "")
    line_span = _coerce_span(line_context.get("line_span"))
    raw_words = line_context.get("raw_words")
    if not source_text or line_span is None or not isinstance(raw_words, list):
        return []

    units: list[_GranularPayloadUnit] = []
    for token_start, token_end in _iter_token_spans(source_text, line_span):
        matching_word_boxes: list[dict[str, Any]] = []
        for raw_word in raw_words:
            if not isinstance(raw_word, dict):
                continue
            word_span = _coerce_span(raw_word.get("span"))
            word_bbox = raw_word.get("bbox")
            if word_span is None or not isinstance(word_bbox, dict):
                continue
            if word_span[1] <= token_start or word_span[0] >= token_end:
                continue
            matching_word_boxes.append(word_bbox)

        if not matching_word_boxes:
            continue

        word_text = _normalize_llamaparse_grounded_text(source_text[token_start:token_end])
        if not word_text:
            continue

        merged_bbox = _merge_llamaparse_bboxes(matching_word_boxes)
        normalized_bbox = _normalize_grounded_bbox(
            merged_bbox,
            page_width=page_width,
            page_height=page_height,
        )
        if normalized_bbox is None:
            continue

        units.append(
            _GranularPayloadUnit(
                text=word_text,
                bbox=normalized_bbox,
                order_index=order_index,
            )
        )

    return units


def _coerce_span(raw_span: Any) -> tuple[int, int] | None:
    if not isinstance(raw_span, list | tuple) or len(raw_span) != 2:
        return None
    try:
        start = int(raw_span[0])
        end = int(raw_span[1])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return (start, end)


def _slice_span_text(source_text: str, span: tuple[int, int]) -> str:
    start = max(span[0], 0)
    end = min(span[1], len(source_text))
    if end <= start:
        return ""
    return source_text[start:end]


def _normalize_llamaparse_grounded_text(text: str) -> str:
    normalized = text.replace("<br/>", "\n").replace("<br />", "\n")
    if "<" in normalized and ">" in normalized:
        normalized = _extract_text_from_html(normalized)
    return normalized.strip()


def _extract_text_from_html(text: str) -> str:
    normalized = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    return html.unescape(normalized)


def _iter_token_spans(source_text: str, line_span: tuple[int, int]) -> list[tuple[int, int]]:
    line_text = _slice_span_text(source_text, line_span)
    return [
        (line_span[0] + match.start(), line_span[0] + match.end())
        for match in re.finditer(r"\S+", line_text, flags=re.UNICODE)
    ]


def _merge_llamaparse_bboxes(raw_bboxes: list[dict[str, Any]]) -> dict[str, float]:
    x1 = min(_as_float(bbox.get("x")) for bbox in raw_bboxes)
    y1 = min(_as_float(bbox.get("y")) for bbox in raw_bboxes)
    x2 = max(_as_float(bbox.get("x")) + _as_float(bbox.get("w")) for bbox in raw_bboxes)
    y2 = max(_as_float(bbox.get("y")) + _as_float(bbox.get("h")) for bbox in raw_bboxes)
    return {"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)}


def _dedupe_granular_units(units: list[_GranularPayloadUnit]) -> list[_GranularPayloadUnit]:
    deduped: list[_GranularPayloadUnit] = []
    seen: set[tuple[str, float, float, float, float]] = set()
    for unit in units:
        key = (
            unit.text,
            round(unit.bbox["x"], 6),
            round(unit.bbox["y"], 6),
            round(unit.bbox["w"], 6),
            round(unit.bbox["h"], 6),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(unit)
    return deduped


def _normalize_grounded_bbox(
    bbox_payload: Any,
    *,
    page_width: float,
    page_height: float,
) -> dict[str, float] | None:
    if not isinstance(bbox_payload, dict) or page_width <= 0 or page_height <= 0:
        return None

    x = bbox_payload.get("x")
    y = bbox_payload.get("y")
    w = bbox_payload.get("w")
    h = bbox_payload.get("h")
    if not all(isinstance(value, (int, float)) for value in (x, y, w, h)):
        return None

    return {
        "x": _as_float(x) / page_width,
        "y": _as_float(y) / page_height,
        "w": _as_float(w) / page_width,
        "h": _as_float(h) / page_height,
    }


def _build_textract_granular_pages(raw_output: dict[str, Any]) -> list[_GranularPayloadPage]:
    textract_response = raw_output.get("textract_response")
    if not isinstance(textract_response, dict):
        return []

    blocks = textract_response.get("Blocks")
    if not isinstance(blocks, list):
        return []

    pages: dict[int, _GranularPayloadPage] = {}
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue

        block_type = str(block.get("BlockType") or "")
        if block_type not in {"LINE", "WORD"}:
            continue

        geometry = block.get("Geometry")
        bbox = geometry.get("BoundingBox") if isinstance(geometry, dict) else None
        if not isinstance(bbox, dict):
            continue

        text = str(block.get("Text") or "")
        if not text:
            continue

        page_number = _as_int(block.get("Page"), fallback=1)
        unit = _GranularPayloadUnit(
            text=text,
            bbox={
                "x": _as_float(bbox.get("Left")),
                "y": _as_float(bbox.get("Top")),
                "w": _as_float(bbox.get("Width")),
                "h": _as_float(bbox.get("Height")),
            },
            order_index=block_index,
            unit_id=str(block.get("Id") or f"textract-{block_type.lower()}-{block_index}"),
        )
        page = pages.setdefault(page_number, _GranularPayloadPage(page_number=page_number, lines=[], words=[]))
        if block_type == "LINE":
            page.lines.append(unit)
        else:
            page.words.append(unit)

    return [pages[page_number] for page_number in sorted(pages)]


def _build_azure_di_granular_pages(raw_output: dict[str, Any]) -> list[_GranularPayloadPage]:
    raw_pages = raw_output.get("pages")
    if not isinstance(raw_pages, list):
        return []

    granular_pages: list[_GranularPayloadPage] = []
    for page_data in raw_pages:
        if not isinstance(page_data, dict):
            continue

        page_number = _as_int(page_data.get("page_number"), fallback=1)
        page_width = _as_float(page_data.get("width"), fallback=1.0)
        page_height = _as_float(page_data.get("height"), fallback=1.0)
        if page_width <= 0 or page_height <= 0:
            continue

        line_units = _build_azure_di_granular_units(
            page_data.get("lines"),
            page_width=page_width,
            page_height=page_height,
            text_key="content",
        )
        word_units = _build_azure_di_granular_units(
            page_data.get("words"),
            page_width=page_width,
            page_height=page_height,
            text_key="content",
        )
        if not line_units and not word_units:
            continue

        granular_pages.append(
            _GranularPayloadPage(
                page_number=page_number,
                lines=line_units,
                words=word_units,
            )
        )

    return granular_pages


def _build_azure_di_granular_units(
    raw_units: Any,
    *,
    page_width: float,
    page_height: float,
    text_key: str,
) -> list[_GranularPayloadUnit]:
    if not isinstance(raw_units, list):
        return []

    units: list[_GranularPayloadUnit] = []
    for index, raw_unit in enumerate(raw_units):
        if not isinstance(raw_unit, dict):
            continue

        polygon = raw_unit.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 8:
            continue

        text = str(raw_unit.get(text_key) or "")
        if not text:
            continue

        x, y, w, h = _polygon_to_normalized_xywh(
            polygon,
            page_width=page_width,
            page_height=page_height,
        )
        units.append(
            _GranularPayloadUnit(
                text=text,
                bbox={"x": x, "y": y, "w": w, "h": h},
                order_index=index,
            )
        )

    return units


def _polygon_to_normalized_xywh(
    polygon: list[float],
    *,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    xs = [_as_float(value) / page_width for value in polygon[0::2]]
    ys = [_as_float(value) / page_height for value in polygon[1::2]]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def _build_payload_granular_pages(payload: Any) -> tuple[dict[int, _GranularPayloadPage], str | None]:
    provider_kind = _infer_granular_provider_kind(payload)
    raw_output = _payload_raw_output(payload)
    if provider_kind is None or raw_output is None:
        return {}, None

    if provider_kind == "llamaparse":
        pages = _build_llamaparse_granular_pages(raw_output)
    elif provider_kind == "textract":
        pages = _build_textract_granular_pages(raw_output)
    else:
        pages = _build_azure_di_granular_pages(raw_output)

    return ({page.page_number: page for page in pages}, _payload_pipeline_name(payload) or provider_kind)


def _extract_cell_layers_from_payload(
    payload: Any,
    *,
    page_dimensions: dict[int, tuple[float, float]],
) -> tuple[dict[int, list[GroundingGranularUnit]], bool, str | None, str | None]:
    provider_kind = _infer_granular_provider_kind(payload)
    raw_output = _payload_raw_output(payload)
    if provider_kind is None or raw_output is None:
        return {}, False, None, None

    source = _payload_pipeline_name(payload) or provider_kind
    if provider_kind == "llamaparse":
        return _extract_llamaparse_cell_layers(raw_output, page_dimensions=page_dimensions), True, source, None
    if provider_kind == "textract":
        textract_response = raw_output.get("textract_response")
        if isinstance(textract_response, dict):
            return _extract_textract_cell_layers(textract_response, page_dimensions=page_dimensions), True, source, None
        return {}, True, source, None

    return {}, False, source, "Azure DI raw output does not preserve exact cell polygons."


def _build_granular_layers(
    pages: list[GroundingPage],
    raw_payload: dict[str, Any] | None,
    result_payload: dict[str, Any] | None,
) -> dict[int, list[GroundingGranularLayer]]:
    page_dimensions = {page.page_number: (page.page_width, page.page_height) for page in pages}
    page_numbers = sorted(page_dimensions)

    granular_pages: dict[int, _GranularPayloadPage] = {}
    granular_source = None
    for payload in (result_payload, raw_payload):
        pages_by_number, source = _build_payload_granular_pages(payload)
        if not pages_by_number:
            continue
        granular_pages = pages_by_number
        granular_source = source
        break

    cell_units_by_page: dict[int, list[GroundingGranularUnit]] = {}
    cell_supported = False
    cell_source: str | None = None
    cell_reason: str | None = None
    for payload in (result_payload, raw_payload):
        cell_units, supported, source, reason = _extract_cell_layers_from_payload(
            payload,
            page_dimensions=page_dimensions,
        )
        if source is None and not supported and reason is None:
            continue
        cell_units_by_page = cell_units
        cell_supported = supported
        cell_source = source
        cell_reason = reason
        break

    granular_layers_by_page: dict[int, list[GroundingGranularLayer]] = {}
    for page_number in page_numbers:
        page_width, page_height = page_dimensions[page_number]
        page_layers: list[GroundingGranularLayer] = []

        if granular_source is not None:
            granular_page = granular_pages.get(page_number)
            if granular_page is None:
                page_layers.append(
                    GroundingGranularLayer(
                        granularity="line",
                        availability="empty",
                        source=granular_source,
                    )
                )
                page_layers.append(
                    GroundingGranularLayer(
                        granularity="word",
                        availability="empty",
                        source=granular_source,
                    )
                )
            else:
                line_units: list[GroundingGranularUnit] = []
                for index, unit in enumerate(granular_page.lines):
                    bbox = _granular_bbox_to_page(unit.bbox, page_width=page_width, page_height=page_height)
                    if bbox is None:
                        continue
                    line_units.append(
                        GroundingGranularUnit(
                            unit_id=unit.unit_id or f"p{page_number}-line-{index}",
                            granularity="line",
                            order_index=unit.order_index,
                            text=unit.text,
                            bbox=bbox,
                            source_path=f"{granular_source}.lines[{index}]",
                            provider=granular_source,
                        )
                    )

                word_units: list[GroundingGranularUnit] = []
                for index, unit in enumerate(granular_page.words):
                    bbox = _granular_bbox_to_page(unit.bbox, page_width=page_width, page_height=page_height)
                    if bbox is None:
                        continue
                    word_units.append(
                        GroundingGranularUnit(
                            unit_id=unit.unit_id or f"p{page_number}-word-{index}",
                            granularity="word",
                            order_index=unit.order_index,
                            text=unit.text,
                            bbox=bbox,
                            source_path=f"{granular_source}.words[{index}]",
                            provider=granular_source,
                        )
                    )
                page_layers.append(
                    GroundingGranularLayer(
                        granularity="line",
                        availability="available" if line_units else "empty",
                        units=line_units,
                        source=granular_source,
                    )
                )
                page_layers.append(
                    GroundingGranularLayer(
                        granularity="word",
                        availability="available" if word_units else "empty",
                        units=word_units,
                        source=granular_source,
                    )
                )
        else:
            page_layers.append(
                GroundingGranularLayer(
                    granularity="line",
                    availability="unavailable",
                    reason="No provider granular adapter was available for this document.",
                )
            )
            page_layers.append(
                GroundingGranularLayer(
                    granularity="word",
                    availability="unavailable",
                    reason="No provider granular adapter was available for this document.",
                )
            )

        if cell_supported:
            cell_units = cell_units_by_page.get(page_number, [])
            page_layers.append(
                GroundingGranularLayer(
                    granularity="cell",
                    availability="available" if cell_units else "empty",
                    units=cell_units,
                    source=cell_source,
                )
            )
        else:
            page_layers.append(
                GroundingGranularLayer(
                    granularity="cell",
                    availability="unavailable",
                    reason=cell_reason
                    or "Cell overlays are not available for this provider because exact cell polygons are missing.",
                    source=cell_source,
                )
            )

        granular_layers_by_page[page_number] = page_layers

    return granular_layers_by_page


def _extract_v2_items_payload(
    doc: IndexedDocumentInternal,
    raw_payload: dict[str, Any] | None,
    result_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], Literal["v2_items", "raw", "result"], Literal["normalized", "legacy"]]:
    result_normalized: dict[str, Any] | None = None
    if isinstance(result_payload, dict):
        result_normalized = _extract_grounding_payload_from_output(result_payload.get("output"))
        if result_normalized is not None and _layout_payload_has_complete_table_content(result_normalized):
            return result_normalized, "result", "normalized"

    raw_normalized: dict[str, Any] | None = None
    if isinstance(raw_payload, dict):
        raw_normalized = _extract_grounding_payload_from_output(raw_payload.get("output"))
        if raw_normalized is not None and _layout_payload_has_complete_table_content(raw_normalized):
            return raw_normalized, "raw", "normalized"

    if doc.v2_items_path is not None:
        display_payload = _read_json(doc.v2_items_path)
        if isinstance(raw_payload, dict):
            raw_output = raw_payload.get("raw_output")
            if isinstance(raw_output, dict):
                grounded_pages = raw_output.get("v2_grounded_items")
                if isinstance(grounded_pages, list):
                    return _merge_llamaparse_items_payload(display_payload, grounded_pages), "v2_items", "legacy"
        return display_payload, "v2_items", "legacy"

    if isinstance(raw_payload, dict):
        extracted = _extract_grounding_payload_from_raw_output(raw_payload.get("raw_output"))
        if extracted is not None:
            return extracted, "raw", "legacy"

    if isinstance(result_payload, dict):
        extracted = _extract_grounding_payload_from_raw_output(result_payload.get("raw_output"))
        if extracted is not None:
            return extracted, "result", "legacy"

    if result_normalized is not None:
        return result_normalized, "result", "normalized"

    if raw_normalized is not None:
        return raw_normalized, "raw", "normalized"

    raise ValueError(f"No grounding payload found for {doc.doc_id}")


def _select_markdown_payload(
    doc: IndexedDocumentInternal,
    selected_grounding_source: Literal["v2_items", "raw", "result"],
    raw_payload: dict[str, Any] | None,
    result_payload: dict[str, Any] | None,
) -> tuple[dict[int, str], str | None, Literal["sidecar_md", "raw", "result"] | None]:
    if doc.markdown_path is not None:
        try:
            document_markdown = doc.markdown_path.read_text(encoding="utf-8")
        except Exception:
            document_markdown = None
        else:
            if document_markdown is not None and document_markdown.strip():
                return {}, document_markdown, "sidecar_md"

    if doc.markdown_json_path is not None:
        try:
            markdown_json_payload = _read_json(doc.markdown_json_path)
        except Exception:
            markdown_json_payload = None
        else:
            page_markdown = _extract_page_markdown_from_pages_payload(markdown_json_payload)
            if page_markdown:
                return page_markdown, None, "sidecar_md"

    source_payloads: list[tuple[Literal["raw", "result"], dict[str, Any] | None]]
    if selected_grounding_source == "result":
        source_payloads = [("result", result_payload), ("raw", raw_payload)]
    else:
        source_payloads = [("raw", raw_payload), ("result", result_payload)]

    for source_name, payload in source_payloads:
        if not isinstance(payload, dict):
            continue

        output = payload.get("output")
        page_markdown = _extract_page_markdown_from_output(output)
        document_markdown = _extract_document_markdown_payload(output)
        if page_markdown or document_markdown:
            return page_markdown, document_markdown, source_name

        raw_output = payload.get("raw_output")
        page_markdown = _extract_page_markdown_payload(raw_output)
        document_markdown = _extract_document_markdown_payload(raw_output)
        if page_markdown or document_markdown:
            return page_markdown, document_markdown, source_name

    return {}, None, None


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_bbox(raw: Any) -> GroundingBbox | None:
    if isinstance(raw, list) and len(raw) == 4:
        raw = {"x": raw[0], "y": raw[1], "w": raw[2], "h": raw[3]}

    if not isinstance(raw, dict):
        return None

    x = raw.get("x")
    y = raw.get("y")
    w = raw.get("w")
    h = raw.get("h")
    if any(val is None for val in [x, y, w, h]):
        return None

    start_index = raw.get("start_index")
    if start_index is None:
        start_index = raw.get("startIndex")

    end_index = raw.get("end_index")
    if end_index is None:
        end_index = raw.get("endIndex")

    return GroundingBbox(
        x=_as_float(x),
        y=_as_float(y),
        w=_as_float(w),
        h=_as_float(h),
        label=raw.get("label") if isinstance(raw.get("label"), str) else None,
        confidence=_as_float(raw.get("confidence"), fallback=0.0) if raw.get("confidence") is not None else None,
        start_index=_as_int(start_index, fallback=0) if start_index is not None else None,
        end_index=_as_int(end_index, fallback=0) if end_index is not None else None,
    )


def _extract_md(item: dict[str, Any]) -> str:
    md = item.get("md")
    if isinstance(md, str) and md.strip():
        return md

    markdown = item.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        return markdown

    html = item.get("html")
    if isinstance(html, str) and html.strip():
        return html

    value = item.get("value")
    if isinstance(value, str):
        return value

    return ""


def _bbox_looks_normalized(box: GroundingBbox) -> bool:
    tolerance = 1.01
    return (
        box.x >= -0.01
        and box.y >= -0.01
        and box.w >= 0.0
        and box.h >= 0.0
        and box.x <= tolerance
        and box.y <= tolerance
        and box.w <= tolerance
        and box.h <= tolerance
    )


def _scale_bbox_to_page(box: GroundingBbox, page_width: float, page_height: float) -> GroundingBbox:
    if not _bbox_looks_normalized(box):
        return box

    safe_width = page_width if page_width > 0 else 1.0
    safe_height = page_height if page_height > 0 else 1.0
    return box.model_copy(
        update={
            "x": box.x * safe_width,
            "y": box.y * safe_height,
            "w": box.w * safe_width,
            "h": box.h * safe_height,
        }
    )


def _extract_field_citation_items(
    result_payload: dict[str, Any] | None,
    pages: list[GroundingPage],
) -> dict[int, list[GroundingItem]]:
    if not isinstance(result_payload, dict):
        return {}

    output = result_payload.get("output")
    if not isinstance(output, dict):
        return {}

    field_citations = output.get("field_citations")
    if not isinstance(field_citations, list):
        return {}

    page_sizes = {page.page_number: (page.page_width, page.page_height) for page in pages}
    counters = {page.page_number: len(page.items) for page in pages}
    items_by_page: dict[int, list[GroundingItem]] = {}

    for citation_index, citation in enumerate(field_citations):
        if not isinstance(citation, dict):
            continue

        page_number = _as_int(citation.get("page"), fallback=1)
        page_width, page_height = page_sizes.get(page_number, (0.0, 0.0))
        raw_bbox = citation.get("bbox")
        normalized_bbox = _normalize_bbox(raw_bbox)
        if normalized_bbox is None:
            continue

        bbox = _scale_bbox_to_page(normalized_bbox, page_width, page_height)
        field_path = citation.get("field_path")
        field_path_text = field_path if isinstance(field_path, str) and field_path else f"citation[{citation_index}]"
        reference_text = citation.get("reference_text")
        matching_text = (
            citation.get("metadata", {}).get("matching_text") if isinstance(citation.get("metadata"), dict) else None
        )
        display_text = (
            reference_text
            if isinstance(reference_text, str) and reference_text.strip()
            else matching_text
            if isinstance(matching_text, str) and matching_text.strip()
            else field_path_text
        )

        item_index = counters.get(page_number, 0)
        counters[page_number] = item_index + 1
        items_by_page.setdefault(page_number, []).append(
            GroundingItem(
                item_id=f"p{page_number}-extract-citation-{citation_index}",
                item_index=item_index,
                page_number=page_number,
                depth=0,
                type="extract_field",
                md=f"**{field_path_text}**\n\n{display_text}",
                value=display_text,
                source_path=f"field_citations.{citation_index}",
                raw_payload=citation,
                bboxes=[bbox.model_copy(update={"label": "extract_field"})],
            )
        )

    return items_by_page


def _extract_item_bboxes(
    raw_item: dict[str, Any],
    page_width: float,
    page_height: float,
    coordinates_are_normalized: bool,
) -> list[GroundingBbox]:
    bboxes: list[GroundingBbox] = []

    raw_layout_segments = raw_item.get("layout_segments")
    if not isinstance(raw_layout_segments, list):
        raw_layout_segments = raw_item.get("layoutAwareBbox")

    if isinstance(raw_layout_segments, list):
        for raw_bbox in raw_layout_segments:
            normalized = _normalize_bbox(raw_bbox)
            if normalized is None:
                continue
            bboxes.append(
                _scale_bbox_to_page(normalized, page_width, page_height) if coordinates_are_normalized else normalized
            )

    if bboxes:
        return bboxes

    raw_bbox = raw_item.get("bbox")
    if raw_bbox is None:
        raw_bbox = raw_item.get("bBox")

    bbox_candidates: list[Any]
    if isinstance(raw_bbox, list):
        bbox_candidates = raw_bbox
    elif isinstance(raw_bbox, dict):
        bbox_candidates = [raw_bbox]
    else:
        bbox_candidates = []

    for bbox_candidate in bbox_candidates:
        normalized = _normalize_bbox(bbox_candidate)
        if normalized is None:
            continue
        bboxes.append(
            _scale_bbox_to_page(normalized, page_width, page_height) if coordinates_are_normalized else normalized
        )

    return bboxes


def _walk_items(
    raw_items: list[Any],
    page_number: int,
    page_width: float,
    page_height: float,
    coordinates_are_normalized: bool,
    page_counter: list[int],
    depth: int,
    source_path: str,
    out_items: list[GroundingItem],
    override_candidates: list[dict[str, Any]] | None = None,
    override_cursor: list[int] | None = None,
) -> None:
    for position, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            continue

        item_index = page_counter[0]
        page_counter[0] += 1

        bboxes = _extract_item_bboxes(
            raw_item=raw_item,
            page_width=page_width,
            page_height=page_height,
            coordinates_are_normalized=coordinates_are_normalized,
        )

        md = _extract_md(raw_item)
        item_type = str(raw_item.get("type") or "unknown")
        item_source_path = f"{source_path}.{position}" if source_path else str(position)
        raw_override = _match_grounded_item_override(raw_item, override_candidates, override_cursor)

        if md or bboxes:
            out_items.append(
                GroundingItem(
                    item_id=f"p{page_number}-i{item_index}",
                    item_index=item_index,
                    page_number=page_number,
                    depth=depth,
                    type=item_type,
                    md=md,
                    value=raw_item.get("value") if isinstance(raw_item.get("value"), str) else None,
                    source_path=item_source_path,
                    raw_payload=raw_override or raw_item,
                    bboxes=bboxes,
                )
            )

        nested = raw_item.get("items")
        if isinstance(nested, list):
            _walk_items(
                raw_items=nested,
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
                coordinates_are_normalized=coordinates_are_normalized,
                page_counter=page_counter,
                depth=depth + 1,
                source_path=f"{item_source_path}.items",
                out_items=out_items,
                override_candidates=override_candidates,
                override_cursor=override_cursor,
            )


def _read_image_size(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        return float(image.width), float(image.height)


def _pdf_page_sizes(path: Path) -> list[tuple[float, float]]:
    with fitz.open(path) as doc:
        return [(float(page.rect.width), float(page.rect.height)) for page in doc]


def _normalize_pages(
    payload: dict[str, Any],
    source_doc: IndexedDocumentInternal,
    payload_kind: Literal["normalized", "legacy"],
    *,
    raw_payload: dict[str, Any] | None = None,
    result_payload: dict[str, Any] | None = None,
) -> list[GroundingPage]:
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raw_pages = []

    pages: list[GroundingPage] = []

    fallback_pdf_sizes: list[tuple[float, float]] = []
    fallback_image_size: tuple[float, float] | None = None

    if source_doc.source_kind == "pdf":
        fallback_pdf_sizes = _pdf_page_sizes(source_doc.source_path)
    else:
        fallback_image_size = _read_image_size(source_doc.source_path)

    grounded_override_items_by_page = _extract_llamaparse_grounded_items_by_page(raw_payload)

    for page_pos, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, dict):
            continue

        page_number = _as_int(
            raw_page.get("page_number") or raw_page.get("page"),
            fallback=_as_int(raw_page.get("page_index"), fallback=page_pos) + 1,
        )
        page_width = _as_float(raw_page.get("page_width"), fallback=_as_float(raw_page.get("width"), fallback=0.0))
        page_height = _as_float(raw_page.get("page_height"), fallback=_as_float(raw_page.get("height"), fallback=0.0))

        if (page_width <= 0 or page_height <= 0) and source_doc.source_kind == "pdf":
            if page_number - 1 < len(fallback_pdf_sizes):
                page_width, page_height = fallback_pdf_sizes[page_number - 1]
        elif (page_width <= 0 or page_height <= 0) and fallback_image_size is not None:
            page_width, page_height = fallback_image_size

        normalized_items: list[GroundingItem] = []
        counter = [0]
        override_candidates = grounded_override_items_by_page.get(page_number)
        override_cursor = [0] if override_candidates else None
        page_items = raw_page.get("items")
        if isinstance(page_items, list):
            _walk_items(
                raw_items=page_items,
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
                coordinates_are_normalized=payload_kind == "normalized",
                page_counter=counter,
                depth=0,
                source_path="items",
                out_items=normalized_items,
                override_candidates=override_candidates,
                override_cursor=override_cursor,
            )

        pages.append(
            GroundingPage(
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
                items=normalized_items,
            )
        )

    if not pages:
        if source_doc.source_kind == "pdf":
            sizes = _pdf_page_sizes(source_doc.source_path)
            pages = [
                GroundingPage(page_number=idx + 1, page_width=size[0], page_height=size[1], items=[])
                for idx, size in enumerate(sizes)
            ]
        else:
            if fallback_image_size is None:
                fallback_image_size = _read_image_size(source_doc.source_path)
            pages = [
                GroundingPage(
                    page_number=1,
                    page_width=fallback_image_size[0],
                    page_height=fallback_image_size[1],
                    items=[],
                )
            ]

    pages.sort(key=lambda p: p.page_number)

    citation_items_by_page = _extract_field_citation_items(result_payload, pages)
    if citation_items_by_page:
        pages = [
            page.model_copy(update={"items": [*page.items, *citation_items_by_page.get(page.page_number, [])]})
            for page in pages
        ]

    granular_layers_by_page = _build_granular_layers(
        pages,
        raw_payload,
        result_payload,
    )
    pages = [
        page.model_copy(update={"granular_layers": granular_layers_by_page.get(page.page_number, [])}) for page in pages
    ]
    return pages


def load_document(doc: IndexedDocumentInternal) -> DocumentResponse:
    raw_payload: dict[str, Any] | None = None
    raw_json: str | None = None
    if doc.raw_path is not None:
        try:
            raw_payload = _read_json(doc.raw_path)
            raw_json = json.dumps(raw_payload, indent=2)
        except Exception:
            raw_payload = None
            raw_json = None

    result_payload: dict[str, Any] | None = None
    result_json: str | None = None
    if doc.result_path is not None:
        try:
            result_payload = _read_json(doc.result_path)
            result_json = json.dumps(result_payload, indent=2)
        except Exception:
            result_payload = None
            result_json = None

    payload, selected_source, payload_kind = _extract_v2_items_payload(
        doc=doc,
        raw_payload=raw_payload,
        result_payload=result_payload,
    )
    pages = _normalize_pages(
        payload,
        doc,
        payload_kind,
        raw_payload=raw_payload,
        result_payload=result_payload,
    )

    page_markdown, document_markdown, selected_markdown_source = _select_markdown_payload(
        doc=doc,
        selected_grounding_source=selected_source,
        raw_payload=raw_payload,
        result_payload=result_payload,
    )
    if document_markdown and not page_markdown and len(pages) == 1:
        page_markdown = {pages[0].page_number: document_markdown}

    pages = [page.model_copy(update={"markdown": page_markdown.get(page.page_number)}) for page in pages]

    page_gt_rules = load_page_gt_rules(
        test_case_path=(
            doc.test_case_path
            if doc.test_case_path is not None and doc.test_case_path.is_file()
            else (doc.source_path.parent / f"{doc.base_name}.test.json")
        ),
        pages=pages,
        result_path=doc.result_path,
        result_payload=result_payload,
    )
    pages = [page.model_copy(update={"gt_rules": page_gt_rules.get(page.page_number, [])}) for page in pages]

    if document_markdown is None and page_markdown:
        document_markdown = (
            "\n\n".join(
                page_markdown[page.page_number]
                for page in pages
                if page.page_number in page_markdown and page_markdown[page.page_number].strip()
            )
            or None
        )

    return DocumentResponse(
        doc_id=doc.doc_id,
        base_name=doc.base_name,
        relative_dir=doc.relative_dir,
        source_kind=doc.source_kind,
        source_ext=doc.source_ext,
        source_file_url=map_host_path_to_files_url(doc.source_path),
        page_count=len(pages),
        pages=pages,
        selected_grounding_source=selected_source,
        selected_markdown_source=selected_markdown_source,
        document_markdown=document_markdown,
        raw_json=raw_json,
        result_json=result_json,
        artifact_flags=doc.artifact_flags,
    )
