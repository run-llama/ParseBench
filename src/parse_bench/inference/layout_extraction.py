"""Utilities for extracting normalized layout predictions from LlamaParse output."""

from __future__ import annotations

import logging
import re
from typing import Any

from parse_bench.layout_label_mapping import (
    detect_llamaparse_label_version,
    map_llamaparse_raw_label_to_canonical,
)
from parse_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    LayoutTableContent,
    LayoutTextContent,
)
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR

logger = logging.getLogger(__name__)

_CHECKBOX_RAW_LABELS = frozenset({"checkbox-selected", "checkbox-unselected"})
_CHECKBOX_CANONICAL_ITEM_TYPES = frozenset({"Checkbox-Selected", "Checkbox-Unselected"})


def _resolve_label_version(
    labels: list[str],
    force_version: str | None = None,
    example_id: str = "",
) -> str:
    """Resolve and log the LlamaParse label version."""
    version = force_version or detect_llamaparse_label_version(labels)
    unique_labels = sorted(set(labels))[:10]
    logger.info(
        "LlamaParse layout version: %s | example_id=%s | sample_labels=%s",
        version.upper(),
        example_id,
        unique_labels,
    )
    return version


def extract_layout_from_llamaparse_output(
    raw_output: dict[str, Any],
    page_index: int = 0,
    example_id: str = "",
    pipeline_name: str = "",
    target_width: int | None = None,
    target_height: int | None = None,
    label_version: str | None = None,
) -> LayoutOutput | None:
    """Extract normalized layout predictions from one page of LlamaParse output."""
    api_pages: list[dict[str, Any]] = raw_output.get("pages", [])
    if page_index >= len(api_pages):
        return None

    page_data = api_pages[page_index]
    labels = _collect_labels(api_pages)
    resolved_label_version = _resolve_label_version(labels, label_version, example_id)

    sdk_width = float(page_data.get("width", 0))
    sdk_height = float(page_data.get("height", 0))

    if target_width is not None and target_height is not None:
        output_width = target_width
        output_height = target_height
    elif len(api_pages) == 1:
        output_width = int(raw_output.get("image_width", sdk_width))
        output_height = int(raw_output.get("image_height", sdk_height))
    else:
        output_width = int(sdk_width)
        output_height = int(sdk_height)

    x_scale = output_width / sdk_width if sdk_width > 0 else 1.0
    y_scale = output_height / sdk_height if sdk_height > 0 else 1.0

    items = page_data.get("items", [])
    page_md = page_data.get("md", "") or page_data.get("text", "") or ""
    predictions, layout_items = _extract_page_predictions(
        items=items,
        page_number=page_index + 1,
        page_md=page_md,
        label_version=resolved_label_version,
        x_scale=x_scale,
        y_scale=y_scale,
    )

    markdown = _page_markdown(page_data)

    return LayoutOutput(
        task_type="layout_detection",
        example_id=example_id,
        pipeline_name=pipeline_name,
        model=LayoutDetectionModel.LLAMAPARSE,
        image_width=max(int(output_width), 1),
        image_height=max(int(output_height), 1),
        layout_pages=[
            ParseLayoutPageIR(
                page_number=page_index + 1,
                width=float(output_width),
                height=float(output_height),
                md=markdown,
                items=layout_items,
            )
        ],
        predictions=predictions,
        markdown=markdown,
    )


def extract_all_layouts_from_llamaparse_output(
    raw_output: dict[str, Any],
    example_id: str = "",
    pipeline_name: str = "",
    label_version: str | None = None,
) -> LayoutOutput:
    """Extract normalized layout predictions from all pages of LlamaParse output."""
    api_pages: list[dict[str, Any]] = raw_output.get("pages", [])
    if not api_pages:
        return LayoutOutput(
            task_type="layout_detection",
            example_id=example_id,
            pipeline_name=pipeline_name,
            model=LayoutDetectionModel.LLAMAPARSE,
            image_width=1,
            image_height=1,
            predictions=[],
            markdown="",
        )

    labels = _collect_labels(api_pages)
    resolved_label_version = _resolve_label_version(labels, label_version, example_id)

    first_page = api_pages[0]
    output_width = int(first_page.get("width", 1))
    output_height = int(first_page.get("height", 1))

    if len(api_pages) == 1:
        output_width = int(raw_output.get("image_width", output_width))
        output_height = int(raw_output.get("image_height", output_height))

    predictions: list[LayoutPrediction] = []
    layout_pages: list[ParseLayoutPageIR] = []
    page_markdowns: list[str] = []

    for page_idx, page_data in enumerate(api_pages):
        page_number = page_idx + 1
        sdk_width = float(page_data.get("width", output_width))
        sdk_height = float(page_data.get("height", output_height))
        x_scale = output_width / sdk_width if sdk_width > 0 else 1.0
        y_scale = output_height / sdk_height if sdk_height > 0 else 1.0

        items = page_data.get("items", [])
        page_md = _page_markdown(page_data)
        if page_md:
            page_markdowns.append(page_md)
        page_predictions, page_layout_items = _extract_page_predictions(
            items=items,
            page_number=page_number,
            page_md=page_md,
            label_version=resolved_label_version,
            x_scale=x_scale,
            y_scale=y_scale,
            order_index_offset=len(predictions),
        )
        predictions.extend(page_predictions)
        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_number,
                width=float(output_width),
                height=float(output_height),
                md=page_md,
                items=page_layout_items,
            )
        )

    return LayoutOutput(
        task_type="layout_detection",
        example_id=example_id,
        pipeline_name=pipeline_name,
        model=LayoutDetectionModel.LLAMAPARSE,
        image_width=max(int(output_width), 1),
        image_height=max(int(output_height), 1),
        layout_pages=layout_pages,
        predictions=predictions,
        markdown="\n\n---\n\n".join(page_markdowns),
    )


def _page_markdown(page_data: dict[str, Any]) -> str:
    """Return the best available markdown/text payload for a page dict."""
    md = page_data.get("md")
    if isinstance(md, str) and md:
        return md
    text = page_data.get("text")
    if isinstance(text, str):
        return text
    return ""


def _extract_page_predictions(
    *,
    items: list[Any],
    page_number: int,
    page_md: str,
    label_version: str,
    x_scale: float,
    y_scale: float,
    order_index_offset: int = 0,
) -> tuple[list[LayoutPrediction], list[LayoutItemIR]]:
    predictions: list[LayoutPrediction] = []
    layout_items: list[LayoutItemIR] = []
    table_htmls = _extract_table_htmls(page_md)
    table_html_idx = 0
    mark_scope_keys = _collect_mark_scope_checkbox_keys(items)

    for item_idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        layout_bboxes = item.get("layoutAwareBbox", [])
        item_type = str(item.get("type") or "text")
        item_text = str(item.get("value") or "")

        for segment_idx, bbox_data in enumerate(layout_bboxes):
            if not isinstance(bbox_data, dict):
                continue
            label = bbox_data.get("label")
            if not isinstance(label, str):
                continue

            map_llamaparse_raw_label_to_canonical(
                label,
                label_version=label_version,
            )

            if _should_skip_duplicate_checkbox_prediction(
                item_type=item_type,
                raw_label=label,
                bbox_data=bbox_data,
                mark_scope_keys=mark_scope_keys,
            ):
                continue

            x = float(bbox_data.get("x", 0)) * x_scale
            y = float(bbox_data.get("y", 0)) * y_scale
            w = float(bbox_data.get("w", 0)) * x_scale
            h = float(bbox_data.get("h", 0)) * y_scale

            content, consumed_table = _build_content(
                item_type=item_type,
                item_text=item_text,
                segment=bbox_data,
                table_htmls=table_htmls,
                table_html_idx=table_html_idx,
            )
            if consumed_table:
                table_html_idx += 1

            attributes = _prediction_attributes(item_type=item_type, raw_label=label)
            prediction = LayoutPrediction(
                bbox=[x, y, x + w, y + h],
                score=float(bbox_data.get("confidence", 0.0)),
                label=label,
                page=page_number,
                content=content,
                attributes=attributes,
                provider_metadata={
                    "label_version": label_version,
                    "item_type": item_type,
                    "item_index": item_idx,
                    "segment_index": segment_idx,
                    "order_index": order_index_offset + len(predictions),
                },
            )
            predictions.append(prediction)
            layout_items.append(_prediction_to_layout_item(prediction, label_version=label_version))

    return predictions, layout_items


def _collect_mark_scope_checkbox_keys(items: list[Any]) -> set[tuple[str, float, float, float, float]]:
    keys: set[tuple[str, float, float, float, float]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "text")
        if item_type not in _CHECKBOX_CANONICAL_ITEM_TYPES:
            continue
        for bbox_data in item.get("layoutAwareBbox", []):
            if not isinstance(bbox_data, dict):
                continue
            raw_label = bbox_data.get("label")
            if not isinstance(raw_label, str) or raw_label not in _CHECKBOX_RAW_LABELS:
                continue
            keys.add(_checkbox_key(raw_label, bbox_data))
    return keys


def _should_skip_duplicate_checkbox_prediction(
    *,
    item_type: str,
    raw_label: str,
    bbox_data: dict[str, Any],
    mark_scope_keys: set[tuple[str, float, float, float, float]],
) -> bool:
    if raw_label not in _CHECKBOX_RAW_LABELS:
        return False
    if item_type in _CHECKBOX_CANONICAL_ITEM_TYPES:
        return False
    return _checkbox_key(raw_label, bbox_data) in mark_scope_keys


def _prediction_attributes(*, item_type: str, raw_label: str) -> dict[str, str]:
    if item_type in _CHECKBOX_CANONICAL_ITEM_TYPES and raw_label in _CHECKBOX_RAW_LABELS:
        return {"scope": "mark"}
    return {}


def _checkbox_key(raw_label: str, bbox_data: dict[str, Any]) -> tuple[str, float, float, float, float]:
    return (
        raw_label,
        round(float(bbox_data.get("x", 0.0)), 4),
        round(float(bbox_data.get("y", 0.0)), 4),
        round(float(bbox_data.get("w", 0.0)), 4),
        round(float(bbox_data.get("h", 0.0)), 4),
    )


def _prediction_to_layout_item(
    prediction: LayoutPrediction,
    *,
    label_version: str,
) -> LayoutItemIR:
    canonical_label, _ = map_llamaparse_raw_label_to_canonical(
        prediction.label,
        label_version=label_version,
    )
    x1, y1, x2, y2 = prediction.bbox
    segment = LayoutSegmentIR(
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        confidence=prediction.score,
        label=prediction.label,
    )
    item_kwargs: dict[str, Any] = {
        "type": canonical_label.value,
        "bbox": segment,
        "layout_segments": [segment],
        "score": prediction.score,
        "attributes": dict(prediction.attributes),
    }
    if isinstance(prediction.content, LayoutTableContent):
        item_kwargs["html"] = prediction.content.html
    elif isinstance(prediction.content, LayoutTextContent):
        item_kwargs["md"] = prediction.content.text
        item_kwargs["value"] = prediction.content.text
    return LayoutItemIR(**item_kwargs)


def _collect_labels(pages: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        items = page.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            layout_aware = item.get("layoutAwareBbox")
            if not isinstance(layout_aware, list):
                continue
            for segment in layout_aware:
                if isinstance(segment, dict) and isinstance(segment.get("label"), str):
                    labels.append(segment["label"])
    return labels


def _build_content(
    *,
    item_type: str,
    item_text: str,
    segment: dict[str, Any],
    table_htmls: list[str],
    table_html_idx: int,
) -> tuple[LayoutTextContent | LayoutTableContent | None, bool]:
    if item_type == "table":
        if table_html_idx < len(table_htmls):
            return LayoutTableContent(html=table_htmls[table_html_idx]), True
        # When no HTML table is available, use per-segment text extraction
        # (same logic as non-table items) to avoid inflating attribution F1
        # with the full table markdown for every segment.
        start = segment.get("startIndex")
        end = segment.get("endIndex")
        if isinstance(start, int) and isinstance(end, int) and end >= start:
            text = item_text[start : end + 1]
            if text:
                return LayoutTextContent(text=text), False
        if item_text:
            return LayoutTextContent(text=item_text), False
        return None, False

    start = segment.get("startIndex")
    end = segment.get("endIndex")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        # Preserve inclusive slicing semantics.
        text = item_text[start : end + 1]
    else:
        text = item_text

    if not text:
        return None, False
    return LayoutTextContent(text=text), False


def _extract_table_htmls(markdown: str) -> list[str]:
    return re.findall(r"<table>.*?</table>", markdown, flags=re.DOTALL | re.IGNORECASE)
