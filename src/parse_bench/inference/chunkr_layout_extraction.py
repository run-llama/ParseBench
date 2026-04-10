"""Utilities for extracting normalized layout predictions from Chunkr output."""

from __future__ import annotations

from typing import Any

from parse_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    LayoutTableContent,
    LayoutTextContent,
)


def extract_layout_from_chunkr_output(
    raw_output: dict[str, Any],
    page_index: int = 0,
    example_id: str = "",
    pipeline_name: str = "",
    target_width: int | None = None,
    target_height: int | None = None,
) -> LayoutOutput | None:
    """Extract normalized layout predictions from one Chunkr page."""
    page_number = page_index + 1
    chunks = raw_output.get("output", {}).get("chunks", [])

    predictions: list[LayoutPrediction] = []
    page_width = 0
    page_height = 0

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        segments = chunk.get("segments", [])
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            if int(segment.get("page_number", 1)) != page_number:
                continue

            if page_width == 0:
                page_width = int(segment.get("page_width", 0))
                page_height = int(segment.get("page_height", 0))

            bbox_data = segment.get("bbox") or {}
            left = float(bbox_data.get("left", 0.0))
            top = float(bbox_data.get("top", 0.0))
            width = float(bbox_data.get("width", 0.0))
            height = float(bbox_data.get("height", 0.0))

            predictions.append(
                LayoutPrediction(
                    bbox=[left, top, left + width, top + height],
                    score=float(segment.get("confidence", 1.0)),
                    label=str(segment.get("segment_type", "Unknown")),
                    page=page_number,
                    content=_build_chunkr_segment_content(segment),
                    provider_metadata={
                        "segment_id": segment.get("segment_id"),
                        "order_index": len(predictions),
                    },
                )
            )

    if not predictions:
        return None

    output_width = target_width if target_width is not None else page_width
    output_height = target_height if target_height is not None else page_height

    return LayoutOutput(
        task_type="layout_detection",
        example_id=example_id,
        pipeline_name=pipeline_name,
        model=LayoutDetectionModel.CHUNKR,
        image_width=max(int(output_width), 1),
        image_height=max(int(output_height), 1),
        predictions=predictions,
    )


def extract_all_layouts_from_chunkr_output(
    raw_output: dict[str, Any],
    example_id: str = "",
    pipeline_name: str = "",
) -> LayoutOutput:
    """Extract normalized layout predictions from all Chunkr pages."""
    chunks = raw_output.get("output", {}).get("chunks", [])

    predictions: list[LayoutPrediction] = []
    output_width = 0
    output_height = 0

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        segments = chunk.get("segments", [])
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, dict):
                continue

            if output_width == 0:
                output_width = int(segment.get("page_width", 0))
                output_height = int(segment.get("page_height", 0))

            bbox_data = segment.get("bbox") or {}
            left = float(bbox_data.get("left", 0.0))
            top = float(bbox_data.get("top", 0.0))
            width = float(bbox_data.get("width", 0.0))
            height = float(bbox_data.get("height", 0.0))

            predictions.append(
                LayoutPrediction(
                    bbox=[left, top, left + width, top + height],
                    score=float(segment.get("confidence", 1.0)),
                    label=str(segment.get("segment_type", "Unknown")),
                    page=int(segment.get("page_number", 1)),
                    content=_build_chunkr_segment_content(segment),
                    provider_metadata={
                        "segment_id": segment.get("segment_id"),
                        "order_index": len(predictions),
                    },
                )
            )

    return LayoutOutput(
        task_type="layout_detection",
        example_id=example_id,
        pipeline_name=pipeline_name,
        model=LayoutDetectionModel.CHUNKR,
        image_width=max(int(output_width), 1),
        image_height=max(int(output_height), 1),
        predictions=predictions,
    )


def _build_chunkr_segment_content(
    segment: dict[str, Any],
) -> LayoutTextContent | LayoutTableContent | None:
    segment_type = str(segment.get("segment_type", "")).strip().lower()
    html = segment.get("html")
    text = segment.get("content") or segment.get("text")

    if segment_type == "table":
        if isinstance(html, str) and html:
            return LayoutTableContent(html=html)
        if isinstance(text, str) and text:
            return LayoutTextContent(text=text)
        return None

    if isinstance(text, str) and text:
        return LayoutTextContent(text=text)
    if isinstance(html, str) and html:
        return LayoutTextContent(text=html)
    return None
