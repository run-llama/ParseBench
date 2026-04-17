"""Concrete layout adapters and registry bindings."""

from __future__ import annotations

import re
from typing import Any

from parse_bench.evaluation.layout_adapters.base import LayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import register_layout_adapter
from parse_bench.evaluation.metrics.attribution.core import (
    PredBlock,
    parse_pred_blocks,
)
from parse_bench.evaluation.metrics.attribution.text_utils import (
    extract_text_from_html,
    normalize_attribution_text,
    tokenize,
)
from parse_bench.inference.layout_extraction import (
    extract_all_layouts_from_llamaparse_output,
)
from parse_bench.inference.providers.layoutdet.adapters import ChunkrLayoutDetLabelAdapter
from parse_bench.inference.providers.parse.llamaparse_v2_normalization import (
    build_pages_from_cli2_raw_payload,
    build_pages_from_sdk_response_payload,
    layout_pages_to_legacy_pages_payload,
)
from parse_bench.layout_label_mapping import (
    UnknownRawLayoutLabelError,
)
from parse_bench.schemas.layout_detection_output import (
    QWEN3VL_STR_TO_LABEL,
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
    LayoutTableContent,
    LayoutTextContent,
)
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline_io import InferenceResult
from parse_bench.test_cases.schema import TestCase


@register_layout_adapter("__default__", priority=-100)
class NormalizedLayoutOutputAdapter(LayoutAdapter):
    """Adapter for providers that already emit `LayoutOutput`."""

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if not isinstance(inference_result.output, LayoutOutput):
            raise ValueError("Inference output is not LayoutOutput and no provider adapter matched.")

        if page_filter is None:
            return inference_result.output

        predictions = [
            prediction for prediction in inference_result.output.predictions if prediction.page == page_filter
        ]
        return inference_result.output.model_copy(update={"predictions": predictions})


@register_layout_adapter(
    "llamaparse",
    "llamaparse_local_cli2",
    "mock_llamacloud_parse",
    "llamaparse_dualpass_internal",
    priority=100,
)
class LlamaParseLayoutAdapter(LayoutAdapter):
    """Adapter for LlamaParse-family outputs with output-first + legacy fallback support."""

    def __init__(self) -> None:
        self._pages_payload: list[dict[str, Any]] | None = None

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if isinstance(inference_result.output, ParseOutput):
            if len(inference_result.output.layout_pages) > 0:
                return True

        if (
            isinstance(inference_result.output, LayoutOutput)
            and inference_result.output.model == LayoutDetectionModel.LLAMAPARSE
        ):
            return True

        raw_output = inference_result.raw_output
        if not isinstance(raw_output, dict):
            return False
        pages = raw_output.get("pages")
        if not isinstance(pages, list) or not pages:
            return False
        first_page = pages[0]
        if not isinstance(first_page, dict):
            return False
        items = first_page.get("items")
        return isinstance(items, list)

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        pages = _resolve_llamaparse_pages(inference_result)
        raw_output = inference_result.raw_output if isinstance(inference_result.raw_output, dict) else {}

        if pages:
            self._pages_payload = pages
            extraction_input: dict[str, Any] = {"pages": pages}
            raw_image_width = raw_output.get("image_width")
            raw_image_height = raw_output.get("image_height")
            if isinstance(raw_image_width, (int, float)) and isinstance(raw_image_height, (int, float)):
                extraction_input["image_width"] = raw_image_width
                extraction_input["image_height"] = raw_image_height

            layout_output = extract_all_layouts_from_llamaparse_output(
                raw_output=extraction_input,
                example_id=inference_result.request.example_id,
                pipeline_name=inference_result.pipeline_name,
            )
            if page_filter is None:
                return layout_output

            predictions = [prediction for prediction in layout_output.predictions if prediction.page == page_filter]
            return layout_output.model_copy(update={"predictions": predictions})

        self._pages_payload = None
        if (
            isinstance(inference_result.output, LayoutOutput)
            and inference_result.output.model == LayoutDetectionModel.LLAMAPARSE
        ):
            if page_filter is None:
                return inference_result.output
            predictions = [
                prediction for prediction in inference_result.output.predictions if prediction.page == page_filter
            ]
            return inference_result.output.model_copy(update={"predictions": predictions})

        raise ValueError("LlamaParse adapter requires ParseOutput.layout_pages or raw_output.pages")

    def to_attribution_blocks(
        self,
        layout_output: LayoutOutput,
        *,
        page_number: int,
        test_case: TestCase | None = None,
    ) -> list[PredBlock]:
        del test_case
        if self._pages_payload is None:
            return super().to_attribution_blocks(
                layout_output,
                page_number=page_number,
                test_case=None,
            )

        raw_page = _find_page_payload(self._pages_payload, page_number)
        if raw_page is None:
            return super().to_attribution_blocks(
                layout_output,
                page_number=page_number,
                test_case=None,
            )

        items = raw_page.get("items")
        if not isinstance(items, list):
            return super().to_attribution_blocks(
                layout_output,
                page_number=page_number,
                test_case=None,
            )

        page_md = raw_page.get("md", "") or raw_page.get("text", "") or ""
        page_width = float(raw_page.get("width") or layout_output.image_width or 1)
        page_height = float(raw_page.get("height") or layout_output.image_height or 1)
        return parse_pred_blocks(items, page_md, page_width, page_height)


@register_layout_adapter("chunkr", priority=90)
class ChunkrLayoutAdapter(LayoutAdapter):
    """Adapter for Chunkr raw parse output (`output.chunks[].segments[]`)."""

    _label_adapter = ChunkrLayoutDetLabelAdapter()

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        raw_output = inference_result.raw_output
        if not isinstance(raw_output, dict):
            return False
        output = raw_output.get("output")
        if not isinstance(output, dict):
            return False
        return isinstance(output.get("chunks"), list)

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        raw_output = inference_result.raw_output
        if not isinstance(raw_output, dict):
            raise ValueError("Chunkr adapter requires dict raw_output")

        chunks = raw_output.get("output", {}).get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError("Chunkr adapter requires raw_output.output.chunks")

        inferred_page_number = _infer_page_number_from_example_id(inference_result.request.example_id)
        predictions: list[LayoutPrediction] = []
        output_width = 0
        output_height = 0

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            segments = chunk.get("segments")
            if not isinstance(segments, list):
                continue

            for segment in segments:
                if not isinstance(segment, dict):
                    continue

                page_number = int(segment.get("page_number", 1))
                # Chunkr single-page inference artifacts frequently report page_number=1,
                # while benchmark example IDs keep original doc page (e.g., "..._page136_...").
                # Use inferred page for this case so cross-evaluation page filtering works.
                if inferred_page_number is not None and page_number == 1:
                    page_number = inferred_page_number
                if page_filter is not None and page_number != page_filter:
                    continue

                segment_label = segment.get("segment_type")
                if not isinstance(segment_label, str):
                    continue

                bbox_data = segment.get("bbox") or {}
                left = float(bbox_data.get("left", 0.0))
                top = float(bbox_data.get("top", 0.0))
                width = float(bbox_data.get("width", 0.0))
                height = float(bbox_data.get("height", 0.0))
                bbox_xyxy = [left, top, left + width, top + height]

                if self._label_adapter.to_canonical(segment_label, 1.0, bbox_xyxy) is None:
                    raise UnknownRawLayoutLabelError(f"Unknown Chunkr raw layout label '{segment_label}'")

                if output_width == 0:
                    output_width = int(segment.get("page_width", 0))
                    output_height = int(segment.get("page_height", 0))

                html = segment.get("html")
                text = segment.get("content") or segment.get("text")
                content = None
                is_table_segment = segment_label.strip().lower() == "table"
                if is_table_segment:
                    if isinstance(html, str) and html:
                        content = LayoutTableContent(html=html)
                    elif isinstance(text, str) and text:
                        content = LayoutTextContent(text=text)  # type: ignore[assignment]
                else:
                    if isinstance(text, str) and text:
                        content = LayoutTextContent(text=text)  # type: ignore[assignment]
                    elif isinstance(html, str) and html:
                        # Fallback when provider omits plain text but includes HTML.
                        content = LayoutTextContent(text=html)  # type: ignore[assignment]

                predictions.append(
                    LayoutPrediction(
                        bbox=bbox_xyxy,
                        score=float(segment.get("confidence", 1.0)),
                        label=segment_label,
                        page=page_number,
                        content=content,
                        provider_metadata={
                            "segment_id": segment.get("segment_id"),
                            "order_index": len(predictions),
                        },
                    )
                )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.CHUNKR,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("dots_ocr_parse", priority=90)
class DotsOcrLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from dots.ocr ParseOutput.layout_pages.

    This enables cross-evaluation: a single dots.ocr PARSE pipeline can be
    evaluated against both parse and layout detection datasets, following the
    same pattern as LlamaParse's ``ours_agentic`` pipeline.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Distinguish from LlamaParse by checking raw_output for dots.ocr markers
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            prompt_mode = raw_output.get("prompt_mode", "")
            return isinstance(prompt_mode, str) and prompt_mode.startswith("prompt_layout")
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        # Handle synthetic LayoutOutput results (e.g. from cross-eval runner)
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("DotsOcrLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("DotsOcrLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_dots_ocr_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.DOTS_OCR,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


def _build_docling_parse_content(item_type: str, text: str) -> LayoutTextContent | LayoutTableContent | None:
    """Build content object for Docling parse-derived layout items."""
    if not text:
        return None
    if item_type == "table":
        return LayoutTableContent(html=text)
    return LayoutTextContent(text=text)


@register_layout_adapter("docling_parse", priority=90)
class DoclingParseLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Docling ParseOutput.layout_pages."""

    def __init__(self) -> None:
        self._current_layout_pages: list[Any] | None = None

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        return isinstance(raw_output, dict) and isinstance(raw_output.get("docling_document"), dict)

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("DoclingParseLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("DoclingParseLayoutAdapter requires non-empty layout_pages")

        selected_pages = [lp for lp in layout_pages if page_filter is None or lp.page_number == page_filter]
        reference_page = selected_pages[0] if selected_pages else layout_pages[0]
        output_width = int(reference_page.width or 1)
        output_height = int(reference_page.height or 1)
        self._current_layout_pages = layout_pages

        predictions: list[LayoutPrediction] = []
        markdown_parts: list[str] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)
            if lp.md:
                markdown_parts.append(lp.md)

            for item_idx, item in enumerate(lp.items):
                segments = item.layout_segments or ([item.bbox] if item.bbox is not None else [])
                for segment_idx, seg in enumerate(segments):
                    label = seg.label or item.type or "text"
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=_build_docling_parse_content(item.type, item.value),
                            provider_metadata={
                                "order_index": len(predictions),
                                "item_index": item_idx,
                                "segment_index": segment_idx,
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.DOCLING_PARSE_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
            markdown="\n\n".join(markdown_parts),
        )

    def to_attribution_blocks(
        self,
        layout_output: LayoutOutput,
        *,
        page_number: int,
        test_case: TestCase | None = None,
    ) -> list[PredBlock]:
        del test_case
        if self._current_layout_pages is None:
            return super().to_attribution_blocks(layout_output, page_number=page_number, test_case=None)

        layout_pages = self._current_layout_pages
        page = next((lp for lp in layout_pages if lp.page_number == page_number), None)
        if page is None:
            return []

        blocks: list[PredBlock] = []
        for item_index, item in enumerate(page.items):
            segments = item.layout_segments or ([item.bbox] if item.bbox is not None else [])
            if not segments:
                continue

            for seg in segments:
                label = seg.label or item.type or "unknown"
                block_type = item.type or "text"
                if item.type == "table":
                    raw_text = extract_text_from_html(item.value)
                else:
                    raw_text = item.value or ""
                    if (
                        isinstance(seg.start_index, int)
                        and isinstance(seg.end_index, int)
                        and seg.end_index >= seg.start_index
                    ):
                        raw_text = raw_text[seg.start_index : seg.end_index + 1]

                normalized_text = normalize_attribution_text(raw_text)
                blocks.append(
                    PredBlock(
                        bbox_xyxy=[seg.x, seg.y, seg.x + seg.w, seg.y + seg.h],
                        block_type=block_type,
                        label=label,
                        text=raw_text,
                        normalized_text=normalized_text,
                        tokens=tokenize(normalized_text),
                        order_index=item_index,
                    )
                )

        return blocks


@register_layout_adapter("qwen3vl_layout", priority=90)
class Qwen3VLLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Qwen3-VL ParseOutput.layout_pages.

    Enables cross-evaluation: the ``qwen3vl_layout`` PARSE pipeline can be
    evaluated against layout detection datasets using the bboxes from
    the structured JSON output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            return "items" in raw_output and "raw_content" in raw_output
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("Qwen3VLLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("Qwen3VLLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    str_label = seg.label or item.type or "Text"

                    # Convert string label to integer label for Qwen3VL evaluator
                    # Map canonical-style "Page-header" → "page_header" for lookup
                    lookup_key = str_label.lower().replace("-", "_")
                    qwen_enum = QWEN3VL_STR_TO_LABEL.get(lookup_key)
                    int_label = str(int(qwen_enum)) if qwen_enum is not None else str_label

                    # Convert normalized [0,1] xywh -> pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_dots_ocr_content(str_label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=int_label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.QWEN3_VL_8B,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


def _parse_with_layout_to_layout_output(
    inference_result: InferenceResult,
    *,
    model: LayoutDetectionModel,
    page_filter: int | None = None,
) -> LayoutOutput:
    """Shared conversion for LLM parse_with_layout adapters (Google/OpenAI/Anthropic)."""
    # Handle LayoutOutput (e.g. from multi-task re-evaluation)
    if isinstance(inference_result.output, LayoutOutput):
        if page_filter is None:
            return inference_result.output
        filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
        return inference_result.output.model_copy(update={"predictions": filtered})

    if not isinstance(inference_result.output, ParseOutput):
        out_type = type(inference_result.output).__name__
        raise ValueError(f"parse_with_layout adapter requires ParseOutput or LayoutOutput, got {out_type}")

    layout_pages = inference_result.output.layout_pages
    if not layout_pages:
        raise ValueError("parse_with_layout adapter requires non-empty layout_pages")

    first_page = layout_pages[0]
    output_width = int(first_page.width or 1)
    output_height = int(first_page.height or 1)

    predictions: list[LayoutPrediction] = []

    for lp in layout_pages:
        page_number = lp.page_number
        if page_filter is not None and page_number != page_filter:
            continue

        page_w = float(lp.width or output_width)
        page_h = float(lp.height or output_height)

        for item in lp.items:
            for seg in item.layout_segments:
                str_label = seg.label or item.type or "Text"

                lookup_key = str_label.lower().replace("-", "_")
                qwen_enum = QWEN3VL_STR_TO_LABEL.get(lookup_key)
                int_label = str(int(qwen_enum)) if qwen_enum is not None else str_label

                # Convert normalized [0,1] xywh -> pixel xyxy
                x1 = seg.x * page_w
                y1 = seg.y * page_h
                x2 = (seg.x + seg.w) * page_w
                y2 = (seg.y + seg.h) * page_h

                content = _build_dots_ocr_content(str_label, item.value)

                predictions.append(
                    LayoutPrediction(
                        bbox=[x1, y1, x2, y2],
                        score=float(seg.confidence or 1.0),
                        label=int_label,
                        page=page_number,
                        content=content,
                        provider_metadata={
                            "order_index": len(predictions),
                        },
                    )
                )

    return LayoutOutput(
        task_type="layout_detection",
        example_id=inference_result.request.example_id,
        pipeline_name=inference_result.pipeline_name,
        model=model,
        image_width=max(output_width, 1),
        image_height=max(output_height, 1),
        predictions=predictions,
    )


@register_layout_adapter("google", priority=90)
class GoogleLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Google Gemini ParseOutput.layout_pages.

    Enables cross-evaluation: the ``google_gemini_*_parse_with_layout`` PARSE pipelines
    can be evaluated against layout detection datasets using the bboxes from
    the div-wrapped output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            model = raw_output.get("model", "")
            return (
                raw_output.get("mode") == "parse_with_layout" and isinstance(model, str) and model.startswith("gemini")
            )
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        return _parse_with_layout_to_layout_output(
            inference_result,
            model=LayoutDetectionModel.GEMINI_LAYOUT,
            page_filter=page_filter,
        )


@register_layout_adapter("gemma4", priority=90)
class Gemma4LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Gemma 4 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``gemma4_*_vllm_with_layout`` PARSE pipelines
    can be evaluated against layout detection datasets using the bboxes from
    the structured layout output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            prompt_mode = raw_output.get("prompt_mode", "")
            config = raw_output.get("_config", {})
            model = config.get("model", "") if isinstance(config, dict) else ""
            return prompt_mode == "layout" and isinstance(model, str) and model.startswith("gemma")
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        return _parse_with_layout_to_layout_output(
            inference_result,
            model=LayoutDetectionModel.GEMMA4_LAYOUT,
            page_filter=page_filter,
        )


@register_layout_adapter("openai", priority=90)
class OpenAILayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from OpenAI ParseOutput.layout_pages.

    Enables cross-evaluation: the ``openai_*_parse_with_layout`` PARSE pipelines
    can be evaluated against layout detection datasets using the bboxes from
    the div-wrapped output.
    """

    # OpenAI model prefixes (gpt-*, o3-*, o4-*, etc.)
    _OPENAI_PREFIXES = ("gpt", "o3", "o4")

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            model = raw_output.get("model", "")
            return (
                raw_output.get("mode") == "parse_with_layout"
                and isinstance(model, str)
                and any(model.startswith(p) for p in cls._OPENAI_PREFIXES)
            )
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        return _parse_with_layout_to_layout_output(
            inference_result,
            model=LayoutDetectionModel.OPENAI_LAYOUT,
            page_filter=page_filter,
        )


@register_layout_adapter("anthropic", priority=90)
class AnthropicLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Anthropic ParseOutput.layout_pages.

    Enables cross-evaluation: the ``anthropic_haiku_parse_with_layout`` PARSE
    pipeline can be evaluated against layout detection datasets using the
    bboxes from the div-wrapped output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            model = raw_output.get("model", "")
            return (
                raw_output.get("mode") == "parse_with_layout" and isinstance(model, str) and model.startswith("claude")
            )
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        return _parse_with_layout_to_layout_output(
            inference_result,
            model=LayoutDetectionModel.ANTHROPIC_LAYOUT,
            page_filter=page_filter,
        )


@register_layout_adapter("reducto", priority=90)
class ReductoLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Reducto ParseOutput.layout_pages.

    Enables cross-evaluation: the ``reducto`` PARSE pipeline can be evaluated
    against layout detection datasets using the block-level bboxes from the
    Reducto API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Reducto by checking raw_output for Reducto-specific markers
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "ocr_system" in config
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        # Handle synthetic LayoutOutput results (e.g. from cross-eval runner)
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("ReductoLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("ReductoLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.REDUCTO_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("textract", priority=89)
class TextractLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Textract ParseOutput.layout_pages.

    Enables cross-evaluation: the ``aws_textract`` PARSE pipeline can be evaluated
    against layout detection datasets using the LAYOUT_* block bboxes from the
    Textract API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Textract by checking raw_output for textract_response key
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            return "textract_response" in raw_output
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("TextractLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("TextractLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.TEXTRACT_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("landingai", priority=89)
class LandingAILayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from LandingAI ParseOutput.layout_pages.

    Enables cross-evaluation: the ``landingai`` PARSE pipeline can be evaluated
    against layout detection datasets using the chunk-level bboxes from the
    LandingAI ADE API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify LandingAI by checking raw_output for grounding + chunks keys
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            return "grounding" in raw_output and "chunks" in raw_output
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("LandingAILayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("LandingAILayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.LANDINGAI_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("extend_parse", priority=89)
class ExtendLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Extend ParseOutput.layout_pages.

    Enables cross-evaluation: the ``extend_parse`` PARSE pipeline can be evaluated
    against layout detection datasets using the block-level bboxes from the
    Extend AI API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Extend by checking raw_output for _extend_metadata key
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            return "_extend_metadata" in raw_output
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("ExtendLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("ExtendLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.EXTEND_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("azure_document_intelligence", priority=89)
class AzureDILayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Azure DI ParseOutput.layout_pages.

    Enables cross-evaluation: the ``azure_document_intelligence`` PARSE pipeline
    can be evaluated against layout detection datasets using the paragraph/table/figure
    bboxes from the Azure Document Intelligence API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Azure DI by checking raw_output for _config with model_id key
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "model_id" in config
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("AzureDILayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("AzureDILayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.AZURE_DI_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("google_docai", priority=89)
class GoogleDocAILayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Google DocAI ParseOutput.layout_pages.

    Enables cross-evaluation: the ``google_docai`` PARSE pipeline can be evaluated
    against layout detection datasets using the paragraph/table bboxes from the
    Google Document AI API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Google DocAI by checking raw_output for mode key and _config
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return (
                isinstance(config, dict)
                and "processor_id" in config
                and raw_output.get("mode")
                in (
                    "ocr",
                    "layout_parser",
                )
            )
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("GoogleDocAILayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("GoogleDocAILayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.GOOGLE_DOCAI_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("unstructured", priority=89)
class UnstructuredLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Unstructured ParseOutput.layout_pages.

    Enables cross-evaluation: the ``unstructured`` PARSE pipeline (hi_res strategy)
    can be evaluated against layout detection datasets using the element-level bboxes
    from the Unstructured API response.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Unstructured by checking raw_output for _config with strategy key
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "strategy" in config
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("UnstructuredLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("UnstructuredLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh → pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.UNSTRUCTURED_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


def _build_vendor_content(label: str, text: str) -> LayoutTextContent | LayoutTableContent | None:
    """Build content object from vendor layout element."""
    if not text:
        return None
    normalized = label.strip().lower()
    if normalized == "table":
        return LayoutTableContent(html=text)
    if normalized == "picture":
        return None
    return LayoutTextContent(text=text)


def _build_dots_ocr_content(label: str, text: str) -> LayoutTextContent | LayoutTableContent | None:
    """Build content object from dots.ocr layout element."""
    if not text:
        return None
    normalized = label.strip().lower()
    if normalized == "table":
        return LayoutTableContent(html=text)
    if normalized == "picture":
        return None
    return LayoutTextContent(text=text)


@register_layout_adapter("deepseekocr2", priority=90)
class DeepSeekOCR2LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from DeepSeek-OCR-2 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``deepseekocr2_vllm`` PARSE pipeline can be
    evaluated against layout detection datasets using the grounding bboxes from
    the model output.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "deepseek" in str(config.get("server_url", "")).lower()
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("DeepSeekOCR2LayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("DeepSeekOCR2LayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.DEEPSEEK_OCR2_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("chandra2", priority=90)
class Chandra2LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Chandra OCR 2 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``chandra2_vllm`` / ``chandra2_sdk`` PARSE pipelines
    can be evaluated against layout detection datasets using the native bboxes from
    the model output. Chandra OCR 2 has 19 fine-grained labels mapping to Canonical17.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "chandra2" in str(config.get("server_url", "")).lower()
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("Chandra2LayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("Chandra2LayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.CHANDRA2_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("qfocr", priority=90)
class QfOcrLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Qianfan-OCR ParseOutput.layout_pages.

    Enables cross-evaluation: the ``qfocr_vllm_thinking`` PARSE pipeline can be
    evaluated against layout detection datasets using the Layout-as-Thought bboxes
    parsed from the model's ``<think>`` block.
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and config.get("thinking") is True
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("QfOcrLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("QfOcrLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.QFOCR_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


def _infer_page_number_from_example_id(example_id: str) -> int | None:
    match = re.search(r"_page(\d+)(?:_|$)", example_id)
    if not match:
        return None
    page_token = int(match.group(1))
    # Dataset IDs are mixed:
    # - most use 1-indexed page tokens (e.g. page136 -> page 136)
    # - some use page0 for first page.
    return page_token if page_token > 0 else 1


def _resolve_llamaparse_pages(inference_result: InferenceResult) -> list[dict[str, Any]]:
    raw_output = inference_result.raw_output
    if isinstance(raw_output, dict):
        raw_pages = raw_output.get("pages")
        if isinstance(raw_pages, list):
            return [page for page in raw_pages if isinstance(page, dict)]

        # CLI2 local provider stores items under v2_items instead of pages.
        # Normalize into the legacy page format so parse_pred_blocks can
        # access layoutAwareBbox segments for per-cell attribution.
        if "v2_items" in raw_output:
            try:
                return build_pages_from_cli2_raw_payload(
                    raw_payload=raw_output,
                    output_tables_as_markdown=False,
                )
            except (ValueError, TypeError):
                pass

        # V2 SDK API responses have items/text/metadata expansions but no
        # pre-normalized pages list.  Normalize them so parse_pred_blocks
        # can access layoutAwareBbox segments for per-cell attribution.
        if "items" in raw_output and "job" in raw_output:
            try:
                return build_pages_from_sdk_response_payload(
                    raw_payload=raw_output,
                    output_tables_as_markdown=False,
                )
            except (ValueError, TypeError):
                pass

    if isinstance(inference_result.output, ParseOutput):
        if len(inference_result.output.layout_pages) > 0:
            return layout_pages_to_legacy_pages_payload(inference_result.output.layout_pages)

    return []


def _find_page_payload(
    pages: list[dict[str, Any]],
    page_number: int,
) -> dict[str, Any] | None:
    for page_index, page in enumerate(pages):
        page_raw = page.get("page")
        page_value = page_raw if isinstance(page_raw, int) and page_raw > 0 else page_index + 1
        if page_value == page_number:
            return page

    return None


@register_layout_adapter("datalab", priority=90)
class DatalabLayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Datalab ParseOutput.layout_pages.

    Enables cross-evaluation: the ``datalab`` PARSE pipeline can be evaluated
    against layout detection datasets using block-level bboxes from the
    Datalab JSON output (powered by Marker/Surya).
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        # Identify Datalab by checking raw_output for Datalab-specific markers
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            # Datalab v0.3.0 returns parse_quality_score in raw_output
            if "parse_quality_score" in raw_output:
                return True
            config = raw_output.get("_config", {})
            return isinstance(config, dict) and "mode" in config and "ocr_system" not in config
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("DatalabLayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("DatalabLayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh -> pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_vendor_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.DATALAB_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )


@register_layout_adapter("qwen3_5", priority=90)
class Qwen35LayoutAdapter(LayoutAdapter):
    """Adapter that extracts LayoutOutput from Qwen3.5 ParseOutput.layout_pages.

    Enables cross-evaluation: the ``qwen3_5_4b_vllm`` PARSE pipeline can be
    evaluated against layout detection datasets using the bboxes from the
    merged layout+content JSON output.

    Bboxes use normalized 0-1000 coordinates (divided by 1000 to [0,1] in the
    provider, then multiplied by page pixel dimensions here).
    """

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        if not isinstance(inference_result.output, ParseOutput):
            return False
        if not inference_result.output.layout_pages:
            return False
        raw_output = inference_result.raw_output
        if isinstance(raw_output, dict):
            config = raw_output.get("_config", {})
            if isinstance(config, dict):
                model = config.get("model", "")
                return isinstance(model, str) and (model.startswith("qwen3.5") or model.startswith("qwen3.6"))
        return False

    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        if isinstance(inference_result.output, LayoutOutput):
            if page_filter is None:
                return inference_result.output
            filtered = [p for p in inference_result.output.predictions if p.page == page_filter]
            return inference_result.output.model_copy(update={"predictions": filtered})

        if not isinstance(inference_result.output, ParseOutput):
            raise ValueError("Qwen35LayoutAdapter requires ParseOutput or LayoutOutput")

        layout_pages = inference_result.output.layout_pages
        if not layout_pages:
            raise ValueError("Qwen35LayoutAdapter requires non-empty layout_pages")

        first_page = layout_pages[0]
        output_width = int(first_page.width or 1)
        output_height = int(first_page.height or 1)

        predictions: list[LayoutPrediction] = []

        for lp in layout_pages:
            page_number = lp.page_number
            if page_filter is not None and page_number != page_filter:
                continue

            page_w = float(lp.width or output_width)
            page_h = float(lp.height or output_height)

            for item in lp.items:
                for seg in item.layout_segments:
                    label = seg.label or item.type or "Text"

                    # Convert normalized [0,1] xywh -> pixel xyxy
                    x1 = seg.x * page_w
                    y1 = seg.y * page_h
                    x2 = (seg.x + seg.w) * page_w
                    y2 = (seg.y + seg.h) * page_h

                    content = _build_dots_ocr_content(label, item.value)

                    predictions.append(
                        LayoutPrediction(
                            bbox=[x1, y1, x2, y2],
                            score=float(seg.confidence or 1.0),
                            label=label,
                            page=page_number,
                            content=content,
                            provider_metadata={
                                "order_index": len(predictions),
                            },
                        )
                    )

        return LayoutOutput(
            task_type="layout_detection",
            example_id=inference_result.request.example_id,
            pipeline_name=inference_result.pipeline_name,
            model=LayoutDetectionModel.QWEN3_5_LAYOUT,
            image_width=max(output_width, 1),
            image_height=max(output_height, 1),
            predictions=predictions,
        )
