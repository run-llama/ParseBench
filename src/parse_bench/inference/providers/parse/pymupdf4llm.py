"""Provider for PyMuPDF4LLM PARSE."""

import importlib
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

_OCR_BACKEND_MODULES = {
    "rapidocr": "pymupdf4llm.ocr.rapidocr_api",
}


@register_provider("pymupdf4llm")
class PyMuPDF4LLMProvider(Provider):
    """Provider for PyMuPDF4LLM native HTML table output. AGPL — runtime dep only."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

    def _markdown_options(self) -> dict[str, Any]:
        """Build serializable options for ``pymupdf4llm.to_markdown``."""
        options: dict[str, Any] = {
            "page_chunks": True,
            "show_progress": False,
        }

        use_ocr = self.base_config.get("use_ocr")
        if use_ocr is not None:
            if not isinstance(use_ocr, bool):
                raise ProviderConfigError("PyMuPDF4LLM 'use_ocr' must be a boolean")
            options["use_ocr"] = use_ocr

        force_ocr = self.base_config.get("force_ocr")
        if force_ocr is not None:
            if not isinstance(force_ocr, bool):
                raise ProviderConfigError("PyMuPDF4LLM 'force_ocr' must be a boolean")
            options["force_ocr"] = force_ocr

        if use_ocr is False and force_ocr is True:
            raise ProviderConfigError("PyMuPDF4LLM cannot set force_ocr=True when use_ocr=False")

        ocr_dpi = self.base_config.get("ocr_dpi")
        if ocr_dpi is not None:
            if isinstance(ocr_dpi, bool) or not isinstance(ocr_dpi, int) or ocr_dpi <= 0:
                raise ProviderConfigError("PyMuPDF4LLM 'ocr_dpi' must be a positive integer")
            options["ocr_dpi"] = ocr_dpi

        ocr_language = self.base_config.get("ocr_language")
        if ocr_language is not None:
            if not isinstance(ocr_language, str) or not ocr_language.strip():
                raise ProviderConfigError("PyMuPDF4LLM 'ocr_language' must be a non-empty string")
            options["ocr_language"] = ocr_language

        table_output = self.base_config.get("table_output", "html")
        if not isinstance(table_output, str) or table_output.strip().lower() != "html":
            raise ProviderConfigError("PyMuPDF4LLM only supports table_output='html'")
        options["table_output"] = "html"

        raw_backend = self.base_config.get("ocr_backend")
        if raw_backend is None:
            return options
        if not isinstance(raw_backend, str):
            raise ProviderConfigError("PyMuPDF4LLM 'ocr_backend' must be a string")
        backend = raw_backend.strip().lower()
        if backend not in _OCR_BACKEND_MODULES:
            supported = ", ".join(_OCR_BACKEND_MODULES)
            raise ProviderConfigError(
                f"Unsupported PyMuPDF4LLM OCR backend '{raw_backend}'. Supported backends: {supported}"
            )
        return options

    def _resolve_ocr_function(self) -> Callable[..., Any] | None:
        """Resolve the configured bundled OCR engine immediately before use."""
        raw_backend = self.base_config.get("ocr_backend")
        if not isinstance(raw_backend, str):
            return None
        module_name = _OCR_BACKEND_MODULES.get(raw_backend.strip().lower())
        if module_name is None:
            return None
        try:
            ocr_module = importlib.import_module(module_name)
        except (ImportError, RuntimeError) as e:
            raise ProviderConfigError(f"PyMuPDF4LLM OCR backend '{raw_backend}' is unavailable: {e}") from e
        ocr_function = getattr(ocr_module, "exec_ocr", None)
        if not callable(ocr_function):
            raise ProviderConfigError(f"PyMuPDF4LLM OCR backend '{raw_backend}' does not expose exec_ocr")
        return cast(Callable[..., Any], ocr_function)

    def _extract(self, pdf_path: str) -> dict[str, Any]:
        try:
            import pymupdf
            import pymupdf4llm  # type: ignore[import-untyped]
        except ImportError as e:
            raise ProviderConfigError("pymupdf4llm not installed. Run: pip install pymupdf4llm") from e

        try:
            markdown_options = self._markdown_options()
            ocr_function = self._resolve_ocr_function()
            if ocr_function is None:
                page_chunks = pymupdf4llm.to_markdown(pdf_path, **markdown_options)
            else:
                page_chunks = pymupdf4llm.to_markdown(pdf_path, ocr_function=ocr_function, **markdown_options)
            with pymupdf.open(pdf_path) as document:
                page_dimensions = [(float(page.rect.width), float(page.rect.height)) for page in document]
        except ProviderConfigError:
            raise
        except Exception as e:
            raise ProviderPermanentError(f"PyMuPDF4LLM error: {e}") from e

        pages = []
        for i, chunk in enumerate(page_chunks):
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            raw_page_number = metadata.get("page_number") if isinstance(metadata, dict) else None
            if isinstance(raw_page_number, (int, float, str)):
                try:
                    page_number = int(raw_page_number)
                except ValueError:
                    page_number = i + 1
            else:
                page_number = i + 1

            dimension_index = page_number - 1
            if not 0 <= dimension_index < len(page_dimensions):
                dimension_index = i
            if 0 <= dimension_index < len(page_dimensions):
                width, height = page_dimensions[dimension_index]
            else:
                width, height = 0.0, 0.0

            page_boxes = chunk.get("page_boxes", []) if isinstance(chunk, dict) else []
            if not isinstance(page_boxes, list):
                page_boxes = []
            pages.append(
                {
                    "page_index": i,
                    "page_number": page_number,
                    "text": text,
                    "width": width,
                    "height": height,
                    "page_boxes": page_boxes,
                }
            )

        return {"pages": pages, "num_pages": len(pages)}

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"PyMuPDF4LLMProvider only supports PARSE, got {request.product_type}")

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"File not found: {pdf_path}")

        started_at = datetime.now()
        try:
            raw_output = self._extract(str(pdf_path))
            completed_at = datetime.now()
            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output=raw_output,
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
            )
        except (ProviderPermanentError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error: {e}") from e

    @staticmethod
    def _coerce_bbox(
        raw_bbox: Any,
        *,
        page_width: float,
        page_height: float,
    ) -> tuple[float, float, float, float] | None:
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return None
        try:
            x0, y0, x1, y1 = (float(value) for value in raw_bbox)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
            return None
        if page_width <= 0 or page_height <= 0:
            return None
        x0 = min(max(x0, 0.0), page_width)
        x1 = min(max(x1, 0.0), page_width)
        y0 = min(max(y0, 0.0), page_height)
        y1 = min(max(y1, 0.0), page_height)
        if x1 <= x0 or y1 <= y0:
            return None
        return (
            x0 / page_width,
            y0 / page_height,
            (x1 - x0) / page_width,
            (y1 - y0) / page_height,
        )

    @staticmethod
    def _coerce_text_range(raw_pos: Any, text_length: int) -> tuple[int, int] | None:
        if not isinstance(raw_pos, (list, tuple)) or len(raw_pos) != 2:
            return None
        start, stop = raw_pos
        if isinstance(start, bool) or isinstance(stop, bool):
            return None
        try:
            start = int(start)
            stop = int(stop)
        except (TypeError, ValueError):
            return None
        start = min(max(start, 0), text_length)
        stop = min(max(stop, 0), text_length)
        if stop < start:
            return None
        return start, stop

    @classmethod
    def _build_layout_page(
        cls,
        page_data: dict[str, Any],
        *,
        raw_markdown: str,
    ) -> ParseLayoutPageIR | None:
        """Convert native PyMuPDF page boxes into benchmark visual-grounding IR."""
        try:
            page_number = int(page_data.get("page_number", 0))
            page_width = float(page_data.get("width", 0.0))
            page_height = float(page_data.get("height", 0.0))
        except (TypeError, ValueError):
            return None
        if page_number < 1 or page_width <= 0 or page_height <= 0:
            return None

        items: list[LayoutItemIR] = []
        for page_box in page_data.get("page_boxes", []):
            if not isinstance(page_box, dict):
                continue
            raw_label = str(page_box.get("class", "")).strip()
            if not raw_label:
                continue
            normalized_class = raw_label.lower().replace("_", "-")

            bbox = cls._coerce_bbox(
                page_box.get("bbox"),
                page_width=page_width,
                page_height=page_height,
            )
            if bbox is None:
                continue
            text_range = cls._coerce_text_range(page_box.get("pos"), len(raw_markdown))
            if text_range is None:
                start_index = None
                end_index = None
                content = ""
            else:
                start_index, end_index = text_range
                content = raw_markdown[start_index:end_index]

            segment = LayoutSegmentIR(
                x=bbox[0],
                y=bbox[1],
                w=bbox[2],
                h=bbox[3],
                label=raw_label,
                start_index=start_index,
                end_index=end_index,
            )
            if normalized_class == "table":
                item_type = "table"
                item_html = content
            elif normalized_class == "picture":
                item_type = "image"
                item_html = ""
            else:
                item_type = "text"
                item_html = ""
            items.append(
                LayoutItemIR(
                    type=item_type,
                    md=content,
                    html=item_html,
                    value=content,
                    bbox=segment,
                    layout_segments=[segment],
                )
            )

        return ParseLayoutPageIR(
            page_number=page_number,
            width=page_width,
            height=page_height,
            md=raw_markdown,
            text=raw_markdown,
            items=items,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        page_texts: list[str] = []
        for page_data in raw_result.raw_output.get("pages", []):
            page_index = page_data.get("page_index", 0)
            raw_markdown = page_data.get("text", "") or ""
            layout_page = self._build_layout_page(page_data, raw_markdown=raw_markdown)
            if layout_page is not None:
                layout_pages.append(layout_page)
            pages.append(PageIR(page_index=page_index, markdown=raw_markdown))
            page_texts.append(raw_markdown)

        full_text = "\n\n".join(page_texts)
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=full_text,
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
