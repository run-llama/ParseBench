"""Provider for PyMuPDF4LLM PARSE."""

import html
import importlib
import math
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from markdown_it import MarkdownIt

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

# Raw HTML is disabled so extracted document content cannot inject markup while
# cell-level Markdown is rendered. A separate parser identifies blocks where
# table recovery must not run.
_MD = MarkdownIt("commonmark", {"html": False}).enable("table")
_BLOCK_MD = MarkdownIt("commonmark")
_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BR_PLACEHOLDER = "\ue000"

_OCR_BACKEND_MODULES = {
    "rapidtess": "pymupdf4llm.ocr.rapidtess_api",
}


def _is_pipe_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    stripped = line.strip().strip("|")
    if not stripped:
        return False
    cells = [cell.strip() for cell in stripped.split("|")]
    return all(cell and re.fullmatch(r":?-+:?", cell) for cell in cells)


def _split_pipe_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    code_delimiter_length = 0
    i = 0
    while i < len(stripped):
        char = stripped[i]
        if char == "\\" and i + 1 < len(stripped):
            current.extend((char, stripped[i + 1]))
            i += 2
            continue
        if char == "`":
            end = i + 1
            while end < len(stripped) and stripped[end] == "`":
                end += 1
            delimiter_length = end - i
            if code_delimiter_length == 0:
                code_delimiter_length = delimiter_length
            elif code_delimiter_length == delimiter_length:
                code_delimiter_length = 0
            current.extend(stripped[i:end])
            i = end
            continue
        if char == "|" and code_delimiter_length == 0:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        i += 1

    cells.append("".join(current).strip())
    return cells


def _render_cell_inline(cell: str) -> str:
    if cell.startswith("**") and not cell.endswith("**"):
        cell = cell[2:]
    if cell.endswith("**") and not cell.startswith("**"):
        cell = cell[:-2]
    # PyMuPDF4LLM uses HTML line breaks inside Markdown table cells. Preserve
    # only that structural tag while keeping arbitrary raw HTML disabled.
    protected_cell = _BR_TAG_RE.sub(_BR_PLACEHOLDER, cell)
    rendered = _MD.renderInline(protected_cell).strip().replace(_BR_PLACEHOLDER, "<br>")
    return rendered if rendered else html.escape(cell)


def _render_html_table(
    rows: list[list[str]],
    *,
    header_rows: int = 1,
    alignments: list[str | None] | None = None,
) -> str:
    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
    normalized_alignments = (alignments or []) + [None] * max_cols

    def render_cell(tag: str, cell: str, column_index: int) -> str:
        alignment = normalized_alignments[column_index]
        style = f' style="text-align:{alignment}"' if alignment else ""
        return f"<{tag}{style}>{_render_cell_inline(cell)}</{tag}>"

    parts = ["<table>"]
    if header_rows > 0:
        parts.append("<thead>")
        for row in normalized_rows[:header_rows]:
            cells = "".join(render_cell("th", cell, index) for index, cell in enumerate(row))
            parts.append(f"<tr>{cells}</tr>")
        parts.append("</thead>")

    body_rows = normalized_rows[header_rows:]
    if body_rows:
        parts.append("<tbody>")
        for row in body_rows:
            cells = "".join(render_cell("td", cell, index) for index, cell in enumerate(row))
            parts.append(f"<tr>{cells}</tr>")
        parts.append("</tbody>")

    parts.append("</table>")
    return "\n".join(parts)


def _render_forgiving_pipe_table(block: list[str], *, min_columns: int = 2) -> str | None:
    if len(block) < 2:
        return None
    separator_idx = next((idx for idx, line in enumerate(block) if _is_separator_row(line)), None)
    if separator_idx is None:
        return None

    alignments: list[str | None] = []
    for cell in _split_pipe_row(block[separator_idx]):
        stripped = cell.strip()
        if stripped.startswith(":") and stripped.endswith(":"):
            alignments.append("center")
        elif stripped.startswith(":"):
            alignments.append("left")
        elif stripped.endswith(":"):
            alignments.append("right")
        else:
            alignments.append(None)

    header = [_split_pipe_row(line) for line in block[:separator_idx]]
    body = [_split_pipe_row(line) for line in block[separator_idx + 1 :]]
    while header and not any(cell.strip() for cell in header[0]):
        header.pop(0)

    rows = [*header, *body]
    non_empty_rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(non_empty_rows) < 2 or max(len(row) for row in non_empty_rows) < min_columns:
        return None
    return _render_html_table(rows, header_rows=len(header), alignments=alignments)


def _protected_block_lines(text: str) -> set[int]:
    protected: set[int] = set()
    for token in _BLOCK_MD.parse(text):
        if token.type in {"fence", "code_block", "html_block"} and token.map:
            start, end = token.map
            protected.update(range(start, end))
    return protected


def _render_strict_pipe_tables(text: str) -> str:
    tokens = _MD.parse(text)
    lines = text.split("\n")
    protected = _protected_block_lines(text)
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type != "table_open" or not token.map:
            i += 1
            continue
        start, end = token.map
        j = i
        while j < len(tokens) and tokens[j].type != "table_close":
            j += 1
        if not any(line_number in protected for line_number in range(start, end)):
            # Markdown-it has already established that this is a structural
            # table, so valid one-column tables are safe to render here.
            rendered = _render_forgiving_pipe_table(lines[start:end], min_columns=1)
            if rendered is not None:
                spans.append((start, end, rendered))
        i = j + 1

    for start, end, rendered in sorted(spans, reverse=True):
        lines[start:end] = [rendered]
    return "\n".join(lines)


def _render_forgiving_pipe_tables(text: str) -> str:
    lines = text.split("\n")
    protected = _protected_block_lines(text)
    result: list[str] = []
    i = 0
    while i < len(lines):
        if i in protected or not _is_pipe_table_line(lines[i]):
            result.append(lines[i])
            i += 1
            continue
        start = i
        while i < len(lines) and i not in protected and _is_pipe_table_line(lines[i]):
            i += 1
        block = lines[start:i]
        rendered = _render_forgiving_pipe_table(block)
        result.extend(block if rendered is None else [rendered])
    return "\n".join(result)


def convert_pipe_tables_to_html(text: str) -> str:
    """Convert structural GFM pipe tables without touching protected blocks."""
    converted = _render_strict_pipe_tables(text) if "|" in text else text
    return _render_forgiving_pipe_tables(converted) if "|" in converted else converted


@register_provider("pymupdf4llm")
class PyMuPDF4LLMProvider(Provider):
    """Provider for PyMuPDF4LLM (markdown). AGPL — runtime dep only."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

    def _markdown_options(self) -> dict[str, Any]:
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
        if getattr(ocr_module, "TESSDATA", True) is None:
            raise ProviderConfigError(
                f"PyMuPDF4LLM OCR backend '{raw_backend}' is unavailable: Tesseract language data was not found"
            )
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
    def _convert_md_tables_to_html(content: str) -> str:
        return convert_pipe_tables_to_html(content)

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

            confidence: float | None = None
            raw_confidence = page_box.get("confidence")
            if raw_confidence is not None:
                try:
                    parsed_confidence = float(raw_confidence)
                except (TypeError, ValueError):
                    parsed_confidence = math.nan
                if math.isfinite(parsed_confidence) and 0.0 <= parsed_confidence <= 1.0:
                    confidence = parsed_confidence

            segment = LayoutSegmentIR(
                x=bbox[0],
                y=bbox[1],
                w=bbox[2],
                h=bbox[3],
                confidence=confidence,
                label=raw_label,
                start_index=start_index,
                end_index=end_index,
            )
            if normalized_class == "table":
                item_type = "table"
                item_html = cls._convert_md_tables_to_html(content)
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
            text = self._convert_md_tables_to_html(raw_markdown)
            pages.append(PageIR(page_index=page_index, markdown=text))
            page_texts.append(text)

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
