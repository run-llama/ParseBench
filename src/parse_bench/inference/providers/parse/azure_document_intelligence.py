"""Provider for Azure Document Intelligence PARSE."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
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

# Azure DI paragraph role -> Canonical17 label string
AZURE_DI_LABEL_MAP: dict[str, str] = {
    "title": "Title",
    "sectionHeading": "Section-header",
    "pageHeader": "Page-header",
    "pageFooter": "Page-footer",
    "footnote": "Footnote",
    "pageNumber": "Page-footer",
}

# Default label for paragraphs without a recognized role
_DEFAULT_PARAGRAPH_LABEL = "Text"

# Azure DI selection mark state -> Canonical17 checkbox label
AZURE_DI_SELECTION_LABEL_MAP: dict[str, str] = {
    "selected": "Checkbox-Selected",
    "unselected": "Checkbox-Unselected",
}

# Virtual page dimensions for normalized coordinate conversion.
# Azure DI polygons are normalized to [0,1] via page width/height, so these cancel out.
_VIRTUAL_PAGE_DIM = 1000.0


@register_provider("azure_document_intelligence")
class AzureDocumentIntelligenceProvider(Provider):
    """
    Provider for Azure Document Intelligence PARSE.

    This provider uses Azure AI Document Intelligence for parsing tasks.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Azure Document Intelligence API key
              (defaults to AZURE_DOCUMENT_INTELLIGENCE_KEY env var)
            - `endpoint`: Azure Document Intelligence endpoint URL
              (defaults to AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT env var)
            - `model_id`: Model to use for analysis (default: "prebuilt-layout")
              Options: "prebuilt-read", "prebuilt-layout", "prebuilt-document"
            - `output_content_format`: Output format - "text" or "markdown"
              (default: "markdown")
        """
        super().__init__(provider_name, base_config)

        # Get API key and endpoint
        self._api_key = self.base_config.get("api_key") or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        self._endpoint = self.base_config.get("endpoint") or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

        if not self._api_key:
            raise ProviderConfigError(
                "Azure Document Intelligence API key is required. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_KEY environment variable "
                "or pass api_key in base_config."
            )

        if not self._endpoint:
            raise ProviderConfigError(
                "Azure Document Intelligence endpoint is required. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable "
                "or pass endpoint in base_config."
            )

        # Get configuration with defaults
        self._model_id = self.base_config.get("model_id", "prebuilt-layout")
        self._output_content_format = self.base_config.get("output_content_format", "markdown")

        # Initialize client
        self._client = DocumentIntelligenceClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )

    def _parse_pdf(self, pdf_path: str) -> dict[str, Any]:
        """
        Parse a PDF using Azure Document Intelligence API.

        :param pdf_path: Path to the PDF file
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        try:
            # Read PDF file
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            # Analyze the document
            poller = self._client.begin_analyze_document(  # type: ignore[call-overload]
                self._model_id,
                body=pdf_bytes,
                output_content_format=self._output_content_format,
            )

            # Wait for completion and get result
            result: AnalyzeResult = poller.result()

            # Convert result to dictionary for raw storage
            raw_response = self._convert_result_to_dict(result)

            # Store configuration for reference
            raw_response["_config"] = {
                "model_id": self._model_id,
                "output_content_format": self._output_content_format,
            }

            return raw_response

        except FileNotFoundError as e:
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}") from e
        except Exception as e:
            # Check if it's a transient error
            error_str = str(e).lower()
            transient_keywords = [
                "timeout",
                "network",
                "connection",
                "503",
                "502",
                "504",
                "429",
                "throttl",
                "rate limit",
            ]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during parsing: {e}") from e
            else:
                raise ProviderPermanentError(f"Error during parsing: {e}") from e

    def _convert_result_to_dict(self, result: AnalyzeResult) -> dict[str, Any]:
        """
        Convert Azure Document Intelligence AnalyzeResult to dictionary.

        :param result: AnalyzeResult from Azure API
        :return: Dictionary representation of the result
        """
        response: dict[str, Any] = {}

        # Extract content (full document text/markdown)
        if result.content:
            response["content"] = result.content

        # Extract pages with their content
        if result.pages:
            pages_data = []
            for page in result.pages:
                page_dict: dict[str, Any] = {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "unit": page.unit,
                }

                # Extract lines if available
                if page.lines:
                    page_dict["lines"] = [
                        {
                            "content": line.content,
                            "polygon": line.polygon if line.polygon else None,
                        }
                        for line in page.lines
                    ]

                # Extract words if available
                if page.words:
                    page_dict["words"] = [
                        {
                            "content": word.content,
                            "polygon": list(word.polygon) if word.polygon else None,
                            "confidence": word.confidence if word.confidence is not None else None,
                        }
                        for word in page.words
                    ]
                    page_dict["word_count"] = len(page.words)

                # Extract selection marks (checkboxes) if available
                selection_marks = getattr(page, "selection_marks", None)
                if selection_marks:
                    page_dict["selection_marks"] = [
                        {
                            "state": mark.state,
                            "polygon": list(mark.polygon) if mark.polygon else None,
                            "confidence": mark.confidence if mark.confidence is not None else None,
                        }
                        for mark in selection_marks
                    ]

                pages_data.append(page_dict)

            response["pages"] = pages_data

        # Extract tables if available
        if result.tables:
            tables_data = []
            for table in result.tables:
                table_dict: dict[str, Any] = {
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "cells": [],
                }

                if table.cells:
                    for cell in table.cells:
                        cell_dict: dict[str, Any] = {
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "content": cell.content,
                            "row_span": cell.row_span,
                            "column_span": cell.column_span,
                            # Azure tags structural roles via cell.kind:
                            # "columnHeader" / "rowHeader" / "stubHead" / "description"
                            # / "content" (default). Used to choose <th> vs <td>
                            # when reconstructing table HTML downstream.
                            "kind": getattr(cell, "kind", None),
                        }
                        # Per-cell polygons let downstream tooling overlay
                        # individual cell bboxes. Some Azure tables only carry a
                        # table-level region, so this is best-effort optional.
                        cell_bounding_regions = getattr(cell, "bounding_regions", None)
                        if cell_bounding_regions:
                            cell_dict["bounding_regions"] = [
                                {
                                    "page_number": br.page_number,
                                    "polygon": list(br.polygon) if br.polygon else None,
                                }
                                for br in cell_bounding_regions
                            ]
                        table_dict["cells"].append(cell_dict)

                tables_data.append(table_dict)

            response["tables"] = tables_data

        # Extract paragraphs if available (with bounding regions for layout)
        if result.paragraphs:
            paragraphs_data = []
            for para in result.paragraphs:
                para_dict: dict[str, Any] = {
                    "content": para.content,
                    "role": para.role if para.role else None,
                }
                if para.bounding_regions:
                    para_dict["bounding_regions"] = [
                        {
                            "page_number": br.page_number,
                            "polygon": list(br.polygon) if br.polygon else None,
                        }
                        for br in para.bounding_regions
                    ]
                paragraphs_data.append(para_dict)
            response["paragraphs"] = paragraphs_data

        # Extract tables if available (with bounding regions for layout)
        if result.tables:
            for i, table in enumerate(result.tables):
                if table.bounding_regions and i < len(response.get("tables", [])):
                    response["tables"][i]["bounding_regions"] = [
                        {
                            "page_number": br.page_number,
                            "polygon": list(br.polygon) if br.polygon else None,
                        }
                        for br in table.bounding_regions
                    ]

        # Extract figures if available (with bounding regions for layout)
        if result.figures:
            response["figures"] = [
                {
                    "caption": fig.caption.content if fig.caption else None,
                    "bounding_regions": [
                        {
                            "page_number": br.page_number,
                            "polygon": list(br.polygon) if br.polygon else None,
                        }
                        for br in fig.bounding_regions
                    ]
                    if fig.bounding_regions
                    else [],
                }
                for fig in result.figures
            ]

        # Extract key-value pairs if available
        if result.key_value_pairs:
            response["key_value_pairs"] = [
                {
                    "key": kvp.key.content if kvp.key else None,
                    "value": kvp.value.content if kvp.value else None,
                }
                for kvp in result.key_value_pairs
            ]

        return response

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AzureDocumentIntelligenceProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}")

        try:
            # Run parsing
            raw_output = self._parse_pdf(str(pdf_path))

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output=raw_output,
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AzureDocumentIntelligenceProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract the main content (markdown or text)
        content = raw_result.raw_output.get("content", "")

        # Build page-level data if available
        pages: list[PageIR] = []
        raw_pages = raw_result.raw_output.get("pages", [])

        if raw_pages:
            # Azure returns page boundaries, we can try to split content by pages
            # For now, we'll create page entries with line-based content
            for page_data in raw_pages:
                page_num = page_data.get("page_number", 1)
                page_index = page_num - 1  # Convert to 0-indexed

                # Reconstruct page content from lines if available
                page_content = ""
                if "lines" in page_data:
                    page_content = "\n".join(line.get("content", "") for line in page_data.get("lines", []))

                pages.append(PageIR(page_index=page_index, markdown=page_content))

        # Build layout_pages for layout cross-evaluation
        layout_pages = _build_layout_pages(raw_result.raw_output)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=content,
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


def _polygon_to_normalized_bbox(
    polygon: list[float],
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    """Convert Azure DI polygon (8 floats, 4 corner points in page units) to normalized [0,1] xywh.

    The polygon contains [x1,y1, x2,y2, x3,y3, x4,y4] in the page's coordinate
    system (typically inches). We take min/max to get axis-aligned bbox, then
    normalize by page dimensions.
    """
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    # Normalize to [0, 1]
    nx = x_min / page_width if page_width > 0 else 0.0
    ny = y_min / page_height if page_height > 0 else 0.0
    nw = (x_max - x_min) / page_width if page_width > 0 else 0.0
    nh = (y_max - y_min) / page_height if page_height > 0 else 0.0

    return (nx, ny, nw, nh)


def _build_table_html_from_cells(
    cells: list[dict[str, Any]],
    row_count: int,
    column_count: int,
) -> str:
    """Reconstruct a ``<table>`` HTML string from Azure DI cell dicts.

    Honors ``row_span`` / ``column_span`` (cells covered by a previous
    cell's span are skipped, not double-emitted) and Azure's ``cell.kind``
    (``"columnHeader"`` / ``"rowHeader"`` → ``<th>``; everything else →
    ``<td>``). Cells without integer ``row_index`` / ``column_index`` are
    skipped silently — they're structural only.

    Returns ``""`` when input is degenerate (no cells, missing row/col
    counts) so callers can fall back to flat-text content.
    """
    from html import escape as _html_escape

    if not cells or row_count <= 0 or column_count <= 0:
        return ""

    grid: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in cells:
        r = cell.get("row_index")
        c = cell.get("column_index")
        if isinstance(r, int) and isinstance(c, int):
            grid[(r, c)] = cell

    covered: set[tuple[int, int]] = set()
    rows_html: list[str] = []
    for r in range(row_count):
        cells_html: list[str] = []
        for c in range(column_count):
            if (r, c) in covered:
                continue
            cell_at_pos = grid.get((r, c))
            if cell_at_pos is None:
                cells_html.append("<td></td>")
                continue
            row_span = int(cell_at_pos.get("row_span") or 1)
            col_span = int(cell_at_pos.get("column_span") or 1)
            for dr in range(row_span):
                for dc in range(col_span):
                    if dr == 0 and dc == 0:
                        continue
                    covered.add((r + dr, c + dc))
            attrs = ""
            if row_span > 1:
                attrs += f' rowspan="{row_span}"'
            if col_span > 1:
                attrs += f' colspan="{col_span}"'
            kind = (cell_at_pos.get("kind") or "").strip()
            tag = "th" if kind in ("columnHeader", "rowHeader") else "td"
            content = _html_escape(cell_at_pos.get("content") or "")
            cells_html.append(f"<{tag}{attrs}>{content}</{tag}>")
        # Always emit a <tr> for every row in row_count, even when every
        # position is covered by a rowspan from a previous row. HTML
        # counts <tr> elements to resolve rowspan targets — dropping
        # the empty <tr> would push the next row's cells into the wrong
        # row index. The empty <tr></tr> renders as an invisible row in
        # browsers and parses cleanly under lxml + BeautifulSoup, which
        # is what the TEDS evaluator uses.
        rows_html.append("<tr>" + "".join(cells_html) + "</tr>")

    return "<table>" + "".join(rows_html) + "</table>"


def _bbox_center_inside_any(
    bbox: tuple[float, float, float, float],
    others: list[tuple[float, float, float, float]],
) -> bool:
    """``True`` when the center of ``bbox`` lies inside any rectangle in
    ``others``. Both rectangles are normalized xywh in the same coord
    system. Used to filter Azure DI cell-as-paragraph duplicates.
    """
    cx = bbox[0] + bbox[2] / 2.0
    cy = bbox[1] + bbox[3] / 2.0
    for ox, oy, ow, oh in others:
        if ox <= cx <= ox + ow and oy <= cy <= oy + oh:
            return True
    return False


def _build_layout_pages(raw_output: dict[str, Any]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from Azure DI paragraphs/tables/figures for layout cross-evaluation.

    Groups elements by page using bounding_regions and converts Azure DI polygon
    coordinates (in page units) into normalized [0,1] LayoutSegmentIR entries.
    """
    from collections import defaultdict

    # Build page dimension lookup from pages data
    page_dims: dict[int, tuple[float, float]] = {}
    for page_data in raw_output.get("pages", []):
        page_num = page_data.get("page_number", 1)
        width = float(page_data.get("width", 1.0))
        height = float(page_data.get("height", 1.0))
        page_dims[page_num] = (width, height)

    # Collect all layout elements grouped by page:
    # (canonical_label, nx, ny, nw, nh, value_text, confidence, html_or_empty).
    # html_or_empty is populated only for table items; other element kinds
    # push an empty string so the tuple stays uniform.
    pages_items: dict[int, list[tuple[str, float, float, float, float, str, float, str]]] = defaultdict(list)

    # Pre-compute table rectangles per page so we can skip paragraphs
    # whose bbox sits inside one. Azure DI returns each table cell *both*
    # as a ``tables[*].cells`` entry (which we use for HTML
    # reconstruction) and as a standalone ``paragraphs[]`` entry —
    # typically with no role and the same bbox as the cell. Without this
    # filter the layout output ends up with one Text bbox per cell on
    # top of the parent Table region.
    table_rects_by_page: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for table in raw_output.get("tables", []):
        for br in table.get("bounding_regions", []):
            page_num = br.get("page_number", 1)
            polygon = br.get("polygon")
            if not polygon or len(polygon) < 8:
                continue
            pw, ph = page_dims.get(page_num, (1.0, 1.0))
            table_rects_by_page[page_num].append(_polygon_to_normalized_bbox(polygon, pw, ph))

    # Process paragraphs
    for para in raw_output.get("paragraphs", []):
        role = para.get("role")
        canonical_label = AZURE_DI_LABEL_MAP.get(role, _DEFAULT_PARAGRAPH_LABEL) if role else _DEFAULT_PARAGRAPH_LABEL
        content = para.get("content", "")

        for br in para.get("bounding_regions", []):
            page_num = br.get("page_number", 1)
            polygon = br.get("polygon")
            if not polygon or len(polygon) < 8:
                continue
            pw, ph = page_dims.get(page_num, (1.0, 1.0))
            nx, ny, nw, nh = _polygon_to_normalized_bbox(polygon, pw, ph)
            if _bbox_center_inside_any((nx, ny, nw, nh), table_rects_by_page.get(page_num, [])):
                # Cell-as-paragraph duplicate. Cell content already lives
                # in the table region's HTML / value via
                # _build_table_html_from_cells; emitting it again would
                # double-count.
                continue
            pages_items[page_num].append((canonical_label, nx, ny, nw, nh, content, 1.0, ""))

    # Process tables
    for table in raw_output.get("tables", []):
        # Flat text fallback for the LayoutItemIR.value field — kept for
        # compatibility with callers that don't read .html. Structured
        # HTML reconstruction (handed to LayoutItemIR.html) preserves
        # <th>/<td> roles and spans for table metrics.
        cells = table.get("cells", [])
        content = " ".join(c.get("content", "") for c in cells if c.get("content"))
        html_content = _build_table_html_from_cells(
            cells,
            int(table.get("row_count") or 0),
            int(table.get("column_count") or 0),
        )

        for br in table.get("bounding_regions", []):
            page_num = br.get("page_number", 1)
            polygon = br.get("polygon")
            if not polygon or len(polygon) < 8:
                continue
            pw, ph = page_dims.get(page_num, (1.0, 1.0))
            nx, ny, nw, nh = _polygon_to_normalized_bbox(polygon, pw, ph)
            pages_items[page_num].append(("Table", nx, ny, nw, nh, content, 1.0, html_content))

    # Process figures
    for fig in raw_output.get("figures", []):
        caption = fig.get("caption") or ""
        for br in fig.get("bounding_regions", []):
            page_num = br.get("page_number", 1)
            polygon = br.get("polygon")
            if not polygon or len(polygon) < 8:
                continue
            pw, ph = page_dims.get(page_num, (1.0, 1.0))
            nx, ny, nw, nh = _polygon_to_normalized_bbox(polygon, pw, ph)
            pages_items[page_num].append(("Picture", nx, ny, nw, nh, caption, 1.0, ""))

    # Process selection marks (checkboxes) — live on page objects, not at document root
    for page_data in raw_output.get("pages", []):
        page_num = page_data.get("page_number", 1)
        pw, ph = page_dims.get(page_num, (1.0, 1.0))
        for mark in page_data.get("selection_marks", []) or []:
            state = mark.get("state") or ""
            checkbox_label = AZURE_DI_SELECTION_LABEL_MAP.get(state)
            if checkbox_label is None:
                continue
            polygon = mark.get("polygon")
            if not polygon or len(polygon) < 8:
                continue
            nx, ny, nw, nh = _polygon_to_normalized_bbox(polygon, pw, ph)
            confidence = mark.get("confidence")
            confidence_val = float(confidence) if confidence is not None else 1.0
            pages_items[page_num].append((checkbox_label, nx, ny, nw, nh, "", confidence_val, ""))

    # Build ParseLayoutPageIR list
    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_items.keys()):
        items_data = pages_items[page_num]
        items: list[LayoutItemIR] = []

        for canonical_label, nx, ny, nw, nh, content, confidence, html_content in items_data:
            seg = LayoutSegmentIR(
                x=nx,
                y=ny,
                w=nw,
                h=nh,
                confidence=confidence,
                label=canonical_label,
            )

            norm_label = canonical_label.strip().lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label == "picture":
                item_type = "image"
            elif canonical_label in AZURE_DI_SELECTION_LABEL_MAP.values():
                item_type = "checkbox"
            else:
                item_type = "text"

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=content,
                    html=html_content,
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_num,
                width=_VIRTUAL_PAGE_DIM,
                height=_VIRTUAL_PAGE_DIM,
                items=items,
            )
        )

    return layout_pages
