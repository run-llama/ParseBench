"""Provider for Databricks ``ai_parse_document`` via the direct REST API."""

from __future__ import annotations

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from parse_bench.evaluation.metrics.parse.chart_json_to_html import (
    chart_description_to_html,
    chart_json_to_html,
)
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

# ai_parse_document element type -> Canonical17 label
DATABRICKS_LABEL_MAP: dict[str, str] = {
    "title": "Title",
    "section_header": "Section-header",
    "text": "Text",
    "table": "Table",
    "figure": "Picture",
    "caption": "Caption",
    "page_header": "Page-header",
    "page_footer": "Page-footer",
    "page_number": "Page-footer",
    "footnote": "Footnote",
    "signature": "Picture",
}

# ai_parse_document returns element bboxes in absolute pixel coordinates of
# the page it rendered internally. For PDFs that render happens at 200 DPI
# (v2 default), so true page dims in that space are points * 200/72; image
# inputs keep their native pixel dims. Normalizing by anything else (e.g. the
# max element extent on the page) shifts and stretches every box.
AI_PARSE_RENDER_DPI = 200.0

# Fallback page dimension when the source file can't be read at normalize
# time (bboxes then stay normalized by element extent — degraded but usable).
_VIRTUAL_PAGE_DIM = 1000.0

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}


@register_provider("databricks_ai_parse")
class DatabricksAiParseProvider(Provider):
    """Parse local documents without a SQL warehouse or staging volume.

    Config:
        - host (str, required): Workspace URL; defaults to DATABRICKS_HOST.
        - token (str, required): Bearer token; defaults to DATABRICKS_TOKEN.
        - version (str, default "2.0"): ai_parse_document schema version.
        - description_element_types (str, default ""): Optional descriptionElementTypes override.
        - timeout (int, default 900): Request timeout in seconds.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        host = self.base_config.get("host") or os.getenv("DATABRICKS_HOST")
        token = self.base_config.get("token") or os.getenv("DATABRICKS_TOKEN")
        if not host:
            raise ProviderConfigError(
                "Databricks host is required. Set DATABRICKS_HOST env var or pass 'host' in base_config."
            )
        if not token:
            raise ProviderConfigError(
                "Databricks token is required. Set DATABRICKS_TOKEN env var or pass 'token' in base_config."
            )
        self._base_url = f"https://{host.rstrip('/').removeprefix('https://').removeprefix('http://')}"
        self._auth_headers = {"Authorization": f"Bearer {token}"}
        self._version = str(self.base_config.get("version", "2.0"))
        self._description_element_types = self.base_config.get("description_element_types", "")
        self._timeout = int(self.base_config.get("timeout", 900))

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"DatabricksAiParseProvider only supports PARSE, got {request.product_type}")
        source = Path(request.source_file_path)
        if not source.is_file():
            raise ProviderPermanentError(f"Source file not found: {source}")
        options = {"version": self._version}
        if self._description_element_types:
            options["descriptionElementTypes"] = self._description_element_types

        started_at = datetime.now()
        response = requests.post(
            f"{self._base_url}/api/2.0/ai-functions/ai_parse_document",
            headers=self._auth_headers,
            json={
                "content": [{"type": "file", "data": base64.b64encode(source.read_bytes()).decode("ascii")}],
                "options": options,
            },
            timeout=self._timeout,
        )
        if not response.ok:
            error = ProviderTransientError if response.status_code in _TRANSIENT_HTTP else ProviderPermanentError
            raise error(f"HTTP {response.status_code} during parse document: {response.text[:500]}")
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise ProviderPermanentError("Databricks ai_parse_document response is not an object.")
        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "ai_parse_document": parsed,
                "_config": {"transport": "rest", **options},
            },
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DatabricksAiParseProvider only supports PARSE, got {raw_result.product_type}"
            )

        variant = raw_result.raw_output.get("ai_parse_document") or {}
        document = variant.get("document") or {}
        elements: list[dict[str, Any]] = document.get("elements") or []

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            layout_pages=_build_layout_pages(document, raw_result.request.source_file_path),
            markdown=_render_markdown(elements),
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


def _primary_page_id(element: dict[str, Any]) -> int:
    bboxes = element.get("bbox") or []
    for box in bboxes:
        pid = box.get("page_id")
        if pid is not None:
            try:
                return int(pid)
            except (TypeError, ValueError):
                continue
    return 0


def _chart_value_to_html(value: Any) -> str:
    if isinstance(value, dict):
        return chart_json_to_html(value)
    if isinstance(value, str):
        return chart_description_to_html(value)
    return ""


def _figure_chart_html(element: dict[str, Any]) -> str:
    """Render current chart content, falling back to legacy descriptions."""
    return _chart_value_to_html(element.get("content")) or _chart_value_to_html(element.get("description"))


def _render_markdown(elements: list[dict[str, Any]]) -> str:
    """Concatenate element content in reading order, grouped by page."""
    from collections import defaultdict

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for el in elements:
        by_page[_primary_page_id(el)].append(el)

    parts: list[str] = []
    for page_id in sorted(by_page.keys()):
        for el in sorted(by_page[page_id], key=lambda e: e.get("id", 0)):
            raw_content = el.get("content")
            content = raw_content.strip() if isinstance(raw_content, str) else ""
            el_type = (el.get("type") or "").lower()
            content_chart_tables = _chart_value_to_html(raw_content) if el_type == "figure" else ""
            if content:
                if el_type == "title":
                    parts.append(f"# {content}")
                elif el_type == "section_header":
                    parts.append(f"## {content}")
                elif not content_chart_tables:
                    parts.append(content)
            if el_type == "figure":
                chart_tables = content_chart_tables or _chart_value_to_html(el.get("description"))
                if chart_tables:
                    parts.append(chart_tables)
    return "\n\n".join(parts)


def _rendered_page_dims(source_file_path: str) -> list[tuple[float, float]] | None:
    """Per-page (width, height) of ai_parse's internally rendered pages, in
    the same pixel space as the returned bbox coordinates.

    Image inputs are processed at native pixel dims; PDFs are rendered at
    ``AI_PARSE_RENDER_DPI``, so true dims are page points * DPI/72. Returns
    None when the source file is missing or unreadable (e.g. renormalizing
    on a machine without the dataset) so callers can fall back gracefully.
    """
    path = Path(source_file_path)
    if not path.is_file():
        return None
    try:
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            from PIL import Image

            with Image.open(path) as im:
                return [(float(im.width), float(im.height))]

        import fitz  # PyMuPDF

        scale = AI_PARSE_RENDER_DPI / 72.0
        with fitz.open(path) as doc:
            return [(page.rect.width * scale, page.rect.height * scale) for page in doc]
    except Exception:  # noqa: BLE001 — never fail normalization over page dims
        return None


def _build_layout_pages(document: dict[str, Any], source_file_path: str) -> list[ParseLayoutPageIR]:
    """Group elements by page and convert bboxes to normalized LayoutSegmentIR.

    Coordinates are normalized into [0,1] by the true rendered page
    dimensions (see ``_rendered_page_dims``). When dims can't be derived the
    page falls back to normalizing by the max element extent, which keeps
    boxes on-page but loses absolute position and aspect ratio.
    """
    from collections import defaultdict

    elements: list[dict[str, Any]] = document.get("elements") or []

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for el in elements:
        for box in el.get("bbox") or []:
            page_id = box.get("page_id")
            if page_id is None:
                continue
            try:
                by_page[int(page_id)].append({"element": el, "coord": box.get("coord")})
            except (TypeError, ValueError):
                continue

    # ai_parse page ids are 0-indexed; derive the base from document.pages
    # (rather than assuming) so 1-indexed responses would still map cleanly.
    declared_ids = [p.get("id") for p in document.get("pages") or [] if isinstance(p.get("id"), int)]
    id_base = min(declared_ids) if declared_ids else 0

    rendered_dims = _rendered_page_dims(source_file_path)

    layout_pages: list[ParseLayoutPageIR] = []
    for page_id in sorted(by_page.keys()):
        entries = by_page[page_id]
        max_x = 1.0
        max_y = 1.0
        for entry in entries:
            coord = entry["coord"] or []
            if len(coord) >= 4:
                max_x = max(max_x, float(coord[2]))
                max_y = max(max_y, float(coord[3]))

        page_index = page_id - id_base
        dims = rendered_dims[page_index] if rendered_dims is not None and 0 <= page_index < len(rendered_dims) else None
        if dims is not None:
            # If an element extends past the computed page edge the render-DPI
            # assumption is off for this file; widen the denominator so boxes
            # stay in [0,1] instead of drifting off-page.
            denom_x = max(dims[0], max_x)
            denom_y = max(dims[1], max_y)
            page_w, page_h = dims
        else:
            denom_x, denom_y = max_x, max_y
            page_w = page_h = _VIRTUAL_PAGE_DIM

        items: list[LayoutItemIR] = []
        for entry in entries:
            el = entry["element"]
            el_type = (el.get("type") or "").lower()
            coord = entry["coord"] or []
            if len(coord) < 4:
                continue
            x1, y1, x2, y2 = (float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3]))
            w = max(x2 - x1, 0.0)
            h = max(y2 - y1, 0.0)

            canonical = DATABRICKS_LABEL_MAP.get(el_type)
            if canonical is None:
                continue

            seg = LayoutSegmentIR(
                x=x1 / denom_x,
                y=y1 / denom_y,
                w=w / denom_x,
                h=h / denom_y,
                confidence=float(el.get("confidence")) if el.get("confidence") is not None else None,
                label=canonical,
            )

            norm_label = canonical.strip().lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label == "picture":
                item_type = "image"
            else:
                item_type = "text"

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=el.get("content") or "",
                    html=(_figure_chart_html(el) if el_type == "figure" else ""),
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        layout_pages.append(
            ParseLayoutPageIR(
                page_number=max(page_index + 1, 1),
                width=page_w,
                height=page_h,
                items=items,
            )
        )

    return layout_pages
