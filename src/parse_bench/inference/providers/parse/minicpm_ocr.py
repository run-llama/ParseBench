"""Provider for a MiniCPM-OCR 1.0 SDK server.

MiniCPM-OCR 1.0 (openbmb/MiniCPM-OCR-1.0, 1.3B) is a region recognizer driven by
the official ``minicpmvocr`` SDK: PP-DocLayoutV3 detects and orders regions, each
crop is recognized with its label's prompt (text, HTML tables, LaTeX formulas,
charts to tables), and the SDK assembles page Markdown. The SDK server takes the
source file as-is, so PDF rasterization is the SDK's own.

Server contract: ``POST {"file_base64", "filename"}`` returns ``{"status",
"markdown", "pages": [{"page_index", "image_width", "image_height", "markdown",
"blocks": [{"label", "bbox" (pixel xyxy), "content", "order"}]}]}``, i.e. each
page's ``results_json`` from the SDK's ``pipelined_run``.
"""

import asyncio
import base64
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._layout_utils import build_layout_pages
from parse_bench.inference.providers.parse.mistral_ocr import _convert_pipe_tables_to_html
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

SUPPORTED_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp")

# PP-DocLayoutV3 label -> canonical label. ``number`` (page numbers) is filed as
# page furniture by position in ``_canonical_label``.
_PP_DOCLAYOUT_V3_TO_CANONICAL: dict[str, str] = {
    "doc_title": "Title",
    "paragraph_title": "Section-header",
    "header": "Page-header",
    "footer": "Page-footer",
    "text": "Text",
    "content": "Text",
    "abstract": "Text",
    "aside_text": "Text",
    "vertical_text": "Text",
    "reference": "Text",
    "reference_content": "Text",
    "formula_number": "Text",
    "footnote": "Footnote",
    "vision_footnote": "Footnote",
    "image": "Picture",
    "chart": "Picture",
    "seal": "Picture",
    "header_image": "Picture",
    "footer_image": "Picture",
    "figure_title": "Caption",
    "table": "Table",
    "display_formula": "Formula",
    "inline_formula": "Formula",
    "algorithm": "Code",
}


def _canonical_label(raw_label: str, bbox: list[float]) -> str | None:
    """Map a PP-DocLayoutV3 label to a canonical label (``bbox`` is 0-1000 xyxy)."""
    key = raw_label.strip().lower()
    if key == "number":
        return "Page-header" if (bbox[1] + bbox[3]) / 2.0 < 500.0 else "Page-footer"
    return _PP_DOCLAYOUT_V3_TO_CANONICAL.get(key)


def _layout_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    """SDK regions (pixel xyxy) -> ``build_layout_pages`` items (canonical labels, 0-1000)."""
    width = float(page.get("image_width") or 0)
    height = float(page.get("image_height") or 0)
    if width <= 0 or height <= 0:
        return []

    items: list[dict[str, Any]] = []
    for block in sorted(page.get("blocks") or [], key=lambda b: b.get("order", 0)):
        bbox = block.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        # Clamped so build_layout_pages never mistakes an edge box for pixel coordinates.
        scaled = (x1 * 1000 / width, y1 * 1000 / height, x2 * 1000 / width, y2 * 1000 / height)
        normalized = [min(1000.0, max(0.0, v)) for v in scaled]
        label = _canonical_label(str(block.get("label") or ""), normalized)
        if label is None:
            continue
        items.append({"label": label, "bbox": normalized, "text": str(block.get("content") or "").strip()})
    return items


@register_provider("minicpm_ocr")
class MiniCPMOCRProvider(Provider):
    """
    Provider for a MiniCPM-OCR 1.0 SDK server.

    Configuration options:
        - server_url (str, required): SDK server /predict URL. Falls back to the
          MINICPM_OCR_SERVER_URL environment variable.
        - timeout (int, default=1200): Request timeout in seconds (one request per document)
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("MINICPM_OCR_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "MiniCPM-OCR provider requires 'server_url' in config or the "
                "MINICPM_OCR_SERVER_URL environment variable (the SDK server /predict URL)."
            )
        self._server_url: str = str(server_url)
        self._timeout = self.base_config.get("timeout", 1200)

    async def _predict(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        payload = {"file_base64": base64.b64encode(file_bytes).decode(), "filename": filename}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._server_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    if resp.status in (408, 429, 500, 502, 503, 504):
                        raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                    raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")
                result: dict[str, Any] = await resp.json()

        if result.get("status") != "success":
            # Failed SDK runs are region requests that timed out or errored on the
            # recognition backend; retry the document rather than score a partial page.
            raise ProviderTransientError(f"MiniCPM-OCR SDK error: {result.get('error', 'unknown error')}")
        if not result.get("pages"):
            raise ProviderPermanentError("MiniCPM-OCR SDK returned no pages")
        return result

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"MiniCPMOCRProvider only supports PARSE product type, got {request.product_type}"
            )

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ProviderPermanentError(f"Unsupported file type: {file_path.suffix}. Supported: {SUPPORTED_SUFFIXES}")

        started_at = datetime.now()
        try:
            result = asyncio.run(self._predict(file_path.read_bytes(), file_path.name))
        except TimeoutError as e:
            raise ProviderTransientError(f"Request timed out after {self._timeout} seconds") from e
        except aiohttp.ClientError as e:
            raise ProviderTransientError(f"Request failed: {e}") from e
        completed_at = datetime.now()

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "markdown": result.get("markdown", ""),
                "page_results": result["pages"],
                "counts": result.get("counts", {}),
                "_config": {"server_url": self._server_url},
            },
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    @staticmethod
    def _sanitize_html_attributes(text: str) -> str:
        """Quote unquoted HTML attributes so tables are valid XML."""

        def _quote_attrs(match: re.Match) -> str:
            return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', match.group(0))

        return re.sub(r"<[^>]+>", _quote_attrs, text)

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"MiniCPMOCRProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        page_markdowns: list[str] = []
        for page in raw_result.raw_output.get("page_results", []):
            page_index = int(page.get("page_index") or 0)
            markdown = page.get("markdown", "")
            if markdown:
                markdown = self._sanitize_html_attributes(_convert_pipe_tables_to_html(markdown))

            pages.append(PageIR(page_index=page_index, markdown=markdown))
            page_markdowns.append(markdown)
            layout_pages.extend(
                build_layout_pages(
                    items=_layout_items(page),
                    image_width=int(page.get("image_width") or 0),
                    image_height=int(page.get("image_height") or 0),
                    markdown=markdown,
                    page_number=page_index + 1,
                )
            )

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown="\n\n".join(page_markdowns),
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
