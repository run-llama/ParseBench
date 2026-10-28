"""Provider for an Unlimited-OCR 2 server.

Unlimited-OCR 2 (baidu/Unlimited-OCR-2) parses a whole document in one request
and emits layout blocks, not markdown:
``<PAGE><BLOCK>label[<LEVELn>][x1,y1,x2,y2]<FORK>content``, one ``<PAGE>`` per
input page, boxes on a 0-999 grid of that page. ``normalize`` turns that string
into markdown with Baidu's official converter and into per-page layout boxes
with the converter's own block parser.

Server contract: the server wraps the model's official SGLang deployment and
sends the README request (page images in order, then
``"<PAGE><image>" * n_pages + "Multi page parsing."``).
``POST {"images_base64": [page1, ...]}`` returns ``{"status", "raw",
"page_sizes": [{"width", "height"}, ...]}`` with ``raw`` the untouched model
output.
"""

import asyncio
import base64
import os
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
from parse_bench.inference.providers.parse._unlimitedocr2_convert import (
    IMAGE_LABELS,
    NOISE_CONTENT,
    PAGE_SPLIT_RE,
    STRAY_TOKEN_RE,
    _base_type,
    _parse_blocks,
    convert,
)
from parse_bench.inference.providers.parse.unlimitedocr import UnlimitedOCRProvider
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# Unlimited-OCR 2 block labels -> keys of the shared _layout_utils.LABEL_MAP.
# text/table/figure already match; title is split by its <LEVELn> and
# page_number by its position below.
_LABEL_ALIASES: dict[str, str] = {
    "header": "page-header",
    "footer": "page-footer",
    "image": "picture",
    "chart": "picture",
    "seal": "picture",
    "image_caption": "caption",
    "image_footnote": "footnote",
    "page_footnote": "footnote",
    "equation": "formula",
    "algorithm": "code",
    "ref_text": "text",
    "aside_text": "text",
}


def _strip_fences(raw: str) -> str:
    """The same ```markdown fence strip ``convert`` applies before parsing blocks."""
    text = (raw or "").strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown") :].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _page_segments(text: str) -> list[str]:
    """Split on <PAGE> exactly as ``_parse_blocks`` numbers pages (page 1 = index 0)."""
    segments = PAGE_SPLIT_RE.split(text)
    if segments and not segments[0].strip():
        segments = segments[1:]
    return segments


def _layout_label(block: dict[str, Any]) -> str:
    base: str = _base_type(block["label"])
    if base == "title":
        return "section-header" if (block.get("level") or 1) >= 2 else "title"
    if base == "page_number":
        # A printed page number is page furniture on whichever edge it sits.
        _x1, y1, _x2, y2 = block["bbox"]
        return "page-header" if (y1 + y2) / 2 < 500 else "page-footer"
    return _LABEL_ALIASES.get(base, base)


@register_provider("unlimitedocr2")
class UnlimitedOCR2Provider(Provider):
    """
    Provider for an Unlimited-OCR 2 server.

    Configuration options:
        - server_url (str, required): Server predict endpoint URL. Falls back to
          the UNLIMITEDOCR2_SERVER_URL environment variable.
        - timeout (int, default=6000): Request timeout in seconds (official README)
        - dpi (int, default=300): DPI for PDF to image conversion (official pdf_to_images)
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("UNLIMITEDOCR2_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "UnlimitedOCR2 provider requires 'server_url' in config or the "
                "UNLIMITEDOCR2_SERVER_URL environment variable."
            )
        self._server_url: str = server_url
        self._timeout = self.base_config.get("timeout", 6000)
        self._dpi = self.base_config.get("dpi", 300)

    def _pdf_to_images(self, pdf_path: Path) -> list[bytes]:
        """Render every page to PNG the way the official ``pdf_to_images`` does (PyMuPDF)."""
        import fitz

        try:
            doc = fitz.open(pdf_path)
            mat = fitz.Matrix(self._dpi / 72, self._dpi / 72)
            pages = [page.get_pixmap(matrix=mat).tobytes("png") for page in doc]
            doc.close()
        except Exception as e:
            raise ProviderPermanentError(f"Error converting PDF to image: {e}") from e
        if not pages:
            raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
        return pages

    async def _call_api(self, pages: list[bytes]) -> dict[str, Any]:
        payload = {"images_base64": [base64.b64encode(p).decode() for p in pages]}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._server_url.rstrip("/"),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    if resp.status in (408, 429, 500, 502, 503, 504):
                        raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                    raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")
                result: dict[str, Any] = await resp.json()
        if result.get("status") == "error":
            raise ProviderPermanentError(result.get("error", "Unknown error from API"))
        if not result.get("raw"):
            raise ProviderPermanentError("Empty output from API")
        return result

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"UnlimitedOCR2Provider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pages = self._pdf_to_images(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            pages = [file_path.read_bytes()]
        else:
            raise ProviderPermanentError(
                f"Unsupported file type: {suffix}. Supported: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
            )

        config = {"server_url": self._server_url, "dpi": self._dpi}
        try:
            # The official client sends the whole document as one request.
            result = asyncio.run(self._call_api(pages))
            raw_output = {**result, "_config": config}
        except (ProviderPermanentError, ProviderTransientError):
            raise
        except TimeoutError as e:
            raise ProviderTransientError(f"Request timed out after {self._timeout} seconds") from e
        except aiohttp.ClientError as e:
            raise ProviderTransientError(f"{type(e).__name__}: {e}") from e

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

    @staticmethod
    def _repair_html(markdown: str) -> str:
        markdown = UnlimitedOCRProvider._close_unclosed_table_tags(markdown)
        return UnlimitedOCRProvider._sanitize_html_attributes(markdown)

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"UnlimitedOCR2Provider only supports PARSE product type, got {raw_result.product_type}"
            )

        raw = raw_result.raw_output.get("raw", "")
        page_sizes: list[dict[str, Any]] = raw_result.raw_output.get("page_sizes") or []
        text = _strip_fences(raw)
        markdown = self._repair_html(convert(raw)) if raw else ""

        blocks_by_page: dict[int, list[dict[str, Any]]] = {}
        for block in _parse_blocks(text, page_base=1):
            if block["label"] == "__raw__" or block["bbox"] is None:
                continue
            content = STRAY_TOKEN_RE.sub("", block["content"]).strip()
            if content.lower() in NOISE_CONTENT:
                content = ""
            label = _layout_label(block)
            # Empty non-figure blocks are containers (e.g. an `other` box around a
            # list's items); ``convert`` renders nothing for them either.
            if not content and _base_type(block["label"]) not in IMAGE_LABELS:
                continue
            blocks_by_page.setdefault(block["page"], []).append(
                {"label": label, "bbox": list(block["bbox"]), "text": content}
            )

        segments = _page_segments(text)
        layout_pages = []
        for page_number, size in enumerate(page_sizes, start=1):
            segment = segments[page_number - 1] if page_number <= len(segments) else ""
            layout_pages.extend(
                build_layout_pages(
                    items=blocks_by_page.get(page_number, []),
                    image_width=size.get("width", 0),
                    image_height=size.get("height", 0),
                    markdown=self._repair_html(convert(segment)) if segment.strip() else "",
                    page_number=page_number,
                )
            )

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            markdown=markdown,
            layout_pages=layout_pages,
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
