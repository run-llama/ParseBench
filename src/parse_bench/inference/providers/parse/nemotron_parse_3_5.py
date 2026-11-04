"""Provider for a Nemotron 3.5 Parse SDK server.

Nemotron 3.5 Parse (nvidia/Nemotron-3.5-Parse, 0.63B) emits a serialization of
classed, content-normalized boxes. The SDK server runs the model repository's
``nemotron_parse`` triple pass (full page + detection + Table/Chart crops)
against a vLLM backend and returns rendered Markdown with HTML tables plus
per-block classes and boxes, so this provider only renders pages and maps
blocks onto the shared layout IR.

Server contract: ``POST {"image_base64", "mode"}`` returns ``{"status",
"markdown", "image_width", "image_height", "blocks": [{"class", "bbox"
(normalized 0-1 xyxy), "text", "content"}]}``, i.e. the ``markdown`` and
``blocks`` of ``DocumentParser.parse_prepared(...).documents[0]``.
"""

import asyncio
import base64
import io
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
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# Nemotron classes that are not already keys of ``_layout_utils.LABEL_MAP``.
# Every other class (Title, Section-header, Text, List-item, Table, Picture,
# Caption, Footnote, Formula, Code, Page-header, Page-footer) maps as is.
_LABEL_ALIASES = {
    "chart": "picture",
    "toc": "document index",
    "bibliography": "text",
}

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp")


@register_provider("nemotron_parse_3_5")
class NemotronParse35Provider(Provider):
    """
    Provider for the Nemotron 3.5 Parse Modal SDK server.

    Configuration options:
        - server_url (str, required): SDK server /predict URL. Falls back to the
          NEMOTRON_PARSE_3_5_SERVER_URL environment variable.
        - mode (str, default="triple"): single | double | triple
        - timeout (int, default=900): Request timeout per page in seconds
        - dpi (int, default=150): PDF render DPI (the model card's ParseBench input contract)
        - jpeg_quality (int, default=92): JPEG quality of rendered PDF pages (same contract)
        - page_concurrency (int, default=4): Pages of one document parsed concurrently
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("NEMOTRON_PARSE_3_5_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "Nemotron 3.5 Parse provider requires 'server_url' in config or the "
                "NEMOTRON_PARSE_3_5_SERVER_URL environment variable (the SDK server /predict URL)."
            )
        self._server_url: str = str(server_url)
        self._mode: str = str(self.base_config.get("mode", "triple"))
        self._timeout = self.base_config.get("timeout", 900)
        self._dpi = self.base_config.get("dpi", 150)
        self._jpeg_quality = self.base_config.get("jpeg_quality", 92)
        self._page_concurrency = self.base_config.get("page_concurrency", 4)

    def _config_summary(self) -> dict[str, Any]:
        return {"server_url": self._server_url, "mode": self._mode, "dpi": self._dpi}

    def _pdf_to_images(self, pdf_path: Path) -> list[bytes]:
        """Render every PDF page to RGB JPEG bytes in source order."""
        import fitz  # PyMuPDF
        from PIL import Image

        try:
            encoded: list[bytes] = []
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=self._dpi, colorspace=fitz.csRGB, alpha=False)
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG", quality=self._jpeg_quality)
                    encoded.append(buf.getvalue())
        except Exception as e:
            raise ProviderPermanentError(f"Error converting PDF to image: {e}") from e
        if not encoded:
            raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
        return encoded

    async def _parse_page(self, session: aiohttp.ClientSession, image_bytes: bytes) -> dict[str, Any]:
        payload = {"image_base64": base64.b64encode(image_bytes).decode(), "mode": self._mode}
        async with session.post(
            self._server_url.rstrip("/"),
            json=payload,
            headers={"Content-Type": "application/json"},
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
        if not result.get("markdown") and not result.get("blocks"):
            raise ProviderPermanentError("Empty response from API")
        return result

    async def _run_inference_pages_async(self, pages: list[bytes]) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(self._page_concurrency)
        async with aiohttp.ClientSession() as session:

            async def one(page: bytes) -> dict[str, Any]:
                async with semaphore:
                    return await self._parse_page(session, page)

            results = await asyncio.gather(*(one(page) for page in pages))
        return {"page_results": list(results), "_config": self._config_summary()}

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"NemotronParse35Provider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            page_images = self._pdf_to_images(file_path)
        elif suffix in _IMAGE_SUFFIXES:
            page_images = [file_path.read_bytes()]
        else:
            raise ProviderPermanentError(
                f"Unsupported file type: {suffix}. Supported: .pdf, {', '.join(_IMAGE_SUFFIXES)}"
            )

        try:
            raw_output = asyncio.run(self._run_inference_pages_async(page_images))
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
    def _layout_items(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Blocks → ``build_layout_pages`` items with 0-1000 xyxy boxes.

        Boxes are clamped so a slightly out-of-frame prediction cannot push a
        coordinate past 1000, which ``build_layout_pages`` would read as pixels.
        """
        items: list[dict[str, Any]] = []
        for block in blocks:
            bbox = block.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            label = str(block.get("class", "Text"))
            items.append(
                {
                    "label": _LABEL_ALIASES.get(label.lower(), label),
                    "bbox": [min(1000.0, max(0.0, float(v) * 1000.0)) for v in bbox],
                    "text": str(block.get("content") or block.get("text") or "").strip(),
                }
            )
        return items

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"NemotronParse35Provider only supports PARSE product type, got {raw_result.product_type}"
            )

        page_markdowns: list[str] = []
        layout_pages: list[ParseLayoutPageIR] = []
        for page_number, page in enumerate(raw_result.raw_output.get("page_results") or [], start=1):
            markdown = page.get("markdown", "")
            if markdown:
                page_markdowns.append(markdown)
            layout_pages.extend(
                build_layout_pages(
                    self._layout_items(page.get("blocks") or []),
                    int(page.get("image_width") or 0),
                    int(page.get("image_height") or 0),
                    markdown,
                    page_number,
                )
            )

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            markdown="\n\n".join(page_markdowns),
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
