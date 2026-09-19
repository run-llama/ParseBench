"""Provider for a self-hosted TeleOCR server.

TeleOCR (StarDoc-AI/TeleOCR) performs layout detection followed by per-block
recognition. The server accepts one page image per request and returns markdown,
recognized blocks, and the source image dimensions.

Each block contains a type, a normalized ``[x1, y1, x2, y2]`` bounding box,
and recognized content. Table and figure content may already be HTML, while
equations may already be wrapped in display-math delimiters.
"""

import base64
import io
import json
import math
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
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._layout_utils import (
    build_layout_pages,
    items_to_markdown,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# TeleOCR block types mapped onto the shared layout utility labels. Types that
# already match the shared labels (text, title, table, code, footnote, formula)
# need no alias.
_LABEL_ALIASES: dict[str, str] = {
    "image": "picture",
    "char": "picture",
    "seal": "picture",
    "algorithm": "code",
    "header": "page-header",
    "footer": "page-footer",
    "page_footnote": "footnote",
    "table_footnote": "footnote",
    "image_footnote": "footnote",
    "table_caption": "caption",
    "image_caption": "caption",
    "code_caption": "caption",
    "equation": "formula",
    "equation_block": "formula",
    "list": "list-item",
    "aside_text": "text",
    "ref_text": "text",
    "phonetic": "text",
    "unknown": "text",
}

# TeleOCR uses one page-number class. Split it into header/footer using the
# vertical center of its normalized bounding box.
_PAGE_NUMBER_SPLIT = 0.5


@register_provider("teleocr")
class TeleOCRProvider(Provider):
    """Provider for a self-hosted TeleOCR inference endpoint.

    Config:
        - server_url (str, required): POST endpoint. May also be supplied via
          the ``TELEOCR_SERVER_URL`` environment variable.
        - timeout (int, default=600): request timeout in seconds.
        - dpi (int, default=300): PDF-to-image render resolution.
        - prompt_appendix (str, optional): extra recognition instruction sent
          to the server as ``custom_prompt``.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("TELEOCR_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "TeleOCR provider requires 'server_url' in config or TELEOCR_SERVER_URL in the environment."
            )
        self._server_url = str(server_url)
        self._timeout = self.base_config.get("timeout", 600)
        self._dpi = self.base_config.get("dpi", 300)
        self._prompt_appendix = str(self.base_config.get("prompt_appendix") or "").strip()

    def _pdf_to_images(self, pdf_path: Path) -> list[bytes]:
        """Render every PDF page to PNG bytes in source order."""
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=self._dpi)
            if not images:
                raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
            encoded: list[bytes] = []
            for image in images:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded.append(buffer.getvalue())
            return encoded
        except ImportError as exc:
            raise ProviderPermanentError("pdf2image is required.") from exc
        except ProviderPermanentError:
            raise
        except Exception as exc:
            raise ProviderPermanentError(f"Error converting PDF to image: {exc}") from exc

    def _read_image(self, file_path: Path) -> bytes:
        try:
            return file_path.read_bytes()
        except Exception as exc:
            raise ProviderPermanentError(f"Error reading image file: {exc}") from exc

    async def _call_api(self, session: aiohttp.ClientSession, image_b64: str) -> dict[str, Any]:
        payload = {"image_base64": image_b64}
        if self._prompt_appendix:
            payload["custom_prompt"] = self._prompt_appendix

        try:
            async with session.post(
                self._server_url.rstrip("/"),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    if response.status == 429:
                        raise ProviderRateLimitError(f"HTTP 429: {error_text[:200]}")
                    if response.status == 408 or 500 <= response.status < 600:
                        raise ProviderTransientError(f"HTTP {response.status}: {error_text[:200]}")
                    raise ProviderPermanentError(f"HTTP {response.status}: {error_text[:200]}")

                try:
                    result = await response.json()
                except ValueError as exc:
                    raise ProviderTransientError(f"TeleOCR returned invalid JSON: {exc}") from exc
                if not isinstance(result, dict):
                    raise ProviderTransientError("TeleOCR returned a non-object JSON response")
                if result.get("status") == "error":
                    raise ProviderPermanentError(result.get("error", "Unknown error from API"))
                markdown = result.get("markdown")
                if not (isinstance(markdown, str) and markdown.strip()) and not self._has_valid_blocks(
                    result.get("blocks")
                ):
                    raise ProviderPermanentError("TeleOCR response contains neither markdown nor valid blocks")
                return result
        except (ProviderPermanentError, ProviderRateLimitError, ProviderTransientError):
            raise
        except TimeoutError as exc:
            raise ProviderTransientError(f"TeleOCR request timed out after {self._timeout} seconds") from exc
        except aiohttp.ClientError as exc:
            raise ProviderTransientError(f"TeleOCR request failed: {exc}") from exc

    async def _run_inference_pages_async(self, pages: list[bytes]) -> dict[str, Any]:
        """Run inference per page in order and preserve the one-page shape."""
        results: list[dict[str, Any]] = []
        async with aiohttp.ClientSession() as session:
            for image_bytes in pages:
                result = await self._call_api(session, base64.b64encode(image_bytes).decode())
                results.append(
                    {
                        "markdown": result.get("markdown", ""),
                        "blocks": result.get("blocks", []),
                        "image_width": result.get("image_width"),
                        "image_height": result.get("image_height"),
                        "timing": result.get("timing"),
                        "_config": self._config_snapshot(),
                    }
                )

        first = results[0]
        if len(results) == 1:
            return first
        merged = dict(first)
        merged["page_results"] = results
        return merged

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "server_url": self._server_url,
            "dpi": self._dpi,
            "prompt_appendix": self._prompt_appendix,
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"TeleOCRProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            page_images = self._pdf_to_images(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            page_images = [self._read_image(file_path)]
        else:
            raise ProviderPermanentError(
                f"Unsupported file type: {suffix}. Supported: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
            )

        try:
            raw_output = self.run_async_from_sync(self._run_inference_pages_async(page_images))
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
        except (ProviderPermanentError, ProviderRateLimitError, ProviderTransientError):
            raise
        except TimeoutError as exc:
            raise ProviderTransientError(f"TeleOCR request timed out after {self._timeout} seconds") from exc
        except aiohttp.ClientError as exc:
            raise ProviderTransientError(f"TeleOCR request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderTransientError(f"TeleOCR returned invalid JSON: {exc}") from exc

    @staticmethod
    def _close_unclosed_table_tags(content: str) -> str:
        opens = content.count("<table")
        closes = content.count("</table>")
        if opens > closes:
            if not content.rstrip().endswith(">"):
                content += "</td></tr>"
            content += "</table>" * (opens - closes)
        return content

    @staticmethod
    def _promote_first_row_to_thead(content: str) -> str:
        """Promote the first all-data-cell table row to a header row."""

        def _promote(match: re.Match[str]) -> str:
            table_html = match.group(0)
            if "<thead" in table_html:
                return table_html
            first_row = re.search(r"<tr>(.*?)</tr>", table_html, re.DOTALL)
            if not first_row:
                return table_html
            header = first_row.group(1).replace("<td>", "<th>").replace("</td>", "</th>")
            header = re.sub(r"<td(\s)", r"<th\1", header)
            return table_html.replace(first_row.group(0), f"<thead><tr>{header}</tr></thead>", 1)

        return re.sub(r"<table[^>]*>.*?</table>", _promote, content, flags=re.DOTALL)

    @staticmethod
    def _sanitize_html_attributes(markdown: str) -> str:
        def _quote_attrs(match: re.Match[str]) -> str:
            return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', match.group(0))

        return re.sub(r"<[^>]+>", _quote_attrs, markdown)

    @classmethod
    def _clean_content(cls, content: str) -> str:
        if not content:
            return ""
        content = cls._close_unclosed_table_tags(content)
        content = cls._promote_first_row_to_thead(content)
        return cls._sanitize_html_attributes(content)

    _FENCED_CODE_BLOCK_RE = re.compile(
        r"(^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\2[ \t]*(?=\n|$))",
        re.MULTILINE | re.DOTALL,
    )
    _HTML_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)

    @classmethod
    def _clean_server_markdown(cls, markdown: str) -> str:
        """Clean HTML tables without altering other server Markdown.

        Fenced code is copied byte-for-byte so examples containing HTML are not
        mistaken for document tables. Outside fences, only complete ``table``
        fragments are normalized; headings, code, images, and other Markdown
        remain server-authored.
        """

        def _clean_segment(segment: str) -> str:
            segment = cls._close_unclosed_table_tags(segment)
            return cls._HTML_TABLE_RE.sub(lambda match: cls._clean_content(match.group(0)), segment)

        parts: list[str] = []
        cursor = 0
        for match in cls._FENCED_CODE_BLOCK_RE.finditer(markdown):
            parts.append(_clean_segment(markdown[cursor : match.start()]))
            parts.append(match.group(0))
            cursor = match.end()
        parts.append(_clean_segment(markdown[cursor:]))
        return "".join(parts)

    @staticmethod
    def _is_valid_bbox(bbox: Any) -> bool:
        return (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in bbox
            )
        )

    @classmethod
    def _has_valid_blocks(cls, blocks: Any) -> bool:
        return isinstance(blocks, list) and any(
            isinstance(block, dict) and cls._is_valid_bbox(block.get("bbox")) for block in blocks
        )

    @staticmethod
    def _blocks_to_items(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert TeleOCR blocks to the shared layout-item shape."""
        items: list[dict[str, Any]] = []
        for block in blocks:
            bbox = block.get("bbox") or []
            if not TeleOCRProvider._is_valid_bbox(bbox):
                continue
            raw_label = str(block.get("type") or "text").lower()
            if raw_label == "page_number":
                y_center = (float(bbox[1]) + float(bbox[3])) / 2.0
                label = "page-header" if y_center < _PAGE_NUMBER_SPLIT else "page-footer"
            else:
                label = _LABEL_ALIASES.get(raw_label, raw_label)
            items.append(
                {
                    "bbox": [float(value) * 1000.0 for value in bbox],
                    "label": label,
                    "text": TeleOCRProvider._clean_content(str(block.get("content") or "")),
                }
            )
        return items

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"TeleOCRProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        page_results = raw_result.raw_output.get("page_results")
        if not isinstance(page_results, list) or not page_results:
            page_results = [raw_result.raw_output]

        page_markdown: list[str] = []
        layout_pages: list[ParseLayoutPageIR] = []
        for page_number, page_raw in enumerate(page_results, start=1):
            items = self._blocks_to_items(page_raw.get("blocks") or [])
            server_markdown = page_raw.get("markdown")
            if isinstance(server_markdown, str) and server_markdown.strip():
                markdown = self._clean_server_markdown(server_markdown)
            else:
                markdown = items_to_markdown(items)
            layout_pages.extend(
                build_layout_pages(
                    items,
                    page_raw.get("image_width") or 0,
                    page_raw.get("image_height") or 0,
                    markdown,
                    page_number=page_number,
                )
            )
            if markdown:
                page_markdown.append(markdown)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            markdown="\n\n".join(page_markdown),
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
