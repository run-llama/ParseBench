"""Provider for PaddleOCR Modal servers."""

import asyncio
import base64
import io
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
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# PP-DocLayout label -> shared ``LABEL_MAP`` key (see ``_layout_utils``).
# Labels NOT in this map are dropped silently (treated as "abandon" / page
# furniture). Verified against actual PaddleOCRVL output: header, text,
# paragraph_title, table, vision_footnote, chart, number, footer.
_PADDLE_LABEL_ALIASES: dict[str, str] = {
    "doc_title": "title",
    "paragraph_title": "section_header",
    "header": "page_header",
    "footer": "page_footer",
    "text": "text",
    "content": "text",
    "abstract": "text",
    "aside_text": "text",
    "number": "text",
    "page_number": "text",
    "formula_number": "text",
    "reference": "text",
    "reference_content": "text",
    "footnote": "footnote",
    "vision_footnote": "footnote",
    "image": "picture",
    "figure": "picture",
    "chart": "picture",
    "seal": "picture",
    "header_image": "picture",
    "footer_image": "picture",
    "figure_title": "caption",
    "figure_caption": "caption",
    "table_title": "caption",
    "table_caption": "caption",
    "chart_title": "caption",
    "chart_caption": "caption",
    "figure_table_title": "caption",
    "list": "list_item",
    "list_item": "list_item",
    "table": "table",
    "formula": "formula",
    "algorithm": "text",
    # NB: "abandon" / page furniture is intentionally NOT mapped — dropped.
}

# Model name expected by vLLM server
SERVED_MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"

# Task-specific prompts for OpenAI API format
TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


@register_provider("paddleocr")
class PaddleOCRProvider(Provider):
    """
    Provider for PaddleOCR Modal servers.

    This provider wraps PaddleOCR-VL models deployed on Modal, supporting both:
    - OpenAI-compatible vLLM API (/v1/chat/completions)
    - Simple pipeline API (/predict with image_base64)

    Configuration options:
        - server_url (str, required): Modal server URL
        - api_format (str, default="openai"): API format - "openai" or "simple"
        - task (str, default="table"): Task prompt for OpenAI API
            Options: "ocr", "table", "formula", "chart"
        - timeout (int, default=600): Request timeout in seconds
        - dpi (int, default=150): DPI for PDF to image conversion
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the PaddleOCR provider.

        :param provider_name: Name of the provider
        :param base_config: Configuration dictionary
        """
        super().__init__(provider_name, base_config)

        # Validate required config
        self._server_url = self.base_config.get("server_url") or os.getenv("PADDLEOCR_SERVER_URL")
        if not self._server_url:
            raise ProviderConfigError(
                "PaddleOCR provider requires 'server_url' in config. "
                "Example: https://llamaindex--paddle-vllm-09b-serve.modal.run"
            )

        # Get configuration with defaults
        self._api_format = self.base_config.get("api_format", "openai")
        if self._api_format not in ("openai", "simple"):
            raise ProviderConfigError(f"Invalid api_format '{self._api_format}'. Must be 'openai' or 'simple'.")

        self._task = self.base_config.get("task", "table")
        if self._task not in TASK_PROMPTS:
            raise ProviderConfigError(f"Invalid task '{self._task}'. Must be one of: {list(TASK_PROMPTS.keys())}")

        self._timeout = self.base_config.get("timeout", 600)
        self._dpi = self.base_config.get("dpi", 150)

        # Model name sent to the vLLM server. Defaults to the 1.5 model; override
        # via the ``served_model_name`` key for other releases (e.g. PaddleOCR-VL-1.6-0.9B).
        self._served_model_name = self.base_config.get("served_model_name", SERVED_MODEL_NAME)

    def _pdf_to_images(self, pdf_path: Path) -> list[bytes]:
        """
        Render every PDF page to PNG bytes in source order.

        :param pdf_path: Path to the PDF file
        :return: PNG image bytes for every source page
        :raises ProviderPermanentError: If conversion fails
        """
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=self._dpi)
            if not images:
                raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")

            encoded: list[bytes] = []
            for image in images:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                encoded.append(buf.getvalue())
            return encoded

        except ImportError as e:
            raise ProviderPermanentError("pdf2image is required. Install with: pip install pdf2image") from e
        except Exception as e:
            if "pdf2image" in str(e).lower():
                raise
            raise ProviderPermanentError(f"Error converting PDF to image: {e}") from e

    def _read_image(self, file_path: Path) -> bytes:
        """
        Read an image file.

        :param file_path: Path to the image file
        :return: Image bytes
        :raises ProviderPermanentError: If reading fails
        """
        try:
            return file_path.read_bytes()
        except Exception as e:
            raise ProviderPermanentError(f"Error reading image file: {e}") from e

    async def _call_openai_api(
        self,
        session: aiohttp.ClientSession,
        image_b64: str,
    ) -> str:
        """
        Call the OpenAI-compatible vLLM API.

        :param session: aiohttp session
        :param image_b64: Base64-encoded image
        :return: Markdown content from the API response
        """
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"  # type: ignore[union-attr]

        payload = {
            "model": self._served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": TASK_PROMPTS.get(self._task, "OCR:")},
                    ],
                }
            ],
            "temperature": 0.0,
            "stream": False,
        }

        async with session.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                # 408 = Modal cold start timeout, 502/503/504 = server errors
                if resp.status in (408, 502, 503, 504):
                    raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

            result = await resp.json()

            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise ProviderPermanentError(f"Invalid response format: {e}") from e

            if not content:
                raise ProviderPermanentError("Empty content response from API")

            return content  # type: ignore[no-any-return]

    async def _call_simple_api(
        self,
        session: aiohttp.ClientSession,
        image_b64: str,
    ) -> dict[str, Any]:
        """Call the simple pipeline API; return ``{markdown, layout_pages?}``."""
        api_url = self._server_url.rstrip("/")  # type: ignore[union-attr]

        payload = {"image_base64": image_b64}

        async with session.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                # 408 = Modal cold start timeout, 502/503/504 = server errors
                if resp.status in (408, 502, 503, 504):
                    raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

            result = await resp.json()

            if result.get("status") == "error":
                raise ProviderPermanentError(result.get("error", "Unknown error from API"))

            content = result.get("markdown", "")
            if not content:
                raise ProviderPermanentError("Empty markdown response from API")

            return {
                "markdown": content,
                "layout_pages": result.get("layout_pages") or [],
            }

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Run async inference on an image.

        :param image_bytes: Image bytes
        :return: Raw response dictionary with markdown (and layout_pages when
            the simple pipeline API is in use).
        """
        image_b64 = base64.b64encode(image_bytes).decode()

        async with aiohttp.ClientSession() as session:
            if self._api_format == "simple":
                resp = await self._call_simple_api(session, image_b64)
                markdown = resp["markdown"]
                layout_pages = resp.get("layout_pages") or []
            else:
                markdown = await self._call_openai_api(session, image_b64)
                layout_pages = []

        return {
            "markdown": markdown,
            "layout_pages": layout_pages,
            "_config": {
                "server_url": self._server_url,
                "api_format": self._api_format,
                "task": self._task,
                "dpi": self._dpi,
            },
        }

    async def _run_inference_pages_async(self, pages: list[bytes]) -> dict[str, Any]:
        """Run every input page in source order, preserving one-page raw output."""
        results = [await self._run_inference_async(page) for page in pages]
        first = results[0]
        if len(results) == 1:
            return first
        merged = dict(first)
        merged["page_results"] = results
        return merged

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
                f"PaddleOCRProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        # Render PDFs into one input image per source page.
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            page_images = self._pdf_to_images(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            page_images = [self._read_image(file_path)]
        else:
            raise ProviderPermanentError(
                f"Unsupported file type: {suffix}. Supported types: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
            )

        try:
            raw_output = asyncio.run(self._run_inference_pages_async(page_images))

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

        except (TimeoutError, ProviderPermanentError, ProviderTransientError, Exception) as e:
            # Return empty result with error info instead of failing
            # This allows workflow to continue while tracking the error
            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            error_msg = str(e)
            if isinstance(e, asyncio.TimeoutError):
                error_msg = f"Request timed out after {self._timeout} seconds"

            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output={
                    "markdown": "",
                    "_error": error_msg,
                    "_error_type": type(e).__name__,
                    "_config": {
                        "server_url": self._server_url,
                        "api_format": self._api_format,
                        "task": self._task,
                        "dpi": self._dpi,
                    },
                },
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

    @staticmethod
    def _sanitize_html_attributes(markdown: str) -> str:
        """Quote unquoted HTML attributes so tables are valid XML.

        PaddleOCR's save_to_markdown() emits attributes like ``border=1``
        without quotes, which is valid HTML5 but not valid XML.  The GriTS
        metric parses tables with ``xml.etree.ElementTree`` (strict XML), so
        unquoted attributes cause parse failures and 0.0 scores.

        This method finds bare attribute values (``name=value`` where value is
        not already quoted) inside HTML tags and wraps them in double quotes.
        """

        def _quote_attrs(match: re.Match) -> str:
            tag_text = match.group(0)
            # Quote unquoted attribute values: attr=value -> attr="value"
            tag_text = re.sub(
                r'(\w+)=([^\s"\'<>=]+)',
                r'\1="\2"',
                tag_text,
            )
            return tag_text

        return re.sub(r"<[^>]+>", _quote_attrs, markdown)

    @staticmethod
    def _otsl_to_html(text: str) -> str:
        """Convert PaddleOCR-VL-1.5 OTSL output to HTML <table>.

        PaddleOCR-VL-1.5 with ``Table Recognition:`` prompt emits OTSL tokens:

        - ``<fcel>cell``  full cell with content
        - ``<ecel>``      empty cell
        - ``<lcel>``      left-merge extension (colspan continuation)
        - ``<ucel>``      up-merge extension (rowspan continuation)
        - ``<xcel>``      diagonal-merge (both row and col extension)
        - ``<ched>cell``  column header cell
        - ``<rhed>cell``  row header cell
        - ``<srow>cell``  section-row cell
        - ``<nl>``        end of row

        Tokens may be wrapped in ``<otsl>...</otsl>`` or appear bare. Any text
        before/after a contiguous OTSL block is preserved verbatim. The whole
        OTSL run is rendered as a single HTML ``<table>``.
        """
        if "<fcel>" not in text and "<ecel>" not in text and "<ched>" not in text:
            return text

        text = re.sub(r"</?otsl[^>]*>", "", text, flags=re.IGNORECASE)

        token_re = re.compile(
            r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>|<ched>|<rhed>|<srow>|<nl>)",
            re.IGNORECASE,
        )
        parts = token_re.split(text)

        out: list[str] = []
        i = 0
        n = len(parts)
        while i < n:
            part = parts[i]
            if not token_re.match(part):
                if part:
                    out.append(part)
                i += 1
                continue

            rows: list[list[tuple[str, str]]] = [[]]
            while i < n:
                tok = parts[i]
                m = token_re.match(tok)
                if not m:
                    break
                kind = tok.lower().strip("<>")
                i += 1
                content = parts[i] if i < n and not token_re.match(parts[i]) else ""
                if content:
                    i += 1
                content = content.strip()
                if kind == "nl":
                    if rows[-1]:
                        rows.append([])
                    continue
                rows[-1].append((kind, content))
            if rows and not rows[-1]:
                rows.pop()

            html: list[str] = ['<table border="1">']
            for r, row in enumerate(rows):
                html.append("<tr>")
                c = 0
                while c < len(row):
                    kind, content = row[c]
                    if kind in ("lcel", "ucel", "xcel"):
                        c += 1
                        continue
                    colspan = 1
                    j = c + 1
                    while j < len(row) and row[j][0] == "lcel":
                        colspan += 1
                        j += 1
                    rowspan = 1
                    rr = r + 1
                    while rr < len(rows) and c < len(rows[rr]) and rows[rr][c][0] in ("ucel", "xcel"):
                        rowspan += 1
                        rr += 1
                    tag = "th" if kind in ("ched", "rhed") else "td"
                    attrs = ""
                    if colspan > 1:
                        attrs += f' colspan="{colspan}"'
                    if rowspan > 1:
                        attrs += f' rowspan="{rowspan}"'
                    html.append(f"<{tag}{attrs}>{content}</{tag}>")
                    c = j
                html.append("</tr>")
            html.append("</table>")
            out.append("".join(html))

        return "".join(out)

    @staticmethod
    def _build_layout_pages(
        raw_layout_pages: list[dict[str, Any]],
        *,
        page_number: int | None = None,
    ) -> list[ParseLayoutPageIR]:
        """Build ``ParseOutput.layout_pages`` from the pipeline server response.

        The server emits per-page ``items`` of ``{bbox: [x1,y1,x2,y2], label,
        text, score}`` in pixel coords matching the input image. Paddle labels
        are aliased onto the shared ``LABEL_MAP`` vocabulary; labels with no
        alias (e.g. ``abandon``, page furniture the evaluator wouldn't score)
        are dropped. When an input page number is supplied, it takes precedence
        over the service's page number because some single-image responses
        always report page 1.
        """
        layout_pages: list[ParseLayoutPageIR] = []
        for page in raw_layout_pages:
            if not isinstance(page, dict):
                continue
            img_w = int(page.get("width") or 0)
            img_h = int(page.get("height") or 0)
            if img_w <= 0 or img_h <= 0:
                continue

            items: list[dict[str, Any]] = []
            for item in page.get("items") or []:
                if not isinstance(item, dict):
                    continue
                bbox = item.get("bbox") or []
                if len(bbox) < 4:
                    continue
                label = str(item.get("label") or "").strip().lower()
                alias = _PADDLE_LABEL_ALIASES.get(label)
                if alias is None:
                    continue  # drop labels we can't map (e.g. abandon)
                items.append({"label": alias, "bbox": list(bbox[:4]), "text": str(item.get("text") or "")})

            if items:
                layout_pages.extend(
                    build_layout_pages(
                        items=items,
                        image_width=img_w,
                        image_height=img_h,
                        markdown="",
                        page_number=page_number
                        if page_number is not None
                        else int(page.get("page_number") or len(layout_pages) + 1),
                        bbox_scale=None,
                    )
                )
        return layout_pages

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"PaddleOCRProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        page_results = raw_result.raw_output.get("page_results")
        if not isinstance(page_results, list) or not page_results:
            page_results = [raw_result.raw_output]

        layout_pages: list[ParseLayoutPageIR] = []
        page_markdowns: list[str] = []
        for page_number, page_raw in enumerate(page_results, start=1):
            markdown = page_raw.get("markdown", "")
            if markdown:
                # Apply model-output repairs independently to each source page.
                # PaddleOCR-VL-1.5 "Table Recognition:" returns OTSL tokens; convert
                # to HTML so GriTS/TEDS can score it. No-op when OTSL tokens absent.
                markdown = self._otsl_to_html(markdown)
                # Quote bare HTML attributes for XML-based metric parsers (e.g. GriTS).
                markdown = self._sanitize_html_attributes(markdown)
                page_markdowns.append(markdown)

            # Bind layout results to the rendered input-page index, rather than
            # trusting the service response's often-constant page number.
            layout_pages.extend(
                self._build_layout_pages(
                    page_raw.get("layout_pages", []),
                    page_number=page_number,
                )
            )

        markdown = "\n\n".join(page_markdowns)

        # Create ParseOutput with document-level markdown
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            layout_pages=layout_pages,
            markdown=markdown,
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
