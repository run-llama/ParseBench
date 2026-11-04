"""Provider for OvisOCR3 served behind an OpenAI-compatible vLLM endpoint.

OvisOCR3 (ATH-MaaS/OvisOCR3) is a unified 0.8B OCR VLM built on Qwen3.5-0.8B.
Its tasks are selected by the prompt files shipped in the model repo
(``prompts/*.txt``); this provider sends two of them verbatim:

* ``document_structured`` (one call per page) returns the page as a JSON array of
  ``{"bbox", "category", "content"}`` regions in reading order, with boxes on a
  [0, 1000) grid, tables in HTML, formulas in LaTeX and other text in Markdown.
  One call therefore feeds both the markdown and the layout output. The page
  markdown is the regions' ``content`` concatenated in reading order, as the
  model card does for cross-page reconstruction.
* ``chart_parsing`` (chart pass) is re-run on each ``chart`` region crop.
  Structured parsing writes charts as ``<img src="images/bbox_...">``
  placeholders, so charts would otherwise score zero; chart parsing is the
  model's own trained chart-reading task. Its JSON is rendered as HTML tables by
  the shared ``databricks_chart_json`` helper.

The server runs the model with the model card's vLLM settings
(``OVISOCR3_SERVER_URL``). Sampling follows the model card: greedy,
``max_tokens=16384``, ``enable_thinking=False``.
"""

import asyncio
import base64
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from parse_bench.evaluation.metrics.parse.chart_json_to_html import chart_description_to_html
from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._layout_utils import LABEL_MAP, build_layout_pages
from parse_bench.inference.providers.parse.mistral_ocr import _convert_pipe_tables_to_html
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# Model name registered in vLLM (--served-model-name)
SERVED_MODEL_NAME = "ovisocr3"

# Official prompts, verbatim from prompts/document_structured.txt and
# prompts/chart_parsing.txt in the model repo.
STRUCTURED_PROMPT = (
    "Analyze the document structure and extract text content. Return a single JSON array in natural "
    "reading order.\n\n"
    "Each element follows this schema:\n"
    '`{"bbox": [x_min, y_min, x_max, y_max], "category": "string", "content": "string"}`\n\n'
    "Rules:\n"
    "1. The bbox coordinates are integers scaled to [0, 1000).\n"
    "2. The category is exactly one of:\n"
    '["abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title", '
    '"figure_title", "footer", "footer_image", "footnote", "formula_number", "header", "header_image", '
    '"image", "inline_formula", "number", "paragraph_title", "reference_content", "seal", "table", '
    '"text", "vertical_text", "vision_footnote"].\n'
    "3. The content formatting depends on category:\n"
    '- For "footer_image", "header_image", "image", or "chart": write as an HTML image tag '
    '<img src="images/bbox_{x_min}_{y_min}_{x_max}_{y_max}.jpg" />.\n'
    '- For "display_formula" or "inline_formula": write in LaTeX.\n'
    '- For "table": write as a standard HTML `<table>...</table>`.\n'
    "- Otherwise: write in Markdown.\n"
    "4. Output must be strictly valid JSON. Properly escape quotes and special characters inside content."
)
CHART_PROMPT = (
    "Parse all key information in the chart and output one valid JSON object using exactly this "
    "top-level schema:\n"
    '{"title":"","source":"","x_title":"","y_title":"","values":{}}\n\n'
    "Rules:\n"
    "1. Preserve visible title, source, axis titles, series or category labels, x-axis labels, and "
    "values without explanation or inference.\n"
    '2. Use the literal string "None" when the chart has no visible value for title, source, or an '
    "axis title. For a dual-axis chart, `y_title` may be an array of two strings in visual order; "
    "otherwise it must be one string.\n"
    "3. `values` must be a non-empty object. Preserve the chart hierarchy and encode every scalar data "
    'value as a JSON string. An empty series name may be represented by the key "" when the chart has '
    "no visible series label.\n"
    "4. Output JSON only. Do not output Python syntax, Markdown fences, comments, prose, or extra keys."
)

# Official sampling parameters (SamplingParams(max_tokens=16384, temperature=0.0))
DEFAULT_MAX_TOKENS = 16384

# Region boxes are integers on this grid (model card, "Coordinates and Visual Regions").
COORD_SCALE = 1000.0

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")
_REGION_START_RE = re.compile(r'\{\s*"bbox"')
_IMG_PLACEHOLDER_RE = re.compile(r'^\s*<img src="images/bbox_[^>]*/>\s*$')
_HEADING_MARKER_RE = re.compile(r"^\s*#{1,6}\s")

# OvisOCR3 categories -> keys of the shared _layout_utils.LABEL_MAP. table, text and
# footnote already match; ``number`` (page number) is placed by position below.
_LABEL_ALIASES: dict[str, str] = {
    "doc_title": "title",
    "paragraph_title": "section_header",
    "header": "page_header",
    "footer": "page_footer",
    "figure_title": "caption",
    "vision_footnote": "footnote",
    "image": "picture",
    "chart": "picture",
    "seal": "picture",
    # ParseBench's layout GT labels an image in the header/footer band as Picture.
    "header_image": "picture",
    "footer_image": "picture",
    "display_formula": "formula",
    "inline_formula": "formula",
    "algorithm": "code",
    # PP-DocLayout's ``content`` is a table of contents.
    "content": "document index",
    "abstract": "text",
    "aside_text": "text",
    "formula_number": "text",
    "reference_content": "text",
    "vertical_text": "text",
}


def parse_regions(content: str) -> list[dict[str, Any]]:
    """Decode the structured answer into ``{bbox, category, content}`` regions.

    A response cut off by ``max_tokens`` is not valid JSON; every region completed
    before the cut is still recovered, in emitted order. A region whose own JSON is
    malformed is skipped without losing the rest of the page.
    """
    text = _FENCE_RE.sub("", content.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, list):
        decoder = json.JSONDecoder()
        data = []
        for match in _REGION_START_RE.finditer(text):
            try:
                region, _end = decoder.raw_decode(text, match.start())
            except json.JSONDecodeError:
                continue
            data.append(region)

    regions: list[dict[str, Any]] = []
    for region in data:
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        regions.append(
            {
                "bbox": [x1, y1, x2, y2],
                "category": str(region.get("category") or "text").strip().lower(),
                "content": str(region.get("content") or ""),
            }
        )
    return regions


def _layout_label(category: str, bbox: list[float]) -> str:
    """Canonical17 label for a region category."""
    if category == "number":
        # A printed page number is page furniture on whichever edge it sits.
        return LABEL_MAP["page_header"] if (bbox[1] + bbox[3]) / 2 < COORD_SCALE / 2 else LABEL_MAP["page_footer"]
    return LABEL_MAP.get(_LABEL_ALIASES.get(category, category), "Text")


def _region_markdown(region: dict[str, Any]) -> str:
    """Markdown for one region; image placeholders carry no page text and are dropped."""
    text = str(region.get("chart_html") or region["content"]).strip()
    if not text or _IMG_PLACEHOLDER_RE.match(text):
        return ""
    category = region["category"]
    if category == "display_formula" and not text.startswith("$"):
        return f"$$\n{text}\n$$"
    if category == "inline_formula" and not text.startswith("$"):
        return f"${text}$"
    if category == "doc_title" and not _HEADING_MARKER_RE.match(text):
        return f"# {text}"
    if category == "paragraph_title" and not _HEADING_MARKER_RE.match(text):
        return f"## {text}"
    return text


@register_provider("ovisocr3")
class OvisOcr3Provider(Provider):
    """Provider for an OvisOCR3 vLLM server.

    Configuration options:
        - server_url (str, required): Server URL. Falls back to the
          ``OVISOCR3_SERVER_URL`` environment variable.
        - served_model_name (str, default="ovisocr3"): Model name in vLLM
        - timeout (int, default=900): Request timeout in seconds
        - dpi (int, default=200): DPI for PDF page rendering
        - api_key_env (str, default="VLLM_API_KEY"): Env var for the API key
        - max_tokens (int, default=16384): Max output tokens
        - chart_pass (bool, default=True): Re-read each ``chart`` region with the
          official ``chart_parsing`` prompt
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("OVISOCR3_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "OvisOCR3 provider requires 'server_url' in config or OVISOCR3_SERVER_URL in the environment."
            )
        self._server_url: str = str(server_url)

        self._served_model_name = self.base_config.get("served_model_name", SERVED_MODEL_NAME)
        self._timeout = self.base_config.get("timeout", 900)
        self._dpi = self.base_config.get("dpi", 200)
        self._max_tokens = self.base_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self._chart_pass = bool(self.base_config.get("chart_pass", True))

        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _pdf_to_images(self, pdf_path: Path) -> list[bytes]:
        """Render every PDF page to PNG bytes in source order."""
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
        try:
            return file_path.read_bytes()
        except Exception as e:
            raise ProviderPermanentError(f"Error reading image file: {e}") from e

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    async def _call_api(self, session: aiohttp.ClientSession, image_png: bytes, prompt: str) -> dict[str, Any]:
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"
        image_b64 = base64.b64encode(image_png).decode()

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
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
            "stream": False,
            # The model card builds the prompt with enable_thinking=False
            "chat_template_kwargs": {"enable_thinking": False},
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with session.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                if resp.status in (408, 429, 502, 503, 504):
                    raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

            result: dict[str, Any] = await resp.json()
            try:
                choice = result["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError) as e:
                raise ProviderPermanentError(f"Invalid response format: {e}") from e

            return {"content": str(content or ""), "finish_reason": choice.get("finish_reason")}

    @staticmethod
    def _crop_region(page_png: bytes, bbox: list[float]) -> bytes | None:
        """Crop a [0, 1000) region box out of the page image, as PNG bytes.

        Uses the model card's crop conversion (``x * width / 1000``) unpadded.
        """
        from PIL import Image

        image = Image.open(io.BytesIO(page_png))
        width, height = image.size
        x1 = max(0, min(width, round(bbox[0] * width / COORD_SCALE)))
        y1 = max(0, min(height, round(bbox[1] * height / COORD_SCALE)))
        x2 = max(0, min(width, round(bbox[2] * width / COORD_SCALE)))
        y2 = max(0, min(height, round(bbox[3] * height / COORD_SCALE)))
        # Too small to hold a readable chart — not worth a request.
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        buf = io.BytesIO()
        image.crop((x1, y1, x2, y2)).convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    async def _run_page_async(self, session: aiohttp.ClientSession, page_png: bytes) -> dict[str, Any]:
        from PIL import Image

        width, height = Image.open(io.BytesIO(page_png)).size

        response = await self._call_api(session, page_png, STRUCTURED_PROMPT)
        regions = parse_regions(response["content"])

        charts_read = 0
        if self._chart_pass:
            for region in regions:
                if region["category"] != "chart":
                    continue
                crop = self._crop_region(page_png, region["bbox"])
                if crop is None:
                    continue
                chart = await self._call_api(session, crop, CHART_PROMPT)
                region["chart_raw"] = chart["content"]
                region["chart_html"] = chart_description_to_html(chart["content"])
                if region["chart_html"]:
                    charts_read += 1

        return {
            "structured_raw": response["content"],
            "finish_reason": response["finish_reason"],
            "regions": regions,
            "charts_read": charts_read,
            "image_width": width,
            "image_height": height,
        }

    async def _run_inference_pages_async(self, pages: list[bytes]) -> dict[str, Any]:
        """Run each input page in order, retaining the single-page shape."""
        async with aiohttp.ClientSession() as session:
            results = [await self._run_page_async(session, page) for page in pages]

        merged = dict(results[0])
        merged["_config"] = {
            "server_url": self._server_url,
            "served_model_name": self._served_model_name,
            "dpi": self._dpi,
            "max_tokens": self._max_tokens,
            "chart_pass": self._chart_pass,
        }
        if len(results) > 1:
            merged["page_results"] = results
        return merged

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"OvisOcr3Provider only supports PARSE product type, got {request.product_type}"
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

        except (ProviderPermanentError, ProviderTransientError):
            raise

        except Exception as e:
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
                    "regions": [],
                    "_error": error_msg,
                    "_error_type": type(e).__name__,
                    "_config": {"server_url": self._server_url, "dpi": self._dpi},
                },
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_html_attributes(markdown: str) -> str:
        """Quote unquoted HTML attributes for XML-based metric parsers."""

        def _quote_attrs(match: re.Match) -> str:
            tag_text = match.group(0)
            return re.sub(r'(\w+)=([^\s"\'<>=]+)', r'\1="\2"', tag_text)

        return re.sub(r"<[^>]+>", _quote_attrs, markdown)

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"OvisOcr3Provider only supports PARSE product type, got {raw_result.product_type}"
            )

        page_results = raw_result.raw_output.get("page_results")
        if not isinstance(page_results, list) or not page_results:
            page_results = [raw_result.raw_output]

        page_markdowns: list[str] = []
        layout_pages = []
        for page_number, page_raw in enumerate(page_results, start=1):
            items: list[dict[str, Any]] = []
            parts: list[str] = []
            for region in page_raw.get("regions", []):
                text = _region_markdown(region)
                if text:
                    # The model is prompted for HTML tables, but convert any pipe
                    # tables it emits anyway so GriTS/TEDS can score them.
                    text = self._sanitize_html_attributes(_convert_pipe_tables_to_html(text))
                    parts.append(text)
                items.append(
                    {
                        "label": _layout_label(region["category"], region["bbox"]),
                        "bbox": region["bbox"],
                        "text": text,
                    }
                )

            markdown = "\n\n".join(parts)
            page_markdowns.append(markdown)
            layout_pages.extend(
                build_layout_pages(
                    items=items,
                    image_width=page_raw.get("image_width", 0),
                    image_height=page_raw.get("image_height", 0),
                    markdown=markdown,
                    page_number=page_number,
                )
            )

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            markdown="\n\n".join(md for md in page_markdowns if md),
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
