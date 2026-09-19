"""Provider for an OvisOCR2 OpenAI-compatible vLLM server.

OvisOCR2 (ATH-MaaS/OvisOCR2) is a 0.8B end-to-end page-level document parsing
VLM post-trained from Qwen/Qwen3.5-0.8B. Given a page image it returns one
Markdown document in natural reading order: Markdown text, LaTeX formulas, HTML
``<table>`` tables, and ``<img src="images/bbox_{l}_{t}_{r}_{b}.jpg" />``
placeholders for charts and pictures.

The prompt, sampling parameters, image-tag filtering, and truncated-repeat
cleanup follow the official ``OvisOCR2Parser`` from the model card.
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

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
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

SERVED_MODEL_NAME = "ovisocr2"

OCR_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    "represent them using an HTML image tag: <"
    'img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, '
    "right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas "
    "as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text "
    "as standard Markdown. Preserve the original text without translation or "
    "paraphrasing."
)

DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.0
MIN_PIXELS = 448 * 448
MAX_PIXELS = 2880 * 2880


@register_provider("ovisocr2")
class OvisOcr2Provider(Provider):
    """Provider for an OvisOCR2 OpenAI-compatible vLLM server.

    Configuration options:
        - server_url (str, required): Server URL. Falls back to the
          ``OVISOCR2_SERVER_URL`` environment variable.
        - served_model_name (str, default="ovisocr2"): Model name in vLLM
        - timeout (int, default=900): Request timeout in seconds
        - dpi (int, default=200): DPI for PDF page rendering
        - api_key_env (str, default="VLLM_API_KEY"): Env var for the API key
        - temperature (float, default=0.0): Sampling temperature
        - max_tokens (int, default=16384): Max output tokens
        - filter_imgtags (bool, default=True): Drop the model's chart/picture
          ``<img src="images/bbox_...">`` placeholder blocks, as the official
          parser does by default
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("OVISOCR2_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "OvisOCR2 provider requires 'server_url' in config or OVISOCR2_SERVER_URL in the environment."
            )
        self._server_url: str = str(server_url)

        self._served_model_name = self.base_config.get("served_model_name", SERVED_MODEL_NAME)
        self._timeout = self.base_config.get("timeout", 900)
        self._dpi = self.base_config.get("dpi", 200)
        self._temperature = self.base_config.get("temperature", DEFAULT_TEMPERATURE)
        self._max_tokens = self.base_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self._filter_imgtags = bool(self.base_config.get("filter_imgtags", True))

        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")

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

    async def _call_openai_api(self, session: aiohttp.ClientSession, image_b64: str) -> str:
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"

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
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {
                "images_kwargs": {
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }
            },
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with session.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    if resp.status == 429:
                        raise ProviderRateLimitError(f"HTTP 429: {error_text[:200]}")
                    if resp.status == 408 or 500 <= resp.status < 600:
                        raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                    raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

                try:
                    result: dict[str, Any] = await resp.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    raise ProviderTransientError("Invalid JSON response from OvisOCR2 server") from e

                try:
                    content = result["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    raise ProviderPermanentError(f"Invalid response format: {e}") from e

                if not content:
                    raise ProviderPermanentError("Empty content response from API")
                return str(content)
        except TimeoutError as e:
            raise ProviderTransientError(f"Request timed out after {self._timeout} seconds") from e
        except aiohttp.ClientError as e:
            raise ProviderTransientError(f"OvisOCR2 transport error: {e}") from e

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode()

        async with aiohttp.ClientSession() as session:
            markdown = await self._call_openai_api(session, image_b64)

        return {
            "markdown": markdown,
            "_config": {
                "server_url": self._server_url,
                "served_model_name": self._served_model_name,
                "dpi": self._dpi,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "filter_imgtags": self._filter_imgtags,
            },
        }

    async def _run_inference_pages_async(self, pages: list[bytes]) -> dict[str, Any]:
        """Run each input page in order, retaining the single-page shape."""
        results = [await self._run_inference_async(page) for page in pages]
        first = results[0]
        if len(results) == 1:
            return first
        merged = dict(first)
        merged["page_results"] = results
        return merged

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"OvisOcr2Provider only supports PARSE product type, got {request.product_type}"
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

        except (ProviderPermanentError, ProviderRateLimitError, ProviderTransientError):
            raise

    @staticmethod
    def _filter_imgtag_blocks(text: str) -> str:
        """Drop the chart/picture placeholder blocks."""
        return "\n\n".join(
            block for block in text.split("\n\n") if not block.strip().startswith('<img src="images/bbox_')
        )

    @staticmethod
    def _clean_truncated_repeats(
        text: str,
        min_text_len: int = 8000,
        max_period: int = 200,
        min_period: int = 1,
        min_repeat_chars: int = 100,
        min_repeat_times: int = 5,
    ) -> str:
        """Trim a degenerate repeating tail."""
        n = len(text)
        if n < min_text_len:
            return text

        max_period = min(max_period, n - 1)
        for unit_len in range(min_period, max_period + 1):
            if text[n - 1] != text[n - 1 - unit_len]:
                continue

            match_len = 1
            idx = n - 2
            while idx >= unit_len and text[idx] == text[idx - unit_len]:
                match_len += 1
                idx -= 1

            total_len = match_len + unit_len
            repeat_times = total_len // unit_len
            tail_len = total_len % unit_len

            if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
                return text[: n - total_len + unit_len] + text[n - tail_len :]

        return text

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
                f"OvisOcr2Provider only supports PARSE product type, got {raw_result.product_type}"
            )

        filter_imgtags = bool(raw_result.raw_output.get("_config", {}).get("filter_imgtags", self._filter_imgtags))

        page_results = raw_result.raw_output.get("page_results")
        if not isinstance(page_results, list) or not page_results:
            page_results = [raw_result.raw_output]

        page_markdowns: list[str] = []
        for page_raw in page_results:
            markdown = page_raw.get("markdown", "")
            if not markdown:
                continue
            markdown = markdown.strip()
            if filter_imgtags:
                markdown = self._filter_imgtag_blocks(markdown)
            markdown = self._clean_truncated_repeats(markdown)
            markdown = _convert_pipe_tables_to_html(markdown)
            markdown = self._sanitize_html_attributes(markdown)
            page_markdowns.append(markdown)

        markdown = "\n\n".join(page_markdowns)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
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
