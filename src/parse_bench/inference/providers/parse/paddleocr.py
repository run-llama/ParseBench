"""Provider for PaddleOCR Modal servers."""

import asyncio
import base64
import io
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
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

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

    def _pdf_to_image(self, pdf_path: Path) -> bytes:
        """
        Convert a PDF to a PNG image (first page only).

        :param pdf_path: Path to the PDF file
        :return: PNG image bytes
        :raises ProviderPermanentError: If conversion fails
        """
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=self._dpi)
            if not images:
                raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")

            # Use first page only
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            return buf.getvalue()

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
            "model": SERVED_MODEL_NAME,
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
    ) -> str:
        """
        Call the simple pipeline API.

        :param session: aiohttp session
        :param image_b64: Base64-encoded image
        :return: Markdown content from the API response
        """
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

            return content  # type: ignore[no-any-return]

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Run async inference on an image.

        :param image_bytes: Image bytes
        :return: Raw response dictionary with markdown
        """
        image_b64 = base64.b64encode(image_bytes).decode()

        async with aiohttp.ClientSession() as session:
            if self._api_format == "simple":
                markdown = await self._call_simple_api(session, image_b64)
            else:
                markdown = await self._call_openai_api(session, image_b64)

        return {
            "markdown": markdown,
            "_config": {
                "server_url": self._server_url,
                "api_format": self._api_format,
                "task": self._task,
                "dpi": self._dpi,
            },
        }

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

        # Convert to image if needed
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            image_bytes = self._pdf_to_image(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            image_bytes = self._read_image(file_path)
        else:
            raise ProviderPermanentError(
                f"Unsupported file type: {suffix}. Supported types: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
            )

        try:
            # Run async inference
            raw_output = asyncio.run(self._run_inference_async(image_bytes))

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

        # Extract markdown from raw output
        markdown = raw_result.raw_output.get("markdown", "")

        # Sanitize HTML attributes for XML-based metric parsers (e.g. GriTS)
        if markdown:
            markdown = self._sanitize_html_attributes(markdown)

        # Create ParseOutput with document-level markdown
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],  # PaddleOCR returns single page/document, leave pages empty
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
