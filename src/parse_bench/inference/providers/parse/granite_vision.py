"""Provider for Granite Vision Modal servers.

ibm-granite/granite-4.0-3b-vision (~4.25B) is an enterprise-grade document
extraction model supporting task tags:
- "<tables_html>" -- extract tables as HTML
- "<chart2csv>" -- extract chart data as CSV
- "<chart2summary>" -- describe chart content
- Free-form text prompts for general OCR

This provider supports two API formats:
- "openai": OpenAI-compatible vLLM API (for granite_vision_server.py)
- "simple": JSON API with image_base64 (for granite_vision_pipeline_server.py)
"""

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
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# Model name registered in vLLM
SERVED_MODEL_NAME = "granite-vision"

# Task-specific prompts / tags for Granite Vision
TASK_PROMPTS = {
    "ocr": "Convert the text in this image to markdown.",
    "tables_html": "<tables_html>",
    "tables_json": "<tables_json>",
    "chart2csv": "<chart2csv>",
    "chart2summary": "<chart2summary>",
}


@register_provider("granite_vision")
class GraniteVisionProvider(Provider):
    """
    Provider for Granite Vision Modal servers.

    Configuration options:
        - server_url (str, required): Modal server URL
        - api_format (str, default="openai"): "openai" or "simple"
        - task (str, default="ocr"): Task prompt -- "ocr", "tables_html", etc.
        - timeout (int, default=600): Request timeout in seconds
        - dpi (int, default=150): DPI for PDF to image conversion
        - api_key_env (str, default="VLLM_API_KEY"): Env var for API key
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url = self.base_config.get("server_url") or os.getenv("GRANITE_VISION_SERVER_URL")
        if not server_url:
            raise ProviderConfigError(
                "GraniteVision provider requires 'server_url' in config. "
                "Example: https://llamaindex--granite-vision-vllm-granitevisionserver-serve.modal.run"
            )
        self._server_url: str = str(server_url)

        self._api_format = self.base_config.get("api_format", "openai")
        if self._api_format not in ("openai", "simple"):
            raise ProviderConfigError(f"Invalid api_format '{self._api_format}'. Must be 'openai' or 'simple'.")

        self._task = self.base_config.get("task", "ocr")
        if self._task not in TASK_PROMPTS:
            raise ProviderConfigError(f"Invalid task '{self._task}'. Must be one of: {list(TASK_PROMPTS.keys())}")

        self._timeout = self.base_config.get("timeout", 600)
        self._dpi = self.base_config.get("dpi", 150)

        # API key for authenticated vLLM endpoints
        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        self._api_key = os.environ.get(api_key_env, "")

    def _pdf_to_image(self, pdf_path: Path) -> bytes:
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path, dpi=self._dpi)
            if not images:
                raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
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
        try:
            return file_path.read_bytes()
        except Exception as e:
            raise ProviderPermanentError(f"Error reading image file: {e}") from e

    async def _call_openai_api(self, session: aiohttp.ClientSession, image_b64: str) -> str:
        api_url = f"{self._server_url.rstrip('/')}/v1/chat/completions"

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
                        {
                            "type": "text",
                            "text": TASK_PROMPTS.get(self._task, TASK_PROMPTS["ocr"]),
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 4096,
            "stream": False,
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
            return str(content)

    async def _call_simple_api(self, session: aiohttp.ClientSession, image_b64: str) -> str:
        api_url = self._server_url.rstrip("/")

        payload: dict[str, str] = {"image_base64": image_b64}

        async with session.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                if resp.status in (408, 502, 503, 504):
                    raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

            result = await resp.json()
            if result.get("status") == "error":
                raise ProviderPermanentError(result.get("error", "Unknown error from API"))

            content = result.get("markdown", "")
            if not content:
                raise ProviderPermanentError("Empty markdown response from API")
            return str(content)

    async def _run_inference_async(self, image_bytes: bytes) -> dict[str, Any]:
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
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"GraniteVisionProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            image_bytes = self._pdf_to_image(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            image_bytes = self._read_image(file_path)
        else:
            raise ProviderPermanentError(
                f"Unsupported file type: {suffix}. Supported: .pdf, .png, .jpg, .jpeg, .webp, .tiff, .bmp"
            )

        try:
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
        """Quote unquoted HTML attributes for XML-based metric parsers."""

        def _quote_attrs(match: re.Match) -> str:
            tag_text = match.group(0)
            tag_text = re.sub(
                r'(\w+)=([^\s"\'<>=]+)',
                r'\1="\2"',
                tag_text,
            )
            return tag_text

        return re.sub(r"<[^>]+>", _quote_attrs, markdown)

    @staticmethod
    def _convert_md_tables_to_html(content: str) -> str:
        """Convert markdown pipe tables to HTML <table> elements.

        Granite Vision's <tables_html> tag outputs HTML tables directly,
        but the OCR prompt may produce markdown pipe tables for inline tables.
        This converts them to HTML for GriTS/TEDS metric evaluation.
        """
        import markdown2

        lines = content.split("\n")
        result_parts: list[str] = []
        table_lines: list[str] = []
        in_table = False

        for line in lines:
            is_table_line = "|" in line and line.strip().startswith("|")
            if is_table_line:
                if not in_table:
                    in_table = True
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                if in_table:
                    if len(table_lines) >= 2:
                        table_md = "\n".join(table_lines)
                        html = markdown2.markdown(table_md, extras=["tables"]).strip()
                        if "<table>" in html.lower():
                            result_parts.append(html)
                        else:
                            result_parts.extend(table_lines)
                    else:
                        result_parts.extend(table_lines)
                    table_lines = []
                    in_table = False
                result_parts.append(line)

        # Handle trailing table
        if in_table and len(table_lines) >= 2:
            table_md = "\n".join(table_lines)
            html = markdown2.markdown(table_md, extras=["tables"]).strip()
            if "<table>" in html.lower():
                result_parts.append(html)
            else:
                result_parts.extend(table_lines)
        elif in_table:
            result_parts.extend(table_lines)

        return "\n".join(result_parts)

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"GraniteVisionProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        markdown = raw_result.raw_output.get("markdown", "")
        if markdown:
            # Convert any markdown pipe tables to HTML so GriTS/TEDS can score them
            markdown = self._convert_md_tables_to_html(markdown)
            markdown = self._sanitize_html_attributes(markdown)

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
