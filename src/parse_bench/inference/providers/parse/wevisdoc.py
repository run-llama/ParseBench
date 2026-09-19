# ruff: noqa: E501
"""Provider for WeVisDoc 2B and 4B OpenAI-compatible vLLM servers.

WeVisDoc is an end-to-end document parser for page images, fine-tuned from
Qwen3-VL-2B/4B-Instruct. One page image in, structured Markdown out, with
LaTeX formulas and HTML ``<table>`` tables.

The prompts, temperature, and token budget below are the official ones from
``wevisdoc/prompts.py`` and ``wevisdoc/client.py`` in
`github.com/Tencent/WeVisDoc <https://github.com/Tencent/WeVisDoc>`_. The
provider talks to an OpenAI-compatible ``/v1/chat/completions`` endpoint.

The official system prompt tells the model to ignore figure content, and the
model emits no bounding boxes, so this provider produces Markdown only.
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

SERVED_MODEL_NAME = "wevisdoc-2b"

DEFAULT_SYSTEM_PROMPT = r"""You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
- Enclose block formulas with \[ \]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
"""

DEFAULT_PROMPT = "Convert this document image to Markdown."


@register_provider("wevisdoc")
class WeVisDocProvider(Provider):
    """Provider for WeVisDoc OpenAI-compatible vLLM servers.

    Configuration options:
        - server_url (str, optional): Server URL. Takes precedence over the
          environment variable named by ``server_url_env``.
        - server_url_env (str, default="WEVISDOC_SERVER_URL"): Environment
          variable containing the server URL.
        - served_model_name (str, default="wevisdoc-2b"): Model name in vLLM.
        - timeout (int, default=600): Request timeout in seconds.
        - dpi (int, default=200): DPI for PDF page rendering.
        - api_key_env (str, default="VLLM_API_KEY"): Environment variable for
          the optional API key.
        - temperature (float, default=0.0): Sampling temperature.
        - max_tokens (int, default=8192): Maximum output tokens.
        - system_prompt (str, default=official): System instruction.
        - prompt (str, default=official): User instruction.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        server_url_env = self.base_config.get("server_url_env", "WEVISDOC_SERVER_URL")
        server_url = self.base_config.get("server_url") or os.getenv(server_url_env)
        if not server_url:
            raise ProviderConfigError(
                f"WeVisDoc provider requires 'server_url' in config or {server_url_env} in the environment."
            )
        self._server_url: str = str(server_url)

        self._served_model_name = self.base_config.get("served_model_name", SERVED_MODEL_NAME)
        self._timeout = self.base_config.get("timeout", 600)
        self._dpi = self.base_config.get("dpi", 200)
        self._temperature = self.base_config.get("temperature", 0.0)
        self._max_tokens = self.base_config.get("max_tokens", 8192)
        self._system_prompt = self.base_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._prompt = self.base_config.get("prompt", DEFAULT_PROMPT)

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

        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": self._prompt},
                ],
            }
        )

        payload = {
            "model": self._served_model_name,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
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
                    if resp.status == 408:
                        raise ProviderTransientError(f"HTTP 408: {error_text[:200]}")
                    if 500 <= resp.status < 600:
                        raise ProviderTransientError(f"HTTP {resp.status}: {error_text[:200]}")
                    raise ProviderPermanentError(f"HTTP {resp.status}: {error_text[:200]}")

                try:
                    result = await resp.json()
                except ValueError as e:
                    raise ProviderTransientError(f"Invalid JSON response: {e}") from e

                try:
                    choice = result["choices"][0]
                except (KeyError, IndexError, TypeError) as e:
                    raise ProviderPermanentError(f"Invalid response format: {e}") from e

                if not isinstance(choice, dict):
                    raise ProviderPermanentError("Invalid response format: choice must be an object")
                if choice.get("finish_reason") is None:
                    raise ProviderPermanentError("Invalid response format: choice is missing finish_reason")

                finish_reason = choice["finish_reason"]
                if finish_reason != "stop":
                    raise ProviderPermanentError(
                        f"Incomplete generation (finish_reason={finish_reason!r}); "
                        "increase max_tokens or the server context limit if the output was truncated"
                    )

                try:
                    content = choice["message"]["content"]
                except (KeyError, TypeError) as e:
                    raise ProviderPermanentError(f"Invalid response format: {e}") from e

                if not content:
                    raise ProviderPermanentError("Empty content response from API")
                return str(content)
        except (ProviderPermanentError, ProviderRateLimitError, ProviderTransientError):
            raise
        except TimeoutError as e:
            raise ProviderTransientError(f"Request timed out after {self._timeout} seconds") from e
        except aiohttp.ClientError as e:
            raise ProviderTransientError(f"Transport error calling WeVisDoc: {e}") from e

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
            },
        }

    async def _run_inference_pages_async(self, pages: list[bytes]) -> dict[str, Any]:
        """Run each input page in order, retaining the one-page shape."""
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
                f"WeVisDocProvider only supports PARSE product type, got {request.product_type}"
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
        except TimeoutError as e:
            raise ProviderTransientError(f"Request timed out after {self._timeout} seconds") from e
        except aiohttp.ClientError as e:
            raise ProviderTransientError(f"Transport error calling WeVisDoc: {e}") from e
        except json.JSONDecodeError as e:
            raise ProviderTransientError(f"Invalid JSON response: {e}") from e
        except Exception as e:
            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output={
                    "markdown": "",
                    "_error": str(e),
                    "_error_type": type(e).__name__,
                    "_config": {
                        "server_url": self._server_url,
                        "served_model_name": self._served_model_name,
                        "dpi": self._dpi,
                    },
                },
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

    @staticmethod
    def _close_unclosed_table_tags(content: str) -> str:
        """Auto-close unclosed HTML table tags from truncated model output."""
        opens = content.count("<table>")
        closes = content.count("</table>")
        if opens > closes:
            if not content.rstrip().endswith(">"):
                content += "</td></tr>"
            content += "</table>" * (opens - closes)
        return content

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
                f"WeVisDocProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        page_results = raw_result.raw_output.get("page_results")
        if not isinstance(page_results, list) or not page_results:
            page_results = [raw_result.raw_output]

        page_markdowns: list[str] = []
        for page_raw in page_results:
            markdown = page_raw.get("markdown", "")
            if markdown:
                markdown = self._close_unclosed_table_tags(markdown)
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
