"""Provider for GLM (z.ai) vision-based PARSE.

GLM-5.3-flash is served through z.ai's OpenAI-compatible chat completions
endpoint (``https://api.z.ai/api/paas/v4``). It is a vision-language model that
accepts documents directly: PDFs ride in a ``file_url`` content block and raw
images in an ``image_url`` block, both as base64 data URLs — z.ai exposes no
Files API, so nothing is uploaded first.

This subclasses :class:`OpenAIProvider` to reuse its ``parse_with_layout_file``
plumbing (per-page PDF splitting, the ``<div data-bbox data-label>`` layout
prompt/parse machinery, and ``normalize``). Only the pieces that are genuinely
z.ai-specific are overridden: the client/auth, the per-page API calls (z.ai uses
``file_url`` / ``image_url`` blocks rather than OpenAI's ``type: file`` blocks),
token accounting, pricing, and error wording.

Thinking is always on for GLM-5.3-flash and cannot be disabled, so the layout
pipeline uses the model's default reasoning; ``reasoning_tokens`` are reported as
part of ``completion_tokens`` and billed at the output rate.
"""

from __future__ import annotations

import base64
import os
import threading
from typing import Any, NoReturn

from PIL import Image

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._layout_utils import (
    SYSTEM_PROMPT_LAYOUT,
    USER_PROMPT_LAYOUT,
    parse_layout_blocks,
)
from parse_bench.inference.providers.parse.openai import OpenAIProvider
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult

# z.ai list pricing: USD per million tokens (input, cached_input, output).
# Cached reads are credited in run_inference (the inherited OpenAIProvider cost
# formula only has input/output terms). A 50%-off promo (0.075 / 0.015 / 0.25)
# runs through 2026-09-09; list price is used so the benchmark cost stays stable
# after it ends. Source: https://docs.z.ai/guides/overview/pricing (verified 2026-08-26)
_GLM_ZAI_PARSE_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    "glm-5.3-flash": (0.15, 0.03, 0.50),
}

_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"


@register_provider("glm_zai")
class GLMZaiParseProvider(OpenAIProvider):
    """GLM-5.3-flash document parsing through z.ai's OpenAI-compatible API."""

    DEFAULT_MODEL = "glm-5.3-flash"

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        # Skip OpenAIProvider.__init__ (it demands OPENAI_API_KEY and an OpenAI
        # client); wire the z.ai client and the fields run_inference/normalize use.
        Provider.__init__(self, provider_name, base_config)

        self._api_key = self.base_config.get("api_key") or os.environ.get("GLM_ZAI_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "GLM z.ai API key is required. Set GLM_ZAI_API_KEY or pass api_key in base_config."
            )

        self._model = self.base_config.get("model", self.DEFAULT_MODEL)
        self._dpi = self.base_config.get("dpi", 150)
        self._max_tokens = self.base_config.get("max_tokens", 32768)
        # Thinking is always on, so a page can take a while — give it more room
        # than the OpenAI default of 120s.
        self._timeout = self.base_config.get("timeout", 600)
        self._reasoning_effort = self.base_config.get("reasoning_effort", None)
        self._temperature = self.base_config.get("temperature", 0)
        self._base_url = self.base_config.get("base_url", _ZAI_BASE_URL)
        self._mode = self.base_config.get("mode", "parse_with_layout_file")
        # The shared layout prompt requests normalized 0-1000 coordinates, and
        # inherited run/normalize code records and consumes this scale.
        self._bbox_scale = self.base_config.get("bbox_scale", 1000)
        self._cached_input_price_per_1m = float(self.base_config.get("cached_input_price_per_1m", self._pricing3()[1]))
        # Per-thread tally of cache-read tokens across a request's per-page API
        # calls. The runner shares one provider instance across a thread pool, so
        # a plain attribute would race between concurrent documents; thread-local
        # state is private to the thread running a single run_inference call.
        self._cache_tls = threading.local()

        if self._mode not in ("image", "file", "parse_with_layout", "parse_with_layout_file"):
            raise ProviderConfigError(
                f"Invalid mode '{self._mode}'. "
                "Must be 'image', 'file', 'parse_with_layout', or 'parse_with_layout_file'."
            )

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)
        except ImportError as e:
            raise ProviderConfigError("openai package not installed. Run: pip install openai") from e

    def _pricing3(self) -> tuple[float, float, float]:
        """Longest-prefix (input, cached_input, output) rate per 1M tokens."""
        matches = [(p, r) for p, r in _GLM_ZAI_PARSE_PRICING_PER_M.items() if self._model.startswith(p)]
        return max(matches, key=lambda x: len(x[0]))[1] if matches else (0.0, 0.0, 0.0)

    def _get_pricing(self) -> tuple[float, float]:
        # The inherited cost formula bills (input, output); the cached-read
        # discount is applied separately in run_inference.
        in_rate, _cached_rate, out_rate = self._pricing3()
        return in_rate, out_rate

    @staticmethod
    def _read_cached_tokens(response) -> int:  # type: ignore[no-untyped-def]
        """Cache-read (hit) tokens the API reports for this call, 0 if none."""
        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
        return int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        # Tally cache-read tokens across this request's per-page calls, run the
        # inherited parse/normalize path (which bills every input token at the
        # full input rate), then credit the cache-read tokens down to the cheaper
        # cached rate. z.ai returns the cache-hit count, so it should not be
        # billed as fresh input.
        self._cache_tls.value = 0
        result = super().run_inference(pipeline, request)
        cached = int(getattr(self._cache_tls, "value", 0) or 0)
        raw = result.raw_output
        raw["cached_input_tokens"] = cached
        if cached > 0:
            in_rate, _out_rate = self._get_pricing()
            credit = cached * (in_rate - self._cached_input_price_per_1m) / 1_000_000
            raw["cost_usd"] = max(0.0, float(raw.get("cost_usd", 0.0)) - credit)
            num_pages = raw.get("num_pages") or 0
            if num_pages > 0:
                raw["cost_per_page_usd"] = raw["cost_usd"] / num_pages
        return result

    def _raise_glm_error(self, e: Exception) -> NoReturn:
        """Classify a z.ai/GLM SDK exception as transient (retried) or permanent."""
        status_code = getattr(e, "status_code", None)
        is_retryable_status = isinstance(status_code, int) and (
            status_code in {408, 409, 429} or 500 <= status_code < 600
        )
        is_retryable_type = isinstance(e, (TimeoutError, ConnectionError)) or type(e).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }
        if is_retryable_status or is_retryable_type:
            raise ProviderTransientError(f"Transient error calling z.ai GLM API: {e}") from e
        raise ProviderPermanentError(f"Error calling z.ai GLM API: {e}") from e

    # OpenAI's chat-completions usage reports the full ``completion_tokens``
    # (visible output *plus* reasoning), with the reasoning count broken out in
    # ``completion_tokens_details``. Splitting them here — output = visible,
    # thinking = reasoning — lets the inherited cost formula bill
    # ``(output + thinking)`` = the full completion at the output rate exactly
    # once, while still recording the reasoning token count.
    @staticmethod
    def _extract_usage(response) -> dict[str, int]:  # type: ignore[no-untyped-def]
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "total_tokens": 0}
        input_tok = getattr(usage, "prompt_tokens", 0) or 0
        completion_tok = getattr(usage, "completion_tokens", 0) or 0
        total_tok = getattr(usage, "total_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        thinking_tok = (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        visible_tok = max(0, completion_tok - thinking_tok)
        return {
            "input_tokens": input_tok,
            "output_tokens": visible_tok,
            "thinking_tokens": thinking_tok,
            "total_tokens": total_tok,
        }

    def _layout_request_kwargs(self, file_block: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_LAYOUT},
                {
                    "role": "user",
                    "content": [file_block, {"type": "text", "text": USER_PROMPT_LAYOUT}],
                },
            ],
        }
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        return kwargs

    def _parse_pdf_page_with_layout(self, pdf_bytes: bytes) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
        """Send a single-page PDF to GLM with the layout prompt via a file_url block."""
        pdf_base64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
        file_block = {"type": "file_url", "file_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
        try:
            response = self._client.chat.completions.create(**self._layout_request_kwargs(file_block))
            self._cache_tls.value = getattr(self._cache_tls, "value", 0) + self._read_cached_tokens(response)
            usage = self._extract_usage(response)
            content = response.choices[0].message.content if response.choices else ""
            text = content or ""
            return parse_layout_blocks(text), text, usage
        except Exception as e:
            self._raise_glm_error(e)

    def _parse_image_with_layout(self, image: Image.Image) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
        """Send a page image to GLM with the layout prompt via an image_url block."""
        img_base64 = self._image_to_base64(image)
        file_block = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        try:
            response = self._client.chat.completions.create(**self._layout_request_kwargs(file_block))
            self._cache_tls.value = getattr(self._cache_tls, "value", 0) + self._read_cached_tokens(response)
            usage = self._extract_usage(response)
            content = response.choices[0].message.content if response.choices else ""
            text = content or ""
            return parse_layout_blocks(text), text, usage
        except Exception as e:
            self._raise_glm_error(e)
