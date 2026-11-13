"""Provider for DeepSeek vision-based PARSE.

DeepSeek-V4.1-Flash is served as ``deepseek-flash`` through DeepSeek's
OpenAI-compatible chat completions endpoint (``https://api.deepseek.com``). Its
image input accepts JPEG/PNG/GIF/WebP only — there is no PDF content block — so
this provider runs the image-based ``parse_with_layout`` mode: pages are
rendered locally and sent in an ``image_url`` block. There is no
``parse_with_layout_file`` counterpart, because the model never sees the PDF.

This subclasses :class:`OpenAIProvider` to reuse its ``parse_with_layout``
plumbing (PDF-to-image rendering, the ``<div data-bbox data-label>`` layout
prompt/parse machinery, and ``normalize``). Only the DeepSeek-specific pieces are
overridden: the client/auth, the per-page API call, the thinking-mode
parameters, token accounting, pricing, and error wording.

Thinking mode is on by default at ``high`` effort; it is switched with
``thinking={"type": "enabled"|"disabled"}`` (which the OpenAI SDK carries in
``extra_body``) and the effort with the top-level ``reasoning_effort``. In
thinking mode ``temperature`` is ignored by the API, so it is not sent.
"""

from __future__ import annotations

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
    parse_layout_blocks,
    resolve_layout_prompts,
)
from parse_bench.inference.providers.parse.openai import OpenAIProvider
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult

# DeepSeek peak pricing: USD per million tokens (input cache miss, input cache
# hit, output). Cache hits are credited in run_inference (the inherited
# OpenAIProvider cost formula only has input/output terms). Off-peak rates are
# half of these; peak is used so the benchmark cost does not swing with the hour
# a run happens to start. Source: https://api-docs.deepseek.com/quick_start/pricing
_DEEPSEEK_PARSE_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.30, 0.006, 1.20),
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
}

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@register_provider("deepseek")
class DeepSeekParseProvider(OpenAIProvider):
    """DeepSeek-V4.1-Flash document parsing through the DeepSeek API."""

    DEFAULT_MODEL = "deepseek-flash"

    # Max image dimension the API accepts per side, and the per-image size cap
    # for inline base64 (base64 adds ~33% overhead, so cap the raw bytes).
    MAX_IMAGE_DIMENSION = 8192
    MAX_IMAGE_SIZE_BYTES = int(32 * 1024 * 1024 * 3 / 4)

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        # Skip OpenAIProvider.__init__ (it demands OPENAI_API_KEY and an OpenAI
        # client); wire the DeepSeek client and the fields run_inference/normalize use.
        Provider.__init__(self, provider_name, base_config)

        self._api_key = self.base_config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "DeepSeek API key is required. Set DEEPSEEK_API_KEY or pass api_key in base_config."
            )

        self._model = self.base_config.get("model", self.DEFAULT_MODEL)
        self._dpi = self.base_config.get("dpi", 150)
        self._max_tokens = self.base_config.get("max_tokens", 32768)
        # Thinking is on by default and a page can take a while — give it more
        # room than the OpenAI default of 120s.
        self._timeout = self.base_config.get("timeout", 600)
        self._reasoning_effort = self.base_config.get("reasoning_effort", None)
        self._thinking = self.base_config.get("thinking", None)
        self._base_url = self.base_config.get("base_url", _DEEPSEEK_BASE_URL)
        self._mode = self.base_config.get("mode", "parse_with_layout")
        # The shared layout prompt requests normalized 0-1000 coordinates, and
        # inherited run/normalize code records and consumes this scale.
        self._bbox_scale = self.base_config.get("bbox_scale", 1000)
        self._layout_system_prompt, self._layout_user_prompt = resolve_layout_prompts(self._bbox_scale, self._mode)
        self._cached_input_price_per_1m = float(self.base_config.get("cached_input_price_per_1m", self._pricing3()[1]))
        # Per-thread tally of cache-hit tokens across a request's per-page API
        # calls. The runner shares one provider instance across a thread pool, so
        # a plain attribute would race between concurrent documents; thread-local
        # state is private to the thread running a single run_inference call.
        self._cache_tls = threading.local()

        if self._mode != "parse_with_layout":
            raise ProviderConfigError(
                f"Invalid mode '{self._mode}'. DeepSeek accepts images only, so 'parse_with_layout' is the only mode."
            )

        if self._thinking is not None and self._thinking not in ("enabled", "disabled"):
            raise ProviderConfigError(f"Invalid thinking '{self._thinking}'. Must be 'enabled' or 'disabled'.")

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout)
        except ImportError as e:
            raise ProviderConfigError("openai package not installed. Run: pip install openai") from e

    def _pricing3(self) -> tuple[float, float, float]:
        """Longest-prefix (input, cached_input, output) rate per 1M tokens."""
        matches = [(p, r) for p, r in _DEEPSEEK_PARSE_PRICING_PER_M.items() if self._model.startswith(p)]
        return max(matches, key=lambda x: len(x[0]))[1] if matches else (0.0, 0.0, 0.0)

    def _get_pricing(self) -> tuple[float, float]:
        # The inherited cost formula bills (input, output); the cache-hit
        # discount is applied separately in run_inference.
        in_rate, _cached_rate, out_rate = self._pricing3()
        return in_rate, out_rate

    @staticmethod
    def _read_cached_tokens(response) -> int:  # type: ignore[no-untyped-def]
        """Cache-hit tokens the API reports for this call, 0 if none.

        DeepSeek reports the split both as its own ``prompt_cache_hit_tokens``
        field and, for OpenAI compatibility, under ``prompt_tokens_details``.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        hit = getattr(usage, "prompt_cache_hit_tokens", None)
        if hit is not None:
            return int(hit or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        return int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        # Tally cache-hit tokens across this request's per-page calls, run the
        # inherited parse/normalize path (which bills every input token at the
        # full cache-miss rate), then credit the cache-hit tokens down to the
        # cheaper rate.
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

    def _raise_deepseek_error(self, e: Exception) -> NoReturn:
        """Classify a DeepSeek SDK exception as transient (retried) or permanent.

        DeepSeek returns 429 when the account's concurrency limit is exceeded
        rather than for a per-minute quota, so it clears as soon as in-flight
        requests drain and is always worth retrying.
        """
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
            raise ProviderTransientError(f"Transient error calling DeepSeek API: {e}") from e
        raise ProviderPermanentError(f"Error calling DeepSeek API: {e}") from e

    # DeepSeek returns the chain of thought in ``message.reasoning_content`` and
    # counts it in ``completion_tokens``, with the count broken out under
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
        if not thinking_tok:
            thinking_tok = int(getattr(usage, "reasoning_tokens", 0) or 0)
        visible_tok = max(0, completion_tok - thinking_tok)
        return {
            "input_tokens": input_tok,
            "output_tokens": visible_tok,
            "thinking_tokens": thinking_tok,
            "total_tokens": total_tok,
        }

    def _layout_request_kwargs(self, image: Image.Image) -> dict[str, Any]:
        img_base64 = self._image_to_base64(image)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": self._layout_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        {"type": "text", "text": self._layout_user_prompt},
                    ],
                },
            ],
        }
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if self._thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": self._thinking}}
        return kwargs

    def _parse_image_with_layout(self, image: Image.Image) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
        """Send a page image to DeepSeek with the layout prompt via an image_url block."""
        try:
            response = self._client.chat.completions.create(**self._layout_request_kwargs(image))
            self._cache_tls.value = getattr(self._cache_tls, "value", 0) + self._read_cached_tokens(response)
            usage = self._extract_usage(response)
            content = response.choices[0].message.content if response.choices else ""
            text = content or ""
            return parse_layout_blocks(text), text, usage
        except Exception as e:
            self._raise_deepseek_error(e)
