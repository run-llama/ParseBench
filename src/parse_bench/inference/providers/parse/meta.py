"""Provider for Meta Muse vision-based PARSE.

The Meta Model API (https://api.meta.ai/v1) is OpenAI-compatible and accepts PDF
``input_file`` parts on the Responses API, so this reuses the OpenAI provider's
Responses layout path (same prompts, page splitting and normalization) with
Meta's endpoint, key and pricing.
"""

import os
from typing import Any, NoReturn

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._layout_utils import resolve_layout_prompts
from parse_bench.inference.providers.parse.openai import OpenAIProvider
from parse_bench.inference.providers.registry import register_provider

# Meta Standard-mode pricing: USD per million tokens (input, cached input, output).
# Reasoning tokens are billed at the output rate. Muse Ember's long-context rates
# start above 200K input tokens, which a one-page request never reaches.
_META_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-ember-1.0": (0.10, 0.004, 0.50),
}


@register_provider("meta")
class MetaProvider(OpenAIProvider):
    """Meta Muse vision-based document parsing (OpenAI-compatible Responses API)."""

    BASE_URL = "https://api.meta.ai/v1"

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `model`: Muse model to use (default: "muse-spark-1.3")
            - `dpi`: DPI for PDF to image conversion (default: 150)
            - `max_tokens`: Max output tokens per response, reasoning included (default: 32768)
            - `timeout`: Request timeout in seconds (default: 300)
            - `reasoning_effort`: "minimal" through "max"; omitted uses the model's own
              level. Muse has no "none" tier.
            - `mode`: "parse_with_layout_file" (default, one single-page PDF per request)
              or "parse_with_layout" (one rendered page image per request)
        """
        # Bypass OpenAIProvider.__init__ (it reads OPENAI_API_KEY and targets the
        # OpenAI base URL); every request goes through the Responses API.
        Provider.__init__(self, provider_name, base_config)

        self._api_key = os.environ.get("META_API_KEY")
        if not self._api_key:
            raise ProviderConfigError("META_API_KEY environment variable not set")

        self._model = self.base_config.get("model", "muse-spark-1.3")
        self._dpi = self.base_config.get("dpi", 150)
        self._max_tokens = self.base_config.get("max_tokens", 32768)
        self._timeout = self.base_config.get("timeout", 300)
        self._reasoning_effort = self.base_config.get("reasoning_effort", None)
        self._mode = self.base_config.get("mode", "parse_with_layout_file")
        self._api = "responses"

        if self._mode not in ("parse_with_layout", "parse_with_layout_file"):
            raise ProviderConfigError(
                f"Invalid mode '{self._mode}'. Must be 'parse_with_layout' or 'parse_with_layout_file'."
            )
        if self._model not in _META_PRICING_PER_M:
            raise ProviderConfigError(f"No Meta pricing for model '{self._model}'")

        self._bbox_scale = 1000
        self._layout_system_prompt, self._layout_user_prompt = resolve_layout_prompts(self._bbox_scale, self._mode)

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, base_url=self.BASE_URL, timeout=self._timeout)
        except ImportError as e:
            raise ProviderConfigError("openai package not installed. Run: pip install openai") from e

    def _get_pricing(self) -> tuple[float, float]:
        input_rate, _, output_rate = _META_PRICING_PER_M[self._model]
        return input_rate, output_rate

    def _get_cached_input_rate(self, input_rate: float) -> float:
        return _META_PRICING_PER_M[self._model][1]

    def _get_cache_write_rate(self, input_rate: float) -> float:
        return input_rate

    def _raise_openai_error(self, e: Exception) -> NoReturn:
        error_str = str(e).lower()
        if any(kw in error_str for kw in ["timeout", "connection", "network"]):
            raise ProviderTransientError(f"Transient error calling Meta API: {e}") from e
        if any(kw in error_str for kw in ["rate_limit", "rate limit", "429"]):
            raise ProviderTransientError(f"Rate limited: {e}") from e
        status_code = getattr(e, "status_code", None)
        if isinstance(status_code, int) and status_code >= 500:
            raise ProviderTransientError(f"Transient Meta server error: {e}") from e
        raise ProviderPermanentError(f"Error calling Meta API: {e}") from e
