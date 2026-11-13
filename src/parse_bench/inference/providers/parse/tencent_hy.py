"""Provider for Tencent Hy vision models via TokenHub.

TokenHub (``https://tokenhub-intl.tencentcloudmaas.com/v1``) is OpenAI
chat-completions compatible, and Hy4 uses the same thinking API as DeepSeek:
reasoning is on by default, returned in ``message.reasoning_content``, counted
in ``completion_tokens_details.reasoning_tokens``, and switched off with
``thinking={"type": "disabled"}``. Cached prompt tokens are reported under
``prompt_tokens_details.cached_tokens``. So this reuses the DeepSeek parse
provider with Tencent's endpoint, key and prices.
"""

from __future__ import annotations

from parse_bench.inference.providers.parse.deepseek import DeepSeekParseProvider
from parse_bench.inference.providers.registry import register_provider

# USD per million tokens (input, cached input, output), the rate TokenHub bills.
# Hy4 is billed at 50% of list through 2026-12-31 23:59:59 Beijing time; it then
# reverts to list (0.834, 0.042, 2.501); update this row then, or costs read half.
# Source: https://www.tencentcloud.com/document/product/1300/78937
_TENCENT_HY_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    "hy4": (0.417, 0.021, 1.2505),
}


@register_provider("tencent_hy")
class TencentHYParseProvider(DeepSeekParseProvider):
    """Hy4 page-image parsing through Tencent TokenHub."""

    DEFAULT_MODEL = "hy4"
    VENDOR = "Tencent TokenHub"
    API_KEY_ENV = "TENCENT_TOKENHUB_API_KEY"
    DEFAULT_BASE_URL = "https://tokenhub-intl.tencentcloudmaas.com/v1"
    PRICING_PER_M = _TENCENT_HY_PRICING_PER_M
