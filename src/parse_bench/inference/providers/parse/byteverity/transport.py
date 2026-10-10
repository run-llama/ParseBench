"""Proposer transport for the public API (what ParseBench maintainers run with their own key).

Mirrors ParseBench's own OpenAI provider conventions: Chat Completions, base64 image_url, max_completion_tokens,
reasoning_effort; cost from ParseBench's price tables (imported, not copied) so the cost column is comparable.
Every call's usage is returned for per-page logging.
"""

from __future__ import annotations

import base64
import os
from typing import Any


def cost_usd(model: str, usage: dict[str, int]) -> float:
    from parse_bench.inference.providers.parse.openai import _OPENAI_CACHED_INPUT_PER_M, _OPENAI_PRICING_PER_M

    m = sorted([p for p in _OPENAI_PRICING_PER_M if model.startswith(p)], key=len)
    if not m:
        return 0.0
    rin, rout = _OPENAI_PRICING_PER_M[m[-1]]
    cm = sorted([p for p in _OPENAI_CACHED_INPUT_PER_M if model.startswith(p)], key=len)
    rcached = _OPENAI_CACHED_INPUT_PER_M[cm[-1]] if cm else rin
    inp = usage.get("input_tokens", 0)
    cached = min(usage.get("cached_tokens", 0), inp)
    out = usage.get("output_tokens", 0) + usage.get("reasoning_tokens", 0)
    return ((inp - cached) * rin + cached * rcached + out * rout) / 1_000_000


class OpenAITransport:
    def __init__(
        self,
        max_tokens: int = 32768,
        timeout: float = 600.0,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        effort_style: str = "openai",
    ):
        """Any OpenAI-compatible endpoint (OpenAI, OpenRouter, DeepInfra, Fireworks, …).
        effort_style: "openai" -> reasoning_effort; "openrouter" -> reasoning={"effort"} plus billed cost
        in usage; "none"."""
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("transport=openai needs the OpenAI SDK: pip install 'parse-bench[openai]'") from e
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"transport=openai needs {api_key_env}")
        self._client = openai.OpenAI(api_key=key, timeout=timeout, **({"base_url": base_url} if base_url else {}))
        self._max_tokens = max_tokens
        self._style = effort_style

    def call(self, image_path: str, prompt: str, model: str, effort: str | None) -> tuple[str, dict[str, Any]]:
        b64 = base64.standard_b64encode(open(image_path, "rb").read()).decode()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": self._max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if self._style == "openai" and effort:
            kwargs["reasoning_effort"] = effort
        elif self._style == "openrouter":
            kwargs["extra_body"] = {"usage": {"include": True}, **({"reasoning": {"effort": effort}} if effort else {})}
        r = self._client.chat.completions.create(**kwargs)
        u = r.usage
        details = getattr(u, "prompt_tokens_details", None)
        cdet = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(cdet, "reasoning_tokens", 0) or 0 if cdet else 0
        usage = {
            "model": model,
            "effort": effort,
            "input_tokens": u.prompt_tokens,
            "cached_tokens": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
            "output_tokens": u.completion_tokens - reasoning,
            "reasoning_tokens": reasoning,
        }
        billed = getattr(u, "cost", None)
        if billed is None and isinstance(getattr(u, "model_extra", None), dict):
            billed = u.model_extra.get("cost")
        if billed is not None:  # provider-reported billed cost (OpenRouter usage.include)
            usage["cost_usd"], usage["cost_source"] = float(billed), "billed"
        else:
            try:
                usage["cost_usd"], usage["cost_source"] = cost_usd(model, usage), "price_table"
            except Exception:  # pricing must never fail a call; report it as unknown
                usage["cost_usd"], usage["cost_source"] = None, "unknown"
        return r.choices[0].message.content or "", usage
