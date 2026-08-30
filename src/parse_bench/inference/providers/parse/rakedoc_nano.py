"""rakedoc-nano — ParseBench provider.

`cloudraker/rakedoc-nano <https://huggingface.co/cloudraker/rakedoc-nano>`_ is a
LoRA fine-tune of `florin-inc/florin-parser-nano
<https://huggingface.co/florin-inc/florin-parser-nano>`_ (itself a fine-tune of
`KDLAI/KDL-Frontier-Parser-nano
<https://huggingface.co/KDLAI/KDL-Frontier-Parser-nano>`_, KoreaDeep, 1.2B,
Qwen2-VL architecture) trained on additional table-structure data (row/column
spans, multi-row headers). Weights are AGPL-3.0, inherited from the base model.
Full attribution to KoreaDeep for the base model and pipeline design, and to
Florin for the inline-formatting fine-tune and markdown emission fixes.

This provider is the ``florin_parser_nano`` provider with different weights and
nothing else: every inference stage and the entire markdown emission are
inherited unchanged from that module. Serve the weights exactly as the parent
models are served:

    vllm serve cloudraker/rakedoc-nano \\
      --served-model-name rakedoc-nano \\
      --max-model-len 8192 --gpu-memory-utilization 0.85 \\
      --max-num-seqs 24 --trust-remote-code \\
      --limit-mm-per-prompt '{"image":1}'

Then:

    RAKEDOC_NANO_ENDPOINT_URL=http://localhost:8000/v1 \\
    uv run parse-bench run rakedoc_nano --input_dir data ...

Config (env):
  RAKEDOC_NANO_ENDPOINT_URL  vLLM base URL ending in /v1   (required)
  RAKEDOC_NANO_MODEL         served model name             (default rakedoc-nano)
  All other knobs are inherited from the ``kdl_frontier_nano`` provider and keep
  their ``KDL_NANO_*`` environment variables and defaults.
"""

from __future__ import annotations

import os
from typing import Any

from parse_bench.inference.providers.parse.florin_parser_nano import (
    FlorinParserNanoProvider,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.inference.providers.base import ProviderConfigError


@register_provider("rakedoc_nano")
class RakedocNanoProvider(FlorinParserNanoProvider):
    """cloudraker/rakedoc-nano: the ``florin_parser_nano`` provider with the
    further fine-tuned weights. Serving requirements, every inference stage,
    and the markdown emission are inherited unchanged."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        cfg = dict(base_config or {})
        cfg["endpoint_url"] = (
            cfg.get("endpoint_url") or os.getenv("RAKEDOC_NANO_ENDPOINT_URL") or ""
        )
        if not cfg["endpoint_url"]:
            raise ProviderConfigError(
                "RAKEDOC_NANO_ENDPOINT_URL is required (vLLM OpenAI-compatible base "
                "URL ending in /v1, serving cloudraker/rakedoc-nano)."
            )
        cfg["model"] = (
            cfg.get("model") or os.getenv("RAKEDOC_NANO_MODEL") or "rakedoc-nano"
        )
        # Skip FlorinParserNanoProvider.__init__ env resolution (FLORIN_NANO_*)
        # but keep everything else it inherits, by calling its parent with the
        # fully-resolved config.
        super(FlorinParserNanoProvider, self).__init__(provider_name, cfg)
