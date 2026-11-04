"""aeddix-alpine-ocr (KDL pipeline): ParseBench provider.

`aeddix-labs/aeddix-alpine-ocr <https://huggingface.co/aeddix-labs/aeddix-alpine-ocr>`_
(tag ``v0.1``) is a fine-tune of `opendatalab/MinerU2.5-Pro-2605-1.2B
<https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B>`_ (OpenDataLab,
1.2B, Qwen2-VL architecture). Weights are Apache-2.0.

This provider is the in-repo ``kdl_frontier_nano`` provider with different
weights and nothing else: the layout stage, the per-region crop and
recognition passes, the retries and the entire markdown emission are
KoreaDeep's pipeline, inherited unchanged. Only the endpoint and served-model
environment variables differ. Full attribution to KoreaDeep for the pipeline
design. Serve the weights exactly as KDLAI/KDL-Frontier-Parser-nano is served:

    vllm serve aeddix-labs/aeddix-alpine-ocr --revision v0.1 \\
      --served-model-name aeddix-alpine-ocr \\
      --max-model-len 8192 --gpu-memory-utilization 0.85 \\
      --max-num-seqs 24 --trust-remote-code \\
      --limit-mm-per-prompt '{"image":1}'

Then:

    AEDDIX_ALPINE_OCR_KDL_ENDPOINT_URL=http://localhost:8000/v1 \\
    uv run parse-bench run aeddix_alpine_ocr_kdl --input_dir data ...

Config (env):
  AEDDIX_ALPINE_OCR_KDL_ENDPOINT_URL  vLLM base URL ending in /v1   (required)
  AEDDIX_ALPINE_OCR_KDL_MODEL         served model name             (default aeddix-alpine-ocr)
  All other knobs are inherited from the ``kdl_frontier_nano`` provider and keep
  their ``KDL_NANO_*`` environment variables and defaults.
"""

from __future__ import annotations

import os
from typing import Any

from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.kdl_frontier_nano import KdlFrontierNanoProvider
from parse_bench.inference.providers.registry import register_provider


@register_provider("aeddix_alpine_ocr_kdl")
class AeddixAlpineOcrKdlProvider(KdlFrontierNanoProvider):
    """aeddix-labs/aeddix-alpine-ocr: the ``kdl_frontier_nano`` provider with
    these weights. Serving requirements, every inference stage, and the
    markdown emission are inherited unchanged."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        cfg = dict(base_config or {})
        cfg["endpoint_url"] = cfg.get("endpoint_url") or os.getenv("AEDDIX_ALPINE_OCR_KDL_ENDPOINT_URL") or ""
        if not cfg["endpoint_url"]:
            raise ProviderConfigError(
                "AEDDIX_ALPINE_OCR_KDL_ENDPOINT_URL is required (vLLM OpenAI-compatible base "
                "URL ending in /v1, serving aeddix-labs/aeddix-alpine-ocr)."
            )
        cfg["model"] = cfg.get("model") or os.getenv("AEDDIX_ALPINE_OCR_KDL_MODEL") or "aeddix-alpine-ocr"
        super().__init__(provider_name, cfg)
