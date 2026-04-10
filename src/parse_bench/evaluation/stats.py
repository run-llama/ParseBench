"""Shared helper to build operational RunStat entries from an InferenceResult."""

from parse_bench.schemas.evaluation import RunStat
from parse_bench.schemas.pipeline_io import InferenceResult

# Stat keys to look for in raw_output, with their units
_RAW_OUTPUT_STATS: list[tuple[str, str]] = [
    ("credits_used", "credits"),
    ("cost_usd", "$"),
    ("cost_per_page_usd", "$/page"),
    ("input_cost_usd", "$"),
    ("tool_use_prompt_cost_usd", "$"),
    ("cached_input_cost_usd", "$"),
    ("output_and_thinking_cost_usd", "$"),
    ("cache_storage_cost_usd", "$"),
    # Token metrics (when available from provider)
    ("input_tokens", "tokens"),
    ("tool_use_prompt_tokens", "tokens"),
    ("cached_content_tokens", "tokens"),
    ("output_tokens", "tokens"),
    ("total_tokens", "tokens"),
    ("thinking_tokens", "tokens"),
    ("num_api_calls", "calls"),
    ("input_tokens_per_page", "tokens/page"),
    ("tool_use_prompt_tokens_per_page", "tokens/page"),
    ("cached_content_tokens_per_page", "tokens/page"),
    ("output_tokens_per_page", "tokens/page"),
]


def build_operational_stats(inference_result: InferenceResult) -> list[RunStat]:
    """Build operational stats from an inference result.

    Extracts latency from the dedicated field and cost-related stats
    from raw_output (pre-computed by the provider).
    """
    stats: list[RunStat] = []
    raw = inference_result.raw_output

    # Latency (from dedicated field)
    if inference_result.latency_in_ms is not None:
        stats.append(RunStat(name="latency_ms", value=float(inference_result.latency_in_ms), unit="ms"))

    # Per-page latency (computed from latency and num_pages)
    num_pages = raw.get("num_pages")
    if inference_result.latency_in_ms is not None and isinstance(num_pages, (int, float)) and num_pages > 0:
        stats.append(
            RunStat(
                name="latency_ms_per_page",
                value=float(inference_result.latency_in_ms) / float(num_pages),
                unit="ms/page",
            )
        )

    # Cost and token stats (from raw_output, pre-computed by provider)
    for key, unit in _RAW_OUTPUT_STATS:
        value = raw.get(key)
        if value is not None:
            stats.append(RunStat(name=key, value=float(value), unit=unit))

    return stats
