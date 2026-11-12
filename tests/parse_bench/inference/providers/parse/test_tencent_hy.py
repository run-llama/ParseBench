from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from parse_bench.evaluation.layout_adapters.adapters import TencentHYLayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.parse.tencent_hy import TencentHYParseProvider
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _run_hy4_page(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str) -> InferenceResult:
    """Run the registered hy4 pipeline on one 1000x2000 page against a fake TokenHub response.

    Usage: 1000 prompt tokens (400 cached) and 500 completion tokens (300 of them reasoning).
    """
    monkeypatch.setenv("TENCENT_TOKENHUB_API_KEY", "sk-test")
    pipeline = get_pipeline("hy4_parse_with_layout")
    provider = TencentHYParseProvider(pipeline.provider_name, pipeline.config)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            prompt_tokens_details=SimpleNamespace(cached_tokens=400),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=300),
        ),
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))
    monkeypatch.setattr(provider, "_client", fake_client)
    png = tmp_path / "page.png"
    Image.new("RGB", (1000, 2000), "white").save(png)
    request = InferenceRequest(example_id="page", source_file_path=str(png), product_type=ProductType.PARSE)
    return provider.normalize(provider.run_inference(pipeline, request))


def test_hy4_uses_tokenhub_endpoint_key_and_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENCENT_TOKENHUB_API_KEY", "sk-test")
    pipeline = get_pipeline("hy4_parse_with_layout")
    provider = TencentHYParseProvider(pipeline.provider_name, pipeline.config)

    assert pipeline.provider_name == "tencent_hy"
    assert provider._base_url == "https://tokenhub-intl.tencentcloudmaas.com/v1"
    assert provider._layout_request_kwargs(Image.new("RGB", (8, 8), "white"))["reasoning_effort"] == "high"


def test_hy4_cost_bills_reasoning_once_and_credits_cached_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reasoning tokens sit inside completion_tokens (billed once at the output rate) and cached
    prompt tokens are credited down to the cached-input rate."""
    result = _run_hy4_page(monkeypatch, tmp_path, '<div data-bbox="[0,0,10,10]" data-label="Text">Hi</div>')

    expected = (600 * 0.417 + 400 * 0.021 + 500 * 1.2505) / 1_000_000
    assert result.raw_output["cost_usd"] == pytest.approx(expected)


def test_hy4_layout_resolves_to_its_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """hy4 results must resolve to the Tencent adapter; the default adapter finds no layout blocks."""
    content = '<div data-bbox="[100, 100, 900, 200]" data-label="Text">Rates rose sharply</div>'
    result = _run_hy4_page(monkeypatch, tmp_path, content)

    assert TencentHYLayoutAdapter.matches(result)
    adapter = create_layout_adapter_for_result(result)
    assert isinstance(adapter, TencentHYLayoutAdapter)
    (block,) = adapter.to_attribution_blocks(adapter.to_layout_output(result), page_number=1)
    assert block.bbox_xyxy == pytest.approx([0.1, 0.1, 0.9, 0.2])
    assert block.text == "Rates rose sharply"
