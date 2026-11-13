"""Anthropic ``layout_bbox_space: "pixels"`` and ``image_resolution`` config."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse._layout_utils import SYSTEM_PROMPT_LAYOUT, USER_PROMPT_LAYOUT
from parse_bench.inference.providers.parse.anthropic import AnthropicProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

_PIXEL_CONFIG = {
    "model": "claude-opus-5-5",
    "max_tokens": 32768,
    "mode": "parse_with_layout",
    "dpi": 150,
    "image_resolution": "high",
    "layout_bbox_space": "pixels",
}


def _provider(monkeypatch: pytest.MonkeyPatch, config: dict, create=None) -> AnthropicProvider:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider("anthropic", config)
    if create is not None:
        provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return provider


def test_pixel_space_prompt_states_the_uploaded_size(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def create(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return SimpleNamespace(content=[], usage=None)

    provider = _provider(monkeypatch, _PIXEL_CONFIG, create)

    provider._parse_image_with_layout(Image.new("RGB", (1240, 1600), "white"))

    system = calls[0]["system"]
    assert "absolute pixel coordinates of the uploaded image" in system
    assert "normalized 0-1000" not in system
    assert system.endswith("\n\nThe uploaded image is exactly 1240 pixels wide by 1600 pixels high.")
    # Everything but the bbox bullet is the shared layout prompt.
    assert system.startswith(SYSTEM_PROMPT_LAYOUT.split('- data-bbox="')[0])
    image_part, text_part = calls[0]["messages"][0]["content"]
    assert image_part["transformations"] == {"oversized_image": "error"}
    assert text_part["text"] == USER_PROMPT_LAYOUT


def test_default_layout_prompt_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def create(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return SimpleNamespace(content=[], usage=None)

    provider = _provider(monkeypatch, {"model": "claude-opus-5-5", "mode": "parse_with_layout"}, create)

    provider._parse_image_with_layout(Image.new("RGB", (100, 100), "white"))

    assert calls[0]["system"] == SYSTEM_PROMPT_LAYOUT
    assert "transformations" not in calls[0]["messages"][0]["content"][0]


def test_high_resolution_sets_resize_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    high = _provider(monkeypatch, _PIXEL_CONFIG)
    standard = _provider(monkeypatch, {"model": "claude-opus-5-5", "mode": "parse_with_layout"})

    assert (high._vision_max_edge, high._vision_max_tokens) == (2576, 4784)
    assert (standard._vision_max_edge, standard._vision_max_tokens) == (1568, 1568)


@pytest.mark.parametrize(
    "config",
    [
        {**_PIXEL_CONFIG, "layout_bbox_space": "gpt_xyxy_999"},
        {**_PIXEL_CONFIG, "mode": "parse_with_layout_file"},
        {**_PIXEL_CONFIG, "bbox_scale": None},
        {**_PIXEL_CONFIG, "image_resolution": "ultra"},
    ],
)
def test_invalid_config_is_refused(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    with pytest.raises(ProviderConfigError):
        _provider(monkeypatch, config)


def test_pixel_pages_are_resized_recorded_and_normalized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = _provider(monkeypatch, _PIXEL_CONFIG)
    sizes: list[tuple[int, int]] = []

    def parse(image: Image.Image) -> tuple[list[dict], str, dict[str, int]]:
        sizes.append(image.size)
        # Right half of the page, in pixels of the uploaded image.
        item = {"bbox": [image.width / 2, 0, image.width, image.height], "label": "Text", "text": "t"}
        return [item], "raw", {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(provider, "_parse_image_with_layout", parse)
    source = tmp_path / "page.png"
    Image.new("RGB", (3000, 4000), "white").save(source)
    pipeline = PipelineSpec(
        pipeline_name="anthropic_opus_5_5_parse_with_layout",
        provider_name="anthropic",
        product_type=ProductType.PARSE,
        config=_PIXEL_CONFIG,
    )
    request = InferenceRequest(example_id="page", source_file_path=str(source), product_type=ProductType.PARSE)

    raw = provider.run_inference(pipeline, request)

    width, height = sizes[0]
    assert (width, height) != (3000, 4000)
    assert max(width, height) <= 2576
    page = raw.raw_output["pages"][0]
    assert (page["width"], page["height"]) == (width, height)
    assert page["bbox_coordinate_space"] == "pixels"
    assert raw.raw_output["config"]["image_resolution"] == "high"
    assert raw.raw_output["config"]["layout_bbox_space"] == "pixels"

    now = datetime.now()
    result = provider.normalize(
        RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=ProductType.PARSE,
            raw_output=raw.raw_output,
            started_at=now,
            completed_at=now,
            latency_in_ms=0,
        )
    )
    segment = result.output.layout_pages[0].items[0].layout_segments[0]
    assert (segment.x, segment.y, segment.w, segment.h) == pytest.approx((0.5, 0.0, 0.5, 1.0), abs=1e-3)
