from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse.glm_zai import GLMZaiParseProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def test_glm_5_3_flash_pricing() -> None:
    provider = object.__new__(GLMZaiParseProvider)
    provider._model = "glm-5.3-flash"

    assert provider._pricing3() == (0.15, 0.03, 0.50)
    assert provider._get_pricing() == (0.15, 0.50)


def test_glm_usage_splits_reasoning_from_visible_output() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=80,
            total_tokens=180,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        )
    )

    assert GLMZaiParseProvider._extract_usage(response) == {
        "input_tokens": 100,
        "output_tokens": 50,
        "thinking_tokens": 30,
        "total_tokens": 180,
    }


def test_glm_run_inference_initializes_inherited_bbox_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(source)
    provider = GLMZaiParseProvider(
        "glm_zai",
        {
            "api_key": "test-key",
            "model": "glm-5.3-flash",
            "mode": "parse_with_layout_file",
        },
    )
    monkeypatch.setattr(
        provider,
        "_parse_image_with_layout",
        lambda _image: (
            [],
            "",
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "thinking_tokens": 2,
                "total_tokens": 17,
            },
        ),
    )
    pipeline = PipelineSpec(
        pipeline_name="glm_5_3_flash_parse_with_layout_file",
        provider_name="glm_zai",
        product_type=ProductType.PARSE,
        config={},
    )
    request = InferenceRequest(
        example_id="glm-run",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )

    raw = provider.run_inference(pipeline, request)

    assert provider._bbox_scale == 1000
    assert raw.raw_output["bbox_scale"] == 1000
    assert raw.raw_output["num_pages"] == 1


class _StatusError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_glm_error_classification_uses_status_not_numeric_message_content() -> None:
    provider = object.__new__(GLMZaiParseProvider)

    with pytest.raises(ProviderTransientError):
        provider._raise_glm_error(_StatusError(500, "service unavailable"))

    with pytest.raises(ProviderPermanentError):
        provider._raise_glm_error(ValueError("maximum context length is 500000 tokens"))
