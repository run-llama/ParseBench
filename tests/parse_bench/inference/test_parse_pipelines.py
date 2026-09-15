"""Tests for the registered parse pipelines."""

from __future__ import annotations

import re

import pytest

from parse_bench.inference.pipelines import get_pipeline, list_pipelines
from parse_bench.schemas.product import ProductType


def test_amazon_nova_with_layout_pipeline_enables_layout_mode() -> None:
    spec = get_pipeline("amazon_nova_2_lite_parse_with_layout")
    assert spec.config["mode"] == "parse_with_layout"


@pytest.mark.parametrize(
    "pipeline_name, provider_name",
    [
        ("reducto_nonagentic_change_tracking", "reducto"),
        ("reducto_agentic_table", "reducto"),
        ("reducto_agentic_chart", "reducto"),
        ("openai_gpt_5_4_reasoning_none_parse", "openai"),
        ("openai_gpt_5_4_reasoning_none_parse_file", "openai"),
        ("anthropic_sonnet_5_parse_with_layout", "anthropic"),
        ("google_gemini_3_6_flash_parse_with_layout", "google"),
        ("google_gemini_3_6_flash_no_thinking_parse_with_layout", "google"),
        ("google_gemini_3_7_flash_thinking_high_parse_with_layout_file", "google"),
        ("google_gemini_3_1_flash_lite_thinking_high_parse_with_layout_file", "google"),
        ("mistral_ocr_4_1", "mistral_ocr"),
        ("mistral_ocr_4_1_annotation", "mistral_ocr"),
        ("nemotron_omni_30b_vllm", "nemotron_omni"),
        ("qwen3_8_flash_next_parse_with_layout", "qwen3_8"),
        ("qwen3_8_flash_next_thinking_parse_with_layout", "qwen3_8"),
    ],
)
def test_ported_pipelines_are_registered(pipeline_name: str, provider_name: str) -> None:
    spec = get_pipeline(pipeline_name)
    assert spec.provider_name == provider_name
    assert spec.product_type == ProductType.PARSE


def test_reducto_variants_differ_only_in_documented_keys() -> None:
    base = get_pipeline("reducto").config
    assert base["agentic"] is False
    change_tracking = get_pipeline("reducto_nonagentic_change_tracking").config
    assert change_tracking == {**base, "formatting_include": ["change_tracking"]}
    table = get_pipeline("reducto_agentic_table").config
    assert table["agentic"] is True and table["agentic_scopes"] == ["text", "table"]
    chart = get_pipeline("reducto_agentic_chart").config
    assert chart["advanced_chart_agent"] is True and chart["agentic_scopes"] == ["text", "table", "figure"]


def test_qwen38_flash_next_pipelines_differ_only_by_thinking() -> None:
    off = get_pipeline("qwen3_8_flash_next_parse_with_layout").config
    on = get_pipeline("qwen3_8_flash_next_thinking_parse_with_layout").config
    assert off["enable_thinking"] is False
    assert on["enable_thinking"] is True
    assert {k: v for k, v in off.items() if k != "enable_thinking"} == {
        k: v for k, v in on.items() if k != "enable_thinking"
    }
    assert off["server_url_env"] == "QWEN3_8_FLASH_NEXT_SERVER_URL"


def test_no_internal_only_pipeline_names_are_registered() -> None:
    # `llamaparse_agentic_granular_bboxes_staging` is a ParseBench-only pipeline,
    # so `_staging` is deliberately not in this list.
    internal = re.compile(r"^ours_|_dev$|_local$|mock")
    assert [name for name in list_pipelines() if internal.search(name)] == []


def test_self_hosted_pipelines_do_not_ship_internal_endpoints() -> None:
    for name in list_pipelines():
        config = get_pipeline(name).config
        for key in ("server_url", "endpoint_url"):
            value = config.get(key)
            if isinstance(value, str):
                assert "modal.run" not in value, f"{name}.{key} points at an internal deployment"
