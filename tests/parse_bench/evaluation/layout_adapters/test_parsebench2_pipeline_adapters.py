"""ParseBench 2 leaderboard pipelines resolve their published outputs to the provider's layout adapter."""

from __future__ import annotations

from datetime import datetime

import pytest

from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from parse_bench.evaluation.layout_adapters.adapters import (
    AnthropicLayoutAdapter,
    GoogleLayoutAdapter,
    InfinityParser2LayoutAdapter,
    LightOnOcr3LayoutAdapter,
    MetaLayoutAdapter,
    OpenAILayoutAdapter,
)
from parse_bench.schemas.layout_detection_output import QWEN3VL_STR_TO_LABEL, LayoutDetectionModel
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _result(pipeline_name: str, raw_output: dict, label: str = "Section-header") -> InferenceResult:
    segment = LayoutSegmentIR(x=0.1, y=0.2, w=0.3, h=0.1, label=label)
    page = ParseLayoutPageIR(
        page_number=1,
        width=1000,
        height=2000,
        items=[LayoutItemIR(type="text", value="Results", bbox=segment, layout_segments=[segment])],
    )
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(example_id="layout/doc", source_file_path="", product_type=ProductType.PARSE),
        pipeline_name=pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        output=ParseOutput(
            task_type="parse",
            example_id="layout/doc",
            pipeline_name=pipeline_name,
            pages=[PageIR(page_index=0, markdown="Results")],
            layout_pages=[page],
            markdown="Results",
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


@pytest.mark.parametrize(
    ("pipeline_name", "adapter"),
    [
        ("anthropic_opus_5_5_parse_with_layout", AnthropicLayoutAdapter),
        ("anthropic_haiku_5_5_parse_with_layout", AnthropicLayoutAdapter),
        ("openai_gpt_5_6_terra_reasoning_low_parse_with_layout", OpenAILayoutAdapter),
        ("openai_gpt_5_6_luna_reasoning_low_parse_with_layout", OpenAILayoutAdapter),
        ("openai_gpt_6_luna_reasoning_low_parse_with_layout", OpenAILayoutAdapter),
        ("gemini_4_argon_thinking_low_parse_with_layout_file", GoogleLayoutAdapter),
        ("muse_ember_1_0_reasoning_low_parse_with_layout_file", MetaLayoutAdapter),
        ("lightonocr3_4b_vllm", LightOnOcr3LayoutAdapter),
        ("lightonocr3_1b_vllm", LightOnOcr3LayoutAdapter),
        ("lightonocr3_0_8b_vllm", LightOnOcr3LayoutAdapter),
        ("infinity_parser2_pro_deep", InfinityParser2LayoutAdapter),
        ("infinity_parser2_flash_deep", InfinityParser2LayoutAdapter),
    ],
)
def test_pipeline_resolves_to_provider_adapter(pipeline_name: str, adapter: type) -> None:
    # An empty raw_output matches no content matcher, so only the registered provider key can pick the adapter.
    assert type(create_layout_adapter_for_result(_result(pipeline_name, {}))) is adapter


@pytest.mark.parametrize(
    ("pipeline_name", "model"),
    [
        ("muse_ember_1_0_reasoning_low_parse_with_layout_file", LayoutDetectionModel.OPENAI_LAYOUT),
        ("lightonocr3_4b_vllm", LayoutDetectionModel.OPENAI_COMPATIBLE_VLM_LAYOUT),
    ],
)
def test_adapter_writes_qwen3vl_index_labels(pipeline_name: str, model: LayoutDetectionModel) -> None:
    result = _result(pipeline_name, {})
    layout = create_layout_adapter_for_result(result).to_layout_output(result)

    assert layout.model == model
    [prediction] = layout.predictions
    assert prediction.label == str(int(QWEN3VL_STR_TO_LABEL["section_header"]))
    assert prediction.bbox == pytest.approx([100.0, 400.0, 400.0, 600.0])


def test_lightonocr3_matcher_reads_served_model() -> None:
    assert LightOnOcr3LayoutAdapter.matches(_result("unregistered", {"_config": {"model": "lightonocr-3-4b"}}))
    assert not LightOnOcr3LayoutAdapter.matches(_result("unregistered", {"_config": {"model": "loocr-grounding"}}))


def test_lightonocr3_collapses_labels_outside_core11() -> None:
    """``Code`` has no Qwen3-VL class; passed through, the label mapper failed the whole page."""
    result = _result("lightonocr3_4b_vllm", {}, label="Code")
    [prediction] = create_layout_adapter_for_result(result).to_layout_output(result).predictions
    assert prediction.label == str(int(QWEN3VL_STR_TO_LABEL["text"]))
