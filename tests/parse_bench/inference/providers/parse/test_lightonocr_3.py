"""Coverage for the LightOnOCR-3 grounding provider and its ``lightonocr_3_vllm_parse`` pipeline."""

from __future__ import annotations

import importlib
from datetime import datetime

from parse_bench.inference.pipelines import get_pipeline
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

# The module is named after the provider, which is not a valid identifier, so import it by name.
_module = importlib.import_module("parse_bench.inference.providers.parse.lightonocr-3")
LightOnOcr3Provider = _module.LightOnOcr3Provider

RAW = (
    "![title](10,10,500,60) # Quarterly report\n\n"
    "![paragraph](10,70,990,120) Revenue grew by 5%.\n\n"
    "![formula](10,130,400,180) $$E = mc^2$$\n\n"
    "![page_number](480,950,520,980) 7"
)


def _provider() -> LightOnOcr3Provider:
    return LightOnOcr3Provider("lightonocr-3", {"server_url": "http://localhost:8000"})


def _raw_result(raw_output: dict) -> RawInferenceResult:
    now = datetime.now()
    request = InferenceRequest(
        example_id="lightonocr-3-test",
        source_file_path="doc.pdf",
        product_type=ProductType.PARSE,
    )
    pipeline = PipelineSpec(
        pipeline_name="lightonocr_3_vllm_parse", provider_name="lightonocr-3", product_type=ProductType.PARSE
    )
    return RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )


def test_parse_blocks_splits_on_markers() -> None:
    blocks = LightOnOcr3Provider.parse_blocks(RAW)

    assert [b["label"] for b in blocks] == ["title", "paragraph", "formula", "page_number"]
    assert blocks[0]["bbox"] == [10, 10, 500, 60]
    assert blocks[0]["text"] == "# Quarterly report"
    assert blocks[1]["text"] == "Revenue grew by 5%."


def test_parse_blocks_returns_nothing_without_markers() -> None:
    assert LightOnOcr3Provider.parse_blocks("plain text with no markers") == []


def test_normalize_builds_markdown_and_layout() -> None:
    layout_items = [
        {
            "bbox": b["bbox"],
            "category": "Title" if b["label"] == "title" else "Text",
            "text": b["text"],
            "raw_label": b["label"],
        }
        for b in LightOnOcr3Provider.parse_blocks(RAW)
    ]
    raw_output = {
        "pages": [{"page_index": 0, "width": 1000, "height": 1400, "raw_response": RAW, "layout_items": layout_items}]
    }

    result = _provider().normalize(_raw_result(raw_output))

    assert result.output.markdown.startswith("# Quarterly report")
    assert "$$\nE = mc^2\n$$" in result.output.markdown
    assert len(result.output.layout_pages) == 1
    # The page number is a dedicated page field, not body text.
    assert result.output.layout_pages[0].printed_page_number == "7"


def test_normalize_repairs_escaped_dollars() -> None:
    layout_items = [
        {"bbox": [0, 0, 500, 100], "category": "Text", "text": "The energy \\$x^2\\$ term.", "raw_label": "paragraph"}
    ]
    raw_output = {
        "pages": [{"page_index": 0, "width": 1000, "height": 1400, "raw_response": "", "layout_items": layout_items}]
    }

    result = _provider().normalize(_raw_result(raw_output))

    assert result.output.markdown == "The energy $x^2$ term."


def test_pipeline_is_registered_with_provider_settings() -> None:
    pipeline = get_pipeline("lightonocr_3_vllm_parse")

    assert pipeline.provider_name == "lightonocr-3"
    assert pipeline.config["dpi"] == 400
    assert pipeline.config["max_pixels"] == 5_000_000
    assert pipeline.config["prompt"] == "grounding"
    assert pipeline.config["server_url_env"] == "LIGHTONOCR_3_SERVER_URL"


def test_layout_adapter_is_registered_for_provider_key() -> None:
    from parse_bench.evaluation.layout_adapters.adapters import QwenLayoutAdapter
    from parse_bench.evaluation.layout_adapters.registry import _lookup_registration_by_key

    registration = _lookup_registration_by_key("lightonocr-3")

    assert registration is not None
    assert registration.adapter_cls is QwenLayoutAdapter
