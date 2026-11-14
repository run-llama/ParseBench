"""Nemotron 3.5 Parse provider: the hand-off from the SDK server's blocks to the layout scorers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers import project_layout_predictions
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.parse.nemotron_parse_3_5 import NemotronParse35Provider
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def test_layout_reaches_the_attribution_scorer_with_normalized_boxes_and_aliased_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches the pipeline falling back to the LlamaParse layout adapter (which read the
    normalized boxes as pixels, so every attribution box collapsed to ~0 and attribution
    scored 0), labels reaching the evaluator's label projection as integer IDs no mapper
    resolves (UnknownRawLayoutLabelError on every document), Chart/TOC dropped to Text as
    unknown labels, and one slightly out-of-frame coordinate (>1.0) flipping the whole page
    into pixel mode."""
    page = {
        "status": "success",
        "markdown": "# Title\n\nchart table",
        "image_width": 1000,
        "image_height": 2000,
        "blocks": [
            {"class": "Title", "bbox": [0.1, 0.05, 0.6, 0.1], "text": "Title", "content": "Title"},
            {"class": "Chart", "bbox": [0.1, 0.2, 1.02, 0.5], "text": "raw", "content": "chart table"},
            {"class": "TOC", "bbox": [0.1, 0.6, 0.9, 0.8], "text": "Contents 1", "content": "Contents 1"},
        ],
    }

    async def fake_pages(_self: NemotronParse35Provider, _pages: list[bytes]) -> dict[str, Any]:
        return {"page_results": [page], "_config": _self._config_summary()}

    monkeypatch.setattr(NemotronParse35Provider, "_run_inference_pages_async", fake_pages)
    source = tmp_path / "page.png"
    source.write_bytes(b"png")
    pipeline = get_pipeline("nemotron_parse_3_5_sdk")
    config = {**pipeline.config, "server_url": "http://sdk.invalid/predict"}
    provider = NemotronParse35Provider(pipeline.provider_name, config)
    request = InferenceRequest(example_id="page", source_file_path=str(source), product_type=ProductType.PARSE)

    result = provider.normalize(provider.run_inference(pipeline, request))

    title, chart, toc = result.output.layout_pages[0].items  # type: ignore[union-attr]
    assert title.bbox.x == pytest.approx(0.1) and title.bbox.h == pytest.approx(0.05)
    assert chart.bbox.label == "Picture" and chart.bbox.x + chart.bbox.w == pytest.approx(1.0)
    assert toc.bbox.label == "Document Index"

    adapter = create_layout_adapter_for_result(result)
    layout_output = adapter.to_layout_output(result)
    blocks = adapter.to_attribution_blocks(layout_output, page_number=1)
    assert [block.bbox_xyxy for block in blocks][0] == pytest.approx([0.1, 0.05, 0.6, 0.1])

    projected = project_layout_predictions(
        result, layout_output, evaluation_view="canonical", target_ontology="canonical"
    )
    assert [p["class_name"] for p in projected] == ["Title", "Picture", "Document Index"]
