"""MiniCPM-OCR provider: the hand-off from the SDK server's regions to the layout scorers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers import project_layout_predictions
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.parse.minicpm_ocr import MiniCPMOCRProvider
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def test_pixel_regions_reach_the_attribution_scorer_normalized_with_pp_doclayout_v3_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches the result falling back to the LlamaParse layout adapter (it read the normalized
    boxes as pixels, so attribution found no overlapping box and scored every element 0 while
    localization passed), PP-DocLayoutV3's ``display_formula`` being dropped as an unknown
    label, and a page number not being filed as page furniture."""
    sdk_result = {
        "status": "success",
        "markdown": "## Results\n\n$$\nE=mc^2\n$$",
        "pages": [
            {
                "page_index": 0,
                "image_width": 1000,
                "image_height": 2000,
                "markdown": "## Results\n\n$$\nE=mc^2\n$$",
                "blocks": [
                    {"label": "paragraph_title", "bbox": [100, 100, 600, 200], "content": "Results", "order": 0},
                    {"label": "display_formula", "bbox": [100, 400, 900, 600], "content": "E=mc^2", "order": 1},
                    {"label": "number", "bbox": [480, 1900, 520, 1950], "content": "7", "order": 2},
                ],
            }
        ],
    }

    async def fake_predict(_self: MiniCPMOCRProvider, _file_bytes: bytes, _filename: str) -> dict[str, Any]:
        return sdk_result

    monkeypatch.setattr(MiniCPMOCRProvider, "_predict", fake_predict)
    source = tmp_path / "page.pdf"
    source.write_bytes(b"%PDF")
    pipeline = get_pipeline("minicpm_ocr_sdk")
    config = {**pipeline.config, "server_url": "http://sdk.invalid/predict"}
    provider = MiniCPMOCRProvider(pipeline.provider_name, config)
    request = InferenceRequest(example_id="page", source_file_path=str(source), product_type=ProductType.PARSE)

    result = provider.normalize(provider.run_inference(pipeline, request))

    title, formula, page_number = result.output.layout_pages[0].items  # type: ignore[union-attr]
    assert title.bbox.x == pytest.approx(0.1) and title.bbox.h == pytest.approx(0.05)
    assert formula.bbox.label == "Formula"
    assert page_number.bbox.label == "Page-footer"

    adapter = create_layout_adapter_for_result(result)
    layout_output = adapter.to_layout_output(result)
    blocks = adapter.to_attribution_blocks(layout_output, page_number=1)
    assert blocks[0].bbox_xyxy == pytest.approx([0.1, 0.05, 0.6, 0.1])

    projected = project_layout_predictions(
        result, layout_output, evaluation_view="canonical", target_ontology="canonical"
    )
    assert [p["class_name"] for p in projected] == ["Section-header", "Formula", "Page-footer"]
