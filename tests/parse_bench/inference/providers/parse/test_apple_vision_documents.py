import json
import subprocess
from datetime import datetime
from unittest.mock import patch

import pytest

from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse.apple_vision_documents import AppleVisionDocumentsProvider, _document_items
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult


def test_table_order_spans_escaping_boxes_and_blank_page():
    box = {"x": 0.1, "y": 0.3, "w": 0.8, "h": 0.2}
    document = {
        "text": "Intro A&B End",
        "tables": [
            {
                "bbox": box,
                "row_count": 2,
                "column_count": 2,
                "cells": [
                    {"text": "A&B", "row": 0, "column": 0, "row_span": 2, "column_span": 1, "bbox": box},
                    {"text": "<value>", "row": 0, "column": 1, "row_span": 1, "column_span": 1, "bbox": box},
                    {"text": "2", "row": 1, "column": 1, "row_span": 1, "column_span": 1, "bbox": box},
                ],
            }
        ],
        "paragraphs": [
            {"text": "Intro", "bbox": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.1}},
            {"text": "A&B", "bbox": box},
            {"text": "End", "bbox": {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.1}},
        ],
    }
    pipeline = get_pipeline("apple_vision_documents")
    request = InferenceRequest(example_id="example", source_file_path="unused.pdf", product_type="parse")
    raw = RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type="parse",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=50,
        raw_output={
            "pages": [
                {"page_index": 0, "width": 600, "height": 800, "documents": [document], "latency_in_ms": 40},
                {"page_index": 1, "width": 600, "height": 800, "documents": [], "latency_in_ms": 10},
            ]
        },
    )
    provider = AppleVisionDocumentsProvider(pipeline.provider_name)
    result = provider.normalize(raw)
    md = result.output.pages[0].markdown
    assert md.index("Intro") < md.index("<table>") < md.index("End")
    assert md.count("A&amp;B") == 1
    assert '<td rowspan="2">A&amp;B</td>' in md
    assert "&lt;value&gt;" in md
    assert result.output.pages[1].markdown == ""
    assert result.raw_output["pages"][0]["latency_in_ms"] == 40
    layout = create_layout_adapter_for_result(result).to_layout_output(result)
    assert layout.model.value == "apple_vision_documents"
    assert layout.predictions[1].bbox == pytest.approx([60, 240, 540, 400])
    assert layout.predictions[1].label == "Table"


@pytest.mark.parametrize(
    "output", ["not json", "{}", json.dumps({"coordinate_system": "bottom_left", "documents": []})]
)
def test_invalid_cli_output_fails(output):
    provider = AppleVisionDocumentsProvider("apple_vision_documents")
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, output, "")):
        with pytest.raises(ProviderPermanentError, match="Invalid Apple Vision JSON"):
            provider._recognize("page.png")


def test_cli_timeout_and_failure_are_not_empty_success():
    provider = AppleVisionDocumentsProvider("apple_vision_documents")
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("vision", 120)):
        with pytest.raises(ProviderPermanentError, match="exceeded"):
            provider._recognize("page.png")
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "recognition failed")):
        with pytest.raises(ProviderPermanentError, match="recognition failed"):
            provider._recognize("page.png")


def test_pdf_pages_rendered_in_order_and_timed(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "two pages.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page(width=72, height=144)
        pdf.new_page(width=144, height=72)
        pdf.save(path)
    pipeline = get_pipeline("apple_vision_documents")
    provider = AppleVisionDocumentsProvider(pipeline.provider_name, {"dpi": 72})
    images = []

    def recognize(image):
        pixmap = pymupdf.Pixmap(str(image))
        images.append((pixmap.width, pixmap.height))
        return {"documents": [], "recognition_latency_ms": 0, "coordinate_system": "normalized_top_left"}

    request = InferenceRequest(example_id="two", source_file_path=str(path), product_type="parse")
    with patch("platform.system", return_value="Darwin"), patch("platform.mac_ver", return_value=("26.0", (), "")):
        with patch.object(provider, "_recognize", side_effect=recognize):
            result = provider.run_inference_normalized(pipeline, request)
    assert images == [(72, 144), (144, 72)]
    assert [p.page_index for p in result.output.pages] == [0, 1]
    for page in result.raw_output["pages"]:
        assert page["latency_in_ms"] >= page["render_latency_ms"] >= 0


def test_list_markers_are_preserved_without_duplicate_paragraphs():
    box = {"x": 0.1, "y": 0.2, "w": 0.8, "h": 0.1}
    document = {
        "text": "3. Third",
        "tables": [],
        "paragraphs": [{"text": "Third", "bbox": box}],
        "lists": [{"bbox": box, "items": [{"text": "Third", "marker": "3.", "bbox": box}]}],
    }
    items = _document_items(document)
    assert len(items) == 1
    assert items[0].md == "3. Third"
