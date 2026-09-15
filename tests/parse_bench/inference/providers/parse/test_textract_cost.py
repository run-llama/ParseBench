"""Textract provider: per-page cost accounting and selection-mark layout items."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from parse_bench.inference.providers.parse import textract as textract_module
from parse_bench.inference.providers.parse.textract import (
    _TEXTRACT_COST_PER_PAGE_USD,
    TEXTRACT_SELECTION_LABEL_MAP,
    TextractProvider,
    _build_layout_pages,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _provider(detect_tables: bool, detect_forms: bool) -> TextractProvider:
    provider = object.__new__(TextractProvider)
    provider._provider_name = "textract"
    provider._base_config = {}
    provider._output_tables_as_html = True
    provider._detect_tables = detect_tables
    provider._detect_forms = detect_forms
    return provider


def _run(provider: TextractProvider, tmp_path: Path, num_pages: int, monkeypatch: pytest.MonkeyPatch):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        provider,
        "_analyze_multipage_document",
        lambda _path: {"DocumentMetadata": {"Pages": num_pages}, "Blocks": []},
    )
    pipeline = PipelineSpec(pipeline_name="aws_textract", provider_name="textract", product_type=ProductType.PARSE)
    request = InferenceRequest(example_id="ex", source_file_path=str(pdf), product_type=ProductType.PARSE)
    return provider.run_inference(pipeline, request)


def test_cost_table_layout_only_is_cheapest_and_forms_is_roughly_5x() -> None:
    assert _TEXTRACT_COST_PER_PAGE_USD[(False, False)] == 0.004
    assert _TEXTRACT_COST_PER_PAGE_USD[(True, False)] == 0.015
    assert _TEXTRACT_COST_PER_PAGE_USD[(False, True)] == 0.050
    assert _TEXTRACT_COST_PER_PAGE_USD[(True, True)] == 0.065


@pytest.mark.parametrize(
    ("detect_tables", "detect_forms", "pages"),
    [(True, False, 3), (False, False, 1), (True, True, 4)],
)
def test_run_inference_writes_cost_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detect_tables: bool, detect_forms: bool, pages: int
) -> None:
    provider = _provider(detect_tables, detect_forms)
    result = _run(provider, tmp_path, pages, monkeypatch)
    per_page = _TEXTRACT_COST_PER_PAGE_USD[(detect_tables, detect_forms)]
    assert result.raw_output["num_pages"] == pages
    assert result.raw_output["cost_per_page_usd"] == per_page
    assert result.raw_output["cost_usd"] == pytest.approx(per_page * pages)
    assert isinstance(result.started_at, datetime)


def test_run_inference_without_page_count_leaves_cost_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(_provider(True, False), tmp_path, 0, monkeypatch)
    assert "cost_usd" not in result.raw_output
    assert "cost_per_page_usd" not in result.raw_output


def test_selection_elements_become_canonical_checkbox_items() -> None:
    blocks = [
        {
            "BlockType": "SELECTION_ELEMENT",
            "SelectionStatus": "SELECTED",
            "Page": 2,
            "Confidence": 90.0,
            "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.2, "Width": 0.05, "Height": 0.04}},
        },
        {
            "BlockType": "SELECTION_ELEMENT",
            "SelectionStatus": "NOT_SELECTED",
            "Page": 2,
            "Confidence": 80.0,
            "Geometry": {"BoundingBox": {"Left": 0.3, "Top": 0.2, "Width": 0.05, "Height": 0.04}},
        },
        {"BlockType": "SELECTION_ELEMENT", "SelectionStatus": "UNKNOWN", "Page": 2},
    ]
    pages = _build_layout_pages(blocks)
    assert [p.page_number for p in pages] == [2]
    labels = [item.bbox.label for item in pages[0].items]
    assert labels == [TEXTRACT_SELECTION_LABEL_MAP["SELECTED"], TEXTRACT_SELECTION_LABEL_MAP["NOT_SELECTED"]]
    assert pages[0].items[0].bbox.confidence == pytest.approx(0.9)


def test_module_has_no_granular_layers_dependency() -> None:
    assert not hasattr(textract_module, "attach_granular_layers")
