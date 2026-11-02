"""Unlimited-OCR 2 provider: the hand-off from the raw block string to the layout scorers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers import project_layout_predictions
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.parse.unlimitedocr2 import UnlimitedOCR2Provider
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

_RAW = (
    "<PAGE><BLOCK>page_number[480, 20, 520, 40]<FORK>38\n"
    "<BLOCK>title<LEVEL2>[100, 60, 600, 80]<FORK>Results\n"
    "<BLOCK>other[100, 100, 900, 300]<FORK>\n"
    "<BLOCK>text[100, 100, 900, 200]<FORK>First item text\n"
    "<BLOCK>cross_table[100, 400, 900, 900]<FORK><table><tr><td>a</td><td>b</td></tr></table>"
    "<PAGE><BLOCK>append_table[100, 50, 900, 300]<FORK><table><tr><td>c</td><td>d</td></tr></table>\n"
    "<BLOCK>chart[100, 400, 900, 800]<FORK>\n"
    "<BLOCK>page_number[480, 960, 520, 980]<FORK>39"
)


def test_blocks_reach_attribution_on_their_own_page_with_canonical_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches blocks landing on the wrong page or page size after the <PAGE> split, labels
    the scorer silently drops (cross_/append_ prefixes, page_number, title levels), empty
    list containers counted as Text regions, the pipeline falling back to the LlamaParse
    layout adapter (which read the normalized boxes as pixels and scored attribution 0), and
    labels reaching the evaluator's label projection in a form no mapper resolves."""

    async def fake_call_api(_self: UnlimitedOCR2Provider, pages: list[bytes]) -> dict[str, Any]:
        assert pages == [b"p1", b"p2"]  # the whole document goes out as one request
        sizes = [{"width": 1000, "height": 2000}, {"width": 1500, "height": 1000}]
        return {"raw": _RAW, "page_sizes": sizes, "status": "success"}

    monkeypatch.setattr(UnlimitedOCR2Provider, "_call_api", fake_call_api)
    monkeypatch.setattr(UnlimitedOCR2Provider, "_pdf_to_images", lambda _self, _path: [b"p1", b"p2"])
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    pipeline = get_pipeline("unlimitedocr2")
    config = {**pipeline.config, "server_url": "http://server.invalid/predict"}
    provider = UnlimitedOCR2Provider(pipeline.provider_name, config)
    request = InferenceRequest(example_id="doc", source_file_path=str(source), product_type=ProductType.PARSE)

    result = provider.normalize(provider.run_inference(pipeline, request))

    page1, page2 = result.output.layout_pages  # type: ignore[union-attr]
    assert [item.bbox.label for item in page1.items] == ["Page-header", "Section-header", "Text", "Table"]
    assert [item.bbox.label for item in page2.items] == ["Table", "Picture", "Page-footer"]
    assert (page2.page_number, page2.width, page2.height) == (2, 1500.0, 1000.0)
    # Baidu's converter merges the cross-page table fragments into one table.
    assert "<td>a</td><td>b</td></tr><tr><td>c</td>" in result.output.markdown  # type: ignore[union-attr]

    adapter = create_layout_adapter_for_result(result)
    layout_output = adapter.to_layout_output(result)
    blocks = adapter.to_attribution_blocks(layout_output, page_number=1)
    assert blocks[0].bbox_xyxy == pytest.approx([0.48, 0.02, 0.52, 0.04])
    assert blocks[0].text == "38"

    projected = project_layout_predictions(
        result, layout_output, evaluation_view="canonical", target_ontology="canonical", page_filter=1
    )
    assert [p["class_name"] for p in projected] == ["Page-header", "Section-header", "Text", "Table"]
