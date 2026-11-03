"""OvisOCR3 provider: the hand-off from the structured JSON answer to the markdown and layout scorers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers import project_layout_predictions
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.parse.ovisocr3 import CHART_PROMPT, OvisOcr3Provider
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

_REGIONS = [
    {"bbox": [100, 50, 900, 80], "category": "paragraph_title", "content": "## Results"},
    {"bbox": [100, 100, 900, 200], "category": "text", "content": 'Rates "rose" sharply'},
    {"bbox": [100, 300, 900, 600], "category": "chart", "content": '<img src="images/bbox_100_300_900_600.jpg" />'},
    {"bbox": [480, 960, 520, 980], "category": "number", "content": "38"},
]
# The answer is cut off by max_tokens inside a fifth region.
_STRUCTURED = json.dumps(_REGIONS)[:-1] + ', {"bbox": [100, 700, 900, 800], "category": "text", "content": "Cut of'
_CHART = '{"title":"None","source":"None","x_title":"None","y_title":"None","values":{"2023":"4.1%","2024":"5.2%"}}'


def test_truncated_page_keeps_regions_and_chart_values_reach_markdown_and_attribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a truncated structured answer dropping the whole page instead of only the cut
    region, chart placeholders reaching the scorer instead of the chart_parsing table, labels
    the scorer silently drops (paragraph_title, chart, number), the pipeline falling back
    to the LlamaParse layout adapter (which read the normalized boxes as pixels and scored
    attribution 0), and labels reaching the evaluator's label projection in a form no mapper
    resolves."""

    buf = io.BytesIO()
    Image.new("RGB", (1000, 2000), "white").save(buf, format="PNG")
    page_png = buf.getvalue()

    async def fake_call_api(_self: OvisOcr3Provider, _session: Any, _image: bytes, prompt: str) -> dict[str, Any]:
        content = _CHART if prompt == CHART_PROMPT else _STRUCTURED
        return {"content": content, "finish_reason": "stop" if prompt == CHART_PROMPT else "length"}

    monkeypatch.setattr(OvisOcr3Provider, "_call_api", fake_call_api)
    monkeypatch.setattr(OvisOcr3Provider, "_pdf_to_images", lambda _self, _path: [page_png])
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    pipeline = get_pipeline("ovisocr3_vllm")
    config = {**pipeline.config, "server_url": "http://server.invalid"}
    provider = OvisOcr3Provider(pipeline.provider_name, config)
    request = InferenceRequest(example_id="doc", source_file_path=str(source), product_type=ProductType.PARSE)

    result = provider.normalize(provider.run_inference(pipeline, request))

    (page,) = result.output.layout_pages  # type: ignore[union-attr]
    assert [item.bbox.label for item in page.items] == ["Section-header", "Text", "Picture", "Page-footer"]
    markdown = result.output.markdown  # type: ignore[union-attr]
    assert markdown.startswith('## Results\n\nRates "rose" sharply')
    assert "<td>2024</td><td>5.2%</td>" in markdown
    assert "images/bbox_" not in markdown
    assert "Cut of" not in markdown

    adapter = create_layout_adapter_for_result(result)
    layout_output = adapter.to_layout_output(result)
    blocks = adapter.to_attribution_blocks(layout_output, page_number=1)
    assert blocks[1].bbox_xyxy == pytest.approx([0.1, 0.1, 0.9, 0.2])
    assert blocks[1].text == 'Rates "rose" sharply'

    projected = project_layout_predictions(
        result, layout_output, evaluation_view="canonical", target_ontology="canonical", page_filter=1
    )
    assert [p["class_name"] for p in projected] == ["Section-header", "Text", "Picture", "Page-footer"]
