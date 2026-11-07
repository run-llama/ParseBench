"""lmkit provider: one request per document carries the structure and its Markdown, every element
becomes exactly one layout box with a canonical label (running heads first, running feet last),
HTTP statuses map onto the error taxonomy, a long parse is followed through its job, and the
layout adapter reads the boxes as they are."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime

import pytest

from parse_bench.evaluation.layout_adapters.adapters import LMKitLayoutAdapter
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse import lmkit
from parse_bench.inference.providers.parse.lmkit import LMKitProvider, project_page
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

_PAGE = {
    "page_index": 0,
    "page_number": 1,
    "width": 600,
    "height": 800,
    "elements": [
        {
            "category": "title",
            "reading_index": 0,
            "confidence": 1,
            "bbox": [60, 80, 540, 120],
            "content": {"type": "text", "format": "markdown", "text": "# Revenue"},
        },
        {
            "category": "header",
            "reading_index": 1,
            "confidence": 0.9,
            "bbox": [60, 10, 300, 30],
            "content": {"type": "text", "format": "markdown", "text": "Annual report"},
        },
        {
            "category": "table",
            "reading_index": 2,
            "confidence": 0.95,
            "bbox": [60, 140, 540, 400],
            "content": {"type": "table", "format": "html", "text": "<table><tr><td>1</td></tr></table>"},
        },
        {
            "category": "figure",
            "reading_index": 3,
            "confidence": 0.8,
            "bbox": [60, 420, 540, 700],
            "content": {"type": "figure", "text": "A bar chart"},
            "printed_words": ["2024", "Sales"],
        },
        {
            "category": "table_caption",
            "reading_index": 4,
            "confidence": 1,
            "bbox": [60, 400, 540, 415],
            "content": {"type": "text", "format": "markdown", "text": "Table 1"},
        },
        {
            "category": "footer",
            "reading_index": 5,
            "confidence": 1,
            "bbox": [280, 770, 320, 790],
            "content": {"type": "text", "format": "markdown", "text": "12"},
        },
        {
            "category": "text",
            "reading_index": 6,
            "confidence": 1,
            "bbox": [60, 710, 60, 760],
            "content": {"type": "text", "format": "markdown", "text": "zero width"},
        },
        {
            "category": "unknown",
            "reading_index": 7,
            "confidence": 0.5,
            "bbox": [60, 720, 200, 740],
            "content": {"type": "repeated_text", "text": "CONFIDENTIAL"},
        },
    ],
}
_ELEMENT_MARKDOWN = [
    "# Revenue",
    "Annual report",
    "<table><tr><td>1</td></tr></table>",
    "![A bar chart]()",
    "Table 1",
    "12",
    "zero width",
    "",
]
_RESPONSE = {
    "document": {"schema_version": "1.0", "pages": [_PAGE]},
    "markdown": "# Revenue\n\n<table><tr><td>1</td></tr></table>",
    "page_markdown": ["# Revenue\n\n<table><tr><td>1</td></tr></table>"],
    "element_markdown": [_ELEMENT_MARKDOWN],
    "output_format": "Json",
    "effort": "High",
    "total_pages": 1,
    "processed_pages": 1,
    "element_count": 8,
    "elapsed_seconds": 3.1,
}


class _Response(io.BytesIO):
    def __init__(self, status: int, payload: dict) -> None:
        super().__init__(json.dumps(payload).encode())
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _request(tmp_path) -> InferenceRequest:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4")
    return InferenceRequest(example_id="doc", source_file_path=str(source), product_type=ProductType.PARSE)


def _pipeline() -> PipelineSpec:
    return PipelineSpec(pipeline_name="lmkit_high", provider_name="lmkit", product_type=ProductType.PARSE, config={})


def test_every_element_is_one_box_with_its_canonical_label_heads_first_feet_last():
    page = project_page(_PAGE, "md", _ELEMENT_MARKDOWN)

    labels = [item.bbox.label for item in page.items]
    assert labels == ["Page-header", "Section-header", "Table", "Picture", "Caption", "Text", "Page-footer"]
    assert all(len(item.layout_segments) == 1 for item in page.items), "one box per element, nothing nested"
    assert page.page_header_markdown == "Annual report"
    assert page.page_footer_markdown == "12"


def test_boxes_are_normalized_and_values_follow_the_content():
    page = project_page(_PAGE, "md", _ELEMENT_MARKDOWN)
    by_label = {item.bbox.label: item for item in page.items}

    table = by_label["Table"]
    assert table.type == "table" and table.value.startswith("<table>")
    assert (table.bbox.x, table.bbox.y, table.bbox.w, table.bbox.h) == pytest.approx((0.1, 0.175, 0.8, 0.325))
    assert by_label["Picture"].type == "image" and by_label["Picture"].value == "2024 Sales"
    assert by_label["Text"].value == "CONFIDENTIAL", "a layout-only element carries its own words"


def _row_page() -> dict:
    """A label and its value read as two elements that the parser joined into one region, beside a
    paragraph standing alone, every region with the box the parser gave it."""
    return {
        "page_index": 0,
        "page_number": 1,
        "width": 600,
        "height": 800,
        "elements": [
            {
                "category": "text",
                "reading_index": 0,
                "confidence": 1,
                "bbox": [60, 100, 160, 110],
                "content": {"type": "text", "format": "markdown", "text": "Net sales:"},
            },
            {
                "category": "text",
                "reading_index": 1,
                "confidence": 0.9,
                "bbox": [300, 100, 400, 110],
                "content": {"type": "text", "format": "markdown", "text": "EUR 4.3 billion"},
            },
            {
                "category": "text",
                "reading_index": 2,
                "confidence": 1,
                "bbox": [60, 140, 540, 170],
                "content": {"type": "text", "format": "markdown", "text": "A paragraph."},
            },
        ],
        "regions": [
            {"category": "text", "bbox": [60, 98, 400, 112], "members": [0, 1], "text": "Net sales: EUR 4.3 billion"},
            {"category": "text", "bbox": [60, 138, 540, 172], "members": [2]},
        ],
    }


def test_each_grounding_region_is_one_box_over_its_members():
    page = project_page(_row_page(), "md", ["Net sales:", "EUR 4.3 billion", "A paragraph."])

    assert [item.value for item in page.items] == ["Net sales:\nEUR 4.3 billion", "A paragraph."]
    assert [item.bbox.label for item in page.items] == ["Text", "Text"]
    joined = page.items[0].bbox
    assert (joined.x, joined.y, joined.w, joined.h) == pytest.approx((0.1, 98 / 800, 340 / 600, 14 / 800))
    assert joined.confidence == pytest.approx(0.9)


def test_without_regions_every_element_is_its_own_box():
    page = _row_page()
    del page["regions"]

    items = project_page(page, "md", ["Net sales:", "EUR 4.3 billion", "A paragraph."]).items

    assert len(items) == 3
    assert (items[0].bbox.y, items[0].bbox.h) == pytest.approx((100 / 800, 10 / 800))


def _unit_pages() -> tuple[dict, dict]:
    """The same parse as two server generations send it: the grounding units (a list as a whole,
    a drawn icon) as layout-only elements, then as regions listing no element."""
    legacy = _row_page()
    legacy["elements"] += [
        {
            "category": "text",
            "reading_index": 3,
            "confidence": 0.9,
            "bbox": [60, 98, 540, 172],
            "content": {"type": "repeated_text", "text": "Net sales: EUR 4.3 billion A paragraph."},
        },
        {
            "category": "figure",
            "reading_index": 4,
            "confidence": 0.75,
            "bbox": [500, 20, 540, 60],
            "content": {"type": "repeated_text", "text": ""},
        },
    ]
    legacy["regions"] += [
        {"category": "text", "bbox": [60, 98, 540, 172], "members": [3]},
        {"category": "figure", "bbox": [500, 20, 540, 60], "members": [4]},
    ]
    current = _row_page()
    current["regions"] += [
        {
            "category": "text",
            "bbox": [60, 98, 540, 172],
            "members": [],
            "text": "Net sales: EUR 4.3 billion A paragraph.",
            "confidence": 0.9,
        },
        {"category": "figure", "bbox": [500, 20, 540, 60], "members": [], "text": "", "confidence": 0.75},
    ]
    return legacy, current


def test_grounding_units_sent_as_regions_are_boxes_with_their_own_words():
    legacy, current = _unit_pages()
    renderings = ["Net sales:", "EUR 4.3 billion", "A paragraph."]

    items = project_page(current, "md", renderings).items

    assert [item.value for item in items] == [
        "Net sales:\nEUR 4.3 billion",
        "A paragraph.",
        "Net sales: EUR 4.3 billion A paragraph.",
        "",
    ]
    assert [(item.type, item.bbox.label) for item in items[2:]] == [("text", "Text"), ("image", "Picture")]
    assert items[3].bbox.confidence == pytest.approx(0.75)
    old = project_page(legacy, "md", renderings + ["", ""]).items
    assert [item.model_dump() for item in items] == [item.model_dump() for item in old], (
        "both server generations read the same"
    )


def test_a_region_listing_no_element_and_no_words_is_not_a_box():
    page = _row_page()
    page["regions"].append({"category": "text", "bbox": [60, 98, 540, 172], "members": []})

    assert len(project_page(page, "md", ["Net sales:", "EUR 4.3 billion", "A paragraph."]).items) == 2


def test_misaligned_renderings_are_refused():
    with pytest.raises(ValueError):
        project_page(_PAGE, "md", _ELEMENT_MARKDOWN[:-1])


def test_one_request_returns_structure_and_markdown(monkeypatch, tmp_path):
    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        sent["auth"] = request.get_header("Authorization")
        return _Response(200, _RESPONSE)

    monkeypatch.setattr(lmkit.urllib.request, "urlopen", fake_urlopen)
    provider = LMKitProvider("lmkit", {"effort": "High", "base_url": "http://server:5189"})
    raw = provider.run_inference(_pipeline(), _request(tmp_path))

    assert sent["url"] == "http://server:5189/lmkit/v1/document-parsing"
    assert sent["body"]["effort"] == "High"
    assert sent["body"]["output_format"] == "Json" and sent["body"]["include_markdown"] is True
    assert sent["auth"] is None, "no key is needed by default"

    result = provider.normalize(raw)
    assert result.output.markdown == _RESPONSE["markdown"]
    assert [p.markdown for p in result.output.pages] == _RESPONSE["page_markdown"]
    assert len(result.output.layout_pages[0].items) == 7


def test_a_long_parse_is_followed_through_its_job(monkeypatch, tmp_path):
    calls = []
    replies = [
        _Response(202, {"job_id": "j1"}),
        _Response(200, {"status": "Processing"}),
        _Response(200, {"status": "Completed", "result": _RESPONSE}),
    ]

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return replies.pop(0)

    monkeypatch.setattr(lmkit.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(lmkit.time, "sleep", lambda seconds: None)
    raw = LMKitProvider("lmkit", {"effort": "Low"}).run_inference(_pipeline(), _request(tmp_path))

    assert calls[1:] == ["http://localhost:5189/lmkit/v1/jobs/j1"] * 2
    assert raw.raw_output["provider"] == "lmkit"


@pytest.mark.parametrize(
    ("status", "error"),
    [(400, ProviderPermanentError), (503, ProviderRateLimitError), (500, ProviderTransientError)],
)
def test_http_statuses_map_onto_the_error_taxonomy(monkeypatch, tmp_path, status, error):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, status, "x", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(lmkit.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(error):
        LMKitProvider("lmkit", {}).run_inference(_pipeline(), _request(tmp_path))


def test_an_unreachable_server_is_left_to_the_runner(monkeypatch, tmp_path):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(lmkit.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ProviderTransientError):
        LMKitProvider("lmkit", {}).run_inference(_pipeline(), _request(tmp_path))


def test_a_server_without_include_markdown_is_refused_clearly(monkeypatch, tmp_path):
    legacy = {key: value for key, value in _RESPONSE.items() if not key.endswith("markdown")}
    monkeypatch.setattr(lmkit.urllib.request, "urlopen", lambda request, timeout: _Response(200, legacy))
    with pytest.raises(ProviderPermanentError, match="2026.10.4"):
        LMKitProvider("lmkit", {}).run_inference(_pipeline(), _request(tmp_path))


def test_an_unknown_effort_is_a_config_error():
    with pytest.raises(ProviderConfigError):
        LMKitProvider("lmkit", {"effort": "Extreme"})


def test_the_layout_adapter_reads_one_prediction_per_box(tmp_path):
    provider = LMKitProvider("lmkit", {})
    raw = RawInferenceResult(
        request=_request(tmp_path),
        pipeline=_pipeline(),
        pipeline_name="lmkit_high",
        product_type=ProductType.PARSE,
        raw_output={"provider": "lmkit", **_RESPONSE},
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=1,
    )
    result = provider.normalize(raw)

    assert LMKitLayoutAdapter.matches(result)
    layout = LMKitLayoutAdapter().to_layout_output(result)
    assert layout.model == LayoutDetectionModel.LMKIT_LAYOUT
    assert len(layout.predictions) == 7
    picture = next(p for p in layout.predictions if p.label == "Picture")
    assert picture.content is not None and picture.content.text == "2024 Sales", "a picture keeps the words it prints"


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_each_effort_level_is_a_pipeline(effort):
    pipeline = get_pipeline(f"lmkit_{effort}")
    assert pipeline.provider_name == "lmkit"
    assert pipeline.config["effort"] == effort.capitalize()
