"""Tests for the providus_vlm_mini provider and its layout adapter. A fake /predict answers; no model runs."""

import base64
import json
from pathlib import Path

import httpx
import pymupdf
import pytest

from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers.projection import project_layout_predictions
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import ProviderConfigError, ProviderTransientError
from parse_bench.inference.providers.parse.providus_vlm_mini import ProvidusVlmMiniProvider
from parse_bench.schemas.layout_detection_output import LayoutOutput
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

PAGE_ANSWERS = [
    {
        "status": "success",
        "markdown": "# Heading\n\nBody text.",
        "image_width": 200,
        "image_height": 400,
        "items": [
            {"category": "Section-header", "bbox": [0.1, 0.1, 0.5, 0.2], "text": "Heading"},
            {"category": "Text", "bbox": [0.1, 0.25, 0.9, 0.5], "text": "Body text."},
        ],
    },
    {
        "status": "success",
        "markdown": "Second page.",
        "image_width": 200,
        "image_height": 400,
        "items": [
            {"category": "Chart", "bbox": [0.0, 0.5, 1.0, 1.0], "text": "| a | b |"},
            {"category": "Table", "bbox": [0.0, 0.0, 0.5, 0.5], "text": "<table><tr><td>1</td></tr></table>"},
        ],
    },
]


def _pdf(path: Path, pages: int = 2) -> Path:
    doc = pymupdf.open()
    for n in range(pages):
        doc.new_page(width=200, height=400).insert_text((20, 40), f"page {n}")
    doc.save(str(path))
    return path


def _provider(answers: list, seen: list | None = None) -> ProvidusVlmMiniProvider:
    queue = list(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        base64.b64decode(body["image_base64"], validate=True)
        if seen is not None:
            seen.append(request.url.path)
        return httpx.Response(200, json=queue.pop(0))

    spec = get_pipeline("providus_vlm_mini")
    provider = ProvidusVlmMiniProvider("providus_vlm_mini", {**spec.config, "server_url": "http://fake:8100"})
    provider._transport = httpx.MockTransport(handler)
    return provider


def _run(provider: ProvidusVlmMiniProvider, pdf: Path):
    spec = get_pipeline("providus_vlm_mini")
    request = InferenceRequest(example_id="doc1", source_file_path=str(pdf), product_type=ProductType.PARSE)
    return provider.run_inference_normalized(spec, request)


def test_pipeline_is_registered_with_defaults() -> None:
    spec = get_pipeline("providus_vlm_mini")
    assert spec.provider_name == "providus_vlm_mini"
    assert spec.config == {"server_url": "", "dpi": 144, "timeout": 900}


def test_two_page_pdf_joins_markdown_and_posts_to_predict(tmp_path: Path) -> None:
    seen: list[str] = []
    result = _run(_provider(PAGE_ANSWERS, seen), _pdf(tmp_path / "a.pdf"))
    assert seen == ["/predict", "/predict"]
    assert result.output.markdown == "# Heading\n\nBody text.\n\nSecond page."
    assert [p.page_index for p in result.output.pages] == [0, 1]
    assert [p.page_number for p in result.output.layout_pages] == [1, 2]
    assert result.output.layout_pages[0].width == 200


def test_pages_render_at_the_configured_dpi(tmp_path: Path) -> None:
    sizes: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        png = base64.b64decode(json.loads(request.content)["image_base64"])
        pix = pymupdf.Pixmap(png)
        sizes.append((pix.width, pix.height))
        return httpx.Response(200, json=PAGE_ANSWERS[0])

    provider = _provider([])
    provider._transport = httpx.MockTransport(handler)
    _run(provider, _pdf(tmp_path / "a.pdf", pages=1))
    assert sizes == [(400, 800)]  # 200 x 400 points at 144 DPI is 2x


def test_error_page_is_a_failed_page_not_a_crash(tmp_path: Path) -> None:
    answers = [{"status": "error", "error": "ValueError: bad layout"}, PAGE_ANSWERS[1]]
    result = _run(_provider(answers), _pdf(tmp_path / "a.pdf"))
    assert result.output.markdown == "Second page."
    assert result.output.pages[0].markdown == ""
    assert result.raw_output["failed_pages"] == 1
    assert result.raw_output["pages"][0]["error"] == "ValueError: bad layout"
    layout = create_layout_adapter_for_result(result).to_layout_output(result)
    assert [p.page_number for p in layout.layout_pages] == [1, 2]
    assert layout.layout_pages[0].items == []


def test_answer_that_is_not_an_object_is_a_failed_page(tmp_path: Path) -> None:
    result = _run(_provider([["not", "a", "dict"], PAGE_ANSWERS[1]]), _pdf(tmp_path / "a.pdf"))
    assert result.raw_output["failed_pages"] == 1


def test_http_error_is_a_transient_error(tmp_path: Path) -> None:
    provider = _provider([])
    provider._transport = httpx.MockTransport(lambda request: httpx.Response(503, text="busy"))
    with pytest.raises(ProviderTransientError):
        _run(provider, _pdf(tmp_path / "a.pdf"))


def test_missing_server_url_is_a_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROVIDUS_PARSE_URL", raising=False)
    provider = ProvidusVlmMiniProvider("providus_vlm_mini", {"server_url": "", "dpi": 144, "timeout": 5})
    with pytest.raises(ProviderConfigError):
        _run(provider, _pdf(tmp_path / "a.pdf"))


def test_server_url_comes_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDUS_PARSE_URL", "http://env-host:8100/")
    provider = _provider(PAGE_ANSWERS)
    provider._base_config["server_url"] = ""
    result = _run(provider, _pdf(tmp_path / "a.pdf"))
    assert result.raw_output["server_url"] == "http://env-host:8100"


def test_adapter_maps_boxes_to_pixels_and_labels_to_canonical(tmp_path: Path) -> None:
    result = _run(_provider(PAGE_ANSWERS), _pdf(tmp_path / "a.pdf"))
    adapter = create_layout_adapter_for_result(result)
    layout = adapter.to_layout_output(result)
    assert isinstance(layout, LayoutOutput)
    assert (layout.image_width, layout.image_height) == (200, 400)
    header, text = layout.layout_pages[0].items
    assert header.type == "Section-header"
    seg = header.bbox
    assert (seg.x, seg.y, seg.w, seg.h) == pytest.approx((20, 40, 80, 40))
    chart, table = layout.layout_pages[1].items
    assert chart.type == "Picture"
    assert chart.attributes == {"picture_type": "chart"}
    assert table.type == "Table"
    flat = [(p.page, p.label) for p in layout.predictions]
    assert flat == [(1, "Section-header"), (1, "Text"), (2, "Picture"), (2, "Table")]
    assert layout.predictions[3].content.html.startswith("<table>")
    assert layout.predictions[1].bbox == pytest.approx([20, 100, 180, 200])


def test_adapter_page_filter_keeps_one_page(tmp_path: Path) -> None:
    result = _run(_provider(PAGE_ANSWERS), _pdf(tmp_path / "a.pdf"))
    layout = create_layout_adapter_for_result(result).to_layout_output(result, page_filter=2)
    assert [p.page_number for p in layout.layout_pages] == [2]
    assert {p.page for p in layout.predictions} == {2}


def test_evaluator_projection_reads_the_adapter_output(tmp_path: Path) -> None:
    result = _run(_provider(PAGE_ANSWERS), _pdf(tmp_path / "a.pdf"))
    layout = create_layout_adapter_for_result(result).to_layout_output(result)
    projected = project_layout_predictions(result, layout, evaluation_view="canonical", target_ontology="canonical")
    assert [(p["page"], p["class_name"]) for p in projected] == [
        (1, "Section-header"),
        (1, "Text"),
        (2, "Picture"),
        (2, "Table"),
    ]
    assert projected[0]["bbox"] == pytest.approx([0.1, 0.1, 0.5, 0.2])


def test_unknown_category_becomes_text_and_bad_boxes_are_skipped(tmp_path: Path) -> None:
    answer = {
        "status": "success",
        "markdown": "x",
        "image_width": 100,
        "image_height": 100,
        "items": [
            {"category": "Mystery", "bbox": [0.1, 0.1, 0.2, 0.2], "text": "a"},
            {"category": "Text", "bbox": [0.5, 0.5, 0.4, 0.6], "text": "flipped"},
            {"category": "Text", "bbox": [0.1], "text": "short"},
        ],
    }
    result = _run(_provider([answer], None), _pdf(tmp_path / "a.pdf", pages=1))
    layout = create_layout_adapter_for_result(result).to_layout_output(result)
    assert [i.type for i in layout.layout_pages[0].items] == ["Text"]


def _bodies(provider: ProvidusVlmMiniProvider, answers: list) -> list[dict]:
    """Keep the request bodies that the fake server gets."""
    bodies: list[dict] = []
    queue = list(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    provider._transport = httpx.MockTransport(handler)
    return bodies


def test_pdf_source_sends_each_page_as_a_one_page_pdf(tmp_path: Path) -> None:
    provider = _provider([])
    bodies = _bodies(provider, PAGE_ANSWERS)
    _run(provider, _pdf(tmp_path / "a.pdf"))
    assert len(bodies) == 2
    for n, body in enumerate(bodies):
        with pymupdf.open(stream=base64.b64decode(body["pdf_base64"]), filetype="pdf") as one:
            assert one.page_count == 1
            assert one[0].get_text().strip() == f"page {n}"


def test_image_source_sends_no_pdf(tmp_path: Path) -> None:
    png = tmp_path / "a.png"
    pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 10), False).save(str(png))
    provider = _provider([])
    bodies = _bodies(provider, PAGE_ANSWERS[:1])
    _run(provider, png)
    assert len(bodies) == 1 and "pdf_base64" not in bodies[0] and bodies[0]["image_base64"]


DECORATED = {
    "status": "success",
    "markdown": "ACME report\n\nBody.\n\nConfidential\n\n7",
    "image_width": 200,
    "image_height": 400,
    "text_layer": {"used": 2, "kept_model": 1},
    "items": [
        {"category": "Page-header", "bbox": [0.1, 0.0, 0.9, 0.05], "text": "ACME report", "type": "header"},
        {"category": "Text", "bbox": [0.1, 0.2, 0.9, 0.5], "text": "Body.", "type": "text"},
        {"category": "Page-footer", "bbox": [0.1, 0.9, 0.7, 0.95], "text": "Confidential", "type": "footer"},
        {"category": "Page-footer", "bbox": [0.8, 0.9, 0.9, 0.95], "text": "7", "type": "page_number"},
    ],
}


def test_normalize_fills_the_page_header_footer_and_printed_page_number(tmp_path: Path) -> None:
    result = _run(_provider([DECORATED, PAGE_ANSWERS[1]]), _pdf(tmp_path / "a.pdf"))
    first, second = result.output.layout_pages
    assert first.page_header_markdown == "ACME report"
    assert first.page_footer_markdown == "Confidential\n\n7"  # the page number item is a Page-footer item too
    assert first.printed_page_number == "7"
    assert (second.page_header_markdown, second.page_footer_markdown, second.printed_page_number) == ("", "", "")
    assert result.raw_output["pages"][0]["text_layer"] == {"used": 2, "kept_model": 1}


def _decorated(items: list) -> dict:
    return {**DECORATED, "items": items}


def test_a_page_number_at_the_top_goes_into_the_header(tmp_path: Path) -> None:
    items = [
        {"category": "Page-header", "bbox": [0.1, 0.0, 0.7, 0.05], "text": "ACME report", "type": "header"},
        {"category": "Page-footer", "bbox": [0.8, 0.0, 0.9, 0.05], "text": "7", "type": "page_number"},
        {"category": "Text", "bbox": [0.1, 0.2, 0.9, 0.5], "text": "Body.", "type": "text"},
        {"category": "Page-footer", "bbox": [0.1, 0.9, 0.7, 0.95], "text": "Confidential", "type": "footer"},
    ]
    result = _run(_provider([_decorated(items)]), _pdf(tmp_path / "a.pdf", pages=1))
    page = result.output.layout_pages[0]
    assert (page.page_header_markdown, page.page_footer_markdown, page.printed_page_number) == (
        "ACME report\n\n7",
        "Confidential",
        "7",
    )


def test_a_field_takes_only_items_at_its_own_page_edge(tmp_path: Path) -> None:
    items = [
        {"category": "Page-header", "bbox": [0.1, 0.9, 0.7, 0.95], "text": "low header", "type": "header"},
        {"category": "Page-footer", "bbox": [0.1, 0.0, 0.7, 0.05], "text": "high footer", "type": "footer"},
        {"category": "Page-footer", "bbox": [0.8, 0.95, 0.9, 1.0], "text": "iv", "type": "page_number"},
    ]
    result = _run(_provider([_decorated(items)]), _pdf(tmp_path / "a.pdf", pages=1))
    page = result.output.layout_pages[0]
    assert (page.page_header_markdown, page.page_footer_markdown, page.printed_page_number) == ("", "iv", "iv")
