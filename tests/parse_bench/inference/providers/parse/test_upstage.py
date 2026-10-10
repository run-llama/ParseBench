"""Tests for the Upstage Document Parse provider."""

import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers import build_mapping_context, resolve_layout_label_mapper
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.upstage import UpstageDocumentParseProvider
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult


def element(category: str, markup: str, page: int = 1, coordinates=None) -> dict:
    return {
        "category": category,
        "page": page,
        "content": {"html": markup, "markdown": markup, "text": markup},
        "coordinates": (
            coordinates
            if coordinates is not None
            else [
                {"x": 0.1, "y": 0.2},
                {"x": 0.8, "y": 0.2},
                {"x": 0.8, "y": 0.6},
                {"x": 0.1, "y": 0.6},
            ]
        ),
    }


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> UpstageDocumentParseProvider:
    monkeypatch.setenv("UPSTAGE_API_KEY", "test-key-not-real")
    monkeypatch.delenv("UPSTAGE_BASE_URL", raising=False)
    spec = get_pipeline("upstage_dpe_v2")
    return UpstageDocumentParseProvider("upstage", spec.config)


def raw(body: dict) -> RawInferenceResult:
    spec = get_pipeline("upstage_dpe_v2")
    now = datetime.now()
    return RawInferenceResult(
        request=InferenceRequest(example_id="test", source_file_path="test.pdf", product_type="parse"),
        pipeline=spec,
        pipeline_name=spec.pipeline_name,
        product_type="parse",
        raw_output=body,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )


def test_normalize_preserves_api_markdown_pages_and_structured_html(provider: UpstageDocumentParseProvider) -> None:
    elements = [
        {
            **element("heading2", "<h2>Title</h2>"),
            "content": {"html": "<h2>Title</h2>", "markdown": "## Title", "text": "Title"},
        },
        element("table", "<table><tr><td>A</td></tr></table>", page=2),
    ]
    result = provider.normalize(raw({"elements": elements, "usage": {"pages": 3}}))

    assert result.output.markdown == "## Title\n\n<table><tr><td>A</td></tr></table>"
    assert [page.page_index for page in result.output.pages] == [0, 1, 2]
    assert result.output.pages[2].markdown == ""
    assert result.output.layout_pages[1].items[0].html == "<table><tr><td>A</td></tr></table>"


def test_normalize_exposes_table_and_chart_semantics(provider: UpstageDocumentParseProvider) -> None:
    elements = [
        {
            **element("caption", "<p>Quarterly revenue</p>"),
            "content": {
                "html": "<p>Quarterly revenue</p>",
                "markdown": "Quarterly revenue",
                "text": "Quarterly revenue",
            },
        },
        {
            **element("chart", ""),
            "content": {
                "html": (
                    '<figure><figcaption><p class="chart-description">Revenue by region</p>'
                    '<p class="chart-ocr-text">North\n2025</p></figcaption>'
                    '<table><tr><td scope="col">Year</td><td scope="col">North</td></tr>'
                    "<tr><td>2025</td><td>42</td></tr></table></figure>"
                ),
                "text": "North 2025 42",
            },
        },
    ]

    markdown = provider.normalize(raw({"elements": elements})).output.markdown

    assert "Quarterly revenue" in markdown
    assert "Revenue by region" not in markdown
    assert "<caption>" not in markdown
    assert "North\n2025" in markdown


def test_normalize_drops_figure_narration_like_chart(provider: UpstageDocumentParseProvider) -> None:
    # The API writes a figure's narration under figure-* classes, the same three fields
    # a chart carries under chart-*. Unmapped, it fell through to body text.
    elements = [
        element("paragraph", "<p>Sales grew in every region.</p>"),
        element(
            "figure",
            "<figure><img /><figcaption>"
            '<p class="figure-type">Figure Type: photo</p>'
            '<p class="figure-description">A photograph showing two mannequins.</p>'
            '<p class="figure-ocr-text">FREE ASSEMBLY</p>'
            "</figcaption></figure>",
        ),
    ]

    markdown = provider.normalize(raw({"elements": elements})).output.markdown

    assert "Sales grew in every region." in markdown
    assert "Figure Type" not in markdown
    assert "two mannequins" not in markdown
    # ocr_text is unwrapped, not dropped: text printed inside the figure stays
    assert "FREE ASSEMBLY" in markdown


def test_normalize_preserves_authored_heading_levels_and_code_language(
    provider: UpstageDocumentParseProvider,
) -> None:
    elements = [
        {
            **element(
                "heading1",
                "<h1>Main title</h1>",
                coordinates=[
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.8, "y": 0.1},
                    {"x": 0.8, "y": 0.2},
                    {"x": 0.1, "y": 0.2},
                ],
            ),
            "content": {"html": "<h1>Main title</h1>", "markdown": "# Main title", "text": "Main title"},
        },
        {
            **element(
                "heading2",
                "<h2>Subsection</h2>",
                coordinates=[
                    {"x": 0.1, "y": 0.3},
                    {"x": 0.4, "y": 0.3},
                    {"x": 0.4, "y": 0.32},
                    {"x": 0.1, "y": 0.32},
                ],
            ),
            "content": {"html": "<h2>Subsection</h2>", "markdown": "## Subsection", "text": "Subsection"},
        },
        {
            **element("code", "<pre><code class=\"language-python\">print('ok')</code></pre>"),
            "content": {
                "html": "<pre><code class=\"language-python\">print('ok')</code></pre>",
                "markdown": "```python\nprint('ok')\n```",
                "text": "print('ok')",
            },
        },
    ]

    markdown = provider.normalize(raw({"elements": elements})).output.markdown

    assert markdown.startswith("# Main title")
    assert "## Subsection" in markdown
    assert "```python" in markdown


def test_normalize_does_not_infer_code_language(provider: UpstageDocumentParseProvider) -> None:
    code = "print('ok')"
    code_element = {
        **element("code", f"<pre><code>{code}</code></pre>"),
        "content": {
            "html": f"<pre><code>{code}</code></pre>",
            "markdown": f"```\n{code}\n```",
            "text": code,
        },
    }

    markdown = provider.normalize(raw({"elements": [code_element]})).output.markdown

    assert "```python" not in markdown
    assert "print('ok')" in markdown
    assert markdown.startswith(chr(96) * 3 + "\n")


def test_layout_adapter_and_label_mapping(provider: UpstageDocumentParseProvider) -> None:
    result = provider.normalize(raw({"elements": [element("heading3", "<h3>Title</h3>")]}))
    adapter = create_layout_adapter_for_result(result)
    output = adapter.to_layout_output(result)

    assert output.model == LayoutDetectionModel.UPSTAGE_LAYOUT
    assert output.predictions[0].bbox == pytest.approx([100, 200, 800, 600])
    context = build_mapping_context(result, output)
    mapper = resolve_layout_label_mapper(context)
    assert (
        mapper.to_canonical(output.predictions[0].label, output.predictions[0], context)
        == CanonicalLabel.SECTION_HEADER
    )


@pytest.mark.parametrize(
    "coordinates",
    [[], [{"x": 0, "y": 0}], [{"x": float("nan"), "y": 0}] * 4, [{"x": 0.1, "y": 0.1}] * 4],
)
def test_invalid_coordinates_do_not_create_layout_predictions(
    provider: UpstageDocumentParseProvider, coordinates: list[dict]
) -> None:
    result = provider.normalize(raw({"elements": [element("paragraph", "<p>Text</p>", coordinates=coordinates)]}))
    assert create_layout_adapter_for_result(result).to_layout_output(result).predictions == []


def test_configuration(provider: UpstageDocumentParseProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTAGE_BASE_URL", "https://example.com/parse")
    configured = UpstageDocumentParseProvider("upstage", provider.base_config)
    assert configured._endpoint == "https://example.com/parse"
    assert configured._form_data() == {
        "model": "document-parse-260930",
        "mode": "enhanced",
        "ocr": "force",
        "output_formats": '["html", "markdown", "text"]',
        "coordinates": "true",
        "chart_recognition": "true",
    }

    monkeypatch.delenv("UPSTAGE_API_KEY")
    with pytest.raises(ProviderConfigError):
        UpstageDocumentParseProvider("upstage")


def mock_http(provider: UpstageDocumentParseProvider, monkeypatch: pytest.MonkeyPatch, handler) -> None:
    original_client = httpx.Client
    monkeypatch.setattr(
        provider._httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )


def request(tmp_path, suffix: str = ".pdf") -> InferenceRequest:
    path = tmp_path / f"test{suffix}"
    if suffix == ".pdf":
        # Valid one-page PDF fixture; the provider needs no client PDF renderer.
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 144] /Resources << >> >>",
        ]
        payload = bytearray(b"%PDF-1.4\n")
        offsets = []
        for number, body in enumerate(objects, 1):
            offsets.append(len(payload))
            payload.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
        xref = len(payload)
        payload.extend(b"xref\n0 4\n0000000000 65535 f \n")
        for offset in offsets:
            payload.extend(f"{offset:010} 00000 n \n".encode())
        payload.extend(f"trailer\n<< /Root 1 0 R /Size 4 >>\nstartxref\n{xref}\n%%EOF\n".encode())
        path.write_bytes(payload)
    else:
        from PIL import Image

        Image.new("RGB", (10, 20), color="white").save(path)
    return InferenceRequest(example_id="test", source_file_path=str(path), product_type="parse")


def test_original_pdf_is_uploaded_without_rasterization(provider: UpstageDocumentParseProvider, tmp_path) -> None:
    source = request(tmp_path)
    path = Path(source.source_file_path)
    name, payload, mime = provider._upload_payload(path)
    assert payload == path.read_bytes()
    assert name == "test.pdf"
    assert mime == "application/pdf"
    assert "rasterize_pdf_dpi" not in provider.base_config


def test_original_image_bytes_are_preserved_when_already_large(
    provider: UpstageDocumentParseProvider, tmp_path
) -> None:
    from PIL import Image

    source = request(tmp_path, ".jpg")
    path = Path(source.source_file_path)
    Image.new("RGB", (2000, 2100), color="white").save(path)
    name, payload, mime = provider._upload_payload(path)
    assert payload == path.read_bytes()
    assert name == "test.jpg"
    assert mime == "image/jpeg"


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, ProviderConfigError),
        (403, ProviderConfigError),
        (429, ProviderRateLimitError),
        (408, ProviderTransientError),
        (503, ProviderTransientError),
        (400, ProviderPermanentError),
    ],
)
def test_error_classification(
    provider: UpstageDocumentParseProvider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    status: int,
    expected: type[Exception],
) -> None:
    mock_http(provider, monkeypatch, lambda _: httpx.Response(status, text="error"))
    with pytest.raises(expected):
        provider.run_inference(get_pipeline("upstage_dpe_v2"), request(tmp_path))


def test_request_shape_cost_and_provenance(
    provider: UpstageDocumentParseProvider, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    body = {"elements": [], "model": "document-parse-260930", "api": "2.0", "usage": {"pages": 2}}

    def respond(req: httpx.Request) -> httpx.Response:
        assert req.headers["Authorization"] == "Bearer test-key-not-real"
        assert req.headers["X-Upstage-Use-Cache"] == "false"
        assert b'name="document"' in req.content
        assert b'filename="test.pdf"' in req.content
        assert b"Content-Type: application/pdf" in req.content
        assert b"document-parse-260930" in req.content
        return httpx.Response(200, json=body)

    mock_http(provider, monkeypatch, respond)
    result = provider.run_inference(get_pipeline("upstage_dpe_v2"), request(tmp_path))

    assert result.raw_output["cost_per_page_usd"] == pytest.approx(0.03)
    assert result.raw_output["cost_usd"] == pytest.approx(0.06)
    assert result.raw_output["_request_config"]["mode"] == "enhanced"
    assert "test-key-not-real" not in json.dumps(result.raw_output)


def test_invalid_json_is_retryable(
    provider: UpstageDocumentParseProvider, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    mock_http(provider, monkeypatch, lambda _: httpx.Response(200, text="truncated"))
    with pytest.raises(ProviderTransientError):
        provider.run_inference(get_pipeline("upstage_dpe_v2"), request(tmp_path))


@pytest.mark.parametrize("suffix", [".pdf", ".jpg", ".png"])
def test_upload_content_type_matches_input(
    provider: UpstageDocumentParseProvider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    suffix: str,
) -> None:
    def respond(req: httpx.Request) -> httpx.Response:
        name = "test.pdf" if suffix == ".pdf" else "test.png"
        assert f'filename="{name}"'.encode() in req.content
        mime = "application/pdf" if suffix == ".pdf" else "image/png"
        assert f"Content-Type: {mime}".encode() in req.content
        return httpx.Response(200, json={"elements": []})

    mock_http(provider, monkeypatch, respond)
    provider.run_inference(get_pipeline("upstage_dpe_v2"), request(tmp_path, suffix))


def test_no_caption_or_emphasis_is_invented(provider: UpstageDocumentParseProvider) -> None:
    authored = "<table><caption>Authored</caption><tr><th>A</th><td><b>B</b></td></tr></table>"
    chart = '<figure><p class="chart-ocr-text">Label Z</p><table><tr><th>X</th><td>1</td></tr></table></figure>'
    result = provider.normalize(
        raw({"elements": [element("caption", "<p>Nearby</p>"), element("chart", chart), element("table", authored)]})
    )
    text = result.output.markdown
    assert text.count("<caption>") == 1
    assert "<caption>Authored</caption>" in text
    assert "<th>X</th>" in text and "<th>A</th>" in text
    assert "<b>B</b>" in text
    assert "<p>Label Z</p>" in text


def test_source_lists_and_cell_formulas_are_preserved(provider: UpstageDocumentParseProvider) -> None:
    source = r"<ol start='3'><li>Parent<ul><li>Child</li></ul></li></ol><table><tr><td>\(x+y\)</td></tr></table>"
    result = provider.normalize(raw({"elements": [element("paragraph", source)]}))
    assert result.output.markdown == source


def test_ocr_and_original_table_caption_are_preserved() -> None:
    from parse_bench.inference.providers.parse.upstage_normalization import normalize_element

    source = (
        '<figure><p class="chart-ocr-text">Axis</p>'
        '<p class="chart-description">Generated</p>'
        "<table><caption>Original</caption><tr><td>Axis</td></tr></table></figure>"
    )
    text = normalize_element(element("chart", source))
    assert text == "<figure><p>Axis</p><table><caption>Original</caption><tr><td>Axis</td></tr></table></figure>"


def test_transport_attributes_and_image_narration_are_not_body_text(provider) -> None:
    source = '<figure id="17"><img data-coord="bottom-right:(10,20)" alt="Generated description" /></figure>'
    result = provider.normalize(raw({"elements": [element("figure", source)]}))
    item = result.output.layout_pages[0].items[0]
    assert result.output.markdown == "<figure><img /></figure>"
    assert item.html == result.output.markdown
    assert item.value == ""
    assert item.layout_segments[0].x == 0.1
    assert result.raw_output["elements"][0]["content"]["html"] == source


@pytest.mark.parametrize(
    "source",
    [
        '```html\n<p id="literal" data-coord="bottom">Example</p>\n```',
        '    <p id="literal" data-coord="bottom">Example</p>\n',
        '`<img id="literal" alt="literal" />`',
        '<pre><code class="language-html">&lt;p id="literal"&gt;Example&lt;/p&gt;</code></pre>',
    ],
)
def test_metadata_cleanup_preserves_literal_code(source) -> None:
    from parse_bench.inference.providers.parse.upstage_normalization import strip_html_metadata

    assert strip_html_metadata(source) == source


def test_html_lists_preserve_literal_markers_and_depth(provider) -> None:
    source = '<ul id="7"><li>- First</li><li>- Second</li></ul>'
    value = element("list", source)
    value["content"]["markdown"] = "- - First\n- - Second"
    assert provider.normalize(raw({"elements": [value]})).output.markdown == (
        "<ul><li>- First</li><li>- Second</li></ul>"
    )


def test_metadata_cleanup_preserves_semantic_attributes() -> None:
    from parse_bench.inference.providers.parse.upstage_normalization import strip_html_metadata

    source = '<a id="4" href="https://example.com/?id=4">Link</a><td colspan="2" data-category="table">X</td>'
    assert strip_html_metadata(source) == '<a href="https://example.com/?id=4">Link</a><td colspan="2">X</td>'


def test_api_markdown_code_is_not_rewritten(provider) -> None:
    content = '```html\n<pre><code class="language-python">literal</code></pre>\n```'
    value = element("code", "<pre><code>unused</code></pre>")
    value["content"]["markdown"] = content
    assert provider.normalize(raw({"elements": [value]})).output.markdown == content


def test_truncated_code_is_not_completed(provider: UpstageDocumentParseProvider) -> None:
    source = '<pre><code class="language-python">x = 1'
    assert provider.normalize(raw({"elements": [element("code", source)]})).output.markdown == source


def test_small_native_images_are_upscaled_with_aspect_ratio(provider: UpstageDocumentParseProvider, tmp_path) -> None:
    import io

    from PIL import Image

    source = request(tmp_path, ".jpg")
    name, payload, mime = provider._upload_payload(Path(source.source_file_path))
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (2000, 4000)
    assert name == "test.png"
    assert mime == "image/png"
