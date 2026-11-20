"""Tests for the Webcite provider: normalization, layout boxes, retries and the layout adapter."""

import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from parse_bench.evaluation.layout_adapters.adapters import WebciteLayoutAdapter
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import ProviderConfigError, ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse import webcite
from parse_bench.inference.providers.parse.webcite import WebciteProvider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

PIPELINE = PipelineSpec(pipeline_name="webcite", provider_name="webcite", product_type=ProductType.PARSE, config={})


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> WebciteProvider:
    monkeypatch.setenv("WEBCITE_API_KEY", "test-key")
    monkeypatch.setattr(webcite.time, "sleep", lambda _: None)
    return WebciteProvider("webcite", {"base_url": "https://api.example.test/api/v1"})


def _raw(extract: dict, path: str = "/tmp/doc.pdf") -> RawInferenceResult:
    now = datetime.now()
    return RawInferenceResult(
        request=InferenceRequest(example_id="doc", source_file_path=path, product_type=ProductType.PARSE),
        pipeline=PIPELINE,
        pipeline_name="webcite",
        product_type=ProductType.PARSE,
        raw_output={"asset_id": "a1", "extract": extract, "_config": {"provider": "webcite"}},
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


SCANNED = {
    "markdown": "# Report\n\n| Year | Revenue |\n| --- | --- |\n| 2024 | 12.4 |\n",
    "state": "complete",
    "units": [
        {
            "provenance": {"page": 1},
            "text": "# Report\n\n| Year | Revenue |\n| --- | --- |\n| 2024 | 12.4 |",
            "geometry": {"coordinateSystem": "normalized-top-left-v1", "width": 612, "height": 792, "rotation": 0},
            "elements": [
                {"type": "title", "markdown": "# Report", "bbox": [0.1, 0.05, 0.9, 0.1]},
                {"type": "table", "markdown": "| Year | Revenue |", "bbox": [0.1, 0.2, 0.9, 0.5]},
                {"type": "picture", "markdown": "", "description": "Logo", "bbox": [0.8, 0.0, 0.95, 0.04]},
                {"type": "page_footer", "markdown": "Page 3", "bbox": [0.4, 0.95, 0.6, 0.98]},
                {"type": "text", "markdown": "bad box", "bbox": [0.9, 0.1, 0.2, 0.3]},
                {"type": "unknown_kind", "markdown": "x", "bbox": [0.1, 0.1, 0.2, 0.2]},
            ],
        }
    ],
}


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEBCITE_API_KEY", raising=False)
    with pytest.raises(ProviderConfigError):
        WebciteProvider("webcite", {})


def test_normalize_passes_markdown_through_and_builds_pages_and_layout(provider: WebciteProvider) -> None:
    result = provider.normalize(_raw(SCANNED))
    output = result.output
    assert output.markdown == SCANNED["markdown"]
    assert [p.page_index for p in output.pages] == [0]
    assert "| Year |" in output.pages[0].markdown
    [page] = output.layout_pages
    assert page.page_number == 1
    assert [item.layout_segments[0].label for item in page.items] == ["Title", "Table", "Picture", "Page-footer"]
    assert page.items[1].type == "table" and page.items[2].type == "image" and page.items[2].value == "Logo"
    assert page.page_footer_markdown == "Page 3"
    title = page.items[0].layout_segments[0]
    assert (title.x, title.y, round(title.w, 6), round(title.h, 6)) == (0.1, 0.05, 0.8, 0.05)


def test_text_layer_page_has_text_but_no_invented_boxes(provider: WebciteProvider) -> None:
    result = provider.normalize(
        _raw(
            {
                "markdown": "Plain text",
                "state": "complete",
                "units": [{"provenance": {"page": 1}, "text": "Plain text"}],
            }
        )
    )
    assert result.output.markdown == "Plain text"
    assert result.output.layout_pages == []


def _layout_values(provider: WebciteProvider, *elements: dict) -> list[str]:
    box = [0.1, 0.1, 0.9, 0.2]
    unit = {"provenance": {"page": 1}, "text": "page text", "elements": [{**e, "bbox": box} for e in elements]}
    result = provider.normalize(_raw({"markdown": "page text", "state": "complete", "units": [unit]}))
    [page] = result.output.layout_pages
    assert result.raw_output["extract"]["units"][0]["elements"] == unit["elements"]  # the response is not edited
    assert page.md == "page text"
    return [item.value for item in page.items]


def test_layout_text_drops_inline_tags_without_a_space_and_decodes_entities(provider: WebciteProvider) -> None:
    values = _layout_values(
        provider,
        {"type": "text", "markdown": 'Report<u>ing</u> <s>old</s> <span class="x">R&amp;D</span>&nbsp;costs &lt; 5'},
        {"type": "section_header", "markdown": "## <b>Scope</b> **bold stays**"},
        {"type": "page_footer", "markdown": "AT&amp;T <br> 3"},
    )
    assert values == ["Reporting old R&D costs < 5", "## Scope **bold stays**", "AT&T <br> 3"]
    assert [item.md for item in provider.normalize(_raw(SCANNED)).output.layout_pages[0].items][0] == "# Report"


def test_layout_text_leaves_tables_and_pictures_untouched(provider: WebciteProvider) -> None:
    table = "<table><tr><td><b>A&amp;B</b></td></tr></table>"
    values = _layout_values(
        provider,
        {"type": "table", "markdown": table},
        {"type": "picture", "markdown": "<b>ACME</b> &amp; $x^2$", "description": "Logo"},
        {"type": "text", "markdown": f"Note <b>x</b> {table}"},
    )
    assert values == [table, "<b>ACME</b> &amp; $x^2$", f"Note <b>x</b> {table}"]


def test_layout_text_writes_formulas_and_inline_maths_as_plain_characters(provider: WebciteProvider) -> None:
    values = _layout_values(
        provider,
        {"type": "formula", "markdown": "$$E = \\frac{a}{b} \\times \\text{gain} + \\sqrt{\\beta^2}$$"},
        {"type": "text", "markdown": "where $\\alpha_{i} \\leq 5\\%$ and the price is $5 or $7 today"},
        {"type": "list_item", "markdown": "- growth of $x^2$ per year"},
    )
    assert values == [
        "E = a/b × gain + β2",
        "where αi ≤ 5% and the price is $5 or $7 today",
        "- growth of x2 per year",
    ]


def test_empty_extraction_is_permanent(provider: WebciteProvider) -> None:
    with pytest.raises(ProviderPermanentError):
        provider.normalize(_raw({"markdown": "", "state": "error", "reason": "empty_source"}))


def test_layout_adapter_converts_segments(provider: WebciteProvider) -> None:
    result = provider.normalize(_raw(SCANNED))
    assert WebciteLayoutAdapter.matches(result)
    layout = WebciteLayoutAdapter().to_layout_output(result)
    assert [p.label for p in layout.predictions] == ["Title", "Table", "Picture", "Page-footer"]
    assert layout.predictions[0].bbox == pytest.approx([100.0, 50.0, 900.0, 100.0])


def _run(provider: WebciteProvider, tmp_path: Path, handler) -> RawInferenceResult:  # type: ignore[no-untyped-def]
    source = tmp_path / "page.pdf"
    source.write_bytes(b"%PDF-1.4 test")
    real_client = httpx.Client
    provider._httpx = type(
        "H",
        (),
        {
            "Client": lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
            "TimeoutException": httpx.TimeoutException,
            "TransportError": httpx.TransportError,
        },
    )
    request = InferenceRequest(example_id="doc", source_file_path=str(source), product_type=ProductType.PARSE)
    return provider.run_inference(PIPELINE, request)


def test_run_inference_uploads_then_extracts_and_retries_transient(provider: WebciteProvider, tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.headers["x-api-key"] == "test-key"
        if request.url.path.endswith("/upload"):
            return (
                httpx.Response(503, text="busy")
                if calls.count(request.url.path) == 1
                else httpx.Response(200, json={"data": {"asset_id": "a1"}})
            )
        assert request.read() == b'{"asset_id":"a1"}'
        return httpx.Response(200, json=SCANNED)

    raw = _run(provider, tmp_path, handler)
    assert calls == ["/api/v1/upload", "/api/v1/upload", "/api/v1/extract"]
    assert raw.raw_output["asset_id"] == "a1" and raw.raw_output["extract"]["state"] == "complete"


def test_client_error_is_permanent_and_persistent_5xx_is_transient(provider: WebciteProvider, tmp_path: Path) -> None:
    with pytest.raises(ProviderPermanentError):
        _run(provider, tmp_path, lambda request: httpx.Response(400, text="bad file"))
    with pytest.raises(ProviderTransientError):
        _run(provider, tmp_path, lambda request: httpx.Response(502, text="down"))


def test_retry_of_a_document_reuses_its_upload_and_records_credits(provider: WebciteProvider, tmp_path: Path) -> None:
    calls: list[str] = []
    extract_status = iter([200, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/upload"):
            return httpx.Response(200, json={"data": {"asset_id": "a1"}, "usage": {"credits": 1}})
        return httpx.Response(next(extract_status), json={**SCANNED, "usage": {"credits": 1}})

    first = _run(provider, tmp_path, handler)
    second = _run(provider, tmp_path, handler)
    assert calls.count("/api/v1/upload") == 1 and calls.count("/api/v1/extract") == 2
    assert first.raw_output["credits_used"] == 2 and first.raw_output["num_pages"] == 1
    assert second.raw_output["asset_id"] == "a1"


def test_rate_limit_waits_for_retry_after_without_spending_attempts(
    provider: WebciteProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(webcite.time, "sleep", waits.append)
    responses = iter(
        [httpx.Response(429, headers={"retry-after": "7"}, json={"retryAfter": 7})] * 5
        + [httpx.Response(200, json=SCANNED)]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload"):
            return httpx.Response(200, json={"data": {"asset_id": "a2"}})
        return next(responses)

    raw = _run(provider, tmp_path, handler)
    assert raw.raw_output["extract"]["state"] == "complete"
    assert waits == [7.0] * 5
    assert raw.raw_output["credits_used"] is None


def test_unknown_credit_charge_is_none_not_zero() -> None:
    assert webcite._credits(1, None) is None and webcite._credits(1, 1) == 2


def test_fractional_credit_charge_is_recorded_as_reported(provider: WebciteProvider, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload"):
            return httpx.Response(200, json={"data": {"asset_id": "a3"}, "usage": {"credits": 0}})
        return httpx.Response(200, json={**SCANNED, "usage": {"credits": 0.7325}})

    raw = _run(provider, tmp_path, handler)
    assert raw.raw_output["credits_used"] == 0.7325
    assert webcite._credits(True, 1) is None and webcite._credits(float("nan"), 1) is None


@pytest.mark.parametrize(("pipeline", "mode"), [("webcite", "standard"), ("webcite_lite", "lite")])
def test_each_pipeline_sends_its_read_mode(pipeline: str, mode: str) -> None:
    spec = get_pipeline(pipeline)
    assert spec.provider_name == "webcite" and spec.product_type == ProductType.PARSE
    assert spec.config["mode"] == mode


@pytest.mark.parametrize("mode", ["standard", "lite"])
def test_extract_request_carries_the_configured_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    monkeypatch.setenv("WEBCITE_API_KEY", "test-key")
    configured = WebciteProvider("webcite", {"base_url": "https://api.example.test/api/v1", "mode": mode})
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload"):
            return httpx.Response(200, json={"data": {"asset_id": "a1"}})
        bodies.append(json.loads(request.read()))
        return httpx.Response(200, json=SCANNED)

    _run(configured, tmp_path, handler)
    assert bodies == [{"asset_id": "a1", "mode": mode}]


def test_unknown_mode_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBCITE_API_KEY", "test-key")
    with pytest.raises(ProviderConfigError):
        WebciteProvider("webcite", {"mode": "fast"})
