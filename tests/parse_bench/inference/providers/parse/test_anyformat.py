"""anyformat provider: a v3 run envelope becomes a ParseOutput with anchors and citation ids
stripped, pages grouped by block anchor, list-price cost from the page count, layout pages from
the blocks' normalized bboxes, and HTTP statuses mapped onto the provider error taxonomy."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.anyformat import (
    CREDITS_PER_PAGE,
    AnyformatProvider,
    clean_markdown,
    split_markdown_by_page,
)
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

_TABLE = (
    '<table><tr><th data-cell-id="r0c0">Item</th><th data-cell-id="r0c1">Qty</th></tr>'
    '<tr><td data-cell-id="r1c0">Bolt</td><td data-cell-id="r1c1">12</td></tr></table>'
)
_MARKDOWN = (
    '<a id="p1_b0"></a>\n\n# Parts list\n\n'
    '<a id="p1_b1"></a>\n\nOrdered for the pump.\n\n'
    f'<a id="p2_b0"></a>\n\n{_TABLE}\n\n'
    '<a id="p2_b1"></a>\n\nEnd of list.'
)
_BLOCKS = [
    {
        "id": "p1_b0",
        "type": "title",
        "page": 1,
        "bbox": {"x0": 0.1, "y0": 0.05, "x1": 0.9, "y1": 0.12},
        "layout_confidence": 0.95,
        "content": "Parts list",
    },
    {
        "id": "p1_b1",
        "type": "text",
        "page": 1,
        "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.9, "y1": 0.3},
        "layout_confidence": 0.9,
        "content": "Ordered for the pump.",
    },
    {
        "id": "p2_b0",
        "type": "table",
        "page": 2,
        "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.5},
        "layout_confidence": 0.8,
        "content": _TABLE,
    },
    {
        "id": "p2_b1",
        "type": "text",
        "page": 2,
        "bbox": {"x0": 0.1, "y0": 0.6, "x1": 0.9, "y1": 0.65},
        "layout_confidence": None,
        "content": "End of list.",
    },
]


def _run(status: str = "processed", *, with_results: bool = True) -> dict[str, Any]:
    return {
        "id": "run-1",
        "workflow_id": "wf-1",
        "document_packet_id": "dp-1",
        "status": status,
        "results": {"parse": {"markdown": _MARKDOWN, "blocks": _BLOCKS}} if with_results else None,
    }


class _Response:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Client:
    """Scripted HTTP client: POSTs and GETs are served from queues, calls are recorded."""

    def __init__(self, posts: list[_Response], gets: list[_Response]) -> None:
        self._posts = list(posts)
        self._gets = list(gets)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append(("POST", url, kwargs))
        return self._posts.pop(0)

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append(("GET", url, kwargs))
        return self._gets.pop(0)


def _provider(client: _Client, **config: Any) -> AnyformatProvider:
    provider = AnyformatProvider("anyformat", {"api_key": "af_test", "poll_interval": 0, **config})
    provider._http = client
    return provider


def _request(tmp_path: Path) -> InferenceRequest:
    pdf = tmp_path / "0000027_page1.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    return InferenceRequest(example_id="table/0000027_page1", source_file_path=str(pdf), product_type=ProductType.PARSE)


_PIPELINE = PipelineSpec(pipeline_name="anyformat_standard", provider_name="anyformat", product_type=ProductType.PARSE)


def test_clean_markdown_drops_anchors_and_citation_ids_but_no_words() -> None:
    cleaned = clean_markdown(_MARKDOWN)

    assert "<a id=" not in cleaned and "data-cell-id" not in cleaned
    assert "# Parts list" in cleaned and "<th>Item</th>" in cleaned and "End of list." in cleaned


def test_markdown_is_grouped_by_the_page_named_in_each_anchor() -> None:
    pages = split_markdown_by_page(_MARKDOWN)

    assert list(pages) == [1, 2]
    assert pages[1] == "# Parts list\n\nOrdered for the pump."
    assert pages[2].startswith("<table>") and pages[2].endswith("End of list.")
    assert split_markdown_by_page("plain text without anchors") == {1: "plain text without anchors"}


def test_a_processed_run_becomes_a_two_page_parse_output_with_layout(tmp_path: Path) -> None:
    client = _Client(
        posts=[_Response({"id": "wf-1"}, 201), _Response({"run_id": "run-1", "status": "queued"}, 202)],
        gets=[_Response(_run("in_progress", with_results=False)), _Response(_run())],
    )
    provider = _provider(client)

    result = provider.normalize(provider.run_inference(_PIPELINE, _request(tmp_path)))

    output = result.output
    assert isinstance(output, ParseOutput)
    assert [p.page_index for p in output.pages] == [0, 1]
    assert "data-cell-id" not in output.markdown and "<a id=" not in output.markdown
    assert output.job_id == "run-1"
    assert [lp.page_number for lp in output.layout_pages] == [1, 2]
    table_item = output.layout_pages[1].items[0]
    assert table_item.type == "table" and table_item.bbox is not None
    assert (table_item.bbox.x, table_item.bbox.y, round(table_item.bbox.w, 6), table_item.bbox.h) == (
        0.1,
        0.1,
        0.8,
        0.4,
    )
    assert table_item.bbox.label == "table" and "data-cell-id" not in table_item.value
    assert output.layout_pages[0].items[0].type == "text"


def test_the_workflow_is_created_once_with_the_tier_and_cache_off(tmp_path: Path) -> None:
    client = _Client(
        posts=[
            _Response({"id": "wf-1"}, 201),
            _Response({"run_id": "run-1"}, 202),
            _Response({"run_id": "run-2"}, 202),
        ],
        gets=[_Response(_run()), _Response(_run())],
    )
    provider = _provider(client, mode="agentic", effort="accurate")

    provider.run_inference(_PIPELINE, _request(tmp_path))
    provider.run_inference(_PIPELINE, _request(tmp_path))

    creates = [c for c in client.calls if c[1] == "/v3/workflows/"]
    assert len(creates) == 1
    assert creates[0][2]["json"]["nodes"] == [
        {"id": "parse_1", "type": "parse", "mode": "agentic", "cache": False, "effort": "accurate"}
    ]
    submit = next(c for c in client.calls if c[1].endswith("/upload/run/"))
    assert submit[1] == "/v3/workflows/wf-1/upload/run/"
    assert submit[2]["data"] == {"on_conflict": "rename"}
    assert "Idempotency-Key" in submit[2]["headers"]


def test_a_configured_workflow_id_is_reused_without_a_create_call(tmp_path: Path) -> None:
    client = _Client(posts=[_Response({"run_id": "run-1"}, 202)], gets=[_Response(_run())])
    provider = _provider(client, workflow_id="wf-existing")

    provider.run_inference(_PIPELINE, _request(tmp_path))

    assert client.calls[0][1] == "/v3/workflows/wf-existing/upload/run/"


def test_cost_is_the_tier_list_price_times_the_pages_the_blocks_span(tmp_path: Path) -> None:
    client = _Client(posts=[_Response({"run_id": "run-1"}, 202)], gets=[_Response(_run())])
    provider = _provider(client, workflow_id="wf", credit_rate_usd=0.002)

    raw = provider.run_inference(_PIPELINE, _request(tmp_path)).raw_output

    assert raw["num_pages"] == 2
    assert raw["credits_used"] == 2 * CREDITS_PER_PAGE["standard"] == 50
    assert raw["cost_per_page_usd"] == pytest.approx(25 * 0.002)
    assert raw["cost_usd"] == pytest.approx(50 * 0.002)


def test_recompute_cost_reprices_a_saved_run_from_its_own_pages_and_tier() -> None:
    provider = AnyformatProvider("anyformat", {"api_key": "af_test", "credit_rate_usd": 0.001})
    raw = {"num_pages": 3, "_config": {"mode": "flash"}, "cost_usd": 999.0}

    provider.recompute_cost(raw)

    assert raw["credits_used"] == 21 and raw["cost_usd"] == pytest.approx(0.021)
    usage_less: dict[str, Any] = {"_config": {"mode": "flash"}, "cost_usd": 999.0}
    provider.recompute_cost(usage_less)
    assert usage_less["cost_usd"] == 999.0


@pytest.mark.parametrize(
    "status_code, error",
    [
        (401, ProviderConfigError),
        (402, ProviderPermanentError),
        (429, ProviderRateLimitError),
        (503, ProviderTransientError),
    ],
)
def test_http_statuses_map_onto_the_provider_error_taxonomy(tmp_path: Path, status_code: int, error: type) -> None:
    client = _Client(posts=[_Response({"error": "nope", "error_code": "X"}, status_code)], gets=[])
    provider = _provider(client, workflow_id="wf")

    with pytest.raises(error):
        provider.run_inference(_PIPELINE, _request(tmp_path))


def test_a_run_that_ends_in_error_is_a_permanent_failure(tmp_path: Path) -> None:
    client = _Client(posts=[_Response({"run_id": "run-1"}, 202)], gets=[_Response(_run("error", with_results=False))])
    provider = _provider(client, workflow_id="wf")

    with pytest.raises(ProviderPermanentError):
        provider.run_inference(_PIPELINE, _request(tmp_path))


def test_missing_api_key_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANYFORMAT_API_KEY", raising=False)

    with pytest.raises(ProviderConfigError):
        AnyformatProvider("anyformat", {})


def test_normalize_rejects_a_non_parse_result() -> None:
    provider = AnyformatProvider("anyformat", {"api_key": "af_test"})
    request = InferenceRequest(example_id="x", source_file_path="/tmp/x.pdf", product_type=ProductType.EXTRACT)
    raw = RawInferenceResult(
        request=request,
        pipeline=_PIPELINE,
        pipeline_name="anyformat_standard",
        product_type=ProductType.EXTRACT,
        raw_output={},
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=0,
    )

    with pytest.raises(ProviderPermanentError):
        provider.normalize(raw)


def test_layout_adapter_scales_normalized_bboxes_and_the_mapper_canonicalizes_labels(tmp_path: Path) -> None:
    from parse_bench.evaluation.layout_adapters.adapters import AnyformatLayoutAdapter
    from parse_bench.evaluation.layout_label_mappers.base import MappingContext
    from parse_bench.evaluation.layout_label_mappers.mappers import AnyformatLabelMapper
    from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
    from parse_bench.schemas.layout_ontology import CanonicalLabel

    client = _Client(posts=[_Response({"run_id": "run-1"}, 202)], gets=[_Response(_run())])
    provider = _provider(client, workflow_id="wf")
    result = provider.normalize(provider.run_inference(_PIPELINE, _request(tmp_path)))

    layout = AnyformatLayoutAdapter().to_layout_output(result, page_filter=2)

    assert layout.model == LayoutDetectionModel.ANYFORMAT_LAYOUT
    assert (layout.image_width, layout.image_height) == (1000, 1000)
    assert [p.label for p in layout.predictions] == ["table", "text"]
    assert [round(v) for v in layout.predictions[0].bbox] == [100, 100, 900, 500]
    assert layout.predictions[0].content is not None and layout.predictions[0].content.type == "table"

    context = MappingContext(
        provider_name="anyformat",
        pipeline_name="anyformat_standard",
        model=LayoutDetectionModel.ANYFORMAT_LAYOUT,
        raw_output=result.raw_output,
        layout_output=layout,
    )
    mapper = AnyformatLabelMapper()
    canonical = [mapper.to_canonical(p.label, p, context) for p in layout.predictions]
    assert canonical == [CanonicalLabel.TABLE, CanonicalLabel.TEXT]
    assert mapper.to_canonical("section-header", layout.predictions[0], context) == CanonicalLabel.SECTION_HEADER
    assert mapper.to_canonical("other", layout.predictions[0], context) == CanonicalLabel.TEXT
    assert mapper.to_canonical("chart", layout.predictions[0], context) == CanonicalLabel.PICTURE


def test_layout_adapter_reads_an_item_that_carries_only_a_bbox():
    """A sidecar that fills `bbox` and not `layout_segments` describes the same region."""
    from parse_bench.evaluation.layout_adapters.adapters import AnyformatLayoutAdapter
    from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR, ParseOutput
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult

    seg = LayoutSegmentIR(x=0.1, y=0.2, w=0.3, h=0.1, label="Title")
    page = ParseLayoutPageIR(
        page_number=1, width=1.0, height=1.0, items=[LayoutItemIR(type="Title", value="Hi", bbox=seg)]
    )
    output = ParseOutput(
        example_id="layout/x", pipeline_name="anyformat_standard", pages=[], markdown="# Hi", layout_pages=[page]
    )
    from datetime import UTC

    now = datetime.now(UTC)
    result = InferenceResult(
        request=InferenceRequest(example_id="layout/x", source_file_path="", product_type="parse"),
        pipeline_name="anyformat_standard",
        product_type="parse",
        raw_output={},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )

    layout = AnyformatLayoutAdapter().to_layout_output(result, page_filter=1)

    assert [p.label for p in layout.predictions] == ["Title"]
