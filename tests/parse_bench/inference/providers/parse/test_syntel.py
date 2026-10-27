"""Tests for the Syntel /v1/parse provider, its layout adapter and label mapper.

``RECORDED_RESULT`` is a ``GET /v1/parse/result/{job_id}`` response recorded on the ParseBench
input ``table/AZ LIC Rate Tables 2.0.v2_page90``, trimmed to the fields the harness reads. No test
reaches the network: HTTP goes through ``httpx.MockTransport``.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any

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
from parse_bench.inference.providers.parse import syntel
from parse_bench.inference.providers.parse.syntel import PROVIDER_NAME, SyntelProvider, upload_name
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel, LayoutTableContent, LayoutTextContent
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult

PIPELINE = "syntel_advanced"
BASE_URL = "https://parse.example.com/api"
JOB_ID = "6b1f6c3e-0d43-4f53-9d55-1f2a1c3b9e10"

RECORDED_RESULT = {
    "success": True,
    "data": {
        "text": "Lemonade Insurance Company Homeowners Program Arizona Edition April 2024\n"
        "\n"
        "**Coverage D Increased Limit (Loss of Use) All Forms**\n"
        "\n"
        '<table><tr><th colspan="1" rowspan="1">Per</th><th colspan="1" '
        'rowspan="1">Rate</th></tr><tr><td colspan="1" rowspan="1">$1,000</td><td colspan="1" '
        'rowspan="1">$2.66</td></tr></table>',
        "total_pages": 1,
        "language": "en",
        "parser_used": "advanced-composed",
        "file_name": "attachment-8c8ee6caeab030d1.pdf",
        "file_size_bytes": 29915,
        "document_format": "pdf",
        "elapsed_ms": 13314.84,
        "regions": [
            {
                "page_number": 1,
                "category": "text",
                "bbox": {"x": 104.5, "y": 117.07090909090908, "w": 298.0, "h": 28.01696969696968},
                "text": "Lemonade Insurance Company",
                "confidence": None,
                "page_width": 1275,
                "page_height": 1651,
                "dpi": 150,
                "coordinate_space": "raster_pixels",
            },
            {
                "page_number": 1,
                "category": "text",
                "bbox": {"x": 104.5, "y": 148.58999999999997, "w": 220.5, "h": 28.01696969696968},
                "text": "Homeowners Program",
                "confidence": None,
                "page_width": 1275,
                "page_height": 1651,
                "dpi": 150,
                "coordinate_space": "raster_pixels",
            },
            {
                "page_number": 1,
                "category": "text",
                "bbox": {"x": 104.5, "y": 181.60999999999999, "w": 84.0, "h": 26.01575757575756},
                "text": "Arizona",
                "confidence": None,
                "page_width": 1275,
                "page_height": 1651,
                "dpi": 150,
                "coordinate_space": "raster_pixels",
            },
            {
                "page_number": 1,
                "category": "text",
                "bbox": {"x": 104.5, "y": 212.62878787878785, "w": 181.0, "h": 25.0151515151515},
                "text": "Edition April 2024",
                "confidence": None,
                "page_width": 1275,
                "page_height": 1651,
                "dpi": 150,
                "coordinate_space": "raster_pixels",
            },
            {
                "page_number": 1,
                "category": "footer",
                "bbox": {"x": 1141.0, "y": 1597.9678787878786, "w": 26.5, "h": 23.01393939393938},
                "text": "90",
                "confidence": None,
                "page_width": 1275,
                "page_height": 1651,
                "dpi": 150,
                "coordinate_space": "raster_pixels",
            },
            {
                "page_number": 1,
                "category": "text",
                "bbox": {"x": 140.76, "y": 273.4056, "w": 377.145, "h": 52.99709999999999},
                "text": "**Coverage D Increased Limit (Loss of Use)** **All Forms**",
                "confidence": None,
                "page_width": 1275,
                "page_height": 1651,
                "dpi": 150,
                "coordinate_space": "raster_pixels",
            },
            {
                "page_number": 1,
                "category": "table",
                "bbox": {"x": 292.6125, "y": 330.20000000000005, "w": 321.17249999999996, "h": 65.70979999999997},
                "text": '<table><tr><th colspan="1" rowspan="1">Per</th><th colspan="1" '
                'rowspan="1">Rate</th></tr><tr><td colspan="1" rowspan="1">$1,000</td><td '
                'colspan="1" rowspan="1">$2.66</td></tr></table>',
                "confidence": None,
                "page_width": 1275,
                "page_height": 1651,
                "dpi": 150,
                "coordinate_space": "raster_pixels",
            },
        ],
        "structured": {
            "schema_version": 1,
            "kind": "canonical_document",
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "markdown": "Lemonade Insurance Company Homeowners Program Arizona "
                    "Edition April 2024\n"
                    "\n"
                    "**Coverage D Increased Limit (Loss of Use) All Forms**\n"
                    "\n"
                    '<table><tr><th colspan="1" rowspan="1">Per</th><th '
                    'colspan="1" rowspan="1">Rate</th></tr><tr><td colspan="1" '
                    'rowspan="1">$1,000</td><td colspan="1" '
                    'rowspan="1">$2.66</td></tr></table>',
                    "width": 1275.0,
                    "height": 1651.0,
                    "coordinate_space": "raster_pixels",
                    "dpi": 150,
                    "blocks": [
                        {
                            "type": "text",
                            "markdown": "Lemonade Insurance Company Homeowners Program Arizona Edition April 2024",
                            "bbox": {
                                "x": 107.10000000000001,
                                "y": 121.1834,
                                "w": 295.03499999999997,
                                "h": 117.38609999999997,
                            },
                            "reading_order": 0,
                            "source_meta": {
                                "model_category": "Text",
                                "normalized_bbox_xyxy": [0.084, 0.0734, 0.3154, 0.1445],
                            },
                        },
                        {
                            "type": "text",
                            "markdown": "**Coverage D Increased Limit (Loss of Use)** **All Forms**",
                            "bbox": {"x": 140.76, "y": 273.4056, "w": 377.145, "h": 52.99709999999999},
                            "reading_order": 1,
                            "source_meta": {
                                "model_category": "Text",
                                "normalized_bbox_xyxy": [0.1104, 0.1656, 0.4062, 0.1977],
                            },
                        },
                        {
                            "type": "table",
                            "markdown": '<table><tr><th colspan="1" '
                            'rowspan="1">Per</th><th colspan="1" '
                            'rowspan="1">Rate</th></tr><tr><td colspan="1" '
                            'rowspan="1">$1,000</td><td colspan="1" '
                            'rowspan="1">$2.66</td></tr></table>',
                            "bbox": {
                                "x": 292.6125,
                                "y": 330.20000000000005,
                                "w": 321.17249999999996,
                                "h": 65.70979999999997,
                            },
                            "reading_order": 2,
                            "source_meta": {
                                "model_category": "Table",
                                "normalized_bbox_xyxy": [0.2295, 0.2, 0.4814, 0.2398],
                            },
                        },
                    ],
                }
            ],
        },
    },
}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(syntel.time, "sleep", lambda _: None)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> SyntelProvider:
    monkeypatch.setenv("SYNTEL_API_KEY", "test-key-not-real")
    monkeypatch.setenv("SYNTEL_BASE_URL", BASE_URL + "/")
    return SyntelProvider(PROVIDER_NAME, get_pipeline(PIPELINE).config)


def use_transport(provider: SyntelProvider, handler: Any) -> None:
    provider._http = httpx.Client(
        base_url=provider._base_url,
        headers=provider._headers,
        transport=httpx.MockTransport(handler),
    )


def request(tmp_path: Any, name: str = "text_multicolumns__2col.pdf") -> InferenceRequest:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4\n% one page\n%%EOF\n")
    return InferenceRequest(example_id="table/example", source_file_path=str(path), product_type="parse")


def raw(body: dict[str, Any]) -> RawInferenceResult:
    spec = get_pipeline(PIPELINE)
    now = datetime.now()
    return RawInferenceResult(
        request=InferenceRequest(example_id="table/example", source_file_path="example.pdf", product_type="parse"),
        pipeline=spec,
        pipeline_name=spec.pipeline_name,
        product_type="parse",
        raw_output=body,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )


def recorded() -> dict[str, Any]:
    return copy.deepcopy(RECORDED_RESULT)


class Service:
    """A scripted /v1/parse service: one job, a list of job statuses, then the result."""

    def __init__(self, statuses: list[dict[str, Any]], result: httpx.Response | None = None) -> None:
        self.statuses = list(statuses)
        self.result = result or httpx.Response(200, json=recorded())
        self.calls: list[httpx.Request] = []
        self.submitted: list[bytes] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        path = req.url.path.removeprefix("/api")
        if req.method == "POST" and path == "/v1/parse":
            self.submitted.append(req.content)
            return httpx.Response(202, json={"success": True, "data": {"job_id": JOB_ID, "status": "queued"}})
        if req.method == "GET" and path == f"/v1/jobs/{JOB_ID}":
            job = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return httpx.Response(200, json={"success": True, "data": {"id": JOB_ID, **job}})
        if req.method == "GET" and path == f"/v1/parse/result/{JOB_ID}":
            return self.result
        if req.method == "POST" and path == f"/v1/jobs/{JOB_ID}/cancel":
            return httpx.Response(200, json={"success": True, "data": {"id": JOB_ID, "status": "cancelled"}})
        return httpx.Response(404, json={"detail": "not found"})


def form_field(body: bytes, name: str) -> str:
    marker = f'name="{name}"\r\n\r\n'.encode()
    start = body.index(marker) + len(marker)
    return body[start : body.index(b"\r\n", start)].decode()


# -- Registration and configuration -------------------------------------------------------------


def test_pipeline_is_registered_with_the_provider_name() -> None:
    spec = get_pipeline(PIPELINE)
    assert spec.provider_name == PROVIDER_NAME
    assert spec.config["parser"] == "advanced"
    assert spec.per_file_timeout is not None
    assert spec.config["job_timeout"] < spec.per_file_timeout


def test_configuration(provider: SyntelProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    assert provider._base_url == BASE_URL
    assert provider._parser == "advanced"
    assert provider._cost_per_page_usd is syntel.LIST_PRICE_PER_PAGE_USD is None

    with pytest.raises(ProviderConfigError):
        SyntelProvider(PROVIDER_NAME, {"parser": "fast"})

    monkeypatch.delenv("SYNTEL_BASE_URL")
    assert syntel.DEFAULT_BASE_URL is None  # no local default
    with pytest.raises(ProviderConfigError, match="SYNTEL_BASE_URL"):
        SyntelProvider(PROVIDER_NAME, {})

    monkeypatch.setenv("SYNTEL_BASE_URL", BASE_URL)
    monkeypatch.delenv("SYNTEL_API_KEY")
    with pytest.raises(ProviderConfigError, match="SYNTEL_API_KEY"):
        SyntelProvider(PROVIDER_NAME, {})


# -- The job flow -------------------------------------------------------------------------------


def test_submit_poll_and_fetch_the_recorded_result(provider: SyntelProvider, tmp_path: Any) -> None:
    service = Service([{"status": "queued"}, {"status": "running"}, {"status": "succeeded"}])
    use_transport(provider, service)
    source = request(tmp_path)

    result = provider.run_inference(get_pipeline(PIPELINE), source)

    post = service.calls[0]
    assert post.headers["Authorization"] == "Bearer test-key-not-real"
    assert str(post.url) == f"{BASE_URL}/v1/parse"
    assert form_field(post.content, "parser") == "advanced"
    assert len(form_field(post.content, "idempotency_key")) == 64
    payload = (tmp_path / "text_multicolumns__2col.pdf").read_bytes()
    assert f'filename="{upload_name(payload, ".pdf")}"'.encode() in post.content
    assert b"text_multicolumns" not in post.content
    assert b"Content-Type: application/pdf" in post.content
    assert [call.url.path for call in service.calls[1:]] == [
        f"/api/v1/jobs/{JOB_ID}",
        f"/api/v1/jobs/{JOB_ID}",
        f"/api/v1/jobs/{JOB_ID}",
        f"/api/v1/parse/result/{JOB_ID}",
    ]

    assert result.raw_output["data"] == RECORDED_RESULT["data"]
    assert result.raw_output["job_id"] == JOB_ID
    assert result.raw_output["num_pages"] == 1
    assert result.raw_output["cost_usd"] is None
    assert result.raw_output["cost_per_page_usd"] is None
    assert "test-key-not-real" not in json.dumps(result.raw_output)
    assert provider._jobs == {}


def test_no_cost_is_recorded_unless_a_price_is_configured(provider: SyntelProvider, tmp_path: Any) -> None:
    use_transport(provider, Service([{"status": "succeeded"}]))
    output = provider.run_inference(get_pipeline(PIPELINE), request(tmp_path)).raw_output
    assert output["cost_per_page_usd"] is None
    assert output["cost_usd"] is None


def test_a_configured_price_is_pages_times_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("SYNTEL_API_KEY", "test-key-not-real")
    monkeypatch.setenv("SYNTEL_BASE_URL", BASE_URL)
    priced = SyntelProvider(PROVIDER_NAME, {**get_pipeline(PIPELINE).config, "cost_per_page_usd": 0.004})
    body = recorded()
    body["data"]["total_pages"] = 3
    use_transport(priced, Service([{"status": "succeeded"}], httpx.Response(200, json=body)))

    output = priced.run_inference(get_pipeline(PIPELINE), request(tmp_path)).raw_output

    assert output["num_pages"] == 3
    assert output["cost_per_page_usd"] == pytest.approx(0.004)
    assert output["cost_usd"] == pytest.approx(0.012)
    priced.recompute_cost(output)
    assert output["cost_usd"] == pytest.approx(0.012)


def test_images_keep_their_type_and_get_a_hashed_name(provider: SyntelProvider, tmp_path: Any) -> None:
    service = Service([{"status": "succeeded"}])
    use_transport(provider, service)
    provider.run_inference(get_pipeline(PIPELINE), request(tmp_path, "chart__bar_scan.JPG"))
    assert b"Content-Type: image/jpeg" in service.submitted[0]
    assert b"chart__bar" not in service.submitted[0]
    with pytest.raises(ProviderPermanentError, match="Unsupported file type"):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path, "notes.docx"))


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, ProviderConfigError),
        (403, ProviderConfigError),
        (429, ProviderRateLimitError),
        (408, ProviderTransientError),
        (500, ProviderTransientError),
        (503, ProviderTransientError),
        (400, ProviderPermanentError),
        (413, ProviderPermanentError),
    ],
)
def test_error_classification(provider: SyntelProvider, tmp_path: Any, status: int, expected: type) -> None:
    use_transport(provider, lambda _: httpx.Response(status, json={"detail": "error"}))
    with pytest.raises(expected):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))


def test_transport_errors_and_bad_json_are_transient(provider: SyntelProvider, tmp_path: Any) -> None:
    def refuse(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    use_transport(provider, refuse)
    with pytest.raises(ProviderTransientError):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))

    use_transport(provider, lambda _: httpx.Response(202, text="truncated"))
    with pytest.raises(ProviderTransientError):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))


def test_rate_limit_while_polling_is_raised_not_retried(provider: SyntelProvider, tmp_path: Any) -> None:
    service = Service([{"status": "running"}])

    def throttle_polls(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            service.calls.append(req)
            return httpx.Response(429, json={"detail": "slow down"})
        return service(req)

    use_transport(provider, throttle_polls)
    with pytest.raises(ProviderRateLimitError):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    assert sum(call.method == "GET" for call in service.calls) == 1


@pytest.mark.parametrize(
    "job, expected",
    [
        (
            {"status": "failed", "error": {"error_type": "PARSE_PROBE_TIMED_OUT", "message": "x", "retryable": True}},
            ProviderTransientError,
        ),
        (
            {"status": "failed", "error": {"error_type": "PARSE_PROBE_FAILED", "message": "x", "retryable": False}},
            ProviderPermanentError,
        ),
        ({"status": "cancelled"}, ProviderPermanentError),
    ],
)
def test_a_finished_job_without_a_result_starts_a_new_job_next_time(
    provider: SyntelProvider, tmp_path: Any, job: dict[str, Any], expected: type
) -> None:
    service = Service([job])
    use_transport(provider, service)
    with pytest.raises(expected):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    service.statuses = [{"status": "succeeded"}]
    provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))

    first, second = (form_field(body, "idempotency_key") for body in service.submitted)
    assert first != second


def test_job_budget_resumes_the_same_job(
    provider: SyntelProvider, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([0.0, 10.0, 20.0, 3000.0])
    monkeypatch.setattr(syntel.time, "monotonic", lambda: next(clock))
    service = Service([{"status": "running"}])
    use_transport(provider, service)
    with pytest.raises(ProviderTransientError, match="still running"):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    assert not any(call.url.path.endswith("/cancel") for call in service.calls)

    monkeypatch.setattr(syntel.time, "monotonic", lambda: 0.0)
    service.statuses = [{"status": "succeeded"}]
    provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    first, second = (form_field(body, "idempotency_key") for body in service.submitted)
    assert first == second


def test_an_expired_result_is_parsed_again(provider: SyntelProvider, tmp_path: Any) -> None:
    gone = httpx.Response(404, json={"error_code": "PARSE_RESULT_EXPIRED"})
    service = Service([{"status": "succeeded"}], gone)
    use_transport(provider, service)
    with pytest.raises(ProviderTransientError, match="expired"):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    service.result = httpx.Response(200, json=recorded())
    provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    first, second = (form_field(body, "idempotency_key") for body in service.submitted)
    assert first != second


def test_cancel_cancels_the_job_in_flight(provider: SyntelProvider, tmp_path: Any) -> None:
    service = Service([{"status": "succeeded"}])
    use_transport(provider, service)
    assert provider.cancel("table/example") is False

    provider._jobs["table/example"] = (JOB_ID, provider._attempt("table/example"))
    assert provider.cancel("table/example") is True
    assert service.calls[-1].method == "POST"
    assert service.calls[-1].url.path == f"/api/v1/jobs/{JOB_ID}/cancel"
    assert provider._jobs == {}


def test_a_new_run_never_resumes_an_old_run_job(provider: SyntelProvider, tmp_path: Any) -> None:
    other_run = SyntelProvider(PROVIDER_NAME, get_pipeline(PIPELINE).config)
    payload = b"%PDF-1.4"
    assert provider._idempotency_key(payload, 0) == provider._idempotency_key(payload, 0)
    assert provider._idempotency_key(payload, 0) != provider._idempotency_key(payload, 1)
    assert provider._idempotency_key(payload, 0) != other_run._idempotency_key(payload, 0)


# -- The runner's per-file timeout: cancel() while a thread still works on the attempt -----------


def test_a_timed_out_thread_stops_and_leaves_the_next_attempt_alone(provider: SyntelProvider, tmp_path: Any) -> None:
    service = Service([{"status": "running"}])
    stale_attempt: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and not stale_attempt:
            # The runner times out while this thread polls: cancel(), then the next attempt
            # registers its own job before the old thread polls again.
            stale_attempt.append(provider._attempt("table/example"))
            provider.cancel("table/example")
            provider._jobs["table/example"] = ("next-job", provider._attempt("table/example"))
        return service(req)

    use_transport(provider, handler)
    with pytest.raises(ProviderPermanentError, match="cancelled by the runner"):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))

    assert stale_attempt == [0]
    assert provider._attempt("table/example") == 1  # moved once, by cancel() alone
    assert provider._jobs == {"table/example": ("next-job", 1)}
    assert sum(call.url.path.endswith("/cancel") for call in service.calls) == 1


def test_the_service_cancelling_an_old_attempt_does_not_move_the_new_one(
    provider: SyntelProvider, tmp_path: Any
) -> None:
    service = Service([{"status": "cancelled"}])

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            # cancel() and the next attempt land between the old thread's check and its poll.
            provider._attempts["table/example"] = 1
            provider._jobs["table/example"] = ("next-job", 1)
        return service(req)

    use_transport(provider, handler)
    with pytest.raises(ProviderPermanentError):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    assert provider._attempt("table/example") == 1
    assert provider._jobs == {"table/example": ("next-job", 1)}


def test_a_timeout_between_retries_still_cancels_the_job(provider: SyntelProvider, tmp_path: Any) -> None:
    service = Service([{"status": "running"}])

    def flaky_poll(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            service.calls.append(req)
            return httpx.Response(503, json={"detail": "unavailable"})
        return service(req)

    use_transport(provider, flaky_poll)
    with pytest.raises(ProviderTransientError):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    # The runner is in its backoff when the per-file timeout fires.
    assert provider.cancel("table/example") is True
    assert service.calls[-1].url.path == f"/api/v1/jobs/{JOB_ID}/cancel"

    use_transport(provider, service)
    service.statuses = [{"status": "succeeded"}]
    provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    first, second = (form_field(body, "idempotency_key") for body in service.submitted)
    assert first != second
    assert provider._jobs == {}


def test_a_job_returned_after_cancel_is_cancelled_and_not_kept(provider: SyntelProvider, tmp_path: Any) -> None:
    service = Service([{"status": "running"}])

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path.endswith("/v1/parse"):
            provider.cancel("table/example")  # the timeout fires while the submit is in flight
        return service(req)

    use_transport(provider, handler)
    with pytest.raises(ProviderPermanentError, match="cancelled by the runner"):
        provider.run_inference(get_pipeline(PIPELINE), request(tmp_path))
    assert service.calls[-1].url.path == f"/api/v1/jobs/{JOB_ID}/cancel"
    assert provider._jobs == {}
    assert provider._attempt("table/example") == 1


# -- Normalization ------------------------------------------------------------------------------


def test_normalize_keeps_the_service_markdown_and_pages(provider: SyntelProvider) -> None:
    result = provider.normalize(raw(recorded()))
    output = result.output

    assert output.markdown == RECORDED_RESULT["data"]["text"]
    assert [page.markdown for page in output.pages] == [RECORDED_RESULT["data"]["structured"]["pages"][0]["markdown"]]
    layout_page = output.layout_pages[0]
    assert (layout_page.width, layout_page.height) == (1275.0, 1651.0)
    assert [item.type for item in layout_page.items] == ["text", "text", "table"]
    table = layout_page.items[2]
    assert table.html.startswith("<table>")
    assert table.bbox is not None
    assert (table.bbox.x, table.bbox.y) == pytest.approx((0.2295, 0.2))


def test_normalize_refuses_other_tiers_and_incomplete_documents(provider: SyntelProvider) -> None:
    body = recorded()
    body["data"]["parser_used"] = "balanced"
    with pytest.raises(ProviderPermanentError, match="Advanced"):
        provider.normalize(raw(body))

    body = recorded()
    del body["data"]["structured"]["pages"][0]["blocks"][0]["source_meta"]
    with pytest.raises(ProviderPermanentError, match="canonical document"):
        provider.normalize(raw(body))


# -- Layout adapter and label mapper ------------------------------------------------------------


def layout(provider: SyntelProvider, body: dict[str, Any]) -> Any:
    result = provider.normalize(raw(body))
    adapter = create_layout_adapter_for_result(result)
    return result, adapter.to_layout_output(result)


def region(category: str, y: float, *, text: str | None = "x") -> dict:
    return {
        "page_number": 1,
        "category": category,
        "bbox": {"x": 100.0, "y": y, "w": 50.0, "h": 20.0},
        "text": text,
        "confidence": None,
        "page_width": 1000,
        "page_height": 1000,
        "dpi": 150,
        "coordinate_space": "raster_pixels",
    }


def test_one_region_is_one_prediction(provider: SyntelProvider) -> None:
    result, output = layout(provider, recorded())
    regions = RECORDED_RESULT["data"]["regions"]

    assert output.model == LayoutDetectionModel.SYNTEL_LAYOUT
    assert (output.image_width, output.image_height) == (1275, 1651)
    assert len(output.predictions) == len(regions)
    assert [p.label for p in output.predictions] == ["Text", "Text", "Text", "Text", "Page-footer", "Text", "Table"]
    first = regions[0]["bbox"]
    assert output.predictions[0].bbox == pytest.approx(
        [first["x"], first["y"], first["x"] + first["w"], first["y"] + first["h"]]
    )
    assert isinstance(output.predictions[-1].content, LayoutTableContent)
    assert isinstance(output.predictions[0].content, LayoutTextContent)

    context = build_mapping_context(result, output)
    mapper = resolve_layout_label_mapper(context)
    canonical = [mapper.to_canonical(p.label, p, context) for p in output.predictions]
    assert canonical == [CanonicalLabel.TEXT] * 4 + [
        CanonicalLabel.PAGE_FOOTER,
        CanonicalLabel.TEXT,
        CanonicalLabel.TABLE,
    ]


def test_page_band_makes_page_numbers_page_furniture(provider: SyntelProvider) -> None:
    body = recorded()
    body["data"]["regions"] = [region("number", 20.0), region("number", 500.0), region("number", 950.0)]
    _, output = layout(provider, body)
    assert [p.label for p in output.predictions] == ["Page-header", "Text", "Page-footer"]


def test_a_region_without_text_carries_no_content(provider: SyntelProvider) -> None:
    body = recorded()
    body["data"]["regions"] = [
        region("picture", 400.0, text=None),
        region("formula", 500.0, text=None),
        region("picture", 700.0),
        region("code", 800.0),
    ]
    _, output = layout(provider, body)
    assert [p.label for p in output.predictions] == ["Picture", "Formula", "Picture", "Code"]
    assert [p.content is None for p in output.predictions] == [True, True, False, False]


def test_a_region_that_encloses_another_is_not_emitted_beside_it(provider: SyntelProvider) -> None:
    figure = region("picture", 100.0)
    figure["bbox"] = {"x": 100.0, "y": 100.0, "w": 400.0, "h": 300.0}
    caption = region("caption", 360.0)
    twin = region("text", 600.0)
    same = region("text", 600.0)
    body = recorded()
    body["data"]["regions"] = [figure, caption, region("text", 450.0), twin, same]
    _, output = layout(provider, body)
    assert [p.provider_metadata["original_category"] for p in output.predictions] == ["caption", "text", "text"]
    assert [p.provider_metadata["order_index"] for p in output.predictions] == [1, 2, 3]


def test_an_empty_region_list_is_an_empty_layout(provider: SyntelProvider) -> None:
    body = recorded()
    body["data"]["regions"] = []
    _, output = layout(provider, body)
    assert output.predictions == []
    assert (output.image_width, output.image_height) == (1275, 1651)

    body["data"]["regions"] = None
    with pytest.raises(ValueError, match="regions"):
        layout(provider, body)


def test_a_result_from_another_tier_is_refused(provider: SyntelProvider) -> None:
    body = recorded()
    body["data"]["parser_used"] = "balanced"
    with pytest.raises(ProviderPermanentError, match="not the Advanced tier"):
        provider.normalize(raw(body))


def test_unknown_labels_are_refused_by_the_mapper(provider: SyntelProvider) -> None:
    result, output = layout(provider, recorded())
    context = build_mapping_context(result, output)
    mapper = resolve_layout_label_mapper(context)
    with pytest.raises(Exception, match="Unknown"):
        mapper.to_canonical("not-a-label", output.predictions[0], context)
