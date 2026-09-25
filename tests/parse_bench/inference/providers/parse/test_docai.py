"""docai provider: pipe tables are swapped for the API's own HTML tables, grounding boxes become
layout pages with canonical labels and page-furniture slots, HTTP statuses map onto the error
taxonomy, connection errors are left to the runner, cost is one credit per page, and the
knowledge base is found by name before it is created."""

from __future__ import annotations

import pytest

from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.docai import (
    DocAIProvider,
    cost_fields,
    layout_pages_from_grounding,
    tables_to_html,
)

_GROUNDING = {
    "pages": [
        {
            "page_number": 1,
            "elements": [
                {"label": "header", "bbox": {"x1": 0.1, "y1": 0.02, "x2": 0.5, "y2": 0.05}, "content": "Annual report"},
                {
                    "label": "paragraph_title",
                    "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.15},
                    "content": "Revenue",
                },
                {
                    "label": "table",
                    "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.5},
                    "content": '<table border="1"><tr><td>Item</td><td colspan="2">Qty</td></tr></table>',
                },
                {"label": "caption", "bbox": {"x1": 0.1, "y1": 0.5, "x2": 0.9, "y2": 0.52}, "content": "Table 1"},
                {"label": "number", "bbox": {"x1": 0.45, "y1": 0.95, "x2": 0.55, "y2": 0.98}, "content": "12"},
            ],
        }
    ]
}
_MARKDOWN = "## Revenue\n\n| Item | Qty |\n| --- | --- |\n| Bolt | 12 |\n"


def test_tables_to_html_swaps_pipe_table_for_grounding_html_with_th_header():
    out = tables_to_html(_MARKDOWN, _GROUNDING)
    assert '<tr><th>Item</th><th colspan="2">Qty</th></tr>' in out
    assert "| Bolt |" not in out


def test_tables_to_html_falls_back_to_pipe_rewrite_when_counts_differ():
    out = tables_to_html(_MARKDOWN, {"pages": []})
    assert "<table>" in out and "<th>Item</th>" in out and "<td>Bolt</td>" in out


def test_layout_pages_have_canonical_labels_and_furniture_slots():
    pages = layout_pages_from_grounding(_GROUNDING)
    assert len(pages) == 1
    assert [item.bbox.label for item in pages[0].items] == [
        "Page-header",
        "Section-header",
        "Table",
        "Caption",
        "Page-footer",
    ]
    assert pages[0].page_header_markdown == "Annual report"
    assert pages[0].printed_page_number == "12"
    seg = pages[0].items[2].bbox
    assert (seg.x, seg.y, seg.w, seg.h) == (0.1, 0.2, pytest.approx(0.8), pytest.approx(0.3))


def test_missing_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("DOCAI_API_KEY", raising=False)
    with pytest.raises(ProviderConfigError):
        DocAIProvider("docai", {})


class _Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        return self._body


def test_status_classification(monkeypatch):
    monkeypatch.setenv("DOCAI_API_KEY", "k")
    p = DocAIProvider("docai", {})
    assert p._check(_Resp(200, {}), "x").status_code == 200
    with pytest.raises(ProviderTransientError):
        p._check(_Resp(503, text="down"), "x")
    with pytest.raises(ProviderPermanentError):
        p._check(_Resp(409, text="busy"), "x")


def test_knowledge_base_is_found_by_name_before_creating(monkeypatch):
    monkeypatch.setenv("DOCAI_API_KEY", "k")
    p = DocAIProvider("docai", {})
    calls = []

    def fake_req(method, path, **kw):
        calls.append((method, path))
        if method == "GET":
            return _Resp(200, {"knowledge_bases": [{"id": "kb-1", "name": "parsebench"}]})  # any case
        raise AssertionError("must not create")

    monkeypatch.setattr(p, "_req", fake_req)
    assert p._knowledge_base() == "kb-1"
    assert p._knowledge_base() == "kb-1"  # cached: one GET in total
    assert calls == [("GET", "/v1/knowledge-bases")]


def test_connection_errors_are_transient_and_not_retried(monkeypatch):
    import httpx

    monkeypatch.setenv("DOCAI_API_KEY", "k")
    p = DocAIProvider("docai", {})
    calls = []

    class _Client:
        def request(self, method, path, **kw):
            calls.append(path)
            raise httpx.ConnectError("dns")

    p._http = _Client()
    with pytest.raises(ProviderTransientError):
        p._req("GET", "/v1/knowledge-bases")
    assert calls == ["/v1/knowledge-bases"]  # one attempt; the runner owns retries


def test_cost_is_one_credit_per_page(monkeypatch):
    assert cost_fields({"pages": [{}, {}, {}]}, 0.01) == {
        "num_pages": 3,
        "credits_used": 3,
        "cost_usd": pytest.approx(0.03),
        "cost_per_page_usd": 0.01,
    }
    assert cost_fields({"pages": []}, 0.01) == {}
    monkeypatch.setenv("DOCAI_API_KEY", "k")
    monkeypatch.delenv("DOCAI_CREDIT_RATE_USD", raising=False)
    assert DocAIProvider("docai", {})._credit_rate == 0.01
    assert DocAIProvider("docai", {"credit_rate_usd": 0.02})._credit_rate == 0.02


class _FakeAPI:
    """Just enough of the DocAI API for run_inference; ``fail`` names paths that raise once."""

    def __init__(self, job_status="completed", existing=None, fail=()):
        self.calls, self.job_status, self.existing, self.fail = [], job_status, existing or [], set(fail)

    def __call__(self, method, path, **kw):
        self.calls.append((method, path.split("?")[0]))
        key = path.split("?")[0]
        if key in self.fail:
            self.fail.discard(key)
            raise ProviderTransientError(f"blip on {key}")
        if key == "/v1/knowledge-bases":
            return _Resp(200, {"knowledge_bases": [{"id": "kb", "name": "ParseBench"}]})
        if key == "/v1/files" and method == "GET":
            return _Resp(200, {"files": self.existing})
        if key == "/v1/files" and method == "POST":
            self.uploaded_name = kw["files"]["file"][0]
            return _Resp(200, {"file": {"id": "f-new"}, "parse_job": {"job_id": "j-new"}})
        if key.endswith("/jobs"):
            return _Resp(200, {"jobs": [{"job_id": "j-old", "kind": "parse", "status": self.job_status}]})
        if "/jobs/" in key:
            return _Resp(200, {"job": {"status": "completed", "result": {}}})
        if key.endswith("/artefacts"):
            return _Resp(
                200, {"artifacts": [{"artifact_type": "parse", "links": {"result_md": "/md", "grounding_json": "/g"}}]}
            )
        if key == "/md":
            return _Resp(200, text="# Hi")
        if key == "/g":
            return _Resp(200, {"pages": [{"page_number": 1, "elements": []}]})
        raise AssertionError(f"unexpected {method} {key}")


def _run(monkeypatch, tmp_path, api):
    from parse_bench.schemas.pipeline_io import InferenceRequest
    from parse_bench.schemas.product import ProductType

    monkeypatch.setenv("DOCAI_API_KEY", "k")
    p = DocAIProvider("docai", {"poll_seconds": 0})
    monkeypatch.setattr(p, "_req", api)
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4")
    req = InferenceRequest(
        example_id="chart/secret_report_p9", source_file_path=str(src), product_type=ProductType.PARSE
    )
    return p, req


def test_upload_name_hides_the_example_id(monkeypatch, tmp_path):
    api = _FakeAPI()
    p, req = _run(monkeypatch, tmp_path, api)
    raw = p.run_inference(p_spec(), req)
    assert "secret" not in api.uploaded_name and api.uploaded_name.endswith(".pdf")
    assert raw.raw_output["cost_usd"] == pytest.approx(0.01)


def test_running_job_from_earlier_run_is_resumed_not_deleted(monkeypatch, tmp_path):
    api = _FakeAPI(job_status="running", existing=[{"id": "f-old", "filename": None}])
    p, req = _run(monkeypatch, tmp_path, api)
    import hashlib

    api.existing[0]["filename"] = hashlib.sha256(req.example_id.encode()).hexdigest()[:16] + ".pdf"
    raw = p.run_inference(p_spec(), req)
    assert raw.raw_output["job_id"] == "j-old"
    assert not [c for c in api.calls if c[0] in ("DELETE", "POST")]


def test_retry_after_a_blip_reuses_the_finished_upload(monkeypatch, tmp_path):
    api = _FakeAPI(fail={"/v1/files/f-new/artefacts"})
    p, req = _run(monkeypatch, tmp_path, api)
    with pytest.raises(ProviderTransientError):
        p.run_inference(p_spec(), req)
    p.run_inference(p_spec(), req)  # the runner's retry
    assert [c for c in api.calls if c[0] in ("DELETE", "POST")] == [("POST", "/v1/files")]  # one upload, one charge


def p_spec():
    from parse_bench.schemas.pipeline import PipelineSpec
    from parse_bench.schemas.product import ProductType

    return PipelineSpec(pipeline_name="docai_default", provider_name="docai", product_type=ProductType.PARSE)
