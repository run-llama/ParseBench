"""X2Knowledge provider against a local mock of its public endpoint (real sockets, 127.0.0.1 only).

Covers one HTTP attempt per page and the exact request shape, the status / timeout / reply-contract
error taxonomy, multi-page splitting with concurrent pages kept in page order, normalize() and the
raw_output whitelist, cost, cancellation, and the shared runner (not the provider) retrying a
transient page error.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfReader, PdfWriter

import parse_bench.inference.runner as runner_module
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse import x2knowledge as x2knowledge_module
from parse_bench.inference.providers.parse.x2knowledge import X2KnowledgeProvider
from parse_bench.inference.providers.registry import create_provider
from parse_bench.inference.runner import InferenceRunner
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 24
_PATH = re.compile(r"/s/([a-z0-9_]+)/v1/chat/completions")
_DATA_URL = re.compile(r"data:([a-z]+/[a-z]+);base64,([A-Za-z0-9+/=]+)")
# Text only the mock's error bodies contain: it must never reach an exception message.
_BODY_ONLY = "portal detail that must not leak"
_PAGE_FIELDS = {"object", "version", "page", "markdown", "layout", "timing", "warnings"}

# scenario -> (status, error code, Retry-After)
_ERRORS: dict[str, tuple[int, str, str | None]] = {
    "rate_limited": (429, "rate_limited", "30"),
    "concurrency_limit": (429, "concurrency_limit", "5"),
    "e500": (500, "internal_error", None),
    "e502": (502, "upstream_model_error", None),
    "e503": (503, "server_busy", "30"),
    "e504": (504, "page_timeout", None),
    "e400": (400, "invalid_request", None),
    "e401": (401, "invalid_api_key", None),
    "e403": (403, "insufficient_quota", None),
    "e404": (404, "model_not_found", None),
    "e413": (413, "payload_too_large", None),
    "e408": (408, "request_timeout", None),
}


def _layout() -> list[dict[str, Any]]:
    return [
        {"label": "Title", "bbox": [0.1, 0.05, 0.5, 0.02], "order": 0, "text": "Title", "score": 0.97, "debug": "x"},
        {"label": "Text", "bbox": [0.1, 0.1, 0.8, 0.1], "order": 1, "text": "Body text", "score": 0.9},
        {"label": "Picture", "bbox": [0.1, 0.25, 0.4, 0.2], "order": 2, "text": "Logo 2024", "score": None},
        {
            "label": "Table",
            "bbox": [0.1, 0.5, 0.8, 0.2],
            "order": 3,
            "text": "a b",
            "html": "<table><tr><td>a</td><td>b</td></tr></table>",
            "score": 0.88,
        },
        {"label": "Table", "bbox": [0.1, 0.75, 0.8, 0.1], "order": 4, "text": "c d", "score": 0.8},
    ]


def _page(markdown: str = "# Title\n\nBody text", layout: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A x2knowledge.page as the service returns it, service-internal keys included."""
    return {
        "object": "x2knowledge.page",
        "version": "v1-test",
        "page": {"index": 1, "width_px": 1000, "height_px": 2000, "dpi": 200, "internal": "x"},
        "markdown": markdown,
        "layout": _layout() if layout is None else layout,
        "timing": {"seconds": 12.5},
        "warnings": [],
        "execution": {"attempts": [{"number": 1, "status": "ok"}], "retries": 0, "deadline_seconds": 1800},
        "internal_note": "x",
    }


def _completion(content: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-x2knowledge-test",
            "object": "chat.completion",
            "created": 0,
            "model": "x2knowledge-parse-v1",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    ).encode()


def _bad_element(**changes: Any) -> dict[str, Any]:
    return {"label": "Text", "bbox": [0.1, 0.1, 0.2, 0.1], "order": 0, "text": "t", "score": 0.5, **changes}


# scenario -> a x2knowledge.page that breaks the contract in one way
_CONTRACT_VIOLATIONS: dict[str, dict[str, Any]] = {
    "bad_label": _page(layout=[_bad_element(label="Chart")]),
    "bbox_range": _page(layout=[_bad_element(bbox=[0.1, 0.1, 1.5, 0.1])]),
    "bbox_outside": _page(layout=[_bad_element(bbox=[0.7, 0.1, 0.5, 0.1])]),
    "bad_score": _page(layout=[_bad_element(score=1.5)]),
    "bad_text": _page(layout=[_bad_element(text=42)]),
    "wrong_object": {**_page(), "object": "chat.completion"},
    "no_geometry": {**_page(), "page": {"index": 1}},
}


@dataclass
class _Request:
    scenario: str
    path: str
    headers: dict[str, str]
    body: Any
    mime: str | None = None
    data: bytes | None = None
    page_id: str | None = None  # "image", or the width of the single PDF page sent
    pdf_pages: int | None = None
    status: int = 0


class MockX2KnowledgeAPI:
    """Stdlib HTTP server emulating ``POST {base_url}/chat/completions`` of the public API.

    The scenario is the base URL's path prefix (``/s/<scenario>/v1``). Every request is recorded
    (headers, parsed body, decoded page, reply status) so tests can assert the number of attempts
    and the request shape.
    """

    def __init__(self) -> None:
        self.requests: list[_Request] = []
        self.completed: list[str] = []  # page ids in the order their replies were sent
        self.release = threading.Event()  # unblocks the "slow" and "hold" scenarios
        self.barrier = threading.Barrier(3)
        self._lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self._thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    def base_url(self, scenario: str) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/s/{scenario}/v1"

    def close(self) -> None:
        self.release.set()
        self.barrier.abort()
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                match = _PATH.fullmatch(self.path)
                request = _Request(
                    scenario=match.group(1) if match else "",
                    path=self.path,
                    headers={key.lower(): value for key, value in self.headers.items()},
                    body=_json_or_none(raw),
                )
                _decode_page(request)
                with api._lock:
                    api.requests.append(request)
                    nth = sum(1 for r in api.requests if r.scenario == request.scenario)
                status, headers, payload, cut = api.reply(request, nth)
                request.status = status
                try:
                    self.send_response(status)
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload if cut is None else payload[:cut])
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client already gave up (timeout scenarios)
                if cut is not None:
                    self.close_connection = True
                with api._lock:
                    api.completed.append(request.page_id or "")

        return Handler

    def reply(self, request: _Request, nth: int) -> tuple[int, dict[str, str], bytes, int | None]:
        """(status, headers, body, cut) for a request; ``cut`` truncates the body mid-stream."""
        scenario = request.scenario
        if scenario == "flaky" and nth == 1:
            scenario = "e503"
        if scenario in _ERRORS:
            return _error_reply(*_ERRORS[scenario])
        if scenario == "fail_page" and request.page_id == "102":
            return _error_reply(*_ERRORS["e502"])
        if scenario == "redirect":
            return 302, {"Location": self.base_url("ok") + "/chat/completions"}, b"", None
        if scenario in ("slow", "hold"):
            self.release.wait(timeout=5)
        if scenario == "concurrent":
            try:
                self.barrier.wait(timeout=5)  # all three pages must be in flight at once
            except threading.BrokenBarrierError:
                return _error_reply(500, "internal_error", None)
            time.sleep({"101": 0.2, "102": 0.1}.get(request.page_id or "", 0.0))  # finish in reverse order
        if scenario in _CONTRACT_VIOLATIONS:
            return 200, _JSON, _completion(json.dumps(_CONTRACT_VIOLATIONS[scenario])), None
        if scenario == "not_json":
            return 200, {"Content-Type": "text/html"}, b"<html>temporarily unavailable</html>", None
        if scenario == "content_not_json":
            return 200, _JSON, _completion("# markdown instead of a x2knowledge.page"), None
        body = _completion(json.dumps(_page(markdown=f"page {request.page_id}")))
        if scenario == "truncated_json":
            return 200, _JSON, body[: len(body) // 2], None
        if scenario == "truncated_stream":
            return 200, _JSON, body, len(body) // 2
        return 200, _JSON, body, None


_JSON = {"Content-Type": "application/json"}


def _error_reply(status: int, code: str, retry_after: str | None) -> tuple[int, dict[str, str], bytes, None]:
    body: dict[str, Any] = {
        "error": {"message": _BODY_ONLY, "type": "x2knowledge_error", "code": code, "retryable": status >= 429}
    }
    if status >= 500:
        body["execution"] = {"attempts": [{"number": 1, "status": "failed"}], "retries": 1, "deadline_seconds": 1800}
    headers = dict(_JSON)
    if retry_after:
        headers["Retry-After"] = retry_after
    return status, headers, json.dumps(body).encode(), None


def _json_or_none(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _decode_page(request: _Request) -> None:
    try:
        url = request.body["messages"][0]["content"][0]["image_url"]["url"]
        match = _DATA_URL.fullmatch(url)
    except (KeyError, IndexError, TypeError):
        return
    if not match:
        return
    request.mime, request.data = match.group(1), base64.b64decode(match.group(2))
    if request.mime == "application/pdf":
        reader = PdfReader(io.BytesIO(request.data))
        request.pdf_pages = len(reader.pages)
        request.page_id = str(int(reader.pages[0].mediabox.width))
    else:
        request.page_id = "image"


@pytest.fixture
def x2knowledge_api() -> Iterator[MockX2KnowledgeAPI]:
    api = MockX2KnowledgeAPI()
    try:
        yield api
    finally:
        api.close()


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No proxy between the provider and 127.0.0.1, and no price unless a test sets one."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("X2KNOWLEDGE_PRICE_PER_PAGE_USD", raising=False)
    monkeypatch.setenv("X2KNOWLEDGE_API_KEY", "test-key")


def _provider(
    monkeypatch: pytest.MonkeyPatch, api: MockX2KnowledgeAPI, scenario: str, **config: Any
) -> X2KnowledgeProvider:
    monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", api.base_url(scenario))
    return X2KnowledgeProvider("x2knowledge", {"model": "x2knowledge-parse-v1", "timeout_s": 10, **config})


def _run(provider: X2KnowledgeProvider, src: Path, example_id: str = "doc-1") -> RawInferenceResult:
    pipeline = PipelineSpec(pipeline_name="x2knowledge_v1", provider_name="x2knowledge", product_type=ProductType.PARSE)
    request = InferenceRequest(example_id=example_id, source_file_path=str(src), product_type=ProductType.PARSE)
    return provider.run_inference(pipeline, request)


def _file(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _pdf(tmp_path: Path, widths: tuple[int, ...] = (101, 102, 103)) -> Path:
    writer = PdfWriter()
    for width in widths:
        writer.add_blank_page(width=width, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return _file(tmp_path, "doc.pdf", buffer.getvalue())


# ---- configuration ----------------------------------------------------------------


def test_missing_api_key_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("X2KNOWLEDGE_API_KEY", raising=False)
    monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", "https://api.example.test/v1")
    with pytest.raises(ProviderConfigError, match="X2KNOWLEDGE_API_KEY"):
        X2KnowledgeProvider("x2knowledge", {})


@pytest.mark.parametrize(
    "base_url",
    [
        "api.example.test/v1",
        "ftp://api.example.test/v1",
        "https:///v1",
        "https://api.example.test/v1?tier=1",
        "https://api.example.test/v1#frag",
        "https://user:secret@api.example.test/v1",
        "https://api.example.test:notaport/v1",
    ],
)
def test_base_url_must_be_an_http_api_root(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", base_url)
    with pytest.raises(ProviderConfigError, match="X2KNOWLEDGE_BASE_URL") as excinfo:
        X2KnowledgeProvider("x2knowledge", {})
    assert "secret" not in str(excinfo.value)


@pytest.mark.parametrize("unset", ["missing", "empty"])
def test_without_a_base_url_the_public_entry_is_used(monkeypatch: pytest.MonkeyPatch, unset: str) -> None:
    if unset == "missing":
        monkeypatch.delenv("X2KNOWLEDGE_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", "")
    provider = X2KnowledgeProvider("x2knowledge", {})
    assert provider._base_url == "https://103.118.252.103/v1"


def test_pipeline_base_url_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", "not a url")
    provider = X2KnowledgeProvider("x2knowledge", {"base_url": "https://api.example.test/v1/"})
    assert provider._base_url == "https://api.example.test/v1"


@pytest.mark.parametrize(
    "config",
    [
        {"page_workers": 0},
        {"page_workers": 9},
        {"page_workers": "4"},
        {"page_workers": True},
        {"timeout_s": 0},
        {"timeout_s": "soon"},
        {"price_per_page_usd": -0.01},
        {"price_per_page_usd": "free"},
    ],
    ids=lambda config: "-".join(f"{key}={value!r}" for key, value in config.items()),
)
def test_invalid_settings_are_config_errors(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]) -> None:
    monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", "https://api.example.test/v1")
    with pytest.raises(ProviderConfigError):
        X2KnowledgeProvider("x2knowledge", config)


# ---- request shape -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "data", "mime"),
    [("page.png", _PNG, "image/png"), ("page.jpg", _JPEG, "image/jpeg"), ("page.JPEG", _JPEG, "image/jpeg")],
    ids=["png", "jpg", "JPEG"],
)
def test_image_is_sent_as_is_in_the_single_image_url_part(
    x2knowledge_api: MockX2KnowledgeAPI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    data: bytes,
    mime: str,
) -> None:
    _run(_provider(monkeypatch, x2knowledge_api, "ok"), _file(tmp_path, name, data))

    [request] = x2knowledge_api.requests
    assert request.path == "/s/ok/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["content-type"] == "application/json"
    body = request.body
    assert set(body) == {"model", "stream", "messages"}  # no options, no sampling knobs
    assert body["model"] == "x2knowledge-parse-v1"
    assert body["stream"] is False
    [message] = body["messages"]
    assert set(message) == {"role", "content"} and message["role"] == "user"
    [part] = message["content"]  # exactly one part: no text options, no category hints
    assert part == {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}
    assert request.mime == mime and request.data == data


def test_requests_use_tcp_keepalive(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The page goes out through a transport with TCP keepalive, so a dead connection is noticed in minutes."""
    import httpx

    transports: list[dict[str, Any]] = []

    class RecordingTransport(httpx.HTTPTransport):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            transports.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "HTTPTransport", RecordingTransport)
    _run(_provider(monkeypatch, x2knowledge_api, "ok"), _file(tmp_path, "page.png", _PNG))

    assert len(x2knowledge_api.requests) == 1
    [kwargs] = transports
    options = kwargs["socket_options"]
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
    probe = socket.socket()
    try:
        for option in options:  # every option is one this platform's sockets accept
            probe.setsockopt(*option)
        assert probe.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
        for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 5)):
            if hasattr(socket, name):
                assert probe.getsockopt(socket.IPPROTO_TCP, getattr(socket, name)) == value
    finally:
        probe.close()


@pytest.mark.parametrize(
    ("environment", "proxy"),
    [
        ({}, None),
        ({"HTTPS_PROXY": "http://proxy.example.test:3128"}, "http://proxy.example.test:3128"),
        ({"ALL_PROXY": "proxy.example.test:3128"}, "http://proxy.example.test:3128"),
        ({"HTTPS_PROXY": "http://proxy.example.test:3128", "NO_PROXY": "localhost, 103.118.252.103"}, None),
        ({"HTTPS_PROXY": "http://proxy.example.test:3128", "NO_PROXY": "*"}, None),
    ],
    ids=["none", "https", "all-without-scheme", "no-proxy-host", "no-proxy-star"],
)
def test_proxy_from_the_environment_still_applies(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], proxy: str | None
) -> None:
    # Only the environment: on macOS and Windows getproxies() also reads the system settings.
    monkeypatch.setattr(
        x2knowledge_module.urllib.request, "getproxies", x2knowledge_module.urllib.request.getproxies_environment
    )
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert x2knowledge_module._environment_proxy("https://103.118.252.103/v1/chat/completions") == proxy


def test_single_page_pdf_is_sent_unchanged(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = _pdf(tmp_path, (150,))

    raw = _run(_provider(monkeypatch, x2knowledge_api, "ok"), src)

    [request] = x2knowledge_api.requests
    assert request.mime == "application/pdf"
    assert request.data == src.read_bytes()
    assert raw.raw_output["num_pages"] == 1


def test_redirect_is_not_followed(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ProviderConfigError, match="HTTP 302"):
        _run(_provider(monkeypatch, x2knowledge_api, "redirect"), _file(tmp_path, "page.png", _PNG))

    assert [r.scenario for r in x2knowledge_api.requests] == ["redirect"]  # the "ok" target was never called


# ---- error taxonomy: one attempt per page, the runner owns retries ---------------------


# scenario -> (error class, parts the message must carry)
_STATUS_CASES: list[tuple[str, type[Exception], list[str]]] = [
    ("rate_limited", ProviderRateLimitError, ["HTTP 429", "code=rate_limited", "Retry-After=30s"]),
    ("concurrency_limit", ProviderRateLimitError, ["HTTP 429", "code=concurrency_limit", "Retry-After=5s"]),
    ("e500", ProviderTransientError, ["HTTP 500", "code=internal_error"]),
    ("e502", ProviderTransientError, ["HTTP 502", "code=upstream_model_error"]),
    ("e503", ProviderTransientError, ["HTTP 503", "code=server_busy", "Retry-After=30s"]),
    ("e504", ProviderTransientError, ["HTTP 504", "code=page_timeout"]),
    ("e408", ProviderTransientError, ["HTTP 408", "code=request_timeout"]),
    ("e401", ProviderConfigError, ["HTTP 401", "code=invalid_api_key"]),
    ("e404", ProviderConfigError, ["HTTP 404", "code=model_not_found"]),
    ("e400", ProviderPermanentError, ["HTTP 400", "code=invalid_request"]),
    ("e403", ProviderPermanentError, ["HTTP 403", "code=insufficient_quota"]),
    ("e413", ProviderPermanentError, ["HTTP 413", "code=payload_too_large"]),
]


@pytest.mark.parametrize(("scenario", "error", "expected"), _STATUS_CASES, ids=[case[0] for case in _STATUS_CASES])
def test_each_status_raises_its_class_after_exactly_one_attempt(
    x2knowledge_api: MockX2KnowledgeAPI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scenario: str,
    error: type[Exception],
    expected: list[str],
) -> None:
    with pytest.raises(error) as excinfo:
        _run(_provider(monkeypatch, x2knowledge_api, scenario), _file(tmp_path, "page.png", _PNG))

    assert len(x2knowledge_api.requests) == 1  # the provider never retries
    message = str(excinfo.value)
    assert all(part in message for part in expected), message
    assert _BODY_ONLY not in message and "attempts" not in message  # no body, no execution metadata


def test_timeout_is_transient_after_exactly_one_attempt(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _provider(monkeypatch, x2knowledge_api, "slow", timeout_s=0.3)
    started = time.monotonic()

    with pytest.raises(ProviderTransientError, match="timed out"):
        _run(provider, _file(tmp_path, "page.png", _PNG))

    assert time.monotonic() - started < 3
    assert len(x2knowledge_api.requests) == 1


@pytest.mark.parametrize(
    "scenario",
    ["truncated_json", "truncated_stream", "not_json", "content_not_json", *_CONTRACT_VIOLATIONS],
)
def test_malformed_replies_are_transient(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> None:
    with pytest.raises(ProviderTransientError) as excinfo:
        _run(_provider(monkeypatch, x2knowledge_api, scenario), _file(tmp_path, "page.png", _PNG))

    assert len(x2knowledge_api.requests) == 1
    assert "Chart" not in str(excinfo.value)  # reply content is never echoed


def test_connection_refused_is_transient(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    api = MockX2KnowledgeAPI()
    base_url = api.base_url("ok")
    api.close()  # nothing listens on that port any more
    monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", base_url)

    with pytest.raises(ProviderTransientError, match="transport error"):
        _run(X2KnowledgeProvider("x2knowledge", {"timeout_s": 5}), _file(tmp_path, "page.png", _PNG))


# ---- input handling -----------------------------------------------------------------


def test_unsupported_missing_or_broken_input_is_permanent_and_never_sent(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _provider(monkeypatch, x2knowledge_api, "ok")

    with pytest.raises(ProviderPermanentError, match="PDF, PNG and JPEG"):
        _run(provider, _file(tmp_path, "doc.docx", b"PK\x03\x04"))
    with pytest.raises(ProviderPermanentError, match="not found"):
        _run(provider, tmp_path / "missing.pdf")
    with pytest.raises(ProviderPermanentError, match="split"):
        _run(provider, _file(tmp_path, "broken.pdf", b"not a pdf"))

    assert x2knowledge_api.requests == []


def test_page_over_the_request_limit_is_permanent_and_never_sent(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(x2knowledge_module, "_MAX_REQUEST_BYTES", 64)

    with pytest.raises(ProviderPermanentError, match="32 MiB"):
        _run(_provider(monkeypatch, x2knowledge_api, "ok"), _file(tmp_path, "page.png", _PNG))

    assert x2knowledge_api.requests == []


def test_non_parse_requests_are_rejected(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _provider(monkeypatch, x2knowledge_api, "ok")
    pipeline = PipelineSpec(pipeline_name="x2knowledge_v1", provider_name="x2knowledge", product_type=ProductType.PARSE)
    request = InferenceRequest(
        example_id="doc-1",
        source_file_path=str(_file(tmp_path, "page.png", _PNG)),
        product_type=ProductType.EXTRACT,
    )
    with pytest.raises(ProviderPermanentError, match="PARSE"):
        provider.run_inference(pipeline, request)
    assert x2knowledge_api.requests == []


# ---- multi-page PDFs ------------------------------------------------------------------


def test_multi_page_pdf_is_split_and_sent_concurrently_in_page_order(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The mock only answers once all three pages are in flight, then answers them in reverse.
    raw = _run(_provider(monkeypatch, x2knowledge_api, "concurrent", page_workers=4), _pdf(tmp_path))

    assert [page["markdown"] for page in raw.raw_output["pages"]] == ["page 101", "page 102", "page 103"]
    assert x2knowledge_api.completed == ["103", "102", "101"]
    assert sorted(r.page_id or "" for r in x2knowledge_api.requests) == ["101", "102", "103"]
    assert {r.pdf_pages for r in x2knowledge_api.requests} == {1}  # one single-page PDF per request
    assert {r.status for r in x2knowledge_api.requests} == {200}
    assert raw.raw_output["num_pages"] == raw.raw_output["num_api_calls"] == 3


def test_one_failing_page_fails_the_file_and_later_pages_are_not_sent(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ProviderTransientError, match="HTTP 502"):
        _run(_provider(monkeypatch, x2knowledge_api, "fail_page", page_workers=1), _pdf(tmp_path))

    assert [r.page_id for r in x2knowledge_api.requests] == ["101", "102"]


def test_cancel_stops_the_remaining_pages_of_a_timed_out_file(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _provider(monkeypatch, x2knowledge_api, "hold", page_workers=1)
    src = _pdf(tmp_path)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            _run(provider, src, example_id="doc-7")
        except Exception as error:  # collected for the assertion below
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 5
    while not x2knowledge_api.requests and time.monotonic() < deadline:
        time.sleep(0.01)

    assert provider.cancel("doc-7") is True
    assert provider.cancel("another-doc") is False
    x2knowledge_api.release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ProviderTransientError)
    assert [r.page_id for r in x2knowledge_api.requests] == ["101"]
    assert provider._active == {}


# ---- normalize and raw_output -----------------------------------------------------------


def test_normalize_builds_pages_markdown_and_one_layout_item_per_element(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _provider(monkeypatch, x2knowledge_api, "ok")
    raw = _run(provider, _pdf(tmp_path, (101, 102)))

    output = provider.normalize(raw).output

    assert [page.page_index for page in output.pages] == [0, 1]
    assert [page.markdown for page in output.pages] == ["page 101", "page 102"]
    assert output.markdown == "page 101\n\npage 102"
    assert [lp.page_number for lp in output.layout_pages] == [1, 2]
    for layout_page in output.layout_pages:
        assert (layout_page.width, layout_page.height) == (1000.0, 2000.0)
        assert layout_page.md == f"page {100 + layout_page.page_number}"
        assert [item.type for item in layout_page.items] == ["Title", "Text", "Picture", "Table", "Table"]
        assert all(len(item.layout_segments) == 1 for item in layout_page.items)
        for item, element in zip(layout_page.items, _layout(), strict=True):
            segment = item.layout_segments[0]
            assert [segment.x, segment.y, segment.w, segment.h] == pytest.approx(element["bbox"])
            assert (segment.label, segment.confidence) == (element["label"], element["score"])
        title, text, picture, table_html, table_text = layout_page.items
        assert (text.value, text.md, text.html) == ("Body text", "Body text", "")
        assert picture.value == "Logo 2024"
        assert table_html.html == table_html.value == "<table><tr><td>a</td><td>b</td></tr></table>"
        assert (table_text.value, table_text.html) == ("c d", "")


def test_raw_output_keeps_only_documented_page_fields(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _run(_provider(monkeypatch, x2knowledge_api, "ok"), _file(tmp_path, "page.png", _PNG)).raw_output

    assert {key: raw[key] for key in ("provider", "object", "model", "num_pages", "num_api_calls")} == {
        "provider": "x2knowledge",
        "object": "x2knowledge.document",
        "model": "x2knowledge-parse-v1",
        "num_pages": 1,
        "num_api_calls": 1,
    }
    [page] = raw["pages"]
    assert set(page) == _PAGE_FIELDS
    assert page["page"] == {"index": 1, "width_px": 1000, "height_px": 2000, "dpi": 200}
    assert page["timing"] == {"seconds": 12.5}
    assert all(set(element) <= {"label", "bbox", "order", "text", "score", "html"} for element in page["layout"])
    serialized = json.dumps(raw)
    assert "execution" not in serialized and "internal" not in serialized and "debug" not in serialized
    assert not {"cost_usd", "cost_per_page_usd", "num_pages_billed"} & set(raw)  # no price configured


def test_normalize_rejects_payloads_from_other_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", "https://api.example.test/v1")
    provider = X2KnowledgeProvider("x2knowledge", {})
    raw = RawInferenceResult.model_construct(raw_output={"results": {}})

    with pytest.raises(ProviderPermanentError):
        provider.normalize(raw)


# ---- cost -----------------------------------------------------------------------------


def test_cost_is_pages_times_the_configured_price(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _run(_provider(monkeypatch, x2knowledge_api, "ok", price_per_page_usd=0.02), _pdf(tmp_path)).raw_output
    assert raw["num_pages_billed"] == 3
    assert raw["cost_per_page_usd"] == pytest.approx(0.02)
    assert raw["cost_usd"] == pytest.approx(0.06)

    monkeypatch.setenv("X2KNOWLEDGE_PRICE_PER_PAGE_USD", "0.05")  # the environment overrides the config
    priced = _provider(monkeypatch, x2knowledge_api, "ok", price_per_page_usd=0.02)
    saved = {"object": "x2knowledge.document", "pages": [], "num_pages": 4, "cost_usd": 0.08}
    priced.recompute_cost(saved)
    assert saved["cost_usd"] == pytest.approx(0.20)
    priced.recompute_cost(saved)  # idempotent
    assert saved["cost_usd"] == pytest.approx(0.20)


def test_without_a_price_recompute_cost_leaves_recorded_cost_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", "https://api.example.test/v1")
    saved = {"object": "x2knowledge.document", "pages": [], "num_pages": 2, "cost_usd": 0.04}

    X2KnowledgeProvider("x2knowledge", {}).recompute_cost(saved)

    assert saved["cost_usd"] == 0.04 and "cost_per_page_usd" not in saved


# ---- the runner owns retries --------------------------------------------------------------


def test_runner_retries_a_transient_page_error_and_the_file_succeeds(
    x2knowledge_api: MockX2KnowledgeAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner_module, "INITIAL_BACKOFF_S", 0.0)
    monkeypatch.setenv("X2KNOWLEDGE_BASE_URL", x2knowledge_api.base_url("flaky"))
    pipeline = get_pipeline("x2knowledge_v1")
    provider = create_provider(pipeline)
    assert isinstance(provider, X2KnowledgeProvider)
    src = _file(tmp_path, "invoice.png", _PNG)
    output_dir = tmp_path / "out"
    runner = InferenceRunner(
        provider=provider, pipeline=pipeline, output_dir=output_dir, max_concurrent=1, use_rich=False
    )
    try:
        summary = asyncio.run(runner.run_files([src], ProductType.PARSE))
    finally:
        runner.shutdown()

    assert (summary.successful, summary.failed) == (1, 0)
    assert [r.status for r in x2knowledge_api.requests] == [503, 200]  # one attempt per runner try
    saved_raw = (output_dir / "invoice.raw.json").read_text()
    assert "execution" not in saved_raw
    saved = json.loads((output_dir / "invoice.result.json").read_text())
    assert saved["output"]["markdown"] == "page image"
