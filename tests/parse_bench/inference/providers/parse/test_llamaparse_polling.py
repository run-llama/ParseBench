"""LlamaParse SDK job creation / polling: timeouts, reconnect, expand, and table-format tri-state."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

import parse_bench.inference.providers.parse.llamaparse as llamaparse_module
from parse_bench.inference.providers.base import ProviderTransientError
from parse_bench.inference.providers.parse.llamaparse import LlamaParseProvider


@pytest.fixture(autouse=True)
def _clean_llama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LLAMA_CLOUD_API_KEY", "LLAMA_CLOUD_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llamaparse_module, "_HAS_V2_SDK", True)
    monkeypatch.setattr(llamaparse_module.time, "sleep", lambda _s: None)


def _provider(**config: Any) -> LlamaParseProvider:
    return LlamaParseProvider("llamaparse", {"api_key": "test-key", "tier": "agentic", **config})


class _FakeResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, mode: str, by_alias: bool) -> dict[str, Any]:
        assert mode == "json"
        assert by_alias is True
        return dict(self._payload)


class _FakeParsing:
    def __init__(self, captured: dict[str, Any], wait_errors: list[Exception] | None = None) -> None:
        self._captured = captured
        self._wait_errors = list(wait_errors or [])

    def create(self, **kwargs: Any) -> Any:
        self._captured["create_kwargs"] = kwargs
        return type("FakeJob", (), {"id": "pjb-1"})()

    def wait_for_completion(self, job_id: str, timeout: float) -> None:
        self._captured.setdefault("wait_calls", []).append((job_id, timeout))
        if self._wait_errors:
            raise self._wait_errors.pop(0)

    def get(self, job_id: str, expand: list[str]) -> _FakeResult:
        self._captured["get"] = (job_id, list(expand))
        return _FakeResult({"job": {"id": job_id, "status": "COMPLETED"}})


def _fake_client_factory(captured: dict[str, Any], wait_errors_per_client: list[list[Exception]] | None = None):
    errors = list(wait_errors_per_client or [])

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.setdefault("clients", []).append(kwargs)
            self.parsing = _FakeParsing(captured, errors.pop(0) if errors else None)
            self.closed = False

        def close(self) -> None:
            self.closed = True
            captured.setdefault("closed", 0)
            captured["closed"] += 1

    return FakeClient


def test_parse_pdf_defaults_polling_timeout_to_1200_and_create_timeout_to_120() -> None:
    provider = _provider()
    captured: dict[str, Any] = {}

    with patch.object(llamaparse_module, "LlamaCloud", _fake_client_factory(captured)):
        payload = provider._parse_pdf("/tmp/sample.pdf")

    assert payload["job"]["id"] == "pjb-1"
    assert captured["create_kwargs"]["timeout"] == 120
    assert "create_timeout" not in provider._sdk_config
    (job_id, timeout) = captured["wait_calls"][0]
    assert job_id == "pjb-1"
    assert 1190 < timeout <= 1200
    assert captured["get"] == ("pjb-1", ["items", "text", "metadata", "debug_logs"])


def test_parse_pdf_forwards_explicit_timeouts() -> None:
    provider = _provider(timeout=30, create_timeout=7)
    captured: dict[str, Any] = {}

    with patch.object(llamaparse_module, "LlamaCloud", _fake_client_factory(captured)):
        provider._parse_pdf("/tmp/sample.pdf")

    assert captured["create_kwargs"]["timeout"] == 7
    assert 29 < captured["wait_calls"][0][1] <= 30


def test_parse_pdf_requests_forms_expand_when_form_pass_is_enabled() -> None:
    provider = _provider(processing_options={"forms": "enrich"})
    captured: dict[str, Any] = {}

    with patch.object(llamaparse_module, "LlamaCloud", _fake_client_factory(captured)):
        provider._parse_pdf("/tmp/sample.pdf")

    assert captured["get"][1] == ["items", "text", "metadata", "debug_logs", "forms"]
    assert "expand" not in captured["create_kwargs"]


def test_wait_for_completion_reconnects_after_transient_failure() -> None:
    provider = _provider()
    captured: dict[str, Any] = {}
    factory = _fake_client_factory(
        captured,
        wait_errors_per_client=[[ConnectionError("connection reset by peer")], []],
    )

    with patch.object(llamaparse_module, "LlamaCloud", factory):
        payload = provider._parse_pdf("/tmp/sample.pdf")

    assert payload["job"]["id"] == "pjb-1"
    # Two clients were created: the original plus one replacement after the transient error.
    assert len(captured["clients"]) == 2
    assert captured["closed"] == 1
    assert [job_id for job_id, _ in captured["wait_calls"]] == ["pjb-1", "pjb-1"]
    # Remaining budget shrinks monotonically rather than restarting from the full timeout.
    assert captured["wait_calls"][1][1] <= captured["wait_calls"][0][1]


def test_wait_for_completion_does_not_reconnect_on_permanent_error() -> None:
    provider = _provider()
    captured: dict[str, Any] = {}
    factory = _fake_client_factory(captured, wait_errors_per_client=[[RuntimeError("job failed: bad input")]])

    with patch.object(llamaparse_module, "LlamaCloud", factory):
        with pytest.raises(llamaparse_module.ProviderPermanentError, match="job_id=pjb-1"):
            provider._parse_pdf("/tmp/sample.pdf")

    assert len(captured["clients"]) == 1


def test_wait_for_completion_gives_up_after_repeated_transient_failures() -> None:
    provider = _provider()
    captured: dict[str, Any] = {}
    factory = _fake_client_factory(
        captured,
        wait_errors_per_client=[[TimeoutError("timed out")] for _ in range(7)],
    )

    with patch.object(llamaparse_module, "LlamaCloud", factory):
        with pytest.raises(ProviderTransientError, match="job_id=pjb-1"):
            provider._parse_pdf("/tmp/sample.pdf")

    # Original + 5 reconnects, then the sixth transient failure is raised.
    assert len(captured["clients"]) == 6


def test_wait_for_completion_exhausted_deadline_is_transient() -> None:
    provider = _provider(timeout=0)
    captured: dict[str, Any] = {}

    with patch.object(llamaparse_module, "LlamaCloud", _fake_client_factory(captured)):
        with pytest.raises(ProviderTransientError, match="Timed out"):
            provider._parse_pdf("/tmp/sample.pdf")

    assert "wait_calls" not in captured


def test_cancelled_error_is_classified_transient() -> None:
    provider = _provider()
    captured: dict[str, Any] = {}
    factory = _fake_client_factory(captured, wait_errors_per_client=[[RuntimeError("request cancelled by caller")]])

    with patch.object(llamaparse_module, "LlamaCloud", factory):
        with pytest.raises(ProviderTransientError, match="cancelled"):
            provider._parse_pdf("/tmp/sample.pdf")


@pytest.mark.parametrize(
    ("config", "explicit", "effective"),
    [
        ({}, None, False),
        ({"output_tables_as_HTML": True}, False, False),
        ({"output_tables_as_HTML": False}, True, True),
        ({"output_options": {"markdown": {"tables": {"output_tables_as_markdown": True}}}}, True, True),
        ({"output_options": {"markdown": {"tables": {"output_tables_as_markdown": False}}}}, False, False),
        (
            {
                "output_tables_as_HTML": False,
                "output_options": {"markdown": {"tables": {"output_tables_as_markdown": False}}},
            },
            False,
            False,
        ),
    ],
)
def test_output_tables_as_markdown_tri_state(config: dict[str, Any], explicit: bool | None, effective: bool) -> None:
    provider = _provider(**config)
    assert provider._explicit_output_tables_as_markdown() is explicit
    assert provider._output_tables_as_markdown() is effective
