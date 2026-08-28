from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from google.api_core import exceptions as google_exceptions
from PIL import Image

from parse_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRetryExhaustedError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.google import GoogleProvider
from parse_bench.inference.providers.parse.google_agentic_vision import (
    USER_PROMPT_AGENTIC_VISION_PREFIX,
    classify_gemini_api_exception,
    parse_page_response,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="google_test",
        provider_name="google",
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def _response(text: str | None) -> SimpleNamespace:
    candidates = []
    if text is not None:
        candidates = [
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
                finish_reason=None,
            )
        ]
    return SimpleNamespace(
        candidates=candidates,
        prompt_feedback=SimpleNamespace(block_reason=None),
        usage_metadata=None,
    )


class _Models:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.calls = 0
        self.requests: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        self.requests.append(kwargs)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, SimpleNamespace)
        return response


class _Caches:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            name="cachedContents/document-prefix",
            usage_metadata=SimpleNamespace(total_token_count=2048),
        )

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


def _provider(mode: str, responses: list[object]) -> tuple[GoogleProvider, _Models]:
    provider = object.__new__(GoogleProvider)
    models = _Models(responses)
    provider._client = SimpleNamespace(models=models)
    provider._types = SimpleNamespace(
        Part=SimpleNamespace(
            from_bytes=lambda **kwargs: kwargs,
            from_text=lambda **kwargs: kwargs,
        ),
        GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        Content=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    provider._model = "gemini-3-flash"
    provider._dpi = 144
    provider._max_tokens = 1024
    provider._thinking_level = None
    provider._mode = mode
    provider._bbox_scale = 1000
    provider._layout_system_prompt = "layout system prompt"
    provider._layout_user_prompt = "layout user prompt"
    return provider, models


def _agentic_provider(responses: list[object]) -> tuple[GoogleProvider, _Models]:
    provider, models = _provider("parse_with_layout_agentic_vision", responses)
    provider._types.ToolCodeExecution = lambda: SimpleNamespace()
    provider._types.Tool = lambda **kwargs: SimpleNamespace(**kwargs)
    provider._enable_explicit_context_cache = False
    provider._context_cache_ttl_seconds = 900
    provider._min_cacheable_tokens = 1024
    return provider, models


@pytest.mark.parametrize(
    ("mode", "page_one", "message"),
    [
        ("image", "page one", "returned no text"),
        (
            "parse_with_layout",
            '<div data-bbox="[0,0,1000,1000]" data-label="Text">page one</div>',
            "returned no layout text",
        ),
    ],
)
def test_google_page_two_empty_responses_abort_without_partial_payload(
    mode: str,
    page_one: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered: list[Image.Image] = []
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        assert Path(path) == source
        assert first_page == last_page
        image = Image.new("RGB", (8 + first_page, 8), "white")
        rendered.append(image)
        return [image]

    monkeypatch.setattr("pdf2image.convert_from_path", render_page)
    provider, models = _provider(mode, [_response(page_one), *[_response(None) for _ in range(3)]])
    monkeypatch.setattr("time.sleep", lambda delay: None)
    successful_result = None

    with pytest.raises(ProviderRetryExhaustedError, match=message):
        successful_result = provider.run_inference(_pipeline(), _request(source))

    assert successful_result is None
    assert models.calls == 4
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


def test_google_retry_ledger_accounts_billed_malformed_response_usage_and_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "page.png"
    with Image.new("RGB", (8, 8), "white") as image:
        image.save(source)
    malformed = _response(None)
    malformed.usage_metadata = SimpleNamespace(
        prompt_token_count=10,
        tool_use_prompt_token_count=0,
        cached_content_token_count=0,
        candidates_token_count=4,
        thoughts_token_count=1,
        total_token_count=15,
    )
    success = _response("parsed")
    success.usage_metadata = SimpleNamespace(
        prompt_token_count=11,
        tool_use_prompt_token_count=0,
        cached_content_token_count=0,
        candidates_token_count=5,
        thoughts_token_count=1,
        total_token_count=17,
    )
    provider, models = _provider("image", [malformed, success])
    monkeypatch.setattr("parse_bench.inference.providers.parse._multipage_image.time.sleep", lambda delay: None)

    result = provider.run_inference(_pipeline(), _request(source))

    assert models.calls == 2
    assert result.raw_output["num_api_calls"] == 2
    assert result.raw_output["input_tokens"] == 21
    assert result.raw_output["output_tokens"] == 9
    assert result.raw_output["thinking_tokens"] == 2
    assert result.raw_output["total_tokens"] == 32
    assert len(result.raw_output["api_attempts"]) == 2
    assert result.raw_output["api_attempts"][0]["status"] == "failed"
    assert result.raw_output["cost_usd"] > 0


def test_google_agentic_page_two_malformed_then_transient_exhausts_one_page_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered: list[Image.Image] = []
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        assert Path(path) == source
        assert first_page == last_page
        image = Image.new("RGB", (8 + first_page, 8), "white")
        rendered.append(image)
        return [image]

    monkeypatch.setattr("pdf2image.convert_from_path", render_page)
    monkeypatch.setattr(
        "parse_bench.inference.providers.parse.google_agentic_vision.time.sleep",
        lambda delay: None,
    )
    valid = '<div data-bbox="[0,0,1000,1000]" data-label="Text">page one</div>'
    malformed_one = _response("not wrapped")
    malformed_one.usage_metadata = SimpleNamespace(
        prompt_token_count=7,
        tool_use_prompt_token_count=0,
        cached_content_token_count=0,
        candidates_token_count=3,
        thoughts_token_count=1,
        total_token_count=11,
    )
    malformed_three = _response("still not wrapped")
    malformed_three.usage_metadata = SimpleNamespace(
        prompt_token_count=8,
        tool_use_prompt_token_count=0,
        cached_content_token_count=0,
        candidates_token_count=4,
        thoughts_token_count=1,
        total_token_count=13,
    )
    provider, models = _agentic_provider(
        [_response(valid), malformed_one, TimeoutError("connection timeout"), malformed_three]
    )

    with pytest.raises(
        ProviderRetryExhaustedError,
        match="Google Agentic Vision page 2 failed after 3 attempts",
    ) as exc_info:
        provider.run_inference(_pipeline(), _request(source))

    assert models.calls == 4
    assert len(models.requests) == 4
    assert len(rendered) == 2
    debug_payload = exc_info.value.debug_payload
    assert isinstance(debug_payload, dict)
    calls = debug_payload["api_calls"]
    assert [call["page_index"] for call in calls] == [0, 1, 1, 1]
    assert [call["attempt"] for call in calls] == [1, 1, 2, 3]
    assert [call["usage"].get("total_tokens") for call in calls] == [None, 11, None, 13]
    assert calls[0]["response"] is not None
    assert calls[1]["response"] is not None
    assert calls[2]["error"]["type"] == "ProviderTransientError"
    assert calls[3]["response"] is not None
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


def test_google_agentic_mixed_failures_can_succeed_on_final_owned_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "page.png"
    with Image.new("RGB", (9, 8), "white") as image:
        image.save(source)
    monkeypatch.setattr(
        "parse_bench.inference.providers.parse.google_agentic_vision.time.sleep",
        lambda delay: None,
    )
    malformed = _response("not wrapped")
    malformed.usage_metadata = SimpleNamespace(
        prompt_token_count=5,
        tool_use_prompt_token_count=0,
        cached_content_token_count=0,
        candidates_token_count=2,
        thoughts_token_count=1,
        total_token_count=8,
    )
    valid = '<div data-bbox="[0,0,1000,1000]" data-label="Text">success</div>'
    successful = _response(valid)
    successful.usage_metadata = SimpleNamespace(
        prompt_token_count=6,
        tool_use_prompt_token_count=0,
        cached_content_token_count=0,
        candidates_token_count=3,
        thoughts_token_count=1,
        total_token_count=10,
    )
    provider, models = _agentic_provider([malformed, TimeoutError("network reset"), successful])

    result = provider.run_inference(_pipeline(), _request(source))

    assert models.calls == 3
    assert result.raw_output["num_api_calls"] == 3
    assert "total_tokens" not in result.raw_output
    assert "cost_usd" not in result.raw_output
    page = result.raw_output["pages"][0]
    assert page["markdown"] == "success"
    calls = page["api_calls"]
    assert [call["attempt"] for call in calls] == [1, 2, 3]
    assert [call["usage"].get("total_tokens") for call in calls] == [8, None, 10]
    assert calls[1]["error"]["type"] == "ProviderTransientError"


@pytest.mark.parametrize(
    "error",
    [
        google_exceptions.InternalServerError("internal"),
        google_exceptions.ServiceUnavailable("unavailable"),
        google_exceptions.GatewayTimeout("deadline"),
        TimeoutError(),
        ConnectionError(),
    ],
)
def test_google_agentic_classifies_real_transient_errors(error: Exception) -> None:
    assert isinstance(classify_gemini_api_exception(error), ProviderTransientError)


@pytest.mark.parametrize(
    "error",
    [
        google_exceptions.BadRequest("bad request"),
        google_exceptions.Unauthorized("unauthorized"),
        google_exceptions.Forbidden("forbidden"),
        google_exceptions.NotFound("not found"),
    ],
)
def test_google_agentic_classifies_real_permanent_errors(error: Exception) -> None:
    assert isinstance(classify_gemini_api_exception(error), ProviderPermanentError)


def test_google_agentic_blank_middle_page_preserves_page_identity_and_raw_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered: list[Image.Image] = []
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 3})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        assert Path(path) == source
        assert first_page == last_page
        image = Image.new("RGB", (8 + first_page, 8), "white")
        rendered.append(image)
        return [image]

    monkeypatch.setattr("pdf2image.convert_from_path", render_page)
    first = '<div data-bbox="[0,0,1000,1000]" data-label="Text">page one</div>'
    third = '<div data-bbox="[0,0,1000,1000]" data-label="Text">page three</div>'
    provider, models = _agentic_provider([_response(first), _response("[]"), _response(third)])

    raw_result = provider.run_inference(_pipeline(), _request(source))
    result = provider.normalize(raw_result)

    assert models.calls == 3
    assert raw_result.raw_output["num_api_calls"] == 3
    assert [page["page_index"] for page in raw_result.raw_output["pages"]] == [0, 1, 2]
    blank = raw_result.raw_output["pages"][1]
    assert blank["items"] == []
    assert blank["markdown"] == ""
    assert blank["raw_content"] == ""
    assert len(blank["api_calls"]) == 1
    assert blank["api_calls"][0]["final_text"] == "[]"
    assert [page.page_index for page in result.output.pages] == [0, 1, 2]
    assert [page.markdown for page in result.output.pages] == ["page one", "", "page three"]
    assert [page.page_number for page in result.output.layout_pages] == [1, 2, 3]
    assert result.output.layout_pages[1].items == []
    assert "exactly []" in USER_PROMPT_AGENTIC_VISION_PREFIX
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize("terminal_failure", [False, True], ids=["success", "failure"])
def test_google_agentic_document_finally_deletes_server_cache(
    terminal_failure: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda path, dpi, first_page, last_page: [Image.new("RGB", (8, 8), "white")],
    )
    monkeypatch.setattr(
        "parse_bench.inference.providers.parse.google_agentic_vision.time.sleep",
        lambda delay: None,
    )
    valid = '<div data-bbox="[0,0,1000,1000]" data-label="Text">page</div>'
    responses = [_response(valid), _response(valid)]
    if terminal_failure:
        responses = [_response(valid), _response("bad"), _response("bad"), _response("bad")]
    provider, _ = _agentic_provider(responses)
    caches = _Caches()
    provider._client.caches = caches
    provider._types.CreateCachedContentConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    provider._enable_explicit_context_cache = True

    if terminal_failure:
        with pytest.raises(ProviderRetryExhaustedError):
            provider.run_inference(_pipeline(), _request(source))
    else:
        result = provider.run_inference(_pipeline(), _request(source))
        assert result.raw_output["explicit_context_cache"]["deleted"] is True

    assert caches.deleted == ["cachedContents/document-prefix"]


@pytest.mark.parametrize(
    "response",
    [
        _response(None),
        _response(""),
        _response("not wrapped"),
        _response("[ ]"),
        _response("[] trailing text"),
    ],
    ids=["missing", "empty", "malformed", "non-exact-array", "array-with-trailing-text"],
)
def test_google_agentic_missing_empty_and_malformed_are_not_blank(response: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="No (valid )?wrapped layout payload"):
        parse_page_response(response)


def test_google_agentic_permanent_api_failure_persists_physical_call() -> None:
    provider, models = _agentic_provider([google_exceptions.BadRequest("bad request")])
    runner = provider._build_agentic_vision_runner(expected_page_calls=1)
    image = Image.new("RGB", (8, 8), "white")

    with pytest.raises(ProviderPermanentError) as caught:
        runner.parse_page(
            page_index=0,
            image=image,
            image_bytes=b"image",
            image_mime_type="image/jpeg",
        )

    payload = caught.value.debug_payload
    assert isinstance(payload, dict)
    assert models.calls == 1
    assert len(payload["api_calls"]) == 1
    assert payload["api_calls"][0]["error"]["type"] == "ProviderPermanentError"
    image.close()


def test_google_agentic_permanent_page_two_failure_keeps_page_one_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda path, dpi, first_page, last_page: [Image.new("RGB", (8, 8), "white")],
    )
    valid = '<div data-bbox="[0,0,1000,1000]" data-label="Text">page one</div>'
    provider, models = _agentic_provider([_response(valid), google_exceptions.BadRequest("bad request")])

    with pytest.raises(ProviderPermanentError) as caught:
        provider.run_inference(_pipeline(), _request(source))

    payload = caught.value.debug_payload
    assert isinstance(payload, dict)
    assert models.calls == 2
    assert len(payload["api_calls"]) == 2
    assert payload["api_calls"][0]["final_text"] == valid
    assert payload["api_calls"][1]["error"]["type"] == "ProviderPermanentError"
