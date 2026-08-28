from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from parse_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRetryExhaustedError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._multipage_image import annotate_attempt_costs, run_page_once
from parse_bench.inference.providers.parse.google_agentic_vision import extract_usage_from_response
from parse_bench.inference.runner import InferenceRunner
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

_USAGE = {
    "input_tokens": 1,
    "output_tokens": 1,
    "thinking_tokens": 0,
    "total_tokens": 2,
}


def _pipeline(provider_name: str) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=f"{provider_name}_layout_file_test",
        provider_name=provider_name,
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def _layout_file_provider(module_name: str, class_name: str) -> Any:
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    provider = object.__new__(getattr(module, class_name))
    provider._mode = "parse_with_layout_file"
    provider._model = "test-model"
    provider._dpi = 150
    provider._max_tokens = 100
    provider._bbox_scale = 1000
    provider._layout_system_prompt = "layout system"
    provider._layout_user_prompt = "layout user"
    provider._reasoning_effort = None
    provider._thinking_level = None
    provider._thinking = None
    provider._effort = None
    provider._supports_temperature = True
    provider._enable_explicit_context_cache = False
    provider._get_pricing = lambda: (0.0, 0.0)
    provider._get_context_cache_pricing = lambda: (0.0, 0.0)
    return provider


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("google", "GoogleProvider"),
        ("openai", "OpenAIProvider"),
        ("anthropic", "AnthropicProvider"),
    ],
)
def test_layout_file_page_two_retry_never_replays_page_one(
    module_name: str,
    class_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    monkeypatch.setattr(module, "split_pdf_to_pages", lambda path: [(b"1", 10, 10), (b"2", 10, 10)])
    monkeypatch.setattr("parse_bench.inference.providers.parse._multipage_image.time.sleep", lambda delay: None)
    provider = _layout_file_provider(module_name, class_name)
    calls: list[int] = []

    def parse_page(pdf_bytes: bytes) -> tuple[list[dict[str, object]], str, dict[str, int]]:
        page_number = int(pdf_bytes)
        calls.append(page_number)
        if calls == [1, 2]:
            raise ProviderTransientError("page two timed out")
        text = f'<div data-bbox="[0,0,10,10]" data-label="Text">page {page_number}</div>'
        return ([{"bbox": [0, 0, 10, 10], "label": "Text", "text": f"page {page_number}"}], text, _USAGE)

    provider._parse_pdf_page_with_layout = parse_page
    raw_result = provider.run_inference(_pipeline(module_name), _request(source))

    assert raw_result.raw_output["num_pages"] == 2
    assert raw_result.raw_output["num_api_calls"] == 3
    assert len(raw_result.raw_output["api_attempts"]) == 3
    assert "total_tokens" not in raw_result.raw_output
    assert "cost_usd" not in raw_result.raw_output
    assert "page_usages" not in raw_result.raw_output
    assert calls == [1, 2, 2]


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("google", "GoogleProvider"),
        ("openai", "OpenAIProvider"),
        ("anthropic", "AnthropicProvider"),
    ],
)
def test_layout_file_exhaustion_is_terminal_to_document_runner(
    module_name: str,
    class_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    monkeypatch.setattr(module, "split_pdf_to_pages", lambda path: [(b"1", 10, 10), (b"2", 10, 10)])
    monkeypatch.setattr("parse_bench.inference.providers.parse._multipage_image.time.sleep", lambda delay: None)
    provider = _layout_file_provider(module_name, class_name)
    calls: list[int] = []
    document_attempts = 0

    def parse_page(pdf_bytes: bytes) -> tuple[list[dict[str, object]], str, dict[str, int]]:
        page_number = int(pdf_bytes)
        calls.append(page_number)
        if page_number == 2:
            raise ProviderTransientError("page two unavailable")
        return ([], "[]", _USAGE)

    provider._parse_pdf_page_with_layout = parse_page
    original_run_inference = provider.run_inference

    def counted_run_inference(pipeline: PipelineSpec, request: InferenceRequest):  # type: ignore[no-untyped-def]
        nonlocal document_attempts
        document_attempts += 1
        return original_run_inference(pipeline, request)

    provider.run_inference = counted_run_inference
    runner = object.__new__(InferenceRunner)
    runner.output_dir = tmp_path
    runner.use_rich = False
    runner.job_statuses = {}
    runner.pipeline = _pipeline(module_name)
    runner.provider = provider
    runner._prepare_source_file_for_provider = lambda example_id, path: path
    runner._fetch_parse_job_logs = lambda raw_result, example_id: None
    runner._save_result = lambda raw_result, normalized_result: None

    raw_result, normalized_result, error = runner._process_document(source, "document", ProductType.PARSE)

    assert raw_result is None
    assert normalized_result is None
    assert error is not None and error[2] == ProviderRetryExhaustedError.__name__
    assert document_attempts == 1
    assert calls == [1, 2, 2, 2]
    payload = json.loads((tmp_path / "document.error.raw.json").read_text())
    assert [(attempt["page_number"], attempt["status"]) for attempt in payload["attempts"]] == [
        (1, "succeeded"),
        (2, "failed"),
        (2, "failed"),
        (2, "failed"),
    ]


def _openai_response(content: object = "page") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )


def _anthropic_response(content: list[object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        content=content if content is not None else [SimpleNamespace(type="text", text="page")],
        usage=None,
    )


@pytest.mark.parametrize(
    ("module_name", "class_name", "response"),
    [
        ("google", "GoogleProvider", SimpleNamespace(usage_metadata=None)),
        ("openai", "OpenAIProvider", SimpleNamespace(usage=None)),
        ("anthropic", "AnthropicProvider", SimpleNamespace(usage=None)),
        ("amazon_nova", "AmazonNovaProvider", {}),
    ],
)
def test_missing_success_usage_remains_unknown(module_name: str, class_name: str, response: object) -> None:
    provider_class = getattr(
        importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}"), class_name
    )

    assert provider_class._extract_usage(response) == {}


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("google", "GoogleProvider"),
        ("openai", "OpenAIProvider"),
        ("anthropic", "AnthropicProvider"),
    ],
)
def test_native_file_success_without_usage_keeps_output_and_physical_ledger(
    module_name: str,
    class_name: str,
    tmp_path: Path,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    provider = _layout_file_provider(module_name, class_name)
    provider._mode = "file"
    provider._get_pricing = lambda: None
    provider._parse_pdf_file = lambda path: ("parsed", {})

    result = provider.run_inference(_pipeline(module_name), _request(source))

    assert result.raw_output["pages"][0]["markdown"] == "parsed"
    assert result.raw_output["num_api_calls"] == 1
    assert result.raw_output["api_attempts"] == [{"page_number": 1, "attempt": 1, "status": "succeeded", "stats": {}}]
    assert "input_tokens" not in result.raw_output
    assert "cost_usd" not in result.raw_output


def test_permanent_physical_call_persists_attempt_before_reraising() -> None:
    attempts: list[dict[str, object]] = []

    def fail() -> None:
        raise ProviderPermanentError("submitted bad request", attempt_stats={"input_tokens": 3})

    with pytest.raises(ProviderPermanentError) as caught:
        run_page_once(fail, page_number=2, attempt_ledger=attempts)

    assert attempts == [
        {
            "page_number": 2,
            "attempt": 1,
            "status": "failed",
            "stats": {"input_tokens": 3},
            "error_type": "ProviderPermanentError",
            "error": "submitted bad request",
        }
    ]
    assert caught.value.debug_payload == {"attempts": attempts}


def test_openai_reasoning_is_not_double_billed_and_cached_input_is_not_full_price() -> None:
    module = importlib.import_module("parse_bench.inference.providers.parse.openai")
    provider = object.__new__(module.OpenAIProvider)
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            prompt_tokens_details=SimpleNamespace(cached_tokens=60),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
        )
    )

    usage = provider._extract_usage(response)

    assert usage == {
        "input_tokens": 100,
        "cached_content_tokens": 60,
        "output_tokens": 40,
        "thinking_tokens": 10,
        "total_tokens": 140,
    }
    attempts: list[dict[str, object]] = [{"stats": dict(usage)}]
    annotate_attempt_costs(
        attempts,
        input_rate_per_million=1.0,
        output_rate_per_million=2.0,
        output_tokens_include_thinking=True,
    )
    assert "cost_usd" not in attempts[0]["stats"]

    uncached_attempts: list[dict[str, object]] = [{"stats": {**usage, "cached_content_tokens": 0}}]
    annotate_attempt_costs(
        uncached_attempts,
        input_rate_per_million=1.0,
        output_rate_per_million=2.0,
        output_tokens_include_thinking=True,
    )
    assert uncached_attempts[0]["stats"]["cost_usd"] == pytest.approx(0.00018)


def test_agentic_missing_or_partial_success_usage_remains_incomplete() -> None:
    assert extract_usage_from_response(SimpleNamespace(usage_metadata=None)) == {}
    partial = extract_usage_from_response(SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=7)))

    assert partial == {"input_tokens": 7}


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("google", "GoogleProvider"),
        ("openai", "OpenAIProvider"),
        ("anthropic", "AnthropicProvider"),
        ("amazon_nova", "AmazonNovaProvider"),
    ],
)
def test_unknown_model_pricing_remains_unknown(module_name: str, class_name: str) -> None:
    provider = object.__new__(
        getattr(importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}"), class_name)
    )
    provider._model = "future-unknown-model"

    assert provider._get_pricing() is None


def test_google_unknown_pricing_preserves_complete_usage_without_cost() -> None:
    google = importlib.import_module("parse_bench.inference.providers.parse.google")
    provider = object.__new__(google.GoogleProvider)
    provider._model = "future-unknown-model"

    summary = provider._compute_usage_cost_summary([dict(_USAGE)], num_pages=1)

    assert summary == {
        "input_tokens": 1,
        "tool_use_prompt_tokens": 0,
        "cached_content_tokens": 0,
        "output_tokens": 1,
        "thinking_tokens": 0,
        "total_tokens": 2,
        "num_api_calls": 1,
        "input_tokens_per_page": 1.0,
        "tool_use_prompt_tokens_per_page": 0.0,
        "cached_content_tokens_per_page": 0.0,
        "output_tokens_per_page": 1.0,
    }


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("google", "GoogleProvider"),
        ("openai", "OpenAIProvider"),
        ("anthropic", "AnthropicProvider"),
    ],
)
def test_unknown_pricing_does_not_abort_completed_file_inference(
    module_name: str,
    class_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    monkeypatch.setattr(module, "split_pdf_to_pages", lambda path: [(b"1", 10, 10)])
    provider = _layout_file_provider(module_name, class_name)
    provider._get_pricing = lambda: None
    if module_name == "google":
        provider._get_context_cache_pricing = lambda: None
    provider._parse_pdf_page_with_layout = lambda pdf_bytes: ([], "[]", dict(_USAGE))

    result = provider.run_inference(_pipeline(module_name), _request(source))

    assert result.raw_output["num_api_calls"] == 1
    assert result.raw_output["total_tokens"] == 2
    assert "cost_usd" not in result.raw_output
    assert "cost_per_page_usd" not in result.raw_output


@pytest.mark.parametrize(
    "bad_response", [SimpleNamespace(choices=[], usage=None), _openai_response(None), _openai_response("")]
)
def test_openai_normal_image_rejects_missing_or_empty_text(bad_response: object) -> None:
    provider = object.__new__(importlib.import_module("parse_bench.inference.providers.parse.openai").OpenAIProvider)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: bad_response))
    )
    provider._model = "test-model"
    provider._max_tokens = 100
    provider._reasoning_effort = None
    provider._image_to_base64 = lambda image: "image"

    with pytest.raises(ProviderTransientError, match="no choices|no message content|empty message content"):
        provider._parse_image(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    "bad_response",
    [
        _anthropic_response([]),
        _anthropic_response([SimpleNamespace(type="thinking", thinking="...")]),
        _anthropic_response([SimpleNamespace(type="text", text="")]),
    ],
)
def test_anthropic_normal_image_rejects_missing_or_empty_text(bad_response: object) -> None:
    provider = object.__new__(
        importlib.import_module("parse_bench.inference.providers.parse.anthropic").AnthropicProvider
    )
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: bad_response))
    provider._model = "test-model"
    provider._max_tokens = 100
    provider._thinking = None
    provider._effort = None
    provider._supports_temperature = True
    provider._image_to_base64 = lambda image: "image"

    with pytest.raises(ProviderTransientError, match="no content blocks|no non-empty text content"):
        provider._parse_image(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    ("module_name", "class_name", "response"),
    [
        ("openai", "OpenAIProvider", _openai_response("not layout markup")),
        (
            "anthropic",
            "AnthropicProvider",
            _anthropic_response([SimpleNamespace(type="text", text="not layout markup")]),
        ),
    ],
)
def test_layout_image_rejects_malformed_output_through_response_parser(
    module_name: str,
    class_name: str,
    response: object,
) -> None:
    provider = object.__new__(
        getattr(importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}"), class_name)
    )
    if module_name == "openai":
        provider._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))
        )
        provider._reasoning_effort = None
    else:
        provider._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))
        provider._thinking = None
        provider._effort = None
        provider._supports_temperature = True
    provider._model = "test-model"
    provider._max_tokens = 100
    provider._layout_system_prompt = "system"
    provider._layout_user_prompt = "user"
    provider._image_to_base64 = lambda image: "image"

    with pytest.raises(ProviderTransientError, match="malformed layout output"):
        provider._parse_image_with_layout(Image.new("RGB", (1, 1)))


@pytest.mark.parametrize(
    ("provider_class", "response"),
    [
        (
            "OpenAIProvider",
            _openai_response("[]"),
        ),
        (
            "AnthropicProvider",
            _anthropic_response([SimpleNamespace(type="text", text="[]")]),
        ),
    ],
)
def test_layout_parser_accepts_only_explicit_structured_blank(provider_class: str, response: object) -> None:
    module_name = "openai" if provider_class == "OpenAIProvider" else "anthropic"
    provider = object.__new__(
        getattr(importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}"), provider_class)
    )
    if module_name == "openai":
        provider._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))
        )
        provider._reasoning_effort = None
    else:
        provider._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))
        provider._thinking = None
        provider._effort = None
        provider._supports_temperature = True
    provider._model = "test-model"
    provider._max_tokens = 100
    provider._layout_system_prompt = "system"
    provider._layout_user_prompt = "user"
    provider._image_to_base64 = lambda image: "image"

    items, text, _ = provider._parse_image_with_layout(Image.new("RGB", (1, 1)))

    assert items == []
    assert text == "[]"


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [("openai", "OpenAIProvider"), ("anthropic", "AnthropicProvider")],
)
def test_layout_file_malformed_page_two_aborts_document_atomically_through_response_parser(
    module_name: str,
    class_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    monkeypatch.setattr(module, "split_pdf_to_pages", lambda path: [(b"1", 10, 10), (b"2", 10, 10)])
    monkeypatch.setattr("parse_bench.inference.providers.parse._multipage_image.time.sleep", lambda delay: None)
    good = '<div data-bbox="[0,0,10,10]" data-label="Text">page one</div>'
    if module_name == "openai":
        responses = iter([_openai_response(good), *[_openai_response("") for _ in range(3)]])
    else:
        responses = iter(
            [
                _anthropic_response([SimpleNamespace(type="text", text=good)]),
                *[_anthropic_response([SimpleNamespace(type="text", text="")]) for _ in range(3)],
            ]
        )
    provider = _layout_file_provider(module_name, class_name)
    if module_name == "openai":
        provider._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: next(responses)))
        )
    else:
        provider._client = SimpleNamespace(
            beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: next(responses)))
        )
    successful_result = None

    with pytest.raises(ProviderRetryExhaustedError, match="page 2 failed after 3 attempts"):
        successful_result = provider.run_inference(_pipeline(module_name), _request(source))

    assert successful_result is None


def test_openai_and_anthropic_sdk_retries_are_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, dict[str, object]] = {}
    openai_sdk = importlib.import_module("openai")
    anthropic_sdk = importlib.import_module("anthropic")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        openai_sdk,
        "OpenAI",
        lambda **kwargs: captured.setdefault("openai", kwargs),
    )
    monkeypatch.setattr(
        anthropic_sdk,
        "Anthropic",
        lambda **kwargs: captured.setdefault("anthropic", kwargs),
    )

    openai_module = importlib.import_module("parse_bench.inference.providers.parse.openai")
    anthropic_module = importlib.import_module("parse_bench.inference.providers.parse.anthropic")
    openai_provider = openai_module.OpenAIProvider("openai")
    anthropic_provider = anthropic_module.AnthropicProvider("anthropic")

    assert captured["openai"]["max_retries"] == 0
    assert captured["anthropic"]["max_retries"] == 0
    assert openai_provider._dpi == openai_module.OpenAIProvider.PDF_RENDER_DPI == 150
    assert anthropic_provider._dpi == anthropic_module.AnthropicProvider.PDF_RENDER_DPI == 150


def test_google_sdk_and_dots_compatible_client_use_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, object]] = {}
    google_module = importlib.import_module("parse_bench.inference.providers.parse.google")
    dots_module = importlib.import_module("parse_bench.inference.providers.parse.dots_ocr")
    genai = importlib.import_module("google.genai")

    monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        genai,
        "Client",
        lambda **kwargs: captured.setdefault("google", kwargs),
    )
    monkeypatch.setattr(
        dots_module,
        "OpenAI",
        lambda **kwargs: captured.setdefault("dots", kwargs),
    )

    google_provider = google_module.GoogleProvider("google")
    dots_provider = dots_module.DotsOcrParseProvider(
        "dots_ocr_parse",
        {"endpoint_url": "https://dots.invalid/v1"},
    )

    http_options = cast(Any, captured["google"]["http_options"])
    assert http_options.retry_options.attempts == 1
    assert captured["dots"]["max_retries"] == 0
    assert google_provider._dpi == google_module.GoogleProvider.PDF_RENDER_DPI == 150
    assert dots_provider._dpi == dots_module.DotsOcrParseProvider.PDF_RENDER_DPI == 150
