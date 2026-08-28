from __future__ import annotations

import asyncio
import io
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from PIL import Image

from parse_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRetryExhaustedError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse import kdl_frontier_nano as kdl
from parse_bench.inference.providers.parse._multipage_image import IMAGE_BACKED_PDF_PROVIDERS
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

KDL_SPEC = next(spec for spec in IMAGE_BACKED_PDF_PROVIDERS if spec.execution == "kdl")


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="kdl_frontier_nano_test",
        provider_name="kdl_frontier_nano",
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def _provider() -> kdl.KdlFrontierNanoProvider:
    provider = object.__new__(kdl.KdlFrontierNanoProvider)
    provider._dpi = KDL_SPEC.dpi
    provider._endpoint_url = "http://provider.invalid/v1"
    provider._model = "test-model"
    provider._max_concurrent = 1
    provider._timeout = 30
    provider._max_pages = 10
    provider._input_cost_per_million = None
    provider._output_cost_per_million = None
    return provider


def _raw_kdl_result(raw_output: dict[str, object]) -> RawInferenceResult:
    now = datetime.now()
    request = InferenceRequest(
        example_id="document",
        source_file_path="document.pdf",
        product_type=ProductType.PARSE,
    )
    return RawInferenceResult(
        request=request,
        pipeline=_pipeline(),
        pipeline_name="kdl_frontier_nano_test",
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_kdl_normalize_rejects_non_contiguous_page_identities() -> None:
    raw = _raw_kdl_result(
        {
            "markdown": "untrusted",
            "pages": [
                {"page_number": 2, "elements": []},
                {"page_number": 3, "elements": []},
            ],
            "markdown_pages": [
                {"page_number": 2, "content": "two"},
                {"page_number": 3, "content": "three"},
            ],
        }
    )

    with pytest.raises(ProviderPermanentError, match="page identities are inconsistent"):
        _provider().normalize(raw)


def test_kdl_normalize_rebuilds_document_markdown_from_canonical_pages() -> None:
    raw = _raw_kdl_result(
        {
            "markdown": "three before two",
            "pages": [
                {"page_number": 1, "elements": []},
                {"page_number": 2, "elements": []},
            ],
            "markdown_pages": [
                {"page_number": 1, "content": "one"},
                {"page_number": 2, "content": "two"},
            ],
        }
    )

    output = _provider().normalize(raw).output

    assert [page.page_index for page in output.pages] == [0, 1]
    assert [page.page_number for page in output.layout_pages] == [1, 2]
    assert output.markdown == "one\n\n---\n\n**Page 2**\n\ntwo"


def _png_bytes(page_number: int) -> bytes:
    with Image.new("RGBA", (8 + page_number, 8), (page_number, 0, 0, 255)) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


class _FakePixmap:
    def __init__(self, page_number: int) -> None:
        self._page_number = page_number

    def tobytes(self, output: str) -> bytes:
        assert output == "png"
        return _png_bytes(self._page_number)


class _FakePage:
    def __init__(self, page_number: int, render: Any) -> None:
        self._page_number = page_number
        self._render = render

    def get_pixmap(self, **kwargs: object) -> _FakePixmap:
        self._render(self._page_number)
        return _FakePixmap(self._page_number)


class _FakeDocument:
    def __init__(self, page_count: int, render: Any, events: list[tuple[str, int]]) -> None:
        self.page_count = page_count
        self._render = render
        self._events = events

    def __enter__(self) -> _FakeDocument:
        return self

    def __exit__(self, *args: object) -> None:
        self._events.append(("close_document", 0))

    def __iter__(self):
        return iter(_FakePage(page_number, self._render) for page_number in range(1, self.page_count + 1))


def _track_opened_and_normalized_images(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Image.Image], list[Image.Image]]:
    opened: list[Image.Image] = []
    normalized: list[Image.Image] = []
    real_open = Image.open
    real_normalize = kdl.normalize_image_mode

    def tracked_open(*args: object, **kwargs: object) -> Image.Image:
        image = real_open(*args, **kwargs)
        opened.append(image)
        return image

    def tracked_normalize(image: Image.Image, target_mode: str = "RGB") -> Image.Image:
        result = real_normalize(image, target_mode)
        if result is not image:
            normalized.append(result)
        return result

    monkeypatch.setattr(kdl.Image, "open", tracked_open)
    monkeypatch.setattr(kdl, "normalize_image_mode", tracked_normalize)
    return opened, normalized


def _assert_closed(image: Image.Image) -> None:
    with pytest.raises(ValueError, match="Operation on closed image"):
        image.getpixel((0, 0))


def _content_image() -> Image.Image:
    image = Image.new("RGB", (64, 64), "white")
    for coordinate in range(8, 56):
        image.putpixel((coordinate, coordinate), (0, 0, 0))
    return image


def _track_pillow_derivatives(monkeypatch: pytest.MonkeyPatch) -> list[Image.Image]:
    derived: list[Image.Image] = []

    def wrap(method_name: str) -> None:
        real_method = getattr(Image.Image, method_name)

        def tracked(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
            result = real_method(image, *args, **kwargs)
            result.close = Mock(wraps=result.close)
            derived.append(result)
            return result

        monkeypatch.setattr(Image.Image, method_name, tracked)

    for method_name in ("convert", "resize", "crop"):
        wrap(method_name)
    return derived


@pytest.mark.parametrize(
    ("status_code", "error_type", "attempts"),
    [(400, ProviderPermanentError, 1), (429, ProviderTransientError, 1), (503, ProviderTransientError, 1)],
)
def test_nano_chat_classifies_http_failures(
    status_code: int,
    error_type: type[Exception],
    attempts: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class Client:
        async def post(self, url: str, **kwargs: object) -> httpx.Response:
            nonlocal calls
            calls += 1
            request = httpx.Request("POST", url)
            return httpx.Response(status_code, request=request, text="provider failure")

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(kdl.asyncio, "sleep", no_sleep)

    with pytest.raises(error_type, match=f"HTTP {status_code}|{status_code}") as caught:
        asyncio.run(kdl._nano_chat(Client(), "http://provider.invalid", {}, asyncio.Semaphore(1)))

    assert calls == attempts
    if status_code >= 500 or status_code == 429:
        assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)


def test_nano_chat_retries_invalid_responses_and_preserves_last_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class Client:
        async def post(self, url: str, **kwargs: object) -> httpx.Response:
            nonlocal calls
            calls += 1
            request = httpx.Request("POST", url)
            return httpx.Response(200, request=request, json={"choices": []})

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(kdl.asyncio, "sleep", no_sleep)

    with pytest.raises(ProviderTransientError, match="Stage request failed") as caught:
        asyncio.run(kdl._nano_chat(Client(), "http://provider.invalid", {}, asyncio.Semaphore(1)))

    assert calls == 1
    assert isinstance(caught.value.__cause__, IndexError)


def test_kdl_persists_every_stage_attempt_and_known_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _content_image()
    monkeypatch.setattr(kdl, "analyze_page_content", lambda image: SimpleNamespace(is_blank=False))
    layout = "<|box_start|>100 100 900 900<|box_end|><|ref_start|>text<|ref_end|>"
    responses = [
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2}},
        {
            "choices": [{"message": {"content": layout}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        },
        {
            "choices": [{"message": {"content": "recognized text"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        },
    ]

    class Client:
        async def post(self, url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            return httpx.Response(200, request=request, json=responses.pop(0))

    class ClientContext:
        async def __aenter__(self) -> Client:
            return Client()

        async def __aexit__(self, *args: object) -> None:
            return None

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(kdl.httpx, "AsyncClient", lambda **kwargs: ClientContext())
    monkeypatch.setattr(kdl.asyncio, "sleep", no_sleep)
    engine = kdl._NanoEngine(
        "http://provider.invalid",
        "test-model",
        1,
        30,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )

    result = asyncio.run(engine.parse_pages([page]))

    assert result["num_api_calls"] == 3
    assert [(attempt["stage"], attempt["attempt"], attempt["status"]) for attempt in result["api_attempts"]] == [
        ("layout", 1, "failed"),
        ("layout", 2, "succeeded"),
        ("text recognition", 1, "succeeded"),
    ]
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 10
    assert result["total_tokens"] == 20
    assert [attempt["stats"]["cost_usd"] for attempt in result["api_attempts"]] == pytest.approx(
        [2 / 1_000_000, 11 / 1_000_000, 17 / 1_000_000]
    )
    assert result["cost_usd"] == pytest.approx(30 / 1_000_000)
    assert result["cost_per_page_usd"] == pytest.approx(30 / 1_000_000)
    page.close()


def test_kdl_retries_semantically_invalid_layout_inside_stage_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _content_image()
    monkeypatch.setattr(kdl, "analyze_page_content", lambda image: SimpleNamespace(is_blank=False))

    async def invalid_layout(*args: object, **kwargs: object) -> kdl._NanoStageResponse:
        return kdl._NanoStageResponse(
            content="not native layout",
            usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        )

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(kdl, "_nano_chat", invalid_layout)
    monkeypatch.setattr(kdl.asyncio, "sleep", no_sleep)
    engine = kdl._NanoEngine("http://provider.invalid", "test-model", 1, 30)

    with pytest.raises(ProviderRetryExhaustedError) as caught:
        asyncio.run(engine._parse_page(SimpleNamespace(), asyncio.Semaphore(1), page, 1))

    attempts = caught.value.debug_payload["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["failed", "failed", "failed"]
    assert [attempt["stats"]["total_tokens"] for attempt in attempts] == [7, 7, 7]
    page.close()


@pytest.mark.parametrize("fail_recognition", [False, True], ids=["monochromatic-skip", "gather-failure"])
def test_kdl_real_page_stage_closes_every_derivative(
    fail_recognition: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _content_image()
    derived = _track_pillow_derivatives(monkeypatch)
    monkeypatch.setattr(kdl, "analyze_page_content", lambda image: SimpleNamespace(is_blank=False))
    layout = "<|box_start|>100 100 900 900<|box_end|><|ref_start|>text<|ref_end|>"
    calls = 0

    async def chat(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return layout
        if fail_recognition:
            raise ProviderTransientError("recognition exhausted")
        return "recognized text"

    if not fail_recognition:
        page.paste("white", (6, 6, 58, 58))

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(kdl, "_nano_chat", chat)
    monkeypatch.setattr(kdl.asyncio, "sleep", no_sleep)
    engine = kdl._NanoEngine("http://provider.invalid", "test-model", 1, 30)

    if fail_recognition:
        with pytest.raises(ProviderRetryExhaustedError, match="KDL page 1 failed after 3 attempts"):
            asyncio.run(engine._parse_page(object(), asyncio.Semaphore(1), page, 1))
    else:
        assert asyncio.run(engine._parse_page(object(), asyncio.Semaphore(1), page, 1)) == []

    assert derived
    assert all(isinstance(image.close, Mock) and image.close.call_count == 1 for image in derived)
    assert page.getpixel((0, 0)) == (255, 255, 255)
    page.close()


def test_kdl_real_page_two_stage_failure_retries_only_page_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [_content_image(), _content_image()]
    derived = _track_pillow_derivatives(monkeypatch)
    monkeypatch.setattr(kdl, "analyze_page_content", lambda image: SimpleNamespace(is_blank=False))
    layout = "<|box_start|>100 100 900 900<|box_end|><|ref_start|>text<|ref_end|>"
    calls = 0

    async def chat(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ProviderTransientError("page two layout timed out")
        return layout if calls in {1, 4} else f"page text {calls}"

    monkeypatch.setattr(kdl, "_nano_chat", chat)

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(kdl.asyncio, "sleep", no_sleep)
    engine = kdl._NanoEngine("http://provider.invalid", "test-model", 1, 30)

    result = asyncio.run(engine.parse_pages(pages))

    assert calls == 5
    assert [page["page_number"] for page in result["pages"]] == [1, 2]
    assert derived
    assert all(isinstance(image.close, Mock) and image.close.call_count == 1 for image in derived)
    for page in pages:
        assert page.getpixel((0, 0)) == (255, 255, 255)
        page.close()


def test_kdl_exhausted_page_two_retry_is_terminal_without_replaying_page_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [_content_image(), _content_image()]
    monkeypatch.setattr(kdl, "analyze_page_content", lambda image: SimpleNamespace(is_blank=False))
    layout = "<|box_start|>100 100 900 900<|box_end|><|ref_start|>text<|ref_end|>"
    calls = 0

    async def chat(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise ProviderTransientError("page two unavailable")
        return layout if calls == 1 else "page one text"

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(kdl, "_nano_chat", chat)
    monkeypatch.setattr(kdl.asyncio, "sleep", no_sleep)
    engine = kdl._NanoEngine("http://provider.invalid", "test-model", 1, 30)

    with pytest.raises(ProviderRetryExhaustedError, match="KDL page 2 failed after 3 attempts") as caught:
        asyncio.run(engine.parse_pages(pages))

    assert calls == 5
    attempts = caught.value.debug_payload["attempts"]
    assert [(attempt["page_number"], attempt["status"]) for attempt in attempts] == [
        (1, "succeeded"),
        (1, "succeeded"),
        (2, "failed"),
        (2, "failed"),
        (2, "failed"),
    ]
    for page in pages:
        page.close()


def test_kdl_recognition_retry_preserves_layout_and_successful_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = (
        "<|box_start|>50 50 450 450<|box_end|><|ref_start|>text<|ref_end|>"
        "<|box_start|>550 550 950 950<|box_end|><|ref_start|>text<|ref_end|>"
    )
    layout_calls = 0
    recognition_calls = 0
    failed_task: asyncio.Task[object] | None = None
    task_attempts: dict[asyncio.Task[object], int] = {}

    async def chat(client: object, url: str, payload: dict[str, object], semaphore: object) -> str:
        nonlocal failed_task, layout_calls, recognition_calls
        prompt = payload["messages"][0]["content"][1]["text"]  # type: ignore[index]
        if prompt == kdl._NANO_PROMPTS["layout"]:
            layout_calls += 1
            return layout

        recognition_calls += 1
        task = asyncio.current_task()
        assert task is not None
        task_attempts[task] = task_attempts.get(task, 0) + 1
        if failed_task is None:
            failed_task = task
        if task is failed_task and task_attempts[task] == 1:
            raise ProviderTransientError("recognition failed once")
        return f"recognized {recognition_calls}"

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(kdl, "_nano_chat", chat)
    monkeypatch.setattr(kdl, "analyze_page_content", lambda image: SimpleNamespace(is_blank=False))
    monkeypatch.setattr(kdl.asyncio, "sleep", no_sleep)
    engine = kdl._NanoEngine("http://provider.invalid", "test-model", 2, 30)

    with _content_image() as page:
        result = asyncio.run(engine._parse_page(object(), asyncio.Semaphore(2), page, 1))

    assert len(result) == 2
    assert layout_calls == 1
    assert recognition_calls == 3


def test_kdl_terminal_recognition_exhaustion_does_not_replay_successful_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = (
        "<|box_start|>50 50 450 450<|box_end|><|ref_start|>text<|ref_end|>"
        "<|box_start|>550 550 950 950<|box_end|><|ref_start|>text<|ref_end|>"
    )
    layout_calls = 0
    recognition_calls = 0
    active_recognitions = 0
    failed_task: asyncio.Task[object] | None = None

    async def chat(client: object, url: str, payload: dict[str, object], semaphore: object) -> str:
        nonlocal active_recognitions, failed_task, layout_calls, recognition_calls
        prompt = payload["messages"][0]["content"][1]["text"]  # type: ignore[index]
        if prompt == kdl._NANO_PROMPTS["layout"]:
            layout_calls += 1
            return layout

        recognition_calls += 1
        task = asyncio.current_task()
        assert task is not None
        if failed_task is None:
            failed_task = task
        active_recognitions += 1
        try:
            if task is failed_task:
                raise ProviderTransientError("recognition failed")
            return "successful sibling"
        finally:
            active_recognitions -= 1

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(kdl, "_nano_chat", chat)
    monkeypatch.setattr(kdl, "analyze_page_content", lambda image: SimpleNamespace(is_blank=False))
    monkeypatch.setattr(kdl.asyncio, "sleep", no_sleep)
    engine = kdl._NanoEngine("http://provider.invalid", "test-model", 2, 30)

    with _content_image() as page:
        with pytest.raises(ProviderRetryExhaustedError, match="KDL page 1 failed after 3 attempts"):
            asyncio.run(engine._parse_page(object(), asyncio.Semaphore(2), page, 1))

    assert layout_calls == 1
    assert recognition_calls == 4
    assert active_recognitions == 0


@pytest.mark.parametrize("fail", [False, True], ids=["success", "encoding-failure"])
def test_nano_data_uri_closes_rgb_conversion(
    fail: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Image.new("HSV", (8, 8), (0, 0, 255))
    real_convert = Image.Image.convert
    real_save = Image.Image.save
    converted: list[Image.Image] = []

    def convert(image: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        result = real_convert(image, *args, **kwargs)
        result.close = Mock(wraps=result.close)
        converted.append(result)
        return result

    def save(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        if fail:
            raise RuntimeError("encoding failed")
        real_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", convert)
    monkeypatch.setattr(Image.Image, "save", save)

    if fail:
        with pytest.raises(RuntimeError, match="encoding failed"):
            kdl._nano_image_to_data_uri(original)
    else:
        assert kdl._nano_image_to_data_uri(original).startswith("data:image/jpeg;base64,")

    assert len(converted) == 1
    assert isinstance(converted[0].close, Mock) and converted[0].close.call_count == 1
    assert original.getpixel((0, 0)) == (0, 0, 255)
    original.close()


def test_kdl_streams_pdf_pages_in_order_and_closes_owned_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    events: list[tuple[str, int]] = []
    opened, normalized = _track_opened_and_normalized_images(monkeypatch)

    def render(page_number: int) -> None:
        if opened:
            _assert_closed(opened[-1])
            _assert_closed(normalized[-1])
        events.append(("render", page_number))

    monkeypatch.setattr(
        "fitz.open",
        lambda path: _FakeDocument(3, render, events),
    )

    async def parse_page(
        self: object,
        client: object,
        semaphore: object,
        image: Image.Image,
        page_number: int,
    ) -> list[dict[str, object]]:
        assert image.mode == "RGB"
        events.append(("infer", page_number))
        if page_number == 2:
            return []
        return [
            {
                "category": "Text",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "content": f"page {page_number}",
                "layout_order": 0,
                "page_number": page_number,
            }
        ]

    monkeypatch.setattr(kdl._NanoEngine, "_parse_page", parse_page)

    raw_result = _provider().run_inference(_pipeline(), _request(source))

    assert raw_result.raw_output["markdown"] == ("page 1\n\n---\n\n**Page 2**\n\n---\n\n**Page 3**\n\npage 3")
    assert [page["page_number"] for page in raw_result.raw_output["pages"]] == [1, 2, 3]
    assert [page["page_number"] for page in raw_result.raw_output["markdown_pages"]] == [1, 2, 3]
    assert raw_result.raw_output["pages"][1]["elements"] == []
    assert raw_result.raw_output["markdown_pages"][1]["content"] == ""
    normalized_result = _provider().normalize(raw_result)
    assert [page.page_index for page in normalized_result.output.pages] == [0, 1, 2]
    assert [page.page_number for page in normalized_result.output.layout_pages] == [1, 2, 3]
    assert events == [
        ("render", 1),
        ("infer", 1),
        ("render", 2),
        ("infer", 2),
        ("render", 3),
        ("infer", 3),
        ("close_document", 0),
    ]
    assert len(opened) == len(normalized) == 3
    for image in [*opened, *normalized]:
        _assert_closed(image)


def test_kdl_non_native_layout_aborts_page(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = kdl._NanoEngine("http://provider.invalid", "test-model", 1, 30)

    async def non_native_layout(*args: object, **kwargs: object) -> str:
        return "diagnostic response instead of layout tokens"

    monkeypatch.setattr(kdl, "_nano_chat", non_native_layout)
    monkeypatch.setattr(kdl, "analyze_page_content", lambda image: SimpleNamespace(is_blank=False))
    with Image.new("RGB", (64, 64), "white") as image:
        with pytest.raises(ProviderPermanentError, match="non-native layout response"):
            asyncio.run(engine._parse_page(object(), asyncio.Semaphore(1), image, 2))


@pytest.mark.parametrize("failure_stage", ["crop", "preprocess"])
def test_kdl_crop_or_preprocess_failure_is_classified(failure_stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with Image.new("RGB", (64, 64), "white") as image:
        if failure_stage == "crop":
            monkeypatch.setattr(
                Image.Image,
                "crop",
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("crop failed")),
            )
        else:
            monkeypatch.setattr(
                kdl,
                "preprocess_for_vlm",
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("preprocess failed")),
            )

        with pytest.raises(
            ProviderPermanentError,
            match=f"Failed to crop or preprocess layout item 0: {failure_stage} failed",
        ):
            kdl._nano_group_by_bucket(
                [
                    {
                        "category": "Text",
                        "bbox": [0.0, 0.0, 0.5, 0.5],
                        "layout_order": 0,
                        "page_number": 1,
                    }
                ],
                image,
                lambda derived: derived,
            )


@pytest.mark.parametrize(
    "failure",
    [
        ProviderPermanentError("invalid page"),
        ProviderTransientError("page timed out"),
    ],
    ids=["permanent", "transient"],
)
def test_kdl_page_two_failure_aborts_and_closes_images(
    failure: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    events: list[tuple[str, int]] = []
    opened, normalized = _track_opened_and_normalized_images(monkeypatch)
    monkeypatch.setattr(
        "fitz.open",
        lambda path: _FakeDocument(3, lambda page: events.append(("render", page)), events),
    )

    async def parse_page(
        self: object,
        client: object,
        semaphore: object,
        image: Image.Image,
        page_number: int,
    ) -> list[dict[str, object]]:
        events.append(("infer", page_number))
        if page_number == 2:
            raise failure
        return []

    monkeypatch.setattr(kdl._NanoEngine, "_parse_page", parse_page)
    successful_result = None

    with pytest.raises(type(failure), match=str(failure)):
        successful_result = _provider().run_inference(_pipeline(), _request(source))

    assert successful_result is None
    assert events == [
        ("render", 1),
        ("infer", 1),
        ("render", 2),
        ("infer", 2),
        ("close_document", 0),
    ]
    assert len(opened) == len(normalized) == 2
    for image in [*opened, *normalized]:
        _assert_closed(image)


def test_kdl_render_failure_closes_prior_page_and_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    events: list[tuple[str, int]] = []
    opened, normalized = _track_opened_and_normalized_images(monkeypatch)

    def render(page_number: int) -> None:
        events.append(("render", page_number))
        if page_number == 2:
            raise RuntimeError("renderer failed")

    monkeypatch.setattr("fitz.open", lambda path: _FakeDocument(3, render, events))

    async def parse_page(*args: object, **kwargs: object) -> list[dict[str, object]]:
        events.append(("infer", len(opened)))
        return []

    monkeypatch.setattr(kdl._NanoEngine, "_parse_page", parse_page)

    with pytest.raises(ProviderPermanentError, match="Failed to render document page 2: renderer failed"):
        _provider().run_inference(_pipeline(), _request(source))

    assert events == [
        ("render", 1),
        ("infer", 1),
        ("render", 2),
        ("close_document", 0),
    ]
    assert len(opened) == len(normalized) == 1
    _assert_closed(opened[0])
    _assert_closed(normalized[0])


def test_kdl_single_image_path_preserves_one_page_behavior_and_closes_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "page.png"
    with Image.new("RGBA", (8, 8), "white") as image:
        image.save(source)
    opened, normalized = _track_opened_and_normalized_images(monkeypatch)
    page_numbers: list[int] = []
    monkeypatch.setattr("fitz.open", lambda path: pytest.fail("single images must not open as PDFs"))

    async def parse_page(
        self: object,
        client: object,
        semaphore: object,
        image: Image.Image,
        page_number: int,
    ) -> list[dict[str, object]]:
        page_numbers.append(page_number)
        return []

    monkeypatch.setattr(kdl._NanoEngine, "_parse_page", parse_page)

    raw_result = _provider().run_inference(_pipeline(), _request(source))

    assert raw_result.raw_output["markdown"] == ""
    assert page_numbers == [1]
    assert len(opened) == len(normalized) == 1
    _assert_closed(opened[0])
    _assert_closed(normalized[0])
