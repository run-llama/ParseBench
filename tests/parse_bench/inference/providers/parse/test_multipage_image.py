from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from parse_bench.evaluation.stats import build_operational_stats
from parse_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRetryExhaustedError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._multipage_image import (
    append_attempt_usages,
    attempt_usages_complete,
    normalize_pdf_pages,
    open_document_page_images,
    run_pdf_pages,
)
from parse_bench.inference.runner import InferenceRunner, RunSummary
from parse_bench.schemas.parse_output import ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="test_image_provider",
        provider_name="test",
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def test_partial_attempt_usage_is_not_treated_as_exact_zero_buckets() -> None:
    usages: list[dict[str, int]] = []

    append_attempt_usages(
        usages,
        [{"stats": {"input_tokens": 4}, "page_number": 1, "attempt": 1, "status": "failed"}],
    )

    assert usages == [
        {
            "input_tokens": 4,
            "_usage_known": 0,
        }
    ]
    assert not attempt_usages_complete(usages)


def test_timeout_drain_blocks_resubmission_until_running_worker_terminates() -> None:
    runner = object.__new__(InferenceRunner)
    runner.provider = object()
    started = threading.Event()
    release = threading.Event()
    drain_returned = threading.Event()

    def worker() -> str:
        started.set()
        release.wait()
        return "late-success"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        assert started.wait(timeout=1)
        outcomes: list[object] = []

        def drain() -> None:
            outcomes.append(runner._cancel_inflight_and_drain("document", future))
            drain_returned.set()

        drain_thread = threading.Thread(target=drain)
        drain_thread.start()
        assert not drain_returned.wait(timeout=0.05)
        release.set()
        drain_thread.join(timeout=1)
        assert drain_returned.is_set()
        assert outcomes == ["late-success"]


def test_async_timeout_drain_awaits_running_worker_termination() -> None:
    runner = object.__new__(InferenceRunner)
    runner.provider = object()
    started = threading.Event()
    release = threading.Event()

    def worker() -> str:
        started.set()
        release.wait()
        return "late-success"

    async def exercise(future: concurrent.futures.Future[None]) -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=0.01)
        drain = asyncio.create_task(runner._cancel_inflight_and_drain_async("document", future))
        await asyncio.sleep(0.05)
        assert not drain.done()
        release.set()
        assert await asyncio.wait_for(drain, timeout=1) == "late-success"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        assert started.wait(timeout=1)
        asyncio.run(exercise(future))


def test_async_timeout_drain_propagates_caller_cancellation() -> None:
    runner = object.__new__(InferenceRunner)
    runner.provider = SimpleNamespace()
    started = threading.Event()
    release = threading.Event()

    def worker() -> str:
        started.set()
        release.wait()
        return "late-success"

    async def exercise(future: concurrent.futures.Future[str]) -> None:
        drain = asyncio.create_task(runner._cancel_inflight_and_drain_async("document", future))
        await asyncio.sleep(0.05)
        assert not drain.done()
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        release.set()
        assert await asyncio.wrap_future(future) == "late-success"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        assert started.wait(timeout=1)
        asyncio.run(exercise(future))


def test_async_timeout_drain_treats_queued_future_cancellation_as_retryable() -> None:
    runner = object.__new__(InferenceRunner)
    runner.provider = SimpleNamespace()
    release = threading.Event()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(release.wait)
        future = executor.submit(lambda: "must not run")
        try:
            outcome = asyncio.run(runner._cancel_inflight_and_drain_async("document", future))
        finally:
            release.set()
            blocker.result(timeout=1)

    assert outcome is None
    assert future.cancelled()


def test_sync_timeout_adopts_late_success_without_resubmission(tmp_path: Path) -> None:
    runner = object.__new__(InferenceRunner)
    runner.provider = SimpleNamespace()
    runner.pipeline = _pipeline()
    runner.output_dir = tmp_path
    runner.use_rich = False
    runner.console = None
    runner.job_statuses = {}
    runner.timeout_retries = 2
    runner.per_file_timeout = 0.01
    runner.max_concurrent = 1
    runner.save_raw = True
    runner.save_normalized = True
    runner.force = True
    runner.tags = []
    runner._is_already_processed = lambda example_id: False
    calls = 0
    late_raw = SimpleNamespace(latency_in_ms=7)

    def process(test_case: object, product_type: ProductType) -> tuple[object, None, None]:
        nonlocal calls
        calls += 1
        time.sleep(0.03)
        return late_raw, None, None

    runner._process_test_case = process
    test_case = SimpleNamespace(test_id="document", file_path=tmp_path / "document.pdf")

    summary = runner._run_test_cases_sync([test_case], ProductType.PARSE)

    assert calls == 1
    assert summary.successful == 1
    assert summary.failed == 0
    assert summary.total_latency_ms == 7


def test_async_timeout_adopts_late_success_without_resubmission(tmp_path: Path) -> None:
    runner = object.__new__(InferenceRunner)
    runner.provider = SimpleNamespace()
    runner.use_rich = False
    runner.job_statuses = {}
    runner.timeout_retries = 2
    runner.per_file_timeout = 0.01
    runner._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    runner._is_already_processed = lambda example_id: False
    calls = 0
    late_raw = SimpleNamespace(latency_in_ms=9)

    def process(pdf_path: Path, example_id: str, product_type: ProductType) -> tuple[object, None, None]:
        nonlocal calls
        calls += 1
        time.sleep(0.03)
        return late_raw, None, None

    runner._process_document = process

    async def exercise() -> RunSummary:
        run_summary = RunSummary()
        await runner._process_with_semaphore(
            asyncio.Semaphore(1),
            tmp_path / "document.pdf",
            "document",
            ProductType.PARSE,
            run_summary,
        )
        return run_summary

    try:
        run_summary = asyncio.run(exercise())
    finally:
        runner._thread_pool.shutdown(wait=True)

    assert calls == 1
    assert run_summary.successful == 1
    assert run_summary.failed == 0
    assert run_summary.total_latency_ms == 9


def test_external_cancellation_signals_and_drains_running_worker(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    cancel_calls: list[str] = []
    runner = object.__new__(InferenceRunner)
    runner.provider = SimpleNamespace(cancel=lambda example_id: (cancel_calls.append(example_id), release.set()))
    runner.use_rich = False
    runner.job_statuses = {}
    runner.timeout_retries = 0
    runner.per_file_timeout = 60
    runner._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    runner._is_already_processed = lambda example_id: False

    def process(pdf_path: Path, example_id: str, product_type: ProductType) -> tuple[None, None, None]:
        started.set()
        release.wait()
        return None, None, None

    runner._process_document = process

    async def exercise() -> None:
        task = asyncio.create_task(
            runner._process_with_semaphore(
                asyncio.Semaphore(1),
                tmp_path / "document.pdf",
                "document",
                ProductType.PARSE,
                RunSummary(),
            )
        )
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise())
    finally:
        runner._thread_pool.shutdown(wait=True)

    assert cancel_calls == ["document"]
    assert release.is_set()


def test_top_level_batch_cancellation_signals_and_drains_every_running_worker(tmp_path: Path) -> None:
    both_started = threading.Event()
    release = threading.Event()
    cancel_calls: list[str] = []
    started = 0
    started_lock = threading.Lock()
    runner = object.__new__(InferenceRunner)
    runner.provider = SimpleNamespace(cancel=lambda example_id: (cancel_calls.append(example_id), release.set()))
    runner.pipeline = _pipeline()
    runner.output_dir = tmp_path
    runner.use_rich = False
    runner.console = None
    runner.job_statuses = {}
    runner.timeout_retries = 0
    runner.per_file_timeout = 60
    runner.max_concurrent = 2
    runner._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    runner._is_already_processed = lambda example_id: False

    def process(pdf_path: Path, example_id: str, product_type: ProductType) -> tuple[None, None, None]:
        nonlocal started
        with started_lock:
            started += 1
            if started == 2:
                both_started.set()
        release.wait()
        return None, None, None

    runner._process_document = process

    async def exercise() -> None:
        task = asyncio.create_task(runner.run_files([tmp_path / "a.pdf", tmp_path / "b.pdf"], ProductType.PARSE))
        while not both_started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        runner._thread_pool.shutdown(wait=True)

    assert sorted(cancel_calls) == ["a", "b"]


def test_timeout_retries_after_cancelled_worker_returns_transient_failure(tmp_path: Path) -> None:
    release = threading.Event()
    calls = 0
    runner = object.__new__(InferenceRunner)
    runner.provider = SimpleNamespace(cancel=lambda example_id: release.set())
    runner.use_rich = False
    runner.job_statuses = {}
    runner.timeout_retries = 1
    runner.per_file_timeout = 0.01
    runner._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    runner._is_already_processed = lambda example_id: False

    def process(pdf_path: Path, example_id: str, product_type: ProductType) -> tuple[None, None, tuple[str, str, str]]:
        nonlocal calls
        calls += 1
        release.wait()
        return None, None, ("cancelled provider request", "", "ProviderTransientError")

    runner._process_document = process

    async def exercise() -> RunSummary:
        summary = RunSummary()
        await runner._process_with_semaphore(
            asyncio.Semaphore(1),
            tmp_path / "document.pdf",
            "document",
            ProductType.PARSE,
            summary,
        )
        return summary

    try:
        summary = asyncio.run(exercise())
    finally:
        runner._thread_pool.shutdown(wait=True)

    assert calls == 2
    assert summary.failed == 1


@pytest.mark.parametrize("provider_name", ["google", "openai", "anthropic"])
def test_runner_file_retry_accounts_failed_usage_cost_and_success(
    provider_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path / "document.pdf")
    calls = 0

    class FileProvider:
        def _get_pricing(self) -> tuple[float, float]:
            return 1.0, 2.0

        def run_inference(self, pipeline: PipelineSpec, current_request: InferenceRequest) -> RawInferenceResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ProviderTransientError(
                    "malformed billed file response",
                    attempt_stats={
                        "input_tokens": 5,
                        "output_tokens": 2,
                        "thinking_tokens": 0,
                        "total_tokens": 7,
                    },
                )
            now = datetime.now()
            return RawInferenceResult(
                request=current_request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=current_request.product_type,
                raw_output={
                    "pages": [{"page_index": 0, "markdown": "success"}],
                    "num_pages": 1,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "thinking_tokens": 0,
                    "total_tokens": 14,
                    "cost_usd": 18 / 1_000_000,
                    "cost_per_page_usd": 18 / 1_000_000,
                    "num_api_calls": 1,
                    "api_attempts": [],
                },
                started_at=now,
                completed_at=now,
                latency_in_ms=1,
            )

    runner = object.__new__(InferenceRunner)
    runner.provider = FileProvider()
    runner.pipeline = PipelineSpec(
        pipeline_name=f"{provider_name}_file",
        provider_name=provider_name,
        product_type=ProductType.PARSE,
    )
    monkeypatch.setattr("parse_bench.inference.runner.time.sleep", lambda delay: None)

    result = runner._run_inference_with_retries(request)

    assert calls == 2
    assert result.raw_output["num_api_calls"] == 2
    assert result.raw_output["input_tokens"] == 15
    assert result.raw_output["output_tokens"] == 6
    assert result.raw_output["total_tokens"] == 21
    assert result.raw_output["cost_usd"] == pytest.approx(27 / 1_000_000)
    assert result.raw_output["cost_per_page_usd"] == pytest.approx(27 / 1_000_000)
    attempts = result.raw_output["api_attempts"]
    assert [attempt["status"] for attempt in attempts] == ["failed", "succeeded"]
    assert attempts[0]["stats"]["cost_usd"] == pytest.approx(9 / 1_000_000)


def test_runner_terminal_retry_persists_every_failed_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path / "document.pdf")

    class FileProvider:
        def _get_pricing(self) -> tuple[float, float]:
            return 1.0, 2.0

        def run_inference(self, pipeline: PipelineSpec, current_request: InferenceRequest) -> RawInferenceResult:
            raise ProviderTransientError(
                "unavailable",
                attempt_stats={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

    runner = object.__new__(InferenceRunner)
    runner.provider = FileProvider()
    runner.pipeline = _pipeline()
    monkeypatch.setattr("parse_bench.inference.runner.time.sleep", lambda delay: None)

    with pytest.raises(ProviderTransientError) as exc_info:
        runner._run_inference_with_retries(request)

    payload = exc_info.value.debug_payload
    assert isinstance(payload, dict)
    assert len(payload["attempts"]) == 6
    assert [attempt["runner_attempt"] for attempt in payload["attempts"]] == [1, 2, 3, 4, 5, 6]


def test_pdf_pages_are_processed_and_combined_in_document_order(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    events: list[tuple[str, int]] = []
    rendered_pages: list[Image.Image] = []

    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 3})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        assert Path(path) == source
        assert dpi == 144
        assert first_page == last_page
        if rendered_pages:
            with pytest.raises(ValueError, match="Operation on closed image"):
                rendered_pages[-1].getpixel((0, 0))
        events.append(("render", first_page))
        image = Image.new("RGB", (20 + first_page, 30), (first_page, 0, 0))
        rendered_pages.append(image)
        return [image]

    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        render_page,
    )

    observed_pages: list[tuple[str, int]] = []

    def run_single_image(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        image_path = Path(request.source_file_path)
        with Image.open(image_path) as image:
            page_number = image.getpixel((0, 0))[0]
        events.append(("infer", page_number))
        observed_pages.append((image_path.name, page_number))
        now = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "markdown": f"page {page_number}",
                "cost_usd": page_number / 10,
                "cost_per_page_usd": page_number / 10,
                "input_tokens": page_number * 10,
                "input_tokens_per_page": page_number * 10,
                "output_tokens": page_number,
                "total_tokens": page_number * 11,
                "num_api_calls": 1,
            },
            started_at=now,
            completed_at=now,
            latency_in_ms=page_number,
        )

    raw_result = run_pdf_pages(
        _pipeline(),
        _request(source),
        dpi=144,
        run_single_image=run_single_image,
    )

    assert raw_result is not None
    assert observed_pages == [
        ("page-000001.png", 1),
        ("page-000002.png", 2),
        ("page-000003.png", 3),
    ]
    # Rendering and inference alternate, so later pages cannot be retained eagerly.
    assert events == [
        ("render", 1),
        ("infer", 1),
        ("render", 2),
        ("infer", 2),
        ("render", 3),
        ("infer", 3),
    ]
    with pytest.raises(ValueError, match="Operation on closed image"):
        rendered_pages[-1].getpixel((0, 0))
    # Raw artifacts must remain checkpoint-safe after temporary images disappear.
    json.dumps(raw_result.model_dump(mode="json"))
    assert raw_result.raw_output["num_pages"] == 3
    assert raw_result.raw_output["cost_usd"] == pytest.approx(0.6)
    assert raw_result.raw_output["cost_per_page_usd"] == pytest.approx(0.2)
    assert raw_result.raw_output["input_tokens"] == 60
    assert raw_result.raw_output["input_tokens_per_page"] == 20
    assert raw_result.raw_output["output_tokens"] == 6
    assert raw_result.raw_output["total_tokens"] == 66
    assert raw_result.raw_output["num_api_calls"] == 3

    envelope = raw_result.raw_output["_parse_bench_multipage"]
    assert isinstance(envelope, dict)
    assert [page["raw_output"]["input_tokens"] for page in envelope["pages"]] == [10, 20, 30]

    def normalize_single_image(raw: RawInferenceResult) -> InferenceResult:
        markdown = raw.raw_output["markdown"]
        output = ParseOutput(
            example_id=raw.request.example_id,
            pipeline_name=raw.pipeline_name,
            markdown=markdown,
            layout_pages=[ParseLayoutPageIR(page_number=1, md=markdown)],
        )
        return InferenceResult(
            request=raw.request,
            pipeline_name=raw.pipeline_name,
            product_type=raw.product_type,
            raw_output=raw.raw_output,
            output=output,
            started_at=raw.started_at,
            completed_at=raw.completed_at,
            latency_in_ms=raw.latency_in_ms,
        )

    result = normalize_pdf_pages(raw_result, normalize_single_image=normalize_single_image)

    assert result is not None
    assert isinstance(result.output, ParseOutput)
    assert result.output.markdown == "page 1\n\npage 2\n\npage 3"
    assert [(page.page_index, page.markdown) for page in result.output.pages] == [
        (0, "page 1"),
        (1, "page 2"),
        (2, "page 3"),
    ]
    assert [page.page_number for page in result.output.layout_pages] == [1, 2, 3]

    stats = {stat.name: stat for stat in build_operational_stats(result)}
    assert stats["latency_ms_per_page"].value == pytest.approx(result.latency_in_ms / 3)
    assert stats["cost_usd"].value == pytest.approx(0.6)
    assert stats["cost_per_page_usd"].value == pytest.approx(0.2)
    assert stats["input_tokens"].value == 60
    assert stats["input_tokens_per_page"].value == 20
    assert stats["num_api_calls"].value == 3


def test_transient_page_retry_does_not_replay_prior_page_or_duplicate_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 3})
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda path, dpi, first_page, last_page: [Image.new("RGB", (8, 8), (first_page, 0, 0))],
    )
    monkeypatch.setattr("parse_bench.inference.providers.parse._multipage_image.time.sleep", lambda delay: None)
    requested_pages: list[int] = []

    def run_single_image(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        with Image.open(request.source_file_path) as image:
            page_number = image.getpixel((0, 0))[0]
        requested_pages.append(page_number)
        if requested_pages == [1, 2]:
            raise ProviderTransientError("page two timed out")
        now = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"markdown": f"page {page_number}", "num_api_calls": 1},
            started_at=now,
            completed_at=now,
            latency_in_ms=1,
        )

    raw_result = run_pdf_pages(_pipeline(), _request(source), dpi=144, run_single_image=run_single_image)

    assert raw_result is not None
    assert requested_pages == [1, 2, 2, 3]
    assert raw_result.raw_output["num_api_calls"] == 4
    envelope = raw_result.raw_output["_parse_bench_multipage"]
    assert isinstance(envelope, dict)
    assert [page["page_index"] for page in envelope["pages"]] == [0, 1, 2]


def test_multipage_retry_ledger_aggregates_failed_attempt_usage_and_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 1})
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda path, dpi, first_page, last_page: [Image.new("RGB", (8, 8), "white")],
    )
    monkeypatch.setattr("parse_bench.inference.providers.parse._multipage_image.time.sleep", lambda delay: None)
    calls = 0

    def run_single_image(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderTransientError(
                "malformed billed response",
                attempt_stats={
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                    "cost_usd": 0.05,
                },
            )
        now = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "markdown": "success",
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "cost_usd": 0.1,
                "num_api_calls": 1,
            },
            started_at=now,
            completed_at=now,
            latency_in_ms=1,
        )

    result = run_pdf_pages(_pipeline(), _request(source), dpi=144, run_single_image=run_single_image)

    assert result is not None
    assert result.raw_output["num_api_calls"] == 2
    assert result.raw_output["input_tokens"] == 15
    assert result.raw_output["output_tokens"] == 6
    assert result.raw_output["total_tokens"] == 21
    assert result.raw_output["cost_usd"] == pytest.approx(0.15)
    assert result.raw_output["cost_per_page_usd"] == pytest.approx(0.15)


def test_exhausted_page_retry_is_terminal_to_document_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda path, dpi, first_page, last_page: [Image.new("RGB", (8, 8), (first_page, 0, 0))],
    )
    monkeypatch.setattr("parse_bench.inference.providers.parse._multipage_image.time.sleep", lambda delay: None)
    requested_pages: list[int] = []
    document_attempts = 0

    def run_single_image(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        with Image.open(request.source_file_path) as image:
            page_number = image.getpixel((0, 0))[0]
        requested_pages.append(page_number)
        if page_number == 2:
            raise ProviderTransientError("still unavailable")
        now = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"markdown": "page 1", "num_api_calls": 1},
            started_at=now,
            completed_at=now,
            latency_in_ms=1,
        )

    class AdapterProvider:
        def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
            nonlocal document_attempts
            document_attempts += 1
            result = run_pdf_pages(pipeline, request, dpi=144, run_single_image=run_single_image)
            assert result is not None
            return result

    runner = object.__new__(InferenceRunner)
    runner.use_rich = False
    runner.job_statuses = {}
    runner.pipeline = _pipeline()
    runner.provider = AdapterProvider()
    runner.output_dir = tmp_path
    runner._prepare_source_file_for_provider = lambda example_id, path: path
    runner._fetch_parse_job_logs = lambda raw_result, example_id: None
    runner._save_result = lambda raw_result, normalized_result: None

    raw_result, normalized_result, error = runner._process_document(source, "document", ProductType.PARSE)

    assert raw_result is None
    assert normalized_result is None
    assert error is not None and error[2] == ProviderRetryExhaustedError.__name__
    assert document_attempts == 1
    assert requested_pages == [1, 2, 2, 2]
    payload = json.loads((tmp_path / "document.error.raw.json").read_text())
    assert [(attempt["page_number"], attempt["status"]) for attempt in payload["attempts"]] == [
        (1, "succeeded"),
        (2, "failed"),
        (2, "failed"),
        (2, "failed"),
    ]


def test_multipage_aggregation_omits_partial_or_invalid_provider_metadata(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda path, dpi, first_page, last_page: [Image.new("RGB", (8, 8), "white")],
    )
    page_number = 0

    def run_single_image(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        nonlocal page_number
        page_number += 1
        now = datetime.now()
        raw_output: dict[str, object] = {
            "markdown": f"page {page_number}",
            "input_tokens": 10 if page_number == 1 else "unknown",
            "output_tokens": page_number,
            "cost_usd": 0.1 if page_number == 1 else float("nan"),
        }
        if page_number == 1:
            raw_output["num_api_calls"] = 1
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=now,
            completed_at=now,
            latency_in_ms=1,
        )

    result = run_pdf_pages(_pipeline(), _request(source), dpi=144, run_single_image=run_single_image)

    assert result is not None
    assert result.raw_output["num_pages"] == 2
    assert result.raw_output["output_tokens"] == 3
    assert "input_tokens" not in result.raw_output
    assert "cost_usd" not in result.raw_output
    assert result.raw_output["num_api_calls"] == 2


def test_pdf_render_failure_identifies_page_and_stops(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    render_calls: list[int] = []

    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 3})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        render_calls.append(first_page)
        if first_page == 2:
            raise RuntimeError("poppler failed")
        return [Image.new("RGB", (8, 8), "white")]

    monkeypatch.setattr("pdf2image.convert_from_path", render_page)

    def run_single_image(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        now = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"markdown": "page 1"},
            started_at=now,
            completed_at=now,
            latency_in_ms=1,
        )

    with pytest.raises(ProviderPermanentError, match="Failed to render PDF page 2: poppler failed"):
        run_pdf_pages(
            _pipeline(),
            _request(source),
            dpi=144,
            run_single_image=run_single_image,
        )

    assert render_calls == [1, 2]


def test_inference_failure_closes_current_page_and_stops_rendering(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered_pages: list[Image.Image] = []
    render_calls: list[int] = []

    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        render_calls.append(first_page)
        image = Image.new("RGB", (8, 8), "white")
        rendered_pages.append(image)
        return [image]

    monkeypatch.setattr("pdf2image.convert_from_path", render_page)

    def fail_inference(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        run_pdf_pages(_pipeline(), _request(source), dpi=144, run_single_image=fail_inference)

    assert render_calls == [1]
    with pytest.raises(ValueError, match="Operation on closed image"):
        rendered_pages[0].getpixel((0, 0))


def test_single_image_context_closes_the_image_after_inference(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(source)
    real_open = Image.open
    close_calls = 0

    class TrackedImageContext:
        def __init__(self, path: str | Path) -> None:
            self.image = real_open(path)

        def __enter__(self) -> Image.Image:
            return self.image

        def __exit__(self, *args: object) -> None:
            nonlocal close_calls
            close_calls += 1
            self.image.close()

    monkeypatch.setattr(Image, "open", TrackedImageContext)

    with open_document_page_images(source, dpi=144) as images:
        (image,) = list(images)
        assert len(images) == 1
        assert image.getpixel((0, 0)) == (255, 255, 255)

    assert close_calls == 1


def test_single_image_input_stays_on_the_provider_path(tmp_path: Path) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(source)
    called = False

    def run_single_image(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        nonlocal called
        called = True
        raise AssertionError("the adapter must not intercept a single image")

    result = run_pdf_pages(
        _pipeline(),
        _request(source),
        dpi=144,
        run_single_image=run_single_image,
    )

    assert result is None
    assert called is False


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"version": 2, "num_pages": 1, "pages": []},
        {"version": 1, "num_pages": True, "pages": []},
        {"version": 1, "num_pages": 0, "pages": []},
        {"version": 1, "num_pages": 1, "pages": "not-a-list"},
        {"version": 1, "num_pages": 2, "pages": [{"page_index": 0, "raw_output": {}}]},
        {"version": 1, "num_pages": 1, "pages": [{"page_index": True, "raw_output": {}}]},
        {"version": 1, "num_pages": 1, "pages": [{"page_index": 0, "raw_output": []}]},
    ],
)
def test_malformed_multipage_envelopes_are_rejected(envelope: dict[str, object]) -> None:
    now = datetime.now()
    raw_result = RawInferenceResult(
        request=_request(Path("document.pdf")),
        pipeline=_pipeline(),
        pipeline_name="test_image_provider",
        product_type=ProductType.PARSE,
        raw_output={"_parse_bench_multipage": envelope},
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )

    with pytest.raises(ProviderPermanentError, match="Invalid multipage raw output"):
        normalize_pdf_pages(raw_result, normalize_single_image=lambda raw: pytest.fail("must not normalize a page"))
