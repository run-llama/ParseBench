"""Tests for the inference runner: filename randomization, timeout precedence, metadata."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from parse_bench.inference.providers.base import ProviderTransientError
from parse_bench.inference.runner import DEFAULT_PER_FILE_TIMEOUT_S, InferenceRunner
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import ParseTestCase


class RecordingProvider:
    """Minimal provider that records the source path visible during inference."""

    def __init__(self, fail_times: int = 0) -> None:
        self.seen_path: Path | None = None
        self.seen_bytes: bytes | None = None
        self.calls = 0
        self._fail_times = fail_times

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise ProviderTransientError("flaky")
        seen_path = Path(request.source_file_path)
        self.seen_path = seen_path
        self.seen_bytes = seen_path.read_bytes()
        now = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"seen_filename": seen_path.name},
            started_at=now,
            completed_at=now,
            latency_in_ms=0,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            markdown="",
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    source_file = tmp_path / "benchmark-set" / "annual_report_2026.pdf"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"%PDF-1.4\nbenchmark fixture\n")
    return source_file


@pytest.fixture
def pipeline_spec() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="openai_parse_file",
        provider_name="openai",
        product_type=ProductType.PARSE,
        config={},
    )


def _runner(provider: RecordingProvider, pipeline: PipelineSpec, output_dir: Path, **kwargs: object) -> InferenceRunner:
    return InferenceRunner(
        provider=provider,  # type: ignore[arg-type]
        pipeline=pipeline,
        output_dir=output_dir,
        max_concurrent=1,
        use_rich=False,
        **kwargs,  # type: ignore[arg-type]
    )


class TestExternalFilenameRandomization:
    def test_external_provider_receives_randomized_filename_while_results_keep_source_path(
        self, output_dir: Path, source_file: Path, pipeline_spec: PipelineSpec
    ) -> None:
        provider = RecordingProvider()
        runner = _runner(provider, pipeline_spec, output_dir)
        raw_result, normalized_result, error_info = runner._process_document(
            pdf_path=source_file,
            example_id="annual_report_2026",
            product_type=ProductType.PARSE,
        )

        assert error_info is None
        assert provider.seen_path is not None
        assert provider.seen_path.name != source_file.name
        assert provider.seen_path.suffix == source_file.suffix
        assert provider.seen_bytes == source_file.read_bytes()
        assert not provider.seen_path.exists()
        assert raw_result is not None
        assert normalized_result is not None
        assert raw_result.request.source_file_path == str(source_file)
        assert normalized_result.request.source_file_path == str(source_file)

        saved_raw = json.loads((output_dir / "annual_report_2026.raw.json").read_text())
        saved_result = json.loads((output_dir / "annual_report_2026.result.json").read_text())
        assert saved_raw["request"]["source_file_path"] == str(source_file)
        assert saved_result["request"]["source_file_path"] == str(source_file)

    def test_llamacloud_provider_keeps_original_provider_visible_filename(
        self, output_dir: Path, source_file: Path
    ) -> None:
        provider = RecordingProvider()
        pipeline = PipelineSpec(
            pipeline_name="llamaparse_agentic",
            provider_name="llamaparse",
            product_type=ProductType.PARSE,
            config={},
        )
        runner = _runner(provider, pipeline, output_dir)

        _, _, error_info = runner._process_document(
            pdf_path=source_file,
            example_id="annual_report_2026",
            product_type=ProductType.PARSE,
        )

        assert error_info is None
        assert provider.seen_path == source_file

    def test_external_filename_randomization_can_be_disabled_with_pipeline_config(
        self, output_dir: Path, source_file: Path, pipeline_spec: PipelineSpec
    ) -> None:
        provider = RecordingProvider()
        pipeline = pipeline_spec.model_copy(update={"config": {"randomize_external_filename": False}})
        runner = _runner(provider, pipeline, output_dir)

        _, _, error_info = runner._process_document(
            pdf_path=source_file,
            example_id="annual_report_2026",
            product_type=ProductType.PARSE,
        )

        assert error_info is None
        assert provider.seen_path == source_file

    def test_test_case_path_also_randomizes_and_rewrites_request(
        self, output_dir: Path, source_file: Path, pipeline_spec: PipelineSpec
    ) -> None:
        provider = RecordingProvider()
        runner = _runner(provider, pipeline_spec, output_dir)
        test_case = ParseTestCase(test_id="annual_report_2026", file_path=source_file, group="benchmark-set")

        raw_result, _, error_info = runner._process_test_case(test_case, ProductType.PARSE)

        assert error_info is None
        assert provider.seen_path is not None
        assert provider.seen_path.name != source_file.name
        assert raw_result is not None
        assert raw_result.request.source_file_path == str(source_file)


class TestProviderRetries:
    def test_transient_errors_are_retried_through_shared_helper(
        self, output_dir: Path, source_file: Path, pipeline_spec: PipelineSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import parse_bench.inference.runner as runner_module

        monkeypatch.setattr(runner_module.time, "sleep", lambda _s: None)
        provider = RecordingProvider(fail_times=2)
        runner = _runner(provider, pipeline_spec, output_dir)

        _, _, error_info = runner._process_document(
            pdf_path=source_file,
            example_id="annual_report_2026",
            product_type=ProductType.PARSE,
        )

        assert error_info is None
        assert provider.calls == 3


class TestPerFileTimeoutResolution:
    """InferenceRunner resolves the effective per-file timeout with precedence:
    explicit run-level value > PipelineSpec.per_file_timeout > global default.

    These build the real runner and read the resolved attribute (exercising
    __init__'s resolution code rather than re-deriving the rule).
    """

    def test_default_when_neither_set(self, output_dir: Path, pipeline_spec: PipelineSpec) -> None:
        runner = _runner(RecordingProvider(), pipeline_spec, output_dir)
        assert runner.per_file_timeout == DEFAULT_PER_FILE_TIMEOUT_S
        assert DEFAULT_PER_FILE_TIMEOUT_S == 1800.0

    def test_pipeline_override_used_when_no_cli(self, output_dir: Path, pipeline_spec: PipelineSpec) -> None:
        pipeline = pipeline_spec.model_copy(update={"per_file_timeout": 500.0})
        runner = _runner(RecordingProvider(), pipeline, output_dir)
        assert runner.per_file_timeout == 500.0

    def test_explicit_cli_overrides_pipeline(self, output_dir: Path, pipeline_spec: PipelineSpec) -> None:
        pipeline = pipeline_spec.model_copy(update={"per_file_timeout": 500.0})
        runner = _runner(RecordingProvider(), pipeline, output_dir, per_file_timeout=9000.0)
        assert runner.per_file_timeout == 9000.0

    def test_explicit_cli_used_when_no_pipeline_override(self, output_dir: Path, pipeline_spec: PipelineSpec) -> None:
        runner = _runner(RecordingProvider(), pipeline_spec, output_dir, per_file_timeout=9000.0)
        assert runner.per_file_timeout == 9000.0


def test_metadata_records_version_timeouts_and_randomization(
    output_dir: Path, source_file: Path, pipeline_spec: PipelineSpec
) -> None:
    from parse_bench import __version__

    runner = _runner(RecordingProvider(), pipeline_spec, output_dir)
    test_case = ParseTestCase(test_id="annual_report_2026", file_path=source_file, group="benchmark-set")
    summary = runner._run_test_cases_sync([test_case], ProductType.PARSE, source_file.parent)
    runner.shutdown()

    assert summary.successful == 1
    metadata = json.loads((output_dir / "_metadata.json").read_text())
    assert metadata["parse_bench_version"] == __version__
    assert metadata["run_config"]["per_file_timeout"] == DEFAULT_PER_FILE_TIMEOUT_S
    assert metadata["run_config"]["timeout_retries"] == runner.timeout_retries
    assert metadata["run_config"]["randomize_external_filenames"] is True
