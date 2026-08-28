from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from parse_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from parse_bench.inference.providers.parse import PARSE_PROVIDER_MODULES
from parse_bench.inference.providers.parse._multipage_image import (
    IMAGE_BACKED_PDF_PROVIDERS,
    PARSE_PROVIDER_PDF_CLASSIFICATIONS,
)
from parse_bench.inference.providers.registry import registered_providers
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

ADAPTER_PROVIDERS = [
    (spec.module_name, spec.class_name, spec.dpi) for spec in IMAGE_BACKED_PDF_PROVIDERS if spec.execution == "adapter"
]


def _pipeline(module_name: str) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=f"{module_name}_test",
        provider_name=module_name,
        product_type=ProductType.PARSE,
    )


def _request(source: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="document",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )


def _provider(module_name: str, class_name: str, dpi: int) -> Any:
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    provider = object.__new__(getattr(module, class_name))
    defaults = {
        "_dpi": dpi,
        "_timeout": 30,
        "_server_url": "http://provider.invalid",
        "_api_format": "openai",
        "_task": "parse",
        "_model": "test-model",
        "_served_model_name": "test-model",
        "_prompt_mode": "parse",
    }
    for name, value in defaults.items():
        setattr(provider, name, value)
    return provider


def _install_local_boundary(
    provider: Any,
    module_name: str,
    *,
    failure: Exception | None = None,
    fail_on_call: int = 1,
) -> list[int]:
    calls: list[int] = []

    def raw_output(page_number: int) -> dict[str, object]:
        if module_name == "infinity_parser2":
            return {
                "result": json.dumps(
                    [
                        {
                            "page": 1,
                            "category": "text",
                            "text": f"page {page_number}",
                            "bbox": [0, 0, 8, 8],
                        }
                    ]
                ),
                "_config": {"page_width": 8, "page_height": 8},
                "input_tokens": page_number * 10,
                "cost_usd": page_number / 10,
                "num_api_calls": 1,
            }
        return {
            "markdown": f"page {page_number}",
            "input_tokens": page_number * 10,
            "cost_usd": page_number / 10,
            "num_api_calls": 1,
        }

    if module_name == "infinity_parser2":

        def parse_document(path: str) -> dict[str, object]:
            calls.append(len(calls) + 1)
            if failure is not None and calls[-1] == fail_on_call:
                raise failure
            return raw_output(calls[-1])

        provider._parse_document = parse_document
    else:

        async def run_inference_async(*args: object, **kwargs: object) -> dict[str, object]:
            calls.append(len(calls) + 1)
            if failure is not None and calls[-1] == fail_on_call:
                raise failure
            return raw_output(calls[-1])

        provider._run_inference_async = run_inference_async

    return calls


def _mock_two_page_pdf(source: Path, monkeypatch: pytest.MonkeyPatch) -> list[Image.Image]:
    rendered: list[Image.Image] = []
    monkeypatch.setattr("pdf2image.pdfinfo_from_path", lambda path: {"Pages": 2})

    def render_page(path: str, dpi: int, first_page: int, last_page: int) -> list[Image.Image]:
        assert Path(path) == source
        assert first_page == last_page
        image = Image.new("RGB", (8 + first_page, 8), "white")
        rendered.append(image)
        return [image]

    monkeypatch.setattr("pdf2image.convert_from_path", render_page)
    return rendered


@pytest.mark.parametrize(("module_name", "class_name", "expected_dpi"), ADAPTER_PROVIDERS)
def test_adapter_provider_runs_and_normalizes_actual_pdf_pages(
    module_name: str,
    class_name: str,
    expected_dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_two_page_pdf(source, monkeypatch)
    provider = _provider(module_name, class_name, expected_dpi)
    boundary_calls = _install_local_boundary(provider, module_name)

    raw_result = provider.run_inference(_pipeline(module_name), _request(source))
    result = provider.normalize(raw_result)

    assert boundary_calls == [1, 2]
    assert raw_result.raw_output["num_pages"] == 2
    assert raw_result.raw_output["input_tokens"] == 30
    assert raw_result.raw_output["cost_usd"] == pytest.approx(0.3)
    assert raw_result.raw_output["num_api_calls"] == 2
    envelope = raw_result.raw_output["_parse_bench_multipage"]
    assert isinstance(envelope, dict)
    assert [page["page_index"] for page in envelope["pages"]] == [0, 1]
    assert isinstance(result.output, ParseOutput)
    assert result.output.markdown == "page 1\n\npage 2"
    assert [(page.page_index, page.markdown) for page in result.output.pages] == [
        (0, "page 1"),
        (1, "page 2"),
    ]
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize(("module_name", "class_name", "expected_dpi"), ADAPTER_PROVIDERS)
def test_adapter_provider_preserves_single_image_path(
    module_name: str,
    class_name: str,
    expected_dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(source)
    provider = _provider(module_name, class_name, expected_dpi)
    boundary_calls = _install_local_boundary(provider, module_name)
    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda *args, **kwargs: pytest.fail("single-image inference must not rasterize a PDF"),
    )

    raw_result = provider.run_inference(_pipeline(module_name), _request(source))
    result = provider.normalize(raw_result)

    assert boundary_calls == [1]
    assert "_parse_bench_multipage" not in raw_result.raw_output
    assert isinstance(result.output, ParseOutput)
    assert result.output.markdown == "page 1"


@pytest.mark.parametrize(("module_name", "class_name", "expected_dpi"), ADAPTER_PROVIDERS)
def test_adapter_provider_closes_rendered_page_when_persistence_fails(
    module_name: str,
    class_name: str,
    expected_dpi: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_two_page_pdf(source, monkeypatch)
    provider = _provider(module_name, class_name, expected_dpi)
    _install_local_boundary(provider, module_name)
    multipage_module = importlib.import_module("parse_bench.inference.providers.parse._multipage_image")

    def fail_save(image: Image.Image, destination: Path) -> None:
        raise ProviderPermanentError("save failed")

    monkeypatch.setattr(multipage_module, "_save_png", fail_save)

    with pytest.raises(ProviderPermanentError, match="save failed"):
        provider.run_inference(_pipeline(module_name), _request(source))

    assert len(rendered) == 1
    with pytest.raises(ValueError, match="Operation on closed image"):
        rendered[0].getpixel((0, 0))


def test_paddleocr_page_two_failure_aborts_without_partial_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_two_page_pdf(source, monkeypatch)
    provider = _provider("paddleocr", "PaddleOCRProvider", 144)
    calls = 0
    successful_result = None

    async def run_inference_async(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProviderPermanentError("page two is invalid")
        return {"markdown": "page 1"}

    provider._run_inference_async = run_inference_async

    with pytest.raises(ProviderPermanentError, match="page two is invalid"):
        successful_result = provider.run_inference(_pipeline("paddleocr"), _request(source))

    assert successful_result is None
    assert calls == 2
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize(("module_name", "class_name", "expected_dpi"), ADAPTER_PROVIDERS)
@pytest.mark.parametrize(
    ("failure", "expected_error", "expected_calls"),
    [
        (ProviderPermanentError("page two is invalid"), ProviderPermanentError, [1, 2]),
        (ProviderTransientError("page two timed out"), None, [1, 2, 3]),
    ],
    ids=["permanent", "transient"],
)
def test_adapter_provider_owns_page_failure_handling(
    module_name: str,
    class_name: str,
    expected_dpi: int,
    failure: Exception,
    expected_error: type[Exception] | None,
    expected_calls: list[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.touch()
    rendered = _mock_two_page_pdf(source, monkeypatch)
    provider = _provider(module_name, class_name, expected_dpi)
    calls = _install_local_boundary(provider, module_name, failure=failure, fail_on_call=2)
    monkeypatch.setattr("parse_bench.inference.providers.parse._multipage_image.time.sleep", lambda delay: None)

    if expected_error is not None:
        with pytest.raises(expected_error, match=str(failure)):
            provider.run_inference(_pipeline(module_name), _request(source))
    else:
        assert provider.run_inference(_pipeline(module_name), _request(source)) is not None

    assert calls == expected_calls
    assert len(rendered) == 2
    for image in rendered:
        with pytest.raises(ValueError, match="Operation on closed image"):
            image.getpixel((0, 0))


@pytest.mark.parametrize(
    "module_name",
    [spec.module_name for spec in IMAGE_BACKED_PDF_PROVIDERS if spec.execution == "direct"],
)
def test_refactored_providers_do_not_call_pdf2image_eagerly(module_name: str) -> None:
    module = importlib.import_module(f"parse_bench.inference.providers.parse.{module_name}")
    tree = ast.parse(inspect.getsource(module))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "convert_from_path"
    ]

    assert calls == []


def test_shared_rasterizer_bounds_every_convert_from_path_call() -> None:
    module = importlib.import_module("parse_bench.inference.providers.parse._multipage_image")
    source_path = Path(inspect.getsourcefile(module) or "")
    tree = ast.parse(source_path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "convert_from_path"
    ]

    assert len(calls) == 1
    assert {keyword.arg for keyword in calls[0].keywords} >= {"first_page", "last_page"}


def test_every_registered_parse_provider_has_explicit_pdf_classification() -> None:
    registered = {
        provider_name: (provider_class.__module__.rsplit(".", 1)[-1], provider_class.__name__)
        for provider_name, provider_class in registered_providers().items()
        if provider_class.__module__.startswith("parse_bench.inference.providers.parse.")
    }

    classified = {spec.provider_name: spec for spec in PARSE_PROVIDER_PDF_CLASSIFICATIONS}
    assert len(classified) == len(PARSE_PROVIDER_PDF_CLASSIFICATIONS)
    assert set(PARSE_PROVIDER_MODULES) == {spec.module_name for spec in classified.values()}
    assert set(classified) == set(registered)
    assert {
        provider_name: (spec.module_name, spec.class_name) for provider_name, spec in classified.items()
    } == registered
    assert {spec.pdf_handling for spec in classified.values()} == {
        "local-page-raster",
        "no-local-page-raster",
    }
    for spec in classified.values():
        if spec.pdf_handling == "local-page-raster":
            assert spec.execution in {"adapter", "direct", "kdl"}
        else:
            assert spec.execution is None

    local_raster = {
        provider_name: (spec.module_name, spec.class_name)
        for provider_name, spec in classified.items()
        if spec.pdf_handling == "local-page-raster"
    }
    assert {
        spec.provider_name: (spec.module_name, spec.class_name) for spec in IMAGE_BACKED_PDF_PROVIDERS
    } == local_raster
    assert {spec.execution for spec in IMAGE_BACKED_PDF_PROVIDERS} == {"adapter", "direct", "kdl"}


def test_local_raster_metadata_reads_declared_provider_defaults() -> None:
    expected_defaults = {
        "amazon_nova": 150,
        "anthropic": 150,
        "chandra2": 192,
        "deepseekocr2": 150,
        "dots_ocr_parse": 150,
        "falconocr": 200,
        "gemma4": 150,
        "google": 150,
        "granite_vision": 150,
        "infinity_parser2": 300,
        "kdl_frontier_nano": 144,
        "mineru25": 150,
        "mineru2605pro": 150,
        "mineru_diffusion": 150,
        "nemotron_omni": 150,
        "openai": 150,
        "paddleocr": 150,
        "qwen3_5": 150,
        "surya2": 192,
        "tesseract": 300,
        "textract": 300,
        "unlimitedocr": 300,
    }

    assert {spec.provider_name: spec.dpi for spec in IMAGE_BACKED_PDF_PROVIDERS} == expected_defaults
    for spec in IMAGE_BACKED_PDF_PROVIDERS:
        module = importlib.import_module(f"parse_bench.inference.providers.parse.{spec.module_name}")
        provider_class = getattr(module, spec.class_name)
        assert provider_class.PDF_RENDER_DPI == expected_defaults[spec.provider_name]
