"""Tests for the PyMuPDF4LLM native-HTML provider."""

import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import parse_bench.inference.providers.parse.pymupdf4llm as provider_module
from parse_bench.evaluation.layout_adapters.adapters import PyMuPDF4LLMLayoutAdapter
from parse_bench.inference.pipelines import get_pipeline, list_pipelines
from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.pymupdf4llm import PyMuPDF4LLMProvider
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

REPO_ROOT = Path(__file__).resolve().parents[5]


def test_only_one_pymupdf4llm_pipeline_is_registered() -> None:
    assert [name for name in list_pipelines() if name.startswith("pymupdf4llm")] == [
        "pymupdf4llm_markdown"
    ]


def test_pipeline_uses_modern_rapidocr_and_native_html() -> None:
    pipeline = get_pipeline("pymupdf4llm_markdown")

    assert pipeline.provider_name == "pymupdf4llm"
    assert pipeline.config == {
        "use_ocr": True,
        "ocr_backend": "rapidocr",
        "ocr_dpi": 150,
        "table_output": "html",
    }


def test_pymupdf4llm_extra_pins_modern_rapidocr_runtime() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["pymupdf4llm"] == ["pymupdf4llm==1.28.2", "rapidocr==3.9.2"]
    assert not any(dependency.startswith("rapidocr-onnxruntime") for dependency in extras["pymupdf4llm"])
    assert {package["version"] for package in lock["package"] if package["name"] == "rapidocr"} == {"3.9.2"}
    assert not any(
        package["name"] == "rapidocr-onnxruntime" and package["version"] == "1.2.3"
        for package in lock["package"]
    )


def test_markdown_options_are_declarative() -> None:
    provider = PyMuPDF4LLMProvider(
        "pymupdf4llm",
        {
            "use_ocr": True,
            "ocr_backend": "rapidocr",
            "ocr_dpi": 150,
            "table_output": "html",
        },
    )

    assert provider._markdown_options() == {
        "page_chunks": True,
        "show_progress": False,
        "use_ocr": True,
        "ocr_dpi": 150,
        "table_output": "html",
    }


def test_resolve_rapidocr_uses_modern_bundled_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def exec_ocr() -> None:
        return None

    requested: list[str] = []

    def import_module(name: str) -> SimpleNamespace:
        requested.append(name)
        return SimpleNamespace(exec_ocr=exec_ocr)

    monkeypatch.setattr(provider_module.importlib, "import_module", import_module)

    provider = PyMuPDF4LLMProvider("pymupdf4llm", {"ocr_backend": "rapidocr"})

    assert provider._resolve_ocr_function() is exec_ocr
    assert requested == ["pymupdf4llm.ocr.rapidocr_api"]


@pytest.mark.parametrize("backend", ["rapidtess", "tesseract", "rapidocr_onnxruntime"])
def test_markdown_options_reject_legacy_ocr_backends(backend: str) -> None:
    provider = PyMuPDF4LLMProvider("pymupdf4llm", {"ocr_backend": backend})

    with pytest.raises(ProviderConfigError, match="Unsupported.*OCR backend"):
        provider._markdown_options()


@pytest.mark.parametrize("ocr_dpi", [True, 0, -1, 150.0, "150"])
def test_markdown_options_reject_invalid_ocr_dpi(ocr_dpi: object) -> None:
    provider = PyMuPDF4LLMProvider("pymupdf4llm", {"ocr_dpi": ocr_dpi})

    with pytest.raises(ProviderConfigError, match="positive integer"):
        provider._markdown_options()


@pytest.mark.parametrize("table_output", [None, 1, "xml"])
def test_markdown_options_reject_invalid_table_output(table_output: object) -> None:
    provider = PyMuPDF4LLMProvider("pymupdf4llm", {"table_output": table_output})

    if table_output is None:
        assert "table_output" not in provider._markdown_options()
    else:
        with pytest.raises(ProviderConfigError, match="table_output"):
            provider._markdown_options()


def test_build_layout_page_preserves_native_html_raw_labels_and_bbox() -> None:
    native_html = "<table><tr><td>value</td></tr></table>"
    page = PyMuPDF4LLMProvider._build_layout_page(
        {
            "page_number": 1,
            "width": 200,
            "height": 100,
            "page_boxes": [
                {
                    "class": "table",
                    "bbox": [20, 10, 180, 90],
                    "pos": [0, len(native_html)],
                    "confidence": 0.8,
                }
            ],
        },
        raw_markdown=native_html,
    )

    assert page is not None
    item = page.items[0]
    assert item.md == native_html
    assert item.html == native_html
    assert item.layout_segments[0].label == "table"
    assert item.layout_segments[0].model_dump(include={"x", "y", "w", "h"}) == {
        "x": 0.1,
        "y": 0.1,
        "w": 0.8,
        "h": 0.8,
    }


def test_normalize_preserves_native_html_and_layout_grounding() -> None:
    native_html = "<table><tr><td>value</td></tr></table>"
    raw_output = {
        "pages": [
            {
                "page_index": 0,
                "page_number": 1,
                "text": native_html,
                "width": 100,
                "height": 100,
                "page_boxes": [
                    {"class": "table", "bbox": [0, 0, 100, 100], "pos": [0, len(native_html)]}
                ],
            }
        ],
        "num_pages": 1,
    }
    pipeline = PipelineSpec(
        pipeline_name="pymupdf4llm_markdown",
        provider_name="pymupdf4llm",
        product_type=ProductType.PARSE,
        config={
            "use_ocr": True,
            "ocr_backend": "rapidocr",
            "ocr_dpi": 150,
            "table_output": "html",
        },
    )
    request = InferenceRequest(
        example_id="example-1",
        source_file_path="/tmp/example.pdf",
        product_type=ProductType.PARSE,
    )
    now = datetime.now()
    raw_result = RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )

    result = PyMuPDF4LLMProvider("pymupdf4llm", pipeline.config).normalize(raw_result)

    assert isinstance(result.output, ParseOutput)
    assert result.output.layout_pages[0].items[0].md == native_html
    assert result.output.layout_pages[0].items[0].html == native_html
    assert result.output.markdown == native_html
    assert result.raw_output == raw_output

    adapter = PyMuPDF4LLMLayoutAdapter()
    assert adapter.matches(result)
    layout_output = adapter.to_layout_output(result)
    assert layout_output.model == LayoutDetectionModel.PYMUPDF4LLM_LAYOUT
    assert layout_output.image_width == 100
    assert layout_output.image_height == 100
    assert layout_output.predictions[0].bbox == [0.0, 0.0, 100.0, 100.0]
