"""Tests for the PyMuPDF4LLM provider and table normalization."""

import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import parse_bench.inference.providers.parse.pymupdf4llm as provider_module
from parse_bench.evaluation.layout_adapters.adapters import PyMuPDF4LLMLayoutAdapter
from parse_bench.inference.pipelines import get_pipeline, list_pipelines
from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.pymupdf4llm import (
    PyMuPDF4LLMProvider,
    convert_pipe_tables_to_html,
)
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from parse_bench.schemas.product import ProductType

REPO_ROOT = Path(__file__).resolve().parents[5]


def test_only_one_pymupdf4llm_pipeline_is_registered() -> None:
    assert [name for name in list_pipelines() if name.startswith("pymupdf4llm")] == ["pymupdf4llm_markdown"]


def test_pipeline_uses_rapidtess_at_150_dpi() -> None:
    pipeline = get_pipeline("pymupdf4llm_markdown")

    assert pipeline.provider_name == "pymupdf4llm"
    assert pipeline.config == {
        "use_ocr": True,
        "ocr_backend": "rapidtess",
        "ocr_dpi": 150,
    }


def test_pymupdf4llm_extra_pins_compatible_rapidocr_runtime() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert "pymupdf4llm==1.28.0" in extras["pymupdf4llm"]
    assert "rapidocr-onnxruntime==1.2.3" in extras["pymupdf4llm"]
    assert not any(dependency.startswith("rapidocr-onnxruntime") for dependency in extras["runners"])
    assert {package["version"] for package in lock["package"] if package["name"] == "rapidocr-onnxruntime"} == {
        "1.2.3",
        "1.4.4",
    }


def test_markdown_options_are_declarative() -> None:
    provider = PyMuPDF4LLMProvider(
        "pymupdf4llm",
        {"use_ocr": True, "ocr_backend": "rapidtess", "ocr_dpi": 150},
    )

    assert provider._markdown_options() == {
        "page_chunks": True,
        "show_progress": False,
        "use_ocr": True,
        "ocr_dpi": 150,
    }


def test_resolve_rapidtess_uses_bundled_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def exec_ocr() -> None:
        return None

    requested: list[str] = []

    def import_module(name: str) -> SimpleNamespace:
        requested.append(name)
        return SimpleNamespace(TESSDATA="/usr/share/tessdata", exec_ocr=exec_ocr)

    monkeypatch.setattr(provider_module.importlib, "import_module", import_module)

    provider = PyMuPDF4LLMProvider("pymupdf4llm", {"ocr_backend": "rapidtess"})

    assert provider._resolve_ocr_function() is exec_ocr
    assert requested == ["pymupdf4llm.ocr.rapidtess_api"]


def test_resolve_rapidtess_requires_tesseract_language_data(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SimpleNamespace(text_detector=lambda: None)
    ocr_module = SimpleNamespace(ENGINE=engine, TESSDATA=None, exec_ocr=lambda: None)
    monkeypatch.setattr(provider_module.importlib, "import_module", lambda _name: ocr_module)

    provider = PyMuPDF4LLMProvider("pymupdf4llm", {"ocr_backend": "rapidtess"})

    with pytest.raises(ProviderConfigError, match="Tesseract language data was not found"):
        provider._resolve_ocr_function()


@pytest.mark.parametrize("ocr_dpi", [True, 0, -1, 150.0, "150"])
def test_markdown_options_reject_invalid_ocr_dpi(ocr_dpi: object) -> None:
    provider = PyMuPDF4LLMProvider("pymupdf4llm", {"ocr_dpi": ocr_dpi})

    with pytest.raises(ProviderConfigError, match="positive integer"):
        provider._markdown_options()


def test_build_layout_page_preserves_raw_labels_and_normalizes_bbox() -> None:
    markdown = "table content"
    page = PyMuPDF4LLMProvider._build_layout_page(
        {
            "page_number": 1,
            "width": 200,
            "height": 100,
            "page_boxes": [
                {
                    "class": "table",
                    "bbox": [20, 10, 180, 90],
                    "pos": [0, len(markdown)],
                    "confidence": 0.8,
                }
            ],
        },
        raw_markdown=markdown,
    )

    assert page is not None
    assert page.items[0].layout_segments[0].label == "table"
    assert page.items[0].layout_segments[0].model_dump(include={"x", "y", "w", "h"}) == {
        "x": 0.1,
        "y": 0.1,
        "w": 0.8,
        "h": 0.8,
    }


def test_convert_pipe_tables_preserves_alignment_and_uneven_rows() -> None:
    markdown = "| Name | Q1 |\n| :--- | ---: |\n| Alpha | 10 | extra |\n| Beta |"

    converted = convert_pipe_tables_to_html(markdown)

    assert '<th style="text-align:left">Name</th>' in converted
    assert '<th style="text-align:right">Q1</th>' in converted
    assert "<td>extra</td>" in converted
    assert converted.count("<td") == 6


def test_convert_pipe_tables_preserves_interior_blank_rows() -> None:
    markdown = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| | |\n| 3 | 4 |"

    converted = convert_pipe_tables_to_html(markdown)

    assert "<tr><td></td><td></td></tr>" in converted
    assert converted.count("<tr>") == 4


def test_convert_pipe_tables_does_not_count_blank_rows_toward_minimum_width() -> None:
    markdown = "| Header |\n| --- | --- |\n| Value |\n| | |"

    assert convert_pipe_tables_to_html(markdown) == markdown


@pytest.mark.parametrize("separator", ["|-|-|", "|:--|--:|"])
def test_convert_pipe_tables_accepts_short_gfm_separators(separator: str) -> None:
    markdown = f"| A | B |\n{separator}\n| 1 | 2 |"

    converted = convert_pipe_tables_to_html(markdown)

    assert "<table>" in converted
    assert ">A</th>" in converted
    assert ">B</th>" in converted
    assert ">1</td>" in converted
    assert ">2</td>" in converted


def test_convert_pipe_tables_drops_empty_fallback_header_without_promoting_body() -> None:
    markdown = "| | |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"

    converted = convert_pipe_tables_to_html(markdown)

    assert "<thead>" not in converted
    assert "<th" not in converted
    assert converted.count("<tr>") == 2
    assert "<td>1</td><td>2</td>" in converted


def test_convert_pipe_tables_preserves_escaped_and_code_pipes() -> None:
    markdown = "| Expression | Meaning |\n| --- | --- |\n| left \\| right | `a|b` |"

    converted = convert_pipe_tables_to_html(markdown)

    assert "left | right" in converted
    assert "<code>a|b</code>" in converted


def test_convert_pipe_tables_escapes_raw_html() -> None:
    markdown = "| Name | Value |\n| --- | --- |\n| <script>alert(1)</script> | safe |"

    converted = convert_pipe_tables_to_html(markdown)

    assert "<script>" not in converted
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in converted


def test_convert_pipe_tables_preserves_line_breaks_in_cells() -> None:
    markdown = "| Header<br>Detail |\n| --- |\n| Value<br/>More |"

    converted = convert_pipe_tables_to_html(markdown)

    assert "<th>Header<br>Detail</th>" in converted
    assert "<td>Value<br>More</td>" in converted
    assert "&lt;br" not in converted


def test_convert_pipe_tables_supports_valid_one_column_tables() -> None:
    markdown = "| Charge |\n| --- |\n| 0% to 15% |"

    converted = convert_pipe_tables_to_html(markdown)

    assert "<table>" in converted
    assert "<th>Charge</th>" in converted
    assert "<td>0% to 15%</td>" in converted


@pytest.mark.parametrize(
    "markdown",
    [
        "Use `left | right` as an example.\n| This is not a table |",
        "```markdown\n| A | B |\n| --- | --- |\n| 1 | 2 |\n```",
        "<div>\n| A | B |\n| --- | --- |\n| 1 | 2 |\n</div>",
        "<table><tbody><tr><td>existing</td></tr></tbody></table>",
    ],
)
def test_convert_pipe_tables_preserves_non_table_or_protected_content(markdown: str) -> None:
    assert convert_pipe_tables_to_html(markdown) == markdown


def test_convert_pipe_tables_is_idempotent() -> None:
    markdown = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    converted = convert_pipe_tables_to_html(markdown)

    assert convert_pipe_tables_to_html(converted) == converted


def test_normalize_builds_layout_before_table_conversion() -> None:
    markdown = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    raw_output = {
        "pages": [
            {
                "page_index": 0,
                "page_number": 1,
                "text": markdown,
                "width": 100,
                "height": 100,
                "page_boxes": [{"class": "table", "bbox": [0, 0, 100, 100], "pos": [0, len(markdown)]}],
            }
        ],
        "num_pages": 1,
    }
    pipeline = PipelineSpec(
        pipeline_name="pymupdf4llm_markdown",
        provider_name="pymupdf4llm",
        product_type=ProductType.PARSE,
        config={"use_ocr": True, "ocr_backend": "rapidtess", "ocr_dpi": 150},
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
    assert result.output.layout_pages[0].items[0].md == markdown
    assert "<table>" in result.output.layout_pages[0].items[0].html
    assert "<table>" in result.output.markdown
    assert result.raw_output == raw_output

    adapter = PyMuPDF4LLMLayoutAdapter()
    assert adapter.matches(result)
    layout_output = adapter.to_layout_output(result)
    assert layout_output.model == LayoutDetectionModel.PYMUPDF4LLM_LAYOUT
    assert layout_output.image_width == 100
    assert layout_output.image_height == 100
    assert layout_output.predictions[0].bbox == [0.0, 0.0, 100.0, 100.0]
