"""superPDF (``superpdf export``) output -> ParseOutput -> LayoutOutput."""

import json
import subprocess
from pathlib import Path

import pytest

from parse_bench.evaluation.layout_adapters.adapters import SuperPdfLayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import ProviderConfigError, ProviderPermanentError
from parse_bench.inference.providers.parse import superpdf
from parse_bench.inference.providers.parse.superpdf import SuperPdfProvider, _build_layout_pages
from parse_bench.schemas.layout_detection_output import LAYOUT_MODEL_INFO, LayoutDetectionModel
from parse_bench.schemas.pipeline_io import InferenceRequest

# One page as `superpdf export --format json` writes it (fields the provider reads).
_PAGE = {
    "page": 1,
    "width": 612,
    "height": 792,
    "blocks": [
        {"type": "header", "text": "Annual report", "bbox": [61.2, 20, 100, 0]},
        {"type": "heading", "level": 1, "text": "Title", "bbox": [61.2, 79.2, 306, 15.84]},
        {"type": "paragraph", "text": "body", "bbox": [61.2, 158.4, 489.6, 79.2]},
        {"type": "table", "rows": [["a", "b"], ["1", "2"]], "text": "a | b", "bbox": [61.2, 396, 489.6, 158.4]},
        {"type": "chart", "text": "Sales by year", "bbox": [61.2, 560, 200, 100]},
        {"type": "list", "text": "- one", "bbox": [61.2, 680, 100, 0]},
        {"type": "footer", "text": "7", "bbox": [300, 760, 6, 9]},
    ],
    "pictures": [{"x": 400, "y": 560, "width": 100, "height": 80, "text": "Logo"}],
}
_PAGE_1_MD = "# Title\n\nbody\n\n<table>\n<tr><td>a</td><td>b</td></tr>\n</table>"
_MARKDOWN = f"{_PAGE_1_MD}\n\n<!-- end of page 1 -->\n\n<!-- end of page 2 -->\n"


def test_build_layout_pages_labels_boxes_and_furniture() -> None:
    (page,) = _build_layout_pages([{**_PAGE, "markdown": "# Title"}])
    assert page.page_number == 1 and page.width == 612.0 and page.md == "# Title"
    # Body blocks in order, then pictures, then page header and footer; a zero-height list item is dropped.
    assert [item.type for item in page.items] == [
        "Section-header",
        "Text",
        "Table",
        "Picture",
        "Picture",
        "Page-header",
        "Page-footer",
    ]
    heading = page.items[0].bbox
    assert (
        heading is not None
        and abs(heading.x - 0.1) < 1e-9
        and abs(heading.y - 0.1) < 1e-9
        and abs(heading.w - 0.5) < 1e-9
    )
    assert page.items[2].value == "a | b\n1 | 2"
    assert page.items[3].value == "Sales by year" and page.items[4].value == "Logo"
    assert page.items[5].bbox is not None and page.items[5].bbox.h > 0  # zero-height header kept 1 pt tall
    assert page.page_header_markdown == "Annual report" and page.page_footer_markdown == "7"


def test_build_layout_pages_skips_pages_without_size_or_regions() -> None:
    assert _build_layout_pages([{"page": 1, "blocks": _PAGE["blocks"]}]) == []
    assert _build_layout_pages([{"page": 1, "width": 612, "height": 792, "blocks": [_PAGE["blocks"][0]]}]) == []


def _fake_cli(monkeypatch: pytest.MonkeyPatch, calls: list[list[str]]) -> None:
    def run(command, **kwargs):
        calls.append(command)
        if "-e" in command:  # image -> PDF through the package's core export
            Path(command[-1]).write_bytes(b"%PDF-1.7")
            out = ""
        elif "json" in command:
            out = json.dumps({"pages": [_PAGE, {**_PAGE, "page": 2, "blocks": [], "pictures": []}]})
        else:
            out = _MARKDOWN
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")

    monkeypatch.setattr(superpdf, "_SUPERPDF_BIN", Path("/opt/bin/superpdf"))
    monkeypatch.setattr(superpdf.subprocess, "run", run)


def _infer(tmp_path, monkeypatch, calls, config=None, name="doc.pdf"):
    source = tmp_path / name
    source.write_bytes(b"%PDF-1.7")
    _fake_cli(monkeypatch, calls)
    pipeline = get_pipeline("superpdf_reader")
    provider = SuperPdfProvider("superpdf", {**pipeline.config, **(config or {})})
    request = InferenceRequest(example_id="layout/doc", source_file_path=str(source), product_type="parse")
    return provider.normalize(provider.run_inference(pipeline, request))


def test_provider_runs_both_exports_and_splits_pages(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    result = _infer(tmp_path, monkeypatch, calls)
    assert calls[0][:2] == ["/opt/bin/superpdf", "export"]
    assert calls[0][3:] == ["--format", "json"] and calls[1][3:] == ["--format", "markdown", "--tables", "html"]
    assert [p.markdown for p in result.output.pages] == [_PAGE_1_MD, ""]
    assert result.output.markdown == f"{_PAGE_1_MD}\n\n"
    assert len(result.output.layout_pages) == 1


def test_provider_wraps_images_in_a_pdf_first(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    _infer(tmp_path, monkeypatch, calls, name="scan.jpg")
    assert "@superpdf/cli/core" in calls[0][3] and calls[0][4].endswith("scan.jpg")
    assert calls[1][2] == calls[0][5] and calls[1][2].endswith("scan.pdf")


def test_provider_no_ocr_flag(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    _infer(tmp_path, monkeypatch, calls, {"ocr_enabled": False})
    assert all(call[-1] == "--no-ocr" for call in calls)


def test_adapter_resolves_to_superpdf_layout(tmp_path, monkeypatch) -> None:
    result = _infer(tmp_path, monkeypatch, [])
    adapter = create_layout_adapter_for_result(result)
    assert isinstance(adapter, SuperPdfLayoutAdapter)
    layout = adapter.to_layout_output(result)
    assert layout.model is LayoutDetectionModel.SUPERPDF_LAYOUT
    assert LAYOUT_MODEL_INFO[LayoutDetectionModel.SUPERPDF_LAYOUT]["name"] == "superPDF Layout"
    assert (layout.image_width, layout.image_height) == (612, 792)
    x1, y1, x2, y2 = layout.predictions[0].bbox
    assert (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)) == (61.2, 79.2, 367.2, 95.04)


def test_missing_cli_and_failed_export_raise(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    request = InferenceRequest(example_id="x", source_file_path=str(pdf), product_type="parse")
    pipeline = get_pipeline("superpdf_reader")
    monkeypatch.setattr(superpdf, "_SUPERPDF_BIN", tmp_path / "no-such-superpdf")
    with pytest.raises(ProviderConfigError, match="npm install -g @superpdf/cli"):
        SuperPdfProvider("superpdf", pipeline.config).run_inference(pipeline, request)

    def fail(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="superPDF: not a PDF")

    monkeypatch.setattr(superpdf.subprocess, "run", fail)
    with pytest.raises(ProviderPermanentError, match="not a PDF"):
        SuperPdfProvider("superpdf", pipeline.config).run_inference(pipeline, request)
