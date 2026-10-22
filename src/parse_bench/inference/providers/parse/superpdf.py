"""Provider for superPDF (``@superpdf/cli`` on npm) PARSE.

superPDF reads PDFs with PDF.js and rules-based layout code (headings, lists,
tables, charts, reading order, page headers and footers). There is no language
model, no API key and no network access; pages with no text layer are read with
Tesseract (tesseract.js, LSTM), which the package ships with. Node.js 22+.

Each document is exported twice with the ``superpdf`` command:

- ``superpdf export <file> --format json``: every block with its type and box in
  PDF points, plus the page's pictures (layout, page headers and footers);
- ``superpdf export <file> --format markdown --tables html``: the Markdown export
  with each table written as an HTML table (header rows and merged cells kept).
  Each page ends with ``<!-- end of page N -->``.

PNG and JPEG inputs are first wrapped in a one-page PDF with the package's own
``imagePagePdf`` (``@superpdf/cli/core``), as superPDF does for images.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# Path to the ``superpdf`` command (``npm install -g @superpdf/cli@0.1.1``). Resolution
# order: ``SUPERPDF_BIN`` env var, then ``superpdf`` on ``PATH``.
SUPERPDF_CLI_VERSION = "0.1.1"
_SUPERPDF_BIN = Path(os.environ.get("SUPERPDF_BIN") or shutil.which("superpdf") or "superpdf")

# Writes a PNG or JPEG (argv[1]) as a one-page PDF (argv[2]), run from the package folder.
_IMAGE_TO_PDF_JS = (
    "const [src, dst] = process.argv.slice(1);"
    "const fs = await import('node:fs/promises');"
    "const { imagePagePdf } = await import('@superpdf/cli/core');"
    "await fs.writeFile(dst, await imagePagePdf(new Uint8Array(await fs.readFile(src))));"
)

# superPDF block types -> canonical layout labels (the layout label mapper is a lookup).
_BLOCK_TYPE_TO_LABEL: dict[str, str] = {
    "heading": "Section-header",
    "paragraph": "Text",
    "list": "List-item",
    "table": "Table",
    "chart": "Picture",
}
_FURNITURE_TO_LABEL: dict[str, str] = {"header": "Page-header", "footer": "Page-footer"}

_PAGE_END = re.compile(r"<!-- end of page (\d+) -->")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _split_pages(markdown: str) -> dict[int, str]:
    """Markdown export -> {page number: that page's Markdown}."""
    pages: dict[int, str] = {}
    start = 0
    for match in _PAGE_END.finditer(markdown):
        pages[int(match.group(1))] = markdown[start : match.start()].strip()
        start = match.end()
    return pages


def _block_text(block: dict[str, Any]) -> str:
    """Text of a block; a table's cells row by row (``a | b``)."""
    if block.get("type") == "table":
        rows = block.get("rows") or []
        return "\n".join(" | ".join(str(cell) for cell in row) for row in rows if isinstance(row, list))
    text = block.get("text")
    return text if isinstance(text, str) else ""


def _item(text: str, box: list[Any], width: float, height: float, label: str) -> LayoutItemIR | None:
    """A ``[x, y, width, height]`` box in points as a layout item with a [0, 1] segment."""
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if label in _FURNITURE_TO_LABEL.values():
        w, h = max(w, 1.0), max(h, 1.0)
    if w <= 0 or h <= 0:
        return None
    segment = LayoutSegmentIR(x=x / width, y=y / height, w=w / width, h=h / height, confidence=1.0, label=label)
    return LayoutItemIR(type=label, value=text, md=text, bbox=segment, layout_segments=[segment])


def _build_layout_pages(pages_raw: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """superPDF's JSON pages -> layout pages: body blocks in reading order, pictures, then headers and footers."""
    layout_pages: list[ParseLayoutPageIR] = []
    for page in pages_raw:
        width, height = page.get("width"), page.get("height")
        if not isinstance(width, int | float) or not isinstance(height, int | float) or width <= 0 or height <= 0:
            continue
        blocks = [b for b in page.get("blocks") or [] if isinstance(b, dict)]
        items: list[LayoutItemIR] = []
        for block in blocks:
            if block.get("type") in _FURNITURE_TO_LABEL:
                continue
            label = _BLOCK_TYPE_TO_LABEL.get(str(block.get("type", "")), "Text")
            item = _item(_block_text(block), block.get("bbox"), width, height, label)
            if item is not None:
                items.append(item)
        # Pictures: embedded images and drawn figures grouped, with the words on them.
        for picture in page.get("pictures") or []:
            box = [picture.get("x"), picture.get("y"), picture.get("width"), picture.get("height")]
            item = _item(str(picture.get("text") or ""), box, width, height, "Picture")
            if item is not None:
                items.append(item)
        if not items:
            continue
        furniture = [b for b in blocks if b.get("type") in _FURNITURE_TO_LABEL]
        for block in furniture:
            item = _item(_block_text(block), block.get("bbox"), width, height, _FURNITURE_TO_LABEL[block["type"]])
            if item is not None:
                items.append(item)
        layout_page = ParseLayoutPageIR(
            page_number=int(page.get("page", 1)),
            width=float(width),
            height=float(height),
            md=page.get("markdown", ""),
            items=items,
        )
        headers = [b["text"] for b in furniture if b["type"] == "header" and b.get("text")]
        footers = [b["text"] for b in furniture if b["type"] == "footer" and b.get("text")]
        if headers:
            layout_page.page_header_markdown = "\n".join(headers)
        if footers:
            layout_page.page_footer_markdown = "\n".join(footers)
        layout_pages.append(layout_page)
    return layout_pages


@register_provider("superpdf")
class SuperPdfProvider(Provider):
    """
    Provider for superPDF (local ``superpdf`` CLI, no API key).

    :param base_config: Recognized keys:
        - ``ocr_enabled``: bool (default True). False passes ``--no-ocr``.
        - ``timeout``: seconds per export (default 900).
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._ocr_enabled = bool(self.base_config.get("ocr_enabled", True))
        self._timeout = float(self.base_config.get("timeout", 900))

    def _run(self, command: list[str], cwd: Path | None = None) -> str:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=self._timeout, check=False, cwd=cwd)
        except FileNotFoundError as e:
            raise ProviderConfigError(
                f"{command[0]} not found. Install Node.js 22+ and "
                f"`npm install -g @superpdf/cli@{SUPERPDF_CLI_VERSION}`, or set SUPERPDF_BIN=/path/to/superpdf."
            ) from e
        if proc.returncode != 0:
            raise ProviderPermanentError(
                f"{' '.join(command[1:4])} failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}"
            )
        return proc.stdout

    def _export(self, path: str, *args: str) -> str:
        return self._run([str(_SUPERPDF_BIN), "export", path, *args, *([] if self._ocr_enabled else ["--no-ocr"])])

    def _image_as_pdf(self, path: Path, directory: str) -> str:
        """A PNG or JPEG as a one-page PDF the size of the image at 150 dpi (superPDF's ``imagePagePdf``)."""
        package = Path(os.path.realpath(shutil.which(str(_SUPERPDF_BIN)) or _SUPERPDF_BIN)).parent.parent
        target = str(Path(directory) / f"{path.stem}.pdf")
        node = shutil.which("node") or "node"
        self._run([node, "--input-type=module", "-e", _IMAGE_TO_PDF_JS, str(path.resolve()), target], cwd=package)
        return target

    def _parse(self, path: Path) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="superpdf_") as directory:
            pdf = self._image_as_pdf(path, directory) if path.suffix.lower() in _IMAGE_SUFFIXES else str(path)
            try:
                structure = json.loads(self._export(pdf, "--format", "json"))
            except json.JSONDecodeError as e:
                raise ProviderPermanentError(f"Failed to parse superpdf JSON output: {e}") from e
            page_markdown = _split_pages(self._export(pdf, "--format", "markdown", "--tables", "html"))
        pages = [
            {**page, "markdown": page_markdown.get(int(page.get("page", 0)), "")}
            for page in structure.get("pages") or []
        ]
        return {
            "pages": pages,
            "num_pages": len(pages),
            "text": "\n\n".join(p["markdown"] for p in pages),
            "output_format": "superpdf",
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"SuperPdfProvider only supports PARSE product type, got {request.product_type}"
            )
        path = Path(request.source_file_path)
        if not path.exists():
            raise ProviderPermanentError(f"File not found: {path}")
        started_at = datetime.now()
        try:
            raw_output = self._parse(path)
        except (ProviderPermanentError, ProviderConfigError):
            raise
        except subprocess.TimeoutExpired as e:
            raise ProviderPermanentError(f"superpdf export timed out after {self._timeout:.0f} s") from e
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e
        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"SuperPdfProvider only supports PARSE product type, got {raw_result.product_type}"
            )
        raw_pages = raw_result.raw_output.get("pages", [])
        pages = [
            PageIR(page_index=int(p.get("page", i + 1)) - 1, markdown=p.get("markdown", ""))
            for i, p in enumerate(raw_pages)
        ]
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=_build_layout_pages(raw_pages),
            markdown=raw_result.raw_output.get("text", ""),
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
