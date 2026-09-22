"""Local Apple Vision document recognition; see docs/apple-vision-documents.md."""

import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

from parse_bench.inference.providers.base import Provider, ProviderConfigError, ProviderPermanentError
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType


def _inside(inner: dict[str, float], outer: dict[str, float]) -> bool:
    """Match paragraph centers to structured regions, allowing OCR edge drift."""
    x, y = inner["x"] + inner["w"] / 2, inner["y"] + inner["h"] / 2
    return outer["x"] <= x <= outer["x"] + outer["w"] and outer["y"] <= y <= outer["y"] + outer["h"]


def _table_html(table: dict[str, Any]) -> str:
    cells = {(c["row"], c["column"]): c for c in table["cells"]}
    covered = set()
    rows = []
    for row in range(table["row_count"]):
        columns = []
        for column in range(table["column_count"]):
            if (row, column) in covered:
                continue
            cell = cells.get((row, column))
            if cell is None:
                columns.append("<td></td>")
                continue
            rs, cs = cell["row_span"], cell["column_span"]
            covered.update((r, c) for r in range(row, row + rs) for c in range(column, column + cs))
            attrs = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
            text = escape(cell["text"]).replace("\n", "<br>")
            columns.append(f"<td{attrs}>{text}</td>")
        rows.append("<tr>" + "".join(columns) + "</tr>")
    return "<table>\n" + "\n".join(rows) + "\n</table>"


def _document_items(document: dict[str, Any]) -> list[LayoutItemIR]:
    """Keep Vision's paragraph order; replace covered prose with tables/lists."""
    structured = [dict(t, kind="Table", md=_table_html(t)) for t in document["tables"]]
    for listing in document.get("lists", []):
        md = "\n".join(f"{escape(item['marker'] or '-')} {escape(item['text'])}" for item in listing["items"])
        structured.append(dict(listing, kind="List-item", md=md))
    paragraphs = document["paragraphs"]
    placed = set()
    blocks = []
    for paragraph in paragraphs:
        matches = [i for i, block in enumerate(structured) if _inside(paragraph["bbox"], block["bbox"])]
        if matches:
            for i in matches:
                if i not in placed:
                    blocks.append(structured[i])
                    placed.add(i)
        else:
            blocks.append(dict(paragraph, kind="Text", md=escape(paragraph["text"])))
    for i, block in enumerate(structured):
        if i not in placed:
            # ponytail: geometry fallback for regions absent from paragraphs; a unified Vision order would replace it.
            position = next((j for j, b in enumerate(blocks) if b["bbox"]["y"] > block["bbox"]["y"]), len(blocks))
            blocks.insert(position, block)
    if not blocks and document["text"]:
        blocks.append(
            {"text": document["text"], "kind": "Text", "md": escape(document["text"]), "bbox": document["bbox"]}
        )
    items = []
    for block in blocks:
        segment = LayoutSegmentIR(**block["bbox"], label=block["kind"])
        items.append(
            LayoutItemIR(
                type=block["kind"],
                md=block["md"],
                value=block.get("text", ""),
                html=block["md"] if block["kind"] == "Table" else "",
                bbox=segment,
                layout_segments=[segment],
            )
        )
    return items


@register_provider("apple_vision_documents")
class AppleVisionDocumentsProvider(Provider):
    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self.binary = str(
            self.base_config.get("binary")
            or os.environ.get("APPLE_VISION_DOCUMENTS_BIN")
            or shutil.which("apple-vision-documents")
            or Path(".build/apple-vision-documents").resolve()
        )
        self.dpi = self.base_config.get("dpi", 200)
        self.page_timeout = self.base_config.get("page_timeout", 120)
        for name, value in (("dpi", self.dpi), ("page_timeout", self.page_timeout)):
            if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ProviderConfigError(f"{name} must be a positive finite number")

    def _recognize(self, image: Path) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                [self.binary, str(image)], capture_output=True, text=True, timeout=self.page_timeout, check=False
            )
        except OSError as exc:
            raise ProviderConfigError(
                "Build scripts/apple_vision_documents.swift and set APPLE_VISION_DOCUMENTS_BIN"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderPermanentError(f"Apple Vision exceeded {self.page_timeout}s for one page") from exc
        if proc.returncode:
            raise ProviderPermanentError(f"Apple Vision exited {proc.returncode}: {proc.stderr.strip()}")
        try:
            result = json.loads(proc.stdout)
            if result["coordinate_system"] != "normalized_top_left" or not isinstance(result["documents"], list):
                raise ValueError("unexpected document schema or coordinate system")
            return result
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderPermanentError(f"Invalid Apple Vision JSON: {exc}") from exc

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if platform.system() != "Darwin" or int(platform.mac_ver()[0].split(".")[0]) < 26:
            raise ProviderConfigError("Apple Vision Documents requires macOS 26 or later")
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError("Apple Vision Documents supports PARSE only")
        path = Path(request.source_file_path)
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ProviderPermanentError(f"Expected an existing PDF: {path}")
        try:
            import pymupdf as fitz
        except ImportError as exc:
            raise ProviderConfigError("Install the local runner dependencies: uv sync --extra local") from exc
        started_at, started = datetime.now(), perf_counter()
        pages = []
        try:
            with fitz.open(path) as pdf, tempfile.TemporaryDirectory(prefix="apple-vision-") as temp:
                if pdf.needs_pass:
                    raise ProviderPermanentError("Password-protected PDF is not supported")
                for index, page in enumerate(pdf):
                    page_started = perf_counter()
                    image = Path(temp) / "page.png"
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72), colorspace=fitz.csRGB, alpha=False
                    )
                    pixmap.save(image)
                    render_ms = (perf_counter() - page_started) * 1000
                    result = self._recognize(image)
                    pages.append(
                        dict(
                            result,
                            page_index=index,
                            width=page.rect.width,
                            height=page.rect.height,
                            render_latency_ms=render_ms,
                            latency_in_ms=(perf_counter() - page_started) * 1000,
                        )
                    )
        except (ProviderConfigError, ProviderPermanentError):
            raise
        except Exception as exc:
            raise ProviderPermanentError(f"Apple Vision PDF processing failed: {exc}") from exc
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"pages": pages, "dpi": self.dpi},
            started_at=started_at,
            completed_at=datetime.now(),
            latency_in_ms=int((perf_counter() - started) * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        pages, layout_pages = [], []
        for page in raw_result.raw_output["pages"]:
            items = [item for document in page["documents"] for item in _document_items(document)]
            markdown = "\n\n".join(item.md for item in items)
            pages.append(PageIR(page_index=page["page_index"], markdown=markdown))
            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=page["page_index"] + 1,
                    width=page["width"],
                    height=page["height"],
                    md=markdown,
                    items=items,
                )
            )
        output = ParseOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown="\n\n".join(p.markdown for p in pages),
        )
        return InferenceResult(**raw_result.model_dump(exclude={"pipeline"}), output=output)
