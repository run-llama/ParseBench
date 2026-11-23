"""Provider for Indra (``indra-ocr``), a local CPU document parser.

Runs the ``indra-ocr`` CLI once per file with ``--format json``: one JSON line per page carrying
the page markdown and Indra's layout blocks (label + bbox in render pixels). Install it with
``pip install indra-ocr`` (models download on first use) and put ``indra-ocr`` on ``PATH``, or
point ``INDRA_OCR_BIN`` at the executable.
"""

import json as _json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.parse.liteparse import LiteParseProvider
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

_INDRA_BIN = os.environ.get("INDRA_OCR_BIN") or shutil.which("indra-ocr") or "indra-ocr"

# Indra block labels -> canonical labels. Pictures come from the detector's picture channel.
INDRA_LABEL_MAP: dict[str, str] = {
    "Text": "Text",
    "Section-Header": "Section-header",
    "Page-Header": "Page-header",
    "Page-Footer": "Page-footer",
    "List-Group": "List-item",
    "Table": "Table",
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Table-Of-Contents": "Document Index",
    "Form": "Form",
    "Picture": "Picture",
}

# Indra renders PDF pages at ``dpi`` pixels per inch, lowered for pages whose render would pass
# this many pixels (indra_ocr.document.MAX_PAGE_PIXELS); its block boxes are in those pixels.
_INDRA_MAX_PAGE_PIXELS = 100e6
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jfif", ".tif", ".tiff", ".bmp", ".webp"}


def _page_frames(path: Path, dpi: float) -> list[tuple[float, float, float]]:
    """(width, height, pixels per point) of each page in Indra's frame; images are 1:1 pixels."""
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        try:
            from PIL import Image
        except ImportError as e:
            raise ProviderConfigError("indra on image inputs needs pillow: pip install 'parse-bench[indra]'") from e
        with Image.open(path) as im:
            return [(float(im.width), float(im.height), 1.0)]
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        frames = []
        for i in range(len(doc)):
            w, h = doc[i].get_size()
            k = min(dpi / 72.0, (_INDRA_MAX_PAGE_PIXELS / max(w * h, 1.0)) ** 0.5)
            frames.append((float(w), float(h), k))
        return frames
    finally:
        doc.close()


def _build_layout_pages(pages_raw: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Indra blocks (bbox in render pixels) -> layout pages in the [0, 1] frame."""
    layout_pages: list[ParseLayoutPageIR] = []
    for page in pages_raw:
        w, h, k = page.get("width"), page.get("height"), page.get("scale") or 1.0
        if not w or not h:
            continue
        items: list[LayoutItemIR] = []
        for block in page.get("blocks") or []:
            label = INDRA_LABEL_MAP.get(str(block.get("label", "")))
            bbox = block.get("bbox")
            if label is None or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            x0, y0, x1, y1 = (float(v) / k for v in bbox[:4])
            if x1 <= x0 or y1 <= y0:
                continue
            seg = LayoutSegmentIR(x=x0 / w, y=y0 / h, w=(x1 - x0) / w, h=(y1 - y0) / h, confidence=1.0, label=label)
            text = block.get("text", "") or ""
            items.append(LayoutItemIR(type=label, value=text, md=text, bbox=seg, layout_segments=[seg]))
        if items:
            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=int(page.get("page_index", 0)) + 1,
                    width=float(w),
                    height=float(h),
                    md=page.get("markdown", "") or "",
                    items=items,
                )
            )
    return layout_pages


@register_provider("indra")
class IndraProvider(Provider):
    """Indra, run locally on CPU through its CLI."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        :param base_config: Recognized keys:
            - `dpi`: float (default 200, Indra's default)
            - `math`: bool, write math as LaTeX (``--math``)
            - `keep_furniture`: bool, keep running headers / footers (``--keep-furniture``)
            - `cores`: int, cores per document (``--cores``)
            - `mode`: "auto" | "ocr" | "text" (``--mode``)
        """
        super().__init__(provider_name, base_config)
        self._dpi = float(self.base_config.get("dpi", 200.0))

    def _cli_args(self, path: str) -> list[str]:
        args = [_INDRA_BIN, path, "--format", "json", "--dpi", str(self._dpi)]
        if self.base_config.get("math"):
            args.append("--math")
        if self.base_config.get("keep_furniture"):
            args.append("--keep-furniture")
        if self.base_config.get("cores"):
            args.extend(["--cores", str(self.base_config["cores"])])
        if self.base_config.get("mode"):
            args.extend(["--mode", str(self.base_config["mode"])])
        return args

    def _parse(self, path: Path) -> dict[str, Any]:
        try:
            proc = subprocess.run(self._cli_args(str(path)), capture_output=True, text=True, check=False)
        except FileNotFoundError as e:
            raise ProviderConfigError(
                f"indra-ocr CLI not found at {_INDRA_BIN}; `pip install indra-ocr` or set INDRA_OCR_BIN"
            ) from e
        if proc.returncode != 0:
            raise ProviderPermanentError(f"indra-ocr failed (exit {proc.returncode}): {proc.stderr.strip()}")
        try:
            lines = [_json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        except _json.JSONDecodeError as e:
            raise ProviderPermanentError(f"Failed to parse indra-ocr JSON output: {e}") from e
        frames = _page_frames(path, self._dpi)
        pages = []
        for line in lines:
            i = int(line.get("page", 0))
            w, h, k = frames[i] if i < len(frames) else (None, None, 1.0)
            pages.append(
                {
                    "page_index": i,
                    "markdown": line.get("markdown") or "",
                    "width": w,
                    "height": h,
                    "scale": k,
                    "source": line.get("source"),
                    "blocks": line.get("blocks") or [],
                }
            )
        return {
            "pages": pages,
            "markdown": "\n\n".join(p["markdown"] for p in pages) + "\n",
            "_config": self.base_config,
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"IndraProvider only supports PARSE, got {request.product_type}")
        path = Path(request.source_file_path)
        if not path.exists():
            raise ProviderPermanentError(f"File not found: {path}")
        started_at = datetime.now()
        raw_output = self._parse(path)
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
        # Pipe tables become HTML so the table metrics can see them, as for LiteParse.
        to_html = LiteParseProvider._convert_md_tables_to_html
        raw_pages = raw_result.raw_output.get("pages", [])
        pages = [PageIR(page_index=p.get("page_index", 0), markdown=to_html(p.get("markdown", ""))) for p in raw_pages]
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=_build_layout_pages(raw_pages),
            markdown=to_html(raw_result.raw_output.get("markdown", "")),
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
