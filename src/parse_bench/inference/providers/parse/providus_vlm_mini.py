"""providus-vl-mini: the ParseBench provider for the providus_parse /predict server.

The provider renders each page, posts it to the server and keeps the markdown and the layout items of each page.
For a PDF source it also posts the page as a one-page PDF, so the server can use the PDF text layer.
Set the server address with the PROVIDUS_PARSE_URL variable or the `server_url` setting.
"""

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _one_page_pdf(doc: Any, index: int) -> bytes | None:
    """Return one page of a PDF as a one-page PDF. If pymupdf cannot copy it, return None (the image path runs)."""
    import pymupdf

    try:
        with pymupdf.open() as one:
            one.insert_pdf(doc, from_page=index, to_page=index)
            return bytes(one.tobytes())
    except Exception:
        return None


def _pages(path: Path, dpi: int) -> list[tuple[bytes, bytes | None]]:
    """Return a PNG (or the image file itself) and a one-page PDF (None for an image source) for each page."""
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        return [(path.read_bytes(), None)]
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return [
            (page.get_pixmap(dpi=dpi).tobytes("png"), _one_page_pdf(doc, page.number) if doc.is_pdf else None)
            for page in doc
        ]


def _page_record(number: int, answer: Any) -> dict[str, Any]:
    """Turn one /predict answer into a page record. An error answer gives a failed page."""
    if not isinstance(answer, dict):
        return {"page_number": number, "status": "error", "error": f"answer is not an object: {type(answer).__name__}"}
    if answer.get("status") != "success":
        return {"page_number": number, "status": "error", "error": str(answer.get("error") or "no error text")}
    items = answer.get("items")
    return {
        "page_number": number,
        "status": "success",
        "markdown": str(answer.get("markdown") or ""),
        "items": items if isinstance(items, list) else [],
        "image_width": answer.get("image_width"),
        "image_height": answer.get("image_height"),
        "text_layer": answer.get("text_layer"),
    }


def _centre_y(item: dict[str, Any]) -> float | None:
    """The 0-1 vertical centre of an item box, or None if the box is not usable."""
    box = item.get("bbox")
    try:
        return (float(box[1]) + float(box[3])) / 2 if isinstance(box, (list, tuple)) and len(box) == 4 else None
    except (TypeError, ValueError):
        return None


def _layout_page(page: dict[str, Any], md: str) -> ParseLayoutPageIR:
    """The layout page of one page record, with the page header, page footer and printed page number from its items.

    A field takes only the items at its own page edge (box centre above or below half the page height):
    the header takes Page-header items and the page number at the top, the footer takes Page-footer items
    (the page number too) at the bottom. So a field never holds text that ParseBench folds in at the other edge.
    """
    header, footer, number = [], [], ""
    for item in (i for i in page.get("items") or [] if isinstance(i, dict)):
        text, kind, y = str(item.get("text") or "").strip(), item.get("type"), _centre_y(item)
        if not text or y is None:
            continue
        if kind == "page_number":
            number = number or text
        if y < 0.5 and (item.get("category") == "Page-header" or kind == "page_number"):
            header.append(text)
        elif y >= 0.5 and item.get("category") == "Page-footer":
            footer.append(text)
    return ParseLayoutPageIR(
        page_number=page["page_number"],
        width=page.get("image_width"),
        height=page.get("image_height"),
        md=md,
        page_header_markdown="\n\n".join(header),
        page_footer_markdown="\n\n".join(footer),
        printed_page_number=number,
    )


@register_provider("providus_vlm_mini")
class ProvidusVlmMiniProvider(Provider):
    """Parse provider for the providus_parse server (pipeline `providus_vlm_mini`)."""

    _transport: httpx.BaseTransport | None = None  # tests set a fake server here

    def _setting(self, request: InferenceRequest, key: str) -> Any:
        return {**self._base_config, **(request.config_override or {})}.get(key)

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError("providus_vlm_mini supports only the parse product")
        url = (self._setting(request, "server_url") or os.environ.get("PROVIDUS_PARSE_URL") or "").rstrip("/")
        if not url:
            raise ProviderConfigError("Set PROVIDUS_PARSE_URL (or server_url) to the providus_parse server address")
        dpi = int(self._setting(request, "dpi") or 144)
        timeout = float(self._setting(request, "timeout") or 900)

        started = datetime.now()
        path = Path(request.source_file_path)
        try:
            images = _pages(path, dpi)
        except Exception as exc:
            raise ProviderPermanentError(f"cannot render {path.name}: {exc}") from exc

        pages: list[dict[str, Any]] = []
        with httpx.Client(timeout=timeout, transport=self._transport) as client:
            for number, (png, pdf) in enumerate(images, start=1):
                body = {"image_base64": base64.b64encode(png).decode()}
                if pdf:
                    body["pdf_base64"] = base64.b64encode(pdf).decode()
                try:
                    response = client.post(f"{url}/predict", json=body)
                    response.raise_for_status()
                    answer = response.json()
                except (httpx.HTTPError, ValueError) as exc:  # a dead server or a bad body: the runner may retry
                    raise ProviderTransientError(f"/predict failed on page {number} of {path.name}: {exc}") from exc
                pages.append(_page_record(number, answer))

        completed = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "server_url": url,
                "dpi": dpi,
                "pages": pages,
                "failed_pages": sum(1 for p in pages if p["status"] != "success"),
            },
            started_at=started,
            completed_at=completed,
            latency_in_ms=int((completed - started).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        pages = raw_result.raw_output.get("pages", [])
        page_markdown = [p.get("markdown", "").strip() for p in pages]
        output = ParseOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[PageIR(page_index=i, markdown=md) for i, md in enumerate(page_markdown)],
            layout_pages=[_layout_page(p, md) for p, md in zip(pages, page_markdown, strict=True)],
            markdown="\n\n".join(md for md in page_markdown if md),
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
