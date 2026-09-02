"""Provider for Cohere document parsing (``/v2/parse``, ``parse-v5.0``).

This provider calls Cohere's hosted parse endpoint
(``POST https://api.cohere.com/v2/parse``) with the parse-v5.0 model.
Each page is rendered to an image and sent as a base64 ``data:`` URI with
``output_format: "raw_generation"``.

Output
------
The model returns raw VLM output with markdown interleaved with
``[visual_element]`` blocks. ``normalize()`` extracts clean markdown (keeping
table HTML inline, stripping non-table VEs) and converts pipe tables to HTML
for GriTS/TEDS scoring. When visual-element blocks carry bounding boxes,
``layout_pages`` are built for layout evaluation.
"""

import base64
import io
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

_PRODUCTION_URL = "https://api.cohere.com/v2/parse"

_SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".jfif"}


# ---------------------------------------------------------------------------
# Visual-element parsing
# ---------------------------------------------------------------------------

_VE_RE = re.compile(
    r"(?:<_visual_element_start_>|\[_visual_element_start_\]|\[visual_element\])"
    r"(.*?)"
    r"(?:<_visual_element_end_>|\[_visual_element_end_\]|\[/visual_element\])",
    re.DOTALL,
)
_HTML_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_BBOX_RE = re.compile(r"bbox\s*:?\s*\[?\s*([-\d.,\s]+?)\s*\]?\s*$", re.IGNORECASE | re.MULTILINE)
_KV_RE = re.compile(r"^\s*([a-zA-Z_][\w\- ]*?)\s*:\s*(.*?)\s*$", re.MULTILINE)

# VE type -> (item_kind, Canonical17 label)
_TYPE_TO_LAYOUT: dict[str, tuple[str, str]] = {
    "table": ("table", "Table"),
    "chart": ("image", "Picture"),
    "image": ("image", "Picture"),
    "picture": ("image", "Picture"),
    "figure": ("image", "Picture"),
    "photo": ("image", "Picture"),
    "diagram": ("image", "Picture"),
    "logo": ("image", "Picture"),
    "formula": ("text", "Formula"),
    "caption": ("text", "Caption"),
    "footnote": ("text", "Footnote"),
    "title": ("text", "Title"),
    "header": ("text", "Page-header"),
    "footer": ("text", "Page-footer"),
    "text": ("text", "Text"),
}


def _parse_ve_block(block: str) -> dict[str, Any]:
    """Extract type, bbox, and html from a VE block body."""
    fields: dict[str, Any] = {}
    for m in _KV_RE.finditer(block):
        key = m.group(1).strip().lower().replace(" ", "_")
        fields[key] = m.group(2).strip()

    bbox_match = _BBOX_RE.search(block)
    if bbox_match:
        try:
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", bbox_match.group(1))]
            if len(nums) >= 4:
                fields["bbox"] = nums[:4]
        except ValueError:
            pass

    html_match = _HTML_TABLE_RE.search(block)
    if html_match:
        fields["html"] = html_match.group(0)

    return fields


def _extract_markdown_and_layout(
    raw_text: str, image_size: tuple[int, int],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract clean markdown + layout items from VE-annotated model output."""
    img_w, img_h = image_size
    parts: list[str] = []
    layout_items: list[dict[str, Any]] = []
    last_end = 0

    for match in _VE_RE.finditer(raw_text):
        start, end = match.span()
        if start > last_end:
            between = raw_text[last_end:start].strip()
            if between:
                parts.append(between)

        body = match.group(1).strip()
        ve = _parse_ve_block(body)

        html_match = _HTML_TABLE_RE.search(body)
        if html_match:
            parts.append(html_match.group(0))

        bbox = ve.get("bbox")
        if bbox and len(bbox) == 4 and img_w > 0 and img_h > 0:
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                x2, y2 = x1 + x2, y1 + y2
            ve_type = str(ve.get("type", "")).strip().lower()
            layout_items.append({
                "bbox_px": [x1, y1, x2, y2],
                "type": ve_type,
                "md": ve.get("html", ""),
            })

        last_end = end

    if last_end < len(raw_text):
        trailing = raw_text[last_end:].strip()
        if trailing:
            parts.append(trailing)

    markdown = "\n\n".join(p for p in parts if p)
    return markdown, layout_items


# ---------------------------------------------------------------------------
# Layout page builder
# ---------------------------------------------------------------------------


def _build_layout_pages(
    pages_data: list[dict[str, Any]],
) -> list[ParseLayoutPageIR]:
    """Build layout pages from all pages' extracted layout items."""
    layout_pages: list[ParseLayoutPageIR] = []

    for page_data in pages_data:
        layout_items = page_data.get("_layout_items") or []
        img_size = tuple(page_data.get("image_size") or (0, 0))
        page_index = int(page_data.get("page_index", 0))

        if not layout_items:
            continue
        img_w, img_h = img_size
        if img_w <= 0 or img_h <= 0:
            continue

        items: list[LayoutItemIR] = []
        for li in layout_items:
            x1, y1, x2, y2 = li["bbox_px"]
            nx = max(0.0, min(1.0, x1 / img_w))
            ny = max(0.0, min(1.0, y1 / img_h))
            nw = max(0.0, min(1.0, (x2 - x1) / img_w))
            nh = max(0.0, min(1.0, (y2 - y1) / img_h))
            item_kind, label = _TYPE_TO_LAYOUT.get(li["type"], ("text", "Text"))
            seg = LayoutSegmentIR(x=nx, y=ny, w=nw, h=nh, confidence=1.0, label=label)
            items.append(LayoutItemIR(type=item_kind, value=li["md"], bbox=seg, layout_segments=[seg]))

        if items:
            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=page_index + 1,
                    width=float(img_w),
                    height=float(img_h),
                    items=items,
                )
            )

    return layout_pages


# ---------------------------------------------------------------------------
# Pipe-table -> HTML conversion
# ---------------------------------------------------------------------------


def _convert_md_tables_to_html(content: str) -> str:
    """Convert markdown pipe tables to HTML <table> so GriTS/TEDS can score them."""
    import markdown2

    def _flush(table_lines: list[str], out: list[str]) -> None:
        if len(table_lines) >= 2:
            html = markdown2.markdown("\n".join(table_lines), extras=["tables"]).strip()
            if "<table>" in html.lower():
                out.append(html)
                return
        out.extend(table_lines)

    result_parts: list[str] = []
    table_lines: list[str] = []
    in_table = False

    for line in content.split("\n"):
        if "|" in line and line.strip().startswith("|"):
            table_lines.append(line)
            in_table = True
            continue
        if in_table:
            _flush(table_lines, result_parts)
            table_lines = []
            in_table = False
        result_parts.append(line)

    if in_table:
        _flush(table_lines, result_parts)

    return "\n".join(result_parts)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


@register_provider("cohere_parse")
class CohereParseProvider(Provider):
    """Provider for Cohere's document parsing via ``/v2/parse``.

    Config keys
    -----------
    api_key : str
        Cohere API key. Falls back to ``COHERE_API_KEY`` env var.
    base_url : str
        Parse endpoint URL (default: production).
    model : str
        Model id (default ``"parse-v5.0"``).
    max_pages : int
        Cap on pages sent per document (default 50).
    timeout : int
        HTTP request timeout in seconds (default 660).
    """

    COST_PER_PAGE_USD = 0.001

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        api_key = (
            self.base_config.get("api_key")
            or os.getenv("COHERE_API_KEY")
        )
        if not api_key:
            raise ProviderConfigError(
                "Cohere API key required. You can get COHERE_API_KEY from https://dashboard.cohere.com/api-keys"
            )
        self._api_key: str = str(api_key)
        self._base_url: str = self.base_config.get("base_url", _PRODUCTION_URL)
        self._model: str = self.base_config.get("model", "parse-v5.0")
        self._max_pages: int = self.base_config.get("max_pages", 50)
        self._timeout: int = self.base_config.get("timeout", 660)
        self._rate_limit_retries: int = self.base_config.get("rate_limit_retries", 6)
        self._rate_limit_base_wait: float = self.base_config.get("rate_limit_base_wait", 2.0)
        self._rate_limit_max_wait: float = self.base_config.get("rate_limit_max_wait", 60.0)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_pages(self, file_path: Path) -> list[Image.Image]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            images = self._render_pdf(file_path)
        else:
            try:
                images = [Image.open(file_path).convert("RGB")]
            except Exception as e:
                raise ProviderPermanentError(f"Unreadable image ({type(e).__name__}): {file_path}") from e
        return images[: self._max_pages]

    def _render_pdf(self, file_path: Path) -> list[Image.Image]:
        try:
            import fitz
        except ImportError:
            fitz = None

        if fitz is not None:
            try:
                images: list[Image.Image] = []
                zoom = 150 / 72.0
                with fitz.open(str(file_path)) as doc:
                    for page in doc:
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                return images
            except Exception as e:
                raise ProviderPermanentError(f"Unreadable PDF ({type(e).__name__}): {file_path}") from e

        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProviderConfigError("Install pymupdf or pdf2image to rasterize PDFs.") from e
        try:
            return convert_from_path(str(file_path), dpi=150)
        except Exception as e:
            raise ProviderPermanentError(f"Failed to convert PDF to images: {e}") from e

    @staticmethod
    def _to_data_uri(image: Image.Image, fmt: str = "png") -> str:
        with io.BytesIO() as buf:
            image.save(buf, format=fmt.upper(), quality=100)
            data = buf.getvalue()
        return f"data:image/{fmt.lower()};base64," + base64.b64encode(data).decode("ascii")

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _parse_page(self, client: httpx.Client, image: Image.Image) -> tuple[str, dict[str, int]]:
        """Call /v2/parse with output_format=raw_generation for a single page."""
        payload = {
            "model": self._model,
            "document": {
                "type": "image_url",
                "image_url": self._to_data_uri(image),
            },
            "output_format": "raw_generation",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Client-Name": "parse-bench",
        }

        resp = None
        for attempt in range(self._rate_limit_retries + 1):
            try:
                resp = client.post(self._base_url, json=payload, headers=headers)
            except httpx.TimeoutException as e:
                raise ProviderTransientError(f"Cohere parse request timed out: {e}") from e
            except httpx.ConnectError as e:
                raise ProviderTransientError(f"Cohere parse connection error: {e}") from e

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self._rate_limit_retries:
                    retry_after = resp.headers.get("retry-after")
                    try:
                        wait = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        wait = 0.0
                    if wait <= 0:
                        wait = self._rate_limit_base_wait * (2.0 ** attempt)
                    time.sleep(min(wait, self._rate_limit_max_wait))
                    continue
                if resp.status_code == 429:
                    raise ProviderRateLimitError(
                        f"Cohere parse rate limit (429) after {self._rate_limit_retries} retries: "
                        f"{resp.text[:300]}"
                    )
                raise ProviderTransientError(
                    f"Cohere parse server error ({resp.status_code}) after "
                    f"{self._rate_limit_retries} retries: {resp.text[:300]}"
                )
            break

        assert resp is not None
        if resp.status_code in (401, 403):
            raise ProviderConfigError(f"Cohere parse unauthorized ({resp.status_code}): {resp.text[:500]}")
        if resp.status_code >= 400:
            raise ProviderPermanentError(f"Cohere parse client error ({resp.status_code}): {resp.text[:500]}")

        try:
            body = resp.json()
            pages = body.get("pages", [])
            text = pages[0].get("raw_generation", "") if pages else ""
        except (ValueError, KeyError, TypeError, IndexError) as e:
            raise ProviderPermanentError(f"Unexpected Cohere parse response shape: {resp.text[:300]}") from e

        meta = body.get("meta", {})
        billed = meta.get("billed_units", {})
        return text, {"pages_billed": int(billed.get("pages", 0))}

    # ------------------------------------------------------------------
    # Provider interface
    # ------------------------------------------------------------------

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"CohereParseProvider only supports PARSE, got {request.product_type}")

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"File not found: {source_path}")
        if source_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ProviderPermanentError(f"Unsupported file type: {source_path.suffix}")

        started_at = datetime.now()
        images = self._render_pages(source_path)

        pages: list[dict[str, Any]] = []
        total_pages_billed = 0
        with httpx.Client(timeout=httpx.Timeout(self._timeout, connect=30.0)) as client:
            for page_index, image in enumerate(images):
                text, usage = self._parse_page(client, image)
                total_pages_billed += usage["pages_billed"]
                pages.append({
                    "page_index": page_index,
                    "markdown": text,
                    "image_size": [image.width, image.height],
                })

        completed_at = datetime.now()
        pages_processed = len(pages)
        raw_output: dict[str, Any] = {
            "pages": pages,
            "num_pages": pages_processed,
            "model": self._model,
            "pages_billed": total_pages_billed,
            "cost_usd": pages_processed * self.COST_PER_PAGE_USD,
            "cost_per_page_usd": self.COST_PER_PAGE_USD,
        }

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
            raise ProviderPermanentError(f"CohereParseProvider only supports PARSE, got {raw_result.product_type}")

        pages: list[PageIR] = []
        page_markdowns: list[str] = []
        pages_with_layout: list[dict[str, Any]] = []

        for page_data in raw_result.raw_output.get("pages") or []:
            page_index = int(page_data.get("page_index", 0))
            raw_text = str(page_data.get("markdown", "") or "")
            img_size = tuple(page_data.get("image_size") or (0, 0))

            markdown, layout_items = _extract_markdown_and_layout(raw_text, img_size)

            pages.append(PageIR(page_index=page_index, markdown=markdown))
            page_markdowns.append(markdown)
            pages_with_layout.append({
                "page_index": page_index,
                "image_size": img_size,
                "_layout_items": layout_items,
            })

        pages.sort(key=lambda p: p.page_index)
        layout_pages = _build_layout_pages(pages_with_layout)
        full_markdown = _convert_md_tables_to_html("\n\n".join(page_markdowns).strip())

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=full_markdown,
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
