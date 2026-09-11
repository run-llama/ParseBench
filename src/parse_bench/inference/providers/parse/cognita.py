"""Provider for Cognita PARSE.

Cognita is a self-hosted, zero-dependency document-understanding engine that
lowers every format into one Intermediate Representation (IR): pages -> blocks
with type, bbox (top-left origin, page points), reading order, confidence and
styled spans, plus a rendered Markdown surface.

This provider talks to a running Cognita server over HTTP:

  * ``POST /v1/parse`` returns the full IR (``document``) and the whole-document
    ``markdown``.
  * ``POST /v1/export?format=markdown`` re-renders an IR document to Markdown.
    We feed it one page at a time to obtain faithful per-page Markdown from
    Cognita's own exporter, rather than re-implementing it here.

The pipeline ``config`` is passed to the provider as ``base_config``:

  * ``server_url`` (str): base URL of the Cognita server (default
    ``https://api.cognita.rahulrawat.in``). Also read from ``COGNITA_SERVER_URL``.
  * ``api_key`` (str, optional): sent as ``X-API-Key`` when auth is enabled.
    Also read from ``COGNITA_API_KEY``.
  * ``timeout`` (int, optional): per-request timeout in seconds (default 300).

The deterministic ``code`` pipeline is used; the optional VLM mode is never
requested, so the run stays pure-code with no cloud calls.
"""

from __future__ import annotations

import html as _html
import mimetypes
import os
import re
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

# Cognita IR block type -> Canonical17 layout label. Headings resolve by level
# (level 1 -> Title, deeper -> Section-header) in _canonical_label().
_BLOCK_LABEL_MAP: dict[str, str] = {
    "paragraph": "Text",
    "table": "Table",
    "image": "Picture",
    "list": "List-item",
    "list_item": "List-item",
    "quote": "Text",
    "code": "Code",
    "header": "Page-header",
    "footer": "Page-footer",
    "caption": "Caption",
    "page_number": "Page-footer",
}

# Item type for LayoutItemIR (coarse), mirroring azure_document_intelligence.
_ITEM_TYPE_BY_LABEL: dict[str, str] = {
    "Table": "table",
    "Picture": "image",
}


_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


def _split_pipe_cells(line: str) -> list[str]:
    """Split a GFM table row into cell texts, honoring escaped pipes."""
    # Protect escaped pipes, split on the rest, then restore.
    parts = re.split(r"(?<!\\)\|", line.strip())
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return [p.strip().replace("\\|", "|") for p in parts]


def _cell_html(text: str) -> str:
    return _html.escape(text, quote=False)


def _gfm_tables_to_html(md: str) -> str:
    """Convert GFM pipe tables in *md* to HTML ``<table>`` blocks.

    ParseBench's table metrics extract predicted tables with
    ``BeautifulSoup(page.markdown).find_all("table")``, so tables must be HTML.
    Prose is left untouched; only contiguous pipe-table blocks are rewritten.
    """
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    fence: str | None = None  # active fenced-code marker ("```" or "~~~"), if any
    while i < n:
        stripped = lines[i].lstrip()
        # Track fenced code blocks so a pipe/separator sequence inside code is
        # never mistaken for a table.
        if fence is not None:
            out.append(lines[i])
            if stripped.startswith(fence):
                fence = None
            i += 1
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            out.append(lines[i])
            i += 1
            continue

        # A table needs a header row, a separator row, then >=0 body rows.
        if (
            i + 1 < n
            and _PIPE_ROW_RE.match(lines[i])
            and _SEPARATOR_RE.match(lines[i + 1])
            and "-" in lines[i + 1]
        ):
            header = _split_pipe_cells(lines[i])
            j = i + 2
            body: list[list[str]] = []
            while j < n and _PIPE_ROW_RE.match(lines[j]) and not _SEPARATOR_RE.match(lines[j]):
                body.append(_split_pipe_cells(lines[j]))
                j += 1

            parts = ["<table>"]
            if header:
                parts.append("<thead><tr>" + "".join(f"<th>{_cell_html(c)}</th>" for c in header) + "</tr></thead>")
            parts.append("<tbody>")
            for row in body:
                parts.append("<tr>" + "".join(f"<td>{_cell_html(c)}</td>" for c in row) + "</tr>")
            parts.append("</tbody></table>")
            out.append("".join(parts))
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _canonical_label(block: dict[str, Any]) -> str:
    btype = str(block.get("type") or "paragraph").lower()
    if btype == "heading":
        level = block.get("level")
        return "Title" if isinstance(level, int) and level <= 1 else "Section-header"
    return _BLOCK_LABEL_MAP.get(btype, "Text")


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalized_segment(block: dict[str, Any], width: float, height: float) -> LayoutSegmentIR | None:
    """Convert a Cognita bbox (points, top-left origin) to a normalized [0,1] segment."""
    bbox = block.get("bbox")
    if not isinstance(bbox, dict) or width <= 0 or height <= 0:
        return None
    try:
        x = float(bbox["x"]) / width
        y = float(bbox["y"]) / height
        w = float(bbox["w"]) / width
        h = float(bbox["h"]) / height
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return LayoutSegmentIR(
        x=_clamp01(x),
        y=_clamp01(y),
        w=_clamp01(w),
        h=_clamp01(h),
        confidence=block.get("confidence"),
        label=_canonical_label(block),
    )


def _iter_layout_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten list containers into their list_item children; keep everything else.

    Cognita nests list items under a ``list`` block via ``children``; the items
    carry their own bboxes, so we surface them directly for attribution.
    """
    out: list[dict[str, Any]] = []
    for b in blocks:
        if str(b.get("type") or "").lower() == "list" and isinstance(b.get("children"), list):
            children = [c for c in b["children"] if isinstance(c, dict)]
            if children:
                out.extend(children)
                continue
        out.append(b)
    return out


@register_provider("cognita")
class CognitaProvider(Provider):
    """Provider for the Cognita document-understanding engine (self-hosted)."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        server_url = (
            self.base_config.get("server_url") or os.getenv("COGNITA_SERVER_URL") or "https://api.cognita.rahulrawat.in"
        )
        self._server_url = str(server_url).rstrip("/")
        self._api_key = self.base_config.get("api_key") or os.getenv("COGNITA_API_KEY") or ""
        self._timeout = float(self.base_config.get("timeout", 300))

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _parse_document(self, client: httpx.Client, file_path: Path) -> dict[str, Any]:
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        try:
            resp = client.post(
                f"{self._server_url}/v1/parse",
                params={"include": "document,markdown", "image_data": "omit"},
                content=data,
                headers=self._headers(content_type),
            )
        except httpx.HTTPError as e:
            raise ProviderTransientError(f"Cognita server unreachable: {e}") from e
        if resp.status_code == 401:
            raise ProviderConfigError("Cognita rejected the API key (401). Set 'api_key' in the pipeline config.")
        if 400 <= resp.status_code < 500:
            raise ProviderPermanentError(f"Cognita parse failed ({resp.status_code}): {resp.text[:300]}")
        if resp.status_code >= 500:
            raise ProviderTransientError(f"Cognita server error ({resp.status_code}): {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderPermanentError(f"Cognita returned non-JSON response: {e}") from e

    def _export_page_markdown(self, client: httpx.Client, document: dict[str, Any], page: dict[str, Any]) -> str:
        """Render one page's Markdown via Cognita's own exporter."""
        single = dict(document)
        single["pages"] = [page]
        try:
            resp = client.post(
                f"{self._server_url}/v1/export",
                params={"format": "markdown"},
                json=single,
                headers=self._headers(),
            )
            if resp.status_code == 200:
                return resp.text
        except httpx.HTTPError:
            pass
        # Fall back to raw block text if export is unavailable for this page.
        return "\n\n".join(str(b.get("text", "")) for b in page.get("blocks", []) if b.get("text"))

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"CognitaProvider only supports PARSE product type, got {request.product_type}"
            )

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        started_at = datetime.now()
        with httpx.Client(timeout=self._timeout) as client:
            parse_resp = self._parse_document(client, file_path)
            document = parse_resp.get("document") or {}
            pages = document.get("pages") or []
            page_markdowns = [self._export_page_markdown(client, document, page) for page in pages]

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)

        raw_output = {"parse": parse_resp, "page_markdowns": page_markdowns}
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"CognitaProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        parse_resp = raw_result.raw_output.get("parse", {})
        document = parse_resp.get("document") or {}
        pages_json = document.get("pages") or []
        page_markdowns = raw_result.raw_output.get("page_markdowns") or []
        full_markdown = _gfm_tables_to_html(parse_resp.get("markdown") or "")

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []

        for idx, page in enumerate(pages_json):
            raw_page_md = page_markdowns[idx] if idx < len(page_markdowns) else ""
            page_md = _gfm_tables_to_html(raw_page_md)
            page_number = int(page.get("number") or (idx + 1))
            width = float(page.get("width") or 0)
            height = float(page.get("height") or 0)

            pages.append(PageIR(page_index=idx, markdown=page_md))

            items: list[LayoutItemIR] = []
            for block in _iter_layout_blocks(page.get("blocks") or []):
                seg = _normalized_segment(block, width, height)
                if seg is None:
                    continue
                label = seg.label or "Text"
                items.append(
                    LayoutItemIR(
                        type=_ITEM_TYPE_BY_LABEL.get(label, "text"),
                        value=str(block.get("text", "")),
                        md=str(block.get("text", "")),
                        bbox=seg,
                        layout_segments=[seg],
                    )
                )

            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=page_number,
                    width=width or None,
                    height=height or None,
                    md=page_md,
                    items=items,
                )
            )

        # Fall back to the whole-document markdown if per-page export produced nothing.
        if not full_markdown and page_markdowns:
            full_markdown = "\n\n".join(page_markdowns)

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
