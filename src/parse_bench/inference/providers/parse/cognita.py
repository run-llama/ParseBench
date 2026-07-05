"""Provider for Cognita — a pure-code document parsing engine.

Cognita is a deterministic, zero-ML document parsing server. A hosted
instance runs at https://cognita.rahulrawat.in (the root URL also serves a
browser test console). This provider sends documents to its REST API
(POST /v1/parse) and normalizes the returned IR + markdown.
"""

import html as html_mod
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

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

# Cognita IR block type -> Canonical17 label string.
# "list" is a container block and is flattened into its list_item children.
COGNITA_LABEL_MAP: dict[str, str] = {
    "heading": "Section-header",
    "paragraph": "Text",
    "table": "Table",
    "image": "Picture",
    "list_item": "List-item",
    "quote": "Text",
    "code": "Code",
    "header": "Page-header",
    "footer": "Page-footer",
    "caption": "Caption",
    "page_number": "Page-footer",
}


@register_provider("cognita")
class CognitaProvider(Provider):
    """
    Provider for Cognita PARSE.

    Sends documents to a locally running Cognita server (POST /v1/parse) and
    normalizes the returned IR + markdown. Cognita is fully self-hosted; no
    external API is involved.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `base_url`: Cognita server URL (defaults to COGNITA_BASE_URL env
              var, then the hosted instance at https://cognita.rahulrawat.in)
            - `api_key`: API key sent as X-API-Key (defaults to COGNITA_API_KEY
              env var; required by the hosted instance)
            - `timeout`: Request timeout in seconds (default: 120)
        """
        super().__init__(provider_name, base_config)

        self._base_url = (
            self.base_config.get("base_url") or os.getenv("COGNITA_BASE_URL") or "https://cognita.rahulrawat.in"
        ).rstrip("/")
        self._api_key = self.base_config.get("api_key") or os.getenv("COGNITA_API_KEY") or ""
        self._timeout = self.base_config.get("timeout", 120)

    def _call_parse(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """
        Call POST /v1/parse on the Cognita server.

        :param file_bytes: Raw document bytes
        :param filename: Original filename
        :return: Raw JSON response ({"document": ..., "markdown": ...})
        :raises ProviderError: For any API errors
        """
        headers = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        # image_data=omit strips embedded image payloads from the IR so the
        # stored raw_output stays small; bbox/dimension metadata is preserved.
        params = {"include": "document,markdown", "image_data": "omit"}

        try:
            response = requests.post(
                f"{self._base_url}/v1/parse",
                headers=headers,
                params=params,
                files={"file": (filename, file_bytes, "application/octet-stream")},
                timeout=self._timeout,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ProviderPermanentError(f"Cognita returned unsupported response type: {type(result).__name__}")
            return result

        except requests.exceptions.Timeout as e:
            raise ProviderTransientError(f"Request timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ProviderTransientError(
                f"Cannot reach Cognita at {self._base_url} — is the server up? (override with COGNITA_BASE_URL): {e}"
            ) from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code in (401, 403):
                raise ProviderConfigError(f"Cognita rejected the API key. Set COGNITA_API_KEY: {e}") from e
            elif status_code == 429:
                raise ProviderRateLimitError(f"Rate limit exceeded: {e}") from e
            elif status_code and 500 <= status_code < 600:
                raise ProviderTransientError(f"Server error ({status_code}): {e}") from e
            elif status_code and 400 <= status_code < 500:
                raise ProviderPermanentError(f"Client error ({status_code}): {e}") from e
            else:
                raise ProviderPermanentError(f"HTTP error: {e}") from e
        except (ProviderPermanentError, ProviderTransientError, ProviderRateLimitError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error calling Cognita: {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"CognitaProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        try:
            raw_output = self._call_parse(source_path.read_bytes(), source_path.name)

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

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

        except (ProviderPermanentError, ProviderTransientError, ProviderRateLimitError, ProviderConfigError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"CognitaProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        markdown = raw_result.raw_output.get("markdown", "") or ""
        document = raw_result.raw_output.get("document") or {}
        layout_pages = _build_layout_pages(document)

        # The table evaluator only recognizes HTML <table> markup in the
        # prediction, and GFM pipe tables cannot express the merged cells that
        # GriTS measures. Cognita's IR has rowspan/colspan, so replace the
        # exporter's pipe tables with HTML rendered from the IR.
        markdown = _replace_pipe_tables_with_html(markdown, document)

        # Cognita's markdown exporter renders the whole document without page
        # breaks, so pages stays empty and markdown carries the full text
        # (same convention as the Reducto provider).
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            layout_pages=layout_pages,
            markdown=markdown,
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


def _build_layout_pages(document: dict[str, Any]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from the Cognita IR for layout cross-evaluation.

    Cognita bboxes are in page units (typically pt, top-left origin) with page
    width/height in the same units. Segments are normalized to [0,1] here —
    the convention layout adapters expect — with the page dims kept on the
    ParseLayoutPageIR so adapters can scale back to pixel space. Pages without
    known dimensions can't be normalized and contribute no layout items.
    """
    layout_pages: list[ParseLayoutPageIR] = []
    for page in document.get("pages") or []:
        width = float(page.get("width") or 0.0)
        height = float(page.get("height") or 0.0)
        items: list[LayoutItemIR] = []
        if width > 0 and height > 0:
            for block in page.get("blocks") or []:
                _collect_items(block, items, width, height)
            items = _prune_layout_items(items)

        # Structured page sections: the header/footer/page-number rules read
        # these fields instead of scanning markdown. Matching is markup-
        # strict and ground truth styles sections inconsistently (some carry
        # **bold**, others are plain), so each section carries a styled AND
        # a plain rendering — the rule regex matches on substrings, so extra
        # variants never hurt.
        headers: list[str] = []
        footers: list[str] = []
        plain_headers: list[str] = []
        plain_footers: list[str] = []
        raw_headers: list[str] = []
        raw_footers: list[str] = []
        page_numbers: list[str] = []
        for block in page.get("blocks") or []:
            text = block.get("text", "") or ""
            if not text:
                continue
            block_type = block.get("type", "")
            if block_type in ("header", "footer", "page_number"):
                styled = _tag_trailing_page_number(_block_styled_markdown(block))
                plain = _tag_trailing_page_number(text)
                if block_type == "header":
                    headers.append(styled)
                    plain_headers.append(plain)
                    raw_headers.append(text)
                elif block_type == "footer":
                    footers.append(styled)
                    plain_footers.append(plain)
                    raw_footers.append(text)
                else:
                    page_numbers.append(text)
                    footers.append(styled)
                    plain_footers.append(plain)
                    raw_footers.append(text)

        header_md = _join_section_variants(headers, plain_headers, raw_headers)
        footer_md = _join_section_variants(footers, plain_footers, raw_footers)
        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page.get("number") or len(layout_pages) + 1,
                width=width or None,
                height=height or None,
                page_header_markdown=header_md,
                page_footer_markdown=footer_md,
                printed_page_number=page_numbers[0] if page_numbers else "",
                items=items,
            )
        )
    return layout_pages


def _join_section_variants(*variant_lists: list[str]) -> str:
    """Join section lines, appending each rendering variant that differs.
    Section matching is substring-based, so extra variants never hurt and
    cover ground truth's inconsistent styling/tagging conventions."""
    seen: list[str] = []
    for lines in variant_lists:
        joined = "\n".join(lines)
        if joined and joined not in seen:
            seen.append(joined)
    return "\n".join(seen)


# A standalone page-number token: 1-4 digits not glued to date/time/decimal
# punctuation or word characters.
_PAGE_NUMBER_TOKEN_RE = re.compile(r"(?<![\w./:\-])(\d{1,4})(?![\w./:%])")


def _tag_trailing_page_number(text: str) -> str:
    """Wrap standalone numbers in <page_number> tags, matching the
    benchmark's convention for page numbers in running headers/footers
    (trailing '... 1', leading '18 ANNUAL REPORT', or '- 2 -' forms)."""

    def _wrap(m: re.Match) -> str:
        return f"<page_number>{m.group(1)}</page_number>"

    return _PAGE_NUMBER_TOKEN_RE.sub(_wrap, text)


def _block_styled_markdown(block: dict[str, Any]) -> str:
    """Render a block's spans with inline styling for page sections
    (bold/italic markers, sup/sub/mark tags, ~~ for strikethrough).
    Underline is deliberately not rendered: ground-truth section text never
    carries <u> tags and section matching is markup-strict."""
    spans = block.get("spans") or []
    if not spans:
        return block.get("text", "") or ""
    parts: list[str] = []
    for span in spans:
        text = span.get("text", "") or ""
        if not text.strip():
            parts.append(text)
            continue
        lead = text[: len(text) - len(text.lstrip(" "))]
        trail = text[len(text.rstrip(" ")) :]
        core = text.strip(" ")
        style = span.get("style") or {}
        if style.get("superscript"):
            core = f"<sup>{core}</sup>"
        elif style.get("subscript"):
            core = f"<sub>{core}</sub>"
        if style.get("bold") and style.get("italic"):
            core = f"***{core}***"
        elif style.get("bold"):
            core = f"**{core}**"
        elif style.get("italic"):
            core = f"*{core}*"
        if style.get("strikethrough"):
            core = f"~~{core}~~"
        if style.get("highlight"):
            core = f"<mark>{core}</mark>"
        parts.append(lead + core + trail)
    return "".join(parts)


def _bbox_area(b: LayoutSegmentIR) -> float:
    return max(0.0, b.w) * max(0.0, b.h)


def _containment(inner: LayoutSegmentIR, outer: LayoutSegmentIR) -> float:
    """Fraction of inner's area covered by outer."""
    ix = max(inner.x, outer.x)
    iy = max(inner.y, outer.y)
    ax = min(inner.x + inner.w, outer.x + outer.w)
    ay = min(inner.y + inner.h, outer.y + outer.h)
    if ax <= ix or ay <= iy:
        return 0.0
    area = _bbox_area(inner)
    if area <= 0:
        return 0.0
    return (ax - ix) * (ay - iy) / area


def _prune_layout_items(items: list[LayoutItemIR]) -> list[LayoutItemIR]:
    """Drop layout noise: sub-images nested inside a larger picture and tiny
    text labels living inside figures (axis labels, callouts)."""
    pictures = [it for it in items if it.type == "image" and it.bbox is not None]
    out: list[LayoutItemIR] = []
    for it in items:
        b = it.bbox
        if b is None:
            out.append(it)
            continue
        if it.type == "image":
            nested = any(
                o is not it
                and o.bbox is not None
                and _bbox_area(o.bbox) > 1.2 * _bbox_area(b)
                and _containment(b, o.bbox) > 0.85
                for o in pictures
            )
            if nested:
                continue
        elif it.type == "text":
            tiny = _bbox_area(b) < 0.004
            inside_figure = any(o.bbox is not None and _containment(b, o.bbox) > 0.7 for o in pictures)
            if tiny and inside_figure:
                continue
        out.append(it)
    return out


def _collect_items(block: dict[str, Any], items: list[LayoutItemIR], page_w: float, page_h: float) -> None:
    """Recursively convert an IR block (and its children) into layout items."""
    block_type = block.get("type", "")
    label = COGNITA_LABEL_MAP.get(block_type)
    if block_type == "heading" and block.get("level", 0) == 1:
        label = "Title"
    bbox = block.get("bbox")

    if label is not None and bbox:
        conf_raw = block.get("confidence")
        try:
            confidence = float(conf_raw) if conf_raw is not None else 1.0
        except (TypeError, ValueError):
            confidence = 1.0

        seg = LayoutSegmentIR(
            x=float(bbox.get("x", 0.0)) / page_w,
            y=float(bbox.get("y", 0.0)) / page_h,
            w=float(bbox.get("w", 0.0)) / page_w,
            h=float(bbox.get("h", 0.0)) / page_h,
            confidence=confidence,
            label=label,
        )

        if label == "Table":
            item_type = "table"
            value = _table_text(block)
        elif label == "Picture":
            item_type = "image"
            value = (block.get("image") or {}).get("alt_text", "") or ""
        else:
            item_type = "text"
            value = block.get("text", "") or ""

        items.append(
            LayoutItemIR(
                type=item_type,
                value=value,
                bbox=seg,
                layout_segments=[seg],
            )
        )

    # Containers (list) and nested structures (lists under list items) carry
    # their content in children.
    for child in block.get("children") or []:
        _collect_items(child, items, page_w, page_h)


def _table_text(block: dict[str, Any]) -> str:
    """Flatten a table block's cell text (tab-separated cells, one row per line)."""
    table = block.get("table") or {}
    lines: list[str] = []
    for row in table.get("rows") or []:
        cells: list[str] = []
        for cell in row.get("cells") or []:
            parts: list[str] = []
            for cell_block in cell.get("blocks") or []:
                text = cell_block.get("text", "") or ""
                if text:
                    parts.append(text)
            cells.append(" ".join(parts))
        lines.append("\t".join(cells))
    return "\n".join(lines)


# =============================================================================
# Pipe table -> HTML table conversion
# =============================================================================


def _replace_pipe_tables_with_html(markdown: str, document: dict[str, Any]) -> str:
    """Replace GFM pipe tables in the exported markdown with HTML tables.

    Primary path: render each IR table (with rowspan/colspan and header rows)
    and substitute it for the corresponding pipe block — Cognita's markdown
    exporter emits exactly one contiguous pipe block per rendered IR table, in
    document order. If the counts disagree for any reason, fall back to
    converting each pipe block textually (no span information, but still HTML).
    """
    blocks = _find_pipe_blocks(markdown)
    if not blocks:
        return markdown

    ir_tables = _collect_rendered_tables(document)
    use_ir = len(ir_tables) == len(blocks)

    lines = markdown.splitlines()
    out: list[str] = []
    cursor = 0
    for i, (start, end) in enumerate(blocks):
        out.extend(lines[cursor:start])
        if use_ir:
            out.append(_ir_table_to_html(ir_tables[i]))
        else:
            out.append(_pipe_block_to_html(lines[start:end]))
        cursor = end
    out.extend(lines[cursor:])
    return "\n".join(out) + ("\n" if markdown.endswith("\n") else "")


def _find_pipe_blocks(markdown: str) -> list[tuple[int, int]]:
    """Return [start, end) line ranges of contiguous pipe-table blocks."""
    blocks: list[tuple[int, int]] = []
    start = None
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            if start is None:
                start = i
        elif start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, len(lines)))
    # A real table block has at least a header and a separator row.
    return [(s, e) for s, e in blocks if e - s >= 2]


def _collect_rendered_tables(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect IR tables in the order Cognita's markdown exporter renders them.

    Mirrors the exporter's traversal: tables under list blocks are never
    rendered as pipe blocks, and empty tables are skipped.
    """
    tables: list[dict[str, Any]] = []
    for page in document.get("pages") or []:
        for block in page.get("blocks") or []:
            _collect_tables(block, tables)
    return tables


def _collect_tables(block: dict[str, Any], tables: list[dict[str, Any]]) -> None:
    block_type = block.get("type", "")
    if block_type == "table":
        table = block.get("table") or {}
        rows = table.get("rows") or []
        if rows and any(row.get("cells") for row in rows):
            tables.append(table)
    if block_type == "list":
        return  # list children are rendered inline, never as pipe tables
    for child in block.get("children") or []:
        _collect_tables(child, tables)


def _cell_plain_text(cell: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in cell.get("blocks") or []:
        text = block.get("text", "") or ""
        if text:
            parts.append(text)
        nested = _table_text(block) if block.get("table") else ""
        if nested:
            parts.append(nested.replace("\t", " ").replace("\n", " "))
        for child in block.get("children") or []:
            child_text = child.get("text", "") or ""
            if child_text:
                parts.append(child_text)
    return " ".join(parts)


def _ir_table_to_html(table: dict[str, Any]) -> str:
    """Render an IR table dict as an HTML table with spans and header rows."""
    rows = table.get("rows") or []
    header_rows = table.get("header_rows") or 0
    # The markdown exporter always renders the first row as the header row;
    # keep that convention when the IR doesn't say otherwise.
    if header_rows <= 0 and len(rows) >= 2:
        header_rows = 1

    parts: list[str] = ["<table>"]
    for ri, row in enumerate(rows):
        tag = "th" if ri < header_rows else "td"
        cells_html: list[str] = []
        for cell in row.get("cells") or []:
            attrs = ""
            row_span = cell.get("row_span") or 0
            col_span = cell.get("col_span") or 0
            if row_span > 1:
                attrs += f' rowspan="{row_span}"'
            if col_span > 1:
                attrs += f' colspan="{col_span}"'
            text = html_mod.escape(_cell_plain_text(cell))
            cells_html.append(f"<{tag}{attrs}>{text}</{tag}>")
        parts.append("<tr>" + "".join(cells_html) + "</tr>")
    parts.append("</table>")
    return "".join(parts)


_PIPE_SPLIT_RE = re.compile(r"(?<!\\)\|")
_SEPARATOR_ROW_RE = re.compile(r"^[\s:|-]+$")


def _pipe_block_to_html(block_lines: list[str]) -> str:
    """Convert a GFM pipe-table block to an HTML table (textual fallback)."""
    parsed_rows: list[list[str]] = []
    for line in block_lines:
        stripped = line.strip()
        if _SEPARATOR_ROW_RE.match(stripped):
            continue
        cells = _PIPE_SPLIT_RE.split(stripped)
        # Leading/trailing pipes produce empty edge cells.
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        parsed_rows.append([c.strip().replace("\\|", "|") for c in cells])

    parts: list[str] = ["<table>"]
    for ri, row in enumerate(parsed_rows):
        tag = "th" if ri == 0 and len(parsed_rows) >= 2 else "td"
        parts.append("<tr>" + "".join(f"<{tag}>{html_mod.escape(c)}</{tag}>" for c in row) + "</tr>")
    parts.append("</table>")
    return "".join(parts)
