"""Provider for Mistral OCR (the standard ``/v1/ocr`` API).

This provider calls Mistral's hosted OCR endpoint
(``POST https://api.mistral.ai/v1/ocr``) with the OCR 4.0 model
(``mistral-ocr-4-0``).  The PDF is sent inline as a base64 ``data:`` URI, so no
temporary file host is required.

Output
------
The API returns one object per page with:

- ``markdown``: page text as markdown (tables rendered as markdown pipe tables)
- ``dimensions``: ``{dpi, height, width}`` in pixels
- ``blocks``: paragraph-level layout blocks (only when ``include_blocks=True``,
  available on OCR 4+).  Each block has pixel coordinates
  ``top_left_x/top_left_y/bottom_right_x/bottom_right_y``, the block ``content``,
  and a ``type`` in {header, footer, title, text, list, table, caption, image}.

``normalize()`` concatenates the per-page markdown (converting markdown pipe
tables to HTML ``<table>`` so GriTS/TEDS can score them) and, when blocks are
present, builds ``layout_pages`` with each block's bbox normalized to ``[0,1]``
xywh and its ``type`` mapped to a Canonical17 label for layout evaluation.
"""

import base64
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown2
import requests
from pypdf import PdfReader

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

_MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"


# ---------------------------------------------------------------------------
# Pipe-table -> HTML conversion
# ---------------------------------------------------------------------------
#
# Mistral OCR renders tables as markdown pipe tables, but ParseBench's table
# metrics (GriTS/TEDS) only score HTML ``<table>`` elements. ``normalize()``
# converts pipe tables to HTML before storing the markdown; non-table content
# is left untouched so rule-based text evaluation still works.

# Regex for a separator row: only pipes, dashes, colons, and whitespace.
_SEPARATOR_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _is_pipe_table_line(line: str) -> bool:
    """Return True if *line* looks like a markdown pipe-table row."""
    stripped = line.strip()
    return "|" in stripped and not stripped.startswith("<!--")


def _convert_pipe_tables_to_html(md_content: str) -> str:
    """Replace markdown pipe tables with HTML ``<table>`` elements.

    Non-table content is left untouched so that rule-based evaluation
    (which operates on the raw markdown text) still works.
    """
    if not md_content or "|" not in md_content:
        return md_content

    lines = md_content.split("\n")
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if _is_pipe_table_line(line):
            # Collect consecutive pipe-table lines.
            table_lines: list[str] = [line]
            j = i + 1
            while j < len(lines) and _is_pipe_table_line(lines[j]):
                table_lines.append(lines[j])
                j += 1

            # A valid table needs >= 3 lines (header + separator + data)
            # and must contain a separator row.
            has_separator = any(_SEPARATOR_RE.match(tl.strip()) for tl in table_lines)
            if len(table_lines) >= 3 and has_separator:
                table_md = "\n".join(table_lines)
                rendered = markdown2.markdown(table_md, extras=["tables"])
                if "<table" in rendered.lower():
                    result.append(rendered.strip())
                    i = j
                    continue

            # Not a valid table; keep the first line and advance by one.
            result.append(line)
            i += 1
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


# Mistral OCR block ``type`` -> Canonical17 label (the dataset GT label set).
#
# The eight types on the left are the full vocabulary Mistral OCR 4 emits in
# ``blocks`` (observed across the tables/charts/layout benchmark documents);
# the rest are defensive entries for types Mistral may emit on other content.
# The mapping was verified against the layout_attribution ground truth, whose
# ontology is Basic7 {Text, Table, Page-header, Section, Picture, ...}:
#   - ``header``  -> running page identifiers (patent no. / page no. at the top)
#                    -> GT ``Page-header``.
#   - ``title``   -> section headings (e.g. "### Example 23") -> GT ``Section``
#                    (Title and Section-header both fold into Section in Basic7).
#   - ``caption`` -> table/figure captions ("TABLE 78") -> Text in Basic7.
# Mistral uses a separate ``title`` type for headings, so bare ``header`` /
# ``footer`` are page furniture (Page-header / Page-footer), not section headers.
# Unknown types fall back to ``Text`` (a valid Canonical17 label, so the
# evaluator never silently drops the region).
MISTRAL_LABEL_MAP: dict[str, str] = {
    # Full observed OCR-4 block vocabulary
    "header": "Page-header",
    "footer": "Page-footer",
    "title": "Title",
    "text": "Text",
    "list": "List-item",
    "table": "Table",
    "caption": "Caption",
    "image": "Picture",
    # Defensive entries for types seen on other document kinds
    "section_header": "Section-header",
    "footnote": "Footnote",
    "formula": "Formula",
    "code": "Code",
    "picture": "Picture",
    "figure": "Picture",
}

# ---------------------------------------------------------------------------
# Layout pages from blocks
# ---------------------------------------------------------------------------


def _build_layout_pages(pages: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Build ``layout_pages`` from Mistral OCR per-page ``blocks``.

    Each block carries pixel coordinates relative to the page ``dimensions``.
    Coordinates are normalized to ``[0,1]`` xywh for cross-evaluation.
    """
    layout_pages: list[ParseLayoutPageIR] = []

    for page_idx, page in enumerate(pages):
        blocks = page.get("blocks") or []
        if not isinstance(blocks, list):
            continue

        dims = page.get("dimensions") or {}
        page_w = float(dims.get("width") or 0.0) or 1.0
        page_h = float(dims.get("height") or 0.0) or 1.0

        items: list[LayoutItemIR] = []
        for block in blocks:
            block_type = str(block.get("type", "")).strip().lower()
            canonical_label = MISTRAL_LABEL_MAP.get(block_type, "Text")

            try:
                x1 = float(block["top_left_x"])
                y1 = float(block["top_left_y"])
                x2 = float(block["bottom_right_x"])
                y2 = float(block["bottom_right_y"])
            except (KeyError, TypeError, ValueError):
                continue

            seg = LayoutSegmentIR(
                x=x1 / page_w,
                y=y1 / page_h,
                w=(x2 - x1) / page_w,
                h=(y2 - y1) / page_h,
                confidence=1.0,
                label=canonical_label,
            )

            norm_label = canonical_label.lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label == "picture":
                item_type = "image"
            else:
                item_type = "text"

            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=block.get("content", "") or "",
                    bbox=seg,
                    layout_segments=[seg],
                )
            )

        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_idx + 1,
                width=page_w,
                height=page_h,
                items=items,
            )
        )

    return layout_pages


@register_provider("mistral_ocr")
class MistralOCRProvider(Provider):
    """Provider for the Mistral OCR ``/v1/ocr`` endpoint.

    Config keys
    -----------
    api_key : str
        Mistral API key. Falls back to ``MISTRAL_API_KEY`` env var.
    model : str
        OCR model id (default ``"mistral-ocr-4-0"``).
    include_blocks : bool
        Request paragraph-level layout blocks (default ``True``; OCR 4+ only).
    max_pages : int
        Cap on pages sent to the API for very long documents (default 50).
    timeout : int
        HTTP request timeout in seconds (default 300).
    """

    # OCR 4.0 standard list price: $4 / 1000 pages.
    COST_PER_PAGE_USD = 0.004

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        api_key = self.base_config.get("api_key") or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ProviderConfigError(
                "Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass api_key in base_config."
            )
        self._api_key: str = str(api_key)

        self._model: str = self.base_config.get("model", "mistral-ocr-4-0")
        self._include_blocks: bool = self.base_config.get("include_blocks", True)
        self._max_pages: int = self.base_config.get("max_pages", 50)
        self._timeout: int = self.base_config.get("timeout", 300)
        # Internal backoff for 429 rate limits and 5xx, honoring the server's
        # Retry-After header. Mistral OCR rate-limits aggressively; retrying in
        # the provider (instead of letting docs exhaust the runner's budget)
        # keeps the per-doc success rate at ~100% even under sustained 429s.
        self._rate_limit_retries: int = self.base_config.get("rate_limit_retries", 6)
        self._rate_limit_base_wait: float = self.base_config.get("rate_limit_base_wait", 2.0)
        self._rate_limit_max_wait: float = self.base_config.get("rate_limit_max_wait", 20.0)

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"MistralOCRProvider only supports PARSE, got {request.product_type}")

        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"File not found: {pdf_path}")

        started_at = datetime.now()

        pdf_bytes = pdf_path.read_bytes()
        try:
            num_pages = len(PdfReader(pdf_path).pages)
        except Exception:
            num_pages = 0

        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        document = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{b64}",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "document": document,
            "include_blocks": self._include_blocks,
            "include_image_base64": False,
        }
        # Cap pages for very long docs (the smoke/benchmark docs are short, so
        # this is a no-op there). Only send an explicit page list when the doc
        # exceeds the cap to avoid an out-of-range page error.
        if num_pages and num_pages > self._max_pages:
            payload["pages"] = list(range(self._max_pages))

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Retry 429 (rate limit) and 5xx in-provider, honoring Retry-After, so a
        # rate-limited document waits and succeeds rather than failing once the
        # runner's outer retry budget is exhausted.
        resp = None
        for attempt in range(self._rate_limit_retries + 1):
            try:
                resp = requests.post(
                    _MISTRAL_OCR_URL,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
            except requests.exceptions.Timeout as e:
                raise ProviderTransientError(f"Mistral OCR request timed out: {e}") from e
            except requests.exceptions.ConnectionError as e:
                raise ProviderTransientError(f"Mistral OCR connection error: {e}") from e

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self._rate_limit_retries:
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        wait = 0.0
                    if wait <= 0:
                        wait = self._rate_limit_base_wait * (2.0**attempt)
                    time.sleep(min(wait, self._rate_limit_max_wait))
                    continue
                # Retries exhausted — fall through to the outer runner retry.
                if resp.status_code == 429:
                    raise ProviderRateLimitError(
                        f"Mistral OCR rate limit (429) after {self._rate_limit_retries} retries: {resp.text[:300]}"
                    )
                raise ProviderTransientError(
                    f"Mistral OCR server error ({resp.status_code}) after "
                    f"{self._rate_limit_retries} retries: {resp.text[:300]}"
                )
            break

        assert resp is not None
        if resp.status_code == 401:
            raise ProviderConfigError(f"Mistral OCR unauthorized (401): {resp.text[:500]}")
        if resp.status_code >= 400:
            raise ProviderPermanentError(f"Mistral OCR client error ({resp.status_code}): {resp.text[:500]}")

        raw_output: dict[str, Any] = resp.json()

        # Cost tracking.
        usage = raw_output.get("usage_info") or {}
        pages_processed = usage.get("pages_processed") or len(raw_output.get("pages", [])) or num_pages
        cost_usd = pages_processed * self.COST_PER_PAGE_USD
        raw_output["cost_usd"] = cost_usd
        raw_output["cost_per_page_usd"] = self.COST_PER_PAGE_USD
        raw_output["_config"] = {
            "model": self._model,
            "include_blocks": self._include_blocks,
            "max_pages": self._max_pages,
            "total_pages": num_pages,
        }

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

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"MistralOCRProvider only supports PARSE, got {raw_result.product_type}")

        pages = raw_result.raw_output.get("pages") or []
        page_markdowns = [str(p.get("markdown", "") or "") for p in pages]
        markdown = "\n\n".join(page_markdowns).strip()
        markdown = _convert_pipe_tables_to_html(markdown)

        layout_pages = _build_layout_pages(pages) if pages else []

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
