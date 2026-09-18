"""Provider for Nutrient DWS Data Extraction PARSE.

Only the options in the published contract are sent (``format``/``formats``,
``includeWords``). Unknown options are accepted and ignored rather than rejected,
and some combinations degrade the markdown silently.

Credits are charged per page. ``usage.data_extraction_credits.cost`` is recorded
on ``raw_output`` as ``credits_used``; its sibling ``remainingCredits`` is not a
running balance (the API returns ``quota - cost_of_this_request`` each call) and
must not be summed. USD uses the Pro plan rate, overridable with
``NUTRIENT_DWS_CREDIT_RATE_USD``.

Config: ``mode``, ``body_source``, ``base_url``, ``timeout_s``, ``retry_count``,
``retry_delay_s``, ``api_version``, ``engine_version``. Key from ``api_key`` or
env NUTRIENT_DWS_API_KEY / DWS_API_KEY / NUTRIENT_DATA_EXTRACTION_API_KEY /
DATA_EXTRACTION_API_KEY.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

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

_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

_API_KEY_ENV = (
    "NUTRIENT_DWS_API_KEY",
    "DWS_API_KEY",
    "NUTRIENT_DATA_EXTRACTION_API_KEY",
    "DATA_EXTRACTION_API_KEY",
)

# Only these modes produce spatial output; `text` is markdown-only and the API
# rejects the text+spatial combination outright.
_SPATIAL_MODES = frozenset({"structure", "understand", "agentic"})
_MODES = frozenset({"text"}) | _SPATIAL_MODES

_NO_ORDER = 2**31 - 1
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?(\d+)(?:\s+of\s+\d+)?$", re.IGNORECASE)

# DWS (type, role) -> ParseBench Canonical layout label. A role missing here
# flattens to "Text" and loses its attribution attrs. Keep in sync with
# _NUTRIENT_DWS_TO_LLAMAPARSE_V3_LABEL in the layout adapter.
_CANONICAL_LABEL = {
    ("paragraph", "Title"): "Title",
    ("paragraph", "SectionHeader"): "Section-header",
    ("paragraph", "Header"): "Page-header",
    ("paragraph", "Footer"): "Page-footer",
    ("paragraph", "Caption"): "Caption",
    ("paragraph", "Footnote"): "Footnote",
    ("paragraph", "ListItem"): "List-item",
    ("paragraph", "DocumentIndex"): "Document Index",
    ("paragraph", "Code"): "Code",
    ("table", None): "Table",
    ("picture", None): "Picture",
    ("chart", None): "Picture",
    ("keyValueRegion", None): "Key-Value Region",
    ("form", None): "Key-Value Region",
    # `handwriting` has no canonical label of its own, so Text is deliberate.
    ("formula", None): "Formula",
    ("handwriting", None): "Text",
}


@register_provider("nutrient_dws")
class NutrientDwsProvider(Provider):
    """Provider that parses documents via the hosted Nutrient DWS parse API."""

    # USD per Data Extraction credit on the Pro plan ($500/mo for 500,000).
    # Data Extraction credits are a separate pool from the Processor API's.
    _DEFAULT_CREDIT_RATE_USD: float | None = 0.0012

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        mode = str(self.base_config.get("mode", "understand")).lower()
        if mode not in _MODES:
            raise ProviderConfigError(f"unknown DWS parse mode '{mode}'; expected one of {sorted(_MODES)}")
        self._mode = mode
        self._spatial = mode in _SPATIAL_MODES

        # Graded body: "graph" renders from output.elements with inline styling
        # from word flags (the default; word flags are only consumed here).
        # "hosted" grades output.markdown verbatim.
        body_source = str(
            os.environ.get("NUTRIENT_DWS_BODY_SOURCE") or self.base_config.get("body_source", "graph")
        ).lower()
        if body_source not in ("graph", "hosted"):
            raise ProviderConfigError(f"body_source must be 'graph' or 'hosted', got '{body_source}'")
        self._body_source = body_source
        # `text` mode returns no elements at all, so there is no graph to render
        # from and the hosted markdown is the only possible body.
        if not self._spatial:
            self._body_source = "hosted"

        self._api_key = self.base_config.get("api_key") or _first_env(*_API_KEY_ENV)
        if not self._api_key:
            raise ProviderConfigError(
                "A DWS API key is required. Set base_config['api_key'] or one of the "
                f"environment variables: {', '.join(_API_KEY_ENV)}."
            )

        self._base_url = (
            self.base_config.get("base_url")
            or _first_env("NUTRIENT_DWS_BASE_URL", "DWS_BASE_URL")
            or "https://api.nutrient.io"
        ).rstrip("/")
        self._timeout = float(os.environ.get("NUTRIENT_DWS_TIMEOUT_SECONDS") or self.base_config.get("timeout_s", 600))
        self._retry_count = int(os.environ.get("NUTRIENT_DWS_RETRY_COUNT") or self.base_config.get("retry_count", 5))
        self._retry_delay = float(
            os.environ.get("NUTRIENT_DWS_RETRY_DELAY_SECONDS") or self.base_config.get("retry_delay_s", 30)
        )
        self._api_version = self.base_config.get("api_version") or os.environ.get("NUTRIENT_DWS_API_VERSION")
        self._engine_version = self.base_config.get("engine_version") or os.environ.get("NUTRIENT_DWS_ENGINE_VERSION")

        rate = os.environ.get("NUTRIENT_DWS_CREDIT_RATE_USD") or self.base_config.get("credit_rate_usd")
        self._credit_rate_usd = float(rate) if rate else self._DEFAULT_CREDIT_RATE_USD

    @property
    def credit_rate_usd(self) -> float | None:
        return self._credit_rate_usd

    # ---- request building --------------------------------------------------
    def _instructions(self) -> str:
        # `text` mode takes the singular `format`; the spatial modes take
        # `formats` and additionally return `output.elements`.
        # Contract options only. `useHtmlTables` /
        # `enableSemanticBlockFormatting` / `includeHeadersAndFooters` are not in
        # the contract and that combination strips chart tables and interleaves
        # running headers into the body. `includeWords` defaults to false and
        # carries the style flags the formatting rules score.
        if self._spatial:
            output: dict[str, Any] = {
                "formats": ["markdown", "spatial"],
                "includeWords": True,
            }
        else:
            output = {"format": "markdown"}
        return json.dumps({"mode": self._mode, "output": output})

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._api_version:
            headers["x-nutrient-api-version"] = self._api_version
        if self._engine_version:
            headers["x-nutrient-engine-version"] = self._engine_version
        return headers

    def _parse(self, src: Path) -> dict[str, Any]:
        content_type = _CONTENT_TYPES.get(src.suffix.lower(), "application/octet-stream")
        instructions = self._instructions()
        url = f"{self._base_url}/extraction/parse"
        last_detail = ""

        for attempt in range(self._retry_count + 1):
            files = {
                "file": (src.name, src.read_bytes(), content_type),
                "instructions": (None, instructions, "application/json"),
            }
            try:
                response = httpx.post(url, headers=self._headers(), files=files, timeout=self._timeout)
            except httpx.TimeoutException as e:
                last_detail = f"timeout after {self._timeout}s ({e})"
                if attempt < self._retry_count:
                    time.sleep(self._backoff(None, attempt))
                    continue
                raise ProviderTransientError(f"DWS parse {last_detail}") from e
            except httpx.HTTPError as e:
                last_detail = repr(e)
                if attempt < self._retry_count:
                    time.sleep(self._backoff(None, attempt))
                    continue
                raise ProviderTransientError(f"DWS parse transport error: {e}") from e

            if response.status_code == 200:
                return response.json()

            body = response.text[:500]
            # 408, 429 and 5xx are worth another attempt; anything else is the
            # caller's fault (bad key, unsupported input) and retrying burns time.
            # 408 is the server giving up on a slow document, not a bad request —
            # treating it as permanent silently drops that document from the run.
            if response.status_code == 429:
                if attempt < self._retry_count:
                    time.sleep(self._backoff(response, attempt))
                    continue
                raise ProviderRateLimitError(f"DWS parse rate limited: {body}")
            if response.status_code == 408 or response.status_code >= 500:
                if attempt < self._retry_count:
                    time.sleep(self._backoff(response, attempt))
                    continue
                raise ProviderTransientError(f"DWS parse HTTP {response.status_code}: {body}")
            raise ProviderPermanentError(f"DWS parse HTTP {response.status_code}: {body}")

        raise ProviderTransientError(f"DWS parse retries exhausted: {last_detail}")

    def _backoff(self, response: "httpx.Response | None", attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return self._retry_delay * (2**attempt)

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _label(el: dict) -> str:
        t, role = el.get("type"), el.get("role")
        return _CANONICAL_LABEL.get((t, role)) or _CANONICAL_LABEL.get((t, None)) or "Text"

    @staticmethod
    def _bbox(b: dict | None, label: str, conf: float | None) -> LayoutSegmentIR | None:
        if not b:
            return None
        return LayoutSegmentIR(
            x=float(b.get("x", 0)),
            y=float(b.get("y", 0)),
            w=float(b.get("width", 0)),
            h=float(b.get("height", 0)),
            confidence=conf if conf is not None else 1.0,
            label=label,
        )

    @staticmethod
    def _order_key(el: dict) -> tuple:
        ro = el.get("readingOrder")
        b = el.get("bounds") or {}
        if ro is None or ro == _NO_ORDER:
            # unordered: fall back to spatial position so it interleaves correctly
            return (1, float(b.get("y", 0)), float(b.get("x", 0)))
        return (0, ro, 0.0)

    @staticmethod
    def _render_table_html(el: dict) -> str:
        cells = el.get("cells") or []
        if not cells:
            return ""
        by_row: dict[int, list[dict]] = {}
        for c in cells:
            by_row.setdefault(c.get("row", 0), []).append(c)
        parts = ["<table>"]
        for r in sorted(by_row):
            parts.append("<tr>")
            for c in sorted(by_row[r], key=lambda x: x.get("column", 0)):
                tag = "th" if (c.get("isHeader") or r == 0) else "td"
                attrs = ""
                if (c.get("rowSpan") or 1) > 1:
                    attrs += f' rowspan="{c["rowSpan"]}"'
                if (c.get("colSpan") or 1) > 1:
                    attrs += f' colspan="{c["colSpan"]}"'
                text = (c.get("text") or "").replace("\n", " ").strip()
                parts.append(f"<{tag}{attrs}>{text}</{tag}>")
            parts.append("</tr>")
        parts.append("</table>")
        return "".join(parts)

    @staticmethod
    def _styled_text(el: dict, plain: str) -> str:
        """Re-render an element's text with inline styling from word-level flags.

        Emits **bold** / *italic* / ~~strike~~ / <u> / <mark> / <sup> / <sub> runs.
        Words are grouped into visual lines by their bounds so a wrapped paragraph
        keeps its line structure. Requires ``includeWords`` on the request.
        """
        words = el.get("words") or []
        if not any(
            w.get("bold")
            or w.get("italic")
            or w.get("underlined")
            or w.get("underline")
            or w.get("strikethrough")
            or w.get("strikeout")
            or w.get("mark")
            or w.get("superscript")
            or w.get("subscript")
            for w in words
        ):
            return plain

        def _yc(w: dict) -> float:
            b = w.get("bounds") or {}
            return float(b.get("y", 0)) + float(b.get("height", 0)) / 2.0

        def _h(w: dict) -> float:
            return float((w.get("bounds") or {}).get("height", 0))

        lines: list[list[dict]] = []
        for w in words:
            if lines and abs(_yc(w) - _yc(lines[-1][-1])) <= max(_h(w), _h(lines[-1][-1]), 1.0) * 0.6:
                lines[-1].append(w)
            else:
                lines.append([w])

        def _key(w: dict) -> tuple:
            return (
                bool(w.get("bold")),
                bool(w.get("italic")),
                bool(w.get("strikethrough") or w.get("strikeout")),
                bool(w.get("superscript")),
                bool(w.get("subscript")),
                bool(w.get("underlined") or w.get("underline")),
                bool(w.get("mark")),
            )

        def _glued(prev: dict | None, curr: dict) -> bool:
            # A script marker split from its base word sits flush against it — attach
            # without a space so markup-stripping reproduces the original text exactly.
            if prev is None or not (
                curr.get("superscript") or curr.get("subscript") or prev.get("superscript") or prev.get("subscript")
            ):
                return False
            pb = prev.get("bounds") or {}
            cb = curr.get("bounds") or {}
            gap = float(cb.get("x", 0)) - (float(pb.get("x", 0)) + float(pb.get("width", 0)))
            height = max(float(pb.get("height", 0)), float(cb.get("height", 0)), 1.0)
            return gap < max(2.0, 0.15 * height)

        def _style_line(line: list[dict]) -> str:
            out = ""
            prev_word: dict | None = None
            i = 0
            while i < len(line):
                key = _key(line[i])
                j = i
                while j + 1 < len(line) and key == _key(line[j + 1]):
                    j += 1
                run_words = line[i : j + 1]
                run = " ".join((w.get("text") or "") for w in run_words).strip()
                first_word = run_words[0]
                i = j + 1
                if not run:
                    continue
                bold, italic, strike, sup, sub, underlined, mark = key
                if sup:
                    run = f"<sup>{run}</sup>"
                elif sub:
                    run = f"<sub>{run}</sub>"
                if strike:
                    run = f"~~{run}~~"
                if underlined:
                    # Markdown has no native underline; <u> is the standard inline-HTML form.
                    run = f"<u>{run}</u>"
                if mark:
                    run = f"<mark>{run}</mark>"
                if bold and italic:
                    run = f"***{run}***"
                elif bold:
                    run = f"**{run}**"
                elif italic:
                    run = f"*{run}*"
                if not out:
                    out = run
                elif _glued(prev_word, first_word):
                    out += run
                else:
                    out += " " + run
                prev_word = run_words[-1]
            return out

        rendered = "\n".join(s for s in (_style_line(line) for line in lines) if s)
        return rendered or plain

    def _graph_body_md(self, el: dict, by_id: dict[str, dict]) -> str:
        """Markdown for one element, rendered from the graph (adapter convention)."""
        etype = el.get("type")
        role = el.get("role") or ""
        text = (el.get("text") or "").strip()
        if etype == "table":
            return self._render_table_html(el)
        if etype == "chart":
            return self._chart_table_html(el, by_id)
        if etype == "picture":
            return ""
        if not text:
            return ""
        if role == "Code":
            return "```\n" + text + "\n```"
        if role in ("Title", "SectionHeader"):
            level = el.get("headingLevel") or (1 if role == "Title" else 2)
            # A markdown heading ends at the newline, so a heading that wrapped in
            # the PDF must be collapsed to one line or its tail becomes body text.
            styled = self._styled_text(el, text)
            return "#" * int(level) + " " + " ".join(styled.split())
        return self._styled_text(el, text)

    @staticmethod
    def _chart_table_html(el: dict, by_id: dict[str, dict]) -> str:
        """A chart element's data table, with its caption attached.

        The chart scorer reads a value's label from its row, its column or the
        table's ``<caption>``, so the caption is load-bearing.
        """
        table = (el.get("htmlTable") or "").strip()
        if not table:
            return ""
        if "<caption" in table:
            return table
        caption = " ".join(
            text
            for text in (((by_id.get(cid) or {}).get("text") or "").strip() for cid in (el.get("captionIds") or []))
            if text
        )
        # `summary` is the VLM's own one-line description ("Horizontal bar chart
        # comparing ..."). It carries the axis/series wording that chart rules match
        # on, so it stands in when the document has no caption element of its own.
        if not caption:
            caption = (el.get("summary") or "").strip()
        if not caption or not table.startswith("<table"):
            return table
        head, rest = table[: len("<table")], table[len("<table") :]
        closing = rest.find(">")
        if closing == -1:
            return table
        return f"{head}{rest[: closing + 1]}<caption>{caption}</caption>{rest[closing + 1 :]}"

    def _chart_tables_md(self, els: list[dict], by_id: dict[str, dict], already_in: str = "") -> str:
        """Chart tables missing from the hosted markdown, in reading order.

        The hosted markdown usually carries them already, so appending
        unconditionally would duplicate every table.
        """
        out: list[str] = []
        for el in els:
            if el.get("type") != "chart":
                continue
            html = self._chart_table_html(el, by_id)
            if not html:
                continue
            if already_in and self._table_present(el, already_in):
                continue
            out.append(html)
        return "\n\n".join(out)

    @staticmethod
    def _table_present(el: dict, markdown: str) -> bool:
        """Is this chart's data already rendered in `markdown`?

        Compares the first few cell values from ``htmlTable`` — a table already
        present shares its numbers whatever the surrounding markup.
        """
        cells = re.findall(r"<t[dh][^>]*>([^<]+)</t[dh]>", el.get("htmlTable") or "")
        probes = [c.strip() for c in cells if c.strip()][:6]
        if len(probes) < 3:
            return False
        hits = sum(1 for p in probes if p in markdown)
        return hits >= max(2, len(probes) // 2)

    @staticmethod
    def _page_number(el: dict) -> int:
        page = el.get("page") or {}
        num = page.get("pageNumber")
        if isinstance(num, int):
            return num
        idx = page.get("pageIndex")
        return (idx + 1) if isinstance(idx, int) else 1

    def _attach_usage(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record the API's reported credit charge on the raw payload.

        `remainingCredits` is deliberately unused: the API returns
        `quota - cost_of_this_request` each call, so it never decreases.
        """
        out = dict(payload)
        pages = ((out.get("metrics") or {}).get("pagesProcessed")) or 0
        credits = ((out.get("usage") or {}).get("data_extraction_credits") or {}).get("cost")

        out.setdefault("dws_mode", self._mode)
        if pages:
            out.setdefault("num_pages", pages)
        if isinstance(credits, (int, float)):
            out.setdefault("credits_used", float(credits))
            if pages:
                out.setdefault("credits_per_page", float(credits) / pages)
            if self._credit_rate_usd:
                cost_usd = float(credits) * self._credit_rate_usd
                out.setdefault("cost_usd", cost_usd)
                if pages:
                    out.setdefault("cost_per_page_usd", cost_usd / pages)
        return out

    # ---- Provider interface ------------------------------------------------
    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"NutrientDwsProvider only supports PARSE, got {request.product_type}")
        src = Path(request.source_file_path).resolve()
        if not src.exists():
            raise ProviderPermanentError(f"input file not found: {src}")

        started_at = datetime.now()
        payload = self._attach_usage(self._parse(src))
        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=payload,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError("NutrientDwsProvider only supports PARSE")

        payload = raw_result.raw_output or {}
        output = payload.get("output") or {}
        doc_markdown = output.get("markdown") or ""
        elements = output.get("elements") or []

        by_page: dict[int, list[dict]] = {}
        page_dims: dict[int, dict] = {}
        by_id: dict[str, dict] = {}
        for el in elements:
            pnum = self._page_number(el)
            by_page.setdefault(pnum, []).append(el)
            page_dims.setdefault(pnum, el.get("page") or {})
            if el.get("id"):
                by_id[el["id"]] = el

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []

        for pnum in sorted(by_page):
            els = sorted(by_page[pnum], key=self._order_key)
            header_parts: list[str] = []
            footer_parts: list[str] = []
            items: list[LayoutItemIR] = []
            printed_page_number = ""

            for el in els:
                role = el.get("role") or ""
                etype = el.get("type")
                text = (el.get("text") or "").strip()
                label = self._label(el)
                is_header = etype == "paragraph" and role == "Header"
                is_footer = etype == "paragraph" and role == "Footer"

                if (is_header or is_footer) and text:
                    (header_parts if is_header else footer_parts).append(text)
                    m = _PAGE_NUMBER_RE.match(text)
                    if m and not printed_page_number:
                        printed_page_number = m.group(1)

                if etype == "table":
                    html = self._render_table_html(el)
                elif etype == "chart":
                    html = self._chart_table_html(el, by_id)
                else:
                    html = ""
                bb = self._bbox(el.get("bounds"), label, el.get("confidence"))
                items.append(
                    LayoutItemIR(
                        type=label,
                        md=(text if (is_header or is_footer) else self._graph_body_md(el, by_id)),
                        html=html,
                        value=text,
                        bbox=bb,
                        layout_segments=[bb] if bb else [],
                    )
                )

            # The graded body is the hosted product's own markdown, plus the chart
            # tables its exporter leaves out (see _chart_table_html). DWS returns
            # markdown per document, not per page, so a single-page document maps
            # directly; for a multi-page document the per-page view falls back to
            # element text in reading order.
            chart_md = self._chart_tables_md(els, by_id, doc_markdown)
            if self._body_source == "graph":
                # Header/footer text stays in the body (headers open the page,
                # footers close it) IN ADDITION to the structured per-page fields,
                # ground truth counts it on letterheads and sparse pages, which
                # otherwise score 0 despite perfect extraction.
                page_md = "\n\n".join(s for s in (self._graph_body_md(el, by_id) for el in els) if s)
            elif len(by_page) == 1:
                page_md = "\n\n".join(s for s in (doc_markdown, chart_md) if s)
            else:
                page_md = "\n\n".join(
                    s
                    for s in (
                        *(
                            self._render_table_html(el) if el.get("type") == "table" else (el.get("text") or "").strip()
                            for el in els
                        ),
                        chart_md,
                    )
                    if s
                )

            dims = page_dims.get(pnum) or {}
            pages.append(PageIR(page_index=pnum - 1, markdown=page_md))
            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=pnum,
                    width=dims.get("width"),
                    height=dims.get("height"),
                    md=page_md,
                    text=page_md,
                    page_header_markdown="\n\n".join(header_parts),
                    page_footer_markdown="\n\n".join(footer_parts),
                    printed_page_number=printed_page_number,
                    items=items,
                )
            )

        # `text` mode returns no elements at all, so there are no pages to build
        # from above — keep the document markdown scorable by emitting one page.
        if not pages and doc_markdown:
            pages.append(PageIR(page_index=0, markdown=doc_markdown))

        # Document body = the hosted markdown followed by every chart table its
        # exporter dropped, pages in order. For a single-page document this is the
        # same string as pages[0].markdown.
        if self._body_source == "graph":
            document_markdown = "\n\n".join(p.markdown for p in pages if p.markdown)
        else:
            all_chart_md = "\n\n".join(
                s for s in (self._chart_tables_md(by_page[p], by_id, doc_markdown) for p in sorted(by_page)) if s
            )
            document_markdown = "\n\n".join(s for s in (doc_markdown, all_chart_md) if s)
        parse_output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=document_markdown,
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=parse_output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None
