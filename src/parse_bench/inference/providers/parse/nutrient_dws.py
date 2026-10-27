"""Provider for Nutrient DWS Data Extraction PARSE.

Only options from the published contract are sent (``format``/``formats``,
``includeWords``, ``enableInlineFormatting``, and with the hosted body
``includeHeadersAndFooters`` and ``extractWordsFromPictures``). Unknown options are accepted and
ignored rather than rejected.

Credits are charged per page. ``usage.data_extraction_credits.cost`` is recorded
on ``raw_output`` as ``credits_used``; its sibling ``remainingCredits`` is not a
running balance (the API returns ``quota - cost_of_this_request`` each call) and
must not be summed. USD uses the Free plan pay-as-you-go rate, the basis every
provider on the leaderboard is priced at, overridable with
``NUTRIENT_DWS_CREDIT_RATE_USD``.

Transient failures (408, 429, 5xx, timeouts, transport errors) are raised as
``ProviderTransientError`` / ``ProviderRateLimitError`` on the first occurrence;
retrying them is left to the shared runner, as for every other provider.

Config: ``mode``, ``body_source``, ``base_url``, ``timeout_s``,
``credit_rate_usd``, ``api_version``, ``engine_version``. Key from ``api_key`` or
env NUTRIENT_DWS_API_KEY / DWS_API_KEY / NUTRIENT_DATA_EXTRACTION_API_KEY /
DATA_EXTRACTION_API_KEY.
"""

import json
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

    # USD per Data Extraction credit at the Free plan pay-as-you-go rate, the
    # common pricing basis of the leaderboard (USD 0.001 since 2026-09-23).
    # Data Extraction credits are a separate pool from the Processor API's.
    _DEFAULT_CREDIT_RATE_USD: float | None = 0.001

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        mode = str(self.base_config.get("mode", "understand")).lower()
        if mode not in _MODES:
            raise ProviderConfigError(f"unknown DWS parse mode '{mode}'; expected one of {sorted(_MODES)}")
        self._mode = mode
        self._spatial = mode in _SPATIAL_MODES

        # Graded body: "hosted" (the default) grades the product's own
        # output.markdown. "graph" renders the body from output.elements instead,
        # with inline styling from the word flags.
        body_source = str(
            os.environ.get("NUTRIENT_DWS_BODY_SOURCE") or self.base_config.get("body_source", "hosted")
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
        # `includeWords` defaults to false and carries the style flags the
        # formatting rules score. `enableInlineFormatting` defaults to false and
        # writes those styles into `output.markdown` (the body when `body_source`
        # is "hosted"); the graph body styles itself from the word flags.
        # When `output.markdown` is the graded body, `includeHeadersAndFooters`
        # and `extractWordsFromPictures` (both default false) are switched on:
        # the graph body keeps running headers, footers and the words inside
        # pictures, so the hosted body is asked for the same content. The graph
        # body never reads `output.markdown`, so its request is unchanged. The
        # other Markdown switches (`useHtmlTables`, `enableSemanticBlockFormatting`)
        # keep the defaults DWS documents and pins, so they aren't sent.
        if self._spatial:
            output: dict[str, Any] = {
                "formats": ["markdown", "spatial"],
                "includeWords": True,
                "enableInlineFormatting": True,
            }
            if self._body_source == "hosted":
                output["includeHeadersAndFooters"] = True
                output["extractWordsFromPictures"] = True
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
        files = {
            "file": (src.name, src.read_bytes(), content_type),
            "instructions": (None, instructions, "application/json"),
        }
        try:
            response = httpx.post(url, headers=self._headers(), files=files, timeout=self._timeout)
        except httpx.TimeoutException as e:
            raise ProviderTransientError(f"DWS parse timeout after {self._timeout}s ({e})") from e
        except httpx.HTTPError as e:
            raise ProviderTransientError(f"DWS parse transport error: {e}") from e

        if response.status_code == 200:
            return response.json()

        body = response.text[:500]
        # 408, 429 and 5xx are worth another attempt; anything else is the
        # caller's fault (bad key, unsupported input) and retrying burns time.
        # 408 is the server giving up on a slow document, not a bad request —
        # treating it as permanent silently drops that document from the run.
        # The runner owns the retries, so each is raised on first occurrence.
        if response.status_code == 429:
            raise ProviderRateLimitError(f"DWS parse rate limited: {body}")
        if response.status_code == 408 or response.status_code >= 500:
            raise ProviderTransientError(f"DWS parse HTTP {response.status_code}: {body}")
        raise ProviderPermanentError(f"DWS parse HTTP {response.status_code}: {body}")

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
        lines = _join_cjk_line_words(_join_wrapped_words(lines))

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
            if curr.get("_cjk_glue"):
                return True
            if prev is None or not (
                curr.get("superscript") or curr.get("subscript") or prev.get("superscript") or prev.get("subscript")
            ):
                return False
            pb = prev.get("bounds") or {}
            cb = curr.get("bounds") or {}
            gap = float(cb.get("x", 0)) - (float(pb.get("x", 0)) + float(pb.get("width", 0)))
            height = max(float(pb.get("height", 0)), float(cb.get("height", 0)), 1.0)
            return gap < max(2.0, 0.15 * height)

        # A heading is collapsed to one line, so every break inside it is a wrap.
        heading = el.get("role") in ("Title", "SectionHeader")
        right_edge = _right(el)
        # One entry per word: the word, whether it opens a visual line, and whether a style
        # run may continue into it from the line above (a soft wrap) or must close first.
        flat: list[tuple[dict, bool, bool]] = []
        for n, line in enumerate(lines):
            wrap = n > 0 and (heading or _is_soft_wrap(lines[n - 1], line, right_edge))
            flat.extend((w, k == 0 and n > 0, wrap) for k, w in enumerate(line))

        out = ""
        prev_word: dict | None = None
        newline_due = False
        i = 0
        while i < len(flat):
            key = _key(flat[i][0])
            j = i
            while j + 1 < len(flat) and key == _key(flat[j + 1][0]) and (not flat[j + 1][1] or flat[j + 1][2]):
                j += 1
            run = ""
            for k in range(i, j + 1):
                w, opens_line, _ = flat[k]
                text = (w.get("text") or "").strip()
                if not text:
                    continue
                if run:
                    run += "\n" if opens_line else ("" if w.get("_cjk_glue") else " ")
                run += text
            first_word, starts_line = flat[i][0], flat[i][1]
            prev_in_line = None if starts_line else prev_word
            newline_due = newline_due or starts_line
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
            elif newline_due:
                out += "\n" + run
            elif _glued(prev_in_line, first_word):
                out += run
            else:
                out += " " + run
            newline_due = False
            prev_word = flat[j][0]

        return out or plain

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
            lang = el.get("codeLanguage") or ""
            return "```" + lang + "\n" + (el.get("text") or "").rstrip() + "\n```"
        text = _join_cjk_lines(_join_wrapped_hyphens(text, el.get("words") or []))
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
        # Only text printed on the page becomes a caption. The chart's `summary` is
        # generated text, not something the document shows, so it is never used.
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


_ADDRESS_CHARS = frozenset("/@_%=")


def _looks_like_address(word: str) -> bool:
    # A URL or path keeps its own hyphens: "…/defesa-civil-" wraps before "contabiliza-…".
    return any(c in _ADDRESS_CHARS for c in word)


def _ends_with_wrap_hyphen(word: str) -> bool:
    return len(word) > 1 and word.endswith("-") and word[-2].isalpha() and not _looks_like_address(word)


def _is_wrap_continuation(word: str) -> bool:
    # A continuation opens in lower case, so "Inter-" before "Provincial" keeps its hyphen.
    return bool(word) and word[0].isalpha() and word[0].islower() and not _looks_like_address(word)


# The product's own decision for a hyphen that ends a line, on that line's last word
# (``hyphenAtLineEnd``): "wrap" is a hyphen the line break inserted, so the halves fuse
# without it ("concen-" + "trations"); "compound" is the word's own hyphen, so they fuse
# keeping it ("performance-" + "based"); "suspended" is a hyphen held before a
# conjunction ("pre-" + "and post-tax"), so the halves stay separate words.
_HYPHEN_DECISIONS = frozenset({"wrap", "compound", "suspended"})


def _hyphen_decision(word: dict) -> str | None:
    decision = word.get("hyphenAtLineEnd")
    return decision if decision in _HYPHEN_DECISIONS else None


def _fuse(left: str, right: str, decision: str | None) -> str:
    # Without the product's decision, the hyphen is taken for a wrap hyphen.
    return left + right if decision == "compound" else left[:-1] + right


def _join_wrapped_hyphens(text: str, words: list[dict] | None = None) -> str:
    """Rejoin words the source broke across lines with a hyphen.

    The graph keeps the page's visual lines, so "aufge-" and "führt den" arrive as two
    lines. The hosted Markdown output already joins such words; this does the same for
    the graph body: "aufgeführt" ends the first line and "den" starts the next.

    When the words carry the product's ``hyphenAtLineEnd`` decision, it is followed.
    Without it, a hyphen after a letter is dropped when the next line opens in lower case.
    """
    # The decisions in reading order, matched to the line ends by the word's text.
    decisions = [((w.get("text") or "").strip(), d) for w in words or [] if (d := _hyphen_decision(w))]
    cursor = 0
    out: list[str] = []
    for line in text.split("\n"):
        if out:
            tail = out[-1].rstrip()
            last = tail.split()[-1] if tail.split() else ""
            decision = None
            for k in range(cursor, len(decisions)):
                if decisions[k][0] == last:
                    decision, cursor = decisions[k][1], k + 1
                    break
            first, _, rest = line.lstrip().partition(" ")
            if decision in ("wrap", "compound") and first:
                fuse = True
            elif decision is None:
                fuse = _ends_with_wrap_hyphen(tail) and _is_wrap_continuation(first)
            else:
                fuse = False
            if fuse:
                out[-1] = _fuse(tail, first, decision)
                if rest.strip():
                    out.append(rest.lstrip())
                continue
        out.append(line)
    return "\n".join(out)


def _join_wrapped_words(lines: list[list[dict]]) -> list[list[dict]]:
    """Word-level form of :func:`_join_wrapped_hyphens` for the styled render.

    The joined word keeps the styling of its first part; the continuation leaves its line.
    """
    for i in range(len(lines) - 1):
        if not lines[i] or not lines[i + 1]:
            continue
        last = (lines[i][-1].get("text") or "").strip()
        first = (lines[i + 1][0].get("text") or "").strip()
        decision = _hyphen_decision(lines[i][-1])
        if decision is None:
            fuse = _ends_with_wrap_hyphen(last) and _is_wrap_continuation(first)
        else:
            fuse = decision != "suspended" and bool(first)
        if fuse:
            lines[i][-1] = {**lines[i][-1], "text": _fuse(last, first, decision)}
            lines[i + 1] = lines[i + 1][1:]
    return [line for line in lines if line]


def _is_cjk(c: str) -> bool:
    """Han, kana and CJK/full-width punctuation: scripts written without spaces between words.

    Hangul is left out on purpose: Korean separates words with spaces, so a line break there
    is a word break.
    """
    o = ord(c)
    return (
        0x3000 <= o <= 0x303F  # CJK symbols and punctuation
        or 0x3040 <= o <= 0x30FF  # hiragana, katakana
        or 0x31F0 <= o <= 0x31FF  # katakana extensions
        or 0x3400 <= o <= 0x4DBF  # CJK extension A
        or 0x4E00 <= o <= 0x9FFF  # CJK unified ideographs
        or 0xF900 <= o <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF00 <= o <= 0xFF60  # full-width forms (，？（）etc.)
        or 0xFF61 <= o <= 0xFF9F  # half-width katakana and punctuation
        or 0x20000 <= o <= 0x2FA1F  # CJK extensions B+
    )


def _cjk_boundary(left: str, right: str) -> bool:
    left, right = left.rstrip(), right.lstrip()
    return bool(left) and bool(right) and _is_cjk(left[-1]) and _is_cjk(right[0])


def _join_cjk_lines(text: str) -> str:
    """Join visual lines with nothing where the break falls between two CJK characters.

    Chinese and Japanese put no space between words, so a line break inside a sentence is not
    a word break: "健康保险的基" + "本信息" is one heading, not "健康保险的基 本信息".
    """
    out: list[str] = []
    for line in text.split("\n"):
        if out and _cjk_boundary(out[-1], line):
            out[-1] = out[-1].rstrip() + line.lstrip()
            continue
        out.append(line)
    return "\n".join(out)


def _join_cjk_line_words(lines: list[list[dict]]) -> list[list[dict]]:
    """Word-level form of :func:`_join_cjk_lines` for the styled render.

    A CJK line that continues a CJK line moves up onto it; its first word is marked to attach
    without a space.
    """
    out: list[list[dict]] = []
    for line in lines:
        if out and out[-1] and line:
            last = out[-1][-1].get("text") or ""
            first = line[0].get("text") or ""
            if _cjk_boundary(last, first):
                out[-1] = out[-1] + [{**line[0], "_cjk_glue": True}] + line[1:]
                continue
        out.append(line)
    return out


def _right(item: dict) -> float:
    b = item.get("bounds") or {}
    return float(b.get("x", 0)) + float(b.get("width", 0))


# A line that ends a sentence or a label ("Total:", "(2020)") closes its style run.
_LINE_END_PUNCT = re.compile(r"[.!?:;”\"’)]$")


def _is_soft_wrap(above: list[dict], below: list[dict], right_edge: float) -> bool:
    """Is the break between two visual lines of one element a soft wrap?

    A style run (a bold sentence) continues across a soft wrap and closes at a hard
    break. The break is soft when the line above doesn't end a sentence or label, and
    either the next line continues it in lower case, or the line above is a full line
    of text: at least 35 characters, with no room left at its end for the next line's
    first word. Short lines in title case (a name over a list of credentials, an
    address block) are separate units even when they share a style.
    """
    last = (above[-1].get("text") or "").strip()
    if _LINE_END_PUNCT.search(last):
        return False
    first = below[0]
    if (first.get("text") or "").strip()[:1].islower():
        return True
    if len(" ".join((w.get("text") or "").strip() for w in above)) < 35:
        return False
    b = first.get("bounds") or {}
    space = 0.5 * max(float(b.get("height", 0)), 1.0)
    return right_edge - _right(above[-1]) < float(b.get("width", 0)) + space
