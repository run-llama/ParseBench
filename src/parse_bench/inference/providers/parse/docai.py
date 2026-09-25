"""Provider for DocAI by ProvidusAI (Providus Technologies), ``https://api.providus.ai``.

Each document goes into a knowledge base named ``ParseBench`` (found by name, created on first
use) with ``auto_parse`` on. The parse job is polled, then the two artifacts the benchmark reads
are fetched: ``result.md`` and ``grounding.json``. Tables arrive in the markdown as pipe tables
and in ``grounding.json`` as HTML with row and column spans; ``normalize`` swaps the HTML in,
since the table and chart metrics only read HTML. Grounding boxes become layout pages.

Config keys
-----------
api_key : str
    DocAI API key. Falls back to ``DOCAI_API_KEY``. Required.
base_url : str
    API origin. Falls back to ``DOCAI_BASE_URL``, else ``https://api.providus.ai``.
knowledge_base : str
    Knowledge base name, matched case-insensitively. Default ``ParseBench``.
parse_options : dict
    Sent with the upload. Default ``{"redact": false}``.
poll_seconds, request_timeout, job_timeout : float
    Seconds. Defaults 5, 120 and 1800.
credit_rate_usd : float
    USD per credit. Falls back to ``DOCAI_CREDIT_RATE_USD``, else the pay-as-you-go rate, 0.01.

One parsed page costs one credit, so ``cost_usd`` is pages x ``credit_rate_usd``. Requests are
single attempts: connection errors and timeouts raise ``ProviderTransientError`` and the runner
retries. A retried document reuses its own upload (running or finished) instead of paying for a
second parse. Uploads are named by a hash of the example id, so the service never sees it.

Recommended ``--max_concurrent``: **8**, twice the service's four parse workers, so each
worker has the next page queued when it finishes one. Extra uploads wait server-side inside
``job_timeout``.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

_DEFAULT_BASE_URL = "https://api.providus.ai"
_ACTIVE = ("queued", "running", "started", "in_progress")  # public status is "running"
_CREDIT_RATE_USD = 0.01  # pay-as-you-go price of one credit (one parsed page)
_CONTENT_TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
# Boxes are normalized to [0, 1]; the evaluator scales them to this frame and back.
_VIRTUAL_PAGE_DIM = 1000.0

# DocAI grounding labels -> ParseBench canonical labels.
LABEL_MAP: dict[str, str] = {
    "text": "Text",
    "list_item": "List-item",
    "aside_text": "Text",
    "content": "Text",
    "abstract": "Text",
    "reference": "Text",
    "reference_content": "Text",
    "paragraph_title": "Section-header",
    "doc_title": "Title",
    "header": "Page-header",
    "footer": "Page-footer",
    "number": "Page-footer",
    "table": "Table",
    "image": "Picture",
    "chart": "Picture",
    "seal": "Picture",
    "caption": "Caption",
    "figure_title": "Caption",
    "table_title": "Caption",
    "chart_title": "Caption",
    "footnote": "Footnote",
    "vision_footnote": "Footnote",
    "formula": "Formula",
    "display_formula": "Formula",
    "inline_formula": "Formula",
    "formula_number": "Text",
    "algorithm": "Code",
}

_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_PIPE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _pipe_blocks(md: str) -> list[tuple[int, int]]:
    """(start, end) line spans of GitHub-style pipe tables."""
    lines = md.splitlines()
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _PIPE_ROW.match(lines[i]):
            j = i
            while j < len(lines) and _PIPE_ROW.match(lines[j]):
                j += 1
            if j - i >= 2 and _PIPE_SEP.match(lines[i + 1]):
                spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def _promote_header(table_html: str) -> str:
    """First row's cells become <th>, as the ground truth writes them."""
    m = re.search(r"<tr>.*?</tr>", table_html, re.S)
    if not m:
        return table_html
    first = re.sub(r"<(/?)td\b", r"<\1th", m.group(0))
    return table_html[: m.start()] + first + table_html[m.end() :]


def pipe_tables_to_html(md: str) -> str:
    """Rewrite pipe tables as <table> blocks. A pipe table has no merged cells, so none are produced."""
    out: list[str] = []
    block: list[str] = []

    def flush() -> None:
        if len(block) >= 2 and _PIPE_SEP.match(block[1]):
            rows = [
                [html.escape(c.strip()) for c in r.strip().strip("|").split("|")] for i, r in enumerate(block) if i != 1
            ]
            body = "\n".join(
                "<tr>"
                + "".join(f"<{'th' if i == 0 else 'td'}>{c}</{'th' if i == 0 else 'td'}>" for c in cells)
                + "</tr>"
                for i, cells in enumerate(rows)
            )
            out.append(f"<table>\n{body}\n</table>")
        else:
            out.extend(block)
        block.clear()

    for line in md.splitlines():
        if _PIPE_ROW.match(line):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def tables_to_html(md: str, grounding: dict[str, Any]) -> str:
    """Swap the markdown's pipe tables for the grounding's HTML tables, matched by order.
    When the counts differ, fall back to a plain pipe-to-HTML rewrite."""
    htmls = [
        str(e.get("content") or "")
        for page in grounding.get("pages") or []
        for e in page.get("elements") or []
        if e.get("label") == "table" and str(e.get("content") or "").lstrip().lower().startswith("<table")
    ]
    spans = _pipe_blocks(md)
    if not spans or len(spans) != len(htmls):
        return pipe_tables_to_html(md)
    lines = md.splitlines()
    out: list[str] = []
    pos = 0
    for (a, b), h in zip(spans, htmls, strict=True):
        out.extend(lines[pos:a])
        out.append(_promote_header(h.replace("\r\n", " ").replace("\r", " ")))
        pos = b
    out.extend(lines[pos:])
    return "\n".join(out)


def layout_pages_from_grounding(grounding: dict[str, Any]) -> list[ParseLayoutPageIR]:
    """grounding.json -> layout pages with normalized xywh boxes, canonical labels and the
    structured page-furniture slots the decoration rules read."""
    out: list[ParseLayoutPageIR] = []
    for page in grounding.get("pages") or []:
        items: list[LayoutItemIR] = []
        furniture: dict[str, list[str]] = {"header": [], "footer": [], "number": []}
        for e in page.get("elements") or []:
            raw_label = str(e.get("label") or e.get("type") or "").lower()
            if raw_label in furniture and e.get("content"):
                furniture[raw_label].append(str(e["content"]).strip())
            label = LABEL_MAP.get(raw_label)
            b = e.get("bbox") or {}
            if not label or not b:
                continue
            seg = LayoutSegmentIR(
                x=float(b["x1"]),
                y=float(b["y1"]),
                w=float(b["x2"]) - float(b["x1"]),
                h=float(b["y2"]) - float(b["y1"]),
                confidence=float((e.get("confidence") or {}).get("score") or 1.0),
                label=label,
            )
            kind = "table" if label == "Table" else "image" if label == "Picture" else "text"
            items.append(LayoutItemIR(type=kind, value=str(e.get("content") or ""), bbox=seg, layout_segments=[seg]))
        out.append(
            ParseLayoutPageIR(
                page_number=int(page.get("page_number") or 1),
                width=_VIRTUAL_PAGE_DIM,
                height=_VIRTUAL_PAGE_DIM,
                items=items,
                page_header_markdown=" | ".join(furniture["header"]),
                page_footer_markdown=" | ".join(furniture["footer"]),
                printed_page_number=" | ".join(furniture["number"]),
            )
        )
    return out


def cost_fields(grounding: dict[str, Any], credit_rate_usd: float) -> dict[str, float]:
    """One credit per parsed page."""
    pages = len(grounding.get("pages") or [])
    if not pages:
        return {}
    return {
        "num_pages": pages,
        "credits_used": pages,
        "cost_usd": pages * credit_rate_usd,
        "cost_per_page_usd": credit_rate_usd,
    }


@register_provider("docai")
class DocAIProvider(Provider):
    """Provider for DocAI (ProvidusAI) via its public REST API."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        api_key = self.base_config.get("api_key") or os.getenv("DOCAI_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ProviderConfigError("DocAI API key is required. Set DOCAI_API_KEY or pass api_key in base_config.")
        self._api_key: str = api_key
        base_url = self.base_config.get("base_url") or os.getenv("DOCAI_BASE_URL") or _DEFAULT_BASE_URL
        self._base_url = str(base_url).strip().rstrip("/")
        self._kb_name = str(self.base_config.get("knowledge_base", "ParseBench"))
        self._options = dict(self.base_config.get("parse_options") or {"redact": False})
        self._poll = float(self.base_config.get("poll_seconds", 5))
        self._request_timeout = float(self.base_config.get("request_timeout", 120))
        self._job_timeout = float(self.base_config.get("job_timeout", 1800))
        self._credit_rate = float(
            self.base_config.get("credit_rate_usd") or os.getenv("DOCAI_CREDIT_RATE_USD") or _CREDIT_RATE_USD
        )
        self._kb_id: str | None = None
        self._kb_lock = threading.Lock()
        self._uploads: dict[str, tuple[str, str]] = {}  # upload name -> (file_id, job_id), for retries
        self._http: Any = None

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.Client(
                base_url=self._base_url,
                headers={"x-api-key": self._api_key},
                timeout=self._request_timeout,
            )
        return self._http

    def _check(self, response: Any, context: str) -> Any:
        status = response.status_code
        if status < 400:
            return response
        detail = response.text[:300]
        if status in (401, 403):
            raise ProviderConfigError(f"DocAI auth failed during {context} ({status}): {detail}")
        if status == 402:
            raise ProviderPermanentError(f"DocAI organization has no credits ({context}): {detail}")
        if status == 429:
            raise ProviderRateLimitError(f"DocAI rate limit during {context}: {detail}")
        if status >= 500:
            raise ProviderTransientError(f"DocAI transient during {context} ({status}): {detail}")
        raise ProviderPermanentError(f"DocAI error during {context} ({status}): {detail}")

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        import httpx

        try:
            response = self._client().request(method, path, **kw)
        except httpx.TransportError as e:  # connection, read and timeout errors; the runner retries
            raise ProviderTransientError(f"DocAI {method} {path}: {e}") from e
        return self._check(response, f"{method} {path}")

    def _knowledge_base(self) -> str:
        """Id of the knowledge base named ``knowledge_base``; created on first use."""
        with self._kb_lock:
            if self._kb_id:
                return self._kb_id
            listing = self._req("GET", "/v1/knowledge-bases").json()
            for kb in listing.get("knowledge_bases") or []:
                if str(kb.get("name") or "").lower() == self._kb_name.lower():
                    self._kb_id = str(kb["id"])
                    return self._kb_id
            created = self._req("POST", "/v1/knowledge-bases", json={"name": self._kb_name}).json()
            self._kb_id = str((created.get("knowledge_base") or created)["id"])
            return self._kb_id

    def _existing(self, kb: str, name: str) -> tuple[str | None, str | None]:
        """A copy left parsing by an earlier run is resumed; a finished or failed one is deleted
        first (filenames are unique per knowledge base). Retries within a run use ``_uploads``."""
        for f in self._req("GET", f"/v1/files?kb_id={kb}&limit=5000").json().get("files", []):
            if f.get("filename") != name or f.get("deleted_at"):
                continue
            jobs = self._req("GET", f"/v1/files/{f['id']}/jobs").json().get("jobs", [])
            active = [j for j in jobs if j.get("kind", "parse") == "parse" and j.get("status") in _ACTIVE]
            if active:
                return f["id"], active[0]["job_id"]
            self._req("DELETE", f"/v1/files/{f['id']}")
        return None, None

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"docai provider handles parse only, got {request.product_type}")
        src = Path(request.source_file_path)
        ctype = _CONTENT_TYPES.get(src.suffix.lower())
        if not ctype:
            raise ProviderPermanentError(f"unsupported file type {src.suffix}")
        started = datetime.now(UTC)
        kb = self._knowledge_base()
        name = hashlib.sha256(request.example_id.encode()).hexdigest()[:16] + src.suffix.lower()
        file_id, job_id = self._uploads.get(name) or self._existing(kb, name)
        if file_id is None:
            with src.open("rb") as fh:
                body = self._req(
                    "POST",
                    "/v1/files",
                    files={"file": (name, fh, ctype)},
                    data={"kb_id": kb, "auto_parse": "true", "parse_options": json.dumps(self._options)},
                ).json()
            file_id, job_id = str(body["file"]["id"]), str(body["parse_job"]["job_id"])
        assert job_id is not None
        self._uploads[name] = (file_id, job_id)
        deadline = time.time() + self._job_timeout
        while True:
            job = self._req("GET", f"/v1/files/{file_id}/jobs/{job_id}").json()["job"]
            if job["status"] in ("completed", "failed", "cancelled"):
                break
            if time.time() > deadline:
                raise ProviderTransientError(f"DocAI job {job_id} still {job['status']} after {self._job_timeout}s")
            time.sleep(self._poll)
        if job["status"] != "completed":
            raise ProviderPermanentError(f"DocAI job {job_id} {job['status']}: {job.get('error')}")
        manifests = [
            m
            for m in self._req("GET", f"/v1/files/{file_id}/artefacts").json()["artifacts"]
            if m["artifact_type"] == "parse"
        ]
        if not manifests:
            raise ProviderPermanentError(f"DocAI file {file_id}: no parse manifest")
        links = manifests[0]["links"]
        markdown = self._req("GET", links["result_md"]).text
        grounding = self._req("GET", links["grounding_json"]).json()
        completed = datetime.now(UTC)
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "provider": "docai",
                "file_id": file_id,
                "job_id": job_id,
                "markdown": markdown,
                "grounding": grounding,
                "usage": (job.get("result") or {}).get("usage"),
                "options": self._options,
                **cost_fields(grounding, self._credit_rate),
            },
            started_at=started,
            completed_at=completed,
            latency_in_ms=int((completed - started).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        raw = raw_result.raw_output
        grounding = raw.get("grounding") or {}
        markdown = tables_to_html(raw.get("markdown") or "", grounding)
        output = ParseOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[PageIR(page_index=0, markdown=markdown)],
            layout_pages=layout_pages_from_grounding(grounding),
            markdown=markdown,
            job_id=raw.get("job_id"),
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
