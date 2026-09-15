"""Provider for the anyformat hosted parse API (``https://api.anyformat.ai``).

The atomic ``POST /v3/parse/`` endpoint always runs the lite tier, so a tier is chosen by
running a one-node parse workflow instead: the provider creates it once per instance
(``POST /v3/workflows/``, node ``{"type": "parse", "mode": <tier>, "cache": false}``),
then submits each document with ``POST /v3/workflows/{id}/upload/run/`` and polls
``GET /v3/runs/{run_id}/`` until the run is terminal — the results envelope arrives inline.

Config keys
-----------
api_key : str
    anyformat API key (``af_...``). Falls back to ``ANYFORMAT_API_KEY``. Required.
base_url : str
    API origin. Falls back to ``ANYFORMAT_BASE_URL``, else ``https://api.anyformat.ai``.
mode : str
    Parse tier: ``standard`` (default), ``agentic``, ``lite`` or ``flash``.
workflow_id : str
    Reuse an existing one-node parse workflow instead of creating one.
effort, prompt_hint, figure_enhancement
    Optional parse-node fields, forwarded verbatim when the workflow is created.
credit_rate_usd : float
    Price per credit behind ``cost_usd``. Defaults to the Business-plan list price.
poll_interval, request_timeout, job_timeout : float
    Seconds. Defaults 3, 120 and 900.

Recommended ``--max_concurrent``: **10** — the submission endpoint allows 60 requests/min.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections import defaultdict
from datetime import datetime
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
    LayoutRegionIR,
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

_DEFAULT_BASE_URL = "https://api.anyformat.ai"
_DEFAULT_POLL_INTERVAL_SECONDS = 3.0
# Pages of the workflow listing walked looking for the harness's own; enough for any real org.
_WORKFLOW_LIST_PAGES = 50
# Consecutive failed polls tolerated before a document is abandoned: asking again is cheap, and
# re-parsing because the question failed is not.
_POLL_BLIPS_TOLERATED = 3
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
_DEFAULT_JOB_TIMEOUT_SECONDS = 900.0

_MODES = ("standard", "agentic", "lite", "flash")
# Public list price in credits per page, per tier (docs/concepts/how-credits-work).
CREDITS_PER_PAGE: dict[str, int] = {"flash": 7, "lite": 12, "standard": 25, "agentic": 100}
# Business-plan list price, EUR 0.001 per credit, at the ECB EUR/USD reference rate of
# 2026-09-10 (1.1616): USD 0.029 for a page of `standard`.
DEFAULT_CREDIT_RATE_USD = 0.001 * 1.1616

# Layout coordinates are stored normalized in [0, 1]; the evaluator scales them to this frame.
_VIRTUAL_PAGE_DIM = 1000.0

# The API marks every block with an anchor and every table cell with a citation id; neither is
# content, and both would count as unexpected words against a ground truth that has none.
_ANCHOR_RE = re.compile(r'<a\s+id="(p(\d+)_b\d+)"\s*>\s*</a>\n?')
_DATA_ATTR_RE = re.compile(r'\s+data-[a-z][a-z0-9-]*="[^"]*"')

_BLANK_RUN_RE = re.compile(r"\n{3,}")

_TERMINAL_FAILURE_STATUSES = {"error", "cancelled"}


def clean_markdown(markdown: str) -> str:
    """Drop block anchors and ``data-*`` attributes; keep every word the parser wrote."""
    return _DATA_ATTR_RE.sub("", _ANCHOR_RE.sub("", markdown or ""))


def split_markdown_by_page(markdown: str) -> dict[int, str]:
    """Group the document markdown by the page named in each block's anchor.

    Text before the first anchor belongs to page 1. A document with no anchors is one page.
    """
    pages: dict[int, list[str]] = defaultdict(list)
    parts = _ANCHOR_RE.split(markdown or "")
    if parts[0].strip():
        pages[1].append(parts[0])
    # split() yields (text, anchor_id, page_number, text, ...) triples after the leading text.
    for i in range(1, len(parts), 3):
        page = int(parts[i + 1])
        body = parts[i + 2] if i + 2 < len(parts) else ""
        pages[page].append(body)
    return {
        page: _BLANK_RUN_RE.sub("\n\n", clean_markdown("".join(chunks))).strip()
        for page, chunks in sorted(pages.items())
    }


def _layout_item_type(block_type: str) -> str:
    kind = block_type.strip().lower()
    if kind == "table":
        return "table"
    if kind in {"picture", "image", "figure", "chart"}:
        return "image"
    return "text"


def _regions_of(block: dict[str, Any], label: str) -> list[LayoutRegionIR]:
    """The detections the block is made of, as the API reports them: each one's box, what it was
    detected as, and the words printed inside it. A block that reports none is its own region."""
    regions = []
    for region in block.get("regions") or []:
        if not isinstance(region, dict):
            continue
        box = region.get("bbox") or {}
        try:
            x0, y0, x1, y1 = (float(box[k]) for k in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError):
            continue
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        detected = str(region.get("type") or label)
        regions.append(
            LayoutRegionIR(
                type=detected,
                bbox=LayoutSegmentIR(x=x0, y=y0, w=x1 - x0, h=y1 - y0, label=detected),
                text=clean_markdown(str(region.get("text") or "")),
            )
        )
    return regions


def build_layout_pages(blocks: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Project the API's blocks (normalized ``{x0, y0, x1, y1}`` bboxes) into layout pages."""
    items_by_page: dict[int, list[LayoutItemIR]] = defaultdict(list)
    for block in blocks:
        bbox = block.get("bbox") or {}
        try:
            x0, y0, x1, y1 = (float(bbox[k]) for k in ("x0", "y0", "x1", "y1"))
            page = int(block.get("page") or 1)
        except (KeyError, TypeError, ValueError):
            continue
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        label = str(block.get("type") or "text")
        confidence = block.get("layout_confidence")
        seg = LayoutSegmentIR(
            x=x0,
            y=y0,
            w=w,
            h=h,
            confidence=float(confidence) if confidence is not None else None,
            label=label,
        )
        value = clean_markdown(str(block.get("content") or ""))
        items_by_page[page].append(
            LayoutItemIR(
                type=_layout_item_type(label),
                value=value,
                bbox=seg,
                layout_segments=[seg],
                regions=_regions_of(block, label),
            )
        )
    return [
        ParseLayoutPageIR(page_number=page, width=_VIRTUAL_PAGE_DIM, height=_VIRTUAL_PAGE_DIM, items=items)
        for page, items in sorted(items_by_page.items())
    ]


@register_provider("anyformat")
class AnyformatProvider(Provider):
    """Provider for anyformat's hosted parse via the v3 REST API."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        api_key = self.base_config.get("api_key") or os.getenv("ANYFORMAT_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ProviderConfigError(
                "anyformat API key is required. Set ANYFORMAT_API_KEY or pass api_key in base_config."
            )
        self._api_key: str = api_key

        base_url = self.base_config.get("base_url") or os.getenv("ANYFORMAT_BASE_URL") or _DEFAULT_BASE_URL
        if not isinstance(base_url, str) or not base_url.strip():
            raise ProviderConfigError("anyformat base_url must be a non-empty string")
        self._base_url = base_url.strip().rstrip("/")

        mode = str(self.base_config.get("mode", "standard")).lower()
        if mode not in _MODES:
            raise ProviderConfigError(f"anyformat mode must be one of {_MODES}, got {mode!r}")
        self._mode = mode

        self._node_extras: dict[str, Any] = {
            key: self.base_config[key]
            for key in ("effort", "prompt_hint", "figure_enhancement")
            if self.base_config.get(key) is not None
        }

        workflow_id = self.base_config.get("workflow_id")
        self._workflow_id: str | None = str(workflow_id) if workflow_id else None
        self._workflow_lock = threading.Lock()

        self._credit_rate_usd = float(self.base_config.get("credit_rate_usd", DEFAULT_CREDIT_RATE_USD))
        self._poll_interval = float(self.base_config.get("poll_interval", _DEFAULT_POLL_INTERVAL_SECONDS))
        self._request_timeout = float(self.base_config.get("request_timeout", _DEFAULT_REQUEST_TIMEOUT_SECONDS))
        self._job_timeout = float(self.base_config.get("job_timeout", _DEFAULT_JOB_TIMEOUT_SECONDS))

        self._http: Any = None

    @property
    def credit_rate_usd(self) -> float:
        return self._credit_rate_usd


    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.Client(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._request_timeout,
            )
        return self._http

    def _classify_bad_response(self, response: Any, context: str) -> None:
        status = response.status_code
        if status < 400:
            return
        detail = response.text[:300]
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("error_code"):
                detail = f"{body.get('error_code')}: {body.get('error')}"
        except ValueError:
            pass
        if status in (401, 403):
            raise ProviderConfigError(f"anyformat auth failed during {context} ({status}): {detail}")
        if status == 402:
            raise ProviderPermanentError(f"anyformat organization has no credit ({context}): {detail}")
        if status == 429:
            raise ProviderRateLimitError(f"anyformat rate limit during {context}: {detail}")
        if status >= 500:
            raise ProviderTransientError(f"anyformat transient during {context} ({status}): {detail}")
        raise ProviderPermanentError(f"anyformat error during {context} ({status}): {detail}")

    def _json(self, response: Any, context: str) -> dict[str, Any]:
        self._classify_bad_response(response, context)
        try:
            body = response.json()
        except ValueError as e:
            raise ProviderTransientError(f"anyformat {context} returned non-JSON response: {e}") from e
        if not isinstance(body, dict):
            raise ProviderTransientError(f"anyformat {context} returned a non-object response: {body!r}")
        return body

    def _ensure_workflow(self) -> str:
        with self._workflow_lock:
            if self._workflow_id:
                return self._workflow_id
            node = {"id": "parse_1", "type": "parse", "mode": self._mode, "cache": False, **self._node_extras}
            body = {
                "name": f"parsebench-{self._mode}",
                "description": "ParseBench harness: one-node parse workflow, cache off",
                "nodes": [node],
                "edges": [],
            }
            response = self._client().post("/v3/workflows/", json=body)
            if response.status_code == 409:
                # One org holds one workflow of a given name, so a second run reuses the one
                # the first run left behind.
                self._workflow_id = self._workflow_named(str(body["name"]))
                return self._workflow_id
            created = self._json(response, "workflow create")
            workflow_id = created.get("id")
            if not isinstance(workflow_id, str) or not workflow_id:
                raise ProviderPermanentError(f"anyformat workflow create returned no id: {created}")
            self._workflow_id = workflow_id
            return workflow_id

    def _workflow_named(self, name: str) -> str:
        """The id of the workflow this harness already owns, by the name it gives it.

        Every page of the listing, because an organization used for anything else holds more
        workflows than one page reports, and ours stops being the newest as soon as one is added.
        """
        cursor: str | None = None
        for _ in range(_WORKFLOW_LIST_PAGES):
            params = {"cursor": cursor} if cursor else None
            listing = self._json(self._client().get("/v3/workflows/", params=params), "workflow list")
            if not isinstance(listing, dict):
                break
            for item in listing.get("items") or []:
                if isinstance(item, dict) and item.get("name") == name and isinstance(item.get("id"), str):
                    return str(item["id"])
            nxt = listing.get("next_cursor")
            cursor = nxt if isinstance(nxt, str) and nxt else None
            if not cursor:
                break
        raise ProviderPermanentError(
            f"anyformat reported a name conflict for {name!r} but no workflow of that name is listed"
        )

    def _submit(self, workflow_id: str, file_path: Path) -> str:
        with open(file_path, "rb") as fh:
            response = self._client().post(
                f"/v3/workflows/{workflow_id}/upload/run/",
                files=[("files", (file_path.name, fh, "application/pdf"))],
                # The same PDF stem recurs across ParseBench groups, and the default on a
                # filename collision is a 409 before anything is uploaded.
                data={"on_conflict": "rename"},
                headers={"Idempotency-Key": str(uuid.uuid4())},
            )
        triggered = self._json(response, "upload/run")
        run_id = triggered.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ProviderPermanentError(f"anyformat upload/run returned no run_id: {triggered}")
        return run_id

    def _poll(self, run_id: str) -> dict[str, Any]:
        import time

        deadline = time.monotonic() + self._job_timeout
        last: dict[str, Any] | None = None
        blips = 0
        while time.monotonic() < deadline:
            time.sleep(max(self._poll_interval, 0.1))
            try:
                run = self._json(self._client().get(f"/v3/runs/{run_id}/"), "run poll")
            except ProviderTransientError:
                # A failed question about a job is not a failed job: abandoning it here pays
                # again for a parse that is almost certainly still running.
                blips += 1
                if blips > _POLL_BLIPS_TOLERATED:
                    raise
                continue
            blips = 0
            last = run
            status = run.get("status")
            if status == "processed":
                return run
            if status in _TERMINAL_FAILURE_STATUSES:
                raise ProviderPermanentError(f"anyformat run {run_id} ended with status={status}: {run.get('error')}")
        raise ProviderTransientError(
            f"anyformat run {run_id} did not finish within {self._job_timeout:.0f}s. Last state: {last}"
        )


    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AnyformatProvider only supports PARSE product type, got {request.product_type}"
            )
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()
        try:
            workflow_id = self._ensure_workflow()
            run_id = self._submit(workflow_id, file_path)
            raw_output = self._poll(run_id)
        except (ProviderConfigError, ProviderPermanentError, ProviderRateLimitError, ProviderTransientError):
            raise
        except Exception as e:  # httpx transport errors and timeouts are retryable
            if e.__class__.__module__.startswith("httpx"):
                raise ProviderTransientError(f"anyformat request failed: {e}") from e
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e
        completed_at = datetime.now()

        raw_output["_config"] = {
            "base_url": self._base_url,
            "mode": self._mode,
            "workflow_id": workflow_id,
            "credit_rate_usd": self._credit_rate_usd,
            **self._node_extras,
        }
        # The public API reports neither a page count nor usage: pages come from the blocks
        # and the price from the tier's public rate card.
        raw_output["num_pages"] = _page_count(_parse_section(raw_output))
        self.recompute_cost(raw_output)

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

    def recompute_cost(self, raw_output: dict[str, Any]) -> None:
        pages = raw_output.get("num_pages")
        mode = (raw_output.get("_config") or {}).get("mode", self._mode)
        if not isinstance(pages, int | float) or pages <= 0 or mode not in CREDITS_PER_PAGE:
            return
        credits_per_page = CREDITS_PER_PAGE[mode]
        raw_output["credits_per_page"] = credits_per_page
        raw_output["credits_used"] = credits_per_page * pages
        raw_output["cost_per_page_usd"] = credits_per_page * self._credit_rate_usd
        raw_output["cost_usd"] = raw_output["credits_used"] * self._credit_rate_usd

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AnyformatProvider only supports PARSE product type, got {raw_result.product_type}"
            )
        parse = _parse_section(raw_result.raw_output)
        markdown = str(parse.get("markdown") or "")
        blocks = [b for b in (parse.get("blocks") or []) if isinstance(b, dict)]
        pages = [PageIR(page_index=page - 1, markdown=md) for page, md in split_markdown_by_page(markdown).items()]
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=build_layout_pages(blocks),
            markdown=clean_markdown(markdown),
            job_id=raw_result.raw_output.get("id"),
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


def _parse_section(raw_output: dict[str, Any]) -> dict[str, Any]:
    results = raw_output.get("results")
    parse = results.get("parse") if isinstance(results, dict) else None
    return parse if isinstance(parse, dict) else {}


def _page_count(parse: dict[str, Any]) -> int:
    """Pages, as the highest one any block names.

    The parse result reports no page count, so this is a floor: a document whose last page holds
    no block reports fewer pages than it has. It matters because the published cost is per page,
    and a floor makes that document look cheaper than it is — an error in the direction that
    flatters us, which is the direction to declare rather than leave implied.
    """
    pages = [int(b.get("page") or 0) for b in (parse.get("blocks") or []) if isinstance(b, dict)]
    return max(pages) if pages and max(pages) > 0 else 1
