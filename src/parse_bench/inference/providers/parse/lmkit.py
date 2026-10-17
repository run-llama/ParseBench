"""Provider for the LM-Kit document parser, served by LM-Kit One (``https://lm-kit.com``).

LM-Kit One is a self-hosted server; the parser runs on your own GPU. Start the public image
(``docker pull lmkitone/lm-kit-one``), then run a pipeline::

    docker run -d --name lmkit --gpus all --network host \\
        -e Security__NetworkAccess=LocalOnly lmkitone/lm-kit-one:latest
    uv run parse-bench run lmkit_high --max_concurrent 4

The server downloads its model on the first request (2.3 GB, once). ``LocalOnly`` keeps the
server on this machine's loopback, where it answers without an API key; nothing is licensed
or locked.

Each document is one ``POST /lmkit/v1/document-parsing`` with ``output_format: Json`` and
``include_markdown: true``: the response carries the parsed document (pages, elements with
category, box and typed content) and the Markdown of the same parse for the document, each page
and each element. A long document may answer ``202`` with a job id, which is polled.

Layout: one box per grounding region (``regions`` in the response: an element, the elements
the parser joined into one unit, such as a label and its value on one row, or a grounding unit
enclosing or splitting them, such as a list as a whole or one of its lines), labelled by its
category; nothing else is emitted. Servers up to 2026.10.5 send the grounding units as layout-only
elements (``repeated_text``); later servers send them as regions listing no element and carrying
their own text and confidence. Both read the same.

Config keys
-----------
effort : str
    ``Low``, ``Medium`` or ``High``.
base_url : str
    Server origin. Falls back to ``LMKIT_ONE_URL``, else ``http://localhost:5189``.
api_key : str
    Optional bearer token, for a server reachable from the network. Falls back to
    ``LMKIT_ONE_API_KEY``.
request_timeout, job_timeout, poll_seconds : float
    Seconds. Defaults 1800, 3600 and 1.

Recommended ``--max_concurrent``: **4**, the server's default number of documents decoded
together. Higher values queue on the server.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
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
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

PROVIDER_NAME = "lmkit"

_DEFAULT_BASE_URL = "http://localhost:5189"
_EFFORTS = ("Low", "Medium", "High")

# LM-Kit element categories -> ParseBench canonical labels; anything else is Text.
LABEL_MAP: dict[str, str] = {
    "text": "Text",
    "title": "Section-header",
    "header": "Page-header",
    "footer": "Page-footer",
    "figure": "Picture",
    "table": "Table",
    "formula": "Formula",
    "figure_caption": "Caption",
    "table_caption": "Caption",
    "formula_caption": "Caption",
    "page_footnote": "Footnote",
    "figure_footnote": "Footnote",
    "table_footnote": "Footnote",
}

# Content kinds whose words stand in the layout view alone: they render nowhere in the reading,
# so the box carries the words as they are.
_LAYOUT_ONLY_CONTENT = ("figure_label", "repeated_text")


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _item_type(element: dict[str, Any]) -> str:
    if (element.get("content") or {}).get("type") == "table":
        return "table"
    return "image" if element.get("category") == "figure" else "text"


def _item_value(element: dict[str, Any], rendering: str) -> str:
    content = element.get("content") or {}
    if content.get("type") in _LAYOUT_ONLY_CONTENT:
        return content.get("text") or ""
    printed = element.get("printed_words") or []
    if element.get("category") == "figure" and printed:
        return " ".join(printed)
    return rendering


def _join_renderings(elements: list[dict[str, Any]], renderings: list[str], category: str) -> str:
    return " ".join(
        rendering
        for element, rendering in zip(elements, renderings, strict=True)
        if element.get("category") == category and rendering
    )


def _rank(category: Any) -> int:
    """Running heads first, running feet last, everything else in reading order between."""
    return 0 if category == "header" else 2 if category == "footer" else 1


def _box(region: dict[str, Any]) -> list[float] | None:
    bbox = region.get("bbox")
    return [float(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None


def _regions(page: dict[str, Any], count: int) -> list[tuple[list[int], list[float] | None]]:
    """The page's grounding regions over its elements: each region the parser formed, its element
    indices and the box it occupies, then every element in no region on its own, boxed by its
    bounds."""
    regions: list[tuple[list[int], list[float] | None]] = []
    taken: set[int] = set()
    for region in page.get("regions") or []:
        members = [m for m in region.get("members") or [] if isinstance(m, int) and 0 <= m < count and m not in taken]
        if members:
            taken.update(members)
            regions.append((members, _box(region)))
    regions.extend(([index], None) for index in range(count) if index not in taken)
    return regions


def _units(page: dict[str, Any]) -> list[dict[str, Any]]:
    """The page's grounding units sent as regions: a region listing no element, with its box and
    its own text (a list as a whole, a line of a paragraph, a drawn icon reading no word)."""
    return [
        region
        for region in page.get("regions") or []
        if not region.get("members") and isinstance(region.get("text"), str) and _box(region) is not None
    ]


def project_page(page: dict[str, Any], page_markdown: str, element_markdown: list[str]) -> ParseLayoutPageIR:
    """One layout page: one item per grounding region (one element, the elements the parser
    joined into one region, or a grounding unit with its own text), in reading order with running
    heads first and running feet last, its box (the region's own when the parser gives one, else
    its elements' bounds) normalized by the page frame."""
    elements = page.get("elements") or []
    if len(element_markdown) != len(elements):
        raise ValueError("element_markdown is not aligned with the page's elements")

    width = float(page.get("width") or 0) or 1.0
    height = float(page.get("height") or 0) or 1.0

    def order(index: int) -> int:
        return _rank(elements[index].get("category"))

    regions = [
        ([m for m in members if len(elements[m].get("bbox") or []) == 4], bbox)
        for members, bbox in _regions(page, len(elements))
    ]
    regions = [(members, bbox) for members, bbox in regions if members]

    # The units read after the page's elements, as the elements they used to be.
    entries: list[tuple[tuple[int, int], list[int], list[float] | None, dict[str, Any] | None]] = [
        (min((order(i), i) for i in members), members, bbox, None) for members, bbox in regions
    ]
    entries.extend(
        ((_rank(unit.get("category")), len(elements) + k), [], _box(unit), unit) for k, unit in enumerate(_units(page))
    )
    entries.sort(key=lambda entry: entry[0])

    items: list[LayoutItemIR] = []
    for _, members, bbox, unit in entries:
        boxes = [bbox] if bbox else [[float(v) for v in elements[m]["bbox"]] for m in members]
        left, top = min(b[0] for b in boxes), min(b[1] for b in boxes)
        right, bottom = max(b[2] for b in boxes), max(b[3] for b in boxes)
        if unit is not None:
            label = LABEL_MAP.get(unit.get("category") or "", "Text")
            confidence = float(unit.get("confidence") if isinstance(unit.get("confidence"), (int, float)) else 1.0)
            item_type = "image" if unit.get("category") == "figure" else "text"
            value = unit.get("text") or ""
        elif len(members) == 1:
            element = elements[members[0]]
            label = LABEL_MAP.get(element.get("category") or "", "Text")
            confidence = float(element.get("confidence", 1.0))
            item_type = _item_type(element)
            value = _item_value(element, element_markdown[members[0]])
        else:
            titles = all(elements[m].get("category") == "title" for m in members)
            label = LABEL_MAP["title"] if titles else LABEL_MAP["text"]
            confidence = min(float(elements[m].get("confidence", 1.0)) for m in members)
            item_type = "text"
            value = "\n".join(element_markdown[m] for m in sorted(members) if element_markdown[m])
        segment = LayoutSegmentIR(
            x=_clamp01(left / width),
            y=_clamp01(top / height),
            w=_clamp01((right - left) / width),
            h=_clamp01((bottom - top) / height),
            label=label,
            confidence=confidence,
        )
        if segment.w <= 0 or segment.h <= 0:
            continue
        items.append(LayoutItemIR(type=item_type, value=value, bbox=segment, layout_segments=[segment]))

    return ParseLayoutPageIR(
        page_number=int(page.get("page_number") or int(page.get("page_index", 0)) + 1),
        width=float(page.get("width") or 0) or None,
        height=float(page.get("height") or 0) or None,
        md=page_markdown,
        page_header_markdown=_join_renderings(elements, element_markdown, "header"),
        page_footer_markdown=_join_renderings(elements, element_markdown, "footer"),
        printed_page_number="",
        items=items,
    )


@register_provider(PROVIDER_NAME)
class LMKitProvider(Provider):
    """Parses each document with a running LM-Kit One server."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        effort = str(self.base_config.get("effort", "High")).capitalize()
        if effort not in _EFFORTS:
            raise ProviderConfigError(f"effort must be one of {', '.join(_EFFORTS)}, got {effort!r}")
        self._effort = effort
        self._base_url = str(
            self.base_config.get("base_url") or os.environ.get("LMKIT_ONE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key = self.base_config.get("api_key") or os.environ.get("LMKIT_ONE_API_KEY")
        self._request_timeout = float(self.base_config.get("request_timeout", 1800))
        self._job_timeout = float(self.base_config.get("job_timeout", 3600))
        self._poll_seconds = float(self.base_config.get("poll_seconds", 1))

    # ── HTTP ───────────────────────────────────────────────────

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(self._base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", "replace")[:500]
            if error.code in (429, 503):
                raise ProviderRateLimitError(f"LM-Kit One busy (HTTP {error.code}): {text}") from error
            if error.code >= 500:
                raise ProviderTransientError(f"LM-Kit One HTTP {error.code}: {text}") from error
            raise ProviderPermanentError(f"LM-Kit One HTTP {error.code}: {text}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise ProviderTransientError(f"LM-Kit One at {self._base_url} unreachable: {error}") from error

    def _parse(self, source: Path) -> dict[str, Any]:
        status, payload = self._call(
            "POST",
            "/lmkit/v1/document-parsing",
            {
                "input": base64.b64encode(source.read_bytes()).decode("ascii"),
                "input_format": "Base64EncodedFile",
                "effort": self._effort,
                "output_format": "Json",
                "include_markdown": True,
            },
        )
        if status != 202:
            return payload

        job_id = payload.get("job_id")
        if not job_id:
            raise ProviderPermanentError("LM-Kit One accepted the parse without a job id")
        deadline = time.monotonic() + self._job_timeout
        while time.monotonic() < deadline:
            time.sleep(self._poll_seconds)
            _, job = self._call("GET", f"/lmkit/v1/jobs/{job_id}")
            state = str(job.get("status", "")).lower()
            if state in ("completed", "1"):
                return job.get("result") or {}
            if state in ("failed", "cancelled", "2", "3"):
                raise ProviderPermanentError(f"LM-Kit One job {job_id} {state}: {job.get('error')}")
        raise ProviderTransientError(f"LM-Kit One job {job_id} did not finish within {self._job_timeout:.0f} s")

    # ── inference ──────────────────────────────────────────────

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"LMKitProvider only supports PARSE, got {request.product_type}")
        source = Path(request.source_file_path)
        if not source.exists():
            raise ProviderPermanentError(f"File not found: {source}")

        started_at = datetime.now()
        result = self._parse(source)
        completed_at = datetime.now()

        if not isinstance(result.get("document"), dict) or result.get("element_markdown") is None:
            raise ProviderPermanentError(
                "LM-Kit One returned no document with its Markdown; "
                "include_markdown needs LM-Kit One 2026.10.4 or later"
            )

        raw_output = {"provider": PROVIDER_NAME, "effort": self._effort, **result}
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

    # ── normalization ──────────────────────────────────────────

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        raw_output = raw_result.raw_output
        document = raw_output.get("document") or {}
        doc_pages = document.get("pages") or []
        page_markdown = raw_output.get("page_markdown") or [""] * len(doc_pages)
        element_markdown = raw_output.get("element_markdown") or [[] for _ in doc_pages]

        pages = [
            PageIR(page_index=int(page.get("page_index", index)), markdown=page_markdown[index] or "")
            for index, page in enumerate(doc_pages)
        ]
        layout_pages = [
            project_page(page, page_markdown[index] or "", list(element_markdown[index]))
            for index, page in enumerate(doc_pages)
        ]
        markdown = raw_output.get("markdown") or "\n\n".join(page.markdown for page in pages)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=markdown,
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
