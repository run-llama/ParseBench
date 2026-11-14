"""Provider for Loamist Parse (https://parse.loamist.com).

Loamist Parse is a hosted document parsing API. One request carries one page (a one-page PDF, or a PNG or JPEG
image) and returns the page as layout blocks: ``<div data-bbox="[top, left, bottom, right]" data-label="...">``
elements in reading order, with boxes normalized to 0-1000 and tables as HTML. The blocks are normalized with the
shared layout helpers (``_layout_utils``). Multi-page PDFs are split locally and sent one page per request.

Sign up for a key at https://parse.loamist.com/signup; API reference at https://parse.loamist.com/docs/.

Config (env):
  LOAMIST_PARSE_API_KEY   API key, ``lp_...``  (required)
  LOAMIST_PARSE_BASE_URL  default ``https://parse.loamist.com``

A page usually takes 1 to 5 minutes and a dense page up to 10, so the request timeout is long. The connection is
idle while the page is parsed, so TCP keep-alive is turned on: some networks drop a connection that is quiet for
about 5 minutes. Recommended ``--max_concurrent``: up to 32.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import socket
import ssl
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._layout_utils import (
    build_layout_pages,
    items_to_markdown,
    parse_layout_blocks,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

_DEFAULT_BASE_URL = "https://parse.loamist.com"
_DEFAULT_TIMEOUT_S = 1200.0
_MEDIA_TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _to_xyxy(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """API boxes are [top, left, bottom, right]; the shared helpers take [x1, y1, x2, y2]."""
    for item in items:
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            top, left, bottom, right = bbox
            item["bbox"] = [left, top, right, bottom]
    return items


class _KeepAliveHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection with TCP keep-alive probes every 30 s while the server works on the page."""

    def connect(self) -> None:
        super().connect()
        sock = self.sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (("TCP_KEEPIDLE", 30), ("TCP_KEEPALIVE", 30), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 8)):
            if hasattr(socket, name):
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
                except OSError:
                    pass


def _split_pdf(data: bytes) -> list[bytes]:
    """One-page PDFs are sent as they are; longer PDFs are split into one-page PDFs (pypdf, if installed)."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return [data]
    reader = PdfReader(io.BytesIO(data))
    if len(reader.pages) <= 1:
        return [data]
    pages = []
    for page in reader.pages:
        writer = PdfWriter()
        writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        pages.append(buf.getvalue())
    return pages


@register_provider("loamist_parse")
class LoamistParseProvider(Provider):
    """Loamist Parse: one page per request, layout blocks normalized with the shared layout helpers."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._api_key = self.base_config.get("api_key") or os.getenv("LOAMIST_PARSE_API_KEY", "")
        if not self._api_key:
            raise ProviderConfigError(
                "LOAMIST_PARSE_API_KEY is required (sign up at https://parse.loamist.com/signup)."
            )
        base = self.base_config.get("base_url") or os.getenv("LOAMIST_PARSE_BASE_URL") or _DEFAULT_BASE_URL
        parts = urlsplit(base)
        if parts.scheme != "https" or not parts.hostname:
            raise ProviderConfigError(f"LOAMIST_PARSE_BASE_URL must be an https URL, got {base!r}")
        self._host = parts.hostname
        self._port = parts.port or 443
        self._timeout = float(self.base_config.get("timeout", _DEFAULT_TIMEOUT_S))
        self._ssl = ssl.create_default_context()

    def _parse_page(self, data: bytes, media_type: str, filename: str) -> dict[str, Any]:
        boundary = uuid.uuid4().hex
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: {media_type}\r\n\r\n"
            ).encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        conn = _KeepAliveHTTPSConnection(self._host, self._port, timeout=self._timeout, context=self._ssl)
        try:
            conn.request(
                "POST",
                "/v1/parse?format=layout",
                body=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "parse-bench-loamist-parse/1",
                },
            )
            res = conn.getresponse()
            status, text = res.status, res.read().decode("utf-8", errors="replace")
        except (OSError, http.client.HTTPException) as e:
            raise ProviderTransientError(f"Loamist Parse request failed: {type(e).__name__}: {e}") from e
        finally:
            conn.close()
        if status == 429:
            raise ProviderRateLimitError(f"Loamist Parse 429: {text[:300]}")
        if status >= 500:
            raise ProviderTransientError(f"Loamist Parse {status}: {text[:300]}")
        if status != 200:
            raise ProviderPermanentError(f"Loamist Parse {status}: {text[:300]}")
        try:
            page: dict[str, Any] = json.loads(text)["pages"][0]
            return page
        except (ValueError, KeyError, IndexError) as e:
            raise ProviderTransientError(f"Loamist Parse returned an unreadable response: {text[:300]}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"Loamist Parse only supports PARSE, got {request.product_type}")
        path = Path(request.source_file_path)
        media_type = _MEDIA_TYPES.get(path.suffix.lower())
        if media_type is None:
            raise ProviderPermanentError(f"Unsupported file type for Loamist Parse: {path.suffix}")
        started_at = datetime.now()
        data = path.read_bytes()
        chunks = _split_pdf(data) if media_type == "application/pdf" else [data]
        pages: list[dict[str, Any]] = []
        for page_index, chunk in enumerate(chunks):
            page = self._parse_page(chunk, media_type, f"page{path.suffix.lower()}")
            raw_content = page.get("layout", "")
            pages.append(
                {
                    "page_index": page_index,
                    "items": _to_xyxy(parse_layout_blocks(raw_content)),
                    "raw_content": raw_content,
                    "width": page.get("width") or 0,
                    "height": page.get("height") or 0,
                }
            )
        completed_at = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"mode": "parse_with_layout_file", "bbox_scale": 1000, "num_pages": len(pages), "pages": pages},
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"Loamist Parse only supports PARSE, got {raw_result.product_type}")
        pages: list[PageIR] = []
        markdowns: list[str] = []
        layout_pages: list[ParseLayoutPageIR] = []
        for page in raw_result.raw_output.get("pages", []):
            page_index = page.get("page_index", 0)
            items = page.get("items", [])
            markdown = items_to_markdown(items)
            layout_pages.extend(
                build_layout_pages(
                    items,
                    page.get("width", 0),
                    page.get("height", 0),
                    markdown,
                    page_number=page_index + 1,
                    bbox_scale=raw_result.raw_output.get("bbox_scale", 1000),
                )
            )
            pages.append(PageIR(page_index=page_index, markdown=markdown))
            markdowns.append(markdown)
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown="\n\n".join(markdowns),
            layout_pages=layout_pages,
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
