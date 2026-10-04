"""Provider for the X2Knowledge hosted document-parsing API.

X2Knowledge is an agentic document parser served behind an OpenAI-compatible
``POST {base_url}/chat/completions`` endpoint. One request parses one page: a
single-page PDF, a PNG or a JPEG, sent as a base64 data URL in the only
``image_url`` part of the only user message (no text options, no category hints).
The reply's ``choices[0].message.content`` is a JSON ``x2knowledge.page`` object: the
page Markdown, the page size in pixels, and one layout element per detection with a
Canonical17 ``label``, a ``bbox`` ``[x, y, w, h]`` as fractions of the page (origin
top-left), a reading ``order``, the element ``text``, an optional detector ``score``
and, for tables, an optional ``html`` that is used when present.

* Multi-page PDFs are split locally with ``pypdf`` and the pages are sent
  concurrently (``page_workers``, 1-8). Results keep the page order; if any page
  fails, the whole file fails and the runner decides whether to retry it.
* Each page is exactly one HTTP attempt: no retries, no sleeping, and redirects are
  not followed. 429 -> ``ProviderRateLimitError``; 5xx, 408, timeouts, transport errors,
  invalid or truncated JSON and replies that break the page contract ->
  ``ProviderTransientError``; 401, 404 and 3xx -> ``ProviderConfigError``; any other
  4xx (400 invalid request, 403 insufficient quota, 413 payload too large) ->
  ``ProviderPermanentError``. Retrying is left to the shared runner, as for every
  other provider. Messages carry the HTTP status, the API error ``code`` and
  ``Retry-After``, never the response body.
* Connections use TCP keepalive (a probe after 60 s without traffic, then every 15 s,
  5 probes). A request is silent until its page is done, so without it a connection that
  dies on the network would only fail at ``timeout_s``; with it, within about 2.5 minutes,
  and the runner retries. Proxies set in ``HTTPS_PROXY`` / ``ALL_PROXY`` (minus
  ``NO_PROXY``) still apply.
* ``raw_output`` keeps only the documented page fields (``object``, ``version``,
  ``page``, ``markdown``, ``layout``, ``timing``, ``warnings``); service-internal
  metadata such as ``execution`` is dropped.
* The service answers every request within its 1800 s hard cap, upload and queueing
  included, so ``timeout_s`` only needs to be a little longer than that.
* Cost is pages x ``price_per_page_usd`` (pipeline config, overridden by
  ``X2KNOWLEDGE_PRICE_PER_PAGE_USD``). Nothing is recorded while no price is set, so a
  missing price never shows up as a zero cost; ``recompute_cost`` re-prices saved runs.

Config: ``base_url`` (else ``X2KNOWLEDGE_BASE_URL``, else the public API root
``https://103.118.252.103/v1``), ``model``, ``timeout_s``, ``page_workers``,
``price_per_page_usd``. API key: ``X2KNOWLEDGE_API_KEY`` (sign up at ``https://103.118.252.103/register``).
Needs ``httpx`` and ``pypdf`` (the ``runners`` extra, or ``local`` plus ``anyformat``).
Recommended ``--max_concurrent``: **30**.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import math
import os
import re
import socket
import threading
import urllib.request
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlsplit

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.layout_ontology import CanonicalLabel
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

# The public API root; the pipeline's ``base_url`` or X2KNOWLEDGE_BASE_URL override it.
_DEFAULT_BASE_URL = "https://103.118.252.103/v1"
_DEFAULT_MODEL = "x2knowledge-parse-v1"
_DEFAULT_TIMEOUT_S = 1860.0
_DEFAULT_PAGE_WORKERS = 4
_MAX_PAGE_WORKERS = 8
# The API answers 413 to an encoded request above this size, so such a page can never succeed.
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_INSTALL_HINT = "pip install 'parse-bench[runners]' (or the 'local' and 'anyformat' extras)"
# TCP keepalive: idle seconds before the first probe, seconds between probes, unanswered probes before the connection
# is dropped. A dead connection is noticed after about 60 + 5 x 15 = 135 s.
_KEEPALIVE_IDLE_S, _KEEPALIVE_INTERVAL_S, _KEEPALIVE_PROBES = 60, 15, 5

_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
_CANONICAL_LABELS = frozenset(label.value for label in CanonicalLabel)
# Documented fields of a ``x2knowledge.page`` and its parts; anything else the service adds
# (such as its internal ``execution`` metadata) is not stored.
_PAGE_FIELDS = ("object", "version", "page", "markdown", "layout", "timing", "warnings")
_GEOMETRY_FIELDS = ("index", "width_px", "height_px", "dpi")
_ELEMENT_FIELDS = ("label", "bbox", "order", "text", "score", "html")
# Boxes are rounded to six decimals, so x + w may overshoot 1 by a rounding step.
_BBOX_SLACK = 2e-6
# Marker the layout adapter keys on, so it never claims another provider's output.
_DOCUMENT_OBJECT = "x2knowledge.document"
# Only values of these shapes are echoed into error messages; the rest of a body is withheld.
_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_RETRY_AFTER_RE = re.compile(r"\d{1,6}")


def _is_unit(value: Any) -> bool:
    """A finite JSON number in [0, 1] (booleans excluded)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validated_base_url(value: Any) -> str:
    url = str(value or "").strip().rstrip("/")
    try:
        parts = urlsplit(url)
        _ = parts.port  # raises ValueError for a malformed port
        valid = (
            parts.scheme in ("http", "https")
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and "?" not in url
            and "#" not in url
        )
    except ValueError:
        valid = False
    if not valid:
        raise ProviderConfigError(
            "Set X2KNOWLEDGE_BASE_URL (or the pipeline's base_url) to the API root, "
            "e.g. https://103.118.252.103/v1: an http(s) URL with a host and no credentials, query or fragment."
        )
    return url


def _keepalive_socket_options() -> list[tuple[int, int, int]]:
    """SO_KEEPALIVE plus idle time, interval and probe count where the platform has them (macOS: TCP_KEEPALIVE)."""
    idle = "TCP_KEEPIDLE" if hasattr(socket, "TCP_KEEPIDLE") else "TCP_KEEPALIVE"
    tuning = ((idle, _KEEPALIVE_IDLE_S), ("TCP_KEEPINTVL", _KEEPALIVE_INTERVAL_S), ("TCP_KEEPCNT", _KEEPALIVE_PROBES))
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    options += [(socket.IPPROTO_TCP, getattr(socket, name), value) for name, value in tuning if hasattr(socket, name)]
    return options


def _environment_proxy(url: str) -> str | None:
    """The proxy the environment names for ``url``, read the way httpx reads it (``urllib.request.getproxies()``).

    Our own transport (for the keepalive options) turns httpx's reading of HTTP(S)_PROXY / ALL_PROXY / NO_PROXY off,
    so the provider passes that proxy to the transport itself.
    """
    parts = urlsplit(url)
    proxies = urllib.request.getproxies()
    host = (parts.hostname or "").lower()
    for raw in proxies.get("no", "").split(","):
        entry = raw.strip().lower().lstrip(".")
        if entry == "*" or (entry and (host == entry or host.endswith("." + entry))):
            return None
    proxy = proxies.get(parts.scheme) or proxies.get("all")
    if not proxy:
        return None
    return proxy if "://" in proxy else f"http://{proxy}"


def _positive_seconds(value: Any) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = math.nan
    if not math.isfinite(seconds) or seconds <= 0:
        raise ProviderConfigError("X2Knowledge timeout_s must be a positive number of seconds")
    return seconds


def _price_per_page(config_value: Any) -> float | None:
    """USD per page from X2KNOWLEDGE_PRICE_PER_PAGE_USD or the pipeline config; None when unset."""
    env_value = os.environ.get("X2KNOWLEDGE_PRICE_PER_PAGE_USD", "").strip()
    value = env_value or config_value
    if value is None or value == "":
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        price = math.nan
    if not math.isfinite(price) or price < 0:
        raise ProviderConfigError("X2Knowledge price_per_page_usd must be a non-negative number of USD")
    return price


def _split_pdf(data: bytes) -> list[bytes]:
    """One single-page PDF per page, in page order. A one-page PDF is sent unchanged."""
    from pypdf import PdfReader, PdfWriter

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("the PDF needs a password")
        if len(reader.pages) == 0:
            raise ValueError("the PDF has no pages")
        if len(reader.pages) == 1 and not reader.is_encrypted:
            return [data]
        pages = []
        for page in reader.pages:
            writer = PdfWriter()
            writer.add_page(page)
            buffer = io.BytesIO()
            writer.write(buffer)
            pages.append(buffer.getvalue())
        return pages
    except Exception as e:
        raise ProviderPermanentError(f"X2Knowledge cannot split the PDF into pages ({type(e).__name__}: {e})") from e


def _request_body(model: str, data: bytes, content_type: str) -> bytes:
    """The request for one page: one user message whose only part is the page as a data URL."""
    url = f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}],
    }
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_REQUEST_BYTES:
        raise ProviderPermanentError(
            f"X2Knowledge page request is {len(encoded) / 2**20:.1f} MiB, over the API's 32 MiB request limit"
        )
    return encoded


def _error_detail(response: Any) -> str:
    """HTTP status plus the API error code and Retry-After when they look like what the API sends."""
    detail = [f"HTTP {response.status_code}"]
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        error = None
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and _ERROR_CODE_RE.fullmatch(code):
        detail.append(f"code={code}")
    retry_after = response.headers.get("Retry-After", "").strip()
    if _RETRY_AFTER_RE.fullmatch(retry_after):
        detail.append(f"Retry-After={retry_after}s")
    return ", ".join(detail)


def _raise_for_status(response: Any) -> NoReturn:
    """Map a non-200 reply onto the error taxonomy the runner retries on."""
    status = response.status_code
    detail = _error_detail(response)
    if status == 429:
        raise ProviderRateLimitError(f"X2Knowledge rate or concurrency limit reached ({detail})")
    # 408: a proxy gave up waiting for a slow upload, not a bad request (as in the Nutrient DWS provider).
    if status == 408 or status >= 500:
        raise ProviderTransientError(f"X2Knowledge service error ({detail})")
    if 300 <= status < 400:
        raise ProviderConfigError(
            f"X2Knowledge answered with a redirect ({detail}); redirects are not followed, "
            "set X2KNOWLEDGE_BASE_URL to the API root"
        )
    if status in (401, 404):
        raise ProviderConfigError(
            f"X2Knowledge rejected the request ({detail}); "
            "check X2KNOWLEDGE_API_KEY, X2KNOWLEDGE_BASE_URL and the model"
        )
    if status >= 400:
        raise ProviderPermanentError(f"X2Knowledge rejected the page ({detail})")
    raise ProviderTransientError(f"X2Knowledge returned an unexpected status ({detail})")


def _validated_element(index: int, element: Any) -> dict[str, Any]:
    where = f"layout[{index}]"
    if not isinstance(element, dict):
        raise ValueError(f"{where} is not an object")
    label = element.get("label")
    if not isinstance(label, str) or label not in _CANONICAL_LABELS:
        raise ValueError(f"{where}.label is not a Canonical17 label")
    box = element.get("bbox")
    if not isinstance(box, list) or len(box) != 4 or not all(_is_unit(v) for v in box):
        raise ValueError(f"{where}.bbox is not four numbers in [0, 1]")
    if box[0] + box[2] > 1 + _BBOX_SLACK or box[1] + box[3] > 1 + _BBOX_SLACK:
        raise ValueError(f"{where}.bbox extends past the page")
    if "order" in element and not _is_int(element["order"]):
        raise ValueError(f"{where}.order is not an integer")
    for key in ("text", "html"):
        if key in element and not isinstance(element[key], str):
            raise ValueError(f"{where}.{key} is not a string")
    score = element.get("score")
    if score is not None and not _is_unit(score):
        raise ValueError(f"{where}.score is not a number in [0, 1]")
    return {key: element[key] for key in _ELEMENT_FIELDS if key in element}


def _validated_page(obj: Any) -> dict[str, Any]:
    """Check a reply against the ``x2knowledge.page`` contract and keep only its documented fields.

    A reply that breaks the contract is a service fault, so it is raised as transient.
    """
    try:
        if not isinstance(obj, dict) or obj.get("object") != "x2knowledge.page":
            raise ValueError("not a x2knowledge.page object")
        if not isinstance(obj.get("markdown"), str):
            raise ValueError("markdown is not a string")
        if not isinstance(obj.get("version", ""), str):
            raise ValueError("version is not a string")
        geometry = obj.get("page")
        if not isinstance(geometry, dict):
            raise ValueError("page geometry is missing")
        for key in _GEOMETRY_FIELDS:
            required = key in ("width_px", "height_px")
            if (required or key in geometry) and not (_is_int(geometry.get(key)) and geometry[key] > 0):
                raise ValueError(f"page.{key} is not a positive integer")
        layout = obj.get("layout")
        if not isinstance(layout, list):
            raise ValueError("layout is not a list")
        elements = [_validated_element(index, element) for index, element in enumerate(layout)]
    except ValueError as e:
        raise ProviderTransientError(f"X2Knowledge returned a page that breaks the API contract: {e}") from None

    page = {key: obj[key] for key in _PAGE_FIELDS if key in obj}
    page["page"] = {key: geometry[key] for key in _GEOMETRY_FIELDS if key in geometry}
    page["layout"] = elements
    timing = obj.get("timing")
    seconds = timing.get("seconds") if isinstance(timing, dict) else None
    if isinstance(seconds, int | float) and not isinstance(seconds, bool) and math.isfinite(seconds):
        page["timing"] = {"seconds": seconds}
    else:
        page.pop("timing", None)
    warnings = obj.get("warnings")
    if isinstance(warnings, list):
        page["warnings"] = [warning for warning in warnings if isinstance(warning, str)]
    else:
        page.pop("warnings", None)
    return page


def _layout_item(element: dict[str, Any]) -> LayoutItemIR:
    """Each API element -> one layout item with one normalized segment and its canonical label."""
    label = element["label"]
    x, y, w, h = (float(v) for v in element["bbox"])
    score = element.get("score")
    segment = LayoutSegmentIR(x=x, y=y, w=w, h=h, label=label, confidence=None if score is None else float(score))
    text = element.get("text") or ""
    html = element.get("html") or ""
    return LayoutItemIR(
        type=label,
        md=text,
        html=html,
        value=html if (label == "Table" and html) else text,
        bbox=segment,
        layout_segments=[segment],
    )


@register_provider("x2knowledge")
class X2KnowledgeProvider(Provider):
    """Parse documents through the hosted X2Knowledge API, one request per page."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        try:
            for module in ("httpx", "pypdf"):
                importlib.import_module(module)
        except ImportError as e:
            raise ProviderConfigError(f"X2Knowledge needs {e.name}: {_INSTALL_HINT}") from e

        self._api_key = os.environ.get("X2KNOWLEDGE_API_KEY", "").strip()
        if not self._api_key:
            raise ProviderConfigError("X2Knowledge API key is required. Set X2KNOWLEDGE_API_KEY.")
        self._base_url = _validated_base_url(
            self.base_config.get("base_url") or os.environ.get("X2KNOWLEDGE_BASE_URL") or _DEFAULT_BASE_URL
        )
        self._model = str(self.base_config.get("model") or _DEFAULT_MODEL)
        self._timeout_s = _positive_seconds(self.base_config.get("timeout_s", _DEFAULT_TIMEOUT_S))
        page_workers = self.base_config.get("page_workers", _DEFAULT_PAGE_WORKERS)
        if not _is_int(page_workers) or not 1 <= page_workers <= _MAX_PAGE_WORKERS:
            raise ProviderConfigError(f"X2Knowledge page_workers must be an integer from 1 to {_MAX_PAGE_WORKERS}")
        self._page_workers: int = page_workers
        self._price_per_page_usd = _price_per_page(self.base_config.get("price_per_page_usd"))
        # example_id -> stop flag of its running file, so the runner can cancel a timed-out file.
        self._active: dict[str, threading.Event] = {}
        self._active_lock = threading.Lock()

    # ---- transport ---------------------------------------------------------------
    def _post_page(self, data: bytes, content_type: str, stop: threading.Event) -> dict[str, Any]:
        """Send one page as exactly one HTTP attempt and return its validated ``x2knowledge.page``."""
        import httpx

        if stop.is_set():
            raise ProviderTransientError("X2Knowledge page not sent: the file was cancelled or another page failed")
        body = _request_body(self._model, data, content_type)
        url = f"{self._base_url}/chat/completions"
        try:
            transport = httpx.HTTPTransport(socket_options=_keepalive_socket_options(), proxy=_environment_proxy(url))
        except (ValueError, ImportError, httpx.InvalidURL) as e:
            # The proxy URL may carry credentials, so only the error type goes into the message.
            raise ProviderConfigError(
                f"X2Knowledge cannot use the proxy set in HTTPS_PROXY / ALL_PROXY ({type(e).__name__})"
            ) from e
        try:
            with httpx.Client(transport=transport, follow_redirects=False) as client:
                response = client.post(
                    url,
                    content=body,
                    headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                    timeout=self._timeout_s,
                )
        except httpx.TimeoutException as e:
            raise ProviderTransientError(f"X2Knowledge request timed out after {self._timeout_s:g}s") from e
        except httpx.HTTPError as e:
            raise ProviderTransientError(f"X2Knowledge transport error: {type(e).__name__}: {e}") from e
        except httpx.InvalidURL as e:
            raise ProviderConfigError("X2Knowledge base URL is not a valid URL; check X2KNOWLEDGE_BASE_URL") from e

        if response.status_code != 200:
            _raise_for_status(response)
        try:
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content is not a string")
            page = json.loads(content)
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderTransientError("X2Knowledge returned invalid or truncated JSON (HTTP 200)") from None
        return _validated_page(page)

    def _parse_file(self, src: Path, stop: threading.Event) -> list[dict[str, Any]]:
        content_type = _CONTENT_TYPES.get(src.suffix.lower())
        if content_type is None:
            raise ProviderPermanentError(
                f"X2Knowledge accepts PDF, PNG and JPEG files, not {src.suffix or 'no suffix'!r}"
            )
        try:
            data = src.read_bytes()
        except OSError as e:
            raise ProviderPermanentError(f"X2Knowledge cannot read the input file: {e.strerror or e}") from e
        chunks = _split_pdf(data) if content_type == "application/pdf" else [data]
        if len(chunks) == 1:
            return [self._post_page(chunks[0], content_type, stop)]

        def send(chunk: bytes) -> dict[str, Any]:
            try:
                return self._post_page(chunk, content_type, stop)
            except BaseException:
                stop.set()  # one failed page fails the file: pages not sent yet are skipped
                raise

        # Leaving the pool waits for the pages already in flight, so a runner retry never overlaps them.
        with ThreadPoolExecutor(max_workers=min(self._page_workers, len(chunks))) as pool:
            futures = [pool.submit(send, chunk) for chunk in chunks]
            wait(futures, return_when=FIRST_EXCEPTION)
            for future in futures:
                future.cancel()  # no-op for pages that already finished or are in flight
        for future in futures:  # page order, not completion order
            if not future.cancelled() and (error := future.exception()) is not None:
                raise error
        return [future.result() for future in futures]

    # ---- cost ----------------------------------------------------------------------
    def recompute_cost(self, raw_output: dict[str, Any]) -> None:
        """Price the recorded page count; leave any recorded cost untouched while no price is set."""
        pages = raw_output.get("num_pages")
        if self._price_per_page_usd is None or not _is_int(pages) or pages <= 0:
            return
        raw_output["num_pages_billed"] = pages
        raw_output["cost_per_page_usd"] = self._price_per_page_usd
        raw_output["cost_usd"] = pages * self._price_per_page_usd

    # ---- Provider API ----------------------------------------------------------------
    def cancel(self, example_id: str) -> bool:
        """Stop sending the remaining pages of a file the runner gave up on.

        Requests already in flight end on their own, within the service's hard cap.
        """
        with self._active_lock:
            stop = self._active.get(example_id)
        if stop is None:
            return False
        stop.set()
        return True

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"X2KnowledgeProvider only supports PARSE, got {request.product_type}")
        src = Path(request.source_file_path)
        if not src.is_file():
            raise ProviderPermanentError(f"X2Knowledge input file not found: {src}")

        stop = threading.Event()
        with self._active_lock:
            self._active[request.example_id] = stop
        started_at = datetime.now()
        try:
            pages = self._parse_file(src, stop)
        finally:
            with self._active_lock:
                if self._active.get(request.example_id) is stop:
                    del self._active[request.example_id]
        completed_at = datetime.now()

        raw_output: dict[str, Any] = {
            "provider": "x2knowledge",
            "object": _DOCUMENT_OBJECT,
            "model": self._model,
            "num_pages": len(pages),
            "num_api_calls": len(pages),
            "pages": pages,
        }
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

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        payload = raw_result.raw_output or {}
        raw_pages = payload.get("pages")
        if payload.get("object") != _DOCUMENT_OBJECT or not isinstance(raw_pages, list):
            raise ProviderPermanentError("X2KnowledgeProvider can only normalize x2knowledge.document payloads")

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        for index, page in enumerate(raw_pages):
            markdown = page.get("markdown") or ""
            geometry = page.get("page") or {}
            width, height = geometry.get("width_px"), geometry.get("height_px")
            pages.append(PageIR(page_index=index, markdown=markdown))
            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=index + 1,
                    width=float(width) if width else None,
                    height=float(height) if height else None,
                    md=markdown,
                    items=[_layout_item(element) for element in page.get("layout") or []],
                )
            )

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown="\n\n".join(page.markdown for page in pages),
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
