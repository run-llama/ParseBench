"""Provider for Syntel's document parse API (``POST /v1/parse``, Advanced tier).

A parse is an asynchronous job. ``POST {base_url}/v1/parse`` (multipart: ``file``, ``parser``,
``idempotency_key``) answers ``202`` with ``{"data": {"job_id": ...}}``; the provider polls
``GET {base_url}/v1/jobs/{job_id}`` until the job is ``succeeded``, ``failed`` or ``cancelled``, then
fetches the parsed document from ``GET {base_url}/v1/parse/result/{job_id}``. That response is kept
as ``raw_output`` unchanged, plus ``job_id``, ``num_pages`` and the cost fields.

Config keys
-----------
api_key : str
    API key, sent as a bearer token. Falls back to ``SYNTEL_API_KEY``. Required.
base_url : str
    API root that ``/v1/parse`` is appended to. Falls back to ``SYNTEL_BASE_URL``, then to
    ``DEFAULT_BASE_URL``. There is no local default.
parser : str
    Parse tier. Only ``advanced`` is offered here.
job_timeout : float
    Seconds the provider waits for one job, from submit to result. Default 2900, below the
    pipeline's ``per_file_timeout`` so a slow job surfaces as a retryable error first.
request_timeout : float
    Seconds for each HTTP call. Default 60.
cost_per_page_usd : float | None
    Price per page. Falls back to ``LIST_PRICE_PER_PAGE_USD``. ``None`` records no cost (a blank,
    never a zero).

Errors are classified for the runner, which owns retries: 401/403 raise ``ProviderConfigError``,
429 ``ProviderRateLimitError``, 408, 5xx, transport errors, timeouts and a job that failed with
``retryable: true`` raise ``ProviderTransientError``, any other 4xx and a non-retryable failed job
``ProviderPermanentError``. The provider never retries on its own.

The idempotency key is a hash of the file's bytes, the tier, a token drawn once per provider
instance (one benchmark run) and a per-document attempt counter. A retried request of the same
attempt therefore resumes the job it already started instead of paying for a second parse; a
cancelled, failed or expired job moves the counter so the next attempt starts a new job.

The runner's per-file timeout calls ``cancel``, which cannot stop the thread still polling. That
thread notices at its next poll that its job was cancelled and stops without retrying, and it never
moves the counter or forgets the job of the attempt that replaced it. A job stays registered for
``cancel`` until it ends, also between the runner's retries, so a timeout during a retry's backoff
still cancels it.

Uploads are named by the first 16 hex characters of the sha256 of their bytes plus the lower-cased
extension, so the service never sees a dataset file name.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderError,
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

#: The provider's public name: the provider, layout adapter and label mapper keys, and the
#: ``SYNTEL_*`` environment variables are derived from it.
PROVIDER_NAME = "syntel"
#: The public API root (``https://<host>/<prefix>``, ``/v1/parse`` is appended). ``None`` until a
#: public endpoint exists; set ``SYNTEL_BASE_URL`` meanwhile.
DEFAULT_BASE_URL: str | None = None
#: Price in USD per page. Syntel publishes none (the service runs in the customer's deployment),
#: so no cost is recorded.
LIST_PRICE_PER_PAGE_USD: float | None = None

_ENV_PREFIX = PROVIDER_NAME.upper()
_SUPPORTED_TIERS = ("advanced",)
#: ``data.parser_used`` values the Advanced tier reports.
_ADVANCED_PARSERS = ("advanced-composed",)
_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
_POLL_INITIAL_S = 1.0
_POLL_MAX_S = 5.0

#: Block classes of the service's canonical document mapped to ParseBench item types.
_ITEM_TYPES = {
    "Table": "table",
    "Picture": "image",
    "Figure": "image",
    "Chart": "image",
    "Formula": "formula",
    "Title": "heading",
}


def upload_name(payload: bytes, suffix: str) -> str:
    """The file name the service sees: a content hash, never the dataset's name."""
    return hashlib.sha256(payload).hexdigest()[:16] + suffix.lower()


def _pages(data: dict[str, Any]) -> tuple[list[PageIR], list[ParseLayoutPageIR]]:
    """Pages and layout pages from the response's canonical document (``data.structured``).

    Page Markdown is the service's page Markdown, verbatim. Each layout page carries the page
    Markdown, the page-header and page-footer channels and one item per block, labelled with the
    model's class and boxed in page-normalized coordinates. Grounding is scored from
    ``data.regions`` through the ``syntel`` layout adapter, not from these items.
    """
    pages: list[PageIR] = []
    layouts: list[ParseLayoutPageIR] = []
    try:
        structured = data["structured"]
        if structured["kind"] != "canonical_document" or not structured["pages"]:
            raise ValueError("missing canonical pages")
        for expected, page in enumerate(structured["pages"], 1):
            if page["page_number"] != expected:
                raise ValueError("unordered canonical pages")
            markdown = page["markdown"]
            items: list[LayoutItemIR] = []
            headers: list[str] = []
            footers: list[str] = []
            for block in page["blocks"]:
                cls = block["source_meta"]["model_category"]
                content = block["markdown"]
                x1, y1, x2, y2 = (float(v) for v in block["source_meta"]["normalized_bbox_xyxy"])
                x1, x2 = sorted((min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))))
                y1, y2 = sorted((min(1.0, max(0.0, y1)), min(1.0, max(0.0, y2))))
                segment = LayoutSegmentIR(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=1.0, label=cls)
                if cls == "Page-header" and content:
                    headers.append(content)
                elif cls == "Page-footer" and content:
                    footers.append(content)
                items.append(
                    LayoutItemIR(
                        type=_ITEM_TYPES.get(cls, "text"),
                        md=content,
                        value=content,
                        html=content if "<table" in content.lower() else "",
                        bbox=segment,
                        layout_segments=[segment],
                    )
                )
            pages.append(PageIR(page_index=expected - 1, markdown=markdown))
            layouts.append(
                ParseLayoutPageIR(
                    page_number=expected,
                    width=page["width"],
                    height=page["height"],
                    md=markdown,
                    text=markdown,
                    page_header_markdown="\n\n".join(headers),
                    page_footer_markdown="\n\n".join(footers),
                    items=items,
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderPermanentError(f"Syntel canonical document is incomplete or invalid: {exc}") from exc
    return pages, layouts


@register_provider(PROVIDER_NAME)
class SyntelProvider(Provider):
    """Syntel ``/v1/parse``: submit, poll the job, fetch the result."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        api_key = self.base_config.get("api_key") or os.getenv(f"{_ENV_PREFIX}_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ProviderConfigError(f"Set {_ENV_PREFIX}_API_KEY to use the {PROVIDER_NAME} provider")
        base_url = self.base_config.get("base_url") or os.getenv(f"{_ENV_PREFIX}_BASE_URL") or DEFAULT_BASE_URL
        if not base_url:
            raise ProviderConfigError(f"Set {_ENV_PREFIX}_BASE_URL to the API root that /v1/parse is appended to")
        self._base_url = str(base_url).strip().rstrip("/")
        self._parser = str(self.base_config.get("parser", "advanced"))
        if self._parser not in _SUPPORTED_TIERS:
            raise ProviderConfigError(f"Unsupported parser tier {self._parser!r}; expected one of {_SUPPORTED_TIERS}")
        self._job_timeout = float(self.base_config.get("job_timeout", 2900))
        self._request_timeout = float(self.base_config.get("request_timeout", 60))
        price = self.base_config.get("cost_per_page_usd", LIST_PRICE_PER_PAGE_USD)
        self._cost_per_page_usd = float(price) if price is not None else None
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._run_token = secrets.token_hex(8)
        self._lock = threading.Lock()
        self._attempts: dict[str, int] = {}  # example_id -> attempt counter in the idempotency key
        self._jobs: dict[str, tuple[str, int]] = {}  # example_id -> (job, attempt) not yet ended
        self._http: Any = None

    # -- HTTP ---------------------------------------------------------------------------------

    def _client(self) -> Any:
        with self._lock:
            if self._http is None:
                try:
                    import httpx
                except ImportError as exc:
                    raise ProviderConfigError(f"Install parse-bench[{PROVIDER_NAME}] to use this provider") from exc
                self._http = httpx.Client(
                    base_url=self._base_url,
                    headers=self._headers,
                    timeout=self._request_timeout,
                )
            return self._http

    def _request(self, method: str, path: str, *, allow: tuple[int, ...] = (), **kwargs: Any) -> Any:
        client = self._client()
        import httpx

        try:
            response = client.request(method, path, **kwargs)
        except httpx.TransportError as exc:  # connect, read and timeout errors
            raise ProviderTransientError(f"{PROVIDER_NAME} {method} {path}: {type(exc).__name__}") from exc
        status = response.status_code
        if status < 400 or status in allow:
            return response
        detail = response.text[:300]
        if status in (401, 403):
            raise ProviderConfigError(f"{PROVIDER_NAME} rejected the API key on {method} {path} ({status})")
        if status == 429:
            raise ProviderRateLimitError(f"{PROVIDER_NAME} rate limit on {method} {path}: {detail}")
        if status == 408 or status >= 500:
            raise ProviderTransientError(f"{PROVIDER_NAME} {method} {path} ({status}): {detail}")
        raise ProviderPermanentError(f"{PROVIDER_NAME} {method} {path} ({status}): {detail}")

    @staticmethod
    def _json(response: Any, context: str) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderTransientError(f"{PROVIDER_NAME} returned invalid JSON on {context}") from exc
        if not isinstance(body, dict):
            raise ProviderPermanentError(f"{PROVIDER_NAME} returned a non-object on {context}")
        return body

    # -- Job lifecycle --------------------------------------------------------------------------

    def _idempotency_key(self, payload: bytes, attempt: int) -> str:
        digest = hashlib.sha256(payload)
        for part in (self._parser, self._run_token, str(attempt)):
            digest.update(b"\0" + part.encode())
        return digest.hexdigest()

    def _attempt(self, example_id: str) -> int:
        with self._lock:
            return self._attempts.get(example_id, 0)

    def _end_attempt(self, example_id: str, attempt: int, job_id: str) -> None:
        """The job of ``attempt`` has ended without a result: the next submit starts a new job.

        Only the attempt still current moves the counter, so a thread that outlived its timeout
        never pushes the attempt that replaced it onto yet another job.
        """
        with self._lock:
            if self._attempts.get(example_id, 0) == attempt:
                self._attempts[example_id] = attempt + 1
            if self._jobs.get(example_id) == (job_id, attempt):
                del self._jobs[example_id]

    def _cancel_job(self, job_id: str) -> None:
        try:
            self._request("POST", f"/v1/jobs/{job_id}/cancel")
        except ProviderError:
            pass  # best effort; the next attempt uses a new key either way

    def cancel(self, example_id: str) -> bool:
        """Cancel the job of ``example_id`` (the runner's per-file timeout).

        The attempt ends here, whether or not a job is registered: a submit still in flight sees
        it and cancels the job it gets back.
        """
        with self._lock:
            entry = self._jobs.pop(example_id, None)
            self._attempts[example_id] = self._attempts.get(example_id, 0) + 1
        if entry is None:
            return False
        self._cancel_job(entry[0])
        return True

    def _submit(self, path: Path, attempt: int) -> str:
        payload = path.read_bytes()
        suffix = path.suffix.lower()
        content_type = _CONTENT_TYPES.get(suffix)
        if content_type is None:
            raise ProviderPermanentError(f"Unsupported file type {path.suffix!r}")
        response = self._request(
            "POST",
            "/v1/parse",
            files={"file": (upload_name(payload, suffix), payload, content_type)},
            data={"parser": self._parser, "idempotency_key": self._idempotency_key(payload, attempt)},
        )
        data = self._json(response, "POST /v1/parse").get("data")
        job_id = data.get("job_id") if isinstance(data, dict) else None
        if response.status_code != 202 or not job_id:
            raise ProviderPermanentError(f"{PROVIDER_NAME} POST /v1/parse returned no job ({response.status_code})")
        return str(job_id)

    def _registered(self, example_id: str, job_id: str, attempt: int) -> bool:
        with self._lock:
            return self._jobs.get(example_id) == (job_id, attempt)

    def _await(self, example_id: str, job_id: str, attempt: int) -> dict[str, Any]:
        deadline = time.monotonic() + self._job_timeout
        interval = _POLL_INITIAL_S
        while True:
            if not self._registered(example_id, job_id, attempt):
                # cancel() ended this attempt (the runner's timeout); permanent, so this thread's
                # own retry loop stops instead of racing the runner's next attempt.
                raise ProviderPermanentError(f"{PROVIDER_NAME} job {job_id} was cancelled by the runner")
            body = self._json(self._request("GET", f"/v1/jobs/{job_id}"), "GET /v1/jobs")
            job = body.get("data") if isinstance(body.get("data"), dict) else body
            status = str(job.get("status") or "").lower()
            if status == "succeeded":
                response = self._request("GET", f"/v1/parse/result/{job_id}", allow=(404,))
                if response.status_code == 404:
                    # The result expired before it was read; the retry parses again.
                    self._end_attempt(example_id, attempt, job_id)
                    raise ProviderTransientError(f"{PROVIDER_NAME} result of job {job_id} has expired")
                return self._json(response, "GET /v1/parse/result")
            if status in ("failed", "cancelled"):
                self._end_attempt(example_id, attempt, job_id)
                error = job.get("error") if isinstance(job.get("error"), dict) else {}
                message = f"{PROVIDER_NAME} job {job_id} {status}: {error.get('error_type')}: {error.get('message')}"
                if status == "failed" and error.get("retryable") is True:
                    raise ProviderTransientError(message)
                raise ProviderPermanentError(message)
            if time.monotonic() >= deadline:
                # The job keeps running; a retry of this attempt resumes it through the same key.
                raise ProviderTransientError(
                    f"{PROVIDER_NAME} job {job_id} still {status or 'unknown'} after {self._job_timeout:.0f}s"
                )
            time.sleep(interval)
            interval = min(interval * 2, _POLL_MAX_S)

    # -- Provider -------------------------------------------------------------------------------

    def recompute_cost(self, raw_output: dict[str, Any]) -> None:
        """Cost is pages x the configured price; nothing to re-derive without both."""
        pages = raw_output.get("num_pages")
        if self._cost_per_page_usd is None or not isinstance(pages, int) or isinstance(pages, bool):
            return
        raw_output["cost_per_page_usd"] = self._cost_per_page_usd
        raw_output["cost_usd"] = pages * self._cost_per_page_usd

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"{PROVIDER_NAME} supports PARSE only, got {request.product_type}")
        path = Path(request.source_file_path)
        if not path.is_file():
            raise ProviderPermanentError(f"Source file not found: {path}")

        started_at = datetime.now()
        example_id = request.example_id
        attempt = self._attempt(example_id)
        job_id = self._submit(path, attempt)
        with self._lock:
            current = self._attempts.get(example_id, 0) == attempt
            if current:
                self._jobs[example_id] = (job_id, attempt)
        if not current:
            # cancel() ran while the submit was in flight: this job belongs to no attempt.
            self._cancel_job(job_id)
            raise ProviderPermanentError(f"{PROVIDER_NAME} job {job_id} was cancelled by the runner")
        # A transient error keeps the job registered: the runner's retry resumes it, and a timeout
        # in between still cancels it.
        body = self._await(example_id, job_id, attempt)
        with self._lock:
            if self._jobs.get(example_id) == (job_id, attempt):
                del self._jobs[example_id]
        completed_at = datetime.now()

        data = body.get("data")
        if not isinstance(data, dict):
            raise ProviderPermanentError(f"{PROVIDER_NAME} result of job {job_id} has no data object")
        pages = data.get("total_pages")
        body["job_id"] = job_id
        body["num_pages"] = pages if isinstance(pages, int) and not isinstance(pages, bool) else None
        body["cost_per_page_usd"] = None
        body["cost_usd"] = None
        self.recompute_cost(body)

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=body,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"{PROVIDER_NAME} supports PARSE only, got {raw_result.product_type}")
        data = raw_result.raw_output.get("data")
        if not isinstance(data, dict):
            raise ProviderPermanentError(f"{PROVIDER_NAME} raw output has no data object")
        parser_used = data.get("parser_used")
        if parser_used not in _ADVANCED_PARSERS:
            raise ProviderPermanentError(f"{PROVIDER_NAME} result came from {parser_used!r}, not the Advanced tier")

        markdown = data.get("text") or data.get("markdown") or ""
        pages, layout_pages = _pages(data)
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
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
