"""Provider for Webcite document extraction (PARSE).

Webcite's public API takes a file in two calls: ``POST /upload`` (multipart field ``file``) stores it
and returns ``data.asset_id``, then ``POST /extract`` with ``{"asset_id": ...}`` reads it into
whole-document ``markdown`` plus per-page ``units``. A unit carries the page ``text`` and, when the
page was read with element geometry, ``elements`` (type, Markdown and a normalized
``[x0, y0, x1, y1]`` box, top-left origin) and the page ``geometry``. Both calls authenticate with
the ``x-api-key`` header and report the credits charged in ``usage.credits``.

A failed read comes back as HTTP 200 with ``state: "error"`` and empty ``markdown``; ``normalize``
raises on it, so the document is recorded as failed and scores zero.

API: https://api.webcite.co/api/v1 (self-serve keys at https://webcite.co). Set ``WEBCITE_API_KEY``;
``WEBCITE_BASE_URL`` overrides the API root.
"""

import hashlib
import html
import math
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
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

DEFAULT_BASE_URL = "https://api.webcite.co/api/v1"

# Webcite element type -> Canonical17 label string
WEBCITE_LABEL_MAP: dict[str, str] = {
    "text": "Text",
    "title": "Title",
    "section_header": "Section-header",
    "table": "Table",
    "picture": "Picture",
    "list": "List-item",
    "list_item": "List-item",
    "page_header": "Page-header",
    "page_footer": "Page-footer",
    "caption": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
}

# Webcite boxes are already normalized [0,1]; these virtual page dimensions cancel out in evaluation.
_VIRTUAL_PAGE_DIM = 1000.0

_MIME_TYPES = {".pdf": "application/pdf", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@register_provider("webcite")
class WebciteProvider(Provider):
    """Provider for Webcite's public extraction API."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        :param base_config: Optional configuration with:
            - base_url: API root (default: WEBCITE_BASE_URL or https://api.webcite.co/api/v1)
            - timeout: per-request timeout in seconds (default: 900; the ``webcite`` pipeline sets 300)
            - max_attempts: attempts per call for transient failures (default: 3)
            - max_rate_limit_waits: extra waits allowed on HTTP 429 (default: 20)
        """
        super().__init__(provider_name, base_config)
        try:
            import httpx
        except ImportError as e:  # pragma: no cover - httpx is a core dependency
            raise ProviderConfigError("httpx is required for the Webcite provider") from e
        self._httpx = httpx
        self._api_key = os.getenv("WEBCITE_API_KEY")
        if not self._api_key:
            raise ProviderConfigError("Webcite API key is required. Set the WEBCITE_API_KEY environment variable.")
        self._base_url = (self.base_config.get("base_url") or os.getenv("WEBCITE_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self._timeout = float(self.base_config.get("timeout", 900))
        self._max_attempts = int(self.base_config.get("max_attempts", 3))
        self._max_rate_limit_waits = int(self.base_config.get("max_rate_limit_waits", 20))
        # A retried document reuses its uploaded asset instead of uploading (and paying for) it again.
        self._assets: dict[str, tuple[str, float | None]] = {}
        self._assets_lock = threading.Lock()

    @staticmethod
    def _retry_after(response: Any) -> float | None:
        """Seconds to wait from a 429's Retry-After header or JSON ``retryAfter``."""
        header = response.headers.get("retry-after")
        try:
            if header is not None:
                return max(0.0, float(header))
            body = response.json()
            value = body.get("retryAfter") if isinstance(body, dict) else None
            return max(0.0, float(value)) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _post(self, client: Any, path: str, **kwargs: Any) -> dict[str, Any]:
        """POST with bounded retries; a 429 waits for the server's Retry-After without spending an attempt."""
        last: Exception | None = None
        attempt = rate_limit_waits = 0
        while attempt < self._max_attempts:
            delay = min(30.0, 2.0 * 2**attempt)
            try:
                response = client.post(path, **kwargs)
            except (self._httpx.TimeoutException, self._httpx.TransportError) as e:
                last = ProviderTransientError(f"Webcite {path} transport error: {e}")
                attempt += 1
            else:
                if response.status_code == 429 and rate_limit_waits < self._max_rate_limit_waits:
                    rate_limit_waits += 1
                    delay = min(60.0, self._retry_after(response) or 5.0)
                    last = ProviderTransientError(f"Webcite {path} HTTP 429: {response.text[:200]}")
                elif response.status_code in _TRANSIENT_STATUS:
                    last = ProviderTransientError(f"Webcite {path} HTTP {response.status_code}: {response.text[:500]}")
                    attempt += 1
                elif response.is_error:
                    raise ProviderPermanentError(f"Webcite {path} HTTP {response.status_code}: {response.text[:500]}")
                else:
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ProviderPermanentError(f"Webcite {path} returned a non-object body")
                    return payload
            if attempt < self._max_attempts:
                time.sleep(delay)
                # A file handle consumed by the failed attempt must be rewound before the retry.
                for _, (_, handle, _) in (kwargs.get("files") or {}).items():
                    handle.seek(0)
        assert last is not None
        raise last

    def _upload(self, client: Any, path: Path, mime: str) -> tuple[str, float | None]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self._assets_lock:
            cached = self._assets.get(digest)
        if cached:
            return cached
        with path.open("rb") as handle:
            uploaded = self._post(client, "/upload", files={"file": (path.name, handle, mime)})
        asset_id = (uploaded.get("data") or {}).get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ProviderPermanentError("Webcite upload returned no asset_id")
        credits = (uploaded.get("usage") or {}).get("credits")
        entry = (asset_id, credits if _is_number(credits) else None)
        with self._assets_lock:
            self._assets[digest] = entry
        return entry

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"Webcite only supports PARSE, got {request.product_type}")
        path = Path(request.source_file_path)
        mime = _MIME_TYPES.get(path.suffix.lower())
        if mime is None:
            raise ProviderPermanentError(f"Unsupported file type for Webcite: {path.suffix}")

        started_at = datetime.now()
        headers = {"x-api-key": self._api_key}
        with self._httpx.Client(base_url=self._base_url, headers=headers, timeout=self._timeout) as client:
            asset_id, upload_credits = self._upload(client, path, mime)
            extracted = self._post(client, "/extract", json={"asset_id": asset_id})
        completed_at = datetime.now()

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={
                "asset_id": asset_id,
                "extract": extracted,
                "num_pages": len(
                    {
                        (u.get("provenance") or {}).get("page")
                        for u in extracted.get("units") or []
                        if isinstance(u, dict) and (u.get("provenance") or {}).get("page")
                    }
                )
                or None,
                "credits_used": _credits(upload_credits, (extracted.get("usage") or {}).get("credits")),
                "_config": {"provider": "webcite", "base_url": self._base_url},
            },
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        data = raw_result.raw_output.get("extract") or {}
        markdown = data.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            raise ProviderPermanentError(
                f"Webcite extraction returned no text: {data.get('state')}; {data.get('reason')}"
            )

        units = [unit for unit in data.get("units") or [] if isinstance(unit, dict)]
        page_texts: dict[int, list[str]] = {}
        for unit in units:
            page = (unit.get("provenance") or {}).get("page")
            text = unit.get("text")
            if isinstance(page, int) and page > 0 and isinstance(text, str):
                page_texts.setdefault(page - 1, []).append(text)
        pages = [
            PageIR(page_index=index, markdown="\n\n".join(texts)) for index, texts in sorted(page_texts.items())
        ] or [PageIR(page_index=0, markdown=markdown)]

        output = ParseOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=_layout_pages(units),
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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _credits(*values: Any) -> float | None:
    """Total credits charged, whole or fractional; None when any call's charge is unknown (never a 0 read as free)."""
    if not all(_is_number(v) for v in values):
        return None
    return sum(values)


_FURNITURE_TYPES = {"page_header", "page_footer"}
_TEXT_TYPES = {"text", "list_item", "list", "section_header", "title", "caption", "footnote"} | _FURNITURE_TYPES
_INLINE_TAG = re.compile(r"</?(?:u|b|i|mark|s|strong|em|ins|del|span|code|strike)\b[^>]*>", re.I)
_INLINE_MATH = re.compile(r"\$([^$]*[\\^_{}][^$]*)\$")
_LATEX_WRAPPER = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|operatorname|textbf|mathcal|overline|bar|hat|tilde|vec|sqrt)\s*\{([^{}]*)\}"
)
_LATEX_FRACTION = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_LATEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "omega": "ω", "Omega": "Ω", "Delta": "Δ", "xi": "ξ",
    "chi": "χ", "Phi": "Φ", "phi": "φ", "mu": "μ", "pi": "π", "sigma": "σ", "lambda": "λ", "theta": "θ", "rho": "ρ",
    "tau": "τ", "epsilon": "ε", "div": "÷", "times": "×", "cdot": "·", "pm": "±", "leq": "≤", "geq": "≥", "neq": "≠",
    "approx": "≈", "oplus": "⊕", "sum": "∑", "infty": "∞",
}  # fmt: skip


def _plain_math(latex: str) -> str:
    """LaTeX as the characters it prints: wrappers and braces removed, fractions as a/b, named symbols as Unicode."""
    text = latex.replace("$$", " ").replace("\\$", "$").replace("\\%", "%").replace("\\&", "&")
    text = text.replace("\\\\", " ").replace("&", " ")
    for _ in range(4):
        text = _LATEX_FRACTION.sub(r"\1/\2", _LATEX_WRAPPER.sub(r"\1", text))
    text = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\([a-zA-Z]+)", lambda m: _LATEX_SYMBOLS.get(m.group(1), " "), text)
    text = re.sub(r"[_^]", "", re.sub(r"[_^]\s*\{([^{}]*)\}", r"\1", text))
    return re.sub(r"\s+", " ", text.replace("{", "").replace("}", "").replace("$", " $")).strip()


def _layout_text(kind: Any, markdown: str) -> str:
    """Element text as sent for layout matching, which compares printed words.

    Inline styling tags are removed without a space (``Report<u>ing</u>`` stays one word) and HTML entities are
    decoded; formulas and inline ``$..$`` maths are written as plain characters. Tables and pictures are untouched.
    The element Markdown in ``raw_output`` and the page Markdown are not changed.
    """
    text = markdown
    if kind not in ("table", "picture") and ("<" in text or "&" in text) and "<table" not in text:
        text = html.unescape(_INLINE_TAG.sub("", text)).replace("\u00a0", " ")
    if kind == "formula" and text.strip():
        return _plain_math(text)
    if kind in _TEXT_TYPES:
        return _INLINE_MATH.sub(lambda m: _plain_math(m.group(1)), text)
    return text


def _layout_pages(units: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Element boxes from the pages that carry them. Pages without valid boxes are omitted, never invented."""
    pages: list[ParseLayoutPageIR] = []
    for unit in units:
        elements = unit.get("elements")
        page = (unit.get("provenance") or {}).get("page")
        geometry = unit.get("geometry") or {}
        if not elements or not isinstance(page, int) or page < 1:
            continue
        if geometry.get("coordinateSystem") not in (None, "normalized-top-left-v1"):
            continue
        items: list[LayoutItemIR] = []
        for element in elements:
            box = element.get("bbox") if isinstance(element, dict) else None
            label = WEBCITE_LABEL_MAP.get(element.get("type")) if isinstance(element, dict) else None
            if label is None or not isinstance(box, list) or len(box) != 4:
                continue
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in box):
                continue
            left, top, right, bottom = box
            if left >= right or top >= bottom:
                continue
            markdown = element.get("markdown")
            value = (
                _layout_text(element.get("type"), markdown)
                if isinstance(markdown, str) and markdown
                else element.get("description") or ""
            )
            kind = "table" if label == "Table" else "image" if label == "Picture" else "text"
            segment = LayoutSegmentIR(x=left, y=top, w=right - left, h=bottom - top, label=label)
            items.append(
                LayoutItemIR(
                    type=kind,
                    value=value,
                    md=value,
                    html=value if kind == "table" else "",
                    bbox=segment,
                    layout_segments=[segment],
                )
            )
        if items:
            furniture = {
                f"page_{part}_markdown": "\n\n".join(
                    e.get("markdown", "") for e in elements if isinstance(e, dict) and e.get("type") == f"page_{part}"
                )
                for part in ("header", "footer")
            }
            pages.append(
                ParseLayoutPageIR(
                    page_number=page,
                    width=_VIRTUAL_PAGE_DIM,
                    height=_VIRTUAL_PAGE_DIM,
                    md=unit.get("text") or "",
                    items=items,
                    **furniture,
                )
            )
    return pages
