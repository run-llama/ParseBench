"""Provider for OCR.space hosted OCR API PARSE.

OCR.space (https://ocr.space) is a hosted OCR API with a generous free tier:
25,000 requests/month with no credit card required. The file is POSTed to
``https://api.ocr.space/parse/image`` and the response carries page text plus
optional word-level overlays (``isOverlayRequired=true``), which become the
per-page markdown. Word boxes are kept in the raw output for debugging.

Config keys
-----------
api_key: str
    OCR.space API key. Falls back to ``OCRSPACE_API_KEY``, else the public
    ``helloworld`` demo key (strictly rate/size limited -- fine for smoke
    tests; get a free key at https://ocr.space/ocrapi for real runs).
base_url: str
    API origin. Falls back to ``OCRSPACE_BASE_URL``, else
    ``https://api.ocr.space``.
language: str
    OCR language code, e.g. ``eng``, ``chs``, ``jpn``. Default ``eng``.
    Full list: https://ocr.space/ocrapi#language
ocr_engine: int
    ``1``, ``2``, ``3`` or ``5``. Default ``2`` (recommended general engine).
scale: bool
    Upscale the image before OCR (helps small text). Default ``False``.
is_table: bool
    Optimize for table OCR. Default ``False``.
detect_orientation: bool
    Auto-detect text orientation. Default ``False``.
timeout: float
    Request timeout in seconds. Default ``120``.
max_file_size_mb: float
    Guard matching the free plan's 1 MB upload cap. Raise for paid keys.

Cost: the free tier costs nothing, so ``cost_usd`` is ``0.0``.
"""

from __future__ import annotations

import os
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
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

_DEFAULT_BASE_URL = "https://api.ocr.space"
_DEMO_API_KEY = "helloworld"
_SUPPORTED_ENGINES = (1, 2, 3, 5)
_SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}


def _is_rate_limit_message(message: str) -> bool:
    lowered = message.lower()
    return any(kw in lowered for kw in ("rate limit", "too many requests", "quota exceeded", "limit exceeded"))


@register_provider("ocrspace")
class OCRSpaceProvider(Provider):
    """OCR.space hosted OCR API document parsing."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._api_key = self.base_config.get("api_key") or os.environ.get("OCRSPACE_API_KEY") or _DEMO_API_KEY
        self._base_url = self.base_config.get("base_url") or os.environ.get("OCRSPACE_BASE_URL", _DEFAULT_BASE_URL)
        self._language = self.base_config.get("language", "eng")
        self._ocr_engine = int(self.base_config.get("ocr_engine", 2))
        if self._ocr_engine not in _SUPPORTED_ENGINES:
            raise ProviderConfigError(f"Invalid ocr_engine '{self._ocr_engine}'. Must be one of {_SUPPORTED_ENGINES}.")
        self._scale = bool(self.base_config.get("scale", False))
        self._is_table = bool(self.base_config.get("is_table", False))
        self._detect_orientation = bool(self.base_config.get("detect_orientation", False))
        self._timeout = float(self.base_config.get("timeout", 120))
        self._max_file_size_mb = float(self.base_config.get("max_file_size_mb", 1.0))

    def _parse_file(self, file_path: Path) -> dict[str, Any]:
        """POST the file to OCR.space and return the decoded JSON response."""
        try:
            import requests
        except ImportError as e:
            raise ProviderConfigError("requests package not installed. Run: pip install requests") from e

        url = self._base_url.rstrip("/") + "/parse/image"
        data = {
            "language": self._language,
            "isOverlayRequired": "true",
            "OCREngine": str(self._ocr_engine),
            "scale": "true" if self._scale else "false",
            "isTable": "true" if self._is_table else "false",
            "detectOrientation": "true" if self._detect_orientation else "false",
        }
        try:
            with open(file_path, "rb") as fh:
                response = requests.post(
                    url,
                    headers={"apikey": self._api_key},
                    files={"file": (file_path.name, fh, "application/octet-stream")},
                    data=data,
                    timeout=self._timeout,
                )
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ("timeout", "timed out")):
                raise ProviderTransientError(f"OCR.space request timed out: {e}") from e
            if any(kw in err for kw in ("connection", "dns", "resolve", "network", "unreachable")):
                raise ProviderTransientError(f"OCR.space connection error: {e}") from e
            raise ProviderTransientError(f"OCR.space request failed: {e}") from e

        if response.status_code == 429:
            raise ProviderRateLimitError(f"OCR.space rate limited (HTTP 429): {response.text[:200]}")
        if 500 <= response.status_code < 600:
            raise ProviderTransientError(f"OCR.space server error (HTTP {response.status_code}): {response.text[:200]}")
        if response.status_code != 200:
            raise ProviderPermanentError(f"OCR.space returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            payload = response.json()
        except Exception as e:
            raise ProviderPermanentError(f"OCR.space returned non-JSON response: {e}") from e
        if not isinstance(payload, dict):
            raise ProviderPermanentError(f"OCR.space returned unexpected JSON: {payload!r}"[:200])
        return payload

    @staticmethod
    def _check_api_error(payload: dict[str, Any]) -> None:
        """Raise for API-level failures reported inside a 200 response."""
        message = str(payload.get("ErrorMessage") or "")
        if message:
            if _is_rate_limit_message(message):
                raise ProviderRateLimitError(f"OCR.space API error: {message}")
            raise ProviderPermanentError(f"OCR.space API error: {message}")
        if str(payload.get("OCRExitCode")) not in ("1",):
            # OCRExitCode 1 = success; anything else without an ErrorMessage is
            # still a failed parse.
            raise ProviderPermanentError(f"OCR.space parse failed (OCRExitCode={payload.get('OCRExitCode')})")

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"OCRSpaceProvider only supports PARSE product type, got {request.product_type}"
            )

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")
        if source_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ProviderPermanentError(
                f"OCRSpaceProvider only supports {_SUPPORTED_EXTENSIONS}, got {source_path.suffix}"
            )
        size_mb = source_path.stat().st_size / (1024 * 1024)
        if size_mb > self._max_file_size_mb:
            raise ProviderPermanentError(
                f"File is {size_mb:.2f} MB, over the {self._max_file_size_mb} MB cap "
                f"(free OCR.space plan limit is 1 MB; raise max_file_size_mb for paid keys)."
            )

        started_at = datetime.now()
        try:
            payload = self._parse_file(source_path)
            self._check_api_error(payload)

            pages: list[dict[str, Any]] = []
            for page_index, result in enumerate(payload.get("ParsedResults") or []):
                text = result.get("ParsedText", "") or ""
                words: list[dict[str, Any]] = []
                for line in (result.get("TextOverlay") or {}).get("Lines", []) or []:
                    for word in line.get("Words", []) or []:
                        words.append(
                            {
                                "text": word.get("WordText", ""),
                                "left": word.get("Left"),
                                "top": word.get("Top"),
                                "width": word.get("Width"),
                                "height": word.get("Height"),
                            }
                        )
                pages.append({"page_index": page_index, "text": text, "words": words})

            raw_output: dict[str, Any] = {
                "provider": "ocrspace",
                "pages": pages,
                "num_pages": len(pages),
                "cost_usd": 0.0,
                "config": {
                    "language": self._language,
                    "ocr_engine": self._ocr_engine,
                    "scale": self._scale,
                    "is_table": self._is_table,
                },
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
        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError, ProviderRateLimitError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """Normalize raw inference result to produce ParseOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"OCRSpaceProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        pages: list[PageIR] = []
        page_texts = []
        for page_data in raw_result.raw_output.get("pages", []):
            text = page_data.get("text", "")
            pages.append(PageIR(page_index=page_data.get("page_index", 0), markdown=text))
            page_texts.append(text)

        full_text = "\n\n".join(page_texts)
        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown=full_text,
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


def cost_fields() -> dict[str, float]:
    """Cost model: the free tier costs nothing."""
    return {"cost_usd": 0.0}
