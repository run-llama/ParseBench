"""Provider for Pulse PARSE.

Uses the Pulse Python SDK (pulse-python-sdk) to convert documents into
markdown with optional HTML table output via the /extract endpoint.
"""

import os
import time
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

# Polling settings for async jobs.
_POLL_INTERVAL = 2.0
_MAX_POLL_ATTEMPTS = 150  # 5 minutes at 2s interval

# Virtual page dimension for normalized [0,1] → pixel coordinate conversion.
# Evaluation divides pixel coords by page dimensions, so these cancel out.
_VIRTUAL_PAGE_DIM = 1000.0

# Map Pulse bounding_boxes label keys to canonical layout labels.
_PULSE_LABEL_MAP: dict[str, str] = {
    "Title": "Title",
    "Text": "Text",
    "Header": "Page-header",
    "Footer": "Page-footer",
    "Page Number": "Page-footer",
    "Images": "Picture",
    "Tables": "Table",
}


@register_provider("pulse")
class PulseProvider(Provider):
    """Provider for Pulse document extraction.

    Uses the Pulse SDK to call the /extract endpoint for document parsing.
    """

    CREDIT_RATE_USD = 0.015  # PAYGO rate: $0.015 per credit

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        api_key = self.base_config.get("api_key") or os.getenv("PULSE_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ProviderConfigError(
                "Pulse API key is required. Set PULSE_API_KEY environment variable or pass api_key in base_config."
            )
        self._api_key: str = api_key

        self._return_html: bool = self.base_config.get("return_html", False)
        self._return_xml: bool = self.base_config.get("return_xml", False)
        self._word_level_bboxes: bool = self.base_config.get(
            "word_level_bboxes", self.base_config.get("wlbb", False)
        )
        self._model: str | None = self.base_config.get("model")
        self._figure_description: bool = self.base_config.get("figure_description", False)
        self._use_async: bool = self.base_config.get("use_async", False)
        self._pages: str | None = self.base_config.get("pages", None)
        self._timeout: float = float(self.base_config.get("timeout", 300))

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _extract(self, file_path: str) -> dict[str, Any]:
        """Extract a document using the Pulse SDK."""
        try:
            from pulse import Pulse
        except ImportError as e:
            raise ProviderConfigError(
                "pulse-python-sdk package not installed. Run: pip install pulse-python-sdk"
            ) from e

        client = Pulse(api_key=self._api_key, timeout=self._timeout)

        # Build optional kwargs
        kwargs: dict[str, Any] = {}
        if self._model:
            kwargs["model"] = self._model
        if self._pages:
            kwargs["pages"] = self._pages
        if self._use_async:
            kwargs["async_"] = True

        # Figure processing
        if self._figure_description:
            from pulse.types import ExtractRequestFigureProcessing

            kwargs["figure_processing"] = ExtractRequestFigureProcessing(description=True)

        # The SDK currently exposes the documented extensions object, but
        # serializes nested multipart fields as a dict. Use the first-class
        # primitive argument for HTML so table eval gets HTML reliably.
        if self._return_html:
            kwargs["return_html"] = True

        if self._return_xml or self._word_level_bboxes:
            from pulse.types import (
                ExtractRequestExtensions,
                ExtractRequestExtensionsAltOutputs,
            )

            kwargs["extensions"] = ExtractRequestExtensions(
                alt_outputs=ExtractRequestExtensionsAltOutputs(
                    wlbb=self._word_level_bboxes or None,
                    return_xml=self._return_xml or None,
                )
            )

        with open(file_path, "rb") as f:
            response = client.extract(file=f, **kwargs)

        # If async mode, poll for completion
        if self._use_async and hasattr(response, "job_id") and response.job_id:
            response = self._poll_job(client, response.job_id)

        # Serialize response to dict
        if hasattr(response, "model_dump"):
            raw: dict[str, Any] = response.model_dump()
        elif hasattr(response, "dict"):
            raw = response.dict()
        elif isinstance(response, dict):
            raw = response
        else:
            raw = {"markdown": getattr(response, "markdown", ""), "raw": str(response)}

        # For large docs (70+ pages), Pulse returns a URL — fetch it
        if raw.get("is_url") and raw.get("url"):
            import requests

            url_resp = requests.get(raw["url"], timeout=300)
            if url_resp.status_code == 200:
                url_result = url_resp.json()
                if "plan_info" in raw or "plan-info" in raw:
                    url_result["plan_info"] = raw.get("plan_info", raw.get("plan-info"))
                raw = url_result

        return raw

    @staticmethod
    def _poll_job(client: Any, job_id: str) -> Any:
        """Poll for async job completion using the SDK."""
        for _ in range(_MAX_POLL_ATTEMPTS):
            job_status = client.jobs.get_job(job_id=job_id)
            status = getattr(job_status, "status", "").lower()
            if status == "completed":
                return getattr(job_status, "result", job_status)
            if status in ("failed", "canceled"):
                raise ProviderPermanentError(f"Pulse job {job_id} {status}: {getattr(job_status, 'message', '')}")
            time.sleep(_POLL_INTERVAL)
        raise ProviderTransientError(f"Pulse job {job_id} timed out after {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL}s")

    # --------------------------------------------------------------------- #
    # Provider interface
    # --------------------------------------------------------------------- #

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"PulseProvider only supports PARSE product type, got {request.product_type}")

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()

        try:
            raw_output = self._extract(str(file_path))

            # Store config for reproducibility
            raw_output["_config"] = {
                "return_html": self._return_html,
                "return_xml": self._return_xml,
                "word_level_bboxes": self._word_level_bboxes,
                "model": self._model,
                "figure_description": self._figure_description,
                "use_async": self._use_async,
                "pages": self._pages,
            }

            # Cost tracking — API returns credits_used at top level and
            # page info under "plan-info" (hyphen) or "plan_info" (underscore).
            plan_info = raw_output.get("plan-info", raw_output.get("plan_info", {}))
            if isinstance(plan_info, dict):
                pages_used = plan_info.get("pages_used", raw_output.get("page_count"))
                if pages_used and pages_used > 0:
                    raw_output["num_pages"] = pages_used

            credits = raw_output.get("credits_used")
            if credits is not None and credits > 0:
                cost_usd = credits * self.CREDIT_RATE_USD
                raw_output["cost_usd"] = cost_usd
                num_pages = raw_output.get("num_pages", raw_output.get("page_count", 0))
                if num_pages and num_pages > 0:
                    raw_output["cost_per_page_usd"] = cost_usd / num_pages

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

        except (
            ProviderPermanentError,
            ProviderTransientError,
            ProviderConfigError,
            ProviderRateLimitError,
        ):
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise ProviderRateLimitError(f"Pulse rate limit: {e}") from e
            if any(kw in error_str for kw in ["timeout", "timed out", "network", "connection", "503", "502"]):
                raise ProviderTransientError(f"Transient error: {e}") from e
            if "401" in error_str or "unauthorized" in error_str:
                raise ProviderConfigError(f"Pulse auth failed: {e}") from e
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"PulseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        raw = raw_result.raw_output

        # Prefer HTML output if available (better for table evaluation)
        html_content = _get_pulse_html(raw)
        markdown = html_content or raw.get("markdown", "")

        # Build layout pages from bounding_boxes for Visual Grounding evaluation
        layout_pages = _build_layout_pages(raw.get("bounding_boxes", {}))

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=[],
            layout_pages=layout_pages,
            markdown=markdown,
            job_id=raw.get("extraction_id"),
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


def _polygon_to_xywh(coords: list[float]) -> tuple[float, float, float, float]:
    """Convert an 8-float polygon [x1,y1, x2,y2, x3,y3, x4,y4] to (x, y, w, h)."""
    xs = [coords[i] for i in range(0, 8, 2)]
    ys = [coords[i] for i in range(1, 8, 2)]
    x = min(xs)
    y = min(ys)
    return x, y, max(xs) - x, max(ys) - y


def _get_pulse_html(raw: dict[str, Any]) -> str:
    """Return Pulse HTML alt output across SDK/API response spellings."""

    extensions = raw.get("extensions")
    if isinstance(extensions, dict):
        for key in ("alt_outputs", "altOutputs"):
            alt_outputs = extensions.get(key)
            if isinstance(alt_outputs, dict):
                html = alt_outputs.get("html")
                if isinstance(html, str) and html:
                    return html

    # Legacy/deprecated response shape when the old return_html field is used.
    html = raw.get("html")
    if isinstance(html, str) and html:
        return html

    return ""


def _build_layout_pages(bounding_boxes: dict[str, Any]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from Pulse bounding_boxes for layout cross-evaluation.

    Pulse returns bounding_boxes grouped by label type (Title, Text, Header,
    Footer, Images, Tables) with normalized [0,1] coordinates as 8-point
    polygons.
    """
    from collections import defaultdict

    pages_items: dict[int, list[LayoutItemIR]] = defaultdict(list)

    for label_key, canonical_label in _PULSE_LABEL_MAP.items():
        elements = bounding_boxes.get(label_key, [])
        if not isinstance(elements, list):
            continue

        for elem in elements:
            # Tables have a different structure
            if label_key == "Tables":
                table_info = elem.get("table_info", {})
                location = table_info.get("location", {})
                coords = location.get("coordinates", [])
                page_num = location.get("page", 1)
                conf_raw = table_info.get("confidence")
                # Reconstruct table text from cell_data
                cell_texts = []
                for cell in elem.get("cell_data", []):
                    text = cell.get("text", "")
                    # Strip the "0t-" prefix Pulse adds
                    if text.startswith("0t-"):
                        text = text[3:]
                    cell_texts.append(text)
                content = " ".join(cell_texts)
            else:
                coords = elem.get("bounding_box", [])
                page_num = elem.get("page_number", 1)
                conf_raw = elem.get("average_word_confidence", elem.get("confidence"))
                content = elem.get("original_content", elem.get("content", ""))

            if not coords or len(coords) < 8:
                continue

            try:
                confidence = float(conf_raw) if conf_raw is not None and conf_raw != "N/A" else 1.0
            except (TypeError, ValueError):
                confidence = 1.0

            x, y, w, h = _polygon_to_xywh(coords)
            seg = LayoutSegmentIR(x=x, y=y, w=w, h=h, confidence=confidence, label=canonical_label)

            norm_label = canonical_label.strip().lower()
            if norm_label == "table":
                item_type = "table"
            elif norm_label == "picture":
                item_type = "image"
            else:
                item_type = "text"

            pages_items[page_num].append(LayoutItemIR(type=item_type, value=content, bbox=seg, layout_segments=[seg]))

    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_items.keys()):
        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_num,
                width=_VIRTUAL_PAGE_DIM,
                height=_VIRTUAL_PAGE_DIM,
                items=pages_items[page_num],
            )
        )
    return layout_pages
