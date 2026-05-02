"""Provider for Pulse PARSE.

Calls the Pulse /extract REST endpoint directly via multipart/form-data.
Exposes the full set of documented controls (model, refine, refine_options,
extract_figure, figure_description, custom_image_prompt, custom_refine_prompt,
additional_prompt) so that pipelines can be configured to match published
runs.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

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

_API_URL = "https://api.runpulse.com/extract"

# Virtual page dimension for normalized [0,1] -> pixel coordinate conversion.
# Evaluation divides pixel coords by page dimensions, so these cancel out.
_VIRTUAL_PAGE_DIM = 1000.0

# Map Pulse bounding_boxes label keys to canonical layout labels. "Header" is
# disambiguated into Page-header vs Section-header by Y-position in
# _build_layout_pages because Pulse lumps both into the same bucket.
_PULSE_LABEL_MAP: dict[str, str] = {
    "Title": "Title",
    "Text": "Text",
    "Header": "Page-header",
    "Footer": "Page-footer",
    "Page Number": "Page-footer",
    "Images": "Picture",
    "Tables": "Table",
    "caption": "Caption",
}

_PAGE_HEADER_TOP_BAND = 0.10
_PAGE_HEADER_BOTTOM_BAND = 0.90


@register_provider("pulse")
class PulseProvider(Provider):
    """Provider for Pulse document extraction via REST."""

    CREDIT_RATE_USD = 0.015  # PAYGO rate: $0.015 per credit

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        api_key = self.base_config.get("api_key") or os.getenv("PULSE_API_KEY")
        if not api_key or not isinstance(api_key, str):
            raise ProviderConfigError(
                "Pulse API key is required. Set PULSE_API_KEY environment variable or pass api_key in base_config."
            )
        self._api_key: str = api_key

        # Core controls
        self._model: str | None = self.base_config.get("model")

        # Refinement
        self._refine: bool = bool(self.base_config.get("refine", False))
        refine_options = self.base_config.get("refine_options")
        if refine_options is not None and not isinstance(refine_options, dict):
            raise ProviderConfigError("refine_options must be a dict")
        self._refine_options: dict[str, bool] | None = refine_options

        # Figures / charts
        self._extract_figure: bool = bool(self.base_config.get("extract_figure", False))
        self._figure_description: bool = bool(self.base_config.get("figure_description", False))

        # Prompt overrides
        self._additional_prompt: str | None = self.base_config.get("additional_prompt")
        self._custom_image_prompt: str | None = self.base_config.get("custom_image_prompt")
        self._custom_refine_prompt: str | None = self.base_config.get("custom_refine_prompt")

        # Misc
        self._pages: str | None = self.base_config.get("pages")
        self._timeout: float = float(self.base_config.get("timeout", 600))

    # --------------------------------------------------------------------- #
    # HTTP call
    # --------------------------------------------------------------------- #

    def _build_form_fields(self) -> list[tuple[str, tuple[None, str]]]:
        """Build the non-file multipart fields for the /extract POST."""
        fields: list[tuple[str, tuple[None, str]]] = []

        def add(name: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, bool):
                fields.append((name, (None, "true" if value else "false")))
            elif isinstance(value, (dict, list)):
                fields.append((name, (None, json.dumps(value))))
            else:
                fields.append((name, (None, str(value))))

        add("model", self._model)
        add("refine", self._refine or None)
        add("refine_options", self._refine_options)
        add("extract_figure", self._extract_figure or None)
        add("figure_description", self._figure_description or None)
        add("additional_prompt", self._additional_prompt)
        add("custom_image_prompt", self._custom_image_prompt)
        add("custom_refine_prompt", self._custom_refine_prompt)
        add("pages", self._pages)

        return fields

    def _extract(self, file_path: str) -> dict[str, Any]:
        headers = {"x-api-key": self._api_key}
        form_fields = self._build_form_fields()

        with open(file_path, "rb") as f:
            files: list[tuple[str, Any]] = [("file", (Path(file_path).name, f, "application/pdf"))]
            files.extend(form_fields)

            response = requests.post(_API_URL, headers=headers, files=files, timeout=self._timeout)

        if response.status_code == 401:
            raise ProviderConfigError(f"Pulse auth failed (401): {response.text[:300]}")
        if response.status_code == 429:
            raise ProviderRateLimitError(f"Pulse rate limit (429): {response.text[:300]}")
        if response.status_code in (502, 503, 504):
            raise ProviderTransientError(f"Pulse transient {response.status_code}: {response.text[:300]}")
        if response.status_code >= 400:
            raise ProviderPermanentError(f"Pulse {response.status_code}: {response.text[:300]}")

        try:
            raw: dict[str, Any] = response.json()
        except ValueError as e:
            raise ProviderPermanentError(f"Pulse returned non-JSON response: {e}") from e

        # For large docs Pulse returns a URL pointer to the full result.
        if raw.get("is_url") and raw.get("url"):
            url_resp = requests.get(raw["url"], timeout=self._timeout)
            if url_resp.status_code != 200:
                raise ProviderTransientError(
                    f"Failed to fetch result URL ({url_resp.status_code}): {url_resp.text[:300]}"
                )
            url_result = url_resp.json()
            if "plan_info" in raw or "plan-info" in raw:
                url_result["plan_info"] = raw.get("plan_info", raw.get("plan-info"))
            raw = url_result

        return raw

    # --------------------------------------------------------------------- #
    # Provider interface
    # --------------------------------------------------------------------- #

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"PulseProvider only supports PARSE product type, got {request.product_type}"
            )

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()

        try:
            raw_output = self._extract(str(file_path))
        except (
            ProviderPermanentError,
            ProviderTransientError,
            ProviderConfigError,
            ProviderRateLimitError,
        ):
            raise
        except requests.Timeout as e:
            raise ProviderTransientError(f"Pulse request timed out: {e}") from e
        except requests.ConnectionError as e:
            raise ProviderTransientError(f"Pulse connection error: {e}") from e
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

        raw_output["_config"] = {
            "model": self._model,
            "refine": self._refine,
            "refine_options": self._refine_options,
            "extract_figure": self._extract_figure,
            "figure_description": self._figure_description,
            "custom_image_prompt": self._custom_image_prompt,
            "custom_refine_prompt": self._custom_refine_prompt,
            "additional_prompt": self._additional_prompt,
            "pages": self._pages,
        }

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

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"PulseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        raw = raw_result.raw_output
        html_content = _get_pulse_html(raw)
        markdown = html_content or raw.get("markdown", "")
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


# ------------------------------------------------------------------------- #
# Output normalization helpers
# ------------------------------------------------------------------------- #


def _polygon_to_xywh(coords: list[float]) -> tuple[float, float, float, float]:
    """Convert an 8-float polygon [x1,y1, x2,y2, x3,y3, x4,y4] to (x, y, w, h)."""
    xs = [coords[i] for i in range(0, 8, 2)]
    ys = [coords[i] for i in range(1, 8, 2)]
    x = min(xs)
    y = min(ys)
    return x, y, max(xs) - x, max(ys) - y


def _get_pulse_html(raw: dict[str, Any]) -> str:
    extensions = raw.get("extensions")
    if isinstance(extensions, dict):
        for key in ("alt_outputs", "altOutputs"):
            alt_outputs = extensions.get(key)
            if isinstance(alt_outputs, dict):
                html = alt_outputs.get("html")
                if isinstance(html, str) and html:
                    return html

    html = raw.get("html")
    if isinstance(html, str) and html:
        return html

    return ""


def _build_layout_pages(bounding_boxes: dict[str, Any]) -> list[ParseLayoutPageIR]:
    pages_items: dict[int, list[LayoutItemIR]] = defaultdict(list)

    for label_key, canonical_label in _PULSE_LABEL_MAP.items():
        elements = bounding_boxes.get(label_key, [])
        if not isinstance(elements, list):
            continue

        for elem in elements:
            if label_key == "Tables":
                table_info = elem.get("table_info", {})
                location = table_info.get("location", {})
                coords = location.get("coordinates", [])
                page_num = location.get("page", 1)
                conf_raw = table_info.get("confidence")
                cell_texts = []
                for cell in elem.get("cell_data", []):
                    text = cell.get("text", "")
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

            elem_label = canonical_label
            if label_key == "Header" and _PAGE_HEADER_TOP_BAND <= y <= _PAGE_HEADER_BOTTOM_BAND:
                elem_label = "Section-header"

            seg = LayoutSegmentIR(x=x, y=y, w=w, h=h, confidence=confidence, label=elem_label)

            norm_label = elem_label.strip().lower()
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
