"""Provider for the Upstage Document Parse API."""

import io
import json
import math
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

from .upstage_normalization import element_html, element_text, normalize_element, normalize_page, strip_html_metadata

_DEFAULT_ENDPOINT = "https://api.upstage.ai/v1/document-digitization"


def _layout_segment(element: dict[str, Any]) -> LayoutSegmentIR | None:
    coordinates = element.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 4:
        return None
    try:
        xs = [float(point["x"]) for point in coordinates]
        ys = [float(point["y"]) for point in coordinates]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in xs + ys):
        return None
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    if right <= left or bottom <= top:
        return None
    return LayoutSegmentIR(
        x=left,
        y=top,
        w=right - left,
        h=bottom - top,
        label=str(element.get("category") or "paragraph"),
        confidence=1.0,
    )


@register_provider("upstage")
class UpstageDocumentParseProvider(Provider):
    """Call Document Parse and map its response to ParseBench output."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        try:
            import httpx
        except ImportError as exc:
            raise ProviderConfigError("Install parse-bench[upstage] to use Upstage") from exc

        self._httpx = httpx
        self._api_key = os.getenv("UPSTAGE_API_KEY")
        if not self._api_key:
            raise ProviderConfigError("Set UPSTAGE_API_KEY to use Upstage")
        self._endpoint = self.base_config.get("endpoint") or os.getenv("UPSTAGE_BASE_URL") or _DEFAULT_ENDPOINT
        self._timeout = float(self.base_config.get("timeout", 600))
        self._cost_per_page_usd = float(self.base_config.get("cost_per_page_usd", 0.03))

    def _form_data(self) -> dict[str, str]:
        return {
            "model": str(self.base_config.get("model", "document-parse-260930")),
            "mode": str(self.base_config.get("mode", "enhanced")),
            "ocr": str(self.base_config.get("ocr", "force")),
            "output_formats": json.dumps(["html", "markdown", "text"]),
            "coordinates": "true",
            "chart_recognition": json.dumps(self.base_config.get("chart_recognition", True)),
        }

    def _upload_payload(self, path: Path) -> tuple[str, bytes, str]:
        """Keep PDFs intact; optionally upscale native images to a minimum edge."""
        minimum = int(self.base_config.get("min_image_short_side", 0))
        if minimum > 0 and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            from PIL import Image

            with Image.open(path) as source:
                if min(source.size) < minimum:
                    scale = minimum / min(source.size)
                    image = source.convert("RGB").resize(
                        (round(source.width * scale), round(source.height * scale)),
                        Image.Resampling.LANCZOS,
                    )
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    return f"{path.stem}.png", buffer.getvalue(), "image/png"
        return path.name, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"Upstage only supports PARSE, got {request.product_type}")

        path = Path(request.source_file_path)
        if not path.is_file():
            raise ProviderPermanentError(f"File not found: {path}")

        started_at = datetime.now()
        fields = self._form_data()
        upload_name, upload_bytes, upload_mime = self._upload_payload(path)
        try:
            with self._httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "X-Upstage-Use-Cache": "false",
                    },
                    data=fields,
                    files={
                        "document": (
                            upload_name,
                            upload_bytes,
                            upload_mime,
                        )
                    },
                )
        except self._httpx.HTTPError as exc:
            raise ProviderTransientError(f"Upstage transport error: {type(exc).__name__}") from exc

        status = response.status_code
        if status in {401, 403}:
            raise ProviderConfigError(f"Upstage authentication/access error ({status})")
        if status == 429:
            raise ProviderRateLimitError("Upstage rate limit (429)")
        if status == 408 or status >= 500:
            raise ProviderTransientError(f"Upstage server error ({status})")
        if status >= 400:
            raise ProviderPermanentError(f"Upstage request error ({status})")

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderTransientError("Upstage returned invalid JSON") from exc
        if not isinstance(body, dict) or not isinstance(body.get("elements"), list):
            raise ProviderPermanentError("Upstage response must contain an elements list")

        num_pages = (body.get("usage") or {}).get("pages") or 1
        body["num_pages"] = num_pages
        body["cost_per_page_usd"] = self._cost_per_page_usd
        body["cost_usd"] = float(num_pages) * self._cost_per_page_usd
        body["_request_config"] = fields

        completed_at = datetime.now()
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
            raise ProviderPermanentError(f"Upstage only supports PARSE, got {raw_result.product_type}")

        body = raw_result.raw_output
        elements_by_page: dict[int, list[dict[str, Any]]] = {}
        for element in body.get("elements") or []:
            page_number = int(element.get("page", 1))
            if page_number < 1:
                raise ProviderPermanentError("Upstage page numbers must be 1-based")
            elements_by_page.setdefault(page_number, []).append(element)
        for page_number in range(1, int((body.get("usage") or {}).get("pages") or 0) + 1):
            elements_by_page.setdefault(page_number, [])

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        for page_number, elements in sorted(elements_by_page.items()):
            page_markup = normalize_page(elements)
            items: list[LayoutItemIR] = []
            for element in elements:
                rendered = normalize_element(element)
                source_html = strip_html_metadata(element_html(element))
                source_text = element_text(element)
                if str(element.get("category") or "").lower() in {"chart", "figure"}:
                    source_html = rendered
                    source_text = BeautifulSoup(rendered, "html.parser").get_text(" ", strip=True)
                segment = _layout_segment(element)
                items.append(
                    LayoutItemIR(
                        type=str(element.get("category") or "paragraph"),
                        md=rendered,
                        html=source_html,
                        value=source_text,
                        bbox=segment,
                        layout_segments=[segment] if segment else [],
                    )
                )
            pages.append(PageIR(page_index=page_number - 1, markdown=page_markup))
            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=page_number,
                    width=1000,
                    height=1000,
                    md=page_markup,
                    text=page_markup,
                    items=items,
                )
            )

        document_html = "\n\n".join(page.markdown for page in pages if page.markdown)
        output = ParseOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=document_html,
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=body,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
