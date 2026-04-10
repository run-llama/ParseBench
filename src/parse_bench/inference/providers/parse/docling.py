"""Provider for Docling parse via a custom HTTP endpoint."""

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from docling_core.types.doc.document import DoclingDocument

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.layout_label_mapping import (
    UnknownRawLayoutLabelError,
    map_docling_raw_label_to_canonical,
)
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

_DOCLING_EXCLUDED_LAYOUT_LABELS = frozenset(
    {
        "empty_value",
        "field_heading",
        "field_hint",
        "field_item",
        "field_key",
        "field_region",
        "field_value",
        "marker",
    }
)
_DOCLING_TABLE_LABELS = frozenset({"document_index", "table"})
_DOCLING_IMAGE_LABELS = frozenset({"chart", "picture"})


def _normalize_docling_label(label: object) -> str | None:
    if label is None:
        return None
    value = getattr(label, "value", label)
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def _should_include_docling_label(raw_label: str) -> bool:
    if raw_label in _DOCLING_EXCLUDED_LAYOUT_LABELS:
        return False
    try:
        map_docling_raw_label_to_canonical(raw_label)
    except UnknownRawLayoutLabelError:
        return False
    return True


def _docling_item_type(raw_label: str) -> str:
    if raw_label in _DOCLING_TABLE_LABELS:
        return "table"
    if raw_label in _DOCLING_IMAGE_LABELS:
        return "image"
    return "text"


def _extract_docling_item_value(item: Any, doc: DoclingDocument, raw_label: str) -> str:
    item_type = _docling_item_type(raw_label)
    if item_type == "image":
        return ""

    if item_type == "table" and hasattr(item, "export_to_html"):
        try:
            html = item.export_to_html(doc=doc, add_caption=True)
            if isinstance(html, str):
                return html
        except Exception:
            pass

    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text

    if hasattr(item, "export_to_markdown"):
        try:
            markdown = item.export_to_markdown()
            if isinstance(markdown, str):
                return markdown
        except Exception:
            pass

    return ""


def _normalize_docling_charspan(
    charspan: object,
    *,
    text_length: int,
    include_span: bool,
) -> tuple[int | None, int | None]:
    if not include_span or not isinstance(charspan, (list, tuple)) or len(charspan) != 2:
        return (None, None)

    start_raw, end_raw = charspan
    if not isinstance(start_raw, int) or not isinstance(end_raw, int):
        return (None, None)

    start = max(0, min(start_raw, text_length))
    end_exclusive = max(start, min(end_raw, text_length))
    if end_exclusive <= start:
        return (None, None)

    # Docling charspan behaves like a Python slice [start, end).
    return (start, end_exclusive - 1)


def _build_docling_segment(
    *,
    prov: Any,
    raw_label: str,
    page_width: float,
    page_height: float,
    include_span: bool,
    text_length: int,
) -> LayoutSegmentIR | None:
    bbox = getattr(prov, "bbox", None)
    if bbox is None or page_width <= 0 or page_height <= 0:
        return None

    bbox_top_left = bbox.to_top_left_origin(page_height=page_height)
    width = bbox_top_left.r - bbox_top_left.l
    height = bbox_top_left.b - bbox_top_left.t
    if width <= 0 or height <= 0:
        return None

    start_index, end_index = _normalize_docling_charspan(
        getattr(prov, "charspan", None),
        text_length=text_length,
        include_span=include_span,
    )

    return LayoutSegmentIR(
        x=bbox_top_left.l / page_width,
        y=bbox_top_left.t / page_height,
        w=width / page_width,
        h=height / page_height,
        confidence=1.0,
        label=raw_label,
        start_index=start_index,
        end_index=end_index,
    )


def _merge_segments(segments: list[LayoutSegmentIR]) -> LayoutSegmentIR | None:
    if not segments:
        return None

    x1 = min(segment.x for segment in segments)
    y1 = min(segment.y for segment in segments)
    x2 = max(segment.x + segment.w for segment in segments)
    y2 = max(segment.y + segment.h for segment in segments)
    return LayoutSegmentIR(
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        confidence=1.0,
        label=segments[0].label,
    )


def _build_docling_layout_pages(
    *,
    doc: DoclingDocument,
    raw_pages: list[dict[str, Any]],
) -> list[ParseLayoutPageIR]:
    page_markdown_by_number: dict[int, str] = {}
    for page_data in raw_pages:
        page_number = page_data.get("page")
        if isinstance(page_number, int) and page_number > 0:
            page_markdown_by_number[page_number] = str(page_data.get("markdown", ""))

    layout_pages: list[ParseLayoutPageIR] = []
    for page_number in sorted(doc.pages.keys()):
        page = doc.pages[page_number]
        page_width = float(page.size.width)
        page_height = float(page.size.height)
        items: list[LayoutItemIR] = []

        for item, _level in doc.iterate_items(page_no=page_number):
            raw_label = _normalize_docling_label(getattr(item, "label", None))
            if raw_label is None or not _should_include_docling_label(raw_label):
                continue

            item_type = _docling_item_type(raw_label)
            item_value = _extract_docling_item_value(item, doc, raw_label)
            include_span = item_type == "text"

            page_provs = [
                prov for prov in getattr(item, "prov", []) or [] if getattr(prov, "page_no", None) == page_number
            ]
            segments = [
                segment
                for prov in page_provs
                if (
                    segment := _build_docling_segment(
                        prov=prov,
                        raw_label=raw_label,
                        page_width=page_width,
                        page_height=page_height,
                        include_span=include_span,
                        text_length=len(item_value),
                    )
                )
                is not None
            ]
            if not segments:
                continue

            merged_bbox = _merge_segments(segments)
            items.append(
                LayoutItemIR(
                    type=item_type,
                    value=item_value,
                    bbox=merged_bbox,
                    layout_segments=segments,
                )
            )

        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_number,
                width=page_width,
                height=page_height,
                md=page_markdown_by_number.get(page_number, ""),
                items=items,
            )
        )

    return layout_pages


@register_provider("docling_parse")
class DoclingParseProvider(Provider):
    """
    Provider for Docling PDF parsing via a custom HTTP endpoint.

    This provider sends PDFs to a Docling endpoint and returns markdown
    with tables formatted as HTML.
    """

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Docling parse provider.

        Args:
            provider_name: Name of the provider
            base_config: Optional configuration with:
                - `api_key`: Optional bearer token for the endpoint
                - `hf_token`: Deprecated alias for `api_key`
                - `endpoint_url`: Endpoint URL (required)
                - `timeout`: Request timeout in seconds (default: 120)
        """
        super().__init__(provider_name, base_config)

        # Optional bearer token; keep `hf_token` / `HF_TOKEN` as a deprecated
        # fallback for backwards compatibility with the previous HF deployment.
        self._api_key = (
            self.base_config.get("api_key")
            or self.base_config.get("hf_token")
            or os.getenv("DOCLING_PARSE_API_KEY")
            or os.getenv("HF_TOKEN")
            or ""
        )

        # Get endpoint URL (from config or env var)
        self._endpoint_url = self.base_config.get("endpoint_url") or os.getenv("DOCLING_PARSE_ENDPOINT_URL")
        if not self._endpoint_url:
            raise ProviderConfigError(
                "Docling endpoint URL is required. "
                "Set DOCLING_PARSE_ENDPOINT_URL environment variable or "
                "pass endpoint_url in pipeline config."
            )

        # Get timeout (default 120 seconds - PDF processing can be slow)
        self._timeout = self.base_config.get("timeout", 120)

    def _call_endpoint(self, pdf_bytes: bytes) -> dict[str, Any]:
        """
        Call the Docling endpoint with PDF bytes.

        Args:
            pdf_bytes: Raw PDF file bytes

        Returns:
            Raw JSON response from endpoint

        Raises:
            ProviderError: For any API errors
        """
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Encode PDF as base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        payload = {
            "inputs": {
                "pdf_base64": pdf_base64,
            }
        }

        try:
            response = requests.post(
                self._endpoint_url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            result_json = response.json()
            if isinstance(result_json, list):
                if not result_json:
                    raise ProviderPermanentError("Endpoint returned an empty list response.")
                first_result = result_json[0]
                if not isinstance(first_result, dict):
                    raise ProviderPermanentError("Endpoint returned a list response with a non-dict payload.")
                result = first_result
            elif isinstance(result_json, dict):
                result = result_json
            else:
                raise ProviderPermanentError(
                    f"Endpoint returned unsupported response type: {type(result_json).__name__}"
                )
            return result

        except requests.exceptions.Timeout as e:
            raise ProviderTransientError(f"Request timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ProviderTransientError(f"Connection error: {e}") from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            if status_code == 429:
                raise ProviderRateLimitError(f"Rate limit exceeded: {e}") from e
            elif status_code and 500 <= status_code < 600:
                raise ProviderTransientError(f"Server error ({status_code}): {e}") from e
            elif status_code and 400 <= status_code < 500:
                raise ProviderPermanentError(f"Client error ({status_code}): {e}") from e
            else:
                raise ProviderPermanentError(f"HTTP error: {e}") from e
        except (ProviderPermanentError, ProviderTransientError, ProviderRateLimitError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error calling endpoint: {e}") from e

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        Args:
            pipeline: Pipeline specification
            request: Inference request

        Returns:
            Raw inference result

        Raises:
            ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DoclingParseProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()

        # Check if file exists
        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        try:
            # Read PDF bytes
            pdf_bytes = source_path.read_bytes()

            # Call endpoint
            raw_output = self._call_endpoint(pdf_bytes)

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

        except (ProviderPermanentError, ProviderTransientError, ProviderRateLimitError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ParseOutput.

        Args:
            raw_result: Raw inference result from run_inference()

        Returns:
            Inference result with ParseOutput

        Raises:
            ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DoclingParseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        # Extract pages from response
        # Response format:
        # {
        #   "pages": [{"page": 1, "markdown": "..."}, ...],
        #   "markdown": "...",
        #   "docling_document": {...},
        # }
        raw_pages = raw_result.raw_output.get("pages", [])
        full_markdown = raw_result.raw_output.get("markdown", "")
        raw_docling_document = raw_result.raw_output.get("docling_document")

        # Convert to PageIR list (0-indexed)
        pages: list[PageIR] = []
        for page_data in raw_pages:
            # Docling uses 1-indexed pages, we use 0-indexed
            page_number = page_data.get("page", 1)
            page_index = page_number - 1 if page_number > 0 else 0
            markdown = page_data.get("markdown", "")

            pages.append(PageIR(page_index=page_index, markdown=markdown))

        # Sort by page index
        pages.sort(key=lambda p: p.page_index)

        # If we have pages but no full markdown, concatenate
        if pages and not full_markdown:
            full_markdown = "\n\n".join(p.markdown for p in pages)

        layout_pages: list[ParseLayoutPageIR] = []
        if raw_docling_document is not None:
            try:
                docling_document = DoclingDocument.model_validate(raw_docling_document)
            except Exception as e:
                raise ProviderPermanentError(f"Failed to validate docling_document payload: {e}") from e

            layout_pages = _build_docling_layout_pages(
                doc=docling_document,
                raw_pages=[page for page in raw_pages if isinstance(page, dict)],
            )

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=full_markdown,
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
