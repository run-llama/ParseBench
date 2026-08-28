"""Provider for dots.ocr parse via Modal OpenAI-compatible API.

Supports two prompt modes:
- ``prompt_parse_markdown``: Returns clean markdown (parse-only, no layout data).
- ``prompt_layout_all_en_v1_5``: Returns structured JSON with bboxes, categories,
  and text.  Markdown is reassembled from the layout elements and ``layout_pages``
  is populated so the same pipeline can be cross-evaluated for layout detection.
"""

import base64
import io
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, cast

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse._layout_utils import validated_sorted_page_records
from parse_bench.inference.providers.parse._multipage_image import (
    append_attempt_usages,
    attempt_usages_complete,
    open_document_page_images,
    run_page_with_retries,
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

# Default model name served by the Modal vLLM deployment
SERVED_MODEL_NAME = "dots-ocr-1.5"

# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

# Markdown-oriented prompt (no layout data)
PROMPT_PARSE_MARKDOWN = (
    "Parse this document image and output its content as clean markdown.\n"
    "- Preserve document structure (headings, paragraphs, lists, tables)\n"
    "- Convert tables to HTML format (<table>, <tr>, <th>, <td>) "
    "with colspan/rowspan for merged cells\n"
    "- Format formulas as LaTeX\n"
    "- Describe images/figures briefly in square brackets "
    "like [Figure: description]\n"
    "- Maintain reading order\n"
    "- Output the original text with no translation\n"
    "- Do not add commentary - only output the parsed content\n"
)

# dots.ocr 1.5 layout+text prompt (Core11 categories, structured JSON output)
PROMPT_LAYOUT_ALL_EN_V1_5 = (
    "Please output the layout information from the PDF image, "
    "including each layout element's bbox, its category, and the "
    "corresponding text content within the bbox.\n"
    "\n"
    "1. Bbox format: [x1, y1, x2, y2]\n"
    "\n"
    "2. Layout Categories: The possible categories are "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', "
    "'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n"
    "\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: If the picture is a chart or graph, extract all data points "
    "and format as an HTML table with flat combined column headers "
    "(e.g., 'Revenue 2023' not nested header rows). Include axis labels "
    "as column/row headers. For non-chart pictures, the text field should "
    "be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n"
    "\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, "
    "with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n"
    "\n"
    "5. Final Output: The entire output must be a single JSON object.\n"
)

PROMPT_CONFIGS: dict[str, str] = {
    "prompt_parse_markdown": PROMPT_PARSE_MARKDOWN,
    "prompt_layout_all_en_v1_5": PROMPT_LAYOUT_ALL_EN_V1_5,
}

# Prompt modes that return structured JSON with layout elements
_LAYOUT_PROMPT_MODES = {"prompt_layout_all_en_v1_5"}


# ---------------------------------------------------------------------------
# dots.ocr layout response schema
# ---------------------------------------------------------------------------


class DotsOcrLayoutItem(BaseModel):
    """Single layout element returned by the dots.ocr layout prompt."""

    bbox: list[float] = Field(..., min_length=4, max_length=4)
    category: str
    text: str = ""


@register_provider("dots_ocr_parse")
class DotsOcrParseProvider(Provider):
    """
    Unified parse provider for dots.ocr deployed on Modal.

    When ``prompt_mode`` is a layout prompt (e.g. ``prompt_layout_all_en_v1_5``),
    the model returns structured JSON with bounding boxes, categories, and text.
    The provider reassembles markdown from the layout elements and populates
    ``ParseOutput.layout_pages`` so the same pipeline can be cross-evaluated
    against layout detection datasets (following the LlamaParse pattern).

    When ``prompt_mode`` is ``prompt_parse_markdown`` (default), the model
    returns clean markdown and no layout data is produced.

    Configuration options:
        - endpoint_url (str, required): Modal server URL (or DOTS_OCR_ENDPOINT_URL env var)
        - model (str, default: "dots-ocr-1.5"): Served model name
        - prompt_mode (str, default: "prompt_parse_markdown"): Prompt selection
        - timeout (int, default: 180): Request timeout in seconds
        - max_tokens (int, default: 32768): Max tokens per response
        - dpi (int, default: 150): DPI for PDF to image conversion
        - temperature (float, default: 0.1): Sampling temperature
        - top_p (float, default: 0.9): Top-p sampling
        - prompt_override (str, optional): Custom prompt text (overrides prompt_mode)
    """

    PDF_RENDER_DPI = 150

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(provider_name, base_config)

        endpoint_url = self.base_config.get("endpoint_url") or os.getenv("DOTS_OCR_ENDPOINT_URL")
        if not endpoint_url:
            raise ProviderConfigError(
                "endpoint_url is required for dots_ocr_parse provider. "
                "Set DOTS_OCR_ENDPOINT_URL or pass endpoint_url in config."
            )

        self._client = OpenAI(
            base_url=endpoint_url,
            api_key=os.getenv("DOTS_OCR_API_KEY", "not-needed"),
            max_retries=0,
        )

        self._model = self.base_config.get("model", SERVED_MODEL_NAME)
        self._timeout = self.base_config.get("timeout", 180)
        self._max_tokens = self.base_config.get("max_tokens", 16384)
        self._dpi = self.base_config.get("dpi", self.PDF_RENDER_DPI)
        self._temperature = self.base_config.get("temperature", 0.1)
        self._top_p = self.base_config.get("top_p", 0.9)

        self._prompt_mode = self.base_config.get("prompt_mode", "prompt_parse_markdown")
        prompt_override = self.base_config.get("prompt_override")
        if prompt_override:
            self._prompt = prompt_override
        else:
            prompt = PROMPT_CONFIGS.get(self._prompt_mode)
            if not prompt:
                raise ProviderConfigError(
                    f"Unknown prompt_mode '{self._prompt_mode}'. Available: {sorted(PROMPT_CONFIGS.keys())}"
                )
            self._prompt = prompt

        self._is_layout_mode = self._prompt_mode in _LAYOUT_PROMPT_MODES

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _call_endpoint(self, image: Image.Image) -> tuple[str, dict[str, int]]:
        """Call dots.ocr and return response text with any reported usage."""
        img_base64 = self._image_to_base64(image)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                            },
                        ],
                    },
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
            )
        except Exception as e:
            self._raise_api_error(e)

        usage = self._extract_usage(response)
        content = response.choices[0].message.content
        if not content:
            raise ProviderTransientError("Empty response from model", attempt_stats=usage or None)
        return cast(str, content), usage

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        """Extract only token fields actually reported by the compatible API."""
        raw_usage = getattr(response, "usage", None)
        if raw_usage is None:
            return {}
        usage: dict[str, int] = {}
        for key, attribute in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = getattr(raw_usage, attribute, None)
            if value is not None:
                usage[key] = int(value)
        details = getattr(raw_usage, "completion_tokens_details", None)
        thinking_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
        if thinking_tokens is not None:
            usage["thinking_tokens"] = int(thinking_tokens)
        return usage

    @staticmethod
    def _raise_api_error(exc: Exception) -> NoReturn:
        """Classify real OpenAI-compatible transport and HTTP failures."""
        from openai import APIConnectionError, APITimeoutError

        if isinstance(exc, (APITimeoutError, APIConnectionError, TimeoutError, ConnectionError)):
            raise ProviderTransientError(f"Transient dots.ocr API failure: {exc}") from exc

        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 429:
            raise ProviderRateLimitError(f"dots.ocr rate limited (429): {exc}") from exc
        if status_code == 408 or isinstance(status_code, int) and status_code >= 500:
            raise ProviderTransientError(f"Transient dots.ocr HTTP {status_code}: {exc}") from exc
        raise ProviderPermanentError(f"Permanent dots.ocr API failure: {exc}") from exc

    # ------------------------------------------------------------------
    # HTML sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_html_attributes(text: str) -> str:
        """Quote unquoted HTML attributes so tables are valid XML."""

        def _quote_attrs(match: re.Match) -> str:
            tag_text = match.group(0)
            tag_text = re.sub(
                r'(\w+)=([^\s"\'<>=]+)',
                r'\1="\2"',
                tag_text,
            )
            return tag_text

        return re.sub(r"<[^>]+>", _quote_attrs, text)

    # ------------------------------------------------------------------
    # Layout JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_layout_items(content: str) -> list[DotsOcrLayoutItem]:
        """Parse dots.ocr layout response into typed items.

        The model is fine-tuned to return a JSON array of
        ``{bbox, category, text}`` objects.  We try ``json.loads``
        first, then fall back to extracting a JSON array from
        markdown fences or raw brackets.
        """
        candidates: list[str] = [content]

        # Fallback: extract from ```json ... ``` fences
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if fence:
            candidates.append(fence.group(1))

        # Fallback: extract outermost [...] bracket
        bracket = re.search(r"\[[\s\S]*\]", content)
        if bracket:
            candidates.append(bracket.group(0))

        from pydantic import TypeAdapter

        adapter = TypeAdapter(list[DotsOcrLayoutItem])
        for candidate in candidates:
            try:
                return cast(list[DotsOcrLayoutItem], adapter.validate_json(candidate))
            except Exception:
                continue

        raise ProviderPermanentError(f"Could not parse layout items from response: {content[:500]}")

    # ------------------------------------------------------------------
    # Per-page inference
    # ------------------------------------------------------------------

    def _call_page_with_retries(
        self,
        image: Image.Image,
        page_number: int,
        attempt_ledger: list[dict[str, object]],
        prior_attempt_ledger: list[dict[str, object]],
    ) -> tuple[str, dict[str, int], list[DotsOcrLayoutItem] | None]:
        """Own transient retries at the billable page request boundary."""

        def call_and_validate() -> tuple[str, dict[str, int], list[DotsOcrLayoutItem] | None]:
            response = self._call_endpoint(image)
            if isinstance(response, tuple):
                raw_text, usage = response
            else:  # compatibility for custom test doubles and provider adapters
                raw_text, usage = response, {}
            if not self._is_layout_mode:
                return raw_text, usage, None
            try:
                layout_items = self._parse_layout_items(raw_text)
            except ProviderPermanentError as exc:
                raise ProviderTransientError(str(exc), attempt_stats=usage or None) from exc
            return raw_text, usage, layout_items

        return run_page_with_retries(
            call_and_validate,
            provider_name="dots.ocr",
            page_number=page_number,
            attempt_ledger=attempt_ledger,
            prior_attempt_ledger=prior_attempt_ledger,
        )

    def _run_inference_pages(self, source_path: Path) -> dict[str, Any]:
        """Convert source file to images and run inference on each page."""
        pages = []
        api_attempts: list[dict[str, object]] = []
        with open_document_page_images(source_path, dpi=self._dpi) as images:
            for page_index, image in enumerate(images):
                page_image = image if image.mode in ("RGB", "RGBA") else image.convert("RGB")
                try:
                    attempts: list[dict[str, object]] = []
                    raw_text, _, layout_items = self._call_page_with_retries(
                        page_image,
                        page_index + 1,
                        attempts,
                        api_attempts,
                    )
                    api_attempts.extend(attempts)

                    page_data: dict[str, Any] = {
                        "page_index": page_index,
                        "width": page_image.width,
                        "height": page_image.height,
                        "raw_response": raw_text,
                    }

                    if self._is_layout_mode:
                        assert layout_items is not None
                        page_data["layout_items"] = [item.model_dump() for item in layout_items]
                        page_data["markdown"] = _reassemble_markdown(layout_items)
                    else:
                        page_data["markdown"] = raw_text
                        page_data["layout_items"] = []

                    pages.append(page_data)
                finally:
                    if page_image is not image:
                        page_image.close()

            num_pages = len(images)

        raw_output: dict[str, Any] = {
            "pages": pages,
            "num_pages": num_pages,
            "model": self._model,
            "prompt_mode": self._prompt_mode,
            "num_api_calls": len(api_attempts),
            "api_attempts": api_attempts,
            "config": {
                "dpi": self._dpi,
                "max_tokens": self._max_tokens,
                "timeout": self._timeout,
            },
        }
        attempt_usages: list[dict[str, int]] = []
        append_attempt_usages(attempt_usages, api_attempts)
        if attempt_usages_complete(attempt_usages):
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                raw_output[field] = sum(int(usage[field]) for usage in attempt_usages)
            if all("thinking_tokens" in usage for usage in attempt_usages):
                raw_output["thinking_tokens"] = sum(int(usage["thinking_tokens"]) for usage in attempt_usages)
        return raw_output

    # ------------------------------------------------------------------
    # run_inference
    # ------------------------------------------------------------------

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DotsOcrParseProvider only supports PARSE product type, got {request.product_type}"
            )

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
        if source_path.suffix.lower() not in supported_extensions:
            raise ProviderPermanentError(
                f"DotsOcrParseProvider supports {supported_extensions}, got {source_path.suffix}"
            )

        started_at = datetime.now()
        try:
            raw_output = self._run_inference_pages(source_path)
        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as exc:
            raise ProviderPermanentError(f"Unexpected error during inference: {exc}") from exc

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

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"DotsOcrParseProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        page_markdowns: list[str] = []

        for page_data in validated_sorted_page_records(raw_result.raw_output.get("pages", [])):
            page_index = page_data.get("page_index", 0)
            markdown = page_data.get("markdown", "")
            img_width = page_data.get("width", 0)
            img_height = page_data.get("height", 0)

            if markdown:
                markdown = self._sanitize_html_attributes(markdown)

            pages.append(PageIR(page_index=page_index, markdown=markdown))
            page_markdowns.append(markdown)

            # Build layout_pages from structured layout items (if available)
            layout_items = page_data.get("layout_items", [])
            is_layout_mode = raw_result.raw_output.get("prompt_mode") in _LAYOUT_PROMPT_MODES
            if is_layout_mode and img_width > 0 and img_height > 0:
                layout_page = _build_layout_page(
                    layout_items=layout_items,
                    page_number=page_index + 1,
                    img_width=img_width,
                    img_height=img_height,
                    page_markdown=markdown,
                )
                layout_pages.append(layout_page)

        full_markdown = "\n\n".join(page_markdowns)

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


# ======================================================================
# Module-level helpers
# ======================================================================


def _reassemble_markdown(layout_items: list[DotsOcrLayoutItem]) -> str:
    """Reassemble page markdown from layout element text fields."""
    parts: list[str] = []
    for item in layout_items:
        label = item.category.strip().lower()
        if not item.text:
            continue

        if label in ("title", "section-header"):
            parts.append(f"## {item.text}")
        elif label == "table":
            parts.append(item.text)  # Already HTML
        elif label == "formula":
            parts.append(f"$${item.text}$$")
        else:
            parts.append(item.text)

    return "\n\n".join(parts)


def _build_layout_page(
    *,
    layout_items: list[dict[str, Any]],
    page_number: int,
    img_width: int,
    img_height: int,
    page_markdown: str,
) -> ParseLayoutPageIR:
    """Convert dots.ocr layout items into a ParseLayoutPageIR for cross-eval."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(list[DotsOcrLayoutItem])
    typed_items = adapter.validate_python(layout_items)

    items: list[LayoutItemIR] = []
    for li in typed_items:
        x1, y1, x2, y2 = li.bbox

        seg = LayoutSegmentIR(
            x=x1 / img_width,
            y=y1 / img_height,
            w=(x2 - x1) / img_width,
            h=(y2 - y1) / img_height,
            confidence=1.0,
            label=li.category,
        )

        norm_label = li.category.strip().lower()
        if norm_label == "table":
            item_type = "table"
        elif norm_label == "picture":
            item_type = "image"
        else:
            item_type = "text"

        items.append(
            LayoutItemIR(
                type=item_type,
                value=li.text,
                bbox=seg,
                layout_segments=[seg],
            )
        )

    return ParseLayoutPageIR(
        page_number=page_number,
        width=float(img_width),
        height=float(img_height),
        md=page_markdown,
        items=items,
    )
