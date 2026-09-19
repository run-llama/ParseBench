"""Provider for HPD-Parsing through an OpenAI-compatible vLLM endpoint.

HPD-Parsing emits a stream of document blocks in this form::

    <BLOCK>{label} [x1, y1, x2, y2]<CHILD>{content}

Coordinates use a normalized 0-1000 grid. Tables arrive as HTML, formulas as
LaTeX, and image or chart blocks may contain only layout geometry.
"""

import base64
import io
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageSequence

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
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

SERVED_MODEL_NAME = "PaddlePaddle/HPD-Parsing"

PROMPT_MODES: dict[str, str] = {
    "fork": "document parsing with fork.",
    "plain": "document parsing.",
}

REPETITION_DETECTION = {
    "min_pattern_size": 64,
    "max_pattern_size": 128,
    "min_count": 10,
}

# HPD follows the Paddle document-layout vocabulary but also emits a few
# model-specific synonyms. Keep this map local so the provider does not import
# PaddleOCR and its unrelated runtime dependencies.
_HPD_LABEL_TO_CANONICAL: dict[str, str] = {
    "doc_title": "Title",
    "title": "Section-header",
    "paragraph_title": "Section-header",
    "header": "Page-header",
    "footer": "Page-footer",
    "text": "Text",
    "content": "Text",
    "abstract": "Text",
    "aside_text": "Text",
    "ref_text": "Text",
    "phonetic": "Text",
    "index": "Text",
    "formula_number": "Text",
    "reference": "Text",
    "reference_content": "Text",
    "footnote": "Footnote",
    "page_footnote": "Footnote",
    "vision_footnote": "Footnote",
    "table_footnote": "Footnote",
    "chart_footnote": "Footnote",
    "image_footnote": "Footnote",
    "image": "Picture",
    "image_block": "Picture",
    "figure": "Picture",
    "chart": "Picture",
    "seal": "Picture",
    "header_image": "Picture",
    "footer_image": "Picture",
    "figure_title": "Caption",
    "figure_caption": "Caption",
    "image_caption": "Caption",
    "table_title": "Caption",
    "table_caption": "Caption",
    "chart_title": "Caption",
    "chart_caption": "Caption",
    "formula_caption": "Caption",
    "code_caption": "Caption",
    "figure_table_title": "Caption",
    "list": "List-item",
    "list_item": "List-item",
    "table": "Table",
    "formula": "Formula",
    "equation": "Formula",
    "code": "Code",
    "algorithm": "Code",
}

_BLOCK_RE = re.compile(
    r"<BLOCK>\s*(?P<label>[A-Za-z_]+)\s*\[(?P<bbox>[^\]]*)\]\s*"
    r"(?:<CHILD>(?P<content>[\s\S]*?))?(?=<BLOCK>|\Z)"
)

_PIPE_TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:|-]+\|?$")

_NO_CONTENT_SNIPPETS = (
    "The image is too blurry to recognize any text content.",
    (
        "The image contains no text or characters. It is a graphical element "
        "(a horizontal line with a vertical line) and does not contain any chart, "
        "graph, or data points that can be extracted. Therefore, the correct OCR "
        "output is an empty string."
    ),
)
_NON_TEXT_SENTINEL = "[Non-Text]"
_TABLE_OPEN_RE = re.compile(r"<table(?:\s[^>]*)?>", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_CONTROL_MARKER_RE = re.compile(r"<(?:FORK|CHILD|BLOCK)>")
_TABLE_TAG_RE = re.compile(
    r"<(?P<closing>/)?(?P<tag>table|thead|tbody|tfoot|tr|th|td|caption|colgroup|col)\b[^<>]*>",
    re.IGNORECASE,
)
_UNQUOTED_ATTRIBUTE_RE = re.compile(
    r"(?P<prefix>\s)(?P<name>[A-Za-z][A-Za-z0-9:_-]*)(?P<equals>\s*=\s*)"
    r"(?P<value>[^\s\"'`=<>]+?)(?=\s|/?>)"
)
_GLOBAL_TABLE_ATTRIBUTES = {"class", "dir", "hidden", "id", "lang", "role", "style", "title"}
_TABLE_ATTRIBUTES_BY_TAG = {
    "table": {"align", "bgcolor", "border", "cellpadding", "cellspacing", "frame", "rules", "summary", "width"},
    "caption": {"align"},
    "colgroup": {"align", "span", "valign", "width"},
    "col": {"align", "span", "valign", "width"},
    "thead": {"align", "char", "charoff", "valign"},
    "tbody": {"align", "char", "charoff", "valign"},
    "tfoot": {"align", "char", "charoff", "valign"},
    "tr": {"align", "bgcolor", "char", "charoff", "valign"},
    "th": {
        "abbr",
        "align",
        "axis",
        "bgcolor",
        "char",
        "charoff",
        "colspan",
        "headers",
        "height",
        "nowrap",
        "rowspan",
        "scope",
        "valign",
        "width",
    },
    "td": {
        "abbr",
        "align",
        "axis",
        "bgcolor",
        "char",
        "charoff",
        "colspan",
        "headers",
        "height",
        "nowrap",
        "rowspan",
        "scope",
        "valign",
        "width",
    },
}


def _canonical_label(raw_label: str, bbox: list[float]) -> str | None:
    """Map an HPD block label to a canonical ParseBench label."""
    key = raw_label.strip().lower()
    if key in ("page_number", "number"):
        if len(bbox) != 4:
            return "Text"
        return "Page-header" if (bbox[1] + bbox[3]) / 2.0 < 500.0 else "Page-footer"
    return _HPD_LABEL_TO_CANONICAL.get(key)


def _is_pipe_table_line(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and not stripped.startswith("<!--")


def _convert_pipe_tables_to_html(markdown: str) -> str:
    """Convert markdown pipe tables without importing another provider."""
    if not markdown or "|" not in markdown:
        return markdown

    import markdown2

    lines = markdown.split("\n")
    result: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not _is_pipe_table_line(line):
            result.append(line)
            index += 1
            continue

        end = index + 1
        while end < len(lines) and _is_pipe_table_line(lines[end]):
            end += 1
        table_lines = lines[index:end]
        has_separator = any(_PIPE_TABLE_SEPARATOR_RE.match(item.strip()) for item in table_lines)
        if len(table_lines) >= 3 and has_separator:
            rendered = markdown2.markdown("\n".join(table_lines), extras=["tables"])
            if "<table" in rendered.lower():
                result.append(rendered.strip())
                index = end
                continue

        result.append(line)
        index += 1
    return "\n".join(result)


def _strip_latex_delimiters(text: str) -> str:
    r"""Remove one outer or unmatched opening display delimiter."""
    stripped = text.strip()
    for opener, closer in (("\\[", "\\]"), ("\\(", "\\)")):
        if not stripped.startswith(opener):
            continue
        if stripped.endswith(closer):
            interior = stripped[len(opener) : -len(closer)].strip()
        elif closer not in stripped[len(opener) :]:
            interior = stripped[len(opener) :].strip()
        else:
            continue
        return interior
    return text


def _clean_no_content_sentinels(text: str) -> str:
    """Remove the no-content messages used by HPD's public postprocessor."""
    cleaned = text.strip()
    for snippet in _NO_CONTENT_SNIPPETS:
        cleaned = cleaned.replace(snippet, "").strip()
    return "" if cleaned == _NON_TEXT_SENTINEL else cleaned


def _close_unclosed_table_tags(text: str) -> str:
    """Close truncated table markup within one HPD block."""
    missing_closes = len(_TABLE_OPEN_RE.findall(text)) - len(_TABLE_CLOSE_RE.findall(text))
    if missing_closes > 0:
        text += "</table>" * missing_closes
    return text


def _quote_unquoted_table_attributes(text: str) -> str:
    """Quote known attributes only inside real HTML table tags."""

    def _sanitize_tag(tag_match: re.Match[str]) -> str:
        if tag_match.group("closing"):
            return tag_match.group(0)
        allowed_attributes = _GLOBAL_TABLE_ATTRIBUTES | _TABLE_ATTRIBUTES_BY_TAG[tag_match.group("tag").lower()]

        def _quote_attribute(attribute_match: re.Match[str]) -> str:
            if attribute_match.group("name").lower() not in allowed_attributes:
                return attribute_match.group(0)
            return (
                f"{attribute_match.group('prefix')}{attribute_match.group('name')}"
                f'{attribute_match.group("equals")}"{attribute_match.group("value")}"'
            )

        return _UNQUOTED_ATTRIBUTE_RE.sub(_quote_attribute, tag_match.group(0))

    return _TABLE_TAG_RE.sub(_sanitize_tag, text)


def _parse_blocks(content: str) -> list[dict[str, Any]]:
    """Parse an HPD block stream into layout-aware text items."""
    items: list[dict[str, Any]] = []
    for match in _BLOCK_RE.finditer(content):
        try:
            bbox = [float(value) for value in re.split(r"[,\s]+", match.group("bbox").strip()) if value]
        except ValueError:
            bbox = []

        label = _canonical_label(match.group("label"), bbox)
        if label is None:
            continue

        text = _CONTROL_MARKER_RE.split(match.group("content") or "", maxsplit=1)[0]
        text = _clean_no_content_sentinels(text)
        if label == "List-item" and not text:
            continue
        if label == "Table" and text:
            text = _close_unclosed_table_tags(text)
            text = _quote_unquoted_table_attributes(text)
        if label == "Formula" and text:
            text = _strip_latex_delimiters(text)
        items.append({"label": label, "bbox": bbox, "text": text})
    return items


@register_provider("hpd_parsing")
class HpdParsingProvider(Provider):
    """Parse documents with an HPD-Parsing vLLM server.

    Configuration options:
        - server_url (str, optional): vLLM base URL ending in ``/v1``
        - server_url_env (str, default ``HPD_PARSING_SERVER_URL``): fallback
          environment variable for the server URL
        - model (str, default ``PaddlePaddle/HPD-Parsing``): served model name
        - prompt_mode (str, default ``fork``): ``fork`` or ``plain``
        - dpi (int, default 150): DPI for PDF rendering
        - max_tokens (int, default 8000): maximum output tokens per page
        - temperature (float, default 0.0): sampling temperature
        - timeout (int, default 900): request timeout in seconds
        - api_key_env (str, default ``VLLM_API_KEY``): optional API key variable
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None) -> None:
        super().__init__(provider_name, base_config)

        server_url_env = self.base_config.get("server_url_env", "HPD_PARSING_SERVER_URL")
        server_url = self.base_config.get("server_url") or os.getenv(server_url_env)
        if not server_url:
            raise ProviderConfigError(f"HPD-Parsing provider requires 'server_url' in config or {server_url_env}.")

        self._model = self.base_config.get("model", SERVED_MODEL_NAME)
        self._dpi = self.base_config.get("dpi", 150)
        self._max_tokens = self.base_config.get("max_tokens", 8000)
        self._temperature = self.base_config.get("temperature", 0.0)
        self._timeout = self.base_config.get("timeout", 900)

        prompt_mode = self.base_config.get("prompt_mode", "fork")
        prompt = PROMPT_MODES.get(prompt_mode)
        if prompt is None:
            raise ProviderConfigError(f"Unknown prompt_mode '{prompt_mode}'. Available: {sorted(PROMPT_MODES)}")
        self._prompt_mode = prompt_mode
        self._prompt = prompt

        api_key_env = self.base_config.get("api_key_env", "VLLM_API_KEY")
        self._client = OpenAI(
            base_url=str(server_url),
            api_key=os.getenv(api_key_env, "not-needed"),
            timeout=float(self._timeout),
        )

    def _pdf_to_images(self, pdf_path: str) -> list[Image.Image]:
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise ProviderConfigError("pdf2image is required to render PDF files.") from exc
        try:
            return convert_from_path(pdf_path, dpi=self._dpi)
        except Exception as exc:
            raise ProviderPermanentError(f"Failed to convert PDF to images: {exc}") from exc

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_endpoint(self, image: Image.Image) -> str:
        image_base64 = self._image_to_base64(image)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                            {"type": "text", "text": self._prompt},
                        ],
                    },
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                extra_body={"repetition_detection": REPETITION_DETECTION},
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if not isinstance(status_code, int):
                status_code = getattr(getattr(exc, "response", None), "status_code", None)

            if status_code == 429:
                raise ProviderRateLimitError(f"HTTP 429 from HPD-Parsing endpoint: {exc}") from exc
            if status_code == 408 or isinstance(status_code, int) and 500 <= status_code < 600:
                raise ProviderTransientError(f"HTTP {status_code} from HPD-Parsing endpoint: {exc}") from exc
            if isinstance(status_code, int) and 400 <= status_code < 500:
                raise ProviderPermanentError(f"HTTP {status_code} from HPD-Parsing endpoint: {exc}") from exc

            error_message = str(exc).lower()
            error_type = type(exc).__name__
            if error_type == "RateLimitError" or "429" in error_message or "rate limit" in error_message:
                raise ProviderRateLimitError(f"Rate limited by HPD-Parsing endpoint: {exc}") from exc
            if (
                isinstance(exc, (TimeoutError, ConnectionError))
                or error_type in {"APIConnectionError", "APITimeoutError", "InternalServerError"}
                or any(marker in error_message for marker in ("timeout", "connection", "502", "503"))
            ):
                raise ProviderTransientError(f"API call failed: {exc}") from exc
            raise ProviderPermanentError(f"API call failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise ProviderPermanentError("Empty response from model")
        return content

    def _run_inference_pages(self, source_path: Path) -> dict[str, Any]:
        if source_path.suffix.lower() == ".pdf":
            images = self._pdf_to_images(str(source_path))
        elif source_path.suffix.lower() in {".tif", ".tiff"}:
            with Image.open(source_path) as source_image:
                images = [frame.copy() for frame in ImageSequence.Iterator(source_image)]
        else:
            with Image.open(source_path) as source_image:
                images = [source_image.copy()]
        if not images:
            raise ProviderPermanentError(f"No pages found in: {source_path}")

        pages: list[dict[str, Any]] = []
        for page_index, image in enumerate(images):
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
            pages.append(
                {
                    "page_index": page_index,
                    "width": image.width,
                    "height": image.height,
                    "raw_response": self._call_endpoint(image),
                }
            )

        return {
            "pages": pages,
            "num_pages": len(images),
            "model": self._model,
            "prompt_mode": self._prompt_mode,
            "config": {
                "dpi": self._dpi,
                "max_tokens": self._max_tokens,
                "timeout": self._timeout,
            },
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"HpdParsingProvider only supports PARSE product type, got {request.product_type}"
            )

        source_path = Path(request.source_file_path)
        if not source_path.exists():
            raise ProviderPermanentError(f"Source file not found: {source_path}")

        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
        if source_path.suffix.lower() not in supported_extensions:
            raise ProviderPermanentError(
                f"HpdParsingProvider supports {supported_extensions}, got {source_path.suffix}"
            )

        started_at = datetime.now()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                raw_output = self._run_inference_pages(source_path)
                completed_at = datetime.now()
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
            except (ProviderTransientError, ProviderRateLimitError) as exc:
                if attempt < max_retries - 1:
                    delay = 15 * (2**attempt)
                    print(
                        f"[hpd-parsing] Transient error on {request.example_id}: {exc}. "
                        f"Retrying in {delay}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    continue
                raise
            except ProviderPermanentError:
                raise
            except Exception as exc:
                raise ProviderPermanentError(f"HPD-Parsing inference failed: {exc}") from exc

        raise AssertionError("HPD-Parsing retry loop exited unexpectedly")

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"HpdParsingProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        page_markdowns: list[str] = []
        prompt_mode = raw_result.raw_output.get("prompt_mode")

        for page_data in raw_result.raw_output.get("pages", []):
            page_index = page_data.get("page_index", 0)
            raw_response = page_data.get("raw_response", "") or ""
            if prompt_mode == "plain" and "<BLOCK>" not in raw_response:
                items: list[dict[str, Any]] = []
                markdown = raw_response
            else:
                items = _parse_blocks(raw_response)
                markdown = items_to_markdown(items)
            if items and markdown:
                markdown = _convert_pipe_tables_to_html(markdown)

            pages.append(PageIR(page_index=page_index, markdown=markdown))
            page_markdowns.append(markdown)
            layout_pages.extend(
                build_layout_pages(
                    items=items,
                    image_width=page_data.get("width", 0),
                    image_height=page_data.get("height", 0),
                    markdown=markdown,
                    page_number=page_index + 1,
                )
            )

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown="\n\n".join(page_markdowns),
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
