r"""Provider for LightOnOCR-3 served by vLLM.

Serve any LightOnOCR checkpoint with the flags used by the LightOn benchmark:

    docker run --gpus all -p 8810:8000 vllm/vllm-openai:v0.30.0 \
        --model <HF repo of a LightOnOCR checkpoint> --revision <commit> \
        --served-model-name loocr-grounding \
        --gpu-memory-utilization 0.85 \
        --limit-mm-per-prompt '{"image": 1}' \
        --max-model-len 24576

Use vLLM v0.30.0 or later: earlier versions' compiled mode makes the model loop. Then set
``LIGHTONOCR_3_SERVER_URL=http://localhost:8810``. ``--max-model-len`` covers a page capped at 5M pixels
(about 4.9k image tokens) plus 12288 output tokens.

The model answers each page as blocks separated by blank lines:

    ![<label>](x1,y1,x2,y2) <text content>

Boxes are 0-1000 normalized (left, top, right, bottom). ``table`` text is HTML, ``formula`` is LaTeX,
``title`` text carries its own ``#`` heading, and figure/chart/image blocks carry a description or nothing.
A ``+`` label suffix (``text+``) marks a block continued from the previous page or column.

One inference gives both the markdown and the layout, so the ``lightonocr_3_vllm_parse`` pipeline scores all
five ParseBench dimensions.
"""

import base64
import io
import math
import re
from pathlib import Path
from typing import Any

import aiohttp

from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse.qwen import QwenProvider, _build_layout_page
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

SERVED_MODEL_NAME = "loocr-grounding"
DEFAULT_PROMPT = "grounding"

_BLOCK_MARKER_RE = re.compile(r"!\[([a-zA-Z_+]+)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*")

# Raw block label -> ParseBench Canonical17 category. Unlisted labels (list, caption, code) map to Text.
LABEL_MAP: dict[str, str] = {
    "paragraph": "Text",
    "title": "Title",
    "figure_title": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "table": "Table",
    "header": "Page-header",
    "footer": "Page-footer",
    "page_number": "Page-footer",
    "figure": "Picture",
    "chart": "Picture",
    "image": "Picture",
    # ParseBench's layout ground truth labels an image in the header/footer band as Picture.
    "header_image": "Picture",
    "footer_image": "Picture",
}

# Page-section fields read these raw labels directly; header/footer images are left out
# because their text describes a visual, not the header.
_PAGE_SECTIONS = {
    "page_header_markdown": {"header", "page_header"},
    "page_footer_markdown": {"footer", "page_footer", "footnote", "page_footnote"},
    "printed_page_number": {"page_number"},
}

# The model writes no language tag on code blocks, which ParseBench's code rule needs. Only the languages
# its text-formatting tests use are detected; anything else gets a bare fence.
_CODE_LANG_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("json", re.compile(r"^\s*[\{\[].*[\}\]]\s*$", re.DOTALL)),
    ("python", re.compile(r"\b(def |import |from \w+ import|self\.|print\()")),
    ("cpp", re.compile(r"#include|::|\bstd::|\bvoid\b.*\(|;\s*$", re.MULTILINE)),
    ("fortran", re.compile(r"^\s{6}\S|\bIF\(|\bEND\s*DO\b|\bCALL\b", re.MULTILINE)),
]
_LEADING_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+")


def _base_label(raw_label: str) -> str:
    """``" Text+ "`` -> ``"text"``: the continuation marker and case are not part of the label."""
    return raw_label.strip().lower().rstrip("+")


def _strip_formula_delimiters(text: str) -> str:
    """Remove one outer ``$$`` or ``$`` pair, if the inside has no other ``$``."""
    stripped = text.strip()
    for delim in ("$$", "$"):
        if len(stripped) > 2 * len(delim) and stripped.startswith(delim) and stripped.endswith(delim):
            interior = stripped[len(delim) : -len(delim)]
            if "$" not in interior and interior.strip():
                return interior.strip()
    return text


def _render_block(raw_label: str, text: str) -> str:
    """Markdown for one block. Adds the markup its label implies when the model left it out.

    A title keeps the model's own ``#`` level: the model uses one ``title`` label for titles and section
    headers, so H1 is added only when the text has no heading marker.
    """
    if not text.strip():
        return ""
    label = _base_label(raw_label)
    if label == "code" and not text.lstrip().startswith("```"):
        lang = next((name for name, pattern in _CODE_LANG_HINTS if pattern.search(text)), "")
        return f"```{lang}\n{text}\n```"
    if label == "title" and not _LEADING_HEADING_RE.match(text):
        return f"# {text.strip()}"
    if label == "formula":
        return f"$$\n{_strip_formula_delimiters(text)}\n$$"
    return text


# ---------------------------------------------------------------------------
# Escaped dollar repair
#
# The model sometimes escapes math delimiters (``\$x^2\$``). This pairs them back into ``$x^2$`` when the
# span reads as math, and leaves prices, prose and code alone. Standard library only; behavior matches
# ``fix_escaped_dollars`` in the LightOnOCR repo.
# ---------------------------------------------------------------------------

_ATOM_RE = re.compile(
    r"\\(?:[A-Za-z]+|[^\r\n])|(?:\d+(?:\.\d*)?|\.\d+)|[^\W\d_]\w*|[-−–—+*/=<>:;,.'!|_^%&()\[\]{}]", re.UNICODE
)
_WORD_RE = re.compile(r"[^\W\d_]\w*")
_NUMERIC_START_RE = re.compile(r"[+\-−]?\s*(?:\d|[.,]\d)")
_DOLLAR_RE = re.compile(r"(?<![\\$])\\?\$(?!\$)")
_INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\$)((?:\\.|[^$\\\n])*)(?<!\\)\$(?!\$)")

# Spans that must never be edited: code, HTML, links, display math and comments.
_PROTECTED_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=fence)[ \t]*(?:\n|$)|\Z)"
    r"|(?P<ticks>`+)[^`\n]*(?P=ticks)"
    r"|^(?: {4}|\t)[^\n]*(?:\n|$)"
    r"|<!--.*?(?:-->|\Z)"
    r"|<(?:pre|code|script|style)\b[^>]*>.*?(?:</(?:pre|code|script|style)\s*>|\Z)"
    r"|<[/!?A-Za-z](?:\"[^\"]*\"|'[^']*'|[^'\">])*>"
    r"|(?:https?://|mailto:|www\.)[^\s<>]+"
    r"|^[ \t]{0,3}\[[^\]\n]+\]:[^\n]*"
    r"|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)",
    re.M | re.S | re.I,
)


def _braces_balanced(body: str) -> bool:
    depth = 0
    for token in re.findall(r"\\.|[{}]", body):
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def _mask_command_arguments(body: str) -> str | None:
    """Replace the braced arguments of each ``\\command`` with ``{}``. None if one is unclosed."""
    out = []
    pos = 0
    while pos < len(body):
        command = re.match(r"\\[A-Za-z]+", body[pos:])
        if not command:
            out.append(body[pos])
            pos += 1
            continue
        out.append(command[0])
        pos += len(command[0])
        while pos < len(body) and body[pos] == "{":
            depth = 1
            pos += 1
            while pos < len(body) and depth:
                if body[pos] == "\\":
                    pos += 2
                    continue
                depth += (body[pos] == "{") - (body[pos] == "}")
                pos += 1
            if depth:
                return None
            out.append("{}")
    return "".join(out)


def _is_lexical_math(body: str) -> bool:
    """Conservative check that ``body`` holds only identifiers, numbers and operators.

    This is lexical evidence, not proof of mathematical correctness. Adjacent words are not accepted as an
    implicit product, but command arguments may hold text (``\\text{a long description}``).
    """
    body = body.strip()
    if not body or "<td" in body.lower() or "</" in body or not _braces_balanced(body):
        return False
    clean = _mask_command_arguments(body)
    if clean is None:
        return False
    previous_word = None
    pos = 0
    for match in _ATOM_RE.finditer(clean):
        if clean[pos : match.start()].strip():
            return False
        is_word = bool(_WORD_RE.fullmatch(match[0]))
        if is_word and previous_word and (len(match[0]) > 1 or len(previous_word) > 1):
            return False
        previous_word = match[0] if is_word else None
        pos = match.end()
    return not clean[pos:].strip()


def _is_plausible_math(body: str) -> bool:
    stripped = body.strip()
    if not _is_lexical_math(body) or re.fullmatch(r"[^\W\d_]{2,}", stripped):
        return False
    if body != stripped and not re.search(r"[\w\\]", stripped):
        return False
    if len(stripped) > 1 and stripped[0] in ",;":
        return False
    # Escaped delimiters and text arguments do not count toward bracket balance.
    simple = re.sub(r"\\text\{[^{}]*\}", "", stripped)
    level = 0
    for token in re.findall(r"\\.|[()\[\]]", simple):
        if token in ("(", "["):
            level += 1
        elif token in (")", "]"):
            level -= 1
        if level < 0:
            return False
    return level == 0


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Protected spans, plus valid-looking inline math (a ``$`` inside it is not a broken delimiter)."""
    spans = [(m.start(), m.end()) for m in _PROTECTED_RE.finditer(text)]
    # Inline link destinations, including relative URLs.
    for m in re.finditer(r"\]\(", text):
        pos = m.end()
        depth = 1
        while pos < len(text) and text[pos] != "\n" and depth:
            if text[pos] == "\\":
                pos += 2
                continue
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1
        if depth == 0:
            spans.append((m.start(), pos))
    for m in _INLINE_MATH_RE.finditer(text):
        if _is_plausible_math(m[1]):
            spans.append((m.start(), m.end()))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _is_ambiguous_pair(body: str, before: str, after: str, left_escaped: bool, right_escaped: bool) -> bool:
    """Whether an escaped-delimiter pair could be a price, prose or a broken span: leave those alone."""
    numeric = bool(_NUMERIC_START_RE.match(body.strip()))
    is_environment = re.fullmatch(r"\s*\\begin\{([A-Za-z*]+)\}.*\\end\{\1\}\s*", body, re.S)
    # A currency code glued to the dollar sign, e.g. US$, S$, HK$.
    code_currency = left_escaped and bool(re.search(r"(?<!\w)[A-Z]{1,3}$", before))
    return bool(
        ("\n" in body or "\r" in body)
        and not is_environment
        or re.search(r"\n[ \t]*\n", body)
        or code_currency
        or (left_escaped and not right_escaped and numeric)
        or (numeric and _NUMERIC_START_RE.match(after.lstrip(" \t")))
        or re.fullmatch(r"[^\W\d_]{2,}", body.strip())
        or not _is_plausible_math(body)
    )


def _delimiter_edits(segment: str, offset: int) -> list[tuple[int, int, str]]:
    """Edits that pair the dollar delimiters of one unprotected segment. ``offset`` is its start in the text."""
    tokens = list(_DOLLAR_RE.finditer(segment))
    edits = []
    i = 0
    while i + 1 < len(tokens):
        left, right = tokens[i], tokens[i + 1]
        body = segment[left.end() : right.start()]
        left_escaped = left[0].startswith("\\")
        right_escaped = right[0].startswith("\\")
        if not left_escaped and not right_escaped:
            i += 2 if _is_plausible_math(body) else 1
            continue
        if _is_ambiguous_pair(
            body,
            before=segment[: left.start()],
            after=segment[right.end() :],
            left_escaped=left_escaped,
            right_escaped=right_escaped,
        ):
            i += 1
            continue
        edits.append((offset + left.start(), offset + right.end(), f"${body}$"))
        i += 2
    return edits


def _repair_dollar_delimiters(text: str) -> str:
    edits: list[tuple[int, int, str]] = []
    previous = 0
    for start, end in _protected_spans(text):
        edits += _delimiter_edits(text[previous:start], previous)
        previous = end
    edits += _delimiter_edits(text[previous:], previous)
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


@register_provider("lightonocr-3")
class LightOnOcr3Provider(QwenProvider):
    """QwenProvider for a vLLM server serving LightOnOCR-3 with its grounding prompt.

    Configuration on top of QwenProvider's (``server_url_env``, ``dpi``, ``max_tokens``, ``temperature``,
    ``timeout``):
        - model (str, default="loocr-grounding"): served model name, as set by ``--served-model-name``
        - prompt (str, default="grounding"): prompt text sent with each page
        - max_pixels (int, optional): cap on pixels per page. A page over the cap renders at the DPI that fits
          it. Unset means no cap.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._model = self.base_config.get("model", SERVED_MODEL_NAME)
        self._prompt = self.base_config.get("prompt", DEFAULT_PROMPT)
        self._max_pixels: int | None = self.base_config.get("max_pixels")

    @staticmethod
    def parse_blocks(content: str) -> list[dict[str, Any]]:
        """The raw response as ordered ``{label, bbox, text}`` blocks."""
        matches = list(_BLOCK_MARKER_RE.finditer(content))
        ends = [m.start() for m in matches[1:]] + [len(content)] if matches else []
        return [
            {"label": m[1], "bbox": [int(m[i]) for i in range(2, 6)], "text": content[m.end() : end].strip()}
            for m, end in zip(matches, ends, strict=True)
        ]

    def _pdf_to_images_with_size(self, pdf_path: Path) -> list[tuple[bytes, int, int]]:
        """Pages at ``self._dpi``. With ``max_pixels``, each page is sized before rasterizing, since rendering
        first and shrinking after hits PIL's decompression-bomb limit on large scans."""
        if self._max_pixels is None:
            return super()._pdf_to_images_with_size(pdf_path)
        from pdf2image import convert_from_path
        from PIL import Image
        from pypdf import PdfReader

        pages = []
        for index, page in enumerate(PdfReader(pdf_path).pages):
            # Some page boxes are inverted, hence abs().
            width_in = abs(float(page.cropbox.width)) / 72
            height_in = abs(float(page.cropbox.height)) / 72
            dpi = min(self._dpi, math.floor(math.sqrt(self._max_pixels / (width_in * height_in))))
            image = convert_from_path(pdf_path, dpi=dpi, first_page=index + 1, last_page=index + 1)[0]
            if image.width * image.height > self._max_pixels:  # poppler rounds page sizes up
                scale = math.sqrt(self._max_pixels / (image.width * image.height))
                image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            pages.append((buffer.getvalue(), image.width, image.height))
        if not pages:
            raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
        return pages

    async def _run_inference_async(self, image_bytes: bytes, img_width: int, img_height: int) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            raw_content = await self._call_api(session, base64.b64encode(image_bytes).decode())
        return {
            "pages": [
                {
                    "page_index": 0,
                    "width": img_width,
                    "height": img_height,
                    "raw_response": raw_content,
                    "layout_items": [
                        {
                            "bbox": block["bbox"],
                            "category": LABEL_MAP.get(_base_label(block["label"]), "Text"),
                            "text": block["text"],
                            # Kept beside the category: footer and page_number both map to Page-footer,
                            # but the page-section fields need them apart.
                            "raw_label": block["label"],
                        }
                        for block in self.parse_blocks(raw_content)
                    ],
                }
            ],
            "_config": {
                "server_url": self._server_url,
                "model": self._model,
                "dpi": self._dpi,
                "max_pixels": self._max_pixels,
                "temperature": self._temperature,
            },
        }

    async def _run_inference_pages_async(self, pages: list[tuple[bytes, int, int]]) -> dict[str, Any]:
        results = [await self._run_inference_async(*page) for page in pages]
        return {**results[0], "pages": [{**r["pages"][0], "page_index": i} for i, r in enumerate(results)]}

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"LightOnOcr3Provider only supports PARSE, got {raw_result.product_type}")

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        for page in sorted(raw_result.raw_output.get("pages") or [], key=lambda p: p.get("page_index", 0)):
            page_index = page.get("page_index", 0)
            markdown, layout_page = self._normalize_page(page)
            pages.append(PageIR(page_index=page_index, markdown=markdown))
            if layout_page is not None:
                layout_pages.append(layout_page)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown="\n\n".join(p.markdown for p in pages),
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

    def _normalize_page(self, page: dict[str, Any]) -> tuple[str, ParseLayoutPageIR | None]:
        """Markdown and layout for one page. The layout page is None when the page has no usable boxes."""
        items = [
            {**item, "text": _repair_dollar_delimiters(str(item.get("text", "")))}
            for item in page.get("layout_items", [])
        ]
        blocks = [_render_block(str(item.get("raw_label", "")), item["text"]) for item in items]
        markdown = "\n\n".join(block for block in blocks if block)
        if markdown:
            markdown = self._sanitize_html_attributes(self._convert_md_tables_to_html(markdown))

        width, height = page.get("width", 0), page.get("height", 0)
        if not items or width <= 0 or height <= 0:
            return markdown, None
        layout_page = _build_layout_page(
            layout_items=items,
            page_number=page.get("page_index", 0) + 1,
            img_width=width,
            img_height=height,
            page_markdown=markdown,
        )
        for field, labels in _PAGE_SECTIONS.items():
            texts = (item["text"].strip() for item in items if _base_label(str(item.get("raw_label", ""))) in labels)
            setattr(layout_page, field, "\n\n".join(text for text in texts if text))
        return markdown, layout_page
