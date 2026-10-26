"""Render API Markdown while preserving structured HTML and source content."""

import re
from html import escape, unescape
from typing import Any

from markdown_it import MarkdownIt

_CODE = re.compile(
    r"<pre\b[^>]*>\s*<code\b(?P<attrs>[^>]*)>(?P<body>.*?)</code>\s*</pre>",
    re.IGNORECASE | re.DOTALL,
)
_LANGUAGE = re.compile(r"""class=["'](?P<classes>[^"']*)["']""", re.IGNORECASE)
_LANGUAGE_TOKEN = re.compile(r"(?:^|\s)language-([A-Za-z0-9_+#.-]+)(?:\s|$)")
_CARRIER = re.compile(
    r"""<p\b(?=[^>]*\bclass=["'][^"']*\b(?:chart|figure)-(?P<field>ocr-text|description|type)\b[^"']*["'])[^>]*>(?P<body>.*?)</p>""",
    re.IGNORECASE | re.DOTALL,
)
_LIST = re.compile(r"<(?:ul|ol)\b", re.IGNORECASE)
_TAG = re.compile(r"""<(?P<tag>[A-Za-z][\w:-]*)(?P<attrs>(?:[^<>"']|"[^"]*"|'[^']*')*)>""")
_ATTR = re.compile(r"""\s+(?P<name>[\w:-]+)(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?""")
_LITERAL_CODE = re.compile(
    r"<(?P<tag>pre|code)\b[^>]*>.*?(?:</(?P=tag)\s*>|\Z)"
    r"|(?<!`)(?P<ticks>`+)(?!`).*?(?<!`)(?P=ticks)(?!`)",
    re.IGNORECASE | re.DOTALL,
)
_MARKDOWN = MarkdownIt("commonmark")
_INLINE_CODE = re.compile(r"(?<!`)(?P<ticks>`+)(?!`).*?(?<!`)(?P=ticks)(?!`)", re.DOTALL)


def element_text(element: dict[str, Any]) -> str:
    content = element.get("content") or {}
    return str(content.get("text") or content.get("markdown") or "")


def element_html(element: dict[str, Any]) -> str:
    content = element.get("content") or {}
    return str(content.get("html") or content.get("markdown") or escape(element_text(element)))


def _markdown_code_ranges(markup: str) -> list[tuple[int, int]]:
    offsets = [0]
    offsets.extend(match.end() for match in re.finditer(r"\r\n|\r|\n", markup))
    if offsets[-1] < len(markup):
        offsets.append(len(markup))
    ranges = []
    for token in _MARKDOWN.parse(markup):
        if token.type in {"fence", "code_block"} and token.map:
            ranges.append((offsets[token.map[0]], offsets[token.map[1]]))
    return ranges


def strip_html_metadata(markup: str) -> str:
    """Remove transport attributes and image narration, preserving literal code."""
    protected = [(m.start(), m.end()) for m in _LITERAL_CODE.finditer(markup)]
    protected.extend(_markdown_code_ranges(markup))
    merged: list[list[int]] = []
    for start, end in sorted(protected):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    def clean_tag(match: re.Match[str]) -> str:
        def attribute(attr: re.Match[str]) -> str:
            name = attr.group("name").lower()
            if name in {"id", "data-category", "data-coord"}:
                return ""
            if match.group("tag").lower() == "img" and name == "alt":
                return ""
            return attr.group(0)

        return "<" + match.group("tag") + _ATTR.sub(attribute, match.group("attrs")) + ">"

    parts = []
    cursor = 0
    for start, end in merged:
        parts.extend([_TAG.sub(clean_tag, markup[cursor:start]), markup[start:end]])
        cursor = end
    parts.append(_TAG.sub(clean_tag, markup[cursor:]))
    return "".join(parts)


def _code_fence(match: re.Match[str]) -> str:
    language = ""
    classes = _LANGUAGE.search(match.group("attrs"))
    if classes:
        explicit = _LANGUAGE_TOKEN.search(classes.group("classes"))
        if explicit:
            language = explicit.group(1).lower()
    body = unescape(match.group("body"))
    longest = max((len(run) for run in re.findall(r"`+", body)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"\n{fence}{language}\n{body}" + ("" if body.endswith("\n") else "\n") + f"{fence}\n"


def normalize_element(element: dict[str, Any]) -> str:
    content = element.get("content") or {}
    category = str(element.get("category") or "").lower()
    source = element_html(element)
    if category in {"chart", "figure"}:
        source = _CARRIER.sub(
            lambda match: "<p>" + match.group("body") + "</p>" if match.group("field").lower() == "ocr-text" else "",
            source,
        )
    if category not in {"table", "chart", "figure"} and not _LIST.search(source) and content.get("markdown"):
        source = str(content["markdown"])
    else:
        source = source.strip()
    blocks = _markdown_code_ranges(source)
    inline = [(m.start(), m.end()) for m in _INLINE_CODE.finditer(source)]

    def closed_code(match: re.Match[str]) -> str:
        if any(start < match.end() and end > match.start() for start, end in blocks):
            return match.group(0)
        if any(start <= match.start() and end >= match.end() for start, end in inline):
            return match.group(0)
        return _code_fence(match)

    converted = _CODE.sub(closed_code, source)
    if converted != source:
        converted = converted.strip()
    return strip_html_metadata(converted)


def normalize_page(elements: list[dict[str, Any]]) -> str:
    return "\n\n".join(normalize_element(element) for element in elements)
