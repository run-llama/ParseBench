"""Shared layout parsing utilities for LLM-based parse_with_layout providers.

These utilities are used by Google, OpenAI, and Anthropic providers that
produce layout-annotated output using <div data-bbox="..." data-label="...">
HTML wrappers with the Core11 label set.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout-annotated prompts (Core11 label set)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LAYOUT = (
    "You are a document parser. Your task is to convert "
    "document images to clean, well-structured markdown."
    "\n\nGuidelines:\n"
    "- Preserve the document structure "
    "(headings, paragraphs, lists, tables)\n"
    "- Convert tables to HTML format "
    "(<table>, <tr>, <th>, <td>)\n"
    "- For existing tables in the document: use colspan "
    "and rowspan attributes to preserve merged cells "
    "and hierarchical headers\n"
    "- For charts/graphs being converted to tables: use "
    "flat combined column headers (e.g., "
    '"Primary 2015" not separate rows) so each data '
    "cell's row contains all its labels\n"
    "- Describe images/figures briefly in square brackets "
    "like [Figure: description]\n"
    "- Preserve any code blocks with appropriate syntax "
    "highlighting\n"
    "- Maintain reading order (left-to-right, "
    "top-to-bottom for Western documents)\n"
    "- Do not add commentary or explanations "
    "- only output the parsed content"
    "\n\n"
    "Additionally, wrap each layout element in a <div> tag with:\n"
    '- data-bbox="[x1, y1, x2, y2]" — bounding box in normalized 0-1000 '
    "coordinates where x is horizontal (left edge = 0, right edge = 1000) "
    "and y is vertical (top = 0, bottom = 1000). "
    "x1,y1 is the top-left corner and x2,y2 is the bottom-right corner.\n"
    '- data-label="<category>" — one of: Caption, Footnote, Formula, '
    "List-item, Page-footer, Page-header, Picture, Section-header, "
    "Table, Text, Title\n\n"
    "Place elements in reading order. Every piece of content must be "
    "inside exactly one <div> wrapper."
)

USER_PROMPT_LAYOUT = (
    "Parse this document page and output its content as "
    "clean markdown, with each layout element wrapped in a "
    '<div data-bbox="[x1,y1,x2,y2]" data-label="Category"> tag. '
    "Use HTML tables for any tabular data. "
    "For charts/graphs, use flat combined column headers. "
    "Output ONLY the parsed content with div wrappers, "
    "no explanations."
)

# ---------------------------------------------------------------------------
# Gemini-specific layout prompts — use native [y_min, x_min, y_max, x_max]
# format to avoid intermittent coordinate inversion when asking for [x1,y1,x2,y2].
# Callers must convert with swap_gemini_bbox() after parse_layout_blocks().
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_LAYOUT_GEMINI = SYSTEM_PROMPT_LAYOUT.replace(
    '"[x1, y1, x2, y2]" — bounding box in normalized 0-1000 '
    "coordinates where x is horizontal (left edge = 0, right edge = 1000) "
    "and y is vertical (top = 0, bottom = 1000). "
    "x1,y1 is the top-left corner and x2,y2 is the bottom-right corner.",
    '"[y_min, x_min, y_max, x_max]" — bounding box in normalized 0-1000 '
    "coordinates where x is horizontal (left edge = 0, right edge = 1000) "
    "and y is vertical (top = 0, bottom = 1000). "
    "The order is [y_min, x_min, y_max, x_max].",
)

USER_PROMPT_LAYOUT_GEMINI = USER_PROMPT_LAYOUT.replace(
    "[x1,y1,x2,y2]",
    "[y_min,x_min,y_max,x_max]",
)


def swap_gemini_bbox(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini native [y_min, x_min, y_max, x_max] to [x1, y1, x2, y2]."""
    for item in items:
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            y_min, x_min, y_max, x_max = bbox
            item["bbox"] = [x_min, y_min, x_max, y_max]
    return items


# Label mapping (case-insensitive raw label -> canonical label string)
LABEL_MAP: dict[str, str] = {
    "caption": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "list-item": "List-item",
    "list_item": "List-item",
    "page-footer": "Page-footer",
    "page_footer": "Page-footer",
    "page-header": "Page-header",
    "page_header": "Page-header",
    "picture": "Picture",
    "figure": "Picture",
    "section-header": "Section-header",
    "section_header": "Section-header",
    "table": "Table",
    "text": "Text",
    "title": "Title",
}


def split_pdf_to_pages(pdf_path: str) -> list[tuple[bytes, int, int]]:
    """Split a PDF into single-page PDF bytes.

    Returns a list of (pdf_bytes, width_px, height_px) tuples, one per page.
    Width/height are at 72 DPI (PDF points).
    """
    import fitz  # PyMuPDF

    src = fitz.open(pdf_path)
    results: list[tuple[bytes, int, int]] = []
    for page_num in range(len(src)):
        page = src[page_num]
        rect = page.rect
        # Create a single-page PDF in memory
        dst = fitz.open()
        dst.insert_pdf(src, from_page=page_num, to_page=page_num)
        pdf_bytes = dst.tobytes()
        dst.close()
        results.append((pdf_bytes, int(rect.width), int(rect.height)))
    src.close()
    return results


def parse_layout_blocks(content: str) -> list[dict[str, Any]]:
    """Parse <div data-bbox="..." data-label="...">content</div> blocks.

    Handles both attribute orderings. Returns list of dicts with
    'bbox' (list[float]), 'label' (str), and 'text' (str) keys.
    """
    blocks: list[dict[str, Any]] = []

    # Match opening div with both attribute orders
    pattern_bbox_first = re.compile(
        r'<div\s+[^>]*?data-bbox=["\'](\[[^\]]+\])["\'][^>]*?data-label=["\']([^"\']+)["\'][^>]*?>'
        r"([\s\S]*?)</div>",
        re.IGNORECASE,
    )
    pattern_label_first = re.compile(
        r'<div\s+[^>]*?data-label=["\']([^"\']+)["\'][^>]*?data-bbox=["\'](\[[^\]]+\])["\'][^>]*?>'
        r"([\s\S]*?)</div>",
        re.IGNORECASE,
    )

    # Collect all matches with their start positions, then sort by
    # position so mixed attribute orderings preserve document order.
    raw_matches: list[tuple[int, str, str, str]] = []  # (pos, bbox_str, label, text)

    for match in pattern_bbox_first.finditer(content):
        raw_matches.append((match.start(), match.group(1), match.group(2), match.group(3)))

    for match in pattern_label_first.finditer(content):
        raw_matches.append((match.start(), match.group(2), match.group(1), match.group(3)))

    raw_matches.sort(key=lambda m: m[0])

    seen_positions: set[int] = set()
    for pos, bbox_str, label, text in raw_matches:
        if pos in seen_positions:
            continue  # skip duplicate from overlapping patterns
        seen_positions.add(pos)
        try:
            bbox = json.loads(bbox_str)
            if isinstance(bbox, list) and len(bbox) == 4:
                blocks.append({"bbox": bbox, "label": label, "text": text.strip()})
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse bbox: {bbox_str}")

    return blocks


def items_to_markdown(items: list[dict[str, Any]]) -> str:
    """Assemble clean markdown from parsed layout items."""
    parts: list[str] = []
    for item in items:
        label = item.get("label", "").lower()
        text = item.get("text", "")
        if not text:
            continue
        if label == "title":
            parts.append(f"# {text}")
        elif label in ("section-header", "section_header"):
            parts.append(f"## {text}")
        elif label == "formula":
            parts.append(f"$$\n{text}\n$$")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def build_layout_pages(
    items: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    markdown: str,
    page_number: int = 1,
) -> list[ParseLayoutPageIR]:
    """Convert parsed layout blocks to ParseLayoutPageIR.

    Args:
        items: Parsed layout blocks from ``parse_layout_blocks``.
        image_width: Page image width in pixels.
        image_height: Page image height in pixels.
        markdown: Page markdown content.
        page_number: 1-indexed page number.
    """
    if not items or not image_width or not image_height:
        return []

    layout_items: list[LayoutItemIR] = []
    for item in items:
        bbox = item.get("bbox", [])
        label_raw = item.get("label", "text")
        text = item.get("text", "")

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Convert from 0-1000 [x1,y1,x2,y2] to normalized [0,1] COCO [x,y,w,h]
        nx = x1 / 1000.0
        ny = y1 / 1000.0
        nw = (x2 - x1) / 1000.0
        nh = (y2 - y1) / 1000.0

        label = LABEL_MAP.get(label_raw.lower(), "Text")
        seg = LayoutSegmentIR(x=nx, y=ny, w=nw, h=nh, confidence=1.0, label=label)

        norm_label = label_raw.lower()
        if norm_label == "table":
            item_type = "table"
        elif norm_label in ("picture", "figure"):
            item_type = "image"
        else:
            item_type = "text"

        layout_items.append(LayoutItemIR(type=item_type, value=text, bbox=seg, layout_segments=[seg]))

    if not layout_items:
        return []

    return [
        ParseLayoutPageIR(
            page_number=page_number,
            width=float(image_width),
            height=float(image_height),
            md=markdown,
            items=layout_items,
        )
    ]
