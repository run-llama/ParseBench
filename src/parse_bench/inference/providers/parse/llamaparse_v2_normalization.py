"""Shared normalization helpers for LlamaParse V2 SDK and local cli2 outputs."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, Literal

from llama_cloud.types import (
    CodeItem,
    FooterItem,
    HeaderItem,
    HeadingItem,
    ImageItem,
    LinkItem,
    ListItem,
    TableItem,
    TextItem,
)
from llama_cloud.types.parsing_get_response import (
    ItemsPage,
    ItemsPageStructuredResultPage,
    Metadata,
    MetadataPage,
    ParsingGetResponse,
    Text,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from parse_bench.layout_label_mapping import (
    LLAMAPARSE_PICTURE_TYPE_LABELS,
    UnknownRawLayoutLabelError,
    detect_llamaparse_label_version,
    map_llamaparse_raw_label_to_canonical,
    normalize_picture_type,
)
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)

JsonItem = HeaderItem | FooterItem | TableItem | ListItem | CodeItem | HeadingItem | ImageItem | LinkItem | TextItem

logger = logging.getLogger(__name__)

_CANONICAL_LABEL_VALUES: frozenset[str] = frozenset(label.value for label in CanonicalLabel)
_CHECKBOX_RAW_LABELS = frozenset({"checkbox-selected", "checkbox-unselected"})
_CHECKBOX_CANONICAL_VALUES = frozenset(
    {
        CanonicalLabel.CHECKBOX_SELECTED.value,
        CanonicalLabel.CHECKBOX_UNSELECTED.value,
    }
)
_CHECKBOX_SELECTED_RE = re.compile(r"^\[[xX]\]$")
_CHECKBOX_UNSELECTED_RE = re.compile(r"^\[\s\]$")

# SDK item types that are known to be non-layout (container wrappers or inline
# structure). Dropping them silently — they're not errors, just not layout
# elements. Distinct from "unmappable" which should still warn.
# - ``list`` is a container that wraps ``ListItem`` children; the children
#   arrive as separate items in the layout stream, not the container itself.
# - ``link`` is inline structure, not a standalone layout element.
_NON_LAYOUT_ITEM_TYPES: frozenset[str] = frozenset({"list", "link"})
# Mark-scope attribute value, set on Checkbox-* items so the evaluator's
# mark-scope dispatch (``_is_region_scope_checkbox``) scores tight-glyph
# geometry instead of loose region bboxes. Same literal referenced in
# ``evaluation/metrics/parse/layout_detection.py`` and GT test_rules.
_SCOPE_MARK = "mark"

# LlamaParse figure classification lands in the markdown as
# ``![label: image description](path)`` alt-text prefixes; signatures use the
# linkless ``[signature: legible text]`` form.
_IMAGE_ALT_TEXT_RE = re.compile(r"!\[([^\][]*)\]\([^)]*\)")
_SIGNATURE_MARKDOWN_RE = re.compile(r"(?<!!)\[\s*signature\s*:", re.IGNORECASE)


def extract_picture_type_from_markdown(markdown: str) -> str | None:
    """Extract the normalized figure-classifier label from item markdown.

    Reads the first image's ``label: description`` alt-text prefix. A colonless
    alt text is treated as a plain description unless it is exactly a known
    classifier label; ``[signature: ...]`` spans map to ``signature``.
    """
    if not markdown:
        return None
    alt_match = _IMAGE_ALT_TEXT_RE.search(markdown)
    if alt_match:
        alt_text = alt_match.group(1)
        if ":" in alt_text:
            return normalize_picture_type(alt_text.split(":", 1)[0])
        bare_label = normalize_picture_type(alt_text)
        if bare_label is not None and alt_text.strip().lower().replace(" ", "_") in LLAMAPARSE_PICTURE_TYPE_LABELS:
            return bare_label
        return None
    if _SIGNATURE_MARKDOWN_RE.search(markdown):
        return "signature"
    return None


class StructuredResultPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_number: int
    items: list[JsonItem]
    page_width: float
    page_height: float
    success: Literal[True] = True


class FailedStructuredPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_number: int
    error: str
    success: Literal[False] = False


class StructuredResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pages: list[StructuredResultPage | FailedStructuredPage]


class MarkdownPage(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    page_number: int
    markdown: str = Field(default="", alias="md")
    header: str = Field(default="", alias="pageHeaderMarkdown")
    footer: str = Field(default="", alias="pageFooterMarkdown")
    printed_page_number: str = Field(default="", alias="printedPageNumber")

    @field_validator("markdown", "header", "footer", "printed_page_number", mode="before")
    @classmethod
    def coalesce_nullable_page_text(cls, value: Any) -> Any:
        """V2 emits JSON null for absent optional page text fields."""
        return "" if value is None else value


class MarkdownResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pages: list[MarkdownPage]


def flatten_v2_items(
    items: Sequence[object],
    output_tables_as_markdown: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], list[str]]:
    """Flatten V2 container items and normalize bbox payloads."""
    page_items: list[dict[str, Any]] = []
    item_markdowns: list[str] = []
    item_texts: list[str] = []
    header_markdowns: list[str] = []
    footer_markdowns: list[str] = []

    for item in items:
        if not isinstance(
            item,
            (
                TextItem,
                HeadingItem,
                ListItem,
                CodeItem,
                TableItem,
                ImageItem,
                LinkItem,
                HeaderItem,
                FooterItem,
            ),
        ):
            continue

        if isinstance(item, (ListItem, HeaderItem, FooterItem)):
            if item.md:
                item_markdowns.append(item.md)
                if isinstance(item, HeaderItem):
                    header_markdowns.append(item.md)
                elif isinstance(item, FooterItem):
                    footer_markdowns.append(item.md)
            child_items, child_mds, child_texts, child_headers, child_footers = flatten_v2_items(
                item.items,
                output_tables_as_markdown,
            )
            page_items.extend(child_items)
            item_markdowns.extend(child_mds)
            item_texts.extend(child_texts)
            header_markdowns.extend(child_headers)
            footer_markdowns.extend(child_footers)
            continue

        item_data: dict[str, Any] = {"type": item.type}

        if isinstance(item, TableItem):
            table_content = item.md if output_tables_as_markdown else (item.html or item.md)
            if table_content:
                item_markdowns.append(table_content)
            # Preserve md on table item_data so that parse_pred_blocks can
            # slice per-segment text using startIndex/endIndex.
            if item.md:
                item_data["md"] = item.md
        elif isinstance(item, LinkItem):
            # Use plain display text instead of linkified markdown so that
            # "www.tdi.texas.gov" stays as-is rather than becoming
            # "[www.tdi.texas.gov](http://www.tdi.texas.gov)".
            if item.text:
                item_markdowns.append(item.text)
        elif item.md:
            item_markdowns.append(item.md)

        if isinstance(item, (TextItem, HeadingItem, CodeItem)):
            item_data["value"] = item.value
            item_texts.append(item.value)

        # Preserve md on ALL items that may have layoutAwareBbox segments so
        # parse_pred_blocks can slice text correctly using startIndex/endIndex
        # (which are computed relative to the md field including markdown
        # formatting, not the stripped value field).
        if item.md and item.bbox:
            item_data["md"] = item.md

        if item.bbox:
            first_bbox = item.bbox[0]
            item_data["bBox"] = {
                "x": first_bbox.x,
                "y": first_bbox.y,
                "w": first_bbox.w,
                "h": first_bbox.h,
                "confidence": first_bbox.confidence,
                "label": first_bbox.label,
            }
            item_data["layoutAwareBbox"] = [
                {
                    "x": bbox.x,
                    "y": bbox.y,
                    "w": bbox.w,
                    "h": bbox.h,
                    "confidence": bbox.confidence,
                    "label": bbox.label,
                    "startIndex": bbox.start_index,
                    "endIndex": bbox.end_index,
                }
                for bbox in item.bbox
            ]

        page_items.append(item_data)

    return page_items, item_markdowns, item_texts, header_markdowns, footer_markdowns


def _build_page(
    *,
    page_number: int,
    items: Sequence[object],
    output_tables_as_markdown: bool,
    page_width: float | None = None,
    page_height: float | None = None,
    include_items: bool = True,
    md_fallback: str = "",
    text_fallback: str = "",
    header: str = "",
    footer: str = "",
    printed_page_number: str = "",
    orientation: int | None = None,
) -> dict[str, Any]:
    page_data: dict[str, Any] = {"page": page_number}

    if page_width is not None:
        page_data["width"] = page_width
    if page_height is not None:
        page_data["height"] = page_height

    if include_items:
        (
            page_items,
            item_markdowns,
            item_texts,
            inferred_headers,
            inferred_footers,
        ) = flatten_v2_items(items, output_tables_as_markdown)
        page_data["items"] = page_items
        if item_markdowns:
            page_data["md"] = "\n\n".join(item_markdowns)
        if item_texts:
            page_data["text"] = "\n\n".join(item_texts)

        # Why: V2 SDK responses commonly expand only items/text/metadata
        # (no markdown expansion), so page header/footer markdown would
        # otherwise be dropped and is_header/is_footer rules fail.
        if not header and inferred_headers:
            page_data["pageHeaderMarkdown"] = "\n\n".join(inferred_headers)
            logger.debug(
                "Inferred pageHeaderMarkdown from HeaderItem(s): page=%s count=%s",
                page_number,
                len(inferred_headers),
            )
        if not footer and inferred_footers:
            page_data["pageFooterMarkdown"] = "\n\n".join(inferred_footers)
            logger.debug(
                "Inferred pageFooterMarkdown from FooterItem(s): page=%s count=%s",
                page_number,
                len(inferred_footers),
            )

    if "md" not in page_data and md_fallback:
        page_data["md"] = md_fallback
    if "text" not in page_data and text_fallback:
        page_data["text"] = text_fallback

    if header:
        page_data["pageHeaderMarkdown"] = header
    if footer:
        page_data["pageFooterMarkdown"] = footer
    if printed_page_number:
        page_data["printedPageNumber"] = printed_page_number
    if orientation is not None:
        page_data["original_orientation_angle"] = orientation

    return page_data


def build_pages_from_cli2_v2_sidecars(
    *,
    items_payload: Any,
    output_tables_as_markdown: bool,
    md_payload: Any | None = None,
    text_payload: Any | None = None,
    metadata_payload: Any | None = None,
) -> list[dict[str, Any]]:
    """Build bench pages from local cli2 V2 sidecar payloads."""
    try:
        structured = StructuredResult.model_validate(items_payload)
        markdown_pages = MarkdownResult.model_validate(md_payload).pages if md_payload is not None else []
        text_pages = Text.model_validate(text_payload).pages if text_payload is not None else []
        metadata_pages = Metadata.model_validate(metadata_payload).pages if metadata_payload is not None else []
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    items_by_page = {page.page_number: page for page in structured.pages}
    md_by_page = {page.page_number: page for page in markdown_pages}
    text_by_page = {page.page_number: page for page in text_pages}
    metadata_by_page = {page.page_number: page for page in metadata_pages}

    page_numbers = set(items_by_page) | set(md_by_page) | set(text_by_page) | set(metadata_by_page)
    pages: list[dict[str, Any]] = []

    for page_number in sorted(page_numbers):
        items_page = items_by_page.get(page_number)
        md_page = md_by_page.get(page_number)
        text_page = text_by_page.get(page_number)
        metadata_page = metadata_by_page.get(page_number)

        md_fallback = md_page.markdown if md_page else ""
        text_fallback = text_page.text if text_page else ""
        header = md_page.header if md_page else ""
        footer = md_page.footer if md_page else ""
        printed_page_number = md_page.printed_page_number if md_page else ""
        orientation = metadata_page.original_orientation_angle if metadata_page else None

        if items_page is None:
            pages.append(
                _build_page(
                    page_number=page_number,
                    items=[],
                    include_items=False,
                    output_tables_as_markdown=output_tables_as_markdown,
                    md_fallback=md_fallback,
                    text_fallback=text_fallback,
                    header=header,
                    footer=footer,
                    printed_page_number=printed_page_number,
                    orientation=orientation,
                )
            )
            continue

        if isinstance(items_page, FailedStructuredPage):
            failed_page: dict[str, Any] = {"page": page_number}
            if text_fallback:
                failed_page["text"] = text_fallback
            if md_fallback:
                failed_page["md"] = md_fallback
            if header:
                failed_page["pageHeaderMarkdown"] = header
            if footer:
                failed_page["pageFooterMarkdown"] = footer
            if printed_page_number:
                failed_page["printedPageNumber"] = printed_page_number
            if orientation is not None:
                failed_page["original_orientation_angle"] = orientation
            pages.append(failed_page)
            continue

        pages.append(
            _build_page(
                page_number=page_number,
                items=items_page.items,
                output_tables_as_markdown=output_tables_as_markdown,
                page_width=items_page.page_width,
                page_height=items_page.page_height,
                md_fallback=md_fallback,
                text_fallback=text_fallback,
                header=header,
                footer=footer,
                printed_page_number=printed_page_number,
                orientation=orientation,
            )
        )

    return pages


def build_pages_from_sdk_expansions(
    *,
    items_pages: Sequence[ItemsPage],
    text_by_page: dict[int, str] | None,
    metadata_by_page: dict[int, MetadataPage] | None,
    output_tables_as_markdown: bool,
    num_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Build bench pages from SDK expansion payloads."""
    text_map = text_by_page or {}
    metadata_map = metadata_by_page or {}
    total_pages = num_pages if num_pages is not None else max(len(items_pages), len(text_map), 1)

    pages: list[dict[str, Any]] = []
    for page_number in range(1, total_pages + 1):
        items_page = items_pages[page_number - 1] if page_number - 1 < len(items_pages) else None
        text_fallback = text_map.get(page_number, "")
        metadata_page = metadata_map.get(page_number)
        orientation = metadata_page.original_orientation_angle if metadata_page else None

        if not isinstance(items_page, ItemsPageStructuredResultPage):
            page_data: dict[str, Any] = {"page": page_number}
            if text_fallback:
                page_data["text"] = text_fallback
            if orientation is not None:
                page_data["original_orientation_angle"] = orientation
            pages.append(page_data)
            continue

        pages.append(
            _build_page(
                page_number=page_number,
                items=items_page.items,
                output_tables_as_markdown=output_tables_as_markdown,
                page_width=items_page.page_width,
                page_height=items_page.page_height,
                text_fallback=text_fallback,
                orientation=orientation,
            )
        )

    return pages


def build_pages_from_sdk_response_payload(
    *,
    raw_payload: Any,
    output_tables_as_markdown: bool,
) -> list[dict[str, Any]]:
    """Build normalized pages from V2 SDK raw payload or legacy normalized payload."""
    if not isinstance(raw_payload, dict):
        return []

    pages_payload = raw_payload.get("pages")
    if isinstance(pages_payload, list) and _looks_like_normalized_pages(pages_payload):
        return pages_payload

    try:
        result = ParsingGetResponse.model_validate(raw_payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    items_pages: list[ItemsPage] = result.items.pages if result.items is not None else []

    text_by_page: dict[int, str] = {}
    if result.text is not None:
        for text_page in result.text.pages:
            text_by_page[text_page.page_number] = text_page.text

    metadata_by_page: dict[int, MetadataPage] = {}
    if result.metadata is not None:
        for metadata_page in result.metadata.pages:
            metadata_by_page[metadata_page.page_number] = metadata_page

    num_pages = max(len(items_pages), len(text_by_page), len(metadata_by_page), 1)
    return build_pages_from_sdk_expansions(
        items_pages=items_pages,
        text_by_page=text_by_page,
        metadata_by_page=metadata_by_page,
        output_tables_as_markdown=output_tables_as_markdown,
        num_pages=num_pages,
    )


def build_pages_from_cli2_raw_payload(
    *,
    raw_payload: Any,
    output_tables_as_markdown: bool,
) -> list[dict[str, Any]]:
    """Build normalized pages from cli2 raw payload (new raw or legacy raw)."""
    if not isinstance(raw_payload, dict):
        return []

    pages_payload = raw_payload.get("pages")
    if isinstance(pages_payload, list) and _looks_like_normalized_pages(pages_payload):
        return pages_payload

    items_payload = raw_payload.get("v2_items", raw_payload)
    md_payload = raw_payload.get("v2_md")
    text_payload = raw_payload.get("v2_txt")
    metadata_payload = raw_payload.get("v2_metadata")
    return build_pages_from_cli2_v2_sidecars(
        items_payload=items_payload,
        output_tables_as_markdown=output_tables_as_markdown,
        md_payload=md_payload,
        text_payload=text_payload,
        metadata_payload=metadata_payload,
    )


def build_pages_from_v1_raw_payload(raw_payload: Any) -> list[dict[str, Any]]:
    """Build normalized pages from V1 raw payload (supports legacy and new raw dumps)."""
    if not isinstance(raw_payload, dict):
        return []

    pages_payload = raw_payload.get("pages")
    if not isinstance(pages_payload, list):
        return []

    normalized_pages: list[dict[str, Any]] = []
    for page_index, page_data in enumerate(pages_payload):
        if not isinstance(page_data, dict):
            continue
        page_copy = dict(page_data)
        if "page" not in page_copy:
            page_number = page_copy.get("page_number")
            if isinstance(page_number, int) and page_number > 0:
                page_copy["page"] = page_number
            else:
                page_copy["page"] = page_index + 1
        normalized_pages.append(page_copy)
    return normalized_pages


def extract_job_id_from_raw_payload(raw_payload: Any) -> str | None:
    """Extract job id from raw payload across old and new payload variants."""
    if not isinstance(raw_payload, dict):
        return None

    direct_job_id = raw_payload.get("job_id")
    if isinstance(direct_job_id, str) and direct_job_id:
        return direct_job_id

    job = raw_payload.get("job")
    if isinstance(job, dict):
        job_id = job.get("id")
        if isinstance(job_id, str) and job_id:
            return job_id

    return None


def build_layout_pages_from_pages_payload(pages_payload: Any) -> list[ParseLayoutPageIR]:
    """Build typed ParseLayoutPageIR entries from normalized pages payload."""
    raw_pages = pages_payload if isinstance(pages_payload, list) else []
    label_version = detect_llamaparse_label_version(_collect_pages_payload_labels(raw_pages))
    layout_pages: list[ParseLayoutPageIR] = []

    for page_index, page_data in enumerate(raw_pages):
        if not isinstance(page_data, dict):
            continue
        page_candidate = dict(page_data)
        if "page" not in page_candidate and "page_number" not in page_candidate:
            page_candidate["page"] = page_index + 1
        try:
            layout_page = ParseLayoutPageIR.model_validate(page_candidate)
        except ValidationError:
            continue
        layout_page = _canonicalize_layout_page_item_types(layout_page, label_version=label_version)
        layout_pages.append(layout_page)

    layout_pages = _synthesize_checkbox_mark_items(layout_pages)
    layout_pages.sort(key=lambda page: page.page_number)
    return layout_pages


def _collect_pages_payload_labels(raw_pages: Sequence[Any]) -> list[str]:
    """Collect raw ``layoutAwareBbox`` labels across pages for version detection."""
    labels: list[str] = []
    for page in raw_pages:
        if not isinstance(page, dict):
            continue
        for item in page.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            for bbox in item.get("layoutAwareBbox", []) or []:
                if isinstance(bbox, dict) and isinstance(bbox.get("label"), str):
                    labels.append(bbox["label"])
    return labels


def _canonicalize_layout_page_item_types(
    layout_page: ParseLayoutPageIR,
    *,
    label_version: str,
) -> ParseLayoutPageIR:
    """Rewrite each ``LayoutItemIR.type`` to its Canonical17 string.

    LlamaParse V2 SDK/cli2 outputs keep raw JsonItem-level types like
    ``"text"`` / ``"heading"`` / ``"table"``. The parse-side layout metrics
    (see ``evaluation/metrics/parse/layout_detection.py``) filter predictions
    by Canonical17 class names, so preserving the raw types produces zero
    matched predictions despite populated ``layout_pages[*].items``. Items
    whose type does not map to Canonical17 are dropped — they cannot
    participate in layout detection scoring anyway.

    This is the sibling of the checkbox-mark synthesis
    (``_synthesize_checkbox_mark_items``) for ``[x]`` / ``[ ]`` tokens: that
    step promotes mark spans into Canonical17 checkbox items, while this
    helper canonicalizes the remaining types.
    """
    canonicalized_items: list[LayoutItemIR] = []
    for item in layout_page.items:
        canonicalized_items.extend(_canonicalize_single_item(item, label_version=label_version))
    return layout_page.model_copy(update={"items": canonicalized_items})


def _canonicalize_single_item(
    item: LayoutItemIR,
    *,
    label_version: str,
) -> list[LayoutItemIR]:
    """Canonicalize one ``LayoutItemIR`` using per-segment labels.

    Label source of truth: ``layout_segments[*].label`` (the detector
    ``bbox.label``). The SDK JsonItem ``item.type`` is a coarse container
    (``"text"`` covers headers/footers/captions; ``"image"`` covers
    pictures/charts) and cannot drive per-box canonical classification —
    using it depresses per-class mAP/F1 vs the legacy adapter path.

    When all segments resolve to the same canonical, keep the original
    item (with updated type). When segments disagree, split into one
    ``LayoutItemIR`` per segment so each detection scores with its own
    ``class_name``. Items without any resolvable segment label are dropped.
    """
    # Single source to iterate: prefer layout_segments, else the top-level bbox.
    segments: list[LayoutSegmentIR] = (
        list(item.layout_segments) if item.layout_segments else ([item.bbox] if item.bbox is not None else [])
    )
    resolved: list[tuple[LayoutSegmentIR, str, dict[str, str]]] = [
        (seg, resolution[0], resolution[1])
        for seg in segments
        if seg.label and (resolution := _canonicalize_item_type(seg.label, label_version=label_version)) is not None
    ]
    if not resolved:
        return []

    unique_canonicals = {canonical for _, canonical, _ in resolved}
    if len(unique_canonicals) == 1:
        (only,) = unique_canonicals
        resolved_segs = [seg for seg, _, _ in resolved]
        updates: dict[str, Any] = {**_canonical_updates(item, only, label_attrs=resolved[0][2])}
        if len(resolved_segs) != len(segments):
            updates["layout_segments"] = resolved_segs
        return [item.model_copy(update=updates)]
    # Segments disagree: split so each detection scores with its own class.
    return [
        item.model_copy(
            update={
                **_canonical_updates(item, canonical, label_attrs=label_attrs),
                "layout_segments": [seg],
                "bbox": seg,
            }
        )
        for seg, canonical, label_attrs in resolved
    ]


def _canonical_updates(
    item: LayoutItemIR,
    canonical: str,
    *,
    label_attrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build ``model_copy(update=...)`` payload for a canonicalized item.

    Stamps ``scope=mark`` on Checkbox-* so the downstream text-token
    synthesizer (``_synthesize_checkbox_mark_items``) dedupes against
    segment-label-derived checkboxes via the same key. Raw-label semantic
    attributes (e.g. ``picture_type=chart`` from a ``chart`` bbox label) are
    carried onto the item, and for Picture items the figure-classifier label
    parsed from the markdown alt text overrides the coarse raw-label value.
    """
    attributes: dict[str, str] = {**(item.attributes or {})}
    if label_attrs:
        attributes.update(label_attrs)
    if canonical == CanonicalLabel.PICTURE.value:
        markdown_picture_type = extract_picture_type_from_markdown(item.md)
        if markdown_picture_type is not None:
            attributes["picture_type"] = markdown_picture_type
    if canonical in _CHECKBOX_CANONICAL_VALUES:
        attributes["scope"] = _SCOPE_MARK
    updates: dict[str, Any] = {"type": canonical}
    if attributes:
        updates["attributes"] = attributes
    return updates


def _canonicalize_item_type(raw_type: str, *, label_version: str) -> tuple[str, dict[str, str]] | None:
    """Map a raw LlamaParse layout item type to its Canonical17 value + attrs.

    Returns None for unknown types so callers can drop them. Canonical17
    values pass through unchanged; raw JsonItem types (``"text"``,
    ``"heading"`` …) resolve via the shared label mapping, whose semantic
    attributes (``picture_type``, ``title_level`` …) are returned alongside
    the canonical value.

    If the detected ``label_version`` mapping misses, falls back to the other
    version before giving up — SDK ``JsonItem`` types like ``"code"`` can
    appear even when ``layoutAwareBbox`` labels look V2-only, so single-version
    lookup silently dropped those items pre-fix. Unmappable types are logged
    at WARNING level so future SDK type drift surfaces instead of disappearing.
    """
    candidate = (raw_type or "").strip()
    if not candidate:
        return None
    if candidate in _CANONICAL_LABEL_VALUES:
        return candidate, {}
    lowered = candidate.lower()
    if lowered in _NON_LAYOUT_ITEM_TYPES:
        return None
    fallback_version = "v3" if label_version == "v2" else "v2"
    for version in (label_version, fallback_version):
        try:
            canonical, attrs = map_llamaparse_raw_label_to_canonical(lowered, label_version=version)
        except UnknownRawLayoutLabelError:
            continue
        return canonical.value, attrs
    logger.warning(
        "Dropping LlamaParse layout item with unmappable type %r (label_version=%r)",
        raw_type,
        label_version,
    )
    return None


def layout_pages_to_legacy_pages_payload(
    layout_pages: Sequence[ParseLayoutPageIR],
    *,
    include_bbox_segment_fallback: bool = True,
) -> list[dict[str, Any]]:
    """Convert typed layout pages into legacy pages payload consumed by layout extractors."""
    legacy_pages: list[dict[str, Any]] = []
    for page in sorted(layout_pages, key=lambda p: p.page_number):
        page_data: dict[str, Any] = {
            "page": page.page_number,
            "items": [],
        }
        if page.width is not None:
            page_data["width"] = page.width
        if page.height is not None:
            page_data["height"] = page.height
        if page.md:
            page_data["md"] = page.md
        if page.text:
            page_data["text"] = page.text
        if page.page_header_markdown:
            page_data["pageHeaderMarkdown"] = page.page_header_markdown
        if page.page_footer_markdown:
            page_data["pageFooterMarkdown"] = page.page_footer_markdown
        if page.printed_page_number:
            page_data["printedPageNumber"] = page.printed_page_number
        if page.original_orientation_angle is not None:
            page_data["original_orientation_angle"] = page.original_orientation_angle

        items: list[dict[str, Any]] = []
        for item in page.items:
            item_data: dict[str, Any] = {"type": item.type}
            if item.md:
                item_data["md"] = item.md
            if item.html:
                item_data["html"] = item.html
            if item.value:
                item_data["value"] = item.value
            if item.bbox is not None:
                item_data["bBox"] = _segment_to_legacy_bbox(item.bbox, include_span=False)
            if item.layout_segments:
                item_data["layoutAwareBbox"] = [
                    _segment_to_legacy_bbox(segment, include_span=True) for segment in item.layout_segments
                ]
            elif include_bbox_segment_fallback and item.bbox is not None:
                # Preserve legacy fallback behavior where a single bBox can act as segment.
                item_data["layoutAwareBbox"] = [_segment_to_legacy_bbox(item.bbox, include_span=True)]
            items.append(item_data)

        page_data["items"] = items
        legacy_pages.append(page_data)

    return legacy_pages


def build_parse_output_from_pages(
    *,
    pages_payload: Any,
    example_id: str,
    pipeline_name: str,
    job_id: str | None = None,
) -> ParseOutput:
    """Build ParseOutput from normalized page payloads."""
    layout_pages = build_layout_pages_from_pages_payload(pages_payload)
    page_irs: list[PageIR] = []
    for page in layout_pages:
        # Keep section metadata (header/footer/page number) in structured layout_pages only.
        # Do not inject tags into markdown body to avoid duplicated content.
        markdown = page.md or page.text

        page_irs.append(PageIR(page_index=page.page_number - 1, markdown=markdown))

    page_irs.sort(key=lambda page: page.page_index)
    full_markdown = "\n\n---\n\n".join(page.markdown for page in page_irs)

    return ParseOutput(
        task_type="parse",
        example_id=example_id,
        pipeline_name=pipeline_name,
        pages=page_irs,
        layout_pages=layout_pages,
        markdown=full_markdown,
        job_id=job_id,
    )


def _looks_like_normalized_pages(pages_payload: list[Any]) -> bool:
    for page in pages_payload:
        if isinstance(page, dict):
            if "page_number" in page:
                return False
            if "page" in page:
                return True
            if any(
                key in page
                for key in (
                    "md",
                    "text",
                    "width",
                    "height",
                    "pageHeaderMarkdown",
                    "pageFooterMarkdown",
                    "printedPageNumber",
                )
            ):
                return True
            return False
    return False


def _segment_to_legacy_bbox(
    segment: LayoutSegmentIR,
    *,
    include_span: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "x": segment.x,
        "y": segment.y,
        "w": segment.w,
        "h": segment.h,
        "confidence": segment.confidence,
        "label": segment.label,
    }
    if include_span:
        payload["startIndex"] = segment.start_index
        payload["endIndex"] = segment.end_index
    return payload


def _synthesize_checkbox_mark_items(layout_pages: list[ParseLayoutPageIR]) -> list[ParseLayoutPageIR]:
    """Promote checkbox mark spans into standalone mark-scope layout items.

    LlamaParse V2 parse output keeps tight checkbox geometry on text-item
    ``layoutAwareBbox`` spans whose markdown slice is ``[x]`` / ``[ ]``.
    The parse-side layout metrics, however, score ``layout_pages[*].items[*]``
    and need these marks as dedicated ``Checkbox-*`` items with
    ``attributes.scope = "mark"`` so they are not conflated with broader
    region-scope checkbox/form detections.
    """
    if not layout_pages:
        return layout_pages

    raw_labels = [
        str(segment.label)
        for page in layout_pages
        for item in page.items
        for segment in item.layout_segments
        if segment.label
    ]
    label_version = detect_llamaparse_label_version(raw_labels)
    existing_keys = _collect_existing_checkbox_mark_keys(layout_pages)

    enriched_pages: list[ParseLayoutPageIR] = []
    for page in layout_pages:
        page_items: list[LayoutItemIR] = []
        for item in page.items:
            page_items.append(item)
            for synthetic_item in _iter_checkbox_mark_items_for_source_item(
                page_number=page.page_number,
                item=item,
                label_version=label_version,
                existing_keys=existing_keys,
            ):
                page_items.append(synthetic_item)
        enriched_pages.append(page.model_copy(update={"items": page_items}))
    return enriched_pages


def _collect_existing_checkbox_mark_keys(
    layout_pages: Sequence[ParseLayoutPageIR],
) -> set[tuple[int, str, float, float, float, float]]:
    keys: set[tuple[int, str, float, float, float, float]] = set()
    for page in layout_pages:
        for item in page.items:
            if item.type not in _CHECKBOX_CANONICAL_VALUES:
                continue
            if str((item.attributes or {}).get("scope", "")).strip().lower() != "mark":
                continue
            segment = item.layout_segments[0] if item.layout_segments else item.bbox
            if segment is None:
                continue
            keys.add(_checkbox_mark_key(page.page_number, item.type, segment))
    return keys


def _iter_checkbox_mark_items_for_source_item(
    *,
    page_number: int,
    item: LayoutItemIR,
    label_version: str,
    existing_keys: set[tuple[int, str, float, float, float, float]],
) -> list[LayoutItemIR]:
    if not item.md or not item.layout_segments:
        return []

    synthetic_items: list[LayoutItemIR] = []
    for segment in item.layout_segments:
        raw_label = str(segment.label or "").strip().lower()
        if raw_label not in _CHECKBOX_RAW_LABELS:
            continue

        span_text = _slice_segment_markdown(item.md, segment)
        if span_text is None or not _matches_checkbox_mark_token(raw_label, span_text):
            continue

        canonical_label, canonical_attrs = map_llamaparse_raw_label_to_canonical(
            raw_label,
            label_version=label_version,
        )
        checkbox_type = canonical_label.value
        key = _checkbox_mark_key(page_number, checkbox_type, segment)
        if key in existing_keys:
            continue

        existing_keys.add(key)
        synthetic_items.append(
            LayoutItemIR(
                type=checkbox_type,
                md=span_text,
                value=span_text,
                bbox=LayoutSegmentIR.model_validate(segment.model_dump()),
                layout_segments=[LayoutSegmentIR.model_validate(segment.model_dump())],
                score=segment.confidence,
                attributes={**canonical_attrs, "scope": _SCOPE_MARK},
            )
        )
    return synthetic_items


def _slice_segment_markdown(markdown: str, segment: LayoutSegmentIR) -> str | None:
    if segment.start_index is None or segment.end_index is None:
        return None
    start_index = int(segment.start_index)
    end_index = int(segment.end_index)
    if start_index < 0 or end_index < start_index or end_index >= len(markdown):
        return None
    return markdown[start_index : end_index + 1]


def _matches_checkbox_mark_token(raw_label: str, span_text: str) -> bool:
    if raw_label == "checkbox-selected":
        return bool(_CHECKBOX_SELECTED_RE.fullmatch(span_text))
    if raw_label == "checkbox-unselected":
        return bool(_CHECKBOX_UNSELECTED_RE.fullmatch(span_text))
    return False


def _checkbox_mark_key(
    page_number: int,
    checkbox_type: str,
    segment: LayoutSegmentIR,
) -> tuple[int, str, float, float, float, float]:
    return (
        page_number,
        checkbox_type,
        round(segment.x, 4),
        round(segment.y, 4),
        round(segment.w, 4),
        round(segment.h, 4),
    )
