"""Unit tests for LlamaParse layout-item canonicalisation and checkbox-mark synthesis."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from parse_bench.inference.providers.parse.llamaparse_v2_normalization import (
    build_parse_output_from_pages,
    layout_pages_to_legacy_pages_payload,
)
from parse_bench.schemas.parse_output import ParseOutput

_NORMALIZATION_LOGGER = "parse_bench.inference.providers.parse.llamaparse_v2_normalization"


def _make_labeled_bbox(label: str) -> dict[str, Any]:
    return {"x": 0, "y": 0, "w": 1, "h": 1, "label": label}


def test_build_parse_output_from_pages_populates_layout_pages() -> None:
    pages = [
        {
            "page": 1,
            "md": "body",
            "pageHeaderMarkdown": "Header 1",
            "pageFooterMarkdown": "Footer 1",
            "printedPageNumber": "1",
            "original_orientation_angle": 90,
            "items": [
                {
                    "type": "text",
                    "value": "abcdef",
                    "bBox": {"x": 1, "y": 2, "w": 3, "h": 4, "label": "text"},
                    "layoutAwareBbox": [
                        {
                            "x": 1,
                            "y": 2,
                            "w": 3,
                            "h": 4,
                            "label": "doc_title",
                            "startIndex": 1,
                            "endIndex": 3,
                            "confidence": 0.99,
                        }
                    ],
                }
            ],
        }
    ]

    output = build_parse_output_from_pages(
        pages_payload=pages,
        example_id="group/doc",
        pipeline_name="ours_agentic",
        job_id="job_123",
    )

    assert output.job_id == "job_123"
    assert len(output.pages) == 1
    assert output.pages[0].markdown == "body"
    assert len(output.layout_pages) == 1
    assert output.layout_pages[0].page_number == 1
    assert output.layout_pages[0].original_orientation_angle == 90
    assert output.layout_pages[0].page_header_markdown == "Header 1"
    assert output.layout_pages[0].page_footer_markdown == "Footer 1"
    assert output.layout_pages[0].items[0].type == "Title"
    assert output.layout_pages[0].items[0].layout_segments[0].start_index == 1
    assert output.layout_pages[0].items[0].layout_segments[0].end_index == 3

    # ParseOutput should serialize/deserialize with layout_pages; segment
    # labels stay raw even though the item type is canonical.
    reloaded = ParseOutput.model_validate(output.model_dump())
    assert reloaded.layout_pages[0].items[0].type == "Title"
    assert reloaded.layout_pages[0].items[0].layout_segments[0].label == "doc_title"


def test_layout_pages_to_legacy_pages_payload_round_trip() -> None:
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "md": "body",
                "items": [
                    {
                        "type": "table",
                        "md": "<table><tr><td>cell</td></tr></table>",
                        "html": "<table><tr><td>cell</td></tr></table>",
                        "value": "abcdef",
                        "bBox": {"x": 10, "y": 20, "w": 30, "h": 40, "label": "text"},
                        "layoutAwareBbox": [
                            {
                                "x": 10,
                                "y": 20,
                                "w": 30,
                                "h": 40,
                                "label": "text",
                                "startIndex": 0,
                                "endIndex": 2,
                            }
                        ],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    legacy_pages = layout_pages_to_legacy_pages_payload(output.layout_pages)
    assert legacy_pages[0]["page"] == 1
    assert legacy_pages[0]["items"][0]["type"] == "Text"
    assert legacy_pages[0]["items"][0]["md"] == "<table><tr><td>cell</td></tr></table>"
    assert legacy_pages[0]["items"][0]["html"] == "<table><tr><td>cell</td></tr></table>"
    assert legacy_pages[0]["items"][0]["layoutAwareBbox"][0]["label"] == "text"
    assert legacy_pages[0]["items"][0]["layoutAwareBbox"][0]["startIndex"] == 0
    assert legacy_pages[0]["items"][0]["layoutAwareBbox"][0]["endIndex"] == 2


def test_build_parse_output_from_pages_synthesizes_mark_scope_checkbox_items() -> None:
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "md": "Question [X] Yes [ ] No",
                "items": [
                    {
                        "type": "text",
                        "md": "Question [X] Yes [ ] No",
                        "value": "Question [X] Yes [ ] No",
                        "layoutAwareBbox": [
                            {
                                "x": 10,
                                "y": 20,
                                "w": 8,
                                "h": 8,
                                "label": "checkbox-selected",
                                "startIndex": 9,
                                "endIndex": 11,
                                "confidence": 0.91,
                            },
                            {
                                "x": 40,
                                "y": 20,
                                "w": 8,
                                "h": 8,
                                "label": "checkbox-unselected",
                                "startIndex": 17,
                                "endIndex": 19,
                                "confidence": 0.83,
                            },
                        ],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    page_items = output.layout_pages[0].items
    checkbox_items = [item for item in page_items if item.type in {"Checkbox-Selected", "Checkbox-Unselected"}]

    # Segment-label canonicalization emits Checkbox-Selected + Checkbox-Unselected
    # directly (split on mixed canonicals); synthesis dedupes against them via
    # the scope=mark key. Result: 2 items, not 3 (parent Text + 2 synthesized).
    assert len(page_items) == 2
    assert [item.type for item in checkbox_items] == ["Checkbox-Selected", "Checkbox-Unselected"]
    assert [item.attributes for item in checkbox_items] == [{"scope": "mark"}, {"scope": "mark"}]
    assert checkbox_items[0].bbox is not None
    assert checkbox_items[0].bbox.label == "checkbox-selected"
    assert checkbox_items[0].layout_segments[0].start_index == 9
    assert checkbox_items[1].bbox is not None
    assert checkbox_items[1].bbox.label == "checkbox-unselected"


def test_build_parse_output_from_pages_ignores_checkbox_segments_without_checkbox_token() -> None:
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "md": "Question Yes No",
                "items": [
                    {
                        "type": "text",
                        "md": "Question Yes No",
                        "value": "Question Yes No",
                        "layoutAwareBbox": [
                            {
                                "x": 10,
                                "y": 20,
                                "w": 8,
                                "h": 8,
                                "label": "checkbox-selected",
                                "startIndex": 9,
                                "endIndex": 11,
                                "confidence": 0.91,
                            }
                        ],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    # Segment label IS "checkbox-selected" so segment-label canonicalization
    # emits Checkbox-Selected (with scope=mark). Text-based synthesis ignores
    # it because the text span isn't a [x]/[ ] token, but the canonicalizer's
    # read from segment.label is the primary class signal.
    items = output.layout_pages[0].items
    assert [item.type for item in items] == ["Checkbox-Selected"]
    assert items[0].attributes == {"scope": "mark"}


def test_synthesizes_checkbox_marks_from_text_items_without_checkbox_segment_type() -> None:
    """A text item whose ``[x]`` span is a checkbox segment alongside a text
    segment is split on mixed canonicals; the checkbox half carries scope=mark
    and no duplicate synthetic item is appended."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "md": "Question [X] Yes",
                "items": [
                    {
                        "type": "text",
                        "md": "Question [X] Yes",
                        "value": "Question [X] Yes",
                        "layoutAwareBbox": [
                            {"x": 0, "y": 20, "w": 100, "h": 8, "label": "text", "startIndex": 0, "endIndex": 15},
                            {
                                "x": 10,
                                "y": 20,
                                "w": 8,
                                "h": 8,
                                "label": "checkbox-selected",
                                "startIndex": 9,
                                "endIndex": 11,
                                "confidence": 0.91,
                            },
                        ],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    items = output.layout_pages[0].items
    assert [item.type for item in items] == ["Text", "Checkbox-Selected"]
    assert items[1].attributes == {"scope": "mark"}
    assert items[1].layout_segments[0].label == "checkbox-selected"


def test_build_parse_output_canonicalizes_from_segment_labels() -> None:
    """Segment ``label`` (not item.type) is the source of truth for Canonical17."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "md": "body",
                "items": [
                    {"type": "text", "value": "hello", "layoutAwareBbox": [_make_labeled_bbox("text")]},
                    {"type": "heading", "value": "title", "layoutAwareBbox": [_make_labeled_bbox("paragraph_title")]},
                    {"type": "table", "md": "<table></table>", "layoutAwareBbox": [_make_labeled_bbox("table")]},
                    {"type": "image", "layoutAwareBbox": [_make_labeled_bbox("image")]},
                    {"type": "text", "value": "footer", "layoutAwareBbox": [_make_labeled_bbox("footer")]},
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    assert [item.type for item in output.layout_pages[0].items] == [
        "Text",
        "Section-header",
        "Table",
        "Picture",
        "Page-footer",
    ]


def test_build_parse_output_drops_items_with_no_segment_labels() -> None:
    """Items without any resolvable segment label are dropped — canonical class
    cannot be determined from SDK ``item.type`` alone (it's a coarse container
    type). Unlabeled items cannot participate in layout-detection scoring."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {"type": "text", "value": "labeled", "layoutAwareBbox": [_make_labeled_bbox("text")]},
                    {"type": "text", "value": "unlabeled", "bBox": {"x": 0, "y": 0, "w": 1, "h": 1}},
                    {"type": "text", "value": "fantasy", "layoutAwareBbox": [_make_labeled_bbox("fantasy-footer")]},
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    # Only the item whose segment.label maps to Canonical17 survives.
    assert [item.type for item in output.layout_pages[0].items] == ["Text"]


def test_build_parse_output_splits_items_with_mixed_segment_labels() -> None:
    """Multi-segment items whose segments carry different canonical classes
    are split into one LayoutItemIR per segment so each detection scores
    against GT with its own class_name."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {
                        "type": "text",
                        "value": "mixed",
                        "layoutAwareBbox": [
                            {"x": 0, "y": 0, "w": 10, "h": 5, "label": "header"},
                            {"x": 0, "y": 10, "w": 10, "h": 5, "label": "text"},
                        ],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    items = output.layout_pages[0].items
    assert [item.type for item in items] == ["Page-header", "Text"]
    assert [len(item.layout_segments) for item in items] == [1, 1]
    assert items[0].bbox is not None and items[0].bbox.label == "header"
    assert items[1].bbox is not None and items[1].bbox.label == "text"


def test_build_parse_output_preserves_checkbox_mark_synthesis() -> None:
    """Checkbox-mark synthesis still fires alongside type canonicalization."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "md": "Question [X] Yes",
                "items": [
                    {
                        "type": "text",
                        "md": "Question [X] Yes",
                        "value": "Question [X] Yes",
                        "layoutAwareBbox": [
                            {
                                "x": 10,
                                "y": 20,
                                "w": 8,
                                "h": 8,
                                "label": "checkbox-selected",
                                "startIndex": 9,
                                "endIndex": 11,
                                "confidence": 0.91,
                            }
                        ],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    types = [item.type for item in output.layout_pages[0].items]
    # Segment-label canonicalization emits Checkbox-Selected (with scope=mark)
    # directly; synthesis dedupes. Single item — no separate parent Text
    # because the sole segment carries a checkbox label.
    assert types == ["Checkbox-Selected"]
    checkbox = output.layout_pages[0].items[0]
    assert checkbox.attributes == {"scope": "mark"}


def test_canonical_item_types_pass_through_unchanged() -> None:
    """Already-canonical segment labels (e.g. re-normalising a normalised
    payload) resolve to themselves rather than hitting the raw-label tables."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {"type": "Page-header", "value": "hdr", "layoutAwareBbox": [_make_labeled_bbox("Page-header")]},
                    {"type": "Picture", "layoutAwareBbox": [_make_labeled_bbox("Picture")]},
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    assert [item.type for item in output.layout_pages[0].items] == ["Page-header", "Picture"]


def test_v2_code_type_canonicalizes_to_code_label() -> None:
    """SDK ``CodeItem`` emits ``type="code"``. Under V2 label_version detection
    (no V3-only labels present) this previously hit
    ``UnknownRawLayoutLabelError`` and the code block was silently dropped from
    ``layout_pages``.
    """
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "md": "```python\nprint('hi')\n```",
                "items": [
                    {
                        "type": "code",
                        "md": "```python\nprint('hi')\n```",
                        "value": "print('hi')",
                        "bBox": {"x": 0, "y": 0, "w": 100, "h": 50},
                        # layoutAwareBbox uses a V2-only label so the
                        # detector classifies this page as V2.
                        "layoutAwareBbox": [
                            {
                                "x": 0,
                                "y": 0,
                                "w": 100,
                                "h": 50,
                                "label": "algorithm",
                                "startIndex": 0,
                                "endIndex": 11,
                            }
                        ],
                    },
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    assert [item.type for item in output.layout_pages[0].items] == ["Code"]


def test_list_and_link_are_silently_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """``list`` is a container wrapper (its children arrive as separate items)
    and ``link`` is inline structure — neither is a standalone layout element.
    Both are silently skipped (no warning) to distinguish known-non-layout from
    unrecognized SDK drift.
    """
    caplog.set_level(logging.WARNING, logger=_NORMALIZATION_LOGGER)

    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {"type": "text", "value": "kept", "bBox": {"x": 0, "y": 0, "w": 1, "h": 1, "label": "text"}},
                    {
                        "type": "list",
                        "md": "- a\n- b",
                        "bBox": {"x": 0, "y": 0, "w": 10, "h": 10},
                        "layoutAwareBbox": [
                            {
                                "x": 0,
                                "y": 0,
                                "w": 10,
                                "h": 10,
                                "label": "list",
                                "startIndex": 0,
                                "endIndex": 1,
                            }
                        ],
                    },
                    {
                        "type": "link",
                        "md": "[x](http://example.com)",
                        "bBox": {"x": 0, "y": 0, "w": 10, "h": 10, "label": "link"},
                    },
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    # Both non-layout items dropped; only the Text item survives.
    assert [item.type for item in output.layout_pages[0].items] == ["Text"]

    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert not any("Dropping LlamaParse layout item" in rec.getMessage() for rec in warnings), (
        "list/link are known non-layout types and must be dropped silently, not warned"
    )


def test_unknown_raw_label_falls_back_to_v3_when_v2_misses() -> None:
    """When ``detect_llamaparse_label_version`` defaults to V2 but a segment's
    raw ``label`` is only defined in V3 (e.g. ``"caption"``), the fallback
    resolves it via the V3 table instead of silently dropping."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {
                        "type": "text",
                        "md": "Figure 1",
                        "layoutAwareBbox": [{"x": 0, "y": 0, "w": 10, "h": 10, "label": "caption"}],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    assert [item.type for item in output.layout_pages[0].items] == ["Caption"]


def test_truly_unknown_label_is_dropped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Segment labels that neither V2 nor V3 recognize are dropped with a
    WARNING so future SDK label drift is visible in logs instead of silently
    corrupting ``layout_pages``."""
    caplog.set_level(logging.WARNING, logger=_NORMALIZATION_LOGGER)

    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {
                        "type": "text",
                        "value": "kept",
                        "layoutAwareBbox": [{"x": 0, "y": 0, "w": 1, "h": 1, "label": "text"}],
                    },
                    {
                        "type": "text",
                        "value": "dropped",
                        "layoutAwareBbox": [{"x": 0, "y": 0, "w": 1, "h": 1, "label": "fantasy-label"}],
                    },
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    assert [item.type for item in output.layout_pages[0].items] == ["Text"]
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("fantasy-label" in rec.getMessage() for rec in warnings), (
        "expected WARNING mentioning unmappable raw label"
    )


def test_picture_items_get_picture_type_from_alt_text_label() -> None:
    """The `![label: description](path)` alt-text prefix lands as a
    ``picture_type`` attribute, overriding the coarse raw-label value
    (`image` -> picture_type=image) with the classifier's specific class."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {
                        "type": "image",
                        "md": "![bar_chart: quarterly revenue](page_1_img_1.png)",
                        "layoutAwareBbox": [_make_labeled_bbox("image")],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    item = output.layout_pages[0].items[0]
    assert item.type == "Picture"
    assert item.attributes["picture_type"] == "bar_chart"


def test_picture_items_fall_back_to_raw_label_picture_type() -> None:
    """Without an alt-text label the raw bbox label's semantic attribute
    still lands on the item."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {
                        "type": "image",
                        "md": "![](page_1_img_1.png)",
                        "layoutAwareBbox": [_make_labeled_bbox("chart")],
                    },
                    {
                        "type": "image",
                        "md": "![a scanned photo of the site](page_1_img_2.png)",
                        "layoutAwareBbox": [_make_labeled_bbox("image")],
                    },
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    items = output.layout_pages[0].items
    assert [item.attributes.get("picture_type") for item in items] == ["chart", "image"]


def test_signature_markdown_sets_signature_picture_type() -> None:
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {
                        "type": "image",
                        "md": "[signature: J. Smith]",
                        "layoutAwareBbox": [_make_labeled_bbox("image")],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    item = output.layout_pages[0].items[0]
    assert item.attributes["picture_type"] == "signature"


def test_non_picture_items_do_not_get_picture_type_from_markdown() -> None:
    """A text item whose markdown mentions an image keeps its attributes clean."""
    output = build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "items": [
                    {
                        "type": "text",
                        "md": "See ![line_chart: trend](img.png) above.",
                        "value": "See above.",
                        "layoutAwareBbox": [_make_labeled_bbox("text")],
                    }
                ],
            }
        ],
        example_id="group/doc",
        pipeline_name="ours_agentic",
    )

    item = output.layout_pages[0].items[0]
    assert item.type == "Text"
    assert "picture_type" not in item.attributes
