from __future__ import annotations

import pytest

from parse_bench.inference.layout_extraction import (
    extract_all_layouts_from_llamaparse_output,
    extract_layout_from_llamaparse_output,
)
from parse_bench.layout_label_mapping import UnknownRawLayoutLabelError
from parse_bench.schemas.layout_detection_output import LayoutOutput

_HAS_LAYOUT_PAGES = "layout_pages" in LayoutOutput.model_fields
_REQUIRES_LAYOUT_PAGES = pytest.mark.skipif(
    not _HAS_LAYOUT_PAGES,
    reason="LayoutOutput.layout_pages schema field not yet present (see CROSS_REQUESTS)",
)


def _make_raw_output(labels: list[str]) -> dict:
    return {
        "image_width": 1000,
        "image_height": 1000,
        "pages": [
            {
                "page": 1,
                "width": 1000,
                "height": 1000,
                "items": [
                    {
                        "type": "text",
                        "value": "Segmented text",
                        "layoutAwareBbox": [
                            {
                                "x": 100,
                                "y": 100,
                                "w": 200,
                                "h": 120,
                                "confidence": 0.9,
                                "label": labels[0],
                            },
                            {
                                "x": 350,
                                "y": 120,
                                "w": 180,
                                "h": 80,
                                "confidence": 0.8,
                                "label": labels[1],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_llamaparse_layout_extraction_is_segment_level() -> None:
    output = extract_layout_from_llamaparse_output(
        _make_raw_output(["section-header", "text"]),
        page_index=0,
        example_id="doc-1",
        pipeline_name="llamaparse_eval",
    )

    assert output is not None
    assert len(output.predictions) == 2
    assert output.predictions[0].label == "section-header"
    assert output.predictions[1].label == "text"


def test_llamaparse_layout_extraction_raises_on_unknown_label() -> None:
    with pytest.raises(UnknownRawLayoutLabelError, match="Unknown LlamaParse raw layout label"):
        extract_layout_from_llamaparse_output(
            _make_raw_output(["section-header", "unexpected-new-label"]),
            page_index=0,
            example_id="doc-2",
            pipeline_name="llamaparse_eval",
        )


def test_llamaparse_layout_extraction_accepts_heading_with_v2_detection() -> None:
    output = extract_layout_from_llamaparse_output(
        _make_raw_output(["doc_title", "heading"]),
        page_index=0,
        example_id="doc-3",
        pipeline_name="llamaparse_eval",
    )
    assert output is not None
    assert len(output.predictions) == 2
    assert output.predictions[1].label == "heading"
    assert output.predictions[1].provider_metadata.get("label_version") == "v2"


def test_llamaparse_layout_extraction_scales_to_target_dimensions() -> None:
    raw_output = _make_raw_output(["doc_title", "text"])
    raw_output["pages"][0]["width"] = 500
    raw_output["pages"][0]["height"] = 250
    raw_output["pages"][0]["items"][0]["layoutAwareBbox"][0]["x"] = 50
    raw_output["pages"][0]["items"][0]["layoutAwareBbox"][0]["y"] = 25
    raw_output["pages"][0]["items"][0]["layoutAwareBbox"][0]["w"] = 100
    raw_output["pages"][0]["items"][0]["layoutAwareBbox"][0]["h"] = 50

    output = extract_layout_from_llamaparse_output(
        raw_output,
        page_index=0,
        example_id="doc-scale",
        pipeline_name="llamaparse_eval",
        target_width=1000,
        target_height=500,
    )

    assert output is not None
    assert output.image_width == 1000
    assert output.image_height == 500
    assert output.predictions[0].bbox == pytest.approx([100.0, 50.0, 300.0, 150.0])


def test_llamaparse_layout_extraction_detects_v3_label_version() -> None:
    output = extract_layout_from_llamaparse_output(
        _make_raw_output(["section-header", "caption"]),
        page_index=0,
        example_id="doc-v3",
        pipeline_name="llamaparse_eval",
    )
    assert output is not None
    assert len(output.predictions) == 2
    assert output.predictions[0].provider_metadata.get("label_version") == "v3"
    assert output.predictions[1].provider_metadata.get("label_version") == "v3"


def test_llamaparse_layout_extraction_promotes_checkbox_mark_scope_without_duplicates() -> None:
    raw_output = {
        "pages": [
            {
                "page": 1,
                "width": 1000,
                "height": 1000,
                "items": [
                    {
                        "type": "text",
                        "value": "Question [x]",
                        "layoutAwareBbox": [
                            {
                                "x": 100,
                                "y": 200,
                                "w": 20,
                                "h": 20,
                                "confidence": 0.91,
                                "label": "checkbox-selected",
                            }
                        ],
                    },
                    {
                        "type": "Checkbox-Selected",
                        "value": "[x]",
                        "layoutAwareBbox": [
                            {
                                "x": 100,
                                "y": 200,
                                "w": 20,
                                "h": 20,
                                "confidence": 0.91,
                                "label": "checkbox-selected",
                            }
                        ],
                    },
                ],
            }
        ]
    }

    output = extract_all_layouts_from_llamaparse_output(
        raw_output,
        example_id="doc-checkbox",
        pipeline_name="llamaparse_eval",
    )

    assert len(output.predictions) == 1
    prediction = output.predictions[0]
    assert prediction.label == "checkbox-selected"
    assert prediction.attributes == {"scope": "mark"}
    assert prediction.provider_metadata["item_type"] == "Checkbox-Selected"


@_REQUIRES_LAYOUT_PAGES
def test_llamaparse_layout_extraction_emits_layout_pages_for_checkbox_marks() -> None:
    raw_output = _make_raw_output(["checkbox-selected", "text"])
    raw_output["pages"][0]["items"].append(
        {
            "type": "Checkbox-Selected",
            "value": "[x]",
            "layoutAwareBbox": [
                {"x": 100, "y": 100, "w": 200, "h": 120, "confidence": 0.9, "label": "checkbox-selected"}
            ],
        }
    )

    output = extract_all_layouts_from_llamaparse_output(
        raw_output,
        example_id="doc-checkbox-pages",
        pipeline_name="llamaparse_eval",
    )

    assert len(output.layout_pages) == 1
    items = output.layout_pages[0].items
    assert [item.type for item in items] == ["Text", "Checkbox-Selected"]
    assert items[1].attributes == {"scope": "mark"}


def test_llamaparse_layout_extraction_keeps_legacy_checkbox_segments_without_mark_item() -> None:
    raw_output = _make_raw_output(["checkbox-unselected", "text"])

    output = extract_layout_from_llamaparse_output(
        raw_output,
        page_index=0,
        example_id="doc-checkbox-legacy",
        pipeline_name="llamaparse_eval",
    )

    assert output is not None
    assert len(output.predictions) == 2
    checkbox_prediction = output.predictions[0]
    assert checkbox_prediction.label == "checkbox-unselected"
    assert checkbox_prediction.attributes == {}
    assert checkbox_prediction.provider_metadata["item_type"] == "text"
    if _HAS_LAYOUT_PAGES:
        assert output.layout_pages[0].items[0].type == "Checkbox-Unselected"
        assert output.layout_pages[0].items[0].attributes == {}


def test_llamaparse_layout_extraction_html_less_table_uses_per_segment_text_slice() -> None:
    """A table item without HTML in the page markdown must not attach the full
    table markdown to every segment (it inflates attribution F1); each segment
    gets its own startIndex/endIndex slice, like non-table items."""
    raw_output = {
        "pages": [
            {
                "page": 1,
                "width": 1000,
                "height": 1000,
                "md": "no html table here",
                "items": [
                    {
                        "type": "table",
                        "value": "row one\nrow two",
                        "layoutAwareBbox": [
                            {
                                "x": 10,
                                "y": 10,
                                "w": 100,
                                "h": 20,
                                "confidence": 0.9,
                                "label": "table",
                                "startIndex": 0,
                                "endIndex": 6,
                            },
                            {
                                "x": 10,
                                "y": 40,
                                "w": 100,
                                "h": 20,
                                "confidence": 0.9,
                                "label": "table",
                                "startIndex": 8,
                                "endIndex": 14,
                            },
                        ],
                    }
                ],
            }
        ]
    }

    output = extract_all_layouts_from_llamaparse_output(
        raw_output,
        example_id="doc-table-slice",
        pipeline_name="llamaparse_eval",
    )

    assert len(output.predictions) == 2
    assert output.predictions[0].content is not None
    assert output.predictions[1].content is not None
    assert output.predictions[0].content.text == "row one"
    assert output.predictions[1].content.text == "row two"
