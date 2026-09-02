"""Tests for Azure Document Intelligence layout reconstruction."""

from __future__ import annotations

from parse_bench.inference.providers.parse.azure_document_intelligence import (
    AZURE_DI_SELECTION_LABEL_MAP,
    _bbox_center_inside_any,
    _build_layout_pages,
    _build_table_html_from_cells,
)


def _square(x: float, y: float, size: float) -> list[float]:
    return [x, y, x + size, y, x + size, y + size, x, y + size]


def test_build_table_html_uses_th_for_header_kinds_and_honors_spans() -> None:
    cells = [
        {"row_index": 0, "column_index": 0, "content": "H1", "kind": "columnHeader", "column_span": 2},
        {"row_index": 1, "column_index": 0, "content": "R", "kind": "rowHeader"},
        {"row_index": 1, "column_index": 1, "content": "a < b", "kind": "content", "row_span": 2},
    ]
    html = _build_table_html_from_cells(cells, row_count=3, column_count=2)
    assert html == (
        "<table>"
        '<tr><th colspan="2">H1</th></tr>'
        '<tr><th>R</th><td rowspan="2">a &lt; b</td></tr>'
        "<tr><td></td></tr>"
        "</table>"
    )


def test_build_table_html_degenerate_inputs_return_empty() -> None:
    assert _build_table_html_from_cells([], 1, 1) == ""
    assert _build_table_html_from_cells([{"row_index": 0, "column_index": 0, "content": "x"}], 0, 1) == ""


def test_bbox_center_inside_any() -> None:
    assert _bbox_center_inside_any((0.1, 0.1, 0.1, 0.1), [(0.0, 0.0, 0.5, 0.5)])
    assert not _bbox_center_inside_any((0.6, 0.6, 0.1, 0.1), [(0.0, 0.0, 0.5, 0.5)])


def test_build_layout_pages_emits_table_html_checkboxes_and_drops_cell_paragraphs() -> None:
    raw_output = {
        "pages": [
            {
                "page_number": 1,
                "width": 10.0,
                "height": 10.0,
                "selection_marks": [
                    {"state": "selected", "polygon": _square(8, 8, 1), "confidence": 0.9},
                    {"state": "unselected", "polygon": _square(8, 6, 1), "confidence": None},
                    {"state": "weird", "polygon": _square(1, 1, 1)},
                ],
            }
        ],
        "paragraphs": [
            {
                "content": "Title",
                "role": "title",
                "bounding_regions": [{"page_number": 1, "polygon": _square(0, 0, 2)}],
            },
            {
                # Sits inside the table region: a cell-as-paragraph duplicate.
                "content": "cell",
                "role": None,
                "bounding_regions": [{"page_number": 1, "polygon": _square(3, 3, 1)}],
            },
        ],
        "tables": [
            {
                "row_count": 1,
                "column_count": 1,
                "cells": [{"row_index": 0, "column_index": 0, "content": "cell", "kind": "columnHeader"}],
                "bounding_regions": [{"page_number": 1, "polygon": _square(2, 2, 4)}],
            }
        ],
        "figures": [],
    }

    pages = _build_layout_pages(raw_output)
    assert len(pages) == 1
    items = pages[0].items
    labels = [item.bbox.label for item in items if item.bbox is not None]
    assert labels.count("Text") == 0
    assert "Table" in labels
    assert "Checkbox-Selected" in labels
    assert "Checkbox-Unselected" in labels
    assert "weird" not in labels

    table = next(item for item in items if item.type == "table")
    assert table.html == "<table><tr><th>cell</th></tr></table>"
    assert table.value == "cell"

    checkboxes = [item for item in items if item.type == "checkbox"]
    assert len(checkboxes) == 2
    selected = next(item for item in checkboxes if item.bbox is not None and item.bbox.label == "Checkbox-Selected")
    assert selected.bbox is not None
    assert selected.bbox.confidence == 0.9
    assert abs(selected.bbox.x - 0.8) < 1e-9
    assert set(AZURE_DI_SELECTION_LABEL_MAP.values()) == {"Checkbox-Selected", "Checkbox-Unselected"}
