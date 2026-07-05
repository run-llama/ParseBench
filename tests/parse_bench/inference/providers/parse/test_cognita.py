"""Unit tests for the Cognita provider's normalization logic."""

from __future__ import annotations

from typing import Any

from parse_bench.inference.providers.parse.cognita import (
    _block_styled_markdown,
    _build_layout_pages,
    _ir_table_to_html,
    _pipe_block_to_html,
    _replace_pipe_tables_with_html,
    _tag_trailing_page_number,
)


def _cell(text: str, col_span: int = 0, row_span: int = 0) -> dict[str, Any]:
    cell: dict[str, Any] = {"blocks": [{"type": "paragraph", "text": text}]}
    if col_span:
        cell["col_span"] = col_span
    if row_span:
        cell["row_span"] = row_span
    return cell


def test_ir_table_to_html_emits_spans_and_header_row() -> None:
    table = {
        "rows": [
            {"cells": [_cell("Metric", col_span=2), _cell("Value")]},
            {"cells": [_cell("Revenue"), _cell("Q3"), _cell("18%")]},
        ],
    }
    html = _ir_table_to_html(table)
    assert html.startswith("<table>")
    # First row is the header by convention when the IR does not say otherwise.
    assert '<th colspan="2">Metric</th>' in html
    assert "<td>Revenue</td>" in html


def test_ir_table_to_html_escapes_cell_text() -> None:
    table = {"rows": [{"cells": [_cell("<b>bold</b> & co"), _cell("x")]}, {"cells": [_cell("y"), _cell("z")]}]}
    html = _ir_table_to_html(table)
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; co" in html


def test_replace_pipe_tables_prefers_ir_tables() -> None:
    document = {
        "pages": [
            {
                "number": 1,
                "blocks": [
                    {
                        "type": "table",
                        "table": {"rows": [{"cells": [_cell("A"), _cell("B")]}, {"cells": [_cell("1"), _cell("2")]}]},
                    }
                ],
            }
        ]
    }
    markdown = "before\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nafter\n"
    out = _replace_pipe_tables_with_html(markdown, document)
    assert "| A | B |" not in out
    assert "<th>A</th>" in out
    assert out.startswith("before")
    assert out.rstrip().endswith("after")


def test_replace_pipe_tables_falls_back_to_textual_conversion() -> None:
    # Document has no IR tables, so the pipe block converts textually.
    markdown = "| H1 | H2 |\n| --- | --- |\n| a \\| b | c |\n"
    out = _replace_pipe_tables_with_html(markdown, {})
    assert "<th>H1</th>" in out
    assert "<td>a | b</td>" in out


def test_pipe_block_to_html_skips_separator_rows() -> None:
    html = _pipe_block_to_html(["| X | Y |", "| --- | --- |", "| 1 | 2 |"])
    assert html.count("<tr>") == 2
    assert "<th>X</th>" in html
    assert "<td>2</td>" in html


def test_tag_page_numbers_wraps_standalone_numbers_only() -> None:
    assert _tag_trailing_page_number("Annual Report 3") == "Annual Report <page_number>3</page_number>"
    assert _tag_trailing_page_number("- 2 -") == "- <page_number>2</page_number> -"
    # Dates, times and glued tokens stay untouched.
    assert _tag_trailing_page_number("10/16/06 4:14:38 PM") == "10/16/06 4:14:38 PM"
    assert _tag_trailing_page_number("ISO9001") == "ISO9001"


def test_block_styled_markdown_renders_span_styles() -> None:
    block = {
        "text": "Ed. 6-11 Page 9",
        "spans": [
            {"text": "Ed. ", "style": {"bold": True}},
            {"text": "6-11", "style": {"highlight": True}},
            {"text": " Page 9"},
        ],
    }
    styled = _block_styled_markdown(block)
    assert "**Ed.**" in styled
    assert "<mark>6-11</mark>" in styled
    assert styled.endswith(" Page 9")


def test_build_layout_pages_normalizes_bboxes_and_sections() -> None:
    document = {
        "pages": [
            {
                "number": 1,
                "width": 600.0,
                "height": 800.0,
                "blocks": [
                    {
                        "type": "header",
                        "text": "ACME Corp 7",
                        "bbox": {"x": 60.0, "y": 40.0, "w": 300.0, "h": 16.0},
                        "spans": [{"text": "ACME Corp 7", "style": {"bold": True}}],
                    },
                    {
                        "type": "heading",
                        "level": 1,
                        "text": "Title",
                        "bbox": {"x": 60.0, "y": 100.0, "w": 480.0, "h": 24.0},
                    },
                    {
                        "type": "page_number",
                        "text": "7",
                        "bbox": {"x": 290.0, "y": 770.0, "w": 20.0, "h": 12.0},
                    },
                ],
            }
        ]
    }
    pages = _build_layout_pages(document)
    assert len(pages) == 1
    page = pages[0]

    # Bboxes are normalized to [0, 1] against the page dimensions.
    header_item = page.items[0]
    assert header_item.bbox is not None
    assert header_item.bbox.x == 0.1
    assert header_item.bbox.y == 0.05
    assert header_item.bbox.label == "Page-header"

    # Level-1 headings map to the Title canonical class.
    assert page.items[1].bbox is not None
    assert page.items[1].bbox.label == "Title"

    # Structured sections carry styled and plain variants with tagged numbers.
    assert "**ACME Corp <page_number>7</page_number>**" in page.page_header_markdown
    assert "ACME Corp 7" in page.page_header_markdown
    assert page.printed_page_number == "7"
    assert "<page_number>7</page_number>" in page.page_footer_markdown


def test_build_layout_pages_skips_items_without_dimensions() -> None:
    document = {
        "pages": [
            {"number": 1, "blocks": [{"type": "paragraph", "text": "x", "bbox": {"x": 1, "y": 1, "w": 1, "h": 1}}]}
        ]
    }
    pages = _build_layout_pages(document)
    assert len(pages) == 1
    assert pages[0].items == []
