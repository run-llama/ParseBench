"""Tests for the Cognita parse provider mapping helpers."""

from __future__ import annotations

from parse_bench.inference.providers.parse.cognita import (
    _canonical_label,
    _gfm_tables_to_html,
    _iter_layout_blocks,
    _normalized_segment,
)


def test_gfm_table_converted_to_html_prose_untouched() -> None:
    md = "# Title\n\nSome prose.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n\nAfter."
    out = _gfm_tables_to_html(md)
    assert "# Title" in out
    assert "Some prose." in out
    assert "After." in out
    assert "<table>" in out and "</table>" in out
    assert "<th>A</th><th>B</th>" in out
    assert "<td>1</td><td>2</td>" in out
    assert "<td>3</td><td>4</td>" in out
    # The pipe syntax for the converted table is gone.
    assert "| A | B |" not in out


def test_gfm_converter_escapes_html_and_honors_escaped_pipes() -> None:
    md = "| Expr | Note |\n| --- | --- |\n| a < b | x \\| y |"
    out = _gfm_tables_to_html(md)
    assert "<td>a &lt; b</td>" in out
    assert "<td>x | y</td>" in out


def test_gfm_converter_leaves_non_tables_alone() -> None:
    md = "Just a line with a | pipe but no table.\n\nAnother line."
    assert _gfm_tables_to_html(md) == md


def test_gfm_converter_ignores_pipe_tables_inside_fenced_code() -> None:
    md = "```\n| A | B |\n| --- | --- |\n| 1 | 2 |\n```\n\n| X | Y |\n| --- | --- |\n| 3 | 4 |"
    out = _gfm_tables_to_html(md)
    # The fenced block is preserved verbatim (no <table>).
    assert "```\n| A | B |\n| --- | --- |\n| 1 | 2 |\n```" in out
    # The real table outside the fence is still converted.
    assert "<th>X</th><th>Y</th>" in out
    assert out.count("<table>") == 1


def test_canonical_label_by_type_and_heading_level() -> None:
    assert _canonical_label({"type": "heading", "level": 1}) == "Title"
    assert _canonical_label({"type": "heading", "level": 3}) == "Section-header"
    assert _canonical_label({"type": "heading"}) == "Section-header"
    assert _canonical_label({"type": "paragraph"}) == "Text"
    assert _canonical_label({"type": "table"}) == "Table"
    assert _canonical_label({"type": "image"}) == "Picture"
    assert _canonical_label({"type": "list_item"}) == "List-item"
    assert _canonical_label({"type": "code"}) == "Code"
    assert _canonical_label({"type": "caption"}) == "Caption"
    assert _canonical_label({"type": "footer"}) == "Page-footer"
    assert _canonical_label({"type": "something_unknown"}) == "Text"


def test_normalized_segment_divides_by_page_dims_and_clamps() -> None:
    block = {"type": "paragraph", "bbox": {"x": 100, "y": 200, "w": 300, "h": 50}, "confidence": 0.9}
    seg = _normalized_segment(block, width=1000.0, height=1000.0)
    assert seg is not None
    assert abs(seg.x - 0.1) < 1e-9
    assert abs(seg.y - 0.2) < 1e-9
    assert abs(seg.w - 0.3) < 1e-9
    assert abs(seg.h - 0.05) < 1e-9
    assert seg.label == "Text"
    assert seg.confidence == 0.9

    # Out-of-range coordinates clamp into [0, 1].
    over = {"type": "paragraph", "bbox": {"x": 1200, "y": 0, "w": 100, "h": 50}}
    seg2 = _normalized_segment(over, width=1000.0, height=1000.0)
    assert seg2 is not None and seg2.x == 1.0


def test_normalized_segment_none_without_bbox_or_dims() -> None:
    assert _normalized_segment({"type": "paragraph"}, 100.0, 100.0) is None
    assert _normalized_segment({"type": "paragraph", "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}, 0, 100.0) is None


def test_iter_layout_blocks_flattens_list_children() -> None:
    blocks = [
        {"type": "paragraph", "text": "p"},
        {
            "type": "list",
            "children": [
                {"type": "list_item", "text": "one"},
                {"type": "list_item", "text": "two"},
            ],
        },
    ]
    flat = _iter_layout_blocks(blocks)
    types = [b["type"] for b in flat]
    assert types == ["paragraph", "list_item", "list_item"]


def test_iter_layout_blocks_keeps_empty_list_container() -> None:
    blocks = [{"type": "list", "children": []}]
    flat = _iter_layout_blocks(blocks)
    assert len(flat) == 1 and flat[0]["type"] == "list"
