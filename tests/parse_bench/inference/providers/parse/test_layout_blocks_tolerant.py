"""``extract_layout_blocks_tolerant``: layout markup that varies from page to page (Gemma 4)."""

from __future__ import annotations

from parse_bench.inference.providers.parse._layout_utils import extract_layout_blocks_tolerant, parse_layout_blocks


def _boxes(blocks: list[dict]) -> list[tuple[list[float], str, str]]:
    return [([float(v) for v in b["bbox"]], b["label"], b["text"]) for b in blocks]


def test_bracketed_flat_markup_matches_the_strict_parser() -> None:
    content = (
        '<div data-bbox="[10, 20, 300, 60]" data-label="Title">Annual report</div>\n'
        '<div data-label="Text" data-bbox="[10, 80, 300, 200]">Revenue grew 12%.</div>'
    )
    assert _boxes(extract_layout_blocks_tolerant(content)) == _boxes(parse_layout_blocks(content))


def test_bbox_without_brackets() -> None:
    content = '<div data-bbox="121, 163, 319, 217" data-label="Section-header">BAOTOU</div>'
    assert parse_layout_blocks(content) == []
    assert _boxes(extract_layout_blocks_tolerant(content)) == [
        ([121.0, 163.0, 319.0, 217.0], "Section-header", "BAOTOU")
    ]


def test_label_on_inner_divs_of_a_bbox_wrapper() -> None:
    content = (
        '<div data-bbox="340, 33, 659, 57">\n'
        '<div data-label="Title">CANARA ROBECO</div>\n'
        '<div data-label="Title">Mutual Fund</div>\n'
        "</div>\n"
        '<div data-bbox="[833, 27, 913, 64]" data-label="Page-header">DRAFT</div>'
    )
    assert _boxes(extract_layout_blocks_tolerant(content)) == [
        ([340.0, 33.0, 659.0, 57.0], "Title", "CANARA ROBECO\n\nMutual Fund"),
        ([833.0, 27.0, 913.0, 64.0], "Page-header", "DRAFT"),
    ]


def test_picture_without_text_keeps_its_box_but_containers_are_skipped() -> None:
    content = (
        '<div data-bbox="[0, 0, 1000, 1000]" data-label="Text">\n'
        '<div data-bbox="[10, 10, 400, 300]" data-label="Picture"></div>\n'
        '<div data-bbox="[10, 320, 400, 360]" data-label="Caption">Figure 1</div>\n'
        "</div>"
    )
    assert _boxes(extract_layout_blocks_tolerant(content)) == [
        ([10.0, 10.0, 400.0, 300.0], "Picture", ""),
        ([10.0, 320.0, 400.0, 360.0], "Caption", "Figure 1"),
    ]
