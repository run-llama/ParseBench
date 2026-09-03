"""A run of abutting inline markers inside a table CELL separates; a lone one joins.

This is the cell-text counterpart of ``test_normalize_text_inline_tag_separator``.
Once ``_augment_with_table_cell_text`` REPLACES a table with its cell text, that
text no longer carries its markup into ``normalize_text``, so the marker-run rule
there cannot reach it.  ``BeautifulSoup.get_text()`` concatenates text nodes with
no separator, so a byte-faithful ``<u>105</u><s>103</s>`` extracted as the welded
``105103`` and every word/sentence rule reported both page numbers as missing.

The asymmetry must match ``normalize_text`` exactly: two abutting markers are the
boundary between two separately-styled tokens and separate; ONE marker between two
content characters is an intra-token style change and must still join.
"""

import pytest

from parse_bench.evaluation.metrics.parse.rules_base import (
    _augment_with_table_cell_text,
    _extract_table_cell_texts,
)
from parse_bench.evaluation.metrics.parse.table_parsing import parse_html_tables
from parse_bench.evaluation.metrics.parse.utils import normalize_text


def _cells(html: str) -> list[list[str]]:
    return [list(row) for table in parse_html_tables(html) for row in table.data]


def test_two_abutting_markers_in_a_cell_separate():
    """The witness weld: `<u>105</u><s>103</s>` is two page numbers, not one."""
    html = "<table><tr><td>Appeals</td><td><u>105</u><s>103</s></td></tr></table>"
    assert _cells(html) == [["Appeals", "105 103"]]


def test_a_lone_marker_in_a_cell_still_joins():
    """One marker inside a token is a style change, not a boundary."""
    html = "<table><tr><td>Shizuok<mark>a</mark> University</td></tr></table>"
    assert _cells(html) == [["Shizuoka University"]]


@pytest.mark.parametrize(
    ("cell_html", "expected"),
    [
        ("<b><u>103</u><s>101</s></b>", "103 101"),
        ("<u>103</u><s>101</s><i>99</i>", "103 101 99"),
        ("<u>103</u> <s>101</s>", "103 101"),
        ("<del>old</del><ins>new</ins>", "old new"),
        ("<span>a</span><span>b</span>", "a b"),
        ("plain text", "plain text"),
        ("<u>whole cell</u>", "whole cell"),
    ],
)
def test_marker_shapes_in_cells(cell_html: str, expected: str):
    assert _cells(f"<table><tr><td>{cell_html}</td></tr></table>") == [[expected]]


def test_intra_word_marker_run_is_the_documented_edge():
    """A run WITHIN a word still separates - same as normalize_text does."""
    html = "<table><tr><td>ab<u>c</u><s>d</s>ef</td></tr></table>"
    assert _cells(html) == [["abc def"]]


def test_augmented_text_carries_the_separated_cell():
    """End to end: the weld does not survive into the augmented document text."""
    md = "<table><tr><td>Appeals</td><td><u>105</u><s>103</s></td></tr></table>"
    augmented = _augment_with_table_cell_text(md)
    assert "105 103" in augmented
    assert "105103" not in augmented


def test_normalized_augmented_text_matches_the_ground_truth_spelling():
    """The GT annotates this page as `~~105~~ 103`; both sides must agree."""
    md = "<table><tr><td><u>105</u><s>103</s></td></tr></table>"
    assert normalize_text(_augment_with_table_cell_text(md)).endswith("105 103")
    assert normalize_text("~~105~~ 103") == "105 103"


def test_a_table_without_markers_is_untouched():
    """Guard: cells with no inline markers extract exactly as before."""
    md = "<table><tr><td>Definitions</td><td>103</td></tr></table>"
    assert _extract_table_cell_texts(md) == ["Definitions", "103"]
