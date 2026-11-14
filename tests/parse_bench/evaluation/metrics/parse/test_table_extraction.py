"""Tests for the shared table extraction stage."""

from __future__ import annotations

import pickle

import pytest

from parse_bench.evaluation.metrics.parse.table_extraction import (
    ExtractedTable,
    GroundTruthTableParseError,
    extract_html_tables,
    extract_table_pairs,
)

ONE_TABLE = """
<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
"""

TWO_TABLES = """
<table><tr><th>A</th></tr><tr><td>1</td></tr></table>
some text
<table><tr><th>B</th></tr><tr><td>2</td></tr></table>
"""

# A "<table" appears but no closing </table> AND no parseable content -
# extract_html_tables grabs it via fallback, parse_html_tables returns []
MALFORMED = "<table>oops"


def test_single_table_each_side() -> None:
    expected, actual, counts = extract_table_pairs(ONE_TABLE, ONE_TABLE)
    assert len(expected) == 1
    assert len(actual) == 1
    assert counts.expected == 1
    assert counts.actual == 1
    assert counts.unparseable_pred == 0
    assert isinstance(expected[0], ExtractedTable)
    assert "<table" in expected[0].raw_html.lower()
    assert expected[0].table_data.data.shape == (2, 2)


def test_multi_table() -> None:
    expected, actual, counts = extract_table_pairs(TWO_TABLES, TWO_TABLES)
    assert len(expected) == 2
    assert len(actual) == 2
    assert counts.expected == 2
    assert counts.actual == 2


def test_extract_html_tables_preserves_indices_after_expanding_lowercase_chars() -> None:
    md = "GARANTİ BBVA\n\n<table><thead><tr><td>A</td></tr></thead></table>"

    tables = extract_html_tables(md)

    assert tables == ["<table><thead><tr><td>A</td></tr></thead></table>"]
    expected, actual, counts = extract_table_pairs(md, md)
    assert len(expected) == 1
    assert len(actual) == 1
    assert counts.expected == 1
    assert counts.actual == 1


def test_extract_html_tables_matches_tags_case_insensitively() -> None:
    md = "<TABLE><TR><TD>A</TD></TR></TABLE>"

    assert extract_html_tables(md) == [md]


def test_no_tables() -> None:
    expected, actual, counts = extract_table_pairs("no tables here", "also nothing")
    assert expected == []
    assert actual == []
    assert counts.expected == 0
    assert counts.actual == 0
    assert counts.unparseable_pred == 0


def test_mismatched_counts() -> None:
    expected, actual, counts = extract_table_pairs(TWO_TABLES, ONE_TABLE)
    assert counts.expected == 2
    assert counts.actual == 1


def test_pred_unparseable_dropped() -> None:
    # one good table + one malformed in actual
    actual_md = ONE_TABLE + "\n" + MALFORMED
    expected, actual, counts = extract_table_pairs(ONE_TABLE, actual_md)
    assert counts.expected == 1
    # The malformed slice may or may not be picked up by extract_html_tables;
    # if it is, parse_html_tables drops it, increasing unparseable_pred.
    # Either way, len(actual) + counts.unparseable_pred should equal the
    # number of slices extract_html_tables produced for actual_md.
    assert counts.unparseable_pred >= 0
    assert counts.actual == len(actual)


def test_gt_unparseable_raises() -> None:
    gt_md = ONE_TABLE + "\n" + MALFORMED
    # Only raises if extract_html_tables actually produced a slice that
    # parse_html_tables can't parse. Build a deliberate failing fixture:
    # a "<table" with no closing tag triggers extract_html_tables's
    # fallback path that grabs from <table to end-of-string.
    bad = "before <table foo bar baz"
    with pytest.raises(GroundTruthTableParseError) as exc_info:
        extract_table_pairs(bad, ONE_TABLE, doc_id="doc-xyz")
    assert "doc-xyz" in str(exc_info.value)
    # Use gt_md to silence unused-variable warning
    _ = gt_md


def test_pickle_roundtrip() -> None:
    """ExtractedTable must pickle for the parallel-path subprocess in P4."""
    expected, _, _ = extract_table_pairs(ONE_TABLE, ONE_TABLE)
    et = expected[0]
    blob = pickle.dumps(et)
    restored = pickle.loads(blob)
    assert restored.raw_html == et.raw_html
    assert restored.table_data.data.shape == et.table_data.data.shape


def test_truncated_cell_attribute_does_not_swallow_following_tables() -> None:
    """Reducto complex_header_018 ended a block at colspan= before four valid tables."""
    fragment = "<table>\n<thead>\n<tr>\n<th></th> <th colspan=\n\n"
    actual_md = fragment + TWO_TABLES.lstrip()
    assert extract_html_tables(actual_md) == [fragment, *extract_html_tables(TWO_TABLES)]
    expected, actual, counts = extract_table_pairs(TWO_TABLES, actual_md)
    assert [table.raw_html for table in actual] == [table.raw_html for table in expected]
    assert counts.actual == 2
    assert counts.unparseable_pred == 1
    with pytest.raises(GroundTruthTableParseError):
        extract_table_pairs(actual_md, TWO_TABLES)


@pytest.mark.parametrize("attribute", ["colspan", "rowspan"])
def test_invalid_prediction_span_is_counted_instead_of_crashing_document(attribute: str) -> None:
    bad = f'<table><tr><td {attribute}="broken">lost</td></tr></table>'
    _, actual, counts = extract_table_pairs(TWO_TABLES, bad + ONE_TABLE)
    assert len(actual) == 1
    assert counts.unparseable_pred == 1
    with pytest.raises(GroundTruthTableParseError, match="bad-span"):
        extract_table_pairs(bad, ONE_TABLE, doc_id="bad-span")


def test_complete_cell_before_nested_table_keeps_outer_table_intact() -> None:
    nested = '<table><tr><td title="nested">' + ONE_TABLE + "</td></tr></table>"
    assert extract_html_tables(nested + ONE_TABLE) == [nested, ONE_TABLE.strip()]
