"""Tests for the upstream table-title-stripping stage.

Focused on the ``max_top_title_rows`` cap semantics that the strip stage
exposes to test cases. Detector-level edge cases are exercised by the
broader TRM test suite.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_extraction import ExtractedTable
from parse_bench.evaluation.metrics.parse.table_parsing import parse_html_tables
from parse_bench.evaluation.metrics.parse.table_title_stripping import strip_title_rows


def _wrap(html: str) -> ExtractedTable:
    """Parse a single HTML table and wrap it in an ExtractedTable."""
    tables = parse_html_tables(f"<table>{html}</table>")
    assert tables, "expected at least one parsed table"
    return ExtractedTable(raw_html=f"<table>{html}</table>", table_data=tables[0])


def test_default_cap_strips_top_td_title_and_top_th_title() -> None:
    """At default cap=1, both leading <td> title and one top <th> title are stripped."""
    et = _wrap(
        '<tr><td colspan="2">Report Title</td></tr>'
        "<thead>"
        '<tr><th colspan="2">(in millions)</th></tr>'
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
    )
    out = strip_title_rows(et)
    # Td title row + top th title row both removed
    assert out.table_data.data.shape[0] == 2
    hints = out.header_hints
    assert hints is not None
    # Stripped titles are stored normalized (lowercased)
    assert "report title" in hints.stripped_titles
    assert "(in millions)" in hints.stripped_titles


def test_cap_zero_strips_nothing_from_top() -> None:
    """``max_top_title_rows=0`` disables both top td and top th stripping.

    The bottom-th title strip is independent and not affected here
    because there is no bottom title in this fixture.
    """
    et = _wrap(
        '<tr><td colspan="2">Report Title</td></tr>'
        "<thead>"
        '<tr><th colspan="2">(in millions)</th></tr>'
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
    )
    out = strip_title_rows(et, max_top_title_rows=0)
    # All four rows survive: td title + th title + th headers + data
    assert out.table_data.data.shape[0] == 4
    hints = out.header_hints
    assert hints is not None
    # Nothing was stripped, so no titles recorded
    assert hints.stripped_titles == frozenset()


def test_cap_two_strips_two_top_th_titles() -> None:
    """``max_top_title_rows=2`` allows two stacked top <th> title rows to be stripped."""
    et = _wrap(
        "<thead>"
        '<tr><th colspan="2">Annual Report</th></tr>'
        '<tr><th colspan="2">(in millions)</th></tr>'
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
    )
    out = strip_title_rows(et, max_top_title_rows=2)
    # Both stacked title rows removed → 2 rows remain (real headers + data)
    assert out.table_data.data.shape[0] == 2
    hints = out.header_hints
    assert hints is not None
    assert "annual report" in hints.stripped_titles
    assert "(in millions)" in hints.stripped_titles


def test_cap_two_with_only_one_title_strips_one() -> None:
    """A higher cap doesn't force extra stripping; only actual titles are removed."""
    et = _wrap(
        "<thead>"
        '<tr><th colspan="2">Annual Report</th></tr>'
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
    )
    out = strip_title_rows(et, max_top_title_rows=2)
    assert out.table_data.data.shape[0] == 2  # one title removed, real header + data
    hints = out.header_hints
    assert hints is not None
    assert "annual report" in hints.stripped_titles


def test_rowspan2_title_strips_both_grid_rows_for_one_slot() -> None:
    """A ``<th rowspan="2" colspan="2">`` title is one logical title.

    With ``max_top_title_rows=1`` it must consume exactly 1 slot from
    the cap AND strip both rowspan-expanded grid rows. Leaving the
    second rowspan row in place would leave a phantom title cell in
    the trimmed grid.
    """
    et = _wrap(
        "<thead>"
        '<tr><th rowspan="2" colspan="2">Big</th></tr>'
        "<tr></tr>"
        "<tr><th>X</th><th>Y</th></tr>"
        "</thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
    )
    out = strip_title_rows(et, max_top_title_rows=1)
    # Original 4 grid rows: 2 from rowspan title + 1 leaf header + 1 data.
    # Both rowspan rows must be stripped → 2 rows remain.
    assert out.table_data.data.shape[0] == 2
    # And the surviving rows are the leaf header + the data row
    surviving = [list(out.table_data.data[r]) for r in range(2)]
    assert surviving == [["X", "Y"], ["1", "2"]]


def test_two_stacked_th_titles_with_cap_one_strips_only_top() -> None:
    """Two stacked single-rowspan ``<th>`` title rows with the same text.

    Unlike the rowspan-2 case, each row originates from its own ``<th>``
    element. With ``max_top_title_rows=1`` only the topmost is stripped;
    the second stays in the trimmed grid because the cap is exhausted.
    This is the contrast to the rowspan test above — same final
    rendering of "Big" repeated twice on top of the header, but
    different DOM source ⇒ different strip outcome.
    """
    et = _wrap(
        "<thead>"
        '<tr><th colspan="2">Big</th></tr>'
        '<tr><th colspan="2">Big</th></tr>'
        "<tr><th>X</th><th>Y</th></tr>"
        "</thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
    )
    out = strip_title_rows(et, max_top_title_rows=1)
    # 4 grid rows → 3: top "Big" stripped, second "Big" + leaf + data remain
    assert out.table_data.data.shape[0] == 3
    surviving = [list(out.table_data.data[r]) for r in range(3)]
    assert surviving == [["Big", "Big"], ["X", "Y"], ["1", "2"]]


def test_top_title_not_contiguous_with_top_is_not_stripped() -> None:
    """Real <th> headers in row 0, span title in row 1, span title in row 2.

    Title rows must be contiguous with the top OR bottom of the header
    block to be stripped:

    - Top-strip walks ``sorted(col_header_rows)`` from row 0 and
      breaks on the first non-title row. Row 0 is real headers ⇒
      top-strip takes nothing.
    - Bottom-strip catches row 2 (it is the bottom of the header
      block AND a title) and removes it from column keys.
    - Row 1 sits between two non-strippable rows (real header above,
      bottom-strip-target below). It is **not** strippable: it stays
      in the grid AND in the column-key construction. Its text gets
      concatenated into every column key.

    This is the deliberate consequence of the contiguity rule. Docs
    that hit this shape need ``max_top_title_rows`` set explicitly or
    structural fixes upstream.
    """
    et = _wrap(
        "<thead>"
        "<tr><th>Mod Code</th><th>Buy Line</th></tr>"
        '<tr><th colspan="2">##CASH ##ASBL</th></tr>'
        '<tr><th colspan="2">MICHAEL BLOOMBERG FOR PRESIDENT</th></tr>'
        "</thead>"
        "<tbody><tr><td></td><td>1</td></tr></tbody>"
    )
    out = strip_title_rows(et, max_top_title_rows=1)
    hints = out.header_hints
    assert hints is not None
    # Nothing physically removed → 4 rows survive (3 header + 1 data).
    assert out.table_data.data.shape[0] == 4
    assert hints.col_header_rows == frozenset({0, 1, 2})
    # Only the bottom title is recorded for key-construction exclusion.
    assert hints.th_title_rows == frozenset({2})
    assert "michael bloomberg for president" in hints.stripped_titles
    # Row 1 (##CASH ##ASBL) was NOT stripped — it must not appear in
    # stripped_titles.
    assert "##cash ##asbl" not in hints.stripped_titles


def test_rowspan_title_does_not_double_consume_cap() -> None:
    """A rowspan-2 title at top + one trailing leaf header still leaves the leaf intact.

    Sanity check that consuming a rowspan title (which strips 2 grid rows
    for 1 slot) doesn't accidentally also consume the next slot.
    """
    et = _wrap(
        "<thead>"
        '<tr><th rowspan="2" colspan="2">Big Title</th></tr>'
        "<tr></tr>"
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
    )
    out = strip_title_rows(et, max_top_title_rows=1)
    # Both rowspan rows stripped (1 slot), leaf header + data remain.
    assert out.table_data.data.shape[0] == 2
    surviving = [list(out.table_data.data[r]) for r in range(2)]
    assert surviving == [["Q1", "Q2"], ["1", "2"]]
