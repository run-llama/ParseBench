"""Regression tests for preserving explicit ``<tbody>`` row provenance."""

from __future__ import annotations

import numpy as np

from parse_bench.evaluation.metrics.parse.table_extraction import ExtractedTable
from parse_bench.evaluation.metrics.parse.table_parsing import TableData, parse_html_tables
from parse_bench.evaluation.metrics.parse.table_record_match_metric import normalize_table
from parse_bench.evaluation.metrics.parse.table_splitting import build_sub_table
from parse_bench.evaluation.metrics.parse.table_title_stripping import _remove_rows


def _table(*, rows: list[list[str]], tbody_rows: set[int]) -> TableData:
    return TableData(data=np.array(rows, dtype=object), tbody_rows=tbody_rows)


def _extracted(td: TableData) -> ExtractedTable:
    return ExtractedTable(raw_html="<table></table>", table_data=td)


def test_normalize_table_preserves_tbody_rows() -> None:
    td = _table(rows=[["H", ""], ["A", "1"]], tbody_rows={1})

    assert normalize_table(td).tbody_rows == {1}


def test_build_sub_table_filters_trimmed_tbody_rows() -> None:
    td = _table(rows=[["H", "H2"], ["A", "1"], ["", ""]], tbody_rows={1, 2})

    sub = build_sub_table(td, 0, 1)

    assert sub.data.shape == (2, 1)
    assert sub.tbody_rows == {1}


def test_title_row_removal_remaps_tbody_rows() -> None:
    td = _table(rows=[["Title"], ["H"], ["A"]], tbody_rows={2})

    removed, old_to_new = _remove_rows(td, frozenset({0}))

    assert old_to_new == {1: 0, 2: 1}
    assert removed.tbody_rows == {1}


def test_parser_marks_explicit_tbody_rows() -> None:
    td = parse_html_tables("<table><tr><th>H</th></tr><tbody><tr><td>A</td></tr></tbody></table>")[0]

    assert td.tbody_rows == {1}


def test_parser_tracks_identical_tbody_rows_by_identity() -> None:
    td = parse_html_tables(
        "<table><tr><th>2022</th></tr><tbody><tr><th>2022</th></tr><tr><td>17</td></tr></tbody></table>"
    )[0]

    assert td.tbody_rows == {1, 2}


def test_parser_marks_explicit_tfoot_rows() -> None:
    td = parse_html_tables("<table><tfoot><tr><th>Footer</th></tr></tfoot><tbody><tr><td>A</td></tr></tbody></table>")[
        0
    ]

    assert td.tfoot_rows == {0}
