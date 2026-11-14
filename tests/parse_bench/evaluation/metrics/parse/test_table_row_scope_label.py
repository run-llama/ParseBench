"""Tests for spanning section labels that declare a row scope.

A section label inside a table body is often written as a full-width
``<th>``::

    <tr><th colspan="3">SPINAL DISORDERS</th></tr>

and HTML lets the author say what that header applies to with ``scope``::

    <tr><th colspan="3" scope="rowgroup">SPINAL DISORDERS</th></tr>

``scope="row"`` / ``scope="rowgroup"`` says the cell heads rows, not columns,
so the parser correctly leaves it out of ``col_headers``. The header-block
scanner, though, still counted the row as part of the column-header block
(it holds a ``<th>``), and because the row contributed no column-header text
the title detector could not recognise it as a spanning title either. The
row ended up neither a column key nor a data row, so its text vanished from
the record set and the prediction lost a record that the same markup without
``scope`` keeps.

A row whose ``<th>`` cells all declare a row scope now ends the column-header
block, so it is read as a body row — the same records as the unscoped
``<th>`` and the ``<td>`` spellings.
"""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.table_extraction import extract_normalized_tables
from parse_bench.evaluation.metrics.parse.table_record_match_metric import TableRecordMatchMetric
from parse_bench.evaluation.metrics.parse.table_title_stripping import strip_title_rows

HEAD = "<tr><th>Disorder</th><th>History</th><th>Exam</th></tr>"
BODY = "<tr><td>Fracture</td><td>Trauma</td><td>Tenderness</td></tr><tr><td>Tumor</td><td>Pain</td><td>Pallor</td></tr>"


def _table(label_row: str, *, groups: bool = False) -> str:
    if groups:
        return f"<table><thead>{HEAD}</thead><tbody>{label_row}{BODY}</tbody></table>"
    return f"<table>{HEAD}{label_row}{BODY}</table>"


PLAIN_TH = '<tr><th colspan="3">SPINAL DISORDERS</th></tr>'
TD = '<tr><td colspan="3">SPINAL DISORDERS</td></tr>'
ROWGROUP = '<tr><th colspan="3" scope="rowgroup">SPINAL DISORDERS</th></tr>'
ROW = '<tr><th colspan="3" scope="row">SPINAL DISORDERS</th></tr>'


def _trm(expected: str, actual: str) -> float:
    values = TableRecordMatchMetric().compute(expected, actual)
    return float(next(v.value for v in values if v.metric_name == "table_record_match"))


def _hints(html: str):  # noqa: ANN202 - local test helper
    tables, _ = extract_normalized_tables(html, side="actual")
    return strip_title_rows(tables[0]).header_hints


class TestRowScopedLabelScoresLikeUnscoped:
    @pytest.mark.parametrize("label", [ROWGROUP, ROW])
    @pytest.mark.parametrize("groups", [False, True])
    def test_against_plain_th_ground_truth(self, label: str, groups: bool) -> None:
        assert _trm(_table(PLAIN_TH), _table(label, groups=groups)) == pytest.approx(1.0)

    @pytest.mark.parametrize("label", [ROWGROUP, ROW])
    def test_against_td_ground_truth(self, label: str) -> None:
        assert _trm(_table(TD), _table(label, groups=True)) == pytest.approx(1.0)

    @pytest.mark.parametrize("label", [ROWGROUP, ROW])
    def test_as_ground_truth(self, label: str) -> None:
        assert _trm(_table(label, groups=True), _table(PLAIN_TH)) == pytest.approx(1.0)

    def test_scoped_label_is_not_a_column_header_row(self) -> None:
        hints = _hints(_table(ROWGROUP, groups=True))
        assert hints.col_header_rows == frozenset({0})


class TestColumnHeadersAreUnchanged:
    def test_unscoped_label_keeps_its_title_handling(self) -> None:
        hints = _hints(_table(PLAIN_TH))
        assert hints.col_header_rows == frozenset({0, 1})
        assert hints.th_title_rows == frozenset({1})

    def test_col_scoped_second_header_row_stays_in_the_block(self) -> None:
        html = (
            "<table><thead>"
            '<tr><th scope="col">Item</th><th colspan="2" scope="colgroup">Year</th></tr>'
            '<tr><th></th><th scope="col">2024</th><th scope="col">2025</th></tr>'
            "</thead><tbody><tr><td>A</td><td>1</td><td>2</td></tr></tbody></table>"
        )
        assert _hints(html).col_header_rows == frozenset({0, 1})

    def test_row_with_row_and_col_scoped_headers_stays_in_the_block(self) -> None:
        html = (
            "<table>"
            "<tr><th>Item</th><th>2024</th><th>2025</th></tr>"
            '<tr><th scope="row">Unit</th><th scope="col">USD</th><th scope="col">USD</th></tr>'
            "<tr><td>A</td><td>1</td><td>2</td></tr></table>"
        )
        assert _hints(html).col_header_rows == frozenset({0, 1})

    def test_row_scoped_cell_in_thead_stays_in_the_block(self) -> None:
        html = (
            "<table><thead>"
            "<tr><th>Item</th><th>2024</th></tr>"
            '<tr><th scope="row" colspan="2">Restated</th></tr>'
            "</thead><tbody><tr><td>A</td><td>1</td></tr></tbody></table>"
        )
        assert _hints(html).col_header_rows == frozenset({0, 1})
