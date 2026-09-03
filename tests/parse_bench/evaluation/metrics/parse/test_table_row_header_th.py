"""Tests for ``<th>`` row headers inside ``<tbody>``.

Marking a row's label cell as ``<th>`` is valid HTML and semantically right —
it is what a row header *is*::

    <tr><th>Cash</th><td>1,777</td><td>555</td></tr>

The header-block scanner used to read every such row as "this row is a header
row", because it only stopped when a non-``<th>`` cell in the row had content.
A balance sheet's section labels break exactly that test::

    <tr><th>Assets</th><td></td><td></td></tr>
    <tr><th>Current assets</th><td></td><td></td></tr>

so those two rows joined the *column*-header block, their text was flattened
into the label column's key (``"(In millions) Assets Current assets"``), and
the column no longer aligned with a ground truth that spells the same table
with ``<td>`` labels and the key ``"(In millions)"``. Every record then lost
its label cell and scored 0.5 instead of 1.0.

``find_row_header_cols`` names the columns that carry row headers — a column
holding a ``<th>`` in a row that is unambiguously data — and a row whose only
``<th>`` content sits in those columns no longer extends the header block.
The rule is content-independent, so it applies identically to the ground truth
and to the prediction.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_extraction import extract_table_pairs
from parse_bench.evaluation.metrics.parse.table_parsing import parse_html_tables
from parse_bench.evaluation.metrics.parse.table_record_match_metric import TableRecordMatchMetric
from parse_bench.evaluation.metrics.parse.table_title_stripping import (
    extract_header_info,
    find_row_header_cols,
)

# The prediction shape: a <thead> column-header row, then section rows and data
# rows whose label cell is a <th>.
TH_ROW_HEADERS = (
    "<table><thead><tr><th>(In millions)</th><th>2025</th><th>2024</th></tr></thead><tbody>"
    "<tr><th>Assets</th><td></td><td></td></tr>"
    "<tr><th>Current assets</th><td></td><td></td></tr>"
    "<tr><th>Cash</th><td>1,777</td><td>555</td></tr>"
    "<tr><th>Inventory</th><td>4,392</td><td>4,227</td></tr>"
    "</tbody></table>"
)

# The ground-truth shape: the same table with <td> labels.
TD_ROW_LABELS = (
    "<table><thead><tr><th>(In millions)</th><th>2025</th><th>2024</th></tr></thead><tbody>"
    "<tr><td>Assets</td><td></td><td></td></tr>"
    "<tr><td>Current assets</td><td></td><td></td></tr>"
    "<tr><td>Cash</td><td>1,777</td><td>555</td></tr>"
    "<tr><td>Inventory</td><td>4,392</td><td>4,227</td></tr>"
    "</tbody></table>"
)


def _header(html: str):  # noqa: ANN202 - local test helper
    tables = parse_html_tables(html)
    assert tables, "expected at least one parsed table"
    return tables[0], extract_header_info(tables[0])


def _trm(expected: str, actual: str) -> float:
    values = TableRecordMatchMetric().compute(expected, actual)
    score = next(v.value for v in values if v.metric_name == "table_record_match")
    return float(score)


class TestRowHeaderColumnDetection:
    def test_label_column_is_recognized(self) -> None:
        table, _ = _header(TH_ROW_HEADERS)
        assert find_row_header_cols(table) == {0}

    def test_a_table_without_body_th_has_no_row_header_column(self) -> None:
        table, _ = _header(TD_ROW_LABELS)
        assert find_row_header_cols(table) == set()


class TestHeaderBlockStopsAtSectionRows:
    def test_section_rows_do_not_join_the_header_block(self) -> None:
        _table, header = _header(TH_ROW_HEADERS)
        assert header.col_header_rows == {0}

    def test_column_keys_stay_the_thead_row(self) -> None:
        _table, header = _header(TH_ROW_HEADERS)
        assert header.keys[0] == "(In millions)"

    def test_the_td_spelling_is_unaffected(self) -> None:
        _table, header = _header(TD_ROW_LABELS)
        assert header.col_header_rows == {0}
        assert header.keys[0] == "(In millions)"


class TestScoringEquivalence:
    def test_th_prediction_against_td_ground_truth(self) -> None:
        assert _trm(TD_ROW_LABELS, TH_ROW_HEADERS) == 1.0

    def test_td_prediction_against_th_ground_truth(self) -> None:
        # Symmetric: the ground truth is the side using <th> row headers.
        assert _trm(TH_ROW_HEADERS, TD_ROW_LABELS) == 1.0

    def test_identical_sides_still_score_perfectly(self) -> None:
        assert _trm(TD_ROW_LABELS, TD_ROW_LABELS) == 1.0
        assert _trm(TH_ROW_HEADERS, TH_ROW_HEADERS) == 1.0

    def test_a_real_cell_error_still_fails(self) -> None:
        broken = TH_ROW_HEADERS.replace("<td>1,777</td>", "<td>9,999</td>")
        assert _trm(TD_ROW_LABELS, broken) < 1.0


class TestGenuineMultiLevelHeadersSurvive:
    """A second header row that is really a column header must still count."""

    MULTI_LEVEL = (
        "<table><thead>"
        '<tr><th rowspan="2">Year</th><th colspan="2">Annual Percentage Change</th></tr>'
        "<tr><th>Berkshire</th><th>S&amp;P 500</th></tr>"
        "</thead><tbody>"
        "<tr><td>1995</td><td>57.4%</td><td>37.6%</td></tr>"
        "<tr><td>1996</td><td>6.2</td><td>23.0</td></tr>"
        "</tbody></table>"
    )

    def test_both_header_rows_are_kept(self) -> None:
        _table, header = _header(self.MULTI_LEVEL)
        assert header.col_header_rows == {0, 1}

    def test_a_thead_row_is_never_read_as_a_section_label(self) -> None:
        """Inside <thead> the document has said the row is a column header.

        The shape is real: a wide permitted-uses matrix names its label column
        in the last <thead> row, with every sibling cell an empty <td>, while
        the body spells its row labels as <th>. Structurally that is identical
        to a balance sheet's section row, and only the authored <thead> tells
        them apart — so <thead> membership exempts the row.
        """
        html = (
            "<table><thead>"
            '<tr><th rowspan="2">TABLE 4-1</th><th colspan="2">Rural</th></tr>'
            "<tr><th>RS-G</th><th>RR</th></tr>"
            "<tr><th>Use Category</th><td></td><td></td></tr>"
            "</thead><tbody>"
            "<tr><th>Household Living</th><td>P</td><td>P</td></tr>"
            "</tbody></table>"
        )
        table, header = _header(html)
        assert find_row_header_cols(table) == {0}
        assert header.col_header_rows == {0, 1, 2}, "<thead> rows stay in the header block"

    def test_an_empty_corner_header_row_is_kept(self) -> None:
        # <td></td> in the corner of an otherwise all-<th> header row: there is
        # no row-header column here (no data row carries a <th>), so the row
        # still joins the header block exactly as before.
        html = (
            "<table><tbody>"
            "<tr><td></td><th>2025</th><th>2024</th></tr>"
            "<tr><td>Cash</td><td>1,777</td><td>555</td></tr>"
            "</tbody></table>"
        )
        table, header = _header(html)
        assert find_row_header_cols(table) == set()
        assert header.col_header_rows == {0}


def test_tables_pair_and_score_through_the_shared_extraction_stage() -> None:
    """The fix holds on the path the evaluator actually runs."""
    expected, actual, counts = extract_table_pairs(TD_ROW_LABELS, TH_ROW_HEADERS)
    assert counts.expected == 1
    assert counts.actual == 1
    assert expected[0].table_data.data.shape == actual[0].table_data.data.shape
