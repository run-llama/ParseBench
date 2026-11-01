"""TRM pairs columns whose keys differ by whole leading or trailing words (a title or unit kept on one side)."""

import pytest

from parse_bench.evaluation.metrics.parse.table_record_match_metric import TableRecordMatchMetric


def _trm(expected: str, actual: str) -> float:
    return (
        TableRecordMatchMetric()
        .compute(expected=f"<table>{expected}</table>", actual=f"<table>{actual}</table>")[0]
        .value
    )


_ROWS = "<tr><td>a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr>"


def test_key_with_an_extra_leading_word_pairs_the_column() -> None:
    gt = "<tr><th>Item</th><th>Age</th></tr>" + _ROWS
    pred = "<tr><th>Item</th><th>Rate Age</th></tr>" + _ROWS
    assert _trm(gt, pred) == pytest.approx(1.0)


def test_key_with_an_extra_trailing_word_pairs_the_column() -> None:
    gt = "<tr><th>Item</th><th>Age</th></tr>" + _ROWS
    pred = "<tr><th>Item</th><th>Age Years</th></tr>" + _ROWS
    assert _trm(gt, pred) == pytest.approx(1.0)


def test_key_prefix_inside_a_word_does_not_pair_the_column() -> None:
    gt = "<tr><th>Item</th><th>Col</th></tr>" + _ROWS
    pred = "<tr><th>Item</th><th>Colour</th></tr>" + _ROWS
    assert _trm(gt, pred) < 0.5
