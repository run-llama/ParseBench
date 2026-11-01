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


def test_key_inside_the_middle_of_another_does_not_pair_the_column() -> None:
    gt = "<tr><th>Item</th><th>Age</th></tr>" + _ROWS
    pred = "<tr><th>Item</th><th>Rate Age Group</th></tr>" + _ROWS
    assert _trm(gt, pred) < 0.5


def test_key_split_by_a_line_break_still_pairs_the_column() -> None:
    gt = "<tr><th>Item</th><th>Age</th></tr>" + _ROWS
    pred = "<tr><th>Item</th><th>Rate<br>Age</th></tr>" + _ROWS
    assert _trm(gt, pred) == pytest.approx(1.0)


def test_exact_key_wins_over_an_affix_key() -> None:
    gt = "<tr><th>Age</th><th>Rate Age</th></tr><tr><td>1</td><td>9</td></tr><tr><td>2</td><td>8</td></tr>"
    pred = "<tr><th>Rate Age</th><th>Age</th></tr><tr><td>9</td><td>1</td></tr><tr><td>8</td><td>2</td></tr>"
    assert _trm(gt, pred) == pytest.approx(1.0)
