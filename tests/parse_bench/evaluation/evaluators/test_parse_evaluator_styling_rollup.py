"""Regression tests for the normalized text-styling rollup."""

import pytest

from parse_bench.evaluation.evaluators.parse import _styling_f_beta_score


def test_styling_f_beta_penalizes_negative_rule_failure_more_heavily() -> None:
    false_styling_score = _styling_f_beta_score(pos_score=1.0, neg_score=0.5)
    missed_styling_score = _styling_f_beta_score(pos_score=0.5, neg_score=1.0)

    assert false_styling_score == pytest.approx(5 / 9)
    assert missed_styling_score == pytest.approx(5 / 6)
    assert false_styling_score < missed_styling_score
