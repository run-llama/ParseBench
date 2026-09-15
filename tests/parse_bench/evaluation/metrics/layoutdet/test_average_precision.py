"""Pins the numpy average-precision implementation to scikit-learn's semantics."""

import numpy as np
import pytest

from parse_bench.evaluation.metrics.layoutdet.classification_utils import average_precision_score


@pytest.mark.parametrize(
    ("y_true", "y_score", "expected"),
    [
        # Reference values computed with sklearn.metrics.average_precision_score
        ([1, 0, 1, 1], [0.9, 0.8, 0.7, 0.6], 0.8055555555555556),
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.8333333333333333),
        ([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6], 1.0),
        ([0, 0, 1, 1], [0.9, 0.8, 0.7, 0.6], 0.41666666666666663),
        # Tied scores are grouped at one threshold (sklearn semantics)
        ([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5], 0.5),
        ([1, 0, 0, 1], [0.9, 0.5, 0.5, 0.5], 0.75),
    ],
)
def test_matches_sklearn_reference_values(y_true, y_score, expected):
    ap = average_precision_score(np.array(y_true, dtype=float), np.array(y_score, dtype=float))
    assert ap == pytest.approx(expected)


def test_degenerate_inputs():
    assert average_precision_score(np.array([]), np.array([])) == 0.0
    assert average_precision_score(np.zeros(3), np.array([0.3, 0.2, 0.1])) == 0.0
    assert average_precision_score(np.ones(3), np.array([0.3, 0.2, 0.1])) == 1.0
