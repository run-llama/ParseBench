"""Tests for the parse ``text_similarity`` metric.

Guards the scale of the reported value: autoevals ``Levenshtein`` already
returns a normalized score in ``[0, 1]``, so a near-identical pair must land in
the ~0.7 range (not ~0.007). A regression here previously divided that score by
100, collapsing every value by two orders of magnitude.
"""

from autoevals.string import Levenshtein

from parse_bench.evaluation.metrics.parse.text_similarity_metric import (
    TextSimilarityMetric,
)


def test_autoevals_levenshtein_score_is_already_normalized() -> None:
    """Document the autoevals contract this metric relies on: ``score`` is in
    ``[0, 1]`` (1.0 identical, 0.0 fully disjoint) — NOT a 0-100 scale."""
    lev = Levenshtein()
    assert lev("hello world", "hello world").score == 1.0
    assert lev("abcdefghij", "zzzzzzzzzz").score == 0.0
    similar = lev("The quick brown fox", "The quick brown fox jumps").score
    assert 0.6 <= similar <= 0.9


def test_text_similarity_near_identical_pair_is_on_0_1_scale() -> None:
    """A near-identical pair must score in the ~0.7 range on the correct 0-1
    scale — not ~0.007, which is what the erroneous ``/100`` produced."""
    metric = TextSimilarityMetric()
    result = metric.compute(
        expected="The quick brown fox",
        actual="The quick brown fox jumps",
    )
    # The load-bearing assertion: the value is on the real 0-1 scale.
    # This FAILS on the buggy ``result.score / 100.0`` (which yields ~0.0076).
    assert result.value > 0.5
    assert 0.6 <= result.value <= 0.9
    # And it must equal the raw autoevals score (no rescaling applied).
    assert result.value == result.metadata["levenshtein_score"]


def test_text_similarity_identical_is_one() -> None:
    """Identical strings score exactly 1.0."""
    metric = TextSimilarityMetric()
    result = metric.compute(expected="same text", actual="same text")
    assert result.value == 1.0


def test_text_similarity_both_empty_is_one() -> None:
    """Two empty inputs are trivially identical."""
    metric = TextSimilarityMetric()
    result = metric.compute(expected="", actual="")
    assert result.value == 1.0


def test_text_similarity_one_empty_is_zero() -> None:
    """One empty, one non-empty scores 0.0."""
    metric = TextSimilarityMetric()
    result = metric.compute(expected="", actual="non-empty")
    assert result.value == 0.0
