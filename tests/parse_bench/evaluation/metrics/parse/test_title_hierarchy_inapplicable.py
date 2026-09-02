"""`title_hierarchy_percent` must be skipped, not scored 0.0, on inapplicable input.

A ``title_hierarchy`` whose labels all normalize away (e.g. harvested from an
ASCII banner: ``{"**": {}}``) carries no evaluable constraint. Scoring it 0.0
zeroes ``rule_title_hierarchy_percent_pass_rate``, which drags
``normalized_title_accuracy`` and then ``semantic_formatting`` to 0 on documents
that parsed correctly. Such a rule is excluded from aggregation instead.

A hierarchy that IS defined but is not satisfied by the output must still score
low - only the undefined case is skipped.
"""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.rule_based_metric import RuleBasedMetric
from parse_bench.evaluation.metrics.parse.rules_base import RuleNotApplicable
from parse_bench.evaluation.metrics.parse.test_rules import TitleHierarchyPercentRule

# Content that genuinely satisfies the well-formed hierarchy used below.
_GOOD_CONTENT = """
# Executive Summary
## Revenue Breakdown
"""

_WELL_FORMED_HIERARCHY = {"Executive Summary": {"Revenue Breakdown": {}}}


def _rule(hierarchy: dict) -> TitleHierarchyPercentRule:
    return TitleHierarchyPercentRule({"type": "title_hierarchy_percent", "title_hierarchy": hierarchy})


# ---------------------------------------------------------------------------
# The degenerate cases: no label survives normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hierarchy", "case"),
    [
        ({"**": {}}, "markdown bold banner"),
        ({"~~": {}}, "strikethrough markers"),
        ({"   ": {}}, "whitespace only"),
        ({"": {}}, "empty string label"),
        ({"**": {}, "~~": {}}, "several degenerate labels"),
    ],
)
def test_degenerate_hierarchy_is_skipped(hierarchy: dict, case: str) -> None:
    """No root label survives normalization -> the rule is inapplicable."""
    with pytest.raises(RuleNotApplicable):
        _rule(hierarchy).run("# Some Real Heading\n\nBody text.\n")


def test_label_that_survives_normalization_is_not_degenerate() -> None:
    """Guard on the boundary: ``***`` normalizes to ``*``, which IS a label.

    Only labels that normalize to nothing make the rule inapplicable; a rule
    with a weird-but-non-empty label stays a real, failable constraint.
    """
    passed, _, score = _rule({"***": {}}).run("# Some Real Heading\n")
    assert not passed
    assert score == 0.0


def test_empty_hierarchy_is_skipped() -> None:
    """An absent/empty hierarchy is undefined, not a failure."""
    with pytest.raises(RuleNotApplicable):
        _rule({}).run("# Some Real Heading\n")


def test_degenerate_root_with_degenerate_children_is_skipped() -> None:
    with pytest.raises(RuleNotApplicable):
        _rule({"**": {"~~": {}}}).run("# A\n## B\n")


# ---------------------------------------------------------------------------
# Guards: defined hierarchies keep their existing behaviour
# ---------------------------------------------------------------------------


def test_correct_hierarchy_still_scores_one() -> None:
    passed, message, score = _rule(_WELL_FORMED_HIERARCHY).run(_GOOD_CONTENT)
    assert passed
    assert message == ""
    assert score == 1.0


def test_real_hierarchy_mismatch_still_scores_below_one() -> None:
    """A defined hierarchy the output violates must still be penalised."""
    content = """
## Revenue Breakdown
# Executive Summary
"""
    passed, _, score = _rule(_WELL_FORMED_HIERARCHY).run(content)
    assert not passed
    assert 0.0 <= score < 1.0


def test_missing_titles_still_score_zero_not_skipped() -> None:
    """Output with no titles at all is a real failure, not an inapplicable rule."""
    passed, _, score = _rule(_WELL_FORMED_HIERARCHY).run("Just a paragraph, no headings.\n")
    assert not passed
    assert score == 0.0


def test_partially_degenerate_hierarchy_is_still_evaluated() -> None:
    """One surviving label is enough to make the rule applicable."""
    hierarchy = {"**": {}, "Executive Summary": {}}
    passed, message, score = _rule(hierarchy).run(_GOOD_CONTENT)
    assert passed, message
    assert score == 1.0


# ---------------------------------------------------------------------------
# Aggregation: a skipped rule must leave no trace in the rollup
# ---------------------------------------------------------------------------


def _degenerate_rule_payload() -> dict:
    return {"type": "title_hierarchy_percent", "title_hierarchy": {"**": {}}}


def _is_title_payload() -> dict:
    return {"type": "is_title", "text": "Executive Summary"}


def test_skipped_rule_absent_from_rule_results() -> None:
    metric = RuleBasedMetric()
    result = metric.compute(
        expected=[_is_title_payload(), _degenerate_rule_payload()],
        actual=_GOOD_CONTENT,
    )
    types = [r.get("type") for r in result.metadata["rule_results"]]
    assert "title_hierarchy_percent" not in types
    assert types == ["is_title"]


def test_skipped_rule_does_not_change_aggregate() -> None:
    """Adding the degenerate rule to a document must not move any number."""
    metric = RuleBasedMetric()
    without = metric.compute(expected=[_is_title_payload()], actual=_GOOD_CONTENT)
    with_degenerate = metric.compute(
        expected=[_is_title_payload(), _degenerate_rule_payload()],
        actual=_GOOD_CONTENT,
    )

    assert with_degenerate.value == without.value == 1.0
    assert with_degenerate.metadata["total"] == without.metadata["total"] == 1
    assert with_degenerate.metadata["passed"] == without.metadata["passed"] == 1


def test_skipped_count_is_reported() -> None:
    metric = RuleBasedMetric()
    result = metric.compute(
        expected=[_is_title_payload(), _degenerate_rule_payload()],
        actual=_GOOD_CONTENT,
    )
    assert result.metadata["skipped_inapplicable"] == 1


def test_document_with_only_inapplicable_rules_is_not_penalised() -> None:
    """Nothing evaluable left -> same convention as a document with no rules."""
    metric = RuleBasedMetric()
    result = metric.compute(expected=[_degenerate_rule_payload()], actual=_GOOD_CONTENT)
    assert result.value == 1.0
    assert result.metadata["total"] == 0
    assert result.metadata["rule_results"] == []


def test_applicable_hierarchy_rule_still_aggregated() -> None:
    """Guard: a defined hierarchy rule keeps contributing to the rollup."""
    metric = RuleBasedMetric()
    result = metric.compute(
        expected=[{"type": "title_hierarchy_percent", "title_hierarchy": _WELL_FORMED_HIERARCHY}],
        actual="Just a paragraph.\n",
    )
    types = [r.get("type") for r in result.metadata["rule_results"]]
    assert types == ["title_hierarchy_percent"]
    assert result.metadata["total"] == 1
    assert result.value == 0.0
