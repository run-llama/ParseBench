"""Text-query rules whose text is pure markdown markers must be skipped, not scored 0.0.

The annotation pass harvests rule text from ASCII banners and decorative rules,
producing rules such as ``is_title`` with text ``"**"`` or ``is_bold`` with text
``"*"``. ``normalize_text`` reduces such text to markers or to nothing, so the
query can never identify content in any output: the rule is unsatisfiable by
construction and permanently scores 0.0. Those zeros pin
``normalized_title_accuracy`` and ``normalized_text_styling``, and through them
``semantic_formatting``, on documents that parsed correctly.

This is the sibling family of the ``title_hierarchy_percent`` case fixed in
#2296, and it reuses that PR's ``RuleNotApplicable`` mechanism: the rule emits
no result at all, so it leaves every numerator and every denominator.

Rules whose text merely *contains* markers around real words stay real,
failable constraints.
"""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.rule_based_metric import RuleBasedMetric
from parse_bench.evaluation.metrics.parse.rules_base import RuleNotApplicable
from parse_bench.evaluation.metrics.parse.test_rules import (
    FormattingRule,
    TitleLevelRule,
)

# Content that genuinely satisfies the well-formed rules used as guards below.
_GOOD_CONTENT = """
# Executive Summary

**Revenue** grew, and *margins* held.
"""


def _rule(rule_type: str, text: str):
    payload = {"type": rule_type, "text": text}
    if rule_type == "is_title":
        return TitleLevelRule(payload)
    return FormattingRule(payload)


# ---------------------------------------------------------------------------
# The degenerate family: text is nothing but markdown markers
# ---------------------------------------------------------------------------

# Every rule type observed carrying degenerate text in text_extended/v1.0.
_DEGENERATE_TYPES = ["is_title", "is_bold", "is_italic", "is_sup"]

# ``"*"`` and ``"**"`` are the texts actually observed in the corpus; the others
# are the rest of the marker class the annotation tool strips (``[*_~]``).
_DEGENERATE_TEXTS = ["*", "**", "~~", "~", "_", "__", "***", "* *"]


@pytest.mark.parametrize("rule_type", _DEGENERATE_TYPES)
@pytest.mark.parametrize("text", _DEGENERATE_TEXTS)
def test_marker_only_text_is_skipped(rule_type: str, text: str) -> None:
    """Pure-marker text carries no query, so the rule is inapplicable."""
    with pytest.raises(RuleNotApplicable):
        _rule(rule_type, text).run(_GOOD_CONTENT)


def test_triple_star_is_skipped_boundary() -> None:
    """Boundary: ``"***"`` normalizes to ``"*"`` - still marker-only, still skipped.

    This is deliberately *wider* than the guard #2296 used for
    ``title_hierarchy_percent``, which skips only when a label normalizes to the
    empty string and therefore keeps ``"***"`` as a real (failing) label. The
    two differ because the shapes differ: a hierarchy is a set of labels where a
    single odd one still participates in real parent/child constraints, whereas
    these rules are a *single* text query - if that query is markers, 100% of
    the rule is degenerate and there is nothing left to measure. On
    text_extended/v1.0 the divergence covers 0 rules: the only hierarchy whose
    labels are all degenerate is the all-empty one #2296 already skips.
    """
    with pytest.raises(RuleNotApplicable):
        _rule("is_bold", "***").run(_GOOD_CONTENT)


def test_negative_polarity_is_also_skipped() -> None:
    """``is_not_bold "*"`` would otherwise trivially *pass* for a degenerate reason.

    The degeneracy is a property of the query, not of the polarity, so it is
    excluded rather than banked as a free pass.
    """
    with pytest.raises(RuleNotApplicable):
        _rule("is_not_bold", "*").run(_GOOD_CONTENT)


def test_whitespace_only_text_still_rejected_at_construction() -> None:
    """Unchanged: blank text never builds a rule in the first place."""
    with pytest.raises(ValueError):
        _rule("is_bold", "   ")


# ---------------------------------------------------------------------------
# Guards: real rules must keep behaving exactly as before
# ---------------------------------------------------------------------------


def test_text_containing_markers_still_runs() -> None:
    """``"*Important*"`` normalizes to ``Important`` - a real query, not markers."""
    rule = _rule("is_bold", "*Important*")
    passed, _ = rule.run("Plain text with no emphasis at all.\n")
    assert passed is False


def test_real_bold_rule_still_passes() -> None:
    passed, _ = _rule("is_bold", "Revenue").run(_GOOD_CONTENT)
    assert passed is True


def test_real_italic_rule_still_passes() -> None:
    passed, _ = _rule("is_italic", "margins").run(_GOOD_CONTENT)
    assert passed is True


def test_real_title_rule_still_passes() -> None:
    passed, _ = _rule("is_title", "Executive Summary").run(_GOOD_CONTENT)
    assert passed is True


def test_real_bold_miss_still_fails() -> None:
    """A well-defined rule the output does not satisfy is a real failure."""
    passed, _ = _rule("is_bold", "Revenue").run("Revenue grew.\n")
    assert passed is False


def test_single_marker_inside_words_is_not_degenerate() -> None:
    """A star adjacent to real text keeps the rule evaluable."""
    with_text = _rule("is_bold", "* Revenue")
    passed, _ = with_text.run("Plain text.\n")
    assert passed is False


# ---------------------------------------------------------------------------
# Aggregation: skipped rules leave numerator *and* denominator
# ---------------------------------------------------------------------------


def _degenerate_payloads() -> list[dict]:
    """The exact family observed on text_ocr/bold and text_sparse/agenda_sparse."""
    return [
        {"type": "is_title", "text": "*"},
        {"type": "is_bold", "text": "*"},
        {"type": "is_italic", "text": "*"},
    ]


def _real_payload() -> dict:
    return {"type": "is_bold", "text": "Revenue"}


def test_degenerate_rules_absent_from_rule_results() -> None:
    metric = RuleBasedMetric()
    result = metric.compute(expected=[_real_payload(), *_degenerate_payloads()], actual=_GOOD_CONTENT)
    types = [r.get("type") for r in result.metadata["rule_results"]]
    assert types == ["is_bold"]


def test_degenerate_rules_do_not_change_aggregate() -> None:
    """Adding the degenerate family to a document must not move any number."""
    metric = RuleBasedMetric()
    without = metric.compute(expected=[_real_payload()], actual=_GOOD_CONTENT)
    with_degenerate = metric.compute(
        expected=[_real_payload(), *_degenerate_payloads()],
        actual=_GOOD_CONTENT,
    )

    assert with_degenerate.value == without.value == 1.0
    assert with_degenerate.metadata["total"] == without.metadata["total"] == 1
    assert with_degenerate.metadata["passed"] == without.metadata["passed"] == 1


def test_skipped_count_is_reported() -> None:
    metric = RuleBasedMetric()
    result = metric.compute(expected=[_real_payload(), *_degenerate_payloads()], actual=_GOOD_CONTENT)
    assert result.metadata["skipped_inapplicable"] == 3


def test_document_with_only_degenerate_rules_is_not_penalised() -> None:
    metric = RuleBasedMetric()
    result = metric.compute(expected=_degenerate_payloads(), actual=_GOOD_CONTENT)
    assert result.value == 1.0
    assert result.metadata["total"] == 0
    assert result.metadata["rule_results"] == []


def test_real_styling_rules_still_aggregated() -> None:
    """Guard: defined styling rules keep contributing, pass or fail."""
    metric = RuleBasedMetric()
    result = metric.compute(
        expected=[{"type": "is_bold", "text": "Revenue"}],
        actual="Revenue grew.\n",
    )
    types = [r.get("type") for r in result.metadata["rule_results"]]
    assert types == ["is_bold"]
    assert result.metadata["total"] == 1
    assert result.value == 0.0
