"""Tests that every FormattingRule styling kind reaches the headline scores.

``_TEXT_STYLING_PAIRS`` used to list only bold/strikeout/sup/sub, so
``is_italic``, ``is_underline`` and ``is_mark`` rules were executed and
reported as ``rule_<type>_pass_rate`` sub-metrics but never folded into
``normalized_text_styling`` — and therefore never into ``semantic_formatting``.
A pipeline scoring 0% on those three kinds got the same headline as one
scoring 100%.
"""

from __future__ import annotations

from datetime import UTC, datetime

from parse_bench.evaluation.evaluators.parse import _TEXT_STYLING_PAIRS, ParseEvaluator
from parse_bench.evaluation.metrics.parse.rules_formatting import FormattingRule
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import ParseTestCase

# Styled so that is_italic / is_underline / is_mark all pass.
STYLED_MD = "The *alpha* and <u>beta</u> and <mark>gamma</mark> terms."
# Same words, no markup: the same three rules all fail.
PLAIN_MD = "The alpha and beta and gamma terms."

STYLING_RULES = [
    {"type": "is_italic", "text": "alpha"},
    {"type": "is_underline", "text": "beta"},
    {"type": "is_mark", "text": "gamma"},
]


def _evaluate(markdown: str, rules: list[dict[str, str]]) -> dict[str, float]:
    """Run the rule-based evaluator and return metric_name -> value."""
    evaluator = ParseEvaluator(
        enable_rule_based=True,
        enable_grits=False,
        enable_structural_consistency=False,
        enable_table_record_match=False,
    )
    test_case = ParseTestCase(
        test_id="text_styling/doc",
        group="text_styling",
        file_path="doc.pdf",
        test_rules=rules,  # type: ignore[arg-type]
    )
    request = InferenceRequest(
        example_id="doc",
        source_file_path="doc.pdf",
        product_type=ProductType.PARSE,
    )
    now = datetime.now(UTC)
    inference_result = InferenceResult(
        request=request,
        pipeline_name="test",
        product_type=ProductType.PARSE,
        raw_output={},
        output=ParseOutput(example_id="doc", pipeline_name="test", markdown=markdown),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    result = evaluator.evaluate(inference_result, test_case)
    return {m.metric_name: m.value for m in result.metrics}


class TestStylingPairsCoverage:
    def test_pairs_cover_every_formatting_kind(self):
        """Guards against a styling kind being added to FormattingRule but not scored."""
        scored_kinds = {pos.removeprefix("is_") for pos, _ in _TEXT_STYLING_PAIRS}
        assert scored_kinds == set(FormattingRule._FORMATTING_PATTERNS)

    def test_pairs_are_well_formed(self):
        for pos, neg in _TEXT_STYLING_PAIRS:
            assert neg == pos.replace("is_", "is_not_", 1)


class TestItalicUnderlineMarkAffectHeadline:
    def test_styling_rules_produce_headline_metrics(self):
        metrics = _evaluate(STYLED_MD, STYLING_RULES)
        assert "normalized_text_styling" in metrics
        assert "semantic_formatting" in metrics

    def test_all_passing_scores_full(self):
        metrics = _evaluate(STYLED_MD, STYLING_RULES)
        assert metrics["normalized_text_styling"] == 1.0
        assert metrics["semantic_formatting"] == 1.0

    def test_all_failing_scores_zero(self):
        metrics = _evaluate(PLAIN_MD, STYLING_RULES)
        assert metrics["normalized_text_styling"] == 0.0
        assert metrics["semantic_formatting"] == 0.0

    def test_headline_separates_passing_from_failing(self):
        """The regression itself: these two used to be indistinguishable."""
        styled = _evaluate(STYLED_MD, STYLING_RULES)
        plain = _evaluate(PLAIN_MD, STYLING_RULES)
        assert styled["semantic_formatting"] > plain["semantic_formatting"]

    def test_each_kind_contributes_individually(self):
        """Failing any single kind must move the headline on its own."""
        for rule in STYLING_RULES:
            styled = _evaluate(STYLED_MD, [rule])
            plain = _evaluate(PLAIN_MD, [rule])
            assert styled["normalized_text_styling"] > plain["normalized_text_styling"], rule["type"]

    def test_negative_rules_are_scored(self):
        """is_not_* counterparts must also reach the headline."""
        negative_rules = [
            {"type": "is_not_italic", "text": "alpha"},
            {"type": "is_not_underline", "text": "beta"},
            {"type": "is_not_mark", "text": "gamma"},
        ]
        # Plain markdown satisfies the negative rules; styled markdown violates them.
        assert _evaluate(PLAIN_MD, negative_rules)["normalized_text_styling"] == 1.0
        assert _evaluate(STYLED_MD, negative_rules)["normalized_text_styling"] == 0.0
