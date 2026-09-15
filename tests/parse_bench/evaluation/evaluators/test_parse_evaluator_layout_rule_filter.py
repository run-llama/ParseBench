"""Regression tests for ParseEvaluator layout-rule filtering.

PR #731 (M5b) widened ``ParseTestCase.test_rules`` to accept layout rule
models alongside parse rules. Without filtering, ``ParseEvaluator`` would
route layout rules through the parse rule factory, which can't build them —
each layout rule was counted as a spurious failure, depressing
``rule_pass_rate`` on any mixed dataset.

See the Devin review on PR #731:
https://github.com/run-llama/experimental/pull/731#discussion_r3120682402
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

from parse_bench.evaluation.evaluators.parse import ParseEvaluator
from parse_bench.schemas.evaluation import MetricValue
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.parse_rule_schemas import coerce_parse_rule
from parse_bench.test_cases.schema import LayoutTestRule, ParseTestCase


def _make_inference_result(
    markdown: str = "alpha beta gamma",
    layout_pages: list[ParseLayoutPageIR] | None = None,
    grounded_pages: list[dict[str, Any]] | None = None,
    raw_output: dict[str, Any] | None = None,
    pipeline_name: str = "test_pipeline",
) -> InferenceResult:
    return InferenceResult(
        request=InferenceRequest(
            example_id="grp/doc",
            source_file_path="doc.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline_name=pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw_output or {},
        output=ParseOutput(
            example_id="grp/doc",
            pipeline_name="test_pipeline",
            markdown=markdown,
            layout_pages=layout_pages or [],
            grounded_pages=grounded_pages or [],
        ),
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        latency_in_ms=1,
    )


def _make_layout_rule(canonical_class: str = "Text") -> LayoutTestRule:
    return LayoutTestRule(
        page=1,
        bbox=[0.1, 0.1, 0.2, 0.2],
        canonical_class=canonical_class,
    )


def _make_parse_test_case(test_rules: list) -> ParseTestCase:
    # ``model_construct`` bypasses ``ParseTestCase``'s parse-rule-only validator
    # so tests can inject mixed-rule lists matching the post-M5b on-disk shape.
    return ParseTestCase.model_construct(
        test_id="grp/doc",
        group="grp",
        file_path="doc.pdf",
        test_rules=test_rules,
    )


def _rule_only_evaluator() -> ParseEvaluator:
    """Evaluator with every table/text metric disabled so ``evaluate`` exercises
    only the rule-based branch under test."""
    return ParseEvaluator(
        enable_rule_based=True,
        enable_text_similarity=False,
        enable_teds=False,
        enable_grits=False,
        enable_header_accuracy=False,
        enable_structural_consistency=False,
        enable_table_record_match=False,
        enable_table_composite=False,
    )


def test_mixed_rules_filter_layout_before_rule_metric() -> None:
    """Layout rules are stripped before the parse rule-based metric runs."""
    parse_rule_a = coerce_parse_rule({"type": "present", "text": "alpha"})
    parse_rule_b = coerce_parse_rule({"type": "present", "text": "beta"})
    layout_rule = _make_layout_rule()

    test_case = _make_parse_test_case([parse_rule_a, layout_rule, parse_rule_b])
    inference_result = _make_inference_result(markdown="alpha beta gamma")

    evaluator = _rule_only_evaluator()

    with patch.object(
        evaluator._rule_metric,
        "compute",
        wraps=evaluator._rule_metric.compute,
    ) as mock_compute:
        result = evaluator.evaluate(inference_result, test_case)

    assert mock_compute.call_count == 1
    forwarded = mock_compute.call_args.kwargs["expected"]
    assert forwarded == [parse_rule_a, parse_rule_b]
    assert all(not isinstance(r, LayoutTestRule) for r in forwarded)

    rule_pass = next(m for m in result.metrics if m.metric_name == "rule_pass_rate")
    # Two parse rules, both present in the markdown → pass_rate 1.0 on the
    # filtered 2-rule denominator (NOT 2/3).
    assert rule_pass.value == 1.0
    assert rule_pass.metadata.get("total") == 2


def test_only_layout_rules_skips_rule_based_branch(caplog) -> None:
    """When only layout rules are present, the rule-based branch is skipped
    entirely — no rule_pass_rate=1.0 emitted for zero parse rules."""
    layout_rule = _make_layout_rule()

    test_case = _make_parse_test_case([layout_rule])
    inference_result = _make_inference_result(markdown="non-empty markdown content")

    evaluator = _rule_only_evaluator()

    with patch.object(evaluator._rule_metric, "compute") as mock_compute:
        with caplog.at_level("DEBUG", logger="parse_bench.evaluation.evaluators.parse"):
            result = evaluator.evaluate(inference_result, test_case)

    assert mock_compute.call_count == 0
    assert all(m.metric_name != "rule_pass_rate" for m in result.metrics)
    assert any("only layout rules present" in rec.message for rec in caplog.records)


def test_only_parse_rules_unchanged_from_pre_fix() -> None:
    """Regression guard: with only parse rules, behavior is identical to
    the pre-fix path."""
    parse_rule = coerce_parse_rule({"type": "present", "text": "alpha"})

    test_case = _make_parse_test_case([parse_rule])
    inference_result = _make_inference_result(markdown="alpha beta gamma")

    evaluator = _rule_only_evaluator()

    with patch.object(
        evaluator._rule_metric,
        "compute",
        wraps=evaluator._rule_metric.compute,
    ) as mock_compute:
        result = evaluator.evaluate(inference_result, test_case)

    assert mock_compute.call_count == 1
    forwarded = mock_compute.call_args.kwargs["expected"]
    assert forwarded == [parse_rule]

    rule_pass = next(m for m in result.metrics if m.metric_name == "rule_pass_rate")
    assert rule_pass.value == 1.0
    assert rule_pass.metadata.get("total") == 1


def test_empty_test_rules_markdown_only_skips_rule_branch(caplog) -> None:
    """Existing behavior: empty ``test_rules`` + ``expected_markdown`` skips
    the rule-based branch and emits the 'not provided' debug log."""
    test_case = ParseTestCase(
        test_id="grp/doc",
        group="grp",
        file_path="doc.pdf",
        test_rules=[],
        expected_markdown="# Ground truth heading\n",
    )
    inference_result = _make_inference_result(markdown="# Ground truth heading\n")

    evaluator = _rule_only_evaluator()

    with patch.object(evaluator._rule_metric, "compute") as mock_compute:
        with caplog.at_level("DEBUG", logger="parse_bench.evaluation.evaluators.parse"):
            _ = evaluator.evaluate(inference_result, test_case)

    assert mock_compute.call_count == 0
    assert any("test_rules not provided" in rec.message for rec in caplog.records)


def test_mixed_rules_layout_failure_does_not_poison_metric() -> None:
    """Without the filter, a failing layout rule (ValueError from the parse
    factory) would be caught and counted as failed, depressing pass_rate.
    With the filter, only the parse rule counts toward the denominator."""
    parse_rule = coerce_parse_rule({"type": "present", "text": "alpha"})
    layout_rules = [_make_layout_rule() for _ in range(5)]

    test_case = _make_parse_test_case([parse_rule, *layout_rules])
    inference_result = _make_inference_result(markdown="alpha beta gamma")

    evaluator = _rule_only_evaluator()

    result = evaluator.evaluate(inference_result, test_case)
    rule_pass = next(
        (m for m in result.metrics if m.metric_name == "rule_pass_rate"),
        None,
    )
    assert rule_pass is not None
    # Without the filter: 1 pass / 6 total = 0.1667. With the filter: 1 / 1 = 1.0.
    assert rule_pass.value == 1.0
    assert rule_pass.metadata.get("total") == 1


def test_metric_value_is_metricvalue_instance() -> None:
    """Guard that the wrapping code still returns a MetricValue (not a list)
    when the filter is in effect — downstream code destructures this object."""
    parse_rule = coerce_parse_rule({"type": "present", "text": "alpha"})
    test_case = _make_parse_test_case([parse_rule, _make_layout_rule()])
    inference_result = _make_inference_result(markdown="alpha")

    evaluator = _rule_only_evaluator()

    result = evaluator.evaluate(inference_result, test_case)
    rule_pass = next(m for m in result.metrics if m.metric_name == "rule_pass_rate")
    assert isinstance(rule_pass, MetricValue)


# ---------------------------------------------------------------------------
# Detector-only vs pure-parse gating (Devin PR #757, Finding 1)
# ---------------------------------------------------------------------------
#
# The ``has_markdown`` gate introduced in M5b was too broad: any pipeline that
# returned empty markdown (including *broken* pure-parse pipelines) skipped
# every text metric and returned ``success=True`` with an empty metric list.
# The intended behaviour is to skip text metrics *only* for legitimate
# detector-only outputs (layout rules present + populated ``layout_pages``
# + empty markdown). Broken pure-parse pipelines must still surface a failure
# signal (``rule_pass_rate=0``).


def _layout_page_with_one_item() -> ParseLayoutPageIR:
    return ParseLayoutPageIR(
        page_number=1,
        width=100.0,
        height=100.0,
        items=[
            LayoutItemIR(
                type="text",
                bbox=LayoutSegmentIR(x=0.1, y=0.1, w=0.2, h=0.2, label="Text"),
            )
        ],
    )


def test_broken_pure_parse_pipeline_emits_zero_rule_pass_rate() -> None:
    """Regression (PR #757 Finding 1): a pure-parse test + broken pipeline
    (empty markdown, empty layout_pages) must still emit ``rule_pass_rate=0``
    rather than silently returning an empty metric list."""
    parse_rule = coerce_parse_rule({"type": "present", "text": "alpha"})
    test_case = _make_parse_test_case([parse_rule])
    inference_result = _make_inference_result(markdown="", layout_pages=[])

    evaluator = _rule_only_evaluator()
    result = evaluator.evaluate(inference_result, test_case)

    rule_pass = next(
        (m for m in result.metrics if m.metric_name == "rule_pass_rate"),
        None,
    )
    assert rule_pass is not None, "rule_pass_rate must be emitted for broken pure-parse runs"
    assert rule_pass.value == 0.0


def test_detector_only_output_still_skips_text_metrics() -> None:
    """Existing M5b behaviour: detector-only output (layout rule + populated
    ``layout_pages`` + empty markdown) skips text metrics entirely."""
    layout_rule = _make_layout_rule()
    test_case = _make_parse_test_case([layout_rule])
    inference_result = _make_inference_result(
        markdown="",
        layout_pages=[_layout_page_with_one_item()],
    )

    evaluator = _rule_only_evaluator()
    result = evaluator.evaluate(inference_result, test_case)

    assert all(m.metric_name != "rule_pass_rate" for m in result.metrics)


def test_line_word_layout_rules_do_not_make_output_detector_only() -> None:
    """Line/word layout rules should not suppress text metrics for parse tests.

    Only region-level layout rules identify detector-only outputs. Granular OCR
    rules can coexist with parse rules, so an empty-markdown parse run should
    still emit text-metric failures instead of being silently skipped.
    """
    parse_rule = coerce_parse_rule({"type": "present", "text": "alpha"})
    word_rule = LayoutTestRule(
        id="word-1",
        page=1,
        bbox=[0.1, 0.1, 0.1, 0.05],
        canonical_class="Text",
        content={"type": "text", "text": "alpha"},
        granularity="word",
    )
    test_case = _make_parse_test_case([parse_rule, word_rule])
    inference_result = _make_inference_result(
        markdown="",
        layout_pages=[_layout_page_with_one_item()],
    )

    evaluator = _rule_only_evaluator()
    result = evaluator.evaluate(inference_result, test_case)

    rule_pass = next((m for m in result.metrics if m.metric_name == "rule_pass_rate"), None)
    assert rule_pass is not None
    assert rule_pass.value == 0.0


def _layout_page_without_items() -> ParseLayoutPageIR:
    return ParseLayoutPageIR(page_number=1, width=100.0, height=100.0, items=[])


def test_pipeline_emitting_no_layout_pages_is_still_skipped() -> None:
    """Text-only providers keep their metrics absent, not zeroed.

    They never claimed a layout capability, so scoring them zero on every
    layout document would be a leaderboard change, not a bug fix.
    """
    test_case = _make_parse_test_case([_make_layout_rule()])
    inference_result = _make_inference_result(layout_pages=[])

    result = _rule_only_evaluator().evaluate(inference_result, test_case)

    assert all(m.metric_name != "layout_element_rule_pass_rate" for m in result.metrics)


def _attributed_layout_page(text: str) -> ParseLayoutPageIR:
    return ParseLayoutPageIR(
        page_number=1,
        width=100.0,
        height=100.0,
        md=text,
        items=[
            LayoutItemIR(
                type="text",
                value=text,
                bbox=LayoutSegmentIR(
                    x=10.0,
                    y=10.0,
                    w=20.0,
                    h=20.0,
                    label="text",
                    confidence=1.0,
                ),
                layout_segments=[
                    LayoutSegmentIR(
                        x=10.0,
                        y=10.0,
                        w=20.0,
                        h=20.0,
                        label="text",
                        confidence=1.0,
                        start_index=0,
                        end_index=max(0, len(text) - 1),
                    )
                ],
            )
        ],
    )


def _attribution_rule(text: str) -> LayoutTestRule:
    return LayoutTestRule(
        page=1,
        bbox=[0.1, 0.1, 0.2, 0.2],
        canonical_class="Text",
        content={"type": "text", "text": text},
        ro_index=0,
    )
