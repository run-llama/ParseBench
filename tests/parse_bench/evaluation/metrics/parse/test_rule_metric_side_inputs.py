"""``RuleBasedMetric`` hands rules the side inputs they declare.

- ``_prepare_rule`` injects ``raw_output`` / ``source_file_path`` /
  ``test_case_path`` into any rule that declares the attribute (built-in or
  extension), and leaves values the rule already carries alone.
- Chart rules on one document share ONE parsed-table cache, and a caller can
  pass its own ``chart_table_cache`` to read the parse back afterwards.
- A subclass can extend ``_prepare_rule`` and sees every created rule.
- ``ParseEvaluator`` forwards the staged source path and the test case's own
  path, so a ``diagram_graph`` rule can find its ``reference_image``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from parse_bench.evaluation.evaluators.parse import ParseEvaluator
from parse_bench.evaluation.metrics.parse import rule_based_metric
from parse_bench.evaluation.metrics.parse.rule_based_metric import ChartTableCache, RuleBasedMetric
from parse_bench.evaluation.metrics.parse.rules_base import ParseTestRule
from parse_bench.evaluation.metrics.parse.rules_diagram import DiagramGraphRule
from parse_bench.extensions import ParseRuleBase, register_rule_type
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.test_cases.schema import ParseTestCase


class _SideInputRuleSchema(ParseRuleBase):
    type: Literal["side_input_probe"] = "side_input_probe"


class _SideInputRule(ParseTestRule):
    """Declares the three side inputs and reports what it received."""

    seen: list[dict[str, Any]] = []

    def __init__(self, rule_data: ParseRuleBase | dict):
        super().__init__(rule_data)
        self.raw_output: dict[str, Any] | None = None
        self.source_file_path: str | None = None
        self.test_case_path: str | None = None

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        _SideInputRule.seen.append(
            {
                "raw_output": self.raw_output,
                "source_file_path": self.source_file_path,
                "test_case_path": self.test_case_path,
                "parse_output": self.parse_output,
            }
        )
        return True, "ok"


register_rule_type("side_input_probe", _SideInputRuleSchema, _SideInputRule)

_CHART_MD = "| Name | Value |\n|---|---|\n| Global | 16 |\n| Local | 4 |\n"
_CHART_RULES = [
    {"type": "chart_data_point", "value": 16, "labels": ["Global"]},
    {"type": "chart_data_point", "value": 4, "labels": ["Local"]},
    {"type": "chart_data_point", "value": 16, "labels": ["Global"], "id": "again"},
]


def test_declared_side_inputs_are_injected() -> None:
    _SideInputRule.seen.clear()
    raw = {"pages": [{"original_orientation_angle": 0}]}
    parse_output = ParseOutput(example_id="e", pipeline_name="p", markdown="hello")
    result = RuleBasedMetric().compute(
        [{"type": "side_input_probe"}],
        "hello",
        raw_output=raw,
        parse_output=parse_output,
        source_file_path=Path("/staged/doc.pdf"),
        test_case_file_path="/data/group/doc.pdf",
    )
    assert result.value == 1.0
    assert _SideInputRule.seen == [
        {
            "raw_output": raw,
            "source_file_path": "/staged/doc.pdf",
            "test_case_path": "/data/group/doc.pdf",
            "parse_output": parse_output,
        }
    ]


def test_rule_without_the_attribute_is_left_alone() -> None:
    # ``present`` declares none of the side inputs: nothing is bolted on.
    captured: list[ParseTestRule] = []

    class _Spy(RuleBasedMetric):
        def _prepare_rule(self, rule: ParseTestRule, actual: str, kwargs: dict[str, Any]) -> None:
            super()._prepare_rule(rule, actual, kwargs)
            captured.append(rule)

    _Spy().compute([{"type": "present", "text": "hello"}], "hello", raw_output={"x": 1}, source_file_path="a")
    assert len(captured) == 1
    assert not hasattr(captured[0], "raw_output")
    assert not hasattr(captured[0], "source_file_path")


def test_subclass_hook_sees_every_rule_and_can_override_inputs() -> None:
    _SideInputRule.seen.clear()

    class _Harness(RuleBasedMetric):
        def _prepare_rule(self, rule: ParseTestRule, actual: str, kwargs: dict[str, Any]) -> None:
            super()._prepare_rule(rule, actual, kwargs)
            if isinstance(rule, _SideInputRule):
                rule.raw_output = {"from": "harness"}

    _Harness().compute([{"type": "side_input_probe"}, {"type": "present", "text": "x"}], "x", raw_output={"a": 1})
    assert _SideInputRule.seen[0]["raw_output"] == {"from": "harness"}


def test_diagram_rule_receives_test_case_path() -> None:
    captured: list[DiagramGraphRule] = []

    class _Spy(RuleBasedMetric):
        def _prepare_rule(self, rule: ParseTestRule, actual: str, kwargs: dict[str, Any]) -> None:
            super()._prepare_rule(rule, actual, kwargs)
            assert isinstance(rule, DiagramGraphRule)
            captured.append(rule)

    rule = {"type": "diagram_graph", "graph": {"nodes": [{"id": "a", "label": "A"}], "edges": []}}
    _Spy().compute([rule], "no diagram here", source_file_path="/staged/d.pdf", test_case_file_path="/tc/d.pdf")
    assert captured[0].source_file_path == "/staged/d.pdf"
    assert captured[0].test_case_path == "/tc/d.pdf"


def test_chart_rules_share_one_table_parse(monkeypatch) -> None:
    calls = 0
    real = rule_based_metric.parse_chart_tables

    def counting(content: str):
        nonlocal calls
        calls += 1
        return real(content)

    monkeypatch.setattr(rule_based_metric, "parse_chart_tables", counting)
    injected: list[Any] = []

    class _Spy(RuleBasedMetric):
        def _prepare_rule(self, rule: ParseTestRule, actual: str, kwargs: dict[str, Any]) -> None:
            super()._prepare_rule(rule, actual, kwargs)
            injected.append(rule.parsed_tables)

    result = _Spy().compute(_CHART_RULES, _CHART_MD)
    assert result.value == 1.0
    assert calls == 1
    assert len(injected) == 3
    assert all(tables is injected[0] for tables in injected)


def test_caller_supplied_cache_is_populated() -> None:
    cache = ChartTableCache()
    RuleBasedMetric().compute(_CHART_RULES, _CHART_MD, chart_table_cache=cache)
    assert cache.populated
    assert len(cache.tables) == 1

    # A cache handed in already populated is reused, not re-parsed.
    again = ChartTableCache(tables=cache.tables, populated=True)
    RuleBasedMetric().compute(_CHART_RULES, _CHART_MD, chart_table_cache=again)
    assert again.tables is cache.tables


def test_evaluator_forwards_source_and_test_case_paths(tmp_path: Path) -> None:
    evaluator = ParseEvaluator(enable_grits=False, enable_structural_consistency=False)
    seen: dict[str, Any] = {}
    original = evaluator._rule_metric.compute

    def spy(expected: Any, actual: str, page: int | None = None, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return original(expected, actual, page, **kwargs)

    evaluator._rule_metric.compute = spy  # type: ignore[method-assign]

    tc_file = tmp_path / "doc.pdf"
    tc_file.write_bytes(b"")
    test_case = ParseTestCase(
        test_id="g/doc",
        group="g",
        file_path=tc_file,
        test_rules=[{"type": "present", "text": "hello"}],
    )
    now = datetime.now(UTC)
    inference_result = InferenceResult(
        request=InferenceRequest(example_id="e", source_file_path="/staged/elsewhere.pdf", product_type="parse"),
        pipeline_name="p",
        product_type="parse",
        raw_output={},
        output=ParseOutput(example_id="e", pipeline_name="p", markdown="hello"),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )
    evaluator.evaluate(inference_result, test_case)
    assert seen["source_file_path"] == "/staged/elsewhere.pdf"
    assert seen["test_case_file_path"] == str(tc_file)
