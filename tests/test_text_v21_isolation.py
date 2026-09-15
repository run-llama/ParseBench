"""Legacy benchmarks must not invoke the opt-in projection, even with side inputs."""

import pytest

from parse_bench.evaluation.metrics.parse import text_v21
from parse_bench.evaluation.metrics.parse.rule_based_metric import RuleBasedMetric
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput


@pytest.mark.parametrize("actual", ["", "alpha", "alpha alpha", "other"])
def test_legacy_rules_never_project_structured_sections(monkeypatch, actual):
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy benchmark invoked v2.1 projection")

    monkeypatch.setattr(text_v21, "delivered_markdown", forbidden)
    output = ParseOutput(
        example_id="example",
        pipeline_name="test",
        markdown=actual,
        pages=[PageIR(page_index=0, markdown=actual)],
        layout_pages=[ParseLayoutPageIR(page_number=1, items=[], page_header_markdown="alpha")],
    )
    rules = [{"type": "missing_word_percent", "bag_of_word": {"alpha": 1}}]
    metric = RuleBasedMetric()
    plain = metric.compute(rules, actual)
    structured = metric.compute(rules, actual, parse_output=output)
    assert plain.value == structured.value
    assert plain.metadata["rule_results"] == structured.metadata["rule_results"]
