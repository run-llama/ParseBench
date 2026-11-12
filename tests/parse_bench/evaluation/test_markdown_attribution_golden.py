"""The scorer's output on real saved outputs stays byte-identical.

The fixture holds saved parse outputs of five v0.3 arms (cost_effective and
agentic, old and new engine, plus an engine-fix arm) on 12 pages, the
``markdown_attribution`` rules those pages had before scorer version 2 (they
say ``version: 1`` and carry no ``alt_texts``), and SHA-256 digests of
``score_markdown_attribution`` and ``compute_markdown_attribution_metrics`` as
scorer version 2 computed them (commit f8fca7f2a3, before the version 1 code
path was removed). Any change to what the scorer returns breaks this test; a
deliberate change regenerates ``digests.json`` and says why in the commit.

``MD_ATTRIBUTION_GOLDEN`` may point at a bundle of the same shape with all
100 pages (``outputs.json.gz``, ``rules.json.gz``, ``digests.json``).
"""

import gzip
import hashlib
import json
import os
import warnings
from pathlib import Path

from parse_bench.evaluation.markdown_attribution import (
    compute_markdown_attribution_metrics,
    score_markdown_attribution,
)
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.test_cases.markdown_attribution_schema import MarkdownAttributionRule

BUNDLE = Path(os.environ.get("MD_ATTRIBUTION_GOLDEN") or Path(__file__).parent / "fixtures/md_attribution_golden")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=repr).encode()).hexdigest()


def _load() -> tuple[dict, dict, dict]:
    with gzip.open(BUNDLE / "outputs.json.gz", "rt") as f:
        outputs = json.load(f)
    with gzip.open(BUNDLE / "rules.json.gz", "rt") as f:
        rules = json.load(f)
    return outputs, rules, json.loads((BUNDLE / "digests.json").read_text())


def _mismatches(rule_of) -> list[str]:
    outputs, rules, expected = _load()
    assert len(expected) >= 60
    mismatches = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # expected_bbox_count notes on a few references
        for key, digests in sorted(expected.items()):
            arm, stem = key.split("/")
            rule = MarkdownAttributionRule.model_validate(rule_of(rules[stem]))
            output = ParseOutput.model_validate(outputs[arm][stem])
            result = score_markdown_attribution(output, rule)
            metrics = [m.model_dump() for m in compute_markdown_attribution_metrics(output, [rule])]
            if [_digest(result), _digest(metrics)] != digests:
                mismatches.append(key)
    return mismatches


def test_results_are_byte_identical_to_the_frozen_version_2_output():
    # The fixture rules say version 1: they are scored by version 2 all the same.
    _, rules, _ = _load()
    assert {rule.get("version") for rule in rules.values()} == {1}
    mismatches = _mismatches(lambda rule: rule)
    assert not mismatches, f"scorer output changed on {len(mismatches)} page(s): {mismatches[:5]}"


def test_a_rule_without_a_version_scores_the_same():
    mismatches = _mismatches(lambda rule: {k: v for k, v in rule.items() if k != "version"})
    assert not mismatches, f"a rule without version scored differently on {mismatches[:5]}"
