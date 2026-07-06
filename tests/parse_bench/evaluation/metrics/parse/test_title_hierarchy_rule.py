"""Tests for TitleHierarchyPercentRule ordering constraints.

Sibling order was only enforced for children under a common parent; the
top-level roots had no order edges, so a multi-root hierarchy scored full
marks no matter which order the roots appeared in.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule

TWO_ROOT_HIERARCHY = {
    "Introduction": ["Background", "Scope"],
    "Methods": ["Sampling", "Analysis"],
}


def _run(md: str) -> tuple[bool, float]:
    rule = create_test_rule({"type": "title_hierarchy_percent", "title_hierarchy": TWO_ROOT_HIERARCHY})
    passed, _, score = rule.run(md)
    return passed, score


def _doc(*headings: str) -> str:
    return "\n\n".join(headings) + "\n"


def test_roots_in_order_scores_full():
    md = _doc(
        "# Introduction",
        "## Background",
        "## Scope",
        "# Methods",
        "## Sampling",
        "## Analysis",
    )
    passed, score = _run(md)
    assert passed
    assert score == 1.0


def test_roots_out_of_order_penalized():
    # Same content, root sections swapped: every title and child edge is
    # satisfied, but the root order edge must fail.
    md = _doc(
        "# Methods",
        "## Sampling",
        "## Analysis",
        "# Introduction",
        "## Background",
        "## Scope",
    )
    passed, score = _run(md)
    assert not passed
    assert score < 1.0


def test_children_out_of_order_still_penalized():
    md = _doc(
        "# Introduction",
        "## Scope",
        "## Background",
        "# Methods",
        "## Sampling",
        "## Analysis",
    )
    passed, score = _run(md)
    assert not passed
    assert score < 1.0


def test_single_root_unaffected():
    rule = create_test_rule(
        {"type": "title_hierarchy_percent", "title_hierarchy": {"Overview": ["Part One", "Part Two"]}}
    )
    passed, _, score = rule.run(_doc("# Overview", "## Part One", "## Part Two"))
    assert passed
    assert score == 1.0


def test_duplicate_roots_after_normalization_do_not_poison_score():
    # "**Intro**" and "Intro" normalize to the same title; the order edges
    # must not include a self-edge that no document can satisfy.
    rule = create_test_rule(
        {"type": "title_hierarchy_percent", "title_hierarchy": {"**Intro**": None, "Intro": None, "Body": None}}
    )
    passed, _, score = rule.run("# Intro\n\n# Body\n")
    assert passed
    assert score == 1.0
