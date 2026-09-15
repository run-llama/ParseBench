"""TextOrderRule must match against HTML-stripped text.

A parser may legitimately render a page as an HTML table where the ground truth
is plain markdown.  Before the strip, every order rule on such a page scored 0
regardless of content fidelity, which caps ``content_faithfulness`` at 0.667
(order carries weight 0.5 of 1.5).

These tests pin BOTH directions: correct reading order inside a table passes,
and genuinely wrong reading order inside a table still fails.  Order rules must
not become vacuous on table-bearing pages.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.test_rules import TextOrderRule

# Ground-truth anchors are authored against plain markdown, so each one spans a
# label and its page number — i.e. it crosses a `</td><td>` boundary once the
# parser renders the page as a table.  That span is what markup blindness breaks.
_TOC_RULE = {
    "type": "order",
    "before": "Introduction to the coverage of the plan 103",
    "after": "Appendix listing the expedited review process 108",
    "max_diffs": 2,
}

_INTRO_ROW = "<tr><td>Introduction to the coverage of the plan</td><td>103</td></tr>"
_APPENDIX_ROW = "<tr><td>Appendix listing the expedited review process</td><td>108</td></tr>"


def test_order_rule_passes_when_table_reading_order_is_correct() -> None:
    """The same sentences, rendered as an HTML table, in the GT reading order."""
    rule = TextOrderRule(_TOC_RULE)

    passed, message = rule.run(f"<table>{_INTRO_ROW}{_APPENDIX_ROW}</table>")

    assert passed, message
    assert message == ""


def test_order_rule_still_fails_when_table_reading_order_is_wrong() -> None:
    """Genuinely inverted reading order inside a table must NOT pass.

    This is the guard against making order rules vacuous on tables: stripping
    markup may not turn a real ordering defect into a pass.  Both anchors are
    findable here, so the failure is an ordering verdict, not a lookup miss.
    """
    rule = TextOrderRule(_TOC_RULE)

    passed, message = rule.run(f"<table>{_APPENDIX_ROW}{_INTRO_ROW}</table>")

    assert not passed
    assert "appears before" in message


def test_order_rule_does_not_weld_adjacent_table_cells() -> None:
    """Tags become a separator, so adjacent cells stay separate tokens."""
    rule = TextOrderRule(
        {
            "type": "order",
            "before": "alpha beta",
            "after": "gamma delta",
            "max_diffs": 0,
        }
    )

    passed, message = rule.run("<table><tr><td>alpha</td><td>beta</td><td>gamma</td><td>delta</td></tr></table>")

    assert passed, message


def test_order_rule_preserves_plain_markdown_pass() -> None:
    """Plain-markdown documents keep their existing behaviour."""
    rule = TextOrderRule(_TOC_RULE)

    passed, message = rule.run(
        "# Handbook\n\n"
        "Introduction to the coverage of the plan .......... 103\n\n"
        "Appendix listing the expedited review process ...... 108\n"
    )

    assert passed, message


def test_order_rule_preserves_plain_markdown_failure() -> None:
    """Plain-markdown documents with inverted order still fail."""
    rule = TextOrderRule(_TOC_RULE)

    passed, _ = rule.run(
        "# Handbook\n\n"
        "Appendix listing the expedited review process ...... 108\n\n"
        "Introduction to the coverage of the plan .......... 103\n"
    )

    assert not passed


def test_order_rule_strips_tags_symmetrically_in_rule_text() -> None:
    """A GT anchor that itself carries markup normalizes the same way."""
    rule = TextOrderRule(
        {
            "type": "order",
            "before": "<td>Introduction to the coverage of the plan</td>",
            "after": "Appendix listing the expedited review process",
            "max_diffs": 2,
        }
    )

    passed, message = rule.run(
        "Introduction to the coverage of the plan\n\nAppendix listing the expedited review process\n"
    )

    assert passed, message
