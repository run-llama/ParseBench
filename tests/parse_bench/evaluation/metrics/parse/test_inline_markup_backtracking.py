"""Regression tests for ``_INLINE_MARKUP_OPT`` backtracking behavior.

The markup-tolerant joiner is inserted between every pair of words in a
rule's text, so a formatting rule evaluated against content containing a
long ``_``/``*`` run (fill-in blanks, horizontal rules, table borders) used
to backtrack exponentially — the ambiguous ``\\*{1,2}``/``__?`` alternatives
gave ``(a|aa)*``-style tokenizations — hanging until the per-rule timeout
fired and scoring the rule 0. The disambiguated alternatives match the same
language with a single tokenization per position.
"""

from __future__ import annotations

import time

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule


def _timed_run(rule_type: str, text: str, md: str) -> tuple[bool, float]:
    rule = create_test_rule({"type": rule_type, "text": text})
    start = time.monotonic()
    passed, _ = rule.run(md)
    return passed, time.monotonic() - start


class TestLongMarkerRunsCompleteQuickly:
    # Each case previously exceeded the 120s per-rule alarm by many orders
    # of magnitude (the blowup is ~phi^n in the run length). The 10s bound
    # leaves ample headroom for loaded CI runners (measured ~0.25s) while
    # still failing decisively if the exponential behavior returns.

    def test_long_underscore_run(self):
        # The '**hello' prefix anchors the bold pattern so the joiner engages
        # the underscore run before 'world' fails to match.
        md = "**hello " + "_" * 3000 + " tail**"
        passed, elapsed = _timed_run("is_bold", "hello world", md)
        assert not passed
        assert elapsed < 10.0

    def test_long_asterisk_run_after_emphasis(self):
        md = "*hello " + "*" * 3000 + " tail"
        passed, elapsed = _timed_run("is_italic", "hello world", md)
        assert not passed
        assert elapsed < 10.0

    def test_long_underscore_run_in_heading(self):
        # The heading arm of the bold patterns engages the joiner too.
        md = "# hello " + "_" * 3000 + " tail"
        passed, elapsed = _timed_run("is_bold", "hello world", md)
        assert not passed
        assert elapsed < 10.0


class TestMarkupToleranceUnchanged:
    # The rewritten token matches the same language: nested/adjacent markup
    # between words is still tolerated.

    def test_bold_with_nested_strikethrough(self):
        rule = create_test_rule({"type": "is_bold", "text": "hello world"})
        passed, _ = rule.run("**hello ~~world~~**")
        assert passed

    def test_bold_with_adjacent_italic_markers(self):
        rule = create_test_rule({"type": "is_bold", "text": "hello world"})
        passed, _ = rule.run("**hello *world***")
        assert passed

    def test_bold_with_html_tag_between_words(self):
        rule = create_test_rule({"type": "is_bold", "text": "hello world"})
        passed, _ = rule.run("<b>hello <mark>world</mark></b>")
        assert passed
