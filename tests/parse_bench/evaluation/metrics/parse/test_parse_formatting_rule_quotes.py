"""Typographic-quote folding regression tests for FormattingRule.

``FormattingRule`` matches the *raw* markdown so the formatting markers are
still present, which meant it was the one rule family that never saw the
quote folding ``normalize_text`` applies for every content rule (and that
``TitleLevelRule`` gets through its ``_normalize_title_label`` fallback).

The consequence: a rule authored with ASCII quotes could not match a faithful
parse of a page printed with typographic quotes, and vice versa. Ground truth
markdown routinely failed its own styling rules for that reason alone.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.test_rules import FormattingRule

# ---------------------------------------------------------------------------
# ASCII rule text vs typographic content
# ---------------------------------------------------------------------------


def test_ascii_quoted_rule_matches_curly_quoted_bold_span() -> None:
    """Rule stored with ASCII ``"``; page, GT, and output all print ``“ ”``."""
    rule = FormattingRule({"type": "is_bold", "text": '"Word of the day"'})

    passed, message = rule.run("**“Word of the day”**")

    assert passed, message


def test_ascii_apostrophe_rule_matches_curly_apostrophe_bold_span() -> None:
    """The faithful parse reproduces the printed ``’``; the rule uses ``'``."""
    rule = FormattingRule({"type": "is_bold", "text": "Kellogg's"})

    passed, message = rule.run("Cereal maker **Kellogg’s** reported growth.")

    assert passed, message


def test_ascii_quoted_rule_matches_curly_quoted_html_bold_span() -> None:
    """Folding is not markdown-specific: the HTML tag arm folds too."""
    rule = FormattingRule({"type": "is_bold", "text": '"Word of the day"'})

    passed, message = rule.run("<b>“Word of the day”</b>")

    assert passed, message


# ---------------------------------------------------------------------------
# Typographic rule text vs ASCII content (the reverse direction)
# ---------------------------------------------------------------------------


def test_curly_quoted_rule_matches_ascii_quoted_bold_span() -> None:
    rule = FormattingRule({"type": "is_bold", "text": "“Word of the day”"})

    passed, message = rule.run('**"Word of the day"**')

    assert passed, message


def test_curly_apostrophe_rule_matches_ascii_apostrophe_italic_span() -> None:
    rule = FormattingRule({"type": "is_italic", "text": "Kellogg’s"})

    passed, message = rule.run("Cereal maker *Kellogg's* reported growth.")

    assert passed, message


def test_curly_quoted_rule_matches_ascii_quoted_strikeout_span() -> None:
    rule = FormattingRule({"type": "is_strikeout", "text": "“Word”"})

    passed, message = rule.run('~~"Word"~~ was removed.')

    assert passed, message


# ---------------------------------------------------------------------------
# Non-Latin scripts: the defect is about the quote characters, not the text
# ---------------------------------------------------------------------------


def test_ascii_quoted_thai_rule_matches_curly_quoted_bold_span() -> None:
    """The observed production case: a Thai motto bolded inside ``“ ”``."""
    motto = "ยึดมั่นธรรมาภิบาล"
    rule = FormattingRule({"type": "is_bold", "text": f'"{motto}"'})

    passed, message = rule.run(f"**“{motto}”**")

    assert passed, message


# ---------------------------------------------------------------------------
# Folding must not manufacture formatting that is not there
# ---------------------------------------------------------------------------


def test_genuinely_missing_bold_still_fails_with_ascii_rule_text() -> None:
    rule = FormattingRule({"type": "is_bold", "text": '"Word of the day"'})

    passed, message = rule.run("“Word of the day” is plain body text.")

    assert not passed
    assert "no bold formatting found" in message


def test_genuinely_missing_bold_still_fails_with_curly_rule_text() -> None:
    rule = FormattingRule({"type": "is_bold", "text": "Kellogg’s"})

    passed, message = rule.run("Cereal maker Kellogg's reported growth.")

    assert not passed
    assert "no bold formatting found" in message


def test_curly_quoted_text_outside_the_bold_span_still_fails() -> None:
    """Folding changes characters, never span boundaries."""
    rule = FormattingRule({"type": "is_bold", "text": '"Bank"'})

    passed, _ = rule.run("“**Bank**” means the institution.")

    assert not passed


def test_is_not_bold_still_passes_when_the_curly_text_is_plain() -> None:
    rule = FormattingRule({"type": "is_not_bold", "text": '"Word of the day"'})

    passed, message = rule.run("“Word of the day” is plain body text.")

    assert passed, message


def test_is_not_bold_fails_once_the_curly_text_is_bold() -> None:
    """The negative polarity has to see the folded match too."""
    rule = FormattingRule({"type": "is_not_bold", "text": '"Word of the day"'})

    passed, message = rule.run("**“Word of the day”**")

    assert not passed
    assert "unexpectedly had bold formatting" in message


# ---------------------------------------------------------------------------
# Quote-free documents are untouched
# ---------------------------------------------------------------------------


def test_quote_free_bold_match_is_unchanged() -> None:
    rule = FormattingRule({"type": "is_bold", "text": "Population"})

    passed, message = rule.run("**Population:** 12,000")

    assert passed, message


def test_quote_free_bold_miss_is_unchanged() -> None:
    rule = FormattingRule({"type": "is_bold", "text": "Population"})

    passed, _ = rule.run("Population: 12,000")

    assert not passed
