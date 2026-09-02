"""Paragraph-spanning regression tests for the HTML tag-pair detectors (LI-8524).

The LI-8524 span-scoping fix rebuilt every flexible query's word gaps with
``_WORD_WS``, which may wrap a single line break but never a paragraph break.
That scope is right for markdown emphasis - a ``**``, ``*``, ``_`` or ``~~``
span ends at a blank line - and for headings, but it silently narrowed the
HTML tag-pair detectors too: a tag pair ends at its closing tag and
legitimately wraps multiple paragraphs, so ``<u>alpha\n\nbeta</u>`` stopped
matching the query ``alpha beta``.

These tests pin the restored behaviour arm by arm: paragraph-spanning queries
match inside tag pairs, the markdown delimiter arms stay paragraph-confined,
and the ``is_not_*`` rules mirror both directions.
"""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.test_rules import (
    FormattingRule,
    MarkColorRule,
    TitleLevelRule,
)

QUERY = "alpha beta"

# Every HTML tag-pair arm, exercised with a blank line inside the pair.
TAG_PAIR_CASES = [
    ("is_underline", "<u>alpha\n\nbeta</u>"),
    ("is_underline", "<ins>alpha\n\nbeta</ins>"),
    ("is_strikeout", "<s>alpha\n\nbeta</s>"),
    ("is_strikeout", "<del>alpha\n\nbeta</del>"),
    ("is_strikeout", "<strike>alpha\n\nbeta</strike>"),
    ("is_sup", "<sup>alpha\n\nbeta</sup>"),
    ("is_sub", "<sub>alpha\n\nbeta</sub>"),
    ("is_mark", "<mark>alpha\n\nbeta</mark>"),
    ("is_bold", "<b>alpha\n\nbeta</b>"),
    ("is_bold", "<strong>alpha\n\nbeta</strong>"),
    ("is_italic", "<i>alpha\n\nbeta</i>"),
    ("is_italic", "<em>alpha\n\nbeta</em>"),
]


def test_reviewers_repro_underline_split_by_blank_line() -> None:
    """The reported regression, verbatim: <u>alpha\n\nbeta</u> is underlined."""
    rule = FormattingRule({"type": "is_underline", "text": QUERY})

    passed, message = rule.run("<u>alpha\n\nbeta</u>")

    assert passed
    assert message == ""


@pytest.mark.parametrize("tag", ["u", "ins"])
def test_underline_rule_matches_target_inside_larger_span(tag: str) -> None:
    """A focused target does not need to equal the producer's tag boundary."""
    rule = FormattingRule({"type": "is_underline", "text": "intellectual"})

    passed, message = rule.run(f"<{tag}>Timely disclose any intellectual property developed.</{tag}>")

    assert passed
    assert message == ""


def test_not_underline_fails_when_target_is_inside_larger_span() -> None:
    rule = FormattingRule({"type": "is_not_underline", "text": "intellectual"})

    passed, message = rule.run("<u>Timely disclose any intellectual property developed.</u>")

    assert not passed
    assert "unexpectedly" in message


def test_underline_containment_does_not_borrow_a_later_closing_tag() -> None:
    """The query must be contained by one tag pair, not adjacent pairs."""
    rule = FormattingRule({"type": "is_underline", "text": QUERY})

    passed, _ = rule.run("<u>alpha</u> plain <u>beta</u>")

    assert not passed


@pytest.mark.parametrize("tag", ["s", "del", "strike"])
def test_strikeout_rule_matches_target_inside_larger_html_span(tag: str) -> None:
    rule = FormattingRule({"type": "is_strikeout", "text": "obsolete clause"})

    passed, message = rule.run(f'<{tag} class="revision">The obsolete clause is removed.</{tag}>')

    assert passed
    assert message == ""


def test_not_strikeout_fails_when_target_is_inside_larger_span() -> None:
    rule = FormattingRule({"type": "is_not_strikeout", "text": "obsolete clause"})

    passed, message = rule.run("~~The obsolete clause is removed.~~")

    assert not passed
    assert "unexpectedly" in message


def test_strikeout_rule_accepts_css_line_through() -> None:
    rule = FormattingRule({"type": "is_strikeout", "text": "obsolete clause"})

    passed, message = rule.run(
        '<span class="revision" style="color:red; text-decoration: line-through">The obsolete clause is removed.</span>'
    )

    assert passed
    assert message == ""


def test_strikeout_rule_rejects_mismatched_html_tags() -> None:
    rule = FormattingRule({"type": "is_strikeout", "text": "obsolete clause"})

    passed, _ = rule.run("<s>The obsolete clause is removed.</del>")

    assert not passed


def test_strikeout_containment_does_not_borrow_a_later_closing_tag() -> None:
    rule = FormattingRule({"type": "is_strikeout", "text": QUERY})

    passed, _ = rule.run("<s>alpha</s> plain <s>beta</s>")

    assert not passed


@pytest.mark.parametrize(("rule_type", "md"), TAG_PAIR_CASES)
def test_tag_pair_matches_phrase_split_by_blank_line(rule_type: str, md: str) -> None:
    """A tag pair wraps multiple paragraphs; the query may span the break."""
    rule = FormattingRule({"type": rule_type, "text": QUERY})

    passed, _ = rule.run(md)

    assert passed


@pytest.mark.parametrize(
    ("kind", "md"),
    [
        ("sup", '<SUP class="x">alpha beta</sup>'),
        ("sub", '<sub data-source="producer">alpha beta</sub>'),
    ],
)
def test_sup_sub_tag_pairs_accept_opening_tag_attributes(kind: str, md: str) -> None:
    positive_rule = FormattingRule({"type": f"is_{kind}", "text": QUERY})
    negative_rule = FormattingRule({"type": f"is_not_{kind}", "text": QUERY})

    positive_passed, positive_message = positive_rule.run(md)
    negative_passed, negative_message = negative_rule.run(md)

    assert positive_passed, positive_message
    assert not negative_passed
    assert "unexpectedly" in negative_message


@pytest.mark.parametrize(
    ("rule_type", "md"),
    [
        ("is_not_underline", "<u>alpha\n\nbeta</u>"),
        ("is_not_strikeout", "<s>alpha\n\nbeta</s>"),
        ("is_not_bold", "<b>alpha\n\nbeta</b>"),
        ("is_not_bold", "<strong>alpha\n\nbeta</strong>"),
    ],
)
def test_negative_rule_fails_when_tag_pair_spans_the_break(rule_type: str, md: str) -> None:
    """Mirror direction: is_not_* must see the paragraph-spanning span too."""
    rule = FormattingRule({"type": rule_type, "text": QUERY})

    passed, message = rule.run(md)

    assert not passed
    assert "unexpectedly" in message


# ---------------------------------------------------------------------------
# The markdown delimiter arms keep their paragraph scope: GFM emphasis cannot
# span a blank line (the delimiters end up in different paragraphs and render
# literally), so a blank-line gap must NOT match even though the same content
# inside an HTML tag pair does.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_type", "md"),
    [
        ("is_strikeout", "~~alpha\n\nbeta~~"),
        ("is_bold", "**alpha\n\nbeta**"),
        ("is_italic", "*alpha\n\nbeta*"),
        ("is_italic", "_alpha\n\nbeta_"),
    ],
)
def test_markdown_delimiters_stay_paragraph_confined(rule_type: str, md: str) -> None:
    rule = FormattingRule({"type": rule_type, "text": QUERY})

    passed, _ = rule.run(md)

    assert not passed


def test_markdown_strikeout_still_wraps_a_single_line_break() -> None:
    """The scoped gap allows one line break - only the blank line is out."""
    rule = FormattingRule({"type": "is_strikeout", "text": QUERY})

    passed, _ = rule.run("~~alpha\nbeta~~")

    assert passed


# ---------------------------------------------------------------------------
# The colour rules search the tag pair's inner text with the same flexible
# query, so they need the paragraph-spanning gaps too.
# ---------------------------------------------------------------------------


def test_mark_color_matches_phrase_split_by_blank_line() -> None:
    rule = MarkColorRule({"type": "mark_color", "text": QUERY, "color": "yellow"})

    passed, message = rule.run('<mark style="background-color: yellow">alpha\n\nbeta</mark>')

    assert passed
    assert message == ""


def test_html_heading_title_matches_phrase_split_by_blank_line() -> None:
    rule = TitleLevelRule({"type": "is_title", "text": QUERY})

    passed, _ = rule.run("<h2>alpha\n\nbeta</h2>")

    assert passed


@pytest.mark.parametrize("tag", ["b", "strong"])
def test_html_bold_title_matches_phrase_split_by_blank_line(tag: str) -> None:
    rule = TitleLevelRule({"type": "is_title", "text": QUERY})

    passed, _ = rule.run(f"<{tag}>alpha\n\nbeta</{tag}>")

    assert passed


def test_markdown_bold_title_stays_paragraph_confined() -> None:
    rule = TitleLevelRule({"type": "is_title", "text": QUERY})

    passed, _ = rule.run("**alpha\n\nbeta**")

    assert not passed


# ---------------------------------------------------------------------------
# Line endings must not change the paragraph scope: a blank line terminated
# with CRLF (or a mix of CRLF and LF) is the same paragraph break as \n\n.
# The guards used to key on bare \n\n, so Windows line endings let every
# markdown arm leak across the break.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_type", "md"),
    [
        ("is_bold", "**alpha\r\n\r\nbeta**"),
        ("is_italic", "*alpha\r\n\r\nbeta*"),
        ("is_italic", "_alpha\r\n\r\nbeta_"),
        ("is_strikeout", "~~alpha\r\n\r\nbeta~~"),
        # Mixed endings on the break, and a blank line with trailing spaces
        ("is_bold", "**alpha\n\r\nbeta**"),
        ("is_bold", "**alpha\r\n\nbeta**"),
        ("is_bold", "**alpha\r\n \t \r\nbeta**"),
    ],
)
def test_markdown_delimiters_stay_confined_across_crlf_breaks(rule_type: str, md: str) -> None:
    rule = FormattingRule({"type": rule_type, "text": QUERY})

    passed, _ = rule.run(md)

    assert not passed


@pytest.mark.parametrize("md", ["**A**x\n\ny**B**", "**A**x\r\n\r\ny**B**"])
def test_tempered_gap_cannot_cross_a_paragraph_break(md: str) -> None:
    """Both ** runs here are both-flanking, so only the blank-line guard keeps
    the text between them from scoring as bold - with either line ending."""
    rule = FormattingRule({"type": "is_bold", "text": "y"})

    passed, _ = rule.run(md)

    assert not passed


@pytest.mark.parametrize(
    ("rule_type", "md"),
    [
        ("is_bold", "**alpha\r\nbeta**"),
        ("is_strikeout", "~~alpha\r\nbeta~~"),
    ],
)
def test_markdown_span_still_wraps_a_single_crlf_line_break(rule_type: str, md: str) -> None:
    """One CRLF line break inside a span is fine - only the blank line is out."""
    rule = FormattingRule({"type": rule_type, "text": QUERY})

    passed, _ = rule.run(md)

    assert passed


def test_tag_pair_matches_phrase_split_by_crlf_blank_line() -> None:
    rule = FormattingRule({"type": "is_bold", "text": QUERY})

    passed, _ = rule.run("<b>alpha\r\n\r\nbeta</b>")

    assert passed
