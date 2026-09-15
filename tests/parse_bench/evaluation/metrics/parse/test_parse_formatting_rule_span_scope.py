"""Span-scoping regression tests for FormattingRule (LI-8524).

The inline-formatting detectors run with ``re.DOTALL`` so a span wrapping a
line break is still detected. DOTALL also used to let the gaps inside those
patterns run past the end of the span they started in, so the query text was
scored as formatted while sitting in plain body prose:

* the heading detector matched *any* heading anywhere earlier in the document
  plus the query anywhere later;
* the paired-delimiter and HTML-tag detectors anchored on one span's closing
  delimiter and borrowed a later span's opening delimiter as their closer.

These tests pin both directions: the false positives stay dead, and legitimate
formatting — including multi-line spans — keeps matching.
"""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.rules_formatting import _strip_other_formatting
from parse_bench.evaluation.metrics.parse.test_rules import FormattingRule, TitleLevelRule

QUERY = "plain paragraph containing the query text"


# ---------------------------------------------------------------------------
# The reported bug: a heading elsewhere in the document must not confer bold
# ---------------------------------------------------------------------------


def test_is_bold_rejects_plain_body_text_below_a_heading() -> None:
    """The original report: one heading + query in plain prose scored as bold.

    Also covers the ``_strip_other_formatting`` fallback path, which used to
    collapse the whole document onto a single line and re-introduce the match
    even once the regexes themselves were scoped to a line.
    """
    rule = FormattingRule({"type": "is_bold", "text": QUERY})

    passed, message = rule.run(f"# Some Title\n\n{QUERY}")

    assert not passed
    assert "no bold formatting found" in message


def test_is_bold_verdict_is_unchanged_by_an_unrelated_heading() -> None:
    """The heading is irrelevant to the query, so it must not flip the verdict."""
    rule = FormattingRule({"type": "is_bold", "text": QUERY})

    without_heading, _ = rule.run(QUERY)
    with_heading, _ = rule.run(f"# Some Title\n\n{QUERY}")

    assert without_heading is False
    assert with_heading == without_heading


def test_is_bold_rejects_text_below_an_empty_heading_marker() -> None:
    """``\\s`` matches newlines regardless of DOTALL, so ``#\\ntext`` also leaked."""
    rule = FormattingRule({"type": "is_bold", "text": QUERY})

    passed, _ = rule.run(f"#\n{QUERY}")

    assert not passed


def test_is_not_bold_passes_for_plain_body_text_below_a_heading() -> None:
    """Mirror direction: is_not_bold used to fail spuriously on correct output."""
    rule = FormattingRule({"type": "is_not_bold", "text": QUERY})

    passed, message = rule.run(f"# Some Title\n\n{QUERY}")

    assert passed
    assert message == ""


# ---------------------------------------------------------------------------
# The query's own word gaps must not straddle a boundary the pattern cannot
# cross: \s matches newlines regardless of DOTALL, so before _WORD_WS a phrase
# starting on a heading line and finishing in the paragraph below scored the
# whole phrase as bold (and as a title).
# ---------------------------------------------------------------------------


def test_is_bold_rejects_query_straddling_from_heading_into_body() -> None:
    rule = FormattingRule({"type": "is_bold", "text": "Total revenue was 4.2M"})

    passed, _ = rule.run("# Total\n\nrevenue was 4.2M")

    assert not passed


def test_is_bold_rejects_query_straddling_a_single_newline_after_heading() -> None:
    """No blank line involved, so only line-confining the query catches this."""
    rule = FormattingRule({"type": "is_bold", "text": "Total revenue"})

    passed, _ = rule.run("# Total\nrevenue was 4.2M")

    assert not passed


def test_is_not_bold_passes_when_query_straddles_from_heading_into_body() -> None:
    rule = FormattingRule({"type": "is_not_bold", "text": "Total revenue was 4.2M"})

    passed, message = rule.run("# Total\n\nrevenue was 4.2M")

    assert passed
    assert message == ""


def test_is_bold_rejects_query_straddling_a_paragraph_break_inside_a_span() -> None:
    """``**alpha\\n\\nbeta**`` is two paragraphs, not one bold span."""
    rule = FormattingRule({"type": "is_bold", "text": "alpha beta"})

    passed, _ = rule.run("text **alpha\n\nbeta** more")

    assert not passed


def test_is_title_rejects_query_straddling_from_heading_into_body() -> None:
    """TitleLevelRule's heading pattern had the identical straddle leak."""
    rule = TitleLevelRule({"type": "is_title", "text": "Total revenue"})

    assert not rule.run("# Total\nrevenue was 4.2M")[0]
    assert not rule.run("# Total\n\nrevenue was 4.2M")[0]


@pytest.mark.parametrize(
    "content",
    ["# Total revenue", "**Total revenue**", "<h2>Total revenue</h2>"],
    ids=["md heading", "standalone bold line", "html heading"],
)
def test_is_title_still_detects_real_titles(content: str) -> None:
    rule = TitleLevelRule({"type": "is_title", "text": "Total revenue"})

    passed, message = rule.run(content)

    assert passed, message


def test_is_bold_still_detects_multiword_query_on_one_heading_line() -> None:
    rule = FormattingRule({"type": "is_bold", "text": "Total revenue was 4.2M"})

    passed, message = rule.run("# Total revenue was 4.2M")

    assert passed, message


def test_is_bold_still_detects_heading_with_markup_between_words() -> None:
    """Line-confining the query must keep the inline-markup tolerance."""
    rule = FormattingRule({"type": "is_bold", "text": "Total revenue"})

    passed, message = rule.run("# **Total** revenue")

    assert passed, message


# ---------------------------------------------------------------------------
# Text sitting between two unrelated formatted spans is not formatted
# ---------------------------------------------------------------------------

SPAN_FORMS = [
    ("is_bold", "**Alpha**", "**Beta**"),
    ("is_bold", "<b>Alpha</b>", "<b>Beta</b>"),
    ("is_bold", "<strong>Alpha</strong>", "<strong>Beta</strong>"),
    ("is_italic", "*Alpha*", "*Beta*"),
    ("is_italic", "_Alpha_", "_Beta_"),
    ("is_italic", "<i>Alpha</i>", "<i>Beta</i>"),
    ("is_italic", "<em>Alpha</em>", "<em>Beta</em>"),
]

# Three separations, because each once defeated a different regex guard. The
# pairing scanner rejects all three the same way - the two runs around the
# plain text belong to different spans - but the shapes stay pinned separately
# so a regression in any one of them is visible on its own.
SEPARATIONS = [
    ("paragraph break", "{a}\n\nplain body text\n\n{b}"),
    ("adjacent lines", "{a}\nplain body text\n{b}"),
    ("same line, table row", "| {a} | plain body text | {b} |"),
]

BETWEEN_SPANS_CASES = [
    (f"{rule_type}: {sep_label}", rule_type, template.format(a=opening, b=closing))
    for rule_type, opening, closing in SPAN_FORMS
    for sep_label, template in SEPARATIONS
]


@pytest.mark.parametrize(
    "label,rule_type,content",
    BETWEEN_SPANS_CASES,
    ids=[case[0] for case in BETWEEN_SPANS_CASES],
)
def test_formatting_rule_rejects_plain_text_between_two_spans(label: str, rule_type: str, content: str) -> None:
    """Text between two spans is not formatted, however the spans are separated.

    The borrowed-delimiter shape: the match starts at one span's CLOSING
    delimiter and uses a later span's OPENING delimiter as its own closer.
    """
    rule = FormattingRule({"type": rule_type, "text": "plain body text"})

    passed, _ = rule.run(content)

    assert not passed, f"{label}: plain text between two spans scored as formatted"


@pytest.mark.parametrize(
    "rule_type,content",
    [
        ("is_bold", "| **Total** | 4.2M |"),
        ("is_italic", "| *Total* | 4.2M |"),
        ("is_italic", "| _Total_ | 4.2M |"),
    ],
)
def test_formatting_rule_still_detects_a_formatted_table_cell(rule_type: str, content: str) -> None:
    """The flanking guards must not reject a genuinely formatted cell."""
    rule = FormattingRule({"type": rule_type, "text": "Total"})

    passed, message = rule.run(content)

    assert passed, message


@pytest.mark.parametrize(
    "rule_type,content",
    [
        ("is_bold", "**A**x y**B**"),
        ("is_italic", "*A*x y*B*"),
    ],
)
def test_pairing_resolves_delimiters_flanked_by_text_on_both_sides(rule_type: str, content: str) -> None:
    """Formerly the pinned KNOWN LIMITATION of the flanking-guard regexes.

    A delimiter run with non-whitespace on BOTH sides is simultaneously left-
    and right-flanking, so no guard local to that run could tell which span it
    belonged to, and ``x y`` here used to score as formatted. Delimiter-run
    pairing resolves the ambiguity positionally - the run after ``A`` closes
    the first span, the run before ``B`` opens the second - so the text
    between the spans is plain and the spans themselves still match.
    """
    rule = FormattingRule({"type": rule_type, "text": "x y"})

    assert not rule.run(content)[0]
    assert FormattingRule({"type": rule_type, "text": "A"}).run(content)[0]
    assert FormattingRule({"type": rule_type, "text": "B"}).run(content)[0]


# ---------------------------------------------------------------------------
# Whitespace padding just INSIDE the delimiters is cosmetic: parser output that
# writes `** Total **` means bold Total, so the padded form must score exactly
# like the tight one. Strict CommonMark would render it as literal asterisks,
# but the evaluator scores parser intent, not CommonMark conformance. Padded
# runs only join the pairing's second pass, over runs the strict pass left
# unpaired, so a delimiter already serving a tight span can never be borrowed
# by a padded match, and a padded pairing that would WRAP a strict span is
# dropped outright - the between-spans false positives stay dead.
# ---------------------------------------------------------------------------

PADDED_EMPHASIS_CASES = [
    ("** padded both sides", "is_bold", "** Total **"),
    ("** padded after text", "is_bold", "**Total: **"),
    ("** padded before text", "is_bold", "** Total**"),
    ("** padded in a table cell", "is_bold", "| ** Total ** | 4.2M |"),
    ("** padded mid-sentence", "is_bold", "Grand ** Total ** shown below."),
    ("* padded both sides", "is_italic", "* Total *"),
    ("_ padded both sides", "is_italic", "_ Total _"),
    ("~~ padded both sides", "is_strikeout", "~~ Total ~~"),
]


@pytest.mark.parametrize(
    "label,rule_type,content",
    PADDED_EMPHASIS_CASES,
    ids=[case[0] for case in PADDED_EMPHASIS_CASES],
)
def test_formatting_rule_treats_padded_delimiters_like_tight_ones(label: str, rule_type: str, content: str) -> None:
    rule = FormattingRule({"type": rule_type, "text": "Total"})

    passed, message = rule.run(content)

    assert passed, f"{label}: {message}"


@pytest.mark.parametrize(
    "rule_type,content,inner,outer_words",
    [
        ("is_bold", "** foo **bold** bar **", "bold", ("foo", "bar")),
        ("is_italic", "* one *mid* two *", "mid", ("one", "two")),
    ],
)
def test_padded_runs_bracketing_a_tight_span_do_not_pair(
    rule_type: str, content: str, inner: str, outer_words: tuple[str, str]
) -> None:
    """Two padded runs AROUND a tight span are its neighbors, not a wrapper.

    The strict pass consumes the inner pair and leaves the outer runs, which
    would then pair in the padded pass into a span swallowing the tight span
    plus the plain text around it. That pairing is dropped: only the tight
    span scores.
    """
    assert FormattingRule({"type": rule_type, "text": inner}).run(content)[0]
    for word in outer_words:
        assert not FormattingRule({"type": rule_type, "text": word}).run(content)[0], word


def test_is_not_bold_fails_on_whitespace_padded_delimiters() -> None:
    """Mirror direction: the padded shape counts as bold, so is_not_bold fails."""
    rule = FormattingRule({"type": "is_not_bold", "text": "Total"})

    passed, message = rule.run("** Total **")

    assert not passed
    assert "unexpectedly" in message


# A padded delimiter run whose outer neighbor is punctuation, not whitespace:
# a padded closer against a table pipe, before a colon, inside parentheses.
# These are the same parser-output shapes as the padded cases above - only the
# character just outside the span differs - so they must score the same way.
PADDED_NEXT_TO_PUNCTUATION_CASES = [
    ("padded closer against a table pipe", "is_bold", "|**Total **|"),
    ("padded opener against a table pipe", "is_bold", "|** Total**|"),
    ("padded closer before a colon", "is_bold", "**Total **: 4.2M"),
    ("padded closer glued to trailing text", "is_bold", "**Total **4.2M"),
    ("padded span inside parentheses", "is_bold", "(** Total **)"),
    ("italic padded opener against a pipe", "is_italic", "|* Total*|"),
    ("italic padded closer against a pipe", "is_italic", "|*Total *|"),
    ("closer alone on the next line", "is_bold", "**Total\n**"),
]


@pytest.mark.parametrize(
    "label,rule_type,content",
    PADDED_NEXT_TO_PUNCTUATION_CASES,
    ids=[case[0] for case in PADDED_NEXT_TO_PUNCTUATION_CASES],
)
def test_padded_emphasis_next_to_punctuation_still_matches(label: str, rule_type: str, content: str) -> None:
    rule = FormattingRule({"type": rule_type, "text": "Total"})

    passed, message = rule.run(content)

    assert passed, f"{label}: {message}"


def test_padded_cells_in_one_table_row_pair_within_their_own_cells() -> None:
    """Two padded cells with plain text between them: each cell's query
    matches, the text between the cells does not."""
    row = "|**Total **| more |**Sum **|"

    assert FormattingRule({"type": "is_bold", "text": "Total"}).run(row)[0]
    assert FormattingRule({"type": "is_bold", "text": "Sum"}).run(row)[0]
    assert not FormattingRule({"type": "is_bold", "text": "more"}).run(row)[0]


def test_list_bullets_never_pair_into_an_italic_span() -> None:
    """A line-leading * is a bullet, not emphasis. Italic padding is
    horizontal-only (unlike ** and ~~, which may wrap one line break), so
    bullets on consecutive lines cannot pair up and italicize the item text."""
    md = "* first item\n* second item\n* third item"

    assert not FormattingRule({"type": "is_italic", "text": "first item"}).run(md)[0]
    assert not FormattingRule({"type": "is_italic", "text": "second item"}).run(md)[0]


BORROWED_WITH_PADDING_CASES = [
    ("is_bold", "**Alpha** plain body text **Beta**"),
    ("is_italic", "*Alpha* plain body text *Beta*"),
    ("is_italic", "_Alpha_ plain body text _Beta_"),
    ("is_strikeout", "~~Alpha~~ plain body text ~~Beta~~"),
    # Fully padded spans: every run here is whitespace-delimited on BOTH
    # sides, so no per-run test can reject the borrowed pairing - only the
    # pairing order keeps the middle text plain.
    ("is_bold", "** Alpha ** plain body text ** Beta **"),
    ("is_bold", "| ** Alpha ** | plain body text | ** Beta ** |"),
    ("is_italic", "* Alpha * plain body text * Beta *"),
    ("is_italic", "_ Alpha _ plain body text _ Beta _"),
    ("is_strikeout", "~~ Alpha ~~ plain body text ~~ Beta ~~"),
]


@pytest.mark.parametrize(
    "rule_type,content",
    BORROWED_WITH_PADDING_CASES,
    ids=[f"{case[0]}: {case[1][:20]}" for case in BORROWED_WITH_PADDING_CASES],
)
def test_padded_branch_does_not_resurrect_borrowed_delimiters(rule_type: str, content: str) -> None:
    """The adversarial shape for padding tolerance: two spans on ONE line
    separated by spaces, so the runs facing the query are whitespace-delimited
    exactly like a padded span's own markers. Pairing resolves it
    positionally: Alpha's closer and Beta's opener each pair within their own
    span, so nothing is left to pair across the middle.
    """
    rule = FormattingRule({"type": rule_type, "text": "plain body text"})

    passed, _ = rule.run(content)

    assert not passed, "plain text between two spans scored as formatted"


@pytest.mark.parametrize(
    "rule_type,content",
    [
        ("is_bold", "<b> Total </b>"),
        ("is_bold", "<strong> Total </strong>"),
        ("is_italic", "<i> Total </i>"),
    ],
)
def test_html_tags_accept_inner_padding(rule_type: str, content: str) -> None:
    """Flanking guards apply to markdown delimiters only, never to tag pairs."""
    rule = FormattingRule({"type": rule_type, "text": "Total"})

    passed, message = rule.run(content)

    assert passed, message


# ---------------------------------------------------------------------------
# True positives: real formatting must keep matching
# ---------------------------------------------------------------------------

TRUE_POSITIVE_CASES = [
    ("query on a level-1 heading line", "is_bold", "Executive Summary", "# Executive Summary"),
    ("closed atx heading", "is_bold", "Executive Summary", "### Executive Summary ###"),
    ("heading surrounded by body text", "is_bold", "Executive Summary", "intro\n\n## Executive Summary\n\nbody"),
    ("indented heading", "is_bold", "Executive Summary", "  ## Executive Summary"),
    ("bolded with **", "is_bold", "Executive Summary", "**Executive Summary**"),
    ("query is a substring of the bold span", "is_bold", "Population", "**Population:** 1,234"),
    ("bolded with <b>", "is_bold", "Executive Summary", "<b>Executive Summary</b>"),
    ("bolded with <strong>", "is_bold", "Executive Summary", "<strong>Executive Summary</strong>"),
    ("italic with *", "is_italic", "Grazing Line", "*Grazing Line, NSW*"),
    ("italic with _", "is_italic", "Grazing Line", "_Grazing Line, NSW_"),
    ("italic with <i>", "is_italic", "Grazing Line", "<i>Grazing Line, NSW</i>"),
    ("italic with <em>", "is_italic", "Grazing Line", "<em>Grazing Line, NSW</em>"),
]


@pytest.mark.parametrize(
    "label,rule_type,text,content",
    TRUE_POSITIVE_CASES,
    ids=[case[0] for case in TRUE_POSITIVE_CASES],
)
def test_formatting_rule_still_detects_real_formatting(label: str, rule_type: str, text: str, content: str) -> None:
    rule = FormattingRule({"type": rule_type, "text": text})

    passed, message = rule.run(content)

    assert passed, f"{label}: {message}"
    assert message == ""


# ---------------------------------------------------------------------------
# Multi-line spans: the reason DOTALL is there in the first place
# ---------------------------------------------------------------------------

MULTILINE_SPAN_CASES = [
    ("** span wrapping a line break", "is_bold", "alpha beta", "text **alpha\nbeta** more"),
    ("<b> span wrapping a line break", "is_bold", "alpha beta", "<b>alpha\nbeta</b>"),
    ("<strong> span wrapping a line break", "is_bold", "alpha beta", "<strong>alpha\nbeta</strong>"),
    ("* span wrapping a line break", "is_italic", "alpha beta", "*alpha\nbeta*"),
    ("<em> span wrapping a line break", "is_italic", "alpha beta", "<em>alpha\nbeta</em>"),
]


@pytest.mark.parametrize(
    "label,rule_type,text,content",
    MULTILINE_SPAN_CASES,
    ids=[case[0] for case in MULTILINE_SPAN_CASES],
)
def test_formatting_rule_still_detects_multiline_spans(label: str, rule_type: str, text: str, content: str) -> None:
    """Scoping a match to one span must not break spans that wrap a line."""
    rule = FormattingRule({"type": rule_type, "text": text})

    passed, message = rule.run(content)

    assert passed, f"{label}: {message}"


def test_is_bold_detects_span_with_nested_markup_between_words() -> None:
    """The markup-tolerance path (and its fallback) still resolves nested tags."""
    rule = FormattingRule({"type": "is_bold", "text": "hello world"})

    passed, message = rule.run("**hello <mark>world</mark>**")

    assert passed
    assert message == ""


# ---------------------------------------------------------------------------
# The fallback path must not flatten the document onto one line
# ---------------------------------------------------------------------------


def test_strip_other_formatting_preserves_line_structure() -> None:
    """Newlines carry the heading/body distinction the detectors rely on."""
    stripped = _strip_other_formatting("# Some Title\n\n<em>plain</em> body", "bold")

    assert "\n" in stripped
    assert stripped.splitlines()[0].strip() == "# Some Title"


def test_strip_other_formatting_still_collapses_horizontal_whitespace() -> None:
    stripped = _strip_other_formatting("a    b\tc", "bold")

    assert stripped == "a b c"
