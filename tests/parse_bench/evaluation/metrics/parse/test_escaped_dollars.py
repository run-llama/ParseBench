"""Escaped dollar signs remain literal in the v2 text and formatting scorers."""

import pytest

from parse_bench.evaluation.metrics.parse.formatting_v2 import parse, score_formatting, styled_chars
from parse_bench.evaluation.metrics.parse.text_next import preprocess, score_text_next, tokenize

REF = (
    "Spending on R&D rose to $4,200 million, while the S&P 500 fell. "
    "Fees were $15 and $20 per share.\n\nThe total was $35."
)


def test_escaped_currency_is_faithful_prose():
    escaped = REF.replace("$", r"\$")
    assert score_text_next(REF, escaped).score == pytest.approx(1.0)
    assert score_text_next(REF, escaped).score > score_text_next(REF, REF.replace("4,200", "4,700")).score
    assert [t.key for t in tokenize(escaped)[0]] == [t.key for t in tokenize(REF)[0]]


def test_escaped_currency_and_real_inline_math():
    ref = r"Fees were $15 and $20; the model uses $x^{2} + y$ and \(z^{3}\)."
    out = r"Fees were \$15 and \$20; the model uses $x^{2} + y$ and \(z^{3}\)."
    assert score_text_next(ref, out).score == pytest.approx(1.0)
    assert preprocess(out).startswith("Fees were $15 and $20;")


@pytest.mark.parametrize(
    "literal",
    [r"\$x^2\$", r"\$^{2}\$", r"\$\$x^2\$\$", r"\$\$\$x^2\$\$\$", r"\$$$x^2\$$$", r"\$x^2$", r"$x^2\$", r"$^{2}\$"],
)
def test_escaped_math_delimiters_remain_literal(literal):
    assert preprocess(literal) == literal.replace(r"\$", "$")


@pytest.mark.parametrize("math", [r"$x^2$", r"$$x^2$$", r"$$$x^2$$$", r"\(x^2\)", r"\[x^2\]"])
def test_real_latex_text_is_unchanged(math):
    assert [t.key for t in tokenize(math)[0]] == ["x", "#2"]


def test_real_fraction_display_math_is_unchanged():
    assert [t.key for t in tokenize(r"$$\frac{a}{b}$$")[0]] == ["a", "b"]


def test_formatting_escaped_currency_preserves_styles():
    assert score_formatting(REF, REF.replace("$", r"\$")).score == pytest.approx(1.0)
    ref = "Fees were **$15** and ~~$20~~; use x<sup>2</sup>."
    out = r"Fees were **\$15** and ~~\$20~~; use $x^{2}$."
    assert score_formatting(ref, out).score == pytest.approx(1.0)
    assert styled_chars(r"\$15") == [("$", 0), ("1", 0), ("5", 0)]


@pytest.mark.parametrize(
    "literal", [r"\$\$**fees**\$\$", r"\$\$\$**fees**\$\$\$", r"\$$**fees**\$$", r"\$$$**fees**\$$$"]
)
def test_formatting_escaped_display_delimiters_preserve_prose(literal):
    assert score_formatting("**fees**", literal).score == pytest.approx(1.0)
    assert parse(literal).toks


@pytest.mark.parametrize("math", [r"$$\frac{a}{b}$$", r"\[\frac{a}{b}\]"])
def test_formatting_real_display_math_is_unchanged(math):
    assert score_formatting("**Fees** paid.", "**Fees** paid.\n\n" + math).score == pytest.approx(1.0)
    assert not parse(math).toks


@pytest.mark.parametrize("math", [r"$x^2$", r"\(x^2\)"])
def test_formatting_real_inline_math_is_unchanged(math):
    assert score_formatting("x<sup>2</sup>", math).score == pytest.approx(1.0)
