"""normalize_attribution_text drops LaTeX math and commands so they do not count as prediction tokens."""

from parse_bench.evaluation.metrics.attribution.text_utils import normalize_attribution_text


def test_display_inline_and_environment_math_are_removed() -> None:
    text = r"display \[ x^2 \] inline \( y_1 \) env \begin{eq} z \end{eq} end"
    assert normalize_attribution_text(text).split() == ["display", "inline", "env", "end"]


def test_latex_commands_and_braced_scripts_are_removed() -> None:
    assert normalize_attribution_text(r"cmd \alpha{b} sub_{ij} end").split() == ["cmd", "sub", "end"]


def test_plain_words_and_amounts_survive() -> None:
    assert normalize_attribution_text("Revenue rose to $100 in 2024").split() == [
        "revenue",
        "rose",
        "to",
        "100",
        "in",
        "2024",
    ]


def test_multiline_display_math_is_removed() -> None:
    assert normalize_attribution_text("before \\[\n x^2 + 1\n\\] after").split() == ["before", "after"]


def test_markdown_escaped_brackets_are_printed_text() -> None:
    # A parser that escapes citation brackets keeps the citation number.
    assert "1" in normalize_attribution_text(r"see \[1\] for details")


def test_bracketed_text_without_latex_markers_survives() -> None:
    text = r"\(unclosed note and a later \) close"
    assert normalize_attribution_text(text).split()[-1] == "close"
    assert "note" in normalize_attribution_text(text)
