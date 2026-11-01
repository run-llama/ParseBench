"""normalize_attribution_text drops LaTeX math and commands so they do not count as prediction tokens."""

from parse_bench.evaluation.metrics.attribution.text_utils import normalize_attribution_text


def test_display_inline_and_environment_math_are_removed() -> None:
    text = r"display \[ x^2 \] inline \( y \) env \begin{eq} z \end{eq} end"
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
