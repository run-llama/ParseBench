"""normalize_attribution_text drops LaTeX math and commands so they do not count as prediction tokens."""

from parse_bench.evaluation.metrics.attribution.text_utils import normalize_attribution_text


def test_display_inline_and_environment_math_are_removed() -> None:
    text = r"display \[ x^2 \] inline \( y_1 \) env \begin{equation} z \end{equation} end"
    assert normalize_attribution_text(text).split() == ["display", "inline", "env", "end"]


def test_command_name_and_braced_scripts_are_removed_and_the_argument_kept() -> None:
    text = normalize_attribution_text(r"\textbf{Net income} rose sub_{ij} end")
    assert "textbf" not in text and "ij" not in text
    assert "net income" in text and text.endswith("rose sub end")


def test_text_environment_keeps_its_prose_and_drops_its_markers() -> None:
    text = normalize_attribution_text(r"\begin{itemize} \item first point \end{itemize}")
    assert "first point" in text and "itemize" not in text


def test_aligned_inside_display_math_is_removed() -> None:
    assert normalize_attribution_text(r"\begin{aligned} x^2 + y \end{aligned} done").split() == ["done"]


def test_backslash_path_keeps_its_words() -> None:
    assert "users documents" in normalize_attribution_text(r"Save to C:\Users\Documents\Report.docx")


def test_unclosed_display_math_does_not_eat_later_math_or_prose() -> None:
    assert normalize_attribution_text(r"\[ open \[x^2\] close").split()[-2:] == ["open", "close"]


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
