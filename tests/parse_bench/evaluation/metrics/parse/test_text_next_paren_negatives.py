"""text_next and table_next: an accounting negative written in parentheses, (8) for -8."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.table_next import score_table_next
from parse_bench.evaluation.metrics.parse.text_next import score_text_next


def sign_changes(ref: str, out: str) -> int:
    return score_text_next(ref, out).events.kinds().get("sign changed", 0)


# --- amounts in parentheses are negatives --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "out"),
    [
        ("Operating result: (8)", "Operating result: 8"),
        ("Net income (loss) for the year was (1,234) thousand.", "Net income (loss) for the year was 1,234 thousand."),
        ("Excise taxes included $(58) million in the year.", "Excise taxes included $58 million in the year."),
        ("Cash used in financing 1,500 (400) (300) 1,200", "Cash used in financing 1,500 400 (300) 1,200"),
        ("Margin change 12.1 (3.4%) 8.7", "Margin change 12.1 3.4% 8.7"),
        ("Dividends paid\n\n(399)\n\n(395)", "Dividends paid\n\n399\n\n(395)"),
    ],
)
def test_dropped_parentheses_on_an_amount_is_a_sign_change(ref: str, out: str) -> None:
    res = score_text_next(ref, out)

    assert res.events.kinds().get("sign changed") == 1
    assert res.score <= 0.9


def test_parentheses_and_minus_are_the_same_negative() -> None:
    assert score_text_next("Operating result: (8)", "Operating result: -8").score == 1.0
    assert score_text_next("Net loss was (1,234) thousand.", "Net loss was -1,234 thousand.").score == 1.0
    assert score_text_next("Operating result: -8", "Operating result: (8)").score == 1.0


def test_dropped_minus_is_still_a_sign_change() -> None:
    assert sign_changes("Operating result: -8", "Operating result: 8") == 1


def test_wrong_magnitude_in_parentheses_is_still_a_wrong_number() -> None:
    res = score_text_next(
        "Net income (loss) for the year was (1,234) thousand.", "Net income (loss) for the year was (1,243) thousand."
    )

    assert res.score < 0.9
    assert res.events.kinds().get("sign changed", 0) == 0


# --- markers in parentheses carry no sign --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "out"),
    [
        ("See note (8) for details.", "See note 8 for details."),
        ("Under section 4 (2) the duty applies.", "Under section 4 2 the duty applies."),
        ("(1) The first condition applies to every lot.", "1 The first condition applies to every lot."),
        (
            "(a) The first condition applies.\n\n(iv) The fourth one too.",
            "a The first condition applies.\n\niv The fourth one too.",
        ),
        ("The survey by Widiawati (2024) found the same.", "The survey by Widiawati 2024 found the same."),
        ("Volume 1: 108 (1794) describes the species.", "Volume 1: 108 1794 describes the species."),
        ("Call us: (555) 123-4567 for help.", "Call us: 555 123-4567 for help."),
        ("Article 6(1) applies to the processing.", "Article 6 1 applies to the processing."),
        ("Balance at year end 12 (0) 4", "Balance at year end 12 0 4"),
    ],
)
def test_marker_in_parentheses_is_not_a_sign_change(ref: str, out: str) -> None:
    assert sign_changes(ref, out) == 0


def test_phone_number_scores_one_against_itself() -> None:
    ref = "Call us: (555) 123-4567 for help."

    assert score_text_next(ref, ref).score == 1.0


# --- table cells ---------------------------------------------------------------------------------------------


def table(cell: str) -> str:
    head = "<tr><th>Item</th><th>2024</th><th>2023</th></tr>"
    rows = [("Revenue", "1,500", "1,400"), ("Operating result", cell, "12"), ("Net result", "900", "850")]
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    return f"<table>{head}{body}</table>"


@pytest.mark.parametrize("ref_cell", ["(1,234)", "(9)%"])
def test_table_cell_negative_in_parentheses(ref_cell: str) -> None:
    ref = table(ref_cell)
    minus = "-" + ref_cell.replace("(", "").replace(")", "")

    assert score_table_next(ref, ref).score == 1.0
    assert score_table_next(ref, table(minus)).score > score_table_next(ref, table(minus.lstrip("-"))).score
    assert score_table_next(ref, table(minus.lstrip("-"))).score < 0.9


def test_table_written_cell_per_line_or_row_per_line_scores_one() -> None:
    cells = "<table>\n<tr>\n<td>Taxes withheld on releases</td>\n<td>(35)</td>\n<td>(32)</td>\n</tr>\n</table>"
    rows = "<table><tr><td>Taxes withheld on releases</td><td>(35)</td><td>(32)</td></tr></table>"
    text = "Cash flows for the period, in millions of dollars.\n\n"

    assert score_text_next(text + cells, text + rows).score == 1.0
    assert score_text_next(text + rows, text + cells).score == 1.0
