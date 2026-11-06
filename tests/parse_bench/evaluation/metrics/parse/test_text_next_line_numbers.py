"""text_next: margin line numbers (pleading paper) are optional page furniture."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.text_next import score_text_next

LINES = [
    "The statute provides that the damages caused by a failure to pay a sum of money are limited",
    "to the amount owed under the agreement, together with interest on that amount, and no more.",
    "interest accrues daily. The landlord charges a $50 late fee when a tenant has paid a rent of",
    "$1,200 two days late, which amounts to an interest rate of 760 percent per annum. By contrast, a",
    "reasonable annual interest rate in this example would dictate a much smaller daily late fee.",
    "The size of the late fee penalty is further shown by the fact that the same amount is",
    "charged whether the rent is one day late or two weeks late, with no regard to the delay.",
    "In other words, when a tenant makes a timely payment of one month's rent but also",
    "has a small outstanding balance made up of earlier late fees or other charges, the landlord",
    "will apply the payment first to the part of the balance that includes the outstanding fees.",
    "This practice is set out in the standard lease, which defines rent to include late fees.",
]
REF = "\n".join(f"{n} {line}" for n, line in enumerate(LINES, start=7))
PROSE = "\n".join(LINES)


def test_dropped_margin_line_numbers_cost_nothing():
    assert score_text_next(REF, PROSE).score == 1.0


def test_margin_line_numbers_written_as_one_block_cost_nothing():
    column = "\n".join(str(n) for n in range(7, 7 + len(LINES)))
    assert score_text_next(REF, column + "\n\n" + PROSE).score == 1.0


def test_margin_line_numbers_kept_inline_cost_nothing():
    assert score_text_next(REF, " ".join(REF.split("\n"))).score == 1.0


def test_wrong_number_on_a_line_numbered_page_is_still_critical():
    wrong = PROSE.replace("$1,200", "$1,300")
    assert score_text_next(REF, wrong).score <= 0.9


def test_numbered_list_items_stay_scored():
    ref = "\n".join(f"{n}. {line}" for n, line in enumerate(LINES, start=1))
    assert score_text_next(ref, PROSE).score < 1.0


def test_a_column_of_numbers_is_content():
    ref = "Totals\n\n" + "\n".join(str(n) for n in range(180, 200))
    assert score_text_next(ref, "Totals").score < 0.5


def test_unnumbered_reference_does_not_drop_output_numbers():
    assert score_text_next(PROSE, REF).score < 1.0
