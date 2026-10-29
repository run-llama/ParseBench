"""text_next: a revision mark (underlined new number glued to a struck old one) read without decoration."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.text_next import score_text_next

REF = "\n".join(
    [
        "Grievance and External Review Procedures <u>103</u> ~~101~~",
        "Appeals and Expedited Review <u>105</u> ~~103~~",
        "Eligibility and Enrollment <u>107</u> ~~105~~",
    ]
)


def test_glued_revision_numbers_without_decoration_are_not_wrong_numbers() -> None:
    hyp = "Grievance and External Review Procedures....103101\nAppeals and Expedited Review 105103\n"
    hyp += "Eligibility and Enrollment <td>107105</td>"
    res = score_text_next(REF, hyp)
    assert res.events.critical == 0
    assert res.score == 1.0


def test_glued_numbers_in_the_wrong_order_or_misread_stay_critical() -> None:
    reversed_pair = score_text_next(REF, REF.replace("<u>103</u> ~~101~~", "101103"))
    misread = score_text_next(REF, REF.replace("<u>103</u> ~~101~~", "103104"))
    assert reversed_pair.events.critical >= 1
    assert misread.events.critical >= 1


def test_plain_adjacent_numbers_are_not_split() -> None:
    res = score_text_next("Section totals 103 101 for the year.", "Section totals 103101 for the year.")
    assert res.events.critical >= 1
