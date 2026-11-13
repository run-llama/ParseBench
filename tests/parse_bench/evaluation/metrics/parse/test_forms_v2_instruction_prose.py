"""forms_v2: a caption followed on its line by a printed instruction takes its value from after the instruction."""

from __future__ import annotations

from typing import Any

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms

PENSION = (
    "**4. Pension income** – If you expect to receive regular pension payments during the year from a plan, "
    "other than old age security, enter the lesser of 2 000 $ or your estimated annual pension income."
)
BASIC = (
    "**1. Basic personal amount** – Every resident of Canada can claim this amount. If your net income will be "
    "more than 181 440 $, you may claim a partial amount and enter the calculated amount here."
)


def _ref() -> Any:
    items = [
        {"type": "field", "field": "text", "id": "1", "label": "Basic personal amount", "isEmpty": True},
        {"type": "field", "field": "text", "id": "4", "label": "Pension income", "value": "111,602"},
    ]
    return reference_from_tree([{"page_number": 1, "items": items}])


def _charged(md: str) -> list[tuple[str, str]]:
    return [(e.label, e.kind) for e in score_forms(_ref(), md).events if e.charged]


@pytest.mark.parametrize(
    "md",
    [
        f"{BASIC}\n\n{PENSION} 111,602\n",  # the amount at the end of the instruction's line
        f"{BASIC}\n\n{PENSION}\n\n111,602\n",  # the amount on its own line under it
        f"{BASIC}: <empty>\n{PENSION}: 111,602\n",  # key: value lines
    ],
)
def test_value_after_an_instruction_is_read(md: str) -> None:
    res = score_forms(_ref(), md)

    assert [e.kind for e in res.events if e.charged] == []
    assert res.score == pytest.approx(1.0)


def test_wrong_value_after_an_instruction_is_still_wrong() -> None:
    assert _charged(f"{BASIC}\n\n{PENSION} 111,692\n") == [("Pension income", "wrong value")]


def test_value_invented_after_an_instruction_is_still_invented() -> None:
    assert _charged(f"{BASIC} 5,000\n\n{PENSION} 111,602\n") == [("Basic personal amount", "invented value")]


def test_instruction_without_its_value_is_still_charged() -> None:
    assert _charged(f"{BASIC}\n\n{PENSION}\n") == [("Pension income", "value lost")]


def test_short_wrong_value_right_after_the_caption_is_not_skipped_for_the_instruction_below() -> None:
    """``Pension income 111`` with the instruction on the lines under it: the value is "111", not a blank field read
    after the instruction (that reading turned a wrong value into an uncharged "value lost")."""
    md = (
        f"{BASIC}\n\n**4. Pension income** 111\n_If you expect to receive regular pension payments during the year "
        "from a plan, other than old age security, enter the lesser of 2 000 $ or your estimated pension income._\n"
    )
    assert ("Pension income", "wrong value") in _charged(md)


def test_amount_left_out_of_a_sentence_that_starts_with_the_caption_is_not_read_as_blank() -> None:
    """A caption that only opens a sentence (no dash before the instruction) with its amount left out: the sentence
    is still charged, not read as a blank field from its end (that reading scored a zero amount as matched)."""
    label = (
        "If your family qualifies as an elderly or disabled family and has unreimbursed health, medical, or "
        "disability assistance expenses of more than"
    )
    items = [{"type": "field", "field": "text", "id": "", "label": label, "value": "0.00"}]
    ref = reference_from_tree([{"page_number": 1, "items": items}])
    md = (
        f"Recertification date: <empty>\n\n{label} a year (10% of your annual income) you may qualify for an income "
        "deduction. This deduction may reduce your monthly payment.\n\nFamily's adjusted annual income: <empty>\n"
    )
    assert [e.kind for e in score_forms(ref, md).events if e.charged] != []
