"""forms_v2: a caption that shares its table cell with a printed instruction takes its value from the next cell."""

from __future__ import annotations

from typing import Any

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms

PENSION = (
    "4. Pension income - If you expect to receive regular pension payments during the year from a plan, "
    "other than old age security, enter the lesser of 2 000 $ or your estimated annual pension income."
)
BASIC = (
    "1. Basic personal amount - Every resident of Canada can claim this amount. If your net income will be "
    "more than 181 440 $, you may claim a partial amount and enter the calculated amount here."
)


def _ref() -> Any:
    items = [
        {"type": "field", "field": "text", "id": "1", "label": "Basic personal amount", "isEmpty": True},
        {"type": "field", "field": "text", "id": "4", "label": "Pension income", "value": "111,602"},
    ]
    return reference_from_tree([{"page_number": 1, "items": items}])


def _md(basic_value: str, pension_value: str) -> str:
    return (
        f'<table><tr><td colspan="4">{BASIC}</td><td>{basic_value}</td></tr>'
        f'<tr><td colspan="4">{PENSION}</td><td>{pension_value}</td></tr></table>'
    )


def test_value_in_the_cell_after_an_instruction_caption_is_read():
    res = score_forms(_ref(), _md("", "111,602"))
    assert [e.kind for e in res.events if e.charged] == []
    assert res.score == pytest.approx(1.0)


def test_wrong_value_after_an_instruction_caption_is_still_wrong():
    res = score_forms(_ref(), _md("", "111,692"))
    assert [(e.label, e.kind) for e in res.events if e.charged] == [("Pension income", "wrong value")]


def test_value_invented_after_an_instruction_caption_is_still_invented():
    res = score_forms(_ref(), _md("5,000", "111,602"))
    assert [(e.label, e.kind) for e in res.events if e.charged] == [("Basic personal amount", "invented value")]
