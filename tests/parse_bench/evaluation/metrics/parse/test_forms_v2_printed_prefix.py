"""A printed IBAN prefix is not an entered account number."""

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_rules, score_forms
from parse_bench.test_cases.parse_rule_schemas import ParseFormFieldRule


@pytest.mark.parametrize("written", ["", "DE", "D E", "DE123", "123", "FR", "DE DE"])
def test_blank_prefix_accepts_print_but_rejects_invented_account(written: str) -> None:
    rule = ParseFormFieldRule.model_validate(
        {"type": "form_field", "label": "30 IBAN", "value": "", "printed_value_prefix": "DE"}
    )
    ref = reference_from_rules([rule])
    assert ref is not None
    result = score_forms(ref, f"30 IBAN: {written}")
    assert (result.score == 1) == (written in {"", "DE", "D E"})


def test_unannotated_blank_still_rejects_printed_looking_value() -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "IBAN", "value": ""}])
    assert ref is not None
    assert score_forms(ref, "IBAN: DE").score < 1


@pytest.mark.parametrize("expected,written,passes", [("", "U S", True), ("", "US42", False), ("42", "US", False)])
def test_prefix_is_generic_and_does_not_replace_a_filled_answer(expected: str, written: str, passes: bool) -> None:
    ref = reference_from_rules(
        [{"type": "form_field", "label": "Account", "value": expected, "printed_value_prefix": "US"}]
    )
    assert ref is not None
    assert (score_forms(ref, f"Account: {written}").score == 1) == passes


@pytest.mark.parametrize("output", ["30 IBAN: DE\n123", "|30 IBAN|DE<br>123|\n|---|---|"])
def test_printed_prefix_cannot_hide_invented_digits_on_next_line(output: str) -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "30 IBAN", "value": "", "printed_value_prefix": "DE"}])
    assert ref is not None
    assert score_forms(ref, output).score < 1


def test_prefix_continuation_stops_at_another_fields_caption() -> None:
    ref = reference_from_rules(
        [
            {"type": "form_field", "label": "30 IBAN", "value": "", "printed_value_prefix": "DE"},
            {"type": "form_field", "label": "31 Amount", "value": "123"},
        ]
    )
    assert ref is not None
    assert score_forms(ref, "30 IBAN: DE\n31 Amount: 123").score == 1
