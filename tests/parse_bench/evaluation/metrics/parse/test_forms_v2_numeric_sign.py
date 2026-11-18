"""Numeric polarity must survive token boundaries and value-frame normalization."""

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_rules, score_forms


@pytest.mark.parametrize(
    "expected,written",
    [
        ("694", "-694"),
        ("694", "−−694"),
        ("694", "--694"),
        ("-694", "694"),
        ("694", "− 694"),
        ("694.00", "-$694.00"),
        ("-694.00", "+694.00"),
    ],
)
@pytest.mark.parametrize(
    "template", ["Tax due: {}", "| Tax due | {} |\n|---|---|", "<table><tr><td>Tax due</td><td>{}</td></tr></table>"]
)
def test_sign_change_is_wrong_in_plain_markdown_and_html(expected: str, written: str, template: str) -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "Tax due", "value_type": "text", "value": expected}])
    assert ref is not None
    scored = score_forms(ref, template.format(written))
    assert any(e.charged and e.kind == "wrong value" for e in scored.events)


@pytest.mark.parametrize(
    "expected,written",
    [("-694", "−694"), ("-694.00", "(694,00)"), ("694", "+694"), ("0", "-0"), ("694", "694"), ("-694", "- 694")],
)
def test_equivalent_signed_amounts_remain_correct(expected: str, written: str) -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "Tax due", "value_type": "text", "value": expected}])
    assert ref is not None
    assert score_forms(ref, f"Tax due: {written}").score == 1


def test_spaced_caption_dash_is_not_a_negative_sign() -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "Tax due", "value_type": "text", "value": "694"}])
    assert ref is not None
    assert score_forms(ref, "Tax due - 694").score == 1


@pytest.mark.parametrize(
    "expected,written,correct",
    [("-694.00", "5 -694 .00", True), ("694.00", "5 -694 .00", False), ("-694.00", "5 694 .00", False)],
)
def test_printed_frame_cannot_erase_numeric_sign(expected: str, written: str, correct: bool) -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "Tax due", "value_type": "text", "value": expected}])
    assert ref is not None
    result = score_forms(ref, f"Tax due: {written}")
    assert (result.score == 1) == correct
    if not correct:
        assert any(e.charged and e.kind == "wrong value" for e in result.events)


@pytest.mark.parametrize("expected,written", [("1 234,50", "-1 234,50"), ("12%", "-12%")])
def test_grouped_and_percent_amounts_cannot_flip_sign(expected: str, written: str) -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "Amount", "value_type": "text", "value": expected}])
    assert ref is not None
    assert any(e.charged and e.kind == "wrong value" for e in score_forms(ref, f"Amount: {written}").events)


@pytest.mark.parametrize("written", ["-6 9 4", "− 6 9 4", "-694 (estimated)"])
def test_digit_boxes_and_trailing_prose_cannot_hide_negative_sign(written: str) -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "Amount", "value_type": "text", "value": "694"}])
    assert ref is not None
    assert any(e.charged and e.kind == "wrong value" for e in score_forms(ref, f"Amount: {written}").events)


@pytest.mark.parametrize(
    "label,expected,written",
    [("Country code", "61", "(61)"), ("Area code", "08", "(08)"), ("A-", "989291149", "989291149")],
)
def test_identifier_punctuation_is_not_a_numeric_negative(label: str, expected: str, written: str) -> None:
    ref = reference_from_rules([{"type": "form_field", "label": label, "value_type": "text", "value": expected}])
    assert ref is not None
    assert score_forms(ref, f"{label}: {written}").score == 1


def test_identifier_prefix_hyphen_does_not_attach_to_value() -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "A-", "value_type": "text", "value": "989291149"}])
    assert ref is not None
    assert score_forms(ref, "A-989291149").score == 1


def test_markdown_list_marker_is_not_a_minus_but_table_minus_is() -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "Tax due", "value_type": "text", "value": "694"}])
    assert ref is not None
    assert score_forms(ref, "Tax due:\n- $694").score == 1
    assert any(
        e.charged and e.kind == "wrong value" for e in score_forms(ref, "| Tax due | - $694 |\n|---|---|").events
    )


def test_dialing_code_frame_normalization_keeps_parentheses_unsigned() -> None:
    ref = reference_from_rules(
        [
            {"type": "form_field", "label": label, "value_type": "text", "value": value}
            for label, value in [("Country code", "61"), ("Area code", "02")]
        ]
    )
    assert ref is not None
    result = score_forms(ref, "| Country code Area code |\n|---|\n| (61) (02) |")
    assert not any(e.charged and e.label == "Area code" for e in result.events)


@pytest.mark.parametrize("expected", ["694.00", "-694.00"])
@pytest.mark.parametrize("written", ["(694)", "(USD 694)", "(694 EUR)"])
def test_parentheses_are_an_accepted_frame_or_accounting_negative(expected: str, written: str) -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "Amount", "value_type": "text", "value": expected}])
    assert ref is not None
    assert score_forms(ref, f"Amount: {written}").score == 1


@pytest.mark.parametrize("written", ["(695)", "(-694)", "−694"])
def test_unsigned_parenthesis_tolerance_rejects_wrong_amount_or_explicit_minus(written: str) -> None:
    ref = reference_from_rules([{"type": "form_field", "label": "Amount", "value_type": "text", "value": "694.00"}])
    assert ref is not None
    assert any(e.charged and e.kind == "wrong value" for e in score_forms(ref, f"Amount: {written}").events)


@pytest.mark.parametrize("written", ["−4,200.15", "-4,200.15", "4,200.15", "5 −4,200.15"])
def test_annotated_external_subtraction_prefix_preserves_unsigned_answer(written: str) -> None:
    from parse_bench.test_cases.parse_rule_schemas import ParseFormFieldRule

    rule = ParseFormFieldRule.model_validate(
        {"type": "form_field", "label": "125 Total credits", "value": "4,200.15", "printed_value_prefix": "−"}
    )
    ref = reference_from_rules([rule])
    assert ref is not None
    assert score_forms(ref, f"125 Total credits: {written}").score == 1


@pytest.mark.parametrize(
    "expected,written", [("4,200.15", "−4,201.15"), ("4,200.15", "−−4,200.15"), ("-1,645.42", "1,645.42")]
)
def test_printed_operator_permission_does_not_hide_wrong_amount_or_entered_sign(expected: str, written: str) -> None:
    ref = reference_from_rules(
        [
            {
                "type": "form_field",
                "label": "Amount",
                "value_type": "text",
                "value": expected,
                "printed_value_prefix": "−",
            }
        ]
    )
    assert ref is not None
    assert any(e.charged and e.kind == "wrong value" for e in score_forms(ref, f"Amount: {written}").events)
