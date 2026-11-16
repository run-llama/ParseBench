"""A caption followed by instructions and an explicit printed-ID value line."""

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_rules, reference_from_tree, score_forms


def reference(tree=True, value="6,413"):
    if tree:
        return reference_from_tree(
            [
                {
                    "page_number": 1,
                    "items": [
                        {
                            "type": "field",
                            "field": "text",
                            "id": "16",
                            "label": "Tax (see instructions).",
                            "value": value,
                        }
                    ],
                }
            ]
        )
    return reference_from_rules([{"type": "form_field", "label": "16 Tax", "value": value, "value_type": "text"}])


def md(value="6,413", identifier="16"):
    return (
        "**16 Tax (see instructions). Check if any from Form(s):**\n"
        "- 1 8814 [ ]\n- 2 4972 [ ]\n- 3 [ ] [___]\n\n"
        f"**{identifier}:** {value}"
    )


@pytest.mark.parametrize("tree", [True, False])
def test_explicit_repeated_id_reads_value_after_instructions(tree):
    assert score_forms(reference(tree), md()).score == 1


@pytest.mark.parametrize("tree", [True, False])
@pytest.mark.parametrize("value", ["9,999", "[___]"])
def test_wrong_and_empty_id_values_are_not_replaced_by_an_answer_elsewhere(tree, value):
    result = score_forms(reference(tree), md(value) + "\n**18:** 6,413")
    assert result.score < 1
    assert any(e.charged and e.cls in ("critical", "major") for e in result.events)


def test_different_id_does_not_supply_the_answer():
    assert score_forms(reference(), md(identifier="18")).score < 1


def test_unmapped_field_between_heading_and_id_stops_lookup():
    text = md().replace("**16:**", "**Other amount:** 900\n**16:**")
    assert score_forms(reference(), text).score < 1


def test_conflicting_repeated_ids_are_ambiguous():
    assert score_forms(reference(), md() + "\n**16:** 9,999").score < 1


def test_direct_wrong_amount_is_not_overridden():
    assert score_forms(reference(), "**16 Tax (see instructions).** 9,999\n**16:** 6,413").score < 1


@pytest.mark.parametrize("table", [False, True])
@pytest.mark.parametrize("value", ["6,413", "9,999"])
def test_instruction_numbers_do_not_hide_repeated_id_value(table, value):
    heading = "16 Tax (see instructions). Check if any from Form(s): 1 [ ] 8814 2 [ ] 4972 3 [ ]"
    text = (
        f"<table><tr><td>{heading}</td><td>16</td><td>{value}</td></tr></table>" if table else f"{heading}\n16 {value}"
    )
    assert (score_forms(reference(False), text).score == 1) == (value == "6,413")


def test_empty_table_value_does_not_skip_to_printed_instructions():
    text = (
        "<table><tr><td>16 Tax (see instructions). Check if any from Forms: 1 [ ] 8814</td>"
        "<td>16</td><td></td><td>Also enter on line 18</td></tr></table>"
    )
    assert score_forms(reference(False, value=""), text).score == 1


def test_printed_amount_is_not_split_into_repeated_id_and_value():
    ref = reference_from_rules(
        [{"type": "form_field", "label": "12 Standard deduction", "value": "", "value_type": "text"}]
    )
    result = score_forms(ref, "12 Standard deduction\n12\nStandard deduction for single filers:\n$12,400")
    assert not any(e.charged and e.got == "400" for e in result.events)


def test_plain_identifier_extension_does_not_override_existing_dual_code_slots(monkeypatch):
    import parse_bench.evaluation.metrics.parse.forms_v2 as forms

    ref = forms.reference_from_rules(
        [{"type": "form_field", "label": "20700 (35) Pension deduction", "value": "", "value_type": "text"}]
    )
    text = "Pension deduction\n20700 35"
    actual = forms.score_forms(ref, text)
    monkeypatch.setattr(forms, "_repeated_id_reading", lambda *args: None)
    assert actual.score == forms.score_forms(ref, text).score


@pytest.mark.parametrize("shared_id", [True, False])
def test_table_identifier_does_not_select_among_multiple_answer_columns(monkeypatch, shared_id):
    import parse_bench.evaluation.metrics.parse.forms_v2 as forms

    rules = [{"type": "form_field", "label": "11 Real estate taxes", "value": "", "value_type": "text"}]
    if shared_id:
        rules.append({"type": "form_field", "label": "11 Indirect taxes", "value": "4,320", "value_type": "text"})
    ref = forms.reference_from_rules(rules)
    text = "<table><tr><td>11 Real estate taxes (see instructions)</td><td>11</td><td></td><td>4,320</td></tr></table>"
    actual = forms.score_forms(ref, text)
    monkeypatch.setattr(forms, "_repeated_id_reading", lambda *args: None)
    assert actual == forms.score_forms(ref, text)


@pytest.mark.parametrize("value", ["6,413", "9,999"])
def test_nonempty_immediate_value_allows_trailing_empty_padding(value):
    text = (
        "<table><tr><td>16 Tax (see instructions). Check if any from Forms: 1 [ ] 8814</td>"
        f"<td>16</td><td>{value}</td><td></td><td></td></tr></table>"
    )
    assert (score_forms(reference(False), text).score == 1) == (value == "6,413")


def test_inline_subfield_with_same_line_number_does_not_hide_the_amount():
    rules = [
        {"type": "form_field", "label": "16 Tax (see instructions)", "value": "6,413", "value_type": "text"},
        {"type": "form_field", "label": "16 (3) form number", "value": "", "value_type": "text"},
    ]
    text = (
        "<table><tr><td>16 Tax (see instructions). Check if any from Forms: 1 [ ] 8814</td>"
        "<td>16</td><td>6,413</td><td></td></tr></table>"
    )
    result = score_forms(reference_from_rules(rules), text)
    assert not any(e.charged and e.item == 0 for e in result.events)


def test_shared_expense_caption_keeps_column_resolver(monkeypatch):
    import parse_bench.evaluation.metrics.parse.forms_v2 as forms

    rules = [
        {"type": "form_field", "label": "11(a) Real estate taxes — direct", "value": "", "value_type": "text"},
        {"type": "form_field", "label": "11(b) Real estate taxes — indirect", "value": "4,320", "value_type": "text"},
    ]
    text = "<table><tr><td>11 Real estate taxes (see instructions)</td><td>11</td><td>4,320</td></tr></table>"
    ref = forms.reference_from_rules(rules)
    actual = forms.score_forms(ref, text)
    monkeypatch.setattr(forms, "_repeated_id_reading", lambda *args: None)
    assert actual == forms.score_forms(ref, text)
