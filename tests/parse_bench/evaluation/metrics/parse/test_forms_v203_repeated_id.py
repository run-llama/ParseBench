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
