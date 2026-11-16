"""Long malformed output must preserve ordinary scoring without repeated work."""

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def test_long_whitespace_without_colon_is_not_an_extra_field():
    ref = reference_from_tree(
        [{"page_number": 1, "items": [{"type": "field", "field": "text", "label": "Name", "value": "Pierre"}]}]
    )
    text = "Name: Pierre\n" + "&nbsp;" * 16000
    assert score_forms(ref, text).score == 1


def test_repeated_prose_does_not_change_field_association():
    ref = reference_from_tree(
        [{"page_number": 1, "items": [{"type": "field", "field": "text", "label": "The full name", "value": "Pierre"}]}]
    )
    text = "The full name: Pierre\n" + 'The "' * 15000
    assert score_forms(ref, text).score == 1
