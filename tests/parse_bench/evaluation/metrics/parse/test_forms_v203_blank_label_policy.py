"""Blank text fields retain their label; an explicit empty marker is optional."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_rules, reference_from_tree, score_forms


@pytest.fixture(params=["rules", "tree"])
def reference(request):
    fields = [("Name", "Pierre"), ("Middle name", ""), ("City", "Paris")]
    if request.param == "rules":
        return reference_from_rules(
            [{"type": "form_field", "label": label, "value": value, "value_type": "text"} for label, value in fields]
        )
    return reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {"type": "field", "field": "text", "label": label, "value": value} for label, value in fields
                ],
            }
        ]
    )


@pytest.mark.parametrize("marker", ["", "   ", "[empty]", "[___]"])
@pytest.mark.parametrize("layout", ["plain", "bold", "table"])
def test_blank_label_with_or_without_marker_is_accepted(reference, marker, layout):
    if layout == "table":
        md = f"| Field | Value |\n|---|---|\n| Name | Pierre |\n| Middle name | {marker} |\n| City | Paris |"
    elif layout == "bold":
        md = f"**Name:** Pierre\n\n**Middle name:** {marker}\n\n**City:** Paris"
    else:
        md = f"Name: Pierre\nMiddle name: {marker}\nCity: Paris"
    result = score_forms(reference, md)
    assert result.score == 1
    assert not [event for event in result.events if event.charged]


def test_omitted_blank_label_is_still_penalized(reference):
    result = score_forms(reference, "Name: Pierre\nCity: Paris")
    assert result.score < 1
    assert any(
        event.kind == "empty field omitted" and event.label == "Middle name" and event.charged
        for event in result.events
    )


@pytest.mark.parametrize("value", ["Marie", "[empty] Marie", "Paris"])
def test_blank_marker_cannot_hide_an_entered_or_neighbouring_value(reference, value):
    result = score_forms(reference, f"Name: Pierre\nMiddle name: {value}\nCity: Paris")
    assert result.score < 1
    assert any(event.label == "Middle name" and event.cls == "critical" and event.charged for event in result.events)
