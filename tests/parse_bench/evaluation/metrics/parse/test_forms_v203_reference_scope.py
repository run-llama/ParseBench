"""Partial GT must not imply exhaustive page annotation merely by using a tree."""

from parse_bench.evaluation.metrics.parse.forms_v2 import forms_reference, score_forms


def ref(scope=None):
    forms = {
        "gt": {
            "pages": [
                {"page_number": 1, "items": [{"type": "field", "field": "text", "label": "Name", "value": "Ada"}]}
            ]
        }
    }
    if scope is not None:
        forms["reference_scope"] = scope
    return forms_reference({"forms": forms})


def test_partial_tree_does_not_charge_unmapped_fields():
    result = score_forms(ref("mapped_fields_only"), "Name: Ada\n\n**Unmapped field**: 123")
    assert not any(e.kind == "extra field" for e in result.events)
    assert result.lanes["extra"] is None
    assert result.score == 1


def test_default_tree_retains_exhaustive_checks():
    result = score_forms(ref(), "Name: Ada\n\n**Invented field**: 123")
    assert any(e.kind == "extra field" for e in result.events)


def test_partial_tree_still_checks_annotated_values():
    result = score_forms(ref("mapped_fields_only"), "Name: Grace\n\n**Unmapped field**: 123")
    assert any(e.cls == "critical" for e in result.events)
    assert result.score < 1
