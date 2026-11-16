"""Explicit caption suffixes remain captions even after an internal colon."""

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_rules, render_reference, score_forms


def ref():
    return reference_from_rules(
        [
            {
                "type": "form_field",
                "label": "TOTAL value of your interest in ALL liabilities: (D1)",
                "value": "12,640.21",
                "value_type": "text",
            }
        ]
    )


def test_caption_suffix_in_canonical_render_is_not_an_answer():
    r = ref()
    assert score_forms(r, render_reference(r)).score == 1


@pytest.mark.parametrize("value", ["9,999", "[___]", "D1"])
def test_suffix_is_not_allowed_to_hide_wrong_or_missing_answer(value):
    r = ref()
    result = score_forms(r, render_reference(r).replace("12,640.21", value))
    assert result.score < 1
    assert any(e.charged and e.cls in ("major", "critical") for e in result.events)
