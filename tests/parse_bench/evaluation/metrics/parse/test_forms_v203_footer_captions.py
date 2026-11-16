"""Value-above-caption footer grids must be associated by column, not answer lookup."""

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_rules, score_forms


def ref():
    return reference_from_rules(
        [
            {"type": "form_field", "id": str(i), "label": label, "value": value, "value_type": "text"}
            for i, (label, value) in enumerate(
                [("(Ort, Datum)", "Hamburg, 04.11.2026"), ("(Unterschrift)", "Marah Maier")]
            )
        ]
    )


def output(left, right):
    return f"{left} | {right}\n---|---\n(Ort, Datum) | (Unterschrift)"


def test_footer_values_above_captions_are_read_in_their_columns():
    assert score_forms(ref(), output("Hamburg, 04.11.2026", "Marah Maier")).score == 1


@pytest.mark.parametrize(
    "left,right",
    [
        ("Berlin, 05.11.2026", "Marah Maier"),
        ("Marah Maier", "Hamburg, 04.11.2026"),
        ("", "Marah Maier"),
    ],
)
def test_wrong_swapped_and_missing_values_remain_errors(left, right):
    result = score_forms(ref(), output(left, right))
    assert result.score < 1
    assert any(e.cls in ("major", "critical") and e.charged for e in result.events)


@pytest.mark.parametrize("value,correct", [("Hamburg, 04.11.2026", True), ("Berlin, 04.11.2026", False), ("", False)])
def test_partial_reference_can_use_explicit_neighbour_caption(value, correct):
    partial = reference_from_rules(
        [{"type": "form_field", "label": "(Ort, Datum)", "value": "Hamburg, 04.11.2026", "value_type": "text"}]
    )
    result = score_forms(partial, output(value, "Marah Maier"))
    assert (result.score == 1) == correct


def test_numeric_parentheses_do_not_establish_caption_row():
    partial = reference_from_rules(
        [{"type": "form_field", "label": "(Ort, Datum)", "value": "Hamburg, 04.11.2026", "value_type": "text"}]
    )
    text = output("Hamburg, 04.11.2026", "Marah Maier").replace("(Unterschrift)", "(123)")
    assert score_forms(partial, text).score < 1
