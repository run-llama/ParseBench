"""Value-above-caption footer grids must be associated by column, not answer lookup."""

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_rules, reference_from_tree, score_forms


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


@pytest.mark.parametrize("tag", ["td", "th"])
def test_stacked_blank_label_rows_are_not_values_above_captions(tag):
    reference = reference_from_rules(
        [
            {"type": "form_field", "label": "CASE / CLIENT ID NUMBER", "value": "", "value_type": "text"},
            {"type": "form_field", "label": "DATE", "value": "", "value_type": "text"},
        ]
    )
    text = (
        f"<table><tr><{tag}>DSHS PHONE NUMBER</{tag}><{tag}>DSHS FAX NUMBER 888-338-7410</{tag}></tr>"
        "<tr><td>CASE / CLIENT ID NUMBER</td><td>DATE</td></tr></table>"
    )
    result = score_forms(reference, text)
    assert not any(e.charged and e.kind == "invented value" for e in result.events)


def test_declaration_above_unsigned_signature_is_not_a_signature():
    reference = reference_from_rules(
        [
            {"type": "form_field", "label": "Employee Signature", "value": False, "value_type": "signature"},
            {"type": "form_field", "label": "Date", "value": "", "value_type": "text"},
        ]
    )
    text = (
        '<table><tr><td colspan="2">I acknowledge that driving violations may result in a fine.</td></tr>'
        "<tr><td>Employee Signature</td><td>Date</td></tr></table>"
    )
    result = score_forms(reference, text)
    assert not any(e.charged and e.kind == "signature invented" for e in result.events)


def test_numbered_blank_multirow_table_keeps_legacy_association():
    text = "Name (print) | Phone Number | Relationship\n---|---|---\n1) [___] | | \n2) [___] | | "
    reference = reference_from_tree(
        [
            {
                "items": [
                    {
                        "type": "table",
                        "columns": ["Name (print)", "Phone Number", "Relationship"],
                        "rows": [
                            [{"items": [{"type": "field", "field": "text", "isEmpty": True}]} for _ in range(3)]
                            for _ in range(2)
                        ],
                    }
                ]
            }
        ]
    )
    result = score_forms(reference, text)
    assert result.score == 1
    assert not any(e.charged for e in result.events)


def test_normal_two_row_grid_does_not_promote_static_tin_zeros_to_entered_value():
    reference = reference_from_rules(
        [{"type": "form_field", "label": "25 Spouse TIN (group 1)", "value": "", "value_type": "text"}]
    )
    text = "(Middle Name) | (Suffix) | **25 Spouse TIN**\n--- | --- | ---\n| | - - - 0 0 0 0 0"
    result = score_forms(reference, text)
    assert not any(e.charged and e.kind == "invented value" for e in result.events)
