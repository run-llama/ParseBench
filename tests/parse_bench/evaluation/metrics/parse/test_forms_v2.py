"""Tests for the form field metric (forms_v2)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import (
    FLOOR,
    forms_reference,
    forms_v2_applies,
    forms_v2_metrics,
    read_output,
    reference_from_rules,
    reference_from_tree,
    render_reference,
    score_forms,
)

TREE: list[dict[str, Any]] = [
    {
        "page_number": 1,
        "items": [
            {
                "type": "section",
                "id": "1",
                "label": "Applicant",
                "items": [
                    {"type": "field", "field": "text", "label": "Full name", "value": "Ingrid Sorensen"},
                    {"type": "field", "field": "text", "label": "Date of birth", "value": "1997-09-18"},
                    {"type": "field", "field": "text", "id": "3", "label": "Annual income", "value": "28,898.28"},
                    {"type": "field", "field": "text", "label": "Middle name", "isEmpty": True},
                    {
                        "type": "field",
                        "field": "single_select",
                        "label": "Residence",
                        "valueItems": [
                            {"type": "field", "field": "checkbox", "label": "Resident", "value": True},
                            {"type": "field", "field": "checkbox", "label": "Non-resident", "value": False},
                        ],
                    },
                ],
            },
            {
                "type": "section",
                "label": "Employer",
                "items": [
                    {"type": "field", "field": "text", "label": "Company", "value": "Juniper Freight Inc"},
                    {"type": "field", "field": "text", "label": "City", "value": "Dayton"},
                    {"type": "field", "field": "text", "label": "Fax number", "isEmpty": True},
                    {"type": "field", "field": "checkbox", "label": "Seasonal work", "value": False},
                    {"type": "field", "field": "checkbox", "label": "Full time", "value": True},
                    {"type": "field", "field": "signature", "label": "Employer signature", "value": False},
                ],
            },
            {
                "type": "table",
                "label": "Dependents",
                "columns": ["Name", "Age"],
                "rows": [
                    [{"items": [{"type": "field", "field": "text", "value": "Ada"}]}, "7"],
                    [None, None],
                    [{"items": [{"type": "field", "field": "text", "value": "Leo"}]}, "4"],
                ],
            },
        ],
    }
]

REF = reference_from_tree(TREE)
PERFECT = render_reference(REF)
FLOOR_MINOR_STEP = FLOOR["minor"] + 1e-9


def score(md: str) -> float:
    return score_forms(REF, md).score


def kinds(md: str) -> list[str]:
    return [e.kind for e in score_forms(REF, md).events if e.charged]


def test_reference_rendered_scores_one():
    res = score_forms(REF, PERFECT)
    assert res.score == 1.0
    assert res.events == []
    assert res.coverage == 1.0
    assert res.atoms == sum(1 for it in REF.items if it.kind != "anchor")


def _lived_with_you(n: int) -> dict[str, Any]:
    """A 1040-style Dependents table: each dependent's cell holds two boxes with printed ids a and b."""
    cell = {
        "items": [
            {"type": "field", "field": "checkbox", "id": "a", "label": "Yes", "value": False},
            {"type": "field", "field": "checkbox", "id": "b", "label": "And in the U.S.", "value": n == 1},
        ]
    }
    return {
        "type": "section",
        "label": "Dependents",
        "items": [
            {
                "type": "table",
                "columns": ["", *[f"Dependent {i}" for i in range(1, n + 1)]],
                "rows": [
                    ["(1) First name", *[None] * n],
                    ["(5) Check if lived with you more than half of 2025", *[cell] * n],
                ],
            }
        ],
    }


def test_reference_rendered_with_printed_ids_in_table_cells_scores_one():
    ref = reference_from_tree([{"page_number": 1, "items": [_lived_with_you(4)]}])
    res = score_forms(ref, render_reference(ref))
    assert [e.kind for e in res.events if e.charged] == []
    assert res.score == 1.0


def test_lanes_are_emitted():
    names = [m.metric_name for m in forms_v2_metrics(REF, PERFECT)]
    assert names == [
        "forms_v2",
        "forms_v2_filled",
        "forms_v2_empty",
        "forms_v2_checkboxes",
        "forms_v2_signatures",
        "forms_v2_ids",
        "forms_v2_extra",
    ]


# ---------------------------------------------------------------- harmless format changes

PLAIN = """## 1 Applicant
Full name: Ingrid Sorensen
Date of birth: 18/09/1997
3 Annual income: 28898.28
Middle name: N/A
Residence: ☒ Resident ☐ Non-resident

## Employer
| Field | Value |
|---|---|
| Company | <u>Juniper Freight Inc</u> |
| City | Dayton |
| Fax number | ________ |
| Seasonal work | ☐ |
| Full time | ☒ |
| Employer signature | |

### Dependents
<table><tr><th>Name</th><th>Age</th></tr><tr><td>Ada</td><td>7</td></tr><tr><td></td><td></td></tr>
<tr><td>Leo</td><td>4</td></tr></table>
"""


def test_format_variants_are_harmless():
    assert score(PLAIN) == pytest.approx(1.0), kinds(PLAIN)


@pytest.mark.parametrize(
    "checked,unchecked",
    [("[x]", "[ ]"), ("[X]", "[]"), ("☑", "☐"), ("✓", ":unselected:"), (":selected:", "□"), ("●", "○")],
)
def test_box_glyph_variants(checked: str, unchecked: str):
    md = PERFECT.replace("[x]", "\u0001").replace("[ ]", unchecked).replace("\u0001", checked)
    assert score(md) == pytest.approx(1.0), kinds(md)


def test_html_checkbox_inputs_and_trailing_boxes():
    md = PERFECT.replace("- [x] Resident", '- Resident <input type="checkbox" checked>').replace(
        "- [ ] Non-resident", '- Non-resident <input type="checkbox">'
    )
    assert score(md) == pytest.approx(1.0), kinds(md)


def test_single_choice_written_as_value():
    md = PERFECT.replace("- **Residence**\n  - [x] Resident\n  - [ ] Non-resident", "- **Residence**: Resident")
    assert md != PERFECT
    assert score(md) == pytest.approx(1.0), kinds(md)


def test_sections_in_another_order():
    head, tail = PERFECT.split("## Employer")
    employer, dependents = tail.split("## Dependents")
    md = "## Employer" + employer + head + "## Dependents" + dependents
    assert score(md) == pytest.approx(1.0), kinds(md)


# ---------------------------------------------------------------- critical errors (>= 10 points)


def drop(md: str) -> float:
    return round((1.0 - score(md)) * 100, 6)


def test_wrong_digit_is_critical():
    md = PERFECT.replace("28,898.28", "28,898.26")
    assert "wrong value" in kinds(md)
    assert drop(md) >= 10


@pytest.mark.parametrize(
    ("expected", "written"),
    [("125000", "12500"), ("12500", "125000"), ("45402", "4540"), ("1HGCM82633A004352", "1HGCM82633A00435")],
)
def test_dropped_or_added_digit_is_critical(expected: str, written: str):
    ref = reference_from_tree([{"page_number": 1, "items": [_text("Units", expected), _text("City", "Dayton")]}])
    res = score_forms(ref, render_reference(ref).replace(f": {expected}", f": {written}"))
    assert [e.kind for e in res.events if e.charged] == ["wrong value"]


def test_value_cut_after_whole_words_is_still_truncated():
    ref = reference_from_tree([{"page_number": 1, "items": [_text("Policy", "PL-4471-2093"), _text("City", "Dayton")]}])
    res = score_forms(ref, render_reference(ref).replace("PL-4471-2093", "PL-4471"))
    assert [e.kind for e in res.events if e.charged] == ["value truncated"]


def test_contradicting_copy_is_charged_whatever_the_order():
    wrong = "- **City**: Akron\n"
    after = score_forms(REF, PERFECT + "\n" + wrong)
    before = score_forms(REF, PERFECT.replace("Dayton", "Akron") + "\n- **City**: Dayton\n")
    assert [e.kind for e in after.events if e.charged] == ["contradicting copy"]
    assert after.score < 1.0
    assert after.score == pytest.approx(before.score)


def test_identical_copy_is_free():
    assert score(PERFECT + "\n- **City**: Dayton\n") == pytest.approx(1.0)


def test_swapped_values_are_critical():
    md = PERFECT.replace("Juniper Freight Inc", "@@").replace("Dayton", "Juniper Freight Inc").replace("@@", "Dayton")
    assert kinds(md).count("value of another field") == 2
    assert drop(md) >= 10


def test_invented_value_in_blank_field_is_critical():
    md = PERFECT.replace("- **Middle name**:", "- **Middle name**: Marie")
    assert "invented value" in kinds(md)
    assert drop(md) >= 10


def test_flipped_box_is_critical():
    md = PERFECT.replace("- [ ] Seasonal work", "- [x] Seasonal work")
    assert "box state flipped" in kinds(md)
    assert drop(md) >= 10


def test_changed_choice_is_one_critical_event():
    md = PERFECT.replace("[x] Resident", "[ ] Resident").replace("[ ] Non-resident", "[x] Non-resident")
    res = score_forms(REF, md)
    charged = [e for e in res.events if e.charged]
    assert [e.kind for e in charged] == ["selected option changed"]
    assert round((1 - res.score) * 100, 6) >= 10


def test_invented_signature_is_critical():
    md = PERFECT.replace("- **Employer signature**:", "- **Employer signature**: /s/ J. Doe")
    assert "signature invented" in kinds(md)


def test_table_value_in_the_wrong_row_is_critical():
    md = PERFECT.replace("| Ada | 7 |", "| Ada | 4 |").replace("| Leo | 4 |", "| Leo | 7 |")
    assert md != PERFECT
    assert drop(md) >= 10


# ---------------------------------------------------------------- major (>= 5) and minor (~1)


def test_dropped_value_is_major():
    md = PERFECT.replace("- **City**: Dayton", "- **City**:")
    assert kinds(md) == ["value lost"]
    assert 5 <= drop(md) < 10


def test_value_without_its_caption_is_major():
    md = PERFECT.replace("- **City**: Dayton", "Dayton")
    assert kinds(md) == ["value without its caption"]


def test_lost_tick_is_major():
    md = PERFECT.replace("- [x] Full time", "- Full time")
    assert kinds(md) == ["checked state lost"]
    assert 5 <= drop(md) < 10


def test_typo_is_minor():
    md = PERFECT.replace("Sorensen", "Sorenson")
    assert kinds(md) == ["value typo"]
    assert 1 <= drop(md) < 5


def test_omitted_blank_field_is_minor():
    md = PERFECT.replace("- **Fax number**: \n", "")
    assert md != PERFECT
    assert kinds(md) == ["empty field omitted"]
    assert 1 <= drop(md) < 5


def test_empty_output_scores_zero():
    assert score("") == 0.0
    assert score("Lorem ipsum dolor sit amet.") == 0.0


# ---------------------------------------------------------------- adversarial


def test_value_found_elsewhere_does_not_count():
    # every value is present on the page, but under the wrong captions: answer-independent matching reads slots
    lines = PERFECT.split("\n")
    values = ["Ingrid Sorensen", "1997-09-18", "28,898.28", "Juniper Freight Inc", "Dayton"]
    md = PERFECT
    for a, b in zip(values, values[1:] + values[:1], strict=True):
        md = md.replace(f": {a}\n", f": \u0002{b}\n")
    md = md.replace("\u0002", "")
    assert len(lines) == len(md.split("\n"))
    assert kinds(md).count("value of another field") == len(values)
    assert score(md) < 0.75


def test_repeated_captions_follow_their_section():
    tree = [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "section",
                    "label": "Applicant",
                    "items": [{"type": "field", "label": "Name", "value": "Alice"}],
                },
                {"type": "section", "label": "Employer", "items": [{"type": "field", "label": "Name", "value": "Bob"}]},
            ],
        }
    ]
    ref = reference_from_tree(tree)
    assert score_forms(ref, "## Employer\nName: Bob\n\n## Applicant\nName: Alice\n").score == 1.0
    swapped = score_forms(ref, "## Applicant\nName: Bob\n\n## Employer\nName: Alice\n")
    assert [e.kind for e in swapped.events if e.charged] == ["value of another field", "value of another field"]


def test_yes_no_rows_keep_their_order():
    tree = [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "field",
                    "field": "single_select",
                    "label": q,
                    "valueItems": [
                        {"type": "field", "field": "checkbox", "label": "Yes", "value": v},
                        {"type": "field", "field": "checkbox", "label": "No", "value": not v},
                    ],
                }
                for q, v in (("Do you smoke?", False), ("Do you drive?", True), ("Do you vote?", True))
            ],
        }
    ]
    ref = reference_from_tree(tree)
    ok = "Do you smoke? Yes [ ] No [x]\nDo you drive? Yes [x] No [ ]\nDo you vote? Yes [x] No [ ]\n"
    assert score_forms(ref, ok).score == 1.0
    bad = ok.replace("Do you drive? Yes [x] No [ ]", "Do you drive? Yes [ ] No [x]")
    assert score_forms(ref, bad).score <= 0.9


def test_printed_hint_after_a_caption_is_not_a_value():
    ref = reference_from_tree([{"page_number": 1, "items": [{"type": "field", "label": "Date", "isEmpty": True}]}])
    assert score_forms(ref, "Date (mm/dd/yyyy): ____\n").score == 1.0
    assert score_forms(ref, "**Date** (see instructions):\n").score == 1.0


def test_markup_injection_and_huge_output_are_safe():
    noise = "<script>alert(1)</script> " + "| a | b |\n" * 2000 + "lorem ipsum " * 5000
    t0 = time.time()
    res = score_forms(REF, PERFECT + "\n" + noise)
    assert time.time() - t0 < 20
    assert res.score == pytest.approx(1.0)


def test_rules_reference_is_spot():
    rules = [
        {"type": "form_field", "label": "Customer name", "value": "Anthony Jackson", "value_type": "text"},
        {"type": "form_field", "label": ["Account type", "Checking"], "value": True, "value_type": "checkbox"},
        {"type": "form_field", "label": "Signature of Employee", "value": True, "value_type": "signature"},
    ]
    ref = reference_from_rules(rules)
    assert ref is not None and ref.source == "rules"
    md = "Customer name: Anthony Jackson\nAccount type: [x] Checking [ ] Savings\nSignature of Employee: (signed)\n"
    assert score_forms(ref, md + "Other printed text: anything\n").score == 1.0
    assert score_forms(ref, md.replace("(signed)", "")).score < 0.96


def test_forms_reference_prefers_the_tree_and_skips_review_excluded_pages():
    meta = {"forms": {"gt": {"pages": TREE}, "review_excluded_pages": [1]}}
    assert forms_reference(meta, []) is None
    meta = {"forms": {"gt": {"pages": TREE}}}
    ref = forms_reference(meta, [{"type": "form_field", "label": "X", "value": "1", "value_type": "text"}])
    assert ref is not None and ref.source == "tree"
    assert not forms_v2_applies(forms_reference({}, []))


def test_output_reader_cells():
    out = read_output("| a | b |\n|---|---|\n| 1 | 2 |\ntext [x] box\n<table><tr><td>c</td></tr></table>")
    assert [c.text.strip() for c in out.cells if c.table == 0] == ["a", "b", "1", "2"]
    assert any(t.box == 1 for t in out.toks)
    assert len(out.grids) == 2


def test_wiring_scores_the_form_tree_of_the_test_case():
    from datetime import UTC, datetime

    from next_wiring import NextScorers

    from parse_bench.schemas.parse_output import PageIR, ParseOutput
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from parse_bench.schemas.product import ProductType
    from parse_bench.test_cases.schema import ParseTestCase

    out = ParseOutput(
        example_id="g/d", pipeline_name="p", markdown=PERFECT, pages=[PageIR(page_index=0, markdown=PERFECT)]
    )
    result = InferenceResult(
        request=InferenceRequest(example_id="g/d", source_file_path="d.pdf", product_type=ProductType.PARSE),
        pipeline_name="p",
        product_type=ProductType.PARSE,
        raw_output={},
        output=out,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        latency_in_ms=1,
    )

    def run(meta: dict[str, Any]) -> dict[str, float]:
        case = ParseTestCase.model_construct(
            test_id="g/d", group="g", file_path="d.pdf", expected_markdown=None, metadata=meta, test_rules=[]
        )
        ev = NextScorers()
        return {
            m.metric_name: m.value for m in ev.evaluate(result, case).metrics if m.metric_name.startswith("forms_v2")
        }

    meta = {"forms": {"gt": {"pages": TREE}, "scope": "markdown"}}
    assert run({"forms": {"gt": {"pages": []}}}) == {}
    scored = run(meta)
    assert scored["forms_v2"] == pytest.approx(1.0)
    assert set(scored) == {
        "forms_v2",
        "forms_v2_filled",
        "forms_v2_empty",
        "forms_v2_checkboxes",
        "forms_v2_signatures",
        "forms_v2_ids",
        "forms_v2_extra",
    }


def test_autolinked_email_is_a_value():
    ref = reference_from_tree(
        [{"page_number": 1, "items": [{"type": "field", "label": "Email", "value": "dmitri.brennan@example.org"}]}]
    )
    assert score_forms(ref, "* **Email**: <u><dmitri.brennan@example.org></u>\n").score == 1.0


# ---------------------------------------------------------------- type-aware values, boxes, tables


@pytest.mark.parametrize("written", ["$28,898.28", "28.898,28", "USD 28,898.28", "28,898.280"])
def test_amount_formats_are_harmless(written: str):
    md = PERFECT.replace("28,898.28", written)
    assert score(md) == pytest.approx(1.0), kinds(md)


def test_decimal_point_lost_is_critical():
    md = PERFECT.replace("28,898.28", "2889828")
    assert kinds(md) == ["wrong value"]
    assert drop(md) >= 10


def test_amount_rounded_is_major():
    md = PERFECT.replace("28,898.28", "28,898")
    assert kinds(md) == ["amount rounded"]
    assert 5 <= drop(md) < 10


@pytest.mark.parametrize(
    "written", ["September 18, 1997", "18 September 1997", "18 de septiembre de 1997", "Sept 18, 1997"]
)
def test_dates_with_month_names_are_harmless(written: str):
    md = PERFECT.replace("1997-09-18", written)
    assert score(md) == pytest.approx(1.0), kinds(md)


def test_date_with_another_month_is_critical():
    md = PERFECT.replace("1997-09-18", "18 October 1997")
    assert "wrong value" in kinds(md)


def test_illegible_marker_is_major_not_critical():
    md = PERFECT.replace("- **City**: Dayton", "- **City**: [illegible]")
    assert kinds(md) == ["value marked illegible"]
    assert 5 <= drop(md) < 10


def test_caption_repeated_as_value_is_major():
    md = PERFECT.replace("- **City**: Dayton", "- **City**: City")
    assert kinds(md) == ["value lost"]


def test_chosen_option_unticked_is_major():
    md = PERFECT.replace("[x] Resident", "[ ] Resident")
    assert kinds(md) == ["choice lost (no option ticked)"]
    assert 5 <= drop(md) < 10


def test_every_option_ticked_is_critical():
    md = PERFECT.replace("[ ] Non-resident", "[x] Non-resident")
    assert kinds(md) == ["selected option changed"]
    assert drop(md) >= 10


@pytest.mark.parametrize("checked,unchecked", [("(X)", "( )"), ("(•)", "( )"), ("◉", "○")])
def test_radio_glyphs(checked: str, unchecked: str):
    md = PERFECT.replace("[x]", "\u0001").replace("[ ]", unchecked).replace("\u0001", checked)
    assert score(md) == pytest.approx(1.0), kinds(md)


def test_area_code_brackets_are_not_a_box():
    ref = reference_from_rules(
        [{"type": "form_field", "label": "Home phone", "value": "503 555-0175", "value_type": "text"}]
    )
    assert ref is not None
    assert score_forms(ref, "Home phone ( ): 503) 555-0175\n").score == 1.0
    assert score_forms(ref, "Home phone ( ) <u>503 555-0175</u>\n").score == 1.0


@pytest.mark.parametrize("on,off", [("Selected", "Not selected"), ("Yes", "No"), ("checked", "unchecked")])
def test_box_states_as_words(on: str, off: str):
    md = PERFECT.replace("  - [x] Resident", f"  - Resident: {on}").replace(
        "  - [ ] Non-resident", f"  - Non-resident: {off}"
    )
    md = md.replace("- [ ] Seasonal work", f"- Seasonal work: {off}").replace("- [x] Full time", f"- Full time: {on}")
    assert score(md) == pytest.approx(1.0), kinds(md)


def test_bracketed_state_words_are_boxes():
    before = PERFECT.replace("[x]", "[checked]").replace("[ ]", "[unchecked]")
    assert score(before) == pytest.approx(1.0), kinds(before)
    after = PERFECT.replace("- [x] Full time", "- Full time [Checked]").replace(
        "- [ ] Seasonal work", "- Seasonal work [not selected]"
    )
    assert score(after) == pytest.approx(1.0), kinds(after)


def test_next_captions_printed_id_is_not_part_of_the_value():
    md = PERFECT.replace(
        "- **Date of birth**: 1997-09-18\n- **3 Annual income**:",
        "**Date of birth**: 1997-09-18    **3 Annual income**:",
    )
    assert md != PERFECT
    assert score(md) == pytest.approx(1.0), kinds(md)


def test_flattened_table_finds_one_character_values():
    md = PERFECT.replace(
        "| Name | Age |\n|---|---|\n| Ada | 7 |\n|  |  |\n| Leo | 4 |", "Row 1: Name Ada; Age 7\nRow 3: Name Leo; Age 4"
    )
    assert md != PERFECT
    assert kinds(md) == ["form table not kept as a grid"]


def test_reordered_table_rows_are_minor():
    md = PERFECT.replace("| Ada | 7 |\n|  |  |\n| Leo | 4 |", "| Leo | 4 |\n|  |  |\n| Ada | 7 |")
    assert md != PERFECT
    assert kinds(md) == ["form table rows reordered"]
    assert 0 < drop(md) <= 2


def test_single_hint_word_value_is_a_value():
    ref = reference_from_rules([{"type": "form_field", "label": "Code", "value": "DD", "value_type": "text"}])
    assert ref is not None
    assert score_forms(ref, "Code: DD\n").score == 1.0
    assert [e.kind for e in score_forms(ref, "Code:\nOther: x\n").events if e.charged] == ["value lost"]


# ---- lanes: signatures, printed ids, extra fields ----


def lanes(md: str) -> dict[str, float | None]:
    return score_forms(REF, md).lanes


def test_signatures_have_their_own_lane():
    assert lanes(PERFECT)["signatures"] == 1.0
    md = PERFECT.replace("- **Employer signature**:", "- **Employer signature**: /s/ J. Doe")
    assert lanes(md)["signatures"] == 0.0
    assert lanes(md)["empty"] == 1.0  # the signature line no longer counts in the empty lane


def test_dropped_printed_id_is_a_small_error_and_counted():
    assert lanes(PERFECT)["ids"] == 1.0  # the one field with a printed id (3 Annual income) came back with it
    md = PERFECT.replace("**3 Annual income**", "**Annual income**")
    res = score_forms(REF, md)
    assert "printed id dropped" in kinds(md)
    assert res.lanes["ids"] == 0.0
    assert res.proportional == score_forms(REF, PERFECT).proportional  # no atom loss
    assert 0 < score_forms(REF, PERFECT).score - res.score <= FLOOR_MINOR_STEP


def test_wrong_printed_id_is_named():
    md = PERFECT.replace("**3 Annual income**", "**4 Annual income**")
    assert "printed id wrong" in kinds(md)


def test_printed_id_in_brackets_or_baked_counts_as_present():
    for variant in (
        "- \\[3\\] **Annual income**: 28,898.28",
        "- **3Annual income**: 28,898.28",
        "- **Annual income (3)**: 28,898.28",
    ):
        md = PERFECT.replace("- **3 Annual income**: 28,898.28", variant)
        assert "printed id dropped" not in kinds(md), variant


def test_extra_fields_are_counted_on_a_tree_reference():
    md = PERFECT + "\n- **OMB No.**: 1545-0112\n- **Account number**: 4471-2290\n"
    res = score_forms(REF, md)
    assert kinds(md).count("extra field") == 2
    assert res.lanes["extra"] is not None and res.lanes["extra"] < 1.0
    assert res.proportional == score_forms(REF, PERFECT).proportional
    assert res.metadata()["extra_fields"] == 2


def test_prose_lead_ins_and_caption_fragments_are_not_extra_fields():
    md = PERFECT + "\n**Key findings include:**\nNote: see the instructions on page 2.\n- **Name**: Ingrid Sorensen\n"
    assert "extra field" not in kinds(md)
    assert lanes(md)["extra"] == 1.0


def test_extra_fields_are_not_judged_on_a_spot_reference():
    rules = [
        {"type": "form_field", "page": 1, "label": "Full name", "value": "Ingrid Sorensen", "value_type": "text"},
    ]
    ref = reference_from_rules(rules)
    assert ref is not None
    res = score_forms(ref, "- **Full name**: Ingrid Sorensen\n- **OMB No.**: 1545-0112\n")
    assert res.lanes["extra"] is None
    assert "extra field" not in [e.kind for e in res.events]


# ---------------------------------------------------------------- layouts written as grids


def _page(*items: dict[str, Any]) -> Any:
    return reference_from_tree([{"page_number": 1, "items": list(items)}])


def _text(label: str, value: str | None = None, pid: str = "") -> dict[str, Any]:
    field: dict[str, Any] = {"type": "field", "field": "text", "label": label}
    if pid:
        field["id"] = pid
    if value is None:
        field["isEmpty"] = True
    else:
        field["value"] = value
    return field


def test_blank_box_does_not_read_the_next_cell():
    ref = _page(_text("Total ordinary dividends", "4,927.48", "1a"), _text("Qualified dividends", None, "1b"))
    md = (
        "<table><tr><td>1a Total ordinary dividends<br />$ 4,927.48</td><td>OMB No. 1545-0110</td></tr>"
        "<tr><td>1b Qualified dividends<br />$</td><td>Form 1099-DIV (Rev. January 2024)</td></tr></table>"
    )
    res = score_forms(ref, md)
    assert [e.kind for e in res.events if e.charged] == []
    assert res.score == pytest.approx(1.0)


def test_next_captions_printed_id_in_its_own_cell_is_not_a_value():
    ref = _page(_text("Other net rental income", None, "3"), _text("Credits", "A 1,200", "15"))
    md = (
        "<table><tr><td>3</td><td>Other net rental income</td><td>15</td><td>Credits</td></tr>\n"
        "<tr><td></td><td></td><td></td><td>A 1,200</td></tr></table>"
    )
    assert "invented value" not in [e.kind for e in score_forms(ref, md).events]


def test_one_digit_value_before_the_next_caption_is_kept():
    ref = _page(_text("Day", "2"), _text("Month", "8"))
    md = "<table><tr><td>Day</td><td>2</td><td>Month</td><td>8</td></tr></table>"
    assert score_forms(ref, md).score == pytest.approx(1.0)


def test_rowspan_keeps_later_cells_under_their_column():
    ref = _page(
        {
            "type": "table",
            "label": "Dependents",
            "columns": ["Name", "Age"],
            "rows": [["Ada", "7"], ["Leo", "4"]],
        }
    )
    md = (
        "<table><tr><th>Group</th><th>Name</th><th>Age</th></tr>"
        '<tr><td rowspan="2">Children</td><td>Ada</td><td>7</td></tr><tr><td>Leo</td><td>4</td></tr></table>'
    )
    res = score_forms(ref, md)
    assert [e.kind for e in res.events if e.charged] == []


def test_repeated_caption_blocks_read_the_value_below():
    ref = _page(
        {
            "type": "table",
            "label": "Dependents",
            "columns": ["Dependent's SSN", "Dependent's first name"],
            "rows": [["579092115", "Elena"], ["095334023", "Caleb"]],
        }
    )
    md = "\n\n".join(
        f"<table><tr><th>{n}. Dependent's SSN</th><th>Dependent's first name</th></tr>"
        f"<tr><td>{ssn}</td><td>{name}</td></tr></table>"
        for n, ssn, name in ((8, "579092115", "Elena"), (9, "095334023", "Caleb"))
    )
    res = score_forms(ref, md)
    assert [e.kind for e in res.events if e.charged] == []


def test_dropped_blank_rows_do_not_cap_coverage():
    ref = _page(_text("Named insured", "Johnson LLC"), *[_text(f"Driver {i} name") for i in range(1, 11)])
    res = score_forms(ref, "**Named insured**: Johnson LLC")
    assert {e.kind.split(" (")[0] for e in res.events if e.charged} == {"empty field omitted"}
    assert res.score >= 1 - FLOOR["minor_cap"] - 1e-9


def _dated(label: str, value: str, other: tuple[str, str] | None = None) -> Any:
    fields = [_text(label, value), *([_text(*other)] if other else []), _text("Policy number", "PN-558213")]
    return _page(*fields)


@pytest.mark.parametrize(
    ("label", "other", "written", "wrong"),
    [
        ("Date of loss (MM/DD/YYYY)", None, "04/03/2025", True),
        ("Date of loss (MM/DD/YYYY)", None, "2025-04-03", True),
        ("Date of loss (MM/DD/YYYY)", None, "April 3, 2025", True),
        ("Date of loss (MM/DD/YYYY)", None, "March 4, 2025", False),
        ("Date of loss", ("Date signed", "25/03/2025"), "04/03/2025", True),
        ("Date of loss", ("Date signed", "25/03/2025"), "2025-04-03", False),
        ("Date of loss", None, "04/03/2025", False),  # no printed order, no other date: both orders accepted
    ],
)
def test_day_month_swap_is_wrong_when_the_form_fixes_the_order(
    label: str, other: tuple[str, str] | None, written: str, wrong: bool
):
    ref = _dated(label, "03/04/2025", other)
    md = render_reference(ref).replace(f"**{label}**: 03/04/2025", f"**{label}**: {written}")
    kinds_ = [e.kind for e in score_forms(ref, md).events if e.charged]
    assert kinds_ == (["wrong value"] if wrong else [])


FRAMED = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                _text("Social Security benefits", "964,240"),
                _text("Exemption amount", "629,536"),
                _text("Date of birth", "05/05/1996"),
            ],
        }
    ]
)


def test_line_id_cents_box_and_date_hint_around_a_value_are_harmless():
    """Tax forms print the line id before the amount box and a ``.00`` cents box after it; dates carry a hint."""
    md = (
        "Social Security benefits: 5 964,240 .00\n"
        "Exemption amount: 10a 629,536 0 0\n"
        "Date of birth: 05/05/1996 MM/DD/YYYY\n"
    )
    assert [(e.kind, e.cls) for e in score_forms(FRAMED, md).events if e.charged] == []


def test_wrong_amount_inside_its_printed_frame_is_critical():
    md = "Social Security benefits: 5 964,241 .00\nExemption amount: 10a 629,536 0 0\nDate of birth: 05/05/1996\n"
    assert [(e.kind, e.cls) for e in score_forms(FRAMED, md).events if e.charged] == [("wrong value", "critical")]


def test_unit_or_word_after_a_value_stays_extra_text():
    """Only date-format hints are a printed frame: ``year`` or ``kg`` after a value is text the parser added."""
    ref = reference_from_tree([{"page_number": 1, "items": [_text("Name", "Alice"), _text("Weight", "75")]}])
    res = score_forms(ref, "Name: Alice year\nWeight: 75 kg\n")
    assert [(e.kind, e.cls) for e in res.events if e.charged] == [("extra text in value", "minor")] * 2


def test_spelled_out_format_hint_after_a_date_is_harmless():
    ref = reference_from_tree([{"page_number": 1, "items": [_text("Date of birth", "05/05/1996")]}])
    assert [e.kind for e in score_forms(ref, "Date of birth: 05/05/1996 Month Day Year\n").events if e.charged] == []
    unit = score_forms(ref, "Date of birth: 05/05/1996 kg\n").events
    assert [(e.kind, e.cls) for e in unit if e.charged] == [("extra text in value", "minor")]


@pytest.mark.parametrize(
    ("expected", "written", "kind"),
    [
        ("964,240", "5,964,240", "wrong value"),  # a longer number, the 5 is no line id
        ("200", "1 200", "extra text in value"),  # digit grouping: 1,200
        ("200", "1 200 .00", "extra text in value"),
        ("200", "1 200.00", "extra text in value"),  # space-grouped 1,200.00
        ("200", "1 200,00", "extra text in value"),  # space-grouped 1 200 with comma cents
        ("12345", "12345 00", "extra text in value"),  # no cents box printed: 00 is extra
        ("00-958", "5 00-958", "extra text in value"),  # an id, not an amount
        ("12", "12 mm", "extra text in value"),  # a lone hint word after a value that is no date
        ("Alice", "Alice yyyy", "extra text in value"),
    ],
)
def test_near_miss_value_frames_are_kept(expected: str, written: str, kind: str) -> None:
    ref = reference_from_tree([{"page_number": 1, "items": [_text("Amount", expected)]}])
    assert [e.kind for e in score_forms(ref, f"Amount: {written}\n").events if e.charged] == [kind]


@pytest.mark.parametrize("expected", ["964240", "964,240"])
def test_line_id_before_a_comma_grouped_amount_is_harmless(expected: str) -> None:
    ref = reference_from_tree([{"page_number": 1, "items": [_text("Amount", expected)]}])
    assert [e.kind for e in score_forms(ref, "Amount: 5 964,240 .00\n").events if e.charged] == []


@pytest.mark.parametrize(
    ("expected", "written"),
    [
        ("26428", "12 26428"),  # a ZIP code
        ("5551234", "1 5551234"),  # a phone number
        ("05/05/1996", "12 05/05/1996"),  # a date
        ("4500", "1 4500"),  # a plain number with no money mark
    ],
)
def test_number_set_apart_by_a_space_before_a_non_money_value_stays_extra_text(expected: str, written: str) -> None:
    ref = reference_from_tree([{"page_number": 1, "items": [_text("Amount", expected)]}])
    assert [e.kind for e in score_forms(ref, f"Amount: {written}\n").events if e.charged] == ["extra text in value"]


@pytest.mark.parametrize(
    ("expected", "written"),
    [
        ("4,500", "1 4,500"),  # the reference writes the amount with grouping
        ("4500", "1 4500 .00"),  # the cents box marks it as money
        ("664", "23a 664"),  # a line id with a letter
        ("664", "23: 664"),  # a line id set apart by punctuation
        ("18/09/1978", "2. 18/09/1978"),  # a numbered line before a date
    ],
)
def test_line_id_before_money_or_marked_by_punctuation_is_harmless(expected: str, written: str) -> None:
    ref = reference_from_tree([{"page_number": 1, "items": [_text("Amount", expected)]}])
    assert [e.kind for e in score_forms(ref, f"Amount: {written}\n").events if e.charged] == []


def test_grid_split_into_tables_that_repeat_its_header_is_one_grid():
    rows = [["1", "120", ""], ["2", "", "4"], ["3", "75", "6"], ["4", "", ""]]
    ref = _page({"type": "table", "columns": ["Day", "Salary", "Hours"], "rows": rows})

    def split(cells: list[list[str]]) -> str:
        head = "<table><tr><th>Day</th><th>Salary</th><th>Hours</th></tr>"
        body = [
            "".join(f"<tr>{''.join(f'<td>{v}</td>' for v in r)}</tr>" for r in part) for part in (cells[:2], cells[2:])
        ]
        return "\n\n".join(f"{head}{b}</table>" for b in body)

    assert [e.kind for e in score_forms(ref, split(rows)).events if e.charged] == []
    wrong = [r[:] for r in rows]
    wrong[2][1] = "57"
    assert "wrong value" in {e.kind for e in score_forms(ref, split(wrong)).events if e.charged}


def _table_html(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{''.join(f'<td>{v}</td>' for v in r)}</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def test_split_grid_header_matches_through_markdown_formatting():
    rows = [["1", "120", ""], ["2", "", "4"], ["3", "75", "6"], ["4", "", ""]]
    ref = _page({"type": "table", "columns": ["Day", "Salary", "Hours"], "rows": rows})
    md = (
        _table_html(["Day", "Salary", "Hours"], rows[:2])
        + "\n\n"
        + _table_html(["Day", "**Salary**", "Hours"], rows[2:])
    )
    assert [e.kind for e in score_forms(ref, md).events if e.charged] == []


def test_reference_tables_sharing_a_header_stay_apart_in_the_output():
    head = ["Item", "Quantity", "Price"]
    filled = [["Apple", "5", "10"], ["Pear", "8", "12"]]
    ref = _page({"type": "table", "columns": head, "rows": filled}, {"type": "table", "columns": head, "rows": filled})
    md = _table_html(head, [["Apple", "", ""], ["Pear", "", ""]]) + "\n\n" + _table_html(head, filled)
    assert "value lost" in {e.kind for e in score_forms(ref, md).events if e.charged}


def test_caption_repeated_per_option_reads_the_column_under_its_header():
    lines = [("Auto", True, "1,200.00"), ("Cargo", False, None), ("Crime", False, None), ("Yacht", False, None)]
    ref = _page(
        {
            "type": "field",
            "field": "multi_select",
            "label": "Lines of business",
            "valueItems": [
                {
                    "type": "section",
                    "label": name,
                    "items": [{"type": "field", "field": "checkbox", "value": on}, _text("Premium", value)],
                }
                for name, on, value in lines
            ],
        }
    )

    def grid(auto: str, cargo: str) -> str:
        return (
            "<table><tr><th colspan='2'>Lines of business</th><th>Premium</th>"
            "<th colspan='2'></th><th>Premium</th></tr>"
            f"<tr><td>[x]</td><td>Auto</td><td>$ {auto}</td><td>[ ]</td><td>Crime</td><td>$</td></tr>"
            f"<tr><td>[ ]</td><td>Cargo</td><td>$ {cargo}</td><td>[ ]</td><td>Yacht</td><td>$</td></tr></table>"
        )

    assert [e.kind for e in score_forms(ref, grid("1,200.00", "")).events if e.charged] == []
    assert "value lost" in {e.kind for e in score_forms(ref, grid("", "1,200.00")).events if e.charged}


def _lines_of_business(lines: list[tuple[str, bool, str | None]]) -> Any:
    sections = [
        {
            "type": "section",
            "label": name,
            "items": [{"type": "field", "field": "checkbox", "value": on}, _text("Premium", value)],
        }
        for name, on, value in lines
    ]
    return _page({"type": "field", "field": "multi_select", "label": "Lines of business", "valueItems": sections})


def _premium_grid(rows: list[tuple[str, str, str]]) -> str:
    body = "".join(f"<tr><td>{box}</td><td>{name}</td><td>{premium}</td></tr>" for box, name, premium in rows)
    return f"<table><tr><th colspan='2'>Lines of business</th><th>Premium</th></tr>{body}</table>"


def test_column_caption_premium_under_an_unprinted_option_is_not_credited():
    ref = _lines_of_business([("Auto", True, "100"), ("Cargo", False, "200")])
    md = _premium_grid([("[x]", "Auto", "100"), ("[ ]", "Other", "200")])
    assert {e.kind for e in score_forms(ref, md).events if e.charged} - {"unchecked option omitted"}


def test_column_caption_repeated_in_each_cell_is_not_value_text():
    ref = _lines_of_business([("Auto", True, "100"), ("Cargo", False, "200")])
    md = _premium_grid([("[x]", "Auto", "Premium: 100"), ("[ ]", "Cargo", "Premium: 200")])
    assert [e.kind for e in score_forms(ref, md).events if e.charged] == []


def test_column_caption_grid_found_past_an_unrelated_table_with_that_header():
    ref = _lines_of_business(
        [("Auto", True, "100"), ("Cargo", False, None), ("Crime", False, None), ("Yacht", False, None)]
    )
    other = "<table><tr><th>Fee</th><th>Premium</th></tr><tr><td>Total</td><td></td></tr></table>\n\n"

    def grid(auto: str, cargo: str) -> str:
        return (
            "<table><tr><th colspan='2'>Lines of business</th><th>Premium</th>"
            "<th colspan='2'></th><th>Premium</th></tr>"
            f"<tr><td>[x]</td><td>Auto</td><td>$ {auto}</td><td>[ ]</td><td>Crime</td><td>$</td></tr>"
            f"<tr><td>[ ]</td><td>Cargo</td><td>$ {cargo}</td><td>[ ]</td><td>Yacht</td><td>$</td></tr></table>"
        )

    assert [e.kind for e in score_forms(ref, other + grid("100", "")).events if e.charged] == []
    assert "value lost" in {e.kind for e in score_forms(ref, other + grid("", "100")).events if e.charged}


def test_column_caption_value_that_starts_with_the_caption_keeps_it():
    ref = _lines_of_business([("Auto", True, "Premium 100"), ("Cargo", False, "Premium 200")])
    md = _premium_grid([("[x]", "Auto", "Premium 100"), ("[ ]", "Cargo", "Premium 200")])
    assert [e.kind for e in score_forms(ref, md).events if e.charged] == []
