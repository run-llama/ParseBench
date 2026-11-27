"""forms_v2: with the page's printed text, a reading made only of printed form text is not a claimed value."""

from __future__ import annotations

from typing import Any

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import forms_reference, reference_from_tree, score_forms

PAGE = (
    "Part 4 Net income\n"
    "Net income (loss) after adjustments if negative, enter 0\n"
    "Property income allowance read the notes\n"
    "Reference number\n"
    "Name of the applicant\n"
    "14,127.55\n"
    "Jane Doe\n"
)


def _ref(page_text: str = PAGE) -> Any:
    items = [
        {"type": "field", "field": "text", "label": "Net income (loss) after adjustments", "value": "14,127.55"},
        {"type": "field", "field": "text", "label": "Property income allowance", "isEmpty": True},
        {"type": "field", "field": "text", "label": "Reference number", "isEmpty": True},
        {"type": "field", "field": "text", "label": "Name of the applicant", "value": "Jane Doe"},
    ]
    ref = reference_from_tree([{"page_number": 1, "items": items}])
    ref.page_text = page_text
    return ref


def _charged(md: str, page_text: str = PAGE) -> dict[str, tuple[str, str]]:
    return {e.label: (e.kind, e.cls) for e in score_forms(_ref(page_text), md).events if e.charged}


INSTRUCTION = (
    "Net income (loss) after adjustments: if negative, enter 0\n\n"
    "Property income allowance: read the notes\n\n"
    "Reference number:\n\n"
    "Name of the applicant: Jane Doe\n"
)


def test_printed_instruction_read_for_a_value_is_a_lost_value_not_a_wrong_one() -> None:
    charged = _charged(INSTRUCTION)

    assert charged["Net income (loss) after adjustments"] == ("value lost", "major")


def test_printed_instruction_read_in_a_blank_field_is_printed_text() -> None:
    charged = _charged(INSTRUCTION)

    assert charged["Property income allowance"] == ("printed text in a blank field", "minor")


def test_without_the_page_text_the_old_verdicts_stand() -> None:
    charged = _charged(INSTRUCTION, page_text="")

    assert charged["Net income (loss) after adjustments"][1] == "critical"
    assert charged["Property income allowance"] == ("invented value", "critical")


@pytest.mark.parametrize(
    "md, label",
    [
        # a number the page does not print is a claimed value
        (
            "Net income (loss) after adjustments: 14,172.55\n\nName of the applicant: Jane Doe\n",
            "Net income (loss) after adjustments",
        ),
        # a word not printed on the page is invented
        ("Reference number: XK-2231\n\nName of the applicant: Jane Doe\n", "Reference number"),
        # words of an expected value are printed only because the form was filled: still invented
        ("Reference number: Doe\n\nName of the applicant: Jane Doe\n", "Reference number"),
    ],
)
def test_claims_that_are_not_printed_form_text_stay_critical(md: str, label: str) -> None:
    assert _charged(md)[label][1] == "critical"


def test_correct_output_scores_one_with_page_text() -> None:
    md = (
        "Net income (loss) after adjustments: 14,127.55\n\n"
        "Property income allowance:\n\n"
        "Reference number:\n\n"
        "Name of the applicant: Jane Doe\n"
    )
    assert score_forms(_ref(), md).score == pytest.approx(1.0)


def test_page_text_is_read_from_the_test_case_metadata() -> None:
    rules = [{"type": "form_field", "label": "Reference number", "value": None, "value_type": "text"}]
    ref = forms_reference({"printed_text": PAGE}, rules)

    assert ref is not None and ref.page_text == PAGE
    assert forms_reference({}, rules).page_text == ""  # type: ignore[union-attr]


def test_page_text_of_a_tree_reference_is_read_beside_its_forms_metadata() -> None:
    gt = {
        "pages": [
            {
                "page_number": 1,
                "items": [{"type": "field", "field": "text", "label": "Reference number", "isEmpty": True}],
            }
        ]
    }
    ref = forms_reference({"printed_text": PAGE, "forms": {"gt": gt}})

    assert ref is not None and ref.page_text == PAGE


def test_a_caption_printed_against_its_box_number_is_printed_text() -> None:
    page = PAGE.replace("Property income allowance read the notes", "412Property income allowance 413read the notes")
    md = "Property income allowance: read the notes\n\nName of the applicant: Jane Doe\n"

    assert _charged(md, page)["Property income allowance"] == ("printed text in a blank field", "minor")


def test_a_wrong_copy_beside_a_printed_text_reading_is_charged_once() -> None:
    md = (
        "Net income (loss) after adjustments: if negative, enter 0\n\n"
        "Name of the applicant: Jane Doe\n\n"
        "Net income (loss) after adjustments: 14,172.55\n"
    )
    events = [(e.kind, e.cls) for e in score_forms(_ref(), md).events if e.charged and "Net income" in e.label]

    assert events == [("wrong value", "critical")]
