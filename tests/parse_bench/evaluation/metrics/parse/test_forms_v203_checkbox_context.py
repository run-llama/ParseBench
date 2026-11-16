"""Question context bounds checkbox labels even when output option order differs."""

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_rules, score_forms

QUESTIONS = [
    "Any professional valuation of stocks and shares",
    "Any professional valuation of household effects",
    "Any professional valuation of houses and land",
]


def ref():
    return reference_from_rules(
        [
            {
                "type": "form_field",
                "label": [question, option],
                "value_type": "checkbox",
                "value": checked == (option == "Yes"),
            }
            for question, checked in zip(QUESTIONS, [True, False, True], strict=True)
            for option in ["Yes", "No"]
        ]
    )


def md(style="inline", middle="No [x] Yes [ ]"):
    options = ["No [ ] Yes [x]", middle, "No [ ] Yes [x]"]

    def fmt(options):
        if style == "bullets":
            return "- " + options.replace(" Yes", "\n- Yes")
        if style == "leading":
            return options.replace("No [x] Yes [ ]", "[x] No [ ] Yes").replace("No [ ] Yes [x]", "[ ] No [x] Yes")
        return options

    return "\n\n".join(f"**{question}**\n{fmt(options)}" for question, options in zip(QUESTIONS, options, strict=True))


@pytest.mark.parametrize("style", ["inline", "bullets", "leading"])
def test_reversed_options_stay_under_their_question(style):
    assert score_forms(ref(), md(style)).score == 1


@pytest.mark.parametrize("middle", ["No [ ] Yes [x]", "No [x]", "No [ ] Yes [ ]"])
def test_wrong_or_missing_option_not_repaired_from_next_question(middle):
    result = score_forms(ref(), md(middle=middle))
    assert result.score < 1
    assert any(e.charged and e.item in (4, 5) for e in result.events)


def test_unscoped_option_is_not_owned_by_previous_question():
    rules = [
        {"type": "form_field", "label": ["First question", "Yes"], "value_type": "checkbox", "value": True},
        {"type": "form_field", "label": ["First question", "No"], "value_type": "checkbox", "value": False},
        {"type": "form_field", "label": "Independent consent", "value_type": "checkbox", "value": True},
    ]
    text = "Independent consent [x]\nFirst question\nNo [ ] Yes [x]"
    result = score_forms(reference_from_rules(rules), text)
    assert not any(e.label == "Independent consent" for e in result.events)
