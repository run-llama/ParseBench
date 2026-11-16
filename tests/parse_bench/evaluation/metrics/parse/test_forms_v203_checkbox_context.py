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


@pytest.mark.parametrize("order", [(2, 0, 1), (1, 2, 0)])
@pytest.mark.parametrize("leading", [False, True])
def test_reordered_questions_with_inline_options_keep_their_own_glyphs(order, leading):
    options = ["No [ ] Yes [x]", "No [x] Yes [ ]", "No [ ] Yes [x]"]
    if leading:
        options = ["[ ] No [x] Yes", "[x] No [ ] Yes", "[ ] No [x] Yes"]
    text = "\n".join(f"{QUESTIONS[i]}? {options[i]}" for i in order)
    result = score_forms(ref(), text)
    assert result.score == 1


def test_inline_wrong_tick_remains_an_error_after_question_reordering():
    text = f"{QUESTIONS[2]}? No [ ] Yes [x]\n{QUESTIONS[0]}? No [x] Yes [ ]\n{QUESTIONS[1]}? No [x] Yes [ ]"
    assert score_forms(ref(), text).score < 1


@pytest.mark.parametrize("instruction", ["If yes, you must register.", "If no, you cannot register."])
def test_instruction_option_words_do_not_displace_a_separate_answer_column(instruction):
    text = "\n".join(f"{question}? {instruction}" for question in QUESTIONS)
    text += "\nYes [x] No [ ]\nYes [ ] No [x]\nYes [x] No [ ]"
    result = score_forms(ref(), text)
    assert not any(
        e.charged and e.kind in ("checked option missing", "unchecked option omitted") for e in result.events
    )


def test_column_question_headings_do_not_bound_each_others_options():
    rules = [
        {"type": "form_field", "label": [question, label], "value_type": "checkbox", "value": state}
        for question, options in [
            ("Hair colour", [("Black", True), ("Other", False)]),
            ("Hair style", [("Curly", True), ("Other", False)]),
            ("Eye colour", [("Blue", True), ("Other", False)]),
        ]
        for label, state in options
    ]
    text = (
        "<table><tr><th>Hair colour</th><th>Eye colour</th></tr>"
        "<tr><td>[x] Black [ ] Other<br>Hair style<br>[x] Curly [ ] Other</td>"
        "<td>[x] Blue [ ] Other</td></tr></table>"
    )
    result = score_forms(reference_from_rules(rules), text)
    assert not any(
        e.charged and e.kind in ("checked option missing", "unchecked option omitted") for e in result.events
    )


@pytest.mark.parametrize("table_cells", [False, True])
def test_ambiguous_or_split_cell_markers_keep_existing_association(monkeypatch, table_cells):
    import parse_bench.evaluation.metrics.parse.forms_v2 as forms

    chunks = []
    for question in QUESTIONS:
        answer = "No ✓ Yes"
        if table_cells:
            answer = "<table><tr><td>No</td><td>[x]</td><td>Yes</td><td>[ ]</td></tr></table>"
        chunks.append(question + "\n" + answer)
    text = "\n\n".join(chunks)
    actual = forms.score_forms(ref(), text)
    monkeypatch.setattr(forms, "_repair_question_options", lambda *args: None)
    existing = forms.score_forms(ref(), text)
    assert actual.score == existing.score
    assert [(e.kind, e.label, e.got) for e in actual.events] == [(e.kind, e.label, e.got) for e in existing.events]


@pytest.mark.parametrize("leading", [True, False])
@pytest.mark.parametrize("wrong", [True, False])
def test_question_then_separate_table_cells_preserve_tick_ownership(leading, wrong):
    chunks = []
    for i, question in enumerate(QUESTIONS):
        yes = (i != 1) != (wrong and i == 1)
        pairs = [("No", "[ ]" if yes else "[x]"), ("Yes", "[x]" if yes else "[ ]")]
        cells = [v for pair in pairs for v in (pair[::-1] if leading else pair)]
        chunks.append(question + "\n<table><tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr></table>")
    result = score_forms(ref(), "\n".join(chunks))
    assert (result.score == 1) == (not wrong)
    if wrong:
        assert any(e.charged and e.kind == "box state flipped" for e in result.events)


def test_full_questions_win_over_identical_instruction_suffix_after_reordering():
    questions = [
        "Were you a director of this company – you must put X in one of these boxes",
        "Was this company a close company – you must put X in one of these boxes",
    ]
    rules = [
        {"type": "form_field", "label": [q, label], "value_type": "checkbox", "value": i == 0 and label == "No"}
        for i, q in enumerate(questions)
        for label in ["Yes", "No"]
    ]
    text = questions[1] + "\nYes [ ] No [ ] If No, go to question 8\n" + questions[0] + "\nYes [ ] No [x]"
    assert score_forms(reference_from_rules(rules), text).score == 1


def test_competing_pairs_do_not_become_unique_by_ignoring_owned_tokens(monkeypatch):
    import parse_bench.evaluation.metrics.parse.forms_v2 as forms

    text = QUESTIONS[0] + "\n" + QUESTIONS[1] + "\nNo [ ] Yes [x]\nNo [x] Yes [ ]\n" + QUESTIONS[2] + "\nNo [ ] Yes [x]"
    out = forms.read_output(text)
    actual, _ = forms.locate(ref(), out)
    monkeypatch.setattr(forms, "_repair_question_options", lambda *args: None)
    existing, _ = forms.locate(ref(), out)
    # The middle question has two competing complete pairs. Neither may be
    # selected merely because another field currently owns the other pair.
    assert actual[4:6] == existing[4:6]


@pytest.mark.parametrize("wrong", [True, False])
def test_question_groups_in_table_columns_use_local_options(wrong):
    rules = [
        {"type": "form_field", "label": [question, label], "value_type": "checkbox", "value": state}
        for question, options in [
            ("Hair colour", [("Black", True), ("Other", False)]),
            ("Eye colour", [("Blue", True), ("Other", False)]),
        ]
        for label, state in options
    ]
    eye = "[ ] Blue [x] Other" if wrong else "[x] Blue [ ] Other"
    text = (
        "<table><tr><th>Eye colour</th><th>Hair colour</th></tr>"
        f"<tr><td>{eye}</td><td>[x] Black [ ] Other</td></tr></table>"
    )
    assert (score_forms(reference_from_rules(rules), text).score == 1) == (not wrong)


def test_split_cell_direction_does_not_skip_an_intervening_mark():
    rules = [
        {"type": "form_field", "label": ["Question", opt], "value_type": "checkbox", "value": opt == "No"}
        for opt in ["Yes", "No"]
    ]
    text = "<table><tr><td>Question</td><td>[ ]</td><td>Yes</td><td>[ ]</td><td>X</td><td>No</td></tr></table>"
    result = score_forms(reference_from_rules(rules), text)
    # X is not one of the parsed box glyphs. Do not jump across it and assign
    # the preceding empty square as an explicit unchecked No.
    assert not any(e.charged and e.label == "No" and e.kind == "box state flipped" for e in result.events)


def test_mixed_inline_and_split_cell_options_keep_existing_glyph_reader():
    rules = [
        {"type": "form_field", "label": label, "value_type": "checkbox", "value": state}
        for label, state in [("office consultation", True), ("doctor visit", False), ("general practitioner", True)]
    ]
    text = (
        "<table><tr><td>office consultation<br>[ ] doctor visit</td><td>[x]</td>"
        "<td>general practitioner</td><td>[x]</td></tr></table>"
    )
    result = score_forms(reference_from_rules(rules), text)
    assert not any(e.charged and e.kind == "box state flipped" for e in result.events)


def test_local_option_with_nested_amount_is_not_replaced_by_later_yes():
    rules = [
        {"type": "form_field", "label": ["Is a deposit required", opt], "value_type": "checkbox", "value": opt == "No"}
        for opt in ["No", "Yes"]
    ]
    rules.append({"type": "form_field", "label": "in the amount of", "value_type": "text", "value": ""})
    text = "Is a deposit required\nNo [x]\nYes, in the amount of [ ]\nOther policy: Yes [x]"
    result = score_forms(reference_from_rules(rules), text)
    assert not any(e.charged and e.label == "Yes" and e.kind == "box state flipped" for e in result.events)


def test_extra_glyph_around_two_options_does_not_trigger_context_repair(monkeypatch):
    import parse_bench.evaluation.metrics.parse.forms_v2 as forms

    rules = [
        {
            "type": "form_field",
            "label": ["Did the business cease?", option],
            "value_type": "checkbox",
            "value": option == "No",
        }
        for option in ["Yes", "No"]
    ]
    reference = forms.reference_from_rules(rules)
    text = (
        '<table><tr><td>Did the business cease? <input type="checkbox"/> Yes '
        '<input type="checkbox"/> No <input type="checkbox" checked/></td></tr></table>'
    )
    actual = forms.score_forms(reference, text)
    monkeypatch.setattr(forms, "_repair_question_options", lambda *args: None)
    assert actual == forms.score_forms(reference, text)
