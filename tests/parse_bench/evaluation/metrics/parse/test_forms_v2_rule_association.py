"""Representation changes must preserve field ownership and reject wrong answers."""

from typing import Any

import pytest

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_rules, score_forms


def field(label: str | list[str], value: str | bool, kind: str = "text") -> dict[str, Any]:
    return {"type": "form_field", "label": label, "value": value, "value_type": kind}


def test_flat_checkbox_labels_match_structured_labels_and_reject_flipped_ticks() -> None:
    question = "Part 9. 18. Have you EVER applied for any kind of relief or protection from removal?"
    output = (
        "18. Have you EVER applied for any kind of relief or protection from removal, exclusion, or deportation? "
        "[ ] Yes [x] No"
    )
    flat = [field(f"{question} — {option}", state, "checkbox") for option, state in [("Yes", False), ("No", True)]]
    structured = [field([question, option], state, "checkbox") for option, state in [("Yes", False), ("No", True)]]
    for rules in (flat, structured):
        ref = reference_from_rules(rules)
        assert ref is not None
        assert score_forms(ref, output).score == 1
        bad = score_forms(ref, output.replace("[ ] Yes [x] No", "[x] Yes [ ] No"))
        assert sum(e.charged and e.kind == "box state flipped" for e in bad.events) == 2


@pytest.mark.parametrize("reverse", [False, True])
def test_split_printed_id_matches_after_gt_reordering_without_losing_id(reverse: bool) -> None:
    rules = [field("14a HARTA PADA AKHIR TAHUN PAJAK", "1.155.250.000"), field("Signature name", "TRI YOLANDA")]
    if reverse:
        rules.reverse()
    ref = reference_from_rules(rules)
    assert ref is not None
    output = "14 a HARTA PADA AKHIR TAHUN PAJAK: 1.155.250.000\nSignature name: TRI YOLANDA"
    assert score_forms(ref, output).score == 1
    wrong = output.replace("1.155.250.000", "1.155.250.001")
    assert any(e.charged and e.cls in ("major", "critical") for e in score_forms(ref, wrong).events)


def table(plus: str, minus: str, row_id: str = "910") -> str:
    return (
        "<table><tr><th>Rubrique</th><th>Plus-value</th><th>Moins-value</th></tr>"
        f"<tr><td>{row_id} Expiration des reports d’imposition : <i>2074-I &gt; lignes 550 et 570</i></td>"
        f"<td>{plus}</td><td>{minus}</td></tr></table>"
    )


@pytest.mark.parametrize("structured", [False, True])
def test_rule_table_reads_empty_cell_not_printed_instructions(structured: bool) -> None:
    row = "910 Expiration des reports d'imposition (2074-I)"
    label = [row, "Plus-value"] if structured else f"{row} — Plus-value"
    ref = reference_from_rules([field(label, "")])
    assert ref is not None
    assert score_forms(ref, table("", "")).score == 1
    assert any(e.charged and e.kind == "invented value" for e in score_forms(ref, table("999", "")).events)


def test_rule_table_swapped_values_and_wrong_row_ids_remain_errors() -> None:
    row = "910 Expiration des reports d'imposition (2074-I)"
    ref = reference_from_rules([field([row, "Plus-value"], "7048"), field([row, "Moins-value"], "1963")])
    assert ref is not None
    assert score_forms(ref, table("7048", "1963")).score == 1
    swapped = score_forms(ref, table("1963", "7048"))
    assert sum(e.charged and e.cls in ("major", "critical") for e in swapped.events) >= 2
    assert score_forms(ref, table("7048", "1963", "911")).score < 1


def test_ambiguous_duplicate_tables_do_not_get_resolved_using_expected_values() -> None:
    ref = reference_from_rules([field(["910 Expiration des reports d'imposition (2074-I)", "Plus-value"], "7048")])
    assert ref is not None
    # The candidate with the correct value is deliberately second: it cannot identify the intended table.
    assert score_forms(ref, table("1963", "") + table("7048", "")).score < 1


def test_rule_table_separate_id_column_keeps_value_ownership() -> None:
    row = "910 Expiration des reports d'imposition (2074-I)"
    ref = reference_from_rules([field([row, "Plus-value"], "7048"), field([row, "Moins-value"], "1963")])
    assert ref is not None
    output = table("7048", "1963").replace("<th>Rubrique</th>", "<th>ID</th><th>Rubrique</th>")
    output = output.replace("<td>910 Expiration", "<td>910</td><td>Expiration")
    assert score_forms(ref, output).score == 1
    swapped = output.replace("7048", "TEMP").replace("1963", "7048").replace("TEMP", "1963")
    assert sum(e.charged and e.cls in ("major", "critical") for e in score_forms(ref, swapped).events) >= 2


def test_sparse_flat_option_does_not_borrow_an_adjacent_questions_tick() -> None:
    rules = [
        field("Family situation — No", False, "checkbox"),
        field("Do you agree? — Yes", True, "checkbox"),
        field("Do you agree? — No", False, "checkbox"),
    ]
    ref = reference_from_rules(rules)
    assert ref is not None
    result = score_forms(ref, "Do you agree? [x] Yes [ ] No")
    sparse = [e for e in result.events if e.label == "Family situation — No" and e.charged]
    assert sparse and all(e.kind != "box state flipped" for e in sparse)


@pytest.mark.parametrize("reverse", [False, True])
def test_unique_fuzzy_caption_keeps_its_slot_when_gt_order_changes(reverse: bool) -> None:
    rules = [field("Annual employee pension contribution", "1234"), field("Contact telephone number", "5678")]
    if reverse:
        rules.reverse()
    ref = reference_from_rules(rules)
    assert ref is not None
    output = "Annual employee pension contributlon: 1234\nContact telephone number: 5678"
    assert score_forms(ref, output).score == 1
    swapped = output.replace("1234", "TMP").replace("5678", "1234").replace("TMP", "5678")
    assert sum(e.charged and e.cls in ("major", "critical") for e in score_forms(ref, swapped).events) >= 2


@pytest.mark.parametrize("caption", ["18. Relief application", "Relief application"])
def test_section_prefix_can_be_context_for_unique_caption(caption: str) -> None:
    ref = reference_from_rules([field("Part 9. 18. Relief application", "Pending")])
    assert ref is not None
    assert score_forms(ref, f"{caption}: Pending").score == 1
    assert score_forms(ref, f"{caption}: Rejected").score < 1


def test_prefix_relaxation_does_not_erase_a_conflicting_printed_id() -> None:
    ref = reference_from_rules([field("Part 9. 18. Relief application", "Pending")])
    assert ref is not None
    assert score_forms(ref, "19. Relief application: Pending").score < 1


def test_repeated_caption_does_not_gain_unscoped_prefix_alias() -> None:
    ref = reference_from_rules([field("Part 1. 7. Name", "Alice"), field("Part 2. 7. Name", "Bob")])
    assert ref is not None
    assert score_forms(ref, "Name: Alice\nName: Bob").score < 1


def test_fuzzy_caption_cannot_steal_a_similar_fields_value() -> None:
    ref = reference_from_rules(
        [
            field("Annual employer pension contribution", "5678"),
            field("Annual employee pension contribution", "1234"),
        ]
    )
    assert ref is not None
    output = "Annual employee pension contribution: 1234"
    result = score_forms(ref, output)
    assert result.score < 1
    assert any(e.charged and e.label == "Annual employer pension contribution" for e in result.events)


def test_prefixed_question_in_table_does_not_block_neighbouring_checkbox_groups() -> None:
    labels = ["First supporting evidence", "Second supporting evidence", "J c Third supporting evidence"]
    rules = [
        field([question, option], state == (option == "No"), "checkbox")
        for question, state in zip(labels, [True, True, False], strict=True)
        for option in ["Yes", "No"]
    ]
    ref = reference_from_rules(rules)
    assert ref is not None
    output = (
        "<table>"
        + "".join(
            f"<tr><td>{code}</td><td>{question}</td><td>{no} No</td><td>{yes} Yes</td></tr>"
            for code, question, no, yes in [
                ("a", labels[0], "[x]", "[ ]"),
                ("b", labels[1], "[x]", "[ ]"),
                ("c", "Third supporting evidence", "[ ]", "[x]"),
            ]
        )
        + "</table>"
    )
    assert score_forms(ref, output).score == 1
    wrong = output.replace("[ ] No</td><td>[x] Yes", "[x] No</td><td>[ ] Yes")
    assert sum(e.charged and e.kind == "box state flipped" for e in score_forms(ref, wrong).events) == 2


def test_checkbox_column_headers_bind_each_tick_to_its_numbered_question_row() -> None:
    rules = [
        field([question, option], state == (option == "Yes"), "checkbox")
        for question, state in [
            ("Part 9. 18. Have you applied for relief?", False),
            ("Part 9. 19. Have you complied with the requirement?", True),
        ]
        for option in ["Yes", "No"]
    ]
    ref = reference_from_rules(rules)
    assert ref is not None
    output = (
        "<table><tr><th>Item</th><th>Question</th><th>Yes</th><th>No</th></tr>"
        "<tr><td>19.</td><td>Have you complied with the requirement?</td><td>[x]</td><td>[ ]</td></tr>"
        "<tr><td>18.</td><td>Have you applied for relief?</td><td>[ ]</td><td>[x]</td></tr></table>"
    )
    assert score_forms(ref, output).score == 1
    wrong = output.replace("<td>[ ]</td><td>[x]</td>", "<td>[x]</td><td>[ ]</td>")
    assert sum(e.charged and e.kind == "box state flipped" for e in score_forms(ref, wrong).events) == 2


def test_spanning_column_header_does_not_turn_printed_tax_code_into_answer() -> None:
    ref = reference_from_rules([field(["254 Maintenance payments", "State tax"], "")])
    assert ref is not None
    output = (
        '<table><tr><th>Deduction</th><th colspan="2">State tax</th></tr>'
        "<tr><td>13.1 Maintenance payments</td><td>254</td><td></td></tr></table>"
    )
    result = score_forms(ref, output)
    assert not any(e.charged and e.kind == "field row id mismatch" for e in result.events)


def test_lettered_subrow_does_not_conflict_with_parent_question_number() -> None:
    ref = reference_from_rules([field(["14 - (a) Current year loss claim", "Self"], "")])
    assert ref is not None
    output = (
        "14. Loss claims\n<table><tr><th>Claim</th><th>Self</th></tr>"
        "<tr><td>(a) Current year loss claim</td><td></td></tr></table>"
    )
    assert not any(e.charged and e.kind == "field row id mismatch" for e in score_forms(ref, output).events)


def test_repeated_inline_table_headers_keep_each_parent_name_with_its_row() -> None:
    rules = [
        field([f"{number} Parent name at birth", column], value)
        for number, given, family in [(4, "Alex", "Brown"), (5, "Maria", "Jones")]
        for column, value in [("Given names", given), ("Family name", family)]
    ]
    ref = reference_from_rules(rules)
    assert ref is not None
    output = (
        "<table>"
        + "".join(
            f"<tr><td>{number}</td><td>Parent name at birth</td><td>Given names</td><td>Family name</td></tr>"
            f"<tr><td></td><td></td><td>{given}</td><td>{family}</td></tr>"
            for number, given, family in [(4, "Alex", "Brown"), (5, "Maria", "Jones")]
        )
        + "</table>"
    )
    assert score_forms(ref, output).score == 1
    swapped = output.replace("Alex", "TEMP").replace("Maria", "Alex").replace("TEMP", "Maria")
    assert sum(e.charged and e.cls in ("major", "critical") for e in score_forms(ref, swapped).events) == 2


def test_recovered_caption_keeps_relocated_code_outside_bracketed_value() -> None:
    ref = reference_from_rules([field("F3 Name of landlord shown on receipt", "Cedar Homes")])
    assert ref is not None
    output = "Name of landlord shown on receipt: F3 [ Cedar Homes ]"
    assert score_forms(ref, output).score == 1
    wrong = output.replace("Cedar Homes", "Maple Homes")
    assert any(e.charged and e.cls in ("major", "critical") for e in score_forms(ref, wrong).events)


def test_prefix_alias_does_not_reinterpret_unmarked_printed_code_as_amount() -> None:
    ref = reference_from_rules([field("Signature name", "Alex"), field("1 Amount withheld by employer", "")])
    assert ref is not None
    result = score_forms(ref, "Amount withheld by employer: 1 <empty>,00\nSignature name: Alex")
    assert not any(e.charged and e.kind == "invented value" for e in result.events)


def test_side_by_side_parent_headers_do_not_borrow_the_other_family_name() -> None:
    rules = [
        field([f"{number} Parent name at birth", column], value)
        for number, given, family in [(4, "Alex", "Brown"), (5, "Maria", "Jones")]
        for column, value in [("Given names", given), ("Family name", family)]
    ]
    ref = reference_from_rules(rules)
    assert ref is not None
    output = (
        '<table><tr><td rowspan="2">4</td><td rowspan="2">Parent name at birth</td>'
        '<td>Given names</td><td colspan="2">Family name</td>'
        '<td rowspan="2">5</td><td rowspan="2">Parent name at birth</td>'
        "<td>Given names</td><td>Family name</td></tr>"
        '<tr><td>Alex</td><td colspan="2">Brown</td><td>Maria</td><td>Jones</td></tr></table>'
    )
    assert score_forms(ref, output).score == 1
    swapped = output.replace("Brown", "TEMP").replace("Jones", "Brown").replace("TEMP", "Jones")
    assert sum(e.charged and e.cls in ("major", "critical") for e in score_forms(ref, swapped).events) == 2


def test_unmarked_relocated_codes_do_not_make_intact_captions_garbled() -> None:
    ref = reference_from_rules(
        [
            field("17 Number of days near shore", ""),
            field("36 Number of days offshore", ""),
            field("76 Basis for gift tax reduction", "10 594"),
        ]
    )
    assert ref is not None
    output = "Number of days near shore: 17\nNumber of days offshore: 36\nBasis for gift tax reduction: 76 10 594"
    result = score_forms(ref, output)
    assert not any(e.charged and e.kind == "caption garbled" for e in result.events)
    wrong = score_forms(ref, output.replace("10 594", "99 999"))
    assert any(e.charged and e.cls in ("major", "critical") and e.expected == "10 594" for e in wrong.events)


def test_short_section_caption_does_not_turn_unidentified_box_codes_into_answer() -> None:
    ref = reference_from_rules([field("Signature name", "Alex"), field("02 Prefix", "")])
    assert ref is not None
    result = score_forms(ref, "Prefix: 3 | 4\nSignature name: Alex")
    assert not any(e.charged and e.kind == "invented value" for e in result.events)


def test_indexed_table_checkbox_rules_keep_shared_header_and_row_ownership() -> None:
    question = "Was a benefit received from that country?"
    rules = [
        field(f"B1 table row {row} — {question} — {option}", selected == option, "checkbox")
        for row, selected in [(1, "No"), (2, "Yes")]
        for option in ["Yes", "No"]
    ]
    ref = reference_from_rules(rules)
    assert ref is not None
    output = (
        '<table><tr><th rowspan="2">Country</th>'
        f'<th colspan="2">{question}</th></tr><tr><th>Yes</th><th>No</th></tr>'
        "<tr><td>2.</td><td>[x]</td><td>[ ]</td></tr>"
        "<tr><td>1.</td><td>[ ]</td><td>[x]</td></tr></table>"
    )
    assert score_forms(ref, output).score == 1
    swapped = output.replace("2.</td>", "TEMP</td>").replace("1.</td>", "2.</td>").replace("TEMP</td>", "1.</td>")
    assert sum(e.charged and e.kind == "box state flipped" for e in score_forms(ref, swapped).events) == 4
    duplicate = output + output
    assert score_forms(ref, duplicate).score < 1


def test_table_row_can_inherit_its_number_from_an_immediate_section_heading() -> None:
    ref = reference_from_rules(
        [
            field(["70 Dividends and fees", "Country"], "Germany"),
            field(["70 Capital gains", "Country"], "Canada"),
        ]
    )
    assert ref is not None
    output = (
        "### 70 Investment income\n<table><tr><th>Income</th><th>Country</th></tr>"
        "<tr><td>Dividends and fees</td><td>Germany</td></tr>"
        "<tr><td>Capital gains</td><td>Canada</td></tr></table>"
    )
    assert score_forms(ref, output).score == 1
    swapped = output.replace("Germany", "TEMP").replace("Canada", "Germany").replace("TEMP", "Canada")
    assert sum(e.charged and e.cls in ("major", "critical") for e in score_forms(ref, swapped).events) == 2
    wrong = output.replace("Canada", "France") + output.replace("### 70", "### 71")
    assert any(
        e.charged and e.cls in ("major", "critical") and e.expected == "Canada" for e in score_forms(ref, wrong).events
    )


def test_printed_code_before_empty_cents_does_not_hide_a_real_amount() -> None:
    ref = reference_from_rules([field("1 Amount withheld by employer", "")])
    assert ref is not None
    assert score_forms(ref, "Amount withheld by employer: 1 <empty>,00").score == 1
    for value in ["1,00", "999,00"]:
        assert any(
            e.charged and e.kind == "invented value"
            for e in score_forms(ref, f"Amount withheld by employer: {value}").events
        )


@pytest.mark.parametrize("value_lines", ["2: 694,00", "2\n694\n,00"])
def test_numbered_subfield_value_is_not_blocked_by_an_intervening_caption(value_lines: str) -> None:
    ref = reference_from_rules(
        [
            field("1 Of which deferred tax", ""),
            field("RN45 col. 2 — Tax due", "694"),
        ]
    )
    assert ref is not None
    output = f"RN45 Tax due\nOf which deferred tax: 1 <empty>,00\n{value_lines}\nRN46 Tax credit"
    assert score_forms(ref, output).score == 1
    wrong = output.replace("694", "695")
    assert any(
        e.charged and e.cls in ("major", "critical") and e.expected == "694" for e in score_forms(ref, wrong).events
    )


def test_empty_html_table_does_not_crash_rule_association() -> None:
    ref = reference_from_rules([field("Signature name", "Alex")])
    assert ref is not None
    assert score_forms(ref, "<table></table>\nSignature name: Alex").score == 1


@pytest.mark.parametrize("label,value", [("4 Name", "Alex"), ("10 Code", "25")])
def test_short_caption_with_one_answer_keeps_prefix_alias_credit(label: str, value: str) -> None:
    ref = reference_from_rules([field(label, value)])
    assert ref is not None
    assert score_forms(ref, f"{label.split(' ', 1)[1]}: {value}").score == 1


def test_repeated_row_in_another_section_does_not_create_a_row_id_conflict() -> None:
    ref = reference_from_rules([field(["51 Is the applicant a sportsperson?", "Self"], False, "checkbox")])
    assert ref is not None
    output = (
        "### 51 Retirement savings\n"
        "Long instructions about completing the retirement savings section before filling this grid.\n"
        "<table><tr><th>Question</th><th>Self</th></tr>"
        "<tr><td>Is the applicant a sportsperson?</td><td>[ ]</td></tr></table>\n"
        "### 52 Annuity contracts\n<table><tr><th>Question</th><th>Self</th></tr>"
        "<tr><td>Is the applicant a sportsperson?</td><td>[x]</td></tr></table>"
    )
    assert not any(e.charged and e.kind == "field row id mismatch" for e in score_forms(ref, output).events)


def test_missing_indexed_section_options_cannot_borrow_previous_sections_tick() -> None:
    rules = [
        field([f"44 Applicant {n} — Moving?", option], n == 2 and option == "No", "checkbox")
        for n in (2, 3)
        for option in ("No", "Yes")
    ]
    ref = reference_from_rules(rules)
    assert ref is not None
    result = score_forms(ref, "## Applicant 2\nMoving? No [x] Yes [ ]\n## Applicant 3\nMoving?")
    assert not any(e.charged and e.kind == "box state flipped" for e in result.events)
    assert sum(e.charged and e.kind == "unchecked option omitted" for e in result.events) == 2


def test_printed_row_code_in_blank_table_cell_is_not_an_entered_answer() -> None:
    ref = reference_from_rules([field("3a Rental income / Federal amount", "")])
    assert ref is not None
    output = (
        "<table><tr><th>Item</th><th>Federal amount</th></tr><tr><td>3 a Rental income</td><td>3a</td></tr></table>"
    )
    result = score_forms(ref, output)
    assert not any(e.charged and e.kind in {"invented value", "printed text in a blank field"} for e in result.events)


def test_printed_allowance_before_field_code_is_not_read_as_blank_input_value() -> None:
    ref = reference_from_rules([field("24.1 Child deduction 370 — State tax", "")])
    assert ref is not None
    output = (
        "<table><tr><th>Item</th><th>State tax</th><th>Code</th></tr>"
        "<tr><td>24.1 Child deduction</td><td>9,300</td><td>370</td></tr></table>"
    )
    result = score_forms(ref, output)
    assert not any(e.charged and e.kind == "invented value" and e.got == "9,300" for e in result.events)


def test_digit_boxes_before_cents_do_not_lose_first_digit_as_line_id() -> None:
    ref = reference_from_rules([field("Pay for USC", "88200")])
    assert ref is not None
    assert score_forms(ref, "Pay for USC: 8 8 2 0 0 .00").score == 1
    assert any(e.charged and e.kind == "wrong value" for e in score_forms(ref, "Pay for USC: 9 8 2 0 0 .00").events)


def test_relocated_complete_field_id_is_not_a_garbled_caption() -> None:
    ref = reference_from_rules([field("D15 End date", "")])
    assert ref is not None
    assert score_forms(ref, "End date: D15 [____]").score == 1


def test_prefix_recovery_preserves_explicit_spouse_context() -> None:
    ref = reference_from_rules([field(["Spouse", "20 Date of birth"], "")])
    assert ref is not None
    result = score_forms(ref, "Date of birth: 09.06.1965\nSpouse\nName: ___ Date of birth: ___")
    assert not any(e.charged and e.kind == "invented value" for e in result.events)


@pytest.mark.parametrize(
    "label,value,output,wrong",
    [
        (
            "23 Manufacturer's list price",
            "63,016.98",
            "Manufacturer's list price: 63,016.98 23",
            "Manufacturer's list price: 63,017.98 23",
        ),
        ("3 Comune", "NAPOLI", "Comune: 3 NAPOLI", "Comune: 3 ROMA"),
        ("8 Categoria", "C", "Categoria: 8 C", "Categoria: 8 D"),
        (
            "1 di cui exit-tax rateizzata (Quadro TR)",
            "",
            "di cui exit-tax rateizzata (Quadro TR): 1 <empty>,00",
            "di cui exit-tax rateizzata (Quadro TR): 1,00",
        ),
        (
            "25B Previous Employer, if applicable",
            "0.00",
            "Previous Employer, if applicable: 25B",
            "Previous Employer, if applicable: 250.00",
        ),
    ],
)
def test_relocated_printed_ids_do_not_become_answers(label: str, value: str, output: str, wrong: str) -> None:
    ref = reference_from_rules([field(label, value)])
    assert ref is not None
    assert score_forms(ref, output).score == 1
    assert score_forms(ref, wrong).score < 1


def test_reordered_possessive_parent_sections_keep_their_own_answers() -> None:
    ref = reference_from_rules(
        [
            field("Part 2. 24.a. Beneficiary's Parent 1 — Family Name (Last Name)", "Phùng"),
            field("Part 2. 29.a. Beneficiary's Parent 2 — Family Name (Last Name)", "Tran"),
        ]
    )
    assert ref is not None
    output = (
        "## Parent 2's Information\n29.a. Family Name (Last Name): Tran\n"
        "## Parent 1's Information\n24.a. Family Name (Last Name): Phùng"
    )
    assert score_forms(ref, output).score == 1
    wrong = output.replace("Tran", "TEMP").replace("Phùng", "Tran").replace("TEMP", "Phùng")
    assert sum(e.charged and e.cls == "critical" for e in score_forms(ref, wrong).events) == 2


@pytest.mark.parametrize("reverse", [False, True])
def test_question_on_only_one_checkbox_option_keeps_parent_ownership(reverse: bool) -> None:
    rules = [
        field("44 Parent 1 — Migrating with you? No", True, "checkbox"),
        field("44 Parent 1 — yes", False, "checkbox"),
        field("44 Parent 2 — Migrating with you? No", False, "checkbox"),
        field("44 Parent 2 — yes", True, "checkbox"),
    ]
    ref = reference_from_rules(rules[::-1] if reverse else rules)
    assert ref is not None
    output = "## Parent 1\nMigrating with you? No [x] Yes [ ]\n## Parent 2\nMigrating with you? No [ ] Yes [x]"
    assert score_forms(ref, output).score == 1
    wrong = output.replace("No [x] Yes [ ]", "No [ ] Yes [x]")
    assert any(e.charged and e.kind == "box state flipped" for e in score_forms(ref, wrong).events)


def test_numbered_amount_caption_is_not_a_section_boundary() -> None:
    ref = reference_from_rules([field("Amount 25:", "45,369.46"), field("amount 21:", "22,684.73")])
    assert ref is not None
    assert score_forms(ref, "Amount 25:\n45,369.46\namount 21:\n22,684.73").score == 1


@pytest.mark.parametrize("expected", ["08/07/1965", ""])
def test_date_header_format_hint_is_not_the_answer_and_blank_rows_are_not_skipped(expected: str) -> None:
    ref = reference_from_rules([field("Name", "Ada"), field("Date of birth", expected), field("City", "Paris")])
    assert ref is not None
    prefix = (
        '<table><tr><th rowspan="2">Name</th><th colspan="3">Date of birth<br>Day Month Year</th>'
        '<th rowspan="2">City</th></tr><tr></tr>'
    )
    suffix = '<tr><td>Other</td><td colspan="3">01/01/2000</td><td>Rome</td></tr></table>'
    output = prefix + f'<tr><td>Ada</td><td colspan="3">{expected}</td><td>Paris</td></tr>' + suffix
    assert not any(e.charged and e.label == "Date of birth" for e in score_forms(ref, output).events)
    wrong = prefix + '<tr><td>Ada</td><td colspan="3">02/02/2001</td><td>Paris</td></tr>' + suffix
    assert any(e.charged and e.label == "Date of birth" for e in score_forms(ref, wrong).events)


def test_parenthesized_caption_on_next_line_cannot_swallow_the_answer() -> None:
    ref = reference_from_rules(
        [field("Family Name (Last Name)", "MacGyver"), field("Given Name (First Name)", "Rahul")]
    )
    assert ref is not None
    output = "Family Name\n(Last Name)\n\nMacGyver\nGiven Name\n(First Name)\n\nRahul"
    assert score_forms(ref, output).score == 1
    assert score_forms(ref, output.replace("MacGyver", "Wrong")).score < 1


@pytest.mark.parametrize("checked,unchecked", [("√", "□"), (r"$\boxtimes$", r"$\square$")])
def test_ocr_glyph_and_latex_boxes_keep_yes_no_ownership(checked: str, unchecked: str) -> None:
    from parse_bench.evaluation.metrics.parse.forms_v2 import clean_text, tokenize

    ref = reference_from_rules(
        [field(["Eligible?", "Yes"], True, "checkbox"), field(["Eligible?", "No"], False, "checkbox")]
    )
    assert ref is not None
    assert score_forms(ref, f"Eligible? Yes {checked} No {unchecked}").score == 1
    assert score_forms(ref, f"Eligible? Yes {unchecked} No {checked}").score < 1
    assert not any(t.box for t in tokenize(clean_text("No √9; result √x")))


def test_adjacent_numbered_empty_slots_do_not_become_entered_numbers() -> None:
    ref = reference_from_rules([field("1 Codice", "")])
    assert ref is not None
    assert score_forms(ref, "Codice: 1 <empty> 2 <empty> ,00").score == 1
    assert score_forms(ref, "Codice: 1 [999] 2 <empty> ,00").score < 1


def test_printed_field_id_before_alphanumeric_answer_is_not_extra_text() -> None:
    ref = reference_from_rules([field("5 Codice Comune", "F839")])
    assert ref is not None
    assert score_forms(ref, "Codice Comune: 5 F839").score == 1
    assert score_forms(ref, "Codice Comune: 5 F838").score < 1


@pytest.mark.parametrize("empty", [False, True])
def test_second_header_tier_does_not_replace_first_data_row(empty: bool) -> None:
    value = "" if empty else "08/07/1965"
    ref = reference_from_rules([field("Name", "Ada"), field("Date of birth", value), field("City", "Paris")])
    assert ref is not None
    output = (
        '<table><tr><th>Name</th><th colspan="3">Date of birth</th><th>City</th></tr>'
        "<tr><td></td><td>Day</td><td>Month</td><td>Year</td><td></td></tr>"
    )
    date = "<td></td><td></td><td></td>" if empty else "<td>08</td><td>07</td><td>1965</td>"
    output += (
        "<tr><td>Ada</td>"
        + date
        + '<td>Paris</td></tr><tr><td>Other</td><td colspan="3">01/01/2000</td><td>Rome</td></tr></table>'
    )
    assert not any(e.charged and e.label == "Date of birth" for e in score_forms(ref, output).events)


def test_numbered_answer_block_after_another_parent_keeps_its_owner() -> None:
    ref = reference_from_rules(
        [
            field("Part 2. 26. Parent 1 Sex — Male", True, "checkbox"),
            field("Part 2. 26. Parent 1 Sex — Female", False, "checkbox"),
            field("Part 2. 32. Parent 2 Sex — Male", False, "checkbox"),
            field("Part 2. 32. Parent 2 Sex — Female", True, "checkbox"),
        ]
    )
    assert ref is not None
    output = (
        "Parent 1's Information\n26. Sex\nParent 2's Information\n"
        "32. Sex [ ] Male [x] Female\n26. Sex [x] Male [ ] Female"
    )
    assert not any(e.charged and e.cls in {"major", "critical"} for e in score_forms(ref, output).events)
    wrong = output.replace("26. Sex [x] Male [ ] Female", "26. Sex [ ] Male [x] Female")
    assert any(e.charged and e.kind == "box state flipped" for e in score_forms(ref, wrong).events)


def test_two_row_numbered_field_grid_uses_its_own_value() -> None:
    ref = reference_from_rules(
        [
            field(["Other Information", "26. Country of Birth"], "Canada"),
            field(["Parent 1's Information", "30. Country of Birth"], "Italy"),
        ]
    )
    assert ref is not None

    def grid(identifier: int, value: str) -> str:
        return (
            f"<table><tr><td>{identifier}.</td><td>Country of Birth</td></tr><tr><td></td><td>{value}</td></tr></table>"
        )

    assert score_forms(ref, grid(30, "Italy") + grid(26, "Canada")).score == 1
    assert score_forms(ref, grid(30, "Canada") + grid(26, "Italy")).score < 1


def test_vertical_field_labels_are_not_header_continuations() -> None:
    ref = reference_from_rules([field("Name", ""), field("Phone", ""), field("Address", ""), field("Code", "")])
    assert ref is not None
    output = (
        "<table><tr><td>Name</td><td></td><td>Code</td><td></td></tr>"
        "<tr><td>Phone</td><td></td><td>Address</td><td></td></tr>"
        '<tr><td colspan="4">Mail the form to the office</td></tr></table>'
    )
    assert score_forms(ref, output).score == 1
    wrong = output.replace("<td>Phone</td><td></td>", "<td>Phone</td><td>12345</td>")
    assert score_forms(ref, wrong).score < 1


def test_dotted_tax_row_id_preserves_year_and_income_columns() -> None:
    ref = reference_from_rules(
        [field(["16.a. Most Recent", "Tax Year"], "2025"), field("16.a. Most Recent — Total Income", "107,864")]
    )
    assert ref is not None
    output = (
        "<table><tr><th></th><th>Tax Year</th><th>Total Income</th></tr>"
        "<tr><td>16.a. Most Recent</td><td>2025</td><td>107,864</td></tr></table>"
    )
    assert score_forms(ref, output).score == 1
    assert score_forms(ref, output.replace("<td>2025</td><td>107,864</td>", "<td>107,864</td><td>2025</td>")).score < 1


def test_currency_slot_after_printed_instructions_does_not_borrow_later_amounts() -> None:
    ref = reference_from_rules([field("101 Nominated assets.", "0"), field("102 Total", "0")])
    assert ref is not None
    output = (
        "101 Nominated assets. Include details of the nominated assets\n"
        "in the additional information on page 19\n- go to the notes\n\n£\n\n0\n102 Total\n0"
    )
    assert score_forms(ref, output).score == 1
    assert score_forms(ref, output.replace("£\n\n0", "£\n\n123")).score < 1
    assert score_forms(ref, output.replace("£\n\n0\n", "")).score < 1


def test_ambiguous_root_tick_between_options_is_not_assigned_to_previous_option() -> None:
    from parse_bench.evaluation.metrics.parse.forms_v2 import clean_text, tokenize

    assert not any(t.box for t in tokenize(clean_text("No √ Yes")))
    assert "√" in clean_text("□ Yes √ NoIf yes, continue")


def test_empty_tag_before_printed_cents_is_not_an_invented_amount() -> None:
    ref = reference_from_rules([field("1 Fondi comuni", "")])
    assert ref is not None
    output = "<table><tr><td>Fondi comuni</td><td>1 <empty> ,00</td></tr></table>"
    assert not any(e.charged and e.kind == "invented value" for e in score_forms(ref, output).events)
    assert any(
        e.charged and e.kind == "invented value" for e in score_forms(ref, output.replace("<empty>", "999")).events
    )


def test_complete_date_in_first_spanned_cell_does_not_absorb_na_neighbor() -> None:
    ref = reference_from_rules([field("Name", "Ada"), field("Date of birth", "08/07/1965"), field("City", "Paris")])
    assert ref is not None
    output = (
        '<table><tr><th>Name</th><th colspan="3">Date of birth Day Month Year</th><th>City</th></tr>'
        "<tr><td>Ada</td><td>08/07/1965</td><td>N/A</td><td></td><td>Paris</td></tr></table>"
    )
    assert not any(e.charged and e.label == "Date of birth" for e in score_forms(ref, output).events)
    assert any(
        e.charged and e.label == "Date of birth"
        for e in score_forms(ref, output.replace("08/07/1965", "09/09/2000")).events
    )


def test_date_hint_below_horizontal_blank_does_not_read_instruction_row() -> None:
    ref = reference_from_rules([field("Date of birth", ""), field("Gender", ""), field("City", "")])
    assert ref is not None
    output = (
        "<table><tr><td>Date of birth</td><td>/___/___</td><td>Gender</td><td>City</td></tr>"
        '<tr><td colspan="4">MM/DD/YYYY</td></tr>'
        '<tr><td colspan="4">Submission of this information is required by law</td></tr></table>'
    )
    assert not any(e.charged and e.label == "Date of birth" for e in score_forms(ref, output).events)


def test_numbered_blank_slots_with_section_id_do_not_become_an_answer() -> None:
    ref = reference_from_rules([field("02 Prefixo", "")])
    assert ref is not None
    assert not any(
        e.charged and e.kind == "invented value" for e in score_forms(ref, "Prefixo: 3 <empty/> 4 <empty/>").events
    )
    assert any(e.charged and e.kind == "invented value" for e in score_forms(ref, "Prefixo: 3 999 4 <empty/>").events)


@pytest.mark.parametrize("scope_kind", ["context", "prefix", "reused_id"])
def test_two_row_grid_cannot_override_a_conflicting_section(scope_kind: str) -> None:
    first = ["Parent 1", "26. Country of Birth"] if scope_kind != "prefix" else "Part 1. 26. Country of Birth"
    rules = [field(first, "Italy")]
    if scope_kind == "reused_id":
        rules.append(field(["Parent 2", "26. Country of Birth"], "Italy"))
    ref = reference_from_rules(rules)
    assert ref is not None
    group = "Part" if scope_kind == "prefix" else "Parent"
    output = (
        f"{group} 1\nCountry of Birth: Canada\n{group} 2\n"
        "<table><tr><td>26.</td><td>Country of Birth</td></tr>"
        "<tr><td></td><td>Italy</td></tr></table>"
    )
    assert score_forms(ref, output).score < 1
