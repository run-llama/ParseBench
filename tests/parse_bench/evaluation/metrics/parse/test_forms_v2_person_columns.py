"""forms_v2: sibling sections written as one row-label table with a value column per section."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _person(first: str, phone: str | None) -> dict[str, Any]:
    fields = [("First name", first), ("Phone", phone)]
    return {
        "type": "section",
        "items": [
            {"type": "field", "field": "text", "label": k, **({"value": v} if v else {"isEmpty": True})}
            for k, v in fields
        ],
    }


PEOPLE = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "section",
                    "id": "3",
                    "label": "Tenants leaving",
                    "items": [_person("Ann", "555-0101"), _person("Ben", None)],
                },
            ],
        }
    ]
)


def _charged(second: str) -> list[tuple[str, str]]:
    md = "3 **Tenants leaving**\n\n| Field | Person 1 | Person 2 |\n|---|---|---|\n"
    md += f"| First name | Ann | {second} |\n| Phone | 555-0101 | |\n"
    return [(e.label, e.expected) for e in score_forms(PEOPLE, md).events if e.charged]


def test_second_section_reads_the_second_value_column_of_a_shared_row_label():
    assert _charged("Ben") == []


def test_a_wrong_value_in_the_second_column_is_charged():
    assert ("First name", "Ben") in _charged("Bo")


def _contact(email: str, tick: bool) -> dict[str, Any]:
    return {
        "type": "section",
        "items": [
            {"type": "field", "field": "text", "label": "Email", "value": email},
            {"type": "field", "field": "checkbox", "label": "Tick to receive notices by email", "value": tick},
        ],
    }


CONTACTS = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "section",
                    "id": "3",
                    "label": "Tenants leaving",
                    "items": [_contact("ann@example.org", False), _contact("ben@example.org", True)],
                },
            ],
        }
    ]
)


def _contacts(first_tick: str, second_tick: str) -> list[tuple[str, str]]:
    md = "3 **Tenants leaving**\n\n| Field | Person 1 | Person 2 |\n|---|---|---|\n"
    md += "| Email | ann@example.org | ben@example.org |\n"
    md += f"| Tick to receive notices by email | {first_tick} | {second_tick} |\n\n"
    md += "**Email completed form to forms@example.org**\n"
    return [(e.label, e.expected) for e in score_forms(CONTACTS, md).events if e.charged]


def test_the_second_column_wins_over_a_later_line_opening_with_the_caption():
    assert ("Email", "ben@example.org") not in _contacts("[ ]", "[x]")


def test_a_box_in_the_second_value_column_is_read():
    assert _contacts("[ ]", "[x]") == []


def test_a_box_state_swapped_between_columns_is_charged():
    assert ("Tick to receive notices by email", "[x]") in _contacts("[x]", "[ ]")


def test_a_sibling_with_its_own_table_reads_that_table():
    md = "3 **Tenants leaving**\n\n| Field | Person 1 | Person 2 |\n|---|---|---|\n"
    md += "| Email | ann@example.org | |\n| Tick to receive notices by email | [ ] | |\n\n"
    md += "| Field | Person 2 |\n|---|---|\n| Email | ben@example.org |\n| Tick to receive notices by email | [x] |\n"
    assert [(e.label, e.expected) for e in score_forms(CONTACTS, md).events if e.charged] == []


def test_a_later_table_repeating_the_row_label_does_not_take_a_filled_column():
    md = "3 **Tenants leaving**\n\n| Field | Person 1 | Person 2 |\n|---|---|---|\n"
    md += "| Email | ann@example.org | ben@example.org |\n| Tick to receive notices by email | [ ] | [x] |\n\n"
    md += "| Notes | |\n|---|---|\n| Email | send reminders |\n"
    assert ("Email", "ben@example.org") not in [
        (e.label, e.expected) for e in score_forms(CONTACTS, md).events if e.charged
    ]


def test_each_sibling_on_its_own_row_of_one_table_reads_its_own_row():
    md = "3 **Tenants leaving**\n\n| Email | ann@example.org | person 1 |\n|---|---|---|\n"
    md += "| Tick to receive notices by email | [ ] | |\n"
    md += "| Email | ben@example.org | person 2 |\n| Tick to receive notices by email | [x] | |\n"
    assert [(e.label, e.expected) for e in score_forms(CONTACTS, md).events if e.charged] == []


def test_a_notes_row_repeating_the_caption_leaves_the_person_column_reading():
    md = "3 **Tenants leaving**\n\n| Field | Person 1 | Person 2 |\n|---|---|---|\n"
    md += "| Email | ann@example.org | ben@example.org |\n| Tick to receive notices by email | [ ] | [x] |\n"
    md += "| Email | send reminders | |\n"
    assert ("Email", "ben@example.org") not in [
        (e.label, e.expected) for e in score_forms(CONTACTS, md).events if e.charged
    ]


def test_sibling_rows_filled_like_the_first_row_each_read_their_own_row():
    md = "3 **Tenants leaving**\n\n| Field | Person 1 | Person 2 |\n|---|---|---|\n"
    md += "| Email | ann@example.org | ann.alt@example.org |\n| Tick to receive notices by email | [ ] | |\n"
    md += "| Email | ben@example.org | ben.alt@example.org |\n| Tick to receive notices by email | [x] | |\n"
    assert ("Email", "ben@example.org") not in [
        (e.label, e.expected) for e in score_forms(CONTACTS, md).events if e.charged
    ]


def _statement(name: str, wages: str) -> dict[str, Any]:
    field = {"type": "field", "field": "text", "id": "4", "label": "GA WAGES / INCOME", "value": wages}
    return {"type": "section", "label": f"(INCOME STATEMENT {name})", "items": [field]}


STATEMENTS = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "section",
                    "label": "INCOME STATEMENT DETAILS",
                    "items": [_statement("A", "100.00"), _statement("B", "200"), _statement("C", "300")],
                },
            ],
        }
    ]
)


def _statements(b: str) -> list[tuple[str, str]]:
    md = "**INCOME STATEMENT DETAILS** Enter income on Line 4 GA Wages/Income. For other statements see below.\n\n"
    md += "| | (INCOME STATEMENT A) | (INCOME STATEMENT B) | (INCOME STATEMENT C) |\n|---|---|---|---|\n"
    md += f"| 4. GA WAGES / INCOME | 100.00 | {b} | 300 |\n"
    return [(e.label, e.expected) for e in score_forms(STATEMENTS, md).events if e.charged]


def test_labelled_sibling_sections_read_the_column_headed_by_their_section():
    assert _statements("200") == []


def test_a_wrong_value_under_a_labelled_section_column_is_charged():
    assert ("GA WAGES / INCOME", "200") in _statements("201")
