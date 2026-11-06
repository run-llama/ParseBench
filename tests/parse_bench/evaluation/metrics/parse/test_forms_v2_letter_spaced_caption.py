"""forms_v2: a short capital caption printed one letter per comb box ("N L 123456789 B")."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms

VAT = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "field", "field": "text", "label": "Name of the company", "value": "Harbour Freight Co"},
                {"type": "field", "field": "text", "label": "NL", "value": "123456789"},
                {"type": "field", "field": "text", "label": "VAT number", "isEmpty": True},
            ],
        }
    ]
)


def _charged(md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(VAT, md).events if e.charged]


def test_a_letter_spaced_caption_reads_its_value():
    assert _charged("**Name of the company:** Harbour Freight Co\n\nN L 123456789\n\n**VAT number:** [___]\n") == []


def test_a_wrong_value_after_a_letter_spaced_caption_is_still_charged():
    md = "**Name of the company:** Harbour Freight Co\n\nN L 123456780\n\n**VAT number:** [___]\n"
    assert ("NL", "123456789") in _charged(md)


def test_single_letters_inside_a_line_are_not_a_caption():
    md = "**Name of the company:** Harbour Freight Co\n\nPlan N L 123456789\n\n**VAT number:** [___]\n"
    assert ("NL", "123456789") in _charged(md)


ACCOUNT = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "field", "field": "text", "label": "Account number", "value": "NL 1234"},
                {"type": "field", "field": "text", "label": "NL", "value": "1234"},
            ],
        }
    ]
)


def test_another_fields_value_spelled_like_the_caption_stays_that_fields_value():
    charged = [
        (e.label, e.expected) for e in score_forms(ACCOUNT, "**Account number:**\nN L 1234\n").events if e.charged
    ]
    assert ("Account number", "NL 1234") not in charged


TWO_PAGES = reference_from_tree(
    [
        {"page_number": p, "items": [{"type": "field", "field": "text", "label": "NL", "value": v}]}
        for p, v in ((1, "123456789"), (2, "987654321"))
    ]
)


def test_a_letter_spaced_caption_repeated_on_each_page_reads_each_value():
    md = "N L 123456789\n\n---\n\nN L 987654321\n"
    assert [e for e in score_forms(TWO_PAGES, md).events if e.charged] == []


def test_a_value_sharing_the_letters_on_its_own_line_leaves_the_caption_alone():
    md = "**Account number:** NL9999\n\nN L 1234\n"
    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {"type": "field", "field": "text", "label": "Account number", "value": "NL9999"},
                    {"type": "field", "field": "text", "label": "NL", "value": "1234"},
                ],
            }
        ]
    )
    assert [e for e in score_forms(ref, md).events if e.charged] == []


def test_a_run_taken_by_another_caption_does_not_count_against_the_free_one():
    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {"type": "field", "field": "text", "label": "N L", "value": "111"},
                    {"type": "field", "field": "text", "label": "NL", "value": "222"},
                ],
            }
        ]
    )
    assert [e for e in score_forms(ref, "N L 111\n\nN L 222\n").events if e.charged] == []


def test_a_value_spelled_like_the_caption_with_a_note_after_it_stays_that_fields_value():
    md = "**Account number:**\nN L 1234 note\n"
    charged = [(e.label, e.got) for e in score_forms(ACCOUNT, md).events if e.charged and e.label == "Account number"]
    assert charged == [("Account number", "N L 1234 note")]  # its own value, the note extra text


def test_a_value_spelled_exactly_like_the_caption_stays_that_fields_value():
    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {"type": "field", "field": "text", "label": "Country code", "value": "NL"},
                    {"type": "field", "field": "text", "label": "NL", "value": "1234"},
                ],
            }
        ]
    )
    charged = [(e.label, e.kind) for e in score_forms(ref, "**Country code:**\nN L\n").events if e.charged]
    assert ("Country code", "value lost") not in charged
