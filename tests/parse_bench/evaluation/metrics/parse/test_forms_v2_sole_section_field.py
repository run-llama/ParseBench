"""forms_v2: the only field of a section, captioned by a bare instruction, written after the section caption."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _section(label: str, *fields: tuple[str, str | None]) -> dict:
    items = [
        {"type": "field", "field": "text", "label": k, **({"value": v} if v else {"isEmpty": True})} for k, v in fields
    ]
    return {"type": "section", "label": label, "items": items}


CONTACT = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                _section("PARTICULARS", ("Name", "Ada Moss")),
                _section("E-CONTACT MEANS", ("Please fill in", "ada@example.com")),
                _section("ADDRESS", ("Flat/Room", None), ("Floor", "7")),
            ],
        }
    ]
)


def _charged(md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(CONTACT, md).events if e.charged]


def _md(contact: str) -> str:
    return (
        f"**PARTICULARS**\n\n**Name:** Ada Moss\n\n**E-CONTACT MEANS:** {contact}\n\n"
        "**ADDRESS**\n\n**Flat/Room:** [___]\n**Floor:** 7\n"
    )


def test_the_sole_field_of_a_section_reads_after_the_section_caption():
    assert _charged(_md("ada@example.com")) == []


def test_a_wrong_value_after_the_section_caption_is_still_charged():
    assert ("Please fill in", "ada@example.com") in _charged(_md("bob@example.com"))


def test_a_one_field_form_read_after_its_section_caption_scores_full():
    ref = reference_from_tree(
        [{"page_number": 1, "items": [_section("E-CONTACT MEANS", ("Please fill in", "ada@example.com"))]}]
    )
    assert score_forms(ref, "**E-CONTACT MEANS:** ada@example.com\n").score == 1.0


def test_a_top_level_field_after_a_section_is_not_read_from_that_section():
    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {
                        "type": "section",
                        "label": "PRIMARY",
                        "items": [{"type": "static_text", "text": "Contact details"}],
                    },
                    {"type": "field", "field": "text", "label": "Backup email", "value": "ada@example.com"},
                ],
            }
        ]
    )
    charged = [(e.label, e.expected) for e in score_forms(ref, "**PRIMARY:** ada@example.com\n").events if e.charged]
    assert ("Backup email", "ada@example.com") in charged
