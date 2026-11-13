"""forms_v2: one value read by a blank field and by the field it belongs to is charged once."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _ref(*fields: tuple[str, str | None]):
    items = [
        {"type": "field", "field": "text", "label": k, **({"value": v} if v else {"isEmpty": True})} for k, v in fields
    ]
    return reference_from_tree([{"page_number": 1, "items": items}])


def _critical(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.got) for e in score_forms(ref, md).events if e.charged and e.cls == "critical"]


PERIODS = _ref(
    ("Course", None),
    ("from", None),
    ("until", None),
    ("Course", None),
    ("from", "980,831"),
    ("until", None),
    ("Course", None),
    ("from", "953,738"),
    ("until", "827,469"),
)
SHIFTED = "**Course:** [___]\n**from:** 980,831\n**from:** 953,738\n**until:** 827,469\n"


def test_a_value_read_by_a_blank_twin_is_credited_to_the_field_it_matches():
    result = score_forms(PERIODS, SHIFTED)
    assert [e.cls for e in result.events if e.charged and e.label == "until"] == ["minor", "minor"]
    assert {s["item"]: s["verdict"] for s in result.states}[5] == "minor"


def test_a_value_in_a_blank_field_that_no_other_field_reads_is_still_critical():
    md = "**Course:** [___]\n**from:** 980,831\n**until:** 05.05.2025\n**from:** 953,738\n**until:** 827,469\n"
    assert ("until", "05.05.2025") in _critical(PERIODS, md)


OTHER = _ref(("Other expenses (specify)", None), ("Other expenses (specify)", "945,345"))


def test_one_caption_with_two_slots_gives_its_value_to_the_slot_it_matches():
    md = "| Rent | 1,200 | 7F |\n| --- | --- | --- |\n| Other expenses (specify) : | 945,345 | 7G |\n"
    assert _critical(OTHER, md) == []


def test_a_value_shifted_into_an_earlier_block_is_still_charged():
    city = _ref(("City", None), ("City", "Paris"))
    assert ("City", "Paris") in _critical(city, "City: Paris\n\nCity: [___]\n")


def test_a_value_before_its_own_caption_is_not_charged_to_the_blank_slot_it_follows():
    ref = _ref(("an amount up to R", None), ("rand", "five thousand"))
    assert _critical(ref, "to pay an amount up to R[___] (five thousand rand) as the Master may claim\n") == []


def test_a_blank_field_in_another_section_does_not_stand_in_for_a_missing_value():
    city = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {
                        "type": "section",
                        "label": sec,
                        "items": [{"type": "field", "field": "text", "label": "City", **v}],
                    }
                    for sec, v in (("Applicant", {"isEmpty": True}), ("Employer", {"value": "Paris"}))
                ],
            }
        ]
    )
    assert ("City", "Paris") in _critical(city, "## Applicant\n\nCity: Paris\n\n## Employer\n\nName: [___]\n")
