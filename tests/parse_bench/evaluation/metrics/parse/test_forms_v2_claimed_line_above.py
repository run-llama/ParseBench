"""forms_v2: a value line between two captions, after one caption and above the next."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _ref(*fields: tuple[str, str | None]):
    items = [
        {"type": "field", "field": "text", "label": k, **({"value": v} if v else {"isEmpty": True})} for k, v in fields
    ]
    return reference_from_tree([{"page_number": 1, "items": items}])


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]


TENANT = _ref(("(Tenant name)", "Ann Brook"), ("(Rented address)", "12 Elm St"))


def test_stacked_under_line_captions_each_read_the_line_above():
    assert _charged(TENANT, "Ann Brook\n\n(Tenant name)\n\n12 Elm St\n\n(Rented address)\n") == []


def test_a_wrong_value_above_an_under_line_caption_is_still_charged():
    assert ("(Rented address)", "12 Elm St") in _charged(
        TENANT, "Ann Brook\n\n(Tenant name)\n\n9 Oak Ave\n\n(Rented address)\n"
    )


HOURS = _ref(("Hourly rate", None), ("hours per week", "38"))


def test_a_value_on_the_next_line_of_a_blank_caption_is_not_credited_to_the_caption_below():
    assert ("hours per week", "38") in _charged(HOURS, "Hourly rate\n\n38\n\nhours per week\n")


TWIN = _ref(("Amount paid", "500"), ("Amount due", "500"))


def test_one_line_does_not_credit_both_the_caption_above_and_the_caption_below():
    assert ("Amount due", "500") in _charged(TWIN, "Amount paid\n\n500\n\nAmount due\n")
