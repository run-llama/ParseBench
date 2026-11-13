"""forms_v2: a page that prints its form twice, a filled copy and a blank copy, with captions over two lines."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]


SLIP = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "field", "field": "text", "label": "Employment income Revenus d'emploi", "value": "312,587"},
                {"type": "field", "field": "text", "label": "Union dues Cotisations syndicales", "value": "6,331"},
            ],
        }
    ]
)
# the filled copy lists the boxes in another order than the blank copy and the reference
SLIP_MD = (
    "44 Union dues\nCotisations syndicales\n{dues}\n\n14 Employment income\nRevenus d'emploi\n312,587\n\n"
    "14\nEmployment income\nRevenus d'emploi\n\n44\nUnion dues\nCotisations syndicales\n"
)


def test_a_two_line_caption_reads_its_filled_copy_when_a_blank_copy_follows():
    assert _charged(SLIP, SLIP_MD.format(dues="6,331")) == []


def test_a_wrong_value_in_the_filled_copy_is_still_charged():
    assert ("Union dues Cotisations syndicales", "6,331") in _charged(SLIP, SLIP_MD.format(dues="9,999"))


def test_a_caption_split_by_a_value_over_two_lines_is_no_copy():
    ref = reference_from_tree(
        [{"page_number": 1, "items": [{"type": "field", "field": "text", "label": "Full name", "value": "Alice"}]}]
    )
    assert _charged(ref, "Full name: Alice\n\nFull: Alice\nName: Smith\n") == []
