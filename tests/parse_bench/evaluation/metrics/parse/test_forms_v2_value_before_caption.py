"""forms_v2: values printed before their caption (caption under or after the blank)."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _field(label: str, value: str | None) -> dict[str, Any]:
    if value is None:
        return {"type": "field", "field": "text", "label": label, "isEmpty": True}
    return {"type": "field", "field": "text", "label": label, "value": value}


def _ref(*fields: dict[str, Any]):
    return reference_from_tree([{"page_number": 1, "items": list(fields)}])


PARTIES = _ref(
    _field("Partie demanderesse", "Example Holdings Ltd"),
    _field("Partie défenderesse", "Sample Partners LP"),
    _field("Dossier", None),
)


def test_value_on_the_line_above_its_caption_reads_like_key_value_lines():
    above = "Example Holdings Ltd\nPartie demanderesse\n\nc.\n\nSample Partners LP\nPartie défenderesse\n\nDossier"
    key_value = "Partie demanderesse: Example Holdings Ltd\nPartie défenderesse: Sample Partners LP\nDossier:"
    got = score_forms(PARTIES, above)
    assert [e.kind for e in got.events] == []
    assert got.score == score_forms(PARTIES, key_value).score == 1.0


def test_value_inside_the_bracket_its_caption_closes():
    ref = _ref(_field("Provincial Division", "AB"), _field("rand", "991,183"))
    md = 'Lower Court ( AB Provincial Division) ("the Clerk") for the due\nthe sum of R 896,115 (991,183 rand); and'
    assert score_forms(ref, md).score == 1.0


def test_wrong_value_above_its_caption_is_not_read_as_right():
    md = "Placeholder Trust\nPartie demanderesse\n\nc.\n\nSample Partners LP\nPartie défenderesse\n\nDossier"
    got = score_forms(PARTIES, md)
    assert [(e.label, e.cls) for e in got.events] == [("Partie demanderesse", "major")]


def test_swapped_values_under_their_captions_stay_wrong():
    # "Rosa" is the next-line slot of "Last name": it is not also read as the value above "First name"
    ref = _ref(_field("Last name", "Marquez"), _field("First name", "Rosa"))
    got = score_forms(ref, "Last name\nRosa\nFirst name\nMarquez")
    assert sorted(e.label for e in got.events) == ["First name", "Last name"]


def test_printed_line_above_a_blank_field_does_not_excuse_an_invented_value():
    ref = _ref(_field("Name", "Ada"), _field("Remarks", None))
    got = score_forms(ref, "Name: Ada\n\nSee the notes overleaf before you sign\nRemarks: 20")
    assert [(e.label, e.cls) for e in got.events] == [("Remarks", "critical")]


COSTS = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "section", "label": "If yes, interest from", "items": []},
                {"type": "field", "field": "text", "label": "(day)", "value": "2009-05-01"},
                {"type": "field", "field": "text", "label": "(event)", "isEmpty": True},
            ],
        }
    ]
)


def _costs(md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(COSTS, md).events if e.charged]


def test_a_value_after_a_box_on_the_line_above_its_caption_is_read():
    assert _costs("If yes, interest from:\n[x] 2009-05-01\n(day)\n\n(event)\n") == []


def test_a_wrong_value_after_a_box_above_its_caption_is_still_charged():
    assert ("(day)", "2009-05-01") in _costs("If yes, interest from:\n[x] 2009-06-01\n(day)\n\n(event)\n")


PASSES = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "field", "field": "text", "label": "Principal passes-C", "value": "3"},
                {"type": "field", "field": "text", "label": "Subsidiary passes-R", "value": "2"},
                {"type": "field", "field": "text", "label": "Mode of study", "value": "Full-Time"},
            ],
        }
    ]
)
PASSES_MD = (
    '**Mode of study:** Full-Time\n\n<table><tr><td rowspan="2">Number of passes</td><td>{c}</td>'
    "<td>Principal passes-C</td></tr><tr><td>2</td><td>Subsidiary passes-R</td></tr></table>\n"
)


def test_a_value_in_the_cell_left_of_its_caption_is_read():
    assert [e for e in score_forms(PASSES, PASSES_MD.format(c="3")).events if e.charged] == []


def test_a_wrong_value_left_of_its_caption_is_still_charged():
    charged = [(e.label, e.expected) for e in score_forms(PASSES, PASSES_MD.format(c="5")).events if e.charged]
    assert ("Principal passes-C", "3") in charged


def test_a_row_number_left_of_a_caption_with_its_own_value_does_not_mask_it():
    md = "**Mode of study:** Full-Time\n\n| 3 | Principal passes-C: 5 |\n|---|---|\n| 2 | Subsidiary passes-R |\n"
    charged = [(e.label, e.expected) for e in score_forms(PASSES, md).events if e.charged]
    assert ("Principal passes-C", "3") in charged


SIGNER = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "field", "field": "text", "label": "Name (please print)", "value": "Noor Okafor"},
                {"type": "field", "field": "text", "label": "Date", "value": "01/02/2020"},
            ],
        }
    ]
)


def test_a_value_before_its_caption_inside_a_table_cell_is_read():
    md = "| 12 | Noor Okafor **Name (please print)** |\n|---|---|\n| 13 | **Date:** 01/02/2020 |\n"
    assert [e for e in score_forms(SIGNER, md).events if e.charged] == []


def test_a_wrong_value_before_its_caption_inside_a_table_cell_is_still_charged():
    md = "| 12 | Nora Okafor **Name (please print)** |\n|---|---|\n| 13 | **Date:** 01/02/2020 |\n"
    charged = [(e.label, e.expected) for e in score_forms(SIGNER, md).events if e.charged]
    assert ("Name (please print)", "Noor Okafor") in charged


QUANTITY = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {"type": "field", "field": "text", "label": "Item", "value": "Bolts"},
                {"type": "field", "field": "text", "id": "12", "label": "Quantity", "value": "12"},
            ],
        }
    ]
)


def test_a_printed_id_before_its_caption_in_a_table_cell_is_not_the_value():
    md = "| Item | Bolts |\n|---|---|\n| 12 Quantity | |\n"
    assert ("Quantity", "12") in [(e.label, e.expected) for e in score_forms(QUANTITY, md).events if e.charged]


def test_a_value_equal_to_the_printed_id_in_the_cell_to_the_left_is_read():
    md = "| Item | Bolts |\n|---|---|\n| 12 | Quantity |\n"
    assert [e.kind for e in score_forms(QUANTITY, md).events if e.charged and e.lane == "filled"] == []
