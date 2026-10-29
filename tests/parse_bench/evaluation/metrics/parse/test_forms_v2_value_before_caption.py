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
