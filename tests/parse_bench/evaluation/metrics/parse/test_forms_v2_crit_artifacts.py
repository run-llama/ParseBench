"""forms_v2: output layouts that were charged as critical value errors although the value is right."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _field(label: str, value: str | None) -> dict[str, Any]:
    if value is None:
        return {"type": "field", "field": "text", "label": label, "isEmpty": True}
    return {"type": "field", "field": "text", "label": label, "value": value}


def _ref(*fields: dict[str, Any]):
    return reference_from_tree([{"page_number": 1, "items": list(fields)}])


def _critical(ref, md: str) -> list[str]:
    return [e.label for e in score_forms(ref, md).events if e.cls == "critical"]


def test_value_before_its_caption_on_the_same_line():
    ref = _ref(_field("additional page(s) are attached.", "57"))
    md = "Check this box if more space is needed. **{}** additional page(s) are attached. Mark each page."
    assert score_forms(ref, md.format("57")).score == 1.0
    assert score_forms(ref, md.format("58")).score < 1.0


def test_box_code_before_a_blank_marker_is_no_value():
    ref = _ref(_field("Surface habitable", None), _field("TASA POR RESTAURACIÓN", None))
    md = "**Surface habitable : F22:** [___] m²\n\nTASA POR RESTAURACIÓN . . . **Casilla RP:** [___]\n"
    assert _critical(ref, md) == []
    assert _critical(ref, "**Surface habitable : F22:** 120 m²\n\nTASA POR RESTAURACIÓN: [___]\n") != []


def test_number_printed_in_its_own_caption_is_no_invented_value():
    label = "For the year Jan. 1-Dec. 31, 2025, or other tax year beginning"
    ref = _ref(_field(label, None), _field("ending", None))
    assert _critical(ref, f"{label} , 2025, ending\n") == []
    assert _critical(ref, f"{label} 03/14/2026, ending\n") != []


def _date(dd: str | None, mm: str | None, yy: str | None) -> dict[str, Any]:
    return {"items": [_field("DD", dd), _field("MM", mm), _field("YY", yy)]}


def _name(value: str) -> dict[str, Any]:
    return {"items": [{"type": "field", "field": "text", "value": value}]}


DATES = reference_from_tree(
    [
        {
            "page_number": 1,
            "items": [
                {
                    "type": "table",
                    "columns": ["Full name", "Date of birth (DD/MM/YY)", "Country of residence"],
                    "rows": [
                        [_name("Ada Lovelace"), _date("A", "D", "2"), _name("France")],
                        [_name("Leo Marsh"), _date(None, "2", None), _name("Sweden")],
                    ],
                }
            ],
        }
    ]
)


def test_date_subfields_written_as_a_list_in_one_cell():
    head = "| Full name | Date of birth (DD/MM/YY) | Country of residence |\n|---|---|---|\n"
    md = head + "| Ada Lovelace | - A - D - 2 | France |\n| Leo Marsh | - [___] - 2 - [___] | Sweden |\n"
    res = score_forms(DATES, md)
    assert [e.kind for e in res.events if e.charged] == []


def test_colspan_caption_reads_the_slot_that_holds_its_values():
    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {
                        "type": "table",
                        "columns": ["INSURER(S) AFFORDING COVERAGE", "NAIC #"],
                        "rows": [
                            [_name("Auto-Owners Insurance Company"), _name("37591")],
                            [_name("Selective"), _name("12572")],
                        ],
                    }
                ],
            }
        ]
    )
    md = (
        '<table><tr><th colspan="2">INSURER(S) AFFORDING COVERAGE</th><th>NAIC #</th></tr>'
        "<tr><td>INSURER A :</td><td>Auto-Owners Insurance Company</td><td>37591</td></tr>"
        "<tr><td>INSURER B :</td><td>Selective</td><td>12572</td></tr></table>"
    )
    assert _critical(ref, md) == []


def test_label_tail_with_a_worded_number_before_a_blank_marker():
    ref = _ref(_field("Any other service stated in item 13.1", None), _field("Name & signature", None))
    md = "- Any other service stated in item 13.1 See special terms (page 12): [___]\n\n"
    md += "**Name & signature (Note 2):** [___]\n"
    assert _critical(ref, md) == []
    assert _critical(ref, md.replace("(Note 2):** [___]", "(Note 2):** Ann Lee")) != []


def test_cross_reference_after_a_blank_marker_is_printed_text():
    ref = _ref(_field("Number of persons allowed to reside at the premises", None))
    md = "**Item 16** **Number of persons allowed to reside at the premises** [___] {}\n"
    assert _critical(ref, md.format("See clause 22")) == []
    assert _critical(ref, md.format("22")) != []


def test_sentence_after_a_blank_marker_is_printed_text():
    ref = _ref(_field("I hereby declare that I have today acquired the vehicle and the price of purchase is", None))
    md = (
        "I hereby declare that I have today acquired the vehicle and the price of purchase is $[___]. I also"
        " declare that I am not less than 18 years old and hereby verify the accuracy of the information.\n"
    )
    assert _critical(ref, md) == []


def test_bracketed_caption_after_its_blank_marker_reads_the_marker():
    ref = _ref(_field("(Name of Principal)", None))
    assert _critical(ref, "[___] (Name of Principal) Corporate (Seal)\n") == []
    assert _critical(ref, "(Name of Principal) Corporate (Seal)\n") != []


def test_printed_century_in_a_blank_date_field():
    ref = _ref(_field("Date (yr, mon, day)", None))
    assert _critical(ref, "**Date (yr, mon, day):** 20\n") == []
    assert _critical(ref, "**Date (yr, mon, day):** 2024\n") != []


def test_value_written_between_the_halves_of_its_caption():
    ref = _ref(_field("am authorized by (name of the Registered Distributor)", "Tobias Novak"))
    md = "I ______ (name), am authorized by {} (name of the Registered Distributor), the Registered Distributor.\n"
    assert _critical(ref, md.format("Tobias Novak")) == []
    assert _critical(ref, md.format("Peter Smith")) != []


def test_value_inside_the_bracket_after_its_caption():
    ref = _ref(_field("High Court", "47,282.22"))
    md = "appointed by the Master of the High Court ({} Division) to administer the above estate and the assets\n"
    assert _critical(ref, md.format("47,282.22")) == []
    assert _critical(ref, md.format("12,000.00")) != []


def test_narrower_output_table_maps_columns_by_their_values():
    rows = [
        ["Filing fee", None, None, None, "3,249"],
        ["Report writing", None, "Preparation of testimony", None, None],
        ["Total", None, None, None, "86,746"],
    ]
    ref = reference_from_tree([{"page_number": 1, "items": [{"type": "table", "columns": [""] * 5, "rows": rows}]}])
    md = "| Filing fee |  | 3,249 |\n|---|---|---|\n"
    md += "| Report writing | Preparation of testimony |  |\n| Total |  | 86,746 |\n"
    assert _critical(ref, md) == []
    assert _critical(ref, md.replace("86,746", "86,745")) != []


def test_rising_line_codes_before_the_value_column_are_printed():
    ref = _ref(
        _field("Basic personal amount", "75,031"),
        _field("Spouse amount", None),
        _field("Eligible dependant", "62,386.90"),
        _field("Caregiver amount", None),
    )
    rows = [("Basic personal amount", "30000", "75,031"), ("Spouse amount", "30300", "")]
    rows += [("Eligible dependant", "30400", "62,386.90"), ("Caregiver amount", "30450", "")]
    md = "| | | | |\n|---|---|---|---|\n" + "".join(f"| {a} | {b} | + | {c} |\n" for a, b, c in rows)
    assert _critical(ref, md) == []
    assert _critical(ref, md.replace("62,386.90", "62,386.99")) != []


def test_line_code_before_a_colon_is_no_part_of_the_value():
    ref = _ref(_field("Federal tuition amount transferred", "797,141"))
    assert _critical(ref, "**Federal tuition amount transferred 32700:** 797,141 **24**\n") == []
    assert _critical(ref, "**Federal tuition amount transferred 32700:** 797,147 **24**\n") != []


def test_caption_inside_another_columns_header_gives_way():
    cols = ["Currency", "Class of shares", "Number of shares", "Aggregate nominal value"]
    ref = reference_from_tree(
        [
            {
                "page_number": 1,
                "items": [
                    {
                        "type": "table",
                        "columns": cols,
                        "rows": [[_name("GBP"), _name("Ordinary"), _name("100"), _name("50")]],
                    },
                    {
                        "type": "table",
                        "columns": cols,
                        "rows": [[_name("EUR"), _name("Preference"), _name("7"), _name("9")]],
                    },
                ],
            }
        ]
    )
    md = (
        "| Currency | Class of shares | Number of shares | Aggregate nominal value Number of shares issued"
        " multiplied by nominal value |\n|---|---|---|---|\n"
        "| GBP | Ordinary | 100 | 50 |\n| EUR | Preference | 7 | 9 |\n"
    )
    assert _critical(ref, md) == []
    assert _critical(ref, md.replace("| 7 |", "| 8 |")) != []


def test_date_parts_written_between_their_unit_captions():
    ref = _ref(_field("Date de naissance Année Mois Jour", "1997-08-22"))
    assert _critical(ref, "**19. Date de naissance:** Année 1997 / Mois 08 / Jour 22\n") == []
    assert _critical(ref, "**19. Date de naissance:** Année 1997 / Mois 08 / Jour 23\n") != []


def test_blank_marker_between_the_words_of_its_caption():
    ref = _ref(_field("Hourly $ per hour", None), _field("hours per week", "40"))
    md = "If regular salary: Hourly $ [___] **per hour** 40 **hours per week;**\n"
    assert _critical(ref, md) == []
    assert _critical(ref, md.replace("[___]", "")) != []


def test_subfields_under_a_colspan_caption_read_one_slot_each():
    md = (
        '<table><tr><th>Full name</th><th colspan="3">Date of birth (DD/MM/YY)</th><th>Country of residence</th></tr>'
        "<tr><td>Ada Lovelace</td><td>A</td><td>D</td><td>2</td><td>France</td></tr>"
        "<tr><td>Leo Marsh</td><td></td><td>2</td><td></td><td>Sweden</td></tr></table>"
    )
    assert _critical(DATES, md) == []
    assert _critical(DATES, md.replace("<td>2</td><td>France", "<td>5</td><td>France")) != []


def test_marker_before_a_wrapped_caption_is_read_from_the_caption_start_line():
    ref = _ref(_field("Name of Principal", None))
    assert _critical(ref, "Applicant Name of\nxxxx ___ (Principal) Ana Lopez\n") != []
