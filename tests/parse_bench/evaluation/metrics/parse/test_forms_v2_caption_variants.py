"""forms_v2: a caption printed in one language of a bilingual label, or without its date-format hint."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _ref(*fields: tuple[str, str, object]):
    items = [
        {"type": "field", "field": f, "label": k, **({"value": v} if v not in (None, "") else {"isEmpty": True})}
        for f, k, v in fields
    ]
    return reference_from_tree([{"page_number": 1, "items": items}])


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]


INSURANCE = _ref(
    ("text", "保險單持有人姓名或名稱 Name of Policy Holder", None),
    ("text", "保險公司（並非代理）的名稱 Name of Insurance Company (not agent)", "Harbour Freight Co"),
    ("text", "屆滿日期 Expiry Date", "14/03/2001"),
)
INSURANCE_MD = (
    "**VALID THIRD PARTY RISKS INSURANCE**\n\n**Name of Policy Holder:** [___]\n"
    "**Name of Insurance Company (not agent):** {company}\n**Expiry Date:** {date}\n"
)


def test_a_caption_printed_in_english_only_reads_its_value():
    assert _charged(INSURANCE, INSURANCE_MD.format(company="Harbour Freight Co", date="14/03/2001")) == []


def test_a_wrong_value_under_an_english_only_caption_is_still_charged():
    charged = _charged(INSURANCE, INSURANCE_MD.format(company="Harbour Freight Co", date="14/03/2011"))
    assert ("屆滿日期 Expiry Date", "14/03/2001") in charged


STATUS = _ref(
    ("checkbox", "單身 Single", False),
    ("checkbox", "已婚 Married", True),
    ("text", "出生地點 Place of birth", "Lisbon"),
)


def test_an_option_printed_in_english_only_after_its_box_is_read():
    assert (
        _charged(STATUS, "**婚姻狀況 Marital status:** [ ] Single [x] Married\n**出生地點 Place of birth:** Lisbon\n")
        == []
    )


def test_an_unticked_english_only_option_is_still_charged():
    charged = _charged(
        STATUS, "**婚姻狀況 Marital status:** [ ] Single [ ] Married\n**出生地點 Place of birth:** Lisbon\n"
    )
    assert ("已婚 Married", "[x]") in charged


INJURY = _ref(("text", "Time of injury", "14:20"), ("text", "Date: M D Y", "1975-05-19"), ("text", "Witness", None))


def test_a_date_caption_printed_without_its_format_hint_reads_its_value():
    assert _charged(INJURY, "**Time of injury:** 14:20\n**Date:** 1975-05-19\n**Witness:** [___]\n") == []


def test_a_wrong_date_under_a_caption_without_its_hint_is_still_charged():
    charged = _charged(INJURY, "**Time of injury:** 14:20\n**Date:** 1975-06-19\n**Witness:** [___]\n")
    assert ("Date: M D Y", "1975-05-19") in charged


PARTIES = _ref(
    ("text", "Nom de l'autorité : Name of the authority", None),
    ("text", "Identité des parties : Particulars of the parties", "Harbour Freight Co\n12 Elm Rd"),
    ("text", "Nature de l'acte : Nature of the document", None),
)
PARTIES_MD = (
    "**Nom de l'autorité :** [___]\nName of the authority: [___]\n\n"
    "**Identité des parties :** {v}\n12 Elm Rd\nParticulars of the parties: (see above)\n\n"
    "**Nature de l'acte :** [___]\nNature of the document: [___]\n"
)


def test_a_value_between_the_two_language_lines_of_its_caption_is_read():
    assert _charged(PARTIES, PARTIES_MD.format(v="Harbour Freight Co")) == []


def test_a_wrong_value_between_the_two_language_lines_is_still_charged():
    charged = _charged(PARTIES, PARTIES_MD.format(v="Harbor Logistics Ltd"))
    assert ("Identité des parties : Particulars of the parties", "Harbour Freight Co\n12 Elm Rd") in charged


POLICY = _ref(
    ("text", "保險單持有人姓名 Name of Policy Holder", "Ada Moss"),
    ("text", "保單號碼 Policy Number", "AB123"),
    ("text", "屆滿日期 Expiry Date", "14/03/2001"),
)


def test_a_half_caption_prefers_the_field_line_over_a_heading():
    md = (
        "**Name of Policy Holder:** Ada Moss\n\n**Policy Number**\n\n"
        "**Policy Number:** AB123\n**Expiry Date:** 14/03/2001\n"
    )
    assert _charged(POLICY, md) == []


AUTHORITY = _ref(
    ("text", "Date", "02/02/2002"),
    ("text", "Nom de l'autorité : Name of the authority", "Alice Moss"),
    ("text", "Lieu : Place", "Lyon"),
)


def test_a_first_language_line_holding_an_instruction_keeps_the_caption_on_the_second_line():
    md = (
        "**Date:** 02/02/2002\n\n**Nom de l'autorité :** Please print clearly\n"
        "Name of the authority: Alice Moss\n\n**Lieu :** Lyon\nPlace: Lyon\n"
    )
    assert _charged(AUTHORITY, md) == []


def test_a_bare_see_above_note_on_the_second_language_line_reads_the_value_above():
    md = PARTIES_MD.format(v="Harbour Freight Co").replace("(see above)", "see above")
    assert _charged(PARTIES, md) == []
