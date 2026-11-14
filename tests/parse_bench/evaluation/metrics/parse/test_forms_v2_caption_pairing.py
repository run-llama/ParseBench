"""forms_v2: which caption occurrence a field pairs with, and where its value is read from (a caption printed
twice, a value block under a caption alone on its line, empty spacer cells between a caption and its value)."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, render_reference, score_forms


def _page(*items: dict[str, Any]) -> Any:
    return reference_from_tree([{"page_number": 1, "items": list(items)}])


def _text(label: str, value: str | None = None, pid: str | None = None) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "field", "field": "text", "label": label}
    node.update({"isEmpty": True} if value is None else {"value": value})
    if pid is not None:
        node["id"] = pid
    return node


def _charged(ref: Any, md: str) -> list[tuple[str, str, str]]:
    return [(e.label, e.kind, e.cls) for e in score_forms(ref, md).events if e.charged]


# ---- a caption printed twice: the form title and the field line ----

NEC = _page(
    _text("PAYER'S TIN", "00-9584744"),
    _text("RECIPIENT'S name", "Brian Williams"),
    _text("Street address (including apt. no.)", "820 Glenn Forges"),
    _text("Nonemployee compensation", "98,272.00", pid="1"),
)


def test_reference_renders_to_one():
    assert score_forms(NEC, render_reference(NEC)).score == 1.0


def test_field_line_wins_over_a_title_printing_the_same_caption():
    # the field line sits out of reference order, so the caption is placed after the reading-order chain
    # bug: the bold title earlier on the page took the caption, so the value read as lost and the id as dropped
    md = (
        "**Nonemployee Compensation**\n\n**PAYER'S TIN:** 00-9584744\n\n"
        "**1 Nonemployee compensation:** $ 98,272.00\n\n**RECIPIENT'S name:** Brian Williams\n\n"
        "**Street address (including apt. no.):** 820 Glenn Forges\n"
    )
    assert _charged(NEC, md) == []


def test_wrong_value_on_the_field_line_stays_critical_with_a_title_above():
    md = (
        "**Nonemployee Compensation**\n\n**PAYER'S TIN:** 00-9584744\n\n"
        "**1 Nonemployee compensation:** $ 98,722.00\n\n**RECIPIENT'S name:** Brian Williams\n\n"
        "**Street address (including apt. no.):** 820 Glenn Forges\n"
    )
    assert _charged(NEC, md) == [("Nonemployee compensation", "wrong value", "critical")]


# page text with every caption on its own line, then a list of caption: value lines (each caption printed twice)

W2_NAME = _page(
    _text("Employee's first name and initial", "Paula", pid="e"),
    _text("Last name", "Stanley"),
    _text("Suff."),
    _text("Employee's address and ZIP code", "16244 Jerry Square", pid="f"),
)


def test_blank_caption_line_followed_by_a_caption_keeps_its_field():
    # bug: "Suff." took the tail of the merged list caption and read its value, an invented value (critical)
    md = (
        "e Employee's first name and initial\n\nSuff.\n\nLast name\n\nPaula\n\nStanley\n\n16244 Jerry Square\n\n"
        "f Employee's address and ZIP code\n\n"
        "- **e Employee's first name and initial Last name Suff.**: Paula Stanley\n"
    )
    assert "Suff." not in [label for label, _, _ in _charged(W2_NAME, md)]


INJURY = _page(
    _text("Injured Worker Name", "Northfield Trust"),
    _text("Date of Injury"),
    _text("Time"),
    _text("Description of Injury", "Fell from ladder"),
)


def test_blank_caption_line_wins_over_the_caption_word_inside_a_sentence():
    # bug: "Time" was read inside "requires time off from work", and the sentence as printed text in its blank
    md = (
        "If an injury requires time off from work, report it to the board.\n\nInjured Worker Name\n\n"
        "Northfield Trust\n\nTime\n\nDate of Injury\n\nDescription of Injury\n\nFell from ladder\n"
    )
    assert _charged(INJURY, md) == []


ENTRY = _page(_text("Arrival date", "2024-03-01"), _text("If Yes, how many?", "42,210"), _text("Port", "Boston"))


def test_field_line_wins_over_its_bare_caption_line_followed_by_a_caption():
    # the list line counts as a field line past the caption's own "?" and its printed id "8a."
    md = (
        "8a. If Yes, how many?\n\nArrival date\n\nPort\n\n2024-03-01\n\nBoston\n\n"
        "- **Arrival date**: 2024-03-01\n- **Port**: Boston\n- **8a. If Yes, how many?**: 42,210\n"
    )
    assert "If Yes, how many?" not in [label for label, _, _ in _charged(ENTRY, md)]


# ---- a value block on the lines after its caption ----

PAYER = _page(
    _text(
        "PAYER'S name, street address, city or town, and telephone no.",
        "Knight-Duncan\n45748 Cynthia Prairie\nNorth Suzanneberg, MD 21205\n(671) 954-5133",
    ),
    _text("OMB No.", "1545-0116"),
)
ADDRESS = "Knight-Duncan\n45748 Cynthia Prairie\nNorth Suzanneberg, MD 21205\n(671) 954-5133\n"


def test_value_block_under_a_caption_alone_on_its_line_is_read_whole():
    # bug: only the first line of the block was read, so the field scored "value lost"
    md = f"**PAYER'S name, street address, city or town, and telephone no.:**\n{ADDRESS}\n**OMB No.:** 1545-0116\n"
    assert _charged(PAYER, md) == []


def test_value_paragraph_starting_on_the_caption_line_is_read_whole():
    # bug: continuation stopped after two lines, so a four-line address scored "value truncated"
    md = f"**PAYER'S name, street address, city or town, and telephone no.:** {ADDRESS}\n**OMB No.:** 1545-0116\n"
    assert _charged(PAYER, md) == []


def test_value_block_with_a_wrong_line_is_not_ok():
    md = (
        "**PAYER'S name, street address, city or town, and telephone no.:**\nKnight-Duncan\n45748 Cynthia Prairie\n"
        "North Suzanneberg, MD 21205\n(671) 954-5135\n\n**OMB No.:** 1545-0116\n"
    )
    assert [c for _, _, c in _charged(PAYER, md)] == ["major"]


def test_value_block_with_a_typo_reads_as_a_typo():
    md = (
        "**PAYER'S name, street address, city or town, and telephone no.:**\nKnight-Duncan\n45748 Cynthia Prairie\n"
        "North Suzaneberg, MD 21205\n(671) 954-5133\n\n**OMB No.:** 1545-0116\n"
    )
    assert [(k, c) for _, k, c in _charged(PAYER, md)] == [("value typo", "minor")]


def test_value_block_does_not_read_past_a_blank_line_into_other_values():
    # over-reading guard: values written as separate paragraphs below a caption list belong to other fields
    ref = _page(_text("Initial", "3"), _text("Your SSN or ITIN", "09279850196"))
    md = "Initial\n\nYour SSN or ITIN\n\n3\n\nIbrahim\n\n09279850196\n"
    assert ("Your SSN or ITIN", "value lost", "major") in _charged(ref, md)


CONTACT = _page(_text("Name"), _text("City", "Dayton"), _text("Notes"))


def test_caption_alone_on_its_line_does_not_swallow_the_next_caption():
    assert _charged(CONTACT, "**Name:**\n**City:** Dayton\n**Notes:**\n") == []


def test_blank_field_alone_on_its_line_does_not_take_the_next_paragraph():
    md = (
        "**Name:**\n**City:** Dayton\n**Notes:**\n\n"
        "This is important tax information and is being furnished to the IRS.\n"
        "If you are required to file a return, a penalty may be imposed.\n"
    )
    assert _charged(CONTACT, md) == []


def test_invented_value_on_a_blank_field_line_stays_critical():
    assert _charged(CONTACT, "**Name:** Ana Lopez\n**City:** Dayton\n**Notes:**\n") == [
        ("Name", "invented value", "critical")
    ]


CITY = _page(_text("City", "Dayton"), _text("Zip", "45402"))
FILLER = "".join(f"filler line number {i} with words\n" for i in range(6))


def test_long_paragraph_ending_with_the_value_on_the_caption_line_stays_wrong():
    # over-reading guard: a paragraph far longer than the value is not read to find the value at its end
    md = f"**City:** Springfield\n{FILLER}Dayton\n**Zip:** 45402\n"
    assert _charged(CITY, md) == [("City", "wrong value", "critical")]


def test_long_paragraph_ending_with_the_value_under_a_caption_alone_stays_lost():
    md = f"**City:**\n{FILLER}Dayton\n\n**Zip:** 45402\n"
    assert _charged(CITY, md) == [("City", "value lost", "major")]


# ---- empty spacer cells between a caption and its value ----

GAINS = _page(
    _text("Net long-term capital gain (loss)", "13,698.72"),
    _text("Qualified dividends"),
)


def _row(*cells: str) -> str:
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def test_value_after_empty_spacer_cells_is_read():
    # bug: the empty cell right of the caption read as the value slot, so the value scored "value lost"
    md = (
        "<table>"
        + _row("9a", "Net long-term capital gain (loss)", "", "13,698.72", "")
        + _row("9b", "Qualified dividends", "", "", "")
        + "</table>"
    )
    assert _charged(GAINS, md) == []


def test_spacer_skip_stops_at_a_cell_holding_other_content():
    md = (
        "<table>"
        + _row("9a", "Net long-term capital gain (loss)", "", "see attached", "13,698.72")
        + _row("9b", "Qualified dividends", "", "", "")
        + "</table>"
    )
    assert [k for _, k, _ in _charged(GAINS, md)] != []


def test_wrong_value_after_spacer_cells_is_not_ok():
    md = (
        "<table>"
        + _row("9a", "Net long-term capital gain (loss)", "", "13,968.72", "")
        + _row("9b", "Qualified dividends", "", "", "")
        + "</table>"
    )
    assert [c for _, _, c in _charged(GAINS, md)] == ["major"]


# ---- a shorter caption whose words a longer caption holds (1099 PAYER and RECIPIENT captions) ----

NESTED = _page(
    _text(
        "PAYER'S name, street address, city or town, state or province, country, ZIP or foreign postal code, "
        "and telephone no.",
        "Acme Corp, 1 Main St, Dover, DE 19901",
    ),
    _text("RECIPIENT'S name", "Ana Diaz"),
    _text("Street address (including apt. no.)", "5 Elm Rd"),
    _text("City or town, state or province, country, and ZIP or foreign postal code", "Lake Gary, WV 26428"),
    _text("Total ordinary dividends", "4,927.48", pid="1a"),
    _text("Qualified dividends", "1,200.00", pid="1b"),
    _text("Total capital gain distr.", "310.00", pid="2a"),
    _text("Unrecap. Sec. 1250 gain", "45.00", pid="2b"),
)
# the output prints the boxes before the RECIPIENT block, so the reading-order chain can't take both
NESTED_MD = (
    "**PAYER'S name, street address, city or town, state or province, country, ZIP or foreign postal code, "
    "and telephone no.:** Acme Corp, 1 Main St, Dover, DE 19901\n\n"
    "**1a Total ordinary dividends:** 4,927.48\n"
    "**1b Qualified dividends:** 1,200.00\n"
    "**2a Total capital gain distr.:** 310.00\n"
    "**2b Unrecap. Sec. 1250 gain:** 45.00\n\n"
    "**RECIPIENT'S name:** Ana Diaz\n"
    "**Street address (including apt. no.):** 5 Elm Rd\n"
    "**City or town, state or province, country, and ZIP or foreign postal code:** Lake Gary, WV 26428\n"
)


def test_partial_caption_inside_a_longer_caption_does_not_take_its_place():
    """The RECIPIENT city caption matched inside the PAYER caption left PAYER unlocated and the real city a copy."""
    assert _charged(NESTED, NESTED_MD) == []


def test_wrong_city_under_a_longer_caption_holding_its_words_is_critical():
    md = NESTED_MD.replace("Lake Gary, WV 26428", "Springfield, OH 45501")
    assert _charged(NESTED, md) == [
        ("City or town, state or province, country, and ZIP or foreign postal code", "wrong value", "critical")
    ]
