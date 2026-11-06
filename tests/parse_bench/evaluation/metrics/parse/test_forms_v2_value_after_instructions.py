"""forms_v2: a value written after the printed instructions or list that follow its caption."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _ref(*fields: tuple[str, str | None]):
    items = [
        {"type": "field", "field": "text", "label": k, **({"value": v} if v else {"isEmpty": True})} for k, v in fields
    ]
    return reference_from_tree([{"page_number": 1, "items": items}])


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]


SPOUSE = _ref(("Spouse amount", "2,721"), ("Dependant amount", None))
SPOUSE_MD = (
    "**7. Spouse amount** - Enter the difference between line 1 and the net income of your spouse for the year "
    "if both conditions are met:\n\n- you support your spouse who lives with you;\n- the net income of your spouse "
    "is below line 1.\n\nIn all cases, go to line 9 if your spouse has an impairment. **Amount:** {v}\n\n"
    "**8. Dependant amount** - Enter the difference between line 1 and the net income of your dependant.\n"
)


def test_a_value_after_a_list_of_instructions_under_its_caption_is_read():
    assert _charged(SPOUSE, SPOUSE_MD.format(v="2,721")) == []


def test_a_wrong_value_after_a_list_of_instructions_is_still_charged():
    assert ("Spouse amount", "2,721") in _charged(SPOUSE, SPOUSE_MD.format(v="9,999"))


RENT = _ref(("Day of last rent increase", "14"), ("Rent payable", None))
RENT_MD = (
    "Day of last rent increase\nInsert the day the rent was last increased for the premises\n\n{v}\n\nRent payable\n"
)


def test_a_value_after_an_instruction_line_under_its_caption_is_read():
    assert _charged(RENT, RENT_MD.format(v="14")) == []


def test_a_wrong_value_after_an_instruction_line_is_still_charged():
    assert ("Day of last rent increase", "14") in _charged(RENT, RENT_MD.format(v="21"))


def test_a_value_past_an_unlocated_caption_line_is_not_credited():
    twin = _ref(("Day of last rent increase", "14"), ("Rent payable", "14"))
    md = "Day of last rent increase\nInsert the day the rent was last increased for the premises\nWeekly sum\n14\n"
    assert ("Day of last rent increase", "14") in _charged(twin, md)


def test_a_time_after_a_label_colon_is_read_whole():
    start = _ref(("Start time", "08:30"))
    md = (
        "Start time - Follow the printed instructions for entering the correct start time below:\n\n**Amount:** 08:30\n"
    )
    assert ("Start time", "08:30") not in _charged(start, md)


def test_a_value_after_a_short_instruction_on_the_caption_line_is_read():
    spouse = _ref(("Spouse amount", "2,721"))
    assert _charged(spouse, "**Spouse amount** - Enter your spouse amount here:\n\n**Amount:** 2,721\n") == []


def test_a_value_right_after_an_unspaced_label_colon_is_read():
    md = "Spouse amount - Enter the difference between your net income and your spouse income if both apply:\n\n"
    assert _charged(_ref(("Spouse amount", "2,721")), md + "Amount:2,721\n") == []
    assert ("Start time", "08:30") not in _charged(
        _ref(("Start time", "08:30")), md.replace("Spouse amount", "Start time") + "Time:08:30\n"
    )
