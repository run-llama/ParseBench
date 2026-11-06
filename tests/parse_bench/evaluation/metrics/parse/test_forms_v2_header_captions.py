"""forms_v2: captions that share a table cell with other printed text, and grids written without a delimiter row."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.forms_v2 import reference_from_tree, score_forms


def _ref(*fields: tuple[str, str | None]):
    items = [
        {"type": "field", "field": "text", "label": k, **({"value": v} if v else {"isEmpty": True})} for k, v in fields
    ]
    return reference_from_tree([{"page_number": 1, "items": items}])


def _charged(ref, md: str) -> list[tuple[str, str]]:
    return [(e.label, e.expected) for e in score_forms(ref, md).events if e.charged]


CLAIM = _ref(("Main claim", "12 Oak Rd"), ("Interest", "300"))
CLAIM_MD = (
    "<table><tr><td colspan='3'>7.1.1. Main claim:</td></tr><tr><td colspan='3'>{v}</td></tr>"
    "<tr><td>Interest</td><td>300</td><td></td></tr></table>"
)


def test_a_dotted_section_number_before_a_caption_still_opens_its_cell():
    assert _charged(CLAIM, CLAIM_MD.format(v="12 Oak Rd")) == []


def test_a_wrong_value_under_a_numbered_caption_is_still_charged():
    assert ("Main claim", "12 Oak Rd") in _charged(CLAIM, CLAIM_MD.format(v="9 Elm St"))


PLACE = _ref(("Province", None), ("Postal code", "K1A 0B1"))
PLACE_MD = "<table><tr><td>City</td><td>Province Postal code</td></tr><tr><td>Ottawa</td><td>{v}</td></tr></table>"


def test_a_caption_after_another_caption_in_its_cell_reads_the_cell_below():
    assert _charged(PLACE, PLACE_MD.format(v="K1A 0B1")) == []


def test_a_wrong_value_under_merged_captions_is_still_charged():
    assert ("Postal code", "K1A 0B1") in _charged(PLACE, PLACE_MD.format(v="K9Z 9Z9"))


CREDIT = _ref(("x 50% =", "8020"))
CREDIT_MD = (
    "<table><tr><td>15</td><td>Blind worker credit. Qualifying expenses 4,000 .00 x 50% =</td><td>{v}</td></tr></table>"
)


def test_a_caption_ending_a_prose_cell_reads_a_matching_value_to_its_right():
    assert _charged(CREDIT, CREDIT_MD.format(v="8020 .00")) == []


def test_a_wrong_value_right_of_a_caption_ending_a_prose_cell_is_still_charged():
    assert ("x 50% =", "8020") in _charged(CREDIT, CREDIT_MD.format(v="9999 .00"))


WORK = _ref(("Activity", None), ("from", "01.03.2024"), ("until", None))
WORK_MD = "Activity | from | until\n\n- [ ] Minor job | {v} | [___]\n- [ ] Other work | [___] | [___]\n"


def test_a_pipe_grid_without_a_delimiter_row_is_read_as_a_table():
    assert _charged(WORK, WORK_MD.format(v="01.03.2024")) == []


def test_a_wrong_value_in_a_pipe_grid_without_a_delimiter_row_is_still_charged():
    assert ("from", "01.03.2024") in _charged(WORK, WORK_MD.format(v="05.07.2023"))


def test_a_value_right_of_merged_captions_is_not_charged_to_the_blank_second_caption():
    ref = _ref(("Province", "Ontario"), ("Postal code", None))
    md = "<table><tr><td>Province Postal code</td><td>Ontario</td></tr></table>"
    assert not [e for e in score_forms(ref, md).events if e.charged and e.cls == "critical"]


def test_a_delimiter_row_in_a_grid_without_outer_pipes_is_not_a_row():
    md = "Activity | from | until\n--- | --- | ---\nMinor job | 01.03.2024 | [___]\nOther work | [___] | [___]\n"
    assert ("from", "01.03.2024") not in _charged(WORK, md)
