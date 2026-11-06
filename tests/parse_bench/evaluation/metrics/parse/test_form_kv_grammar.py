"""Grammar pins for `form_kv_scoring`: each guide construct, what is deliberately unparsed, and the
lossless render -> parse round trip ground truth relies on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parse_bench.evaluation.metrics.parse.form_kv_scoring import (
    COVERAGE_SPOT,
    SELECT_KIND,
    KvRecord,
    pages_from_tree,
    parse_kv_markdown,
    render_guide,
    tree_units,
)

FX = Path(__file__).parent / "fixtures" / "form_kv"


def parse(md: str) -> list[KvRecord]:
    return parse_kv_markdown(md).records


def groups(records: list[KvRecord]) -> list[int]:
    seen: dict[int, int] = {}
    return [seen.setdefault(r.parent, len(seen)) for r in records]


def lines(*xs: str) -> str:
    return "\n".join(xs)


def test_simple_field_line() -> None:
    (r,) = parse("- \\[1a\\] **Total amount**: 4,200")
    assert (r.id, r.label, r.kind, r.value) == ("1a", "Total amount", "text", "4,200")
    assert r.raw_label == "Total amount" and r.raw_value == "4,200" and r.source_line == 1


def test_escaped_id_is_an_id_and_bare_brackets_are_a_tick() -> None:
    printed_x, single, married = parse(lines("- \\[x\\] **Label**: 1", "- [x] Single", "- [ ] Married"))
    assert (printed_x.id, printed_x.kind, printed_x.value) == ("x", "text", "1")
    assert (single.id, single.kind, single.value, single.label) == (None, "checkbox", True, "Single")
    assert (married.kind, married.value, married.label) == ("checkbox", False, "Married")


def test_unescaped_brackets_are_not_ids() -> None:
    (r,) = parse("- [a] Employee's social security number: 920-24-6595")
    assert (r.id, r.label, r.value) == (None, None, "[a] Employee's social security number: 920-24-6595")


def test_blank_tokens_and_bare_values() -> None:
    tips, nine, bare_blank, bare_value, empty_bullet = parse(
        lines("- \\[7\\] **Social security tips**: [___]", "- \\[9\\]: [___]", "- [___]", "- 4,200", "- ")
    )
    assert (tips.id, tips.label, tips.value) == ("7", "Social security tips", None)
    assert (nine.id, nine.label, nine.value) == ("9", None, None)
    assert (bare_blank.id, bare_blank.label, bare_blank.value) == (None, None, None)
    assert (bare_value.label, bare_value.value) == (None, "4,200")
    assert (empty_bullet.label, empty_bullet.value, empty_bullet.kind) == (None, None, "text")


def test_blank_marker_and_older_underscore_spelling_are_both_blanks() -> None:
    new, old, two_on_a_line = parse(
        lines("- **From**: [blank]", "- **To**: [___]", "- **Period**: from [blank] to [blank]")
    )
    assert (new.label, new.value) == ("From", None)
    assert (old.label, old.value) == ("To", None)
    assert two_on_a_line.value == "from [blank] to [blank]"


def test_colon_inside_the_bold_label_stays_in_the_label() -> None:
    fax, contact = parse(lines("- **FAX (A/C, No):** [___]", "- **CONTACT NAME:** Melinda Clayton"))
    assert (fax.label, fax.value) == ("FAX (A/C, No):", None)
    assert (contact.label, contact.value) == ("CONTACT NAME:", "Melinda Clayton")


def test_checkbox_field_and_write_in() -> None:
    fatca, other_box, other_text, blank_box, blank_text = parse(
        lines("- **FATCA filing requirement**: [x]", "- **Other (specify):** [x] consulting", "- **X**: [ ] [___]")
    )
    assert (fatca.kind, fatca.value, fatca.label) == ("checkbox", True, "FATCA filing requirement")
    assert (other_box.kind, other_box.value, other_box.label) == ("checkbox", True, "Other (specify):")
    assert (other_text.kind, other_text.value, other_text.label) == ("text", "consulting", "Other (specify):")
    assert (blank_box.kind, blank_box.value) == ("checkbox", False)
    assert (blank_text.kind, blank_text.value, blank_text.label) == ("text", None, "X")


def test_signature_tokens() -> None:
    named, bare, blank = parse(
        lines("- **Your signature**: [signature: John Doe]", "- **Spouse**: [signature]", "- **Date**: [___]")
    )
    assert (named.kind, named.value, named.signer) == ("signature", True, "John Doe")
    assert (bare.kind, bare.value, bare.signer) == ("signature", True, None)
    assert (blank.kind, blank.value) == ("text", None)


def test_heading_opens_a_section_scope_until_the_next_heading() -> None:
    recs = parse(
        lines(
            "### \\[Part II\\] Information About the Partner",
            "",
            "- \\[E\\] **Partner's SSN or TIN**: 27-4162256",
            "- \\[I1\\] **What type of entity is this partner?**: CORPORATION",
            "",
            "### POLICY INFORMATION",
            "",
            "- **PROPOSED EFF DATE**: 08/23/2026",
            "",
            "# Form W-2 Wage and Tax Statement",
        )
    )
    assert [r.path for r in recs] == [("Part II",), ("Part II",), ("POLICY INFORMATION",)]
    assert groups(recs) == [0, 0, 1]
    assert [r.value for r in recs] == ["27-4162256", "CORPORATION", "08/23/2026"]


def test_small_captioned_boxes_nest() -> None:
    recs = parse(
        lines(
            "- \\[e\\]",
            "  - **Employee's first name and initial**: Joan Jackson",
            "  - **Last name**: [___]",
            "  - **Suff.**: [___]",
            "- \\[12a\\]",
            "  - **Code**: D",
            "  - 1,500.00",
            "- \\[15\\] **State**",
            "  - TX",
            "  - **Employer's state ID number**: 00-0000056",
        )
    )
    assert [r.path for r in recs] == [("e",)] * 3 + [("12a",)] * 2 + [("15",)] * 2
    assert groups(recs) == [0, 0, 0, 1, 1, 2, 2]
    assert [(r.label, r.value) for r in recs[3:5]] == [("Code", "D"), (None, "1,500.00")]
    assert [(r.label, r.value) for r in recs[5:]] == [(None, "TX"), ("Employer's state ID number", "00-0000056")]


def test_labeled_section_as_nested_parent_item() -> None:
    recs = parse(lines("- \\[Part I\\] **Information About the Partnership**", "  - \\[A\\] **EIN**: 27-4162308"))
    assert [(r.path, r.id, r.value) for r in recs] == [(("Part I",), "A", "27-4162308")]


def test_option_groups() -> None:
    recs = parse(
        lines(
            "- **Filing Status**",
            "  - [x] Single",
            "  - [ ] Married filing jointly",
            "- \\[13\\]",
            "  - [x] Statutory employee",
            "  - [ ] Retirement plan",
            "- \\[16\\] **Tax (see instructions). Check if any from Form(s):**",
            "  - \\[1\\] [ ] 8814",
            "  - \\[2\\] [ ] 4972",
            "  - \\[3\\] [x] 8888",
            "  - \\[4\\] [ ] [___]",
            "- \\[16\\]: 0.00",
        )
    )
    filing, box13, line16, amount = recs
    assert filing.kind == SELECT_KIND and filing.label == "Filing Status" and filing.id is None
    assert [(o["label"], o["value"]) for o in filing.options] == [("Single", True), ("Married filing jointly", False)]
    assert (box13.id, box13.label) == ("13", None)
    assert [(o["label"], o["value"]) for o in box13.options] == [
        ("Statutory employee", True),
        ("Retirement plan", False),
    ]
    assert (line16.id, line16.label) == ("16", "Tax (see instructions). Check if any from Form(s):")
    assert [(o["id"], o["label"], o["value"]) for o in line16.options] == [
        ("1", "8814", False),
        ("2", "4972", False),
        ("3", "8888", True),
        ("4", None, False),  # a blank write-in is not a label
    ]
    assert (amount.id, amount.kind, amount.value) == ("16", "text", "0.00")


def test_html_table_indented_under_its_key() -> None:
    recs = parse(
        lines(
            "- \\[J\\] **Partner's share of profit, loss, and capital (see instructions):**",
            "",
            "  <table>",
            "    <tr><th></th><th>Beginning</th><th>Ending</th></tr>",
            "    <tr><td>Profit</td><td>1.0000000000 %</td><td>1.0000000000 %</td></tr>",
            "    <tr><td>Loss</td><td>[___]</td><td></td></tr>",
            "  </table>",
            "",
            "- **Check if decrease is due to sale or exchange of partnership interest.**: [ ]",
        )
    )
    cells, tail = recs[:-1], recs[-1]
    assert [(r.path, r.label, r.value) for r in cells] == [
        (("J",), None, "Profit"),
        (("J",), "Beginning", "1.0000000000 %"),
        (("J",), "Ending", "1.0000000000 %"),
        (("J",), None, "Loss"),
        (("J",), "Beginning", None),
        (("J",), "Ending", None),
    ]
    assert groups(cells) == [0, 0, 0, 1, 1, 1]  # one parent identity per row
    assert (tail.path, tail.kind, tail.value) == ((), "checkbox", False)


def test_html_table_flush_left_is_anonymous_and_cells_use_form_notation() -> None:
    recs = parse(
        lines(
            "- **Before**: 1",
            "<table>",
            "<tr><th>Name</th><th>Credit</th><th>Other</th><th>Both</th></tr>",
            "<tr><td>Jane</td><td>- [x] Child tax credit</td><td>[ ]</td><td>- [ ]<br>- [x]</td></tr>",
            "<tr><td></td><td>- [ ]</td><td>- [ ]</td><td></td></tr>",
            "</table>",
        )
    )
    before, *cells = recs
    assert before.value == "1"
    assert [(r.path, r.label, r.kind, r.value) for r in cells] == [
        ((), "Name", "text", "Jane"),
        ((), "Child tax credit", "checkbox", True),
        ((), "Other", "checkbox", False),
        ((), "Both", "checkbox", False),
        ((), "Both", "checkbox", True),
        ((), "Name", "text", None),
        ((), "Credit", "checkbox", False),
        ((), "Other", "checkbox", False),
        ((), "Both", "text", None),
    ]


def test_continuation_lines_extend_the_value_until_a_blank_line() -> None:
    (r,) = parse(
        lines(
            "- \\[c\\] **Employer's name, address, and ZIP code**: Wilson and Sons",
            "  5435 Bradshaw Trail",
            "  South Howardhaven, TX 76177",
            "",
            "Loose text after a blank line is not a continuation.",
        )
    )
    assert r.value == "Wilson and Sons\n5435 Bradshaw Trail\nSouth Howardhaven, TX 76177"
    page = parse_kv_markdown("- **A**: 1\n\nloose")
    assert page.diagnostics["unparsed_lines"] == [3] and page.diagnostics["continuation_lines"] == 0


def test_non_list_lines_are_unparsed_not_normalised() -> None:
    page = parse_kv_markdown(
        lines(
            "**a Employee's social security number**:",
            "",
            "920-24-6595",
            "",
            "[x] Final K-1 [ ] Amended K-1",
            "",
            "Instructions: see separate instructions.",
            "",
            "- \\[b\\] **EIN**: 00-3438130",
        )
    )
    assert [r.id for r in page.records] == ["b"]
    assert page.diagnostics["unparsed_lines"] == [1, 3, 5, 7]
    assert page.diagnostics["bold_colon_unparsed"] == 1


def test_todays_page_markdown_scores_as_list_loss() -> None:
    page = parse_kv_markdown((FX / "w2.p1.page.md").read_text())
    assert all(r.path == () for r in page.records)
    assert {r.label for r in page.records} >= {"15 State", "Employer's state ID number", "20 Locality name"}
    assert page.diagnostics["bold_colon_unparsed"] >= 20


def test_coverage_tag_and_override() -> None:
    assert parse_kv_markdown("- **A**: 1").coverage == "exhaustive"
    assert parse_kv_markdown("<!-- coverage: spot -->\n- **A**: 1").coverage == COVERAGE_SPOT
    assert parse_kv_markdown("<!-- coverage: spot -->\n- **A**: 1", coverage="exhaustive").coverage == "exhaustive"


def test_render_guide_matches_the_guide_examples() -> None:
    line16 = [
        {
            "type": "field",
            "field": "multi_select",
            "id": "16",
            "label": "Tax (see instructions). Check if any from Form(s):",
            "valueItems": [
                {"type": "field", "field": "checkbox", "id": "1", "label": "8814", "value": False},
                {"type": "field", "field": "checkbox", "id": "2", "label": "4972", "value": False},
                {
                    "type": "section",
                    "id": "3",
                    "items": [
                        {"type": "field", "field": "checkbox", "value": False},
                        {"type": "field", "field": "text", "isEmpty": True},
                    ],
                },
            ],
        },
        {"type": "field", "field": "text", "id": "16", "value": "0.00"},
    ]
    assert render_guide(line16) == lines(
        "- \\[16\\] **Tax (see instructions). Check if any from Form(s):**",
        "  - \\[1\\] [ ] 8814",
        "  - \\[2\\] [ ] 4972",
        "  - \\[3\\] [ ] [blank]",
        "- \\[16\\]: 0.00",
    )
    w2_boxes = [
        {
            "type": "section",
            "id": "e",
            "items": [
                {"type": "field", "field": "text", "label": "Employee's first name and initial", "value": "Joan"},
                {"type": "field", "field": "text", "label": "Last name", "isEmpty": True},
            ],
        },
        {
            "type": "section",
            "id": "12a",
            "items": [
                {"type": "field", "field": "text", "label": "Code", "value": "D"},
                {"type": "field", "field": "text", "value": "1,500.00"},
            ],
        },
        {
            "type": "field",
            "field": "multi_select",
            "id": "13",
            "valueItems": [{"type": "field", "field": "checkbox", "label": "Statutory employee", "value": True}],
        },
        {"type": "field", "field": "text", "id": "9", "isEmpty": True},
        {"type": "field", "field": "text", "label": "FAX (A/C, No):", "isEmpty": True},
        {"type": "field", "field": "checkbox", "id": "D", "label": "Check if this is a PTP", "value": False},
        {"type": "field", "field": "signature", "label": "Your signature", "value": True},
        {"type": "field", "field": "signature", "label": "Spouse's signature", "value": False},
        {"type": "field", "field": "text", "id": "c", "label": "Employer's name", "value": "Wilson\n5435 Bradshaw"},
    ]
    assert render_guide(w2_boxes) == lines(
        "- \\[e\\]",
        "  - **Employee's first name and initial**: Joan",
        "  - **Last name**: [blank]",
        "- \\[12a\\]",
        "  - **Code**: D",
        "  - 1,500.00",
        "- \\[13\\]",
        "  - [x] Statutory employee",
        "- \\[9\\]: [blank]",
        "- **FAX (A/C, No):** [blank]",
        "- \\[D\\] **Check if this is a PTP**: [ ]",
        "- **Your signature**: [signature]",
        "- **Spouse's signature**: [blank]",
        "- \\[c\\] **Employer's name**: Wilson",
        "  5435 Bradshaw",
    )
    part_and_grid = [
        {
            "type": "section",
            "id": "Part II",
            "label": "Information About the Partner",
            "items": [{"type": "field", "field": "text", "id": "E", "label": "Partner's SSN", "value": "27-4162256"}],
        },
        {
            "type": "table",
            "id": "J",
            "label": "Partner's share of profit, loss, and capital (see instructions):",
            "columns": ["", "Beginning", "Ending"],
            "rows": [["Profit", "1.0 %", {"items": [{"type": "field", "field": "text", "value": "1.0 %"}]}]],
        },
        {"type": "field", "field": "checkbox", "label": "Check if decrease is due to sale", "value": False},
    ]
    assert render_guide(part_and_grid) == lines(
        "- \\[Part II\\] **Information About the Partner**",
        "  - \\[E\\] **Partner's SSN**: 27-4162256",
        "- \\[J\\] **Partner's share of profit, loss, and capital (see instructions):**",
        "",
        "  <table>",
        "  <tr><th></th><th>Beginning</th><th>Ending</th></tr>",
        "  <tr><td>Profit</td><td>1.0 %</td><td>1.0 %</td></tr>",
        "  </table>",
        "",
        "- **Check if decrease is due to sale**: [ ]",
    )
    recs = parse(render_guide(part_and_grid))
    assert [(r.path, r.label, r.value) for r in recs] == [
        (("Part II",), "Partner's SSN", "27-4162256"),
        (("J",), None, "Profit"),
        (("J",), "Beginning", "1.0 %"),
        (("J",), "Ending", "1.0 %"),
        ((), "Check if decrease is due to sale", False),
    ]


def test_fixture_trees_round_trip_without_loss() -> None:
    for short in ("w2", "k1", "f1040"):
        tree = json.loads((FX / f"{short}.tree.json").read_text())
        for pg in tree["pages"]:
            page = parse_kv_markdown(render_guide(pg["items"]), pg["page_number"])
            assert page.diagnostics["unparsed"] == 0, f"{short} p{pg['page_number']}"
            assert len(page.records) == tree_units(pg["items"]), f"{short} p{pg['page_number']}"
            assert len(page.records) > 40


def test_rows_of_different_tables_never_share_a_parent_identity() -> None:
    table = (
        "<table>\n<tr><th>A</th><th>B</th></tr>\n<tr><td>1</td><td>2</td></tr>\n<tr><td>3</td><td>4</td></tr>\n</table>"
    )
    for _ in range(5):
        page = parse_kv_markdown(f"- **One**\n\n  {table}\n\n- **Two**\n\n  {table}\n")
        assert len({r.parent for r in page.records}) == 4


def test_option_group_without_id_or_caption_keeps_its_shape() -> None:
    tree = [
        {
            "type": "field",
            "field": "single_select",
            "valueItems": [
                {"type": "field", "field": "checkbox", "label": "Yes", "value": True},
                {"type": "field", "field": "checkbox", "label": "No", "value": False},
            ],
        },
        {"type": "field", "field": "checkbox", "label": "Final K-1", "value": True},
    ]
    md = render_guide(tree)
    assert md == lines("-", "  - [x] Yes", "  - [ ] No", "- **Final K-1**: [x]")
    group, final = parse(md)
    assert group.kind == SELECT_KIND and (group.id, group.label) == (None, None)
    assert [(o["label"], o["value"]) for o in group.options] == [("Yes", True), ("No", False)]
    assert (final.kind, final.label, final.value) == ("checkbox", "Final K-1", True)


LINES_OF_BUSINESS: list[dict[str, Any]] = [
    {
        "type": "field",
        "field": "multi_select",
        "label": "INDICATE LINES OF BUSINESS",
        "valueItems": [
            {
                "type": "section",
                "items": [
                    {"type": "field", "field": "checkbox", "label": "BOILER & MACHINERY", "value": False},
                    {"type": "field", "field": "text", "label": "PREMIUM", "isEmpty": True},
                ],
            },
            {
                "type": "section",
                "items": [
                    {"type": "field", "field": "checkbox", "label": "BUSINESS AUTO", "value": True},
                    {"type": "field", "field": "text", "label": "PREMIUM", "value": "1,200.00"},
                ],
            },
            {
                "type": "section",
                "id": "3",
                "items": [
                    {"type": "field", "field": "checkbox", "value": True},
                    {"type": "field", "field": "text", "value": "8888"},
                ],
            },
            {
                "type": "section",
                "items": [
                    {"type": "field", "field": "checkbox", "value": False},
                    {"type": "field", "field": "text", "isEmpty": True},
                    {"type": "field", "field": "text", "label": "AMOUNT", "isEmpty": True},
                ],
            },
        ],
    },
    {"type": "field", "field": "text", "label": "After", "value": "x"},
]


def test_option_groups_with_labeled_members_round_trip_losslessly() -> None:
    md = render_guide(LINES_OF_BUSINESS)
    assert md == lines(
        "- **INDICATE LINES OF BUSINESS**",
        "  - [ ] BOILER & MACHINERY",
        "    - **PREMIUM**: [blank]",
        "  - [x] BUSINESS AUTO",
        "    - **PREMIUM**: 1,200.00",
        "  - \\[3\\] [x] 8888",
        "  - [ ]",
        "    - [blank]",
        "    - **AMOUNT**: [blank]",
        "- **After**: x",
    )
    group, *rest = parse(md)
    assert group.kind == SELECT_KIND and group.label == "INDICATE LINES OF BUSINESS"
    assert [(o["id"], o["label"], o["value"]) for o in group.options] == [
        (None, "BOILER & MACHINERY", False),
        (None, "BUSINESS AUTO", True),
        ("3", "8888", True),
        (None, None, False),
    ]
    assert [(r.path, r.label, r.value) for r in rest] == [
        (("INDICATE LINES OF BUSINESS",), "PREMIUM", None),
        (("INDICATE LINES OF BUSINESS",), "PREMIUM", "1,200.00"),
        (("INDICATE LINES OF BUSINESS",), None, None),
        (("INDICATE LINES OF BUSINESS",), "AMOUNT", None),
        ((), "After", "x"),
    ]
    assert groups(rest) == [0, 1, 2, 2, 3]  # one parent per option's members
    assert not any(isinstance(r.value, str) and r.value.startswith("**") for r in rest)
    (page,) = pages_from_tree([{"page_number": 1, "items": LINES_OF_BUSINESS}])
    assert len(page.records) == tree_units(LINES_OF_BUSINESS) == 6 and page.diagnostics["roundtrip_loss"] == 0


def test_text_only_option_is_a_reported_round_trip_loss() -> None:
    tree = [
        {
            "type": "field",
            "field": "single_select",
            "label": "Type",
            "valueItems": [
                {"type": "field", "field": "checkbox", "label": "Checking", "value": True},
                {"type": "field", "field": "text", "label": "Other", "value": "trust"},
            ],
        }
    ]
    (page,) = pages_from_tree([{"page_number": 1, "items": tree}])
    assert page.diagnostics["roundtrip_loss"] != 0  # the group degrades to a section: visible, never silent


def test_unknown_coverage_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown coverage 'sopt'"):
        parse_kv_markdown("- **A**: 1", coverage="sopt")
    with pytest.raises(ValueError, match="unknown coverage 'partial'"):
        parse_kv_markdown("<!-- coverage: partial -->\n- **A**: 1")
