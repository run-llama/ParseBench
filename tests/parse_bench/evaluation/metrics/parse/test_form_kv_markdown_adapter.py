"""Adapter pins for `form_kv_scoring.kv_markdown_adapter`: each page-markdown shape becomes the
guide's list line, content is re-emitted verbatim, and what is not a form shape is left alone."""

from __future__ import annotations

from pathlib import Path

from parse_bench.evaluation.metrics.parse.form_kv_scoring import parse_kv_markdown, render_guide
from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_markdown_adapter import (
    adapt_page_markdown,
    adapter_summary,
)

FX = Path(__file__).parent / "fixtures" / "form_kv"


def adapt(*lines: str) -> str:
    return adapt_page_markdown("\n".join(lines))[0]


def records(*lines: str):
    return parse_kv_markdown(adapt(*lines)).records


def test_bold_colon_paragraphs_become_list_lines_verbatim() -> None:
    out, diag = adapt_page_markdown("**a Name**: Ada Lovelace\n\n**Suffix**:     \n\n**Phone:** 555-0100\n")
    assert out == "- **a Name**: Ada Lovelace\n\n- **Suffix**:\n\n- **Phone:** 555-0100\n"
    assert diag["md_key_lines"] == 3 and diag["md_adapted_lines"] == 3
    a, suffix, phone = records("**a Name**: Ada Lovelace", "**Suffix**:     ", "**Phone:** 555-0100")
    assert (a.label, a.value) == ("a Name", "Ada Lovelace")
    assert (suffix.label, suffix.value) == ("Suffix", None)
    assert (phone.label, phone.value) == ("Phone:", "555-0100")


def test_several_pairs_on_one_line_split_at_their_keys() -> None:
    line = "**Your first name**: Jennifer H. **Last name**: Hale **Your social security number**:"
    assert adapt(line) == (
        "- **Your first name**: Jennifer H.\n- **Last name**: Hale\n- **Your social security number**:"
    )
    assert adapt("**Estate of** [________], **Deceased**") == "- **Estate of** [________]\n- **Deceased**"
    assert [(r.label, r.value) for r in records("**Estate of** [________], **Deceased**")] == [
        ("Estate of", None),
        ("Deceased", None),
    ]


def test_bold_inside_a_value_is_emphasis_not_a_key() -> None:
    assert adapt("**Note**: see **Part II** for details") == "- **Note**: see **Part II** for details"
    (r,) = records("**Note**: see **Part II** for details")
    assert r.value == "see **Part II** for details"


def test_printed_line_numbers_become_ids_and_their_second_occurrence_marks_the_value() -> None:
    line = "**1a** Total amount from Form(s) W-2, box 1 (see instructions) **1a** 163,641.64"
    assert adapt(line) == "- \\[1a\\] **Total amount from Form(s) W-2, box 1 (see instructions)**: 163,641.64"
    (r,) = records(line)
    assert (r.id, r.label, r.value) == ("1a", "Total amount from Form(s) W-2, box 1 (see instructions)", "163,641.64")
    # an empty value cell is a blank; a sub-line's marker may carry the parent number
    (blank,) = records("**b** Nontaxable combat pay election      **66b**")
    assert (blank.id, blank.label, blank.value) == ("b", "Nontaxable combat pay election", None)
    # a bold section word before the line number is its own (blank) key
    section, item = records("**Income** **1a** Total amount **1a** 5")
    assert (section.label, section.value) == ("Income", None)
    assert (item.id, item.value) == ("1a", "5")
    # the marker after a colon key is the value cell, and bold emphasis in a label is dropped
    (r,) = records("**13** **FUTA tax deposited for the year** **13**")
    assert (r.id, r.label, r.value) == ("13", "FUTA tax deposited for the year", None)
    assert adapt("* **4a IRAs, pensions, and annuities**:      **4a**      **b Taxable amount**:      **4b**") == (
        "* **4a IRAs, pensions, and annuities**:\n* **b Taxable amount**:"
    )


def test_boxes_after_a_key_form_an_option_group() -> None:
    assert adapt("**Filing Status** [ ] Single [x] Head of household (HOH)") == (
        "- **Filing Status**\n  - [ ] Single\n  - [x] Head of household (HOH)"
    )
    (group,) = records("**Filing Status** [ ] Single [x] Head of household (HOH)")
    assert group.label == "Filing Status"
    assert [(o["label"], o["value"]) for o in group.options] == [("Single", False), ("Head of household (HOH)", True)]
    # one box after the key stays a checkbox value; the grammar reads its write-in as usual
    box, write_in = records("**Other (specify):** [x] consulting")
    assert (box.kind, box.value) == ("checkbox", True) and (write_in.kind, write_in.value) == ("text", "consulting")


def test_trailing_and_inline_boxes_without_a_key() -> None:
    assert adapt("* CHECKING [ ]", "* SAVINGS [x]") == "* [ ] CHECKING\n* [x] SAVINGS"  # the item's marker stays
    assert adapt("Other [ ] (specify)     ") == "- [ ] Other (specify)"
    assert adapt("[ ] Single [x] Head of household") == "- \n  - [ ] Single\n  - [x] Head of household"
    assert adapt("1. a. Have you paid wages? [ ] Yes [ ] No") == (
        "- **a. Have you paid wages?**\n  - [ ] Yes\n  - [ ] No"
    )
    (group,) = records("1. a. Have you paid wages? [ ] Yes [ ] No")
    assert group.label == "a. Have you paid wages?" and [o["label"] for o in group.options] == ["Yes", "No"]


def test_a_captioned_group_with_trailing_box_options_parses_as_the_guide_group() -> None:
    lines = ("* **40. Plumbing system?**:", "    - Yes [ ]", "    - No [x]", "    - NA [ ]")
    assert adapt(*lines) == "* **40. Plumbing system?**:\n    - [ ] Yes\n    - [x] No\n    - [ ] NA"
    (group,) = records(*lines)
    assert group.kind == "multi_select" and group.label == "40. Plumbing system?"
    assert [(o["label"], o["value"]) for o in group.options] == [("Yes", False), ("No", True), ("NA", False)]


def test_box_first_paragraphs_and_numbered_fields_become_list_lines() -> None:
    assert adapt("[ ] Certified Copies ($10 per document)") == "- [ ] Certified Copies ($10 per document)"
    (box,) = records("[x] Certified Copies ($10 per document)")
    assert (box.kind, box.label, box.value) == ("checkbox", "Certified Copies ($10 per document)", True)
    # a numbered item is a field only when it carries a key, a box or a fill marker
    assert adapt("2. Did you acquire the assets?") == "2. Did you acquire the assets?"
    assert adapt("2. **Amount paid**: 4,200") == "- **Amount paid**: 4,200"
    assert adapt("3) Fee [___]") == "- **Fee**: [blank]"


def test_unbolded_captions_with_a_fill_marker_are_blank_fields() -> None:
    assert adapt("Name: ____________________") == "- **Name:** [blank]"
    assert adapt("Date [___]") == "- **Date**: [blank]"
    (name,) = records("Name: ____________________")
    assert (name.label, name.value) == ("Name:", None)
    # a colon in prose, or a value, is not a field
    assert adapt("I understand that:") == "I understand that:"
    assert adapt("File Online : see the portal") == "File Online : see the portal"
    assert adapt("I, ____, hereby authorize ____ to release to:") == "I, ____, hereby authorize ____ to release to:"


def test_a_caption_printed_under_its_line_captions_that_blank() -> None:
    out, diag = adapt_page_markdown("[________________________]\n(Signature)\n\n[________]\n(City, State, Zip)\n")
    assert out == "- **(Signature)** [blank]\n\n- **(City, State, Zip)** [blank]\n"
    assert diag["md_under_captions"] == 2
    (sig, city) = parse_kv_markdown(out).records
    assert (sig.label, sig.value, city.label) == ("(Signature)", None, "(City, State, Zip)")


def test_labelled_boxes_on_one_line_stay_separate_fields() -> None:
    assert adapt("**Approve** [ ] **Disapprove** [x]") == "- **Approve** [ ]\n- **Disapprove** [x]"
    approve, disapprove = records("**Approve** [ ] **Disapprove** [x]")
    assert (approve.kind, approve.value, disapprove.value) == ("checkbox", False, True)
    # an id, a box, then the option's bold label: one option, the bold is its label
    (box, write_in) = records("* **6a** [ ] **Yourself**. If someone can claim you as a dependent")
    assert (box.kind, box.id, box.value) == ("checkbox", "6a", False)
    assert write_in.value == "Yourself. If someone can claim you as a dependent"


def test_bold_captions_and_pipe_tables_are_left_alone() -> None:
    for line in (
        "**FOR COURT USE ONLY**",
        "**Additional Details**",
        "| **Name** | **Phone** |",
        "**Name (print)** | **Phone Number** | **Relationship**",
    ):
        assert adapt(line) == line
    # a bold caption with a colon or a value after it is still a field
    assert adapt("**Suffix**:") == "- **Suffix**:"
    assert adapt("**Date**:\\") == "- **Date**:"


def test_emphasised_numbers_and_continued_captions() -> None:
    assert adapt("**8 Total** income from the worksheet. Enter **0** if not applicable. **8**") == (
        "- **8 Total income from the worksheet. Enter 0 if not applicable.**:"
    )
    (r,) = records("**7** List the income (*see instructions*) of each owner **7** 4,200")
    assert (r.id, r.label, r.value) == ("7", "List the income (*see instructions*) of each owner", "4,200")


GRID = (
    "<table><tr><th>Name</th><th>Relationship</th><th>Age</th></tr>"
    "<tr><td>Ada</td><td>Daughter</td><td>12</td></tr><tr><td>Bob</td><td>Son</td><td>9</td></tr></table>"
)


def test_headings_prose_and_grids_pass_through() -> None:
    untouched = "\n".join(
        [
            "# Form W-2",
            "### **Contact Information**",
            "Plain prose with a colon: not a pair.",
            "Label: value without bold",
            GRID,
            "![logo](page_1_image_1.jpg)",
            "",
        ]
    )
    out, diag = adapt_page_markdown(untouched)
    assert out == untouched and diag["md_adapted_lines"] == 0 and adapter_summary(diag) is None


def test_bold_lead_ins_and_long_values_are_text_not_fields() -> None:
    """Bold lead-ins, bold sentences and long values in ordinary documents stay unchanged."""
    for line in (
        "**The bug.** When a core is disabled and then re-enabled, load balancing stops.",
        "**42%** of our staff work from home at least one day a week",
        "**Artículo 1º.-** Aprobar el reglamento general del programa EJEMPLO",
        "**Regulatory Matters**: The partnership may face potential regulatory fines, penalties, or "
        "compliance costs that could reduce the distributions available to the partners.",
        "**Key:** Presentation ![icon: presentation](page_1_image_10_v2.jpg) Drop-in",
        "1. **Reducing Emissions**",
        "2. **Exhibit A**",
        "* **Program**: Other Liability - occurrence",
        "* **RV** – Right Ventricle",
    ):
        assert adapt(line) == line, line
    # a short value after a colon key is a field, and so is any pair with a fill marker or a box
    assert adapt("**Line Of Business**: Commercial Package Policy Property") == (
        "- **Line Of Business**: Commercial Package Policy Property"
    )
    assert adapt("**Deceased** [___]") == "- **Deceased** [___]"


def test_guide_list_lines_are_left_as_they_are() -> None:
    guide = render_guide(
        [
            {"type": "field", "field": "text", "id": "1", "label": "Wages", "value": "29,513"},
            {"type": "field", "field": "checkbox", "id": "b", "label": "Retired", "value": True},
            {"type": "field", "field": "checkbox", "label": "Spouse retired", "value": False},
            {"type": "field", "field": "text", "id": "c", "label": "Suffix", "isEmpty": True},
            {
                "type": "field",
                "field": "single_select",
                "label": "Filing Status",
                "valueItems": [
                    {"type": "field", "field": "checkbox", "label": "Single", "value": False},
                    {"type": "field", "field": "checkbox", "label": "Head of household", "value": True},
                ],
            },
        ]
    )
    out, diag = adapt_page_markdown(guide)
    assert out == guide and diag["md_adapted_lines"] == 0


def test_todays_w2_page_markdown_is_read_as_records() -> None:
    page = parse_kv_markdown(adapt_page_markdown((FX / "w2.p1.page.md").read_text())[0])
    labels = {r.label for r in page.records}
    assert {"a Employee's social security number", "13 Statutory employee", "Retirement plan"} <= labels
    assert page.diagnostics["bold_colon_unparsed"] == 0
    assert next(r.value for r in page.records if r.label == "13 Statutory employee") is True


def test_summary_line_counts_what_was_rewritten() -> None:
    _, diag = adapt_page_markdown("**A**: 1 **B**: 2\n\n[ ] x [x] y\n\n3. **Fee**: 5\n\nName: ____\n")
    assert adapter_summary(diag) == (
        "4 page-markdown lines read through the markdown-dialect adapter (1 key paragraphs, "
        "1 pairs split off shared lines, 1 box runs grouped, 1 numbered items, 1 unbolded blank fields)"
    )


def test_contents_page_numbers_are_not_printed_ids() -> None:
    # a bold number after a dot leader ends a contents line; a value-cell marker still binds
    for line in (
        "**Relaxing** ..................................**14**",
        "**Eating** ..................................... **84**",
    ):
        assert adapt_page_markdown(line)[0] == line
    assert adapt_page_markdown("**b** Nontaxable combat pay election . . . . . **66b**")[0] == (
        "- \\[b\\] **Nontaxable combat pay election . . . . .**:"
    )


def test_a_form_laid_out_as_a_table_is_read_as_fields() -> None:
    table = (
        "<table><tr><td><strong>A</strong> <strong>NAME OF PAYEE</strong> (<em>last, first</em>)</td>"
        "<td><strong>D</strong> <strong>TYPE OF DEPOSITOR ACCOUNT</strong> [ ] CHECKING [x] SAVINGS</td></tr>"
        "<tr><td><strong>CITY</strong>  <strong>STATE</strong>  <strong>ZIP CODE</strong></td>"
        "<td><strong>F</strong> <strong>TYPE OF PAYMENT</strong></td></tr>"
        "<tr><td>[ ] Social Security</td><td>[x] Mil. Active</td></tr>"
        "<tr><td>Street Address - Do not list a P.O. Box</td><td>City (no abbreviations)</td></tr></table>"
    )
    out, diag = adapt_page_markdown(table)
    assert out.split("\n") == [
        "- \\[A\\] **NAME OF PAYEE (\\*last, first\\*)**:",
        "- \\[D\\] **TYPE OF DEPOSITOR ACCOUNT**:",
        "  - [ ] CHECKING",
        "  - [x] SAVINGS",
        "- **CITY**:",
        "- **STATE**:",
        "- **ZIP CODE**:",
        "- \\[F\\] **TYPE OF PAYMENT**",
        "  - [ ] Social Security",
        "  - [x] Mil. Active",
        "- **Street Address - Do not list a P.O. Box**:",
        "- **City (no abbreviations)**:",
    ]
    assert diag["md_tables_unwrapped"] == 1 and "table" in (adapter_summary(diag) or "")


def test_a_caption_cell_beside_a_value_cell_is_one_pair() -> None:
    out, _ = adapt_page_markdown("<table><tr><td><strong>Name:</strong></td><td>Ada Lovelace</td></tr></table>")
    assert out == "- **Name**: Ada Lovelace"
