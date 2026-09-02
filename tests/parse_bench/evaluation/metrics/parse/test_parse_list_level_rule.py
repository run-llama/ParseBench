"""Tests for the `list_level` rule (CommonMark list nesting).

The fixture strings mirror the cases from the design discussion: nesting is
defined by the CommonMark list-items spec (marker column vs the parent item's
first content column), not by a fixed number of spaces.
"""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.rules_base import (
    RuleNotApplicable,
    create_test_rule,
)
from parse_bench.evaluation.metrics.parse.rules_list import (
    ListLevelRule,
    extract_list_items,
)


def _rule(text: str, level: int) -> ListLevelRule:
    return ListLevelRule({"type": "list_level", "text": text, "level": level})


# ---------------------------------------------------------------------------
# CommonMark nesting arithmetic (the spec examples from the design thread)
# ---------------------------------------------------------------------------


def test_child_at_parent_content_column_is_nested() -> None:
    md = "1. foo\n   - hi\n"
    assert _rule("hi", 2).run(md)[0]
    assert _rule("foo", 1).run(md)[0]
    passed, message = _rule("hi", 1).run(md)
    assert not passed
    assert "level(s) [2]" in message


def test_wide_ordered_marker_moves_the_required_column() -> None:
    md = "123. abc\n     1. hi\n"
    assert _rule("abc", 1).run(md)[0]
    assert _rule("hi", 2).run(md)[0]


def test_one_space_indent_does_not_nest_under_a_dash_parent() -> None:
    # The original defect: sub-items indented by one space parse FLAT.
    md = "- parent clause\n - first condition\n - second condition\n"
    passed, message = _rule("first condition", 2).run(md)
    assert not passed
    assert "level(s) [1]" in message
    assert _rule("first condition", 1).run(md)[0]


def test_two_space_convention_does_not_nest_under_an_ordered_parent() -> None:
    # "1. foo" puts content at column 3, so a 2-space child is a sibling list.
    md = "1. foo\n  - hi\n"
    assert not _rule("hi", 2).run(md)[0]
    assert _rule("hi", 1).run(md)[0]


def test_loose_list_with_blank_lines_keeps_nesting() -> None:
    # Ground-truth markdown joins blocks with blank lines; nesting must survive.
    md = "1. alpha step\n\n   - beta detail\n\n2. gamma step\n"
    assert _rule("alpha step", 1).run(md)[0]
    assert _rule("beta detail", 2).run(md)[0]
    assert _rule("gamma step", 1).run(md)[0]


def test_paren_ordered_marker_is_a_valid_commonmark_marker() -> None:
    md = "1) foo\n   - hi\n"
    assert _rule("foo", 1).run(md)[0]
    assert _rule("hi", 2).run(md)[0]


def test_item_flattened_to_a_paragraph_fails() -> None:
    passed, message = _rule("first condition", 1).run("parent clause\n\nfirst condition\n")
    assert not passed
    assert "does not appear as a markdown or HTML list item" in message


def test_parent_own_text_excludes_nested_children() -> None:
    md = "- parent clause\n  - child detail\n"
    assert _rule("parent clause", 1).run(md)[0]
    assert _rule("child detail", 2).run(md)[0]
    # The child text must not be attributed to the parent item.
    assert not _rule("child detail", 1).run(md)[0]


# ---------------------------------------------------------------------------
# Typographic bullet glyphs
# ---------------------------------------------------------------------------


def test_unicode_bullets_are_treated_as_dash_markers() -> None:
    md = "• top item\n  ◦ nested item\n"
    assert _rule("top item", 1).run(md)[0]
    assert _rule("nested item", 2).run(md)[0]


def test_en_dash_bullet_counts_as_a_marker() -> None:
    assert _rule("budget note", 1).run("– budget note\n")[0]


def test_bullet_glyph_under_ordered_parent_needs_the_content_column() -> None:
    md = "1. numbered step\n   • sub note\n"
    assert _rule("sub note", 2).run(md)[0]
    md_flat = "1. numbered step\n  • sub note\n"
    assert not _rule("sub note", 2).run(md_flat)[0]


# ---------------------------------------------------------------------------
# HTML lists
# ---------------------------------------------------------------------------


def test_html_nested_lists_report_levels() -> None:
    md = "<ul><li>outer point<ul><li>inner point</li></ul></li></ul>"
    assert _rule("outer point", 1).run(md)[0]
    assert _rule("inner point", 2).run(md)[0]
    assert not _rule("inner point", 1).run(md)[0]


def test_html_implicit_li_close_between_siblings() -> None:
    md = "<ol><li>first entry<li>second entry</ol>"
    assert _rule("first entry", 1).run(md)[0]
    assert _rule("second entry", 1).run(md)[0]


# ---------------------------------------------------------------------------
# Tables are out of scope
# ---------------------------------------------------------------------------


def test_pipe_table_cell_bullets_are_not_list_items() -> None:
    md = "| item | note |\n| --- | --- |\n| - cell bullet | x |\n"
    passed, message = _rule("cell bullet", 1).run(md)
    assert not passed
    assert "does not appear" in message


def test_html_table_content_is_masked() -> None:
    md = "<table><tr><td><ul><li>cell entry</li></ul></td></tr></table>\n\n- real entry\n"
    assert not _rule("cell entry", 1).run(md)[0]
    assert _rule("real entry", 1).run(md)[0]


def test_masking_tables_does_not_shift_surrounding_levels() -> None:
    md = "1. before table\n\n   | a | b |\n   | - | - |\n\n   - after table\n"
    assert _rule("after table", 2).run(md)[0]


# ---------------------------------------------------------------------------
# Matching and rule hygiene
# ---------------------------------------------------------------------------


def test_match_uses_normalized_substring() -> None:
    md = "- The **quarterly** report   was filed.\n"
    assert _rule("quarterly report was filed", 1).run(md)[0]


def test_match_ignores_whitespace_differences() -> None:
    # Styling a number routinely inserts a space the source never had.
    md = "2. 申請者が連絡先を変更したときは、 **15** 日以内に届け出てください。\n"
    assert _rule("申請者が連絡先を変更したときは、15 日以内に届け出てください。", 1).run(md)[0]


def test_duplicate_text_passes_when_any_occurrence_has_the_level() -> None:
    md = "- shared phrase\n  - shared phrase\n"
    assert _rule("shared phrase", 1).run(md)[0]
    assert _rule("shared phrase", 2).run(md)[0]


def test_degenerate_marker_text_is_not_applicable() -> None:
    with pytest.raises(RuleNotApplicable):
        _rule("**", 1).run("- item\n")


def test_empty_text_and_bad_level_are_invalid() -> None:
    with pytest.raises(ValueError):
        ListLevelRule({"type": "list_level", "text": "  ", "level": 1})
    with pytest.raises(ValueError):
        ListLevelRule({"type": "list_level", "text": "item", "level": 0})
    with pytest.raises(ValueError):
        ListLevelRule({"type": "list_level", "text": "item"})


def test_factory_dispatch_builds_the_rule_from_a_raw_dict() -> None:
    rule = create_test_rule({"id": "list-level:x", "type": "list_level", "text": "hi", "level": 2})
    assert isinstance(rule, ListLevelRule)
    assert rule.run("1. foo\n   - hi\n")[0]


def test_extract_list_items_reports_text_and_levels() -> None:
    items = set(extract_list_items("1. foo\n   - hi\n\n<ul><li>web</li></ul>"))
    assert ("foo", 1) in items
    assert ("hi", 2) in items
    assert ("web", 1) in items
