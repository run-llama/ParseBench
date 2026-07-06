"""Tests for FormattingRule italic detection (CommonMark flanking rules).

Two classes of markdown ``*``/``_`` characters are not emphasis delimiters:

* a ``*`` followed by whitespace cannot open emphasis and one preceded by
  whitespace cannot close it — so list bullets and arithmetic (``5 * 3``)
  are not delimiters, and
* an intraword ``_`` (``snake_case``) is not a delimiter under CommonMark's
  flanking rules.

Without these distinctions, ``is_italic`` rules pass spuriously on bulleted
lists and identifier-heavy text, and ``is_not_italic`` rules fail on them.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule


def _run(rule_type: str, text: str, md: str) -> bool:
    passed, _ = create_test_rule({"type": rule_type, "text": text}).run(md)
    return passed


class TestStarFlanking:
    def test_bulleted_list_is_not_emphasis(self):
        md = "* item one\n* item two\n"
        assert not _run("is_italic", "item one", md)
        assert _run("is_not_italic", "item one", md)

    def test_indented_bullets_are_not_emphasis(self):
        md = "  * nested item\n  * second item\n"
        assert not _run("is_italic", "nested item", md)

    def test_blockquoted_bullets_are_not_emphasis(self):
        md = "> * item one\n> * item two\n"
        assert not _run("is_italic", "item one", md)

    def test_trailing_bullet_at_end_of_string(self):
        # A bare bullet as the last character is not a closing delimiter.
        md = "*alpha beta\n*"
        assert _run("is_not_italic", "alpha", md)

    def test_midline_arithmetic_stars_are_not_emphasis(self):
        md = "cost is 5 * 3 apples and 7 * 2 pears total"
        assert not _run("is_italic", "3 apples and 7", md)

    def test_real_emphasis_inside_bullet_still_detected(self):
        md = "* item with *emph* here\n"
        assert _run("is_italic", "emph", md)

    def test_plain_star_emphasis_still_detected(self):
        assert _run("is_italic", "real italic", "some *real italic* text")

    def test_star_emphasis_substring_query(self):
        assert _run("is_italic", "Grazing Line", "*Grazing Line, Macquarie Marshes, NSW*")

    def test_bold_detection_unaffected_by_bullets(self):
        md = "* item with **bold** here\n"
        assert _run("is_bold", "bold", md)


class TestUnderscoreFlanking:
    def test_snake_case_is_not_emphasis(self):
        md = "the snake_case_word identifier"
        assert not _run("is_italic", "case", md)
        assert _run("is_not_italic", "case", md)

    def test_span_between_identifiers_is_not_emphasis(self):
        md = "Call parse_config and load_data now"
        assert not _run("is_italic", "config and load", md)

    def test_trailing_underscore_identifiers_are_not_emphasis(self):
        md = "the file_ and dir_ paths"
        assert not _run("is_italic", "and", md)

    def test_leading_underscore_identifiers_are_not_emphasis(self):
        md = "call _init and _load helpers"
        assert not _run("is_italic", "init and", md)

    def test_real_underscore_emphasis_still_detected(self):
        assert _run("is_italic", "this", "see _this_ word")

    def test_underscore_emphasis_at_string_edges(self):
        assert _run("is_italic", "leading", "_leading_ emphasis")
        assert _run("is_italic", "trailing", "emphasis _trailing_")

    def test_underscore_emphasis_with_punctuation(self):
        assert _run("is_italic", "title", "(_title_)")

    def test_emphasis_containing_snake_case_still_detected(self):
        # An intraword ``_`` inside a real italic span is content, not a
        # delimiter — the span must still be found around and across it.
        md = "_see config_yaml for details_"
        assert _run("is_italic", "details", md)
        assert _run("is_italic", "see", md)

    def test_double_underscore_is_bold_not_italic(self):
        assert not _run("is_italic", "bold", "some __bold__ text")


class TestStripConsistency:
    def test_bold_query_containing_snake_case_survives_fallback(self):
        # The fallback path strips italic markup from content when testing
        # bold; flanking-aware stripping must not eat intraword underscores.
        md = "**use ~~snake_case_word~~ now** here"
        assert _run("is_bold", "use snake_case_word now", md)
