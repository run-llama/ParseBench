"""Tests for normalize_cell_text() in parse evaluation utils."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from parse_bench.evaluation.metrics.parse.utils import normalize_cell_text  # noqa: E402


class TestHTMLFormattingStripping:
    """Formatting markers are stripped on the GriTS side. Models routinely
    emphasize totals/headers with bold while ground truth doesn't — treating
    that as a content mismatch would penalize parsing quality for an
    unrelated convention. ``<b>Total</b>``, ``**Total**``, and plain
    ``Total`` all compare equal.
    """

    def test_mark_tag(self) -> None:
        assert normalize_cell_text("<mark>highlighted</mark>") == "highlighted"

    def test_bold_tag(self) -> None:
        assert normalize_cell_text("<b>bold</b>") == "bold"

    def test_italic_tag(self) -> None:
        assert normalize_cell_text("<i>italic</i>") == "italic"

    def test_em_tag(self) -> None:
        assert normalize_cell_text("<em>em</em>") == "em"

    def test_strong_tag(self) -> None:
        assert normalize_cell_text("<strong>strong</strong>") == "strong"

    def test_mixed_html_formatting(self) -> None:
        assert normalize_cell_text("<b>bold</b> and <i>italic</i>") == "bold and italic"

    def test_case_insensitive_tags(self) -> None:
        assert normalize_cell_text("<B>bold</B>") == "bold"

    def test_html_bold_equals_plain(self) -> None:
        assert normalize_cell_text("<b>Total</b>") == normalize_cell_text("Total")


class TestMarkdownStripping:
    def test_bold_asterisks(self) -> None:
        assert normalize_cell_text("**bold**") == "bold"

    def test_bold_underscores(self) -> None:
        assert normalize_cell_text("__bold__") == "bold"

    def test_italic_asterisk(self) -> None:
        assert normalize_cell_text("*italic*") == "italic"

    def test_italic_underscore(self) -> None:
        assert normalize_cell_text("_italic_") == "italic"

    def test_strikethrough(self) -> None:
        assert normalize_cell_text("~~struck~~") == "struck"

    def test_md_bold_equals_plain(self) -> None:
        assert normalize_cell_text("**Total**") == normalize_cell_text("Total")


class TestSubSupConversion:
    """P5: GriTS now applies TRM-style sup/sub conversion."""

    def test_sup_tag_digit(self) -> None:
        assert normalize_cell_text("x<sup>2</sup>") == "x2"

    def test_sub_tag_digit(self) -> None:
        assert normalize_cell_text("H<sub>2</sub>O") == "H2O"

    def test_sub_tag_letter(self) -> None:
        assert normalize_cell_text("H<sub>x</sub>O") == "HxO"

    def test_unicode_superscript_to_ascii(self) -> None:
        assert normalize_cell_text("x²") == "x2"

    def test_unicode_subscript_to_ascii(self) -> None:
        assert normalize_cell_text("H₂O") == "H2O"


class TestNoLowercaseOrAccentStripping:
    """Negative tests: GriTS does NOT lowercase or strip accents (TRM does)."""

    def test_no_lowercasing(self) -> None:
        assert normalize_cell_text("Hello World") == "Hello World"

    def test_no_accent_stripping(self) -> None:
        assert normalize_cell_text("Café") == "Café"


class TestDotLeaderStripping:
    def test_trailing_dots(self) -> None:
        assert normalize_cell_text("Revenue.........") == "Revenue"

    def test_trailing_dots_with_spaces(self) -> None:
        assert normalize_cell_text("Revenue......   ") == "Revenue"

    def test_single_dot_preserved(self) -> None:
        assert normalize_cell_text("Inc.") == "Inc."

    def test_mid_text_dots_preserved(self) -> None:
        assert normalize_cell_text("A..B") == "A..B"


class TestDashNormalization:
    def test_dash_only_triple(self) -> None:
        assert normalize_cell_text("---") == "-"

    def test_dash_only_em_dash(self) -> None:
        assert normalize_cell_text("\u2014") == "-"  # em-dash

    def test_dash_only_en_dash_spaced(self) -> None:
        assert normalize_cell_text("\u2013 \u2013") == "-"  # en-dash spaced

    def test_dash_only_mixed(self) -> None:
        assert normalize_cell_text("- - -") == "-"

    def test_mixed_content_with_dashes_preserved(self) -> None:
        assert normalize_cell_text("2020-01") == "2020-01"

    def test_en_dash_in_content(self) -> None:
        assert normalize_cell_text("2020\u201301") == "2020-01"

    def test_minus_sign_normalized(self) -> None:
        assert normalize_cell_text("\u2212" + "5") == "-5"  # minus sign


class TestExistingBehavior:
    def test_whitespace_collapsing(self) -> None:
        assert normalize_cell_text("  hello   world  ") == "hello world"

    def test_bullet_equivalence(self) -> None:
        # BLACK CIRCLE → BULLET
        assert normalize_cell_text("\u25cf item") == "\u2022 item"

    def test_quote_normalization(self) -> None:
        assert normalize_cell_text("\u201chello\u201d") == '"hello"'

    def test_empty_string(self) -> None:
        assert normalize_cell_text("") == ""

    def test_no_lowercasing(self) -> None:
        assert normalize_cell_text("Hello World") == "Hello World"

    def test_no_accent_removal(self) -> None:
        assert normalize_cell_text("caf\u00e9") == "caf\u00e9"


class TestBooleanMarkerNormalization:
    def test_check_and_cross_match_bracketed_yes_no(self) -> None:
        assert normalize_cell_text("✓") == "yes"
        assert normalize_cell_text("[yes]") == "yes"
        assert normalize_cell_text("X") == "yes"
        assert normalize_cell_text("x") == "yes"
        assert normalize_cell_text("✗") == "no"
        assert normalize_cell_text("[no]") == "no"

    def test_filled_dot_matches_yes(self) -> None:
        assert normalize_cell_text("●") == "yes"
        assert normalize_cell_text("[yes]") == "yes"

    def test_open_circle_matches_no(self) -> None:
        assert normalize_cell_text("○") == "no"
        assert normalize_cell_text("[no]") == "no"

    def test_mixed_bullet_text_is_not_boolean_marker(self) -> None:
        assert normalize_cell_text("● item") == "• item"

    def test_numeric_cells_are_not_boolean_markers(self) -> None:
        assert normalize_cell_text("1") == "1"
        assert normalize_cell_text("0") == "0"
