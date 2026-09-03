"""Behavior tests for the text_color rule (colored revision text)."""

import pytest

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.evaluation.metrics.parse.rules_formatting import (
    TextColorRule,
    _rgb_to_hue_family,
)


def _rule(text: str, color: str) -> TextColorRule:
    rule = create_test_rule({"type": "text_color", "text": text, "color": color})
    assert isinstance(rule, TextColorRule)
    return rule


class TestHueFamily:
    def test_pure_colors(self) -> None:
        assert _rgb_to_hue_family(255, 0, 0) == "red"
        assert _rgb_to_hue_family(0, 128, 0) == "green"
        assert _rgb_to_hue_family(0, 0, 255) == "blue"

    def test_light_red_is_red(self) -> None:
        # LibreOffice track-changes salmon (#ffa6a6)
        assert _rgb_to_hue_family(0xFF, 0xA6, 0xA6) == "red"

    def test_amber_is_orange(self) -> None:
        # LibreOffice track-changes amber (#c69200)
        assert _rgb_to_hue_family(0xC6, 0x92, 0x00) == "orange"

    def test_grayscale_has_no_family(self) -> None:
        assert _rgb_to_hue_family(0, 0, 0) is None
        assert _rgb_to_hue_family(120, 120, 120) is None
        assert _rgb_to_hue_family(255, 255, 255) is None


class TestTextColorRule:
    def test_hex_span_matches_family(self) -> None:
        rule = _rule("five (5) days in advance", "red")
        ok, _ = rule.run('<span style="color:#ffa6a6">five (5) days in advance</span>')
        assert ok

    def test_named_font_color(self) -> None:
        rule = _rule("five (5) days in advance", "red")
        ok, _ = rule.run('<font color="red">five (5) days in advance</font>')
        assert ok

    def test_rgb_value(self) -> None:
        rule = _rule("prior written agreement", "orange")
        ok, _ = rule.run('<span style="color: rgb(198,146,0)">requires prior written agreement between</span>')
        assert ok

    def test_adjacent_family_accepted(self) -> None:
        # amber may reasonably be called yellow or orange
        rule = _rule("prior written agreement", "yellow")
        ok, _ = rule.run('<span style="color:#c69200">prior written agreement</span>')
        assert ok

    def test_wrong_family_fails_with_found_families(self) -> None:
        rule = _rule("five (5) days in advance", "red")
        ok, msg = rule.run('<span style="color:blue">five (5) days in advance</span>')
        assert not ok
        assert "blue" in msg

    def test_plain_strikeout_is_not_color(self) -> None:
        rule = _rule("five (5) days in advance", "red")
        ok, _ = rule.run("~~five (5) days in advance~~")
        assert not ok

    def test_unmarked_text_fails(self) -> None:
        rule = _rule("five (5) days in advance", "red")
        ok, msg = rule.run("must be notified five (5) days in advance")
        assert not ok
        assert "no colored markup" in msg

    def test_nested_markup_inside_span_tolerated(self) -> None:
        rule = _rule("five days in advance", "red")
        ok, _ = rule.run('<span style="color:#ffa6a6">five ~~days~~ in advance</span>')
        assert ok

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError):
            create_test_rule({"type": "text_color", "text": " ", "color": "red"})

    def test_empty_color_rejected(self) -> None:
        with pytest.raises(ValueError):
            create_test_rule({"type": "text_color", "text": "hello world", "color": ""})


class TestAbsentUnlessStrikeoutRule:
    def _rule(self, text: str):
        return create_test_rule({"type": "absent_unless_strikeout", "text": text})

    def test_absent_passes(self) -> None:
        ok, _ = self._rule("five (5) days in advance").run("The remaining agreement text.")
        assert ok

    def test_struck_tilde_passes(self) -> None:
        ok, _ = self._rule("five (5) days in advance").run(
            "~~must be notified at least five (5) days in advance~~ remains"
        )
        assert ok

    def test_struck_html_passes(self) -> None:
        ok, _ = self._rule("five (5) days in advance").run("<del>five (5) days in advance</del>")
        assert ok

    def test_line_through_style_passes(self) -> None:
        ok, _ = self._rule("five (5) days in advance").run(
            '<span style="text-decoration: line-through">five (5) days in advance</span>'
        )
        assert ok

    def test_plain_occurrence_fails(self) -> None:
        ok, msg = self._rule("five (5) days in advance").run("must be notified at least five (5) days in advance.")
        assert not ok
        assert "regular content" in msg

    def test_plain_occurrence_outside_struck_region_fails(self) -> None:
        ok, _ = self._rule("five (5) days in advance").run(
            "~~other struck text~~ but five (5) days in advance stays plain"
        )
        assert not ok

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError):
            create_test_rule({"type": "absent_unless_strikeout", "text": "  "})


class TestPresentAsStrikeoutRule:
    def _rule(self, text: str):
        return create_test_rule({"type": "present_as_strikeout", "text": text})

    def test_struck_passes(self) -> None:
        ok, _ = self._rule("five (5) days in advance").run("~~notified at least five (5) days in advance~~")
        assert ok

    def test_struck_html_passes(self) -> None:
        ok, _ = self._rule("five (5) days in advance").run("<del>five (5) days in advance</del>")
        assert ok

    def test_absent_fails(self) -> None:
        ok, msg = self._rule("five (5) days in advance").run("Unrelated text.")
        assert not ok
        assert "dropped" in msg

    def test_plain_fails(self) -> None:
        ok, msg = self._rule("five (5) days in advance").run("five (5) days in advance stays plain")
        assert not ok
        assert "not marked as struck" in msg

    def test_struck_copy_passes_even_with_plain_copy(self) -> None:
        # The struck copy satisfies retention+marking; the plain leak is
        # absent_unless_strikeout's job to punish.
        ok, _ = self._rule("five (5) days").run("~~five (5) days~~ and five (5) days plain")
        assert ok

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError):
            create_test_rule({"type": "present_as_strikeout", "text": " "})
