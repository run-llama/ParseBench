"""Behavior tests for the mark_color rule's color comparison (hex, rgb, named)."""

import pytest

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.evaluation.metrics.parse.rules_formatting import MarkColorRule, TextColorRule

TEXT = "amended effective date"


def _mark(color: str) -> MarkColorRule:
    rule = create_test_rule({"type": "mark_color", "text": TEXT, "color": color})
    assert isinstance(rule, MarkColorRule)
    return rule


def _run_mark(color: str, attrs: str) -> bool:
    ok, _ = _mark(color).run(f"<mark {attrs}>{TEXT}</mark>")
    return ok


def test_hex_yellow_passes_yellow() -> None:
    assert _run_mark("yellow", 'style="background-color:#ffff00"')
    assert _run_mark("yellow", 'style="background-color: #FF0"')


def test_hex_red_fails_yellow() -> None:
    ok, msg = _mark("yellow").run(f'<mark style="background-color:#ff0000">{TEXT}</mark>')
    assert not ok
    assert "yellow" in msg


def test_named_color_still_passes() -> None:
    assert _run_mark("yellow", 'style="background-color: yellow"')
    assert _run_mark("yellow", 'background="yellow"')
    assert not _run_mark("yellow", 'style="background-color: red"')


def test_rgb_passes() -> None:
    assert _run_mark("yellow", 'style="background-color: rgb(255, 255, 0)"')
    assert not _run_mark("yellow", 'style="background-color: rgb(0, 0, 255)"')


def test_pale_highlight_fill_passes_its_family() -> None:
    assert _run_mark("green", 'style="background-color:#ecfdd7"')


def test_non_family_expected_keeps_substring_match() -> None:
    assert _run_mark("gold", 'style="background-color: gold"')
    assert not _run_mark("gold", 'style="background-color:#ffd700"')


@pytest.mark.parametrize(
    ("expected", "value"),
    [
        ("yellow", "#c69200"),  # orange family, adjacent to yellow
        ("yellow", "#fff2cc"),  # pale orange family, adjacent to yellow
        ("yellow", "#00ff00"),  # green family, not adjacent to yellow
        ("red", "#0000ff"),  # blue family, not adjacent to red
        ("blue", "#800080"),  # purple family, adjacent to blue
    ],
)
def test_adjacent_family_agrees_with_text_color(expected: str, value: str) -> None:
    text_rule = create_test_rule({"type": "text_color", "text": TEXT, "color": expected})
    assert isinstance(text_rule, TextColorRule)
    text_ok, _ = text_rule.run(f'<span style="color:{value}">{TEXT}</span>')
    assert _run_mark(expected, f'style="background-color:{value}"') == text_ok
