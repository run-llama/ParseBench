"""Regression tests for LlamaParse V2 label aliases and picture-type helpers."""

from __future__ import annotations

import pytest

from parse_bench.layout_label_mapping import (
    LLAMAPARSE_V2_RAW_TO_CANONICAL,
    PICTURE_TYPE_ALIASES,
    is_chart_family_picture_type,
    normalize_picture_type,
    picture_types_match,
)
from parse_bench.schemas.layout_ontology import CanonicalLabel


def test_llamaparse_v2_code_alias_maps_to_canonical_code() -> None:
    """The V2 SDK ``CodeItem`` emits ``type="code"``; without this alias code
    blocks were silently dropped from the canonicalized layout."""
    assert LLAMAPARSE_V2_RAW_TO_CANONICAL["code"] == (CanonicalLabel.CODE, {})
    assert LLAMAPARSE_V2_RAW_TO_CANONICAL["algorithm"] == (CanonicalLabel.CODE, {})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bar Chart", "bar_chart"),
        ("photograph", "image"),
        ("screenshot-from-computer", "screenshot"),
        ("geographical_map", "map"),
        ("scatter_plot", "scatter_chart"),
        ("logo", "logo"),
        ("true", None),
        ("caption", None),
        ("", None),
        (None, None),
        ("this is a very long leaked description of the picture", None),
    ],
)
def test_normalize_picture_type(raw: object, expected: str | None) -> None:
    assert normalize_picture_type(raw) == expected


def test_picture_type_aliases_are_normalized_targets() -> None:
    for alias_target in PICTURE_TYPE_ALIASES.values():
        assert normalize_picture_type(alias_target) == alias_target


def test_chart_family_membership() -> None:
    assert is_chart_family_picture_type("chart")
    assert is_chart_family_picture_type("bar_chart")
    assert is_chart_family_picture_type("scatter_chart")
    assert not is_chart_family_picture_type("flow_chart")
    assert not is_chart_family_picture_type("org_chart")
    assert not is_chart_family_picture_type("logo")


def test_picture_types_match_generic_specific_tolerance() -> None:
    assert picture_types_match("chart", "bar_chart")
    assert picture_types_match("pie_chart", "chart")
    assert picture_types_match("logo", "logo")
    assert not picture_types_match("chart", "flow_chart")
    assert not picture_types_match("bar_chart", "pie_chart")
