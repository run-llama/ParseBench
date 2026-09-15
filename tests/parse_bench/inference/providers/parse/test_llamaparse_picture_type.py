"""``extract_picture_type_from_markdown`` reads the figure-classifier alt-text prefix."""

from __future__ import annotations

import pytest

from parse_bench.inference.providers.parse.llamaparse_v2_normalization import extract_picture_type_from_markdown


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("", None),
        ("plain text", None),
        ("![bar chart: revenue by quarter](img.png)", "bar_chart"),
        ("![Logo: ACME](logo.png)", "logo"),
        ("![pie_chart](x.png)", "pie_chart"),
        ("![a photo of a dog](x.png)", None),
        ("see [signature: J. Doe]", "signature"),
        ("![signature: J. Doe](sig.png)", "signature"),
    ],
)
def test_extract_picture_type_from_markdown(markdown: str, expected: str | None) -> None:
    assert extract_picture_type_from_markdown(markdown) == expected
