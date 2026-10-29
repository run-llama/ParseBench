"""chart_next reads a ``{series: {category: value}}`` JSON object written anywhere on the page."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.chart_next import score_chart_next

# two bonds issued at the same tenor: the category "15yr" appears twice, in order
YIELDS = {
    "title": "Yields at issuance",
    "chart_type": "bar",
    "data_labels_shown": True,
    "categories": ["10yr", "15yr", "5yr", "15yr"],
    "series": [
        {"name": "Bond", "values": ["-0.2", "-0.1", "-0.5", "1.3"]},
        {"name": "Benchmark", "values": ["-0.3", "-0.3", "-0.6", "1.1"]},
    ],
}


def page(bond: str, benchmark: str) -> str:
    """Prose, a heading, then an unfenced JSON object holding the series under ``values``, then more prose."""
    return (
        "Some text before the figure.\n\n## Yields at issuance\n\n{\n"
        '  "title": "Yield comparison",\n'
        f'  "values": {{\n    "Bond": {{{bond}}},\n    "Benchmark": {{{benchmark}}}\n  }}\n}}\n\n'
        "Source: issuer data.\n"
    )


def test_unfenced_series_object_mid_page_is_read_with_repeated_categories():
    md = page(
        '"10yr": -0.2, "15yr": -0.1, "5yr": -0.5, "15yr": 1.3',
        '"10yr": -0.3, "15yr": -0.3, "5yr": -0.6, "15yr": 1.1',
    )
    res = score_chart_next(md, [YIELDS])
    assert res is not None and res.score == 1.0


def test_wrong_values_in_a_series_object_still_score_low():
    md = page(
        '"10yr": 2.2, "15yr": 3.1, "5yr": 4.5, "15yr": 5.3',
        '"10yr": 2.3, "15yr": 3.3, "5yr": 4.6, "15yr": 5.1',
    )
    res = score_chart_next(md, [YIELDS])
    assert res is not None and res.score < 0.5


def test_single_series_inside_named_panels_is_read():
    chart = {
        "title": "Exports",
        "chart_type": "bar",
        "data_labels_shown": True,
        "categories": ["2020", "2021", "2022"],
        "series": [{"name": "Exports", "values": ["34.6", "40.1", "54"]}],
    }
    md = (
        '{\n  "title": "None",\n  "panels": {\n'
        '    "Exports": {"values": {"2020": 34.6, "2021": 40.1, "2022": 54.0}}\n  }\n}\n'
    )
    res = score_chart_next(md, [chart])
    assert res is not None and res.score == 1.0
