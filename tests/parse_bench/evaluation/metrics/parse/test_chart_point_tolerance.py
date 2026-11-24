"""Opt-in point margins must not relax neighbours or expand with a wrong prediction."""

import copy

import pytest

from parse_bench.evaluation.metrics.parse.chart_next import ChartSpec, score_chart_next

CHART = {
    "title": "Ownership",
    "chart_type": "bar",
    "data_labels_shown": True,
    "unit": "%",
    "categories": ["Global", "Europe"],
    "series": [
        {"name": "Printed", "values": ["78", "52"]},
        {
            "name": "Estimated",
            "values": ["2.5", "6"],
            "point_tolerances": [
                {"relative_tolerance": 0.1, "absolute_tolerance": 2},
                None,
            ],
        },
    ],
}


def score(chart: dict, printed: str = "78", estimate: str = "4.5", neighbour: str = "6") -> float:
    md = (
        "## Ownership\n\n| Region | Printed | Estimated |\n|---|---|---|\n"
        f"| Global | {printed} | {estimate} |\n| Europe | 52 | {neighbour} |"
    )
    result = score_chart_next(md, [chart])
    assert result is not None
    return result.score


def test_point_margin_accepts_estimate_without_relaxing_printed_or_null_neighbour() -> None:
    assert score(CHART) == 1
    assert score(CHART, printed="85") < 1
    assert score(CHART, neighbour="8") < 1


@pytest.mark.parametrize("estimate", ["0.5", "4.5", "20", "-20"])
def test_absolute_bound_does_not_expand_with_prediction(estimate: str) -> None:
    assert (score(CHART, estimate=estimate) == 1) == (estimate in {"0.5", "4.5"})


def test_absent_and_null_overrides_keep_legacy_score() -> None:
    absent = copy.deepcopy(CHART)
    del absent["series"][1]["point_tolerances"]
    null = copy.deepcopy(absent)
    null["series"][1]["point_tolerances"] = [None, None]
    assert score(absent) == score(null) < 1


@pytest.mark.parametrize(
    "override",
    [
        [{}],
        [{"relative_tolerance": -1, "absolute_tolerance": 2}, None],
        [{"relative_tolerance": float("nan"), "absolute_tolerance": 2}, None],
        [{"relative_tolerance": 0.1}, None],
    ],
)
def test_invalid_override_fails_instead_of_silently_loosening_gt(override: list) -> None:
    chart = copy.deepcopy(CHART)
    chart["series"][1]["point_tolerances"] = override
    with pytest.raises(ValueError):
        ChartSpec.from_dict(chart)


def test_colour_group_merge_keeps_tolerance_on_its_original_category() -> None:
    chart = {
        "title": "Ownership",
        "chart_type": "bar",
        "data_labels_shown": True,
        "categories": ["Global", "Europe"],
        "series": [
            {
                "name": "Group A",
                "values": ["2.5", ""],
                "point_tolerances": [{"relative_tolerance": 0.1, "absolute_tolerance": 2}, None],
            },
            {"name": "Group B", "values": ["", "52"]},
        ],
    }
    for value, expected in [("52", True), ("60", False)]:
        md = f"## Ownership\n\n| Region | Group A / Group B |\n|---|---|\n| Global | 4.5 |\n| Europe | {value} |"
        result = score_chart_next(md, [chart])
        assert result is not None
        assert (result.score >= 0.99) == expected
