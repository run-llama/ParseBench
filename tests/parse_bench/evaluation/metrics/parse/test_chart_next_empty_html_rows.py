"""Empty rows in per-series HTML tables must not abort chart scoring."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.chart_next import score_chart_next

CHART = {
    "title": "Sales",
    "categories": ["Alpha", "Beta"],
    "series": [{"name": "Revenue", "values": ["10", "20"]}],
    "data_labels_shown": True,
}
REVENUE = (
    "<table><tr><th>Item</th><th>Revenue</th></tr>"
    "<tr><td>Alpha</td><td>10</td></tr><tr><td>Beta</td><td>20</td></tr>{empty}</table>"
)
COST = "<table><tr><th>Item</th><th>Cost</th></tr><tr><td>Alpha</td><td>4</td></tr>{empty}</table>"


@pytest.mark.parametrize("empty", ["<tr></tr>", "<tr> \n </tr>"])
@pytest.mark.parametrize("location", ["first", "second", "both"])
def test_empty_html_rows_keep_chart_score(empty: str, location: str) -> None:
    control = score_chart_next("## Sales\n\n" + REVENUE.format(empty="") + COST.format(empty=""), [CHART])
    output = (
        "## Sales\n\n"
        + REVENUE.format(empty=empty if location in ("first", "both") else "")
        + COST.format(empty=empty if location in ("second", "both") else "")
    )
    result = score_chart_next(output, [CHART])

    assert control is not None and result is not None
    assert result.score == control.score == 1.0
    assert result.events.kinds() == control.events.kinds() == {}


def test_empty_markdown_rows_remain_supported() -> None:
    output = (
        "## Sales\n\n| Item | Revenue |\n|---|---|\n|Alpha|10|\n|Beta|20|\n||\n\n"
        "| Item | Cost |\n|---|---|\n|Alpha|4|\n||\n"
    )
    result = score_chart_next(output, [CHART])
    assert result is not None and result.score == 1.0


def test_empty_row_does_not_hide_a_wrong_value() -> None:
    output = (
        "## Sales\n\n"
        + REVENUE.format(empty="<tr></tr>").replace("<td>10</td>", "<td>99</td>")
        + COST.format(empty="<tr></tr>")
    )
    result = score_chart_next(output, [CHART])
    assert result is not None and result.score < 1.0
    assert result.events.kinds().get("wrong value") == 1
