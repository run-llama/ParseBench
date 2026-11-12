"""chart_next: cells nobody can read are not required, and model-estimated references get a wider axis tolerance."""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.chart_next import chart_tables_of, score_chart_next

# An unlabeled line chart read off the axis (values 0-100, so the axis tolerance is 2% of 100 = 2).
TREND: dict[str, Any] = {
    "title": "Share of respondents",
    "chart_type": "line",
    "unit": "%",
    "data_labels_shown": False,
    "categories": ["2019", "2020", "2021", "2022"],
    "series": [{"name": "Online", "values": ["20", "40", "60", "100"]}],
}

# A histogram whose empty bins the reference writes as 0.
HISTOGRAM: dict[str, Any] = {
    "title": "Distribution of annual losses",
    "chart_type": "bar",
    "unit": "years",
    "data_labels_shown": False,
    "categories": ["-50%", "-25%", "0%", "25%", "50%"],
    "series": [
        {"name": "Primary perils", "values": ["0", "6", "3", "0", "2"]},
        {"name": "Secondary perils", "values": ["0", "8", "11", "0", "1"]},
    ],
}

# A labeled stacked bar whose thin top segments (1%) carry no label.
OWNERSHIP: dict[str, Any] = {
    "title": "Future car ownership plans",
    "chart_type": "stacked_bar",
    "unit": "%",
    "data_labels_shown": True,
    "categories": ["Global", "China", "Europe"],
    "series": [
        {"name": "Buy new car", "values": ["78", "96", "52"]},
        {"name": "Buy used car", "values": ["17", "1", "36"]},
        {"name": "Private leasing", "values": ["1", "1", "6"]},
        {"name": "Company car", "values": ["1", "1", "1"]},
    ],
}


def md_table(title: str, rows: list[list[str]]) -> str:
    lines = [f"## {title}", "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines) + "\n"


def charts(spec: dict[str, Any], source: str | None = None):
    meta: dict[str, Any] = {"chart_tables": [spec]}
    if source:
        meta["chart_tables_source"] = source
    return chart_tables_of(meta)


def scored(md: str, specs):
    res = score_chart_next(md, specs)
    assert res is not None
    return res


# a reading 4 points off (2x the 2% axis tolerance): a near miss for a measured reference
TREND_OFF = md_table(
    TREND["title"], [["Year", "Online"], ["2019", "24"], ["2020", "40"], ["2021", "60"], ["2022", "100"]]
)


def test_near_miss_counts_against_a_measured_reference():
    res = scored(TREND_OFF, charts(TREND, "auto_vector"))
    assert res.events.kinds().get("value off") == 1
    assert res.score < 1


def test_near_miss_passes_against_a_model_estimated_reference():
    for source in ("review_accept", "review_fixed", "auto_spot"):
        res = scored(TREND_OFF, charts(TREND, source))
        assert res.events.kinds() == {}, source
        assert res.score == 1


def test_estimated_reference_still_charges_a_wrong_reading():
    md = md_table(TREND["title"], [["Year", "Online"], ["2019", "45"], ["2020", "40"], ["2021", "60"], ["2022", "100"]])
    res = scored(md, charts(TREND, "review_fixed"))
    assert res.events.kinds().get("wrong value") == 1


def test_printed_labels_keep_their_tolerance_on_an_estimated_page():
    labeled = {**TREND, "data_labels_shown": True}
    res = scored(TREND_OFF, charts(labeled, "review_accept"))
    assert res.score < 1


def test_zero_bins_left_blank_or_out_are_not_missing():
    blank = md_table(
        HISTOGRAM["title"],
        [
            ["Deviation", "Primary perils", "Secondary perils"],
            ["-50%", "", ""],
            ["-25%", "6", "8"],
            ["0%", "3", "11"],
            ["25%", "", ""],
            ["50%", "2", "1"],
        ],
    )
    rowless = md_table(
        HISTOGRAM["title"],
        [["Deviation", "Primary perils", "Secondary perils"], ["-25%", "6", "8"], ["0%", "3", "11"], ["50%", "2", "1"]],
    )
    for md in (blank, rowless):
        res = scored(md, charts(HISTOGRAM))
        assert res.events.kinds() == {}
        assert res.score == 1


def test_a_wrong_number_in_a_zero_bin_is_still_scored():
    md = md_table(
        HISTOGRAM["title"],
        [
            ["Deviation", "Primary perils", "Secondary perils"],
            ["-50%", "4", "0"],
            ["-25%", "6", "8"],
            ["0%", "3", "11"],
            ["25%", "0", "0"],
            ["50%", "2", "1"],
        ],
    )
    assert scored(md, charts(HISTOGRAM)).score < 1


def test_unlabeled_sliver_segments_left_out_are_not_missing():
    md = md_table(
        OWNERSHIP["title"],
        [
            ["Region", "Buy new car", "Buy used car", "Private leasing", "Company car"],
            ["Global", "78", "17", "", ""],
            ["China", "96", "1", "", ""],
            ["Europe", "52", "36", "6", ""],
        ],
    )
    res = scored(md, charts(OWNERSHIP))
    assert "series missing" not in res.events.kinds()
    assert "point missing" not in res.events.kinds()
    assert res.score == 1


def test_a_readable_segment_left_out_is_still_missing():
    md = md_table(
        OWNERSHIP["title"],
        [
            ["Region", "Buy new car", "Buy used car", "Private leasing", "Company car"],
            ["Global", "78", "17", "", ""],
            ["China", "96", "1", "", ""],
            ["Europe", "52", "36", "", ""],
        ],
    )
    res = scored(md, charts(OWNERSHIP))
    assert res.events.kinds()  # Europe's 6% leasing segment is readable: its series is charged
    assert res.score < 1


# Extra columns next to a complete chart (chart_full_series_design.md §3-4: printed values are complete, "25 (13.0%)";
# printed totals are annotated; whisker ends carry no rules).
STRAIN: dict[str, Any] = {
    "title": "Job strain by occupation",
    "chart_type": "bar",
    "unit": "%",
    "data_labels_shown": False,
    "categories": ["Green-driven", "Green new", "GHG-intensive"],
    "series": [{"name": "value", "values": ["-8.6", "-11.9", "-1"]}],
}
EGDI: dict[str, Any] = {
    "title": "Countries in each EGDI group",
    "chart_type": "stacked_bar",
    "data_labels_shown": True,
    "categories": ["2014", "2016"],
    "series": [{"name": "Low EGDI", "values": ["32", "32"]}, {"name": "High EGDI", "values": ["62", "75"]}],
}


def test_error_bar_bounds_next_to_a_complete_chart_are_free():
    md = md_table(
        STRAIN["title"],
        [
            ["Occupation", "Point estimate (%)", "95% CI lower (%)", "95% CI upper (%)"],
            ["Green-driven", "-8.6", "-15", "-2.5"],
            ["Green new", "-11.8", "-17.1", "-6.5"],
            ["GHG-intensive", "-1", "-9.3", "7.4"],
        ],
    )
    res = scored(md, charts(STRAIN))
    assert "invented series or category" not in res.events.kinds()
    assert res.score == 1


def test_printed_share_and_total_columns_are_free():
    md = md_table(
        EGDI["title"],
        [
            ["Year", "Low EGDI", "Low EGDI (percent)", "High EGDI", "High EGDI (percent)", "Total"],
            ["2014", "32", "16.6", "62", "32.1", "94"],
            ["2016", "32", "16.6", "75", "38.9", "107"],
        ],
    )
    res = scored(md, charts(EGDI))
    assert res.events.kinds() == {}
    assert res.score == 1


def test_an_unexplained_extra_series_is_still_invented():
    md = md_table(
        EGDI["title"],
        [["Year", "Low EGDI", "High EGDI", "Medium EGDI"], ["2014", "32", "62", "40"], ["2016", "32", "75", "41"]],
    )
    assert "invented series or category" in scored(md, charts(EGDI)).events.kinds()
