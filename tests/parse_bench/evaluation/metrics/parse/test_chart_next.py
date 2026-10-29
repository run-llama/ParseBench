"""Tests for the chart data metric (``chart_next``)."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.chart_next import (
    FLOOR,
    chart_next_metrics,
    chart_rules,
    chart_tables_of,
    full_label,
    hungarian,
    label_bind,
    nx_label,
    score_chart_next,
    score_chart_rules,
    unit_desc,
)

REVENUE = {
    "title": "Quarterly revenue by region",
    "chart_type": "bar",
    "x_label": "Quarter",
    "y_label": "Revenue",
    "unit": "USD millions",
    "data_labels_shown": True,
    "categories": ["Q1 2023", "Q2 2023", "Q3 2023"],
    "series": [
        {"name": "North", "values": ["412.5", "430", "455"]},
        {"name": "South", "values": ["300", "310.2", "299"]},
    ],
}


def table(rows: list[list[str]], title: str | None = "Quarterly revenue by region") -> str:
    head = f"## {title}\n\n" if title else ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return head + "\n".join(lines) + "\n"


GOOD = [
    ["Quarter", "North", "South"],
    ["Q1 2023", "412.5", "300"],
    ["Q2 2023", "430", "310.2"],
    ["Q3 2023", "455", "299"],
]


def score(md: str, charts=None) -> object:
    res = score_chart_next(md, [REVENUE] if charts is None else charts)
    assert res is not None
    return res


def kinds(res) -> dict[str, int]:
    return res.events.kinds()


def with_cell(r: int, c: int, value: str, rows=GOOD) -> list[list[str]]:
    out = [list(x) for x in rows]
    out[r][c] = value
    return out


# --- primitives -------------------------------------------------------------------------------------------------


def test_label_normalization_quarters_years_units_and_plurals():
    assert nx_label("1Q23") == nx_label("Q1 2023") == nx_label("2023 Q1") == "2023q1"
    assert nx_label("'23") == "2023"
    assert nx_label("Revenue (USD m)") == "revenue"  # unit parenthetical dropped
    assert nx_label("SIDS (Yes)") == "sids (yes)"  # not a unit: kept
    assert nx_label("Years") == nx_label("year")


def test_unit_desc_reads_kind_and_scale():
    assert unit_desc("USD bn") == {"kind": "cur", "scale": 1e9}
    assert unit_desc("%") == {"kind": "pct"}
    assert unit_desc("percentage points") == {"kind": "pp"}
    assert unit_desc("thousands") == {"scale": 1e3}
    assert unit_desc("Region") is None


def test_hungarian_finds_the_minimum_cost_assignment():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    pairs = sorted(hungarian(cost))
    assert sum(cost[i][j] for i, j in pairs) == 5
    assert sorted(hungarian([[1, 0]])) == [(0, 1)]  # rectangular


# --- full chart_tables mode ---------------------------------------------------------------------------------------


def test_exact_match_scores_one():
    res = score(table(GOOD))
    assert res.score == 1.0
    assert res.mode == "tables" and res.n_points == 6 and res.charts_found == 1
    assert res.events.critical == res.events.major == res.events.minor == 0


def test_value_beyond_tolerance_is_a_critical_wrong_value():
    res = score(table(with_cell(1, 1, "512.5")))
    assert kinds(res) == {"wrong value": 1}
    assert res.events.critical == 1
    assert res.score <= 0.90  # critical floor: at least 10 points


def test_value_slightly_off_is_major():
    res = score(table(with_cell(1, 1, "420")))  # 1.8% off, tolerance 1%
    assert kinds(res) == {"value off": 1}
    assert res.score <= 0.95


def test_unit_scale_from_header_suffix_and_full_digits():
    bn = [["Quarter", "North (USD bn)", "South (USD bn)"], ["Q1 2023", "0.4125", "0.3"], ["Q2 2023", "0.43", "0.3102"]]
    bn.append(["Q3 2023", "0.455", "0.299"])
    assert score(table(bn)).score == 1.0
    suffixed = [
        GOOD[0],
        ["Q1 2023", "$412.5M", "$300M"],
        ["Q2 2023", "$430M", "$310.2M"],
        ["Q3 2023", "$455M", "$299M"],
    ]
    assert score(table(suffixed)).score == 1.0
    raw = [GOOD[0], ["Q1 2023", "412,500,000", "300,000,000"], ["Q2 2023", "430,000,000", "310,200,000"]]
    raw.append(["Q3 2023", "455,000,000", "299,000,000"])
    assert score(table(raw)).score == 1.0


def test_unit_contradiction_is_wrong():
    pct = [["Quarter", "North (%)", "South (%)"], *GOOD[1:]]
    res = score(table(pct))
    assert res.events.critical == 6 and res.score == pytest.approx(0.0)
    millions_as_billions = [["Quarter", "North (USD bn)", "South (USD bn)"], *GOOD[1:]]
    assert score(table(millions_as_billions)).events.critical == 6


def test_transposed_table_is_reoriented():
    transposed = [[GOOD[r][c] for r in range(len(GOOD))] for c in range(len(GOOD[0]))]
    transposed[0][0] = "Region"
    res = score(table(transposed))
    # the corner "Region" names the rows of the transposed table: a generic label header, not a wrong axis name
    assert res.score == 1.0
    assert kinds(res) == {}
    transposed[0][0] = "Revenue growth"  # a specific, unrelated name in the corner still costs one minor
    res = score(table(transposed))
    assert res.score == 1.0 - 0.01 and kinds(res) == {"category axis mislabeled": 1}


def test_missing_series_is_critical():
    res = score(table([[r[0], r[1]] for r in GOOD]))
    assert kinds(res) == {"series missing": 1}
    assert res.events.critical == 1 and res.events.major == 0
    assert res.recall == pytest.approx(0.5) and res.precision == pytest.approx(1.0)
    assert res.score == pytest.approx(min(2 / 3, 0.9))


def test_missing_category_is_one_major():
    res = score(table(GOOD[:3]))
    assert kinds(res) == {"category missing": 1}
    assert res.events.major == 1 and res.events.critical == 0


def test_renamed_series_is_a_minor_alias():
    renamed = with_cell(0, 1, "Northern region")
    res = score(table(renamed))
    assert kinds(res) == {"series name reworded": 1}
    assert res.score == pytest.approx(0.99)


def test_placeholder_legend_is_a_lost_legend_not_an_alias():
    # a placeholder name carrying the series' values binds it; with two or more series the values can no
    # longer be attributed, so the lost legend is one critical per chart
    placeholder = with_cell(0, 2, "Series 2")
    res = score(table(placeholder))
    assert kinds(res) == {"legend lost": 1} and res.events.critical == 1
    assert res.score == pytest.approx(0.90)
    # a single series under "Series 1" keeps its values readable: one major
    single = {**REVENUE, "series": REVENUE["series"][:1]}
    res = score(table([[r[0], r[1]] for r in with_cell(0, 1, "Series 1")]), [single])
    assert kinds(res) == {"legend lost": 1} and res.events.major == 1


def test_label_typo_is_tolerated_as_minor():
    res = score(table(with_cell(0, 1, "Nroth")))  # swapped neighbours: one OSA edit
    assert res.events.critical == 0 and res.events.major == 0
    assert kinds(res) == {"label typo": 1}  # one misspelt series name: one typo, not one per point
    # binding strength (OSA 0.8) weights the point; the floor only charges the minor
    assert res.floor == pytest.approx(0.99) and res.score == pytest.approx(0.9)


def test_missing_or_wrong_title_is_major():
    res = score(table(GOOD, title=None))
    assert kinds(res) == {"title missing": 1} and res.score == pytest.approx(0.95)
    res = score(table(GOOD, title="Employment by sector"))
    assert kinds(res) == {"wrong title": 1} and res.title == 0.0
    # a "Figure 3:" prefix or a caption in bold is still the title
    assert score("**Figure 3: Quarterly revenue by region**\n\n" + table(GOOD, title=None)).score == 1.0


def test_totals_row_is_derived_major_not_invented():
    res = score(table([*GOOD, ["Total", "1297.5", "909.2"]]))
    assert kinds(res) == {"derived total": 1}
    assert res.events.critical == 0 and res.events.major == 1
    assert res.precision == pytest.approx(1.0)


def test_range_containing_the_value_is_one_minor():
    res = score(table(with_cell(2, 1, "425-435")))  # contains 430
    assert kinds(res) == {"range instead of value": 1}
    assert res.events.critical == res.events.major == 0
    assert res.score == pytest.approx(0.99)
    ranges = [GOOD[0], *([r[0], f"{float(r[1]) - 2}-{float(r[1]) + 2}", r[2]] for r in GOOD[1:])]
    assert kinds(score(table(ranges))) == {"range instead of value": 1}  # once per chart
    # a range that misses the value: its midpoint is scored as a value
    res = score(table(with_cell(2, 1, "500-520")))
    assert kinds(res) == {"wrong value": 1}


def test_quarter_label_formats_are_equivalent():
    rows = [GOOD[0], ["1Q23", "412.5", "300"], ["2Q23", "430", "310.2"], ["Q3 '23", "455", "299"]]
    assert score(table(rows)).score == 1.0


def test_unknown_reference_cells_are_skipped():
    chart = {**REVENUE, "series": [{"name": "North", "values": ["412.5", "", "455"]}, REVENUE["series"][1]]}
    res = score(table(GOOD), [chart])
    assert res.n_points == 5 and res.score == 1.0  # the 430 in the output is neither scored nor invented
    res = score(table(with_cell(2, 1, "999")), [chart])
    assert res.score == 1.0


def test_multiple_charts_are_assigned_to_their_tables():
    jobs = {
        "title": "Unemployment rate by sex",
        "x_label": "Year",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["2021", "2022", "2023"],
        "series": [
            {"name": "Men", "values": ["5.1", "4.2", "3.9"]},
            {"name": "Women", "values": ["5.6", "4.8", "4.1"]},
        ],
    }
    jobs_md = table(
        [["Year", "Men", "Women"], ["2021", "5.1%", "5.6%"], ["2022", "4.2%", "4.8%"], ["2023", "3.9%", "4.1%"]],
        title="Unemployment rate by sex",
    )
    md = jobs_md + "\n" + table(GOOD)  # tables in the reverse order of the reference charts
    res = score(md, [REVENUE, jobs])
    assert res.n_charts == 2 and res.charts_found == 2
    assert res.score == 1.0
    res = score(table(GOOD), [REVENUE, jobs])
    assert kinds(res) == {"chart missing": 1} and res.recall == pytest.approx(0.5)


def test_json_output_is_read_by_the_content_adapter():
    md = (
        "## Quarterly revenue by region\n\n```json\n"
        '[{"Quarter": "Q1 2023", "North": 412.5, "South": 300}, {"Quarter": "Q2 2023", "North": 430, "South": 310.2},'
        ' {"Quarter": "Q3 2023", "North": 455, "South": 299}]\n```\n'
    )
    assert score(md).score == 1.0


def test_unlabeled_chart_tolerates_axis_reading_noise():
    chart = {**REVENUE, "data_labels_shown": False}
    rows = [GOOD[0], ["Q1 2023", "420", "295"], ["Q2 2023", "425", "315"], ["Q3 2023", "460", "300"]]
    assert score(table(rows), [chart]).score == 1.0
    assert score(table(rows)).score < 0.9  # the same reading on a labeled chart is wrong


def test_full_labels_bind_fiscal_years_footnotes_months_and_quarters():
    assert full_label("FY2011") == full_label("2011") == full_label("FY 11")
    assert full_label("Poland*") == full_label("Poland¹") == full_label("Poland [a]") == full_label("poland")
    assert full_label("Jan-23") == full_label("January 2023") == full_label("2023 Jan")
    assert full_label("Q1-23") == full_label("1Q 2023") == full_label("2023 Q1")
    assert full_label("Investment (Trillion USD (2023, MER))") == "investment"
    assert full_label("Net additions (right axis)") == "net addition"
    assert label_bind("2011", "FY2011") >= 0.95
    assert label_bind("SIDS (Yes %)", "SIDS (Yes %)") > label_bind("SIDS (Yes %)", "SIDS (No %)")


def test_floor_first_event_full_then_scaled_by_page_size():
    # the first wrong value costs the full 10 points; the next ones cost 10 x min(1, 10 / cells on the page)
    cats = [str(2000 + i) for i in range(50)]
    big = {
        "title": "Big chart",
        "data_labels_shown": True,
        "categories": cats,
        "series": [
            {"name": "A", "values": [str(100 + i) for i in range(50)]},
            {"name": "B", "values": [str(200 + i) for i in range(50)]},
        ],
    }
    rows = [["Year", "A", "B"]] + [[c, str(100 + i), str(200 + i)] for i, c in enumerate(cats)]
    for r in range(1, 11):
        rows[r][1] = "999"
    res = score(table(rows, title="Big chart"), [big])
    assert kinds(res) == {"wrong value": 10}
    assert res.floor == pytest.approx(1 - FLOOR["critical"] - FLOOR["critical"] * 9 * 10 / 100)
    small = score(table(with_cell(1, 1, "512.5")))
    assert small.floor == pytest.approx(0.9)  # one wrong value always costs at least 10 points
    # structural events are not scaled
    clean = [["Year", "A"]] + [[c, str(100 + i)] for i, c in enumerate(cats)]
    res = score(table(clean, title="Big chart"), [big])
    assert kinds(res) == {"series missing": 1} and res.floor == pytest.approx(1 - FLOOR["critical"])


def test_floor_budget_counts_the_cells_of_the_page():
    # two 6-cell panels on one page: after the first wrong value (full), each costs 0.1 x 10 / 12
    second = {**REVENUE, "title": "Quarterly costs by region"}
    md = table(with_cell(1, 1, "512.5")) + "\n" + table(with_cell(1, 1, "512.5"), title="Quarterly costs by region")
    res = score(md, [REVENUE, second])
    assert kinds(res) == {"wrong value": 2}
    assert res.floor == pytest.approx(1 - FLOOR["critical"] - FLOOR["critical"] * 10 / 12)


def test_unlabeled_tolerance_is_a_share_of_the_value_range():
    chart = {
        "title": "Shares",
        "data_labels_shown": False,
        "categories": ["A", "B", "C"],
        "series": [{"name": "Value", "values": ["100", "50", "10"]}],
    }

    def one(c_value: str):
        return score(table([["Item", "Value"], ["A", "100"], ["B", "50"], ["C", c_value]], title="Shares"), [chart])

    # range 0..100 (bar baseline included): within 2 points is exact, up to 3x is off, beyond is wrong
    assert one("11.9").score == 1.0  # 19% relative error, but 1.9% of the range
    assert kinds(one("15")) == {"value off": 1}
    assert kinds(one("20")) == {"wrong value": 1}
    # a labeled chart keeps the relative rule, and a value rounded to the printed precision is exact
    labeled = {**chart, "data_labels_shown": True, "series": [{"name": "Value", "values": ["100", "50", "10.4"]}]}
    md = table([["Item", "Value"], ["A", "100"], ["B", "50"], ["C", "10"]], title="Shares")
    assert score(md, [labeled]).score == 1.0
    md = table([["Item", "Value"], ["A", "100"], ["B", "50"], ["C", "11.9"]], title="Shares")
    assert kinds(score(md, [labeled])) == {"wrong value": 1}


def test_prose_units_are_not_a_contradiction():
    pct = {
        "title": "Unemployment rate",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["2021", "2022"],
        "series": [{"name": "Rate", "values": ["5.1", "4.2"]}],
    }
    prose = "Spending on benefits reached USD 5 million in 2023 (in millions).\n\n"
    md = prose + table([["Year", "Rate"], ["2021", "5.1"], ["2022", "4.2"]], title="Unemployment rate")
    assert score(md, [pct]).score == 1.0
    prose = "Revenue in USD millions grew 5% in 2023, with a 12% margin.\n\n"
    assert score(prose + table(GOOD)).score == 1.0
    # "Top 5%" is a label, not a percentage unit; a reference without unit has nothing to contradict
    quantiles = {
        "title": "Markups",
        "data_labels_shown": True,
        "categories": ["1997", "1998"],
        "series": [{"name": "Top 5%", "values": ["1.91", "1.96"]}, {"name": "Median", "values": ["0.96", "0.95"]}],
    }
    md = table([["Year", "Top 5%", "Median"], ["1997", "1.91", "0.96"], ["1998", "1.96", "0.95"]], title="Markups")
    assert score(md, [quantiles]).score == 1.0


def html_table(rows: list[list[str]], row_th: bool = True, thead: bool = False) -> str:
    head = "".join(f"<th>{x}</th>" for x in rows[0])
    body = []
    for r in rows[1:]:
        label = f"<th>{r[0]}</th>" if row_th else f"<td>{r[0]}</td>"
        body.append("<tr>" + label + "".join(f"<td>{x}</td>" for x in r[1:]) + "</tr>")
    if thead:
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def test_html_row_header_cells_are_not_header_rows():
    title = "## Quarterly revenue by region\n\n"
    for thead in (False, True):
        res = score(title + html_table(GOOD, row_th=True, thead=thead))
        assert res.score == 1.0, thead
    # a row group header spanning the table labels the rows below it; a footer note is not data
    md = (
        title + "<table><tr><th>Quarter</th><th>North</th><th>South</th></tr>"
        '<tr><th colspan="3">2023</th></tr>'
        + "".join(f"<tr><th>{r[0]}</th><td>{r[1]}</td><td>{r[2]}</td></tr>" for r in GOOD[1:])
        + '<tr><td colspan="3">Source: company filings, figures in USD millions</td></tr></table>'
    )
    assert score(md).score == 1.0


def test_a_non_chart_table_on_the_page_is_not_charged():
    other = table(
        [["Quarter", "Headcount", "Stores"], ["Q1 2023", "1,204", "88"], ["Q2 2023", "1,250", "90"]],
        title="Key operating figures",
    )
    res = score(other + "\n" + table(GOOD) + "\n" + table([["Item", "Count"], ["Apples", "3"], ["Pears", "4"]], None))
    assert res.score == 1.0 and res.charts_found == 1
    assert res.events.critical == res.events.major == res.events.minor == 0


def test_multi_panel_charts_in_one_output_table():
    panels = [
        {
            "title": title,
            "data_labels_shown": True,
            "categories": ["2021", "2022", "2023"],
            "series": [{"name": "Employment rate", "values": vals}],
        }
        for title, vals in (("A. Men", ["70.1", "71.0", "71.5"]), ("B. Women", ["60.2", "61.4", "62.0"]))
    ]
    md = table(
        [["Year", "A. Men", "B. Women"], ["2021", "70.1", "60.2"], ["2022", "71.0", "61.4"], ["2023", "71.5", "62.0"]],
        title="Employment rate by sex",
    )
    res = score(md, panels)
    assert res.charts_found == 2 and res.score == 1.0
    # multi-series panels merged into one long table with a panel column
    panels = [
        {
            "title": title,
            "data_labels_shown": True,
            "categories": ["2021", "2022"],
            "series": [{"name": "North", "values": n}, {"name": "South", "values": s_}],
        }
        for title, n, s_ in (("Crude steel", ["10", "11"], ["5", "6"]), ("Iron", ["20", "21"], ["7", "8"]))
    ]
    rows = [
        ["Product", "Year", "North", "South"],
        ["Crude steel", "2021", "10", "5"],
        ["Crude steel", "2022", "11", "6"],
    ]
    rows += [["Iron", "2021", "20", "7"], ["Iron", "2022", "21", "8"]]
    res = score(table(rows, title="Production"), panels)
    assert res.charts_found == 2 and res.score == 1.0
    # a reference that splits a category by series is read from a two-level header ("Women - Researchers")
    chart = {
        "title": "R&D personnel",
        "data_labels_shown": True,
        "categories": ["EU Women", "EU Men", "BE Women", "BE Men"],
        "series": [
            {"name": "Researchers", "values": ["61", "64", "60", "63"]},
            {"name": "Technicians", "values": ["39", "36", "40", "37"]},
        ],
    }
    rows = [["Country", "Women - Researchers", "Women - Technicians", "Men - Researchers", "Men - Technicians"]]
    rows += [["EU", "61", "39", "64", "36"], ["BE", "60", "40", "63", "37"]]
    assert score(table(rows, title="R&D personnel"), [chart]).score == 1.0


def test_lost_legend_binds_by_values_once_per_chart():
    renamed = with_cell(0, 2, "Orange Line", with_cell(0, 1, "Dark Blue Line"))
    res = score(table(renamed))
    assert kinds(res) == {"legend lost": 1} and res.score == pytest.approx(0.90)
    unnamed = with_cell(0, 2, "", with_cell(0, 1, ""))
    assert kinds(score(table(unnamed))) == {"legend lost": 1}
    # a lost legend whose values do not match is not bound: the series is missing, the column invented
    wrong = [list(r) for r in with_cell(0, 2, "Orange Line")]
    for r in wrong[1:]:
        r[2] = "999"
    res = score(table(wrong))
    assert kinds(res)["series missing"] == 1 and kinds(res)["invented series or category"] == 1
    assert "legend lost" not in kinds(res)


# --- sparse chart_data_point mode ---------------------------------------------------------------------------------


def rules(*items: tuple[list[str], str]) -> list[dict]:
    return [{"type": "chart_data_point", "labels": labels, "value": value} for labels, value in items]


def test_sparse_rules_fallback_scores_points_on_any_table():
    other = "| Item | Count |\n|---|---|\n| Apples | 3 |\n| Pears | 4 |\n\n"
    md = other + table(GOOD)
    res = score_chart_next(md, None, rules((["Q1 2023", "North"], "412.5"), (["Q3 2023", "South"], "299")))
    assert res is not None and res.mode == "rules"
    assert res.score == 1.0 and res.strict_pass == 1.0


def test_sparse_wrong_value_is_critical_and_missing_is_major():
    res = score_chart_rules(table(with_cell(1, 1, "512.5")), chart_rules(rules((["Q1 2023", "North"], "412.5"))))
    assert res.events.critical == 1 and res.score == 0.0
    res = score_chart_rules(
        table(GOOD), chart_rules(rules((["Q4 2023", "North"], "470"), (["Q1 2023", "North"], "412.5")))
    )
    assert res.events.major == 1 and res.score == pytest.approx(0.5)


def test_sparse_rules_have_no_contradiction_penalty():
    # the same labels bind a second, different cell (a long-format table with the year omitted from the rule)
    md = table([["Region", "Year", "Value"], ["North", "2023", "412.5"], ["North", "2024", "500"]])
    res = score_chart_rules(md, chart_rules(rules((["North"], "412.5"))))
    assert res.score == 1.0 and res.events.critical == 0


def test_metrics_and_metadata_entry_points():
    assert chart_next_metrics(table(GOOD)) == []
    charts = chart_tables_of({"chart_tables": [REVENUE]})
    metrics = chart_next_metrics(table(GOOD), charts)
    names = {m.metric_name: m.value for m in metrics}
    assert names["chart_next"] == 1.0 and names["chart_next_recall"] == 1.0 and "chart_next_precision" in names
    assert metrics[0].metadata["mode"] == "tables"


def test_single_series_extra_columns_are_invented():
    single = {**REVENUE, "series": REVENUE["series"][:1]}
    rows = [[r[0], r[1], r[1]] for r in GOOD]  # the series duplicated
    res = score(table(rows), [single])
    assert kinds(res) == {"invented series or category": 1} and res.score <= 0.95
    rows = [[r[0], r[1], r[2]] for r in with_cell(0, 2, "Zenith")]  # an extra series
    assert kinds(score(table(rows), [single])) == {"invented series or category": 1}
    # the same series in another unit is not invented
    rows = [["Quarter", "North (USD m)", "North (EUR m)"]] + [[r[0], r[1], r[1]] for r in GOOD[1:]]
    assert score(table(rows), [single]).score == 1.0


def test_continuous_extra_categories_are_free_only_between_reference_categories():
    chart = {
        "title": "Unemployment rate",
        "chart_type": "line",
        "unit": "%",
        "data_labels_shown": False,
        "categories": ["2016", "2018", "2020"],
        "series": [{"name": "Rate", "values": ["5.1", "4.0", "8.3"]}],
    }
    dense = [["Year", "Rate"], ["2016", "5.1"], ["2017", "4.6"], ["2018", "4.0"], ["2019", "3.8"], ["2020", "8.3"]]
    assert score(table(dense, title="Unemployment rate"), [chart]).score == 1.0
    beyond = [*dense, ["2024", "3.1"]]
    assert kinds(score(table(beyond, title="Unemployment rate"), [chart])) == {"invented series or category": 1}


def test_two_digit_years_bind_as_truncated_categories():
    chart = {
        "title": "Unemployment rate by sex",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["2016", "2017"],
        "series": [{"name": "Men", "values": ["5.1", "4.6"]}, {"name": "Women", "values": ["4.6", "4.2"]}],
    }
    md = table([["Year", "Men", "Women"], ["16", "5.1", "4.6"], ["17", "4.6", "4.2"]], title="Unemployment rate by sex")
    res = score(md, [chart])
    assert res.charts_found == 1 and kinds(res) == {"category label truncated": 2}
    assert 0.85 <= res.score <= 0.95
    md = md.replace("| 16 |", "| '16 |").replace("| 17 |", "| '17 |")  # the printed abbreviation is exact
    assert score(md, [chart]).score == 1.0


# --- harder page layouts (one synthetic fixture each) --------------------------------------------------------

FUNDRAISING = {
    "title": "Figure 4.1 – Latest Fundraisings",
    "chart_type": "bar",
    "unit": "GBP millions",
    "data_labels_shown": False,
    "categories": ["2017", "2020", "2023"],
    "series": [{"name": "Fundraisings (GBP)", "values": ["22", "189", "218"]}],
}
FUND_ROWS = [["Year", "Fundraisings (GBP)"], ["2017", "22"], ["2020", "189"], ["2023", "218"]]


def test_numbered_title_identical_to_the_heading_is_found():
    # the heading equals the title, "Figure 4.1" stripped from one side only
    assert score(table(FUND_ROWS, title="Figure 4.1 – Latest Fundraisings"), [FUNDRAISING]).score == 1.0
    # the title as a plain line right above the table
    md = "Figure 4.1 – Latest Fundraisings\n\n" + table(FUND_ROWS, title=None)
    assert score(md, [FUNDRAISING]).score == 1.0
    # a caption that goes on with a subtitle
    md = "**Figure 4.1 – Latest Fundraisings**; GBP millions\n\n" + table(FUND_ROWS, title=None)
    assert score(md, [FUNDRAISING]).score == 1.0
    # prose that mentions the figure is not its title
    md = "Figure 4.1 shows the value of fundraisings secured within their latest rounds.\n\n" + table(FUND_ROWS, None)
    assert kinds(score(md, [FUNDRAISING])) == {"title missing": 1}


def test_hedged_values_are_read_as_numbers():
    # hedged cells ("Approximately £22M", "~£189M", "c. 218") are read as their numbers
    rows = [["Year", "Fundraisings (GBP)"], ["2017", "Approximately £22M"], ["2020", "~£189M"], ["2023", "c. 218"]]
    res = score(table(rows, title=FUNDRAISING["title"]), [FUNDRAISING])
    assert res.score == 1.0 and kinds(res) == {}
    rows = [["Year", "Fundraisings (GBP)"], ["2017", "about 22"], ["2020", "189 (est.)"], ["2023", "approx. 218"]]
    assert score(table(rows, title=FUNDRAISING["title"]), [FUNDRAISING]).score == 1.0
    # an upper bound containing the value is a range (one minor), a bound equal to the value is exact
    pie = {
        **FUNDRAISING,
        "data_labels_shown": True,
        "series": [{"name": "Fundraisings (GBP)", "values": ["0.4", "1", "218"]}],
    }
    rows = [["Year", "Fundraisings (GBP)"], ["2017", "<1"], ["2020", "<1"], ["2023", "218"]]
    res = score(table(rows, title=FUNDRAISING["title"]), [pie])
    assert kinds(res) == {"range instead of value": 1} and res.score == pytest.approx(0.99)
    # a tilde inside a cell is not a hedge ("9~0" is not 90)
    rows = [["Year", "Fundraisings (GBP)"], ["2017", "22~0"], ["2020", "189"], ["2023", "218"]]
    assert "wrong value" not in kinds(score(table(rows, title=FUNDRAISING["title"]), [FUNDRAISING]))


def test_excel_trendlines_are_optional_and_bind_their_series_name():
    # a trendline "Linear (global average score)" written "Global Average", the panel's own series (named like
    # the chart) under a "Value" column
    chart = {
        "title": "LDCs",
        "chart_type": "bar",
        "data_labels_shown": True,
        "categories": ["IF", "CP", "SP"],
        "series": [
            {"name": "LDCs", "values": ["0.6284", "0.4272", "0.2878"]},
            {"name": "Linear (global average score)", "values": ["0.5754", "0.5754", "0.5754"]},
            {"name": "Linear (average score for LDCs)", "values": ["0.3264", "0.3264", "0.3264"]},
        ],
    }
    rows = [
        ["Subindex", "Value", "Group Average", "Global Average"],
        ["IF", "0.6284", "0.3264", "0.5754"],
        ["CP", "0.4272", "0.3264", "0.5754"],
        ["SP", "0.2878", "0.3264", "0.5754"],
    ]
    res = score(table(rows, title="LDCs"), [chart])
    assert res.score == 1.0 and kinds(res) == {}
    # a trendline left out costs nothing (it is derived from the data)
    res = score(table([r[:2] for r in rows], title="LDCs"), [chart])
    assert res.score == 1.0 and res.n_points == 3


def test_split_labels_bind_across_row_and_column():
    # series "ETS 2012" written as a Year column (2012) and an "ETS" column
    chart = {
        "title": "Effective carbon rates",
        "unit": "EUR",
        "data_labels_shown": True,
        "categories": ["CHE", "DNK"],
        "series": [
            {"name": "ETS 2012", "values": ["2.9", "2.6"]},
            {"name": "Carbon taxes 2012", "values": ["12.6", "12.6"]},
            {"name": "ETS 2021", "values": ["3.5", "20"]},
            {"name": "Carbon taxes 2021", "values": ["37.7", "14.5"]},
        ],
    }
    rows = [
        ["Country", "Year", "ETS", "Carbon taxes"],
        ["CHE", "2012", "2.9", "12.6"],
        ["CHE", "2021", "3.5", "37.7"],
        ["DNK", "2012", "2.6", "12.6"],
        ["DNK", "2021", "20", "14.5"],
    ]
    res = score(table(rows, title="Effective carbon rates"), [chart])
    assert res.score == 1.0 and kinds(res) == {}


def test_integer_column_that_tells_repeated_rows_apart_is_a_label():
    # categories "Germany 1".."Germany 4" as a Country column plus a decile column
    cats = [f"{c} {d}" for c in ("Germany", "France") for d in (1, 2)]
    chart = {
        "title": "Growth rates by decile",
        "data_labels_shown": True,
        "categories": cats,
        "series": [
            {"name": "ECR", "values": ["0.2", "0.3", "0.5", "0.6"]},
            {"name": "CPI", "values": ["1.5", "1.5", "1", "1"]},
        ],
    }
    rows = [["Country", "Income decile", "ECR", "CPI"]]
    rows += [
        ["Germany", "1", "0.2", "1.5"],
        ["Germany", "2", "0.3", "1.5"],
        ["France", "1", "0.5", "1"],
        ["France", "2", "0.6", "1"],
    ]
    res = score(table(rows, title="Growth rates by decile"), [chart])
    assert res.score == 1.0 and kinds(res) == {}


def test_middle_header_level_melts_into_the_rows():
    # categories "Canada Dec 2023" over months x (group > country > measure) headers
    chart = {
        "title": "Posted wages",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["Canada Dec 2023", "Canada Jan 2024", "France Dec 2023", "France Jan 2024"],
        "series": [
            {"name": "Wage growth", "values": ["4", "3.7", "4", "3.7"]},
            {"name": "Inflation", "values": ["3.2", "3.1", "3.8", "3.5"]},
        ],
    }
    md = (
        "## Posted wages\n\n<table><thead>"
        "<tr><th colspan='5'>Posted wage growth and inflation</th></tr>"
        "<tr><th>Country</th><th colspan='2'>Canada</th><th colspan='2'>France</th></tr>"
        "<tr><th>Month</th><th>Wage growth</th><th>Inflation</th><th>Wage growth</th><th>Inflation</th></tr>"
        "</thead><tbody>"
        "<tr><td>Dec 2023</td><td>4</td><td>3.2</td><td>4</td><td>3.8</td></tr>"
        "<tr><td>Jan 2024</td><td>3.7</td><td>3.1</td><td>3.7</td><td>3.5</td></tr>"
        "</tbody></table>\n"
    )
    res = score(md, [chart])
    assert res.score == 1.0 and kinds(res) == {}


def test_blank_cell_for_a_zero_segment_is_correct():
    # a stacked segment of 0 is not drawn; the output leaves its cell empty
    chart = {
        "title": "Tariff rates",
        "chart_type": "stacked_bar",
        "data_labels_shown": True,
        "categories": ["IND", "BRA"],
        "series": [
            {"name": "Compliance", "values": ["0", "0", "1.5"]},
            {"name": "New tariffs", "values": ["24.7", "21.3", "9"]},
        ],
    }
    chart["categories"] = ["IND", "BRA", "LUX"]
    rows = [["Country", "Compliance", "New tariffs"], ["IND", "", "24.7"], ["BRA", "-", "21.3"], ["LUX", "1.5", "9"]]
    assert score(table(rows, title="Tariff rates"), [chart]).score == 1.0
    # a blank where the reference is not 0 is still missing (here the whole column is empty: the series)
    rows = [["Country", "Compliance", "New tariffs"], ["IND", "", "24.7"], ["BRA", "", "21.3"], ["LUX", "", "9"]]
    assert kinds(score(table(rows, title="Tariff rates"), [chart])) == {"series missing": 1}


def test_figure_title_with_panel_name():
    # title "<figure title> – Brazil" = page heading + the Brazil column
    chart = {
        "title": "PROPORTION INTERESTED IN SPORT (2015–22) – Brazil",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["2015", "2022"],
        "series": [{"name": "Brazil", "values": ["82", "57"]}],
    }
    md = "## PROPORTION INTERESTED IN SPORT (2015–22)\n\n" + table(
        [["Year", "Brazil"], ["2015", "82"], ["2022", "57"]], None
    )
    assert score(md, [chart]).score == 1.0


def test_series_named_by_their_distinguishing_words():
    # "Nominal" / "Real" for "Nominal minimum wage" / "Real minimum wage"
    chart = {
        "title": "Australia",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["May-19", "Nov-19", "May-20"],
        "series": [
            {"name": "Nominal minimum wage", "values": ["0", "2.9", "2.9"]},
            {"name": "Real minimum wage", "values": ["0", "2.0", "3.1"]},
        ],
    }
    rows = [["Date", "Nominal", "Real"], ["May-19", "0", "0"], ["Nov-19", "2.9", "1.5"], ["May-20", "2.9", "3.1"]]
    res = score(table(rows, title="Australia"), [chart])
    assert kinds(res) == {"wrong value": 1}  # the misread point, not a lost series plus an invented one


def test_header_written_as_the_first_body_row():
    # a "[tsv]" placeholder header, the real header in the first row
    rows = [["[tsv]", "[tsv]", "[tsv]"], *GOOD]
    assert score(table(rows)).score == 1.0


def test_generic_corner_or_value_axis_name_is_not_a_mislabeled_axis():
    rows = with_cell(0, 0, "Category")
    assert score(table(rows)).score == 1.0
    value_axis = {**REVENUE, "x_label": "Revenue (USD m)"}  # a horizontal bar chart's value-axis title
    assert score(table(with_cell(0, 0, "Region")), [value_axis]).score == 1.0


def test_stop_words_do_not_reword_a_label():
    # "Score in 2018" written "Score 2018"
    chart = {**REVENUE, "series": [{"name": "Score in 2018", "values": ["412.5", "430", "455"]}, REVENUE["series"][1]]}
    assert score(table(with_cell(0, 1, "Score 2018")), [chart]).score == 1.0


def test_colour_group_series_merged_into_one_column_is_one_minor():
    # bars coloured by group (one series per group, disjoint countries) in one output column
    chart = {
        "title": "Innovation scores",
        "data_labels_shown": True,
        "categories": ["Ireland", "Denmark", "Malta", "Italy"],
        "series": [
            {"name": "Strong innovators", "values": ["251", "", "", ""]},
            {"name": "Innovation leaders", "values": ["", "191", "", ""]},
            {"name": "Moderate innovators", "values": ["", "", "183", "172"]},
            {"name": "Score in 2018", "values": ["166", "138", "135", "146"]},
        ],
    }
    rows = [["Country", "Score in 2025", "Score in 2018"], ["Ireland", "251", "166"], ["Denmark", "191", "138"]]
    rows += [["Malta", "183", "135"], ["Italy", "172", "146"]]
    res = score(table(rows, title="Innovation scores"), [chart])
    assert kinds(res) == {"colour groups merged": 1} and res.score == pytest.approx(0.99)


def test_change_column_under_a_spanning_header_is_not_a_repeated_point():
    # "DGI" spans the value and its % change
    single = {**REVENUE, "series": [{"name": "DGI", "values": ["412.5", "430", "455"]}]}
    md = (
        "## Quarterly revenue by region\n\n<table><thead><tr><th>Quarter</th><th colspan='2'>DGI</th></tr></thead>"
        "<tbody><tr><td>Q1 2023</td><td>412.5</td><td>4.6%</td></tr><tr><td>Q2 2023</td><td>430</td><td>4.2%</td></tr>"
        "<tr><td>Q3 2023</td><td>455</td><td>5.8%</td></tr></tbody></table>\n"
    )
    assert "repeated point" not in kinds(score(md, [single]))


def test_partially_labeled_chart_uses_the_axis_tolerance_for_estimated_points():
    # only the last points are printed; the sparse rules
    # mark the others "need_estimate", so they are read with the axis (range) tolerance
    chart = {
        "title": "Storm insured losses",
        "chart_type": "line",
        "unit": "USD bn",
        "data_labels_shown": True,
        "categories": ["2000", "2010", "2025"],
        "series": [{"name": "United States", "values": ["8", "138", "648"]}],
    }
    rules = [
        {"type": "chart_data_point", "labels": ["2025", "United States"], "value": "648"},
        {"type": "chart_data_point", "labels": ["2010", "United States"], "value": "138", "tags": ["need_estimate"]},
    ]
    md = table(
        [["Year", "United States"], ["2000", "10"], ["2010", "130"], ["2025", "648"]], title="Storm insured losses"
    )
    labeled = score_chart_next(md, [chart])
    partial = score_chart_next(md, [chart], rules)
    assert labeled is not None and partial is not None
    assert labeled.events.critical >= 2 and partial.score == 1.0
    md = md.replace("| 648 |", "| 600 |")  # the printed label keeps the 1% tolerance
    res = score_chart_next(md, [chart], rules)
    assert res is not None and res.events.critical == 1


def test_category_written_as_its_unique_code_or_name_part():
    # "AR: ARGENTINA" written "AR"; a part shared by two labels does not bind on its own
    chart = {
        "title": "Energy use per capita",
        "data_labels_shown": True,
        "categories": ["AR: ARGENTINA", "BR: BRAZIL", "CA: CANADA"],
        "series": [
            {"name": "Energy use", "values": ["64", "51", "365"]},
            {"name": "GDP", "values": ["2600", "2300", "11500"]},
        ],
    }
    rows = [["Country", "Energy use", "GDP"], ["AR", "64", "2600"], ["BR", "51", "2300"], ["CA", "365", "11500"]]
    assert score(table(rows, title="Energy use per capita"), [chart]).score == 1.0


def test_trailing_zero_decimal_in_a_dual_unit_chart_is_exact():
    # unit "% ; USD (2024)/bbl": the WTI column in USD/bbl is not held to "%", and "30" for "30.0" is exact
    chart = {
        "title": "Car imports and oil price",
        "unit": "% ; USD (2024)/bbl",
        "data_labels_shown": False,
        "categories": ["1964", "1965", "1966"],
        "series": [
            {"name": "Imports as share of demand", "values": ["0.2", "0.3", "0.6"]},
            {"name": "WTI price", "values": ["30.0", "29.1", "28.2"]},
        ],
    }
    rows = [
        ["Year", "Imports as share of demand", "WTI price (USD/bbl)"],
        ["1964", "0.2", "30"],
        ["1965", "0.3", "29.1"],
    ]
    rows.append(["1966", "0.6", "28.2"])
    assert score(table(rows, title="Car imports and oil price"), [chart]).score == 1.0


def test_series_unit_takes_the_scale_of_the_matching_chart_unit():
    # "Deal Value ($)" in a "# of transactions; $ in bn" chart
    chart = {
        "title": "Deal volume and value",
        "unit": "# of transactions; $ in bn",
        "data_labels_shown": True,
        "categories": ["2020 Q2", "2020 Q3", "2021 Q1"],
        "series": [
            {"name": "Deal Volume (#)", "values": ["806", "1090", "1528"]},
            {"name": "Deal Value ($)", "values": ["24.9", "81.3", "66.1"]},
        ],
    }
    # the category split over a Quarter and a Year column
    rows = [["Quarter", "Year", "Deal Volume (#)", "Deal Value ($ bn)"], ["Q2", "2020", "806", "$24.9"]]
    rows += [["Q3", "2020", "1,090", "$81.3"], ["Q1", "2021", "1,528", "$66.1"]]
    assert score(table(rows, title="Deal volume and value"), [chart]).score == 1.0


def test_signed_percent_series_in_a_currency_chart():
    # a "+6%" change series inside a chart whose unit is Euro billions
    chart = {
        "title": "Premiums & Other Revenues",
        "chart_type": "stacked_bar",
        "unit": "Euro billions",
        "data_labels_shown": True,
        "categories": ["Commercial lines", "Retail lines"],
        "series": [{"name": "FY24", "values": ["34.9", "19.1"]}, {"name": "Change", "values": ["+6%", "+7%"]}],
    }
    rows = [["Segment", "FY24", "Change"], ["Commercial lines", "34.9", "+6%"], ["Retail lines", "19.1", "+7%"]]
    assert score(table(rows, title="Premiums & Other Revenues"), [chart]).score == 1.0


def test_compound_count_and_share_cells():
    # the reference stores "32 (16.6%)"
    chart = {
        "title": "Countries per DGI group",
        "unit": "countries",
        "data_labels_shown": True,
        "categories": ["2014", "2016"],
        "series": [
            {"name": "Low DGI", "values": ["32 (16.6%)", "15 (7.8%)"]},
            {"name": "High DGI", "values": ["62 (32.1%)", "75 (38.9%)"]},
        ],
    }
    rows = [
        ["Year", "Low DGI", "High DGI"],
        ["2014", "32 (16.6%)", "62 (32.1%)"],
        ["2016", "15 (7.8%)", "75 (38.9%)"],
    ]
    assert score(table(rows, title="Countries per DGI group"), [chart]).score == 1.0
    counts = [["Year", "Low DGI", "High DGI"], ["2014", "32", "62"], ["2016", "15", "75"]]
    assert score(table(counts, title="Countries per DGI group"), [chart]).score == 1.0
    # count and share are two reference series, the output writes "135 countries, 70%" (series down the rows)
    two = {
        "title": "E-procurement platforms",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["World", "Africa"],
        "series": [
            {"name": "2024 number of countries", "values": ["135", "21"]},
            {"name": "2024 percentage of countries", "values": ["70", "39"]},
        ],
    }
    rows = [["Year", "World", "Africa"], ["2024", "135 countries, 70%", "21 countries, 39%"]]
    res = score(table(rows, title="E-procurement platforms"), [two])
    assert res.events.critical == res.events.major == 0 and res.proportional == 1.0


def test_reference_value_n_a_is_not_a_point():
    # "N/A" printed for one bar
    chart = {**REVENUE, "series": [{"name": "North", "values": ["412.5", "N/A", "455"]}, REVENUE["series"][1]]}
    assert score(table(with_cell(2, 1, "N/A")), [chart]).score == 1.0


def test_integer_value_column_headed_like_a_label_next_to_unique_rows():
    # integer columns headed "25th percentile", "<30 years" or "Start year" hold values, not labels
    chart = {
        "title": "Debt ratios",
        "data_labels_shown": False,
        "categories": ["2001", "2002", "2003"],
        "series": [
            {"name": "25th percentile", "values": ["45", "44", "46"]},
            {"name": "Median", "values": ["58", "60", "64"]},
        ],
    }
    rows = [["Year", "25th percentile", "Median"], ["2001", "45", "58"], ["2002", "44", "60"], ["2003", "46", "64"]]
    assert score(table(rows, title="Debt ratios"), [chart]).score == 1.0


def test_decimal_comma_percent_is_not_a_compound_cell():
    # "18,7%" is 18.7 with a European decimal comma, not "18" plus "7%"
    share = {
        "title": "Market share",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["A", "B", "C"],
        "series": [{"name": "Share", "values": ["18.7", "41.3", "40"]}],
    }
    rows = [["Company", "Share"], ["A", "18,7%"], ["B", "41,3%"], ["C", "40%"]]
    assert score(table(rows, title="Market share"), [share]).score == 1.0


def test_wrong_category_kind_in_the_corner_is_a_mislabeled_axis():
    # the reference axis is "Quarter", the output corner names another kind of category
    res = score(table(with_cell(0, 0, "Region")))
    assert kinds(res) == {"category axis mislabeled": 1}
    assert score(table(with_cell(0, 0, "Category"))).score == 1.0  # a placeholder names no kind


def test_word_hedged_value_column_is_not_a_label_column():
    # every cell "approximately ..." (one plain) - the column holds values, not labels
    rows = [GOOD[0], ["Q1 2023", "approximately 412.5", "about 300"], ["Q2 2023", "approximately 430", "about 310.2"]]
    rows.append(["Q3 2023", "455", "299"])
    assert score(table(rows)).score == 1.0


def test_bar_series_swapped_with_its_own_trendline_is_wrong():
    # the column named like the bars holds the trendline's values, and vice versa
    chart = {
        "title": "Sales",
        "data_labels_shown": True,
        "categories": ["2021", "2022", "2023"],
        "series": [
            {"name": "Sales", "values": ["10", "14", "21"]},
            {"name": "Linear (Sales)", "values": ["9.5", "15", "20.5"]},
        ],
    }
    rows = [["Year", "Sales", "Linear (Sales)"], ["2021", "9.5", "10"], ["2022", "15", "14"], ["2023", "20.5", "21"]]
    res = score(table(rows, title="Sales"), [chart])
    assert res.events.critical >= 3 and res.score <= 0.5
    good = [["Year", "Sales", "Linear (Sales)"], ["2021", "10", "9.5"], ["2022", "14", "15"], ["2023", "21", "20.5"]]
    assert score(table(good, title="Sales"), [chart]).score == 1.0


def test_bound_words_in_a_label_column_stay_labels():
    # "Above 60%" / "Below 60%" are labels
    chart = {
        "title": "Unable to keep home warm",
        "unit": "%",
        "data_labels_shown": True,
        "categories": [
            "Single parent - Above 60%",
            "Single parent - Below 60%",
            "Couple - Above 60%",
            "Couple - Below 60%",
        ],
        "series": [
            {"name": "2021", "values": ["6.9", "18.8", "4.3", "17.1"]},
            {"name": "2022", "values": ["12.8", "22.3", "6.7", "21.1"]},
        ],
    }
    rows = [["Household", "Threshold", "2021", "2022"], ["Single parent", "Above 60%", "6.9", "12.8"]]
    rows += [
        ["Single parent", "Below 60%", "18.8", "22.3"],
        ["Couple", "Above 60%", "4.3", "6.7"],
        ["Couple", "Below 60%", "17.1", "21.1"],
    ]
    assert score(table(rows, title="Unable to keep home warm"), [chart]).score == 1.0


def test_unitless_series_next_to_series_with_units_is_not_held_to_the_chart_unit():
    # "Payout ratio" in % in a combo chart whose unit is EUR, the other series naming their own unit
    # ("Earnings per share (EUR)")
    chart = {
        "title": "Earnings and dividend per share",
        "unit": "EUR",
        "data_labels_shown": True,
        "categories": ["2023", "2024"],
        "series": [
            {"name": "Earnings per share (EUR)", "values": ["5.15", "5.02"]},
            {"name": "Payout ratio", "values": ["48.5", "51.8"]},
        ],
    }
    rows = [["Year", "Earnings per share (EUR)", "Payout ratio"], ["2023", "5,15", "48,5%"], ["2024", "5,02", "51,8%"]]
    assert score(table(rows, title="Earnings and dividend per share"), [chart]).score == 1.0


# --- adversarial parser errors ---------------------------------------------------------------------------------

QUARTERS = {
    **REVENUE,
    "categories": ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"],
    "series": [
        {"name": "North", "values": ["412.5", "430", "455", "470"]},
        {"name": "South", "values": ["300", "310.2", "299", "320"]},
    ],
}
Q_ROWS = [
    ["Quarter", "North", "South"],
    ["Q1 2023", "412.5", "300"],
    ["Q2 2023", "430", "310.2"],
    ["Q3 2023", "455", "299"],
    ["Q4 2023", "470", "320"],
]


def test_adversarial_series_swapped_by_a_confused_legend_is_critical():
    rows = [Q_ROWS[0], *([r[0], r[2], r[1]] for r in Q_ROWS[1:])]
    res = score(table(rows), [QUARTERS])
    assert kinds(res) == {"wrong value": 8} and res.score == 0.0


def test_adversarial_categories_shifted_by_one_is_critical():
    rows = [Q_ROWS[0], ["Q1 2023", "", ""], *([Q_ROWS[k + 1][0], *Q_ROWS[k][1:]] for k in range(1, 4))]
    res = score(table(rows), [QUARTERS])
    assert res.events.critical >= 3 and res.score < 0.5


def test_adversarial_stacked_segments_reported_as_cumulative_totals_is_critical():
    stacked = {**QUARTERS, "chart_type": "stacked_bar"}
    rows = [Q_ROWS[0], *([r[0], r[1], f"{float(r[1]) + float(r[2]):g}"] for r in Q_ROWS[1:])]
    res = score(table(rows), [stacked])
    assert kinds(res) == {"wrong value": 4} and res.score == pytest.approx(0.5)


def test_adversarial_pie_slice_dropped_and_renormalized_is_critical():
    pie = {
        "title": "Market share 2023",
        "chart_type": "pie",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["A", "B", "C", "D"],
        "series": [{"name": "Share", "values": ["40", "30", "20", "10"]}],
    }
    rows = [["Company", "Share"], ["A", "44.4%"], ["B", "33.3%"], ["C", "22.2%"]]
    res = score(table(rows, title="Market share 2023"), [pie])
    assert kinds(res) == {"wrong value": 3, "point missing": 1} and res.score == 0.0


def test_adversarial_secondary_axis_read_on_the_primary_scale_is_critical():
    dual = {
        "title": "Revenue and margin",
        "chart_type": "combo",
        "data_labels_shown": False,
        "categories": ["2020", "2021", "2022", "2023"],
        "series": [
            {"name": "Revenue (USD bn)", "values": ["100", "120", "150", "160"]},
            {"name": "Margin (%, right axis)", "values": ["10", "12", "15", "14"]},
        ],
    }
    rows = [["Year", "Revenue (USD bn)", "Margin (%)"], ["2020", "100", "100"], ["2021", "120", "120"]]
    rows += [["2022", "150", "150"], ["2023", "160", "140"]]
    res = score(table(rows, title="Revenue and margin"), [dual])
    assert kinds(res) == {"wrong value": 4} and res.score == pytest.approx(0.5)


def test_adversarial_reverse_chronological_rows_are_harmless():
    rows = [Q_ROWS[0], *reversed(Q_ROWS[1:])]
    assert score(table(rows), [QUARTERS]).score == 1.0


def test_adversarial_values_copied_from_the_neighbouring_chart_are_critical():
    costs = {
        **QUARTERS,
        "title": "Quarterly costs by region",
        "series": [
            {"name": "North", "values": ["200", "210", "220", "230"]},
            {"name": "South", "values": ["150", "155", "160", "170"]},
        ],
    }
    md = table(Q_ROWS) + "\n" + table(Q_ROWS, title="Quarterly costs by region")
    res = score(md, [QUARTERS, costs])
    assert kinds(res) == {"wrong value": 8} and res.score < 0.5


# --- several charts merged into one output table ------------------------------------------------------------------

TENURE = [
    {
        "title": title,
        "data_labels_shown": True,
        "unit": "%",
        "categories": ["Less than 1 year", "1-3 years", "4-5 years"],
        "series": [{"name": "2024", "values": a}, {"name": "2022", "values": b}],
    }
    for title, a, b in (
        ("Years as a partner", ["2", "9", "12"], ["6", "13", "18"]),
        ("Years with the firm", ["8", "26", "19"], ["11", "22", "17"]),
    )
]
# a panel of text categories next to panels whose categories read as numbers (years, ranges, projections)
MIXED = [
    {"title": t, "data_labels_shown": True, "categories": c, "series": [{"name": "Contribution ratio", "values": v}]}
    for t, c, v in (
        ("By maturity level", ["Initiated", "Piloted", "Deployed"], ["15.4", "26.4", "37.7"]),
        ("Five year trend", ["2017", "2018", "2029 P"], ["31.7", "34.7", "35.4"]),
        ("By wage percentile", ["5", "10", "96-99"], ["2.76", "1.79", "0.92"]),
    )
]


def _merged_layout(name: str) -> str:
    cats, years = TENURE[0]["categories"], ["2024", "2022"]
    if name == "long":  # Chart | Category | Series | Value
        rows = [["Chart", "Category", "Series", "Value"]]
        rows += [
            [c["title"], cat, c["series"][0]["name"], v]
            for c in MIXED
            for cat, v in zip(c["categories"], c["series"][0]["values"], strict=True)
        ]
        return table(rows, title=None)
    if name == "row groups":  # the chart in a rowspan cell
        html = "<table><tr><th>Chart</th><th>Category</th><th>Contribution ratio</th></tr>"
        for c in MIXED:
            for i, (cat, v) in enumerate(zip(c["categories"], c["series"][0]["values"], strict=True)):
                lead = f"<th rowspan='3'>{c['title']}</th>" if i == 0 else ""
                html += f"<tr>{lead}<td>{cat}</td><td>{v}</td></tr>"
        return html + "</table>"
    if name == "one table per series":  # charts as columns, the series in the heading
        return "\n".join(
            table(
                [["", *(c["title"] for c in TENURE)]]
                + [[cat, *(c["series"][k]["values"][i] for c in TENURE)] for i, cat in enumerate(cats)],
                title=y,
            )
            for k, y in enumerate(years)
        )
    if name == "side by side":  # each chart its own column group: label column + values, under its title
        html = "<table><tr>" + "".join(f"<th colspan='2'>{c['title']}</th>" for c in MIXED) + "</tr><tr>"
        html += "<th>Category</th><th>Contribution ratio</th>" * len(MIXED) + "</tr>"
        for i in range(3):
            html += "<tr>" + "".join(
                f"<td>{c['categories'][i]}</td><td>{c['series'][0]['values'][i]}</td>" for c in MIXED
            )
            html += "</tr>"
        return html + "</table>"
    if name == "header sections":  # one <thead> per chart, stacked in one <table>
        html = "<table>"
        for c in MIXED:
            html += f"<thead><tr><th colspan='2'>{c['title']}: a long panel title over its own column headers</th></tr>"
            html += "<tr><th>Category</th><th>Contribution ratio</th></tr></thead><tbody>"
            html += "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in zip(c["categories"], c["series"][0]["values"], strict=True)
            )
            html += "</tbody>"
        return html + "</table>"
    # charts as columns, one line per series in each cell, the legend above the table
    html = "**2024** **2022**\n\n<table><tr><th></th>" + "".join(f"<th>{c['title']}</th>" for c in TENURE) + "</tr>"
    for i, cat in enumerate(cats):
        cells = "".join("<td>" + "<br />".join(s["values"][i] for s in c["series"]) + "</td>" for c in TENURE)
        html += f"<tr><td>{cat}</td>{cells}</tr>"
    return html + "</table>"


@pytest.mark.parametrize(
    "layout", ["long", "row groups", "one table per series", "stacked cells", "side by side", "header sections"]
)
def test_charts_merged_into_one_output_table_with_every_value_right_score_one(layout):
    """A key column of years or ranges was read as values (long, row groups); one table per series, ``2<br>6``
    cells, charts side by side with their own label columns and one <thead> section per chart were read as missing
    charts or series."""
    charts = MIXED if layout in ("long", "row groups", "side by side", "header sections") else TENURE
    res = score(_merged_layout(layout), charts)
    assert res.charts_found == len(charts) and res.score == 1.0, kinds(res)


def test_stacked_cells_without_a_legend_line_lose_the_legend():
    md = _merged_layout("stacked cells").replace("**2024** **2022**\n\n", "")
    assert kinds(score(md, TENURE)).get("legend lost") == 2


def test_space_grouped_thousands_are_one_number_not_stacked_cells():
    chart = {
        "title": "Merger notifications",
        "categories": ["2015", "2016"],
        "series": [{"name": "Members", "values": ["7300", "7500"]}],
    }
    md = table([["Year", "Members"], ["2015", "~7 300"], ["2016", "~7 500"]], title="Merger notifications")
    assert score(md, [chart]).score == 1.0


def test_a_panel_does_not_read_the_rows_of_another_panel():
    """A panel whose own rows are wrong was scored on the rows of a neighbouring panel that happened to fit."""
    rows = [["Panel", "Year", "North", "South"], ["Steel", "2021", "20", "5"], ["Steel", "2022", "21", "6"]]
    rows += [["Iron", "2021", "10", "1"], ["Iron", "2022", "11", "2"]]
    charts = [
        {
            "title": t,
            "data_labels_shown": True,
            "categories": ["2021", "2022"],
            "series": [{"name": "North", "values": n}, {"name": "South", "values": s_}],
        }
        for t, n, s_ in (("Steel", ["10", "11"], ["1", "2"]), ("Iron", ["30", "31"], ["7", "8"]))
    ]
    assert kinds(score(table(rows, title="Production"), charts)) == {"wrong value": 8}


def test_trendline_column_is_never_invented():
    """A reference that leaves the trendline out charged the parser's ``Linear (...)`` column as invented data."""
    charts = [
        {"title": t, "data_labels_shown": True, "categories": ["IF", "CP"], "series": [{"name": t, "values": v}]}
        for t, v in (("Africa", ["0.68", "0.48"]), ("Asia", ["0.85", "0.74"]))
    ]
    html = "<table><tr><th rowspan='2'>Subindex</th><th colspan='2'>Africa</th><th colspan='2'>Asia</th></tr>"
    html += "<tr><th>Value</th><th>Linear (average score for Africa)</th>"
    html += "<th>Value</th><th>Linear (average score for Asia)</th></tr>"
    html += "<tr><th>IF</th><td>0.68</td><td>0.38</td><td>0.85</td><td>0.64</td></tr>"
    html += "<tr><th>CP</th><td>0.48</td><td>0.38</td><td>0.74</td><td>0.64</td></tr></table>"
    assert score(html, charts).score == 1.0


def test_category_split_over_row_and_column_is_read_cell_by_cell():
    """``Green occupations - Total`` printed at row ``Total``, column ``Green occupations`` was a missing point."""
    chart = {
        "title": "Marginal effect by occupation",
        "data_labels_shown": True,
        "categories": [
            "Green new - Total",
            "Green new - Low-skill",
            "Green enhanced - Total",
            "Green enhanced - Low-skill",
        ],
        "series": [{"name": "Marginal effect", "values": ["2.7", "-3.8", "0.4", "9.9"]}],
    }
    rows = [["Skill level", "Green new", "Green enhanced"], ["Total", "2.7", "0.4"], ["Low-skill", "-3.8", "9.9"]]
    assert score(table(rows, title="Marginal effect by occupation"), [chart]).score == 1.0


def test_two_digit_year_tick_binds_a_finer_period_of_that_year():
    """A line chart annotated at its yearly ticks (``19``, ``20``) found nothing in ``2019 (Start)`` rows."""
    chart = {
        "title": "Consumer prices",
        "chart_type": "line",
        "data_labels_shown": False,
        "categories": ["19", "20"],
        "series": [{"name": "Total", "values": ["1.6", "2.5"]}],
    }
    rows = [["Period", "Total"], ["2019 (Start)", "1.6"], ["2019 (Mid)", "1.8"], ["2020 (Start)", "2.5"]]
    assert score(table(rows, title="Consumer prices"), [chart]).score == 1.0


def test_optional_series_is_scored_when_given_never_required_and_never_invented():
    """A series flagged ``optional`` (the 100 - Yes of a Yes/No bar, a reference line) was required, and the
    parser's row for it was charged as invented."""
    chart = {
        "title": "Countries with an open data portal",
        "data_labels_shown": True,
        "categories": ["SIDS", "LDCs"],
        "series": [{"name": "Yes", "values": ["73", "67"]}, {"name": "No", "values": ["27", "33"], "optional": True}],
    }
    only_yes = table([["Group", "Yes"], ["SIDS", "73"], ["LDCs", "67"]], title=chart["title"])
    assert score(only_yes, [chart]).score == 1.0
    wrong_no = table([["Group", "Yes", "No"], ["SIDS", "73", "50"], ["LDCs", "67", "33"]], title=chart["title"])
    assert kinds(score(wrong_no, [chart])) == {"wrong value": 1}
    as_a_row = table([["Series", "SIDS", "LDCs"], ["Yes", "73", "67"], ["No", "27", "33"]], title=chart["title"])
    assert score(as_a_row, [chart]).score == 1.0


def test_header_rowspan_stops_at_the_end_of_thead():
    # a rowspan=2 corner cell in a one-row <thead> covers the header only, as browsers render it
    md = (
        "## Quarterly revenue by region\n\n<table><thead><tr><th rowspan='2'>Quarter</th><th>North</th>"
        "<th>South</th></tr></thead><tbody>"
        + "".join(f"<tr><th>{r[0]}</th><td>{r[1]}</td><td>{r[2]}</td></tr>" for r in GOOD[1:])
        + "</tbody></table>\n"
    )
    assert score(md).score == 1.0


def test_colspan_copies_of_one_value_are_one_point():
    md = (
        "## Quarterly revenue by region\n\n<table><thead><tr><th>Quarter</th><th colspan='2'>North</th>"
        "<th colspan='2'>South</th></tr></thead><tbody>"
        + "".join(f"<tr><th>{r[0]}</th><td colspan='2'>{r[1]}</td><td colspan='2'>{r[2]}</td></tr>" for r in GOOD[1:])
        + "</tbody></table>\n"
    )
    assert score(md).score == 1.0


def test_percentage_point_columns_read_on_a_percent_chart():
    chart = {**REVENUE, "unit": "%"}
    rows = [["Quarter", "North (percentage points)", "South (percentage points)"], *GOOD[1:]]
    assert score(table(rows), [chart]).score == 1.0


def test_daily_axis_bare_days_take_the_month_above_them():
    chart = {
        "title": "Futures price performance",
        "chart_type": "line",
        "unit": "%",
        "data_labels_shown": True,
        "categories": ["Jan. 2", "Jan. 5", "Jan. 6", "Feb. 2", "Feb. 3"],
        "series": [{"name": "Silver", "values": ["1", "4", "9", "12", "15"]}],
    }
    days = [["Day", "Silver"], ["Jan. 2", "1"], ["5", "4"], ["6", "9"], ["Feb. 2", "12"], ["3", "15"]]
    assert score(table(days, title="Futures price performance"), [chart]).score == 1.0
    header = [["Date (Jan. 2026)", "Silver"], ["2", "1"], ["5", "4"], ["6", "9"]]
    three = {**chart, "categories": chart["categories"][:3], "series": [{"name": "Silver", "values": ["1", "4", "9"]}]}
    assert score(table(header, title="Futures price performance"), [three]).score == 1.0
    # "Income decile" names no month: the deciles stay numbers
    deciles = {**three, "categories": ["1", "2", "3"]}
    rows = [["Income decile (2026)", "Silver"], ["1", "1"], ["2", "4"], ["3", "9"]]
    assert score(table(rows, title="Futures price performance"), [deciles]).score == 1.0


def test_typeset_year_marks_read_as_years():
    assert full_label("’20") == full_label("'20") == "2020"
