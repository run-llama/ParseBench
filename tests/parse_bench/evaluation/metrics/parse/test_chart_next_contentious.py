"""chart_next reads correct tables that spread a point's labels or several series over unusual cell layouts.

Each layout comes from a ParseBench v2 chart page where a correct transcription scored near 0: a Year x Q1..Q4
grid, a placeholder ``Value`` header, a markdown header that writes each year once over its quarters, and
small-multiple cells holding two named series. Each test pairs the correct reading with wrong ones that must
still be charged.
"""

from __future__ import annotations

from collections.abc import Sequence

from parse_bench.evaluation.metrics.parse.chart_next import score_chart_next

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
SHARES = "78 83 -89 37 -5 -24 -69 35 45 30 30 31".split()

# unlabeled quarterly bars: one series, categories "2007 Q1" ... "2009 Q4"
TRADING = {
    "title": "Figure 1. Bank Trading Revenue Share",
    "chart_type": "bar",
    "data_labels_shown": False,
    "unit": "%",
    "categories": [f"{y} {q}" for y in (2007, 2008, 2009) for q in QUARTERS],
    "series": [{"name": "value", "values": SHARES}],
}


def year_grid(values: Sequence[str], years: tuple[int, ...] = (2007, 2008, 2009)) -> str:
    rows = "".join(f"| {y} | " + " | ".join(values[4 * k : 4 * k + 4]) + " |\n" for k, y in enumerate(years))
    return "**Figure 1. Bank Trading Revenue Share**\n\n| Year | Q1 | Q2 | Q3 | Q4 |\n|---|---|---|---|---|\n" + rows


def test_year_by_quarter_grid_binds_each_cell_to_its_year_and_quarter():
    res = score_chart_next(year_grid(SHARES), [TRADING])
    assert res is not None and res.score == 1.0


def test_year_by_quarter_grid_with_swapped_quarters_is_charged():
    v = SHARES
    swapped = [v[i ^ 2] for i in range(len(v))]  # Q1 <-> Q3, Q2 <-> Q4 within each year
    res = score_chart_next(year_grid(swapped), [TRADING])
    assert res is not None and res.score < 0.5
    assert any(kind == "wrong value" for kind, *_ in res.events.examples)


def test_year_by_quarter_grid_with_shifted_years_is_charged():
    res = score_chart_next(year_grid(SHARES, years=(2008, 2009, 2010)), [TRADING])
    assert res is not None and res.score < 0.5


# printed labels, two series in different units; the reference names them "Deal Volume (#)" / "Deal Value ($)"
QUARTER_LABELS = ["2020 Q2", "2020 Q3", "2020 Q4", "2021 Q1", "2021 Q2"]
VOLUME = ["806", "1090", "1398", "1528", "1352"]
DEAL_VALUE = ["24.9", "81.3", "141.3", "66.1", "80.3"]
DEALS = {
    "title": "Announced Deal Volume and Value",
    "chart_type": "bar",
    "data_labels_shown": True,
    "unit": "# of transactions; $ in bn",
    "categories": QUARTER_LABELS,
    "series": [{"name": "Deal Volume (#)", "values": VOLUME}, {"name": "Deal Value ($)", "values": DEAL_VALUE}],
}


def long_table(header: tuple[str, str], first: list[str], second: list[str]) -> str:
    rows = "".join(f"| {c} | {a} | ${b} |\n" for c, a, b in zip(QUARTER_LABELS, first, second, strict=True))
    return f"## Announced Deal Volume and Value\n\n| Quarter | {header[0]} | {header[1]} |\n|---|---|---|\n" + rows


def test_placeholder_value_header_over_the_volumes_binds_by_values():
    # "Value" names no series: the volumes under it are the volume series with its legend lost (not 5 wrong
    # deal values), and the "Total" column holding the deal values is that series renamed (not a derived sum)
    res = score_chart_next(long_table(("Value", "Total"), VOLUME, DEAL_VALUE), [DEALS])
    assert res is not None and res.score >= 0.8
    kinds = {kind for kind, *_ in res.events.examples}
    assert "wrong value" not in kinds and "legend lost" in kinds


def test_placeholder_value_header_with_wrong_numbers_is_still_charged():
    wrong = ["412", "377", "905", "120", "640"]
    res = score_chart_next(long_table(("Value", "Total"), wrong, DEAL_VALUE), [DEALS])
    assert res is not None and res.score < 0.5


def test_short_legend_names_still_bind_by_name():
    res = score_chart_next(long_table(("Volume", "Value"), VOLUME, DEAL_VALUE), [DEALS])
    assert res is not None and res.score == 1.0


def spanned_years(volume: list[str], value: list[str]) -> str:
    """A markdown table that writes each year once and leaves its other quarters' top header cells blank."""
    return (
        "## Announced Deal Volume and Value\n\n"
        "| Series | 2020 |  |  | 2021 |  |\n|---|---|---|---|---|---|\n"
        "|  | Q2 | Q3 | Q4 | Q1 | Q2 |\n"
        f"| Deal Volume (#) | {' | '.join(volume)} |\n"
        f"| Deal Value ($) | {' | '.join('$' + v for v in value)} |\n"
    )


def test_blank_top_header_cells_take_the_year_written_to_their_left():
    res = score_chart_next(spanned_years(VOLUME, DEAL_VALUE), [DEALS])
    assert res is not None and res.score == 1.0


def test_blank_top_header_cells_with_values_in_the_wrong_quarter_are_charged():
    res = score_chart_next(spanned_years(VOLUME[1:] + VOLUME[:1], DEAL_VALUE[1:] + DEAL_VALUE[:1]), [DEALS])
    assert res is not None and res.score < 0.5


# an HTML header row that names fewer groups than the columns under it (one group left out)
GROUP_ITEMS = ["Digital ID", "Cloud", "AI", "Cybersecurity"]
GROUP_YES = {"SIDS": ["54", "32", "19", "65"], "LDCs": ["51", "13", "11", "64"], "LLDCs": ["72", "25", "22", "91"]}
GROUPS = {
    "title": "Legislative frameworks",
    "chart_type": "stacked_bar",
    "data_labels_shown": True,
    "unit": "%",
    "categories": GROUP_ITEMS,
    "series": [{"name": f"{g} - Yes", "values": v} for g, v in GROUP_YES.items()],
}


def test_short_html_header_row_does_not_span_its_padding():
    """``<th>SIDS</th><th>LLDCs</th>`` over six Yes/No columns: an HTML header can write a colspan, so the cells
    after "LLDCs" are not one, and the LDCs and LLDCs columns bind by their values (not all as "LLDCs": 0.80)."""
    head = "<tr><th></th><th>SIDS</th><th>LLDCs</th></tr><tr><th></th>" + "<th>Yes</th><th>No</th>" * 3 + "</tr>"
    body = "".join(
        f"<tr><td>{it}</td>" + "".join(f"<td>{v[k]}%</td><td></td>" for v in GROUP_YES.values()) + "</tr>"
        for k, it in enumerate(GROUP_ITEMS)
    )
    res = score_chart_next(f"**Legislative frameworks**\n\n<table>{head}{body}</table>", [GROUPS])
    assert res is not None and res.score > 0.95


# small multiples: one stacked bar per item and panel; "Machine-readable" is printed for one item only
ITEMS = ["OGD portal", "Real-time data", "Budget data"]


def panel(title: str, yes: list[str], machine: list[str]) -> dict:
    return {
        "title": title,
        "chart_type": "stacked_bar",
        "data_labels_shown": True,
        "unit": "%",
        "categories": ITEMS,
        "series": [{"name": "Yes", "values": yes}, {"name": "Machine-readable", "values": machine}],
    }


PANELS = [panel("SIDS", ["73", "14", "68"], ["", "5", "19"]), panel("LDCs", ["67", "16", "62"], ["", "11", "18"])]
FIGURE = "**Figure 2.36 Countries with OGD portals**\n\n"


def named_cells(cells: dict[str, list[str]]) -> str:
    rows = "".join(f"| {it} | {cells['SIDS'][k]} | {cells['LDCs'][k]} |\n" for k, it in enumerate(ITEMS))
    return FIGURE + "| Item | SIDS | LDCs |\n|---|---|---|\n" + rows


def test_cells_naming_two_series_are_read_as_one_value_per_series():
    md = named_cells(
        {
            "SIDS": ["73% Yes", "14% Yes, 5% Machine-readable", "68% Yes, 19% Machine-readable"],
            "LDCs": ["67% Yes", "16% Yes, 11% Machine-readable", "62% Yes, 18% Machine-readable"],
        }
    )
    res = score_chart_next(md, PANELS)
    assert res is not None and res.score == 1.0


def test_slash_cells_under_a_slash_header_are_read_as_one_value_per_series():
    rows = "".join(
        f"| {it} | {a}% / {b or '-'} | {c}% / {d or '-'} |\n"
        for it, a, b, c, d in zip(
            ITEMS, ["73", "14", "68"], ["", "5%", "19%"], ["67", "16", "62"], ["", "11%", "18%"], strict=True
        )
    )
    md = FIGURE + "| Item | SIDS (Yes / Machine-readable) | LDCs (Yes / Machine-readable) |\n|---|---|---|\n" + rows
    res = score_chart_next(md, PANELS)
    assert res is not None and res.score == 1.0


def test_named_cells_with_swapped_series_are_charged():
    md = named_cells(
        {
            "SIDS": ["73% Yes", "5% Yes, 14% Machine-readable", "19% Yes, 68% Machine-readable"],
            "LDCs": ["67% Yes", "11% Yes, 16% Machine-readable", "18% Yes, 62% Machine-readable"],
        }
    )
    res = score_chart_next(md, PANELS)
    assert res is not None and res.score < 0.5


def test_named_cells_with_a_wrong_second_value_are_charged():
    md = named_cells(
        {
            "SIDS": ["73% Yes", "14% Yes, 45% Machine-readable", "68% Yes, 19% Machine-readable"],
            "LDCs": ["67% Yes", "16% Yes, 11% Machine-readable", "62% Yes, 18% Machine-readable"],
        }
    )
    res = score_chart_next(md, PANELS)
    assert res is not None and res.score < 1.0
    assert ("wrong value", "Real-time data / Machine-readable: 5", "45%") in res.events.examples
