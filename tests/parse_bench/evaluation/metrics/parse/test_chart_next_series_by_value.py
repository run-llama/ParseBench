"""chart_next: series whose legend names do not bind are assigned to output columns by their values."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.chart_next import score_chart_next

CATS = ["1960-64", "1965-69", "1970-74", "1975-79", "1980-84", "1985-89", "1990-94", "1995-99"]
EDU = ["-14.1", "-11.2", "-10.8", "-7.6", "-3.2", "-3.7", "-0.4", "2.1"]
INC = ["-10.6", "-14.4", "-14.1", "-15.7", "-12.9", "-16.0", "-13.3", "-10.1"]
CHART = {
    "title": "Education and income gaps, 1960-1999",
    "chart_type": "line",
    "unit": "%",
    "data_labels_shown": False,
    "categories": CATS,
    "series": [
        {"name": "(% of top 10% educated buying online) minus (% of bottom 90% educated buying online)", "values": EDU},
        {"name": "(% of top 10% earners buying online) minus (% of bottom 90% earners buying online)", "values": INC},
    ],
}
CALLOUTS = ["Higher-educated shoppers buying online-only brands", "Top-income shoppers buying store-only brands"]


def page(header: list[str], cols: list[list[str]]) -> str:
    rows = [header, *([c, *(col[i] for col in cols)] for i, c in enumerate(CATS))]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return f"## {CHART['title']}\n\n" + "\n".join(lines) + "\n"


def test_columns_named_after_callouts_bind_their_series_by_value() -> None:
    # integer readings off the axis: about half the points fall outside the 2%-of-range tolerance
    edu = ["-14", "-10", "-10", "-7", "-3", "-4", "0", "3"]
    inc = ["-10", "-15", "-15", "-16", "-14", "-16", "-13", "-9"]
    res = score_chart_next(page(["Period", *CALLOUTS], [edu, inc]), [CHART])
    assert res is not None
    assert res.metadata()["charts_found"] == 1
    assert res.events.kinds().get("series renamed") == 2
    assert 0.3 < res.score < 0.9


def test_columns_with_other_values_stay_unbound() -> None:
    # the same callout names over values that follow neither series: still a missing chart
    other = ["5", "8", "10", "12", "15", "18", "20", "22"]
    res = score_chart_next(page(["Period", *CALLOUTS], [other, list(reversed(other))]), [CHART])
    assert res is not None
    assert res.score == 0.0


def test_one_series_following_its_values_is_not_enough() -> None:
    # all or nothing: the education column follows its series, the income column follows neither
    edu = ["-14", "-10", "-10", "-7", "-3", "-4", "0", "3"]
    other = ["5", "8", "10", "12", "15", "18", "20", "22"]
    res = score_chart_next(page(["Period", *CALLOUTS], [edu, other]), [CHART])
    assert res is not None
    assert res.score == 0.0


def test_reference_names_still_score_one() -> None:
    names = [s["name"] for s in CHART["series"]]
    res = score_chart_next(page(["Period", *names], [EDU, INC]), [CHART])
    assert res is not None
    assert res.score == 1.0
