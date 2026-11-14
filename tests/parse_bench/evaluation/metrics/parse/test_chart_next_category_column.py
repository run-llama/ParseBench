"""chart_next: the category label column found after other label columns, in long tables and in grouped tables."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.chart_next import score_chart_next

MATURITIES = {
    "title": "Yields at issuance",
    "chart_type": "scatter",
    "unit": "%",
    "data_labels_shown": False,
    "categories": ["10yr", "20yr", "5yr", "30yr", "15yr"],
    "series": [
        {"name": "Bond", "values": ["-0.24", "0.13", "-0.51", "0.32", "-0.10"]},
        {"name": "Benchmark", "values": ["-0.34", "0.12", "-0.56", "0.26", "-0.36"]},
    ],
}

ISSUES = [
    ("Issue #1", "20/10/20", "10yr"),
    ("Issue #1", "20/10/20", "20yr"),
    ("Issue #2", "10/11/20", "5yr"),
    ("Issue #2", "10/11/20", "30yr"),
    ("Issue #3", "24/11/20", "15yr"),
]


def md(header: list[str], rows: list[list[str]], title: str) -> str:
    lines = [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join([*lines, *("| " + " | ".join(r) + " |" for r in rows)]) + "\n"


def maturity_table(bond: list[str]) -> str:
    rows = [[*k, b, m] for k, b, m in zip(ISSUES, bond, MATURITIES["series"][1]["values"], strict=True)]
    return md(["Issue", "Date", "Maturity", "Bond", "Benchmark"], rows, "Yields at issuance")


def test_maturity_column_after_other_label_columns_holds_the_categories():
    res = score_chart_next(maturity_table(MATURITIES["series"][0]["values"]), [MATURITIES])
    assert res is not None and res.score == 1.0


def test_maturity_table_with_wrong_values_is_still_charged():
    res = score_chart_next(maturity_table(["0.50", "0.13", "-0.51", "0.32", "-0.10"]), [MATURITIES])
    assert res is not None and res.events.kinds().get("wrong value") == 1


def _deciles(title: str, base: int) -> dict:
    return {
        "title": title,
        "chart_type": "line",
        "data_labels_shown": True,
        "categories": ["D1", "D2", "D3", "D4"],
        "series": [
            {"name": "Share of women", "values": [str(base + k) for k in range(4)]},
            {"name": "Wage gap", "values": [str(base + 10 + 2 * k) for k in range(4)]},
        ],
    }


GROUPS = [_deciles("Low-income countries", 40), _deciles("High-income countries", 50)]


def grouped_table(gap_shift: int = 0) -> str:
    rows = []
    for chart in GROUPS:
        share, gap = (s["values"] for s in chart["series"])
        for k, cat in enumerate(chart["categories"]):
            g = str(int(gap[k]) + gap_shift) if chart is GROUPS[1] else gap[k]
            rows.append([chart["title"] if k == 0 else "", cat, share[k], g])
    return md(["Group", "Decile", "Share of women", "Wage gap"], rows, "Share of women and wage gap by decile")


def test_group_label_written_once_labels_the_blank_cells_below_it():
    res = score_chart_next(grouped_table(), GROUPS)
    assert res is not None and res.score == 1.0


def test_grouped_table_with_a_wrong_panel_is_still_charged():
    res = score_chart_next(grouped_table(gap_shift=20), GROUPS)
    assert res is not None and res.score < 0.8 and res.events.kinds().get("wrong value", 0) >= 1


def test_long_table_with_four_label_columns_reads_the_series_column():
    charts = [
        {
            "title": f"Change in scores - {panel}",
            "chart_type": "bar",
            "data_labels_shown": True,
            "categories": ["Finland", "Denmark", "Sweden"],
            "series": [
                {"name": "Unadjusted", "values": [str(v) for v in vals]},
                {"name": "Adjusted", "values": [str(v + 1) for v in vals]},
            ],
        }
        for panel, vals in (("Native-born adults", (16, 9, 10)), ("Foreign-born adults", (19, 12, 4)))
    ]
    rows = [
        [panel.split(" - ")[1], country, f"{panel.split(' - ')[1]} vs. parents", s["name"], s["values"][k]]
        for c in charts
        for panel in [c["title"]]
        for k, country in enumerate(c["categories"])
        for s in c["series"]
    ]
    res = score_chart_next(md(["Panel", "Country", "Comparison", "Series", "Value"], rows, "Change in scores"), charts)
    assert res is not None and res.score == 1.0
