"""chart_next robustness: outputs that game or crash the score (synthetic pages, one behaviour per test)."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.chart_next import score_chart_next

YEARS = ["2021", "2022", "2023", "2024"]
HARDWARE = ["12.4", "15.1", "18.2", "21.7"]
SERVICES = ["7.1", "8.4", "9.9", "11.5"]
SEGMENTS = {
    "title": "Revenue by segment",
    "categories": YEARS,
    "series": [{"name": "Hardware", "values": HARDWARE}, {"name": "Services", "values": SERVICES}],
}


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join([*lines, *("| " + " | ".join(r) + " |" for r in rows)])


def segments(hardware: list[str] = HARDWARE, services: list[str] = SERVICES, extra: list[list[str]] = ()) -> str:
    rows = [[y, a, b] for y, a, b in zip(YEARS, hardware, services, strict=True)]
    return "Revenue by segment\n\n" + md_table(["Year", "Hardware", "Services"], [*rows, *extra])


def score(md: str, charts: list[dict]) -> float:
    res = score_chart_next(md, charts)
    assert res is not None
    return res.score


def test_correct_output_scores_one():
    assert score(segments(), [SEGMENTS]) == 1.0


def test_unnamed_series_of_an_untitled_chart_is_scored():
    chart = {"categories": ["A", "B", "C"], "series": [{"name": "", "values": ["1", "2", "3"]}]}
    assert score(md_table(["Category", "Value"], [["A", "1"], ["B", "2"], ["C", "3"]]), [chart]) == 1.0
    assert score(md_table(["Category", "Value"], [["A", "1"], ["B", "2"], ["C", "9"]]), [chart]) < 0.9


def test_currency_code_inside_a_word_is_no_unit():
    # "eur" in "Europe" is no currency: a "Share (Europe)" column on a % chart holds percentages
    chart = {
        "title": "Market share by region",
        "unit": "%",
        "categories": ["2021", "2022"],
        "series": [{"name": "Europe", "values": ["40", "42"]}, {"name": "Asia", "values": ["30", "31"]}],
    }
    rows = [["2021", "40", "30"], ["2022", "42", "31"]]
    assert (
        score("Market share by region\n\n" + md_table(["Year", "Share (Europe)", "Share (Asia)"], rows), [chart]) == 1.0
    )
    # a real currency header still contradicts the percentage unit
    euro = md_table(["Year", "Europe (EURm)", "Asia (EURm)"], rows)
    assert score("Market share by region\n\n" + euro, [chart]) == 0.0


def test_parentheses_do_not_hedge_the_sign_on_a_mixed_sign_chart():
    chart = {
        "title": "Net change",
        "categories": ["Q1", "Q2", "Q3", "Q4"],
        "series": [{"name": "Change", "values": ["-5", "3", "-2", "4"]}],
    }

    def out(values: list[str]) -> str:
        return "Net change\n\n" + md_table(
            ["Quarter", "Change"], [[q, v] for q, v in zip(chart["categories"], values, strict=True)]
        )

    assert score(out(["(5)", "3", "(2)", "4"]), [chart]) == 1.0  # accounting negatives
    assert score(out(["(5)", "(3)", "(2)", "(4)"]), [chart]) < 0.9  # "(3)" is -3, not the reference 3
    # on a chart of positive values only, "(3)" still reads as a comparison label
    positive = {**chart, "series": [{"name": "Change", "values": ["5", "3", "2", "4"]}]}
    assert score(out(["(5)", "(3)", "(2)", "(4)"]), [positive]) == 1.0
    # a grid that writes "(-2)" wraps every value in parentheses: "(3)" is +3, "(-2)" is -2
    assert score(out(["(-5)", "(3)", "(-2)", "(4)"]), [chart]) == 1.0


def test_only_a_percentage_twin_of_a_compound_cell_is_free():
    shares = [
        [f"{a} ({b}%)" for a, b in zip(col, ["40", "45", "50", "55"], strict=True)] for col in (HARDWARE, SERVICES)
    ]
    assert score(segments(*shares), [SEGMENTS]) == 1.0  # "12.4 (40%)": a value and its share
    junk = [[f"{a} (999)" for a in col] for col in (HARDWARE, SERVICES)]
    assert score(segments(*junk), [SEGMENTS]) < 0.9  # "12.4 (999)": the second number is invented


def test_a_wide_range_does_not_match_a_single_value():
    wide = segments(["0–1000"] * 4, ["0–1000"] * 4)
    assert score(wide, [SEGMENTS]) == 0.0
    bound = segments([f"<{float(x) * 2:g}" for x in HARDWARE], [f"<{float(x) * 2:g}" for x in SERVICES])
    assert score(bound, [SEGMENTS]) == 0.0
    # a range as narrow as the tolerance still reads as the value (one minor for the chart)
    narrow = segments([f"{float(x) - 0.05:.2f}–{float(x) + 0.05:.2f}" for x in HARDWARE], SERVICES)
    assert score(narrow, [SEGMENTS]) == 0.99


def test_rows_appended_to_the_chart_table_are_invented_however_many():
    def padded(k: int) -> str:
        return segments(extra=[[f"Invented {i}", "99", "99"] for i in range(k)])

    scores = [score(padded(k), [SEGMENTS]) for k in (1, 4, 5, 50)]
    assert scores == sorted(scores, reverse=True) and scores[0] < 1.0
    assert scores[2] < scores[1] and scores[3] == 0.0
    # a page table under its own caption that also holds the chart's rows is not the chart's table
    table = "Table 2. Revenue, 2015-2024\n\n" + md_table(
        ["Year", "Hardware", "Services"],
        [[str(y), "1", "1"] for y in range(2015, 2021)]
        + [[y, a, b] for y, a, b in zip(YEARS, HARDWARE, SERVICES, strict=True)],
    )
    res = score_chart_next(table, [SEGMENTS])
    assert res is not None and "invented series or category" not in res.events.kinds()


def test_a_wrong_copy_of_the_chart_beside_the_right_one_is_charged():
    wrong = segments(["16.1", "19.6", "23.7", "28.2"], ["4.3", "5.0", "5.9", "6.9"])
    assert score(wrong, [SEGMENTS]) == 0.0
    assert score(wrong + "\n\n" + segments(), [SEGMENTS]) < 0.5
    assert score(segments() + "\n\n" + wrong, [SEGMENTS]) < 0.5
    # the same chart written twice (a table and its copy) restates no error
    assert score(segments() + "\n\n" + segments(), [SEGMENTS]) == 1.0
