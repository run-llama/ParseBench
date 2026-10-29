"""Tests for the table metric (``table_next``)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from next_wiring import NextScorers

from parse_bench.evaluation.metrics.parse.table_next import (
    build_triples,
    clean_grid,
    decode_entities,
    infer_schema,
    key_sim,
    nx_norm,
    parse_tables,
    score_table_next,
    table_next_metrics,
    value_sim,
)
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import ParseTestCase

HEAD = ["Region", "2023", "2024"]
BODY = [
    ["North America", "1,284", "1,352"],
    ["Europe", "846", "901"],
    ["Asia-Pacific", "512", "618"],
    ["Latin America", "133", "176"],
    ["Middle East", "97", "121"],
    ["Africa", "64", "88"],
]


def html(rows: list[list[str]], head: int = 1, title: str | None = None) -> str:
    out = ["<table>"]
    if title is not None:
        out.append(f'<tr><th colspan="{len(rows[0])}">{title}</th></tr>')
    for i, r in enumerate(rows):
        tag = "th" if i < head else "td"
        out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in r) + "</tr>")
    return "\n".join(out + ["</table>"])


def md(rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows[1:]])


REF = html([HEAD, *BODY])


def score(out: str, ref: str = REF, **flags: object) -> tuple[float, dict[str, int]]:
    res = score_table_next(ref, out, **flags)  # type: ignore[arg-type]
    types: dict[str, int] = {}
    for kind, _, _, sev in res.events.items:
        if sev != "detail":
            types[kind] = types.get(kind, 0) + 1
    return res.score, types


def with_body(body: list[list[str]], head: list[str] = HEAD) -> str:
    return html([head, *body])


# --- parsing and normalization ----------------------------------------------------------------------------------


def test_parses_html_spans_markdown_and_unclosed_tables():
    t = parse_tables(
        '<table><tr><th colspan="2">A</th></tr><tr><td>1</td><td>2</td></tr></table>\n\n| x | y |\n|---|---|\n| 1 | 2 |'
    )
    assert [x.kind for x in t] == ["html", "md"]
    assert (t[0].cells[0].cs, t[0].cells[1].c) == (2, 0)
    assert len(parse_tables("<table><tr><td>a</td><td>b</td></tr>")) == 1  # truncated output still counts


def test_entities_links_and_typography_normalize():
    assert decode_entities("a &#x3C; b &#38; c &#xD800;") == "a < b & c &#xD800;"
    assert nx_norm("[Revenue](http://example.com/1)") == "revenue"
    assert nx_norm("2022 ............") == "2022"
    assert nx_norm("$ 1,824") == nx_norm("$1,824")
    assert nx_norm("V / D") == nx_norm("V/D")
    assert nx_norm("Company Name:") == "company name"
    assert nx_norm("**Total**") == "total"


def test_value_similarity_grades():
    assert value_sim("1,284", "1,284") == 1
    assert value_sim("Europe", "EUROPE") == 0.95
    assert value_sim("(21,330)", "-21330") == 0.9
    assert value_sim("450 W", "450") == 0.5
    assert value_sim("1,284", "1,285") == 0
    assert value_sim("—", "-") == 1  # both empty markers


def test_label_similarity_typos_digits_and_extra_levels():
    assert key_sim("operating income", "operating incme") > 0.8
    assert key_sim("2023 revenue", "2024 revenue") == 0
    assert key_sim("attained age", "male rates attained age") == 0.6
    assert key_sim("region", "area") == 0


# --- structure inference ----------------------------------------------------------------------------------------


def test_schema_from_th_title_and_label_columns():
    g = clean_grid(parse_tables(html([HEAD, *BODY], title="Sales by region"))[0]).g
    s = infer_schema(g, 1)
    assert (s.T, s.H, s.L) == (1, 1, 1)
    assert infer_schema(g, 0).T == 0  # max_top_title_rows = 0 keeps the title row in the header


def test_schema_without_th_uses_rows_without_numbers():
    g = clean_grid(parse_tables(html([HEAD, *BODY], head=0))[0]).g
    assert infer_schema(g, 1).H == 1


def test_row_number_column_is_a_label_and_section_rows_qualify_labels():
    rows = [
        ["", "", "Invested", "Yield"],
        ["(1)", "Bonds", "", ""],
        ["", "Short", "100", "1.0%"],
        ["", "Long", "200", "2.0%"],
        ["(2)", "Stocks", "", ""],
        ["", "Short", "300", "3.0%"],
        ["", "Long", "400", "4.0%"],
    ]
    tg = clean_grid(parse_tables(html(rows))[0])
    s = infer_schema(tg.g, 1)
    assert s.sec and s.L == 2
    keys = {x.rk for x in build_triples(tg, s).triples}
    assert keys == {"(1) bonds short", "(1) bonds long", "(2) stocks short", "(2) stocks long"}


def test_all_text_table_uses_the_shortest_unique_label_prefix():
    rows = [["Code", "Name", "City"], ["A1", "Acme", "Paris"], ["B2", "Bolt", "Lyon"], ["C3", "Core", "Nice"]]
    tg = clean_grid(parse_tables(html(rows))[0])
    assert infer_schema(tg.g, 1).L == 1


# --- scores and severity --------------------------------------------------------------------------------------


def test_identical_and_format_variants_score_one():
    assert score(REF)[0] == pytest.approx(1.0)
    assert score(md([HEAD, *BODY]))[0] == pytest.approx(1.0)  # Markdown is not an error
    no_th = REF.replace("<th>", "<td>").replace("</th>", "</td>")
    assert score(no_th)[0] == pytest.approx(1.0)  # th vs td markup is not an error
    styled = REF.replace("<table>", '<table class="x"><thead>').replace("<td>", '<td style="a">')
    assert score(styled)[0] == pytest.approx(1.0)


def test_no_reference_table_and_missing_output():
    assert score("anything", ref="just text")[0] == 1.0
    s, ev = score("Table could not be extracted.")
    assert s == 0 and ev == {"table missing": 1}


def test_wrong_number_is_critical():
    body = [r[:] for r in BODY]
    body[1][1] = "864"
    s, ev = score(with_body(body))
    assert s == pytest.approx(0.9)
    assert ev == {"cell errors in row": 1}


def test_missing_row_and_invented_row_are_critical():
    s, ev = score(with_body(BODY[:-1]))
    assert ev["row missing"] == 1 and s <= 0.9
    s, ev = score(with_body([*BODY, ["Antarctica", "1", "2"]]))
    assert ev["invented row"] == 1 and s <= 0.9


def test_lost_unit_and_reworded_label_are_major():
    ref = with_body([["Power", "450 W", "500 W"], ["Weight", "1.6 kg", "1.8 kg"]])
    s, ev = score(with_body([["Power", "450", "500 W"], ["Weight", "1.6 kg", "1.8 kg"]]), ref=ref)
    assert ev == {"cell errors in row": 1} and s == pytest.approx(0.875)  # content 3.5 / 4 is below the 0.95 floor
    s, ev = score(with_body(BODY, head=["Area", "2023", "2024"]))
    assert ev == {"row-label header differs": 1} and s == pytest.approx(0.95)


def test_number_format_and_case_are_minor():
    body = [r[:] for r in BODY]
    body[0][1] = "1284"
    s, ev = score(with_body(body))
    assert ev == {"number format changed": 1} and s == pytest.approx(0.99)
    body = [[r[0].upper(), *r[1:]] for r in BODY]
    s, _ = score(with_body(body))
    assert 0.94 <= s < 1


def test_row_reorder_and_column_swap_are_errors():
    s, ev = score(with_body([BODY[3], *BODY[:3], *BODY[4:]]))
    assert ev["row out of order"] >= 1 and s < 0.96
    swapped = html([[r[0], r[2], r[1]] for r in [HEAD, *BODY]])
    s, ev = score(swapped)
    assert ev["column out of order"] == 1 and s == pytest.approx(0.95)


def test_transposition_and_header_flattening_are_errors():
    rows = [HEAD, *BODY]
    s, ev = score(html([list(col) for col in zip(*rows, strict=True)], head=0))
    assert ev["table transposed"] == 1 and s == pytest.approx(0.9)
    ref = html(
        [
            ["", "Q1", "Q1", "Q2", "Q2"],
            ["", "Sales", "Cost", "Sales", "Cost"],
            ["A", "1", "2", "3", "4"],
            ["B", "5", "6", "7", "8"],
        ],
        head=2,
    )
    flat = html(
        [["", "Q1 Sales", "Q1 Cost", "Q2 Sales", "Q2 Cost"], ["A", "1", "2", "3", "4"], ["B", "5", "6", "7", "8"]]
    )
    s, ev = score(flat, ref=ref)
    assert ev == {"header rows differ": 1} and s == pytest.approx(0.95)


def test_split_and_merged_tables_are_major():
    # the continuation after a break has no header of its own: it is read with the reference's column keys
    s, ev = score(with_body(BODY[:3]) + "\n\n" + html(BODY[3:], head=0))
    assert ev == {"table split": 1} and s == pytest.approx(0.95)
    s, ev = score(html(BODY, head=0))
    assert ev == {"header rows missing": 1} and s == pytest.approx(0.9)
    second = html([["Office", "Staff"], ["Toronto", "412"], ["Berlin", "305"], ["Paris", "211"]])
    merged = html(
        [HEAD, *BODY, ["Office", "Staff", ""], ["Toronto", "412", ""], ["Berlin", "305", ""], ["Paris", "211", ""]]
    )
    s, ev = score(merged, ref=REF + "\n\n" + second)
    assert ev == {"tables merged": 1} and s == pytest.approx(0.95)  # split back at the second table's header row


def test_unannotated_extra_table_is_ignored():
    extra = html([["Metric", "Value"], ["Headcount", "1,284"], ["Offices", "12"]]).replace("1,284", "7,777")
    s, ev = score(REF + "\n\n" + extra)
    assert s == pytest.approx(1.0) and ev == {"unpaired output table ignored": 1}


def test_repeated_events_scale_down_on_large_tables():
    big_body = [[f"Item {i}", str(100 + i), str(200 + i)] for i in range(120)]
    ref = with_body(big_body, head=["Item", "A", "B"])
    bad = [r[:] for r in big_body]
    for i in range(0, 30, 3):
        bad[i][1] = str(900 + i)
    small_s, _ = score(with_body([[r[0], "9" + r[1], r[2]] for r in BODY[:3]] + BODY[3:]))
    big_s, ev = score(with_body(bad, head=["Item", "A", "B"]), ref=ref)
    assert ev == {"systematic column error": 1}  # one failing column counts once
    assert big_s > small_s


# --- test.json flags ---------------------------------------------------------------------------------------------


def test_max_top_title_rows():
    ref = html([HEAD, *BODY], title="Sales by region")
    assert score(REF, ref=ref)[1].get("title missing") == 1
    assert score("## Sales by region\n\n" + REF, ref=ref)[0] == pytest.approx(1.0)  # title written above the table
    # with no title stripping the spanning row is a header level shared by every column: no column identity
    assert score(ref, ref=ref, max_top_title_rows=0)[0] == pytest.approx(1.0)


def test_allow_splitting_ambiguous_merged_tables():
    left = html([["Zip", "Rate"], ["90001", "0.96"], ["90002", "0.97"], ["90003", "0.98"]])
    right = html([["Zip", "Rate"], ["90059", "1.06"], ["90061", "0.97"], ["90062", "1.01"]])
    wide = html(
        [
            ["Zip", "Rate", "Zip", "Rate"],
            ["90001", "0.96", "90059", "1.06"],
            ["90002", "0.97", "90061", "0.97"],
            ["90003", "0.98", "90062", "1.01"],
        ]
    )
    s, ev = score(wide, ref=left + "\n\n" + right)
    assert ev == {"side-by-side tables merged": 1} and s == pytest.approx(0.95)
    s, ev = score(wide, ref=left + "\n\n" + right, allow_splitting_ambiguous_merged_tables=True)
    assert s == pytest.approx(1.0) and not ev
    s, _ = score(left + "\n\n" + right, ref=wide, allow_splitting_ambiguous_merged_tables=True)
    assert s == pytest.approx(1.0)  # symmetric: the reference is split too


def test_trm_unsupported_matches_by_row_and_column_position():
    out = with_body(BODY, head=["Region", "Year", "Year"])  # both columns lose their distinct header
    strict, _ = score(out)
    relaxed, ev = score(out, trm_unsupported=True)
    assert strict < 0.6
    assert relaxed > 0.85 and ev.get("column headers differ") == 1


# --- evaluator wiring -------------------------------------------------------------------------------------------


def _inference_result(markdown: str) -> InferenceResult:
    output = ParseOutput(
        example_id="grp/doc", pipeline_name="p", markdown=markdown, pages=[PageIR(page_index=0, markdown=markdown)]
    )
    return InferenceResult(
        request=InferenceRequest(example_id="grp/doc", source_file_path="doc.pdf", product_type=ProductType.PARSE),
        pipeline_name="p",
        product_type=ProductType.PARSE,
        raw_output={},
        output=output,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        latency_in_ms=1,
    )


def _case(metadata: dict | None, expected: str = REF, **flags: object) -> ParseTestCase:
    return ParseTestCase.model_construct(
        test_id="grp/doc", group="grp", file_path="doc.pdf", expected_markdown=expected, metadata=metadata, **flags
    )


def test_metrics_only_when_reference_has_tables():
    assert table_next_metrics("no table here", REF) == []
    names = [m.metric_name for m in table_next_metrics(REF, REF)]
    assert names == ["table_next", "table_next_content", "table_next_row_order"]


def test_wiring_emits_table_next_against_the_reference_tables():
    evaluator = NextScorers()
    body = [r[:] for r in BODY]
    body[2][2] = "681"
    result = evaluator.evaluate(_inference_result(with_body(body)), _case({"scoring": "next"}))
    by_name = {m.metric_name: m for m in result.metrics}
    assert by_name["table_next"].value == pytest.approx(0.9)
    assert by_name["table_next"].metadata["events"] == {"critical": 1, "major": 0, "minor": 0}
    assert by_name["table_next_content"].value < 1
    assert by_name["table_next_row_order"].value == pytest.approx(1.0)

    no_table = evaluator.evaluate(_inference_result(REF), _case(None, expected="Only prose here."))
    assert not any(m.metric_name.startswith("table_next") for m in no_table.metrics)


def test_wiring_passes_table_flags():
    evaluator = NextScorers()
    ref = html([HEAD, *BODY], title="Sales by region")
    case = _case({"scoring": "next"}, expected=ref, max_top_title_rows=0)
    result = evaluator.evaluate(_inference_result(ref), case)
    meta = {m.metric_name: m for m in result.metrics}["table_next"].metadata
    assert meta["flags"]["max_top_title_rows"] == 0


GROUPED_HEAD = (
    '<tr><th rowspan="2">Region</th><th colspan="2">Revenue (USD m)</th></tr><tr><th>2023</th><th>2024</th></tr>'
)


def grouped(rows: list[list[str]], head: str = GROUPED_HEAD) -> str:
    return "<table>" + head + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


def test_span_copy_row_is_not_a_repeated_row():
    # a header cell with rowspan over an otherwise empty row: the row has no content of its own
    out = '<table><tr><th rowspan="2">Region</th><th>2023</th><th>2024</th></tr><tr><th></th><th></th></tr>'
    out += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in BODY) + "</table>"
    s, ev = score(out)
    assert "repeated row" not in ev and s == pytest.approx(1.0)


def test_wrong_rowspan_shifting_cells_is_one_major():
    out = '<table><tr><th rowspan="2">Region</th><th>2023</th><th>2024</th></tr>'
    out += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in BODY) + "</table>"
    s, ev = score(out)
    assert ev == {"wrong row span": 1} and s == pytest.approx(0.95)


def test_dropped_label_span_reads_like_the_span():
    ref = grouped([["Group A", "1", "2"], ["Group A", "3", "4"], ["Group B", "5", "6"]])
    ref = ref.replace("<tr><td>Group A</td><td>1</td>", '<tr><td rowspan="2">Group A</td><td>1</td>', 1)
    ref = ref.replace("<tr><td>Group A</td><td>3</td>", "<tr><td>3</td>", 1)
    out = grouped([["Group A", "1", "2"], ["", "3", "4"], ["Group B", "5", "6"]])
    assert score_table_next(ref, out).score == pytest.approx(1.0)


def test_row_whose_label_did_not_pair_costs_one_event():
    # the label of one row merged with a note: the same values under another label, not a missing + repeated row
    out = html([HEAD, *BODY[:2], ["Asia-Pacific (Note 7)", "512", "618"], *BODY[3:]])
    s, ev = score(out)
    assert "row missing" not in ev and "repeated row" not in ev and s >= 0.95


def test_stacked_tables_split_back_and_charged_once():
    second = html([["Office", "Staff"], ["Toronto", "412"], ["Berlin", "305"]])
    merged = html([HEAD, *BODY, ["Office", "Staff", ""], ["Toronto", "412", ""], ["Berlin", "305", ""]])
    s, ev = score(merged, ref=REF + "\n\n" + second)
    assert ev == {"tables merged": 1} and s == pytest.approx(0.95)


def test_spanning_header_joined_into_one_row_is_major():
    ref = grouped(BODY)
    joined = html([["Region", "Revenue (USD m) 2023", "Revenue (USD m) 2024"], *BODY])
    s, ev = score(joined, ref=ref)
    assert ev == {"header rows joined": 1} and s == pytest.approx(0.95)
    assert score(ref, ref=ref)[0] == pytest.approx(1.0)


def test_percent_sign_in_cells_under_a_percent_header_is_free():
    ref = html([["Region", "Share (%)"], ["Europe", "30.4"], ["Asia", "18.4"], ["Africa", "5.1"]])
    out = html([["Region", "Share (%)"], ["Europe", "30.4%"], ["Asia", "18.4%"], ["Africa", "5.1%"]])
    assert score(out, ref=ref) == (pytest.approx(1.0), {})
    assert value_sim("30.4", "30.4%") == 0.5  # without a percent label the unit change still counts


def test_glyph_variants_and_single_asterisk_emphasis_are_format():
    assert nx_norm("⊗") == nx_norm("⮾") == nx_norm("✗") and nx_norm("✔") == nx_norm("✓")
    assert nx_norm("*Updated Zip Code Factors*") == nx_norm("Updated Zip Code Factors")
    assert nx_norm("Revenue*") == "revenue*"  # a footnote asterisk is content


def test_shortened_header_still_pairs_its_column():
    assert key_sim("change", "change updated zip code factors effective with new business") == 0.6
    assert key_sim("a", "a b c d e") == 0.0  # a short, generic word alone is not enough


def test_row_label_typo_is_minor():
    out = html([HEAD, BODY[0], ["Eurpoe", "846", "901"], *BODY[2:]])
    s, ev = score(out)
    assert ev == {"label reworded": 1} and s == pytest.approx(0.99)


def test_missing_table_floor_reports_the_critical():
    res = score_table_next(REF, "no table here")
    assert res.score == 0.0 and res.floor == pytest.approx(0.9)


# --- adversarial cases: what a parser can get wrong, and the severity class each must land in ---------------------

ADV_HEAD = ["Region", "2023", "2024"]
ADV_BODY = [
    ["North America", "1,284", "1,352"],
    ["Europe", "846", "901"],
    ["Asia-Pacific", "512", "618"],
    ["Latin America", "(133)", "176"],
    ["Middle East", "97", "121"],
    ["Total", "2,606", "3,168"],
]
ADV_REF = html([ADV_HEAD, *ADV_BODY])


def _adv(rows: list[list[str]], head: list[str] = ADV_HEAD) -> tuple[float, dict[str, int]]:
    return score(html([head, *rows]), ref=ADV_REF)


def _with(i: int, row: list[str]) -> list[list[str]]:
    return [row if k == i else r for k, r in enumerate(ADV_BODY)]


@pytest.mark.parametrize(
    ("rows", "head", "kind"),
    [
        (_with(1, ["Europe", "901", "846"]), ADV_HEAD, "cell errors in row"),  # two years swapped in one row
        (_with(3, ["Latin America", "133", "176"]), ADV_HEAD, "cell errors in row"),  # accounting negative lost
        (_with(0, ["North America", "1,2841", "1,352"]), ADV_HEAD, "cell errors in row"),  # footnote mark glued on
        (_with(4, ["Middle East", "9.7", "121"]), ADV_HEAD, "cell errors in row"),  # decimal point inserted
        ([*ADV_BODY[:-1], ["Other", "45", "63"], ADV_BODY[-1]], ADV_HEAD, "invented row"),  # plausible extra row
        (ADV_BODY[:-1], ADV_HEAD, "row missing"),  # total row dropped
        (ADV_BODY, ["Region", "2022", "2024"], "label number changed"),  # wrong year in a column header
    ],
)
def test_adversarial_cases_are_critical(rows: list[list[str]], head: list[str], kind: str) -> None:
    s, ev = _adv(rows, head)
    assert kind in ev and s <= 0.9


def test_values_shifted_one_row_are_wrong_values_not_relabelled_rows():
    vals = [r[1:] for r in ADV_BODY]
    shifted = [[ADV_BODY[i][0], *vals[(i + 1) % len(ADV_BODY)]] for i in range(len(ADV_BODY))]
    s, ev = _adv(shifted)
    assert "label partly differs" not in ev and s <= 0.5


def test_label_with_a_note_number_added_is_not_a_changed_number():
    assert "label number changed" not in _adv(_with(2, ["Asia-Pacific (Note 7)", "512", "618"]))[1]


def test_spaced_sign_before_a_currency_is_the_same_number():
    ref = html([["Item", "7%", "3%"], ["Costs", "−$75.5", "−$69.9"], ["Savings", "$40.1", "$37.2"]])
    out = html([["Item", "7%", "3%"], ["Costs", "– $75.5", "– 69.9"], ["Savings", "$40.1", "$37.2"]])
    s, ev = score(out, ref=ref)
    assert "small text difference" not in ev and "cell errors in row" not in ev
    assert value_sim("−$75.5", "– $75.5") == 1.0


def test_label_wrapped_over_two_rows_matches_the_spanning_label():
    head = "<tr><th>Combination</th><th>MIC A</th><th>MIC B</th></tr>"
    ref = (
        f"<table>{head}"
        '<tr><td rowspan="2">Alphazole + Betamab</td><td>50</td><td>6</td></tr><tr><td>12</td><td>0.8</td></tr>'
        '<tr><td rowspan="2">Gammacin + Betamab</td><td>25</td><td>19</td></tr><tr><td>4</td><td>2</td></tr>'
        "</table>"
    )
    out = (
        f"<table>{head}"
        "<tr><td>Alphazole +</td><td>50</td><td>6</td></tr><tr><td>Betamab</td><td>12</td><td>0.8</td></tr>"
        "<tr><td>Gammacin +</td><td>25</td><td>19</td></tr><tr><td>Betamab</td><td>4</td><td>2</td></tr>"
        "</table>"
    )
    s, ev = score(out, ref=ref)
    assert ev == {"label split across rows": 1} and s == pytest.approx(0.99)


def test_column_with_only_a_header_does_not_shift_the_values():
    rows = [["Classification", "20%", "20%"], ["Employees", "20%", "15%"], ["Premises", "15%", "10%"]]
    ref = html([["Risk", "Credit", "Debit"], *rows])
    out = html([["Risk", "Credit", "to", "Debit"], *([r[0], r[1], "", r[2]] for r in rows)])
    s, ev = score(out, ref=ref)
    assert ev == {"empty rows or columns added": 1} and s == pytest.approx(0.99)
