"""table_next: correct outputs the ParseBench v2 contentious-pages audit found misread, and wrong outputs that
must still be charged.

* HTML5 named character references ("&dollar;") decode like numeric ones.
* A header label followed by blank header cells reads as that label's colspan.
* A row label written once above blank label cells reads as its rowspan, also across a row that holds no value.
* An empty stub column beside the row labels (the header words left in it) joins the labels, one minor.
* Twin reference tables (same keys, other values) allow twin output tables: one is not a copy of the other.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import TableNextResult, score_table_next


def html(rows: list[list[str]], head: int = 1, skip: int = 0) -> str:
    """An HTML table; a cell "x|2" spans two columns, "x^3" three rows. Rows ``skip`` to ``head`` - 1 are th."""

    def cell(x: str, tag: str) -> str:
        if "|" in x:
            t, n = x.split("|")
            return f'<{tag} colspan="{n}">{t}</{tag}>'
        if "^" in x:
            t, n = x.split("^")
            return f'<{tag} rowspan="{n}">{t}</{tag}>'
        return f"<{tag}>{x}</{tag}>"

    return (
        "<table>"
        + "".join(
            "<tr>" + "".join(cell(x, "th" if skip <= i < head else "td") for x in r) + "</tr>"
            for i, r in enumerate(rows)
        )
        + "</table>"
    )


def kinds(res: TableNextResult) -> set[str]:
    return {k for k, *_ in res.events.items}


# --- named character references -------------------------------------------------------------------------------

DOLLARS = html(
    [["Item", "2024", "2023"], ["Revenue", "$ 235,717", "$ 246,252"], ["Net income", "$ 17,661", "$ 21,369"]]
)


def test_named_dollar_reference_reads_as_the_dollar_sign() -> None:
    out = DOLLARS.replace("$", "&dollar;")
    assert score_table_next(DOLLARS, out).score == 1.0


def test_named_reference_with_a_wrong_number_is_still_wrong() -> None:
    out = DOLLARS.replace("$ 235,717", "&dollar; 235,718")
    assert score_table_next(DOLLARS, out).score < 0.95


def test_reference_without_semicolon_stays_text() -> None:
    ref = html([["Item", "Value"], ["A&para", "1"], ["B", "2"], ["C", "3"]])
    out = html([["Item", "Value"], ["A¶", "1"], ["B", "2"], ["C", "3"]])
    assert score_table_next(ref, out).score < 1.0


# --- header label + blank cells -------------------------------------------------------------------------------

BODY = [
    ["Net sales", "$21.9", "$21.4", "$0.4", "2.0%", "$86.2", "$84.3", "$1.8", "2.2%"],
    ["Net sales (ex. fuel)", "$19.4", "$18.8", "$0.6", "3.3%", "$75.1", "$71.7", "$3.4", "4.7%"],
    ["Operating income", "$0.6", "$0.5", "$0.1", "20.4%", "$2.2", "$2.0", "$0.2", "11.6%"],
]
SPANNED = html([["Sam's Club U.S.", "FY'24Q4", "FY'23Q4", "Change|2", "FY'24", "FY'23", "Change|2"], *BODY])


def test_header_label_then_blank_cell_reads_as_its_colspan() -> None:
    out = html([["Sam's Club U.S.", "FY'24Q4", "FY'23Q4", "Change", "", "FY'24", "FY'23", "Change", ""], *BODY])
    assert score_table_next(SPANNED, out).score == 1.0


def test_blank_header_over_a_column_the_reference_heads_apart_is_charged() -> None:
    ref = html([["Sam's Club U.S.", "FY'24Q4", "FY'23Q4", "$", "%", "FY'24", "FY'23", "$", "%"], *BODY])
    out = html([["Sam's Club U.S.", "FY'24Q4", "FY'23Q4", "$", "", "FY'24", "FY'23", "$", ""], *BODY])
    res = score_table_next(ref, out)
    assert "label reworded" in kinds(res)
    assert res.score < 1.0


def test_blank_header_cells_over_swapped_columns_cost_what_a_colspan_does() -> None:
    swapped = [[r[0], *r[1:3], r[4], r[3], *r[5:7], r[8], r[7]] for r in BODY]  # each Change pair swapped
    blank = html([["Sam's Club U.S.", "FY'24Q4", "FY'23Q4", "Change", "", "FY'24", "FY'23", "Change", ""], *swapped])
    spanned = html([["Sam's Club U.S.", "FY'24Q4", "FY'23Q4", "Change|2", "FY'24", "FY'23", "Change|2"], *swapped])
    res = score_table_next(SPANNED, blank)
    assert "column out of order" in kinds(res)
    assert res.score == score_table_next(SPANNED, spanned).score < 1.0


# --- blank row-label cells ------------------------------------------------------------------------------------


def form_page(spans: bool, second_label: str = "(54)") -> str:
    """A form page: a table of "Rate Impact" / "Rule" rows, then one block per item whose item cell spans its
    Proposed, Current (no value), Rate Impact and Rule rows. ``spans=False`` writes the item once over blank cells."""

    def block(item: str, form: str, rate: str, rule: str) -> str:
        rows = [["Current", "", "", ""], ["Rate Impact", rate, "", ""], ["Rule", rule, "", ""]]
        if spans:
            return html([[f"{item}^4", "Proposed", form, "2", "Y"], *rows], head=0)
        return html([[item, "Proposed", form, "2", "Y"], *[["", *r] for r in rows]], head=0)

    first = html([["Rate Impact", "No impact"], ["Rule", "ISO Rule 14"]], head=0)
    second = block(second_label if not spans else "(54)", "CP 03 21", "Rate change", "Rule 82")
    return "\n\n".join([first, block("(53)", "CP 01 40", "No impact", "Rule A9"), second])


def test_blank_item_cells_read_as_the_item_span_across_a_value_less_row() -> None:
    assert score_table_next(form_page(spans=True), form_page(spans=False)).score == 1.0


def test_item_label_left_out_is_still_charged() -> None:
    res = score_table_next(form_page(spans=True), form_page(spans=False, second_label=""))
    assert "row missing" in kinds(res)
    assert res.score < 0.7


# --- empty stub column ----------------------------------------------------------------------------------------

RATES = [
    ["8.000", "101.500", "101.375", "100.875", "99.500"],
    ["7.875", "101.375", "101.250", "100.750", "99.375"],
    ["7.750", "101.250", "101.125", "100.625", "99.250"],
]
STUB_REF = html(
    [
        ["LENDING RATES|5"],
        ["BASE RATE", "TIER 1|2", "TIER 2|2"],
        ["RATE %", "21 DAY", "30 DAY", "60 DAY", "FLOAT"],
        *RATES,
    ],
    head=3,
)


def stub_out(rates: list[list[str]], stub: str = "") -> str:
    """The wide first column read as an empty stub column plus the rate column; its header words stay in the stub."""
    rows = [["BASE RATE", "", "TIER 1|2", "TIER 2|2"], ["", "RATE %", "21 DAY", "30 DAY", "60 DAY", "FLOAT"]]
    body = [[stub if i == 1 else "", *r] for i, r in enumerate(rates)]
    return html([["", "LENDING RATES|5"], *rows, *body], head=3, skip=1)


def test_empty_stub_column_joins_the_row_labels_at_one_minor() -> None:
    res = score_table_next(STUB_REF, stub_out(RATES))
    assert kinds(res) == {"empty rows or columns added"}
    assert res.score == 0.99


def test_empty_stub_column_with_a_wrong_value_is_charged() -> None:
    wrong = [[*RATES[0][:3], "100.975", RATES[0][4]], *RATES[1:]]
    res = score_table_next(STUB_REF, stub_out(wrong))
    assert "wrong or misplaced number" in kinds(res)
    assert res.score < 0.9


def test_stub_column_holding_text_is_not_joined() -> None:
    res = score_table_next(STUB_REF, stub_out(RATES, stub="note"))
    assert "empty rows or columns added" not in kinds(res)
    assert res.score < 0.5


# --- twin tables ----------------------------------------------------------------------------------------------

LABELS = ["Vehicle manufacturing", "Of which enabling", "Non eligible", "Total"]


def twin(title: str, values: list[str], header: str = "€m") -> str:
    return f"# {title}\n\n" + html(
        [["Activities", header, "%"], *[[lab, v, "1%"] for lab, v in zip(LABELS, values, strict=True)]]
    )


CAPEX, OPEX = ["4,341", "1,000", "267", "5,608"], ["1,400", "300", "100", "1,800"]
TWINS = twin("Capex", CAPEX) + "\n\n" + twin("Opex", OPEX)


def test_twin_tables_with_one_header_slip_each_are_not_copies() -> None:
    out = twin("Capex", CAPEX, "€M total") + "\n\n" + twin("Opex", OPEX, "€M total")
    res = score_table_next(TWINS, out)
    assert "duplicated table" not in kinds(res)
    assert res.score >= 0.9


def test_one_twin_written_twice_is_still_charged() -> None:
    out = twin("Capex", CAPEX) + "\n\n" + twin("Capex", CAPEX)
    res = score_table_next(TWINS, out)
    assert "wrong or misplaced number" in kinds(res)
    assert res.score < 0.8


def test_a_third_copy_of_twin_tables_is_a_duplicate() -> None:
    out = TWINS + "\n\n" + twin("Opex", OPEX)
    assert "duplicated table" in kinds(score_table_next(TWINS, out))


def test_a_twin_written_twice_next_to_its_real_twin_is_a_duplicate() -> None:
    """Capex, Capex again, then Opex (each with the header slip): Opex owns the second slot of the layout, so the
    verbatim second Capex is a copy (it was kept and scored as extra columns)."""
    ref = twin("Capex", CAPEX) + "\n\n" + twin("Opex", OPEX).replace("1%", "2%")
    capex = twin("Capex", CAPEX, "€M total")
    out = capex + "\n\n" + capex + "\n\n" + twin("Opex", OPEX, "€M total").replace("1%", "2%")
    res = score_table_next(ref, out)
    assert "duplicated table" in kinds(res)
    assert res.score >= 0.9
