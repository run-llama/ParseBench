"""table_next: values found under a label that names something else are misattributed, not reworded."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.table_next import TableNextResult, score_table_next


def _html(head: list[list[str]], body: list[list[str]]) -> str:
    th = "".join("<tr>" + "".join(f"<th>{x}</th>" for x in r) + "</tr>" for r in head)
    td = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in body)
    return f"<table>{th}{td}</table>"


def _kinds(res: TableNextResult, severity: str) -> set[str]:
    return {kind for kind, _, _, sev in res.events.items if sev == severity}


AGES = [["35", "0", "1.34", "11.69"], ["40", "0", "1.52", "12.10"], ["45", "1", "1.71", "12.64"]]
RATES = [["101", "Local", "1.25", "1.01"], ["205", "Local", "0.89", "1.00"], ["317", "Zone", "1.02", "0.98"]]
RATE_HEAD = ["Territory", "Radius", "Collision", "Comprehensive"]


@pytest.mark.parametrize(("label", "misread"), [("35", "3.5"), ("2016", "201.6")])
def test_decimal_point_inserted_in_a_numeric_row_label_is_critical(label: str, misread: str) -> None:
    # was a "label reworded" major or minor: the digit template saw "#" against "#.#" and missed the changed number
    ref = _html([["Age", "Count", "Factor", "Rate"]], [[label, "0", "1.34", "11.69"], *AGES[1:]])
    out = ref.replace(f"<td>{label}</td>", f"<td>{misread}</td>")

    res = score_table_next(ref, out)

    assert "label number changed" in _kinds(res, "critical")
    assert res.score <= 0.9


def test_label_in_the_other_decimal_convention_is_not_a_changed_number() -> None:
    ref = _html([["Note", "Amount", "Due"]], [["4.625% notes", "600", "2029"], ["5.125% notes", "750", "2031"]])
    out = ref.replace("4.625%", "4,625%")

    assert "label number changed" not in _kinds(score_table_next(ref, out), "critical")


def test_value_column_under_the_row_label_header_is_critical() -> None:
    # "Territory" and "Collision" trade places: was two majors ("label reworded", "row-label header differs")
    out = _html([["Collision", "Radius", "Territory", "Comprehensive"]], RATES)

    res = score_table_next(_html([RATE_HEAD], RATES), out)

    assert "values under another label" in _kinds(res, "critical")
    assert res.score <= 0.85


def test_headers_of_two_label_columns_swapped_is_critical() -> None:
    # was one major "row-label header differs", the same cost as a reworded corner cell
    head = ["Schedule", "Schedule Item", "Status", "Public"]
    body = [["Form", "Jacket", "Filed", "Yes"], ["Form", "Rate pages", "Filed", "No"], ["Rule", "Manual", "Open", "No"]]
    out = _html([["Schedule Item", "Schedule", "Status", "Public"]], body)

    res = score_table_next(_html([head], body), out)

    assert "column headers swapped" in _kinds(res, "critical")
    assert res.score <= 0.9


def test_headers_shifted_under_a_spanning_header_are_critical() -> None:
    # each column reads "limit <left neighbor>": the alias pass took every one for a reworded label (one major)
    top = '<tr><th>Rule</th><th colspan="3">Limit</th></tr>'
    body = "".join(f"<tr><td>{r}</td><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for r, a, b, c in (
        ("Base", "110", "125", "140"), ("Fire", "12", "15", "19"), ("Theft", "31", "36", "44")))  # fmt: skip
    ref = f"<table>{top}<tr><th></th><th>$2,000</th><th>$3,000</th><th>$4,000</th></tr>{body}</table>"
    out = f"<table>{top}<tr><th></th><th></th><th>$2,000</th><th>$3,000</th></tr>{body}</table>"

    res = score_table_next(ref, out)

    assert "values under another label" in _kinds(res, "critical")
    assert res.score <= 0.85


def test_abbreviated_label_holding_the_header_word_stays_a_rewording() -> None:
    # "panel" is the row-label header and a word of the abbreviated label: a rewording, never a swapped header
    head = ["Panel", "Test", "Result", "Units"]
    body = [["Lipid panel", "Total cholesterol", "212", "mg/dL"], ["Lipid panel", "LDL cholesterol", "138", "mg/dL"]]
    out = _html([head], [["Lipid panel", "Total chol", "212", "mg/dL"], ["Lipid panel", "LDL chol", "138", "mg/dL"]])

    res = score_table_next(_html([head], body), out)

    assert not _kinds(res, "critical")
    assert "label reworded" in _kinds(res, "major")


def test_header_swap_is_seen_under_a_column_number_row_with_a_formula() -> None:
    # "(3) = (1) x (2)" kept the numbering row from being dropped and made it a data row, which left the table
    # without header rows: each row became a bag of values and swapped headers scored 1.0
    numbers = ["(1)", "(2)", "(3)", "(4) = (2) + (3)"]
    body = [["BI", "1,458", "6,908", "8,366"], ["CL", "539", "4,252", "4,791"], ["PD", "2,851", "13,319", "16,170"]]
    ref = _html([numbers, ["Coverage", "2006", "2007", "Total"]], body)
    out = _html([numbers, ["Coverage", "2007", "2006", "Total"]], body)

    assert score_table_next(ref, ref).score == 1.0
    assert score_table_next(ref, out).score <= 0.9
