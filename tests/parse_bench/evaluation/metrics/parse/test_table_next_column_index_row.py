"""table_next: a printed column-index row "(1) | (2) | ... | (n)" is not table content."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import score_table_next


def _html(head: list[list[str]], body: list[list[str]]) -> str:
    th = "".join("<tr>" + "".join(f"<th>{x}</th>" for x in r) + "</tr>" for r in head)
    td = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in body)
    return f"<table><thead>{th}</thead><tbody>{td}</tbody></table>"


HEAD = ["Discount", "Inforce Premium", "Current Factor", "Proposed Factor", "Relative Chg Factor"]
BODY_A = [["No", "68,412,307", "1.000", "1.000", "1.000"], ["Yes", "314,859", "0.950", "1.000", "1.053"]]
BODY_B = [["No", "62,904,135", "1.000", "1.000", "1.000"], ["Yes", "7,836,214", "0.900", "1.000", "1.111"]]
INDEX = ["(1)", "(2)", "(3)", "(4)", "(5)"]
REF = _html([HEAD], BODY_A) + "\n\n" + _html([HEAD], BODY_B)


def test_column_index_row_above_the_header_costs_nothing() -> None:
    out = _html([INDEX], [HEAD, *BODY_A]) + "\n\n" + _html([HEAD], BODY_B)
    assert score_table_next(REF, out).score == 1.0


def test_reference_with_index_row_matches_output_without_it() -> None:
    ref = _html([INDEX, HEAD], BODY_A)
    assert score_table_next(ref, ref).score == 1.0
    assert score_table_next(ref, _html([HEAD], BODY_A)).score == 1.0


def test_parenthesized_values_that_do_not_count_up_stay_content() -> None:
    # Negative amounts in accounting style: a real data row, so dropping it from the output still costs.
    ref = _html([HEAD], [*BODY_A, ["Adj", "(41,862)", "(2)", "(7)", "(1)"]])
    assert score_table_next(ref, _html([HEAD], BODY_A)).score < 0.9
