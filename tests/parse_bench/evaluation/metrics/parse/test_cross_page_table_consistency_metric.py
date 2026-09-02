"""Unit tests for the cross-page table-consistency metric."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from bs4 import BeautifulSoup

from parse_bench.evaluation.metrics.parse.cross_page_table_consistency import (
    _describe,
    _select_cluster_tables,
    compute_cross_page_table_metrics,
)


def _tbl(html: str) -> Any:
    return BeautifulSoup(html, "lxml").find("table")


def _po(pages: dict[int, str]) -> Any:
    return SimpleNamespace(pages=[SimpleNamespace(page_index=p - 1, markdown=md) for p, md in sorted(pages.items())])


def _by_name(metrics: list) -> dict[str, float]:
    return {m.metric_name: m.value for m in metrics}


def _table(header: list[str], ncols: int, nrows: int = 2) -> str:
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>c{j}</td>" for j in range(ncols)) + "</tr>" for _ in range(nrows))
    return f"<table>{head}{body}</table>"


# --- _describe -------------------------------------------------------------


def test_describe_colspan_and_header_rows():
    html = (
        '<table><thead><tr><th colspan="3">Title</th></tr>'
        "<tr><th>a</th><th>b</th><th>c</th></tr></thead>"
        "<tbody><tr><td>1</td><td>2</td><td>3</td></tr><tr><td>4</td><td>5</td><td>6</td></tr></tbody></table>"
    )
    n_cols, n_hrows, header = _describe(_tbl(html))
    assert n_cols == 3  # modal body width, not the colspan=3 title
    assert n_hrows == 2  # both thead rows
    assert "a | b | c" in header


def test_describe_colspan_title_not_counted_as_cols():
    # a colspan-everything section row must not inflate the column count
    html = (
        "<table><tr><td colspan='9'>section</td></tr><tr><td>x</td><td>y</td></tr><tr><td>z</td><td>w</td></tr></table>"
    )
    n_cols, _, _ = _describe(_tbl(html))
    assert n_cols == 2


# --- _select_cluster_tables ------------------------------------------------


def test_select_best_match_disambiguates_multi_table_page():
    by_page = {6: [(2, 1, "vin | stock"), (8, 1, "adjustments | loss")]}
    assert _select_cluster_tables([{"page": 6, "header": "vin | stock"}], by_page) == [(2, 1, "vin | stock")]
    assert _select_cluster_tables([{"page": 6, "header": "adjustments | loss"}], by_page) == [
        (8, 1, "adjustments | loss")
    ]


def test_poor_match_is_kept_not_dropped():
    # Fix A: a mangled header on one page must NOT be dropped — it's exactly the
    # inconsistency the metric exists to catch.
    by_page = {5: [(4, 1, "revenue header")], 6: [(7, 2, "garbled zzz")]}
    members = [{"page": 5, "header": "revenue header"}, {"page": 6, "header": "revenue header"}]
    sel = _select_cluster_tables(members, by_page)
    assert len(sel) == 2  # both kept
    assert {t[0] for t in sel} == {4, 7}  # differing cols stay visible


def test_member_skipped_only_when_page_has_no_tables():
    by_page = {5: [(4, 1, "h")]}  # page 6 produced no parser tables
    sel = _select_cluster_tables([{"page": 5, "header": "h"}, {"page": 6, "header": "h"}], by_page)
    assert len(sel) == 1


def test_legacy_page_list_pools_all():
    by_page = {1: [(2, 1, "a"), (3, 1, "b")]}
    assert len(_select_cluster_tables([1], by_page)) == 2


# --- compute_cross_page_table_metrics --------------------------------------


def test_count_accuracy_uses_expected_not_logical():
    # Fix B: target is the per-page-fair expected count, not the de-segmented logical.
    po = _po({1: _table(["x", "y"], 2) + _table(["x", "y"], 2)})  # 2 parser tables on page 1
    gt = {
        "clusters": {},
        "logical_table_count": 1,
        "expected_page_table_count": 2,
        "logical_tables_per_page": {"1": 1},
    }
    m = _by_name(compute_cross_page_table_metrics(po, gt))
    assert m["cross_page_table_count_accuracy"] == 1.0  # 2 parser == 2 expected (not 1 logical)


def test_consistent_cluster_scores_one():
    t = _table(["zip", "factor"], 2)
    po = _po({1: t, 2: t, 3: t})
    gt = {
        "clusters": {"zip": [{"page": p, "header": "zip | factor"} for p in (1, 2, 3)]},
        "logical_table_count": 3,
        "expected_page_table_count": 3,
        "logical_tables_per_page": {"1": 1, "2": 1, "3": 1},
    }
    m = _by_name(compute_cross_page_table_metrics(po, gt))
    assert m["cross_page_table_col_consistency"] == 1.0
    assert m["cross_page_table_header_row_consistency"] == 1.0
    assert m["cross_page_table_header_similarity"] == 1.0


def test_inconsistent_cluster_flagged():
    po = _po({1: _table(["zip", "factor"], 2), 2: _table(["zip", "factor", "extra"], 3)})
    gt = {
        "clusters": {"zip": [{"page": 1, "header": "zip | factor"}, {"page": 2, "header": "zip | factor"}]},
        "logical_table_count": 2,
        "expected_page_table_count": 2,
        "logical_tables_per_page": {"1": 1, "2": 1},
    }
    m = _by_name(compute_cross_page_table_metrics(po, gt))
    assert m["cross_page_table_col_consistency"] == 0.0  # 2 vs 3 columns
