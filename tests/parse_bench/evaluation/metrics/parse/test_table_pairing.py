"""Unit tests for the shared page-aware table assignment helper."""

from __future__ import annotations

import numpy as np

from parse_bench.evaluation.metrics.parse.table_pairing import (
    assign_tables,
    page_blocking_active,
)


def _pairs(row_ind: list[int], col_ind: list[int]) -> set[tuple[int, int]]:
    return set(zip(row_ind, col_ind, strict=True))


# --- page_blocking_active --------------------------------------------------


def test_page_blocking_active_requires_both_lists_and_lengths():
    assert page_blocking_active([1, 2], [1, 2], 2, 2) is True
    assert page_blocking_active(None, [1, 2], 2, 2) is False
    assert page_blocking_active([1, 2], None, 2, 2) is False
    assert page_blocking_active([1], [1, 2], 2, 2) is False  # exp length != n_expected
    assert page_blocking_active([1, 2], [1], 2, 2) is False  # act length != n_actual


# --- assign_tables: global (no pages) --------------------------------------


def test_global_assignment_matches_linear_sum_assignment():
    # cost = -score; the optimal global assignment pairs the diagonal.
    cost = np.array([[-1.0, 0.0], [0.0, -1.0]])
    row_ind, col_ind = assign_tables(cost)
    assert _pairs(row_ind, col_ind) == {(0, 0), (1, 1)}


def test_global_assignment_picks_cross_pairs_when_cheaper():
    # Best global match is the anti-diagonal — proves no page constraint applies.
    cost = np.array([[0.0, -1.0], [-1.0, 0.0]])
    row_ind, col_ind = assign_tables(cost)
    assert _pairs(row_ind, col_ind) == {(0, 1), (1, 0)}


# --- assign_tables: per-page -----------------------------------------------


def test_per_page_forbids_cross_page_even_when_cheaper():
    # Anti-diagonal is globally cheapest, but pages force the diagonal.
    cost = np.array([[0.0, -1.0], [-1.0, 0.0]])
    row_ind, col_ind = assign_tables(cost, expected_pages=[1, 2], actual_pages=[1, 2])
    assert _pairs(row_ind, col_ind) == {(0, 0), (1, 1)}


def test_per_page_gt_without_prediction_is_unmatched():
    # GT page 2 has no prediction -> that GT row is left unmatched.
    cost = np.array([[-1.0], [-1.0]])  # 2 GT x 1 pred
    row_ind, col_ind = assign_tables(cost, expected_pages=[1, 2], actual_pages=[1])
    assert _pairs(row_ind, col_ind) == {(0, 0)}
    assert 1 not in row_ind


def test_per_page_pred_only_page_is_ignored():
    # Pred on page 3 has no GT -> not force-matched.
    cost = np.array([[-1.0, -1.0]])  # 1 GT x 2 pred
    row_ind, col_ind = assign_tables(cost, expected_pages=[1], actual_pages=[1, 3])
    assert _pairs(row_ind, col_ind) == {(0, 0)}


def test_per_page_multiple_tables_same_page():
    # Two GT and two pred all on page 1 -> behaves like a global 2x2 solve.
    cost = np.array([[-1.0, 0.0], [0.0, -1.0]])
    row_ind, col_ind = assign_tables(cost, expected_pages=[1, 1], actual_pages=[1, 1])
    assert _pairs(row_ind, col_ind) == {(0, 0), (1, 1)}


def test_length_mismatch_falls_back_to_global():
    cost = np.array([[0.0, -1.0], [-1.0, 0.0]])
    # expected_pages too short -> page blocking disabled -> global anti-diagonal.
    row_ind, col_ind = assign_tables(cost, expected_pages=[1], actual_pages=[1, 2])
    assert _pairs(row_ind, col_ind) == {(0, 1), (1, 0)}
