"""Shared GT->pred table assignment for the table-similarity metrics.

GriTS and TEDS both build an ``n_expected x n_actual`` cost matrix and solve a
single document-global Hungarian assignment over it. On multi-page documents
that recur near-identical tables across pages, the global assignment mis-pairs a
GT table on one page with a prediction on another. When per-table page labels
are available, ``assign_tables`` instead solves the assignment **per page** so a
GT table only ever pairs with a prediction on the same page.

The return shape mirrors ``scipy.optimize.linear_sum_assignment`` — a
``(row_ind, col_ind)`` pair of equal-length index lists — so callers consume it
exactly as before.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def page_blocking_active(
    expected_pages: list[int] | None,
    actual_pages: list[int] | None,
    n_expected: int,
    n_actual: int,
) -> bool:
    """Return True when page-constrained matching can and should be applied.

    Requires both page-label lists to be present and length-consistent with the
    table counts. Any mismatch (missing labels, wrong length) falls back to the
    document-global assignment, so the feature is self-disabling.
    """
    return (
        expected_pages is not None
        and actual_pages is not None
        and len(expected_pages) == n_expected
        and len(actual_pages) == n_actual
    )


def assign_tables(
    cost_matrix: np.ndarray,
    expected_pages: list[int] | None = None,
    actual_pages: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Solve the GT->pred table assignment minimizing total cost.

    When ``expected_pages`` / ``actual_pages`` are provided and consistent with
    ``cost_matrix.shape`` (see :func:`page_blocking_active`), the assignment is
    solved independently per page: a GT table on page P can only pair with a
    prediction on page P. GT tables whose page has no predictions are left
    unmatched (they appear in neither returned list, so the caller scores them
    as unmatched-expected). Otherwise a single global assignment is solved.

    Returns ``(row_ind, col_ind)`` as index lists, mirroring
    ``scipy.optimize.linear_sum_assignment``.
    """
    n_expected, n_actual = cost_matrix.shape

    if not page_blocking_active(expected_pages, actual_pages, n_expected, n_actual):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind.tolist(), col_ind.tolist()

    assert expected_pages is not None and actual_pages is not None  # narrowed by the guard above

    pred_by_page: dict[int, list[int]] = defaultdict(list)
    for j, page in enumerate(actual_pages):
        pred_by_page[page].append(j)

    gt_by_page: dict[int, list[int]] = defaultdict(list)
    for i, page in enumerate(expected_pages):
        gt_by_page[page].append(i)

    row_out: list[int] = []
    col_out: list[int] = []
    for page, gt_idxs in gt_by_page.items():
        pred_idxs = pred_by_page.get(page)
        if not pred_idxs:
            continue
        sub = cost_matrix[np.ix_(gt_idxs, pred_idxs)]
        sub_rows, sub_cols = linear_sum_assignment(sub)
        for sr, sc in zip(sub_rows.tolist(), sub_cols.tolist(), strict=True):
            row_out.append(gt_idxs[sr])
            col_out.append(pred_idxs[sc])
    return row_out, col_out
