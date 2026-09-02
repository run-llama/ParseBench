"""GriTS (Grid Table Similarity) metric for HTML table comparison.

Computes content similarity between HTML tables using a grid-based
representation. GriTS_Con evaluates tables in their natural matrix form
via the factored 2D most-similar substructures (2D-MSS) algorithm.

Core algorithm adapted from the reference implementation at:
    https://github.com/microsoft/table-transformer/blob/main/src/grits.py

Reference paper:
    Smock, Pesala, Abraham. "GriTS: Grid Table Similarity Metric for
    Table Structure Recognition." ICDAR 2023.
    https://arxiv.org/abs/2203.12555
"""

import itertools
import os
from collections import defaultdict
from dataclasses import replace
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Literal, overload

import numpy as np
from lxml import html

from parse_bench.evaluation.metrics.base import Metric
from parse_bench.evaluation.metrics.parse.table_pairing import assign_tables, page_blocking_active
from parse_bench.evaluation.metrics.parse.table_parsing import (
    _ASCII_TO_SUBSCRIPT,
    _ASCII_TO_SUPERSCRIPT,
    TableData,
)
from parse_bench.evaluation.metrics.parse.utils import normalize_cell_text
from parse_bench.schemas.evaluation import MetricValue

# `pairing` schema in MetricValue.metadata (load-bearing for TRM consumption):
#   list[tuple[int, int | None]] of length n_gt
#   (gt_idx, pred_idx) for matched, (gt_idx, None) for unmatched GT.

# When one table has many cells AND the row/column counts differ by more
# than this factor, skip GriTS — the prediction is structurally wrong and
# the O(R1*C1*R2*C2) algorithm would take minutes for no useful signal.
DEFAULT_MIN_CELLS_FOR_MISMATCH_SKIP = 2500
DEFAULT_MAX_DIMENSION_RATIO = 1.5
DEFAULT_MISMATCH_SKIP_SCORE = 0.0

# =============================================================================
# Core GriTS algorithm (adapted from microsoft/table-transformer)
# =============================================================================


def _is_scalar(val: Any) -> bool:
    """Check if a value is a scalar (unoccupied grid cell), not a bbox list."""
    try:
        len(val)
        return False
    except TypeError:
        return True


def _bbox_iou(bbox1: Any, bbox2: Any) -> float:
    """Compute intersection-over-union of two [x1, y1, x2, y2] bounding boxes.

    Uses bounding-box union (area of the smallest enclosing rectangle) to
    match the reference GriTS implementation, which uses PyMuPDF
    Rect.include_rect for the union.

    Handles numpy arrays and scalar 0 (unoccupied grid cells).
    """
    bbox1_scalar = _is_scalar(bbox1)
    bbox2_scalar = _is_scalar(bbox2)

    # Both unoccupied → both tables agree "no cell here" → perfect match
    if bbox1_scalar and bbox2_scalar:
        return 1.0
    # One occupied, one not → structural mismatch
    if bbox1_scalar or bbox2_scalar:
        return 0.0

    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0

    # Intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    # Bounding-box union (smallest enclosing rectangle)
    union = (max(bbox1[2], bbox2[2]) - min(bbox1[0], bbox2[0])) * (max(bbox1[3], bbox2[3]) - min(bbox1[1], bbox2[1]))

    if union <= 0:
        return 0.0
    return intersection / union  # type: ignore[no-any-return]


@lru_cache(maxsize=200_000)
def _lcs_similarity_cached(s1: str, s2: str) -> float:
    """Ratcliff/Obershelp similarity of two strings, memoized.

    This is the numeric core of ``_lcs_similarity`` — kept byte-for-byte
    identical to the reference implementation (``difflib.SequenceMatcher``'s
    matching-block sum, i.e. ``2*|matches| / (|s1| + |s2|)``), so scores are
    unchanged. It is factored out purely so ``factored_2dmss`` — which calls
    this ``O(R1*C1*R2*C2)`` times per table pair, mostly on *repeated* cell
    strings (empty cells, spanned headers, duplicated values) — pays the
    ``SequenceMatcher`` cost only once per distinct ``(s1, s2)`` pair instead
    of once per grid position. Pure function ⇒ the cache is always exact.

    Measured (``scripts/bench_table_metrics.py``, GriTS-Con, random tables)::

        size    difflib ms   memoized ms   speedup   |Δ score|
        8x5          4.3         1.5         2.9x     0.00e+00
        15x8        38.5        12.7         3.0x     0.00e+00
        25x10      168.7        56.2         3.0x     0.00e+00
        40x12      644.7       243.6         2.6x     0.00e+00
    """
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    s = SequenceMatcher(None, s1, s2)
    lcs = "".join([s1[block.a : (block.a + block.size)] for block in s.get_matching_blocks()])
    return 2 * len(lcs) / (len(s1) + len(s2))


def _lcs_similarity(string1: Any, string2: Any) -> float:
    """Compute longest-common-subsequence similarity between two strings.

    Returns 2*|LCS| / (|s1| + |s2|), ranging from 0.0 (no overlap) to
    1.0 (identical strings). Returns 1.0 when both strings are empty.
    Handles non-string grid values (e.g., scalar 0 for unoccupied cells).

    Delegates to the memoized :func:`_lcs_similarity_cached` after coercing
    non-string grid values to ``str`` exactly as before, so the result is
    identical while avoiding recomputation for repeated cell pairs.
    """
    s1 = str(string1) if not isinstance(string1, str) else string1
    s2 = str(string2) if not isinstance(string2, str) else string2
    return _lcs_similarity_cached(s1, s2)


def _compute_fscore(num_true_positives: float, num_true: int, num_positives: int) -> tuple[float, float, float]:
    """Compute F-score, precision, and recall.

    Conventions (from the reference implementation):
    - precision is 1 when there are no predicted instances
    - recall is 1 when there are no true instances
    - fscore is 0 when recall or precision is 0
    """
    precision = num_true_positives / num_positives if num_positives > 0 else 1.0
    recall = num_true_positives / num_true if num_true > 0 else 1.0

    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return fscore, precision, recall


def _initialize_dp(seq1_len: int, seq2_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize dynamic programming score and pointer tables."""
    scores = np.zeros((seq1_len + 1, seq2_len + 1))
    pointers = np.zeros((seq1_len + 1, seq2_len + 1))

    for i in range(1, seq1_len + 1):
        pointers[i, 0] = -1  # up
    for j in range(1, seq2_len + 1):
        pointers[0, j] = 1  # left

    return scores, pointers


def _traceback(pointers: np.ndarray) -> tuple[list[int], list[int]]:
    """Traceback through DP pointer table to get aligned indices.

    Convention: -1 = up, 1 = left, 0 = diagonal (match).
    """
    i = pointers.shape[0] - 1
    j = pointers.shape[1] - 1
    seq1_indices: list[int] = []
    seq2_indices: list[int] = []

    while not (i == 0 and j == 0):
        if pointers[i, j] == -1:
            i -= 1
        elif pointers[i, j] == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
            seq1_indices.append(i)
            seq2_indices.append(j)

    return seq1_indices[::-1], seq2_indices[::-1]


def _align_1d(
    sequence1: list[tuple[int, int]],
    sequence2: list[tuple[int, int]],
    reward_lookup: dict[tuple[int, int, int, int], float],
    return_alignment: bool = False,
) -> float | tuple[list[int], list[int], float]:
    """1D sequence alignment with pre-computed rewards.

    Sequences are index tuples into the reward lookup table.
    """
    seq1_len = len(sequence1)
    seq2_len = len(sequence2)
    scores, pointers = _initialize_dp(seq1_len, seq2_len)

    for i in range(1, seq1_len + 1):
        for j in range(1, seq2_len + 1):
            reward = reward_lookup[sequence1[i - 1] + sequence2[j - 1]]
            diag = scores[i - 1, j - 1] + reward
            skip_seq2 = scores[i, j - 1]
            skip_seq1 = scores[i - 1, j]

            best = max(diag, skip_seq1, skip_seq2)
            scores[i, j] = best
            if diag == best:
                pointers[i, j] = 0
            elif skip_seq1 == best:
                pointers[i, j] = -1
            else:
                pointers[i, j] = 1

    score = float(scores[-1, -1])

    if not return_alignment:
        return score

    seq1_indices, seq2_indices = _traceback(pointers)
    return seq1_indices, seq2_indices, score


def _align_2d_outer(
    true_shape: tuple[int, int],
    pred_shape: tuple[int, int],
    reward_lookup: dict[tuple[int, int, int, int], float],
) -> tuple[list[int], list[int], float]:
    """2D sequence-of-sequences alignment.

    Aligns two outer sequences (rows) where match reward between entries
    is their 1D column alignment score.
    """
    scores, pointers = _initialize_dp(true_shape[0], pred_shape[0])

    for row_idx in range(1, true_shape[0] + 1):
        for col_idx in range(1, pred_shape[0] + 1):
            reward_result = _align_1d(
                [(row_idx - 1, tcol) for tcol in range(true_shape[1])],
                [(col_idx - 1, prow) for prow in range(pred_shape[1])],
                reward_lookup,
            )
            assert isinstance(reward_result, float)
            reward = reward_result
            diag = scores[row_idx - 1, col_idx - 1] + reward
            same_row = scores[row_idx, col_idx - 1]
            same_col = scores[row_idx - 1, col_idx]

            best = max(diag, same_col, same_row)
            scores[row_idx, col_idx] = best
            if diag == best:
                pointers[row_idx, col_idx] = 0
            elif same_col == best:
                pointers[row_idx, col_idx] = -1
            else:
                pointers[row_idx, col_idx] = 1

    score = float(scores[-1, -1])
    true_indices, pred_indices = _traceback(pointers)
    return true_indices, pred_indices, score


def _kernel_mode() -> str:
    """Select the ``factored_2dmss`` implementation: ``array`` | ``memo`` | ``slow``.

    ``array`` (default) builds the reward as a vectorized numpy tensor over a
    unique-cell-value similarity matrix and runs the alignment DP on Python lists
    — no per-pair LCS redundancy, no R1·C1·R2·C2 dict build, no numpy scalar
    indexing. ``memo`` is the previous fast path (memoized dict build). ``slow`` is
    the original dict path. All three return bit-identical scores; the non-array
    modes exist only for before/after benchmarking.

    ``BENCH_GRITS_KERNEL`` selects explicitly; the legacy boolean
    ``BENCH_GRITS_FAST_KERNEL=0`` forces ``slow``.
    """
    mode = os.environ.get("BENCH_GRITS_KERNEL", "").strip().lower()
    if mode in ("array", "memo", "slow"):
        return mode
    return "slow" if os.environ.get("BENCH_GRITS_FAST_KERNEL", "1") == "0" else "array"


def _precompute_rewards(
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    reward_function: Any,
    memoize: bool,
) -> tuple[dict[tuple[int, int, int, int], float], dict[tuple[int, int, int, int], float]]:
    """Original dict reward lookups over every cell pair (``slow`` / ``memo`` modes).

    The reward depends only on the two cell VALUES, but the grids carry the same
    string in many cells, so the naive ``range(R1·C1·R2·C2)`` loop calls
    ``reward_function`` (an LCS over strings) up to 1.7M times per pair with massive
    redundancy. ``memoize`` caches by ``(cell_value, cell_value)`` (a pure-function
    cache → identical rewards). The ``array`` kernel avoids this loop entirely.
    """
    pre_computed: dict[tuple[int, int, int, int], float] = {}
    transpose_rewards: dict[tuple[int, int, int, int], float] = {}
    product = itertools.product(
        range(true_grid.shape[0]),
        range(true_grid.shape[1]),
        range(pred_grid.shape[0]),
        range(pred_grid.shape[1]),
    )
    if memoize:
        cache: dict[tuple[Any, Any], float] = {}
        for trow, tcol, prow, pcol in product:
            value_key = (true_grid[trow, tcol], pred_grid[prow, pcol])
            reward = cache.get(value_key)
            if reward is None:
                reward = reward_function(*value_key)
                cache[value_key] = reward
            pre_computed[(trow, tcol, prow, pcol)] = reward
            transpose_rewards[(tcol, trow, pcol, prow)] = reward
    else:
        for trow, tcol, prow, pcol in product:
            reward = reward_function(true_grid[trow, tcol], pred_grid[prow, pcol])
            pre_computed[(trow, tcol, prow, pcol)] = reward
            transpose_rewards[(tcol, trow, pcol, prow)] = reward
    return pre_computed, transpose_rewards


def _reward_tensor(
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    reward_function: Any,
) -> np.ndarray:
    """Reward tensor ``R[trow, tcol, prow, pcol] = reward(true[trow,tcol], pred[prow,pcol])``.

    ``reward_function`` is evaluated once per *distinct* (true_value, pred_value)
    pair over a unique-string similarity matrix, then broadcast-gathered into the
    full R1·C1·R2·C2 tensor — no per-cell Python loop, no LCS redundancy. Values
    are the same doubles the dict path would store, so downstream scores are
    bit-identical.
    """
    true_uids: dict[Any, int] = {}
    pred_uids: dict[Any, int] = {}
    uid_true = np.empty(true_grid.shape, dtype=np.intp)
    uid_pred = np.empty(pred_grid.shape, dtype=np.intp)
    unique_true: list[Any] = []
    unique_pred: list[Any] = []
    for r in range(true_grid.shape[0]):
        for c in range(true_grid.shape[1]):
            value = true_grid[r, c]
            uid = true_uids.get(value)
            if uid is None:
                uid = len(unique_true)
                true_uids[value] = uid
                unique_true.append(value)
            uid_true[r, c] = uid
    for r in range(pred_grid.shape[0]):
        for c in range(pred_grid.shape[1]):
            value = pred_grid[r, c]
            uid = pred_uids.get(value)
            if uid is None:
                uid = len(unique_pred)
                pred_uids[value] = uid
                unique_pred.append(value)
            uid_pred[r, c] = uid

    sim = np.empty((len(unique_true), len(unique_pred)), dtype=np.float64)
    for a, value_a in enumerate(unique_true):
        for b, value_b in enumerate(unique_pred):
            sim[a, b] = reward_function(value_a, value_b)

    # Broadcast-gather: (R1,C1,1,1) x (1,1,R2,C2) -> (R1,C1,R2,C2)
    return sim[uid_true[:, :, None, None], uid_pred[None, None, :, :]]


def _traceback_list(pointers: list[list[int]]) -> tuple[list[int], list[int]]:
    """List-based traceback (mirror of :func:`_traceback`). -1=up, 1=left, 0=match."""
    i = len(pointers) - 1
    j = len(pointers[0]) - 1
    seq1_indices: list[int] = []
    seq2_indices: list[int] = []
    while not (i == 0 and j == 0):
        ptr = pointers[i][j]
        if ptr == -1:
            i -= 1
        elif ptr == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
            seq1_indices.append(i)
            seq2_indices.append(j)
    return seq1_indices[::-1], seq2_indices[::-1]


@overload
def _align_matrix(reward_matrix: list[list[float]], return_alignment: Literal[False] = False) -> float: ...


@overload
def _align_matrix(
    reward_matrix: list[list[float]], return_alignment: Literal[True]
) -> tuple[list[int], list[int], float]: ...


def _align_matrix(
    reward_matrix: list[list[float]],
    return_alignment: bool = False,
) -> float | tuple[list[int], list[int], float]:
    """Max-reward monotonic alignment over a dense ``M×N`` reward matrix.

    Identical DP and tie-breaking to :func:`_align_1d` / :func:`_align_2d_outer`
    (``best = max(diag, up, left)``; tie prefers diag, then up, then left), but on
    plain Python lists — no numpy scalar indexing, no 4-tuple dict keys.
    """
    m = len(reward_matrix)
    n = len(reward_matrix[0]) if m else 0
    scores = [[0.0] * (n + 1) for _ in range(m + 1)]
    pointers = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        pointers[i][0] = -1  # up
    for j in range(1, n + 1):
        pointers[0][j] = 1  # left

    for i in range(1, m + 1):
        reward_row = reward_matrix[i - 1]
        scores_prev = scores[i - 1]
        scores_cur = scores[i]
        pointers_cur = pointers[i]
        for j in range(1, n + 1):
            diag = scores_prev[j - 1] + reward_row[j - 1]
            skip_up = scores_prev[j]
            skip_left = scores_cur[j - 1]
            best = max(diag, skip_up, skip_left)
            scores_cur[j] = best
            if diag == best:
                pointers_cur[j] = 0
            elif skip_up == best:
                pointers_cur[j] = -1
            else:
                pointers_cur[j] = 1

    score = scores[m][n]
    if not return_alignment:
        return score
    seq1_indices, seq2_indices = _traceback_list(pointers)
    return seq1_indices, seq2_indices, score


def _factored_2dmss_array(
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    reward_function: Any,
) -> tuple[float, float, float, float, dict[int, int], dict[int, int]]:
    """Array-kernel ``factored_2dmss`` — bit-identical to the dict path.

    Builds the reward tensor once (vectorized), then does the same factored
    row/column alignments. The per-row-pair column score and per-col-pair row
    score matrices (``RR`` / ``CC``) are exactly the inner ``_align_1d`` scores the
    dict path recomputes inside ``_align_2d_outer``. ``positive_match`` is summed in
    the original nested order so floating-point rounding matches to the last bit.
    """
    reward = _reward_tensor(true_grid, pred_grid, reward_function)
    n_true_rows, n_true_cols, n_pred_rows, n_pred_cols = reward.shape

    # Row alignment: reward between true row i and pred row j is their column
    # alignment score over reward[i, :, j, :] (C1×C2).
    rr = [[_align_matrix(reward[i, :, j, :].tolist()) for j in range(n_pred_rows)] for i in range(n_true_rows)]
    true_row_nums, pred_row_nums, row_score = _align_matrix(rr, return_alignment=True)

    # Column alignment: reward between true col i and pred col j is their row
    # alignment score over reward[:, i, :, j] (R1×R2).
    cc = [[_align_matrix(reward[:, i, :, j].tolist()) for j in range(n_pred_cols)] for i in range(n_true_cols)]
    true_col_nums, pred_col_nums, col_score = _align_matrix(cc, return_alignment=True)

    num_true = n_true_rows * n_true_cols
    num_pos = n_pred_rows * n_pred_cols

    upper_bound = min(row_score, col_score)
    ub_fscore, _, _ = _compute_fscore(upper_bound, num_true, num_pos)

    # Sum matched-cell rewards in the same (row-pair outer, col-pair inner) order
    # as the dict path so the float reduction is bit-identical.
    positive_match = 0.0
    for true_row, pred_row in zip(true_row_nums, pred_row_nums, strict=True):
        for true_col, pred_col in zip(true_col_nums, pred_col_nums, strict=True):
            positive_match += float(reward[true_row, true_col, pred_row, pred_col])

    fscore, precision, recall = _compute_fscore(positive_match, num_true, num_pos)
    row_map = dict(zip(true_row_nums, pred_row_nums, strict=True))
    col_map = dict(zip(true_col_nums, pred_col_nums, strict=True))
    return fscore, precision, recall, ub_fscore, row_map, col_map


def factored_2dmss(
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    reward_function: Any,
) -> tuple[float, float, float, float]:
    """Factored 2D most-similar substructures (2D-MSS).

    A polynomial-time heuristic for the NP-hard 2D-MSS problem. Finds
    the substructures of two matrices with the greatest total similarity.

    Returns (fscore, precision, recall, upper_bound_score).
    """
    mode = _kernel_mode()
    if mode == "array":
        fscore, precision, recall, ub_fscore, _, _ = _factored_2dmss_array(true_grid, pred_grid, reward_function)
        return fscore, precision, recall, ub_fscore

    pre_computed, transpose_rewards = _precompute_rewards(
        true_grid, pred_grid, reward_function, memoize=(mode == "memo")
    )

    num_pos = pred_grid.shape[0] * pred_grid.shape[1]
    num_true = true_grid.shape[0] * true_grid.shape[1]

    true_row_nums, pred_row_nums, row_score = _align_2d_outer(true_grid.shape[:2], pred_grid.shape[:2], pre_computed)

    true_col_nums, pred_col_nums, col_score = _align_2d_outer(
        true_grid.shape[:2][::-1],
        pred_grid.shape[:2][::-1],
        transpose_rewards,
    )

    upper_bound = min(row_score, col_score)
    ub_fscore, _, _ = _compute_fscore(upper_bound, num_true, num_pos)

    positive_match = 0.0
    for true_row, pred_row in zip(true_row_nums, pred_row_nums, strict=False):
        for true_col, pred_col in zip(true_col_nums, pred_col_nums, strict=False):
            positive_match += pre_computed[(true_row, true_col, pred_row, pred_col)]

    fscore, precision, recall = _compute_fscore(positive_match, num_true, num_pos)
    return fscore, precision, recall, ub_fscore


def factored_2dmss_with_alignment(
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    reward_function: Any,
) -> tuple[float, float, float, float, dict[int, int], dict[int, int]]:
    """Like factored_2dmss, but also returns row and column alignment maps.

    Returns (fscore, precision, recall, upper_bound, row_map, col_map)
    where row_map = {true_row: pred_row} and col_map = {true_col: pred_col}.
    """
    mode = _kernel_mode()
    if mode == "array":
        return _factored_2dmss_array(true_grid, pred_grid, reward_function)

    pre_computed, transpose_rewards = _precompute_rewards(
        true_grid, pred_grid, reward_function, memoize=(mode == "memo")
    )

    num_pos = pred_grid.shape[0] * pred_grid.shape[1]
    num_true = true_grid.shape[0] * true_grid.shape[1]

    true_row_nums, pred_row_nums, row_score = _align_2d_outer(true_grid.shape[:2], pred_grid.shape[:2], pre_computed)

    true_col_nums, pred_col_nums, col_score = _align_2d_outer(
        true_grid.shape[:2][::-1],
        pred_grid.shape[:2][::-1],
        transpose_rewards,
    )

    row_map = dict(zip(true_row_nums, pred_row_nums, strict=True))
    col_map = dict(zip(true_col_nums, pred_col_nums, strict=True))

    upper_bound = min(row_score, col_score)
    ub_fscore, _, _ = _compute_fscore(upper_bound, num_true, num_pos)

    positive_match = 0.0
    for true_row, pred_row in zip(true_row_nums, pred_row_nums, strict=False):
        for true_col, pred_col in zip(true_col_nums, pred_col_nums, strict=False):
            positive_match += pre_computed[(true_row, true_col, pred_row, pred_col)]

    fscore, precision, recall = _compute_fscore(positive_match, num_true, num_pos)
    return fscore, precision, recall, ub_fscore, row_map, col_map


# =============================================================================
# HTML table parsing
# =============================================================================


def html_to_cells(table_html: str) -> list[dict[str, Any]] | None:
    """Parse an HTML table string into a list of cell dictionaries.

    Each cell dict has keys: row_nums, column_nums, is_column_header, cell_text.
    Returns None if parsing fails.
    """
    try:
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        doc = html.fromstring(table_html, parser=parser)
    except Exception:
        return None

    # Find the <table> element (lxml may wrap in <html><body>)
    if doc.tag == "table":
        tree = doc
    else:
        tables = doc.xpath(".//table")
        if not tables:
            return None
        tree = tables[0]

    table_cells: list[dict[str, Any]] = []
    occupied_columns_by_row: dict[int, set[int]] = defaultdict(set)
    current_row = -1

    stack: list[tuple[Any, bool]] = [(tree, False)]
    while stack:
        current, in_header = stack.pop()

        if current.tag == "tr":
            current_row += 1

        if current.tag in ("td", "th"):
            colspan = int(current.attrib.get("colspan", "1"))
            rowspan = int(current.attrib.get("rowspan", "1"))
            row_nums = list(range(current_row, current_row + rowspan))

            occupied = occupied_columns_by_row[current_row]
            if occupied:
                max_occ = max(occupied)
                current_column = min(set(range(max_occ + 2)).difference(occupied))
            else:
                current_column = 0

            column_nums = list(range(current_column, current_column + colspan))
            for rn in row_nums:
                occupied_columns_by_row[rn].update(column_nums)

            # Convert <sup>/<sub> digit content to Unicode equivalents
            # so that "Name<sup>1</sup>" becomes "Name¹", matching sources
            # that already use Unicode superscripts.
            _map = {"sup": _ASCII_TO_SUPERSCRIPT, "sub": _ASCII_TO_SUBSCRIPT}
            for sup_sub in current.xpath(".//sup | .//sub"):
                char_map = _map[sup_sub.tag]
                converted = "".join(char_map.get(c, "") for c in (sup_sub.text or ""))
                # Replace element with converted text in parent's tree
                prev = sup_sub.getprevious()
                if prev is not None:
                    prev.tail = (prev.tail or "") + converted + (sup_sub.tail or "")
                else:
                    sup_sub.getparent().text = (sup_sub.getparent().text or "") + converted + (sup_sub.tail or "")
                sup_sub.getparent().remove(sup_sub)
            # Gather text from element and all descendants
            cell_text = normalize_cell_text(" ".join(current.itertext()))

            table_cells.append(
                {
                    "row_nums": row_nums,
                    "column_nums": column_nums,
                    "is_column_header": current.tag == "th" or in_header,
                    "cell_text": cell_text,
                }
            )

        children = list(current)
        for child in children[::-1]:
            stack.append((child, in_header or current.tag in ("th", "thead")))

    return table_cells


def cells_to_grid(cells: list[dict[str, Any]], key: str = "cell_text") -> list[list[Any]]:
    """Convert cell list to a 2D grid keyed by 'cell_text' or 'bbox'.

    For GriTS_Con, use key='cell_text'.
    """
    if not cells:
        return [[]]
    num_rows = max(max(c["row_nums"]) for c in cells) + 1
    num_cols = max(max(c["column_nums"]) for c in cells) + 1
    grid: list[list[Any]] = [[0] * num_cols for _ in range(num_rows)]
    for cell in cells:
        for rn in cell["row_nums"]:
            for cn in cell["column_nums"]:
                grid[rn][cn] = cell[key]
    return grid


# =============================================================================
# High-level GriTS computation from HTML
# =============================================================================


def grits_con(true_text_grid: np.ndarray, pred_text_grid: np.ndarray) -> tuple[float, float, float, float]:
    """Compute GriTS_Con (content) from text grids."""
    return factored_2dmss(true_text_grid, pred_text_grid, _lcs_similarity)


def grits_con_with_alignment(
    true_text_grid: np.ndarray, pred_text_grid: np.ndarray
) -> tuple[float, float, float, float, dict[int, int], dict[int, int]]:
    """GriTS_Con that also returns row/col alignment maps."""
    return factored_2dmss_with_alignment(true_text_grid, pred_text_grid, _lcs_similarity)


def grits_from_html(
    true_html: str,
    pred_html: str,
    min_cells_for_mismatch_skip: int = DEFAULT_MIN_CELLS_FOR_MISMATCH_SKIP,
    max_dimension_ratio: float = DEFAULT_MAX_DIMENSION_RATIO,
    mismatch_skip_score: float = DEFAULT_MISMATCH_SKIP_SCORE,
) -> dict[str, Any] | None:
    """Compute GriTS_Con from two HTML table strings.

    Args:
        true_html: Ground-truth HTML table string.
        pred_html: Predicted HTML table string.

    Returns a dict with keys: grits_con and its precision/recall/upper_bound
    variants, plus alignment maps. Returns None if parsing fails.
    """
    true_cells = html_to_cells(true_html)
    pred_cells = html_to_cells(pred_html)

    if true_cells is None or pred_cells is None:
        return None
    if not true_cells or not pred_cells:
        return None

    true_text = np.array(cells_to_grid(true_cells, key="cell_text"), dtype=object)
    pred_text = np.array(cells_to_grid(pred_cells, key="cell_text"), dtype=object)

    true_rows = max(max(c["row_nums"]) for c in true_cells) + 1
    true_cols = max(max(c["column_nums"]) for c in true_cells) + 1
    pred_rows = max(max(c["row_nums"]) for c in pred_cells) + 1
    pred_cols = max(max(c["column_nums"]) for c in pred_cells) + 1

    true_cells_count = true_rows * true_cols
    pred_cells_count = pred_rows * pred_cols

    # Skip when tables are large and dimensions are badly mismatched —
    # the prediction is structurally wrong so GriTS won't be informative,
    # and the O(R1*C1*R2*C2) cost would be extreme.
    larger_cells = max(true_cells_count, pred_cells_count)
    if larger_cells >= min_cells_for_mismatch_skip:
        row_ratio = max(true_rows, pred_rows) / max(min(true_rows, pred_rows), 1)
        col_ratio = max(true_cols, pred_cols) / max(min(true_cols, pred_cols), 1)
        if row_ratio > max_dimension_ratio or col_ratio > max_dimension_ratio:
            print(
                f"  GriTS: skipping — large table ({true_rows}x{true_cols} vs "
                f"{pred_rows}x{pred_cols}) with dimension ratio "
                f"{max(row_ratio, col_ratio):.1f}x > {max_dimension_ratio}x threshold, "
                f"scoring {mismatch_skip_score}",
                flush=True,
            )
            s = mismatch_skip_score
            return {
                "grits_con": s,
                "grits_precision_con": s,
                "grits_recall_con": s,
                "grits_con_upper_bound": s,
                "_con_row_alignment": {},
                "_con_col_alignment": {},
            }

    metrics: dict[str, Any] = {}
    (
        metrics["grits_con"],
        metrics["grits_precision_con"],
        metrics["grits_recall_con"],
        metrics["grits_con_upper_bound"],
        row_map,
        col_map,
    ) = grits_con_with_alignment(true_text, pred_text)
    metrics["_con_row_alignment"] = row_map
    metrics["_con_col_alignment"] = col_map

    return metrics


def _leading_title_band_text(td: TableData) -> str | None:
    """Normalized text of a leading full-width title band, if the table has one.

    A title band is row 0 rendered as one colspan cell: after span expansion
    every column of row 0 carries the same non-empty text. Tables with only
    one row, or only one column, are excluded — the first has no body left to
    score once the band is dropped, the second cannot distinguish a band from
    an ordinary cell.
    """
    n_rows, n_cols = td.data.shape
    if n_rows < 2 or n_cols < 2:
        return None
    texts = {normalize_cell_text(str(td.data[0, c])) for c in range(n_cols)}
    if len(texts) != 1:
        return None
    text = texts.pop()
    return text or None


def _drop_caption_matched_title_band(td: TableData, other_caption: str) -> TableData:
    """Drop ``td``'s leading title band when ``other_caption`` says the same thing.

    ``<caption>`` is the semantically correct place for a table's title, and a
    prediction that uses it must not be scored as having lost the title row
    that a ground truth spells as a full-width header band (or vice versa).
    The caption never enters either grid — it is not a ``<td>``/``<th>`` — so
    the band on the other side is the only asymmetry, and removing it makes
    the two structures comparable.

    Guarded by text equality under the ordinary cell normalization, so a band
    that says something the caption does not is left in place and still
    scores. Applied in both directions by the caller, so the treatment is
    symmetric between ground truth and prediction.
    """
    caption = normalize_cell_text(other_caption)
    if not caption:
        return td
    band = _leading_title_band_text(td)
    if band is None or band != caption:
        return td
    return replace(td, data=td.data[1:, :])


def grits_con_from_table_data(
    gt_td: TableData,
    pred_td: TableData,
    min_cells_for_mismatch_skip: int = DEFAULT_MIN_CELLS_FOR_MISMATCH_SKIP,
    max_dimension_ratio: float = DEFAULT_MAX_DIMENSION_RATIO,
    mismatch_skip_score: float = DEFAULT_MISMATCH_SKIP_SCORE,
) -> dict[str, Any] | None:
    """Compute GriTS_Con from two parsed ``TableData`` objects.

    Reads the resolved 2D grid from ``td.data`` directly (no HTML re-parsing)
    and applies the upgraded ``normalize_cell_text``. P5 entry point — replaces
    the older ``grits_from_html`` path on the GriTS hot path.
    """
    if gt_td.data.size == 0 or pred_td.data.size == 0:
        return None

    # A <caption> on one side and a full-width title band saying the same
    # thing on the other are the same title rendered two ways; neutralize the
    # band so the difference costs nothing. Both directions, so the treatment
    # is symmetric.
    if pred_td.caption:
        gt_td = _drop_caption_matched_title_band(gt_td, pred_td.caption)
    if gt_td.caption:
        pred_td = _drop_caption_matched_title_band(pred_td, gt_td.caption)
    if gt_td.data.size == 0 or pred_td.data.size == 0:
        return None

    true_rows, true_cols = gt_td.data.shape
    pred_rows, pred_cols = pred_td.data.shape

    # Skip when tables are large and dimensions are badly mismatched —
    # the prediction is structurally wrong so GriTS won't be informative,
    # and the O(R1*C1*R2*C2) cost would be extreme.
    larger_cells = max(true_rows * true_cols, pred_rows * pred_cols)
    if larger_cells >= min_cells_for_mismatch_skip:
        row_ratio = max(true_rows, pred_rows) / max(min(true_rows, pred_rows), 1)
        col_ratio = max(true_cols, pred_cols) / max(min(true_cols, pred_cols), 1)
        if row_ratio > max_dimension_ratio or col_ratio > max_dimension_ratio:
            print(
                f"  GriTS: skipping — large table ({true_rows}x{true_cols} vs "
                f"{pred_rows}x{pred_cols}) with dimension ratio "
                f"{max(row_ratio, col_ratio):.1f}x > {max_dimension_ratio}x threshold, "
                f"scoring {mismatch_skip_score}",
                flush=True,
            )
            s = mismatch_skip_score
            return {
                "grits_con": s,
                "grits_precision_con": s,
                "grits_recall_con": s,
                "grits_con_upper_bound": s,
                "_con_row_alignment": {},
                "_con_col_alignment": {},
            }

    true_text = np.empty_like(gt_td.data)
    for r in range(true_rows):
        for c in range(true_cols):
            true_text[r, c] = normalize_cell_text(str(gt_td.data[r, c]))
    pred_text = np.empty_like(pred_td.data)
    for r in range(pred_rows):
        for c in range(pred_cols):
            pred_text[r, c] = normalize_cell_text(str(pred_td.data[r, c]))

    metrics: dict[str, Any] = {}
    (
        metrics["grits_con"],
        metrics["grits_precision_con"],
        metrics["grits_recall_con"],
        metrics["grits_con_upper_bound"],
        row_map,
        col_map,
    ) = grits_con_with_alignment(true_text, pred_text)
    metrics["_con_row_alignment"] = row_map
    metrics["_con_col_alignment"] = col_map

    return metrics


# =============================================================================
# Module-level helper for parallel pairwise computation
# (must be top-level so ProcessPoolExecutor can pickle it)
# =============================================================================

_ZERO_RESULT: dict[str, Any] = {
    "grits_con": 0.0,
    "grits_precision_con": 0.0,
    "grits_recall_con": 0.0,
    "grits_con_upper_bound": 0.0,
    "_con_row_alignment": {},
    "_con_col_alignment": {},
}


# =============================================================================
# GriTSMetric class (Metric interface)
# =============================================================================


class GriTSMetric(Metric):
    """GriTS metric for comparing HTML tables in markdown content.

    Computes Grid Table Similarity (content / GriTS_Con) between expected
    and actual HTML tables. Uses the Hungarian algorithm for optimal table
    matching when documents contain multiple tables.
    """

    @property
    def name(self) -> str:
        """Return the name of this metric."""
        return "grits"

    def compute(  # type: ignore[override]
        self,
        expected_tables: list[Any],
        actual_tables: list[Any],
        *,
        expected_pages: list[int] | None = None,
        actual_pages: list[int] | None = None,
        **kwargs: Any,
    ) -> list[MetricValue]:
        """Compute GriTS_Con scores between expected and actual table sets.

        Consumes pre-extracted ``ExtractedTable`` lists from the shared
        ``extract_table_pairs`` stage so that GriTS and TRM provably see
        the same tables. The lift is purely "stop calling extract_html_tables
        yourself" — internal scoring (html_to_cells → cells_to_grid → grits_con,
        Hungarian assignment) is unchanged from main.

        Args:
            expected_tables: Pre-extracted GT tables (``list[ExtractedTable]``).
            actual_tables: Pre-extracted predicted tables (``list[ExtractedTable]``).
            expected_pages: Optional page number (1-indexed) per GT table, aligned
                with ``expected_tables``. When supplied together with
                ``actual_pages`` (and length-consistent), the GT->pred assignment
                is solved per page so a GT table only pairs with a prediction on
                the same page. Omit for the document-global assignment.
            actual_pages: Optional page number per predicted table, aligned with
                ``actual_tables``.
            kwargs: Additional parameters (not used)

        Returns:
            List with a single MetricValue for grits_con.
        """
        # P5: read TableData directly from the ExtractedTable inputs and
        # apply the upgraded normalize_cell_text. The raw_html field is no
        # longer touched on the GriTS hot path.
        expected_td = [et.table_data for et in expected_tables]
        actual_td = [et.table_data for et in actual_tables]

        shared_meta: dict[str, Any] = {}

        if not expected_td:
            shared_meta = {
                "note": "No tables found in expected markdown",
                "tables_found_expected": 0,
                "tables_found_actual": len(actual_td),
                "pairing": [],
            }
            return [MetricValue(metric_name="grits_con", value=0.0, metadata=shared_meta)]

        if not actual_td:
            shared_meta = {
                "note": "No tables found in actual markdown",
                "tables_found_expected": len(expected_td),
                "tables_found_actual": 0,
                "tables_matched": 0,
                "pairing": [(i, None) for i in range(len(expected_td))],
            }
            return [MetricValue(metric_name="grits_con", value=0.0, metadata=shared_meta)]

        n_expected = len(expected_td)
        n_actual = len(actual_td)

        # When per-table page labels are supplied, only same-page pairs are
        # eligible to match — so only those need a GriTS comparison. This both
        # constrains the assignment to within-page pairs and avoids the O(n_gt *
        # n_pred) blow-up on multi-page documents.
        blocked = page_blocking_active(expected_pages, actual_pages, n_expected, n_actual)
        allowed_pairs = [
            (i, j)
            for i in range(n_expected)
            for j in range(n_actual)
            if not blocked or expected_pages[i] == actual_pages[j]  # type: ignore[index]
        ]
        total_pairs = len(allowed_pairs)

        print(
            f"  GriTS: comparing {n_expected} expected x {n_actual} actual = {total_pairs} table pair(s)"
            f"{' (same-page only)' if blocked else ''}",
            flush=True,
        )

        # Compute pairwise GriTS scores for the eligible pairs only. Disallowed
        # (cross-page) cells keep cost 0.0 and are never read — the per-page
        # assignment only ever selects same-page pairs.
        results_cache: dict[tuple[int, int], dict[str, Any]] = {}
        cost_matrix = np.zeros((n_expected, n_actual))

        for pair_idx, (i, j) in enumerate(allowed_pairs, start=1):
            if total_pairs > 1:
                print(f"  GriTS: table pair {pair_idx}/{total_pairs}", flush=True)
            maybe_result = grits_con_from_table_data(expected_td[i], actual_td[j])
            result = maybe_result if maybe_result is not None else dict(_ZERO_RESULT)
            results_cache[(i, j)] = result
            cost_matrix[i, j] = -result["grits_con"]

        # Solve assignment (per page when page labels are supplied, else global).
        row_ind, col_ind = assign_tables(cost_matrix, expected_pages, actual_pages)

        per_table_details: list[dict[str, Any]] = []
        con_scores: list[float] = []
        matched_gt: set[int] = set()

        for gt_idx, pred_idx in zip(row_ind, col_ind, strict=True):
            gi, pi = int(gt_idx), int(pred_idx)
            result = results_cache[(gi, pi)]
            con_scores.append(result["grits_con"])
            detail: dict[str, Any] = {
                "gt_table_index": gi,
                "pred_table_index": pi,
                "grits_con": result["grits_con"],
                "grits_precision_con": result["grits_precision_con"],
                "grits_recall_con": result["grits_recall_con"],
                "_con_row_alignment": result.get("_con_row_alignment", {}),
                "_con_col_alignment": result.get("_con_col_alignment", {}),
            }
            if blocked:
                detail["gt_page"] = expected_pages[gi]  # type: ignore[index]
                detail["pred_page"] = actual_pages[pi]  # type: ignore[index]
            per_table_details.append(detail)
            matched_gt.add(gi)

        # Unmatched expected tables score 0
        for i in range(n_expected):
            if i not in matched_gt:
                con_scores.append(0.0)
                unmatched_detail: dict[str, Any] = {
                    "gt_table_index": i,
                    "pred_table_index": None,
                    "grits_con": 0.0,
                    "note": "No matching table in actual",
                }
                if blocked:
                    unmatched_detail["gt_page"] = expected_pages[i]  # type: ignore[index]
                per_table_details.append(unmatched_detail)

        avg_con = sum(con_scores) / len(con_scores) if con_scores else 0.0
        print(f"  GriTS: done, con = {avg_con:.4f}", flush=True)

        # Build the load-bearing pairing key consumed by TRM and the
        # evaluator's count metrics: list[(gt_idx, pred_idx | None)] of
        # length n_expected. Unmatched GT tables get None.
        pairing: list[tuple[int, int | None]] = []
        for i in range(n_expected):
            if i in matched_gt:
                # Find the matched pred index from row_ind/col_ind
                for gi, pi in zip(row_ind, col_ind, strict=True):
                    if int(gi) == i:
                        pairing.append((i, int(pi)))
                        break
            else:
                pairing.append((i, None))

        shared_meta = {
            "tables_found_expected": n_expected,
            "tables_found_actual": n_actual,
            "tables_matched": len(row_ind),
            "per_table_details": per_table_details,
            "pairing": pairing,
        }

        # Build human-readable detail strings
        details: list[str] = []
        details.append(f"{n_expected} table(s) expected, {n_actual} found, {len(row_ind)} matched")
        for td in per_table_details:
            gt_i: int = td["gt_table_index"]
            pr_i: int | None = td.get("pred_table_index")
            if pr_i is None:
                details.append(f"Table {gt_i + 1}: no match found in prediction")
            else:
                details.append(
                    f"Table {gt_i + 1}: con={td['grits_con']:.3f}"
                    f" (precision={td.get('grits_precision_con', 0):.2f},"
                    f" recall={td.get('grits_recall_con', 0):.2f})"
                )

        return [
            MetricValue(
                metric_name="grits_con",
                value=avg_con,
                metadata=shared_meta,
                details=details,
            ),
        ]
