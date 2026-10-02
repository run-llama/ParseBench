"""Ambiguous-merged-table splitting, lifted upstream of GriTS/TEDS.

When a model concatenates several side-by-side tables into one wide table
with a repeating column header period, this module detects the period and
splits merged preds into sub-tables. Runs after ``extract_table_pairs`` so
GriTS sees the split sub-tables instead of the merged blobs.

Selection across pred tables on a page is performed jointly: each pred
table contributes a list of candidate ``SplitOption``s (always including a
no-split sentinel), and ``select_joint_split`` enumerates the Cartesian
product over the variable tables only and picks the combination whose
total post-split table count is closest to ``len(expected)``, breaking ties
by total repeating-header rows then total period. The chosen combination
is only applied if it strictly beats the all-no-split baseline.

The TRM-side primitives (``normalize_table``, ``extract_header_info``,
``HeaderInfo``, ``_resolve_header_row_values``, ``_COLUMN_MATCH_THRESHOLD``)
are imported lazily inside function bodies to avoid a module-level
circular import with ``table_record_match_metric``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from parse_bench.evaluation.metrics.parse.table_extraction import ExtractedTable
from parse_bench.evaluation.metrics.parse.table_parsing import TableData

if TYPE_CHECKING:
    from parse_bench.evaluation.metrics.parse.table_record_match_metric import HeaderInfo


_SAFETY_CAP = 256


def _row_repeats_with_period(row_vals: list[str], period: int) -> bool:
    """Check if a header row's values repeat with the given period."""
    from parse_bench.evaluation.metrics.parse.table_record_match_metric import (
        _COLUMN_MATCH_THRESHOLD,
    )

    n_cols = len(row_vals)
    n_segments = n_cols // period
    first_segment = row_vals[0:period]
    for seg_idx in range(1, n_segments):
        segment = row_vals[seg_idx * period : (seg_idx + 1) * period]
        matches = sum(
            1
            for a, b in zip(first_segment, segment, strict=True)
            if (fuzz.ratio(a.lower(), b.lower()) / 100.0 >= _COLUMN_MATCH_THRESHOLD or (not a and not b))
        )
        if matches < period * 0.8:
            return False
    return True


def _detect_period_candidates(
    table: TableData,
    header: HeaderInfo,
) -> list[tuple[int, int]]:
    """Return all valid ``(period, n_repeating_rows)`` candidates for ``table``.

    Unlike the previous version, no filtering by GT table count is applied —
    callers (specifically the joint selector) are responsible for choosing
    among the candidates.
    """
    from parse_bench.evaluation.metrics.parse.table_record_match_metric import (
        _resolve_header_row_values,
    )

    if not header.col_header_rows:
        return []

    n_cols = table.data.shape[1]
    if n_cols < 2:
        return []

    header_row_values = _resolve_header_row_values(table, header)
    if not header_row_values:
        return []

    candidates: list[tuple[int, int]] = []
    for P in range(1, n_cols // 2 + 1):
        if n_cols % P != 0:
            continue
        n_segments = n_cols // P
        if n_segments < 2:
            continue

        n_repeating_rows = sum(1 for row_vals in header_row_values if _row_repeats_with_period(row_vals, P))

        if n_repeating_rows > 0:
            candidates.append((P, n_repeating_rows))

    return candidates


def _trim_trailing_padding(table: TableData) -> TableData:
    """Remove only empty tail rows, preserving metadata for retained rows."""
    end = table.data.shape[0]
    # Metadata can carry header text even when the grid cell is empty.
    header_text_rows = {r for entries in table.col_headers.values() for r, text in entries if str(text).strip()}
    header_text_rows.update(
        r for r, entries in table.row_headers.items() if any(str(text).strip() for _, text in entries)
    )
    while end and end - 1 not in header_text_rows and not any(str(v).strip() for v in table.data[end - 1]):
        end -= 1
    if end == table.data.shape[0]:
        return table
    row_sets = ("header_rows", "thead_rows", "tbody_rows", "tfoot_rows", "column_scope_rows", "row_scope_rows")
    return replace(
        table,
        data=table.data[:end],
        **{name: {r for r in getattr(table, name) if r < end} for name in row_sets},
        col_headers={c: [(r, text) for r, text in entries if r < end] for c, entries in table.col_headers.items()},
        row_headers={r: entries for r, entries in table.row_headers.items() if r < end},
        header_cells={(r, c) for r, c in table.header_cells if r < end},
        spanned_cells={(r, c) for r, c in table.spanned_cells if r < end},
    )


def build_sub_table(
    pred_table: TableData,
    start: int,
    end: int,
) -> TableData:
    """Slice columns and trim blank tail rows without losing header relationships."""
    return _trim_trailing_padding(
        replace(
            pred_table,
            data=pred_table.data[:, start:end],
            header_cols={c - start for c in pred_table.header_cols if start <= c < end},
            col_headers={c - start: entries for c, entries in pred_table.col_headers.items() if start <= c < end},
            row_headers={
                r: [(c - start, text) for c, text in entries if start <= c < end]
                for r, entries in pred_table.row_headers.items()
                if any(start <= c < end for c, _ in entries)
            },
            header_cells={(r, c - start) for r, c in pred_table.header_cells if start <= c < end},
            spanned_cells={(r, c - start) for r, c in pred_table.spanned_cells if start <= c < end},
        )
    )


@dataclass(frozen=True)
class SplitOption:
    """One possible outcome for a single pred table.

    The no-split sentinel is ``SplitOption(n_segments=1, n_repeating_rows=0,
    period=0, sub_tables=None)``. A real split has ``sub_tables`` populated.
    """

    n_segments: int
    n_repeating_rows: int
    period: int
    sub_tables: tuple[TableData, ...] | None


_NO_SPLIT = SplitOption(n_segments=1, n_repeating_rows=0, period=0, sub_tables=None)


def enumerate_split_options(pred_table: TableData) -> list[SplitOption]:
    """Return all split options for ``pred_table``, always including no-split.

    The first element is always the no-split sentinel. Each detected period
    contributes one additional option whose ``sub_tables`` are the actual
    column-sliced ``TableData`` instances.
    """
    from parse_bench.evaluation.metrics.parse.table_title_stripping import (
        extract_header_info,
    )

    options: list[SplitOption] = [_NO_SPLIT]

    header = extract_header_info(pred_table)
    candidates = _detect_period_candidates(pred_table, header)
    if not candidates:
        return options

    n_cols = pred_table.data.shape[1]
    for period, n_repeating_rows in candidates:
        n_segments = n_cols // period
        sub_tables = tuple(
            build_sub_table(pred_table, seg_idx * period, (seg_idx + 1) * period) for seg_idx in range(n_segments)
        )
        options.append(
            SplitOption(
                n_segments=n_segments,
                n_repeating_rows=n_repeating_rows,
                period=period,
                sub_tables=sub_tables,
            )
        )
    return options


def select_joint_split(
    actual: list[ExtractedTable],
    n_expected: int,
) -> list[SplitOption] | None:
    """Pick a per-table SplitOption for each pred table jointly.

    Returns one ``SplitOption`` per input table (same order, same length)
    when the chosen combination strictly beats the all-no-split baseline
    under the lexicographic objective ``(|total_segments - n_expected|,
    -sum(n_repeating_rows), -sum(period))``. Returns ``None`` when no
    improvement exists or when the variable-tables Cartesian product
    exceeds the safety cap.
    """
    from parse_bench.evaluation.metrics.parse.table_record_match_metric import (
        normalize_table,
    )

    if not actual:
        return None

    per_table_options: list[list[SplitOption]] = [
        enumerate_split_options(normalize_table(t.table_data)) for t in actual
    ]

    variable_indices = [i for i, opts in enumerate(per_table_options) if len(opts) >= 2]
    fixed_indices = [i for i, opts in enumerate(per_table_options) if len(opts) == 1]

    if not variable_indices:
        return None

    cap_product = 1
    for i in variable_indices:
        cap_product *= len(per_table_options[i])
        if cap_product > _SAFETY_CAP:
            return None

    n_fixed_segments = sum(per_table_options[i][0].n_segments for i in fixed_indices)

    variable_option_lists = [per_table_options[i] for i in variable_indices]

    best_score: tuple[int, int, int] | None = None
    best_combo: tuple[SplitOption, ...] | None = None
    for combo in itertools.product(*variable_option_lists):
        total_segments = n_fixed_segments + sum(opt.n_segments for opt in combo)
        score = (
            abs(total_segments - n_expected),
            -sum(opt.n_repeating_rows for opt in combo),
            -sum(opt.period for opt in combo),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_combo = combo

    assert best_score is not None
    assert best_combo is not None

    baseline_score = (abs(len(actual) - n_expected), 0, 0)
    if best_score >= baseline_score:
        return None

    chosen: list[SplitOption] = [_NO_SPLIT] * len(actual)
    for i in fixed_indices:
        chosen[i] = per_table_options[i][0]
    for var_pos, table_idx in enumerate(variable_indices):
        chosen[table_idx] = best_combo[var_pos]
    return chosen


def split_ambiguous_merged_pred(
    expected: list[ExtractedTable],
    actual: list[ExtractedTable],
) -> tuple[list[ExtractedTable], bool]:
    """Split merged pred tables on a page when GT has more tables than pred.

    Trigger: ``len(expected) > len(actual)``. Delegates to
    ``select_joint_split`` which jointly chooses a ``SplitOption`` per pred
    table on the page, optimizing total table count toward ``len(expected)``
    under a lexicographic objective and only applying the result when it
    strictly beats the all-no-split baseline. Capped by ``_SAFETY_CAP`` on
    the variable-tables Cartesian product.

    Untouched ``ExtractedTable``s preserve their original ``raw_html``;
    sub-tables emitted from a split have ``raw_html=""`` since they have no
    meaningful HTML to attribute back to the source.

    When a split is applied, normalize trailing blank padding on both sides.
    The peer ``expected`` list is updated in place with replacement tables;
    original TableData objects and source HTML remain unchanged. No-split
    calls leave both inputs untouched.

    Returns ``(possibly_rewritten_actual, did_split)``.
    """
    from parse_bench.evaluation.metrics.parse.table_record_match_metric import normalize_table

    if len(expected) <= len(actual):
        return actual, False

    chosen = select_joint_split(actual, len(expected))
    if chosen is None:
        return actual, False

    new_actual: list[ExtractedTable] = []
    for original, opt in zip(actual, chosen, strict=True):
        if opt.sub_tables is None:
            new_actual.append(original)
        else:
            # Keep the candidate geometry: normalization may remove spacer
            # columns, so its offsets cannot slice the original grid directly.
            source = normalize_table(original.table_data, preserve_text=True)
            new_actual.extend(
                ExtractedTable(
                    raw_html="",
                    table_data=build_sub_table(source, i * opt.period, (i + 1) * opt.period),
                )
                for i in range(opt.n_segments)
            )
    expected[:] = [replace(t, table_data=_trim_trailing_padding(t.table_data)) for t in expected]
    new_actual = [replace(t, table_data=_trim_trailing_padding(t.table_data)) for t in new_actual]
    return new_actual, True
