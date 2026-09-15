"""Equivalence guards for the GriTS performance optimizations.

Both optimizations are perf-only and MUST NOT change any metric value:

- The ``factored_2dmss`` kernel has three modes (``BENCH_GRITS_KERNEL``):
  ``slow`` (original dict path), ``memo`` (memoized dict build), and ``array``
  (vectorized reward tensor + list DP, the default). All three must return
  bit-identical scores AND alignment maps.
- ``BENCH_GRITS_PAIR_WORKERS`` distributes the per-table-pair scoring across
  processes, page by page — ``GriTSMetric.compute`` must return the same
  ``grits_con`` and the same ``pairing`` as the sequential path.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import parse_bench.evaluation.metrics.parse.grits_metric as G  # noqa: E402
from parse_bench.evaluation.metrics.parse.grits_metric import (  # noqa: E402
    grits_con_from_table_data,
)
from parse_bench.evaluation.metrics.parse.table_extraction import (  # noqa: E402
    extract_table_pairs,
)

# Four distinct tables: repeated cell values (so the memo cache is exercised),
# different dimensions, and mismatched content (so grits_con != 1.0 and the
# alignment tie-breaking matters).
T_A = "<table><tr><td>A</td><td>A</td></tr><tr><td>B</td><td>A</td></tr></table>"
T_A_PERTURBED = "<table><tr><td>A</td><td>A</td></tr><tr><td>B</td><td>X</td></tr></table>"
T_WIDE = "<table><tr><td>h</td><td>h</td><td>h</td></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>"
T_TALL = "<table><tr><td>k</td><td>v</td></tr><tr><td>k</td><td>w</td></tr><tr><td>k</td><td>x</td></tr></table>"


def _table_datas(html: str):
    exp, act, _ = extract_table_pairs(html, html)
    return exp[0].table_data, act[0].table_data


def test_kernel_modes_are_bit_identical(monkeypatch):
    """slow / memo / array kernels return identical scores AND alignment maps."""
    pairs = [
        (T_A, T_A),  # identical
        (T_A, T_A_PERTURBED),  # one cell differs (con < 1.0)
        (T_A, T_WIDE),  # dimension mismatch
        (T_TALL, T_WIDE),  # both dims differ
    ]
    keys = [
        "grits_con",
        "grits_precision_con",
        "grits_recall_con",
        "grits_con_upper_bound",
        "_con_row_alignment",
        "_con_col_alignment",
    ]
    for gt_html, pred_html in pairs:
        gt = _table_datas(gt_html)[0]
        pred = _table_datas(pred_html)[0]

        results = {}
        for mode in ("slow", "memo", "array"):
            monkeypatch.setenv("BENCH_GRITS_KERNEL", mode)
            results[mode] = grits_con_from_table_data(gt, pred)
            assert results[mode] is not None

        for mode in ("memo", "array"):
            for k in keys:
                assert results["slow"][k] == results[mode][k], (
                    f"{mode}.{k} diverged from slow on {gt_html!r} vs {pred_html!r}: "
                    f"{results['slow'][k]} != {results[mode][k]}"
                )


def test_legacy_fast_kernel_flag_still_disables(monkeypatch):
    """BENCH_GRITS_FAST_KERNEL=0 forces the slow path (back-compat)."""
    gt, pred = _table_datas(T_A)[0], _table_datas(T_A_PERTURBED)[0]
    monkeypatch.delenv("BENCH_GRITS_KERNEL", raising=False)
    monkeypatch.setenv("BENCH_GRITS_FAST_KERNEL", "0")
    assert G._kernel_mode() == "slow"
    slow = grits_con_from_table_data(gt, pred)
    monkeypatch.setenv("BENCH_GRITS_FAST_KERNEL", "1")
    assert G._kernel_mode() == "array"
    arr = grits_con_from_table_data(gt, pred)
    assert slow["grits_con"] == arr["grits_con"]


def _paged_tables(n_pages: int, per_page: int):
    """``per_page`` GT/pred tables on each of ``n_pages`` pages, mismatched so
    scores are non-trivial and the assignment is not the identity."""
    variants = [T_A, T_A_PERTURBED, T_WIDE, T_TALL]
    expected, actual, exp_pages, act_pages = [], [], [], []
    for page in range(1, n_pages + 1):
        for k in range(per_page):
            gt_html = variants[(page + k) % len(variants)]
            pred_html = variants[(page + k + 1) % len(variants)]
            exp, act, _ = extract_table_pairs(gt_html, pred_html)
            expected.append(exp[0])
            actual.append(act[0])
            exp_pages.append(page)
            act_pages.append(page)
    return expected, actual, exp_pages, act_pages


def test_page_parallel_pair_workers_match_sequential(monkeypatch) -> None:
    # 3 pages x 3 tables = 27 same-page pairs, above the parallel threshold.
    expected, actual, exp_pages, act_pages = _paged_tables(3, 3)
    assert len(exp_pages) * 3 >= G._PAIR_PARALLEL_MIN_PAIRS

    sequential = G.GriTSMetric(pair_workers=1).compute(
        expected, actual, expected_pages=exp_pages, actual_pages=act_pages
    )
    parallel = G.GriTSMetric(pair_workers=2).compute(expected, actual, expected_pages=exp_pages, actual_pages=act_pages)
    monkeypatch.setenv("BENCH_GRITS_PAIR_WORKERS", "2")
    via_env = G.GriTSMetric().compute(expected, actual, expected_pages=exp_pages, actual_pages=act_pages)

    for candidate in (parallel, via_env):
        assert [m.value for m in candidate] == [m.value for m in sequential]
        assert candidate[0].metadata["pairing"] == sequential[0].metadata["pairing"]
        assert candidate[0].metadata["per_table_details"] == sequential[0].metadata["per_table_details"]


def test_pair_workers_env_fallback_is_at_least_one(monkeypatch) -> None:
    monkeypatch.setenv("BENCH_GRITS_PAIR_WORKERS", "0")
    assert G._pair_workers() == 1
    monkeypatch.setenv("BENCH_GRITS_PAIR_WORKERS", "junk")
    assert G._pair_workers() == 1
    monkeypatch.delenv("BENCH_GRITS_PAIR_WORKERS")
    assert G._pair_workers() == 1
