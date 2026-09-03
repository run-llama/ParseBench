"""Unit tests for the table-merging (junction merge) metric.

Each case is a tiny hand-built GT (atoms + logical_tables) plus a fake ParseOutput
standing in for a parser with a specific merge behavior. Tables are single-column
with distinct header/cell text so GriTS containment-recall assigns each atom to the
intended predicted table.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from parse_bench.evaluation.metrics.parse.table_merging import compute_table_merging_metrics


def _t(header: str, *cells: str) -> str:
    """A single-column table: one header row + one row per cell."""
    rows = "".join(f"<tr><td>{c}</td></tr>" for c in cells)
    return f"<table><tr><th>{header}</th></tr>{rows}</table>"


def _po(pages: dict[int, str]) -> Any:
    """Fake ParseOutput; each page's markdown is a concatenation of <table> HTML."""
    return SimpleNamespace(pages=[SimpleNamespace(page_index=p - 1, markdown=md) for p, md in sorted(pages.items())])


def _atom(key: str, page: int, idx: int, html: str) -> dict[str, Any]:
    return {"key": key, "page": page, "idx": idx, "html": html}


def _by_name(metrics: list) -> dict[str, float]:
    return {m.metric_name: m.value for m in metrics}


# Distinct fragments used across cases.
A = _t("alpha", "a1")
B = _t("beta", "b1")
C = _t("gamma", "c1")
D = _t("delta", "d1")
B2 = _t("beta", "b2")  # a continuation of B (same header)
BC_MERGED = _t("beta", "b1", "b2")


# --- 1. perfect merge ------------------------------------------------------


def test_perfect_merge():
    atoms = [_atom("1:0", 1, 0, A), _atom("1:1", 1, 1, B), _atom("2:0", 2, 0, B2)]
    gt = {
        "atoms": atoms,
        "logical_tables": [
            {"id": "L1", "fragments": ["1:0"], "html": A},
            {"id": "L2", "fragments": ["1:1", "2:0"], "html": BC_MERGED},
        ],
    }
    m = _by_name(compute_table_merging_metrics(_po({1: A + BC_MERGED, 2: ""}), gt))
    assert m["table_merge_accuracy"] == 1.0
    assert m["table_merge_recall"] == 1.0
    assert m["table_merge_specificity"] == 1.0
    assert m["table_merge_tp"] == 1.0 and m["table_merge_tn"] == 1.0
    assert m["table_merge_fp"] == 0.0 and m["table_merge_fn"] == 0.0


# --- 2. cross-page continuation split at the page break --------------------


def test_cross_page_split_is_fn_on_cross_slice_only():
    atoms = [_atom("1:0", 1, 0, A), _atom("1:1", 1, 1, B), _atom("2:0", 2, 0, B2)]
    gt = {
        "atoms": atoms,
        "logical_tables": [
            {"id": "L1", "fragments": ["1:0"], "html": A},
            {"id": "L2", "fragments": ["1:1", "2:0"], "html": BC_MERGED},
        ],
    }
    # Parser splits at the page break: B on page 1, its continuation on page 2.
    m = _by_name(compute_table_merging_metrics(_po({1: A + B, 2: B2}), gt))
    assert m["table_merge_fn"] == 1.0  # the cross-page merge was missed
    assert m["table_merge_accuracy_cross_page"] == 0.0
    assert m["table_merge_accuracy_within_page"] == 1.0  # within-page junction unaffected


# --- 3. within-page section-break split ------------------------------------


def test_within_page_split_is_fn_on_within_slice():
    atoms = [_atom("1:0", 1, 0, B), _atom("1:1", 1, 1, B2)]
    gt = {"atoms": atoms, "logical_tables": [{"id": "L2", "fragments": ["1:0", "1:1"], "html": BC_MERGED}]}
    m = _by_name(compute_table_merging_metrics(_po({1: B + B2}), gt))  # kept separate
    assert m["table_merge_fn"] == 1.0
    assert m["table_merge_recall"] == 0.0
    assert m["table_merge_accuracy_within_page"] == 0.0
    assert "table_merge_accuracy_cross_page" not in m  # no cross-page junctions → slice omitted


# --- 4. false merge (two distinct tables emitted as one) -------------------


def test_false_merge_is_fp():
    atoms = [_atom("1:0", 1, 0, A), _atom("1:1", 1, 1, D)]
    gt = {
        "atoms": atoms,
        "logical_tables": [
            {"id": "L1", "fragments": ["1:0"], "html": A},
            {"id": "L3", "fragments": ["1:1"], "html": D},
        ],
    }
    merged_ad = _t("alpha", "a1").replace("</table>", "") + "<tr><th>delta</th></tr><tr><td>d1</td></tr></table>"
    m = _by_name(compute_table_merging_metrics(_po({1: merged_ad}), gt))
    assert m["table_merge_fp"] == 1.0
    assert m["table_merge_precision"] < 1.0
    assert m["table_merge_specificity"] < 1.0


# --- 5. correct separate (distinct tables kept apart) ----------------------


def test_correct_separate_is_tn_and_counts_toward_accuracy():
    atoms = [_atom("1:0", 1, 0, A), _atom("1:1", 1, 1, D)]
    gt = {
        "atoms": atoms,
        "logical_tables": [
            {"id": "L1", "fragments": ["1:0"], "html": A},
            {"id": "L3", "fragments": ["1:1"], "html": D},
        ],
    }
    m = _by_name(compute_table_merging_metrics(_po({1: A + D}), gt))
    assert m["table_merge_tn"] == 1.0
    assert m["table_merge_accuracy"] == 1.0  # TN counts toward accuracy
    assert m["table_merge_specificity"] == 1.0


# --- 6. multi-fragment logical table (3 atoms => 2 junctions), partial ------


def test_partial_merge_accuracy_strictly_between_0_and_1():
    atoms = [_atom("1:0", 1, 0, B), _atom("1:1", 1, 1, B2), _atom("2:0", 2, 0, _t("beta", "b3"))]
    merged_b12 = _t("beta", "b1", "b2")
    gt = {
        "atoms": atoms,
        "logical_tables": [{"id": "L2", "fragments": ["1:0", "1:1", "2:0"], "html": _t("beta", "b1", "b2", "b3")}],
    }
    # Parser merges the first two (within page) but splits the cross-page third.
    m = _by_name(compute_table_merging_metrics(_po({1: merged_b12, 2: _t("beta", "b3")}), gt))
    assert m["table_merge_tp"] == 1.0  # within-page merge caught
    assert m["table_merge_fn"] == 1.0  # cross-page continuation missed
    assert 0.0 < m["table_merge_accuracy"] < 1.0
    assert m["table_merge_accuracy"] == pytest.approx(0.5)


# --- 7. missing predicted table for one atom -> FN -------------------------


def test_missing_atom_table_is_fn():
    # B and its continuation belong to one logical table, but the parser dropped
    # the continuation entirely (its content appears in no predicted table).
    cont = _t("zulu", "q9")  # shares no characters with B, so it matches nothing when dropped
    atoms = [_atom("1:0", 1, 0, B), _atom("2:0", 2, 0, cont)]
    gt = {"atoms": atoms, "logical_tables": [{"id": "L2", "fragments": ["1:0", "2:0"], "html": BC_MERGED}]}
    m = _by_name(compute_table_merging_metrics(_po({1: B, 2: ""}), gt))  # only B emitted
    assert m["table_merge_fn"] == 1.0
    assert m["table_merge_recall"] == 0.0


# --- 8. <2 atoms is empty; zero-merge doc is still emitted (all TN) ---------


def test_junction_free_doc_emits_nothing():
    gt = {"atoms": [_atom("1:0", 1, 0, A)], "logical_tables": [{"id": "L1", "fragments": ["1:0"], "html": A}]}
    assert compute_table_merging_metrics(_po({1: A}), gt) == []


def test_zero_merge_doc_is_emitted_all_tn():
    atoms = [_atom("1:0", 1, 0, A), _atom("1:1", 1, 1, D), _atom("1:2", 1, 2, C)]
    gt = {
        "atoms": atoms,
        "logical_tables": [
            {"id": "L1", "fragments": ["1:0"], "html": A},
            {"id": "L3", "fragments": ["1:1"], "html": D},
            {"id": "L4", "fragments": ["1:2"], "html": C},
        ],
    }
    metrics = compute_table_merging_metrics(_po({1: A + D + C}), gt)
    assert metrics  # non-empty: a doc with junctions always emits, even all-separate
    m = _by_name(metrics)
    assert m["table_merge_tn"] == 2.0
    assert m["table_merge_accuracy"] == 1.0


# --- 9. page-agnostic headline = micro over ALL junctions ------------------


def test_headline_accuracy_is_page_agnostic_micro():
    # 2 within-page junctions (both correct TN) + 1 cross-page junction (wrong FP).
    # Pooled accuracy = (0+2)/3 = 0.667, which is NOT the mean of the slice
    # accuracies (1.0 and 0.0 → 0.5): the headline pools junctions page-agnostically.
    atoms = [_atom("1:0", 1, 0, A), _atom("1:1", 1, 1, B), _atom("1:2", 1, 2, C), _atom("2:0", 2, 0, D)]
    gt = {
        "atoms": atoms,
        "logical_tables": [
            {"id": "L1", "fragments": ["1:0"], "html": A},
            {"id": "L2", "fragments": ["1:1"], "html": B},
            {"id": "L3", "fragments": ["1:2"], "html": C},
            {"id": "L4", "fragments": ["2:0"], "html": D},
        ],
    }
    merged_cd = _t("gamma", "c1").replace("</table>", "") + "<tr><th>delta</th></tr><tr><td>d1</td></tr></table>"
    m = _by_name(compute_table_merging_metrics(_po({1: A + B + merged_cd, 2: ""}), gt))
    assert m["table_merge_accuracy"] == pytest.approx(2 / 3)
    assert m["table_merge_accuracy_within_page"] == 1.0
    assert m["table_merge_accuracy_cross_page"] == 0.0
    assert m["table_merge_fp"] == 1.0 and m["table_merge_tn"] == 2.0
