"""Scoring pins for `form_kv_scoring`: perfect iff 1.0, one defect moves one axis, coverage
semantics, and the counter schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parse_bench.evaluation.metrics.parse.form_kv_scoring import (
    AXES,
    COUNTER_KEYS,
    KvPage,
    pages_from_tree,
    parse_kv_markdown,
    render_guide,
    score_counts,
    score_pages,
)

FX = Path(__file__).parent / "fixtures" / "form_kv"

BASE: list[dict[str, Any]] = [
    {"type": "field", "field": "text", "id": "1", "label": "Wages, tips, other compensation", "value": "29,513"},
    {"type": "field", "field": "text", "id": "2", "label": "Federal income tax withheld", "value": "3,241.82"},
    {"type": "field", "field": "text", "id": "7", "label": "Social security tips", "isEmpty": True},
    {"type": "field", "field": "checkbox", "id": "D", "label": "Check if this is a PTP", "value": False},
    {"type": "field", "field": "signature", "label": "Your signature", "value": True},
    {
        "type": "section",
        "id": "15",
        "label": "State",
        "items": [
            {"type": "field", "field": "text", "value": "SC"},
            {"type": "field", "field": "text", "label": "Employer's state ID number", "value": "00-0000056"},
        ],
    },
    {
        "type": "field",
        "field": "multi_select",
        "id": "13",
        "valueItems": [
            {"type": "field", "field": "checkbox", "label": "Statutory employee", "value": True},
            {"type": "field", "field": "checkbox", "label": "Retirement plan", "value": False},
            {"type": "field", "field": "checkbox", "label": "Third-party sick pay", "value": False},
        ],
    },
]
GT_MD = render_guide(BASE)
SCORED = [f"{a}_f1" for a in AXES] + ["structure", "composite", "checkbox_signature_f1"]


def gt_pages() -> list[KvPage]:
    return pages_from_tree([{"page_number": 1, "items": BASE}])


def score_md(pred_md: str, gt: list[KvPage] | None = None) -> dict[str, Any]:
    return score_pages(gt or gt_pages(), [parse_kv_markdown(pred_md, 1)])


def edited(old: str, new: str) -> str:
    assert old in GT_MD, old
    return GT_MD.replace(old, new)


def ones(r: dict[str, Any], *axes: str) -> None:
    for a in axes:
        assert r[a] == 1.0, (a, r[a])


def below(r: dict[str, Any], *axes: str) -> None:
    for a in axes:
        assert r[a] is not None and r[a] < 1.0, (a, r[a])


def test_perfect_prediction_is_one_on_every_axis() -> None:
    r = score_md(GT_MD)
    ones(r, *SCORED, "spot_composite", "spot_declared_precision", "grouping_f1", "order_kendall", "select_shape")
    for short in ("w2", "k1", "f1040"):
        tree = json.loads((FX / f"{short}.tree.json").read_text())
        r = score_pages(pages_from_tree(tree["pages"]), pages_from_tree(tree["pages"]))
        assert all(r[k] in (None, 1.0) for k in SCORED), (short, {k: r[k] for k in SCORED if r[k] not in (None, 1.0)})
        assert r["composite"] == 1.0 and r["core_kv_f1"] == 1.0


def test_wrong_value_moves_value_and_core_only() -> None:
    r = score_md(edited(": 29,513", ": 29,514"))
    below(r, "value_f1", "value_strict_f1", "core_kv_f1", "composite")
    ones(r, "label_f1", "label_strict_f1", "id_f1", "id_lenient_f1", "checkbox_f1", "signature_f1", "structure")


def test_value_formatting_moves_the_strict_dual_only() -> None:
    r = score_md(edited(": 29,513", ": 29513"))
    ones(r, "value_f1", "core_kv_f1", "composite")
    below(r, "value_strict_f1")


def test_label_typo_is_graded_and_case_moves_the_strict_dual_only() -> None:
    typo = score_md(edited("**Federal income tax withheld**", "**Federal income tax witheld**"))
    below(typo, "label_f1", "label_strict_f1", "composite")
    assert typo["label_f1"] > 0.9
    ones(typo, "core_kv_f1", "value_f1", "id_f1")
    case = score_md(edited("**Wages, tips, other compensation**", "**wages, tips, other compensation**"))
    ones(case, "label_f1", "core_kv_f1")
    below(case, "label_strict_f1")


def test_baked_id_is_a_formatting_defect() -> None:
    r = score_md(edited("- \\[1\\] **Wages, tips, other compensation**", "- **1 Wages, tips, other compensation**"))
    below(r, "id_f1", "label_f1", "composite")
    ones(r, "id_lenient_f1", "value_f1", "core_kv_f1", "checkbox_f1")
    assert r["counts"]["exhaustive"]["baked_ids"] == 1


def test_flipped_box_moves_checkbox_only() -> None:
    r = score_md(edited("  - [ ] Retirement plan", "  - [x] Retirement plan"))
    below(r, "checkbox_f1", "checkbox_signature_f1", "core_kv_f1", "composite")
    ones(r, "value_f1", "label_f1", "id_f1", "signature_f1", "structure")
    assert r["checkbox_recall"] == 3 / 4  # units: 3 options + the standalone box D; one option disagrees


def test_dropped_signature_is_a_signature_miss_and_an_invented_blank() -> None:
    r = score_md(edited("**Your signature**: [signature]", "**Your signature**: [blank]"))
    assert r["signature_recall"] == 0.0 and r["signature_f1"] == 0.0
    ones(r, "checkbox_f1", "label_f1", "value_recall")
    below(r, "value_precision", "checkbox_signature_f1")


def test_missing_record_moves_recall_and_invented_record_moves_precision() -> None:
    missing = score_md(edited("- \\[2\\] **Federal income tax withheld**: 3,241.82\n", ""))
    below(missing, "value_recall", "label_recall", "id_recall", "core_kv_recall")
    ones(missing, "value_precision", "label_precision", "id_precision", "checkbox_f1", "structure")
    invented = score_md(GT_MD + "\n- **Bonus**: 5")
    below(invented, "value_precision", "label_precision", "core_kv_precision")
    ones(invented, "value_recall", "label_recall", "id_f1", "checkbox_f1")


def test_moved_record_moves_order_and_flattened_section_moves_grouping() -> None:
    line = "- \\[2\\] **Federal income tax withheld**: 3,241.82\n"
    moved = score_md(edited(line, "") + "\n" + line.rstrip("\n"))
    below(moved, "order_kendall", "structure", "composite")
    ones(moved, "grouping_f1", "select_shape", "value_f1", "label_f1", "id_f1", "core_kv_f1")
    flat = score_md(edited("- \\[15\\] **State**\n  - SC\n  - **Employer's", "- SC\n- **Employer's"))
    below(flat, "grouping_f1", "structure")
    ones(flat, "order_kendall", "value_f1", "label_f1", "core_kv_f1")


def test_option_group_emitted_as_loose_boxes_is_rescued() -> None:
    loose = edited(
        "- \\[13\\]\n  - [x] Statutory employee\n  - [ ] Retirement plan\n  - [ ] Third-party sick pay",
        "- [x] Statutory employee\n- [ ] Retirement plan\n- [ ] Third-party sick pay",
    )
    r = score_md(loose)
    ones(r, "checkbox_f1", "core_kv_f1", "value_f1")
    assert r["select_shape"] == 0.0 and r["counts"]["exhaustive"]["rescued_selects"] == 1
    below(r, "id_recall", "structure")


def test_spot_page_scores_recall_only() -> None:
    spot_gt = parse_kv_markdown(
        "<!-- coverage: spot -->\n- \\[1\\] **Wages, tips, other compensation**: 29,513\n"
        "- \\[7\\] **Social security tips**: [blank]\n- \\[D\\] **Check if this is a PTP**: [ ]",
        1,
    )
    r = score_pages([spot_gt], [parse_kv_markdown(GT_MD, 1)])
    assert all(r[k] is None for k in SCORED)
    assert r["exhaustive_pages"] == 0 and r["spot_pages"] == 1
    ones(r, "spot_value_recall", "spot_label_recall", "spot_id_recall", "spot_checkbox_recall")
    ones(r, "spot_declared_precision", "spot_composite")
    filled = score_pages(
        [spot_gt], [parse_kv_markdown(edited("**Social security tips**: [blank]", "**Social security tips**: 5"), 1)]
    )
    below(filled, "spot_declared_precision", "spot_value_recall", "spot_composite")
    ones(filled, "spot_label_recall", "spot_id_recall")


def test_mixed_doc_reads_exhaustive_axes_from_exhaustive_pages_only() -> None:
    gt = gt_pages() + [
        parse_kv_markdown("<!-- coverage: spot -->\n- \\[1\\] **Wages, tips, other compensation**: 29,513", 2)
    ]
    pred = [parse_kv_markdown(GT_MD, 1), parse_kv_markdown(GT_MD + "\n- **Extra**: 1", 2)]
    r = score_pages(gt, pred)
    assert (r["exhaustive_pages"], r["spot_pages"]) == (1, 1)
    ones(r, "composite", "value_precision", "spot_value_recall")


def test_pred_page_without_gt_page_is_all_inventions() -> None:
    r = score_pages(gt_pages(), [parse_kv_markdown(GT_MD, 1), parse_kv_markdown("- **Ghost**: 1", 2)])
    below(r, "value_precision", "label_precision")
    ones(r, "value_recall")


def test_counter_schema_is_pinned_and_sums() -> None:
    assert COUNTER_KEYS[:6] == (
        "gt_records",
        "pred_records",
        "matched_gt",
        "matched_pred",
        "rescued_selects",
        "baked_ids",
    )
    assert all(f"{a}_{p}" in COUNTER_KEYS for a in AXES for p in ("rec_num", "rec_den", "prec_num", "prec_den"))
    assert COUNTER_KEYS[-9:] == (
        "grp_tp",
        "grp_gt_pairs",
        "grp_pred_pairs",
        "order_num",
        "order_den",
        "select_n",
        "select_shaped",
        "declared_num",
        "declared_den",
    )
    two = gt_pages() + [parse_kv_markdown(edited(": 29,513", ": 1"), 2)]
    pred = [parse_kv_markdown(GT_MD, 1), parse_kv_markdown(GT_MD, 2)]
    doc = score_pages(two, pred)["counts"]["exhaustive"]
    p1 = score_counts(list(two[0].records), list(pred[0].records))
    p2 = score_counts(list(two[1].records), list(pred[1].records))
    assert doc == {k: p1[k] + p2[k] for k in COUNTER_KEYS}
    for k in COUNTER_KEYS:
        assert doc[k] >= 0
    for a in AXES:
        assert doc[f"{a}_rec_num"] <= doc[f"{a}_rec_den"] and doc[f"{a}_prec_num"] <= doc[f"{a}_prec_den"]


def test_wrong_baked_id_counts_against_lenient_precision() -> None:
    gt = parse_kv_markdown("- \\[1\\] **Wages**: 10\n- \\[2\\] **Tax**: 20", 1)
    pred = parse_kv_markdown("- **7 Wages**: 10\n- **2 Tax**: 20", 1)
    r = score_pages([gt], [pred])
    assert (r["id_lenient_recall"], r["id_lenient_precision"], r["id_lenient_f1"]) == (0.5, 0.5, 0.5)
    assert r["id_f1"] == 0.0  # nothing in the id slot


def test_baked_looking_label_is_an_id_claim_only_when_gt_has_an_id() -> None:
    gt = parse_kv_markdown("- **14 State**: TX\n- **Y / N**: [x]", 1)
    r = score_pages([gt], [parse_kv_markdown("- **14 State**: TX\n- **Y / N**: [x]", 1)])
    assert r["id_lenient_precision"] is None and r["id_lenient_f1"] is None  # no ids anywhere: no axis
    mixed_gt = parse_kv_markdown("- \\[1\\] **Wages**: 10\n- **14 State**: TX", 1)
    r = score_pages([mixed_gt], [parse_kv_markdown("- \\[1\\] **Wages**: 10\n- **14 State**: TX", 1)])
    assert r["id_lenient_precision"] == 1.0 and r["id_lenient_f1"] == 1.0


def test_strict_label_folds_encoding_variants_like_the_strict_value_dual() -> None:
    r = score_md(edited("**Employer's state ID number**", "**Employer\u2019s state ID number**"))
    ones(r, "label_f1", "label_strict_f1")
