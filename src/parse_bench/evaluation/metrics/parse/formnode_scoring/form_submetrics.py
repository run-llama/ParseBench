"""Submetric counters and aggregation.

Counters are raw and summable: aggregate pages/docs/groups by summing counter dicts,
then `finalize()` — so every aggregate is exposure-weighted (a 100-field doc moves the
group number 10x more than a 10-field doc).

Every content axis is an F1 with BOTH denominators, so absence can never fall out:

- **axis recall** = credited weight / total GT-side weight of the fields belonging to
  the axis — a missed GT field contributes denominator and no numerator (coverage folds
  into every axis).
- **axis precision** = credited weight / total pred-side weight — an invented pred field
  contributes denominator and no numerator.
- A matched-but-wrong pair sits in both denominators and neither numerator.
- **Membership**: GT kind decides recall-side membership, predicted kind decides
  precision-side membership — a GT checkbox predicted as text hurts checkbox-recall,
  value-precision, and the kind axis. The kind and label/id axes span all fields
  (label/id membership = that side carries a label/printed id).
- **Exposure weights carry to both sides**: empty fields weigh 1/2 and near-duplicate
  empties weigh 1/4 in every axis denominator (and symmetrically in the numerator when
  credited), matched or not. Among the non-select empty fields sharing a
  (label, id, kind) key on a page, the first in flat order keeps the half weight and
  every later one a quarter — a repeated anonymous empty grid is hard to emit
  exhaustively and slots past the first carry little information. Inventing a value in
  a blank box is priced at the pred side's FULL weight (hallucination punished harder
  than omission).
- Selects contribute **per-option units** (weight 1.0 each) to the checkbox axis on
  their side, so an invented option inside a matched select is visible; the select
  field itself carries the kind/label/id axes.
- Labels are the one graded axis (`text_sim` numerator); every other axis credits 0/1.
- **Slot-literal**: the label axis reads the RAW emitted label and the id axis credits
  only two-sided id-slot agreement. Baking the printed id into the label is an error —
  graded label ding + dropped id (the strip still serves matching, and the
  `id_exact_rate`/`baked_ids` diagnostics name the failure mode). Symmetrically,
  label text emitted in the id slot is a dropped label + hallucinated id.

Not F1-able and unchanged: order (Kendall tau) and the structure mean
(grouping/order/select-shape). Known residual: grouping is co-membership only, so
section ANCHOR renames are headline-invisible — `skeleton_teds` is their diagnostic.

Matched-pair conditional accuracies (`value_acc`, `checkbox_acc`, `kind_acc`,
`label_sim_mean`, ...) and the standalone field-coverage F1 remain as diagnostics; the
composite reads the per-axis F1s. Strictness duals (`kind_acc_strict`,
`select_shape_strict`) are emitted beside the agnostic primaries.
"""

from typing import Any

from parse_bench.evaluation.metrics.parse.formnode_scoring.bbox_geometry import best_pair_iou
from parse_bench.evaluation.metrics.parse.formnode_scoring.field_flatten import (
    SELECT_KINDS,
    Field,
    flatten_pages,
)
from parse_bench.evaluation.metrics.parse.formnode_scoring.field_matching import (
    _id_component,
    align,
    kind_credit,
    match_options,
)
from parse_bench.evaluation.metrics.parse.formnode_scoring.form_skeleton_teds import page_skeleton_teds
from parse_bench.evaluation.metrics.parse.formnode_scoring.text_normalization import (
    fold,
    is_empty_value,
    norm_value_eq,
    numeric_or_none,
    text_sim,
)

AXES = ("value", "label", "checkbox", "kind", "id")

# Grounded family (field-grounding IoU, module constant like SIM_THRESHOLD). NOT an
# AXES entry: AXES membership is kind-based, grounded membership is bbox-presence.
IOU_THRESHOLD = 0.5

COUNTER_KEYS = (
    "gt_fields",
    "pred_fields",
    "matched_gt",
    "matched_pred",
    "miss_w",
    "fp_w",
    "missed_valued",
    "missed_empty",
    "spurious_empty",
    "duplicate_empty_gt",
    "duplicate_empty_pred",
    "hallucinated_value_in_blank",
    "hallucinated_labels",
    "baked_ids",
    "rescued_selects",
    "label_n",
    "label_gt_on_match",
    "label_sim_sum",
    "label_exact",
    "id_n",
    "id_exact",
    "id_mismatch_on_match",
    "value_n",
    "value_ok",
    "value_n_numeric",
    "value_ok_numeric",
    "value_n_text",
    "value_ok_text",
    "cb_n",
    "cb_ok",
    "kind_n",
    "kind_credit",
    "kind_strict_credit",
    "grp_tp",
    "grp_gt_pairs",
    "grp_pred_pairs",
    "order_num",
    "order_den",
    "select_n",
    "select_shaped",
    "select_shaped_strict",
    "select_kind_swap",
    # per-axis F1 counters: recall over GT-side weight, precision over pred-side weight
    *(f"{axis}_{part}" for axis in AXES for part in ("rec_num", "rec_den", "prec_num", "prec_den")),
    # grounded family (value-gated bbox-IoU): recall den = box-bearing
    # GT fields/options; precision den = pred box claims aligned to a box-bearing GT
    # counterpart; nums = content-correct AND any-pair IoU >= IOU_THRESHOLD. iou_* is
    # the ungated localization diagnostic over two-sided pairs.
    "grounded_rec_num",
    "grounded_rec_den",
    "grounded_prec_num",
    "grounded_prec_den",
    "iou_sum",
    "iou_n",
    # filled by score_pages (page-level diagnostic, node-weighted) — declared here so
    # every counter dict carries the same keys and aggregates by plain summing
    "skeleton_teds_num",
    "skeleton_teds_den",
)


def _kendall_order(pairs: list[tuple[int, int]]) -> float | None:
    n = len(pairs)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = pairs[i][0] - pairs[j][0]
            b = pairs[i][1] - pairs[j][1]
            if a * b > 0:
                conc += 1
            elif a * b < 0:
                disc += 1
    total = n * (n - 1) / 2
    return (1 + (conc - disc) / total) / 2


def _empty_group_rank(fields: list[Field]) -> dict[int, int]:
    """Rank of each non-select EMPTY field within its duplicate group, keyed by `id()`.

    Groups are `(folded label, folded printed id, kind)`; members are ranked in flat
    order, matched or not, so rank 0 is the group's first slot on the page. Valued fields
    and selects are absent from the mapping (they never discount).
    """
    ranks: dict[int, int] = {}
    seen: dict[tuple[str, str, str], int] = {}
    for f in sorted(fields, key=lambda x: x.flat_idx):
        if f.kind in SELECT_KINDS or not is_empty_value(f.value):
            continue
        key = (fold(f.label or ""), fold(f.id or ""), f.kind)
        rank = seen.get(key, 0)
        ranks[id(f)] = rank
        seen[key] = rank + 1
    return ranks


def _exposure(f: Field, empty_rank: dict[int, int]) -> float:
    """A field's weight in every axis denominator: valued/select 1.0, empty 0.5,
    duplicate empty 0.25 — matched or not."""
    rank = empty_rank.get(id(f))
    if rank is None:
        return 1.0
    return 0.5 if rank == 0 else 0.25


def _is_checkbox_axis(kind: str, value: Any) -> bool:
    """Standalone binary-state fields (selects are per-option and handled apart)."""
    return kind == "checkbox" or (kind == "signature" and isinstance(value, bool))


def score_counts(gt: list[Field], pred: list[Field]) -> dict[str, float]:
    """Raw counters for one page pair (aggregate by summing, then finalize())."""
    align(gt, pred)
    c = dict.fromkeys(COUNTER_KEYS, 0.0)
    c["gt_fields"], c["pred_fields"] = float(len(gt)), float(len(pred))

    rescued_gt = [g for g in gt if g.rescued]
    matched: list[tuple[Field, Field]] = [(g, m) for g in gt if (m := g.match) is not None and not g.rescued]
    c["matched_gt"] = float(len(matched) + len(rescued_gt))
    c["matched_pred"] = float(sum(1 for p in pred if p.match is not None))
    c["rescued_selects"] = float(len(rescued_gt))

    gt_empty_rank = _empty_group_rank(gt)
    pred_empty_rank = _empty_group_rank(pred)

    def axis_membership(f: Field, side: str, w: float) -> None:
        """Denominator contributions of one field on its side (`rec` or `prec`)."""
        c[f"kind_{side}_den"] += w
        if f.id:
            c[f"id_{side}_den"] += w
        if f.label:
            c[f"label_{side}_den"] += w
        if f.kind in SELECT_KINDS:
            c[f"checkbox_{side}_den"] += len(f.options)
        elif _is_checkbox_axis(f.kind, f.value):
            c[f"checkbox_{side}_den"] += w
        else:
            c[f"value_{side}_den"] += w

    for g in gt:
        if g.match is not None or g.rescued:
            continue
        w_g = _exposure(g, gt_empty_rank)
        axis_membership(g, "rec", w_g)
        # missed box-bearing GT: grounded-recall denominator with no numerator
        # (unmatched-row rule); selects charge per box-bearing option
        if g.kind in SELECT_KINDS:
            c["grounded_rec_den"] += sum(1 for opt in g.options if opt.get("bbox"))
        elif g.bbox:
            c["grounded_rec_den"] += w_g
        rank = gt_empty_rank.get(id(g))
        if rank is None:  # valued field or select: full miss
            c["missed_valued"] += 1
            c["miss_w"] += 1.0
        else:
            c["missed_empty"] += 1
            c["miss_w"] += 0.5 if rank == 0 else 0.25
            if rank:
                c["duplicate_empty_gt"] += 1
    for p in pred:
        if p.match is not None:
            continue
        axis_membership(p, "prec", _exposure(p, pred_empty_rank))
        rank = pred_empty_rank.get(id(p))
        if rank is None:
            c["fp_w"] += 1.0
        else:
            c["spurious_empty"] += 1
            c["fp_w"] += 0.5 if rank == 0 else 0.25
            if rank:
                c["duplicate_empty_pred"] += 1

    for g, p in matched:
        w_g, w_p = _exposure(g, gt_empty_rank), _exposure(p, pred_empty_rank)
        # The id-in-label strip serves MATCHING (one mistake must not become a miss
        # plus a false positive) and the recoverability diagnostics (id_n/id_exact,
        # baked_ids). The AXES are slot-literal: baking the printed id into the label
        # is an error — the label axis compares the raw emitted label (graded ding)
        # and the id axis treats the id slot as dropped.
        id_score, stripped = _id_component(g, p)
        if stripped is not None:
            c["baked_ids"] += 1
        if g.label:
            c["label_rec_den"] += w_g
            c["label_gt_on_match"] += 1
        if p.label:
            c["label_prec_den"] += w_p
        if g.label and p.label:
            sim = text_sim(g.label, p.label)
            c["label_n"] += 1
            c["label_sim_sum"] += sim
            c["label_exact"] += fold(g.label) == fold(p.label)
            c["label_rec_num"] += sim * w_g
            c["label_prec_num"] += sim * w_p
        elif p.label:
            c["hallucinated_labels"] += 1
        # id axis, slot-literal: credit only a two-sided id-slot agreement. A dropped
        # or baked id is a recall-side 0; an id invented on a GT id-less field is a
        # precision-side 0.
        slot_id_score = id_score if p.id else None
        if g.id:
            c["id_rec_den"] += w_g
            if slot_id_score:
                c["id_rec_num"] += w_g
        if p.id:
            c["id_prec_den"] += w_p
            if slot_id_score:
                c["id_prec_num"] += w_p
        if id_score is not None:
            c["id_n"] += 1
            c["id_exact"] += id_score
            if not id_score:
                c["id_mismatch_on_match"] += 1

        ck = kind_credit(g.kind, p.kind)
        c["kind_n"] += 1
        c["kind_credit"] += ck
        c["kind_strict_credit"] += g.kind == p.kind
        c["kind_rec_den"] += w_g
        c["kind_rec_num"] += ck * w_g
        c["kind_prec_den"] += w_p
        c["kind_prec_num"] += ck * w_p
        if g.kind not in SELECT_KINDS and is_empty_value(g.value) and not is_empty_value(p.value):
            c["hallucinated_value_in_blank"] += 1

        # content credit for the axis F1s: scalar agreement only when both sides are
        # non-select AND claim the SAME content axis — a select's `value` slot is a
        # None artifact (its content is the options), and a state emitted as a value
        # (or vice versa) is not a transcription of it, whatever norm_value_eq says.
        # The matched-pair diagnostics (value_ok/cb_ok) keep their ungated semantics.
        ok_diag = norm_value_eq(g.value, p.value)
        ok = (
            g.kind not in SELECT_KINDS
            and p.kind not in SELECT_KINDS
            and _is_checkbox_axis(g.kind, g.value) == _is_checkbox_axis(p.kind, p.value)
            and ok_diag
        )

        # grounded family, field level: graded only where GT carries a box (a pred
        # claim against box-less GT can never be verified — the sparse-annotation
        # rule). Matched selects grade per OPTION in the branch below; grading the
        # parent's union box too would double-count the same regions.
        if g.kind not in SELECT_KINDS and g.bbox:
            c["grounded_rec_den"] += w_g
            if p.bbox:
                c["grounded_prec_den"] += w_p
                pair_iou = best_pair_iou(g.bbox, p.bbox)
                c["iou_n"] += 1
                c["iou_sum"] += pair_iou
                if ok and pair_iou >= IOU_THRESHOLD:
                    c["grounded_rec_num"] += w_g
                    c["grounded_prec_num"] += w_p

        if g.kind in SELECT_KINDS:
            c["select_n"] += 1
            if p.kind in SELECT_KINDS:
                c["select_shaped"] += 1
                if p.kind == g.kind:
                    c["select_shaped_strict"] += 1
                else:
                    c["select_kind_swap"] += 1
            for go, po in match_options(g, p):
                c["cb_n"] += 1
                c["checkbox_rec_den"] += 1
                state_ok = po is not None and norm_value_eq(go["value"], po["value"])
                if state_ok:
                    c["cb_ok"] += 1
                    c["checkbox_rec_num"] += 1
                    c["checkbox_prec_num"] += 1  # po is one of p's options
                # per-option grounding, weight 1.0/option like the checkbox axis; an
                # invented pred option has no GT counterpart here and stays ungraded
                if go.get("bbox"):
                    c["grounded_rec_den"] += 1
                    if po is not None and po.get("bbox"):
                        c["grounded_prec_den"] += 1
                        pair_iou = best_pair_iou(go["bbox"], po["bbox"])
                        c["iou_n"] += 1
                        c["iou_sum"] += pair_iou
                        if state_ok and pair_iou >= IOU_THRESHOLD:
                            c["grounded_rec_num"] += 1
                            c["grounded_prec_num"] += 1
        elif _is_checkbox_axis(g.kind, g.value):
            c["cb_n"] += 1
            c["cb_ok"] += ok_diag
            c["checkbox_rec_den"] += w_g
            c["checkbox_rec_num"] += ok * w_g
        else:
            c["value_n"] += 1
            c["value_ok"] += ok_diag
            c["value_rec_den"] += w_g
            c["value_rec_num"] += ok * w_g
            # split by GT transcription type; empty-GT pairs stay in the overall
            # counters only (their diagnostic is hallucinated_value_in_blank)
            if not is_empty_value(g.value):
                if numeric_or_none(g.value) is not None:
                    c["value_n_numeric"] += 1
                    c["value_ok_numeric"] += ok_diag
                else:
                    c["value_n_text"] += 1
                    c["value_ok_text"] += ok_diag

        # precision side of the same pair, membership by the PREDICTED kind (matched
        # agreeing options were credited to checkbox_prec_num in the loop above)
        if p.kind in SELECT_KINDS:
            c["checkbox_prec_den"] += len(p.options)
        elif _is_checkbox_axis(p.kind, p.value):
            c["checkbox_prec_den"] += w_p
            c["checkbox_prec_num"] += ok * w_p
        else:
            c["value_prec_den"] += w_p
            c["value_prec_num"] += ok * w_p

    for g in rescued_gt:  # option-level rescue: states score, shape is a violation
        w_g = _exposure(g, gt_empty_rank)
        c["kind_n"] += 1  # kind credit 0 (select emitted as checkboxes)
        c["select_n"] += 1
        c["kind_rec_den"] += w_g
        if g.id:  # the select-level id/label were not emitted — recall-side 0s
            c["id_rec_den"] += w_g
        if g.label:
            c["label_rec_den"] += w_g
        rescuer_ok: dict[int, bool] = {}
        for opt in g.options:
            c["cb_n"] += 1
            c["checkbox_rec_den"] += 1
            p_rescuer = opt.get("_rescued_by")
            agree = p_rescuer is not None and norm_value_eq(opt["value"], p_rescuer.value)
            if p_rescuer is not None:
                rescuer_ok[id(p_rescuer)] = agree
                if agree:
                    c["cb_ok"] += 1
                    c["checkbox_rec_num"] += 1
            # option grounding under rescue: the rescuer checkbox IS the pred option
            if opt.get("bbox"):
                c["grounded_rec_den"] += 1
                if p_rescuer is not None and p_rescuer.bbox:
                    c["grounded_prec_den"] += 1
                    pair_iou = best_pair_iou(opt["bbox"], p_rescuer.bbox)
                    c["iou_n"] += 1
                    c["iou_sum"] += pair_iou
                    if agree and pair_iou >= IOU_THRESHOLD:
                        c["grounded_rec_num"] += 1
                        c["grounded_prec_num"] += 1
        # rescuers stay OUT of the label/id axes: their labels name options, not
        # fields, and the shape violation is already charged to kind + structure
        for p in pred:
            if p.match is g:
                w_p = _exposure(p, pred_empty_rank)
                c["kind_prec_den"] += w_p
                c["checkbox_prec_den"] += w_p
                if rescuer_ok.get(id(p)):
                    c["checkbox_prec_num"] += w_p

    # structure: grouping co-membership F1 over matched (non-rescued) field pairs
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            g1, g2 = matched[i][0], matched[j][0]
            p1, p2 = matched[i][1], matched[j][1]
            gt_same, pred_same = g1.parent == g2.parent, p1.parent == p2.parent
            c["grp_gt_pairs"] += gt_same
            c["grp_pred_pairs"] += pred_same
            c["grp_tp"] += gt_same and pred_same
    order = _kendall_order([(g.flat_idx, p.flat_idx) for g, p in matched])
    if order is not None:
        c["order_num"] += order
        c["order_den"] += 1
    return c


COMPOSITE_WEIGHTS = {
    "value_f1": 0.25,
    "label_f1": 0.20,
    "checkbox_f1": 0.15,
    "kind_f1": 0.10,
    "id_f1": 0.10,
    "structure": 0.20,
}


def finalize(c: dict[str, float]) -> dict[str, Any]:
    def ratio(num: float, den: float) -> float | None:
        return num / den if den else None

    def f1_of(precision: float | None, recall: float | None) -> float | None:
        if precision and recall:
            return 2 * precision * recall / (precision + recall)
        return 0.0 if (recall == 0.0 or precision == 0.0) else None

    recall = ratio(c["matched_gt"], c["matched_gt"] + c["miss_w"])
    precision = ratio(c["matched_pred"], c["matched_pred"] + c["fp_w"])

    grp_p = ratio(c["grp_tp"], c["grp_pred_pairs"])
    grp_r = ratio(c["grp_tp"], c["grp_gt_pairs"])
    grouping: float | None = None
    if grp_p is not None and grp_r is not None:
        grouping = 2 * grp_p * grp_r / (grp_p + grp_r) if (grp_p + grp_r) else 0.0
    elif c["grp_gt_pairs"] or c["grp_pred_pairs"]:
        grouping = 0.0
    order = ratio(c["order_num"], c["order_den"])
    shape = ratio(c["select_shaped"], c["select_n"])
    structure_parts = [x for x in (grouping, order, shape) if x is not None]
    structure = sum(structure_parts) / len(structure_parts) if structure_parts else None
    kind_acc = ratio(c["kind_credit"], c["kind_n"])

    r: dict[str, Any] = {
        "field_recall": recall,
        "field_precision": precision,
        "field_f1": f1_of(precision, recall),
        "value_acc": ratio(c["value_ok"], c["value_n"]),
        "checkbox_acc": ratio(c["cb_ok"], c["cb_n"]),
        "kind_acc": kind_acc,
        "structure": structure,
        # per-axis diagnostics (never in the composite)
        "label_sim_mean": ratio(c["label_sim_sum"], c["label_n"]),
        "label_exact_rate": ratio(c["label_exact"], c["label_n"]),
        "label_coverage": ratio(c["label_n"], c["label_gt_on_match"]),
        "id_exact_rate": ratio(c["id_exact"], c["id_n"]),
        "value_acc_numeric": ratio(c["value_ok_numeric"], c["value_n_numeric"]),
        "value_acc_text": ratio(c["value_ok_text"], c["value_n_text"]),
        "kind_acc_strict": ratio(c["kind_strict_credit"], c["kind_n"]),
        "kind_acc_select_agnostic": kind_acc,
        "grouping_f1": grouping,
        "order_kendall": order,
        "select_shape_agnostic": shape,
        "select_shape_strict": ratio(c["select_shaped_strict"], c["select_n"]),
        "skeleton_teds": ratio(c["skeleton_teds_num"], c["skeleton_teds_den"]),
    }
    for axis in AXES:
        axis_rec = ratio(c[f"{axis}_rec_num"], c[f"{axis}_rec_den"])
        axis_prec = ratio(c[f"{axis}_prec_num"], c[f"{axis}_prec_den"])
        r[f"{axis}_recall"] = axis_rec
        r[f"{axis}_precision"] = axis_prec
        r[f"{axis}_f1"] = f1_of(axis_prec, axis_rec)
    # grounded family (never in the composite): zero denominators — a doc whose GT
    # carries no boxes — emit None rather than a punitive 0.0, at every rollup level
    grounded_rec = ratio(c["grounded_rec_num"], c["grounded_rec_den"])
    grounded_prec = ratio(c["grounded_prec_num"], c["grounded_prec_den"])
    r["grounded_recall"] = grounded_rec
    r["grounded_precision"] = grounded_prec
    r["grounded_f1"] = f1_of(grounded_prec, grounded_rec)
    r["bbox_iou_mean"] = ratio(c["iou_sum"], c["iou_n"])
    avail = {k: w for k, w in COMPOSITE_WEIGHTS.items() if r.get(k) is not None}
    r["composite"] = sum(r[k] * w for k, w in avail.items()) / sum(avail.values()) if avail else None
    r["counts"] = c
    return r


def score_pages(gt_pages: list[dict[str, Any]] | None, pred_pages: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Score a whole doc: pages paired by page_number, counters summed."""
    gt_fields, gt_statics = flatten_pages(gt_pages)
    pred_fields, _ = flatten_pages(pred_pages)

    def items_by_page(pages: list[dict[str, Any]] | None) -> dict[int, list[dict[str, Any]]]:
        """Page number -> its items, concatenated if a number repeats (keeping flatten
        order, so the skeleton walk still consumes that page's fields one-for-one)."""
        out: dict[int, list[dict[str, Any]]] = {}
        for pg in pages or []:
            out.setdefault(pg.get("page_number", 1), []).extend(pg.get("items") or [])
        return out

    gt_items, pred_items = items_by_page(gt_pages), items_by_page(pred_pages)
    total: dict[str, float] | None = None
    page_nos = sorted({f.page for f in gt_fields} | {f.page for f in pred_fields})
    for pg in page_nos:
        gt_page_fields = [f for f in gt_fields if f.page == pg]
        pred_page_fields = [f for f in pred_fields if f.page == pg]
        c = score_counts(gt_page_fields, pred_page_fields)
        # AFTER score_counts: the skeleton reads the alignment state align() just set
        teds, weight = page_skeleton_teds(gt_items.get(pg), pred_items.get(pg), gt_page_fields, pred_page_fields)
        c["skeleton_teds_num"] += teds * weight
        c["skeleton_teds_den"] += weight
        total = c if total is None else {k: total[k] + c[k] for k in c}
    r = finalize(total if total is not None else score_counts([], []))
    r["gt_static_text"] = gt_statics
    return r
