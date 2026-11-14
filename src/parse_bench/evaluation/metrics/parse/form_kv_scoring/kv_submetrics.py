"""Counters, finalize and doc scoring for the key-value list metric, in `formnode_scoring`'s idiom:
per-page counters summed, then `finalize()`; alignment and exposure weights are its `align()`
and `_exposure` unchanged.

Axes are F1s with a GT-side (recall) and a pred-side (precision) denominator, membership by each
side's own kind: value / value_strict, label / label_strict, id / id_lenient (a baked-looking label is an
id claim only when the GT record has an id), checkbox (boxes plus one unit per option), signature, core_kv (every
unit; label agrees and value/state agrees). Structure = mean of grouping, order, select shape.
A doc keeps two counter dicts, exhaustive pages and all pages: the F1 axes read the first, the
`spot_*` recalls and `spot_declared_precision` the second.
"""

from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_normalization import (
    baked_id,
    id_eq,
    label_eq_strict,
    label_sim,
    value_eq,
    value_eq_strict,
)
from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_records import (
    COVERAGE_EXHAUSTIVE,
    KvPage,
    KvRecord,
)
from parse_bench.evaluation.metrics.parse.formnode_scoring.field_flatten import SELECT_KINDS, Field
from parse_bench.evaluation.metrics.parse.formnode_scoring.field_matching import (
    _id_component,
    align,
    match_options,
)
from parse_bench.evaluation.metrics.parse.formnode_scoring.form_submetrics import (
    _empty_group_rank,
    _exposure,
    _kendall_order,
)
from parse_bench.evaluation.metrics.parse.formnode_scoring.text_normalization import fold, is_empty_value
from parse_bench.evaluation.metrics.parse.rules_base import CELL_FUZZY_MATCH_THRESHOLD

AXES = ("value", "value_strict", "label", "label_strict", "id", "id_lenient", "checkbox", "signature", "core_kv")
_PARTS = ("rec_num", "rec_den", "prec_num", "prec_den")

COUNTER_KEYS = (
    "gt_records",
    "pred_records",
    "matched_gt",
    "matched_pred",
    "rescued_selects",
    "baked_ids",
    *(f"{axis}_{part}" for axis in AXES for part in _PARTS),
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

COMPOSITE_WEIGHTS = {
    "value_f1": 0.35,
    "label_f1": 0.25,
    "checkbox_signature_f1": 0.15,
    "id_f1": 0.10,
    "structure": 0.15,
}
SPOT_COMPOSITE_WEIGHTS = {
    "spot_value_recall": 0.35,
    "spot_label_recall": 0.25,
    "spot_checkbox_signature_recall": 0.15,
    "spot_id_recall": 0.10,
    "spot_structure": 0.15,
}


def zero_counts() -> dict[str, float]:
    return dict.fromkeys(COUNTER_KEYS, 0.0)


def add_counts(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: a[k] + b[k] for k in COUNTER_KEYS}


def _content_class(r: KvRecord) -> str:
    if r.kind in SELECT_KINDS:
        return "select"
    if r.kind in ("checkbox", "signature"):
        return r.kind
    return "text"


def _label_agree(g: KvRecord, p_label: str | None) -> bool:
    if not g.raw_label:
        return not p_label
    if not p_label:
        return False
    return label_sim(g.raw_label, p_label) >= CELL_FUZZY_MATCH_THRESHOLD


def _option_label_agree(go: dict[str, Any], po: dict[str, Any]) -> bool:
    if go.get("id") and po.get("id") and fold(str(go["id"])) == fold(str(po["id"])):
        return True
    if not go.get("label"):
        return not po.get("label")
    if not po.get("label"):
        return False
    return label_sim(go["label"], po["label"]) >= CELL_FUZZY_MATCH_THRESHOLD


class _Counter:
    def __init__(self) -> None:
        self.c = zero_counts()

    def bump(self, key: str, w: float = 1.0) -> None:
        self.c[key] += w

    def axis(self, axis: str, side: str, w: float, credit: bool | float) -> None:
        self.c[f"{axis}_{side}_den"] += w
        self.c[f"{axis}_{side}_num"] += float(credit) * w

    def membership(self, r: KvRecord, side: str, w: float) -> None:
        """Denominators of an unmatched record: every axis it belongs to, no credit."""
        if r.raw_label:
            self.axis("label", side, w, False)
            self.axis("label_strict", side, w, False)
        if r.id:
            self.axis("id", side, w, False)
            self.axis("id_lenient", side, w, False)
        self.content_membership(r, side, w)

    def content_membership(self, r: KvRecord, side: str, w: float) -> None:
        cls = _content_class(r)
        if cls == "select":
            self.axis("checkbox", side, float(len(r.options)), False)
            self.axis("core_kv", side, float(len(r.options)), False)
            return
        self.axis("core_kv", side, w, False)
        if cls == "checkbox":
            self.axis("checkbox", side, w, False)
        elif cls == "signature":
            self.axis("signature", side, w, False)
        else:
            self.axis("value", side, w, False)
            self.axis("value_strict", side, w, False)


def _reset(records: list[KvRecord]) -> None:
    for r in records:
        r.match = None
        r.rescued = False
        for o in r.options:
            o.pop("_rescued_by", None)


def score_counts(gt: list[KvRecord], pred: list[KvRecord]) -> dict[str, float]:
    _reset(gt)
    _reset(pred)
    gt_fields: list[Field] = list(gt)
    pred_fields: list[Field] = list(pred)
    align(gt_fields, pred_fields)
    k = _Counter()
    c = k.c
    c["gt_records"], c["pred_records"] = float(len(gt)), float(len(pred))
    rescued = [g for g in gt if g.rescued]
    matched: list[tuple[KvRecord, KvRecord]] = []
    for g in gt:
        if g.match is not None and not g.rescued and isinstance(g.match, KvRecord):
            matched.append((g, g.match))
    c["matched_gt"] = float(len(matched) + len(rescued))
    c["matched_pred"] = float(sum(1 for p in pred if p.match is not None))
    c["rescued_selects"] = float(len(rescued))
    gt_rank, pred_rank = _empty_group_rank(list(gt)), _empty_group_rank(list(pred))

    for g in gt:
        if g.match is None and not g.rescued:
            k.membership(g, "rec", _exposure(g, gt_rank))
    for p in pred:
        if p.match is None:
            k.membership(p, "prec", _exposure(p, pred_rank))

    for g, p in matched:
        w_g, w_p = _exposure(g, gt_rank), _exposure(p, pred_rank)
        gc, pc = _content_class(g), _content_class(p)
        id_score, stripped = _id_component(g, p)
        if stripped is not None:
            c["baked_ids"] += 1
        p_label_lenient = stripped if stripped is not None else p.raw_label

        # both label duals read the raw emitted text: a baked id dings the graded axis
        both = bool(g.raw_label and p.raw_label)
        sim = label_sim(g.raw_label, p.raw_label) if both else 0.0
        strict = both and label_eq_strict(g.raw_label, p.raw_label)
        if g.raw_label:
            k.axis("label", "rec", w_g, sim)
            k.axis("label_strict", "rec", w_g, strict)
        if p.raw_label:
            k.axis("label", "prec", w_p, sim)
            k.axis("label_strict", "prec", w_p, strict)

        slot_ok = bool(g.id and p.id and id_eq(g.id, p.id))
        lenient_ok = id_score == 1.0
        if g.id:
            k.axis("id", "rec", w_g, slot_ok)
            k.axis("id_lenient", "rec", w_g, lenient_ok)
        if p.id:
            k.axis("id", "prec", w_p, slot_ok)
        if p.id or stripped is not None or (g.id and baked_id(p.raw_label)):
            k.axis("id_lenient", "prec", w_p, lenient_ok)

        if gc == "select" or pc == "select":
            if gc == "select":
                c["select_n"] += 1
                c["select_shaped"] += pc == "select"
            if gc == "select" and pc == "select":
                for go, po in match_options(g, p):
                    state_ok = po is not None and value_eq(go["value"], po["value"])
                    core_ok = state_ok and po is not None and _option_label_agree(go, po)
                    k.axis("checkbox", "rec", 1.0, state_ok)
                    k.axis("core_kv", "rec", 1.0, core_ok)
                    c["checkbox_prec_num"] += state_ok
                    c["core_kv_prec_num"] += core_ok
                    if go["value"] is False:
                        c["declared_den"] += 1
                        c["declared_num"] += po is not None and po["value"] is False
                c["checkbox_prec_den"] += len(p.options)
                c["core_kv_prec_den"] += len(p.options)
            else:
                k.content_membership(g, "rec", w_g)
                k.content_membership(p, "prec", w_p)
            continue

        same_axis = gc == pc
        ok = same_axis and value_eq(g.value, p.value)
        ok_strict = same_axis and value_eq_strict(g.value, p.value)
        core_ok = ok and _label_agree(g, p_label_lenient)
        k.axis("core_kv", "rec", w_g, core_ok)
        k.axis("core_kv", "prec", w_p, core_ok)
        for r, side, w in ((g, "rec", w_g), (p, "prec", w_p)):
            cls = _content_class(r)
            if cls == "checkbox":
                k.axis("checkbox", side, w, ok)
            elif cls == "signature":
                k.axis("signature", side, w, ok)
            else:
                k.axis("value", side, w, ok)
                k.axis("value_strict", side, w, ok_strict)
        if gc == "text" and is_empty_value(g.value):
            c["declared_den"] += 1
            c["declared_num"] += pc == "text" and is_empty_value(p.value)
        elif gc == "checkbox" and g.value is False:
            c["declared_den"] += 1
            c["declared_num"] += pc == "checkbox" and p.value is False

    for g in rescued:
        w_g = _exposure(g, gt_rank)
        c["select_n"] += 1
        if g.id:
            k.axis("id", "rec", w_g, False)
            k.axis("id_lenient", "rec", w_g, False)
        if g.raw_label:
            k.axis("label", "rec", w_g, False)
            k.axis("label_strict", "rec", w_g, False)
        rescuer_ok: dict[int, tuple[bool, bool]] = {}
        for opt in g.options:
            rescuer = opt.get("_rescued_by")
            agree = isinstance(rescuer, KvRecord) and value_eq(opt["value"], rescuer.value)
            core_ok = False
            if isinstance(rescuer, KvRecord):
                anchored = bool(
                    opt.get("id")
                    and (rescuer.id or rescuer.raw_label)
                    and id_eq(opt["id"], rescuer.id or rescuer.raw_label)
                )
                core_ok = agree and (anchored or _option_label_agree(opt, {"label": rescuer.raw_label}))
                rescuer_ok[id(rescuer)] = (agree, core_ok)
            k.axis("checkbox", "rec", 1.0, agree)
            k.axis("core_kv", "rec", 1.0, core_ok)
            if opt["value"] is False:
                c["declared_den"] += 1
                c["declared_num"] += agree
        for p in pred:
            if p.match is g:
                w_p = _exposure(p, pred_rank)
                agree, core_ok = rescuer_ok.get(id(p), (False, False))
                k.axis("checkbox", "prec", w_p, agree)
                k.axis("core_kv", "prec", w_p, core_ok)

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


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision and recall:
        return 2 * precision * recall / (precision + recall)
    return 0.0 if (recall == 0.0 or precision == 0.0) else None


def _structure(c: dict[str, float]) -> dict[str, float | None]:
    grp_p, grp_r = _ratio(c["grp_tp"], c["grp_pred_pairs"]), _ratio(c["grp_tp"], c["grp_gt_pairs"])
    grouping: float | None = None
    if grp_p is not None and grp_r is not None:
        grouping = 2 * grp_p * grp_r / (grp_p + grp_r) if (grp_p + grp_r) else 0.0
    elif c["grp_gt_pairs"] or c["grp_pred_pairs"]:
        grouping = 0.0
    order = _ratio(c["order_num"], c["order_den"])
    shape = _ratio(c["select_shaped"], c["select_n"])
    parts = [x for x in (grouping, order, shape) if x is not None]
    return {
        "grouping_f1": grouping,
        "order_kendall": order,
        "select_shape": shape,
        "structure": sum(parts) / len(parts) if parts else None,
    }


def _weighted(r: dict[str, Any], weights: dict[str, float]) -> float | None:
    avail = {k: w for k, w in weights.items() if r.get(k) is not None}
    return sum(r[k] * w for k, w in avail.items()) / sum(avail.values()) if avail else None


def finalize(exhaustive: dict[str, float], every: dict[str, float]) -> dict[str, Any]:
    r: dict[str, Any] = {}
    for axis in AXES:
        rec = _ratio(exhaustive[f"{axis}_rec_num"], exhaustive[f"{axis}_rec_den"])
        prec = _ratio(exhaustive[f"{axis}_prec_num"], exhaustive[f"{axis}_prec_den"])
        r[f"{axis}_recall"], r[f"{axis}_precision"], r[f"{axis}_f1"] = rec, prec, _f1(prec, rec)
    cs_rec = _ratio(
        exhaustive["checkbox_rec_num"] + exhaustive["signature_rec_num"],
        exhaustive["checkbox_rec_den"] + exhaustive["signature_rec_den"],
    )
    cs_prec = _ratio(
        exhaustive["checkbox_prec_num"] + exhaustive["signature_prec_num"],
        exhaustive["checkbox_prec_den"] + exhaustive["signature_prec_den"],
    )
    r["checkbox_signature_f1"] = _f1(cs_prec, cs_rec)
    r.update(_structure(exhaustive))
    r["composite"] = _weighted(r, COMPOSITE_WEIGHTS)

    for axis in AXES:
        r[f"spot_{axis}_recall"] = _ratio(every[f"{axis}_rec_num"], every[f"{axis}_rec_den"])
    r["spot_checkbox_signature_recall"] = _ratio(
        every["checkbox_rec_num"] + every["signature_rec_num"],
        every["checkbox_rec_den"] + every["signature_rec_den"],
    )
    r["spot_declared_precision"] = _ratio(every["declared_num"], every["declared_den"])
    r["spot_structure"] = _structure(every)["structure"]
    r["spot_composite"] = _weighted(r, SPOT_COMPOSITE_WEIGHTS)
    r["counts"] = {"exhaustive": exhaustive, "all": every}
    return r


def score_pages(gt_pages: list[KvPage], pred_pages: list[KvPage]) -> dict[str, Any]:
    """Pages pair by page_number; a pred page with no GT page scores against an empty exhaustive page."""
    gt_by = {p.page_number: p for p in gt_pages}
    pred_by = {p.page_number: p for p in pred_pages}
    exhaustive, every = zero_counts(), zero_counts()
    n_exh = 0
    page_nos = sorted(set(gt_by) | set(pred_by))
    for pn in page_nos:
        g, p = gt_by.get(pn), pred_by.get(pn)
        c = score_counts(list(g.records) if g else [], list(p.records) if p else [])
        every = add_counts(every, c)
        if g is None or g.coverage == COVERAGE_EXHAUSTIVE:
            exhaustive = add_counts(exhaustive, c)
            n_exh += 1
    r = finalize(exhaustive, every)
    r["exhaustive_pages"] = n_exh
    r["spot_pages"] = len(page_nos) - n_exh
    return r
