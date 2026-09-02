"""Table-merging (junction merge) metric — parse.

Scores whether a parser joins table fragments that belong to one logical table —
both within a page (section-break continuations) and across a page break — and
keeps genuinely distinct tables apart. Reads ``metadata["table_merging"]``:

    atoms          : ordered split GT tables  (reading order = sorted by page,idx)
    logical_tables : the merged grouping; ``fragments`` lists the atom keys it joins

Each consecutive atom pair ``(aᵢ, aᵢ₊₁)`` is a *junction*. Ground-truth says
"merge" iff both atoms belong to the same logical table; the parser says "merge"
iff both atoms land in the same predicted table (assignment = global argmax of
GriTS containment-recall, **no page blocking, no threshold**). Each junction is a
merge/separate decision scored over all four confusion quadrants — correct
*non*-merges (TN) count too, so over-, under-, and appropriate-merging all
register.

Metrics (rates 0..1, higher better):
  - table_merge_accuracy   : HEADLINE — (TP+TN)/N over all junctions (page-agnostic)
  - table_merge_precision  : over-merge sensitivity   (TP / (TP+FP))
  - table_merge_recall     : under-merge sensitivity  (TP / (TP+FN))
  - table_merge_specificity: TN / (TN+FP)
  - table_merge_f1
  - table_merge_accuracy_within_page / _cross_page : per-kind slices
  - table_merge_tp/fp/fn/tn: confusion-matrix counts (metadata["count"] ⇒ the
    runner SUMS them at corpus level → total_table_merge_{tp,fp,fn,tn})
  - table_merge_grits_con  : secondary content signal (merged logical-table GriTS)

Pure over ``(parse_output, gt)``; no I/O.
"""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup
from lxml import etree
from lxml import html as lxml_html

from parse_bench.evaluation.metrics.parse._vendor_grits_reference import grits_from_html
from parse_bench.schemas.evaluation import MetricValue


def _all_predicted_tables(parse_output: Any) -> list[str]:
    """Flat, reading-ordered list of predicted ``<table>`` HTML across all pages.

    Page-agnostic by design: a cross-page merge surfaces as one predicted table
    (on the first page), and index identity is what "same predicted table" means.
    """
    out: list[str] = []
    for page in getattr(parse_output, "pages", []) or []:
        md = getattr(page, "markdown", "") or ""
        for t in BeautifulSoup(md, "lxml").find_all("table"):
            out.append(str(t))
    return out


def _xml_safe(table_html: str) -> str | None:
    """Normalize a table fragment to well-formed XML the vendor GriTS can parse.

    ``_vendor_grits_reference.html_to_cells`` uses ``xml.etree.ElementTree``
    (strict XML — raises on unclosed tags / bare ``&``), unlike the lenient lxml
    used elsewhere in the bench. Re-serialize through lxml (which repairs and
    escapes) and return just the ``<table>``. ``None`` if there is no table.
    """
    if not table_html or not table_html.strip():
        return None
    try:
        node = lxml_html.fromstring(table_html)
    except Exception:
        return None
    table = node if node.tag == "table" else node.find(".//table")
    if table is None:
        return None
    return str(etree.tostring(table, encoding="unicode", method="xml"))


def _grits(true_html: str, pred_html: str, key: str) -> float:
    """One GriTS sub-score between two table fragments; 0.0 on any parse failure."""
    t = _xml_safe(true_html)
    p = _xml_safe(pred_html)
    if t is None or p is None:
        return 0.0
    try:
        return float(grits_from_html(t, p).get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision / recall / F1; a 0-denominator component is vacuously 1.0."""
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _slice_accuracy(slice_name: str, b: dict[str, int]) -> list[MetricValue]:
    """Per-kind accuracy slice; omitted entirely when that kind has no junctions."""
    n = b["tp"] + b["fp"] + b["fn"] + b["tn"]
    if n == 0:
        return []
    acc = (b["tp"] + b["tn"]) / n
    return [
        MetricValue(
            metric_name=f"table_merge_accuracy_{slice_name}",
            value=acc,
            metadata={**b, "n_junctions": n},
        )
    ]


def _logical_table_grits_con(logical: list[dict[str, Any]], pred: list[str]) -> float:
    """Secondary content signal: mean GriTS-Con of each merged logical table vs its
    best-matching predicted table. 1.0 vacuously when there are no logical tables."""
    if not logical:
        return 1.0
    if not pred:
        return 0.0
    scores = [max((_grits(lt.get("html", ""), p, "grits_con") for p in pred), default=0.0) for lt in logical]
    return sum(scores) / len(scores)


def compute_table_merging_metrics(parse_output: Any, gt: dict[str, Any]) -> list[MetricValue]:
    atoms: list[dict[str, Any]] = gt.get("atoms") or []
    logical: list[dict[str, Any]] = gt.get("logical_tables") or []
    frag_logical = {f: lt["id"] for lt in logical for f in lt.get("fragments", [])}

    pred = _all_predicted_tables(parse_output)

    # Assign each atom to the predicted table that best CONTAINS it (recall-oriented,
    # global argmax, no page blocking, no threshold). None when the parser produced
    # no tables OR no predicted table contains the atom at all (dropped fragment).
    assign: dict[str, int | None] = {}
    for a in atoms:
        if not pred:
            assign[a["key"]] = None
            continue
        scores = [_grits(a["html"], p, "grits_recall_con") for p in pred]
        best = max(range(len(pred)), key=lambda j: scores[j])
        assign[a["key"]] = best if scores[best] > 0.0 else None

    if len(atoms) < 2:
        return []  # junction-free doc — emit nothing, don't pollute corpus averages

    tp = fp = fn = tn = 0
    within = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    cross = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for a, b in zip(atoms, atoms[1:], strict=False):
        la, lb = frag_logical.get(a["key"]), frag_logical.get(b["key"])
        gt_merge = la is not None and la == lb
        ja, jb = assign[a["key"]], assign[b["key"]]
        pred_merge = ja is not None and ja == jb
        bucket = within if a["page"] == b["page"] else cross
        if gt_merge and pred_merge:
            tp += 1
            bucket["tp"] += 1
        elif gt_merge and not pred_merge:
            fn += 1
            bucket["fn"] += 1
        elif not gt_merge and pred_merge:
            fp += 1
            bucket["fp"] += 1
        else:
            tn += 1
            bucket["tn"] += 1

    n = tp + fp + fn + tn  # == len(atoms) - 1
    acc = (tp + tn) / n
    prec, rec, f1 = _prf(tp, fp, fn)
    spec = tn / (tn + fp) if (tn + fp) else 1.0
    grits_con = _logical_table_grits_con(logical, pred)

    cm = f"TP={tp} FP={fp} FN={fn} TN={tn}"
    md = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_atoms": len(atoms), "n_logical": len(logical)}

    results = [
        # HEADLINE (doc-averaged at corpus level via avg_table_merge_accuracy) + inline CM.
        MetricValue(metric_name="table_merge_accuracy", value=acc, metadata=md, details=[cm]),
        MetricValue(metric_name="table_merge_precision", value=prec, metadata=md),
        MetricValue(metric_name="table_merge_recall", value=rec, metadata=md),
        MetricValue(metric_name="table_merge_specificity", value=spec, metadata=md),
        MetricValue(metric_name="table_merge_f1", value=f1, metadata=md),
        # Confusion counts as first-class metrics. metadata["count"] makes the runner
        # SUM them at corpus level (total_table_merge_{tp,fp,fn,tn}) rather than average,
        # so over-/under-/appropriate-merging stay visible beside the headline.
        MetricValue(metric_name="table_merge_tp", value=float(tp), metadata={"count": tp, "n_atoms": len(atoms)}),
        MetricValue(metric_name="table_merge_fp", value=float(fp), metadata={"count": fp, "n_atoms": len(atoms)}),
        MetricValue(metric_name="table_merge_fn", value=float(fn), metadata={"count": fn, "n_atoms": len(atoms)}),
        MetricValue(metric_name="table_merge_tn", value=float(tn), metadata={"count": tn, "n_atoms": len(atoms)}),
        MetricValue(metric_name="table_merge_grits_con", value=grits_con, metadata=md),
    ]
    results += _slice_accuracy("within_page", within)
    results += _slice_accuracy("cross_page", cross)
    return results
