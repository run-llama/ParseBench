"""Field alignment: pairwise similarity, Hungarian assignment, loose-checkbox rescue.

Evidence weights:

    label 0.40 | value 0.25 | printed id 0.20 | position 0.10 | path anchor 0.05

renormalized over the parts AVAILABLE for a pair, so absent evidence is neutral rather
than penalizing. Path anchors stay neutral when one-sided: flattened sections are charged
to the structure axis, not to alignment.

- Printed ids do not veto. Disagreeing ids are a 0.0-scored component, not an early
  return, so a mis-read id stays an id error instead of becoming a silent miss + false
  positive. Anchor: exact label + exact value + WRONG id + position 0.9 and no path
  anchor -> (0.40 + 0.25 + 0.00 + 0.09) / 0.95 ≈ 0.78, comfortably matched.
- Value evidence is positive-only: the part exists only on agreement — exact
  transcription equality, or BOTH sides blank (two empty slots agree on blankness; that
  is how anonymous empty grids pair up, ranked by position). A wrong transcription or
  one-sided emptiness must not push a pair apart — those land on the value/coverage axes
  after matching. Exception: a select facing an option-less partner keeps a 0.0-scored
  shape part, so a plain field can never outbid a real select for a select, and
  the loose-checkbox rescue survives.
- Kind is not matching evidence. Kind confusion is what the kind axis measures; letting
  it steer alignment turns one error into three.
- Position is rank-based (`page_rank`, the field's index within its own page), so one
  spurious prediction shifts later ranks by a slot instead of skewing the whole page.
- CONTENT GATE: a pair is only assignable if at least one content-bearing part is present
  (a printed-id comparison, a two-sided label, exact value agreement, or two-sided option
  overlap). Position, path, and one-sided labels rank candidates but cannot create a
  match — otherwise renormalization lets an all-anonymous pair score ~1.0 on position
  alone and capture fields (or whole selects) it shares nothing with.

Assignment is one optimal Hungarian solve over a square (n+m) x (n+m) matrix: real pairs
cost `1 - similarity`, and each field carries a private dummy priced at
`d = (1 - SIM_THRESHOLD) / 2`. A real pair beats leaving both sides unmatched iff
`1 - s < 2d`, i.e. `s > 1 - 2d = SIM_THRESHOLD`, so sub-threshold pairs are never
assigned — the optimizer strictly prefers the two dummies. Off-diagonal dummy entries are
large but finite, keeping the solve feasible.

A GT select emitted as loose lone checkboxes is rescued at the option level (states
still score) — the structure axis records the shape violation.
"""

import re
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from parse_bench.evaluation.metrics.parse.formnode_scoring.field_flatten import SELECT_KINDS, Field
from parse_bench.evaluation.metrics.parse.formnode_scoring.text_normalization import (
    fold,
    norm_value_eq,
    text_sim,
)

W_LABEL, W_VALUE, W_ID, W_POS, W_PATH = 0.40, 0.25, 0.20, 0.10, 0.05
SIM_THRESHOLD = 0.5
_FORBIDDEN = 1e6  # off-diagonal dummy cost: unusable, but finite


def kind_credit(gk: str, pk: str) -> float:
    """single_select vs multi_select is FULL credit: the primary metric is agnostic to
    the select flavor; the swap is still counted separately as the `select_kind_swap`
    diagnostic."""
    if gk == pk or {gk, pk} <= SELECT_KINDS:
        return 1.0
    return 0.0


def _option_anchor(po: dict[str, Any]) -> str | None:
    """A pred option's id anchor: its printed id, else its label — the model often
    emits the printed option id in the LABEL slot of an id-less checkbox."""
    anchor = po.get("id") if po.get("id") is not None else po.get("label")
    return str(anchor) if anchor is not None and str(anchor).strip() else None


def _opt_sim(go: dict[str, Any], po: dict[str, Any]) -> float:
    """Option-pair similarity: printed ids anchor first (including ids emitted in the
    label slot), then label similarity."""
    p_anchor = _option_anchor(po)
    if go.get("id") and p_anchor and fold(str(go["id"])) == fold(p_anchor):
        return 1.0
    return text_sim(go.get("label") or "", po.get("label") or "")


def _id_component(g: Field, p: Field) -> tuple[float | None, str | None]:
    """(score or None if unavailable, pred label with id-prefix stripped or None)."""
    if g.id and p.id:
        return (1.0 if fold(g.id) == fold(p.id) else 0.0), None
    if g.id and not p.id and p.label:
        m = re.match(re.escape(g.id) + r"\s*[.):\-]?\s+", p.label, re.I)
        if m:  # failure mode: id baked into the label — treat as an id match
            return 1.0, p.label[m.end() :]
    return None, None


def similarity(g: Field, p: Field, gt_n: int, pred_n: int) -> float:
    parts: list[tuple[float, float]] = []  # (weight, score)
    content = False  # any content-bearing comparison: ids, two-sided labels, value/option agreement

    id_score, stripped = _id_component(g, p)
    if id_score is not None:
        parts.append((W_ID, id_score))
        content = True

    p_label = stripped if stripped is not None else p.label
    if g.label and p_label:
        parts.append((W_LABEL, text_sim(g.label, p_label)))
        content = True
    elif bool(g.label) != bool(p_label):
        parts.append((W_LABEL, 0.15))  # one-sided label: weak evidence of mismatch

    g_anchor = g.path[-1] if g.path else None
    p_anchor = p.path[-1] if p.path else None
    if g_anchor and p_anchor:
        parts.append((W_PATH, text_sim(g_anchor, p_anchor)))

    # value evidence is positive-only: no agreement -> no part (renormalized away).
    # A select against an option-less partner keeps a 0.0-scored shape part so a plain
    # field never looks like a better home for a select than a real select does — and
    # the loose-checkbox rescue keeps its chance on unmatched selects.
    if g.kind in SELECT_KINDS or p.kind in SELECT_KINDS:
        if g.kind in SELECT_KINDS and p.kind in SELECT_KINDS and not g.options and not p.options:
            # two option-less selects agree on option-emptiness (the select analogue
            # of the both-blank rule below) — otherwise an identical empty select
            # cannot match itself and a perfect doc scores below 1.0
            parts.append((W_VALUE, 1.0))
            content = True
        else:
            opt_sim = 0.0
            if g.options and p.options:
                sims = [max((_opt_sim(go, po) for po in p.options), default=0.0) for go in g.options]
                opt_sim = sum(sims) / len(sims)
                content = True
            parts.append((W_VALUE, opt_sim))
    elif norm_value_eq(g.value, p.value):
        # covers both-valued exact agreement AND both-empty agreement (two blank slots
        # agree on blankness — that is how anonymous empty grids pair up, by rank);
        # one-sided emptiness or a mismatch contributes nothing
        parts.append((W_VALUE, 1.0))
        content = True

    if gt_n > 1 and pred_n > 1:
        parts.append((W_POS, 1.0 - abs(g.page_rank / (gt_n - 1) - p.page_rank / (pred_n - 1))))

    if not content:
        # Position, path, and one-sided labels RANK candidate pairs; they can never
        # CREATE a match on their own. Without this gate, renormalizing over available
        # parts lets an all-anonymous pair score ~1.0 on position + path alone.
        return 0.0
    total_w = sum(w for w, _ in parts)
    return sum(w * s for w, s in parts) / total_w if total_w else 0.0


def align(gt: list[Field], pred: list[Field]) -> None:
    """Optimal Hungarian assignment on pairwise similarity, then loose-checkbox rescue.

    Both lists arrive in flatten order for a single page, so enumeration order is the
    page rank the position component reads.
    """
    for i, f in enumerate(gt):
        f.page_rank = i
    for i, f in enumerate(pred):
        f.page_rank = i

    n, m = len(gt), len(pred)
    if n and m:
        d = (1.0 - SIM_THRESHOLD) / 2.0
        cost = np.zeros((n + m, n + m), dtype=float)
        for i, g in enumerate(gt):
            for j, p in enumerate(pred):
                cost[i, j] = 1.0 - similarity(g, p, n, m)
        cost[:n, m:] = _FORBIDDEN  # each gt field reaches only its own dummy
        cost[n:, :m] = _FORBIDDEN  # likewise each pred field
        for i in range(n):
            cost[i, m + i] = d
        for j in range(m):
            cost[n + j, j] = d
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
            if i < n and j < m:
                gt[i].match, pred[j].match = pred[j], gt[i]

    # rescue: GT select emitted as loose pred checkboxes — options still score,
    # the structure axis records the shape violation
    loose = [p for p in pred if p.match is None and p.kind == "checkbox"]
    for g in gt:
        if g.match is not None or g.kind not in SELECT_KINDS or not g.options:
            continue
        taken: list[Field] = []
        for opt in g.options:
            best, best_p = 0.0, None
            for p in loose:
                if p in taken:
                    continue
                p_anchor = p.id if p.id else p.label  # id often lands in the label slot
                if opt.get("id") and p_anchor and fold(str(opt["id"])) == fold(str(p_anchor)):
                    s = 1.0  # printed option id = strong anchor
                elif opt.get("label") and p.label:
                    s = text_sim(opt["label"], p.label)
                else:
                    continue
                if s > best:
                    best, best_p = s, p
            if best >= 0.55 and best_p is not None:
                taken.append(best_p)
                opt["_rescued_by"] = best_p
        if taken:
            g.rescued = True
            for p in taken:
                p.match = g
                loose.remove(p)


def match_options(g: Field, p: Field) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Align a matched select pair's options: printed ids are strong anchors first,
    then greedy label similarity for the rest."""
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    used: set[int] = set()
    remaining: list[dict[str, Any]] = []
    by_id: dict[str, int] = {}
    for j, po in enumerate(p.options):
        anchor = _option_anchor(po)
        if anchor is not None:
            by_id.setdefault(anchor.casefold(), j)
    for go in g.options:
        anchored_j = by_id.get(str(go["id"]).casefold()) if go.get("id") else None
        if anchored_j is not None and anchored_j not in used:
            used.add(anchored_j)
            pairs.append((go, p.options[anchored_j]))
        else:
            remaining.append(go)
    for go in remaining:
        best, best_j = 0.0, None
        for j, po in enumerate(p.options):
            if j in used:
                continue
            s = text_sim(go.get("label") or "", po.get("label") or "")
            if s > best:
                best, best_j = s, j
        if best >= 0.5 and best_j is not None:
            used.add(best_j)
            pairs.append((go, p.options[best_j]))
        else:
            pairs.append((go, None))
    return pairs
