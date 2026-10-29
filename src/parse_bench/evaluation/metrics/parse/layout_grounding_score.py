"""Page model, tokenizer and per-page scoring for ``layout_grounding_pass_rate``."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from parse_bench.evaluation.metrics.attribution.text_utils import normalize_attribution_text, tokenize
from parse_bench.evaluation.metrics.layoutdet.iou import compute_iou_matrix, xyxy_to_rotated_polygon
from parse_bench.evaluation.metrics.layoutdet.overlap import compute_layout_iou, compute_layout_overlaps

METRIC_NAME = "layout_grounding_pass_rate"

TEXTLIKE = frozenset({"Text", "List-item", "Caption", "Footnote", "Code", "Key-Value Region", "Document Index"})
L_IOA_GT = 0.5
L_IOA_PRED = 0.2
A_F1 = 0.8
TABLE_WORD_COVERAGE = 0.9
MERGE_AWARE_IOA = 0.3
DUPLICATE_IOU = 0.5
NESTED_IOA = 0.5
TIGHT_IOU = 0.5
GROUP_RECALL = 0.5
RAMP_START = 0.5
RAMP_WIDTH = 0.4
HYPHEN_GAP_POINTS = 30.0
HYPHEN_GAP_LINE_HEIGHTS = 4.0
CHAIN_OVERLAP_POINTS = 1.0  # vertical overlap of successive chained lines, in page points

HYPHENS = "-‐‑"
_PRED_HYPHEN_SPACE = re.compile(r"([^\W\d_])[-‐‑][ \t]+([^\W\d_])")
# markdown `_x_` / `__x__`; normalize_attribution_text would strip `_x` as a LaTeX subscript
_UNDERSCORE_EMPHASIS = re.compile(r"(?<![\w$\\])_{1,2}(?=[^\s_])(.+?)(?<=[^\s_])_{1,2}(?!\w)")
_HYPHEN = re.compile(r"([^\W\d_])[-‐‑]([^\W\d_])")
# normalize_attribution_text strips math but its patterns only match a doubled backslash; this is the
# single-backslash strip. Sub/sup markup glues to its base: `L$_{1}$`, `L<sub>1</sub>`, `L₁` all give `l1`.
_SCRIPT_MARKUP = re.compile(r"\$\s*[_^]\{?([^\W_]+)\}?\s*\$|(?<=\w)\^\{?([^\W_]+)\}?|</?su[bp]>", re.IGNORECASE)
_LATEX = re.compile(
    r"\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{([^}]*)\}.*?\\end\{\1\}|\\[a-zA-Z]+\*?(?:\s*\{[^}]*\})*", re.DOTALL
)
# text-formatting commands keep their argument (`\textbf{x}` is `x`); other commands drop whole, innermost
# first, so `\frac{\sqrt{2}}{\text{b}}` loses `b` with the fraction instead of leaving it as prose
_TEXT_CMD = r"\\(?:text(?:bf|it|tt|sc|rm|sf|up|md|normal)?|emph|underline)"
_LATEX_TEXT = re.compile(_TEXT_CMD + r"\s*\{([^{}]*)\}")
_LATEX_LEAF = re.compile(r"\\(?!" + _TEXT_CMD[2:] + r"\s*\{)[a-zA-Z]+\*?(?:\s*\{[^{}]*\})+")
_SENTINEL = re.compile(r"zzx([0-9a-f]+)xzz")
_CJK_PREFIXES = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")


def is_cjk(ch: str) -> bool:
    return unicodedata.name(ch, "").startswith(_CJK_PREFIXES)


def content_tokens(text: str | None, *, pred: bool = False) -> list[str]:
    """Return the attribution tokens of ``text``, keeping non-ASCII letters and digits.

    ``pred=True`` first joins ``word- word`` (a hyphen followed by spaces or tabs
    between two letters) into one word. Non-ASCII alphanumerics are lowercased and
    kept; each CJK, kana or Hangul character is its own token. Markdown underscore
    emphasis is unwrapped on both sides.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text)  # first, so `⁶` is a digit before any hyphen join
    if pred:
        text = _PRED_HYPHEN_SPACE.sub(r"\1\2", text)
    text = _UNDERSCORE_EMPHASIS.sub(r"\1", text)
    while (unwrapped := _LATEX_TEXT.sub(r"\1", _LATEX_LEAF.sub(" ", text))) != text:
        text = unwrapped
    text = _LATEX.sub(" ", _SCRIPT_MARKUP.sub(lambda m: m.group(1) or m.group(2) or "", text))
    text = _HYPHEN.sub(r"\1\2", text)
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ß", "ss")
    parts = []
    for ch in text:
        if ch.isascii() or not ch.isalnum():
            parts.append(ch)
            continue
        code = f"zzx{ord(ch.lower()[0]):x}xzz"
        parts.append(f" {code} " if is_cjk(ch) else code)
    tokens = tokenize(normalize_attribution_text("".join(parts)))
    return [_SENTINEL.sub(lambda m: chr(int(m.group(1), 16)), token) for token in tokens]


# ---------------------------------------------------------------- page model


@dataclass
class Element:
    """One scored GT element. ``box`` is xyxy in page units; ``words`` index the page words.

    ``tokens`` is the content text checked; the words stand in when the element has no content.
    """

    id: str
    cls: str
    box: tuple[float, float, float, float]
    words: list[int]
    tokens: tuple[str, ...] = ()
    angle: float | None = None
    caption: bool = False


@dataclass
class Page:
    """One page in page units. Elements are in reading order; word tokens index ``token_text``."""

    elements: list[Element]
    word_boxes: np.ndarray  # (W, 4) xyxy
    word_tokens: list[tuple[int, ...]]
    token_text: list[str]
    pred_boxes: np.ndarray  # (P, 4) xyxy
    pred_tokens: list[list[str]]
    pred_units: list[Hashable]
    pred_angles: list[float | None] = field(default_factory=list)
    skip_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    skip_angles: list[float | None] = field(default_factory=list)
    page: int = 1
    chain_tolerance: float = CHAIN_OVERLAP_POINTS


@dataclass
class PageScore:
    """Weighted numerator, element weight sum, extra penalty sum and per-row diagnostics."""

    weighted_score: float
    weight: float
    extra: float
    n_extra: int
    element_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


def _area(boxes: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(boxes[..., 2] - boxes[..., 0], 0, None) * np.clip(boxes[..., 3] - boxes[..., 1], 0, None))


def _ubox(boxes: np.ndarray) -> np.ndarray:
    return np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(compute_iou_matrix(a[None], b[None])[0, 0])


def _hulls(boxes: np.ndarray, angles: Sequence[float | None]) -> np.ndarray:
    """Axis-aligned hull of each box after its rotation; unrotated boxes are returned unchanged."""
    out = boxes.copy()
    for i, angle in enumerate(angles):
        if angle:
            poly = np.array(xyxy_to_rotated_polygon(list(boxes[i]), angle))
            out[i] = [*poly.min(axis=0), *poly.max(axis=0)]
    return out


def _prf(gt: list[str], pred: list[str]) -> tuple[float, float, float]:
    if not gt or not pred:
        return 0.0, 0.0, 0.0
    tp = sum((Counter(gt) & Counter(pred)).values())
    if not tp:
        return 0.0, 0.0, 0.0
    p, r = tp / len(pred), tp / len(gt)
    return p, r, 2 * p * r / (p + r)


def _ms_diff(a: list[str], b: list[str]) -> list[str]:
    left = Counter(b)
    out = []
    for token in a:
        if left[token]:
            left[token] -= 1
        else:
            out.append(token)
    return out


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


@dataclass
class _Match:
    b3: int = -1
    localized: bool = False
    applicable: bool = False
    f1: float | None = None
    passed: bool = False
    m: list[int] = field(default_factory=list)
    lock: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    group: int | None = None
    iou: float = 0.0
    why: str = ""


def score_page(page: Page, h_min: float) -> PageScore:
    """Score one page in page units; ``h_min`` is the minimum element weight."""
    elems = page.elements
    n_g, n_p = len(elems), len(page.pred_boxes)
    g_boxes = np.array([e.box for e in elems], dtype=float).reshape(n_g, 4)
    p_boxes = np.asarray(page.pred_boxes, dtype=float).reshape(n_p, 4)
    g_angles = [e.angle for e in elems]
    p_angles = list(page.pred_angles) or [None] * n_p
    s_boxes = np.asarray(page.skip_boxes, dtype=float).reshape(-1, 4)
    s_angles = list(page.skip_angles) or [None] * len(s_boxes)
    frame = {"page_width": 1.0, "page_height": 1.0}
    iou_gp, ioa_g, ioa_pg = compute_layout_overlaps(g_boxes, p_boxes, g_angles, p_angles, **frame)
    iou_pp, ioa_pp, _ = compute_layout_overlaps(p_boxes, p_boxes, p_angles, p_angles, **frame)
    _, _, ioa_ps = compute_layout_overlaps(s_boxes, p_boxes, s_angles, p_angles, **frame)
    w_boxes = np.asarray(page.word_boxes, dtype=float).reshape(-1, 4)
    centers = (w_boxes[:, :2] + w_boxes[:, 2:]) / 2
    in_pred = (
        (centers[:, None, 0] >= p_boxes[None, :, 0])
        & (centers[:, None, 0] <= p_boxes[None, :, 2])
        & (centers[:, None, 1] >= p_boxes[None, :, 1])
        & (centers[:, None, 1] <= p_boxes[None, :, 3])
    )
    for j, angle in enumerate(p_angles):
        if angle:  # rotated prediction: centre inside the convex polygon, either winding
            poly = np.array(xyxy_to_rotated_polygon(list(p_boxes[j]), angle))
            edges = np.roll(poly, -1, axis=0) - poly
            rel = centers[:, None, :] - poly[None]
            cross = edges[None, :, 0] * rel[..., 1] - edges[None, :, 1] * rel[..., 0]
            in_pred[:, j] = (cross >= 0).all(axis=1) | (cross <= 0).all(axis=1)
    g_hull, p_hull = _hulls(g_boxes, g_angles), _hulls(p_boxes, p_angles)

    def shape_iou(gs: Sequence[int], ps: Sequence[int]) -> float:
        """A lone box keeps its rotation (polygon IoU); a set of boxes is the axis-aligned hull of their hulls."""
        g, ga = (g_boxes[gs[0]], g_angles[gs[0]]) if len(gs) == 1 else (_ubox(g_hull[list(gs)]), None)
        p, pa = (p_boxes[ps[0]], p_angles[ps[0]]) if len(ps) == 1 else (_ubox(p_hull[list(ps)]), None)
        return float(compute_layout_iou(g[None], p[None], [ga], [pa], page_width=1.0, page_height=1.0)[0, 0])

    word_owner = np.full(len(w_boxes), -1)
    for i, e in enumerate(elems):
        word_owner[e.words] = i

    def toks(words: Sequence[int] | np.ndarray) -> list[str]:
        ids = dict.fromkeys(t for w in words for t in page.word_tokens[w])
        return [page.token_text[t] for t in ids]

    def gt_toks(gi: int) -> list[str]:
        return list(elems[gi].tokens) or toks(elems[gi].words)

    def merge_aware(gi: int, j: int) -> list[str]:
        others = [t for k in np.flatnonzero(ioa_g[:, j] >= MERGE_AWARE_IOA) if k != gi for t in gt_toks(k)]
        if not others:
            return page.pred_tokens[j]
        return _ms_diff(page.pred_tokens[j], _ms_diff(others, gt_toks(gi)))

    matches = [_element_test(gi, e, iou_gp, ioa_g, ioa_pg, in_pred, toks, merge_aware) for gi, e in enumerate(elems)]
    _group_text(page, matches, ioa_g, in_pred, toks, g_hull, p_hull, shape_iou)

    used: dict[int, int] = {}
    for gi, mt in enumerate(matches):
        for j in dict.fromkeys(mt.m):
            used.setdefault(j, gi)
    pred_rows = []
    extra = 0.0
    for j in range(n_p):
        if j in used:
            continue
        kind, reason = _triage(j, used, elems, matches, iou_pp, ioa_pp, ioa_pg, ioa_ps, in_pred, w_boxes, p_boxes)
        penalty = float(np.sqrt(_area(p_boxes[j]))) if kind == "extra" else 0.0
        extra += penalty
        pred_rows.append({"page": page.page, "pred_index": j, "kind": kind, "reason": reason, "penalty": penalty})

    weights = _weights(elems, g_boxes, w_boxes, h_min, page.chain_tolerance)
    rows = []
    total_ws = 0.0
    for gi, (e, mt) in enumerate(zip(elems, matches, strict=True)):
        if mt.group is None and mt.b3 >= 0:
            own = np.flatnonzero(ioa_g[:, mt.b3] >= L_IOA_GT)
            mt.iou = shape_iou(own.tolist(), [mt.b3]) if len(own) else 0.0
        tightness = min(1.0, max(0.0, (mt.iou - RAMP_START) / RAMP_WIDTH))
        score = tightness if mt.passed else 0.0
        total_ws += weights[gi] * score
        rows.append(
            {
                "element_id": e.id,
                "page": page.page,
                "gt_class": e.cls,
                "L": mt.localized,
                "A": (mt.f1 is not None and mt.f1 >= A_F1) if mt.applicable else None,
                "token_f1": mt.f1,
                "tightness": tightness,
                "score": score,
                "weight": float(weights[gi]),
                "group_id": None if mt.group is None else f"p{page.page}g{mt.group}",
                "why": mt.why,
            }
        )
    n_extra = sum(row["kind"] == "extra" for row in pred_rows)
    return PageScore(total_ws, float(weights.sum()), extra, n_extra, rows, pred_rows)


def _element_test(
    gi: int,
    e: Element,
    iou_gp: np.ndarray,
    ioa_g: np.ndarray,
    ioa_pg: np.ndarray,
    in_pred: np.ndarray,
    toks: Callable[[Sequence[int] | np.ndarray], list[str]],
    merge_aware: Callable[[int, int], list[str]],
) -> _Match:
    mt = _Match()
    words = np.asarray(e.words, dtype=int)
    gt = list(e.tokens) or toks(words)
    if iou_gp.shape[1]:
        valid = (ioa_g[gi] >= L_IOA_GT) & (ioa_pg[:, gi] >= L_IOA_PRED) & (iou_gp[gi] > 0)
        if valid.any():
            best = np.flatnonzero(valid & (iou_gp[gi] >= iou_gp[gi][valid].max()))

            # equal-IoU boxes: table coverage, then the best text wins, so list order never decides
            def rank(j: int) -> tuple[bool, float]:
                covered = e.cls != "Table" or not len(words) or in_pred[words, j].mean() >= TABLE_WORD_COVERAGE
                return bool(covered), _prf(gt, merge_aware(gi, j))[2] if gt else 0.0

            mt.b3 = int(max(best, key=rank))
    mt.localized = mt.b3 >= 0
    mt.why = "pass" if mt.localized else "no_localization"
    if e.cls == "Table" and mt.localized and len(words):
        if in_pred[words, mt.b3].mean() < TABLE_WORD_COVERAGE:
            mt.localized, mt.why = False, "table_word_coverage"
    if mt.b3 >= 0:
        mt.m = [mt.b3]
    elif iou_gp.shape[1]:
        counts = in_pred[words].sum(axis=0) if len(words) else np.zeros(iou_gp.shape[1])
        key = np.where(counts + iou_gp[gi] > 0, counts + iou_gp[gi] / 2, -1.0)
        if key.max() >= 0:
            mt.m = [int(np.argmax(key))]
    mt.lock = words[in_pred[words][:, mt.m].any(axis=1)] if mt.m and len(words) else np.zeros(0, dtype=int)
    mt.applicable = bool(gt) and not e.caption
    if mt.applicable and mt.m:
        mt.f1 = _prf(gt, [t for j in mt.m for t in merge_aware(gi, j)])[2]
    if mt.localized and mt.applicable and not (mt.f1 is not None and mt.f1 >= A_F1):
        mt.why = "token_f1_below_threshold"
    mt.passed = mt.localized and (not mt.applicable or (mt.f1 is not None and mt.f1 >= A_F1))
    return mt


def _is_text(e: Element) -> bool:
    return e.cls in TEXTLIKE and bool(e.words)


def _group_text(
    page: Page,
    matches: list[_Match],
    ioa_g: np.ndarray,
    in_pred: np.ndarray,
    toks: Callable[[Sequence[int]], list[str]],
    g_hull: np.ndarray,
    p_hull: np.ndarray,
    shape_iou: Callable[[Sequence[int], Sequence[int]], float],
) -> None:
    elems = page.elements
    unit_map: dict[Hashable, list[int]] = {}
    for j, key in enumerate(page.pred_units):
        unit_map.setdefault(key, []).append(j)
    units = list(unit_map.values())
    if not units:
        return
    in_unit = np.stack([in_pred[:, js].any(axis=1) for js in units], axis=1)  # (W, U)
    exclusive = in_unit.sum(axis=1) == 1
    text_ids = [i for i, e in enumerate(elems) if _is_text(e)]
    uf = _UnionFind(len(elems) + len(units))
    for k, js in enumerate(units):
        unit_words = in_unit[:, k]
        if not (unit_words & exclusive).any():
            continue
        n_unit = unit_words.sum()
        held = {i: unit_words[elems[i].words].sum() for i in text_ids}
        if not any(held[i] > len(elems[i].words) / 2 or held[i] > n_unit / 2 for i in text_ids):
            continue
        unit_tokens = [t for j in js for t in page.pred_tokens[j]]
        for i in text_ids:
            hw = [w for w in elems[i].words if unit_words[w]]
            if hw and _prf(toks(hw), unit_tokens)[1] > GROUP_RECALL:
                uf.union(i, len(elems) + k)
    comps: dict[int, tuple[list[int], list[list[int]]]] = {}
    for i in text_ids:
        comps.setdefault(uf.find(i), ([], []))[0].append(i)
    for k, js in enumerate(units):
        comp = comps.get(uf.find(len(elems) + k))
        if comp:
            comp[1].append(js)
    group_id = 0
    for mem, unit_list in comps.values():
        segs = [j for js in unit_list for j in js]
        if not segs or (len(mem) < 2 and len(segs) < 2):
            continue
        mem.sort()
        if any(b - a != 1 for a, b in zip(mem, mem[1:], strict=False)):
            continue
        seg_words = np.asarray(in_pred[:, segs].any(axis=1), dtype=bool)
        swallows_non_text = any(
            seg_words[e.words].sum() > len(e.words) / 2 if e.words else (ioa_g[i, segs] >= L_IOA_GT).any()
            for i, e in enumerate(elems)
            if not _is_text(e)
        )
        if swallows_non_text:
            continue
        member_words = [w for i in mem for w in elems[i].words]
        f1 = _prf(toks(member_words), [t for j in segs for t in page.pred_tokens[j]])[2]
        nodes = np.concatenate([g_hull[mem], p_hull[segs]])
        overlap = np.minimum(nodes[:, None, 2], nodes[None, :, 2]) > np.maximum(nodes[:, None, 0], nodes[None, :, 0])
        cols = _UnionFind(len(nodes))
        for a, b in np.argwhere(np.triu(overlap, 1)):
            cols.union(int(a), int(b))
        roots = np.array([cols.find(x) for x in range(len(nodes))])
        for x, i in enumerate(mem):
            same = roots == roots[x]
            gs = [g for g, s in zip(mem, same[: len(mem)], strict=True) if s]
            ps = [j for j, s in zip(segs, same[len(mem) :], strict=True) if s]
            mt = matches[i]
            mt.group = group_id
            mt.iou = shape_iou(gs, ps) if ps else 0.0
            mt.localized, mt.applicable, mt.f1, mt.passed = True, True, f1, f1 >= A_F1
            mt.m = list(segs)
            words = np.asarray(elems[i].words, dtype=int)
            mt.lock = words[seg_words[words]]
            mt.why = "group" if mt.passed else "group_token_f1_below_threshold"
        group_id += 1


def _triage(
    j: int,
    used: dict[int, int],
    elems: list[Element],
    matches: list[_Match],
    iou_pp: np.ndarray,
    ioa_pp: np.ndarray,
    ioa_pg: np.ndarray,
    ioa_ps: np.ndarray,
    in_pred: np.ndarray,
    w_boxes: np.ndarray,
    p_boxes: np.ndarray,
) -> tuple[str, str]:
    for k in used:
        if iou_pp[j, k] >= DUPLICATE_IOU:
            return "extra", f"duplicate of #{k}"
    held = np.flatnonzero(in_pred[:, j])
    for k, gi in used.items():
        if elems[gi].cls != "Table" or ioa_pp[j, k] < NESTED_IOA:
            continue
        if len(held) and _iou(p_boxes[j], _ubox(w_boxes[held])) >= TIGHT_IOU:
            return "neutral", f"tight child of table #{k}"
        return "extra", f"garbage child of table #{k}"
    if ioa_ps.shape[1] and (ioa_ps[j] >= NESTED_IOA).any():
        return "neutral", "on ignore/optional element"
    for gi, mt in enumerate(matches):
        if j not in mt.m and len(mt.lock) and in_pred[mt.lock, j].any():
            return "extra", f"holds words locked to #{mt.m[0]} ({elems[gi].id})"
    hosts = np.flatnonzero(ioa_pg[j] >= NESTED_IOA) if ioa_pg.shape[1] else np.zeros(0, dtype=int)
    if len(hosts):
        host = elems[int(hosts[0])]
        if host.cls == "Picture":
            return "extra", f"nested in {host.id}, no optional GT box"
        return "neutral", f"inside {host.id}"
    if (ioa_pg.shape[1] and (ioa_pg[j] > 0).any()) or (ioa_ps.shape[1] and (ioa_ps[j] > 0).any()):
        return "extra", "partial overlap"
    return "extra", "orphan words" if len(held) else "blank area"


def _weights(
    elems: list[Element], g_boxes: np.ndarray, w_boxes: np.ndarray, h_min: float, tolerance: float
) -> np.ndarray:
    all_words = [w for e in elems for w in e.words]
    heights = np.sort(w_boxes[all_words, 3] - w_boxes[all_words, 1]) if all_words else np.zeros(0)
    h = float(heights[len(heights) // 2]) if len(heights) else 0.0
    areas = _area(g_boxes)
    w = np.zeros(len(elems))
    chain: list[int] = []

    def flush() -> None:
        if chain:
            total = areas[chain].sum()
            share = areas[chain] / total if total > 0 else np.full(len(chain), 1 / len(chain))
            w[chain] = np.sqrt(_area(_ubox(g_boxes[chain]))) * share
            chain.clear()

    for i, e in enumerate(elems):
        if not _is_text(e):
            flush()
            w[i] = np.sqrt(areas[i])
            continue
        if chain:
            g, last = g_boxes[i], g_boxes[chain[-1]]
            continues = g[1] >= last[3] - tolerance and g[1] - last[3] < h and min(g[2], last[2]) > max(g[0], last[0])
            if not continues:
                flush()
        chain.append(i)
    flush()
    return np.maximum(w, h_min)
