"""Layout grounding score: weighted per-element grounding score for one document.

Public API:
    ``compute_layout_grounding_metrics(layout_rules, read_blocks, page_sizes=None) -> list[MetricValue]``
    ``content_tokens(text, pred=False) -> list[str]``

Emits ``layout_grounding_pass_rate`` = sum(w * s) / (sum(w) + sum(extra penalty)),
pooled over the document's pages. ``w`` is the element weight (square root of an
area, in page points when ``page_sizes`` holds the page, else on the unit square;
each page's sums are rescaled to the unit square so scores pool across frames),
``s`` the tightness credit of an element that passes
localization and attribution, and every extra prediction adds the square root of
its area to the denominator. Nothing is emitted when the denominator is 0.

Inputs per page: region rules (granularity ``None``/``"region"``) are the elements,
``"word"`` rules are the word boxes and their text, ``"line"`` rules order words
into lines for the hyphen join. ``optional``/``ignore`` regions and their words
are not scored; their boxes only neutralize predictions on them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium

from parse_bench.evaluation.metrics.attribution.core import PredBlock, is_truthy, normalize_layout_attributes
from parse_bench.evaluation.metrics.attribution.text_utils import extract_text_from_html
from parse_bench.evaluation.metrics.parse.layout_grounding_score import (
    CHAIN_OVERLAP_POINTS,
    HYPHEN_GAP_LINE_HEIGHTS,
    HYPHEN_GAP_POINTS,
    HYPHENS,
    METRIC_NAME,
    Element,
    Page,
    PageScore,
    content_tokens,
    is_cjk,
    score_page,
)
from parse_bench.schemas.evaluation import MetricValue
from parse_bench.test_cases.schema import LayoutTestRule

__all__ = [
    "METRIC_NAME",
    "Element",
    "Page",
    "PageScore",
    "compute_layout_grounding_metrics",
    "content_tokens",
    "score_page",
]

# ------------------------------------------------------------ rules to pages


def layout_angle(value: object) -> float | None:
    """Return the rotation ``r`` of a layout model, or None when it carries none."""
    angle = getattr(value, "r", None)
    return None if angle is None else float(angle)


def _is_region(rule: LayoutTestRule) -> bool:
    return rule.granularity in (None, "region")


def _attr(rule: LayoutTestRule, key: str) -> bool:
    return is_truthy(normalize_layout_attributes(rule.attributes).get(key))


def _grade(rule: LayoutTestRule) -> str | None:
    """Return ``"none"`` (not scored), ``"box"`` (box only) or None (box and text).

    Reads the ``grade`` attribute and the legacy ``ignore``/``optional``/``caption`` flags.
    """
    attrs = normalize_layout_attributes(rule.attributes)
    grade = str(attrs.get("grade") or "").strip().lower()
    if grade == "none" or _attr(rule, "ignore") or _attr(rule, "optional"):
        return "none"
    if grade == "box" or is_truthy(attrs.get("caption")):
        return "box"
    return None


def _is_mark_checkbox(rule: LayoutTestRule) -> bool:
    return rule.canonical_class in ("Checkbox-Selected", "Checkbox-Unselected") and (
        str(normalize_layout_attributes(rule.attributes).get("scope", "")).strip().lower() == "mark"
    )


def _rule_text(rule: LayoutTestRule) -> str:
    content = rule.content
    if content is None:
        return ""
    if content.type == "table":
        return extract_text_from_html(
            content.html, ignore_thead=_attr(rule, "exclude_table_header") or _attr(rule, "ignore_thead")
        )
    return content.text or ""


def _xyxy(rule: LayoutTestRule, pw: float, ph: float) -> tuple[float, float, float, float]:
    x, y, w, h = rule.bbox
    return (x * pw, y * ph, (x + w) * pw, (y + h) * ph)


def _build_page(
    rules: list[LayoutTestRule], blocks: list[PredBlock], page: int, size: tuple[float, float] | None
) -> tuple[Page, list[float]]:
    """Return the page model and its word heights, in points of ``size`` or on the unit square when it is None."""
    page_frame = size is not None
    pw, ph = size if size is not None else (1.0, 1.0)
    order = {id(r): n for n, r in enumerate(rules)}
    by_id = {r.id: r for r in rules if r.id}

    def ro_key(r: LayoutTestRule) -> tuple[bool, int, float, float]:
        return (r.ro_index is None, r.ro_index or 0, r.bbox[1], r.bbox[0])

    regions = [r for r in rules if _is_region(r) and not _is_mark_checkbox(r)]
    skip = [r for r in regions if _grade(r) == "none"]
    # Mark checkboxes own no words and are not scored; a prediction on one is neutral.
    shields = skip + [r for r in rules if _is_region(r) and _is_mark_checkbox(r)]
    scored = sorted(
        (r for r in regions if _grade(r) != "none"),
        key=lambda r: (ro_key(r)[:2], order[id(r)]),
    )
    skip_ids = {id(r) for r in skip}
    region_ids = {id(r) for r in regions}

    def region_of(word: LayoutTestRule) -> LayoutTestRule | None:
        node, seen = word, set()
        while node.parent_test_id and node.parent_test_id in by_id and node.parent_test_id not in seen:
            seen.add(node.parent_test_id)
            node = by_id[node.parent_test_id]
            if id(node) in region_ids:
                return node
        cx, cy = word.bbox[0] + word.bbox[2] / 2, word.bbox[1] + word.bbox[3] / 2
        hosts = [
            r for r in regions if r.bbox[0] <= cx <= r.bbox[0] + r.bbox[2] and r.bbox[1] <= cy <= r.bbox[1] + r.bbox[3]
        ]
        return min(hosts, key=lambda r: r.bbox[2] * r.bbox[3]) if hosts else None

    words = []
    for r in rules:
        if r.granularity != "word":
            continue
        region = region_of(r)
        if region is not None and id(region) not in skip_ids:
            words.append((r, region))
    token_text: list[str] = []
    word_tokens: list[list[int]] = []
    for r, _ in words:
        ids = []
        for token in content_tokens(_rule_text(r)):
            ids.append(len(token_text))
            token_text.append(token)
        word_tokens.append(ids)
    word_boxes = np.array([_xyxy(r, pw, ph) for r, _ in words], dtype=float).reshape(-1, 4)
    _join_hyphens(words, word_tokens, token_text, word_boxes, by_id, pw, ph, page_frame, ro_key)
    _keep_content_tokens(words, word_tokens, token_text)

    word_index = {id(r): n for n, (r, _) in enumerate(words)}
    elements = []
    for r in scored:
        own = [word_index[id(w)] for w, reg in words if reg is r]
        elements.append(
            Element(
                id=r.id or f"rule-{order[id(r)]}",
                cls=r.canonical_class,
                box=_xyxy(r, pw, ph),
                words=own,
                # pred=True: content keeps line-break hyphens as `infor- mation`, joined like the prediction side
                tokens=() if r.canonical_class == "Picture" else tuple(content_tokens(_rule_text(r), pred=True)),
                angle=r.r,
                caption=_grade(r) == "box" or r.canonical_class == "Picture",
            )
        )
    model = Page(
        elements=elements,
        word_boxes=word_boxes,
        word_tokens=[tuple(t) for t in word_tokens],
        token_text=token_text,
        pred_boxes=np.array(
            [[b.bbox_xyxy[0] * pw, b.bbox_xyxy[1] * ph, b.bbox_xyxy[2] * pw, b.bbox_xyxy[3] * ph] for b in blocks],
            dtype=float,
        ).reshape(-1, 4),
        pred_tokens=[content_tokens(b.text, pred=True) for b in blocks],
        pred_units=[b.order_index for b in blocks],
        pred_angles=[layout_angle(b) for b in blocks],
        skip_boxes=np.array([_xyxy(r, pw, ph) for r in shields], dtype=float).reshape(-1, 4),
        skip_angles=[r.r for r in shields],
        page=page,
        chain_tolerance=CHAIN_OVERLAP_POINTS if page_frame else 0.0,
    )
    return model, list(word_boxes[:, 3] - word_boxes[:, 1])


def _join_hyphens(
    words: list[tuple[LayoutTestRule, LayoutTestRule]],
    word_tokens: list[list[int]],
    token_text: list[str],
    word_boxes: np.ndarray,
    by_id: dict[str, LayoutTestRule],
    pw: float,
    ph: float,
    page_frame: bool,
    ro_key: Callable[[LayoutTestRule], tuple[bool, int, float, float]],
) -> None:
    """Join the last word of each line with the first word of the next line across a hyphen."""
    lines: dict[int, tuple[LayoutTestRule, list[int]]] = {}
    for n, (r, _) in enumerate(words):
        parent = by_id.get(r.parent_test_id or "")
        line = parent if parent is not None and parent.granularity == "line" else r
        lines.setdefault(id(line), (line, []))[1].append(n)
    ordered = sorted(lines.values(), key=lambda item: ro_key(item[0]))
    for (left_line, left_words), (right_line, right_words) in zip(ordered, ordered[1:], strict=False):
        lb, rb = _xyxy(left_line, pw, ph), _xyxy(right_line, pw, ph)
        max_h = max(lb[3] - lb[1], rb[3] - rb[1])
        limit = HYPHEN_GAP_LINE_HEIGHTS * max_h
        if page_frame:
            limit = max(HYPHEN_GAP_POINTS, limit)
        if not (rb[1] > lb[1] and rb[1] - lb[3] <= limit):
            continue
        if words[left_words[0]][1] is not words[right_words[0]][1]:
            continue  # a hyphen never joins words of two regions
        left = max(left_words, key=lambda n: (ro_key(words[n][0])[:2], word_boxes[n, 0]))
        right = min(right_words, key=lambda n: (ro_key(words[n][0])[:2], word_boxes[n, 0]))
        lt, rt = _rule_text(words[left][0]).strip(), _rule_text(words[right][0]).strip()
        stem = lt[:-1]
        if not (
            lt
            and lt[-1] in HYPHENS
            and stem
            and stem[-1].isalpha()
            and not stem[0].isdigit()
            and rt
            and rt[0].isalpha()
            and not any(is_cjk(ch) for ch in stem + rt)
        ):
            continue
        if not word_tokens[left] or not word_tokens[right]:
            continue
        joined = len(token_text)
        token_text.append(token_text[word_tokens[left][-1]] + token_text[word_tokens[right][0]])
        word_tokens[left][-1] = joined
        word_tokens[right][0] = joined


def _keep_content_tokens(
    words: list[tuple[LayoutTestRule, LayoutTestRule]], word_tokens: list[list[int]], token_text: list[str]
) -> None:
    """Drop word tokens absent from the region's normalized content, which is the text scored.

    OCR text of math is noise (``$...$`` is stripped from content). A region with content None keeps every token.
    """
    budgets: dict[int, Counter[str]] = {}
    kept: set[int] = set()
    for n, (_, region) in enumerate(words):
        if region.content is None:
            continue
        if id(region) not in budgets:
            text = _rule_text(region)  # pred=True also joins "infor- mation"
            budgets[id(region)] = Counter(content_tokens(text)) | Counter(content_tokens(text, pred=True))
        budget = budgets[id(region)]
        ids = []
        for t in word_tokens[n]:
            if t in kept or budget[token_text[t]] > 0:
                if t not in kept:
                    budget[token_text[t]] -= 1
                    kept.add(t)
                ids.append(t)
        word_tokens[n] = ids


def compute_layout_grounding_metrics(
    layout_rules: list[LayoutTestRule],
    read_blocks: Callable[[int], list[PredBlock]],
    page_sizes: Mapping[int, tuple[float, float]] | None = None,
) -> list[MetricValue]:
    """Return ``layout_grounding_pass_rate`` for one document.

    ``page_sizes`` maps a 1-based page number to its (width, height) in points.

    Returns nothing when no rule carries content or the denominator is zero.
    """
    if all(r.content is None for r in layout_rules):
        return []
    pages = sorted({r.page for r in layout_rules if _is_region(r)})
    built = []
    heights: list[float] = []
    for page in pages:
        model, page_heights = _build_page(
            [r for r in layout_rules if r.page == page], read_blocks(page), page, (page_sizes or {}).get(page)
        )
        built.append(model)
        heights.extend(page_heights)
    h_min = min(heights) if heights else 0.0
    weighted = denom = 0.0
    element_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for page, model in zip(pages, built, strict=True):
        result = score_page(model, h_min)
        pw, ph = (page_sizes or {}).get(page, (1.0, 1.0))
        unit = float(np.sqrt(pw * ph))
        weighted += result.weighted_score / unit
        denom += (result.weight + result.extra) / unit
        element_rows.extend(result.element_rows)
        prediction_rows.extend(result.prediction_rows)
    if denom <= 0:
        return []
    return [
        MetricValue(
            metric_name=METRIC_NAME,
            value=weighted / denom,
            metadata={
                "passed": float(weighted),
                "total": float(denom),
                "numerator_label": "score",
                "element_results": element_rows,
                "prediction_results": prediction_rows,
            },
        )
    ]


def source_page_sizes(file_path: Path) -> dict[int, tuple[float, float]]:
    """Return 1-based page sizes in PDF points; empty when unreadable, which scores on the unit square."""
    try:
        pdf = pdfium.PdfDocument(file_path)
    except (OSError, pdfium.PdfiumError):
        return {}
    try:
        return {n: pdf[n - 1].get_size() for n in range(1, len(pdf) + 1)}
    finally:
        pdf.close()
