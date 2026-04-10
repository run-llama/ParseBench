"""Core attribution metrics: LAP, LAR, AF1.

LAP (Local Attribution Precision): For each predicted block, checks whether
its text tokens are found in the GT elements that spatially overlap with it.
Penalizes hallucinated content or text copied from another region.

LAR (Local Attribution Recall): For each GT element, checks whether its
text tokens are recovered by predicted blocks that spatially overlap with it.
Penalizes missing content.

AF1 (Attribution F1): Harmonic mean of LAP and LAR.

All metrics use bidirectional overlap (max of IoA in both directions) for
spatial matching, which provides both merge and split tolerance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from parse_bench.evaluation.metrics.attribution.geometry import (
    coco_to_xyxy,
    compute_overlap_matrix,
    normalize_bbox_to_unit,
)
from parse_bench.evaluation.metrics.attribution.text_utils import (
    extract_text_from_html,
    normalize_attribution_text,
    strip_leading_markdown_image,
    tokenize,
)
from parse_bench.schemas.layout_ontology import BasicLabel, CanonicalLabel

IMAGE_LABELS = frozenset(
    {
        CanonicalLabel.PICTURE.value.lower(),
        "image",
    }
)

IMAGE_ITEM_TYPE = "image"


@dataclass
class GTElement:
    """A ground truth layout element."""

    bbox_coco: list[float]  # normalized [x, y, w, h] in [0,1]
    bbox_xyxy: list[float]  # normalized [x1, y1, x2, y2] in [0,1]
    canonical_class: str
    text: str  # raw text
    normalized_text: str  # normalized text
    tokens: list[str]  # tokenized
    ro_index: int  # reading order index
    content_type: str  # "text" or "table"
    html: str | None = None  # original HTML for tables
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class PredBlock:
    """A predicted output block with bounding box(es)."""

    bbox_xyxy: list[float]  # normalized [x1, y1, x2, y2] in [0,1]
    block_type: str  # "text", "heading", "table"
    label: str  # layout label from the service
    text: str  # raw text/value
    normalized_text: str  # normalized
    tokens: list[str]  # tokenized
    order_index: int  # position in output list


@dataclass
class AttributionResult:
    """Results from attribution evaluation for a single page."""

    # Core metrics
    lap: float = 0.0  # Local Attribution Precision
    lar: float = 0.0  # Local Attribution Recall
    af1: float = 0.0  # Attribution F1

    # Reading order
    order_adjacent_accuracy: float = 0.0
    order_pairwise_accuracy: float = 0.0

    # Per-class LAR breakdown
    per_class_lar: dict[str, float] = field(default_factory=dict)
    per_class_lap: dict[str, float] = field(default_factory=dict)
    per_class_af1: dict[str, float] = field(default_factory=dict)

    # Grounding accuracy (element-level pass/fail)
    grounding_accuracy: float = 0.0  # fraction of GT elements correctly attributed
    grounded_count: int = 0  # number of GT elements that passed
    total_count: int = 0  # total GT elements evaluated
    per_class_grounding: dict[str, float] = field(default_factory=dict)  # accuracy per class
    per_class_grounded_count: dict[str, int] = field(default_factory=dict)  # pass count per class
    per_class_total_count: dict[str, int] = field(default_factory=dict)  # total count per class

    # Diagnostics
    num_gt_elements: int = 0
    num_pred_blocks: int = 0
    num_gt_tokens: int = 0
    num_pred_tokens: int = 0
    unmatched_gt_elements: int = 0  # GT elements with no overlapping pred
    unmatched_pred_blocks: int = 0  # pred blocks with no overlapping GT


def _multiset_intersection_size(a: list[str], b: list[str]) -> int:
    """Compute the size of the multiset intersection of two token lists.

    For each token, min(count_in_a, count_in_b) contributes to the intersection.

    :param a: First token list
    :param b: Second token list
    :return: Size of multiset intersection
    """
    counter_a = Counter(a)
    counter_b = Counter(b)
    return sum((counter_a & counter_b).values())


def normalize_layout_attributes(attributes: dict[str, str | bool] | None) -> dict[str, str]:
    """Normalize known layout-annotation attribute aliases/typos."""
    normalized = {str(key): str(value) for key, value in (attributes or {}).items()}

    if normalized.get("content", "").strip().lower() == "explicit":
        normalized.pop("content", None)
        normalized["explicit"] = "true"
    if "ignore_head" in normalized:
        ignore_head_value = normalized.pop("ignore_head")
        if is_truthy(ignore_head_value):
            normalized["ignore_thead"] = "true"

    return normalized


def is_truthy(value: object | None) -> bool:
    """Return True when a value matches the accepted truthy spellings."""
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def layout_element_is_formula(canonical_class: str | None, attributes: dict[str, str] | None) -> bool:
    """Return True for legacy Formula labels or collapsed formula attributes."""
    canonical_class_norm = str(canonical_class or "").strip().lower()
    text_role = str((attributes or {}).get("text_role", "")).strip().lower()
    formula_labels = {
        CanonicalLabel.FORMULA.value.lower(),
        BasicLabel.FORMULA.value.lower(),
    }
    return canonical_class_norm in formula_labels or text_role == "formula"


def layout_element_uses_image_alt_text_normalization(canonical_class: str | None, content_type: str | None) -> bool:
    """Return True for GT elements that should normalize image markup to alt text."""
    canonical_class_norm = str(canonical_class or "").strip().lower()
    content_type_norm = str(content_type or "").strip().lower()
    page_furniture_labels = {
        CanonicalLabel.PAGE_HEADER.value.lower(),
        CanonicalLabel.PAGE_FOOTER.value.lower(),
        BasicLabel.PAGE_HEADER.value.lower(),
        BasicLabel.PAGE_FOOTER.value.lower(),
    }
    return (
        canonical_class_norm in IMAGE_LABELS
        or content_type_norm in IMAGE_LABELS
        or canonical_class_norm in page_furniture_labels
    )


def pred_block_is_image_like(block_type: str | None, label: str | None) -> bool:
    """Return True when a predicted block should be normalized as an image."""
    block_type_norm = str(block_type or "").strip().lower()
    label_norm = str(label or "").strip().lower()
    return block_type_norm == IMAGE_ITEM_TYPE or label_norm in IMAGE_LABELS


def gt_element_is_ignored(element: GTElement) -> bool:
    """Return True when the GT element should be fully excluded from grading."""
    return is_truthy(element.attributes.get("ignore"))


def gt_element_is_explicit(element: GTElement) -> bool:
    """Return True when attribution should use recall-only semantics."""
    return is_truthy(element.attributes.get("explicit"))


def gt_element_skips_attribution(element: GTElement) -> bool:
    """Return True when attribution should be skipped for this GT element."""
    return layout_element_is_formula(element.canonical_class, element.attributes) or is_truthy(
        element.attributes.get("caption")
    )


def _filter_gt_elements_for_attribution(gt_elements: list[GTElement]) -> list[GTElement]:
    """Remove GT elements that should not participate in attribution grading."""
    return [gt for gt in gt_elements if not gt_element_is_ignored(gt) and not gt_element_skips_attribution(gt)]


def _filter_pred_blocks_for_lap(
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_threshold: float,
) -> list[PredBlock]:
    """Exclude predictions that overlap only explicit GT elements from LAP.

    Explicit GT elements are recall-only. Predictions that overlap only explicit GT
    elements should not be penalized for extra tokens in precision-oriented metrics.
    Predictions with no GT overlap remain in-scope so hallucinated text is still penalized.
    """
    if not gt_elements or not pred_blocks:
        return pred_blocks

    gt_boxes = np.array([g.bbox_xyxy for g in gt_elements])
    pred_boxes = np.array([p.bbox_xyxy for p in pred_blocks])
    ioa_matrix = compute_overlap_matrix(gt_boxes, pred_boxes)
    explicit_mask = [gt_element_is_explicit(gt) for gt in gt_elements]

    filtered: list[PredBlock] = []
    for pred_idx, pred in enumerate(pred_blocks):
        overlapping_gt_indices = np.where(ioa_matrix[:, pred_idx] >= ioa_threshold)[0]
        if len(overlapping_gt_indices) == 0:
            filtered.append(pred)
            continue
        if all(explicit_mask[int(gt_idx)] for gt_idx in overlapping_gt_indices):
            continue
        filtered.append(pred)

    return filtered


def parse_gt_elements(test_rules: list[dict]) -> list[GTElement]:
    """Parse ground truth test rules into GTElement objects.

    :param test_rules: List of test rule dicts from .test.json
    :return: List of GTElement objects
    """
    elements = []
    for rule in test_rules:
        if rule.get("type") != "layout":
            continue

        attributes = normalize_layout_attributes(rule.get("attributes"))
        content = rule.get("content") or {}
        content_type = content.get("type", "text")

        if content_type == "table":
            html = content.get("html", "")
            raw_text = extract_text_from_html(
                html,
                ignore_thead=is_truthy(attributes.get("ignore_thead")),
            )
        else:
            html = None
            raw_text = content.get("text", "")

        normalized = normalize_attribution_text(
            raw_text,
            strip_image_markup=layout_element_uses_image_alt_text_normalization(
                rule.get("canonical_class"),
                content_type,
            ),
        )
        tokens = tokenize(normalized)

        bbox_coco = rule["bbox"]
        bbox_xyxy = coco_to_xyxy(bbox_coco)

        elements.append(
            GTElement(
                bbox_coco=bbox_coco,
                bbox_xyxy=bbox_xyxy,
                canonical_class=rule.get("canonical_class", "Unknown"),
                attributes=attributes,
                text=raw_text,
                normalized_text=normalized,
                tokens=tokens,
                ro_index=rule.get("ro_index", 0),
                content_type=content_type,
                html=html,
            )
        )

    return elements


def parse_pred_blocks(
    items: list[dict],
    page_md: str,
    page_width: float,
    page_height: float,
) -> list[PredBlock]:
    """Parse predicted output items into PredBlock objects.

    :param items: List of item dicts from result.json pages[].items
    :param page_md: Full page markdown (used for table content extraction)
    :param page_width: Page width in pixels
    :param page_height: Page height in pixels
    :return: List of PredBlock objects
    """
    blocks = []
    # Split page markdown into table HTML sections for table content matching
    table_htmls = _extract_table_htmls(page_md)
    table_html_idx = 0

    for idx, item in enumerate(items):
        bbox_dict = item.get("bBox")
        if not bbox_dict:
            continue

        # Normalize pixel bbox to [0,1]
        layout_aware = item.get("layoutAwareBbox") or []
        primary_layout_label = None
        if layout_aware:
            primary_layout_label = layout_aware[0].get("label")
        if layout_aware:
            for segment in layout_aware:
                bbox_norm = normalize_bbox_to_unit(segment, page_width, page_height)
                bbox_xyxy = coco_to_xyxy(bbox_norm)
                block_type = item.get("type", "text")
                label = segment.get("label") or primary_layout_label or bbox_dict.get("label", "unknown")

                # Prefer "md" over "value" because startIndex/endIndex on bbox segments
                # are computed relative to the md field (which includes markdown formatting).
                # Using "value" (formatting stripped) would cause index-based slicing to
                # extract the wrong substring.
                item_text = item.get("md") or item.get("value") or ""
                start = segment.get("startIndex")
                end = segment.get("endIndex")
                if isinstance(start, int) and isinstance(end, int) and end >= start:
                    raw_text = item_text[start : end + 1]
                else:
                    raw_text = item_text

                # For non-image text items, strip a leading inline image reference
                # (e.g. "![icon](image.jpg) text") so attribution compares only the
                # text content. Image-type blocks keep their full markdown so alt-text
                # extraction via strip_image_markup still works correctly.
                is_image_block = pred_block_is_image_like(block_type, label)
                if not is_image_block:
                    raw_text = strip_leading_markdown_image(raw_text)

                normalized = normalize_attribution_text(
                    raw_text,
                    strip_image_markup=is_image_block,
                )
                tokens = tokenize(normalized)

                blocks.append(
                    PredBlock(
                        bbox_xyxy=bbox_xyxy,
                        block_type=block_type,
                        label=label,
                        text=raw_text,
                        normalized_text=normalized,
                        tokens=tokens,
                        order_index=idx,
                    )
                )
            continue

        bbox_norm = normalize_bbox_to_unit(bbox_dict, page_width, page_height)
        bbox_xyxy = coco_to_xyxy(bbox_norm)

        block_type = item.get("type", "text")
        label = primary_layout_label or bbox_dict.get("label", "unknown")

        # Get text content
        if block_type == "table":
            # Tables have value=null, content is in page markdown
            if table_html_idx < len(table_htmls):
                raw_text = extract_text_from_html(table_htmls[table_html_idx])
                table_html_idx += 1
            else:
                raw_text = ""
        else:
            raw_text = item.get("value", "") or ""

        normalized = normalize_attribution_text(
            raw_text,
            strip_image_markup=pred_block_is_image_like(block_type, label),
        )
        tokens = tokenize(normalized)

        blocks.append(
            PredBlock(
                bbox_xyxy=bbox_xyxy,
                block_type=block_type,
                label=label,
                text=raw_text,
                normalized_text=normalized,
                tokens=tokens,
                order_index=idx,
            )
        )

    return blocks


def _extract_table_htmls(md: str) -> list[str]:
    """Extract all <table>...</table> sections from markdown.

    :param md: Page markdown text
    :return: List of HTML table strings
    """
    import re

    pattern = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)
    return pattern.findall(md)


def compute_lap(
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_threshold: float = 0.3,
) -> tuple[float, dict[str, float], int]:
    """Compute Local Attribution Precision (LAP).

    For each predicted block, find GT elements whose boxes overlap (IoA >= threshold),
    then check what fraction of predicted tokens appear in those GT elements' tokens.

    :param gt_elements: Ground truth elements
    :param pred_blocks: Predicted blocks
    :param ioa_threshold: IoA threshold for considering overlap
    :return: (LAP score, per-pred-label breakdown, total pred tokens)
    """
    if not pred_blocks:
        return 1.0, {}, 0

    # Build IoA matrix: shape (num_gt, num_pred)
    gt_boxes = np.array([g.bbox_xyxy for g in gt_elements]) if gt_elements else np.zeros((0, 4))
    pred_boxes = np.array([p.bbox_xyxy for p in pred_blocks])
    ioa_matrix = compute_overlap_matrix(gt_boxes, pred_boxes)  # (N_gt, N_pred)

    total_weighted_prec = 0.0
    total_tokens = 0
    label_numerator: dict[str, float] = {}
    label_denominator: dict[str, int] = {}

    for j, pred in enumerate(pred_blocks):
        if not pred.tokens:
            continue

        # Find GT elements overlapping this pred block
        if gt_elements:
            overlapping_gt_indices = np.where(ioa_matrix[:, j] >= ioa_threshold)[0]
        else:
            overlapping_gt_indices = []  # type: ignore[assignment]

        # Union of GT tokens from overlapping elements
        gt_token_union: list[str] = []
        for gi in overlapping_gt_indices:
            gt_token_union.extend(gt_elements[gi].tokens)

        # Token precision: what fraction of pred tokens appear in GT union
        matched = _multiset_intersection_size(pred.tokens, gt_token_union)
        prec = matched / len(pred.tokens) if pred.tokens else 1.0

        weight = len(pred.tokens)
        total_weighted_prec += weight * prec
        total_tokens += weight

        # Per-label accumulation
        lbl = pred.label
        label_numerator[lbl] = label_numerator.get(lbl, 0.0) + weight * prec
        label_denominator[lbl] = label_denominator.get(lbl, 0) + weight

    lap = total_weighted_prec / total_tokens if total_tokens > 0 else 1.0

    per_label = {}
    for lbl in label_numerator:
        per_label[lbl] = label_numerator[lbl] / label_denominator[lbl] if label_denominator[lbl] > 0 else 0.0

    return lap, per_label, total_tokens


def compute_per_class_lap_by_gt(
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_threshold: float = 0.3,
) -> dict[str, float]:
    """Compute per-class LAP (precision) by GT class.

    Each predicted block contributes token-precision to every overlapping GT class
    (IoA >= threshold), weighted by predicted token count.

    :param gt_elements: Ground truth elements
    :param pred_blocks: Predicted blocks
    :param ioa_threshold: IoA threshold for considering overlap
    :return: Dict mapping GT class -> LAP
    """
    if not pred_blocks or not gt_elements:
        return {}

    gt_boxes = np.array([g.bbox_xyxy for g in gt_elements])
    pred_boxes = np.array([p.bbox_xyxy for p in pred_blocks])
    ioa_matrix = compute_overlap_matrix(gt_boxes, pred_boxes)

    class_numerator: dict[str, float] = {}
    class_denominator: dict[str, int] = {}

    for j, pred in enumerate(pred_blocks):
        if not pred.tokens:
            continue

        overlapping_gt_indices = np.where(ioa_matrix[:, j] >= ioa_threshold)[0]
        if len(overlapping_gt_indices) == 0:
            continue

        tokens_by_class: dict[str, list[str]] = {}
        for gi in overlapping_gt_indices:
            cls = gt_elements[gi].canonical_class
            tokens_by_class.setdefault(cls, []).extend(gt_elements[gi].tokens)

        weight = len(pred.tokens)
        for cls, gt_tokens in tokens_by_class.items():
            matched = _multiset_intersection_size(pred.tokens, gt_tokens)
            prec = matched / len(pred.tokens) if pred.tokens else 1.0
            class_numerator[cls] = class_numerator.get(cls, 0.0) + weight * prec
            class_denominator[cls] = class_denominator.get(cls, 0) + weight

    per_class = {}
    for cls in class_numerator:
        per_class[cls] = class_numerator[cls] / class_denominator[cls] if class_denominator[cls] > 0 else 0.0

    return per_class


def compute_lar(
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_threshold: float = 0.3,
) -> tuple[float, dict[str, float], int]:
    """Compute Local Attribution Recall (LAR).

    For each GT element, find predicted blocks whose boxes overlap (IoA >= threshold),
    then check what fraction of GT tokens are recovered by those predicted blocks' tokens.

    :param gt_elements: Ground truth elements
    :param pred_blocks: Predicted blocks
    :param ioa_threshold: IoA threshold for considering overlap
    :return: (LAR score, per-class breakdown, total GT tokens)
    """
    if not gt_elements:
        return 1.0, {}, 0

    gt_boxes = np.array([g.bbox_xyxy for g in gt_elements])
    pred_boxes = np.array([p.bbox_xyxy for p in pred_blocks]) if pred_blocks else np.zeros((0, 4))
    ioa_matrix = compute_overlap_matrix(gt_boxes, pred_boxes)  # (N_gt, N_pred)

    total_weighted_rec = 0.0
    total_tokens = 0
    class_numerator: dict[str, float] = {}
    class_denominator: dict[str, int] = {}

    for i, gt in enumerate(gt_elements):
        if not gt.tokens:
            continue

        # Find pred blocks overlapping this GT element
        if pred_blocks:
            overlapping_pred_indices = np.where(ioa_matrix[i, :] >= ioa_threshold)[0]
        else:
            overlapping_pred_indices = []  # type: ignore[assignment]

        # Union of pred tokens from overlapping blocks
        pred_token_union: list[str] = []
        for pj in overlapping_pred_indices:
            pred_token_union.extend(pred_blocks[pj].tokens)

        # Token recall: what fraction of GT tokens are in pred union
        matched = _multiset_intersection_size(gt.tokens, pred_token_union)
        rec = matched / len(gt.tokens) if gt.tokens else 1.0

        weight = len(gt.tokens)
        total_weighted_rec += weight * rec
        total_tokens += weight

        # Per-class accumulation
        cls = gt.canonical_class
        class_numerator[cls] = class_numerator.get(cls, 0.0) + weight * rec
        class_denominator[cls] = class_denominator.get(cls, 0) + weight

    lar = total_weighted_rec / total_tokens if total_tokens > 0 else 1.0

    per_class = {}
    for cls in class_numerator:
        per_class[cls] = class_numerator[cls] / class_denominator[cls] if class_denominator[cls] > 0 else 0.0

    return lar, per_class, total_tokens


def compute_af1(lap: float, lar: float) -> float:
    """Compute Attribution F1 score (harmonic mean of LAP and LAR).

    :param lap: Local Attribution Precision
    :param lar: Local Attribution Recall
    :return: AF1 score
    """
    if lap + lar <= 0:
        return 0.0
    return 2.0 * lap * lar / (lap + lar)


def compute_grounding_accuracy(
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_threshold: float = 0.3,
    recall_threshold: float = 0.8,
) -> tuple[float, int, int, dict[str, float], dict[str, int], dict[str, int]]:
    """Compute grounding accuracy: percentage of GT elements correctly attributed.

    A GT element is "correctly grounded" when overlapping predictions recover
    at least `recall_threshold` of its tokens. This is a strict, binary,
    element-level metric designed for executive reporting.

    :param gt_elements: Ground truth elements
    :param pred_blocks: Predicted blocks
    :param ioa_threshold: Minimum spatial overlap to match GT to pred
    :param recall_threshold: Minimum token recall to count as correctly grounded
    :return: (accuracy, grounded_count, total_count, per_class_accuracy,
              per_class_pass_count, per_class_total_count)
    """
    eligible_gt = _filter_gt_elements_for_attribution(gt_elements)
    if not eligible_gt:
        return 1.0, 0, 0, {}, {}, {}

    gt_boxes = np.array([g.bbox_xyxy for g in eligible_gt])
    pred_boxes = np.array([p.bbox_xyxy for p in pred_blocks]) if pred_blocks else np.zeros((0, 4))
    overlap = compute_overlap_matrix(gt_boxes, pred_boxes)

    # Per-class counters
    class_pass: dict[str, int] = {}
    class_total: dict[str, int] = {}

    grounded = 0
    total = 0

    for i, gt in enumerate(eligible_gt):
        # Skip elements with no meaningful text content
        if not gt.tokens:
            continue

        total += 1
        cls = gt.canonical_class
        class_total[cls] = class_total.get(cls, 0) + 1

        # Find overlapping predictions
        if pred_blocks:
            overlapping = np.where(overlap[i, :] >= ioa_threshold)[0]
        else:
            overlapping = []  # type: ignore[assignment]

        # Pool tokens from all overlapping predictions
        pred_token_union: list[str] = []
        for j in overlapping:
            pred_token_union.extend(pred_blocks[j].tokens)

        # Check recall
        matched = _multiset_intersection_size(gt.tokens, pred_token_union)
        recall = matched / len(gt.tokens)

        if recall >= recall_threshold:
            grounded += 1
            class_pass[cls] = class_pass.get(cls, 0) + 1

    accuracy = grounded / total if total > 0 else 1.0

    per_class_acc = {}
    for cls in class_total:
        per_class_acc[cls] = class_pass.get(cls, 0) / class_total[cls]

    return accuracy, grounded, total, per_class_acc, class_pass, class_total


def compute_reading_order(
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_threshold: float = 0.3,
) -> tuple[float, float]:
    """Compute reading order accuracy metrics.

    Maps each GT element to the earliest predicted block index that overlaps it,
    then checks whether the predicted order preserves GT reading order.

    :param gt_elements: GT elements sorted by ro_index
    :param pred_blocks: Predicted blocks
    :param ioa_threshold: IoA threshold for spatial matching
    :return: (adjacent_order_accuracy, pairwise_order_accuracy)
    """
    eligible_gt = _filter_gt_elements_for_attribution(gt_elements)
    if len(eligible_gt) < 2 or not pred_blocks:
        return 1.0, 1.0

    # Sort GT elements by reading order
    sorted_gt = sorted(eligible_gt, key=lambda g: g.ro_index)

    gt_boxes = np.array([g.bbox_xyxy for g in sorted_gt])
    pred_boxes = np.array([p.bbox_xyxy for p in pred_blocks])
    ioa_matrix = compute_overlap_matrix(gt_boxes, pred_boxes)

    # Map each GT element to earliest pred index that covers it
    positions: list[int | None] = []
    for i in range(len(sorted_gt)):
        overlapping = np.where(ioa_matrix[i, :] >= ioa_threshold)[0]
        if len(overlapping) > 0:
            # Use the minimum order_index among overlapping preds
            min_order = min(pred_blocks[j].order_index for j in overlapping)
            positions.append(min_order)
        else:
            positions.append(None)

    # Filter out unmapped GT elements
    mapped = [(i, pos) for i, pos in enumerate(positions) if pos is not None]

    if len(mapped) < 2:
        return 1.0, 1.0

    # Adjacent order accuracy
    adj_correct = 0
    adj_total = 0
    for k in range(len(mapped) - 1):
        _, pos_curr = mapped[k]
        _, pos_next = mapped[k + 1]
        adj_total += 1
        if pos_curr <= pos_next:
            adj_correct += 1

    adj_acc = adj_correct / adj_total if adj_total > 0 else 1.0

    # Pairwise order accuracy
    pair_correct = 0
    pair_total = 0
    for a in range(len(mapped)):
        for b in range(a + 1, len(mapped)):
            _, pos_a = mapped[a]
            _, pos_b = mapped[b]
            pair_total += 1
            if pos_a <= pos_b:
                pair_correct += 1

    pair_acc = pair_correct / pair_total if pair_total > 0 else 1.0

    return adj_acc, pair_acc


def compute_attribution_metrics(
    gt_elements: list[GTElement],
    pred_blocks: list[PredBlock],
    ioa_threshold: float = 0.3,
) -> AttributionResult:
    """Compute all attribution metrics for a single page.

    :param gt_elements: Ground truth elements
    :param pred_blocks: Predicted blocks
    :param ioa_threshold: IoA threshold for spatial matching
    :return: AttributionResult with all metrics
    """
    attribution_gt = _filter_gt_elements_for_attribution(gt_elements)
    lap_gt = [gt for gt in attribution_gt if not gt_element_is_explicit(gt)]
    lap_pred_blocks = _filter_pred_blocks_for_lap(attribution_gt, pred_blocks, ioa_threshold)

    lap, _per_label_lap, num_pred_tokens = compute_lap(lap_gt, lap_pred_blocks, ioa_threshold)
    lar, per_class_lar, num_gt_tokens = compute_lar(attribution_gt, pred_blocks, ioa_threshold)
    af1 = compute_af1(lap, lar)

    adj_order, pair_order = compute_reading_order(attribution_gt, pred_blocks, ioa_threshold)

    ga, ga_pass, ga_total, per_class_ga, per_class_ga_pass, per_class_ga_total = compute_grounding_accuracy(
        attribution_gt, pred_blocks, ioa_threshold
    )

    # Compute per-class LAP (by GT class) and AF1
    per_class_lap = compute_per_class_lap_by_gt(lap_gt, lap_pred_blocks, ioa_threshold)
    all_classes = set(per_class_lar.keys()) | set(per_class_lap.keys())
    per_class_af1 = {}
    for cls in all_classes:
        lap_cls = per_class_lap.get(cls, 0.0)
        lar_cls = per_class_lar.get(cls, 0.0)
        per_class_af1[cls] = compute_af1(lap_cls, lar_cls)

    # Count unmatched elements
    gt_boxes = np.array([g.bbox_xyxy for g in attribution_gt]) if attribution_gt else np.zeros((0, 4))
    pred_boxes = np.array([p.bbox_xyxy for p in pred_blocks]) if pred_blocks else np.zeros((0, 4))

    unmatched_gt = 0
    unmatched_pred = 0
    if attribution_gt and pred_blocks:
        ioa_matrix = compute_overlap_matrix(gt_boxes, pred_boxes)
        for i in range(len(attribution_gt)):
            if not np.any(ioa_matrix[i, :] >= ioa_threshold):
                unmatched_gt += 1
        for j in range(len(pred_blocks)):
            if not np.any(ioa_matrix[:, j] >= ioa_threshold):
                unmatched_pred += 1
    else:
        unmatched_gt = len(attribution_gt)
        unmatched_pred = len(pred_blocks)

    return AttributionResult(
        lap=lap,
        lar=lar,
        af1=af1,
        order_adjacent_accuracy=adj_order,
        order_pairwise_accuracy=pair_order,
        per_class_lar=per_class_lar,
        per_class_lap=per_class_lap,
        per_class_af1=per_class_af1,
        grounding_accuracy=ga,
        grounded_count=ga_pass,
        total_count=ga_total,
        per_class_grounding=per_class_ga,
        per_class_grounded_count=per_class_ga_pass,
        per_class_total_count=per_class_ga_total,
        num_gt_elements=len(attribution_gt),
        num_pred_blocks=len(pred_blocks),
        num_gt_tokens=num_gt_tokens,
        num_pred_tokens=num_pred_tokens,
        unmatched_gt_elements=unmatched_gt,
        unmatched_pred_blocks=unmatched_pred,
    )
