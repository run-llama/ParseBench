"""Projection helpers that map unified layout predictions to evaluation labels."""

from __future__ import annotations

from typing import Any, Literal

from parse_bench.evaluation.layout_adapters.base import normalize_bbox_xyxy
from parse_bench.evaluation.layout_label_mappers.registry import (
    build_mapping_context,
    resolve_layout_label_mapper,
)
from parse_bench.layout_label_mapping import (
    map_canonical_label_to_target_ontology,
)
from parse_bench.schemas.layout_detection_output import LayoutOutput
from parse_bench.schemas.layout_ontology import CANONICAL_TO_CORE, CanonicalLabel
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR
from parse_bench.schemas.pipeline_io import InferenceResult


def _segment_to_xyxy(item: LayoutItemIR) -> list[float] | None:
    """Extract a pixel xyxy bbox from a ``LayoutItemIR``.

    Prefers ``layout_segments[0]`` (primary segment) and falls back to
    ``bbox`` if no segments are present. Returns ``None`` if neither is
    usable — such items are skipped by the caller.
    """
    seg: LayoutSegmentIR | None = None
    if item.layout_segments:
        seg = item.layout_segments[0]
    elif item.bbox is not None:
        seg = item.bbox
    if seg is None:
        return None
    return [seg.x, seg.y, seg.x + seg.w, seg.y + seg.h]


def project_layout_predictions(
    inference_result: InferenceResult,
    layout_output: LayoutOutput,
    *,
    evaluation_view: Literal["core", "canonical"] = "core",
    target_ontology: str = "basic",
    page_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Project unified layout predictions to evaluator-ready class labels.

    Dual-read during the ParseOutput-schema migration: prefers
    ``layout_output.layout_pages[*].items[*]`` (target shape) when any
    page has items, otherwise falls back to the legacy flat
    ``layout_output.predictions`` list. Emits the same projected dict
    shape in both branches so downstream evaluator code is unaffected.
    """
    if layout_output.image_width <= 0 or layout_output.image_height <= 0:
        return []

    projected: list[dict[str, Any]] = []

    # New path: iterate per-page items. Migrated providers emit canonical
    # strings into ``item.type``; we coerce directly to ``CanonicalLabel``
    # without the legacy mapper chain (canonical-is-canonical by construction).
    if any(page.items for page in layout_output.layout_pages):
        for page in layout_output.layout_pages:
            if page_filter is not None and page.page_number != page_filter:
                continue
            for item in page.items:
                bbox_xyxy = _segment_to_xyxy(item)
                if bbox_xyxy is None:
                    continue
                try:
                    canonical = CanonicalLabel(item.type)
                except ValueError:
                    # Unknown label: skip (consistent with legacy mapper
                    # behavior when ``to_canonical`` returns None).
                    continue

                label_for_view: CanonicalLabel = canonical
                if evaluation_view == "core":
                    core_class = CANONICAL_TO_CORE.get(canonical)
                    if core_class is None:
                        continue
                    label_for_view = core_class

                class_name = map_canonical_label_to_target_ontology(label_for_view, target_ontology)
                # Downstream metric code (classification_utils, layoutdet
                # evaluator) assumes ``score`` is always a float — it calls
                # ``float(p["score"])`` and stuffs the list into
                # ``np.array(..., dtype=float)``. ``LayoutItemIR.score`` is
                # ``float | None``, so coerce ``None`` to 0.0 here rather
                # than letting it leak downstream and crash or silently
                # become ``nan``. Parse-origin items (which legitimately
                # have no detector confidence) sort last in score-ordered
                # matching, which is the right behavior.
                score = item.score if item.score is not None else 0.0
                projected.append(
                    {
                        "bbox": normalize_bbox_xyxy(
                            bbox_xyxy,
                            width=layout_output.image_width,
                            height=layout_output.image_height,
                        ),
                        "class_name": class_name,
                        "score": score,
                        "page": page.page_number,
                        "order_index": None,
                        "attributes": dict(item.attributes) if item.attributes else {},
                    }
                )
        return projected

    # Legacy path: flat predictions list. Unchanged from pre-migration behavior.
    context = build_mapping_context(inference_result, layout_output)
    mapper = resolve_layout_label_mapper(context)

    for prediction in layout_output.predictions:
        if page_filter is not None and prediction.page != page_filter:
            continue
        if not mapper.should_include_prediction(prediction, context):
            continue

        canonical = mapper.to_canonical(prediction.label, prediction, context)
        label_for_view = canonical
        if evaluation_view == "core":
            core_class = CANONICAL_TO_CORE.get(canonical)
            if core_class is None:
                continue
            label_for_view = core_class

        class_name = mapper.to_target_ontology(label_for_view, target_ontology)
        raw_order_index = prediction.provider_metadata.get("order_index")
        order_index = raw_order_index if isinstance(raw_order_index, int) else None
        # Copy attributes so downstream consumers can mutate freely without
        # leaking into the original prediction.
        attributes = dict(prediction.attributes) if prediction.attributes else {}
        # Downstream metric code (classification_utils, layoutdet evaluator)
        # assumes ``score`` is always a float. Coerce a missing score to 0.0
        # here rather than letting ``None`` leak downstream and silently
        # become ``nan`` in score-sorted matching.
        score = prediction.score if prediction.score is not None else 0.0
        projected.append(
            {
                "bbox": normalize_bbox_xyxy(
                    prediction.bbox,
                    width=layout_output.image_width,
                    height=layout_output.image_height,
                ),
                "class_name": class_name,
                "score": score,
                "page": prediction.page,
                "order_index": order_index,
                "attributes": attributes,
            }
        )

    return projected
