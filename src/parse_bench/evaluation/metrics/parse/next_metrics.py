"""Per-document wiring of the extended scorers (``evaluation run --next``).

``compute_next_metrics(inference_result, test_case)`` returns every extended metric that applies to one
document, reading the same test-case fields as the scorers expect:

- ``text_next`` and ``formatting_v2``: the reference Markdown (``expected_markdown``, or a text rule's
  ``original_md``), a full-page decoration inventory (``metadata.decorations``) or, with neither, the
  formatting rules themselves (``heading_structure``, ``present_as_strikeout``, ``is_underline``, ...).
  Chart pages and table-only references are left to the chart and table scorers.
- ``forms_v2``: the form tree under ``metadata.forms.gt``, else the ``form_field`` rules.
- ``chart_next``: full reference tables under ``metadata.chart_tables``, else the ``chart_data_point`` rules.
- ``table_next``: the tables of ``expected_markdown`` with the test case's table flags
  (``max_top_title_rows``, ``allow_splitting_ambiguous_merged_tables``, ``trm_unsupported``,
  ``allow_merged_sub_rows``).
- ``form_kv_*``: the key-value list metric over ``metadata.forms`` (or the ``form_field`` rules).
- ``layout_grounding_pass_rate``: the layout rules (regions, plus their ``line`` and ``word`` boxes).

Nothing here changes the metrics the evaluators already emit; the runner appends these rows when asked.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import cache
from typing import Any

from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from parse_bench.evaluation.layout_adapters.base import LayoutAdapter, normalize_bbox_xyxy
from parse_bench.evaluation.metrics.attribution.core import PredBlock
from parse_bench.evaluation.metrics.parse.chart_next import chart_next_metrics, chart_tables_of
from parse_bench.evaluation.metrics.parse.form_kv_scoring.bench_metric import (
    compute_form_kv_metrics,
    forms_meta_from_rules,
)
from parse_bench.evaluation.metrics.parse.formatting_v2 import FULL as FULL_SCOPE
from parse_bench.evaluation.metrics.parse.formatting_v2 import (
    decoration_reference,
    formatting_v2_applies,
    formatting_v2_metrics,
    rules_reference,
)
from parse_bench.evaluation.metrics.parse.forms_v2 import forms_reference, forms_v2_applies, forms_v2_metrics
from parse_bench.evaluation.metrics.parse.layout_grounding import (
    compute_layout_grounding_metrics,
    source_page_sizes,
)
from parse_bench.evaluation.metrics.parse.table_next import table_next_metrics
from parse_bench.evaluation.metrics.parse.text_next import (
    output_hypothesis,
    reference_markdown,
    text_next_applies,
    text_next_metrics,
)
from parse_bench.schemas.evaluation import EvaluationResult, MetricValue
from parse_bench.schemas.parse_output import ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceResult
from parse_bench.test_cases.schema import LayoutDetectionTestCase, LayoutTestRule, ParseTestCase, TestCase

logger = logging.getLogger(__name__)

# Layout attribute spellings and the equivalent keys the layout scorer reads.
_LAYOUT_KEY_ALIASES = {
    ("grade", "none"): "ignore",
    ("grade", "box"): "caption",
    ("text_match", "recall"): "explicit",
    ("exclude_table_header", "true"): "ignore_thead",
}

# Provider labels of box-only picture elements (no text content), compared lower-cased.
_PICTURE_LABELS = frozenset({"picture", "figure", "image", "chart", "graphic", "logo"})


def compute_next_metrics(inference_result: InferenceResult, test_case: TestCase) -> list[MetricValue]:
    """Return the extended metrics of one document (empty when none applies)."""
    output = inference_result.output
    if not isinstance(output, ParseOutput) or not isinstance(test_case, ParseTestCase | LayoutDetectionTestCase):
        return []
    rules = list(test_case.test_rules or [])
    metadata = test_case.metadata if isinstance(test_case.metadata, dict) else {}
    layout_rules = [_with_layout_key_aliases(r) for r in rules if isinstance(r, LayoutTestRule)]
    region_rules = [r for r in layout_rules if r.granularity in (None, "region")]

    metrics: list[MetricValue] = []
    if isinstance(test_case, ParseTestCase):
        # A layout-only pipeline (layout boxes, no Markdown) is not scored on text.
        has_markdown = bool((output.markdown or "").strip())
        has_layout_content = any(len(p.items) > 0 for p in output.layout_pages or [])
        run_text = not (bool(region_rules) and has_layout_content and not has_markdown)
        metrics.extend(_document_metrics(inference_result, test_case, rules, metadata, run_text))
    if region_rules:
        metrics.extend(_layout_metrics(inference_result, test_case, layout_rules))
    return metrics


def _document_metrics(
    inference_result: InferenceResult,
    test_case: ParseTestCase,
    rules: list[Any],
    metadata: dict[str, Any],
    run_text: bool,
) -> list[MetricValue]:
    """Text, formatting, forms, charts, tables and form key-value rows of one Markdown document."""
    output = inference_result.output
    assert isinstance(output, ParseOutput)
    metrics: list[MetricValue] = []
    if run_text:
        reference = reference_markdown(test_case.expected_markdown, rules)
        if reference is not None and text_next_applies(reference, rules, metadata):
            metrics.extend(text_next_metrics(reference, output))

        # A full-page decoration inventory takes precedence over the reference Markdown and the rules.
        decorations = decoration_reference(metadata)
        fmt_md = reference_markdown(test_case.expected_markdown, rules)
        fmt_ref = None if fmt_md else rules_reference(rules)
        if fmt_md and text_next_applies(fmt_md, rules, metadata):
            fmt_ref = (fmt_md, FULL_SCOPE) if formatting_v2_applies(fmt_md) else None
        neutral: list[str] = []
        if decorations is not None:
            fmt_ref, neutral = (decorations[0], decorations[1]), decorations[2]
        if fmt_ref is not None:
            metrics.extend(formatting_v2_metrics(fmt_ref[0], output_hypothesis(output), fmt_ref[1], neutral))

    forms_ref = forms_reference(metadata, rules)
    if forms_ref is not None and forms_v2_applies(forms_ref):
        metrics.extend(forms_v2_metrics(forms_ref, output.markdown or ""))

    metrics.extend(chart_next_metrics(output.markdown or "", chart_tables_of(metadata), rules))

    if run_text:
        metrics.extend(
            table_next_metrics(
                test_case.expected_markdown,
                output.markdown,
                max_top_title_rows=test_case.max_top_title_rows,
                allow_splitting_ambiguous_merged_tables=test_case.allow_splitting_ambiguous_merged_tables,
                trm_unsupported=test_case.trm_unsupported,
                allow_merged_sub_rows=test_case.allow_merged_sub_rows,
            )
        )

    forms_meta = metadata.get("forms")
    if isinstance(forms_meta, dict):
        metrics.extend(compute_form_kv_metrics(inference_result, forms_meta))
    else:
        rules_meta = forms_meta_from_rules(rules)
        if rules_meta is not None:
            metrics.extend(compute_form_kv_metrics(inference_result, rules_meta))
    return metrics


def _layout_metrics(
    inference_result: InferenceResult, test_case: TestCase, layout_rules: list[LayoutTestRule]
) -> list[MetricValue]:
    """``layout_grounding_pass_rate`` of one document.

    A pipeline that returned no layout pages for this document scores 0 on it; the row is flagged
    ``no_predictions`` and scored on the source PDF's page sizes.
    """
    output = inference_result.output
    assert isinstance(output, ParseOutput)
    if output.layout_pages:
        return compute_layout_grounding_metrics(
            layout_rules,
            _attribution_block_reader(inference_result, test_case),
            _layout_page_dimensions(output.layout_pages),
        )
    return [
        metric.model_copy(update={"metadata": {**metric.metadata, "no_predictions": True}})
        for metric in compute_layout_grounding_metrics(
            layout_rules, lambda _page: [], source_page_sizes(test_case.file_path)
        )
    ]


def _with_layout_key_aliases(rule: LayoutTestRule) -> LayoutTestRule:
    """Add the equivalent key for each aliased layout attribute, so both spellings score the same."""
    attrs = rule.attributes
    extra = {
        key: "true"
        for (alias, value), key in _LAYOUT_KEY_ALIASES.items()
        if str(attrs.get(alias, "")).strip().lower() == value and key not in attrs
    }
    return rule.model_copy(update={"attributes": {**attrs, **extra}}) if extra else rule


def _layout_page_dimensions(layout_pages: list[ParseLayoutPageIR]) -> dict[int, tuple[float, float]]:
    """1-based page number -> (width, height) of every layout page that declares a positive size."""
    dimensions: dict[int, tuple[float, float]] = {}
    for page in layout_pages:
        if page.width is None or page.height is None or page.width <= 0 or page.height <= 0:
            continue
        dimensions[int(page.page_number)] = (float(page.width), float(page.height))
    return dimensions


def _attribution_block_reader(
    inference_result: InferenceResult, test_case: TestCase
) -> Callable[[int], list[PredBlock]]:
    """Return a 1-based page number -> prediction blocks reader; the layout output is built once."""

    @cache
    def adapter_output() -> tuple[LayoutAdapter, Any]:
        adapter = create_layout_adapter_for_result(inference_result)
        return adapter, adapter.to_layout_output(inference_result)

    def read(page_number: int) -> list[PredBlock]:
        adapter, layout_output = adapter_output()
        blocks = adapter.to_attribution_blocks(layout_output, page_number=page_number, test_case=test_case)
        return _with_picture_boxes(adapter, layout_output, page_number, blocks)

    return read


def _with_picture_boxes(
    adapter: LayoutAdapter, layout_output: Any, page_number: int, blocks: list[PredBlock]
) -> list[PredBlock]:
    """Add the box of every content-less picture prediction on the page.

    The generic ``LayoutAdapter.to_attribution_blocks`` skips predictions without content, so a picture
    (which has no text) never reaches the layout scorer, which grades a reference picture on its box alone.
    Adapters with their own ``to_attribution_blocks`` keep their pictures and are left unchanged; a box
    already present is not added twice.
    """
    if type(adapter).to_attribution_blocks is not LayoutAdapter.to_attribution_blocks:
        return blocks
    width, height = layout_output.image_width, layout_output.image_height
    if width <= 0 or height <= 0:
        return blocks
    seen = {tuple(round(v, 6) for v in b.bbox_xyxy) for b in blocks}
    out = list(blocks)
    for idx, prediction in enumerate(layout_output.predictions):
        if (prediction.page if prediction.page is not None else 1) != page_number or prediction.content is not None:
            continue
        if prediction.label.strip().lower() not in _PICTURE_LABELS:
            continue
        box = normalize_bbox_xyxy(prediction.bbox, width=width, height=height)
        key = tuple(round(v, 6) for v in box)
        if key in seen:
            continue
        seen.add(key)
        order = prediction.provider_metadata.get("order_index")
        out.append(
            PredBlock(
                bbox_xyxy=box,
                block_type="text",
                label=prediction.label,
                text="",
                normalized_text="",
                tokens=[],
                order_index=order if isinstance(order, int) else idx,
            )
        )
    return out


def attach_next_metrics(
    result: EvaluationResult, inference_result: InferenceResult, test_case: TestCase
) -> EvaluationResult:
    """Append the extended metrics of one document to its evaluation result.

    A layout test case the layout evaluator could not score because the output has no layout for it becomes
    a scored result when an extended metric scores it (layout grounding counts it as 0). A failure inside the
    extended scorers is reported as an evaluation error on the result.
    """
    try:
        extra = compute_next_metrics(inference_result, test_case)
    except Exception as e:  # noqa: BLE001 - reported on the result like any evaluator error
        logger.exception("extended metrics failed for %s", test_case.test_id)
        return result.model_copy(update={"success": False, "error": f"Evaluation error: extended metrics: {e}"})
    if not extra:
        return result
    update: dict[str, Any] = {"metrics": [*result.metrics, *extra]}
    if not result.success and not result.metrics and _layout_unavailable(result, inference_result, test_case):
        update.update(success=True, error=None)
    return result.model_copy(update=update)


def _layout_unavailable(result: EvaluationResult, inference_result: InferenceResult, test_case: TestCase) -> bool:
    """Whether the layout evaluator failed only because the output has no layout for this document."""
    output = inference_result.output
    if not isinstance(test_case, LayoutDetectionTestCase) or not isinstance(output, ParseOutput):
        return False
    return not output.layout_pages or "No layout data" in (result.error or "")
