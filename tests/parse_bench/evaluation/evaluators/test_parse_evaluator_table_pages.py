"""Tests for page-native table matching wired through ParseEvaluator.

Covers ``_predicted_markdown_and_pages`` (predicted tables sourced from
``output.pages``) and the end-to-end behavior when a parse test case carries the
native per-page GT field ``metadata["expected_pages"]``: GriTS pairs tables within
the same page, and the predicted side does not depend on the flat
``output.markdown`` ordering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from parse_bench.evaluation.evaluators.parse import (
    ParseEvaluator,
    _gt_markdown_and_pages,
    _predicted_markdown_and_pages,
)
from parse_bench.evaluation.metrics.parse.grits_metric import GriTSMetric
from parse_bench.evaluation.metrics.parse.table_extraction import extract_html_tables, extract_normalized_tables
from parse_bench.evaluation.metrics.parse.teds_metric import TEDSMetric
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import ParseTestCase


class _PageAwareParseTestCase(ParseTestCase):
    """``ParseTestCase`` carrying the optional ``metadata`` bag the evaluator reads
    ``expected_pages`` from. Declared here so the page-aware path is exercised
    even while the public ``ParseTestCase`` schema has no ``metadata`` field."""

    metadata: dict[str, Any] | None = None


# Two same-structure 2x2 tables with disjoint content.
_T_ALPHA = "<table><tr><td>a1</td><td>a2</td></tr><tr><td>a3</td><td>a4</td></tr></table>"
_T_BETA = "<table><tr><td>b1</td><td>b2</td></tr><tr><td>b3</td><td>b4</td></tr></table>"
_T_CHECKMARK = (
    "<table><thead><tr><th>Options</th><th>Comp 1 (*)</th></tr></thead>"
    "<tbody><tr><td>Automatic Transmission</td><td>[no]</td></tr>"
    "<tr><td>Overdrive</td><td>[yes]</td></tr></tbody></table>"
)
_T_CHECKMARK_FLIPPED = _T_CHECKMARK.replace(
    "<td>Automatic Transmission</td><td>[no]</td>",
    "<td>Automatic Transmission</td><td>[yes]</td>",
)


def _parse_output(
    page_markdowns: list[str],
    *,
    document_markdown: str | None = None,
    job_id: str | None = None,
    page_indices: list[int] | None = None,
) -> ParseOutput:
    """ParseOutput with per-page markdown. ``document_markdown`` overrides the
    flat ``output.markdown`` (defaults to the page concatenation) so tests can
    deliberately desync it from the pages."""
    indices = page_indices if page_indices is not None else list(range(len(page_markdowns)))
    pages = [PageIR(page_index=i, markdown=md) for i, md in zip(indices, page_markdowns, strict=True)]
    return ParseOutput(
        example_id="grp/doc",
        pipeline_name="test_pipeline",
        markdown=document_markdown if document_markdown is not None else "\n\n".join(page_markdowns),
        pages=pages,
        job_id=job_id,
    )


def _inference_result(output: ParseOutput) -> InferenceResult:
    return InferenceResult(
        request=InferenceRequest(
            example_id="grp/doc",
            source_file_path="doc.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline_name="test_pipeline",
        product_type=ProductType.PARSE,
        raw_output={},
        output=output,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        latency_in_ms=1,
    )


def _test_case_pages(pages: list[tuple[int, str]], *, document_markdown: str | None = None) -> ParseTestCase:
    """Page-aware GT: ``pages`` = [(page_num, markdown), ...]. ``document_markdown``
    overrides the flat ``expected_markdown`` (defaults to the per-page concatenation)
    so tests can deliberately desync it from the per-page GT."""
    metadata: dict = {"expected_pages": [{"page": p, "markdown": md} for p, md in pages]}
    return _PageAwareParseTestCase.model_construct(
        test_id="grp/doc",
        group="grp",
        file_path="doc.pdf",
        expected_markdown=document_markdown if document_markdown is not None else "\n\n".join(md for _, md in pages),
        metadata=metadata,
    )


def _test_case_blob(expected_markdown: str) -> ParseTestCase:
    """Non-page-aware GT: a single concatenated blob with no page metadata."""
    return ParseTestCase.model_construct(
        test_id="grp/doc",
        group="grp",
        file_path="doc.pdf",
        expected_markdown=expected_markdown,
        metadata=None,
    )


def _by_name(metrics) -> dict[str, float]:
    return {m.metric_name: m.value for m in metrics}


def _grits_meta(metrics):
    return next(m.metadata for m in metrics if m.metric_name == "grits_con")


# --- metadata-only evaluator routing --------------------------------------


def test_predicted_markdown_and_pages_sources_from_pages():
    out = _parse_output([_T_ALPHA, "", _T_BETA + _T_BETA])  # p1:1 table, p2:0, p3:2 tables
    md, labels = _predicted_markdown_and_pages(out)
    assert labels == [1, 3, 3]
    assert len(extract_html_tables(md)) == 3


def test_predicted_markdown_and_pages_none_when_no_pages():
    out = ParseOutput(example_id="grp/doc", pipeline_name="p", markdown=_T_ALPHA, pages=[])
    assert _predicted_markdown_and_pages(out) is None


def test_gt_markdown_and_pages_built_together_order_agnostic():
    # Pages listed out of order: labels follow the entries and the markdown is
    # concatenated to match — no assumption that pages ascend.
    field = [{"page": 5, "markdown": _T_BETA}, {"page": 2, "markdown": _T_ALPHA + _T_ALPHA}]
    md, labels = _gt_markdown_and_pages(field)
    assert labels == [5, 2, 2]
    assert len(extract_html_tables(md)) == 3


# --- end-to-end through ParseEvaluator -------------------------------------


def test_page_metadata_constrains_grits_pairing():
    # GT: alpha@1, beta@2. Prediction: beta@1, alpha@2 (cross-page look-alikes).
    evaluator = ParseEvaluator()  # grits on, teds off (sequential path)
    test_case = _test_case_pages([(1, _T_ALPHA), (2, _T_BETA)])
    result = evaluator.evaluate(_inference_result(_parse_output([_T_BETA, _T_ALPHA])), test_case)

    meta = _grits_meta(result.metrics)
    # Same-page pairing only — no detail crosses pages.
    for d in meta["per_table_details"]:
        if d.get("pred_table_index") is not None:
            assert d["gt_page"] == d["pred_page"]
    # alpha-vs-beta on each page is an imperfect match, so grits_con < 1.0.
    assert _by_name(result.metrics)["grits_con"] < 1.0


def test_table_metrics_penalize_a_flipped_checkmark():
    """Content metrics must see a single mark flip even when shape is unchanged."""

    expected, _ = extract_normalized_tables(_T_CHECKMARK, side="expected")
    actual, _ = extract_normalized_tables(_T_CHECKMARK_FLIPPED, side="actual")

    grits = GriTSMetric().compute(expected, actual)
    teds = TEDSMetric().compute(_T_CHECKMARK, _T_CHECKMARK_FLIPPED)

    assert next(metric.value for metric in grits if metric.metric_name == "grits_con") < 1.0
    assert next(metric.value for metric in teds if metric.metric_name == "teds") < 1.0
    # Structural-only scores are expected to remain perfect: the regression
    # guard is specifically the content-bearing variants above.
    assert next(metric.value for metric in teds if metric.metric_name == "teds_struct") == 1.0


def test_page_aware_metrics_surface_missing_and_extra_page_tables():
    """A missing table cannot pair across pages; cardinalities expose an extra."""

    evaluator = ParseEvaluator(enable_teds=True)
    test_case = _test_case_pages([(11, _T_ALPHA), (14, _T_BETA)])
    # Page 11 is missing; page 14 has the correct table plus an extra table.
    output = _parse_output(["", _T_BETA + _T_ALPHA], page_indices=[10, 13])
    result = evaluator.evaluate(_inference_result(output), test_case)

    grits = _grits_meta(result.metrics)
    teds = next(metric for metric in result.metrics if metric.metric_name == "teds")
    assert _by_name(result.metrics)["grits_con"] == 0.5
    assert teds.value == 0.5
    assert grits["tables_found_expected"] == 2
    assert grits["tables_found_actual"] == 2
    assert grits["tables_matched"] == 1
    matched = [detail for detail in grits["per_table_details"] if detail.get("pred_table_index") is not None]
    assert len(matched) == 1
    assert matched[0]["gt_page"] == matched[0]["pred_page"] == 14
    assert teds.metadata["tables_found_actual"] == 2
    assert teds.metadata["tables_matched"] == 1


def test_predicted_tables_sourced_from_pages_not_document_markdown():
    # THE regression that proves the fix: output.markdown is ordered DIFFERENTLY
    # from output.pages. The old approach (tables from output.markdown, labels
    # from output.pages, joined by index) would mislabel the pages and mispair;
    # sourcing predicted tables from output.pages keeps it correct.
    evaluator = ParseEvaluator()
    test_case = _test_case_pages([(1, _T_ALPHA), (2, _T_BETA)])
    output = _parse_output(
        [_T_ALPHA, _T_BETA],  # pages: alpha@1, beta@2 (correct)
        document_markdown=_T_BETA + _T_ALPHA,  # flat markdown reversed (would mislead a positional join)
    )
    result = evaluator.evaluate(_inference_result(output), test_case)

    assert _by_name(result.metrics)["grits_con"] == 1.0  # alpha@1<->alpha@1, beta@2<->beta@2
    for d in _grits_meta(result.metrics)["per_table_details"]:
        if d.get("pred_table_index") is not None:
            assert d["gt_page"] == d["pred_page"]


def test_gt_tables_sourced_from_expected_pages_not_blob():
    # GT side, mirror of the predicted-side regression: expected_pages is listed
    # out of page order AND the expected_markdown blob is in yet another order.
    # GT tables must be sourced from expected_pages (page intrinsic), so matching
    # stays correct regardless of order.
    evaluator = ParseEvaluator()
    test_case = _test_case_pages(
        [(2, _T_BETA), (1, _T_ALPHA)],  # pages listed out of order
        document_markdown=_T_ALPHA + _T_BETA,  # flat blob in a different order again
    )
    result = evaluator.evaluate(_inference_result(_parse_output([_T_ALPHA, _T_BETA])), test_case)

    assert _by_name(result.metrics)["grits_con"] == 1.0  # alpha@1<->alpha@1, beta@2<->beta@2
    for d in _grits_meta(result.metrics)["per_table_details"]:
        if d.get("pred_table_index") is not None:
            assert d["gt_page"] == d["pred_page"]


def test_no_page_metadata_matches_globally():
    # No page labels -> global matching cross-pairs the identical content and
    # scores 1.0 (normal-dataset behavior).
    evaluator = ParseEvaluator()
    test_case = _test_case_blob(_T_ALPHA + _T_BETA)
    result = evaluator.evaluate(_inference_result(_parse_output([_T_BETA, _T_ALPHA])), test_case)

    assert _by_name(result.metrics)["grits_con"] == 1.0


def test_normal_multipage_doc_single_blob_gt_stays_global():
    # Backward-compat guard: a multi-page prediction but a single-blob GT with no
    # page metadata must match GLOBALLY (not be page-blocked), so the cross-page
    # look-alikes still pair and score 1.0 — identical to pre-page-aware behavior.
    evaluator = ParseEvaluator()
    test_case = _test_case_blob(_T_ALPHA + _T_BETA)
    output = _parse_output([_T_BETA, _T_ALPHA])  # prediction genuinely spans 2 pages
    result = evaluator.evaluate(_inference_result(output), test_case)
    assert _by_name(result.metrics)["grits_con"] == 1.0


def test_empty_pages_falls_back_to_global():
    # output.pages empty -> page-blocking disabled (global matching, score 1.0).
    evaluator = ParseEvaluator()
    test_case = _test_case_pages([(1, _T_ALPHA), (2, _T_BETA)])
    output = ParseOutput(
        example_id="grp/doc",
        pipeline_name="test_pipeline",
        markdown=_T_BETA + _T_ALPHA,
        pages=[],
    )
    result = evaluator.evaluate(_inference_result(output), test_case)

    assert _by_name(result.metrics)["grits_con"] == 1.0  # fell back to global cross-pairing
