"""Cross-evaluation scores attribution from the adapter that read the parse output.

The runner adapts a PARSE result to a ``LayoutOutput`` and evaluates a synthetic
LAYOUT_DETECTION result built from it. Re-resolving an adapter from that synthetic
result only sees ``raw_output``, so a LlamaParse-style adapter lost the pages it had
read from ``ParseOutput.layout_pages`` and fell back to prediction-level blocks, which
drop chart and table content, unless the raw SDK response happened to be present.
Published results carry no SDK response, so their Visual Grounding was lower than a
local run of the same outputs.
"""

from datetime import datetime

from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import LayoutDetectionTestCase

_LINES = [("alpha beta", 10), ("gamma delta", 40)]


def _parse_result() -> InferenceResult:
    # One item whose two segments each own a span of its text, as LlamaParse emits
    # for a block that wraps across regions. Only the pages read from layout_pages
    # slice the item text per segment.
    text = "\n".join(t for t, _ in _LINES)
    segments, start = [], 0
    for line, y in _LINES:
        segments.append(
            LayoutSegmentIR(
                x=10, y=y, w=80, h=10, label="text", confidence=1.0, start_index=start, end_index=start + len(line) - 1
            )
        )
        start += len(line) + 1
    items = [LayoutItemIR(type="Text", md=text, value=text, bbox=segments[0], layout_segments=segments)]
    output = ParseOutput(
        task_type="parse",
        example_id="layout/probe",
        pipeline_name="llamaparse_agentic",
        pages=[PageIR(page_index=0, markdown="\n\n".join(t for t, _ in _LINES))],
        layout_pages=[ParseLayoutPageIR(page_number=1, width=100, height=100, items=items)],
        markdown="\n\n".join(t for t, _ in _LINES),
    )
    return InferenceResult(
        request=InferenceRequest(
            example_id="layout/probe", source_file_path="/tmp/probe.pdf", product_type=ProductType.PARSE
        ),
        pipeline_name="llamaparse_agentic",
        product_type=ProductType.PARSE,
        raw_output={},  # as published: no SDK response
        output=output,
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
        latency_in_ms=0,
    )


def _case() -> LayoutDetectionTestCase:
    rules = [
        {
            "type": "layout",
            "page": 1,
            "bbox": [0.1, y / 100, 0.8, 0.1],
            "canonical_class": "Text",
            "content": {"type": "text", "text": text},
            "ro_index": i,
        }
        for i, (text, y) in enumerate(_LINES)
    ]
    return LayoutDetectionTestCase(test_id="probe", group="test", file_path="/tmp/probe.pdf", test_rules=rules)


def _cross_eval(pass_adapter: bool) -> dict[str, float]:
    parse_result = _parse_result()
    adapter = create_layout_adapter_for_result(parse_result)
    layout_output = adapter.to_layout_output(parse_result)
    synthetic = parse_result.model_copy(update={"product_type": ProductType.LAYOUT_DETECTION, "output": layout_output})
    evaluator = LayoutDetectionEvaluator()
    result = (
        evaluator.evaluate(synthetic, _case(), layout_adapter=adapter)
        if pass_adapter
        else evaluator.evaluate(synthetic, _case())
    )
    return {m.metric_name: m.value for m in result.metrics}


def test_attribution_uses_the_pages_read_from_the_parse_output() -> None:
    metrics = _cross_eval(pass_adapter=True)
    assert metrics["layout_localization_pass_rate"] == 1.0
    assert metrics["layout_attribution_pass_rate"] == 1.0


def test_evaluator_uses_the_given_adapter_instead_of_re_resolving(monkeypatch) -> None:
    # On real LlamaParse outputs the re-resolved adapter dropped chart and table
    # blocks (1,477 vs 1,610 on 60 pages), costing up to ~12 points of grounding.
    import parse_bench.evaluation.evaluators.layoutdet as layoutdet

    def _no_re_resolve(_result):
        raise AssertionError("the evaluator re-resolved an adapter from the synthetic layout result")

    monkeypatch.setattr(layoutdet, "create_layout_adapter_for_result", _no_re_resolve)
    metrics = _cross_eval(pass_adapter=True)
    assert metrics["layout_attribution_pass_rate"] == 1.0
