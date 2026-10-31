"""Adapter picture boxes reach layout grounding (the generic adapter path drops content-less boxes)."""

from __future__ import annotations

from datetime import UTC, datetime

from layout_grounding_rule_builders import _block, _rule

from parse_bench.evaluation.layout_adapters.base import LayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.evaluation.metrics.parse.layout_grounding import compute_layout_grounding_metrics
from parse_bench.evaluation.metrics.parse.next_metrics import _attribution_block_reader, _with_picture_boxes
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import ParseTestCase

PICTURE = [0.1, 0.2, 0.5, 0.3]  # x, y, w, h (normalized)
TEXT = [0.1, 0.6, 0.5, 0.05]


def _item(kind: str, label: str, box: list[float], value: str) -> LayoutItemIR:
    x, y, w, h = box
    return LayoutItemIR(
        type=kind, value=value, layout_segments=[LayoutSegmentIR(x=x, y=y, w=w, h=h, label=label, confidence=1.0)]
    )


def _vendor_result() -> InferenceResult:
    page = ParseLayoutPageIR(
        page_number=1,
        width=612,
        height=792,
        items=[
            _item("figure", "Picture", PICTURE, "A bar chart of revenue"),
            _item("text", "Text", TEXT, "Revenue grew"),
        ],
    )
    return InferenceResult(
        request=InferenceRequest(example_id="p", source_file_path="p.pdf", product_type=ProductType.PARSE),
        pipeline_name="reducto",
        product_type=ProductType.PARSE,
        raw_output={"_config": {"ocr_system": "standard"}},  # shape the adapter registry recognizes
        output=ParseOutput(example_id="p", pipeline_name="reducto", markdown="Revenue grew", layout_pages=[page]),
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        latency_in_ms=1,
    )


def _case() -> ParseTestCase:
    return ParseTestCase.model_construct(test_id="p", group="g", file_path="p.pdf", test_rules=[], metadata={})


def test_picture_box_without_content_reaches_layout_grounding():
    """A Picture with no content is dropped by the generic block builder unless its box is restored."""
    blocks = _attribution_block_reader(_vendor_result(), _case())(1)
    pictures = [b for b in blocks if b.label == "Picture"]
    assert len(pictures) == 1
    assert pictures[0].bbox_xyxy == [0.1, 0.2, 0.6, 0.5]
    assert pictures[0].tokens == []


def test_reference_picture_is_scored_on_the_restored_box():
    """With the box back, a reference Picture at the same place is localized (it needs no text)."""
    rules = [_rule("P", PICTURE, "Picture"), _rule("T", TEXT, content={"type": "text", "text": "Revenue grew"})]
    (metric,) = compute_layout_grounding_metrics(rules, _attribution_block_reader(_vendor_result(), _case()))
    picture = next(r for r in metric.metadata["element_results"] if r["gt_class"] == "Picture")
    assert picture["score"] > 0.9


def test_adapters_with_their_own_block_builder_are_left_alone_and_boxes_are_not_doubled():
    """Adapters that build their own blocks keep them; the generic path does not add a second picture copy."""
    result = _vendor_result()
    adapter = create_layout_adapter_for_result(result)
    layout = adapter.to_layout_output(result)

    class OwnBlocks(type(adapter)):
        def to_attribution_blocks(self, layout_output, *, page_number, test_case=None):
            return []

    assert _with_picture_boxes(OwnBlocks(), layout, 1, []) == []
    already = [_block([0.1, 0.2, 0.5, 0.3], "", 0)]
    assert type(adapter).to_attribution_blocks is LayoutAdapter.to_attribution_blocks
    assert len(_with_picture_boxes(adapter, layout, 1, already)) == 1
