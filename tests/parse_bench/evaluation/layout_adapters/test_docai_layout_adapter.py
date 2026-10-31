"""DocAI's layout adapter keeps the text read from a Picture (chart table, figure text); LandingAI's does not."""

from datetime import datetime

from parse_bench.evaluation.layout_adapters import adapters
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult


def _result(provider: str) -> InferenceResult:
    def item(kind: str, label: str, value: str, y: float) -> LayoutItemIR:
        seg = LayoutSegmentIR(x=0.1, y=y, w=0.8, h=0.2, label=label)
        return LayoutItemIR(type=kind, value=value, bbox=seg, layout_segments=[seg])

    pages = [
        ParseLayoutPageIR(
            page_number=n,
            width=1000,
            height=1000,
            items=[
                item("image", "Picture", f"Sales {n}\n\n| Year | Sales |\n|---|---|\n| 2024 | 12 |", 0.1),
                item("image", "Picture", "", 0.4),
                item("text", "Text", f"body {n}", 0.7),
            ],
        )
        for n in (1, 2)
    ]
    return InferenceResult(
        request=InferenceRequest(example_id="d", source_file_path="d.pdf", product_type="parse"),
        pipeline_name="docai_default",
        product_type="parse",
        raw_output={"provider": provider, "grounding": {}, "chunks": []},
        output=ParseOutput(example_id="d", pipeline_name="docai_default", markdown="", layout_pages=pages),
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
        latency_in_ms=0,
    )


def _texts(layout) -> list[str | None]:
    return [getattr(p.content, "text", None) for p in layout.predictions]


def test_docai_keeps_picture_content_and_attributes_it():
    adapter = adapters.DocAILayoutAdapter()
    result = _result("docai")
    assert adapter.matches(result)
    layout = adapter.to_layout_output(result)
    texts = _texts(layout)
    assert texts[0].startswith("Sales 1") and texts[3].startswith("Sales 2")
    assert texts[1] is None and texts[4] is None  # an empty Picture still has no content
    assert texts[2] == "body 1"
    blocks = adapter.to_attribution_blocks(layout, page_number=2)
    assert [b.label for b in blocks] == ["Picture", "Text"]
    assert "2024" in blocks[0].tokens

    one_page = adapter.to_layout_output(result, page_filter=2)
    assert [t and t.split("\n")[0] for t in _texts(one_page)] == ["Sales 2", None, "body 2"]


def test_landingai_still_drops_picture_content():
    layout = adapters.LandingAILayoutAdapter().to_layout_output(_result("landingai"))
    assert _texts(layout) == [None, None, "body 1", None, None, "body 2"]
