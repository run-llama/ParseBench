"""Every retained typed IR writer must preserve geometry and confidence."""

from datetime import datetime

import pytest

from parse_bench.evaluation.layout_adapters import adapters
from parse_bench.evaluation.metrics.attribution.core import compute_attribution_metrics, parse_gt_elements
from parse_bench.schemas.layout_detection_output import LayoutOutput
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult


@pytest.mark.parametrize(
    "name",
    [
        "AnthropicLayoutAdapter",
        "AzureDILayoutAdapter",
        "Chandra2LayoutAdapter",
        "DatabricksAiParseLayoutAdapter",
        "DatalabLayoutAdapter",
        "DeepSeekLayoutAdapter",
        "DeepSeekOCR2LayoutAdapter",
        "DoclingParseLayoutAdapter",
        "DotsOcrLayoutAdapter",
        "ExtendLayoutAdapter",
        "Gemma4LayoutAdapter",
        "GoogleDocAILayoutAdapter",
        "GoogleLayoutAdapter",
        "InfinityParser2LayoutAdapter",
        "KdlFrontierNanoLayoutAdapter",
        "LandingAILayoutAdapter",
        "LiteParseLayoutAdapter",
        "MinerU25LayoutAdapter",
        "OIParserLayoutAdapter",
        "OpenAILayoutAdapter",
        "PulseLayoutAdapter",
        "PyMuPDF4LLMLayoutAdapter",
        "QfOcrLayoutAdapter",
        "Qwen3VLLayoutAdapter",
        "QwenLayoutAdapter",
        "ReductoLayoutAdapter",
        "TextractLayoutAdapter",
        "UnstructuredLayoutAdapter",
    ],
)
@pytest.mark.parametrize("angle", [None, 0, 30, -30, 90, -90, 180])
@pytest.mark.parametrize("confidence", [None, 0, 0.87])
@pytest.mark.parametrize("reverse", [False, True])
def test_retained_typed_writer(name, angle, confidence, reverse):
    pages = []
    for number, width, height in [(1, 200, 100), (2, 100, 200)]:
        sx, sy = (width, height) if name in {"OIParserLayoutAdapter", "InfinityParser2LayoutAdapter"} else (1, 1)
        segment = LayoutSegmentIR(
            x=0.2 * sx,
            y=0.45 * sy,
            w=0.6 * sx,
            h=0.05 * sy,
            r=angle,
            label="text",
            confidence=confidence,
            start_index=0,
            end_index=4,
        )
        pages.append(
            ParseLayoutPageIR(
                page_number=number,
                width=width,
                height=height,
                items=[LayoutItemIR(type="text", value="alpha", bbox=segment, layout_segments=[segment])],
            )
        )
    if reverse:
        pages.reverse()
    result = InferenceResult(
        request=InferenceRequest(example_id="typed", source_file_path="typed.pdf", product_type="parse"),
        pipeline_name="typed",
        product_type="parse",
        raw_output={},
        output=ParseOutput(example_id="typed", pipeline_name="typed", markdown="alpha", layout_pages=pages),
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
        latency_in_ms=0,
    )
    adapter = getattr(adapters, name)()
    layout = adapter.to_layout_output(result)
    assert [p.r for p in layout.predictions] == [angle, angle]
    assert [p.score for p in layout.predictions] == [1 if confidence is None else confidence] * 2
    # JSON transport and filtering must preserve the same page frame.
    restored = LayoutOutput.model_validate_json(layout.model_dump_json())
    for number, width, height in [(1, 200, 100), (2, 100, 200)]:
        for output in [restored, adapter.to_layout_output(result, page_filter=number)]:
            blocks = adapter.to_attribution_blocks(output, page_number=number)
            assert len(blocks) == 1
            block = blocks[0]
            assert block.r == angle
            assert (block.page_width, block.page_height) == (width, height)
            assert block.bbox_xyxy == pytest.approx([0.2, 0.45, 0.8, 0.5])
            assert block.text == "alpha"
            assert block.tokens == ["alpha"]
            gt = parse_gt_elements(
                [
                    {
                        "type": "layout",
                        "bbox": [0.2, 0.45, 0.6, 0.05],
                        "r": angle,
                        "canonical_class": "Text",
                        "content": {"type": "text", "text": "alpha"},
                    }
                ]
            )
            assert compute_attribution_metrics(gt, blocks).af1 == pytest.approx(1)


@pytest.mark.parametrize("angle", [None, 0, 30, -30, 90, -90, 180])
def test_public_canonical_converter(angle):
    from parse_bench.inference.providers.layoutdet.adapters import canonical_to_core
    from parse_bench.schemas.layout_detection_output import CanonicalLayoutPrediction
    from parse_bench.schemas.layout_ontology import CanonicalLabel

    prediction = CanonicalLayoutPrediction(
        bbox=[20, 45, 80, 50],
        r=angle,
        score=0,
        canonical_class=CanonicalLabel.TEXT,
        original_label="text",
        attributes={"scope": "mark"},
    )
    core = canonical_to_core(prediction)
    assert core is not None
    assert core.r == angle
    assert core.bbox == prediction.bbox
    assert core.score == 0
    assert core.attributes == prediction.attributes
    assert core.original_label == "text"
    prediction.canonical_class = CanonicalLabel.FORM
    assert canonical_to_core(prediction) is None
