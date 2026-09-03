"""The LlamaParse layout adapter accepts canonicalised ``ParseOutput.layout_pages``."""

from __future__ import annotations

from datetime import datetime

from parse_bench.evaluation.layout_adapters.adapters import LlamaParseLayoutAdapter
from parse_bench.inference.providers.parse.llamaparse_v2_normalization import build_parse_output_from_pages
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _make_result(output: ParseOutput) -> InferenceResult:
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(
            example_id=output.example_id,
            source_file_path="/tmp/doc-1.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline_name=output.pipeline_name,
        product_type=ProductType.PARSE,
        raw_output={},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _canonicalised_output() -> ParseOutput:
    return build_parse_output_from_pages(
        pages_payload=[
            {
                "page": 1,
                "width": 1000,
                "height": 1000,
                "md": "Header\n\nTitle\n\nBody [X] Yes\n\n![chart: sales](img.png)",
                "items": [
                    {
                        "type": "text",
                        "value": "Header",
                        "layoutAwareBbox": [
                            {
                                "x": 0,
                                "y": 0,
                                "w": 100,
                                "h": 10,
                                "label": "header",
                                "startIndex": 0,
                                "endIndex": 5,
                                "confidence": 0.9,
                            },
                        ],
                    },
                    {
                        "type": "heading",
                        "value": "Title",
                        "layoutAwareBbox": [
                            {
                                "x": 0,
                                "y": 20,
                                "w": 100,
                                "h": 10,
                                "label": "paragraph_title",
                                "startIndex": 8,
                                "endIndex": 12,
                                "confidence": 0.9,
                            },
                        ],
                    },
                    {
                        "type": "text",
                        "md": "Body [X] Yes",
                        "value": "Body [X] Yes",
                        "layoutAwareBbox": [
                            {
                                "x": 0,
                                "y": 40,
                                "w": 100,
                                "h": 10,
                                "label": "text",
                                "startIndex": 0,
                                "endIndex": 11,
                                "confidence": 0.9,
                            },
                            {
                                "x": 30,
                                "y": 40,
                                "w": 8,
                                "h": 8,
                                "label": "checkbox-selected",
                                "startIndex": 5,
                                "endIndex": 7,
                                "confidence": 0.9,
                            },
                        ],
                    },
                    {
                        "type": "image",
                        "md": "![chart: sales](img.png)",
                        "layoutAwareBbox": [{"x": 0, "y": 60, "w": 100, "h": 100, "label": "image", "confidence": 0.9}],
                    },
                ],
            }
        ],
        example_id="doc-1",
        pipeline_name="llamaparse",
    )


def test_adapter_matches_and_extracts_canonicalised_layout_pages() -> None:
    output = _canonicalised_output()
    # Precondition: the normaliser canonicalised item types.
    assert [item.type for item in output.layout_pages[0].items] == [
        "Page-header",
        "Section-header",
        "Text",
        "Checkbox-Selected",
        "Picture",
    ]

    result = _make_result(output)
    assert LlamaParseLayoutAdapter.matches(result)

    layout_output = LlamaParseLayoutAdapter().to_layout_output(result)

    # Predictions keep the raw detector labels (the mapper canonicalises them)
    # while the canonical item type rides along in provider metadata.
    assert [prediction.label for prediction in layout_output.predictions] == [
        "header",
        "paragraph_title",
        "text",
        "checkbox-selected",
        "image",
    ]
    assert [prediction.provider_metadata["item_type"] for prediction in layout_output.predictions] == [
        "Page-header",
        "Section-header",
        "Text",
        "Checkbox-Selected",
        "Picture",
    ]
    # The canonical Checkbox-* item type is what promotes the checkbox
    # prediction to mark scope on the evaluator side.
    checkbox = layout_output.predictions[3]
    assert checkbox.attributes == {"scope": "mark"}
    assert layout_output.layout_pages[0].items[3].attributes == {"scope": "mark"}


def test_adapter_page_filter_on_canonicalised_output() -> None:
    result = _make_result(_canonicalised_output())
    filtered = LlamaParseLayoutAdapter().to_layout_output(result, page_filter=2)
    assert filtered.predictions == []
    kept = LlamaParseLayoutAdapter().to_layout_output(result, page_filter=1)
    assert len(kept.predictions) == 5
