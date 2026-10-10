"""X2Knowledge layout: one prediction per API element, element text as content (Picture included),
tables as HTML when present, per-page geometry, page_filter, Canonical17 pass-through and the
adapter / label-mapper registries."""

from __future__ import annotations

from datetime import datetime

import pytest

from parse_bench.evaluation.layout_adapters import registry as registry_module
from parse_bench.evaluation.layout_adapters.adapters import X2KnowledgeLayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.evaluation.layout_label_mappers.mappers import X2KnowledgeLabelMapper
from parse_bench.evaluation.layout_label_mappers.projection import project_layout_predictions
from parse_bench.evaluation.layout_label_mappers.registry import build_mapping_context, resolve_layout_label_mapper
from parse_bench.layout_label_mapping import UnknownRawLayoutLabelError
from parse_bench.schemas.layout_detection_output import (
    LAYOUT_MODEL_INFO,
    LayoutDetectionModel,
    LayoutTableContent,
    LayoutTextContent,
)
from parse_bench.schemas.layout_ontology import CanonicalLabel
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType

_TABLE_HTML = "<table><tr><td>a</td><td>b</td></tr></table>"


def _item(label: str, box: tuple[float, float, float, float], text: str = "", html: str = "") -> LayoutItemIR:
    """A layout item exactly as X2KnowledgeProvider.normalize() builds it from a single API element."""
    segment = LayoutSegmentIR(x=box[0], y=box[1], w=box[2], h=box[3], label=label, confidence=0.9)
    value = html if (label == "Table" and html) else text
    return LayoutItemIR(type=label, md=text, html=html, value=value, bbox=segment, layout_segments=[segment])


def _result(
    pages: list[list[LayoutItemIR]],
    *,
    sizes: list[tuple[float, float]] | None = None,
    raw: dict | None = None,
    pipeline_name: str = "x2knowledge_v1",
) -> InferenceResult:
    sizes = sizes or [(1000.0, 2000.0)] * len(pages)
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(example_id="doc-1", source_file_path="/tmp/doc-1.pdf", product_type=ProductType.PARSE),
        pipeline_name=pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw
        if raw is not None
        else {"provider": "x2knowledge", "object": "x2knowledge.document", "pages": []},
        output=ParseOutput(
            task_type="parse",
            example_id="doc-1",
            pipeline_name=pipeline_name,
            pages=[PageIR(page_index=i, markdown=f"page {i + 1}") for i in range(len(pages))],
            layout_pages=[
                ParseLayoutPageIR(page_number=i + 1, width=w, height=h, md=f"page {i + 1}", items=items)
                for i, (items, (w, h)) in enumerate(zip(pages, sizes, strict=True))
            ],
            markdown="\n\n".join(f"page {i + 1}" for i in range(len(pages))),
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_each_element_is_exactly_one_prediction_in_page_pixels() -> None:
    picture = _item("Picture", (0.5, 0.5, 0.4, 0.3), "Logo 2024")
    # A second segment on an item must not become a second prediction: one element, one box.
    picture.layout_segments.append(LayoutSegmentIR(x=0.0, y=0.0, w=0.1, h=0.1, label="Picture"))
    items = [_item("Text", (0.1, 0.1, 0.2, 0.05), "Alpha"), picture]

    layout = X2KnowledgeLayoutAdapter().to_layout_output(_result([items]))

    assert layout.model == LayoutDetectionModel.X2KNOWLEDGE_LAYOUT
    assert (layout.image_width, layout.image_height) == (1000, 2000)
    assert [p.label for p in layout.predictions] == ["Text", "Picture"]
    assert [p.bbox for p in layout.predictions] == [
        pytest.approx([100.0, 200.0, 300.0, 300.0]),
        pytest.approx([500.0, 1000.0, 900.0, 1600.0]),
    ]
    assert [p.score for p in layout.predictions] == [pytest.approx(0.9)] * 2
    assert [p.provider_metadata["order_index"] for p in layout.predictions] == [0, 1]


def test_element_text_is_the_content_for_every_label_and_tables_prefer_html() -> None:
    items = [
        _item("Picture", (0.0, 0.0, 0.5, 0.5), "Logo 2024"),
        _item("Table", (0.0, 0.5, 0.5, 0.2), "a b", html=_TABLE_HTML),
        _item("Table", (0.5, 0.5, 0.5, 0.2), "c d"),
        _item("Section-header", (0.5, 0.0, 0.5, 0.1), "Revenue"),
        _item("Text", (0.5, 0.2, 0.5, 0.1)),
    ]

    contents = [p.content for p in X2KnowledgeLayoutAdapter().to_layout_output(_result([items])).predictions]

    assert contents == [
        LayoutTextContent(text="Logo 2024"),
        LayoutTableContent(html=_TABLE_HTML),
        LayoutTextContent(text="c d"),
        LayoutTextContent(text="Revenue"),
        None,
    ]


def test_picture_text_is_one_switch_away_from_being_withheld(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(X2KnowledgeLayoutAdapter, "include_picture_text", False)
    items = [_item("Picture", (0.0, 0.0, 0.5, 0.5), "Logo 2024"), _item("Text", (0.5, 0.5, 0.2, 0.1), "Body")]

    contents = [p.content for p in X2KnowledgeLayoutAdapter().to_layout_output(_result([items])).predictions]

    assert contents == [None, LayoutTextContent(text="Body")]


def test_page_filter_selects_one_page_and_its_geometry() -> None:
    pages = [[_item("Text", (0.1, 0.1, 0.2, 0.05), "one")], [_item("Text", (0.1, 0.1, 0.2, 0.05), "two")]]
    result = _result(pages, sizes=[(1000.0, 2000.0), (2000.0, 1000.0)])

    second = X2KnowledgeLayoutAdapter().to_layout_output(result, page_filter=2)

    assert [(p.page, p.content) for p in second.predictions] == [(2, LayoutTextContent(text="two"))]
    assert second.predictions[0].bbox == pytest.approx([200.0, 100.0, 600.0, 150.0])
    assert (second.image_width, second.image_height) == (2000, 1000)
    assert second.markdown == "page 2"
    assert [(lp.page_number, lp.width, lp.height, lp.items) for lp in second.layout_pages] == [(2, 2000.0, 1000.0, [])]

    missing = X2KnowledgeLayoutAdapter().to_layout_output(result, page_filter=3)
    assert missing.predictions == [] and missing.layout_pages == []


def test_pages_of_different_sizes_normalize_back_to_the_api_boxes() -> None:
    pages = [[_item("Text", (0.1, 0.2, 0.3, 0.1), "one")], [_item("Table", (0.5, 0.5, 0.25, 0.25), "two")]]
    result = _result(pages, sizes=[(1000.0, 2000.0), (3000.0, 1500.0)])
    layout = X2KnowledgeLayoutAdapter().to_layout_output(result)

    projected = project_layout_predictions(result, layout, evaluation_view="canonical")

    assert [(p["page"], p["class_name"]) for p in projected] == [(1, "Text"), (2, "Table")]
    assert projected[0]["bbox"] == pytest.approx([0.1, 0.2, 0.4, 0.3])
    assert projected[1]["bbox"] == pytest.approx([0.5, 0.5, 0.75, 0.75])


def test_empty_layout_still_yields_a_layout_output() -> None:
    layout = X2KnowledgeLayoutAdapter().to_layout_output(_result([[]]))

    assert layout.predictions == []
    assert layout.model == LayoutDetectionModel.X2KNOWLEDGE_LAYOUT


def test_matches_only_x2knowledge_payloads() -> None:
    assert X2KnowledgeLayoutAdapter.matches(_result([[]]))
    assert not X2KnowledgeLayoutAdapter.matches(_result([[]], raw={"results": {}}))
    assert not X2KnowledgeLayoutAdapter.matches(_result([[]], raw={"provider": "x2knowledge", "pages": []}))


def test_registry_resolves_the_adapter_by_provider_key() -> None:
    assert isinstance(create_layout_adapter_for_result(_result([[]])), X2KnowledgeLayoutAdapter)


def test_shape_matcher_fallback_picks_the_adapter_for_an_unregistered_pipeline_name() -> None:
    def _resolver(pipeline_name: str) -> PipelineSpec | None:
        if pipeline_name != "x2knowledge_renamed":
            return None
        return PipelineSpec(
            pipeline_name=pipeline_name, provider_name="x2knowledge_renamed", product_type=ProductType.PARSE
        )

    registry_module.register_pipeline_resolver(_resolver)
    try:
        result = _result([[_item("Text", (0.1, 0.1, 0.2, 0.05), "one")]], pipeline_name="x2knowledge_renamed")
        assert isinstance(create_layout_adapter_for_result(result), X2KnowledgeLayoutAdapter)
    finally:
        registry_module._PIPELINE_RESOLVERS.remove(_resolver)


def test_label_mapper_passes_canonical_labels_through() -> None:
    result = _result([[_item("Section-header", (0.0, 0.0, 1.0, 0.1), "Revenue")]])
    layout = X2KnowledgeLayoutAdapter().to_layout_output(result)
    [prediction] = layout.predictions
    context = build_mapping_context(result, layout)

    mapper = resolve_layout_label_mapper(context)

    assert isinstance(mapper, X2KnowledgeLabelMapper)
    assert all(mapper.to_canonical(label.value, prediction, context) == label for label in CanonicalLabel)
    with pytest.raises(UnknownRawLayoutLabelError):
        mapper.to_canonical("section_header", prediction, context)


def test_layout_model_is_registered_for_display() -> None:
    assert LayoutDetectionModel.X2KNOWLEDGE_LAYOUT.value == "x2knowledge_layout"
    assert LAYOUT_MODEL_INFO[LayoutDetectionModel.X2KNOWLEDGE_LAYOUT]["name"] == "X2Knowledge"
