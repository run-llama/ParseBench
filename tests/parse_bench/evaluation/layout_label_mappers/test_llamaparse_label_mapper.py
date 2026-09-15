"""The LlamaParse label mapper accepts raw ``layoutAwareBbox`` labels and Canonical17 strings."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.layout_label_mappers.base import MappingContext
from parse_bench.evaluation.layout_label_mappers.mappers import LlamaParseRawLabelMapper
from parse_bench.layout_label_mapping import UnknownRawLayoutLabelError
from parse_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
)
from parse_bench.schemas.layout_ontology import CanonicalLabel


def _context(labels: list[str]) -> MappingContext:
    predictions = [LayoutPrediction(bbox=[0.0, 0.0, 1.0, 1.0], score=1.0, label=label, page=1) for label in labels]
    layout_output = LayoutOutput(
        example_id="ex",
        pipeline_name="llamaparse",
        model=LayoutDetectionModel.LLAMAPARSE,
        image_width=100,
        image_height=100,
        predictions=predictions,
    )
    return MappingContext(
        provider_name="llamaparse",
        pipeline_name="llamaparse",
        model=LayoutDetectionModel.LLAMAPARSE,
        raw_output={},
        layout_output=layout_output,
    )


@pytest.mark.parametrize(
    "canonical",
    sorted(CanonicalLabel, key=lambda label: label.value),
)
def test_canonical_labels_map_to_themselves(canonical: CanonicalLabel) -> None:
    """Canonicalised ``layout_pages`` carry Canonical17 item types; the mapper
    must pass them through instead of failing the raw-label lookup."""
    context = _context([canonical.value])
    prediction = context.layout_output.predictions[0]
    assert LlamaParseRawLabelMapper().to_canonical(canonical.value, prediction, context) is canonical


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("header", CanonicalLabel.PAGE_HEADER),
        ("paragraph_title", CanonicalLabel.SECTION_HEADER),
        ("image", CanonicalLabel.PICTURE),
        ("section-header", CanonicalLabel.SECTION_HEADER),
        ("picture", CanonicalLabel.PICTURE),
    ],
)
def test_raw_labels_still_map(raw: str, expected: CanonicalLabel) -> None:
    context = _context([raw])
    prediction = context.layout_output.predictions[0]
    assert LlamaParseRawLabelMapper().to_canonical(raw, prediction, context) == expected


def test_mixed_raw_and_canonical_labels_in_one_output() -> None:
    context = _context(["Page-header", "text"])
    mapper = LlamaParseRawLabelMapper()
    first, second = context.layout_output.predictions
    assert mapper.to_canonical("Page-header", first, context) == CanonicalLabel.PAGE_HEADER
    assert mapper.to_canonical("text", second, context) == CanonicalLabel.TEXT


def test_unknown_label_still_raises() -> None:
    context = _context(["fantasy-label"])
    prediction = context.layout_output.predictions[0]
    with pytest.raises(UnknownRawLayoutLabelError):
        LlamaParseRawLabelMapper().to_canonical("fantasy-label", prediction, context)
