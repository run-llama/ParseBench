"""Tests for the PyMuPDF4LLM layout label mapper."""

import pytest

from parse_bench.evaluation.layout_label_mappers.base import MappingContext
from parse_bench.evaluation.layout_label_mappers.mappers import PyMuPDF4LLMLabelMapper
from parse_bench.evaluation.layout_label_mappers.registry import (
    list_layout_label_mappers,
    resolve_layout_label_mapper,
)
from parse_bench.layout_label_mapping import UnknownRawLayoutLabelError
from parse_bench.schemas.layout_detection_output import LayoutDetectionModel, LayoutOutput
from parse_bench.schemas.layout_ontology import CanonicalLabel


def _context(*, provider_name: str | None) -> MappingContext:
    layout_output = LayoutOutput(
        example_id="ex",
        pipeline_name="pymupdf4llm_markdown",
        model=LayoutDetectionModel.PYMUPDF4LLM_LAYOUT,
        image_width=100,
        image_height=100,
        predictions=[],
    )
    return MappingContext(
        provider_name=provider_name,
        pipeline_name="pymupdf4llm_markdown",
        model=LayoutDetectionModel.PYMUPDF4LLM_LAYOUT,
        raw_output={},
        layout_output=layout_output,
    )


def test_mapper_registered_for_provider_and_model() -> None:
    keys = list_layout_label_mappers()
    assert "pymupdf4llm" in keys
    assert "model:pymupdf4llm_layout" in keys


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("table", CanonicalLabel.TABLE),
        ("text", CanonicalLabel.TEXT),
        ("Section_Header", CanonicalLabel.SECTION_HEADER),
        ("list_item", CanonicalLabel.LIST_ITEM),
        ("picture", CanonicalLabel.PICTURE),
        ("caption", CanonicalLabel.CAPTION),
    ],
)
def test_maps_known_raw_labels(raw: str, expected: CanonicalLabel) -> None:
    assert PyMuPDF4LLMLabelMapper().to_canonical(raw, None, None) == expected  # type: ignore[arg-type]


def test_unknown_label_raises() -> None:
    with pytest.raises(UnknownRawLayoutLabelError):
        PyMuPDF4LLMLabelMapper().to_canonical("unknown", None, None)  # type: ignore[arg-type]


def test_resolver_selects_mapper_by_provider_or_model() -> None:
    assert isinstance(resolve_layout_label_mapper(_context(provider_name="pymupdf4llm")), PyMuPDF4LLMLabelMapper)
    assert isinstance(resolve_layout_label_mapper(_context(provider_name=None)), PyMuPDF4LLMLabelMapper)
