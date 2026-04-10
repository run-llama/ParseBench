"""Schema definitions for test cases."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator

from parse_bench.test_cases.parse_rule_schemas import (
    ParseRule,
    coerce_parse_rule,
    coerce_parse_rule_list_or_none,
)

# =============================================================================
# Layout Content Types
# =============================================================================


class LayoutTextContent(BaseModel):
    """Text content for layout elements (paragraphs, headers, captions, etc.)."""

    type: Literal["text"] = "text"
    text: str = Field(description="Aggregated text content from PDF cells")


class LayoutTableContent(BaseModel):
    """Table content with HTML representation."""

    type: Literal["table"] = "table"
    html: str = Field(description="HTML table representation")


# Union type for layout content with discriminator
LayoutContent = Annotated[
    Annotated[LayoutTextContent, Tag("text")] | Annotated[LayoutTableContent, Tag("table")],
    Discriminator("type"),
]


# =============================================================================
# Test Case Schemas
# =============================================================================


class QAConfig(BaseModel):
    """Configuration for question-answering test cases."""

    question: str = Field(description="The question text to be answered")
    answer: str = Field(description="Expected answer(s)")
    question_type: str = Field(description="Type of question: 'single_choice', 'multiple_choice', or 'numerical'")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional QA metadata (options, tolerance, etc.)"
    )


class LayoutAnnotation(BaseModel):
    """Single layout region annotation."""

    id: str | None = Field(
        default=None,
        description="Optional stable identifier for this layout element",
    )
    bbox: list[float] = Field(description="Bounding box [x, y, width, height] in COCO format")
    canonical_class: str = Field(description="Canonical class name from ontology")
    page: int = Field(default=0, description="Page index (0-based) for multi-page documents")
    attributes: dict[str, str | bool] = Field(
        default_factory=dict, description="Optional attributes that refine the class"
    )
    source_label: str | None = Field(default=None, description="Original label from source dataset")
    source_category_id: int | None = Field(default=None, description="Original category ID from source dataset")
    content: LayoutContent | None = Field(
        default=None,
        description="Attributed content for this layout element",
    )
    ro_index: int | None = Field(
        default=None,
        description="Reading order index (0-based position in reading sequence)",
    )
    tags: list[str] = Field(default_factory=list, description="Optional per-rule tags")


class LayoutTestRule(BaseModel):
    """Layout annotation as test rule with normalized bbox."""

    type: Literal["layout"] = "layout"
    id: str | None = Field(
        default=None,
        description="Optional stable identifier for this layout element",
    )
    page: int = Field(ge=1, description="Page number (1-indexed)")
    bbox: list[float] = Field(description="Normalized bbox [x, y, w, h] in [0,1] range (COCO format)")
    canonical_class: str = Field(description="Class from ontology")
    attributes: dict[str, str | bool] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, description="Optional per-rule tags")
    source_label: str | None = Field(default=None)
    source_category_id: int | None = Field(default=None)
    content: LayoutContent | None = Field(
        default=None,
        description="Attributed content (text or table HTML)",
    )
    ro_index: int | None = Field(
        default=None,
        description="Reading order index (0-based position in reading sequence)",
    )

    def to_layout_annotation(self) -> LayoutAnnotation:
        """Convert to LayoutAnnotation (coordinates remain normalized)."""
        return LayoutAnnotation(
            id=self.id,
            bbox=self.bbox,  # Keep normalized
            canonical_class=self.canonical_class,
            page=self.page - 1,  # Convert to 0-indexed for internal use
            attributes=self.attributes,
            source_label=self.source_label,
            source_category_id=self.source_category_id,
            content=self.content,
            ro_index=self.ro_index,
            tags=self.tags,
        )


class BaseTestCase(BaseModel):
    """Base test case with common fields."""

    model_config = ConfigDict(populate_by_name=True)

    test_id: str = Field(description="Unique identifier (e.g., 'group/pdf_name')")
    group: str = Field(description="Group/subfolder name")
    file_path: Path = Field(description="Path to the input file (PDF, image, etc.)")
    tags: list[str] = Field(
        default_factory=list,
        description="Optional top-level tags for document-level provenance, filtering, and grouping.",
    )


class ParseTestCase(BaseTestCase):
    """Test case for PARSE product type."""

    test_rules: list[ParseRule] | None = Field(
        default=None,
        description="List of rule-based test definitions (from test.json test_rules)",
    )
    expected_markdown: str | None = Field(
        default=None,
        description="Ground truth markdown for comparison (from test.json expected_markdown)",
    )
    qa_config: QAConfig | None = Field(
        default=None,
        description=(
            "Optional QA configuration for question-answering evaluation (from test.json question/answer fields)"
        ),
    )
    qa_configs: list[QAConfig] | None = Field(
        default=None,
        description=(
            "Optional list of QA configurations (from test.json qa_configs array). "
            "Expanded into per-question evaluation tasks by the evaluation runner."
        ),
    )
    allow_splitting_ambiguous_merged_tables: bool = Field(
        default=False,
        description=(
            "When True, try splitting a merged predicted table to match GT table "
            "structure for ambiguous side-by-side layouts"
        ),
    )
    trm_unsupported: bool = Field(
        default=False,
        description=(
            "When True, the table_record_match metric is unreliable for this "
            "document's tables; grits_trm_composite falls back to grits_con."
        ),
    )
    max_top_title_rows: int = Field(
        default=1,
        description=(
            "Cap on how many top <th> spanning title rows the strip stage "
            "removes from the header block of each table (both GT and "
            "predicted) before any table metric runs. Default 1 matches "
            "today's behavior. Set to 0 to disable all top-of-table "
            "stripping (no leading <td> title rows, no top <th> titles). "
            "A rowspanning title row consumes 1 slot from the cap "
            "regardless of its original rowspan. The bottom-<th> title "
            "strip is independent and not governed by this field."
        ),
    )

    @field_validator("test_rules", mode="before")
    @classmethod
    def _coerce_parse_rules(cls, value: list[dict[str, Any]] | None) -> list[ParseRule] | None:
        if value is None:
            return None
        return coerce_parse_rule_list_or_none(value)


class LayoutDetectionTestCase(BaseTestCase):
    """Test case for LAYOUT_DETECTION product type.

    Layout annotations are stored in test_rules with type="layout" and normalized
    bounding boxes in [0,1] range. This enables unified loading with other test types.
    """

    test_rules: list[LayoutTestRule | ParseRule] = Field(
        default_factory=list,
        description="Test rules including layout annotations (type='layout')",
    )
    ontology: str | None = Field(
        default=None,
        description="Target ontology for evaluation (e.g., basic, canonical)",
    )
    source_ontology: str | None = Field(
        default=None,
        description="Original ontology of the source dataset (if provided)",
    )
    source_dataset: str | None = Field(default=None, description="Source dataset identifier (e.g., 'DocLayNet-v1.2')")
    source_id: str | None = Field(default=None, description="Original ID in source dataset")
    page_index: int = Field(default=0, description="Page index for single-page test (0-indexed)")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata (doc_category, complexity_rank, etc.)"
    )

    @field_validator("test_rules", mode="before")
    @classmethod
    def _coerce_layout_or_parse_rules(cls, value: list[dict[str, Any]] | None) -> list[LayoutTestRule | ParseRule]:
        if value is None:
            return []

        typed_rules: list[LayoutTestRule | ParseRule] = []
        for rule in value:
            if isinstance(rule, (LayoutTestRule,)):
                typed_rules.append(rule)
                continue

            if isinstance(rule, dict):
                rule_type = rule.get("type")
                if rule_type == "layout":
                    typed_rules.append(LayoutTestRule.model_validate(rule))
                else:
                    typed_rules.append(coerce_parse_rule(rule))
                continue

            typed_rules.append(coerce_parse_rule(rule))

        return typed_rules

    def get_layout_rules(self, page: int | None = None) -> list[LayoutTestRule]:
        """Get layout test rules, optionally filtered by page.

        :param page: 1-indexed page number to filter by (None for all pages)
        :return: List of LayoutTestRule objects
        """
        layout_rules: list[LayoutTestRule] = []
        for rule in self.test_rules:
            if isinstance(rule, LayoutTestRule):
                if page is None or rule.page == page:
                    layout_rules.append(rule)
        return layout_rules

    def get_layout_annotations(self, page: int | None = None) -> list[LayoutAnnotation]:
        """Get layout annotations as LayoutAnnotation objects.

        :param page: 1-indexed page number to filter by (None for all pages)
        :return: List of LayoutAnnotation objects (with normalized coordinates)
        """
        return [rule.to_layout_annotation() for rule in self.get_layout_rules(page)]

    def get_page_indices(self) -> list[int]:
        """Get list of unique page indices (0-indexed) with layout annotations."""
        pages: set[int] = set()
        for rule in self.test_rules:
            if isinstance(rule, LayoutTestRule):
                pages.add(rule.page - 1)  # Convert to 0-indexed
        return sorted(pages)


# Union type for backward compatibility
TestCase = ParseTestCase | LayoutDetectionTestCase
