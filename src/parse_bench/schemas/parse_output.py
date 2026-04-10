"""
Normalized output schema for parse tasks.
"""

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class PageIR(BaseModel):
    """Intermediate representation of a single page."""

    page_index: int = Field(ge=0, description="0-indexed page number")
    markdown: str = Field(description="Markdown content of the page")


class LayoutSegmentIR(BaseModel):
    """Normalized layout segment coordinates and attribution span metadata."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    x: float
    y: float
    w: float
    h: float
    confidence: float | None = None
    label: str | None = None
    start_index: int | None = Field(
        default=None,
        validation_alias=AliasChoices("start_index", "startIndex"),
    )
    end_index: int | None = Field(
        default=None,
        validation_alias=AliasChoices("end_index", "endIndex"),
    )


class LayoutItemIR(BaseModel):
    """Normalized layout item used for attribution/layout reconstruction."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    type: str = "text"
    md: str = Field(
        default="",
        validation_alias=AliasChoices("md", "markdown"),
    )
    html: str = ""
    value: str = ""
    bbox: LayoutSegmentIR | None = Field(
        default=None,
        validation_alias=AliasChoices("bbox", "bBox"),
    )
    layout_segments: list[LayoutSegmentIR] = Field(
        default_factory=list,
        validation_alias=AliasChoices("layout_segments", "layoutAwareBbox"),
    )

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, value: object) -> str:
        if value is None:
            return "text"
        return str(value)

    @field_validator("md", "html", "value", mode="before")
    @classmethod
    def _normalize_value(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)


class ParseLayoutPageIR(BaseModel):
    """Normalized per-page layout payload embedded in ParseOutput."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    page_number: int = Field(
        ge=1,
        validation_alias=AliasChoices("page_number", "page"),
    )
    width: float | None = Field(
        default=None,
        validation_alias=AliasChoices("width", "page_width"),
    )
    height: float | None = Field(
        default=None,
        validation_alias=AliasChoices("height", "page_height"),
    )
    md: str = ""
    text: str = ""
    page_header_markdown: str = Field(
        default="",
        validation_alias=AliasChoices("page_header_markdown", "pageHeaderMarkdown"),
    )
    page_footer_markdown: str = Field(
        default="",
        validation_alias=AliasChoices("page_footer_markdown", "pageFooterMarkdown"),
    )
    printed_page_number: str = Field(
        default="",
        validation_alias=AliasChoices("printed_page_number", "printedPageNumber"),
    )
    original_orientation_angle: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "original_orientation_angle",
            "originalOrientationAngle",
        ),
    )
    items: list[LayoutItemIR] = Field(default_factory=list)

    @field_validator(
        "md",
        "text",
        "page_header_markdown",
        "page_footer_markdown",
        "printed_page_number",
        mode="before",
    )
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)


class ParseOutput(BaseModel):
    """Normalized output for parse tasks."""

    task_type: Literal["parse"] = Field(default="parse", frozen=True, description="Task type discriminator")
    example_id: str = Field(description="Unique identifier for the example")
    pipeline_name: str = Field(description="Name of the pipeline that produced this output")
    pages: list[PageIR] = Field(default_factory=list, description="List of parsed pages")
    layout_pages: list[ParseLayoutPageIR] = Field(
        default_factory=list,
        description=("Normalized page/item/segment layout payload used by layout attribution and overlays"),
    )
    markdown: str = Field(description="Markdown content of the entire document")
    job_id: str | None = Field(default=None, description="Optional job ID from the provider (e.g., LlamaParse job ID)")
