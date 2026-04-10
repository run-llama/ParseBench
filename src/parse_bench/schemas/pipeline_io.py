from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Discriminator, Field

from parse_bench.schemas.layout_detection_output import LayoutOutput
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.product import ProductType


class InferenceRequest(BaseModel):
    """Request for running inference on a document."""

    example_id: str = Field(description="Unique identifier for the example")
    source_file_path: str = Field(description="Path to the source file (PDF, etc.)")
    product_type: ProductType = Field(description="Type of product task to run")
    schema_override: dict[str, Any] | None = Field(
        default=None,
        description="Optional schema override",
    )
    config_override: dict[str, Any] | None = Field(
        default=None,
        description=("Optional configuration override to merge with pipeline config"),
    )


PipelineOutputType = Annotated[
    ParseOutput | LayoutOutput,
    Discriminator("task_type"),
]


class RawInferenceResult(BaseModel):
    """Raw result from provider before normalization."""

    request: InferenceRequest = Field(description="Original inference request")
    pipeline: PipelineSpec = Field(description="Pipeline used")
    pipeline_name: str = Field(description="Name of the pipeline used")
    product_type: ProductType = Field(description="Type of product task that was run")
    raw_output: dict = Field(description="Raw output from the provider API")
    started_at: datetime = Field(description="Timestamp when inference started")
    completed_at: datetime = Field(description="Timestamp when inference completed")
    latency_in_ms: int = Field(ge=0, description="Latency in milliseconds")


class InferenceResult(BaseModel):
    """Result of running inference on a document with both raw and normalized outputs."""

    request: InferenceRequest = Field(description="Original inference request")
    pipeline_name: str = Field(description="Name of the pipeline used")
    product_type: ProductType = Field(description="Type of product task that was run")

    # Both outputs stored here
    raw_output: dict = Field(description="Raw output from the provider (for debugging/re-normalization)")
    output: PipelineOutputType = Field(description="Normalized output from the pipeline")

    # metadata
    started_at: datetime = Field(description="Timestamp when inference started")
    completed_at: datetime = Field(description="Timestamp when inference completed")
    latency_in_ms: int = Field(ge=0, description="Latency in milliseconds")
