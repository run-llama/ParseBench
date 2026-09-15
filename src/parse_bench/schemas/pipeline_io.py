from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Discriminator, Field, SerializeAsAny

from parse_bench.schemas.extract_output import ExtractOutput
from parse_bench.schemas.layout_detection_output import LayoutOutput
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.product import ProductTypeName


class InferenceRequest(BaseModel):
    """Request for running inference on a document."""

    example_id: str = Field(description="Unique identifier for the example")
    source_file_path: str = Field(description="Path to the source file (PDF, etc.)")
    product_type: ProductTypeName = Field(description="Type of product task to run")
    schema_override: dict[str, Any] | None = Field(
        default=None,
        description="Optional schema override",
    )
    config_override: dict[str, Any] | None = Field(
        default=None,
        description=("Optional configuration override to merge with pipeline config"),
    )


PipelineOutputType = Annotated[
    ParseOutput | LayoutOutput | ExtractOutput,
    Discriminator("task_type"),
]
"""Union of the built-in normalized output models (for type annotations)."""

_OUTPUT_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "parse": ParseOutput,
    "layout_detection": LayoutOutput,
    "extract": ExtractOutput,
}


def register_output_model(task_type: str, model: type[BaseModel]) -> None:
    """Register a normalized output model for ``task_type``.

    ``InferenceResult.output`` dispatches on the ``task_type`` discriminator, so
    extensions that add a product (e.g. a QA task) register the model their
    provider returns here and saved results round-trip through JSON unchanged.
    Re-registering a built-in task type raises ``ValueError``.
    """
    if not isinstance(task_type, str) or not task_type:
        raise ValueError("task_type must be a non-empty string")
    if task_type in {"parse", "layout_detection", "extract"} and _OUTPUT_MODEL_REGISTRY[task_type] is not model:
        raise ValueError(f"{task_type!r} is a built-in output model and cannot be replaced")
    _OUTPUT_MODEL_REGISTRY[task_type] = model


def registered_output_models() -> dict[str, type[BaseModel]]:
    """Snapshot of ``task_type`` -> model for every registered output."""
    return dict(_OUTPUT_MODEL_REGISTRY)


def coerce_pipeline_output(value: Any) -> BaseModel:
    """Validate a normalized output payload against the registered model for its ``task_type``."""
    if isinstance(value, BaseModel):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Pipeline output must be a dict or model, got {type(value).__name__}")
    task_type = value.get("task_type")
    model = _OUTPUT_MODEL_REGISTRY.get(task_type) if isinstance(task_type, str) else None
    if model is None:
        raise ValueError(f"Unknown output task_type {task_type!r}; registered: {sorted(_OUTPUT_MODEL_REGISTRY)}")
    return model.model_validate(value)


PipelineOutput = Annotated[SerializeAsAny[BaseModel], BeforeValidator(coerce_pipeline_output)]
"""Field annotation for a normalized output: dispatches on ``task_type`` via the registry."""


class RawInferenceResult(BaseModel):
    """Raw result from provider before normalization."""

    request: InferenceRequest = Field(description="Original inference request")
    pipeline: PipelineSpec = Field(description="Pipeline used")
    pipeline_name: str = Field(description="Name of the pipeline used")
    product_type: ProductTypeName = Field(description="Type of product task that was run")
    raw_output: dict = Field(description="Raw output from the provider API")
    started_at: datetime = Field(description="Timestamp when inference started")
    completed_at: datetime = Field(description="Timestamp when inference completed")
    latency_in_ms: int = Field(ge=0, description="Latency in milliseconds")


class InferenceResult(BaseModel):
    """Result of running inference on a document with both raw and normalized outputs."""

    request: InferenceRequest = Field(description="Original inference request")
    pipeline_name: str = Field(description="Name of the pipeline used")
    product_type: ProductTypeName = Field(description="Type of product task that was run")

    # Both outputs stored here
    raw_output: dict = Field(description="Raw output from the provider (for debugging/re-normalization)")
    output: PipelineOutput = Field(description="Normalized output from the pipeline")

    # metadata
    started_at: datetime = Field(description="Timestamp when inference started")
    completed_at: datetime = Field(description="Timestamp when inference completed")
    latency_in_ms: int = Field(ge=0, description="Latency in milliseconds")
