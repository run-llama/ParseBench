"""Extract pipelines - structured data extraction from documents."""

from typing import Any

from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.product import ProductType


def _pipeline_spec(
    *,
    pipeline_name: str,
    provider_name: str,
    config: dict[str, Any],
) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=pipeline_name,
        provider_name=provider_name,
        product_type=ProductType.EXTRACT,
        config=config,
    )


def register_extract_pipelines(register_fn) -> None:  # type: ignore[no-untyped-def]
    """Register the implementation-target extract pipelines."""

    register_fn(
        _pipeline_spec(
            pipeline_name="extend_extract",
            provider_name="extend",
            config={
                "baseProcessor": "extraction_performance",
                "baseVersion": "4.1.1",
                "advancedOptions": {
                    "citationsEnabled": True,
                    "advancedFigureParsingEnabled": True,
                },
            },
        )
    )
