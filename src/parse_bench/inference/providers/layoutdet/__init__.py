"""Layout detection providers for inference endpoints."""

# Import providers to register them
from parse_bench.inference.providers.layoutdet import (
    chandra,  # noqa: F401
    docling,  # noqa: F401
    dots_ocr,  # noqa: F401
    layout_v3,  # noqa: F401
    paddle,  # noqa: F401
    qwen3vl,  # noqa: F401
    surya,  # noqa: F401
    yolo,  # noqa: F401
)

__all__ = [
    "chandra",
    "docling",
    "dots_ocr",
    "layout_v3",
    "paddle",
    "qwen3vl",
    "surya",
    "yolo",
]
