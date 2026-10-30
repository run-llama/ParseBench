"""Layout detection providers, imported on first use for their registry side effects."""

import importlib
import logging

logger = logging.getLogger(__name__)

_PROVIDER_MODULES = [
    "chandra",
    "docling",
    "dots_ocr",
    "layout_v3",
    "paddle",
    "qwen3vl",
    "surya",
    "yolo",
]


def load_providers() -> None:
    """Import each provider module so it registers itself; skip ones whose SDK is missing.

    Called by ``registry.load_builtin_providers`` on first use rather than on package
    import, so helpers under this package (such as label adapters) stay cheap to import.
    """
    for module in _PROVIDER_MODULES:
        try:
            importlib.import_module(f"parse_bench.inference.providers.layoutdet.{module}")
        except ImportError:
            logger.debug("Skipping layout provider %s (missing dependency)", module)


__all__ = _PROVIDER_MODULES
