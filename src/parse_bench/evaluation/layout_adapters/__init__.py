"""Layout adapter registry and provider-specific adapter bindings."""

# Import adapters for registration side effects.
from . import adapters as _adapters  # noqa: F401
from .base import LayoutAdapter
from .registry import (
    create_layout_adapter,
    create_layout_adapter_for_result,
    list_layout_adapters,
    register_layout_adapter,
    resolve_layout_provider_name,
)

__all__ = [
    "LayoutAdapter",
    "create_layout_adapter",
    "create_layout_adapter_for_result",
    "list_layout_adapters",
    "register_layout_adapter",
    "resolve_layout_provider_name",
]
