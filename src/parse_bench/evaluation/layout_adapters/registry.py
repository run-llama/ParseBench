"""Decorator-driven registry for layout adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from parse_bench.evaluation.layout_adapters.base import LayoutAdapter
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.schemas.pipeline_io import InferenceResult

_DEFAULT_LAYOUT_ADAPTER_KEY = "__default__"


@dataclass(frozen=True)
class _LayoutAdapterRegistration:
    keys: tuple[str, ...]
    priority: int
    adapter_cls: type[LayoutAdapter]


_LAYOUT_ADAPTER_REGISTRY: list[_LayoutAdapterRegistration] = []


def register_layout_adapter(
    *provider_keys: str,
    priority: int = 0,
) -> Callable[[type[LayoutAdapter]], type[LayoutAdapter]]:
    """Register a layout adapter class for one or more provider keys."""
    if not provider_keys:
        raise ValueError("register_layout_adapter requires at least one provider key")

    def decorator(cls: type[LayoutAdapter]) -> type[LayoutAdapter]:
        existing_keys = {key for entry in _LAYOUT_ADAPTER_REGISTRY for key in entry.keys}
        for key in provider_keys:
            if key in existing_keys:
                raise ValueError(f"Layout adapter already registered for provider key '{key}'")

        _LAYOUT_ADAPTER_REGISTRY.append(
            _LayoutAdapterRegistration(
                keys=tuple(provider_keys),
                priority=priority,
                adapter_cls=cls,
            )
        )
        return cls

    return decorator


def list_layout_adapters() -> list[str]:
    """List all registered adapter keys."""
    keys = {key for registration in _LAYOUT_ADAPTER_REGISTRY for key in registration.keys}
    return sorted(keys)


def resolve_layout_provider_name(inference_result: InferenceResult) -> str | None:
    """Resolve provider key from pipeline metadata when available."""
    try:
        pipeline_spec = get_pipeline(inference_result.pipeline_name)
        return pipeline_spec.provider_name
    except Exception:
        return None


def create_layout_adapter(provider_name: str | None) -> LayoutAdapter:
    """Instantiate a layout adapter for an optional provider key."""
    if provider_name is not None:
        candidates = [registration for registration in _LAYOUT_ADAPTER_REGISTRY if provider_name in registration.keys]
        if candidates:
            chosen = sorted(candidates, key=lambda entry: entry.priority, reverse=True)[0]
            return chosen.adapter_cls()

    default_candidates = [
        registration for registration in _LAYOUT_ADAPTER_REGISTRY if _DEFAULT_LAYOUT_ADAPTER_KEY in registration.keys
    ]
    if default_candidates:
        chosen = sorted(
            default_candidates,
            key=lambda entry: entry.priority,
            reverse=True,
        )[0]
        return chosen.adapter_cls()

    available = ", ".join(list_layout_adapters())
    raise ValueError(f"No layout adapter registered for provider '{provider_name}'. Available adapters: {available}")


def create_layout_adapter_for_result(inference_result: InferenceResult) -> LayoutAdapter:
    """Resolve and instantiate adapter using provider key first, matcher fallback second."""
    provider_name = resolve_layout_provider_name(inference_result)
    if provider_name is not None:
        try:
            return create_layout_adapter(provider_name)
        except ValueError:
            pass

    candidates: list[_LayoutAdapterRegistration] = []
    for registration in _LAYOUT_ADAPTER_REGISTRY:
        if _DEFAULT_LAYOUT_ADAPTER_KEY in registration.keys:
            continue
        if registration.adapter_cls.matches(inference_result):
            candidates.append(registration)

    if candidates:
        chosen = sorted(candidates, key=lambda entry: entry.priority, reverse=True)[0]
        return chosen.adapter_cls()

    return create_layout_adapter(_DEFAULT_LAYOUT_ADAPTER_KEY)
