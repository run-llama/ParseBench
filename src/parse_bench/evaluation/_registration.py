"""Shared replacement semantics for keyed layout registrations."""

from collections.abc import Sequence
from typing import Protocol


class _KeyedRegistration(Protocol):
    @property
    def keys(self) -> tuple[str, ...]: ...


def updated_registrations[RegistrationT: _KeyedRegistration](
    registrations: Sequence[RegistrationT],
    replacement: RegistrationT,
    *,
    dangerous_override: bool,
) -> Sequence[RegistrationT]:
    """Replace at the first matched position, retaining unrelated tie order."""
    requested = set(replacement.keys)
    updated: list[RegistrationT] = []
    inserted = False
    for entry in registrations:
        overlap = requested.intersection(entry.keys)
        if len(overlap) == 0:
            updated.append(entry)
            continue
        if not dangerous_override:
            raise ValueError(f"Layout implementation already registered for keys {sorted(overlap)}")
        omitted = set(entry.keys) - requested
        if len(omitted) > 0:
            raise ValueError(f"Override must include all aliases of the replaced registration: {sorted(omitted)}")
        if not inserted:
            updated.append(replacement)
            inserted = True
    if not inserted:
        updated.append(replacement)
    return updated
