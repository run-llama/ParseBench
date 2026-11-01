"""Shared replacement semantics for keyed layout registrations."""

from collections.abc import Sequence
from typing import Protocol


class _KeyedRegistration(Protocol):
    @property
    def keys(self) -> tuple[str, ...]: ...


def retain_registrations[RegistrationT: _KeyedRegistration](
    registrations: Sequence[RegistrationT],
    keys: Sequence[str],
    *,
    dangerous_override: bool,
) -> list[RegistrationT]:
    """Validate conflicts before returning registrations retained by an update.

    A registration owns both its aliases and its shape matcher. Replacing all
    its keys together removes the old matcher without orphaning an alias.
    """
    requested = set(keys)
    retained: list[RegistrationT] = []
    for entry in registrations:
        overlap = requested.intersection(entry.keys)
        if len(overlap) == 0:
            retained.append(entry)
            continue
        if not dangerous_override:
            raise ValueError(f"Layout implementation already registered for keys {sorted(overlap)}")
        omitted = set(entry.keys) - requested
        if len(omitted) > 0:
            raise ValueError(f"Override must include all aliases of the replaced registration: {sorted(omitted)}")
    return retained
