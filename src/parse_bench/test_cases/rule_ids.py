"""Shared helpers for deterministic test-rule identifiers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

# Fields that describe a rule's *review state* rather than its content, and so
# must never take part in the identity of a rule.
#
# ``verified`` is the human sign-off marker: machine-generated rules arrive as
# ``verified: false`` and an annotator flips them to ``true``.  Hashing it would
# make the same assertion hash three different ways (absent / false / true), and
# because the merge dedups incoming rules against a dataset by this signature,
# a re-run of the same document would append duplicates of every rule a human
# had already accepted.
_NON_CONTENT_KEYS = ("id", "verified")


def canonical_rule_signature(rule: dict[str, Any]) -> str:
    """Return a canonical JSON signature for a rule's content.

    Excludes the rule's own ``id`` (which is derived from this signature) and
    its ``verified`` review flag, so a rule keeps one identity across the
    machine-generated → human-verified transition.
    """
    payload = dict(rule)
    for key in _NON_CONTENT_KEYS:
        payload.pop(key, None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_rule_id(rule: dict[str, Any], hash_len: int) -> str:
    """Compute the deterministic rule id used by `scripts/assign_rule_ids.py`."""
    signature = canonical_rule_signature(rule)
    page = rule.get("page")
    page_prefix = str(page) if page is not None else ""
    payload = f"{page_prefix}\u0000{signature}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:hash_len]


def _canonical_signature(rule: dict[str, Any], *, exclude_keys: set[str]) -> str:
    if not exclude_keys:
        return canonical_rule_signature(rule)

    payload = dict(rule)
    payload.pop("id", None)
    for key in exclude_keys:
        payload.pop(key, None)
    return canonical_rule_signature(payload)


def _compute_rule_id_with_excludes(
    rule: dict[str, Any],
    *,
    hash_len: int,
    exclude_keys: set[str],
) -> str:
    if not exclude_keys:
        return compute_rule_id(rule, hash_len)

    payload = dict(rule)
    payload.pop("id", None)
    for key in exclude_keys:
        payload.pop(key, None)
    return compute_rule_id(payload, hash_len)


def assign_deterministic_ids(
    rules: list[dict[str, Any]],
    *,
    hash_len: int,
    reserved_ids: set[str] | None = None,
    exclude_keys: set[str] | None = None,
) -> None:
    """Assign deterministic ``id`` fields in-place, disambiguating collisions."""
    if not rules:
        return

    reserved = set(reserved_ids or set())
    excluded = set(exclude_keys or set())
    indexed_rules: list[tuple[int, dict[str, Any], str]] = [
        (index, rule, _canonical_signature(rule, exclude_keys=excluded)) for index, rule in enumerate(rules)
    ]

    by_base_id: dict[str, list[tuple[int, dict[str, Any], str]]] = defaultdict(list)
    for entry in indexed_rules:
        _, rule, _ = entry
        base_id = _compute_rule_id_with_excludes(rule, hash_len=hash_len, exclude_keys=excluded)
        rule["id"] = base_id
        by_base_id[base_id].append(entry)

    for base_id, entries in by_base_id.items():
        if len(entries) == 1 and base_id not in reserved:
            reserved.add(base_id)
            continue

        for prefix_counter, entry in enumerate(sorted(entries, key=lambda item: (item[2], item[0]))):
            entry[1]["id"] = f"{prefix_counter:03d}-{base_id}"
            reserved.add(entry[1]["id"])
