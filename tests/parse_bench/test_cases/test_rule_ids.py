from __future__ import annotations

from typing import Any

from parse_bench.test_cases.rule_ids import (
    assign_deterministic_ids,
    canonical_rule_signature,
    compute_rule_id,
)


def _layout_rule(**overrides: Any) -> dict[str, Any]:
    rule: dict[str, Any] = {
        "type": "layout",
        "page": 1,
        "bbox": [0.1, 0.2, 0.3, 0.05],
        "canonical_class": "Text",
        "content": {"type": "text", "text": "hello"},
    }
    rule.update(overrides)
    return rule


def test_assign_deterministic_ids_assigns_stable_ids() -> None:
    rules = [_layout_rule(), _layout_rule(page=2, content={"type": "text", "text": "world"})]
    assign_deterministic_ids(rules, hash_len=16)

    assert rules[0]["id"] == compute_rule_id(rules[0], 16)
    assert rules[1]["id"] == compute_rule_id(rules[1], 16)
    assert rules[0]["id"] != rules[1]["id"]


def test_assign_deterministic_ids_disambiguates_collisions() -> None:
    rules = [_layout_rule(), _layout_rule()]
    assign_deterministic_ids(rules, hash_len=16)

    base_id = compute_rule_id(rules[0], 16)
    assert rules[0]["id"] == f"000-{base_id}"
    assert rules[1]["id"] == f"001-{base_id}"


def test_assign_deterministic_ids_respects_reserved_ids() -> None:
    rules = [_layout_rule()]
    base_id = compute_rule_id(rules[0], 16)

    assign_deterministic_ids(rules, hash_len=16, reserved_ids={base_id})

    assert rules[0]["id"] == f"000-{base_id}"


def test_canonical_rule_signature_ignores_verified_flag() -> None:
    """``verified`` is review state, not content, so it must not change identity.

    A machine-generated rule enters the queue as ``verified: false`` and comes
    back from the annotator as ``verified: true``.  If that flipped the
    signature, the merge's dedup would treat the accepted rule as new and append
    a duplicate of every rule a human had already signed off.
    """
    absent = _layout_rule()
    unverified = _layout_rule(verified=False)
    verified = _layout_rule(verified=True)

    signature = canonical_rule_signature(absent)
    assert canonical_rule_signature(unverified) == signature
    assert canonical_rule_signature(verified) == signature
    assert "verified" not in signature

    assert compute_rule_id(unverified, 16) == compute_rule_id(verified, 16)
    assert compute_rule_id(absent, 16) == compute_rule_id(verified, 16)


def test_canonical_rule_signature_still_separates_real_content() -> None:
    """The verified exemption must not blunt the signature for real differences."""
    assert canonical_rule_signature(_layout_rule(verified=True)) != canonical_rule_signature(
        _layout_rule(verified=False, content={"type": "text", "text": "different"})
    )


def test_assign_deterministic_ids_ignores_verified_flag() -> None:
    rules = [_layout_rule(verified=False), _layout_rule(verified=True)]
    assign_deterministic_ids(rules, hash_len=16)

    # Same content, so they collide and get the collision prefixes — which is
    # exactly the point: the ``verified`` flag did not make them distinct.
    base_id = compute_rule_id(_layout_rule(), 16)
    assert rules[0]["id"] == f"000-{base_id}"
    assert rules[1]["id"] == f"001-{base_id}"


def test_assign_deterministic_ids_honors_exclude_keys() -> None:
    rules = [
        _layout_rule(parent_test_id="parent-a"),
        _layout_rule(parent_test_id="parent-b"),
    ]
    assign_deterministic_ids(rules, hash_len=16, exclude_keys={"parent_test_id"})

    without_parent = _layout_rule()
    expected_base = compute_rule_id(without_parent, 16)
    assert rules[0]["id"] == f"000-{expected_base}"
    assert rules[1]["id"] == f"001-{expected_base}"
