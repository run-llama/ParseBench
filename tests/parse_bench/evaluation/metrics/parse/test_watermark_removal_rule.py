import pytest
from pydantic import ValidationError

from parse_bench.evaluation.metrics.parse.rules_watermark import WatermarkRemovalRule


def _rule() -> WatermarkRemovalRule:
    return WatermarkRemovalRule(
        {
            "type": "watermark_removal",
            "id": "watermark-1",
            "page": 1,
            "watermark_texts": ["CONFIDENTIAL DRAFT"],
            "preserve_texts": [
                "Quarterly operating results",
                "Revenue increased during the period",
                "The board approved the proposal",
                "Notes to the financial statements",
                "Authorized representative",
            ],
            "watermark_match_threshold": 0.8,
            "preserve_match_threshold": 0.85,
            "removal_pass_threshold": 1.0,
            "preservation_pass_threshold": 0.8,
        }
    )


def test_watermark_rule_passes_when_mark_is_removed_and_body_survives() -> None:
    rule = _rule()

    passed, explanation, score = rule.run(
        "Quarterly operating results\nRevenue increased during the period\n"
        "The board approved the proposal\nNotes to the financial statements"
    )

    assert passed is True
    assert score == 0.8
    assert "removed=1/1" in explanation
    assert "preserved=4/5" in explanation
    assert rule.result_details["removal_score"] == 1.0
    assert rule.result_details["preservation_score"] == 0.8


def test_watermark_rule_fails_when_watermark_leaks() -> None:
    rule = _rule()

    passed, _, score = rule.run(
        "CONFIDENTIAL DRAFT\nQuarterly operating results\nRevenue increased during the period\n"
        "The board approved the proposal\nNotes to the financial statements\nAuthorized representative"
    )

    assert passed is False
    assert score == 0.0
    assert rule.result_details["watermark_matches"][0]["removed"] is False


def test_watermark_rule_fails_when_body_is_destroyed() -> None:
    rule = _rule()

    passed, _, score = rule.run("Quarterly operating results")

    assert passed is False
    assert score == 0.2
    assert rule.result_details["removal_score"] == 1.0
    assert rule.result_details["preservation_score"] == 0.2


def test_watermark_rule_allows_legitimate_body_occurrences_but_rejects_overlay_leak() -> None:
    payload = _rule()._rule_data.model_dump()
    payload["watermark_texts"] = ["DRAFT"]
    payload["allowed_occurrences"] = [1]
    rule = WatermarkRemovalRule(payload)

    passed, _, _ = rule.run(
        "The DRAFT policy remains under review.\nQuarterly operating results\n"
        "Revenue increased during the period\nThe board approved the proposal\n"
        "Notes to the financial statements\nAuthorized representative"
    )
    assert passed is True
    assert rule.result_details["watermark_matches"][0]["occurrences"] == 1

    passed, _, _ = rule.run(
        "DRAFT\nThe DRAFT policy remains under review.\nQuarterly operating results\n"
        "Revenue increased during the period\nThe board approved the proposal\n"
        "Notes to the financial statements\nAuthorized representative"
    )
    assert passed is False
    assert rule.result_details["watermark_matches"][0]["occurrences"] == 2


def test_watermark_rule_rejects_misaligned_occurrence_limits() -> None:
    payload = _rule()._rule_data.model_dump()
    payload["allowed_occurrences"] = [0, 1]
    with pytest.raises(ValidationError, match="one value per watermark_texts"):
        WatermarkRemovalRule(payload)
