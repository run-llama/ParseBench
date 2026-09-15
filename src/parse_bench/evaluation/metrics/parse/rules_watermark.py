"""Watermark-removal rule with an explicit content-preservation guard."""

from __future__ import annotations

import re
from typing import Any, cast

from rapidfuzz import fuzz

from parse_bench.evaluation.metrics.parse.rules_base import ParseTestRule
from parse_bench.evaluation.metrics.parse.utils import normalize_text
from parse_bench.test_cases.parse_rule_schemas import ParseWatermarkRemovalRule


def _partial_similarity(needle: str, content: str) -> float:
    normalized_needle = normalize_text(needle).casefold().strip()
    normalized_content = normalize_text(content).casefold().strip()
    if not normalized_needle or not normalized_content:
        return 0.0
    return float(fuzz.partial_ratio(normalized_needle, normalized_content) / 100.0)


def _occurrence_count(needle: str, content: str) -> int:
    """Count normalized, non-overlapping literal phrase occurrences."""
    normalized_needle = normalize_text(needle).casefold().strip()
    normalized_content = normalize_text(content).casefold().strip()
    if not normalized_needle or not normalized_content:
        return 0
    pattern = re.escape(normalized_needle).replace(r"\ ", r"\s+")
    return sum(1 for _ in re.finditer(pattern, normalized_content))


class WatermarkRemovalRule(ParseTestRule):
    """Require watermark text to disappear while sampled body anchors survive."""

    def __init__(self, rule_data: ParseWatermarkRemovalRule | dict[str, Any]):
        super().__init__(rule_data)
        if not isinstance(self._rule_data, ParseWatermarkRemovalRule):
            raise TypeError("watermark_removal requires ParseWatermarkRemovalRule")

    def run(
        self,
        md_content: str,
        normalized_content: str | None = None,
    ) -> tuple[bool, str, float]:
        rule = cast(ParseWatermarkRemovalRule, self._rule_data)
        actual_markdown = md_content
        if self.parse_output is not None:
            for markdown_page in self.parse_output.pages:
                if markdown_page.page_index + 1 == rule.page:
                    actual_markdown = markdown_page.markdown
                    break

        allowed_occurrences = rule.allowed_occurrences or [0] * len(rule.watermark_texts)
        watermark_matches: list[dict[str, Any]] = [
            {
                "text": text,
                "similarity": _partial_similarity(text, actual_markdown),
                "occurrences": _occurrence_count(text, actual_markdown),
                "allowed_occurrences": allowed,
            }
            for text, allowed in zip(rule.watermark_texts, allowed_occurrences, strict=True)
        ]
        for match in watermark_matches:
            # Legacy rules have no allowed body occurrences and retain fuzzy
            # leak detection. Occurrence-aware rules distinguish a removed
            # overlay from legitimate copies of the same phrase in body text.
            if rule.allowed_occurrences is None:
                match["removed"] = match["similarity"] < rule.watermark_match_threshold
            else:
                match["removed"] = match["occurrences"] <= match["allowed_occurrences"]

        preservation_matches: list[dict[str, Any]] = [
            {
                "text": text,
                "similarity": _partial_similarity(text, actual_markdown),
            }
            for text in rule.preserve_texts
        ]
        for match in preservation_matches:
            match["preserved"] = match["similarity"] >= rule.preserve_match_threshold

        removal_score = sum(bool(match["removed"]) for match in watermark_matches) / len(watermark_matches)
        preservation_score = sum(bool(match["preserved"]) for match in preservation_matches) / len(preservation_matches)
        combined_score = min(removal_score, preservation_score)
        passed = removal_score >= rule.removal_pass_threshold and preservation_score >= rule.preservation_pass_threshold

        self.result_details = {
            "removal_score": removal_score,
            "preservation_score": preservation_score,
            "watermark_match_threshold": rule.watermark_match_threshold,
            "preserve_match_threshold": rule.preserve_match_threshold,
            "watermark_matches": watermark_matches,
            "preservation_matches": preservation_matches,
        }
        explanation = (
            f"watermark_removal={removal_score:.4f} "
            f"body_preservation={preservation_score:.4f} "
            f"removed={sum(bool(match['removed']) for match in watermark_matches)}/{len(watermark_matches)} "
            f"preserved={sum(bool(match['preserved']) for match in preservation_matches)}/{len(preservation_matches)}"
        )
        return passed, explanation, combined_score


__all__ = ["WatermarkRemovalRule"]
