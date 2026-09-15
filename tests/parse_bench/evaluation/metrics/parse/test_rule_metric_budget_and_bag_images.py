"""Focused regressions for ported fixes that had no dedicated internal test.

- ``rules_bag``: markdown image alt text is stripped from BOTH the word-bag and
  the sentence-bag extraction paths.
- ``rule_based_metric``: ``normalize_text`` runs under the per-rule timeout and
  the aggregate per-document budget skips (scores 0) the remaining rules.
- ``utils.strip_dot_leaders``: the trailing dot-leader strip is linear, so a
  degenerate OCR page of dots cannot stall a worker.
"""

from __future__ import annotations

import time

import pytest

from parse_bench.evaluation.metrics.parse import rule_based_metric
from parse_bench.evaluation.metrics.parse.rule_based_metric import RuleBasedMetric, _RuleTimeoutError
from parse_bench.evaluation.metrics.parse.rules_bag import SentenceBagRule, WordBagRule
from parse_bench.evaluation.metrics.parse.utils import normalize_cell_text, strip_dot_leaders

_IMG_MD = "Real prose sentence here. ![spurious alt caption](figure.png) Another sentence."


def test_word_bag_strips_markdown_image_alt_text() -> None:
    words = WordBagRule._extract_normalized_words_static(_IMG_MD)
    assert "real" in words
    assert "spurious" not in words
    assert "caption" not in words


def test_sentence_bag_strips_markdown_image_alt_text() -> None:
    sentences = SentenceBagRule._extract_normalized_sentences_static(_IMG_MD)
    joined = " ".join(sentences)
    assert "real prose sentence here" in joined
    assert "spurious" not in joined


def _present_rules(n: int) -> list[dict]:
    return [{"type": "present", "text": f"word{i}"} for i in range(n)]


def test_document_rule_budget_skips_remaining_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    # A vanishingly small budget is exhausted after the first rule; the rest are
    # skipped with a score of 0 and an explanation, never evaluated.
    monkeypatch.setenv("BENCH_DOC_RULE_BUDGET_SECONDS", "1e-9")
    result = RuleBasedMetric().compute(expected=_present_rules(3), actual="word0 word1 word2")
    meta = result.metadata
    assert meta["total"] == 3
    assert len(meta["rule_results"]) == 3
    assert meta["skipped_over_budget"] >= 1
    skipped = [r for r in meta["rule_results"] if r["explanation"].startswith("Skipped: document rule budget")]
    assert len(skipped) == meta["skipped_over_budget"]
    assert all(r["score"] == 0.0 and r["passed"] is False for r in skipped)


def test_document_rule_budget_disabled_when_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_DOC_RULE_BUDGET_SECONDS", "0")
    result = RuleBasedMetric().compute(expected=_present_rules(3), actual="word0 word1 word2")
    assert result.value == 1.0
    assert result.metadata["skipped_over_budget"] == 0


def test_normalize_timeout_skips_whole_document(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_: str) -> str:
        raise _RuleTimeoutError()

    monkeypatch.setattr(rule_based_metric, "normalize_text", _boom)
    result = RuleBasedMetric().compute(expected=_present_rules(2), actual="word0 word1")
    assert result.value == 0.0
    meta = result.metadata
    assert meta["total"] == 2
    assert meta["skipped_over_budget"] == 2
    assert "normalization exceeded" in meta["note"]
    assert len(meta["rule_results"]) == 2
    assert all(r["score"] == 0.0 for r in meta["rule_results"])


def test_trailing_dot_leader_strip_is_linear() -> None:
    page = "Item " + "." * 130_000
    start = time.monotonic()
    assert strip_dot_leaders(page).rstrip(". ") == "Item"
    normalize_cell_text(page)
    assert time.monotonic() - start < 2.0
