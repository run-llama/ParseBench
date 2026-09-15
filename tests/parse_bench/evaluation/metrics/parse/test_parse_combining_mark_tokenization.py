"""Combining marks must survive normalization and stay attached to their base.

Regression tests for the Thai/Indic grading defect: annotation-time tokenization
split words at every combining mark while eval-time ``normalize_text`` deleted
those marks, so a stored ground-truth "word" could never equal a document token.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from parse_bench.evaluation.metrics.parse.rules_bag import (
    _tokenize_unicode_words,
    _word_boundary_count,
)
from parse_bench.evaluation.metrics.parse.test_rules import MissingSpecificWordRule
from parse_bench.evaluation.metrics.parse.utils import normalize_text

# One real word per affected script.  Each contains at least one combining mark
# (category Mn or Mc) that changes the word when deleted.
THAI_WORD = "เทศบาลเมืองเพชรบูรณ์"  # "municipality of Phetchabun"
THAI_SENTENCE = "สำนักงานเทศบาลเมืองเพชรบูรณ์ ขอแสดงความนับถือ"
KHMER_WORD = "ភាសាខ្មែរ"  # "Khmer language"
LAO_WORD = "ຄົນລາວ"  # "Lao people"
MYANMAR_WORD = "မြန်မာဘာသာ"  # "Burmese language"
DEVANAGARI_WORD = "हिन्दी"  # "Hindi"
KANNADA_WORD = "ಕನ್ನಡ"  # "Kannada"
TAMIL_WORD = "தமிழ்"  # "Tamil"

MARKED_WORDS = [
    pytest.param(THAI_WORD, id="thai"),
    pytest.param(KHMER_WORD, id="khmer"),
    pytest.param(LAO_WORD, id="lao"),
    pytest.param(MYANMAR_WORD, id="myanmar"),
    pytest.param(DEVANAGARI_WORD, id="devanagari"),
    pytest.param(KANNADA_WORD, id="kannada"),
    pytest.param(TAMIL_WORD, id="tamil"),
]

# The junk fragment the old annotation tokenizer stored for THAI_WORD: it stops
# at the first combining mark (◌ื) instead of keeping the word whole.
THAI_GT_FRAGMENT = "เทศบาลเม"

_ANNOTATION_TOOL = Path(__file__).resolve().parents[4] / "text_annotation_tools" / "toBagOfWord.js"


def _marks(text: str) -> list[str]:
    return [ch for ch in text if unicodedata.category(ch) in {"Mn", "Mc", "Me"}]


@pytest.mark.parametrize("word", MARKED_WORDS)
def test_normalize_text_preserves_combining_marks(word: str) -> None:
    """normalize_text must not delete marks that carry meaning in the script."""
    assert _marks(word), "fixture word must contain combining marks"
    assert normalize_text(word) == word


@pytest.mark.parametrize("word", MARKED_WORDS)
def test_marked_word_tokenizes_whole(word: str) -> None:
    """A marked word survives normalization + tokenization as a single token."""
    tokens = _tokenize_unicode_words(normalize_text(word))
    assert tokens == [word]


def test_thai_sentence_tokenizes_into_whole_words() -> None:
    tokens = _tokenize_unicode_words(normalize_text(THAI_SENTENCE))
    assert tokens == ["สำนักงานเทศบาลเมืองเพชรบูรณ์", "ขอแสดงความนับถือ"]
    assert THAI_GT_FRAGMENT not in tokens


def test_stacked_thai_marks_survive() -> None:
    """Thai stacks a tone mark on top of a vowel mark; both must be kept."""
    word = "เมื่อ"  # ม + ◌ื (vowel) + ◌่ (tone) + อ
    assert len(_marks(word)) == 2
    assert normalize_text(word) == word
    assert _tokenize_unicode_words(normalize_text(word)) == [word]


def test_thai_fragment_is_not_a_token_and_does_not_match_by_boundary() -> None:
    """The old GT fragment must not spuriously pass either matching path."""
    normalized = normalize_text(THAI_SENTENCE)
    assert THAI_GT_FRAGMENT not in _tokenize_unicode_words(normalized)
    assert _word_boundary_count(THAI_GT_FRAGMENT, normalized) == 0


def test_missing_specific_word_rule_matches_whole_thai_word() -> None:
    rule = MissingSpecificWordRule({"type": "missing_specific_word", "word": THAI_WORD})
    passed, message = rule.run(f"# หัวข้อ\n\n{THAI_WORD} ขอแสดงความนับถือ\n")
    assert passed, message


@pytest.mark.parametrize("word", MARKED_WORDS)
def test_missing_specific_word_rule_round_trips_every_script(word: str) -> None:
    rule = MissingSpecificWordRule({"type": "missing_specific_word", "word": word})
    passed, message = rule.run(f"เอกสาร {word} ทดสอบ")
    assert passed, message


# --- guards: unaffected scripts keep their existing behaviour -----------------


def test_latin_accents_are_still_folded_to_ascii() -> None:
    assert normalize_text("café Ångström naïve") == "cafe angstrom naive"
    assert _tokenize_unicode_words(normalize_text("café")) == ["cafe"]


def test_cjk_and_kana_behaviour_unchanged() -> None:
    assert normalize_text("が ぱ 検査結果") == "が ぱ 検査結果"
    assert _tokenize_unicode_words(normalize_text("検査abc結果")) == ["検査", "abc", "結果"]


def test_latin_tokenization_unchanged() -> None:
    assert _tokenize_unicode_words(normalize_text("Total revenue: 1,234 USD (net)")) == [
        "total",
        "revenue",
        "234",
        "usd",
        "net",
    ]


# --- annotation side: the JS rule generator must agree with the evaluator ----
