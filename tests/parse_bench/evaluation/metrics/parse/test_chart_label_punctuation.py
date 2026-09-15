"""Encoding-only punctuation must never decide a chart label match (issue #2314).

Two halves are covered here:

1. The deterministic label-matching path already folds typographic punctuation via
   ``normalize_text``. These tests lock that in so a future normalization change
   cannot silently reintroduce the defect.
2. The judge evidence builder now folds the same punctuation on BOTH sides, so the
   LLM judge is never handed a pure encoding difference to arbitrate.
"""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.utils import normalize_label_punctuation, normalize_text

# --------------------------------------------------------------------------
# normalize_label_punctuation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typographic", "ascii_form"),
    [
        # Curly vs straight apostrophe -- the defect reported in issue #2314.
        ("’26F", "'26F"),
        ("Bachelor’s degree", "Bachelor's degree"),
        ("‘26F", "'26F"),
        # Curly vs straight double quote.
        ("“Net zero” scenario", '"Net zero" scenario'),
        # En dash and em dash vs ASCII hyphen.
        ("2024–2030", "2024-2030"),
        ("2024—2030", "2024-2030"),
    ],
)
def test_typographic_punctuation_folds_to_ascii(typographic: str, ascii_form: str) -> None:
    assert normalize_label_punctuation(typographic) == normalize_label_punctuation(ascii_form)


@pytest.mark.parametrize(
    ("typographic", "ascii_form"),
    [
        ("′26F", "'26F"),
        ("2024‑2030", "2024-2030"),
        ("2024‒2030", "2024-2030"),
        ("2024−2030", "2024-2030"),
    ],
)
def test_deterministic_quote_and_dash_variants_fold_for_judge_evidence(
    typographic: str,
    ascii_form: str,
) -> None:
    assert normalize_label_punctuation(typographic) == normalize_label_punctuation(ascii_form)


@pytest.mark.parametrize(
    ("typographic", "expected"),
    [
        ("’26F", "'26f"),
        ("“Net zero” scenario", '"net zero" scenario'),
        ("2024‑2030", "2024-2030"),
        ("2024‒2030", "2024-2030"),
        ("2024–2030", "2024-2030"),
        ("2024—2030", "2024-2030"),
        ("2024−2030", "2024-2030"),
    ],
)
def test_normalize_text_folds_same_quote_and_dash_variants(
    typographic: str,
    expected: str,
) -> None:
    """Prove the deterministic utility emits the intended canonical form."""
    assert normalize_text(typographic) == expected


def test_internal_whitespace_runs_collapse_and_trim() -> None:
    assert normalize_label_punctuation("  Data   center\tsemiconductors \n") == "Data center semiconductors"


def test_case_and_wording_are_preserved() -> None:
    """Unlike normalize_text, this must stay legible for judge-facing evidence."""
    assert normalize_label_punctuation("Data Center Semiconductors") == "Data Center Semiconductors"


def test_none_normalizes_to_empty_string() -> None:
    assert normalize_label_punctuation(None) == ""


def test_genuinely_different_labels_still_differ() -> None:
    """Negative case: normalization must not collapse distinct labels."""
    assert normalize_label_punctuation("Bachelor’s degree") != normalize_label_punctuation("Master's degree")
    assert normalize_label_punctuation("’26F") != normalize_label_punctuation("'27F")


# --------------------------------------------------------------------------
# Deterministic rule path -- regression lock
# --------------------------------------------------------------------------
