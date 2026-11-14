"""text_next: inside a URL an underscore and a space read the same, on both sides."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.text_next import score_text_next

BASE = "https://example.org/sites/default/files/inline-files/"
REF = (
    "Adopted principles:\n\n"
    f"- {BASE}Two_ABC%20Principles_Adopted_XYZ.pdf\n"
    f"- {BASE}Three_ABC_Communication_Principles_Adopted_XYZ.pdf\n"
)


def test_url_with_underscores_rendered_as_spaces_scores_one() -> None:
    assert score_text_next(REF, REF).score == 1.0
    assert score_text_next(REF, REF.replace("_", " ")).score == 1.0


def test_reference_with_spaces_matches_an_output_with_underscores() -> None:
    spaced = REF.replace("_", " ")
    assert score_text_next(spaced, REF).score == 1.0


def test_wrong_url_is_still_charged() -> None:
    res = score_text_next(REF, REF.replace("Two_ABC", "Four_ABC").replace("_", " "))
    assert res.score < 1.0
    assert res.events.critical >= 1
