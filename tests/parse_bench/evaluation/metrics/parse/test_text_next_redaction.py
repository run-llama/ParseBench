"""Redaction markers: every spelling of the marker reads the same."""

import pytest

from parse_bench.evaluation.metrics.parse.text_next import score_text_next

REFERENCE = (
    "VISTO el expediente electrónico formado con motivo del recurso de revisión interpuesto por [REDACTED] "
    "en contra de la respuesta del Ayuntamiento de Amecameca, se procede a dictar la presente resolución.\n\n"
    "Con fecha cuatro de octubre de dos mil dieciséis, el C. [REDACTED] presentó solicitud de acceso."
)


@pytest.mark.parametrize(
    "marker",
    [
        "[REDACTED]",
        "[redacted]",
        "[Redacted]",
        "[censored]",
        "[CENSORED]",
        "[redaction]",
        "[blacked out]",
        "████████",
        "███ ████",
    ],
)
def test_redaction_marker_spellings_are_equivalent(marker: str) -> None:
    assert score_text_next(REFERENCE, REFERENCE.replace("[REDACTED]", marker)).score == 1.0


def test_reference_spelling_does_not_matter() -> None:
    assert score_text_next(REFERENCE.replace("[REDACTED]", "[censored]"), REFERENCE).score == 1.0


@pytest.mark.parametrize("other", ["[illegible]", "", "John Smith"])
def test_other_text_in_place_of_a_redaction_is_still_charged(other: str) -> None:
    assert score_text_next(REFERENCE, REFERENCE.replace("[REDACTED]", other)).score < 1.0


def test_prose_use_of_the_word_is_unchanged() -> None:
    reference = "The names were redacted before release and the censored pages were withheld from the public file."
    assert score_text_next(reference, reference).score == 1.0
    assert score_text_next(reference, reference.replace("redacted", "censored", 1)).score < 1.0


def test_wrong_marker_is_a_wrong_word_not_a_changed_name() -> None:
    result = score_text_next(REFERENCE, REFERENCE.replace("[REDACTED]", "[illegible]"))
    assert result.events.critical == 0
    assert result.score >= 0.9
