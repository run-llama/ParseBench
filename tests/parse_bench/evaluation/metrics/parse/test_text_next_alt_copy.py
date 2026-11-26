"""Page text copied inside an image description: the copy is text, the words around it are description."""

import pytest

from parse_bench.evaluation.metrics.parse.text_next import score_text_next

TAGLINE = "Instituto de Transparencia, Acceso a la Información Pública y Protección de Datos Personales"
BODY = (
    f"Resolución del Pleno del {TAGLINE}, con domicilio en Metepec; a siete de diciembre de dos mil dieciséis.\n\n"
    "VISTO el expediente electrónico formado con motivo del recurso de revisión 03304/INFOEM/IP/RR/2016, "
    "se procede a dictar la presente resolución."
)
# The line under the wordmark is printed again in the first paragraph.
REFERENCE = f"![logo](image)\n{TAGLINE}\n\n{BODY}"


@pytest.mark.parametrize(
    "logo",
    [
        f"![logo]({{}})\n{TAGLINE}",
        f"![{TAGLINE}](logo.jpg)",
        f"![logo: {TAGLINE}](logo.jpg)",
        f"![logo: INFOEM — {TAGLINE}](logo.jpg)",
        f"![Infoem logo with the tagline {TAGLINE}](logo.jpg)",
        f"[logo: infoem {TAGLINE}]",
    ],
)
def test_line_printed_twice_may_sit_in_the_logo_alt_text(logo: str) -> None:
    assert score_text_next(REFERENCE, f"{logo}\n\n{BODY}").score == 1.0


def test_line_left_out_is_still_charged() -> None:
    assert score_text_next(REFERENCE, f"![logo](logo.jpg)\n\n{BODY}").score < 1.0


def test_wrong_word_inside_the_copy_is_still_charged() -> None:
    wrong = TAGLINE.replace("Protección", "Proyección")
    assert score_text_next(REFERENCE, f"![logo: INFOEM — {wrong}](logo.jpg)\n\n{BODY}").score < 1.0


def test_number_outside_the_copy_keeps_the_whole_alt_text() -> None:
    reference = f"![logo](image)\n{TAGLINE} 2016\n\n{BODY}"
    result = score_text_next(reference, f"![logo: {TAGLINE} 2061](logo.jpg)\n\n{BODY}")
    assert result.events.critical >= 1


def test_line_printed_once_and_repeated_in_alt_text_is_ignored() -> None:
    reference = f"![logo](image)\n\n{BODY}"
    assert score_text_next(reference, f"![logo: INFOEM — {TAGLINE}](logo.jpg)\n\n{BODY}").score == 1.0
