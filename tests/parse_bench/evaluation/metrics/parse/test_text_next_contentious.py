"""text_next: math written in an HTML ``<math>`` element or a ``$$$`` fence, and digits of other scripts."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.text_next import score_text_next

REF_MATH = (
    "Má-li funkce $g$ v bodě $a$ derivaci $g'(a)$, pak platí\n\n"
    "$$f'(a) = h'(g(a)) \\cdot g'(a)$$\n\n"
    "a tedy $(\\sin(x^2+1))' = \\cos(x^2+1) \\cdot 2x$ pro každé x."
)


# --- <math> elements and $$$ fences read as math ------------------------------------------------------------


def test_latex_in_math_elements_reads_like_dollar_math() -> None:
    out = (
        "Má-li funkce <math>g</math> v bodě <math>a</math> derivaci <math>g'(a)</math>, pak platí\n\n"
        "<math display=\"block\">f'(a) = h'(g(a)) \\cdot g'(a)</math>\n\n"
        "a tedy <math>(\\sin(x^2+1))' = \\cos(x^2+1) \\cdot 2x</math> pro každé x."
    )

    assert score_text_next(REF_MATH, out).score == 1.0


def test_triple_dollar_fence_reads_like_display_math() -> None:
    out = REF_MATH.replace("$$f'(a)", "$$$f'(a)").replace("g'(a)$$", "g'(a)$$$")

    assert score_text_next(REF_MATH, out).score == 1.0


def test_wrong_latex_inside_math_element_is_still_charged() -> None:
    out = (
        "Má-li funkce <math>g</math> v bodě <math>a</math> derivaci <math>g'(a)</math>, pak platí\n\n"
        "<math display=\"block\">f'(a) = h'(g(a)) \\cdot g'(a)</math>\n\n"
        "a tedy <math>(\\sin(x^2+1))' = \\cos(x^3+7) \\cdot 5x</math> pro každé x."
    )

    assert score_text_next(REF_MATH, out).score < 0.9


def test_garbage_inside_math_element_is_still_charged() -> None:
    out = (
        "Má-li funkce <math>g</math> v bodě <math>a</math> derivaci <math>g'(a)</math>, pak platí\n\n"
        '<math display="block">\\mathcal{Q} \\oplus \\Xi \\otimes 42 \\wedge 17 \\vee 99</math>\n\n'
        "a tedy <math>(\\sin(x^2+1))' = \\cos(x^2+1) \\cdot 2x</math> pro každé x."
    )

    assert score_text_next(REF_MATH, out).score < 0.9


@pytest.mark.parametrize("math", [r"$\gamma$", r"$$\gamma$$", r"\(\gamma\)", r"\[\gamma\]", r"<math>\gamma</math>"])
def test_latex_greek_letter_is_not_deleted_or_charged_as_invented_text(math: str) -> None:
    reference = f"Marker {math} remains active."
    output = "Marker γ remains active."

    assert score_text_next(reference, output).score == 1.0
    assert score_text_next(output, reference).score == 1.0


@pytest.mark.parametrize(
    "output", ["Marker remains active.", r"Marker $\delta$ remains active.", "Marker δ remains active."]
)
def test_missing_or_substituted_greek_letter_is_not_hidden_by_command_stripping(output: str) -> None:
    assert score_text_next(r"Marker $\gamma$ remains active.", output).score < 1.0


def test_greek_command_conversion_keeps_existing_critical_operator_and_number_checks() -> None:
    reference = r"Reading $\gamma \leq 5$ remains valid."

    assert score_text_next(reference, "Reading γ ≤ 5 remains valid.").score == 1.0
    for output in ("Reading γ ≥ 5 remains valid.", "Reading γ 5 remains valid.", "Reading γ ≤ 6 remains valid."):
        assert score_text_next(reference, output).score <= 0.9


@pytest.mark.parametrize("output", [r"Marker $\gammamax$ remains active.", "Marker gamma remains active."])
def test_greek_command_requires_a_whole_math_command_not_a_prefix_or_prose_name(output: str) -> None:
    assert score_text_next("Marker γ remains active.", output).score < 1.0


def test_other_greek_letters_and_html_entities_share_the_unicode_reading() -> None:
    reference = r"Markers $\alpha$, $\beta$, $\Delta$, $\lambda$ and $\mu$ remain active."
    output = "Markers &alpha;, β, Δ, λ and μ remain active."

    assert score_text_next(reference, output).score == 1.0
    assert score_text_next(output, reference).score == 1.0


# --- decimal digits of another script ---------------------------------------------------------------------------

REF_THAI = "ติดต่อสอบถาม โทร ๐ ๕๖๗๑ ๑๐๐๗ หรือ ๐๘ ๑๒๓๔ ๕๖๗๘ ทุกวัน"


def test_thai_digits_written_in_ascii_are_the_same_number() -> None:
    assert score_text_next(REF_THAI, "ติดต่อสอบถาม โทร 0 5671 1007 หรือ 08 1234 5678 ทุกวัน").score == 1.0


def test_ascii_digits_written_in_devanagari_are_the_same_number() -> None:
    assert score_text_next("कलम 4 आणि 28 मध्ये 2009 साली", "कलम ४ आणि २८ मध्ये २००९ साली").score == 1.0


def test_wrong_digit_in_either_script_is_still_a_wrong_number() -> None:
    ascii_wrong = score_text_next(REF_THAI, "ติดต่อสอบถาม โทร 0 5671 1008 หรือ 08 1234 5678 ทุกวัน")
    native_wrong = score_text_next(REF_THAI, "ติดต่อสอบถาม โทร ๐ ๕๖๗๑ ๑๐๐๘ หรือ ๐๘ ๑๒๓๔ ๕๖๗๘ ทุกวัน")

    assert ascii_wrong.score < 0.9
    assert native_wrong.score < 0.9


# a phone number, an amount with Arabic separators where the script has them, and a year; each script's zero
ZEROS = {"thai": 0x0E50, "devanagari": 0x0966, "bengali": 0x09E6, "arabic-indic": 0x0660, "persian": 0x06F0}
ZEROS |= {"fullwidth": 0xFF10, "gujarati": 0x0AE6}


def native(text: str, zero: int) -> str:
    out = "".join(chr(zero + int(c)) if c.isascii() and c.isdigit() else c for c in text)
    return out.replace(",", "\u066c").replace(".", "\u066b") if zero in (0x0660, 0x06F0) else out


AMOUNTS = "Call 0 5671 1007 before 2009 to pay 12,345.6 for the form"
MISREAD = "Call 0 5671 1008 before 2009 to pay 12,345.6 for the form"


@pytest.mark.parametrize("zero", ZEROS.values(), ids=ZEROS.keys())
def test_native_digits_and_their_ascii_transcription_are_both_exact_and_misreads_stay_critical(zero: int) -> None:
    """The reference keeps native digits. A native output and an ASCII transcription both score exact, in either
    direction, and one misread digit in either script is a critical wrong number."""
    ref = native(AMOUNTS, zero)

    assert score_text_next(ref, ref).score == 1.0
    assert score_text_next(ref, AMOUNTS).score == 1.0
    assert score_text_next(AMOUNTS, ref).score == 1.0
    for wrong in (MISREAD, native(MISREAD, zero)):
        events = score_text_next(ref, wrong).events
        assert events.critical == 1
        assert ("wrong number or identifier", "#1007", "#1008") in events.examples
