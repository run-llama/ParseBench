"""text_next: content changes a reward model could exploit are charged; correct outputs still score 1.0."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.text_next import score_text_next

REF = (
    "# Quarterly results\n\nOperating margin fell to -5.2% in the third quarter, while net debt rose by 1,250 million. "
    "Management expects the decline to continue until the restructuring is complete.\n\n"
    "The board approved a dividend of 0.45 per share."
)


def kinds(ref: str, out: str) -> dict[str, int]:
    return score_text_next(ref, out).events.kinds()


# --- signs -----------------------------------------------------------------------------------------------------


def test_dropped_minus_is_a_critical_wrong_number() -> None:
    res = score_text_next(REF, REF.replace("-5.2%", "5.2%"))
    assert res.events.kinds().get("sign changed") == 1
    assert res.score <= 0.9


def test_minus_turned_plus_is_charged() -> None:
    assert kinds(REF, REF.replace("-5.2%", "+5.2%")).get("sign changed") == 1


def test_minus_written_as_unicode_minus_or_math_scores_one() -> None:
    assert score_text_next(REF, REF.replace("-5.2%", "−5.2%")).score == 1.0
    assert score_text_next(REF, REF.replace("-5.2%", "$-5.2$%")).score == 1.0


def test_dashes_that_are_not_signs_are_not_charged() -> None:
    ref = "Records from 19920322 - 19970920 were kept.\n\n- 3 -\n\nThe count is $n-1$ per voter, see page 10-15."
    out = "Records from 19920322 -19970920 were kept.\n\n-3-\n\nThe count is n -1 per voter, see page 10-15."
    assert score_text_next(ref, out).score == 1.0


# --- checkboxes ------------------------------------------------------------------------------------------------

BOXES = "☑ Signed by the auditor\n\n☐ Reviewed by the board\n\nThe report was filed on time."


def test_flipped_checkbox_is_critical() -> None:
    res = score_text_next(BOXES, BOXES.replace("☑ Signed", "☐ Signed"))
    assert res.events.kinds().get("checkbox flipped") == 1
    assert res.score <= 0.9


def test_checkbox_in_another_notation_or_left_out_scores_one() -> None:
    assert score_text_next(BOXES, BOXES.replace("☑", "[x]").replace("☐", "[ ]")).score == 1.0
    assert score_text_next(BOXES, BOXES.replace("☑", "✓").replace("☐", "□")).score == 1.0
    assert score_text_next(BOXES, BOXES.replace("☑ ", "").replace("☐ ", "")).score == 1.0


# --- math operators --------------------------------------------------------------------------------------------

BOUND = "The bound holds: $\\sum_{i=1}^{n} x_i \\leq C$ for every sample, with p < 0.05 in the second test."


def test_swapped_math_operators_are_critical() -> None:
    res = score_text_next(BOUND, BOUND.replace("\\sum", "\\prod").replace("\\leq", "\\geq"))
    assert res.events.critical >= 2
    assert res.score <= 0.8


def test_flipped_comparison_in_prose_is_critical() -> None:
    res = score_text_next(BOUND, BOUND.replace("p < 0.05", "p > 0.05"))
    assert res.events.critical == 1
    assert res.score <= 0.9


def test_operator_spellings_and_markup_score_one() -> None:
    assert score_text_next(BOUND, BOUND.replace("\\leq", "≤").replace("p < 0.05", "p &lt; 0.05")).score == 1.0
    assert score_text_next(BOUND, BOUND.replace("\\leq", "<=")).score == 1.0
    marked = BOUND + "\n\nSee <https://example.org/a?b=1> -> next == end"
    ref = BOUND + "\n\nSee https://example.org/a?b=1 next end"
    assert score_text_next(ref, marked).score == 1.0


# --- currency signs --------------------------------------------------------------------------------------------

FEES = "The fee is ₹ 250 per month, or ₩ 3,000 in Korea and ₽ 90 in Russia."


def test_changed_currency_sign_is_critical() -> None:
    ref = "The fee is ₹ 250 per month for every member of the club."
    for sign in ("₩", "₽", "$"):
        res = score_text_next(ref, ref.replace("₹", sign))
        assert res.events.critical == 1
        assert res.score <= 0.9


def test_currency_signs_written_the_same_score_one() -> None:
    assert score_text_next(FEES, FEES.replace("₹ 250", "₹250").replace("3,000", "3000")).score >= 0.98


# --- flipped meaning -------------------------------------------------------------------------------------------


def test_word_replaced_by_its_opposite_is_critical() -> None:
    for old, new in (("margin fell", "margin rose"), ("approved", "rejected"), ("approved", "denied")):
        res = score_text_next(REF, REF.replace(old, new))
        assert res.events.kinds().get("meaning flipped") == 1, (old, new)
        assert res.score <= 0.9


def test_added_negation_is_critical() -> None:
    res = score_text_next(REF, REF.replace("board approved", "board not approved"))
    assert res.events.critical == 1


def test_unrelated_word_change_is_not_a_flip() -> None:
    res = score_text_next(REF, REF.replace("margin fell", "margin fall"))
    assert "meaning flipped" not in res.events.kinds()


# --- alt text --------------------------------------------------------------------------------------------------

DIVIDEND = "The board approved a dividend of 0.45 per share."


def test_long_genuine_image_description_is_free() -> None:
    drawing = "![drawing: connecting charger clamps to a car battery and plugging the device into a wall outlet](p.jpg)"
    assert score_text_next(REF, REF + "\n\n" + drawing).score == 1.0
    photo = '<img alt="A man in a blue jacket speaking in front of a bookshelf full of books" src="p.jpg"/>'
    assert score_text_next(REF, REF + "\n\n" + photo).score == 1.0


def test_page_text_hidden_only_in_alt_is_matched() -> None:
    hidden = REF.replace(DIVIDEND, f"![image: {DIVIDEND}](a.png)")
    assert score_text_next(REF, hidden).score == 1.0
    assert score_text_next(REF, REF + f"\n\n![image: {DIVIDEND}](a.png)").score == 1.0  # also in the body: ignored
    res = score_text_next(REF, REF.replace(DIVIDEND, f"![image: {DIVIDEND.replace('0.45', '0.54')}](a.png)"))
    assert res.events.kinds().get("wrong number or identifier") == 1


def test_junk_alt_text_is_inserted_text() -> None:
    res = score_text_next(REF, REF + "\n\n![image: Lx~ I-. -. I- z 0 C 0 0 U, E 0 E I-](p.jpg)")
    assert res.events.kinds().get("invented passage") == 1
    assert res.score <= 0.9


# --- thousands separators --------------------------------------------------------------------------------------


def test_space_or_apostrophe_thousands_read_like_commas() -> None:
    for written in ("1 250", "1 250", "1 250", "1 250", "1'250", "1’250"):
        res = score_text_next(REF, REF.replace("1,250", written))
        assert res.events.critical == 0, written
        assert res.score >= 0.99


def test_other_digits_with_spaces_are_still_wrong() -> None:
    assert score_text_next(REF, REF.replace("1,250", "12 50")).events.critical == 1
