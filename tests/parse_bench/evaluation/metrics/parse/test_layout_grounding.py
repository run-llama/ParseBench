"""layout_grounding_pass_rate: reference presets, invariances, tokenizer and rule-to-page wiring."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from parse_bench.evaluation.metrics.parse.layout_grounding import (
    Element,
    Page,
    content_tokens,
    score_page,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "layout_grounding_presets.json").read_text())
PW, PH = 612.0, 792.0


def _xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def _preset_page(gt: list[dict[str, Any]], preds: list[dict[str, Any]]) -> tuple[Page, float]:
    word_boxes, word_tokens, token_text, elements = [], [], [], []
    for g in gt:
        ids = []
        for w in g["words"]:
            ids.append(len(word_boxes))
            word_boxes.append(_xyxy(w["box"]))
            word_tokens.append((len(token_text),))
            token_text.append(w["t"])
        elements.append(Element(id=g["id"], cls=g["cls"], box=_xyxy(g["box"]), words=ids))
    page = Page(
        elements=elements,
        word_boxes=np.array(word_boxes, dtype=float).reshape(-1, 4),
        word_tokens=word_tokens,
        token_text=token_text,
        pred_boxes=np.array([_xyxy(p["box"]) for p in preds], dtype=float).reshape(-1, 4),
        pred_tokens=[list(p["tok"]) for p in preds],
        pred_units=[p["item"] if p["item"] is not None else f"#{j}" for j, p in enumerate(preds)],
        skip_boxes=np.array([_xyxy(o["box"]) for o in FIXTURE["optional"]], dtype=float).reshape(-1, 4),
    )
    h_min = min(w["box"][3] for g in gt for w in g["words"])
    return page, h_min


def _score(gt: list[dict[str, Any]], preds: list[dict[str, Any]]) -> float:
    page, h_min = _preset_page(gt, preds)
    result = score_page(page, h_min)
    return result.weighted_score / (result.weight + result.extra)


@pytest.mark.parametrize("preset", FIXTURE["presets"], ids=lambda p: p["name"])
def test_preset_matches_reference_score_and_extra_count(preset: dict[str, Any]) -> None:
    """A change in matching, grouping, triage or weighting moves a reference preset score."""
    page, h_min = _preset_page(FIXTURE["gts"][preset["gt"]], preset["preds"])
    result = score_page(page, h_min)
    assert result.weighted_score / (result.weight + result.extra) == pytest.approx(preset["score"], abs=1e-6)
    assert result.n_extra == preset["n_extra"]
    assert result.weight == pytest.approx(preset["total_weight"], rel=1e-9)


def test_layout_only_perfect_boxes_score_box_credit_of_non_text_elements_only() -> None:
    """A zero-text branch or a text element passing without text changes the layout-only score."""
    preds = [{**p, "tok": []} for p in FIXTURE["presets"][0]["preds"]]
    assert _score(FIXTURE["gts"]["base"], preds) == pytest.approx(FIXTURE["layout_only_perfect"], abs=1e-6)


def test_prediction_order_permutation_does_not_change_perfect_score() -> None:
    """Confidence is ignored: shuffling predictions (no ties) must not change a tie-free preset."""
    gt = FIXTURE["gts"]["base"]
    for name in ("perfect", "junk", "merge5", "picChildren"):
        preds = next(p for p in FIXTURE["presets"] if p["name"] == name)["preds"]
        shuffled = preds[:]
        random.Random(3).shuffle(shuffled)
        assert _score(gt, shuffled) == pytest.approx(_score(gt, preds), abs=1e-9), name


def test_weight_floor_lifts_tiny_elements_to_min_word_height() -> None:
    """Without the floor, a 1x1 picture weighs 1 instead of the document's minimum word height."""
    word = (0.0, 0.0, 10.0, 8.0)
    page = Page(
        elements=[
            Element(id="T", cls="Text", box=word, words=[0]),
            Element(id="DOT", cls="Picture", box=(50.0, 50.0, 51.0, 51.0), words=[]),
        ],
        word_boxes=np.array([word]),
        word_tokens=[(0,)],
        token_text=["alpha"],
        pred_boxes=np.zeros((0, 4)),
        pred_tokens=[],
        pred_units=[],
    )
    rows = score_page(page, h_min=8.0).element_rows
    assert [row["weight"] for row in rows] == pytest.approx([np.sqrt(80.0), 8.0])


def test_rows_report_every_element_and_reason_every_unused_prediction() -> None:
    """A dropped row or a triage arm returning no reason hides why a document scored as it did."""
    for preset in FIXTURE["presets"]:
        page, h_min = _preset_page(FIXTURE["gts"][preset["gt"]], preset["preds"])
        result = score_page(page, h_min)
        assert [r["element_id"] for r in result.element_rows] == [e.id for e in page.elements]
        assert all(r["reason"] and r["kind"] in {"extra", "neutral"} for r in result.prediction_rows)
        assert sum(r["kind"] == "extra" for r in result.prediction_rows) == preset["n_extra"]
        assert sum(r["penalty"] for r in result.prediction_rows) == pytest.approx(result.extra)


def _text_page(
    elements: list[tuple[str, tuple[float, float, float, float], list[tuple[float, float, float, float]]]],
    preds: list[tuple[tuple[float, float, float, float], list[str], Any]],
) -> Page:
    """Elements as (cls, box, word boxes), word n reading 'w<n>'; preds as (box, tokens, unit)."""
    word_boxes, elems = [], []
    for n, (cls, box, words) in enumerate(elements):
        elems.append(
            Element(id=f"E{n}", cls=cls, box=box, words=list(range(len(word_boxes), len(word_boxes) + len(words))))
        )
        word_boxes.extend(words)
    return Page(
        elements=elems,
        word_boxes=np.array(word_boxes, dtype=float).reshape(-1, 4),
        word_tokens=[(n,) for n in range(len(word_boxes))],
        token_text=[f"w{n}" for n in range(len(word_boxes))],
        pred_boxes=np.array([p[0] for p in preds], dtype=float).reshape(-1, 4),
        pred_tokens=[p[1] for p in preds],
        pred_units=[p[2] for p in preds],
    )


def _row_words(y: float, x0: float, n: int) -> list[tuple[float, float, float, float]]:
    return [(x0 + 20 * k, y, x0 + 20 * k + 15, y + 10) for k in range(n)]


def test_unit_holding_a_minority_of_each_paragraph_does_not_link_them() -> None:
    """Without the majority gate, a box over one word of each paragraph merges both into one group."""
    e0, e1 = _row_words(0, 0, 4), _row_words(12, 0, 4)
    page = _text_page(
        [("Text", (0, 0, 75, 10), e0), ("Text", (0, 12, 75, 22), e1)],
        [
            ((0, 0, 55, 10), ["w0", "w1", "w2"], "a"),
            ((60, 0, 75, 22), ["w3", "w7"], "b"),
            ((0, 12, 55, 22), ["w4", "w5", "w6"], "c"),
        ],
    )
    rows = score_page(page, h_min=10.0).element_rows
    assert [r["group_id"] for r in rows] == [None, None]


def test_single_element_single_unit_never_forms_a_group_that_skips_localization() -> None:
    """A one-element group is forced localized; a page-sized box with the right text would pass L."""
    page = _text_page([("Text", (0, 0, 35, 10), _row_words(0, 0, 2))], [((0, 0, 600, 800), ["w0", "w1"], "a")])
    (row,) = score_page(page, h_min=10.0).element_rows
    assert (row["L"], row["group_id"], row["why"]) == (False, None, "no_localization")


def test_group_members_must_be_consecutive_in_reading_order() -> None:
    """A unit over paragraphs 1 and 3 (2 predicted apart) must not group across paragraph 2."""
    words = [_row_words(0, 0, 2), _row_words(12, 0, 2), _row_words(24, 0, 2)]
    page = _text_page(
        [("Text", (0, 12 * k, 35, 12 * k + 10), w) for k, w in enumerate(words)],
        [
            ((0, 0, 35, 10), ["w0", "w1"], "a"),
            ((0, 24, 35, 34), ["w4", "w5"], "a"),
            ((0, 12, 35, 22), ["w2", "w3"], "b"),
        ],
    )
    assert [r["group_id"] for r in score_page(page, h_min=10.0).element_rows] == [None, None, None]


def test_column_crossing_paragraph_scores_tightness_per_column() -> None:
    """A two-segment item over a two-column paragraph scores 1; a loose right segment costs only its column."""
    left, right = _row_words(0, 0, 2), _row_words(0, 300, 2)
    elements = [("Text", (0, 0, 35, 10), left), ("Text", (300, 0, 335, 10), right)]
    exact = [((0, 0, 35, 10), ["w0", "w1"], "a"), ((300, 0, 335, 10), ["w2", "w3"], "a")]
    loose = [exact[0], ((300, 0, 335, 14), ["w2", "w3"], "a")]
    result = score_page(_text_page(elements, exact), h_min=10.0)
    assert result.weighted_score / result.weight == pytest.approx(1.0)
    rows = score_page(_text_page(elements, loose), h_min=10.0).element_rows
    assert rows[0]["group_id"] == rows[1]["group_id"] is not None
    assert rows[0]["tightness"] == pytest.approx(1.0)
    assert rows[1]["tightness"] < 1.0


LINE = (100.0, 400.0, 400.0, 460.0)  # 5:1 line, unrotated frame; its one word shares its centre
HULL_90 = (220.0, 280.0, 280.0, 580.0)


@pytest.mark.parametrize(
    ("angle", "preds", "tightness"),
    [
        (15.0, [LINE], 0.05),  # unrotated box on a 15 degree line: polygon IoU 0.52, not the box's 1.0
        (90.0, [HULL_90], 1.0),  # axis-aligned hull of a vertical line: polygon IoU 1.0, not the box's 0.11
        (15.0, [(100.0, 400.0, 250.0, 460.0), (250.0, 400.0, 400.0, 460.0)], 0.05),  # same, split in a group
    ],
)
def test_tightness_of_a_rotated_element_is_polygon_iou(angle: float, preds: list[Any], tightness: float) -> None:
    """Tightness on the unrotated boxes credits a tilted line's unrotated box and zeroes a vertical line's hull."""
    page = Page(
        elements=[Element(id="L", cls="Text", box=LINE, words=[0], angle=angle)],
        word_boxes=np.array([LINE]),
        word_tokens=[(0,)],
        token_text=["w0"],
        pred_boxes=np.array(preds),
        pred_tokens=[["w0"]] + [[]] * (len(preds) - 1),
        pred_units=["a"] * len(preds),
        pred_angles=[None] * len(preds),
    )
    (row,) = score_page(page, h_min=10.0).element_rows
    assert row["score"] == pytest.approx(tightness, abs=0.01)
    assert (row["group_id"] is not None) == (len(preds) > 1)


def test_unused_prediction_reasons_name_partial_overlap_orphan_words_and_blank_area() -> None:
    """A triage arm that falls through to the wrong reason hides what kind of extra a box is."""
    elements = [
        ("Text", (0, 0, 35, 10), _row_words(0, 0, 2)),
        ("Text", (0, 100, 35, 110), [(0, 100, 15, 110), (400, 400, 415, 410)]),
    ]
    preds = [
        ((30, 5, 60, 30), [], 0),
        ((398, 398, 420, 412), [], 1),
        ((200, 200, 220, 220), [], 2),
        ((0, 100, 35, 110), ["w2"], 3),
        ((0, 0, 35, 10), ["w0", "w1"], 4),
    ]
    rows = score_page(_text_page(elements, preds), h_min=10.0).prediction_rows
    assert [(r["pred_index"], r["reason"]) for r in rows] == [
        (0, "partial overlap"),
        (1, "orphan words"),
        (2, "blank area"),
    ]


def test_text_chain_weight_needs_vertical_succession_and_shared_x_range() -> None:
    """Chained lines share sqrt(area of their union) by area; side-by-side or overlapping lines do not chain."""

    def weights(boxes: list[tuple[float, float, float, float]]) -> list[float]:
        page = _text_page([("Text", b, [b]) for b in boxes], [])
        return [r["weight"] for r in score_page(page, h_min=0.0).element_rows]

    chained = weights([(0, 0, 100, 10), (0, 12, 50, 22)])
    assert sum(chained) == pytest.approx(np.sqrt(100 * 22))
    assert chained[0] == pytest.approx(2 * chained[1])
    assert weights([(0, 0, 100, 10), (200, 12, 250, 22)]) == pytest.approx([np.sqrt(1000), np.sqrt(500)])
    assert weights([(0, 0, 100, 10), (0, 5, 50, 15)]) == pytest.approx([np.sqrt(1000), np.sqrt(500)])


def test_text_chain_tolerates_one_point_of_overlap_independent_of_word_height() -> None:
    """The reference chains lines overlapping by under 1 pt; a 0 or height-relative tolerance chains others."""

    def chained(overlap: float) -> bool:
        boxes = [(0.0, 0.0, 100.0, 20.0), (0.0, 20.0 - overlap, 100.0, 40.0 - overlap)]
        page = _text_page([("Text", b, [b]) for b in boxes], [])
        w = [r["weight"] for r in score_page(page, h_min=0.0).element_rows]
        return bool(sum(w) != pytest.approx(np.sqrt(2000) * 2))

    assert chained(0.5)
    assert not chained(1.5)


# ----------------------------------------------------------------- tokenizer


def test_cjk_characters_are_single_tokens_and_cyrillic_words_are_kept() -> None:
    """The ASCII-only normalizer drops non-Latin text, leaving CJK/Cyrillic elements with no tokens."""
    assert content_tokens("日本語") == ["日", "本", "語"]
    assert content_tokens("Привет") == ["привет"]
    assert content_tokens("Café 42") == ["cafe", "42"]
    assert content_tokens("한국") == ["한", "국"]


def test_prediction_side_joins_hyphen_followed_by_spaces() -> None:
    """Parsers flatten 'infor-\\nmation' to 'infor- mation'; without the join it splits into two tokens."""
    assert content_tokens("infor- mation", pred=True) == ["information"]
    assert content_tokens("infor- mation") == ["infor", "mation"]
    assert content_tokens("pages 3- 4", pred=True) == ["pages", "3", "4"]


@pytest.mark.parametrize("pred", [False, True])
def test_hyphenated_word_tokenizes_the_same_with_or_without_a_line_break(pred: bool) -> None:
    """'in-\\nclass' joined to 'inclass' while 'in-class' split into 'in', 'class', so GT and prediction disagreed."""
    assert content_tokens("in-class", pred=pred) == content_tokens("in-\nclass", pred=pred) == ["inclass"]


@pytest.mark.parametrize(
    ("latex", "plain"),
    [
        ("L$_{1}$", "L₁"),
        ("L$_{1}$", "L<sub>1</sub>"),
        ("x$^{2}$ + y", "x<sup>2</sup> + y"),
        (r"\mathcal{F} \subset X", "X"),
        (r"a \(x_{i}\) and \[y^2\] b", "a and b"),
        (r"the \textbf{deadline} is", "the deadline is"),
        (r"\emph{very \textit{late}} \mathcal{F}", "very late"),
        (r"alpha \frac{\sqrt{2}}{\text{bottom}} beta", "alpha beta"),  # math args stay math, not prose
        (r"the \textbf{see \cite{x}} end", "the see end"),
    ],
)
def test_latex_markup_tokenizes_like_the_same_text_written_without_latex(latex: str, plain: str) -> None:
    """Upstream strips only '\\\\' LaTeX ('\\mathcal' leaked, 'L$_{1}$' lost its '1') and '\\textbf{x}' lost 'x'."""
    assert content_tokens(latex) == content_tokens(plain)


@pytest.mark.parametrize("pred", [False, True])
@pytest.mark.parametrize("text", ["10⁶-kilobase", "10<sup>6</sup>-kilobase", "10^6-kilobase", "10^{6}-kilobase"])
def test_superscript_before_a_hyphen_tokenizes_the_same_in_every_markup(text: str, pred: bool) -> None:
    """'⁶' counted as a letter before NFKC, so '10⁶-kilobase' joined to '106kilobase'; bare '^6' was dropped."""
    assert content_tokens(text, pred=pred) == ["106", "kilobase"]
