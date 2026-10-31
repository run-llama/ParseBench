"""layout_grounding_pass_rate: rule-level scoring and region attributes."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from layout_grounding_rule_builders import _block, _line, _math_doc, _rule, _single_score, _word

from parse_bench.evaluation.metrics.attribution.core import PredBlock
from parse_bench.evaluation.metrics.attribution.text_utils import normalize_attribution_text, tokenize
from parse_bench.evaluation.metrics.parse.layout_grounding import (
    METRIC_NAME,
    compute_layout_grounding_metrics,
    content_tokens,
)
from parse_bench.test_cases.schema import LayoutTestRule

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "layout_grounding_presets.json").read_text())
PW, PH = 612.0, 792.0

# ------------------------------------------------------- rules through blocks


def _hyphen_doc(gap: float) -> list[LayoutTestRule]:
    """One paragraph of two lines, 'infor-' ending line 1 and 'mation' starting line 2 ``gap`` below."""
    h = 0.01
    y2 = 0.1 + h + gap
    return [
        _rule("P", [0.1, 0.1, 0.3, y2 + h - 0.1], ro_index=0),
        _line("L1", "P", [0.1, 0.1, 0.2, h], 0),
        _line("L2", "P", [0.1, y2, 0.2, h], 1),
        _word("W1", "L1", [0.1, 0.1, 0.1, h], "infor-"),
        _word("W2", "L2", [0.1, y2, 0.1, h], "mation"),
    ]


def _f1(rules: list[LayoutTestRule], text: str = "information") -> Any:
    return _single_score(rules, [_block(rules[0].bbox, text, 0)]).metadata["element_results"][0]["token_f1"]


def test_gt_hyphen_join_needs_a_hyphen_and_a_following_line_below() -> None:
    """Joining without a hyphen, or onto a line beside instead of below, fuses two real words."""
    plain = _hyphen_doc(gap=0.002)
    plain[3] = _word("W1", "L1", plain[3].bbox, "infor")
    assert _f1(plain) == 0.0
    beside = [
        _rule("P", [0.1, 0.1, 0.3, 0.01], ro_index=0),
        _line("L1", "P", [0.1, 0.1, 0.1, 0.01], 0),
        _line("L2", "P", [0.25, 0.1, 0.1, 0.01], 1),
        _word("W1", "L1", [0.1, 0.1, 0.1, 0.01], "infor-"),
        _word("W2", "L2", [0.25, 0.1, 0.1, 0.01], "mation"),
    ]
    assert _f1(beside) == 0.0


def test_gt_hyphen_join_gate_has_a_thirty_unit_floor_in_a_page_frame() -> None:
    """With page dimensions known, a 25 pt gap between 5 pt lines still joins (30 pt floor)."""
    h, gap = 5 / 792, 25 / 792
    rules = [
        _rule("P", [0.1, 0.1, 0.3, 2 * h + gap], ro_index=0),
        _line("L1", "P", [0.1, 0.1, 0.2, h], 0),
        _line("L2", "P", [0.1, 0.1 + h + gap, 0.2, h], 1),
        _word("W1", "L1", [0.1, 0.1, 0.1, h], "infor-"),
        _word("W2", "L2", [0.1, 0.1 + h + gap, 0.1, h], "mation"),
    ]
    blocks = [_block(rules[0].bbox, "information", 0)]
    (metric,) = compute_layout_grounding_metrics(rules, lambda page: blocks, {1: (PW, PH)})
    assert metric.metadata["element_results"][0]["token_f1"] == pytest.approx(1.0)


@pytest.mark.parametrize("attrs", [{"caption": True}, {"grade": "box"}], ids=["legacy", "grade"])
def test_caption_text_is_scored_on_its_box_without_a_text_test(attrs: dict[str, Any]) -> None:
    """Caption (grade box) text is not read as body text; a text test would fail a well-placed caption box."""
    rules = [
        _rule("C", [0.1, 0.1, 0.3, 0.02], ro_index=0, attributes=attrs),
        _word("W", "C", [0.1, 0.1, 0.1, 0.02], "alpha"),
    ]
    metric = _single_score(rules, [_block([0.1, 0.1, 0.3, 0.02], "other", 0)])
    assert metric.value == pytest.approx(1.0)
    assert metric.metadata["element_results"][0]["A"] is None


def test_word_belongs_to_its_parent_region_not_the_smallest_box_around_its_center() -> None:
    """Assigning by center alone gives a word inside a nested picture to the picture, failing the paragraph text."""
    rules = [
        _rule("P", [0.1, 0.1, 0.4, 0.1], ro_index=0),
        _rule("PIC", [0.3, 0.1, 0.2, 0.1], cls="Picture", ro_index=1),
        _line("L", "P", [0.1, 0.1, 0.4, 0.02], 0),
        _word("W1", "L", [0.1, 0.1, 0.1, 0.02], "alpha"),
        _word("W2", "L", [0.35, 0.1, 0.1, 0.02], "beta"),
    ]
    metric = _single_score(rules, [_block(rules[0].bbox, "alpha beta", 0), _block(rules[1].bbox, "", 1)])
    paragraph, picture = metric.metadata["element_results"]
    assert paragraph["token_f1"] == pytest.approx(1.0)
    assert picture["element_id"] == "PIC" and picture["A"] is None


def test_weight_floor_is_the_smallest_scored_word_height_across_pages() -> None:
    """A floor from one page only, or from ignored words, gives a tiny picture a different weight."""
    rules = [
        _rule("T1", [0.1, 0.1, 0.3, 0.01], ro_index=0),
        _word("W1", "T1", [0.1, 0.1, 0.1, 0.01], "alpha"),
        _rule("DOT", [0.6, 0.6, 0.001, 0.001], cls="Picture", ro_index=1),
        _rule("IGN", [0.1, 0.8, 0.3, 0.1], attributes={"ignore": True}),
        _word("W3", "IGN", [0.1, 0.8, 0.1, 0.005], "watermark"),
        LayoutTestRule(id="T2", page=2, bbox=[0.1, 0.1, 0.3, 0.02], canonical_class="Text", ro_index=0),
        LayoutTestRule(
            id="W2",
            page=2,
            bbox=[0.1, 0.1, 0.1, 0.02],
            canonical_class="Text",
            granularity="word",
            parent_test_id="T2",
            content={"type": "text", "text": "beta"},
        ),
    ]
    (metric,) = compute_layout_grounding_metrics(rules, lambda page: [])
    dot = next(r for r in metric.metadata["element_results"] if r["element_id"] == "DOT")
    assert dot["weight"] == pytest.approx(0.01)


def test_gt_hyphen_join_lets_joined_prediction_text_pass_attribution() -> None:
    """Without the GT join, 'infor- mation' against 'information' fails token F1."""
    rules = _hyphen_doc(gap=0.002)
    box = rules[0].bbox
    metric = _single_score(rules, [_block(box, "information", 0)])
    (row,) = metric.metadata["element_results"]
    assert row["token_f1"] == pytest.approx(1.0)


def test_gt_hyphen_join_gate_is_four_line_heights_on_the_unit_square() -> None:
    """The gate must scale with line height when no page frame is known; 30 page units would join anything."""
    near, far = _hyphen_doc(gap=0.039), _hyphen_doc(gap=0.041)
    joined = _single_score(near, [_block(near[0].bbox, "information", 0)])
    split = _single_score(far, [_block(far[0].bbox, "information", 0)])
    assert joined.metadata["element_results"][0]["token_f1"] == pytest.approx(1.0)
    assert split.metadata["element_results"][0]["token_f1"] == 0.0


@pytest.mark.parametrize(
    ("opt", "ign"),
    [({"optional": "true"}, {"ignore": True}), ({"grade": "none"}, {"grade": "none", "grade_reason": "watermark"})],
    ids=["legacy", "grade"],
)
def test_missing_optional_and_ignored_regions_are_not_penalized(opt: dict[str, Any], ign: dict[str, Any]) -> None:
    """Scoring an optional or ignored (grade none) region as an element turns a perfect page into a miss."""
    rules = [
        _rule("P", [0.1, 0.1, 0.3, 0.02], ro_index=0),
        _word("W1", "P", [0.1, 0.1, 0.1, 0.02], "alpha"),
        _rule("OPT", [0.5, 0.5, 0.2, 0.2], cls="Picture", attributes=opt),
        _rule("IGN", [0.5, 0.8, 0.2, 0.1], attributes=ign),
        _word("W2", "IGN", [0.5, 0.8, 0.1, 0.02], "watermark"),
    ]
    metric = _single_score(rules, [_block([0.1, 0.1, 0.3, 0.02], "alpha", 0)])
    assert metric.metric_name == METRIC_NAME
    assert metric.value == pytest.approx(1.0)
    assert [r["element_id"] for r in metric.metadata["element_results"]] == ["P"]
    assert metric.metadata["numerator_label"] == "score"
    assert metric.metadata["passed"] == pytest.approx(metric.metadata["total"])


def test_no_scored_elements_and_no_predictions_emit_nothing() -> None:
    """A zero denominator must not emit 0/0 or a fake 0 that drags the average."""
    rules = [
        _rule("OPT", [0.5, 0.5, 0.2, 0.2], attributes={"optional": "true"}),
        _word("W", "OPT", [0.55, 0.55, 0.05, 0.02], "hello"),
    ]
    assert compute_layout_grounding_metrics(rules, lambda page: []) == []


def test_ground_truth_without_any_content_emits_nothing_and_reads_no_predictions() -> None:
    """Box-only GT has no content to ground: scoring it reads the adapter and crashes detector-only outputs."""

    def read(page: int) -> list[PredBlock]:
        raise AssertionError("prediction blocks read for content-less ground truth")

    rules = [_rule("T", [0.1, 0.1, 0.2, 0.2]), _rule("P", [0.5, 0.5, 0.2, 0.2], cls="Picture")]
    assert compute_layout_grounding_metrics(rules, read) == []


def _para(rid: str, y: float, ro: int, words: list[str], page: int = 1) -> list[LayoutTestRule]:
    """A one-line paragraph at height ``y`` whose words sit 0.12 apart."""
    out = [LayoutTestRule(id=rid, page=page, bbox=[0.1, y, 0.3, 0.02], canonical_class="Text", ro_index=ro)]
    for k, t in enumerate(words):
        out.append(
            LayoutTestRule(
                id=f"{rid}w{k}",
                page=page,
                bbox=[0.1 + 0.12 * k, y, 0.1, 0.02],
                canonical_class="Text",
                granularity="word",
                parent_test_id=rid,
                content={"type": "text", "text": t},
            )
        )
    return out


def _ro_doc() -> tuple[list[LayoutTestRule], list[PredBlock]]:
    """P1, P2, P3 in reading order, listed P1, P3, P2; one block reads P1 and P2, another P3."""
    rules = _para("P1", 0.1, 0, ["aa", "bb"]) + _para("P3", 0.16, 2, ["ee", "ff"]) + _para("P2", 0.13, 1, ["cc", "dd"])
    return rules, [_block([0.1, 0.1, 0.3, 0.05], "aa bb cc dd", 0), _block([0.1, 0.16, 0.3, 0.02], "ee ff", 1)]


def test_relabeling_prediction_blocks_does_not_change_the_score() -> None:
    """The headline is classification-free: reading a block label or type anywhere in the path moves it."""
    rules, blocks = _ro_doc()
    rules += [_rule("PIC", [0.6, 0.6, 0.2, 0.2], cls="Picture", ro_index=3)]
    blocks += [_block([0.6, 0.6, 0.2, 0.2], "", 2)]
    base = _single_score(rules, blocks).value
    assert base > 0.5
    for label in ("picture", "table", "title", "page_header"):
        relabeled = [dataclasses.replace(b, label=label, block_type="table") for b in blocks]
        assert _single_score(rules, relabeled).value == pytest.approx(base, abs=1e-12), label


def test_elements_group_in_ground_truth_reading_order_not_rule_order() -> None:
    """Sorting elements by rule order puts P3 between P1 and P2, so the block reading P1 and P2 forms no group."""
    rules, blocks = _ro_doc()
    rows = _single_score(rules, blocks).metadata["element_results"]
    groups = {r["element_id"]: r["group_id"] for r in rows}
    assert [r["element_id"] for r in rows] == ["P1", "P2", "P3"]
    assert groups["P1"] == groups["P2"] is not None
    assert groups["P3"] is None


def test_blocks_sharing_an_order_index_are_one_prediction_unit() -> None:
    """Two segments of one column-crossing item must group as one unit, not score as two partial blocks."""
    rules = [
        _rule("L", [0.1, 0.1, 0.3, 0.02], ro_index=0),
        _word("w0", "L", [0.1, 0.1, 0.1, 0.02], "alpha"),
        _word("w1", "L", [0.25, 0.1, 0.1, 0.02], "beta"),
        _rule("R", [0.55, 0.1, 0.3, 0.02], ro_index=1),
        _word("w2", "R", [0.55, 0.1, 0.1, 0.02], "gamma"),
        _word("w3", "R", [0.7, 0.1, 0.1, 0.02], "delta"),
    ]
    blocks = [_block([0.1, 0.1, 0.3, 0.02], "alpha beta", 0), _block([0.55, 0.1, 0.3, 0.02], "gamma delta", 0)]
    rows = _single_score(rules, blocks).metadata["element_results"]
    assert rows[0]["group_id"] == rows[1]["group_id"] is not None


def test_attribution_fails_below_token_f1_of_point_eight() -> None:
    """An element whose prediction reads half its words (F1 0.67) must fail A; a 0.5 gate passes it."""
    rules = _para("P", 0.1, 0, ["aa", "bb", "cc", "dd"])
    (row,) = _single_score(rules, [_block(rules[0].bbox, "aa bb", 0)]).metadata["element_results"]
    assert row["token_f1"] == pytest.approx(2 / 3)
    assert (row["L"], row["A"]) == (True, False)


@pytest.mark.parametrize(
    ("pred", "passes"),
    [
        ("alpha beta gamma delta epsilon- zeta", True),
        ("alpha beta", False),
        ("alpha beta gamma delta epsilon-zeta", True),  # content may keep a line-break hyphen as `epsilon- zeta`
    ],
)
def test_region_text_is_graded_against_its_content_not_only_its_annotated_words(pred: str, passes: bool) -> None:
    """With words annotated for only part of a region, the full text must pass and the annotated part must not."""
    rules = _math_doc("alpha beta gamma delta epsilon- zeta", ["alpha", "beta"])
    row = _single_score(rules, [_block(rules[0].bbox, pred, 0)]).metadata["element_results"][0]
    assert row["A"] is passes


def test_ocr_tokens_of_inline_math_absent_from_content_are_not_scored() -> None:
    """OCR words of `$...$` math (stripped from content) would sink token F1 of a clean prediction."""
    rules = _math_doc(r"a matrix $Q_k \in \mathbb{R}^{k}$ satisfies", ["a", "matrix", "Qk", "E_R(k-1)xk,", "satisfies"])
    row = _single_score(rules, [_block(rules[0].bbox, "a matrix satisfies", 0)]).metadata["element_results"][0]
    assert row["token_f1"] == pytest.approx(1.0)
    assert row["A"] is True


def test_display_math_element_is_box_only() -> None:
    """A `$$...$$` element has no content tokens; scoring its OCR words fails a well-placed box on attribution."""
    rules = _math_doc(r"$$Q_k = \begin{bmatrix} V \end{bmatrix}$$", ["Qk", "=", "VE", "Vk(k-1)"])
    metric = _single_score(rules, [_block(rules[0].bbox, "Q_k = V", 0)])
    row = metric.metadata["element_results"][0]
    assert row["A"] is None
    assert metric.value == pytest.approx(1.0)


def test_mark_scope_checkbox_rules_are_not_scored() -> None:
    """Mark-scope checkboxes are near-zero boxes; scoring them adds unearnable weight."""
    rules = _math_doc("alpha beta", ["alpha", "beta"])
    rules.append(_rule("C", [0.5, 0.5, 1e-5, 1e-5], "Checkbox-Selected", attributes={"scope": "mark"}))
    metric = _single_score(rules, [_block(rules[0].bbox, "alpha beta", 0)])
    assert [r["element_id"] for r in metric.metadata["element_results"]] == ["R"]
    assert metric.value == pytest.approx(1.0)


def test_weights_are_square_roots_of_areas_in_page_points_per_page_size() -> None:
    """On the unit square every weight is a fraction; mixed page sizes must each weigh in their own points."""
    rules = _para("T", 0.1, 0, ["alpha"]) + [
        _rule("PIC1", [0.5, 0.5, 0.4, 0.4], cls="Picture", ro_index=1),
        LayoutTestRule(id="PIC2", page=2, bbox=[0.5, 0.5, 0.4, 0.4], canonical_class="Picture", ro_index=0),
    ]
    (metric,) = compute_layout_grounding_metrics(rules, lambda page: [], {1: (PW, PH), 2: (2 * PW, PH)})
    w = {r["element_id"]: r["weight"] for r in metric.metadata["element_results"]}
    assert w["PIC1"] == pytest.approx(np.sqrt(0.16 * PW * PH))
    assert w["PIC2"] == pytest.approx(np.sqrt(0.16 * 2 * PW * PH))
    (unit,) = compute_layout_grounding_metrics(rules, lambda page: [])
    assert {r["element_id"]: r["weight"] for r in unit.metadata["element_results"]}["PIC1"] == pytest.approx(0.4)


def test_rule_lines_chain_within_one_point_in_points_and_never_overlapping_on_the_unit_square() -> None:
    """1 pt is a whole page on the unit square: a fixed tolerance there chains overlapping lines."""

    def weights(overlap: float, sizes: dict[int, tuple[float, float]] | None) -> list[float]:
        h, y2 = 0.02, 0.12 - overlap
        rules = _para("A", 0.1, 0, ["aa"]) + [
            _rule("B", [0.1, y2, 0.3, h], ro_index=1),
            _word("Bw", "B", [0.1, y2, 0.1, h], "bb"),
        ]
        (metric,) = compute_layout_grounding_metrics(rules, lambda page: [], sizes)
        return [r["weight"] for r in metric.metadata["element_results"]]

    scale = np.sqrt(PW * PH)
    assert weights(0.5 / PH, {1: (PW, PH)}) != pytest.approx([scale * np.sqrt(0.006)] * 2)
    assert weights(0.005, None) == pytest.approx([np.sqrt(0.006)] * 2)


def test_eszett_tokens_agree_with_the_attribution_normalizer() -> None:
    """The non-ASCII sentinel bypasses the normalizer's ss fold, so 'Straße' would not match 'Strasse'."""
    assert content_tokens("Straße") == content_tokens("Strasse") == tokenize(normalize_attribution_text("Straße"))


# ------------------------------------------------------------ evaluator wiring
