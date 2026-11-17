"""Regression checks for annotation-driven tightness without changing the other scoring gates."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from layout_grounding_rule_builders import _block, _rule, _word
from pydantic import ValidationError

from parse_bench.evaluation.metrics.parse.layout_grounding import _build_page
from parse_bench.evaluation.metrics.parse.layout_grounding_score import score_page
from parse_bench.test_cases.layout_bbox_tolerance import word_outward_tolerances
from parse_bench.test_cases.schema import LayoutBboxTolerance


def test_opt_in_flows_from_rules_to_tightness_without_changing_matching_or_penalties() -> None:
    rules = [_rule("r", [0.2, 0.2, 0.1, 0.02]), _word("w", "r", [0.2, 0.2, 0.1, 0.02], "hello")]
    prediction = _block([0.196, 0.198, 0.108, 0.024], "hello", 0)
    legacy, hs = _build_page(rules, [prediction], 1, (100, 200))
    a = score_page(legacy, min(hs))
    rules[1].bbox_tolerance = word_outward_tolerances([rules[1]], (100, 200))[0]
    tolerant, _ = _build_page(rules, [prediction], 1, (100, 200))
    b = score_page(tolerant, min(hs))
    assert b.element_rows[0]["tightness"] == pytest.approx(1)
    assert a.element_rows[0]["tightness"] < 1
    assert b.weight == a.weight and b.extra == a.extra and b.prediction_rows == a.prediction_rows
    assert {k: v for k, v in a.element_rows[0].items() if k not in ("score", "tightness")} == {
        k: v for k, v in b.element_rows[0].items() if k not in ("score", "tightness")
    }
    tolerant.word_outward = None
    assert score_page(tolerant, min(hs)) == a


@pytest.mark.parametrize("prediction", [[0.204, 0.204, 0.092, 0.012], [0.15, 0.15, 0.2, 0.12]])
def test_outward_does_not_forgive_inward_clipping_or_unlimited_padding(prediction: list[float]) -> None:
    rules = [_rule("r", [0.2, 0.2, 0.1, 0.02]), _word("w", "r", [0.2, 0.2, 0.1, 0.02], "hello")]
    legacy, hs = _build_page(rules, [_block(prediction, "hello", 0)], 1, (100, 200))
    tolerant = copy.deepcopy(legacy)
    tolerant.word_outward = np.array([[0.4, 0.4, 0.4, 0.4]])
    before, after = score_page(legacy, min(hs)), score_page(tolerant, min(hs))
    if prediction[0] > 0.2:
        assert before == after
    else:
        assert after.element_rows[0]["tightness"] < 1


def test_neighbor_caps_use_physical_height_and_zero_existing_overlap_and_rotation() -> None:
    words = [_word("a", "r", [0.2, 0.2, 0.1, 0.02], "a"), _word("b", "r", [0.302, 0.2, 0.1, 0.02], "b")]
    values = word_outward_tolerances(words, (100, 200))
    assert values[0].outward_normalized == pytest.approx((0.004, 0.002, 0.001, 0.002))
    assert values[1].outward_normalized[0] == pytest.approx(0.001)
    words[1].bbox[0] = 0.29
    assert all(not any(v.outward_normalized) for v in word_outward_tolerances(words, (100, 200)))
    words[1].bbox[0] = 0.4
    words[0].r = 90
    assert not any(word_outward_tolerances(words, (100, 200))[0].outward_normalized)
    assert word_outward_tolerances([], (100, 200)) == []


def test_punctuation_borrows_letter_height_and_page_edge_still_caps_it() -> None:
    words = [_word("a", "r", [0, 0.2, 0.1, 0.02], "alpha"), _word("dash", "r", [0.11, 0.209, 0.01, 0.002], "-")]
    a, dash = word_outward_tolerances(words, (100, 200))
    assert a.outward_normalized[0] == 0
    assert dash.outward_normalized == pytest.approx((0.004, 0.002, 0.004, 0.002))


@pytest.mark.parametrize("margins", [[-1, 0, 0, 0], [float("nan"), 0, 0, 0], [0, 0, 0]])
def test_invalid_margin_metadata_cannot_enter_scorer(margins: list[float]) -> None:
    with pytest.raises(ValidationError):
        LayoutBboxTolerance.model_validate({"version": "outward-word-height-10-v1", "outward_normalized": margins})


@pytest.mark.parametrize(
    "cls,attrs,angle", [("Picture", {}, None), ("Table", {}, None), ("Text", {"grade": "box"}, None), ("Text", {}, 90)]
)
def test_box_only_and_rotated_regions_do_not_inherit_word_tolerance(
    cls: str, attrs: dict[str, str], angle: int | None
) -> None:
    rules = [
        _rule("r", [0.2, 0.2, 0.1, 0.02], cls=cls, attributes=attrs, r=angle),
        _word("w", "r", [0.2, 0.2, 0.1, 0.02], "hello"),
    ]
    blocks = [_block([0.196, 0.198, 0.108, 0.024], "hello", 0)]
    a, hs = _build_page(rules, blocks, 1, (100, 200))
    rules[1].bbox_tolerance = word_outward_tolerances([rules[1]], (100, 200))[0]
    b, _ = _build_page(rules, blocks, 1, (100, 200))
    assert score_page(a, min(hs)) == score_page(b, min(hs))


def test_diagonal_neighbors_cannot_gain_new_overlap_from_both_expansions() -> None:
    words = [_word("a", "r", [0.2, 0.2, 0.1, 0.02], "a"), _word("b", "s", [0.301, 0.2205, 0.1, 0.02], "b")]
    a, b = word_outward_tolerances(words, (100, 200))
    assert a.outward_normalized[2] + b.outward_normalized[0] <= 0.001 + 1e-12 or (
        a.outward_normalized[3] + b.outward_normalized[1] <= 0.0005 + 1e-12
    )
