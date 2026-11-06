"""layout_grounding_pass_rate: table, chart and checkbox regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from layout_grounding_rule_builders import _block, _line, _math_doc, _rule, _single_score, _word

from parse_bench.evaluation.metrics.parse.layout_grounding import (
    Element,
    Page,
    score_page,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "layout_grounding_presets.json").read_text())
PW, PH = 612.0, 792.0


def test_html_table_with_word_annotations_fails_wrong_cell_text() -> None:
    """Table content is HTML with no `text`; dropping every word token made the table box-only."""
    box = [0.1, 0.1, 0.4, 0.1]
    rules = [
        _rule("T", box, "Table", content={"type": "table", "html": "<table><tr><td>100</td></tr></table>"}),
        _word("W", "T", [0.2, 0.12, 0.05, 0.03], "100"),
    ]
    (row,) = _single_score(rules, [_block(box, "900", 0)]).metadata["element_results"]
    assert row["A"] is False
    assert row["score"] == 0.0


def test_text_region_without_word_rules_is_checked_against_its_content() -> None:
    """A region with content but no word children skipped A, so a well-placed box with wrong text scored 1."""
    box = [0.1, 0.1, 0.4, 0.05]
    rules = [_rule("R", box, content={"type": "text", "text": "alpha"})]
    assert _single_score(rules, [_block(box, "omega", 0)]).metadata["element_results"][0]["A"] is False
    assert _single_score(rules, [_block(box, "alpha", 0)]).value == pytest.approx(1.0)


def test_prediction_on_mark_scope_checkbox_is_not_an_extra() -> None:
    """Dropping mark checkboxes from the skip boxes charged a correct checkbox prediction as an extra."""
    rules = _math_doc("alpha beta", ["alpha", "beta"])
    rules.append(_rule("C", [0.5, 0.5, 0.02, 0.02], "Checkbox-Selected", attributes={"scope": "mark"}))
    metric = _single_score(rules, [_block(rules[0].bbox, "alpha beta", 0), _block([0.5, 0.5, 0.02, 0.02], "", 1)])
    assert metric.value == pytest.approx(1.0)


def test_equal_iou_candidates_pick_the_one_whose_text_passes_regardless_of_order() -> None:
    """np.argmax took the first of two identical boxes, so list order decided attribution."""
    rules = _math_doc("alpha", ["alpha"])
    box = rules[0].bbox
    for texts in (["omega", "alpha"], ["alpha", "omega"]):
        metric = _single_score(rules, [_block(box, t, n) for n, t in enumerate(texts)])
        assert metric.metadata["element_results"][0]["A"] is True


def test_gt_hyphen_join_stays_inside_one_region() -> None:
    """A hyphen ending region A's line fused with region B's first word, failing both regions' text."""
    h = 0.01
    rules = [
        _rule("A", [0.1, 0.1, 0.3, h], ro_index=0),
        _rule("B", [0.1, 0.12, 0.3, h], ro_index=1),
        _line("LA", "A", [0.1, 0.1, 0.2, h], 0),
        _line("LB", "B", [0.1, 0.12, 0.2, h], 1),
        _word("WA", "LA", [0.1, 0.1, 0.1, h], "alpha-"),
        _word("WB", "LB", [0.1, 0.12, 0.1, h], "beta"),
    ]
    blocks = [_block(rules[0].bbox, "alpha-", 0), _block(rules[1].bbox, "beta", 1)]
    rows = _single_score(rules, blocks).metadata["element_results"]
    assert [r["A"] for r in rows] == [True, True]


def test_rotated_table_word_coverage_uses_the_rotated_prediction() -> None:
    """Word centres were tested against the literal box, so an exact rotated prediction failed table coverage."""
    box = (0.4, 0.1, 0.6, 0.9)  # tall literal box, rotated 90 degrees it spans x 0.1-0.9 at y 0.4-0.6
    words = [(x, 0.49, x + 0.04, 0.51) for x in np.linspace(0.15, 0.8, 8)]
    page = Page(
        elements=[Element(id="T", cls="Table", box=box, words=list(range(8)), angle=90.0)],
        word_boxes=np.array(words),
        word_tokens=[(n,) for n in range(8)],
        token_text=[f"w{n}" for n in range(8)],
        pred_boxes=np.array([box]),
        pred_tokens=[[f"w{n}" for n in range(8)]],
        pred_units=[0],
        pred_angles=[90.0],
    )
    result = score_page(page, 0.0)
    assert result.element_rows[0]["L"] is True
    assert result.weighted_score / (result.weight + result.extra) == pytest.approx(1.0)


def test_equal_iou_table_tie_prefers_the_candidate_that_covers_the_words() -> None:
    """The F1 tie-break picked a box missing the header word centres, failing a table another tied box passes."""
    rules = [
        _rule("T", [0.125, 0.125, 0.5, 0.5], "Table", content={"type": "text", "text": "head line a b c d e f g h"}),
        _word("H1", "T", [0.13, 0.13, 0.05, 0.02], "head"),
        _word("H2", "T", [0.2, 0.13, 0.05, 0.02], "line"),
        *[_word(f"W{n}", "T", [0.13 + 0.05 * n, 0.3, 0.02, 0.02], c) for n, c in enumerate("abcdefgh")],
    ]
    top = _block([0.125, 0.125, 0.5, 0.4375], "head line a b c d e f", 0)  # covers every centre, F1 0.89
    bottom = _block([0.125, 0.1875, 0.5, 0.4375], "head line a b c d e f g h", 1)  # same IoU, misses the header
    for blocks in ([top, bottom], [bottom, top]):
        (row,) = _single_score(rules, blocks).metadata["element_results"]
        assert row["L"] is True and row["A"] is True


def test_one_prediction_over_two_wordless_paragraphs_credits_both() -> None:
    """Merge-aware text subtraction used only word tokens, so wordless paragraphs failed a correct merged box."""
    rules = [
        _rule("A", [0.1, 0.1, 0.4, 0.05], content={"type": "text", "text": "alpha beta"}),
        _rule("B", [0.1, 0.16, 0.4, 0.05], content={"type": "text", "text": "gamma delta"}),
    ]
    rows = _single_score(rules, [_block([0.1, 0.1, 0.4, 0.11], "alpha beta gamma delta", 0)]).metadata[
        "element_results"
    ]
    assert [r["A"] for r in rows] == [True, True]


def test_underscore_italic_words_in_the_prediction_keep_their_tokens() -> None:
    """normalize_attribution_text strips `_word` as a LaTeX subscript, so markdown italics like `_quick_` vanished."""
    rules = _math_doc("the quick brown fox", ["the", "quick", "brown", "fox"])
    metric = _single_score(rules, [_block(rules[0].bbox, "the _quick_ __brown__ fox", 0)])
    assert metric.metadata["element_results"][0]["A"] is True


@pytest.mark.parametrize("attrs", [{"ignore_thead": True}, {"exclude_table_header": True}], ids=["legacy", "alias"])
def test_excluded_table_header_words_are_not_graded(attrs: dict[str, Any]) -> None:
    """Ignoring the header flag keeps header words in the table text, failing a prediction that omits the header."""
    box = [0.1, 0.1, 0.4, 0.2]
    head = "<thead><tr><th>head</th><th>one</th><th>two</th></tr></thead>"
    html = f"<table>{head}<tr><td>alpha</td></tr><tr><td>beta</td></tr></table>"
    rules = [
        _rule("T", box, "Table", content={"type": "table", "html": html}, attributes=attrs),
        *[_word(f"H{n}", "T", [0.12 + 0.1 * n, 0.11, 0.05, 0.03], t) for n, t in enumerate(["head", "one", "two"])],
        _word("A", "T", [0.2, 0.18, 0.05, 0.03], "alpha"),
        _word("B", "T", [0.2, 0.24, 0.05, 0.03], "beta"),
    ]
    (row,) = _single_score(rules, [_block(box, "alpha beta", 0)]).metadata["element_results"]
    assert row["A"] is True


def test_chart_text_inside_a_picture_is_not_graded() -> None:
    """Legacy chart GT carries words on its Picture; grading them failed a well-placed chart box."""
    rules = _math_doc("sales 2024", ["sales", "2024"])
    rules[0] = rules[0].model_copy(update={"canonical_class": "Picture", "attributes": {"picture_type": "chart"}})
    metric = _single_score(rules, [_block(rules[0].bbox, "", 0)])
    assert metric.metadata["element_results"][0]["A"] is None
    assert metric.value == pytest.approx(1.0)
