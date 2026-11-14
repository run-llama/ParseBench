"""Contract tests for the additive v0.3 metric, including real evaluator routing.

The metric names keep their ``v1`` prefix; every rule is scored by scorer
version 2. Its additions (linking passes, dilation, row pieces, charts by
role, furniture) are tested one by one in test_markdown_attribution_v2.py.
"""

import math
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest

from parse_bench.evaluation import markdown_attribution
from parse_bench.evaluation.evaluators.parse import ParseEvaluator
from parse_bench.evaluation.markdown_attribution import (
    _clamped_area,
    canonical_latex,
    cell_f1,
    chart_cell_f1,
    chart_cells,
    compute_markdown_attribution_metrics,
    geometry,
    headline_rows,
    markdown_elements,
    normalize,
    score_markdown_attribution,
    table_cells,
)
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.markdown_attribution_schema import (
    AttributionReference,
    AttributionRegion,
    MarkdownAttributionRule,
)
from parse_bench.test_cases.parse_rule_schemas import coerce_parse_rule
from parse_bench.test_cases.schema import ParseTestCase

A = "Revenue increased by twelve percent this year."
B = "Operating expenses remained stable across the company."
BOX_A = [0.1, 0.1, 0.7, 0.2]
BOX_B = [0.1, 0.6, 0.7, 0.2]


@pytest.fixture(autouse=True)
def _undilated(monkeypatch):
    # These tests check IoU and weighting arithmetic on exact boxes. The 1 pt
    # dilation would move every value a little; it has its own tests.
    monkeypatch.setattr(markdown_attribution, "DILATION_PT", 0.0)


def reference(text, box, index=0, **kwargs):
    return dict(id=f"ref-{index}", text=text, order=index, regions=[{"page": 1, "bbox": box}], **kwargs)


def reference_model(data):
    return AttributionReference.model_validate(data)


def rule(refs=None, **kwargs):
    return MarkdownAttributionRule(references=refs or [reference(A, BOX_A), reference(B, BOX_B, 1)], **kwargs)


def output(texts, boxes, *, markdown=None, page=1):
    items = []
    for text, box in zip(texts, boxes, strict=True):
        regions = [] if box is None else (box if isinstance(box[0], list) else [box])
        items.append(
            LayoutItemIR(md=text, layout_segments=[LayoutSegmentIR(x=b[0], y=b[1], w=b[2], h=b[3]) for b in regions])
        )
    return ParseOutput(
        example_id="test",
        pipeline_name="test",
        markdown="\n\n".join(texts) if markdown is None else markdown,
        layout_pages=[ParseLayoutPageIR(page_number=page, width=1, height=1, items=items)],
    )


def test_no_reference_recall_and_no_duplicate_penalty():
    for texts, boxes in [([A], [BOX_A]), ([A, B], [BOX_A, BOX_B]), ([A, A, B], [BOX_A, BOX_A, BOX_B])]:
        result = score_markdown_attribution(output(texts, boxes), rule())
        assert result["quality"] == pytest.approx(1)
        assert result["assessment_coverage"] == 1


def test_missing_box_and_empty_output_are_not_zero_or_perfect_quality():
    result = score_markdown_attribution(output([A, B], [BOX_A, None]), rule())
    assert result["quality"] == pytest.approx(1)
    assert result["box_coverage"] == 0.5
    for actual in [output([A], [None]), output([], [])]:
        result = score_markdown_attribution(actual, rule())
        assert result["quality"] is None
        metrics = compute_markdown_attribution_metrics(actual, [rule()])
        assert "md_attribution_v1_quality" not in {m.metric_name for m in metrics}


def test_wrong_location_and_full_page_and_wrong_page():
    assert score_markdown_attribution(output([A], [BOX_B]), rule())["quality"] == 0
    assert score_markdown_attribution(output([A], [[0, 0, 1, 1]]), rule())["quality"] == pytest.approx(0.14)
    assert score_markdown_attribution(output([A], [BOX_A], page=2), rule())["quality"] == 0


def test_unicode_formatting_and_small_ocr_drift():
    for expected, actual in [
        ("这是一个中文段落。", "这是一个中文段落。"),
        (A, A.replace("increased", "increasd")),
        ("**Revenue** increased.", "Revenue increased."),
    ]:
        result = score_markdown_attribution(output([actual], [BOX_A]), rule([reference(expected, BOX_A)]))
        assert result["quality"] == pytest.approx(1)


def test_numeric_and_negation_changes_are_not_text_matches():
    # The digit and negation gates keep the per-element text match from
    # resolving either pair. With one free reference in reading order, the
    # later passes still link them (a changed number through the fuzzy
    # fallback, which has no digit gate; a negation through a relaxed link),
    # and the row says which pass did.
    for expected, actual, match in [
        ("Revenue increased 12 percent this year.", "Revenue increased 13 percent this year.", "fuzzy"),
        (A, "Revenue did not increase by twelve percent this year.", "relaxed"),
    ]:
        result = score_markdown_attribution(output([actual], [BOX_A]), rule([reference(expected, BOX_A)]))
        assert result["elements"][0]["match"] == match
        assert result["elements"][0]["text_similarity"] < 0.92


@pytest.mark.parametrize(
    ("gt", "md"),
    [
        (
            r"$\Phi_{ij} \in C^5_{1-n}(N_\infty)$ and $A \geqslant 0$",
            r"$\Phi_{ij} \in C_{1-n}^5(N_\infty)$ and $A \ge 0$",
        ),
        (r"$$ W_T = \left(\hat{S}_T\right)^{-1}. \tag{4} $$", r"$$W_T = (\hat{S}_T)^{-1}. \quad (4)$$"),
        (r"$\rho: N \to \mathbf{R}$, $a \qquad \text{with} \, b$", r"$\rho: N \rightarrow R$, $a \quad with b$"),
        (r"$$ \begin{aligned} L &= a & (14) \\ &= b \end{aligned} $$", "$$L = a \\quad (14)$$\n$$= b$$"),
        (r"\(\square_R \varphi\)", r"$\Box_R \varphi$"),
    ],
)
def test_latex_idioms_normalize_to_one_form(gt, md):
    assert normalize(gt) == normalize(md)


def test_latex_canonicalization_keeps_the_formula():
    # Different symbols, digits or scripts stay different.
    for a, b in [(r"$x^2_1$", r"$x^1_2$"), (r"$a \ge b$", r"$a \le b$"), (r"$\mathbf{R}^3$", r"$\mathbf{R}^4$")]:
        assert normalize(a) != normalize(b)
    # Outside math nothing is rewritten.
    assert canonical_latex(r"costs \left 5 dollars") == r"costs \left 5 dollars"


def test_latex_script_order_resolves_despite_the_digit_gate():
    gt = r"where $A$ is a constant and $\Phi_{ij} \in C^5_{1-n}(N_\infty)$ holds for every chart"
    md = r"where $A$ is a constant and $\Phi_{ij} \in C_{1-n}^5(N_\infty)$ holds for every chart"
    result = score_markdown_attribution(output([md], [BOX_A]), rule([reference(gt, BOX_A)]))
    assert result["elements"][0]["status"] == "resolved"


def picture(text, box, index=0):
    return reference(text, box, index, kind="Picture")


def test_alt_text_prefix_and_bracket_markers_read_as_their_words():
    assert normalize("![photo: The Wedding Feast](img.jpg)") == normalize("THE WEDDING FEAST")
    assert normalize("![icon: signature](img.jpg)") == normalize("[signature]")
    assert normalize("[seal: ROYAL DENTAL COLLEGE]") == "royal dental college"
    # A link is not a marker.
    assert normalize("[logo](https://example.com)") == "logo"


def test_image_resolves_to_its_picture_not_to_a_caption_with_the_same_words():
    refs = [picture("THE WEDDING FEAST", BOX_A), reference("The wedding feast", BOX_B, 1, kind="Caption")]
    image = "![photo: The Wedding Feast](page_1_image_1.jpg)"
    element = score_markdown_attribution(output([image], [BOX_A]), rule(refs))["elements"][0]
    assert element["status"] == "resolved"
    assert element["reference_ids"] == ["ref-0"]
    # With only the caption on the page the image stays unresolved.
    element = score_markdown_attribution(output([image], [BOX_A]), rule(refs[1:]))["elements"][0]
    assert element["status"] == "unresolved_content"


def test_marker_reference_does_not_collide_with_a_heading_or_another_mark():
    refs = [picture("[seal: ROYAL DENTAL COLLEGE]", BOX_A), reference("ROYAL DENTAL COLLEGE", BOX_B, 1)]
    result = score_markdown_attribution(
        output(["ROYAL DENTAL COLLEGE", "![logo: Royal Dental College](logo.jpg)"], [BOX_B, [0.1, 0.9, 0.1, 0.05]]),
        rule(refs),
    )
    heading, logo = result["elements"]
    # The heading still has one source; the seal's words do not make it a duplicate.
    assert heading["status"] == "resolved"
    assert heading["reference_ids"] == ["ref-1"]
    # A logo is not the seal that prints the same words.
    assert logo["status"] == "unresolved_content"
    seal = score_markdown_attribution(output(["![seal: Royal Dental College](s.jpg)"], [BOX_A]), rule(refs))
    assert seal["elements"][0]["reference_ids"] == ["ref-0"]


def test_line_break_hyphen_is_joined_before_the_negation_gate():
    gt = "Markets feared that monetary policy might neither stay loose nor tighten as planned by the bank."
    md = gt.replace("neither", "nei- ther")
    result = score_markdown_attribution(output([md], [BOX_A]), rule([reference(gt, BOX_A)]))
    assert result["elements"][0]["status"] == "resolved"
    assert normalize("Offi- cer") == "officer"
    # A suspended compound keeps its words apart.
    assert normalize("pre- and post-war") == "pre and post war"


def test_digit_gate_ignores_a_leading_list_ordinal_on_one_side_only():
    gt = "1. The requesting entity is an eligible entity as defined in Regulations section 301.7701-3(a);"
    md = "1. The requesting entity is an eligible entity as defined in Regulations section 301.7701-3(a);"
    # The Markdown writes the number as an ordered-list marker, which the element text drops.
    result = score_markdown_attribution(output([md], [BOX_A]), rule([reference(gt, BOX_A)]))
    assert result["elements"][0]["status"] == "resolved"
    # Any other number difference still fails, including a different ordinal on both sides.
    for other in [
        "2 The requesting entity is an eligible entity as defined in Regulations section 301.7701-3(a);",
        "The requesting entity is an eligible entity as defined in Regulations section 301.7702-3(a);",
    ]:
        result = score_markdown_attribution(output([other], [BOX_A]), rule([reference(gt, BOX_A)]))
        assert result["elements"][0]["status"] != "resolved"


def test_repeated_reference_text_does_not_use_predicted_box_to_disambiguate():
    refs = [reference(A, BOX_A), reference(A, BOX_B, 1)]
    for box in [BOX_A, BOX_B]:
        result = score_markdown_attribution(output([A], [box]), rule(refs))
        assert result["elements"][0]["status"] == "ambiguous_source"
        assert result["quality"] is None


def test_split_needs_span_geometry_but_annotated_span_can_score():
    split = "Revenue increased by twelve percent"
    result = score_markdown_attribution(output([split], [BOX_A]), rule())
    assert result["elements"][0]["status"] == "needs_span_geometry"
    refs = [reference(A, BOX_A, spans=[reference(split, BOX_B, 10)])]
    assert score_markdown_attribution(output([split], [BOX_B]), rule(refs))["quality"] == pytest.approx(1)


def test_substring_that_clears_the_threshold_resolves_after_fuzzy_matching():
    # The Markdown split a short heading off the paragraph the reference holds.
    body = "We also constrain the predicted latent features to closely match the true ones."
    refs = [reference("Latent loss " + body, BOX_A)]
    element = score_markdown_attribution(output([body], [BOX_A]), rule(refs))["elements"][0]
    assert element["status"] == "resolved"
    assert element["reference_ids"] == ["ref-0"]


def test_substring_of_an_unrelated_reference_stays_needs_span_geometry():
    # "employer identification number" is in two form rows; the fuzzy best is
    # the shorter row, but the element may be the other row's piece.
    refs = [
        reference("Name: Employer identification number:", BOX_A),
        reference("Trust's name and address: Employer identification number:", BOX_B, 1),
    ]
    element = score_markdown_attribution(output(["Employer identification number:"], [BOX_B]), rule(refs))
    assert element["elements"][0]["status"] == "needs_span_geometry"


def test_exact_source_identity_wins_over_a_fuzzy_merged_superset():
    long_text = "This is a long uniquely identifiable source paragraph. " * 10
    refs = [reference(long_text, BOX_A), reference("Next", [0.1, 0.32, 0.7, 0.1], 1)]
    result = score_markdown_attribution(output([long_text], [BOX_A]), rule(refs))
    assert result["quality"] == pytest.approx(1)


def test_near_tie_between_a_reference_and_a_merge_containing_it_resolves():
    long_text = "These metrics may have negative scalar curvature at some points of the manifold. " * 4
    heading = reference("Proof.", [0.1, 0.32, 0.7, 0.02], 1)
    refs = [reference(long_text, BOX_A), heading]
    # One typo: the paragraph alone scores ~0.99, "paragraph + Proof." ~0.97,
    # inside the 0.04 margin. Both name the paragraph; the better one wins.
    typo = long_text.replace("negative", "negatve", 1)
    element = score_markdown_attribution(output([typo], [BOX_A]), rule(refs))["elements"][0]
    assert element["status"] == "resolved"
    assert element["reference_ids"] == ["ref-0"]
    # A near tie with an unrelated reference stays ambiguous.
    twin = reference(long_text.replace("points", "place", 1), BOX_B, 2)
    element = score_markdown_attribution(output([typo], [BOX_A]), rule([*refs, twin]))["elements"][0]
    assert element["status"] == "ambiguous_source"


def test_contiguous_merge_and_no_cross_column_envelope():
    next_box = [0.1, 0.32, 0.7, 0.2]
    merged = A + " " + B
    result = score_markdown_attribution(
        output([merged], [[0.1, 0.1, 0.7, 0.42]]), rule([reference(A, BOX_A), reference(B, next_box, 1)])
    )
    assert result["quality"] == pytest.approx(1)
    # A and B are not consecutive in one column (BOX_A and BOX_B are far apart):
    # no merge, so the text match leaves the element; a relaxed link takes one reference.
    envelope = score_markdown_attribution(output([merged], [[0, 0, 1, 1]]), rule())["elements"][0]
    assert envelope["match"] == "relaxed" and len(envelope["reference_ids"]) == 1


def test_multibox_within_expected_count_uses_union_geometry():
    r = reference(A, BOX_A, expected_bbox_count=2)
    r["regions"].append({"page": 1, "bbox": BOX_B})
    r["exception"] = "multi_column"
    actual = output([A], [[BOX_A, BOX_B]])
    assert score_markdown_attribution(actual, rule([r]))["quality"] == pytest.approx(1)
    ordinary = score_markdown_attribution(actual, rule())
    assert ordinary["quality"] == 0
    assert ordinary["elements"][0]["status"] == "multi_box_penalized"
    assert ordinary["valid_boxed"] == 0
    a, b = AttributionRegion(page=1, bbox=BOX_A), AttributionRegion(page=2, bbox=BOX_A)
    assert geometry([a, b], [a])[0] == pytest.approx(0.5)


# A left and right half of BOX_A: together they cover it exactly (union IoU 1).
HALF_L = [0.1, 0.1, 0.35, 0.2]
HALF_R = [0.45, 0.1, 0.35, 0.2]
# A top and bottom half of BOX_A. Side-by-side halves are row pieces and count
# as one box against the budget; stacked halves count as two.
HALF_T = [0.1, 0.1, 0.7, 0.1]
HALF_B = [0.1, 0.2, 0.7, 0.1]


def column_paragraph(expected_bbox_count=2, **kwargs):
    """A paragraph whose GT is its two column halves (HALF_L, HALF_R)."""
    r = reference(A, HALF_L, expected_bbox_count=expected_bbox_count, exception="multi_column", **kwargs)
    r["regions"].append({"page": 1, "bbox": HALF_R})
    return r


def test_default_budget_is_one_box_and_penalises_an_ordinary_two_box_claim():
    assert reference_model(reference(A, BOX_A)).expected_bbox_count == 1
    actual = output([A, B], [BOX_A, [HALF_T, HALF_B]])
    result = score_markdown_attribution(actual, rule())
    row = result["elements"][1]
    assert row["status"] == "multi_box_penalized"
    assert row["quality"] == 0
    assert row["expected_bbox_count"] == 1
    assert row["unpenalized_quality"] == pytest.approx(0)  # halves of BOX_A, target is BOX_B
    assert row["contract_valid"] is False
    assert result["assessed"] == 2
    assert result["assessment_coverage"] == 1
    assert result["valid_boxed"] == 1
    assert result["quality"] == pytest.approx(0.5)
    assert result["multi_box_penalized"] == 1
    assert "multi_box_policy" not in result["parameters"]
    metrics = {m.metric_name: m for m in compute_markdown_attribution_metrics(actual, [rule()])}
    assert metrics["md_attribution_v1_quality"].value == pytest.approx(0.5)
    assert "multi_box_policy" not in metrics["md_attribution_v1_quality"].metadata
    assert metrics["md_attribution_v1_multi_box_penalized_count"].value == 1
    # A perfectly placed pair of boxes is still penalized: the contract is one box.
    exact = score_markdown_attribution(output([A], [[HALF_T, HALF_B]]), rule())
    assert exact["elements"][0]["quality"] == 0
    assert exact["elements"][0]["unpenalized_quality"] == pytest.approx(1)


def test_penalty_count_metric_is_always_emitted():
    metrics = {m.metric_name: m for m in compute_markdown_attribution_metrics(output([A], [BOX_A]), [rule()])}
    assert metrics["md_attribution_v1_multi_box_penalized_count"].value == 0


def test_expected_bbox_count_under_at_and_over_the_budget():
    third = [0.1, 0.35, 0.2, 0.05]
    cases = {
        "under": ([BOX_A], "resolved"),  # one box over both columns: scored, not penalised
        "at": ([HALF_L, HALF_R], "resolved"),
        "over": ([HALF_T, HALF_B, third], "multi_box_penalized"),
    }
    for name, (boxes, status) in cases.items():
        result = score_markdown_attribution(output([A], [boxes]), rule([column_paragraph()]))
        row = result["elements"][0]
        assert row["status"] == status, name
        assert row["expected_bbox_count"] == 2
        assert result["multi_box_penalized"] == int(status == "multi_box_penalized")
    assert score_markdown_attribution(output([A], [[HALF_L, HALF_R]]), rule([column_paragraph()]))[
        "quality"
    ] == pytest.approx(1)
    over = score_markdown_attribution(output([A], [[HALF_T, HALF_B, third]]), rule([column_paragraph()]))
    assert over["quality"] == 0
    assert over["elements"][0]["unpenalized_quality"] == pytest.approx(0.14 / 0.15)


def test_one_box_over_a_two_column_paragraph_is_charged_for_the_gutter():
    left, right = [0.1, 0.1, 0.3, 0.2], [0.5, 0.1, 0.3, 0.2]  # 0.1-wide gutter
    r = reference(A, left, expected_bbox_count=2, exception="multi_column")
    r["regions"].append({"page": 1, "bbox": right})
    result = score_markdown_attribution(output([A], [[0.1, 0.1, 0.7, 0.2]]), rule([r]))
    assert result["elements"][0]["status"] == "resolved"
    assert result["quality"] == pytest.approx(0.12 / 0.14)


# A form row: three labels side by side, far apart, on one line.
ROW = [
    ("City:", [0.10, 0.50, 0.05, 0.02]),
    ("State:", [0.45, 0.50, 0.06, 0.02]),
    ("ZIP Code:", [0.75, 0.50, 0.09, 0.02]),
]


def row_rule(**kwargs):
    return rule([reference(text, box, i) for i, (text, box) in enumerate(ROW)], **kwargs)


def test_same_row_merge_resolves_in_left_to_right_order():
    line = "City: [____] State: [____] ZIP Code: [____]"
    result = score_markdown_attribution(output([line], [[box for _, box in ROW]]), row_rule())
    element = result["elements"][0]
    assert element["status"] == "resolved"
    assert element["reference_ids"] == ["ref-0", "ref-1", "ref-2"]
    # One box per field is within the row's budget and matches the union exactly.
    assert element["expected_bbox_count"] == 3
    assert element["quality"] == pytest.approx(1)
    assert result["sqrt_area_coverage"] == pytest.approx(1)


def test_same_row_merge_is_scored_on_the_union_not_the_envelope():
    line = "City: State: ZIP Code:"
    row_box = [0.10, 0.50, 0.74, 0.02]
    element = score_markdown_attribution(output([line], [row_box]), row_rule())["elements"][0]
    # One box over the whole row covers the blank space between the labels.
    assert element["status"] == "resolved"
    assert element["quality"] == pytest.approx((0.05 + 0.06 + 0.09) / 0.74)


def test_same_row_merge_over_budget_is_penalised():
    # Four stacked strips: none are row pieces, so four boxes against a budget of three.
    boxes = [[0.10, 0.50 + 0.005 * i, 0.74, 0.005] for i in range(4)]
    element = score_markdown_attribution(output(["City: State: ZIP Code:"], [boxes]), row_rule())["elements"][0]
    assert element["status"] == "multi_box_penalized"


def test_margin_note_written_before_its_paragraph_merges():
    paragraph = reference(A + " " + B, [0.25, 0.40, 0.65, 0.20], 0)
    note = reference("42 USC 2133.", [0.08, 0.45, 0.12, 0.02], 1)
    # GT order puts the note after the paragraph; the Markdown writes it first.
    element = score_markdown_attribution(
        output(["42 USC 2133. " + A + " " + B], [[0.08, 0.40, 0.82, 0.20]]), rule([paragraph, note])
    )["elements"][0]
    assert element["status"] == "resolved"
    assert element["reference_ids"] == ["ref-0", "ref-1"]


def test_rows_need_vertical_overlap_and_no_horizontal_overlap():
    offset = [reference("City:", [0.1, 0.50, 0.05, 0.02]), reference("State:", [0.45, 0.53, 0.06, 0.02], 1)]
    stacked = [reference("City:", [0.1, 0.50, 0.3, 0.02]), reference("State:", [0.2, 0.505, 0.3, 0.02], 1)]
    for refs in (offset, stacked):
        element = score_markdown_attribution(output(["City: State:"], [[0.1, 0.5, 0.4, 0.05]]), rule(refs))
        # Not merged into one row: at most a relaxed link to one of the two.
        assert element["elements"][0]["match"] != "text"
        assert element["elements"][0]["reference_ids"] != ["ref-0", "ref-1"]


def test_default_merge_cap_is_six_stacked_lines():
    lines = [f"Line {word} of the scorecard" for word in ("one", "two", "three", "four", "five", "six")]
    refs = [reference(t, [0.1, 0.1 + 0.03 * i, 0.5, 0.02], i) for i, t in enumerate(lines)]
    merged = "\n".join(lines)
    result = score_markdown_attribution(output([merged], [[0.1, 0.1, 0.5, 0.17]]), rule(refs))
    assert result["elements"][0]["status"] == "resolved"
    assert len(result["elements"][0]["reference_ids"]) == 6
    capped = score_markdown_attribution(output([merged], [[0.1, 0.1, 0.5, 0.17]]), rule(refs, max_merge_elements=5))
    # The text match cannot merge six; the fuzzy fallback's merges obey the same cap.
    assert capped["elements"][0]["match"] == "fuzzy"
    assert len(capped["elements"][0]["reference_ids"]) == 5


def test_merged_references_have_a_one_box_budget():
    next_box = [0.1, 0.32, 0.7, 0.2]
    refs = [reference(A, BOX_A, expected_bbox_count=2), reference(B, next_box, 1)]
    with pytest.warns(UserWarning):
        r = rule(refs)
    result = score_markdown_attribution(output([A + " " + B], [[BOX_A, next_box]]), r)
    assert result["elements"][0]["status"] == "multi_box_penalized"
    assert result["elements"][0]["expected_bbox_count"] == 1


def test_expected_bbox_count_validation():
    for bad in [0, -1, 1.5, "2", True]:
        with pytest.raises(ValueError):
            reference_model(reference(A, BOX_A, expected_bbox_count=bad))
    with pytest.warns(UserWarning, match="exceeds its 1 ground-truth region"):
        reference_model(reference(A, BOX_A, expected_bbox_count=2))
    with pytest.warns(UserWarning, match="2 ground-truth regions but expected_bbox_count 1"):
        reference_model(column_paragraph(expected_bbox_count=1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert reference_model(column_paragraph()).expected_bbox_count == 2
    parsed = coerce_parse_rule(rule([column_paragraph()]).model_dump())
    assert isinstance(parsed, MarkdownAttributionRule)
    assert parsed.references[0].expected_bbox_count == 2


def test_column_pieces_as_spans_credit_the_parent_by_area():
    r = column_paragraph()
    left_text, right_text = "Revenue increased by twelve", "percent this year."
    r["spans"] = [reference(left_text, HALF_L, 10), reference(right_text, HALF_R, 11)]
    both = score_markdown_attribution(output([left_text, right_text], [HALF_L, HALF_R]), rule([r]))
    assert [e["reference_ids"] for e in both["elements"]] == [["ref-10"], ["ref-11"]]
    assert both["area_quality"] == pytest.approx(1)
    one = score_markdown_attribution(output([left_text], [HALF_L]), rule([r]))
    assert one["references"][0]["status"] == "assessed"
    assert one["area_quality"] == pytest.approx(0.5)  # half the paragraph's area


def test_single_box_and_unresolved_claims_are_never_penalised():
    for actual in [output([A, B], [BOX_A, [0, 0, 1, 1]]), output(["Unknown words here."], [[HALF_L, HALF_R]])]:
        result = score_markdown_attribution(actual, rule())
        assert result["multi_box_penalized"] == 0
        assert all(row["status"] != "multi_box_penalized" for row in result["elements"])


def test_exact_duplicate_boxes_are_one_box_not_a_multi_box_claim():
    actual = output([A], [[BOX_A, BOX_A]])
    result = score_markdown_attribution(actual, rule())
    assert len(result["elements"][0]["predicted_regions"]) == 1
    assert result["elements"][0]["status"] == "resolved"
    assert result["quality"] == pytest.approx(1)
    assert result["multi_box_penalized"] == 0


# Area weighting: one big paragraph (0.4 of the page) and three tiny page numbers
# (0.0002 each). Short texts must match exactly, so the numbers are unique.
BIG = [0.1, 0.1, 0.8, 0.5]
TINY = [[0.05 + 0.3 * i, 0.95, 0.02, 0.01] for i in range(3)]
TINY_ELSEWHERE = [[0.05 + 0.3 * i, 0.02, 0.02, 0.01] for i in range(3)]
NUMBERS = ["12", "13", "14"]


def big_and_tiny_rule(**kwargs):
    refs = [reference(A, BIG)] + [
        reference(n, box, i + 1) for i, (n, box) in enumerate(zip(NUMBERS, TINY, strict=True))
    ]
    return rule(refs, **kwargs)


def area_metrics(actual, r):
    return {m.metric_name: m for m in compute_markdown_attribution_metrics(actual, [r])}


def test_area_weighting_big_right_small_wrong_scores_high():
    actual = output([A, *NUMBERS], [BIG, *TINY_ELSEWHERE])
    result = score_markdown_attribution(actual, big_and_tiny_rule())
    assert result["quality"] == pytest.approx(0.25)  # count-weighted: 1 of 4 elements right
    assert result["area_total"] == pytest.approx(0.4006)
    assert result["area_quality"] == pytest.approx(0.4 / 0.4006)
    assert [r["status"] for r in result["references"]] == ["assessed"] * 4
    metric = area_metrics(actual, big_and_tiny_rule())["md_attribution_v1_area_quality"]
    assert metric.value == pytest.approx(0.4 / 0.4006)
    assert metric.metadata["passed"] == pytest.approx(0.4)
    assert metric.metadata["total"] == pytest.approx(0.4006)
    assert metric.metadata["numerator_label"] == "score"


def test_area_weighting_big_wrong_small_right_scores_low():
    actual = output([A, *NUMBERS], [[0.1, 0.1, 0.8, 0.05], *TINY])
    result = score_markdown_attribution(actual, big_and_tiny_rule())
    assert result["elements"][0]["quality"] == pytest.approx(0.1)
    assert result["quality"] == pytest.approx((0.1 + 3) / 4)
    assert result["area_quality"] == pytest.approx((0.1 * 0.4 + 0.0006) / 0.4006)
    assert result["area_quality"] < 0.12


def test_area_weight_is_union_area_clipped_and_missed_references_score_zero():
    left, right = [0.1, 0.1, 0.4, 0.2], [0.3, 0.1, 0.4, 0.2]  # overlap 0.2 wide
    split = reference(A, left, expected_bbox_count=2)
    split["regions"].append({"page": 1, "bbox": right})
    split["exception"] = "multi_column"
    other = reference(B, [0.1, 0.5, 0.6, 0.2], 1)
    # B is never output: a missed reference is weighted at quality 0.
    result = score_markdown_attribution(output([A], [[left, right]]), rule([split, other]))
    areas = {r["reference_id"]: r for r in result["references"]}
    assert areas["ref-0"]["area"] == pytest.approx(0.6 * 0.2)  # union, not 0.16 (sum)
    assert areas["ref-1"]["status"] == "missed"
    assert result["area_by_status"]["missed"] == pytest.approx(0.12)
    assert result["area_quality"] == pytest.approx(0.5)
    assert result["quality"] == pytest.approx(1)  # count-weighted view ignores the miss
    outside = AttributionRegion.model_construct(page=1, bbox=[-0.1, 0.8, 0.3, 0.4])
    assert _clamped_area([outside]) == pytest.approx(0.2 * 0.2)


def test_zero_scorable_area_page_is_excluded_and_counted():
    # References with no text that are not pictures (pictures are linked by
    # reading order) can never be matched, so they carry no weight.
    pictures = [reference("", BIG, kind="Text"), reference("![](img.png)", TINY[0], 1, kind="Formula")]
    actual = output([A], [BIG])
    result = score_markdown_attribution(actual, rule(pictures))
    assert result["area_quality"] is None
    assert result["area_total"] == 0
    assert result["area_by_status"]["unscorable"] == pytest.approx(0.4002)
    metrics = area_metrics(actual, rule(pictures))
    assert "md_attribution_v1_area_quality" not in metrics
    assert metrics["md_attribution_v1_zero_area_excluded_count"].value == 1
    assert area_metrics(actual, rule())["md_attribution_v1_zero_area_excluded_count"].value == 0
    # Repeated identical text is scorable: reading-order anchors can link it.
    # Here no anchor separates the copies, so both are missed.
    repeated = score_markdown_attribution(actual, rule([reference(A, BOX_A), reference(A, BOX_B, 1)]))
    assert repeated["area_quality"] == 0
    assert {r["status"] for r in repeated["references"]} == {"missed"}


def test_area_weighting_charges_penalized_multi_box_by_its_area():
    refs = [reference(A, BIG), reference(NUMBERS[0], TINY[0], 1)]
    halves = [[0.1, 0.1, 0.8, 0.25], [0.1, 0.35, 0.8, 0.25]]  # stacked, together exactly BIG
    actual = output([A, NUMBERS[0]], [halves, TINY[0]])
    penalized = score_markdown_attribution(actual, rule(refs))
    assert penalized["elements"][0]["status"] == "multi_box_penalized"
    assert penalized["quality"] == pytest.approx(0.5)
    assert penalized["area_quality"] == pytest.approx(0.0002 / 0.4002)
    assert penalized["references"][0]["status"] == "assessed"
    assert set(penalized["area_by_status"]) == {"assessed", "missed", "unscorable", "optional", "not_requested"}


# Sqrt-area weighting: the area machinery with weight sqrt(area) instead of area.
SQRT_BIG, SQRT_TINY = math.sqrt(0.4), math.sqrt(0.0002)


def test_sqrt_area_big_right_small_wrong_sits_between_count_and_area():
    actual = output([A, *NUMBERS], [BIG, *TINY_ELSEWHERE])
    result = score_markdown_attribution(actual, big_and_tiny_rule())
    expected = SQRT_BIG / (SQRT_BIG + 3 * SQRT_TINY)
    assert result["sqrt_area_quality"] == pytest.approx(expected)
    assert round(expected, 3) == 0.937
    assert result["quality"] < result["sqrt_area_quality"] < result["area_quality"]
    assert (round(result["quality"], 3), round(result["area_quality"], 3)) == (0.25, 0.999)
    metric = area_metrics(actual, big_and_tiny_rule())["md_attribution_v1_sqrt_area_quality"]
    assert metric.value == pytest.approx(expected)
    assert metric.metadata["passed"] == pytest.approx(SQRT_BIG)
    assert metric.metadata["total"] == pytest.approx(SQRT_BIG + 3 * SQRT_TINY)
    assert metric.metadata["numerator_label"] == "score"
    assert metric.metadata["weighting"] == "sqrt_area"


def test_sqrt_area_big_wrong_small_right():
    actual = output([A, *NUMBERS], [[0.1, 0.1, 0.8, 0.05], *TINY])
    result = score_markdown_attribution(actual, big_and_tiny_rule())
    expected = (0.1 * SQRT_BIG + 3 * SQRT_TINY) / (SQRT_BIG + 3 * SQRT_TINY)
    assert result["sqrt_area_quality"] == pytest.approx(expected)
    assert result["area_quality"] < result["sqrt_area_quality"] < result["quality"]


def test_sqrt_area_uses_union_area_and_leaves_missed_references_to_coverage():
    left, right = [0.1, 0.1, 0.4, 0.2], [0.3, 0.1, 0.4, 0.2]  # overlap 0.2 wide
    split = reference(A, left, expected_bbox_count=2)
    split["regions"].append({"page": 1, "bbox": right})
    split["exception"] = "multi_column"
    other = reference(B, [0.1, 0.5, 0.6, 0.2], 1)
    result = score_markdown_attribution(output([A], [[left, right]]), rule([split, other]))
    union = math.sqrt(0.6 * 0.2)  # sqrt of the union 0.12, not of the 0.16 sum
    assert result["sqrt_area_total"] == pytest.approx(union)
    assert result["sqrt_area_scorable"] == pytest.approx(union + math.sqrt(0.12))
    assert result["sqrt_area_quality"] == pytest.approx(1)  # B missed: not scored
    assert result["sqrt_area_coverage"] == pytest.approx(0.5)
    assert result["area_quality"] == pytest.approx(0.5)  # the area metric still charges it


def test_missed_reference_lowers_coverage_not_the_headline():
    both = area_metrics(output([A, *NUMBERS], [BIG, *TINY]), big_and_tiny_rule())
    # The paragraph is not output at all: missed.
    missing = area_metrics(output(NUMBERS, TINY), big_and_tiny_rule())
    assert both["md_attribution_v1_sqrt_area_quality"].value == pytest.approx(1)
    assert missing["md_attribution_v1_sqrt_area_quality"].value == pytest.approx(1)
    assert both["md_attribution_v1_sqrt_area_coverage"].value == pytest.approx(1)
    coverage = missing["md_attribution_v1_sqrt_area_coverage"]
    assert coverage.value == pytest.approx(3 * SQRT_TINY / (SQRT_BIG + 3 * SQRT_TINY))
    assert coverage.metadata["passed"] == pytest.approx(3 * SQRT_TINY)
    assert coverage.metadata["total"] == pytest.approx(SQRT_BIG + 3 * SQRT_TINY)
    assert missing["md_attribution_v1_area_quality"].value == pytest.approx(0.0006 / 0.4006)
    # Every reference missed: coverage 0, no headline value (absent, not 0 or 1).
    none = area_metrics(output(["Unrelated words entirely here."], [BIG]), big_and_tiny_rule())
    assert "md_attribution_v1_sqrt_area_quality" not in none
    assert none["md_attribution_v1_sqrt_area_coverage"].value == 0


def test_optional_reference_is_out_of_coverage_when_missed_and_scored_when_matched():
    # Not a furniture kind: furniture absent from the Markdown is not_requested whatever the flag.
    header = reference("ANNUAL REPORT 2025 PAGE HEADER BAND", [0.1, 0.02, 0.6, 0.02], 2, kind="Text", optional=True)
    refs = [reference(A, BOX_A), reference(B, BOX_B, 1), header]
    dropped = score_markdown_attribution(output([A, B], [BOX_A, BOX_B]), rule(refs))
    by_id = {r["reference_id"]: r for r in dropped["references"]}
    assert by_id["ref-2"]["status"] == "optional"
    assert dropped["sqrt_area_coverage"] == pytest.approx(1)
    assert dropped["area_quality"] == pytest.approx(1)
    assert dropped["area_by_status"]["optional"] == pytest.approx(0.6 * 0.02)
    # Written and boxed, it is scored like any other reference.
    kept = score_markdown_attribution(
        output([header["text"], A, B], [[0.1, 0.02, 0.3, 0.02], BOX_A, BOX_B]), rule(refs)
    )
    by_id = {r["reference_id"]: r for r in kept["references"]}
    assert by_id["ref-2"]["status"] == "assessed"
    assert by_id["ref-2"]["quality"] == pytest.approx(0.5)
    # Without the flag the same omission is a miss.
    plain = score_markdown_attribution(output([A, B], [BOX_A, BOX_B]), rule([*refs[:2], {**header, "optional": False}]))
    assert plain["sqrt_area_coverage"] < 1


def test_sqrt_area_counts_a_penalized_element_at_zero():
    refs = [reference(A, BIG), reference(NUMBERS[0], TINY[0], 1)]
    halves = [[0.1, 0.1, 0.8, 0.25], [0.1, 0.35, 0.8, 0.25]]
    penalized = score_markdown_attribution(output([A, NUMBERS[0]], [halves, TINY[0]]), rule(refs))
    assert penalized["elements"][0]["status"] == "multi_box_penalized"
    assert penalized["sqrt_area_quality"] == pytest.approx(SQRT_TINY / (SQRT_BIG + SQRT_TINY))


def test_sqrt_area_excludes_unscorable_references_and_empty_pages():
    empty = reference("", TINY[0], 1, kind="Text")
    result = score_markdown_attribution(output([A], [BIG]), rule([reference(A, BIG), empty]))
    assert result["references"][1]["status"] == "unscorable"
    assert result["sqrt_area_total"] == pytest.approx(SQRT_BIG)  # the empty reference carries no weight
    assert result["sqrt_area_quality"] == pytest.approx(1)
    assert result["sqrt_area_coverage"] == pytest.approx(1)
    pictures = rule([reference("", BIG, kind="Text")])
    assert score_markdown_attribution(output([A], [BIG]), pictures)["sqrt_area_quality"] is None
    metrics = area_metrics(output([A], [BIG]), pictures)
    assert "md_attribution_v1_sqrt_area_quality" not in metrics
    assert "md_attribution_v1_sqrt_area_coverage" not in metrics
    assert metrics["md_attribution_v1_zero_area_excluded_count"].value == 1


def test_sqrt_area_micro_pools_weights_across_pages():
    # micro_ is sum(passed) / sum(total) over pages: sum(q * sqrt(a)) / sum(sqrt(a)).
    pages = [
        area_metrics(output([A, *NUMBERS], [BIG, *TINY_ELSEWHERE]), big_and_tiny_rule()),
        area_metrics(output([NUMBERS[0]], [TINY[0]]), rule([reference(NUMBERS[0], TINY[0])])),
    ]
    sqrt_metrics = [page["md_attribution_v1_sqrt_area_quality"] for page in pages]
    pooled = sum(m.metadata["passed"] for m in sqrt_metrics) / sum(m.metadata["total"] for m in sqrt_metrics)
    assert pooled == pytest.approx((SQRT_BIG + SQRT_TINY) / (SQRT_BIG + 4 * SQRT_TINY))


def html_table(rows, header=None):
    head = "<tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr>" if header else ""
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table>{head}{body}</table>"


def pipe_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


SALES = [["North", "12.5", "0"], ["South", "0", "—"], ["East", "7.1", "0"], ["West", "n/a", "—"]]
HEADER = ["Region", "2024", "2025"]
TOP, BOTTOM = [0.1, 0.1, 0.8, 0.3], [0.1, 0.55, 0.8, 0.3]


def test_table_cells_parse_html_and_pipe_to_the_same_multiset():
    html, pipe = html_table(SALES, HEADER), pipe_table(HEADER, SALES)
    assert table_cells(html) == table_cells(pipe)
    assert table_cells(html)["0"] == 3 and table_cells(html)["-"] == 2  # repeats counted, dashes unified
    assert table_cells("Just a paragraph.") is None
    sup = table_cells(html_table([["Europe<sup>2</sup>", "5"]]))
    assert sup == table_cells(pipe_table(["Europe 2", "5"], []))


def test_repeated_values_match_one_to_one():
    gt = table_cells(html_table(SALES, HEADER))
    fewer_zeros = table_cells(html_table([r if r[0] != "East" else ["East", "7.1", "9"] for r in SALES], HEADER))
    assert cell_f1(gt, gt) == 1
    # One "0" became "9": 14 of 15 cells still match, the repeated zeros are not double counted.
    assert cell_f1(fewer_zeros, gt) == pytest.approx(14 / 15)


def test_colspan_expanded_into_repeats_is_one_cell():
    html = '<table><tr><th colspan="3">Amount of Coverage</th></tr><tr><td>A</td><td>B</td><td>C</td></tr></table>'
    pipe = pipe_table(["Amount of Coverage"] * 3, [["A", "B", "C"]])
    assert table_cells(html) == table_cells(pipe)
    assert table_cells(pipe)["amount of coverage"] == 1


def test_table_resolves_by_cells_despite_row_order_and_format():
    refs = [reference(pipe_table(HEADER, SALES), TOP, kind="Table")]
    shuffled = html_table(SALES[::-1], HEADER)
    result = score_markdown_attribution(output([shuffled], [TOP]), rule(refs))
    row = result["elements"][0]
    assert row["status"] == "resolved" and row["match"] == "table_cells"
    assert row["text_similarity"] == pytest.approx(1)
    assert result["quality"] == pytest.approx(1)


def test_two_similar_tables_are_assigned_one_to_one():
    other = [["North", "12.5", "3"], ["South", "0", "—"], ["East", "7.1", "0"], ["West", "n/a", "4"]]
    refs = [reference(pipe_table(HEADER, SALES), TOP, kind="Table"), reference(pipe_table(HEADER, other), BOTTOM, 1)]
    # Both predicted tables score well against the first reference; each still gets its own.
    assert cell_f1(table_cells(html_table(other, HEADER)), table_cells(refs[0]["text"])) > 0.6
    actual = output([html_table(other, HEADER), html_table(SALES, HEADER)], [BOTTOM, TOP])
    result = score_markdown_attribution(actual, rule(refs))
    assert [e["reference_ids"] for e in result["elements"]] == [["ref-1"], ["ref-0"]]
    assert result["quality"] == pytest.approx(1)


def test_a_different_table_below_the_threshold_stays_unresolved():
    refs = [reference(pipe_table(HEADER, SALES), TOP, kind="Table")]
    unrelated = html_table([["Alpha", "1"], ["Beta", "2"], ["Gamma", "3"]], ["Name", "Rank"])
    result = score_markdown_attribution(output([unrelated], [TOP]), rule(refs))
    assert result["elements"][0]["status"] == "unresolved_content"
    assert result["quality"] is None


def test_pipe_table_item_pairs_with_html_markdown():
    # The provider item carries the pipe form, the final Markdown the HTML form.
    html, pipe = html_table(SALES, HEADER), pipe_table(HEADER, SALES)
    refs = [reference(pipe, TOP, kind="Table")]
    actual = output([pipe], [TOP], markdown=html)
    result = score_markdown_attribution(actual, rule(refs))
    assert result["elements"][0]["boxed"] is True
    assert result["quality"] == pytest.approx(1)
    assert result["unlinked_claims"] == 0


CHART = [["2", "1000"], ["4", "350"], ["6", "150"], ["8", "50"]]
CHART_HEADER = ["pH", "Rate"]


def test_chart_reference_accepts_estimates_within_tolerance():
    # A chart reference stores values read off the marks; another careful reading differs a
    # little: within 2% of the value or 5% of the value range (950 of 1000 is 2.6% of the range).
    refs = [reference(pipe_table(CHART_HEADER, CHART), TOP, kind="Picture")]
    reread = [["2", "970"], ["4", "380"], ["6", "140"], ["8", "55"]]
    result = score_markdown_attribution(output([pipe_table(CHART_HEADER, reread)], [TOP]), rule(refs))
    row = result["elements"][0]
    assert row["status"] == "resolved" and row["match"] == "table_cells"
    assert row["text_similarity"] == pytest.approx(1)


def test_chart_values_outside_tolerance_do_not_match():
    chart = chart_cells(pipe_table(CHART_HEADER, CHART))
    far = table_cells(pipe_table(CHART_HEADER, [["2", "600"], ["4", "350"], ["6", "150"], ["8", "50"]]))
    assert chart_cell_f1(far, chart) == pytest.approx(9 / 10)  # 1000 vs 600 is 40% off


def test_ordinary_table_numbers_stay_exact():
    refs = [reference(pipe_table(CHART_HEADER, CHART), TOP, kind="Table")]
    reread = [["2", "950"], ["4", "380"], ["6", "140"], ["8", "55"]]
    gt, pred = table_cells(refs[0]["text"]), table_cells(pipe_table(CHART_HEADER, reread))
    # Headers and row labels match, the four changed values earn nothing: 6 of 10 cells.
    assert cell_f1(pred, gt) == pytest.approx(0.6)
    result = score_markdown_attribution(output([pipe_table(CHART_HEADER, reread)], [TOP]), rule(refs))
    assert result["elements"][0]["text_similarity"] == pytest.approx(0.6)


def test_headline_rows_put_sqrt_area_micro_first():
    aggregate = {
        "avg_md_attribution_v1_generated_elements": 3.0,
        "avg_md_attribution_v1_quality": 0.8,
        "micro_md_attribution_v1_quality": 0.7,
        "avg_md_attribution_v1_area_quality": 0.9,
        "micro_md_attribution_v1_area_quality": 0.85,
        "total_md_attribution_v1_zero_area_excluded_count": 1.0,
        "avg_md_attribution_v1_sqrt_area_quality": 0.75,
        "micro_md_attribution_v1_sqrt_area_quality": 0.78,
        "micro_md_attribution_v1_sqrt_area_coverage": 0.83,
    }
    rows = headline_rows(aggregate)
    assert [label for label, _ in rows] == [
        "Md Attribution Sqrt-Area-Weighted Quality H (Micro, Assessed Only, Headline)",
        "Md Attribution Sqrt-Area Coverage C (Micro, Assessed / Scorable)",
        "Md Attribution F (Combined 2HC/(H+C) of Micro H and C)",
        "Md Attribution Sqrt-Area-Weighted Quality H (Page Mean, Secondary)",
        "Md Attribution Count-Weighted Quality (Micro Element-Weighted, Secondary)",
        "Md Attribution Count-Weighted Quality (Page Mean, Secondary)",
        "Md Attribution Area-Weighted Quality (Page Mean, Missed Count 0, Secondary)",
        "Md Attribution Area-Weighted Quality (Micro, Missed Count 0, Secondary)",
        "Md Attribution Pages Excluded (Zero Scorable Area)",
    ]
    f = 2 * 0.78 * 0.83 / (0.78 + 0.83)
    assert [value for _, value in rows] == pytest.approx([0.78, 0.83, f, 0.75, 0.7, 0.8, 0.9, 0.85, 1.0])
    assert headline_rows({"avg_rule_pass_rate": 1.0}) == []


def test_md_inventory_includes_unboxed_blocks_not_just_layout_items():
    result = score_markdown_attribution(output([A], [BOX_A], markdown=A + "\n\n" + B), rule())
    assert result["generated"] == 2
    assert result["box_coverage"] == 0.5
    assert result["quality"] == pytest.approx(1)
    elements = markdown_elements("# Heading\n\n- Item one\n- Item two\n\n| A | B |\n|---|---|\n|1|2|\n")
    assert [e.kind for e in elements] == ["heading", "paragraph", "paragraph", "table"]


def test_invalid_geometry_is_coverage_not_quality_and_no_box_hunting():
    actual = output([A], [[-0.1, 0.1, 0.2, 0.2]])
    result = score_markdown_attribution(actual, rule())
    assert result["boxed"] == 1
    assert result["valid_boxed"] == 0
    assert result["quality"] is None
    assert result["elements"][0]["status"] == "invalid_geometry"


def test_schema_roundtrip_and_unknown_version_rejected():
    parsed = coerce_parse_rule(rule().model_dump())
    assert isinstance(parsed, MarkdownAttributionRule)
    assert coerce_parse_rule(parsed.model_dump()).model_dump() == parsed.model_dump()
    with pytest.raises(ValueError):
        MarkdownAttributionRule(**{**rule().model_dump(), "version": 3})
    # The penalty is always on; the retired opt-in switch is rejected, not ignored.
    with pytest.raises(ValueError):
        MarkdownAttributionRule(**{**rule().model_dump(), "multi_box_policy": "penalize"})
    with pytest.raises(ValueError):
        AttributionRegion(page=1, bbox=[0, 0, float("nan"), 1])


def test_real_evaluator_new_rule_is_not_a_binary_parse_rule():
    actual = output([A], [BOX_A])
    case = ParseTestCase(test_id="test", group="v0.3", file_path=Path("test.pdf"), test_rules=[rule().model_dump()])
    inference = InferenceResult(
        request=InferenceRequest(example_id="test", source_file_path="test.pdf", product_type=ProductType.PARSE),
        pipeline_name="test",
        product_type=ProductType.PARSE,
        raw_output={},
        output=actual,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        latency_in_ms=0,
    )
    evaluator = ParseEvaluator(
        enable_text_similarity=False,
        enable_teds=False,
        enable_grits=False,
        enable_header_accuracy=False,
        enable_structural_consistency=False,
        enable_table_record_match=False,
        enable_table_composite=False,
    )
    result = evaluator.evaluate(inference, case)
    values = {m.metric_name: m.value for m in result.metrics}
    assert values["md_attribution_v1_quality"] == pytest.approx(1)
    assert values["md_attribution_v1_area_quality"] == pytest.approx(0.14 / 0.28)  # B is missed
    assert values["md_attribution_v1_sqrt_area_quality"] == pytest.approx(1)  # B missed: not scored
    assert values["md_attribution_v1_sqrt_area_coverage"] == pytest.approx(0.5)  # equal areas
    assert "rule_pass_rate" not in values
    assert not any(name.startswith("layout_") for name in values)
    case.test_rules = []
    case.expected_markdown = A
    assert not any(m.metric_name.startswith("md_attribution_") for m in evaluator.evaluate(inference, case).metrics)


# ---------------------------------------------------------------------------
# Display fields for the report preview. They are added to the per-example
# output only; every metric value and every pre-existing metadata field must stay
# byte-identical. The headline values below were produced by the scorer before
# the display fields existed (merge df88d41e42); scorer version 2 left them
# unchanged on this page (with the dilation off) and added the diagnostics.
# ---------------------------------------------------------------------------

C = "Liabilities were refinanced at a lower rate during the second half."
D = "Dividends are paid twice a year to shareholders of record."
E = "Headcount grew in every region except the smallest one."
F = "An unrelated sentence that no reference in this rule contains."
BOX_C, BOX_D, BOX_E = [0.1, 0.35, 0.35, 0.1], [0.55, 0.35, 0.35, 0.1], [0.1, 0.85, 0.7, 0.05]


def every_status_page():
    """A: misplaced box. B: exact. C: two stacked boxes over a one-box budget. D: no box.
    F: text no reference holds. A claim missing from the final Markdown. E: never output."""
    refs = [reference(t, b, i) for i, (t, b) in enumerate(((A, BOX_A), (B, BOX_B), (C, BOX_C), (D, BOX_D), (E, BOX_E)))]
    actual = output(
        [A, B, C, D, F, "Stray caption text"],
        [
            [0.3, 0.15, 0.7, 0.2],
            BOX_B,
            [[0.1, 0.35, 0.35, 0.05], [0.1, 0.4, 0.35, 0.05]],
            None,
            [0.5, 0.6, 0.4, 0.1],
            [0.1, 0.95, 0.3, 0.03],
        ],
        markdown="\n\n".join([A, B, C, D, F]),
    )
    return actual, rule(refs)


GOLDEN_VALUES = {
    "md_attribution_v1_sqrt_area_quality": 0.5463414634146343,
    "md_attribution_v1_sqrt_area_coverage": 0.7142857142857142,
    "md_attribution_v1_area_quality": 0.49667405764966754,
    "md_attribution_v1_generated_elements": 5.0,
    "md_attribution_v1_quality": 0.4552845528455285,
    "md_attribution_v1_box_coverage": 0.8,
    "md_attribution_v1_valid_contract_coverage": 0.6,
    "md_attribution_v1_assessment_coverage": 0.75,
    "md_attribution_v1_missed_absent_share": 0.14285714285714288,
    "md_attribution_v1_missed_unlinked_share": 0.1428571428571428,
    "md_attribution_v1_order_tau": 1.0,
    "md_attribution_v1_fuzzy_weight_share": 0.0,
    "md_attribution_v1_fuzzy_links_count": 0.0,
    "md_attribution_v1_relaxed_links_count": 0.0,
    "md_attribution_v1_boxed_count": 4.0,
    "md_attribution_v1_assessed_count": 3.0,
    "md_attribution_v1_unlinked_claims_count": 1.0,
    "md_attribution_v1_multi_box_penalized_count": 1.0,
    "md_attribution_v1_zero_area_excluded_count": 0.0,
}
GOLDEN_SCALAR_METADATA = {
    "version": 2,
    "parameters": {
        "text_match_threshold": 0.92,
        "ambiguity_margin": 0.04,
        "max_merge_elements": 6,
        "table_match_threshold": 0.6,
    },
    "multi_box_penalized": 1,
    "generated": 5,
    "boxed": 4,
    "assessed": 3,
    "valid_boxed": 3,
    "quality": 0.4552845528455285,
    "box_coverage": 0.8,
    "valid_contract_coverage": 0.6,
    "assessment_coverage": 0.75,
    "unlinked_claims": 1,
    "statuses": {"resolved": 2, "multi_box_penalized": 1, "unboxed": 1, "unresolved_content": 1},
    "area_total": 0.385,
    "area_score": 0.191219512195122,
    "area_quality": 0.49667405764966754,
    "area_by_status": {
        "assessed": 0.315,
        "missed": 0.07000000000000002,
        "unscorable": 0.0,
        "optional": 0.0,
        "not_requested": 0.0,
    },
    "sqrt_area_total": 0.9354143466934853,
    "sqrt_area_score": 0.5110556430715628,
    "sqrt_area_scorable": 1.3095800853708797,
    "sqrt_area_quality": 0.5463414634146343,
    "sqrt_area_coverage": 0.7142857142857142,
    "p0_score": 1.3658536585365855,
    "p0_total": 3.0,
    "p1_score": 0.191219512195122,
    "p1_total": 0.31500000000000006,
    "sqrt_area_missed_absent": 0.18708286933869714,
    "sqrt_area_fuzzy": 0.0,
    "sqrt_area_missed_unlinked": 0.18708286933869703,
    "furniture": {"score": 0.0, "total": 0.0, "scorable": 0.0, "quality": None, "coverage": None, "references": []},
    "relaxed_links": 0,
    "fuzzy_links": 0,
    "fuzzy_weight_share": 0.0,
    "kind_agreement": {
        "score": 0.0,
        "total": 0.0,
        "confusion": {},
        "classes": {
            "heading": {"tp": 0, "fp": 0, "fn": 0},
            "paragraph": {"tp": 0, "fp": 0, "fn": 0},
            "table": {"tp": 0, "fp": 0, "fn": 0},
            "image": {"tp": 0, "fp": 0, "fn": 0},
            "formula": {"tp": 0, "fp": 0, "fn": 0},
        },
    },
    "order_tau": {"tau": 1.0, "concordant": 3, "discordant": 0, "ties": 0},
    "count": 5,
}
GOLDEN_OTHER_METADATA = [
    {
        "passed": 0.5110556430715628,
        "total": 0.9354143466934853,
        "numerator_label": "score",
        "weighting": "sqrt_area",
        "scorer_version": 2,
        "p0_score": 1.3658536585365855,
        "p0_total": 3.0,
        "p1_score": 0.191219512195122,
        "p1_total": 0.31500000000000006,
    },
    {
        "passed": 0.9354143466934853,
        "total": 1.3095800853708797,
        "numerator_label": "score",
        "weighting": "sqrt_area",
        "scorer_version": 2,
    },
    {
        "passed": 0.191219512195122,
        "total": 0.385,
        "numerator_label": "score",
        "weighting": "area",
        "area_by_status": {
            "assessed": 0.315,
            "missed": 0.07000000000000002,
            "unscorable": 0.0,
            "optional": 0.0,
            "not_requested": 0.0,
        },
        "scorer_version": 2,
    },
    {"passed": 1.3658536585365855, "total": 3, "numerator_label": "score", "scorer_version": 2},
    {"passed": 4, "total": 5, "numerator_label": "passed", "scorer_version": 2},
    {"passed": 3, "total": 5, "numerator_label": "passed", "scorer_version": 2},
    {"passed": 3, "total": 4, "numerator_label": "passed", "scorer_version": 2},
    {"passed": 0.18708286933869714, "total": 1.3095800853708797, "numerator_label": "score", "scorer_version": 2},
    {"passed": 0.18708286933869703, "total": 1.3095800853708797, "numerator_label": "score", "scorer_version": 2},
    {"concordant": 3, "discordant": 0, "ties": 0, "scorer_version": 2},
    {
        "passed": 0.0,
        "total": 0.9354143466934853,
        "numerator_label": "score",
        "scorer_version": 2,
        "fuzzy_links": 0,
        "relaxed_links": 0,
    },
    {"count": 0, "scorer_version": 2},
    {"count": 0, "scorer_version": 2},
    {"count": 4},
    {"count": 3},
    {"count": 1},
    {"count": 1},
    {"count": 0},
]


def test_display_fields_leave_every_metric_byte_identical():
    actual, r = every_status_page()
    metrics = compute_markdown_attribution_metrics(actual, [r])
    # Exact float equality on purpose: the display fields must not move any value.
    assert {m.metric_name: m.value for m in metrics} == GOLDEN_VALUES
    assert [m.metric_name for m in metrics] == list(GOLDEN_VALUES)
    generated = next(m for m in metrics if m.metric_name == "md_attribution_v1_generated_elements").metadata
    assert {k: v for k, v in generated.items() if not isinstance(v, list)} == GOLDEN_SCALAR_METADATA
    assert [m.metadata for m in metrics if m.metric_name != "md_attribution_v1_generated_elements"] == (
        GOLDEN_OTHER_METADATA
    )
    assert [(e["element_id"], e["status"], e["quality"], e["reference_ids"]) for e in generated["elements"]] == [
        ("md-1", "resolved", 0.3658536585365854, ["ref-0"]),
        ("md-2", "resolved", 1.0, ["ref-1"]),
        ("md-3", "multi_box_penalized", 0.0, ["ref-2"]),
        ("md-4", "unboxed", None, []),
        ("md-5", "unresolved_content", None, []),
    ]
    assert [(r["reference_id"], r["status"], r["quality"]) for r in generated["references"]] == [
        ("ref-0", "assessed", 0.3658536585365854),
        ("ref-1", "assessed", 1.0),
        ("ref-2", "assessed", 0.0),
        ("ref-3", "missed", 0.0),
        ("ref-4", "missed", 0.0),
    ]


def test_output_carries_reference_regions_spans_and_unlinked_claims_for_drawing():
    result = score_markdown_attribution(*every_status_page())
    missed = result["references"][4]
    assert missed["text"] == E
    assert missed["regions"] == [{"page": 1, "bbox": BOX_E}]
    assert missed["expected_bbox_count"] == 1 and missed["order"] == 4 and missed["spans"] == []
    penalized = result["elements"][2]
    assert len(penalized["predicted_regions"]) == 2 and penalized["target_regions"] == [{"page": 1, "bbox": BOX_C}]
    assert result["unlinked"] == [
        {
            "kind": "paragraph",
            "text": "Stray caption text",
            "invalid": False,
            "regions": [{"page": 1, "bbox": [0.1, 0.95, 0.3, 0.03]}],
        }
    ]
    spanned = reference(A + " " + B, [0.1, 0.1, 0.7, 0.7], spans=[reference(A, BOX_A, 1) | {"id": "span-a"}])
    spans = score_markdown_attribution(output([A], [BOX_A]), rule([spanned]))["references"][0]["spans"]
    assert spans == [
        {
            "text": A,
            "order": 1,
            "expected_bbox_count": 1,
            "regions": [{"page": 1, "bbox": BOX_A}],
            "reference_id": "span-a",
        }
    ]
    long_text = "word " * 100
    snippet = score_markdown_attribution(output([A], [BOX_A]), rule([reference(long_text, BOX_A)]))["references"][0][
        "text"
    ]
    assert len(snippet) == 240 and snippet.endswith("…")
