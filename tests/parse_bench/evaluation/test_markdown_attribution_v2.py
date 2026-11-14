"""Scorer version 2 of markdown_attribution: each change, with its negative case.

Version 2 is the only scorer: a rule with no ``version``, or ``version: 1``, is
scored the same way. test_markdown_attribution_golden.py freezes its output on
real saved outputs.
"""

import math

import pytest

from parse_bench.evaluation.markdown_attribution import (
    _box_count,
    cell_f1,
    chart_cell_f1,
    chart_cells,
    compute_markdown_attribution_metrics,
    geometry,
    score_markdown_attribution,
    table_cells,
)
from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, ParseLayoutPageIR, ParseOutput
from parse_bench.test_cases.markdown_attribution_schema import AttributionRegion, MarkdownAttributionRule

A = "Revenue increased by twelve percent this year."
B = "Operating expenses remained stable across the company."
BOX_A = [0.1, 0.1, 0.7, 0.2]
BOX_B = [0.1, 0.6, 0.7, 0.2]


def ref(text, box, index=0, **kwargs):
    boxes = box if isinstance(box[0], list) else [box]
    return {
        "id": f"ref-{index}",
        "text": text,
        "order": index,
        "regions": [{"page": 1, "bbox": b} for b in boxes],
        **kwargs,
    }


def rule(refs, **kwargs):
    return MarkdownAttributionRule(references=refs, **kwargs)


def output(texts, boxes, *, markdown=None, size=(612, 792), types=None):
    items = []
    for i, (text, box) in enumerate(zip(texts, boxes, strict=True)):
        regions = [] if box is None else (box if isinstance(box[0], list) else [box])
        items.append(
            LayoutItemIR(
                md=text,
                type=types[i] if types else None,
                layout_segments=[LayoutSegmentIR(x=b[0], y=b[1], w=b[2], h=b[3]) for b in regions],
            )
        )
    return ParseOutput(
        example_id="test",
        pipeline_name="test",
        markdown="\n\n".join(t for t in texts if t) if markdown is None else markdown,
        layout_pages=[ParseLayoutPageIR(page_number=1, width=size[0], height=size[1], items=items)],
    )


def score(texts, boxes, refs, **kwargs):
    return score_markdown_attribution(output(texts, boxes, **kwargs), rule(refs))


def region(box):
    return AttributionRegion(page=1, bbox=box)


# ---------------------------------------------------------------- 6: row pieces

LINE = [0.1, 0.2, 0.6, 0.02]
LEFT_PIECE, RIGHT_PIECE = [0.1, 0.2, 0.28, 0.02], [0.4, 0.2, 0.3, 0.02]  # a gap between them


def test_row_piece_test_on_the_three_cases_of_the_diagram():
    # Left: two pieces on one baseline, horizontally disjoint -> one box.
    assert _box_count([region(LEFT_PIECE), region(RIGHT_PIECE)]) == 1
    # Middle: a paragraph cut into two stacked halves -> two boxes.
    assert _box_count([region([0.1, 0.2, 0.6, 0.05]), region([0.1, 0.25, 0.6, 0.05])]) == 2
    # Right: a wrapper box plus a line box inside it -> two boxes.
    assert _box_count([region([0.1, 0.2, 0.6, 0.1]), region([0.1, 0.22, 0.6, 0.02])]) == 2
    # A chain through a tall box would join two stacked lines: not one row.
    chain = [region([0.1, 0.2, 0.2, 0.02]), region([0.3, 0.2, 0.2, 0.05]), region([0.1, 0.23, 0.2, 0.02])]
    assert _box_count(chain) == 3


def test_line_claimed_as_two_row_pieces_is_scored_on_the_union():
    line = "Name of the applicant and the mailing address of the office"
    refs = [ref(line, LINE)]
    new = score([line], [[LEFT_PIECE, RIGHT_PIECE]], refs)["elements"][0]
    assert new["status"] == "resolved" and new["row_pieces"] == 2
    union = (0.28 + 0.3) * 0.02 / (0.6 * 0.02)
    assert new["raw_quality"] == pytest.approx(union)
    assert new["contract_valid"] is True


def test_stacked_halves_and_many_boxes_stay_penalized():
    refs = [ref(A, BOX_A)]
    halves = [[0.1, 0.1, 0.7, 0.1], [0.1, 0.2, 0.7, 0.1]]
    element = score([A], [halves], refs)["elements"][0]
    assert element["status"] == "multi_box_penalized" and element["quality"] == 0
    # Three rows of a reference entry against a budget of 2 stay at 0.
    rows = [[0.1, 0.1 + 0.03 * i, 0.7, 0.02] for i in range(3)]
    budget_two = ref(A, [[0.1, 0.1, 0.3, 0.2], [0.5, 0.1, 0.3, 0.2]], exception="multi_column", expected_bbox_count=2)
    element = score([A], [rows], [budget_two])["elements"][0]
    assert element["status"] == "multi_box_penalized"
    # Thirteen boxes over a paragraph, lines stacked, stay at 0.
    lines = [[0.1, 0.1 + 0.015 * i, 0.7, 0.012] for i in range(13)]
    assert score([A], [lines], refs)["elements"][0]["status"] == "multi_box_penalized"


# ---------------------------------------------------------------- 7: dilation

PT = 1 / 612, 1 / 792  # one point as a page fraction on US Letter


def test_one_point_of_padding_on_a_small_label_is_mostly_forgiven():
    # A 6 pt line number, ink-fitted in the GT, boxed with 1 pt of padding.
    ink = [0.05, 0.5, 6 * PT[0], 6 * PT[1]]
    padded = [ink[0] - PT[0], ink[1] - PT[1], ink[2] + 2 * PT[0], ink[3] + 2 * PT[1]]
    refs = [ref("45", ink)]
    element = score(["45"], [padded], refs)["elements"][0]
    assert element["raw_quality"] == pytest.approx(36 / 64)
    # Both dilated by 1 pt: 8 x 8 inside 10 x 10.
    assert element["quality"] == pytest.approx(64 / 100)


def test_dilation_hardly_moves_a_large_box_and_does_not_rescue_a_wrong_one():
    refs = [ref(A, BOX_A)]
    shifted = [0.1, 0.12, 0.7, 0.2]
    element = score([A], [shifted], refs)["elements"][0]
    assert element["quality"] == pytest.approx(element["raw_quality"], abs=0.01)
    assert element["quality"] > element["raw_quality"]
    assert score([A], [BOX_B], refs)["elements"][0]["quality"] == 0


def test_dilation_uses_the_page_size_in_points_and_clips_to_the_page():
    ink = [0.0, 0.0, 0.01, 0.01]
    refs = [ref("7", ink)]
    # On an A5 page (420 x 595 pt) one point is a larger fraction than on Letter.
    letter = score(["7"], [[0.0, 0.0, 0.02, 0.02]], refs)["elements"][0]["quality"]
    a5 = score(["7"], [[0.0, 0.0, 0.02, 0.02]], refs, size=(420, 595))["elements"][0]["quality"]
    assert a5 > letter
    # At the page corner the dilated box is clipped: nothing lies outside [0, 1].
    x = 1 / 612
    expected = geometry([region([0, 0, 0.02 + x, 0.02 + 1 / 792])], [region([0, 0, 0.01 + x, 0.01 + 1 / 792])])[0]
    assert letter == pytest.approx(expected)


def test_dilation_shrinks_the_gap_between_padded_and_ink_fitted_ground_truth(monkeypatch):
    import parse_bench.evaluation.markdown_attribution as ma

    # The same 8 pt label annotated ink-fitted (v0.3 convention) and padded by 1.5 pt (v0.2).
    ink = [0.3, 0.4, 20 * PT[0], 8 * PT[1]]
    padded = [ink[0] - 1.5 * PT[0], ink[1] - 1.5 * PT[1], ink[2] + 3 * PT[0], ink[3] + 3 * PT[1]]
    prediction = [ink[0] - 0.5 * PT[0], ink[1], ink[2] + PT[0], ink[3]]

    def gap():
        a = score(["Total"], [prediction], [ref("Total", ink)])["elements"][0]["quality"]
        b = score(["Total"], [prediction], [ref("Total", padded)])["elements"][0]["quality"]
        return abs(a - b)

    with_dilation = gap()
    monkeypatch.setattr(ma, "DILATION_PT", 0.0)
    assert with_dilation < gap()


# ---------------------------------------------------------------- 8: charts


def pipe(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


CHART = [["2", "1000"], ["4", "350"], ["6", "150"], ["8", "50"], ["10", "20"]]
HEAD = ["pH", "Rate"]


def test_chart_label_cells_match_exactly():
    gt = pipe(HEAD, CHART)
    # The pH axis read 12 instead of 10: a label, not an estimate.
    wrong_label = table_cells(pipe(HEAD, [*CHART[:4], ["12", "20"]]))
    assert chart_cell_f1(wrong_label, chart_cells(gt)) == pytest.approx(11 / 12)


def test_chart_value_cells_use_a_tolerance_scaled_to_the_value_range():
    chart = chart_cells(pipe(HEAD, CHART))
    assert chart.spread == 980
    # 5% of the 980 range is 49: 1000 read as 955 and 20 read as 60 both match.
    near = table_cells(pipe(HEAD, [["2", "955"], *CHART[1:4], ["10", "60"]]))
    assert chart_cell_f1(near, chart) == 1
    # 1000 read as 900 is outside both 2% of the value and 5% of the range.
    far = table_cells(pipe(HEAD, [["2", "900"], *CHART[1:]]))
    assert chart_cell_f1(far, chart) == pytest.approx(11 / 12)


def test_chart_roles_come_from_the_reference_so_a_transposed_chart_scores_the_same():
    chart = chart_cells(pipe(HEAD, CHART))
    transposed = pipe(["pH", *[r[0] for r in CHART]], [["Rate", *[r[1] for r in CHART]]])
    near = [["2", "955"], *CHART[1:]]
    near_transposed = pipe(["pH", *[r[0] for r in near]], [["Rate", *[r[1] for r in near]]])
    assert chart_cell_f1(table_cells(transposed), chart) == 1
    assert chart_cell_f1(table_cells(near_transposed), chart) == 1


def test_estimate_mark_limits_the_tolerance_to_marked_cells():
    marked = pipe(HEAD, [["2", "≈1000"], ["4", "350"], ["6", "≈150"], ["8", "50"], ["10", "20"]])
    chart = chart_cells(marked)
    assert sum(chart.cells.values()) == 12 and "1000" in chart.cells  # the mark is not part of the text
    # The estimated 1000 may be read as 960; the printed 350 may not be read as 340.
    assert chart_cell_f1(table_cells(pipe(HEAD, [["2", "960"], *CHART[1:]])), chart) == 1
    assert chart_cell_f1(table_cells(pipe(HEAD, [CHART[0], ["4", "340"], *CHART[2:]])), chart) < 1
    # Without any mark, every value cell may use the tolerance.
    assert (
        chart_cell_f1(table_cells(pipe(HEAD, [CHART[0], ["4", "340"], *CHART[2:]])), chart_cells(pipe(HEAD, CHART)))
        == 1
    )


def test_chart_resolution_in_the_scorer_and_ordinary_tables_unchanged():
    refs = [ref(pipe(HEAD, CHART), BOX_A, kind="Picture")]
    reread = pipe(HEAD, [["2", "960"], ["4", "340"], ["6", "160"], ["8", "45"], ["10", "25"]])
    element = score([reread], [BOX_A], refs)["elements"][0]
    assert element["status"] == "resolved" and element["match"] == "table_cells"
    assert element["text_similarity"] == pytest.approx(1)
    table_refs = [ref(pipe(HEAD, CHART), BOX_A, kind="Table")]
    element = score([reread], [BOX_A], table_refs)["elements"][0]
    # Ordinary table numbers stay exact: no direct cell match (a relaxed link may still take it).
    assert element.get("match") != "table_cells"
    assert cell_f1(table_cells(reread), table_cells(pipe(HEAD, CHART))) < 1


# ---------------------------------------------------------------- 2a: absent vs unlinked


def metrics_of(texts, boxes, refs, **kwargs):
    return {
        m.metric_name: m for m in compute_markdown_attribution_metrics(output(texts, boxes, **kwargs), [rule(refs)])
    }


def by_id(result):
    return {r["reference_id"]: r for r in result["references"]}


def test_missed_reference_is_absent_or_unlinked():
    refs = [ref(A, BOX_A), ref(B, BOX_B, 1)]
    # B is written inside a longer unboxed paragraph (so no element resolves to it); A is not written at all.
    markdown = "Intro words. " + B + " And a closing remark that makes the paragraph longer."
    result = score_markdown_attribution(output([markdown], [None]), rule(refs))
    rows = by_id(result)
    assert rows["ref-0"]["missed_reason"] == "absent"
    assert rows["ref-1"]["missed_reason"] == "unlinked"
    assert result["sqrt_area_missed_absent"] + result["sqrt_area_missed_unlinked"] == pytest.approx(
        result["sqrt_area_scorable"] - result["sqrt_area_total"]
    )


def test_absent_test_uses_whole_words_for_short_text_and_cells_for_tables():
    refs = [ref("12", [0.1, 0.9, 0.02, 0.01]), ref(pipe(HEAD, CHART), BOX_A, 1, kind="Table")]
    # "12" occurs only inside "2012"; half of the table's cells occur as words.
    markdown = "Report 2012. pH Rate 2 1000 4 350 6 150"
    rows = by_id(score_markdown_attribution(output([markdown], [None]), rule(refs)))
    assert rows["ref-0"]["missed_reason"] == "absent"
    assert rows["ref-1"]["missed_reason"] == "unlinked"
    rows = by_id(score_markdown_attribution(output(["Report 2012. pH Rate"], [None]), rule(refs)))
    assert rows["ref-1"]["missed_reason"] == "absent"


def test_missed_share_metrics():
    refs = [ref(A, BOX_A), ref(B, BOX_B, 1)]
    metrics = metrics_of([B], [BOX_B], refs)
    absent = metrics["md_attribution_v1_missed_absent_share"]
    assert absent.value == pytest.approx(0.5) and absent.metadata["scorer_version"] == 2
    assert metrics["md_attribution_v1_missed_unlinked_share"].value == 0


# ---------------------------------------------------------------- 15: furniture

HEADER_BOX = [0.1, 0.02, 0.5, 0.02]
HEADER = "ANNUAL REPORT OF THE COMPANY 2025"


def test_furniture_absent_from_the_markdown_is_not_requested_by_default():
    refs = [ref(A, BOX_A), ref(HEADER, HEADER_BOX, 1, kind="Page-header")]
    result = score([A], [BOX_A], refs)
    assert by_id(result)["ref-1"]["status"] == "not_requested"
    assert result["sqrt_area_coverage"] == pytest.approx(1)
    assert result["area_by_status"]["not_requested"] == pytest.approx(0.01)


def test_furniture_written_in_the_markdown_is_scored_or_missed_as_usual():
    refs = [ref(A, BOX_A), ref(HEADER, HEADER_BOX, 1, kind="Page-header")]
    written = score([HEADER, A], [HEADER_BOX, BOX_A], refs)
    assert by_id(written)["ref-1"]["status"] == "assessed"
    # Written but unboxed: present in the Markdown, so it is an unlinked miss, not out of coverage.
    unboxed = score([HEADER, A], [None, BOX_A], refs)
    assert by_id(unboxed)["ref-1"]["status"] == "missed"
    assert by_id(unboxed)["ref-1"]["missed_reason"] == "unlinked"
    # A body reference absent from the Markdown stays missed.
    assert by_id(score([HEADER], [HEADER_BOX], refs))["ref-0"]["status"] == "missed"


def test_furniture_items_are_scored_separately_by_type_and_text():
    refs = [
        ref(A, BOX_A),
        ref(HEADER, HEADER_BOX, 1, kind="Page-header"),
        ref("Page 7", [0.45, 0.96, 0.1, 0.02], 2, kind="Page-footer"),
    ]
    # The header is kept as a typed item but left out of the Markdown; the footer is not kept at all.
    actual = output([A, HEADER], [BOX_A, HEADER_BOX], markdown=A, types=["Text", "Page-header"])
    result = score_markdown_attribution(actual, rule(refs))
    furniture = result["furniture"]
    assert furniture["quality"] == pytest.approx(1)
    assert furniture["coverage"] == pytest.approx(math.sqrt(0.01) / (math.sqrt(0.01) + math.sqrt(0.002)))
    assert by_id(result)["ref-1"]["status"] == "not_requested"  # out of the headline, but boxed as an item
    # An item of another type with the same text does not count.
    untyped = output([A, HEADER], [BOX_A, HEADER_BOX], markdown=A, types=["Text", "Text"])
    assert score_markdown_attribution(untyped, rule(refs))["furniture"]["total"] == 0
    metrics = {m.metric_name for m in compute_markdown_attribution_metrics(actual, [rule(refs)])}
    assert {"md_attribution_v1_furniture_quality", "md_attribution_v1_furniture_coverage"} <= metrics


# ---------------------------------------------------------------- 2b: alternative texts


def test_element_resolves_through_an_alternative_text():
    # A GT typo on a short heading (under 20 characters, so only an exact match resolves).
    refs = [ref("MEOWNERS INSURANCE", BOX_A, alt_texts=["HOMEOWNERS INSURANCE"])]
    element = score(["Homeowners Insurance"], [BOX_A], refs)["elements"][0]
    assert element["status"] == "resolved" and element["reference_ids"] == ["ref-0"]
    # Without the alternative the text path fails (only the looser relaxed pass can link it).
    plain = [ref("MEOWNERS INSURANCE", BOX_A)]
    assert score(["Homeowners Insurance"], [BOX_A], plain)["elements"][0].get("match") != "text"


def test_two_texts_of_one_reference_are_not_rivals():
    # The element is within the ambiguity margin of both the text and the alternative.
    text = "The quarterly revenue of the company increased by twelve percent this year overall"
    alt = text.replace("quarterly", "quarterlv")
    element = score([text.replace("company", "cornpany")], [BOX_A], [ref(text, BOX_A, alt_texts=[alt])])
    assert element["elements"][0]["status"] == "resolved"


def test_alternative_equal_to_another_references_text_is_ambiguous_but_not_a_duplicate():
    refs = [ref(A, BOX_A), ref("Revenue grew by twelve percent this year.", BOX_B, 1, alt_texts=[A])]
    result = score([A], [BOX_A], refs)
    assert result["elements"][0]["status"] == "ambiguous_source"
    # Both references stay scorable: an alternative never makes a reference a duplicate.
    assert {r["status"] for r in result["references"]} == {"missed"}


# ---------------------------------------------------------------- 5: reading-order anchors

TOP_LINE = "Section one introduces the quarterly results in detail."
MID_LINE = "Section two explains the regional breakdown of sales."
END_LINE = "Section three closes with the outlook for next year."
LABEL = "Total amount due"
BOXES5 = [[0.1, 0.1 + 0.1 * i, 0.6, 0.04] for i in range(6)]


def anchored_refs():
    # GT order: TOP, LABEL(1), MID, LABEL(3), END. The two labels are exact duplicates.
    texts = [TOP_LINE, LABEL, MID_LINE, LABEL, END_LINE]
    return [ref(t, BOXES5[i], i) for i, t in enumerate(texts)]


def test_duplicate_text_resolves_inside_its_reading_order_window():
    texts = [TOP_LINE, LABEL, MID_LINE, LABEL, END_LINE]
    result = score(texts, BOXES5[:5], anchored_refs())
    first, second = result["elements"][1], result["elements"][3]
    assert (first["reference_ids"], first["match"]) == (["ref-1"], "anchor")
    assert (second["reference_ids"], second["match"]) == (["ref-3"], "anchor")
    assert first["quality"] == pytest.approx(1) and second["quality"] == pytest.approx(1)


def test_swapped_duplicate_boxes_lower_the_score():
    texts = [TOP_LINE, LABEL, MID_LINE, LABEL, END_LINE]
    swapped = [BOXES5[0], BOXES5[3], BOXES5[2], BOXES5[1], BOXES5[4]]
    result = score(texts, swapped, anchored_refs())
    # The boxes do not pick the reference: each label is still linked by its place, and scored 0.
    assert result["elements"][1]["reference_ids"] == ["ref-1"] and result["elements"][1]["quality"] == 0
    assert result["sqrt_area_quality"] < score(texts, BOXES5[:5], anchored_refs())["sqrt_area_quality"]


def test_duplicates_in_one_window_stay_ambiguous_and_count_as_missed():
    # Without the middle anchor both labels sit in one window (TOP, END).
    texts = [TOP_LINE, LABEL, END_LINE]
    result = score(texts, [BOXES5[0], BOXES5[1], BOXES5[4]], anchored_refs())
    assert result["elements"][1]["status"] == "ambiguous_source"
    assert {by_id(result)[r]["status"] for r in ("ref-1", "ref-3")} == {"missed"}


def picture(index, box, **kwargs):
    return ref("", box, index, kind="Picture", **kwargs)


PIC_A, PIC_B = [0.1, 0.3, 0.35, 0.3], [0.55, 0.3, 0.35, 0.3]


def picture_page(image_boxes, n_images=2, **pic_kwargs):
    refs = [
        ref(TOP_LINE, BOXES5[0], 0),
        picture(1, PIC_A, **pic_kwargs),
        picture(2, PIC_B),
        ref(END_LINE, BOXES5[5], 3),
    ]
    images = [f"![photo: picture {i}](img{i}.jpg)" for i in range(n_images)]
    return score([TOP_LINE, *images, END_LINE], [BOXES5[0], *image_boxes, BOXES5[5]], refs)


def test_images_pair_with_pictures_in_order_and_swapping_them_lowers_the_score():
    right = picture_page([PIC_A, PIC_B])
    assert [e["reference_ids"] for e in right["elements"][1:3]] == [["ref-1"], ["ref-2"]]
    assert all(e["match"] == "picture_anchor" for e in right["elements"][1:3])
    assert right["sqrt_area_quality"] == pytest.approx(1)
    swapped = picture_page([PIC_B, PIC_A])
    assert swapped["elements"][1]["reference_ids"] == ["ref-1"] and swapped["elements"][1]["quality"] == 0
    assert swapped["sqrt_area_quality"] < right["sqrt_area_quality"]


def test_unequal_image_and_picture_counts_stay_ambiguous():
    result = picture_page([PIC_A], n_images=1)
    assert result["elements"][1]["status"] == "ambiguous_picture"
    assert {by_id(result)[r]["status"] for r in ("ref-1", "ref-2")} == {"missed"}
    # Pictures are scorable: an output that drops them loses coverage.
    assert by_id(result)["ref-1"]["missed_reason"] == "unlinked"


def test_decorative_picture_is_out_of_the_weights_unless_an_image_is_written_for_it():
    dropped = score(
        [TOP_LINE, "![photo: p](i.jpg)", END_LINE],
        [BOXES5[0], PIC_B, BOXES5[5]],
        [
            ref(TOP_LINE, BOXES5[0], 0),
            picture(1, PIC_A, decorative=True),
            picture(2, PIC_B),
            ref(END_LINE, BOXES5[5], 3),
        ],
    )
    # One image, two pictures (the decorative one included): ambiguous, and the decorative one is optional.
    assert by_id(dropped)["ref-1"]["status"] == "optional"
    assert by_id(dropped)["ref-2"]["status"] == "missed"
    written = picture_page([PIC_A, PIC_B], decorative=True)
    assert by_id(written)["ref-1"]["status"] == "assessed"


def test_same_text_items_are_linked_by_raw_text_and_neighbours():
    # Two items print the same words; the Markdown orders them differently from the items.
    refs = [ref("Orientations and priorities", BOX_A, 0), ref("ORIENTATIONS AND PRIORITIES", BOX_B, 1)]
    actual = output(
        ["ORIENTATIONS AND PRIORITIES", "Orientations and priorities"],
        [BOX_B, BOX_A],
        markdown="Orientations and priorities\n\nORIENTATIONS AND PRIORITIES",
    )
    # Taking the items in item order would swap the two boxes.
    new = score_markdown_attribution(actual, rule(refs))
    assert [e["predicted_regions"][0]["bbox"] for e in new["elements"]] == [BOX_A, BOX_B]


# ---------------------------------------------------------------- 4: many-to-one groups

COURT = "UNITED STATES DISTRICT COURT\nSOUTHERN DISTRICT OF NEW YORK"
COURT_BOX = [0.3, 0.05, 0.4, 0.05]
LINE_1, LINE_2 = [0.3, 0.05, 0.4, 0.022], [0.3, 0.078, 0.4, 0.022]


def court_page(boxes, extra_refs=()):
    refs = [ref(COURT, COURT_BOX, 0), ref(A, BOX_A, 1), *extra_refs]
    return score(["UNITED STATES DISTRICT COURT", "SOUTHERN DISTRICT OF NEW YORK", A], [*boxes, BOX_A], refs)


def test_consecutive_unresolved_elements_resolve_as_one_group():
    result = court_page([LINE_1, LINE_2])
    heading, line = result["elements"][:2]
    assert heading["status"] == line["status"] == "resolved_group"
    assert heading["reference_ids"] == line["reference_ids"] == ["ref-0"]
    assert heading["group_size"] == 2 and heading["match"] == "group"
    assert heading["quality"] > 0.9  # the two line boxes cover the reference's box but the leading
    assert by_id(result)["ref-0"]["status"] == "assessed"


def test_group_is_scored_on_the_union_and_each_member_keeps_its_one_box_budget():
    half = court_page([LINE_1, [0.8, 0.5, 0.1, 0.02]])
    assert 0 < half["elements"][0]["quality"] < 0.6
    over = court_page([[LINE_1, [0.3, 0.2, 0.4, 0.02]], LINE_2])  # stacked boxes on one member
    assert over["elements"][0]["status"] == "multi_box_penalized" and over["elements"][0]["quality"] == 0


def test_group_needs_the_joined_text_and_a_missed_reference():
    # Only the first line is written: no group.
    alone = score(["UNITED STATES DISTRICT COURT", A], [LINE_1, BOX_A], [ref(COURT, COURT_BOX, 0), ref(A, BOX_A, 1)])
    assert alone["elements"][0]["status"] != "resolved_group"
    # A reference already credited by a single element is not credited again by a group.
    both = score(
        [COURT.replace("\n", " "), "UNITED STATES DISTRICT COURT", "SOUTHERN DISTRICT OF NEW YORK", A],
        [COURT_BOX, LINE_1, LINE_2, BOX_A],
        [ref(COURT, COURT_BOX, 0), ref(A, BOX_A, 1)],
    )
    assert [e["status"] for e in both["elements"][1:3]] != ["resolved_group", "resolved_group"]


def test_table_reference_written_as_paragraphs_groups_by_cells():
    rows = [["North", "12.5"], ["South", "7.1"], ["East", "3.2"]]
    table = pipe(["Region", "Share"], rows)
    refs = [ref(TOP_LINE, BOXES5[0], 0), ref(table, [0.1, 0.2, 0.6, 0.2], 1, kind="Table"), ref(END_LINE, BOXES5[5], 2)]
    pieces = ["Region", "Share", "North", "12.5", "South", "7.1", "East", "3.2"]
    boxes = [[0.1 + 0.3 * (i % 2), 0.2 + 0.05 * (i // 2), 0.3, 0.05] for i in range(8)]
    result = score([TOP_LINE, *pieces, END_LINE], [BOXES5[0], *boxes, BOXES5[5]], refs)
    assert {e["status"] for e in result["elements"][1:9]} == {"resolved_group"}
    assert result["elements"][1]["quality"] == pytest.approx(1)
    # Unrelated short paragraphs do not reach the cell threshold.
    other = ["Alpha", "Beta", "Gamma", "Delta"]
    result = score([TOP_LINE, *other, END_LINE], [BOXES5[0], *boxes[:4], BOXES5[5]], refs)
    assert "resolved_group" not in {e["status"] for e in result["elements"]}


def test_an_unboxed_piece_inside_the_run_adds_text_but_no_box():
    refs = [ref(COURT, COURT_BOX, 0), ref(A, BOX_A, 1)]
    result = score(["UNITED STATES DISTRICT COURT", "SOUTHERN DISTRICT OF NEW YORK", A], [LINE_1, None, BOX_A], refs)
    heading, line = result["elements"][:2]
    assert heading["status"] == "resolved_group" and line["reference_ids"] == ["ref-0"]
    assert line["quality"] is None  # unboxed: linked, not assessed


# ---------------------------------------------------------------- 2c: relaxed links


def relaxed_page(middle_text, middle_ref, box=None, middle_kind="text"):
    refs = [ref(TOP_LINE, BOXES5[0], 0), ref(middle_ref, BOXES5[2], 1, kind=middle_kind), ref(END_LINE, BOXES5[5], 2)]
    return score([TOP_LINE, middle_text, END_LINE], [BOXES5[0], box or BOXES5[2], BOXES5[5]], refs)


def test_relaxed_link_inside_the_window():
    # A heading written with its section number: under 20 characters, so the text match needs it exact.
    result = relaxed_page("1 Green Bonds", "Green Bonds")
    element = result["elements"][1]
    assert (element["status"], element["match"], element["reference_ids"]) == ("resolved", "relaxed", ["ref-1"])
    assert result["relaxed_links"] == 1


def test_relaxed_link_needs_the_window_the_digits_and_comparable_lengths():
    # Outside the window: the candidate's GT order is before the element's previous anchor.
    refs = [ref("Green Bonds", BOXES5[1], 0), ref(TOP_LINE, BOXES5[0], 1), ref(END_LINE, BOXES5[5], 2)]
    outside = score([TOP_LINE, "1 Green Bonds", END_LINE], [BOXES5[0], BOXES5[2], BOXES5[5]], refs)
    assert outside["elements"][1]["status"] == "unresolved_content"
    # Different numbers ("55" is not "5").
    assert relaxed_page("55", "5")["elements"][1]["status"] == "unresolved_content"
    # A merged block that only contains the reference: the lengths differ too much.
    merged = relaxed_page("SPATULA PASTE PURE LIME 950 RUB PER 15 KG BUCKET", "950 RUB PER 15 KG")
    assert merged["elements"][1]["status"] == "unresolved_content"


def test_relaxed_table_links_only_to_a_table_reference():
    gt = pipe(HEAD, CHART)
    # Header and labels match, every value differs: cell F1 7/12, under the 0.6 of a direct match.
    loose = pipe(HEAD, [["2", "1100"], ["4", "900"], ["6", "700"], ["8", "300"], ["10", "600"]])
    assert relaxed_page(loose, gt, middle_kind="Table")["elements"][1]["match"] == "relaxed"
    assert relaxed_page(loose, "pH Rate 2 1000 4", middle_kind="Text")["elements"][1]["status"] == "unresolved_content"


def test_a_split_piece_is_not_relaxed_linked_to_its_whole_paragraph():
    # "needs_span_geometry" pieces stay unresolved rather than being scored against the whole box.
    result = relaxed_page("Revenue increased by twelve percent", A)
    assert result["elements"][1]["status"] == "needs_span_geometry"
    assert result["relaxed_links"] == 0


# ---------------------------------------------------------------- 3: regressions from real pages


def real_page(name, version=None):
    import json
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures/md_attribution_regressions.json").read_text())[name]
    items = [
        LayoutItemIR(md=item["md"], layout_segments=[LayoutSegmentIR(**s) for s in item["layout_segments"]])
        for item in fixture["items"]
    ]
    actual = ParseOutput(
        example_id=name,
        pipeline_name="ce",
        markdown=fixture["markdown"],
        layout_pages=[ParseLayoutPageIR(page_number=1, **fixture["page"], items=items)],
    )
    raw = {k: v for k, v in fixture["rule"].items() if k != "version"}
    return score_markdown_attribution(
        actual, MarkdownAttributionRule.model_validate(raw if version is None else {**raw, "version": version})
    )


@pytest.mark.parametrize("version", [None, 1, 2])
def test_layout_0065_paragraph_beats_the_merge_that_contains_it(version):
    # The paragraph alone scores 0.987, "heading + paragraph" 0.954: both name the paragraph.
    result = real_page("layout_0065", version)
    paragraph = result["elements"][1]
    assert paragraph["text"].startswith("We believe voluntary assurance")
    assert paragraph["status"] == "resolved"
    assert paragraph["reference_ids"] == ["30fdcf55ae60f599"]  # order 17, the paragraph alone
    assert by_id(result)["30fdcf55ae60f599"]["status"] == "assessed"


@pytest.mark.parametrize("version", [None, 1, 2])
def test_layout_0050_court_name_written_as_heading_and_paragraph_resolves_as_a_group(version):
    # The text match alone leaves both pieces at needs_span_geometry; the group pass links them.
    new = real_page("layout_0050", version)
    heading, line = new["elements"][:2]
    assert heading["status"] == line["status"] == "resolved_group"
    assert heading["reference_ids"] == line["reference_ids"]
    assert heading["quality"] > 0.8


# ---------------------------------------------------------------- 9: kind agreement


def test_kind_agreement_reports_a_caption_written_as_a_heading():
    refs = [
        ref("Figure 3: Quarterly revenue by region", BOX_A, 0, kind="Caption"),
        ref("Results", [0.1, 0.4, 0.2, 0.03], 1, kind="Section-header"),
    ]
    actual = output(
        ["Figure 3: Quarterly revenue by region", "Results"],
        [BOX_A, [0.1, 0.4, 0.2, 0.03]],
        markdown="## Figure 3: Quarterly revenue by region\n\n## Results",
    )
    result = score_markdown_attribution(actual, rule(refs))
    kinds = result["kind_agreement"]
    assert kinds["confusion"] == {"paragraph->heading": 1, "heading->heading": 1}
    caption, heading = math.sqrt(0.14), math.sqrt(0.006)
    assert kinds["score"] / kinds["total"] == pytest.approx(heading / (caption + heading))
    assert kinds["classes"]["paragraph"] == {"tp": 0, "fp": 0, "fn": 1}
    assert kinds["classes"]["heading"] == {"tp": 1, "fp": 1, "fn": 0}
    metrics = {m.metric_name: m for m in compute_markdown_attribution_metrics(actual, [rule(refs)])}
    assert metrics["md_attribution_v1_kind_agreement"].metadata["confusion"] == kinds["confusion"]
    assert metrics["md_attribution_v1_kind_heading_f1"].metadata == {"tp": 1, "fp": 1, "fn": 0, "scorer_version": 2}


def test_kind_agreement_accepts_a_chart_as_table_or_image_and_skips_forms():
    chart = ref(pipe(HEAD, CHART), BOX_A, 0, kind="Picture")
    form = ref("Name of applicant and mailing address", BOX_B, 1, kind="Form")
    result = score([pipe(HEAD, CHART), "Name of applicant and mailing address"], [BOX_A, BOX_B], [chart, form])
    assert result["kind_agreement"]["confusion"] == {"table->table": 1}
    bold = score(["**Results of the survey**"], [BOX_A], [ref("Results of the survey", BOX_A, kind="Section-header")])
    assert bold["kind_agreement"]["confusion"] == {"heading->paragraph": 1}  # bold is not a heading


def test_macro_kind_f1_is_the_mean_of_pooled_class_f1s():
    from parse_bench.evaluation.markdown_attribution import derived_aggregates

    aggregate = {"micro_md_attribution_v1_kind_heading_f1": 0.8, "micro_md_attribution_v1_kind_paragraph_f1": 0.9}
    assert derived_aggregates(aggregate)["macro_md_attribution_v1_kind_f1"] == pytest.approx(0.85)


# ---------------------------------------------------------------- 14: reading order


FOUR = [f"Paragraph {w} of a page whose order is under test here." for w in ("one", "two", "three", "four")]
FOUR_BOXES = [[0.1, 0.1 + 0.2 * i, 0.7, 0.1] for i in range(4)]


def four_refs():
    return [ref(t, b, i) for i, (t, b) in enumerate(zip(FOUR, FOUR_BOXES, strict=True))]


def test_order_tau_is_one_in_order_and_drops_when_two_paragraphs_swap():
    assert score(FOUR, FOUR_BOXES, four_refs())["order_tau"]["tau"] == pytest.approx(1)
    swapped = [FOUR[0], FOUR[2], FOUR[1], FOUR[3]]
    tau = score(swapped, [FOUR_BOXES[0], FOUR_BOXES[2], FOUR_BOXES[1], FOUR_BOXES[3]], four_refs())["order_tau"]
    assert tau["discordant"] == 1 and tau["tau"] == pytest.approx(4 / 6)
    reversed_ = score(FOUR[::-1], FOUR_BOXES[::-1], four_refs())["order_tau"]
    assert reversed_["tau"] == pytest.approx(-1)


def test_order_tau_needs_three_comparable_pairs():
    assert score(FOUR[:2], FOUR_BOXES[:2], four_refs())["order_tau"] is None
    three = score(FOUR[:3], FOUR_BOXES[:3], four_refs())
    assert three["order_tau"]["tau"] == pytest.approx(1)
    metrics = {
        m.metric_name for m in compute_markdown_attribution_metrics(output(FOUR, FOUR_BOXES), [rule(four_refs())])
    }
    assert "md_attribution_v1_order_tau" in metrics


# ---------------------------------------------------------------- 2d: fuzzy fallback

NUMBERS = "Total revenue for the financial year reached 1,234.5 million across the three regions."


# Lines of one column: each 0.01 below the previous, so consecutive references merge as a stack.
STACK = [[0.1, 0.2 + 0.05 * i, 0.6, 0.04] for i in range(3)]


def fuzzy_page(middle_text, middle_refs, **rule_kwargs):
    """TOP and END link by text and bound the window; the middle reference(s) sit between them."""
    refs = [ref(TOP_LINE, BOXES5[0], 0)]
    refs += [ref(text, STACK[i], 1 + i) for i, text in enumerate(middle_refs)]
    refs.append(ref(END_LINE, BOXES5[5], len(refs)))
    return score_markdown_attribution(
        output([TOP_LINE, middle_text, END_LINE], [BOXES5[0], STACK[0], BOXES5[5]]), rule(refs, **rule_kwargs)
    )


def test_fuzzy_links_a_number_written_without_its_thousands_separator():
    # "1,234.5" and "1234.5" have different digit groups, so the text match and
    # the relaxed link (digit gate) both refuse the pair.
    result = fuzzy_page(NUMBERS.replace("1,234.5", "1234.5"), [NUMBERS])
    element = result["elements"][1]
    assert (element["status"], element["match"], element["reference_ids"]) == ("fuzzy_resolved", "fuzzy", ["ref-1"])
    assert element["quality"] > 0.9
    assert element["fuzzy_similarity"] >= 0.7 and element["fuzzy_margin"] >= 0.1
    assert result["fuzzy_links"] == 1 and result["relaxed_links"] == 0
    assert by_id(result)["ref-1"]["fuzzy"] is True


@pytest.mark.parametrize("version", [None, 1, 2])
def test_every_rule_is_scored_by_version_2_whatever_its_version_field(version):
    # Older v0.3 sidecars say version 1, newer ones 2, a new rule nothing: the
    # removed version 1 scorer had no fuzzy fallback, anchors or dilation.
    refs = [ref(TOP_LINE, BOXES5[0], 0), ref(NUMBERS, STACK[0], 1), ref(END_LINE, BOXES5[5], 2)]
    raw = {"references": refs} if version is None else {"references": refs, "version": version}
    parsed = MarkdownAttributionRule.model_validate(raw)
    assert parsed.version == 2
    actual = output([TOP_LINE, NUMBERS.replace("1,234.5", "1234.5"), END_LINE], [BOXES5[0], STACK[0], BOXES5[5]])
    result = score_markdown_attribution(actual, parsed)
    assert result["version"] == 2
    element = result["elements"][1]
    assert (element["status"], element["match"], element["reference_ids"]) == ("fuzzy_resolved", "fuzzy", ["ref-1"])
    assert "raw_quality" in element  # the quality is the dilated IoU, the raw IoU beside it
    names = {m.metric_name for m in compute_markdown_attribution_metrics(actual, [parsed])}
    assert {"md_attribution_v1_fuzzy_weight_share", "md_attribution_v1_order_tau"} <= names
    # alt_texts and decorative, once version 2 only, are accepted on any rule.
    extra = [ref(A, BOX_A, 3, alt_texts=["alternative"]), picture(4, PIC_A, decorative=True)]
    MarkdownAttributionRule.model_validate({**raw, "references": [*refs, *extra]})


def test_unknown_scorer_versions_are_rejected():
    for bad in (0, 3, "1", True):
        with pytest.raises(ValueError):
            MarkdownAttributionRule.model_validate({"references": [ref(A, BOX_A)], "version": bad})


def test_fuzzy_links_dropped_punctuation_that_splits_a_decimal():
    gt = "The rate rose from 12.5 to 13.75 percent, well above the target set in the plan for the year."
    result = fuzzy_page(
        "The rate rose from 125 to 1375 percent well above the target set in the plan for the year", [gt]
    )
    assert result["elements"][1]["match"] == "fuzzy"


def test_fuzzy_links_two_swapped_words_that_reorder_the_numbers():
    # A plain word swap is already a relaxed link (token_set_ratio); a swap of two
    # numbers fails its digit gate.
    gt = "Sample sizes of 12 15 and 18 units were tested at each of the regional sites."
    result = fuzzy_page("Sample sizes of 15 12 and 18 units were tested at each of the regional sites.", [gt])
    assert result["elements"][1]["match"] == "fuzzy"


def test_fuzzy_containment_refuses_an_element_that_holds_a_neighbouring_block():
    # The element is the paragraph plus a caption that is its own reference. With
    # merges off, the paragraph alone is the only target: the cosine is high, but
    # the element holds 3-grams the paragraph lacks.
    caption = "Figure 12 shows the monthly trend of sales for every region."
    element = NUMBERS.replace("1,234.5", "1234.5") + " " + caption
    refused = fuzzy_page(element, [NUMBERS, caption], max_merge_elements=1)
    assert refused["elements"][1]["status"] == "unresolved_content"
    assert refused["fuzzy_links"] == 0
    # With merges on, the element meets the paragraph + caption merge instead.
    merged = fuzzy_page(element, [NUMBERS, caption])
    assert merged["elements"][1]["match"] == "fuzzy"
    assert merged["elements"][1]["reference_ids"] == ["ref-1", "ref-2"]


def test_containment_is_the_smaller_of_the_two_shares():
    from parse_bench.evaluation.markdown_attribution import containment, normalize

    assert containment("green bonds", "green bonds") == 1.0
    # "green bonds" is inside "1 green bonds", but 2 of the 13 3-grams of the longer text are not in it.
    assert containment(normalize("1 Green Bonds"), normalize("Green Bonds")) == pytest.approx(11 / 13)
    assert containment("", "green bonds") == 0.0


def test_fuzzy_margin_leaves_a_near_tie_unresolved():
    # Two references in the window differ in one year; the element misses the
    # separator of both, so neither text match nor relaxed link applies and the
    # two cosines are within the margin.
    first = NUMBERS.replace("financial year", "financial year 2023")
    second = NUMBERS.replace("financial year", "financial year 2024")
    element = first.replace("1,234.5", "1234.5").replace("2023", "2025")
    result = fuzzy_page(element, [first, second])
    assert result["elements"][1]["status"] == "unresolved_content"
    assert result["fuzzy_links"] == 0


def test_fuzzy_does_not_link_a_table_to_a_text_reference():
    table = pipe(["Region", "Revenue"], [["North", "1234.5"], ["South", "2345.5"], ["East", "3456.5"]])
    flat = "Region Revenue North 1,234.5 South 2,345.5 East 3,456.5"
    result = fuzzy_page(table, [flat])
    element = result["elements"][1]
    assert element["kind"] == "table"
    assert element["status"] == "unresolved_content" and result["fuzzy_links"] == 0


def test_fuzzy_links_are_never_reading_order_anchors():
    from parse_bench.evaluation.markdown_attribution import TEXT_MATCHES, _windows

    assert "fuzzy" not in TEXT_MATCHES
    rows = [{"match": "fuzzy", "reference_ids": ["a"]}, {"match": None, "reference_ids": []}]
    assert _windows(rows, {"a": 3}) == [(-math.inf, math.inf), (-math.inf, math.inf)]
    # On a page: two exact duplicates around a fuzzy-linked paragraph stay
    # ambiguous. Were the fuzzy link an anchor, each would have one duplicate
    # left in its window and resolve.
    refs = [
        ref(TOP_LINE, BOXES5[0], 0),
        ref(LABEL, BOXES5[1], 1),
        ref(NUMBERS, BOXES5[2], 2),
        ref(LABEL, BOXES5[3], 3),
        ref(END_LINE, BOXES5[5], 4),
    ]
    texts = [TOP_LINE, LABEL, NUMBERS.replace("1,234.5", "1234.5"), LABEL, END_LINE]
    result = score(texts, [BOXES5[0], BOXES5[1], BOXES5[2], BOXES5[3], BOXES5[5]], refs)
    statuses = [e["status"] for e in result["elements"]]
    assert statuses[2] == "fuzzy_resolved"
    assert statuses[1] == statuses[3] == "ambiguous_source"


def test_fuzzy_tfidf_matches_scikit_learn_and_ignores_corpus_order():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    from parse_bench.evaluation.markdown_attribution import tfidf_cosine

    corpus = ["total revenue 1234 5", "a de la", "green bonds 2025", "x", "the total amount due"]
    left, right = ["total revenue 1 234 5", "green bond"], ["total amount", "green bonds", "a"]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), lowercase=False, sublinear_tf=True)
    vectorizer.fit(corpus)
    expected = (vectorizer.transform(left) @ vectorizer.transform(right).T).toarray()
    assert np.allclose(tfidf_cosine(corpus, left, right), expected)
    assert np.array_equal(tfidf_cosine(corpus, left, right), tfidf_cosine(corpus[::-1], left, right))


def test_fuzzy_results_are_deterministic():
    first = fuzzy_page(NUMBERS.replace("1,234.5", "1234.5"), [NUMBERS])
    second = fuzzy_page(NUMBERS.replace("1,234.5", "1234.5"), [NUMBERS])
    assert first == second


def test_fuzzy_weight_share_and_link_count_metrics():
    from parse_bench.evaluation.markdown_attribution import headline_rows

    refs = [ref(TOP_LINE, BOXES5[0], 0), ref(NUMBERS, BOXES5[1], 1), ref(END_LINE, BOXES5[5], 2)]
    texts = [TOP_LINE, NUMBERS.replace("1,234.5", "1234.5"), END_LINE]
    metrics = metrics_of(texts, [BOXES5[0], BOXES5[1], BOXES5[5]], refs)
    share = metrics["md_attribution_v1_fuzzy_weight_share"]
    # Three references of equal area, all assessed; one through the fallback.
    assert share.value == pytest.approx(1 / 3)
    assert share.metadata["fuzzy_links"] == 1 and share.metadata["relaxed_links"] == 0
    assert metrics["md_attribution_v1_fuzzy_links_count"].metadata["count"] == 1
    assert metrics["md_attribution_v1_relaxed_links_count"].metadata["count"] == 0
    inventory = metrics["md_attribution_v1_generated_elements"].metadata
    assert inventory["fuzzy_links"] == 1 and inventory["fuzzy_weight_share"] == pytest.approx(1 / 3)
    rows = headline_rows(
        {
            "avg_md_attribution_v1_generated_elements": 3.0,
            "micro_md_attribution_v1_fuzzy_weight_share": 0.004,
            "total_md_attribution_v1_fuzzy_links_count": 3.0,
        }
    )
    assert any("Fuzzy" in label and value == 0.004 for label, value in rows)


# A compact port of the rewrite stress test that chose the fallback's thresholds:
# rewrite what version 2 links by text on real saved outputs (number formatting,
# dropped punctuation, swapped numbers), then check that the fallback keeps more
# elements on their reference and never links one to other content.

STRESS_SEED = 7
STRESS_RATE = 0.5


def _rewrite(text: str) -> str:
    import random
    import re

    rng = random.Random(f"{STRESS_SEED}|{text}")

    def number(match: re.Match) -> str:
        n = match.group(0)
        if rng.random() >= STRESS_RATE:
            return n
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", n):
            return n.replace(",", "")
        if re.fullmatch(r"\d{4,}", n):
            return f"{int(n):,}"
        if re.fullmatch(r"\d+\.\d+", n):
            return n.replace(".", ",")
        return n

    text = re.sub(r"\d[\d,.]*\d|\d", number, text)
    return "".join(c for c in text if not (c in ".,;:()" and rng.random() < STRESS_RATE))


def _stressed(output_: ParseOutput, indices: list[int]) -> tuple[ParseOutput, set[int]]:
    """The output with the elements at ``indices`` rewritten in the Markdown and in their item."""
    from parse_bench.evaluation.markdown_attribution import markdown_elements

    out = output_.model_copy(deep=True)
    elements = markdown_elements(out.markdown)
    lines = out.markdown.splitlines(keepends=True)
    changed: dict[int, tuple[str, str]] = {}
    for index in sorted(indices, reverse=True):
        element = elements[index]
        new = _rewrite(element.text)
        block = "".join(lines[element.start_line : element.end_line])
        if new == element.text or element.kind != "paragraph" or element.text not in block:
            continue
        lines[element.start_line : element.end_line] = [block.replace(element.text, new, 1)] + [""] * (
            element.end_line - element.start_line - 1
        )
        changed[index] = (element.text, new)
    out.markdown = "".join(lines)
    for old, new in changed.values():
        for page in out.layout_pages:
            for item in page.items:
                if item.md and old in item.md:
                    item.md = item.md.replace(old, new)
                    break
    return out, set(changed)


def test_fuzzy_fallback_under_rewrites_keeps_more_elements_on_their_reference(monkeypatch):
    import gzip
    import json
    import warnings
    from pathlib import Path

    from parse_bench.evaluation import markdown_attribution as ma

    bundle = Path(__file__).parent / "fixtures/md_attribution_golden"
    with gzip.open(bundle / "outputs.json.gz", "rt") as f:
        outputs = json.load(f)
    with gzip.open(bundle / "rules.json.gz", "rt") as f:
        rules = {stem: MarkdownAttributionRule.model_validate(r) for stem, r in json.load(f).items()}
    same = {"v2": 0, "v2+A": 0}
    rewritten = fuzzy_links = 0
    fallback = ma._fuzzy_pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for arm in ("ce-headnew", "agentic-headnew"):
            for stem, raw in sorted(outputs[arm].items()):
                page = ParseOutput.model_validate(raw)
                base = {
                    i: row["reference_ids"]
                    for i, row in enumerate(score_markdown_attribution(page, rules[stem])["elements"])
                    if row.get("match") == "text" and row["quality"] is not None
                }
                stressed, changed = _stressed(page, list(base))
                rewritten += len(changed)
                for variant in same:
                    monkeypatch.setattr(ma, "_fuzzy_pass", fallback if variant == "v2+A" else lambda *args: None)
                    rows = score_markdown_attribution(stressed, rules[stem])["elements"]
                    same[variant] += sum(rows[i]["reference_ids"] == base[i] for i in changed)
                    if variant == "v2+A":
                        taken = {r for i in base for r in base[i]}
                        for i, row in enumerate(rows):
                            if row.get("match") != "fuzzy":
                                continue
                            fuzzy_links += 1
                            # A fuzzy link joins the same content: a rewritten element
                            # gets back its own reference, any other element takes none
                            # that a text link held before the rewrite.
                            if i in changed:
                                assert row["reference_ids"] == base[i], (stem, row["element_id"])
                            else:
                                assert not set(row["reference_ids"]) & taken, (stem, row["element_id"])
    assert rewritten >= 50
    assert fuzzy_links > 0
    assert same["v2+A"] >= same["v2"]
