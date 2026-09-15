"""page_decoration: header / footer / printed page number per page, plus leakage into the body."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.evaluation.metrics.parse.rules_page_decoration import (
    PageDecorationRule,
    canonical_page_number,
    page_number_equal,
)
from parse_bench.schemas.parse_output import ParseOutput

RULE = {
    "type": "page_decoration",
    "page": 1,
    "header": "Annual Report 2024 | Governance",
    "footer": "Example plc",
    "page_number": "12",
    "page_number_raw": "Page 12 of 48",
}
TAGGED_MD = (
    "<page_header>Governance — Annual Report 2024</page_header>\n\n# Board of directors\n\n"
    "The board met four times.\n\n<page_footer>Example plc</page_footer>\n<page_number>Page 12 of 48</page_number>\n"
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("Page 12 of 48", "12"),
        ("– iv –", "iv"),
        ("A-3", "a-3"),
        ("3 / 10", "3"),
        ("007", "7"),
        ("p. 5", "5"),
        ("", None),
        (None, None),
    ],
)
def test_canonical_page_number(raw: str | None, canonical: str | None) -> None:
    assert canonical_page_number(raw) == canonical


def test_page_number_equal_bridges_roman_and_arabic() -> None:
    assert page_number_equal("iv", "4") and page_number_equal("12", "Page 12 of 48")
    assert not page_number_equal(None, "3") and not page_number_equal("3", "4")


def test_markdown_tags_reordered_header_pieces_pass() -> None:
    rule = PageDecorationRule(RULE)
    passed, expl, score = rule.run(TAGGED_MD)
    assert passed, expl
    assert score == 1.0 and rule.result_details["source"] == "markdown_tags"
    assert rule.result_details["axes"]["leak"]["passed"] is True


def test_furniture_left_in_body_fails_leak_and_slots() -> None:
    rule = PageDecorationRule(RULE)
    passed, expl, score = rule.run("# Annual Report 2024 Governance\n\nBody. Page 12 of 48\n\nExample plc\n")
    assert not passed
    axes = rule.result_details["axes"]
    assert axes["header"]["reason"] == "missing" and axes["footer"]["reason"] == "missing"
    assert axes["page_number"]["passed"] is False
    assert sorted(axes["leak"]["leaked"]) == ["Annual Report 2024", "Example plc", "Governance", "Page 12 of 48"]
    assert score == 0.0


def test_negative_page_must_stay_empty() -> None:
    rule = PageDecorationRule({"type": "page_decoration", "page": 1})
    assert rule.run("# Cover\n\nText")[0] is True
    passed, expl, _ = rule.run("<page_header>Cover</page_header>\nText")
    assert not passed and rule.result_details["axes"]["header"]["reason"] == "hallucinated"
    assert rule.result_details["axes"]["leak"]["passed"] is None


def test_structured_fields_win_over_markdown_tags() -> None:
    rule = create_test_rule(RULE)
    rule.parse_output = ParseOutput.model_validate(
        {
            "example_id": "positives/sample",
            "pipeline_name": "ours_agentic",
            "markdown": "# Board of directors\n\nThe board met four times.",
            "layout_pages": [
                {
                    "page_number": 1,
                    "page_header_markdown": "Annual Report 2024 · Governance",
                    "page_footer_markdown": "Example plc",
                    "printed_page_number": "12",
                    "items": [],
                }
            ],
        }
    )
    passed, expl, score = rule.run("# Board of directors\n\nThe board met four times.")
    assert passed, expl
    assert rule.result_details["source"] == "structured" and score == 1.0


def test_partial_header_gets_partial_credit_but_fails_threshold() -> None:
    rule = PageDecorationRule(RULE)
    rule.run(
        "<page_header>Annual Report 2024</page_header>\n\nBody\n\n"
        "<page_footer>Example plc</page_footer><page_number>12</page_number>"
    )
    header = rule.result_details["axes"]["header"]
    assert header["passed"] is False and 0.5 < header["score"] < 1.0
    assert header["recall"] == 0.75 and header["precision"] == 1.0


def test_page_number_inside_header_text_counts() -> None:
    rule = PageDecorationRule(RULE)
    passed, expl, _ = rule.run(
        "<page_header>12 | Annual Report 2024 | Governance</page_header>\n\nBody\n\n"
        "<page_footer>Example plc</page_footer>"
    )
    axis = rule.result_details["axes"]["page_number"]
    assert passed, expl
    assert axis["passed"] is True and axis["score"] == 1.0 and axis["in_furniture_text"] is True
    assert "ok (in header/footer text)" in expl
