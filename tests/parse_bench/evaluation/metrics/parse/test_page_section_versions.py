import pytest
from pydantic import ValidationError

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.test_cases.parse_rule_schemas import ParsePageSectionRule


@pytest.mark.parametrize("kind", ["header", "footer"])
@pytest.mark.parametrize(
    ("content", "target", "legacy", "nested"),
    [
        ("Report title", "Report title", True, True),
        ("Report <page_number>7</page_number>", "Report", False, True),
        ("<u>Other</u> Report", "Report", False, True),
        ("<u>Report</u> <u>Other</u>", "<u>Report</u>", False, True),
        ("<page_number>7</page_number>", "<page_number>7</page_number>", True, True),
        ("Different text", "Report", False, False),
    ],
)
def test_nested_tags_require_opt_in(kind: str, content: str, target: str, legacy: bool, nested: bool) -> None:
    payload = {"type": f"is_{kind}", "text": target}
    md = f"<page_{kind}>{content}</page_{kind}>"
    assert create_test_rule(payload).run(md)[0] is legacy
    assert create_test_rule({**payload, "section_matching": "legacy"}).run(md)[0] is legacy
    model = ParsePageSectionRule.model_validate({**payload, "section_matching": "nested-v1"})
    assert create_test_rule(model.model_dump()).run(md)[0] is nested


@pytest.mark.parametrize(
    ("md", "target"),
    [
        ("<page_footer>Other</page_footer> Report", "Report"),
        ("<page_header>Report</page_header>", "Report"),
        ("<page_footer>One</page_footer><page_footer>Two</page_footer>", "One Two"),
        ("<page_footer>Other<page_header>Report</page_header></page_footer>", "Report"),
        ("<page_footer>Other</sub_page>Report</page_footer>", "Report"),
        ("<page_footer>Other<page_footer>Report</page_footer>", "Other"),
        ("<page_footer>Report", "Report"),
    ],
)
def test_nested_matcher_cannot_cross_section_boundaries(md: str, target: str) -> None:
    rule = create_test_rule({"type": "is_footer", "text": target, "section_matching": "nested-v1"})
    assert not rule.run(md)[0]


def test_unknown_matching_contract_is_rejected() -> None:
    with pytest.raises(ValidationError):
        create_test_rule({"type": "is_footer", "text": "Report", "section_matching": "typo"})
