import pytest

from parse_bench.evaluation.metrics.parse.rule_based_metric import RuleBasedMetric
from parse_bench.evaluation.metrics.parse.text_content_projection import (
    canonicalize_tables_for_text_content,
)


def test_canonicalizes_html_table_once_in_row_major_order() -> None:
    markdown = """
Before
<table>
  <tr><th>Name</th><th>Value</th></tr>
  <tr><td>Alpha</td><td>42</td></tr>
</table>
After
"""

    projected = canonicalize_tables_for_text_content(markdown)

    assert "<table" not in projected
    assert projected.count("Name") == 1
    assert projected.count("Alpha") == 1
    assert "Name\tValue\nAlpha\t42" in projected
    assert projected.index("Before") < projected.index("Name") < projected.index("After")


def test_preserves_anchor_continuity_across_html_cells() -> None:
    markdown = "<table><tr><td>Chapter 3.</td><td>Installation</td><td>8</td></tr></table>"

    projected = canonicalize_tables_for_text_content(markdown)

    assert "Chapter 3.\tInstallation\t8" == projected.strip()


def test_preserves_caption_and_cells_without_explicit_rows() -> None:
    markdown = "<table><caption>Summary</caption><td>Alpha</td><td>42</td></table>"

    projected = canonicalize_tables_for_text_content(markdown)

    assert projected.strip() == "Summary\nAlpha\t42"


def test_preserves_document_order_for_orphan_cells_and_section_rows() -> None:
    markdown = """
<table>
  <th>Orphan header</th>
  <thead><tr><th>Section header</th></tr></thead>
  <tbody><tr><td>Body value</td></tr></tbody>
  <tfoot><tr><td>Footer value</td></tr></tfoot>
</table>
"""

    projected = canonicalize_tables_for_text_content(markdown)

    assert projected.strip() == "Orphan header\nSection header\nBody value\nFooter value"


def test_projects_valid_table_after_unclosed_table_start() -> None:
    markdown = "Before <table> malformed\nMiddle\n<table><tr><td>Valid</td></tr></table>\nAfter"

    projected = canonicalize_tables_for_text_content(markdown)

    assert "<table> malformed" in projected
    assert projected.count("Valid") == 1
    assert "<table><tr><td>Valid" not in projected


def test_canonicalizes_markdown_table_without_separator_row() -> None:
    markdown = """
Before
| Name | Value |
| :--- | ---: |
| Alpha | `a|b` |
| Beta | 42 |
After
"""

    projected = canonicalize_tables_for_text_content(markdown)

    assert "| :--- | ---: |" not in projected
    assert "Name\tValue\nAlpha\t`a|b`\nBeta\t42" in projected
    assert projected.count("Alpha") == 1


def test_canonicalizes_gfm_table_with_short_delimiter_cells() -> None:
    markdown = "| A | B |\n| :-: | -- |\n| Alpha | Beta |"

    projected = canonicalize_tables_for_text_content(markdown)

    assert projected == "A\tB\nAlpha\tBeta"


def test_leaves_non_table_pipe_text_unchanged() -> None:
    markdown = "Run `left | right` and keep this line.\nA | B"

    assert canonicalize_tables_for_text_content(markdown) == markdown


def test_leaves_malformed_html_table_unchanged() -> None:
    markdown = "Before <table><tr><td>Unclosed After"

    assert canonicalize_tables_for_text_content(markdown) == markdown


def test_nested_table_text_is_not_duplicated() -> None:
    markdown = """
<table>
  <tr><td>Outer</td><td><table><tr><td>Nested</td></tr></table></td></tr>
</table>
"""

    projected = canonicalize_tables_for_text_content(markdown)

    assert projected.count("Outer") == 1
    assert projected.count("Nested") == 1


def _run_rule(rule: dict[str, object], markdown: str) -> dict[str, object]:
    projected = canonicalize_tables_for_text_content(markdown)
    result = RuleBasedMetric().compute([rule], projected)
    return result.metadata["rule_results"][0]


@pytest.mark.parametrize(
    ("rule_type", "bag_field", "bag"),
    [
        ("unexpected_word", "bag_of_word", {"allowed": 1}),
        ("unexpected_word_percent", "bag_of_word", {"allowed": 1}),
        ("unexpected_sentence", "bag_of_sentence", {"Allowed sentence": 1}),
        ("unexpected_sentence_percent", "bag_of_sentence", {"Allowed sentence": 1}),
    ],
)
def test_unexpected_rules_include_canonical_table_payload(
    rule_type: str,
    bag_field: str,
    bag: dict[str, int],
) -> None:
    rule_result = _run_rule(
        {"type": rule_type, bag_field: bag},
        "<table><tr><td>Novel table sentence.</td></tr></table>",
    )

    assert rule_result["passed"] is False


@pytest.mark.parametrize(
    ("rule_type", "bag_field", "bag", "cell_text"),
    [
        ("too_many_word_occurence", "bag_of_word", {"repeated": 1}, "Repeated"),
        ("too_many_word_occurence_percent", "bag_of_word", {"repeated": 1}, "Repeated"),
        (
            "too_many_sentence_occurence",
            "bag_of_sentence",
            {"Repeated table sentence": 1},
            "Repeated table sentence.",
        ),
        (
            "too_many_sentence_occurence_percent",
            "bag_of_sentence",
            {"Repeated table sentence": 1},
            "Repeated table sentence.",
        ),
    ],
)
def test_too_many_rules_do_not_count_one_table_cell_as_duplicates(
    rule_type: str,
    bag_field: str,
    bag: dict[str, int],
    cell_text: str,
) -> None:
    rule_result = _run_rule(
        {"type": rule_type, bag_field: bag},
        f"<table><tr><td>{cell_text}</td></tr></table>",
    )

    assert rule_result["passed"] is True


@pytest.mark.parametrize(
    ("rule_type", "bag_field", "bag", "cell_text"),
    [
        ("too_many_word_occurence", "bag_of_word", {"repeated": 1}, "Repeated"),
        ("too_many_word_occurence_percent", "bag_of_word", {"repeated": 1}, "Repeated"),
        (
            "too_many_sentence_occurence",
            "bag_of_sentence",
            {"Repeated table sentence": 1},
            "Repeated table sentence.",
        ),
        (
            "too_many_sentence_occurence_percent",
            "bag_of_sentence",
            {"Repeated table sentence": 1},
            "Repeated table sentence.",
        ),
    ],
)
def test_too_many_rules_detect_genuinely_duplicated_table_rows(
    rule_type: str,
    bag_field: str,
    bag: dict[str, int],
    cell_text: str,
) -> None:
    rule_result = _run_rule(
        {"type": rule_type, bag_field: bag},
        f"<table><tr><td>{cell_text}</td></tr><tr><td>{cell_text}</td></tr></table>",
    )

    assert rule_result["passed"] is False
