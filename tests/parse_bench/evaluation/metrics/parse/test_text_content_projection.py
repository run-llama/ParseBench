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
