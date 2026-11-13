"""Markdown cell content has the same reading as equivalent HTML cells."""

import pytest

from parse_bench.evaluation.metrics.parse.table_next import parse_tables, score_table_next


@pytest.mark.parametrize("leading_pipe", [True, False])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"A \| B", "A | B"),
        (r"A \|", "A |"),
        (r"A \\| B", "A | B"),
        (r"A \\\| B", r"A \| B"),
        ("A &amp; B &lt; C", "A & B < C"),
        (r"\$4,200", "$4,200"),
        (r"A \* B \_ C", "A * B _ C"),
        ("North<br>America", "North America"),
        ("North<br/>America", "North America"),
        ("<strong>A &amp; B</strong>", "A & B"),
        (r"C:\Reports", r"C:\Reports"),
        ("&amp;lt;", "&lt;"),
        (r"\<br>", "<br>"),
        (r"`A \| B`", "A | B"),
        (r"``A ` \| B``", "A ` | B"),
        (r"`\$ &amp; <br>`", r"\$ &amp; <br>"),
        ('<span title="`">A</span> `B`', "A B"),
        ("`x^{2}`", "x^{2}"),
        (r"\^{2}", "^{2}"),
    ],
)
def test_markdown_cell_reading(raw: str, expected: str, leading_pipe: bool) -> None:
    prefix = "|" if leading_pipe else ""
    table = f"{prefix} Name | Value |\n{prefix} --- | --- |\n{prefix} {raw} | 42 |"
    cells = parse_tables(table)[0].cells
    assert [(c.r, c.c, c.text) for c in cells] == [(0, 0, "Name"), (0, 1, "Value"), (1, 0, expected), (1, 1, "42")]


@pytest.mark.parametrize("leading_pipe", [True, False])
def test_escaped_final_pipe_is_cell_content(leading_pipe: bool) -> None:
    prefix = "|" if leading_pipe else ""
    table = f"{prefix} Name | Value\n{prefix} --- | ---\n{prefix} Item | A " + r"\|"
    assert [c.text for c in parse_tables(table)[0].cells] == ["Name", "Value", "Item", "A |"]


def test_html_markdown_score_parity() -> None:
    reference = (
        "<table><tr><th>Item</th><th>Value</th></tr>"
        "<tr><td>A | B</td><td>$4,200</td></tr>"
        "<tr><td>A &amp; B &lt; C</td><td>20</td></tr>"
        "<tr><td>North<br>America</td><td>30</td></tr>"
        "<tr><td><code>Formula A | B</code></td><td>40</td></tr></table>"
    )
    markdown = (
        "| Item | Value |\n| --- | --- |\n"
        r"| A \| B | \$4,200 |"
        "\n"
        "| A &amp; B &lt; C | 20 |\n| North<br/>America | 30 |\n"
        r"| `Formula A \| B` | 40 |"
    )
    assert score_table_next(reference, markdown).score == score_table_next(reference, reference).score == 1.0
