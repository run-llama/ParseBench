"""Tests for merge_preceding_titles_into_tables."""

from bs4 import BeautifulSoup

from parse_bench.evaluation.metrics.parse.table_parsing import (
    merge_preceding_titles_into_tables,
)


def _extract_first_table_rows(html_content: str) -> list[list[str]]:
    """Helper: extract text of each cell in each row of the first table."""
    soup = BeautifulSoup(html_content, "lxml")
    table = soup.find("table")
    if not table:
        return []
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
        rows.append(cells)
    return rows


def test_title_merged_when_gt_has_colspan_row():
    """Preceding heading in prediction should be merged when GT has a colspan title row."""
    expected = """
<table>
    <tr><th colspan="3">My Table Title</th></tr>
    <tr><th>A</th><th>B</th><th>C</th></tr>
    <tr><td>1</td><td>2</td><td>3</td></tr>
</table>
"""
    actual = """
<h2>My Table Title</h2>
<table>
    <tr><th>A</th><th>B</th><th>C</th></tr>
    <tr><td>1</td><td>2</td><td>3</td></tr>
</table>
"""
    result = merge_preceding_titles_into_tables(expected, actual)
    rows = _extract_first_table_rows(result)
    # First row should be the merged title
    assert rows[0] == ["My Table Title"]
    # Second row should be headers
    assert rows[1] == ["A", "B", "C"]
    # Title heading should be removed from content
    assert "<h2>" not in result or "My Table Title" not in result.split("<table")[0]


def test_no_merge_when_gt_has_no_colspan_row():
    """Should not merge when GT table doesn't start with a full-width row."""
    expected = """
<table>
    <tr><th>A</th><th>B</th><th>C</th></tr>
    <tr><td>1</td><td>2</td><td>3</td></tr>
</table>
"""
    actual = """
<h2>Some Heading</h2>
<table>
    <tr><th>A</th><th>B</th><th>C</th></tr>
    <tr><td>1</td><td>2</td><td>3</td></tr>
</table>
"""
    result = merge_preceding_titles_into_tables(expected, actual)
    rows = _extract_first_table_rows(result)
    # First row should still be headers, no title row added
    assert rows[0] == ["A", "B", "C"]


def test_no_merge_when_prediction_already_has_colspan_row():
    """Should not merge when prediction already has a full-width first row."""
    expected = """
<table>
    <tr><th colspan="3">Title</th></tr>
    <tr><th>A</th><th>B</th><th>C</th></tr>
</table>
"""
    actual = """
<table>
    <tr><th colspan="3">Title</th></tr>
    <tr><th>A</th><th>B</th><th>C</th></tr>
</table>
"""
    result = merge_preceding_titles_into_tables(expected, actual)
    rows = _extract_first_table_rows(result)
    # Should not have duplicated the title row
    assert rows[0] == ["Title"]
    assert rows[1] == ["A", "B", "C"]
    assert len(rows) == 2


def test_fuzzy_match_title():
    """Titles should match even with minor differences."""
    expected = """
<table>
    <tr><th colspan="3">Rate Schedule Item Changes</th></tr>
    <tr><th>A</th><th>B</th><th>C</th></tr>
</table>
"""
    actual = """
<h2>Rate Schedule Item Changes</h2>
<table>
    <tr><th>A</th><th>B</th><th>C</th></tr>
    <tr><td>1</td><td>2</td><td>3</td></tr>
</table>
"""
    result = merge_preceding_titles_into_tables(expected, actual)
    rows = _extract_first_table_rows(result)
    assert rows[0] == ["Rate Schedule Item Changes"]


def test_no_merge_when_preceding_text_doesnt_match():
    """Should not merge when preceding text doesn't match any GT title."""
    expected = """
<table>
    <tr><th colspan="3">Rate Schedule Item Changes</th></tr>
    <tr><th>A</th><th>B</th><th>C</th></tr>
</table>
"""
    actual = """
<h2>Completely Different Title</h2>
<table>
    <tr><th>A</th><th>B</th><th>C</th></tr>
</table>
"""
    result = merge_preceding_titles_into_tables(expected, actual)
    rows = _extract_first_table_rows(result)
    # No title should have been merged
    assert rows[0] == ["A", "B", "C"]


def test_empty_inputs():
    """Should handle empty inputs gracefully."""
    assert merge_preceding_titles_into_tables("", "some content") == "some content"
    assert merge_preceding_titles_into_tables("some content", "") == ""


def test_no_tables():
    """Should return actual unchanged when no tables are present."""
    expected = "<p>No tables here</p>"
    actual = "<p>No tables here either</p>"
    result = merge_preceding_titles_into_tables(expected, actual)
    assert "No tables here either" in result


def test_merge_with_thead():
    """Title row should be inserted into thead when present."""
    expected = """
<table>
    <tr><th colspan="2">Title Row</th></tr>
    <tr><th>X</th><th>Y</th></tr>
</table>
"""
    actual = """
<p>Title Row</p>
<table>
    <thead>
        <tr><th>X</th><th>Y</th></tr>
    </thead>
    <tbody>
        <tr><td>1</td><td>2</td></tr>
    </tbody>
</table>
"""
    result = merge_preceding_titles_into_tables(expected, actual)
    rows = _extract_first_table_rows(result)
    assert rows[0] == ["Title Row"]
    assert rows[1] == ["X", "Y"]
