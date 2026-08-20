"""Canonical plain-text projection for ParseBench Text Content rules."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

_HTML_TABLE_TAG = re.compile(r"<table(?=[\s>])[^>]*>|</table\s*>", re.IGNORECASE)
_MARKDOWN_DELIMITER_CELL = re.compile(r"^:?-+:?$")


def _html_table_spans(content: str) -> list[tuple[int, int]]:
    """Return matched top-level HTML table spans, recovering after malformed starts."""
    stack: list[int] = []
    matched: list[tuple[int, int]] = []

    for tag in _HTML_TABLE_TAG.finditer(content):
        if tag.group().lower().startswith("</"):
            if stack:
                matched.append((stack.pop(), tag.end()))
        else:
            stack.append(tag.start())

    # A valid nested table is projected through its outer table. If an unmatched
    # outer start precedes a later valid table, the later matched span survives.
    return [
        span
        for span in matched
        if not any(outer_start < span[0] and span[1] < outer_end for outer_start, outer_end in matched)
    ]


def _normalize_cell_text(cell: Tag) -> str:
    return " ".join(cell.get_text(" ", strip=True).split())


def _canonicalize_html_table(table_html: str) -> str:
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return table_html

    rows: list[str] = []
    caption = table.find("caption")
    if caption is not None and caption.find_parent("table") is table:
        caption_text = _normalize_cell_text(caption)
        if caption_text:
            rows.append(caption_text)

    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = [_normalize_cell_text(cell) for cell in row.find_all(["th", "td"]) if cell.find_parent("tr") is row]
        if cells:
            rows.append("\t".join(cells))

    orphan_cells = [
        _normalize_cell_text(cell)
        for cell in table.find_all(["th", "td"])
        if cell.find_parent("table") is table and cell.find_parent("tr") is None
    ]
    if orphan_cells:
        rows.append("\t".join(orphan_cells))

    return "\n".join(rows)


def _split_markdown_row(line: str) -> list[str]:
    """Split a Markdown table row without treating escaped/code pipes as separators."""
    cells: list[str] = []
    buffer: list[str] = []
    code_ticks = 0
    index = 0

    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line):
            buffer.extend((char, line[index + 1]))
            index += 2
            continue
        if char == "`":
            run_end = index + 1
            while run_end < len(line) and line[run_end] == "`":
                run_end += 1
            run_length = run_end - index
            if code_ticks == 0:
                code_ticks = run_length
            elif code_ticks == run_length:
                code_ticks = 0
            buffer.extend(line[index:run_end])
            index = run_end
            continue
        if char == "|" and code_ticks == 0:
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
        index += 1

    cells.append("".join(buffer).strip())
    if line.lstrip().startswith("|") and cells and not cells[0]:
        cells = cells[1:]
    if line.rstrip().endswith("|") and cells and not cells[-1]:
        cells = cells[:-1]
    return cells


def _is_markdown_delimiter(line: str) -> bool:
    cells = _split_markdown_row(line)
    return len(cells) > 0 and all(_MARKDOWN_DELIMITER_CELL.fullmatch(cell.replace(" ", "")) for cell in cells)


def _canonicalize_markdown_tables(content: str) -> str:
    lines = content.splitlines()
    output: list[str] = []
    index = 0
    changed = False

    while index < len(lines):
        if (
            index + 1 < len(lines)
            and "|" in lines[index]
            and "|" in lines[index + 1]
            and _is_markdown_delimiter(lines[index + 1])
        ):
            changed = True
            table_lines = [lines[index]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            output.extend("\t".join(_split_markdown_row(row)) for row in table_lines)
            continue

        output.append(lines[index])
        index += 1

    return "\n".join(output) if changed else content


def canonicalize_tables_for_text_content(markdown: str) -> str:
    """Replace each table with one row-major text view for Text Content rules.

    Every source cell is emitted once. Tabs separate cells and newlines separate
    rows, preserving anchor continuity without adding alternate cell/row copies.
    The caller retains the original Markdown for table and formatting metrics.
    """
    if not markdown:
        return markdown

    spans = _html_table_spans(markdown)
    for start, end in reversed(spans):
        replacement = _canonicalize_html_table(markdown[start:end])
        markdown = f"{markdown[:start]}\n{replacement}\n{markdown[end:]}"

    return _canonicalize_markdown_tables(markdown)
