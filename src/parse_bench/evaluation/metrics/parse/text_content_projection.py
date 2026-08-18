"""Canonical plain-text projection for ParseBench Text Content rules."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

_MARKDOWN_DELIMITER_CELL = re.compile(r"^:?-{3,}:?$")


def _html_table_spans(content: str) -> list[tuple[int, int]]:
    """Return non-overlapping top-level HTML table spans."""
    spans: list[tuple[int, int]] = []
    lower = content.lower()
    search_start = 0

    while True:
        start = lower.find("<table", search_start)
        if start == -1:
            break

        tag_end = start + len("<table")
        if tag_end < len(lower) and lower[tag_end] not in (">", " ", "\t", "\n", "\r"):
            search_start = start + 1
            continue

        depth = 0
        position = start
        end = -1
        while position < len(lower):
            next_open = lower.find("<table", position + 1)
            next_close = lower.find("</table>", position + 1)
            if next_close == -1:
                break

            if next_open != -1 and next_open < next_close:
                nested_end = next_open + len("<table")
                if nested_end < len(lower) and lower[nested_end] not in (">", " ", "\t", "\n", "\r"):
                    position = next_open
                    continue
                depth += 1
                position = next_open
                continue

            if depth == 0:
                end = next_close + len("</table>")
                break
            depth -= 1
            position = next_close

        if end == -1:
            break
        spans.append((start, end))
        search_start = end

    return spans


def _normalize_cell_text(cell: Tag) -> str:
    return " ".join(cell.get_text(" ", strip=True).split())


def _canonicalize_html_table(table_html: str) -> str:
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return table_html

    rows: list[str] = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = [
            _normalize_cell_text(cell)
            for cell in row.find_all(["th", "td"])
            if cell.find_parent("tr") is row
        ]
        if cells:
            rows.append("\t".join(cells))

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
