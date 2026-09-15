"""Shared table identification + parsing stage.

Run once per (expected_md, actual_md) so all table metrics consume the
same set of tables, paired the same way. Normalization stays inside each
metric — GriTS and TRM apply their own per-cell normalization downstream.

GT parse failures raise loudly (dataset bug). Pred parse failures are
dropped silently (model bug — dropped tables count as unmatched_expected
for any GT they would have paired with, so the model still pays the score).
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from parse_bench.evaluation.metrics.parse.table_parsing import TableData, parse_html_tables

if TYPE_CHECKING:
    from parse_bench.evaluation.metrics.parse.table_title_stripping import HeaderHints

_TABLE_OPEN_RE = re.compile(r"<table(?=[>\s])", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)


def extract_html_tables(content: str) -> list[str]:
    """Extract all top-level HTML table strings from markdown/HTML content.

    Uses depth-aware string scanning to correctly handle nested tables.
    Nested tables (inside <td> cells) are included as part of the outer
    table's HTML string, not extracted as separate entries.
    """
    if not content:
        return []

    tables: list[str] = []
    search_start = 0
    while True:
        open_match = _TABLE_OPEN_RE.search(content, search_start)
        if open_match is None:
            break
        start = open_match.start()

        # Track nesting depth to find the matching </table>
        depth = 0
        pos = open_match.end()
        end = -1
        while pos < len(content):
            next_open = _TABLE_OPEN_RE.search(content, pos)
            next_close = _TABLE_CLOSE_RE.search(content, pos)
            if next_close is None:
                break
            if next_open is not None and next_open.start() < next_close.start():
                depth += 1
                pos = next_open.end()
            else:
                if depth == 0:
                    end = next_close.end()
                    break
                depth -= 1
                pos = next_close.end()
        if end == -1:
            tables.append(content[start:])
            break
        tables.append(content[start:end])
        search_start = end

    return tables


class GroundTruthTableParseError(RuntimeError):
    """Raised when a ground-truth table cannot be parsed. Dataset bug."""


@dataclass(frozen=True)
class ExtractedTable:
    """One table, identified once. Cell content is NOT normalized.

    - GriTS reads ``raw_html`` and runs it through its own ``html_to_cells`` +
      ``normalize_cell_text`` path (P2). In P5 it switches to reading
      ``table_data`` directly with upgraded normalization.
    - TRM reads ``table_data`` and runs it through its own ``normalize_table``
      (P3 onward).
    """

    raw_html: str
    table_data: TableData  # raw parse_html_tables output, NOT normalized
    header_hints: "HeaderHints | None" = None  # populated by strip_title_rows


@dataclass(frozen=True)
class TableExtractionCounts:
    """Per-doc table counts surfaced as MetricValues."""

    expected: int
    actual: int
    unparseable_pred: int  # dropped pred tables (gt is always 0 — they raise)


def extract_normalized_tables(
    md: str,
    *,
    side: str,  # "expected" or "actual"
    doc_id: str | None = None,
) -> tuple[list[ExtractedTable], int]:
    """Extract and parse all tables on one side of a doc.

    Returns ``(tables, n_unparseable)``. ``n_unparseable`` is always 0 for the
    expected side because GT parse failures raise.

    Note: despite the name, this stage does **not** normalize cell content —
    each metric applies its own normalization downstream. The name is kept
    for backward compatibility with the plan.
    """
    raw_slices = extract_html_tables(md)
    if not raw_slices:
        return [], 0

    # Parse each slice independently rather than calling parse_html_tables
    # on the whole doc and zipping by index. The two parsers (depth-aware
    # string scanner in extract_html_tables vs. lxml/bs4 in parse_html_tables)
    # can disagree on what counts as a top-level table for malformed HTML,
    # which would silently mis-pair raw_html with table_data. Parsing each
    # slice individually makes the (raw_html, table_data) pairing correct
    # by construction.
    tables: list[ExtractedTable] = []
    unparseable = 0
    for i, raw in enumerate(raw_slices):
        parsed_one = parse_html_tables(raw)
        if not parsed_one:
            if side == "expected":
                raise GroundTruthTableParseError(f"Failed to parse expected table {i} in doc {doc_id!r}")
            unparseable += 1
            continue
        tables.append(ExtractedTable(raw_html=raw, table_data=parsed_one[0]))
    return tables, unparseable


def extract_table_pairs(
    expected_md: str,
    actual_md: str,
    *,
    doc_id: str | None = None,
) -> tuple[list[ExtractedTable], list[ExtractedTable], TableExtractionCounts]:
    """Extract both sides for one doc."""
    expected, _ = extract_normalized_tables(expected_md, side="expected", doc_id=doc_id)
    actual, n_unparseable_pred = extract_normalized_tables(actual_md, side="actual", doc_id=doc_id)
    counts = TableExtractionCounts(
        expected=len(expected),
        actual=len(actual),
        unparseable_pred=n_unparseable_pred,
    )
    return expected, actual, counts
