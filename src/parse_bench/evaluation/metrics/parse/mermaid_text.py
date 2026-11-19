"""Visible-text projection of supported Mermaid diagrams for Text/TAL scoring."""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

from parse_bench.evaluation.metrics.parse.mermaid_graph import parse_mermaid


def project_mermaid_text(markdown: str) -> str:
    """Read complete flowcharts as labels; retain unsupported or malformed source verbatim.

    Markdown's block parser protects examples nested inside ordinary code fences. The existing
    graph parser resolves explicit node labels and repeated edge references. Bare node names
    remain visible labels, as they are in Mermaid. This projection does not score graph topology.
    """
    if "mermaid" not in markdown.lower():
        return markdown
    lines = markdown.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    for token in MarkdownIt("commonmark").parse(markdown):
        if (
            token.type != "fence"
            or token.info.strip().casefold().split(maxsplit=1)[:1] != ["mermaid"]
            or token.map is None
        ):
            continue
        start, end = token.map
        # An unclosed Markdown fence consumes the rest of a page. Preserve that text rather
        # than trusting it as a diagram. YAML front matter can carry a visible title too.
        closing = rf"[ \t]{{0,3}}{re.escape(token.markup[0])}{{{len(token.markup)},}}[ \t]*"
        if end <= start + 1 or not re.fullmatch(closing, lines[end - 1].rstrip("\r\n")):
            continue
        if token.content.lstrip().startswith("---"):
            continue
        graph = parse_mermaid(token.content)
        # Other parsers intentionally omit details (e.g. sequence notes and class members).
        # Until those have complete text projections, leave their source on the legacy path.
        if graph.type != "flowchart" or graph.errors or graph.is_empty or not graph.text_projection_safe:
            continue
        labels = list(graph.groups.values())
        labels.extend(node.label for node in graph.nodes.values() if node.id not in graph.groups)
        labels.extend(edge.label for edge in graph.edges if edge.label)
        replacements.append((start, end, "\n" + "\n".join(labels) + "\n"))
    for start, end, text in reversed(replacements):
        lines[start:end] = [text]
    return "".join(lines)
