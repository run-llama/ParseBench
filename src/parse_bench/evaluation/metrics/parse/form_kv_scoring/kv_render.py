"""FormNode tree -> key-value list markdown in the annotation guide's dialect.

`render_guide` is the ground-truth producer: GT is rendered and parsed by `kv_grammar` like a
prediction, so both sides share one grammar. Shapes: `- \\[id\\] **Label**: value`, `[___]`
blanks, `[x]`/`[ ]`, `[signature]`, option groups as a caption (a bare `- ` when nothing is
printed) with nested box-first lines (a section option's other members nest under its box
line), labeled sections as nested parent items (never `###`: a
heading's scope runs to the next heading), keyed grids as HTML tables indented under the key.
"""

from __future__ import annotations

import html
from typing import Any

from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_grammar import parse_kv_markdown
from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_records import (
    BLANK_TOKEN,
    COVERAGE_EXHAUSTIVE,
    KvPage,
)

INDENT = "  "
SELECT_KINDS = ("single_select", "multi_select")


def _val_of(node: dict[str, Any]) -> Any:
    if node.get("isEmpty") is True:
        return None
    v = node.get("value")
    return None if v is None or (isinstance(v, str) and v.strip() == "") else v


def _is_compound(node: dict[str, Any]) -> bool:
    vi = node.get("valueItems")
    return isinstance(vi, list) and len(vi) > 0


def _esc(text: Any) -> str:
    """A printed `*` is content, not emphasis (the guide's escaping rule)."""
    return str(text).replace("*", "\\*")


def _eid(id_: Any) -> str:
    return f"\\[{_esc(id_)}\\]"


def _key(id_: Any, label: Any) -> str:
    parts: list[str] = []
    if id_ is not None and str(id_) != "":
        parts.append(_eid(id_))
    if label:
        parts.append(f"**{_esc(label)}**")
    return " ".join(parts)


def _guide_value(node: dict[str, Any]) -> str:
    kind = node.get("field") or "text"
    if kind == "checkbox":
        return "[x]" if node.get("value") is True else "[ ]"
    if kind == "signature":
        return "[signature]" if node.get("value") is True else BLANK_TOKEN
    v = _val_of(node)
    return BLANK_TOKEN if v is None else _esc(v)


def _guide_field_text(node: dict[str, Any]) -> str:
    """`\\[id\\] **Label**: value` with absent parts omitted; id-only keeps the colon."""
    id_, label = node.get("id"), node.get("label")
    value = _guide_value(node)
    if label:
        key = _key(id_, label)
        sep = " " if str(label).endswith(":") else ": "
        return f"{key}{sep}{value}"
    if id_ is not None and str(id_) != "":
        return f"{_eid(id_)}: {value}"
    return value


def _guide_member_line(m: dict[str, Any]) -> str:
    if m.get("type") != "field":
        return ""
    return _guide_field_text(dict(m, id=None)) if m.get("label") else _guide_value(m)


def _is_box(m: dict[str, Any]) -> bool:
    return m.get("type") == "field" and m.get("field") == "checkbox"


def _inline_write_in(box: dict[str, Any], rest: list[dict[str, Any]]) -> bool:
    """`\\[3\\] [x] 8888`: an unlabeled box with a single unlabeled text member joins on one line."""
    if box.get("label") or len(rest) != 1:
        return False
    m = rest[0]
    return (
        m.get("type") == "field" and (m.get("field") or "text") == "text" and not m.get("label") and m.get("id") is None
    )


def _guide_option_lines(item: dict[str, Any]) -> tuple[str, list[str]]:
    """An option's box-first line plus the lines nested under it (a section option's other members)."""
    t = item.get("type")
    id_ = item.get("id")
    prefix = f"{_eid(id_)} " if id_ is not None and str(id_) != "" else ""
    if t == "section":
        members = [m for m in (item.get("items") or []) if isinstance(m, dict)]
        boxes = [m for m in members if _is_box(m)]
        if not boxes:
            first, *others = members or [{}]
            return f"{prefix}{_guide_member_line(first)}".strip(), [n for n in map(_guide_member_line, others) if n]
        box = boxes[0]
        rest: list[dict[str, Any]] = [m for m in members if m is not box]
        state = "[x]" if box.get("value") is True else "[ ]"
        inline: str | None = box.get("label") or item.get("label")
        if _inline_write_in(box, rest):
            inline = _guide_value(rest[0])
            rest = []
        line = f"{prefix}{state} {_esc(inline)}".strip() if inline else f"{prefix}{state}"
        return line, [n for n in map(_guide_member_line, rest) if n]
    if t == "table":
        return f"{prefix}{'**' + _esc(item['label']) + '**' if item.get('label') else ''}".strip(), []
    if (item.get("field") or "text") == "checkbox":
        mark = "[x]" if item.get("value") is True else "[ ]"
        return (f"{prefix}{mark} {_esc(item['label'])}" if item.get("label") else f"{prefix}{mark}"), []
    return _guide_field_text(item), []


def tree_units(items: list[dict[str, Any]] | None) -> int:
    """Records a lossless render of `items` parses to: one per field (an option group is one
    record), one per grid cell, plus a section option's members not joined on its box line."""
    n = 0
    for node in items or []:
        if not isinstance(node, dict):
            continue
        t = node.get("type")
        if t == "section":
            n += tree_units(node.get("items"))
        elif t == "table":
            for row in node.get("rows") or []:
                for cell in row if isinstance(row, list) else []:
                    n += (
                        (tree_units(cell.get("items")) or 1) if isinstance(cell, dict) else 1
                    )  # no controls = a blank cell
        elif t == "field":
            n += 1
            for vi in node.get("valueItems") or [] if _is_compound(node) else []:
                if isinstance(vi, dict) and vi.get("type") == "section":
                    members = [m for m in (vi.get("items") or []) if isinstance(m, dict)]
                    boxes = [m for m in members if _is_box(m)]
                    rest = [m for m in members if m is not boxes[0]] if boxes else members[1:]
                    n += 0 if (boxes and _inline_write_in(boxes[0], rest)) else len(rest)
    return n


def _with_continuation(first: str, depth: int) -> list[str]:
    lines = first.split("\n")
    pad = INDENT * (depth + 1)
    return [lines[0]] + [f"{pad}{line}" for line in lines[1:]]


def _guide_cell_html(cell: Any) -> str:
    """A cell is a plain value, or list markdown (lines joined by `<br>`) when it holds
    form controls; a lone unlabeled control reads as the plain value. Static text is
    not scored and is not rendered."""
    if cell is None:
        return ""
    if isinstance(cell, str):
        return html.escape(_esc(cell), quote=False)
    items = [m for m in (cell.get("items") or []) if isinstance(m, dict)] if isinstance(cell, dict) else []
    items = [m for m in items if m.get("type") != "static_text"]
    if len(items) == 1 and items[0].get("type") == "field" and not items[0].get("label") and items[0].get("id") is None:
        only = items[0]
        if not _is_compound(only):
            return html.escape(_guide_value(only), quote=False)
    lines = [line for m in items for line in _guide_lines(m, 0)]
    return "<br>".join(html.escape(line, quote=False) for line in lines if line.strip())


def _guide_table_html(node: dict[str, Any]) -> list[str]:
    lines = ["<table>"]
    columns = node.get("columns") if isinstance(node.get("columns"), list) else []
    if columns:
        lines.append("<tr>" + "".join(f"<th>{html.escape(_esc(c), quote=False)}</th>" for c in columns) + "</tr>")
    for row in node.get("rows") or []:
        if not isinstance(row, list):
            continue
        lines.append("<tr>" + "".join(f"<td>{_guide_cell_html(c)}</td>" for c in row) + "</tr>")
    lines.append("</table>")
    return lines


def _guide_lines(node: dict[str, Any], depth: int) -> list[str]:
    pad = INDENT * depth
    t = node.get("type")
    if t == "static_text":
        return []
    if t == "section":
        key = _key(node.get("id"), node.get("label"))
        children = [c for c in (node.get("items") or []) if isinstance(c, dict)]
        if not key:
            return [line for c in children for line in _guide_lines(c, depth)]
        nested = [line for c in children for line in _guide_lines(c, depth + 1)]
        return [f"{pad}- {key}", *nested] if nested else []  # a section with nothing to score is not a record
    if t == "table":
        key = _key(node.get("id"), node.get("label"))
        table_depth = depth + 1 if key else depth
        table_pad = INDENT * table_depth
        out = [f"{pad}- {key}"] if key else []
        out.append("")
        out.extend(f"{table_pad}{line}" for line in _guide_table_html(node))
        out.append("")
        return out
    if t != "field":
        return []
    if _is_compound(node):
        key = _key(node.get("id"), node.get("label"))
        child_pad = INDENT * (depth + 1)
        out = [f"{pad}- {key}".rstrip()]
        for item in node.get("valueItems") or []:
            if isinstance(item, dict):
                line, nested = _guide_option_lines(item)
                out.append(f"{child_pad}- {line}")
                out.extend(f"{INDENT * (depth + 2)}- {n}" for n in nested)
        return out
    first, *rest = _with_continuation(_guide_field_text(node), depth)
    return [f"{pad}- {first}", *rest]


def render_guide(nodes: list[dict[str, Any]] | None) -> str:
    return "\n".join(line for n in nodes or [] if isinstance(n, dict) for line in _guide_lines(n, 0))


def pages_from_tree(pages: list[dict[str, Any]] | None, coverage: str = COVERAGE_EXHAUSTIVE) -> list[KvPage]:
    """GT pages: each page rendered in the guide's dialect and parsed like a prediction.
    `diagnostics["roundtrip_loss"]` is records minus `tree_units`: nonzero means the list
    could not express the tree (a text-only option, a section option without a box)."""
    out = []
    for pg in pages or []:
        if not isinstance(pg, dict):
            continue
        items = pg.get("items") or []
        page = parse_kv_markdown(render_guide(items), pg.get("page_number", 1), coverage)
        page.diagnostics["tree_units"] = tree_units(items)
        page.diagnostics["roundtrip_loss"] = len(page.records) - page.diagnostics["tree_units"]
        out.append(page)
    return out
