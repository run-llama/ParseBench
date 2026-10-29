"""Key-value list markdown -> `KvRecord`s, one record per list line, in the annotation
guide's dialect: `- \\[1a\\] **Label**: value`, `[x]`/`[ ]`, `[blank]` (or the older `[___]`),
option groups as a caption
with nested box-first lines, nested captioned boxes, `#` bands, HTML grids indented under their
key. Lines that are not list items are unparsed and counted, never normalised.

Pass 1 builds an item tree by indentation; pass 2 lets children decide a parent's role (all
box-first: option group; all grids: captioned table; else section) and flattens grid cells
exactly like `formnode_scoring.field_flatten.walk_table`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Tag

from parse_bench.evaluation.metrics.parse.form_kv_scoring.kv_records import (
    COVERAGE_EXHAUSTIVE,
    COVERAGE_SPOT,
    SELECT_KIND,
    KvPage,
    KvRecord,
    classify_value,
    is_blank_token,
    split_box,
)

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<text>.*?)[ \t]*$")
_BULLET_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-*+])(?: (?P<body>.*))?$")
_TABLE_OPEN_RE = re.compile(r"^(?P<indent>[ \t]*)<table\b", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_ID_RE = re.compile(r"^\\\[(?P<id>[^\]\n]+?)\\\]")
_BOLD_LABEL_RE = re.compile(r"^\*\*(?P<label>(?:\\\*|[^*\n])+?)\*\*")
_BOLD_COLON_RE = re.compile(r"^\s*\*\*[^*\n]+\*\*\s*:")
_COVERAGE_RE = re.compile(r"<!--\s*coverage\s*:\s*(?P<cov>\w+)\s*-->", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _unesc(text: str) -> str:
    return text.replace("\\*", "*")


@dataclass
class _Node:
    line: int
    indent: int = 0
    id: str | None = None
    label: str | None = None
    value_text: str | None = None  # None = nothing after the key
    box_first: bool = False
    heading: bool = False
    table_html: str | None = None
    children: list[_Node] = dc_field(default_factory=list)


# ── pass 1: lines -> item tree ──────────────────────────────────────────────────


def _take_id(text: str) -> tuple[str | None, str]:
    m = _ID_RE.match(text)
    if m is None:
        return None, text
    return _unesc(m.group("id").strip()), text[m.end() :]


def _parse_body(node: _Node, body: str, *, heading: bool = False) -> None:
    """Fill id/label/value/box_first from a bullet body (or a heading's text)."""
    node.id, rest = _take_id(body.strip())
    rest = rest.strip()
    if not rest:
        return
    if heading:
        node.label = _unesc(rest)
        return
    if rest.startswith(":"):
        node.value_text = _unesc(rest[1:].strip()) or None
        return
    bold = _BOLD_LABEL_RE.match(rest)
    if bold is not None:
        node.label = _unesc(bold.group("label").strip()) or None
        after = rest[bold.end() :].strip()
        if after.startswith(":"):
            after = after[1:].strip()
        node.value_text = _unesc(after) or None
        return
    if split_box(rest) is not None:
        node.box_first = True
    node.value_text = _unesc(rest)


def _owner_for(stack: list[_Node], indent: int) -> _Node | None:
    """The innermost open item whose bullet sits left of `indent`."""
    owner = None
    for node in stack:
        if node.indent < indent:
            owner = node
    return owner


def _build_tree(markdown: str, diag: dict[str, Any]) -> _Node:
    root = _Node(line=0)
    scope: _Node = root  # the open heading section
    stack: list[_Node] = []  # open list items, shallowest first
    last_item: _Node | None = None
    prev_blank = True
    lines = markdown.split("\n")
    unparsed: list[int] = []
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\r")
        line_no = i + 1
        if not raw.strip():
            prev_blank = True
            last_item = None
            i += 1
            continue
        heading = _HEADING_RE.match(raw)
        if heading is not None:
            node = _Node(line=line_no, heading=True)
            _parse_body(node, heading.group("text"), heading=True)
            root.children.append(node)
            scope = node
            stack.clear()
            last_item = None
            prev_blank = False
            diag["headings"] += 1
            i += 1
            continue
        table = _TABLE_OPEN_RE.match(raw)
        if table is not None:
            indent = len(table.group("indent").expandtabs(4))
            block = [raw]
            while not _TABLE_CLOSE_RE.search(block[-1]) and i + 1 < len(lines):
                i += 1
                block.append(lines[i].rstrip("\r"))
            node = _Node(line=line_no, indent=indent, table_html="\n".join(block))
            owner = _owner_for(stack, indent)
            (owner.children if owner is not None else scope.children).append(node)
            last_item = None
            prev_blank = False
            i += 1
            continue
        bullet = _BULLET_RE.match(raw)
        if bullet is not None:
            indent = len(bullet.group("indent").expandtabs(4))
            node = _Node(line=line_no, indent=indent)
            _parse_body(node, bullet.group("body") or "")
            while stack and stack[-1].indent >= indent:
                stack.pop()
            (stack[-1].children if stack else scope.children).append(node)
            stack.append(node)
            last_item = node
            prev_blank = False
            i += 1
            continue
        if last_item is not None and not prev_blank:
            text = _unesc(raw.strip())
            last_item.value_text = f"{last_item.value_text}\n{text}" if last_item.value_text else text
            diag["continuation_lines"] += 1
            i += 1
            continue
        unparsed.append(line_no)
        if _BOLD_COLON_RE.match(raw):
            diag["bold_colon_unparsed"] += 1
        prev_blank = False
        i += 1
    diag["unparsed_lines"] = unparsed
    diag["unparsed"] = len(unparsed)
    return root


# ── pass 2: item tree -> records ────────────────────────────────────────────────


class _Emitter:
    def __init__(self, page: int, diag: dict[str, Any]) -> None:
        self.page = page
        self.diag = diag
        self.records: list[KvRecord] = []
        self._parents = 0
        self.cell_caption: str | None = None  # column caption inherited by unlabeled cell records

    def new_parent(self) -> int:
        """Fresh container identity; never `id()` of parser objects, whose addresses get recycled."""
        self._parents += 1
        return self._parents

    def add(
        self,
        line: int,
        path: tuple[str, ...],
        parent: int,
        *,
        kind: str,
        value: Any,
        id_: str | None = None,
        label: str | None = None,
        raw_value: str | None = None,
        options: list[dict[str, Any]] | None = None,
        signer: str | None = None,
    ) -> KvRecord:
        rec = KvRecord(
            page=self.page,
            path=path,
            parent=parent,
            id=id_,
            label=label,
            kind=kind,
            value=value,
            options=options or [],
            flat_idx=len(self.records),
            raw_label=label,
            raw_value=raw_value,
            source_line=line,
            signer=signer,
        )
        if id_ is not None:
            self.diag["ids"] += 1
        self.records.append(rec)
        return rec

    def emit_nodes(self, nodes: list[_Node], path: tuple[str, ...], parent: int) -> None:
        for node in nodes:
            if node.table_html is not None:
                self.emit_table(node, None, path)
            else:
                self.emit_node(node, path, parent)

    def emit_node(self, node: _Node, path: tuple[str, ...], parent: int) -> None:
        anchor = node.id if node.id else node.label
        if node.children and all(c.table_html is not None for c in node.children):
            if node.value_text:
                self.emit_leaf(node, path, parent)
            for child in node.children:
                self.emit_table(child, anchor, path)
            return
        if node.children and all(c.box_first for c in node.children):
            self.emit_select(node, path, parent)
            return
        if node.children or node.heading:
            if node.value_text and not node.box_first:
                self.emit_leaf(node, path, parent)
            self.emit_nodes(node.children, path + ((anchor,) if anchor else ()), self.new_parent())
            return
        self.emit_leaf(node, path, parent)

    def emit_select(self, node: _Node, path: tuple[str, ...], parent: int) -> None:
        options: list[dict[str, Any]] = []
        for child in node.children:
            box = split_box(child.value_text or "")
            state, rest = box if box is not None else (False, "")
            options.append(
                {
                    "id": child.id,
                    "label": None if (not rest or is_blank_token(rest)) else rest,
                    "value": state,
                    "bbox": (),
                }
            )
        label = node.label if node.label is not None else self.cell_caption
        self.add(node.line, path, parent, kind=SELECT_KIND, value=None, id_=node.id, label=label, options=options)
        anchor = node.id if node.id else node.label
        for child in node.children:
            if child.children:
                self.emit_nodes(child.children, path + ((anchor,) if anchor else ()), self.new_parent())

    def emit_leaf(self, node: _Node, path: tuple[str, ...], parent: int) -> None:
        self.emit_value(node.line, path, parent, node.id, node.label, node.value_text, box_first=node.box_first)

    def emit_value(
        self,
        line: int,
        path: tuple[str, ...],
        parent: int,
        id_: str | None,
        label: str | None,
        value_text: str | None,
        *,
        box_first: bool = False,
    ) -> None:
        tok = classify_value(value_text)
        if label is None and not box_first:
            label = self.cell_caption
        if tok.kind == "checkbox":
            if box_first:
                label = tok.write_in if tok.write_in is not None else self.cell_caption
                self.add(line, path, parent, kind="checkbox", value=tok.value, id_=id_, label=label)
                return
            self.add(line, path, parent, kind="checkbox", value=tok.value, id_=id_, label=label)
            if tok.write_in_present:
                self.add(
                    line, path, parent, kind="text", value=tok.write_in, id_=id_, label=label, raw_value=tok.write_in
                )
            return
        if tok.kind == "signature":
            raw = value_text.strip() if value_text else None
            self.add(
                line, path, parent, kind="signature", value=True, id_=id_, label=label, raw_value=raw, signer=tok.signer
            )
            return
        self.add(line, path, parent, kind="text", value=tok.value, id_=id_, label=label, raw_value=tok.value)

    def emit_cell(self, line: int, path: tuple[str, ...], row_parent: int, caption: str | None, text: str) -> None:
        """A cell is list markdown when it holds bullets (nested groups included), else one
        plain value; unlabeled records inherit the column caption."""
        if not any(_BULLET_RE.match(ln) for ln in text.split("\n")):
            self.emit_value(line, path, row_parent, None, caption, _unesc(text.strip()) or None)
            return
        sub = new_diagnostics()
        root = _build_tree(text, sub)
        for ln in root.children:
            _shift_lines(ln, line)
        prev, self.cell_caption = self.cell_caption, caption
        self.emit_nodes(root.children, path, row_parent)
        self.cell_caption = prev
        self.diag["unparsed"] += sub["unparsed"]

    def emit_table(self, node: _Node, anchor: str | None, path: tuple[str, ...]) -> None:
        tpath = path + ((anchor,) if anchor else ())
        soup = BeautifulSoup(node.table_html or "", "lxml")
        table = soup.find("table")
        if not isinstance(table, Tag):
            return
        rows = [tr for tr in table.find_all("tr") if _closest_table(tr) is table]
        captions: list[str | None] = []
        body_rows = rows
        if rows and _cells(rows[0]) and all(c.name == "th" for c in _cells(rows[0])):
            captions = [_unesc(c.get_text(" ", strip=True)) or None for c in _cells(rows[0])]
            body_rows = rows[1:]
        for tr in body_rows:
            row_parent = self.new_parent()
            for ci, cell in enumerate(_cells(tr)):
                caption = captions[ci] if ci < len(captions) else None
                text = _unescape_html(_HTML_TAG_RE.sub("", _BR_RE.sub("\n", cell.decode_contents())))
                self.emit_cell(node.line, tpath, row_parent, caption, text)
        self.diag["tables"] += 1


def _shift_lines(node: _Node, line: int) -> None:
    node.line = line
    for child in node.children:
        _shift_lines(child, line)


def _closest_table(tag: Tag) -> Tag | None:
    parent = tag.parent
    while parent is not None:
        if isinstance(parent, Tag) and parent.name == "table":
            return parent
        parent = parent.parent
    return None


def _cells(tr: Tag) -> list[Tag]:
    return [c for c in tr.find_all(["td", "th"], recursive=False) if isinstance(c, Tag)]


def _unescape_html(text: str) -> str:
    return BeautifulSoup(text, "lxml").get_text() if "&" in text else text


# ── public API ──────────────────────────────────────────────────────────────────


def new_diagnostics() -> dict[str, Any]:
    return {
        "records": 0,
        "ids": 0,
        "unparsed": 0,
        "unparsed_lines": [],
        "bold_colon_unparsed": 0,
        "continuation_lines": 0,
        "headings": 0,
        "tables": 0,
    }


def parse_kv_markdown(markdown: str, page_number: int = 1, coverage: str | None = None) -> KvPage:
    """Coverage defaults to exhaustive; a `<!-- coverage: spot -->` comment on the page overrides it."""
    diag = new_diagnostics()
    root = _build_tree(markdown or "", diag)
    emitter = _Emitter(page_number, diag)
    emitter.emit_nodes(root.children, (), emitter.new_parent())
    diag["records"] = len(emitter.records)
    if coverage is None:
        m = _COVERAGE_RE.search(markdown or "")
        coverage = m.group("cov").lower() if m else COVERAGE_EXHAUSTIVE
    if coverage not in (COVERAGE_EXHAUSTIVE, COVERAGE_SPOT):
        raise ValueError(f"page {page_number}: unknown coverage {coverage!r} (expected 'exhaustive' or 'spot')")
    return KvPage(page_number=page_number, records=emitter.records, coverage=coverage, diagnostics=diag)
