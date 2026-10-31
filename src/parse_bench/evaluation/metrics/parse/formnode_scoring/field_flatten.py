"""Flatten FormNode trees to alignable field records.

Both GT and prediction flatten depth-first to `Field` records carrying their
section-anchor ancestry (printed section ids/labels; anonymous sections contribute no
anchor). `static_text` nodes are excluded from scoring (count reported). Table cells
flatten to fields (column caption as label, per-row parent identity) so table-vs-table
and table-vs-legacy-sections both go through the standard alignment machinery.
"""

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from parse_bench.evaluation.metrics.parse.formnode_scoring.bbox_geometry import Box, normalize_node_bboxes
from parse_bench.evaluation.metrics.parse.formnode_scoring.text_normalization import is_empty_value

SELECT_KINDS = {"single_select", "multi_select"}


@dataclass
class Field:
    page: int
    path: tuple[str, ...]  # non-anonymous section anchors (ids/labels) on the ancestry
    parent: int  # identity of the immediate container (for grouping F1)
    id: str | None
    label: str | None
    kind: str
    value: Any  # str | bool | None (None == empty); GT files may carry raw numbers
    options: list[dict[str, Any]]  # [{id, label, value, bbox}] for selects
    flat_idx: int = 0
    # normalized-xyxy boxes (each side normalized by its OWN page dims); () = no claim
    bbox: tuple[Box, ...] = ()
    # alignment state
    page_rank: int = 0  # index within the field's own page (set by align)
    match: "Field | None" = dc_field(default=None, repr=False)
    rescued: bool = False


def flatten_pages(pages: list[dict[str, Any]] | None) -> tuple[list[Field], int]:
    fields: list[Field] = []
    statics = 0
    # current page's declared pixel space, rebound per page by the loop at the bottom;
    # None dims make every bbox on the page normalize to () (ungradeable, not an error)
    page_w: Any = None
    page_h: Any = None

    def val_of(node: dict[str, Any]) -> Any:
        if node.get("isEmpty") is True:
            return None
        v = node.get("value")
        return None if is_empty_value(v) else v

    def expand_options(value_items: list[Any] | None) -> list[dict[str, Any]]:
        """Options are fields; a section groups several controls into ONE option —
        members inherit the section's id/label when they lack their own. (Legacy
        control-dict options in old predicted files still read fine: they carry the
        same id/label/value keys.) Each option keeps its OWN bbox — options carry the
        grounding signal for selects, the parent's union box is never graded."""
        out: list[dict[str, Any]] = []
        for vi in value_items or []:
            if not isinstance(vi, dict):
                continue
            if vi.get("type") in ("section", "option_group"):
                for m in vi.get("items") or []:
                    out.append(
                        {
                            "id": m.get("id") or vi.get("id"),
                            "label": m.get("label") or vi.get("label"),
                            "value": val_of(m),
                            "bbox": normalize_node_bboxes(m.get("bbox"), page_w, page_h),
                        }
                    )
            else:
                out.append(
                    {
                        "id": vi.get("id"),
                        "label": vi.get("label"),
                        "value": val_of(vi),
                        "bbox": normalize_node_bboxes(vi.get("bbox"), page_w, page_h),
                    }
                )
        return out

    def walk_table(node: dict[str, Any], page: int, path: tuple[str, ...]) -> None:
        """Table cells flatten to Fields so the standard alignment machinery applies
        (table-vs-table pairs by column caption + value + position; a table emitted as
        legacy row-sections still cross-matches on label/value).
        - scalar cell -> text Field (column caption as label; null == empty)
        - {items} cell -> its nodes walk normally (unlabeled members inherit the
          column caption, like option-group members inherit the section's)
        - each ROW gets its own parent identity, mirroring per-row sections, so the
          grouping axis keeps working. Malformed rows/cells are skipped."""
        anchor = node.get("id") or node.get("label")
        tpath = path + ((str(anchor),) if anchor else ())
        columns = node.get("columns") if isinstance(node.get("columns"), list) else []

        def col_label(ci: int) -> str | None:
            if columns is not None and ci < len(columns) and isinstance(columns[ci], str) and columns[ci].strip():
                return str(columns[ci])
            return None

        for row in node.get("rows") or []:
            if not isinstance(row, list):
                continue
            row_parent = id(row)
            for ci, cell in enumerate(row):
                if isinstance(cell, dict):
                    cell_items = cell.get("items")
                    if not isinstance(cell_items, list):
                        continue
                    inherited = [
                        dict(m, label=m.get("label") or col_label(ci))
                        if isinstance(m, dict) and m.get("type") == "field"
                        else m
                        for m in cell_items
                    ]
                    walk(inherited, page, tpath, row_parent)
                elif isinstance(cell, str) or cell is None:
                    fields.append(
                        Field(
                            page=page,
                            path=tpath,
                            parent=row_parent,
                            id=None,
                            label=col_label(ci),
                            kind="text",
                            value=None if is_empty_value(cell) else cell,
                            options=[],
                        )
                    )

    def walk(items: list[Any] | None, page: int, path: tuple[str, ...], parent: int) -> None:
        nonlocal statics
        for node in items or []:
            if not isinstance(node, dict):
                continue
            t = node.get("type")
            if t == "static_text":
                statics += 1
            elif t == "section":
                anchor = node.get("id") or node.get("label")
                walk(node.get("items"), page, path + ((str(anchor),) if anchor else ()), id(node))
            elif t == "table":
                walk_table(node, page, path)
            elif t == "field":
                kind = node.get("field") or "text"
                opts = expand_options(node.get("valueItems")) if kind in SELECT_KINDS else []
                fields.append(
                    Field(
                        page=page,
                        path=path,
                        parent=parent,
                        # printed ids are legal as JSON numbers; everything downstream
                        # (regex strip, folding, group keys) expects str
                        id=None if node.get("id") is None else str(node.get("id")),
                        label=node.get("label"),
                        kind=kind,
                        value=val_of(node),
                        options=opts,
                        bbox=normalize_node_bboxes(node.get("bbox"), page_w, page_h),
                    )
                )

    for pg in pages or []:
        page_w, page_h = pg.get("page_width"), pg.get("page_height")
        walk(pg.get("items"), pg.get("page_number", 1), (), id(pg))
    for i, f in enumerate(fields):
        f.flat_idx = i
    return fields, statics
