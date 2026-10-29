"""Structure-only skeleton-TEDS diagnostic for FormNode pages.

Answers what the field-level axes cannot: for the fields both sides agree on, did the
HIERARCHY come out right? Field content is deliberately invisible — leaves are labeled by
MATCHED-PAIR IDENTITY, so the score moves only when PLACEMENT moves. An extra emitted
diagnostic: never the headline, never in the composite.

Tree construction (one tree per page; GT and prediction built identically):
  - the page root is labeled `page`;
  - internal nodes are sections and tables, labeled by their folded anchor
    (`fold(id or label)`, `""` when anonymous);
  - table rows are anonymous internal nodes (label `""`), mirroring the per-row parent
    identity the flattener gives table cells;
  - leaves are fields, consumed from the page's already-flattened `Field` list in
    `flatten_pages` DFS order — same recursion and same skip rules (`static_text`
    skipped, malformed nodes/rows/cells skipped, `{items}` cells walked as nodes), so the
    k-th leaf produced IS the k-th field. A cursor/length disagreement means the fields
    list did not come from these items and raises ValueError;
  - a leaf is labeled `m{k}` for the k-th one-to-one matched GT field and for the pred
    field it matched, and `only` for everything else — unmatched fields on either side,
    rescued GT selects, and the loose checkboxes that rescued them.

Normalization: anchors go through `text_normalization.fold` (casefold, collapsed
whitespace). Leaf labels carry no field text at all.

Scoring: APTED tree edit distance with unit insert/delete and a 0/1 label rename, then
`1 - distance / max(gt_nodes, pred_nodes)` clamped at 0 (node counts include the root and
all internal nodes). The returned weight is that same node maximum, so callers aggregate
node-weighted — `sum(teds * w) / sum(w)` — and a 40-node page moves a document number ten
times as much as a 4-node page.
"""

from collections.abc import Callable
from typing import Any

from apted import APTED, Config
from apted.helpers import Tree

from parse_bench.evaluation.metrics.parse.formnode_scoring.field_flatten import Field
from parse_bench.evaluation.metrics.parse.formnode_scoring.text_normalization import fold

PAGE_LABEL = "page"
ANON_LABEL = ""
UNPAIRED_LABEL = "only"


class LabelConfig(Config):
    """Unit-cost edit config: renaming costs 1 unless the labels are identical."""

    def rename(self, node1: Any, node2: Any) -> int:
        return 0 if node1.name == node2.name else 1

    def delete(self, node: Any) -> int:
        return 1

    def insert(self, node: Any) -> int:
        return 1


def _anchor_label(node: dict[str, Any]) -> str:
    """A section/table anchor label: folded printed id, else folded label, else `""`."""
    anchor = node.get("id") or node.get("label")
    return fold(anchor) if anchor else ANON_LABEL


def _leaf_labels(gt_fields: list[Field], pred_fields: list[Field]) -> tuple[dict[int, str], dict[int, str]]:
    """Matched-pair leaf labels, keyed by `id()` (Field is not hashable by value).

    The k-th one-to-one matched GT field and its pred partner share `m{k}`. Rescued GT
    selects (and the loose checkboxes pointing at them) are not one-to-one pairs, so both
    sides fall through to `only` along with every unmatched field.
    """
    gt_labels: dict[int, str] = {}
    k = 0
    for g in gt_fields:
        if g.match is not None and not g.rescued:
            gt_labels[id(g)] = f"m{k}"
            k += 1
        else:
            gt_labels[id(g)] = UNPAIRED_LABEL

    pred_labels: dict[int, str] = {}
    for p in pred_fields:
        matched_gt = p.match
        pred_labels[id(p)] = UNPAIRED_LABEL if matched_gt is None else gt_labels.get(id(matched_gt), UNPAIRED_LABEL)
    return gt_labels, pred_labels


def _build_skeleton(
    items: list[dict[str, Any]] | None,
    fields: list[Field],
    leaf_label: Callable[[Field], str],
    side: str,
) -> tuple[Any, int]:
    """Build one page's skeleton tree, returning (root, node_count).

    Walks `items` in `flatten_pages` DFS order with its skip rules, consuming `fields`
    with a cursor as each field/cell leaf is produced.
    """
    cursor = 0
    node_count = 0

    def node(label: str) -> Any:
        nonlocal node_count
        node_count += 1
        return Tree(label)

    def leaf() -> Any:
        nonlocal cursor
        if cursor >= len(fields):
            raise ValueError(
                f"{side} skeleton produced more leaves than flattened fields "
                f"({cursor + 1} > {len(fields)}): the fields list did not come from these items"
            )
        f = fields[cursor]
        cursor += 1
        return node(leaf_label(f))

    def walk_table(table: dict[str, Any], parent: Any) -> None:
        table_node = node(_anchor_label(table))
        parent.children.append(table_node)
        for row in table.get("rows") or []:
            if not isinstance(row, list):
                continue
            row_node = node(ANON_LABEL)
            table_node.children.append(row_node)
            for cell in row:
                if isinstance(cell, dict):
                    cell_items = cell.get("items")
                    if not isinstance(cell_items, list):
                        continue
                    walk(cell_items, row_node)
                elif isinstance(cell, str) or cell is None:
                    row_node.children.append(leaf())

    def walk(nodes: list[Any] | None, parent: Any) -> None:
        for entry in nodes or []:
            if not isinstance(entry, dict):
                continue
            t = entry.get("type")
            if t == "section":
                section_node = node(_anchor_label(entry))
                parent.children.append(section_node)
                walk(entry.get("items"), section_node)
            elif t == "table":
                walk_table(entry, parent)
            elif t == "field":
                parent.children.append(leaf())

    root = node(PAGE_LABEL)
    walk(items, root)
    if cursor != len(fields):
        raise ValueError(
            f"{side} skeleton consumed {cursor} of {len(fields)} flattened fields: "
            "the fields list did not come from these items"
        )
    return root, node_count


def page_skeleton_teds(
    gt_page_items: list[dict[str, Any]] | None,
    pred_page_items: list[dict[str, Any]] | None,
    gt_page_fields: list[Field],
    pred_page_fields: list[Field],
) -> tuple[float, int]:
    """Skeleton-TEDS for one page pair: (teds, node weight).

    `*_page_items` are that page's FormNode `items` lists; `*_page_fields` are the same
    page's `Field` objects in `flatten_pages` DFS order, AFTER `align()` has populated
    `.match`/`.rescued`. Two pages with no items at all agree trivially: `(1.0, 1)`.
    """
    if not gt_page_items and not pred_page_items:
        return (1.0, 1)

    gt_labels, pred_labels = _leaf_labels(gt_page_fields, pred_page_fields)
    gt_tree, gt_nodes = _build_skeleton(gt_page_items, gt_page_fields, lambda f: gt_labels[id(f)], "ground truth")
    pred_tree, pred_nodes = _build_skeleton(
        pred_page_items, pred_page_fields, lambda f: pred_labels[id(f)], "prediction"
    )

    weight = max(gt_nodes, pred_nodes)
    distance = float(APTED(gt_tree, pred_tree, LabelConfig()).compute_edit_distance())
    return (max(0.0, 1.0 - distance / weight), weight)
