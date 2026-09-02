"""Diagram rules: score mermaid blocks by graph semantics, never by source text.

Three rule types share one pipeline. Every fenced ``mermaid`` block on the page is parsed into
a :class:`~parse_bench.evaluation.metrics.parse.mermaid_graph.Graph` (labelled nodes,
labelled/directed edges). The ground truth is the same shape. Nodes are aligned by label with a
fuzzy ratio (rapidfuzz ``token_set_ratio``, so "Customer submits application form" still meets
"Submit application") under a one-to-one Hungarian assignment; edges are then compared on the
aligned ids, honouring direction only when the reference edge is directed.

* ``diagram_graph`` — per diagram: a block exists and parses, its type is acceptable, node
  recall/precision and edge F1 clear their thresholds, and the block sits between the text
  anchors. Graded score = mean of the axis scores; ``result_details`` carries both graphs, the
  alignment and every axis so the report can draw them side by side.
* ``diagram_edge`` — one anchor relation (``edge`` / ``path`` / ``no_edge``) between two labelled
  nodes; robust to however creatively the rest of the diagram was transcribed.
* ``diagram_count`` — page-level: the expected number of parseable blocks (0 on negatives).
"""

from __future__ import annotations

import re
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from parse_bench.evaluation.metrics.parse.mermaid_graph import (
    Graph,
    extract_mermaid_blocks,
    graph_from_dict,
    parse_mermaid,
)
from parse_bench.evaluation.metrics.parse.rules_base import ParseTestRule
from parse_bench.evaluation.metrics.parse.test_types import TestType
from parse_bench.evaluation.metrics.parse.utils import normalize_text
from parse_bench.test_cases.parse_rule_schemas import (
    ParseDiagramCountRule,
    ParseDiagramEdgeRule,
    ParseDiagramGraphRule,
)

# ``graph``/``flowchart`` are one family; org charts and trees are commonly drawn as either.
_TYPE_ALIASES: dict[str, set[str]] = {
    "flowchart": {"flowchart", "graph"},
    "state": {"state", "statediagram", "statediagram-v2"},
    "sequence": {"sequence", "sequencediagram"},
    "mindmap": {"mindmap"},
    "class": {"class", "classdiagram"},
    "er": {"er", "erdiagram"},
}


def _canonical_type(name: str) -> str:
    low = name.strip().lower()
    for canon, names in _TYPE_ALIASES.items():
        if low in names:
            return canon
    return low


def norm_label(label: str) -> str:
    """Comparison form of a node/edge label: bench text normalisation + lowercase + no punctuation."""
    s = normalize_text(label or "").lower()
    s = re.sub(r"[^\w\s%/&+-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def label_similarity(a: str, b: str) -> float:
    """0–100 similarity tolerant to word order and abbreviation (``token_set_ratio``)."""
    na, nb = norm_label(a), norm_label(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 100.0
    # Very short single-word labels ("End", "Yes", "Send") are only a match when identical: any
    # ratio-based score treats a one-letter difference as near-equal, which is exactly wrong here.
    shortest = min(na, nb, key=len)
    if len(shortest) <= 5 and len(shortest.split()) == 1:
        return 100.0 if shortest in (na.split() + nb.split()) and na.split() == nb.split() else 40.0
    score = float(fuzz.token_set_ratio(na, nb))
    # Abbreviations: "Submit application" for "Customer submits application form". Accept the
    # shorter label as a substring match only when it is substantial enough that a stray
    # short word cannot hide inside another ("End" in "Send" must not count).
    shorter = min(na, nb, key=len)
    if len(shorter) >= 8 and len(shorter.split()) >= 2:
        score = max(score, float(fuzz.partial_ratio(na, nb)))
    # Multi-line box labels often come out of the parser with the line breaks dropped and no
    # space in their place ("Office ofState Grand JuryReview"); compare without whitespace too.
    da, db = na.replace(" ", ""), nb.replace(" ", "")
    if len(min(da, db, key=len)) >= 8:
        score = max(score, float(fuzz.ratio(da, db)))
    return score


def page_blocks(md_content: str, page: int | None) -> list[dict[str, Any]]:
    """Mermaid blocks on the rule's page, each with its parsed graph."""
    page_md = _page_markdown(md_content, page)
    blocks = extract_mermaid_blocks(page_md)
    for b in blocks:
        b["graph"] = parse_mermaid(b["source"])
    return blocks


def _page_markdown(md_content: str, page: int | None) -> str:
    if page is None or "\f" not in md_content:
        return md_content
    parts = md_content.split("\f")
    if 1 <= page <= len(parts):
        return parts[page - 1]
    return md_content


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def align_nodes(expected: Graph, predicted: Graph, threshold: int) -> dict[str, Any]:
    """One-to-one node alignment maximising total label similarity; pairs below ``threshold`` are dropped."""
    exp_ids = list(expected.nodes)
    pred_ids = list(predicted.nodes)
    if not exp_ids or not pred_ids:
        return {"pairs": [], "unmatched_expected": exp_ids, "unmatched_predicted": pred_ids, "matrix": []}
    matrix = np.zeros((len(exp_ids), len(pred_ids)), dtype=float)
    for i, e in enumerate(exp_ids):
        for j, p in enumerate(pred_ids):
            matrix[i, j] = label_similarity(expected.nodes[e].label, predicted.nodes[p].label)
    rows, cols = linear_sum_assignment(-matrix)
    pairs = []
    matched_e: set[str] = set()
    matched_p: set[str] = set()
    for i, j in zip(rows, cols, strict=True):
        if matrix[i, j] >= threshold:
            pairs.append(
                {"expected": exp_ids[i], "predicted": pred_ids[j], "similarity": round(float(matrix[i, j]), 1)}
            )
            matched_e.add(exp_ids[i])
            matched_p.add(pred_ids[j])
    return {
        "pairs": pairs,
        "unmatched_expected": [e for e in exp_ids if e not in matched_e],
        "unmatched_predicted": [p for p in pred_ids if p not in matched_p],
    }


def _edge_key(source: str, target: str, directed: bool) -> tuple[str, str]:
    return (source, target) if directed else tuple(sorted((source, target)))  # type: ignore[return-value]


def score_edges(
    expected: Graph,
    predicted: Graph,
    pairs: list[dict[str, Any]],
    label_threshold: int,
) -> dict[str, Any]:
    """Edge recall / precision / F1 over the aligned nodes.

    A reference edge is found when the predicted graph links the two aligned nodes: same
    direction when the reference is directed (a predicted undirected or bidirectional link
    also counts), any direction otherwise. When the reference edge carries text and the
    predicted one too, the texts must fuzzy-match; a predicted edge with no text is accepted
    (LLMs often fold "Yes"/"No" into the target node). Predicted edges between two aligned
    nodes that match no reference edge, and edges touching an unaligned node, are extras.
    """
    e2p = {p["expected"]: p["predicted"] for p in pairs}
    pred_edges = list(predicted.edges)
    used = [False] * len(pred_edges)
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for ref in expected.edges:
        s, t = e2p.get(ref.source), e2p.get(ref.target)
        found = None
        if s is not None and t is not None:
            for k, pe in enumerate(pred_edges):
                if used[k]:
                    continue
                forward = pe.source == s and pe.target == t
                backward = pe.source == t and pe.target == s
                if not (forward or backward):
                    continue
                if ref.directed and not ref.bidirectional and backward and pe.directed and not pe.bidirectional:
                    continue
                if ref.label and pe.label and label_similarity(ref.label, pe.label) < label_threshold:
                    continue
                found = k
                break
        entry = {
            "from": ref.source,
            "to": ref.target,
            "label": ref.label,
            "directed": ref.directed,
            "from_label": expected.nodes[ref.source].label,
            "to_label": expected.nodes[ref.target].label,
        }
        if found is None:
            missing.append(entry)
        else:
            used[found] = True
            matched.append(
                {
                    **entry,
                    "predicted": {
                        "from": pred_edges[found].source,
                        "to": pred_edges[found].target,
                        "label": pred_edges[found].label,
                    },
                }
            )
    extra = [
        {
            "from": pe.source,
            "to": pe.target,
            "label": pe.label,
            "from_label": predicted.nodes[pe.source].label,
            "to_label": predicted.nodes[pe.target].label,
        }
        for k, pe in enumerate(pred_edges)
        if not used[k]
    ]
    n_exp, n_pred = len(expected.edges), len(pred_edges)
    recall = len(matched) / n_exp if n_exp else 1.0
    precision = len(matched) / n_pred if n_pred else (1.0 if not n_exp else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "matched": matched,
        "missing": missing,
        "extra": extra,
    }


def find_node(graph: Graph, label: str, threshold: int) -> str | None:
    """Best node id for a label, or None when nothing reaches the threshold."""
    best_id, best = None, 0.0
    for node_id, node in graph.nodes.items():
        s = label_similarity(label, node.label)
        if s > best:
            best_id, best = node_id, s
    return best_id if best >= threshold else None


def has_edge(graph: Graph, source: str, target: str, directed: bool, label: str | None, label_threshold: int) -> bool:
    for e in graph.edges:
        forward = e.source == source and e.target == target
        backward = e.source == target and e.target == source
        if not (forward or backward):
            continue
        if directed and backward and e.directed and not e.bidirectional:
            continue
        if label and e.label and label_similarity(label, e.label) < label_threshold:
            continue
        return True
    return False


def has_path(graph: Graph, source: str, target: str, directed: bool) -> bool:
    adj: dict[str, set[str]] = {n: set() for n in graph.nodes}
    for e in graph.edges:
        adj.setdefault(e.source, set()).add(e.target)
        if not directed or not e.directed or e.bidirectional:
            adj.setdefault(e.target, set()).add(e.source)
    seen = {source}
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        if cur == target:
            return True
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def placement(page_md: str, offset: int, text_before: list[str], text_after: list[str]) -> dict[str, Any]:
    """The block must come after the last ``text_before`` line and before the first ``text_after`` line."""
    lowered = page_md.lower()

    def find(line: str) -> int | None:
        needle = re.sub(r"\s+", " ", line.strip().lower())
        if len(needle) < 4:
            return None
        idx = lowered.find(needle)
        if idx >= 0:
            return idx
        best, best_idx = 0.0, None
        step = max(1, len(needle) // 4)
        for i in range(0, max(1, len(lowered) - len(needle)), step):
            r = SequenceMatcher(None, needle, lowered[i : i + len(needle)]).ratio()
            if r > best:
                best, best_idx = r, i
        return best_idx if best >= 0.8 else None

    before_idx = [i for i in (find(t) for t in text_before) if i is not None]
    after_idx = [i for i in (find(t) for t in text_after) if i is not None]
    if not before_idx and not after_idx:
        return {"passed": None, "score": None, "reason": "anchors not found in markdown"}
    ok = not (before_idx and offset < max(before_idx)) and not (after_idx and offset > min(after_idx))
    return {
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "block_offset": offset,
        "before_offset": max(before_idx) if before_idx else None,
        "after_offset": min(after_idx) if after_idx else None,
    }


def node_leakage(page_md: str, expected: Graph, min_words: int = 2) -> dict[str, Any]:
    """Share of multi-word node labels that also appear as plain text outside any mermaid block.

    Informational in v0.1 (``passed`` is None): a caption or a legend may legitimately repeat a
    label, so the threshold is calibrated from data before it can fail a page.
    """
    stripped = re.sub(r"```.*?```", " ", page_md, flags=re.S)
    haystack = norm_label(stripped)
    labels = [n.label for n in expected.nodes.values() if len(n.label.split()) >= min_words]
    if not labels:
        return {"passed": None, "score": None, "reason": "no multi-word labels"}
    leaked = [lab for lab in labels if norm_label(lab) and norm_label(lab) in haystack]
    return {"passed": None, "score": None, "leaked": leaked, "ratio": round(len(leaked) / len(labels), 3)}


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


class DiagramGraphRule(ParseTestRule):
    """One expected diagram, scored as a graph against the best mermaid block on the page."""

    def __init__(self, rule_data: ParseDiagramGraphRule | dict):
        super().__init__(rule_data)
        if self.type != TestType.DIAGRAM_GRAPH.value:
            raise ValueError(f"Invalid type for DiagramGraphRule: {self.type}")
        self.rule = cast(ParseDiagramGraphRule, self._rule_data)
        self.expected = graph_from_dict(self.rule.graph)
        self.raw_output: dict[str, Any] | None = None
        self.source_file_path: str | None = None
        self.test_case_path: str | None = None

    def _reference_path(self) -> Path | None:
        if not self.rule.reference_image:
            return None
        for p in (self.test_case_path, self.source_file_path):
            if p:
                cand = Path(p).parent / self.rule.reference_image
                if cand.exists():
                    return cand
        return None

    def _best_block(self, blocks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """The block whose node alignment covers most of the reference (ties: most similar)."""
        best: tuple[dict[str, Any], dict[str, Any]] | None = None
        best_key = (-1, -1.0)
        for b in blocks:
            g: Graph = b["graph"]
            if g.is_empty:
                continue
            al = align_nodes(self.expected, g, self.rule.node_match_threshold)
            key = (len(al["pairs"]), sum(p["similarity"] for p in al["pairs"]))
            if key > best_key:
                best_key, best = key, (b, al)
        return best

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        page_md = _page_markdown(md_content, self.page)
        blocks = page_blocks(md_content, self.page)
        ref_path = self._reference_path()
        details: dict[str, Any] = {
            "expected": {
                "graph": self.expected.to_dict(),
                "diagram_class": self.rule.diagram_class,
                "accepted_types": self.rule.accepted_types,
                "bbox": self.rule.bbox,
                "caption": self.rule.caption,
                "reference_image": self.rule.reference_image,
                "text_before": self.rule.text_before,
                "text_after": self.rule.text_after,
                "mermaid": graph_to_mermaid(self.expected),
            },
            "blocks_on_page": [
                {
                    "type": b["graph"].type,
                    "nodes": len(b["graph"].nodes),
                    "edges": len(b["graph"].edges),
                    "errors": b["graph"].errors[:5],
                }
                for b in blocks
            ],
            "axes": {},
        }
        if ref_path is not None:
            from parse_bench.evaluation.metrics.parse.rules_image import thumbnail_data_uri

            details["expected"]["thumb"] = thumbnail_data_uri(ref_path)
        self.result_details = details
        axes = details["axes"]

        if not blocks:
            axes["present"] = {"passed": False, "score": 0.0, "reason": "no mermaid block on page"}
            return False, "No mermaid block on the page for the expected diagram", 0.0
        best = self._best_block(blocks)
        if best is None:
            axes["present"] = {
                "passed": False,
                "score": 0.0,
                "reason": "mermaid block(s) present but none parses into a graph",
            }
            details["predicted"] = {"source": blocks[0]["source"], "errors": blocks[0]["graph"].errors[:10]}
            return False, "Mermaid block does not parse into any node", 0.0
        block, alignment = best
        graph: Graph = block["graph"]
        details["predicted"] = {"graph": graph.to_dict(), "source": block["source"], "offset": block["offset"]}
        details["alignment"] = alignment
        axes["present"] = {"passed": True, "score": 1.0}

        accepted = {_canonical_type(t) for t in self.rule.accepted_types}
        type_ok = _canonical_type(graph.type) in accepted
        axes["type"] = {
            "passed": type_ok,
            "score": 1.0 if type_ok else 0.0,
            "predicted": graph.type,
            "accepted": sorted(accepted),
        }

        n_exp, n_pred, n_match = len(self.expected.nodes), len(graph.nodes), len(alignment["pairs"])
        recall = n_match / n_exp if n_exp else 1.0
        precision = n_match / n_pred if n_pred else 0.0
        node_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        nodes_ok = recall >= self.rule.node_recall_threshold and precision >= self.rule.node_precision_threshold
        axes["nodes"] = {
            "passed": nodes_ok,
            "score": round(node_f1, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "matched": n_match,
            "expected": n_exp,
            "predicted": n_pred,
        }

        edges = score_edges(self.expected, graph, alignment["pairs"], self.rule.edge_label_threshold)
        axes["edges"] = {"passed": edges["f1"] >= self.rule.edge_f1_threshold, "score": edges["f1"], **edges}

        axes["placement"] = placement(page_md, block["offset"], self.rule.text_before, self.rule.text_after)
        details["leakage"] = node_leakage(page_md, self.expected)

        applicable = [a for a in axes.values() if a.get("passed") is not None]
        passed = all(a["passed"] for a in applicable)
        score = float(np.mean([a["score"] for a in applicable])) if applicable else 0.0
        failed = [k for k, a in axes.items() if a.get("passed") is False]
        expl = f"{graph.type}: nodes {n_match}/{n_exp} (P {precision:.2f}), edges F1 {edges['f1']:.2f}" + (
            f"; failed: {', '.join(failed)}" if failed else "; all axes pass"
        )
        return passed, expl, round(score, 4)


class DiagramEdgeRule(ParseTestRule):
    """One anchor relation between two labelled nodes, checked in any mermaid block on the page."""

    def __init__(self, rule_data: ParseDiagramEdgeRule | dict):
        super().__init__(rule_data)
        if self.type != TestType.DIAGRAM_EDGE.value:
            raise ValueError(f"Invalid type for DiagramEdgeRule: {self.type}")
        self.rule = cast(ParseDiagramEdgeRule, self._rule_data)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        blocks = [b for b in page_blocks(md_content, self.page) if not b["graph"].is_empty]
        r = self.rule
        self.result_details = {
            "source": r.source,
            "target": r.target,
            "relation": r.relation,
            "label": r.label,
            "checks": [],
        }
        if not blocks:
            if r.relation == "no_edge":
                return False, "No mermaid block on the page (no_edge needs a diagram to hold)", 0.0
            return False, "No parseable mermaid block on the page", 0.0
        for b in blocks:
            g: Graph = b["graph"]
            s = find_node(g, r.source, r.node_match_threshold)
            t = find_node(g, r.target, r.node_match_threshold)
            check: dict[str, Any] = {
                "source_node": g.nodes[s].label if s else None,
                "target_node": g.nodes[t].label if t else None,
            }
            self.result_details["checks"].append(check)
            if s is None or t is None:
                check["result"] = "node(s) not found"
                if r.relation == "no_edge" and (s is None) != (t is None):
                    # One endpoint exists, the other does not: there is certainly no edge between them.
                    return True, f"no_edge holds: `{r.source if s is None else r.target}` absent", 1.0
                continue
            if r.relation == "edge":
                ok = has_edge(g, s, t, r.directed, r.label, r.edge_label_threshold)
            elif r.relation == "path":
                ok = has_path(g, s, t, r.directed)
            else:
                ok = not has_edge(g, s, t, r.directed, None, r.edge_label_threshold)
            check["result"] = "holds" if ok else "violated"
            if ok:
                return True, f"{r.relation} `{r.source}` → `{r.target}` holds", 1.0
        last = self.result_details["checks"][-1]
        if last.get("result") == "node(s) not found":
            missing = [
                lab for lab, node in ((r.source, last["source_node"]), (r.target, last["target_node"])) if node is None
            ]
            return False, f"node(s) not found in any diagram: {', '.join(f'`{m}`' for m in missing)}", 0.0
        return False, f"{r.relation} `{r.source}` → `{r.target}` violated", 0.0


class DiagramCountRule(ParseTestRule):
    """Page-level count of parseable mermaid blocks (0 on negative pages)."""

    def __init__(self, rule_data: ParseDiagramCountRule | dict):
        super().__init__(rule_data)
        if self.type != TestType.DIAGRAM_COUNT.value:
            raise ValueError(f"Invalid type for DiagramCountRule: {self.type}")
        self.rule = cast(ParseDiagramCountRule, self._rule_data)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        blocks = page_blocks(md_content, self.page)
        parseable = [b for b in blocks if not b["graph"].is_empty]
        broken = [b for b in blocks if b["graph"].is_empty]
        expected = max(0, int(self.rule.expected_count))
        self.result_details = {
            "expected_count": expected,
            "strict": self.rule.strict,
            "blocks": [
                {
                    "type": b["graph"].type,
                    "nodes": len(b["graph"].nodes),
                    "edges": len(b["graph"].edges),
                    "parseable": not b["graph"].is_empty,
                    "errors": b["graph"].errors[:5],
                }
                for b in blocks
            ],
            "parseable": len(parseable),
            "broken": len(broken),
        }
        problems: list[str] = []
        if broken:
            problems.append(f"{len(broken)} mermaid block(s) do not parse")
        extra = len(parseable) - expected
        if expected == 0 and parseable:
            problems.append(f"{len(parseable)} mermaid block(s) on a page that should have none")
        elif self.rule.strict and extra > 0:
            problems.append(f"{extra} more mermaid block(s) than the {expected} expected")
        if problems:
            offending = len(broken) + (max(0, extra) if (self.rule.strict or expected == 0) else 0)
            score = max(0.0, 1.0 - offending / max(1, expected + offending))
            return False, "; ".join(problems), round(score, 4)
        return True, f"{len(parseable)} parseable mermaid block(s) for {expected} expected", 1.0


# ---------------------------------------------------------------------------
# rendering the reference for reports
# ---------------------------------------------------------------------------


def _mermaid_escape(label: str) -> str:
    return '"' + label.replace('"', "#quot;") + '"'


def graph_to_mermaid(graph: Graph) -> str:
    """Render a reference graph as a flowchart so the report can draw GT next to the prediction."""
    ids = {node_id: f"n{i}" for i, node_id in enumerate(graph.nodes)}
    lines = ["flowchart TD"]
    by_group: dict[str | None, list[str]] = {}
    for node_id, node in graph.nodes.items():
        by_group.setdefault(node.group, []).append(f"{ids[node_id]}[{_mermaid_escape(node.label)}]")
    for group, decls in by_group.items():
        if group is None:
            lines.extend(f"    {d}" for d in decls)
        else:
            lines.append(f"    subgraph {_mermaid_escape(graph.groups.get(group, group))}")
            lines.extend(f"        {d}" for d in decls)
            lines.append("    end")
    for e in graph.edges:
        arrow = "<-->" if e.bidirectional else ("-->" if e.directed else "---")
        text = f"|{_mermaid_escape(e.label)}|" if e.label else ""
        lines.append(f"    {ids[e.source]} {arrow}{text} {ids[e.target]}")
    return "\n".join(lines)
