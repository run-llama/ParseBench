"""Diagram (mermaid) rules: parser, alignment, and the three rule types."""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.mermaid_graph import (
    extract_mermaid_blocks,
    graph_from_dict,
    parse_mermaid,
)
from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.evaluation.metrics.parse.rules_diagram import (
    DiagramCountRule,
    DiagramEdgeRule,
    DiagramGraphRule,
    align_nodes,
    graph_to_mermaid,
    label_similarity,
    score_edges,
)

GT_GRAPH = {
    "nodes": [
        {"id": "a", "label": "Receive application"},
        {"id": "b", "label": "Application complete?"},
        {"id": "c", "label": "Approve"},
        {"id": "d", "label": "Return to applicant"},
    ],
    "edges": [
        {"from": "a", "to": "b"},
        {"from": "b", "to": "c", "label": "Yes"},
        {"from": "b", "to": "d", "label": "No"},
        {"from": "d", "to": "a"},
    ],
}

# Same diagram, transcribed "creatively": other ids, LR, different shapes, inline edge text,
# a <br/> in a label, a subgraph, and a shortened label.
CREATIVE_MD = """# Process

Figure 2 shows the intake process.

```mermaid
graph LR
    start([Receive<br/>application]) --> q{Is the application complete?}
    q -- Yes --> ok[Approve]
    q -->|No| back[Return to applicant]
    subgraph Loop
    back -.-> start
    end
```

Once approved, the file is archived.
"""


def _graph_rule(**overrides):
    data = {
        "type": "diagram_graph",
        "page": 1,
        "id": "d1",
        "graph": GT_GRAPH,
        "accepted_types": ["flowchart"],
        "text_before": ["Figure 2 shows the intake process."],
        "text_after": ["Once approved, the file is archived."],
    }
    data.update(overrides)
    return data


# --- parser ------------------------------------------------------------------------------


def test_parser_reads_every_flowchart_edge_form() -> None:
    src = """flowchart TD
    A[Start] --> B{Ok?}
    B -- Yes --> C([Go])
    B -->|No| D[Stop<br/>now]
    C ==> E((End)) & F[Log]
    D -.-> A
    G["Quoted `label`"] --- H
    A ~~~ H
    classDef x fill:#f00
    """
    g = parse_mermaid(src)
    assert g.type == "flowchart" and g.direction == "TD"
    assert {n.label for n in g.nodes.values()} == {"Start", "Ok?", "Go", "Stop now", "End", "Log", "Quoted label", "H"}
    edges = {(e.source, e.target, e.label, e.directed) for e in g.edges}
    assert ("B", "C", "Yes", True) in edges
    assert ("B", "D", "No", True) in edges
    assert ("C", "E", "", True) in edges and ("C", "F", "", True) in edges
    assert ("G", "H", "", False) in edges
    assert not any(e.source == "A" and e.target == "H" for e in g.edges), "invisible links are not edges"
    assert g.errors == []


def test_parser_handles_compact_syntax_and_subgraphs() -> None:
    g = parse_mermaid("graph LR; a-->b; b-->c\nsubgraph Ops [Operations]\n  c --> d-1[Done]\nend")
    assert [n.id for n in g.nodes.values()] == ["a", "b", "c", "d-1"]
    assert g.nodes["d-1"].label == "Done" and g.nodes["d-1"].group == "Ops"
    assert g.groups == {"Ops": "Operations"}


@pytest.mark.parametrize(
    ("src", "kind", "n_nodes", "n_edges"),
    [
        ("stateDiagram-v2\n[*] --> Idle\nIdle --> Running : start\nRunning --> [*]", "state", 4, 3),
        ("sequenceDiagram\nparticipant U as User\nU->>S: Login\nS-->>U: Token", "sequence", 2, 2),
        ("mindmap\n  root((Plan))\n    A\n      A1\n    B", "mindmap", 4, 3),
        ("classDiagram\nAnimal <|-- Duck\nAnimal --> Food : eats", "class", 3, 2),
        ("erDiagram\nCUSTOMER ||--o{ ORDER : places", "er", 2, 1),
    ],
)
def test_parser_other_dialects(src: str, kind: str, n_nodes: int, n_edges: int) -> None:
    g = parse_mermaid(src)
    assert g.type == kind
    assert len(g.nodes) == n_nodes and len(g.edges) == n_edges


def test_unknown_header_is_unparseable() -> None:
    g = parse_mermaid("not a diagram\nA --> B")
    assert g.is_empty and g.errors


def test_extract_blocks_is_case_insensitive_and_ordered() -> None:
    md = "x\n```mermaid\nflowchart LR\nA-->B\n```\ny\n```Mermaid\ngraph TD\nX-->Y\n```\n"
    blocks = extract_mermaid_blocks(md)
    assert [b["source"].splitlines()[0] for b in blocks] == ["flowchart LR", "graph TD"]
    assert blocks[0]["offset"] < blocks[1]["offset"]


# --- alignment ---------------------------------------------------------------------------


def test_label_similarity_tolerates_abbreviation_and_order() -> None:
    assert label_similarity("Receive application", "Receive<br/>application") == 100.0
    assert label_similarity("Customer submits application form", "Submit application") >= 80
    assert label_similarity("Approve", "Reject") < 50
    # line breaks dropped without a space by the parser
    assert label_similarity("Office of State Grand Jury Review", "Office ofState Grand JuryReview") >= 90
    assert label_similarity("End", "Send") < 80


def test_alignment_is_one_to_one_and_edges_score_on_aligned_ids() -> None:
    expected = graph_from_dict(GT_GRAPH)
    predicted = parse_mermaid(extract_mermaid_blocks(CREATIVE_MD)[0]["source"])
    al = align_nodes(expected, predicted, 80)
    assert len(al["pairs"]) == 4 and not al["unmatched_expected"]
    edges = score_edges(expected, predicted, al["pairs"], 70)
    assert edges["f1"] == 1.0, edges


def test_edge_scoring_reports_missing_and_extra() -> None:
    expected = graph_from_dict(GT_GRAPH)
    predicted = parse_mermaid(
        "flowchart TD\nA[Receive application] --> B{Application complete?}\nB --> C[Approve]\nC --> Z[Archive]"
    )
    al = align_nodes(expected, predicted, 80)
    edges = score_edges(expected, predicted, al["pairs"], 70)
    assert edges["recall"] == 0.5
    assert [e["to_label"] for e in edges["missing"]] == ["Return to applicant", "Receive application"]
    assert [e["to_label"] for e in edges["extra"]] == ["Archive"]


def test_wrong_direction_fails_directed_reference_edge() -> None:
    expected = graph_from_dict(
        {"nodes": [{"id": "a", "label": "CEO"}, {"id": "b", "label": "CFO"}], "edges": [{"from": "a", "to": "b"}]}
    )
    predicted = parse_mermaid("flowchart TD\nCFO --> CEO")
    al = align_nodes(expected, predicted, 80)
    assert score_edges(expected, predicted, al["pairs"], 70)["f1"] == 0.0
    undirected = parse_mermaid("flowchart TD\nCFO --- CEO")
    al = align_nodes(expected, undirected, 80)
    assert score_edges(expected, undirected, al["pairs"], 70)["f1"] == 1.0


# --- diagram_graph -------------------------------------------------------------------------


def test_graph_rule_passes_creative_but_equivalent_transcription() -> None:
    rule = DiagramGraphRule(_graph_rule())
    passed, expl, score = rule.run(CREATIVE_MD)
    assert passed, expl
    assert score == 1.0
    axes = rule.result_details["axes"]
    assert axes["type"]["passed"] and axes["nodes"]["recall"] == 1.0 and axes["edges"]["f1"] == 1.0
    assert axes["placement"]["passed"] is True
    assert rule.result_details["expected"]["mermaid"].startswith("flowchart TD")


def test_graph_rule_fails_without_block_and_when_block_is_broken() -> None:
    rule = DiagramGraphRule(_graph_rule())
    passed, expl, score = rule.run("# Process\n\nJust prose.")
    assert not passed and score == 0.0 and "No mermaid block" in expl
    passed, expl, _ = rule.run("```mermaid\nsomething odd\n```")
    assert not passed and "does not parse" in expl


def test_graph_rule_flags_missing_nodes_and_edges() -> None:
    md = "```mermaid\nflowchart TD\n  A[Receive application] --> B{Application complete?}\n  B --> C[Approve]\n```"
    rule = DiagramGraphRule(_graph_rule())
    passed, expl, score = rule.run(md)
    assert not passed
    axes = rule.result_details["axes"]
    assert axes["nodes"]["recall"] == 0.75 and axes["nodes"]["passed"] is False
    # 2 of 4 reference edges with no extras: F1 0.67 clears the 0.6 edge threshold on its own;
    # the page still fails through the node axis.
    assert axes["edges"]["recall"] == 0.5 and axes["edges"]["precision"] == 1.0
    assert axes["edges"]["passed"] is True
    assert axes["placement"]["passed"] is None  # anchors absent from this markdown
    assert 0.0 < score < 1.0


def test_graph_rule_type_axis_uses_accepted_types() -> None:
    md = "```mermaid\nmindmap\n  root((Receive application))\n    Application complete?\n      Approve\n      Return to applicant\n```"  # noqa: E501
    strict = DiagramGraphRule(_graph_rule())
    passed, _, _ = strict.run(md)
    assert not passed and strict.result_details["axes"]["type"]["passed"] is False
    lenient = DiagramGraphRule(_graph_rule(accepted_types=["flowchart", "mindmap"]))
    lenient.run(md)
    assert lenient.result_details["axes"]["type"]["passed"] is True


def test_graph_rule_picks_the_best_block_among_several() -> None:
    md = "```mermaid\nflowchart LR\n  X[Other chart] --> Y[Stuff]\n```\n" + CREATIVE_MD
    rule = DiagramGraphRule(_graph_rule(text_before=[], text_after=[]))
    passed, _, _ = rule.run(md)
    assert passed
    assert rule.result_details["predicted"]["graph"]["nodes"][0]["label"] == "Receive application"


# --- diagram_edge --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "expected_pass"),
    [
        ({"source": "Application complete?", "target": "Approve", "relation": "edge", "label": "Yes"}, True),
        ({"source": "Application complete?", "target": "Approve", "relation": "edge", "label": "No"}, False),
        ({"source": "Approve", "target": "Application complete?", "relation": "edge"}, False),
        ({"source": "Approve", "target": "Application complete?", "relation": "edge", "directed": False}, True),
        ({"source": "Return to applicant", "target": "Approve", "relation": "path"}, True),
        ({"source": "Approve", "target": "Return to applicant", "relation": "path"}, False),
        ({"source": "Receive application", "target": "Approve", "relation": "no_edge"}, True),
        ({"source": "Application complete?", "target": "Approve", "relation": "no_edge"}, False),
        ({"source": "Nonexistent step", "target": "Approve", "relation": "edge"}, False),
        ({"source": "Nonexistent step", "target": "Approve", "relation": "no_edge"}, True),
    ],
)
def test_edge_rule_relations(rule: dict, expected_pass: bool) -> None:
    r = DiagramEdgeRule({"type": "diagram_edge", "page": 1, **rule})
    passed, expl, score = r.run(CREATIVE_MD)
    assert passed is expected_pass, expl
    assert score == (1.0 if expected_pass else 0.0)


def test_edge_rule_without_diagram_fails() -> None:
    r = DiagramEdgeRule({"type": "diagram_edge", "page": 1, "source": "A", "target": "B"})
    assert r.run("no diagram here")[0] is False


# --- diagram_count -------------------------------------------------------------------------


def test_count_rule_negative_page_and_broken_blocks() -> None:
    neg = DiagramCountRule({"type": "diagram_count", "page": 1, "expected_count": 0})
    assert neg.run("# Chart page\n\n| a | b |\n|---|---|\n| 1 | 2 |")[0] is True
    passed, expl, score = neg.run(CREATIVE_MD)
    assert not passed and "should have none" in expl and score == 0.0

    one = DiagramCountRule({"type": "diagram_count", "page": 1, "expected_count": 1})
    assert one.run(CREATIVE_MD) == (True, "1 parseable mermaid block(s) for 1 expected", 1.0)
    passed, expl, score = one.run(CREATIVE_MD + "\n```mermaid\nbroken\n```\n")
    assert not passed and "do not parse" in expl and score == 0.5


def test_count_rule_strict_flag() -> None:
    two = CREATIVE_MD + "\n```mermaid\nflowchart LR\nA-->B\n```\n"
    assert (
        DiagramCountRule({"type": "diagram_count", "page": 1, "expected_count": 1, "strict": True}).run(two)[0] is False
    )
    assert (
        DiagramCountRule({"type": "diagram_count", "page": 1, "expected_count": 1, "strict": False}).run(two)[0] is True
    )


# --- wiring --------------------------------------------------------------------------------


def test_registry_creates_diagram_rules_and_validates_graph() -> None:
    assert isinstance(create_test_rule(_graph_rule()), DiagramGraphRule)
    assert isinstance(
        create_test_rule({"type": "diagram_edge", "page": 1, "source": "A", "target": "B"}), DiagramEdgeRule
    )
    assert isinstance(create_test_rule({"type": "diagram_count", "page": 1, "expected_count": 1}), DiagramCountRule)
    with pytest.raises(Exception, match="unknown node id"):
        create_test_rule(
            _graph_rule(graph={"nodes": [{"id": "a", "label": "A"}], "edges": [{"from": "a", "to": "zz"}]})
        )


def test_graph_to_mermaid_round_trips_through_parser() -> None:
    expected = graph_from_dict({**GT_GRAPH, "groups": [{"id": "g", "label": "Back office", "members": ["c", "d"]}]})
    rendered = graph_to_mermaid(expected)
    again = parse_mermaid(rendered)
    assert {n.label for n in again.nodes.values()} == {n.label for n in expected.nodes.values()}
    assert len(again.edges) == len(expected.edges)
    assert again.groups == {"Back office": "Back office"}
