"""Mermaid syntax must not become TAL content, without hiding wrong diagram labels."""

import pytest

from parse_bench.evaluation.metrics.parse.mermaid_text import project_mermaid_text
from parse_bench.evaluation.metrics.parse.text_next import score_text_next


def test_mermaid_ids_and_repeated_edge_endpoints_do_not_change_text_score() -> None:
    reference = (
        "The following system handles incoming requests.\n\n"
        "![Processing stack: User Interface, Contract Functionality, Notification, Recording Ledger.](image)"
    )
    scores = []
    for ui, cf, notification, ledger in [("UI", "CF", "NO", "RL"), ("node1", "node2", "node3", "node4")]:
        diagram = f"""```mermaid
flowchart TB
subgraph S ["Processing stack"]
{ui}["User Interface"]
{cf}["Contract Functionality"]
end
{ui} --- {cf}
{cf} --- {notification}["Notification"]
{cf} --- {ledger}["Recording Ledger"]
```
"""
        output = "The following system handles incoming requests.\n\n" + diagram
        scores.append(score_text_next(reference, output).score)
    assert scores == [1.0, 1.0]


@pytest.mark.parametrize(
    "mutated",
    [
        'A["Amount 150"] -->|approved| B["Payment pending"]',
        'A["Amount 100"] -->|rejected| B["Payment pending"]',
        'A["Amount 100"] -->|approved| B["Payment not pending"]',
        'A["Amount 100"] -->|approved| B["Payment pending"]\nC["Invented charge 999 dollars"]',
        'A["Amount 100"]',
    ],
)
def test_wrong_missing_and_invented_mermaid_labels_remain_penalized(mutated: str) -> None:
    reference = '```mermaid\nflowchart LR\nA["Amount 100"] -->|approved| B["Payment pending"]\n```'
    assert score_text_next(reference, reference).score == 1.0
    assert score_text_next(reference, "```mermaid\nflowchart LR\n" + mutated + "\n```").score < 1.0


def test_bare_nodes_retain_visible_names_and_distinct_equal_labels() -> None:
    diagram = '```mermaid\ngraph LR\nStart --> A["Review"]\nA --> B["Review"]\nB --> End\n```'
    assert project_mermaid_text(diagram).split() == ["Start", "Review", "Review", "End"]
    assert score_text_next("Start Review Review End", diagram).score == 1.0


@pytest.mark.parametrize(
    "markdown",
    [
        '```python\nprint("mermaid")\n```',
        '````text\n```mermaid\nflowchart LR\nA["Visible label"]\n```\n````',
        '```mermaid\nflowchart LR\nA["Broken label"\n```',
        '```mermaid\nflowchart LR\nA["Visible label"]\nthis is not a diagram statement\n```',
        '```mermaid\nflowchart LR\nA["Visible label"]',
        "```mermaid\nsequenceDiagram\nparticipant Alice\nNote over Alice: Important visible note\n```",
        '```mermaid\n---\ntitle: Visible title\n---\nflowchart LR\nA["Visible label"]\n```',
        "```mermaid\nflowchart LR\n```",
    ],
)
def test_unsupported_or_partial_projection_does_not_silently_drop_content(markdown: str) -> None:
    assert project_mermaid_text(markdown) == markdown


def test_multiple_fences_keep_surrounding_text_and_ignore_style_identifiers() -> None:
    markdown = """Before the first diagram.
~~~mermaid
flowchart LR
A["First label"] -->|Proceed| B["Second label"]
style A fill:#123456
~~~
Between the diagrams.
```mermaid
flowchart TB
C["Third label"]
````
After the last diagram.
"""
    assert (
        project_mermaid_text(markdown).split()
        == (
            "Before the first diagram. First label Second label Proceed Between the diagrams. "
            "Third label After the last diagram."
        ).split()
    )


def test_forward_declared_underscore_ids_resolve_to_their_visible_labels() -> None:
    diagram = "\n".join(
        [
            "```MERMAID",
            "flowchart LR",
            "node_one --> node_two",
            'node_one["Start here"]',
            'node_two["Finish here"]',
            "```",
        ]
    )
    assert score_text_next("Start here Finish here", diagram).score == 1.0


@pytest.mark.parametrize(
    "body",
    [
        'A["Amount 100"]\nA["Amount 999"]',
        'A["Amount 100"]\nend Invented 999',
        'subgraph S["Amount 100"]\nA["Amount 100"]',
        'A["Amount 100"]\nend',
    ],
)
def test_conflicting_labels_and_unbalanced_groups_preserve_source(body: str) -> None:
    markdown = "```mermaid\nflowchart LR\n" + body + "\n```"
    assert project_mermaid_text(markdown) == markdown
