"""table_next: a row label that lost the space at a line break still matches its reference row."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import key_sim, score_table_next

REF = (
    "<table><tr><th>Product<br/>(active substance)</th><th>Meeting date</th><th>Change</th></tr>"
    "<tr><td>Zentrol<br/>(alphacide/betacide)</td><td>11-13 May 2020</td><td>Section 4.6 updated</td></tr>"
    "</table>"
)


def _out(label: str) -> str:
    """One value differs slightly, so the label-alias pass (most values equal) cannot rescue the row."""
    return (
        "<table><tr><th>Product(active substance)</th><th>Meeting date</th><th>Change</th></tr>"
        f"<tr><td>{label}</td><td>11-13 May 2020</td><td>Section 4.6 update</td></tr></table>"
    )


def test_label_joined_without_space_still_matches() -> None:
    assert key_sim("zentrol (alphacide/betacide)", "zentrol(alphacide/betacide)") == 0.9
    assert score_table_next(REF, _out("Zentrol(alphacide/betacide)")).score > 0.8
    assert score_table_next(REF, REF).score == 1.0


def test_different_label_still_scores_zero() -> None:
    assert key_sim("zentrol (alphacide)", "mirovax(alphacide)") == 0.0
    assert key_sim("item 1 2", "item 12") == 0.0  # digits regrouped are a different label
    assert score_table_next(REF, _out("Kortiva(gammacide)")).score == 0.0
