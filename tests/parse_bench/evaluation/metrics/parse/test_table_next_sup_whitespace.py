"""table_next: whitespace around a <sup>/<sub> marker in a pretty-printed reference cell."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import parse_tables, score_table_next

VALUES = [["14/33 (42)", "ab"], ["9/33 (27)", "a"], ["8/14 (57)", "b"], ["3/20 (15)", "ab"]]


def _table(cells: list[str]) -> str:
    rows = "".join(f"<tr><td>Group {i}</td><td>{c}</td></tr>" for i, c in enumerate(cells))
    return f"<table><tr><td></td><td>Rate</td></tr>{rows}</table>"


# Reference cells pretty-printed with line breaks and indentation around the marker.
REF = _table([f"\n      {v}\n      <sup>\n       {m}\n      </sup>\n" for v, m in VALUES])


def test_pretty_printed_sup_reads_as_attached_marker() -> None:
    cells = [c.text for c in parse_tables(REF)[0].cells]
    assert "14/33 (42)ab" in cells


def test_inline_sup_and_unicode_superscript_outputs_score_full_credit() -> None:
    assert score_table_next(REF, REF).score == 1.0
    assert score_table_next(REF, _table([f"{v}<sup>{m}</sup>" for v, m in VALUES])).score == 1.0
    sup = str.maketrans("ab", "ᵃᵇ")
    assert score_table_next(REF, _table([f"{v}{m.translate(sup)}" for v, m in VALUES])).score == 1.0


def test_wrong_marker_still_costs() -> None:
    wrong = _table([f"{v}<sup>c</sup>" for v, _ in VALUES])
    assert score_table_next(REF, wrong).score < 0.9
