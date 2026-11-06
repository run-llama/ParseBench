"""table_next on a long list printed in side-by-side column groups (an index of names and page numbers)."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import score_table_next

ROWS = [(f"Name{i:02d}", str(10 + 7 * i)) for i in range(12)]


def _html(rows: list[list[str]]) -> str:
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{body}</table>"


def _wrapped(rows: list[tuple[str, str]], groups: int) -> str:
    """The list read top to bottom, then left to right, as the page prints it: one row per printed line."""
    n = -(-len(rows) // groups)
    cols = [rows[b * n : (b + 1) * n] for b in range(groups)]
    return _html([[c for col in cols for c in (col[i] if i < len(col) else ("", ""))] for i in range(n)])


REF = _html([list(r) for r in ROWS])


def test_list_wrapped_into_column_groups_reads_group_after_group() -> None:
    res = score_table_next(REF, _wrapped(ROWS, 3))
    assert 0.85 <= res.score < 1.0
    assert res.content == 1.0 and res.row_order == 1.0
    assert any(kind == "column groups wrapped" for kind, *_ in res.events.items)


def test_wrapped_groups_score_like_the_same_list_split_into_tables() -> None:
    n = len(ROWS) // 3
    split = "\n\n".join(_html([list(r) for r in ROWS[b * n : (b + 1) * n]]) for b in range(3))
    assert abs(score_table_next(REF, _wrapped(ROWS, 3)).score - score_table_next(REF, split).score) < 0.02


def test_wrapped_groups_with_wrong_values_still_lose_content() -> None:
    bad = [(name, str(int(page) + 1)) for name, page in ROWS]
    res = score_table_next(REF, _wrapped(bad, 3))
    assert res.score < 0.1


def test_reference_width_output_is_not_unwrapped() -> None:
    res = score_table_next(REF, REF)
    assert res.score == 1.0
    assert not any(kind == "column groups wrapped" for kind, *_ in res.events.items)
