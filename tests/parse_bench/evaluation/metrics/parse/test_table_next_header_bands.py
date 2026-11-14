"""table_next on a reference that stacks sub-tables in one table, each under its own copy of the header band."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.table_next import score_table_next

GROUPS = [
    ("Inpatients (n = 40)", "Baseline", "6 months", [["Mobility", "38.7", "49.5", "44.1", "51.2"]]),
    ("Discharged (n = 30)", "Basal", "2 months", [["Mobility", "46.9", "65.3", "69.8", "72.0"]]),
]
LABELS = ["Cognition", "Daily living"]


def _bodies() -> list[list[list[str]]]:
    out = []
    for k, (_, _, _, rows) in enumerate(GROUPS):
        extra = [[lab, *(f"{k}.{i}{j}" for j in range(4))] for i, lab in enumerate(LABELS)]
        out.append(rows + extra)
    return out


def _band(name: str, a: str, b: str, *, empty_corner: bool, tag: str) -> str:
    """Header band: group name, two periods, Int/Ctl under each period."""
    o, c = f"<{tag}>", f"</{tag}>"
    first = f"<tr>{o}{c}<{tag} colspan='4'>{name}{c}</tr>" if empty_corner else f"<tr><{tag} colspan='5'>{name}{c}</tr>"
    return (
        first
        + f"<tr>{o}{c}<{tag} colspan='2'>{a}{c}<{tag} colspan='2'>{b}{c}</tr>"
        + f"<tr>{o}{c}{o}Int{c}{o}Ctl{c}{o}Int{c}{o}Ctl{c}</tr>"
    )


def _rows(body: list[list[str]]) -> str:
    return "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in body)


def _reference() -> str:
    """One table, the second group's header band restated mid-body (td cells, empty top-left corner)."""
    parts = [
        _band(n, a, b, empty_corner=True, tag="td") + _rows(body)
        for (n, a, b, _), body in zip(GROUPS, _bodies(), strict=True)
    ]
    return "<table>" + "".join(parts) + "</table>"


def _one_table_per_group(bodies: list[list[list[str]]]) -> str:
    return "\n\n".join(
        "<table>" + _band(n, a, b, empty_corner=False, tag="th") + _rows(body) + "</table>"
        for (n, a, b, _), body in zip(GROUPS, bodies, strict=True)
    )


def test_reference_with_header_bands_scores_itself_one() -> None:
    ref = _reference()
    assert score_table_next(ref, ref).score == 1.0


def test_groups_emitted_as_separate_tables_match_the_bands() -> None:
    res = score_table_next(_reference(), _one_table_per_group(_bodies()))
    assert res.score == 1.0
    assert res.events.critical == 0


def test_bands_kept_in_one_table_with_full_width_names_match() -> None:
    out = (
        "<table>"
        + "".join(
            _band(n, a, b, empty_corner=False, tag="th") + _rows(body)
            for (n, a, b, _), body in zip(GROUPS, _bodies(), strict=True)
        )
        + "</table>"
    )
    assert score_table_next(_reference(), out).score == 1.0


def test_values_swapped_between_groups_still_lose() -> None:
    bodies = _bodies()
    bodies[0], bodies[1] = bodies[1], bodies[0]
    # the swapped groups read as two moved tables (row order 0.5); the floor charges a moved table once
    assert score_table_next(_reference(), _one_table_per_group(bodies)).score < 0.7
