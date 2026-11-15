"""align() on oversized pages: the rectangular solve must reach the square dummy solve's optimum."""

import random
import tracemalloc

import pytest

from parse_bench.evaluation.metrics.parse.formnode_scoring import field_matching
from parse_bench.evaluation.metrics.parse.formnode_scoring.field_flatten import Field

WORDS = ["name", "date", "amount", "address", "city", "code", "total", "tax", "id", "phone"]


def _field(rng: random.Random, i: int) -> Field:
    label = " ".join(rng.sample(WORDS, rng.randint(1, 3))) if rng.random() < 0.8 else None
    value = rng.choice([None, "", "1", "2", "yes", f"v{rng.randint(0, 5)}"])
    fid = f"f{rng.randint(0, 8)}" if rng.random() < 0.3 else None
    kind = rng.choice(["text", "text", "checkbox"])
    return Field(page=1, path=(), parent=0, id=fid, label=label, kind=kind, value=value, options=[], flat_idx=i)


def _pages(seed: int, n: int, m: int) -> tuple[list[Field], list[Field]]:
    rng = random.Random(seed)
    return [_field(rng, i) for i in range(n)], [_field(rng, i) for i in range(m)]


def _matched_score(gt: list[Field], pred: list[Field]) -> float:
    n, m = len(gt), len(pred)
    return sum(field_matching.similarity(g, g.match, n, m) for g in gt if g.match is not None)


@pytest.mark.parametrize("seed", range(200))
def test_rectangular_solve_reaches_square_optimum(seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
    n, m = random.Random(seed).randint(1, 12), random.Random(seed + 1000).randint(1, 30)
    gt, pred = _pages(seed, n, m)
    field_matching.align(gt, pred)
    square_score, square_count = _matched_score(gt, pred), sum(g.match is not None for g in gt)

    gt2, pred2 = _pages(seed, n, m)
    monkeypatch.setattr(field_matching, "_MAX_SQUARE_CELLS", 0)
    field_matching.align(gt2, pred2)
    rect_score, rect_count = _matched_score(gt2, pred2), sum(g.match is not None for g in gt2)

    # the square cost is d*(n+m) - sum over matched pairs of (s - SIM_THRESHOLD): equal optimum
    threshold = field_matching.SIM_THRESHOLD
    assert square_score - threshold * square_count == pytest.approx(rect_score - threshold * rect_count, abs=1e-9)
    for g in gt2:
        if g.match is not None:
            assert field_matching.similarity(g, g.match, n, m) > threshold
            assert g.match.match is g


def test_a_prediction_with_many_fields_is_aligned_in_bounded_memory() -> None:
    gt, pred = _pages(7, 5, 6000)
    assert (len(gt) + len(pred)) ** 2 > field_matching._MAX_SQUARE_CELLS  # the square matrix would be ~290 MB
    tracemalloc.start()
    try:
        field_matching.align(gt, pred)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 50_000_000
    matched = [g for g in gt if g.match is not None]
    assert matched and all(g.match.match is g for g in matched)
