"""text_next: consecutive <sup>/<sub> tags are separate marks, not one glued identifier."""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.text_next import score_text_next, tokenize

AUTHORS = (
    "Ana Maria Moreau<sup>1</sup>, Omar Karimi<sup>2</sup>, Nikos Petrakis<sup>3,2</sup>, "
    "Paul Grant<sup>7</sup> and Nora Bell<sup>8</sup> wrote the report on sample screening methods."
)


def keys(md: str) -> list[str]:
    return [t.key for t in tokenize(md)[0]]


def test_second_sup_tag_starts_a_new_token():
    assert keys("Moreau<sup>1</sup><sup>ID</sup> and") == ["moreau", "#1", "id", "and"]
    assert keys("Moreau<sup>1</sup>") == ["moreau", "#1"]  # a single glued footnote still splits off


def test_icon_sup_after_each_footnote_costs_only_inserted_words():
    hyp = AUTHORS.replace("</sup>,", "</sup><sup>ID</sup>,").replace("</sup> and", "</sup><sup>ID</sup> and")
    assert score_text_next(AUTHORS, AUTHORS).score == 1.0
    res = score_text_next(AUTHORS, hyp)
    assert res.events.critical == 0
    assert res.score > 0.8


def test_wrong_footnote_number_in_adjacent_sup_is_still_critical():
    hyp = AUTHORS.replace("Moreau<sup>1</sup>", "Moreau<sup>4</sup><sup>ID</sup>")
    assert score_text_next(AUTHORS, hyp).events.critical >= 1
