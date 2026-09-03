"""Tests for relaxed_normalize_cell_text() — the relaxed-only cell-text folds
(LI-8223 items 2 & 3): tag-strip separator injection and homoglyph/equivalence
canonicalization.

These folds run ONLY on the relaxed metric path (via ``normalize_for_relaxed``)
and compose with the strict ``normalize_cell_text`` that runs on top.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.utils import (
    normalize_cell_text,
    relaxed_normalize_cell_text,
)


class TestTagStripSeparatorInjection:
    """Block/line tags become separators so list items don't concatenate."""

    def test_ul_li_list_does_not_fuse(self) -> None:
        # Without separator injection get_text-style stripping would fuse to "AB".
        assert relaxed_normalize_cell_text("<ul><li>A</li><li>B</li></ul>") == "• A • B"

    def test_ul_li_matches_literal_bullets(self) -> None:
        # Motivating case Low-Back-Guideline (1)_page2: a <ul><li> prediction and
        # a literal ■-bulleted GT cell must converge to the same shape.
        pred = relaxed_normalize_cell_text("<ul><li>A</li><li>B</li></ul>")
        gt = relaxed_normalize_cell_text("■ A<br>■ B")
        assert pred == gt == "• A • B"

    def test_br_becomes_space(self) -> None:
        assert relaxed_normalize_cell_text("line1<br>line2") == "line1 line2"

    def test_paragraph_tags_become_space(self) -> None:
        assert relaxed_normalize_cell_text("<p>one</p><p>two</p>") == "one two"

    def test_ordered_list(self) -> None:
        assert relaxed_normalize_cell_text("<ol><li>A</li><li>B</li></ol>") == "• A • B"

    def test_plain_text_unchanged(self) -> None:
        assert relaxed_normalize_cell_text("plain value") == "plain value"


class TestHomoglyphFolding:
    """Visually-identical codepoints fold to a canonical form."""

    def test_micro_sign_to_greek_mu(self) -> None:
        # µ U+00B5 MICRO SIGN vs μ U+03BC GREEK SMALL LETTER MU
        assert relaxed_normalize_cell_text("5 µm") == relaxed_normalize_cell_text("5 μm") == "5 μm"

    def test_non_breaking_hyphen(self) -> None:
        assert relaxed_normalize_cell_text("A‑B") == "A-B"

    def test_figure_dash(self) -> None:
        assert relaxed_normalize_cell_text("A‒B") == "A-B"

    def test_en_dash(self) -> None:
        assert relaxed_normalize_cell_text("A–B") == "A-B"

    def test_nbsp_to_space(self) -> None:
        assert relaxed_normalize_cell_text("A B") == "A B"

    def test_square_bullets_fold_to_canonical(self) -> None:
        # Only the SQUARE bullets fold at the relaxed layer (strict already folds
        # circle bullets in mixed-content cells).
        canonical = relaxed_normalize_cell_text("■ item")
        for bullet in ("▪", "•"):
            assert relaxed_normalize_cell_text(f"{bullet} item") == canonical == "• item"

    def test_circle_markers_not_folded_boolean_distinction_preserved(self) -> None:
        # Whole-cell ● (yes) vs ○ (no) is a REAL content distinction: the strict
        # boolean-marker pass must still see them unchanged after relaxed runs.
        # (Folding them to • would make checked-vs-unchecked score as a match,
        # since • is itself a truthy marker glyph.)
        for glyph in ("●", "○", "◦"):
            assert relaxed_normalize_cell_text(glyph) == glyph
        from parse_bench.evaluation.metrics.parse.utils import normalize_cell_text

        assert normalize_cell_text(relaxed_normalize_cell_text("●")) != normalize_cell_text(
            relaxed_normalize_cell_text("○")
        )


class TestComposesWithStrict:
    """The relaxed fold runs before the strict normalize_cell_text; the
    composition is stable (running strict on top is a no-op on the fold's
    output for these cases)."""

    def test_bullet_survives_strict(self) -> None:
        folded = relaxed_normalize_cell_text("■ A<br>■ B")
        assert normalize_cell_text(folded) == "• A • B"

    def test_idempotent(self) -> None:
        once = relaxed_normalize_cell_text("<ul><li>A</li><li>B</li></ul>")
        assert relaxed_normalize_cell_text(once) == once

    def test_empty_string(self) -> None:
        assert relaxed_normalize_cell_text("") == ""
