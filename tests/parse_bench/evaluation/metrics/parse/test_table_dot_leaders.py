"""Tests for dot-leader stripping in the two table-cell normalizers.

A dot leader is the row of periods a typesetter runs between a label and its
value so the eye can track across the column::

    1995 . . . . . . . . . . . . . . . . . . .    57.4%    37.6%

The dot count is a function of column width, not of content — the same label
reaches the evaluator as ``Total assets`` from one document and
``Total assets . . . . . . .`` from another, and neither spelling is more
correct than the other.

``_normalize_trm_cell_text`` already stripped both the contiguous
(``.....``) and the spaced (``. . . . .``) form, but ``normalize_cell_text``
— the normalizer GriTS, TEDS and header accuracy use — stripped only a
*trailing contiguous* run. A prediction that reproduces a spaced leader
faithfully therefore scored a full cell miss under GriTS while scoring a
clean match under TRM. Both now call the same ``strip_dot_leaders`` helper,
so the two metrics cannot drift apart on the question again.

The run definition requires a *second* period reachable across blanks alone.
That is what keeps real content intact: in a decimal (``1.5``), a dotted
abbreviation (``U.S.``), a European thousands separator (``1.234.567``) and a
version string (``3.14.1``) the periods are separated by non-blank characters,
so none of them is a leader.


The acceptance property: indifference to leader decoration
==========================================================

The grader must not be able to tell whether a leader was drawn, or how its
dots were spaced. Formally, for a label ``L`` and any leader decoration ``d``::

    normalize(L + d) == normalize(L)

This has to hold with the decoration on *either* side, because which side
carries it is not fixed: human-annotated ground truth usually omits leaders
while a faithful prediction reproduces them, but ground-truth corpora
transcribed from the page carry them too.

How much a residual difference costs depends on the metric, and the two
families differ sharply:

* **Fuzzy** — GriTS-Con scores cells with ``_lcs_similarity``
  (``2·|LCS| / (|s1| + |s2|)``, ``grits_metric.py:102``) and TEDS with a
  normalized Levenshtein distance (``teds_metric.py:120``). A one-character
  residual costs a fraction of one cell, and that fraction grows as the cell
  gets shorter — 0.06 on ``"Acme Inc."`` but 0.33 on ``"3."``.
* **Exact** — table record match compares cells with ``cell_score``
  (``table_record_match_metric.py``) and header accuracy's
  ``header_content_bag`` / ``header_data_alignment`` / ``header_perfect``
  submetrics (``header_accuracy_metric.py``). There a one-character residual
  would cost the **entire cell**.

Because the exact-match family exists, "the residual is small" is not an
available defence, and these tests assert the property rather than a
similarity floor.


Two layers: normalization, then leader-insensitive comparison
=============================================================

One family of cells cannot satisfy the property by normalization alone: a
label whose own last character is a period, so that a single period sits
between the label and the run and belongs, by shape alone, to either.
``"Acme Inc. ...."`` normalizes to ``"Acme Inc"`` while an undecorated
``"Acme Inc."`` keeps its period.

Three candidate resolutions were weighed:

1. **Keep the label's period during normalization.** Rejected.
   ``"Acme Inc....."`` is one undifferentiated glyph run and nothing marks how
   many dots belonged to ``Inc.``; ``"Employed. . . ."`` has precisely the
   same shape. Keeping the period therefore forces the *spaced* and
   *contiguous* spellings of one leader to normalize differently, which is the
   exact artefact this module exists to remove and which was measured at 0.046
   grits_trm_composite of false regression on a single page.

2. **Drop every cell-final period on both sides during normalization.** This
   is an exact closure, and it was prototyped. Rejected anyway, because
   ``normalize_cell_text`` is shared, and
   ``table_normalization_relaxed._EMPTY_TEMPLATES`` documents a deliberate
   dependency on the opposite convention — ``"ft."`` / ``"in."`` are unit
   residue precisely *because* of the period, and its comment states that the
   trailing-period strip "is intentionally NOT applied" since bare ``"ft"`` /
   ``"in"`` would sweep up legitimate English columns. Mutating every
   period-terminated table cell also reaches far past "repeated dots", the
   property actually under repair, and feeds the mutated text downstream.

3. **Decide it at comparison time.** Adopted. ``cells_match_leader_insensitive``
   (``utils.py``) asks of a *pair* whether their difference is confined to a
   trailing dot/period tail — each side gives up its own tail, so the period's
   owner never has to be decided. It rewrites no cell and feeds nothing
   downstream, so the "intentionally NOT applied" comment in the relaxed
   normalizer still governs the mutation pipeline unchanged; this layer sits
   beside it, not against it. Only the tail is discounted, which is what keeps
   ``"3.1.4"`` off ``"3.14"`` and ``"192.168.1.1"`` off ``"192.168.1"``.

So a *normalization* boundary remains — ``strip_dot_leaders("Acme Inc. ....")``
is still ``"Acme Inc"`` while ``"Acme Inc."`` keeps its period — but it costs
nothing on the exact-match metrics. ``TestTheNormalizationBoundary`` pins the
text, and ``TestComparisonLevelLeaderIndifference`` pins that the metrics no
longer see it. The boundary's two shapes:

* a final token that ends in a period and contains no other one (``"Inc."``,
  ``"Jr."``, ``"No."``, ``"ft."``) — absorbed under every leader spelling;
* a *multi-period* abbreviation with the leader written contiguously
  (``"U.S....."`` -> ``"U.S"``). A blank between them is evidence that the
  period is the label's, and where that evidence exists it is used:
  ``"U.S. ....."`` and ``"E.O. ....."`` keep their period.

The GriTS / TEDS residual is unchanged and still real
=====================================================

Those metrics score cells with a similarity function, not equality, and are
deliberately *not* routed through the comparison layer: a fuzzy score is
already tolerant of a one-character tail, and blending an equality override
into it would make its numbers non-comparable with published GriTS. The
measured residual is pinned below at ~0.06 (``"Acme Inc."``) to ~0.33
(``"3."``) of one cell.

Numeric labels are exempt, and that is a fix in this change
==========================================================

A number has the same *shape* as an abbreviation — ``"3.14."`` and ``"E.O."``
are both a period-containing token ending in a period — but not the same
ambiguity, because a number never legitimately ends in a period. Before this
change the shape rule was applied to both, so ``"3.14. . . ."`` normalized to
``"3.14."``: a period present in neither the content nor the leader, and one
that the merely spaced spelling ``"3.14 . . . ."`` did not produce. Numeric
labels now resolve the adjacent period to the leader
(``_ATTACHED_NUMERIC_DOT_LEADER_RE``), which restores full indifference for
every numeric cell — see ``TestNumericCells``.
"""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.grits_metric import _lcs_similarity
from parse_bench.evaluation.metrics.parse.header_accuracy_metric import (
    HeaderCell,
    _header_content_bag_score,
    _header_data_alignment_score,
    _header_perfect_score,
    _normalize_header_text,
)
from parse_bench.evaluation.metrics.parse.table_record_match_metric import (
    _normalize_trm_cell_text,
    align_columns,
    cell_score,
)
from parse_bench.evaluation.metrics.parse.utils import (
    cells_match_leader_insensitive,
    leader_insensitive_core,
    normalize_cell_text,
    strip_dot_leaders,
)

#: Every way one leader run has been observed spelled. Reused by the
#: indifference sweeps below, with the decoration appended to a label.
DECORATIONS = [
    "",  # undecorated — the usual ground-truth spelling
    "..",
    "....",
    "...........",
    " ..",
    " ....",
    ". . . .",
    " . . . .",
    ".  .  .",
    ".. ..",
    "…",
    " …",
]


def trm_cell_match(gt: str, pred: str) -> float:
    """Score one cell pair the way table record match does.

    TRM normalizes with ``normalize_text`` + ``_normalize_trm_cell_text`` and
    then compares for **equality**, so this returns 1.0 or 0.0 and nothing in
    between. It is the strictest consumer of ``strip_dot_leaders`` and the
    reason the tests below assert equality rather than a similarity floor.
    """
    return cell_score(
        _normalize_trm_cell_text(normalize_cell_text(gt)),
        _normalize_trm_cell_text(normalize_cell_text(pred)),
    )


def _header_cells(texts: list[str]) -> list[HeaderCell]:
    """One row of header cells, normalized the way the metric normalizes them."""
    return [HeaderCell(_normalize_header_text(t), 0, i, 1, 1) for i, t in enumerate(texts)]


def header_content_bag(gt_texts: list[str], pred_texts: list[str]) -> float:
    """Score one header row pair the way ``header_content_bag`` does."""
    return _header_content_bag_score(_header_cells(gt_texts), _header_cells(pred_texts))


def header_data_alignment(gt_texts: list[str], pred_texts: list[str]) -> float:
    """Score one header row pair the way ``header_data_alignment`` does.

    The grid mapping is the identity here — the point under test is the text
    comparison at the mapped position, not the alignment search.
    """
    lookup = {(0, i): _normalize_header_text(t) for i, t in enumerate(pred_texts)}
    return _header_data_alignment_score(
        _header_cells(gt_texts),
        lookup,
        {0: 0},
        {i: i for i in range(len(pred_texts))},
    )


def header_perfect(gt_texts: list[str], pred_texts: list[str]) -> float:
    """Score one header row pair the way ``header_perfect`` does."""
    return _header_perfect_score(_header_cells(gt_texts), _header_cells(pred_texts))


class TestSpacedLeadersAreStripped:
    """The form that only TRM used to handle."""

    @pytest.mark.parametrize(
        "cell",
        [
            "Total assets . . . . . . .",
            "Total assets .  .  .  .",
            "Total assets ..",
            "Total assets.......",
        ],
    )
    def test_grits_normalizer_matches_the_bare_label(self, cell: str) -> None:
        assert normalize_cell_text(cell) == normalize_cell_text("Total assets")

    @pytest.mark.parametrize(
        "cell",
        [
            "Total assets . . . . . . .",
            "Total assets.......",
        ],
    )
    def test_trm_normalizer_matches_the_bare_label(self, cell: str) -> None:
        assert _normalize_trm_cell_text(cell) == _normalize_trm_cell_text("Total assets")

    def test_the_motivating_cell(self) -> None:
        # Verbatim from the prediction for a market-performance table: the
        # year label carries the leader that the page prints.
        cell = "1995 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ."
        assert normalize_cell_text(cell) == "1995"

    def test_the_two_normalizers_agree(self) -> None:
        # The bug was a disagreement, not a missing feature. Pin the agreement.
        cell = "Total assets . . . . . . ."
        assert strip_dot_leaders(cell).strip() == "Total assets"
        assert normalize_cell_text(cell).lower() == _normalize_trm_cell_text(normalize_cell_text(cell)).lower()


class TestIndifferenceToLeaderDecoration:
    """The acceptance property, swept over every spelling and both sides.

    A label whose last character is a period is excluded here and handled by
    ``TestTheKnownBoundary`` instead — see the module docstring.
    """

    LABELS = [
        "Total assets",
        "Employed",
        "1995",
        "Cash and equivalents",
        "PHS Act",
        "FD&C Act",
        "Net income (loss)",
        # Numeric cells hold across every spelling as of this change.
        "3.14",
        "1,234.56",
        "$1,234.56",
        "192.168.1.1",
    ]

    @pytest.mark.parametrize("label", LABELS)
    def test_every_decoration_reduces_to_one_text(self, label: str) -> None:
        """normalize(label + d) is the same text for every decoration d."""
        results = {normalize_cell_text(label + d) for d in DECORATIONS}
        assert len(results) == 1, f"{label!r} normalized {len(results)} ways: {sorted(results)}"

    @pytest.mark.parametrize("label", LABELS)
    @pytest.mark.parametrize("decoration", DECORATIONS)
    def test_decoration_on_the_prediction_side(self, label: str, decoration: str) -> None:
        """GT plain, prediction decorated — the usual human-annotation case."""
        assert trm_cell_match(label, label + decoration) == 1.0

    @pytest.mark.parametrize("label", LABELS)
    @pytest.mark.parametrize("decoration", DECORATIONS)
    def test_decoration_on_the_ground_truth_side(self, label: str, decoration: str) -> None:
        """GT decorated, prediction plain — GT corpora carry leaders too."""
        assert trm_cell_match(label + decoration, label) == 1.0

    @pytest.mark.parametrize("label", LABELS)
    def test_two_different_decorations_agree(self, label: str) -> None:
        """Both sides decorated, differently — the paired-replay case."""
        assert trm_cell_match(label + " ....", label + ". . . .") == 1.0


class TestTheNormalizationBoundary:
    """A label ending in a period does not normalize to the same *text* as its
    decorated form — but the exact-match metrics no longer see the difference.

    Pinned deliberately, so the trade-off recorded in the module docstring is
    visible in the suite rather than folklore. What the metrics score is
    pinned in ``TestComparisonLevelLeaderIndifference``.
    """

    #: (label whose final token is period-free but ends in a period, decorated form)
    BOUNDARY_PAIRS = [
        ("Acme Inc.", "Acme Inc. ...."),
        ("Acme Inc.", "Acme Inc....."),
        ("Jr.", "Jr. .."),
        ("No.", "No. ..."),
        ("3.", "3. ...."),
        ("ft.", "ft. ...."),
    ]

    @pytest.mark.parametrize(("plain", "decorated"), BOUNDARY_PAIRS)
    def test_the_decorated_form_loses_the_label_period(self, plain: str, decorated: str) -> None:
        assert normalize_cell_text(plain) == plain
        assert normalize_cell_text(decorated) == plain.rstrip(".")

    @pytest.mark.parametrize(("plain", "decorated"), BOUNDARY_PAIRS)
    def test_the_exact_match_metrics_no_longer_pay_for_it(self, plain: str, decorated: str) -> None:
        """The two texts differ, yet TRM scores them a full match.

        This is the whole point of the comparison layer: the residual is
        confined to a trailing period, which ``cells_match_leader_insensitive``
        discounts on both sides at once.
        """
        assert normalize_cell_text(plain) != normalize_cell_text(decorated)
        assert trm_cell_match(plain, decorated) == 1.0
        assert trm_cell_match(decorated, plain) == 1.0

    @pytest.mark.parametrize(
        ("plain", "decorated", "expected_similarity"),
        [
            # The fuzzy metrics pay only the missing character, and the shorter
            # the cell the larger that fraction is.
            ("Acme Inc.", "Acme Inc. ....", 2 * 8 / (9 + 8)),  # ~0.941
            ("Jr.", "Jr. ..", 2 * 2 / (3 + 2)),  # 0.800
            ("3.", "3. ....", 2 * 1 / (2 + 1)),  # ~0.667
        ],
    )
    def test_the_fuzzy_metrics_pay_only_the_character(
        self, plain: str, decorated: str, expected_similarity: float
    ) -> None:
        """GriTS-Con cost, quantified — it scales inversely with cell length.

        Unchanged by the comparison layer, which the fuzzy metrics do not use.
        """
        similarity = _lcs_similarity(normalize_cell_text(plain), normalize_cell_text(decorated))
        assert similarity == pytest.approx(expected_similarity)
        assert similarity < 1.0

    @pytest.mark.parametrize(
        ("plain", "decorated"),
        [
            # A blank between the abbreviation and the run is evidence that the
            # period is the label's, and the rules use it: the attached rule's
            # label token is ``[^\s.…]+`` and cannot span the abbreviation's own
            # periods, so only the blank-introduced run is stripped.
            ("E.O.", "E.O. ....."),
            ("U.S.A.", "U.S.A. ..."),
            ("Ph.D.", "Ph.D. ...."),
            ("U.S.", "U.S. . . . ."),
        ],
    )
    def test_a_spaced_leader_leaves_a_multi_period_abbreviation_intact(self, plain: str, decorated: str) -> None:
        assert normalize_cell_text(decorated) == plain
        assert trm_cell_match(plain, decorated) == 1.0

    @pytest.mark.parametrize(
        ("plain", "decorated"),
        [
            # Written contiguously there is no such evidence — the run and the
            # abbreviation's period are one glyph run — so the period goes.
            ("E.O.", "E.O....."),
            ("U.S.A.", "U.S.A...."),
            ("U.S.", "U.S...."),
            ("Ph.D.", "Ph.D.. . . ."),
        ],
    )
    def test_a_contiguous_leader_absorbs_the_abbreviation_period(self, plain: str, decorated: str) -> None:
        # The normalized *text* loses the period ...
        assert normalize_cell_text(decorated) == plain.rstrip(".")
        # ... and the comparison layer makes that free on the exact metrics.
        assert trm_cell_match(plain, decorated) == 1.0
        assert trm_cell_match(decorated, plain) == 1.0


class TestComparisonLevelLeaderIndifference:
    """The exact-match metrics ignore a trailing dot/period tail.

    ``cells_match_leader_insensitive`` is asked of a *pair*, so both sides give
    up their own tail and nobody has to decide whether a period closed an
    abbreviation or opened a leader. It rewrites no cell — every normalizer in
    this file still produces exactly the text pinned elsewhere in the module.
    """

    #: The shapes the normalizer cannot reconcile, which this layer must.
    RESIDUAL_PAIRS = [
        ("Acme Inc.", "Acme Inc. ...."),
        ("Acme Inc.", "Acme Inc....."),
        ("Jr.", "Jr. .."),
        ("No.", "No. ..."),
        ("ft.", "ft. ...."),
        ("3.", "3. ...."),
        ("E.O.", "E.O....."),
        ("U.S.", "U.S...."),
        ("U.S.A.", "U.S.A...."),
        ("Ph.D.", "Ph.D.. . . ."),
    ]

    @pytest.mark.parametrize(("plain", "decorated"), RESIDUAL_PAIRS)
    def test_trm_scores_a_full_match(self, plain: str, decorated: str) -> None:
        assert trm_cell_match(plain, decorated) == 1.0
        assert trm_cell_match(decorated, plain) == 1.0

    @pytest.mark.parametrize(("plain", "decorated"), RESIDUAL_PAIRS)
    def test_header_content_bag_scores_a_full_match(self, plain: str, decorated: str) -> None:
        assert header_content_bag([plain], [decorated]) == 1.0
        assert header_content_bag([decorated], [plain]) == 1.0

    @pytest.mark.parametrize(("plain", "decorated"), RESIDUAL_PAIRS)
    def test_header_data_alignment_scores_a_full_match(self, plain: str, decorated: str) -> None:
        assert header_data_alignment([plain], [decorated]) == 1.0
        assert header_data_alignment([decorated], [plain]) == 1.0

    @pytest.mark.parametrize(("plain", "decorated"), RESIDUAL_PAIRS)
    def test_header_perfect_scores_a_full_match(self, plain: str, decorated: str) -> None:
        # The fourth exact-match submetric. Held to the same rule so the
        # header family cannot disagree with itself about one cell.
        assert header_perfect([plain], [decorated]) == 1.0
        assert header_perfect([decorated], [plain]) == 1.0

    @pytest.mark.parametrize(("plain", "decorated"), RESIDUAL_PAIRS)
    def test_the_normalized_texts_really_do_still_differ(self, plain: str, decorated: str) -> None:
        """Guards against the layer being tested against a no-op.

        If a later change closed the gap in the normalizer instead, these
        assertions fail and the tests above stop proving anything.
        """
        assert normalize_cell_text(plain) != normalize_cell_text(decorated)

    def test_it_is_a_comparison_and_not_a_mutation(self) -> None:
        # No normalizer output moved. The relaxed metric's dependency on a
        # cell-final period ("ft." / "in." as unit residue) is untouched.
        assert normalize_cell_text("ft.") == "ft."
        assert _normalize_trm_cell_text("ft.") == "ft."
        assert strip_dot_leaders("Acme Inc. ....").strip() == "Acme Inc"


class TestTheCoreOnlyReachesTheTail:
    """``leader_insensitive_core`` is a right-strip and nothing more."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Acme Inc.", "Acme Inc"),
            ("Acme Inc", "Acme Inc"),
            ("3.", "3"),
            ("3.14", "3.14"),
            ("3.1.4", "3.1.4"),
            ("192.168.1.1", "192.168.1.1"),
            ("192.168.1", "192.168.1"),
            ("$1,234.56.", "$1,234.56"),
            ("$1,234.56", "$1,234.56"),
            # Interior periods are never in reach — the pattern is anchored
            # at the end of the string.
            ("Note 4. Income taxes", "Note 4. Income taxes"),
            ("e.g. deferred taxes", "e.g. deferred taxes"),
            ("A..B", "A..B"),
            # A run with digits after it is content, not a tail.
            ("...5", "...5"),
            ("Introduction ..... 3", "Introduction ..... 3"),
            # Whole-cell dots have no core at all.
            (".", ""),
            ("...", ""),
            ("…", ""),
            ("", ""),
        ],
    )
    def test_core(self, text: str, expected: str) -> None:
        assert leader_insensitive_core(text) == expected


class TestNumericEdgeCasesUnderComparison:
    """The cases that make "just drop the dots" wrong.

    Discounting the *tail* is safe; discounting dots anywhere is not. These
    pin the difference.
    """

    @pytest.mark.parametrize(
        ("gt", "pred"),
        [
            ("3.14", "3.14..."),
            ("3.14", "3.14."),
            ("3.1.4", "3.1.4......"),
            ("3.", "3"),
            ("$1,234.56", "$1,234.56."),
            ("1.5", "1.5."),
            ("2.0.1", "2.0.1…"),
        ],
    )
    def test_a_trailing_tail_is_free(self, gt: str, pred: str) -> None:
        assert trm_cell_match(gt, pred) == 1.0
        assert trm_cell_match(pred, gt) == 1.0
        assert header_content_bag([gt], [pred]) == 1.0
        assert header_content_bag([pred], [gt]) == 1.0

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            # The user-named trap: a version string must never collapse onto
            # the decimal that shares its digits.
            ("3.1.4......", "3.14"),
            ("3.1.4", "3.14"),
            ("3.1.4.", "3.14."),
            # An interior period is content on both sides.
            ("192.168.1.1", "192.168.1"),
            ("192.168.1.1.", "192.168.1."),
            ("1.234.567", "1.234.56"),
            # Different values behind identical decoration.
            ("3.14 ...", "3.15 ..."),
            ("1,234.56 ..", "1,234.57 .."),
            ("Introduction ..... 3", "Introduction ..... 4"),
        ],
    )
    def test_distinct_numbers_stay_distinct(self, left: str, right: str) -> None:
        assert trm_cell_match(left, right) == 0.0
        assert trm_cell_match(right, left) == 0.0
        assert header_content_bag([left], [right]) == 0.0


class TestDotsOnlyCellsUnderComparison:
    """A cell with no core cannot borrow one from the other side.

    The pinned normalization decision is that a dots-only cell (``"..."``,
    ``"…"``) becomes the empty string, and that a lone ``"."`` is not a run and
    survives. The comparison layer keeps faith with both: two dots-only cells
    still compare equal *because they are both empty*, and a cell whose core is
    empty is never matched against a label through the fallback.
    """

    @pytest.mark.parametrize("cell", ["...", "…", "..", "  ..  "])
    def test_two_dots_only_cells_still_match(self, cell: str) -> None:
        assert normalize_cell_text(cell) == ""
        assert trm_cell_match(cell, "...") == 1.0

    @pytest.mark.parametrize("other", ["Total assets", "3", "Acme Inc.", "Jr.", "."])
    def test_a_dots_only_cell_matches_nothing_with_content(self, other: str) -> None:
        assert trm_cell_match("...", other) == 0.0
        assert trm_cell_match(other, "...") == 0.0

    @pytest.mark.parametrize("sentinel", ["-", "", "n/a", "nan"])
    def test_a_dots_only_cell_does_match_a_missing_value_marker(self, sentinel: str) -> None:
        """Not this layer's doing, and pinned so it is not mistaken for it.

        ``"..."`` normalizes to ``""``, and ``cell_score`` treats an empty cell
        and a missing-value sentinel (``"-"``, ``"n/a"``, ...) as equal in its
        own first branch, before any equality check. That predates the
        comparison layer and is unchanged by it.
        """
        assert normalize_cell_text("...") == ""
        assert trm_cell_match("...", sentinel) == 1.0

    def test_a_lone_period_keeps_its_pinned_behavior(self) -> None:
        # ``"."`` normalizes to ``"."`` (not a run) while ``"..."`` normalizes
        # to ``""``. Both cores are empty, so the fallback declines and the
        # two stay unequal — exactly as before this layer existed.
        assert normalize_cell_text(".") == "."
        assert trm_cell_match(".", "...") == 0.0
        assert trm_cell_match(".", "") == 0.0
        assert cells_match_leader_insensitive(".", "") is False
        # Against itself it matches on plain equality, not on the fallback.
        assert trm_cell_match(".", ".") == 1.0

    def test_the_fallback_never_manufactures_a_match_from_nothing(self) -> None:
        assert cells_match_leader_insensitive("...", "Total") is False
        assert cells_match_leader_insensitive("", "Total") is False
        assert cells_match_leader_insensitive(".", "Total") is False


class TestColumnKeyingAgreesWithCellScoring:
    """Row pairing and column keying use the same equivalence as ``cell_score``.

    TRM has no separate record key: rows are paired by a Hungarian assignment
    over ``_record_similarity``, which is a sum of ``cell_score`` — so fixing
    ``cell_score`` fixes row pairing by construction, and there is nothing else
    to extend there. Columns *are* keyed, by header text, but through a fuzzy
    ratio rather than equality. That ratio is length-relative and a short key
    pair such as ``"No."`` / ``"No"`` scores 0.80, below the 0.9 match
    threshold, so without the same equivalence the column would go unmatched
    and every cell under it would score 0 — the metric would pair a row on
    cells it then refused to score. ``_align_columns_header_core`` therefore
    consults ``cells_match_leader_insensitive`` too.
    """

    @pytest.mark.parametrize(
        ("gt_key", "pred_key"),
        [
            ("No.", "No. ..."),
            ("Jr.", "Jr. .."),
            ("Item", "Item ...."),
            ("Acme Inc.", "Acme Inc. ...."),
            ("ft.", "ft. ...."),
        ],
    )
    def test_a_decorated_key_still_matches_its_column(self, gt_key: str, pred_key: str) -> None:
        gt_keys = [normalize_cell_text(gt_key), "Amount"]
        pred_keys = [normalize_cell_text(pred_key), "Amount"]
        mapping, _ = align_columns(gt_keys, pred_keys)
        assert mapping.get(gt_keys[0]) == pred_keys[0]
        # And with the decoration on the other side.
        mapping_rev, _ = align_columns(pred_keys, gt_keys)
        assert mapping_rev.get(pred_keys[0]) == gt_keys[0]

    def test_distinct_keys_are_still_distinct(self) -> None:
        mapping, _ = align_columns(["Employed", "Amount"], ["Unemployed", "Amount"])
        assert "Employed" not in mapping

    def test_an_empty_key_pair_is_left_to_the_fuzzy_path(self) -> None:
        """The fallback is skipped when both keys are empty.

        Two empty keys already pair on ``fuzz.ratio("", "") == 100``, so the
        outcome below is the pre-existing one. The point of the ``not
        gk_empty`` guard is that the fallback must not be what decides it —
        ``cells_match_leader_insensitive("", "")`` is trivially true, and
        routing empty keys through it would silently pin a perfect similarity
        for a case the fuzzy path owns.
        """
        mapping, _ = align_columns(["", "Amount"], ["", "Amount"])
        assert mapping == {"": "", "Amount": "Amount"}
        # An empty key against a real one is still refused, as before.
        mapping_mixed, _ = align_columns(["", "Amount"], ["Item", "Amount"])
        assert "" not in mapping_mixed


class TestLeadersInsideACell:
    """A leader between a label and a value in one cell, not only at the end."""

    def test_label_value_pair_in_one_cell(self) -> None:
        assert normalize_cell_text("Total assets . . . . . 5,061") == "Total assets 5,061"

    def test_a_run_wedged_between_word_characters_survives(self) -> None:
        # No whitespace on either side, so this is punctuation rather than
        # typography. Pinned by ``test_normalize_cell_text`` as ``A..B`` and
        # kept here at cell scale.
        assert normalize_cell_text("Total assets.....5,061") == "Total assets.....5,061"


class TestTableOfContentsRows:
    """Leader followed by trailing content — the TOC shape.

    The run rules end at ``(?=\\s|$)``, so a leader that is followed by a
    space and then a page number is stripped where it stands and the number
    survives. A run with no blank on either side is punctuation and is left
    alone, which is why ``"Introduction...3"`` is untouched.
    """

    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            ("Introduction ..... 3", "Introduction 3"),
            ("Employed. . . . 12", "Employed 12"),
            ("Total assets . . . . . 5,061", "Total assets 5,061"),
            ("Introduction ...3", "Introduction 3"),
            ("Introduction... 3", "Introduction 3"),
            # No blank on either side of the run — punctuation, not typography.
            ("Introduction...3", "Introduction...3"),
        ],
    )
    def test_toc_row_normalization(self, cell: str, expected: str) -> None:
        assert normalize_cell_text(cell) == expected

    @pytest.mark.parametrize(
        ("gt", "pred"),
        [
            ("Introduction 3", "Introduction ..... 3"),
            ("Employed 12", "Employed. . . . 12"),
            ("Total assets 5,061", "Total assets . . . . . 5,061"),
        ],
    )
    def test_a_toc_row_is_indifferent_on_both_sides(self, gt: str, pred: str) -> None:
        assert trm_cell_match(gt, pred) == 1.0
        assert trm_cell_match(pred, gt) == 1.0

    def test_the_page_number_is_still_content(self) -> None:
        # Stripping the leader must not make two different pages compare equal.
        assert normalize_cell_text("Introduction ..... 3") != normalize_cell_text("Introduction ..... 4")


class TestMultipleRunsAndPositions:
    """More than one run per cell, and runs at the start of a cell."""

    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            ("A .. B .. C", "A B C"),
            ("Chapter 1 ... 7 ... 8", "Chapter 1 7 8"),
            ("a. . . b. . . c", "a b c"),
        ],
    )
    def test_every_run_in_the_cell_is_stripped(self, cell: str, expected: str) -> None:
        assert normalize_cell_text(cell) == expected

    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            (".... Total", "Total"),
            ("... Total", "Total"),
            ("…Total", "Total"),
            (".. ..Total", "Total"),
            # A single leading period is not a run, so it is left alone.
            (".Total", ".Total"),
        ],
    )
    def test_a_leading_leader_is_stripped(self, cell: str, expected: str) -> None:
        assert normalize_cell_text(cell) == expected


class TestWhitespaceInsideASpacedRun:
    """The blanks separating a run's dots may be any non-newline whitespace.

    The run pattern separates dots with ``[^\\S\\n]*``, so a transcriber that
    emits NBSP, a tab, a thin space or a figure space between the dots — all
    of which occur when a PDF's inter-glyph gaps are reconstructed — produces
    the same normalized text as one that emits plain spaces.
    """

    SPELLINGS = [
        "Employed. . . .",
        "Employed.\xa0.\xa0.\xa0.",  # NBSP
        "Employed.\t.\t.",  # tab
        "Employed.  .  .",  # thin space
        "Employed. . .",  # figure space
        "Employed\xa0. . . .",  # NBSP before the run
        "Employed.  .  .",  # doubled plain spaces
    ]

    @pytest.mark.parametrize("cell", SPELLINGS)
    def test_every_blank_kind_reduces_to_the_bare_label(self, cell: str) -> None:
        assert normalize_cell_text(cell) == "Employed"

    @pytest.mark.parametrize("cell", SPELLINGS)
    def test_indifferent_against_the_plain_label(self, cell: str) -> None:
        assert trm_cell_match("Employed", cell) == 1.0
        assert trm_cell_match(cell, "Employed") == 1.0

    def test_a_newline_does_not_join_a_run(self) -> None:
        # ``[^\S\n]`` excludes newlines deliberately: two dots on separate
        # lines are two cells' worth of content, not one leader.
        assert normalize_cell_text("foo ....\nbar ....") == "foo bar"


class TestDotOnlyCells:
    """A cell that is nothing but dots — ditto / missing-value marks.

    A run needs two dots, so ``".."``, ``"..."`` and ``"…"`` are complete runs
    and the cell empties; a lone ``"."`` is not a run and survives. Emptying is
    safe for grading precisely because it happens on both sides: a ditto mark
    in ground truth and the same mark in a prediction still compare equal. It
    does mean a ditto cell and a genuinely blank cell become
    indistinguishable, which is recorded here as a decision, not an accident.
    """

    @pytest.mark.parametrize("cell", ["..", "...", "....", "…", " ... ", "…...", "  ..  "])
    def test_a_dot_only_cell_empties(self, cell: str) -> None:
        assert normalize_cell_text(cell) == ""

    def test_a_lone_period_is_not_a_run(self) -> None:
        assert normalize_cell_text(".") == "."

    @pytest.mark.parametrize("cell", ["...", "…", ".."])
    def test_a_ditto_cell_matches_the_same_ditto_on_the_other_side(self, cell: str) -> None:
        # Indifference still holds: both sides normalize identically.
        assert trm_cell_match(cell, cell) == 1.0

    def test_a_ditto_cell_becomes_indistinguishable_from_a_blank(self) -> None:
        # The accepted cost of the rule above. Pinned so it stays a decision.
        assert normalize_cell_text("...") == normalize_cell_text("")


class TestRealContentSurvives:
    """Periods that are content, not typography."""

    @pytest.mark.parametrize(
        "cell",
        [
            "1.5",
            "1.234.567",
            "3.14.1",
            "U.S.",
            "U.S. Treasury",
            "Acme Inc.",
            "e.g. deferred taxes",
            "Note 4. Income taxes",
        ],
    )
    def test_cell_is_unchanged(self, cell: str) -> None:
        assert normalize_cell_text(cell) == cell

    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            # An abbreviation period that *introduces* a leader keeps its own
            # period: a spaced run may not begin mid-token, so only the
            # contiguous tail is a leader. Verbatim from an acronym glossary.
            ("E.O. .....................", "E.O."),
            ("PHS Act ..................", "PHS Act"),
            ("FD&C Act .................", "FD&C Act"),
        ],
    )
    def test_an_abbreviation_before_a_leader_keeps_its_period(self, cell: str, expected: str) -> None:
        assert normalize_cell_text(cell) == expected

    def test_a_sentence_ending_period_survives(self) -> None:
        assert normalize_cell_text("Amounts are in millions.") == "Amounts are in millions."

    def test_an_ellipsis_in_prose_is_treated_as_a_leader(self) -> None:
        # Known and accepted: an ellipsis is indistinguishable from a short
        # leader by shape alone, and this is already the shipped TRM behavior.
        # Recorded here so the trade-off is a decision and not a surprise.
        assert normalize_cell_text("continued ... see note 4") == "continued see note 4"


class TestNumericCells:
    """Numbers keep every period that carries value.

    A leader needs a second dot reachable across blanks alone, so a decimal
    point, a version separator and a European thousands separator are all
    immune — their neighbours are digits, not blanks.
    """

    @pytest.mark.parametrize(
        "cell",
        [
            "3.14",
            "2.0.1",
            "1.5",
            "0.046",
            "192.168.1.1",
            "1,234.56",
            "$1,234.56",
            "1.234.567",
            "(1,234.56)",
            "-3.14",
            "3.14%",
        ],
    )
    def test_numeric_cell_is_unchanged(self, cell: str) -> None:
        assert normalize_cell_text(cell) == cell

    @pytest.mark.parametrize(
        ("plain", "decorated"),
        [
            ("$1,234.56", "$1,234.56 ...."),
            ("3.14", "3.14 ..."),
            ("192.168.1.1", "192.168.1.1 .."),
            ("2.0.1", "2.0.1…"),
        ],
    )
    def test_a_decorated_number_matches_the_plain_number(self, plain: str, decorated: str) -> None:
        assert normalize_cell_text(decorated) == plain
        assert trm_cell_match(plain, decorated) == 1.0
        assert trm_cell_match(decorated, plain) == 1.0

    @pytest.mark.parametrize(
        ("plain", "decorated"),
        [
            # The leader's first dot is glued to the number and the rest is
            # spaced. The number already contains a period, so the period-free
            # attached rule cannot fire and the glued dot used to survive as a
            # spurious trailing period ("3.14." — in neither content nor
            # leader). ``_ATTACHED_NUMERIC_DOT_LEADER_RE`` now resolves it to
            # the leader, because a number never ends in a period.
            ("3.14", "3.14. . . ."),
            ("1,234.56", "1,234.56. . . ."),
            ("$1,234.56", "$1,234.56. . . ."),
            ("192.168.1.1", "192.168.1.1. . . ."),
            ("2.0.1", "2.0.1. . . ."),
            ("0.046", "0.046. . ."),
            ("v1.2.3", "v1.2.3. . . ."),
        ],
    )
    def test_a_glued_first_dot_does_not_leave_a_spurious_period(self, plain: str, decorated: str) -> None:
        assert normalize_cell_text(decorated) == plain
        assert trm_cell_match(plain, decorated) == 1.0
        assert trm_cell_match(decorated, plain) == 1.0

    def test_the_numeric_rule_does_not_reach_a_period_that_ends_a_clause(self) -> None:
        # It only fires when actual dots follow, so a number that merely ends a
        # sentence or introduces a caption keeps its period.
        assert normalize_cell_text("Note 4. Income taxes") == "Note 4. Income taxes"
        assert normalize_cell_text("Rev. 2.0") == "Rev. 2.0"
        assert normalize_cell_text("3.") == "3."

    def test_a_bare_ordinal_is_the_boundary_case(self) -> None:
        # "3." is a label whose final token is period-free but ends in a
        # period, so it sits inside the accepted boundary above, not here.
        assert normalize_cell_text("3.") == "3."
        assert normalize_cell_text("3. ....") == "3"

    def test_stripping_never_merges_two_different_numbers(self) -> None:
        assert normalize_cell_text("3.14 ...") != normalize_cell_text("3.15 ...")
        assert normalize_cell_text("1,234.56 ..") != normalize_cell_text("1,234.57 ..")


class TestSpacingStyleIsNotContent:
    """The leader's *spacing* must not move a metric.

    A dot leader is one glyph run on the page; whether a transcriber renders
    it ``....``, ``. . . .``, ``.. ..`` or ``…`` is a decision about the
    transcription, never about the document. Every spelling therefore has to
    reduce to the same text.

    This was a live source of false regression signal: on one benchmark page
    two arms of a paired replay transcribed the same stub column as
    ``Employed............`` and ``Employed. . . . . . . .``. The spaced
    spelling kept a stray period (``Employed.``) because a spaced run was only
    recognised when a blank introduced it, so an identical reading of an
    identical glyph run cost 0.046 grits_trm_composite.
    """

    #: Every way the same leader run has been observed in the wild.
    SPELLINGS = [
        "Employed............",
        "Employed. . . . . . . .",
        "Employed . . . . . . . .",
        "Employed ...........",
        "Employed.. ..",
        "Employed…",
        "Employed …",
    ]

    @pytest.mark.parametrize("cell", SPELLINGS)
    def test_every_spelling_reduces_to_the_bare_label(self, cell: str) -> None:
        assert normalize_cell_text(cell) == "Employed"

    @pytest.mark.parametrize("cell", SPELLINGS)
    def test_the_two_normalizers_still_agree(self, cell: str) -> None:
        # GriTS and TRM must not drift apart on any spelling.
        assert _normalize_trm_cell_text(normalize_cell_text(cell)) == "Employed"

    def test_the_motivating_pair_scores_identical(self) -> None:
        # The exact two cells from the replay bench, which used to differ.
        assert normalize_cell_text("Employed............") == normalize_cell_text("Employed. . . . . . . .")


class TestEllipsisIsALeader:
    """U+2026 is one glyph standing for three periods."""

    def test_ellipsis_equals_three_dots(self) -> None:
        assert normalize_cell_text("Total assets…") == normalize_cell_text("Total assets...")

    def test_spaced_ellipsis_equals_three_dots(self) -> None:
        assert normalize_cell_text("Total assets …") == normalize_cell_text("Total assets ...")

    def test_ellipsis_between_word_characters_survives(self) -> None:
        # Same rule as ``A..B``: no blank on either side makes it punctuation.
        assert normalize_cell_text("1…5") == "1…5"

    def test_a_mixed_ellipsis_and_period_run_is_one_leader(self) -> None:
        assert normalize_cell_text("Total assets …...") == "Total assets"
        assert normalize_cell_text("Total assets ...…") == "Total assets"


class TestLeaderStrippingIsNotTextEquivalence:
    """Stripping a leader must not make different labels compare equal."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Employed....", "Unemployed...."),
            ("Employed....", "Employer...."),
            ("Employed. . . .", "Employed elsewhere . . . ."),
            ("Total assets . . . 5,061", "Total assets . . . 5,062"),
        ],
    )
    def test_distinct_labels_stay_distinct(self, left: str, right: str) -> None:
        assert normalize_cell_text(left) != normalize_cell_text(right)


class TestVersionsAndDecimalsSurviveTheAttachedRule:
    """The label-attached rule must not reach into dotted content."""

    @pytest.mark.parametrize(
        "cell",
        [
            "v1.2.3",
            "1.2.3",
            "Python 3.14.1",
            "1.5",
            "0.046",
            "Rev. 2.0",
            "192.168.1.1",
        ],
    )
    def test_cell_is_unchanged(self, cell: str) -> None:
        assert normalize_cell_text(cell) == cell


class TestIdempotence:
    """Normalizing twice must equal normalizing once.

    ``strip_dot_leaders`` may run on text another pass already touched — the
    relaxed metric path composes folds — so a rule that keeps finding new runs
    in its own output would make a cell's score depend on how many times it
    was normalized.
    """

    CASES = [
        # Labels crossed with every decoration.
        *[
            label + decoration
            for label in ("Total assets", "Employed", "Acme Inc.", "U.S.A.", "E.O.", "Jr.", "No.", "3.", "ft.")
            for decoration in DECORATIONS
        ],
        # Every other shape pinned in this file.
        "...",
        "…",
        "..",
        ".",
        ".... Total",
        ".Total",
        "Introduction ..... 3",
        "Employed. . . . 12",
        "A .. B .. C",
        "Chapter 1 ... 7 ... 8",
        "3.14",
        "2.0.1",
        "$1,234.56",
        "192.168.1.1",
        "Total assets.....5,061",
        "1…5",
        "Amounts are in millions.",
        "continued ... see note 4",
        "Employed.\xa0.\xa0.",
        "foo ....\nbar ....",
        "",
    ]

    @pytest.mark.parametrize("cell", CASES)
    def test_strip_dot_leaders_is_idempotent(self, cell: str) -> None:
        once = strip_dot_leaders(cell)
        assert strip_dot_leaders(once) == once

    #: ``"No."`` is excluded from the two cell normalizers below and pinned on
    #: its own — it is the single shape where a second pass is not a no-op.
    IDEMPOTENT_CASES = [c for c in CASES if not c.lower().startswith("no.")]

    @pytest.mark.parametrize("cell", IDEMPOTENT_CASES)
    def test_trm_cell_normalizer_is_idempotent(self, cell: str) -> None:
        once = _normalize_trm_cell_text(cell)
        assert _normalize_trm_cell_text(once) == once

    @pytest.mark.parametrize("cell", IDEMPOTENT_CASES)
    def test_normalize_cell_text_is_idempotent(self, cell: str) -> None:
        once = normalize_cell_text(cell)
        assert normalize_cell_text(once) == once

    @pytest.mark.parametrize("cell", ["No. ...", "No. . . . .", "No.…", "No....."])
    def test_a_decorated_boolean_token_is_the_one_non_idempotent_shape(self, cell: str) -> None:
        """``"No. ..."`` -> ``"No"`` -> ``"no"``, because ``no`` is a boolean marker.

        ``_normalize_table_boolean_marker`` runs *before* leader stripping and
        must stay there — it has to see ``○`` / ``●`` before
        ``_normalize_unicode_symbols`` folds them onto one bullet, or a
        checked box and an unchecked box become the same cell. Leader
        stripping can therefore expose a bare ``No`` that only a second pass
        would fold. Harmless in production, where each side is normalized
        exactly once, and pinned here so the ordering constraint is visible.

        ``strip_dot_leaders`` itself stays idempotent on these — the extra
        pass comes from the boolean fold sitting upstream of it, which is why
        the sweep above covers the helper without an exclusion list.
        """
        assert strip_dot_leaders(strip_dot_leaders(cell)) == strip_dot_leaders(cell)
        once = normalize_cell_text(cell)
        assert once == "No"
        assert normalize_cell_text(once) == "no"


class TestSymmetry:
    """Ground truth and prediction go through the same function.

    ``normalize_cell_text`` is applied to the ground-truth grid at
    ``grits_metric.py:861`` and to the prediction grid at
    ``grits_metric.py:865``; TEDS normalizes both trees at
    ``teds_metric.py:120-121``; TRM routes both tables through
    ``normalize_table`` -> ``_normalize_trm_table_text`` ->
    ``strip_dot_leaders`` (``table_record_match_metric.py:105``). There is no
    path that normalizes one side and not the other.
    """

    def test_leader_on_either_side_normalizes_alike(self) -> None:
        gt = "Cash and equivalents"
        pred = "Cash and equivalents . . . . . . . . . ."
        assert normalize_cell_text(gt) == normalize_cell_text(pred)
        # And with the sides swapped — GT corpora carry leaders too.
        assert normalize_cell_text(pred) == normalize_cell_text(gt)

    @pytest.mark.parametrize("decoration", DECORATIONS)
    def test_both_normalizers_are_indifferent_in_both_directions(self, decoration: str) -> None:
        label = "Cash and equivalents"
        decorated = label + decoration
        assert normalize_cell_text(label) == normalize_cell_text(decorated)
        assert _normalize_trm_cell_text(label) == _normalize_trm_cell_text(decorated)
        assert trm_cell_match(label, decorated) == 1.0
        assert trm_cell_match(decorated, label) == 1.0
