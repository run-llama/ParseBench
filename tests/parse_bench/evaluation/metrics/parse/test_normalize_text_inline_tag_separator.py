"""Tests for inline-styling-tag stripping in ``normalize_text``.

``normalize_text`` deletes inline styling markers (``<b> <i> <u> <s> <del>
<strike> <ins> <mark> <span>`` and ``~~``) while keeping their content. Deleting
them with the *empty string* welds the two sides together, so a byte-faithful
parser output like ``<u>103</u><s>101</s>`` normalizes to ``103101`` and every
downstream word/sentence/order rule reports the page numbers as missing. The GT
for the same page escapes the weld only because its annotation happened to be
``~~103~~ 101`` — an author-typed space.

**The rule, and why it is a *run* of markers rather than every marker.**
Injecting a separator for *every* stripped tag is not safe: inline tags are used
intra-word in the ground truth. Surveying the shipped text corpora
(``text_extended/v1.0`` + ``text_core/v0.10``, 1120 GT markdown files) for tags
with an alphanumeric character on *both* sides:

* 22 single-tag intra-word hits — 21 of them Japanese, where ``<u>`` opens and
  closes mid-sentence with no whitespace at all (``…とともに、<u>開催地としての
  魅力…</u>取り組みが…`` in ``text_dense/underline.md``), and one Latin,
  ``Shizuok<mark>a University)</mark>`` in ``text_misc/marks.md``, where the
  highlight opens *inside* the word "Shizuoka".
* **0** hits for a run of two or more abutting markers.

So a single marker between two content characters marks a style change *within*
one token and must stay a zero-width join; two or more abutting markers mark the
boundary *between* two separately-styled tokens (``</u><s>``) and must become a
separator. That rule fixes the weld and provably changes nothing in the GT
corpus, which contains no multi-marker run with content on both sides.
"""

from __future__ import annotations

from parse_bench.evaluation.metrics.parse.utils import normalize_text


class TestMarkerRunSeparates:
    """Two or more abutting inline markers separate two styled tokens."""

    def test_close_then_open_does_not_weld(self) -> None:
        # The motivating case: text_simple/strikeUnderline emits the underlined
        # page number and the struck one adjacent, with no whitespace on the
        # page. Both tokens must survive as separate tokens.
        assert normalize_text("<u>103</u><s>101</s>") == "103 101"

    def test_weld_inside_a_table_cell(self) -> None:
        # Verbatim from the failing document; <td> is not stripped here (that is
        # the caller's job) — only the welded numbers are this function's bug.
        assert normalize_text("<td><u>103</u><s>101</s></td>") == "<td>103 101</td>"

    def test_matches_the_ground_truth_spelling(self) -> None:
        # GT wrote the same page as "~~103~~ 101". After the fix both spellings
        # normalize to the same string, which is the whole point.
        assert normalize_text("<u>103</u><s>101</s>") == normalize_text("~~103~~ 101")

    def test_tilde_run_also_separates(self) -> None:
        assert normalize_text("~~103~~<s>101</s>") == "103 101"

    def test_open_open_run_separates(self) -> None:
        assert normalize_text("103<u><s>101</s></u>") == "103 101"

    def test_span_participates_in_a_run(self) -> None:
        assert normalize_text('<span c="1">103</span><u>101</u>') == "103 101"

    def test_run_at_the_string_edges_adds_no_padding(self) -> None:
        # A run with no content on one side separates nothing; emitting a space
        # there would only add leading/trailing padding for callers that compare
        # normalized strings for equality.
        assert normalize_text("<u><b>103</b></u>") == "103"


class TestGroundTruthSideIsEssentiallyUnmoved:
    """The same function normalizes GT rule text and prediction, so the fix must
    barely move the GT side. Sweeping every rule payload in the shipped corpora
    (562 ``.test.json`` files, 165946 rules, 343380 strings) exactly **one**
    string normalizes differently — the nested run below, in
    ``text_misc/edit2.test.json``. Re-scoring that document against the run it
    came from leaves all 93 rules at the same verdict (70 pass) and CF at 0.8140.
    """

    def test_nested_run_separates(self) -> None:
        # text_misc/edit2.md: "HO 04 61-<mark>~~10 00~~</mark>." — <mark>~~ and
        # ~~</mark> are both runs, so the highlighted-and-struck token detaches
        # from the hyphen before it and the period after it.
        assert normalize_text("HO 04 61-<mark>~~10 00~~</mark>.") == "ho 04 61- 10 00 ."


class TestSingleMarkerStillJoins:
    """A lone marker is an intra-token style change — it must not split words."""

    def test_latin_intra_word_tag_keeps_the_word_whole(self) -> None:
        # text_misc/marks.md: "Shizuok<mark>a University)</mark>" — the mark
        # opens inside "Shizuoka". A separator here would emit "shizuok a".
        assert normalize_text("Shizuok<mark>a University)</mark>") == "shizuoka university)"

    def test_latin_intra_word_underline(self) -> None:
        assert normalize_text("re<u>do</u>ne") == "redone"

    def test_japanese_mid_sentence_underline_is_not_split(self) -> None:
        # text_dense/underline.md pattern: CJK runs have no whitespace at all,
        # so every <u> boundary sits between two content characters. Splitting
        # there would desynchronize GT and output on 21 separate occurrences.
        assert normalize_text("魅力や集客力を高める</u>取り組みが") == "魅力や集客力を高める取り組みが"

    def test_single_marker_next_to_a_space_is_unchanged(self) -> None:
        assert normalize_text("<u>103</u> 101") == "103 101"


class TestTagFreeContentIsUntouched:
    """Guard: content with no inline markers normalizes exactly as before."""

    def test_plain_sentence(self) -> None:
        assert normalize_text("The quick brown fox.") == "the quick brown fox."

    def test_table_markup_without_inline_tags(self) -> None:
        assert normalize_text("<td>20121234</td><td>20121235</td>") == ("<td>20121234</td><td>20121235</td>")

    def test_br_still_becomes_a_space(self) -> None:
        assert normalize_text("line1<br>line2") == "line1 line2"

    def test_sup_content_removal_is_unchanged(self) -> None:
        # <sup>/<sub> delete their *content* too — a different decision, left
        # alone here so this change stays confined to content-preserving tags.
        assert normalize_text("84.1<sup>(2)</sup>") == "84.1"
