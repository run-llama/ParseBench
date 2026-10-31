from __future__ import annotations

import unittest

from parse_bench.inference.providers.parse.nutrient_dws import NutrientDwsProvider


class TestInlineStyleMarkdown(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = NutrientDwsProvider(
            "nutrient_dws", {"mode": "agentic", "body_source": "graph", "api_key": "test"}
        )

    def test_section_heading_preserves_underlined_word_run(self) -> None:
        element = {
            "type": "paragraph",
            "role": "SectionHeader",
            "headingLevel": 2,
            "text": "The Complaints\nfor Adjudication",
            "words": [
                _word("The", 0, underlined=True),
                _word("Complaints", 40, underlined=True),
                _word("for", 150, underlined=True),
                _word("Adjudication", 185, underlined=True),
            ],
        }

        markdown = self.provider._graph_body_md(element, {})

        self.assertEqual(markdown, "## <u>The Complaints for Adjudication</u>")

    def test_highlighted_word_run_uses_mark_tag(self) -> None:
        element = {
            "type": "paragraph",
            "text": "to supply all relevant documents and",
            "words": [
                _word(t, x, mark=True)
                for t, x in (
                    ("to", 0),
                    ("supply", 25),
                    ("all", 85),
                    ("relevant", 120),
                    ("documents", 205),
                    ("and", 305),
                )
            ],
        }

        markdown = self.provider._graph_body_md(element, {})

        self.assertEqual(markdown, "<mark>to supply all relevant documents and</mark>")

    def test_bold_italic_run_uses_triple_asterisk(self) -> None:
        element = {
            "type": "paragraph",
            "text": "Very plain",
            "words": [_word("Very", 0, bold=True, italic=True), _word("plain", 40)],
        }

        self.assertEqual(self.provider._graph_body_md(element, {}), "***Very*** plain")

    def test_superscript_marker_glues_to_its_base_word(self) -> None:
        # A footnote marker sits flush against the word it annotates, so stripping
        # the markup has to reproduce "Note1" rather than "Note 1".
        element = {
            "type": "paragraph",
            "text": "Note 1",
            "words": [_word("Note", 0, width=24), _word("1", 24, superscript=True)],
        }

        self.assertEqual(self.provider._graph_body_md(element, {}), "Note<sup>1</sup>")

    def test_legacy_style_flag_spellings_are_honoured(self) -> None:
        # The API has shipped both spellings for these two flags; neither may be dropped.
        element = {
            "type": "paragraph",
            "text": "A B",
            "words": [_word("A", 0, underline=True), _word("B", 20, strikeout=True)],
        }

        self.assertEqual(self.provider._graph_body_md(element, {}), "<u>A</u> ~~B~~")

    def test_unstyled_text_is_returned_unchanged(self) -> None:
        element = {
            "type": "paragraph",
            "text": "just text",
            "words": [_word("just", 0), _word("text", 40)],
        }

        self.assertEqual(self.provider._graph_body_md(element, {}), "just text")

    def test_missing_words_falls_back_to_element_text(self) -> None:
        # `includeWords` off: there is nothing to style from, and the plain text stands.
        element = {"type": "paragraph", "text": "no word data"}

        self.assertEqual(self.provider._graph_body_md(element, {}), "no word data")


class TestWrapHyphenMarkdown(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = NutrientDwsProvider(
            "nutrient_dws", {"mode": "agentic", "body_source": "graph", "api_key": "test"}
        )

    def test_word_broken_at_a_line_end_is_rejoined(self) -> None:
        element = {"type": "paragraph", "text": "die aufge-\nführten Daten\nsind da"}

        self.assertEqual(self.provider._graph_body_md(element, {}), "die aufgeführten\nDaten\nsind da")

    def test_continuation_that_is_the_whole_line_leaves_no_empty_line(self) -> None:
        element = {"type": "paragraph", "text": "a regula-\ntory\nbody"}

        self.assertEqual(self.provider._graph_body_md(element, {}), "a regulatory\nbody")

    def test_hyphen_before_a_capitalised_line_is_kept(self) -> None:
        element = {"type": "paragraph", "text": "Inter-\nProvincial Committee"}

        self.assertEqual(self.provider._graph_body_md(element, {}), "Inter-\nProvincial Committee")

    def test_dash_after_a_space_is_kept(self) -> None:
        element = {"type": "paragraph", "text": "Steve Morrow -\nOffice"}

        self.assertEqual(self.provider._graph_body_md(element, {}), "Steve Morrow -\nOffice")

    def test_url_wrapped_at_its_own_hyphen_is_kept(self) -> None:
        element = {
            "type": "paragraph",
            "text": (
                "see https://example.org/defesa-civil-\ncontabiliza-estragos and https://a.org/inline-\nfiles/x.pdf"
            ),
        }

        self.assertEqual(
            self.provider._graph_body_md(element, {}),
            "see https://example.org/defesa-civil-\ncontabiliza-estragos and https://a.org/inline-\nfiles/x.pdf",
        )

    def test_wrapped_heading_joins_before_it_collapses(self) -> None:
        element = {
            "type": "paragraph",
            "role": "SectionHeader",
            "headingLevel": 3,
            "text": "National Association of Pharmacy Regula-\ntory Authorities (NAPRA)",
        }

        self.assertEqual(
            self.provider._graph_body_md(element, {}),
            "### National Association of Pharmacy Regulatory Authorities (NAPRA)",
        )

    def test_styled_words_join_across_lines(self) -> None:
        element = {
            "type": "paragraph",
            "text": "of Pharmacy Regula-\ntory Authorities",
            "words": [
                _word("of", 0, bold=True),
                _word("Pharmacy", 25, bold=True),
                _word("Regula-", 90, bold=True),
                _word("tory", 0, y=30, bold=True),
                _word("Authorities", 40, y=30, bold=True),
            ],
        }

        self.assertEqual(
            self.provider._graph_body_md(element, {}),
            "**of Pharmacy Regulatory**\n**Authorities**",
        )

    def test_code_keeps_its_hyphens(self) -> None:
        element = {"type": "paragraph", "role": "Code", "text": "run --no-\nverify"}

        self.assertEqual(self.provider._graph_body_md(element, {}), "```\nrun --no-\nverify\n```")

    def test_code_fence_carries_the_element_language(self) -> None:
        element = {"type": "paragraph", "role": "Code", "text": "if x:\n    print(x)", "codeLanguage": "python"}

        self.assertEqual(self.provider._graph_body_md(element, {}), "```python\nif x:\n    print(x)\n```")

    def test_cjk_line_break_joins_without_a_space(self) -> None:
        element = {"type": "paragraph", "role": "Text", "text": "健康保险的基\n本信息"}

        self.assertEqual(self.provider._graph_body_md(element, {}), "健康保险的基本信息")

    def test_hangul_line_break_stays_a_word_break(self) -> None:
        element = {"type": "paragraph", "role": "Text", "text": "한국어\n문장"}

        self.assertEqual(self.provider._graph_body_md(element, {}), "한국어\n문장")

    def test_styled_cjk_words_join_across_lines(self) -> None:
        element = {
            "type": "paragraph",
            "role": "SectionHeader",
            "headingLevel": 2,
            "text": "健康保险的基\n本信息",
            "words": [
                _word("健康保险的基", 0, width=60, bold=True),
                _word("本信息", 0, width=30, y=30, bold=True),
            ],
        }

        self.assertEqual(self.provider._graph_body_md(element, {}), "## **健康保险的基本信息**")


class TestStyleRunAcrossLineWraps(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = NutrientDwsProvider(
            "nutrient_dws", {"mode": "agentic", "body_source": "graph", "api_key": "test"}
        )

    def test_bold_sentence_over_a_full_line_stays_one_span(self) -> None:
        element = _element(
            [
                [("We now acknowledge this as a limitation", True), ("of", True)],
                [("the survey design", True)],
            ],
            right=300,
        )

        self.assertEqual(
            self.provider._graph_body_md(element, {}),
            "**We now acknowledge this as a limitation of\nthe survey design**",
        )

    def test_lower_case_continuation_keeps_the_span_open(self) -> None:
        element = _element([[("the vibrant and", False, True)], [("traditional flavors", False, True)]], right=300)

        self.assertEqual(self.provider._graph_body_md(element, {}), "*the vibrant and\ntraditional flavors*")

    def test_span_that_starts_and_ends_mid_line_crosses_the_wrap(self) -> None:
        element = _element(
            [
                [("This limitation is now described as", False), ("Although we", True)],
                [("examined several predictors", True), ("in the study.", False)],
            ],
            right=300,
        )

        self.assertEqual(
            self.provider._graph_body_md(element, {}),
            "This limitation is now described as **Although we\nexamined several predictors** in the study.",
        )

    def test_line_that_ends_a_sentence_closes_the_span(self) -> None:
        element = _element(
            [[("The questions did not differentiate the care.", True)], [("Methods were updated", True)]],
            right=300,
        )

        self.assertEqual(
            self.provider._graph_body_md(element, {}),
            "**The questions did not differentiate the care.**\n**Methods were updated**",
        )

    def test_short_title_case_lines_stay_separate_spans(self) -> None:
        # A name over its credentials: two units that share a style, not a wrapped sentence.
        element = _element([[("Victoria Powell RN CCM", True)], [("CNLCP MSCC CEAS", True)]], right=150)

        self.assertEqual(self.provider._graph_body_md(element, {}), "**Victoria Powell RN CCM**\n**CNLCP MSCC CEAS**")

    def test_line_with_room_left_for_the_next_word_is_a_hard_break(self) -> None:
        element = _element(
            [[("Office of the Attorney General of the State", True)], [("Department of Justice", True)]],
            right=900,
        )

        self.assertEqual(
            self.provider._graph_body_md(element, {}),
            "**Office of the Attorney General of the State**\n**Department of Justice**",
        )

    def test_heading_lines_form_one_span(self) -> None:
        element = {**_element([[("Sunday Masses", True)], [("29/30 September", True)]], right=150)}
        element["role"] = "SectionHeader"
        element["headingLevel"] = 4

        self.assertEqual(self.provider._graph_body_md(element, {}), "#### **Sunday Masses 29/30 September**")


class TestProductHyphenDecision(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = NutrientDwsProvider(
            "nutrient_dws", {"mode": "agentic", "body_source": "graph", "api_key": "test"}
        )

    def _element(self, decision: str | None, bold: bool = False) -> dict:
        styles = {"bold": True} if bold else {}
        hyphen = {"hyphenAtLineEnd": decision} if decision else {}
        return {
            "type": "paragraph",
            "text": "a performance-\nbased plan",
            "words": [
                _word("a", 0, **styles),
                _word("performance-", 25, **styles, **hyphen),
                _word("based", 0, y=30, **styles),
                _word("plan", 30, y=30, **styles),
            ],
        }

    def test_compound_keeps_its_hyphen(self) -> None:
        self.assertEqual(self.provider._graph_body_md(self._element("compound"), {}), "a performance-based\nplan")

    def test_wrap_drops_the_hyphen(self) -> None:
        self.assertEqual(self.provider._graph_body_md(self._element("wrap"), {}), "a performancebased\nplan")

    def test_suspended_keeps_the_words_apart(self) -> None:
        self.assertEqual(self.provider._graph_body_md(self._element("suspended"), {}), "a performance-\nbased plan")

    def test_no_decision_keeps_the_old_rule(self) -> None:
        # Payloads without the field render as before: the hyphen is taken for a wrap hyphen.
        self.assertEqual(self.provider._graph_body_md(self._element(None), {}), "a performancebased\nplan")

    def test_unknown_decision_is_ignored(self) -> None:
        self.assertEqual(self.provider._graph_body_md(self._element("other"), {}), "a performancebased\nplan")

    def test_compound_before_a_capital_fuses(self) -> None:
        element = {
            "type": "paragraph",
            "text": "Franco-\nGerman talks",
            "words": [
                _word("Franco-", 0, hyphenAtLineEnd="compound"),
                _word("German", 0, y=30),
                _word("talks", 50, y=30),
            ],
        }

        self.assertEqual(self.provider._graph_body_md(element, {}), "Franco-German\ntalks")

    def test_styled_words_follow_the_decision(self) -> None:
        for decision, expected in (
            ("compound", "**a performance-based\nplan**"),
            ("wrap", "**a performancebased\nplan**"),
            ("suspended", "**a performance-\nbased plan**"),
            (None, "**a performancebased\nplan**"),
        ):
            with self.subTest(decision=decision):
                self.assertEqual(self.provider._graph_body_md(self._element(decision, bold=True), {}), expected)


def _word(text: str, x: float, width: float = 20, y: float = 10, **styles: bool) -> dict:
    return {
        "text": text,
        "bounds": {"x": x, "y": y, "width": width, "height": 10},
        **styles,
    }


def _element(lines: list[list[tuple]], right: float) -> dict:
    """A paragraph from (text, bold[, italic]) runs, one list per visual line, laid out from x=0.

    Each run is split into words 8 units per character wide; the element ends at ``right``.
    """
    words = []
    for n, line in enumerate(lines):
        x = 0.0
        for run in line:
            text, bold = run[0], run[1]
            italic = run[2] if len(run) > 2 else False
            for token in text.split():
                styles = {k: True for k, v in (("bold", bold), ("italic", italic)) if v}
                words.append(_word(token, x, width=8 * len(token), y=10 + 30 * n, **styles))
                x += 8 * len(token) + 5
    return {
        "type": "paragraph",
        "text": "\n".join(" ".join(run[0] for run in line) for line in lines),
        "bounds": {"x": 0, "y": 0, "width": right, "height": 30 * len(lines)},
        "words": words,
    }


if __name__ == "__main__":
    unittest.main()
