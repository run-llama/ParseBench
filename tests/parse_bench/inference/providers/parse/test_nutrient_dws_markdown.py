from __future__ import annotations

import unittest

from parse_bench.inference.providers.parse.nutrient_dws import NutrientDwsProvider


class TestInlineStyleMarkdown(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = NutrientDwsProvider("nutrient_dws", {"mode": "agentic", "api_key": "test"})

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
        self.provider = NutrientDwsProvider("nutrient_dws", {"mode": "agentic", "api_key": "test"})

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
            "text": "see https://example.org/defesa-civil-\ncontabiliza-estragos and https://a.org/inline-\nfiles/x.pdf",
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


def _word(text: str, x: float, width: float = 20, y: float = 10, **styles: bool) -> dict:
    return {
        "text": text,
        "bounds": {"x": x, "y": y, "width": width, "height": 10},
        **styles,
    }


if __name__ == "__main__":
    unittest.main()
