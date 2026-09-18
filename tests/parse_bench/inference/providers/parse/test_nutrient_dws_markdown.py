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


def _word(text: str, x: float, width: float = 20, **styles: bool) -> dict:
    return {
        "text": text,
        "bounds": {"x": x, "y": 10, "width": width, "height": 10},
        **styles,
    }


if __name__ == "__main__":
    unittest.main()
