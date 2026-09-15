"""Table cell text must be counted exactly once by the bag rules.

``_augment_with_table_cell_text`` used to *append* extracted cell text to
content that still contained the table, so every table word was counted at
least twice (original + cell) and usually three times (original + cell +
row join).  Colspan/rowspan cells were replicated once per covered grid
position on top of that.  The occurrence-counting rules
(``too_many_word_occurence``, ``unexpected_sentence``, ``bag_of_digit``)
read those inflated counts directly.
"""

import re

from parse_bench.evaluation.metrics.parse.rules_bag import (
    TooManyWordOccurencePercentRule,
    WordBagRule,
)
from parse_bench.evaluation.metrics.parse.rules_base import (
    _augment_with_table_cell_text,
    _extract_table_cell_texts,
)
from parse_bench.test_cases.parse_rule_schemas import ParseTooManyWordOccurrencePercentRule


def _count(word: str, text: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(word)}(?!\w)", text, flags=re.IGNORECASE))


HTML_DOC = """Intro of the doc.

<table>
<tr><td>Table of contents</td><td>Page of 5</td></tr>
<tr><td>Chapter of one</td><td>12</td></tr>
</table>

Outro of the doc."""

MARKDOWN_DOC = """Intro of the doc.

| Name of item | Qty |
| --- | --- |
| Bolt of steel | 3 |

Outro of the doc."""


class TestAugmentedTextCountsCellsOnce:
    def test_html_table_words_counted_once(self):
        augmented = _augment_with_table_cell_text(HTML_DOC)
        # 2 in prose + 3 in table cells
        assert _count("of", augmented) == 5
        assert _count("contents", augmented) == 1
        assert _count("12", augmented) == 1

    def test_markdown_table_words_counted_once(self):
        augmented = _augment_with_table_cell_text(MARKDOWN_DOC)
        # 2 in prose + 2 in table cells
        assert _count("of", augmented) == 4
        assert _count("Bolt", augmented) == 1
        assert _count("Qty", augmented) == 1

    def test_colspan_cell_counted_once(self):
        md = (
            '<table><tr><td colspan="3">Wide header of cells</td></tr>'
            "<tr><td>a1</td><td>b1</td><td>c1</td></tr></table>"
        )
        augmented = _augment_with_table_cell_text(md)
        assert _count("Wide", augmented) == 1
        assert _count("of", augmented) == 1
        assert _extract_table_cell_texts(md) == ["Wide header of cells", "a1", "b1", "c1"]

    def test_rowspan_cell_counted_once(self):
        md = '<table><tr><td rowspan="2">Spanning of rows</td><td>x1</td></tr><tr><td>y1</td></tr></table>'
        augmented = _augment_with_table_cell_text(md)
        assert _count("Spanning", augmented) == 1
        assert _count("of", augmented) == 1

    def test_nested_table_cell_counted_once(self):
        md = "<table><tr><td>Outer of cell</td><td><table><tr><td>Inner of cell</td></tr></table></td></tr></table>"
        augmented = _augment_with_table_cell_text(md)
        assert _count("Outer", augmented) == 1
        assert _count("Inner", augmented) == 1
        assert _count("of", augmented) == 2

    def test_word_in_both_prose_and_cell_counted_twice(self):
        md = "Invoice total is due.\n\n<table><tr><td>Invoice</td><td>42</td></tr></table>"
        augmented = _augment_with_table_cell_text(md)
        assert _count("Invoice", augmented) == 2

    def test_caption_text_survives_and_is_counted_once(self):
        md = "<table><caption>Ozonesonde of note</caption><tr><td>a1</td></tr></table>"
        augmented = _augment_with_table_cell_text(md)
        assert _count("Ozonesonde", augmented) == 1
        assert _count("of", augmented) == 1

    def test_table_free_content_is_unchanged(self):
        md = "Just prose of the doc.\n\nNo tables of any kind here."
        assert _augment_with_table_cell_text(md) == md

    def test_row_spanning_text_stays_contiguous_in_normalized_full_text(self):
        """Cells of a row must remain adjacent so cross-cell sentences match."""
        full_text = WordBagRule._normalize_full_word_text(HTML_DOC)
        full_text = re.sub(r"\s+", " ", full_text)
        assert "table of contents page of 5" in full_text.lower()


class TestTooManyWordOccurenceNotInflatedByTables:
    def test_table_word_does_not_exceed_its_allowed_count(self):
        rule = TooManyWordOccurencePercentRule(
            ParseTooManyWordOccurrencePercentRule(
                type="too_many_word_occurence_percent",
                bag_of_word={"of": 5, "contents": 1},
            )
        )
        passed, explanation, score = rule.run(HTML_DOC)
        assert passed, explanation
        assert score == 1.0
