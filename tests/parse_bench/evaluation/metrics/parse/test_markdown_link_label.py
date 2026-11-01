"""A Markdown link ``[label](url)`` is scored as its label, in text rules and in table cells."""

import pytest

from parse_bench.evaluation.metrics.parse.rule_based_metric import RuleBasedMetric
from parse_bench.evaluation.metrics.parse.table_record_match_metric import TableRecordMatchMetric
from parse_bench.evaluation.metrics.parse.test_rules import MissingSpecificSentenceRule
from parse_bench.evaluation.metrics.parse.utils import normalize_cell_text, normalize_text


def test_linked_words_satisfy_the_sentence_rule() -> None:
    rule = MissingSpecificSentenceRule(
        {"type": "missing_specific_sentence", "sentence": "See the annual report for details."}
    )
    passed, _ = rule.run("See the [annual report](https://example.com/ar_2024.pdf) for details.")
    assert passed


def test_link_label_replaces_the_link_in_normalized_text() -> None:
    assert normalize_text("see [the report](<https://example.com/a b.pdf>) now") == "see the report now"


def test_image_markdown_is_not_read_as_a_link() -> None:
    assert normalize_text("![chart](img.png)") == "![chart](img.png)"


def test_link_in_a_table_cell_scores_as_its_label() -> None:
    gt = "<table><tr><th>Doc</th><th>Year</th></tr><tr><td>Rate filing</td><td>2024</td></tr></table>"
    pred = (
        "<table><tr><th>Doc</th><th>Year</th></tr>"
        "<tr><td>[Rate filing](https://example.com/f.pdf)</td><td>2024</td></tr></table>"
    )
    assert TableRecordMatchMetric().compute(expected=gt, actual=pred)[0].value == pytest.approx(1.0)
    assert normalize_cell_text("[Rate filing](https://example.com/f.pdf)") == "Rate filing"


def test_link_url_does_not_split_sentences_in_the_rule_metric() -> None:
    rules = [
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {"See the annual report for details.": 1},
        }
    ]
    actual = "See the [annual report](https://example.com/ar_2024.pdf) for details."
    assert RuleBasedMetric().compute(expected=rules, actual=actual).value == pytest.approx(1.0)
