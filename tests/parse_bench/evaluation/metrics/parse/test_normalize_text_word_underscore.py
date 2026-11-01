"""An underscore between two letters/digits separates words; it never pairs with another as italics."""

from parse_bench.evaluation.metrics.parse.test_rules import MissingSpecificSentenceRule
from parse_bench.evaluation.metrics.parse.utils import normalize_text


def test_underscores_of_two_filenames_do_not_pair_as_italics() -> None:
    assert (
        normalize_text("Files img_001.jpg and img_002.jpg were sent.") == "files img 001.jpg and img 002.jpg were sent."
    )


def test_filename_matches_a_rule_written_with_one_of_the_two_filenames() -> None:
    rule = MissingSpecificSentenceRule({"type": "missing_specific_sentence", "sentence": "img_001.jpg and"})
    passed, _ = rule.run("Files img_001.jpg and img_002.jpg were sent.")
    assert passed


def test_italic_word_is_still_unwrapped() -> None:
    assert normalize_text("This is _important_ text.") == "this is important text."


def test_snake_case_and_dunder_names_split_into_words() -> None:
    assert normalize_text("x = snake_case_name + __init__") == "x = snake case name + init"
