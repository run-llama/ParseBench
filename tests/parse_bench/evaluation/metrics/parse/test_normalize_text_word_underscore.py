"""An underscore between two letters/digits separates words; it never pairs with another as italics."""

import pytest

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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a_b_c", "a b c"),  # two inner underscores used to pair as italics
        ("x_1 + y_2", "x 1 + y 2"),  # subscripts across an operator
        ("1_000_000", "1 000 000"),  # digit grouping
        ("file__name", "file name"),  # a run of underscores is one separator
        ("_a_b_", "a b"),  # italic around a welded word still unwraps
        ("_a_ and _b_", "a and b"),  # separate italic words are unchanged
        ("Name: ______", "name: "),  # a fill-in blank touches no word on its right
    ],
)
def test_word_underscore_near_misses(text: str, expected: str) -> None:
    assert normalize_text(text) == expected
