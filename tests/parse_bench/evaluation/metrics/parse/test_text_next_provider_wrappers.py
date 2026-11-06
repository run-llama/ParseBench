"""Regression coverage for provider-specific markup in text_next."""

import pytest

from parse_bench.evaluation.metrics.parse.text_next import score_text_next, tokenize


@pytest.mark.parametrize("tag", ["change", "verbose", "signature", "empty"])
def test_provider_wrapper_names_do_not_become_inserted_words(tag: str) -> None:
    reference = "The change preserves all the original words in this complete sentence."
    output = f"The change preserves all the <{tag}>original words</{tag}> in this complete sentence."
    assert [token.key for token in tokenize(output)[0]] == [token.key for token in tokenize(reference)[0]]
    assert score_text_next(reference, output).score == score_text_next(reference, reference).score


def test_revision_wrappers_do_not_hide_added_or_deleted_text() -> None:
    reference = "The agreement contains the original important words and a final clause."
    output = (
        "The agreement contains the <change><del>original</del><ins>altered</ins></change>"
        " important words and a final clause."
    )
    assert score_text_next(reference, output).score < score_text_next(reference, reference).score


@pytest.mark.parametrize("tag", ["signature:id", "empty-field", "change-log", "verbose:details"])
def test_similarly_named_unknown_tags_keep_their_words(tag: str) -> None:
    reference = f"This complete sentence includes a literal label <{tag}>."
    output = "This complete sentence includes a literal label."
    assert score_text_next(reference, output).score < score_text_next(reference, reference).score
