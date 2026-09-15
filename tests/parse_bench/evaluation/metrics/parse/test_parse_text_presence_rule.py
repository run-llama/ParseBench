from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.parse.test_rules import (
    BagOfDigitPercentRule,
    FormattingRule,
    LatexRule,
    MarkColorRule,
    MissingSentencePercentRule,
    MissingSentenceRule,
    MissingSpecificSentenceRule,
    MissingSpecificWordRule,
    MissingWordPercentRule,
    MissingWordRule,
    NotLatexRule,
    TextOrderRule,
    TextPresenceRule,
    TitleHierarchyPercentRule,
    TitleLevelRule,
    TooManySentenceOccurencePercentRule,
    TooManySentenceOccurenceRule,
    TooManyWordOccurencePercentRule,
    TooManyWordOccurenceRule,
    UnexpectedSentencePercentRule,
    UnexpectedSentenceRule,
    UnexpectedWordPercentRule,
    UnexpectedWordRule,
)


def test_present_rule_default_behavior_with_count_omitted() -> None:
    rule = TextPresenceRule({"type": "present", "text": "hello", "max_diffs": 0})

    passed, message = rule.run("prefix hello suffix")
    assert passed
    assert message == ""


def test_present_rule_keeps_default_behavior_with_count_zero() -> None:
    rule = TextPresenceRule({"type": "present", "text": "hello", "count": 0, "max_diffs": 0})

    passed, message = rule.run("hello hello hello")

    assert passed
    assert message == ""


def test_present_rule_with_exact_count_passes() -> None:
    rule = TextPresenceRule({"type": "present", "text": "hello", "count": 2, "max_diffs": 0})

    passed, message = rule.run("hello world\nhello again")

    assert passed
    assert message == ""


def test_present_rule_normalizes_unicode_double_quotes() -> None:
    rule = TextPresenceRule({"type": "present", "text": 'he said "hello"', "max_diffs": 0})

    passed, message = rule.run("He said “hello” in the report.")

    assert passed
    assert message == ""


def test_present_rule_normalizes_unicode_single_quotes_with_keep_formatting() -> None:
    rule = TextPresenceRule(
        {
            "type": "present",
            "text": "it's correct",
            "keep_formatting_text_normalisation": True,
            "max_diffs": 0,
        }
    )

    passed, message = rule.run("It’s correct")

    assert passed
    assert message == ""


def test_order_rule_matches_latex_formula_simplified() -> None:
    """LaTeX formulas are simplified to numbers/operators/variables, not replaced with 'LATEX'."""
    rule = TextOrderRule(
        {
            "type": "order",
            "before": "Variance of a population is",
            "after": "where sigma is standard deviation",
            "max_diffs": 2,
        }
    )

    passed, message = rule.run("Variance of a population is $\\sigma^2$. where sigma is standard deviation")

    assert passed
    assert message == ""


def test_is_latex_rule_passes_on_inline_dollar_formula() -> None:
    rule = LatexRule(
        {
            "type": "is_latex",
            "formula": r"\frac{a}{b}",
        }
    )

    passed, message = rule.run(r"The ratio is $\frac{a}{b}$ in this line.")

    assert passed
    assert message == ""


def test_is_latex_rule_passes_on_block_formula_with_delimiters_in_query() -> None:
    rule = LatexRule(
        {
            "type": "is_latex",
            "formula": r"$$E = mc^2$$",
        }
    )

    passed, message = rule.run(
        """
Some intro text.
$$
E = mc^2
$$
"""
    )

    assert passed
    assert message == ""


def test_is_latex_rule_fails_when_formula_not_present() -> None:
    rule = LatexRule(
        {
            "type": "is_latex",
            "formula": r"\int_0^1 x\,dx",
        }
    )

    passed, message = rule.run(r"Only $a+b$ is present here.")

    assert not passed
    assert "not found" in message.lower()


def test_is_latex_rule_uses_raw_md_content_not_normalized_content() -> None:
    rule = LatexRule(
        {
            "type": "is_latex",
            "formula": r"\frac{a}{b}",
        }
    )

    passed, message = rule.run(
        r"Value is $\frac{a}{b}$.",
        normalized_content="LATEX",
    )

    assert passed
    assert message == ""


def test_is_latex_rule_surfaces_placeholder_preprocessing_hint() -> None:
    rule = LatexRule(
        {
            "type": "is_latex",
            "formula": r"\frac{a}{b}",
        }
    )

    passed, message = rule.run("This output was preprocessed as LATEX token only.")

    assert not passed
    assert "placeholder" in message.lower()


def test_is_latex_rule_ignores_presentation_only_differences() -> None:
    rule = LatexRule(
        {
            "type": "is_latex",
            "formula": r"(\frac{a}{b}) = c",
        }
    )

    passed, message = rule.run(r"$$\left(\dfrac{a}{b}\right) \ = c \tag{7}$$")

    assert passed, message


def test_is_latex_rule_ignores_equivalent_comparison_and_text_commands() -> None:
    rule = LatexRule(
        {
            "type": "is_latex",
            "formula": r"x \le 4, s.t. y \ge 1",
        }
    )

    passed, message = rule.run(r"$x \leq 4, \text{s.t.} y \geq 1$")

    assert passed, message


def test_is_latex_rule_allows_raw_inequality_characters_inside_math() -> None:
    rule = LatexRule(
        {
            "type": "is_latex",
            "formula": r"\Delta = x < 0.93",
        }
    )

    passed, message = rule.run(r"First $|x| < 1$, then $\Delta = x < 0.93$, and finally $y > 0$.")

    assert passed, message


def test_is_latex_rule_does_not_pair_currency_across_html() -> None:
    rule = LatexRule({"type": "is_latex", "formula": "10 million in revenue"})

    passed, _ = rule.run("<td>$10 million</td><td>$20 million</td>")

    assert not passed


def test_is_latex_rule_does_not_pair_currency_prose_on_one_line() -> None:
    rule = LatexRule({"type": "is_latex", "formula": "10 million and"})

    passed, _ = rule.run("Revenue rose from $10 million and expenses reached $20 million.")

    assert not passed


def test_is_not_latex_rule_passes_for_plain_formula_like_text() -> None:
    rule = NotLatexRule({"type": "is_not_latex", "text": "N = 806"})

    passed, message = rule.run("Phase II/III total N = 806 participants.")

    assert passed
    assert message == ""


def test_is_not_latex_rule_fails_for_target_inside_latex() -> None:
    rule = NotLatexRule({"type": "is_not_latex", "text": "$400 billion"})

    passed, message = rule.run("Expected investment is $400 billion$ next year.")

    assert not passed
    assert "ordinary text" in message


def test_is_not_latex_rule_is_paired_with_presence_for_missing_text() -> None:
    rule = NotLatexRule({"type": "is_not_latex", "text": "$400 billion"})

    passed, message = rule.run("The amount was omitted.")

    assert passed
    assert message == ""


def test_is_title_passes_for_markdown_heading_without_level() -> None:
    rule = TitleLevelRule(
        {
            "type": "is_title",
            "text": "Executive Summary",
        }
    )

    passed, message = rule.run("## Executive Summary")

    assert passed
    assert message == ""


def test_is_title_passes_for_bold_text_without_heading() -> None:
    rule = TitleLevelRule(
        {
            "type": "is_title",
            "text": "Executive Summary",
        }
    )

    passed, message = rule.run("**Executive Summary**")

    assert passed
    assert message == ""


def test_is_title_ignores_level_for_heading_matching() -> None:
    rule = TitleLevelRule(
        {
            "type": "is_title",
            "text": "Executive Summary",
            "level": 1,
        }
    )

    passed_other_heading, _ = rule.run("## Executive Summary")
    passed_bold, _ = rule.run("<b>Executive Summary</b>")

    assert passed_other_heading
    assert passed_bold


def test_is_title_fails_for_inline_bold_not_standalone_line() -> None:
    rule = TitleLevelRule(
        {
            "type": "is_title",
            "text": "Executive Summary",
        }
    )

    passed, message = rule.run("This paragraph has **Executive Summary** inline.")

    assert not passed
    assert "title" in message.lower()


def test_is_title_fails_for_bold_prefix_with_trailing_text() -> None:
    rule = TitleLevelRule(
        {
            "type": "is_title",
            "text": "Executive Summary",
        }
    )

    passed, _ = rule.run("**Executive Summary** details")

    assert not passed


def test_is_bold_rule_accepts_markdown_heading_text() -> None:
    rule = FormattingRule(
        {
            "type": "is_bold",
            "text": "Executive Summary",
        }
    )

    passed, message = rule.run("# Executive Summary")

    assert passed
    assert message == ""


def test_is_not_bold_rule_fails_for_markdown_heading_text() -> None:
    rule = FormattingRule(
        {
            "type": "is_not_bold",
            "text": "Executive Summary",
        }
    )

    passed, message = rule.run("## Executive Summary")

    assert not passed
    assert "unexpectedly had" in message


def test_is_title_passes_when_html_heading_is_entity_escaped() -> None:
    rule = TitleLevelRule(
        {
            "type": "is_title",
            "text": "Executive Summary",
            "level": 2,
        }
    )

    passed, message = rule.run("&lt;h2&gt;Executive Summary&lt;/h2&gt;")

    assert passed
    assert message == ""


def test_is_title_passes_when_markdown_markers_are_escaped() -> None:
    rule = TitleLevelRule(
        {
            "type": "is_title",
            "text": "Executive Summary",
        }
    )

    passed_heading, message_heading = rule.run(r"\# Executive Summary")
    passed_bold, message_bold = rule.run(r"\*\*Executive Summary\*\*")

    assert passed_heading
    assert message_heading == ""
    assert passed_bold
    assert message_bold == ""


@pytest.mark.parametrize("product_split", ["**Product Split**", "<strong>Product Split</strong>"])
def test_title_hierarchy_percent_passes_on_valid_nested_structure(product_split: str) -> None:
    rule = TitleHierarchyPercentRule(
        {
            "type": "title_hierarchy_percent",
            "title_hierarchy": {
                "Executive Summary": {
                    "Revenue Breakdown": ["Regional Split", "Product Split"],
                    "Outlook": {},
                }
            },
        }
    )

    content = f"""
# Executive Summary
## Revenue Breakdown
### Regional Split
{product_split}
## Outlook
"""

    passed, message, score = rule.run(content)

    assert passed
    assert message == ""
    assert score == 1.0


def test_title_hierarchy_percent_penalizes_bad_order_or_depth() -> None:
    rule = TitleHierarchyPercentRule(
        {
            "type": "title_hierarchy_percent",
            "title_hierarchy": {
                "Executive Summary": {
                    "Revenue Breakdown": ["Regional Split", "Product Split"],
                }
            },
        }
    )

    content = """
**Product Split**
## Revenue Breakdown
# Executive Summary
"""

    passed, message, score = rule.run(content)

    assert not passed
    assert 0.0 <= score < 1.0
    assert "score=" in message


def test_present_rule_with_exact_count_fails_when_mismatch() -> None:
    rule = TextPresenceRule({"type": "present", "text": "hello", "count": 2, "max_diffs": 0})

    passed, message = rule.run("hello once")

    assert not passed
    assert "exactly 2 time(s), but found 1" in message


def test_present_rule_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="Count field cannot be negative"):
        TextPresenceRule({"type": "present", "text": "hello", "count": -1})


def test_present_rule_rejects_non_integer_count() -> None:
    with pytest.raises(ValueError, match="integer"):
        TextPresenceRule({"type": "present", "text": "hello", "count": 1.5})


def test_unexpected_sentence_rule_passes_when_only_whitelisted_sentences_exist() -> None:
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "sentence one": 1,
                "another sentence": 1,
            },
        }
    )

    passed, message = rule.run("Sentence one. another sentence")

    assert passed
    assert message == ""


def test_unexpected_sentence_rule_fails_on_unknown_sentence() -> None:
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "allowed sentence": 1,
            },
        }
    )

    passed, message = rule.run("Allowed sentence\nCompletely new sentence")

    assert not passed
    assert "completely new sentence" in message


def test_sentence_rules_merge_short_fragments_with_next() -> None:
    """Short fragments (< 7 chars) are merged with the next chunk instead of
    being silently dropped.  Here "ok" and "tiny" merge with subsequent text."""
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "allowed sentence": 1,
                "ok tiny short": 1,
            },
        }
    )

    # "ok", "tiny", and "short" are merged into one sentence instead of dropped.
    passed, message = rule.run("Allowed sentence. ok. tiny\nshort")

    assert passed
    assert message == ""


def test_sentence_rules_strip_markdown_markers_before_matching() -> None:
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "article": 1,
                "section": 1,
            },
        }
    )

    passed, message = rule.run("# article\n## section")

    assert passed
    assert message == ""


def test_sentence_rules_strip_html_tags_before_matching() -> None:
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "article section": 1,
            },
        }
    )

    passed, message = rule.run("<h1>article section</h1>")

    assert passed
    assert message == ""


def test_sentence_rules_normalize_encoded_html_tags_and_lt_entity() -> None:
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "&lt;h1&gt;value &lt; threshold&lt;/h1&gt;": 1,
            },
        }
    )

    passed, message = rule.run("&lt;div&gt;value &lt; threshold&lt;/div&gt;")

    assert passed
    assert message == ""


def test_sentence_rules_remove_multi_dot_runs_in_bag_and_content() -> None:
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "net income .. adjusted . . basis": 1,
            },
        }
    )

    passed, message = rule.run("net income adjusted basis")

    assert passed
    assert message == ""


def test_too_many_sentence_occurence_rule_fails_on_over_limit() -> None:
    rule = TooManySentenceOccurenceRule(
        {
            "type": "too_many_sentence_occurence",
            "bag_of_sentence": {
                "sentence one": 1,
            },
        }
    )

    passed, message = rule.run("sentence one. sentence one")

    assert not passed
    assert "sentence one" in message


def test_missing_sentence_rule_fails_on_under_limit() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "required sentence": 2,
            },
        }
    )

    passed, message = rule.run("required sentence")

    assert not passed
    assert "required sentence" in message


def test_missing_sentence_rule_matches_sentence_in_markdown_table_cell() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "required sentence": 1,
            },
        }
    )

    passed, message = rule.run("| header |\n|---|\n| required sentence |")

    assert passed
    assert message == ""


def test_missing_sentence_rule_splits_multiple_sentences_from_one_table_cell() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "first sentence": 1,
                "second sentence": 1,
            },
        }
    )

    md = "<table><tr><td>First sentence. Second sentence.</td></tr></table>"
    passed, message = rule.run(md)

    assert passed
    assert message == ""


def test_unexpected_sentence_percent_rule_returns_partial_score() -> None:
    rule = UnexpectedSentencePercentRule(
        {
            "type": "unexpected_sentence_percent",
            "bag_of_sentence": {
                "allowed sentence": 1,
            },
        }
    )

    passed, message, score = rule.run("Allowed sentence. Completely new sentence")

    assert not passed
    assert 0.0 < score < 1.0
    assert "score=" in message


def test_too_many_sentence_occurence_percent_rule_returns_zero_when_all_over_limit() -> None:
    rule = TooManySentenceOccurencePercentRule(
        {
            "type": "too_many_sentence_occurence_percent",
            "bag_of_sentence": {
                "sentence one": 0,
            },
        }
    )

    passed, message, score = rule.run("sentence one. sentence one")

    assert not passed
    assert score == 0.0
    assert "score=" in message


def test_missing_sentence_percent_rule_returns_partial_score() -> None:
    rule = MissingSentencePercentRule(
        {
            "type": "missing_sentence_percent",
            "bag_of_sentence": {
                "required sentence": 2,
                "other sentence": 1,
            },
        }
    )

    passed, message, score = rule.run("required sentence")

    assert not passed
    assert score == pytest.approx(1 / 3)
    assert "score=" in message


def test_sentence_percent_rules_treat_quoted_and_unquoted_sentence_as_same() -> None:
    md = '"ACM SIGPLAN Notices."\n"ACM SIGPLAN Notices."'
    bag = {"acm sigplan notices": 2}

    missing_rule = MissingSentencePercentRule(
        {
            "type": "missing_sentence_percent",
            "bag_of_sentence": bag,
        }
    )
    unexpected_rule = UnexpectedSentencePercentRule(
        {
            "type": "unexpected_sentence_percent",
            "bag_of_sentence": bag,
        }
    )

    missing_passed, missing_message, missing_score = missing_rule.run(md)
    unexpected_passed, unexpected_message, unexpected_score = unexpected_rule.run(md)

    assert missing_passed
    assert missing_message == ""
    assert missing_score == pytest.approx(1.0)

    assert unexpected_passed
    assert unexpected_message == ""
    assert unexpected_score == pytest.approx(1.0)


def test_missing_sentence_rule_handles_decimal_section_heading() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "10.1 General 296": 1,
                "PDF 32000-1:2008": 1,
            },
        }
    )

    section_line = f"10.1 General {'.' * 145} 296"
    passed, message = rule.run(
        f"""
        {section_line}
        PDF 32000-1:2008
        """
    )

    assert passed
    assert message == ""


def test_missing_sentence_rule_splits_on_question_and_exclamation_marks() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "are you ready": 1,
                "lets go": 1,
            },
        }
    )

    passed, message = rule.run("Are you ready? Lets go!")

    assert passed
    assert message == ""


def test_missing_sentence_rule_splits_on_cjk_ideographic_full_stop() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "这是第一句话": 1,
                "这是第二句话": 1,
            },
        }
    )

    passed, message = rule.run("这是第一句话。这是第二句话")

    assert passed
    assert message == ""


def test_missing_sentence_rule_splits_on_cjk_ideographic_comma() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "这是第一部分": 1,
                "这是第二部分": 1,
            },
        }
    )

    passed, message = rule.run("这是第一部分、这是第二部分")

    assert passed
    assert message == ""


def test_missing_sentence_rule_splits_on_fullwidth_comma() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "这是第一部分": 1,
                "这是第二部分": 1,
            },
        }
    )

    passed, message = rule.run("这是第一部分，这是第二部分")

    assert passed
    assert message == ""


def test_missing_sentence_rule_splits_on_fullwidth_exclamation_and_question() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "你好世界": 1,
                "你准备好了吗": 1,
            },
        }
    )

    passed, message = rule.run("你好世界！你准备好了吗？")

    assert passed
    assert message == ""


def test_missing_sentence_rule_splits_on_horizontal_ellipsis() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "first part here": 1,
                "second part here": 1,
            },
        }
    )

    passed, message = rule.run("first part here…second part here")

    assert passed
    assert message == ""


def test_missing_sentence_rule_splits_on_fullwidth_semicolon() -> None:
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "这是第一部分": 1,
                "这是第二部分": 1,
            },
        }
    )

    passed, message = rule.run("这是第一部分；这是第二部分")

    assert passed
    assert message == ""


def test_missing_specific_sentence_rule_matches_sentence_in_table_cell() -> None:
    rule = MissingSpecificSentenceRule(
        {
            "type": "missing_specific_sentence",
            "sentence": "required sentence",
        }
    )

    passed, message = rule.run("| header |\n|---|\n| required sentence |")

    assert passed
    assert message == ""


def test_missing_specific_sentence_rule_matches_sentence_split_across_table_cells() -> None:
    """Sentence text split across multiple cells in the same row should still match."""
    rule = MissingSpecificSentenceRule(
        {
            "type": "missing_specific_sentence",
            "sentence": "hello world foo bar",
        }
    )

    md = "| col1 | col2 |\n|---|---|\n| hello world | foo bar |"
    passed, message = rule.run(md)

    assert passed
    assert message == ""


def test_sentence_bag_rules_reject_invalid_bag_definition() -> None:
    with pytest.raises(ValueError, match="non-empty dictionary"):
        UnexpectedSentenceRule({"type": "unexpected_sentence", "bag_of_sentence": {}})

    with pytest.raises(ValueError, match="integer"):
        TooManySentenceOccurenceRule(
            {
                "type": "too_many_sentence_occurence",
                "bag_of_sentence": {"sentence": 1.2},
            }
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        MissingSentenceRule(
            {
                "type": "missing_sentence",
                "bag_of_sentence": {"sentence": -1},
            }
        )


def test_sentence_bag_rule_skips_invalid_noise_entries_in_bag() -> None:
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "..": 2,
                ". . .": 3,
                "allowed sentence": 1,
            },
        }
    )

    passed, message = rule.run("allowed sentence")

    assert passed
    assert message == ""


def test_sentence_bag_rule_executes_with_empty_normalized_bag() -> None:
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "..": 1,
                ". .": 1,
                "tiny": 1,
            },
        }
    )

    passed, message = rule.run("this sentence would normally be unexpected")

    assert not passed
    assert "this sentence would normally be unexpected" in message


def test_unexpected_word_rule_fails_on_unknown_word() -> None:
    rule = UnexpectedWordRule(
        {
            "type": "unexpected_word",
            "bag_of_word": {
                "allowed": 5,
                "word": 5,
            },
        }
    )

    passed, message = rule.run("Allowed word and surprise")

    assert not passed
    assert "surprise" in message


def test_unexpected_sentence_rule_reports_all_unexpected_sentences() -> None:
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "allowed sentence": 1,
            },
        }
    )

    passed, message = rule.run("Allowed sentence. Extra sentence one. Extra sentence two")

    assert not passed
    assert "extra sentence one" in message
    assert "extra sentence two" in message


def test_unexpected_word_rule_reports_all_unexpected_words() -> None:
    rule = UnexpectedWordRule(
        {
            "type": "unexpected_word",
            "bag_of_word": {
                "allowed": 5,
                "words": 5,
            },
        }
    )

    passed, message = rule.run("allowed words surprise mystery")

    assert not passed
    assert "surprise" in message
    assert "mystery" in message


def test_too_many_word_occurence_rule_fails_on_over_limit() -> None:
    rule = TooManyWordOccurenceRule(
        {
            "type": "too_many_word_occurence",
            "bag_of_word": {
                "total": 1,
            },
        }
    )

    passed, message = rule.run("total total")

    assert not passed
    assert "total" in message


def test_word_rules_ignore_words_shorter_than_two_characters() -> None:
    rule = UnexpectedWordRule(
        {
            "type": "unexpected_word",
            "bag_of_word": {
                "allowed": 5,
                "words": 5,
                "ok": 5,
                "big": 5,
            },
        }
    )

    # Single-char words like "a" and "I" are ignored because they are shorter than 2 chars.
    # "ok" (2 chars) and "big" (3 chars) now pass the MIN_WORD_LENGTH=2 filter.
    passed, message = rule.run("allowed words ok big a I")

    assert passed
    assert message == ""


def test_word_rule_skips_invalid_markdown_noise_in_bag() -> None:
    rule = UnexpectedWordRule(
        {
            "type": "unexpected_word",
            "bag_of_word": {
                "##": 3,
                "allowed": 1,
            },
        }
    )

    passed, message = rule.run("allowed")

    assert passed
    assert message == ""


def test_word_rules_strip_html_tags_in_bag_and_content() -> None:
    rule = UnexpectedWordRule(
        {
            "type": "unexpected_word",
            "bag_of_word": {
                "<b>allowed</b>": 1,
            },
        }
    )

    passed, message = rule.run("<span>allowed</span>")

    assert passed
    assert message == ""


def test_word_rules_strip_encoded_html_tags_in_bag_and_content() -> None:
    rule = UnexpectedWordRule(
        {
            "type": "unexpected_word",
            "bag_of_word": {
                "&lt;b&gt;allowed&lt;/b&gt;": 1,
            },
        }
    )

    passed, message = rule.run("&lt;span&gt;allowed&lt;/span&gt;")

    assert passed
    assert message == ""


def test_missing_word_rule_fails_on_under_limit() -> None:
    rule = MissingWordRule(
        {
            "type": "missing_word",
            "bag_of_word": {
                "required": 2,
            },
        }
    )

    passed, message = rule.run("required")

    assert not passed
    assert "required" in message


def test_missing_word_rule_matches_word_in_markdown_table_cell() -> None:
    rule = MissingWordRule(
        {
            "type": "missing_word",
            "bag_of_word": {
                "required": 1,
            },
        }
    )

    passed, message = rule.run("| item |\n|---|\n| required |")

    assert passed
    assert message == ""


def test_missing_specific_word_rule_matches_word_in_table_cell() -> None:
    rule = MissingSpecificWordRule(
        {
            "type": "missing_specific_word",
            "word": "required",
        }
    )

    passed, message = rule.run("<table><tr><td>required</td></tr></table>")

    assert passed
    assert message == ""


def test_missing_specific_word_rule_keeps_words_around_currency_amounts() -> None:
    md = (
        "In 2023, our net income was $15.00 billion, representing a favorable change of $2.44 "
        "billion. This included a one-time non-cash tax benefit of $5.93 billion for the release "
        "of valuation allowance on certain deferred tax assets. "
        "Our cash flows provided by operating activities in 2023 and 2022 were $13.26 billion and "
        "$14.72 billion, respectively."
    )

    for word in ["assets", "activities", "2022", "2023"]:
        rule = MissingSpecificWordRule(
            {
                "type": "missing_specific_word",
                "word": word,
            }
        )

        passed, message = rule.run(md)

        assert passed
        assert message == ""


def test_missing_specific_word_rule_still_fails_for_absent_word() -> None:
    md = "valuation allowance on certain deferred tax assets"
    rule = MissingSpecificWordRule(
        {
            "type": "missing_specific_word",
            "word": "allowances",
        }
    )

    passed, message = rule.run(md)

    assert not passed
    assert "allowances" in message


def test_missing_specific_word_rule_handles_apostrophe_word() -> None:
    """Words with apostrophes like d'équipage should not raise during init."""
    rule = MissingSpecificWordRule(
        {
            "type": "missing_specific_word",
            "word": "d'équipage",
            "page": 1,
        }
    )
    assert rule.normalized_word == "equipage"

    passed, _ = rule.run("Les membres d'équipage sont présents.")
    assert passed

    passed, msg = rule.run("Some unrelated content.")
    assert not passed
    assert "equipage" in msg


def test_missing_specific_word_rule_handles_contraction() -> None:
    """Contractions like can't should pick the longest valid fragment ('can') with MIN_WORD_LENGTH=2."""
    rule = MissingSpecificWordRule(
        {
            "type": "missing_specific_word",
            "word": "can't",
            "page": 1,
        }
    )
    assert rule.normalized_word == "can"

    passed, _ = rule.run("You can't do that.")
    assert passed

    passed, msg = rule.run("Some unrelated content.")
    assert not passed
    assert "can" in msg


def test_unexpected_word_percent_rule_returns_partial_score() -> None:
    rule = UnexpectedWordPercentRule(
        {
            "type": "unexpected_word_percent",
            "bag_of_word": {
                "allowed": 5,
                "words": 5,
            },
        }
    )

    passed, message, score = rule.run("allowed words surprise mystery")

    assert not passed
    assert 0.0 < score < 1.0
    assert "score=" in message


def test_too_many_word_occurence_percent_rule_returns_zero_when_all_over_limit() -> None:
    rule = TooManyWordOccurencePercentRule(
        {
            "type": "too_many_word_occurence_percent",
            "bag_of_word": {
                "total": 0,
            },
        }
    )

    passed, message, score = rule.run("total total")

    assert not passed
    assert score == 0.0
    assert "score=" in message


def test_missing_word_percent_rule_returns_partial_score() -> None:
    rule = MissingWordPercentRule(
        {
            "type": "missing_word_percent",
            "bag_of_word": {
                "required": 2,
                "other": 1,
            },
        }
    )

    passed, message, score = rule.run("required")

    assert not passed
    assert score == pytest.approx(1 / 3)
    assert "score=" in message


def test_word_bag_rules_reject_invalid_bag_definition() -> None:
    with pytest.raises(ValueError, match="non-empty dictionary"):
        UnexpectedWordRule({"type": "unexpected_word", "bag_of_word": {}})

    with pytest.raises(ValueError, match="integer"):
        TooManyWordOccurenceRule(
            {
                "type": "too_many_word_occurence",
                "bag_of_word": {"word": 1.2},
            }
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        MissingWordRule(
            {
                "type": "missing_word",
                "bag_of_word": {"word": -1},
            }
        )


# ---- substring fallback tests ----


def test_missing_sentence_rule_finds_sentence_via_substring_fallback() -> None:
    """A sentence split across line boundaries by the parser should still be found
    via substring matching in the normalized full text."""
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "the quick brown fox jumps over the lazy dog": 1,
            },
        }
    )
    # The sentence split by "." so bag extraction splits it differently,
    # but substring matching in normalized full text should still find it.
    md = "The quick brown fox jumps\nover the lazy dog"
    passed, message = rule.run(md)

    assert passed, f"Expected pass via substring fallback, got: {message}"


def test_missing_sentence_rule_still_fails_when_truly_missing() -> None:
    """Substring fallback should not produce false positives."""
    rule = MissingSentenceRule(
        {
            "type": "missing_sentence",
            "bag_of_sentence": {
                "this sentence is not in the content": 1,
            },
        }
    )
    passed, message = rule.run("Completely unrelated content here instead")

    assert not passed


def test_too_many_sentence_rule_uses_substring_count() -> None:
    """Substring fallback should detect repeated occurrences even when
    bag extraction splits them differently."""
    rule = TooManySentenceOccurenceRule(
        {
            "type": "too_many_sentence_occurence",
            "bag_of_sentence": {
                "repeated sentence here": 1,
            },
        }
    )
    # Two occurrences separated by newline — bag splitting may merge or split differently
    md = "Repeated sentence here\nsome filler text\nRepeated sentence here"
    passed, message = rule.run(md)

    assert not passed, "Expected failure for over-limit, got pass"


def test_unexpected_sentence_rule_accepts_substring_of_bag_entry() -> None:
    """An actual sentence fragment that is a substring of a bag entry should not
    be flagged as unexpected (handles boundary misalignment)."""
    rule = UnexpectedSentenceRule(
        {
            "type": "unexpected_sentence",
            "bag_of_sentence": {
                "the quick brown fox jumps over the lazy dog": 1,
            },
        }
    )
    # "quick brown fox jumps over the lazy dog" is a sub-piece of the bag entry
    md = "Quick brown fox jumps over the lazy dog"
    passed, message = rule.run(md)

    assert passed, f"Expected pass for substring of bag entry, got: {message}"


def test_missing_word_rule_finds_word_via_substring_fallback() -> None:
    """Word-boundary substring matching should find words that tokenization may miss."""
    rule = MissingWordRule(
        {
            "type": "missing_word",
            "bag_of_word": {
                "revenue": 2,
            },
        }
    )
    # Content has "revenue" twice but in contexts that might tokenize differently
    md = "Total revenue was high. The revenue grew."
    passed, message = rule.run(md)

    assert passed, f"Expected pass, got: {message}"


def test_missing_word_percent_rule_uses_substring_fallback() -> None:
    """MissingWordPercentRule should use substring fallback for higher accuracy."""
    rule = MissingWordPercentRule(
        {
            "type": "missing_word_percent",
            "bag_of_word": {
                "revenue": 1,
                "expenses": 1,
            },
        }
    )
    md = "Total revenue and total expenses"
    passed, message, score = rule.run(md)

    assert passed
    assert score == 1.0


def test_too_many_word_rule_uses_substring_count() -> None:
    """TooManyWordOccurenceRule should detect over-limit via substring matching."""
    rule = TooManyWordOccurenceRule(
        {
            "type": "too_many_word_occurence",
            "bag_of_word": {
                "revenue": 1,
            },
        }
    )
    md = "Revenue from sales. Total revenue. Net revenue."
    passed, message = rule.run(md)

    assert not passed, "Expected failure for over-limit, got pass"


# ---- bag_of_digit_percent rule tests ----


def test_bag_of_digit_percent_rule_perfect_match() -> None:
    """All expected digits are present with correct counts → score 1.0."""
    rule = BagOfDigitPercentRule(
        {
            "type": "bag_of_digit_percent",
            "bag_of_digit": {"1": 2, "5": 1, "0": 1},
        }
    )

    passed, message, score = rule.run("Revenue was 150 and cost was 11")

    assert passed
    assert score == 1.0


def test_bag_of_digit_percent_rule_partial_match() -> None:
    """Some digits are missing → score between 0 and 1."""
    rule = BagOfDigitPercentRule(
        {
            "type": "bag_of_digit_percent",
            "bag_of_digit": {"1": 3, "5": 2, "0": 1},
        }
    )

    # Content has: 1 appears 2x, 5 appears 1x, 0 appears 1x → matched=4 / expected=6
    passed, message, score = rule.run("15 and 10")

    assert not passed
    assert 0.0 < score < 1.0
    assert "score=" in message


def test_bag_of_digit_percent_rule_no_digits_in_content() -> None:
    """No digits in content → score 0.0."""
    rule = BagOfDigitPercentRule(
        {
            "type": "bag_of_digit_percent",
            "bag_of_digit": {"3": 2, "7": 1},
        }
    )

    passed, message, score = rule.run("No digits here at all")

    assert not passed
    assert score == 0.0


def test_bag_of_digit_percent_rule_ignores_html_attributes() -> None:
    """Digits in HTML tag attributes (e.g. colspan='2') must NOT be counted."""
    rule = BagOfDigitPercentRule(
        {
            "type": "bag_of_digit_percent",
            "bag_of_digit": {"2": 1},
        }
    )

    # The only "2" is inside an HTML attribute — should NOT match
    passed, _, score = rule.run('<td colspan="2">text</td>')

    assert not passed
    assert score == 0.0


def test_bag_of_digit_percent_rule_counts_digits_in_table_cell_text() -> None:
    """Digits inside table cell text content should be counted."""
    rule = BagOfDigitPercentRule(
        {
            "type": "bag_of_digit_percent",
            "bag_of_digit": {"4": 1, "2": 1},
        }
    )

    md = '<table><tr><td colspan="3">42</td></tr></table>'
    passed, _, score = rule.run(md)

    assert passed
    assert score == 1.0


def test_bag_of_digit_percent_rule_counts_digits_in_markdown_table() -> None:
    """Digits in markdown pipe tables should be counted."""
    rule = BagOfDigitPercentRule(
        {
            "type": "bag_of_digit_percent",
            "bag_of_digit": {"9": 1, "8": 1},
        }
    )

    md = "| Col A | Col B |\n|-------|-------|\n| 9 | 8 |"
    passed, _, score = rule.run(md)

    assert passed
    assert score == 1.0


def test_bag_of_digit_percent_rule_rejects_non_digit_keys() -> None:
    """Keys must be single digit characters (0-9)."""
    with pytest.raises(ValueError, match="single digit"):
        BagOfDigitPercentRule(
            {
                "type": "bag_of_digit_percent",
                "bag_of_digit": {"abc": 1},
            }
        )


def test_bag_of_digit_percent_rule_rejects_empty_bag() -> None:
    """Empty bag_of_digit is rejected."""
    with pytest.raises(ValueError, match="non-empty dictionary"):
        BagOfDigitPercentRule(
            {
                "type": "bag_of_digit_percent",
                "bag_of_digit": {},
            }
        )


def test_bag_of_digit_percent_rule_rejects_negative_count() -> None:
    """Negative counts are rejected."""
    with pytest.raises(ValueError, match="cannot be negative"):
        BagOfDigitPercentRule(
            {
                "type": "bag_of_digit_percent",
                "bag_of_digit": {"1": -1},
            }
        )


def test_bag_of_digit_percent_rule_complex_html_table() -> None:
    """Digits in cell content are counted, digits in HTML attrs are not."""
    rule = BagOfDigitPercentRule(
        {
            "type": "bag_of_digit_percent",
            "bag_of_digit": {"1": 2, "0": 2, "5": 1},
        }
    )

    md = '<table><tr><td rowspan="2" colspan="3">100</td><td>15</td></tr></table>'

    passed, _, score = rule.run(md)

    assert passed
    assert score == 1.0


# ---- New tests for bugfixes ----


def test_fuzzy_match_threshold_floor_for_short_query() -> None:
    """Short queries should have a threshold floor of 0.7 to avoid matching everything."""
    # "hi" with max_diffs=1 would give threshold=0.5 without floor — too permissive
    rule = TextPresenceRule({"type": "present", "text": "hi", "max_diffs": 1})
    # "zz" should NOT match "hi" even with fuzzy matching
    passed, _ = rule.run("zz completely different text")
    assert not passed


def test_latex_simplification_distinguishes_different_formulas() -> None:
    """Different LaTeX formulas should produce different simplified tokens."""
    from parse_bench.evaluation.metrics.parse.rules_base import _strip_and_replace_latex

    result = _strip_and_replace_latex(r"$\frac{a}{b}$ and $x^2 + y^2$")
    # Should contain distinguishable simplified forms, not just "LATEX" twice
    assert "LATEX" not in result or result.count("LATEX") == 0
    assert "ab" in result.lower() or "a" in result  # frac{a}{b} → contains a and b
    assert "x2" in result or "x" in result  # x^2 → contains x and 2


def test_sentence_substring_word_boundary_avoids_false_positives() -> None:
    """Short sentence substring matching should use word boundaries."""
    from parse_bench.evaluation.metrics.parse.rules_bag import SentenceBagRule

    # "the cat" should not match inside "other category"
    count = SentenceBagRule._count_sentence_in_full_text("the cat", "other category is here")
    assert count == 0

    # But should match when actually present
    count = SentenceBagRule._count_sentence_in_full_text("the cat", "the cat is here")
    assert count == 1


def test_word_rules_handle_cjk_characters() -> None:
    """CJK characters stay grouped as multi-char tokens (matching JS annotation tool)."""
    # Multi-char CJK bag entries should match as substring in content
    rule = MissingWordRule(
        {
            "type": "missing_word",
            "bag_of_word": {
                "\u4f60\u597d": 1,  # 你好
                "\u4e16\u754c": 1,  # 世界
            },
        }
    )

    passed, message = rule.run("你好世界")

    assert passed
    assert message == ""


def test_word_rules_handle_short_words_with_new_min_length() -> None:
    """Words with 2-3 characters should now be recognized (MIN_WORD_LENGTH=2)."""
    rule = MissingWordRule(
        {
            "type": "missing_word",
            "bag_of_word": {
                "ok": 1,
                "the": 1,
            },
        }
    )

    passed, message = rule.run("This is ok and the end")

    assert passed
    assert message == ""


def test_repetition_detection_catches_late_repetition() -> None:
    """Repetition detection should catch repetitive content anywhere in the document."""
    from parse_bench.evaluation.metrics.parse.rules_text import BaselineRule

    rule = BaselineRule({"type": "baseline"})

    # Normal start, then repetitive content filling the document
    content = "Normal start here. " + "x" * 5000
    passed, message = rule.run(content)

    assert not passed
    assert "repetitive" in message.lower()


def test_formatting_rule_tolerates_mark_tags_between_words() -> None:
    """Markup tolerance should handle <mark>, <sup>, <em> etc. between words."""
    rule = FormattingRule(
        {
            "type": "is_bold",
            "text": "hello world",
        }
    )

    # Words separated by a <mark> tag should still match
    passed, message = rule.run("**hello <mark>world</mark>**")

    assert passed
    assert message == ""


# ---------------------------------------------------------------------------
# MarkColorRule tests
# ---------------------------------------------------------------------------


def test_mark_color_rule_passes_with_style_attribute() -> None:
    rule = MarkColorRule({"type": "mark_color", "text": "important", "color": "yellow"})

    passed, message = rule.run('Some text <mark style="background-color: yellow">important</mark> here.')
    assert passed
    assert message == ""


def test_is_mark_rule_accepts_color_attributes_without_weakening_color_check() -> None:
    semantic_rule = FormattingRule({"type": "is_mark", "text": "important"})
    color_rule = MarkColorRule({"type": "mark_color", "text": "important", "color": "yellow"})
    markdown = '<mark style="background-color: yellow">important</mark>'

    assert semantic_rule.run(markdown)[0]
    assert color_rule.run(markdown)[0]


def test_mark_color_rule_passes_with_background_attribute() -> None:
    rule = MarkColorRule({"type": "mark_color", "text": "important", "color": "yellow"})

    passed, message = rule.run('Some text <mark background="yellow">important</mark> here.')
    assert passed
    assert message == ""


def test_mark_color_rule_passes_with_backgroundcolor_attribute() -> None:
    rule = MarkColorRule({"type": "mark_color", "text": "important", "color": "yellow"})

    passed, message = rule.run('Some text <mark backgroundColor="yellow">important</mark> here.')
    assert passed
    assert message == ""


def test_mark_color_rule_fails_when_wrong_color() -> None:
    rule = MarkColorRule({"type": "mark_color", "text": "important", "color": "yellow"})

    passed, message = rule.run('Some text <mark style="background-color: red">important</mark> here.')
    assert not passed
    assert "yellow" in message


def test_mark_color_rule_fails_when_no_mark_tag() -> None:
    rule = MarkColorRule({"type": "mark_color", "text": "important", "color": "yellow"})

    passed, message = rule.run("Some text important here.")
    assert not passed


def test_mark_color_rule_fails_when_mark_has_no_attributes() -> None:
    rule = MarkColorRule({"type": "mark_color", "text": "important", "color": "yellow"})

    passed, message = rule.run("Some text <mark>important</mark> here.")
    assert not passed


def test_mark_color_rule_case_insensitive_color() -> None:
    rule = MarkColorRule({"type": "mark_color", "text": "note", "color": "yellow"})

    passed, message = rule.run('Text <mark style="background-color: Yellow">note</mark>.')
    assert passed
    assert message == ""


def test_mark_color_rule_hex_color() -> None:
    rule = MarkColorRule({"type": "mark_color", "text": "highlighted", "color": "#ff0"})

    passed, message = rule.run('Text <mark style="background-color: #ff0">highlighted</mark>.')
    assert passed
    assert message == ""


def test_mark_color_rule_text_not_in_mark() -> None:
    """Text exists but not inside the colored mark tag."""
    rule = MarkColorRule({"type": "mark_color", "text": "other", "color": "yellow"})

    passed, message = rule.run('Before other <mark style="background-color: yellow">important</mark> after.')
    assert not passed


def test_mark_color_rule_with_nested_formatting() -> None:
    """Text inside mark tag has nested bold formatting."""
    rule = MarkColorRule({"type": "mark_color", "text": "hello world", "color": "green"})

    passed, message = rule.run('<mark style="background-color: green">**hello** world</mark>')
    assert passed
    assert message == ""
