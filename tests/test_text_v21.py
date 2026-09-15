"""Contract tests use source text and adversarial edits, not saved model scores."""

import pytest

from parse_bench.evaluation.metrics.parse.rules_base import create_test_rule
from parse_bench.evaluation.metrics.parse.text_v21 import delivered_markdown, reference_rules, visible_text
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput


def scores(reference, actual):
    return {rule["type"]: create_test_rule(rule).run(actual)[2] for rule in reference_rules(reference)}


@pytest.mark.parametrize(
    "reference",
    [
        "delete_ability() rest_svc",
        "053_20090718白山祭り042\n054_20090718白山祭り043",
        "```python\nprint(value)\n```\n181",
        "<table><tr><td rowspan='2'>Alpha</td><td>20</td></tr><tr><td>30</td></tr></table>",
        "| Label | Value |\n| --- | --- |\n| Alpha | 20 |",
        "<본 문서는 외부 비공개 문서입니다>\nD < 4 and MVP > 2\nMVP",
        "[Visible label](https://example.org/not-printed) ![invented narration](crop.jpg)",
        "ಕನ್ನಡ भाषा ภาษาไทย 日本語 がぎぐ",
        "# Section one\n1. c\n2. a\n# Section two\n1. a\n2. c",
    ],
)
def test_exact_reference(reference):
    assert set(scores(reference, reference).values()) == {1.0}


def test_code_and_table_text_cannot_disappear():
    for reference in ["```python\nprint(value)\n```\nfooter", "<table><tr><td>Important</td></tr></table>\nfooter"]:
        assert scores(reference, "footer")["missing_word_percent"] < 1


def test_deletion_insertion_and_duplication_are_distinct():
    reference = "Alpha beta gamma.\nDelta epsilon zeta."
    assert scores(reference, "Alpha beta.")["missing_word_percent"] < 1
    assert scores(reference, reference + " hallucinated")["unexpected_word_percent"] < 1
    assert scores(reference, reference + "\nAlpha beta gamma.")["too_many_word_occurence_percent"] < 1
    assert scores(reference, reference + "\nAlpha beta gamma.")["too_many_sentence_occurence_percent"] < 1
    assert scores(reference, "")["missing_word_percent"] == 0


def test_answer_associations_are_not_a_bag_of_letters():
    result = scores("1. c\n2. a", "1. a\n2. c")
    assert result["missing_word_percent"] == 1
    assert result["missing_sentence_percent"] < 1


def test_cjk_wrapping_and_mixed_suffix_deletion():
    assert scores("健康保险的基本信息", "# 健康保险的基\n# 本信息")["missing_word_percent"] == 1
    assert scores("健康保险的基本信息", "健康保险的基信息")["missing_word_percent"] < 1
    assert scores("20090718白山祭り042", "20090718白山祭り")["missing_word_percent"] < 1


def test_images_exclude_alt_but_labels_and_literals_survive():
    text = visible_text("[Label](https://secret.invalid) ![Invented](photo.jpg)\n<본 문서>\nMVP < 3")
    assert "label" in text and "본 문서" in text and "mvp" in text
    assert "invented" not in text and "secret" not in text


def test_header_footer_folding_is_page_local_and_idempotent():
    output = ParseOutput(
        example_id="example",
        pipeline_name="test",
        markdown="Body",
        pages=[PageIR(page_index=0, markdown="Body")],
        layout_pages=[
            ParseLayoutPageIR(page_number=1, items=[], page_header_markdown="Header", page_footer_markdown="Footer")
        ],
    )
    folded = delivered_markdown(output.markdown, output)
    assert folded == "Header\n\nBody\n\nFooter"
    output.pages[0].markdown = folded
    assert delivered_markdown(folded, output) == folded
    rule = create_test_rule(reference_rules(folded)[0])
    rule.parse_output = output
    assert rule.run(folded)[2] == 1
    with pytest.raises(ValueError, match="consistent document and page"):
        rule.run(folded + " hallucination")


def test_legacy_dispatch_is_unchanged():
    rule = create_test_rule({"type": "missing_word_percent", "bag_of_word": {"alpha": 1}})
    assert type(rule).__name__ == "MissingWordPercentRule"
    with pytest.raises(ValueError):
        create_test_rule({"type": "present", "text": "alpha", "text_normalization": "text-v2.1"})
