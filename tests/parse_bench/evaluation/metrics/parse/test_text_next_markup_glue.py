"""text_next: markup that splits or hides a word on one side only reads like the plain text on the other."""

from __future__ import annotations

import random

from parse_bench.evaluation.metrics.parse.text_next import _long, score_text_next, tokenize, with_page_sections

random.seed(3)
_WORDS = "the board reviewed the annual plan and agreed several measures for regional offices during the year".split()
BODY = " ".join(random.choice(_WORDS) for _ in range(120)) + "."


def score(ref: str, out: str) -> float:
    """Score with a neutral 120-word body on both sides, so one event does not dominate the page."""
    return score_text_next(f"{BODY}\n\n{ref}", f"{BODY}\n\n{out}").score


def keys(md: str) -> list[str]:
    return [t.key for t in tokenize(md)[0]]


def test_ordinal_markup_and_spaced_date_suffix_do_not_split_the_number():
    assert score("this 26<sup>th</sup> day", "this <u>26</u><sup>th</sup> day") == 1.0
    assert score("April 22<sup>nd</sup>, 2020", "April <u>22</u><u><sup>nd</sup></u>, 2020") == 1.0
    assert score("on 31<sup>st</sup> March 2022", "on 31 st March 2022") == 1.0


def test_ordinal_changes_and_non_date_suffixes_remain_distinct():
    assert score("on 31st March 2022", "on 30 st March 2022") < 1.0
    assert score("this 26th day", "this <u>26</u><sup>nd</sup> day") < 1.0
    assert keys("31 St John Street") != keys("31st John Street")
    assert keys("10<sup>9</sup> cells") != keys("109 cells")


FOOTER = "http://medicalxpress.com/news/2012-07-hair-samples-infants-exposure-anti-hiv.html[12/14/2012 11:15:25 AM]"
AD_LINK = "[Ads by Google](http://g.com/url?u=http://medicalxpress.com/news/2012-07-hair-samples-infants-exposure-anti-hiv.html)"


def test_page_footer_found_only_in_a_link_target_or_image_alt_is_added():
    page = f"Story text about infants.\n\n{AD_LINK}"
    assert with_page_sections(page, [None], [FOOTER]).endswith(FOOTER)
    logo = "Body text.\n\n![icon: FASL Fondation pour l'animation socioculturelle lausannoise logo](p1.jpg)"
    footer = "Fondation pour l'animation socioculturelle lausannoise\nChemin de Malley 28, CH - 1007 Lausanne"
    assert with_page_sections(logo, [None], [footer]).endswith(footer)


def test_page_footer_printed_in_the_page_is_not_added_twice():
    url = FOOTER.split("[")[0]
    for page in (f"Story.\n\n<{url}>", f"Story.\n\n{url}"):
        assert with_page_sections(page, [None], [FOOTER]) == page


def test_superscript_after_closing_bold_or_holding_a_link_reads_like_the_reference():
    ref = "**Om Dobariya<sup>1</sup> and Akhil Kumar<sup>1</sup>** wrote"
    assert score(ref, "**Om Dobariya**<sup>1</sup> and **Akhil Kumar**<sup>1</sup> wrote") == 1.0
    linked = (
        "Qingliang Li<sup>[11](https://orcid.org/x)</sup> and Kelsey Ng<sup>[3,](https://orcid.org/y)42</sup> wrote"
    )
    assert score("Qingliang Li<sup>11</sup> and Kelsey Ng<sup>3,42</sup> wrote", linked) == 1.0


def test_wrong_affiliation_digit_after_closing_bold_is_still_charged():
    ref = "**Om Dobariya<sup>1</sup> and Akhil Kumar<sup>1</sup>** wrote"
    assert score(ref, "**Om Dobariya**<sup>2</sup> and **Akhil Kumar**<sup>1</sup> wrote") < 0.9


def test_inline_tag_inside_a_word_does_not_split_it():
    assert score("the noncore gas market", "the <u>n</u>oncore gas <u>marke</u>t") == 1.0
    span = 'See https://example.<span style="color:#1">com/abc/</span> for details'
    assert score("See https://example.com/abc/ for details", span) == 1.0
    assert score("the noncore gas market", "the <u>m</u>oncore gas market") < 1.0


def test_attributed_tag_inside_a_word_joins_only_when_inline():
    assert keys('hel<span style="color:red">lo</span>world') == ["helloworld"]
    assert keys('Hello<p class="x">World</p>') == ["hello", "world"]


def test_footnote_after_closing_bold_or_underscores_splits_from_the_name():
    assert keys("**Alex**<sup>[1](https://orcid.org/x)</sup> wrote") == ["alex", "#1", "wrote"]
    assert keys("__init__<sup>1</sup>") == ["init", "#1"]


def test_inline_tags_between_two_numbers_keep_them_apart():
    assert score("page 103 101 end", "page <u>103</u><s>101</s> end") == 1.0


def test_url_glued_after_a_colon_is_one_url():
    ref = "see repository: https://a.example.org/id/1/ ok"
    assert score(ref, "see repository:https://a.example.org/id/1/ ok") == 1.0
    assert score(ref, "see repository:https://a.example.org/id/7/ ok") < 0.9
    assert keys("Ref:www.example.org/a") == ["ref", "#www.example.org/a"]


def test_chinese_alt_caption_printed_as_text_is_free():
    caption = "昨天，广西百色芒果推介会在新发地市场举行。推介会展台上，百色市田阳香芒等13个品种亮相。"
    assert score(f"![{caption}](image)", caption) == 1.0
    assert score(f"![{caption}](image)", "") == 1.0  # an undescribed picture is free


def test_cjk_alt_is_long_only_past_eight_characters():
    assert not _long("一二三四五六七八")
    assert _long("一二三四五六七八九")


def test_times_sign_glued_to_numbers_reads_like_a_spaced_one():
    ref = "count 3.2×10<sup>9</sup> cells"
    assert score(ref, "count 3.2 × 10<sup>9</sup> cells") == 1.0
    assert score(ref, "count 3.2 × 10<sup>8</sup> cells") < 0.9
    assert keys("3×3 grid") == keys("3 × 3 grid") == ["#3", "#×", "#3", "grid"]


# --- near misses: shapes that look like the patterns above and must keep their old reading ---


def test_block_and_unlisted_tags_inside_a_word_stay_a_space():
    assert keys("foo<br>bar") == ["foo", "bar"]
    assert keys("foo<ul><li>bar</li></ul>") == ["foo", "bar"]
    assert keys('foo<img src="x">bar') == ["foo", "bar"]
    assert keys('x <span style="c">y</span> z') == ["x", "y", "z"]


def test_superscript_after_an_opening_bold_stays_with_the_next_word():
    assert keys("text **<sup>1</sup>bold** x") == ["text", "#1bold", "x"]
    assert keys("***Alex***<sup>1</sup>, Bo") == keys("**Alex<sup>1</sup>**, Bo") == ["alex", "#1", "bo"]


def test_colon_glue_only_splits_before_a_url():
    assert keys("at 10:30 today") == ["at", "#10:30", "today"]
    assert keys("isbn:978-3-16") == ["isbn", "#978-3-16"]


def test_letter_x_between_numbers_is_not_the_times_sign():
    assert keys("3x3 grid") == ["#3x3", "grid"]


def test_page_footer_hidden_in_an_html_comment_is_added():
    page = "Body <!-- Footer text that is hidden --> x"
    assert with_page_sections(page, [None], ["Footer text that is hidden"]) != page


# --- review follow-ups: header/footer projection, glued tags next to digits, strikethrough, ×, alt length ---


def test_page_footer_with_markup_already_in_the_page_is_not_added_twice():
    cases = [
        (
            "Rulemaking text.\n\n<page_number>2</page_number>\n\nQRX042FY19370 000000512",
            "<page_number>2</page_number>\n\nQRX042FY19370 000000512",
        ),
        (
            "Exhibit text.\n\nNorthwind Mutual Company<br>2021 Small Group Rate Filing\n\n11",
            "Northwind Mutual Company<br>2021 Small Group Rate Filing<br><page_number>11</page_number>",
        ),
        ("Board text.\n\n556120 03-04-21\n\n<b>Schedule K (Form 12) 2021</b>", "<b>Schedule K (Form 12) 2021</b>"),
    ]
    for page, footer in cases:
        assert with_page_sections(page, [None], [footer]) == page


def test_page_number_footer_printed_as_a_body_line_is_not_added_twice():
    page = "Body text of the page.\n\n88"
    assert with_page_sections(page, [None], ["<page_number>88</page_number>"]) == page
    assert with_page_sections("Body text about 88 items.", [None], ["<page_number>88</page_number>"]) != (
        "Body text about 88 items."
    )


def test_short_footer_over_several_lines_is_found_line_by_line():
    page = "Body.\n\nAppendix A\n\n1"
    assert with_page_sections(page, [None], ["Appendix A\n\n<page_number>1</page_number>"]) == page
    assert with_page_sections("Body.\n\nAppendix A", [None], ["Appendix A\n\n<page_number>1</page_number>"]) != (
        "Body.\n\nAppendix A"
    )


def test_html_and_markdown_emphasis_after_a_superscript_score_alike():
    ref = "Ann Lee<sup>1</sup>*CNC Program, University of Utah*"
    assert score(ref, "Ann Lee<sup>1</sup><i>CNC Program, University of Utah</i>") == 1.0


def test_inline_tag_next_to_a_digit_reads_like_the_plain_text():
    assert keys("Table<span>3</span> shows") == keys("Table3 shows")
    assert keys("10<span>kg</span> bag") == keys("10kg bag")
    assert score("see Table3 here", "see Table<span>3</span> here") == 1.0
    assert keys("H<span>2</span>O and v<b>2</b> and A<b>1</b>") == keys("H2O and v2 and A1")
    assert keys("page <u>103</u><b>101</b> end") == ["page", "#103", "#101", "end"]


def test_markdown_emphasis_inside_a_word_scores_like_the_html_tag():
    pairs = [
        ("see 1*CNC* here", "see 1<i>CNC</i> here"),
        ("water H*2*O here", "water H<i>2</i>O here"),
        ("see Table**3** here", "see Table<b>3</b> here"),
        ("bag 10*kg* here", "bag 10<em>kg</em> here"),
        ("the n*o*ncore gas", "the n<u>o</u>ncore gas"),
    ]
    for md, html in pairs:
        assert keys(md) == keys(html)
        assert score(md, html) == score(html, md) == 1.0
    assert score("see 1*CNC* here", "see 1<i>CMC</i> here") < 1.0


def test_stars_inside_code_stay_literal_while_the_text_around_still_joins():
    assert keys("water `H*2*O` and H*2*O here") == ["water", "h", "#2", "o", "and", "#h2o", "here"]
    assert keys("water ``H*2*O`` here") != keys("water H2O here")
    for fence in ("```", "~~~"):
        assert keys(f"x\n{fence}py\nH*2*O\n{fence}\ny") != keys(f"x\n{fence}py\nH2O\n{fence}\ny")
    assert score("see `H*2*O` here", "see `H2O` here") < 1.0


def test_lone_or_spaced_star_and_intraword_underscore_are_not_emphasis():
    assert keys("x a*b y") != keys("x ab y")
    assert keys("the O*NET data and **O**utlook") == ["the", "o", "net", "data", "and", "outlook"]
    assert keys("the O*NET data (*i.e.*, tasks)") == ["the", "o", "net", "data", "ie", "tasks"]
    assert keys("a * b * c") == ["a", "b", "c"]
    assert keys("* item one*two") != keys("item onetwo")
    assert keys("snake_case_name") == ["snake", "case", "name"]


def test_html_strikethrough_reads_like_markdown_strikethrough():
    assert keys("15-170-<s>GMS</s> filed") == keys("15-170-~~GMS~~ filed")
    assert score("case 15-170-~~GMS~~ filed", "case 15-170-<del>GMS</del> filed") == 1.0


def test_dropped_times_sign_is_charged_like_a_dropped_equals_sign():
    ref = "count 5×10<sup>9</sup> cells at 1920×1080"
    assert score(ref, "count 5 × 10<sup>9</sup> cells at 1920 × 1080") == 1.0
    assert score(ref, "count 5 10<sup>9</sup> cells at 1920 1080") < 1.0
    assert keys("× Not covered: 5 × 10 units") == ["not", "covered", "#5", "#×", "#10", "units"]
    assert keys(r"brine (2$\times$100 mL)") == keys("brine (2×100 mL)")


def test_english_alt_text_counts_whitespace_words():
    assert _long("ACME Logo - ACME The bank for every town")  # 9 words with the dash, as on main
    assert not _long("ACME Logo ACME The bank for every town")


def test_page_number_footer_is_found_after_a_less_than_sign_in_the_page():
    page = "Values p < 0.05 were kept.\n\nThe motifs in a-\n\n<page_number>6/179</page_number>"
    assert with_page_sections(page, [None], ["<page_number>6/179</page_number>"]) == page


def test_footer_whose_lines_the_page_prints_apart_is_not_added_twice():
    page = "News body.\n\nNews code: C-150302011022\n\n<page_number>Page: 1/1</page_number>\n\n![logo](l.png)\n\n"
    page += "Example Wire Ltd, 412/9 Harbor Plaza, Springfield 20450"
    footer = "News code: C-150302011022\n\n<page_number>Page: 1/1</page_number>\n\nExample Wire Ltd, 412/9 Harbor"
    assert with_page_sections(page, [None], [footer]) == page
    partly = "Example Wire Ltd, 412/9 Harbor\n\nPrinted 2015-03-02 in Springfield\n\nNews code: C-150302011022"
    assert with_page_sections(page, [None], [partly]) != page
