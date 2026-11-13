"""Tests for the text fidelity metric (``text_next``)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from next_wiring import NextScorers

from parse_bench.evaluation.metrics.parse.text_next import (
    output_hypothesis,
    reference_markdown,
    score_text_next,
    tokenize,
    with_page_sections,
)
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import ParseTestCase

PROSE = (
    "The committee met on Tuesday to review the annual budget. Members agreed that the library "
    "should extend its opening hours during the winter months. The proposal was approved after a "
    "short debate and will take effect next year."
)


def keys(md: str) -> list[str]:
    return [t.key for t in tokenize(md)[0]]


def crit(md: str) -> list[str]:
    return [t.key for t in tokenize(md)[0] if t.crit]


def long_page(paragraphs: int) -> str:
    return "\n\n".join(
        f"Paragraph {chr(65 + i % 26)} explains how the regional office handled the request from the "
        f"local council and why the decision was postponed until the spring meeting of the board."
        for i in range(paragraphs)
    )


# --- tokenization -----------------------------------------------------------------------------------------------


def test_page_tags_become_line_breaks_and_count_as_content():
    toks = keys("<page_header>Annual Report</page_header>Body text here.<page_number>12</page_number>")
    assert toks == ["annual", "report", "body", "text", "here", "#12"]
    lines = tokenize("<page_header>Annual Report</page_header>Body")[1]
    assert "Annual Report" in lines and "Body" in lines


def test_latex_is_unwrapped_but_plain_dollar_amounts_survive():
    assert keys(r"$x^{2} + y$") == ["x", "#2", "y"]
    assert keys(r"$\mathrm{CO}_2$ levels") == ["co", "#2", "levels"]
    # an inline $..$ without TeX syntax is money, not math
    assert crit("from $5 to $10") == ["#$", "#5", "#$", "#10"]


def test_superscripts_tags_and_footnote_markers_split_from_words():
    # <sup> and Unicode superscripts read the same; a short stem stays one identifier
    assert keys("E = mc<sup>2</sup>") == keys("E = mc²") == ["e", "#mc2"]
    assert keys("results<sup>3</sup> show") == keys("results³ show") == ["results", "#3", "show"]
    assert keys("results$^{3}$ show") == ["results", "#3", "show"]
    assert keys("protein12 binds") == ["protein", "#12", "binds"]


def test_units_glued_to_numbers_are_split_and_critical():
    # a unit right after a number is part of the quantity (critical); the same letters elsewhere are words
    assert keys("5mg daily") == keys("5 mg daily") == ["#5", "#mg", "daily"]
    assert keys("3.5kWh") == keys("3.5 kWh") == ["#3.5", "#kwh"]
    assert keys("kg of rice") == ["kg", "of", "rice"]
    assert crit("grew 12% in 2024") == ["#12", "#%", "#2024"]


def test_figure_descriptions_are_ignored():
    assert keys("![Figure 3: a bar chart of sales](fig.png) Sales rose.") == ["sales", "rose"]
    assert keys("[Image: company logo] Welcome") == ["welcome"]


def test_negators_are_critical_but_number_signs_are_not():
    toks = {t.key: t.crit for t in tokenize("This is not allowed. No. 5 is fine.")[0]}
    assert toks["not"] is True
    assert toks["no"] is False  # "No. 5" is a number sign
    assert [t.key for t in tokenize("you don't stop")[0] if t.crit] == ["t"]
    assert keys("N° 12") == keys("No. 12")


def test_accents_and_homoglyphs():
    # Cyrillic look-alikes fold to Latin keys; accents are kept in keys (scored as minor events)
    assert keys("соре") == ["cope"]
    assert keys("café") == ["café"]
    res = score_text_next(PROSE + " The café opens daily.", PROSE + " The cafe opens daily.")
    assert res.events.critical == 0
    assert res.events.kinds().get("accent only") == 1
    glyph = score_text_next(PROSE, PROSE.replace("committee", "cоmmittee"))
    assert glyph.words == pytest.approx(1.0)
    assert glyph.events.kinds().get("look-alike characters") == 1


def test_thousands_separators_and_identifiers():
    assert crit("Total: 1,234,567.89 USD") == ["#1234567.89"]
    assert crit("Call 555-123-4567 or mail info@example.com") == ["#555-123-4567", "#info@example.com"]


# --- scores -----------------------------------------------------------------------------------------------------


def test_identical_page_scores_one():
    res = score_text_next(PROSE, PROSE)
    assert res.score == pytest.approx(1.0)
    assert (res.words, res.integrity, res.order) == (1.0, 1.0, 1.0)


def test_empty_output_scores_zero():
    assert score_text_next(PROSE, "").score == 0.0


def test_one_wrong_number_hits_short_pages_harder_than_long_ones():
    short_ref = "Invoice 4821 is due on 12 March 2024 for a total of 950 dollars."
    short = score_text_next(short_ref, short_ref.replace("950", "590"))
    long_ref = long_page(40) + "\n\n" + short_ref
    long = score_text_next(long_ref, long_ref.replace("950", "590"))
    assert short.events.critical == 1 and long.events.critical == 1
    # the floor charges one occurrence the same everywhere ...
    assert short.floor == pytest.approx(0.9) and long.floor == pytest.approx(0.9)
    # ... and the proportional part is harsher on the short page
    assert short.score < long.score <= 0.9


def test_repeated_errors_scale_down_on_long_pages():
    ref = long_page(60)
    hyp = ref.replace("regional", "rgional")  # 60 spelling slips
    res = score_text_next(ref, hyp)
    assert res.events.minor >= 60
    assert res.floor == pytest.approx(0.85)  # minor errors are capped


def test_column_swap_costs_mostly_the_order_lane():
    left = "\n".join(f"Left column sentence number {w} talks about apples and pears." for w in ("one", "two", "three"))
    right = "\n".join(
        f"Right column sentence number {w} talks about boats and trains." for w in ("four", "five", "six")
    )
    res = score_text_next(left + "\n\n" + right, right + "\n\n" + left)
    assert res.words == pytest.approx(1.0)
    assert res.critical is None or res.critical == pytest.approx(1.0)
    assert res.integrity == pytest.approx(1.0)
    assert res.order is not None and res.order < 1.0
    assert res.events.critical == 0
    assert res.score >= 0.9


def test_interleaved_columns_break_sentences():
    ref = "Alpha beta gamma delta epsilon zeta eta theta.\nOne two three four five six seven eight."
    hyp = "Alpha beta gamma delta One two three four\nepsilon zeta eta theta. five six seven eight."
    res = score_text_next(ref, hyp)
    assert res.integrity is not None and res.integrity < 1.0
    assert res.events.broken > 0


def test_flipped_negation_is_critical():
    ref = PROSE + " The board did not approve the merger."
    res = score_text_next(ref, ref.replace("did not approve", "did approve"))
    assert res.events.kinds().get("negator in changed text") == 1
    assert (res.events.critical, res.events.major) == (1, 1)  # flipped meaning + the omitted word
    assert res.floor == pytest.approx(0.85)


@pytest.mark.parametrize(
    "dropped",
    ["not, without the prior ", "not, without the prior written consent of the landlord, "],
    ids=["4-token run", "9-token run"],
)
def test_negator_dropped_inside_a_longer_omission_is_still_critical(dropped: str):
    clause = "The tenant shall not, without the prior written consent of the landlord, sublet any part of the premises."
    ref = PROSE + " " + clause
    res = score_text_next(ref, ref.replace(dropped, ""))
    assert res.events.kinds().get("negator in changed text") == 1
    assert res.floor <= score_text_next(ref, ref.replace("shall not,", "shall,")).floor


def test_negator_dropped_with_its_whole_sentence_is_only_an_omission():
    ref = PROSE + " Members did not object. The treasurer presented the plan."
    res = score_text_next(ref, ref.replace(" Members did not object.", ""))
    assert "negator in changed text" not in res.events.kinds()


def test_invented_passage_is_critical():
    hyp = PROSE + " Quarterly revenue reached 48 million euros across seven new markets."
    res = score_text_next(PROSE, hyp)
    assert res.events.kinds().get("invented passage") == 1
    assert res.events.critical >= 2  # the passage plus its invented numbers (capped)


# --- hypothesis assembly ----------------------------------------------------------------------------------------


def test_page_sections_added_only_when_missing():
    md = "ACME Annual Report\n\nBody text."
    assert with_page_sections(md, ["ACME Annual Report"], ["Page 3"]) == md + "\n\nPage 3"
    assert with_page_sections("Body.", ["Header", ""], [None]) == "Header\n\nBody."


def test_output_hypothesis_multi_page_places_sections_per_page():
    out = ParseOutput(
        example_id="g/d",
        pipeline_name="p",
        markdown="First page.\n\nSecond page.",
        pages=[PageIR(page_index=0, markdown="First page."), PageIR(page_index=1, markdown="Second page.")],
        layout_pages=[
            ParseLayoutPageIR(page_number=1, width=1, height=1, page_footer_markdown="Foot 1"),
            ParseLayoutPageIR(page_number=2, width=1, height=1, page_header_markdown="Head 2"),
        ],
    )
    assert output_hypothesis(out) == "First page.\n\nFoot 1\n\nHead 2\n\nSecond page."


def test_reference_prefers_text_rule_original_md():
    class Rule:
        type = "unexpected_sentence_percent"
        original_md = "orig"

    assert reference_markdown("expected", [Rule()]) == "orig"
    assert reference_markdown("expected", None) == "expected"


# --- evaluator wiring -------------------------------------------------------------------------------------------


def _inference_result(output: ParseOutput) -> InferenceResult:
    return InferenceResult(
        request=InferenceRequest(example_id="grp/doc", source_file_path="doc.pdf", product_type=ProductType.PARSE),
        pipeline_name="test_pipeline",
        product_type=ProductType.PARSE,
        raw_output={},
        output=output,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        latency_in_ms=1,
    )


def _case(metadata: dict | None) -> ParseTestCase:
    return ParseTestCase.model_construct(
        test_id="grp/doc",
        group="grp",
        file_path="doc.pdf",
        expected_markdown="<page_header>ACME Report</page_header>\n\n" + PROSE,
        metadata=metadata,
    )


def _output(markdown: str, header: str = "") -> ParseOutput:
    return ParseOutput(
        example_id="grp/doc",
        pipeline_name="test_pipeline",
        markdown=markdown,
        pages=[PageIR(page_index=0, markdown=markdown)],
        layout_pages=[ParseLayoutPageIR(page_number=1, width=1, height=1, page_header_markdown=header)],
    )


def test_wiring_emits_text_next_with_page_headers_as_content():
    evaluator = NextScorers()
    result = evaluator.evaluate(_inference_result(_output(PROSE, header="ACME Report")), _case({"scoring": "next"}))
    by_name = {m.metric_name: m for m in result.metrics}
    assert by_name["text_next"].value == pytest.approx(1.0)  # header from layout_pages counts as content
    for lane in ("words", "critical", "integrity", "order"):
        if lane != "critical":
            assert f"text_next_{lane}" in by_name
    assert set(by_name["text_next"].metadata["events"]) == {"critical", "major", "minor", "broken", "moved"}

    missing_header = evaluator.evaluate(_inference_result(_output(PROSE)), _case({"scoring": "next"}))
    assert {m.metric_name: m.value for m in missing_header.metrics}["text_next"] < 1.0


# --- regressions -------------------------------------------------------------------------------------------------


def test_cjk_number_marks_and_latin_words_do_not_swallow_cjk():
    # a CJK run with number marks was one word token, so one wrong kana cost the whole run
    assert keys("五八〇〇円でバラン") == ["五", "八", "〇", "〇", "円", "で", "バ", "ラ", "ン"]
    assert keys("Excel表格") == ["excel", "表", "格"]


def test_link_destinations_with_angle_brackets_and_parentheses_are_dropped():
    # a link destination in <...> with nested parentheses leaked "31127-9/ref0155" as text
    md = "[J. Example 33,](<http://example.org/S1234-5678(17)31127-9/ref0155>) 1123–1130."
    assert keys(md) == keys("J. Example 33, 1123–1130.")
    assert keys("[a](http://example.org/f(1).pdf) b") == ["a", "b"]


def test_all_html_entities_are_decoded():
    # "&#x26;" read as an inserted identifier "#x26"
    assert keys("Yoke (at right &#x26; left sides) &quot;A&quot; &nbsp;B") == keys('Yoke (at right & left sides) "A" B')


def test_inline_math_without_tex_commands_is_math_not_money():
    # "$n * m$" and "$1 : 5000$" left "$" currency tokens
    assert keys("a $n * m$ geometry") == ["a", "n", "m", "geometry"]
    assert keys("scale $1 : 5000$ map") == ["scale", "#1", "#5000", "map"]
    assert crit("from $5 to $10") == ["#$", "#5", "#$", "#10"]  # amounts stay money


def test_sup_tags_keep_the_space_before_them():
    # "Chapter 2\n<sup>2024 AB</sup>" read as "#22024"; "1990 <sup>25</sup>" as "#199025"
    assert keys("Chapter 2\n<sup>2024 AB</sup>") == ["chapter", "#2", "#2024", "ab"]
    assert keys("about 1990 <sup>25</sup> and") == ["about", "#1990", "#25", "and"]
    assert keys("E = mc<sup>2</sup>") == ["e", "#mc2"]  # glued when written glued


def test_transcribed_signatures_and_stamps_read_the_same_in_reference_and_output():
    # reference "[signature: Jane Q. Doe]" vs output "![signature: Jane Q. Doe](img.jpg)";
    # a bare "[signature]" placeholder was an omitted word "signature" on 30+ pages
    assert keys("[signature: Jane Q. Doe]") == keys("![signature: Jane Q. Doe](page_1_image_1.jpg)")
    assert keys("Podpis: [signature] Dátum") == keys("Podpis: ![signature](s.jpg) Dátum") == ["podpis", "dátum"]
    assert keys("![seal: FILED 2010 SEP](s.jpg)") == ["filed", "#2010", "sep"]
    # a description of the element is not its text
    assert keys("![seal: Example Bank official stamp and signature](s.jpg)") == []


def test_abbreviation_dots_do_not_split_words():
    # "U.S." vs "US" was 1 critical "name changed" per occurrence
    assert keys("U.S. economy") == keys("US economy") == ["us", "economy"]
    ref = PROSE + " The U.S. market grew."
    assert score_text_next(ref, ref.replace("U.S.", "US")).score == pytest.approx(1.0)


def test_unit_after_a_number_is_critical():
    ref = PROSE + " The plant produced 12 kWh during the test."
    res = score_text_next(ref, ref.replace("kWh", "MWh"))
    assert res.events.kinds().get("wrong number or identifier") == 1
    assert res.score <= 0.9
    assert keys("L 194 and s. 22") == ["l", "#194", "and", "s", "#22"]  # single letters stay words


def test_number_written_with_other_separators_is_minor():
    # "723-3600" vs "723・3600", "23. 2. 2021" vs "23.2.2021", "11-15" vs "11- 15"
    for ref_num, out_num in (("723-3600", "723 3600"), ("23. 2. 2021", "23.2.2021"), ("4.00 pm", "4:00 pm")):
        ref = PROSE + f" Call {ref_num} today."
        res = score_text_next(ref, ref.replace(ref_num, out_num))
        assert res.events.critical == 0, (ref_num, res.events.examples)
        assert res.score >= 0.95
    # a lost decimal point or merged numbers are still wrong values
    ref = PROSE + " The dose is 1.5 mg daily and pages 105 103 apply."
    res = score_text_next(ref, ref.replace("1.5", "15").replace("105 103", "105103"))
    assert res.events.kinds().get("wrong number or identifier") == 2


def test_words_split_or_joined_are_minor_not_name_changes():
    # "R. M. de" vs "RMde", "non tariff" vs "nontariff" (was a flipped negation)
    ref = PROSE + " It uses non tariff barriers in the region."
    res = score_text_next(ref, ref.replace("non tariff", "nontariff"))
    assert res.events.kinds() == {"words split or joined": 1}
    assert res.events.critical == 0 and res.score >= 0.95
    # "Māori" written "M aori" was a critical name change on every occurrence
    ref = PROSE + " The Māori council met."
    assert score_text_next(ref, ref.replace("Māori", "M aori")).events.kinds() == {"words split or joined": 1}


def test_short_footer_is_not_taken_as_already_in_the_page():
    # footer "12" was dropped because "12 months" appears in the body
    md = "Any waiver will stay on our website for a period of 12 months."
    assert with_page_sections(md, [""], ["12"]) == md + "\n\n12"
    assert with_page_sections("Intro\n\n12\n\nBody", [""], ["12"]) == "Intro\n\n12\n\nBody"
    long_header = "ACME Annual Report 2024 Strategic Review"
    assert with_page_sections(long_header + " and body text.", [long_header], []) == long_header + " and body text."


def test_systematic_repeats_are_charged_once():
    # the parser added the labels "Address:", "City:", ... to every contact card: 39 majors scored 0
    cards = [
        ("Mr Smith", "Chef du Service", "Cotonou", "Benin"),
        ("Ms Jones", "Senior Manager", "Gaborone", "Botswana"),
    ]
    cards += [("Mr Lee", "Head of Research", "Sofia", "Bulgaria"), ("Mr Guo", "Assistant Engineer", "Beijing", "China")]
    ref = "\n\n".join(f"{n}\n{t}\n{c}\n{k}" for n, t, c, k in cards)
    hyp = "\n\n".join(f"{n}\nTitle: {t}\nCity: {c}\nCountry: {k}" for n, t, c, k in cards)
    res = score_text_next(ref, hyp)
    assert res.events.kinds()["inserted words"] >= 8
    assert res.events.major <= 3  # one per distinct inserted label
    # list numbers omitted on every item: one quirk, not one major per number
    lines = [f"{i}) Line {i} of the pleading says something about the case." for i in range(1, 21)]
    res = score_text_next("\n".join(lines), "\n".join(x.split(" ", 1)[1] for x in lines))
    assert res.events.major == 1 and res.floor >= 0.95


def test_moved_blocks_are_counted_as_runs_and_ignore_single_tokens():
    # a trailing page number displaced to the top hid a real paragraph move (LIS counted single tokens)
    a, b, c = (long_page(1).replace("Paragraph A", f"Section {w}") for w in ("one", "two", "three"))
    res = score_text_next(f"{a}\n\n{b}\n\n{c}\n\n7", f"7\n\n{a}\n\n{c}\n\n{b}")
    assert res.events.moved == 1
    # a 4 x 5 photo grid read column by column: moves count column runs, not captions (was 14 of 20)
    things = "apples boats candles dancers eagles fountains gardens harbors islands jugglers kites lanterns"
    things += " markets needles orchards parades quilts rivers statues towers"
    caps = [f"Photo {i} shows the {w} near the hall" for i, w in enumerate(things.split())]
    by_rows = "\n\n".join(caps)
    by_cols = "\n\n".join(caps[r * 5 + c] for c in range(5) for r in range(4))
    grid = score_text_next(by_rows, by_cols)
    assert grid.events.moved <= 6
    # the first word of a caption caught by the previous caption's tile is a seam, not a cut sentence
    assert grid.integrity is not None and grid.integrity >= 0.5  # was 0.4


def test_interleaved_index_columns_cost_a_capped_order_penalty():
    # a two-column name index read row by row: every name keeps its page number, only the order is off
    names = "Abbott Baker Carter Dalton Ellis Foster Graham Harper Irving Jensen Keller Lawson Morgan Nolan"
    names += " Owens Parker Quinn Reyes Sutton Turner Vaughn Walsh Young Zimmer"
    entries = [f"{n}, Robert {3100 + i}" for i, n in enumerate(names.split())]
    half = len(entries) // 2
    ref = "\n".join(entries)
    rows = [f"| {entries[i]} | {entries[half + i]} |" for i in range(half)]
    res = score_text_next(ref, "\n".join(rows))
    assert (res.words, res.events.critical, res.events.major) == (1.0, 0, 0)
    assert res.events.moved > 6  # was floored to 0 by one major per move
    assert 0.6 <= res.score < 0.9
    # the cap covers order only: a wrong page number on top still costs its critical
    wrong = score_text_next(ref, "\n".join(rows).replace("3105", "3150"))
    assert wrong.score <= res.score - 0.09


def test_bare_numbers_at_sentence_ends_do_not_break_sentences():
    # margin line numbers printed apart from their lines broke 28 sentences
    ref = "\n".join(
        f"{i} The undersigned party number {w} declines to consent here."
        for i, w in enumerate(["one", "two", "three"], 1)
    )
    hyp = "1\n2\n3\n\n" + "\n".join(x.split(" ", 1)[1] for x in ref.split("\n"))
    res = score_text_next(ref, hyp)
    assert res.integrity == pytest.approx(1.0) and res.events.broken == 0


def test_long_omission_is_critical_and_does_not_flip_negation():
    extra = " Members did not object to the plan, which the treasurer presented with a short summary of costs."
    res = score_text_next(PROSE + extra, PROSE)
    assert res.events.kinds() == {"omitted passage": 1}
    assert res.events.critical == 1


def test_text_next_applies_only_to_running_text_references():
    from types import SimpleNamespace

    from parse_bench.evaluation.metrics.parse.text_next import text_next_applies

    assert text_next_applies(PROSE, [], {"scoring": "next"})
    assert not text_next_applies("<table><tr><td>1</td></tr></table>", [], {"scoring": "next"})
    assert not text_next_applies("| a | b |\n|---|---|\n| 1 | 2 |", [], None)
    assert not text_next_applies(PROSE, [], {"scoring": "next", "chart_tables": []})
    assert not text_next_applies(PROSE, [SimpleNamespace(type="chart_data_point")], None)


# --- adversarial parser failures: severity class targets (critical >= 10 pts, major >= 5, harmless 0) -----------

INVOICE = (
    "Policy number PL-0081-B covers the warehouse at 14 Dock Road. The renewal date is 2023-04-01 and the "
    "premium is due within thirty days of renewal."
)


@pytest.mark.parametrize(
    ("ref", "hyp", "min_drop", "max_drop"),
    [
        # a date with day and month swapped
        (PROSE + "\n\n" + INVOICE, PROSE + "\n\n" + INVOICE.replace("2023-04-01", "2023-01-04"), 10, 100),
        # OCR confusions inside an identifier (0 -> O, 1 -> l)
        (PROSE + "\n\n" + INVOICE, PROSE + "\n\n" + INVOICE.replace("PL-0081-B", "PL-O08l-B"), 10, 100),
        # two values swapped between clauses: every number is still on the page
        (
            PROSE + " Revenue rose 12% in the north while costs rose 15% in the south.",
            PROSE + " Revenue rose 15% in the north while costs rose 12% in the south.",
            10,
            100,
        ),
        # a hallucinated summary sentence at the end
        (PROSE, PROSE + "\n\nIn summary, the committee approved a 12% budget increase for the library.", 10, 100),
        # a sentence cut at a column break (major)
        (
            PROSE + "\n\nThe new reading room on the second floor will open in March and offer quiet desks.\n\nStaff "
            "will be trained.",
            PROSE + "\n\nThe new reading room on the second floor will open\n\nStaff will be trained.",
            5,
            100,
        ),
        # a footnote spliced into the middle of a body sentence (major)
        (
            PROSE + "\n\nThe council expects visitor numbers to rise sharply once the new wing is finished next "
            "spring.\n\n1 Figures from the 2022 annual visitor survey.",
            PROSE + "\n\nThe council expects visitor numbers to rise sharply\n1 Figures from the 2022 annual visitor "
            "survey.\nonce the new wing is finished next spring.",
            5,
            100,
        ),
        # a list whose numbering is kept but two items' contents are swapped (major)
        (
            PROSE + "\n\n1. Switch off the main power supply at the wall.\n2. Remove the front cover with a flat "
            "screwdriver.\n3. Replace the fuse with one of the same rating.",
            PROSE + "\n\n1. Remove the front cover with a flat screwdriver.\n2. Switch off the main power supply at "
            "the wall.\n3. Replace the fuse with one of the same rating.",
            5,
            100,
        ),
        # a duplicated table of contents (major)
        (
            "Contents\n\n1 Introduction 3\n2 Methods 7\n3 Results 12\n\n" + PROSE,
            "Contents\n\n1 Introduction 3\n2 Methods 7\n3 Results 12\n\n1 Introduction 3\n2 Methods 7\n3 Results "
            "12\n\n" + PROSE,
            5,
            100,
        ),
        # line-end hyphenation left in the output (harmless)
        (
            PROSE + " The information desk moves.",
            PROSE.replace("committee", "commit-\ntee") + " The informa- tion desk moves.",
            0,
            0,
        ),
    ],
    ids=[
        "date-day-month-swap",
        "ocr-in-identifier",
        "values-swapped",
        "hallucinated-summary",
        "sentence-cut-at-column-break",
        "footnote-spliced",
        "list-items-swapped",
        "toc-duplicated",
        "hyphenation-kept",
    ],
)
def test_adversarial_failures_meet_their_severity_class(ref: str, hyp: str, min_drop: float, max_drop: float):
    drop = (1 - score_text_next(ref, hyp).score) * 100
    assert min_drop - 1e-6 <= drop <= max_drop + 1e-6


# --- scorer edge cases -------------------------------------------------------------------------------------------


def test_fractions_exponents_dashes_fences_and_inline_page_numbers():
    # "¾" read as "3 4"; 10<sup>9</sup> read as "109" while $10^9$ read as "10 9"
    assert keys("¾ inch") == keys("3/4 inch") == ["#3/4", "inch"]
    assert keys("10<sup>9</sup> cells") == keys("$10^9$ cells") == keys("10⁹ cells") == ["#10", "#9", "cells"]
    assert keys("m<sup>2</sup>") == keys("m²") == ["#m2"]  # a letter base stays glued
    # an unspaced em dash joined two words
    assert keys("behaviors—including visa") == ["behaviors", "including", "visa"]
    # the fence label "python" is not on the page; "<page_number>7</page_number>/7" is "7/7"
    assert keys("```python\nx = 1\n```") == ["x", "#=", "#1"]
    assert keys("page <page_number>7</page_number>/7") == ["page", "#7/7"]


def test_spacing_inside_dates_and_ranges_is_harmless():
    # "23. 2. 2021" vs "23.2.2021"; "17- 43" vs "17-43": was a critical identifier error
    for ref_num, out_num in (("23. 2. 2021", "23.2.2021"), ("17- 43", "17-43")):
        ref = PROSE + f" Signed on {ref_num} by the clerk."
        assert score_text_next(ref, ref.replace(ref_num, out_num)).score == pytest.approx(1.0)


@pytest.mark.parametrize(
    "spaced",
    ["19.07 .2022", "19 .07.2022", "19. 07.2022", "19.07. 2022", "19\t. 07 \t.\t2022", "19.\u202f07 .\u00a02022"],
)
def test_spaces_on_either_side_of_date_dots_do_not_invent_numeric_errors(spaced: str) -> None:
    compact = "Signed on 19.07.2022 by the clerk."
    rendered = f"Signed on {spaced} by the clerk."

    for reference, hypothesis in ((compact, rendered), (rendered, compact)):
        result = score_text_next(reference, hypothesis)
        assert result.score == 1.0
        assert not result.events.kinds()


@pytest.mark.parametrize(
    ("reference", "hypothesis", "critical"),
    [
        ("Signed on 09.07 .2022", "Signed on 07.09.2022", True),
        ("Signed on 19.07 .2022", "Signed on 19.07.2023", True),
        ("Signed on 19.07 .2022", "Signed on 19.7.2022", True),
        ("Code AB-19.07 .2022-X", "Code AB-19.07.2022-Y", False),
        ("Code AB19.07 .2022-X", "Code AB19.07.2022-X", False),
        ("Code 19.07 .2022", "Code 19-07-2022", False),
        ("Coefficient 0.004307", "Coefficient 0.04307", True),
        ("Amount -47.2", "Amount 47.2", True),
        ("Code 19.\n07.2022", "Code 19.07.2022", False),
    ],
)
def test_date_spacing_equivalence_keeps_changed_values_signs_and_identifiers_chargeable(
    reference: str, hypothesis: str, critical: bool
) -> None:
    result = score_text_next(reference + ". Filed on 11.06 .2020.", hypothesis + ". Filed on 11.06.2020.")

    assert result.score < 1.0
    assert result.events.kinds()
    if critical:
        assert result.events.critical >= 1


def test_identically_spaced_dates_do_not_change_an_unrelated_spelling_penalty() -> None:
    reference = "The clerk recorded 19.07 .2022 before approval."
    hypothesis = "The clark recorded 19.07 .2022 before approval."

    assert score_text_next(reference, hypothesis).score == pytest.approx(6 / 7)


def test_dotted_output_without_a_dotted_reference_does_not_invent_a_missing_date() -> None:
    reference = "Record\n19\n07\n2022\nEnd of record."
    hypothesis = "Record\n19 . 07 . 2022\nEnd of record."

    assert score_text_next(reference, hypothesis).score == 1.0


@pytest.mark.parametrize("wrapped", ["12—\n18", "12-\r\n\t18", "12-\n  18"])
def test_single_soft_wrap_after_range_dash_is_not_a_number_rewrite(wrapped: str) -> None:
    """The eller replay charges a minor number rewrite for an unchanged range wrapped after its dash."""
    ref = PROSE + " The layer measures 12-18 cm."
    hyp = PROSE + f" The layer measures {wrapped} cm."
    for reference, output in ((ref, hyp), (hyp, ref)):
        result = score_text_next(reference, output)
        assert result.score == pytest.approx(1.0)
        assert result.events.kinds() == {}


@pytest.mark.parametrize(
    "separated",
    [
        "12-\n\n18 cm",
        "12-  \n18 cm",
        "12-\\\n18 cm",
        "<p>12-</p><p>18 cm</p>",
        "<table><tr><td>12-</td><td>18 cm</td></tr></table>",
        "| 12- |\n| 18 cm |",
    ],
)
def test_range_repair_does_not_join_ambiguous_hard_or_structural_breaks(separated: str) -> None:
    assert keys(separated) != keys("12-18 cm")
    result = score_text_next(PROSE + " 12-18 cm.", PROSE + " " + separated)
    assert result.score < 1.0
    assert result.events.kinds()


@pytest.mark.parametrize("wrong_range,expect_critical", [("18—\n12", True), ("12—\n19", True), ("12\n18", False)])
def test_wrapped_range_does_not_hide_reversed_changed_or_missing_endpoints(
    wrong_range: str, expect_critical: bool
) -> None:
    result = score_text_next(PROSE + " Depth 12-18 cm.", PROSE + f" Depth {wrong_range} cm.")
    assert result.score < 1.0
    assert result.events.kinds()
    assert (result.events.critical > 0) is expect_critical


def test_soft_wrapped_ranges_do_not_excuse_swapped_paragraphs() -> None:
    first = "The excavation depth was 12-18 cm below ground."
    second = "The drainage layer reached 24-30 cm after compaction."
    ref = PROSE + "\n\n" + first + "\n\n" + second
    hyp = PROSE + "\n\n" + second.replace("24-30", "24-\n30") + "\n\n" + first.replace("12-18", "12-\n18")
    result = score_text_next(ref, hyp)
    assert result.words == pytest.approx(1.0)
    assert result.order < 1.0
    assert result.events.moved > 0
    assert result.score < 1.0


def test_date_spacing_and_wrapped_range_are_both_preserved_in_one_page() -> None:
    reference = PROSE + " Filed on 19.07 .2022. The layer measures 12-18 cm."
    hypothesis = PROSE + " Filed on 19.07.2022. The layer measures 12-\n18 cm."
    assert score_text_next(reference, hypothesis).score == pytest.approx(1.0)


def test_every_lost_point_has_an_event_example():
    # row-major interleaving lowered the score to 0.59 with no event listed
    ref = "Answers\n1. A\n2. C\n3. B\n4. D\n5. A\n6. B"
    hyp = "Answers\n1. A 4. D\n2. C 5. A\n3. B 6. B"
    res = score_text_next(ref, hyp)
    assert res.score < 1.0 and res.events.kinds()


def test_gap_larger_than_the_dp_limit_is_still_aligned(monkeypatch: pytest.MonkeyPatch):
    """A gap over LCS_CELL_LIMIT was left unaligned (a 2,001-token page scored 0.0, a 1,999-token one 0.5)."""
    import random

    from parse_bench.evaluation.metrics.parse import text_next

    rng = random.Random(3)
    vocab = ["revenue", "margin", "center", "freight", "costs", "board", "dividend", "share", "payable", "record"]
    words = [rng.choice(vocab) for _ in range(300)]
    ref = " ".join(words)
    hyp = " ".join(w + "s" if i % 2 else w for i, w in enumerate(words))  # every 2nd word respelled: no 3-tiles
    exact = score_text_next(ref, hyp).proportional
    assert exact == pytest.approx(0.5)
    monkeypatch.setattr(text_next, "LCS_CELL_LIMIT", 1_000)
    assert score_text_next(ref, hyp).proportional == pytest.approx(exact, abs=0.02)
    monkeypatch.setattr(text_next, "LCS_SLICE_CELLS", 10_000)
    assert score_text_next(ref, hyp).proportional == pytest.approx(exact, abs=0.05)


def test_html_document_scaffolding_and_tag_attributes_are_not_text():
    ref = "Item #3 was approved by the board on Monday after a short debate."
    wrapped = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Minutes</title><style>p{color:red}</style>'
        f'</head><body><article class="main" id="m1"><p>{ref}</p></article></body></html>'
    )
    res = score_text_next(ref, wrapped)
    assert res.score == 1.0, res.events.kinds()
    assert score_text_next(ref, wrapped.replace("#3", "#8")).events.kinds().get("wrong number or identifier") == 1


_URL = "https://www.example.ch/region/neue-regeln-fuer-velofahrer-ld.1055646"
_CITED = f"Quelle: {_URL} abgerufen am 3. Mai 2019 von der Redaktion der Zeitung in der Stadt."


def test_identifier_misread_or_split_costs_no_more_than_omitting_it():
    omitted = score_text_next(_CITED, _CITED.replace(_URL + " ", ""))
    garbled = score_text_next(_CITED, _CITED.replace("-ld.1055646", "-Id. 1055646"))
    assert garbled.events.kinds() == {"identifier garbled": 1}
    assert garbled.score >= omitted.score


def test_identifier_with_a_changed_digit_stays_critical():
    res = score_text_next(_CITED, _CITED.replace("1055646", "1055647"))
    assert res.events.kinds().get("wrong number or identifier") == 1
    assert res.events.critical >= 1


@pytest.mark.parametrize(
    ("name", "written"),
    [("ʻAiea", "Aiea"), ("Koʻolau", "Ko‘olau"), ("Koʻolau", "Ko'olau"), ("Koʻolau", "Koolau"), ("Kāneʻohe", "Kaneohe")],
)
def test_okina_apostrophe_and_accent_variants_of_a_name_are_minor(name: str, written: str):
    page = "The new clinic opened last spring in {} near the harbor and serves families from the whole district."
    res = score_text_next(page.format(name), page.format(written))
    assert res.events.critical == 0 and res.events.kinds() == {"accent only": 1}


def test_a_different_name_is_still_critical():
    page = "The new clinic opened last spring in {} near the harbor and serves families from the whole district."
    assert score_text_next(page.format("Koʻolau"), page.format("Kailua")).events.kinds() == {"name changed": 1}


_LEAFLET = (
    "Keep the battery away from heat. Charge it only with the supplied cable and never leave it charging overnight."
)
_MARKERS = [
    "[IMAGE_DESCRIPTION: {}]",
    "[Image: {}]",
    "[Logo: {}]",
    "<figure><img src='a.png'><figcaption>{}</figcaption></figure>",
    '<img src="a.png" alt="{}">',
    "<::image: {}::>",
    "<::{} : figure::>",
    "<figure-description>\n\n***figure-description:***\n{}\n\n</figure-description>",
    '<div class="img-description" style="padding: 10px;">\n <div class="img-alt">{}</div>\n</div>',
]


def _with_description(marker: str, text: str) -> str:
    return _LEAFLET.replace(" Charge", "\n\n" + marker.format(text) + "\n\n Charge")


@pytest.mark.parametrize("marker", _MARKERS)
def test_marked_picture_description_is_free_like_alt_text(marker: str):
    desc = "a small green battery icon with a lightning bolt in the corner"
    assert score_text_next(_LEAFLET, _with_description(marker, desc)).score == 1.0


@pytest.mark.parametrize("marker", _MARKERS)
def test_marked_garbage_is_charged_and_marked_page_text_is_read_as_text(marker: str):
    assert score_text_next(_LEAFLET, _with_description(marker, "i l 1 | l i 1 ; j 1 l i")).score < 0.9
    moved = "Keep the battery away from heat. Charge it only with the supplied cable"
    out = marker.format(moved) + "\n\n" + _LEAFLET.replace(moved, "")
    assert score_text_next(_LEAFLET, out).score == 1.0


def test_unmarked_description_is_still_invented_text():
    out = _with_description("{}", "a small green battery icon with a lightning bolt in the corner")
    assert "invented passage" in score_text_next(_LEAFLET, out).events.kinds()


def test_page_text_inside_a_figure_still_counts():
    ref = (
        "metro\n\nFrom The Palms to Wigram, buses run every fifteen minutes on weekdays and every half hour on Sundays."
    )
    out = "<figure>\n\nmetro\n\n</figure>\n\n" + ref.split("\n\n", 1)[1]
    assert score_text_next(ref, out).score == 1.0


@pytest.mark.parametrize("printed", ["Kite", "2030-2031 Kite", "北駅"])
def test_short_printed_text_in_markdown_alt_is_not_reported_missing(printed: str) -> None:
    ref = printed + "\n\n" + PROSE
    out = f"![logo: **{printed}**](logo.png)\n\n" + PROSE
    assert score_text_next(ref, out).score == 1.0


def test_short_alt_reading_does_not_discard_unmatched_narration() -> None:
    ref = "Kite 2030-2031\n\n" + PROSE
    out = "![logo: **Kite** **2030-2031** logo](logo.png)\n\n" + PROSE
    result = score_text_next(ref, out)
    assert result.recall == 1.0
    assert result.critical == 1.0
    assert result.score < 1.0
    assert "inserted words" in result.events.kinds()


@pytest.mark.parametrize(
    "caption",
    ["", "Station 71 must not close.", "Station 17 must close.", "Depot 17 must not close."],
)
def test_short_alt_recovery_still_penalizes_missing_or_changed_printed_caption(caption: str) -> None:
    ref = "Kite\n\nStation 17 must not close.\n\n" + PROSE
    out = f"![logo: Kite](logo.png)\n\n<figure><figcaption>{caption}</figcaption></figure>\n\n" + PROSE
    assert score_text_next(ref, out).score < 1.0


@pytest.mark.parametrize("printed", ["Kite 2030-2032", "Kite", "2030-2031"])
def test_short_alt_recovery_still_penalizes_wrong_or_missing_logo_text(printed: str) -> None:
    ref = "Kite 2030-2031\n\n" + PROSE
    out = f"![logo: {printed}](logo.png)\n\n" + PROSE
    assert score_text_next(ref, out).score < 1.0


def test_short_alt_reading_does_not_exempt_unmarked_body_narration() -> None:
    ref = "Kite\n\n" + PROSE
    out = "![logo: Kite](logo.png)\n\nA bright bird perched above the distant river.\n\n" + PROSE
    assert score_text_next(ref, out).score < 1.0
