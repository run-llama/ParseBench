"""Tests for the formatting metric (``formatting_v2``)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from next_wiring import NextScorers

from parse_bench.evaluation.metrics.parse.formatting_v2 import (
    SUP,
    formatting_v2_applies,
    formatting_v2_metrics,
    parse,
    rules_reference,
    score_formatting,
)
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType
from parse_bench.test_cases.schema import ParseTestCase

PAGE = """# Annual Report

## Results

Revenue grew **12%** to $5 billion<sup>1</sup>. The *new* plant opened. Capacity is 10<sup>9</sup> units.

- First item
  - Nested item
- Second item

The fee is ~~$100~~ $80 per month.
"""

HTML_PAGE = (
    "<h1>Annual Report</h1><h2>Results</h2><p>Revenue grew <strong>12%</strong> to $5 billion¹. The <em>new</em> "
    "plant opened. Capacity is 10⁹ units.</p><ul><li>First item<ul><li>Nested item</li></ul></li><li>Second item</li>"
    "</ul><p>The fee is <del>$100</del> $80 per month.</p>"
)


def kinds(ref: str, out: str) -> list[tuple[str, str]]:
    return [(e.kind, e.cls) for e in score_formatting(ref, out).events if e.charged]


def score(ref: str, out: str) -> float:
    return score_formatting(ref, out).score


# --- syntax never matters ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out",
    [
        PAGE,
        HTML_PAGE,
        PAGE.replace("**12%**", "__12%__").replace("*new*", "_new_"),
        PAGE.replace("# Annual Report", "Annual Report\n=============").replace("## Results", "Results\n-------"),
        PAGE.replace("- First", "• First").replace("  - Nested", "    * Nested").replace("- Second", "• Second"),
        PAGE.replace("10<sup>9</sup>", "$10^{9}$").replace("billion<sup>1</sup>", "billion^1^"),
        PAGE.replace("# ", "## ").replace("## Results", "### Results"),  # uniform level shift
        "```markdown\n" + PAGE + "\n```",
        PAGE.replace("## Results", "## **Results** ##").replace(
            "~~$100~~", '<span style="text-decoration: line-through">$100</span>'
        ),
        PAGE.replace("\n\n", "\n\n\n").replace("  - Nested", "    - Nested"),
    ],
)
def test_syntax_variants_are_free(out):
    assert score(PAGE, out) == pytest.approx(1.0)


# --- severity classes -----------------------------------------------------------------------------------------


def test_strikethrough_lost_or_added_is_critical():
    assert kinds(PAGE, PAGE.replace("~~$100~~", "$100")) == [("strikethrough lost", "critical")]
    assert kinds(PAGE, PAGE.replace("~~$100~~ $80", "~~$100 $80~~")) == [("strikethrough added", "critical")]
    assert score(PAGE, PAGE.replace("~~$100~~", "$100")) <= 0.9


def test_exponent_lost_is_critical_footnote_marker_minor_ordinal_free():
    assert kinds(PAGE, PAGE.replace("10<sup>9</sup>", "109")) == [("exponent lost", "critical")]
    assert kinds(PAGE, PAGE.replace("billion<sup>1</sup>", "billion1")) == [("superscript lost", "minor")]
    neg = "A yield of 4.2 × 10<sup>-3</sup> was found."
    assert kinds(neg, neg.replace("<sup>-3</sup>", "-3")) == [("exponent lost", "critical")]
    ordinal = "The 1<sup>st</sup> cycle and **92%** activity."
    assert kinds(ordinal, ordinal.replace("<sup>st</sup>", "st")) == []


def test_flattened_letter_scripts_are_charged_like_digits():
    page = "The cost grows as 2<sup>n</sup> with depth, and x<sub>i</sub> is the input."
    flat = "The cost grows as 2n with depth, and xi is the input."
    assert kinds(page, flat) == [("exponent lost", "critical"), ("subscript lost", "minor")]
    assert score(page, page.replace("2<sup>n</sup>", "2ⁿ")) == pytest.approx(1.0)
    marks = "The 2<sup>nd</sup> and 1<sup>er</sup> editions of Acme<sup>TM</sup> ship with **Pro**."
    assert kinds(marks, marks.replace("<sup>", "").replace("</sup>", "")) == []


def test_exponent_swapped_or_added():
    assert kinds(PAGE, PAGE.replace("10<sup>9</sup>", "10<sub>9</sub>")) == [
        ("superscript read as subscript", "critical")
    ]
    assert kinds(PAGE, PAGE.replace("$80", "$8<sup>0</sup>")) == [("exponent added", "critical")]


def test_headings():
    assert kinds(PAGE, PAGE.replace("## Results", "Results")) == [("heading lost", "major")]
    assert kinds(PAGE, PAGE.replace("## Results", "**Results**")) == [("heading shown as bold line", "minor")]
    assert kinds(PAGE, PAGE.replace("## Results", "# Results")) == [("heading level relation changed", "minor")]
    three = "# A\n\nx\n\n## B\n\ny\n\n### C\n\nz"
    assert kinds(three, three.replace("## B", "### B").replace("### C", "## C")) == [
        ("heading hierarchy inverted", "major")
    ]
    assert kinds(three, three.replace("## B", "# B").replace("### C", "# C")) == [
        ("heading hierarchy flattened", "major")
    ]


def test_two_headings_merged_into_one_line_are_charged():
    page = "## Results\n\n### Revenue\n\nRevenue rose 5% in the year."
    assert kinds(page, "## Results Revenue\n\nRevenue rose 5% in the year.") == [("headings merged", "major")]
    assert score(page, "# Results\n\n## Revenue\n\nRevenue rose 5% in the year.") == pytest.approx(1.0)


def test_false_headings():
    page = (
        "# Title\n\nShort caption line\n\n"
        "This is a long paragraph of body text that goes on for quite a while about the plant."
    )
    assert kinds(page, page.replace("Short caption", "### Short caption")) == [("text promoted to heading", "minor")]
    assert kinds(page, page.replace("This is", "## This is")) == [("body text turned into heading", "major")]
    labels = "# Title\n\n**Procedure**\n\nDo the thing."
    assert kinds(labels, labels.replace("**Procedure**", "### Procedure")) == []  # a bold label is a pseudo heading


def test_lists():
    assert kinds(
        PAGE, PAGE.replace("- First item\n  - Nested item\n- Second item", "First item Nested item Second item")
    ) == [("list items merged", "major")]
    assert kinds(PAGE, PAGE.replace("  - Nested", "- Nested")) == [("list nesting flattened", "major")]
    steps = "1. Close valve V1 now\n2. Vent the line through port P3\n3. Remove the screws"
    assert kinds(steps, steps.replace("1. ", "- ").replace("2. ", "- ").replace("3. ", "- ")) == [
        ("list type changed", "minor")
    ]
    dropped = steps.replace("1. ", "").replace("2. ", "").replace("3. ", "")
    assert [k for k, _ in kinds(steps, dropped)] == ["list marker lost (systematic, 3x)"]
    split = steps.replace("2. Vent the line through port P3", "2. Vent the line\n9. through port P3")
    assert kinds(steps, split) == [("list item split", "minor")]


def test_emphasis_is_minor_and_style_swap_is_one_event():
    assert kinds(PAGE, PAGE.replace("**12%**", "12%")) == [("bold lost", "minor")]
    assert kinds(PAGE, PAGE.replace("**12%**", "*12%*")) == [("emphasis style changed", "minor")]
    assert kinds(PAGE, PAGE.replace("The *new* plant", "The *new plant*")) == [("italic span shifted", "minor")]
    assert 0.98 <= score(PAGE, PAGE.replace("**12%**", "12%")) < 1.0


def test_underline_is_major_on_a_redline():
    red = "The fee is ~~$100~~ <u>$80</u> per month."
    assert kinds(red, red.replace("<u>$80</u>", "$80")) == [("underline lost", "major")]
    plain = "See <u>Note 14</u> for details."
    assert kinds(plain, plain.replace("<u>Note 14</u>", "Note 14")) == [("underline lost", "minor")]


def test_systematic_quirk_is_charged_once():
    page = "\n\n".join(f"Paragraph {w} has an *italic phrase {w}* inside it." for w in "abcdef")
    res = score_formatting(page, page.replace("*", ""))
    assert res.counts() == {"critical": 0, "major": 0, "minor": 1}
    assert res.proportional < res.floor  # the extent is in the proportional part


@pytest.mark.parametrize(
    "hide",
    [lambda s: "", lambda s: f"<!-- {s} -->", lambda s: f"$$\\text{{{s}}}$$"],
    ids=["deleted", "comment", "display-math"],
)
def test_a_formatted_item_dropped_with_its_text_is_lost(hide):
    ref = "The monthly fee is ~~$25~~ $15 for all **Standard** plans. Customers keep their renewal date."
    out = ref.replace("~~$25~~", hide("$25"))
    assert kinds(ref, out) == [("strikethrough missing", "critical")]
    assert score(ref, out) == pytest.approx(score(ref, ref.replace("~~$25~~", "$25")))
    assert score(ref, out) < 1.0
    dropped = ref.replace("**Standard** ", "")
    assert kinds(ref, dropped) == [("bold missing", "minor")]


def test_a_respelled_formatted_word_is_located_and_costs_one_minor_event():
    ref = "The fee is **Standard** for all plans and customers in every region."
    res = score_formatting(ref, ref.replace("Standard", "Standrad"))
    assert [(e.kind, e.cls) for e in res.events] == [("formatted text respelled", "minor")]
    assert res.coverage == 1.0 and res.score > 0.9
    assert kinds(ref, ref.replace("**Standard**", "Standrad")) == [
        ("formatted text respelled", "minor"),
        ("bold lost", "minor"),
    ]
    assert kinds(ref, ref.replace("**Standard**", "**Premium**")) == [("bold missing", "minor")]  # another word


def test_a_style_split_inside_a_word_is_matched_across_the_split():
    ref = "The monthly fee is $15 for all **Standard** plans in every region."
    split = ref.replace("**Standard**", "S<i>tandard</i>")
    assert kinds(ref, split) == [("emphasis style changed", "minor")]
    assert score(ref, split) == pytest.approx(score(ref, ref.replace("**Standard**", "*Standard*")))
    assert kinds(ref, ref.replace("**Standard**", "**S**tandard")) == [("bold lost", "minor")]
    exp = "Capacity is 10<sup>9</sup> units per year."
    assert score(exp, exp.replace("10<sup>9</sup>", "10 <sup>9</sup>")) == pytest.approx(1.0)


def test_dropped_headings_and_list_items_are_charged_and_empty_output_scores_zero():
    half = PAGE.split("- First")[0]
    assert kinds(PAGE, half) == [("strikethrough missing", "critical"), ("list item missing (systematic, 3x)", "minor")]
    assert kinds(PAGE, PAGE.replace("## Results\n", "")) == [("heading missing", "major")]
    res = score_formatting(PAGE, "")
    assert res.score == 0.0 and res.coverage == 0.0


# --- adversarial parsing ----------------------------------------------------------------------------------------


def test_money_is_not_math_and_snake_case_is_not_italic():
    assert kinds("Pay $5 and $10 today.", "Pay $5 and $10 today.") == []
    page = "Set **max_value** in config_file_name before use."
    assert score(page, page) == pytest.approx(1.0)
    assert kinds(page, page.replace("**max_value**", "max_value")) == [("bold lost", "minor")]


def test_code_fence_content_is_not_parsed_as_markdown():
    page = "Run this:\n\n```bash\necho **not bold** ~~x~~\n```\n\nDone **now**."
    doc = parse(page)
    assert [u.role for u in doc.units] == ["p", "code", "p"]
    assert kinds(page, page) == []
    # unfenced, the same characters are Markdown again: that output really shows bold and struck text
    assert ("strikethrough added", "critical") in kinds(page, page.replace("```bash\n", "").replace("\n```", ""))
    plain = "Run this:\n\n```bash\ncal --sensor ps2 --zero\n```\n\nDone **now**."
    assert kinds(plain, plain.replace("```bash\n", "").replace("\n```", "")) == [("code block lost", "minor")]


def test_html_tables_attributes_and_nesting():
    ref = "<table><tr><td><b>Total</b></td><td>10<sup>3</sup></td></tr></table>"
    out = "| **Total** | 10³ |\n|---|---|"
    assert score(ref, out) == pytest.approx(1.0)
    assert kinds(ref, "| Total | 103 |\n|---|---|") == [("bold lost", "minor"), ("exponent lost", "critical")]
    nested = '<p><b class="x">Net <i>income</i></b> rose</p>'
    assert score(nested, "**Net *income*** rose") == pytest.approx(1.0)


def test_unbalanced_markers_and_cjk():
    assert score("Revenue* rose 5%.", "Revenue* rose 5%.") == pytest.approx(1.0)
    cjk = "## 摘要\n\n**本报讯** 国务院公布了意见。"
    assert score(cjk, cjk) == pytest.approx(1.0)
    assert kinds(cjk, cjk.replace("**本报讯**", "本报讯")) == [("bold lost", "minor")]


def test_emphasis_wrapping_lines_inside_a_paragraph():
    ref = "**A bold sentence that wraps\nonto the next line** and ends."
    assert score(ref, "**A bold sentence that wraps onto the next line** and ends.") == pytest.approx(1.0)


def test_superscript_bits_are_per_character():
    tok = parse("10<sup>9</sup>").toks[0]
    assert tok.key == "109" and tok.mask(SUP) == "001"


def test_applies_only_with_formatting():
    assert formatting_v2_applies(PAGE)
    assert not formatting_v2_applies("Just plain text with no formatting at all.")


# --- rule-mode references (rule-only formatting test cases) ---------------------------------------------------


def rule(**kw):
    return SimpleNamespace(**kw)


def test_rules_reference_heading_structure_is_complete():
    ref, scope = rules_reference(
        [rule(type="heading_structure", headings=[{"text": "Scope", "level": 2}, {"text": "Terms", "level": 3}])]
    )
    assert scope.complete_headings and not scope.lists
    out = "## Scope\n\nbody\n\n### Terms\n\n## Signatures"
    assert [(e.kind, e.cls) for e in score_formatting(ref, out, scope).events] == [
        ("heading not in the page's heading list", "minor")
    ]


def test_rules_reference_judges_only_named_kinds():
    ref, scope = rules_reference(
        [rule(type="present_as_strikeout", text="five days notice"), rule(type="present", text="thirty days notice")]
    )
    assert score_formatting(ref, "~~five days notice~~ **thirty days notice**", scope).score == pytest.approx(1.0)
    assert [e.kind for e in score_formatting(ref, "five days notice thirty days notice", scope).events] == [
        "strikethrough lost"
    ]
    assert rules_reference([rule(type="present", text="plain")]) is None


HARD_NEGATIVE = [rule(type="present", text="Bank of Fairview"), rule(type="is_not_strikeout", text="Bank of Fairview")]


@pytest.mark.parametrize(
    ("out", "expected", "kinds"),
    [
        ("Bank of Fairview plc", 1.0, {}),
        ("~~Bank of Fairview~~ plc", 0.75, {"strikethrough added": 1}),
        ("", 0.0, {"plain text missing (3 of 3 tokens not found)": 1}),
    ],
)
def test_hard_negative_rules_are_scored_and_a_dropped_span_does_not_pass(out, expected, kinds):
    """Bugs: a page whose rules hold no styled span got no reference (its hard negative went unscored), and with
    no formatted atoms to locate, coverage passed an output that dropped the very span the negative tests."""
    ref, scope = rules_reference(HARD_NEGATIVE)
    metric = formatting_v2_metrics(ref, out, scope)[0]
    assert metric.value == pytest.approx(expected)
    assert metric.metadata["event_kinds"] == kinds


def test_empty_heading_list_makes_any_output_heading_false():
    """Bug: a headings page without headings got no reference, so invented headings were free."""
    ref, scope = rules_reference([rule(type="heading_structure", headings=[])])
    assert formatting_v2_metrics(ref, "Body text only.", scope)[0].value == pytest.approx(1.0)
    assert formatting_v2_metrics(ref, "# Invented\n\nBody text only.", scope)[0].value < 1.0


# --- evaluator wiring -------------------------------------------------------------------------------------------


def _result(markdown: str) -> InferenceResult:
    out = ParseOutput(
        example_id="g/d", pipeline_name="p", markdown=markdown, pages=[PageIR(page_index=0, markdown=markdown)]
    )
    return InferenceResult(
        request=InferenceRequest(example_id="g/d", source_file_path="d.pdf", product_type=ProductType.PARSE),
        pipeline_name="p",
        product_type=ProductType.PARSE,
        raw_output={},
        output=out,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        latency_in_ms=1,
    )


def _case(metadata: dict | None, expected: str | None = PAGE, rules: list | None = None) -> ParseTestCase:
    return ParseTestCase.model_construct(
        test_id="g/d", group="g", file_path="d.pdf", expected_markdown=expected, metadata=metadata, test_rules=rules
    )


def test_wiring_emits_formatting_v2_only_when_the_reference_is_formatted():
    ev = NextScorers()
    by_name = {m.metric_name: m for m in ev.evaluate(_result(HTML_PAGE), _case({"scoring": "next"})).metrics}
    assert by_name["formatting_v2"].value == pytest.approx(1.0)
    assert {"formatting_v2_inline", "formatting_v2_headings", "formatting_v2_lists"} <= set(by_name)
    plain = {
        m.metric_name for m in ev.evaluate(_result("text"), _case({"scoring": "next"}, "Only plain text here.")).metrics
    }
    assert "formatting_v2" not in plain


def test_evaluator_uses_rules_when_there_is_no_reference_markdown():
    from parse_bench.test_cases.parse_rule_schemas import coerce_parse_rule

    rules = [coerce_parse_rule({"type": "present_as_strikeout", "text": "development guidelines"})]
    ev = NextScorers()
    by_name = {
        m.metric_name: m
        for m in ev.evaluate(
            _result("the development guidelines apply"), _case({"scoring": "next"}, None, rules)
        ).metrics
    }
    # one critical atom on a one-atom page: proportional 1 - 1 / (1 + 3) sets the score, below the 0.90 floor
    assert by_name["formatting_v2"].value == pytest.approx(0.75)
    assert by_name["formatting_v2"].metadata["event_kinds"] == {"strikethrough lost": 1}


def test_evaluator_scores_a_hard_negative_case_without_reference_markdown():
    """Contract at the evaluator boundary: a rule-only hard negative (present + is_not_strikeout, no
    expected Markdown) emits formatting_v2 and charges strikethrough the output adds to its span."""
    from parse_bench.test_cases.parse_rule_schemas import coerce_parse_rule

    rules = [coerce_parse_rule({"type": "present", "text": "Bank of Fairview"})]
    rules.append(coerce_parse_rule({"type": "is_not_strikeout", "text": "Bank of Fairview"}))
    ev = NextScorers()
    case = _case({"scoring": "next"}, None, rules)
    by_name = {m.metric_name: m for m in ev.evaluate(_result("~~Bank of Fairview~~ plc"), case).metrics}
    assert by_name["formatting_v2"].metadata["event_kinds"] == {"strikethrough added": 1}
    plain = {m.metric_name: m for m in ev.evaluate(_result("Bank of Fairview plc"), case).metrics}
    assert plain["formatting_v2"].value == pytest.approx(1.0)


# ---- full-page decoration inventories (metadata.decorations) and the strikeout / underline lanes ----

from parse_bench.evaluation.metrics.parse.formatting_v2 import decoration_reference  # noqa: E402

REDLINE = (
    "2. A Public Nuisance ~~includes~~may include but is not limited to:\n\n"
    "~~9.~~8. Underage Person: Any person under 21 years of age.\n\n"
    "The fee is ~~one hundred dollars~~ eighty dollars per month."
)  # struck tokens: includes, 9, one, hundred, dollars


def _decoration_case(kind: str, md: str, neutral: list[str] | None = None) -> dict:
    return {"scoring": "next", "decorations": {"kind": kind, "reference_md": md, "neutral": neutral or []}}


def _fmt(meta: dict, output: str) -> dict:
    ev = NextScorers()
    case = _case(meta, None, [])
    return {m.metric_name: m for m in ev.evaluate(_result(output), case).metrics}


def test_inventory_mode_scores_every_decorated_run_in_any_syntax():
    out = (
        "2. A Public Nuisance <del>includes</del>may include but is not limited to:\n\n"
        "<s>9.</s>8. Underage Person: Any person under 21 years of age.\n\n"
        'The fee is <span style="text-decoration: line-through">one hundred dollars</span> eighty dollars per month.'
    )
    by = _fmt(_decoration_case("strikeout", REDLINE), out)
    assert by["formatting_v2"].value == 1.0
    assert by["formatting_v2"].metadata["decoration"]["strikeout"] == {"tp": 5, "fp": 0, "fn": 0}


def test_inventory_mode_charges_strikethrough_added_outside_the_annotated_runs():
    """Rule mode only judged the annotated span; the inventory makes decoration added anywhere a false positive,
    and invented strikethrough keeps its critical floor."""
    by = _fmt(_decoration_case("strikeout", REDLINE), REDLINE.replace("eighty dollars", "~~eighty dollars~~"))
    meta = by["formatting_v2"].metadata
    assert meta["decoration"]["strikeout"] == {"tp": 5, "fp": 2, "fn": 0}
    assert meta["proportional"] == pytest.approx(10 / 12)
    assert by["formatting_v2"].value == pytest.approx(min(10 / 12, meta["floor"]))
    assert meta["floor"] <= 0.90


def test_an_underline_page_whose_underlines_are_all_dropped_scores_low():
    """Atom loss rated a lost underline minor, so a parser that never writes underline kept about 0.92."""
    ref = "Insert <u>the new clause</u> here.\n\nSee <u>Schedule B</u> for the <u>updated fees</u>."
    by = _fmt(_decoration_case("underline", ref), ref.replace("<u>", "").replace("</u>", ""))
    assert by["formatting_v2"].value == 0.0
    assert by["formatting_v2_underline"].value == 0.0


def test_inventory_hard_negative_passes_plain_text_and_fails_an_empty_output():
    page = "Bank of Fairview plc\n\nRegistered office: The Quay, Westport"
    meta = _decoration_case("strikeout", page)
    assert _fmt(meta, page)["formatting_v2"].value == 1.0
    struck = _fmt(meta, "~~Bank of Fairview~~ plc\n\nRegistered office: The Quay, Westport")["formatting_v2"]
    assert struck.metadata["proportional"] == pytest.approx(1 - 3 / 9)
    assert struck.value <= 0.90  # invented strikethrough keeps its critical floor
    assert _fmt(meta, "")["formatting_v2"].value == 0.0


def test_hard_negative_charges_the_share_of_wrongly_decorated_tokens():
    """F1 on a page with no decoration was 0 for any false positive: one underlined link cost the whole page."""
    page = " ".join(f"word{i}" for i in range(40)) + "\n\nSee the annual report for details."
    meta = _decoration_case("underline", page)
    one_link = _fmt(meta, page.replace("annual report", "<u>annual report</u>"))["formatting_v2"]
    assert one_link.metadata["proportional"] == pytest.approx(1 - 2 / 46)
    assert one_link.value > 0.9
    everything = _fmt(meta, "<u>" + page.replace("\n\n", "</u>\n\n<u>") + "</u>")["formatting_v2"]
    assert everything.metadata["proportional"] == 0.0 and everything.value == 0.0


def test_neutral_inventory_text_is_scored_neither_way():
    ref = "Fees are listed in <u>Schedule B</u> and at -------- the bottom."
    out = "Fees are listed in <u>Schedule B</u> and at <u>-------- the</u> bottom."
    assert _fmt(_decoration_case("underline", ref), out)["formatting_v2"].metadata["decoration"]["underline"]["fp"] == 1
    assert _fmt(_decoration_case("underline", ref, ["-------- the"]), out)["formatting_v2"].value == 1.0


def test_inventory_judges_only_its_own_decoration():
    ref = "Insert <u>new clause</u> here and delete the old one."
    by = _fmt(_decoration_case("underline", ref), "Insert <u>new clause</u> here and **delete** ~~the old one~~.")
    assert by["formatting_v2"].value == 1.0
    assert "formatting_v2_strikeout" not in by


def test_decoration_lanes_on_a_full_reference_page_leave_the_headline_unchanged():
    """The lanes are diagnostics on ordinary pages: the atom-based headline is computed as before."""
    ref = "The fee is ~~$100~~ $80 per month, see <u>Annex 2</u>."
    lost = score_formatting(ref, "The fee is $100 $80 per month, see Annex 2.")
    assert lost.lanes["strikeout"] == 0.0 and lost.lanes["underline"] == 0.0
    # strikethrough lost is critical (1.0); on a page with strikethrough a lost underline is major (0.5)
    assert lost.proportional == pytest.approx(1 - (1.0 + 0.5) / (2 + 3))


def test_decoration_reference_needs_a_known_kind_and_markdown():
    assert decoration_reference({"decorations": {"kind": "bold", "reference_md": "x"}}) is None
    assert decoration_reference({"decorations": {"kind": "underline", "reference_md": ""}}) is None
    assert decoration_reference(None) is None
