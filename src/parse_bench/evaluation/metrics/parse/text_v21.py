"""Versioned visible-text contract for text_extended v2.1.

The same projection authors the reference bags and scores predictions. Ordinary
code and table cells are content; generated diagram syntax and image alt text
are not. Word coverage is case-insensitive, preserves script marks, and counts
CJK characters individually so line wrapping cannot change completeness.
These are transcription signals, not mathematical-equivalence or layout tests.
"""

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from html import escape

from lxml import html
from markdown_it import MarkdownIt

from parse_bench.evaluation.metrics.parse.rules_base import ParseTestRule
from parse_bench.schemas.parse_output import ParseOutput

PROFILE = "text-v2.1"
RULE_TYPES = {
    "missing_word_percent",
    "unexpected_word_percent",
    "too_many_word_occurence_percent",
    "missing_sentence_percent",
    "unexpected_sentence_percent",
    "too_many_sentence_occurence_percent",
    "bag_of_digit_percent",
}
_MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")
_TAGS = re.compile(
    r"</?(?:p|div|span|br|hr|h[1-6]|b|i|u|s|del|ins|mark|strong|em|sub|sup|"
    r"table|thead|tbody|tfoot|tr|td|th|caption|ul|ol|li|blockquote|pre|code|"
    r"header|footer|img|a)\b[^>]*>",
    re.I,
)
_ANGLE = re.compile(r"<[^\n>]*>")
_AUTOLINK = re.compile(r"<(?:https?://|mailto:)[^>]+>|<[^ >@]+@[^ >@]+>")


def _protect_literals(text: str) -> str:
    # An inequality or Korean notice in angle brackets is not an HTML tag.
    return _ANGLE.sub(lambda m: m[0] if _TAGS.fullmatch(m[0]) or _AUTOLINK.fullmatch(m[0]) else escape(m[0]), text)


def _html_text(value: str) -> str:
    root = html.fragment_fromstring(value, create_parent="div")
    for image in root.xpath(".//img"):
        image.drop_tree()
    for node in root.iter():
        if node.tag in {
            "br",
            "p",
            "div",
            "tr",
            "li",
            "header",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "pre",
            "caption",
        }:
            node.tail = "\n" + (node.tail or "")
        elif node.tag in {"td", "th"}:
            node.tail = " " + (node.tail or "")
    return str(root.text_content())


@lru_cache(maxsize=64)
def visible_text(markdown: str) -> str:
    """Project Markdown to printed text without parsing identifiers as emphasis."""
    markdown = re.sub(r"</?(?:page_header|page_footer|page_number)\b[^>]*>", "\n", markdown, flags=re.I)
    parts: list[str] = []
    lists: list[int | None] = []
    for token in _MARKDOWN.parse(_protect_literals(markdown)):
        if token.type in {"ordered_list_open", "bullet_list_open"}:
            lists.append(int(token.attrGet("start") or 1) if token.type == "ordered_list_open" else None)
        elif token.type in {"ordered_list_close", "bullet_list_close"}:
            lists.pop()
        elif token.type == "list_item_open" and lists and lists[-1] is not None:
            number = lists[-1]
            assert number is not None
            parts.append(f"\n{number}. ")
            lists[-1] = number + 1
        elif token.type == "inline":
            for child in token.children or []:
                if child.type in {"text", "code_inline"}:
                    parts.append(child.content)
                elif child.type in {"softbreak", "hardbreak"}:
                    parts.append("\n")
                elif child.type == "html_inline":
                    # Inline tags are styling, not content. Keep word boundaries
                    # around explicit breaks; b/i spans can split a printed word.
                    if re.match(r"<br\b", child.content, re.I):
                        parts.append("\n")
            parts.append("\n")
        elif token.type in {"fence", "code_block"}:
            if token.info.strip().lower() not in {"mermaid", "description"}:
                parts.extend([token.content, "\n"])
        elif token.type == "html_block":
            parts.extend([_html_text(token.content), "\n"])
    return unicodedata.normalize("NFC", "".join(parts)).lower()


def _cjk(ch: str) -> bool:
    return any(
        lo <= ord(ch) <= hi
        for lo, hi in (
            (0x3400, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x3040, 0x30FF),
            (0xAC00, 0xD7AF),
            (0x20000, 0x323AF),
        )
    )


def tokens(text: str) -> list[str]:
    """Keep single letters/digits (answer keys), marks, and every mixed-script suffix."""
    result: list[str] = []
    word = ""
    for ch in text:
        if _cjk(ch):
            if word:
                result.append(word)
                word = ""
            result.append(ch)
        elif unicodedata.category(ch)[0] in "LMN":
            word += ch
        elif word:
            result.append(word)
            word = ""
    if word:
        result.append(word)
    return result


def sequence(text: str) -> str:
    return " ".join(tokens(text))


def count_sequence(needle: str, haystack: str) -> int:
    return len(re.findall(r"(?<!\S)" + re.escape(needle) + r"(?!\S)", haystack))


def sentence_bag(text: str) -> Counter[str]:
    full = sequence(text)
    # Sentence coverage uses ordered token spans, allowing line reflow. Count
    # each anchor throughout the reference too, so nested/repeated anchors do
    # not manufacture duplicate failures on the reference itself.
    anchors = {sequence(line) for line in re.split(r"\n+|(?<!\d)[.!?。！？](?!\d)", text)}
    return Counter({anchor: count_sequence(anchor, full) for anchor in sorted(anchors) if " " in anchor})


def reference_rules(markdown: str) -> list[dict]:
    """Author all bags from this contract, never from a model prediction."""
    text = visible_text(markdown)
    bags = {
        "word": Counter(tokens(text)),
        "sentence": sentence_bag(text),
        "digit": Counter(ch for ch in text if ch in "0123456789"),
    }
    rules = []
    for kind in ("word", "sentence", "digit"):
        if not bags[kind]:
            continue
        types = (
            ["bag_of_digit_percent"]
            if kind == "digit"
            else [
                f"missing_{kind}_percent",
                f"unexpected_{kind}_percent",
                f"too_many_{kind}_occurence_percent",
            ]
        )
        for rule_type in types:
            rule = {"type": rule_type, "text_normalization": PROFILE, f"bag_of_{kind}": dict(bags[kind])}
            if kind == "sentence":
                rule["reference_text"] = sequence(text)
            rules.append(rule)
    return rules


def delivered_markdown(markdown: str, output: ParseOutput | None) -> str:
    """Include structured page sections once; preserve real repetitions in the body."""
    if output is None or not output.pages or not output.layout_pages:
        return markdown
    page_body = "\n".join(page.markdown for page in output.pages)
    if sequence(visible_text(markdown)) != sequence(visible_text(page_body)):
        # Otherwise replacing document text with page text could silently drop
        # hallucinations or restore omissions before measuring them.
        raise ValueError("text-v2.1 requires consistent document and page Markdown before section folding")
    layouts = {page.page_number: page for page in output.layout_pages}
    pages = []
    for page in output.pages:
        body = page.markdown
        layout = layouts.get(page.page_index + 1)
        if layout:
            header = layout.page_header_markdown or ""
            footer = layout.page_footer_markdown or ""
            # Only boundary copies are redundant. A matching phrase inside the
            # body can be a real occurrence and must not hide an omitted header.
            body_text = sequence(visible_text(body))
            header_text = sequence(visible_text(header))
            footer_text = sequence(visible_text(footer))
            if header_text and not (body_text == header_text or body_text.startswith(header_text + " ")):
                body = header + "\n\n" + body
            if footer_text and not (body_text == footer_text or body_text.endswith(" " + footer_text)):
                body += "\n\n" + footer
        pages.append(body)
    return "\n\n".join(pages)


class VisibleTextBagRule(ParseTestRule):
    """Opt-in scoring with one reference/prediction projection and explicit counts."""

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        text = visible_text(delivered_markdown(md_content, self.parse_output))
        kind = "digit" if self.type == "bag_of_digit_percent" else ("sentence" if "sentence" in self.type else "word")
        expected = Counter(self._rule_data.get(f"bag_of_{kind}", {}))
        if not expected or any(not isinstance(n, int) or n < 1 for n in expected.values()):
            raise ValueError("Visible-text bags must contain positive reference counts")
        if kind == "word":
            actual = Counter(tokens(text))
        elif kind == "digit":
            actual = Counter(ch for ch in text if ch in "0123456789")
        else:
            full = sequence(text)
            actual = Counter({key: count_sequence(key, full) for key in expected})
        missing = expected - actual
        excess = actual - expected
        if self.type.startswith("missing"):
            errors, denominator = sum(missing.values()), sum(expected.values())
        elif self.type.startswith("unexpected"):
            if kind == "sentence":
                reference = self._rule_data.get("reference_text")
                if not reference:
                    raise ValueError("Sentence precision requires reference_text")
                observed = sentence_bag(text)
                excess = Counter({k: n for k, n in observed.items() if not count_sequence(k, reference)})
                denominator = sum(observed.values())
            else:
                excess = Counter({k: n for k, n in actual.items() if k not in expected})
                denominator = sum(actual.values())
            errors = sum(excess.values())
        elif kind == "digit":
            errors = sum(missing.values()) + sum(excess.values())
            denominator = sum(expected.values()) + sum(actual.values())
        else:
            excess = Counter({k: n for k, n in excess.items() if k in expected})
            errors = sum(excess.values())
            denominator = sum(expected.values()) + errors
        score = 1 - errors / denominator if denominator else 1.0
        self.result_details = {
            "text_normalization": PROFILE,
            "errors": errors,
            "denominator": denominator,
            "missing": dict(missing.most_common(10)),
            "excess": dict(excess.most_common(10)),
        }
        return (
            errors == 0,
            "" if errors == 0 else f"{kind} content: {errors}/{denominator} errors; {self.result_details}",
            score,
        )
