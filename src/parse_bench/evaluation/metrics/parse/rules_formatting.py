"""Formatting, LaTeX, code block, title, and page section test rules."""

import re
import unicodedata
from typing import Any, cast

from parse_bench.evaluation.metrics.parse.emphasis_spans import EmphasisSpanMatcher
from parse_bench.evaluation.metrics.parse.rules_base import (
    ParseTestRule,
    RuleNotApplicable,
    _unescape_html_entities,
    is_degenerate_marker_text,
)
from parse_bench.evaluation.metrics.parse.test_types import TestType
from parse_bench.evaluation.metrics.parse.utils import _normalize_quotes, normalize_text
from parse_bench.test_cases.parse_rule_schemas import (
    ParseAbsentUnlessStrikeoutRule,
    ParseCodeBlockRule,
    ParseFormattingRule,
    ParseLatexRule,
    ParseMarkColorRule,
    ParseNotLatexRule,
    ParsePageSectionRule,
    ParsePresentAsStrikeoutRule,
    ParseTextColorRule,
    ParseTitleHierarchyPercentRule,
    ParseTitleRule,
)

# ---------------------------------------------------------------------------
# Unicode superscript / subscript character tables
# ---------------------------------------------------------------------------
# Inline markup tolerance – allows regex patterns built from stripped rule text
# to still match raw markdown that contains nested formatting markers.
# E.g. rule text "hello world" matches raw "hello ~~world~~".
# ---------------------------------------------------------------------------

# Matches optional inline formatting tokens that may appear between words
# in raw markdown: strikethrough (~~), bold (**), italic (* or _), and
# any HTML open/close/self-closing tags (<b>, </b>, <mark>, <sup>, etc.).
_INLINE_MARKUP_OPT = r"(?:\*{1,2}|~~|__?|</?\w+(?:\s[^>]*)?>)*"

# Markdown allows backslash-escaping of ASCII punctuation characters.
# Model output often contains these (e.g. ``\~``, ``\*``) which prevent
# rule text from matching.  This pattern strips such escapes.
_MD_BACKSLASH_ESCAPE_RE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")


# ---------------------------------------------------------------------------
# Span scoping for the inline-formatting detectors
# ---------------------------------------------------------------------------
# The markdown emphasis arms (``**``, ``*``, ``_``, ``~~``) are not regexes:
# no guard local to one delimiter run can tell a padded span's own markers
# from two padded spans' markers seen back to back, so those arms resolve
# spans by delimiter-run pairing (see ``emphasis_spans``) and search the query
# inside the paired spans. The HTML tag-pair and heading arms below remain
# regexes; the guards that follow keep their matches confined to one span.


def _tempered(closing: str) -> str:
    """Lazy any-char run that cannot cross *closing* (a regex literal).

    Without this, ``<b>.*?query.*?</b>`` happily matches by starting at one
    span's opening tag and borrowing a *later* span's tag as its closer.
    """
    return r"(?:(?!" + closing + r").)*?"


# Whitespace between the query's own words. A bare ``\s+`` here matches
# newlines regardless of the DOTALL flag, which lets the QUERY straddle
# boundaries the surrounding pattern is forbidden to cross - e.g. a query
# starting on a heading line and finishing in the paragraph below. This run
# may wrap a single line break but never a paragraph break.
_WORD_WS = r"(?:(?!\r?\n[ \t]*\r?\n)\s)+"


def _confined_to_line(query: str) -> str:
    """Rewrite a flexible query so its word gaps cannot leave the line.

    Heading text lives on a single line, so a query embedded in a heading
    pattern must not continue onto the next one. ``_WORD_WS`` is a generated
    token that cannot arise from ``re.escape``'d rule text (escaping doubles
    every backslash), so the replacement is unambiguous.
    """
    return query.replace(_WORD_WS, r"[ \t]+")


def _paragraph_spanning(query: str) -> str:
    """Rewrite a flexible query so its word gaps may cross paragraph breaks.

    A markdown emphasis span ends at a blank line, but an HTML tag pair ends
    at its closing tag and legitimately wraps multiple paragraphs - so inside
    a tag-pair arm the containment scope comes from the tags, and the query's
    word gaps must stay fully permissive (``<u>alpha\\n\\nbeta</u>`` is
    underlined). The token replacement is unambiguous for the same reason as
    in ``_confined_to_line``.
    """
    return query.replace(_WORD_WS, r"\s+")


def _make_markup_tolerant(escaped_text: str) -> str:
    """Make an escaped regex pattern tolerant of inline formatting markup.

    Between each word boundary (space) and at the start/end of the text,
    allow optional inline markup tokens (``~~``, HTML tags) so that
    clean rule text can match raw content with nested formatting.
    """
    # Split on escaped spaces (re.escape turns " " into "\\ ")
    # and rejoin with markup-tolerant whitespace
    parts = escaped_text.split(r"\ ")
    joiner = _INLINE_MARKUP_OPT + _WORD_WS + _INLINE_MARKUP_OPT
    tolerant = joiner.join(parts)
    # Allow leading and trailing markup adjacent to the text
    return _INLINE_MARKUP_OPT + tolerant + _INLINE_MARKUP_OPT


# Regex patterns for stripping inline formatting by kind.
# Used by _strip_other_formatting() to remove all formatting EXCEPT the type
# being tested, so the outer markers remain for pattern matching.
_STRIP_PATTERNS: dict[str, list[tuple[re.Pattern, str]]] = {
    "bold": [
        (re.compile(r"\*\*\*(.+?)\*\*\*", re.DOTALL), r"\1"),
        (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),
        (re.compile(r"</?(?:b|strong)>", re.IGNORECASE), ""),
    ],
    "italic": [
        (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"\1"),
        (re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)"), r"\1"),
        (re.compile(r"</?(?:i|em)>", re.IGNORECASE), ""),
    ],
    "underline": [
        (re.compile(r"</?(?:u|ins)>", re.IGNORECASE), ""),
    ],
    "strikeout": [
        (re.compile(r"~~"), ""),
        (re.compile(r"</?(?:s|del|strike)>", re.IGNORECASE), ""),
    ],
    "mark": [
        (re.compile(r"</?mark\b[^>]*>", re.IGNORECASE), ""),
    ],
    "sup": [
        (re.compile(r"</?sup>", re.IGNORECASE), ""),
    ],
    "sub": [
        (re.compile(r"</?sub>", re.IGNORECASE), ""),
    ],
}


def _strip_other_formatting(text: str, keep_kind: str) -> str:
    """Strip all inline formatting markers EXCEPT those of *keep_kind*.

    This lets us re-try pattern matching after removing nested markup that
    would otherwise prevent the outer-marker regex from matching the clean
    rule text.
    """
    result = text
    for kind, replacements in _STRIP_PATTERNS.items():
        if kind == keep_kind:
            continue
        for pattern, repl in replacements:
            result = pattern.sub(repl, result)
    # Collapse *horizontal* whitespace only. Line structure has to survive:
    # the detectors distinguish a heading line from body text, and the query's
    # _WORD_WS gaps wrap line breaks anyway. Collapsing newlines here would
    # join the whole document into one line and defeat that scoping.
    result = re.sub(r"[^\S\n]+", " ", result)
    result = re.sub(r" ([,;:!?.\)\]\}])", r"\1", result)
    return result


# Used by FormattingRule to detect Unicode-encoded sup/sub alongside HTML tags
# ---------------------------------------------------------------------------
_UNICODE_SUPERSCRIPT_CHARS = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ")

_UNICODE_SUBSCRIPT_CHARS = set("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")


class FormattingRule(ParseTestRule):
    """Test rule to verify that specific text has (or lacks) a formatting style.

    Each formatting type defines regex patterns to detect formatting markers
    (markdown syntax or HTML tags) wrapping the target text. The rule searches
    the *raw* markdown content (before normalize_text strips markers) so that
    formatting information is still available.

    Supported formatting kinds and their detection patterns:
    - bold:      **text**, <b>text</b>, or <strong>text</strong>
    - italic:    *text* or _text_ or <i>text</i>
    - underline: <u>text</u> or <ins>text</ins>
    - strikeout: ~~text~~
    - mark:      <mark>text</mark>
    - sup:       <sup>text</sup> or Unicode superscript characters
    - sub:       <sub>text</sub> or Unicode subscript characters
    """

    # Map formatting kind -> list of regex-builder functions.
    # Each function receives the escaped query text and returns a compiled
    # pattern that should match the formatted occurrence in raw content.
    _FORMATTING_PATTERNS: dict[str, list] = {}  # populated below after class body

    def __init__(self, rule_data: ParseFormattingRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseFormattingRule, self._rule_data)

        # Derive formatting kind and polarity from the test type value.
        # E.g. "is_bold" -> kind="bold", expect_present=True
        #      "is_not_bold" -> kind="bold", expect_present=False
        type_value = self.type
        if type_value.startswith("is_not_"):
            self.formatting_kind = type_value[len("is_not_") :]
            self.expect_present = False
        elif type_value.startswith("is_"):
            self.formatting_kind = type_value[len("is_") :]
            self.expect_present = True
        else:
            raise ValueError(f"Invalid type for FormattingRule: {type_value}")

        if self.formatting_kind not in self._FORMATTING_PATTERNS:
            raise ValueError(f"Unsupported formatting kind: {self.formatting_kind}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

    # ------------------------------------------------------------------
    # Pattern builders – called with re.escape(query) to build detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bold_patterns(escaped_query: str) -> list[re.Pattern | EmphasisSpanMatcher]:
        """Detect **text**, <b>/<strong> text, or markdown headings (# text).

        The query may be a substring of the bold span (e.g. query
        ``Population`` matches ``**Population:**``).
        """
        return [
            # ** spans come from delimiter-run pairing, so the query can only
            # ever match text that sits between an opener and its own closer.
            EmphasisSpanMatcher(("bold",), escaped_query),
            # Tag pairs may wrap multiple paragraphs, so the query's word gaps
            # go back to \s+ here: containment is scoped by the closing tag.
            re.compile(
                r"<b>" + _tempered(r"</b>") + _paragraph_spanning(escaped_query) + _tempered(r"</b>") + r"</b>",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"<strong>"
                + _tempered(r"</strong>")
                + _paragraph_spanning(escaped_query)
                + _tempered(r"</strong>")
                + r"</strong>",
                re.IGNORECASE | re.DOTALL,
            ),
            # Heading text must live on the heading's own line: no DOTALL, the
            # intra-line separators use [ \t] because \s matches newlines
            # regardless of the DOTALL flag, and the query itself is confined
            # so its own word gaps cannot continue past the heading line.
            re.compile(
                r"^[ \t]*#{1,6}[ \t]+[^\n]*?" + _confined_to_line(escaped_query) + r"[^\n]*?[ \t]*(?:#+[ \t]*)?$",
                re.IGNORECASE | re.MULTILINE,
            ),
        ]

    @staticmethod
    def _build_italic_patterns(escaped_query: str) -> list[re.Pattern | EmphasisSpanMatcher]:
        """Detect *text* (not **), _text_ (not __), or <i>text</i>.

        The query may be a substring of the italic span (e.g. query
        ``Grazing Line`` matches ``*Grazing Line, Macquarie Marshes, NSW*``).
        """
        return [
            # Lone-* and lone-_ spans from delimiter-run pairing. Run length
            # is exact, so ** and __ never feed the italic arm.
            EmphasisSpanMatcher(("italic_star", "italic_under"), escaped_query),
            # Tag pairs may wrap multiple paragraphs: paragraph-spanning gaps.
            re.compile(
                r"<i>" + _tempered(r"</i>") + _paragraph_spanning(escaped_query) + _tempered(r"</i>") + r"</i>",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"<em>" + _tempered(r"</em>") + _paragraph_spanning(escaped_query) + _tempered(r"</em>") + r"</em>",
                re.IGNORECASE | re.DOTALL,
            ),
        ]

    @staticmethod
    def _build_underline_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect <u>text</u> or <ins>text</ins>.

        The rule text may be a substring of the underlined span. This mirrors
        the other inline formatting rules: a focused assertion such as
        ``intellectual`` should match when the parser correctly wraps the
        complete sentence containing it. Tag pairs may wrap multiple
        paragraphs, so the closing tag scopes containment.
        """
        query = _paragraph_spanning(escaped_query)
        return [
            re.compile(
                r"<u>" + _tempered(r"</u>") + query + _tempered(r"</u>") + r"</u>",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"<ins>" + _tempered(r"</ins>") + query + _tempered(r"</ins>") + r"</ins>",
                re.IGNORECASE | re.DOTALL,
            ),
        ]

    @staticmethod
    def _build_strikeout_patterns(escaped_query: str) -> list[re.Pattern | EmphasisSpanMatcher]:
        """Detect the query inside one continuous strikeout span."""
        query = _paragraph_spanning(escaped_query)
        return [
            # ~~ is GFM strikethrough - markdown emphasis, so it pairs and
            # scopes exactly like the ** arm: paragraph-confined spans, inner
            # padding (``~~ text ~~``) accepted on the same terms.
            EmphasisSpanMatcher(("strikeout",), escaped_query),
            # HTML strike tags may carry attributes and wrap multiple
            # paragraphs. Backreference the opener so malformed mixed pairs
            # do not count as producer-supported strikeout markup.
            re.compile(
                r"<(s|del|strike)\b[^>]*>" + _tempered(r"</\1>") + query + _tempered(r"</\1>") + r"</\1>",
                re.IGNORECASE | re.DOTALL,
            ),
            # Some producers retain CSS line-through instead of translating
            # it to Markdown or a semantic strike tag.
            re.compile(
                r"<(\w+)\b(?=[^>]*text-decoration\s*:\s*[^>]*line-through)[^>]*>"
                + _tempered(r"</\1>")
                + query
                + _tempered(r"</\1>")
                + r"</\1>",
                re.IGNORECASE | re.DOTALL,
            ),
        ]

    @staticmethod
    def _build_mark_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect ``<mark>`` with or without attributes around the target text."""
        return [
            re.compile(r"<mark(?:\s+[^>]*)?>" + _paragraph_spanning(escaped_query) + r"</mark>", re.IGNORECASE),
        ]

    @staticmethod
    def _build_sup_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect <sup>text</sup>. Unicode superscripts handled separately."""
        return [
            re.compile(r"<sup[^>]*>" + _paragraph_spanning(escaped_query) + r"</sup>", re.IGNORECASE),
        ]

    @staticmethod
    def _build_sub_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect <sub>text</sub>. Unicode subscripts handled separately."""
        return [
            re.compile(r"<sub[^>]*>" + _paragraph_spanning(escaped_query) + r"</sub>", re.IGNORECASE),
        ]

    def _has_unicode_superscript(self, content: str) -> bool:
        """Check if content contains consecutive Unicode superscript chars
        that correspond to self.text (case-insensitive)."""
        query_lower = self.text.lower()
        # Build a string of superscript chars found consecutively
        for i in range(len(content)):
            if content[i] in _UNICODE_SUPERSCRIPT_CHARS:
                # Collect the full run of superscript characters
                run = []
                j = i
                while j < len(content) and content[j] in _UNICODE_SUPERSCRIPT_CHARS:
                    run.append(content[j])
                    j += 1
                # Transliterate to plain ASCII/Latin and compare
                transliterated = unicodedata.normalize("NFKD", "".join(run)).lower()
                if query_lower in transliterated:
                    return True
        return False

    def _has_unicode_subscript(self, content: str) -> bool:
        """Check if content contains consecutive Unicode subscript chars
        that correspond to self.text (case-insensitive)."""
        query_lower = self.text.lower()
        for i in range(len(content)):
            if content[i] in _UNICODE_SUBSCRIPT_CHARS:
                run = []
                j = i
                while j < len(content) and content[j] in _UNICODE_SUBSCRIPT_CHARS:
                    run.append(content[j])
                    j += 1
                transliterated = unicodedata.normalize("NFKD", "".join(run)).lower()
                if query_lower in transliterated:
                    return True
        return False

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if the target text has the expected formatting in raw markdown.

        We search the *raw* md_content (not the normalized version) because
        normalize_text strips all formatting markers.

        The query is made markup-tolerant so that clean rule text (e.g.
        ``"hello world"``) still matches raw content where the words are
        wrapped in nested formatting (e.g. ``"hello ~~world~~"``).

        As a fallback, other formatting markers are stripped from the content
        while keeping the markers for the kind being tested, so the regex
        can match even when nested markup is adjacent to words (no space).

        Both sides are quote-folded first. Content rules get that folding from
        ``normalize_text``; matching raw markdown skips it, so an ASCII-quoted
        rule could not match a faithful parse of a page printed with
        typographic quotes (``"Bank"`` vs ``“Bank”``, ``Kellogg's`` vs
        ``Kellogg’s``) — ground truth routinely failed its own styling rules
        on nothing but the quote encoding. The fold is a 1:1 character
        translation, so span boundaries and offsets are untouched.
        """
        # The query is nothing but markdown markers (harvested from an ASCII
        # banner), so it identifies no content and no output could ever satisfy
        # it. Skip rather than score 0.0 - see ``is_degenerate_marker_text``.
        # This covers both polarities: a degenerate ``is_not_*`` query would
        # otherwise bank a free pass for an equally undefined reason.
        if is_degenerate_marker_text(self.text):
            raise RuleNotApplicable(f"{self.type} text {self.text!r} is markdown markers only")

        # Un-escape markdown backslash sequences (e.g. \~ → ~) so that
        # clean rule text can match content produced by parsers that emit
        # backslash-escaped punctuation.
        md_clean = _normalize_quotes(_MD_BACKSLASH_ESCAPE_RE.sub(r"\1", md_content))

        escaped_query = re.escape(_normalize_quotes(self.text))
        # Allow flexible whitespace *and* optional inline markup between words
        flexible_query = _make_markup_tolerant(escaped_query)

        patterns = self._FORMATTING_PATTERNS[self.formatting_kind](flexible_query)  # type: ignore[operator]
        found = any(p.search(md_clean) for p in patterns)

        # Fallback: strip OTHER formatting from content, keeping only the kind
        # being tested, then retry with a simple flexible-whitespace pattern.
        if not found:
            stripped = _strip_other_formatting(md_clean, self.formatting_kind)
            # str.replace, not re.sub: _WORD_WS contains \s, which re.sub
            # rejects as a bad escape in a replacement template.
            simple_query = escaped_query.replace(r"\ ", _WORD_WS)
            simple_patterns = self._FORMATTING_PATTERNS[self.formatting_kind](simple_query)  # type: ignore[operator]
            found = any(p.search(stripped) for p in simple_patterns)

        # For sup/sub, also check Unicode superscript/subscript characters
        if not found and self.formatting_kind == "sup":
            found = self._has_unicode_superscript(md_content)
        if not found and self.formatting_kind == "sub":
            found = self._has_unicode_subscript(md_content)

        if self.expect_present:
            if found:
                return True, ""
            return (
                False,
                f"Expected '{self.text[:40]}' to be formatted as {self.formatting_kind}, "
                f"but no {self.formatting_kind} formatting found",
            )
        else:
            if not found:
                return True, ""
            return (
                False,
                f"Expected '{self.text[:40]}' NOT to be formatted as {self.formatting_kind}, "
                f"but unexpectedly had {self.formatting_kind} formatting",
            )


# Wire up pattern builders after class body is defined
FormattingRule._FORMATTING_PATTERNS = {
    "bold": FormattingRule._build_bold_patterns,  # type: ignore[dict-item]
    "italic": FormattingRule._build_italic_patterns,  # type: ignore[dict-item]
    "underline": FormattingRule._build_underline_patterns,  # type: ignore[dict-item]
    "strikeout": FormattingRule._build_strikeout_patterns,  # type: ignore[dict-item]
    "mark": FormattingRule._build_mark_patterns,  # type: ignore[dict-item]
    "sup": FormattingRule._build_sup_patterns,  # type: ignore[dict-item]
    "sub": FormattingRule._build_sub_patterns,  # type: ignore[dict-item]
}

# Collect all formatting TestType values handled by FormattingRule
_FORMATTING_TEST_TYPES = {
    TestType.IS_UNDERLINE.value,
    TestType.IS_NOT_UNDERLINE.value,
    TestType.IS_BOLD.value,
    TestType.IS_NOT_BOLD.value,
    TestType.IS_STRIKEOUT.value,
    TestType.IS_NOT_STRIKEOUT.value,
    TestType.IS_ITALIC.value,
    TestType.IS_NOT_ITALIC.value,
    TestType.IS_MARK.value,
    TestType.IS_NOT_MARK.value,
    TestType.IS_SUP.value,
    TestType.IS_NOT_SUP.value,
    TestType.IS_SUB.value,
    TestType.IS_NOT_SUB.value,
    TestType.MARK_COLOR.value,
    TestType.TEXT_COLOR.value,
}


# Regex to match <mark ...>text</mark> capturing the opening tag attributes and inner text.
_MARK_TAG_PATTERN = re.compile(
    r"<mark\b([^>]*)>([\s\S]+?)</mark>",
    re.IGNORECASE,
)


class MarkColorRule(ParseTestRule):
    """Test rule to verify that text is wrapped in a ``<mark>`` tag with a specific color.

    Passes when:
    1. The text is found inside a ``<mark>`` tag, AND
    2. The ``<mark>`` tag contains the expected color string in any of its attributes
       (e.g. ``style="background-color: yellow"``, ``background="yellow"``,
       ``backgroundColor="yellow"``).
    """

    def __init__(self, rule_data: ParseMarkColorRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseMarkColorRule, self._rule_data)

        if self.type != TestType.MARK_COLOR.value:
            raise ValueError(f"Invalid type for MarkColorRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

        raw_color = rule_data.color
        if not raw_color.strip():
            raise ValueError("Color field cannot be empty")
        self.color = raw_color.strip().lower()

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if text is inside a <mark> tag that contains the expected color."""
        escaped_query = re.escape(self.text)
        # A <mark> tag pair may wrap multiple paragraphs, so the query's word
        # gaps must be paragraph-spanning to match anywhere in the inner text.
        flexible_query = _paragraph_spanning(_make_markup_tolerant(escaped_query))
        text_pattern = re.compile(flexible_query, re.IGNORECASE)

        for match in _MARK_TAG_PATTERN.finditer(md_content):
            attrs_str = match.group(1)
            inner_text = match.group(2)

            # Check if the target text is inside this <mark> tag
            if not text_pattern.search(inner_text):
                continue

            # Check if the color string appears in the tag attributes
            if self.color in attrs_str.lower():
                return True, ""

        # Fallback: strip other formatting and retry
        stripped = _strip_other_formatting(md_content, "mark")
        for match in _MARK_TAG_PATTERN.finditer(stripped):
            attrs_str = match.group(1)
            inner_text = match.group(2)

            if not text_pattern.search(inner_text):
                continue

            if self.color in attrs_str.lower():
                return True, ""

        return (
            False,
            f"Expected '{self.text[:40]}' to be inside a <mark> tag with color '{self.color}', "
            f"but no matching <mark> tag found",
        )


# ---------------------------------------------------------------------------
# absent_unless_strikeout rule
# ---------------------------------------------------------------------------

# Regions of raw markdown that carry explicit strikeout markup. Text inside
# these regions is "marked as deleted" and therefore allowed to remain.
_STRUCK_REGION_PATTERNS = [
    re.compile(r"~~[\s\S]+?~~"),
    re.compile(r"<(s|del|strike)\b[^>]*>[\s\S]*?</\1>", re.IGNORECASE),
    re.compile(
        r"<(\w+)\b[^>]*text-decoration\s*:\s*line-through[^>]*>[\s\S]*?</\1>",
        re.IGNORECASE,
    ),
]


class AbsentUnlessStrikeoutRule(ParseTestRule):
    """Deleted (struck) source text must not surface as regular content.

    Passes when the text is absent from the output entirely, or when every
    occurrence sits inside strikeout markup. Fails when the text appears as
    plain, unmarked content — the case where deleted contract language reads
    identically to current language downstream.
    """

    def __init__(self, rule_data: ParseAbsentUnlessStrikeoutRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseAbsentUnlessStrikeoutRule, self._rule_data)

        if self.type != TestType.ABSENT_UNLESS_STRIKEOUT.value:
            raise ValueError(f"Invalid type for AbsentUnlessStrikeoutRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        # Remove every explicitly-struck region, then look for the text in
        # what remains. Whatever survives the removal is unmarked content.
        visible = _MD_BACKSLASH_ESCAPE_RE.sub(r"\1", md_content)
        for pattern in _STRUCK_REGION_PATTERNS:
            visible = pattern.sub(" ", visible)

        norm_query = normalize_text(self.text)
        norm_visible = normalize_text(visible)
        if norm_query and norm_query in norm_visible:
            return (
                False,
                f"Deleted text '{self.text[:40]}' appears as regular content (not struck through and not removed)",
            )
        return True, ""


class PresentAsStrikeoutRule(ParseTestRule):
    """Deleted (struck) source text must be retained AND marked as struck.

    Stricter counterpart of :class:`AbsentUnlessStrikeoutRule`: dropping the
    text also fails. Passes only when the text appears inside strikeout
    markup somewhere in the output. Region-based, so a struck span wider
    than the rule text passes.
    """

    def __init__(self, rule_data: ParsePresentAsStrikeoutRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParsePresentAsStrikeoutRule, self._rule_data)

        if self.type != TestType.PRESENT_AS_STRIKEOUT.value:
            raise ValueError(f"Invalid type for PresentAsStrikeoutRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        unescaped = _MD_BACKSLASH_ESCAPE_RE.sub(r"\1", md_content)
        struck_parts: list[str] = []
        for pattern in _STRUCK_REGION_PATTERNS:
            struck_parts.extend(m.group(0) for m in pattern.finditer(unescaped))

        norm_query = normalize_text(self.text)
        if norm_query and norm_query in normalize_text(" ".join(struck_parts)):
            return True, ""

        if norm_query and norm_query in normalize_text(unescaped):
            return (
                False,
                f"Deleted text '{self.text[:40]}' is present but not marked as struck",
            )
        return (
            False,
            f"Deleted text '{self.text[:40]}' was dropped from the output (expected retained inside strikeout markup)",
        )


# ---------------------------------------------------------------------------
# text_color rule
# ---------------------------------------------------------------------------

# Any HTML tag with attributes wrapping inner text. Tag-agnostic because
# parsers may express colored text via <span>, <font>, <mark>, <del>, <ins>, …
_COLORABLE_TAG_PATTERN = re.compile(
    r"<(\w+)\b([^>]*)>([\s\S]+?)</\1>",
    re.IGNORECASE,
)

# Color value candidates inside a tag's attribute string: hex codes, rgb()
# triples, or values following a color-ish attribute (style="color: red",
# color="red", data-text-color="red").
_HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_RGB_COLOR_RE = re.compile(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_NAMED_COLOR_ATTR_RE = re.compile(r"color\s*[:=]\s*[\"']?\s*([a-zA-Z]+)", re.IGNORECASE)

# Representative RGB for named colors parsers plausibly emit.
_NAMED_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "darkred": (139, 0, 0),
    "crimson": (220, 20, 60),
    "salmon": (250, 128, 114),
    # Palette names LlamaParse's structured `colors` annotation normalizes to
    # (rehydrated into md by color_annotations.py) that were missing here.
    "lightcoral": (240, 128, 128),
    "goldenrod": (218, 165, 32),
    "olive": (128, 128, 0),
    "orange": (255, 165, 0),
    "amber": (255, 191, 0),
    "gold": (255, 215, 0),
    "yellow": (255, 255, 0),
    "green": (0, 128, 0),
    "darkgreen": (0, 100, 0),
    "lime": (0, 255, 0),
    "teal": (0, 128, 128),
    "cyan": (0, 255, 255),
    "blue": (0, 0, 255),
    "navy": (0, 0, 128),
    "royalblue": (65, 105, 225),
    "purple": (128, 0, 128),
    "violet": (238, 130, 238),
    "magenta": (255, 0, 255),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "maroon": (128, 0, 0),
}


def _rgb_to_hue_family(r: int, g: int, b: int) -> str | None:
    """Classify an RGB value into a coarse hue family; None for grayscale."""
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 30:  # too desaturated to carry revision meaning
        return None
    # Hue in degrees (0-360)
    d = float(mx - mn)
    if mx == r:
        h = (60.0 * ((g - b) / d)) % 360.0
    elif mx == g:
        h = 60.0 * ((b - r) / d) + 120.0
    else:
        h = 60.0 * ((r - g) / d) + 240.0
    if h < 20 or h >= 335:
        return "red"
    if h < 48:
        return "orange"
    if h < 72:
        return "yellow"
    if h < 165:
        return "green"
    if h < 200:
        return "cyan"
    if h < 262:
        return "blue"
    return "purple"


# Families close enough on the hue wheel that a parser naming either is
# accepted (e.g. amber #c69200 may reasonably be called orange or yellow).
_ADJACENT_FAMILIES: set[frozenset[str]] = {
    frozenset({"red", "orange"}),
    frozenset({"orange", "yellow"}),
    frozenset({"green", "cyan"}),
    frozenset({"cyan", "blue"}),
    frozenset({"blue", "purple"}),
    frozenset({"purple", "red"}),
}


def _families_match(expected: str, found: str) -> bool:
    if expected == found:
        return True
    return frozenset({expected, found}) in _ADJACENT_FAMILIES


def _extract_color_families(attrs_str: str) -> set[str]:
    """Collect hue families from every color-like value in a tag attribute string."""
    families: set[str] = set()
    for m in _HEX_COLOR_RE.finditer(attrs_str):
        hx = m.group(1)
        if len(hx) == 3:
            hx = "".join(c * 2 for c in hx)
        fam = _rgb_to_hue_family(int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
        if fam:
            families.add(fam)
    for m in _RGB_COLOR_RE.finditer(attrs_str):
        fam = _rgb_to_hue_family(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if fam:
            families.add(fam)
    for m in _NAMED_COLOR_ATTR_RE.finditer(attrs_str):
        name = m.group(1).lower()
        if name in _NAMED_COLOR_RGB:
            fam = _rgb_to_hue_family(*_NAMED_COLOR_RGB[name])
            if fam:
                families.add(fam)
    return families


class TextColorRule(ParseTestRule):
    """Test rule verifying that text carries a specific text color.

    Passes when the text is found inside any HTML tag whose attributes carry a
    color in the expected hue family. The expected ``color`` is a family name
    (``red``, ``orange``, ``yellow``, ``green``, ``cyan``, ``blue``,
    ``purple``); attribute values may be named colors, hex codes or ``rgb()``
    triples, and adjacent hue families are accepted.
    """

    def __init__(self, rule_data: ParseTextColorRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTextColorRule, self._rule_data)

        if self.type != TestType.TEXT_COLOR.value:
            raise ValueError(f"Invalid type for TextColorRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

        raw_color = rule_data.color
        if not raw_color.strip():
            raise ValueError("Color field cannot be empty")
        self.color = raw_color.strip().lower()

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        escaped_query = re.escape(self.text)
        # Colorable tag pairs may wrap multiple paragraphs: the query's word
        # gaps must be paragraph-spanning to match anywhere in the inner text.
        flexible_query = _paragraph_spanning(_make_markup_tolerant(escaped_query))
        text_pattern = re.compile(flexible_query, re.IGNORECASE)

        found_families: set[str] = set()
        for match in _COLORABLE_TAG_PATTERN.finditer(md_content):
            attrs_str = match.group(2)
            inner_text = match.group(3)
            if not attrs_str.strip():
                continue
            if not text_pattern.search(inner_text):
                continue
            families = _extract_color_families(attrs_str)
            found_families |= families
            if any(_families_match(self.color, f) for f in families):
                return True, ""

        if found_families:
            return (
                False,
                f"Expected '{self.text[:40]}' to have text color '{self.color}', "
                f"but found color families {sorted(found_families)} instead",
            )
        return (
            False,
            f"Expected '{self.text[:40]}' to have text color '{self.color}', but no colored markup found around it",
        )


def _strip_latex_delimiters(formula: str) -> str:
    stripped = formula.strip()
    if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        return stripped[1:-1].strip()
    if stripped.startswith(r"\(") and stripped.endswith(r"\)") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    if stripped.startswith(r"\[") and stripped.endswith(r"\]") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    return stripped


def _normalize_latex_formula(formula: str) -> str:
    body = _strip_latex_delimiters(formula)
    body = _unescape_html_entities(body)
    # These commands only control presentation. The benchmark cares about the
    # mathematical expression, not delimiter sizing, display style, spacing,
    # or whether an equation number was emitted as ``\tag`` versus ``\eqno``.
    body = re.sub(r"\\(?:left|right)(?=\s|[()\[\]{}|.]|$)", "", body)
    body = re.sub(r"\\[dt]frac\b", r"\\frac", body)
    body = re.sub(r"\\(?:leq|le)\b", r"\\le", body)
    body = re.sub(r"\\(?:geq|ge)\b", r"\\ge", body)
    body = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", body)
    body = re.sub(r"\\(?:display|text|script|scriptscript)style\b", "", body)
    body = re.sub(r"\\(?:quad|qquad|,|;|:|!|>|enspace|hspace\*?\{[^{}]*\})", "", body)
    body = re.sub(r"\\[ \t]+", "", body)
    body = re.sub(r"\\tag\s*\{[^{}]*\}", "", body)
    body = re.sub(r"\\eqno\s*(?:\([^()]*\)|\{[^{}]*\}|\S+)", "", body)
    for unicode_operator, latex_operator in (
        ("−", "-"),
        ("×", r"\times"),
        ("÷", r"\div"),
        ("≤", r"\le"),
        ("≥", r"\ge"),
    ):
        body = body.replace(unicode_operator, latex_operator)
    body = re.sub(r"\s+", "", body)
    return body


def _looks_like_inline_latex(body: str) -> bool:
    """Reject prose/currency accidentally paired by two dollar signs.

    Inline math cannot cross a Markdown line or an HTML tag boundary. Within
    that scope, a multi-word body needs an actual LaTeX/math signal; this keeps
    ``$10 million ... $20 million`` from becoming one giant fake formula while
    retaining valid simple expressions such as ``$x$``.
    """

    stripped = body.strip()
    if not stripped:
        return False
    if not re.search(r"\s", stripped):
        return True
    return bool(re.search(r"\\[A-Za-z]+|[=+\-*/^_<>|{}]", stripped))


def _extract_latex_formula_spans(md_content: str, *, include_implausible_inline: bool = False) -> list[tuple[str, str]]:
    formulas: list[tuple[str, str]] = []
    block_dollar = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)
    # Inline math is line-local and cannot consume HTML markup. A bounded body
    # also prevents one unmatched currency sign from pairing far downstream.
    inline_dollar = re.compile(r"(?<!\\)\$(?!\$)([^$\r\n]{1,500}?)(?<!\\)\$(?!\$)")
    inline_paren = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
    block_bracket = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)

    for candidate in _title_content_candidates(md_content):
        for kind, pattern, search_areas in (
            ("block", block_dollar, (candidate,)),
            # Split only at syntactically plausible HTML tags. A raw ``<`` or
            # ``>`` is common inside math and must not break dollar pairing,
            # while tag boundaries must never let currency in adjacent table
            # cells masquerade as one inline formula.
            (
                "inline",
                inline_dollar,
                re.split(r"<[/!?A-Za-z][^>\r\n]*>", candidate),
            ),
            ("inline", inline_paren, (candidate,)),
            ("block", block_bracket, (candidate,)),
        ):
            for search_area in search_areas:
                for match in pattern.finditer(search_area):
                    body = match.group(1)
                    if kind == "inline" and not include_implausible_inline and not _looks_like_inline_latex(body):
                        continue
                    normalized = _normalize_latex_formula(body)
                    if normalized:
                        formulas.append((body, normalized))
    return formulas


def _extract_latex_formulas(md_content: str) -> set[str]:
    return {normalized for _, normalized in _extract_latex_formula_spans(md_content)}


def _normalize_formula_text_for_negative_match(text: str) -> str:
    text = _strip_latex_delimiters(_unescape_html_entities(text))
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"\_", "_").replace(r"\$", "$")
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    for marker in "{}$":
        text = text.replace(marker, " ")
    return normalize_text(text).strip()


class LatexRule(ParseTestRule):
    """Check that a specific inline or block LaTeX formula is present."""

    def __init__(self, rule_data: ParseLatexRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseLatexRule, self._rule_data)

        if self.type != TestType.IS_LATEX.value:
            raise ValueError(f"Invalid type for LatexRule: {self.type}")

        raw_formula = rule_data.formula
        if not isinstance(raw_formula, str) or not raw_formula.strip():
            raise ValueError("formula must be a non-empty string")

        self.formula = raw_formula.strip()
        self.normalized_formula = _normalize_latex_formula(self.formula)
        if not self.normalized_formula:
            raise ValueError("formula is empty after normalization")

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        found_formulas = _extract_latex_formulas(md_content)
        if self.normalized_formula in found_formulas:
            return True, ""

        preview = ", ".join(sorted(found_formulas)[:3])
        placeholder_hint = ""
        if "LATEX" in md_content and not found_formulas:
            placeholder_hint = (
                " Content appears to contain 'LATEX' placeholder tokens "
                "and no raw formula delimiters,"
                " suggesting upstream preprocessing replaced formulas before this rule."
            )
        return (
            False,
            (
                f"Expected LaTeX formula '{self.formula[:80]}' not found. "
                f"Detected formulas (normalized preview): {preview}"
                if preview
                else f"Expected LaTeX formula '{self.formula[:80]}' not found.{placeholder_hint}"
            ),
        )


class NotLatexRule(ParseTestRule):
    """Check that formula-like source text was not wrapped as LaTeX."""

    def __init__(self, rule_data: ParseNotLatexRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseNotLatexRule, self._rule_data)
        if self.type != TestType.IS_NOT_LATEX.value:
            raise ValueError(f"Invalid type for NotLatexRule: {self.type}")
        if not isinstance(rule_data.text, str) or not rule_data.text.strip():
            raise ValueError("text must be a non-empty string")
        self.text = rule_data.text.strip()
        self.normalized_text = _normalize_formula_text_for_negative_match(self.text)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        # Negative rules are intentionally stricter than positive discovery:
        # even an implausible ``$400 billion$`` span is wrong attribution for
        # an annotated currency target and must not be ignored as prose.
        for body, _ in _extract_latex_formula_spans(md_content, include_implausible_inline=True):
            normalized_body = _normalize_formula_text_for_negative_match(body)
            if self.normalized_text and self.normalized_text in normalized_body:
                return False, f"Expected '{self.text[:80]}' to remain ordinary text, but it was emitted as LaTeX"
        return True, ""


def _extract_fenced_code_blocks(md_content: str) -> list[tuple[str, str]]:
    """Extract markdown fenced code blocks as (language, code)."""
    blocks: list[tuple[str, str]] = []
    # ```lang\n...\n``` (language optional)
    pattern = re.compile(r"(?ms)^[ \t]*```(?P<lang>[^\n`]*)\n(?P<body>.*?)[ \t]*\n[ \t]*```[ \t]*$")

    for candidate in _title_content_candidates(md_content):
        for match in pattern.finditer(candidate):
            lang = match.group("lang").strip().lower()
            body = match.group("body").strip()
            blocks.append((lang, body))
    return blocks


class CodeBlockRule(ParseTestRule):
    """Check that a fenced code block with a given language contains target code.

    Matching is permissive: whitespace is collapsed before comparison so that
    minor indentation or line-break differences do not cause failures.
    """

    _WS_COLLAPSE = re.compile(r"\s+")

    def __init__(self, rule_data: ParseCodeBlockRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseCodeBlockRule, self._rule_data)

        if self.type != TestType.IS_CODE_BLOCK.value:
            raise ValueError(f"Invalid type for CodeBlockRule: {self.type}")

        raw_language = rule_data.language
        raw_code = rule_data.code

        if not isinstance(raw_language, str) or not raw_language.strip():
            raise ValueError("language must be a non-empty string")
        if not isinstance(raw_code, str) or not raw_code.strip():
            raise ValueError("code must be a non-empty string")

        self.language = raw_language.strip().lower()
        self.code = raw_code.strip()
        self._code_normalized = self._WS_COLLAPSE.sub(" ", self.code)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        blocks = _extract_fenced_code_blocks(md_content)

        matching_lang_blocks = [body for lang, body in blocks if lang == self.language]
        if not matching_lang_blocks:
            available = ", ".join(sorted({lang for lang, _ in blocks if lang}))
            return (
                False,
                (
                    f"No fenced code block found with language '{self.language}'."
                    + (f" Available languages: {available}" if available else "")
                ),
            )

        for body in matching_lang_blocks:
            # Exact substring first, then whitespace-normalized fallback
            if self.code in body:
                return True, ""
            if self._code_normalized in self._WS_COLLAPSE.sub(" ", body):
                return True, ""

        return (
            False,
            f"Found '{self.language}' code block(s), but none contained snippet '{self.code[:80]}'",
        )


def _title_content_candidates(md_content: str) -> list[str]:
    """Return raw + decoded/de-escaped content variants for title matching."""
    candidates: list[str] = []

    def _append_unique(value: str) -> None:
        if value not in candidates:
            candidates.append(value)

    _append_unique(md_content)
    unescaped_content = _unescape_html_entities(md_content)
    _append_unique(unescaped_content)

    markdown_unescaped = re.sub(r"(?m)^\\(#{1,6}\s+)", r"\1", unescaped_content)
    markdown_unescaped = markdown_unescaped.replace(r"\*\*", "**")
    _append_unique(markdown_unescaped)
    return candidates


def _normalize_title_label(text: str) -> str:
    """Normalize title text for case/spacing-insensitive comparisons."""
    return normalize_text(text).strip()


def _extract_title_events(md_content: str) -> list[tuple[int, int, str]]:
    """Extract title events as (line_index, level, normalized_text).

    Levels: 1-6 for heading levels, 7 for bold-title lines (lowest title level).
    A bold title line must start at line beginning and contain only the bold text.
    """
    events: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()

    html_heading_regex = re.compile(r"<h([1-6])[^>]*>\s*(.*?)\s*</h\1>", re.IGNORECASE)
    html_bold_line_regex = re.compile(
        r"^\s*<(?P<tag>b|strong)[^>]*>\s*(?P<text>.*?)\s*</(?P=tag)>\s*$",
        re.IGNORECASE,
    )
    md_heading_regex = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
    md_bold_line_regex = re.compile(r"^\s*\*\*\s*(.+?)\s*\*\*\s*$")

    for candidate in _title_content_candidates(md_content):
        for line_idx, line in enumerate(candidate.splitlines()):
            md_heading_match = md_heading_regex.match(line)
            if md_heading_match:
                level = len(md_heading_match.group(1))
                title_text = re.sub(r"\s+#+\s*$", "", md_heading_match.group(2)).strip()
                normalized = _normalize_title_label(title_text)
                if normalized:
                    key = (line_idx, level, normalized)
                    if key not in seen:
                        seen.add(key)
                        events.append(key)

            for match in html_heading_regex.finditer(line):
                level = int(match.group(1))
                title_text = re.sub(r"<[^>]+>", " ", match.group(2)).strip()
                normalized = _normalize_title_label(title_text)
                if normalized:
                    key = (line_idx, level, normalized)
                    if key not in seen:
                        seen.add(key)
                        events.append(key)

            md_bold_match = md_bold_line_regex.match(line)
            if md_bold_match:
                normalized = _normalize_title_label(md_bold_match.group(1))
                if normalized:
                    key = (line_idx, 7, normalized)
                    if key not in seen:
                        seen.add(key)
                        events.append(key)

            html_bold_match = html_bold_line_regex.match(line)
            if html_bold_match:
                bold_text = re.sub(r"<[^>]+>", " ", html_bold_match.group("text")).strip()
                normalized = _normalize_title_label(bold_text)
                if normalized:
                    key = (line_idx, 7, normalized)
                    if key not in seen:
                        seen.add(key)
                        events.append(key)

    events.sort(key=lambda item: (item[0], item[1], item[2]))
    return events


class TitleLevelRule(ParseTestRule):
    """Test rule to verify that text appears as a title.

    A title is satisfied if the text appears either:
    - as a markdown/HTML heading (`#`, `##`, ..., `<h1>`, ..., `<h6>`), or
    - as bold text (`**text**`, `<b>text</b>`, or `<strong>text</strong>`).

    The `level` field is currently ignored for matching; any heading level
    (1-6) or standalone bold title line can satisfy the rule.
    """

    def __init__(self, rule_data: ParseTitleRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTitleRule, self._rule_data)

        if self.type != TestType.IS_TITLE.value:
            raise ValueError(f"Invalid type for TitleLevelRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

        self.level = rule_data.level

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if text appears as heading or bold title in raw markdown.

        Also tolerates escaped markup payloads (e.g. `&lt;h1&gt;...&lt;/h1&gt;`,
        `\\# Title`, `\\*\\*Title\\*\\*`) by checking decoded/de-escaped variants.
        """
        # A title query that is nothing but markdown markers names no title -
        # it came from an ASCII banner. Skip rather than score 0.0, which would
        # otherwise pin ``normalized_title_accuracy`` on a correct parse.
        if is_degenerate_marker_text(self.text):
            raise RuleNotApplicable(f"is_title text {self.text!r} is markdown markers only")

        escaped = re.escape(self.text)
        # Allow flexible whitespace and optional inline markup between words
        flexible = _make_markup_tolerant(escaped)

        # [ \t] and a line-confined query: heading text lives on one line, and
        # both \s+ and the query's own word gaps would otherwise cross the
        # newline and match a phrase that continues into the paragraph below.
        md_heading_pattern = re.compile(
            r"^#{1,6}[ \t]+" + _confined_to_line(flexible),
            re.MULTILINE | re.IGNORECASE,
        )
        # HTML tag pairs may wrap multiple paragraphs, so their arms use
        # paragraph-spanning query gaps; the markdown ** arm keeps the scoped
        # gaps because emphasis cannot cross a blank line.
        html_heading_pattern = re.compile(
            r"<h[1-6][^>]*>\s*" + _paragraph_spanning(flexible) + r"\s*</h[1-6]>",
            re.IGNORECASE,
        )

        # Bold title forms (standalone line, bold at line beginning)
        md_bold_pattern = re.compile(
            r"^\s*\*\*\s*" + flexible + r"\s*\*\*\s*$",
            re.MULTILINE | re.IGNORECASE,
        )
        html_bold_pattern = re.compile(
            r"^\s*<(?P<tag>b|strong)[^>]*>\s*" + _paragraph_spanning(flexible) + r"\s*</(?P=tag)>\s*$",
            re.MULTILINE | re.IGNORECASE,
        )

        for candidate in _title_content_candidates(md_content):
            if (
                md_heading_pattern.search(candidate)
                or html_heading_pattern.search(candidate)
                or md_bold_pattern.search(candidate)
                or html_bold_pattern.search(candidate)
            ):
                return True, ""

        # Fallback: normalized comparison using title event extraction.
        # This catches cases where inner markup prevents literal regex matching.
        normalized_self = _normalize_title_label(self.text)
        if normalized_self:
            events = _extract_title_events(md_content)
            for _, _, normalized_title in events:
                if normalized_title == normalized_self:
                    return True, ""

        return (
            False,
            (f"Expected '{self.text[:40]}' to be a title, but no matching heading or bold formatting found"),
        )


class TitleHierarchyPercentRule(ParseTestRule):
    """Score title hierarchy compliance using expected nested title map.

    Expected hierarchy is provided via `title_hierarchy` as nested dict/list.
    Bold title lines are treated as the lowest heading level (level 7).
    """

    def __init__(self, rule_data: ParseTitleHierarchyPercentRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTitleHierarchyPercentRule, self._rule_data)
        if self.type != TestType.TITLE_HIERARCHY_PERCENT.value:
            raise ValueError(f"Invalid type for TitleHierarchyPercentRule: {self.type}")

        if not isinstance(rule_data.title_hierarchy, dict):
            raise ValueError("title_hierarchy must be a dictionary")
        # Emptiness is not rejected here: an empty hierarchy carries no
        # constraint, which ``run`` reports as inapplicable rather than failed.
        self.title_hierarchy = rule_data.title_hierarchy

    @staticmethod
    def _collect_constraints(
        hierarchy: dict[str, Any],
    ) -> tuple[set[str], list[tuple[str, str, bool]]]:
        titles: set[str] = set()
        # (parent, child, require_deeper_level)
        edges: list[tuple[str, str, bool]] = []

        def normalize_title(value: str) -> str:
            return _normalize_title_label(value)

        def walk_children(parent_title: str, children: Any) -> None:
            if children is None:
                return

            child_titles_in_order: list[str] = []

            if isinstance(children, str):
                normalized_child = normalize_title(children)
                if normalized_child:
                    titles.add(normalized_child)
                    edges.append((parent_title, normalized_child, True))
                    child_titles_in_order.append(normalized_child)
            elif isinstance(children, dict):
                for child_raw, grand_children in children.items():
                    if not isinstance(child_raw, str):
                        continue
                    normalized_child = normalize_title(child_raw)
                    if not normalized_child:
                        continue
                    titles.add(normalized_child)
                    edges.append((parent_title, normalized_child, True))
                    child_titles_in_order.append(normalized_child)
                    walk_children(normalized_child, grand_children)
            elif isinstance(children, list):
                for child in children:
                    if isinstance(child, str):
                        normalized_child = normalize_title(child)
                        if not normalized_child:
                            continue
                        titles.add(normalized_child)
                        edges.append((parent_title, normalized_child, True))
                        child_titles_in_order.append(normalized_child)
                    elif isinstance(child, dict):
                        for child_raw, grand_children in child.items():
                            if not isinstance(child_raw, str):
                                continue
                            normalized_child = normalize_title(child_raw)
                            if not normalized_child:
                                continue
                            titles.add(normalized_child)
                            edges.append((parent_title, normalized_child, True))
                            child_titles_in_order.append(normalized_child)
                            walk_children(normalized_child, grand_children)

            # Preserve sibling ordering when children are explicitly ordered.
            for i in range(len(child_titles_in_order) - 1):
                edges.append((child_titles_in_order[i], child_titles_in_order[i + 1], False))

        for root_raw, root_children in hierarchy.items():
            if not isinstance(root_raw, str):
                continue
            normalized_root = normalize_title(root_raw)
            if not normalized_root:
                continue
            titles.add(normalized_root)
            walk_children(normalized_root, root_children)

        return titles, edges

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        events = _extract_title_events(md_content)
        first_pos: dict[str, int] = {}
        first_level: dict[str, int] = {}
        for idx, level, title in events:
            if title not in first_pos:
                first_pos[title] = idx
                first_level[title] = level

        expected_titles, edges = self._collect_constraints(self.title_hierarchy)
        # No label survived normalization - the hierarchy is pure markup,
        # whitespace, non-string keys, or empty. There is no constraint to
        # check, so the rule is skipped rather than scored 0.0. (A hierarchy
        # that IS defined but absent from the output falls through below and
        # still scores 0.0, which is a real failure.)
        if not expected_titles:
            raise RuleNotApplicable("title_hierarchy has no valid titles after normalization")

        total_constraints = len(expected_titles) + len(edges)
        if total_constraints == 0:
            raise RuleNotApplicable("title_hierarchy has no evaluable constraints")

        satisfied = 0
        failures: list[str] = []

        for title in sorted(expected_titles):
            if title in first_pos:
                satisfied += 1
            else:
                failures.append(f"missing title '{title}'")

        for parent, child, require_deeper_level in edges:
            if parent not in first_pos or child not in first_pos:
                failures.append(f"missing edge '{parent}' -> '{child}'")
                continue
            parent_pos = first_pos[parent]
            child_pos = first_pos[child]
            parent_level = first_level[parent]
            child_level = first_level[child]

            order_ok = parent_pos < child_pos
            depth_ok = parent_level < child_level if require_deeper_level else True
            if order_ok and depth_ok:
                satisfied += 1
            else:
                violation_kind = "order/level" if require_deeper_level else "order"
                failures.append(
                    f"{violation_kind} violation '{parent}'(line={parent_pos},lvl={parent_level}) "
                    f"-> '{child}'(line={child_pos},lvl={child_level})"
                )

        score = max(0.0, min(1.0, satisfied / total_constraints))
        passed = score >= 0.999
        if passed:
            return True, "", score

        preview = "; ".join(failures[:5])
        return False, f"Title hierarchy score={score:.3f}; {preview}", score


_PAGE_SECTION_DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_PAGE_SECTION_DASH_PATTERN = f"[{re.escape(_PAGE_SECTION_DASH_CHARS)}]"


def _build_page_section_query_pattern(text: str) -> str:
    parts: list[str] = []
    for char in text:
        if char in _PAGE_SECTION_DASH_CHARS:
            parts.append(_PAGE_SECTION_DASH_PATTERN)
        elif char == " ":
            parts.append(r"\s+")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


class PageSectionRule(ParseTestRule):
    """Test rule to verify that text appears in page header/footer sections.

    Structured page metadata (parse_output.layout_pages) is the primary source.
    Raw markdown tag scanning is retained as a backward-compatible fallback.
    """

    # Map type value -> tag name
    _TAG_MAP = {
        TestType.IS_HEADER.value: "page_header",
        TestType.IS_FOOTER.value: "page_footer",
    }

    def __init__(self, rule_data: ParsePageSectionRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParsePageSectionRule, self._rule_data)

        if self.type not in self._TAG_MAP:
            raise ValueError(f"Invalid type for PageSectionRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()
        self.tag = self._TAG_MAP[self.type]

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if the target text appears inside the expected page section."""
        flexible = _build_page_section_query_pattern(self.text)

        # Primary path: evaluate against structured per-page sections.
        structured_sections = self._get_structured_page_sections()
        if structured_sections is not None:
            section_values = structured_sections.get(self.tag, [])
            section_pattern = re.compile(
                r"[^<]*?" + flexible + r"[^<]*?",
                re.IGNORECASE | re.DOTALL,
            )

            for section_value in section_values:
                if section_pattern.search(section_value):
                    return True, ""

            section_label = "header" if self.tag == "page_header" else "footer"
            preview = "; ".join(repr(value[:80]) for value in section_values[:3])
            return (
                False,
                (
                    f"Expected '{self.text[:40]}' to appear in structured page {section_label} "
                    f"content, but it was not found" + (f" (sample: {preview})" if preview else "")
                ),
            )

        # Backward-compatible path for artifacts/rules that only provide markdown.
        pattern = re.compile(
            r"<" + self.tag + r">" + r"[^<]*?" + flexible + r"[^<]*?" + r"</" + self.tag + r">",
            re.IGNORECASE | re.DOTALL,
        )

        if pattern.search(md_content):
            return True, ""

        section_label = "header" if self.tag == "page_header" else "footer"
        return (
            False,
            f"Expected '{self.text[:40]}' to appear inside a page {section_label} "
            f"(<{self.tag}>...</{self.tag}>), but it was not found",
        )
