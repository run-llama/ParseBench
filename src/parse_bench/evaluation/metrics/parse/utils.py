"""Utility functions for parse evaluation."""

import re
import unicodedata
from collections.abc import Callable

_SINGLE_QUOTE_CHARS = (
    "‘’‚‛"  # curly / low-9 / high-reversed-9
    "`´"  # grave accent, acute accent
    "ʼʹ＇"  # modifier apostrophe, modifier prime, fullwidth apostrophe
    "′‵"  # prime (U+2032), reversed prime (U+2035)
    "ʻˊˋ"  # turned comma (U+02BB), modifier acute (U+02CA), modifier grave (U+02CB)
)
_DOUBLE_QUOTE_CHARS = (
    "“”"  # left / right double quotation marks
    "„‟"  # double low-9 / high-reversed-9
    "〝〞"  # reversed double prime / double prime quotation
    "＂"  # fullwidth quotation mark
    "″‶"  # double prime (U+2033), reversed double prime (U+2036)
    "ˮ"  # modifier letter double apostrophe (U+02EE)
)
# Fullwidth punctuation forms (U+FF01..U+FF5E) → ASCII (U+0021..U+007E).
# Only the most common punctuation is listed explicitly; extend as needed.
_FULLWIDTH_PUNCT_CHARS = {
    "\uff0c": ",",  # fullwidth comma
    "\uff0e": ".",  # fullwidth full stop
    "\uff1a": ":",  # fullwidth colon
    "\uff1b": ";",  # fullwidth semicolon
    "\uff01": "!",  # fullwidth exclamation mark
    "\uff1f": "?",  # fullwidth question mark
    "\uff08": "(",  # fullwidth left parenthesis
    "\uff09": ")",  # fullwidth right parenthesis
    "\u3001": ",",  # ideographic comma (、)
    "\u3002": ".",  # ideographic full stop (。)
}

_QUOTE_TRANSLATION_TABLE = str.maketrans(
    {
        **dict.fromkeys(_SINGLE_QUOTE_CHARS, "'"),
        **dict.fromkeys(_DOUBLE_QUOTE_CHARS, '"'),
        **_FULLWIDTH_PUNCT_CHARS,
    }
)

# Ranges of base characters whose combining marks are load-bearing: deleting the
# mark changes the word.  Covers the Japanese voicing marks (dakuten/handakuten)
# and the abugida scripts, where vowel signs, tone marks and viramas are written
# as combining marks (Thai, Khmer, Devanagari matras, Indic virama).
# Latin / Greek / Cyrillic are deliberately absent: folding their accents to
# ASCII is the intended normalization.  Arabic/Hebrew are absent too, because
# their harakat / niqqud are optional pointing, so stripping them is forgiving.
_MARK_PRESERVING_BASE_RANGES = (
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),  # CJK Extension A
    ("\uf900", "\ufaff"),  # CJK Compatibility Ideographs
    ("\uac00", "\ud7af"),  # Hangul Syllables
    ("\u1100", "\u11ff"),  # Hangul Jamo
    ("\u0900", "\u097f"),  # Devanagari
    ("\u0980", "\u09ff"),  # Bengali
    ("\u0a00", "\u0a7f"),  # Gurmukhi
    ("\u0a80", "\u0aff"),  # Gujarati
    ("\u0b00", "\u0b7f"),  # Oriya
    ("\u0b80", "\u0bff"),  # Tamil
    ("\u0c00", "\u0c7f"),  # Telugu
    ("\u0c80", "\u0cff"),  # Kannada
    ("\u0d00", "\u0d7f"),  # Malayalam
    ("\u0d80", "\u0dff"),  # Sinhala
    ("\u0e00", "\u0e7f"),  # Thai
    ("\u0e80", "\u0eff"),  # Lao
    ("\u1000", "\u109f"),  # Myanmar
    ("\u1780", "\u17ff"),  # Khmer
)

_COMBINING_MARK_CATEGORIES = frozenset({"Mn", "Mc", "Me"})


def _is_mark_preserving_base_char(ch: str) -> bool:
    """Return True if combining marks on *ch* must survive normalization."""
    return any(lo <= ch <= hi for lo, hi in _MARK_PRESERVING_BASE_RANGES)


def _build_preserved_mark_class() -> str:
    """Regex character-class body for the marks kept by :func:`normalize_text`.

    Derived from :data:`_MARK_PRESERVING_BASE_RANGES` so the tokenizer and the
    normalizer can never drift: a mark that survives normalization is also a
    word character, and therefore never splits the word it belongs to.
    """
    codepoints = [
        cp
        for lo, hi in _MARK_PRESERVING_BASE_RANGES
        for cp in range(ord(lo), ord(hi) + 1)
        if unicodedata.category(chr(cp)) in _COMBINING_MARK_CATEGORIES
    ]
    codepoints.sort()
    parts: list[str] = []
    start = prev = codepoints[0]
    for cp in codepoints[1:] + [0]:
        if cp == prev + 1:
            prev = cp
            continue
        parts.append(f"\\u{start:04x}" if start == prev else f"\\u{start:04x}-\\u{prev:04x}")
        start = prev = cp
    return "".join(parts)


PRESERVED_COMBINING_MARK_CLASS = _build_preserved_mark_class()

# The shared word-character class.  Must be used by every tokenizer and every
# word-boundary check on both the evaluation and the annotation side so that a
# ground-truth "word" and a document token are cut the same way.
WORD_CHAR_CLASS = rf"[\w{PRESERVED_COMBINING_MARK_CLASS}]"
NON_WORD_CHAR_CLASS = rf"[^\w{PRESERVED_COMBINING_MARK_CLASS}]"


# ---------------------------------------------------------------------------
# Unicode symbol equivalence classes
#
# Each entry maps a set of visually-similar Unicode characters to a single
# canonical character.  Used by normalize_cell_text() (and transitively by
# normalize_text / normalize_text_light) so that TEDS, GriTS, and other
# cell-level comparisons treat these variants as identical.
# ---------------------------------------------------------------------------

_UNICODE_SYMBOL_CLASSES: list[tuple[str, str]] = [
    # Bullet-like dots → standard bullet (U+2022)
    (
        "●"  # U+25CF BLACK CIRCLE
        "○"  # U+25CB WHITE CIRCLE
        "◦"  # U+25E6 WHITE BULLET
        "∙"  # U+2219 BULLET OPERATOR
        "⦁"  # U+2981 Z NOTATION SPOT
        "·",  # U+00B7 MIDDLE DOT
        "•",  # U+2022 BULLET (canonical)
    ),
    # Circled x / cross marks → ⊗ (U+2297 CIRCLED TIMES)
    (
        "⮾"  # U+2BBE CIRCLED X
        "ⓧ"  # U+24E7 CIRCLED LATIN SMALL LETTER X
        "⨂",  # U+2A02 N-ARY CIRCLED TIMES OPERATOR
        "⊗",  # U+2297 CIRCLED TIMES (canonical)
    ),
]

_UNICODE_SYMBOL_TABLE = str.maketrans(
    {char: canonical for chars, canonical in _UNICODE_SYMBOL_CLASSES for char in chars}
)

_TRUE_TABLE_MARKER_GLYPHS = frozenset(
    {
        "☑",
        "▣",
        "✓",
        "✔",
        "◉",
        "●",
        "•",
        "⦿",
    }
)
_FALSE_TABLE_MARKER_GLYPHS = frozenset({"☐", "□", "○", "◯", "✗", "✘", "✕", "✖", "×"})
_TRUE_TABLE_MARKER_TEXT = frozenset({"yes", "x"})
_FALSE_TABLE_MARKER_TEXT = frozenset({"no"})


def _normalize_unicode_symbols(text: str) -> str:
    """Collapse Unicode symbol variants to their canonical forms."""
    return text.translate(_UNICODE_SYMBOL_TABLE)


def _normalize_table_boolean_marker(text: str) -> str:
    """Normalize whole-cell boolean markers to yes/no.

    This is intentionally exact-cell scoped. It treats values like ``✓``,
    ``✗``, ``●``, ``X``, ``[yes]``, and ``[no]`` as boolean table-cell markers,
    but leaves mixed content such as ``● item`` untouched.
    """
    stripped = re.sub(r"\s+", " ", text).strip()
    if not stripped:
        return text

    token = stripped.lower()
    bracket_match = re.fullmatch(r"\[\s*(.*?)\s*\]", stripped)
    if bracket_match:
        stripped = bracket_match.group(1).strip()
        token = stripped.lower()

    if (
        stripped in _TRUE_TABLE_MARKER_GLYPHS
        or token in _TRUE_TABLE_MARKER_TEXT
        or (bracket_match and token in {"x", "yes"})
    ):
        return "yes"
    if (
        stripped in _FALSE_TABLE_MARKER_GLYPHS
        or token in _FALSE_TABLE_MARKER_TEXT
        or (bracket_match and token in {"", "no"})
    ):
        return "no"
    return text


def _normalize_quotes(text: str) -> str:
    """Map common Unicode quote/punctuation variants to ASCII equivalents."""
    return text.translate(_QUOTE_TRANSLATION_TABLE)


# ---------------------------------------------------------------------------
# Formatting / markup patterns stripped by normalize_cell_text()
# ---------------------------------------------------------------------------

# HTML formatting tags to strip (same set as header_accuracy_metric._FORMATTING_RE)
_HTML_FORMATTING_RE = re.compile(
    r"</?(?:b|i|u|s|em|strong|del|strike|mark|ins)>",
    re.IGNORECASE,
)

# <span> tags (with optional attributes like style, color) — strip tag, keep content
_HTML_SPAN_RE = re.compile(r"</?span\b[^>]*>", re.IGNORECASE)

# Inline styling markers that ``normalize_text`` deletes while keeping their
# content: the content-preserving HTML tags plus markdown ``~~``. ``<sup>`` and
# ``<sub>`` are deliberately excluded — they delete their content too, which is
# a different decision. The ``\b`` after each name keeps ``<b>`` from matching
# ``<br>`` and ``<s>`` from matching ``<span>``/``<strike>``/``<sub>``.
_INLINE_STYLE_MARKER = r"</?(?:b|i|u|s|del|strike|ins|mark|span)\b[^>]*>|~~"
# A *run* of two or more abutting markers. One marker between two content
# characters is an intra-token style change (Japanese ``<u>`` mid-sentence,
# ``Shizuok<mark>a``) and must join; two abutting markers are the boundary
# between two separately-styled tokens (``<u>103</u><s>101</s>``) and must
# separate. The shipped GT corpora contain zero runs with content on both sides,
# so this only ever changes prediction-side welds.
_INLINE_STYLE_MARKER_RUN_RE = re.compile(f"(?:{_INLINE_STYLE_MARKER}){{2,}}", re.IGNORECASE)


def _inline_style_run_to_separator(match: re.Match[str]) -> str:
    """Replace a run of abutting inline markers with a single space.

    A run at either end of the string separates nothing, so it collapses to the
    empty string rather than padding the result — callers compare normalized
    strings for equality.
    """
    if match.start() == 0 or match.end() == len(match.string):
        return ""
    return " "


# Markdown bold: **text** or __text__
_MD_BOLD_RE = re.compile(r"\*\*(.*?)\*\*|__(.*?)__")
# Markdown italic: *text* or _text_  (must not match ** or __)
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.*?)(?<!_)_(?!_)")
# Markdown strikethrough: ~~text~~
_MD_STRIKETHROUGH_RE = re.compile(r"~~(.*?)~~")

# ---------------------------------------------------------------------------
# Sup/sub conversion (shared between GriTS normalize_cell_text and TRM
# normalize_table). Converts <sup>x</sup> / <sub>x</sub> tags to plain text
# and translates Unicode super/subscript characters to ASCII equivalents.
# ---------------------------------------------------------------------------

# Unicode superscript → ASCII mappings (digits + common letters/symbols)
_SUPERSCRIPT_TO_ASCII = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ",
    "0123456789+-=()ni",
)

# Unicode subscript → ASCII mappings
_SUBSCRIPT_TO_ASCII = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ",
    "0123456789+-=()aehijklmnoprstuvx",
)


def _normalize_sub_sup_for_table(text: str) -> str:
    """Convert sub/sup tags and Unicode chars to plain text for table comparison.

    Unlike ``normalize_text()`` which strips sup/sub entirely (correct for
    footnote markers in prose), tables use sup/sub for meaningful content
    like chemical formulas (H₂O) and exponents (x²).
    """
    text = re.sub(r"<sup[^>]*>(.*?)</sup>", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"<sub[^>]*>(.*?)</sub>", r"\1", text, flags=re.IGNORECASE)
    text = text.translate(_SUPERSCRIPT_TO_ASCII)
    text = text.translate(_SUBSCRIPT_TO_ASCII)
    return text


# Dash-like characters to normalize to ASCII hyphen
_DASH_CHARS = str.maketrans(
    {
        "–": "-",  # en-dash U+2013
        "—": "-",  # em-dash U+2014
        "‑": "-",  # non-breaking hyphen U+2011
        "‒": "-",  # figure dash U+2012
        "−": "-",  # minus sign U+2212
    }
)


def _normalize_encoding_punctuation(text: str) -> str:
    """Apply the shared quote and dash folds used by text and label matching."""
    return _normalize_quotes(text).translate(_DASH_CHARS)


_LABEL_WHITESPACE_RE = re.compile(r"\s+")


def normalize_label_punctuation(text: object | None) -> str:
    """Fold encoding-only punctuation while preserving judge-facing label text.

    This composes the same quote and dash equivalence tables used by the
    deterministic text matcher, but deliberately keeps case, wording, and
    markup intact for evidence shown to the judge.
    """
    if text is None:
        return ""
    normalized = _normalize_encoding_punctuation(str(text))
    return _LABEL_WHITESPACE_RE.sub(" ", normalized).strip()


# A cell consisting entirely of dash-like characters and whitespace
_DASH_ONLY_RE = re.compile(r"^[\s\-–—‑‒−]+$")

# A dot-leader run: two or more periods separated by nothing but blanks, e.g.
# ``.....`` or ``. . . . .``. Typographic leaders whose dot count is a function
# of column width, not of content, so the same label reaches the evaluator as
# ``Total assets`` from one page and ``Total assets . . . . . . .`` from
# another.
#
# Two alternatives, because a leader may or may not be introduced by a space:
#
# 1. A run *starting* at whitespace or the start of the string, whose periods
#    may then be separated by blanks — ``Total assets . . . . .``.
# 2. A *contiguous* run of two or more periods that ends at whitespace or the
#    end of the string — ``Revenue.........``.
#
# Between them these spare every period that is content. In a decimal
# (``1.5``), a European thousands separator (``1.234.567``), a version string
# (``3.14.1``) and a dotted abbreviation (``U.S.``) the periods are single and
# separated by non-blank characters. A run wedged between two word characters
# (``A..B``) is punctuation, not typography, and matches neither alternative.
#
# A Unicode ellipsis (U+2026) is one glyph standing for three periods, so it
# counts as a dot *and* as a complete run on its own: a transcriber choosing
# ``…`` over ``...`` is making a font decision, not a content one.
_DOT_CHAR = r"[.…]"
# A "run" is either two-or-more dot characters (possibly blank-separated) or a
# single ellipsis, which is already three dots' worth of glyph.
_DOT_RUN = rf"(?:{_DOT_CHAR}(?:[^\S\n]*{_DOT_CHAR})+|…)"
_DOT_LEADER_RE = re.compile(rf"(?:(?<=\s)|^){_DOT_RUN}|(?<=\S)(?:\.{{2,}}|…+)(?=\s|$)")

# A leader that begins *attached* to its label, with no space between the label
# and the first dot, and is then spaced out: ``Employed. . . . . . . .``. The
# contiguous spelling of the very same glyph run (``Employed............``)
# already loses every period to alternative 2 above, so without this the two
# spellings normalize differently and a prediction is scored down purely for
# how it spaced a filler. Requiring the label to be a period-free token
# (``[^\s.…]+``) is what keeps a dotted abbreviation introducing a leader
# intact — in ``E.O. .....................`` the token cannot span the ``.``,
# so the pattern never engages and ``E.O.`` keeps its own period.
_ATTACHED_DOT_LEADER_RE = re.compile(rf"(?<!\S)(?P<label>[^\s.…]+)\.(?:[^\S\n]*{_DOT_CHAR})+(?=\s|$)")

# The same attached-leader shape, but for a label that already contains periods
# of its own and so cannot match the period-free token above: ``3.14. . . .``,
# ``1,234.56. . . .``, ``192.168.1.1. . . .``. Without this the leader's first
# dot stays glued to the number and the cell normalizes to ``"3.14."`` — a
# period that is in neither the content nor the leader, and one that the merely
# spaced spelling (``3.14 . . . .``) does not produce.
#
# The label is required to end in a *digit*, which is what makes the assignment
# unambiguous here while it is genuinely ambiguous for letters. A numeric token
# never legitimately ends in a period, so a period between a digit and a run
# can only belong to the run; ``E.O.`` and ``3.14.`` are the same *shape*, and
# only the digit tells them apart. Alphabetic labels keep the conservative
# treatment above, which is why ``"ft."`` and ``"in."`` — unit residue that
# ``table_normalization_relaxed`` recognises *by* the period — are untouched.
_ATTACHED_NUMERIC_DOT_LEADER_RE = re.compile(rf"(?<!\S)(?P<label>\S*\d)\.(?:[^\S\n]*{_DOT_CHAR})+(?=\s|$)")


def strip_dot_leaders(text: str) -> str:
    """Remove dot-leader runs from a table cell's text.

    Shared by ``normalize_cell_text`` (GriTS / TEDS / header accuracy) and
    ``_normalize_trm_cell_text`` (table record match) so the two table
    metrics agree on what a leader run is. Applied symmetrically to ground
    truth and prediction.

    A run is replaced by the empty string, and the whitespace either side is
    collapsed by the caller, so ``"Total assets . . . 5"`` and
    ``"Total assets 5"`` normalize alike. A run wedged between two word
    characters (``"A..B"``) is punctuation, not typography, and survives.

    The result is independent of how the leader was *spaced*: ``"Employed...."``,
    ``"Employed. . . ."``, ``"Employed ...."`` and ``"Employed…"`` all reduce to
    ``"Employed"``. That invariant is the point — dot spacing is a property of
    the transcription, never of the document, so it must not move a metric.

    Idempotent: ``strip_dot_leaders(strip_dot_leaders(x)) == strip_dot_leaders(x)``.

    Known boundary — a label whose own last character is a period
    ------------------------------------------------------------
    ``"Acme Inc. ...."`` reduces to ``"Acme Inc"``, dropping the
    abbreviation's own period, while an undecorated ``"Acme Inc."`` keeps it.
    The two are therefore *not* interchangeable **as text**.

    That residual no longer costs anything on the exact-match submetrics: they
    compare through :func:`cells_match_leader_insensitive`, which asks whether
    two cells differ only in their trailing dot/period tail and never has to
    decide who owns the period. The boundary below is a statement about this
    function's *output text*, not about what the metrics score.

    This is irreducible rather than an oversight. ``"Acme Inc....."`` is one
    undifferentiated glyph run in which nothing marks how many of those dots
    belonged to ``Inc.``, and ``"Employed. . . ."`` has exactly the same shape,
    so no rule can separate the two cases. The tie is broken toward absorbing
    the period because the alternative — keeping it — makes the *spaced* and
    *contiguous* spellings of one leader normalize differently, which is a
    strictly larger and already-measured harm.

    The boundary covers exactly the cells where a period sits between the label
    and the run and the two readings are indistinguishable by shape:

    * a label whose final token ends in a period and has no other period
      (``"Acme Inc."``, ``"Jr."``, ``"No."``, ``"ft."``) — absorbed for every
      leader spelling;
    * a *multi-period* abbreviation with a leader written contiguously
      (``"U.S....."`` -> ``"U.S"``). Given a blank to separate them
      (``"U.S. ....."``, ``"E.O. ....."``) the abbreviation keeps its period,
      because the blank is the evidence that the period is the label's.

    Numeric labels are *not* in the boundary, even though they have the same
    shape, because a number never legitimately ends in a period — see
    ``_ATTACHED_NUMERIC_DOT_LEADER_RE``. ``tests/.../test_table_dot_leaders.py``
    pins the boundary, its per-cell cost, and the numeric exemption.
    """
    # Absorb the label-attached period first, so the spaced spelling of a
    # leader reduces to the same text as its contiguous spelling.
    text = _ATTACHED_NUMERIC_DOT_LEADER_RE.sub(r"\g<label>", text)
    text = _ATTACHED_DOT_LEADER_RE.sub(r"\g<label>", text)
    return _DOT_LEADER_RE.sub("", text)


# Trailing decoration a dot leader can leave behind on either side of a
# comparison: dot characters, blanks, and — crucially — a *single* final
# period, which is below the two-dot threshold a run needs and so survives
# ``strip_dot_leaders`` untouched. Anchored at the end of the cell, so no
# interior period is ever in reach.
_TRAILING_LEADER_RESIDUE_RE = re.compile(r"[.…\s]+$")


def leader_insensitive_core(text: str) -> str:
    """Return *text* with any trailing dot / ellipsis / blank run removed.

    Unlike :func:`strip_dot_leaders` this also removes a *lone* final period,
    so ``"Acme Inc."``, ``"Acme Inc"`` and ``"Acme Inc. ...."`` all reduce to
    ``"Acme Inc"``. That is deliberately too aggressive to use as a
    normalization — see :func:`cells_match_leader_insensitive` for why it is
    only ever used to compare two cells, never to rewrite one.

    Only the tail is touched: ``"192.168.1.1"``, ``"3.1.4"`` and ``"1.5"``
    come back unchanged, because their periods are interior.
    """
    return _TRAILING_LEADER_RESIDUE_RE.sub("", text)


def cells_match_leader_insensitive(
    a: str,
    b: str,
    *,
    normalize: Callable[[str], str] | None = None,
) -> bool:
    """Compare two already-normalized table cells, ignoring leader decoration.

    Why this is a *comparison* and not another normalization
    -------------------------------------------------------
    When a single period sits between a label and a dot-leader run
    (``"Acme Inc. ...."``), nothing in the glyph run says whether that period
    closes the abbreviation or opens the leader — the two readings have the
    same shape. :func:`strip_dot_leaders` has to pick one, and it absorbs the
    period, so a decorated ``"Acme Inc. ...."`` reduces to ``"Acme Inc"``
    while an undecorated ``"Acme Inc."`` keeps its period. On the exact-match
    submetrics that residual costs the whole cell.

    Mutation-level stripping cannot close that gap. The only exact closure is
    to drop *every* cell-final period, and ``normalize_cell_text`` is shared
    with ``table_normalization_relaxed``, whose ``_EMPTY_TEMPLATES`` depend on
    the opposite convention: ``"ft."`` / ``"in."`` are recognised as unit
    residue *by* their period, and that module's comment records that the
    trailing-period strip "is intentionally NOT applied". This helper does not
    contradict that comment — it changes no cell's text and feeds nothing
    downstream. It only decides, at the moment two cells are compared for
    equality, whether their difference is confined to leader decoration.

    Because the question is asked symmetrically about a *pair*, the period's
    owner never has to be decided: both sides give up their tail, so the two
    readings can no longer disagree.

    Dots-only cells
    ---------------
    A cell whose core is empty — ``"."``, or a cell that already normalized to
    ``""`` — is **not** matched through this fallback against arbitrary text;
    it falls back to plain equality. This keeps faith with the pinned decision
    that a dots-only cell (``"..."``, ``"…"``) normalizes to the empty string:
    two such cells still compare equal because they are *both* empty after
    normalization, not because a stray dot was allowed to match a label.

    Args:
        a: One normalized cell's text.
        b: The other normalized cell's text.
        normalize: Optional re-normalizer applied to both cores. Needed where
            trimming the tail can expose a token an *earlier* fold would have
            rewritten — TRM's boolean-marker fold runs before leader stripping,
            so ``"No."`` survives as ``"No."`` while ``"No. ..."`` reaches
            ``"no"``; re-normalizing both cores lands them on the same text.
    """
    if a == b:
        return True
    core_a = leader_insensitive_core(a)
    core_b = leader_insensitive_core(b)
    if normalize is not None:
        core_a = normalize(core_a)
        core_b = normalize(core_b)
    if not core_a or not core_b:
        return False
    return core_a == core_b


def normalize_cell_text(text: str) -> str:
    """Normalize a table cell's text content for metric comparison.

    Applies transformations suitable for cell-level comparison
    in TEDS, GriTS, and header-accuracy metrics:
    - Sup/sub tag conversion (<sup>x</sup> → x, ¹ → 1, etc.)
    - HTML formatting tag removal (<b>, <i>, <mark>, <em>, <strong>, etc.)
    - Markdown bold/italic/strikethrough removal
    - Whole-cell boolean marker equivalence (✓/✗/●/[yes]/[no])
    - Unicode symbol equivalence (bullets, circled-x, etc.)
    - Quote / fullwidth punctuation canonicalization
    - Dash character normalization (en-dash, em-dash, etc. → ASCII hyphen)
    - Dash-only cells collapsed to a single "-"
    - Dot-leader stripping (runs of 2+ dots, contiguous or space-separated)
    - Whitespace collapsing and stripping

    Formatting is intentionally stripped (not preserved as a signal): in
    practice, models routinely emphasize totals/headers with bold while
    ground-truth tables don't, and treating that mismatch as a content
    error penalizes parsing quality for an unrelated convention. This
    intentionally does NOT lowercase or strip accents.
    """
    # Sup/sub: tag conversion + Unicode → ASCII (shared with TRM normalization)
    text = _normalize_sub_sup_for_table(text)
    # Strip HTML formatting tags
    text = _HTML_FORMATTING_RE.sub("", text)
    # Strip <span> tags (keep content) — e.g. <span color="red">text</span> → text
    text = _HTML_SPAN_RE.sub("", text)
    # Strip markdown bold, then italic, then strikethrough
    text = _MD_BOLD_RE.sub(r"\1\2", text)
    text = _MD_ITALIC_RE.sub(r"\1\2", text)
    text = _MD_STRIKETHROUGH_RE.sub(r"\1", text)
    # Whole-cell boolean markers before generic symbol folding, so open/closed
    # circles remain distinguishable.
    text = _normalize_table_boolean_marker(text)
    # Unicode symbol equivalence
    text = _normalize_unicode_symbols(text)
    # Quote / dash / fullwidth punctuation canonicalization shared with the
    # plain-text and judge-evidence normalizers.
    text = _normalize_encoding_punctuation(text)
    # Strip dot-leaders — contiguous ("Total assets.....") or spaced
    # ("Total assets . . . . ."). Shared with the TRM cell normalizer.
    text = strip_dot_leaders(text)
    # Collapse whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()
    # If cell is entirely dashes (after normalization), collapse to single dash
    if _DASH_ONLY_RE.match(text):
        return "-"
    return text


# ---------------------------------------------------------------------------
# Relaxed-metric cell-text normalization (LI-8223)
#
# These transformations are applied ONLY on the relaxed metric path (via
# ``table_normalization_relaxed.normalize_for_relaxed``) — never by the strict
# GriTS / TRM / header-accuracy metrics, which keep using ``normalize_cell_text``
# unchanged. ``relaxed_normalize_cell_text`` runs BEFORE the strict
# ``normalize_cell_text`` (they compose), so it only needs to canonicalize the
# few extra equivalences the strict pass leaves alone.
# ---------------------------------------------------------------------------

# Block / line-level tags whose removal must not concatenate adjacent tokens.
# A list-item OPENING tag becomes a canonical bullet so that a ``<ul><li>`` list
# and a literal ■-bulleted cell converge to the same "• A • B" shape; every
# other structural tag becomes a plain space. Motivating page:
# ``Low-Back-Guideline (1)_page2`` — GT emits literal ■ bullets while the
# prediction emits ``<ul><li>`` items that ``get_text()`` fuses without spaces.
_RELAXED_LIST_ITEM_OPEN_RE = re.compile(r"<li\b[^>]*>", re.IGNORECASE)
_RELAXED_BLOCK_TAG_RE = re.compile(
    r"</li>|</?ul\b[^>]*>|</?ol\b[^>]*>|</?p\b[^>]*>|<br\s*/?>",
    re.IGNORECASE,
)

# Canonical bullet used by both the list-item injection and the geometric-bullet
# homoglyph fold below (U+2022 BULLET).
_RELAXED_CANONICAL_BULLET = "•"

# Visually-equivalent codepoint folds applied only on the relaxed path.
# Kept small, explicit, and commented with the motivating equivalence.
_RELAXED_HOMOGLYPH_TABLE = str.maketrans(
    {
        "µ": "μ",  # µ MICRO SIGN -> μ GREEK SMALL LETTER MU
        " ": " ",  # NBSP -> space
        # Geometric bullet variants -> canonical bullet (U+2022).
        # ONLY the square bullets fold here: strict _normalize_unicode_symbols already
        # folds circle bullets for mixed-content cells, and folding whole-cell ●/○/◦
        # would collide with _normalize_table_boolean_marker (● = yes, ○ = no) —
        # a checked-vs-unchecked disagreement is a REAL content error and must not
        # be normalized away on the relaxed path.
        "■": _RELAXED_CANONICAL_BULLET,  # ■ BLACK SQUARE
        "▪": _RELAXED_CANONICAL_BULLET,  # ▪ BLACK SMALL SQUARE
    }
)


def relaxed_normalize_cell_text(text: str) -> str:
    """Extra relaxed-only cell-text canonicalization (LI-8223 items 2 & 3).

    Applied only on the relaxed metric path, before the strict
    ``normalize_cell_text`` runs on top. Two folds:

    * **Tag-strip separator injection** — replace block/line tags with a
      separator so list items don't concatenate into one token. A ``<li>``
      opener injects the canonical bullet (``•``); ``</li>``, ``<ul>``,
      ``<ol>``, ``<p>``, ``<br>`` inject a space. Repeated whitespace is
      collapsed afterwards.
    * **Encoding punctuation folding** — use the shared quote/dash mapping so
      this path cannot drift from strict table, plain-text, or judge matching.
    * **Relaxed-only homoglyph folding** — map µ→μ, NBSP→space, and geometric
      bullet variants → ``•``.

    Idempotent and symmetric (applied identically to GT and prediction).
    """
    if not text:
        return text
    text = _RELAXED_LIST_ITEM_OPEN_RE.sub(f" {_RELAXED_CANONICAL_BULLET} ", text)
    text = _RELAXED_BLOCK_TAG_RE.sub(" ", text)
    # Keep quote/dash equivalence on the same canonical mapping as strict
    # table, plain-text, and judge-evidence normalization.
    text = _normalize_encoding_punctuation(text)
    text = text.translate(_RELAXED_HOMOGLYPH_TABLE)
    # Collapse the whitespace introduced by tag/NBSP substitution.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(md_content: str | None) -> str:
    """
    Normalize markdown text for comparison.

    This function:
    - Normalizes whitespace
    - Removes markdown formatting (bold, italics)
    - Normalizes unicode characters
    - Replaces fancy quotes and dashes with ASCII equivalents

    :param md_content: Markdown content to normalize
    :return: Normalized text
    """
    if md_content is None:
        return ""

    # Strip autolink angle brackets: <http://foo.bar> → http://foo.bar
    # Also handles mailto: and bare email autolinks (<user@host.tld>)
    md_content = re.sub(
        r"<((?:https?://|mailto:)[^>\s]+|[^>@\s]+@[^>@\s]+\.[^>@\s]+)>",
        r"\1",
        md_content,
        flags=re.IGNORECASE,
    )

    # Normalize <br>, <br/>, and <br /> to spaces
    md_content = re.sub(r"<br\s*/?>", " ", md_content)

    # Inline styling markers are deleted below (empty string), which is right for
    # a single marker inside a token but welds two tokens together where two
    # markers abut: "<u>103</u><s>101</s>" would become "103101" and every
    # word/sentence/order rule would then report both page numbers as missing.
    # Turn a run of abutting markers into a separator first; the whitespace
    # collapse right below folds it into surrounding space.
    md_content = _INLINE_STYLE_MARKER_RUN_RE.sub(_inline_style_run_to_separator, md_content)

    # Canonicalize Unicode quote and dash variants early using the same mapping
    # as table-cell and judge-evidence normalization.
    md_content = _normalize_encoding_punctuation(md_content)

    # Normalize whitespace in the md_content
    md_content = re.sub(r"\s+", " ", md_content)

    # Remove markdown bold formatting (** or __ for bold)
    md_content = re.sub(r"\*\*(.*?)\*\*", r"\1", md_content)
    md_content = re.sub(r"__(.*?)__", r"\1", md_content)
    md_content = re.sub(r"</?b>", "", md_content)  # Remove <b> tags if they exist
    md_content = re.sub(r"</?i>", "", md_content)  # Remove <i> tags if they exist

    # Remove markdown italics formatting (* or _ for italics)
    md_content = re.sub(r"\*(.*?)\*", r"\1", md_content)
    md_content = re.sub(r"_(.*?)_", r"\1", md_content)

    # Replace remaining underscores with spaces so filenames like
    # "099_20090718白山祭り088" split into separate tokens.  Paired italic
    # markers (_..._) are already stripped above; any leftover _ is a literal
    # underscore (e.g. in image filenames embedded in OCR output).
    # Stays aligned with JS annotation tool which does text.replace(/[*_~]+/g, " ").
    md_content = md_content.replace("_", " ")

    # Convert accented letters to ASCII equivalents (e.g., é -> e)
    # NFD decomposing separates base characters from combining marks
    md_content = unicodedata.normalize("NFD", md_content)
    # Remove combining characters (accents, diacritics) but KEEP combining marks
    # whose base character needs them: the Japanese voicing marks (they
    # distinguish ka from ga, ha from pa) and the vowel signs, tone marks and
    # viramas of the abugida scripts, where deleting a mark changes the word.
    # The base is tracked across stacks: Thai writes a tone mark on top of a
    # vowel mark, so the character immediately preceding a mark is not the one
    # that decides whether that mark is load-bearing.
    result_chars: list[str] = []
    base_preserves_marks = False
    for char in md_content:
        if unicodedata.category(char) != "Mn":
            result_chars.append(char)
            base_preserves_marks = _is_mark_preserving_base_char(char)
        elif base_preserves_marks:
            result_chars.append(char)
        # else: strip (Latin/Cyrillic accents -> ASCII)
    md_content = "".join(result_chars)
    # Convert back to NFC form for consistency
    md_content = unicodedata.normalize("NFC", md_content)

    # Dictionary of characters to replace: keys are fancy characters, values are ASCII equivalents
    replacements = {
        "＿": "_",
        "…": "...",
        "<ins>": "",
        "</ins>": "",
        "<u>": "",
        "</u>": "",
        "~~": "",
        "<mark>": "",
        "</mark>": "",
        "<br/>": " ",
        "<br />": " ",
        "\n": " ",
        "$$": "",  # Remove $$ signs as Latex delimiters are that way
        "\u00b5": "\u03bc",  # micro sign to greek mu
    }

    # Apply all replacements from the dictionary
    for fancy_char, ascii_char in replacements.items():
        md_content = md_content.replace(fancy_char, ascii_char)

    # Normalize Unicode symbol variants (bullets, circled-x, etc.)
    md_content = _normalize_unicode_symbols(md_content)

    # Strip <s>, <del>, <strike> tags (keep content) — equivalent to ~~ stripping above
    md_content = re.sub(r"</?(?:s|del|strike)>", "", md_content, flags=re.IGNORECASE)

    # Strip <span> tags with any attributes (keep content)
    # e.g. <span color="red">text</span> → text
    md_content = _HTML_SPAN_RE.sub("", md_content)

    # Remove <sup>...</sup> and <sub>...</sub> tags AND their content
    # (e.g., footnote markers like "84.1<sup>(2)</sup>" → "84.1")
    md_content = re.sub(r"<sup[^>]*>.*?</sup>", "", md_content, flags=re.IGNORECASE)
    md_content = re.sub(r"<sub[^>]*>.*?</sub>", "", md_content, flags=re.IGNORECASE)

    # Strip Unicode superscript digits (footnote markers like "84.1¹" → "84.1").
    # These are standalone codepoints that NFD decomposition does not decompose.
    # We strip rather than convert to regular digits to avoid changing values
    # (e.g., "84.1¹" → "84.11" would be wrong). Consistent with <sup> removal above.
    md_content = re.sub(r"[\u00b9\u00b2\u00b3\u2070\u2074-\u2079]+", "", md_content)

    # Strip Unicode subscript digits (e.g. "H₂O" → "HO"), consistent with
    # <sub> removal above and superscript digit stripping.
    md_content = re.sub(r"[\u2080-\u2089]+", "", md_content)

    # Normalize multiple consecutive dashes to single dash
    # This handles cases like "--" or "---" becoming "-"
    md_content = re.sub(r"-{2,}", "-", md_content)

    # Strip trailing dot-leaders (e.g., "Operating income.........." → "Operating income").
    # These are formatting dots used in tables to connect labels to values.
    # Only strip 2+ consecutive dots at the end to preserve grammatical periods ("Inc.").
    #
    # Implemented as a bounded right-to-left scan rather than the regex
    # r"\.{2,}\s*$": that regex backtracks quadratically on pathological content
    # holding one very long run of dots (e.g. a degenerate OCR page that emits
    # 130k dot characters). On such input the greedy `\.{2,}` grabs the whole run
    # at every start position and then gives it back a dot at a time when `\s*$`
    # fails — ~66s for a single 130k-char string, and the metric re-normalizes
    # per rule, so a single document could burn the entire 6h eval budget. The
    # scan below is linear in the length of the trailing run and behaviourally
    # identical.
    _end = len(md_content)
    while _end > 0 and md_content[_end - 1].isspace():
        _end -= 1
    _dot_start = _end
    while _dot_start > 0 and md_content[_dot_start - 1] == ".":
        _dot_start -= 1
    if _end - _dot_start >= 2:
        md_content = md_content[:_dot_start]

    # lowerCase the content for case-insensitive comparison
    md_content = md_content.lower()
    return md_content


def normalize_text_light(md_content: str | None) -> str:
    """
    Light normalization that preserves text formatting/styling.

    Unlike normalize_text(), this function:
    - KEEPS markdown formatting (bold **, italics *)
    - KEEPS HTML styling tags (<i>, <b>, <u>, <sup>, <sub>, <mark>, <ins>)
    - KEEPS dots/periods
    - KEEPS original case
    - Still normalizes whitespace and unicode quotes/dashes for reliable matching

    Use this when testing that formatting is correctly preserved in the output.

    :param md_content: Markdown content to normalize
    :return: Lightly normalized text with formatting preserved
    """
    if md_content is None:
        return ""

    # Strip autolink angle brackets: <http://foo.bar> → http://foo.bar
    md_content = re.sub(
        r"<((?:https?://|mailto:)[^>\s]+|[^>@\s]+@[^>@\s]+\.[^>@\s]+)>",
        r"\1",
        md_content,
        flags=re.IGNORECASE,
    )

    # Normalize <br> and <br/> to spaces (these are layout, not styling)
    md_content = re.sub(r"<br/?>", " ", md_content)

    # Strip <span> tags (keep content) — these are layout wrappers, not styling
    md_content = _HTML_SPAN_RE.sub("", md_content)

    # Canonicalize Unicode quote and dash variants using the shared mapping
    # while preserving styling.
    md_content = _normalize_encoding_punctuation(md_content)

    # Normalize whitespace (collapse multiple spaces/newlines to single space)
    md_content = re.sub(r"\s+", " ", md_content)

    # Convert accented letters to ASCII equivalents (e.g., é -> e)
    # This helps with matching even when accents differ
    md_content = unicodedata.normalize("NFD", md_content)
    md_content = "".join(
        char
        for char in md_content
        if unicodedata.category(char) != "Mn"  # Mn = Nonspacing_Mark (accents)
    )
    md_content = unicodedata.normalize("NFC", md_content)

    # Only normalize dashes/symbols to ASCII equivalents
    # Keep dots, keep case, keep formatting tags
    replacements = {
        "＿": "_",
        "…": "...",
        "\u00b5": "\u03bc",  # micro sign to greek mu
    }

    for fancy_char, ascii_char in replacements.items():
        md_content = md_content.replace(fancy_char, ascii_char)

    # Normalize Unicode symbol variants (bullets, circled-x, etc.)
    md_content = _normalize_unicode_symbols(md_content)

    # Normalize multiple consecutive dashes to single dash
    md_content = re.sub(r"-{2,}", "-", md_content)

    return md_content.strip()
