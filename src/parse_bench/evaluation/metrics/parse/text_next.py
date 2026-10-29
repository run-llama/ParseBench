"""Text fidelity metric (``text_next``): move-tolerant text fidelity with an error budget.

The page is read as one token stream. Numbers, identifiers, emails, URLs, domains,
units, currency signs and negators are *critical* tokens; other words are
NFKC + lower-cased; Han, Kana and Thai are one token per character.

Alignment: greedy string tiling (runs of at least ``K`` tokens, longest first)
finds the blocks wherever they moved; an in-order LCS inside each gap between
tiles matches the rest in context; short reference lines are matched as whole
output lines; leftover identical words are matched as *displaced*.

Lanes (each in [0, 1], ``None`` when undefined):

* ``words``: F1 of matched tokens (missing or extra text).
* ``critical``: F1 of the ordered critical tokens after undoing block moves.
* ``integrity``: share of reference sentence words whose sentence is not cut by
  foreign output content.
* ``order``: 1 - matched weight outside the heaviest in-order chain of blocks.

Headline: ``min(proportional, floor)``. The proportional score subtracts weighted
lane losses; the floor subtracts a fixed budget per error *occurrence* (critical,
major, minor), so one wrong number on a long page still costs points. Reading-order
occurrences (moved blocks, cut sentences) cost at most ``FLOOR["order_cap"]`` together.

Regexes follow ECMAScript semantics: ``\\s`` is the
ECMAScript white-space set, ``\\d``/``\\w``/``\\b`` are ASCII, ``$`` without the
multiline flag is the end of the input, and ``\\p{..}`` classes come from the
``regex`` package.
"""

from __future__ import annotations

import html
import math
import unicodedata
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import regex
from rapidfuzz.distance import LCSseq, Levenshtein

from parse_bench.schemas.evaluation import MetricValue
from parse_bench.schemas.parse_output import ParseOutput

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K = 3  # minimum tile length (tokens)
NEXT_W = {"words": 1.0, "critical": 1.5, "integrity": 0.25, "order": 0.1}
FLOOR = {"critical": 0.10, "major": 0.05, "minor": 0.01, "minor_cap": 0.15, "order_cap": 0.30}
NOVEL_CAP = 2  # invented critical tokens counted per inserted run
CRIT_RAMP = 10  # the critical lane reaches full weight at this many critical tokens
FLOOR_L0 = 400  # occurrences after the first scale by min(1, FLOOR_L0 / reference tokens)
LCS_CELL_LIMIT = 4_000_000  # gap LCS skipped above this many DP cells
OMIT_PASSAGE = 12  # an omitted run of this many reference tokens is a lost passage (critical), not a phrase
NEG_OMIT_MAX = 3  # a negator inside a longer omitted run is part of the omission, not a flipped meaning
CHAIN_LIMIT = 3000  # moved-block runs use an O(n^2) chain below this many blocks
SWAP_SLOT = 3  # a displaced number whose slot (up to this many output tokens) holds another one: swapped values
LINE_NO_RUN = 10  # consecutive lines numbered n, n+1, ... that make a margin line numbering (pleading paper)

# ---------------------------------------------------------------------------
# JavaScript regex compatibility
# ---------------------------------------------------------------------------

_JS_WS = "\t\n\x0b\x0c\r \xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_S = f"[{_JS_WS}]"
_NS = f"[^{_JS_WS}]"
_EOL_M = r"(?=[\n\r\u2028\u2029]|\Z)"  # `$` with the m flag
_BOL_M = r"(?:(?<=[\n\r\u2028\u2029])|\A)"  # `^` with the m flag
_W = "[A-Za-z0-9_]"  # JS `\w`
_NW = "(?![A-Za-z0-9_])"  # JS `\b` after a word character


def _rx(pattern: str, flags: int = 0) -> Any:
    return regex.compile(pattern, flags | regex.V0)


_I = regex.IGNORECASE


_JS_TRIM = " \t\n\x0b\x0c\r\xa0\u1680\u2028\u2029\u202f\u205f\u3000\ufeff" + "".join(map(chr, range(0x2000, 0x200B)))


def _js_trim(s: str) -> str:
    """``String.prototype.trim``: ECMAScript white space and line terminators."""
    return s.strip(_JS_TRIM)


def _utf16_len(s: str) -> int:
    return len(s) + sum(1 for c in s if ord(c) > 0xFFFF)


# ---------------------------------------------------------------------------
# Markdown to plain text
# ---------------------------------------------------------------------------

# A link destination: ``(<url>)`` (the ``<`` may already be the ``‹`` placeholder) or a URL with
# one level of balanced parentheses, e.g. ``(http://host/a-1(17)2-3)``.
_DEST = r"\((?:[<\u2039][^>\n]*>|(?:[^()\n]|\([^()\n]*\))*)\)"

_MD_STEPS: list[tuple[Any, str]] = [
    (_rx(rf"<br{_S}*/?>", _I), " "),
    (_rx(r"</(td|th)>", _I), " "),
    (_rx(r"</(tr|p|div|li|h[0-9]|table|thead|tbody)>", _I), "\n"),
    (_rx(r"<[^>]+>"), " "),
    (_rx(rf"!\[[^\]]*\]{_DEST}"), " "),
    (_rx(rf"\[([^\]]*)\]{_DEST}"), r"\1"),
    (_rx(rf"{_BOL_M}[ \t]*\|?[ \t]*:?-{{3,}}[-|: \t]*{_EOL_M}"), ""),
    (_rx(rf"{_BOL_M}[ \t]*(?:```|~~~)[^\n]*{_EOL_M}"), ""),  # code fence with its language label
    (_rx(r"\|"), " "),
    (_rx(rf"{_BOL_M}[ \t]{{0,3}}(#{{1,6}}[ \t]+|>[ \t]?|[-*+][ \t]+)"), ""),
    (_rx(r"\*\*|__"), ""),
    (_rx(rf"(\A|{_S})[*_](?={_NS})"), r"\1"),
    (_rx(rf"(?<={_NS})[*_](?={_S}|\Z|[.,;:!?])"), ""),
    (_rx("[ \t\xa0]+"), " "),
]


def md_plain(md: str) -> str:
    """Markdown/HTML to visible plain text, one block per line."""
    for pat, rep in _MD_STEPS:
        md = pat.sub(rep, md)
    return md


_SUP_MAP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_SUP_TAG = _rx(r"<sup>(.*?)</sup>", _I | regex.DOTALL)
_SUP_RUN = _rx("[\u2070\xb9\xb2\xb3\u2074-\u2079]+")
_FULLWIDTH_DIGIT = _rx("[\uff10-\uff19]")
_DASHES = _rx("[\u2010-\u2014\u2212\ufe63\uff0d]")
_SPACES = _rx("[\xa0\u202f\u2009]")
# A punctuation dash between letters: "behaviors—including" is two words ("4-G-5—Alternate" keeps its dash).
_EM_DASH = _rx(r"(?<=\p{L})[\u2014\u2015](?=\p{L})")
_FRACTION_SLASH = _rx("\u2044")  # NFKC of "¾" is "3⁄4": read as 3/4
_SPACED_DATE = _rx(r"(?<![0-9.])([0-9]{1,2})\.[ \t]+([0-9]{1,2})\.[ \t]+([0-9]{2,4})(?![0-9])")  # 23. 2. 2021
_SPACED_RANGE = _rx(r"([0-9])-[ \t]+([0-9])")  # "17- 43": a range wrapped after its dash


def tok_normalize(md: str) -> str:
    t = unicodedata.normalize("NFC", md)
    t = _SUP_TAG.sub(r" \1 ", t)
    t = _SUP_RUN.sub(lambda m: " " + "".join(_SUP_MAP[c] for c in m.group(0)) + " ", t)
    t = _FULLWIDTH_DIGIT.sub(lambda m: str(ord(m.group(0)) - 0xFF10), t)
    t = _EM_DASH.sub(" ", t)
    t = _FRACTION_SLASH.sub("/", t)
    t = _DASHES.sub("-", t)
    t = _SPACES.sub(" ", t)
    t = _SPACED_DATE.sub(r"\1.\2.\3", t)
    t = _SPACED_RANGE.sub(r"\1-\2", t)
    return md_plain(t)


# ---------------------------------------------------------------------------
# Token scanner
# ---------------------------------------------------------------------------

_AN = "[0-9A-Za-z\xc0-\u024f\u0370-\u03ff\u0400-\u04ff]"
_JOIN = r"[.,:/\-]"
_TLDS = "com|org|net|gov|edu|int|io|ai|info|co|uk|de|fr|eu|ca|au|us"
_URL = rf"(?:https?://|www\.)[^{_JS_WS}<>()\[\]\"'|“”‘’]+"
_TOKEN = (
    r"(?<email>[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+)"
    rf"|(?<url>{_URL})"
    rf"|(?<domain>(?<![@A-Za-z0-9_.])[A-Za-z0-9][A-Za-z0-9\-]*(?:\.[A-Za-z0-9\-]+)*\.(?:{_TLDS})(?!{_W}))"
    rf"|(?<num>{_AN}+(?:{_JOIN}{_AN}+)*)"
)
_CJK = "[\u3005-\u3007\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u0e00-\u0e7f]"
# A math operator (see _MATH_CMDS). = < > count only before a number and not at a line start or escaped (\\>):
# "p < 0.05", "x = 5". Elsewhere they are markup, arrows or quote marks (<!-- -->, <url>, ->, ==, "> Formatted").
_REL = r"(?<![=<>\u2039\\-])(?<!\A[ \t]*)[=<>\u2039](?![=<>\u2039])"
_OP = rf"[∑∏∫≤≥≠±∓]|{_REL}(?=[ \t]*[-+±]?\.?[0-9])"
_SCAN = _rx(
    rf"(?<crit>{_TOKEN})|(?<sym>[%‰\p{{Sc}}])|(?<op>{_OP})|(?<char>{_CJK})"
    rf"|(?<word>[\p{{L}}\p{{N}}](?:(?!{_CJK})[\p{{L}}\p{{N}}\p{{M}}])*)",
    _I,
)
_CJK1 = _rx(rf"\A{_CJK}\Z")
_GLYPH = str.maketrans(
    {
        "\u0430": "a",
        "\u0435": "e",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0445": "x",
        "\u0443": "y",
        "\u0456": "i",
        "\u03bf": "o",
        "\u03b1": "a",
    }
)
UNITS = frozenset("mg µg μg ug g kg km cm mm m mi ml l pci mbps gb mb kb kwh kw mw w v hz °c °f".split())
# Units that are critical right after a number ("12 MWh"); as bare words they are ordinary. The
# one-letter ones of UNITS count only when they do not read as a reference ("s. 22", "L 194").
NUMBER_UNITS = frozenset(u for u in UNITS if len(u) > 1) | frozenset(
    "min nm µm μm dl cl gal lb lbs oz ft yd ha mwh gwh twh kj mj kcal cal ghz mhz khz tb gbps kbps "
    "psi kpa mpa bar ppm ppb mmol mol kn mph kph".split()
)
NEGATORS = frozenset(
    (
        "not no never nor neither none nobody nothing nowhere without cannot nicht kein keine keinen keiner keinem "
        "nie niemals pas jamais aucun aucune sans non nunca ningún ninguna sin nem não sem niet geen nooit "
        "нет не ни 不 没 沒 無 无 未 非 別 别"
    ).split()
)
_NT = _rx(r"(do|does|did|is|are|was|were|has|have|had|ca|can|could|should|would|wo|must|need|ai|might)n\Z")
# Opposite words: one written for the other flips the meaning (critical), as a negation does. A short curated
# English list with the inflections written out; add a pair only when no context can make the swap harmless.
_ANTONYM_PAIRS = (
    ("rise rises rose risen rising", "fall falls fell fallen falling"),
    ("increase increased increases increasing", "decrease decreased decreases decreasing"),
    ("approve approved approves approval", "reject rejected rejects rejection"),
    ("approve approved approves approval", "deny denied denies denial"),
    ("accept accepted accepts acceptance", "reject rejected rejects rejection"),
    ("profit profits", "loss losses"),
    ("gain gains", "loss losses"),
    ("above", "below"),
    ("higher", "lower"),
    ("positive", "negative"),
    ("include included includes including", "exclude excluded excludes excluding"),
    ("allow allowed allows", "prohibit prohibited prohibits"),
)
_ANTONYM_SIDE: dict[str, set[tuple[int, int]]] = {}  # word -> {(pair index, side)}
for _n, _side, _w in ((n, side, w) for n, p in enumerate(_ANTONYM_PAIRS) for side in (0, 1) for w in p[side].split()):
    _ANTONYM_SIDE.setdefault(_w, set()).add((_n, _side))


def _flipped(rk: Sequence[str], hk: Sequence[str]) -> tuple[str, str] | None:
    """A reference word replaced by its opposite in the output (rose / fell), or None."""
    only_r, only_h = set(rk) - set(hk), set(hk) - set(rk)
    for a in only_r:
        for b in only_h:
            if any((n, 1 - side) in _ANTONYM_SIDE.get(b, ()) for n, side in _ANTONYM_SIDE.get(a, ())):
                return a, b
    return None


_SPLIT_RUN = _rx(r"([.,:/\-])")
_DIGIT = _rx("[0-9]")
_TRAIL_NUM = _rx(r"[.,:;\-/]+\Z")
_TRAIL_ID = _rx(r"[.,;:)]+\Z")
_THOUSANDS = _rx(r"\A[+-]?[0-9]{1,3}(,[0-9]{3})+(\.[0-9]+)?\Z")
_THOUSANDS_DOT = _rx(r"\A[+-]?[0-9]{1,3}(\.[0-9]{3}){2,}(,[0-9]+)?\Z")  # 200.000.000 (two or more groups)
_ABBREV = _rx(r"\A\p{L}(?:\.\p{L})+\Z")  # U.S, e.g, U.S.A: one word without the dots
_PURE_NUM = _rx(r"\A#[0-9]+\Z")
_SC = _rx(r"\p{Sc}")
_QUANTITY = _rx(r"\A(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]*)(?:\.[0-9]+)?\Z")  # 5, 0.45, 1,250.5
_TAG_TAIL = _rx(rf"(?:{_S}|\u2039[^>\u203a\n]*[>\u203a])+\Z")  # spaces and leftover tags (``‹math›``) at the end


def _sign(line: str, s0: int, e0: int, piece: str) -> str:
    """The sign written before a quantity: "-" or "+", "" when no dash or plus precedes it, "?" when unclear.

    A sign is glued to the quantity (-5.2, -.9, -$5, ``<math>-70``) after an opening mark or a word of two or
    more letters. A spaced dash ("- 2"), a dash after a number or a one-letter word ("1993 -1994", "n -1"), a
    dash run ("---1"), a number in dashes ("-3-") or alone on its line (a page number) are unclear. Dates and
    IDs carry no sign.
    """
    j = s0 - 1
    if j >= 1 and line[j] == "." and line[j - 1] in "-+":
        j -= 1
    elif j >= 0 and _SC.match(line[j]):
        j -= 1
    before = line[: j + 1].rstrip()
    if before[-1:] not in ("-", "+"):
        return "?" if before[-2:] in ("-.", "+.") else ""
    if not _QUANTITY.search(piece) or j < 0 or line[j] not in "-+" or line[e0:].lstrip()[:1] == "-":
        return "?"
    if j > 0 and line[j - 1] not in _JS_TRIM and line[j - 1] not in "([=:;>\u203a":
        return "?"
    head = _TAG_TAIL.sub("", line[:j])
    if head[-1:].isdigit() or (head[-1:].isalpha() and not head[-2:-1].isalpha()):
        return "?"
    return "?" if not head and not _js_trim(line[e0:]) else line[j]


_LN = _rx(r"[\p{L}\p{N}]+")
_SENT_END = _rx(rf"[.!?。！？]['\")\]]*({_S}|\Z)")
_LOWER_START = _rx(r"\A\p{Ll}")
_UPPER_START = _rx(r"\A\p{Lu}")
_LETTER_LINE = _rx(rf"\A{_S}*\p{{L}}")
_GREEK_CYR = _rx("[\u0400-\u04ff\u0370-\u03ff]")
_MARKS = _rx(r"\p{M}")

# Fixes applied before tokenization (tags, fig, sup, latex, unit, neg).
_PAGE_TAGS = _rx(r"</?(?:page_header|page_footer)>", _I)
_PAGE_NUMBER_TAG = _rx(r"<page_number>([^<]*)</page_number>", _I)
_FIGURE = _rx(rf"!?\[(?:Figure|Image|Logo|Photo|Chart|Graphic|Icon|Diagram){_NW}[^\]]*\](?:{_DEST})?", _I)
# A transcribed visual element: ``[signature: Name]`` in a reference, ``![signature: Name](img.png)`` in an
# output. Both read as the transcribed text alone; a bare ``[signature]`` placeholder reads as nothing.
_TRANSCRIBED = _rx(
    rf"!?\[{_S}*(?:[A-Za-z]+[ \t]+){{0,2}}(?:signature|seal|stamp|handwriting|handwritten(?:[ \t]+[A-Za-z]+)?)"
    rf"{_S}*(?::([^\]]*))?\](?:{_DEST})?",
    _I,
)
_DOLLAR_SUP = _rx(rf"\$\^\{{?{_S}*([\p{{L}}\p{{N}}*†‡,]{{1,6}}){_S}*\}}?\$")
# Glued to the previous word only when written glued: ``mc<sup>2</sup>`` is one token, while
# ``2\n<sup>2024 AB</sup>`` or ``1990 <sup>25</sup>`` keep their space.
_SUP_SUB_TAG = _rx(rf"({_S}*)<(sup|sub)>{_S}*([^<]{{1,8}}?){_S}*</\2>", _I)
_SUP_SUB_CHAR = _rx("[\u2070\xb9\xb2\xb3\u2074-\u2079\u2080-\u2089]")
_WORD_NUM = _rx(r"(\p{L}{3,})([0-9]{1,3})(?![\p{L}\p{N}])")
_UNIT_GLUE = _rx(r"([0-9])(mm|cm|km|kg|mg|ml|µm|μm|nm|kwh|kw|mw|gb|mb|kb|hz|ghz|mhz|%)(?![\p{L}\p{N}])", _I)
_NUMERO = _rx(rf"(?<![A-Za-z0-9_])([Nn]){_S}*\.?{_S}*[º°]")
_LATEX = _rx(r"\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^$\n]+?)\$")
_LATEX_KEEP = _rx(r"[\\^_{}=]")
_LATEX_LETTER = _rx(rf"\A{_S}*[A-Za-z]{_S}*\Z")
_LATEX_WRAP = _rx(rf"\\(?:mathrm|text|mathbf|mathit|operatorname|left|right|big|Big){_NW}")
_LATEX_CMD = _rx(r"\\[a-zA-Z]+")
_LATEX_BRACES = _rx(r"[{}^_\\&]")
_LATEX_OPS = _rx(r"[-+=<>*/]")
# Math operators are critical tokens (_OP): a swapped relation (≤ / ≥, = / ≠) or big operator (∑ / ∏) changes
# the meaning. LaTeX commands (in math) and ASCII spellings (<=, !=, +/-) read as the symbol; ‹ is a < (_NOT_TAG).
_MATH_CMDS = {
    **dict.fromkeys(("le", "leq", "leqq", "leqslant"), "≤"),
    **dict.fromkeys(("ge", "geq", "geqq", "geqslant"), "≥"),
    **dict.fromkeys(("ne", "neq"), "≠"),
    **{"pm": "±", "mp": "∓", "sum": "∑", "prod": "∏", "int": "∫", "lt": "<", "gt": ">"},
}
_MATH_CMD = _rx(r"\\(" + "|".join(sorted(_MATH_CMDS, key=len, reverse=True)) + r")(?![A-Za-z])")
# A tag left in the text (an unknown ``<math display="inline">`` read as ``‹math ...>``, an escaped one, or a
# bracketed label such as <확인방법>) reads as ``‹math display "inline"›``: markup, not operators; words stay.
_TAG_LIKE = _rx(r"[<\u2039]/?(?![A-Za-z]+://)\p{L}[\p{L}\p{N}_:-]*(?:[ \t/][^<>\u2039\n]{0,300})?>")
_TAG_PUNCT = {"<": "\u2039", "\u2039": "\u2039", ">": "\u203a", "=": " "}
_ASCII_OPS = {"<=": "≤", ">=": "≥", "!=": "≠", "+/-": "±"}
_ASCII_OP = _rx(r"(?<![<>=!\u2039-])(?:[<\u2039]=|>=|!=)(?![<>=])|\+/-")
_QUOTE_S = _rx("[‘’]")
_QUOTE_D = _rx("[“”]")
_NOT_TAG = _rx(
    r"<(?!/?(?:p|div|span|br|b|i|u|s|em|strong|sup|sub|table|thead|tbody|tr|td|th|ul|ol|li|h[1-6]|a|img|code|pre|"
    r"blockquote|hr|figure|figcaption|caption|section|header|footer|page_number|del|ins|mark|small|font|center)"
    rf"{_NW}[^<>]*>)",
    _I,
)
_CHECKBOX = _rx(r"\[(?:yes|no|x| )\]|[✓✔✗✘☐☑☒■□]", _I)
# A checkbox's state (checked: ☑ ☒ [x] ✓ ✔; unchecked: ☐ □ [ ]) is kept on the token after it, so a flipped box
# is charged; a box written or left out costs nothing (pages use them as bullets). Crosses and ■ have no state.
_BOX_STATE = {"☑": "x", "☒": "x", "[x]": "x", "✓": "x", "✔": "x", "☐": "o", "□": "o", "[ ]": "o"}
_BOX_MARK = {"x": "\ue000", "o": "\ue001"}  # private-use placeholders, read back by _box_marks
_HYPHEN_BREAK = _rx(rf"(?<=\p{{L}})[-\xad](?:{_S}|[\ue000\ue001])*(?=\p{{L}})")  # across checkbox marks too


_DESCRIPTION = _rx(r"\b(?:signature|signed|stamp|seal|logo|handwritten|handwriting|illegible|official)\b", _I)


def _transcribed_sub(m: Any) -> str:
    """Keep what a visual element says (a name, a stamp's text), drop what describes it."""
    x = m.group(1) or ""
    return " " if _DESCRIPTION.search(x) else " " + x + " "


def _dollar_is_money(m: Any) -> bool:
    """``$5 to $10`` is two amounts, ``$n * m$`` or ``$1 : 5000$`` is inline math."""
    x = m.group(4)
    nxt = m.string[m.end() : m.end() + 1]
    return bool(x[:1].isspace() or x[-1:].isspace() or nxt.isdigit())


def _latex_unary(o: Any) -> bool:
    """A minus opening a number in math (``$-5$``, ``x = -5``) stays glued to it: it is the number's sign."""
    before = o.string[: o.start()].rstrip()
    return o.group(0) == "-" and o.string[o.end() : o.end() + 1].isdigit() and (not before or before[-1] in "=(,<>")


def _latex_sub(m: Any) -> str:
    x = next((g for g in m.groups() if g is not None), "")
    if m.group(4) is not None and not _LATEX_KEEP.search(x) and not _LATEX_LETTER.search(x) and _dollar_is_money(m):
        return str(m.group(0))
    x = _MATH_CMD.sub(lambda c: " " + _MATH_CMDS[c.group(1)] + " ", x)
    x = _ASCII_OP.sub(lambda c: _ASCII_OPS[c.group(0)], x)  # before the spacing below splits <= into < =
    x = _LATEX_WRAP.sub(" ", x)
    x = _LATEX_CMD.sub(" ", x)
    x = _LATEX_BRACES.sub(" ", x)
    x = _LATEX_OPS.sub(lambda o: " " + o.group(0) + ("" if _latex_unary(o) else " "), x)
    return " " + str(x) + " "


def _word_key(w: str) -> str:
    return unicodedata.normalize("NFKC", w).lower().translate(_GLYPH)


def split_run(run: str) -> list[str]:
    """Split an alphanumeric run at separators that do not join digits."""
    parts = _SPLIT_RUN.split(run)
    pieces, seps = parts[0::2], parts[1::2]
    has = [bool(_DIGIT.search(p)) for p in pieces]
    out: list[str] = []
    cur = pieces[0]
    for i, sep in enumerate(seps):
        bound = (has[i] and has[i + 1]) if sep in ".,:" else (has[i] or has[i + 1])
        if bound:
            cur += sep + pieces[i + 1]
        else:
            out.append(cur)
            cur = pieces[i + 1]
    out.append(cur)
    return out


@dataclass
class Token:
    key: str
    surf: str
    crit: bool
    b: int  # line index
    s: int  # start offset in the line
    e: int  # end offset in the line
    end: bool = False  # a sentence ends after this token
    sign: str = ""  # sign written before a number: "-", "+", "" (none) or "?" (unclear), see _sign
    box: str = ""  # state of a checkbox written just before the token: "x" checked, "o" unchecked, "" none


def _page_number_sub(m: Any) -> str:
    """A page number is its own line, unless written glued to text: "<page_number>7</page_number>/7" is 7/7."""
    before = m.string[m.start() - 1 : m.start()]
    after = m.string[m.end() : m.end() + 1]
    glued = (before and not before.isspace()) or (after and not after.isspace())
    return str(m.group(1)) if glued else "\n" + str(m.group(1)) + "\n"


def _after_digit(m: Any, content: str) -> bool:
    """A superscript digit right after a digit is an exponent (10<sup>9</sup>, 10⁹), not part of the number."""
    before = m.string[m.start() - 1 : m.start()]
    return "0" <= before <= "9" and "0" <= unicodedata.normalize("NFKC", content[:1]) <= "9"


def _sup_sub_spaced(m: Any) -> bool:
    """A ``<sup>``/``<sub>`` keeps a space before it when written spaced, after a digit, or right after another one.

    Consecutive tags are separate marks: ``Moreau<sup>1</sup><sup>ID</sup>`` (footnote then ORCID icon) reads
    ``Moreau1 ID``, so the footnote still splits from the name.
    """
    after_tag = m.string[max(0, m.start() - 6) : m.start()].lower() in ("</sup>", "</sub>")
    return bool(m.group(1)) or after_tag or _after_digit(m, m.group(3))


def preprocess(md: str) -> str:
    """Fixes applied before tokenization."""
    md = _PAGE_TAGS.sub("\n", md)
    md = _PAGE_NUMBER_TAG.sub(_page_number_sub, md)
    md = _TRANSCRIBED.sub(_transcribed_sub, md)
    md = _FIGURE.sub(" ", md)
    md = _DOLLAR_SUP.sub(r"<sup>\1</sup>", md)
    md = _SUP_SUB_TAG.sub(lambda m: (" " if _sup_sub_spaced(m) else "") + m.group(3), md)
    md = _SUP_SUB_CHAR.sub(
        lambda m: (" " if _after_digit(m, m.group(0)) else "") + unicodedata.normalize("NFKC", m.group(0)), md
    )
    md = _LATEX.sub(_latex_sub, md)
    md = _WORD_NUM.sub(r"\1 \2", md)
    md = _UNIT_GLUE.sub(r"\1 \2", md)
    md = _NUMERO.sub(r"\1o.", md)
    return md


_URL_SPAN = _rx(_URL, _I)


def to_lines(md: str) -> list[str]:
    """Plain-text lines of a Markdown page (checkbox marks read as spaces)."""
    return _box_marks(_raw_lines(md))[0]


def _raw_lines(md: str) -> list[str]:
    safe = unicodedata.normalize("NFKC", preprocess(md))
    safe = _QUOTE_S.sub("'", safe)
    safe = _QUOTE_D.sub('"', safe)
    safe = _NOT_TAG.sub("\u2039", safe)
    text = html.unescape(tok_normalize(safe))
    text = _TAG_LIKE.sub(lambda m: "".join(_TAG_PUNCT.get(c, c) for c in m.group(0)), text)
    text = _ASCII_OP.sub(lambda m: _ASCII_OPS[m.group(0).replace("\u2039", "<")], text)
    text = _CHECKBOX.sub(lambda m: " " + _BOX_MARK.get(_BOX_STATE.get(m.group(0).lower(), ""), "") + " ", text)
    text = str(_HYPHEN_BREAK.sub("", text))
    # inside a URL an underscore reads as a space: a parser may render "Two_RBO" as "Two RBO" (or the reverse)
    return [_URL_SPAN.sub(lambda m: m.group(0).replace("_", " "), line) for line in text.split("\n")]


_BOX_RX = _rx("[\ue000\ue001]")


def _box_marks(lines: list[str]) -> tuple[list[str], list[tuple[int, int, str]]]:
    """Checkbox placeholders as (line, offset, state), and the lines with each placeholder turned into a space."""
    marks = [
        (b, m.start(), "x" if m.group(0) == "\ue000" else "o")
        for b, line in enumerate(lines)
        for m in _BOX_RX.finditer(line)
    ]
    return ([_BOX_RX.sub(" ", line) for line in lines] if marks else lines), marks


_LINE_NO = _rx(rf"\A{_S}*([0-9]{{1,3}})(?={_S}|\Z)")


def _line_number_lines(lines: Sequence[str]) -> list[int]:
    """Indexes of the lines that open with a margin line number.

    A margin line number is a bare integer (no "." or ")", so Markdown lists are left alone) that counts
    up by one over LINE_NO_RUN or more consecutive non-blank lines, as on pleading paper and numbered
    manuscripts. Lines holding only the number (a column of numbers written as its own block) count too.
    """
    found: list[int] = []
    run: list[int] = []
    prev: int | None = None
    for i, line in enumerate(lines):
        if not _js_trim(line):
            continue
        m = _LINE_NO.search(line)
        n = int(m.group(1)) if m else None
        if n is not None and prev is not None and n == prev + 1 and run:
            run.append(i)
        else:
            if len(run) >= LINE_NO_RUN:
                found.extend(run)
            run = [i] if n is not None else []
        prev = n
    if len(run) >= LINE_NO_RUN:
        found.extend(run)
    return found


def has_line_numbers(md: str) -> bool:
    """Whether the page carries margin line numbers (see ``_line_number_lines``).

    At least half of the numbered lines must carry text after the number, so a page whose content is a
    column of numbers keeps them as content.
    """
    lines = to_lines(md)
    found = _line_number_lines(lines)
    with_text = sum(1 for i in found if _js_trim(_LINE_NO.sub("", lines[i], count=1)))
    return bool(found) and 2 * with_text >= len(found)


def _unit_letter_ok(toks: Sequence[Token], i: int, lines: Sequence[str]) -> bool:
    """A one-letter unit ("20 m", "5 g/L") unless it reads as a reference ("s. 22", "L 194")."""
    t = toks[i]
    after = lines[t.b][t.e :].lstrip()
    nxt = toks[i + 1] if i + 1 < len(toks) else None
    return not after.startswith(".") and not (nxt is not None and nxt.b == t.b and _DIGIT.search(nxt.key))


def _attach_boxes(toks: Sequence[Token], marks: Sequence[tuple[int, int, str]]) -> None:
    """Give each token the state of the last checkbox mark between the previous token and itself."""
    q = 0
    for t in toks:
        while q < len(marks) and (marks[q][0], marks[q][1]) < (t.b, t.s):
            t.box = marks[q][2]
            q += 1


def tokenize(md: str, drop_line_numbers: bool = False) -> tuple[list[Token], list[str]]:
    """Token stream of a Markdown page and the plain-text lines it was read from.

    ``drop_line_numbers`` removes margin line numbers (``_line_number_lines``) before tokenizing.
    """
    lines, marks = _box_marks(_raw_lines(md))
    if drop_line_numbers:
        cut: dict[int, int] = {}
        for i in _line_number_lines(lines):
            width = len(lines[i])
            lines[i] = _LINE_NO.sub("", lines[i], count=1)
            cut[i] = width - len(lines[i])
        marks = [(b, max(0, x - cut.get(b, 0)), st) for b, x, st in marks]
    toks: list[Token] = []
    for b, line in enumerate(lines):
        for m in _SCAN.finditer(line):
            s0, e0 = m.span()
            g = m.group(0)
            if m.group("sym") is not None or m.group("op") is not None:
                g = g.replace("\u2039", "<")
                toks.append(Token("#" + g, g, True, b, s0, s0 + 1))
                continue
            if m.group("crit") is None:
                w = unicodedata.normalize("NFKC", g)
                lw = w.lower().translate(_GLYPH)
                if lw in UNITS:
                    toks.append(Token("#" + lw, w, True, b, s0, e0))
                else:
                    toks.append(Token(lw, w, False, b, s0, e0))
                continue
            if m.group("num") is None:
                k = _TRAIL_ID.sub("", g)
                toks.append(Token("#" + k.lower(), k, True, b, s0, e0))
                continue
            if _ABBREV.search(g):
                w = "".join(g.split("."))
                toks.append(Token(_word_key(w), w, False, b, s0, e0))
                continue
            pos = s0
            pieces = split_run(g)
            for p in pieces:
                s = line.find(p, pos)
                pos = s + len(p)
                if not _DIGIT.search(p):
                    for w_m in _LN.finditer(p):
                        toks.append(
                            Token(_word_key(w_m.group(0)), w_m.group(0), False, b, s + w_m.start(), s + w_m.end())
                        )
                    continue
                k = _TRAIL_NUM.sub("", p)
                if _THOUSANDS.search(k):
                    k = k.replace(",", "")
                elif _THOUSANDS_DOT.search(k):
                    k = k.replace(".", "").replace(",", ".")
                if k:
                    num = Token("#" + k.lower(), _TRAIL_NUM.sub("", p), True, b, s, s + len(p))
                    num.sign = _sign(line, s, s + len(p), p) if s == s0 else "?"
                    toks.append(num)
    # A unit right after a number is part of the quantity: "12 kWh" vs "12 MWh" is a wrong value.
    for i in range(1, len(toks)):
        t, prev = toks[i], toks[i - 1]
        if (
            not t.crit
            and (t.key in NUMBER_UNITS or (t.key in UNITS and _unit_letter_ok(toks, i, lines)))
            and prev.crit
            and prev.b == t.b
            and _DIGIT.search(prev.key)
            and not _js_trim(lines[t.b][prev.e : t.s])
        ):
            t.crit = True
            t.key = "#" + t.key
    _attach_boxes(toks, marks)
    blk_len: dict[int, int] = {}
    for t in toks:
        blk_len[t.b] = blk_len.get(t.b, 0) + 1
    n_tok = len(toks)
    for i, t in enumerate(toks):
        if i + 1 == n_tok:
            t.end = True
            continue
        n = toks[i + 1]
        if n.b != t.b:
            tail = _js_trim(lines[t.b][t.e :])
            cont = (
                tail in ("", ",")
                and not t.crit
                and bool(_LOWER_START.search(n.surf))
                and blk_len[t.b] >= 6
                and bool(_LETTER_LINE.search(lines[n.b]))
            )
            t.end = not cont
        else:
            t.end = bool(_SENT_END.search(lines[t.b][t.e : n.s] + " "))
    for i, t in enumerate(toks):
        if t.key == "no":
            line = lines[t.b]
            if (t.e < len(line) and line[t.e] == ".") or (i + 1 < n_tok and toks[i + 1].key[:1] == "#"):
                continue  # "No. 5" is a number sign, not a negation
        if t.key in NEGATORS or (t.key == "t" and i > 0 and _NT.search(toks[i - 1].key)):
            t.crit = True
    return toks, lines


# ---------------------------------------------------------------------------
# Alignment primitives
# ---------------------------------------------------------------------------


def _tile(
    r: Sequence[str], h: Sequence[str], rm: list[bool], hm: list[bool], k: int, tiles: list[tuple[int, int, int]]
) -> None:
    """Greedy string tiling, longest first; ties go to the closest relative position."""
    n_r, n_h = len(r), len(h)
    while True:
        idx: dict[tuple[str, ...], list[int]] = {}
        for j in range(n_h - k + 1):
            if not any(hm[j : j + k]):
                idx.setdefault(tuple(h[j : j + k]), []).append(j)
        cands: list[tuple[int, float, int, int]] = []
        for i in range(n_r - k + 1):
            if any(rm[i : i + k]):
                continue
            js = idx.get(tuple(r[i : i + k]))
            if not js:
                continue
            for j in js:
                if i and j and r[i - 1] == h[j - 1] and not rm[i - 1] and not hm[j - 1]:
                    continue  # not left-maximal
                length = k
                while (
                    i + length < n_r
                    and j + length < n_h
                    and r[i + length] == h[j + length]
                    and not rm[i + length]
                    and not hm[j + length]
                ):
                    length += 1
                cands.append((-length, abs(i / n_r - j / n_h), i, j))
        cands.sort()
        took = False
        for neg_len, _, i, j in cands:
            length = -neg_len
            if any(rm[i : i + length]) or any(hm[j : j + length]):
                if took:
                    break  # a longer candidate lost part of its span: recompute first
                continue
            for x in range(length):
                rm[i + x] = hm[j + x] = True
            tiles.append((i, j, length))
            took = True
        if not took:
            return


def _lcs_pairs(a: Sequence[str], b: Sequence[str]) -> list[tuple[int, int]]:
    """In-order LCS pairs; the backtrack prefers skipping in ``a`` on ties."""
    n, m = len(a), len(b)
    if not n or not m or n * m > LCS_CELL_LIMIT:
        return []
    rows: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ai = a[i]
        cur = rows[i]
        nxt = rows[i + 1]
        for j in range(m - 1, -1, -1):
            if ai == b[j]:
                cur[j] = nxt[j + 1] + 1
            else:
                x, y = nxt[j], cur[j + 1]
                cur[j] = x if x >= y else y
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i += 1
            j += 1
        elif rows[i + 1][j] >= rows[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    return int(LCSseq.similarity(list(a), list(b)))


def _f1(m: int, a: int, b: int) -> float | None:
    return 2 * m / (a + b) if a + b else None


def _heaviest_chain(items: Sequence[tuple[int, int, int]]) -> int:
    """Max total weight of items increasing in both hyp (given order) and ref position."""
    refs = sorted({it[1] for it in items})
    rank = {r: x + 1 for x, r in enumerate(refs)}
    tree = [0] * (len(refs) + 1)
    best = 0
    for _, ref, w in items:
        q, x = 0, rank[ref] - 1
        while x > 0:
            q = max(q, tree[x])
            x -= x & -x
        v = q + w
        best = max(best, v)
        x = rank[ref]
        while x <= len(refs):
            if tree[x] < v:
                tree[x] = v
            x += x & -x
    return best


def _lis_len(seq: Sequence[int]) -> int:
    tails: list[int] = []
    for v in seq:
        p = bisect_left(tails, v)
        if p == len(tails):
            tails.append(v)
        else:
            tails[p] = v
    return len(tails)


def _runs(n: int, used: Sequence[bool]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    st = -1
    for i in range(n + 1):
        if i < n and not used[i]:
            if st < 0:
                st = i
        elif st >= 0:
            out.append((st, i))
            st = -1
    return out


def _fold(z: str) -> str:
    return str(_MARKS.sub("", unicodedata.normalize("NFD", z)))


_SMALL_INT = _rx(r"\A#[0-9]{1,3}\Z")
_NON_ALNUM = _rx(r"[^\p{L}\p{N}]+")


def _shape(keys: Sequence[str]) -> tuple[str, ...]:
    """Event signature: the keys, with small bare integers (line, list, page numbers) collapsed so a run of
    omitted line numbers is one quirk; other numbers keep their value."""
    return tuple("#0" if _SMALL_INT.search(k) else k for k in keys)


def _joined(keys: Sequence[str]) -> str:
    """Keys glued together: equal for the same characters split or joined differently."""
    return "".join(k.lstrip("#") for k in keys)


def _same_number(rc: Sequence[Token], hc: Sequence[Token]) -> bool:
    """Same digit groups and signs, other separators: 723-3600 / 723 3600, 23. 2. 2021 / 23.2.2021, 4.00 / 4:00.

    Groups are read from the written form, so thousands written with a comma, a space, a no-break or thin space
    or an apostrophe (1,250 / 1 250 / 1'250) are the same number written differently.
    """

    def groups(ts: Sequence[Token]) -> tuple[list[str], list[str]]:
        text = " ".join(t.surf.lower() for t in ts)
        return [g for g in _NON_ALNUM.split(text) if g], [t.key for t in ts if not _LN.search(t.key)]

    return groups(rc) == groups(hc)


def _in_order(ha: int, hb: int, pref: Sequence[int]) -> bool:
    """Consecutive sentence tokens stay together: no backward jump, no other matched output between."""
    return hb > ha and pref[hb] - pref[ha + 1] == 0


def _moved_runs(big: Sequence[tuple[int, int, int]]) -> int:
    """Blocks out of order, counted as runs: blocks (in output order) outside the heaviest in-order chain,
    where consecutive out-of-chain blocks that keep their own reference order are one move."""
    n = len(big)
    if n < 2:
        return 0
    refs = [it[1] for it in big]
    if n > CHAIN_LIMIT:
        return n - min(_lis_len(refs), n)
    best = [it[2] for it in big]
    back = [-1] * n
    for i in range(n):
        for j in range(i):
            if refs[j] < refs[i] and best[j] + big[i][2] > best[i]:
                best[i] = best[j] + big[i][2]
                back[i] = j
    in_chain = [False] * n
    i = max(range(n), key=lambda x: best[x])
    while i >= 0:
        in_chain[i] = True
        i = back[i]
    runs = 0
    for i in range(n):
        if in_chain[i]:
            continue
        if not (i > 0 and not in_chain[i - 1] and refs[i - 1] < refs[i]):
            runs += 1
    return runs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class TextNextEvents:
    critical: int = 0
    major: int = 0
    minor: int = 0
    broken: int = 0
    moved: int = 0
    uncharged: int = 0  # major occurrences repeating an already charged one (systematic quirks)
    examples: list[tuple[str, str, str]] = field(default_factory=list)

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kind, _, _ in self.examples:
            out[kind] = out.get(kind, 0) + 1
        return out


@dataclass
class TextNextResult:
    score: float
    proportional: float
    floor: float
    words: float | None
    critical: float | None
    integrity: float | None
    order: float | None
    recall: float | None
    precision: float | None
    case: float | None
    displaced: int
    n_crit: int
    n_ref: int
    n_hyp: int
    events: TextNextEvents

    def metadata(self, max_examples: int = 20) -> dict[str, Any]:
        return {
            "proportional": self.proportional,
            "floor": self.floor,
            "words": self.words,
            "critical": self.critical,
            "integrity": self.integrity,
            "order": self.order,
            "recall": self.recall,
            "precision": self.precision,
            "case": self.case,
            "displaced": self.displaced,
            "n_critical_tokens": self.n_crit,
            "n_ref_tokens": self.n_ref,
            "n_hyp_tokens": self.n_hyp,
            "events": {
                "critical": self.events.critical,
                "major": self.events.major,
                "minor": self.events.minor,
                "broken": self.events.broken,
                "moved": self.events.moved,
            },
            "event_kinds": self.events.kinds(),
            "uncharged_repeats": self.events.uncharged,
            "event_examples": [
                {"kind": k, "reference": r[:160], "output": h[:160]} for k, r, h in self.events.examples[:max_examples]
            ],
        }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _fix_eff(n: float, n_ref: int) -> float:
    """First occurrence counts in full, later ones scale down on long pages."""
    return min(n, 1.0) + max(0.0, n - 1) * min(1.0, FLOOR_L0 / max(1, n_ref))


def next_floor(ev: TextNextEvents, n_ref: int) -> float:
    """Severity floor. Moved blocks and cut sentences are charged as majors, but their share of the
    deduction is capped at ``FLOOR["order_cap"]``: one reading-order choice (two columns interleaved
    row by row) produces dozens of moves while every word, number and record is intact."""
    major = FLOOR["major"] * _fix_eff(ev.major, n_ref)
    with_order = FLOOR["major"] * _fix_eff(ev.major + 0.5 * ev.broken + ev.moved, n_ref)
    return _clamp01(
        1
        - FLOOR["critical"] * _fix_eff(ev.critical, n_ref)
        - major
        - min(FLOOR["order_cap"], with_order - major)
        - min(FLOOR["minor_cap"], FLOOR["minor"] * ev.minor)
    )


def next_proportional(
    words: float | None, critical: float | None, integrity: float | None, order: float | None, n_crit: int
) -> float:
    ramp = min(1.0, n_crit / CRIT_RAMP)
    loss = (
        NEXT_W["words"] * (1 - (1.0 if words is None else words))
        + NEXT_W["critical"] * ramp * (1 - (1.0 if critical is None else critical))
        + NEXT_W["integrity"] * (1 - (1.0 if integrity is None else integrity))
        + NEXT_W["order"] * (1 - (1.0 if order is None else order))
    )
    return _clamp01(1 - loss)


# A number wrapped in underline (<u>, <ins>) or strikeout (~~, <s>, <del>, <strike>) markup.
_DECO_NUM = _rx(r"<(u|ins|s|del|strike)>[ \t]*([0-9]+)[ \t]*</\1>|~~[ \t]*([0-9]+)[ \t]*~~", _I)
_UNDERLINE_TAGS = frozenset(("u", "ins"))


def _split_glued_decorated_numbers(ref_md: str, hyp_md: str) -> str:
    """Split output numbers that glue two adjacent reference numbers of different decoration.

    A revision mark such as ``<u>103</u> ~~101~~`` prints as "103101" with no gap on the page, so an
    output that drops the decoration reads one number. Each glued pair found in the reference (an
    underlined and a struck number separated only by spaces) is split back in the output, in the
    reference order only; the lost decoration is scored by formatting_v2.
    """
    marks = list(_DECO_NUM.finditer(ref_md))
    pairs: set[tuple[str, str]] = set()
    for a, b in zip(marks, marks[1:], strict=False):
        under_a = (a.group(1) or "").lower() in _UNDERLINE_TAGS
        under_b = (b.group(1) or "").lower() in _UNDERLINE_TAGS
        if under_a != under_b and not ref_md[a.end() : b.start()].strip(" \t"):
            pairs.add((a.group(2) or a.group(3), b.group(2) or b.group(3)))
    for left, right in pairs:
        hyp_md = regex.sub(rf"(?<![0-9]|[0-9][.,]){left}{right}(?![0-9]|[.,][0-9])", f"{left} {right}", hyp_md)
    return hyp_md


ALT_FREE_WORDS = 8  # reference alt text over this many words may hold page text (second reading, score_text_next)
ALT_DUP_MIN = 5  # an output alt text of at least this many tokens ...
ALT_DUP_SHARE = 0.5  # ... with this share of them in runs of K+ tokens the reference has duplicates page text
ALT_JUNK_MIN = 4  # garbage is judged on alt texts of at least this many tokens
_ALT_IMAGE = _rx(rf"!\[([^\]\n]*)\](?:{_DEST})?")
_HTML_OPEN = _rx(r"<[A-Za-z][^<>]*>")
_ALT_ATTR = _rx(r"""\b(?:alt|title|aria-label)[ \t]*=[ \t]*(?:"([^"]*)"|'([^']*)')""", _I)
_ALT_LABEL = _rx(r"\A[ \t]*[A-Za-z]+[ \t]*:")  # "image:", "logo:" prefixes an output adds to alt text


def _long(text: str) -> bool:
    return len(text.split()) > ALT_FREE_WORDS


def _covered(keys: Sequence[str], other: Sequence[str]) -> float:
    """Share of ``keys`` inside a run of K or more tokens that ``other`` also has (common words alone do not count)."""
    if len(keys) < K:
        return 0.0
    grams = {tuple(other[i : i + K]) for i in range(len(other) - K + 1)}
    hit = [False] * len(keys)
    for i in range(len(keys) - K + 1):
        if tuple(keys[i : i + K]) in grams:
            hit[i : i + K] = [True] * K
    return sum(hit) / len(keys)


def _junk(text: str, toks: Sequence[Token]) -> bool:
    """OCR junk: under half of the visible characters are letters or digits, or most tokens are one character.

    A barcode or reference number (``barcode 9 781234 567890``) is not junk.
    """
    chars = [c for c in text if not c.isspace()]
    alnum = sum(1 for c in chars if c.isalnum())
    lone = sum(1 for t in toks if len(t.key.lstrip("#")) == 1 and t.key[-1:].isalnum() and not _CJK1.search(t.key))
    return len(toks) >= ALT_JUNK_MIN and (2 * alnum < len(chars) or 2 * lone > len(toks))


def alt_as_text(ref_keys: Sequence[str], md: str) -> str:
    """The output page with alt text that carries page content or junk written out as body text.

    Image alt text and alt / title / aria-label attributes are free: a picture description costs nothing
    whatever its length. Two cases are read as text. Alt text that duplicates page text (ALT_DUP_SHARE of its
    tokens, at least ALT_DUP_MIN, in runs of K or more tokens the reference also has) and that the body does
    not already give (same test), so page content placed only inside alt is matched normally; alt text
    repeating the body is ignored. And junk (``_junk``), charged as inserted text.
    """
    if not _ALT_IMAGE.search(md) and not _ALT_ATTR.search(md):
        return md
    body = [t.key for t in tokenize(md)[0]]

    def as_text(text: str) -> str | None:
        plain = _ALT_LABEL.sub("", text)
        toks = tokenize(plain)[0]
        keys = [t.key for t in toks]
        dup = len(keys) >= ALT_DUP_MIN and _covered(keys, ref_keys) >= ALT_DUP_SHARE
        keep = (dup and _covered(keys, body) < ALT_DUP_SHARE) or (not dup and _junk(plain, toks))
        if keep and dup:
            body.extend(["\n", *keys])  # a second alt repeating it is then a repeat of the body
        return "\n\n" + plain + "\n\n" if keep else None

    def image(m: Any) -> str:
        return as_text(str(m.group(1))) or str(m.group(0))

    def tag(m: Any) -> str:
        vals = [a.group(1) if a.group(1) is not None else a.group(2) for a in _ALT_ATTR.finditer(m.group(0))]
        return str(m.group(0)) + "".join(x for x in (as_text(str(v)) for v in vals) if x)

    return str(_HTML_OPEN.sub(tag, _ALT_IMAGE.sub(image, md)))


def _ref_alt_as_text(md: str) -> str:
    """The reference with its alt text over ALT_FREE_WORDS words written out as text (it may hold page text)."""

    def image(m: Any) -> str:
        return "\n\n" + str(m.group(1)) + "\n\n" if _long(m.group(1)) else str(m.group(0))

    return str(_ALT_IMAGE.sub(image, md))


def score_text_next(ref_md: str, hyp_md: str) -> TextNextResult:
    """Score an output page (``hyp_md``) against its reference page (``ref_md``).

    The output's alt text is free except page text placed only there, or junk (``alt_as_text``). A reference's
    long alt text reads as nothing (an output may leave a picture undescribed), but annotators sometimes put
    page text there: the page is then also scored with it as text, and the better reading counts.
    """
    rt, _ = tokenize(ref_md)
    hyp_md = alt_as_text([t.key for t in rt], _split_glued_decorated_numbers(ref_md, hyp_md))
    ht, _ = tokenize(hyp_md)
    res = score_tokens(rt, ht)
    ref_alt = _ref_alt_as_text(ref_md)
    if res.score < 1.0 and ref_alt != ref_md:
        alt = score_tokens(tokenize(ref_alt)[0], ht)
        if alt.score > res.score:
            res, ref_md = alt, ref_alt
    # Margin line numbers are optional page furniture: when the reference has them, the page is also
    # scored with them left out on both sides, and the better reading counts.
    if res.score < 1.0 and has_line_numbers(ref_md):
        alt = score_tokens(tokenize(ref_md, True)[0], tokenize(hyp_md, True)[0])
        if alt.score > res.score:
            return alt
    return res


def score_tokens(rt: list[Token], ht: list[Token]) -> TextNextResult:  # noqa: C901 - one scoring pass
    r_keys = [t.key for t in rt]
    h_keys = [t.key for t in ht]
    n_r, n_h = len(r_keys), len(h_keys)
    rm = [False] * n_r
    hm = [False] * n_h
    r2h = [-1] * n_r
    how: list[str | None] = [None] * n_r
    tiles: list[tuple[int, int, int]] = []

    def link(i: int, j: int, w: str) -> None:
        r2h[i] = j
        how[i] = w
        rm[i] = hm[j] = True

    _tile(r_keys, h_keys, rm, hm, K, tiles)
    for i, j, n in tiles:
        for x in range(n):
            link(i + x, j + x, "tile")
    tiles.sort(key=lambda t: t[0])

    # Gaps: reference between tiles vs output after the left copy / before the right copy.
    for t in range(len(tiles) + 1):
        left = tiles[t - 1] if t > 0 else None
        right = tiles[t] if t < len(tiles) else None
        r_idx = [x for x in range(left[0] + left[2] if left else 0, right[0] if right else n_r) if not rm[x]]
        if not r_idx:
            continue
        h_idx: list[int] = []
        y = left[1] + left[2] if left else 0
        while y < n_h and not hm[y]:
            h_idx.append(y)
            y += 1
        if right:
            seen = set(h_idx)
            before: list[int] = []
            y = right[1] - 1
            while y >= 0 and not hm[y]:
                if y not in seen:
                    before.append(y)
                y -= 1
            h_idx.extend(reversed(before))
        for a, b in _lcs_pairs([r_keys[x] for x in r_idx], [h_keys[y] for y in h_idx]):
            link(r_idx[a], h_idx[b], "gap")

    # Short reference lines matched as whole output lines, nearest relative position first.
    blocks: dict[int, list[int]] = {}
    for i, tk in enumerate(rt):
        blocks.setdefault(tk.b, []).append(i)
    h_pos: dict[str, list[int]] = {}
    for j, k in enumerate(h_keys):
        h_pos.setdefault(k, []).append(j)
    for idx in blocks.values():
        if len(idx) >= K or any(rm[i] for i in idx):
            continue
        size = len(idx)
        best: tuple[float, int] | None = None
        for j in h_pos.get(r_keys[idx[0]], ()):
            e = j + size
            if e > n_h:
                break
            if (
                all(h_keys[j + x] == r_keys[i] for x, i in enumerate(idx))
                and not any(hm[j:e])
                and (j == 0 or ht[j - 1].b != ht[j].b)
                and (e == n_h or ht[e].b != ht[e - 1].b)
            ):
                d = abs(j / n_h - idx[0] / n_r)
                if best is None or d < best[0]:
                    best = (d, j)
        if best is not None:
            tiles.append((idx[0], best[1], size))
            for x, i in enumerate(idx):
                link(i, best[1] + x, "tile")

    # Displaced: leftover identical keys anywhere (content credit only).
    pool: dict[str, list[int]] = {}
    for j, k in enumerate(h_keys):
        if not hm[j]:
            pool.setdefault(k, []).append(j)
    for i, k in enumerate(r_keys):
        c = pool.get(k)
        if rm[i] or not c or _CJK1.search(k):
            continue
        target = i / n_r
        bi = 0
        bd = abs(c[0] / n_h - target)
        for x in range(1, len(c)):
            d = abs(c[x] / n_h - target)
            if d < bd:
                bi, bd = x, d
        link(i, c.pop(bi), "disp")

    matched = sum(1 for x in r2h if x >= 0)

    # Error occurrences for the severity floor. Computed before the lanes: a number written with other
    # separators (723-3600 vs 723 3600) or words split/joined (non tariff vs nontariff) also count as
    # present in the words and critical lanes.
    ev = TextNextEvents()
    h_used = [False] * n_h
    h_override: dict[int, list[str]] = {}  # output token -> critical keys it stands for in the critical lane
    credit_r = credit_h = 0  # reference / output tokens present up to formatting
    charged: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()

    def is_neg(t: Token) -> bool:
        return t.crit and (t.key in NEGATORS or t.key == "t")

    def major(kind: str, rk: Sequence[str], hk: Sequence[str]) -> None:
        """One major occurrence; a systematic repeat (same kind and text, digits ignored) is charged once."""
        ev.examples.append((kind, " ".join(rk), " ".join(hk)))
        sig = (kind, _shape(rk), _shape(hk))
        if sig in charged:
            ev.uncharged += 1
            return
        charged.add(sig)
        ev.major += 1

    def as_present(rr: Sequence[Token], hy_idx: Sequence[int]) -> None:
        nonlocal credit_r, credit_h
        credit_r += len(rr)
        credit_h += len(hy_idx)
        if hy_idx:
            h_override[hy_idx[0]] = [t.key for t in rr if t.crit]
            for y in hy_idx[1:]:
                h_override[y] = []

    ref_bag = set(r_keys)
    for a, b in _runs(n_r, rm):
        rr = rt[a:b]
        hp = r2h[a - 1] if a > 0 else -1
        hn = r2h[b] if b < n_r else n_h
        hy_idx: list[int] = []
        if hn > hp:
            for y in range(hp + 1, hn):
                if not hm[y] and not h_used[y]:
                    h_used[y] = True
                    hy_idx.append(y)
        hy = [ht[y] for y in hy_idx]
        rk = [t.key for t in rr]
        hk = [t.key for t in hy]
        if (
            hy
            and not any(_DIGIT.search(k) for k in rk)
            and (_joined(rk) == _joined(hk) or (len(rk) != len(hk) and _fold(_joined(rk)) == _fold(_joined(hk))))
        ):
            ev.minor += 1
            ev.examples.append(("words split or joined", " ".join(rk), " ".join(hk)))
            as_present(rr, hy_idx)
            continue
        rc = [t for t in rr if t.crit and not is_neg(t)]
        hc = [t for t in hy if t.crit and not is_neg(t)]
        rw = [t for t in rr if not t.crit]
        hw = [t for t in hy if not t.crit]
        r_neg = sum(1 for t in rr if is_neg(t))
        h_neg = sum(1 for t in hy if is_neg(t))
        if r_neg != h_neg and (hy or len(rr) <= NEG_OMIT_MAX):
            ev.critical += 1
            ev.examples.append(("negation changed", " ".join(rk), " ".join(hk)))
        if not hy:
            if len(rr) >= OMIT_PASSAGE:
                ev.critical += 1
                ev.examples.append(("omitted passage", " ".join(rk), ""))
            elif rr:
                major("omitted", rk, [])
            continue
        if rc and hc and _same_number(rc, hc):
            ev.minor += 1
            ev.examples.append(("number written differently", " ".join(t.key for t in rc), " ".join(t.key for t in hc)))
            as_present(rc, [y for y in hy_idx if ht[y].crit and not is_neg(ht[y])])
        else:
            sub = min(len(rc), len(hc))
            if sub:
                ev.critical += sub
                ev.examples.append(
                    ("wrong number or identifier", " ".join(t.key for t in rc), " ".join(t.key for t in hc))
                )
            if len(rc) > len(hc):
                major("omitted", [t.key for t in rc[len(hc) :]], [])
        flip = _flipped([t.key for t in rw], [t.key for t in hw])
        if flip:
            ev.critical += 1
            ev.examples.append(("meaning flipped", flip[0], flip[1]))
            continue
        if rw:
            rs = " ".join(t.key for t in rw)
            hs = " ".join(t.key for t in hw)
            if not hs:
                major("omitted", [t.key for t in rw], [])
                continue
            sim = 1 - Levenshtein.distance(rs, hs) / max(_utf16_len(rs), _utf16_len(hs))
            name = any(_UPPER_START.search(t.surf or "") for t in rw) and a > 0 and not rt[a - 1].end
            if _fold(rs) == _fold(hs):
                ev.minor += len(rw)
                ev.examples.append(("accent only", rs, hs))
            elif name and sim < 0.85:
                ev.critical += 1
                ev.examples.append(("name changed", rs, hs))
            elif sim >= 0.7:
                ev.minor += len(rw)
                ev.examples.append(("spelling", rs, hs))
            else:
                major("wrong words", [t.key for t in rw], [t.key for t in hw])
    for a, b in _runs(n_h, [hm[j] or h_used[j] for j in range(n_h)]):
        hh = ht[a:b]
        if not hh:
            continue
        hk = [t.key for t in hh]
        if any(is_neg(t) for t in hh):
            ev.critical += 1
            ev.examples.append(("negation added", "", " ".join(hk)))
        novel = [t for t in hh if t.key not in ref_bag]
        novel_crit = [t for t in novel if t.crit and not is_neg(t)]
        ev.critical += min(NOVEL_CAP, len(novel_crit))
        if len(hh) >= 4 and len(novel) >= len(hh) / 2:
            ev.critical += 1
            ev.examples.append(("invented passage", "", " ".join(hk)))
        else:
            major("inserted words" if novel or len(hh) < K else "repeated text", [], hk)

    # Critical: ordered LCS after undoing tile moves (output tokens travel with the preceding tile).
    by_ref = sorted(tiles, key=lambda t: t[0])
    rank = {t: r for r, t in enumerate(by_ref)}
    by_h = sorted(tiles, key=lambda t: t[1])
    hkey: list[int] = []
    x = -1
    for j in range(n_h):
        while x + 1 < len(by_h) and by_h[x + 1][1] <= j:
            x += 1
        hkey.append(rank[by_h[x]] if x >= 0 else -1)
    h_prime = sorted(range(n_h), key=lambda j: (hkey[j], j))
    c_r = [t.key for t in rt if t.crit]
    c_h: list[str] = []
    for j in h_prime:
        if j in h_override:
            c_h.extend(h_override[j])
        elif ht[j].crit:
            c_h.append(h_keys[j])
    critical = _f1(_lcs_len(c_r, c_h), len(c_r), len(c_h))

    # Integrity: a reference sentence is intact unless foreign matched output (or a backward jump) splits it.
    # A bare number at either end (margin line number, list number, footnote marker) may sit elsewhere.
    h_matched = [0] * n_h
    for j in r2h:
        if j >= 0:
            h_matched[j] = 1
    pref = [0]
    for v in h_matched:
        pref.append(pref[-1] + v)
    sent_w = intact_w = 0
    cur: list[int] = []
    for i, tk in enumerate(rt):
        if r2h[i] >= 0:
            cur.append(i)
        if tk.end:
            if len(cur) >= 2:
                lo, hi = 0, len(cur)
                while lo < hi and _PURE_NUM.search(rt[cur[lo]].key):
                    lo += 1
                while hi > lo and _PURE_NUM.search(rt[cur[hi - 1]].key):
                    hi -= 1
                if hi - lo > 3:  # one stray word at either end is an alignment seam, not a cut
                    if not _in_order(r2h[cur[lo]], r2h[cur[lo + 1]], pref):
                        lo += 1
                    if not _in_order(r2h[cur[hi - 2]], r2h[cur[hi - 1]], pref):
                        hi -= 1
                ok = all(_in_order(r2h[cur[q - 1]], r2h[cur[q]], pref) for q in range(lo + 1, hi))
                sent_w += len(cur)
                intact_w += len(cur) if ok else 0
            cur = []

    # Order: matched weight outside the heaviest chain of tiles / displaced tokens.
    items = [(j, i, n) for i, j, n in tiles] + [(r2h[i], i, 1) for i in range(n_r) if how[i] == "disp"]
    items.sort(key=lambda it: it[0])
    chain = _heaviest_chain(items) if items else 0
    out_w = sum(it[2] for it in items) - chain
    moved_blocks = _moved_runs([it for it in items if it[2] >= K])

    m_r, m_h = matched + credit_r, matched + credit_h
    case_m = [i for i in range(n_r) if r2h[i] >= 0 and not rt[i].crit]
    words = (m_r + m_h) / (n_r + n_h) if n_r + n_h else None
    integrity = intact_w / sent_w if sent_w else None
    order = 1 - out_w / matched if matched else None
    n_crit = max(len(c_r), len(c_h))

    # Swapped values: a number matched only as displaced while its own slot (between its in-place
    # neighbours) holds another displaced number ("rose 12% ... rose 15%" written with 15 and 12).
    for i in range(1, n_r - 1):
        if how[i] != "disp" or not rt[i].crit or not _DIGIT.search(r_keys[i]):
            continue
        if how[i - 1] in (None, "disp") or how[i + 1] in (None, "disp"):
            continue
        lo, hi = r2h[i - 1], r2h[i + 1]
        if not 0 < hi - lo <= SWAP_SLOT + 1:
            continue
        for y in range(lo + 1, hi):
            if y != r2h[i] and ht[y].crit and _DIGIT.search(h_keys[y]) and h_keys[y] != r_keys[i]:
                src = next((q for q in range(n_r) if r2h[q] == y), -1)
                if src >= 0 and how[src] == "disp":
                    ev.critical += 1
                    ev.examples.append(("numbers swapped", r_keys[i], h_keys[y]))
                    break
    # A number matched in context whose sign changed (-5.2% read as 5.2%, or -5 as +5) is a wrong number. An
    # unclear dash on either side ("- 2", "-3-") and a displaced match (maybe another occurrence) are not charged.
    for i, j in enumerate(r2h):
        if how[i] in ("tile", "gap") and {rt[i].sign, ht[j].sign} in ({"-", ""}, {"-", "+"}):
            ev.critical += 1
            ev.examples.append(("sign changed", rt[i].sign + rt[i].surf, ht[j].sign + ht[j].surf))
    # A checkbox flipped between checked and unchecked on a token matched in context is a wrong value.
    for i, j in enumerate(r2h):
        if how[i] in ("tile", "gap") and rt[i].box and ht[j].box and rt[i].box != ht[j].box:
            ev.critical += 1
            ev.examples.append(("checkbox flipped", rt[i].box + " " + rt[i].surf, ht[j].box + " " + ht[j].surf))
    ev.moved = max(0, moved_blocks)
    if ev.moved:
        ev.examples.append(("blocks out of order", str(ev.moved), ""))
    fmt = glyph = 0
    for i, j in enumerate(r2h):
        if j < 0:
            continue
        if rt[i].crit and rt[i].surf != ht[j].surf:
            fmt += 1
        elif not rt[i].crit and _GREEK_CYR.search(ht[j].surf) and not _GREEK_CYR.search(rt[i].surf):
            glyph += 1
    if glyph:
        ev.minor += glyph
        ev.examples.append(("look-alike characters", str(glyph), ""))
    if fmt:
        ev.minor += fmt
        ev.examples.append(("number written differently", str(fmt), ""))
    ends = sum(1 for t in rt if t.end) or 1
    ev.broken = math.floor((sent_w - intact_w) / sent_w * ends + 0.5) if sent_w else 0
    if ev.broken:
        ev.examples.append(("sentences cut", str(ev.broken), ""))

    prop = next_proportional(words, critical, integrity, order, n_crit)
    floor = next_floor(ev, n_r)
    # When a lane loss sets the score below the floor, name it, so every lost point has an explanation.
    if prop < floor:
        if order is not None and order < 1 and not ev.moved:
            ev.examples.append(("reading order", f"{1 - order:.0%} of matched text out of order", ""))
        if critical is not None and critical < 1 and not ev.critical:
            ev.examples.append(("numbers out of sequence", f"critical-token F1 {critical:.2f}", ""))
    return TextNextResult(
        score=min(prop, floor),
        proportional=prop,
        floor=floor,
        words=words,
        critical=critical,
        integrity=integrity,
        order=order,
        recall=m_r / n_r if n_r else None,
        precision=m_h / n_h if n_h else None,
        case=(sum(1 for i in case_m if rt[i].surf == ht[r2h[i]].surf) / len(case_m)) if case_m else None,
        displaced=sum(1 for w in how if w == "disp"),
        n_crit=n_crit,
        n_ref=n_r,
        n_hyp=n_h,
        events=ev,
    )


# ---------------------------------------------------------------------------
# Hypothesis assembly (headers and footers count as content)
# ---------------------------------------------------------------------------

_NZ = _rx(r"[^\p{L}\p{N}]+")
SECTION_SUBSTRING_MIN = 12


def _nz(x: str) -> str:
    return str(_NZ.sub("", x)).lower()


def with_page_sections(markdown: str, headers: Sequence[str | None], footers: Sequence[str | None]) -> str:
    """Add the parser's page header/footer text that the page Markdown does not already contain.

    A section of at least ``SECTION_SUBSTRING_MIN`` normalized characters is present when its first 40
    appear anywhere in the page; a shorter one (a page number, "Page 3") only when a whole line of the
    page reads the same, so a footer "12" is not taken for the "12" of "12 months".
    """
    body = _nz(markdown)
    body_lines = {_nz(line) for line in md_plain(markdown).split("\n")}

    def present(x: str) -> bool:
        nz = _nz(x)
        return nz in body_lines if len(nz) < SECTION_SUBSTRING_MIN else nz[:40] in body

    def missing(parts: Sequence[str | None]) -> list[str]:
        return [x for x in parts if x and _js_trim(x) and not present(x)]

    head = "\n".join(missing(headers))
    foot = "\n".join(missing(footers))
    return "\n\n".join(p for p in (head, markdown, foot) if p)


def reference_markdown(expected_markdown: str | None, rules: Sequence[Any] | None) -> str | None:
    """The page reference: a text rule's ``original_md`` when present, else the expected Markdown."""
    for rule in rules or ():
        if getattr(rule, "type", None) == "unexpected_sentence_percent":
            original = getattr(rule, "original_md", None)
            if isinstance(original, str) and original:
                return original
    return expected_markdown


_HTML_TABLE = _rx(r"<table[\s\S]*?</table>", _I)
_PIPE_ROW = _rx(r"^[ \t]*\|.*\|[ \t]*$", regex.MULTILINE)


def text_next_applies(reference: str, rules: Sequence[Any] | None, metadata: dict[str, Any] | None) -> bool:
    """Whether the page reference is running text that text_next can score.

    Chart pages (``chart_tables`` metadata or ``chart_data_point`` rules) and table-only references (no
    word outside HTML / pipe tables) are scored by chart_next / table_next: a chart's data or a table read
    in another layout (transposed, row- vs column-major, prose) is not a text error, and a table-only
    reference against a whole-page output reads every paragraph as inserted text.
    """
    if "chart_tables" in (metadata or {}):
        return False
    if any(getattr(r, "type", None) == "chart_data_point" for r in rules or ()):
        return False
    prose = _PIPE_ROW.sub("", _HTML_TABLE.sub("", reference))
    return bool(tokenize(prose)[0])


def output_hypothesis(output: ParseOutput) -> str:
    """Output Markdown with the parser's page headers/footers that it does not already contain.

    Multi-page outputs are assembled page by page so a header lands at the top of its own page;
    single-page outputs use the document Markdown.
    """
    layouts = output.layout_pages or []
    if len(output.pages) > 1 and layouts:
        by_number = {lp.page_number: lp for lp in layouts}
        parts = []
        for page in output.pages:
            lp = by_number.get(page.page_index + 1)
            if lp is None:
                parts.append(page.markdown)
            else:
                parts.append(with_page_sections(page.markdown, [lp.page_header_markdown], [lp.page_footer_markdown]))
        return "\n\n".join(p for p in parts if p)
    return with_page_sections(
        output.markdown or "",
        [lp.page_header_markdown for lp in layouts],
        [lp.page_footer_markdown for lp in layouts],
    )


def text_next_metrics(reference: str, output: ParseOutput) -> list[MetricValue]:
    """``text_next`` headline plus its lane diagnostics as metric values."""
    res = score_text_next(reference, output_hypothesis(output))
    meta = res.metadata()
    metrics = [MetricValue(metric_name="text_next", value=res.score, metadata=meta)]
    for lane in ("words", "critical", "integrity", "order"):
        value = getattr(res, lane)
        if value is not None:
            metrics.append(MetricValue(metric_name=f"text_next_{lane}", value=float(value)))
    return metrics
