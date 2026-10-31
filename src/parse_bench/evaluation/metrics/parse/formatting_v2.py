"""Formatting metric (``formatting_v2``): semantic formatting with severity floors.

The reference Markdown and the output are both read into the same format-neutral model: a stream of
word tokens, each character carrying style bits (bold, italic, underline, strikethrough, superscript,
subscript, highlight, inline code), and line units with a role (paragraph line, heading with level,
standalone bold line, list item with depth and ordered flag, table row, code block). The syntax never
matters: ``**`` / ``__`` / ``<b>`` / ``<strong>``, ATX / setext / ``<hN>``, ``-`` / ``*`` / ``•`` /
``<li>``, ``<sup>`` / Unicode superscripts / ``^`` / LaTeX ``^{}`` all read the same.

Tokens are aligned move-tolerantly (greedy string tiling, LCS inside the gaps, then unique leftovers, then
the words of still unlocated formatted items, joining a word split by a style change and allowing one
typo); formatting is compared on aligned text. Invented words stay text_next's business; a formatted item
whose text is missing is lost with it (deleting struck text must not beat keeping it plain).

Atoms: every reference style span (or superscript / subscript token), heading, list item and code block,
plus every formatting the output adds on aligned text. Each atom that is lost, missing, added or changed
triggers one event with a severity class:

* critical (0.10 floor, atom loss 1.0): strikethrough lost or added (deleted text reads as live text or
  the reverse); a superscript / subscript glued to a digit lost or added (10^9 read as 109, 2^n as 2n).
* major (0.05, 0.5): heading turned into body text; body text turned into a heading; headings of different
  levels merged into one line; list items merged onto one line; list nesting flattened; heading hierarchy
  inverted or off by 2+ ranks; underline on a redline page (insertion marks).
* minor (0.01, capped at 0.15; 0.2): bold / italic / underline / highlight / inline code lost, added,
  shifted or swapped; footnote-marker superscripts and chemistry subscripts; heading shown as a bold
  line; heading level off by one rank; list markers lost; ordered vs unordered; spurious list or nesting.
* harmless (not an atom): ordinal suffixes after a digit (1st, 2e) and TM / SM marks, a uniform heading-level
  shift, any syntax.

Headline: ``min(proportional, floor)``. Proportional = 1 - sum(atom losses) / (atoms + PRIOR_ATOMS).
Floor = 1 - 0.10 * fix(critical) - 0.05 * fix(major) - min(0.15, 0.01 * minor) where the first event
counts in full and later ones scale by min(1, FLOOR_K / atoms). A minor or major event kind that hits
every instance of its kind on the page (3+) is a systematic quirk, charged once. When less than half of
the reference's formatted items can be located in the output, the score is capped at 2 x coverage.

Regexes are written so Python ``re`` and JavaScript (``u`` flag) behave the same: explicit character
classes instead of ``\\s`` / ``\\w`` / ``\\d``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from parse_bench.schemas.evaluation import MetricValue

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FLOOR = {"critical": 0.10, "major": 0.05, "minor": 0.01, "minor_cap": 0.15}
LOSS = {"critical": 1.0, "major": 0.5, "minor": 0.2}
PRIOR_ATOMS = 3  # smoothing: one lost minor atom on a page with a single atom costs 0.2 / 4
FLOOR_K = 10  # events after the first scale by min(1, FLOOR_K / atoms)
SYSTEMATIC_MIN = 3  # a minor/major kind hitting every instance (at least this many) is charged once
COVERAGE_FREE = 0.5  # below this share of located reference atoms the score is capped at coverage / 0.5
TILE_K = 3
LCS_CELL_LIMIT = 4_000_000
LABEL_MAX_TOKENS = 20  # a standalone fully bold line up to this many tokens is a bold label (pseudo heading)
SHORT_LINE_TOKENS = 12  # a false heading made of a whole short line is minor, of body text major

B, I, U, S, SUP, SUB, M, C = 1, 2, 4, 8, 16, 32, 64, 128  # noqa: E741 - style bits, named as the tags
F = 256  # inside an inline formula ($..$): a context bit, not a scored style
KIND_NAMES = {
    B: "bold",
    I: "italic",
    U: "underline",
    S: "strikethrough",
    SUP: "superscript",
    SUB: "subscript",
    M: "highlight",
    C: "inline code",
}
SPAN_KINDS = (B, I, U, S, M, C)
EMPHASIS = B | I | U | M
WORD_SPLIT = B | I | U | S | M | C
LANE_OF = {"inline": "inline", "heading": "headings", "list": "lists"}

# Private-use sentinels: open/close of each style, a neutral non-word placeholder and a cell break.
_TRACKED = (B, I, U, S, SUP, SUB, M, C, F)
_OPEN = {k: chr(0xE100 + 2 * n) for n, k in enumerate(_TRACKED)}
_CLOSE = {k: chr(0xE101 + 2 * n) for n, k in enumerate(_TRACKED)}
_SENT = {**{v: (k, 1) for k, v in _OPEN.items()}, **{v: (k, -1) for k, v in _CLOSE.items()}}
NEUTRAL = "\ue0ff"

_SUP_CHARS = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
_SUB_CHARS = set("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ")
_CJK = re.compile("[\u3005-\u3007\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u0e00-\u0e7f]")
_ENTITIES = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": " ",
    "ndash": "–",
    "mdash": "—",
    "hellip": "…",
    "rsquo": "’",
    "lsquo": "‘",
    "ldquo": "“",
    "rdquo": "”",
    "laquo": "«",
    "raquo": "»",
    "bull": "•",
    "middot": "·",
    "times": "×",
    "minus": "−",
    "deg": "°",
    "copy": "©",
    "reg": "®",
    "trade": "™",
    "sup1": "¹",
    "sup2": "²",
    "sup3": "³",
    "euro": "€",
    "pound": "£",
    "sect": "§",
    "para": "¶",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    """A line unit: paragraph line, heading, bold label, list item, table row or code block."""

    role: str  # "p" | "h" | "label" | "item" | "row" | "code"
    level: int = 0  # heading level 1-6, 7 for a bold label
    depth: int = 0  # list depth
    ordered: bool = False
    group: int = -1  # list group / code block id
    block: int = 0  # paragraph block id (blank-line separated)
    text: str = ""
    toks: list[int] = field(default_factory=list)


@dataclass
class Tok:
    key: str
    bits: list[int]  # style bits per key character
    unit: int
    lead: int = 0  # scripts whose run starts with a sign glued to a preceding digit (10<sup>-3</sup>)

    def has(self, k: int) -> bool:
        """Token-level style: at least half of its characters carry it (any for scripts)."""
        n = sum(1 for b in self.bits if b & k)
        if k in (SUP, SUB):
            return n > 0
        return 2 * n >= len(self.bits) and n > 0

    def mask(self, k: int) -> str:
        return "".join("1" if b & k else "0" for b in self.bits)


@dataclass
class Doc:
    toks: list[Tok]
    units: list[Unit]


# ---------------------------------------------------------------------------
# Inline parsing
# ---------------------------------------------------------------------------

_NS = "[^ \\t\\n]"  # a non-space character (explicit: identical in Python and JavaScript)


def _rx(p: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(p, flags)


_CODE_SPAN = _rx(r"(`+)([^`][\s\S]*?)\1(?!`)")
_ESCAPE = _rx(r"\\([!-/:-@\[-`{-~])")
_INLINE_MATH = _rx(r"\\\(([\s\S]+?)\\\)|\$([^$\n]+?)\$")
_MATH_TEXTCMD = _rx(r"\\(?:mathrm|text|textbf|textit|mathbf|mathit|operatorname|rm|bf|it)\{([^{}]*)\}")
_MATH_SUPB = _rx(r"\^\{([^{}]*)\}")
_MATH_SUP1 = _rx(r"\^([A-Za-z0-9])")
_MATH_SUBB = _rx(r"_\{([^{}]*)\}")
_MATH_SUB1 = _rx(r"_([A-Za-z0-9])")
_MATH_CMD = _rx(r"\\[A-Za-z]+|\\.")
_TRIPLE = _rx(r"(\*\*\*|___)(?=" + _NS + r")([\s\S]+?)(?<=" + _NS + r")\1")
_BOLD_STAR = _rx(r"\*\*(?=" + _NS + r")([\s\S]+?)(?<=" + _NS + r")\*\*")
_BOLD_UND = _rx(r"(?<![A-Za-z0-9_])__(?=" + _NS + r")([\s\S]+?)(?<=" + _NS + r")__(?![A-Za-z0-9_])")
_STRIKE = _rx(r"~~(?=[^ \t\n~])([\s\S]+?)(?<=[^ \t\n~])~~")
_MARK = _rx(r"==(?=[^ \t\n=])([\s\S]+?)(?<=[^ \t\n=])==")
_ITAL_STAR = _rx(r"\*(?=[^ \t\n*])([\s\S]+?)(?<=[^ \t\n*])\*")
_ITAL_UND = _rx(r"(?<![A-Za-z0-9_])_(?=[^ \t\n_])([\s\S]+?)(?<=[^ \t\n_])_(?![A-Za-z0-9_])")
_PANDOC_SUP = _rx(r"\^([^ \t\n^]+?)\^")
_CARET_SUPB = _rx(r"\^\{([^{}\n]*)\}")
_CARET_SUP = _rx(r"(?<=[A-Za-z0-9)\]])\^([-+−]?[A-Za-z0-9]+)")
_PANDOC_SUB = _rx(r"(?<!~)~([^ \t\n~]+?)~(?!~)")
_TAG = _rx(r"<(/?)([A-Za-z][A-Za-z0-9]*)([^<>]*?)(/?)>")
_ENTITY = _rx(r"&(#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,7});")
_TAG_BITS = {
    "b": B,
    "strong": B,
    "i": I,
    "em": I,
    "u": U,
    "ins": U,
    "s": S,
    "del": S,
    "strike": S,
    "sup": SUP,
    "sub": SUB,
    "mark": M,
    "code": C,
    "kbd": C,
    "tt": C,
}


def _wrap(k: int, s: str) -> str:
    return _OPEN[k] + s + _CLOSE[k]


def _neutralize(s: str) -> str:
    return re.sub(r"[*_~^=<>$\\`]", NEUTRAL, s)


def _math(body: str) -> str:
    body = _MATH_TEXTCMD.sub(lambda m: m.group(1), body)
    body = _MATH_SUPB.sub(lambda m: _wrap(SUP, m.group(1)), body)
    body = _MATH_SUP1.sub(lambda m: _wrap(SUP, m.group(1)), body)
    body = _MATH_SUBB.sub(lambda m: _wrap(SUB, m.group(1)), body)
    body = _MATH_SUB1.sub(lambda m: _wrap(SUB, m.group(1)), body)
    body = _MATH_CMD.sub(" ", body)
    return body.replace("{", "").replace("}", "")


def _inline_math(m: re.Match[str]) -> str:
    """``$..$`` / ``\\(..\\)`` with TeX syntax inside; a plain ``$5 .. $10`` stays money."""
    body = m.group(1) if m.group(1) is not None else m.group(2)
    if m.group(1) is None and not re.search(r"[\\^_{}]", body):
        return m.group(0)
    return _wrap(F, _math(body))


def _markdown_inline(text: str) -> str:
    """Markdown emphasis and script syntaxes -> sentinels (HTML tags stay for the walker)."""
    text = _CODE_SPAN.sub(lambda m: _wrap(C, _neutralize(m.group(2))), text)
    text = _ESCAPE.sub(NEUTRAL, text)
    text = _INLINE_MATH.sub(_inline_math, text)
    text = _TRIPLE.sub(lambda m: _OPEN[B] + _OPEN[I] + m.group(2) + _CLOSE[I] + _CLOSE[B], text)
    text = _BOLD_STAR.sub(lambda m: _wrap(B, m.group(1)), text)
    text = _BOLD_UND.sub(lambda m: _wrap(B, m.group(1)), text)
    text = _STRIKE.sub(lambda m: _wrap(S, m.group(1)), text)
    text = _MARK.sub(lambda m: _wrap(M, m.group(1)), text)
    text = _ITAL_STAR.sub(lambda m: _wrap(I, m.group(1)), text)
    text = _ITAL_UND.sub(lambda m: _wrap(I, m.group(1)), text)
    text = _PANDOC_SUP.sub(lambda m: _wrap(SUP, m.group(1)), text)
    text = _CARET_SUPB.sub(lambda m: _wrap(SUP, m.group(1)), text)
    text = _CARET_SUP.sub(lambda m: _wrap(SUP, m.group(1)), text)
    return _PANDOC_SUB.sub(lambda m: _wrap(SUB, m.group(1)), text)


def _style_bits(name: str, attrs: str) -> int:
    bits = _TAG_BITS.get(name, 0)
    a = attrs.lower()
    if "line-through" in a:
        bits |= S
    if "underline" in a:
        bits |= U
    if re.search(r"font-weight[ \t]*:[ \t]*(?:bold|[6-9]00)", a):
        bits |= B
    if re.search(r"font-style[ \t]*:[ \t]*italic", a):
        bits |= I
    if re.search(r"vertical-align[ \t]*:[ \t]*super", a):
        bits |= SUP
    if re.search(r"vertical-align[ \t]*:[ \t]*sub", a):
        bits |= SUB
    return bits


def _entity(m: re.Match[str]) -> str:
    e = m.group(1)
    if e[0] == "#":
        try:
            cp = int(e[2:], 16) if e[1] in "xX" else int(e[1:])
            return chr(cp) if 0 < cp < 0x110000 and not 0xD800 <= cp < 0xE000 else " "
        except ValueError:
            return " "
    return _ENTITIES.get(e, m.group(0))


def styled_chars(text: str) -> list[tuple[str, int]]:
    """Block text -> characters with their style bits (newlines kept)."""
    text = _markdown_inline(text)
    out: list[tuple[str, int]] = []
    count = dict.fromkeys(_TRACKED, 0)
    stack: list[tuple[str, int]] = []

    def bump(bits: int, d: int) -> None:
        for k in _TRACKED:
            if bits & k:
                count[k] = max(0, count[k] + d)

    def emit(s: str) -> None:
        cur = sum(k for k in _TRACKED if count[k] > 0)
        for ch in _ENTITY.sub(_entity, s):
            if ch in _SENT:
                k, d = _SENT[ch]
                count[k] = max(0, count[k] + d)
                cur = sum(k for k in _TRACKED if count[k] > 0)
            elif ch in _SUP_CHARS:
                out.append((ch, cur | SUP))
            elif ch in _SUB_CHARS:
                out.append((ch, cur | SUB))
            else:
                out.append((ch, cur))

    pos = 0
    for m in _TAG.finditer(text):
        emit(text[pos : m.start()])
        pos = m.end()
        closing, name, attrs, selfclose = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
        if selfclose:
            continue
        if closing:
            for x in range(len(stack) - 1, -1, -1):
                if stack[x][0] == name:
                    bump(stack[x][1], -1)
                    del stack[x]
                    break
        else:
            bits = _style_bits(name, attrs)
            stack.append((name, bits))
            bump(bits, 1)
    emit(text[pos:])
    return out


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _is_word(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LNM"


def _key_chars(ch: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", ch) if unicodedata.category(c) != "Mn").lower()


def _lead(chars: Sequence[tuple[str, int]], ci: int) -> int:
    """Script kinds whose run, starting at ``ci`` after a sign (10<sup>-3</sup>), is glued to a digit."""
    lead = 0
    for k in (SUP, SUB):
        x = ci - 1
        while x >= 0 and chars[x][0] in "-−+–" and chars[x][1] & k:
            x -= 1
        if x < ci - 1 and x >= 0 and chars[x][0] in "0123456789" and not chars[x][1] & k:
            lead |= k
    return lead


def tokens_of(chars: Sequence[tuple[str, int]], unit: int) -> list[Tok]:
    toks: list[Tok] = []
    key: list[str] = []
    bits: list[int] = []
    lead = 0

    def flush() -> None:
        if key:
            toks.append(Tok("".join(key), list(bits), unit, lead))
            key.clear()
            bits.clear()

    for ci, (ch, b) in enumerate(chars):
        if _CJK.match(ch):
            flush()
            k = _key_chars(ch)
            if k:
                toks.append(Tok(k, [b] * len(k), unit))
        elif _is_word(ch):
            kc = _key_chars(ch)
            if kc and key and (b ^ bits[-1]) & WORD_SPLIT:
                flush()  # <u>103</u>~~101~~: abutting styled runs are two words (scripts stay glued: 10<sup>9</sup>)
            if kc and not key:
                lead = _lead(chars, ci)
            for c in kc:
                key.append(c)
                bits.append(b)
        else:
            flush()
    flush()
    return toks


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------

_HTML_COMMENT = _rx(r"<!--[\s\S]*?-->")
_DISPLAY_MATH = _rx(r"\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]")
_IMAGE = _rx(r"!\[[^\]\n]*\]\([^)\n]*\)")
_LINK = _rx(r"\[([^\]\n]*)\]\((?:[^()\n]|\([^()\n]*\))*\)")
_HTML_HEADING = _rx(r"<h([1-6])(?![A-Za-z0-9_])[^>]*>([\s\S]*?)</h\1[ \t]*>", re.IGNORECASE)
_HTML_LIST_TAG = _rx(r"<(/?)(ul|ol|li)(?![A-Za-z0-9_])[^>]*>", re.IGNORECASE)
_TABLE = _rx(r"<table(?![A-Za-z0-9_])[^>]*>([\s\S]*?)</table[ \t]*>", re.IGNORECASE)
_ROW_END = _rx(r"</tr[ \t]*>", re.IGNORECASE)
_CELL_TAG = _rx(r"</?(?:td|th)(?![A-Za-z0-9_])[^>]*>", re.IGNORECASE)
_TABLE_TAG = _rx(r"</?(?:tr|thead|tbody|tfoot|caption|colgroup|col)(?![A-Za-z0-9_])[^>]*>", re.IGNORECASE)
_BR = _rx(r"<br[ \t]*/?>", re.IGNORECASE)
_BLOCK_TAG = _rx(
    r"</?(?:p|div|section|article|blockquote|figure|figcaption|header|footer|page_header|page_footer|page_number|hr)(?![A-Za-z0-9_])[^>]*>",
    re.IGNORECASE,
)
_FENCE = _rx(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^`\n]*)$")
_ATX = _rx(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_ATX_CLOSE = _rx(r"[ \t]+#+$")
_SETEXT = _rx(r"^ {0,3}(=+|-+)[ \t]*$")
_HR = _rx(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_QUOTE = _rx(r"^ {0,3}>[ \t]?")
_ITEM = _rx(r"^([ \t]*)([-*+•◦▪●○■‣∙]|[0-9]{1,3}[.)]|\([0-9]{1,3}\)|\([a-z]{1,4}\)|\([A-Z]\)|[a-z]\))[ \t]+(.*)$")
_PIPE_ROW = _rx(r"^[ \t]*\|.*\|[ \t]*$")
_PIPE_SEP = _rx(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$")


def _indent(s: str) -> int:
    n = 0
    for ch in s:
        n += 4 if ch == "\t" else 1
    return n


def _html_lists(md: str) -> str:
    out: list[str] = []
    stack: list[str] = []
    pos = 0
    for m in _HTML_LIST_TAG.finditer(md):
        out.append(md[pos : m.start()])
        pos = m.end()
        closing, name = m.group(1), m.group(2).lower()
        if name in ("ul", "ol"):
            if closing:
                if stack:
                    stack.pop()
            else:
                stack.append(name)
            out.append("\n")
        elif not closing:
            depth = max(0, len(stack) - 1)
            marker = "1. " if stack and stack[-1] == "ol" else "- "
            out.append("\n" + "  " * depth + marker)
        else:
            out.append("\n")
    out.append(md[pos:])
    return "".join(out)


def _tables(md: str) -> str:
    def one(m: re.Match[str]) -> str:
        body = _ROW_END.sub("\n", m.group(1).replace("\n", " "))
        body = _CELL_TAG.sub(" | ", body)
        body = _TABLE_TAG.sub(" ", body)
        rows = body.split("\n")
        return "\n\n" + "\n".join("|" + r + "|" for r in rows if r.strip(" \t|")) + "\n\n"

    return _TABLE.sub(one, md)


def _prepare(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = unicodedata.normalize("NFC", md)
    md = _HTML_COMMENT.sub(" ", md)
    md = _DISPLAY_MATH.sub("\n\n", md)
    md = _IMAGE.sub(" ", md)
    md = _LINK.sub(lambda m: m.group(1), md)
    md = _HTML_HEADING.sub(lambda m: "\n\n" + "#" * int(m.group(1)) + " " + m.group(2).replace("\n", " ") + "\n\n", md)
    md = _tables(md)
    md = _html_lists(md)
    md = _BR.sub("\n", md)
    return _BLOCK_TAG.sub("\n\n", md)


def parse(md: str) -> Doc:
    """Markdown / HTML page -> tokens and line units (format-neutral)."""
    lines = _prepare(md or "").split("\n")
    units: list[Unit] = []
    raw: list[list[str]] = []  # per unit, its raw lines
    block = 0
    group = 0
    stack: list[int] = []  # list indents
    in_list = False
    blank_before = True
    code: Unit | None = None
    code_fence = ""
    transparent = ""  # an outer ```markdown fence is read as Markdown
    para_open = False

    def new_unit(u: Unit, text: str) -> None:
        units.append(u)
        raw.append([text])

    for line in lines:
        fm = _FENCE.match(line)
        if code is not None:
            if fm and fm.group(1)[0] == code_fence[0] and len(fm.group(1)) >= len(code_fence) and not fm.group(2):
                code = None
                block += 1
                blank_before = True
            else:
                raw[-1].append(line)
            continue
        if fm:
            lang = fm.group(2).strip().lower()
            if transparent and not lang and fm.group(1)[0] == transparent[0]:
                transparent = ""
                continue
            if lang in ("markdown", "md"):
                transparent = fm.group(1)
                continue
            block += 1
            group += 1
            code = Unit("code", group=group, block=block)
            code_fence = fm.group(1)
            units.append(code)
            raw.append([])
            para_open = False
            in_list = False
            continue
        while _QUOTE.match(line):
            line = _QUOTE.sub("", line, count=1)
        if not line.strip(" \t"):
            blank_before = True
            para_open = False
            continue
        sm = _SETEXT.match(line)
        if sm and para_open and units and units[-1].role == "p":
            level = 1 if sm.group(1)[0] == "=" else 2
            first = len(units) - 1
            while first > 0 and units[first - 1].role == "p" and units[first - 1].block == units[-1].block:
                first -= 1
            text = " ".join(x for u in raw[first:] for x in u)
            del units[first:]
            del raw[first:]
            new_unit(Unit("h", level=level, block=block), text)
            block += 1
            para_open = False
            blank_before = False
            continue
        if _HR.match(line):
            blank_before = True
            para_open = False
            continue
        am = _ATX.match(line)
        if am:
            text = _ATX_CLOSE.sub("", am.group(2) or "").strip(" \t")
            block += 1
            if text and text.strip("#"):
                new_unit(Unit("h", level=len(am.group(1)), block=block), text)
            block += 1
            para_open = False
            in_list = False
            blank_before = False
            continue
        if _PIPE_ROW.match(line):
            if not _PIPE_SEP.match(line):
                if not para_open or units[-1].role != "row":
                    block += 1
                new_unit(Unit("row", block=block), line.strip(" \t").strip("|").replace("|", " " + NEUTRAL + " "))
                para_open = True
            in_list = False
            blank_before = False
            continue
        im = _ITEM.match(line)
        if im:
            ind = _indent(im.group(1))
            if not in_list:
                group += 1
                stack = [ind]
                in_list = True
            elif ind > stack[-1] + 1:
                stack.append(ind)
            else:
                while len(stack) > 1 and stack[-1] > ind + 1:
                    stack.pop()
            marker = im.group(2)
            ordered = marker not in "-*+•◦▪●○■‣∙"
            block += 1
            new_unit(Unit("item", depth=len(stack) - 1, ordered=ordered, group=group, block=block), im.group(3))
            para_open = True
            blank_before = False
            continue
        ind = _indent(line[: len(line) - len(line.lstrip(" \t"))])
        if in_list and units and units[-1].role == "item" and (not blank_before or ind >= 2):
            raw[-1].append(line.strip(" \t"))
            blank_before = False
            continue
        if blank_before or not para_open or units[-1].role not in ("p", "label"):
            block += 1
        in_list = False
        new_unit(Unit("p", block=block), line.strip(" \t"))
        para_open = True
        blank_before = False

    # Inline styles: parsed per paragraph block (emphasis may wrap lines), per unit otherwise.
    toks: list[Tok] = []
    x = 0
    while x < len(units):
        y = x + 1
        if units[x].role == "p":
            while y < len(units) and units[y].role == "p" and units[y].block == units[x].block:
                y += 1
        text = "\n".join(" ".join(raw[z]) if units[z].role != "code" else "\n".join(raw[z]) for z in range(x, y))
        if units[x].role == "code":
            chars = [(ch, 0) for ch in text]
            lines_chars = [chars]
        else:
            chars = styled_chars(text)
            lines_chars = [[]]
            for ch, b in chars:
                if ch == "\n" and len(lines_chars) < y - x:
                    lines_chars.append([])
                else:
                    lines_chars[-1].append((ch, b))
        for z in range(x, y):
            lc = lines_chars[z - x] if z - x < len(lines_chars) else []
            units[z].text = "".join(ch for ch, _ in lc).strip()
            for t in tokens_of(lc, z):
                units[z].toks.append(len(toks))
                toks.append(t)
        x = y
    for u in units:
        if u.role == "p" and 1 <= len(u.toks) <= LABEL_MAX_TOKENS and all(toks[t].has(B) for t in u.toks):
            u.role = "label"
            u.level = 7
    return Doc(toks, units)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def _tile(r: Sequence[str], h: Sequence[str], rm: list[bool], hm: list[bool], k: int) -> list[tuple[int, int, int]]:
    """Greedy string tiling, longest first; ties go to the closest relative position."""
    n_r, n_h = len(r), len(h)
    tiles: list[tuple[int, int, int]] = []
    while True:
        idx: dict[str, list[int]] = {}
        for j in range(n_h - k + 1):
            if not any(hm[j : j + k]):
                idx.setdefault("\u0001".join(h[j : j + k]), []).append(j)
        cands: list[tuple[int, float, int, int]] = []
        for i in range(n_r - k + 1):
            if any(rm[i : i + k]):
                continue
            js = idx.get("\u0001".join(r[i : i + k]))
            if not js:
                continue
            for j in js:
                if i and j and r[i - 1] == h[j - 1] and not rm[i - 1] and not hm[j - 1]:
                    continue
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
        for neg, _, i, j in cands:
            length = -neg
            if any(rm[i : i + length]) or any(hm[j : j + length]):
                if took:
                    break
                continue
            for x in range(length):
                rm[i + x] = hm[j + x] = True
            tiles.append((i, j, length))
            took = True
        if not took:
            return tiles


def _lcs_pairs(a: Sequence[str], b: Sequence[str]) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    if not n or not m or n * m > LCS_CELL_LIMIT:
        return []
    rows = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        cur, nxt, ai = rows[i], rows[i + 1], a[i]
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


def align(r: Sequence[str], h: Sequence[str], pinned: Sequence[tuple[int, int]] = ()) -> list[int]:
    """For each reference token, the aligned output token index or -1. ``pinned`` (ref, out) pairs are fixed
    first and anchor the gap alignment like tiles."""
    n, m = len(r), len(h)
    rm, hm = [False] * n, [False] * m
    match = [-1] * n
    for i, j in pinned:
        match[i] = j
        rm[i] = hm[j] = True
    for i, j, length in _tile(r, h, rm, hm, TILE_K):
        for x in range(length):
            match[i + x] = j + x
    # LCS inside the gaps between consecutive tiles (in reference order) whose output order agrees.
    anchors = [(-1, -1)] + sorted((i, match[i]) for i in range(n) if match[i] >= 0) + [(n, m)]
    for (i0, j0), (i1, j1) in zip(anchors, anchors[1:], strict=False):
        if i1 - i0 > 1 and j1 - j0 > 1:
            ri = [i for i in range(i0 + 1, i1) if not rm[i]]
            hj = [j for j in range(j0 + 1, j1) if not hm[j]]
            for a, b in _lcs_pairs([r[i] for i in ri], [h[j] for j in hj]):
                match[ri[a]] = hj[b]
                rm[ri[a]] = hm[hj[b]] = True
    for i, j, length in _tile(r, h, rm, hm, 2):
        for x in range(length):
            match[i + x] = j + x
    # Unique leftovers: a key left exactly once on each side.
    left_r: dict[str, list[int]] = {}
    left_h: dict[str, list[int]] = {}
    for i in range(n):
        if not rm[i]:
            left_r.setdefault(r[i], []).append(i)
    for j in range(m):
        if not hm[j]:
            left_h.setdefault(h[j], []).append(j)
    for key, ii in left_r.items():
        jj = left_h.get(key)
        if len(ii) == 1 and jj is not None and len(jj) == 1:
            match[ii[0]] = jj[0]
    return match


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class Event:
    kind: str
    cls: str  # critical | major | minor
    family: str  # inline | heading | list
    ref: str
    out: str
    group: str  # systematic-collapse group (event kind + instance kind)
    charged: bool = True


@dataclass
class FormattingResult:
    score: float
    proportional: float
    floor: float
    coverage: float
    lanes: dict[str, float | None]
    atoms: int
    events: list[Event]
    n_ref_tokens: int
    n_out_tokens: int
    decoration: dict[str, dict[str, int]] = field(default_factory=dict)  # kind -> tp, fp, fn

    def counts(self) -> dict[str, int]:
        c = {"critical": 0, "major": 0, "minor": 0}
        for e in self.events:
            if e.charged:
                c[e.cls] += 1
        return c

    def metadata(self, max_examples: int = 25) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for e in self.events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        return {
            "proportional": self.proportional,
            "floor": self.floor,
            "coverage": self.coverage,
            "atoms": self.atoms,
            "lanes": self.lanes,
            "decoration": self.decoration,
            "events": self.counts(),
            "event_kinds": kinds,
            "uncharged_repeats": sum(1 for e in self.events if not e.charged),
            "event_examples": [
                {
                    "kind": e.kind,
                    "severity": e.cls,
                    "reference": e.ref[:160],
                    "output": e.out[:160],
                    "charged": e.charged,
                }
                for e in self.events[:max_examples]
            ],
            "n_ref_tokens": self.n_ref_tokens,
            "n_out_tokens": self.n_out_tokens,
        }


def _numeric(s: str) -> bool:
    return any(c in "0123456789" for c in s) and all(c in "0123456789" for c in s)


_ORDINALS = frozenset(("st", "nd", "rd", "th", "e", "er", "re", "eme", "o", "a", "os", "as"))
_MARKS = frozenset(("tm", "sm"))


def _script_class(t: Tok, k: int) -> str | None:
    """Severity of losing / adding the superscript (or subscript) chars of a token; None = harmless.

    Digits always count. Letters count when glued to a base in the same token (2^n flattened to 2n is another
    number, x_i flattened to xi another symbol), except an ordinal suffix after a digit (1st, 2e) or a TM / SM
    mark; a standalone letter (footnote a) and letters inside an inline formula (its own metric) are harmless."""
    chars = "".join(t.key[x] for x in range(len(t.key)) if t.bits[x] & k)
    first = next((x for x in range(len(t.key)) if t.bits[x] & k), 0)
    if not chars:
        return None
    if not _numeric(chars):
        if not first or chars in _MARKS or t.has(F):
            return None
        if chars in _ORDINALS and t.key[first - 1] in "0123456789" and t.key.endswith(chars):
            return None
    if t.lead & k and t.bits[0] & k:
        return "critical"
    for x in range(1, len(t.key)):
        if t.bits[x] & k and not t.bits[x - 1] & k and t.key[x - 1] in "0123456789":
            return "critical"
    return "minor"


def _excerpt(doc: Doc, toks: Sequence[int]) -> str:
    return " ".join(doc.toks[t].key for t in toks[:12]) + (" …" if len(toks) > 12 else "")


def _majority(units: Sequence[int]) -> int:
    best, best_n = -1, 0
    seen: dict[int, int] = {}
    for u in units:
        seen[u] = seen.get(u, 0) + 1
    for u in units:  # first occurrence order breaks ties
        if seen[u] > best_n:
            best, best_n = u, seen[u]
    return best


def _head(u: Unit) -> str:
    return ("[bold line] " if u.level == 7 else f"[h{u.level}] ") + u.text[:60]


def _rank(levels: Sequence[int]) -> dict[int, int]:
    return {lv: x for x, lv in enumerate(sorted(set(levels)))}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


@dataclass(frozen=True)
class Scope:
    """What a reference can judge. A full reference Markdown judges everything; a reference built from rules
    judges only the kinds its rules name, and a ``heading_structure`` rule lists every heading of the page."""

    kinds: int = B | I | U | S | SUP | SUB | M | C
    headings: bool = True
    lists: bool = True
    complete_headings: bool = False
    # A full-page decoration inventory (metadata.decorations): every strikethrough or underline run on the page
    # is in the reference, so the proportional part is the token F1 of the scoped decoration (see score_docs).
    decoration: bool = False


FULL = Scope()


def _pin_heading_lines(ref: Doc, out: Doc, rk: Sequence[str], hk: Sequence[str]) -> list[tuple[int, int]]:
    """(ref, out) token pairs tying each reference heading to an output heading or bold-label line with the same
    token keys, for ``align`` to fix before it runs.

    Used when the reference is a sparse heading list (rule mode): its lines carry no position, so the token
    alignment can bind a heading to the same words in a body sentence or table row and charge the real heading
    twice (lost + false). Equal-text lines pair in order of appearance.
    """
    lines: dict[tuple[str, ...], list[int]] = {}
    for y, o in enumerate(out.units):
        if o.role in ("h", "label") and o.toks:
            lines.setdefault(tuple(hk[j] for j in o.toks), []).append(y)
    pairs: list[tuple[int, int]] = []
    for u in ref.units:
        ys = lines.get(tuple(rk[t] for t in u.toks)) if u.role == "h" and u.toks else None
        if not ys:
            continue
        pairs.extend(zip(u.toks, out.units[ys.pop(0)].toks, strict=True))
    return pairs


def _spans(doc: Doc, k: int) -> list[list[int]]:
    """Runs of consecutive tokens of one unit carrying style ``k`` (bold inside a heading is the heading's)."""
    spans: list[list[int]] = []
    for t, tok in enumerate(doc.toks):
        ok = tok.has(k) and not (k == B and doc.units[tok.unit].role == "h")
        if ok and spans and spans[-1][-1] == t - 1 and doc.toks[t - 1].unit == tok.unit:
            spans[-1].append(t)
        elif ok:
            spans.append([t])
    return spans


def _typo_distance(a: str, b: str, cap: int) -> int:
    """Edit distance counting an adjacent transposition as one edit; any value above ``cap`` is cap + 1."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for x in range(1, len(a) + 1):
        cur = [x] + [0] * len(b)
        for y in range(1, len(b) + 1):
            cur[y] = min(prev[y] + 1, cur[y - 1] + 1, prev[y - 1] + (a[x - 1] != b[y - 1]))
            if x > 1 and y > 1 and a[x - 1] == b[y - 2] and a[x - 2] == b[y - 1]:
                cur[y] = min(cur[y], prev2[y - 2] + 1)
        prev2, prev = prev, cur
    return min(prev[-1], cap + 1)


def _unlocated_items(ref: Doc, match: Sequence[int], scope: Scope) -> list[int]:
    """Reference tokens of the formatted items alignment could not locate: spans, scripts and code blocks with no
    aligned token, headings and list items with too few to map their line. These are charged as missing."""
    need: set[int] = set()
    for k in SPAN_KINDS:
        if k & scope.kinds:
            need.update(t for sp in _spans(ref, k) if all(match[t] < 0 for t in sp) for t in sp)
    for t, tok in enumerate(ref.toks):
        if match[t] < 0 and any(tok.has(k) and k & scope.kinds and _script_class(tok, k) for k in (SUP, SUB)):
            need.add(t)
    for u in ref.units:
        n_al = sum(1 for t in u.toks if match[t] >= 0)
        if (
            (u.role == "h" and scope.headings and 2 * n_al < len(u.toks))
            or (u.role == "item" and scope.lists and 2 * n_al < len(u.toks))
            or (u.role == "code" and C & scope.kinds and not n_al)
        ):
            need.update(t for t in u.toks if match[t] < 0)
    return sorted(need)


def _relocate(ref: Doc, out: Doc, match: list[int], scope: Scope) -> set[int]:
    """Locate the tokens of unlocated formatted items on output tokens alignment left unused; returns the
    reference tokens located through a respelling.

    Each token is searched between the output positions of its nearest aligned neighbours (anywhere when they
    are out of order, or in rule mode where the reference has no page order). An equal key wins, or consecutive
    output tokens of one line whose keys join into it: a style change inside a word (S<i>tandard</i>) or a
    space before a script (10 <sup>9</sup>) splits it, and the pieces are merged back into one output token
    (``out.toks`` is replaced in place) so its style is judged as a whole. Else a word of letters with one typo
    (two from 8 letters), the closest such one, so a respelled bold word is judged as bold rather than as a
    missing span, and the respelling costs a minor event instead of the page.
    """
    n, m = len(ref.toks), len(out.toks)
    used = [False] * m
    for j in match:
        if j >= 0:
            used[j] = True
    if all(used):
        return set()
    nxt = [m] * (n + 1)
    for i in range(n - 1, -1, -1):
        nxt[i] = match[i] if match[i] >= 0 else nxt[i + 1]
    need = _unlocated_items(ref, match, scope)
    respelled: set[int] = set()
    prev_j, x = -1, 0
    for i in need:
        while x < i:
            prev_j = match[x] if match[x] >= 0 else prev_j
            x += 1
        lo, hi = prev_j, nxt[i + 1]
        cands = range(lo + 1, hi) if scope == FULL and lo < hi else range(m)
        key = ref.toks[i].key
        fuzzy = len(key) >= 4 and key.isalpha()
        cap = 2 if len(key) >= 8 else 1
        end = hi if scope == FULL and lo < hi else m
        best: tuple[int, int, int, int] | None = None  # (distance, offset, first, last output token)
        for j in cands:
            if used[j]:
                continue
            h, z = out.toks[j].key, j
            while key.startswith(h) and h != key and z + 1 < end and not used[z + 1]:
                if out.toks[z + 1].unit != out.toks[j].unit:
                    break
                z += 1
                h += out.toks[z].key
            if h != key:
                h, z = out.toks[j].key, j
            d = 0 if h == key else (_typo_distance(key, h, cap) if fuzzy and h.isalpha() and len(h) >= 4 else cap + 1)
            if d <= cap and (best is None or (d, abs(j - lo)) < best[:2]):
                best = (d, abs(j - lo), j, z)
        if best is not None:
            d, _, j, z = best
            if z > j:
                pieces = out.toks[j : z + 1]
                bits = [b for p in pieces for b in p.bits]
                out.toks[j] = Tok(key, bits, pieces[0].unit, pieces[0].lead)
            match[i] = j
            for y in range(j, z + 1):
                used[y] = True
            if d:
                respelled.add(i)
    return respelled


def score_docs(ref: Doc, out: Doc, scope: Scope = FULL, skip: frozenset[int] = frozenset()) -> FormattingResult:  # noqa: C901 - one scoring pass
    rk = [t.key for t in ref.toks]
    hk = [t.key for t in out.toks]
    pinned = _pin_heading_lines(ref, out, rk, hk) if scope.complete_headings else []
    match = align(rk, hk, pinned)
    out = Doc(list(out.toks), out.units)  # _relocate may merge split output tokens
    respelled = set() if scope.decoration else _relocate(ref, out, match, scope)
    back = [-1] * len(hk)
    for i, j in enumerate(match):
        if j >= 0:
            back[j] = i
    events: list[Event] = []
    atoms: dict[str, list[float]] = {"inline": [], "heading": [], "list": []}  # per atom, its loss
    located = 0
    ref_atoms = 0
    redline = any(t.has(S) for t in ref.toks)

    def in_heading(doc: Doc, t: int) -> bool:
        return doc.units[doc.toks[t].unit].role == "h"

    def ev(kind: str, cls: str, fam: str, r: str, o: str, group: str = "") -> None:
        events.append(Event(kind, cls, fam, r, o, group or kind))

    def missing(what: str, cls: str, fam: str, r: str, group: str) -> None:
        """A reference atom whose text the output dropped (or hid in a comment or display math) is lost with
        it: deleting struck text must not beat keeping it plain. The inventory mode counts it as a FN instead."""
        if not scope.decoration:
            ev(f"{what} missing", cls, fam, r, "", group)
            atoms[fam].append(LOSS[cls])

    by_line: dict[int, list[int]] = {}  # one typo-level event per reference line, however many words it respells
    for t in sorted(respelled):
        by_line.setdefault(ref.toks[t].unit, []).append(t)
    for x, ts in by_line.items():
        fam = {"h": "heading", "item": "list"}.get(ref.units[x].role, "inline")
        ev("formatted text respelled", "minor", fam, _excerpt(ref, ts), _excerpt(out, [match[t] for t in ts]))
        atoms[fam].append(LOSS["minor"])

    # ---- inline spans: bold, italic, underline, strikethrough, highlight, inline code ----
    used_out: dict[int, set[int]] = {k: set() for k in SPAN_KINDS}  # output tokens explained by a style change
    for k in SPAN_KINDS:
        if not k & scope.kinds:
            continue
        name = KIND_NAMES[k]
        cls_lost = "critical" if k == S else ("major" if k == U and redline else "minor")
        for sp in _spans(ref, k):
            ref_atoms += 1
            if all(match[t] < 0 for t in sp):
                missing(name, cls_lost, "inline", _excerpt(ref, sp), f"{name} lost")
                continue
            located += 1
            al = [t for t in sp if match[t] >= 0 and not (k == B and in_heading(out, match[t]))]
            if not al:  # a bold label shown as a heading: judged by the heading lane
                atoms["inline"].append(0.0)
                continue
            kept = [t for t in al if out.toks[match[t]].has(k)]
            frac = len(kept) / len(al)
            if frac >= 1:
                atoms["inline"].append(0.0)
                continue
            other = [
                t
                for t in al
                if t not in kept and out.toks[match[t]].has(EMPHASIS & ~k) and not ref.toks[t].has(EMPHASIS & ~k)
            ]
            o_ex = _excerpt(out, [match[t] for t in al])
            if k & EMPHASIS and not kept and len(other) == len(al):
                ev(
                    "emphasis style changed",
                    "minor",
                    "inline",
                    f"{name}: {_excerpt(ref, sp)}",
                    o_ex,
                    "emphasis style changed",
                )
                atoms["inline"].append(LOSS["minor"])
                for t in other:
                    for k2 in SPAN_KINDS:
                        if out.toks[match[t]].has(k2):
                            used_out[k2].add(match[t])
                continue
            if k == S or frac < 0.5:
                kind = f"{name} lost" if not kept else f"{name} partly lost"
                ev(kind, cls_lost, "inline", _excerpt(ref, sp), o_ex, f"{name} lost")
                atoms["inline"].append(LOSS[cls_lost] * (1 - frac))
            else:
                ev(f"{name} span shifted", "minor", "inline", _excerpt(ref, sp), o_ex, f"{name} shifted")
                atoms["inline"].append(LOSS["minor"] * (1 - frac))
        # output spans adding the style on aligned text
        for sp in _spans(out, k):
            al = [j for j in sp if back[j] >= 0 and not (k == B and in_heading(ref, back[j]))]
            extra = [j for j in al if not ref.toks[back[j]].has(k) and j not in used_out[k] and back[j] not in skip]
            if not extra:
                continue
            overlap = len(extra) < len(al)
            cls = "critical" if k == S else ("major" if k == U and redline else "minor")
            if overlap and k != S:
                ev(
                    f"{name} span shifted",
                    "minor",
                    "inline",
                    _excerpt(ref, [back[j] for j in al]),
                    _excerpt(out, sp),
                    f"{name} shifted",
                )
                atoms["inline"].append(LOSS["minor"] * len(extra) / len(al))
            else:
                ev(
                    f"{name} added",
                    cls,
                    "inline",
                    _excerpt(ref, [back[j] for j in al]),
                    _excerpt(out, sp),
                    f"{name} added",
                )
                atoms["inline"].append(LOSS[cls] * len(extra) / len(al))

    # ---- superscripts and subscripts: per token, exact character positions ----
    for k, other_k in ((SUP, SUB), (SUB, SUP)):
        if not k & scope.kinds:
            continue
        name = KIND_NAMES[k]
        for t, tok in enumerate(ref.toks):
            if not tok.has(k):
                continue
            scls = _script_class(tok, k)
            if scls is None:
                continue
            j = match[t]
            if j < 0 and tok.has(F):
                continue  # a formula the output wrote as display math or dropped: the formula metric's business
            ref_atoms += 1
            if j < 0:
                missing("exponent" if scls == "critical" else name, scls, "inline", tok.key, f"{name} lost")
                continue
            located += 1
            ot = out.toks[j]
            if ot.mask(k) == tok.mask(k):
                atoms["inline"].append(0.0)
                continue
            label = "exponent" if scls == "critical" else name
            if ot.mask(other_k) == tok.mask(k):
                kind = f"{name} read as {KIND_NAMES[other_k]}"
            elif ot.has(k):
                kind = f"{label} misplaced"
            else:
                kind = f"{label} lost"
            ev(kind, scls, "inline", tok.key, ot.key, f"{name} lost")
            atoms["inline"].append(LOSS[scls])
        for j, ot in enumerate(out.toks):
            if not ot.has(k) or back[j] < 0:
                continue
            r = ref.toks[back[j]]
            if r.has(k) or r.mask(other_k) == ot.mask(k):
                continue  # judged above (misplaced / swapped)
            scls = _script_class(ot, k)
            if scls is None:
                continue
            ev(("exponent" if scls == "critical" else name) + " added", scls, "inline", r.key, ot.key, f"{name} added")
            atoms["inline"].append(LOSS[scls])

    # ---- code blocks ----
    def code_units(doc: Doc) -> list[int]:
        return [x for x, u in enumerate(doc.units) if u.role == "code" and u.toks and C & scope.kinds]

    for x in code_units(ref):
        ref_atoms += 1
        al = [match[t] for t in ref.units[x].toks if match[t] >= 0]
        if not al:
            missing("code block", "minor", "inline", _excerpt(ref, ref.units[x].toks), "code block lost")
            continue
        located += 1
        n_kept = sum(1 for j in al if out.units[out.toks[j].unit].role == "code")
        if 2 * n_kept >= len(al):
            atoms["inline"].append(0.0)
        else:
            ev("code block lost", "minor", "inline", _excerpt(ref, ref.units[x].toks), _excerpt(out, al))
            atoms["inline"].append(LOSS["minor"])
    for x in code_units(out):
        al = [back[j] for j in out.units[x].toks if back[j] >= 0]
        if al and 2 * sum(1 for t in al if ref.units[ref.toks[t].unit].role == "code") < len(al):
            ev("code block added", "minor", "inline", _excerpt(ref, al), _excerpt(out, out.units[x].toks))
            atoms["inline"].append(LOSS["minor"])

    # ---- unit mapping (reference unit -> majority output unit of its aligned tokens) ----
    def target(x: int) -> tuple[int, int]:
        toks = ref.units[x].toks
        al = [out.toks[match[t]].unit for t in toks if match[t] >= 0]
        if not toks or 2 * len(al) < len(toks):
            return -1, len(al)
        return _majority(al), len(al)

    def source(y: int) -> int:
        toks = out.units[y].toks
        al = [ref.toks[back[j]].unit for j in toks if back[j] >= 0]
        if not toks or 2 * len(al) < len(toks):
            return -1
        return _majority(al)

    ref_target = [target(x)[0] for x in range(len(ref.units))]

    # ---- headings ----
    heads = [x for x, u in enumerate(ref.units) if u.role in ("h", "label") and u.toks and scope.headings]
    chain: list[tuple[int, int]] = []  # (ref unit, out unit) for matched heading-like units, in reference order
    claimed: set[int] = set()
    head_atom: dict[int, int] = {}
    merged_into: dict[int, int] = {}  # output heading line -> first reference heading read into it
    merged_charged: set[int] = set()
    for x in heads:
        u = ref.units[x]
        if u.role == "h":
            ref_atoms += 1
        y = ref_target[x]
        if y < 0:
            if u.role == "h":
                missing("heading", "major", "heading", u.text, "heading lost")
            continue
        o = out.units[y]
        if o.role in ("h", "label"):
            claimed.add(y)
            if u.role == "h":
                located += 1
                head_atom[x] = len(atoms["heading"])
                # A heading of another level read into the same output line loses a section boundary; same-level
                # lines are one heading wrapped over several lines, free.
                first = merged_into.setdefault(y, x)
                if ref.units[first].level != u.level:
                    if y not in merged_charged:
                        merged_charged.add(y)
                        ev("headings merged", "major", "heading", f"{ref.units[first].text} / {u.text}", o.text)
                    atoms["heading"].append(LOSS["major"])
                    continue
                if o.role == "label":
                    ev("heading shown as bold line", "minor", "heading", u.text, o.text)
                    atoms["heading"].append(LOSS["minor"])
                else:
                    chain.append((x, y))  # levels are compared between real headings only
                    atoms["heading"].append(0.0)
        elif u.role == "h":
            located += 1
            ev("heading lost", "major", "heading", u.text, o.text)
            atoms["heading"].append(LOSS["major"])
    if len(chain) >= 2:
        rr = _rank([ref.units[x].level for x, _ in chain])
        orank = _rank([out.units[y].level for _, y in chain])
        # every heading at one level while the reference has several: one systematic major event
        flat = len(chain) >= 3 and len(rr) >= 2 and len(orank) == 1
        flat_hits = 0
        for (xa, ya), (xb, yb) in zip(chain, chain[1:], strict=False):
            sr = _sign(ref.units[xb].level - ref.units[xa].level)
            so = _sign(out.units[yb].level - out.units[ya].level)
            if ya == yb or sr == so:
                continue
            gap = abs(rr[ref.units[xb].level] - orank[out.units[yb].level])
            cls = "major" if sr * so < 0 or gap >= 2 or flat else "minor"
            kind = "heading hierarchy inverted" if sr * so < 0 else "heading level relation changed"
            a = head_atom[xb]
            atoms["heading"][a] = max(atoms["heading"][a], LOSS[cls])
            if flat:
                flat_hits += 1
                continue
            ev(
                kind,
                cls,
                "heading",
                f"{_head(ref.units[xa])} > {_head(ref.units[xb])}",
                f"{_head(out.units[ya])} > {_head(out.units[yb])}",
                "heading level",
            )
        if flat_hits:
            ev(
                "heading hierarchy flattened",
                "major",
                "heading",
                " / ".join(_head(ref.units[x]) for x, _ in chain[:4]),
                " / ".join(_head(out.units[y]) for _, y in chain[:4]),
            )
    promoted: set[int] = set()
    for y, o in enumerate(out.units):
        if o.role != "h" or y in claimed or not o.toks or not scope.headings:
            continue
        x = source(y)
        if x < 0 and scope.complete_headings and all(back[j] < 0 for j in o.toks):
            ev("heading not in the page's heading list", "minor", "heading", "", o.text, "false heading")
            atoms["heading"].append(LOSS["minor"])
            continue
        if x < 0 or x in promoted:
            continue
        ru = ref.units[x]
        if ru.role in ("h", "label"):
            continue  # a split heading: the claimed part carries the judgement
        promoted.add(x)  # one event per reference line, however many headings it was cut into
        al = [back[j] for j in o.toks if back[j] >= 0]
        short = len(ru.toks) <= SHORT_LINE_TOKENS
        bold_label = all(ref.toks[t].has(B) for t in al)
        cls = "minor" if short or bold_label else "major"
        ev(
            "text promoted to heading" if cls == "minor" else "body text turned into heading",
            cls,
            "heading",
            ru.text,
            o.text,
            "false heading",
        )
        atoms["heading"].append(LOSS[cls])

    # ---- lists ----
    items = [x for x, u in enumerate(ref.units) if u.role == "item" and u.toks and scope.lists]
    by_target: dict[int, list[int]] = {}
    for x in items:
        ref_atoms += 1
        y = ref_target[x]
        if y < 0:
            missing("list item", "minor", "list", ref.units[x].text, "list marker lost")
            continue
        located += 1
        by_target.setdefault(y, []).append(x)
    item_loss: dict[int, float] = {}
    merged_seen: set[int] = set()
    for x in items:
        y = ref_target[x]
        if y < 0:
            continue
        o = out.units[y]
        if len(by_target[y]) > 1 and o.role != "h":
            if y not in merged_seen:
                merged_seen.add(y)
                ev(
                    "list items merged",
                    "major",
                    "list",
                    " / ".join(ref.units[z].text for z in by_target[y][:3]),
                    o.text,
                )
            item_loss[x] = LOSS["major"]
        elif o.role != "item":
            ev("list marker lost", "minor", "list", ref.units[x].text, o.text)
            item_loss[x] = LOSS["minor"]
        else:
            item_loss[x] = 0.0
            hits = [out.toks[match[t]].unit for t in ref.units[x].toks if match[t] >= 0]
            split = [
                z
                for z in dict.fromkeys(hits)
                if z != y and out.units[z].role == "item" and hits.count(z) >= 2 and z not in by_target
            ]
            if split:
                ev(
                    "list item split",
                    "minor",
                    "list",
                    ref.units[x].text,
                    " / ".join(out.units[z].text for z in [y, *split][:3]),
                )
                item_loss[x] = LOSS["minor"]
    # ordered vs unordered (one event per reference list) and nesting relations
    prev: dict[int, int] = {}
    typed: set[int] = set()
    for x in items:
        y = ref_target[x]
        if y < 0 or out.units[y].role != "item" or len(by_target[y]) > 1:
            continue
        u, o = ref.units[x], out.units[y]
        if u.ordered != o.ordered and u.group not in typed:
            typed.add(u.group)
            ev("list type changed", "minor", "list", u.text, o.text)
            item_loss[x] = max(item_loss[x], LOSS["minor"])
        p = prev.get(u.group)
        prev[u.group] = x
        if p is None:
            continue
        py = ref_target[p]
        sr = _sign(u.depth - ref.units[p].depth)
        so = _sign(o.depth - out.units[py].depth)
        if sr == so or (sr < 0 and so == 0):  # back to a shallower level: follows from the flattening
            continue
        if sr > 0:
            ev("list nesting flattened", "major", "list", u.text, o.text, "list nesting")
            item_loss[x] = max(item_loss[x], LOSS["major"])
        else:
            ev("list nesting changed", "minor", "list", u.text, o.text, "list nesting")
            item_loss[x] = max(item_loss[x], LOSS["minor"])
    atoms["list"].extend(item_loss[x] for x in items if x in item_loss)
    for y, o in enumerate(out.units):
        if o.role != "item" or not o.toks or not scope.lists:
            continue
        x = source(y)
        if x < 0 or ref.units[x].role in ("item", "h"):
            continue
        ev("text turned into list item", "minor", "list", ref.units[x].text, o.text)
        atoms["list"].append(LOSS["minor"])

    # ---- floor: systematic quirks, first event in full, later ones scaled ----
    n_atoms = sum(len(v) for v in atoms.values())
    by_group: dict[str, list[Event]] = {}
    for e in events:
        by_group.setdefault(e.group, []).append(e)
    instances = _instances(ref, atoms)
    for g, es in by_group.items():
        if es[0].cls != "critical" and len(es) >= SYSTEMATIC_MIN and len(es) >= instances.get(g, 10**9):
            for e in es[1:]:
                e.charged = False
            es[0].kind += f" (systematic, {len(es)}x)"
    cnt = {"critical": 0, "major": 0, "minor": 0}
    for e in events:
        if e.charged:
            cnt[e.cls] += 1
    scale = min(1.0, FLOOR_K / max(1, n_atoms))

    def fix(c: int) -> float:
        return min(c, 1) + max(0, c - 1) * scale

    floor = _clamp01(
        1
        - FLOOR["critical"] * fix(cnt["critical"])
        - FLOOR["major"] * fix(cnt["major"])
        - min(FLOOR["minor_cap"], FLOOR["minor"] * cnt["minor"])
    )
    total_loss = sum(sum(v) for v in atoms.values())
    proportional = _clamp01(1 - total_loss / (n_atoms + PRIOR_ATOMS))
    coverage = located / ref_atoms if ref_atoms else 1.0
    decoration = _decoration_counts(ref, out, match, scope, skip)
    if scope.decoration:
        # Full-page inventory: the decoration's token F1 is the proportional part, and coverage is the share of
        # scored reference tokens the output carries, so an output that drops the text cannot pass.
        tp, fp, fn = (sum(c[x] for c in decoration.values()) for x in ("tp", "fp", "fn"))
        proportional = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
        scored = [i for i in range(len(ref.toks)) if i not in skip]
        if tp + fn == 0 and fp:
            # Hard negative (no decoration on the page): F1 is 0 for any false positive, so charge the share of
            # scored tokens wrongly decorated instead; the severity floor still caps the score.
            proportional = _clamp01(1 - fp / max(1, len(scored)))
        coverage = sum(1 for i in scored if match[i] >= 0) / len(scored) if scored else 1.0
    score = min(proportional, floor)
    if coverage < COVERAGE_FREE:
        score = min(score, coverage / COVERAGE_FREE)
        events.append(
            Event(
                f"page text missing ({1 - coverage:.0%} of the reference tokens not found)"
                if scope.decoration
                else f"formatted text missing ({ref_atoms - located} of {ref_atoms} items not found)",
                "major",
                "coverage",
                "",
                "",
                "coverage",
                False,
            )
        )
    lanes: dict[str, float | None] = {}
    for fam, lane in LANE_OF.items():
        v = atoms[fam]
        lanes[lane] = _clamp01(1 - sum(v) / len(v)) if v else None
    for name, c in decoration.items():
        n = c["tp"] + c["fp"] + c["fn"]
        lanes[name] = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if n else None
    return FormattingResult(
        score=score,
        proportional=proportional,
        floor=floor,
        coverage=coverage,
        lanes=lanes,
        atoms=n_atoms,
        events=events,
        n_ref_tokens=len(ref.toks),
        n_out_tokens=len(out.toks),
        decoration=decoration,
    )


DECORATION_LANES = {S: "strikeout", U: "underline"}


def _decoration_counts(
    ref: Doc, out: Doc, match: Sequence[int], scope: Scope, skip: frozenset[int]
) -> dict[str, dict[str, int]]:
    """Token counts of strikethrough and underline, per reference token (``skip``: neutral tokens).

    TP: decorated in the reference and in its aligned output token; FN: decorated in the reference, output
    token undecorated or missing; FP: output decorates an aligned token the reference leaves plain. Output
    tokens with no reference counterpart are not judged. A kind appears only when the scope covers it.
    """
    counts: dict[str, dict[str, int]] = {}
    for k, name in DECORATION_LANES.items():
        if not k & scope.kinds:
            continue
        c = {"tp": 0, "fp": 0, "fn": 0}
        for i, tok in enumerate(ref.toks):
            if i in skip:
                continue
            d = tok.has(k)
            o = match[i] >= 0 and out.toks[match[i]].has(k)
            if d and o:
                c["tp"] += 1
            elif d:
                c["fn"] += 1
            elif o:
                c["fp"] += 1
        counts[name] = c
    return counts


def _instances(ref: Doc, atoms: dict[str, list[float]]) -> dict[str, int]:
    """How many instances a systematic group needs to cover: all reference items of that kind."""
    n: dict[str, int] = {}
    for k in SPAN_KINDS:
        name = KIND_NAMES[k]
        cnt = 0
        prev_on = False
        for t, tok in enumerate(ref.toks):
            on = tok.has(k) and not (k == B and ref.units[tok.unit].role == "h")
            if on and not (prev_on and ref.toks[t - 1].unit == tok.unit):
                cnt += 1
            prev_on = on
        n[f"{name} lost"] = cnt
    for k in (SUP, SUB):
        n[f"{KIND_NAMES[k]} lost"] = sum(1 for t in ref.toks if t.has(k) and _script_class(t, k) is not None)
    n["heading lost"] = sum(1 for u in ref.units if u.role == "h" and u.toks)
    n["heading shown as bold line"] = n["heading lost"]
    n["list marker lost"] = sum(1 for u in ref.units if u.role == "item" and u.toks)
    return n


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def has_formatting(doc: Doc) -> bool:
    """Whether a reference page carries any scored formatting."""
    if any(u.role in ("h", "item", "code") and u.toks for u in doc.units):
        return True
    for t in doc.toks:
        if any(t.has(k) for k in SPAN_KINDS):
            return True
        for k in (SUP, SUB):
            if t.has(k) and _script_class(t, k) is not None:
                return True
    return False


def score_formatting(ref_md: str, out_md: str, scope: Scope = FULL) -> FormattingResult:
    """Score an output page (``out_md``) against its reference page (``ref_md``)."""
    return score_docs(parse(ref_md), parse(out_md), scope)


def formatting_v2_applies(reference: str | None) -> bool:
    return bool(reference) and has_formatting(parse(reference or ""))


_RULE_WRAP = {
    "is_bold": "**{}**",
    "is_italic": "*{}*",
    "is_underline": "<u>{}</u>",
    "is_strikeout": "~~{}~~",
    "present_as_strikeout": "~~{}~~",
    "is_sup": "<sup>{}</sup>",
    "is_sub": "<sub>{}</sub>",
    "is_mark": "<mark>{}</mark>",
}
_RULE_PLAIN = {
    "is_not_bold",
    "is_not_italic",
    "is_not_underline",
    "is_not_strikeout",
    "is_not_sup",
    "is_not_sub",
    "is_not_mark",
    "present",
}


_RULE_KIND = {
    "is_bold": B,
    "is_not_bold": B,
    "is_italic": I,
    "is_not_italic": I,
    "is_underline": U,
    "is_not_underline": U,
    "is_strikeout": S,
    "is_not_strikeout": S,
    "present_as_strikeout": S,
    "is_sup": SUP,
    "is_not_sup": SUP,
    "is_sub": SUB,
    "is_not_sub": SUB,
    "is_mark": M,
    "is_not_mark": M,
}


def rules_reference(rules: Sequence[Any] | None) -> tuple[str, Scope] | None:
    """A sparse reference page built from formatting rules, for cases without reference Markdown.

    Rule-only formatting test cases (headings, strikeout_text, underlined_text) carry only rules: ``heading_structure``
    (the complete ordered heading list of the page), positive span rules (``present_as_strikeout``,
    ``is_underline``, ...) and negative or plain-presence controls (``is_not_*``, ``present``). Each becomes one
    paragraph of the reference: a heading line, a styled span or plain text. Alignment is move tolerant, so
    paragraph order does not matter; output text absent from the reference is simply not judged. The scope
    limits the judgement to what the rules name (an underline on a headings page is not an error), and a
    ``heading_structure`` rule, being the page's complete heading list, makes any other output heading a
    false one. A page whose rules hold no styled span (a hard negative such as ``present`` + ``is_not_strikeout``,
    or an empty heading list) is still scored: adding the named formatting to its spans, or any heading, is an
    error. Only rules that name no formatting at all (``present`` alone) give no reference.
    """
    parts: list[str] = []
    kinds = 0
    headings = complete = False
    for r in rules or ():
        t = str(getattr(r, "type", "") or "")
        kinds |= _RULE_KIND.get(t, 0)
        if t in ("heading_structure", "is_title"):
            headings = True
        if t == "heading_structure":
            complete = True
            for h in getattr(r, "headings", None) or []:
                text = h.get("text") if isinstance(h, dict) else getattr(h, "text", None)
                level = h.get("level") if isinstance(h, dict) else getattr(h, "level", None)
                if text and isinstance(level, int) and 1 <= level <= 6:
                    parts.append("#" * level + " " + " ".join(str(text).split()))
            continue
        text = getattr(r, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        text = " ".join(text.split())
        if t in _RULE_WRAP:
            parts.append(_RULE_WRAP[t].format(text))
        elif t in _RULE_PLAIN and text not in parts:  # a hard negative pairs present + is_not_* on one span
            parts.append(text)
        elif t == "is_title":
            level = getattr(r, "level", None)
            parts.append(("#" * level + " " + text) if isinstance(level, int) and 1 <= level <= 6 else f"**{text}**")
    if not kinds and not headings:
        return None
    reference = "\n\n".join(parts)
    return reference, Scope(kinds=kinds, headings=headings, lists=False, complete_headings=complete)


def _cap_unlocated_plain_text(ref: Doc, out: Doc, res: FormattingResult) -> FormattingResult:
    """A reference without formatted items (a rule-mode hard negative) has no atoms to locate, so coverage alone
    would pass an output that dropped the span it tests. Locate its plain tokens instead and apply the same cap."""
    need = Counter(t.key for t in ref.toks)
    total = sum(need.values())
    if not total:
        return res
    have = Counter(t.key for t in out.toks)
    coverage = sum(min(n, have[k]) for k, n in need.items()) / total
    res.coverage = coverage
    if coverage < COVERAGE_FREE:
        res.score = min(res.score, coverage / COVERAGE_FREE)
        missing = total - sum(min(n, have[k]) for k, n in need.items())
        res.events.append(
            Event(
                f"plain text missing ({missing} of {total} tokens not found)",
                "major",
                "coverage",
                "",
                "",
                "coverage",
                False,
            )
        )
    return res


DECORATION_KINDS = {"strikeout": S, "underline": U}


def decoration_reference(metadata: Mapping[str, Any] | None) -> tuple[str, Scope, list[str]] | None:
    """(reference Markdown, scope, neutral texts) from a full-page decoration inventory, or None.

    ``metadata.decorations`` = {kind: "strikeout" | "underline", reference_md, neutral}: the page text with
    every run of that decoration wrapped, and texts scored neither way (runs the reviewers could not decide).
    Only that decoration is judged; headings and lists are not.
    """
    dec = (metadata or {}).get("decorations")
    if not isinstance(dec, Mapping) or dec.get("kind") not in DECORATION_KINDS or not dec.get("reference_md"):
        return None
    neutral = [str(t) for t in dec.get("neutral") or [] if str(t).strip()]
    scope = Scope(kinds=DECORATION_KINDS[dec["kind"]], headings=False, lists=False, decoration=True)
    return str(dec["reference_md"]), scope, neutral


def _neutral_tokens(ref: Doc, neutral: Sequence[str]) -> frozenset[int]:
    """Reference token indices covered by a neutral text (every occurrence of its key sequence)."""
    keys = [t.key for t in ref.toks]
    out: set[int] = set()
    for text in neutral:
        seq = [t.key for t in parse(text).toks]
        if not seq:
            continue
        for i in range(len(keys) - len(seq) + 1):
            if keys[i : i + len(seq)] == seq:
                out.update(range(i, i + len(seq)))
    return frozenset(out)


def formatting_v2_metrics(
    reference: str, output_markdown: str, scope: Scope = FULL, neutral: Sequence[str] = ()
) -> list[MetricValue]:
    """``formatting_v2`` headline plus its lane diagnostics as metric values."""
    ref_doc, out_doc = parse(reference), parse(output_markdown)
    res = score_docs(ref_doc, out_doc, scope, _neutral_tokens(ref_doc, neutral))
    if not scope.decoration and not has_formatting(ref_doc):
        res = _cap_unlocated_plain_text(ref_doc, out_doc, res)
    metrics = [MetricValue(metric_name="formatting_v2", value=res.score, metadata=res.metadata())]
    for lane, value in res.lanes.items():
        if value is not None:
            metrics.append(MetricValue(metric_name=f"formatting_v2_{lane}", value=float(value)))
    return metrics
