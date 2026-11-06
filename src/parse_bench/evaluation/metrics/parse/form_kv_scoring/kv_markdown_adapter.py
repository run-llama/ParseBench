"""Ordinary page markdown -> the guide's list lines, for the `form_kv_md_*` arm only.

The page markdown writes form fields in the shape the parse prompt asks for (`**Label**: value`
pairs, `[x]`/`[ ]` boxes, `[___]` blanks) but not as the guide's list: pairs sit in paragraphs,
several pairs share a line, a printed line number is a bold marker (`**13** Wages… **13** 4,200`),
an option's box trails its caption (`Yes [ ]`) or a run of boxes sits inline (`[ ] Single [x] Head
of household`), and questions are numbered items. `kv_grammar` leaves every such line unparsed
on purpose, so scoring page markdown through it measured the dialect gap, not the parse. This
adapter rewrites those shapes into list lines, line by line, and touches nothing else: labels,
values and states are re-emitted verbatim, tables pass through untouched, prose stays unparsed.
The enriched `list.md` arm never goes through it.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<text>.*?)[ \t]*$")
_BULLET_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-*+]|\d{1,3}[.)])[ \t]+(?P<body>\S.*)$")
_TABLE_OPEN_RE = re.compile(r"^[ \t]*<table\b", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_BOLD_RE = re.compile(r"\*\*(?P<key>(?:\\\*|[^*\n])+?)\*\*")
_BOX_RE = re.compile(r"\[[ xX]\](?=\s|$)")
_BLANK_RE = re.compile(r"^\[(?:blank|_+)\]")
_TRAIL_SEP_RE = re.compile(r"[\s,;]+$")
_PAIR_HEAD_RE = re.compile(r"^(?:\\\[[^\]]+\\\]\s*)?(?:\*\*(?:\\\*|[^*\n])+?\*\*)?")
_GUIDE_ID_RE = re.compile(r"^\\\[[^\]\n]+?\\\]")  # an escaped printed id: the line is already the guide's
_PIPE_ROW_RE = re.compile(r"^\s*\||\s\|\s.*\s\|\s|^\s*:?-{3,}\s*\|")  # a markdown pipe-table row or rule
_BLANK_LINE_RE = re.compile(r"^\[(?:blank|_{2,})\]\s*$")
# `Name: ______`, `Date [___]`: a short unbolded caption followed only by a fill marker
_PLAIN_BLANK_RE = re.compile(
    r"^(?P<label>[A-Za-z][^:*\[\]_|\n]{0,60}?)\s*(?P<colon>:)?\s*(?:_{3,}|\[_{2,}\]|\[blank\])\s*$"
)
_UNDER_CAPTION_RE = re.compile(r"^\((?P<caption>[^()\n]{1,60})\)\s*$")
# a printed line number on its own in bold: `**13**`, `**1a**`, `**b**`, `**III**`
_ID_KEY_RE = re.compile(r"^(?:\d{1,3}[a-z]?|[A-Za-z]|[IVX]{1,4})$")
_BAKED_ID_RE = re.compile(r"^(?P<id>\d{1,3}[a-z]?|[a-z])[.)]?\s+\S")
# a dot leader (`Relaxing ........ 14`): what follows it at the end of the line is a page number
_DOT_LEADER_RE = re.compile(r"(?:\.{4,}|…{2,}|(?:\. ){3,}\.?)\s*$")

DIAG_KEYS = (
    "md_adapted_lines",
    "md_key_lines",
    "md_split_pairs",
    "md_id_markers",
    "md_box_groups",
    "md_box_moved",
    "md_box_lines",
    "md_numbered_items",
    "md_under_captions",
    "md_plain_fields",
    "md_tables_unwrapped",
    "md_table_fields",
)
# a form value is short: a name, a number, a date, an address line. Longer text after a key is
# a definition, a note or a paragraph with a bold lead-in, and stays text.
_FIELD_VALUE_MAX_WORDS = 12


def new_adapter_diagnostics() -> dict[str, int]:
    return dict.fromkeys(DIAG_KEYS, 0)


def _unbold(text: str) -> str:
    return _BOLD_RE.sub(lambda m: m.group("key"), text)


@dataclass
class _Pair:
    """One key-value unit of a line: a bold key (`**Label**`, `**Label:**`) or a bold printed
    id (`**13**`) followed by label text, an optional value-cell marker (`**13**` again) and
    the value."""

    key: str = ""  # the bold run, verbatim, when the pair opens with a label
    id: str | None = None  # the printed id when the pair opens with an id marker
    text: list[str] = field(default_factory=list)  # pieces after the key, before any marker
    value: list[str] | None = None  # pieces after the value-cell marker
    colon: bool = False  # the key carries a colon, inside or outside the bold

    def anchor_id(self) -> str | None:
        if self.id is not None:
            return self.id
        m = _BAKED_ID_RE.match(_unbold(self.key)) if self.key else None
        return m.group("id") if m else None

    def add(self, piece: str) -> None:
        (self.value if self.value is not None else self.text).append(piece)

    def takes_marker(self, id_: str) -> bool:
        """`**13**` after `**13** Wages…` (or right after `**4a Label**:`) marks the value cell;
        a sub-line's marker may carry the parent number (`**b** Nontaxable combat pay **66b**`)."""
        if self.value is not None:
            return False
        anchor = self.anchor_id()
        if anchor is not None:
            if id_ == anchor:
                return True
            if len(anchor) == 1 and anchor.isalpha() and id_.endswith(anchor) and id_[:-1].isdigit():
                return True
        return self.colon and not "".join(self.text).strip()

    def render(self) -> str:
        text = _TRAIL_SEP_RE.sub("", "".join(self.text)).strip()
        value = _TRAIL_SEP_RE.sub("", "".join(self.value)).strip() if self.value is not None else None
        if self.id is not None:
            head = f"\\[{self.id}\\]"
            if text and (_BOX_RE.match(text) or _BLANK_RE.match(text)):
                value, text = _unbold(text if value is None else f"{text} {value}"), ""
            elif text and value is None and (box := _BOX_RE.search(text)) is not None:
                # `**D** TYPE OF ACCOUNT [ ] CHECKING [ ] SAVINGS`: the label ends at the first box
                text, value = text[: box.start()].strip(), text[box.start() :].strip()
            if text:
                head = f"{head} **{_plain_label(text)}**"
            return f"{head}: {value}".rstrip() if value is not None else head
        inner = _unbold(self.key).strip()
        if value is not None and not self.colon and text:
            # `**8 Total** income from the worksheet **8** 4,200`: the caption continues past
            # the bold run up to the value-cell marker
            return f"**{_plain_label(inner + ' ' + text)}**: {value}".rstrip()
        head = f"{self.key}:" if self.colon and not inner.endswith(":") else self.key
        body = value if value is not None else text
        return f"{head} {body}".rstrip()


def _plain_label(text: str) -> str:
    """A label is plain text in the guide: bold inside it is the page's own emphasis, and a
    printed `*` (italics markers included) is escaped so the grammar keeps the label whole."""
    return _unbold(text).strip().replace("\\*", "*").replace("*", "\\*")


def _is_emphasised_number(body: str, m: re.Match[str]) -> bool:
    """`Enter **0** if not applicable`: a bold number inside prose, one space each side and a
    lowercase word after it, is emphasis, not a printed id."""
    if m.start() == 0:
        return False
    before, after = body[: m.start()], body[m.end() :]
    return before.endswith(" ") and not before.endswith("  ") and bool(re.match(r"\s+[a-z]", after))


def _starts_pair(body: str, m: re.Match[str], prev_end: int) -> bool:
    """A bold label starts a new pair when it opens the line, carries its own colon, or is set
    off from the previous pair by a gap or a token; bold inside a value is emphasis."""
    if m.start() == 0:
        return True
    key, after = m.group("key"), body[m.end() :]
    if key.rstrip().endswith(":") or after.startswith(":"):
        return True
    gap = body[prev_end : m.start()]
    if not gap.strip():
        return len(gap) >= 2
    if _BOX_RE.search(gap.strip()[-3:]):
        # `[ ] **Yourself**`: the bold right after a box is that box's label, unless it is a
        # key of its own with a box or colon after it (`**Approve** [ ] **Disapprove** [ ]`)
        return bool(re.match(r"\s*(\[[ xX]\]|:)", after))
    return _BLANK_RE.match(gap.strip().split()[-1]) is not None or _BOX_RE.search(gap.strip()) is not None


def _split_pairs(body: str, diag: dict[str, int]) -> list[str] | None:
    """`**A**: x **B**: y` -> [`**A**: x`, `**B**: y`]; `**13** Wages **13** 4,200` ->
    [`\\[13\\] **Wages**: 4,200`]; None when the body does not open with a bold run."""
    if _BOLD_RE.match(body) is None:
        return None
    pairs: list[_Pair] = []
    current: _Pair | None = None
    pos = prev_end = 0
    for m in _BOLD_RE.finditer(body):
        key = m.group("key").strip()
        if _ID_KEY_RE.match(key) and not (current is not None and _is_emphasised_number(body, m)):
            if current is not None:
                current.add(body[pos : m.start()])
            owner = next((p for p in reversed(pairs) if p.takes_marker(key)), None)
            if (
                owner is None
                and current is not None
                and not body[m.end() :].strip()
                and _DOT_LEADER_RE.search(body[: m.start()])
            ):
                current.add(body[pos : m.end()])  # `**Relaxing** ....... **14**`: a contents page number
                pos = prev_end = m.end()
                continue
            if owner is not None:
                owner.value = []
                current = owner
                diag["md_id_markers"] += 1
            else:
                current = _Pair(id=key)
                pairs.append(current)
            pos = prev_end = m.end()
        elif current is None or _starts_pair(body, m, prev_end):
            if current is not None:
                current.add(body[pos : m.start()])
            rest = body[m.end() :]
            colon = key.endswith(":") or rest.startswith(":")
            current = _Pair(key=m.group(0), colon=colon)
            pairs.append(current)
            # a colon outside the bold is re-emitted by render()
            pos = prev_end = m.end() + (1 if rest.startswith(":") and not key.endswith(":") else 0)
        else:
            current.add(body[pos : m.end()])
            pos = prev_end = m.end()
    assert current is not None
    current.add(body[pos:])
    return [p.render() for p in pairs]


def _split_boxes(text: str) -> tuple[str, list[str]]:
    """`Filing [ ] Single [x] HOH` -> (`Filing`, [`[ ] Single`, `[x] HOH`])."""
    boxes = list(_BOX_RE.finditer(text))
    pre = text[: boxes[0].start()].strip()
    options = []
    for b, nxt in zip(boxes, [*boxes[1:], None], strict=True):
        end = nxt.start() if nxt is not None else len(text)
        options.append(_TRAIL_SEP_RE.sub("", text[b.start() : end]).strip())
    return pre, options


def _is_bold_caption(pair: str) -> bool:
    """`**FOR COURT USE ONLY**`: a bold run with no colon and nothing after it."""
    m = _PAIR_HEAD_RE.match(pair)
    key = m.group(0) if m else ""
    return (
        bool(key)
        and not key.startswith("\\[")
        and not _unbold(key).strip().endswith(":")
        and not pair[len(key) :].strip()
    )


def _key_and_rest(pair: str) -> tuple[str, str]:
    m = _PAIR_HEAD_RE.match(pair)
    key = m.group(0) if m else ""
    rest = pair[len(key) :]
    return key.rstrip(), rest[1:].strip() if rest.startswith(":") else rest.strip()


def _is_field_pair(pair: str) -> bool:
    """A pair is a form field when it is an id pair, an empty fill-in, a box or blank, or a
    colon key with a short value. A bold lead-in on prose (`**The bug.** When a core…`), a bold
    sentence, or a key whose value is an image is text, not a field."""
    key, rest = _key_and_rest(pair)
    if key.startswith("\\["):
        return True
    if "![" in rest:
        return False
    if _BOX_RE.match(rest) or _BLANK_RE.match(rest):
        return True
    colon = _unbold(key).strip().endswith(":") or pair[len(key) :].lstrip().startswith(":")
    if not rest:
        return colon  # `**Suffix**:` is an empty fill-in; `**Section Overview**` is a caption
    return colon and len(rest.split()) <= _FIELD_VALUE_MAX_WORDS


def _adapt_pair(pair: str, pad: str, marker: str, diag: dict[str, int]) -> list[str]:
    key, rest = _key_and_rest(pair)
    boxes = list(_BOX_RE.finditer(rest))
    if not key or not boxes or (len(boxes) == 1 and boxes[0].start() == 0):
        return [f"{pad}{marker} {pair}"]  # a value, a blank, or one box with an optional write-in
    if len(boxes) == 1:
        # `**Label**: unless you check this box: [ ]`: a lone labelled box goes in the value
        # position; the prose before it is instruction, the text after a write-in
        diag["md_box_moved"] += 1
        return [f"{pad}{marker} {key}: {rest[boxes[0].start() :].strip()}"]
    _pre, options = _split_boxes(rest)
    diag["md_box_groups"] += 1
    return [f"{pad}{marker} {key}", *(f"{pad}  - {opt}" for opt in options)]


def _split_labelled_boxes(body: str) -> tuple[str, list[str]]:
    """`You: Yes [ ] No [ ]` -> (`You:`, [`[ ] Yes`, `[ ] No`]): each box closes the option
    printed before it; a colon in the first stretch separates the group's caption."""
    caption, options, pos = "", [], 0
    for i, b in enumerate(_BOX_RE.finditer(body)):
        label = body[pos : b.start()].strip()
        if i == 0 and ":" in label:
            caption, _, label = label.rpartition(":")
            caption, label = caption.strip() + ":", label.strip()
        options.append(f"{b.group(0)} {label}".strip())
        pos = b.end()
    return caption, options


def _adapt_boxes(body: str, pad: str, marker: str, diag: dict[str, int]) -> list[str]:
    """A line of boxes without a key: one box becomes a box-first line (`Other [ ] (specify)`
    -> `[ ] Other (specify)`), a run becomes a group captioned by the text before the first box
    (`[ ] Single [x] HOH`) or, when every box trails its label (`Yes [ ] No [ ]`), by the text
    before the first label."""
    pre, options = _split_boxes(body)
    if len(options) == 1:
        box, after = options[0][:3], options[0][3:].strip()
        diag["md_box_moved"] += bool(pre)
        return [f"{pad}{marker} {' '.join(part for part in (box, pre, after) if part)}"]
    diag["md_box_groups"] += 1
    if _BOX_RE.match(body) is None and _BOX_RE.search(body.rstrip()[-3:]) is not None:
        pre, options = _split_labelled_boxes(body)
    caption = f"{pad}{marker} **{pre}**" if pre else f"{pad}{marker} "
    return [caption, *(f"{pad}  - {opt}" for opt in options)]


def _adapt_line(raw: str, diag: dict[str, int]) -> list[str]:
    if _HEADING_RE.match(raw):
        return [raw]  # a heading is a scope on both sides; its styling is not the adapter's business
    bullet = _BULLET_RE.match(raw)
    if bullet is not None:
        indent, marker, body = bullet.group("indent"), bullet.group("marker"), bullet.group("body")
        pad = indent.expandtabs(4)
        numbered = marker[0].isdigit()
        marker = "-" if numbered else marker  # a field in a numbered item takes the guide's marker
    else:
        stripped = raw.strip()
        if not stripped:
            return [raw]
        pad = raw[: len(raw) - len(raw.lstrip())].expandtabs(4)
        body, numbered, marker = stripped, False, "-"
    if _GUIDE_ID_RE.match(body) or _PIPE_ROW_RE.search(body):
        return [raw]  # already the guide's, or a markdown pipe table (tables are not rewritten)
    body = body[:-1].rstrip() if body.endswith("\\") else body  # a markdown hard line break

    pairs = _split_pairs(body, diag)
    if pairs is not None and bullet is None and len(pairs) == 1 and _is_bold_caption(pairs[0]):
        return [raw]  # `**FOR COURT USE ONLY**`: a bold line with no colon and no value is a caption
    if pairs is not None and not any(_is_field_pair(p) for p in pairs):
        return [raw]  # a bold lead-in on prose, a bold sentence: text, not a field
    if pairs is not None:
        out = [line for pair in pairs for line in _adapt_pair(pair, pad, marker, diag)]
        diag["md_split_pairs"] += len(pairs) - 1
        diag["md_key_lines"] += bullet is None
    elif _BOX_RE.search(body) is not None:
        if _BOX_RE.match(body) is not None and len(_BOX_RE.findall(body)) == 1:
            out = [f"{pad}{marker} {body}"]
        else:
            out = _adapt_boxes(body, pad, marker, diag)
        diag["md_box_lines"] += bullet is None
    elif (plain := _PLAIN_BLANK_RE.match(body)) is not None:
        # `Name: ____________`, `Date [___]`, `3. Fee [___]`: an unbolded caption followed by an
        # explicit fill marker is a blank field; a numbered item without one stays prose
        label, colon = plain.group("label").strip(), plain.group("colon") or ""
        out = [f"{pad}{marker} **{label}{colon}**{'' if colon else ':'} [blank]"]
        diag["md_plain_fields"] += 1
    else:
        return [raw]
    if out == [raw]:
        return [raw]  # a bullet that already had the guide's shape
    diag["md_numbered_items"] += numbered
    return out


# a table cell that is a caption: short, not a sentence
_CAPTION_MAX_WORDS = 25
_BOLD_ONLY_RE = re.compile(r"^\*\*[^*]+\*\*:?$")
_BOX_LINE_RE = re.compile(r"^- \[[ xX]\](?=\s|$)")
_EMPTY_CAPTION_RE = re.compile(r"^- (?:\\\[[^\]]+\\\] )?\*\*(?:\\\*|[^*])+\*\*:?$")
# a list line that is only a caption: `- **CITY**`, `- \[A\] **NAME OF PAYEE**`
_BARE_CAPTION_RE = re.compile(r"^- (?:\\\[[^\]]+\\\] )?\*\*(?:\\\*|[^*])+\*\*$")
# captions printed side by side in one cell: bold runs with only space between them, or a wide gap
_CELL_GAP_RE = re.compile(r"(?<=\*\*)\s+(?=\*\*)|\s{3,}")


def _cell_markdown(cell: Tag) -> str:
    """A table cell's contents as markdown: bold and italics kept as markers, `<br>` as a line break."""

    def walk(node: Any) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        if not isinstance(node, Tag):
            return ""
        if node.name == "br":
            return "\n"
        inner = "".join(walk(c) for c in node.children)
        if node.name in ("strong", "b") and inner.strip():
            return f"**{inner.strip()}**"
        if node.name in ("em", "i") and inner.strip():
            return f"*{inner.strip()}*"
        if node.name in ("p", "div", "li") and inner.strip():
            return inner + "\n"
        return inner

    text = html.unescape("".join(walk(c) for c in cell.children))
    return text.replace("****", "").replace("\xa0", " ")


def _is_data_cell(text: str) -> bool:
    """A filled cell of a data grid: text with no bold, no box, no fill marker, no trailing colon."""
    t = text.strip()
    return (
        bool(t)
        and "**" not in t
        and not _BOX_RE.search(t)
        and not re.search(r"_{3,}|\[(?:blank|_+)\]", t)
        and not t.endswith(":")
    )


def _is_grid(rows: list[list[str]], header_th: bool) -> bool:
    """A genuine grid: a header row over at least two rows of data. It stays a table."""
    if len(rows) < 3:
        return False
    head = [c for c in rows[0] if c.strip()]
    header = header_th or (len(head) >= 2 and all(_BOLD_ONLY_RE.match(c.strip()) for c in head))
    if not header:
        return False
    data_rows = 0
    for row in rows[1:]:
        cells = [c for c in row if c.strip()]
        if cells and sum(_is_data_cell(c) for c in cells) * 2 >= len(cells):
            data_rows += 1
    return data_rows >= 2


def _table_line(text: str, diag: dict[str, int]) -> list[str]:
    """One line of a form-table cell as guide list lines. Captions printed side by side in one cell
    (`**CITY**   **STATE**`) are separate fields; a caption with nothing after it is an empty fill-in;
    boxes, printed ids and pairs go through the line rules; a sentence stays text."""
    out: list[str] = []
    segs: list[str] = []
    for seg in _CELL_GAP_RE.split(text.strip()):
        # a printed id (`**D**`) belongs to the caption after it
        if segs and _ID_KEY_RE.match(_unbold(segs[-1]).strip()) and segs[-1].startswith("**"):
            segs[-1] = f"{segs[-1]} {seg}"
        else:
            segs.append(seg)
    for seg in segs:
        seg = seg.strip()
        if not seg:
            continue
        got = _adapt_line(seg, diag)
        if got == [seg]:
            plain = _plain_label(seg).rstrip(":").strip()
            words = plain.split()
            sentence = plain.endswith((".", "?", "!")) and len(words) > 8
            got = [seg] if not words or len(words) > _CAPTION_MAX_WORDS or sentence else [f"- **{plain}**:"]
        out.extend(f"{g}:" if _BARE_CAPTION_RE.match(g) else g for g in got)
    return out


def _adapt_table(block: str, diag: dict[str, int]) -> list[str] | None:
    """A form laid out as an HTML table, rewritten as guide list lines, cell by
    cell in reading order: a caption cell is a field, a caption next to a value cell is a pair, boxes
    and printed ids follow the line rules. A genuine grid returns None and stays a table."""
    soup = BeautifulSoup(block, "lxml")
    table = soup.find("table")
    if not isinstance(table, Tag):
        return None
    trs = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    raw_rows = [[c for c in tr.find_all(["td", "th"]) if c.find_parent("tr") is tr] for tr in trs]
    rows = [[_cell_markdown(c) for c in r] for r in raw_rows]
    if not rows or _is_grid(rows, bool(raw_rows[0]) and all(c.name == "th" for c in raw_rows[0])):
        return None
    out: list[str] = []
    caption_at: int | None = None  # the last caption line with nothing after it
    for row in rows:
        cells = [c.strip() for c in row if c.strip()]
        head = cells[0] if cells else ""
        if len(cells) == 2 and ("**" in head or head.endswith(":")) and "\n" not in head and _is_data_cell(cells[1]):
            # `**Name** | Ada`: a caption cell and the value cell beside it
            out.append(f"- **{_plain_label(head).rstrip(':').strip()}**: {cells[1]}")
            diag["md_table_fields"] += 1
            continue
        for cell in cells:
            for line in cell.split("\n"):
                if not line.strip():
                    continue
                got = _table_line(line, diag)
                if caption_at is not None and got and got[0] == "- ":
                    # a run of boxes with no caption of its own: the options of the caption cell above
                    out[caption_at] = out[caption_at].rstrip(":")
                    got = got[1:]
                elif caption_at is not None and got and all(_BOX_LINE_RE.match(g) for g in got):
                    # a cell holding one box: an option of the caption cell above
                    out[caption_at] = out[caption_at].rstrip(":")
                    got = [f"  {g}" for g in got]
                diag["md_table_fields"] += sum(1 for g in got if g.lstrip().startswith(("- ", "* ")))
                out.extend(got)
                if got and _EMPTY_CAPTION_RE.match(got[-1]):
                    caption_at = len(out) - 1
                elif got and not got[-1].startswith("  "):
                    caption_at = None
    diag["md_tables_unwrapped"] += 1
    return out


def adapt_page_markdown(markdown: str) -> tuple[str, dict[str, int]]:
    """The page markdown with its form shapes rewritten as guide list lines, plus counts of
    what was rewritten (`md_*`). Tables and prose pass through unchanged."""
    diag = new_adapter_diagnostics()
    out: list[str] = []
    lines = (markdown or "").split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\r")
        if _TABLE_OPEN_RE.match(raw):
            block = [raw]
            while not _TABLE_CLOSE_RE.search(block[-1]) and i + 1 < len(lines):
                i += 1
                block.append(lines[i].rstrip("\r"))
            i += 1
            unwrapped = _adapt_table("\n".join(block), diag)
            if unwrapped is None:
                out.extend(block)  # a genuine grid stays a table
            else:
                out.extend(unwrapped)
                diag["md_adapted_lines"] += len(block)
            continue
        # a blank line with its caption printed under it (`[______]` then `(Signature)`) is
        # one field captioned by that line
        if _BLANK_LINE_RE.match(raw.strip()) and i + 1 < len(lines):
            caption = _UNDER_CAPTION_RE.match(lines[i + 1].strip())
            if caption is not None:
                out.append(f"- **({caption.group('caption').strip()})** [blank]")
                diag["md_adapted_lines"] += 1
                diag["md_under_captions"] += 1
                i += 2
                continue
        adapted = _adapt_line(raw, diag)
        if adapted != [raw]:
            diag["md_adapted_lines"] += 1
        out.extend(adapted)
        i += 1
    return "\n".join(out), diag


def adapter_summary(diag: dict[str, Any]) -> str | None:
    """One detail line for a metric row, or None when nothing was rewritten."""
    n = diag.get("md_adapted_lines", 0)
    if not n:
        return None
    parts = [
        f"{diag.get(k, 0)} {label}"
        for k, label in (
            ("md_key_lines", "key paragraphs"),
            ("md_split_pairs", "pairs split off shared lines"),
            ("md_id_markers", "printed-id markers"),
            ("md_box_groups", "box runs grouped"),
            ("md_box_moved", "trailing boxes moved first"),
            ("md_numbered_items", "numbered items"),
            ("md_plain_fields", "unbolded blank fields"),
            ("md_under_captions", "captions under a line"),
            ("md_tables_unwrapped", "form tables read as fields"),
        )
        if diag.get(k, 0)
    ]
    return f"{n} page-markdown lines read through the markdown-dialect adapter ({', '.join(parts)})"
