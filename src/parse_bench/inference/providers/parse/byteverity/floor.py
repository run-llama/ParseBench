"""Deterministic PDF floor for synthesa-codex.

Everything here is read from the PDF itself (PyMuPDF), never from the model:
word geometry for grounding, and the text-layer recall signal the routing
oracle consumes. All boxes are on the 0-1000 page grid used by the layout
prompt, so VLM proposals and PDF facts are directly comparable.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .rules import rules as _rules

_TOKEN_RE = re.compile(r"[0-9a-z]+")
_TAG_RE = re.compile(r"<[^>]+>")


def norm_tokens(text: str) -> list[str]:
    text = _TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text).lower()
    return _TOKEN_RE.findall(text)


def page_words(page: Any) -> list[dict[str, Any]]:
    """Words on the 0-1000 grid with their normalized tokens."""
    W, H = page.rect.width, page.rect.height
    out = []
    for x0, y0, x1, y1, w, b, li, _n in page.get_text("words"):
        toks = norm_tokens(w)
        if not toks:
            continue
        out.append(
            {
                "bbox": [x0 / W * 1000, y0 / H * 1000, x1 / W * 1000, y1 / H * 1000],
                "toks": toks,
                "block": b,
                "line": li,
            }
        )
    return out


def text_layer_class(words: list[dict[str, Any]], page: Any) -> str:
    """none / sparse / full — is there a trustworthy text layer?

    A scanned page often carries an invisible OCR layer of junk or nothing at
    all; we only call it 'full' when it has a reasonable number of real words.
    """
    n = len(words)
    if n == 0:
        return "none"
    real = sum(1 for w in words if any(len(t) >= 3 and t.isalpha() for t in w["toks"]))
    if n < 15 or real / n < 0.4:
        return "sparse"
    return "full"


def recall_class(words: list[dict[str, Any]], vlm_text: str) -> str:
    """Share of text-layer tokens that appear in the VLM transcription."""
    if not words:
        return "na"
    layer = [t for w in words for t in w["toks"] if len(t) >= 2]
    if len(layer) < 20:
        return "na"
    have = set(norm_tokens(vlm_text))
    r = sum(1 for t in layer if t in have) / len(layer)
    if r >= 0.9:
        return "high"
    if r >= 0.7:
        return "medium"
    return "low"


def _inside(c: tuple[float, float], bb: list[float], pad: float) -> bool:
    return bb[0] - pad <= c[0] <= bb[2] + pad and bb[1] - pad <= c[1] <= bb[3] + pad


def snap_items(
    items: list[dict[str, Any]], words: list[dict[str, Any]], pad: float = 20.0, min_cover: float = 0.5
) -> tuple[list[dict[str, Any]], int]:
    """Replace each VLM box with the union of the PDF words that make up its text.

    A word is claimed by a block only if (a) its centre lies inside the VLM
    box grown by *pad* and (b) its token occurs in the block's text. The box
    is replaced only when the claimed words cover at least *min_cover* of the
    block's tokens — otherwise the VLM box stands (fail-safe: no evidence, no
    change). Pictures/tables keep their own extent but are grown to include
    claimed words.
    """
    used: set[int] = set()
    snapped = 0
    out = []
    for it in items:
        bb = it.get("bbox")
        label = (it.get("label") or "").lower()
        toks = norm_tokens(it.get("text", ""))
        if not (isinstance(bb, list) and len(bb) == 4) or not toks:
            out.append(it)
            continue
        want = set(toks)
        claim = []
        for i, w in enumerate(words):
            if i in used:
                continue
            c = ((w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2)
            if _inside(c, bb, pad) and any(t in want for t in w["toks"]):
                claim.append(i)
        covered = {t for i in claim for t in words[i]["toks"]} & want
        if claim and len(covered) / max(1, len(want)) >= min_cover:
            xs0 = [words[i]["bbox"][0] for i in claim]
            ys0 = [words[i]["bbox"][1] for i in claim]
            xs1 = [words[i]["bbox"][2] for i in claim]
            ys1 = [words[i]["bbox"][3] for i in claim]
            nb = [min(xs0), min(ys0), max(xs1), max(ys1)]
            if label in ("picture", "figure", "table"):
                nb = [min(nb[0], bb[0]), min(nb[1], bb[1]), max(nb[2], bb[2]), max(nb[3], bb[3])]
            used.update(claim)
            it = {**it, "bbox": nb, "vlm_bbox": bb}
            snapped += 1
        out.append(it)
    return out, snapped


# ---------------------------------------------------------------------------
# Formatting facts from the PDF (font flags, drawn rules, annotations)
# ---------------------------------------------------------------------------

_BOLD_FONT_RE = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)


def _hsegments(page: Any) -> list[tuple[float, float, float, str]]:
    """Horizontal rules drawn on the page: (x0, x1, y, kind) in PDF points."""
    segs: list[tuple[float, float, float, str]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    for d in drawings:
        for it in d.get("items", []):
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) < 1.0 and abs(p1.x - p2.x) > 3:
                    segs.append((min(p1.x, p2.x), max(p1.x, p2.x), (p1.y + p2.y) / 2, "line"))
            elif it[0] == "re":
                r = it[1]
                if r.height < 2.0 and r.width > 3:
                    segs.append((r.x0, r.x1, (r.y0 + r.y1) / 2, "line"))
    # vertical rules: a horizontal rule meeting one at either end is a cell border
    verts: list[tuple[float, float, float]] = []
    for d in drawings:
        for it in d.get("items", []):
            if it[0] == "l" and abs(it[1].x - it[2].x) < 1.0 and abs(it[1].y - it[2].y) > 3:
                verts.append((it[1].x, min(it[1].y, it[2].y), max(it[1].y, it[2].y)))
            elif it[0] == "re" and it[1].width < 2.0 and it[1].height > 3:
                r = it[1]
                verts.append(((r.x0 + r.x1) / 2, r.y0, r.y1))

    def _is_cell_edge(x0: float, x1: float, y: float) -> bool:
        return any(abs(vx - ex) < 2.5 and vy0 - 2.5 <= y <= vy1 + 2.5 for vx, vy0, vy1 in verts for ex in (x0, x1))

    segs = [sg for sg in segs if not _is_cell_edge(sg[0], sg[1], sg[2])]
    for a in page.annots() or []:
        t = a.type[1].lower()
        if t in ("strikeout", "underline", "highlight"):
            for v in (a.vertices and [a.vertices[i : i + 4] for i in range(0, len(a.vertices), 4)]) or [
                [a.rect.tl, a.rect.tr, a.rect.bl, a.rect.br]
            ]:
                xs = [p[0] for p in v]
                ys = [p[1] for p in v]
                segs.append((min(xs), max(xs), (min(ys) + max(ys)) / 2, t))
    return segs


def _highlight_rects(page: Any) -> list[Any]:
    """Filled, coloured (non-grey) rectangles that text sits on — a highlighter mark."""
    out = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return out
    for d in drawings:
        fill = d.get("fill")
        if not fill or d.get("type") not in ("f", "fs"):
            continue
        r, g, b = fill[:3]
        saturated = max(r, g, b) - min(r, g, b) > 0.25 and max(r, g, b) > 0.6
        if saturated and d["rect"].height < 40:
            out.append(d["rect"])
    return out


def styled_runs(page: Any) -> list[dict[str, Any]]:
    """Runs of text with PDF-proven styling: bold / italic / sup / sub / strike / underline / mark."""
    segs = _hsegments(page)
    marks = _highlight_rects(page)
    runs: list[dict[str, Any]] = []
    d = page.get_text("dict")
    R0 = _rules()
    body_size, body_color = 0.0, None
    if R0 is not None:
        import collections as _c

        sz: _c.Counter = _c.Counter()
        col: _c.Counter = _c.Counter()
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    n = len(sp.get("text", "").strip())
                    if n:
                        sz[round(sp["size"], 1)] += n
                        col[sp.get("color", 0)] += n
        if sz:
            tot, acc = sum(sz.values()) / 2, 0
            for k in sorted(sz):
                acc += sz[k]
                if acc >= tot:
                    body_size = k
                    break
            body_color = col.most_common(1)[0][0]
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            base_size = max(s["size"] for s in spans)
            base_y = max(s["bbox"][3] for s in spans)
            for s in spans:
                x0, y0, x1, y1 = s["bbox"]
                h = max(1.0, y1 - y0)
                w = max(1.0, x1 - x0)
                st: set[str] = set()
                if R0 is not None:
                    fn = s.get("font", "")
                    name = (
                        "bold"
                        if _BOLD_FONT_RE.search(fn)
                        else "medium"
                        if re.search(r"medium|bd(?![a-z])", fn, re.IGNORECASE)
                        else "regular"
                    )
                    rs = s["size"] / body_size if body_size else 1.0
                    if (
                        R0.decide(
                            "bold_evidence",
                            flag=bool(s["flags"] & 16),
                            name=name,
                            size="smaller"
                            if rs < 0.95
                            else "body"
                            if rs < 1.15
                            else "larger"
                            if rs < 1.6
                            else "much_larger",
                            colored=s.get("color", 0) != body_color,
                            standalone=len(spans) == 1,
                        )
                        == "bold"
                    ):
                        st.add("bold")
                elif s["flags"] & 16 or _BOLD_FONT_RE.search(s.get("font", "")):
                    st.add("bold")
                if s["flags"] & 2:
                    st.add("italic")
                if s["size"] < base_size * 0.85:
                    if (y1) < base_y - h * 0.25:
                        st.add("sup")
                    elif y0 > base_y - base_size * 0.9 and y1 > base_y + 0.5:
                        st.add("sub")
                R = _rules()
                for sx0, sx1, sy, kind in segs:
                    ov = min(x1, sx1) - max(x0, sx0)
                    if R is not None:
                        rel = (sy - y0) / h
                        pos = (
                            "above"
                            if rel < 0.35
                            else "mid"
                            if rel <= 0.75
                            else "gap"
                            if rel < 0.8
                            else "base"
                            if rel <= 1.25
                            else "below"
                        )
                        k = {
                            "line": "rule",
                            "strikeout": "strike_annot",
                            "underline": "underline_annot",
                            "highlight": "highlight_annot",
                        }[kind]
                        dec = R.decide(
                            "decoration", kind=k, overlap=ov >= 0.6 * w, too_wide=(sx1 - sx0) > 1.6 * w + 12, pos=pos
                        )
                        if dec != "none":
                            st.add(dec)
                        continue
                    if ov < 0.6 * w:
                        continue
                    if kind == "line" and (sx1 - sx0) > 1.6 * w + 12:
                        continue  # a table/cell rule, not a text decoration
                    rel = (sy - y0) / h
                    if kind == "strikeout" or (kind == "line" and 0.35 <= rel <= 0.75):
                        st.add("strike")
                    elif kind == "underline" or (kind == "line" and 0.8 <= rel <= 1.25):
                        st.add("underline")
                    elif kind == "highlight":
                        st.add("mark")
                for r in marks:
                    if r.x0 <= x0 + 1 and r.x1 >= x1 - 1 and r.y0 <= y0 + h * 0.3 and r.y1 >= y1 - h * 0.3:
                        st.add("mark")
                runs.append(
                    {
                        "text": s["text"],
                        "styles": st,
                        "size": s["size"],
                        "bbox": (x0, y0, x1, y1),
                        "line_key": (id(block), id(line)),
                    }
                )
    return runs


def merge_runs(runs: list[dict[str, Any]], style: str) -> list[str]:
    """Contiguous same-line text carrying *style*, merged into phrases."""
    phrases: list[str] = []
    cur: list[str] = []
    cur_key = None
    for r in runs:
        has = style in r["styles"]
        if has and (cur_key is None or r["line_key"] == cur_key):
            cur.append(r["text"])
            cur_key = r["line_key"]
            continue
        if cur:
            phrases.append(" ".join(" ".join(cur).split()))
        cur, cur_key = ([r["text"]], r["line_key"]) if has else ([], None)
    if cur:
        phrases.append(" ".join(" ".join(cur).split()))
    return [p for p in phrases if len(p.strip()) >= 1]


# ---------------------------------------------------------------------------
# Inject PDF-proven markup into the proposer's text (prose only, never tables)
# ---------------------------------------------------------------------------

_TABLE_SPLIT_RE = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)


def _phrase_re(phrase: str) -> re.Pattern[str] | None:
    words = phrase.split()
    if not words:
        return None
    parts = [re.escape(w) for w in words]
    return re.compile(r"(?<![\w*~])" + r"[ \t]+".join(parts) + r"(?![\w*~])")


def _wrap_outside_tables(
    text: str, pat: re.Pattern[str], left: str, right: str, already: re.Pattern[str]
) -> tuple[str, int]:
    n = 0
    chunks = _TABLE_SPLIT_RE.split(text)
    for i, ch in enumerate(chunks):
        if i % 2 == 1:
            continue  # table chunk

        def _sub(m: re.Match[str], ch: str = ch) -> str:
            nonlocal n
            # skip if this occurrence is already styled
            pre = ch[max(0, m.start() - 4) : m.start()]
            if already.search(pre + "\0"):
                return m.group(0)
            n += 1
            return f"{left}{m.group(0)}{right}"

        chunks[i] = pat.sub(_sub, ch, count=1)
    return "".join(chunks), n


_ALREADY = {
    "bold": re.compile(r"(\*\*|<b>|<strong>)\0$"),
    "strike": re.compile(r"(~~|<s>|<del>)\0$"),
}


def inject_markup(
    items: list[dict[str, Any]], runs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Wrap PDF-bold / strike phrases, and sup/sub glyph runs, inside matching blocks."""
    stats = {"bold": 0, "strike": 0, "sup": 0, "sub": 0}
    out = [dict(it) for it in items]
    for style, (li, r) in (("bold", ("**", "**")), ("strike", ("~~", "~~"))):
        for phrase in merge_runs(runs, style):
            if len(phrase) < 2 or not any(c.isalnum() for c in phrase):
                continue
            pat = _phrase_re(phrase)
            if pat is None:
                continue
            R = _rules()
            for it in out:
                lab = (it.get("label") or "").lower()
                if R is not None:
                    blk = (
                        "table_picture_formula"
                        if lab in ("table", "picture", "formula")
                        else "heading"
                        if lab in ("title", "section-header")
                        else "other"
                    )
                    if blk == "table_picture_formula":
                        if R.decide("markup_inject", style=style, block=blk, already=False) == "skip_block":
                            continue
                    txt = it.get("text", "")
                    if not pat.search(txt):
                        continue
                    act = R.decide("markup_inject", style=style, block=blk, already=False)
                    if act == "skip_block":
                        continue
                    if act == "wrap":
                        new, k = _wrap_outside_tables(txt, pat, li, r, _ALREADY[style])
                        if k:
                            it["text"] = new
                            stats[style] += k
                    break
                if lab in ("table", "picture", "formula"):
                    continue
                txt = it.get("text", "")
                if pat.search(txt):
                    # headings already count as bold; don't double-mark them
                    if style == "bold" and lab in ("title", "section-header"):
                        break
                    new, k = _wrap_outside_tables(txt, pat, li, r, _ALREADY[style])
                    if k:
                        it["text"] = new
                        stats[style] += k
                    break
    # super/subscript: anchor on the preceding glyphs of the same line
    prev = None
    for rr in runs:
        for style in ("sup", "sub"):
            if style in rr["styles"] and prev is not None and prev["line_key"] == rr["line_key"]:
                g = rr["text"].strip()
                anchor = (prev["text"].rstrip().split() or [""])[-1][-12:]
                if not g or not anchor or len(g) > 12:
                    continue
                pat = re.compile(re.escape(anchor) + r"\s?" + re.escape(g) + r"(?![\w<])")
                R = _rules()
                for it in out:
                    lab = (it.get("label") or "").lower()
                    if R is not None:
                        blk = (
                            "table_picture_formula"
                            if lab in ("table", "picture", "formula")
                            else "heading"
                            if lab in ("title", "section-header")
                            else "other"
                        )
                        if R.decide("markup_inject", style=style, block=blk, already=False) != "wrap":
                            continue
                    elif lab in ("table", "picture", "formula"):
                        continue
                    txt = it.get("text", "")
                    m = pat.search(txt)
                    if m and f"<{style}>" not in txt[m.start() : m.end() + 6]:
                        it["text"] = txt[: m.start()] + anchor + f"<{style}>{g}</{style}>" + txt[m.end() :]
                        stats[style] += 1
                        break
        if not ({"sup", "sub"} & rr["styles"]):
            prev = rr
    return out, stats


# ---------------------------------------------------------------------------
# Grounding from the PDF: exact segments, classed by the proposer's regions
# ---------------------------------------------------------------------------

_LIST_RE = re.compile(r"^\s*([•\-–·▪■●◦]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s")
_BOLD_NAME_RE = re.compile(r"bold|black|semibold|heavy", re.IGNORECASE)


def _line_text(li: dict[str, Any]) -> str:
    return "".join(s["text"] for s in li["spans"])


def _line_bold(li: dict[str, Any]) -> bool:
    sp = [s for s in li["spans"] if s["text"].strip()]
    return bool(sp) and all((s["flags"] & 16) or _BOLD_NAME_RE.search(s["font"]) for s in sp)


def _line_size(li: dict[str, Any]) -> float:
    return max((s["size"] for s in li["spans"] if s["text"].strip()), default=0.0)


def pdf_segments(page: Any) -> list[dict[str, Any]]:
    """Paragraph-level segments: PDF blocks split at list markers, bold/regular and size changes, big gaps."""
    W, H = page.rect.width, page.rect.height
    out: list[dict[str, Any]] = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines = [li for li in b["lines"] if _line_text(li).strip()]
        if not lines:
            continue
        groups: list[list[dict[str, Any]]] = []
        cur = [lines[0]]
        R = _rules()
        for prev, li in zip(lines, lines[1:], strict=False):
            gap = li["bbox"][1] - prev["bbox"][3]
            if R is not None:
                ps = _line_size(prev)
                ratio = gap / ps if ps > 0 else (float("inf") if gap > 0 else 0.0)
                b = R.decide(
                    "line_boundary",
                    list_marker=bool(_LIST_RE.match(_line_text(li))),
                    bold_change=_line_bold(li) != _line_bold(prev),
                    size_jump=abs(_line_size(li) - ps) > 1,
                    gap="tight" if ratio < 0.3 else "normal" if ratio <= 0.8 else "wide",
                )
                if b == "split":
                    groups.append(cur)
                    cur = [li]
                else:
                    cur.append(li)
                continue
            if (
                _LIST_RE.match(_line_text(li))
                or _line_bold(li) != _line_bold(prev)
                or abs(_line_size(li) - _line_size(prev)) > 1
                or gap > 0.8 * _line_size(prev)
            ):
                groups.append(cur)
                cur = [li]
            else:
                cur.append(li)
        groups.append(cur)
        for g in groups:
            x0 = min(li["bbox"][0] for li in g)
            y0 = min(li["bbox"][1] for li in g)
            x1 = max(li["bbox"][2] for li in g)
            y1 = max(li["bbox"][3] for li in g)
            out.append(
                {
                    "bbox": [x0 / W * 1000, y0 / H * 1000, x1 / W * 1000, y1 / H * 1000],
                    "text": " ".join(" ".join(_line_text(li).split()) for li in g),
                    "bold": all(_line_bold(li) for li in g),
                    "nlines": len(g),
                    "size": max(_line_size(li) for li in g),
                }
            )
    return out


def _ioa(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


_REGION_LABELS = ("table", "picture", "figure", "page-header", "page-footer")
KEEP_INNER = __import__("os").environ.get("SX_KEEPINNER", "1") == "1"  # admitted by ratchet R6


def grounding_items(vlm_items: list[dict[str, Any]], segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Layout elements = VLM regions for tables/pictures/furniture + PDF segments for everything else."""
    regions = [
        (i, it)
        for i, it in enumerate(vlm_items)
        if (it.get("label") or "").lower() in _REGION_LABELS and isinstance(it.get("bbox"), list)
    ]
    textual = [
        (i, it)
        for i, it in enumerate(vlm_items)
        if (it.get("label") or "").lower() not in _REGION_LABELS and isinstance(it.get("bbox"), list)
    ]
    out: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    import os as _os

    sec_rules = _os.environ.get("SX_SECTIONRULES", "0") == "1"
    # body size = length-weighted median segment font size
    sizes = sorted((sg["size"], len(sg["text"])) for sg in segs)
    half, acc, body = sum(n for _, n in sizes) / 2, 0, 0.0
    for sz, n in sizes:
        acc += n
        if acc >= half:
            body = sz
            break
    for i, it in regions:
        out.append(((i, it["bbox"][1], it["bbox"][0]), it))
    R = _rules()
    for sg in segs:
        sb = sg["bbox"]
        if R is not None:
            inside = [it for _, it in regions if _ioa(sb, it["bbox"]) >= 0.5]
            region = (
                "header_footer"
                if any((it.get("label") or "").lower() in ("page-header", "page-footer") for it in inside)
                else "table_picture"
                if inside
                else "none"
            )
            best, best_ov = None, 0.0
            for i, it in textual:
                ov = _ioa(sb, it["bbox"])
                if ov > best_ov:
                    best, best_ov = (i, it), ov
            order = 10_000.0
            vlm = "none"
            if best is not None and best_ov >= 0.3:
                order = best[0]
                vl = (best[1].get("label") or "text").lower()
                vlm = {"title": "title", "section-header": "section", "text": "text"}.get(vl, "other")
            t = sg["text"].strip()
            toks = t.split()
            numeric = bool(toks) and sum(1 for w in toks if re.fullmatch(r"[\d.,%$€£()+\-–/]+", w)) / len(toks) >= 0.5
            caption = bool(re.match(r"(?i)^(table|figure|fig\.?|chart|exhibit|source|note)s?\s*[\dA-Z]", t))
            rs = sg["text"].rstrip()
            role = R.decide(
                "element_role",
                region=region,
                vlm=vlm,
                bold=bool(sg["bold"]),
                lines="one" if sg["nlines"] == 1 else "two" if sg["nlines"] == 2 else "many",
                short=len(sg["text"].split()) <= 14,
                end_punct="sentence" if rs.endswith((".", ",", ";")) else "colon" if rs.endswith(":") else "none",
                numeric=numeric,
                caption=caption,
                paren=t.startswith("("),
                size="smaller"
                if body and sg["size"] < 0.95 * body
                else "larger"
                if body and sg["size"] >= 1.15 * body
                else "body",
            )
            if role == "drop":
                continue
            label = {"title": "Title", "section": "Section-header", "text": "Text"}.get(role)
            if label is None:  # keep_vlm
                label = best[1].get("label") or "Text" if best is not None else "Text"
            out.append(((order, sb[1], sb[0]), {"bbox": sb, "label": label, "text": sg["text"]}))
            continue
        inside = [it for _, it in regions if _ioa(sb, it["bbox"]) >= 0.5]
        if inside and (
            not KEEP_INNER or any((it.get("label") or "").lower() in ("page-header", "page-footer") for it in inside)
        ):
            continue  # belongs to a header / footer region (or to any region, when KEEP_INNER is off)
        best, best_ov = None, 0.0
        for i, it in textual:
            ov = _ioa(sb, it["bbox"])
            if ov > best_ov:
                best, best_ov = (i, it), ov
        label = "Text"
        order = 10_000.0
        if best is not None and best_ov >= 0.3:
            order = best[0]
            vl = (best[1].get("label") or "text").lower()
            label = {"title": "Title", "section-header": "Section-header"}.get(vl, best[1].get("label") or "Text")
            if vl in ("title", "section-header") and not sg["bold"] and sg["nlines"] > 2:
                label = "Text"  # a paragraph swallowed into a heading box
        if (
            label == "Text"
            and sg["bold"]
            and sg["nlines"] <= 2
            and len(sg["text"].split()) <= 14
            and not sg["text"].rstrip().endswith((".", ",", ";"))
        ):
            label = "Section-header"
        if sec_rules:
            t = sg["text"].strip()
            toks = t.split()
            numeric = toks and sum(1 for w in toks if re.fullmatch(r"[\d.,%$€£()+\-–/]+", w)) / len(toks) >= 0.5
            caption = re.match(r"(?i)^(table|figure|fig\.?|chart|exhibit|source|note)s?\s*[\dA-Z]", t)
            if label in ("Section-header", "Title") and (numeric or caption or t.startswith("(")):
                label = "Text"
            elif (
                label == "Text"
                and body
                and sg["size"] >= 1.15 * body
                and sg["nlines"] <= 2
                and len(toks) <= 14
                and not numeric
                and not caption
                and not t.startswith("(")
                and not t.endswith((".", ",", ";", ":"))
            ):
                label = "Section-header"
        out.append(((order, sb[1], sb[0]), {"bbox": sb, "label": label, "text": sg["text"]}))
    out.sort(key=lambda t: t[0])
    return [it for _, it in out]


def pdf_images(page: Any) -> list[list[float]]:
    """Raster image placements on the 0-1000 grid (tiny/full-page backgrounds dropped)."""
    W, H = page.rect.width, page.rect.height
    out = []
    for im in page.get_image_info():
        x0, y0, x1, y1 = im["bbox"]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        a = (x1 - x0) * (y1 - y0) / (W * H)
        if a < 0.0004 or a > 0.85:
            continue
        out.append([x0 / W * 1000, y0 / H * 1000, x1 / W * 1000, y1 / H * 1000])
    return out


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def reconcile_pictures(items: list[dict[str, Any]], images: list[list[float]]) -> list[dict[str, Any]]:
    """Snap VLM picture boxes to PDF image rects; add images the proposer missed."""
    out = [dict(it) for it in items]
    used = set()
    for it in out:
        if (it.get("label") or "").lower() not in ("picture", "figure") or not isinstance(it.get("bbox"), list):
            continue
        best, bi = 0.0, -1
        for j, im in enumerate(images):
            v = _iou(it["bbox"], im)
            if v > best:
                best, bi = v, j
        R = _rules()
        snap = (
            (
                R.decide(
                    "picture_reconcile",
                    subject="vlm_picture",
                    iou="high" if best >= 0.5 else "low",
                    covered=False,
                    ink_fit="none",
                    text_heavy=False,
                )
                == "snap_to_image"
            )
            if R is not None
            else best >= 0.5
        )
        if snap:
            it["bbox"] = images[bi]
            used.add(bi)
    for j, im in enumerate(images):
        if j in used:
            continue
        covered = any(
            isinstance(it.get("bbox"), list)
            and (_ioa(im, it["bbox"]) > 0.5 or _ioa(it["bbox"], im) > 0.5)
            and (it.get("label") or "").lower() in ("picture", "figure", "table")
            for it in out
        )
        R = _rules()
        add = (
            (
                R.decide(
                    "picture_reconcile",
                    subject="pdf_image",
                    iou="low",
                    covered=bool(covered),
                    ink_fit="none",
                    text_heavy=False,
                )
                == "add_picture"
            )
            if R is not None
            else not covered
        )
        if add:
            out.append({"bbox": im, "label": "Picture", "text": ""})
    return out


# ---------------------------------------------------------------------------
# Graphic-ink components (vector drawings + raster images), on the 0-1000 grid
# ---------------------------------------------------------------------------


def ink_components(page: Any, gap: float = 6.0) -> list[list[float]]:
    """Cluster graphic ink into figure-sized components.

    Dropped before clustering: page/panel backgrounds (a fill covering >25% of the
    page), long thin rules (separators, table borders), and white fills. Rects that
    come within *gap* points of each other merge (union-find over rects).
    """
    W, H = page.rect.width, page.rect.height
    PA = W * H
    rects: list[list[float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    line_centres = []
    for b in page.get_text("dict")["blocks"]:
        for li in b.get("lines", []):
            x0, y0, x1, y1 = li["bbox"]
            line_centres.append(((x0 + x1) / 2, (y0 + y1) / 2))
    for d in drawings:
        r = d["rect"]
        if r.is_empty and (r.width < 0.5 and r.height < 0.5):
            continue
        a = r.width * r.height
        if a > 0.25 * PA:
            continue
        if (
            d.get("fill") is not None
            and a > 0.004 * PA
            and any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for cx, cy in line_centres)
        ):
            continue  # a panel / box behind text, not figure ink
        thin = min(r.width, r.height) < 2.0
        if thin and max(r.width, r.height) > 0.3 * W:
            continue  # separator / rule
        fill = d.get("fill")
        if fill is not None and min(fill[:3]) > 0.97 and d.get("color") is None:
            continue  # white knock-out
        rects.append([r.x0, r.y0, r.x1, r.y1])
    images = []
    for im in page.get_image_info():
        x0, y0, x1, y1 = im["bbox"]
        if (x1 - x0) * (y1 - y0) < 0.85 * PA:
            images.append([max(0, x0), max(0, y0), min(W, x1), min(H, y1)])
    n = len(rects)
    if n > 4000:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: rects[i][0])
    for ai, i in enumerate(order):
        ri = rects[i]
        for j in order[ai + 1 :]:
            rj = rects[j]
            if rj[0] > ri[2] + gap:
                break
            if rj[1] <= ri[3] + gap and ri[1] <= rj[3] + gap:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups: dict[int, list[float]] = {}
    for i in range(n):
        g = find(i)
        r = rects[i]
        if g in groups:
            b = groups[g]
            groups[g] = [min(b[0], r[0]), min(b[1], r[1]), max(b[2], r[2]), max(b[3], r[3])]
        else:
            groups[g] = list(r)
    out = []
    for b in list(groups.values()) + images:  # raster images stand alone: adjacent photos are separate figures
        a = (b[2] - b[0]) * (b[3] - b[1])
        if a < 0.0008 * PA or (b[2] - b[0]) < 8 or (b[3] - b[1]) < 8:
            continue
        out.append([b[0] / W * 1000, b[1] / H * 1000, b[2] / W * 1000, b[3] / H * 1000])
    return out


def reconcile_ink(
    items: list[dict[str, Any]], comps: list[list[float]], segs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Tighten oversized VLM picture boxes to the ink they contain; add ink figures the proposer missed.

    A component that is mostly text (>=60% of its area under PDF text segments) is a
    text panel/table grid, not a figure, and is ignored.
    """

    def text_share(c: list[float]) -> float:
        A = max(1e-9, (c[2] - c[0]) * (c[3] - c[1]))
        s = 0.0
        for sg in segs:
            b = sg["bbox"]
            ix = max(0.0, min(c[2], b[2]) - max(c[0], b[0]))
            iy = max(0.0, min(c[3], b[3]) - max(c[1], b[1]))
            s += ix * iy
        return min(1.0, s / A)

    R = _rules()
    if R is not None:
        comps = [
            c
            for c in comps
            if R.decide(
                "picture_reconcile",
                subject="ink_component",
                iou="low",
                covered=False,
                ink_fit="none",
                text_heavy=text_share(c) >= 0.6,
            )
            == "add_picture"
        ]
    else:
        comps = [c for c in comps if text_share(c) < 0.6]
    out = [dict(it) for it in items]
    used: set[int] = set()
    for it in out:
        if (it.get("label") or "").lower() not in ("picture", "figure") or not isinstance(it.get("bbox"), list):
            continue
        bb = it["bbox"]
        inside = [k for k, c in enumerate(comps) if _ioa(c, bb) >= 0.8]
        if not inside:
            continue
        u = [
            min(comps[k][0] for k in inside),
            min(comps[k][1] for k in inside),
            max(comps[k][2] for k in inside),
            max(comps[k][3] for k in inside),
        ]

        def A(b: list[float]) -> float:
            return max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))

        if R is not None:
            fit = "tight" if A(u) < 0.5 * A(bb) else "loose"
            if (
                R.decide(
                    "picture_reconcile", subject="vlm_picture", iou="low", covered=False, ink_fit=fit, text_heavy=False
                )
                == "shrink_to_ink"
            ):
                it["bbox"] = u
        elif A(u) < 0.5 * A(bb):
            it["bbox"] = u
        used.update(inside)
    import os as _os

    if _os.environ.get("SX_INKALL", "1") == "1":  # admitted by ratchet R7
        for c in comps:
            dup = any(isinstance(it.get("bbox"), list) and _iou(c, it["bbox"]) > 0.9 for it in out)
            add = (
                (
                    R.decide(
                        "picture_reconcile",
                        subject="ink_component",
                        iou="low",
                        covered=dup,
                        ink_fit="none",
                        text_heavy=False,
                    )
                    == "add_picture"
                )
                if R is not None
                else not dup
            )
            if add:
                out.append({"bbox": c, "label": "Picture", "text": ""})
        return out
    for k, c in enumerate(comps):
        if k in used:
            continue
        covered = any(
            isinstance(it.get("bbox"), list)
            and (_ioa(c, it["bbox"]) > 0.5 or _ioa(it["bbox"], c) > 0.5)
            and (it.get("label") or "").lower() in ("picture", "figure", "table")
            for it in out
        )
        if not covered:
            out.append({"bbox": c, "label": "Picture", "text": ""})
    return out


def merged_segments(segs: list[dict[str, Any]], gapk: float = 0.6) -> list[dict[str, Any]]:
    """Coarser candidates: vertically adjacent, column-aligned, same-style non-bold segments merged.

    Emitted *alongside* the fine segments (extra candidates are not penalized; each
    GT element takes its best eligible match), never instead of them.
    """
    out: list[dict[str, Any]] = []
    merged_any: list[bool] = []
    for s in sorted(segs, key=lambda s: (round(s["bbox"][0] / 40), s["bbox"][1])):
        if out:
            p = out[-1]
            pb, sb = p["bbox"], s["bbox"]
            xov = min(pb[2], sb[2]) - max(pb[0], sb[0])
            gap = sb[1] - pb[3]
            lh = (pb[3] - pb[1]) / max(1, p["nlines"])
            if (
                xov > 0.6 * min(pb[2] - pb[0], sb[2] - sb[0])
                and 0 <= gap < gapk * lh
                and p["bold"] == s["bold"]
                and abs(p["size"] - s["size"]) <= 1
                and not p["bold"]
            ):
                p["bbox"] = [min(pb[0], sb[0]), pb[1], max(pb[2], sb[2]), sb[3]]
                p["nlines"] += s["nlines"]
                p["text"] += " " + s["text"]
                merged_any[-1] = True
                continue
        out.append(dict(s, bbox=list(s["bbox"])))
        merged_any.append(False)
    return [s for s, m in zip(out, merged_any, strict=False) if m]


# ---------------------------------------------------------------------------
# Text-layer validity facts (O0) — PDF-only
# ---------------------------------------------------------------------------

_BAD_CHAR_RE = re.compile(r"[-�\x00-\x08\x0b\x0c\x0e-\x1f]")


def text_layer_validity_facts(page: Any) -> dict[str, Any]:
    import collections as _c

    import numpy as np

    W, H = page.rect.width, page.rect.height
    words = page.get_text("words")
    text = "".join(w[4] for w in words)
    bad = (len(_BAD_CHAR_RE.findall(text)) + 5 * text.count("(cid:")) / max(1, len(text))
    alpha0 = tot = 0
    for b in page.get_text("dict")["blocks"]:
        for li in b.get("lines", []):
            for s in li["spans"]:
                k = len(s["text"].strip())
                tot += k
                alpha0 += k if s.get("alpha", 255) == 0 else 0
    keys = _c.Counter((w[4], round(w[0]), round(w[1])) for w in words)
    dup = sum(c - 1 for c in keys.values() if c > 1) / max(1, len(words))
    align = 1.0
    if words:
        pix = page.get_pixmap(dpi=50, alpha=False)
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].min(axis=2)
        sx, sy = pix.width / W, pix.height / H
        hit = cnt = 0
        for i in range(0, len(words), max(1, len(words) // 200)):
            x0, y0, x1, y1 = words[i][:4]
            r = a[max(0, int(y0 * sy)) : int(y1 * sy) + 1, max(0, int(x0 * sx)) : int(x1 * sx) + 1]
            cnt += 1
            hit += bool(r.size) and (r < 160).mean() > 0.03
        align = hit / max(1, cnt)
    return {
        "tl_class": text_layer_class(page_words(page), page),
        "encoding": "clean" if bad < 0.005 else "degraded" if bad < 0.05 else "garbage",
        "visibility": "mostly_invisible" if alpha0 / max(1, tot) > 0.5 else "visible",
        "scan_overlay": any(
            (im["bbox"][2] - im["bbox"][0]) * (im["bbox"][3] - im["bbox"][1]) >= 0.85 * W * H
            for im in page.get_image_info()
        ),
        "alignment": "aligned" if align >= 0.85 else "partial" if align >= 0.6 else "misaligned",
        "duplicated": dup > 0.10,
    }


# ---------------------------------------------------------------------------
# Content from the admitted text layer (O9): word ownership, PDF block text, orphan lines
# ---------------------------------------------------------------------------

_OWNER_LABELS_SELF = ("table", "picture", "figure", "formula", "page-header", "page-footer")


def word_ownership(
    page: Any, items: list[dict[str, Any]], pad: float = 4.0
) -> tuple[dict[int, list[tuple]], list[tuple]]:
    """Assign each PDF word to the smallest item box containing its centre. Returns ({item_idx: words}, orphans).

    Words are raw PyMuPDF tuples (x0,y0,x1,y1,text,block,line,wordno) with coords on the 0-1000 grid.
    """
    W, H = page.rect.width, page.rect.height
    owned: dict[int, list[tuple]] = {}
    orphans: list[tuple] = []
    boxes = [(i, it["bbox"]) for i, it in enumerate(items) if isinstance(it.get("bbox"), list) and len(it["bbox"]) == 4]
    for w in page.get_text("words", sort=False):
        x0, y0, x1, y1 = w[0] / W * 1000, w[1] / H * 1000, w[2] / W * 1000, w[3] / H * 1000
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best, area = None, None
        for i, b in boxes:
            if b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad:
                a = (b[2] - b[0]) * (b[3] - b[1])
                if area is None or a < area:
                    best, area = i, a
        t = (x0, y0, x1, y1, w[4], w[5], w[6], w[7])
        if best is None:
            orphans.append(t)
        else:
            owned.setdefault(best, []).append(t)
    return owned, orphans


def words_to_text(words: list[tuple]) -> str:
    """PDF reading order inside a block (block, line, word); lines joined with de-hyphenation."""
    lines: dict[tuple[int, int], list[tuple]] = {}
    for w in sorted(words, key=lambda w: (w[5], w[6], w[7])):
        lines.setdefault((w[5], w[6]), []).append(w)
    out = ""
    for ws in lines.values():
        ln = " ".join(w[4] for w in ws)
        if out.endswith("-") and ln[:1].islower() and len(out) > 1 and out[-2].isalpha():
            out = out[:-1] + ln
        else:
            out = (out + " " + ln) if out else ln
    return out


def label_class(label: str) -> str:
    li = (label or "").lower()
    if li in ("title", "section-header", "section_header"):
        return "heading"
    if li in ("text",):
        return "prose"
    if li in ("list-item", "list"):
        return "list"
    if li == "caption":
        return "caption"
    if li == "footnote":
        return "footnote"
    return "other"


def agreement_band(vlm_text: str, pdf_text: str) -> str:
    a, b = set(norm_tokens(vlm_text)), set(norm_tokens(pdf_text))
    if not b:
        return "empty"
    j = len(a & b) / max(1, len(a | b))
    return "high" if j >= 0.9 else "mid" if j >= 0.6 else "low"


def orphan_items(orphans: list[tuple], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group orphan words by PDF (block, line) runs into Text blocks.

    Each block is inserted before the first same-column item below it.
    """
    by_block: dict[int, list[tuple]] = {}
    for w in orphans:
        by_block.setdefault(w[5], []).append(w)
    out = [dict(it) for it in items]
    for ws in sorted(by_block.values(), key=lambda ws: (min(w[1] for w in ws), min(w[0] for w in ws))):
        txt = words_to_text(ws)
        if len(norm_tokens(txt)) < 3:
            continue  # stray page numbers / marks
        bb = [min(w[0] for w in ws), min(w[1] for w in ws), max(w[2] for w in ws), max(w[3] for w in ws)]
        pos = len(out)
        for k, it in enumerate(out):
            b = it.get("bbox")
            if isinstance(b, list) and b[1] >= bb[3] - 2 and min(b[2], bb[2]) - max(b[0], bb[0]) > 0:
                pos = k
                break
        out.insert(pos, {"bbox": bb, "label": "Text", "text": txt, "orphan": True})
    return out


# ---------------------------------------------------------------------------
# Raster floor (O10): ink lines from the rendered page, for pages without text-layer geometry
# ---------------------------------------------------------------------------


def page_ink(page: Any, dpi: int = 100) -> tuple[Any, str]:
    """Binary ink mask (dark pixels) and the page's ink-coverage band."""
    import numpy as np

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    g = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].min(axis=2)
    mask = g < 150
    cov = float(mask.mean())
    band = "none" if cov < 0.005 else "light" if cov < 0.03 else "normal" if cov < 0.25 else "heavy"
    return mask, band


def _ink_lines(mask: Any, x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    """Text lines inside a pixel box: runs of ink rows (1-row gaps bridged), each with its ink x-extent."""
    import numpy as np

    sub = mask[y0:y1, x0:x1]
    if sub.size == 0:
        return []
    rows = sub.sum(axis=1) > max(2, 0.01 * sub.shape[1])
    lines, start, gap = [], None, 0
    for r, on in enumerate(list(rows) + [False, False]):
        if on:
            if start is None:
                start = r
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 1:
                end = r - gap + 1
                if end - start >= 4:
                    cols = np.where(sub[start:end].any(axis=0))[0]
                    if cols.size:
                        lines.append([x0 + int(cols[0]), y0 + start, x0 + int(cols[-1]) + 1, y0 + end])
                start, gap = None, 0
    return lines


def _split_text(text: str, weights: list[float]) -> list[str]:
    k = len(weights)
    if k == 1:
        return [text]
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) == k:
        return parts
    words = text.split()
    tot = sum(weights) or 1.0
    out, i = [], 0
    for j, w in enumerate(weights):
        n = len(words) - i if j == k - 1 else round(len(words) * w / tot)
        out.append(" ".join(words[i : i + n]))
        i += n
    return out


def raster_segments(page: Any, items: list[dict[str, Any]], mask: Any, R: Any) -> list[dict[str, Any]]:
    """Grounding elements from ink: textual VLM blocks split into ink paragraphs (O1 decides boundaries),
    region boxes tightened to their ink. Text comes from the proposer, split across paragraphs."""
    Hp, Wp = mask.shape
    sx, sy = Wp / 1000.0, Hp / 1000.0
    out: list[dict[str, Any]] = []
    for it in items:
        bb = it.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        px0 = max(0, int(bb[0] * sx) - 6)
        py0 = max(0, int(bb[1] * sy) - 6)
        px1 = min(Wp, int(bb[2] * sx) + 6)
        py1 = min(Hp, int(bb[3] * sy) + 6)
        lines = _ink_lines(mask, px0, py0, px1, py1)
        if not lines:
            out.append(it)
            continue

        def to1000(b: list[float]) -> list[float]:
            return [b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy]

        lab = (it.get("label") or "").lower()
        if lab in _REGION_LABELS or lab == "formula":
            u = [
                min(li[0] for li in lines),
                min(li[1] for li in lines),
                max(li[2] for li in lines),
                max(li[3] for li in lines),
            ]
            out.append({**it, "bbox": to1000(u)})
            continue
        if __import__("os").environ.get("SX_PIXSPLIT", "1") == "0":
            u = [
                min(li[0] for li in lines),
                min(li[1] for li in lines),
                max(li[2] for li in lines),
                max(li[3] for li in lines),
            ]
            out.append({**it, "bbox": to1000(u)})
            continue
        groups, cur = [], [lines[0]]
        for prev, li in zip(lines, lines[1:], strict=False):
            h = max(1, prev[3] - prev[1])
            ratio = (li[1] - prev[3]) / h
            b = R.decide(
                "line_boundary",
                list_marker=False,
                bold_change=False,
                size_jump=abs((li[3] - li[1]) - h) > 0.3 * h,
                gap="tight" if ratio < 0.3 else "normal" if ratio <= 0.8 else "wide",
            )
            if b == "split":
                groups.append(cur)
                cur = [li]
            else:
                cur.append(li)
        groups.append(cur)
        texts = _split_text(it.get("text", ""), [sum(li[2] - li[0] for li in g) for g in groups])
        for g, t in zip(groups, texts, strict=False):
            u = [min(li[0] for li in g), min(li[1] for li in g), max(li[2] for li in g), max(li[3] for li in g)]
            out.append({"bbox": to1000(u), "label": it.get("label") or "Text", "text": t})
    return out


# ---------------------------------------------------------------------------
# Deterministic tables from ruling lines (O12)
# ---------------------------------------------------------------------------


def _cell_text(page: Any, bbox: tuple) -> str:
    import html as _h

    import pymupdf

    t = page.get_text("text", clip=pymupdf.Rect(bbox)) if bbox else ""
    return _h.escape(" ".join(t.split()))


def pdf_tables(page: Any) -> list[dict[str, Any]]:
    """Tables found from ruling lines, as HTML with row/colspans rebuilt from cell geometry.

    Returns [{bbox (0-1000), html, rows, cols, text}], the first grid row emitted as <thead>.
    """
    W, H = page.rect.width, page.rect.height
    out: list[dict[str, Any]] = []
    try:
        found = page.find_tables()
        tables = list(found.tables)
        if not tables and __import__("os").environ.get("SX_TABLE_TEXT", "0") == "1":
            tables = list(page.find_tables(strategy="text").tables)  # borderless tables by text alignment
    except Exception:
        return out
    for t in tables:
        cells = [c for c in (t.cells or []) if c]
        if not cells:
            continue
        xs = sorted({round(c[0], 1) for c in cells} | {round(c[2], 1) for c in cells})
        ys = sorted({round(c[1], 1) for c in cells} | {round(c[3], 1) for c in cells})

        def idx(v: float, grid: list[float]) -> int:
            return min(range(len(grid)), key=lambda i: abs(grid[i] - v))

        ncol, nrow = len(xs) - 1, len(ys) - 1
        if ncol < 1 or nrow < 1:
            continue
        occ = [[None] * ncol for _ in range(nrow)]
        for c in cells:
            c0, c1 = idx(c[0], xs), idx(c[2], xs)
            r0, r1 = idx(c[1], ys), idx(c[3], ys)
            if c1 <= c0 or r1 <= r0 or occ[r0][c0] is not None:
                continue
            for r in range(r0, r1):
                for cc in range(c0, c1):
                    occ[r][cc] = "x"
            occ[r0][c0] = {"rs": r1 - r0, "cs": c1 - c0, "text": _cell_text(page, c)}
        rows_html = []
        for r in range(nrow):
            tag = "th" if r == 0 else "td"
            cells_html = []
            for cc in range(ncol):
                v = occ[r][cc]
                if isinstance(v, dict):
                    attrs = (f' rowspan="{v["rs"]}"' if v["rs"] > 1 else "") + (
                        f' colspan="{v["cs"]}"' if v["cs"] > 1 else ""
                    )
                    cells_html.append(f"<{tag}{attrs}>{v['text']}</{tag}>")
                elif v is None:
                    cells_html.append(f"<{tag}></{tag}>")
            rows_html.append("<tr>" + "".join(cells_html) + "</tr>")
        html = "<table><thead>" + rows_html[0] + "</thead><tbody>" + "".join(rows_html[1:]) + "</tbody></table>"
        x0, y0, x1, y1 = t.bbox
        out.append(
            {
                "bbox": [x0 / W * 1000, y0 / H * 1000, x1 / W * 1000, y1 / H * 1000],
                "html": html,
                "rows": nrow,
                "cols": ncol,
                "text": " ".join(str(v["text"]) for row in occ for v in row if isinstance(v, dict)),
            }
        )
    return out


_ROW_RE = re.compile(r"<tr[^>]*>([\s\S]*?)</tr>", re.IGNORECASE)
_CELL_HTML_RE = re.compile(r"<t[hd][^>]*>", re.IGNORECASE)


def html_table_shape(html: str) -> tuple[int, int]:
    rows = _ROW_RE.findall(html)
    return len(rows), max((len(_CELL_HTML_RE.findall(r)) for r in rows), default=0)


def html_grid_consistent(html: str) -> bool:
    """Every row spans the same number of columns once rowspan/colspan are applied."""
    rows = _ROW_RE.findall(html)
    if not rows:
        return False
    carry: dict[int, int] = {}  # column -> remaining rowspan
    widths = []
    for r in rows:
        col, w = 0, 0
        for m in re.finditer(r"<t[hd]([^>]*)>", r, re.IGNORECASE):
            while carry.get(col, 0) > 0:
                carry[col] -= 1
                col += 1
                w += 1
            a = m.group(1)
            cs = int(re.search(r'colspan="?(\d+)', a).group(1)) if re.search(r'colspan="?(\d+)', a) else 1
            rs = int(re.search(r'rowspan="?(\d+)', a).group(1)) if re.search(r'rowspan="?(\d+)', a) else 1
            for c in range(col, col + cs):
                if rs > 1:
                    carry[c] = rs - 1
            col += cs
            w += cs
        while carry.get(col, 0) > 0:
            carry[col] -= 1
            col += 1
            w += 1
        widths.append(w)
    return len(set(widths)) == 1


# ---------------------------------------------------------------------------
# Layout evidence from an open-source detector (docling-layout-heron, Apache-2.0)
# ---------------------------------------------------------------------------

DET_LABEL = {
    "caption": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "list_item": "List-item",
    "page_footer": "Page-footer",
    "page_header": "Page-header",
    "picture": "Picture",
    "section_header": "Section-header",
    "table": "Table",
    "text": "Text",
    "title": "Title",
    "document_index": "Table",
    "code": "Text",
    "checkbox_selected": "Text",
    "checkbox_unselected": "Text",
    "form": "Table",
    "key_value_region": "Text",
}


def detector_items(
    page: Any, dets: list[dict[str, Any]], vlm_items: list[dict[str, Any]], authority: str, min_score: float = 0.5
) -> list[dict[str, Any]]:
    """Grounding elements = detector boxes/classes. Text: PDF words owned by each box (trusted layer),
    else the proposer's blocks whose centre falls in the box. Order follows the proposer's reading order."""
    boxes = [d for d in dets if d["score"] >= min_score]
    items = [{"bbox": d["bbox"], "label": DET_LABEL.get(d["label"], "Text"), "text": ""} for d in boxes]
    if not items:
        return []
    if authority in ("full_authority", "geometry_only"):
        owned, _ = word_ownership(page, items, pad=2.0)
        for k, it in enumerate(items):
            if it["label"] not in ("Picture",):
                it["text"] = words_to_text(owned.get(k, []))
    else:
        for it in items:
            b = it["bbox"]
            parts = [
                v.get("text", "")
                for v in vlm_items
                if isinstance(v.get("bbox"), list)
                and b[0] <= (v["bbox"][0] + v["bbox"][2]) / 2 <= b[2]
                and b[1] <= (v["bbox"][1] + v["bbox"][3]) / 2 <= b[3]
            ]
            it["text"] = " ".join(parts)
    for it in items:  # tables keep the proposer's HTML when they overlap one
        if it["label"] == "Table":
            for v in vlm_items:
                if (
                    (v.get("label") or "").lower() == "table"
                    and isinstance(v.get("bbox"), list)
                    and _iou(v["bbox"], it["bbox"]) >= 0.3
                ):
                    it["text"] = v.get("text", "")
                    break

    def order(it: dict[str, Any]) -> tuple:
        best, bo = 10_000, 0.0
        for k, v in enumerate(vlm_items):
            if isinstance(v.get("bbox"), list):
                o = _ioa(it["bbox"], v["bbox"])
                if o > bo:
                    best, bo = k, o
        return (best if bo >= 0.3 else 10_000, it["bbox"][1], it["bbox"][0])

    return sorted(items, key=order)
