"""Page-decoration rule: running header, running footer and printed page number, per page.

LlamaParse lifts page furniture out of the body markdown into per-page structured fields
(``page_header_markdown``, ``page_footer_markdown``, ``printed_page_number``); the prompts ask
for ``<page_header>`` / ``<page_footer>`` / ``<page_number>`` tags which post-processing removes.
This rule scores that contract on one page with four axes:

* ``header`` / ``footer`` — predicted text vs annotated text after normalisation (markdown and
  punctuation stripped, whitespace collapsed, lowercase) with ``token_set_ratio`` so that the
  left / centre / right pieces of a header may come in any order. An annotated ``None`` means
  the slot must be empty: a hallucinated header on a cover page fails.
* ``page_number`` — both sides canonicalised (arabic digits, lowercase roman, ``Page 3 of 10`` →
  ``3``, ``– 3 –`` → ``3``, compound ids such as ``A-3`` kept) and compared exactly; a number returned
  inside the header/footer text instead of the dedicated field also counts.
* ``leak`` — none of the annotated furniture strings may remain in the body markdown; the body
  is the page markdown with any ``<page_*>`` tags removed.

Prediction source: the structured page fields when ``parse_output.layout_pages`` is available,
else the markdown tags — the same preference order as the ``is_header`` / ``is_footer`` rules.
"""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
from rapidfuzz import fuzz

from parse_bench.evaluation.metrics.parse.rules_base import ParseTestRule
from parse_bench.evaluation.metrics.parse.test_types import TestType
from parse_bench.evaluation.metrics.parse.utils import normalize_text
from parse_bench.test_cases.parse_rule_schemas import ParsePageDecorationRule

_TAG_RE = {
    "header": re.compile(r"<page_header>(.*?)</page_header>", re.S | re.I),
    "footer": re.compile(r"<page_footer>(.*?)</page_footer>", re.S | re.I),
    "page_number": re.compile(r"<page_number>(.*?)</page_number>", re.S | re.I),
}
_ANY_TAG_RE = re.compile(r"</?page_(?:header|footer|number)>", re.I)
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def norm_furniture(text: str | None) -> str:
    """Comparison form of a header/footer: bench text normalisation, no punctuation, one space."""
    s = normalize_text(text or "").lower()
    s = re.sub(r"[|•·–—\-_/\\,:;.()\[\]{}\"'`*#>]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or not _ROMAN_RE.match(s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN_VALUES[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def canonical_page_number(raw: str | None) -> str | None:
    """``Page 12 of 48`` → ``12``; ``– iv –`` → ``iv``; ``A-3`` → ``a-3``; ``3 / 10`` → ``3``."""
    if raw is None:
        return None
    s = normalize_text(str(raw)).strip().lower()
    s = re.sub(r"^(page|p\.?|pg\.?|seite|página|pagina)\s*", "", s)
    s = re.sub(r"\s*(of|/|sur|de|von)\s*\d+\s*$", "", s)
    s = s.strip(" -–—|.·•")
    if not s:
        return None
    m = re.fullmatch(r"0*(\d+)", s)
    if m:
        return str(int(m.group(1)))
    if _ROMAN_RE.match(s):
        return s
    m = re.fullmatch(r"([a-z]{1,3})[\s\-–.]?0*(\d+)", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2))}"
    return s


def page_number_equal(a: str | None, b: str | None) -> bool:
    ca, cb = canonical_page_number(a), canonical_page_number(b)
    if ca is None or cb is None:
        return ca == cb
    if ca == cb:
        return True
    ra, rb = roman_to_int(ca), roman_to_int(cb)
    # ``iv`` printed vs ``4`` predicted (or the reverse) counts as the same number.
    return ra is not None and str(ra) == cb or rb is not None and str(rb) == ca


def _page_markdown(md_content: str, page: int | None) -> str:
    if page is None or "\f" not in md_content:
        return md_content
    parts = md_content.split("\f")
    if 1 <= page <= len(parts):
        return parts[page - 1]
    return md_content


class PageDecorationRule(ParseTestRule):
    """Header / footer / printed page number for one page, plus leakage into the body."""

    def __init__(self, rule_data: ParsePageDecorationRule | dict):
        super().__init__(rule_data)
        if self.type != TestType.PAGE_DECORATION.value:
            raise ValueError(f"Invalid type for PageDecorationRule: {self.type}")
        self.rule = cast(ParsePageDecorationRule, self._rule_data)

    def _predicted(self, page_md: str) -> tuple[dict[str, str | None], str]:
        """Predicted slots and the source they came from (``structured`` or ``markdown_tags``)."""
        if self.parse_output is not None and self.parse_output.layout_pages:
            pages = self.parse_output.layout_pages
            if self.page is not None:
                pages = [p for p in pages if p.page_number == self.page] or pages[:1]
            if pages:
                p = pages[0]
                return {
                    "header": p.page_header_markdown or None,
                    "footer": p.page_footer_markdown or None,
                    "page_number": p.printed_page_number or None,
                }, "structured"
        out: dict[str, str | None] = {}
        for slot, pattern in _TAG_RE.items():
            found = [m.group(1).strip() for m in pattern.finditer(page_md)]
            out[slot] = " | ".join(f for f in found if f) or None
        return out, "markdown_tags"

    def _text_axis(
        self, expected: str | None, predicted: str | None, threshold: int, ignore: set[str] | None = None
    ) -> dict[str, Any]:
        ne, npd = norm_furniture(expected), norm_furniture(predicted)
        if ignore:
            # The printed page number may sit in the header/footer line on either side; it is scored by
            # its own axis and must not cost precision or recall here.
            ne = " ".join(t for t in ne.split() if t not in ignore)
            npd = " ".join(t for t in npd.split() if t not in ignore)
        if not ne and not npd:
            return {
                "passed": True,
                "score": 1.0,
                "expected": expected,
                "predicted": predicted,
                "reason": "none expected, none predicted",
            }
        if not ne:
            return {
                "passed": False,
                "score": 0.0,
                "expected": expected,
                "predicted": predicted,
                "reason": "hallucinated",
            }
        if not npd:
            return {"passed": False, "score": 0.0, "expected": expected, "predicted": predicted, "reason": "missing"}
        # Token-level F1 with fuzzy token matching: order-free (header pieces may be reordered), but a
        # dropped piece lowers recall and body text swept into the header lowers precision. A plain
        # token_set_ratio would score a strict subset 100 and hide missing pieces.
        exp_tokens, pred_tokens = ne.split(), npd.split()
        matched_pred: set[int] = set()
        hits = 0
        for tok in exp_tokens:
            best, best_j = 0.0, -1
            for j, cand in enumerate(pred_tokens):
                if j in matched_pred:
                    continue
                score = 100.0 if tok == cand else float(fuzz.ratio(tok, cand))
                if score > best:
                    best, best_j = score, j
            if best >= 85 and best_j >= 0:
                matched_pred.add(best_j)
                hits += 1
        recall = hits / len(exp_tokens)
        precision = hits / len(pred_tokens) if pred_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        # Both directions must clear the threshold: F1 alone would forgive one dropped piece in four.
        return {
            "passed": recall * 100 >= threshold and precision * 100 >= threshold,
            "score": round(f1, 4),
            "expected": expected,
            "predicted": predicted,
            "recall": round(recall, 3),
            "precision": round(precision, 3),
        }

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        page_md = _page_markdown(md_content, self.page)
        predicted, source = self._predicted(page_md)
        r = self.rule
        page_tokens = {
            t for t in (canonical_page_number(r.page_number), canonical_page_number(predicted["page_number"])) if t
        }
        axes: dict[str, dict[str, Any]] = {
            "header": self._text_axis(r.header, predicted["header"], r.text_threshold, page_tokens),
            "footer": self._text_axis(r.footer, predicted["footer"], r.text_threshold, page_tokens),
        }
        pn_field_ok = page_number_equal(r.page_number, predicted["page_number"])
        # A page number printed inside the running header or footer is legitimately returned as part of
        # that text; the dedicated field is preferred but not required. It counts when the canonical
        # number appears as a whole token in the predicted header/footer.
        in_furniture = False
        if not pn_field_ok and r.page_number and not predicted["page_number"]:
            furniture = norm_furniture(" ".join(v for v in (predicted["header"], predicted["footer"]) if v))
            canon = canonical_page_number(r.page_number) or ""
            in_furniture = bool(canon) and re.search(rf"(?<!\w){re.escape(canon)}(?!\w)", furniture) is not None
        pn_ok = pn_field_ok or in_furniture
        axes["page_number"] = {
            "passed": pn_ok,
            "score": 1.0 if pn_ok else 0.0,
            "expected": r.page_number,
            "predicted": predicted["page_number"],
            "expected_canonical": canonical_page_number(r.page_number),
            "predicted_canonical": canonical_page_number(predicted["page_number"]),
            "in_furniture_text": in_furniture,
        }

        # leak: annotated furniture must not remain in the body
        body = page_md
        for pattern in _TAG_RE.values():  # tagged furniture is not body
            body = pattern.sub(" ", body)
        body = _ANY_TAG_RE.sub(" ", body)  # then any unbalanced stray tag
        body_norm = " " + norm_furniture(body) + " "
        leaked: list[str] = []
        checked = 0
        for slot in ("header", "footer"):
            value = getattr(r, slot)
            if value:
                pieces = [p for p in re.split(r"\s*\|\s*", value) if len(norm_furniture(p)) >= r.leak_min_chars]
                for piece in pieces:
                    checked += 1
                    if norm_furniture(piece) in body_norm:
                        leaked.append(piece)
        if r.page_number_raw or r.page_number:
            raw = norm_furniture(r.page_number_raw or r.page_number)
            if raw:
                checked += 1
                if re.search(rf"(?<!\S){re.escape(raw)}(?!\S)", body_norm):
                    leaked.append(r.page_number_raw or r.page_number or "")
        if checked == 0:
            axes["leak"] = {"passed": None, "score": None, "reason": "no furniture expected"}
        else:
            axes["leak"] = {
                "passed": not leaked,
                "score": round(1.0 - len(leaked) / checked, 4),
                "leaked": leaked,
                "checked": checked,
            }

        self.result_details = {
            "source": source,
            "expected": {
                "header": r.header,
                "footer": r.footer,
                "page_number": r.page_number,
                "page_number_raw": r.page_number_raw,
            },
            "predicted": predicted,
            "axes": axes,
        }
        applicable = [a for a in axes.values() if a.get("passed") is not None]
        passed = all(a["passed"] for a in applicable)
        score = float(np.mean([a["score"] for a in applicable])) if applicable else 1.0
        failed = [k for k, a in axes.items() if a.get("passed") is False]

        def verdict(k: str) -> str:
            if axes[k].get("passed"):
                return "ok (in header/footer text)" if k == "page_number" and axes[k].get("in_furniture_text") else "ok"
            return axes[k].get("reason") or "mismatch"

        summary = ", ".join(f"{k}:{verdict(k)}" for k in ("header", "footer", "page_number"))
        expl = f"[{source}] {summary}" + (f"; failed: {', '.join(failed)}" if failed else "; all axes pass")
        return passed, expl, round(score, 4)
