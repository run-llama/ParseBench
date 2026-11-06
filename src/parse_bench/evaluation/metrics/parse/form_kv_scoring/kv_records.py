"""Record model for the forms key-value list metric.

A `KvRecord` is a `formnode_scoring` `Field` (so `field_matching.align` applies unchanged) plus the raw
emitted label/value the strict duals read. Kinds: `text` (None when blank), `checkbox` (bool),
`signature` (True only: an unsigned line is a blank), `SELECT_KIND` for option groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from parse_bench.evaluation.metrics.parse.formnode_scoring.field_flatten import Field

SELECT_KIND = "multi_select"
COVERAGE_EXHAUSTIVE = "exhaustive"
COVERAGE_SPOT = "spot"
BLANK_TOKEN = "[blank]"

_BOX_RE = re.compile(r"^\[(?P<mark>[ xX])\](?=\s|$)")
# `[blank]` is the protocol's blank marker; `[___]` is the older spelling, still accepted.
_BLANK_RE = re.compile(r"^\[(?:blank|_+)\]$")
_SIGNATURE_RE = re.compile(r"^\[signature(?:\s*:\s*(?P<signer>[^\]]*?))?\s*\]$", re.IGNORECASE)


@dataclass
class KvRecord(Field):
    """A `Field` plus the raw emitted strings the axes read."""

    raw_label: str | None = None  # emitted caption without bold markers (may carry a baked id)
    raw_value: str | None = None  # emitted value text, None when blank / not a text value
    source_line: int = 0  # 1-based line in the page text
    signer: str | None = None  # `[signature: Name]`


@dataclass
class KvPage:
    """Records of one page; `spot` coverage feeds the recall side only."""

    page_number: int
    records: list[KvRecord]
    coverage: str = COVERAGE_EXHAUSTIVE
    diagnostics: dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class ValueToken:
    """What a value string denotes; a box may carry a write-in after it."""

    kind: str  # text | checkbox | signature
    value: Any  # str | None (text), bool (checkbox), True (signature)
    write_in: str | None = None  # text after a box on the same value (`[x] consulting`)
    write_in_present: bool = False  # a box followed by text or by a blank token
    signer: str | None = None


def is_blank_token(text: str) -> bool:
    return _BLANK_RE.match(text.strip()) is not None


def split_box(text: str) -> tuple[bool, str] | None:
    """`(state, rest)` when `text` starts with a box token, else None."""
    m = _BOX_RE.match(text.strip())
    if m is None:
        return None
    return m.group("mark") in "xX", text.strip()[m.end() :].strip()


def classify_value(text: str | None) -> ValueToken:
    """Blank tokens are blanks, `[x]`/`[ ]` are states (with an optional write-in), `[signature…]` is a signature."""
    s = (text or "").strip()
    if not s or is_blank_token(s):
        return ValueToken("text", None)
    box = split_box(s)
    if box is not None:
        state, rest = box
        if not rest:
            return ValueToken("checkbox", state)
        if is_blank_token(rest):
            return ValueToken("checkbox", state, write_in=None, write_in_present=True)
        return ValueToken("checkbox", state, write_in=rest, write_in_present=True)
    sig = _SIGNATURE_RE.match(s)
    if sig is not None:
        signer = (sig.group("signer") or "").strip() or None
        return ValueToken("signature", True, signer=signer)
    return ValueToken("text", s)
