"""Text/value normalization shared by field matching and submetric scoring.

Value comparison is transcription, not interpretation: numeric equivalence applies only
when BOTH sides parse as numbers after stripping `$ , %` and whitespace ("590.50" ==
590.5, "25%" == "25"); anything else is casefolded whitespace-collapsed string equality,
so "(284,791)" is literal text, NOT -284791.
"""

import re
from typing import Any

import Levenshtein

_NUMERIC_CHROME = re.compile(r"[$,%\s]")
_WS = re.compile(r"\s+")


def fold(s: Any) -> str:
    """Casefold and collapse whitespace for order-insensitive text comparison."""
    return _WS.sub(" ", str(s)).strip().casefold()


def numeric_or_none(v: Any) -> float | None:
    """The numeric reading of a value, or None when it isn't cleanly numeric.

    Booleans are never numeric; strings must be a plain (optionally signed, optionally
    decimal) number after `$ , %`/whitespace stripping.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    s = _NUMERIC_CHROME.sub("", str(v))
    if not s or not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    return float(s)


def is_empty_value(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def norm_value_eq(gt: Any, pred: Any) -> bool:
    """Transcription equality: empties match empties; booleans compare as booleans;
    numeric compare only when BOTH sides parse as numbers; otherwise casefolded
    whitespace-collapsed string equality (parens stay literal)."""
    if is_empty_value(gt) or is_empty_value(pred):
        return is_empty_value(gt) and is_empty_value(pred)
    if isinstance(gt, bool) or isinstance(pred, bool):
        return isinstance(gt, bool) and isinstance(pred, bool) and gt == pred
    gn, pn = numeric_or_none(gt), numeric_or_none(pred)
    if gn is not None and pn is not None:
        return abs(gn - pn) < 1e-9
    return fold(gt) == fold(pred)


def text_sim(a: Any, b: Any) -> float:
    """Normalized Levenshtein similarity on folded text, 0..1."""
    fa, fb = fold(a), fold(b)
    if fa == fb:
        return 1.0
    if not fa or not fb:
        return 0.0
    return 1.0 - Levenshtein.distance(fa, fb) / max(len(fa), len(fb))
