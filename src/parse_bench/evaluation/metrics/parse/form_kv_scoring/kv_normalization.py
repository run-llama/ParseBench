"""Comparison rules for the key-value axes. Defaults follow `formnode_scoring` (`norm_value_eq`; labels by
normalised Levenshtein over `normalize_text`, punctuation kept); the `_strict` duals fold only
whitespace and encoding variants and keep case, so `29,513` != `29513`. `baked_id` spots an
id-looking label prefix for the lenient id dual on unmatched predictions.
"""

from __future__ import annotations

import re
from typing import Any

from parse_bench.evaluation.metrics.parse.formnode_scoring.text_normalization import (
    fold,
    is_empty_value,
    norm_value_eq,
    text_sim,
)
from parse_bench.evaluation.metrics.parse.utils import _normalize_encoding_punctuation, normalize_text

_WS = re.compile(r"\s+")
_BAKED_ID_RE = re.compile(r"^(?:\d{1,3}[a-z]?|[A-Za-z]\d?|[IVX]+|Part [IVX]+)[.):\-]?\s+\S")


def fold_label(label: Any) -> str:
    return normalize_text(str(label)) if label else ""


def label_sim(a: Any, b: Any) -> float:
    return text_sim(fold_label(a), fold_label(b))


def label_eq_strict(a: Any, b: Any) -> bool:
    return _strict_text(a or "") == _strict_text(b or "")


def value_eq(gt: Any, pred: Any) -> bool:
    return norm_value_eq(gt, pred)


def value_eq_strict(gt: Any, pred: Any) -> bool:
    if is_empty_value(gt) or is_empty_value(pred):
        return is_empty_value(gt) and is_empty_value(pred)
    if isinstance(gt, bool) or isinstance(pred, bool):
        return isinstance(gt, bool) and isinstance(pred, bool) and gt == pred
    return _strict_text(gt) == _strict_text(pred)


def _strict_text(v: Any) -> str:
    return _WS.sub(" ", _normalize_encoding_punctuation(str(v))).strip()


def id_eq(a: Any, b: Any) -> bool:
    return fold(a) == fold(b)


def baked_id(label: Any) -> bool:
    return bool(label) and _BAKED_ID_RE.match(str(label)) is not None
