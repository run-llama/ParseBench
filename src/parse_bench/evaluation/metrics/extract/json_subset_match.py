"""JSON subset matching metric for extract evaluation.

Ports json_subset_match_score from extract-tests with date normalization support.
"""

import re
from typing import Any

import numpy as np
from autoevals.number import NumericDiff  # type: ignore[import-untyped]
from autoevals.string import EmbeddingSimilarity, Levenshtein  # type: ignore[import-untyped]
from dateutil import parser as date_parser  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment

_COMMA_WS_RE = re.compile(r"\s*,\s*")


def normalize_date_string(date_str: Any) -> Any:
    """Normalise a date string to ISO format (YYYY-MM-DD).

    Returns the input unchanged if it does not look like a date.
    """
    if not isinstance(date_str, str):
        return date_str

    if len(date_str) < 4 or len(date_str) > 50:
        return date_str

    if date_str.isdigit():
        return date_str

    # Long digit runs (10+) are almost certainly IDs, not dates.
    if re.search(r"\d{10,}", date_str):
        return date_str

    # Spacing around the comma carries no meaning: "March 28,1956" and
    # "March 27 ,1956" name the same date as "March 27, 1956". Normalize the
    # comma to ", " on a probe copy so these variants clear the pattern gate
    # and parse; the original is still what we return unchanged when it is not
    # a date. Comma-gated and comma-only on purpose: this function runs on
    # every string leaf of every dataset, and a wider rewrite (e.g. collapsing
    # doubled spaces) would silently widen what counts as a date everywhere.
    probe = _COMMA_WS_RE.sub(", ", date_str) if "," in date_str else date_str

    date_patterns = [
        r"\d{4}-\d{1,2}-\d{1,2}",  # YYYY-MM-DD
        r"\d{1,2}/\d{1,2}/\d{4}",  # MM/DD/YYYY
        r"\d{1,2}/\d{1,2}/\d{2}\b",  # MM/DD/YY (v5 GT canonical format)
        r"\d{1,2}-\d{1,2}-\d{4}",  # MM-DD-YYYY
        r"\d{1,2}-\d{1,2}-\d{2}\b",  # MM-DD-YY
        r"[A-Za-z]+ \d{1,2},? \d{4}",  # Month DD, YYYY
        r"[A-Za-z]+\.? [A-Za-z]+\.? \d{1,2},? \d{4}",  # Weekday Month DD YYYY
        r"\d{1,2} [A-Za-z]+ \d{4}",  # DD Month YYYY
    ]
    if not any(re.search(p, probe) for p in date_patterns):
        return date_str

    try:
        parsed = date_parser.parse(probe, fuzzy=False)
        if parsed.year < 1900 or parsed.year > 2100:
            return date_str
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError, date_parser.ParserError, OverflowError):
        return date_str


def _is_nullable_numeric_field(schema_node: Any) -> bool:
    """True when a field's JSON Schema shape is a nullable number.

    Recognizes both idiomatic Pydantic/JSON Schema encodings used across
    extract test cases:

    * ``{"anyOf": [{"type": "number"}, {"type": "null"}], "default": null}``
    * ``{"anyOf": [{"type": "null"}, {"type": "number"}], ...}`` (order-agnostic)
    * ``{"type": ["number", "null"]}`` / ``{"type": ["null", "number"]}``

    Only the shape gate — the ``default`` clause is not required. Callers use
    this to treat ``0`` / ``0.0`` and ``None`` as equivalent on such fields,
    because "not present on this row" and "amount = zero" are not distinguished
    semantically in financial extraction ground truth.
    """
    if not isinstance(schema_node, dict):
        return False
    any_of = schema_node.get("anyOf")
    if isinstance(any_of, list) and len(any_of) == 2:
        types = {branch.get("type") for branch in any_of if isinstance(branch, dict)}
        if types == {"number", "null"}:
            return True
    type_field = schema_node.get("type")
    if isinstance(type_field, list) and set(type_field) == {"number", "null"}:
        return True
    return False


def _normalize_nullable_numeric(value: Any) -> Any:
    """Collapse ``0`` / ``0.0`` to ``None`` on nullable numeric fields.

    Everything else (non-zero numbers, non-numeric types) is returned as-is.
    Booleans are explicitly excluded from the zero collapse so that ``False``
    is never conflated with an absent value.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value == 0:
        return None
    return value


# Budget for the order-invariant assignment's n×m cost matrix. Beyond it
# (e.g. a 3,000-row shuffled array is 9M pairs and ~3s of Python-level cost
# building, recursing into nested lists) index pairing applies instead of
# stalling the evaluator.
_ASSIGNMENT_MAX_PAIRS = 250_000


def _count_leaves(value: Any) -> int:
    """Count leaf nodes in a JSON-like value.

    A leaf is any primitive (str, number, bool, None) or an empty dict / list.
    Used to weight a missing field by the structural cost of what it contains
    in `expected`, so that dropping a 15-claim array isn't scored the same
    as dropping a single scalar.

    Conservative behavior:
    - Empty dict or empty list contribute weight 1 (matches the existing
      "empty == empty" success path which also uses weight 1).
    - Any non-dict / non-list value contributes weight 1.
    """
    if isinstance(value, dict):
        if not value:
            return 1
        return sum(_count_leaves(v) for v in value.values())
    if isinstance(value, list):
        if not value:
            return 1
        return sum(_count_leaves(item) for item in value)
    return 1


def _flatten_normalized_leaves(
    value: Any,
    *,
    case_sensitive: bool,
    normalize_dates: bool,
    _path: tuple[Any, ...] = (),
) -> dict[tuple[Any, ...], Any]:
    """Flatten a JSON-like value into {path: normalized_leaf}.

    Used to build the assignment cost matrix for order-invariant list
    pairing. Strings are normalized the same way the scalar scoring path
    normalizes them (lowercase unless case_sensitive, then date
    normalization) so the approximate cost agrees with the real scorer on
    exact matches. Empty dicts / lists are kept as leaves, matching
    `_count_leaves` semantics.
    """
    if isinstance(value, dict) and value:
        flat: dict[tuple[Any, ...], Any] = {}
        for k, v in value.items():
            flat.update(
                _flatten_normalized_leaves(
                    v, case_sensitive=case_sensitive, normalize_dates=normalize_dates, _path=(*_path, k)
                )
            )
        return flat
    if isinstance(value, list) and value:
        flat = {}
        for i, v in enumerate(value):
            flat.update(
                _flatten_normalized_leaves(
                    v, case_sensitive=case_sensitive, normalize_dates=normalize_dates, _path=(*_path, i)
                )
            )
        return flat
    if isinstance(value, str):
        normalized = value if case_sensitive else value.lower()
        if normalize_dates:
            normalized = normalize_date_string(normalized)
        return {_path: normalized}
    return {_path: value}


def _descend_schema_for_key(schema_node: Any, key: str) -> Any:
    """Return the child schema for ``key`` under ``schema_node`` if declared."""
    if not isinstance(schema_node, dict):
        return None
    props = schema_node.get("properties")
    if isinstance(props, dict) and key in props:
        return props[key]
    return None


def _descend_schema_for_items(schema_node: Any) -> Any:
    """Return the ``items`` schema for an array node."""
    if not isinstance(schema_node, dict):
        return None
    items = schema_node.get("items")
    if isinstance(items, dict):
        return items
    return None


def _compute_score_with_weight(
    expected: Any,
    actual: Any,
    weighted: bool,
    case_sensitive: bool,
    cosine_similarity: bool,
    normalize_dates: bool,
    string_scorer: Any,
    number_scorer: Any,
    schema_node: Any = None,
) -> tuple[float, int]:
    """
    Recursively compute match score and weight.

    :param expected: Expected JSON structure
    :param actual: Actual JSON structure
    :param weighted: If True, aggregate by leaf node weights; if False, simple average
    :param case_sensitive: Whether string comparison should be case-sensitive
    :param cosine_similarity: Use embedding similarity for strings
    :param normalize_dates: Normalize date strings before comparison
    :param string_scorer: Scorer for string comparison
    :param number_scorer: Scorer for number comparison
    :param schema_node: JSON Schema node describing ``expected`` (when known);
        used to gate the nullable-numeric ``0 == None`` collapse
    :return: (score, weight) where weight is the number of leaf nodes in expected

    When `expected` has structure (dict/list) but `actual` is missing or of
    a non-matching type (most commonly: the key was absent from a parent dict
    so `dict.get(k)` returned None), the score is 0 with weight equal to the
    full leaf count of `expected`. This ensures a dropped 15-claim array is
    penalized by 15 × per-claim leaves rather than weight=1.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            # actual is missing the key entirely (or wrong type). Score 0
            # with weight equal to expected's full leaf count.
            return (0.0, _count_leaves(expected))
        if len(expected) == 0 and len(actual) == 0:
            return (1.0, 1)
        if len(expected) == 0:
            return (1.0, 1)

        # Compute scores and weights for each key
        results: list[tuple[float, int]] = []
        for k in expected.keys():
            score, weight = _compute_score_with_weight(
                expected.get(k),
                actual.get(k),
                weighted=weighted,
                case_sensitive=case_sensitive,
                cosine_similarity=cosine_similarity,
                normalize_dates=normalize_dates,
                string_scorer=string_scorer,
                number_scorer=number_scorer,
                schema_node=_descend_schema_for_key(schema_node, k),
            )
            results.append((score, weight))

        if not results:
            return (0.0, 1)

        total_weight = sum(w for _, w in results)
        # When weighted=False, treat each field as weight=1
        effective_weights = [w if weighted else 1 for _, w in results]
        total_eff_weight = sum(effective_weights)
        if total_eff_weight == 0:
            return (0.0, max(total_weight, 1))
        weighted_sum = sum(s * ew for (s, _), ew in zip(results, effective_weights, strict=True))
        agg_score = weighted_sum / total_eff_weight

        return (agg_score, max(total_weight, 1))

    elif isinstance(expected, list):
        if not isinstance(actual, list):
            # actual is missing the key entirely (or wrong type). Score 0
            # with weight equal to expected's full leaf count, recursively.
            return (0.0, _count_leaves(expected))
        if len(expected) == 0 and len(actual) == 0:
            return (1.0, 1)
        if len(expected) == 0:
            return (1.0, 1)
        if len(actual) == 0:
            # All expected items missing - weight by expected's full leaf
            # count so a dropped 15-claim array is penalized by 15 × per-claim
            # leaves, not just len(expected).
            return (0.0, max(_count_leaves(expected), 1))

        item_schema = _descend_schema_for_items(schema_node)

        # Order-invariant pairing: optimal one-to-one assignment between
        # expected and actual elements (Hungarian algorithm). The assignment
        # cost is an approximate mismatch count over pre-normalized leaves
        # (cheap exact equality, computed once per element so the n×m matrix
        # stays cheap); the score for each assigned pair is then the full
        # recursive partial-credit score. Unmatched expected elements score 0
        # with their full leaf weight; extra actual elements are ignored in
        # weighted mode (subset semantics) and penalized through the
        # max-length denominator in unweighted mode.
        if len(expected) * len(actual) > _ASSIGNMENT_MAX_PAIRS:
            # Pathologically long arrays: building the cost matrix is
            # O(n·m·leaves) in Python, so beyond the pair budget pair by
            # index instead of stalling the evaluator.
            assigned = {i: i for i in range(min(len(expected), len(actual)))}
        else:
            expected_flat = [
                _flatten_normalized_leaves(item, case_sensitive=case_sensitive, normalize_dates=normalize_dates)
                for item in expected
            ]
            actual_flat = [
                _flatten_normalized_leaves(item, case_sensitive=case_sensitive, normalize_dates=normalize_dates)
                for item in actual
            ]
            # Tie-break toward index order: scipy does not document which
            # optimal assignment it returns when several are tied (e.g.
            # every row is a near-miss and the cost matrix is uniform), so
            # without this a scipy upgrade could legally swap aligned
            # near-miss rows for crossed ones and silently drop their
            # Levenshtein partial credit. The index distance term is scaled
            # so its total over any assignment stays below 1 (sum of |i-j|
            # is bounded by n*m), which means it can never override a real
            # integer mismatch-count difference.
            tiebreak_eps = 1.0 / (len(expected) * len(actual) + 1)
            cost = np.empty((len(expected), len(actual)))
            for i, exp_leaves in enumerate(expected_flat):
                for j, act_leaves in enumerate(actual_flat):
                    mismatches = sum(
                        1
                        for leaf_path, leaf in exp_leaves.items()
                        if leaf_path not in act_leaves or act_leaves[leaf_path] != leaf
                    )
                    cost[i, j] = mismatches + tiebreak_eps * abs(i - j)
            exp_indices, act_indices = linear_sum_assignment(cost)
            assigned = dict(zip(exp_indices.tolist(), act_indices.tolist(), strict=True))

        list_results: list[tuple[float, int]] = []
        for i, exp_item in enumerate(expected):
            paired_index = assigned.get(i)
            if paired_index is None:
                # More expected than actual: this element went unmatched.
                # Weight by full leaf count so dropping rows of a long
                # array isn't under-penalized.
                list_results.append((0.0, _count_leaves(exp_item)))
                continue
            score, weight = _compute_score_with_weight(
                exp_item,
                actual[paired_index],
                weighted=weighted,
                case_sensitive=case_sensitive,
                cosine_similarity=cosine_similarity,
                normalize_dates=normalize_dates,
                string_scorer=string_scorer,
                number_scorer=number_scorer,
                schema_node=item_schema,
            )
            list_results.append((score, weight))

        if not list_results:
            return (0.0, 1)

        total_weight = sum(w for _, w in list_results)
        if weighted:
            # Weighted: each element contributes proportionally to its leaf count
            if total_weight == 0:
                return (0.0, 1)
            agg_score = sum(s * w for s, w in list_results) / total_weight
        else:
            # Unweighted: divide by max length to penalize extra items in actual
            agg_score = sum(s for s, _ in list_results) / max(len(expected), len(actual))

        return (agg_score, max(total_weight, 1))

    elif isinstance(expected, str):
        if not isinstance(actual, str):
            return (0.0, 1)

        expected_normalized = expected
        actual_normalized = actual

        if not case_sensitive:
            expected_normalized = expected_normalized.lower()
            actual_normalized = actual_normalized.lower()

        if normalize_dates:
            expected_normalized = normalize_date_string(expected_normalized)
            actual_normalized = normalize_date_string(actual_normalized)

        result = string_scorer.eval(expected_normalized, actual_normalized)
        score = result.score if hasattr(result, "score") else 0.0
        return (score, 1)

    elif isinstance(expected, (int, float)):
        # Nullable numeric fields treat 0 / 0.0 and None as equivalent — the
        # semantic on financial extraction ground truth is that null means
        # "not present on this row" and there is no meaningful distinction
        # between "not present" and "amount = zero". Gated strictly on the
        # JSON Schema shape so plain numeric fields keep strict semantics.
        # Booleans are excluded from the collapse (they are a subclass of int
        # but ``False`` is never conflated with an absent value).
        if not isinstance(expected, bool) and _is_nullable_numeric_field(schema_node):
            norm_expected = _normalize_nullable_numeric(expected)
            norm_actual = _normalize_nullable_numeric(actual)
            if norm_expected is None and norm_actual is None:
                return (1.0, 1)
        if not isinstance(actual, (int, float)):
            return (0.0, 1)
        result = number_scorer.eval(expected, actual)
        score = result.score if hasattr(result, "score") else 0.0
        return (score, 1)

    elif expected is None:
        if actual is None:
            return (1.0, 1)
        # Same nullable-numeric gate for the None-expected direction: 0/0.0
        # predicted against a null-expected cell scores as a match.
        if (
            _is_nullable_numeric_field(schema_node)
            and isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and _normalize_nullable_numeric(actual) is None
        ):
            return (1.0, 1)
        return (0.0, 1)

    else:
        # Type mismatch or unsupported type
        return (0.0, 1)


def json_subset_match_score(
    expected: Any,
    actual: Any,
    case_sensitive: bool = True,
    cosine_similarity: bool = False,
    normalize_dates: bool = True,
    weighted: bool = True,
    data_schema: dict[str, Any] | None = None,
) -> float:
    """
    Calculate similarity score between expected and actual JSON structures.

    Adapted from autoevals.JsonDiff to only test on the subset of keys within
    the expected json. This means extra keys in actual are ignored.

    :param expected: Expected JSON structure (dict, list, or primitive)
    :param actual: Actual JSON structure to compare
    :param case_sensitive: Whether string comparison should be case-sensitive
    :param cosine_similarity: Use embedding similarity for strings (slower but more semantic)
    :param normalize_dates: Normalize date strings before comparison
    :param weighted: If True (default), weight fields by their number of leaf nodes.
                     If False, use simple averaging (each field/element counts equally).
    :param data_schema: Optional JSON Schema for ``expected``; when provided,
        nullable numeric fields (``anyOf: [number, null]`` / ``type: [number, null]``)
        treat ``0`` / ``0.0`` and ``None`` as equivalent
    :return: Similarity score between 0.0 and 1.0
    """
    string_scorer = Levenshtein() if not cosine_similarity else EmbeddingSimilarity()
    number_scorer = NumericDiff()

    score, _ = _compute_score_with_weight(
        expected=expected,
        actual=actual,
        weighted=weighted,
        case_sensitive=case_sensitive,
        cosine_similarity=cosine_similarity,
        normalize_dates=normalize_dates,
        string_scorer=string_scorer,
        number_scorer=number_scorer,
        schema_node=data_schema,
    )
    return score
