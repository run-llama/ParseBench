from __future__ import annotations

from typing import Any

from parse_bench.evaluation.metrics.extract.json_subset_match import (
    json_subset_match_score,
    normalize_date_string,
)
from parse_bench.evaluation.metrics.extract.json_subset_match_metric import JsonSubsetMatchMetric


def test_normalize_date_string_handles_weekday_prefix_with_periods() -> None:
    assert normalize_date_string("Mon. Jan. 02 2023") == "2023-01-02"
    assert normalize_date_string("Fri. Dec. 29 2023") == "2023-12-29"


def test_normalize_date_string_tolerates_comma_spacing() -> None:
    assert normalize_date_string("March 27 ,1956") == "1956-03-27"
    assert normalize_date_string("March 28,1956") == "1956-03-28"


def test_normalize_date_string_two_digit_year_patterns() -> None:
    assert normalize_date_string("01/02/23") == "2023-01-02"
    assert normalize_date_string("01-02-23") == "2023-01-02"


def test_normalize_date_string_rejects_implausible_years() -> None:
    assert normalize_date_string("0001-01-01") == "0001-01-01"
    assert normalize_date_string("3000-01-01") == "3000-01-01"


def test_normalize_date_string_non_string_passthrough() -> None:
    assert normalize_date_string(None) is None
    assert normalize_date_string(42) == 42
    assert normalize_date_string("12345678") == "12345678"


def test_normalize_date_string_probe_is_comma_scoped() -> None:
    """The comma-respacing probe must not widen what counts as a date: strings
    without a comma are left exactly as before (doubled spaces included)."""
    assert normalize_date_string("March  27 1956") == "March  27 1956"
    assert normalize_date_string("March 27 1956") == "1956-03-27"


def test_json_subset_match_scores_weekday_date_equivalence() -> None:
    score = json_subset_match_score(
        {"attendance_records": [{"date": "2023-01-02"}]},
        {"attendance_records": [{"date": "Mon. Jan. 02 2023"}]},
    )

    assert score == 1.0


# Regression tests: missing arrays/dicts must be weighted by the full leaf
# count of `expected`, not weight=1. Otherwise a pipeline that drops a whole
# claims array can score the same as one that extracts it.


def test_missing_top_level_array_weighted_by_full_leaf_count() -> None:
    """A 15-claim array dropped entirely must outweigh a 1-leaf scalar field."""
    expected = {
        "scalar_field": "x",
        "claims": [{"a": 1, "b": 2, "c": 3} for _ in range(15)],
    }
    actual = {"scalar_field": "x"}

    score = json_subset_match_score(expected, actual, weighted=True)
    # 1 leaf right out of 46 total => ~0.022. Pre-fix this was ~0.5 because
    # the missing claims array was treated as a single weight=1 leaf.
    assert score < 0.05, f"expected score <0.05, got {score}"


def test_missing_array_same_as_empty_array() -> None:
    """An absent key and `[]` should score identically when expected is non-empty."""
    expected = {"claims": [{"a": 1, "b": 2}]}
    score_missing = json_subset_match_score(expected, {}, weighted=True)
    score_empty = json_subset_match_score(expected, {"claims": []}, weighted=True)
    assert score_missing == score_empty == 0.0


def test_missing_nested_dict_weighted_by_subtree() -> None:
    """A dropped nested dict must be weighted by its leaf count."""
    expected = {
        "id": "x",
        "patient": {
            "first_name": "A",
            "last_name": "B",
            "address": {"city": "X", "zip": "1"},
        },
    }
    actual = {"id": "x"}
    score = json_subset_match_score(expected, actual, weighted=True)
    # 1/5 = 0.20
    assert 0.18 < score < 0.22, f"expected ~0.20, got {score}"


def test_partial_array_drop_weights_each_missing_item_recursively() -> None:
    """Dropping the tail of an array must penalize by per-item leaves."""
    expected = {
        "claims": [
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        ]
    }
    actual = {"claims": [{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}]}

    score = json_subset_match_score(expected, actual, weighted=True)
    # 5/15 = ~0.33. Pre-fix this would be ~0.78 because each missing tail
    # item only contributed weight=1.
    assert 0.30 < score < 0.40, f"expected ~0.33, got {score}"


def test_dropped_large_array_scores_low() -> None:
    """When a pipeline returns one section correctly but doesn't emit a
    15-record array with ~10 fields each, the score should reflect that
    ~150 leaves are missing, not weight=1."""
    expected = {
        "header": [
            {
                "check": {"check_number": "X", "check_amount": 100, "check_date": "2026-01-01"},
                "payer": {"payer_name": "P", "payer_phone": "555"},
                "payee": {"payee_name": "Q"},
            }
        ],
        "records": [
            {
                "record_number": f"C{i}",
                "name": f"P{i}",
                "total_paid": i * 10.0,
                "total_submitted": i * 12.0,
                "from_date": "2026-01-01",
                "to_date": "2026-01-31",
                "account_number": f"A{i}",
                "plan_type": "PPO",
                "status": "PAID",
                "responsibility": 0.0,
            }
            for i in range(15)
        ],
    }
    actual = {"header": expected["header"]}
    score = json_subset_match_score(expected, actual, weighted=True)
    # header has 7 leaves, records has 15 × 10 = 150 leaves. 7/157 = ~0.045.
    # Pre-fix this scored ~0.875 (header perfect / records weight=1).
    assert score < 0.10, f"dropped 15-record array should score <0.10, got {score}"


# Order-invariant pairing: lists must be paired by optimal assignment
# (Hungarian), not by index. Shuffled-but-correct extractions should not be
# penalized for row order.


def test_anonymous_record_list_is_order_invariant() -> None:
    expected = {
        "rows": [
            {"col_a": 1, "col_b": "x"},
            {"col_a": 2, "col_b": "y"},
            {"col_a": 3, "col_b": "z"},
        ]
    }
    actual = {"rows": list(reversed(expected["rows"]))}

    assert json_subset_match_score(expected, actual, weighted=True) == 1.0


def test_scalar_list_is_order_invariant() -> None:
    assert json_subset_match_score({"chunks": ["a", "b", "c"]}, {"chunks": ["c", "a", "b"]}) == 1.0


def test_exact_lists_score_perfectly() -> None:
    expected = {"chunks": ["a", "b", "c"]}
    actual = {"chunks": ["a", "b", "c"]}
    assert json_subset_match_score(expected, actual, weighted=True) == 1.0

    expected2 = {"rows": [{"col_a": 1, "col_b": 2}, {"col_a": 3, "col_b": 4}]}
    actual2 = {"rows": [{"col_a": 1, "col_b": 2}, {"col_a": 3, "col_b": 4}]}
    assert json_subset_match_score(expected2, actual2, weighted=True) == 1.0


def test_order_invariant_pairing_recovers_dropped_middle_row() -> None:
    """A dropped middle row must cost only that row, not cascade an
    off-by-one penalty onto every subsequent index-paired row."""
    expected = {"rows": [{"a": i, "b": f"r{i}"} for i in range(4)]}
    actual = {"rows": [expected["rows"][0], expected["rows"][2], expected["rows"][3]]}

    score = json_subset_match_score(expected, actual, weighted=True)
    # 3 of 4 rows match perfectly, each row has 2 leaves: 6/8 = 0.75.
    assert score == 0.75, f"expected 0.75, got {score}"


def test_tied_assignment_prefers_index_order() -> None:
    """When every pairing is a near-miss (uniform approximate cost matrix),
    index order must win the tie deterministically. Pins the eps*|i-j|
    tie-break: without it, which optimal assignment wins is an undocumented
    scipy implementation detail, and a scipy upgrade could silently swap
    aligned near-miss rows for crossed ones, dropping the Levenshtein
    partial credit with no code change."""
    expected = {"rows": [{"text": "alpha bravo"}, {"text": "charlie delta"}]}
    # Aligned rows with one-character typos: no leaf matches exactly, so the
    # approximate cost is 1 for every pairing — a fully tied matrix.
    actual = {"rows": [{"text": "alpha brivo"}, {"text": "charlie delte"}]}

    score = json_subset_match_score(expected, actual, weighted=True)
    # Index-aligned pairing keeps ~0.92 Levenshtein partial credit; the
    # crossed pairing would score near zero.
    assert score > 0.8, f"aligned near-miss rows should keep partial credit, got {score}"


def test_order_invariant_pairing_does_not_reward_wrong_values() -> None:
    """Optimal assignment must not inflate scores when values are wrong —
    a fully mismatched list still scores low."""
    expected = {"rows": [{"a": 1}, {"a": 2}]}
    actual = {"rows": [{"a": 7}, {"a": 8}]}

    score = json_subset_match_score(expected, actual, weighted=True)
    assert score < 0.5, f"expected <0.5, got {score}"


def test_unweighted_mode_penalizes_extra_actual_items() -> None:
    expected = {"rows": ["a", "b"]}
    actual = {"rows": ["b", "a", "c", "d"]}
    assert json_subset_match_score(expected, actual, weighted=False) == 0.5


def test_assignment_pairing_falls_back_to_index_beyond_pair_cap(monkeypatch: Any) -> None:
    """Beyond the bounded pair budget the n×m assignment is skipped and
    index pairing applies, so pathological very-long arrays don't stall the
    evaluator."""
    import parse_bench.evaluation.metrics.extract.json_subset_match as jsm

    expected = {"rows": [{"a": i} for i in range(3)]}
    actual = {"rows": [{"a": i} for i in reversed(range(3))]}

    monkeypatch.setattr(jsm, "_ASSIGNMENT_MAX_PAIRS", 4)
    capped = json_subset_match_score(expected, actual, weighted=True)
    assert capped < 1.0, f"beyond the cap, index pairing should penalize reorder, got {capped}"

    monkeypatch.setattr(jsm, "_ASSIGNMENT_MAX_PAIRS", 250_000)
    assert json_subset_match_score(expected, actual, weighted=True) == 1.0


# Nullable-numeric collapse: gated strictly on the JSON Schema shape.

_NULLABLE_NUMBER_SCHEMA = {
    "type": "object",
    "properties": {
        "amount": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": None},
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"fee": {"type": ["null", "number"]}},
            },
        },
    },
}


def test_nullable_numeric_zero_matches_null_with_schema() -> None:
    assert json_subset_match_score({"amount": 0}, {"amount": None}, data_schema=_NULLABLE_NUMBER_SCHEMA) == 1.0
    assert json_subset_match_score({"amount": None}, {"amount": 0.0}, data_schema=_NULLABLE_NUMBER_SCHEMA) == 1.0
    assert (
        json_subset_match_score(
            {"rows": [{"fee": 0.0}]},
            {"rows": [{"fee": None}]},
            data_schema=_NULLABLE_NUMBER_SCHEMA,
        )
        == 1.0
    )


def test_nullable_numeric_collapse_requires_schema() -> None:
    assert json_subset_match_score({"amount": 0}, {"amount": None}) == 0.0
    assert json_subset_match_score({"amount": None}, {"amount": 0}) == 0.0


def test_nullable_numeric_collapse_does_not_apply_to_plain_numbers_or_bools() -> None:
    plain_schema = {"type": "object", "properties": {"amount": {"type": "number"}}}
    assert json_subset_match_score({"amount": 0}, {"amount": None}, data_schema=plain_schema) == 0.0
    bool_schema = {"type": "object", "properties": {"flag": {"anyOf": [{"type": "number"}, {"type": "null"}]}}}
    assert json_subset_match_score({"flag": False}, {"flag": None}, data_schema=bool_schema) == 0.0


def test_nullable_numeric_nonzero_still_compares_numerically() -> None:
    assert json_subset_match_score({"amount": 5}, {"amount": None}, data_schema=_NULLABLE_NUMBER_SCHEMA) == 0.0
    assert json_subset_match_score({"amount": 5}, {"amount": 5}, data_schema=_NULLABLE_NUMBER_SCHEMA) == 1.0


def test_metric_passes_data_schema_through() -> None:
    metric = JsonSubsetMatchMetric()
    without = metric.compute(expected={"amount": 0}, actual={"amount": None})
    with_schema = metric.compute(expected={"amount": 0}, actual={"amount": None}, data_schema=_NULLABLE_NUMBER_SCHEMA)
    assert without.value == 0.0
    assert with_schema.value == 1.0
