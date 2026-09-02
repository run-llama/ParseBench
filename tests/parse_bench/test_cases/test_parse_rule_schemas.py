from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from parse_bench.test_cases.loader import load_test_case
from parse_bench.test_cases.parse_rule_schemas import (
    ParseChartDataArrayLabelsRule,
    ParsePresenceRule,
    ParseRuleBase,
    ParseTableRule,
    coerce_parse_rule,
    coerce_parse_rule_list,
    get_rule_id,
    get_rule_layout_bindings,
    get_rule_layout_id,
    get_rule_layout_ids,
    get_rule_page,
    get_rule_type,
)
from parse_bench.test_cases.schema import LayoutTestRule, ParseTestCase


def test_coerce_parse_rule_accepts_unknown_fields_and_optional_id() -> None:
    rule = coerce_parse_rule(
        {
            "type": "present",
            "text": "hello",
            "id": "rule-id-1",
            "legacy_tag": "kept",
        }
    )

    assert isinstance(rule, ParsePresenceRule)
    # Direct attribute access
    assert rule.id == "rule-id-1"
    assert rule.type == "present"
    assert rule.text == "hello"
    # Extra fields accessible via get()
    assert rule.get("legacy_tag") == "kept"
    # get() still works for regular fields
    assert rule.get("id") == "rule-id-1"


def test_coerce_parse_rule_list_for_parse_test_case(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    test_json = tmp_path / "sample.test.json"
    test_json.write_text(
        json.dumps(
            {
                "test_rules": [
                    {"type": "present", "text": "hello"},
                    {"type": "unexpected_word", "bag_of_word": {"sample": 1}},
                ]
            }
        )
    )

    case = load_test_case(pdf_path, test_json)
    assert isinstance(case, ParseTestCase)
    assert case.test_rules is not None
    assert all(not isinstance(rule, dict) and isinstance(rule, BaseModel) for rule in case.test_rules)

    present_rule, word_rule = case.test_rules
    assert present_rule.type == "present"
    assert word_rule.type == "unexpected_word"


def test_layout_test_rule_accepts_tags_and_round_trips_to_annotation() -> None:
    rule = LayoutTestRule.model_validate(
        {
            "type": "layout",
            "page": 2,
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "canonical_class": "Text",
            "tags": ["table_related", "Needs Review"],
        }
    )

    assert rule.tags == ["table_related", "Needs Review"]

    annotation = rule.to_layout_annotation()
    assert annotation.page == 1
    assert annotation.tags == ["table_related", "Needs Review"]


def test_coerce_parse_rule_list_type_validation(tmp_path) -> None:
    rules = coerce_parse_rule_list(
        [
            {"type": "present", "text": "hello"},
            {"type": "unexpected_sentence", "bag_of_sentence": {"sample": 1}},
        ]
    )

    assert len(rules) == 2
    assert all(isinstance(rule, BaseModel) for rule in rules)


def test_coerce_parse_rule_supports_word_percent_rules() -> None:
    rules = coerce_parse_rule_list(
        [
            {"type": "unexpected_word_percent", "bag_of_word": {"sample": 1}},
            {"type": "too_many_word_occurence_percent", "bag_of_word": {"sample": 1}},
            {"type": "missing_word_percent", "bag_of_word": {"sample": 1}},
        ]
    )

    assert [get_rule_type(rule) for rule in rules] == [
        "unexpected_word_percent",
        "too_many_word_occurence_percent",
        "missing_word_percent",
    ]


def test_coerce_parse_rule_supports_sentence_percent_rules() -> None:
    rules = coerce_parse_rule_list(
        [
            {"type": "unexpected_sentence_percent", "bag_of_sentence": {"sample sentence": 1}},
            {
                "type": "too_many_sentence_occurence_percent",
                "bag_of_sentence": {"sample sentence": 1},
            },
            {"type": "missing_sentence_percent", "bag_of_sentence": {"sample sentence": 1}},
        ]
    )

    assert [get_rule_type(rule) for rule in rules] == [
        "unexpected_sentence_percent",
        "too_many_sentence_occurence_percent",
        "missing_sentence_percent",
    ]


def test_table_and_chart_rules_accept_nullable_fields() -> None:
    rule = coerce_parse_rule(
        {
            "type": "table",
            "cell": "A1",
            "up": None,
            "down": None,
            "left": None,
            "right": None,
            "top_heading": None,
            "left_heading": None,
        }
    )
    assert isinstance(rule, ParseTableRule)


def test_coerce_parse_rule_supports_is_title_without_level() -> None:
    rule = coerce_parse_rule(
        {
            "type": "is_title",
            "text": "Executive Summary",
        }
    )

    assert get_rule_type(rule) == "is_title"
    assert rule.get("level") is None


def test_coerce_parse_rule_supports_title_hierarchy_percent() -> None:
    rule = coerce_parse_rule(
        {
            "type": "title_hierarchy_percent",
            "title_hierarchy": {"Executive Summary": {"Revenue Breakdown": ["Regional Split", "Product Split"]}},
        }
    )

    assert get_rule_type(rule) == "title_hierarchy_percent"
    assert isinstance(rule.get("title_hierarchy"), dict)


def test_coerce_parse_rule_supports_heading_structure() -> None:
    rule = coerce_parse_rule(
        {
            "type": "heading_structure",
            "headings": [{"text": "Executive Summary", "level": 2}],
        }
    )

    assert get_rule_type(rule) == "heading_structure"
    assert rule.get("headings")[0].text == "Executive Summary"
    assert rule.get("headings")[0].level == 2


def test_heading_structure_requires_an_explicit_inventory() -> None:
    with pytest.raises(ValueError, match="headings"):
        coerce_parse_rule({"type": "heading_structure"})


def test_coerce_parse_rule_supports_is_latex() -> None:
    rule = coerce_parse_rule(
        {
            "type": "is_latex",
            "formula": r"\frac{a}{b}",
        }
    )

    assert get_rule_type(rule) == "is_latex"
    assert rule.get("formula") == r"\frac{a}{b}"


def test_coerce_parse_rule_supports_is_not_latex() -> None:
    rule = coerce_parse_rule(
        {
            "type": "is_not_latex",
            "text": "$400 billion",
        }
    )

    assert get_rule_type(rule) == "is_not_latex"
    assert rule.get("text") == "$400 billion"


def test_coerce_parse_rule_supports_is_code_block() -> None:
    rule = coerce_parse_rule(
        {
            "type": "is_code_block",
            "language": "c",
            "code": "jklnhj",
        }
    )

    assert get_rule_type(rule) == "is_code_block"
    assert rule.get("language") == "c"
    assert rule.get("code") == "jklnhj"

    chart_rule = coerce_parse_rule(
        {
            "type": "chart_data_point",
            "labels": ["A", "B"],
            "value": 310,
            "relative_tolerance": 0.01,
        }
    )
    assert chart_rule.value == 310
    assert str(chart_rule.get("value")) == "310"


def test_coerce_parse_rule_idempotent() -> None:
    """Already-typed rules pass through coerce_parse_rule unchanged."""
    rule = coerce_parse_rule({"type": "present", "text": "hello"})
    assert isinstance(rule, ParsePresenceRule)

    # Coercing again returns the same object
    same_rule = coerce_parse_rule(rule)
    assert same_rule is rule


def test_direct_attribute_access_on_typed_rules() -> None:
    """Direct attribute access works for all field types."""
    rule = coerce_parse_rule(
        {
            "type": "present",
            "text": "hello world",
            "id": "test-1",
            "page": 3,
            "max_diffs": 2,
            "tags": ["tag1", "tag2"],
            "case_sensitive": False,
            "first_n": 100,
        }
    )
    assert isinstance(rule, ParsePresenceRule)
    assert rule.type == "present"
    assert rule.text == "hello world"
    assert rule.id == "test-1"
    assert rule.page == 3
    assert rule.max_diffs == 2
    assert rule.tags == ["tag1", "tag2"]
    assert rule.case_sensitive is False
    assert rule.first_n == 100
    assert rule.last_n is None
    assert rule.count is None


def test_csv_path_alias_accessible_after_coercion() -> None:
    """csv_path field (alias _csv_path) accessible by field name after coercion."""
    rule = coerce_parse_rule(
        {
            "type": "chart_data_array_labels",
            "data": [["A", "B"], [1, 2]],
            "_csv_path": "/tmp/test.csv",
        }
    )
    assert isinstance(rule, ParseChartDataArrayLabelsRule)
    # Access by field name
    assert rule.csv_path == "/tmp/test.csv"
    # Access by alias via get()
    assert rule.get("_csv_path") == "/tmp/test.csv"


def test_tag_mutation_on_typed_models() -> None:
    """Tags can be mutated directly on typed Pydantic models."""
    rule = coerce_parse_rule({"type": "present", "text": "hello", "tags": ["a"]})
    assert rule.tags == ["a"]

    # Direct assignment works (Pydantic v2 non-frozen models)
    rule.tags = ["a", "b", "c"]
    assert rule.tags == ["a", "b", "c"]


def test_helper_functions_use_direct_access() -> None:
    """get_rule_type/id/page use direct attribute access on typed rules."""
    rule = coerce_parse_rule({"type": "present", "text": "hello", "id": "r1", "page": 5})
    assert get_rule_type(rule) == "present"
    assert get_rule_id(rule) == "r1"
    assert get_rule_page(rule) == 5

    # Also works with None values
    rule2 = coerce_parse_rule({"type": "absent", "text": "bye"})
    assert get_rule_id(rule2) is None
    assert get_rule_page(rule2) is None


def test_get_returns_default_for_missing_keys() -> None:
    """get() returns default for keys not in model fields or extra."""
    rule = coerce_parse_rule({"type": "present", "text": "hello"})
    assert rule.get("nonexistent") is None
    assert rule.get("nonexistent", "fallback") == "fallback"


def test_loader_produces_typed_rules(tmp_path) -> None:
    """End-to-end: loader produces typed rules with tags merged."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    test_json = tmp_path / "sample.test.json"
    test_json.write_text(
        json.dumps(
            {
                "tags": ["doc-tag"],
                "test_rules": [
                    {"type": "present", "text": "hello", "tags": ["rule-tag"]},
                    {"type": "absent", "text": "bye"},
                ],
            }
        )
    )

    case = load_test_case(pdf_path, test_json)
    assert isinstance(case, ParseTestCase)
    assert case.test_rules is not None

    # All rules are typed (not dicts)
    for rule in case.test_rules:
        assert isinstance(rule, ParseRuleBase)

    # Tags are merged: doc-level tags propagated into rules
    assert "doc-tag" in case.test_rules[0].tags
    assert "rule-tag" in case.test_rules[0].tags
    assert "doc-tag" in case.test_rules[1].tags
    assert case.tags == ["doc-tag", "rule-tag"]


def test_layout_grounding_normalizes_layout_id_to_layout_ids() -> None:
    rule = coerce_parse_rule(
        {
            "type": "present",
            "text": "hello",
            "layout_id": "layout-1",
        }
    )
    assert isinstance(rule, ParsePresenceRule)
    assert rule.layout_id == "layout-1"
    assert rule.layout_ids == ["layout-1"]
    assert rule.layout_bindings == {}


def test_layout_grounding_merges_layout_bindings_and_dedupes_layout_ids() -> None:
    rule = coerce_parse_rule(
        {
            "type": "order",
            "before": "a",
            "after": "b",
            "layout_id": "layout-a",
            "layout_ids": ["layout-b", "layout-a"],
            "layout_bindings": {
                "before": "layout-a",
                "after": "layout-c",
            },
        }
    )
    assert rule.layout_id == "layout-a"
    assert rule.layout_ids == ["layout-a", "layout-b", "layout-c"]
    assert rule.layout_bindings == {"before": "layout-a", "after": "layout-c"}


def test_layout_grounding_sets_primary_id_from_layout_ids_when_missing() -> None:
    rule = coerce_parse_rule(
        {
            "type": "present",
            "text": "hello",
            "layout_ids": ["layout-2", "layout-3"],
        }
    )
    assert rule.layout_id == "layout-2"
    assert rule.layout_ids == ["layout-2", "layout-3"]


def test_layout_grounding_helper_accessors_for_typed_and_dict_rules() -> None:
    typed_rule = coerce_parse_rule(
        {
            "type": "order",
            "before": "a",
            "after": "b",
            "layout_bindings": {"before": "layout-1", "after": "layout-2"},
        }
    )
    assert get_rule_layout_ids(typed_rule) == ["layout-1", "layout-2"]
    assert get_rule_layout_id(typed_rule) == "layout-1"
    assert get_rule_layout_bindings(typed_rule) == {"before": "layout-1", "after": "layout-2"}

    dict_rule = {
        "type": "present",
        "text": "hello",
        "layout_id": "layout-3",
    }
    assert get_rule_layout_ids(dict_rule) == ["layout-3"]
    assert get_rule_layout_id(dict_rule) == "layout-3"
    assert get_rule_layout_bindings(dict_rule) == {}


def test_order_layout_bindings_reject_unknown_roles() -> None:
    with pytest.raises(ValueError, match="only supports roles 'before' and 'after'"):
        coerce_parse_rule(
            {
                "type": "order",
                "before": "a",
                "after": "b",
                "layout_bindings": {
                    "middle": "layout-1",
                },
            }
        )


def test_order_layout_bindings_reject_list_values() -> None:
    with pytest.raises(ValueError, match="must be a single layout id string"):
        coerce_parse_rule(
            {
                "type": "order",
                "before": "a",
                "after": "b",
                "layout_bindings": {
                    "before": ["layout-1", "layout-2"],
                },
            }
        )


def test_order_layout_id_is_canonicalized_to_before_binding_when_both_set() -> None:
    rule = coerce_parse_rule(
        {
            "type": "order",
            "before": "a",
            "after": "b",
            "layout_id": "layout-primary",
            "layout_bindings": {
                "before": "layout-other",
                "after": "layout-secondary",
            },
        }
    )
    assert rule.layout_id == "layout-other"
    assert rule.layout_ids == ["layout-other", "layout-primary", "layout-secondary"]


def test_layout_test_rule_without_r_omits_key_on_dump() -> None:
    rule = LayoutTestRule.model_validate(
        {
            "type": "layout",
            "page": 1,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "canonical_class": "Text",
        }
    )

    dumped = rule.model_dump(exclude_none=True)
    assert "r" not in dumped
