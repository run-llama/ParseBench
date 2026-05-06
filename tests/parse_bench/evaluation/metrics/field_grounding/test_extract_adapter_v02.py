"""Tests for v0.2 schema extensions and the page-grounded / coverage metrics.

Phase A scope. Phase B (comparator dispatch, structural rules) is not yet
exercised here — those tests should land alongside the dispatch table.
"""

from __future__ import annotations

import pytest

from parse_bench.evaluation.metrics.field_grounding.extract_adapter import (
    compute_extract_field_grounding_metrics,
)
from parse_bench.evaluation.metrics.field_grounding.value_compare import (
    candidate_values_for_rule,
    compare_field_with_rule,
    compare_value_against_rule,
)
from parse_bench.schemas.extract_output import FieldCitation
from parse_bench.test_cases.schema import (
    ExtractFieldBbox,
    ExtractFieldTestRule,
    FieldEvidence,
    iter_rule_evidence,
)


def _named(metrics):
    return {m.metric_name: m for m in metrics}


class TestIterRuleEvidence:
    def test_legacy_bboxes_synthesize_evidence(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.2])],
        )
        evidence = iter_rule_evidence(rule)
        assert len(evidence) == 1
        assert evidence[0].page == 1
        assert evidence[0].value == "bar"
        assert evidence[0].coarse is False

    def test_v02_evidence_list_returned_verbatim(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="UNII",
            evidence=[
                FieldEvidence(page=2, bbox=[0.5, 0.5, 0.1, 0.1], value="LABEL"),
                FieldEvidence(page=3, value="UNII", coarse=True),
            ],
        )
        evidence = iter_rule_evidence(rule)
        assert len(evidence) == 2
        assert evidence[1].coarse is True
        assert evidence[1].bbox is None

    def test_empty_rule_returns_empty(self):
        rule = ExtractFieldTestRule(field_path="foo", expected_value=None)
        assert iter_rule_evidence(rule) == []


class TestComparatorShim:
    def test_falls_through_to_attributed_value_when_no_comparator(self):
        rule = ExtractFieldTestRule(field_path="foo", expected_value="bar")
        result = compare_field_with_rule(rule, "bar", "bar", expected_type="string")
        assert result.passed

    def test_rule_none_falls_through(self):
        result = compare_field_with_rule(None, "bar", "bar", expected_type="string")
        assert result.passed


class TestCandidateValuesForRule:
    def test_legacy_returns_single_expected(self):
        rule = ExtractFieldTestRule(field_path="foo", expected_value="bar")
        assert candidate_values_for_rule(rule) == ["bar"]

    def test_v02_evidence_emits_all_unique_values(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="canonical",
            evidence=[
                FieldEvidence(page=1, value="alt-1"),
                FieldEvidence(page=2, value="canonical"),
                FieldEvidence(page=3, value="alt-1"),
            ],
        )
        cands = candidate_values_for_rule(rule)
        assert "alt-1" in cands and "canonical" in cands
        assert len(cands) == 2

    def test_none_rule_returns_none(self):
        assert candidate_values_for_rule(None) == [None]


class TestOrOverEvidence:
    def test_pipeline_value_matches_alternate_evidence_value(self):
        rule = ExtractFieldTestRule(
            field_path="ingredients[0].name",
            expected_value="UNII-CANONICAL",
            evidence=[
                FieldEvidence(page=1, value="LABEL-PRINTED"),
                FieldEvidence(page=2, value="UNII-CANONICAL"),
            ],
        )
        result = compare_value_against_rule(rule, "LABEL-PRINTED", expected_type="string")
        assert result.passed

    def test_no_match_against_any_candidate(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            evidence=[FieldEvidence(page=1, value="baz")],
        )
        result = compare_value_against_rule(rule, "qux", expected_type="string")
        assert not result.passed


class TestPageGroundedMetric:
    def test_passes_when_page_matches_no_bbox(self):
        rule = ExtractFieldTestRule(
            field_path="ndc[0]",
            expected_value="37000-439",
            evidence=[FieldEvidence(page=4, value="37000-439")],
        )
        # Page-only citation (Extend pattern)
        citations = [FieldCitation(field_path="ndc[0]", page=4, bbox=None, source="extend")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"ndc": ["37000-439"]},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_page_grounded_pass_rate"].value == 1.0
        assert metrics["extract_page_grounded_coverage"].value == 1.0
        # Page-only citation does NOT contribute to bbox coverage
        assert metrics["extract_attribution_coverage"].value == 0.0

    def test_fails_when_page_wrong(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            evidence=[FieldEvidence(page=1, value="bar")],
        )
        citations = [FieldCitation(field_path="foo", page=99, bbox=None, source="x")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        # Citation has wrong page → page_pass=False but still qualifies for coverage
        assert metrics["extract_page_grounded_pass_rate"].value == 0.0
        assert metrics["extract_page_grounded_coverage"].value == 1.0


class TestCoarseParentPrefixWalk:
    def test_parent_cite_counts_for_m2a_not_m2b(self):
        rule = ExtractFieldTestRule(
            field_path="warnings[0].text",
            expected_value="Do not use",
            evidence=[FieldEvidence(page=2, value="Do not use")],
        )
        citations = [
            FieldCitation(field_path="warnings", page=2, bbox=[0.1, 0.1, 0.5, 0.5], source="reducto"),
        ]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"warnings": [{"text": "Do not use"}]},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_page_grounded_pass_rate"].value == 1.0
        assert metrics["extract_attribution_coverage"].value == 0.0


class TestCoverage:
    def test_zero_when_no_citations(self):
        rule = ExtractFieldTestRule(field_path="foo", expected_value="bar")
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_page_grounded_coverage"].value == 0.0
        assert metrics["extract_attribution_coverage"].value == 0.0

    def test_one_when_all_rules_have_citations(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            evidence=[FieldEvidence(page=1, bbox=[0.1, 0.1, 0.2, 0.2], value="bar")],
        )
        citations = [FieldCitation(field_path="foo", page=1, bbox=[0.1, 0.1, 0.2, 0.2], source="x")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_page_grounded_coverage"].value == 1.0
        assert metrics["extract_attribution_coverage"].value == 1.0


class TestPerRuleIouThreshold:
    def test_loose_threshold_lets_imperfect_iou_pass(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.4, 0.4])],
            iou_threshold=0.2,
        )
        # Pred bbox shifted slightly → IoU below 0.5 default but above 0.2
        citations = [FieldCitation(field_path="foo", page=1, bbox=[0.15, 0.15, 0.4, 0.4], source="x")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_attribution_pass_rate"].value == 1.0


class TestBackwardCompat:
    def test_legacy_bboxes_path_unchanged(self):
        """Legacy rules without v0.2 evidence keep producing the same M2b results."""
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            bboxes=[ExtractFieldBbox(page=1, bbox=[0.1, 0.1, 0.2, 0.2])],
        )
        citations = [FieldCitation(field_path="foo", page=1, bbox=[0.1, 0.1, 0.2, 0.2], source="x")]
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=citations,
                data_schema={"type": "object"},
            )
        )
        assert metrics["extract_attribution_pass_rate"].value == 1.0
        assert metrics["extract_value_f1"].value == 1.0


class TestV02StrayAndNullSemantics:
    """v0.2 rules carry the canonical value in evidence, not expected_value.

    A rule with ``expected_value=None`` but populated ``evidence[].value`` is
    NOT stray and NOT null-expected — it prescribes the value via evidence.
    """

    def test_v02_evidence_only_rule_scores_value_f1(self):
        rule = ExtractFieldTestRule(
            field_path="drug_name",
            expected_value=None,  # v0.2 leaves expected_value empty
            evidence=[FieldEvidence(page=1, bbox=[0.1, 0.1, 0.2, 0.2], value="Aspirin")],
            comparator="case_insensitive",
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"drug_name": "Aspirin"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        # Should match via OR-over-evidence even though expected_value is None
        assert metrics["extract_value_f1"].value == 1.0

    def test_v02_evidence_only_rule_not_treated_as_hallucination(self):
        rule = ExtractFieldTestRule(
            field_path="drug_name",
            expected_value=None,
            evidence=[FieldEvidence(page=1, value="Aspirin")],
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"drug_name": "Aspirin"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        # Should NOT emit null_hallucination_rate (no null-expected rules)
        assert "null_hallucination_rate" not in metrics

    def test_no_evidence_no_expected_value_is_null_expected(self):
        """Legacy null-expected behavior preserved when evidence is also empty."""
        rule = ExtractFieldTestRule(
            field_path="adverse_reactions",
            expected_value=None,
            evidence=None,
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"adverse_reactions": "made up"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        assert metrics["null_hallucination_rate"].value == 1.0

    def test_explicit_stray_tag_still_treated_stray(self):
        rule = ExtractFieldTestRule(
            field_path="foo",
            expected_value="bar",
            evidence=[FieldEvidence(page=1, value="bar")],
            tags=["stray"],
        )
        metrics = _named(
            compute_extract_field_grounding_metrics(
                extracted_data={"foo": "bar"},
                field_rules=[rule],
                field_citations=[],
                data_schema={"type": "object"},
            )
        )
        # Stray rules excluded from value F1
        assert "extract_value_f1" not in metrics


class TestLoaderDictShape:
    def test_field_rules_dict_converts_to_list(self, tmp_path):
        import json

        from parse_bench.test_cases.loader import load_test_case

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        config = {
            "_schema_version": "extract_core/v0.2",
            "data_schema": {"type": "object", "properties": {"foo": {"type": "string"}}},
            "expected_output": {"foo": "bar"},
            "_field_rules": {
                "foo": {
                    "expected_value": "bar",
                    "evidence": [{"page": 1, "value": "bar", "bbox": [0.1, 0.1, 0.2, 0.2]}],
                    "iou_threshold": 0.4,
                    "comparator": "exact",
                },
            },
        }
        (tmp_path / "doc.test.json").write_text(json.dumps(config))
        case = load_test_case(pdf)
        assert case is not None
        assert case.schema_version == "extract_core/v0.2"
        rules = case.test_rules
        assert len(rules) == 1
        rule = rules[0]
        assert rule.field_path == "foo"
        assert rule.iou_threshold == 0.4
        assert rule.comparator == "exact"
        assert rule.evidence is not None and rule.evidence[0].value == "bar"

    def test_conflict_both_shapes_errors(self, tmp_path):
        import json

        from parse_bench.test_cases.loader import load_test_case

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        config = {
            "data_schema": {"type": "object"},
            "expected_output": {},
            "_field_rules": {"foo": {"expected_value": "bar"}},
            "test_rules": [{"type": "extract_field", "field_path": "foo", "expected_value": "bar"}],
        }
        (tmp_path / "doc.test.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="cannot specify both"):
            load_test_case(pdf)
